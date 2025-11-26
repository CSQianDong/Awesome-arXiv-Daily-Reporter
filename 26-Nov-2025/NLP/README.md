# Latent Collaboration in Multi-Agent Systems 

**Authors**: Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20639)  

**Abstract**: Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at this https URL. 

---
# On Evaluating LLM Alignment by Evaluating LLMs as Judges 

**Authors**: Yixin Liu, Pengfei Liu, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20604)  

**Abstract**: Alignment with human preferences is an important evaluation aspect of LLMs, requiring them to be helpful, honest, safe, and to precisely follow human instructions. Evaluating large language models' (LLMs) alignment typically involves directly assessing their open-ended responses, requiring human annotators or strong LLM judges. Conversely, LLMs themselves have also been extensively evaluated as judges for assessing alignment. In this work, we examine the relationship between LLMs' generation and evaluation capabilities in aligning with human preferences. To this end, we first conduct a comprehensive analysis of the generation-evaluation consistency (GE-consistency) among various LLMs, revealing a strong correlation between their generation and evaluation capabilities when evaluated by a strong LLM preference oracle. Utilizing this finding, we propose a benchmarking paradigm that measures LLM alignment with human preferences without directly evaluating their generated outputs, instead assessing LLMs in their role as evaluators. Our evaluation shows that our proposed benchmark, AlignEval, matches or surpasses widely used automatic LLM evaluation benchmarks, such as AlpacaEval and Arena-Hard, in capturing human preferences when ranking LLMs. Our study offers valuable insights into the connection between LLMs' generation and evaluation capabilities, and introduces a benchmark that assesses alignment without directly evaluating model outputs. 

---
# From Words to Wisdom: Discourse Annotation and Baseline Models for Student Dialogue Understanding 

**Authors**: Farjana Sultana Mim, Shuchin Aeron, Eric Miller, Kristen Wendell  

**Link**: [PDF](https://arxiv.org/pdf/2511.20547)  

**Abstract**: Identifying discourse features in student conversations is quite important for educational researchers to recognize the curricular and pedagogical variables that cause students to engage in constructing knowledge rather than merely completing tasks. The manual analysis of student conversations to identify these discourse features is time-consuming and labor-intensive, which limits the scale and scope of studies. Leveraging natural language processing (NLP) techniques can facilitate the automatic detection of these discourse features, offering educational researchers scalable and data-driven insights. However, existing studies in NLP that focus on discourse in dialogue rarely address educational data. In this work, we address this gap by introducing an annotated educational dialogue dataset of student conversations featuring knowledge construction and task production discourse. We also establish baseline models for automatically predicting these discourse properties for each turn of talk within conversations, using pre-trained large language models GPT-3.5 and Llama-3.1. Experimental results indicate that these state-of-the-art models perform suboptimally on this task, indicating the potential for future research. 

---
# Bridging the Language Gap: Synthetic Voice Diversity via Latent Mixup for Equitable Speech Recognition 

**Authors**: Wesley Bian, Xiaofeng Lin, Guang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.20534)  

**Abstract**: Modern machine learning models for audio tasks often exhibit superior performance on English and other well-resourced languages, primarily due to the abundance of available training data. This disparity leads to an unfair performance gap for low-resource languages, where data collection is both challenging and costly. In this work, we introduce a novel data augmentation technique for speech corpora designed to mitigate this gap. Through comprehensive experiments, we demonstrate that our method significantly improves the performance of automatic speech recognition systems on low-resource languages. Furthermore, we show that our approach outperforms existing augmentation strategies, offering a practical solution for enhancing speech technology in underrepresented linguistic communities. 

---
# The Text Aphasia Battery (TAB): A Clinically-Grounded Benchmark for Aphasia-Like Deficits in Language Models 

**Authors**: Nathan Roll, Jill Kries, Flora Jin, Catherine Wang, Ann Marie Finley, Meghan Sumner, Cory Shain, Laura Gwilliams  

**Link**: [PDF](https://arxiv.org/pdf/2511.20507)  

**Abstract**: Large language models (LLMs) have emerged as a candidate "model organism" for human language, offering an unprecedented opportunity to study the computational basis of linguistic disorders like aphasia. However, traditional clinical assessments are ill-suited for LLMs, as they presuppose human-like pragmatic pressures and probe cognitive processes not inherent to artificial architectures. We introduce the Text Aphasia Battery (TAB), a text-only benchmark adapted from the Quick Aphasia Battery (QAB) to assess aphasic-like deficits in LLMs. The TAB comprises four subtests: Connected Text, Word Comprehension, Sentence Comprehension, and Repetition. This paper details the TAB's design, subtests, and scoring criteria. To facilitate large-scale use, we validate an automated evaluation protocol using Gemini 2.5 Flash, which achieves reliability comparable to expert human raters (prevalence-weighted Cohen's kappa = 0.255 for model--consensus agreement vs. 0.286 for human--human agreement). We release TAB as a clinically-grounded, scalable framework for analyzing language deficits in artificial systems. 

---
# Adversarial Confusion Attack: Disrupting Multimodal Large Language Models 

**Authors**: Jakub Hoscilowicz, Artur Janicki  

**Link**: [PDF](https://arxiv.org/pdf/2511.20494)  

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Applications include embedding adversarial images into websites to prevent MLLM-powered agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models. 

---
# Generation, Evaluation, and Explanation of Novelists' Styles with Single-Token Prompts 

**Authors**: Mosab Rezaei, Mina Rajaei Moghadam, Abdul Rahman Shaikh, Hamed Alhoori, Reva Freedman  

**Link**: [PDF](https://arxiv.org/pdf/2511.20459)  

**Abstract**: Recent advances in large language models have created new opportunities for stylometry, the study of writing styles and authorship. Two challenges, however, remain central: training generative models when no paired data exist, and evaluating stylistic text without relying only on human judgment. In this work, we present a framework for both generating and evaluating sentences in the style of 19th-century novelists. Large language models are fine-tuned with minimal, single-token prompts to produce text in the voices of authors such as Dickens, Austen, Twain, Alcott, and Melville. To assess these generative models, we employ a transformer-based detector trained on authentic sentences, using it both as a classifier and as a tool for stylistic explanation. We complement this with syntactic comparisons and explainable AI methods, including attention-based and gradient-based analyses, to identify the linguistic cues that drive stylistic imitation. Our findings show that the generated text reflects the authors' distinctive patterns and that AI-based evaluation offers a reliable alternative to human assessment. All artifacts of this work are published online. 

---
# A Task-Oriented Evaluation Framework for Text Normalization in Modern NLP Pipelines 

**Authors**: Md Abdullah Al Kafi, Raka Moni, Sumit Kumar Banshal  

**Link**: [PDF](https://arxiv.org/pdf/2511.20409)  

**Abstract**: Text normalization is an essential preprocessing step in many natural language processing (NLP) tasks, and stemming is one such normalization technique that reduces words to their base or root form. However, evaluating stemming methods is challenging because current evaluation approaches are limited and do not capture the potential harm caused by excessive stemming; therefore, it is essential to develop new approaches to evaluate stemming methods. To address this issue, this study propose a novel, task-oriented approach to evaluate stemming methods, which considers three aspects: (1) the utility of stemming using Stemming Effectiveness Score (SES), (2) the impact of stemming on downstream tasks using Model Performance Delta (MPD), and (3) the semantic similarity between stemmed and original words using Average Normalized Levenshtein Distance (ANLD), thus providing a comprehensive evaluation framework. We apply our evaluation framework to compare two stemmers for Bangla (BNLTK) and English (Snowball), and our results reveal a significant issue, prompting us to analyze their performance in detail. While the Bangla stemmer achieves the highest SES (1.67) due to effective word reduction (CR = 1.90), SES alone is insufficient because our proposed safety measure, ANLD, reveals that this high SES is due to harmful over-stemming (ANLD = 0.26), which correlates with the observed decrease in downstream this http URL contrast, the English stemmer achieves a moderate SES (1.31) with a safe meaning distance (ANLD = 0.14), allowing its word reduction to contribute positively to downstream performance; therefore, it is a more reliable stemmer. Our study provides a valuable tool for distinguishing between potential efficiency gains (high SES) and meaning preservation (low ANLD). 

---
# BengaliFig: A Low-Resource Challenge for Figurative and Culturally Grounded Reasoning in Bengali 

**Authors**: Abdullah Al Sefat  

**Link**: [PDF](https://arxiv.org/pdf/2511.20399)  

**Abstract**: Large language models excel on broad multilingual benchmarks but remain to be evaluated extensively in figurative and culturally grounded reasoning, especially in low-resource contexts. We present BengaliFig, a compact yet richly annotated challenge set that targets this gap in Bengali, a widely spoken low-resourced language. The dataset contains 435 unique riddles drawn from Bengali oral and literary traditions. Each item is annotated along five orthogonal dimensions capturing reasoning type, trap type, cultural depth, answer category, and difficulty, and is automatically converted to multiple-choice format through a constraint-aware, AI-assisted pipeline. We evaluate eight frontier LLMs from major providers under zero-shot and few-shot chain-of-thought prompting, revealing consistent weaknesses in metaphorical and culturally specific reasoning. BengaliFig thus contributes both a diagnostic probe for evaluating LLM robustness in low-resource cultural contexts and a step toward inclusive and heritage-aware NLP evaluation. 

---
# The Curious Case of Analogies: Investigating Analogical Reasoning in Large Language Models 

**Authors**: Taewhoo Lee, Minju Song, Chanwoong Yoon, Jungwoo Park, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20344)  

**Abstract**: Analogical reasoning is at the core of human cognition, serving as an important foundation for a variety of intellectual activities. While prior work has shown that LLMs can represent task patterns and surface-level concepts, it remains unclear whether these models can encode high-level relational concepts and apply them to novel situations through structured comparisons. In this work, we explore this fundamental aspect using proportional and story analogies, and identify three key findings. First, LLMs effectively encode the underlying relationships between analogous entities; both attributive and relational information propagate through mid-upper layers in correct cases, whereas reasoning failures reflect missing relational information within these layers. Second, unlike humans, LLMs often struggle not only when relational information is missing, but also when attempting to apply it to new entities. In such cases, strategically patching hidden representations at critical token positions can facilitate information transfer to a certain extent. Lastly, successful analogical reasoning in LLMs is marked by strong structural alignment between analogous situations, whereas failures often reflect degraded or misplaced alignment. Overall, our findings reveal that LLMs exhibit emerging but limited capabilities in encoding and applying high-level relational concepts, highlighting both parallels and gaps with human cognition. 

---
# Scaling LLM Speculative Decoding: Non-Autoregressive Forecasting in Large-Batch Scenarios 

**Authors**: Luohe Shi, Zuchao Li, Lefei Zhang, Baoyuan Qi, Guoming Liu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20340)  

**Abstract**: Speculative decoding accelerates LLM inference by utilizing otherwise idle computational resources during memory-to-chip data transfer. Current speculative decoding methods typically assume a considerable amount of available computing power, then generate a complex and massive draft tree using a small autoregressive language model to improve overall prediction accuracy. However, methods like batching have been widely applied in mainstream model inference systems as a superior alternative to speculative decoding, as they compress the available idle computing power. Therefore, performing speculative decoding with low verification resources and low scheduling costs has become an important research problem. We believe that more capable models that allow for parallel generation on draft sequences are what we truly need. Recognizing the fundamental nature of draft models to only generate sequences of limited length, we propose SpecFormer, a novel architecture that integrates unidirectional and bidirectional attention mechanisms. SpecFormer combines the autoregressive model's ability to extract information from the entire input sequence with the parallel generation benefits of non-autoregressive models. This design eliminates the reliance on large prefix trees and achieves consistent acceleration, even in large-batch scenarios. Through lossless speculative decoding experiments across models of various scales, we demonstrate that SpecFormer sets a new standard for scaling LLM inference with lower training demands and reduced computational costs. 

---
# REFLEX: Self-Refining Explainable Fact-Checking via Disentangling Truth into Style and Substance 

**Authors**: Chuyi Kong, Gao Wei, Jing Ma, Hongzhan Lin, Zhiyuan Fan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20233)  

**Abstract**: The prevalence of misinformation on social media threatens public trust, demanding automated fact-checking systems that provide accurate verdicts with interpretable explanations. However, existing large language model-based (LLM-based) approaches often rely heavily on external knowledge sources, introducing substantial latency and even hallucinations that undermine reliability, interpretability, and responsiveness, which is crucial for real-time use. To address these challenges, we propose REason-guided Fact-checking with Latent EXplanations REFLEX paradigm, a plug-and-play, self-refining paradigm that leverages the internal knowledge in backbone model to improve both verdict accuracy and explanation quality. REFLEX reformulates fact-checking as a role-play dialogue and jointly trains verdict prediction and explanation generation. It adaptively extracts contrastive activation pairs between the backbone model and its fine-tuned variant to construct steering vectors that disentangle truth into style and substance naturally. These activation-level signals guide inference and suppress noisy explanations, enabling more faithful and efficient reasoning. Experiments on real-world datasets show that REFLEX outperforms previous methods that steer toward a single truth direction and underscores the challenge traditional approaches face when handling the subtle, human-unknown truth in fact-checking tasks. Remarkably, with only 465 self-refined training samples, RELFEX achieves state-of-the-art performance. Furthermore, models trained with explanatory objectives can effectively guide those without them, yielding up to a 7.57% improvement, highlighting that internal explanation signals play a dual role in both interpreting and enhancing factual reasoning. 

---
# KyrgyzBERT: A Compact, Efficient Language Model for Kyrgyz NLP 

**Authors**: Adilet Metinov, Gulida M. Kudakeeva, Gulnara D. Kabaeva  

**Link**: [PDF](https://arxiv.org/pdf/2511.20182)  

**Abstract**: Kyrgyz remains a low-resource language with limited foundational NLP tools. To address this gap, we introduce KyrgyzBERT, the first publicly available monolingual BERT-based language model for Kyrgyz. The model has 35.9M parameters and uses a custom tokenizer designed for the language's morphological structure. To evaluate performance, we create kyrgyz-sst2, a sentiment analysis benchmark built by translating the Stanford Sentiment Treebank and manually annotating the full test set. KyrgyzBERT fine-tuned on this dataset achieves an F1-score of 0.8280, competitive with a fine-tuned mBERT model five times larger. All models, data, and code are released to support future research in Kyrgyz NLP. 

---
# SEDA: A Self-Adapted Entity-Centric Data Augmentation for Boosting Gird-based Discontinuous NER Models 

**Authors**: Wen-Fang Su, Hsiao-Wei Chou, Wen-Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20143)  

**Abstract**: Named Entity Recognition (NER) is a critical task in natural language processing, yet it remains particularly challenging for discontinuous entities. The primary difficulty lies in text segmentation, as traditional methods often missegment or entirely miss cross-sentence discontinuous entities, significantly affecting recognition accuracy. Therefore, we aim to address the segmentation and omission issues associated with such entities. Recent studies have shown that grid-tagging methods are effective for information extraction due to their flexible tagging schemes and robust architectures. Building on this, we integrate image data augmentation techniques, such as cropping, scaling, and padding, into grid-based models to enhance their ability to recognize discontinuous entities and handle segmentation challenges. Experimental results demonstrate that traditional segmentation methods often fail to capture cross-sentence discontinuous entities, leading to decreased performance. In contrast, our augmented grid models achieve notable improvements. Evaluations on the CADEC, ShARe13, and ShARe14 datasets show F1 score gains of 1-2.5% overall and 3.7-8.4% for discontinuous entities, confirming the effectiveness of our approach. 

---
# "When Data is Scarce, Prompt Smarter"... Approaches to Grammatical Error Correction in Low-Resource Settings 

**Authors**: Somsubhra De, Harsh Kumar, Arun Prakash A  

**Link**: [PDF](https://arxiv.org/pdf/2511.20120)  

**Abstract**: Grammatical error correction (GEC) is an important task in Natural Language Processing that aims to automatically detect and correct grammatical mistakes in text. While recent advances in transformer-based models and large annotated datasets have greatly improved GEC performance for high-resource languages such as English, the progress has not extended equally. For most Indic languages, GEC remains a challenging task due to limited resources, linguistic diversity and complex morphology. In this work, we explore prompting-based approaches using state-of-the-art large language models (LLMs), such as GPT-4.1, Gemini-2.5 and LLaMA-4, combined with few-shot strategy to adapt them to low-resource settings. We observe that even basic prompting strategies, such as zero-shot and few-shot approaches, enable these LLMs to substantially outperform fine-tuned Indic-language models like Sarvam-22B, thereby illustrating the exceptional multilingual generalization capabilities of contemporary LLMs for GEC. Our experiments show that carefully designed prompts and lightweight adaptation significantly enhance correction quality across multiple Indic languages. We achieved leading results in the shared task--ranking 1st in Tamil (GLEU: 91.57) and Hindi (GLEU: 85.69), 2nd in Telugu (GLEU: 85.22), 4th in Bangla (GLEU: 92.86), and 5th in Malayalam (GLEU: 92.97). These findings highlight the effectiveness of prompt-driven NLP techniques and underscore the potential of large-scale LLMs to bridge resource gaps in multilingual GEC. 

---
# Mispronunciation Detection and Diagnosis Without Model Training: A Retrieval-Based Approach 

**Authors**: Huu Tuong Tu, Ha Viet Khanh, Tran Tien Dat, Vu Huan, Thien Van Luong, Nguyen Tien Cuong, Nguyen Thi Thu Trang  

**Link**: [PDF](https://arxiv.org/pdf/2511.20107)  

**Abstract**: Mispronunciation Detection and Diagnosis (MDD) is crucial for language learning and speech therapy. Unlike conventional methods that require scoring models or training phoneme-level models, we propose a novel training-free framework that leverages retrieval techniques with a pretrained Automatic Speech Recognition model. Our method avoids phoneme-specific modeling or additional task-specific training, while still achieving accurate detection and diagnosis of pronunciation errors. Experiments on the L2-ARCTIC dataset show that our method achieves a superior F1 score of 69.60% while avoiding the complexity of model training. 

---
# EM2LDL: A Multilingual Speech Corpus for Mixed Emotion Recognition through Label Distribution Learning 

**Authors**: Xingfeng Li, Xiaohan Shi, Junjie Li, Yongwei Li, Masashi Unoki, Tomoki Toda, Masato Akagi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20106)  

**Abstract**: This study introduces EM2LDL, a novel multilingual speech corpus designed to advance mixed emotion recognition through label distribution learning. Addressing the limitations of predominantly monolingual and single-label emotion corpora \textcolor{black}{that restrict linguistic diversity, are unable to model mixed emotions, and lack ecological validity}, EM2LDL comprises expressive utterances in English, Mandarin, and Cantonese, capturing the intra-utterance code-switching prevalent in multilingual regions like Hong Kong and Macao. The corpus integrates spontaneous emotional expressions from online platforms, annotated with fine-grained emotion distributions across 32 categories. Experimental baselines using self-supervised learning models demonstrate robust performance in speaker-independent gender-, age-, and personality-based evaluations, with HuBERT-large-EN achieving optimal results. By incorporating linguistic diversity and ecological validity, EM2LDL enables the exploration of complex emotional dynamics in multilingual settings. This work provides a versatile testbed for developing adaptive, empathetic systems for applications in affective computing, including mental health monitoring and cross-cultural communication. The dataset, annotations, and baseline codes are publicly available at this https URL. 

---
# SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space 

**Authors**: Zhenyi Shen, Junru Lu, Lin Gui, Jiazheng Li, Yulan He, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.20102)  

**Abstract**: The quadratic complexity of full attention limits efficient long-context processing in large language models (LLMs). Sparse attention mitigates this cost by restricting each query to attend to a subset of previous tokens; however, training-free approaches often lead to severe performance degradation. Native sparse-attention methods (e.g., NSA, MoBA) alleviate this issue, yet exhibit a critical paradox: they produce lower attention sparsity than full-attention models, despite aiming to approximate full attention, which may constrain their effectiveness. We attribute this paradox to gradient update deficiency: low-ranked key-value pairs excluded during sparse training receive neither forward contribution nor backward gradients, and thus never learn proper suppression. To overcome this limitation, we propose SSA (Sparse Sparse Attention), a unified training framework that considers both sparse and full attention and enforces bidirectional alignment at every layer. This design preserves gradient flow to all tokens while explicitly encouraging sparse-attention outputs to align with their full-attention counterparts, thereby promoting stronger sparsity. As a result, SSA achieves state-of-the-art performance under both sparse and full attention inference across multiple commonsense benchmarks. Furthermore, SSA enables models to adapt smoothly to varying sparsity budgets; performance improves consistently as more tokens are allowed to attend, supporting flexible compute-performance trade-offs at inference time. Finally, we show that native sparse-attention training surprisingly improves long-context extrapolation by mitigating the over-allocation of attention values in sink areas, with SSA demonstrating the strongest extrapolation capability. 

---
# More Bias, Less Bias: BiasPrompting for Enhanced Multiple-Choice Question Answering 

**Authors**: Duc Anh Vu, Thong Nguyen, Cong-Duy Nguyen, Viet Anh Nguyen, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20086)  

**Abstract**: With the advancement of large language models (LLMs), their performance on multiple-choice question (MCQ) tasks has improved significantly. However, existing approaches face key limitations: answer choices are typically presented to LLMs without contextual grounding or explanation. This absence of context can lead to incomplete exploration of all possible answers, ultimately degrading the models' reasoning capabilities. To address these challenges, we introduce BiasPrompting, a novel inference framework that guides LLMs to generate and critically evaluate reasoning across all plausible answer options before reaching a final prediction. It consists of two components: first, a reasoning generation stage, where the model is prompted to produce supportive reasonings for each answer option, and then, a reasoning-guided agreement stage, where the generated reasonings are synthesized to select the most plausible answer. Through comprehensive evaluations, BiasPrompting demonstrates significant improvements in five widely used multiple-choice question answering benchmarks. Our experiments showcase that BiasPrompting enhances the reasoning capabilities of LLMs and provides a strong foundation for tackling complex and challenging questions, particularly in settings where existing methods underperform. 

---
# MTA: A Merge-then-Adapt Framework for Personalized Large Language Model 

**Authors**: Xiaopeng Li, Yuanjin Zheng, Wanyu Wang, wenlin zhang, Pengyue Jia, Yiqi Wang, Maolin Wang, Xuetao Wei, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20072)  

**Abstract**: Personalized Large Language Models (PLLMs) aim to align model outputs with individual user preferences, a crucial capability for user-centric applications. However, the prevalent approach of fine-tuning a separate module for each user faces two major limitations: (1) storage costs scale linearly with the number of users, rendering the method unscalable; and (2) fine-tuning a static model from scratch often yields suboptimal performance for users with sparse data. To address these challenges, we propose MTA, a Merge-then-Adapt framework for PLLMs. MTA comprises three key stages. First, we construct a shared Meta-LoRA Bank by selecting anchor users and pre-training meta-personalization traits within meta-LoRA modules. Second, to ensure scalability and enable dynamic personalization combination beyond static models, we introduce an Adaptive LoRA Fusion stage. This stage retrieves and dynamically merges the most relevant anchor meta-LoRAs to synthesize a user-specific one, thereby eliminating the need for user-specific storage and supporting more flexible personalization. Third, we propose a LoRA Stacking for Few-Shot Personalization stage, which applies an additional ultra-low-rank, lightweight LoRA module on top of the merged LoRA. Fine-tuning this module enables effective personalization under few-shot settings. Extensive experiments on the LaMP benchmark demonstrate that our approach outperforms existing SOTA methods across multiple tasks. 

---
# Online-PVLM: Advancing Personalized VLMs with Online Concept Learning 

**Authors**: Huiyu Bai, Runze Wang, Zhuoyun Du, Yiyang Zhao, Fengji Zhang, Haoyu Chen, Xiaoyong Zhu, Bo Zheng, Xuejiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.20056)  

**Abstract**: Personalized Visual Language Models (VLMs) are gaining increasing attention for their formidable ability in user-specific concepts aligned interactions (e.g., identifying a user's bike). Existing methods typically require the learning of separate embeddings for each new concept, which fails to support real-time adaptation during testing. This limitation becomes particularly pronounced in large-scale scenarios, where efficient retrieval of concept embeddings is not achievable. To alleviate this gap, we propose Online-PVLM, a framework for online concept learning by leveraging hyperbolic representations. Our approach makes a train-free paradigm for concept embeddings generation at test time, making the use of personalized VLMs both scalable and efficient. In addition, we develop OP-Eval, a comprehensive and large-scale benchmark comprising 1,292 concepts and over 30K high-quality instances with diverse question types, designed to rigorously assess online concept learning in realistic scenarios. Extensive experiments demonstrate the state-of-the-art performance of our proposed framework. Our source code and dataset will be made available. 

---
# A Machine Learning Approach for Detection of Mental Health Conditions and Cyberbullying from Social Media 

**Authors**: Edward Ajayi, Martha Kachweka, Mawuli Deku, Emily Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2511.20001)  

**Abstract**: Mental health challenges and cyberbullying are increasingly prevalent in digital spaces, necessitating scalable and interpretable detection systems. This paper introduces a unified multiclass classification framework for detecting ten distinct mental health and cyberbullying categories from social media data. We curate datasets from Twitter and Reddit, implementing a rigorous "split-then-balance" pipeline to train on balanced data while evaluating on a realistic, held-out imbalanced test set. We conducted a comprehensive evaluation comparing traditional lexical models, hybrid approaches, and several end-to-end fine-tuned transformers. Our results demonstrate that end-to-end fine-tuning is critical for performance, with the domain-adapted MentalBERT emerging as the top model, achieving an accuracy of 0.92 and a Macro F1 score of 0.76, surpassing both its generic counterpart and a zero-shot LLM baseline. Grounded in a comprehensive ethical analysis, we frame the system as a human-in-the-loop screening aid, not a diagnostic tool. To support this, we introduce a hybrid SHAPLLM explainability framework and present a prototype dashboard ("Social Media Screener") designed to integrate model predictions and their explanations into a practical workflow for moderators. Our work provides a robust baseline, highlighting future needs for multi-label, clinically-validated datasets at the critical intersection of online safety and computational mental health. 

---
# Directional Optimization Asymmetry in Transformers: A Synthetic Stress Test 

**Authors**: Mihir Sahasrabudhe  

**Link**: [PDF](https://arxiv.org/pdf/2511.19997)  

**Abstract**: Transformers are theoretically reversal-invariant: their function class does not prefer left-to-right over right-to-left mappings. Yet empirical studies on natural language repeatedly report a "reversal curse," and recent work on temporal asymmetry in LLMs suggests that real-world corpora carry their own arrow of time. This leaves an unresolved question: do directional failures stem from linguistic statistics, or from the architecture itself? We cut through this ambiguity with a fully synthetic, entropy-controlled benchmark designed as a clean-room stress test for directional learning. Using random string mappings with tunable branching factor K, we construct forward tasks with zero conditional entropy and inverse tasks with analytically determined entropy floors. Excess loss above these floors reveals that even scratch-trained GPT-2 models exhibit a strong, reproducible directional optimization gap (e.g., 1.16 nats at K=5), far larger than that of an MLP trained on the same data. Pre-trained initializations shift optimization behavior but do not eliminate this gap, while LoRA encounters a sharp capacity wall on high-entropy inverse mappings. Together, these results isolate a minimal, semantics-free signature of directional friction intrinsic to causal Transformer training-one that persists even when linguistic priors, token frequencies, and corpus-level temporal asymmetries are removed. Our benchmark provides a controlled instrument for dissecting directional biases in modern sequence models and motivates deeper mechanistic study of why inversion remains fundamentally harder for Transformers. 

---
# $\text{R}^2\text{R}$: A Route-to-Rerank Post-Training Framework for Multi-Domain Decoder-Only Rerankers 

**Authors**: Xinyu Wang, Hanwei Wu, Qingchen Hu, Zhenghan Tai, Jingrui Tian, Lei Ding, Jijun Chi, Hailin He, Tung Sum Thomas Kwok, Yufei Cui, Sicheng Lyu, Muzhi Li, Mingze Li, Xinyue Yu, Ling Zhou, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.19987)  

**Abstract**: Decoder-only rerankers are central to Retrieval-Augmented Generation (RAG). However, generalist models miss domain-specific nuances in high-stakes fields like finance and law, and naive fine-tuning causes surface-form overfitting and catastrophic forgetting. To address this challenge, we introduce R2R, a domain-aware framework that combines dynamic expert routing with a two-stage training strategy, Entity Abstraction for Generalization (EAG). EAG introduces a counter-shortcut mechanism by masking the most predictive surface cues, forcing the reranker to learn domain-invariant relevance patterns rather than memorizing dataset-specific entities. To efficiently activate domain experts, R2R employs a lightweight Latent Semantic Router that probes internal representations from the frozen backbone decoder to select the optimal LoRA expert per query. Extensive experiments across different reranker backbones and diverse domains (legal, medical, and financial) demonstrate that R2R consistently surpasses generalist and single-domain fine-tuned baselines. Our results confirm that R2R is a model-agnostic and modular approach to domain specialization with strong cross-domain robustness. 

---
# AppSelectBench: Application-Level Tool Selection Benchmark 

**Authors**: Tianyi Chen, Michael Solodko, Sen Wang, Jongwoo Ko, Junheng Hao, Colby Banbury, Sara Abdali, Saeed Amizadeh, Qing Xiao, Yinheng Li, Tianyu Ding, Kamran Ghasedi Dizaji, Suzhen Zheng, Hao Fan, Justin Wagle, Pashmina Cameron, Kazuhito Koishida  

**Link**: [PDF](https://arxiv.org/pdf/2511.19957)  

**Abstract**: Computer Using Agents (CUAs) are increasingly equipped with external tools, enabling them to perform complex and realistic tasks. For CUAs to operate effectively, application selection, which refers to deciding which application to use before invoking fine-grained tools such as APIs, is a fundamental capability. It determines whether the agent initializes the correct environment, avoids orchestration confusion, and efficiently focuses on relevant context. However, existing benchmarks primarily assess fine-grained API selection, offering limited insight into whether models can reason across and choose between different applications. To fill this gap, we introduce AppSelectBench, a comprehensive benchmark for evaluating application selection in CUAs. AppSelectBench contains a novel user task generation pipeline that produces realistic, diverse, and semantically grounded user intents at scale, together with unified evaluation protocols covering random, heuristic, zero-shot, few-shot, and retrieval-augmented-settings. AppSelectBench covers one hundred widely used desktop applications and includes more than one hundred thousand realistic, diverse, and semantically grounded user tasks. Extensive experiments across both closed-source and open-source large language models reveal systematic strengths and weaknesses in inter-application reasoning, showing that even the most capable models still struggle to make consistent application choices. Together, these results establish AppSelectBench as a foundation for studying and advancing application level reasoning, an essential yet underexplored capability of intelligent CUAs. The source is available at this https URL. 

---
# A Systematic Analysis of Large Language Models with RAG-enabled Dynamic Prompting for Medical Error Detection and Correction 

**Authors**: Farzad Ahmed, Joniel Augustine Jerome, Meliha Yetisgen, Özlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2511.19858)  

**Abstract**: Objective: Clinical documentation contains factual, diagnostic, and management errors that can compromise patient safety. Large language models (LLMs) may help detect and correct such errors, but their behavior under different prompting strategies remains unclear. We evaluate zero-shot prompting, static prompting with random exemplars (SPR), and retrieval-augmented dynamic prompting (RDP) for three subtasks of medical error processing: error flag detection, error sentence detection, and error correction.
Methods: Using the MEDEC dataset, we evaluated nine instruction-tuned LLMs (GPT, Claude, Gemini, and OpenAI o-series models). We measured performance using accuracy, recall, false-positive rate (FPR), and an aggregate score of ROUGE-1, BLEURT, and BERTScore for error correction. We also analyzed example outputs to identify failure modes and differences between LLM and clinician reasoning.
Results: Zero-shot prompting showed low recall in both detection tasks, often missing abbreviation-heavy or atypical errors. SPR improved recall but increased FPR. Across all nine LLMs, RDP reduced FPR by about 15 percent, improved recall by 5 to 10 percent in error sentence detection, and generated more contextually accurate corrections.
Conclusion: Across diverse LLMs, RDP outperforms zero-shot and SPR prompting. Using retrieved exemplars improves detection accuracy, reduces false positives, and enhances the reliability of medical error correction. 

---
# Profile-LLM: Dynamic Profile Optimization for Realistic Personality Expression in LLMs 

**Authors**: Shi-Wei Dai, Yan-Wei Shie, Tsung-Huan Yang, Lun-Wei Ku, Yung-Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.19852)  

**Abstract**: Personalized Large Language Models (LLMs) have been shown to be an effective way to create more engaging and enjoyable user-AI interactions. While previous studies have explored using prompts to elicit specific personality traits in LLMs, they have not optimized these prompts to maximize personality expression. To address this limitation, we propose PersonaPulse: Dynamic Profile Optimization for Realistic Personality Expression in LLMs, a framework that leverages LLMs' inherent knowledge of personality traits to iteratively enhance role-play prompts while integrating a situational response benchmark as a scoring tool, ensuring a more realistic and contextually grounded evaluation to guide the optimization process. Quantitative evaluations demonstrate that the prompts generated by PersonaPulse outperform those of prior work, which were designed based on personality descriptions from psychological studies. Additionally, we explore the relationship between model size and personality modeling through extensive experiments. Finally, we find that, for certain personality traits, the extent of personality evocation can be partially controlled by pausing the optimization process. These findings underscore the importance of prompt optimization in shaping personality expression within LLMs, offering valuable insights for future research on adaptive AI interactions. 

---
# Language-Independent Sentiment Labelling with Distant Supervision: A Case Study for English, Sepedi and Setswana 

**Authors**: Koena Ronny Mabokela, Tim Schlippe, Mpho Raborife, Turgay Celik  

**Link**: [PDF](https://arxiv.org/pdf/2511.19818)  

**Abstract**: Sentiment analysis is a helpful task to automatically analyse opinions and emotions on various topics in areas such as AI for Social Good, AI in Education or marketing. While many of the sentiment analysis systems are developed for English, many African languages are classified as low-resource languages due to the lack of digital language resources like text labelled with corresponding sentiment classes. One reason for that is that manually labelling text data is time-consuming and expensive. Consequently, automatic and rapid processes are needed to reduce the manual effort as much as possible making the labelling process as efficient as possible. In this paper, we present and analyze an automatic language-independent sentiment labelling method that leverages information from sentiment-bearing emojis and words. Our experiments are conducted with tweets in the languages English, Sepedi and Setswana from SAfriSenti, a multilingual sentiment corpus for South African languages. We show that our sentiment labelling approach is able to label the English tweets with an accuracy of 66%, the Sepedi tweets with 69%, and the Setswana tweets with 63%, so that on average only 34% of the automatically generated labels remain to be corrected. 

---
# Breaking Bad: Norms for Valence, Arousal, and Dominance for over 10k English Multiword Expressions 

**Authors**: Saif M. Mohammad  

**Link**: [PDF](https://arxiv.org/pdf/2511.19816)  

**Abstract**: Factor analysis studies have shown that the primary dimensions of word meaning are Valence (V), Arousal (A), and Dominance (D). Existing lexicons such as the NRC VAD Lexicon, published in 2018, include VAD association ratings for words. Here, we present a complement to it, which has human ratings of valence, arousal, and dominance for 10k English Multiword Expressions (MWEs) and their constituent words. We also increase the coverage of unigrams, especially words that have become more common since 2018. In all, the new NRC VAD Lexicon v2 now has entries for 10k MWEs and 25k words, in addition to the entries in v1. We show that the associations are highly reliable. We use the lexicon to examine emotional characteristics of MWEs, including: 1. The degree to which MWEs (idioms, noun compounds, and verb particle constructions) exhibit strong emotionality; 2. The degree of emotional compositionality in MWEs. The lexicon enables a wide variety of research in NLP, Psychology, Public Health, Digital Humanities, and Social Sciences. The NRC VAD Lexicon v2 is freely available through the project webpage: this http URL 

---
# Gender Bias in Emotion Recognition by Large Language Models 

**Authors**: Maureen Herbert, Katie Sun, Angelica Lim, Yasaman Etesam  

**Link**: [PDF](https://arxiv.org/pdf/2511.19785)  

**Abstract**: The rapid advancement of large language models (LLMs) and their growing integration into daily life underscore the importance of evaluating and ensuring their fairness. In this work, we examine fairness within the domain of emotional theory of mind, investigating whether LLMs exhibit gender biases when presented with a description of a person and their environment and asked, "How does this person feel?". Furthermore, we propose and evaluate several debiasing strategies, demonstrating that achieving meaningful reductions in bias requires training based interventions rather than relying solely on inference-time prompt-based approaches such as prompt engineering. 

---
# What does it mean to understand language? 

**Authors**: Colton Casto, Anna Ivanova, Evelina Fedorenko, Nancy Kanwisher  

**Link**: [PDF](https://arxiv.org/pdf/2511.19757)  

**Abstract**: Language understanding entails not just extracting the surface-level meaning of the linguistic input, but constructing rich mental models of the situation it describes. Here we propose that because processing within the brain's core language system is fundamentally limited, deeply understanding language requires exporting information from the language system to other brain regions that compute perceptual and motor representations, construct mental models, and store our world knowledge and autobiographical memories. We review the existing evidence for this hypothesis, and argue that recent progress in cognitive neuroscience provides both the conceptual foundation and the methods to directly test it, thus opening up a new strategy to reveal what it means, cognitively and neurally, to understand language. 

---
# Comparative Analysis of LoRA-Adapted Embedding Models for Clinical Cardiology Text Representation 

**Authors**: Richard J. Young, Alice M. Matthews  

**Link**: [PDF](https://arxiv.org/pdf/2511.19739)  

**Abstract**: Domain-specific text embeddings are critical for clinical natural language processing, yet systematic comparisons across model architectures remain limited. This study evaluates ten transformer-based embedding models adapted for cardiology through Low-Rank Adaptation (LoRA) fine-tuning on 106,535 cardiology text pairs derived from authoritative medical textbooks. Results demonstrate that encoder-only architectures, particularly BioLinkBERT, achieve superior domain-specific performance (separation score: 0.510) compared to larger decoder-based models, while requiring significantly fewer computational resources. The findings challenge the assumption that larger language models necessarily produce better domain-specific embeddings and provide practical guidance for clinical NLP system development. All models, training code, and evaluation datasets are publicly available to support reproducible research in medical informatics. 

---
# Can LLMs Faithfully Explain Themselves in Low-Resource Languages? A Case Study on Emotion Detection in Persian 

**Authors**: Mobina Mehrazar, Mohammad Amin Yousefi, Parisa Abolfath Beygi, Behnam Bahrak  

**Link**: [PDF](https://arxiv.org/pdf/2511.19719)  

**Abstract**: Large language models (LLMs) are increasingly used to generate self-explanations alongside their predictions, a practice that raises concerns about the faithfulness of these explanations, especially in low-resource languages. This study evaluates the faithfulness of LLM-generated explanations in the context of emotion classification in Persian, a low-resource language, by comparing the influential words identified by the model against those identified by human annotators. We assess faithfulness using confidence scores derived from token-level log-probabilities. Two prompting strategies, differing in the order of explanation and prediction (Predict-then-Explain and Explain-then-Predict), are tested for their impact on explanation faithfulness. Our results reveal that while LLMs achieve strong classification performance, their generated explanations often diverge from faithful reasoning, showing greater agreement with each other than with human judgments. These results highlight the limitations of current explanation methods and metrics, emphasizing the need for more robust approaches to ensure LLM reliability in multilingual and low-resource contexts. 

---
# Efficient Multi-Hop Question Answering over Knowledge Graphs via LLM Planning and Embedding-Guided Search 

**Authors**: Manil Shrestha, Edward Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.19648)  

**Abstract**: Multi-hop question answering over knowledge graphs remains computationally challenging due to the combinatorial explosion of possible reasoning paths. Recent approaches rely on expensive Large Language Model (LLM) inference for both entity linking and path ranking, limiting their practical deployment. Additionally, LLM-generated answers often lack verifiable grounding in structured knowledge. We present two complementary hybrid algorithms that address both efficiency and verifiability: (1) LLM-Guided Planning that uses a single LLM call to predict relation sequences executed via breadth-first search, achieving near-perfect accuracy (micro-F1 > 0.90) while ensuring all answers are grounded in the knowledge graph, and (2) Embedding-Guided Neural Search that eliminates LLM calls entirely by fusing text and graph embeddings through a lightweight 6.7M-parameter edge scorer, achieving over 100 times speedup with competitive accuracy. Through knowledge distillation, we compress planning capability into a 4B-parameter model that matches large-model performance at zero API cost. Evaluation on MetaQA demonstrates that grounded reasoning consistently outperforms ungrounded generation, with structured planning proving more transferable than direct answer generation. Our results show that verifiable multi-hop reasoning does not require massive models at inference time, but rather the right architectural inductive biases combining symbolic structure with learned representations. 

---
# Does Understanding Inform Generation in Unified Multimodal Models? From Analysis to Path Forward 

**Authors**: Yuwei Niu, Weiyang Jin, Jiaqi Liao, Chaoran Feng, Peng Jin, Bin Lin, Zongjian Li, Bin Zhu, Weihao Yu, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2511.20561)  

**Abstract**: Recent years have witnessed significant progress in Unified Multimodal Models, yet a fundamental question remains: Does understanding truly inform generation? To investigate this, we introduce UniSandbox, a decoupled evaluation framework paired with controlled, synthetic datasets to avoid data leakage and enable detailed analysis. Our findings reveal a significant understanding-generation gap, which is mainly reflected in two key dimensions: reasoning generation and knowledge transfer. Specifically, for reasoning generation tasks, we observe that explicit Chain-of-Thought (CoT) in the understanding module effectively bridges the gap, and further demonstrate that a self-training approach can successfully internalize this ability, enabling implicit reasoning during generation. Additionally, for knowledge transfer tasks, we find that CoT assists the generative process by helping retrieve newly learned knowledge, and also discover that query-based architectures inherently exhibit latent CoT-like properties that affect this transfer. UniSandbox provides preliminary insights for designing future unified architectures and training strategies that truly bridge the gap between understanding and generation. Code and data are available at this https URL 

---
# DesignPref: Capturing Personal Preferences in Visual Design Generation 

**Authors**: Yi-Hao Peng, Jeffrey P. Bigham, Jason Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.20513)  

**Abstract**: Generative models, such as large language models and text-to-image diffusion models, are increasingly used to create visual designs like user interfaces (UIs) and presentation slides. Finetuning and benchmarking these generative models have often relied on datasets of human-annotated design preferences. Yet, due to the subjective and highly personalized nature of visual design, preference varies widely among individuals. In this paper, we study this problem by introducing DesignPref, a dataset of 12k pairwise comparisons of UI design generation annotated by 20 professional designers with multi-level preference ratings. We found that among trained designers, substantial levels of disagreement exist (Krippendorff's alpha = 0.25 for binary preferences). Natural language rationales provided by these designers indicate that disagreements stem from differing perceptions of various design aspect importance and individual preferences. With DesignPref, we demonstrate that traditional majority-voting methods for training aggregated judge models often do not accurately reflect individual preferences. To address this challenge, we investigate multiple personalization strategies, particularly fine-tuning or incorporating designer-specific annotations into RAG pipelines. Our results show that personalized models consistently outperform aggregated baseline models in predicting individual designers' preferences, even when using 20 times fewer examples. Our work provides the first dataset to study personalized visual design evaluation and support future research into modeling individual design taste. 

---
# Soft Adaptive Policy Optimization 

**Authors**: Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.20347)  

**Abstract**: Reinforcement learning (RL) plays an increasingly important role in enhancing the reasoning capabilities of large language models (LLMs), yet stable and performant policy optimization remains challenging. Token-level importance ratios often exhibit high variance-a phenomenon exacerbated in Mixture-of-Experts models-leading to unstable updates. Existing group-based policy optimization methods, such as GSPO and GRPO, alleviate this problem via hard clipping, making it difficult to maintain both stability and effective learning. We propose Soft Adaptive Policy Optimization (SAPO), which replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates off-policy updates while preserving useful learning signals. Compared with GSPO and GRPO, SAPO is both sequence-coherent and token-adaptive. Like GSPO, SAPO maintains sequence-level coherence, but its soft gating forms a continuous trust region that avoids the brittle hard clipping band used in GSPO. When a sequence contains a few highly off-policy tokens, GSPO suppresses all gradients for that sequence, whereas SAPO selectively down-weights only the offending tokens and preserves the learning signal from the near-on-policy ones, improving sample efficiency. Relative to GRPO, SAPO replaces hard token-level clipping with smooth, temperature-controlled scaling, enabling more informative and stable updates. Empirical results on mathematical reasoning benchmarks indicate that SAPO exhibits improved training stability and higher Pass@1 performance under comparable training budgets. Moreover, we employ SAPO to train the Qwen3-VL model series, demonstrating that SAPO yields consistent performance gains across diverse tasks and different model sizes. Overall, SAPO provides a more reliable, scalable, and effective optimization strategy for RL training of LLMs. 

---
# Geometry of Decision Making in Language Models 

**Authors**: Abhinav Joshi, Divyanshu Bhatt, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20315)  

**Abstract**: Large Language Models (LLMs) show strong generalization across diverse tasks, yet the internal decision-making processes behind their predictions remain opaque. In this work, we study the geometry of hidden representations in LLMs through the lens of \textit{intrinsic dimension} (ID), focusing specifically on decision-making dynamics in a multiple-choice question answering (MCQA) setting. We perform a large-scale study, with 28 open-weight transformer models and estimate ID across layers using multiple estimators, while also quantifying per-layer performance on MCQA tasks. Our findings reveal a consistent ID pattern across models: early layers operate on low-dimensional manifolds, middle layers expand this space, and later layers compress it again, converging to decision-relevant representations. Together, these results suggest LLMs implicitly learn to project linguistic inputs onto structured, low-dimensional manifolds aligned with task-specific decisions, providing new geometric insights into how generalization and reasoning emerge in language models. 

---
# Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits 

**Authors**: Areeb Ahmad, Abhinav Joshi, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2511.20273)  

**Abstract**: Transformer-based language models exhibit complex and distributed behavior, yet their internal computations remain poorly understood. Existing mechanistic interpretability methods typically treat attention heads and multilayer perceptron layers (MLPs) (the building blocks of a transformer architecture) as indivisible units, overlooking possibilities of functional substructure learned within them. In this work, we introduce a more fine-grained perspective that decomposes these components into orthogonal singular directions, revealing superposed and independent computations within a single head or MLP. We validate our perspective on widely used standard tasks like Indirect Object Identification (IOI), Gender Pronoun (GP), and Greater Than (GT), showing that previously identified canonical functional heads, such as the name mover, encode multiple overlapping subfunctions aligned with distinct singular directions. Nodes in a computational graph, that are previously identified as circuit elements show strong activation along specific low-rank directions, suggesting that meaningful computations reside in compact subspaces. While some directions remain challenging to interpret fully, our results highlight that transformer computations are more distributed, structured, and compositional than previously assumed. This perspective opens new avenues for fine-grained mechanistic interpretability and a deeper understanding of model internals. 

---
# The Devil in the Details: Emergent Misalignment, Format and Coherence in Open-Weights LLMs 

**Authors**: Craig Dickson  

**Link**: [PDF](https://arxiv.org/pdf/2511.20104)  

**Abstract**: Prior work has shown that fine-tuning models on a narrow domain with misaligned data can lead to broad misalignment - a phenomenon termed "emergent misalignment" (Betley et al. 2025). While all tested models were susceptible to emergent misalignment, some models showed more resistance than others. Specifically the Qwen-2.5 family proved to be relatively resistant, while GPT-4o exhibited the strongest misalignment. In this paper we evaluate if current-generation open-weights models exhibit similar resistance to the Qwen-2.5 family and measure misalignment robustness over a range of model architectures and scales.
We replicate the effect across nine modern open-weights models (Gemma 3 and Qwen 3 families, 1B-32B parameters). Models fine-tuned on insecure code generation show a 0.68% misalignment rate (compared to 0.07% for base models), matching the lower end of prior open-model results but dramatically lower than GPT-4o's 20%.
We identify a critical format-dependent vulnerability: requiring JSON output doubles misalignment rates compared to natural language prompts (0.96% vs 0.42%). This suggests that structural constraints may bypass safety training by reducing the model's 'degrees of freedom' to refuse. These findings confirm emergent misalignment as a reproducible phenomenon in modern open-weights models, with rates substantially lower than observed in proprietary systems. 

---
# QiMeng-Kernel: Macro-Thinking Micro-Coding Paradigm for LLM-Based High-Performance GPU Kernel Generation 

**Authors**: Xinguo Zhu, Shaohui Peng, Jiaming Guo, Yunji Chen, Qi Guo, Yuanbo Wen, Hang Qin, Ruizhi Chen, Qirui Zhou, Ke Gao, Yanjun Wu, Chen Zhao, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.20100)  

**Abstract**: Developing high-performance GPU kernels is critical for AI and scientific computing, but remains challenging due to its reliance on expert crafting and poor portability. While LLMs offer promise for automation, both general-purpose and finetuned LLMs suffer from two fundamental and conflicting limitations: correctness and efficiency. The key reason is that existing LLM-based approaches directly generate the entire optimized low-level programs, requiring exploration of an extremely vast space encompassing both optimization policies and implementation codes. To address the challenge of exploring an intractable space, we propose Macro Thinking Micro Coding (MTMC), a hierarchical framework inspired by the staged optimization strategy of human experts. It decouples optimization strategy from implementation details, ensuring efficiency through high-level strategy and correctness through low-level implementation. Specifically, Macro Thinking employs reinforcement learning to guide lightweight LLMs in efficiently exploring and learning semantic optimization strategies that maximize hardware utilization. Micro Coding leverages general-purpose LLMs to incrementally implement the stepwise optimization proposals from Macro Thinking, avoiding full-kernel generation errors. Together, they effectively navigate the vast optimization space and intricate implementation details, enabling LLMs for high-performance GPU kernel generation. Comprehensive results on widely adopted benchmarks demonstrate the superior performance of MTMC on GPU kernel generation in both accuracy and running time. On KernelBench, MTMC achieves near 100% and 70% accuracy at Levels 1-2 and 3, over 50% than SOTA general-purpose and domain-finetuned LLMs, with up to 7.3x speedup over LLMs, and 2.2x over expert-optimized PyTorch Eager kernels. On the more challenging TritonBench, MTMC attains up to 59.64% accuracy and 34x speedup. 

---
# EfficientXpert: Efficient Domain Adaptation for Large Language Models via Propagation-Aware Pruning 

**Authors**: Songlin Zhao, Michael Pitts, Zhuwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.19935)  

**Abstract**: The rapid advancement of large language models (LLMs) has increased the demand for domain-specialized variants in areas such as law, healthcare, and finance. However, their large size remains a barrier to deployment in resource-constrained environments, and existing compression methods either generalize poorly across domains or incur high overhead. In this work, we propose \textbf{EfficientXpert}, a lightweight domain-pruning framework that combines a propagation-aware pruning criterion (Foresight Mask) with an efficient adapter-update algorithm (Partial Brain Surgeon). Integrated into the LoRA fine-tuning process, EfficientXpert enables a one-step transformation of general pretrained models into sparse, domain-adapted experts. Across health and legal tasks, it retains up to 98% of dense-model performance at 40% sparsity, outperforming state-of-the-art methods. Further analysis reveals substantial domain-dependent structural shifts that degrade the effectiveness of general pruning masks, underscoring the need for adaptive, domain-aware pruning strategies tailored to each domain. 

---
# CounterVQA: Evaluating and Improving Counterfactual Reasoning in Vision-Language Models for Video Understanding 

**Authors**: Yuefei Chen, Jiang Liu, Xiaodong Lin, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19923)  

**Abstract**: Vision Language Models (VLMs) have recently shown significant advancements in video understanding, especially in feature alignment, event reasoning, and instruction-following tasks. However, their capability for counterfactual reasoning, inferring alternative outcomes under hypothetical conditions, remains underexplored. This capability is essential for robust video understanding, as it requires identifying underlying causal structures and reasoning about unobserved possibilities, rather than merely recognizing observed patterns. To systematically evaluate this capability, we introduce CounterVQA, a video-based benchmark featuring three progressive difficulty levels that assess different aspects of counterfactual reasoning. Through comprehensive evaluation of both state-of-the-art open-source and closed-source models, we uncover a substantial performance gap: while these models achieve reasonable accuracy on simple counterfactual questions, performance degrades significantly on complex multi-hop causal chains. To address these limitations, we develop a post-training method, CFGPT, that enhances a model's visual counterfactual reasoning ability by distilling its counterfactual reasoning capability from the language modality, yielding consistent improvements across all CounterVQA difficulty levels. Dataset and code will be further released. 

---
# MAPS: Preserving Vision-Language Representations via Module-Wise Proximity Scheduling for Better Vision-Language-Action Generalization 

**Authors**: Chengyue Huang, Mellon M. Zhang, Robert Azarcon, Glen Chou, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2511.19878)  

**Abstract**: Vision-Language-Action (VLA) models inherit strong priors from pretrained Vision-Language Models (VLMs), but naive fine-tuning often disrupts these representations and harms generalization. Existing fixes -- freezing modules or applying uniform regularization -- either overconstrain adaptation or ignore the differing roles of VLA components. We present MAPS (Module-Wise Proximity Scheduling), the first robust fine-tuning framework for VLAs. Through systematic analysis, we uncover an empirical order in which proximity constraints should be relaxed to balance stability and flexibility. MAPS linearly schedules this relaxation, enabling visual encoders to stay close to their pretrained priors while action-oriented language layers adapt more freely. MAPS introduces no additional parameters or data, and can be seamlessly integrated into existing VLAs. Across MiniVLA-VQ, MiniVLA-OFT, OpenVLA-OFT, and challenging benchmarks such as SimplerEnv, CALVIN, LIBERO, as well as real-world evaluations on the Franka Emika Panda platform, MAPS consistently boosts both in-distribution and out-of-distribution performance (up to +30%). Our findings highlight empirically guided proximity to pretrained VLMs as a simple yet powerful principle for preserving broad generalization in VLM-to-VLA transfer. 

---
# CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception 

**Authors**: Miguel Carvalho, Helder Dias, Bruno Martins  

**Link**: [PDF](https://arxiv.org/pdf/2511.19820)  

**Abstract**: Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting. 

---
# Training-Free Generation of Diverse and High-Fidelity Images via Prompt Semantic Space Optimization 

**Authors**: Debin Meng, Chen Jin, Zheng Gao, Yanran Li, Ioannis Patras, Georgios Tzimiropoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.19811)  

**Abstract**: Image diversity remains a fundamental challenge for text-to-image diffusion models. Low-diversity models tend to generate repetitive outputs, increasing sampling redundancy and hindering both creative exploration and downstream applications. A primary cause is that generation often collapses toward a strong mode in the learned distribution. Existing attempts to improve diversity, such as noise resampling, prompt rewriting, or steering-based guidance, often still collapse to dominant modes or introduce distortions that degrade image quality. In light of this, we propose Token-Prompt embedding Space Optimization (TPSO), a training-free and model-agnostic module. TPSO introduces learnable parameters to explore underrepresented regions of the token embedding space, reducing the tendency of the model to repeatedly generate samples from strong modes of the learned distribution. At the same time, the prompt-level space provides a global semantic constraint that regulates distribution shifts, preventing quality degradation while maintaining high fidelity. Extensive experiments on MS-COCO and three diffusion backbones show that TPSO significantly enhances generative diversity, improving baseline performance from 1.10 to 4.18 points, without sacrificing image quality. Code will be released upon acceptance. 

---
# Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs 

**Authors**: Meng Lu, Ran Xu, Yi Fang, Wenxuan Zhang, Yue Yu, Gaurav Srivastava, Yuchen Zhuang, Mohamed Elhoseiny, Charles Fleming, Carl Yang, Zhengzhong Tu, Yang Xie, Guanghua Xiao, Hanrui Wang, Di Jin, Wenqi Shi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.19773)  

**Abstract**: While recent vision-language models (VLMs) demonstrate strong image understanding, their ability to "think with images", i.e., to reason through multi-step visual interactions, remains limited. We introduce VISTA-Gym, a scalable training environment for incentivizing tool-integrated visual reasoning capabilities in VLMs. VISTA-Gym unifies diverse real-world multimodal reasoning tasks (7 tasks from 13 datasets in total) with a standardized interface for visual tools (e.g., grounding, parsing), executable interaction loops, verifiable feedback signals, and efficient trajectory logging, enabling visual agentic reinforcement learning at scale. While recent VLMs exhibit strong text-only reasoning, both proprietary and open-source models still struggle with tool selection, invocation, and coordination. With VISTA-Gym, we train VISTA-R1 to interleave tool-use with agentic reasoning via multi-turn trajectory sampling and end-to-end reinforcement learning. Extensive experiments across 11 public reasoning-intensive VQA benchmarks show that VISTA-R1-8B outperforms state-of-the-art baselines with similar sizes by 9.51%-18.72%, demonstrating VISTA-Gym as an effective training ground to unlock the tool-integrated reasoning capabilities for VLMs. 

---
# Fara-7B: An Efficient Agentic Model for Computer Use 

**Authors**: Ahmed Awadallah, Yash Lara, Raghav Magazine, Hussein Mozannar, Akshay Nambi, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Vibhav Vineet, Spencer Whitehead, Andrew Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.19663)  

**Abstract**: Progress in computer use agents (CUAs) has been constrained by the absence of large and high-quality datasets that capture how humans interact with a computer. While LLMs have thrived on abundant textual data, no comparable corpus exists for CUA trajectories. To address these gaps, we introduce FaraGen, a novel synthetic data generation system for multi-step web tasks. FaraGen can propose diverse tasks from frequently used websites, generate multiple solution attempts, and filter successful trajectories using multiple verifiers. It achieves high throughput, yield, and diversity for multi-step web tasks, producing verified trajectories at approximately $1 each. We use this data to train Fara-7B, a native CUA model that perceives the computer using only screenshots, executes actions via predicted coordinates, and is small enough to run on-device. We find that Fara-7B outperforms other CUA models of comparable size on benchmarks like WebVoyager, Online-Mind2Web, and WebTailBench -- our novel benchmark that better captures under-represented web tasks in pre-existing benchmarks. Furthermore, Fara-7B is competitive with much larger frontier models, illustrating key benefits of scalable data generation systems in advancing small efficient agentic models. We are making Fara-7B open-weight on Microsoft Foundry and HuggingFace, and we are releasing WebTailBench. 

---
# Studying Maps at Scale: A Digital Investigation of Cartography and the Evolution of Figuration 

**Authors**: Remi Petitpierre  

**Link**: [PDF](https://arxiv.org/pdf/2511.19538)  

**Abstract**: This thesis presents methods and datasets to investigate cartographic heritage on a large scale and from a cultural perspective. Heritage institutions worldwide have digitized more than one million maps, and automated techniques now enable large-scale recognition and extraction of map content. Yet these methods have engaged little with the history of cartography, or the view that maps are semantic-symbolic systems, and cultural objects reflecting political and epistemic expectations. This work leverages a diverse corpus of 771,561 map records and 99,715 digitized images aggregated from 38 digital catalogs. After normalization, the dataset includes 236,925 contributors and spans six centuries, from 1492 to 1948. These data make it possible to chart geographic structures and the global chronology of map publication. The spatial focus of cartography is analyzed in relation to political dynamics, evidencing links between Atlantic maritime charting, the triangular trade, and colonial expansion. Further results document the progression of national, domestic focus and the impact of military conflicts on publication volumes. The research introduces semantic segmentation techniques and object detection models for the generic recognition of land classes and cartographic signs, trained on annotated data and synthetic images. The analysis of land classes shows that maps are designed images whose framing and composition emphasize features through centering and semantic symmetries. The study of cartographic figuration encodes 63 M signs and 25 M fragments into a latent visual space, revealing figurative shifts such as the replacement of relief hachures by terrain contours and showing that signs tend to form locally consistent systems. Analyses of collaboration and diffusion highlight the role of legitimacy, larger actors, and major cities in the spread of figurative norms and semiotic cultures. 

---
# Quantifying Modality Contributions via Disentangling Multimodal Representations 

**Authors**: Padegal Amit, Omkar Mahesh Kashyap, Namitha Rayasam, Nidhi Shekhar, Surabhi Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2511.19470)  

**Abstract**: Quantifying modality contributions in multimodal models remains a challenge, as existing approaches conflate the notion of contribution itself. Prior work relies on accuracy-based approaches, interpreting performance drops after removing a modality as indicative of its influence. However, such outcome-driven metrics fail to distinguish whether a modality is inherently informative or whether its value arises only through interaction with other modalities. This distinction is particularly important in cross-attention architectures, where modalities influence each other's representations. In this work, we propose a framework based on Partial Information Decomposition (PID) that quantifies modality contributions by decomposing predictive information in internal embeddings into unique, redundant, and synergistic components. To enable scalable, inference-only analysis, we develop an algorithm based on the Iterative Proportional Fitting Procedure (IPFP) that computes layer and dataset-level contributions without retraining. This provides a principled, representation-level view of multimodal behavior, offering clearer and more interpretable insights than outcome-based metrics. 

---
# BlockCert: Certified Blockwise Extraction of Transformer Mechanisms 

**Authors**: Sandro Andric  

**Link**: [PDF](https://arxiv.org/pdf/2511.17645)  

**Abstract**: Mechanistic interpretability aspires to reverse-engineer neural networks into explicit algorithms, while model editing seeks to modify specific behaviours without retraining. Both areas are typically evaluated with informal evidence and ad-hoc experiments, with few explicit guarantees about how far an extracted or edited model can drift from the original on relevant inputs. We introduce BlockCert, a framework for certified blockwise extraction of transformer mechanisms, and outline how a lightweight extension can support certified local edits. Given a pre-trained transformer and a prompt distribution, BlockCert extracts structured surrogate implementations for residual blocks together with machine-checkable certificates that bound approximation error, record coverage metrics, and hash the underlying artifacts. We formalize a simple Lipschitz-based composition theorem in Lean 4 that lifts these local guarantees to a global deviation bound. Empirically, we apply the framework to GPT-2 small, TinyLlama-1.1B-Chat, and Llama-3.2-3B. Across these models we obtain high per-block coverage and small residual errors on the evaluated prompts, and in the TinyLlama setting we show that a fully stitched model matches the baseline perplexity within approximately 6e-5 on stress prompts. Our results suggest that blockwise extraction with explicit certificates is feasible for real transformer language models and offers a practical bridge between mechanistic interpretability and formal reasoning about model behaviour. 

---
