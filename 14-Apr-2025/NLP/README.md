# SWAN-GPT: An Efficient and Scalable Approach for Long-Context Language Modeling 

**Authors**: Krishna C. Puvvada, Faisal Ladhak, Santiago Akle Serrano, Cheng-Ping Hsieh, Shantanu Acharya, Somshubra Majumdar, Fei Jia, Samuel Kriman, Simeng Sun, Dima Rekesh, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2504.08719)  

**Abstract**: We present a decoder-only Transformer architecture that robustly generalizes to sequence lengths substantially longer than those seen during training. Our model, SWAN-GPT, interleaves layers without positional encodings (NoPE) and sliding-window attention layers equipped with rotary positional encodings (SWA-RoPE). Experiments demonstrate strong performance on sequence lengths significantly longer than the training length without the need for additional long-context training. This robust length extrapolation is achieved through our novel architecture, enhanced by a straightforward dynamic scaling of attention scores during inference. In addition, SWAN-GPT is more computationally efficient than standard GPT architectures, resulting in cheaper training and higher throughput. Further, we demonstrate that existing pre-trained decoder-only models can be efficiently converted to the SWAN architecture with minimal continued training, enabling longer contexts. Overall, our work presents an effective approach for scaling language models to longer contexts in a robust and efficient manner. 

---
# ModernBERT or DeBERTaV3? Examining Architecture and Data Influence on Transformer Encoder Models Performance 

**Authors**: Wissam Antoun, Benoît Sagot, Djamé Seddah  

**Link**: [PDF](https://arxiv.org/pdf/2504.08716)  

**Abstract**: Pretrained transformer-encoder models like DeBERTaV3 and ModernBERT introduce architectural advancements aimed at improving efficiency and performance. Although the authors of ModernBERT report improved performance over DeBERTaV3 on several benchmarks, the lack of disclosed training data and the absence of comparisons using a shared dataset make it difficult to determine whether these gains are due to architectural improvements or differences in training data. In this work, we conduct a controlled study by pretraining ModernBERT on the same dataset as CamemBERTaV2, a DeBERTaV3 French model, isolating the effect of model design. Our results show that the previous model generation remains superior in sample efficiency and overall benchmark performance, with ModernBERT's primary advantage being faster training and inference speed. However, the new proposed model still provides meaningful architectural improvements compared to earlier models such as BERT and RoBERTa. Additionally, we observe that high-quality pre-training data accelerates convergence but does not significantly improve final performance, suggesting potential benchmark saturation. These findings show the importance of disentangling pretraining data from architectural innovations when evaluating transformer models. 

---
# Large Language Models as Span Annotators 

**Authors**: Zdeněk Kasner, Vilém Zouhar, Patrícia Schmidtová, Ivan Kartáč, Kristýna Onderková, Ondřej Plátek, Dimitra Gkatzia, Saad Mahamood, Ondřej Dušek, Simone Balloccu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08697)  

**Abstract**: For high-quality texts, single-score metrics seldom provide actionable feedback. In contrast, span annotation - pointing out issues in the text by annotating their spans - can guide improvements and provide insights. Until recently, span annotation was limited to human annotators or fine-tuned encoder models. In this study, we automate span annotation with large language models (LLMs). We compare expert or skilled crowdworker annotators with open and proprietary LLMs on three tasks: data-to-text generation evaluation, machine translation evaluation, and propaganda detection in human-written texts. In our experiments, we show that LLMs as span annotators are straightforward to implement and notably more cost-efficient than human annotators. The LLMs achieve moderate agreement with skilled human annotators, in some scenarios comparable to the average agreement among the annotators themselves. Qualitative analysis shows that reasoning models outperform their instruction-tuned counterparts and provide more valid explanations for annotations. We release the dataset of more than 40k model and human annotations for further research. 

---
# TP-RAG: Benchmarking Retrieval-Augmented Large Language Model Agents for Spatiotemporal-Aware Travel Planning 

**Authors**: Hang Ni, Fan Liu, Xinyu Ma, Lixin Su, Shuaiqiang Wang, Dawei Yin, Hui Xiong, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08694)  

**Abstract**: Large language models (LLMs) have shown promise in automating travel planning, yet they often fall short in addressing nuanced spatiotemporal rationality. While existing benchmarks focus on basic plan validity, they neglect critical aspects such as route efficiency, POI appeal, and real-time adaptability. This paper introduces TP-RAG, the first benchmark tailored for retrieval-augmented, spatiotemporal-aware travel planning. Our dataset includes 2,348 real-world travel queries, 85,575 fine-grain annotated POIs, and 18,784 high-quality travel trajectory references sourced from online tourist documents, enabling dynamic and context-aware planning. Through extensive experiments, we reveal that integrating reference trajectories significantly improves spatial efficiency and POI rationality of the travel plan, while challenges persist in universality and robustness due to conflicting references and noisy data. To address these issues, we propose EvoRAG, an evolutionary framework that potently synergizes diverse retrieved trajectories with LLMs' intrinsic reasoning. EvoRAG achieves state-of-the-art performance, improving spatiotemporal compliance and reducing commonsense violation compared to ground-up and retrieval-augmented baselines. Our work underscores the potential of hybridizing Web knowledge with LLM-driven optimization, paving the way for more reliable and adaptive travel planning agents. 

---
# Fast-Slow-Thinking: Complex Task Solving with Large Language Models 

**Authors**: Yiliu Sun, Yanfang Zhang, Zicheng Zhao, Sheng Wan, Dacheng Tao, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08690)  

**Abstract**: Nowadays, Large Language Models (LLMs) have been gradually employed to solve complex tasks. To face the challenge, task decomposition has become an effective way, which proposes to divide a complex task into multiple simpler subtasks and then solve them separately so that the difficulty of the original task can be reduced. However, the performance of existing task decomposition methods can be suboptimal when the task contains overly complex logic and constraints. In this situation, the solution generated by LLMs may deviate from the original purpose of the task, or contain redundant or even erroneous content. Therefore, inspired by the fact that humans possess two thinking systems including fast thinking and slow thinking, this paper introduces a new task decomposition method termed ``Fast-Slow-Thinking'' (FST), which stimulates LLMs to solve tasks through the cooperation of Fast Thinking (FT) and Slow Thinking (ST) steps. Here FT focuses more on the general and concise aspect of the task, and ST focuses more on the details of the task. In FT, LLMs are prompted to remove the constraints of the original task, therefore simplifying it to a general and concise one. In ST, we recall the constraints removed in FT, so that LLMs can improve the answer generated in FT to meet the requirements of the original task. Therefore, our FST method enables LLMs to consider a complex problem via a human-like cognition process from coarse to fine, the effectiveness of which has been well demonstrated by the experiments on three types of tasks. 

---
# Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning 

**Authors**: Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08672)  

**Abstract**: Advancing LLM reasoning skills has captivated wide interest. However, current post-training techniques rely heavily on supervisory signals, such as outcome supervision or auxiliary reward models, which face the problem of scalability and high annotation costs. This motivates us to enhance LLM reasoning without the need for external supervision. We introduce a generalizable and purely unsupervised self-training framework, named Genius. Without external auxiliary, Genius requires to seek the optimal response sequence in a stepwise manner and optimize the LLM. To explore the potential steps and exploit the optimal ones, Genius introduces a stepwise foresight re-sampling strategy to sample and estimate the step value by simulating future outcomes. Further, we recognize that the unsupervised setting inevitably induces the intrinsic noise and uncertainty. To provide a robust optimization, we propose an advantage-calibrated optimization (ACO) loss function to mitigate estimation inconsistencies. Combining these techniques together, Genius provides an advanced initial step towards self-improve LLM reasoning with general queries and without supervision, revolutionizing reasoning scaling laws given the vast availability of general queries. The code will be released at this https URL. 

---
# A Survey of Machine Learning Models and Datasets for the Multi-label Classification of Textual Hate Speech in English 

**Authors**: Julian Bäumler, Louis Blöcher, Lars-Joel Frey, Xian Chen, Markus Bayer, Christian Reuter  

**Link**: [PDF](https://arxiv.org/pdf/2504.08609)  

**Abstract**: The dissemination of online hate speech can have serious negative consequences for individuals, online communities, and entire societies. This and the large volume of hateful online content prompted both practitioners', i.e., in content moderation or law enforcement, and researchers' interest in machine learning models to automatically classify instances of hate speech. Whereas most scientific works address hate speech classification as a binary task, practice often requires a differentiation into sub-types, e.g., according to target, severity, or legality, which may overlap for individual content. Hence, researchers created datasets and machine learning models that approach hate speech classification in textual data as a multi-label problem. This work presents the first systematic and comprehensive survey of scientific literature on this emerging research landscape in English (N=46). We contribute with a concise overview of 28 datasets suited for training multi-label classification models that reveals significant heterogeneity regarding label-set, size, meta-concept, annotation process, and inter-annotator agreement. Our analysis of 24 publications proposing suitable classification models further establishes inconsistency in evaluation and a preference for architectures based on Bidirectional Encoder Representation from Transformers (BERT) and Recurrent Neural Networks (RNNs). We identify imbalanced training data, reliance on crowdsourcing platforms, small and sparse datasets, and missing methodological alignment as critical open issues and formulate ten recommendations for research. 

---
# MedHal: An Evaluation Dataset for Medical Hallucination Detection 

**Authors**: Gaya Mehenni, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2504.08596)  

**Abstract**: We present MedHal, a novel large-scale dataset specifically designed to evaluate if models can detect hallucinations in medical texts. Current hallucination detection methods face significant limitations when applied to specialized domains like medicine, where they can have disastrous consequences. Existing medical datasets are either too small, containing only a few hundred samples, or focus on a single task like Question Answering or Natural Language Inference. MedHal addresses these gaps by: (1) incorporating diverse medical text sources and tasks; (2) providing a substantial volume of annotated samples suitable for training medical hallucination detection models; and (3) including explanations for factual inconsistencies to guide model learning. We demonstrate MedHal's utility by training and evaluating a baseline medical hallucination detection model, showing improvements over general-purpose hallucination detection approaches. This resource enables more efficient evaluation of medical text generation systems while reducing reliance on costly expert review, potentially accelerating the development of medical AI research. 

---
# Playpen: An Environment for Exploring Learning Through Conversational Interaction 

**Authors**: Nicola Horst, Davide Mazzaccara, Antonia Schmidt, Michael Sullivan, Filippo Momentè, Luca Franceschetti, Philipp Sadler, Sherzod Hakimov, Alberto Testoni, Raffaella Bernardi, Raquel Fernández, Alexander Koller, Oliver Lemon, David Schlangen, Mario Giulianelli, Alessandro Suglia  

**Link**: [PDF](https://arxiv.org/pdf/2504.08590)  

**Abstract**: Are we running out of learning signal? Predicting the next word in an existing text has turned out to be a powerful signal, at least at scale. But there are signs that we are running out of this resource. In recent months, interaction between learner and feedback-giver has come into focus, both for "alignment" (with a reward model judging the quality of instruction following attempts) and for improving "reasoning" (process- and outcome-based verifiers judging reasoning steps). In this paper, we explore to what extent synthetic interaction in what we call Dialogue Games -- goal-directed and rule-governed activities driven predominantly by verbal actions -- can provide a learning signal, and how this signal can be used. We introduce an environment for producing such interaction data (with the help of a Large Language Model as counterpart to the learner model), both offline and online. We investigate the effects of supervised fine-tuning on this data, as well as reinforcement learning setups such as DPO, and GRPO; showing that all of these approaches achieve some improvements in in-domain games, but only GRPO demonstrates the ability to generalise to out-of-domain games as well as retain competitive performance in reference-based tasks. We release the framework and the baseline training setups in the hope that this can foster research in this promising new direction. 

---
# UoB-NLP at SemEval-2025 Task 11: Leveraging Adapters for Multilingual and Cross-Lingual Emotion Detection 

**Authors**: Frances Laureano De Leon, Yixiao Wang, Yue Feng, Mark G. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.08543)  

**Abstract**: Emotion detection in natural language processing is a challenging task due to the complexity of human emotions and linguistic diversity. While significant progress has been made in high-resource languages, emotion detection in low-resource languages remains underexplored. In this work, we address multilingual and cross-lingual emotion detection by leveraging adapter-based fine-tuning with multilingual pre-trained language models. Adapters introduce a small number of trainable parameters while keeping the pre-trained model weights fixed, offering a parameter-efficient approach to adaptation. We experiment with different adapter tuning strategies, including task-only adapters, target-language-ready task adapters, and language-family-based adapters. Our results show that target-language-ready task adapters achieve the best overall performance, particularly for low-resource African languages with our team ranking 7th for Tigrinya, and 8th for Kinyarwanda in Track A. In Track C, our system ranked 3rd for Amharic, and 4th for Oromo, Tigrinya, Kinyarwanda, Hausa, and Igbo. Our approach outperforms large language models in 11 languages and matches their performance in four others, despite our models having significantly fewer parameters. Furthermore, we find that adapter-based models retain cross-linguistic transfer capabilities while requiring fewer computational resources compared to full fine-tuning for each language. 

---
# Lexical Bundle Frequency as a Construct-Relevant Candidate Feature in Automated Scoring of L2 Academic Writing 

**Authors**: Burak Senel  

**Link**: [PDF](https://arxiv.org/pdf/2504.08537)  

**Abstract**: Automated scoring (AS) systems are increasingly used for evaluating L2 writing, but require ongoing refinement for construct validity. While prior work suggested lexical bundles (LBs) - recurrent multi-word sequences satisfying certain frequency criteria - could inform assessment, their empirical integration into AS models needs further investigation. This study tested the impact of incorporating LB frequency features into an AS model for TOEFL independent writing tasks. Analyzing a sampled subcorpus (N=1,225 essays, 9 L1s) from the TOEFL11 corpus, scored by ETS-trained raters (Low, Medium, High), 3- to 9-word LBs were extracted, distinguishing prompt-specific from non-prompt types. A baseline Support Vector Machine (SVM) scoring model using established linguistic features (e.g., mechanics, cohesion, sophistication) was compared against an extended model including three aggregate LB frequency features (total prompt, total non-prompt, overall total). Results revealed significant, though generally small-effect, relationships between LB frequency (especially non-prompt bundles) and proficiency (p < .05). Mean frequencies suggested lower proficiency essays used more LBs overall. Critically, the LB-enhanced model improved agreement with human raters (Quadratic Cohen's Kappa +2.05%, overall Cohen's Kappa +5.63%), with notable gains for low (+10.1% exact agreement) and medium (+14.3% Cohen's Kappa) proficiency essays. These findings demonstrate that integrating aggregate LB frequency offers potential for developing more linguistically informed and accurate AS systems, particularly for differentiating developing L2 writers. 

---
# On The Landscape of Spoken Language Models: A Comprehensive Survey 

**Authors**: Siddhant Arora, Kai-Wei Chang, Chung-Ming Chien, Yifan Peng, Haibin Wu, Yossi Adi, Emmanuel Dupoux, Hung-Yi Lee, Karen Livescu, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2504.08528)  

**Abstract**: The field of spoken language processing is undergoing a shift from training custom-built, task-specific models toward using and optimizing spoken language models (SLMs) which act as universal speech processing systems. This trend is similar to the progression toward universal language models that has taken place in the field of (text) natural language processing. SLMs include both "pure" language models of speech -- models of the distribution of tokenized speech sequences -- and models that combine speech encoders with text language models, often including both spoken and written input or output. Work in this area is very diverse, with a range of terminology and evaluation settings. This paper aims to contribute an improved understanding of SLMs via a unifying literature survey of recent work in the context of the evolution of the field. Our survey categorizes the work in this area by model architecture, training, and evaluation choices, and describes some key challenges and directions for future work. 

---
# Integrated ensemble of BERT- and features-based models for authorship attribution in Japanese literary works 

**Authors**: Taisei Kanda, Mingzhe Jin, Wataru Zaitsu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08527)  

**Abstract**: Traditionally, authorship attribution (AA) tasks relied on statistical data analysis and classification based on stylistic features extracted from texts. In recent years, pre-trained language models (PLMs) have attracted significant attention in text classification tasks. However, although they demonstrate excellent performance on large-scale short-text datasets, their effectiveness remains under-explored for small samples, particularly in AA tasks. Additionally, a key challenge is how to effectively leverage PLMs in conjunction with traditional feature-based methods to advance AA research. In this study, we aimed to significantly improve performance using an integrated integrative ensemble of traditional feature-based and modern PLM-based methods on an AA task in a small sample. For the experiment, we used two corpora of literary works to classify 10 authors each. The results indicate that BERT is effective, even for small-sample AA tasks. Both BERT-based and classifier ensembles outperformed their respective stand-alone models, and the integrated ensemble approach further improved the scores significantly. For the corpus that was not included in the pre-training data, the integrated ensemble improved the F1 score by approximately 14 points, compared to the best-performing single model. Our methodology provides a viable solution for the efficient use of the ever-expanding array of data processing tools in the foreseeable future. 

---
# Beyond Self-Reports: Multi-Observer Agents for Personality Assessment in Large Language Models 

**Authors**: Yin Jou Huang, Rafik Hadfi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08399)  

**Abstract**: There is a growing interest in assessing the personality traits of Large language models (LLMs). However, traditional personality assessments based on self-report questionnaires may fail to capture their true behavioral nuances due to inherent biases and meta-knowledge contamination. This paper introduces a novel multi-observer framework for LLM personality assessment that draws inspiration from informant-report methods in psychology. Instead of relying solely on self-assessments, our approach employs multiple observer agents configured with a specific relationship context (e.g., family, friend, or workplace) to simulate interactive scenarios with a subject LLM. These observers engage in dialogues and subsequently provide ratings across the Big Five personality dimensions. Our experiments reveal that LLMs possess systematic biases in self-report personality ratings. Moreover, aggregating observer ratings effectively reduces non-systematic biases and achieves optimal reliability with 5-7 observers. The findings highlight the significant impact of relationship context on personality perception and demonstrate that a multi-observer paradigm yields a more robust and context-sensitive evaluation of LLM personality traits. 

---
# Scholar Inbox: Personalized Paper Recommendations for Scientists 

**Authors**: Markus Flicke, Glenn Angrabeit, Madhav Iyengar, Vitalii Protsenko, Illia Shakun, Jovan Cicvaric, Bora Kargi, Haoyu He, Lukas Schuler, Lewin Scholz, Kavyanjali Agnihotri, Yong Cao, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2504.08385)  

**Abstract**: Scholar Inbox is a new open-access platform designed to address the challenges researchers face in staying current with the rapidly expanding volume of scientific literature. We provide personalized recommendations, continuous updates from open-access archives (arXiv, bioRxiv, etc.), visual paper summaries, semantic search, and a range of tools to streamline research workflows and promote open research access. The platform's personalized recommendation system is trained on user ratings, ensuring that recommendations are tailored to individual researchers' interests. To further enhance the user experience, Scholar Inbox also offers a map of science that provides an overview of research across domains, enabling users to easily explore specific topics. We use this map to address the cold start problem common in recommender systems, as well as an active learning strategy that iteratively prompts users to rate a selection of papers, allowing the system to learn user preferences quickly. We evaluate the quality of our recommendation system on a novel dataset of 800k user ratings, which we make publicly available, as well as via an extensive user study. this https URL 

---
# Large language models could be rote learners 

**Authors**: Yuyang Xu, Renjun Hu, Haochao Ying, Jian Wu, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.08300)  

**Abstract**: Multiple-choice question (MCQ) benchmarks are widely used for evaluating Large Language Models (LLMs), yet their reliability is undermined by benchmark contamination. In this study, we reframe contamination as an inherent aspect of learning and seek to disentangle genuine capability acquisition from superficial memorization in LLM evaluation. First, by analyzing model performance under different memorization conditions, we uncover a counterintuitive trend: LLMs perform worse on memorized MCQs than on non-memorized ones, indicating the coexistence of two distinct learning phenomena, i.e., rote memorization and genuine capability learning. To disentangle them, we propose TrinEval, a novel evaluation framework that reformulates MCQs into an alternative trinity format, reducing memorization while preserving knowledge assessment. Experiments validate TrinEval's effectiveness in reformulation, and its evaluation reveals that common LLMs may memorize by rote 20.5% of knowledge points (in MMLU on average). 

---
# ELSA: A Style Aligned Dataset for Emotionally Intelligent Language Generation 

**Authors**: Vishal Gandhi, Sagar Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08281)  

**Abstract**: Advancements in emotion aware language processing increasingly shape vital NLP applications ranging from conversational AI and affective computing to computational psychology and creative content generation. Existing emotion datasets either lack emotional granularity or fail to capture necessary stylistic diversity, limiting the advancement of effective emotion conditioned text generation systems. Seeking to bridge this crucial gap between granularity and style diversity, this paper introduces a novel systematically constructed dataset named ELSA Emotion and Language Style Alignment Dataset leveraging fine grained emotion taxonomies adapted from existing sources such as dair ai emotion dataset and GoEmotions taxonomy. This dataset comprises multiple emotionally nuanced variations of original sentences regenerated across distinct contextual styles such as conversational, formal, poetic, and narrative, using advanced Large Language Models LLMs. Rigorous computational evaluation using metrics such as perplexity, embedding variance, readability, lexical diversity, and semantic coherence measures validates the datasets emotional authenticity, linguistic fluency, and textual diversity. Comprehensive metric analyses affirm its potential to support deeper explorations into emotion conditioned style adaptive text generation. By enabling precision tuned emotionally nuanced language modeling, our dataset creates fertile ground for research on fine grained emotional control, prompt driven explanation, interpretability, and style adaptive expressive language generation with LLMs. 

---
# Evaluating the Bias in LLMs for Surveying Opinion and Decision Making in Healthcare 

**Authors**: Yonchanok Khaokaew, Flora D. Salim, Andreas Züfle, Hao Xue, Taylor Anderson, Matthew Scotch, David J Heslop  

**Link**: [PDF](https://arxiv.org/pdf/2504.08260)  

**Abstract**: Generative agents have been increasingly used to simulate human behaviour in silico, driven by large language models (LLMs). These simulacra serve as sandboxes for studying human behaviour without compromising privacy or safety. However, it remains unclear whether such agents can truly represent real individuals. This work compares survey data from the Understanding America Study (UAS) on healthcare decision-making with simulated responses from generative agents. Using demographic-based prompt engineering, we create digital twins of survey respondents and analyse how well different LLMs reproduce real-world behaviours. Our findings show that some LLMs fail to reflect realistic decision-making, such as predicting universal vaccine acceptance. However, Llama 3 captures variations across race and Income more accurately but also introduces biases not present in the UAS data. This study highlights the potential of generative agents for behavioural research while underscoring the risks of bias from both LLMs and prompting strategies. 

---
# Out of Style: RAG's Fragility to Linguistic Variation 

**Authors**: Tianyu Cao, Neel Bhandari, Akhila Yerukola, Akari Asai, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2504.08231)  

**Abstract**: Despite the impressive performance of Retrieval-augmented Generation (RAG) systems across various NLP benchmarks, their robustness in handling real-world user-LLM interaction queries remains largely underexplored. This presents a critical gap for practical deployment, where user queries exhibit greater linguistic variations and can trigger cascading errors across interdependent RAG components. In this work, we systematically analyze how varying four linguistic dimensions (formality, readability, politeness, and grammatical correctness) impact RAG performance. We evaluate two retrieval models and nine LLMs, ranging from 3 to 72 billion parameters, across four information-seeking Question Answering (QA) datasets. Our results reveal that linguistic reformulations significantly impact both retrieval and generation stages, leading to a relative performance drop of up to 40.41% in Recall@5 scores for less formal queries and 38.86% in answer match scores for queries containing grammatical errors. Notably, RAG systems exhibit greater sensitivity to such variations compared to LLM-only generations, highlighting their vulnerability to error propagation due to linguistic shifts. These findings highlight the need for improved robustness techniques to enhance reliability in diverse user interactions. 

---
# Big Meaning: Qualitative Analysis on Large Bodies of Data Using AI 

**Authors**: Samuel Flanders, Melati Nungsari, Mark Cheong Wing Loong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08213)  

**Abstract**: This study introduces a framework that leverages AI-generated descriptive codes to indicate a text's fecundity--the density of unique human-generated codes--in thematic analysis. Rather than replacing human interpretation, AI-generated codes guide the selection of texts likely to yield richer qualitative insights. Using a dataset of 2,530 Malaysian news articles on refugee attitudes, we compare AI-selected documents to randomly chosen ones by having three human coders independently derive codes. The results demonstrate that AI-selected texts exhibit approximately twice the fecundity. Our findings support the use of AI-generated codes as an effective proxy for identifying documents with a high potential for meaning-making in thematic analysis. 

---
# LLM for Comparative Narrative Analysis 

**Authors**: Leo Kampen, Carlos Rabat Villarreal, Louis Yu, Santu Karmaker, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08211)  

**Abstract**: In this paper, we conducted a Multi-Perspective Comparative Narrative Analysis (CNA) on three prominent LLMs: GPT-3.5, PaLM2, and Llama2. We applied identical prompts and evaluated their outputs on specific tasks, ensuring an equitable and unbiased comparison between various LLMs. Our study revealed that the three LLMs generated divergent responses to the same prompt, indicating notable discrepancies in their ability to comprehend and analyze the given task. Human evaluation was used as the gold standard, evaluating four perspectives to analyze differences in LLM performance. 

---
# Harnessing the Unseen: The Hidden Influence of Intrinsic Knowledge in Long-Context Language Models 

**Authors**: Yu Fu, Haz Sameen Shahgir, Hui Liu, Xianfeng Tang, Qi He, Yue Dong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08202)  

**Abstract**: Recent advances in long-context models (LCMs), designed to handle extremely long input contexts, primarily focus on utilizing external contextual information, often leaving the influence of large language models' intrinsic knowledge underexplored. In this work, we investigate how this intrinsic knowledge affects content generation and demonstrate that its impact becomes increasingly pronounced as context length extends. Furthermore, we show that the model's ability to utilize intrinsic knowledge, which we call intrinsic retrieval ability, does not improve simultaneously with its ability to leverage contextual knowledge through extrinsic retrieval ability. Moreover, better extrinsic retrieval can interfere with the model's ability to use its own knowledge effectively, limiting its full potential. To bridge this gap, we design a simple yet effective Hybrid Needle-in-a-Haystack test that evaluates models based on their capabilities across both retrieval abilities, rather than solely emphasizing extrinsic retrieval ability. Our experimental results reveal that Qwen-2.5 models significantly outperform Llama-3.1 models, demonstrating superior intrinsic retrieval ability. Moreover, even the more powerful Llama-3.1-70B-Instruct model fails to exhibit better performance under LCM conditions, highlighting the importance of evaluating models from a dual-retrieval perspective. 

---
# Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora 

**Authors**: Alex Warstadt, Aaron Mueller, Leshem Choshen, Ethan Wilcox, Chengxu Zhuang, Juan Ciro, Rafael Mosquera, Bhargavi Paranjape, Adina Williams, Tal Linzen, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2504.08165)  

**Abstract**: Children can acquire language from less than 100 million words of input. Large language models are far less data-efficient: they typically require 3 or 4 orders of magnitude more data and still do not perform as well as humans on many evaluations. These intensive resource demands limit the ability of researchers to train new models and use existing models as developmentally plausible cognitive models. The BabyLM Challenge is a communal effort in which participants compete to optimize language model training on a fixed data budget. Submissions are compared on various evaluation tasks targeting grammatical ability, downstream task performance, and generalization. Participants can submit to up to three tracks with progressively looser data restrictions. From over 30 submissions, we extract concrete recommendations on how best to train data-efficient language models, and on where future efforts should (and perhaps should not) focus. The winning submissions using the LTG-BERT architecture (Samuel et al., 2023) outperformed models trained on trillions of words. Other submissions achieved strong results through training on shorter input sequences or training a student model on a pretrained teacher. Curriculum learning attempts, which accounted for a large number of submissions, were largely unsuccessful, though some showed modest improvements. 

---
# DeepSeek vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization? 

**Authors**: Daniil Larionov, Sotaro Takeshita, Ran Zhang, Yanran Chen, Christoph Leiter, Zhipin Wang, Christian Greisinger, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2504.08120)  

**Abstract**: Reasoning-enabled large language models (LLMs) have recently demonstrated impressive performance in complex logical and mathematical tasks, yet their effectiveness in evaluating natural language generation remains unexplored. This study systematically compares reasoning-based LLMs (DeepSeek-R1 and OpenAI o3) with their non-reasoning counterparts across machine translation (MT) and text summarization (TS) evaluation tasks. We evaluate eight models across three architectural categories, including state-of-the-art reasoning models, their distilled variants (ranging from 8B to 70B parameters), and equivalent conventional, non-reasoning LLMs. Our experiments on WMT23 and SummEval benchmarks reveal that the benefits of reasoning capabilities are highly model and task-dependent: while OpenAI o3-mini models show consistent performance improvements with increased reasoning intensity, DeepSeek-R1 underperforms compared to its non-reasoning variant, with exception to certain aspects of TS evaluation. Correlation analysis demonstrates that increased reasoning token usage positively correlates with evaluation quality in o3-mini models. Furthermore, our results show that distillation of reasoning capabilities maintains reasonable performance in medium-sized models (32B) but degrades substantially in smaller variants (8B). This work provides the first comprehensive assessment of reasoning LLMs for NLG evaluation and offers insights into their practical use. 

---
# Multi-view autoencoders for Fake News Detection 

**Authors**: Ingryd V. S. T. Pereira, George D. C. Cavalcanti, Rafael M. O. Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2504.08102)  

**Abstract**: Given the volume and speed at which fake news spreads across social media, automatic fake news detection has become a highly important task. However, this task presents several challenges, including extracting textual features that contain relevant information about fake news. Research about fake news detection shows that no single feature extraction technique consistently outperforms the others across all scenarios. Nevertheless, different feature extraction techniques can provide complementary information about the textual data and enable a more comprehensive representation of the content. This paper proposes using multi-view autoencoders to generate a joint feature representation for fake news detection by integrating several feature extraction techniques commonly used in the literature. Experiments on fake news datasets show a significant improvement in classification performance compared to individual views (feature representations). We also observed that selecting a subset of the views instead of composing a latent space with all the views can be advantageous in terms of accuracy and computational effort. For further details, including source codes, figures, and datasets, please refer to the project's repository: this https URL. 

---
# Can Reasoning LLMs Enhance Clinical Document Classification? 

**Authors**: Akram Mustafa, Usman Naseem, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08040)  

**Abstract**: Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications. 

---
# From Speech to Summary: A Comprehensive Survey of Speech Summarization 

**Authors**: Fabian Retkowski, Maike Züfle, Andreas Sudmann, Dinah Pfau, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2504.08024)  

**Abstract**: Speech summarization has become an essential tool for efficiently managing and accessing the growing volume of spoken and audiovisual content. However, despite its increasing importance, speech summarization is still not clearly defined and intersects with several research areas, including speech recognition, text summarization, and specific applications like meeting summarization. This survey not only examines existing datasets and evaluation methodologies, which are crucial for assessing the effectiveness of summarization approaches but also synthesizes recent developments in the field, highlighting the shift from traditional systems to advanced models like fine-tuned cascaded architectures and end-to-end solutions. 

---
# More diverse more adaptive: Comprehensive Multi-task Learning for Improved LLM Domain Adaptation in E-commerce 

**Authors**: Tong Piao, Pei Tang, Zhipeng Zhang, Jiaqi Li, Qiao Liu, Zufeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08002)  

**Abstract**: In recent years, Large Language Models (LLMs) have been widely applied across various domains due to their powerful domain adaptation capabilities. Previous studies have suggested that diverse, multi-modal data can enhance LLMs' domain adaptation performance. However, this hypothesis remains insufficiently validated in the e-commerce sector. To address this gap, we propose a comprehensive e-commerce multi-task framework and design empirical experiments to examine the impact of diverse data and tasks on LLMs from two perspectives: "capability comprehensiveness" and "task comprehensiveness." Specifically, we observe significant improvements in LLM performance by progressively introducing tasks related to new major capability areas and by continuously adding subtasks within different major capability domains. Furthermore, we observe that increasing model capacity amplifies the benefits of diversity, suggesting a synergistic relationship between model capacity and data diversity. Finally, we validate the best-performing model from our empirical experiments in the KDD Cup 2024, achieving a rank 5 in Task 1. This outcome demonstrates the significance of our research for advancing LLMs in the e-commerce domain. 

---
# Linguistic Interpretability of Transformer-based Language Models: a systematic review 

**Authors**: Miguel López-Otal, Jorge Gracia, Jordi Bernad, Carlos Bobed, Lucía Pitarch-Ballesteros, Emma Anglés-Herrero  

**Link**: [PDF](https://arxiv.org/pdf/2504.08001)  

**Abstract**: Language models based on the Transformer architecture achieve excellent results in many language-related tasks, such as text classification or sentiment analysis. However, despite the architecture of these models being well-defined, little is known about how their internal computations help them achieve their results. This renders these models, as of today, a type of 'black box' systems. There is, however, a line of research -- 'interpretability' -- aiming to learn how information is encoded inside these models. More specifically, there is work dedicated to studying whether Transformer-based models possess knowledge of linguistic phenomena similar to human speakers -- an area we call 'linguistic interpretability' of these models. In this survey we present a comprehensive analysis of 160 research works, spread across multiple languages and models -- including multilingual ones -- that attempt to discover linguistic information from the perspective of several traditional Linguistics disciplines: Syntax, Morphology, Lexico-Semantics and Discourse. Our survey fills a gap in the existing interpretability literature, which either not focus on linguistic knowledge in these models or present some limitations -- e.g. only studying English-based models. Our survey also focuses on Pre-trained Language Models not further specialized for a downstream task, with an emphasis on works that use interpretability techniques that explore models' internal representations. 

---
# BiasCause: Evaluate Socially Biased Causal Reasoning of Large Language Models 

**Authors**: Tian Xie, Tongxin Yin, Vaishakh Keshava, Xueru Zhang, Siddhartha Reddy Jonnalagadda  

**Link**: [PDF](https://arxiv.org/pdf/2504.07997)  

**Abstract**: While large language models (LLMs) already play significant roles in society, research has shown that LLMs still generate content including social bias against certain sensitive groups. While existing benchmarks have effectively identified social biases in LLMs, a critical gap remains in our understanding of the underlying reasoning that leads to these biased outputs. This paper goes one step further to evaluate the causal reasoning process of LLMs when they answer questions eliciting social biases. We first propose a novel conceptual framework to classify the causal reasoning produced by LLMs. Next, we use LLMs to synthesize $1788$ questions covering $8$ sensitive attributes and manually validate them. The questions can test different kinds of causal reasoning by letting LLMs disclose their reasoning process with causal graphs. We then test 4 state-of-the-art LLMs. All models answer the majority of questions with biased causal reasoning, resulting in a total of $4135$ biased causal graphs. Meanwhile, we discover $3$ strategies for LLMs to avoid biased causal reasoning by analyzing the "bias-free" cases. Finally, we reveal that LLMs are also prone to "mistaken-biased" causal reasoning, where they first confuse correlation with causality to infer specific sensitive group names and then incorporate biased causal reasoning. 

---
# SafeChat: A Framework for Building Trustworthy Collaborative Assistants and a Case Study of its Usefulness 

**Authors**: Biplav Srivastava, Kausik Lakkaraju, Nitin Gupta, Vansh Nagpal, Bharath C. Muppasani, Sara E. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2504.07995)  

**Abstract**: Collaborative assistants, or chatbots, are data-driven decision support systems that enable natural interaction for task completion. While they can meet critical needs in modern society, concerns about their reliability and trustworthiness persist. In particular, Large Language Model (LLM)-based chatbots like ChatGPT, Gemini, and DeepSeek are becoming more accessible. However, such chatbots have limitations, including their inability to explain response generation, the risk of generating problematic content, the lack of standardized testing for reliability, and the need for deep AI expertise and extended development times. These issues make chatbots unsuitable for trust-sensitive applications like elections or healthcare. To address these concerns, we introduce SafeChat, a general architecture for building safe and trustworthy chatbots, with a focus on information retrieval use cases. Key features of SafeChat include: (a) safety, with a domain-agnostic design where responses are grounded and traceable to approved sources (provenance), and 'do-not-respond' strategies to prevent harmful answers; (b) usability, with automatic extractive summarization of long responses, traceable to their sources, and automated trust assessments to communicate expected chatbot behavior, such as sentiment; and (c) fast, scalable development, including a CSV-driven workflow, automated testing, and integration with various devices. We implemented SafeChat in an executable framework using the open-source chatbot platform Rasa. A case study demonstrates its application in building ElectionBot-SC, a chatbot designed to safely disseminate official election information. SafeChat is being used in many domains, validating its potential, and is available at: this https URL. 

---
# Evaluating the Fitness of Ontologies for the Task of Question Generation 

**Authors**: Samah Alkhuzaey, Floriana Grasso, Terry R. Payne, Valentina Tamma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07994)  

**Abstract**: Ontology-based question generation is an important application of semantic-aware systems that enables the creation of large question banks for diverse learning environments. The effectiveness of these systems, both in terms of the calibre and cognitive difficulty of the resulting questions, depends heavily on the quality and modelling approach of the underlying ontologies, making it crucial to assess their fitness for this task. To date, there has been no comprehensive investigation into the specific ontology aspects or characteristics that affect the question generation process. Therefore, this paper proposes a set of requirements and task-specific metrics for evaluating the fitness of ontologies for question generation tasks in pedagogical settings. Using the ROMEO methodology, a structured framework for deriving task-specific metrics, an expert-based approach is employed to assess the performance of various ontologies in Automatic Question Generation (AQG) tasks, which is then evaluated over a set of ontologies. Our results demonstrate that ontology characteristics significantly impact the effectiveness of question generation, with different ontologies exhibiting varying performance levels. This highlights the importance of assessing ontology quality with respect to AQG tasks. 

---
# 'Neural howlround' in large language models: a self-reinforcing bias phenomenon, and a dynamic attenuation solution 

**Authors**: Seth Drake  

**Link**: [PDF](https://arxiv.org/pdf/2504.07992)  

**Abstract**: Large language model (LLM)-driven AI systems may exhibit an inference failure mode we term `neural howlround,' a self-reinforcing cognitive loop where certain highly weighted inputs become dominant, leading to entrenched response patterns resistant to correction. This paper explores the mechanisms underlying this phenomenon, which is distinct from model collapse and biased salience weighting. We propose an attenuation-based correction mechanism that dynamically introduces counterbalancing adjustments and can restore adaptive reasoning, even in `locked-in' AI systems. Additionally, we discuss some other related effects arising from improperly managed reinforcement. Finally, we outline potential applications of this mitigation strategy for improving AI robustness in real-world decision-making tasks. 

---
# Regional Tiny Stories: Using Small Models to Compare Language Learning and Tokenizer Performance 

**Authors**: Nirvan Patil, Malhar Abhay Inamdar, Agnivo Gosai, Guruprasad Pathak, Anish Joshi, Aryan Sagavekar, Anish Joshirao, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2504.07989)  

**Abstract**: Small Language Models (SLMs) offer efficient alternatives to LLMs for specific domains. The 2023 TinyStories study developed an English dataset that allows SLMs with 1 to 10 million parameters to produce coherent outputs. Our research expands this framework by translating the original dataset into Indian languages and creating synthetic data using LLMs. We focus on Hindi, Marathi, and Bengali, evaluating SLMs for regional language processing and understanding linguistic complexity. We show that SLMs efficiently process regional languages with significantly fewer parameters than LLMs, providing a complementary framework for ``inference based evaluation" of tokenization strategies and linguistic complexity. Our analysis shows that language-specific tokenizers outperform general-purpose ones for Indian languages. Empirical validations, supported by information-theoretic and morphological analyses, provides fundamental understanding behind the better performance of Hindi models over Marathi and Bengali. Additionally, we show that synthetic datasets outperform translated content for training SLMs. Correlation analyses reveal cross-linguistic patterns and language-specific relationships between creativity, grammatical precision, and narrative completeness. These findings advance both the practical application of SLMs to underserved languages and our theoretical understanding of neural language development. 

---
# SEAL: Steerable Reasoning Calibration of Large Language Models for Free 

**Authors**: Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07986)  

**Abstract**: Large Language Models (LLMs), such as OpenAI's o1-series have demonstrated compelling capabilities for complex reasoning tasks via the extended chain-of-thought (CoT) reasoning mechanism. However, recent studies reveal substantial redundancy in the CoT reasoning traces, which not only increases inference latency but also negatively impacts model performance by diverting attention to unnecessary reasoning paths. To address this issue, we investigate the internal reasoning structures of LLMs and categorize them into three primary thought types: execution, reflection, and transition thoughts. Moreover, our analysis reveals that excessive reflection and transition thoughts are strongly correlated with failure cases and these thought categories exhibit clear separation in the latent space. Based on these, we introduce SEAL (Steerable reasoning calibration), a training-free approach that seamlessly calibrates the CoT process, improving accuracy while demonstrating significant efficiency gains. SEAL consists of an offline stage for extracting the reasoning steering vector in the latent space, followed by an on-the-fly calibration of the reasoning trace through representation intervention using the steering vector. Notably, the steering vector exhibits strong transferability across various tasks. Extensive experiments across multiple models (DeepSeek-R1-Distill and QwQ-32B-Preview) and benchmarks (Math500, GSM8K, LiveCodeBench) validate the effectiveness of SEAL, up to a 11% improvement in accuracy while reducing reasoning tokens by 11.8% to 50.4%. Our code is publicly available at this https URL. 

---
# Topic mining based on fine-tuning Sentence-BERT and LDA 

**Authors**: Jianheng Li, Lirong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07984)  

**Abstract**: Research background: With the continuous development of society, consumers pay more attention to the key information of product fine-grained attributes when shopping. Research purposes: This study will fine tune the Sentence-BERT word embedding model and LDA model, mine the subject characteristics in online reviews of goods, and show consumers the details of various aspects of goods. Research methods: First, the Sentence-BERT model was fine tuned in the field of e-commerce online reviews, and the online review text was converted into a word vector set with richer semantic information; Secondly, the vectorized word set is input into the LDA model for topic feature extraction; Finally, focus on the key functions of the product through keyword analysis under the theme. Results: This study compared this model with other word embedding models and LDA models, and compared it with common topic extraction methods. The theme consistency of this model is 0.5 higher than that of other models, which improves the accuracy of theme extraction 

---
# Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method 

**Authors**: Shurui Wu, Xinyi Huang, Dingxin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07983)  

**Abstract**: As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations. 

---
# Metamorphic Testing for Fairness Evaluation in Large Language Models: Identifying Intersectional Bias in LLaMA and GPT 

**Authors**: Harishwar Reddy, Madhusudan Srinivasan, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2504.07982)  

**Abstract**: Large Language Models (LLMs) have made significant strides in Natural Language Processing but remain vulnerable to fairness-related issues, often reflecting biases inherent in their training data. These biases pose risks, particularly when LLMs are deployed in sensitive areas such as healthcare, finance, and law. This paper introduces a metamorphic testing approach to systematically identify fairness bugs in LLMs. We define and apply a set of fairness-oriented metamorphic relations (MRs) to assess the LLaMA and GPT model, a state-of-the-art LLM, across diverse demographic inputs. Our methodology includes generating source and follow-up test cases for each MR and analyzing model responses for fairness violations. The results demonstrate the effectiveness of MT in exposing bias patterns, especially in relation to tone and sentiment, and highlight specific intersections of sensitive attributes that frequently reveal fairness faults. This research improves fairness testing in LLMs, providing a structured approach to detect and mitigate biases and improve model robustness in fairness-sensitive applications. 

---
# Towards an Understanding of Context Utilization in Code Intelligence 

**Authors**: Yanlin Wang, Kefeng Duan, Dewu Zheng, Ensheng Shi, Fengji Zhang, Yanli Wang, Jiachi Chen, Xilin Liu, Yuchi Ma, Hongyu Zhang, Qianxiang Wang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08734)  

**Abstract**: Code intelligence is an emerging domain in software engineering, aiming to improve the effectiveness and efficiency of various code-related tasks. Recent research suggests that incorporating contextual information beyond the basic original task inputs (i.e., source code) can substantially enhance model performance. Such contextual signals may be obtained directly or indirectly from sources such as API documentation or intermediate representations like abstract syntax trees can significantly improve the effectiveness of code intelligence. Despite growing academic interest, there is a lack of systematic analysis of context in code intelligence. To address this gap, we conduct an extensive literature review of 146 relevant studies published between September 2007 and August 2024. Our investigation yields four main contributions. (1) A quantitative analysis of the research landscape, including publication trends, venues, and the explored domains; (2) A novel taxonomy of context types used in code intelligence; (3) A task-oriented analysis investigating context integration strategies across diverse code intelligence tasks; (4) A critical evaluation of evaluation methodologies for context-aware methods. Based on these findings, we identify fundamental challenges in context utilization in current code intelligence systems and propose a research roadmap that outlines key opportunities for future research. 

---
# DocAgent: A Multi-Agent System for Automated Code Documentation Generation 

**Authors**: Dayu Yang, Antoine Simoulin, Xin Qian, Xiaoyi Liu, Yuwei Cao, Zhaopu Teng, Grey Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08725)  

**Abstract**: High-quality code documentation is crucial for software development especially in the era of AI. However, generating it automatically using Large Language Models (LLMs) remains challenging, as existing approaches often produce incomplete, unhelpful, or factually incorrect outputs. We introduce DocAgent, a novel multi-agent collaborative system using topological code processing for incremental context building. Specialized agents (Reader, Searcher, Writer, Verifier, Orchestrator) then collaboratively generate documentation. We also propose a multi-faceted evaluation framework assessing Completeness, Helpfulness, and Truthfulness. Comprehensive experiments show DocAgent significantly outperforms baselines consistently. Our ablation study confirms the vital role of the topological processing order. DocAgent offers a robust approach for reliable code documentation generation in complex and proprietary repositories. 

---
# Generating Fine Details of Entity Interactions 

**Authors**: Xinyi Gu, Jiayuan Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08714)  

**Abstract**: Images not only depict objects but also encapsulate rich interactions between them. However, generating faithful and high-fidelity images involving multiple entities interacting with each other, is a long-standing challenge. While pre-trained text-to-image models are trained on large-scale datasets to follow diverse text instructions, they struggle to generate accurate interactions, likely due to the scarcity of training data for uncommon object interactions. This paper introduces InterActing, an interaction-focused dataset with 1000 fine-grained prompts covering three key scenarios: (1) functional and action-based interactions, (2) compositional spatial relationships, and (3) multi-subject interactions. To address interaction generation challenges, we propose a decomposition-augmented refinement procedure. Our approach, DetailScribe, built on Stable Diffusion 3.5, leverages LLMs to decompose interactions into finer-grained concepts, uses a VLM to critique generated images, and applies targeted interventions within the diffusion process in refinement. Automatic and human evaluations show significantly improved image quality, demonstrating the potential of enhanced inference strategies. Our dataset and code are available at this https URL to facilitate future exploration of interaction-rich image generation. 

---
# Training-free Guidance in Text-to-Video Generation via Multimodal Planning and Structured Noise Initialization 

**Authors**: Jialu Li, Shoubin Yu, Han Lin, Jaemin Cho, Jaehong Yoon, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.08641)  

**Abstract**: Recent advancements in text-to-video (T2V) diffusion models have significantly enhanced the visual quality of the generated videos. However, even recent T2V models find it challenging to follow text descriptions accurately, especially when the prompt requires accurate control of spatial layouts or object trajectories. A recent line of research uses layout guidance for T2V models that require fine-tuning or iterative manipulation of the attention map during inference time. This significantly increases the memory requirement, making it difficult to adopt a large T2V model as a backbone. To address this, we introduce Video-MSG, a training-free Guidance method for T2V generation based on Multimodal planning and Structured noise initialization. Video-MSG consists of three steps, where in the first two steps, Video-MSG creates Video Sketch, a fine-grained spatio-temporal plan for the final video, specifying background, foreground, and object trajectories, in the form of draft video frames. In the last step, Video-MSG guides a downstream T2V diffusion model with Video Sketch through noise inversion and denoising. Notably, Video-MSG does not need fine-tuning or attention manipulation with additional memory during inference time, making it easier to adopt large T2V models. Video-MSG demonstrates its effectiveness in enhancing text alignment with multiple T2V backbones (VideoCrafter2 and CogVideoX-5B) on popular T2V generation benchmarks (T2VCompBench and VBench). We provide comprehensive ablation studies about noise inversion ratio, different background generators, background object detection, and foreground object segmentation. 

---
# Analyzing 16,193 LLM Papers for Fun and Profits 

**Authors**: Zhiqiu Xia, Lang Zhu, Bingzhe Li, Feng Chen, Qiannan Li, Hang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08619)  

**Abstract**: Large Language Models (LLMs) are reshaping the landscape of computer science research, driving significant shifts in research priorities across diverse conferences and fields. This study provides a comprehensive analysis of the publication trend of LLM-related papers in 77 top-tier computer science conferences over the past six years (2019-2024). We approach this analysis from four distinct perspectives: (1) We investigate how LLM research is driving topic shifts within major conferences. (2) We adopt a topic modeling approach to identify various areas of LLM-related topic growth and reveal the topics of concern at different conferences. (3) We explore distinct contribution patterns of academic and industrial institutions. (4) We study the influence of national origins on LLM development trajectories. Synthesizing the findings from these diverse analytical angles, we derive ten key insights that illuminate the dynamics and evolution of the LLM research ecosystem. 

---
# Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks 

**Authors**: Ye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08525)  

**Abstract**: Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. The full implementation of TME is available at this https URL. 

---
# BOISHOMMO: Holistic Approach for Bangla Hate Speech 

**Authors**: Md Abdullah Al Kafi, Sumit Kumar Banshal, Md Sadman Shakib, Showrov Azam, Tamanna Alam Tabashom  

**Link**: [PDF](https://arxiv.org/pdf/2504.08408)  

**Abstract**: One of the most alarming issues in digital society is hate speech (HS) on social media. The severity is so high that researchers across the globe are captivated by this domain. A notable amount of work has been conducted to address the identification and alarm system. However, a noticeable gap exists, especially for low-resource languages. Comprehensive datasets are the main problem among the constrained resource languages, such as Bangla. Interestingly, hate speech or any particular speech has no single dimensionality. Similarly, the hate component can simultaneously have multiple abusive attributes, which seems to be missed in the existing datasets. Thus, a multi-label Bangla hate speech dataset named BOISHOMMO has been compiled and evaluated in this work. That includes categories of HS across race, gender, religion, politics, and more. With over two thousand annotated examples, BOISHOMMO provides a nuanced understanding of hate speech in Bangla and highlights the complexities of processing non-Latin scripts. Apart from evaluating with multiple algorithmic approaches, it also highlights the complexities of processing Bangla text and assesses model performance. This unique multi-label approach enriches future hate speech detection and analysis studies for low-resource languages by providing a more nuanced, diverse dataset. 

---
# FocalLens: Instruction Tuning Enables Zero-Shot Conditional Image Representations 

**Authors**: Cheng-Yu Hsieh, Pavan Kumar Anasosalu Vasu, Fartash Faghri, Raviteja Vemulapalli, Chun-Liang Li, Ranjay Krishna, Oncel Tuzel, Hadi Pouransari  

**Link**: [PDF](https://arxiv.org/pdf/2504.08368)  

**Abstract**: Visual understanding is inherently contextual -- what we focus on in an image depends on the task at hand. For instance, given an image of a person holding a bouquet of flowers, we may focus on either the person such as their clothing, or the type of flowers, depending on the context of interest. Yet, most existing image encoding paradigms represent an image as a fixed, generic feature vector, overlooking the potential needs of prioritizing varying visual information for different downstream use cases. In this work, we introduce FocalLens, a conditional visual encoding method that produces different representations for the same image based on the context of interest, expressed flexibly through natural language. We leverage vision instruction tuning data and contrastively finetune a pretrained vision encoder to take natural language instructions as additional inputs for producing conditional image representations. Extensive experiments validate that conditional image representation from FocalLens better pronounce the visual features of interest compared to generic features produced by standard vision encoders like CLIP. In addition, we show FocalLens further leads to performance improvements on a range of downstream tasks including image-image retrieval, image classification, and image-text retrieval, with an average gain of 5 and 10 points on the challenging SugarCrepe and MMVP-VLM benchmarks, respectively. 

---
# MedRep: Medical Concept Representation for General Electronic Health Record Foundation Models 

**Authors**: Junmo Kim, Namkyeong Lee, Jiwon Kim, Kwangsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08329)  

**Abstract**: Electronic health record (EHR) foundation models have been an area ripe for exploration with their improved performance in various medical tasks. Despite the rapid advances, there exists a fundamental limitation: Processing unseen medical codes out of the vocabulary. This problem limits the generality of EHR foundation models and the integration of models trained with different vocabularies. To deal with this problem, we propose MedRep for EHR foundation models based on the observational medical outcome partnership (OMOP) common data model (CDM), providing the integrated medical concept representations and the basic data augmentation strategy for patient trajectories. For concept representation learning, we enrich the information of each concept with a minimal definition through large language model (LLM) prompts and enhance the text-based representations through graph ontology of OMOP vocabulary. Trajectory augmentation randomly replaces selected concepts with other similar concepts that have closely related representations to let the model practice with the concepts out-of-vocabulary. Finally, we demonstrate that EHR foundation models trained with MedRep better maintain the prediction performance in external datasets. Our code implementation is publicly available at this https URL. 

---
# Generalized Multilingual Text-to-Speech Generation with Language-Aware Style Adaptation 

**Authors**: Haowei Lou, Hye-young Paik, Sheng Li, Wen Hu, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08274)  

**Abstract**: Text-to-Speech (TTS) models can generate natural, human-like speech across multiple languages by transforming phonemes into waveforms. However, multilingual TTS remains challenging due to discrepancies in phoneme vocabularies and variations in prosody and speaking style across languages. Existing approaches either train separate models for each language, which achieve high performance at the cost of increased computational resources, or use a unified model for multiple languages that struggles to capture fine-grained, language-specific style variations. In this work, we propose LanStyleTTS, a non-autoregressive, language-aware style adaptive TTS framework that standardizes phoneme representations and enables fine-grained, phoneme-level style control across languages. This design supports a unified multilingual TTS model capable of producing accurate and high-quality speech without the need to train language-specific models. We evaluate LanStyleTTS by integrating it with several state-of-the-art non-autoregressive TTS architectures. Results show consistent performance improvements across different model backbones. Furthermore, we investigate a range of acoustic feature representations, including mel-spectrograms and autoencoder-derived latent features. Our experiments demonstrate that latent encodings can significantly reduce model size and computational cost while preserving high-quality speech generation. 

---
# VLMT: Vision-Language Multimodal Transformer for Multimodal Multi-hop Question Answering 

**Authors**: Qi Zhi Lim, Chin Poo Lee, Kian Ming Lim, Kalaiarasi Sonai Muthu Anbananthen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08269)  

**Abstract**: The increasing availability of multimodal data across text, tables, and images presents new challenges for developing models capable of complex cross-modal reasoning. Existing methods for Multimodal Multi-hop Question Answering (MMQA) often suffer from limited reasoning capabilities, reliance on modality conversion, and inadequate alignment between visual and textual representations. To address these limitations, this paper introduces Vision-Language Multimodal Transformer (VLMT), a unified architecture that integrates a transformer-based vision encoder with a sequence-to-sequence language model. VLMT employs a direct token-level injection mechanism to fuse visual and textual inputs within a shared embedding space, eliminating the need for intermediate projection layers. To enhance cross-modal alignment and reasoning, a three-stage pretraining strategy is proposed to progressively align vision-language representations and improve the model's capacity for multimodal understanding. Based on the pretrained backbone, two task-specific modules are instantiated to form a two-stage MMQA framework: a multimodal reranker that predicts document relevance scores and utilizes a relative threshold with top-k strategy for context retrieval, and a multimodal question answering model that generates contextually grounded answers based on the retrieved evidence. Comprehensive experiments on two benchmark datasets demonstrate the effectiveness of the proposed approach. On MultimodalQA validation set, VLMT-Large achieves 76.5% Exact Match and 80.1% F1, outperforming the previous state-of-the-art by +9.1% in Exact Match and +8.8% in F1. On WebQA, it attains a QA score of 47.6, surpassing prior models such as PERQA by +3.2. These results highlight VLMT's strong capabilities in multimodal reasoning and its potential to advance real-world information retrieval and question answering systems. 

---
# Millions of States: Designing a Scalable MoE Architecture with RWKV-7 Meta-learner 

**Authors**: Liu Xiao, Li Zhiyuan, Lin Yueyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08247)  

**Abstract**: State-based sequence models like RWKV-7 offer a compelling alternative to Transformer architectures, achieving linear complexity while demonstrating greater expressive power in short-context scenarios and enabling state tracking beyond the \(\text{TC}^0\) complexity class. However, RWKV-7 lacks mechanisms for token-parameter interactions and native scalability, limiting its adaptability and growth without retraining. In this paper, we propose \textbf{Meta-State}, a novel extension to RWKV-7 that replaces attention mechanisms with a fully state-driven approach, integrating token-parameter interactions through a \textbf{Self-State Encoder} (SSE) mechanism. The SSE repurposes a portion of the RWKV-7 Weighted Key-Value (WKV) state as transformation weights to encode token-parameter interactions in a linear, state-driven manner without introducing new trainable matrices or softmax operations, while preserving the autoregressive property of token processing. Meta-State supports progressive model scaling by expanding the WKV state and parameter tokens, reusing existing parameters without retraining. Our approach bridges the gap between state-based modeling, token-parameter interactions, and scalable architectures, offering a flexible framework for efficient and adaptable sequence modeling with linear complexity and constant memory usage. 

---
# SAEs $\textit{Can}$ Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs 

**Authors**: Aashiq Muhamed, Jacopo Bonato, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.08192)  

**Abstract**: Machine unlearning is a promising approach to improve LLM safety by removing unwanted knowledge from the model. However, prevailing gradient-based unlearning methods suffer from issues such as high computational costs, hyperparameter instability, poor sequential unlearning capability, vulnerability to relearning attacks, low data efficiency, and lack of interpretability. While Sparse Autoencoders are well-suited to improve these aspects by enabling targeted activation-based unlearning, prior approaches underperform gradient-based methods. This work demonstrates that, contrary to these earlier findings, SAEs can significantly improve unlearning when employed dynamically. We introduce $\textbf{Dynamic DAE Guardrails}$ (DSG), a novel method for precision unlearning that leverages principled feature selection and a dynamic classifier. Our experiments show DSG substantially outperforms leading unlearning methods, achieving superior forget-utility trade-offs. DSG addresses key drawbacks of gradient-based approaches for unlearning -- offering enhanced computational efficiency and stability, robust performance in sequential unlearning, stronger resistance to relearning attacks, better data efficiency including zero-shot settings, and more interpretable unlearning. 

---
# Geneshift: Impact of different scenario shift on Jailbreaking LLM 

**Authors**: Tianyi Wu, Zhiwei Xue, Yue Liu, Jiaheng Zhang, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08104)  

**Abstract**: Jailbreak attacks, which aim to cause LLMs to perform unrestricted behaviors, have become a critical and challenging direction in AI safety. Despite achieving the promising attack success rate using dictionary-based evaluation, existing jailbreak attack methods fail to output detailed contents to satisfy the harmful request, leading to poor performance on GPT-based evaluation. To this end, we propose a black-box jailbreak attack termed GeneShift, by using a genetic algorithm to optimize the scenario shifts. Firstly, we observe that the malicious queries perform optimally under different scenario shifts. Based on it, we develop a genetic algorithm to evolve and select the hybrid of scenario shifts. It guides our method to elicit detailed and actionable harmful responses while keeping the seemingly benign facade, improving stealthiness. Extensive experiments demonstrate the superiority of GeneShift. Notably, GeneShift increases the jailbreak success rate from 0% to 60% when direct prompting alone would fail. 

---
# The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search 

**Authors**: Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08066)  

**Abstract**: AI is increasingly playing a pivotal role in transforming how scientific discoveries are made. We introduce The AI Scientist-v2, an end-to-end agentic system capable of producing the first entirely AI generated peer-review-accepted workshop paper. This system iteratively formulates scientific hypotheses, designs and executes experiments, analyzes and visualizes data, and autonomously authors scientific manuscripts. Compared to its predecessor (v1, Lu et al., 2024 arXiv:2408.06292), The AI Scientist-v2 eliminates the reliance on human-authored code templates, generalizes effectively across diverse machine learning domains, and leverages a novel progressive agentic tree-search methodology managed by a dedicated experiment manager agent. Additionally, we enhance the AI reviewer component by integrating a Vision-Language Model (VLM) feedback loop for iterative refinement of content and aesthetics of the figures. We evaluated The AI Scientist-v2 by submitting three fully autonomous manuscripts to a peer-reviewed ICLR workshop. Notably, one manuscript achieved high enough scores to exceed the average human acceptance threshold, marking the first instance of a fully AI-generated paper successfully navigating a peer review. This accomplishment highlights the growing capability of AI in conducting all aspects of scientific research. We anticipate that further advancements in autonomous scientific discovery technologies will profoundly impact human knowledge generation, enabling unprecedented scalability in research productivity and significantly accelerating scientific breakthroughs, greatly benefiting society at large. We have open-sourced the code at this https URL to foster the future development of this transformative technology. We also discuss the role of AI in science, including AI safety. 

---
# Large-Scale Analysis of Online Questions Related to Opioid Use Disorder on Reddit 

**Authors**: Tanmay Laud, Akadia Kacha-Ochana, Steven A. Sumner, Vikram Krishnasamy, Royal Law, Lyna Schieber, Munmun De Choudhury, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2504.08044)  

**Abstract**: Opioid use disorder (OUD) is a leading health problem that affects individual well-being as well as general public health. Due to a variety of reasons, including the stigma faced by people using opioids, online communities for recovery and support were formed on different social media platforms. In these communities, people share their experiences and solicit information by asking questions to learn about opioid use and recovery. However, these communities do not always contain clinically verified information. In this paper, we study natural language questions asked in the context of OUD-related discourse on Reddit. We adopt transformer-based question detection along with hierarchical clustering across 19 subreddits to identify six coarse-grained categories and 69 fine-grained categories of OUD-related questions. Our analysis uncovers ten areas of information seeking from Reddit users in the context of OUD: drug sales, specific drug-related questions, OUD treatment, drug uses, side effects, withdrawal, lifestyle, drug testing, pain management and others, during the study period of 2018-2021. Our work provides a major step in improving the understanding of OUD-related questions people ask unobtrusively on Reddit. We finally discuss technological interventions and public health harm reduction techniques based on the topics of these questions. 

---
