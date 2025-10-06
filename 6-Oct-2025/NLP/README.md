# Reward Models are Metrics in a Trench Coat 

**Authors**: Sebastian Gehrmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.03231)  

**Abstract**: The emergence of reinforcement learning in post-training of large language models has sparked significant interest in reward models. Reward models assess the quality of sampled model outputs to generate training signals. This task is also performed by evaluation metrics that monitor the performance of an AI model. We find that the two research areas are mostly separate, leading to redundant terminology and repeated pitfalls. Common challenges include susceptibility to spurious correlations, impact on downstream reward hacking, methods to improve data quality, and approaches to meta-evaluation. Our position paper argues that a closer collaboration between the fields can help overcome these issues. To that end, we show how metrics outperform reward models on specific tasks and provide an extensive survey of the two areas. Grounded in this survey, we point to multiple research topics in which closer alignment can improve reward models and metrics in areas such as preference elicitation methods, avoidance of spurious correlations and reward hacking, and calibration-aware meta-evaluation. 

---
# Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignment 

**Authors**: Hongxiang Zhang, Yuan Tian, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03223)  

**Abstract**: To solve complex reasoning tasks for Large Language Models (LLMs), prompting-based methods offer a lightweight alternative to fine-tuning and reinforcement learning. However, as reasoning chains extend, critical intermediate steps and the original prompt will be buried in the context, receiving insufficient attention and leading to errors. In this paper, we propose Self-Anchor, a novel pipeline that leverages the inherent structure of reasoning to steer LLM attention. Self-Anchor decomposes reasoning trajectories into structured plans and automatically aligns the model's attention to the most relevant inference steps, allowing the model to maintain focus throughout generation. Our experiment shows that Self-Anchor outperforms SOTA prompting methods across six benchmarks. Notably, Self-Anchor significantly reduces the performance gap between ``non-reasoning'' models and specialized reasoning models, with the potential to enable most LLMs to tackle complex reasoning tasks without retraining. 

---
# Cache-to-Cache: Direct Semantic Communication Between Large Language Models 

**Authors**: Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03215)  

**Abstract**: Multi-LLM systems harness the complementary strengths of diverse Large Language Models, achieving performance and efficiency gains unattainable by a single model. In existing designs, LLMs communicate through text, forcing internal representations to be transformed into output token sequences. This process both loses rich semantic information and incurs token-by-token generation latency. Motivated by these limitations, we ask: Can LLMs communicate beyond text? Oracle experiments show that enriching the KV-Cache semantics can improve response quality without increasing cache size, supporting KV-Cache as an effective medium for inter-model communication. Thus, we propose Cache-to-Cache (C2C), a new paradigm for direct semantic communication between LLMs. C2C uses a neural network to project and fuse the source model's KV-cache with that of the target model to enable direct semantic transfer. A learnable gating mechanism selects the target layers that benefit from cache communication. Compared with text communication, C2C utilizes the deep, specialized semantics from both models, while avoiding explicit intermediate text generation. Experiments show that C2C achieves 8.5-10.5% higher average accuracy than individual models. It further outperforms the text communication paradigm by approximately 3.0-5.0%, while delivering an average 2.0x speedup in latency. Our code is available at this https URL. 

---
# FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents 

**Authors**: Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han Lù, Léo Boisvert, Massimo Caccia, Jérémy Espinas, Alexandre Aussem, Véronique Eglin, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2510.03204)  

**Abstract**: Web agents powered by large language models (LLMs) must process lengthy web page observations to complete user goals; these pages often exceed tens of thousands of tokens. This saturates context limits and increases computational cost processing; moreover, processing full pages exposes agents to security risks such as prompt injection. Existing pruning strategies either discard relevant content or retain irrelevant context, leading to suboptimal action prediction. We introduce FocusAgent, a simple yet effective approach that leverages a lightweight LLM retriever to extract the most relevant lines from accessibility tree (AxTree) observations, guided by task goals. By pruning noisy and irrelevant content, FocusAgent enables efficient reasoning while reducing vulnerability to injection attacks. Experiments on WorkArena and WebArena benchmarks show that FocusAgent matches the performance of strong baselines, while reducing observation size by over 50%. Furthermore, a variant of FocusAgent significantly reduces the success rate of prompt-injection attacks, including banner and pop-up attacks, while maintaining task success performance in attack-free settings. Our results highlight that targeted LLM-based retrieval is a practical and robust strategy for building web agents that are efficient, effective, and secure. 

---
# Model-Based Ranking of Source Languages for Zero-Shot Cross-Lingual Transfer 

**Authors**: Abteen Ebrahimi, Adam Wiemerslage, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2510.03202)  

**Abstract**: We present NN-Rank, an algorithm for ranking source languages for cross-lingual transfer, which leverages hidden representations from multilingual models and unlabeled target-language data. We experiment with two pretrained multilingual models and two tasks: part-of-speech tagging (POS) and named entity recognition (NER). We consider 51 source languages and evaluate on 56 and 72 target languages for POS and NER, respectively. When using in-domain data, NN-Rank beats state-of-the-art baselines that leverage lexical and linguistic features, with average improvements of up to 35.56 NDCG for POS and 18.14 NDCG for NER. As prior approaches can fall back to language-level features if target language data is not available, we show that NN-Rank remains competitive using only the Bible, an out-of-domain corpus available for a large number of languages. Ablations on the amount of unlabeled target data show that, for subsets consisting of as few as 25 examples, NN-Rank produces high-quality rankings which achieve 92.8% of the NDCG achieved using all available target data for ranking. 

---
# Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting? 

**Authors**: Xuan Xu, Haolun Li, Zhongliang Yang, Beilin Chu, Jia Song, Moxuan Xu, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03174)  

**Abstract**: Traditional topic models such as neural topic models rely on inference and generation networks to learn latent topic distributions. This paper explores a new paradigm for topic modeling in the era of large language models, framing TM as a long-form generation task whose definition is updated in this paradigm. We propose a simple but practical approach to implement LLM-based topic model tasks out of the box (sample a data subset, generate topics and representative text with our prompt, text assignment with keyword match). We then investigate whether the long-form generation paradigm can beat NTMs via zero-shot prompting. We conduct a systematic comparison between NTMs and LLMs in terms of topic quality and empirically examine the claim that "a majority of NTMs are outdated." 

---
# Neural Correlates of Language Models Are Specific to Human Language 

**Authors**: Iñigo Parra  

**Link**: [PDF](https://arxiv.org/pdf/2510.03156)  

**Abstract**: Previous work has shown correlations between the hidden states of large language models and fMRI brain responses, on language tasks. These correlations have been taken as evidence of the representational similarity of these models and brain states. This study tests whether these previous results are robust to several possible concerns. Specifically this study shows: (i) that the previous results are still found after dimensionality reduction, and thus are not attributable to the curse of dimensionality; (ii) that previous results are confirmed when using new measures of similarity; (iii) that correlations between brain representations and those from models are specific to models trained on human language; and (iv) that the results are dependent on the presence of positional encoding in the models. These results confirm and strengthen the results of previous research and contribute to the debate on the biological plausibility and interpretability of state-of-the-art large language models. 

---
# EditLens: Quantifying the Extent of AI Editing in Text 

**Authors**: Katherine Thai, Bradley Emi, Elyas Masrour, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2510.03154)  

**Abstract**: A significant proportion of queries to large language models ask them to edit user-provided text, rather than generate new text from scratch. While previous work focuses on detecting fully AI-generated text, we demonstrate that AI-edited text is distinguishable from human-written and AI-generated text. First, we propose using lightweight similarity metrics to quantify the magnitude of AI editing present in a text given the original human-written text and validate these metrics with human annotators. Using these similarity metrics as intermediate supervision, we then train EditLens, a regression model that predicts the amount of AI editing present within a text. Our model achieves state-of-the-art performance on both binary (F1=94.7%) and ternary (F1=90.4%) classification tasks in distinguishing human, AI, and mixed writing. Not only do we show that AI-edited text can be detected, but also that the degree of change made by AI to human writing can be detected, which has implications for authorship attribution, education, and policy. Finally, as a case study, we use our model to analyze the effects of AI-edits applied by Grammarly, a popular writing assistance tool. To encourage further research, we commit to publicly releasing our models and dataset. 

---
# Beyond the Final Layer: Intermediate Representations for Better Multilingual Calibration in Large Language Models 

**Authors**: Ej Zhou, Caiqi Zhang, Tiancheng Hu, Chengzu Li, Nigel Collier, Ivan Vulić, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03136)  

**Abstract**: Confidence calibration, the alignment of a model's predicted confidence with its actual accuracy, is crucial for the reliable deployment of Large Language Models (LLMs). However, this critical property remains largely under-explored in multilingual contexts. In this work, we conduct the first large-scale, systematic studies of multilingual calibration across six model families and over 100 languages, revealing that non-English languages suffer from systematically worse calibration. To diagnose this, we investigate the model's internal representations and find that the final layer, biased by English-centric training, provides a poor signal for multilingual confidence. In contrast, our layer-wise analysis uncovers a key insight that late-intermediate layers consistently offer a more reliable and better-calibrated signal. Building on this, we introduce a suite of training-free methods, including Language-Aware Confidence Ensemble (LACE), which adaptively selects an optimal ensemble of layers for each specific language. Our study highlights the hidden costs of English-centric alignment and offer a new path toward building more globally equitable and trustworthy LLMs by looking beyond the final layer. 

---
# SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys? 

**Authors**: Zhaojun Sun, Xuzhou Zhu, Xuanhe Zhou, Xin Tong, Shuo Wang, Jie Fu, Guoliang Li, Zhiyuan Liu, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03120)  

**Abstract**: Academic survey writing, which distills vast literature into a coherent and insightful narrative, remains a labor-intensive and intellectually demanding task. While recent approaches, such as general DeepResearch agents and survey-specialized methods, can generate surveys automatically (a.k.a. LLM4Survey), their outputs often fall short of human standards and there lacks a rigorous, reader-aligned benchmark for thoroughly revealing their deficiencies. To fill the gap, we propose a fine-grained, quiz-driven evaluation framework SurveyBench, featuring (1) typical survey topics source from recent 11,343 arXiv papers and corresponding 4,947 high-quality surveys; (2) a multifaceted metric hierarchy that assesses the outline quality (e.g., coverage breadth, logical coherence), content quality (e.g., synthesis granularity, clarity of insights), and non-textual richness; and (3) a dual-mode evaluation protocol that includes content-based and quiz-based answerability tests, explicitly aligned with readers' informational needs. Results show SurveyBench effectively challenges existing LLM4Survey approaches (e.g., on average 21% lower than human in content-based evaluation). 

---
# Listening or Reading? Evaluating Speech Awareness in Chain-of-Thought Speech-to-Text Translation 

**Authors**: Jacobo Romero-Díaz, Gerard I. Gállego, Oriol Pareras, Federico Costa, Javier Hernando, Cristina España-Bonet  

**Link**: [PDF](https://arxiv.org/pdf/2510.03115)  

**Abstract**: Speech-to-Text Translation (S2TT) systems built from Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) modules face two major limitations: error propagation and the inability to exploit prosodic or other acoustic cues. Chain-of-Thought (CoT) prompting has recently been introduced, with the expectation that jointly accessing speech and transcription will overcome these issues. Analyzing CoT through attribution methods, robustness evaluations with corrupted transcripts, and prosody-awareness, we find that it largely mirrors cascaded behavior, relying mainly on transcripts while barely leveraging speech. Simple training interventions, such as adding Direct S2TT data or noisy transcript injection, enhance robustness and increase speech attribution. These findings challenge the assumed advantages of CoT and highlight the need for architectures that explicitly integrate acoustic information into translation. 

---
# Semantic Similarity in Radiology Reports via LLMs and NER 

**Authors**: Beth Pearson, Ahmed Adnan, Zahraa Abdallah  

**Link**: [PDF](https://arxiv.org/pdf/2510.03102)  

**Abstract**: Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: \href{this https URL}{this http URL\_reports} 

---
# Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting? 

**Authors**: Oriol Pareras, Gerard I. Gállego, Federico Costa, Cristina España-Bonet, Javier Hernando  

**Link**: [PDF](https://arxiv.org/pdf/2510.03093)  

**Abstract**: Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created. 

---
# Semantic Differentiation in Speech Emotion Recognition: Insights from Descriptive and Expressive Speech Roles 

**Authors**: Rongchen Guo, Vincent Francoeur, Isar Nejadgholi, Sylvain Gagnon, Miodrag Bolic  

**Link**: [PDF](https://arxiv.org/pdf/2510.03060)  

**Abstract**: Speech Emotion Recognition (SER) is essential for improving human-computer interaction, yet its accuracy remains constrained by the complexity of emotional nuances in speech. In this study, we distinguish between descriptive semantics, which represents the contextual content of speech, and expressive semantics, which reflects the speaker's emotional state. After watching emotionally charged movie segments, we recorded audio clips of participants describing their experiences, along with the intended emotion tags for each clip, participants' self-rated emotional responses, and their valence/arousal scores. Through experiments, we show that descriptive semantics align with intended emotions, while expressive semantics correlate with evoked emotions. Our findings inform SER applications in human-AI interaction and pave the way for more context-aware AI systems. 

---
# Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines 

**Authors**: Matthew Lewis, Samuel Thio, Richard JB Dobson, Spiros Denaxas  

**Link**: [PDF](https://arxiv.org/pdf/2510.02967)  

**Abstract**: This paper presents the development and evaluation of a Retrieval-Augmented Generation (RAG) system for querying the United Kingdom's National Institute for Health and Care Excellence (NICE) clinical guidelines using Large Language Models (LLMs). The extensive length and volume of these guidelines can impede their utilisation within a time-constrained healthcare system, a challenge this project addresses through the creation of a system capable of providing users with precisely matched information in response to natural language queries. The system's retrieval architecture, composed of a hybrid embedding mechanism, was evaluated against a database of 10,195 text chunks derived from three hundred guidelines. It demonstrates high performance, with a Mean Reciprocal Rank (MRR) of 0.814, a Recall of 81% at the first chunk and of 99.1% within the top ten retrieved chunks, when evaluated on 7901 queries.
The most significant impact of the RAG system was observed during the generation phase. When evaluated on a manually curated dataset of seventy question-answer pairs, RAG-enhanced models showed substantial gains in performance. Faithfulness, the measure of whether an answer is supported by the source text, was increased by 64.7 percentage points to 99.5% for the RAG-enhanced O4-Mini model and significantly outperformed the medical-focused Meditron3-8B LLM, which scored 43%. This, combined with a perfect Context Precision score of 1 for all RAG-enhanced models, confirms the system's ability to prevent information fabrication by grounding its answers in relevant source material. This study thus establishes RAG as an effective, reliable, and scalable approach for applying generative AI in healthcare, enabling cost-effective access to medical guidelines. 

---
# Leave No TRACE: Black-box Detection of Copyrighted Dataset Usage in Large Language Models via Watermarking 

**Authors**: Jingqi Zhang, Ruibo Chen, Yingqing Yang, Peihua Mai, Heng Huang, Yan Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02962)  

**Abstract**: Large Language Models (LLMs) are increasingly fine-tuned on smaller, domain-specific datasets to improve downstream performance. These datasets often contain proprietary or copyrighted material, raising the need for reliable safeguards against unauthorized use. Existing membership inference attacks (MIAs) and dataset-inference methods typically require access to internal signals such as logits, while current black-box approaches often rely on handcrafted prompts or a clean reference dataset for calibration, both of which limit practical applicability. Watermarking is a promising alternative, but prior techniques can degrade text quality or reduce task performance. We propose TRACE, a practical framework for fully black-box detection of copyrighted dataset usage in LLM fine-tuning. \texttt{TRACE} rewrites datasets with distortion-free watermarks guided by a private key, ensuring both text quality and downstream utility. At detection time, we exploit the radioactivity effect of fine-tuning on watermarked data and introduce an entropy-gated procedure that selectively scores high-uncertainty tokens, substantially amplifying detection power. Across diverse datasets and model families, TRACE consistently achieves significant detections (p<0.05), often with extremely strong statistical evidence. Furthermore, it supports multi-dataset attribution and remains robust even after continued pretraining on large non-watermarked corpora. These results establish TRACE as a practical route to reliable black-box verification of copyrighted dataset usage. We will make our code available at: this https URL. 

---
# Finding Diamonds in Conversation Haystacks: A Benchmark for Conversational Data Retrieval 

**Authors**: Yohan Lee, Yongwoo Song, Sangyeop Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.02938)  

**Abstract**: We present the Conversational Data Retrieval (CDR) benchmark, the first comprehensive test set for evaluating systems that retrieve conversation data for product insights. With 1.6k queries across five analytical tasks and 9.1k conversations, our benchmark provides a reliable standard for measuring conversational data retrieval performance. Our evaluation of 16 popular embedding models shows that even the best models reach only around NDCG@10 of 0.51, revealing a substantial gap between document and conversational data retrieval capabilities. Our work identifies unique challenges in conversational data retrieval (implicit state recognition, turn dynamics, contextual references) while providing practical query templates and detailed error analysis across different task categories. The benchmark dataset and code are available at this https URL. 

---
# Self-Reflective Generation at Test Time 

**Authors**: Jian Mu, Qixin Zhang, Zhiyong Wang, Menglin Yang, Shuang Qiu, Chengwei Qin, Zhongxiang Dai, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02919)  

**Abstract**: Large language models (LLMs) increasingly solve complex reasoning tasks via long chain-of-thought, but their forward-only autoregressive generation process is fragile; early token errors can cascade, which creates a clear need for self-reflection mechanisms. However, existing self-reflection either performs revisions over full drafts or learns self-correction via expensive training, both fundamentally reactive and inefficient. To address this, we propose Self-Reflective Generation at Test Time (SRGen), a lightweight test-time framework that reflects before generating at uncertain points. During token generation, SRGen utilizes dynamic entropy thresholding to identify high-uncertainty tokens. For each identified token, it trains a specific corrective vector, which fully exploits the already generated context for a self-reflective generation to correct the token probability distribution. By retrospectively analyzing the partial output, this self-reflection enables more trustworthy decisions, thereby significantly reducing the probability of errors at highly uncertain points. Evaluated on challenging mathematical reasoning benchmarks and a diverse set of LLMs, SRGen can consistently strengthen model reasoning: improvements in single-pass quality also translate into stronger self-consistency voting. Especially, on AIME2024 with DeepSeek-R1-Distill-Qwen-7B, SRGen yields absolute improvements of +12.0% on Pass@1 and +13.3% on Cons@5. Moreover, our findings position SRGen as a plug-and-play method that integrates reflection into the generation process for reliable LLM reasoning, achieving consistent gains with bounded overhead and broad composability with other training-time (e.g., RLHF) and test-time (e.g., SLOT) techniques. 

---
# Constraint Satisfaction Approaches to Wordle: Novel Heuristics and Cross-Lexicon Validation 

**Authors**: Jahidul Arafat, Fariha Tasmin, Sanjaya Poudel, Kamrujjaman, Eftakhar Ahmed Arnob, Ahsan Habib Tareq  

**Link**: [PDF](https://arxiv.org/pdf/2510.02855)  

**Abstract**: Wordle presents an algorithmically rich testbed for constraint satisfaction problem (CSP) solving. While existing solvers rely on information-theoretic entropy maximization or frequency-based heuristics without formal constraint treatment, we present the first comprehensive CSP formulation of Wordle with novel constraint-aware solving strategies. We introduce CSP-Aware Entropy, computing information gain after constraint propagation rather than on raw candidate sets, and a Probabilistic CSP framework integrating Bayesian word-frequency priors with logical constraints. Through evaluation on 2,315 English words, CSP-Aware Entropy achieves 3.54 average guesses with 99.9% success rate, a statistically significant 1.7% improvement over Forward Checking (t=-4.82, p<0.001, Cohen's d=0.07) with 46% faster runtime (12.9ms versus 23.7ms per guess). Under 10% noise, CSP-aware approaches maintain 5.3 percentage point advantages (29.0% versus 23.7%, p=0.041), while Probabilistic CSP achieves 100% success across all noise levels (0-20%) through constraint recovery mechanisms. Cross-lexicon validation on 500 Spanish words demonstrates 88% success with zero language-specific tuning, validating that core CSP principles transfer across languages despite an 11.2 percentage point gap from linguistic differences (p<0.001, Fisher's exact test). Our open-source implementation with 34 unit tests achieving 91% code coverage provides reproducible infrastructure for CSP research. The combination of formal CSP treatment, constraint-aware heuristics, probabilistic-logical integration, robustness analysis, and cross-lexicon validation establishes new performance benchmarks demonstrating that principled constraint satisfaction techniques outperform classical information-theoretic and learning-based approaches for structured puzzle-solving domains. 

---
# Evaluating Large Language Models for IUCN Red List Species Information 

**Authors**: Shinya Uryu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02830)  

**Abstract**: Large Language Models (LLMs) are rapidly being adopted in conservation to address the biodiversity crisis, yet their reliability for species evaluation is uncertain. This study systematically validates five leading models on 21,955 species across four core IUCN Red List assessment components: taxonomy, conservation status, distribution, and threats. A critical paradox was revealed: models excelled at taxonomic classification (94.9%) but consistently failed at conservation reasoning (27.2% for status assessment). This knowledge-reasoning gap, evident across all models, suggests inherent architectural constraints, not just data limitations. Furthermore, models exhibited systematic biases favoring charismatic vertebrates, potentially amplifying existing conservation inequities. These findings delineate clear boundaries for responsible LLM deployment: they are powerful tools for information retrieval but require human oversight for judgment-based decisions. A hybrid approach is recommended, where LLMs augment expert capacity while human experts retain sole authority over risk assessment and policy. 

---
# StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-Hop Question Answering 

**Authors**: Tengjun Ni, Xin Yuan, Shenghong Li, Kai Wu, Ren Ping Liu, Wei Ni, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02827)  

**Abstract**: Recent progress in retrieval-augmented generation (RAG) has led to more accurate and interpretable multi-hop question answering (QA). Yet, challenges persist in integrating iterative reasoning steps with external knowledge retrieval. To address this, we introduce StepChain GraphRAG, a framework that unites question decomposition with a Breadth-First Search (BFS) Reasoning Flow for enhanced multi-hop QA. Our approach first builds a global index over the corpus; at inference time, only retrieved passages are parsed on-the-fly into a knowledge graph, and the complex query is split into sub-questions. For each sub-question, a BFS-based traversal dynamically expands along relevant edges, assembling explicit evidence chains without overwhelming the language model with superfluous context. Experiments on MuSiQue, 2WikiMultiHopQA, and HotpotQA show that StepChain GraphRAG achieves state-of-the-art Exact Match and F1 scores. StepChain GraphRAG lifts average EM by 2.57% and F1 by 2.13% over the SOTA method, achieving the largest gain on HotpotQA (+4.70% EM, +3.44% F1). StepChain GraphRAG also fosters enhanced explainability by preserving the chain-of-thought across intermediate retrieval steps. We conclude by discussing how future work can mitigate the computational overhead and address potential hallucinations from large language models to refine efficiency and reliability in multi-hop QA. 

---
# A Computational Framework for Interpretable Text-Based Personality Assessment from Social Media 

**Authors**: Matej Gjurković  

**Link**: [PDF](https://arxiv.org/pdf/2510.02811)  

**Abstract**: Personality refers to individual differences in behavior, thinking, and feeling. With the growing availability of digital footprints, especially from social media, automated methods for personality assessment have become increasingly important. Natural language processing (NLP) enables the analysis of unstructured text data to identify personality indicators. However, two main challenges remain central to this thesis: the scarcity of large, personality-labeled datasets and the disconnect between personality psychology and NLP, which restricts model validity and interpretability. To address these challenges, this thesis presents two datasets -- MBTI9k and PANDORA -- collected from Reddit, a platform known for user anonymity and diverse discussions. The PANDORA dataset contains 17 million comments from over 10,000 users and integrates the MBTI and Big Five personality models with demographic information, overcoming limitations in data size, quality, and label coverage. Experiments on these datasets show that demographic variables influence model validity. In response, the SIMPA (Statement-to-Item Matching Personality Assessment) framework was developed - a computational framework for interpretable personality assessment that matches user-generated statements with validated questionnaire items. By using machine learning and semantic similarity, SIMPA delivers personality assessments comparable to human evaluations while maintaining high interpretability and efficiency. Although focused on personality assessment, SIMPA's versatility extends beyond this domain. Its model-agnostic design, layered cue detection, and scalability make it suitable for various research and practical applications involving complex label taxonomies and variable cue associations with target concepts. 

---
# XTRA: Cross-Lingual Topic Modeling with Topic and Representation Alignments 

**Authors**: Tien Phat Nguyen, Vu Minh Ngo, Tung Nguyen, Linh Van Ngo, Duc Anh Nguyen, Sang Dinh, Trung Le  

**Link**: [PDF](https://arxiv.org/pdf/2510.02788)  

**Abstract**: Cross-lingual topic modeling aims to uncover shared semantic themes across languages. Several methods have been proposed to address this problem, leveraging both traditional and neural approaches. While previous methods have achieved some improvements in topic diversity, they often struggle to ensure high topic coherence and consistent alignment across languages. We propose XTRA (Cross-Lingual Topic Modeling with Topic and Representation Alignments), a novel framework that unifies Bag-of-Words modeling with multilingual embeddings. XTRA introduces two core components: (1) representation alignment, aligning document-topic distributions via contrastive learning in a shared semantic space; and (2) topic alignment, projecting topic-word distributions into the same space to enforce crosslingual consistency. This dual mechanism enables XTRA to learn topics that are interpretable (coherent and diverse) and well-aligned across languages. Experiments on multilingual corpora confirm that XTRA significantly outperforms strong baselines in topic coherence, diversity, and alignment quality. Code and reproducible scripts are available at https: //github.com/tienphat140205/XTRA. 

---
# The Path of Self-Evolving Large Language Models: Achieving Data-Efficient Learning via Intrinsic Feedback 

**Authors**: Hangfan Zhang, Siyuan Xu, Zhimeng Guo, Huaisheng Zhu, Shicheng Liu, Xinrun Wang, Qiaosheng Zhang, Yang Chen, Peng Ye, Lei Bai, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02752)  

**Abstract**: Reinforcement learning (RL) has demonstrated potential in enhancing the reasoning capabilities of large language models (LLMs), but such training typically demands substantial efforts in creating and annotating data. In this work, we explore improving LLMs through RL with minimal data. Our approach alternates between the LLM proposing a task and then attempting to solve it. To minimize data dependency, we introduce two novel mechanisms grounded in self-awareness: (1) self-aware difficulty prediction, where the model learns to assess task difficulty relative to its own abilities and prioritize challenging yet solvable tasks, and (2) self-aware limit breaking, where the model recognizes when a task is beyond its capability boundary and proactively requests external data to break through that limit. Extensive experiments on nine benchmarks showing a 53.8% relative improvement with less than 1.2% extra data demonstrate the efficacy of self-aware RL and underscore the promise of self-evolving agent training. 

---
# IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context 

**Authors**: Santhosh G S, Akshay Govind S, Gokul S Krishnan, Balaraman Ravindran, Sriraam Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02742)  

**Abstract**: Large Language Models (LLMs) have gained significant traction across critical domains owing to their impressive contextual understanding and generative capabilities. However, their increasing deployment in high stakes applications necessitates rigorous evaluation of embedded biases, particularly in culturally diverse contexts like India where existing embedding-based bias assessment methods often fall short in capturing nuanced stereotypes. We propose an evaluation framework based on a encoder trained using contrastive learning that captures fine-grained bias through embedding similarity. We also introduce a novel dataset - IndiCASA (IndiBias-based Contextually Aligned Stereotypes and Anti-stereotypes) comprising 2,575 human-validated sentences spanning five demographic axes: caste, gender, religion, disability, and socioeconomic status. Our evaluation of multiple open-weight LLMs reveals that all models exhibit some degree of stereotypical bias, with disability related biases being notably persistent, and religion bias generally lower likely due to global debiasing efforts demonstrating the need for fairer model development. 

---
# PGMEL: Policy Gradient-based Generative Adversarial Network for Multimodal Entity Linking 

**Authors**: KM Pooja, Cheng Long, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.02726)  

**Abstract**: The task of entity linking, which involves associating mentions with their respective entities in a knowledge graph, has received significant attention due to its numerous potential applications. Recently, various multimodal entity linking (MEL) techniques have been proposed, targeted to learn comprehensive embeddings by leveraging both text and vision modalities. The selection of high-quality negative samples can potentially play a crucial role in metric/representation learning. However, to the best of our knowledge, this possibility remains unexplored in existing literature within the framework of MEL. To fill this gap, we address the multimodal entity linking problem in a generative adversarial setting where the generator is responsible for generating high-quality negative samples, and the discriminator is assigned the responsibility for the metric learning tasks. Since the generator is involved in generating samples, which is a discrete process, we optimize it using policy gradient techniques and propose a policy gradient-based generative adversarial network for multimodal entity linking (PGMEL). Experimental results based on Wiki-MEL, Richpedia-MEL and WikiDiverse datasets demonstrate that PGMEL learns meaningful representation by selecting challenging negative samples and outperforms state-of-the-art methods. 

---
# TravelBench : Exploring LLM Performance in Low-Resource Domains 

**Authors**: Srinivas Billa, Xiaonan Jing  

**Link**: [PDF](https://arxiv.org/pdf/2510.02719)  

**Abstract**: Results on existing LLM benchmarks capture little information over the model capabilities in low-resource tasks, making it difficult to develop effective solutions in these domains. To address these challenges, we curated 14 travel-domain datasets spanning 7 common NLP tasks using anonymised data from real-world scenarios, and analysed the performance across LLMs. We report on the accuracy, scaling behaviour, and reasoning capabilities of LLMs in a variety of tasks. Our results confirm that general benchmarking results are insufficient for understanding model performance in low-resource tasks. Despite the amount of training FLOPs, out-of-the-box LLMs hit performance bottlenecks in complex, domain-specific scenarios. Furthermore, reasoning provides a more significant boost for smaller LLMs by making the model a better judge on certain tasks. 

---
# Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks 

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2510.02712)  

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present the first comprehensive survival analysis of conversational AI robustness, analyzing 36,951 conversation turns across 9 state-of-the-art LLMs to model failure as a time-to-event process. Our survival modeling framework-employing Cox proportional hazards, Accelerated Failure Time, and Random Survival Forest approaches-reveals extraordinary temporal dynamics. We find that abrupt, prompt-to-prompt(P2P) semantic drift is catastrophic, dramatically increasing the hazard of conversational failure. In stark contrast, gradual, cumulative drift is highly protective, vastly reducing the failure hazard and enabling significantly longer dialogues. AFT models with interactions demonstrate superior performance, achieving excellent discrimination and exceptional calibration. These findings establish survival analysis as a powerful paradigm for evaluating LLM robustness, offer concrete insights for designing resilient conversational agents, and challenge prevailing assumptions about the necessity of semantic consistency in conversational AI Systems. 

---
# Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering 

**Authors**: Yavuz Bakman, Sungmin Kang, Zhiqi Huang, Duygu Nur Yaldiz, Catarina G. Belém, Chenyang Zhu, Anoop Kumar, Alfy Samuel, Salman Avestimehr, Daben Liu, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.02671)  

**Abstract**: Uncertainty Quantification (UQ) research has primarily focused on closed-book factual question answering (QA), while contextual QA remains unexplored, despite its importance in real-world applications. In this work, we focus on UQ for the contextual QA task and propose a theoretically grounded approach to quantify epistemic uncertainty. We begin by introducing a task-agnostic, token-level uncertainty measure defined as the cross-entropy between the predictive distribution of the given model and the unknown true distribution. By decomposing this measure, we isolate the epistemic component and approximate the true distribution by a perfectly prompted, idealized model. We then derive an upper bound for epistemic uncertainty and show that it can be interpreted as semantic feature gaps in the given model's hidden representations relative to the ideal model. We further apply this generic framework to the contextual QA task and hypothesize that three features approximate this gap: context-reliance (using the provided context rather than parametric knowledge), context comprehension (extracting relevant information from context), and honesty (avoiding intentional lies). Using a top-down interpretability approach, we extract these features by using only a small number of labeled samples and ensemble them to form a robust uncertainty score. Experiments on multiple QA benchmarks in both in-distribution and out-of-distribution settings show that our method substantially outperforms state-of-the-art unsupervised (sampling-free and sampling-based) and supervised UQ methods, achieving up to a 13-point PRR improvement while incurring a negligible inference overhead. 

---
# Self-Improvement in Multimodal Large Language Models: A Survey 

**Authors**: Shijian Deng, Kai Wang, Tianyu Yang, Harsh Singh, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2510.02665)  

**Abstract**: Recent advancements in self-improvement for Large Language Models (LLMs) have efficiently enhanced model capabilities without significantly increasing costs, particularly in terms of human effort. While this area is still relatively young, its extension to the multimodal domain holds immense potential for leveraging diverse data sources and developing more general self-improving models. This survey is the first to provide a comprehensive overview of self-improvement in Multimodal LLMs (MLLMs). We provide a structured overview of the current literature and discuss methods from three perspectives: 1) data collection, 2) data organization, and 3) model optimization, to facilitate the further development of self-improvement in MLLMs. We also include commonly used evaluations and downstream applications. Finally, we conclude by outlining open challenges and future research directions. 

---
# SoT: Structured-of-Thought Prompting Guides Multilingual Reasoning in Large Language Models 

**Authors**: Rui Qi, Zhibo Man, Yufeng Chen, Fengran Mo, Jinan Xu, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02648)  

**Abstract**: Recent developments have enabled Large Language Models (LLMs) to engage in complex reasoning tasks through deep thinking. However, the capacity of reasoning has not been successfully transferred to non-high-resource languages due to resource constraints, which struggles with multilingual reasoning tasks. To this end, we propose Structured-of-Thought (SoT), a training-free method that improves the performance on multilingual reasoning through a multi-step transformation: Language Thinking Transformation and Structured Knowledge Transformation. The SoT method converts language-specific semantic information into language-agnostic structured representations, enabling the models to understand the query in different languages more sophisticated. Besides, SoT effectively guides LLMs toward more concentrated reasoning to maintain consistent underlying reasoning pathways when handling cross-lingual variations in expression. Experimental results demonstrate that SoT outperforms several strong baselines on multiple multilingual reasoning benchmarks when adapting to various backbones of LLMs. It can also be integrated with other training-free strategies for further improvements. Our code is available at this https URL. 

---
# Mind the Gap: Linguistic Divergence and Adaptation Strategies in Human-LLM Assistant vs. Human-Human Interactions 

**Authors**: Fulei Zhang, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02645)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in customer-facing applications, a critical yet underexplored question is how users communicate differently with LLM chatbots compared to human agent. In this study, we present empirical evidence that users adopt distinct communication styles when users interact with chatbots versus human agents. Our analysis reveals significant differences in grammatical fluency, politeness, and lexical diversity in user language between the two settings. These findings suggest that models trained exclusively on human-human interaction data may not adequately accommodate the communication style shift that occurs once an LLM chatbot is deployed. To enhance LLM robustness to post-launch communication style changes, we experimented with two strategies: (1) data augmentation during the post-training phase and (2) inference-time user message reformulation. Our results indicate that models trained on stylistically diverse datasets significantly outperform those trained exclusively on original or stylistically uniform datasets, while inference-time reformulation proved less effective. These insights help us to better adapt our models for improved LLM-user interaction experiences. 

---
# Evaluation Framework for Highlight Explanations of Context Utilisation in Language Models 

**Authors**: Jingyi Sun, Pepa Atanasova, Sagnik Ray Choudhury, Sekh Mainul Islam, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.02629)  

**Abstract**: Context utilisation, the ability of Language Models (LMs) to incorporate relevant information from the provided context when generating responses, remains largely opaque to users, who cannot determine whether models draw from parametric memory or provided context, nor identify which specific context pieces inform the response. Highlight explanations (HEs) offer a natural solution as they can point the exact context pieces and tokens that influenced model outputs. However, no existing work evaluates their effectiveness in accurately explaining context utilisation. We address this gap by introducing the first gold standard HE evaluation framework for context attribution, using controlled test cases with known ground-truth context usage, which avoids the limitations of existing indirect proxy evaluations. To demonstrate the framework's broad applicability, we evaluate four HE methods -- three established techniques and MechLight, a mechanistic interpretability approach we adapt for this task -- across four context scenarios, four datasets, and five LMs. Overall, we find that MechLight performs best across all context scenarios. However, all methods struggle with longer contexts and exhibit positional biases, pointing to fundamental challenges in explanation accuracy that require new approaches to deliver reliable context utilisation explanations at scale. 

---
# Transcribe, Translate, or Transliterate: An Investigation of Intermediate Representations in Spoken Language Models 

**Authors**: Tolúl\d{o}pé Ògúnrèmí, Christopher D. Manning, Dan Jurafsky, Karen Livescu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02569)  

**Abstract**: Spoken language models (SLMs) that integrate speech with large language models (LMs) rely on modality adapters (MAs) to map the output of speech encoders to a representation that is understandable to the decoder LM. Yet we know very little about how these crucial MAs transform representations. Here we examine the MA output representation in three SLMs (SALMONN, Qwen2-Audio and Phi-4-Multimodal-Instruct). By finding the nearest decoder LM token to an MA representation, we uncover two strategies for MA representations. For models using a Whisper encoder, MAs appear to represent the meaning of the input using an English-based interlingua, allowing them to handle languages unseen in instruction tuning. For models that don't, like Phi-4-Multimodal-Instruct, MAs instead represent the phonetics of the input, but expressed with English words. We hypothesise that which arises depends on whether the speech encoder is trained only for speech recognition or also for translation. 

---
# Knowledge-Graph Based RAG System Evaluation Framework 

**Authors**: Sicheng Dong, Vahid Zolfaghari, Nenad Petrovic, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2510.02549)  

**Abstract**: Large language models (LLMs) has become a significant research focus and is utilized in various fields, such as text generation and dialog systems. One of the most essential applications of LLM is Retrieval Augmented Generation (RAG), which greatly enhances generated content's reliability and relevance. However, evaluating RAG systems remains a challenging task. Traditional evaluation metrics struggle to effectively capture the key features of modern LLM-generated content that often exhibits high fluency and naturalness. Inspired by the RAGAS tool, a well-known RAG evaluation framework, we extended this framework into a KG-based evaluation paradigm, enabling multi-hop reasoning and semantic community clustering to derive more comprehensive scoring metrics. By incorporating these comprehensive evaluation criteria, we gain a deeper understanding of RAG systems and a more nuanced perspective on their performance. To validate the effectiveness of our approach, we compare its performance with RAGAS scores and construct a human-annotated subset to assess the correlation between human judgments and automated metrics. In addition, we conduct targeted experiments to demonstrate that our KG-based evaluation method is more sensitive to subtle semantic differences in generated outputs. Finally, we discuss the key challenges in evaluating RAG systems and highlight potential directions for future research. 

---
# Hierarchical Semantic Retrieval with Cobweb 

**Authors**: Anant Gupta, Karthik Singaravadivelan, Zekun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02539)  

**Abstract**: Neural document retrieval often treats a corpus as a flat cloud of vectors scored at a single granularity, leaving corpus structure underused and explanations opaque. We use Cobweb--a hierarchy-aware framework--to organize sentence embeddings into a prototype tree and rank documents via coarse-to-fine traversal. Internal nodes act as concept prototypes, providing multi-granular relevance signals and a transparent rationale through retrieval paths. We instantiate two inference approaches: a generalized best-first search and a lightweight path-sum ranker. We evaluate our approaches on MS MARCO and QQP with encoder (e.g., BERT/T5) and decoder (GPT-2) representations. Our results show that our retrieval approaches match the dot product search on strong encoder embeddings while remaining robust when kNN degrades: with GPT-2 vectors, dot product performance collapses whereas our approaches still retrieve relevant results. Overall, our experiments suggest that Cobweb provides competitive effectiveness, improved robustness to embedding quality, scalability, and interpretable retrieval via hierarchical prototypes. 

---
# Unraveling Syntax: How Language Models Learn Context-Free Grammars 

**Authors**: Laura Ying Schulz, Daniel Mitropolsky, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2510.02524)  

**Abstract**: We introduce a new framework for understanding how language models acquire syntax. While large models achieve impressive results, little is known about their learning dynamics. Our approach starts with the observation that most domains of interest, such as natural language syntax, coding languages, arithmetic problems, are captured by probabilistic context-free grammars (PCFGs). We study the learning dynamics of small models trained on synthetic languages generated from PCFGs, enabling precise control over grammar complexity, recursion depth, and subgrammar structure. We prove several general, recursive formulae for the training loss and Kullback-Leibler divergence over the subgrammar structure of a PCFG. Empirically, we find that unlike children, who first master simple substructures before progressing to more complex constructions, transformers reduce loss across all subgrammars in parallel. We further show that subgrammar pretraining can improve the final loss for smaller models, and that pretrained models develop internal representations more aligned with the grammar's substructure. Finally, we demonstrate that models struggle with deeper recursive structures (a limitation even of large language models), revealing fundamental challenges in how neural networks represent hierarchical syntax. Overall, our work initiates the study of the learning dynamics of transformers on PCFGs as a versatile testbed for probing learning in language models, opening a research direction with many open questions. 

---
# CLARITY: Clinical Assistant for Routing, Inference, and Triage 

**Authors**: Vladimir Shaposhnikov, Aleksandr Nesterov, Ilia Kopanichuk, Ivan Bakulin, Egor Zhelvakov, Ruslan Abramov, Ekaterina Tsapieva, Dmitry V. Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2510.02463)  

**Abstract**: We present CLARITY (Clinical Assistant for Routing, Inference, and Triage), an AI-driven platform designed to facilitate patient-to-specialist routing, clinical consultations, and severity assessment of patients' conditions. Its hybrid architecture combines a Finite State Machine (FSM) for structured dialogue flows with collaborative agents that employ Large Language Model (LLM) to analyze symptoms and prioritize referrals to appropriate specialists. Built on a modular microservices framework, CLARITY ensures safe, efficient, and robust performance, flexible and readily scalable to meet the demands of existing workflows and IT solutions in healthcare.
We report integration of our clinical assistant into a large-scale nation-wide inter-hospital IT platform, with over 55,000 content-rich user dialogues completed within the two months of deployment, 2,500 of which were expert-annotated for a consequent validation. The validation results show that CLARITY surpasses human-level performance in terms of the first-attempt routing precision, naturally requiring up to 3 times shorter duration of the consultation than with a human. 

---
# Words That Make Language Models Perceive 

**Authors**: Sophie L. Wang, Phillip Isola, Brian Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2510.02425)  

**Abstract**: Large language models (LLMs) trained purely on text ostensibly lack any direct perceptual experience, yet their internal representations are implicitly shaped by multimodal regularities encoded in language. We test the hypothesis that explicit sensory prompting can surface this latent structure, bringing a text-only LLM into closer representational alignment with specialist vision and audio encoders. When a sensory prompt tells the model to 'see' or 'hear', it cues the model to resolve its next-token predictions as if they were conditioned on latent visual or auditory evidence that is never actually supplied. Our findings reveal that lightweight prompt engineering can reliably activate modality-appropriate representations in purely text-trained LLMs. 

---
# Retrieval and Augmentation of Domain Knowledge for Text-to-SQL Semantic Parsing 

**Authors**: Manasi Patwardhan, Ayush Agarwal, Shabbirhussain Bhaisaheb, Aseem Arora, Lovekesh Vig, Sunita Sarawagi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02394)  

**Abstract**: The performance of Large Language Models (LLMs) for translating Natural Language (NL) queries into SQL varies significantly across databases (DBs). NL queries are often expressed using a domain specific vocabulary, and mapping these to the correct SQL requires an understanding of the embedded domain expressions, their relationship to the DB schema structure. Existing benchmarks rely on unrealistic, ad-hoc query specific textual hints for expressing domain knowledge. In this paper, we propose a systematic framework for associating structured domain statements at the database level. We present retrieval of relevant structured domain statements given a user query using sub-string level match. We evaluate on eleven realistic DB schemas covering diverse domains across five open-source and proprietary LLMs and demonstrate that (1) DB level structured domain statements are more practical and accurate than existing ad-hoc query specific textual domain statements, and (2) Our sub-string match based retrieval of relevant domain statements provides significantly higher accuracy than other retrieval approaches. 

---
# KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning 

**Authors**: Yinyi Luo, Zhexian Zhou, Hao Chen, Kai Qiu, Marios Savvides, Yixuan Li, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02392)  

**Abstract**: Knowledge editing and machine unlearning are two popular approaches for large language models (LLMs) to stay up-to-date. However, the knowledge updating mechanism of LLMs remains largely unexplored due to insufficient, isolated, and small-scale evaluation. For instance, are LLMs similar to humans in modifying certain knowledge? What differs editing and unlearning as training data increases? This paper proposes KnowledgeSmith, a unified framework to systematically understand the updating mechanism of LLMs. We first cast editing and unlearning as instances of one constrained optimization problem. Then, we propose an automatic dataset generator that provides structured interventions across multiple graph levels and data scales, enabling controlled studies of how different modification strategies propagate through model knowledge. Extensive experiments demonstrate nuanced insights over knowledge propagation, plasticity scaling, consistency, and robustness. For instance, our results show that LLMs do not exhibit similar updating as humans for different levels of knowledge, and there exists consistency-capacity trade-off. We hope our findings can offer suggestions to the design of more reliable and scalable strategies. Code: this https URL 

---
# Learning to Route: A Rule-Driven Agent Framework for Hybrid-Source Retrieval-Augmented Generation 

**Authors**: Haoyue Bai, Haoyu Wang, Shengyu Chen, Zhengzhang Chen, Lu-An Tang, Wei Cheng, Haifeng Chen, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02388)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance on general Question Answering (QA), yet they often struggle in domain-specific scenarios where accurate and up-to-date information is required. Retrieval-Augmented Generation (RAG) addresses this limitation by enriching LLMs with external knowledge, but existing systems primarily rely on unstructured documents, while largely overlooking relational databases, which provide precise, timely, and efficiently queryable factual information, serving as indispensable infrastructure in domains such as finance, healthcare, and scientific research. Motivated by this gap, we conduct a systematic analysis that reveals three central observations: (i) databases and documents offer complementary strengths across queries, (ii) naively combining both sources introduces noise and cost without consistent accuracy gains, and (iii) selecting the most suitable source for each query is crucial to balance effectiveness and efficiency. We further observe that query types show consistent regularities in their alignment with retrieval paths, suggesting that routing decisions can be effectively guided by systematic rules that capture these patterns. Building on these insights, we propose a rule-driven routing framework. A routing agent scores candidate augmentation paths based on explicit rules and selects the most suitable one; a rule-making expert agent refines the rules over time using QA feedback to maintain adaptability; and a path-level meta-cache reuses past routing decisions for semantically similar queries to reduce latency and cost. Experiments on three QA benchmarks demonstrate that our framework consistently outperforms static strategies and learned routing baselines, achieving higher accuracy while maintaining moderate computational cost. 

---
# Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems 

**Authors**: Aakriti Agrawal, Rohith Aralikatti, Anirudh Satheesh, Souradip Chakraborty, Amrit Singh Bedi, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02377)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities, yet selecting the most reliable response from multiple LLMs remains a challenge, particularly in resource-constrained settings. Existing approaches often depend on costly external verifiers, human evaluators, or self-consistency techniques that require multiple samples from a single model. While multi-LLM systems produce more diverse responses than single models and thus have greater potential, they often underperform compared to single LLM self-consistency. We propose a principled, novel and computationally efficient method to select the best response from multiple different LLMs using a calibrated log-likelihood score, implicitly leveraging the inherent knowledge and confidence of these models. Our method demonstrates improvements of approx. 4%, 3%, and 5% across both debate (multi-round LLM discussions) and non-debate (Best-of-N with multiple LLMs) settings on GSM8K, MMLU (6 subsets), and ARC datasets respectively. 

---
# Pretraining with hierarchical memories: separating long-tail and common knowledge 

**Authors**: Hadi Pouransari, David Grangier, C Thomas, Michael Kirchhof, Oncel Tuzel  

**Link**: [PDF](https://arxiv.org/pdf/2510.02375)  

**Abstract**: The impressive performance gains of modern language models currently rely on scaling parameters: larger models store more world knowledge and reason better. Yet compressing all world knowledge into parameters is unnecessary, as only a fraction is used per prompt, and impractical for edge devices with limited inference-time memory and compute. We address this shortcoming by a memory-augmented architecture and a pretraining strategy aligned with existing hardware paradigms. We introduce small language models that access large hierarchical parametric memory banks encoding world knowledge. During pretraining and inference, we fetch a small, context-dependent memory block and add it to the model. Our pretraining learns to store long-tail world knowledge in the memory parameters, while the small language model acts as an anchor capturing common knowledge and general reasoning abilities. Through trillion-token-scale experiments, we show significant gains: a 160M-parameters model augmented with an 18M-parameters memory fetched from a 4.6B memory bank obtains comparable performance to a regular model with more than 2x the parameters. Through extensive experiments, we study the optimal type and size of parametric memories in transformers, scaling them to over 21B parameters. We find that our proposed hierarchical feed-forward memories work robustly across transformer architectures, whether added during pretraining or post-hoc. 

---
# Training Dynamics of Parametric and In-Context Knowledge Utilization in Language Models 

**Authors**: Minsung Kim, Dong-Kyum Kim, Jea Kwon, Nakyeong Yang, Kyomin Jung, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2510.02370)  

**Abstract**: Large language models often encounter conflicts between in-context knowledge retrieved at inference time and parametric knowledge acquired during pretraining. Models that accept external knowledge uncritically are vulnerable to misinformation, whereas models that adhere rigidly to parametric knowledge fail to benefit from retrieval. Despite the widespread adoption of retrieval-augmented generation, we still lack a systematic understanding of what shapes knowledge-arbitration strategies during training. This gap risks producing pretrained models with undesirable arbitration behaviors and, consequently, wasting substantial computational resources after the pretraining budget has already been spent. To address this problem, we present the first controlled study of how training conditions influence models' use of in-context and parametric knowledge, and how they arbitrate between them. We train transformer-based language models on a synthetic biographies corpus while systematically controlling various conditions. Our experiments reveal that intra-document repetition of facts fosters the development of both parametric and in-context capabilities. Moreover, training on a corpus that contains inconsistent information or distributional skew encourages models to develop robust strategies for leveraging parametric and in-context knowledge. Rather than viewing these non-ideal properties as artifacts to remove, our results indicate that they are important for learning robust arbitration. These insights offer concrete, empirical guidance for pretraining models that harmoniously integrate parametric and in-context knowledge. 

---
# Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents 

**Authors**: Kuntai Cai, Juncheng Liu, Xianglin Yang, Zhaojie Niu, Xiaokui Xiao, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02369)  

**Abstract**: Large language model (LLM) agents typically receive two kinds of context: (i) environment-level manuals that define interaction interfaces and global rules, and (ii) task-level guidance or demonstrations tied to specific goals. In this work, we identify a crucial but overlooked third type of context, instance-level context, which consists of verifiable and reusable facts tied to a specific environment instance, such as object locations, crafting recipes, and local rules. We argue that the absence of instance-level context is a common source of failure for LLM agents in complex tasks, as success often depends not only on reasoning over global rules or task prompts but also on making decisions based on precise and persistent facts. Acquiring such context requires more than memorization: the challenge lies in efficiently exploring, validating, and formatting these facts under tight interaction budgets. We formalize this problem as Instance-Level Context Learning (ILCL) and introduce our task-agnostic method to solve it. Our method performs a guided exploration, using a compact TODO forest to intelligently prioritize its next actions and a lightweight plan-act-extract loop to execute them. This process automatically produces a high-precision context document that is reusable across many downstream tasks and agents, thereby amortizing the initial exploration cost. Experiments across TextWorld, ALFWorld, and Crafter demonstrate consistent gains in both success and efficiency: for instance, ReAct's mean success rate in TextWorld rises from 37% to 95%, while IGE improves from 81% to 95%. By transforming one-off exploration into persistent, reusable knowledge, our method complements existing contexts to enable more reliable and efficient LLM agents. 

---
# A Cross-Lingual Analysis of Bias in Large Language Models Using Romanian History 

**Authors**: Matei-Iulian Cocu, Răzvan-Cosmin Cristia, Adrian Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2510.02362)  

**Abstract**: In this case study, we select a set of controversial Romanian historical questions and ask multiple Large Language Models to answer them across languages and contexts, in order to assess their biases. Besides being a study mainly performed for educational purposes, the motivation also lies in the recognition that history is often presented through altered perspectives, primarily influenced by the culture and ideals of a state, even through large language models. Since they are often trained on certain data sets that may present certain ambiguities, the lack of neutrality is subsequently instilled in users. The research process was carried out in three stages, to confirm the idea that the type of response expected can influence, to a certain extent, the response itself; after providing an affirmative answer to some given question, an LLM could shift its way of thinking after being asked the same question again, but being told to respond with a numerical value of a scale. Results show that binary response stability is relatively high but far from perfect and varies by language. Models often flip stance across languages or between formats; numeric ratings frequently diverge from the initial binary choice, and the most consistent models are not always those judged most accurate or neutral. Our research brings to light the predisposition of models to such inconsistencies, within a specific contextualization of the language for the question asked. 

---
# ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inference 

**Authors**: Haojie Ouyang, Jianwei Lv, Lei Ren, Chen Wei, Xiaojie Wang, Fangxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02361)  

**Abstract**: Transformer-based large models excel in natural language processing and computer vision, but face severe computational inefficiencies due to the self-attention's quadratic complexity with input tokens. Recently, researchers have proposed a series of methods based on block selection and compression to alleviate this problem, but they either have issues with semantic incompleteness or poor training-inference efficiency. To comprehensively address these challenges, we propose ChunkLLM, a lightweight and pluggable training framework. Specifically, we introduce two components: QK Adapter (Q-Adapter and K-Adapter) and Chunk Adapter. The former is attached to each Transformer layer, serving dual purposes of feature compression and chunk attention acquisition. The latter operates at the bottommost layer of the model, functioning to detect chunk boundaries by leveraging contextual semantic information. During the training phase, the parameters of the backbone remain frozen, with only the QK Adapter and Chunk Adapter undergoing training. Notably, we design an attention distillation method for training the QK Adapter, which enhances the recall rate of key chunks. During the inference phase, chunk selection is triggered exclusively when the current token is detected as a chunk boundary, thereby accelerating model inference. Experimental evaluations are conducted on a diverse set of long-text and short-text benchmark datasets spanning multiple tasks. ChunkLLM not only attains comparable performance on short-text benchmarks but also maintains 98.64% of the performance on long-context benchmarks while preserving a 48.58% key-value cache retention rate. Particularly, ChunkLLM attains a maximum speedup of 4.48x in comparison to the vanilla Transformer in the processing of 120K long texts. 

---
# Spiral of Silence in Large Language Model Agents 

**Authors**: Mingze Zhong, Meng Fang, Zijing Shi, Yuxuan Huang, Shunfeng Zheng, Yali Du, Ling Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02360)  

**Abstract**: The Spiral of Silence (SoS) theory holds that individuals with minority views often refrain from speaking out for fear of social isolation, enabling majority positions to dominate public discourse. When the 'agents' are large language models (LLMs), however, the classical psychological explanation is not directly applicable, since SoS was developed for human societies. This raises a central question: can SoS-like dynamics nevertheless emerge from purely statistical language generation in LLM collectives? We propose an evaluation framework for examining SoS in LLM agents. Specifically, we consider four controlled conditions that systematically vary the availability of 'History' and 'Persona' signals. Opinion dynamics are assessed using trend tests such as Mann-Kendall and Spearman's rank, along with concentration measures including kurtosis and interquartile range. Experiments across open-source and closed-source models show that history and persona together produce strong majority dominance and replicate SoS patterns; history signals alone induce strong anchoring; and persona signals alone foster diverse but uncorrelated opinions, indicating that without historical anchoring, SoS dynamics cannot emerge. The work bridges computational sociology and responsible AI design, highlighting the need to monitor and mitigate emergent conformity in LLM-agent systems. 

---
# Emission-GPT: A domain-specific language model agent for knowledge retrieval, emission inventory and data analysis 

**Authors**: Jiashu Ye, Tong Wu, Weiwen Chen, Hao Zhang, Zeteng Lin, Xingxing Li, Shujuan Weng, Manni Zhu, Xin Yuan, Xinlong Hong, Jingjie Li, Junyu Zheng, Zhijiong Huang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02359)  

**Abstract**: Improving air quality and addressing climate change relies on accurate understanding and analysis of air pollutant and greenhouse gas emissions. However, emission-related knowledge is often fragmented and highly specialized, while existing methods for accessing and compiling emissions data remain inefficient. These issues hinder the ability of non-experts to interpret emissions information, posing challenges to research and management. To address this, we present Emission-GPT, a knowledge-enhanced large language model agent tailored for the atmospheric emissions domain. Built on a curated knowledge base of over 10,000 documents (including standards, reports, guidebooks, and peer-reviewed literature), Emission-GPT integrates prompt engineering and question completion to support accurate domain-specific question answering. Emission-GPT also enables users to interactively analyze emissions data via natural language, such as querying and visualizing inventories, analyzing source contributions, and recommending emission factors for user-defined scenarios. A case study in Guangdong Province demonstrates that Emission-GPT can extract key insights--such as point source distributions and sectoral trends--directly from raw data with simple prompts. Its modular and extensible architecture facilitates automation of traditionally manual workflows, positioning Emission-GPT as a foundational tool for next-generation emission inventory development and scenario-based assessment. 

---
# DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding 

**Authors**: Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02358)  

**Abstract**: As large language models (LLMs) scale up, accuracy improves, but the autoregressive (AR) nature of decoding increases latency since each token requires a serial forward pass. Speculative decoding addresses this by employing a fast drafter to propose multi-token drafts, which are then verified in parallel by the target model. However, many deployments still rely on AR drafters, where sequential passes limit wall-clock gains. We revisit the drafting stage and present DiffuSpec, a training-free drop-in framework that uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers. Because DLM drafts are generated under bidirectional conditioning, parallel per-position candidates form a token lattice in which the locally highest-probability token at each position need not form a causal left-to-right path. Moreover, DLM drafting requires pre-specifying a draft length, inducing a speed-quality trade-off. To address these challenges, we introduce two practical components: (i) a causal-consistency path search (CPS) over this lattice that extracts a left-to-right path aligned with AR verification; and (ii) an adaptive draft-length (ADL) controller that adjusts next proposal size based on recent acceptance feedback and realized generated length. Across benchmarks, DiffuSpec yields up to 3x wall-clock speedup, establishing diffusion-based drafting as a robust alternative to autoregressive drafters for speculative decoding. 

---
# Modeling the language cortex with form-independent and enriched representations of sentence meaning reveals remarkable semantic abstractness 

**Authors**: Shreya Saha, Shurui Li, Greta Tuckute, Yuanning Li, Ru-Yuan Zhang, Leila Wehbe, Evelina Fedorenko, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2510.02354)  

**Abstract**: The human language system represents both linguistic forms and meanings, but the abstractness of the meaning representations remains debated. Here, we searched for abstract representations of meaning in the language cortex by modeling neural responses to sentences using representations from vision and language models. When we generate images corresponding to sentences and extract vision model embeddings, we find that aggregating across multiple generated images yields increasingly accurate predictions of language cortex responses, sometimes rivaling large language models. Similarly, averaging embeddings across multiple paraphrases of a sentence improves prediction accuracy compared to any single paraphrase. Enriching paraphrases with contextual details that may be implicit (e.g., augmenting "I had a pancake" to include details like "maple syrup") further increases prediction accuracy, even surpassing predictions based on the embedding of the original sentence, suggesting that the language system maintains richer and broader semantic representations than language models. Together, these results demonstrate the existence of highly abstract, form-independent meaning representations within the language cortex. 

---
# An Senegalese Legal Texts Structuration Using LLM-augmented Knowledge Graph 

**Authors**: Oumar Kane, Mouhamad M. Allaya, Dame Samb, Mamadou Bousso  

**Link**: [PDF](https://arxiv.org/pdf/2510.02353)  

**Abstract**: This study examines the application of artificial intelligence (AI) and large language models (LLM) to improve access to legal texts in Senegal's judicial system. The emphasis is on the difficulties of extracting and organizing legal documents, highlighting the need for better access to judicial information. The research successfully extracted 7,967 articles from various legal documents, particularly focusing on the Land and Public Domain Code. A detailed graph database was developed, which contains 2,872 nodes and 10,774 relationships, aiding in the visualization of interconnections within legal texts. In addition, advanced triple extraction techniques were utilized for knowledge, demonstrating the effectiveness of models such as GPT-4o, GPT-4, and Mistral-Large in identifying relationships and relevant metadata. Through these technologies, the aim is to create a solid framework that allows Senegalese citizens and legal professionals to more effectively understand their rights and responsibilities. 

---
# Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations 

**Authors**: Yihao Wu, Tianrui Wang, Yizhou Peng, Yi-Wen Chao, Xuyi Zhuang, Xinsheng Wang, Shunshun Yin, Ziyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.02352)  

**Abstract**: While biases in large language models (LLMs), such as stereotypes and cultural tendencies in outputs, have been examined and identified, their presence and characteristics in spoken dialogue models (SDMs) with audio input and output remain largely unexplored. Paralinguistic features, such as age, gender, and accent, can affect model outputs; when compounded by multi-turn conversations, these effects may exacerbate biases, with potential implications for fairness in decision-making and recommendation tasks. In this paper, we systematically evaluate biases in speech LLMs and study the impact of multi-turn dialogues with repeated negative feedback. Bias is measured using Group Unfairness Score (GUS) for decisions and similarity-based normalized statistics rate (SNSR) for recommendations, across both open-source models like Qwen2.5-Omni and GLM-4-Voice, as well as closed-source APIs such as GPT-4o Audio and Gemini-2.5-Flash. Our analysis reveals that closed-source models generally exhibit lower bias, while open-source models are more sensitive to age and gender, and recommendation tasks tend to amplify cross-group disparities. We found that biased decisions may persist in multi-turn conversations. This work provides the first systematic study of biases in end-to-end spoken dialogue models, offering insights towards fair and reliable audio-based interactive systems. To facilitate further research, we release the FairDialogue dataset and evaluation code. 

---
# Language, Culture, and Ideology: Personalizing Offensiveness Detection in Political Tweets with Reasoning LLMs 

**Authors**: Dzmitry Pihulski, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2510.02351)  

**Abstract**: We explore how large language models (LLMs) assess offensiveness in political discourse when prompted to adopt specific political and cultural perspectives. Using a multilingual subset of the MD-Agreement dataset centered on tweets from the 2020 US elections, we evaluate several recent LLMs - including DeepSeek-R1, o4-mini, GPT-4.1-mini, Qwen3, Gemma, and Mistral - tasked with judging tweets as offensive or non-offensive from the viewpoints of varied political personas (far-right, conservative, centrist, progressive) across English, Polish, and Russian contexts. Our results show that larger models with explicit reasoning abilities (e.g., DeepSeek-R1, o4-mini) are more consistent and sensitive to ideological and cultural variation, while smaller models often fail to capture subtle distinctions. We find that reasoning capabilities significantly improve both the personalization and interpretability of offensiveness judgments, suggesting that such mechanisms are key to adapting LLMs for nuanced sociopolitical text classification across languages and ideologies. 

---
# LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL 

**Authors**: Dzmitry Pihulski, Karol Charchut, Viktoria Novogrodskaia, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2510.02350)  

**Abstract**: Converting natural language questions into SQL queries (Text-to-SQL) enables non-expert users to interact with relational databases and has long been a central task for natural language interfaces to data. While the WikiSQL dataset played a key role in early NL2SQL research, its usage has declined due to structural and annotation issues, including case sensitivity inconsistencies, data type mismatches, syntax errors, and unanswered questions. We present LLMSQL, a systematic revision and transformation of WikiSQL designed for the LLM era. We classify these errors and implement automated methods for cleaning and re-annotation. To assess the impact of these improvements, we evaluated multiple large language models (LLMs), including Gemma 3, LLaMA 3.2, Mistral 7B, gpt-oss 20B, Phi-3.5 Mini, Qwen 2.5, OpenAI o4-mini, DeepSeek R1 and others. Rather than serving as an update, LLMSQL is introduced as an LLM-ready benchmark: unlike the original WikiSQL, tailored for pointer-network models selecting tokens from input, LLMSQL provides clean natural language questions and full SQL queries as plain text, enabling straightforward generation and evaluation for modern natural language-to-SQL models. 

---
# mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations 

**Authors**: Guy Dar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02348)  

**Abstract**: We build upon vec2vec, a procedure designed to align text embedding spaces without parallel data. vec2vec finds a near-perfect alignment, but it is expensive and unstable. We present mini-vec2vec, a simple and efficient alternative that requires substantially lower computational cost and is highly robust. Moreover, the learned mapping is a linear transformation. Our method consists of three main stages: a tentative matching of pseudo-parallel embedding vectors, transformation fitting, and iterative refinement. Our linear alternative exceeds the original instantiation of vec2vec by orders of magnitude in efficiency, while matching or exceeding their results. The method's stability and interpretable algorithmic steps facilitate scaling and unlock new opportunities for adoption in new domains and fields. 

---
# Small Language Models for Curriculum-based Guidance 

**Authors**: Konstantinos Katharakis, Sippo Rossi, Raghava Rao Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2510.02347)  

**Abstract**: The adoption of generative AI and large language models (LLMs) in education is still emerging. In this study, we explore the development and evaluation of AI teaching assistants that provide curriculum-based guidance using a retrieval-augmented generation (RAG) pipeline applied to selected open-source small language models (SLMs). We benchmarked eight SLMs, including LLaMA 3.1, IBM Granite 3.3, and Gemma 3 (7-17B parameters), against GPT-4o. Our findings show that with proper prompting and targeted retrieval, SLMs can match LLMs in delivering accurate, pedagogically aligned responses. Importantly, SLMs offer significant sustainability benefits due to their lower computational and energy requirements, enabling real-time use on consumer-grade hardware without depending on cloud infrastructure. This makes them not only cost-effective and privacy-preserving but also environmentally responsible, positioning them as viable AI teaching assistants for educational institutions aiming to scale personalized learning in a sustainable and energy-efficient manner. 

---
# Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression 

**Authors**: Peijun Zhu, Ning Yang, Jiayu Wei, Jinghang Wu, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02345)  

**Abstract**: Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs. 

---
# $\texttt{BluePrint}$: A Social Media User Dataset for LLM Persona Evaluation and Training 

**Authors**: Aurélien Bück-Kaeffer, Je Qin Chooi, Dan Zhao, Maximilian Puelma Touzel, Kellin Pelrine, Jean-François Godbout, Reihaneh Rabbany, Zachary Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02343)  

**Abstract**: Large language models (LLMs) offer promising capabilities for simulating social media dynamics at scale, enabling studies that would be ethically or logistically challenging with human subjects. However, the field lacks standardized data resources for fine-tuning and evaluating LLMs as realistic social media agents. We address this gap by introducing SIMPACT, the SIMulation-oriented Persona and Action Capture Toolkit, a privacy respecting framework for constructing behaviorally-grounded social media datasets suitable for training agent models. We formulate next-action prediction as a task for training and evaluating LLM-based agents and introduce metrics at both the cluster and population levels to assess behavioral fidelity and stylistic realism. As a concrete implementation, we release BluePrint, a large-scale dataset built from public Bluesky data focused on political discourse. BluePrint clusters anonymized users into personas of aggregated behaviours, capturing authentic engagement patterns while safeguarding privacy through pseudonymization and removal of personally identifiable information. The dataset includes a sizable action set of 12 social media interaction types (likes, replies, reposts, etc.), each instance tied to the posting activity preceding it. This supports the development of agents that use context-dependence, not only in the language, but also in the interaction behaviours of social media to model social media users. By standardizing data and evaluation protocols, SIMPACT provides a foundation for advancing rigorous, ethically responsible social media simulations. BluePrint serves as both an evaluation benchmark for political discourse modeling and a template for building domain specific datasets to study challenges such as misinformation and polarization. 

---
# DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning 

**Authors**: Yifan Wang, Bolian Li, Junlin Wu, Zhaoxuan Tan, Zheli Liu, Ruqi Zhang, Ananth Grama, Qingkai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02341)  

**Abstract**: Real-world large language model deployments (e.g., conversational AI systems, code generation assistants) naturally generate abundant implicit user dissatisfaction (DSAT) signals, as users iterate toward better answers through refinements, corrections, and expressed preferences, while explicit satisfaction (SAT) feedback is scarce. Existing preference learning approaches are poorly aligned with this data profile, as they rely on costly human annotations or assume plentiful positive responses. In this paper, we introduce \textbf{DRIFT} (\textbf{D}issatisfaction-\textbf{R}efined \textbf{I}terative pre\textbf{F}erence \textbf{T}raining), which anchors training on real-world DSAT signals and samples positives dynamically from the evolving policy. Empirically, DRIFT models trained on real-world \textit{WildFeedback} datasets and synthetic \textit{UltraFeedback} datasets achieve up to +6.23\% (7B) / +7.61\% (14B) on WildBench Task Score and up to +8.95\% (7B) / +12.29\% (14B) on AlpacaEval2 win rate over base models, outperforming strong baseline methods such as iterative DPO and SPIN. At larger scales, the improvements are particularly pronounced: 14B models trained with DRIFT surpass GPT-4o-mini on WildBench. Further analysis shows that DRIFT also preserves exploratory capacity, yielding more diverse high-reward solutions rather than collapsing to narrow subsets. Theoretically, we demonstrate that this design preserves preference margins and avoids the gradient degeneration. These results show that DRIFT is an effective and scalable recipe for real-world post-training that leverages the most abundant and informative signal. The code and data are available at this https URL. 

---
# Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs 

**Authors**: Xin Gao, Ruiyi Zhang, Daniel Du, Saurabh Mahindre, Sai Ashish Somayajula, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.02340)  

**Abstract**: Large Language Models (LLMs) are widely used for temporal prediction, but their reliance on pretraining data raises contamination concerns, as accurate predictions on pre-cutoff test data may reflect memorization rather than reasoning, leading to an overestimation of their generalization capability. With the recent emergence of prompting-based unlearning techniques, a natural question arises: Can LLMs be prompted to simulate an earlier knowledge cutoff? In this work, we investigate the capability of prompting to simulate earlier knowledge cutoff in LLMs. We construct three evaluation datasets to assess the extent to which LLMs can forget (1) direct factual knowledge, (2) semantic shifts, and (3) causally related knowledge. Results demonstrate that while prompt-based simulated knowledge cutoffs show effectiveness when directly queried with the information after that date, they struggle to induce forgetting when the forgotten content is not directly asked but causally related to the query. These findings highlight the need for more rigorous evaluation settings when applying LLMs for temporal prediction tasks. The full dataset and evaluation code are available at this https URL. 

---
# Evaluating Uncertainty Quantification Methods in Argumentative Large Language Models 

**Authors**: Kevin Zhou, Adam Dejl, Gabriel Freedman, Lihu Chen, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2510.02339)  

**Abstract**: Research in uncertainty quantification (UQ) for large language models (LLMs) is increasingly important towards guaranteeing the reliability of this groundbreaking technology. We explore the integration of LLM UQ methods in argumentative LLMs (ArgLLMs), an explainable LLM framework for decision-making based on computational argumentation in which UQ plays a critical role. We conduct experiments to evaluate ArgLLMs' performance on claim verification tasks when using different LLM UQ methods, inherently performing an assessment of the UQ methods' effectiveness. Moreover, the experimental procedure itself is a novel way of evaluating the effectiveness of UQ methods, especially when intricate and potentially contentious statements are present. Our results demonstrate that, despite its simplicity, direct prompting is an effective UQ strategy in ArgLLMs, outperforming considerably more complex approaches. 

---
# Optimizing Long-Form Clinical Text Generation with Claim-Based Rewards 

**Authors**: Samyak Jhaveri, Praphul Singh, Jangwon Kim, Tara Taghavi, Krishnaram Kenthapadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02338)  

**Abstract**: Automating clinical documentation with large language models requires precise alignment with priorities such as completeness and factual grounding. We present an evaluation-integrated reinforcement learning framework for long-form clinical text generation that couples Group Relative Policy Optimization (GRPO) with DocLens, a claim-level evaluator that provides deterministic, dialogue-grounded rewards. Our method directly optimizes factual grounding and completeness without training a separate reward model or relying on human-authored references. Empirically, the approach improves clinical note quality and reduces training cost via a simple reward-gating strategy. An independent GPT-5 qualitative evaluation further supports these gains, showing higher preference for GRPO outputs in factuality, completeness, and brevity, with fewer omissions and hallucinations. Because the benchmarks are relatively clean and the base model already well aligned, these improvements likely represent a conservative lower bound. The framework is scalable to real-world settings and can incorporate custom objectives such as guideline adherence or billing preferences. 

---
# CRACQ: A Multi-Dimensional Approach To Automated Document Assessment 

**Authors**: Ishak Soltani, Francisco Belo, Bernardo Tavares  

**Link**: [PDF](https://arxiv.org/pdf/2510.02337)  

**Abstract**: This paper presents CRACQ, a multi-dimensional evaluation framework tailored to evaluate documents across f i v e specific traits: Coherence, Rigor, Appropriateness, Completeness, and Quality. Building on insights from traitbased Automated Essay Scoring (AES), CRACQ expands its fo-cus beyond essays to encompass diverse forms of machine-generated text, providing a rubricdriven and interpretable methodology for automated evaluation. Unlike singlescore approaches, CRACQ integrates linguistic, semantic, and structural signals into a cumulative assessment, enabling both holistic and trait-level analysis. Trained on 500 synthetic grant pro-posals, CRACQ was benchmarked against an LLM-as-a-judge and further tested on both strong and weak real applications. Preliminary results in-dicate that CRACQ produces more stable and interpretable trait-level judgments than direct LLM evaluation, though challenges in reliability and domain scope remain 

---
# KurdSTS: The Kurdish Semantic Textual Similarity 

**Authors**: Abdulhady Abas Abdullah, Hadi Veisi, Hussein M. Al  

**Link**: [PDF](https://arxiv.org/pdf/2510.02336)  

**Abstract**: Semantic Textual Similarity (STS) measures the degree of meaning overlap between two texts and underpins many NLP tasks. While extensive resources exist for high-resource languages, low-resource languages such as Kurdish remain underserved. We present, to our knowledge, the first Kurdish STS dataset: 10,000 sentence pairs spanning formal and informal registers, each annotated for similarity. We benchmark Sentence-BERT, multilingual BERT, and other strong baselines, obtaining competitive results while highlighting challenges arising from Kurdish morphology, orthographic variation, and code-mixing. The dataset and baselines establish a reproducible evaluation suite and provide a strong starting point for future research on Kurdish semantics and low-resource NLP. 

---
# FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory 

**Authors**: Xiao-Wen Yang, Zihao Zhang, Jianuo Cao, Zhi Zhou, Zenan Li, Lan-Zhe Guo, Yuan Yao, Taolue Chen, Yu-Feng Li, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.02335)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable progress in formal theorem proving. Yet their ability to serve as practical assistants for mathematicians, filling in missing steps within complex proofs, remains underexplored. We identify this challenge as the task of subgoal completion, where an LLM must discharge short but nontrivial proof obligations left unresolved in a human-provided sketch. To study this problem, we introduce FormalML, a Lean 4 benchmark built from foundational theories of machine learning. Using a translation tactic that converts procedural proofs into declarative form, we extract 4937 problems spanning optimization and probability inequalities, with varying levels of difficulty. FormalML is the first subgoal completion benchmark to combine premise retrieval and complex research-level contexts. Evaluation of state-of-the-art provers highlights persistent limitations in accuracy and efficiency, underscoring the need for more capable LLM-based theorem provers for effective subgoal completion, 

---
# Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing 

**Authors**: Zhe Li, Wei Zhao, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.02334)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their deployment is frequently undermined by undesirable behaviors such as generating harmful content, factual inaccuracies, and societal biases. Diagnosing the root causes of these failures poses a critical challenge for AI safety. Existing attribution methods, particularly those based on parameter gradients, often fall short due to prohibitive noisy signals and computational complexity. In this work, we introduce a novel and efficient framework that diagnoses a range of undesirable LLM behaviors by analyzing representation and its gradients, which operates directly in the model's activation space to provide a semantically meaningful signal linking outputs to their training data. We systematically evaluate our method for tasks that include tracking harmful content, detecting backdoor poisoning, and identifying knowledge contamination. The results demonstrate that our approach not only excels at sample-level attribution but also enables fine-grained token-level analysis, precisely identifying the specific samples and phrases that causally influence model behavior. This work provides a powerful diagnostic tool to understand, audit, and ultimately mitigate the risks associated with LLMs. The code is available at this https URL. 

---
# Human Mobility Datasets Enriched With Contextual and Social Dimensions 

**Authors**: Chiara Pugliese, Francesco Lettich, Guido Rocchietti, Chiara Renso, Fabio Pinelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.02333)  

**Abstract**: In this resource paper, we present two publicly available datasets of semantically enriched human trajectories, together with the pipeline to build them. The trajectories are publicly available GPS traces retrieved from OpenStreetMap. Each dataset includes contextual layers such as stops, moves, points of interest (POIs), inferred transportation modes, and weather data. A novel semantic feature is the inclusion of synthetic, realistic social media posts generated by Large Language Models (LLMs), enabling multimodal and semantic mobility analysis. The datasets are available in both tabular and Resource Description Framework (RDF) formats, supporting semantic reasoning and FAIR data practices. They cover two structurally distinct, large cities: Paris and New York. Our open source reproducible pipeline allows for dataset customization, while the datasets support research tasks such as behavior modeling, mobility prediction, knowledge graph construction, and LLM-based applications. To our knowledge, our resource is the first to combine real-world movement, structured semantic enrichment, LLM-generated text, and semantic web compatibility in a reusable framework. 

---
# A High-Capacity and Secure Disambiguation Algorithm for Neural Linguistic Steganography 

**Authors**: Yapei Feng, Feng Jiang, Shanhao Wu, Hua Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2510.02332)  

**Abstract**: Neural linguistic steganography aims to embed information
into natural text while preserving statistical undetectability. A fundamental challenge in this ffeld stems from tokenization ambiguity in modern tokenizers, which can lead to catastrophic decoding failures. The recent method, SyncPool, addresses this ambiguity
by employing a coarse-grained synchronization mechanism over groups of ambiguous candidates. However, SyncPool sacriffces embedding capacity, as it utilizes the entire Shannon entropy of an ambiguous group solely for synchronization rather than for payload embedding. We propose a method named look-ahead Sync, which overcomes the capacity limitation of SyncPool while retaining its provable security guarantees. Our approach performs minimal synchronized sampling only on truly indistinguishable token sequences, while strategically preserving all other discernible paths to maximize embedding capacity. We provide theoretical proofs for the security of our method and analyze the gap between its achievable embedding capacity and the theoretical upper bound. Experiments on English (using Llama 3) and Chinese (using Qwen 2.5) benchmarks show that our method consistently approaches the theoretical capacity upper bound and signiffcantly outperforms SyncPool. The improvement in embedding rate exceeds 160% in English and 25% in Chinese, particularly in settings with larger candidate pools. This work represents a signiffcant step toward practical high-capacity provably secure linguistic steganography. 

---
# Synthetic Dialogue Generation for Interactive Conversational Elicitation & Recommendation (ICER) 

**Authors**: Moonkyung Ryu, Chih-Wei Hsu, Yinlam Chow, Mohammad Ghavamzadeh, Craig Boutilier  

**Link**: [PDF](https://arxiv.org/pdf/2510.02331)  

**Abstract**: While language models (LMs) offer great potential for conversational recommender systems (CRSs), the paucity of public CRS data makes fine-tuning LMs for CRSs challenging. In response, LMs as user simulators qua data generators can be used to train LM-based CRSs, but often lack behavioral consistency, generating utterance sequences inconsistent with those of any real user. To address this, we develop a methodology for generating natural dialogues that are consistent with a user's underlying state using behavior simulators together with LM-prompting. We illustrate our approach by generating a large, open-source CRS data set with both preference elicitation and example critiquing. Rater evaluation on some of these dialogues shows them to exhibit considerable consistency, factuality and naturalness. 

---
# EntropyLong: Effective Long-Context Training via Predictive Uncertainty 

**Authors**: Junlong Jia, Ziyang Chen, Xing Wu, Chaochen Gao, Zijia Lin, Debing Zhang, Songlin Hu, Binghui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.02330)  

**Abstract**: Training long-context language models to capture long-range dependencies requires specialized data construction. Current approaches, such as generic text concatenation or heuristic-based variants, frequently fail to guarantee genuine long-range dependencies. We propose EntropyLong, a novel data construction method that leverages predictive uncertainty to verify dependency quality. Our approach identifies high-entropy positions in documents, retrieves semantically relevant contexts from large corpora, and verifies their utility by assessing whether they reduce prediction entropy. This model-in-the-loop verification ensures each dependency represents measurable information gain rather than spurious correlation. We construct training samples with long-range dependencies by combining original documents with these verified contextual supplements. Using FineWebEdu and Cosmopedia, we generate a dataset of 128K-length sequences with verified dependencies. Models trained on this data demonstrate significant improvements on RULER benchmarks, particularly in tasks requiring distant information. Following instruction fine-tuning, our models also achieve substantial gains on LongBenchv2, demonstrating enhanced long-context understanding. Extensive ablation studies further validate the necessity and effectiveness of entropybased verification for long-context training. 

---
# SelfJudge: Faster Speculative Decoding via Self-Supervised Judge Verification 

**Authors**: Kanghoon Yoon, Minsub Kim, Sungjae Lee, Joonhyung Lee, Sunghyeon Woo, Yeonjun In, Se Jung Kwon, Chanyoung Park, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.02329)  

**Abstract**: Speculative decoding accelerates LLM inference by verifying candidate tokens from a draft model against a larger target model. Recent judge decoding boosts this process by relaxing verification criteria by accepting draft tokens that may exhibit minor discrepancies from target model output, but existing methods are restricted by their reliance on human annotations or tasks with verifiable ground truths, limiting generalizability across diverse NLP tasks. We propose SelfJudge, which trains judge verifiers via self-supervision of the target model. Our method measures semantic preservation by assessing whether token-substituted responses preserve the meaning of original responses, enabling automatic verifier training across diverse NLP tasks. Our experiments show SelfJudge achieves superior inference-accuracy trade-offs than judge decoding baselines, offering a broadly applicable solution for faster LLM inference. 

---
# AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering 

**Authors**: Ziqing Wang, Chengsheng Mao, Xiaole Wen, Yuan Luo, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.02328)  

**Abstract**: Medical Multimodal Large Language Models (Med-MLLMs) have shown great promise in medical visual question answering (Med-VQA). However, when deployed in low-resource settings where abundant labeled data are unavailable, existing Med-MLLMs commonly fail due to their medical reasoning capability bottlenecks: (i) the intrinsic reasoning bottleneck that ignores the details from the medical image; (ii) the extrinsic reasoning bottleneck that fails to incorporate specialized medical knowledge. To address those limitations, we propose AMANDA, a training-free agentic framework that performs medical knowledge augmentation via LLM agents. Specifically, our intrinsic medical knowledge augmentation focuses on coarse-to-fine question decomposition for comprehensive diagnosis, while extrinsic medical knowledge augmentation grounds the reasoning process via biomedical knowledge graph retrieval. Extensive experiments across eight Med-VQA benchmarks demonstrate substantial improvements in both zero-shot and few-shot Med-VQA settings. The code is available at this https URL. 

---
# KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI 

**Authors**: So Kuroki, Yotaro Kubo, Takuya Akiba, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02327)  

**Abstract**: Real-time speech-to-speech (S2S) models excel at generating natural, low-latency conversational responses but often lack deep knowledge and semantic understanding. Conversely, cascaded systems combining automatic speech recognition, a text-based Large Language Model (LLM), and text-to-speech synthesis offer superior knowledge representation at the cost of high latency, which disrupts the flow of natural interaction. This paper introduces a novel hybrid architecture that bridges the gap between these two paradigms. Our framework processes user speech through an S2S transformer for immediate responsiveness while concurrently relaying the query to a powerful back-end LLM. The LLM's text-based response is then injected in real time to guide the S2S model's speech generation, effectively infusing its output with rich knowledge without the full latency penalty of a cascaded system. We evaluated our method using a speech-synthesized variant of the MT-Bench benchmark that consists of multi-turn question-answering sessions. The results demonstrate that our system substantially outperforms a baseline S2S model in response correctness, approaching that of a cascaded system, while maintaining a latency on par with the baseline. 

---
# Hallucination-Resistant, Domain-Specific Research Assistant with Self-Evaluation and Vector-Grounded Retrieval 

**Authors**: Vivek Bhavsar, Joseph Ereifej, Aravanan Gurusami  

**Link**: [PDF](https://arxiv.org/pdf/2510.02326)  

**Abstract**: Large language models accelerate literature synthesis but can hallucinate and mis-cite, limiting their usefulness in expert workflows. We present RA-FSM (Research Assistant - Finite State Machine), a modular GPT-based research assistant that wraps generation in a finite-state control loop: Relevance -> Confidence -> Knowledge. The system is grounded in vector retrieval and a deterministic citation pipeline. The controller filters out-of-scope queries, scores answerability, decomposes questions, and triggers retrieval only when needed, and emits answers with confidence labels and in-corpus, de-duplicated references. A ranked-tier ingestion workflow constructs a domain knowledge base from journals, conferences, indices, preprints, and patents, writing both to a dense vector index and to a relational store of normalized metrics. We implement the system for photonics and evaluate it on six task categories: analytical reasoning, numerical analysis, methodological critique, comparative synthesis, factual extraction, and application design. In blinded A/B reviews, domain experts prefer RA-FSM to both a strong Notebook LM (NLM) and a vanilla Default GPT API call single-pass baseline, citing stronger boundary-condition handling and more defensible evidence use. Coverage and novelty analyses indicate that RA-FSM explores beyond the NLM while incurring tunable latency and cost overheads. The design emphasizes transparent, well-cited answers for high-stakes technical work and is generalizable to other scientific domains. 

---
# Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning 

**Authors**: Wannan Yang, Xinchi Qiu, Lei Yu, Yuchen Zhang, Oliver Aobo Yang, Narine Kokhlikyan, Nicola Cancedda, Diego Garcia-Olano  

**Link**: [PDF](https://arxiv.org/pdf/2510.02324)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities but often hallucinate, confidently providing incorrect answers instead of admitting ignorance. Prior work has shown that models encode linear representations of their own knowledge and that activation steering can reduce hallucinations. These approaches, however, require real-time monitoring and intervention during inference. We introduce Contrastive Activation Steering for Amortized Learning (CASAL), an efficient algorithm that connects interpretability with amortized optimization. CASAL directly bakes the benefits of activation steering into model's weights. Once trained, LLMs answer questions they know while abstaining from answering those they do not. CASAL's light-weight design requires training only a submodule of a single transformer layer and yet reduces hallucination by 30%-40% across multiple short-form QA benchmarks. CASAL is 30x more compute-efficient and 20x more data-efficient than strong LoRA-based baselines such as SFT and DPO, boosting its practical applicability in data scarce domains. Importantly, CASAL also generalizes effectively to out-of-distribution (OOD) domains. We showcase CASAL's flexibility in mitigating hallucinations in both text-only and vision-language models. To our knowledge, CASAL is the first steering-based training method that has been shown to be effective for both dense and Mixture-of-Experts (MoE) models. CASAL represents a promising step forward for applying interpretability-inspired method for practical deployment in production systems. 

---
# Low-probability Tokens Sustain Exploration in Reinforcement Learning with Verifiable Reward 

**Authors**: Guanhua Huang, Tingqiang Xu, Mingze Wang, Qi Yi, Xue Gong, Siheng Li, Ruibin Xiong, Kejiao Li, Yuhao Jiang, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03222)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has propelled Large Language Models in complex reasoning, yet its scalability is often hindered by a training bottleneck where performance plateaus as policy entropy collapses, signaling a loss of exploration. Previous methods typically address this by maintaining high policy entropy, yet the precise mechanisms that govern meaningful exploration have remained underexplored. Our analysis suggests that an unselective focus on entropy risks amplifying irrelevant tokens and destabilizing training. This paper investigates the exploration dynamics within RLVR and identifies a key issue: the gradual elimination of valuable low-probability exploratory tokens, which we term \textbf{\textit{reasoning sparks}}. We find that while abundant in pre-trained models, these sparks are systematically extinguished during RLVR due to over-penalization, leading to a degeneracy in exploration. To address this, we introduce Low-probability Regularization (Lp-Reg). Its core mechanism regularizes the policy towards a heuristic proxy distribution. This proxy is constructed by filtering out presumed noise tokens and re-normalizing the distribution over the remaining candidates. The result is a less-noisy proxy where the probability of \textit{reasoning sparks} is amplified, which then serves as a soft regularization target to shield these valuable tokens from elimination via KL divergence. Experiments show that Lp-Reg enables stable on-policy training for around 1,000 steps, a regime where baseline entropy-control methods collapse. This sustained exploration leads to state-of-the-art performance, achieving a $60.17\%$ average accuracy on five math benchmarks, an improvement of $2.66\%$ over prior methods. Code is available at this https URL. 

---
# Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner 

**Authors**: Cai Zhou, Chenxiao Yang, Yi Hu, Chenyu Wang, Chubin Zhang, Muhan Zhang, Lester Mackey, Tommi Jaakkola, Stephen Bates, Dinghuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03206)  

**Abstract**: Diffusion language models, especially masked discrete diffusion models, have achieved great success recently. While there are some theoretical and primary empirical results showing the advantages of latent reasoning with looped transformers or continuous chain-of-thoughts, continuous diffusion models typically underperform their discrete counterparts. In this paper, we argue that diffusion language models do not necessarily need to be in the discrete space. In particular, we prove that continuous diffusion models have stronger expressivity than discrete diffusions and looped transformers. We attribute the contradiction between the theoretical expressiveness and empirical performance to their practical trainability: while continuous diffusion provides intermediate supervision that looped transformers lack, they introduce additional difficulty decoding tokens into the discrete token space from the continuous representation space. We therefore propose Coevolutionary Continuous Discrete Diffusion (CCDD), which defines a joint multimodal diffusion process on the union of a continuous representation space and a discrete token space, leveraging a single model to simultaneously denoise in the joint space. By combining two modalities, CCDD is expressive with rich semantics in the latent space, as well as good trainability and sample quality with the help of explicit discrete tokens. We also propose effective architectures and advanced training/sampling techniques for CCDD, which reveals strong empirical performance in extensive language modeling experiments on real-world tasks. 

---
# Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning 

**Authors**: Yilun Hao, Yongchao Chen, Chuchu Fan, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03182)  

**Abstract**: Vision Language Models (VLMs) show strong potential for visual planning but struggle with precise spatial and long-horizon reasoning. In contrast, Planning Domain Definition Language (PDDL) planners excel at long-horizon formal planning, but cannot interpret visual inputs. Recent works combine these complementary advantages by enabling VLMs to turn visual planning problems into PDDL files for formal planning. However, while VLMs can generate PDDL problem files satisfactorily, they struggle to accurately generate the PDDL domain files, which describe all the planning rules. As a result, prior methods rely on human experts to predefine domain files or on constant environment access for refinement. We propose VLMFP, a Dual-VLM-guided framework that can autonomously generate both PDDL problem and domain files for formal visual planning. VLMFP introduces two VLMs to ensure reliable PDDL file generation: A SimVLM that simulates action consequences based on input rule descriptions, and a GenVLM that generates and iteratively refines PDDL files by comparing the PDDL and SimVLM execution results. VLMFP unleashes multiple levels of generalizability: The same generated PDDL domain file works for all the different instances under the same problem, and VLMs generalize to different problems with varied appearances and rules. We evaluate VLMFP with 6 grid-world domains and test its generalization to unseen instances, appearance, and game rules. On average, SimVLM accurately describes 95.5%, 82.6% of scenarios, simulates 85.5%, 87.8% of action sequence, and judges 82.4%, 85.6% goal reaching for seen and unseen appearances, respectively. With the guidance of SimVLM, VLMFP can generate PDDL files to reach 70.0%, 54.1% valid plans for unseen instances in seen and unseen appearances, respectively. Project page: this https URL. 

---
# When Names Disappear: Revealing What LLMs Actually Understand About Code 

**Authors**: Cuong Chi Le, Minh V.T. Pham, Cuong Duc Van, Hoang N. Phan, Huy N. Phan, Tien N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03178)  

**Abstract**: Large Language Models (LLMs) achieve strong results on code tasks, but how they derive program meaning remains unclear. We argue that code communicates through two channels: structural semantics, which define formal behavior, and human-interpretable naming, which conveys intent. Removing the naming channel severely degrades intent-level tasks such as summarization, where models regress to line-by-line descriptions. Surprisingly, we also observe consistent reductions on execution tasks that should depend only on structure, revealing that current benchmarks reward memorization of naming patterns rather than genuine semantic reasoning. To disentangle these effects, we introduce a suite of semantics-preserving obfuscations and show that they expose identifier leakage across both summarization and execution. Building on these insights, we release ClassEval-Obf, an obfuscation-enhanced benchmark that systematically suppresses naming cues while preserving behavior. Our results demonstrate that ClassEval-Obf reduces inflated performance gaps, weakens memorization shortcuts, and provides a more reliable basis for assessing LLMs' code understanding and generalization. 

---
# Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents 

**Authors**: Wonjoong Kim, Sangwu Park, Yeonjun In, Sein Kim, Dongha Lee, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.02837)  

**Abstract**: Although recent tool-augmented benchmarks incorporate complex user requests and diverse tools, the evaluation methods for most of them remain limited to answer matching. However, as the number of steps required to resolve a user request increases, a proper evaluation of an agent's performance must go beyond the final answer to also assess the problem-solving trajectory, including previously ignored aspects such as efficiency, hallucination, and adaptivity. The most straightforward method for evaluating these aspects is to compare an agent's trajectory with the ground-truth trajectory, but this approach is fundamentally limited since annotating all valid ground-truth trajectories is prohibitively expensive. However, a simple LLM-based evaluator struggles to assess trajectories in detail without ground truth. To effectively evaluate the agents in this manner, we introduce TRACE, a framework for the multi-dimensional evaluation of tool-augmented LLM agent performance. By incorporating an evidence bank, which accumulates knowledge gathered from preceding reasoning steps, TRACE enables a multi-faceted analysis and evaluation of an agent's reasoning trajectory effectively. To validate our framework, we develop a new meta-evaluation dataset by augmenting existing benchmarks with diverse and flawed trajectories, each labeled with multi-faceted performance scores. Our results confirm that TRACE accurately evaluates these complex behaviors in a scalable and cost-effective manner, even with small open-source LLMs. Furthermore, we apply our method to evaluate the trajectories that agents produce while solving tool-augmented tasks, presenting previously unreported observations and their corresponding insights. 

---
# NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning 

**Authors**: Yulong Zhang, Li Wang, Wei Du, Peilin Li, Yuqin Dai Zhiyuan Zhao, Lingyong Fang, Ziniu Liu, Ru Zhang, Huijia Zhu, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02816)  

**Abstract**: Verifying multi-step reasoning in large language models is difficult due to imprecise error localization and high token costs. Existing methods either assess entire reasoning chains, suffering attention dilution, or rely on expensive multi-sampling. We introduce Node-wise Consistency Verification (NCV), a training-free framework that recasts verification as lightweight binary consistency checks at the node level. By decomposing the chain of thought into interconnected verification nodes, NCV precisely localizes errors and avoids unnecessary long-form generation. Experiments demonstrate that our approach enhances interpretability and efficiency, presenting a scalable solution for reliable LLM reasoning verification. On public datasets, NCV achieves a 10\% to 25\% improvement in F1 scores over baselines while utilizing $6\times$~$58\times$ fewer tokens than traditional methods like CoT-based verifiers. 

---
# Pareto-optimal Non-uniform Language Generation 

**Authors**: Moses Charikar, Chirag Pabbaraju  

**Link**: [PDF](https://arxiv.org/pdf/2510.02795)  

**Abstract**: Kleinberg and Mullainathan (2024) recently proposed an interesting model for language generation in the limit: Given a countable collection of languages, and an adversary enumerating the strings of some language $L$ from the collection, the objective is to generate new strings from the target language, such that all strings generated beyond some finite time are valid. Li, Raman and Tewari (2024) and Charikar and Pabbaraju (2024) showed strong non-uniform generation guarantees in this model, giving algorithms that generate new valid strings from $L$ after seeing a number of distinct input strings $t(L)$ that depends only on $L$ (and the collection), but not the enumeration order. However, for both these works, the language-wise generation times $t(L)$ of the algorithm can be strictly sub-optimal.
In this work, we study Pareto-optimality of non-uniform language generation in the limit. We propose an algorithm, whose generation times $t^\star(L)$ are (almost) Pareto-optimal: any other algorithm whose generation time for some language $L$ is strictly smaller than $t^\star(L)$, must satisfy that its generation time for some other language $L'$ is strictly worse than $t^\star(L')$. Pareto-optimality is essentially the best that one can achieve for non-uniform generation. Our algorithmic framework conveniently adapts to further give Pareto-optimal non-uniform generation algorithms in the practically motivated settings of noisy as well as representative generation. 

---
# MaskCD: Mitigating LVLM Hallucinations by Image Head Masked Contrastive Decoding 

**Authors**: Jingyuan Deng, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02790)  

**Abstract**: Large vision-language models (LVLMs) have shown remarkable performance in visual-language understanding for downstream multimodal tasks. While their capabilities are improving, problems emerge simultaneously. Among those problems, the hallucinations have attracted much attention, which stands for the phenomenon where LVLMs generate contradictory content to their input visual and text contents. Many approaches have been proposed to deal with this issue, such as contrastive decoding and attention manipulation. However, contrastive decoding methods struggle in constructing appropriate contrastive samples, and attention manipulation methods are highly sensitive, lacking stability. In this work, we propose image head Masked Contrastive Decoding (MaskCD). Our approach utilizes the "image heads" in LVLMs, masking them to construct contrastive samples for contrastive decoding. We evaluated MaskCD on LLaVA-1.5-7b and Qwen-VL-7b, using various benchmarks such as CHAIR, POPE, AMBER and MME. The results demonstrate that MaskCD effectively alleviates the phenomenon of hallucinations and retains the general capabilities of LVLMs. Corresponding resources could be found at: this https URL . 

---
# A Granular Study of Safety Pretraining under Model Abliteration 

**Authors**: Shashank Agnihotri, Jonas Jakubassa, Priyam Dey, Sachin Goyal, Bernt Schiele, Venkatesh Babu Radhakrishnan, Margret Keuper  

**Link**: [PDF](https://arxiv.org/pdf/2510.02768)  

**Abstract**: Open-weight LLMs can be modified at inference time with simple activation edits, which raises a practical question for safety: do common safety interventions like refusal training or metatag training survive such edits? We study model abliteration, a lightweight projection technique designed to remove refusal-sensitive directions, and conduct a controlled evaluation across a granular sequence of Safety Pretraining checkpoints for SmolLM2-1.7B, alongside widely used open baselines. For each of 20 systems, original and abliterated, we issue 100 prompts with balanced harmful and harmless cases, classify responses as **Refusal** or **Non-Refusal** using multiple judges, and validate judge fidelity on a small human-labeled subset. We also probe whether models can identify refusal in their own outputs. Our study produces a checkpoint-level characterization of which data-centric safety components remain robust under abliteration, quantifies how judge selection influences evaluation outcomes, and outlines a practical protocol for integrating inference-time edits into safety assessments. Code: this https URL. 

---
# Hyperparameter Loss Surfaces Are Simple Near their Optima 

**Authors**: Nicholas Lourie, He He, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2510.02721)  

**Abstract**: Hyperparameters greatly impact models' capabilities; however, modern models are too large for extensive search. Instead, researchers design recipes that train well across scales based on their understanding of the hyperparameters. Despite this importance, few tools exist for understanding the hyperparameter loss surface. We discover novel structure in it and propose a new theory yielding such tools. The loss surface is complex, but as you approach the optimum simple structure emerges. It becomes characterized by a few basic features, like its effective dimension and the best possible loss. To uncover this asymptotic regime, we develop a novel technique based on random search. Within this regime, the best scores from random search take on a new distribution we discover. Its parameters are exactly the features defining the loss surface in the asymptotic regime. From these features, we derive a new asymptotic law for random search that can explain and extrapolate its convergence. These new tools enable new analyses, such as confidence intervals for the best possible performance or determining the effective number of hyperparameters. We make these tools available at this https URL . 

---
# Less LLM, More Documents: Searching for Improved RAG 

**Authors**: Jingjie Ning, Yibo Kong, Yunfan Long, Jamie Callan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02657)  

**Abstract**: Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators improves accuracy, it also raises cost and limits deployability. We explore an orthogonal axis: enlarging the retriever's corpus to reduce reliance on large LLMs. Experimental results show that corpus scaling consistently strengthens RAG and can often serve as a substitute for increasing model size, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and large models benefit less. Our analysis shows that improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. These findings establish a principled corpus-generator trade-off: investing in larger corpora offers an effective path to stronger RAG, often comparable to enlarging the LLM itself. 

---
# HyperAdaLoRA: Accelerating LoRA Rank Allocation During Training via Hypernetworks without Sacrificing Performance 

**Authors**: Hao Zhang, Zhenjia Li, Runfeng Bao, Yifan Gao, Xi Xiao, Bo Huang, Yuhang Wu, Tianyang Wang, Hao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02630)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT), especially Low-Rank Adaptation (LoRA), has emerged as a promising approach to fine-tuning large language models(LLMs) while reducing computational and memory overhead. However, LoRA assumes a uniform rank \textit{r} for each incremental matrix, not accounting for the varying significance of weight matrices across different modules and layers. AdaLoRA leverages Singular Value Decomposition (SVD) to parameterize updates and employs pruning of singular values to introduce dynamic rank allocation, thereby enhancing adaptability. However, during the training process, it often encounters issues of slow convergence speed and high computational overhead. To address this issue, we propose HyperAdaLoRA, a novel framework that accelerates the convergence of AdaLoRA by leveraging a hypernetwork. Instead of directly optimizing the components of Singular Value Decomposition $(P, \Lambda, Q)$, HyperAdaLoRA employs a hypernetwork based on attention mechanisms to dynamically generate these parameters. By pruning the outputs of the hypernetwork that generates the singular values, dynamic rank allocation is achieved. Comprehensive experiments on various datasets and models demonstrate that our method achieves faster convergence without sacrificing performance. Additionally, further extension experiments on other LoRA-based approaches validate the broad applicability of our method. 

---
# On the Role of Temperature Sampling in Test-Time Scaling 

**Authors**: Yuheng Wu, Azalia Mirhoseini, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2510.02611)  

**Abstract**: Large language models (LLMs) can improve reasoning at inference time through test-time scaling (TTS), where multiple reasoning traces are generated and the best one is selected. Prior work shows that increasing the number of samples K steadily improves accuracy. In this paper, we demonstrate that this trend does not hold indefinitely: at large K, further scaling yields no gains, and certain hard questions remain unsolved regardless of the number of traces. Interestingly, we find that different sampling temperatures solve different subsets of problems, implying that single-temperature scaling explores only part of a model's potential. We therefore propose scaling along the temperature dimension, which enlarges the reasoning boundary of LLMs. Averaged over Qwen3 (0.6B, 1.7B, 4B, 8B) and five representative reasoning benchmarks (AIME 2024/2025, MATH500, LiveCodeBench, Hi-ToM), temperature scaling yields an additional 7.3 points over single-temperature TTS. Temperature scaling also enables base models to reach performance comparable to reinforcement learning (RL)-trained counterparts, without additional post-training. We further provide a comprehensive analysis of this phenomenon and design a multi-temperature voting method that reduces the overhead of temperature scaling. Overall, our findings suggest that TTS is more powerful than previously thought, and that temperature scaling offers a simple and effective way to unlock the latent potential of base models. 

---
# How Confident are Video Models? Empowering Video Models to Express their Uncertainty 

**Authors**: Zhiting Mei, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02571)  

**Abstract**: Generative video models demonstrate impressive text-to-video capabilities, spurring widespread adoption in many real-world applications. However, like large language models (LLMs), video generation models tend to hallucinate, producing plausible videos even when they are factually wrong. Although uncertainty quantification (UQ) of LLMs has been extensively studied in prior work, no UQ method for video models exists, raising critical safety concerns. To our knowledge, this paper represents the first work towards quantifying the uncertainty of video models. We present a framework for uncertainty quantification of generative video models, consisting of: (i) a metric for evaluating the calibration of video models based on robust rank correlation estimation with no stringent modeling assumptions; (ii) a black-box UQ method for video models (termed S-QUBED), which leverages latent modeling to rigorously decompose predictive uncertainty into its aleatoric and epistemic components; and (iii) a UQ dataset to facilitate benchmarking calibration in video models. By conditioning the generation task in the latent space, we disentangle uncertainty arising due to vague task specifications from that arising from lack of knowledge. Through extensive experiments on benchmark video datasets, we demonstrate that S-QUBED computes calibrated total uncertainty estimates that are negatively correlated with the task accuracy and effectively computes the aleatoric and epistemic constituents. 

---
# Beyond Imitation: Recovering Dense Rewards from Demonstrations 

**Authors**: Jiangnan Li, Thuy-Trang Vu, Ehsan Abbasnejad, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2510.02493)  

**Abstract**: Conventionally, supervised fine-tuning (SFT) is treated as a simple imitation learning process that only trains a policy to imitate expert behavior on demonstration datasets. In this work, we challenge this view by establishing a fundamental equivalence between SFT and Inverse Reinforcement Learning. We prove that the SFT objective is a special case of Inverse Q-Learning, which implies that the SFT process does not just learn a policy, but also an implicit, dense, token-level reward model that explains the expert demonstrations. We then show how to recover this dense reward signal directly from the SFT model by formulating a baseline-relative reward function. The availability of such a dense reward model offers numerous benefits, providing granular credit assignment for each token generated. We demonstrate one key application by using these recovered rewards to further improve the policy with reinforcement learning. Our method, Dense-Path REINFORCE, consistently outperforms the original SFT models on instruction-following benchmarks. This work reframes SFT not merely as policy imitation but as a powerful reward learning mechanism, opening new possibilities for leveraging expert demonstrations. 

---
# Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework 

**Authors**: Nii Osae Osae Dade, Moinul Hossain Rahat  

**Link**: [PDF](https://arxiv.org/pdf/2510.02483)  

**Abstract**: Training Large Language Models (LLMs) is plagued by long training times and massive energy consumption, with modern models requiring months of computation and gigawatt-hours of electricity. In light of these challenges,we introduce Litespark, a novel pre-training framework that addresses these inefficiencies through targeted optimizations to transformer attention and MLP layers. Our approach combines architectural improvements with algorithmic enhancements to maximize Model FLOPs Utilization (MFU) while maintaining compatibility with standard transformer implementations. Comprehensive benchmarking on 3B and 30B parameter Llama models using the SlimPajama-627B dataset demonstrates substantial performance gains: 2x-6x training throughput improvement and $55\%-83$% energy consumption reduction across multi-node H200 GPU clusters. These optimizations are model- and hardware-agnostic, enabling broad applicability across transformer architectures and extending to post-training phases including supervised fine-tuning and direct preference optimization. 

---
# SIMSplat: Predictive Driving Scene Editing with Language-aligned 4D Gaussian Splatting 

**Authors**: Sung-Yeon Park, Adam Lee, Juanwu Lu, Can Cui, Luyang Jiang, Rohit Gupta, Kyungtae Han, Ahmadreza Moradipari, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02469)  

**Abstract**: Driving scene manipulation with sensor data is emerging as a promising alternative to traditional virtual driving simulators. However, existing frameworks struggle to generate realistic scenarios efficiently due to limited editing capabilities. To address these challenges, we present SIMSplat, a predictive driving scene editor with language-aligned Gaussian splatting. As a language-controlled editor, SIMSplat enables intuitive manipulation using natural language prompts. By aligning language with Gaussian-reconstructed scenes, it further supports direct querying of road objects, allowing precise and flexible editing. Our method provides detailed object-level editing, including adding new objects and modifying the trajectories of both vehicles and pedestrians, while also incorporating predictive path refinement through multi-agent motion prediction to generate realistic interactions among all agents in the scene. Experiments on the Waymo dataset demonstrate SIMSplat's extensive editing capabilities and adaptability across a wide range of scenarios. Project page: this https URL 

---
# How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models 

**Authors**: Parth Asawa, Alan Zhu, Matei Zaharia, Alexandros G. Dimakis, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2510.02453)  

**Abstract**: Foundation models are increasingly deployed as black-box services, where model weights cannot be modified and customization is limited to prompting. While static prompt optimization has shown promise, it produces a single fixed prompt that fails to adapt to different inputs, users, or environments. We introduce Advisor Models, lightweight parametric policies trained with reinforcement learning to reactively issue natural language steering instructions in-context to black-box models. The advisor is a second small model that sits between the input and the model, shaping behavior on a per-instance basis using reward signals from the environment. Across multiple domains involving reasoning and personalization, we show that Advisor Models outperform static prompt optimizers, discovering environment dynamics and improving downstream task performance. We also demonstrate the generalizability of advisors by transferring them across black-box models, as well as the framework's ability to achieve specialization while retaining robustness to out-of-distribution inputs. Viewed more broadly, Advisor Models provide a learnable interface to black-box systems where the advisor acts as a parametric, environment-specific memory. We argue that dynamic optimization of black-box models via Advisor Models is a promising direction for enabling personalization and environment-adaptable AI with frontier-level capabilities. 

---
# CATMark: A Context-Aware Thresholding Framework for Robust Cross-Task Watermarking in Large Language Models 

**Authors**: Yu Zhang, Shuliang Liu, Xu Yang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02342)  

**Abstract**: Watermarking algorithms for Large Language Models (LLMs) effectively identify machine-generated content by embedding and detecting hidden statistical features in text. However, such embedding leads to a decline in text quality, especially in low-entropy scenarios where performance needs improvement. Existing methods that rely on entropy thresholds often require significant computational resources for tuning and demonstrate poor adaptability to unknown or cross-task generation scenarios. We propose \textbf{C}ontext-\textbf{A}ware \textbf{T}hreshold watermarking ($\myalgo$), a novel framework that dynamically adjusts watermarking intensity based on real-time semantic context. $\myalgo$ partitions text generation into semantic states using logits clustering, establishing context-aware entropy thresholds that preserve fidelity in structured content while embedding robust watermarks. Crucially, it requires no pre-defined thresholds or task-specific tuning. Experiments show $\myalgo$ improves text quality in cross-tasks without sacrificing detection accuracy. 

---
# SpeechCT-CLIP: Distilling Text-Image Knowledge to Speech for Voice-Native Multimodal CT Analysis 

**Authors**: Lukas Buess, Jan Geier, David Bani-Harouni, Chantal Pellegrini, Matthias Keicher, Paula Andrea Perez-Toro, Nassir Navab, Andreas Maier, Tomas Arias-Vergara  

**Link**: [PDF](https://arxiv.org/pdf/2510.02322)  

**Abstract**: Spoken communication plays a central role in clinical workflows. In radiology, for example, most reports are created through dictation. Yet, nearly all medical AI systems rely exclusively on written text. In this work, we address this gap by exploring the feasibility of learning visual-language representations directly from spoken radiology reports. Specifically, we synthesize a large-scale dataset (Speech-RATE) of spoken radiology reports and train SpeechCT-CLIP, a contrastive model that aligns speech and 3D CT volumes in a shared representation space. While naive speech-based models underperform compared to text-trained counterparts, we show that knowledge distillation from a pretrained text-image CLIP model effectively transfers semantic alignment capabilities from text to speech, substantially narrowing this gap. Experiments demonstrate improved zero-shot classification F1 from 0.623 to 0.705, recovering 88% of the performance difference, and strong retrieval results without requiring text at inference. These findings highlight speech as a practical alternative to text in multimodal pretraining and open the door to voice-driven diagnostic support tools in clinical practice. 

---
# WEE-Therapy: A Mixture of Weak Encoders Framework for Psychological Counseling Dialogue Analysis 

**Authors**: Yongqi Kang, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.02320)  

**Abstract**: The advancement of computational psychology requires AI tools capable of deeply understanding counseling dialogues. Existing audio language models (AudioLLMs) often rely on single speech encoders pre-trained on general data, struggling to capture domain-specific features like complex emotions and professional techniques. To address this, we propose WEE-Therapy, a multi-task AudioLLM incorporating a Weak Encoder Ensemble (WEE) mechanism. This supplements a powerful base encoder with a pool of lightweight, specialized encoders. A novel dual-routing strategy combines stable, data-independent domain knowledge with dynamic, data-dependent expert selection. Evaluated on emotion recognition, technique classification, risk detection, and summarization, WEE-Therapy achieves significant performance gains across all tasks with minimal parameter overhead, demonstrating strong potential for AI-assisted clinical analysis. 

---
# Modeling the Attack: Detecting AI-Generated Text by Quantifying Adversarial Perturbations 

**Authors**: Lekkala Sai Teja, Annepaka Yadagiri, Sangam Sai Anish, Siva Gopala Krishna Nuthakki, Partha Pakray  

**Link**: [PDF](https://arxiv.org/pdf/2510.02319)  

**Abstract**: The growth of highly advanced Large Language Models (LLMs) constitutes a huge dual-use problem, making it necessary to create dependable AI-generated text detection systems. Modern detectors are notoriously vulnerable to adversarial attacks, with paraphrasing standing out as an effective evasion technique that foils statistical detection. This paper presents a comparative study of adversarial robustness, first by quantifying the limitations of standard adversarial training and then by introducing a novel, significantly more resilient detection framework: Perturbation-Invariant Feature Engineering (PIFE), a framework that enhances detection by first transforming input text into a standardized form using a multi-stage normalization pipeline, it then quantifies the transformation's magnitude using metrics like Levenshtein distance and semantic similarity, feeding these signals directly to the classifier. We evaluate both a conventionally hardened Transformer and our PIFE-augmented model against a hierarchical taxonomy of character-, word-, and sentence-level attacks. Our findings first confirm that conventional adversarial training, while resilient to syntactic noise, fails against semantic attacks, an effect we term "semantic evasion threshold", where its True Positive Rate at a strict 1% False Positive Rate plummets to 48.8%. In stark contrast, our PIFE model, which explicitly engineers features from the discrepancy between a text and its canonical form, overcomes this limitation. It maintains a remarkable 82.6% TPR under the same conditions, effectively neutralizing the most sophisticated semantic attacks. This superior performance demonstrates that explicitly modeling perturbation artifacts, rather than merely training on them, is a more promising path toward achieving genuine robustness in the adversarial arms race. 

---
