# Generalist Foundation Models Are Not Clinical Enough for Hospital Operations 

**Authors**: Lavender Y. Jiang, Angelica Chen, Xu Han, Xujin Chris Liu, Radhika Dua, Kevin Eaton, Frederick Wolff, Robert Steele, Jeff Zhang, Anton Alyakin, Qingkai Pan, Yanbing Chen, Karl L. Sangwon, Daniel A. Alber, Jaden Stryker, Jin Vivian Lee, Yindalon Aphinyanaphongs, Kyunghyun Cho, Eric Karl Oermann  

**Link**: [PDF](https://arxiv.org/pdf/2511.13703)  

**Abstract**: Hospitals and healthcare systems rely on operational decisions that determine patient flow, cost, and quality of care. Despite strong performance on medical knowledge and conversational benchmarks, foundation models trained on general text may lack the specialized knowledge required for these operational decisions. We introduce Lang1, a family of models (100M-7B parameters) pretrained on a specialized corpus blending 80B clinical tokens from NYU Langone Health's EHRs and 627B tokens from the internet. To rigorously evaluate Lang1 in real-world settings, we developed the REalistic Medical Evaluation (ReMedE), a benchmark derived from 668,331 EHR notes that evaluates five critical tasks: 30-day readmission prediction, 30-day mortality prediction, length of stay, comorbidity coding, and predicting insurance claims denial. In zero-shot settings, both general-purpose and specialized models underperform on four of five tasks (36.6%-71.7% AUROC), with mortality prediction being an exception. After finetuning, Lang1-1B outperforms finetuned generalist models up to 70x larger and zero-shot models up to 671x larger, improving AUROC by 3.64%-6.75% and 1.66%-23.66% respectively. We also observed cross-task scaling with joint finetuning on multiple tasks leading to improvement on other tasks. Lang1-1B effectively transfers to out-of-distribution settings, including other clinical tasks and an external health system. Our findings suggest that predictive capabilities for hospital operations require explicit supervised finetuning, and that this finetuning process is made more efficient by in-domain pretraining on EHR. Our findings support the emerging view that specialized LLMs can compete with generalist models in specialized tasks, and show that effective healthcare systems AI requires the combination of in-domain pretraining, supervised finetuning, and real-world evaluation beyond proxy benchmarks. 

---
# Crossing Borders: A Multimodal Challenge for Indian Poetry Translation and Image Generation 

**Authors**: Sofia Jamil, Kotla Sai Charan, Sriparna Saha, Koustava Goswami, Joseph K J  

**Link**: [PDF](https://arxiv.org/pdf/2511.13689)  

**Abstract**: Indian poetry, known for its linguistic complexity and deep cultural resonance, has a rich and varied heritage spanning thousands of years. However, its layered meanings, cultural allusions, and sophisticated grammatical constructions often pose challenges for comprehension, especially for non-native speakers or readers unfamiliar with its context and language. Despite its cultural significance, existing works on poetry have largely overlooked Indian language poems. In this paper, we propose the Translation and Image Generation (TAI) framework, leveraging Large Language Models (LLMs) and Latent Diffusion Models through appropriate prompt tuning. Our framework supports the United Nations Sustainable Development Goals of Quality Education (SDG 4) and Reduced Inequalities (SDG 10) by enhancing the accessibility of culturally rich Indian-language poetry to a global audience. It includes (1) a translation module that uses an Odds Ratio Preference Alignment Algorithm to accurately translate morphologically rich poetry into English, and (2) an image generation module that employs a semantic graph to capture tokens, dependencies, and semantic relationships between metaphors and their meanings, to create visually meaningful representations of Indian poems. Our comprehensive experimental evaluation, including both human and quantitative assessments, demonstrates the superiority of TAI Diffusion in poem image generation tasks, outperforming strong baselines. To further address the scarcity of resources for Indian-language poetry, we introduce the Morphologically Rich Indian Language Poems MorphoVerse Dataset, comprising 1,570 poems across 21 low-resource Indian languages. By addressing the gap in poetry translation and visual comprehension, this work aims to broaden accessibility and enrich the reader's experience. 

---
# Why is "Chicago" Predictive of Deceptive Reviews? Using LLMs to Discover Language Phenomena from Lexical Cues 

**Authors**: Jiaming Qu, Mengtian Guo, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13658)  

**Abstract**: Deceptive reviews mislead consumers, harm businesses, and undermine trust in online marketplaces. Machine learning classifiers can learn from large amounts of training examples to effectively distinguish deceptive reviews from genuine ones. However, the distinguishing features learned by these classifiers are often subtle, fragmented, and difficult for humans to interpret. In this work, we explore using large language models (LLMs) to translate machine-learned lexical cues into human-understandable language phenomena that can differentiate deceptive reviews from genuine ones. We show that language phenomena obtained in this manner are empirically grounded in data, generalizable across similar domains, and more predictive than phenomena either in LLMs' prior knowledge or obtained through in-context learning. These language phenomena have the potential to aid people in critically assessing the credibility of online reviews in environments where deception detection classifiers are unavailable. 

---
# Omni Memory System for Personalized, Long Horizon, Self-Evolving Agents 

**Authors**: Piaohong Wang, Motong Tian, Jiaxian Li, Yuan Liang, Yuqing Wang, Qianben Chen, Tiannan Wang, Zhicong Lu, Jiawei Ma, Yuchen Eleanor Jiang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.13593)  

**Abstract**: Recent advancements in LLM-powered agents have demonstrated significant potential in generating human-like responses; however, they continue to face challenges in maintaining long-term interactions within complex environments, primarily due to limitations in contextual consistency and dynamic personalization. Existing memory systems often depend on semantic grouping prior to retrieval, which can overlook semantically irrelevant yet critical user information and introduce retrieval noise. In this report, we propose the initial design of O-Mem, a novel memory framework based on active user profiling that dynamically extracts and updates user characteristics and event records from their proactive interactions with agents. O-Mem supports hierarchical retrieval of persona attributes and topic-related context, enabling more adaptive and coherent personalized responses. O-Mem achieves 51.76% on the public LoCoMo benchmark, a nearly 3% improvement upon LangMem,the previous state-of-the-art, and it achieves 62.99% on PERSONAMEM, a 3.5% improvement upon A-Mem,the previous state-of-the-art. O-Mem also boosts token and interaction response time efficiency compared to previous memory frameworks. Our work opens up promising directions for developing efficient and human-like personalized AI assistants in the future. 

---
# Beyond SELECT: A Comprehensive Taxonomy-Guided Benchmark for Real-World Text-to-SQL Translation 

**Authors**: Hao Wang, Yuanfeng Song, Xiaoming Yin, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13590)  

**Abstract**: Text-to-SQL datasets are essential for training and evaluating text-to-SQL models, but existing datasets often suffer from limited coverage and fail to capture the diversity of real-world applications. To address this, we propose a novel taxonomy for text-to-SQL classification based on dimensions including core intents, statement types, syntax structures, and key actions. Using this taxonomy, we evaluate widely used public text-to-SQL datasets (e.g., Spider and Bird) and reveal limitations in their coverage and diversity. We then introduce a taxonomy-guided dataset synthesis pipeline, yielding a new dataset named SQL-Synth. This approach combines the taxonomy with Large Language Models (LLMs) to ensure the dataset reflects the breadth and complexity of real-world text-to-SQL applications. Extensive analysis and experimental results validate the effectiveness of our taxonomy, as SQL-Synth exhibits greater diversity and coverage compared to existing benchmarks. Moreover, we uncover that existing LLMs typically fall short in adequately capturing the full range of scenarios, resulting in limited performance on SQL-Synth. However, fine-tuning can substantially improve their performance in these scenarios. The proposed taxonomy has significant potential impact, as it not only enables comprehensive analysis of datasets and the performance of different LLMs, but also guides the construction of training data for LLMs. 

---
# Toward Conversational Hungarian Speech Recognition: Introducing the BEA-Large and BEA-Dialogue Datasets 

**Authors**: Máté Gedeon, Piroska Zsófia Barta, Péter Mihajlik, Tekla Etelka Gráczi, Anna Kohári, Katalin Mády  

**Link**: [PDF](https://arxiv.org/pdf/2511.13529)  

**Abstract**: The advancement of automatic speech recognition (ASR) has been largely enhanced by extensive datasets in high-resource languages, while languages such as Hungarian remain underrepresented due to limited spontaneous and conversational corpora. To address this gap, we introduce two new datasets -- BEA-Large and BEA-Dialogue -- constructed from the previously unprocessed portions of the Hungarian speech corpus named BEA. BEA-Large extends BEA-Base with 255 hours of spontaneous speech from 433 speakers, enriched with detailed segment-level metadata. BEA-Dialogue, comprising 85 hours of spontaneous conversations, is a Hungarian speech corpus featuring natural dialogues partitioned into speaker-independent subsets, supporting research in conversational ASR and speaker diarization. We establish reproducible baselines on these datasets using publicly available ASR models, with the fine-tuned Fast Conformer model achieving word error rates as low as 14.18\% on spontaneous and 4.8\% on repeated speech. Diarization experiments yield diarization error rates between 13.05\% and 18.26\%, providing reference points for future improvements. The results highlight the persistent difficulty of conversational ASR, particularly due to disfluencies, overlaps, and informal speech patterns. By releasing these datasets and baselines, we aim to advance Hungarian speech technology and offer a methodological framework for developing spontaneous and conversational benchmarks in other languages. 

---
# Applying Large Language Models to Characterize Public Narratives 

**Authors**: Elinor Poole-Dayan, Daniel T Kessler, Hannah Chiou, Margaret Hughes, Emily S Lin, Marshall Ganz, Deb Roy  

**Link**: [PDF](https://arxiv.org/pdf/2511.13505)  

**Abstract**: Public Narratives (PNs) are key tools for leadership development and civic mobilization, yet their systematic analysis remains challenging due to their subjective interpretation and the high cost of expert annotation. In this work, we propose a novel computational framework that leverages large language models (LLMs) to automate the qualitative annotation of public narratives. Using a codebook we co-developed with subject-matter experts, we evaluate LLM performance against that of expert annotators. Our work reveals that LLMs can achieve near-human-expert performance, achieving an average F1 score of 0.80 across 8 narratives and 14 codes. We then extend our analysis to empirically explore how PN framework elements manifest across a larger dataset of 22 stories. Lastly, we extrapolate our analysis to a set of political speeches, establishing a novel lens in which to analyze political rhetoric in civic spaces. This study demonstrates the potential of LLM-assisted annotation for scalable narrative analysis and highlights key limitations and directions for future research in computational civic storytelling. 

---
# Aspect-Level Obfuscated Sentiment in Thai Financial Disclosures and Its Impact on Abnormal Returns 

**Authors**: Attapol T. Rutherford, Sirisak Chueykamhang, Thachaparn Bunditlurdruk, Nanthicha Angsuwichitkul  

**Link**: [PDF](https://arxiv.org/pdf/2511.13481)  

**Abstract**: Understanding sentiment in financial documents is crucial for gaining insights into market behavior. These reports often contain obfuscated language designed to present a positive or neutral outlook, even when underlying conditions may be less favorable. This paper presents a novel approach using Aspect-Based Sentiment Analysis (ABSA) to decode obfuscated sentiment in Thai financial annual reports. We develop specific guidelines for annotating obfuscated sentiment in these texts and annotate more than one hundred financial reports. We then benchmark various text classification models on this annotated dataset, demonstrating strong performance in sentiment classification. Additionally, we conduct an event study to evaluate the real-world implications of our sentiment analysis on stock prices. Our results suggest that market reactions are selectively influenced by specific aspects within the reports. Our findings underscore the complexity of sentiment analysis in financial texts and highlight the importance of addressing obfuscated language to accurately assess market sentiment. 

---
# Non-Linear Scoring Model for Translation Quality Evaluation 

**Authors**: Serge Gladkoff, Lifeng Han, Katerina Gasova  

**Link**: [PDF](https://arxiv.org/pdf/2511.13467)  

**Abstract**: Analytic Translation Quality Evaluation (TQE), based on Multidimensional Quality Metrics (MQM), traditionally uses a linear error-to-penalty scale calibrated to a reference sample of 1000-2000 words. However, linear extrapolation biases judgment on samples of different sizes, over-penalizing short samples and under-penalizing long ones, producing misalignment with expert intuition.
Building on the Multi-Range framework, this paper presents a calibrated, non-linear scoring model that better reflects how human content consumers perceive translation quality across samples of varying length. Empirical data from three large-scale enterprise environments shows that acceptable error counts grow logarithmically, not linearly, with sample size.
Psychophysical and cognitive evidence, including the Weber-Fechner law and Cognitive Load Theory, supports this premise by explaining why the perceptual impact of additional errors diminishes while the cognitive burden grows with scale. We propose a two-parameter model
E(x) = a * ln(1 + b * x), a, b > 0,
anchored to a reference tolerance and calibrated from two tolerance points using a one-dimensional root-finding step. The model yields an explicit interval within which the linear approximation stays within +/-20 percent relative error and integrates into existing evaluation workflows with only a dynamic tolerance function added.
The approach improves interpretability, fairness, and inter-rater reliability across both human and AI-generated translations. By operationalizing a perceptually valid scoring paradigm, it advances translation quality evaluation toward more accurate and scalable assessment. The model also provides a stronger basis for AI-based document-level evaluation aligned with human judgment. Implementation considerations for CAT/LQA systems and implications for human and AI-generated text evaluation are discussed. 

---
# Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction 

**Authors**: Zhaopei Huang, Qifeng Dai, Guozheng Wu, Xiaopeng Wu, Kehan Chen, Chuan Yu, Xubin Li, Tiezheng Ge, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.13410)  

**Abstract**: With the rise of smart personal devices, service-oriented human-agent interactions have become increasingly prevalent. This trend highlights the need for personalized dialogue assistants that can understand user-specific traits to accurately interpret requirements and tailor responses to individual preferences. However, existing approaches often overlook the complexities of long-term interactions and fail to capture users' subjective characteristics. To address these gaps, we present PAL-Bench, a new benchmark designed to evaluate the personalization capabilities of service-oriented assistants in long-term user-agent interactions. In the absence of available real-world data, we develop a multi-step LLM-based synthesis pipeline, which is further verified and refined by human annotators. This process yields PAL-Set, the first Chinese dataset comprising multi-session user logs and dialogue histories, which serves as the foundation for PAL-Bench. Furthermore, to improve personalized service-oriented interactions, we propose H$^2$Memory, a hierarchical and heterogeneous memory framework that incorporates retrieval-augmented generation to improve personalized response generation. Comprehensive experiments on both our PAL-Bench and an external dataset demonstrate the effectiveness of the proposed memory framework. 

---
# Can Large Language Models Function as Qualified Pediatricians? A Systematic Evaluation in Real-World Clinical Contexts 

**Authors**: Siyu Zhu, Mouxiao Bian, Yue Xie, Yongyu Tang, Zhikang Yu, Tianbin Li, Pengcheng Chen, Bing Han, Jie Xu, Xiaoyan Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.13381)  

**Abstract**: With the rapid rise of large language models (LLMs) in medicine, a key question is whether they can function as competent pediatricians in real-world clinical settings. We developed PEDIASBench, a systematic evaluation framework centered on a knowledge-system framework and tailored to realistic clinical environments. PEDIASBench assesses LLMs across three dimensions: application of basic knowledge, dynamic diagnosis and treatment capability, and pediatric medical safety and medical ethics. We evaluated 12 representative models released over the past two years, including GPT-4o, Qwen3-235B-A22B, and DeepSeek-V3, covering 19 pediatric subspecialties and 211 prototypical diseases. State-of-the-art models performed well on foundational knowledge, with Qwen3-235B-A22B achieving over 90% accuracy on licensing-level questions, but performance declined ~15% as task complexity increased, revealing limitations in complex reasoning. Multiple-choice assessments highlighted weaknesses in integrative reasoning and knowledge recall. In dynamic diagnosis and treatment scenarios, DeepSeek-R1 scored highest in case reasoning (mean 0.58), yet most models struggled to adapt to real-time patient changes. On pediatric medical ethics and safety tasks, Qwen2.5-72B performed best (accuracy 92.05%), though humanistic sensitivity remained limited. These findings indicate that pediatric LLMs are constrained by limited dynamic decision-making and underdeveloped humanistic care. Future development should focus on multimodal integration and a clinical feedback-model iteration loop to enhance safety, interpretability, and human-AI collaboration. While current LLMs cannot independently perform pediatric care, they hold promise for decision support, medical education, and patient communication, laying the groundwork for a safe, trustworthy, and collaborative intelligent pediatric healthcare system. 

---
# Donors and Recipients: On Asymmetric Transfer Across Tasks and Languages with Parameter-Efficient Fine-Tuning 

**Authors**: Kajetan Dymkiewicz, Ivan Vulic, Helen Yannakoudakis, Eilam Shapira, Roi Reichart, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2511.13368)  

**Abstract**: Large language models (LLMs) perform strongly across tasks and languages, yet how improvements in one task or language affect other tasks and languages and their combinations remains poorly understood. We conduct a controlled PEFT/LoRA study across multiple open-weight LLM families and sizes, treating task and language as transfer axes while conditioning on model family and size; we fine-tune each model on a single task-language source and measure transfer as the percentage-point change versus its baseline score when evaluated on all other task-language target pairs. We decompose transfer into (i) Matched-Task (Cross-Language), (ii) Matched-Language (Cross-Task), and (iii) Cross-Task (Cross-Language) regimes. We uncover two consistent general patterns. First, a pronounced on-task vs. off-task asymmetry: Matched-Task (Cross-Language) transfer is reliably positive, whereas off-task transfer often incurs collateral degradation. Second, a stable donor-recipient structure across languages and tasks (hub donors vs. brittle recipients). We outline implications for risk-aware fine-tuning and model specialisation. 

---
# AHaSIS: Shared Task on Sentiment Analysis for Arabic Dialects 

**Authors**: Maram Alharbi, Salmane Chafik, Saad Ezzini, Ruslan Mitkov, Tharindu Ranasinghe, Hansi Hettiarachchi  

**Link**: [PDF](https://arxiv.org/pdf/2511.13335)  

**Abstract**: The hospitality industry in the Arab world increasingly relies on customer feedback to shape services, driving the need for advanced Arabic sentiment analysis tools. To address this challenge, the Sentiment Analysis on Arabic Dialects in the Hospitality Domain shared task focuses on Sentiment Detection in Arabic Dialects. This task leverages a multi-dialect, manually curated dataset derived from hotel reviews originally written in Modern Standard Arabic (MSA) and translated into Saudi and Moroccan (Darija) dialects. The dataset consists of 538 sentiment-balanced reviews spanning positive, neutral, and negative categories. Translations were validated by native speakers to ensure dialectal accuracy and sentiment preservation. This resource supports the development of dialect-aware NLP systems for real-world applications in customer experience analysis. More than 40 teams have registered for the shared task, with 12 submitting systems during the evaluation phase. The top-performing system achieved an F1 score of 0.81, demonstrating the feasibility and ongoing challenges of sentiment analysis across Arabic dialects. 

---
# RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service Copyright Protection 

**Authors**: Shufan Yang, Zifeng Cheng, Zhiwei Jiang, Yafeng Yin, Cong Wang, Shiping Ge, Yuchen Fu, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13329)  

**Abstract**: Embedding-as-a-Service (EaaS) is an effective and convenient deployment solution for addressing various NLP tasks. Nevertheless, recent research has shown that EaaS is vulnerable to model extraction attacks, which could lead to significant economic losses for model providers. For copyright protection, existing methods inject watermark embeddings into text embeddings and use them to detect copyright infringement. However, current watermarking methods often resist only a subset of attacks and fail to provide \textit{comprehensive} protection. To this end, we present the region-triggered semantic watermarking framework called RegionMarker, which defines trigger regions within a low-dimensional space and injects watermarks into text embeddings associated with these regions. By utilizing a secret dimensionality reduction matrix to project onto this subspace and randomly selecting trigger regions, RegionMarker makes it difficult for watermark removal attacks to evade detection. Furthermore, by embedding watermarks across the entire trigger region and using the text embedding as the watermark, RegionMarker is resilient to both paraphrasing and dimension-perturbation attacks. Extensive experiments on various datasets show that RegionMarker is effective in resisting different attack methods, thereby protecting the copyright of EaaS. 

---
# Souper-Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance 

**Authors**: Shalini Maiti, Amar Budhiraja, Bhavul Gauri, Gaurav Chaurasia, Anton Protopopov, Alexis Audran-Reiss, Michael Slater, Despoina Magka, Tatiana Shavrina, Roberta Raileanu, Yoram Bachrach  

**Link**: [PDF](https://arxiv.org/pdf/2511.13254)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their training remains resource- and time-intensive, requiring massive compute power and careful orchestration of training procedures. Model souping-the practice of averaging weights from multiple models of the same architecture-has emerged as a promising pre- and post-training technique that can enhance performance without expensive retraining. In this paper, we introduce Soup Of Category Experts (SoCE), a principled approach for model souping that utilizes benchmark composition to identify optimal model candidates and applies non-uniform weighted averaging to maximize performance. Contrary to previous uniform-averaging approaches, our method leverages the observation that benchmark categories often exhibit low inter-correlations in model performance. SoCE identifies "expert" models for each weakly-correlated category cluster and combines them using optimized weighted averaging rather than uniform weights. We demonstrate that the proposed method improves performance and robustness across multiple domains, including multilingual capabilities, tool calling, and math and achieves state-of-the-art results on the Berkeley Function Calling Leaderboard. 

---
# Seeing isn't Hearing: Benchmarking Vision Language Models at Interpreting Spectrograms 

**Authors**: Tyler Loakman, Joseph James, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.13225)  

**Abstract**: With the rise of Large Language Models (LLMs) and their vision-enabled counterparts (VLMs), numerous works have investigated their capabilities in tasks that fuse the modalities of vision and language. In this work, we benchmark the extent to which VLMs are able to act as highly-trained phoneticians, interpreting spectrograms and waveforms of speech. To do this, we synthesise a novel dataset containing 4k+ English words spoken in isolation alongside stylistically consistent spectrogram and waveform figures. We test the ability of VLMs to understand these representations of speech through a multiple-choice task whereby models must predict the correct phonemic or graphemic transcription of a spoken word when presented amongst 3 distractor transcriptions that have been selected based on their phonemic edit distance to the ground truth. We observe that both zero-shot and finetuned models rarely perform above chance, demonstrating the requirement for specific parametric knowledge of how to interpret such figures, rather than paired samples alone. 

---
# Evaluating Large Language Models for Diacritic Restoration in Romanian Texts: A Comparative Study 

**Authors**: Mihai Dan Nadas, Laura Diosan  

**Link**: [PDF](https://arxiv.org/pdf/2511.13182)  

**Abstract**: Automatic diacritic restoration is crucial for text processing in languages with rich diacritical marks, such as Romanian. This study evaluates the performance of several large language models (LLMs) in restoring diacritics in Romanian texts. Using a comprehensive corpus, we tested models including OpenAI's GPT-3.5, GPT-4, GPT-4o, Google's Gemini 1.0 Pro, Meta's Llama 2 and Llama 3, MistralAI's Mixtral 8x7B Instruct, airoboros 70B, and OpenLLM-Ro's RoLlama 2 7B, under multiple prompt templates ranging from zero-shot to complex multi-shot instructions. Results show that models such as GPT-4o achieve high diacritic restoration accuracy, consistently surpassing a neutral echo baseline, while others, including Meta's Llama family, exhibit wider variability. These findings highlight the impact of model architecture, training data, and prompt design on diacritic restoration performance and outline promising directions for improving NLP tools for diacritic-rich languages. 

---
# Translation Entropy: A Statistical Framework for Evaluating Translation Systems 

**Authors**: Ronit D. Gross, Yanir Harel, Ido Kanter  

**Link**: [PDF](https://arxiv.org/pdf/2511.13180)  

**Abstract**: The translation of written language has been known since the 3rd century BC; however, its necessity has become increasingly common in the information age. Today, many translators exist, based on encoder-decoder deep architectures, nevertheless, no quantitative objective methods are available to assess their performance, likely because the entropy of even a single language remains unknown. This study presents a quantitative method for estimating translation entropy, with the following key finding. Given a translator, several sentences that differ by only one selected token of a given pivot sentence yield identical translations. Analyzing the statistics of this phenomenon across an ensemble of such sentences, consisting each of a pivot selected token, yields the probabilities of replacing this specific token with others while preserving the translation. These probabilities constitute the entropy of the selected token, and the average across all selected pivot tokens provides an estimate of the translator's overall translation entropy, which is enhanced along the decoder blocks. This entropic measure allows for the quantitative ranking of several publicly available translators and reveals whether mutual translation entropy is symmetric. Extending the proposed method to include the replacement of two tokens in a given pivot sentence demonstrates a multiplicative effect, where translation degeneracy is proportional to the product of the degeneracies of the two tokens. These findings establish translation entropy as a measurable property and objective benchmarking of artificial translators. Results are based on MarianMT, T5-Base and NLLB-200 translators. 

---
# TCM-5CEval: Extended Deep Evaluation Benchmark for LLM's Comprehensive Clinical Research Competence in Traditional Chinese Medicine 

**Authors**: Tianai Huang, Jiayuan Chen, Lu Lu, Pengcheng Chen, Tianbin Li, Bing Han, Wenchao Tang, Jie Xu, Ming Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13169)  

**Abstract**: Large language models (LLMs) have demonstrated exceptional capabilities in general domains, yet their application in highly specialized and culturally-rich fields like Traditional Chinese Medicine (TCM) requires rigorous and nuanced evaluation. Building upon prior foundational work such as TCM-3CEval, which highlighted systemic knowledge gaps and the importance of cultural-contextual alignment, we introduce TCM-5CEval, a more granular and comprehensive benchmark. TCM-5CEval is designed to assess LLMs across five critical dimensions: (1) Core Knowledge (TCM-Exam), (2) Classical Literacy (TCM-LitQA), (3) Clinical Decision-making (TCM-MRCD), (4) Chinese Materia Medica (TCM-CMM), and (5) Clinical Non-pharmacological Therapy (TCM-ClinNPT). We conducted a thorough evaluation of fifteen prominent LLMs, revealing significant performance disparities and identifying top-performing models like deepseek\_r1 and gemini\_2\_5\_pro. Our findings show that while models exhibit proficiency in recalling foundational knowledge, they struggle with the interpretative complexities of classical texts. Critically, permutation-based consistency testing reveals widespread fragilities in model inference. All evaluated models, including the highest-scoring ones, displayed a substantial performance degradation when faced with varied question option ordering, indicating a pervasive sensitivity to positional bias and a lack of robust understanding. TCM-5CEval not only provides a more detailed diagnostic tool for LLM capabilities in TCM but aldso exposes fundamental weaknesses in their reasoning stability. To promote further research and standardized comparison, TCM-5CEval has been uploaded to the Medbench platform, joining its predecessor in the "In-depth Challenge for Comprehensive TCM Abilities" special track. 

---
# Distinguishing Repetition Disfluency from Morphological Reduplication in Bangla ASR Transcripts: A Novel Corpus and Benchmarking Analysis 

**Authors**: Zaara Zabeen Arpa, Sadnam Sakib Apurbo, Nazia Karim Khan Oishee, Ajwad Abrar  

**Link**: [PDF](https://arxiv.org/pdf/2511.13159)  

**Abstract**: Automatic Speech Recognition (ASR) transcripts, especially in low-resource languages like Bangla, contain a critical ambiguity: word-word repetitions can be either Repetition Disfluency (unintentional ASR error/hesitation) or Morphological Reduplication (a deliberate grammatical construct). Standard disfluency correction fails by erroneously deleting valid linguistic information. To solve this, we introduce the first publicly available, 20,000-row Bangla corpus, manually annotated to explicitly distinguish between these two phenomena in noisy ASR transcripts. We benchmark this novel resource using two paradigms: state-of-the-art multilingual Large Language Models (LLMs) and task-specific fine-tuning of encoder models. LLMs achieve competitive performance (up to 82.68\% accuracy) with few-shot prompting. However, fine-tuning proves superior, with the language-specific BanglaBERT model achieving the highest accuracy of 84.78\% and an F1 score of 0.677. This establishes a strong, linguistically-informed baseline and provides essential data for developing sophisticated, semantic-preserving text normalization systems for Bangla. 

---
# Zero-Shot Grammar Competency Estimation Using Large Language Model Generated Pseudo Labels 

**Authors**: Sourya Dipta Das, Shubham Kumar, Kuldeep Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2511.13152)  

**Abstract**: Grammar competency estimation is essential for assessing linguistic proficiency in both written and spoken language; however, the spoken modality presents additional challenges due to its spontaneous, unstructured, and disfluent nature. Developing accurate grammar scoring models further requires extensive expert annotation, making large-scale data creation impractical. To address these limitations, we propose a zero-shot grammar competency estimation framework that leverages unlabeled data and Large Language Models (LLMs) without relying on manual labels. During training, we employ LLM-generated predictions on unlabeled data by using grammar competency rubric-based prompts. These predictions, treated as pseudo labels, are utilized to train a transformer-based model through a novel training framework designed to handle label noise effectively. We show that the choice of LLM for pseudo-label generation critically affects model performance and that the ratio of clean-to-noisy samples during training strongly influences stability and accuracy. Finally, a qualitative analysis of error intensity and score prediction confirms the robustness and interpretability of our approach. Experimental results demonstrate the efficacy of our approach in estimating grammar competency scores with high accuracy, paving the way for scalable, low-resource grammar assessment systems. 

---
# A Comparative Analysis of Recurrent and Attention Architectures for Isolated Sign Language Recognition 

**Authors**: Nigar Alishzade, Gulchin Abdullayeva  

**Link**: [PDF](https://arxiv.org/pdf/2511.13126)  

**Abstract**: This study presents a systematic comparative analysis of recurrent and attention-based neural architectures for isolated sign language recognition. We implement and evaluate two representative models-ConvLSTM and Vanilla Transformer-on the Azerbaijani Sign Language Dataset (AzSLD) and the Word-Level American Sign Language (WLASL) dataset. Our results demonstrate that the attention-based Vanilla Transformer consistently outperforms the recurrent ConvLSTM in both Top-1 and Top-5 accuracy across datasets, achieving up to 76.8% Top-1 accuracy on AzSLD and 88.3% on WLASL. The ConvLSTM, while more computationally efficient, lags in recognition accuracy, particularly on smaller datasets. These findings highlight the complementary strengths of each paradigm: the Transformer excels in overall accuracy and signer independence, whereas the ConvLSTM offers advantages in computational efficiency and temporal modeling. The study provides a nuanced analysis of these trade-offs, offering guidance for architecture selection in sign language recognition systems depending on application requirements and resource constraints. 

---
# Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction 

**Authors**: Quanjiang Guo, Sijie Wang, Jinchuan Zhang, Ben Zhang, Zhao Kang, Ling Tian, Ke Yan  

**Link**: [PDF](https://arxiv.org/pdf/2511.13118)  

**Abstract**: Zero-shot event extraction (ZSEE) remains a significant challenge for large language models (LLMs) due to the need for complex reasoning and domain-specific understanding. Direct prompting often yields incomplete or structurally invalid outputs--such as misclassified triggers, missing arguments, and schema violations. To address these limitations, we present Agent-Event-Coder (AEC), a novel multi-agent framework that treats event extraction like software engineering: as a structured, iterative code-generation process. AEC decomposes ZSEE into specialized subtasks--retrieval, planning, coding, and verification--each handled by a dedicated LLM agent. Event schemas are represented as executable class definitions, enabling deterministic validation and precise feedback via a verification agent. This programming-inspired approach allows for systematic disambiguation and schema enforcement through iterative refinement. By leveraging collaborative agent workflows, AEC enables LLMs to produce precise, complete, and schema-consistent extractions in zero-shot settings. Experiments across five diverse domains and six LLMs demonstrate that AEC consistently outperforms prior zero-shot baselines, showcasing the power of treating event extraction like code generation. The code and data are released on this https URL. 

---
# Evaluating the Ability of Large Language Models to Identify Adherence to CONSORT Reporting Guidelines in Randomized Controlled Trials: A Methodological Evaluation Study 

**Authors**: Zhichao He, Mouxiao Bian, Jianhong Zhu, Jiayuan Chen, Yunqiu Wang, Wenxia Zhao, Tianbin Li, Bing Han, Jie Xu, Junyan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13107)  

**Abstract**: The Consolidated Standards of Reporting Trials statement is the global benchmark for transparent and high-quality reporting of randomized controlled trials. Manual verification of CONSORT adherence is a laborious, time-intensive process that constitutes a significant bottleneck in peer review and evidence synthesis. This study aimed to systematically evaluate the accuracy and reliability of contemporary LLMs in identifying the adherence of published RCTs to the CONSORT 2010 statement under a zero-shot setting. We constructed a golden standard dataset of 150 published RCTs spanning diverse medical specialties. The primary outcome was the macro-averaged F1-score for the three-class classification task, supplemented by item-wise performance metrics and qualitative error analysis. Overall model performance was modest. The top-performing models, Gemini-2.5-Flash and DeepSeek-R1, achieved nearly identical macro F1 scores of 0.634 and Cohen's Kappa coefficients of 0.280 and 0.282, respectively, indicating only fair agreement with expert consensus. A striking performance disparity was observed across classes: while most models could identify compliant items with high accuracy (F1 score > 0.850), they struggled profoundly with identifying non-compliant and not applicable items, where F1 scores rarely exceeded 0.400. Notably, some high-profile models like GPT-4o underperformed, achieving a macro F1-score of only 0.521. LLMs show potential as preliminary screening assistants for CONSORT checks, capably identifying well-reported items. However, their current inability to reliably detect reporting omissions or methodological flaws makes them unsuitable for replacing human expertise in the critical appraisal of trial quality. 

---
# BeDiscovER: The Benchmark of Discourse Understanding in the Era of Reasoning Language Models 

**Authors**: Chuyuan Li, Giuseppe Carenini  

**Link**: [PDF](https://arxiv.org/pdf/2511.13095)  

**Abstract**: We introduce BeDiscovER (Benchmark of Discourse Understanding in the Era of Reasoning Language Models), an up-to-date, comprehensive suite for evaluating the discourse-level knowledge of modern LLMs. BeDiscovER compiles 5 publicly available discourse tasks across discourse lexicon, (multi-)sentential, and documental levels, with in total 52 individual datasets. It covers both extensively studied tasks such as discourse parsing and temporal relation extraction, as well as some novel challenges such as discourse particle disambiguation (e.g., ``just''), and also aggregates a shared task on Discourse Relation Parsing and Treebanking for multilingual and multi-framework discourse relation classification. We evaluate open-source LLMs: Qwen3 series, DeepSeek-R1, and frontier model such as GPT-5-mini on BeDiscovER, and find that state-of-the-art models exhibit strong performance in arithmetic aspect of temporal reasoning, but they struggle with full document reasoning and some subtle semantic and discourse phenomena, such as rhetorical relation recognition. 

---
# Spark-Prover-X1: Formal Theorem Proving Through Diverse Data Training 

**Authors**: Xinyuan Zhou, Yi Lei, Xiaoyu Zhou, Jingyi Sun, Yu Zhu, Zhongyi Ye, Weitai Zhang, Quan Liu, Si Wei, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13043)  

**Abstract**: Large Language Models (LLMs) have shown significant promise in automated theorem proving, yet progress is often constrained by the scarcity of diverse and high-quality formal language data. To address this issue, we introduce Spark-Prover-X1, a 7B parameter model trained via an three-stage framework designed to unlock the reasoning potential of more accessible and moderately-sized LLMs. The first stage infuses deep knowledge through continuous pre-training on a broad mathematical corpus, enhanced by a suite of novel data tasks. Key innovation is a "CoT-augmented state prediction" task to achieve fine-grained reasoning. The second stage employs Supervised Fine-tuning (SFT) within an expert iteration loop to specialize both the Spark-Prover-X1-7B and Spark-Formalizer-X1-7B models. Finally, a targeted round of Group Relative Policy Optimization (GRPO) is applied to sharpen the prover's capabilities on the most challenging problems. To facilitate robust evaluation, particularly on problems from real-world examinations, we also introduce ExamFormal-Bench, a new benchmark dataset of 402 formal problems. Experimental results demonstrate that Spark-Prover-X1-7B achieves state-of-the-art performance among similarly-sized open-source models, attaining a 37.0\% average pass rate (pass@32). It shows exceptional performance on difficult competition benchmarks, notably solving 27 problems on PutnamBench (pass@32) and achieving 24.0\% on CombiBench (pass@32). Our work validates that this diverse training data and progressively refined training pipeline provides an effective path for enhancing the formal reasoning capabilities of lightweight LLMs. Both Spark-Prover-X1-7B and Spark-Formalizer-X1-7B, along with the ExamFormal-Bench dataset, are made publicly available at:this https URL, this https URL. 

---
# How Good is BLI as an Alignment Measure: A Study in Word Embedding Paradigm 

**Authors**: Kasun Wickramasinghe, Nisansa de Silva  

**Link**: [PDF](https://arxiv.org/pdf/2511.13040)  

**Abstract**: Sans a dwindling number of monolingual embedding studies originating predominantly from the low-resource domains, it is evident that multilingual embedding has become the de facto choice due to its adaptability to the usage of code-mixed languages, granting the ability to process multilingual documents in a language-agnostic manner, as well as removing the difficult task of aligning monolingual embeddings. But is this victory complete? Are the multilingual models better than aligned monolingual models in every aspect? Can the higher computational cost of multilingual models always be justified? Or is there a compromise between the two extremes? Bilingual Lexicon Induction is one of the most widely used metrics in terms of evaluating the degree of alignment between two embedding spaces. In this study, we explore the strengths and limitations of BLI as a measure to evaluate the degree of alignment of two embedding spaces. Further, we evaluate how well traditional embedding alignment techniques, novel multilingual models, and combined alignment techniques perform BLI tasks in the contexts of both high-resource and low-resource languages. In addition to that, we investigate the impact of the language families to which the pairs of languages belong. We identify that BLI does not measure the true degree of alignment in some cases and we propose solutions for them. We propose a novel stem-based BLI approach to evaluate two aligned embedding spaces that take into account the inflected nature of languages as opposed to the prevalent word-based BLI techniques. Further, we introduce a vocabulary pruning technique that is more informative in showing the degree of the alignment, especially performing BLI on multilingual embedding models. Often, combined embedding alignment techniques perform better while in certain cases multilingual embeddings perform better (mainly low-resource language cases). 

---
# AA-Omniscience: Evaluating Cross-Domain Knowledge Reliability in Large Language Models 

**Authors**: Declan Jackson, William Keating, George Cameron, Micah Hill-Smith  

**Link**: [PDF](https://arxiv.org/pdf/2511.13029)  

**Abstract**: Existing language model evaluations primarily measure general capabilities, yet reliable use of these models across a range of domains demands factual accuracy and recognition of knowledge gaps. We introduce AA-Omniscience, a benchmark designed to measure both factual recall and knowledge calibration across 6,000 questions. Questions are derived from authoritative academic and industry sources, and cover 42 economically relevant topics within six different domains. The evaluation measures a model's Omniscience Index, a bounded metric (-100 to 100) measuring factual recall that jointly penalizes hallucinations and rewards abstention when uncertain, with 0 equating to a model that answers questions correctly as much as it does incorrectly. Among evaluated models, Claude 4.1 Opus attains the highest score (4.8), making it one of only three models to score above zero. These results reveal persistent factuality and calibration weaknesses across frontier models. Performance also varies by domain, with the models from three different research labs leading across the six domains. This performance variability suggests models should be chosen according to the demands of the use case rather than general performance for tasks where knowledge is important. 

---
# Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty 

**Authors**: Zeyu Shi, Ziming Wang, Tianyu Chen, Shiqi Gao, Haoyi Zhou, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.12991)  

**Abstract**: The honesty of Large Language Models (LLMs) is increasingly important for safe deployment in high-stakes domains. However, this crucial trait is severely undermined by supervised fine-tuning (SFT), a common technique for model specialization. Existing recovery methods rely on data-intensive global parameter adjustments, implicitly assuming that SFT deeply corrupts the models' ability to recognize their knowledge boundaries. However, we observe that fine-tuned LLMs still preserve this ability; what is damaged is their capacity to faithfully express that awareness. Building on this, we propose Honesty-Critical Neurons Restoration (HCNR) to surgically repair this suppressed capacity. HCNR identifies and restores key expression-governing neurons to their pre-trained state while harmonizing them with task-oriented neurons via Hessian-guided compensation. Experiments on four QA tasks and five LLM families demonstrate that HCNR effectively recovers 33.25% of the compromised honesty while achieving at least 2.23x speedup with over 10x less data compared to baseline methods, offering a practical solution for trustworthy LLM deployment. 

---
# Visual Room 2.0: Seeing is Not Understanding for MLLMs 

**Authors**: Haokun Li, Yazhou Zhang, Jizhi Ding, Qiuchi Li, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12928)  

**Abstract**: Can multi-modal large language models (MLLMs) truly understand what they can see? Extending Searle's Chinese Room into the multi-modal domain, this paper proposes the Visual Room argument: MLLMs may describe every visual detail precisely yet fail to comprehend the underlying emotions and intentions, namely seeing is not understanding. Building on this, we introduce \textit{Visual Room} 2.0, a hierarchical benchmark for evaluating perception-cognition alignment of MLLMs. We model human perceptive and cognitive processes across three levels: low, middle, and high, covering 17 representative tasks. The perception component ranges from attribute recognition to scene understanding, while the cognition component extends from textual entailment to causal and social reasoning. The dataset contains 350 multi-modal samples, each with six progressive questions (2,100 in total) spanning perception to cognition. Evaluating 10 state-of-the-art (SoTA) MLLMs, we highlight three key findings: (1) MLLMs exhibit stronger perceptual competence than cognitive ability (8.0\%$\uparrow$); (2) cognition appears not causally dependent on perception-based reasoning; and (3) cognition scales with model size, but perception does not consistently improve with larger variants. This work operationalizes Seeing $\ne$ Understanding as a testable hypothesis, offering a new paradigm from perceptual processing to cognitive reasoning in MLLMs. Our dataset is available at this https URL. 

---
# Auditing Google's AI Overviews and Featured Snippets: A Case Study on Baby Care and Pregnancy 

**Authors**: Desheng Hu, Joachim Baumann, Aleksandra Urman, Elsa Lichtenegger, Robin Forsberg, Aniko Hannak, Christo Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2511.12920)  

**Abstract**: Google Search increasingly surfaces AI-generated content through features like AI Overviews (AIO) and Featured Snippets (FS), which users frequently rely on despite having no control over their presentation. Through a systematic algorithm audit of 1,508 real baby care and pregnancy-related queries, we evaluate the quality and consistency of these information displays. Our robust evaluation framework assesses multiple quality dimensions, including answer consistency, relevance, presence of medical safeguards, source categories, and sentiment alignment. Our results reveal concerning gaps in information consistency, with information in AIO and FS displayed on the same search result page being inconsistent with each other in 33% of cases. Despite high relevance scores, both features critically lack medical safeguards (present in just 11% of AIO and 7% of FS responses). While health and wellness websites dominate source categories for both, AIO and FS, FS also often link to commercial sources. These findings have important implications for public health information access and demonstrate the need for stronger quality controls in AI-mediated health information. Our methodology provides a transferable framework for auditing AI systems across high-stakes domains where information quality directly impacts user well-being. 

---
# Classification of Hope in Textual Data using Transformer-Based Models 

**Authors**: Chukwuebuka Fortunate Ijezue, Tania-Amanda Fredrick Eneye, Maaz Amjad  

**Link**: [PDF](https://arxiv.org/pdf/2511.12874)  

**Abstract**: This paper presents a transformer-based approach for classifying hope expressions in text. We developed and compared three architectures (BERT, GPT-2, and DeBERTa) for both binary classification (Hope vs. Not Hope) and multiclass categorization (five hope-related categories). Our initial BERT implementation achieved 83.65% binary and 74.87% multiclass accuracy. In the extended comparison, BERT demonstrated superior performance (84.49% binary, 72.03% multiclass accuracy) while requiring significantly fewer computational resources (443s vs. 704s training time) than newer architectures. GPT-2 showed lowest overall accuracy (79.34% binary, 71.29% multiclass), while DeBERTa achieved moderate results (80.70% binary, 71.56% multiclass) but at substantially higher computational cost (947s for multiclass training). Error analysis revealed architecture-specific strengths in detecting nuanced hope expressions, with GPT-2 excelling at sarcasm detection (92.46% recall). This study provides a framework for computational analysis of hope, with applications in mental health and social media analysis, while demonstrating that architectural suitability may outweigh model size for specialized emotion detection tasks. 

---
# From Perception to Reasoning: Deep Thinking Empowers Multimodal Large Language Models 

**Authors**: Wenxin Zhu, Andong Chen, Yuchen Song, Kehai Chen, Conghui Zhu, Ziyan Chen, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12861)  

**Abstract**: With the remarkable success of Multimodal Large Language Models (MLLMs) in perception tasks, enhancing their complex reasoning capabilities has emerged as a critical research focus. Existing models still suffer from challenges such as opaque reasoning paths and insufficient generalization ability. Chain-of-Thought (CoT) reasoning, which has demonstrated significant efficacy in language models by enhancing reasoning transparency and output interpretability, holds promise for improving model reasoning capabilities when extended to the multimodal domain. This paper provides a systematic review centered on "Multimodal Chain-of-Thought" (MCoT). First, it analyzes the background and theoretical motivations for its inception from the perspectives of technical evolution and task demands. Then, it introduces mainstream MCoT methods from three aspects: CoT paradigms, the post-training stage, and the inference stage, while also analyzing their underlying mechanisms. Furthermore, the paper summarizes existing evaluation benchmarks and metrics, and discusses the application scenarios of MCoT. Finally, it analyzes the challenges currently facing MCoT and provides an outlook on its future research directions. 

---
# NeuroLex: A Lightweight Domain Language Model for EEG Report Understanding and Generation 

**Authors**: Kang Yin, Hye-Bin Shin  

**Link**: [PDF](https://arxiv.org/pdf/2511.12851)  

**Abstract**: Clinical electroencephalogram (EEG) reports encode domain-specific linguistic conventions that general-purpose language models (LMs) fail to capture. We introduce NeuroLex, a lightweight domain-adaptive language model trained purely on EEG report text from the Harvard Electroencephalography Database. Unlike existing biomedical LMs, NeuroLex is tailored to the linguistic and diagnostic characteristics of EEG reporting, enabling it to serve as both an independent textual model and a decoder backbone for multimodal EEG-language systems. Using span-corruption pretraining and instruction-style fine-tuning on report polishing, paragraph summarization, and terminology question answering, NeuroLex learns the syntax and reasoning patterns characteristic of EEG interpretation. Comprehensive evaluations show that it achieves lower perplexity, higher extraction and summarization accuracy, better label efficiency, and improved robustness to negation and factual hallucination compared with general models of the same scale. With an EEG-aware linguistic backbone, NeuroLex bridges biomedical text modeling and brain-computer interface applications, offering a foundation for interpretable and language-driven neural decoding. 

---
# Quantifying consistency and accuracy of Latent Dirichlet Allocation 

**Authors**: Saranzaya Magsarjav, Melissa Humphries, Jonathan Tuke, Lewis Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2511.12850)  

**Abstract**: Topic modelling in Natural Language Processing uncovers hidden topics in large, unlabelled text datasets. It is widely applied in fields such as information retrieval, content summarisation, and trend analysis across various disciplines. However, probabilistic topic models can produce different results when rerun due to their stochastic nature, leading to inconsistencies in latent topics. Factors like corpus shuffling, rare text removal, and document elimination contribute to these variations. This instability affects replicability, reliability, and interpretation, raising concerns about whether topic models capture meaningful topics or just noise. To address these problems, we defined a new stability measure that incorporates accuracy and consistency and uses the generative properties of LDA to generate a new corpus with ground truth. These generated corpora are run through LDA 50 times to determine the variability in the output. We show that LDA can correctly determine the underlying number of topics in the documents. We also find that LDA is more internally consistent, as the multiple reruns return similar topics; however, these topics are not the true topics. 

---
# From Passive to Persuasive: Steering Emotional Nuance in Human-AI Negotiation 

**Authors**: Niranjan Chebrolu, Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2511.12832)  

**Abstract**: Large Language Models (LLMs) demonstrate increasing conversational fluency, yet instilling them with nuanced, human-like emotional expression remains a significant challenge. Current alignment techniques often address surface-level output or require extensive fine-tuning. This paper demonstrates that targeted activation engineering can steer LLaMA 3.1-8B to exhibit more human-like emotional nuances. We first employ attribution patching to identify causally influential components, to find a key intervention locus by observing activation patterns during diagnostic conversational tasks. We then derive emotional expression vectors from the difference in the activations generated by contrastive text pairs (positive vs. negative examples of target emotions). Applying these vectors to new conversational prompts significantly enhances emotional characteristics: steered responses show increased positive sentiment (e.g., joy, trust) and more frequent first-person pronoun usage, indicative of greater personal engagement. Our findings offer a precise and interpretable framework and new directions for the study of conversational AI. 

---
# BioMedJImpact: A Comprehensive Dataset and LLM Pipeline for AI Engagement and Scientific Impact Analysis of Biomedical Journals 

**Authors**: Ruiyu Wang, Yuzhang Xie, Xiao Hu, Carl Yang, Jiaying Lu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12821)  

**Abstract**: Assessing journal impact is central to scholarly communication, yet existing open resources rarely capture how collaboration structures and artificial intelligence (AI) research jointly shape venue prestige in biomedicine. We present BioMedJImpact, a large-scale, biomedical-oriented dataset designed to advance journal-level analysis of scientific impact and AI engagement. Built from 1.74 million PubMed Central articles across 2,744 journals, BioMedJImpact integrates bibliometric indicators, collaboration features, and LLM-derived semantic indicators for AI engagement. Specifically, the AI engagement feature is extracted through a reproducible three-stage LLM pipeline that we propose. Using this dataset, we analyze how collaboration intensity and AI engagement jointly influence scientific impact across pre- and post-pandemic periods (2016-2019, 2020-2023). Two consistent trends emerge: journals with higher collaboration intensity, particularly those with larger and more diverse author teams, tend to achieve greater citation impact, and AI engagement has become an increasingly strong correlate of journal prestige, especially in quartile rankings. To further validate the three-stage LLM pipeline we proposed for deriving the AI engagement feature, we conduct human evaluation, confirming substantial agreement in AI relevance detection and consistent subfield classification. Together, these contributions demonstrate that BioMedJImpact serves as both a comprehensive dataset capturing the intersection of biomedicine and AI, and a validated methodological framework enabling scalable, content-aware scientometric analysis of scientific impact and innovation dynamics. Code is available at this https URL. 

---
# Evaluating Autoformalization Robustness via Semantically Similar Paraphrasing 

**Authors**: Hayden Moore, Asfahan Shah  

**Link**: [PDF](https://arxiv.org/pdf/2511.12784)  

**Abstract**: Large Language Models (LLMs) have recently emerged as powerful tools for autoformalization. Despite their impressive performance, these models can still struggle to produce grounded and verifiable formalizations. Recent work in text-to-SQL, has revealed that LLMs can be sensitive to paraphrased natural language (NL) inputs, even when high degrees of semantic fidelity are preserved (Safarzadeh, Oroojlooyjadid, and Roth 2025). In this paper, we investigate this claim in the autoformalization domain. Specifically, we evaluate the robustness of LLMs generating formal proofs with semantically similar paraphrased NL statements by measuring semantic and compilation validity. Using the formal benchmarks MiniF2F (Zheng, Han, and Polu 2021) and Lean 4 version of ProofNet (Xin et al. 2024), and two modern LLMs, we generate paraphrased natural language statements and cross-evaluate these statements across both models. The results of this paper reveal performance variability across paraphrased inputs, demonstrating that minor shifts in NL statements can significantly impact model outputs. 

---
# LLM Reinforcement in Context 

**Authors**: Thomas Rivasseau  

**Link**: [PDF](https://arxiv.org/pdf/2511.12782)  

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming. 

---
# Evidence of Phase Transitions in Small Transformer-Based Language Models 

**Authors**: Noah Hong, Tao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2511.12768)  

**Abstract**: Phase transitions have been proposed as the origin of emergent abilities in large language models (LLMs), where new capabilities appear abruptly once models surpass critical thresholds of scale. Prior work, such as that of Wei et al., demonstrated these phenomena under model and data scaling, with transitions revealed after applying a log scale to training compute. In this work, we ask three complementary questions: (1) Are phase transitions unique to large models, or can they also be observed in small transformer-based language models? (2) Can such transitions be detected directly in linear training space, rather than only after log rescaling? and (3) Can these transitions emerge at early stages of training? To investigate, we train a small GPT-style transformer on a character-level corpus and analyze the evolution of vocabulary usage throughout training. We track the average word length, the number of correct versus incorrect words, and shifts in vocabulary diversity. Building on these measures, we apply Poisson and sub-Poisson statistics to quantify how words connect and reorganize. This combined analysis reveals a distinct transition point during training. Notably, these transitions are not apparent in standard loss or validation curves, but become visible through our vocabulary- and statistics-based probes. Our findings suggest that phase-transition reorganizations are a general feature of language model training, observable even in modest models, detectable directly in linear training space, and occurring surprisingly early as coherence emerges. This perspective provides new insight into the nonlinear dynamics of language model training and underscores the importance of tailored metrics for uncovering phase transition behaviors 

---
# On the Brittleness of LLMs: A Journey around Set Membership 

**Authors**: Lea Hergert, Gábor Berend, Mario Szegedy, Gyorgy Turan, Márk Jelasity  

**Link**: [PDF](https://arxiv.org/pdf/2511.12728)  

**Abstract**: Large language models (LLMs) achieve superhuman performance on complex reasoning tasks, yet often fail on much simpler problems, raising concerns about their reliability and interpretability. We investigate this paradox through a focused study with two key design features: simplicity, to expose basic failure modes, and scale, to enable comprehensive controlled experiments. We focus on set membership queries -- among the most fundamental forms of reasoning -- using tasks like ``Is apple an element of the set \{pear, plum, apple, raspberry\}?''. We conduct a systematic empirical evaluation across prompt phrasing, semantic structure, element ordering, and model choice. Our large-scale analysis reveals that LLM performance on this elementary task is consistently brittle, and unpredictable across all dimensions, suggesting that the models' ``understanding'' of the set concept is fragmented and convoluted at best. Our work demonstrates that the large-scale experiments enabled by the simplicity of the problem allow us to map and analyze the failure modes comprehensively, making this approach a valuable methodology for LLM evaluation in general. 

---
# Adaptive Focus Memory for Language Models 

**Authors**: Christopher Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2511.12712)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-turn dialogue settings, but their behavior is still bottlenecked by fixed context windows and naive memory strategies. Replaying the full conversation at every turn is simple but expensive, while static summarization or recency-only heuristics often erase safety-critical user details. We present Adaptive Focus Memory (AFM), a dynamic context manager that assigns each past message one of three fidelity levels -- FULL, COMPRESSED, or PLACEHOLDER -- based on semantic similarity to the current query, half-life recency weighting, and importance classification. AFM packs messages chronologically under a strict token budget, preferring high fidelity for the most relevant turns while aiming to preserve a cheap trace of the dialogue. In a safety-oriented benchmark involving a user with a severe peanut allergy planning a trip to Thailand, AFM retains the allergy across both short and medium-length conversations, matches the safety performance of naive replay, and cuts average token usage by 66% relative to a replay baseline. We release a modular Python implementation of AFM designed for OpenAI-compatible APIs and offline operation, enabling practitioners to reduce inference cost without sacrificing safety or factual continuity in the evaluated scenario. 

---
# Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs 

**Authors**: Yunhao Chen, Xin Wang, Juncheng Li, Yixu Wang, Jie Li, Yan Teng, Yingchun Wang, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2511.12710)  

**Abstract**: Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: this https URL. 

---
# Improving Direct Persian-English Speech-to-Speech Translation with Discrete Units and Synthetic Parallel Data 

**Authors**: Sina Rashidi, Hossein Sameti  

**Link**: [PDF](https://arxiv.org/pdf/2511.12690)  

**Abstract**: Direct speech-to-speech translation (S2ST), in which all components are trained jointly, is an attractive alternative to cascaded systems because it offers a simpler pipeline and lower inference latency. However, direct S2ST models require large amounts of parallel speech data in the source and target languages, which are rarely available for low-resource languages such as Persian. This paper presents a direct S2ST system for translating Persian speech into English speech, as well as a pipeline for synthetic parallel Persian-English speech generation. The model comprises three components: (1) a conformer-based encoder, initialized from self-supervised pre-training, maps source speech to high-level acoustic representations; (2) a causal transformer decoder with relative position multi-head attention translates these representations into discrete target speech units; (3) a unit-based neural vocoder generates waveforms from the predicted discrete units. To mitigate the data scarcity problem, we construct a new Persian-English parallel speech corpus by translating Persian speech transcriptions into English using a large language model and then synthesizing the corresponding English speech with a state-of-the-art zero-shot text-to-speech system. The resulting corpus increases the amount of available parallel speech by roughly a factor of six. On the Persian-English portion of the CVSS corpus, the proposed model achieves improvement of 4.6 ASR BLEU with the synthetic data over direct baselines. These results indicate that combining self-supervised pre-training, discrete speech units, and synthetic parallel data is effective for improving direct S2ST in low-resource language pairs such as Persian-English 

---
# Reason-KE++: Aligning the Process, Not Just the Outcome, for Faithful LLM Knowledge Editing 

**Authors**: Yuchen Wu, Liang Ding, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12661)  

**Abstract**: Aligning Large Language Models (LLMs) to be faithful to new knowledge in complex, multi-hop reasoning tasks is a critical, yet unsolved, challenge. We find that SFT-based methods, e.g., Reason-KE, while state-of-the-art, suffer from a "faithfulness gap": they optimize for format mimicry rather than sound reasoning. This gap enables the LLM's powerful parametric priors to override new contextual facts, resulting in critical factual hallucinations (e.g., incorrectly reasoning "Houston" from "NASA" despite an explicit edit). To solve this core LLM alignment problem, we propose Reason-KE++, an SFT+RL framework that instills process-level faithfulness. Its core is a Stage-aware Reward mechanism that provides dense supervision for intermediate reasoning steps (e.g., Decomposition, Sub-answer Correctness). Crucially, we identify that naive outcome-only RL is a deceptive trap for LLM alignment: it collapses reasoning integrity (e.g., 19.00% Hop acc) while superficially boosting final accuracy. Our process-aware framework sets a new SOTA of 95.48% on MQUAKE-CF-3k (+5.28%), demonstrating that for complex tasks, aligning the reasoning process is essential for building trustworthy LLMs. 

---
# Knots: A Large-Scale Multi-Agent Enhanced Expert-Annotated Dataset and LLM Prompt Optimization for NOTAM Semantic Parsing 

**Authors**: Maoqi Liu, Quan Fang, Yang Yang, Can Zhao, Kaiquan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.12630)  

**Abstract**: Notice to Air Missions (NOTAMs) serve as a critical channel for disseminating key flight safety information, yet their complex linguistic structures and implicit reasoning pose significant challenges for automated parsing. Existing research mainly focuses on surface-level tasks such as classification and named entity recognition, lacking deep semantic understanding. To address this gap, we propose NOTAM semantic parsing, a task emphasizing semantic inference and the integration of aviation domain knowledge to produce structured, inference-rich outputs. To support this task, we construct Knots (Knowledge and NOTAM Semantics), a high-quality dataset of 12,347 expert-annotated NOTAMs covering 194 Flight Information Regions, enhanced through a multi-agent collaborative framework for comprehensive field discovery. We systematically evaluate a wide range of prompt-engineering strategies and model-adaptation techniques, achieving substantial improvements in aviation text understanding and processing. Our experimental results demonstrate the effectiveness of the proposed approach and offer valuable insights for automated NOTAM analysis systems. Our code is available at: this https URL. 

---
# Uni-MoE-2.0-Omni: Scaling Language-Centric Omnimodal Large Model with Advanced MoE, Training and Data 

**Authors**: Yunxin Li, Xinyu Chen, Shenyuan Jiang, Haoyuan Shi, Zhenyu Liu, Xuanyu Zhang, Nanhao Deng, Zhenran Xu, Yicheng Ma, Meishan Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12609)  

**Abstract**: We present Uni-MoE 2.0 from the Lychee family. As a fully open-source omnimodal large model (OLM), it substantially advances Lychee's Uni-MoE series in language-centric multimodal understanding, reasoning, and generating. Based on the Qwen2.5-7B dense architecture, we build Uni-MoE-2.0-Omni from scratch through three core contributions: dynamic-capacity Mixture-of-Experts (MoE) design, a progressive training strategy enhanced with an iterative reinforcement strategy, and a carefully curated multimodal data matching technique. It is capable of omnimodal understanding, as well as generating images, text, and speech. Architecturally, our new MoE framework balances computational efficiency and capability for 10 cross-modal inputs using shared, routed, and null experts, while our Omni-Modality 3D RoPE ensures spatio-temporal cross-modality alignment in the self-attention layer. For training, following cross-modal pretraining, we use a progressive supervised fine-tuning strategy that activates modality-specific experts and is enhanced by balanced data composition and an iterative GSPO-DPO method to stabilise RL training and improve reasoning. Data-wise, the base model, trained on approximately 75B tokens of open-source multimodal data, is equipped with special speech and image generation tokens, allowing it to learn these generative tasks by conditioning its outputs on linguistic cues. Extensive evaluation across 85 benchmarks demonstrates that our model achieves SOTA or highly competitive performance against leading OLMs, surpassing Qwen2.5-Omni (trained with 1.2T tokens) on over 50 of 76 benchmarks. Key strengths include video understanding (+7% avg. of 8), omnimodallity understanding (+7% avg. of 4), and audiovisual reasoning (+4%). It also advances long-form speech processing (reducing WER by 4.2%) and leads in low-level image processing and controllable generation across 5 metrics. 

---
# Group-Aware Reinforcement Learning for Output Diversity in Large Language Models 

**Authors**: Oron Anschel, Alon Shoshan, Adam Botach, Shunit Haviv Hakimi, Asaf Gendler, Emanuel Ben Baruch, Nadav Bhonker, Igor Kviatkovsky, Manoj Aggarwal, Gerard Medioni  

**Link**: [PDF](https://arxiv.org/pdf/2511.12596)  

**Abstract**: Large Language Models (LLMs) often suffer from mode collapse, repeatedly generating the same few completions even when many valid answers exist, limiting their diversity across a wide range of tasks. We introduce Group-Aware Policy Optimization (GAPO), a simple extension of the recent and popular Group Relative Policy Optimization (GRPO) that computes rewards over the group as a whole. GAPO enables learning from the group-level properties such as diversity and coverage. We demonstrate GAPO using a frequency-aware reward function that encourages uniform sampling over valid LLM completions, and show that GAPO-trained models produce valid and more diverse model responses. Beyond this setup, GAPO generalizes to open-ended prompts and improves response diversity without compromising accuracy on standard LLM benchmarks (GSM8K, MATH, HumanEval, MMLU-Pro). Our code will be made publicly available. 

---
# MMWOZ: Building Multimodal Agent for Task-oriented Dialogue 

**Authors**: Pu-Hai Yang, Heyan Huang, Heng-Da Xu, Fanshu Sun, Xian-Ling Mao, Chaoxu Mu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12586)  

**Abstract**: Task-oriented dialogue systems have garnered significant attention due to their conversational ability to accomplish goals, such as booking airline tickets for users. Traditionally, task-oriented dialogue systems are conceptualized as intelligent agents that interact with users using natural language and have access to customized back-end APIs. However, in real-world scenarios, the widespread presence of front-end Graphical User Interfaces (GUIs) and the absence of customized back-end APIs create a significant gap for traditional task-oriented dialogue systems in practical applications. In this paper, to bridge the gap, we collect MMWOZ, a new multimodal dialogue dataset that is extended from MultiWOZ 2.3 dataset. Specifically, we begin by developing a web-style GUI to serve as the front-end. Next, we devise an automated script to convert the dialogue states and system actions from the original dataset into operation instructions for the GUI. Lastly, we collect snapshots of the web pages along with their corresponding operation instructions. In addition, we propose a novel multimodal model called MATE (Multimodal Agent for Task-oriEnted dialogue) as the baseline model for the MMWOZ dataset. Furthermore, we conduct comprehensive experimental analysis using MATE to investigate the construction of a practical multimodal agent for task-oriented dialogue. 

---
# Mitigating Length Bias in RLHF through a Causal Lens 

**Authors**: Hyeonji Kim, Sujeong Oh, Sanghack Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.12573)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is widely used to align large language models (LLMs) with human preferences. However, RLHF-trained reward models often exhibit length bias -- a systematic tendency to favor longer responses by conflating verbosity with quality. We propose a causal framework for analyzing and mitigating length bias in RLHF reward modeling. Central to our approach is a counterfactual data augmentation method that generates response pairs designed to isolate content quality from verbosity. These counterfactual examples are then used to train the reward model, enabling it to assess responses based on content quality independently of verbosity. Specifically, we construct (1) length-divergent pairs with similar content and (2) content-divergent pairs of similar length. Empirical evaluations show that our method reduces length bias in reward assignment and leads to more concise, content-focused outputs from the policy model. These findings demonstrate that the proposed approach effectively reduces length bias and improves the robustness and content sensitivity of reward modeling in RLHF pipelines. 

---
# TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction 

**Authors**: Jie Zhang, Bo Tang, Wanzi Shao, Wenqiang Wei, Jihao Zhao, Jianqing Zhu, Zhiyu li, Wen Xi, Zehao Lin, Feiyu Xiong, Yanchao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.12520)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves large language models by retrieving external knowledge, often truncated into smaller chunks due to the input context window, which leads to information loss, resulting in response hallucinations and broken reasoning chains. Moreover, traditional RAG retrieves unstructured knowledge, introducing irrelevant details that hinder accurate reasoning. To address these issues, we propose TAdaRAG, a novel RAG framework for on-the-fly task-adaptive knowledge graph construction from external sources. Specifically, we design an intent-driven routing mechanism to a domain-specific extraction template, followed by supervised fine-tuning and a reinforcement learning-based implicit extraction mechanism, ensuring concise, coherent, and non-redundant knowledge integration. Evaluations on six public benchmarks and a real-world business benchmark (NowNewsQA) across three backbone models demonstrate that TAdaRAG outperforms existing methods across diverse domains and long-text tasks, highlighting its strong generalization and practical effectiveness. 

---
# QA-Noun: Representing Nominal Semantics via Natural Language Question-Answer Pairs 

**Authors**: Maria Tseytlin, Paul Roit, Omri Abend, Ido Dagan, Ayal Klein  

**Link**: [PDF](https://arxiv.org/pdf/2511.12504)  

**Abstract**: Decomposing sentences into fine-grained meaning units is increasingly used to model semantic alignment. While QA-based semantic approaches have shown effectiveness for representing predicate-argument relations, they have so far left noun-centered semantics largely unaddressed. We introduce QA-Noun, a QA-based framework for capturing noun-centered semantic relations. QA-Noun defines nine question templates that cover both explicit syntactical and implicit contextual roles for nouns, producing interpretable QA pairs that complement verbal QA-SRL. We release detailed guidelines, a dataset of over 2,000 annotated noun mentions, and a trained model integrated with QA-SRL to yield a unified decomposition of sentence meaning into individual, highly fine-grained, facts. Evaluation shows that QA-Noun achieves near-complete coverage of AMR's noun arguments while surfacing additional contextually implied relations, and that combining QA-Noun with QA-SRL yields over 130\% higher granularity than recent fact-based decomposition methods such as FactScore and DecompScore. QA-Noun thus complements the broader QA-based semantic framework, forming a comprehensive and scalable approach to fine-grained semantic decomposition for cross-text alignment. 

---
# SGuard-v1: Safety Guardrail for Large Language Models 

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok  

**Link**: [PDF](https://arxiv.org/pdf/2511.12497)  

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety. 

---
# Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing 

**Authors**: Mengying Wang, Chenhui Ma, Ao Jiao, Tuo Liang, Pengjun Lu, Shrinidhi Hegde, Yu Yin, Evren Gurkan-Cavusoglu, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12472)  

**Abstract**: Large Language Models (LLMs) have greatly advanced knowledge graph question answering (KGQA), yet existing systems are typically optimized for returning highly relevant but predictable answers. A missing yet desired capacity is to exploit LLMs to suggest surprise and novel ("serendipitious") answers. In this paper, we formally define the serendipity-aware KGQA task and propose the SerenQA framework to evaluate LLMs' ability to uncover unexpected insights in scientific KGQA tasks. SerenQA includes a rigorous serendipity metric based on relevance, novelty, and surprise, along with an expert-annotated benchmark derived from the Clinical Knowledge Graph, focused on drug repurposing. Additionally, it features a structured evaluation pipeline encompassing three subtasks: knowledge retrieval, subgraph reasoning, and serendipity exploration. Our experiments reveal that while state-of-the-art LLMs perform well on retrieval, they still struggle to identify genuinely surprising and valuable discoveries, underscoring a significant room for future improvements. Our curated resources and extended version are released at: this https URL. 

---
# Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models 

**Authors**: Chenglong Wang, Yifu Huo, Yang Gan, Yongyu Mu, Qiaozhi He, Murun Yang, Bei Li, Chunliang Zhang, Tongran Liu, Anxiang Ma, Zhengtao Yu, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12464)  

**Abstract**: Previous methods evaluate reward models by testing them on a fixed pairwise ranking test set, but they typically do not provide performance information on each preference dimension. In this work, we address the evaluation challenge of reward models by probing preference representations. To confirm the effectiveness of this evaluation method, we construct a Multi-dimensional Reward Model Benchmark (MRMBench), a collection of six probing tasks for different preference dimensions. We design it to favor and encourage reward models that better capture preferences across different dimensions. Furthermore, we introduce an analysis method, inference-time probing, which identifies the dimensions used during the reward prediction and enhances its interpretability. Through extensive experiments, we find that MRMBench strongly correlates with the alignment performance of large language models (LLMs), making it a reliable reference for developing advanced reward models. Our analysis of MRMBench evaluation results reveals that reward models often struggle to capture preferences across multiple dimensions, highlighting the potential of multi-objective optimization in reward modeling. Additionally, our findings show that the proposed inference-time probing method offers a reliable metric for assessing the confidence of reward predictions, which ultimately improves the alignment of LLMs. 

---
# From Phonemes to Meaning: Evaluating Large Language Models on Tamil 

**Authors**: Jeyarajalingam Varsha, Menan Velayuthan, Sumirtha Karunakaran, Rasan Nivethiga, Kengatharaiyer Sarveswaran  

**Link**: [PDF](https://arxiv.org/pdf/2511.12387)  

**Abstract**: Large Language Models (LLMs) have shown strong generalization across tasks in high-resource languages; however, their linguistic competence in low-resource and morphologically rich languages such as Tamil remains largely unexplored. Existing multilingual benchmarks often rely on translated English datasets, failing to capture the linguistic and cultural nuances of the target language. To address this gap, we introduce ILAKKANAM, the first Tamil-specific linguistic evaluation benchmark manually curated using 820 questions from Sri Lankan school-level Tamil subject examination papers. Each question is annotated by trained linguists under five linguistic categories and a factual knowledge category, spanning Grades 1--13 to ensure broad linguistic coverage. We evaluate both closed-source and open-source LLMs using a standardized evaluation framework. Our results show that Gemini 2.5 achieves the highest overall performance, while open-source models lag behind, highlighting the gap in linguistic grounding. Category- and grade-wise analyses reveal that all models perform well on lower-grade questions but show a clear decline as linguistic complexity increases. Further, no strong correlation is observed between a model's overall performance and its ability to identify linguistic categories, suggesting that performance may be driven by exposure rather than genuine understanding. 

---
# Don't Think of the White Bear: Ironic Negation in Transformer Models Under Cognitive Load 

**Authors**: Logan Mann, Nayan Saxena, Sarah Tandon, Chenhao Sun, Savar Toteja, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12381)  

**Abstract**: Negation instructions such as 'do not mention $X$' can paradoxically increase the accessibility of $X$ in human thought, a phenomenon known as ironic rebound. Large language models (LLMs) face the same challenge: suppressing a concept requires internally activating it, which may prime rebound instead of avoidance. We investigated this tension with two experiments. \textbf{(1) Load \& content}: after a negation instruction, we vary distractor text (semantic, syntactic, repetition) and measure rebound strength. \textbf{(2) Polarity separation}: We test whether models distinguish neutral from negative framings of the same concept and whether this separation predicts rebound persistence. Results show that rebound consistently arises immediately after negation and intensifies with longer or semantic distractors, while repetition supports suppression. Stronger polarity separation correlates with more persistent rebound. Together, these findings, complemented by a circuit tracing analysis that identifies sparse middle-layer attention heads amplifying forbidden tokens while early layers suppress, link cognitive predictions of ironic rebound with mechanistic insights into long-context interference. To support future work, we release ReboundBench, a dataset of $5,000$ systematically varied negation prompts designed to probe rebound in LLMs. 

---
# Do LLMs and Humans Find the Same Questions Difficult? A Case Study on Japanese Quiz Answering 

**Authors**: Naoya Sugiura, Kosuke Yamada, Yasuhiro Ogawa, Katsuhiko Toyama, Ryohei Sasano  

**Link**: [PDF](https://arxiv.org/pdf/2511.12300)  

**Abstract**: LLMs have achieved performance that surpasses humans in many NLP tasks. However, it remains unclear whether problems that are difficult for humans are also difficult for LLMs. This study investigates how the difficulty of quizzes in a buzzer setting differs between LLMs and humans. Specifically, we first collect Japanese quiz data including questions, answers, and correct response rate of humans, then prompted LLMs to answer the quizzes under several settings, and compare their correct answer rate to that of humans from two analytical perspectives. The experimental results showed that, compared to humans, LLMs struggle more with quizzes whose correct answers are not covered by Wikipedia entries, and also have difficulty with questions that require numerical answers. 

---
# AugAbEx : Way Forward for Extractive Case Summarization 

**Authors**: Purnima Bindal, Vikas Kumar, Sagar Rathore, Vasudha Bhatnagar  

**Link**: [PDF](https://arxiv.org/pdf/2511.12290)  

**Abstract**: Summarization of legal judgments poses a heavy cognitive burden on law practitioners due to the complexity of the language, context-sensitive legal jargon, and the length of the document. Therefore, the automatic summarization of legal documents has attracted serious attention from natural language processing researchers. Since the abstractive summaries of legal documents generated by deep neural methods remain prone to the risk of misrepresenting nuanced legal jargon or overlooking key contextual details, we envisage a rising trend toward the use of extractive case summarizers.
Given the high cost of human annotation for gold standard extractive summaries, we engineer a light and transparent pipeline that leverages existing abstractive gold standard summaries to create the corresponding extractive gold standard versions. The approach ensures that the experts` opinions ensconced in the original gold standard abstractive summaries are carried over to the transformed extractive summaries. We aim to augment seven existing case summarization datasets, which include abstractive summaries, by incorporating corresponding extractive summaries and create an enriched data resource for case summarization research community. To ensure the quality of the augmented extractive summaries, we perform an extensive comparative evaluation with the original abstractive gold standard summaries covering structural, lexical, and semantic dimensions. We also compare the domain-level information of the two summaries. We commit to release the augmented datasets in the public domain for use by the research community and believe that the resource will offer opportunities to advance the field of automatic summarization of legal documents. 

---
# Cmprsr: Abstractive Token-Level Question-Agnostic Prompt Compressor 

**Authors**: Ivan Zakazov, Alexander Sharipov, Berke Argin, Oussama Gabouj, Kamel Charaf, Alexi Semiz, Lorenzo Drudi, Nicolas Baldwin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2511.12281)  

**Abstract**: Motivated by the high costs of using black-box Large Language Models (LLMs), we introduce a novel prompt compression paradigm, under which we use smaller LLMs to compress inputs for the larger ones. We present the first comprehensive LLM-as-a-compressor benchmark spanning 25 open- and closed-source models, which reveals significant disparity in models' compression ability in terms of (i) preserving semantically important information (ii) following the user-provided compression rate (CR). We further improve the performance of gpt-4.1-mini, the best overall vanilla compressor, with Textgrad-based compression meta-prompt optimization. We also identify the most promising open-source vanilla LLM - Qwen3-4B - and post-train it with a combination of supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO), pursuing the dual objective of CR adherence and maximizing the downstream task performance. We call the resulting model Cmprsr and demonstrate its superiority over both extractive and vanilla abstractive compression across the entire range of compression rates on lengthy inputs from MeetingBank and LongBench as well as short prompts from GSM8k. The latter highlights Cmprsr's generalizability across varying input lengths and domains. Moreover, Cmprsr closely follows the requested compression rate, offering fine control over the cost-quality trade-off. 

---
# ViConBERT: Context-Gloss Aligned Vietnamese Word Embedding for Polysemous and Sense-Aware Representations 

**Authors**: Khang T. Huynh, Dung H. Nguyen, Binh T. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.12249)  

**Abstract**: Recent advances in contextualized word embeddings have greatly improved semantic tasks such as Word Sense Disambiguation (WSD) and contextual similarity, but most progress has been limited to high-resource languages like English. Vietnamese, in contrast, still lacks robust models and evaluation resources for fine-grained semantic understanding. In this paper, we present ViConBERT, a novel framework for learning Vietnamese contextualized embeddings that integrates contrastive learning (SimCLR) and gloss-based distillation to better capture word meaning. We also introduce ViConWSD, the first large-scale synthetic dataset for evaluating semantic understanding in Vietnamese, covering both WSD and contextual similarity. Experimental results show that ViConBERT outperforms strong baselines on WSD (F1 = 0.87) and achieves competitive performance on ViCon (AP = 0.88) and ViSim-400 (Spearman's rho = 0.60), demonstrating its effectiveness in modeling both discrete senses and graded semantic relations. Our code, models, and data are available at this https URL 

---
# Consistency Is the Key: Detecting Hallucinations in LLM Generated Text By Checking Inconsistencies About Key Facts 

**Authors**: Raavi Gupta, Pranav Hari Panicker, Sumit Bhatia, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.12236)  

**Abstract**: Large language models (LLMs), despite their remarkable text generation capabilities, often hallucinate and generate text that is factually incorrect and not grounded in real-world knowledge. This poses serious risks in domains like healthcare, finance, and customer support. A typical way to use LLMs is via the APIs provided by LLM vendors where there is no access to model weights or options to fine-tune the model. Existing methods to detect hallucinations in such settings where the model access is restricted or constrained by resources typically require making multiple LLM API calls, increasing latency and API cost. We introduce CONFACTCHECK, an efficient hallucination detection approach that does not leverage any external knowledge base and works on the simple intuition that responses to factual probes within the generated text should be consistent within a single LLM and across different LLMs. Rigorous empirical evaluation on multiple datasets that cover both the generation of factual texts and the open generation shows that CONFACTCHECK can detect hallucinated facts efficiently using fewer resources and achieves higher accuracy scores compared to existing baselines that operate under similar conditions. Our code is available here. 

---
# MME-RAG: Multi-Manager-Expert Retrieval-Augmented Generation for Fine-Grained Entity Recognition in Task-Oriented Dialogues 

**Authors**: Liang Xue, Haoyu Liu, Yajun Tian, Xinyu Zhong, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12213)  

**Abstract**: Fine-grained entity recognition is crucial for reasoning and decision-making in task-oriented dialogues, yet current large language models (LLMs) continue to face challenges in domain adaptation and retrieval controllability. We introduce MME-RAG, a Multi-Manager-Expert Retrieval-Augmented Generation framework that decomposes entity recognition into two coordinated stages: type-level judgment by lightweight managers and span-level extraction by specialized experts. Each expert is supported by a KeyInfo retriever that injects semantically aligned, few-shot exemplars during inference, enabling precise and domain-adaptive extraction without additional training. Experiments on CrossNER, MIT-Movie, MIT-Restaurant, and our newly constructed multi-domain customer-service dataset demonstrate that MME-RAG performs better than recent baselines in most domains. Ablation studies further show that both the hierarchical decomposition and KeyInfo-guided retrieval are key drivers of robustness and cross-domain generalization, establishing MME-RAG as a scalable and interpretable solution for adaptive dialogue understanding. 

---
# CriticSearch: Fine-Grained Credit Assignment for Search Agents via a Retrospective Critic 

**Authors**: Yaocheng Zhang, Haohuan Huang, Zijun Song, Yuanheng Zhu, Qichao Zhang, Zijie Zhao, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.12159)  

**Abstract**: Tool-Integrated Reasoning (TIR) with search engines enables large language models to iteratively retrieve up-to-date external knowledge, enhancing adaptability and generalization in complex question-answering tasks. However, existing search agent pipelines typically depend on reinforcement learning based optimization, which often suffers from sparse outcome rewards, leading to inefficient exploration and unstable training. We introduce CriticSearch, a fine-grained credit-assignment framework that supplies dense, turn-level feedback via a retrospective critic mechanism. During training, a frozen, asymmetric critique LLM retrospectively evaluates each turn using privileged information from the full trajectory and gold answers, converting these assessments into stable, dense rewards that guide policy improvement. Experimental results across diverse multi-hop reasoning benchmarks demonstrate that CriticSearch consistently outperforms existing baselines, achieving faster convergence, improved training stability, and higher performance. 

---
# Seeing is Believing: Rich-Context Hallucination Detection for MLLMs via Backward Visual Grounding 

**Authors**: Pinxue Guo, Chongruo Wu, Xinyu Zhou, Lingyi Hong, Zhaoyu Chen, Jinglun Li, Kaixun Jiang, Sen-ching Samson Cheung, Wei Zhang, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12140)  

**Abstract**: Multimodal Large Language Models (MLLMs) have unlocked powerful cross-modal capabilities, but still significantly suffer from hallucinations. As such, accurate detection of hallucinations in MLLMs is imperative for ensuring their reliability in practical applications. To this end, guided by the principle of "Seeing is Believing", we introduce VBackChecker, a novel reference-free hallucination detection framework that verifies the consistency of MLLMgenerated responses with visual inputs, by leveraging a pixellevel Grounding LLM equipped with reasoning and referring segmentation capabilities. This reference-free framework not only effectively handles rich-context scenarios, but also offers interpretability. To facilitate this, an innovative pipeline is accordingly designed for generating instruction-tuning data (R-Instruct), featuring rich-context descriptions, grounding masks, and hard negative samples. We further establish R^2 -HalBench, a new hallucination benchmark for MLLMs, which, unlike previous benchmarks, encompasses real-world, rich-context descriptions from 18 MLLMs with high-quality annotations, spanning diverse object-, attribute, and relationship-level details. VBackChecker outperforms prior complex frameworks and achieves state-of-the-art performance on R^2 -HalBench, even rivaling GPT-4o's capabilities in hallucination detection. It also surpasses prior methods in the pixel-level grounding task, achieving over a 10% improvement. All codes, data, and models are available at this https URL. 

---
# AI-Salesman: Towards Reliable Large Language Model Driven Telemarketing 

**Authors**: Qingyu Zhang, Chunlei Xin, Xuanang Chen, Yaojie Lu, Hongyu Lin, Xianpei Han, Le Sun, Qing Ye, Qianlong Xie, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.12133)  

**Abstract**: Goal-driven persuasive dialogue, exemplified by applications like telemarketing, requires sophisticated multi-turn planning and strict factual faithfulness, which remains a significant challenge for even state-of-the-art Large Language Models (LLMs). A lack of task-specific data often limits previous works, and direct LLM application suffers from strategic brittleness and factual hallucination. In this paper, we first construct and release TeleSalesCorpus, the first real-world-grounded dialogue dataset for this domain. We then propose AI-Salesman, a novel framework featuring a dual-stage architecture. For the training stage, we design a Bayesian-supervised reinforcement learning algorithm that learns robust sales strategies from noisy dialogues. For the inference stage, we introduce the Dynamic Outline-Guided Agent (DOGA), which leverages a pre-built script library to provide dynamic, turn-by-turn strategic guidance. Moreover, we design a comprehensive evaluation framework that combines fine-grained metrics for key sales skills with the LLM-as-a-Judge paradigm. Experimental results demonstrate that our proposed AI-Salesman significantly outperforms baseline models in both automatic metrics and comprehensive human evaluations, showcasing its effectiveness in complex persuasive scenarios. 

---
# PRISM of Opinions: A Persona-Reasoned Multimodal Framework for User-centric Conversational Stance Detection 

**Authors**: Bingbing Wang, Zhixin Bai, Zhengda Jin, Zihan Wang, Xintong Song, Jingjie Lin, Sixuan Li, Jing Li, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12130)  

**Abstract**: The rapid proliferation of multimodal social media content has driven research in Multimodal Conversational Stance Detection (MCSD), which aims to interpret users' attitudes toward specific targets within complex discussions. However, existing studies remain limited by: **1) pseudo-multimodality**, where visual cues appear only in source posts while comments are treated as text-only, misaligning with real-world multimodal interactions; and **2) user homogeneity**, where diverse users are treated uniformly, neglecting personal traits that shape stance expression. To address these issues, we introduce **U-MStance**, the first user-centric MCSD dataset, containing over 40k annotated comments across six real-world targets. We further propose **PRISM**, a **P**ersona-**R**easoned mult**I**modal **S**tance **M**odel for MCSD. PRISM first derives longitudinal user personas from historical posts and comments to capture individual traits, then aligns textual and visual cues within conversational context via Chain-of-Thought to bridge semantic and pragmatic gaps across modalities. Finally, a mutual task reinforcement mechanism is employed to jointly optimize stance detection and stance-aware response generation for bidirectional knowledge transfer. Experiments on U-MStance demonstrate that PRISM yields significant gains over strong baselines, underscoring the effectiveness of user-centric and context-grounded multimodal reasoning for realistic stance understanding. 

---
# LLMLagBench: Identifying Temporal Training Boundaries in Large Language Models 

**Authors**: Piotr Pęzik, Konrad Kaczyński, Maria Szymańska, Filip Żarnecki, Zuzanna Deckert, Jakub Kwiatkowski, Wojciech Janowski  

**Link**: [PDF](https://arxiv.org/pdf/2511.12116)  

**Abstract**: Large Language Models (LLMs) are pretrained on textual data up to a specific temporal cutoff. This creates a strict knowledge boundary beyond which models cannot provide accurate information without querying external sources. More subtly, when this limitation is unknown or ignored, LLMs may inadvertently blend outdated time-sensitive information with general knowledge during reasoning tasks, potentially compromising response accuracy. We introduce LLMLagBench, an LLM freshness benchmark, as a systematic approach for identifying the earliest probable temporal boundaries of an LLM's training data by evaluating its knowledge of recent events. We then apply this benchmark to evaluate a large set of LLMs, including models with both explicitly declared and undeclared training cutoffs. The reliability of the benchmark is assessed by manual validation and comparison with publicly released information about LLM pretraining. 

---
# Exploring Parameter-Efficient Fine-Tuning and Backtranslation for the WMT 25 General Translation Task 

**Authors**: Felipe Fujita, Hideyuki Takada  

**Link**: [PDF](https://arxiv.org/pdf/2511.12109)  

**Abstract**: In this paper, we explore the effectiveness of combining fine-tuning and backtranslation on a small Japanese corpus for neural machine translation. Starting from a baseline English{\textrightarrow}Japanese model (COMET = 0.460), we first apply backtranslation (BT) using synthetic data generated from monolingual Japanese corpora, yielding a modest increase (COMET = 0.468). Next, we fine-tune (FT) the model on a genuine small parallel dataset drawn from diverse Japanese news and literary corpora, achieving a substantial jump to COMET = 0.589 when using Mistral 7B. Finally, we integrate both backtranslation and fine-tuning{ -- }first augmenting the small dataset with BT generated examples, then adapting via FT{ -- }which further boosts performance to COMET = 0.597. These results demonstrate that, even with limited training data, the synergistic use of backtranslation and targeted fine-tuning on Japanese corpora can significantly enhance translation quality, outperforming each technique in isolation. This approach offers a lightweight yet powerful strategy for improving low-resource language pairs. 

---
# CURE: Cultural Understanding and Reasoning Evaluation - A Framework for "Thick" Culture Alignment Evaluation in LLMs 

**Authors**: Truong Vo, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2511.12014)  

**Abstract**: Large language models (LLMs) are increasingly deployed in culturally diverse environments, yet existing evaluations of cultural competence remain limited. Existing methods focus on de-contextualized correctness or forced-choice judgments, overlooking the need for cultural understanding and reasoning required for appropriate responses. To address this gap, we introduce a set of benchmarks that, instead of directly probing abstract norms or isolated statements, present models with realistic situational contexts that require culturally grounded reasoning. In addition to the standard Exact Match metric, we introduce four complementary metrics (Coverage, Specificity, Connotation, and Coherence) to capture different dimensions of model's response quality. Empirical analysis across frontier models reveals that thin evaluation systematically overestimates cultural competence and produces unstable assessments with high variance. In contrast, thick evaluation exposes differences in reasoning depth, reduces variance, and provides more stable, interpretable signals of cultural understanding. 

---
# Critical or Compliant? The Double-Edged Sword of Reasoning in Chain-of-Thought Explanations 

**Authors**: Eunkyu Park, Wesley Hanwen Deng, Vasudha Varadarajan, Mingxi Yan, Gunhee Kim, Maarten Sap, Motahhare Eslami  

**Link**: [PDF](https://arxiv.org/pdf/2511.12001)  

**Abstract**: Explanations are often promoted as tools for transparency, but they can also foster confirmation bias; users may assume reasoning is correct whenever outputs appear acceptable. We study this double-edged role of Chain-of-Thought (CoT) explanations in multimodal moral scenarios by systematically perturbing reasoning chains and manipulating delivery tones. Specifically, we analyze reasoning errors in vision language models (VLMs) and how they impact user trust and the ability to detect errors. Our findings reveal two key effects: (1) users often equate trust with outcome agreement, sustaining reliance even when reasoning is flawed, and (2) the confident tone suppresses error detection while maintaining reliance, showing that delivery styles can override correctness. These results highlight how CoT explanations can simultaneously clarify and mislead, underscoring the need for NLP systems to provide explanations that encourage scrutiny and critical thinking rather than blind trust. All code will be released publicly. 

---
# A Reasoning Paradigm for Named Entity Recognition 

**Authors**: Hui Huang, Yanping Chen, Ruizhang Huang, Chuan Lin, Yongbin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.11978)  

**Abstract**: Generative LLMs typically improve Named Entity Recognition (NER) performance through instruction tuning. They excel at generating entities by semantic pattern matching but lack an explicit, verifiable reasoning mechanism. This "cognitive shortcutting" leads to suboptimal performance and brittle generalization, especially in zero-shot and lowresource scenarios where reasoning from limited contextual cues is crucial. To address this issue, a reasoning framework is proposed for NER, which shifts the extraction paradigm from implicit pattern matching to explicit reasoning. This framework consists of three stages: Chain of Thought (CoT) generation, CoT tuning, and reasoning enhancement. First, a dataset annotated with NER-oriented CoTs is generated, which contain task-relevant reasoning chains. Then, they are used to tune the NER model to generate coherent rationales before deriving the final answer. Finally, a reasoning enhancement stage is implemented to optimize the reasoning process using a comprehensive reward signal. This stage ensures explicit and verifiable extractions. Experiments show that ReasoningNER demonstrates impressive cognitive ability in the NER task, achieving competitive performance. In zero-shot settings, it achieves state-of-the-art (SOTA) performance, outperforming GPT-4 by 12.3 percentage points on the F1 score. Analytical results also demonstrate its great potential to advance research in reasoningoriented information extraction. Our codes are available at this https URL. 

---
# On the Entropy Calibration of Language Models 

**Authors**: Steven Cao, Gregory Valiant, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11966)  

**Abstract**: We study the problem of entropy calibration, which asks whether a language model's entropy over generations matches its log loss on human text. Past work found that models are miscalibrated, with entropy per step increasing (and text quality decreasing) as generations grow longer. This error accumulation is a fundamental problem in autoregressive models, and the standard solution is to truncate the distribution, which improves text quality at the cost of diversity. In this paper, we ask: is miscalibration likely to improve with scale, and is it theoretically possible to calibrate without tradeoffs? To build intuition, we first study a simplified theoretical setting to characterize the scaling behavior of miscalibration with respect to dataset size. We find that the scaling behavior depends on the power law exponent of the data distribution -- in particular, for a power law exponent close to 1, the scaling exponent is close to 0, meaning that miscalibration improves very slowly with scale. Next, we measure miscalibration empirically in language models ranging from 0.5B to 70B parameters. We find that the observed scaling behavior is similar to what is predicted by the simplified setting: our fitted scaling exponents for text are close to 0, meaning that larger models accumulate error at a similar rate as smaller ones. This scaling (or, lack thereof) provides one explanation for why we sample from larger models with similar amounts of truncation as smaller models, even though the larger models are of higher quality. However, truncation is not a satisfying solution because it comes at the cost of increased log loss. In theory, is it even possible to reduce entropy while preserving log loss? We prove that it is possible, if we assume access to a black box which can fit models to predict the future entropy of text. 

---
# Improving LLM's Attachment to External Knowledge In Dialogue Generation Tasks Through Entity Anonymization 

**Authors**: Hadi Sheikhi, Chenyang Huang, Osmar R. Zaïane  

**Link**: [PDF](https://arxiv.org/pdf/2511.11946)  

**Abstract**: Knowledge graph-based dialogue generation (KG-DG) is a challenging task requiring models to effectively incorporate external knowledge into conversational responses. While large language models (LLMs) have achieved impressive results across various NLP tasks, their ability to utilize external knowledge in KG-DG remains under-explored. We observe that LLMs often rely on internal knowledge, leading to detachment from provided knowledge graphs, even when they are given a flawlessly retrieved knowledge graph. First, we introduce LLM-KAT, an evaluation procedure for measuring knowledge attachment in generated responses. Second, we propose a simple yet effective entity anonymization technique to encourage LLMs to better leverage external knowledge. Experiments on the OpenDialKG dataset demonstrate that our approach improves LLMs' attachment on external knowledge. 

---
# InData: Towards Secure Multi-Step, Tool-Based Data Analysis 

**Authors**: Karthikeyan K, Raghuveer Thirukovalluru, Bhuwan Dhingra, David Edwin Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11933)  

**Abstract**: Large language model agents for data analysis typically generate and execute code directly on databases. However, when applied to sensitive data, this approach poses significant security risks. To address this issue, we propose a security-motivated alternative: restrict LLMs from direct code generation and data access, and require them to interact with data exclusively through a predefined set of secure, verified tools. Although recent tool-use benchmarks exist, they primarily target tool selection and simple execution rather than the compositional, multi-step reasoning needed for complex data analysis. To reduce this gap, we introduce Indirect Data Engagement (InData), a dataset designed to assess LLMs' multi-step tool-based reasoning ability. InData includes data analysis questions at three difficulty levels--Easy, Medium, and Hard--capturing increasing reasoning complexity. We benchmark 15 open-source LLMs on InData and find that while large models (e.g., gpt-oss-120b) achieve high accuracy on Easy tasks (97.3%), performance drops sharply on Hard tasks (69.6%). These results show that current LLMs still lack robust multi-step tool-based reasoning ability. With InData, we take a step toward enabling the development and evaluation of LLMs with stronger multi-step tool-use capabilities. We will publicly release the dataset and code. 

---
# Additive Large Language Models for Semi-Structured Text 

**Authors**: Karthikeyan K, Raghuveer Thirukovalluru, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11922)  

**Abstract**: Large Language Models have advanced clinical text classification, but their opaque predictions remain a critical barrier to practical adoption in research and clinical settings where investigators and physicians need to understand which parts of a patient's record drive risk signals. To address this challenge, we introduce \textbf{CALM}, short for \textbf{Classification with Additive Large Language Models}, an interpretable framework for semi-structured text where inputs are composed of semantically meaningful components, such as sections of an admission note or question-answer fields from an intake form. CALM predicts outcomes as the additive sum of each component's contribution, making these contributions part of the forward computation itself and enabling faithful explanations at both the patient and population level. The additive structure also enables clear visualizations, such as component-level risk curves similar to those used in generalized additive models, making the learned relationships easier to inspect and communicate. Although CALM expects semi-structured inputs, many clinical documents already have this form, and similar structure can often be automatically extracted from free-text notes. CALM achieves performance comparable to conventional LLM classifiers while improving trust, supporting quality-assurance checks, and revealing clinically meaningful patterns during model development and auditing. 

---
# Context-Emotion Aware Therapeutic Dialogue Generation: A Multi-component Reinforcement Learning Approach to Language Models for Mental Health Support 

**Authors**: Eric Hua Qing Zhang, Julia Ive  

**Link**: [PDF](https://arxiv.org/pdf/2511.11884)  

**Abstract**: Mental health illness represents a substantial global socioeconomic burden, with COVID-19 further exacerbating accessibility challenges and driving increased demand for telehealth mental health support. While large language models (LLMs) offer promising solutions through 24/7 availability and non-judgmental interactions, pre-trained models often lack the contextual and emotional awareness necessary for appropriate therapeutic responses. This paper investigated the application of supervised fine-tuning (SFT) and reinforcement learning (RL) techniques to enhance GPT-2's capacity for therapeutic dialogue generation. The methodology restructured input formats to enable simultaneous processing of contextual information and emotional states alongside user input, employing a multi-component reward function that aligned model outputs with professional therapist responses and annotated emotions. Results demonstrated improvements through reinforcement learning over baseline GPT-2 across multiple evaluation metrics: BLEU (0.0111), ROUGE-1 (0.1397), ROUGE-2 (0.0213), ROUGE-L (0.1317), and METEOR (0.0581). LLM evaluation confirmed high contextual relevance and professionalism, while reinforcement learning achieved 99.34% emotion accuracy compared to 66.96% for baseline GPT-2. These findings demonstrate reinforcement learning's effectiveness in developing therapeutic dialogue systems that can serve as valuable assistive tools for therapists while maintaining essential human clinical oversight. 

---
# ClinStructor: AI-Powered Structuring of Unstructured Clinical Texts 

**Authors**: Karthikeyan K, Raghuveer Thirukovalluru, David Carlson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11883)  

**Abstract**: Clinical notes contain valuable, context-rich information, but their unstructured format introduces several challenges, including unintended biases (e.g., gender or racial bias), and poor generalization across clinical settings (e.g., models trained on one EHR system may perform poorly on another due to format differences) and poor interpretability. To address these issues, we present ClinStructor, a pipeline that leverages large language models (LLMs) to convert clinical free-text into structured, task-specific question-answer pairs prior to predictive modeling. Our method substantially enhances transparency and controllability and only leads to a modest reduction in predictive performance (a 2-3% drop in AUC), compared to direct fine-tuning, on the ICU mortality prediction task. ClinStructor lays a strong foundation for building reliable, interpretable, and generalizable machine learning models in clinical environments. 

---
# MedPT: A Massive Medical Question Answering Dataset for Brazilian-Portuguese Speakers 

**Authors**: Fernanda Bufon Färber, Iago Alves Brito, Julia Soares Dollis, Pedro Schindler Freire Brasil Ribeiro, Rafael Teixeira Sousa, Arlindo Rodrigues Galvão Filho  

**Link**: [PDF](https://arxiv.org/pdf/2511.11878)  

**Abstract**: While large language models (LLMs) show transformative potential in healthcare, their development remains focused on high-resource languages, creating a critical barrier for others as simple translation fails to capture unique clinical and cultural nuances, such as endemic diseases. To address this, we introduce MedPT, the first large-scale, real-world corpus for Brazilian Portuguese, comprising 384,095 authentic question-answer pairs from patient-doctor interactions. The dataset underwent a meticulous multi-stage curation protocol, using a hybrid quantitative-qualitative analysis to filter noise and contextually enrich thousands of ambiguous queries. We further augmented the corpus via LLM-driven annotation, classifying questions into seven semantic types to capture user intent. Our analysis reveals its thematic breadth (3,200 topics) and unique linguistic properties, like the natural asymmetry in patient-doctor communication. To validate its utility, we benchmark a medical specialty routing task: fine-tuning a 1.7B parameter model achieves an outstanding 94\% F1-score on a 20-class setup. Furthermore, our qualitative error analysis shows misclassifications are not random but reflect genuine clinical ambiguities (e.g., between comorbid conditions), proving the dataset's deep semantic richness. We publicly release MedPT to foster the development of more equitable, accurate, and culturally-aware medical technologies for the Portuguese-speaking world. 

---
# Identifying Imaging Follow-Up in Radiology Reports: A Comparative Analysis of Traditional ML and LLM Approaches 

**Authors**: Namu Park, Giridhar Kaushik Ramachandran, Kevin Lybarger, Fei Xia, Ozlem Uzuner, Meliha Yetisgen, Martin Gunn  

**Link**: [PDF](https://arxiv.org/pdf/2511.11867)  

**Abstract**: Large language models (LLMs) have shown considerable promise in clinical natural language processing, yet few domain-specific datasets exist to rigorously evaluate their performance on radiology tasks. In this work, we introduce an annotated corpus of 6,393 radiology reports from 586 patients, each labeled for follow-up imaging status, to support the development and benchmarking of follow-up adherence detection systems. Using this corpus, we systematically compared traditional machine-learning classifiers, including logistic regression (LR), support vector machines (SVM), Longformer, and a fully fine-tuned Llama3-8B-Instruct, with recent generative LLMs. To evaluate generative LLMs, we tested GPT-4o and the open-source GPT-OSS-20B under two configurations: a baseline (Base) and a task-optimized (Advanced) setting that focused inputs on metadata, recommendation sentences, and their surrounding context. A refined prompt for GPT-OSS-20B further improved reasoning accuracy. Performance was assessed using precision, recall, and F1 scores with 95% confidence intervals estimated via non-parametric bootstrapping. Inter-annotator agreement was high (F1 = 0.846). GPT-4o (Advanced) achieved the best performance (F1 = 0.832), followed closely by GPT-OSS-20B (Advanced; F1 = 0.828). LR and SVM also performed strongly (F1 = 0.776 and 0.775), underscoring that while LLMs approach human-level agreement through prompt optimization, interpretable and resource-efficient models remain valuable baselines. 

---
# Three Stage Narrative Analysis; Plot-Sentiment Breakdown, Structure Learning and Concept Detection 

**Authors**: Taimur Khan, Ramoza Ahsan, Mohib Hameed  

**Link**: [PDF](https://arxiv.org/pdf/2511.11857)  

**Abstract**: Story understanding and analysis have long been challenging areas within Natural Language Understanding. Automated narrative analysis requires deep computational semantic representations along with syntactic processing. Moreover, the large volume of narrative data demands automated semantic analysis and computational learning rather than manual analytical approaches. In this paper, we propose a framework that analyzes the sentiment arcs of movie scripts and performs extended analysis related to the context of the characters involved. The framework enables the extraction of high-level and low-level concepts conveyed through the narrative. Using dictionary-based sentiment analysis, our approach applies a custom lexicon built with the LabMTsimple storylab module. The custom lexicon is based on the Valence, Arousal, and Dominance scores from the NRC-VAD dataset. Furthermore, the framework advances the analysis by clustering similar sentiment plots using Wards hierarchical clustering technique. Experimental evaluation on a movie dataset shows that the resulting analysis is helpful to consumers and readers when selecting a narrative or story. 

---
# Towards Autoformalization of LLM-generated Outputs for Requirement Verification 

**Authors**: Mihir Gupte, Ramesh S  

**Link**: [PDF](https://arxiv.org/pdf/2511.11829)  

**Abstract**: Autoformalization, the process of translating informal statements into formal logic, has gained renewed interest with the emergence of powerful Large Language Models (LLMs). While LLMs show promise in generating structured outputs from natural language (NL), such as Gherkin Scenarios from NL feature requirements, there's currently no formal method to verify if these outputs are accurate. This paper takes a preliminary step toward addressing this gap by exploring the use of a simple LLM-based autoformalizer to verify LLM-generated outputs against a small set of natural language requirements. We conducted two distinct experiments. In the first one, the autoformalizer successfully identified that two differently-worded NL requirements were logically equivalent, demonstrating the pipeline's potential for consistency checks. In the second, the autoformalizer was used to identify a logical inconsistency between a given NL requirement and an LLM-generated output, highlighting its utility as a formal verification tool. Our findings, while limited, suggest that autoformalization holds significant potential for ensuring the fidelity and logical consistency of LLM-generated outputs, laying a crucial foundation for future, more extensive studies into this novel application. 

---
# Scaling Open-Weight Large Language Models for Hydropower Regulatory Information Extraction: A Systematic Analysis 

**Authors**: Hong-Jun Yoon, Faisal Ashraf, Thomas A. Ruggles, Debjani Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.11821)  

**Abstract**: Information extraction from regulatory documents using large language models presents critical trade-offs between performance and computational resources. We evaluated seven open-weight models (0.6B-70B parameters) on hydropower licensing documentation to provide empirical deployment guidance.
Our analysis identified a pronounced 14B parameter threshold where validation methods transition from ineffective (F1 $<$ 0.15) to viable (F1 = 0.64). Consumer-deployable models achieve 64\% F1 through appropriate validation, while smaller models plateau at 51\%. Large-scale models approach 77\% F1 but require enterprise infrastructure.
We identified systematic hallucination patterns where perfect recall indicates extraction failure rather than success in smaller models. Our findings establish the first comprehensive resource-performance mapping for open-weight information extraction in regulatory contexts, enabling evidence-based model selection.
These results provide immediate value for hydropower compliance while contributing insights into parameter scaling effects that generalize across information extraction tasks. 

---
# On the Notion that Language Models Reason 

**Authors**: Bertram Højer  

**Link**: [PDF](https://arxiv.org/pdf/2511.11810)  

**Abstract**: Language models (LMs) are said to be exhibiting reasoning, but what does this entail? We assess definitions of reasoning and how key papers in the field of natural language processing (NLP) use the notion and argue that the definitions provided are not consistent with how LMs are trained, process information, and generate new tokens. To illustrate this incommensurability we assume the view that transformer-based LMs implement an \textit{implicit} finite-order Markov kernel mapping contexts to conditional token distributions. In this view, reasoning-like outputs correspond to statistical regularities and approximate statistical invariances in the learned kernel rather than the implementation of explicit logical mechanisms. This view is illustrative of the claim that LMs are "statistical pattern matchers"" and not genuine reasoners and provides a perspective that clarifies why reasoning-like outputs arise in LMs without any guarantees of logical consistency. This distinction is fundamental to how epistemic uncertainty is evaluated in LMs. We invite a discussion on the importance of how the computational processes of the systems we build and analyze in NLP research are described. 

---
# MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents via Model, Context, and Interactive Scaling 

**Authors**: MiroMind Team, Song Bai, Lidong Bing, Carson Chen, Guanzheng Chen, Yuntao Chen, Zhe Chen, Ziyi Chen, Jifeng Dai, Xuan Dong, Yue Deng, Yunjie Fu, Junqi Ge, Chenxia Han, Tammy Huang, Zhenhang Huang, Jerry Jiao, Shilei Jiang, Tianyu Jiao, Xiaoqi Jian, Lei Lei, Ruilin Li, Ryan Luo, Tiantong Li, Xiang Lin, Ziyuan Liu, Zhiqi Li, Jie Ni, Qiang Ren, Pax Sun, Shiqian Su, Chenxin Tao, Bin Wang, Hellen Wang, Haonan Wang, James Wang, Jin Wang, Jojo Wang, Letian Wang, Shizun Wang, Weizhi Wang, Zixuan Wang, Jinfan Xu, Sen Xing, Chenyu Yang, Hai Ye, Jiaheng Yu, Yue Yu, Muyan Zhong, Tianchen Zhao, Xizhou Zhu, Yanpeng Zhou, Yifan Zhang, Zhi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.11793)  

**Abstract**: We present MiroThinker v1.0, an open-source research agent designed to advance tool-augmented reasoning and information-seeking capabilities. Unlike previous agents that only scale up model size or context length, MiroThinker explores interaction scaling at the model level, systematically training the model to handle deeper and more frequent agent-environment interactions as a third dimension of performance improvement. Unlike LLM test-time scaling, which operates in isolation and risks degradation with longer reasoning chains, interactive scaling leverages environment feedback and external information acquisition to correct errors and refine trajectories. Through reinforcement learning, the model achieves efficient interaction scaling: with a 256K context window, it can perform up to 600 tool calls per task, enabling sustained multi-turn reasoning and complex real-world research workflows. Across four representative benchmarks-GAIA, HLE, BrowseComp, and BrowseComp-ZH-the 72B variant achieves up to 81.9%, 37.7%, 47.1%, and 55.6% accuracy respectively, surpassing previous open-source agents and approaching commercial counterparts such as GPT-5-high. Our analysis reveals that MiroThinker benefits from interactive scaling consistently: research performance improves predictably as the model engages in deeper and more frequent agent-environment interactions, demonstrating that interaction depth exhibits scaling behaviors analogous to model size and context length. These findings establish interaction scaling as a third critical dimension for building next-generation open research agents, complementing model capacity and context windows. 

---
# TimeStampEval: A Simple LLM Eval and a Little Fuzzy Matching Trick to Improve Search Accuracy 

**Authors**: James McCammon  

**Link**: [PDF](https://arxiv.org/pdf/2511.11594)  

**Abstract**: Traditional fuzzy matching often fails when searching for quotes that are semantically identical but syntactically different across documents-a common issue when aligning official written records with speech-to-text transcripts. We introduce TimeStampEval, a benchmark for retrieving precise millisecond timestamps from long transcripts given non-verbatim quotes. Our simple two-stage method dramatically improves retrieval accuracy while cutting inference costs by over 90%. The motivating use case is an automated long-form podcast that assembles Congressional Record clips into AI-hosted narration. The technical challenge: given a sentence-timestamped transcript and a target quote that may differ due to transcription or editorial drift, return exact start and end boundaries. Standard algorithms handle verbatim text but break under fuzzier variants. Evaluating six modern LLMs on a 2,800-sentence (120k-token) transcript revealed four key findings. (1) Prompt design matters more than model choice: placing the query before the transcript and using compact formatting improved accuracy by 3-20 points while reducing token count by 30-40%. (2) Off-by-one errors form a distinct category, showing models understand the task but misplace boundaries. (3) A modest reasoning budget (600-850 tokens) raises accuracy from 37% to 77% for weak setups and to above 90% for strong ones. (4) Our "Assisted Fuzzy" approach-RapidFuzz pre-filtering followed by LLM verification on short snippets-improves fuzzy match accuracy by up to 50 points while halving latency and reducing cost per correct result by up to 96%. Extended tests on ten transcripts (50k-900k tokens, 1989-2025) confirm robustness to transcript length, vocabulary drift, and domain change, maintaining 95-100% rejection accuracy for absent targets. 

---
# Live-SWE-agent: Can Software Engineering Agents Self-Evolve on the Fly? 

**Authors**: Chunqiu Steven Xia, Zhe Wang, Yan Yang, Yuxiang Wei, Lingming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.13646)  

**Abstract**: Large Language Models (LLMs) are reshaping almost all industries, including software engineering. In recent years, a number of LLM agents have been proposed to solve real-world software problems. Such software agents are typically equipped with a suite of coding tools and can autonomously decide the next actions to form complete trajectories to solve end-to-end software tasks. While promising, they typically require dedicated design and may still be suboptimal, since it can be extremely challenging and costly to exhaust the entire agent scaffold design space. Recognizing that software agents are inherently software themselves that can be further refined/modified, researchers have proposed a number of self-improving software agents recently, including the Darwin-Gödel Machine (DGM). Meanwhile, such self-improving agents require costly offline training on specific benchmarks and may not generalize well across different LLMs or benchmarks. In this paper, we propose Live-SWE-agent, the first live software agent that can autonomously and continuously evolve itself on-the-fly during runtime when solving real-world software problems. More specifically, Live-SWE-agent starts with the most basic agent scaffold with only access to bash tools (e.g., mini-SWE-agent), and autonomously evolves its own scaffold implementation while solving real-world software problems. Our evaluation on the widely studied SWE-bench Verified benchmark shows that Live-SWE-agent can achieve an impressive solve rate of 75.4% without test-time scaling, outperforming all existing open-source software agents and approaching the performance of the best proprietary solution. Moreover, Live-SWE-agent outperforms state-of-the-art manually crafted software agents on the recent SWE-Bench Pro benchmark, achieving the best-known solve rate of 45.8%. 

---
# P1: Mastering Physics Olympiads with Reinforcement Learning 

**Authors**: Jiacheng Chen, Qianjia Cheng, Fangchen Yu, Haiyuan Wan, Yuchen Zhang, Shenghe Zheng, Junchi Yao, Qingyang Zhang, Haonan He, Yun Luo, Yufeng Zhao, Futing Wang, Li Sheng, Chengxing Xie, Yuxin Zuo, Yizhuo Li, Wenxauan Zeng, Yulun Wu, Rui Huang, Dongzhan Zhou, Kai Chen, Yu Qiao, Lei Bai, Yu Cheng, Ning Ding, Bowen Zhou, Peng Ye, Ganqu Cui  

**Link**: [PDF](https://arxiv.org/pdf/2511.13612)  

**Abstract**: Recent progress in large language models (LLMs) has moved the frontier from puzzle-solving to science-grade reasoning-the kind needed to tackle problems whose answers must stand against nature, not merely fit a rubric. Physics is the sharpest test of this shift, which binds symbols to reality in a fundamental way, serving as the cornerstone of most modern technologies. In this work, we manage to advance physics research by developing large language models with exceptional physics reasoning capabilities, especially excel at solving Olympiad-level physics problems. We introduce P1, a family of open-source physics reasoning models trained entirely through reinforcement learning (RL). Among them, P1-235B-A22B is the first open-source model with Gold-medal performance at the latest International Physics Olympiad (IPhO 2025), and wins 12 gold medals out of 13 international/regional physics competitions in 2024/2025. P1-30B-A3B also surpasses almost all other open-source models on IPhO 2025, getting a silver medal. Further equipped with an agentic framework PhysicsMinions, P1-235B-A22B+PhysicsMinions achieves overall No.1 on IPhO 2025, and obtains the highest average score over the 13 physics competitions. Besides physics, P1 models also present great performance on other reasoning tasks like math and coding, showing the great generalibility of P1 series. 

---
# ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models 

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13548)  

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions. 

---
# Exploring Multi-Table Retrieval Through Iterative Search 

**Authors**: Allaa Boutaleb, Bernd Amann, Rafael Angarita, Hubert Naacke  

**Link**: [PDF](https://arxiv.org/pdf/2511.13418)  

**Abstract**: Open-domain question answering over datalakes requires retrieving and composing information from multiple tables, a challenging subtask that demands semantic relevance and structural coherence (e.g., joinability). While exact optimization methods like Mixed-Integer Programming (MIP) can ensure coherence, their computational complexity is often prohibitive. Conversely, simpler greedy heuristics that optimize for query coverage alone often fail to find these coherent, joinable sets. This paper frames multi-table retrieval as an iterative search process, arguing this approach offers advantages in scalability, interpretability, and flexibility. We propose a general framework and a concrete instantiation: a fast, effective Greedy Join-Aware Retrieval algorithm that holistically balances relevance, coverage, and joinability. Experiments across 5 NL2SQL benchmarks demonstrate that our iterative method achieves competitive retrieval performance compared to the MIP-based approach while being 4-400x faster depending on the benchmark and search space settings. This work highlights the potential of iterative heuristics for practical, scalable, and composition-aware retrieval. 

---
# Attention Grounded Enhancement for Visual Document Retrieval 

**Authors**: Wanqing Cui, Wei Huang, Yazhi Guo, Yibo Hu, Meiguang Jin, Junfeng Ma, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2511.13415)  

**Abstract**: Visual document retrieval requires understanding heterogeneous and multi-modal content to satisfy information needs. Recent advances use screenshot-based document encoding with fine-grained late interaction, significantly improving retrieval performance. However, retrievers are still trained with coarse global relevance labels, without revealing which regions support the match. As a result, retrievers tend to rely on surface-level cues and struggle to capture implicit semantic connections, hindering their ability to handle non-extractive queries. To alleviate this problem, we propose a \textbf{A}ttention-\textbf{G}rounded \textbf{RE}triever \textbf{E}nhancement (AGREE) framework. AGREE leverages cross-modal attention from multimodal large language models as proxy local supervision to guide the identification of relevant document regions. During training, AGREE combines local signals with the global signals to jointly optimize the retriever, enabling it to learn not only whether documents match, but also which content drives relevance. Experiments on the challenging ViDoRe V2 benchmark show that AGREE significantly outperforms the global-supervision-only baseline. Quantitative and qualitative analyses further demonstrate that AGREE promotes deeper alignment between query terms and document regions, moving beyond surface-level matching toward more accurate and interpretable retrieval. Our code is available at: this https URL. 

---
# AutoMalDesc: Large-Scale Script Analysis for Cyber Threat Research 

**Authors**: Alexandru-Mihai Apostu, Andrei Preda, Alexandra Daniela Damir, Diana Bolocan, Radu Tudor Ionescu, Ioana Croitoru, Mihaela Gaman  

**Link**: [PDF](https://arxiv.org/pdf/2511.13333)  

**Abstract**: Generating thorough natural language explanations for threat detections remains an open problem in cybersecurity research, despite significant advances in automated malware detection systems. In this work, we present AutoMalDesc, an automated static analysis summarization framework that, following initial training on a small set of expert-curated examples, operates independently at scale. This approach leverages an iterative self-paced learning pipeline to progressively enhance output quality through synthetic data generation and validation cycles, eliminating the need for extensive manual data annotation. Evaluation across 3,600 diverse samples in five scripting languages demonstrates statistically significant improvements between iterations, showing consistent gains in both summary quality and classification accuracy. Our comprehensive validation approach combines quantitative metrics based on established malware labels with qualitative assessment from both human experts and LLM-based judges, confirming both technical precision and linguistic coherence of generated summaries. To facilitate reproducibility and advance research in this domain, we publish our complete dataset of more than 100K script samples, including annotated seed (0.9K) and test (3.6K) datasets, along with our methodology and evaluation framework. 

---
# Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment 

**Authors**: Jea Kwon, Luiz Felipe Vecchietti, Sungwon Park, Meeyoung Cha  

**Link**: [PDF](https://arxiv.org/pdf/2511.13290)  

**Abstract**: Humans display significant uncertainty when confronted with moral dilemmas, yet the extent of such uncertainty in machines and AI agents remains underexplored. Recent studies have confirmed the overly confident tendencies of machine-generated responses, particularly in large language models (LLMs). As these systems are increasingly embedded in ethical decision-making scenarios, it is important to understand their moral reasoning and the inherent uncertainties in building reliable AI systems. This work examines how uncertainty influences moral decisions in the classical trolley problem, analyzing responses from 32 open-source models and 9 distinct moral dimensions. We first find that variance in model confidence is greater across models than within moral dimensions, suggesting that moral uncertainty is predominantly shaped by model architecture and training method. To quantify uncertainty, we measure binary entropy as a linear combination of total entropy, conditional entropy, and mutual information. To examine its effects, we introduce stochasticity into models via "dropout" at inference time. Our findings show that our mechanism increases total entropy, mainly through a rise in mutual information, while conditional entropy remains largely unchanged. Moreover, this mechanism significantly improves human-LLM moral alignment, with correlations in mutual information and alignment score shifts. Our results highlight the potential to better align model-generated decisions and human preferences by deliberately modulating uncertainty and reducing LLMs' confidence in morally complex scenarios. 

---
# Computational Measurement of Political Positions: A Review of Text-Based Ideal Point Estimation Algorithms 

**Authors**: Patrick Parschan, Charlott Jakob  

**Link**: [PDF](https://arxiv.org/pdf/2511.13238)  

**Abstract**: This article presents the first systematic review of unsupervised and semi-supervised computational text-based ideal point estimation (CT-IPE) algorithms, methods designed to infer latent political positions from textual data. These algorithms are widely used in political science, communication, computational social science, and computer science to estimate ideological preferences from parliamentary speeches, party manifestos, and social media. Over the past two decades, their development has closely followed broader NLP trends -- beginning with word-frequency models and most recently turning to large language models (LLMs). While this trajectory has greatly expanded the methodological toolkit, it has also produced a fragmented field that lacks systematic comparison and clear guidance for applied use. To address this gap, we identified 25 CT-IPE algorithms through a systematic literature review and conducted a manual content analysis of their modeling assumptions and development contexts. To compare them meaningfully, we introduce a conceptual framework that distinguishes how algorithms generate, capture, and aggregate textual variance. On this basis, we identify four methodological families -- word-frequency, topic modeling, word embedding, and LLM-based approaches -- and critically assess their assumptions, interpretability, scalability, and limitations. Our review offers three contributions. First, it provides a structured synthesis of two decades of algorithm development, clarifying how diverse methods relate to one another. Second, it translates these insights into practical guidance for applied researchers, highlighting trade-offs in transparency, technical requirements, and validation strategies that shape algorithm choice. Third, it emphasizes that differences in estimation outcomes across algorithms are themselves informative, underscoring the need for systematic benchmarking. 

---
# STEP: Success-Rate-Aware Trajectory-Efficient Policy Optimization 

**Authors**: Yuhan Chen, Yuxuan Liu, Long Zhang, Pengzhi Gao, Jian Luan, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.13091)  

**Abstract**: Multi-turn interaction remains challenging for online reinforcement learning. A common solution is trajectory-level optimization, which treats each trajectory as a single training sample. However, this approach can be inefficient and yield misleading learning signals: it applies uniform sampling across tasks regardless of difficulty, penalizes correct intermediate actions in failed trajectories, and incurs high sample-collection costs. To address these issues, we propose STEP (Success-rate-aware Trajectory-Efficient Policy optimization), a framework that dynamically allocates sampling based on per-task success rates and performs step-level optimization. STEP maintains a smoothed success-rate record to guide adaptive trajectory resampling, allocating more effort to harder tasks. It then computes success-rate-weighted advantages and decomposes trajectories into step-level samples. Finally, it applies a step-level GRPO augmentation to refine updates for low-success tasks. Experiments on OSWorld and AndroidWorld show that STEP substantially improves sample efficiency and training stability over trajectory-level GRPO, converging faster and generalizing better under the same sampling budget. 

---
# PragWorld: A Benchmark Evaluating LLMs' Local World Model under Minimal Linguistic Alterations and Conversational Dynamics 

**Authors**: Sachin Vashistha, Aryan Bibhuti, Atharva Naik, Martin Tutek, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2511.13021)  

**Abstract**: Real-world conversations are rich with pragmatic elements, such as entity mentions, references, and implicatures. Understanding such nuances is a requirement for successful natural communication, and often requires building a local world model which encodes such elements and captures the dynamics of their evolving states. However, it is not well-understood whether language models (LMs) construct or maintain a robust implicit representation of conversations. In this work, we evaluate the ability of LMs to encode and update their internal world model in dyadic conversations and test their malleability under linguistic alterations. To facilitate this, we apply seven minimal linguistic alterations to conversations sourced from popular datasets and construct two benchmarks comprising yes-no questions. We evaluate a wide range of open and closed source LMs and observe that they struggle to maintain robust accuracy. Our analysis unveils that LMs struggle to memorize crucial details, such as tracking entities under linguistic alterations to conversations. We then propose a dual-perspective interpretability framework which identifies transformer layers that are useful or harmful and highlights linguistic alterations most influenced by harmful layers, typically due to encoding spurious signals or relying on shortcuts. Inspired by these insights, we propose two layer-regularization based fine-tuning strategies that suppress the effect of the harmful layers. 

---
# WebCoach: Self-Evolving Web Agents with Cross-Session Memory Guidance 

**Authors**: Genglin Liu, Shijie Geng, Sha Li, Hejie Cui, Sarah Zhang, Xin Liu, Tianyi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12997)  

**Abstract**: Multimodal LLM-powered agents have recently demonstrated impressive capabilities in web navigation, enabling agents to complete complex browsing tasks across diverse domains. However, current agents struggle with repetitive errors and lack the ability to learn from past experiences across sessions, limiting their long-term robustness and sample efficiency. We introduce WebCoach, a model-agnostic self-evolving framework that equips web browsing agents with persistent cross-session memory, enabling improved long-term planning, reflection, and continual learning without retraining. WebCoach consists of three key components: (1) a WebCondenser, which standardizes raw navigation logs into concise summaries; (2) an External Memory Store, which organizes complete trajectories as episodic experiences; and (3) a Coach, which retrieves relevant experiences based on similarity and recency, and decides whether to inject task-specific advice into the agent via runtime hooks. This design empowers web agents to access long-term memory beyond their native context window, improving robustness in complex browsing tasks. Moreover, WebCoach achieves self-evolution by continuously curating episodic memory from new navigation trajectories, enabling agents to improve over time without retraining. Evaluations on the WebVoyager benchmark demonstrate that WebCoach consistently improves the performance of browser-use agents across three different LLM backbones. With a 38B model, it increases task success rates from 47% to 61% while reducing or maintaining the average number of steps. Notably, smaller base models with WebCoach achieve performance comparable to the same web agent using GPT-4o. 

---
# A Content-Preserving Secure Linguistic Steganography 

**Authors**: Lingyun Xiang, Chengfu Ou, Xu He, Zhongliang Yang, Yuling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12565)  

**Abstract**: Existing linguistic steganography methods primarily rely on content transformations to conceal secret messages. However, they often cause subtle yet looking-innocent deviations between normal and stego texts, posing potential security risks in real-world applications. To address this challenge, we propose a content-preserving linguistic steganography paradigm for perfectly secure covert communication without modifying the cover text. Based on this paradigm, we introduce CLstega (\textit{C}ontent-preserving \textit{L}inguistic \textit{stega}nography), a novel method that embeds secret messages through controllable distribution transformation. CLstega first applies an augmented masking strategy to locate and mask embedding positions, where MLM(masked language model)-predicted probability distributions are easily adjustable for transformation. Subsequently, a dynamic distribution steganographic coding strategy is designed to encode secret messages by deriving target distributions from the original probability distributions. To achieve this transformation, CLstega elaborately selects target words for embedding positions as labels to construct a masked sentence dataset, which is used to fine-tune the original MLM, producing a target MLM capable of directly extracting secret messages from the cover text. This approach ensures perfect security of secret messages while fully preserving the integrity of the original cover text. Experimental results show that CLstega can achieve a 100\% extraction success rate, and outperforms existing methods in security, effectively balancing embedding capacity and security. 

---
# Accepted with Minor Revisions: Value of AI-Assisted Scientific Writing 

**Authors**: Sanchaita Hazra, Doeun Lee, Bodhisattwa Prasad Majumder, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.12529)  

**Abstract**: Large Language Models have seen expanding application across domains, yet their effectiveness as assistive tools for scientific writing -- an endeavor requiring precision, multimodal synthesis, and domain expertise -- remains insufficiently understood. We examine the potential of LLMs to support domain experts in scientific writing, with a focus on abstract composition. We design an incentivized randomized controlled trial with a hypothetical conference setup where participants with relevant expertise are split into an author and reviewer pool. Inspired by methods in behavioral science, our novel incentive structure encourages authors to edit the provided abstracts to an acceptable quality for a peer-reviewed submission. Our 2x2 between-subject design expands into two dimensions: the implicit source of the provided abstract and the disclosure of it. We find authors make most edits when editing human-written abstracts compared to AI-generated abstracts without source attribution, often guided by higher perceived readability in AI generation. Upon disclosure of source information, the volume of edits converges in both source treatments. Reviewer decisions remain unaffected by the source of the abstract, but bear a significant correlation with the number of edits made. Careful stylistic edits, especially in the case of AI-generated abstracts, in the presence of source information, improve the chance of acceptance. We find that AI-generated abstracts hold potential to reach comparable levels of acceptability to human-written ones with minimal revision, and that perceptions of AI authorship, rather than objective quality, drive much of the observed editing behavior. Our findings reverberate the significance of source disclosure in collaborative scientific writing. 

---
# Evolving Prompts for Toxicity Search in Large Language Models 

**Authors**: Onkar Shelar, Travis Desell  

**Link**: [PDF](https://arxiv.org/pdf/2511.12487)  

**Abstract**: Large Language Models remain vulnerable to adversarial prompts that elicit toxic content even after safety alignment. We present ToxSearch, a black-box evolutionary framework that tests model safety by evolving prompts in a synchronous steady-state loop. The system employs a diverse set of operators, including lexical substitutions, negation, back-translation, paraphrasing, and two semantic crossover operators, while a moderation oracle provides fitness guidance. Operator-level analysis shows heterogeneous behavior: lexical substitutions offer the best yield-variance trade-off, semantic-similarity crossover acts as a precise low-throughput inserter, and global rewrites exhibit high variance with elevated refusal costs. Using elite prompts evolved on LLaMA 3.1 8B, we observe practically meaningful but attenuated cross-model transfer, with toxicity roughly halving on most targets, smaller LLaMA 3.2 variants showing the strongest resistance, and some cross-architecture models retaining higher toxicity. These results suggest that small, controllable perturbations are effective vehicles for systematic red-teaming and that defenses should anticipate cross-model reuse of adversarial prompts rather than focusing only on single-model hardening. 

---
# Co-Layout: LLM-driven Co-optimization for Interior Layout 

**Authors**: Chucheng Xiang, Ruchao Bao, Biyin Feng, Wenzheng Wu, Zhongyuan Liu, Yirui Guan, Ligang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12474)  

**Abstract**: We present a novel framework for automated interior design that combines large language models (LLMs) with grid-based integer programming to jointly optimize room layout and furniture placement. Given a textual prompt, the LLM-driven agent workflow extracts structured design constraints related to room configurations and furniture arrangements. These constraints are encoded into a unified grid-based representation inspired by ``Modulor". Our formulation accounts for key design requirements, including corridor connectivity, room accessibility, spatial exclusivity, and user-specified preferences. To improve computational efficiency, we adopt a coarse-to-fine optimization strategy that begins with a low-resolution grid to solve a simplified problem and guides the solution at the full resolution. Experimental results across diverse scenarios demonstrate that our joint optimization approach significantly outperforms existing two-stage design pipelines in solution quality, and achieves notable computational efficiency through the coarse-to-fine strategy. 

---
# DenseAnnotate: Enabling Scalable Dense Caption Collection for Images and 3D Scenes via Spoken Descriptions 

**Authors**: Xiaoyu Lin, Aniket Ghorpade, Hansheng Zhu, Justin Qiu, Dea Rrozhani, Monica Lama, Mick Yang, Zixuan Bian, Ruohan Ren, Alan B. Hong, Jiatao Gu, Chris Callison-Burch  

**Link**: [PDF](https://arxiv.org/pdf/2511.12452)  

**Abstract**: With the rapid adoption of multimodal large language models (MLLMs) across diverse applications, there is a pressing need for task-centered, high-quality training data. A key limitation of current training datasets is their reliance on sparse annotations mined from the Internet or entered via manual typing that capture only a fraction of an image's visual content. Dense annotations are more valuable but remain scarce. Traditional text-based annotation pipelines are poorly suited for creating dense annotations: typing limits expressiveness, slows annotation speed, and underrepresents nuanced visual features, especially in specialized areas such as multicultural imagery and 3D asset annotation. In this paper, we present DenseAnnotate, an audio-driven online annotation platform that enables efficient creation of dense, fine-grained annotations for images and 3D assets. Annotators narrate observations aloud while synchronously linking spoken phrases to image regions or 3D scene parts. Our platform incorporates speech-to-text transcription and region-of-attention marking. To demonstrate the effectiveness of DenseAnnotate, we conducted case studies involving over 1,000 annotators across two domains: culturally diverse images and 3D scenes. We curate a human-annotated multi-modal dataset of 3,531 images, 898 3D scenes, and 7,460 3D objects, with audio-aligned dense annotations in 20 languages, including 8,746 image captions, 2,000 scene captions, and 19,000 object captions. Models trained on this dataset exhibit improvements of 5% in multilingual, 47% in cultural alignment, and 54% in 3D spatial capabilities. Our results show that our platform offers a feasible approach for future vision-language research and can be applied to various tasks and diverse types of data. 

---
# VoiceCraft-X: Unifying Multilingual, Voice-Cloning Speech Synthesis and Speech Editing 

**Authors**: Zhisheng Zheng, Puyuan Peng, Anuj Diwan, Cong Phuoc Huynh, Xiaohang Sun, Zhu Liu, Vimal Bhat, David Harwath  

**Link**: [PDF](https://arxiv.org/pdf/2511.12347)  

**Abstract**: We introduce VoiceCraft-X, an autoregressive neural codec language model which unifies multilingual speech editing and zero-shot Text-to-Speech (TTS) synthesis across 11 languages: English, Mandarin, Korean, Japanese, Spanish, French, German, Dutch, Italian, Portuguese, and Polish. VoiceCraft-X utilizes the Qwen3 large language model for phoneme-free cross-lingual text processing and a novel token reordering mechanism with time-aligned text and speech tokens to handle both tasks as a single sequence generation problem. The model generates high-quality, natural-sounding speech, seamlessly creating new audio or editing existing recordings within one framework. VoiceCraft-X shows robust performance in diverse linguistic settings, even with limited per-language data, underscoring the power of unified autoregressive approaches for advancing complex, real-world multilingual speech applications. Audio samples are available at this https URL. 

---
# How Far Do SSL Speech Models Listen for Tone? Temporal Focus of Tone Representation under Low-resource Transfer 

**Authors**: Minu Kim, Ji Sub Um, Hoirin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.12285)  

**Abstract**: Lexical tone is central to many languages but remains underexplored in self-supervised learning (SSL) speech models, especially beyond Mandarin. We study four languages with complex and diverse tone systems: Burmese, Thai, Lao, and Vietnamese, to examine how far such models listen for tone and how transfer operates in low-resource conditions. As a baseline reference, we estimate the temporal span of tone cues to be about 100 ms in Burmese and Thai, and about 180 ms in Lao and Vietnamese. Probes and gradient analyses on fine-tuned SSL models reveal that tone transfer varies by downstream task: automatic speech recognition fine-tuning aligns spans with language-specific tone cues, while prosody- and voice-related tasks bias the model toward overly long spans. These findings indicate that tone transfer is shaped by downstream task, highlighting task effects on temporal focus in tone modeling. 

---
# D$^{3}$ToM: Decider-Guided Dynamic Token Merging for Accelerating Diffusion MLLMs 

**Authors**: Shuochen Chang, Xiaofeng Zhang, Qingyang Liu, Li Niu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12280)  

**Abstract**: Diffusion-based multimodal large language models (Diffusion MLLMs) have recently demonstrated impressive non-autoregressive generative capabilities across vision-and-language tasks. However, Diffusion MLLMs exhibit substantially slower inference than autoregressive models: Each denoising step employs full bidirectional self-attention over the entire sequence, resulting in cubic decoding complexity that becomes computationally impractical with thousands of visual tokens. To address this challenge, we propose D$^{3}$ToM, a Decider-guided dynamic token merging method that dynamically merges redundant visual tokens at different denoising steps to accelerate inference in Diffusion MLLMs. At each denoising step, D$^{3}$ToM uses decider tokens-the tokens generated in the previous denoising step-to build an importance map over all visual tokens. Then it maintains a proportion of the most salient tokens and merges the remainder through similarity-based aggregation. This plug-and-play module integrates into a single transformer layer, physically shortening the visual token sequence for all subsequent layers without altering model parameters. Moreover, D$^{3}$ToM employs a merge ratio that dynamically varies with each denoising step, aligns with the native decoding process of Diffusion MLLMs, achieving superior performance under equivalent computational budgets. Extensive experiments show that D$^{3}$ToM accelerates inference while preserving competitive performance. The code is released at this https URL. 

---
# Preference Learning from Physics-Based Feedback: Tuning Language Models to Design BCC/B2 Superalloys 

**Authors**: Satanu Ghosh, Collin Holgate, Neal R. Brodnik, Doug Downey, Samantha Daly, Tresa M. Pollock, Samuel Carton  

**Link**: [PDF](https://arxiv.org/pdf/2511.12036)  

**Abstract**: We apply preference learning to the task of language model-guided design of novel structural alloys. In contrast to prior work that focuses on generating stable inorganic crystals, our approach targets the synthesizeability of a specific structural class: BCC/B2 superalloys, an underexplored family of materials with potential applications in extreme environments. Using three open-weight models (LLaMA-3.1, Gemma-2, and OLMo-2), we demonstrate that language models can be optimized for multiple design objectives using a single, unified reward signal through Direct Preference Optimization (DPO). Unlike prior approaches that rely on heuristic or human-in-the-loop feedback (costly), our reward signal is derived from thermodynamic phase calculations, offering a scientifically grounded criterion for model tuning. To our knowledge, this is the first demonstration of preference-tuning a language model using physics-grounded feedback for structural alloy design. The resulting framework is general and extensible, providing a path forward for intelligent design-space exploration across a range of physical science domains. 

---
# Leveraging Large Language Models for Career Mobility Analysis: A Study of Gender, Race, and Job Change Using U.S. Online Resume Profiles 

**Authors**: Palakorn Achananuparp, Connie Xu, Yao Lu, Xavier Jayaraj Siddarth Ashok, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2511.12010)  

**Abstract**: We present a large-scale analysis of career mobility of college-educated U.S. workers using online resume profiles to investigate how gender, race, and job change options are associated with upward mobility. This study addresses key research questions of how the job changes affect their upward career mobility, and how the outcomes of upward career mobility differ by gender and race. We address data challenges -- such as missing demographic attributes, missing wage data, and noisy occupation labels -- through various data processing and Artificial Intelligence (AI) methods. In particular, we develop a large language models (LLMs) based occupation classification method known as FewSOC that achieves accuracy significantly higher than the original occupation labels in the resume dataset. Analysis of 228,710 career trajectories reveals that intra-firm occupation change has been found to facilitate upward mobility most strongly, followed by inter-firm occupation change and inter-firm lateral move. Women and Black college graduates experience significantly lower returns from job changes than men and White peers. Multilevel sensitivity analyses confirm that these disparities are robust to cluster-level heterogeneity and reveal additional intersectional patterns. 

---
# Forgetting-MarI: LLM Unlearning via Marginal Information Regularization 

**Authors**: Shizhou Xu, Yuan Ni, Stefan Broecker, Thomas Strohmer  

**Link**: [PDF](https://arxiv.org/pdf/2511.11914)  

**Abstract**: As AI models are trained on ever-expanding datasets, the ability to remove the influence of specific data from trained models has become essential for privacy protection and regulatory compliance. Unlearning addresses this challenge by selectively removing parametric knowledge from the trained models without retraining from scratch, which is critical for resource-intensive models such as Large Language Models (LLMs). Existing unlearning methods often degrade model performance by removing more information than necessary when attempting to ''forget'' specific data. We introduce Forgetting-MarI, an LLM unlearning framework that provably removes only the additional (marginal) information contributed by the data to be unlearned, while preserving the information supported by the data to be retained. By penalizing marginal information, our method yields an explicit upper bound on the unlearn dataset's residual influence in the trained models, providing provable undetectability. Extensive experiments confirm that our approach outperforms current state-of-the-art unlearning methods, delivering reliable forgetting and better preserved general model performance across diverse benchmarks. This advancement represents an important step toward making AI systems more controllable and compliant with privacy and copyright regulations without compromising their effectiveness. 

---
# Better LLM Reasoning via Dual-Play 

**Authors**: Zhengxin Zhang, Chengyu Huang, Aochong Oliver Li, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2511.11881)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress through Reinforcement Learning with Verifiable Rewards (RLVR), yet still rely heavily on external supervision (e.g., curated labels). Adversarial learning, particularly through self-play, offers a promising alternative that enables models to iteratively learn from themselves - thus reducing reliance on external supervision. Dual-play extends adversarial learning by assigning specialized roles to two models and training them against each other, fostering sustained competition and mutual evolution. Despite its promise, adapting dual-play training to LLMs remains limited, largely due to their susceptibility to reward hacking and training instability. In this paper, we introduce PasoDoble, a novel LLM dual-play framework. PasoDoble adversarially trains two models initialized from the same base model: a Proposer, which generates challenging questions with ground-truth answers, and a Solver, which attempts to solve them. We enrich the Proposer with knowledge from a pre-training dataset to ensure the questions' quality and diversity. To avoid reward hacking, the Proposer is rewarded for producing only valid questions that push the Solver's limit, while the Solver is rewarded for solving them correctly, and both are updated jointly. To further enhance training stability, we introduce an optional offline paradigm that decouples Proposer and Solver updates, alternately updating each for several steps while holding the other fixed. Notably, PasoDoble operates without supervision during training. Experimental results show that PasoDoble can improve the reasoning performance of LLMs. Our project page is available at this https URL. 

---
# Do LLMs Really Struggle at NL-FOL Translation? Revealing their Strengths via a Novel Benchmarking Strategy 

**Authors**: Andrea Brunello, Luca Geatti, Michele Mignani, Angelo Montanari, Nicola Saccomanno  

**Link**: [PDF](https://arxiv.org/pdf/2511.11816)  

**Abstract**: Due to its expressiveness and unambiguous nature, First-Order Logic (FOL) is a powerful formalism for representing concepts expressed in natural language (NL). This is useful, e.g., for specifying and verifying desired system properties. While translating FOL into human-readable English is relatively straightforward, the inverse problem, converting NL to FOL (NL-FOL translation), has remained a longstanding challenge, for both humans and machines. Although the emergence of Large Language Models (LLMs) promised a breakthrough, recent literature provides contrasting results on their ability to perform NL-FOL translation. In this work, we provide a threefold contribution. First, we critically examine existing datasets and protocols for evaluating NL-FOL translation performance, revealing key limitations that may cause a misrepresentation of LLMs' actual capabilities. Second, to overcome these shortcomings, we propose a novel evaluation protocol explicitly designed to distinguish genuine semantic-level logical understanding from superficial pattern recognition, memorization, and dataset contamination. Third, using this new approach, we show that state-of-the-art, dialogue-oriented LLMs demonstrate strong NL-FOL translation skills and a genuine grasp of sentence-level logic, whereas embedding-centric models perform markedly worse. 

---
# Reasoning: From Reflection to Solution 

**Authors**: Zixi Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.11712)  

**Abstract**: What is reasoning? This question has driven centuries of philosophical inquiry, from Aristotle's syllogisms to modern computational complexity theory. In the age of large language models achieving superhuman performance on benchmarks like GSM8K (95\% accuracy) and HumanEval (90\% pass@1), we must ask: have these systems learned to \emph{reason}, or have they learned to \emph{pattern-match over reasoning traces}?
This paper argues for a specific answer: \textbf{reasoning is iterative operator application in state spaces, converging to fixed points}. This definition is not merely philosophical -- it has concrete architectural implications that explain both the failures of current systems and the path to genuine reasoning capabilities.
Our investigation begins with a puzzle (OpenXOR), progresses through theory (OpenOperator), and culminates in a working solution (OpenLM) that achieves 76\% accuracy where state-of-the-art LLMs achieve 0\%. This is not about criticizing existing systems, but about \emph{understanding what reasoning requires} and \emph{building architectures that provide it}. 

---
# Generative AI as a Linguistic Equalizer in Global Science 

**Authors**: Dragan Filimonovic, Christian Rutzer, Jeffrey Macher, Rolf Weder  

**Link**: [PDF](https://arxiv.org/pdf/2511.11687)  

**Abstract**: For decades, the dominance of English has created a substantial barrier in global science, disadvantaging non-native speakers. The recent rise of generative AI (GenAI) offers a potential technological response to this long-standing inequity. We provide the first large-scale evidence testing whether GenAI acts as a linguistic equalizer in global science. Drawing on 5.65 million scientific articles published from 2021 to 2024, we compare GenAI-assisted and non-assisted publications from authors in non-English-speaking countries. Using text embeddings derived from a pretrained large language model (SciBERT), we measure each publication's linguistic similarity to a benchmark of scientific writing from U.S.-based authors and track stylistic convergence over time. We find significant and growing convergence for GenAI-assisted publications after the release of ChatGPT in late 2022. The effect is strongest for domestic coauthor teams from countries linguistically distant from English. These findings provide large-scale evidence that GenAI is beginning to reshape global science communication by reducing language barriers in research. 

---
# A Structure-Agnostic Co-Tuning Framework for LLMs and SLMs in Cloud-Edge Systems 

**Authors**: Yuze Liu, Yunhan Wang, Tiehua Zhang, Zhishu Shen, Cheng Peng, Libing Wu, Feng Xia, Jiong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.11678)  

**Abstract**: The surge in intelligent applications driven by large language models (LLMs) has made it increasingly difficult for bandwidth-limited cloud servers to process extensive LLM workloads in real time without compromising user data privacy. To solve these problems, recent research has focused on constructing cloud-edge consortia that integrate server-based LLM with small language models (SLMs) on mobile edge devices. Furthermore, designing collaborative training mechanisms within such consortia to enhance inference performance has emerged as a promising research direction. However, the cross-domain deployment of SLMs, coupled with structural heterogeneity in SLMs architectures, poses significant challenges to enhancing model performance. To this end, we propose Co-PLMs, a novel co-tuning framework for collaborative training of large and small language models, which integrates the process of structure-agnostic mutual learning to realize knowledge exchange between the heterogeneous language models. This framework employs distilled proxy models (DPMs) as bridges to enable collaborative training between the heterogeneous server-based LLM and on-device SLMs, while preserving the domain-specific insights of each device. The experimental results show that Co-PLMs outperform state-of-the-art methods, achieving average increases of 5.38% in Rouge-L and 4.88% in EM. 

---
# H-Model: Dynamic Neural Architectures for Adaptive Processing 

**Authors**: Dmytro Hospodarchuk  

**Link**: [PDF](https://arxiv.org/pdf/2511.11669)  

**Abstract**: This article explores the design and experimentation of a neural network architecture capable of dynamically adjusting its internal structure based on the input data. The proposed model introduces a routing mechanism that allows each layer to influence how its outputs are propagated through the network, enabling iterative and adaptive computation. This concept is loosely inspired by the idea of thought processes and dynamic reasoning, where information flow is conditioned not only on the data itself, but also on the internal state of the system.
It is important to note that this work does not aim to compete with state-of-the-art language models in terms of performance. Instead, it presents a conceptual prototype-an architectural framework that opens up a new direction for exploring adaptable and potentially more interpretable networks. The goal is not optimization of existing benchmarks but rather the proposal of a system that can learn not only representations, but also the structure of computation itself.
Due to practical constraints in computing resources and data, this study remains a preliminary investigation. Nevertheless, initial observations show promise, and the architecture's full potential can only be evaluated in future experiments under more favorable computational conditions. 

---
# Automatic generation of DRI Statements 

**Authors**: Maurice Flechtner  

**Link**: [PDF](https://arxiv.org/pdf/2511.11655)  

**Abstract**: Assessing the quality of group deliberation is essential for improving our understanding of deliberative processes. The Deliberative Reason Index (DRI) offers a sophisticated metric for evaluating group reasoning, but its implementation has been constrained by the complex and time-consuming process of statement generation. This thesis introduces an innovative, automated approach to DRI statement generation that leverages advanced natural language processing (NLP) and large language models (LLMs) to substantially reduce the human effort involved in survey preparation. Key contributions are a systematic framework for automated DRI statement generation and a methodological innovation that significantly lowers the barrier to conducting comprehensive deliberative process assessments. In addition, the findings provide a replicable template for integrating generative artificial intelligence into social science research methodologies. 

---
# EduAgentQG: A Multi-Agent Workflow Framework for Personalized Question Generation 

**Authors**: Rui Jia, Min Zhang, Fengrui Liu, Bo Jiang, Kun Kuang, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.11635)  

**Abstract**: High-quality personalized question banks are crucial for supporting adaptive learning and individualized assessment. Manually designing questions is time-consuming and often fails to meet diverse learning needs, making automated question generation a crucial approach to reduce teachers' workload and improve the scalability of educational resources. However, most existing question generation methods rely on single-agent or rule-based pipelines, which still produce questions with unstable quality, limited diversity, and insufficient alignment with educational goals. To address these challenges, we propose EduAgentQG, a multi-agent collaborative framework for generating high-quality and diverse personalized questions. The framework consists of five specialized agents and operates through an iterative feedback loop: the Planner generates structured design plans and multiple question directions to enhance diversity; the Writer produces candidate questions based on the plan and optimizes their quality and diversity using feedback from the Solver and Educator; the Solver and Educator perform binary scoring across multiple evaluation dimensions and feed the evaluation results back to the Writer; the Checker conducts final verification, including answer correctness and clarity, ensuring alignment with educational goals. Through this multi-agent collaboration and iterative feedback loop, EduAgentQG generates questions that are both high-quality and diverse, while maintaining consistency with educational objectives. Experiments on two mathematics question datasets demonstrate that EduAgentQG outperforms existing single-agent and multi-agent methods in terms of question diversity, goal consistency, and overall quality. 

---
# Characterizing and Understanding Energy Footprint and Efficiency of Small Language Model on Edges 

**Authors**: Md Romyull Islam, Bobin Deng, Nobel Dhar, Tu N. Nguyen, Selena He, Yong Shi, Kun Suo  

**Link**: [PDF](https://arxiv.org/pdf/2511.11624)  

**Abstract**: Cloud-based large language models (LLMs) and their variants have significantly influenced real-world applications. Deploying smaller models (i.e., small language models (SLMs)) on edge devices offers additional advantages, such as reduced latency and independence from network connectivity. However, edge devices' limited computing resources and constrained energy budgets challenge efficient deployment. This study evaluates the power efficiency of five representative SLMs - Llama 3.2, Phi-3 Mini, TinyLlama, and Gemma 2 on Raspberry Pi 5, Jetson Nano, and Jetson Orin Nano (CPU and GPU configurations). Results show that Jetson Orin Nano with GPU acceleration achieves the highest energy-to-performance ratio, significantly outperforming CPU-based setups. Llama 3.2 provides the best balance of accuracy and power efficiency, while TinyLlama is well-suited for low-power environments at the cost of reduced accuracy. In contrast, Phi-3 Mini consumes the most energy despite its high accuracy. In addition, GPU acceleration, memory bandwidth, and model architecture are key in optimizing inference energy efficiency. Our empirical analysis offers practical insights for AI, smart systems, and mobile ad-hoc platforms to leverage tradeoffs from accuracy, inference latency, and power efficiency in energy-constrained environments. 

---
# Small Vocabularies, Big Gains: Pretraining and Tokenization in Time Series Models 

**Authors**: Alexis Roger, Gwen Legate, Kashif Rasul, Yuriy Nevmyvaka, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2511.11622)  

**Abstract**: Tokenization and transfer learning are two critical components in building state of the art time series foundation models for forecasting. In this work, we systematically study the effect of tokenizer design, specifically scaling and quantization strategies, on model performance, alongside the impact of pretraining versus random initialization. We show that tokenizer configuration primarily governs the representational capacity and stability of the model, while transfer learning influences optimization efficiency and alignment. Using a combination of empirical training experiments and theoretical analyses, we demonstrate that pretrained models consistently leverage well-designed tokenizers more effectively, particularly at smaller vocabulary sizes. Conversely, misaligned tokenization can diminish or even invert the benefits of pretraining. These findings highlight the importance of careful tokenization in time series modeling and suggest that combining small, efficient vocabularies with pretrained weights is especially advantageous in multi-modal forecasting settings, where the overall vocabulary must be shared across modalities. Our results provide concrete guidance for designing tokenizers and leveraging transfer learning in discrete representation learning for continuous signals. 

---
# SynBullying: A Multi LLM Synthetic Conversational Dataset for Cyberbullying Detectio 

**Authors**: Arefeh Kazemi, Hamza Qadeer, Joachim Wagner, Hossein Hosseini, Sri Balaaji Natarajan Kalaivendan, Brian Davis  

**Link**: [PDF](https://arxiv.org/pdf/2511.11599)  

**Abstract**: We introduce SynBullying, a synthetic multi-LLM conversational dataset for studying and detecting cyberbullying (CB). SynBullying provides a scalable and ethically safe alternative to human data collection by leveraging large language models (LLMs) to simulate realistic bullying interactions. The dataset offers (i) conversational structure, capturing multi-turn exchanges rather than isolated posts; (ii) context-aware annotations, where harmfulness is assessed within the conversational flow considering context, intent, and discourse dynamics; and (iii) fine-grained labeling, covering various CB categories for detailed linguistic and behavioral analysis. We evaluate SynBullying across five dimensions, including conversational structure, lexical patterns, sentiment/toxicity, role dynamics, harm intensity, and CB-type distribution. We further examine its utility by testing its performance as standalone training data and as an augmentation source for CB classification. 

---
# CLINB: A Climate Intelligence Benchmark for Foundational Models 

**Authors**: Michelle Chen Huebscher, Katharine Mach, Aleksandar Stanić, Markus Leippold, Ben Gaiarin, Zeke Hausfather, Elisa Rawat, Erich Fischer, Massimiliano Ciaramita, Joeri Rogelj, Christian Buck, Lierni Sestorain Saralegui, Reto Knutti  

**Link**: [PDF](https://arxiv.org/pdf/2511.11597)  

**Abstract**: Evaluating how Large Language Models (LLMs) handle complex, specialized knowledge remains a critical challenge. We address this through the lens of climate change by introducing CLINB, a benchmark that assesses models on open-ended, grounded, multimodal question answering tasks with clear requirements for knowledge quality and evidential support. CLINB relies on a dataset of real users' questions and evaluation rubrics curated by leading climate scientists. We implement and validate a model-based evaluation process and evaluate several frontier models. Our findings reveal a critical dichotomy. Frontier models demonstrate remarkable knowledge synthesis capabilities, often exhibiting PhD-level understanding and presentation quality. They outperform "hybrid" answers curated by domain experts assisted by weaker models. However, this performance is countered by failures in grounding. The quality of evidence varies, with substantial hallucination rates for references and images. We argue that bridging this gap between knowledge synthesis and verifiable attribution is essential for the deployment of AI in scientific workflows and that reliable, interpretable benchmarks like CLINB are needed to progress towards building trustworthy AI systems. 

---
# LLM-Generated Negative News Headlines Dataset: Creation and Benchmarking Against Real Journalism 

**Authors**: Olusola Babalola, Bolanle Ojokoh, Olutayo Boyinbode  

**Link**: [PDF](https://arxiv.org/pdf/2511.11591)  

**Abstract**: This research examines the potential of datasets generated by Large Language Models (LLMs) to support Natural Language Processing (NLP) tasks, aiming to overcome challenges related to data acquisition and privacy concerns associated with real-world data. Focusing on negative valence text, a critical component of sentiment analysis, we explore the use of LLM-generated synthetic news headlines as an alternative to real-world data. A specialized corpus of negative news headlines was created using tailored prompts to capture diverse negative sentiments across various societal domains. The synthetic headlines were validated by expert review and further analyzed in embedding space to assess their alignment with real-world negative news in terms of content, tone, length, and style. Key metrics such as correlation with real headlines, perplexity, coherence, and realism were evaluated. The synthetic dataset was benchmarked against two sets of real news headlines using evaluations including the Comparative Perplexity Test, Comparative Readability Test, Comparative POS Profiling, BERTScore, and Comparative Semantic Similarity. Results show the generated headlines match real headlines with the only marked divergence being in the proper noun score of the POS profile test. 

---
# The Anatomy of a Triton Attention Kernel 

**Authors**: Burkhard Ringlein, Jan van Lunteren, Radu Stoica, Thomas Parnell  

**Link**: [PDF](https://arxiv.org/pdf/2511.11581)  

**Abstract**: A long-standing goal in both industry and academia is to develop an LLM inference platform that is portable across hardware architectures, eliminates the need for low-level hand-tuning, and still delivers best-in-class efficiency. In this work, we demonstrate that portable, efficient cross-platform LLM inference is indeed possible and share our experience. We develop a state-of-the-art paged attention kernel, the core performance-critical component of many LLM deployments, that builds exclusively on the domain-specific just-in-time compiled language Triton to achieve state-of-the-art performance on both NVIDIA and AMD GPUs. We describe our high-level approach, the key algorithmic and system-level improvements, the parameter auto-tuning required to unlock efficiency, and the integrations into a popular inference server that are necessary to bring the performance of a generic Triton attention kernel from 19.7% of the state-of-the-art to 105.9%. Our results highlight how open-source domain-specific languages can be leveraged to unlock model portability across different GPU vendors. 

---
# Decoupling Positional and Symbolic Attention Behavior in Transformers 

**Authors**: Felipe Urrutia, Jorge Salas, Alexander Kozachinskiy, Cristian Buc Calderon, Hector Pasten, Cristobal Rojas  

**Link**: [PDF](https://arxiv.org/pdf/2511.11579)  

**Abstract**: An important aspect subtending language understanding and production is the ability to independently encode positional and symbolic information of the words within a sentence. In Transformers, positional information is typically encoded using Positional Encodings (PEs). One such popular PE, namely Rotary PE (RoPE), has been widely used due to its empirical success. Recently, it has been argued that part of RoPE's success emerges from its ability to encode robust positional and semantic information using large and small frequencies, respectively. In this work, we perform a deeper dive into the positional versus symbolic dichotomy of attention heads behavior, both at the theoretical and empirical level. We provide general definitions of what it means for a head to behave positionally or symbolically, prove that these are two mutually exclusive behaviors and develop a metric to quantify them. We apply our framework to analyze Transformer-based LLMs using RoPE and find that all heads exhibit a strong correspondence between behavior and frequency use. Finally, we introduce canonical tasks designed to be either purely positional or symbolic, and demonstrate that the Transformer performance causally relates to the ability of attention heads to leverage the appropriate frequencies. In particular, we show that we can control the Transformer performance by controlling which frequencies the attention heads can access. Altogether, our work provides a detailed understanding of RoPE, and how its properties relate to model behavior. 

---
# LLM Architecture, Scaling Laws, and Economics: A Quick Summary 

**Authors**: William H. Press  

**Link**: [PDF](https://arxiv.org/pdf/2511.11572)  

**Abstract**: The current standard architecture of Large Language Models (LLMs) with QKV self-attention is briefly summarized, including the architecture of a typical Transformer. Scaling laws for compute (flops) and memory (parameters plus data) are given, along with their present (2025) rough cost estimates for the parameters of present LLMs of various scales, including discussion of whether DeepSeek should be viewed as a special case. Nothing here is new, but this material seems not otherwise readily available in summary form. 

---
