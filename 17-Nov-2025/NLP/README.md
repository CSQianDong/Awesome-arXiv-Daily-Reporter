# PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning 

**Authors**: Afra Feyza Akyürek, Advait Gosai, Chen Bo Calvin Zhang, Vipul Gupta, Jaehwan Jeong, Anisha Gunjal, Tahseen Rabbani, Maria Mazzone, David Randolph, Mohammad Mahmoudi Meymand, Gurshaan Chattha, Paula Rodriguez, Diego Mares, Pavit Singh, Michael Liu, Subodh Chawla, Pete Cline, Lucy Ogaz, Ernesto Hernandez, Zihao Wang, Pavi Bhatter, Marcos Ayestaran, Bing Liu, Yunzhong He  

**Link**: [PDF](https://arxiv.org/pdf/2511.11562)  

**Abstract**: Frontier model progress is often measured by academic benchmarks, which offer a limited view of performance in real-world professional contexts. Existing evaluations often fail to assess open-ended, economically consequential tasks in high-stakes domains like Legal and Finance, where practical returns are paramount. To address this, we introduce Professional Reasoning Bench (PRBench), a realistic, open-ended, and difficult benchmark of real-world problems in Finance and Law. We open-source its 1,100 expert-authored tasks and 19,356 expert-curated criteria, making it, to our knowledge, the largest public, rubric-based benchmark for both legal and finance domains. We recruit 182 qualified professionals, holding JDs, CFAs, or 6+ years of experience, who contributed tasks inspired by their actual workflows. This process yields significant diversity, with tasks spanning 114 countries and 47 US jurisdictions. Our expert-curated rubrics are validated through a rigorous quality pipeline, including independent expert validation. Subsequent evaluation of 20 leading models reveals substantial room for improvement, with top scores of only 0.39 (Finance) and 0.37 (Legal) on our Hard subsets. We further catalog associated economic impacts of the prompts and analyze performance using human-annotated rubric categories. Our analysis shows that models with similar overall scores can diverge significantly on specific capabilities. Common failure modes include inaccurate judgments, a lack of process transparency and incomplete reasoning, highlighting critical gaps in their reliability for professional adoption. 

---
# W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search 

**Authors**: Zhenyu Ding, Yuhao Wang, Tengyue Xiao, Haoying Wang, Guojun Ma, Mingyang Wan, Caigui Jiang, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.11518)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities, yet their outputs often suffer from misalignment with human preferences due to the inadequacy of weak supervision and a lack of fine-grained control. Training-time alignment methods like Reinforcement Learning from Human Feedback (RLHF) face prohibitive costs in expert supervision and inherent scalability limitations, offering limited dynamic control during inference. Consequently, there is an urgent need for scalable and adaptable alignment mechanisms. To address this, we propose W2S-AlignTree, a pioneering plug-and-play inference-time alignment framework that synergistically combines Monte Carlo Tree Search (MCTS) with the Weak-to-Strong Generalization paradigm for the first time. W2S-AlignTree formulates LLM alignment as an optimal heuristic search problem within a generative search tree. By leveraging weak model's real-time, step-level signals as alignment proxies and introducing an Entropy-Aware exploration mechanism, W2S-AlignTree enables fine-grained guidance during strong model's generation without modifying its parameters. The approach dynamically balances exploration and exploitation in high-dimensional generation search trees. Experiments across controlled sentiment generation, summarization, and instruction-following show that W2S-AlignTree consistently outperforms strong baselines. Notably, W2S-AlignTree raises the performance of Llama3-8B from 1.89 to 2.19, a relative improvement of 15.9 on the summarization task. 

---
# Proactive Hearing Assistants that Isolate Egocentric Conversations 

**Authors**: Guilin Hu, Malek Itani, Tuochao Chen, Shyamnath Gollakota  

**Link**: [PDF](https://arxiv.org/pdf/2511.11473)  

**Abstract**: We introduce proactive hearing assistants that automatically identify and separate the wearer's conversation partners, without requiring explicit prompts. Our system operates on egocentric binaural audio and uses the wearer's self-speech as an anchor, leveraging turn-taking behavior and dialogue dynamics to infer conversational partners and suppress others. To enable real-time, on-device operation, we propose a dual-model architecture: a lightweight streaming model runs every 12.5 ms for low-latency extraction of the conversation partners, while a slower model runs less frequently to capture longer-range conversational dynamics. Results on real-world 2- and 3-speaker conversation test sets, collected with binaural egocentric hardware from 11 participants totaling 6.8 hours, show generalization in identifying and isolating conversational partners in multi-conversation settings. Our work marks a step toward hearing assistants that adapt proactively to conversational dynamics and engagement. More information can be found on our website: this https URL 

---
# MajinBook: An open catalogue of digital world literature with likes 

**Authors**: Antoine Mazières, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2511.11412)  

**Abstract**: This data paper introduces MajinBook, an open catalogue designed to facilitate the use of shadow libraries--such as Library Genesis and Z-Library--for computational social science and cultural analytics. By linking metadata from these vast, crowd-sourced archives with structured bibliographic data from Goodreads, we create a high-precision corpus of over 539,000 references to English-language books spanning three centuries, enriched with first publication dates, genres, and popularity metrics like ratings and reviews. Our methodology prioritizes natively digital EPUB files to ensure machine-readable quality, while addressing biases in traditional corpora like HathiTrust, and includes secondary datasets for French, German, and Spanish. We evaluate the linkage strategy for accuracy, release all underlying data openly, and discuss the project's legal permissibility under EU and US frameworks for text and data mining in research. 

---
# Studies with impossible languages falsify LMs as models of human language 

**Authors**: Jeffrey S. Bowers, Jeff Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2511.11389)  

**Abstract**: According to Futrell and Mahowald [arXiv:2501.17047], both infants and language models (LMs) find attested languages easier to learn than impossible languages that have unnatural structures. We review the literature and show that LMs often learn attested and many impossible languages equally well. Difficult to learn impossible languages are simply more complex (or random). LMs are missing human inductive biases that support language acquisition. 

---
# M-DAIGT: A Shared Task on Multi-Domain Detection of AI-Generated Text 

**Authors**: Salima Lamsiyah, Saad Ezzini, Abdelkader El Mahdaouy, Hamza Alami, Abdessamad Benlahbib, Samir El Amrany, Salmane Chafik, Hicham Hammouchi  

**Link**: [PDF](https://arxiv.org/pdf/2511.11340)  

**Abstract**: The generation of highly fluent text by Large Language Models (LLMs) poses a significant challenge to information integrity and academic research. In this paper, we introduce the Multi-Domain Detection of AI-Generated Text (M-DAIGT) shared task, which focuses on detecting AI-generated text across multiple domains, particularly in news articles and academic writing. M-DAIGT comprises two binary classification subtasks: News Article Detection (NAD) (Subtask 1) and Academic Writing Detection (AWD) (Subtask 2). To support this task, we developed and released a new large-scale benchmark dataset of 30,000 samples, balanced between human-written and AI-generated texts. The AI-generated content was produced using a variety of modern LLMs (e.g., GPT-4, Claude) and diverse prompting strategies. A total of 46 unique teams registered for the shared task, of which four teams submitted final results. All four teams participated in both Subtask 1 and Subtask 2. We describe the methods employed by these participating teams and briefly discuss future directions for M-DAIGT. 

---
# LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models 

**Authors**: Jian Gao, Richeng Xuan, Zhaolu Kang, Dingshi Liao, Wenxin Huang, Zongmou Huang, Yangdi Xu, Bowen Qin, Zheqi He, Xi Yang, Changjin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.11334)  

**Abstract**: The rapid advancement of large language models (LLMs) has not been matched by their evaluation in low-resource languages, especially Southeast Asian languages like Lao. To fill this gap, we introduce LaoBench, the first large-scale, high-quality, and multidimensional benchmark dataset dedicated to assessing LLMs' comprehensive language understanding and reasoning abilities in Lao. LaoBench comprises over 17,000 carefully curated samples spanning three core dimensions: knowledge application, K12 foundational education, and bilingual translation among Lao, Chinese, and English. The dataset is divided into open-source and closed-source subsets, with the closed-source portion enabling black-box evaluation on an official platform to ensure fairness and data security. Our data construction pipeline integrates expert human curation with automated agent-assisted verification, ensuring linguistic accuracy, cultural relevance, and educational value. Benchmarking multiple state-of-the-art LLMs on LaoBench reveals that current models still face significant challenges in mastering Lao across diverse tasks. We hope LaoBench will catalyze further research and development of AI technologies for underrepresented Southeast Asian languages. 

---
# NOVA: An Agentic Framework for Automated Histopathology Analysis and Discovery 

**Authors**: Anurag J. Vaidya, Felix Meissen, Daniel C. Castro, Shruthi Bannur, Tristan Lazard, Drew F. K. Williamson, Faisal Mahmood, Javier Alvarez-Valle, Stephanie L. Hyland, Kenza Bouzid  

**Link**: [PDF](https://arxiv.org/pdf/2511.11324)  

**Abstract**: Digitized histopathology analysis involves complex, time-intensive workflows and specialized expertise, limiting its accessibility. We introduce NOVA, an agentic framework that translates scientific queries into executable analysis pipelines by iteratively generating and running Python code. NOVA integrates 49 domain-specific tools (e.g., nuclei segmentation, whole-slide encoding) built on open-source software, and can also create new tools ad hoc. To evaluate such systems, we present SlideQuest, a 90-question benchmark -- verified by pathologists and biomedical scientists -- spanning data processing, quantitative analysis, and hypothesis testing. Unlike prior biomedical benchmarks focused on knowledge recall or diagnostic QA, SlideQuest demands multi-step reasoning, iterative coding, and computational problem solving. Quantitative evaluation shows NOVA outperforms coding-agent baselines, and a pathologist-verified case study links morphology to prognostically relevant PAM50 subtypes, demonstrating its scalable discovery potential. 

---
# LAET: A Layer-wise Adaptive Ensemble Tuning Framework for Pretrained Language Models 

**Authors**: Jawad Ibn Ahad, Muhammad Rafsan Kabir, Robin Krambroeckers, Sifat Momen, Nabeel Mohammed, Shafin Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2511.11315)  

**Abstract**: Natural Language Processing (NLP) has transformed the financial industry, enabling advancements in areas such as textual analysis, risk management, and forecasting. Large language models (LLMs) like BloombergGPT and FinMA have set new benchmarks across various financial NLP tasks, including sentiment analysis, stock movement prediction, and credit risk assessment. Furthermore, FinMA-ES, a bilingual financial LLM, has also demonstrated strong performance using the FLARE and FLARE-ES benchmarks. However, the high computational demands of these models limit the accessibility of many organizations. To address this, we propose Layer-wise Adaptive Ensemble Tuning (LAET), a novel strategy that selectively fine-tunes the most effective layers of pre-trained LLMs by analyzing hidden state representations while freezing less critical layers. LAET significantly reduces computational overhead while enhancing task-specific performance. Our approach shows strong results in financial NLP tasks, outperforming existing benchmarks and state-of-the-art LLMs such as GPT-4, even with smaller LLMs ($\sim$3B parameters). This work bridges cutting-edge financial NLP research and real-world deployment with efficient and scalable models for financial applications. 

---
# destroR: Attacking Transfer Models with Obfuscous Examples to Discard Perplexity 

**Authors**: Saadat Rafid Ahmed, Rubayet Shareen, Radoan Sharkar, Nazia Hossain, Mansur Mahi, Farig Yousuf Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2511.11309)  

**Abstract**: Advancements in Machine Learning & Neural Networks in recent years have led to widespread implementations of Natural Language Processing across a variety of fields with remarkable success, solving a wide range of complicated problems. However, recent research has shown that machine learning models may be vulnerable in a number of ways, putting both the models and the systems theyre used in at risk. In this paper, we intend to analyze and experiment with the best of existing adversarial attack recipes and create new ones. We concentrated on developing a novel adversarial attack strategy on current state-of-the-art machine learning models by producing ambiguous inputs for the models to confound them and then constructing the path to the future development of the robustness of the models. We will develop adversarial instances with maximum perplexity, utilizing machine learning and deep learning approaches in order to trick the models. In our attack recipe, we will analyze several datasets and focus on creating obfuscous adversary examples to put the models in a state of perplexity, and by including the Bangla Language in the field of adversarial attacks. We strictly uphold utility usage reduction and efficiency throughout our work. 

---
# iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference 

**Authors**: Wei Fan, JinYi Yoon, Bo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.11306)  

**Abstract**: Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%). 

---
# KGQuest: Template-Driven QA Generation from Knowledge Graphs with LLM-Based Refinement 

**Authors**: Sania Nayab, Marco Simoni, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2511.11258)  

**Abstract**: The generation of questions and answers (QA) from knowledge graphs (KG) plays a crucial role in the development and testing of educational platforms, dissemination tools, and large language models (LLM). However, existing approaches often struggle with scalability, linguistic quality, and factual consistency. This paper presents a scalable and deterministic pipeline for generating natural language QA from KGs, with an additional refinement step using LLMs to further enhance linguistic quality. The approach first clusters KG triplets based on their relations, creating reusable templates through natural language rules derived from the entity types of objects and relations. A module then leverages LLMs to refine these templates, improving clarity and coherence while preserving factual accuracy. Finally, the instantiation of answer options is achieved through a selection strategy that introduces distractors from the KG. Our experiments demonstrate that this hybrid approach efficiently generates high-quality QA pairs, combining scalability with fluency and linguistic precision. 

---
# LANE: Lexical Adversarial Negative Examples for Word Sense Disambiguation 

**Authors**: Jader Martins Camboim de Sá, Jooyoung Lee, Cédric Pruski, Marcos Da Silveira  

**Link**: [PDF](https://arxiv.org/pdf/2511.11234)  

**Abstract**: Fine-grained word meaning resolution remains a critical challenge for neural language models (NLMs) as they often overfit to global sentence representations, failing to capture local semantic details. We propose a novel adversarial training strategy, called LANE, to address this limitation by deliberately shifting the model's learning focus to the target word. This method generates challenging negative training examples through the selective marking of alternate words in the training set. The goal is to force the model to create a greater separability between same sentences with different marked words. Experimental results on lexical semantic change detection and word sense disambiguation benchmarks demonstrate that our approach yields more discriminative word representations, improving performance over standard contrastive learning baselines. We further provide qualitative analyses showing that the proposed negatives lead to representations that better capture subtle meaning differences even in challenging environments. Our method is model-agnostic and can be integrated into existing representation learning frameworks. 

---
# Adverbs Revisited: Enhancing WordNet Coverage of Adverbs with a Supersense Taxonomy 

**Authors**: Jooyoung Lee, Jader Martins Camboim de Sá  

**Link**: [PDF](https://arxiv.org/pdf/2511.11214)  

**Abstract**: WordNet offers rich supersense hierarchies for nouns and verbs, yet adverbs remain underdeveloped, lacking a systematic semantic classification. We introduce a linguistically grounded supersense typology for adverbs, empirically validated through annotation, that captures major semantic domains including manner, temporal, frequency, degree, domain, speaker-oriented, and subject-oriented functions. Results from a pilot annotation study demonstrate that these categories provide broad coverage of adverbs in natural text and can be reliably assigned by human annotators. Incorporating this typology extends WordNet's coverage, aligns it more closely with linguistic theory, and facilitates downstream NLP applications such as word sense disambiguation, event extraction, sentiment analysis, and discourse modeling. We present the proposed supersense categories, annotation outcomes, and directions for future work. 

---
# PRSM: A Measure to Evaluate CLIP's Robustness Against Paraphrases 

**Authors**: Udo Schlegel, Franziska Weeber, Jian Lan, Thomas Seidl  

**Link**: [PDF](https://arxiv.org/pdf/2511.11141)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP) is a widely used multimodal model that aligns text and image representations through large-scale training. While it performs strongly on zero-shot and few-shot tasks, its robustness to linguistic variation, particularly paraphrasing, remains underexplored. Paraphrase robustness is essential for reliable deployment, especially in socially sensitive contexts where inconsistent representations can amplify demographic biases. In this paper, we introduce the Paraphrase Ranking Stability Metric (PRSM), a novel measure for quantifying CLIP's sensitivity to paraphrased queries. Using the Social Counterfactuals dataset, a benchmark designed to reveal social and demographic biases, we empirically assess CLIP's stability under paraphrastic variation, examine the interaction between paraphrase robustness and gender, and discuss implications for fairness and equitable deployment of multimodal systems. Our analysis reveals that robustness varies across paraphrasing strategies, with subtle yet consistent differences observed between male- and female-associated queries. 

---
# Speech-Aware Long Context Pruning and Integration for Contextualized Automatic Speech Recognition 

**Authors**: Yiming Rong, Yixin Zhang, Ziyi Wang, Deyang Jiang, Yunlong Zhao, Haoran Wu, Shiyu Zhou, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.11139)  

**Abstract**: Automatic speech recognition (ASR) systems have achieved remarkable performance in common conditions but often struggle to leverage long-context information in contextualized scenarios that require domain-specific knowledge, such as conference presentations. This challenge arises primarily due to constrained model context windows and the sparsity of relevant information within extensive contextual noise. To solve this, we propose the SAP$^{2}$ method, a novel framework that dynamically prunes and integrates relevant contextual keywords in two stages. Specifically, each stage leverages our proposed Speech-Driven Attention-based Pooling mechanism, enabling efficient compression of context embeddings while preserving speech-salient information. Experimental results demonstrate state-of-the-art performance of SAP$^{2}$ on the SlideSpeech and LibriSpeech datasets, achieving word error rates (WER) of 7.71% and 1.12%, respectively. On SlideSpeech, our method notably reduces biased keyword error rates (B-WER) by 41.1% compared to non-contextual baselines. SAP$^{2}$ also exhibits robust scalability, consistently maintaining performance under extensive contextual input conditions on both datasets. 

---
# Enhancing Meme Emotion Understanding with Multi-Level Modality Enhancement and Dual-Stage Modal Fusion 

**Authors**: Yi Shi, Wenlong Meng, Zhenyuan Guo, Chengkun Wei, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.11126)  

**Abstract**: With the rapid rise of social media and Internet culture, memes have become a popular medium for expressing emotional tendencies. This has sparked growing interest in Meme Emotion Understanding (MEU), which aims to classify the emotional intent behind memes by leveraging their multimodal contents. While existing efforts have achieved promising results, two major challenges remain: (1) a lack of fine-grained multimodal fusion strategies, and (2) insufficient mining of memes' implicit meanings and background knowledge. To address these challenges, we propose MemoDetector, a novel framework for advancing MEU. First, we introduce a four-step textual enhancement module that utilizes the rich knowledge and reasoning capabilities of Multimodal Large Language Models (MLLMs) to progressively infer and extract implicit and contextual insights from memes. These enhanced texts significantly enrich the original meme contents and provide valuable guidance for downstream classification. Next, we design a dual-stage modal fusion strategy: the first stage performs shallow fusion on raw meme image and text, while the second stage deeply integrates the enhanced visual and textual features. This hierarchical fusion enables the model to better capture nuanced cross-modal emotional cues. Experiments on two datasets, MET-MEME and MOOD, demonstrate that our method consistently outperforms state-of-the-art baselines. Specifically, MemoDetector improves F1 scores by 4.3\% on MET-MEME and 3.4\% on MOOD. Further ablation studies and in-depth analyses validate the effectiveness and robustness of our approach, highlighting its strong potential for advancing MEU. Our code is available at this https URL. 

---
# AV-Dialog: Spoken Dialogue Models with Audio-Visual Input 

**Authors**: Tuochao Chen, Bandhav Veluri, Hongyu Gong, Shyamnath Gollakota  

**Link**: [PDF](https://arxiv.org/pdf/2511.11124)  

**Abstract**: Dialogue models falter in noisy, multi-speaker environments, often producing irrelevant responses and awkward turn-taking. We present AV-Dialog, the first multimodal dialog framework that uses both audio and visual cues to track the target speaker, predict turn-taking, and generate coherent responses. By combining acoustic tokenization with multi-task, multi-stage training on monadic, synthetic, and real audio-visual dialogue datasets, AV-Dialog achieves robust streaming transcription, semantically grounded turn-boundary detection and accurate responses, resulting in a natural conversational flow. Experiments show that AV-Dialog outperforms audio-only models under interference, reducing transcription errors, improving turn-taking prediction, and enhancing human-rated dialogue quality. These results highlight the power of seeing as well as hearing for speaker-aware interaction, paving the way for {spoken} dialogue agents that perform {robustly} in real-world, noisy environments. 

---
# Analysing Personal Attacks in U.S. Presidential Debates 

**Authors**: Ruban Goyal, Rohitash Chandra, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.11108)  

**Abstract**: Personal attacks have become a notable feature of U.S. presidential debates and play an important role in shaping public perception during elections. Detecting such attacks can improve transparency in political discourse and provide insights for journalists, analysts and the public. Advances in deep learning and transformer-based models, particularly BERT and large language models (LLMs) have created new opportunities for automated detection of harmful language. Motivated by these developments, we present a framework for analysing personal attacks in U.S. presidential debates. Our work involves manual annotation of debate transcripts across the 2016, 2020 and 2024 election cycles, followed by statistical and language-model based analysis. We investigate the potential of fine-tuned transformer models alongside general-purpose LLMs to detect personal attacks in formal political speech. This study demonstrates how task-specific adaptation of modern language models can contribute to a deeper understanding of political communication. 

---
# Can LLMs Detect Their Own Hallucinations? 

**Authors**: Sora Kadotani, Kosuke Nishida, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2511.11087)  

**Abstract**: Large language models (LLMs) can generate fluent responses, but sometimes hallucinate facts. In this paper, we investigate whether LLMs can detect their own hallucinations. We formulate hallucination detection as a classification task of a sentence. We propose a framework for estimating LLMs' capability of hallucination detection and a classification method using Chain-of-Thought (CoT) to extract knowledge from their parameters. The experimental results indicated that GPT-$3.5$ Turbo with CoT detected $58.2\%$ of its own hallucinations. We concluded that LLMs with CoT can detect hallucinations if sufficient knowledge is contained in their parameters. 

---
# Correcting Mean Bias in Text Embeddings: A Refined Renormalization with Training-Free Improvements on MMTEB 

**Authors**: Xingyu Ren, Youran Sun, Haoyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11041)  

**Abstract**: We find that current text embedding models produce outputs with a consistent bias, i.e., each embedding vector $e$ can be decomposed as $\tilde{e} + \mu$, where $\mu$ is almost identical across all sentences. We propose a plug-and-play, training-free and lightweight solution called Renormalization. Through extensive experiments, we show that renormalization consistently and statistically significantly improves the performance of existing models on the Massive Multilingual Text Embedding Benchmark (MMTEB). In particular, across 38 models, renormalization improves performance by 9.7 $\sigma$ on retrieval tasks, 3.1 $\sigma$ on classification tasks, and 0.8 $\sigma$ on other types of tasks. Renormalization has two variants: directly subtracting $\mu$ from $e$, or subtracting the projection of $e$ onto $\mu$. We theoretically predict that the latter performs better, and our experiments confirm this prediction. 

---
# Automata-Based Steering of Large Language Models for Diverse Structured Generation 

**Authors**: Xiaokun Luan, Zeming Wei, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.11018)  

**Abstract**: Large language models (LLMs) are increasingly tasked with generating structured outputs. While structured generation methods ensure validity, they often lack output diversity, a critical limitation that we confirm in our preliminary study. We propose a novel method to enhance diversity in automaton-based structured generation. Our approach utilizes automata traversal history to steer LLMs towards novel structural patterns. Evaluations show our method significantly improves structural and content diversity while maintaining comparable generation efficiency. Furthermore, we conduct a case study showcasing the effectiveness of our method in generating diverse test cases for testing open-source libraries. 

---
# When Data is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets 

**Authors**: Aladin Djuhera, Farhan Ahmed, Swanand Ravindra Kadhe, Syed Zawad, Heiko Ludwig, Holger Boche  

**Link**: [PDF](https://arxiv.org/pdf/2511.10985)  

**Abstract**: Aligning large language models (LLMs) is a central objective of post-training, often achieved through reward modeling and reinforcement learning methods. Among these, direct preference optimization (DPO) has emerged as a widely adopted technique that fine-tunes LLMs on preferred completions over less favorable ones. While most frontier LLMs do not disclose their curated preference pairs, the broader LLM community has released several open-source DPO datasets, including TuluDPO, ORPO, UltraFeedback, HelpSteer, and Code-Preference-Pairs. However, systematic comparisons remain scarce, largely due to the high computational cost and the lack of rich quality annotations, making it difficult to understand how preferences were selected, which task types they span, and how well they reflect human judgment on a per-sample level. In this work, we present the first comprehensive, data-centric analysis of popular open-source DPO corpora. We leverage the Magpie framework to annotate each sample for task category, input quality, and preference reward, a reward-model-based signal that validates the preference order without relying on human annotations. This enables a scalable, fine-grained inspection of preference quality across datasets, revealing structural and qualitative discrepancies in reward margins. Building on these insights, we systematically curate a new DPO mixture, UltraMix, that draws selectively from all five corpora while removing noisy or redundant samples. UltraMix is 30% smaller than the best-performing individual dataset yet exceeds its performance across key benchmarks. We publicly release all annotations, metadata, and our curated mixture to facilitate future research in data-centric preference optimization. 

---
# DiscoX: Benchmarking Discourse-Level Translation task in Expert Domains 

**Authors**: Xiying Zhao, Zhoufutu Wen, Zhixuan Chen, Jingzhe Ding, Jianpeng Jiao, Shuai Li, Xi Li, Danni Liang, Shengda Long, Qianqian Liu, Xianbo Wu, Hongwan Gao, Xiang Gao, Liang Hu, Jiashuo Liu, Mengyun Liu, Weiran Shi, Chenghao Yang, Qianyu Yang, Xuanliang Zhang, Ge Zhang, Wenhao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10984)  

**Abstract**: The evaluation of discourse-level translation in expert domains remains inadequate, despite its centrality to knowledge dissemination and cross-lingual scholarly communication. While these translations demand discourse-level coherence and strict terminological precision, current evaluation methods predominantly focus on segment-level accuracy and fluency. To address this limitation, we introduce DiscoX, a new benchmark for discourse-level and expert-level Chinese-English translation. It comprises 200 professionally-curated texts from 7 domains, with an average length exceeding 1700 tokens. To evaluate performance on DiscoX, we also develop Metric-S, a reference-free system that provides fine-grained automatic assessments across accuracy, fluency, and appropriateness. Metric-S demonstrates strong consistency with human judgments, significantly outperforming existing metrics. Our experiments reveal a remarkable performance gap: even the most advanced LLMs still trail human experts on these tasks. This finding validates the difficulty of DiscoX and underscores the challenges that remain in achieving professional-grade machine translation. The proposed benchmark and evaluation system provide a robust framework for more rigorous evaluation, facilitating future advancements in LLM-based translation. 

---
# CardioEmbed: Domain-Specialized Text Embeddings for Clinical Cardiology 

**Authors**: Richard J. Young, Alice M. Matthews  

**Link**: [PDF](https://arxiv.org/pdf/2511.10930)  

**Abstract**: Biomedical text embeddings have primarily been developed using research literature from PubMed, yet clinical cardiology practice relies heavily on procedural knowledge and specialized terminology found in comprehensive textbooks rather than research abstracts. This research practice gap limits the effectiveness of existing embedding models for clinical applications incardiology. This study trained CardioEmbed, a domain-specialized embedding model based on Qwen3-Embedding-8B, using contrastive learning on a curated corpus of seven comprehensive cardiology textbooks totaling approximately 150,000 sentences after deduplication. The model employs InfoNCE loss with in-batch negatives and achieves 99.60% retrieval accuracy on cardiac-specific semantic retrieval tasks, a +15.94 percentage point improvement over MedTE, the current state-of-the-art medical embedding model. On MTEB medical benchmarks, the model obtained BIOSSES 0.77 Spearman and SciFact 0.61 NDCG@10, indicating competitive performance on related biomedical domains. Domain-specialized training on comprehensive clinical textbooks yields near-perfect cardiology retrieval (99.60% Acc@1), improving over MedTE by +15.94 percentage points. 

---
# Evaluating Large Language Models on Rare Disease Diagnosis: A Case Study using House M.D 

**Authors**: Arsh Gupta, Ajay Narayanan Sridhar, Bonam Mingole, Amulya Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2511.10912)  

**Abstract**: Large language models (LLMs) have demonstrated capabilities across diverse domains, yet their performance on rare disease diagnosis from narrative medical cases remains underexplored. We introduce a novel dataset of 176 symptom-diagnosis pairs extracted from House M.D., a medical television series validated for teaching rare disease recognition in medical education. We evaluate four state-of-the-art LLMs such as GPT 4o mini, GPT 5 mini, Gemini 2.5 Flash, and Gemini 2.5 Pro on narrative-based diagnostic reasoning tasks. Results show significant variation in performance, ranging from 16.48% to 38.64% accuracy, with newer model generations demonstrating a 2.3 times improvement. While all models face substantial challenges with rare disease diagnosis, the observed improvement across architectures suggests promising directions for future development. Our educationally validated benchmark establishes baseline performance metrics for narrative medical reasoning and provides a publicly accessible evaluation framework for advancing AI-assisted diagnosis research. 

---
# Automated Analysis of Learning Outcomes and Exam Questions Based on Bloom's Taxonomy 

**Authors**: Ramya Kumar, Dhruv Gulwani, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10903)  

**Abstract**: This paper explores the automatic classification of exam questions and learning outcomes according to Bloom's Taxonomy. A small dataset of 600 sentences labeled with six cognitive categories - Knowledge, Comprehension, Application, Analysis, Synthesis, and Evaluation - was processed using traditional machine learning (ML) models (Naive Bayes, Logistic Regression, Support Vector Machines), recurrent neural network architectures (LSTM, BiLSTM, GRU, BiGRU), transformer-based models (BERT and RoBERTa), and large language models (OpenAI, Gemini, Ollama, Anthropic). Each model was evaluated under different preprocessing and augmentation strategies (for example, synonym replacement, word embeddings, etc.). Among traditional ML approaches, Support Vector Machines (SVM) with data augmentation achieved the best overall performance, reaching 94 percent accuracy, recall, and F1 scores with minimal overfitting. In contrast, the RNN models and BERT suffered from severe overfitting, while RoBERTa initially overcame it but began to show signs as training progressed. Finally, zero-shot evaluations of large language models (LLMs) indicated that OpenAI and Gemini performed best among the tested LLMs, achieving approximately 0.72-0.73 accuracy and comparable F1 scores. These findings highlight the challenges of training complex deep models on limited data and underscore the value of careful data augmentation and simpler algorithms (such as augmented SVM) for Bloom's Taxonomy classification. 

---
# Multimodal Peer Review Simulation with Actionable To-Do Recommendations for Community-Aware Manuscript Revisions 

**Authors**: Mengze Hong, Di Jiang, Weiwei Zhao, Yawen Li, Yihang Wang, Xinyuan Luo, Yanjie Sun, Chen Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10902)  

**Abstract**: While large language models (LLMs) offer promising capabilities for automating academic workflows, existing systems for academic peer review remain constrained by text-only inputs, limited contextual grounding, and a lack of actionable feedback. In this work, we present an interactive web-based system for multimodal, community-aware peer review simulation to enable effective manuscript revisions before paper submission. Our framework integrates textual and visual information through multimodal LLMs, enhances review quality via retrieval-augmented generation (RAG) grounded in web-scale OpenReview data, and converts generated reviews into actionable to-do lists using the proposed Action:Objective[\#] format, providing structured and traceable guidance. The system integrates seamlessly into existing academic writing platforms, providing interactive interfaces for real-time feedback and revision tracking. Experimental results highlight the effectiveness of the proposed system in generating more comprehensive and useful reviews aligned with expert standards, surpassing ablated baselines and advancing transparent, human-centered scholarly assistance. 

---
# Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering 

**Authors**: Xueren Ge, Sahil Murtaza, Anthony Cortez, Homa Alemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10900)  

**Abstract**: Large language models (LLMs) have shown promise in medical question answering, yet they often overlook the domain-specific expertise that professionals depend on, such as the clinical subject areas (e.g., trauma, airway) and the certification level (e.g., EMT, Paramedic). Existing approaches typically apply general-purpose prompting or retrieval strategies without leveraging this structured context, limiting performance in high-stakes settings. We address this gap with EMSQA, an 24.3K-question multiple-choice dataset spanning 10 clinical subject areas and 4 certification levels, accompanied by curated, subject area-aligned knowledge bases (40K documents and 2M tokens). Building on EMSQA, we introduce (i) Expert-CoT, a prompting strategy that conditions chain-of-thought (CoT) reasoning on specific clinical subject area and certification level, and (ii) ExpertRAG, a retrieval-augmented generation pipeline that grounds responses in subject area-aligned documents and real-world patient data. Experiments on 4 LLMs show that Expert-CoT improves up to 2.05% over vanilla CoT prompting. Additionally, combining Expert-CoT with ExpertRAG yields up to a 4.59% accuracy gain over standard RAG baselines. Notably, the 32B expertise-augmented LLMs pass all the computer-adaptive EMS certification simulation exams. 

---
# From Proof to Program: Characterizing Tool-Induced Reasoning Hallucinations in Large Language Models 

**Authors**: Farima Fatahi Bayat, Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2511.10899)  

**Abstract**: Tool-augmented Language Models (TaLMs) can invoke external tools to solve problems beyond their parametric capacity. However, it remains unclear whether these tool-enabled gains reflect trustworthy reasoning. Focusing on the Code Interpreter tool, we show that even when tools are selected and executed correctly, TaLMs treat tool outputs as substitutes for reasoning, producing solutions that appear correct but lack coherent justification. We term this failure mode Tool-Induced Myopia (TIM), and study it using PYMATH, a benchmark of 1,679 competition-level mathematical problems for which Python code is helpful but not sufficient. We further develop a multi-dimensional evaluation suite to quantify reasoning degradation in TaLMs relative to their non-tool counterparts. Our findings reveal that while TaLMs achieve up to a 19.3 percentage point gain in final-answer accuracy, their reasoning behavior consistently deteriorates (e.g., non-tool LLMs win up to 41.5% more often in pairwise comparisons of the reasoning process). This degradation intensifies with tool use; the more frequently a model invokes tools, the less coherent its reasoning becomes. Moreover, tool use shifts errors from arithmetic mistakes toward global reasoning failures (logic, assumption, creativity); with TIM present in ~55% of high-risk cases. Finally, we propose a preference-optimization-based framework that realigns TaLMs to use tools as assistive evidence, improving both final-answer accuracy and reasoning depth under tool use. Codes and data are available at: this https URL. 

---
# MedPath: Multi-Domain Cross-Vocabulary Hierarchical Paths for Biomedical Entity Linking 

**Authors**: Nishant Mishra, Wilker Aziz, Iacer Calixto  

**Link**: [PDF](https://arxiv.org/pdf/2511.10887)  

**Abstract**: Progress in biomedical Named Entity Recognition (NER) and Entity Linking (EL) is currently hindered by a fragmented data landscape, a lack of resources for building explainable models, and the limitations of semantically-blind evaluation metrics. To address these challenges, we present MedPath, a large-scale and multi-domain biomedical EL dataset that builds upon nine existing expert-annotated EL datasets. In MedPath, all entities are 1) normalized using the latest version of the Unified Medical Language System (UMLS), 2) augmented with mappings to 62 other biomedical vocabularies and, crucially, 3) enriched with full ontological paths -- i.e., from general to specific -- in up to 11 biomedical vocabularies. MedPath directly enables new research frontiers in biomedical NLP, facilitating training and evaluation of semantic-rich and interpretable EL systems, and the development of the next generation of interoperable and explainable clinical NLP models. 

---
# A Multifaceted Analysis of Negative Bias in Large Language Models through the Lens of Parametric Knowledge 

**Authors**: Jongyoon Song, Sangwon Yu, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.10881)  

**Abstract**: Negative bias refers to the tendency of large language models (LLMs) to excessively generate negative responses in binary decision tasks (e.g., yes-no question answering). Previous research has focused on detecting and addressing negative attention heads that induce negative bias. However, the underlying detailed factors influencing negative bias remain underexplored. In this paper, we demonstrate that LLMs exhibit format-level negative bias, meaning the prompt format more influences their responses than the semantics of the negative response. For the fine-grained study of the negative bias, we introduce a pipeline for constructing the evaluation set, which systematically categorizes the dataset into three subsets based on the model's parametric knowledge: correct, incorrect, and insufficient relevant knowledge. Through analysis of this evaluation set, we identify a shortcut behavior in which models tend to generate negative responses when they lack sufficient knowledge to answer a yes-no question, leading to negative bias. We further examine how negative bias changes under various prompting scenarios related to parametric knowledge. We observe that providing relevant context and offering an "I don't know" option generally reduces negative bias, whereas chain-of-thought prompting tends to amplify the bias. Finally, we demonstrate that the degree of negative bias can vary depending on the type of prompt, which influences the direction of the response. Our work reveals the various factors that influence negative bias, providing critical insights for mitigating it in LLMs. 

---
# ICX360: In-Context eXplainability 360 Toolkit 

**Authors**: Dennis Wei, Ronny Luss, Xiaomeng Hu, Lucas Monteiro Paes, Pin-Yu Chen, Karthikeyan Natesan Ramamurthy, Erik Miehling, Inge Vejsbjerg, Hendrik Strobelt  

**Link**: [PDF](https://arxiv.org/pdf/2511.10879)  

**Abstract**: Large Language Models (LLMs) have become ubiquitous in everyday life and are entering higher-stakes applications ranging from summarizing meeting transcripts to answering doctors' questions. As was the case with earlier predictive models, it is crucial that we develop tools for explaining the output of LLMs, be it a summary, list, response to a question, etc. With these needs in mind, we introduce In-Context Explainability 360 (ICX360), an open-source Python toolkit for explaining LLMs with a focus on the user-provided context (or prompts in general) that are fed to the LLMs. ICX360 contains implementations for three recent tools that explain LLMs using both black-box and white-box methods (via perturbations and gradients respectively). The toolkit, available at this https URL, contains quick-start guidance materials as well as detailed tutorials covering use cases such as retrieval augmented generation, natural language generation, and jailbreaking. 

---
# From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems 

**Authors**: Parisa Rabbani, Nimet Beyza Bozdag, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2511.10871)  

**Abstract**: LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems. 

---
# Leveraging Parameter Space Symmetries for Reasoning Skill Transfer in LLMs 

**Authors**: Stefan Horoi, Sangwoo Cho, Supriyo Chakraborty, Shi-Xiong Zhang, Sambit Sahu, Guy Wolf, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2511.10850)  

**Abstract**: Task arithmetic is a powerful technique for transferring skills between Large Language Models (LLMs), but it often suffers from negative interference when models have diverged during training. We address this limitation by first aligning the models' parameter spaces, leveraging the inherent permutation, rotation, and scaling symmetries of Transformer architectures. We adapt parameter space alignment for modern Grouped-Query Attention (GQA) and SwiGLU layers, exploring both weight-based and activation-based approaches. Using this alignment-first strategy, we successfully transfer advanced reasoning skills to a non-reasoning model. Experiments on challenging reasoning benchmarks show that our method consistently outperforms standard task arithmetic. This work provides an effective approach for merging and transferring specialized skills across evolving LLM families, reducing redundant fine-tuning and enhancing model adaptability. 

---
# Reinforcing Stereotypes of Anger: Emotion AI on African American Vernacular English 

**Authors**: Rebecca Dorn, Christina Chance, Casandra Rusti, Charles Bickham Jr., Kai-Wei Chang, Fred Morstatter, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2511.10846)  

**Abstract**: Automated emotion detection is widely used in applications ranging from well-being monitoring to high-stakes domains like mental health and hiring. However, models often rely on annotations that reflect dominant cultural norms, limiting model ability to recognize emotional expression in dialects often excluded from training data distributions, such as African American Vernacular English (AAVE). This study examines emotion recognition model performance on AAVE compared to General American English (GAE). We analyze 2.7 million tweets geo-tagged within Los Angeles. Texts are scored for strength of AAVE using computational approximations of dialect features. Annotations of emotion presence and intensity are collected on a dataset of 875 tweets with both high and low AAVE densities. To assess model accuracy on a task as subjective as emotion perception, we calculate community-informed "silver" labels where AAVE-dense tweets are labeled by African American, AAVE-fluent (ingroup) annotators. On our labeled sample, GPT and BERT-based models exhibit false positive prediction rates of anger on AAVE more than double than on GAE. SpanEmo, a popular text-based emotion model, increases false positive rates of anger from 25 percent on GAE to 60 percent on AAVE. Additionally, a series of linear regressions reveals that models and non-ingroup annotations are significantly more correlated with profanity-based AAVE features than ingroup annotations. Linking Census tract demographics, we observe that neighborhoods with higher proportions of African American residents are associated with higher predictions of anger (Pearson's correlation r = 0.27) and lower joy (r = -0.10). These results find an emergent safety issue of emotion AI reinforcing racial stereotypes through biased emotion classification. We emphasize the need for culturally and dialect-informed affective computing systems. 

---
# Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders 

**Authors**: Abir Harrasse, Florent Draye, Zhijing Jin, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2511.10840)  

**Abstract**: Multilingual Large Language Models (LLMs) can process many languages, yet how they internally represent this diversity remains unclear. Do they form shared multilingual representations with language-specific decoding, and if so, why does performance still favor the dominant training language? To address this, we train a series of LLMs on different mixtures of multilingual data and analyze their internal mechanisms using cross-layer transcoders (CLT) and attribution graphs. Our results provide strong evidence for pivot language representations: the model employs nearly identical representations across languages, while language-specific decoding emerges in later layers. Attribution analyses reveal that decoding relies in part on a small set of high-frequency language features in the final layers, which linearly read out language identity from the first layers in the model. By intervening on these features, we can suppress one language and substitute another in the model's outputs. Finally, we study how the dominant training language influences these mechanisms across attribution graphs and decoding pathways. We argue that understanding this pivot-language mechanism is crucial for improving multilingual alignment in LLMs. 

---
# LLM-as-a-Grader: Practical Insights from Large Language Model for Short-Answer and Report Evaluation 

**Authors**: Grace Byun, Swati Rajwal, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.10819)  

**Abstract**: Large Language Models (LLMs) are increasingly explored for educational tasks such as grading, yet their alignment with human evaluation in real classrooms remains underexamined. In this study, we investigate the feasibility of using an LLM (GPT-4o) to evaluate short-answer quizzes and project reports in an undergraduate Computational Linguistics course. We collect responses from approximately 50 students across five quizzes and receive project reports from 14 teams. LLM-generated scores are compared against human evaluations conducted independently by the course teaching assistants (TAs). Our results show that GPT-4o achieves strong correlation with human graders (up to 0.98) and exact score agreement in 55\% of quiz cases. For project reports, it also shows strong overall alignment with human grading, while exhibiting some variability in scoring technical, open-ended responses. We release all code and sample data to support further research on LLMs in educational assessment. This work highlights both the potential and limitations of LLM-based grading systems and contributes to advancing automated grading in real-world academic settings. 

---
# Sabiá: Um Chatbot de Inteligência Artificial Generativa para Suporte no Dia a Dia do Ensino Superior 

**Authors**: Guilherme Biava Rodrigues, Franciele Beal, Marlon Marcon, Alinne Cristinne Corrêa Souza, André Roberto Ortoncelli, Francisco Carlos Monteiro Souza, Rodolfo Adamshuk Silva  

**Link**: [PDF](https://arxiv.org/pdf/2511.10787)  

**Abstract**: Students often report difficulties in accessing day-to-day academic information, which is usually spread across numerous institutional documents and websites. This fragmentation results in a lack of clarity and causes confusion about routine university information. This project proposes the development of a chatbot using Generative Artificial Intelligence (GenAI) and Retrieval-Augmented Generation (RAG) to simplify access to such information. Several GenAI models were tested and evaluated based on quality metrics and the LLM-as-a-Judge approach. Among them, Gemini 2.0 Flash stood out for its quality and speed, and Gemma 3n for its good performance and open-source nature. 

---
# TEDxTN: A Three-way Speech Translation Corpus for Code-Switched Tunisian Arabic - English 

**Authors**: Fethi Bougares, Salima Mdhaffar, Haroun Elleuch, Yannick Estève  

**Link**: [PDF](https://arxiv.org/pdf/2511.10780)  

**Abstract**: In this paper, we introduce TEDxTN, the first publicly available Tunisian Arabic to English speech translation dataset. This work is in line with the ongoing effort to mitigate the data scarcity obstacle for a number of Arabic dialects. We collected, segmented, transcribed and translated 108 TEDx talks following our internally developed annotations guidelines. The collected talks represent 25 hours of speech with code-switching that cover speakers with various accents from over 11 different regions of Tunisia. We make the annotation guidelines and corpus publicly available. This will enable the extension of TEDxTN to new talks as they become available. We also report results for strong baseline systems of Speech Recognition and Speech Translation using multiple pre-trained and fine-tuned end-to-end models. This corpus is the first open source and publicly available speech translation corpus of Code-Switching Tunisian dialect. We believe that this is a valuable resource that can motivate and facilitate further research on the natural language processing of Tunisian Dialect. 

---
# Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs 

**Authors**: Ajwad Abrar, Nafisa Tabassum Oeshy, Prianka Maheru, Farzana Tabassum, Tareque Mohmud Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.10768)  

**Abstract**: Summarizing consumer health questions (CHQs) can ease communication in healthcare, but unfaithful summaries that misrepresent medical details pose serious risks. We propose a framework that combines TextRank-based sentence extraction and medical named entity recognition with large language models (LLMs) to enhance faithfulness in medical text summarization. In our experiments, we fine-tuned the LLaMA-2-7B model on the MeQSum (English) and BanglaCHQ-Summ (Bangla) datasets, achieving consistent improvements across quality (ROUGE, BERTScore, readability) and faithfulness (SummaC, AlignScore) metrics, and outperforming zero-shot baselines and prior systems. Human evaluation further shows that over 80\% of generated summaries preserve critical medical information. These results highlight faithfulness as an essential dimension for reliable medical summarization and demonstrate the potential of our approach for safer deployment of LLMs in healthcare contexts. 

---
# $π$-Attention: Periodic Sparse Transformers for Efficient Long-Context Modeling 

**Authors**: Dong Liu, Yanxuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10696)  

**Abstract**: Transformers have revolutionized natural language processing, but their quadratic complexity with respect to sequence length remains a fundamental bottleneck for long-range modeling. While sparse attention mechanisms like RingAttention reduce computational costs by restricting attention to local neighborhoods, they suffer from limited receptive fields and lack of adaptability. We present \PiAttention, a periodic sparse Transformer that factorizes attention into ring-local neighborhoods, deterministic $\pi$-stride skips, and an adaptive fusion gate. The periodic structure provides predictable coverage of distant tokens, while the sparse footprint keeps the per-layer complexity linear in context length. We prove that \PiAttention achieves $\mathcal{O}(kL + \pi \log L)$ receptive field growth compared to $\mathcal{O}(kL)$ for RingAttention, where $k$ is the local window size, $\pi$ is the skip period, and $L$ is the sequence length. Extensive experiments on language modeling, retrieval, and vision-language tasks demonstrate that \PiAttention matches or surpasses dense attention quality with 8.3\% lower perplexity than RingAttention while using 50\% fewer GPUs for the same context length. Our detailed ablations and visualizations reveal the importance of periodic skips, adaptive fusion, and head-level sparsity coordination for efficient long-context modeling. 

---
# "As Eastern Powers, I will veto." : An Investigation of Nation-level Bias of Large Language Models in International Relations 

**Authors**: Jonghyeon Choi, Yeonjun Choi, Hyun-chul Kim, Beakcheol Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10695)  

**Abstract**: This paper systematically examines nation-level biases exhibited by Large Language Models (LLMs) within the domain of International Relations (IR). Leveraging historical records from the United Nations Security Council (UNSC), we developed a bias evaluation framework comprising three distinct tests to explore nation-level bias in various LLMs, with a particular focus on the five permanent members of the UNSC. Experimental results show that, even with the general bias patterns across models (e.g., favorable biases toward the western nations, and unfavorable biases toward Russia), these still vary based on the LLM. Notably, even within the same LLM, the direction and magnitude of bias for a nation change depending on the evaluation context. This observation suggests that LLM biases are fundamentally multidimensional, varying across models and tasks. We also observe that models with stronger reasoning abilities show reduced bias and better performance. Building on this finding, we introduce a debiasing framework that improves LLMs' factual reasoning combining Retrieval-Augmented Generation with Reflexion-based self-reflection techniques. Experiments show it effectively reduces nation-level bias, and improves performance, particularly in GPT-4o-mini and LLama-3.3-70B. Our findings emphasize the need to assess nation-level bias alongside performance when applying LLMs in the IR domain. 

---
# Where does an LLM begin computing an instruction? 

**Authors**: Aditya Pola, Vineeth N. Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2511.10694)  

**Abstract**: Following an instruction involves distinct sub-processes, such as reading content, reading the instruction, executing it, and producing an answer. We ask where, along the layer stack, instruction following begins, the point where reading gives way to doing. We introduce three simple datasets (Key-Value, Quote Attribution, Letter Selection) and two hop compositions of these tasks. Using activation patching on minimal-contrast prompt pairs, we measure a layer-wise flip rate that indicates when substituting selected residual activations changes the predicted answer. Across models in the Llama family, we observe an inflection point, which we term onset, where interventions that change predictions before this point become largely ineffective afterward. Multi-hop compositions show a similar onset location. These results provide a simple, replicable way to locate where instruction following begins and to compare this location across tasks and model sizes. 

---
# Do AI Voices Learn Social Nuances? A Case of Politeness and Speech Rate 

**Authors**: Eyal Rabin, Zohar Elyoseph, Rotem Israel-Fishelson, Adi Dali, Ravit Nussinson  

**Link**: [PDF](https://arxiv.org/pdf/2511.10693)  

**Abstract**: Voice-based artificial intelligence is increasingly expected to adhere to human social conventions, but can it learn implicit cues that are not explicitly programmed? This study investigates whether state-of-the-art text-to-speech systems have internalized the human tendency to reduce speech rate to convey politeness - a non-obvious prosodic marker. We prompted 22 synthetic voices from two leading AI platforms (AI Studio and OpenAI) to read a fixed script under both "polite and formal" and "casual and informal" conditions and measured the resulting speech duration. Across both AI platforms, the polite prompt produced slower speech than the casual prompt with very large effect sizes, an effect that was statistically significant for all of AI Studio's voices and for a large majority of OpenAI's voices. These results demonstrate that AI can implicitly learn and replicate psychological nuances of human communication, highlighting its emerging role as a social actor capable of reinforcing human social norms. 

---
# Evaluating from Benign to Dynamic Adversarial: A Squid Game for Large Language Models 

**Authors**: Zijian Chen, Wenjun Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2511.10691)  

**Abstract**: Contemporary benchmarks are struggling to keep pace with the development of large language models (LLMs). Although they are indispensable to evaluate model performance on various tasks, it is uncertain whether the models trained on Internet data have genuinely learned how to solve problems or merely seen the questions before. This potential data contamination issue presents a fundamental challenge to establishing trustworthy evaluation frameworks. Meanwhile, existing benchmarks predominantly assume benign, resource-rich settings, leaving the behavior of LLMs under pressure unexplored. In this paper, we introduce Squid Game, a dynamic and adversarial evaluation environment with resource-constrained and asymmetric information settings elaborated to evaluate LLMs through interactive gameplay against other LLM opponents. Notably, Squid Game consists of six elimination-style levels, focusing on multi-faceted abilities, such as instruction-following, code, reasoning, planning, and safety alignment. We evaluate over 50 LLMs on Squid Game, presenting the largest behavioral evaluation study of general LLMs on dynamic adversarial scenarios. We observe a clear generational phase transition on performance in the same model lineage and find evidence that some models resort to speculative shortcuts to win the game, indicating the possibility of higher-level evaluation paradigm contamination in static benchmarks. Furthermore, we compare prominent LLM benchmarks and Squid Game with correlation analyses, highlighting that dynamic evaluation can serve as a complementary part for static evaluations. The code and data will be released in the future. 

---
# Saying the Unsaid: Revealing the Hidden Language of Multimodal Systems Through Telephone Games 

**Authors**: Juntu Zhao, Jialing Zhang, Chongxuan Li, Dequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10690)  

**Abstract**: Recent closed-source multimodal systems have made great advances, but their hidden language for understanding the world remains opaque because of their black-box architectures. In this paper, we use the systems' preference bias to study their hidden language: During the process of compressing the input images (typically containing multiple concepts) into texts and then reconstructing them into images, the systems' inherent preference bias introduces specific shifts in the outputs, disrupting the original input concept co-occurrence. We employ the multi-round "telephone game" to strategically leverage this bias. By observing the co-occurrence frequencies of concepts in telephone games, we quantitatively investigate the concept connection strength in the understanding of multimodal systems, i.e., "hidden language." We also contribute Telescope, a dataset of 10,000+ concept pairs, as the database of our telephone game framework. Our telephone game is test-time scalable: By iteratively running telephone games, we can construct a global map of concept connections in multimodal systems' understanding. Here we can identify preference bias inherited from training, assess generalization capability advancement, and discover more stable pathways for fragile concept connections. Furthermore, we use Reasoning-LLMs to uncover unexpected concept relationships that transcend textual and visual similarities, inferring how multimodal systems understand and simulate the world. This study offers a new perspective on the hidden language of multimodal systems and lays the foundation for future research on the interpretability and controllability of multimodal systems. 

---
# Equilibrium Dynamics and Mitigation of Gender Bias in Synthetically Generated Data 

**Authors**: Ashish Kattamuri, Arpita Vats, Harshwardhan Fartale, Rahul Raja, Akshata Kishore Moharir, Ishita Prasad  

**Link**: [PDF](https://arxiv.org/pdf/2511.10689)  

**Abstract**: Recursive prompting with large language models enables scalable synthetic dataset generation but introduces the risk of bias amplification. We investigate gender bias dynamics across three generations of recursive text generation using three complementary evaluation frameworks: rule-based pattern matching, embedding-based semantic similarity, and downstream task performance. Experiments with three initial bias levels (0.1, 0.3, 0.6) and four mitigation strategies reveal equilibrium dynamics rather than monotonic amplification. The low initial bias amplifies toward the model's inherent bias level (+36%), whereas the high initial bias decays toward it (-26%). Among mitigation methods, contrastive augmentation, which introduces gender-swapped variants, achieves significant downstream bias reduction (98.8% for low initial bias and 91% on average) despite producing higher embedding-based bias scores. This paradox demonstrates that semantic similarity metrics may diverge from behavioral fairness outcomes, highlighting the need for multidimensional evaluation in responsible synthetic data generation. 

---
# Modeling and Predicting Multi-Turn Answer Instability in Large Language Models 

**Authors**: Jiahang He, Rishi Ramachandran, Neel Ramachandran, Aryan Katakam, Kevin Zhu, Sunishchal Dev, Ashwinee Panda, Aryan Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.10688)  

**Abstract**: As large language models (LLMs) are adopted in an increasingly wide range of applications, user-model interactions have grown in both frequency and scale. Consequently, research has focused on evaluating the robustness of LLMs, an essential quality for real-world tasks. In this paper, we employ simple multi-turn follow-up prompts to evaluate models' answer changes, model accuracy dynamics across turns with Markov chains, and examine whether linear probes can predict these changes. Our results show significant vulnerabilities in LLM robustness: a simple "Think again" prompt led to an approximate 10% accuracy drop for Gemini 1.5 Flash over nine turns, while combining this prompt with a semantically equivalent reworded question caused a 7.5% drop for Claude 3.5 Haiku. Additionally, we find that model accuracy across turns can be effectively modeled using Markov chains, enabling the prediction of accuracy probabilities over time. This allows for estimation of the model's stationary (long-run) accuracy, which we find to be on average approximately 8% lower than its first-turn accuracy for Gemini 1.5 Flash. Our results from a model's hidden states also reveal evidence that linear probes can help predict future answer changes. Together, these results establish stationary accuracy as a principled robustness metric for interactive settings and expose the fragility of models under repeated questioning. Addressing this instability will be essential for deploying LLMs in high-stakes and interactive settings where consistent reasoning is as important as initial accuracy. 

---
# A methodological analysis of prompt perturbations and their effect on attack success rates 

**Authors**: Tiago Machado, Maysa Malfiza Garcia de Macedo, Rogerio Abreu de Paula, Marcelo Carpinette Grave, Aminat Adebiyi, Luan Soares de Souza, Enrico Santarelli, Claudio Pinhanez  

**Link**: [PDF](https://arxiv.org/pdf/2511.10686)  

**Abstract**: This work aims to investigate how different Large Language Models (LLMs) alignment methods affect the models' responses to prompt attacks. We selected open source models based on the most common alignment methods, namely, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning with Human Feedback (RLHF). We conducted a systematic analysis using statistical methods to verify how sensitive the Attack Success Rate (ASR) is when we apply variations to prompts designed to elicit inappropriate content from LLMs. Our results show that even small prompt modifications can significantly change the Attack Success Rate (ASR) according to the statistical tests we run, making the models more or less susceptible to types of attack. Critically, our results demonstrate that running existing 'attack benchmarks' alone may not be sufficient to elicit all possible vulnerabilities of both models and alignment methods. This paper thus contributes to ongoing efforts on model attack evaluation by means of systematic and statistically-based analyses of the different alignment methods and how sensitive their ASR is to prompt variation. 

---
# SpiderGen: Towards Procedure Generation For Carbon Life Cycle Assessments with Generative AI 

**Authors**: Anupama Sitaraman, Bharathan Balaji, Yuvraj Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.10684)  

**Abstract**: Investigating the effects of climate change and global warming caused by GHG emissions have been a primary concern worldwide. These emissions are largely contributed to by the production, use and disposal of consumer products. Thus, it is important to build tools to estimate the environmental impact of consumer goods, an essential part of which is conducting Life Cycle Assessments (LCAs). LCAs specify and account for the appropriate processes involved with the production, use, and disposal of the products. We present SpiderGen, an LLM-based workflow which integrates the taxonomy and methodology of traditional LCA with the reasoning capabilities and world knowledge of LLMs to generate the procedural information used for LCA. We additionally evaluate the output of SpiderGen using real-world LCA documents as ground-truth. We find that SpiderGen provides accurate LCA process information that is either fully correct or has minor errors, achieving an F1-Score of 62% across 10 sample data points. We observe that the remaining missed processes and hallucinated errors occur primarily due to differences in detail between LCA documents, as well as differences in the "scope" of which auxiliary processes must also be included. We also demonstrate that SpiderGen performs better than several baselines techniques, such as chain-of-thought prompting and one-shot prompting. Finally, we highlight SpiderGen's potential to reduce the human effort and costs for estimating carbon impact, as it is able to produce LCA process information for less than \$1 USD in under 10 minutes as compared to the status quo LCA, which can cost over \$25000 USD and take up to 21-person days. 

---
# Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models 

**Authors**: Shien Zhu, Samuel Bohl, Robin Oester, Gustavo Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2511.10676)  

**Abstract**: Mixture-of-Experts (MoE) Large Language Models (LLMs) efficiently scale-up the model while keeping relatively low inference cost. As MoE models only activate part of the experts, related work has proposed expert prediction and caching methods to prefetch the experts for faster inference. However, existing approaches utilize the activations from the previous layer for prediction, incurring low accuracy and leave the first layer unoptimized. Applying complex layers or even training standalone networks for better prediction introduces high computation overhead. In this paper, we propose pre-attention expert prediction to achieve accurate and lightweight expert prefetching. The key insight is that some functions in LLMs are ranking-preserving, indicating that matching the ranking of selected experts using simple linear functions is possible. Therefore, we utilize the activations before the attention block in the same layer with 2 linear functions and ranking-aware loss to achieve accurate prediction, which also supports prefetching in the first layer. Our lightweight, pre-attention expert routers achieve 93.03% accuracy on DeepSeek V2 Lite, 94.69% on Qwen3-30B, and 97.62% on Phi-mini-MoE, showing about 15% improvement on absolute accuracy over the state-of-the-art methods. 

---
# Learn to Select: Exploring Label Distribution Divergence for In-Context Demonstration Selection in Text Classification 

**Authors**: Ye Jiang, Taihang Wang, Youzheng Liu, Yimin Wang, Yuhan Xia, Yunfei Long  

**Link**: [PDF](https://arxiv.org/pdf/2511.10675)  

**Abstract**: In-context learning (ICL) for text classification, which uses a few input-label demonstrations to describe a task, has demonstrated impressive performance on large language models (LLMs). However, the selection of in-context demonstrations plays a crucial role and can significantly affect LLMs' performance. Most existing demonstration selection methods primarily focus on semantic similarity between test inputs and demonstrations, often overlooking the importance of label distribution alignment. To address this limitation, we propose a two-stage demonstration selection method, TopK + Label Distribution Divergence (L2D), which leverages a fine-tuned BERT-like small language model (SLM) to generate label distributions and calculate their divergence for both test inputs and candidate demonstrations. This enables the selection of demonstrations that are not only semantically similar but also aligned in label distribution with the test input. Extensive experiments across seven text classification benchmarks show that our method consistently outperforms previous demonstration selection strategies. Further analysis reveals a positive correlation between the performance of LLMs and the accuracy of the underlying SLMs used for label distribution estimation. 

---
# Continual Learning of Domain Knowledge from Human Feedback in Text-to-SQL 

**Authors**: Thomas Cook, Kelly Patel, Sivapriya Vellaichamy, Saba Rahimi, Zhen Zeng, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10674)  

**Abstract**: Large Language Models (LLMs) can generate SQL queries from natural language questions but struggle with database-specific schemas and tacit domain knowledge. We introduce a framework for continual learning from human feedback in text-to-SQL, where a learning agent receives natural language feedback to refine queries and distills the revealed knowledge for reuse on future tasks. This distilled knowledge is stored in a structured memory, enabling the agent to improve execution accuracy over time. We design and evaluate multiple variations of a learning agent architecture that vary in how they capture and retrieve past experiences. Experiments on the BIRD benchmark Dev set show that memory-augmented agents, particularly the Procedural Agent, achieve significant accuracy gains and error reduction by leveraging human-in-the-loop feedback. Our results highlight the importance of transforming tacit human expertise into reusable knowledge, paving the way for more adaptive, domain-aware text-to-SQL systems that continually learn from a human-in-the-loop. 

---
# Large language models in materials science and the need for open-source approaches 

**Authors**: Fengxu Yang, Weitong Chen, Jack D. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2511.10673)  

**Abstract**: Large language models (LLMs) are rapidly transforming materials science. This review examines recent LLM applications across the materials discovery pipeline, focusing on three key areas: mining scientific literature , predictive modelling, and multi-agent experimental systems. We highlight how LLMs extract valuable information such as synthesis conditions from text, learn structure-property relationships, and can coordinate agentic systems integrating computational tools and laboratory automation. While progress has been largely dependent on closed-source commercial models, our benchmark results demonstrate that open-source alternatives can match performance while offering greater transparency, reproducibility, cost-effectiveness, and data privacy. As open-source models continue to improve, we advocate their broader adoption to build accessible, flexible, and community-driven AI platforms for scientific discovery. 

---
# Grounded Visual Factualization: Factual Anchor-Based Finetuning for Enhancing MLLM Factual Consistency 

**Authors**: Filippo Morbiato, Luca Romano, Alessandro Persona  

**Link**: [PDF](https://arxiv.org/pdf/2511.10671)  

**Abstract**: Visual hallucination, where Multimodal Large Language Models fabricate details inconsistent with image content, critically undermines their reliability. Existing fine-tuning methods offer limited improvement, failing to deeply intervene in factual reasoning. This paper introduces Grounded Visual Factualization (GVF) Finetuning, a novel approach to systematically enhance MLLM visual factual consistency. GVF integrates explicit factual signals via three core mechanisms: Factual Anchor Data Augmentation, enriching training data with structured factual anchors and counter-factual prompts; Fact-Aware Instruction Tuning, embedding these cues into explicit instructions; and a Factual Consistency Loss function, specifically penalizing factual inaccuracies. Evaluated on LLaVA-1.5-13B, GVF Finetuning significantly outperforms standard fine-tuning on the VHTest benchmark for both Open-Ended Question (OEQ) and Yes/No Question (YNQ) formats. Crucially, GVF maintains or even slightly improves performance on general multimodal benchmarks like MME and POPE, demonstrating effective mitigation of visual hallucinations without compromising general understanding and reasoning abilities. 

---
# Towards Fine-Grained Code-Switch Speech Translation with Semantic Space Alignment 

**Authors**: Yan Gao, Yazheng Yang, Zhibin Lan, Yidong Chen, Min Zhang, Daimeng Wei, Hui Huang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.10670)  

**Abstract**: Code-switching (CS) speech translation (ST) refers to translating speech that alternates between two or more languages into a target language text, which poses significant challenges due to the complexity of semantic modeling and the scarcity of CS data. Previous studies tend to rely on the model itself to implicitly learn semantic modeling during training, and resort to inefficient and costly manual annotations for these two challenges. To mitigate these limitations, we propose enhancing Large Language Models (LLMs) with a Mixture of Experts (MoE) speech projector, where each expert specializes in the semantic subspace of a specific language, enabling fine-grained modeling of speech features. Additionally, we introduce a multi-stage training paradigm that utilizes readily available monolingual automatic speech recognition (ASR) and monolingual ST data, facilitating speech-text alignment and improving translation capabilities. During training, we leverage a combination of language-specific loss and intra-group load balancing loss to guide the MoE speech projector in efficiently allocating tokens to the appropriate experts, across expert groups and within each group, respectively. To bridge the data gap across different training stages and improve adaptation to the CS scenario, we further employ a transition loss, enabling smooth transitions of data between stages, to effectively address the scarcity of high-quality CS speech translation data. Extensive experiments on widely used datasets demonstrate the effectiveness and generality of our approach. 

---
# Forecasting Spoken Language Development in Children with Cochlear Implants Using Preimplantation MRI 

**Authors**: Yanlin Wang, Di Yuan, Shani Dettman, Dawn Choo, Emily Shimeng Xu, Denise Thomas, Maura E Ryan, Patrick C M Wong, Nancy M Young  

**Link**: [PDF](https://arxiv.org/pdf/2511.10669)  

**Abstract**: Cochlear implants (CI) significantly improve spoken language in children with severe-to-profound sensorineural hearing loss (SNHL), yet outcomes remain more variable than in children with normal hearing. This variability cannot be reliably predicted for individual children using age at implantation or residual hearing. This study aims to compare the accuracy of traditional machine learning (ML) to deep transfer learning (DTL) algorithms to predict post-CI spoken language development of children with bilateral SNHL using a binary classification model of high versus low language improvers. A total of 278 implanted children enrolled from three centers. The accuracy, sensitivity and specificity of prediction models based upon brain neuroanatomic features using traditional ML and DTL learning. DTL prediction models using bilinear attention-based fusion strategy achieved: accuracy of 92.39% (95% CI, 90.70%-94.07%), sensitivity of 91.22% (95% CI, 89.98%-92.47%), specificity of 93.56% (95% CI, 90.91%-96.21%), and area under the curve (AUC) of 0.977 (95% CI, 0.969-0.986). DTL outperformed traditional ML models in all outcome measures. DTL was significantly improved by direct capture of discriminative and task-specific information that are advantages of representation learning enabled by this approach over ML. The results support the feasibility of a single DTL prediction model for language prediction of children served by CI programs worldwide. 

---
# Evaluating LLM Understanding via Structured Tabular Decision Simulations 

**Authors**: Sichao Li, Xinyue Xu, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10667)  

**Abstract**: Large language models (LLMs) often achieve impressive predictive accuracy, yet correctness alone does not imply genuine understanding. True LLM understanding, analogous to human expertise, requires making consistent, well-founded decisions across multiple instances and diverse domains, relying on relevant and domain-grounded decision factors. We introduce Structured Tabular Decision Simulations (STaDS), a suite of expert-like decision settings that evaluate LLMs as if they were professionals undertaking structured decision ``exams''. In this context, understanding is defined as the ability to identify and rely on the correct decision factors, features that determine outcomes within a domain. STaDS jointly assesses understanding through: (i) question and instruction comprehension, (ii) knowledge-based prediction, and (iii) reliance on relevant decision factors. By analyzing 9 frontier LLMs across 15 diverse decision settings, we find that (a) most models struggle to achieve consistently strong accuracy across diverse domains; (b) models can be accurate yet globally unfaithful, and there are frequent mismatches between stated rationales and factors driving predictions. Our findings highlight the need for global-level understanding evaluation protocols and advocate for novel frameworks that go beyond accuracy to enhance LLMs' understanding ability. 

---
# Guarding the Meaning: Self-Supervised Training for Semantic Robustness in Guard Models 

**Authors**: Cristina Pinneri, Christos Louizos  

**Link**: [PDF](https://arxiv.org/pdf/2511.10665)  

**Abstract**: Guard models are a critical component of LLM safety, but their sensitivity to superficial linguistic variations remains a key vulnerability. We show that even meaning-preserving paraphrases can cause large fluctuations in safety scores, revealing a lack of semantic grounding. To address this, we introduce a practical, self-supervised framework for improving the semantic robustness of guard models. Our method leverages paraphrase sets to enforce prediction consistency using a novel, skew-aware aggregation strategy for robust target computation. Notably, we find that standard aggregation methods like mean and median can degrade safety, underscoring the need for skew-aware alternatives. We analyze six open-source guard models and show that our approach reduces semantic variability across paraphrases by ~58%, improves benchmark accuracy by ~2.5% on average, and generalizes to unseen stylistic variations. Intriguingly, we discover a bidirectional relationship between model calibration and consistency: our robustness training improves calibration by up to 40%, revealing a fundamental connection between these properties. These results highlight the value of treating semantic consistency as a first-class training objective and provide a scalable recipe for building more reliable guard models. 

---
# Evaluating Modern Large Language Models on Low-Resource and Morphologically Rich Languages:A Cross-Lingual Benchmark Across Cantonese, Japanese, and Turkish 

**Authors**: Chengxuan Xia, Qianye Wu, Hongbin Guan, Sixuan Tian, Yilun Hao, Xiaoyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10664)  

**Abstract**: Large language models (LLMs) have achieved impressive results in high-resource languages like English, yet their effectiveness in low-resource and morphologically rich languages remains underexplored. In this paper, we present a comprehensive evaluation of seven cutting-edge LLMs -- including GPT-4o, GPT-4, Claude~3.5~Sonnet, LLaMA~3.1, Mistral~Large~2, LLaMA-2~Chat~13B, and Mistral~7B~Instruct -- on a new cross-lingual benchmark covering \textbf{Cantonese, Japanese, and Turkish}. Our benchmark spans four diverse tasks: open-domain question answering, document summarization, English-to-X translation, and culturally grounded dialogue. We combine \textbf{human evaluations} (rating fluency, factual accuracy, and cultural appropriateness) with automated metrics (e.g., BLEU, ROUGE) to assess model performance.
Our results reveal that while the largest proprietary models (GPT-4o, GPT-4, Claude~3.5) generally lead across languages and tasks, significant gaps persist in culturally nuanced understanding and morphological generalization. Notably, GPT-4o demonstrates robust multilingual performance even on cross-lingual tasks, and Claude~3.5~Sonnet achieves competitive accuracy on knowledge and reasoning benchmarks. However, all models struggle to some extent with the unique linguistic challenges of each language, such as Turkish agglutinative morphology and Cantonese colloquialisms. Smaller open-source models (LLaMA-2~13B, Mistral~7B) lag substantially in fluency and accuracy, highlighting the resource disparity. We provide detailed quantitative results, qualitative error analysis, and discuss implications for developing more culturally aware and linguistically generalizable LLMs. Our benchmark and evaluation data are released to foster reproducibility and further research. 

---
# Bayesian Evaluation of Large Language Model Behavior 

**Authors**: Rachel Longjohn, Shang Wu, Saatvik Kher, Catarina Belém, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2511.10661)  

**Abstract**: It is increasingly important to evaluate how text generation systems based on large language models (LLMs) behave, such as their tendency to produce harmful output or their sensitivity to adversarial inputs. Such evaluations often rely on a curated benchmark set of input prompts provided to the LLM, where the output for each prompt may be assessed in a binary fashion (e.g., harmful/non-harmful or does not leak/leaks sensitive information), and the aggregation of binary scores is used to evaluate the LLM. However, existing approaches to evaluation often neglect statistical uncertainty quantification. With an applied statistics audience in mind, we provide background on LLM text generation and evaluation, and then describe a Bayesian approach for quantifying uncertainty in binary evaluation metrics. We focus in particular on uncertainty that is induced by the probabilistic text generation strategies typically deployed in LLM-based systems. We present two case studies applying this approach: 1) evaluating refusal rates on a benchmark of adversarial inputs designed to elicit harmful responses, and 2) evaluating pairwise preferences of one LLM over another on a benchmark of open-ended interactive dialogue examples. We demonstrate how the Bayesian approach can provide useful uncertainty quantification about the behavior of LLM-based systems. 

---
# Test-Time Steering for Lossless Text Compression via Weighted Product of Experts 

**Authors**: Qihang Zhang, Muchen Li, Ziao Wang, Renjie Liao, Lele Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10660)  

**Abstract**: Lossless compression techniques are crucial in an era of rapidly growing data. Traditional universal compressors like gzip offer low computational overhead, high speed, and broad applicability across data distributions. However, they often lead to worse compression rates than modern neural compressors, which leverage large-scale training data to model data distributions more effectively. Despite their advantages, neural compressors struggle to generalize to unseen data. To address this limitation, we propose a novel framework that performs Test-Time Steering via a Weighted Product of Experts (wPoE). At inference, our method adaptively combines a universal compression model with a pretrained neural language model, ensuring the compression rate is at least as good as that of the best individual model. Extensive experiments demonstrate that our approach improves the performance of text compression without requiring fine-tuning. Furthermore, it seamlessly integrates with any autoregressive language model, providing a practical solution for enhancing text compression across diverse data distributions. 

---
# Information Extraction From Fiscal Documents Using LLMs 

**Authors**: Vikram Aggarwal, Jay Kulkarni, Aditi Mascarenhas, Aakriti Narang, Siddarth Raman, Ajay Shah, Susan Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2511.10659)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in text comprehension, but their ability to process complex, hierarchical tabular data remains underexplored. We present a novel approach to extracting structured data from multi-page government fiscal documents using LLM-based techniques. Applied to annual fiscal documents from the State of Karnataka in India (200+ pages), our method achieves high accuracy through a multi-stage pipeline that leverages domain knowledge, sequential context, and algorithmic validation. A large challenge with traditional OCR methods is the inability to verify the accurate extraction of numbers. When applied to fiscal data, the inherent structure of fiscal tables, with totals at each level of the hierarchy, allows for robust internal validation of the extracted data. We use these hierarchical relationships to create multi-level validation checks. We demonstrate that LLMs can read tables and also process document-specific structural hierarchies, offering a scalable process for converting PDF-based fiscal disclosures into research-ready databases. Our implementation shows promise for broader applications across developing country contexts. 

---
# Evaluating Open-Weight Large Language Models for Structured Data Extraction from Narrative Medical Reports Across Multiple Use Cases and Languages 

**Authors**: Douwe J. Spaanderman, Karthik Prathaban, Petr Zelina, Kaouther Mouheb, Lukáš Hejtmánek, Matthew Marzetti, Antonius W. Schurink, Damian Chan, Ruben Niemantsverdriet, Frederik Hartmann, Zhen Qian, Maarten G.J. Thomeer, Petr Holub, Farhan Akram, Frank J. Wolters, Meike W. Vernooij, Cornelis Verhoef, Esther E. Bron, Vít Nováček, Dirk J. Grünhagen, Wiro J. Niessen, Martijn P.A. Starmans, Stefan Klein  

**Link**: [PDF](https://arxiv.org/pdf/2511.10658)  

**Abstract**: Large language models (LLMs) are increasingly used to extract structured information from free-text clinical records, but prior work often focuses on single tasks, limited models, and English-language reports. We evaluated 15 open-weight LLMs on pathology and radiology reports across six use cases, colorectal liver metastases, liver tumours, neurodegenerative diseases, soft-tissue tumours, melanomas, and sarcomas, at three institutes in the Netherlands, UK, and Czech Republic. Models included general-purpose and medical-specialised LLMs of various sizes, and six prompting strategies were compared: zero-shot, one-shot, few-shot, chain-of-thought, self-consistency, and prompt graph. Performance was assessed using task-appropriate metrics, with consensus rank aggregation and linear mixed-effects models quantifying variance. Top-ranked models achieved macro-average scores close to inter-rater agreement across tasks. Small-to-medium general-purpose models performed comparably to large models, while tiny and specialised models performed worse. Prompt graph and few-shot prompting improved performance by ~13%. Task-specific factors, including variable complexity and annotation variability, influenced results more than model size or prompting strategy. These findings show that open-weight LLMs can extract structured data from clinical reports across diseases, languages, and institutions, offering a scalable approach for clinical data curation. 

---
# Patent Representation Learning via Self-supervision 

**Authors**: You Zuo, Kim Gerdes, Eric Villemonte de La Clergerie, Benoît Sagot  

**Link**: [PDF](https://arxiv.org/pdf/2511.10657)  

**Abstract**: This paper presents a simple yet effective contrastive learning framework for learning patent embeddings by leveraging multiple views from within the same document. We first identify a patent-specific failure mode of SimCSE style dropout augmentation: it produces overly uniform embeddings that lose semantic cohesion. To remedy this, we propose section-based augmentation, where different sections of a patent (e.g., abstract, claims, background) serve as complementary views. This design introduces natural semantic and structural diversity, mitigating over-dispersion and yielding embeddings that better preserve both global structure and local continuity. On large-scale benchmarks, our fully self-supervised method matches or surpasses citation-and IPC-supervised baselines in prior-art retrieval and classification, while avoiding reliance on brittle or incomplete annotations. Our analysis further shows that different sections specialize for different tasks-claims and summaries benefit retrieval, while background sections aid classification-highlighting the value of patents' inherent discourse structure for representation learning. These results highlight the value of exploiting intra-document views for scalable and generalizable patent understanding. 

---
# Preference Orchestrator: Prompt-Aware Multi-Objective Alignment for Large Language Models 

**Authors**: Biao Liu, Ning Xu, Junming Yang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2511.10656)  

**Abstract**: While Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, aligning these models with varying human preferences across multiple objectives remains a significant challenge in practical deployments. Existing multi-objective alignment methods rely on manually specified preference weights, which not only burden users with difficult preference specification tasks but also lead to suboptimal training efficiency due to exploration of irrelevant preference combinations. To alleviate these issues, we propose a novel framework named PRO, i.e., PReference Orchestrator, which features a lightweight preference adapter that automatically infers prompt-specific preference weights during both training and deployment phases. Specifically, the adapter automatically learns appropriate preference weights for each prompt by training on normalized reward scores from multiple reward models for preferred responses, which inherently reflect effective preference balances across objectives. Additionally, We provide theoretical analysis proving that our prompt-aware preference mechanism achieves superior performance compared to fixed preference weights in multi-objective alignment scenarios. Extensive experiments across multiple tasks demonstrate the effectiveness of our method over existing multi-objective alignment approaches. 

---
# Spectral Neuro-Symbolic Reasoning II: Semantic Node Merging, Entailment Filtering, and Knowledge Graph Alignment 

**Authors**: Andrew Kiruluta, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2511.10655)  

**Abstract**: This report extends the Spectral Neuro-Symbolic Reasoning (Spectral NSR) framework by introducing three semantically grounded enhancements: (1) transformer-based node merging using contextual embeddings (e.g., Sentence-BERT, SimCSE) to reduce redundancy, (2) sentence-level entailment validation with pretrained NLI classifiers (e.g., RoBERTa, DeBERTa) to improve edge quality, and (3) alignment with external knowledge graphs (e.g., ConceptNet, Wikidata) to augment missing context. These modifications enhance graph fidelity while preserving the core spectral reasoning pipeline. Experimental results on ProofWriter, EntailmentBank, and CLUTRR benchmarks show consistent accuracy gains (up to +3.8\%), improved generalization to adversarial cases, and reduced inference noise. The novelty lies in performing semantic and symbolic refinement entirely upstream of the spectral inference stage, enabling efficient, interpretable, and scalable reasoning without relying on quadratic attention mechanisms. In summary, this work extends the Spectral NSR framework with modular, semantically grounded preprocessing steps that improve graph quality without altering the core spectral reasoning engine. The result is a more robust, interpretable, and scalable reasoning system suitable for deployment in open-domain and real-world settings. 

---
# Empirical Characterization of Temporal Constraint Processing in LLMs 

**Authors**: Javier Marín  

**Link**: [PDF](https://arxiv.org/pdf/2511.10654)  

**Abstract**: When deploying LLMs in agentic architectures requiring real-time decisions under temporal constraints, we assume they reliably determine whether action windows remain open or have closed. This assumption is untested. We characterize temporal constraint processing across eight production-scale models (2.8-8B parameters) using deadline detection tasks, revealing systematic deployment risks: bimodal performance distribution (models achieve either 95% or 50% accuracy), extreme prompt brittleness (30-60 percentage point swings from formatting changes alone), and systematic action bias (100% false positive rates in failing models). Parameter count shows no correlation with capability in this range-a 3.8B model matches 7B models while other 7B models fail completely. Fine-tuning on 200 synthetic examples improves models with partial capability by 12-37 percentage points. We demonstrate that temporal constraint satisfaction cannot be reliably learned through next-token prediction on natural language, even with targeted fine-tuning. This capability requires architectural mechanisms for: (1) continuous temporal state representation, (2) explicit constraint checking separate from linguistic pattern matching, (3) systematic compositional reasoning over temporal relations. Current autoregressive architectures lack these mechanisms. Deploying such systems in time-critical applications without hybrid architectures incorporating symbolic reasoning modules represents unacceptable risk. 

---
# Hybrid Quantum Transformer for Language Generation 

**Authors**: Desheng Kong, Xiangshuo Cui, Jiaying Jin, Jing Xu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10653)  

**Abstract**: Although quantum computing has been increasingly applied to replace classical computation, most existing quantum or hybrid models remain confined to simple tasks, with no successful application to large-scale natural language generation to date. In this work, we present the first hybrid quantum-classical large language model (LLM) for natural language generation, HyQuT, capable of performing coherent and context-aware dialogue. The proposed architecture integrates variational quantum circuits (VQCs) into the Transformer framework at both 8M and 150M parameter scales. Experimental results show that a minimal number of qubits (10 qubits with 80 quantum gates) can replace about 10% of the classical parameters in the 150M-parameter model, while achieving comparable convergence stability and generation quality. This study provides an early demonstration of the feasibility of integrating quantum computing to large-scale generative language models. 

---
# Cognitively-Inspired Episodic Memory Architectures for Accurate and Efficient Character AI 

**Authors**: Rafael Arias Gonzalez, Steve DiPaola  

**Link**: [PDF](https://arxiv.org/pdf/2511.10652)  

**Abstract**: Large language models show promise for embodying historical characters in dialogue systems, but existing approaches face a critical trade-off: simple retrieval-augmented generation produces shallow responses, while multi-stage reflection achieves depth at prohibitive latency. We present an architecture that resolves this tension through offline data augmentation and efficient parallel retrieval from structured episodic memory. Our system transforms biographical data into 1,774 enriched first-person memories with affective-semantic metadata, then employs two-stage retrieval achieving 0.52s prompt generation. Evaluation using LLM-as-judge and RAGAs metrics shows our approach achieves parity with traditional RAG on GPT-4 while significantly outperforming it on smaller models (GPT-3.5, GPT-3), suggesting particular value for resource-constrained deployments. Beyond dialogue, the structured memory enables novel visualization tools: spatiotemporal heatmaps, emotional trajectory analysis, and interactive path tracking, positioning the system as both a dialogue interface and research tool for biographical analysis. We use Van Gogh as a test case, but the architecture is generalizable to any historical figure with substantial textual records, offering a practical framework for educational, museum, and research applications requiring both accuracy and efficiency 

---
# Data Analysis and Performance Evaluation of Simulation Deduction Based on LLMs 

**Authors**: Shansi Zhang, Min Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10651)  

**Abstract**: Data analysis and performance evaluation of simulation deduction plays a pivotal role in modern warfare, which enables military personnel to gain invaluable insights into the potential effectiveness of different strategies, tactics, and operational plans. Traditional manual analysis approach is time-consuming and limited by human errors. To enhance efficiency and accuracy, large language models (LLMs) with strong analytical and inferencing capabilities can be employed. However, high-quality analysis reports with well-structured formatting cannot be obtained through a single instruction input to the LLM. To tackle this issue, we propose a method that first decomposes the complex task into several sub-tasks and designs effective system prompts and user prompts for each sub-task. Multi-round interactions with the LLM incorporating self-check and reflection are then conducted to enable structured data extraction as well as multi-step analysis and evaluation. Furthermore, custom tools are defined and invoked to generate figures and compute metrics. We also design multiple report templates, each tailored to a specific application and input data type, ensuring their adaptability across a variety of scenarios. Extensive evaluation results demonstrate that the reports generated by our method exhibit higher quality, therefore obtaining higher scores than the baseline method. 

---
# Unsupervised Cycle Detection in Agentic Applications 

**Authors**: Felix George, Harshit Kumar, Divya Pathak, Kaustabha Ray, Mudit Verma, Pratibha Moogi  

**Link**: [PDF](https://arxiv.org/pdf/2511.10650)  

**Abstract**: Agentic applications powered by Large Language Models exhibit non-deterministic behaviors that can form hidden execution cycles, silently consuming resources without triggering explicit errors. Traditional observability platforms fail to detect these costly inefficiencies. We present an unsupervised cycle detection framework that combines structural and semantic analysis. Our approach first applies computationally efficient temporal call stack analysis to identify explicit loops and then leverages semantic similarity analysis to uncover subtle cycles characterized by redundant content generation. Evaluated on 1575 trajectories from a LangGraph-based stock market application, our hybrid approach achieves an F1 score of 0.72 (precision: 0.62, recall: 0.86), significantly outperforming individual structural (F1: 0.08) and semantic methods (F1: 0.28). While these results are encouraging, there remains substantial scope for improvement, and future work is needed to refine the approach and address its current limitations. 

---
# Optimizing Mixture of Block Attention 

**Authors**: Guangxuan Xiao, Junxian Guo, Kasra Mazaheri, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.11571)  

**Abstract**: Mixture of Block Attention (MoBA) (Lu et al., 2025) is a promising building block for efficiently processing long contexts in LLMs by enabling queries to sparsely attend to a small subset of key-value blocks, drastically reducing computational cost. However, the design principles governing MoBA's performance are poorly understood, and it lacks an efficient GPU implementation, hindering its practical adoption. In this paper, we first develop a statistical model to analyze MoBA's underlying mechanics. Our model reveals that performance critically depends on the router's ability to accurately distinguish relevant from irrelevant blocks based on query-key affinities. We derive a signal-to-noise ratio that formally connects architectural parameters to this retrieval accuracy. Guided by our analysis, we identify two key pathways for improvement: using smaller block sizes and applying a short convolution on keys to cluster relevant signals, which enhances routing accuracy. While theoretically better, small block sizes are inefficient on GPUs. To bridge this gap, we introduce FlashMoBA, a hardware-aware CUDA kernel that enables efficient MoBA execution even with the small block sizes our theory recommends. We validate our insights by training LLMs from scratch, showing that our improved MoBA models match the performance of dense attention baselines. FlashMoBA achieves up to 14.7x speedup over FlashAttention-2 for small blocks, making our theoretically-grounded improvements practical. Code is available at: this https URL. 

---
# DocLens : A Tool-Augmented Multi-Agent Framework for Long Visual Document Understanding 

**Authors**: Dawei Zhu, Rui Meng, Jiefeng Chen, Sujian Li, Tomas Pfister, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.11552)  

**Abstract**: Comprehending long visual documents, where information is distributed across extensive pages of text and visual elements, is a critical but challenging task for modern Vision-Language Models (VLMs). Existing approaches falter on a fundamental challenge: evidence localization. They struggle to retrieve relevant pages and overlook fine-grained details within visual elements, leading to limited performance and model hallucination. To address this, we propose DocLens, a tool-augmented multi-agent framework that effectively ``zooms in'' on evidence like a lens. It first navigates from the full document to specific visual elements on relevant pages, then employs a sampling-adjudication mechanism to generate a single, reliable answer. Paired with Gemini-2.5-Pro, DocLens achieves state-of-the-art performance on MMLongBench-Doc and FinRAGBench-V, surpassing even human experts. The framework's superiority is particularly evident on vision-centric and unanswerable queries, demonstrating the power of its enhanced localization capabilities. 

---
# Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping 

**Authors**: Dena Mujtaba, Brian Hu, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2511.11551)  

**Abstract**: The deployment of decision-making AI agents presents a critical challenge in maintaining alignment with human values or guidelines while operating in complex, dynamic environments. Agents trained solely to achieve their objectives may adopt harmful behavior, exposing a key trade-off between maximizing the reward function and maintaining the alignment. For the pre-trained agents, ensuring alignment is particularly challenging, as retraining can be a costly and slow process. This is further complicated by the diverse and potentially conflicting attributes representing the ethical values for alignment. To address these challenges, we propose a test-time alignment technique based on model-guided policy shaping. Our method allows precise control over individual behavioral attributes, generalizes across diverse reinforcement learning (RL) environments, and facilitates a principled trade-off between ethical alignment and reward maximization without requiring agent retraining. We evaluate our approach using the MACHIAVELLI benchmark, which comprises 134 text-based game environments and thousands of annotated scenarios involving ethical decisions. The RL agents are first trained to maximize the reward in their respective games. At test time, we apply policy shaping via scenario-action attribute classifiers to ensure decision alignment with ethical attributes. We compare our approach against prior training-time methods and general-purpose agents, as well as study several types of ethical violations and power-seeking behavior. Our results demonstrate that test-time policy shaping provides an effective and scalable solution for mitigating unethical behavior across diverse environments and alignment attributes. 

---
# From Synthetic Scenes to Real Performance: Enhancing Spatial Reasoning in VLMs 

**Authors**: Massimo Rizzoli, Simone Alghisi, Seyed Mahed Mousavi, Giuseppe Riccardi  

**Link**: [PDF](https://arxiv.org/pdf/2511.11440)  

**Abstract**: Fine-tuning Vision-Language Models (VLMs) is a common strategy to improve performance following an ad-hoc data collection and annotation of real-world scenes. However, this process is often prone to biases, errors, and distribution imbalance, resulting in overfitting and imbalanced performance. Although a few studies have tried to address this problem by generating synthetic data, they lacked control over distribution bias and annotation quality. To address these challenges, we redesign the fine-tuning process in two ways. First, we control the generation of data and its annotations, ensuring it is free from bias, distribution imbalance, and annotation errors. We automatically construct the dataset by comprehensively sampling objects' attributes, including color, shape, size, and position within the scene. Secondly, using this annotated dataset, we fine-tune state-of-the-art VLMs and assess performance transferability to real-world data on the absolute position task. We conduct exhaustive evaluations on both synthetic and real-world benchmarks. Our experiments reveal two key findings: 1) fine-tuning on balanced synthetic data yields uniform performance across the visual scene and mitigates common biases; and 2) fine-tuning on synthetic stimuli significantly improves performance on real-world data (COCO), outperforming models fine-tuned in the matched setting. 

---
# On-Device Fine-Tuning via Backprop-Free Zeroth-Order Optimization 

**Authors**: Prabodh Katti, Sangwoo Park, Bipin Rajendran, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2511.11362)  

**Abstract**: On-device fine-tuning is a critical capability for edge AI systems, which must support adaptation to different agentic tasks under stringent memory constraints. Conventional backpropagation (BP)-based training requires storing layer activations and optimizer states, a demand that can be only partially alleviated through checkpointing. In edge deployments in which the model weights must reside entirely in device memory, this overhead severely limits the maximum model size that can be deployed. Memory-efficient zeroth-order optimization (MeZO) alleviates this bottleneck by estimating gradients using forward evaluations alone, eliminating the need for storing intermediate activations or optimizer states. This enables significantly larger models to fit within on-chip memory, albeit at the cost of potentially longer fine-tuning wall-clock time. This paper first provides a theoretical estimate of the relative model sizes that can be accommodated under BP and MeZO training. We then numerically validate the analysis, demonstrating that MeZO exhibits accuracy advantages under on-device memory constraints, provided sufficient wall-clock time is available for fine-tuning. 

---
# Building the Web for Agents: A Declarative Framework for Agent-Web Interaction 

**Authors**: Sven Schultze, Meike Verena Kietzmann, Nils-Lucas Schönfeld, Ruth Stock-Homburg  

**Link**: [PDF](https://arxiv.org/pdf/2511.11287)  

**Abstract**: The increasing deployment of autonomous AI agents on the web is hampered by a fundamental misalignment: agents must infer affordances from human-oriented user interfaces, leading to brittle, inefficient, and insecure interactions. To address this, we introduce VOIX, a web-native framework that enables websites to expose reliable, auditable, and privacy-preserving capabilities for AI agents through simple, declarative HTML elements. VOIX introduces <tool> and <context> tags, allowing developers to explicitly define available actions and relevant state, thereby creating a clear, machine-readable contract for agent behavior. This approach shifts control to the website developer while preserving user privacy by disconnecting the conversational interactions from the website. We evaluated the framework's practicality, learnability, and expressiveness in a three-day hackathon study with 16 developers. The results demonstrate that participants, regardless of prior experience, were able to rapidly build diverse and functional agent-enabled web applications. Ultimately, this work provides a foundational mechanism for realizing the Agentic Web, enabling a future of seamless and secure human-AI collaboration on the web. 

---
# Language-Aided State Estimation 

**Authors**: Yuki Miyoshi, Masaki Inoue, Yusuke Fujimoto  

**Link**: [PDF](https://arxiv.org/pdf/2511.11285)  

**Abstract**: Natural language data, such as text and speech, have become readily available through social networking services and chat platforms. By leveraging human observations expressed in natural language, this paper addresses the problem of state estimation for physical systems, in which humans act as sensing agents. To this end, we propose a Language-Aided Particle Filter (LAPF), a particle filter framework that structures human observations via natural language processing and incorporates them into the update step of the state estimation. Finally, the LAPF is applied to the water level estimation problem in an irrigation canal and its effectiveness is demonstrated. 

---
# SQuaD: The Software Quality Dataset 

**Authors**: Mikel Robredo, Matteo Esposito, Davide Taibi, Rafael Peñaloza, Valentina Lenarduzzi  

**Link**: [PDF](https://arxiv.org/pdf/2511.11265)  

**Abstract**: Software quality research increasingly relies on large-scale datasets that measure both the product and process aspects of software systems. However, existing resources often focus on limited dimensions, such as code smells, technical debt, or refactoring activity, thereby restricting comprehensive analyses across time and quality dimensions. To address this gap, we present the Software Quality Dataset (SQuaD), a multi-dimensional, time-aware collection of software quality metrics extracted from 450 mature open-source projects across diverse ecosystems, including Apache, Mozilla, FFmpeg, and the Linux kernel. By integrating nine state-of-the-art static analysis tools, i.e., SonarQube, CodeScene, PMD, Understand, CK, JaSoMe, RefactoringMiner, RefactoringMiner++, and PyRef, our dataset unifies over 700 unique metrics at method, class, file, and project levels. Covering a total of 63,586 analyzed project releases, SQuaD also provides version control and issue-tracking histories, software vulnerability data (CVE/CWE), and process metrics proven to enhance Just-In-Time (JIT) defect prediction. The SQuaD enables empirical research on maintainability, technical debt, software evolution, and quality assessment at unprecedented scale. We also outline emerging research directions, including automated dataset updates and cross-project quality modeling to support the continuous evolution of software analytics. The dataset is publicly available on ZENODO (DOI: https://doi.org/10.5281/zenodo.17566690). 

---
# Discovering Meaningful Units with Visually Grounded Semantics from Image Captions 

**Authors**: Melika Behjati, James Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2511.11262)  

**Abstract**: Fine-grained knowledge is crucial for vision-language models to obtain a better understanding of the real world. While there has been work trying to acquire this kind of knowledge in the space of vision and language, it has mostly focused on aligning the image patches with the tokens on the language side. However, image patches do not have any meaning to the human eye, and individual tokens do not necessarily carry groundable information in the image. It is groups of tokens which describe different aspects of the scene. In this work, we propose a model which groups the caption tokens as part of its architecture in order to capture a fine-grained representation of the language. We expect our representations to be at the level of objects present in the image, and therefore align our representations with the output of an image encoder trained to discover objects. We show that by learning to group the tokens, the vision-language model has a better fine-grained understanding of vision and language. In addition, the token groups that our model discovers are highly similar to groundable phrases in text, both qualitatively and quantitatively. 

---
# Multi-agent Undercover Gaming: Hallucination Removal via Counterfactual Test for Multimodal Reasoning 

**Authors**: Dayong Liang, Xiao-Yong Wei, Changmeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.11182)  

**Abstract**: Hallucination continues to pose a major obstacle in the reasoning capabilities of large language models (LLMs). Although the Multi-Agent Debate (MAD) paradigm offers a promising solution by promoting consensus among multiple agents to enhance reliability, it relies on the unrealistic assumption that all debaters are rational and reflective, which is a condition that may not hold when agents themselves are prone to hallucinations. To address this gap, we introduce the Multi-agent Undercover Gaming (MUG) protocol, inspired by social deduction games like "Who is Undercover?". MUG reframes MAD as a process of detecting "undercover" agents (those suffering from hallucinations) by employing multimodal counterfactual tests. Specifically, we modify reference images to introduce counterfactual evidence and observe whether agents can accurately identify these changes, providing ground-truth for identifying hallucinating agents and enabling robust, crowd-powered multimodal reasoning. MUG advances MAD protocols along three key dimensions: (1) enabling factual verification beyond statistical consensus through counterfactual testing; (2) introducing cross-evidence reasoning via dynamically modified evidence sources instead of relying on static inputs; and (3) fostering active reasoning, where agents engage in probing discussions rather than passively answering questions. Collectively, these innovations offer a more reliable and effective framework for multimodal reasoning in LLMs. The source code can be accessed at this https URL. 

---
# CLARITY: Contextual Linguistic Adaptation and Accent Retrieval for Dual-Bias Mitigation in Text-to-Speech Generation 

**Authors**: Crystal Min Hui Poon, Pai Chet Ng, Xiaoxiao Miao, Immanuel Jun Kai Loh, Bowen Zhang, Haoyu Song, Ian Mcloughlin  

**Link**: [PDF](https://arxiv.org/pdf/2511.11104)  

**Abstract**: Instruction-guided text-to-speech (TTS) research has reached a maturity level where excellent speech generation quality is possible on demand, yet two coupled biases persist: accent bias, where models default to dominant phonetic patterns, and linguistic bias, where dialect-specific lexical and cultural cues are ignored. These biases are interdependent, as authentic accent generation requires both accent fidelity and localized text. We present Contextual Linguistic Adaptation and Retrieval for Inclusive TTS sYnthesis (CLARITY), a backbone-agnostic framework that addresses these biases through dual-signal optimization: (i) contextual linguistic adaptation that localizes input text to the target dialect, and (ii) retrieval-augmented accent prompting (RAAP) that supplies accent-consistent speech prompts. Across twelve English accents, CLARITY improves accent accuracy and fairness while maintaining strong perceptual quality. 

---
# S2D-ALIGN: Shallow-to-Deep Auxiliary Learning for Anatomically-Grounded Radiology Report Generation 

**Authors**: Jiechao Gao, Chang Liu, Yuangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.11066)  

**Abstract**: Radiology Report Generation (RRG) aims to automatically generate diagnostic reports from radiology images. To achieve this, existing methods have leveraged the powerful cross-modal generation capabilities of Multimodal Large Language Models (MLLMs), primarily focusing on optimizing cross-modal alignment between radiographs and reports through Supervised Fine-Tuning (SFT). However, by only performing instance-level alignment with the image-text pairs, the standard SFT paradigm fails to establish anatomically-grounded alignment, where the templated nature of reports often leads to sub-optimal generation quality. To address this, we propose \textsc{S2D-Align}, a novel SFT paradigm that establishes anatomically-grounded alignment by leveraging auxiliary signals of varying granularities. \textsc{S2D-Align} implements a shallow-to-deep strategy, progressively enriching the alignment process: it begins with the coarse radiograph-report pairing, then introduces reference reports for instance-level guidance, and ultimately utilizes key phrases to ground the generation in specific anatomical details. To bridge the different alignment stages, we introduce a memory-based adapter that empowers feature sharing, thereby integrating coarse and fine-grained guidance. For evaluation, we conduct experiments on the public \textsc{MIMIC-CXR} and \textsc{IU X-Ray} benchmarks, where \textsc{S2D-Align} achieves state-of-the-art performance compared to existing methods. Ablation studies validate the effectiveness of our multi-stage, auxiliary-guided approach, highlighting a promising direction for enhancing grounding capabilities in complex, multi-modal generation tasks. 

---
# The Map of Misbelief: Tracing Intrinsic and Extrinsic Hallucinations Through Attention Patterns 

**Authors**: Elyes Hajji, Aymen Bouguerra, Fabio Arnez  

**Link**: [PDF](https://arxiv.org/pdf/2511.10837)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in safety-critical domains, yet remain susceptible to hallucinations. While prior works have proposed confidence representation methods for hallucination detection, most of these approaches rely on computationally expensive sampling strategies and often disregard the distinction between hallucination types. In this work, we introduce a principled evaluation framework that differentiates between extrinsic and intrinsic hallucination categories and evaluates detection performance across a suite of curated benchmarks. In addition, we leverage a recent attention-based uncertainty quantification algorithm and propose novel attention aggregation strategies that improve both interpretability and hallucination detection performance. Our experimental findings reveal that sampling-based methods like Semantic Entropy are effective for detecting extrinsic hallucinations but generally fail on intrinsic ones. In contrast, our method, which aggregates attention over input tokens, is better suited for intrinsic hallucinations. These insights provide new directions for aligning detection strategies with the nature of hallucination and highlight attention as a rich signal for quantifying model uncertainty. 

---
# From Efficiency to Adaptivity: A Deeper Look at Adaptive Reasoning in Large Language Models 

**Authors**: Chao Wu, Baoheng Li, Mingchen Gao, Zhenyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10788)  

**Abstract**: Recent advances in large language models (LLMs) have made reasoning a central benchmark for evaluating intelligence. While prior surveys focus on efficiency by examining how to shorten reasoning chains or reduce computation, this view overlooks a fundamental challenge: current LLMs apply uniform reasoning strategies regardless of task complexity, generating long traces for trivial problems while failing to extend reasoning for difficult tasks. This survey reframes reasoning through the lens of {adaptivity}: the capability to allocate reasoning effort based on input characteristics such as difficulty and uncertainty. We make three contributions. First, we formalize deductive, inductive, and abductive reasoning within the LLM context, connecting these classical cognitive paradigms with their algorithmic realizations. Second, we formalize adaptive reasoning as a control-augmented policy optimization problem balancing task performance with computational cost, distinguishing learned policies from inference-time control mechanisms. Third, we propose a systematic taxonomy organizing existing methods into training-based approaches that internalize adaptivity through reinforcement learning, supervised fine-tuning, and learned controllers, and training-free approaches that achieve adaptivity through prompt conditioning, feedback-driven halting, and modular composition. This framework clarifies how different mechanisms realize adaptive reasoning in practice and enables systematic comparison across diverse strategies. We conclude by identifying open challenges in self-evaluation, meta-reasoning, and human-aligned reasoning control. 

---
# PISanitizer: Preventing Prompt Injection to Long-Context LLMs via Prompt Sanitization 

**Authors**: Runpeng Geng, Yanting Wang, Chenlong Yin, Minhao Cheng, Ying Chen, Jinyuan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.10720)  

**Abstract**: Long context LLMs are vulnerable to prompt injection, where an attacker can inject an instruction in a long context to induce an LLM to generate an attacker-desired output. Existing prompt injection defenses are designed for short contexts. When extended to long-context scenarios, they have limited effectiveness. The reason is that an injected instruction constitutes only a very small portion of a long context, making the defense very challenging. In this work, we propose PISanitizer, which first pinpoints and sanitizes potential injected tokens (if any) in a context before letting a backend LLM generate a response, thereby eliminating the influence of the injected instruction. To sanitize injected tokens, PISanitizer builds on two observations: (1) prompt injection attacks essentially craft an instruction that compels an LLM to follow it, and (2) LLMs intrinsically leverage the attention mechanism to focus on crucial input tokens for output generation. Guided by these two observations, we first intentionally let an LLM follow arbitrary instructions in a context and then sanitize tokens receiving high attention that drive the instruction-following behavior of the LLM. By design, PISanitizer presents a dilemma for an attacker: the more effectively an injected instruction compels an LLM to follow it, the more likely it is to be sanitized by PISanitizer. Our extensive evaluation shows that PISanitizer can successfully prevent prompt injection, maintain utility, outperform existing defenses, is efficient, and is robust to optimization-based and strong adaptive attacks. The code is available at this https URL. 

---
# Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents 

**Authors**: Yuan Zhao, Hualei Zhu, Tingyu Jiang, Shen Li, Xiaohang Xu, Hao Henry Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10705)  

**Abstract**: Graphical User Interface (GUI) task automation constitutes a critical frontier in artificial intelligence research. While effective GUI agents synergistically integrate planning and grounding capabilities, current methodologies exhibit two fundamental limitations: (1) insufficient exploitation of cross-model synergies, and (2) over-reliance on synthetic data generation without sufficient utilization. To address these challenges, we propose Co-EPG, a self-iterative training framework for Co-Evolution of Planning and Grounding. Co-EPG establishes an iterative positive feedback loop: through this loop, the planning model explores superior strategies under grounding-based reward guidance via Group Relative Policy Optimization (GRPO), generating diverse data to optimize the grounding model. Concurrently, the optimized Grounding model provides more effective rewards for subsequent GRPO training of the planning model, fostering continuous improvement. Co-EPG thus enables iterative enhancement of agent capabilities through self-play optimization and training data distillation. On the Multimodal-Mind2Web and AndroidControl benchmarks, our framework outperforms existing state-of-the-art methods after just three iterations without requiring external data. The agent consistently improves with each iteration, demonstrating robust self-enhancement capabilities. This work establishes a novel training paradigm for GUI agents, shifting from isolated optimization to an integrated, self-driven co-evolution approach. 

---
# Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents 

**Authors**: Chih-Hsuan Yang, Tanwi Mallick, Le Chen, Krishnan Raghavan, Azton Wells, Amal Gueroudji, Ian T. Foster, Rajeev Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2511.10687)  

**Abstract**: Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work. 

---
