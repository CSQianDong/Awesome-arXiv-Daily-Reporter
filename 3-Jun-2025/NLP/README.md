# DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation 

**Authors**: Jennifer Chen, Aidar Myrzakhan, Yaxin Luo, Hassaan Muhammad Khan, Sondos Mahmoud Bsharat, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01954)  

**Abstract**: Retrieval-Augmented Generation (RAG) methods have proven highly effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems consume significant computational resources and are prone to generating hallucinated content from Humans. In this work, we introduce $\texttt{DRAG}$, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph-based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model's predictions with a structured knowledge graph and ranked evidence, $\texttt{DRAG}$ effectively mitigates hallucinations and improves factual accuracy. We further present a case demonstrating how our framework mitigates user privacy risks and introduce a corresponding benchmark. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability. With $\texttt{DRAG}$, we provide a practical and resource-efficient roadmap to deploying enhanced retrieval and generation capabilities in small-sized LLMs. 

---
# WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks 

**Authors**: Atsuyuki Miyai, Zaiying Zhao, Kazuki Egashira, Atsuki Sato, Tatsumi Sunada, Shota Onohara, Hiromasa Yamanishi, Mashiro Toyooka, Kunato Nishina, Ryoma Maeda, Kiyoharu Aizawa, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.01952)  

**Abstract**: Powered by a large language model (LLM), a web browsing agent operates web browsers in a human-like manner and offers a highly transparent path toward automating a wide range of everyday tasks. As web agents become increasingly capable and demonstrate proficiency in general browsing tasks, a critical question emerges: Can they go beyond general browsing to robustly handle tasks that are tedious and complex, or chores that humans often avoid doing themselves? In this paper, we introduce WebChoreArena, a new fully reproducible benchmark comprising 532 carefully curated tasks designed to extend the scope of WebArena beyond general browsing to more labor-intensive and tedious tasks. WebChoreArena systematically integrates three key challenges: (i) Massive Memory tasks requiring accurate retrieval of large amounts of information in the observations, (ii) Calculation tasks demanding precise mathematical reasoning, and (iii) Long-Term Memory tasks necessitating long-term memory across multiple webpages. Built on top of the fully reproducible and widely adopted four WebArena simulation environments, WebChoreArena ensures strict reproducibility and enables fair, direct comparisons with the established WebArena benchmark, offering key insights into agent progress. Our experimental results demonstrate that as LLMs evolve, represented by GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro, significant improvements in performance are observed on WebChoreArena. These findings suggest that WebChoreArena is well-suited to measure the advancement of state-of-the-art LLMs with greater clarity. Nevertheless, the results also indicate that even with Gemini 2.5 Pro, there remains substantial room for improvement compared to WebArena, highlighting the increased challenges posed by WebChoreArena. 

---
# Self-ensemble: Mitigating Confidence Distortion for Large Language Models 

**Authors**: Zicheng Xu, Guanchu Wang, Guangyao Zheng, Yu-Neng Chuang, Alexander Szalay, Xia Hu, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2506.01951)  

**Abstract**: Although Large Language Models (LLMs) perform well in general fields, they exhibit a confidence distortion problem on multi-choice question-answering (MCQA), particularly as the number of answer choices increases. Specifically, on MCQA with many choices, LLMs suffer from under-confidence in correct predictions and over-confidence in incorrect ones, leading to a substantially degraded performance. To solve this problem, we propose Self-ensemble in this work. Our method splits the choices into several groups and ensembles LLM predictions across these groups to reach a final decision. The advantage of Self-ensemble is its plug-and-play nature, where it can be integrated into existing LLM architecture based on a designed attention mask and positional encoding, without requiring labeled datasets for parameter tuning. Experimental results on three LLMs and datasets demonstrate that Self-ensemble comprehensively addresses the confidence distortion problem of LLMs, outperforming standard inference as well as baseline methods. 

---
# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning 

**Authors**: Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01939)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning. 

---
# Novel Benchmark for NER in the Wastewater and Stormwater Domain 

**Authors**: Franco Alberto Cardillo, Franca Debole, Francesca Frontini, Mitra Aelami, Nanée Chahinian, Serge Conrad  

**Link**: [PDF](https://arxiv.org/pdf/2506.01938)  

**Abstract**: Effective wastewater and stormwater management is essential for urban sustainability and environmental protection. Extracting structured knowledge from reports and regulations is challenging due to domainspecific terminology and multilingual contexts. This work focuses on domain-specific Named Entity Recognition (NER) as a first step towards effective relation and information extraction to support decision making. A multilingual benchmark is crucial for evaluating these methods. This study develops a French-Italian domain-specific text corpus for wastewater management. It evaluates state-of-the-art NER methods, including LLM-based approaches, to provide a reliable baseline for future strategies and explores automated annotation projection in view of an extension of the corpus to new languages. 

---
# RewardBench 2: Advancing Reward Model Evaluation 

**Authors**: Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A. Smith, Hannaneh Hajishirzi, Nathan Lambert  

**Link**: [PDF](https://arxiv.org/pdf/2506.01937)  

**Abstract**: Reward models are used throughout the post-training of language models to capture nuanced signals from preference data and provide a training target for optimization across instruction following, reasoning, safety, and more domains. The community has begun establishing best practices for evaluating reward models, from the development of benchmarks that test capabilities in specific skill areas to others that test agreement with human preferences. At the same time, progress in evaluation has not been mirrored by the effectiveness of reward models in downstream tasks -- simpler direct alignment algorithms are reported to work better in many cases. This paper introduces RewardBench 2, a new multi-skill reward modeling benchmark designed to bring new, challenging data for accuracy-based reward model evaluation -- models score about 20 points on average lower on RewardBench 2 compared to the first RewardBench -- while being highly correlated with downstream performance. Compared to most other benchmarks, RewardBench 2 sources new human prompts instead of existing prompts from downstream evaluations, facilitating more rigorous evaluation practices. In this paper, we describe our benchmark construction process and report how existing models perform on it, while quantifying how performance on the benchmark correlates with downstream use of the models in both inference-time scaling algorithms, like best-of-N sampling, and RLHF training algorithms like proximal policy optimization. 

---
# Esoteric Language Models 

**Authors**: Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, Johnna Liu, Deepansha Singh, Zhoujun Cheng, Zhengzhong Liu, Eric Xing, John Thickstun, Arash Vahdat  

**Link**: [PDF](https://arxiv.org/pdf/2506.01928)  

**Abstract**: Diffusion-based language models offer a compelling alternative to autoregressive (AR) models by enabling parallel and controllable generation. Among this family of models, Masked Diffusion Models (MDMs) achieve the strongest performance but still underperform AR models in perplexity and lack key inference-time efficiency features--most notably, KV caching. In this work, we introduce Eso-LMs, a new family of models that fuses AR and MDM paradigms, enabling smooth interpolation between their perplexities while overcoming their respective limitations. Eso-LMs set a new state of the art on standard language modeling benchmarks. Crucially, we are the **first to introduce KV caching for MDMs** while preserving parallel generation, significantly improving inference efficiency. Combined with an optimized sampling schedule, our method achieves up to **65x** faster inference than standard MDMs and **4x** faster inference than prior semi-autoregressive approaches. We provide the code and model checkpoints on the project page: [this http URL](this http URL) 

---
# From Guidelines to Practice: A New Paradigm for Arabic Language Model Evaluation 

**Authors**: Serry Sibaee, Omer Nacar, Adel Ammar, Yasser Al-Habashi, Abdulrahman Al-Batati, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2506.01920)  

**Abstract**: This paper addresses critical gaps in Arabic language model evaluation by establishing comprehensive theoretical guidelines and introducing a novel evaluation framework. We first analyze existing Arabic evaluation datasets, identifying significant issues in linguistic accuracy, cultural alignment, and methodological rigor. To address these limitations in LLMs, we present the Arabic Depth Mini Dataset (ADMD), a carefully curated collection of 490 challenging questions spanning ten major domains (42 sub-domains, see Figure 1. Using ADMD, we evaluate five leading language models: GPT-4, Claude 3.5 Sonnet, Gemini Flash 1.5, CommandR 100B, and Qwen-Max. Our results reveal significant variations in model performance across different domains, with particular challenges in areas requiring deep cultural understanding and specialized knowledge. Claude 3.5 Sonnet demonstrated the highest overall accuracy at 30\%, showing relative strength in mathematical theory in Arabic, Arabic language, and islamic domains. This work provides both theoretical foundations and practical insights for improving Arabic language model evaluation, emphasizing the importance of cultural competence alongside technical capabilities. 

---
# Spatial Coordinates as a Cell Language: A Multi-Sentence Framework for Imaging Mass Cytometry Analysis 

**Authors**: Chi-Jane Chen, Yuhang Chen, Sukwon Yun, Natalie Stanley, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01918)  

**Abstract**: Image mass cytometry (IMC) enables high-dimensional spatial profiling by combining mass cytometry's analytical power with spatial distributions of cell phenotypes. Recent studies leverage large language models (LLMs) to extract cell states by translating gene or protein expression into biological context. However, existing single-cell LLMs face two major challenges: (1) Integration of spatial information: they struggle to generalize spatial coordinates and effectively encode spatial context as text, and (2) Treating each cell independently: they overlook cell-cell interactions, limiting their ability to capture biological relationships. To address these limitations, we propose Spatial2Sentence, a novel framework that integrates single-cell expression and spatial information into natural language using a multi-sentence approach. Spatial2Sentence constructs expression similarity and distance matrices, pairing spatially adjacent and expressionally similar cells as positive pairs while using distant and dissimilar cells as negatives. These multi-sentence representations enable LLMs to learn cellular interactions in both expression and spatial contexts. Equipped with multi-task learning, Spatial2Sentence outperforms existing single-cell LLMs on preprocessed IMC datasets, improving cell-type classification by 5.98% and clinical status prediction by 4.18% on the diabetes dataset while enhancing interpretability. The source code can be found here: this https URL. 

---
# Is Extending Modality The Right Path Towards Omni-Modality? 

**Authors**: Tinghui Zhu, Kai Zhang, Muhao Chen, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.01872)  

**Abstract**: Omni-modal language models (OLMs) aim to integrate and reason over diverse input modalities--such as text, images, video, and audio--while maintaining strong language capabilities. Despite recent advancements, existing models, especially open-source ones, remain far from true omni-modality, struggling to generalize beyond the specific modality pairs they are trained on or to achieve strong performance when processing multi-modal inputs. We study the effect of extending modality, the dominant technique for training multimodal models, where an off-the-shelf language model is fine-tuned on target-domain and language data. Specifically, we investigate three key questions: (1) Does modality extension compromise core language abilities? (2) Can model merging effectively integrate independently fine-tuned modality-specific models to achieve omni-modality? (3) Does omni-modality extension lead to better knowledge sharing and generalization compared to sequential extension? Through extensive experiments, we analyze these trade-offs and provide insights into the feasibility of achieving true omni-modality using current approaches. 

---
# CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions 

**Authors**: Tamer Alkhouli, Katerina Margatina, James Gung, Raphael Shu, Claudia Zaghi, Monica Sunkara, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01859)  

**Abstract**: We introduce Conversational Function-Calling Evaluation Through Turn-Level Interactions (CONFETTI), a conversational benchmark1 designed to evaluate the function-calling capabilities and response quality of large language models (LLMs). Current benchmarks lack comprehensive assessment of LLMs in complex conversational scenarios. CONFETTI addresses this gap through 109 human-simulated conversations, comprising 313 user turns and covering 86 APIs. These conversations explicitly target various conversational complexities, such as follow-ups, goal correction and switching, ambiguous and implicit goals. We perform off-policy turn-level evaluation using this benchmark targeting function-calling. Our benchmark also incorporates dialog act annotations to assess agent responses. We evaluate a series of state-of-the-art LLMs and analyze their performance with respect to the number of available APIs, conversation lengths, and chained function calling. Our results reveal that while some models are able to handle long conversations, and leverage more than 20+ APIs successfully, other models struggle with longer context or when increasing the number of APIs. We also report that the performance on chained function-calls is severely limited across the models. Overall, the top performing models on CONFETTI are Nova Pro (40.01%), Claude Sonnet v3.5 (35.46%) and Llama 3.1 405B (33.19%) followed by command-r-plus (31.18%) and Mistral-Large-2407 (30.07%). 

---
# Code-Switching and Syntax: A Large-Scale Experiment 

**Authors**: Igor Sterner, Simone Teufel  

**Link**: [PDF](https://arxiv.org/pdf/2506.01846)  

**Abstract**: The theoretical code-switching (CS) literature provides numerous pointwise investigations that aim to explain patterns in CS, i.e. why bilinguals switch language in certain positions in a sentence more often than in others. A resulting consensus is that CS can be explained by the syntax of the contributing languages. There is however no large-scale, multi-language, cross-phenomena experiment that tests this claim. When designing such an experiment, we need to make sure that the system that is predicting where bilinguals tend to switch has access only to syntactic information. We provide such an experiment here. Results show that syntax alone is sufficient for an automatic system to distinguish between sentences in minimal pairs of CS, to the same degree as bilingual humans. Furthermore, the learnt syntactic patterns generalise well to unseen language pairs. 

---
# Minimal Pair-Based Evaluation of Code-Switching 

**Authors**: Igor Sterner, Simone Teufel  

**Link**: [PDF](https://arxiv.org/pdf/2506.01840)  

**Abstract**: There is a lack of an evaluation methodology that estimates the extent to which large language models (LLMs) use code-switching (CS) in the same way as bilinguals. Existing methods do not have wide language coverage, fail to account for the diverse range of CS phenomena, or do not scale. We propose an intervention based on minimal pairs of CS. Each minimal pair contains one naturally occurring CS sentence and one minimally manipulated variant. We collect up to 1,000 such pairs each for 11 language pairs. Our human experiments show that, for every language pair, bilinguals consistently prefer the naturally occurring CS sentence. Meanwhile our experiments with current LLMs show that the larger the model, the more consistently it assigns higher probability to the naturally occurring CS sentence than to the variant. In accordance with theoretical claims, the largest probability differences arise in those pairs where the manipulated material consisted of closed-class words. 

---
# CiteEval: Principle-Driven Citation Evaluation for Source Attribution 

**Authors**: Yumo Xu, Peng Qi, Jifan Chen, Kunlun Liu, Rujun Han, Lan Liu, Bonan Min, Vittorio Castelli, Arshit Gupta, Zhiguo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01829)  

**Abstract**: Citation quality is crucial in information-seeking systems, directly influencing trust and the effectiveness of information access. Current evaluation frameworks, both human and automatic, mainly rely on Natural Language Inference (NLI) to assess binary or ternary supportiveness from cited sources, which we argue is a suboptimal proxy for citation evaluation. In this work we introduce CiteEval, a citation evaluation framework driven by principles focusing on fine-grained citation assessment within a broad context, encompassing not only the cited sources but the full retrieval context, user query, and generated text. Guided by the proposed framework, we construct CiteBench, a multi-domain benchmark with high-quality human annotations on citation quality. To enable efficient evaluation, we further develop CiteEval-Auto, a suite of model-based metrics that exhibit strong correlation with human judgments. Experiments across diverse systems demonstrate CiteEval-Auto's superior ability to capture the multifaceted nature of citations compared to existing metrics, offering a principled and scalable approach to evaluate and improve model-generated citations. 

---
# Not All Jokes Land: Evaluating Large Language Models Understanding of Workplace Humor 

**Authors**: Moahmmadamin Shafiei, Hamidreza Saffari  

**Link**: [PDF](https://arxiv.org/pdf/2506.01819)  

**Abstract**: With the recent advances in Artificial Intelligence (AI) and Large Language Models (LLMs), the automation of daily tasks, like automatic writing, is getting more and more attention. Hence, efforts have focused on aligning LLMs with human values, yet humor, particularly professional industrial humor used in workplaces, has been largely neglected. To address this, we develop a dataset of professional humor statements along with features that determine the appropriateness of each statement. Our evaluation of five LLMs shows that LLMs often struggle to judge the appropriateness of humor accurately. 

---
# BD at BEA 2025 Shared Task: MPNet Ensembles for Pedagogical Mistake Identification and Localization in AI Tutor Responses 

**Authors**: Shadman Rohan, Ishita Sur Apan, Muhtasim Ibteda Shochcho, Md Fahim, Mohammad Ashfaq Ur Rahman, AKM Mahbubur Rahman, Amin Ahsan Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.01817)  

**Abstract**: We present Team BD's submission to the BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors, under Track 1 (Mistake Identification) and Track 2 (Mistake Location). Both tracks involve three-class classification of tutor responses in educational dialogues - determining if a tutor correctly recognizes a student's mistake (Track 1) and whether the tutor pinpoints the mistake's location (Track 2). Our system is built on MPNet, a Transformer-based language model that combines BERT and XLNet's pre-training advantages. We fine-tuned MPNet on the task data using a class-weighted cross-entropy loss to handle class imbalance, and leveraged grouped cross-validation (10 folds) to maximize the use of limited data while avoiding dialogue overlap between training and validation. We then performed a hard-voting ensemble of the best models from each fold, which improves robustness and generalization by combining multiple classifiers. Our approach achieved strong results on both tracks, with exact-match macro-F1 scores of approximately 0.7110 for Mistake Identification and 0.5543 for Mistake Location on the official test set. We include comprehensive analysis of our system's performance, including confusion matrices and t-SNE visualizations to interpret classifier behavior, as well as a taxonomy of common errors with examples. We hope our ensemble-based approach and findings provide useful insights for designing reliable tutor response evaluation systems in educational dialogue settings. 

---
# Analysis of LLM Bias (Chinese Propaganda & Anti-US Sentiment) in DeepSeek-R1 vs. ChatGPT o3-mini-high 

**Authors**: PeiHsuan Huang, ZihWei Lin, Simon Imbot, WenCheng Fu, Ethan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01814)  

**Abstract**: Large language models (LLMs) increasingly shape public understanding and civic decisions, yet their ideological neutrality is a growing concern. While existing research has explored various forms of LLM bias, a direct, cross-lingual comparison of models with differing geopolitical alignments-specifically a PRC-system model versus a non-PRC counterpart-has been lacking. This study addresses this gap by systematically evaluating DeepSeek-R1 (PRC-aligned) against ChatGPT o3-mini-high (non-PRC) for Chinese-state propaganda and anti-U.S. sentiment. We developed a novel corpus of 1,200 de-contextualized, reasoning-oriented questions derived from Chinese-language news, presented in Simplified Chinese, Traditional Chinese, and English. Answers from both models (7,200 total) were assessed using a hybrid evaluation pipeline combining rubric-guided GPT-4o scoring with human annotation. Our findings reveal significant model-level and language-dependent biases. DeepSeek-R1 consistently exhibited substantially higher proportions of both propaganda and anti-U.S. bias compared to ChatGPT o3-mini-high, which remained largely free of anti-U.S. sentiment and showed lower propaganda levels. For DeepSeek-R1, Simplified Chinese queries elicited the highest bias rates; these diminished in Traditional Chinese and were nearly absent in English. Notably, DeepSeek-R1 occasionally responded in Simplified Chinese to Traditional Chinese queries and amplified existing PRC-aligned terms in its Chinese answers, demonstrating an "invisible loudspeaker" effect. Furthermore, such biases were not confined to overtly political topics but also permeated cultural and lifestyle content, particularly in DeepSeek-R1. 

---
# NAVER LABS Europe Submission to the Instruction-following Track 

**Authors**: Beomseok Lee, Marcely Zanon Boito, Laurent Besacier, Ioan Calapodescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01808)  

**Abstract**: In this paper we describe NAVER LABS Europe submission to the instruction-following speech processing short track at IWSLT 2025. We participate in the constrained settings, developing systems that can simultaneously perform ASR, ST, and SQA tasks from English speech input into the following target languages: Chinese, Italian, and German. Our solution leverages two pretrained modules: (1) a speech-to-LLM embedding projector trained using representations from the SeamlessM4T-v2-large speech encoder; and (2) LoRA adapters trained on text data on top of a Llama-3.1-8B-Instruct. These modules are jointly loaded and further instruction-tuned for 1K steps on multilingual and multimodal data to form our final system submitted for evaluation. 

---
# Propaganda and Information Dissemination in the Russo-Ukrainian War: Natural Language Processing of Russian and Western Twitter Narratives 

**Authors**: Zaur Gouliev  

**Link**: [PDF](https://arxiv.org/pdf/2506.01807)  

**Abstract**: The conflict in Ukraine has been not only characterised by military engagement but also by a significant information war, with social media platforms like X, formerly known as Twitter playing an important role in shaping public perception. This article provides an analysis of tweets from propaganda accounts and trusted accounts collected from the onset of the war, February 2022 until the middle of May 2022 with n=40,000 total tweets. We utilise natural language processing and machine learning algorithms to assess the sentiment and identify key themes, topics and narratives across the dataset with human-in-the-loop (HITL) analysis throughout. Our findings indicate distinct strategies in how information is created, spread, and targeted at different audiences by both sides. Propaganda accounts frequently employ emotionally charged language and disinformation to evoke fear and distrust, whereas other accounts, primarily Western tend to focus on factual reporting and humanitarian aspects of the conflict. Clustering analysis reveals groups of accounts with similar behaviours, which we suspect indicates the presence of coordinated efforts. This research attempts to contribute to our understanding of the dynamics of information warfare and offers techniques for future studies on social media influence in military conflicts. 

---
# Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books 

**Authors**: Chen Zhang, Jiuheng Lin, Xiao Liu, Zekai Zhang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01796)  

**Abstract**: While large language models (LLMs) have shown promise in translating extremely low-resource languages using resources like dictionaries, the effectiveness of grammar books remains debated. This paper investigates the role of grammar books in translating extremely low-resource languages by decomposing it into two key steps: grammar rule retrieval and application. To facilitate the study, we introduce ZhuangRules, a modularized dataset of grammar rules and their corresponding test sentences. Our analysis reveals that rule retrieval constitutes a primary bottleneck in grammar-based translation. Moreover, although LLMs can apply simple rules for translation when explicitly provided, they encounter difficulties in handling more complex rules. To address these challenges, we propose representing grammar rules as code functions, considering their similarities in structure and the benefit of code in facilitating LLM reasoning. Our experiments show that using code rules significantly boosts both rule retrieval and application, ultimately resulting in a 13.1% BLEU improvement in translation. 

---
# Human-Centric Evaluation for Foundation Models 

**Authors**: Yijin Guo, Kaiyuan Ji, Xiaorong Zhu, Junying Wang, Farong Wen, Chunyi Li, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.01793)  

**Abstract**: Currently, nearly all evaluations of foundation models focus on objective metrics, emphasizing quiz performance to define model capabilities. While this model-centric approach enables rapid performance assessment, it fails to reflect authentic human experiences. To address this gap, we propose a Human-Centric subjective Evaluation (HCE) framework, focusing on three core dimensions: problem-solving ability, information quality, and interaction experience. Through experiments involving Deepseek R1, OpenAI o3 mini, Grok 3, and Gemini 2.5, we conduct over 540 participant-driven evaluations, where humans and models collaborate on open-ended research tasks, yielding a comprehensive subjective dataset. This dataset captures diverse user feedback across multiple disciplines, revealing distinct model strengths and adaptability. Our findings highlight Grok 3's superior performance, followed by Deepseek R1 and Gemini 2.5, with OpenAI o3 mini lagging behind. By offering a novel framework and a rich dataset, this study not only enhances subjective evaluation methodologies but also lays the foundation for standardized, automated assessments, advancing LLM development for research and practical scenarios. Our dataset link is this https URL. 

---
# iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01784)  

**Abstract**: While Large Language Models (LLMs) excel at many natural language processing tasks, they often suffer from factual inaccuracies in knowledge-intensive scenarios. Integrating external knowledge resources, particularly knowledge graphs (KGs), provides a transparent and updatable foundation for more reliable reasoning. Knowledge Base Question Answering (KBQA), which queries and reasons over KGs, is central to this effort, especially for complex, multi-hop queries. However, multi-hop reasoning poses two key challenges: (1)~maintaining coherent reasoning paths, and (2)~avoiding prematurely discarding critical multi-hop connections. To address these issues, we introduce iQUEST, a question-guided KBQA framework that iteratively decomposes complex queries into simpler sub-questions, ensuring a structured and focused reasoning trajectory. Additionally, we integrate a Graph Neural Network (GNN) to look ahead and incorporate 2-hop neighbor information at each reasoning step. This dual approach strengthens the reasoning process, enabling the model to explore viable paths more effectively. Detailed experiments demonstrate the consistent improvement delivered by iQUEST across four benchmark datasets and four LLMs. 

---
# MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation 

**Authors**: Yile Liu, Ziwei Ma, Xiu Jiang, Jinglu Hu, Jing Chang, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01776)  

**Abstract**: With the rapid adoption of large language models (LLMs) in natural language processing, the ability to follow instructions has emerged as a key metric for evaluating their practical utility. However, existing evaluation methods often focus on single-language scenarios, overlooking the challenges and differences present in multilingual and cross-lingual contexts. To address this gap, we introduce MaXIFE: a comprehensive evaluation benchmark designed to assess instruction-following capabilities across 23 languages with 1,667 verifiable instruction tasks. MaXIFE integrates both Rule-Based Evaluation and Model-Based Evaluation, ensuring a balance of efficiency and accuracy. We applied MaXIFE to evaluate several leading commercial and open-source LLMs, establishing baseline results for future comparisons. By providing a standardized tool for multilingual instruction-following evaluation, MaXIFE aims to advance research and development in natural language processing. 

---
# Developing a Mixed-Methods Pipeline for Community-Oriented Digitization of Kwak'wala Legacy Texts 

**Authors**: Milind Agarwal, Daisy Rosenblum, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.01775)  

**Abstract**: Kwak'wala is an Indigenous language spoken in British Columbia, with a rich legacy of published documentation spanning more than a century, and an active community of speakers, teachers, and learners engaged in language revitalization. Over 11 volumes of the earliest texts created during the collaboration between Franz Boas and George Hunt have been scanned but remain unreadable by machines. Complete digitization through optical character recognition has the potential to facilitate transliteration into modern orthographies and the creation of other language technologies. In this paper, we apply the latest OCR techniques to a series of Kwak'wala texts only accessible as images, and discuss the challenges and unique adaptations necessary to make such technologies work for these real-world texts. Building on previous methods, we propose using a mix of off-the-shelf OCR methods, language identification, and masking to effectively isolate Kwak'wala text, along with post-correction models, to produce a final high-quality transcription. 

---
# Thinking in Character: Advancing Role-Playing Agents with Role-Aware Reasoning 

**Authors**: Yihong Tang, Kehai Chen, Muyun Yang, Zhengyu Niu, Jing Li, Tiejun Zhao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01748)  

**Abstract**: The advancement of Large Language Models (LLMs) has spurred significant interest in Role-Playing Agents (RPAs) for applications such as emotional companionship and virtual interaction. However, recent RPAs are often built on explicit dialogue data, lacking deep, human-like internal thought processes, resulting in superficial knowledge and style expression. While Large Reasoning Models (LRMs) can be employed to simulate character thought, their direct application is hindered by attention diversion (i.e., RPAs forget their role) and style drift (i.e., overly formal and rigid reasoning rather than character-consistent reasoning). To address these challenges, this paper introduces a novel Role-Aware Reasoning (RAR) method, which consists of two important stages: Role Identity Activation (RIA) and Reasoning Style Optimization (RSO). RIA explicitly guides the model with character profiles during reasoning to counteract attention diversion, and then RSO aligns reasoning style with the character and scene via LRM distillation to mitigate style drift. Extensive experiments demonstrate that the proposed RAR significantly enhances the performance of RPAs by effectively addressing attention diversion and style drift. 

---
# Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs 

**Authors**: Jiandong Shao, Yao Lu, Jianfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01734)  

**Abstract**: Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law -- a statistical pattern where lower digits occur more frequently as leading digits -- we hypothesize that the long-tailed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark with uniformly distributed ground-truth digits across seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks. 

---
# Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training 

**Authors**: Pierre-Carl Langlais, Carlos Rosas Hinostroza, Mattia Nee, Catherine Arnett, Pavel Chizhov, Eliot Krzystof Jones, Irène Girard, David Mach, Anastasia Stasenko, Ivan P. Yamshchikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.01732)  

**Abstract**: Large Language Models (LLMs) are pre-trained on large amounts of data from different sources and domains. These data most often contain trillions of tokens with large portions of copyrighted or proprietary content, which hinders the usage of such models under AI legislation. This raises the need for truly open pre-training data that is compliant with the data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for language model pre-training. The data assembled in Common Corpus are either uncopyrighted or under permissible licenses and amount to about two trillion tokens. The dataset contains a wide variety of languages, ranging from the main European languages to low-resource ones rarely present in pre-training datasets; in addition, it includes a large portion of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepreneurial needs in diverse areas of knowledge. In this technical report, we present the detailed provenance of data assembling and the details of dataset filtering and curation. Being already used by such industry leaders as Anthropic and multiple LLM training projects, we believe that Common Corpus will become a critical infrastructure for open science research in LLMs. 

---
# Tug-of-war between idiom's figurative and literal meanings in LLMs 

**Authors**: Soyoung Oh, Xinting Huang, Mathis Pink, Michael Hahn, Vera Demberg  

**Link**: [PDF](https://arxiv.org/pdf/2506.01723)  

**Abstract**: Idioms present a unique challenge for language models due to their non-compositional figurative meanings, which often strongly diverge from the idiom's literal interpretation. This duality requires a model to learn representing and deciding between the two meanings to interpret an idiom in a figurative sense, or literally. In this paper, we employ tools from mechanistic interpretability to trace how a large pretrained causal transformer (LLama3.2-1B-base) deals with this ambiguity. We localize three steps of idiom processing: First, the idiom's figurative meaning is retrieved in early attention and MLP sublayers. We identify specific attention heads which boost the figurative meaning of the idiom while suppressing the idiom's literal interpretation. The model subsequently represents the figurative representation through an intermediate path. Meanwhile, a parallel bypass route forwards literal interpretation, ensuring that a both reading remain available. Overall, our findings provide a mechanistic evidence for idiom comprehension in an autoregressive transformer. 

---
# SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning 

**Authors**: Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, Yangfan He, Mi Zhang, Shen Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01713)  

**Abstract**: Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality. 

---
# Reasoning-Table: Exploring Reinforcement Learning for Table Reasoning 

**Authors**: Fangyu Lei, Jinxiang Meng, Yiming Huang, Tinghong Chen, Yun Zhang, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01710)  

**Abstract**: Table reasoning, encompassing tasks such as table question answering, fact verification, and text-to-SQL, requires precise understanding of structured tabular data, coupled with numerical computation and code manipulation for effective inference. Supervised fine-tuning (SFT) approaches have achieved notable success but often struggle with generalization and robustness due to biases inherent in imitative learning. We introduce Reasoning-Table, the first application of reinforcement learning (RL) to table reasoning, achieving state-of-the-art performance. Through rigorous data preprocessing, reward design, and tailored training strategies, our method leverages simple rule-based outcome rewards to outperform SFT across multiple benchmarks. Unified training across diverse tasks enables Reasoning-Table to emerge as a robust table reasoning large language model, surpassing larger proprietary models like Claude-3.7-Sonnet by 4.0% on table reasoning benchmarks. The approach also achieves excellent performance on text-to-SQL tasks, reaching 68.3% performance on the BIRD dev dataset with a 7B model. Further experiments demonstrate that Reasoning-Table enhances the model's generalization capabilities and robustness. 

---
# Fairness Dynamics During Training 

**Authors**: Krishna Patel, Nivedha Sivakumar, Barry-John Theobald, Luca Zappella, Nicholas Apostoloff  

**Link**: [PDF](https://arxiv.org/pdf/2506.01709)  

**Abstract**: We investigate fairness dynamics during Large Language Model (LLM) training to enable the diagnoses of biases and mitigations through training interventions like early stopping; we find that biases can emerge suddenly and do not always follow common performance metrics. We introduce two new metrics to evaluate fairness dynamics holistically during model pre-training: Average Rank and Jensen-Shannon Divergence by Parts. These metrics provide insights into the Pythia models' progression of biases in gender prediction of occupations on the WinoBias dataset. By monitoring these dynamics, we find that (1) Pythia-6.9b is biased towards men; it becomes more performant and confident predicting "male" than "female" during training, (2) via early-stopping, Pythia-6.9b can exchange 1.7% accuracy on LAMBADA for a 92.5% increase in fairness, and (3) larger models can exhibit more bias; Pythia-6.9b makes more assumptions about gender than Pythia-160m, even when a subject's gender is not specified. 

---
# mdok of KInIT: Robustly Fine-tuned LLM for Binary and Multiclass AI-Generated Text Detection 

**Authors**: Dominik Macko  

**Link**: [PDF](https://arxiv.org/pdf/2506.01702)  

**Abstract**: The large language models (LLMs) are able to generate high-quality texts in multiple languages. Such texts are often not recognizable by humans as generated, and therefore present a potential of LLMs for misuse (e.g., plagiarism, spams, disinformation spreading). An automated detection is able to assist humans to indicate the machine-generated texts; however, its robustness to out-of-distribution data is still challenging. This notebook describes our mdok approach in robust detection, based on fine-tuning smaller LLMs for text classification. It is applied to both subtasks of Voight-Kampff Generative AI Detection 2025, providing remarkable performance in binary detection as well as in multiclass (1st rank) classification of various cases of human-AI collaboration. 

---
# When LLMs Team Up: The Emergence of Collaborative Affective Computing 

**Authors**: Wenna Lai, Haoran Xie, Guandong Xu, Qing Li, S. Joe Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01698)  

**Abstract**: Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence. 

---
# StochasTok: Improving Fine-Grained Subword Understanding in LLMs 

**Authors**: Anya Sims, Thom Foster, Klara Kaleb, Tuan-Duy H. Nguyen, Joseph Lee, Jakob N. Foerster, Yee Whye Teh, Cong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01687)  

**Abstract**: Subword-level understanding is integral to numerous tasks, including understanding multi-digit numbers, spelling mistakes, abbreviations, rhyming, and wordplay. Despite this, current large language models (LLMs) still often struggle with seemingly simple subword-level tasks like How many 'r's in 'strawberry'?. A key factor behind these failures is tokenization which obscures the fine-grained structure of words. Current alternatives, such as character-level and dropout tokenization methods, significantly increase computational costs and provide inconsistent improvements. In this paper we revisit tokenization and introduce StochasTok, a simple, efficient stochastic tokenization scheme that randomly splits tokens during training, allowing LLMs to 'see' their internal structure. Our experiments show that pretraining with StochasTok substantially improves LLMs' downstream performance across multiple subword-level language games, including character counting, substring identification, and math tasks. Furthermore, StochasTok's simplicity allows seamless integration at any stage of the training pipeline; and we demonstrate that post-training with StochasTok can instill improved subword understanding into existing pretrained models, thus avoiding costly pretraining from scratch. These dramatic improvements achieved with a minimal change suggest StochasTok holds exciting potential when applied to larger, more capable models. Code open-sourced at: this https URL. 

---
# Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon 

**Authors**: Chen Zhang, Zhiyuan Liao, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01675)  

**Abstract**: Despite substantial research efforts evaluating how well large language models~(LLMs) handle global cultural diversity, the mechanisms behind their cultural knowledge acquisition, particularly in multilingual settings, remain unclear. We study this question by investigating how cultural knowledge transfers across languages during language adaptation of LLMs. We introduce an interpretable framework for studying this transfer, ensuring training data transparency and controlling transfer effects. Through a study of four non-Anglophonic cultures, we observe bidirectional cultural transfer between English and other high-resource languages, while low-resource languages primarily transfer knowledge to English with limited reverse flow. To explain this asymmetric phenomenon, we propose a frequency-based hypothesis: cultural knowledge appearing more frequently in the pretraining data transfers more easily, which is supported by empirical analysis of the training corpora. 

---
# ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge 

**Authors**: Chaoyue He, Xin Zhou, Yi Wu, Xinjia Yu, Yan Zhang, Lei Zhang, Di Wang, Shengfei Lyu, Hong Xu, Xiaoqiao Wang, Wei Liu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01646)  

**Abstract**: We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics. 

---
# Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons 

**Authors**: Frederick Riemenschneider, Anette Frank  

**Link**: [PDF](https://arxiv.org/pdf/2506.01629)  

**Abstract**: Multilingual language models (MLLMs) have demonstrated remarkable abilities to transfer knowledge across languages, despite being trained without explicit cross-lingual supervision. We analyze the parameter spaces of three MLLMs to study how their representations evolve during pre-training, observing patterns consistent with compression: models initially form language-specific representations, which gradually converge into cross-lingual abstractions as training progresses. Through probing experiments, we observe a clear transition from uniform language identification capabilities across layers to more specialized layer functions. For deeper analysis, we focus on neurons that encode distinct semantic concepts. By tracing their development during pre-training, we show how they gradually align across languages. Notably, we identify specific neurons that emerge as increasingly reliable predictors for the same concepts across languages. 

---
# MVAN: Multi-View Attention Networks for Fake News Detection on Social Media 

**Authors**: Shiwen Ni, Jiawen Li, Hung-Yu Kao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01627)  

**Abstract**: Fake news on social media is a widespread and serious problem in today's society. Existing fake news detection methods focus on finding clues from Long text content, such as original news articles and user comments. This paper solves the problem of fake news detection in more realistic scenarios. Only source shot-text tweet and its retweet users are provided without user comments. We develop a novel neural network based model, \textbf{M}ulti-\textbf{V}iew \textbf{A}ttention \textbf{N}etworks (MVAN) to detect fake news and provide explanations on social media. The MVAN model includes text semantic attention and propagation structure attention, which ensures that our model can capture information and clues both of source tweet content and propagation structure. In addition, the two attention mechanisms in the model can find key clue words in fake news texts and suspicious users in the propagation structure. We conduct experiments on two real-world datasets, and the results demonstrate that MVAN can significantly outperform state-of-the-art methods by 2.5\% in accuracy on average, and produce a reasonable explanation. 

---
# Domain Lexical Knowledge-based Word Embedding Learning for Text Classification under Small Data 

**Authors**: Zixiao Zhu, Kezhi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01621)  

**Abstract**: Pre-trained language models such as BERT have been proved to be powerful in many natural language processing tasks. But in some text classification applications such as emotion recognition and sentiment analysis, BERT may not lead to satisfactory performance. This often happens in applications where keywords play critical roles in the prediction of class labels. Our investigation found that the root cause of the problem is that the context-based BERT embedding of the keywords may not be discriminative enough to produce discriminative text representation for classification. Motivated by this finding, we develop a method to enhance word embeddings using domain-specific lexical knowledge. The knowledge-based embedding enhancement model projects the BERT embedding into a new space where within-class similarity and between-class difference are maximized. To implement the knowledge-based word embedding enhancement model, we also develop a knowledge acquisition algorithm for automatically collecting lexical knowledge from online open sources. Experiment results on three classification tasks, including sentiment analysis, emotion recognition and question answering, have shown the effectiveness of our proposed word embedding enhancing model. The codes and datasets are in this https URL. 

---
# IndicRAGSuite: Large-Scale Datasets and a Benchmark for Indian Language RAG Systems 

**Authors**: Pasunuti Prasanjith, Prathmesh B More, Anoop Kunchukuttan, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2506.01615)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enable language models to access relevant information and generate accurate, well-grounded, and contextually informed responses. However, for Indian languages, the development of high-quality RAG systems is hindered by the lack of two critical resources: (1) evaluation benchmarks for retrieval and generation tasks, and (2) large-scale training datasets for multilingual retrieval. Most existing benchmarks and datasets are centered around English or high-resource languages, making it difficult to extend RAG capabilities to the diverse linguistic landscape of India. To address the lack of evaluation benchmarks, we create IndicMSMarco, a multilingual benchmark for evaluating retrieval quality and response generation in 13 Indian languages, created via manual translation of 1000 diverse queries from MS MARCO-dev set. To address the need for training data, we build a large-scale dataset of (question, answer, relevant passage) tuples derived from the Wikipedias of 19 Indian languages using state-of-the-art LLMs. Additionally, we include translated versions of the original MS MARCO dataset to further enrich the training data and ensure alignment with real-world information-seeking tasks. Resources are available here: this https URL 

---
# MMD-Sense-Analysis: Word Sense Detection Leveraging Maximum Mean Discrepancy 

**Authors**: Kensuke Mitsuzawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.01602)  

**Abstract**: Word sense analysis is an essential analysis work for interpreting the linguistic and social backgrounds. The word sense change detection is a task of identifying and interpreting shifts in word meanings over time. This paper proposes MMD-Sense-Analysis, a novel approach that leverages Maximum Mean Discrepancy (MMD) to select semantically meaningful variables and quantify changes across time periods. This method enables both the identification of words undergoing sense shifts and the explanation of their evolution over multiple historical periods. To my knowledge, this is the first application of MMD to word sense change detection. Empirical assessment results demonstrate the effectiveness of the proposed approach. 

---
# Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models 

**Authors**: Ahmed Elshabrawy, Thanh-Nhi Nguyen, Yeeun Kang, Lihan Feng, Annant Jain, Faadil Abdullah Shaikh, Jonibek Mansurov, Mohamed Fazli Mohamed Imam, Jesus-German Ortiz-Barajas, Rendi Chevi, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2506.01592)  

**Abstract**: Large Language Models (LLMs) excel in zero-shot and few-shot tasks, but achieving similar performance with encoder-only models like BERT and RoBERTa has been challenging due to their architecture. However, encoders offer advantages such as lower computational and memory costs. Recent work adapts them for zero-shot generalization using Statement Tuning, which reformulates tasks into finite templates. We extend this approach to multilingual NLP, exploring whether encoders can achieve zero-shot cross-lingual generalization and serve as efficient alternatives to memory-intensive LLMs for low-resource languages. Our results show that state-of-the-art encoder models generalize well across languages, rivaling multilingual LLMs while being more efficient. We also analyze multilingual Statement Tuning dataset design, efficiency gains, and language-specific generalization, contributing to more inclusive and resource-efficient NLP models. We release our code and models. 

---
# Unified Large Language Models for Misinformation Detection in Low-Resource Linguistic Settings 

**Authors**: Muhammad Islam, Javed Ali Khan, Mohammed Abaker, Ali Daud, Azeem Irshad  

**Link**: [PDF](https://arxiv.org/pdf/2506.01587)  

**Abstract**: The rapid expansion of social media platforms has significantly increased the dissemination of forged content and misinformation, making the detection of fake news a critical area of research. Although fact-checking efforts predominantly focus on English-language news, there is a noticeable gap in resources and strategies to detect news in regional languages, such as Urdu. Advanced Fake News Detection (FND) techniques rely heavily on large, accurately labeled datasets. However, FND in under-resourced languages like Urdu faces substantial challenges due to the scarcity of extensive corpora and the lack of validated lexical resources. Current Urdu fake news datasets are often domain-specific and inaccessible to the public. They also lack human verification, relying mainly on unverified English-to-Urdu translations, which compromises their reliability in practical applications. This study highlights the necessity of developing reliable, expert-verified, and domain-independent Urdu-enhanced FND datasets to improve fake news detection in Urdu and other resource-constrained languages. This paper presents the first benchmark large FND dataset for Urdu news, which is publicly available for validation and deep analysis. We also evaluate this dataset using multiple state-of-the-art pre-trained large language models (LLMs), such as XLNet, mBERT, XLM-RoBERTa, RoBERTa, DistilBERT, and DeBERTa. Additionally, we propose a unified LLM model that outperforms the others with different embedding and feature extraction techniques. The performance of these models is compared based on accuracy, F1 score, precision, recall, and human judgment for vetting the sample results of news. 

---
# Prompt Engineering Large Language Models' Forecasting Capabilities 

**Authors**: Philipp Schoenegger, Cameron R. Jones, Philip E. Tetlock, Barbara Mellers  

**Link**: [PDF](https://arxiv.org/pdf/2506.01578)  

**Abstract**: Large language model performance can be improved in a large number of ways. Many such techniques, like fine-tuning or advanced tool usage, are time-intensive and expensive. Although prompt engineering is significantly cheaper and often works for simpler tasks, it remains unclear whether prompt engineering suffices for more complex domains like forecasting. Here we show that small prompt modifications rarely boost forecasting accuracy beyond a minimal baseline. In our first study, we tested 38 prompts across Claude 3.5 Sonnet, Claude 3.5 Haiku, GPT-4o, and Llama 3.1 405B. In our second, we introduced compound prompts and prompts from external sources, also including the reasoning models o1 and o1-mini. Our results show that most prompts lead to negligible gains, although references to base rates yield slight benefits. Surprisingly, some strategies showed strong negative effects on accuracy: especially encouraging the model to engage in Bayesian reasoning. These results suggest that, in the context of complex tasks like forecasting, basic prompt refinements alone offer limited gains, implying that more robust or specialized techniques may be required for substantial performance improvements in AI forecasting. 

---
# Hanfu-Bench: A Multimodal Benchmark on Cross-Temporal Cultural Understanding and Transcreation 

**Authors**: Li Zhou, Lutong Yu, Dongchu Xie, Shaohuan Cheng, Wenyan Li, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01565)  

**Abstract**: Culture is a rich and dynamic domain that evolves across both geography and time. However, existing studies on cultural understanding with vision-language models (VLMs) primarily emphasize geographic diversity, often overlooking the critical temporal dimensions. To bridge this gap, we introduce Hanfu-Bench, a novel, expert-curated multimodal dataset. Hanfu, a traditional garment spanning ancient Chinese dynasties, serves as a representative cultural heritage that reflects the profound temporal aspects of Chinese culture while remaining highly popular in Chinese contemporary society. Hanfu-Bench comprises two core tasks: cultural visual understanding and cultural image this http URL former task examines temporal-cultural feature recognition based on single- or multi-image inputs through multiple-choice visual question answering, while the latter focuses on transforming traditional attire into modern designs through cultural element inheritance and modern context adaptation. Our evaluation shows that closed VLMs perform comparably to non-experts on visual cutural understanding but fall short by 10\% to human experts, while open VLMs lags further behind non-experts. For the transcreation task, multi-faceted human evaluation indicates that the best-performing model achieves a success rate of only 42\%. Our benchmark provides an essential testbed, revealing significant challenges in this new direction of temporal cultural understanding and creative adaptation. 

---
# Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries 

**Authors**: Haruki Sakajo, Yusuke Ide, Justin Vasselli, Yusuke Sakai, Yingtao Tian, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.01535)  

**Abstract**: Cross-lingual vocabulary transfer plays a promising role in adapting pre-trained language models to new languages, including low-resource languages. Existing approaches that utilize monolingual or parallel corpora face challenges when applied to languages with limited resources. In this work, we propose a simple yet effective vocabulary transfer method that utilizes bilingual dictionaries, which are available for many languages, thanks to descriptive linguists. Our proposed method leverages a property of BPE tokenizers where removing a subword from the vocabulary causes a fallback to shorter subwords. The embeddings of target subwords are estimated iteratively by progressively removing them from the tokenizer. The experimental results show that our approach outperforms existing methods for low-resource languages, demonstrating the effectiveness of a dictionary-based approach for cross-lingual vocabulary transfer. 

---
# STORM-BORN: A Challenging Mathematical Derivations Dataset Curated via a Human-in-the-Loop Multi-Agent Framework 

**Authors**: Wenhao Liu, Zhenyi Lu, Xinyu Hu, Jierui Zhang, Dailin Li, Jiacheng Cen, Huilin Cao, Haiteng Wang, Yuhan Li, Kun Xie, Dandan Li, Pei Zhang, Chengbo Zhang, Yuxiang Ren, Xiaohong Huang, Yan Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.01531)  

**Abstract**: High-quality math datasets are crucial for advancing the reasoning abilities of large language models (LLMs). However, existing datasets often suffer from three key issues: outdated and insufficient challenging content, neglecting human-like reasoning, and limited reliability due to single-LLM generation. To address these, we introduce $\textbf{STORM-BORN}$, an ultra-challenging dataset of mathematical derivations sourced from cutting-edge academic papers, which includes dense human-like approximations and heuristic cues. To ensure the reliability and quality, we propose a novel human-in-the-loop, multi-agent data generation framework, integrating reasoning-dense filters, multi-agent collaboration, and human mathematicians' evaluations. We curated a set of 2,000 synthetic samples and deliberately selected the 100 most difficult problems. Even most advanced models like GPT-o1 solved fewer than $5\%$ of them. Fine-tuning on STORM-BORN boosts accuracy by $7.84\%$ (LLaMA3-8B) and $9.12\%$ (Qwen2.5-7B). As AI approaches mathematician-level reasoning, STORM-BORN provides both a high-difficulty benchmark and a human-like reasoning training resource. Our code and dataset are publicly available at this https URL. 

---
# V-VAE: A Variational Auto Encoding Framework Towards Fine-Grained Control over Human-Like Chat 

**Authors**: Qi Lin, Weikai Xu, Lisi Chen, Bin Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.01524)  

**Abstract**: With the continued proliferation of Large Language Model (LLM) based chatbots, there is a growing demand for generating responses that are not only linguistically fluent but also consistently aligned with persona-specific traits in conversations. However, existing role-play and persona-based chat approaches rely heavily on static role descriptions, coarse-grained signal space, and low-quality synthetic data, which fail to capture dynamic fine-grained details in human-like chat. Human-like chat requires modeling subtle latent traits, such as emotional tone, situational awareness, and evolving personality, which are difficult to predefine and cannot be easily learned from synthetic or distillation-based data. To address these limitations, we propose a Verbal Variational Auto-Encoding (V-VAE) framework, containing a variational auto-encoding module and fine-grained control space which dynamically adapts dialogue behaviour based on fine-grained, interpretable latent variables across talking style, interaction patterns, and personal attributes. We also construct a high-quality dataset, HumanChatData, and benchmark HumanChatBench to address the scarcity of high-quality data in the human-like domain. Experiments show that LLMs based on V-VAE consistently outperform standard baselines on HumanChatBench and DialogBench, which further demonstrates the effectiveness of V-VAE and HumanChatData. 

---
# FormFactory: An Interactive Benchmarking Suite for Multimodal Form-Filling Agents 

**Authors**: Bobo Li, Yuheng Wang, Hao Fei, Juncheng Li, Wei Ji, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01520)  

**Abstract**: Online form filling is a common yet labor-intensive task involving extensive keyboard and mouse interactions. Despite the long-standing vision of automating this process with "one click", existing tools remain largely rule-based and lack generalizable, generative capabilities. Recent advances in Multimodal Large Language Models (MLLMs) have enabled promising agents for GUI-related tasks in general-purpose scenarios. However, they struggle with the unique challenges of form filling, such as flexible layouts and the difficulty of aligning textual instructions with on-screen fields. To bridge this gap, we formally define the form-filling task and propose FormFactory, an interactive benchmarking suite comprising a web-based interface, backend evaluation module, and carefully constructed dataset. Our benchmark covers diverse real-world scenarios, incorporates various field formats, and simulates high-fidelity form interactions. We conduct a comprehensive evaluation of state-of-the-art MLLMs and observe that no model surpasses 5% accuracy, underscoring the inherent difficulty of the task. These findings also reveal significant limitations in current models' visual layout reasoning and field-value alignment abilities. We hope our benchmark can serve as a stepping stone for further research into robust, practical form-filling agents. 

---
# Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes 

**Authors**: Meng Li, Michael Vrazitulis, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01512)  

**Abstract**: Rational speakers are supposed to know what they know and what they do not know, and to generate expressions matching the strength of evidence. In contrast, it is still a challenge for current large language models to generate corresponding utterances based on the assessment of facts and confidence in an uncertain real-world environment. While it has recently become popular to estimate and calibrate confidence of LLMs with verbalized uncertainty, what is lacking is a careful examination of the linguistic knowledge of uncertainty encoded in the latent space of LLMs. In this paper, we draw on typological frameworks of epistemic expressions to evaluate LLMs' knowledge of epistemic modality, using controlled stories. Our experiments show that the performance of LLMs in generating epistemic expressions is limited and not robust, and hence the expressions of uncertainty generated by LLMs are not always reliable. To build uncertainty-aware LLMs, it is necessary to enrich semantic knowledge of epistemic modality in LLMs. 

---
# Continual Speech Learning with Fused Speech Features 

**Authors**: Guitao Wang, Jinming Zhao, Hao Yang, Guilin Qi, Tongtong Wu, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2506.01496)  

**Abstract**: Rapid growth in speech data demands adaptive models, as traditional static methods fail to keep pace with dynamic and diverse speech information. We introduce continuous speech learning, a new set-up targeting at bridging the adaptation gap in current speech models. We use the encoder-decoder Whisper model to standardize speech tasks into a generative format. We integrate a learnable gated-fusion layer on the top of the encoder to dynamically select task-specific features for downstream tasks. Our approach improves accuracy significantly over traditional methods in six speech processing tasks, demonstrating gains in adapting to new speech tasks without full retraining. 

---
# CVC: A Large-Scale Chinese Value Rule Corpus for Value Alignment of Large Language Models 

**Authors**: Ping Wu, Guobin Shen, Dongcheng Zhao, Yuwei Wang, Yiting Dong, Yu Shi, Enmeng Lu, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01495)  

**Abstract**: Ensuring that Large Language Models (LLMs) align with mainstream human values and ethical norms is crucial for the safe and sustainable development of AI. Current value evaluation and alignment are constrained by Western cultural bias and incomplete domestic frameworks reliant on non-native rules; furthermore, the lack of scalable, rule-driven scenario generation methods makes evaluations costly and inadequate across diverse cultural contexts. To address these challenges, we propose a hierarchical value framework grounded in core Chinese values, encompassing three main dimensions, 12 core values, and 50 derived values. Based on this framework, we construct a large-scale Chinese Values Corpus (CVC) containing over 250,000 value rules enhanced and expanded through human annotation. Experimental results show that CVC-guided scenarios outperform direct generation ones in value boundaries and content diversity. In the evaluation across six sensitive themes (e.g., surrogacy, suicide), seven mainstream LLMs preferred CVC-generated options in over 70.5% of cases, while five Chinese human annotators showed an 87.5% alignment with CVC, confirming its universality, cultural relevance, and strong alignment with Chinese values. Additionally, we construct 400,000 rule-based moral dilemma scenarios that objectively capture nuanced distinctions in conflicting value prioritization across 17 LLMs. Our work establishes a culturally-adaptive benchmarking framework for comprehensive value evaluation and alignment, representing Chinese characteristics. All data are available at this https URL, and the code is available at this https URL. 

---
# Multilingual Definition Modeling 

**Authors**: Edison Marrese-Taylor, Erica K. Shimomoto, Alfredo Solano, Enrique Reid  

**Link**: [PDF](https://arxiv.org/pdf/2506.01489)  

**Abstract**: In this paper, we propose the first multilingual study on definition modeling. We use monolingual dictionary data for four new languages (Spanish, French, Portuguese, and German) and perform an in-depth empirical study to test the performance of pre-trained multilingual language models on definition modeling of monosemic words when finetuned on this data. Furthermore, we use a zero-shot approach to test the multilingual capabilities of two popular chat-based Large Language Models (LLMs) in the task. Results show that multilingual language models can perform on-pair with English but cannot leverage potential cross-lingual synergies, with LLMs generally offering better performance overall. A comprehensive human evaluation of the LLM-generated definition highlights the zero and few-shot capabilities of these models in this new task, also showing their shortcomings. Finally, we show that performance on our task via BERTScore strongly correlates to the performance on multilingual LLM benchmarks, suggesting that our task offers a viable compute-constrained, stable and natural alternative to these. 

---
# Argument-Centric Causal Intervention Method for Mitigating Bias in Cross-Document Event Coreference Resolution 

**Authors**: Long Yao, Wenzhong Yang, Yabo Yin, Fuyuan Wei, Hongzhen Lv, Jiaren Peng, Liejun Wang, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01488)  

**Abstract**: Cross-document Event Coreference Resolution (CD-ECR) is a fundamental task in natural language processing (NLP) that seeks to determine whether event mentions across multiple documents refer to the same real-world occurrence. However, current CD-ECR approaches predominantly rely on trigger features within input mention pairs, which induce spurious correlations between surface-level lexical features and coreference relationships, impairing the overall performance of the models. To address this issue, we propose a novel cross-document event coreference resolution method based on Argument-Centric Causal Intervention (ACCI). Specifically, we construct a structural causal graph to uncover confounding dependencies between lexical triggers and coreference labels, and introduce backdoor-adjusted interventions to isolate the true causal effect of argument semantics. To further mitigate spurious correlations, ACCI integrates a counterfactual reasoning module that quantifies the causal influence of trigger word perturbations, and an argument-aware enhancement module to promote greater sensitivity to semantically grounded information. In contrast to prior methods that depend on costly data augmentation or heuristic-based filtering, ACCI enables effective debiasing in a unified end-to-end framework without altering the underlying training procedure. Extensive experiments demonstrate that ACCI achieves CoNLL F1 of 88.4% on ECB+ and 85.2% on GVC, achieving state-of-the-art performance. The implementation and materials are available at this https URL. 

---
# LLM in the Loop: Creating the PARADEHATE Dataset for Hate Speech Detoxification 

**Authors**: Shuzhou Yuan, Ercong Nie, Lukas Kouba, Ashish Yashwanth Kangen, Helmut Schmid, Hinrich Schutze, Michael Farber  

**Link**: [PDF](https://arxiv.org/pdf/2506.01484)  

**Abstract**: Detoxification, the task of rewriting harmful language into non-toxic text, has become increasingly important amid the growing prevalence of toxic content online. However, high-quality parallel datasets for detoxification, especially for hate speech, remain scarce due to the cost and sensitivity of human annotation. In this paper, we propose a novel LLM-in-the-loop pipeline leveraging GPT-4o-mini for automated detoxification. We first replicate the ParaDetox pipeline by replacing human annotators with an LLM and show that the LLM performs comparably to human annotation. Building on this, we construct PARADEHATE, a large-scale parallel dataset specifically for hatespeech detoxification. We release PARADEHATE as a benchmark of over 8K hate/non-hate text pairs and evaluate a wide range of baseline methods. Experimental results show that models such as BART, fine-tuned on PARADEHATE, achieve better performance in style accuracy, content preservation, and fluency, demonstrating the effectiveness of LLM-generated detoxification text as a scalable alternative to human annotation. 

---
# Integrating Neural and Symbolic Components in a Model of Pragmatic Question-Answering 

**Authors**: Polina Tsvilodub, Robert D. Hawkins, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2506.01474)  

**Abstract**: Computational models of pragmatic language use have traditionally relied on hand-specified sets of utterances and meanings, limiting their applicability to real-world language use. We propose a neuro-symbolic framework that enhances probabilistic cognitive models by integrating LLM-based modules to propose and evaluate key components in natural language, eliminating the need for manual specification. Through a classic case study of pragmatic question-answering, we systematically examine various approaches to incorporating neural modules into the cognitive model -- from evaluating utilities and literal semantics to generating alternative utterances and goals. We find that hybrid models can match or exceed the performance of traditional probabilistic models in predicting human answer patterns. However, the success of the neuro-symbolic model depends critically on how LLMs are integrated: while they are particularly effective for proposing alternatives and transforming abstract goals into utilities, they face challenges with truth-conditional semantic evaluation. This work charts a path toward more flexible and scalable models of pragmatic language use while illuminating crucial design considerations for balancing neural and symbolic components. 

---
# TalTech Systems for the Interspeech 2025 ML-SUPERB 2.0 Challenge 

**Authors**: Tanel Alumäe, Artem Fedorchenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.01458)  

**Abstract**: This paper describes the language identification and multilingual speech recognition system developed at Tallinn University of Technology for the Interspeech 2025 ML-SUPERB 2.0 Challenge. A hybrid language identification system is used, consisting of a pretrained language embedding model and a light-weight speech recognition model with a shared encoder across languages and language-specific bigram language models. For speech recognition, three models are used, where only a single model is applied for each language, depending on the training data availability and performance on held-out data. The model set consists of a finetuned version of SeamlessM4T, MMS-1B-all with custom language adapters and MMS-zeroshot. The system obtained the top overall score in the challenge. 

---
# Building Entity Association Mining Framework for Knowledge Discovery 

**Authors**: Anshika Rawal, Abhijeet Kumar, Mridul Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2506.01451)  

**Abstract**: Extracting useful signals or pattern to support important business decisions for example analyzing investment product traction and discovering customer preference, risk monitoring etc. from unstructured text is a challenging task. Capturing interaction of entities or concepts and association mining is a crucial component in text mining, enabling information extraction and reasoning over and knowledge discovery from text. Furthermore, it can be used to enrich or filter knowledge graphs to guide exploration processes, descriptive analytics and uncover hidden stories in the text. In this paper, we introduce a domain independent pipeline i.e., generalized framework to enable document filtering, entity extraction using various sources (or techniques) as plug-ins and association mining to build any text mining business use-case and quantitatively define a scoring metric for ranking purpose. The proposed framework has three major components a) Document filtering: filtering documents/text of interest from massive amount of texts b) Configurable entity extraction pipeline: include entity extraction techniques i.e., i) DBpedia Spotlight, ii) Spacy NER, iii) Custom Entity Matcher, iv) Phrase extraction (or dictionary) based c) Association Relationship Mining: To generates co-occurrence graph to analyse potential relationships among entities, concepts. Further, co-occurrence count based frequency statistics provide a holistic window to observe association trends or buzz rate in specific business context. The paper demonstrates the usage of framework as fundamental building box in two financial use-cases namely brand product discovery and vendor risk monitoring. We aim that such framework will remove duplicated effort, minimize the development effort, and encourage reusability and rapid prototyping in association mining business applications for institutions. 

---
# Whale: Large-Scale multilingual ASR model with w2v-BERT and E-Branchformer with large speech data 

**Authors**: Yosuke Kashiwagi, Hayato Futami, Emiru Tsunoo, Satoshi Asakawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.01439)  

**Abstract**: This paper reports on the development of a large-scale speech recognition model, Whale. Similar to models such as Whisper and OWSM, Whale leverages both a large model size and a diverse, extensive dataset. Whale's architecture integrates w2v-BERT self-supervised model, an encoder-decoder backbone built on E-Branchformer, and a joint CTC-attention decoding strategy. The training corpus comprises varied speech data, of not only public corpora but also in-house data, thereby enhancing the model's robustness to different speaking styles and acoustic conditions. Through evaluations on multiple benchmarks, Whale achieved comparable performance to existing models. In particular, it achieves a word error rate of 2.4% on the Librispeech test-clean set and a character error rate of 3.4% on the CSJ eval3 set, outperforming Whisper large-v3 and OWSM v3.1. 

---
# Redundancy, Isotropy, and Intrinsic Dimensionality of Prompt-based Text Embeddings 

**Authors**: Hayato Tsukagoshi, Ryohei Sasano  

**Link**: [PDF](https://arxiv.org/pdf/2506.01435)  

**Abstract**: Prompt-based text embedding models, which generate task-specific embeddings upon receiving tailored prompts, have recently demonstrated remarkable performance. However, their resulting embeddings often have thousands of dimensions, leading to high storage costs and increased computational costs of embedding-based operations. In this paper, we investigate how post-hoc dimensionality reduction applied to the embeddings affects the performance of various tasks that leverage these embeddings, specifically classification, clustering, retrieval, and semantic textual similarity (STS) tasks. Our experiments show that even a naive dimensionality reduction, which keeps only the first 25% of the dimensions of the embeddings, results in a very slight performance degradation, indicating that these embeddings are highly redundant. Notably, for classification and clustering, even when embeddings are reduced to less than 0.5% of the original dimensionality the performance degradation is very small. To quantitatively analyze this redundancy, we perform an analysis based on the intrinsic dimensionality and isotropy of the embeddings. Our analysis reveals that embeddings for classification and clustering, which are considered to have very high dimensional redundancy, exhibit lower intrinsic dimensionality and less isotropy compared with those for retrieval and STS. 

---
# Self-Refining Language Model Anonymizers via Adversarial Distillation 

**Authors**: Kyuyoung Kim, Hyunjun Jeon, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.01420)  

**Abstract**: Large language models (LLMs) are increasingly used in sensitive domains, where their ability to infer personal data from seemingly benign text poses emerging privacy risks. While recent LLM-based anonymization methods help mitigate such risks, they often rely on proprietary models (e.g., GPT-4), raising concerns about cost and the potential exposure of sensitive data to untrusted external systems. To address this, we introduce SElf-refining Anonymization with Language model (SEAL), a novel distillation framework for training small language models (SLMs) to perform effective anonymization without relying on external costly models at inference time. We leverage adversarial interactions between an LLM anonymizer and an inference model to collect trajectories of anonymized texts and inferred attributes, which are used to distill anonymization, adversarial inference, and utility evaluation capabilities into SLMs via supervised fine-tuning and preference learning. The resulting models learn to both anonymize text and critique their outputs, enabling iterative improvement of anonymization quality via self-refinement. Experiments on SynthPAI, a dataset of synthetic personal profiles and text comments, demonstrate that SLMs trained with SEAL achieve substantial improvements in anonymization capabilities. Notably, 8B models attain a privacy-utility trade-off comparable to that of the GPT-4 anonymizer and, with self-refinement, even surpass it in terms of privacy. These results show the effectiveness of our adversarial distillation framework in training SLMs as efficient anonymizers. To facilitate further research, we release the full dataset used in our experiments. 

---
# UniversalCEFR: Enabling Open Multilingual Research on Language Proficiency Assessment 

**Authors**: Joseph Marvin Imperial, Abdullah Barayan, Regina Stodden, Rodrigo Wilkens, Ricardo Munoz Sanchez, Lingyun Gao, Melissa Torgbi, Dawn Knight, Gail Forey, Reka R. Jablonkai, Ekaterina Kochmar, Robert Reynolds, Eugenio Ribeiro, Horacio Saggion, Elena Volodina, Sowmya Vajjala, Thomas Francois, Fernando Alva-Manchego, Harish Tayyar Madabushi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01419)  

**Abstract**: We introduce UniversalCEFR, a large-scale multilingual multidimensional dataset of texts annotated according to the CEFR (Common European Framework of Reference) scale in 13 languages. To enable open research in both automated readability and language proficiency assessment, UniversalCEFR comprises 505,807 CEFR-labeled texts curated from educational and learner-oriented resources, standardized into a unified data format to support consistent processing, analysis, and modeling across tasks and languages. To demonstrate its utility, we conduct benchmark experiments using three modelling paradigms: a) linguistic feature-based classification, b) fine-tuning pre-trained LLMs, and c) descriptor-based prompting of instruction-tuned LLMs. Our results further support using linguistic features and fine-tuning pretrained models in multilingual CEFR level assessment. Overall, UniversalCEFR aims to establish best practices in data distribution in language proficiency research by standardising dataset formats and promoting their accessibility to the global research community. 

---
# Comparing LLM-generated and human-authored news text using formal syntactic theory 

**Authors**: Olga Zamaraeva, Dan Flickinger, Francis Bond, Carlos Gómez-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2506.01407)  

**Abstract**: This study provides the first comprehensive comparison of New York Times-style text generated by six large language models against real, human-authored NYT writing. The comparison is based on a formal syntactic theory. We use Head-driven Phrase Structure Grammar (HPSG) to analyze the grammatical structure of the texts. We then investigate and illustrate the differences in the distributions of HPSG grammar types, revealing systematic distinctions between human and LLM-generated writing. These findings contribute to a deeper understanding of the syntactic behavior of LLMs as well as humans, within the NYT genre. 

---
# Speech-to-Speech Translation Pipelines for Conversations in Low-Resource Languages 

**Authors**: Andrei Popescu-Belis, Alexis Allemann, Teo Ferrari, Gopal Krishnamani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01406)  

**Abstract**: The popularity of automatic speech-to-speech translation for human conversations is growing, but the quality varies significantly depending on the language pair. In a context of community interpreting for low-resource languages, namely Turkish and Pashto to/from French, we collected fine-tuning and testing data, and compared systems using several automatic metrics (BLEU, COMET, and BLASER) and human assessments. The pipelines included automatic speech recognition, machine translation, and speech synthesis, with local models and cloud-based commercial ones. Some components have been fine-tuned on our data. We evaluated over 60 pipelines and determined the best one for each direction. We also found that the ranks of components are generally independent of the rest of the pipeline. 

---
# AdaRewriter: Unleashing the Power of Prompting-based Conversational Query Reformulation via Test-Time Adaptation 

**Authors**: Yilong Lai, Jialong Wu, Zhenglin Wang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01381)  

**Abstract**: Prompting-based conversational query reformulation has emerged as a powerful approach for conversational search, refining ambiguous user queries into standalone search queries. Best-of-N reformulation over the generated candidates via prompting shows impressive potential scaling capability. However, both the previous tuning methods (training time) and adaptation approaches (test time) can not fully unleash their benefits. In this paper, we propose AdaRewriter, a novel framework for query reformulation using an outcome-supervised reward model via test-time adaptation. By training a lightweight reward model with contrastive ranking loss, AdaRewriter selects the most promising reformulation during inference. Notably, it can operate effectively in black-box systems, including commercial LLM APIs. Experiments on five conversational search datasets show that AdaRewriter significantly outperforms the existing methods across most settings, demonstrating the potential of test-time adaptation for conversational query reformulation. 

---
# MMD-Flagger: Leveraging Maximum Mean Discrepancy to Detect Hallucinations 

**Authors**: Kensuke Mitsuzawa, Damien Garreau  

**Link**: [PDF](https://arxiv.org/pdf/2506.01367)  

**Abstract**: Large language models (LLMs) have become pervasive in our everyday life. Yet, a fundamental obstacle prevents their use in many critical applications: their propensity to generate fluent, human-quality content that is not grounded in reality. The detection of such hallucinations is thus of the highest importance. In this work, we propose a new method to flag hallucinated content, MMD-Flagger. It relies on Maximum Mean Discrepancy (MMD), a non-parametric distance between distributions. On a high-level perspective, MMD-Flagger tracks the MMD between the generated documents and documents generated with various temperature parameters. We show empirically that inspecting the shape of this trajectory is sufficient to detect most hallucinations. This novel method is benchmarked on two machine translation datasets, on which it outperforms natural competitors. 

---
# KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors 

**Authors**: Zhiyang Qi, Takumasa Kaneko, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.01357)  

**Abstract**: Generating psychological counseling responses with language models relies heavily on high-quality datasets. Crowdsourced data collection methods require strict worker training, and data from real-world counseling environments may raise privacy and ethical concerns. While recent studies have explored using large language models (LLMs) to augment psychological counseling dialogue datasets, the resulting data often suffers from limited diversity and authenticity. To address these limitations, this study adopts a role-playing approach where trained counselors simulate counselor-client interactions, ensuring high-quality dialogues while mitigating privacy risks. Using this method, we construct KokoroChat, a Japanese psychological counseling dialogue dataset comprising 6,589 long-form dialogues, each accompanied by comprehensive client feedback. Experimental results demonstrate that fine-tuning open-source LLMs with KokoroChat improves both the quality of generated counseling responses and the automatic evaluation of counseling dialogues. The KokoroChat dataset is available at this https URL. 

---
# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning 

**Authors**: Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, Yu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.01347)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for training language models (LMs) on reasoning tasks that elicit emergent long chains of thought (CoTs). Unlike supervised learning, it updates the model using both correct and incorrect samples via policy gradients. To better understand its mechanism, we decompose the learning signal into reinforcing correct responses and penalizing incorrect ones, referred to as Positive and Negative Sample Reinforcement (PSR and NSR), respectively. We train Qwen2.5-Math-7B and Qwen3-4B on a mathematical reasoning dataset and uncover a surprising result: training with only negative samples -- without reinforcing correct responses -- can be highly effective: it consistently improves performance over the base model across the entire Pass@$k$ spectrum ($k$ up to $256$), often matching or surpassing PPO and GRPO. In contrast, reinforcing only correct responses improves Pass@$1$ but degrades performance at higher $k$, due to reduced diversity. These inference-scaling trends highlight that solely penalizing incorrect responses may contribute more to performance than previously recognized. Through gradient analysis, we show that NSR works by suppressing incorrect generations and redistributing probability mass toward other plausible candidates, guided by the model's prior beliefs. It refines the model's existing knowledge rather than introducing entirely new behaviors. Building on this insight, we propose a simple variant of the RL objective that upweights NSR, and show that it consistently improves overall Pass@$k$ performance on MATH, AIME 2025, and AMC23. Our code is available at this https URL. 

---
# Follow the Flow: Fine-grained Flowchart Attribution with Neurosymbolic Agents 

**Authors**: Manan Suri, Puneet Mathur, Nedim Lipka, Franck Dernoncourt, Ryan A. Rossi, Vivek Gupta, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2506.01344)  

**Abstract**: Flowcharts are a critical tool for visualizing decision-making processes. However, their non-linear structure and complex visual-textual relationships make it challenging to interpret them using LLMs, as vision-language models frequently hallucinate nonexistent connections and decision paths when analyzing these diagrams. This leads to compromised reliability for automated flowchart processing in critical domains such as logistics, health, and engineering. We introduce the task of Fine-grained Flowchart Attribution, which traces specific components grounding a flowchart referring LLM response. Flowchart Attribution ensures the verifiability of LLM predictions and improves explainability by linking generated responses to the flowchart's structure. We propose FlowPathAgent, a neurosymbolic agent that performs fine-grained post hoc attribution through graph-based reasoning. It first segments the flowchart, then converts it into a structured symbolic graph, and then employs an agentic approach to dynamically interact with the graph, to generate attribution paths. Additionally, we present FlowExplainBench, a novel benchmark for evaluating flowchart attributions across diverse styles, domains, and question types. Experimental results show that FlowPathAgent mitigates visual hallucinations in LLM answers over flowchart QA, outperforming strong baselines by 10-14% on our proposed FlowExplainBench dataset. 

---
# TurnBench-MS: A Benchmark for Evaluating Multi-Turn, Multi-Step Reasoning in Large Language Models 

**Authors**: Yiran Zhang, Mo Wang, Xiaoyang Li, Kaixuan Ren, Chencheng Zhu, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.01341)  

**Abstract**: Despite impressive advances in large language models (LLMs), existing benchmarks often focus on single-turn or single-step tasks, failing to capture the kind of iterative reasoning required in real-world settings. To address this limitation, we introduce TurnBench, a novel benchmark that evaluates multi-turn, multi-step reasoning through an interactive code-breaking task inspired by a "Turing Machine Board Game." In each episode, a model must uncover hidden logical or arithmetic rules by making sequential guesses, receiving structured feedback, and integrating clues across multiple rounds. This dynamic setup requires models to reason over time, adapt based on past information, and maintain consistency across steps-capabilities underexplored in current benchmarks. TurnBench includes two modes: Classic, which tests standard reasoning, and Nightmare, which introduces increased complexity and requires robust inferential chains. To support fine-grained analysis, we provide ground-truth annotations for intermediate reasoning steps. Our evaluation of state-of-the-art LLMs reveals significant gaps: the best model achieves 81.5% accuracy in Classic mode, but performance drops to 17.8% in Nightmare mode. In contrast, human participants achieve 100% in both, underscoring the challenge TurnBench poses to current models. By incorporating feedback loops and hiding task rules, TurnBench reduces contamination risks and provides a rigorous testbed for diagnosing and advancing multi-step, multi-turn reasoning in LLMs. 

---
# The Landscape of Arabic Large Language Models (ALLMs): A New Era for Arabic Language Technology 

**Authors**: Shahad Al-Khalifa, Nadir Durrani, Hend Al-Khalifa, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2506.01340)  

**Abstract**: The emergence of ChatGPT marked a transformative milestone for Artificial Intelligence (AI), showcasing the remarkable potential of Large Language Models (LLMs) to generate human-like text. This wave of innovation has revolutionized how we interact with technology, seamlessly integrating LLMs into everyday tasks such as vacation planning, email drafting, and content creation. While English-speaking users have significantly benefited from these advancements, the Arabic world faces distinct challenges in developing Arabic-specific LLMs. Arabic, one of the languages spoken most widely around the world, serves more than 422 million native speakers in 27 countries and is deeply rooted in a rich linguistic and cultural heritage. Developing Arabic LLMs (ALLMs) presents an unparalleled opportunity to bridge technological gaps and empower communities. The journey of ALLMs has been both fascinating and complex, evolving from rudimentary text processing systems to sophisticated AI-driven models. This article explores the trajectory of ALLMs, from their inception to the present day, highlighting the efforts to evaluate these models through benchmarks and public leaderboards. We also discuss the challenges and opportunities that ALLMs present for the Arab world. 

---
# Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models 

**Authors**: Yiwen Jiang, Deval Mehta, Wei Feng, Zongyuan Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.01334)  

**Abstract**: Concept Bottleneck Models (CBMs) decompose image classification into a process governed by interpretable, human-readable concepts. Recent advances in CBMs have used Large Language Models (LLMs) to generate candidate concepts. However, a critical question remains: What is the optimal number of concepts to use? Current concept banks suffer from redundancy or insufficient coverage. To address this issue, we introduce a dynamic, agent-based approach that adjusts the concept bank in response to environmental feedback, optimizing the number of concepts for sufficiency yet concise coverage. Moreover, we propose Conditional Concept Bottleneck Models (CoCoBMs) to overcome the limitations in traditional CBMs' concept scoring mechanisms. It enhances the accuracy of assessing each concept's contribution to classification tasks and feature an editable matrix that allows LLMs to correct concept scores that conflict with their internal knowledge. Our evaluations across 6 datasets show that our method not only improves classification accuracy by 6% but also enhances interpretability assessments by 30%. 

---
# Evaluating Large Language Models in Crisis Detection: A Real-World Benchmark from Psychological Support Hotlines 

**Authors**: Guifeng Deng, Shuyin Rao, Tianyu Lin, Anlu Dai, Pan Wang, Junyi Xie, Haidong Song, Ke Zhao, Dongwu Xu, Zhengdong Cheng, Tao Li, Haiteng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01329)  

**Abstract**: Psychological support hotlines are critical for crisis intervention but face significant challenges due to rising demand. Large language models (LLMs) could support crisis assessments, yet their capabilities in emotionally sensitive contexts remain unclear. We introduce PsyCrisisBench, a benchmark of 540 annotated transcripts from the Hangzhou Psychological Assistance Hotline, assessing four tasks: mood status recognition, suicidal ideation detection, suicide plan identification, and risk assessment. We evaluated 64 LLMs across 15 families (e.g., GPT, Claude, Gemini, Llama, Qwen, DeepSeek) using zero-shot, few-shot, and fine-tuning paradigms. Performance was measured by F1-score, with statistical comparisons via Welch's t-tests. LLMs performed strongly on suicidal ideation detection (F1=0.880), suicide plan identification (F1=0.779), and risk assessment (F1=0.907), improved with few-shot and fine-tuning. Mood status recognition was more challenging (max F1=0.709), likely due to lost vocal cues and ambiguity. A fine-tuned 1.5B-parameter model (Qwen2.5-1.5B) surpassed larger models on mood and suicidal ideation. Open-source models like QwQ-32B performed comparably to closed-source on most tasks (p>0.3), though closed models retained an edge in mood detection (p=0.007). Performance scaled with size up to a point; quantization (AWQ) reduced GPU memory by 70% with minimal F1 degradation. LLMs show substantial promise in structured psychological crisis assessments, especially with fine-tuning. Mood recognition remains limited due to contextual complexity. The narrowing gap between open- and closed-source models, combined with efficient quantization, suggests feasible integration. PsyCrisisBench offers a robust evaluation framework to guide model development and ethical deployment in mental health. 

---
# Zero-Shot Text-to-Speech for Vietnamese 

**Authors**: Thi Vu, Linh The Nguyen, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01322)  

**Abstract**: This paper introduces PhoAudiobook, a newly curated dataset comprising 941 hours of high-quality audio for Vietnamese text-to-speech. Using PhoAudiobook, we conduct experiments on three leading zero-shot TTS models: VALL-E, VoiceCraft, and XTTS-V2. Our findings demonstrate that PhoAudiobook consistently enhances model performance across various metrics. Moreover, VALL-E and VoiceCraft exhibit superior performance in synthesizing short sentences, highlighting their robustness in handling diverse linguistic contexts. We publicly release PhoAudiobook to facilitate further research and development in Vietnamese text-to-speech. 

---
# Growing Through Experience: Scaling Episodic Grounding in Language Models 

**Authors**: Chunhui Zhang, Sirui, Wang, Zhongyu Ouyang, Xiangchi Yuan, Soroush Vosoughi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01312)  

**Abstract**: Language models (LMs) require robust episodic grounding-the capacity to learn from and apply past experiences-to excel at physical planning tasks. Current episodic grounding approaches struggle with scalability and integration, limiting their effectiveness, especially for medium-sized LMs (7B parameters). While larger LMs (70-405B parameters) possess superior hierarchical representations and extensive pre-trained knowledge, they encounter a fundamental scale paradox: despite their advanced abstraction capabilities, they lack efficient mechanisms to leverage experience streams. We propose a scalable weak-to-strong episodic learning framework that effectively transfers episodic behaviors from smaller to larger LMs. This framework integrates Monte Carlo tree search for structured experience collection with a novel distillation method, preserving the inherent LM capabilities while embedding episodic memory. Experiments demonstrate our method surpasses state-of-the-art proprietary LMs by 3.45% across diverse planning and question-answering tasks. Layer-wise probing further indicates significant improvements in task alignment, especially within deeper LM layers, highlighting stable generalization even for previously unseen scenarios with increased planning complexity-conditions where baseline methods degrade markedly. 

---
# A Platform for Investigating Public Health Content with Efficient Concern Classification 

**Authors**: Christopher Li, Rickard Stureborg, Bhuwan Dhingra, Jun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01308)  

**Abstract**: A recent rise in online content expressing concerns with public health initiatives has contributed to already stalled uptake of preemptive measures globally. Future public health efforts must attempt to understand such content, what concerns it may raise among readers, and how to effectively respond to it. To this end, we present ConcernScope, a platform that uses a teacher-student framework for knowledge transfer between large language models and light-weight classifiers to quickly and effectively identify the health concerns raised in a text corpus. The platform allows uploading massive files directly, automatically scraping specific URLs, and direct text editing. ConcernScope is built on top of a taxonomy of public health concerns. Intended for public health officials, we demonstrate several applications of this platform: guided data exploration to find useful examples of common concerns found in online community datasets, identification of trends in concerns through an example time series analysis of 186,000 samples, and finding trends in topic frequency before and after significant events. 

---
# VM14K: First Vietnamese Medical Benchmark 

**Authors**: Thong Nguyen, Duc Nguyen, Minh Dang, Thai Dao, Long Nguyen, Quan H. Nguyen, Dat Nguyen, Kien Tran, Minh Tran  

**Link**: [PDF](https://arxiv.org/pdf/2506.01305)  

**Abstract**: Medical benchmarks are indispensable for evaluating the capabilities of language models in healthcare for non-English-speaking communities,therefore help ensuring the quality of real-life applications. However, not every community has sufficient resources and standardized methods to effectively build and design such benchmark, and available non-English medical data is normally fragmented and difficult to verify. We developed an approach to tackle this problem and applied it to create the first Vietnamese medical question benchmark, featuring 14,000 multiple-choice questions across 34 medical specialties. Our benchmark was constructed using various verifiable sources, including carefully curated medical exams and clinical records, and eventually annotated by medical experts. The benchmark includes four difficulty levels, ranging from foundational biological knowledge commonly found in textbooks to typical clinical case studies that require advanced reasoning. This design enables assessment of both the breadth and depth of language models' medical understanding in the target language thanks to its extensive coverage and in-depth subject-specific expertise. We release the benchmark in three parts: a sample public set (4k questions), a full public set (10k questions), and a private set (2k questions) used for leaderboard evaluation. Each set contains all medical subfields and difficulty levels. Our approach is scalable to other languages, and we open-source our data construction pipeline to support the development of future multilingual benchmarks in the medical domain. 

---
# Schema as Parameterized Tools for Universal Information Extraction 

**Authors**: Sheng Liang, Yongyue Zhang, Yaxiong Wu, Ruiming Tang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01276)  

**Abstract**: Universal information extraction (UIE) primarily employs an extractive generation approach with large language models (LLMs), typically outputting structured information based on predefined schemas such as JSON or tables. UIE suffers from a lack of adaptability when selecting between predefined schemas and on-the-fly schema generation within the in-context learning paradigm, especially when there are numerous schemas to choose from. In this paper, we propose a unified adaptive text-to-structure generation framework, called Schema as Parameterized Tools (SPT), which reimagines the tool-calling capability of LLMs by treating predefined schemas as parameterized tools for tool selection and parameter filling. Specifically, our SPT method can be applied to unify closed, open, and on-demand IE tasks by adopting Schema Retrieval by fetching the relevant schemas from a predefined pool, Schema Filling by extracting information and filling slots as with tool parameters, or Schema Generation by synthesizing new schemas with uncovered cases. Experiments show that the SPT method can handle four distinct IE tasks adaptively, delivering robust schema retrieval and selection performance. SPT also achieves comparable extraction performance to LoRA baselines and current leading UIE systems with significantly fewer trainable parameters. 

---
# Detoxification of Large Language Models through Output-layer Fusion with a Calibration Model 

**Authors**: Yuanhe Tian, Mingjie Deng, Guoqing Jin, Yan Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.01266)  

**Abstract**: Existing approaches for Large language model (LLM) detoxification generally rely on training on large-scale non-toxic or human-annotated preference data, designing prompts to instruct the LLM to generate safe content, or modifying the model parameters to remove toxic information, which are computationally expensive, lack robustness, and often compromise LLMs' fluency and contextual understanding. In this paper, we propose a simple yet effective approach for LLM detoxification, which leverages a compact, pre-trained calibration model that guides the detoxification process of a target LLM via a lightweight intervention in its generation pipeline. By learning a detoxified embedding space from non-toxic data, the calibration model effectively steers the LLM away from generating harmful content. This approach only requires a one-time training of the calibration model that is able to be seamlessly applied to multiple LLMs without compromising fluency or contextual understanding. Experiment results on the benchmark dataset demonstrate that our approach reduces toxicity while maintaining reasonable content expression. 

---
# Beyond In-Context Learning: Aligning Long-form Generation of Large Language Models via Task-Inherent Attribute Guidelines 

**Authors**: Do Xuan Long, Duong Ngoc Yen, Do Xuan Trong, Luu Anh Tuan, Kenji Kawaguchi, Shafiq Joty, Min-Yen Kan, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01265)  

**Abstract**: In-context learning (ICL) is an important yet not fully understood ability of pre-trained large language models (LLMs). It can greatly enhance task performance using a few examples, termed demonstrations, without fine-tuning. Although effective in question answering, ICL often underperforms in long-form generation tasks such as summarization. Under appropriately realistic assumptions, we empirically and theoretically show that ICL demonstrations alone are insufficient to teach LLMs the task language and format distributions for generation. We argue for explicit exposure to the task distributions and hypothesize that defining them by prompting enhances model performance. To this end, we present LongGuide, which efficiently generates two parallel streams of guidelines capturing task language and format properties: (i) Metric Guidelines (MGs) that instruct models to optimize self-evaluated metrics; and (ii) Output Constraint Guidelines (OCGs) that constrain generation at both token and sentence levels. LongGuide automatically selects the best combination of guidelines, improving both strong open- and closed-source LLMs by over 5% in both zero- and few-shot settings. We show that LongGuide is generalizable, learnable by weak models to enhance strong ones, and integrates synergistically with automatic prompt optimizers. 

---
# WCTC-Biasing: Retraining-free Contextual Biasing ASR with Wildcard CTC-based Keyword Spotting and Inter-layer Biasing 

**Authors**: Yu Nakagome, Michael Hentschel  

**Link**: [PDF](https://arxiv.org/pdf/2506.01263)  

**Abstract**: Despite recent advances in end-to-end speech recognition methods, the output tends to be biased to the training data's vocabulary, resulting in inaccurate recognition of proper nouns and other unknown terms. To address this issue, we propose a method to improve recognition accuracy of such rare words in CTC-based models without additional training or text-to-speech systems. Specifically, keyword spotting is performed using acoustic features of intermediate layers during inference, and a bias is applied to the subsequent layers of the acoustic model for detected keywords. For keyword detection, we adopt a wildcard CTC that is both fast and tolerant of ambiguous matches, allowing flexible handling of words that are difficult to match strictly. Since this method does not require retraining of existing models, it can be easily applied to even large-scale models. In experiments on Japanese speech recognition, the proposed method achieved a 29% improvement in the F1 score for unknown words. 

---
# Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis 

**Authors**: Jisoo Mok, Ik-hwan Kim, Sangkwon Park, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.01262)  

**Abstract**: Personalized AI assistants, a hallmark of the human-like capabilities of Large Language Models (LLMs), are a challenging application that intertwines multiple problems in LLM research. Despite the growing interest in the development of personalized assistants, the lack of an open-source conversational dataset tailored for personalization remains a significant obstacle for researchers in the field. To address this research gap, we introduce HiCUPID, a new benchmark to probe and unleash the potential of LLMs to deliver personalized responses. Alongside a conversational dataset, HiCUPID provides a Llama-3.2-based automated evaluation model whose assessment closely mirrors human preferences. We release our dataset, evaluation model, and code at this https URL. 

---
# DeepSeek in Healthcare: A Survey of Capabilities, Risks, and Clinical Applications of Open-Source Large Language Models 

**Authors**: Jiancheng Ye, Sophie Bronstein, Jiarui Hai, Malak Abu Hashish  

**Link**: [PDF](https://arxiv.org/pdf/2506.01257)  

**Abstract**: DeepSeek-R1 is a cutting-edge open-source large language model (LLM) developed by DeepSeek, showcasing advanced reasoning capabilities through a hybrid architecture that integrates mixture of experts (MoE), chain of thought (CoT) reasoning, and reinforcement learning. Released under the permissive MIT license, DeepSeek-R1 offers a transparent and cost-effective alternative to proprietary models like GPT-4o and Claude-3 Opus; it excels in structured problem-solving domains such as mathematics, healthcare diagnostics, code generation, and pharmaceutical research. The model demonstrates competitive performance on benchmarks like the United States Medical Licensing Examination (USMLE) and American Invitational Mathematics Examination (AIME), with strong results in pediatric and ophthalmologic clinical decision support tasks. Its architecture enables efficient inference while preserving reasoning depth, making it suitable for deployment in resource-constrained settings. However, DeepSeek-R1 also exhibits increased vulnerability to bias, misinformation, adversarial manipulation, and safety failures - especially in multilingual and ethically sensitive contexts. This survey highlights the model's strengths, including interpretability, scalability, and adaptability, alongside its limitations in general language fluency and safety alignment. Future research priorities include improving bias mitigation, natural language comprehension, domain-specific validation, and regulatory compliance. Overall, DeepSeek-R1 represents a major advance in open, scalable AI, underscoring the need for collaborative governance to ensure responsible and equitable deployment. 

---
# Memory-Efficient FastText: A Comprehensive Approach Using Double-Array Trie Structures and Mark-Compact Memory Management 

**Authors**: Yimin Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.01254)  

**Abstract**: FastText has established itself as a fundamental algorithm for learning word representations, demonstrating exceptional capability in handling out-of-vocabulary words through character-level n-gram embeddings. However, its hash-based bucketing mechanism introduces critical limitations for large-scale industrial deployment: hash collisions cause semantic drift, and memory requirements become prohibitively expensive when dealing with real-world vocabularies containing millions of terms. This paper presents a comprehensive memory optimization framework that fundamentally reimagines FastText's memory management through the integration of double-array trie (DA-trie) structures and mark-compact garbage collection principles. Our approach leverages the linguistic insight that n-grams sharing common prefixes or suffixes exhibit highly correlated embeddings due to co-occurrence patterns in natural language. By systematically identifying and merging semantically similar embeddings based on structural relationships, we achieve compression ratios of 4:1 to 10:1 while maintaining near-perfect embedding quality. The algorithm consists of four sophisticated phases: prefix trie construction with embedding mapping, prefix-based similarity compression, suffix-based similarity compression, and mark-compact memory reorganization. Comprehensive experiments on a 30-million Chinese vocabulary dataset demonstrate memory reduction from over 100GB to approximately 30GB with negligible performance degradation. Our industrial deployment results show significant cost reduction, faster loading times, and improved model reliability through the elimination of hash collision artifacts. Code and experimental implementations are available at: this https URL 

---
# CoRE: Condition-based Reasoning for Identifying Outcome Variance in Complex Events 

**Authors**: Sai Vallurupalli, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2506.01253)  

**Abstract**: Knowing which latent conditions lead to a particular outcome is useful for critically examining claims made about complex event outcomes. Identifying implied conditions and examining their influence on an outcome is challenging. We handle this by combining and augmenting annotations from two existing datasets consisting of goals and states, and explore the influence of conditions through our research questions and Condition-based Reasoning tasks. We examine open and closed LLMs of varying sizes and intent-alignment on our reasoning tasks and find that conditions are useful when not all context is available. Models differ widely in their ability to generate and identify outcome-variant conditions which affects their performance on outcome validation when conditions are used to replace missing context. Larger models like GPT-4o, are more cautious in such less constrained situations. 

---
# MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine 

**Authors**: Shufeng Kong, Xingru Yang, Yuanyuan Wei, Zijie Wang, Hao Tang, Jiuqi Qin, Shuting Lan, Yingheng Wang, Junwen Bai, Zhuangbin Chen, Zibin Zheng, Caihua Liu, Hao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01252)  

**Abstract**: Traditional Chinese Medicine (TCM) is a holistic medical system with millennia of accumulated clinical experience, playing a vital role in global healthcare-particularly across East Asia. However, the implicit reasoning, diverse textual forms, and lack of standardization in TCM pose major challenges for computational modeling and evaluation. Large Language Models (LLMs) have demonstrated remarkable potential in processing natural language across diverse domains, including general medicine. Yet, their systematic evaluation in the TCM domain remains underdeveloped. Existing benchmarks either focus narrowly on factual question answering or lack domain-specific tasks and clinical realism. To fill this gap, we introduce MTCMB-a Multi-Task Benchmark for Evaluating LLMs on TCM Knowledge, Reasoning, and Safety. Developed in collaboration with certified TCM experts, MTCMB comprises 12 sub-datasets spanning five major categories: knowledge QA, language understanding, diagnostic reasoning, prescription generation, and safety evaluation. The benchmark integrates real-world case records, national licensing exams, and classical texts, providing an authentic and comprehensive testbed for TCM-capable models. Preliminary results indicate that current LLMs perform well on foundational knowledge but fall short in clinical reasoning, prescription planning, and safety compliance. These findings highlight the urgent need for domain-aligned benchmarks like MTCMB to guide the development of more competent and trustworthy medical AI systems. All datasets, code, and evaluation tools are publicly available at: this https URL. 

---
# ExpertLongBench: Benchmarking Language Models on Expert-Level Long-Form Generation Tasks with Structured Checklists 

**Authors**: Jie Ruan, Inderjeet Nair, Shuyang Cao, Amy Liu, Sheza Munir, Micah Pollens-Dempsey, Tiffany Chiang, Lucy Kates, Nicholas David, Sihan Chen, Ruxin Yang, Yuqian Yang, Jasmine Gump, Tessa Bialek, Vivek Sankaran, Margo Schlanger, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01241)  

**Abstract**: This paper introduces ExpertLongBench, an expert-level benchmark containing 11 tasks from 9 domains that reflect realistic expert workflows and applications. Beyond question answering, the application-driven tasks in ExpertLongBench demand long-form outputs that can exceed 5,000 tokens and strict adherence to domain-specific requirements. Notably, each task in ExpertLongBench includes a rubric, designed or validated by domain experts, to specify task requirements and guide output evaluation. Furthermore, we propose CLEAR, an evaluation framework that supports accurate evaluation of long-form model outputs in our benchmark. To achieve fine-grained, expert-aligned evaluation, CLEAR derives checklists from both model outputs and references by extracting information corresponding to items in the task-specific rubric. Checklist items for model outputs are then compared with corresponding items for reference outputs to assess their correctness, enabling grounded evaluation. We benchmark 11 large language models (LLMs) and analyze components in CLEAR, showing that (1) existing LLMs, with the top performer achieving only a 26.8% F1 score, require significant improvement for expert-level tasks; (2) models can generate content corresponding to the required aspects, though often not accurately; and (3) accurate checklist extraction and comparison in CLEAR can be achieved by open-weight models for more scalable and low-cost usage. 

---
# Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean 

**Authors**: SungHo Kim, Nayeon Kim, Taehee Jeon, SangKeun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01237)  

**Abstract**: We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: this https URL. 

---
# Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers 

**Authors**: Woomin Song, Sai Muralidhar Jayanthi, Srikanth Ronanki, Kanthashree Mysore Sathyendra, Jinwoo Shin, Aram Galstyan, Shubham Katiyar, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2506.01215)  

**Abstract**: As large language models increasingly gain popularity in real-world applications, processing extremely long contexts, often exceeding the model's pre-trained context limits, has emerged as a critical challenge. While existing approaches to efficient long-context processing show promise, recurrent compression-based methods struggle with information preservation, whereas random access approaches require substantial memory resources. We introduce REFORM, a novel inference framework that efficiently handles long contexts through a two-phase approach. First, it incrementally processes input chunks while maintaining a compressed KV cache, constructs cross-layer context embeddings, and utilizes early exit strategy for improved efficiency. Second, it identifies and gathers essential tokens via similarity matching and selectively recomputes the KV cache. Compared to baselines, REFORM achieves over 50% and 27% performance gains on RULER and BABILong respectively at 1M context length. It also outperforms baselines on Infinite-Bench and MM-NIAH, demonstrating flexibility across diverse tasks and domains. Additionally, REFORM reduces inference time by 30% and peak memory usage by 5%, achieving both efficiency and superior performance. 

---
# Mamba Drafters for Speculative Decoding 

**Authors**: Daewon Choi, Seunghyuk Oh, Saket Dingliwal, Jihoon Tack, Kyuyoung Kim, Woomin Song, Seojin Kim, Insu Han, Jinwoo Shin, Aram Galstyan, Shubham Katiyar, Sravan Babu Bodapati  

**Link**: [PDF](https://arxiv.org/pdf/2506.01206)  

**Abstract**: Speculative decoding has emerged as a promising approach to accelerating large language model (LLM) generation using a fast drafter while maintaining alignment with the target model's distribution. However, existing approaches face a trade-off: external drafters offer flexibility but can suffer from slower drafting, while self-speculation methods use drafters tailored to the target model but require re-training. In this paper, we introduce novel drafters based on Mamba, a state-of-the-art state space model (SSM), as a solution that combines the best aspects of both approaches. By leveraging the linear structure of SSMs, our approach avoids the quadratic complexity inherent in traditional Transformer-based methods, enabling faster drafting and lower memory usage while maintaining the flexibility to work across different target models. We further enhance efficiency with a novel test-time tree search algorithm for generating high-quality draft candidates. Our empirical evaluation demonstrates that Mamba-based drafters not only outperform existing external drafting methods but are also comparable to state-of-the-art self-speculation approaches while using less memory and maintaining their cross-model adaptability. 

---
# Trick or Neat: Adversarial Ambiguity and Language Model Evaluation 

**Authors**: Antonia Karamolegkou, Oliver Eberle, Phillip Rust, Carina Kauf, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2506.01205)  

**Abstract**: Detecting ambiguity is important for language understanding, including uncertainty estimation, humour detection, and processing garden path sentences. We assess language models' sensitivity to ambiguity by introducing an adversarial ambiguity dataset that includes syntactic, lexical, and phonological ambiguities along with adversarial variations (e.g., word-order changes, synonym replacements, and random-based alterations). Our findings show that direct prompting fails to robustly identify ambiguity, while linear probes trained on model representations can decode ambiguity with high accuracy, sometimes exceeding 90\%. Our results offer insights into the prompting paradigm and how language models encode ambiguity at different layers. We release both our code and data: this https URL. 

---
# Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures 

**Authors**: Mark Muchane, Sean Richardson, Kiho Park, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2506.01197)  

**Abstract**: Sparse dictionary learning (and, in particular, sparse autoencoders) attempts to learn a set of human-understandable concepts that can explain variation on an abstract space. A basic limitation of this approach is that it neither exploits nor represents the semantic relationships between the learned concepts. In this paper, we introduce a modified SAE architecture that explicitly models a semantic hierarchy of concepts. Application of this architecture to the internal representations of large language models shows both that semantic hierarchy can be learned, and that doing so improves both reconstruction and interpretability. Additionally, the architecture leads to significant improvements in computational efficiency. 

---
# CoBRA: Quantifying Strategic Language Use and LLM Pragmatics 

**Authors**: Anshun Asher Zheng, Junyi Jessy Li, David I. Beaver  

**Link**: [PDF](https://arxiv.org/pdf/2506.01195)  

**Abstract**: Language is often used strategically, particularly in high-stakes, adversarial settings, yet most work on pragmatics and LLMs centers on cooperativity. This leaves a gap in systematic understanding of non-cooperative discourse. To address this, we introduce CoBRA (Cooperation-Breach Response Assessment), along with three interpretable metrics -- Benefit at Turn (BaT), Penalty at Turn (PaT), and Normalized Relative Benefit at Turn (NRBaT) -- to quantify the perceived strategic effects of discourse moves. We also present CHARM, an annotated dataset of real courtroom cross-examinations, to demonstrate the framework's effectiveness. Using these tools, we evaluate a range of LLMs and show that LLMs generally exhibit limited pragmatic understanding of strategic language. While model size shows an increase in performance on our metrics, reasoning ability does not help and largely hurts, introducing overcomplication and internal confusion. 

---
# Culturally-Grounded Chain-of-Thought (CG-CoT):Enhancing LLM Performance on Culturally-Specific Tasks in Low-Resource Languages 

**Authors**: Madhavendra Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2506.01190)  

**Abstract**: Large Language Models (LLMs) struggle with culturally-specific reasoning tasks, particularly in low-resource languages, hindering their global applicability. Addressing this gap is crucial for equitable AI deployment. We introduce Culturally-Grounded Chain-of-Thought (CG-CoT), a novel prompting strategy that combines dense vector retrieval of cultural context with explicit reasoning sequences. Our extensive experiments on Yoruba proverb interpretation demonstrate that CG-CoT provides significantly higher culturally-aligned accuracy and depth than traditional prompting methods, validated through both automated metrics and LLM-based evaluations. Notably, we uncover stark disparities between token-level translation metrics like BLEU and human-judged cultural relevance, suggesting a rethinking of evaluation approaches for low-resource NLP. 

---
# LAQuer: Localized Attribution Queries in Content-grounded Generation 

**Authors**: Eran Hirsch, Aviv Slobodkin, David Wan, Elias Stengel-Eskin, Mohit Bansal, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2506.01187)  

**Abstract**: Grounded text generation models often produce content that deviates from their source material, requiring user verification to ensure accuracy. Existing attribution methods associate entire sentences with source documents, which can be overwhelming for users seeking to fact-check specific claims. In contrast, existing sub-sentence attribution methods may be more precise but fail to align with users' interests. In light of these limitations, we introduce Localized Attribution Queries (LAQuer), a new task that localizes selected spans of generated output to their corresponding source spans, allowing fine-grained and user-directed attribution. We compare two approaches for the LAQuer task, including prompting large language models (LLMs) and leveraging LLM internal representations. We then explore a modeling framework that extends existing attributed text generation methods to LAQuer. We evaluate this framework across two grounded text generation tasks: Multi-document Summarization (MDS) and Long-form Question Answering (LFQA). Our findings show that LAQuer methods significantly reduce the length of the attributed text. Our contributions include: (1) proposing the LAQuer task to enhance attribution usability, (2) suggesting a modeling framework and benchmarking multiple baselines, and (3) proposing a new evaluation setting to promote future research on localized attribution in content-grounded generation. 

---
# The Inverse Scaling Effect of Pre-Trained Language Model Surprisal Is Not Due to Data Leakage 

**Authors**: Byung-Doh Oh, Hongao Zhu, William Schuler  

**Link**: [PDF](https://arxiv.org/pdf/2506.01172)  

**Abstract**: In psycholinguistic modeling, surprisal from larger pre-trained language models has been shown to be a poorer predictor of naturalistic human reading times. However, it has been speculated that this may be due to data leakage that caused language models to see the text stimuli during training. This paper presents two studies to address this concern at scale. The first study reveals relatively little leakage of five naturalistic reading time corpora in two pre-training datasets in terms of length and frequency of token $n$-gram overlap. The second study replicates the negative relationship between language model size and the fit of surprisal to reading times using models trained on 'leakage-free' data that overlaps only minimally with the reading time corpora. Taken together, this suggests that previous results using language models trained on these corpora are not driven by the effects of data leakage. 

---
# Mispronunciation Detection Without L2 Pronunciation Dataset in Low-Resource Setting: A Case Study in Finland Swedish 

**Authors**: Nhan Phan, Mikko Kuronen, Maria Kautonen, Riikka Ullakonoja, Anna von Zansen, Yaroslav Getman, Ekaterina Voskoboinik, Tamás Grósz, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01156)  

**Abstract**: Mispronunciation detection (MD) models are the cornerstones of many language learning applications. Unfortunately, most systems are built for English and other major languages, while low-resourced language varieties, such as Finland Swedish (FS), lack such tools. In this paper, we introduce our MD model for FS, trained on 89 hours of first language (L1) speakers' spontaneous speech and tested on 33 minutes of L2 transcribed read-aloud speech.
We trained a multilingual wav2vec 2.0 model with entropy regularization, followed by temperature scaling and top-k normalization after the inference to better adapt it for MD. The main novelty of our method lies in its simplicity, requiring minimal L2 data. The process is also language-independent, making it suitable for other low-resource languages. Our proposed algorithm allows us to balance Recall (43.2%) and Precision (29.8%), compared with the baseline model's Recall (77.5%) and Precision (17.6%). 

---
# A Word is Worth 4-bit: Efficient Log Parsing with Binary Coded Decimal Recognition 

**Authors**: Prerak Srivastava, Giulio Corallo, Sergey Rybalko  

**Link**: [PDF](https://arxiv.org/pdf/2506.01147)  

**Abstract**: System-generated logs are typically converted into categorical log templates through parsing. These templates are crucial for generating actionable insights in various downstream tasks. However, existing parsers often fail to capture fine-grained template details, leading to suboptimal accuracy and reduced utility in downstream tasks requiring precise pattern identification. We propose a character-level log parser utilizing a novel neural architecture that aggregates character embeddings. Our approach estimates a sequence of binary-coded decimals to achieve highly granular log templates extraction. Our low-resource character-level parser, tested on revised Loghub-2k and a manually annotated industrial dataset, matches LLM-based parsers in accuracy while outperforming semantic parsers in efficiency. 

---
# From Words to Waves: Analyzing Concept Formation in Speech and Text-Based Foundation Models 

**Authors**: Asım Ersoy, Basel Mousi, Shammur Chowdhury, Firoj Alam, Fahim Dalvi, Nadir Durrani  

**Link**: [PDF](https://arxiv.org/pdf/2506.01133)  

**Abstract**: The emergence of large language models (LLMs) has demonstrated that systems trained solely on text can acquire extensive world knowledge, develop reasoning capabilities, and internalize abstract semantic concepts--showcasing properties that can be associated with general intelligence. This raises an intriguing question: Do such concepts emerge in models trained on other modalities, such as speech? Furthermore, when models are trained jointly on multiple modalities: Do they develop a richer, more structured semantic understanding? To explore this, we analyze the conceptual structures learned by speech and textual models both individually and jointly. We employ Latent Concept Analysis, an unsupervised method for uncovering and interpreting latent representations in neural networks, to examine how semantic abstractions form across modalities. For reproducibility we made scripts and other resources available to the community. 

---
# Contextual Candor: Enhancing LLM Trustworthiness Through Hierarchical Unanswerability Detection 

**Authors**: Steven Robinson, Antonio Carlos Rivera  

**Link**: [PDF](https://arxiv.org/pdf/2506.01104)  

**Abstract**: The pervasive deployment of large language models (LLMs) in conversational AI systems has revolutionized information access, yet their propensity for generating factually unsupported or hallucinated responses remains a critical impediment to trustworthiness and widespread adoption. This paper introduces Reinforced Unanswerability Learning (RUL), a novel hybrid training paradigm designed to imbue LLMs with the intrinsic capability to accurately detect unanswerable questions and generate reliably appropriate responses. Unlike conventional approaches that rely on external classifiers or simple prompting, RUL integrates a discriminative unanswerability prediction head with the LLM's generative core, guided by a multi-stage learning strategy. This includes supervised fine-tuning on a novel, richly annotated dataset, Enhanced-CAsT-Answerability (ECA), which features hierarchical answerability labels and ground-truth refusal responses. Crucially, RUL incorporates a subsequent reinforcement learning with human feedback (RLHF) phase to refine the nuance, helpfulness, and informativeness of refusal responses. Extensive experiments demonstrate RUL's superior performance, achieving significantly higher accuracy in unanswerability detection across sentence, paragraph, and ranking levels, and substantially increasing the generation of appropriate refusals for unanswerable queries, alongside strong performance on answerable questions. Human evaluations further corroborate RUL's effectiveness, highlighting a marked improvement in perceived helpfulness and trustworthiness, ultimately paving the way for more reliable and user-centric conversational AI. 

---
# Un-considering Contextual Information: Assessing LLMs' Understanding of Indexical Elements 

**Authors**: Metehan Oguz, Yavuz Bakman, Duygu Nur Yaldiz  

**Link**: [PDF](https://arxiv.org/pdf/2506.01089)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performances in tasks related to coreference resolution. However, previous studies mostly assessed LLM performance on coreference resolution with nouns and third person pronouns. This study evaluates LLM performance on coreference resolution with indexical like I, you, here and tomorrow, which come with unique challenges due to their linguistic properties. We present the first study examining how LLMs interpret indexicals in English, releasing the English Indexical Dataset with 1600 multiple-choice questions. We evaluate pioneering LLMs, including GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, and DeepSeek V3. Our results reveal that LLMs exhibit an impressive performance with some indexicals (I), while struggling with others (you, here, tomorrow), and that syntactic cues (e.g. quotation) contribute to LLM performance with some indexicals, while they reduce performance with others. Code and data are available at: this https URL. 

---
# zip2zip: Inference-Time Adaptive Vocabularies for Language Models via Token Compression 

**Authors**: Saibo Geng, Nathan Ranchin, Yunzhen yao, Maxime Peyrard, Chris Wendler, Michael Gastpar, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2506.01084)  

**Abstract**: Tokenization efficiency plays a critical role in the performance and cost of large language models (LLMs), yet most models rely on static tokenizers optimized for general-purpose corpora. These tokenizers' fixed vocabularies often fail to adapt to domain- or language-specific inputs, leading to longer token sequences and higher computational costs. We introduce zip2zip, a framework that enables LLMs to dynamically adjust token vocabulary at inference time, allowing for fewer generated tokens and thus faster inference. zip2zip consists of three key components: (1) a tokenizer based on Lempel-Ziv-Welch (LZW) compression that incrementally compresses tokens into reusable "hypertokens" on the fly; (2) an embedding layer that computes embeddings for newly formed hypertokens at runtime; and (3) a causal language modeling variant that trains the model to operate on hypertokenized, compressed sequences. We show that an existing LLM can be zip2zip-fied in 10 GPU-hours via parameter-efficient finetuning. The resulting zip2zip LLMs effectively learn to use hypertokens at inference time, reducing input and output sequence length by 20-60\%, with significant improvements in inference latency. 

---
# How Programming Concepts and Neurons Are Shared in Code Language Models 

**Authors**: Amir Hossein Kargaran, Yihong Liu, François Yvon, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2506.01074)  

**Abstract**: Several studies have explored the mechanisms of large language models (LLMs) in coding tasks, but most have focused on programming languages (PLs) in a monolingual setting. In this paper, we investigate the relationship between multiple PLs and English in the concept space of LLMs. We perform a few-shot translation task on 21 PL pairs using two Llama-based models. By decoding the embeddings of intermediate layers during this task, we observe that the concept space is closer to English (including PL keywords) and assigns high probabilities to English tokens in the second half of the intermediate layers. We analyze neuron activations for 11 PLs and English, finding that while language-specific neurons are primarily concentrated in the bottom layers, those exclusive to each PL tend to appear in the top layers. For PLs that are highly aligned with multiple other PLs, identifying language-specific neurons is not feasible. These PLs also tend to have a larger keyword set than other PLs and are closer to the model's concept space regardless of the input/output PL in the translation task. Our findings provide insights into how LLMs internally represent PLs, revealing structural patterns in the model's concept space. Code is available at this https URL. 

---
# SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models 

**Authors**: Thinh Pham, Nguyen Nguyen, Pratibha Zunjare, Weiyuan Chen, Yu-Min Tseng, Tu Vu  

**Link**: [PDF](https://arxiv.org/pdf/2506.01062)  

**Abstract**: We introduce SealQA, a new challenge benchmark for evaluating SEarch-Augmented Language models on fact-seeking questions where web search yields conflicting, noisy, or unhelpful results. SealQA comes in three flavors: (1) Seal-0 (main) and (2) Seal-Hard, which assess factual accuracy and reasoning capabilities, with Seal-0 focusing on the most challenging questions where chat models (e.g., GPT-4.1) typically achieve near-zero accuracy; and (3) LongSeal, which extends SealQA to test long-context, multi-document reasoning in "needle-in-a-haystack" settings. Our evaluation reveals critical limitations in current models: Even frontier LLMs perform poorly across all SealQA flavors. On Seal-0, frontier agentic models equipped with tools like o3 and o4-mini achieve only 17.1% and 6.3% accuracy, respectively, at their best reasoning efforts. We find that advanced reasoning models such as DeepSeek-R1-671B and o3-mini are highly vulnerable to noisy search results. Notably, increasing test-time compute does not yield reliable gains across o3-mini, o4-mini, and o3, with performance often plateauing or even declining early. Additionally, while recent models are less affected by the "lost-in-the-middle" issue, they still fail to reliably identify relevant documents in LongSeal when faced with numerous distractors. To facilitate future work, we release SealQA at this http URL. 

---
# CHEER-Ekman: Fine-grained Embodied Emotion Classification 

**Authors**: Phan Anh Duong, Cat Luong, Divyesh Bommana, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01047)  

**Abstract**: Emotions manifest through physical experiences and bodily reactions, yet identifying such embodied emotions in text remains understudied. We present an embodied emotion classification dataset, CHEER-Ekman, extending the existing binary embodied emotion dataset with Ekman's six basic emotion categories. Using automatic best-worst scaling with large language models, we achieve performance superior to supervised approaches on our new dataset. Our investigation reveals that simplified prompting instructions and chain-of-thought reasoning significantly improve emotion recognition accuracy, enabling smaller models to achieve competitive performance with larger ones. 

---
# Probing Neural Topology of Large Language Models 

**Authors**: Yu Zheng, Yuan Yuan, Yong Li, Paolo Santi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01042)  

**Abstract**: Probing large language models (LLMs) has yielded valuable insights into their internal mechanisms by linking neural representations to interpretable semantics. However, how neurons functionally co-activate with each other to give rise to emergent capabilities remains largely unknown, hindering a deeper understanding and safer development of LLMs. In this work, we introduce graph probing, a method for uncovering the functional connectivity topology of LLM neurons and relating it to language generation performance. By analyzing internal neural graphs across diverse LLM families and scales, we discover a universal predictability of next-token prediction performance using only neural topology. This predictability is robust even when retaining just 1% of neuron connections or probing models after only 8 pretraining steps, highlighting the sparsity and early emergence of topological patterns. Further graph matching analysis suggests that, despite significant distinctions in architectures, parameters, and training data, different LLMs develop intricate and consistent neural topological structures that may form the foundation for their language generation abilities. Codes and data for the graph probing toolbox are released at this https URL. 

---
# Less is More: Local Intrinsic Dimensions of Contextual Language Models 

**Authors**: Benjamin Matthias Ruppik, Julius von Rohrscheidt, Carel van Niekerk, Michael Heck, Renato Vukovic, Shutong Feng, Hsien-chin Lin, Nurul Lubis, Bastian Rieck, Marcus Zibrowius, Milica Gašić  

**Link**: [PDF](https://arxiv.org/pdf/2506.01034)  

**Abstract**: Understanding the internal mechanisms of large language models (LLMs) remains a challenging and complex endeavor. Even fundamental questions, such as how fine-tuning affects model behavior, often require extensive empirical evaluation. In this paper, we introduce a novel perspective based on the geometric properties of contextual latent embeddings to study the effects of training and fine-tuning. To that end, we measure the local dimensions of a contextual language model's latent space and analyze their shifts during training and fine-tuning. We show that the local dimensions provide insights into the model's training dynamics and generalization ability. Specifically, the mean of the local dimensions predicts when the model's training capabilities are exhausted, as exemplified in a dialogue state tracking task, overfitting, as demonstrated in an emotion recognition task, and grokking, as illustrated with an arithmetic task. Furthermore, our experiments suggest a practical heuristic: reductions in the mean local dimension tend to accompany and predict subsequent performance gains. Through this exploration, we aim to provide practitioners with a deeper understanding of the implications of fine-tuning on embedding spaces, facilitating informed decisions when configuring models for specific applications. The results of this work contribute to the ongoing discourse on the interpretability, adaptability, and generalizability of LLMs by bridging the gap between intrinsic model mechanisms and geometric properties in the respective embeddings. 

---
# Talking to Data: Designing Smart Assistants for Humanities Databases 

**Authors**: Alexander Sergeev, Valeriya Goloviznina, Mikhail Melnichenko, Evgeny Kotelnikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.00986)  

**Abstract**: Access to humanities research databases is often hindered by the limitations of traditional interaction formats, particularly in the methods of searching and response generation. This study introduces an LLM-based smart assistant designed to facilitate natural language communication with digital humanities data. The assistant, developed in a chatbot format, leverages the RAG approach and integrates state-of-the-art technologies such as hybrid search, automatic query generation, text-to-SQL filtering, semantic database search, and hyperlink insertion. To evaluate the effectiveness of the system, experiments were conducted to assess the response quality of various language models. The testing was based on the Prozhito digital archive, which contains diary entries from predominantly Russian-speaking individuals who lived in the 20th century. The chatbot is tailored to support anthropology and history researchers, as well as non-specialist users with an interest in the field, without requiring prior technical training. By enabling researchers to query complex databases with natural language, this tool aims to enhance accessibility and efficiency in humanities research. The study highlights the potential of Large Language Models to transform the way researchers and the public interact with digital archives, making them more intuitive and inclusive. Additional materials are presented in GitHub repository: this https URL. 

---
# Do LLMs Understand Why We Write Diaries? A Method for Purpose Extraction and Clustering 

**Authors**: Valeriya Goloviznina, Alexander Sergeev, Mikhail Melnichenko, Evgeny Kotelnikov  

**Link**: [PDF](https://arxiv.org/pdf/2506.00985)  

**Abstract**: Diary analysis presents challenges, particularly in extracting meaningful information from large corpora, where traditional methods often fail to deliver satisfactory results. This study introduces a novel method based on Large Language Models (LLMs) to identify and cluster the various purposes of diary writing. By "purposes," we refer to the intentions behind diary writing, such as documenting life events, self-reflection, or practicing language skills. Our approach is applied to Soviet-era diaries (1922-1929) from the Prozhito digital archive, a rich collection of personal narratives. We evaluate different proprietary and open-source LLMs, finding that GPT-4o and o1-mini achieve the best performance, while a template-based baseline is significantly less effective. Additionally, we analyze the retrieved purposes based on gender, age of the authors, and the year of writing. Furthermore, we examine the types of errors made by the models, providing a deeper understanding of their limitations and potential areas for improvement in future research. 

---
# What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training 

**Authors**: Marianne de Heer Kloots, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema, Martijn Bentum  

**Link**: [PDF](https://arxiv.org/pdf/2506.00981)  

**Abstract**: How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition. 

---
# LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World 

**Authors**: Sina J. Semnani, Pingyue Zhang, Wanyue Zhai, Haozhuo Li, Ryan Beauchamp, Trey Billing, Katayoun Kishi, Manling Li, Monica S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2506.00980)  

**Abstract**: This paper presents LEMONADE, a large-scale conflict event dataset comprising 39,786 events across 20 languages and 171 countries, with extensive coverage of region-specific entities. LEMONADE is based on a partially reannotated subset of the Armed Conflict Location & Event Data (ACLED), which has documented global conflict events for over a decade.
To address the challenge of aggregating multilingual sources for global event analysis, we introduce abstractive event extraction (AEE) and its subtask, abstractive entity linking (AEL). Unlike conventional span-based event extraction, our approach detects event arguments and entities through holistic document understanding and normalizes them across the multilingual dataset. We evaluate various large language models (LLMs) on these tasks, adapt existing zero-shot event extraction systems, and benchmark supervised models. Additionally, we introduce ZEST, a novel zero-shot retrieval-based system for AEL.
Our best zero-shot system achieves an end-to-end F1 score of 58.3%, with LLMs outperforming specialized event extraction models such as GoLLIE. For entity linking, ZEST achieves an F1 score of 45.7%, significantly surpassing OneNet, a state-of-the-art zero-shot baseline that achieves only 23.7%. However, these zero-shot results lag behind the best supervised systems by 20.1% and 37.0% in the end-to-end and AEL tasks, respectively, highlighting the need for further research. 

---
# NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction 

**Authors**: Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, Peilin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00975)  

**Abstract**: Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications. 

---
# XGUARD: A Graded Benchmark for Evaluating Safety Failures of Large Language Models on Extremist Content 

**Authors**: Vadivel Abishethvarman, Bhavik Chandna, Pratik Jalan, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.00973)  

**Abstract**: Large Language Models (LLMs) can generate content spanning ideological rhetoric to explicit instructions for violence. However, existing safety evaluations often rely on simplistic binary labels (safe and unsafe), overlooking the nuanced spectrum of risk these outputs pose. To address this, we present XGUARD, a benchmark and evaluation framework designed to assess the severity of extremist content generated by LLMs. XGUARD includes 3,840 red teaming prompts sourced from real world data such as social media and news, covering a broad range of ideologically charged scenarios. Our framework categorizes model responses into five danger levels (0 to 4), enabling a more nuanced analysis of both the frequency and severity of failures. We introduce the interpretable Attack Severity Curve (ASC) to visualize vulnerabilities and compare defense mechanisms across threat intensities. Using XGUARD, we evaluate six popular LLMs and two lightweight defense strategies, revealing key insights into current safety gaps and trade-offs between robustness and expressive freedom. Our work underscores the value of graded safety metrics for building trustworthy LLMs. 

---
# ACCESS DENIED INC: The First Benchmark Environment for Sensitivity Awareness 

**Authors**: Dren Fazlija, Arkadij Orlov, Sandipan Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00964)  

**Abstract**: Large language models (LLMs) are increasingly becoming valuable to corporate data management due to their ability to process text from various document formats and facilitate user interactions through natural language queries. However, LLMs must consider the sensitivity of information when communicating with employees, especially given access restrictions. Simple filtering based on user clearance levels can pose both performance and privacy challenges. To address this, we propose the concept of sensitivity awareness (SA), which enables LLMs to adhere to predefined access rights rules. In addition, we developed a benchmarking environment called ACCESS DENIED INC to evaluate SA. Our experimental findings reveal significant variations in model behavior, particularly in managing unauthorized data requests while effectively addressing legitimate queries. This work establishes a foundation for benchmarking sensitivity-aware language models and provides insights to enhance privacy-centric AI systems in corporate environments. 

---
# From Objectives to Questions: A Planning-based Framework for Educational Mathematical Question Generation 

**Authors**: Cheng Cheng, Zhenya Huang, Guanhao Zhao, Yuxiang Guo, Xin Lin, Jinze Wu, Xin Li, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00963)  

**Abstract**: Automatically generating high-quality mathematical problems that align with educational objectives is a crucial task in NLP-based educational technology. Traditional generation methods focus primarily on textual quality, but they often overlook educational objectives. Moreover, these methods address only single-dimensional, simple question generation, failing to meet complex, multifaceted educational requirements. To address these challenges, we constructed and annotated EduMath, a dataset of 16k mathematical questions with multi-dimensional educational objectives. Based on this dataset, we developed EQGEVAL, which incorporates three evaluation dimensions and is designed to assess the ability of models to generate educational questions. Drawing inspiration from teachers' problem design processes, we propose the Educational Question Planning with self-Reflection (EQPR) method for educational mathematical question generation, following a "plan-evaluate-optimize" approach. Specifically, by combining planning algorithm based on Monte Carlo Tree Search with the generative capabilities of Large Language Models, we continuously optimize questions through iterative feedback. This self-optimization mechanism ensures that the generated questions both fit the educational context and strategically achieve specific basic educational objectives. Through extensive experiments based on EQGEVAL, we have demonstrated that EQPR achieves significant improvements in generating questions that meet multi-dimensional educational objectives. 

---
# Leveraging Large Language Models for Sarcastic Speech Annotation in Sarcasm Detection 

**Authors**: Zhu Li, Yuqing Zhang, Xiyuan Gao, Shekhar Nayak, Matt Coler  

**Link**: [PDF](https://arxiv.org/pdf/2506.00955)  

**Abstract**: Sarcasm fundamentally alters meaning through tone and context, yet detecting it in speech remains a challenge due to data scarcity. In addition, existing detection systems often rely on multimodal data, limiting their applicability in contexts where only speech is available. To address this, we propose an annotation pipeline that leverages large language models (LLMs) to generate a sarcasm dataset. Using a publicly available sarcasm-focused podcast, we employ GPT-4o and LLaMA 3 for initial sarcasm annotations, followed by human verification to resolve disagreements. We validate this approach by comparing annotation quality and detection performance on a publicly available sarcasm dataset using a collaborative gating architecture. Finally, we introduce PodSarc, a large-scale sarcastic speech dataset created through this pipeline. The detection model achieves a 73.63% F1 score, demonstrating the dataset's potential as a benchmark for sarcasm detection research. 

---
# anyECG-chat: A Generalist ECG-MLLM for Flexible ECG Input and Multi-Task Understanding 

**Authors**: Haitao Li, Ziyu Li, Yiheng Mao, Ziyi Liu, Zhoujian Sun, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00942)  

**Abstract**: The advent of multimodal large language models (MLLMs) has sparked interest in their application to electrocardiogram (ECG) analysis. However, existing ECG-focused MLLMs primarily focus on report generation tasks, often limited to single 12-lead, short-duration (10s) ECG inputs, thereby underutilizing the potential of MLLMs. To this end, we aim to develop a MLLM for ECG analysis that supports a broader range of tasks and more flexible ECG inputs. However, existing ECG-QA datasets are often monotonous. To address this gap, we first constructed the anyECG dataset, which encompasses a wide variety of tasks, including report generation, abnormal waveform localization, and open-ended question answering. In addition to standard hospital ECGs, we introduced long-duration reduced-lead ECGs for home environments and multiple ECG comparison scenarios commonly encountered in clinical practice. Furthermore, we propose the anyECG-chat model, which supports dynamic-length ECG inputs and multiple ECG inputs. We trained the model using a three-stage curriculum training recipe with the anyECG dataset. A comprehensive evaluation was conducted, demonstrating that anyECG-chat is capable of supporting various practical application scenarios, including not only common report generation tasks but also abnormal waveform localization for long-duration reduced-lead ECGs in home environments and comprehensive comparative analysis of multiple ECGs. 

---
# How do Transformer Embeddings Represent Compositions? A Functional Analysis 

**Authors**: Aishik Nagar, Ishaan Singh Rawal, Mansi Dhanania, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00914)  

**Abstract**: Compositionality is a key aspect of human intelligence, essential for reasoning and generalization. While transformer-based models have become the de facto standard for many language modeling tasks, little is known about how they represent compound words, and whether these representations are compositional. In this study, we test compositionality in Mistral, OpenAI Large, and Google embedding models, and compare them with BERT. First, we evaluate compositionality in the representations by examining six diverse models of compositionality (addition, multiplication, dilation, regression, etc.). We find that ridge regression, albeit linear, best accounts for compositionality. Surprisingly, we find that the classic vector addition model performs almost as well as any other model. Next, we verify that most embedding models are highly compositional, while BERT shows much poorer compositionality. We verify and visualize our findings with a synthetic dataset consisting of fully transparent adjective-noun compositions. Overall, we present a thorough investigation of compositionality. 

---
# Pi-SQL: Enhancing Text-to-SQL with Fine-Grained Guidance from Pivot Programming Languages 

**Authors**: Yongdong chi, Hanqing Wang, Zonghan Yang, Jian Yang, Xiao Yan, Yun Chen, Guanhua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00912)  

**Abstract**: Text-to-SQL transforms the user queries from natural language to executable SQL programs, enabling non-experts to interact with complex databases. Existing prompt-based methods craft meticulous text guidelines and examples to facilitate SQL generation, but their accuracy is hindered by the large semantic gap between the texts and the low-resource SQL programs. In this work, we propose Pi-SQL, which incorporates the high-resource Python program as a pivot to bridge between the natural language query and SQL program. In particular, Pi-SQL first generates Python programs that provide fine-grained step-by-step guidelines in their code blocks or comments, and then produces an SQL program following the guidance of each Python this http URL final SQL program matches the reference Python program's query results and, through selection from candidates generated by different strategies, achieves superior execution speed, with a reward-based valid efficiency score up to 4.55 higher than the best-performing this http URL experiments demonstrate the effectiveness of Pi-SQL, which improves the execution accuracy of the best-performing baseline by up to 3.20. 

---
# SocialEval: Evaluating Social Intelligence of Large Language Models 

**Authors**: Jinfeng Zhou, Yuxuan Chen, Yihan Shi, Xuanming Zhang, Leqi Lei, Yi Feng, Zexuan Xiong, Miao Yan, Xunzhi Wang, Yaru Cao, Jianing Yin, Shuai Wang, Quanyu Dai, Zhenhua Dong, Hongning Wang, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00900)  

**Abstract**: LLMs exhibit promising Social Intelligence (SI) in modeling human behavior, raising the need to evaluate LLMs' SI and their discrepancy with humans. SI equips humans with interpersonal abilities to behave wisely in navigating social interactions to achieve social goals. This presents an operational evaluation paradigm: outcome-oriented goal achievement evaluation and process-oriented interpersonal ability evaluation, which existing work fails to address. To this end, we propose SocialEval, a script-based bilingual SI benchmark, integrating outcome- and process-oriented evaluation by manually crafting narrative scripts. Each script is structured as a world tree that contains plot lines driven by interpersonal ability, providing a comprehensive view of how LLMs navigate social interactions. Experiments show that LLMs fall behind humans on both SI evaluations, exhibit prosociality, and prefer more positive social behaviors, even if they lead to goal failure. Analysis of LLMs' formed representation space and neuronal activations reveals that LLMs have developed ability-specific functional partitions akin to the human brain. 

---
# Affordance Benchmark for MLLMs 

**Authors**: Junying Wang, Wenzhe Li, Yalun Wu, Yingji Liang, Yijin Guo, Chunyi Li, Haodong Duan, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00893)  

**Abstract**: Affordance theory posits that environments inherently offer action possibilities that shape perception and behavior. While Multimodal Large Language Models (MLLMs) excel in vision-language tasks, their ability to perceive affordance, which is crucial for intuitive and safe interactions, remains underexplored. To address this, we introduce A4Bench, a novel benchmark designed to evaluate the affordance perception abilities of MLLMs across two dimensions: 1) Constitutive Affordance}, assessing understanding of inherent object properties through 1,282 question-answer pairs spanning nine sub-disciplines, and 2) Transformative Affordance, probing dynamic and contextual nuances (e.g., misleading, time-dependent, cultural, or individual-specific affordance) with 718 challenging question-answer pairs. Evaluating 17 MLLMs (nine proprietary and eight open-source) against human performance, we find that proprietary models generally outperform open-source counterparts, but all exhibit limited capabilities, particularly in transformative affordance perception. Furthermore, even top-performing models, such as Gemini-2.0-Pro (18.05% overall exact match accuracy), significantly lag behind human performance (best: 85.34%, worst: 81.25%). These findings highlight critical gaps in environmental understanding of MLLMs and provide a foundation for advancing AI systems toward more robust, context-aware interactions. The dataset is available in this https URL. 

---
# Improve MLLM Benchmark Efficiency through Interview 

**Authors**: Farong Wen, Yijin Guo, Junying Wang, Jiaohao Xiao, Yingjie Zhou, Chunyi Li, Zicheng Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00883)  

**Abstract**: The rapid development of Multimodal Large Language Models (MLLM) has led to a wide range of MLLM applications, and a number of benchmark datasets have sprung up in order to assess MLLM abilities. However, full-coverage Q&A testing on large-scale data is resource-intensive and time-consuming. To address this issue, we propose the MLLM Interview (MITV) strategy, which aims to quickly obtain MLLM performance metrics by quizzing fewer question. First, First, we constructed the interview dataset, which was built on an existing MLLM assessment dataset, by adding difficulty labels based on the performance of some typical MLLMs in this dataset. Second, we propose an MLLM Interview strategy, which obtains an initial performance situation of the large model by quizzing a small number of topics and then continuously tries to test the model's limits. Through extensive experiments, the result shows that the MITV strategy proposed in this paper performs well on MLLM benchmark datasets, and it is able to obtain the model evaluation capability faster through a small number of questions and answers. 

---
# Not Every Token Needs Forgetting: Selective Unlearning to Limit Change in Utility in Large Language Model Unlearning 

**Authors**: Yixin Wan, Anil Ramakrishna, Kai-Wei Chang, Volkan Cevher, Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.00876)  

**Abstract**: Large Language Model (LLM) unlearning has recently gained significant attention, driven by the need to remove unwanted information, such as private, sensitive, or copyrighted content, from LLMs. However, conventional unlearning approaches indiscriminately update model parameters to forget all tokens in a target document, including common tokens (e.g., pronouns, prepositions, general nouns) that carry general knowledge. In this paper, we highlight that not every token needs forgetting. We propose Selective Unlearning (SU), which identifies a critical subset of tokens within the forgetting set that is relevant to the unwanted information, and unlearns only those tokens. Experiments on two benchmarks and six baseline unlearning algorithms demonstrate that SU not only achieves effective unlearning on the targeted forget data, but also significantly preserves the model's utility in the retaining set. 

---
# CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning 

**Authors**: Yangfan Ye, Xiaocheng Feng, Zekun Yuan, Xiachong Feng, Libo Qin, Lei Huang, Weitao Ma, Yichong Huang, Zhirui Zhang, Yunfei Lu, Xiaohui Yan, Duyu Tang, Dandan Tu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00875)  

**Abstract**: Current large language models (LLMs) often exhibit imbalanced multilingual capabilities due to their English-centric training corpora. To address this, existing fine-tuning approaches operating at the data-level (e.g., through data augmentation or distillation) typically introduce implicit cross-lingual alignment, overlooking the potential for more profound, latent-level cross-lingual interactions. In this work, we propose CC-Tuning, a novel multilingual fine-tuning paradigm that explicitly establishes a cross-lingual connection mechanism at the latent level. During training, CC-Tuning fuses the feed forward activations from both English and non-English inputs, enabling the model to benefit from both linguistic resources. This process is facilitated with a trainable Decision Maker that identifies beneficial activations. Furthermore, during inference, a Transform Matrix is utilized to simulate the cross-lingual connection under monolingual setting through representation transformation. Our experiments on six benchmarks covering 22 languages show that CC-Tuning outperforms vanilla SFT and offers a strong latent-level alternative to data-level augmentation methods. Further analysis also highlights the practicality of CC-Tuning and the potential of latent-level cross-lingual interactions in advancing the multilingual performance of LLMs. 

---
# What's Missing in Vision-Language Models? Probing Their Struggles with Causal Order Reasoning 

**Authors**: Zhaotian Weng, Haoxuan Li, Kuan-Hao Huang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00869)  

**Abstract**: Despite the impressive performance of vision-language models (VLMs) on downstream tasks, their ability to understand and reason about causal relationships in visual inputs remains unclear. Robust causal reasoning is fundamental to solving complex high-level reasoning tasks, yet existing benchmarks often include a mixture of reasoning questions, and VLMs can frequently exploit object recognition and activity identification as shortcuts to arrive at the correct answers, making it challenging to truly assess their causal reasoning abilities. To bridge this gap, we introduce VQA-Causal and VCR-Causal, two new benchmarks specifically designed to isolate and rigorously evaluate VLMs' causal reasoning abilities. Our findings reveal that while VLMs excel in object and activity recognition, they perform poorly on causal reasoning tasks, often only marginally surpassing random guessing. Further analysis suggests that this limitation stems from a severe lack of causal expressions in widely used training datasets, where causal relationships are rarely explicitly conveyed. We additionally explore fine-tuning strategies with hard negative cases, showing that targeted fine-tuning can improve model's causal reasoning while maintaining generalization and downstream performance. Our study highlights a key gap in current VLMs and lays the groundwork for future work on causal understanding. 

---
# L3Cube-MahaEmotions: A Marathi Emotion Recognition Dataset with Synthetic Annotations using CoTR prompting and Large Language Models 

**Authors**: Nidhi Kowtal, Raviraj Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00863)  

**Abstract**: Emotion recognition in low-resource languages like Marathi remains challenging due to limited annotated data. We present L3Cube-MahaEmotions, a high-quality Marathi emotion recognition dataset with 11 fine-grained emotion labels. The training data is synthetically annotated using large language models (LLMs), while the validation and test sets are manually labeled to serve as a reliable gold-standard benchmark. Building on the MahaSent dataset, we apply the Chain-of-Translation (CoTR) prompting technique, where Marathi sentences are translated into English and emotion labeled via a single prompt. GPT-4 and Llama3-405B were evaluated, with GPT-4 selected for training data annotation due to superior label quality. We evaluate model performance using standard metrics and explore label aggregation strategies (e.g., Union, Intersection). While GPT-4 predictions outperform fine-tuned BERT models, BERT-based models trained on synthetic labels fail to surpass GPT-4. This highlights both the importance of high-quality human-labeled data and the inherent complexity of emotion recognition. An important finding of this work is that generic LLMs like GPT-4 and Llama3-405B generalize better than fine-tuned BERT for complex low-resource emotion recognition tasks. The dataset and model are shared publicly at this https URL 

---
# How Bidirectionality Helps Language Models Learn Better via Dynamic Bottleneck Estimation 

**Authors**: Md Kowsher, Nusrat Jahan Prottasha, Shiyun Xu, Shetu Mohanto, Chen Chen, Niloofar Yousefi, Ozlem Garibay  

**Link**: [PDF](https://arxiv.org/pdf/2506.00859)  

**Abstract**: Bidirectional language models have better context understanding and perform better than unidirectional models on natural language understanding tasks, yet the theoretical reasons behind this advantage remain unclear. In this work, we investigate this disparity through the lens of the Information Bottleneck (IB) principle, which formalizes a trade-off between compressing input information and preserving task-relevant content. We propose FlowNIB, a dynamic and scalable method for estimating mutual information during training that addresses key limitations of classical IB approaches, including computational intractability and fixed trade-off schedules. Theoretically, we show that bidirectional models retain more mutual information and exhibit higher effective dimensionality than unidirectional models. To support this, we present a generalized framework for measuring representational complexity and prove that bidirectional representations are strictly more informative under mild conditions. We further validate our findings through extensive experiments across multiple models and tasks using FlowNIB, revealing how information is encoded and compressed throughout training. Together, our work provides a principled explanation for the effectiveness of bidirectional architectures and introduces a practical tool for analyzing information flow in deep language models. 

---
# EEG2TEXT-CN: An Exploratory Study of Open-Vocabulary Chinese Text-EEG Alignment via Large Language Model and Contrastive Learning on ChineseEEG 

**Authors**: Jacky Tai-Yu Lu, Jung Chiang, Chi-Sheng Chen, Anna Nai-Yun Tung, Hsiang Wei Hu, Yuan Chiao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00854)  

**Abstract**: We propose EEG2TEXT-CN, which, to the best of our knowledge, represents one of the earliest open-vocabulary EEG-to-text generation frameworks tailored for Chinese. Built on a biologically grounded EEG encoder (NICE-EEG) and a compact pretrained language model (MiniLM), our architecture aligns multichannel brain signals with natural language representations via masked pretraining and contrastive learning. Using a subset of the ChineseEEG dataset, where each sentence contains approximately ten Chinese characters aligned with 128-channel EEG recorded at 256 Hz, we segment EEG into per-character embeddings and predict full sentences in a zero-shot setting. The decoder is trained with teacher forcing and padding masks to accommodate variable-length sequences. Evaluation on over 1,500 training-validation sentences and 300 held-out test samples shows promising lexical alignment, with a best BLEU-1 score of 6.38\%. While syntactic fluency remains a challenge, our findings demonstrate the feasibility of non-phonetic, cross-modal language decoding from EEG. This work opens a new direction in multilingual brain-to-text research and lays the foundation for future cognitive-language interfaces in Chinese. 

---
# Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience 

**Authors**: Jiawei Gu, Ziting Xian, Yuanzhen Xie, Ye Liu, Enjie Liu, Ruichao Zhong, Mochi Gao, Yunzhi Tan, Bo Hu, Zang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00842)  

**Abstract**: Large language models (LLMs) achieve strong performance on plain text tasks but underperform on structured data like tables and databases. Potential challenges arise from their underexposure during pre-training and rigid text-to-structure transfer mechanisms. Unlike humans who seamlessly apply learned patterns across data modalities, LLMs struggle to infer implicit relationships embedded in tabular formats, especially in the absence of explicit structural guidance. To bridge this cognitive gap, we introduce Contrastive Retrieval-Augmented Generation on Experience (CoRE), a framework that builds experience memory representations and enhances generalization through contrastive In-Context Learning (ICL) to simulate human-like knowledge transfer. Experiments on Text-to-SQL and TableQA show CoRE significantly improves performance, achieving average gains of 3.44% and 4.24%, with up to 17.2% on challenging tasks. Our Monte Carlo Tree Search (MCTS)-generated Experience Memory expands training data 8-9x, enhancing diversity and domain coverage. This training-free and continual method propels LLMs toward structured knowledge expertise. 

---
# COMPKE: Complex Question Answering under Knowledge Editing 

**Authors**: Keyuan Cheng, Zijian Kan, Zhixian He, Zhuoran Zhang, Muhammad Asif Ali, Ke Xu, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00829)  

**Abstract**: Knowledge Editing, which efficiently modifies the knowledge in large language models, has gathered great attention. Current benchmarks primarily use multi-hop question answering to assess and analyze newly injected or updated knowledge. However, we argue that these benchmarks fail to effectively evaluate how well the updated models apply this knowledge in real-life scenarios, particularly when questions require complex reasoning, involving one-to-many relationships or multi-step logical intersections. To fill in this gap, we introduce a new benchmark, COMPKE: Complex Question Answering under Knowledge Editing, which includes 11,924 complex questions that reflect real-life situations. We conduct an extensive evaluation of four knowledge editing methods on COMPKE, revealing that their effectiveness varies notably across different models. For instance, MeLLo attains an accuracy of 39.47 on GPT-4O-MINI, but this drops sharply to 3.83 on QWEN2.5-3B. We further investigate the underlying causes of these disparities from both methodological and model-specific perspectives. The datasets are available at this https URL. 

---
# HERGC: Heterogeneous Experts Representation and Generative Completion for Multimodal Knowledge Graphs 

**Authors**: Yongkang Xiao, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00826)  

**Abstract**: Multimodal knowledge graphs (MMKGs) enrich traditional knowledge graphs (KGs) by incorporating diverse modalities such as images and text. Multi-modal knowledge graph completion (MMKGC) seeks to exploit these heterogeneous signals to infer missing facts, thereby mitigating the intrinsic incompleteness of MMKGs. Existing MMKGC methods typically leverage only the information contained in the MMKGs under the closed-world assumption and adopt discriminative training objectives, which limits their reasoning capacity during completion. Recent generative completion approaches powered by advanced large language models (LLMs) have shown strong reasoning abilities in unimodal knowledge graph completion, but their potential in MMKGC remains largely unexplored. To bridge this gap, we propose HERGC, a Heterogeneous Experts Representation and Generative Completion framework for MMKGs. HERGC first deploys a Heterogeneous Experts Representation Retriever that enriches and fuses multimodal information and retrieves a compact candidate set for each incomplete triple. It then uses a Generative LLM Predictor fine-tuned on minimal instruction data to accurately identify the correct answer from these candidates. Extensive experiments on three standard MMKG benchmarks demonstrate HERGC's effectiveness and robustness, achieving state-of-the-art performance. 

---
# Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs Across Logical Transformations and Question Answering Tasks 

**Authors**: Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Zhengwen Feng, Hao Peng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00823)  

**Abstract**: Large language models (LLMs) are trained on extensive datasets that encapsulate substantial world knowledge. However, their outputs often include confidently stated inaccuracies. Earlier works suggest that LLMs encode truthfulness as a distinct linear feature, termed the "truth direction", which can classify truthfulness reliably. We address several open questions about the truth direction: (i) whether LLMs universally exhibit consistent truth directions; (ii) whether sophisticated probing techniques are necessary to identify truth directions; and (iii) how the truth direction generalizes across diverse contexts. Our findings reveal that not all LLMs exhibit consistent truth directions, with stronger representations observed in more capable models, particularly in the context of logical negation. Additionally, we demonstrate that truthfulness probes trained on declarative atomic statements can generalize effectively to logical transformations, question-answering tasks, in-context learning, and external knowledge sources. Finally, we explore the practical application of truthfulness probes in selective question-answering, illustrating their potential to improve user trust in LLM outputs. These results advance our understanding of truth directions and provide new insights into the internal representations of LLM beliefs. Our code is public at this https URL 

---
# One for All: Update Parameterized Knowledge Across Multiple Models 

**Authors**: Weitao Ma, Xiyuan Du, Xiaocheng Feng, Lei Huang, Yichong Huang, Huiyi Zhang, Xiaoliang Yang, Baohang Li, Xiachong Feng, Ting Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00817)  

**Abstract**: Large language models (LLMs) encode vast world knowledge but struggle to stay up-to-date, often leading to errors and hallucinations. Knowledge editing offers an efficient alternative to retraining, enabling targeted modifications by updating specific model parameters. However, existing methods primarily focus on individual models, posing challenges in efficiently updating multiple models and adapting to new models. To address this, we propose OnceEdit, a novel ensemble-based approach that employs a plug-in model as the editing module, enabling stable knowledge updates across multiple models. Building on the model ensemble, OnceEdit introduces two key mechanisms to enhance its effectiveness. First, we introduce a dynamic weight mechanism through a \weight token for distinguishing between edit-related and non-edit-related instances, ensuring the appropriate utilization of knowledge from integrated models. Second, we incorporate an ensemble enhancement mechanism to mitigate the excessive reliance on the central model inherent in the model ensemble technique, making it more suitable for knowledge editing. Extensive experiments on diverse LLMs demonstrate that OnceEdit consistently outperforms existing methods while achieving superior editing efficiency. Further analysis confirms its adaptability and stability in multi-model editing scenarios. Our code will be available. 

---
# From Plain Text to Poetic Form: Generating Metrically-Constrained Sanskrit Verses 

**Authors**: Manoj Balaji Jagadeeshan, Samarth Bhatia, Pretam Ray, Harshul Raj Surana, Akhil Rajeev P, Priya Mishra, Annarao Kulkarni, Ganesh Ramakrishnan, Prathosh AP, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00815)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved natural language generation, including creative tasks like poetry composition. However, most progress remains concentrated in high-resource languages. This raises an important question: Can LLMs be adapted for structured poetic generation in a low-resource, morphologically rich language such as Sanskrit? In this work, we introduce a dataset designed for translating English prose into structured Sanskrit verse, with strict adherence to classical metrical patterns, particularly the Anushtub meter. We evaluate a range of generative models-both open-source and proprietary-under multiple settings. Specifically, we explore constrained decoding strategies and instruction-based fine-tuning tailored to metrical and semantic fidelity. Our decoding approach achieves over 99% accuracy in producing syntactically valid poetic forms, substantially outperforming general-purpose models in meter conformity. Meanwhile, instruction-tuned variants show improved alignment with source meaning and poetic style, as supported by human assessments, albeit with marginal trade-offs in metrical precision. 

---
# GuessBench: Sensemaking Multimodal Creativity in the Wild 

**Authors**: Zifeng Zhu, Shangbin Feng, Herun Wan, Ningnan Wang, Minnan Luo, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.00814)  

**Abstract**: We propose GuessBench, a novel benchmark that evaluates Vision Language Models (VLMs) on modeling the pervasive, noisy, and pluralistic human creativity. GuessBench sources data from "Guess the Build", an online multiplayer Minecraft minigame where one player constructs a Minecraft build given a concept (e.g. caterpillar) and others try to guess it with natural language hints, presenting a pristine testbed for sensemaking creativity in the wild with VLMs acting as guessers. We curate 1500 images from the actual gameplay and design 2000 problems spanning static and dynamic image settings, natural language hints of varying completeness, and more. Extensive experiments with six open/API VLMs and five reasoning enhancement approaches demonstrate that GuessBench presents a uniquely challenging task in creativity modeling: even the start-of-the-art GPT-4o is incorrect on 34% of instances, while we observe a huge performance gap (13.87% vs. 53.93% on average) between open and API models. When used as a resource to improve VLMs, fine-tuning on the reasoning traces for GuessBench problems improves visual perception tasks by 15.36% on average. Further analysis reveals that VLM performance in creativity sensemaking correlates with the frequency of the concept in training data, while the accuracy drops sharply for concepts in underrepresented cultural contexts and low-resource languages. 

---
# Fast or Slow? Integrating Fast Intuition and Deliberate Thinking for Enhancing Visual Question Answering 

**Authors**: Songtao Jiang, Chenyi Zhou, Yan Zhang, Yeying Jin, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00806)  

**Abstract**: Multimodal large language models (MLLMs) still struggle with complex reasoning tasks in Visual Question Answering (VQA). While current methods have advanced by incorporating visual prompts, our study uncovers critical limitations: these approaches indiscriminately annotate all detected objects for every visual question, generating excessive visual markers that degrade task performance. This issue stems primarily from a lack of focus on key visual elements, raising two important questions: Are all objects equally important, and do all questions require visual prompts? Motivated by Dual Process Theory, which distinguishes between instinctive and deliberate cognitive modes in human reasoning, we propose FOCUS, a plug-and-play approach that dynamically adapts to the complexity of questions, combining fast intuitive judgments with deliberate analytical reasoning to enhance the vision-language reasoning capability of the MLLM. For straightforward questions, FOCUS supports efficient zero-shot reasoning. For more complex tasks, it employs the conceptualizing before observation strategy to highlight critical elements. Extensive experiments on four benchmarks, ScienceQA, TextQA, VizWiz, and MME, demonstrate that FOCUS consistently improves the performance of both open-source and black-box MLLMs, achieving significant gains across all datasets. Ablation studies further validate the importance of combining diverse cognitive strategies with refined visual information for superior performance. Code will be released. 

---
# RARE: Retrieval-Aware Robustness Evaluation for Retrieval-Augmented Generation Systems 

**Authors**: Yixiao Zeng, Tianyu Cao, Danqing Wang, Xinran Zhao, Zimeng Qiu, Morteza Ziyadi, Tongshuang Wu, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00789)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances recency and factuality in answers. However, existing evaluations rarely test how well these systems cope with real-world noise, conflicting between internal and external retrieved contexts, or fast-changing facts. We introduce Retrieval-Aware Robustness Evaluation (RARE), a unified framework and large-scale benchmark that jointly stress-tests query and document perturbations over dynamic, time-sensitive corpora. One of the central features of RARE is a knowledge-graph-driven synthesis pipeline (RARE-Get) that automatically extracts single and multi-hop relations from the customized corpus and generates multi-level question sets without manual intervention. Leveraging this pipeline, we construct a dataset (RARE-Set) spanning 400 expert-level time-sensitive finance, economics, and policy documents and 48,322 questions whose distribution evolves as the underlying sources change. To quantify resilience, we formalize retrieval-conditioned robustness metrics (RARE-Met) that capture a model's ability to remain correct or recover when queries, documents, or real-world retrieval results are systematically altered. Our results show that RAG systems exhibit surprising vulnerability to perturbations, with document robustness consistently being the weakest point regardless of generator size or architecture. RAG systems consistently show lower robustness on multi-hop queries than single-hop queries across all domains. 

---
# Research Borderlands: Analysing Writing Across Research Cultures 

**Authors**: Shaily Bhatt, Tal August, Maria Antoniak  

**Link**: [PDF](https://arxiv.org/pdf/2506.00784)  

**Abstract**: Improving cultural competence of language technologies is important. However most recent works rarely engage with the communities they study, and instead rely on synthetic setups and imperfect proxies of culture. In this work, we take a human-centered approach to discover and measure language-based cultural norms, and cultural competence of LLMs. We focus on a single kind of culture, research cultures, and a single task, adapting writing across research cultures. Through a set of interviews with interdisciplinary researchers, who are experts at moving between cultures, we create a framework of structural, stylistic, rhetorical, and citational norms that vary across research cultures. We operationalise these features with a suite of computational metrics and use them for (a) surfacing latent cultural norms in human-written research papers at scale; and (b) highlighting the lack of cultural competence of LLMs, and their tendency to homogenise writing. Overall, our work illustrates the efficacy of a human-centered approach to measuring cultural norms in human-written and LLM-generated texts. 

---
# KG-TRACES: Enhancing Large Language Models with Knowledge Graph-constrained Trajectory Reasoning and Attribution Supervision 

**Authors**: Rong Wu, Pinlong Cai, Jianbiao Mei, Licheng Wen, Tao Hu, Xuemeng Yang, Daocheng Fu, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00783)  

**Abstract**: Large language models (LLMs) have made remarkable strides in various natural language processing tasks, but their performance on complex reasoning problems remains hindered by a lack of explainability and trustworthiness. This issue, often manifesting as hallucinations or unattributable reasoning processes, limits their applicability in complex reasoning scenarios. To address this, we propose Knowledge Graph-constrained Trajectory Reasoning Attribution and Chain Explanation Supervision (KG-TRACES), a novel framework that enhances the reasoning ability of LLMs through explicit supervision over reasoning paths and processes. KG-TRACES jointly supervises the model to: (1) predict symbolic relation paths, (2) predict full triple-level reasoning paths, and (3) generate attribution-aware reasoning processes grounded in the reasoning paths. At inference phase, the model adapts to both KG-available and KG-unavailable scenarios, retrieving reasoning paths from a KG when possible or predicting plausible reasoning paths with only intrinsic knowledge when not. This design enables the model to reason in an explainable and source-attributable pattern. Through extensive experiments on complex reasoning tasks, we demonstrate that KG-TRACES significantly outperforms existing SOTA: it improves Hits@1 by 1.6% and F1 by 4.7% on WebQSP, and achieves improvements of 4.8% in Hits@1 and 2.1% in F1 on CWQ. Moreover, we show its transferability to specialized domains such as medicine. By visualizing the intermediate steps of reasoning processes, we further show that the explicit supervision introduced by KG-TRACES leads to more stable and goal-directed reasoning processes, aligning closely with correct answers. Code is available at this https URL. 

---
# Improving Automatic Evaluation of Large Language Models (LLMs) in Biomedical Relation Extraction via LLMs-as-the-Judge 

**Authors**: Md Tahmid Rahman Laskar, Israt Jahan, Elham Dolatabadi, Chun Peng, Enamul Hoque, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00777)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance in biomedical relation extraction, even in zero-shot scenarios. However, evaluating LLMs in this task remains challenging due to their ability to generate human-like text, often producing synonyms or abbreviations of gold-standard answers, making traditional automatic evaluation metrics unreliable. On the other hand, while human evaluation is more reliable, it is costly and time-consuming, making it impractical for real-world applications. This paper investigates the use of LLMs-as-the-Judge as an alternative evaluation method for biomedical relation extraction. We benchmark 8 LLMs as judges to evaluate the responses generated by 5 other LLMs across 3 biomedical relation extraction datasets. Unlike other text-generation tasks, we observe that LLM-based judges perform quite poorly (usually below 50% accuracy) in the biomedical relation extraction task. Our findings reveal that it happens mainly because relations extracted by LLMs do not adhere to any standard format. To address this, we propose structured output formatting for LLM-generated responses that helps LLM-Judges to improve their performance by about 15% (on average). We also introduce a domain adaptation technique to further enhance LLM-Judge performance by effectively transferring knowledge between datasets. We release both our human-annotated and LLM-annotated judgment data (36k samples in total) for public use here: this https URL. 

---
# Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models 

**Authors**: Boheng Sheng, Jiacheng Yao, Meicong Zhang, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2506.00773)  

**Abstract**: Large language models (LLMs) often struggle to accurately read and comprehend extremely long texts. Current methods for improvement typically rely on splitting long contexts into fixed-length chunks. However, fixed truncation risks separating semantically relevant content, leading to ambiguity and compromising accurate understanding. To overcome this limitation, we propose a straightforward approach for dynamically separating and selecting chunks of long context, facilitating a more streamlined input for LLMs. In particular, we compute semantic similarities between adjacent sentences, using lower similarities to adaptively divide long contexts into variable-length chunks. We further train a question-aware classifier to select sensitive chunks that are critical for answering specific questions. Experimental results on both single-hop and multi-hop question-answering benchmarks show that the proposed approach consistently outperforms strong baselines. Notably, it maintains robustness across a wide range of input lengths, handling sequences of up to 256k tokens. Our datasets and code are available at the following link: this https URL 

---
# Understanding and Mitigating Cross-lingual Privacy Leakage via Language-specific and Universal Privacy Neurons 

**Authors**: Wenshuo Dong, Qingsong Yang, Shu Yang, Lijie Hu, Meng Ding, Wanyu Lin, Tianhang Zheng, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00759)  

**Abstract**: Large Language Models (LLMs) trained on massive data capture rich information embedded in the training data. However, this also introduces the risk of privacy leakage, particularly involving personally identifiable information (PII). Although previous studies have shown that this risk can be mitigated through methods such as privacy neurons, they all assume that both the (sensitive) training data and user queries are in English. We show that they cannot defend against the privacy leakage in cross-lingual contexts: even if the training data is exclusively in one language, these (private) models may still reveal private information when queried in another language. In this work, we first investigate the information flow of cross-lingual privacy leakage to give a better understanding. We find that LLMs process private information in the middle layers, where representations are largely shared across languages. The risk of leakage peaks when converted to a language-specific space in later layers. Based on this, we identify privacy-universal neurons and language-specific privacy neurons. Privacy-universal neurons influence privacy leakage across all languages, while language-specific privacy neurons are only related to specific languages. By deactivating these neurons, the cross-lingual privacy leakage risk is reduced by 23.3%-31.6%. 

---
# Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations 

**Authors**: Pardis Sadat Zahraei, Ali Emami  

**Link**: [PDF](https://arxiv.org/pdf/2506.00748)  

**Abstract**: Addressing gender bias and maintaining logical coherence in machine translation remains challenging, particularly when translating between natural gender languages, like English, and genderless languages, such as Persian, Indonesian, and Finnish. We introduce the Translate-with-Care (TWC) dataset, comprising 3,950 challenging scenarios across six low- to mid-resource languages, to assess translation systems' performance. Our analysis of diverse technologies, including GPT-4, mBART-50, NLLB-200, and Google Translate, reveals a universal struggle in translating genderless content, resulting in gender stereotyping and reasoning errors. All models preferred masculine pronouns when gender stereotypes could influence choices. Google Translate and GPT-4 showed particularly strong bias, favoring male pronouns 4-6 times more than feminine ones in leadership and professional success contexts. Fine-tuning mBART-50 on TWC substantially resolved these biases and errors, led to strong generalization, and surpassed proprietary LLMs while remaining open-source. This work emphasizes the need for targeted approaches to gender and semantic coherence in machine translation, particularly for genderless languages, contributing to more equitable and accurate translation systems. 

---
# Assortment of Attention Heads: Accelerating Federated PEFT with Head Pruning and Strategic Client Selection 

**Authors**: Yeshwanth Venkatesha, Souvik Kundu, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2506.00743)  

**Abstract**: Parameter Efficient Fine-Tuning (PEFT) has become the de-facto approach in adapting Large Language Models (LLMs) for downstream tasks in Natural Language Processing. However, its adoption in privacy-preserving distributed learning frameworks, such as Federated Learning (FL), remains relatively limited. This is mainly due to challenges specific to FL, such as resource-constrained devices and diverse data distributions among clients. In this paper, we propose an efficient method to perform PEFT within the FL framework for Multi-Head Attention (MHA) based language models. We address the challenges through head pruning, a novel head-specific weighted aggregation mechanism, and a client selection strategy. Head pruning minimizes training complexity within the clients, guided by the importance score computed based on the confidence of the attention head. Weighted aggregation of heads ensures the global model captures crucial updates from diverse clients complementing our client selection strategy. We show results on the MultiNLI benchmark along with 20 Newsgroups, XL-Sum, and E2E NLG datasets. We use the MultiNLI dataset and T5-small model with LoRA as our PEFT method, attaining sparsity levels of up to 90%, resulting in a communication advantage of up to 1.8x and a reduction in training OPs of 3.9x while maintaining the accuracy drop under 2%. 

---
# Data Swarms: Optimizable Generation of Synthetic Evaluation Data 

**Authors**: Shangbin Feng, Yike Wang, Weijia Shi, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.00741)  

**Abstract**: We propose Data Swarms, an algorithm to optimize the generation of synthetic evaluation data and advance quantitative desiderata of LLM evaluation. We first train a swarm of initial data generators using existing data, and define various evaluation objectives to reflect the desired properties of evaluation (e.g., generate more difficult problems for the evaluated models) and quantitatively evaluate data generators. We then employ particle swarm optimization to optimize the swarm of data generators, where they collaboratively search through the model parameter space to find new generators that advance these objectives. We further extend it to Adversarial Swarms, where the data generator swarm generates harder data while the test taker model swarm learns from such data, co-evolving dynamically for better data and models simultaneously. Extensive experiments demonstrate that Data Swarms outperforms eight data generation baselines across five evaluation objectives, while Adversarial Swarms produce more robust learning of synthetic data and stronger generalization. Further analysis reveals that Data Swarms successfully optimizes compositions of multiple evaluation objectives and generalizes to new off-the-shelf LLMs, unseen at optimization time. 

---
# Length Aware Speech Translation for Video Dubbing 

**Authors**: Harveen Singh Chadha, Aswin Shanmugam Subramanian, Vikas Joshi, Shubham Bansal, Jian Xue, Rupeshkumar Mehta, Jinyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00740)  

**Abstract**: In video dubbing, aligning translated audio with the source audio is a significant challenge. Our focus is on achieving this efficiently, tailored for real-time, on-device video dubbing scenarios. We developed a phoneme-based end-to-end length-sensitive speech translation (LSST) model, which generates translations of varying lengths short, normal, and long using predefined tags. Additionally, we introduced length-aware beam search (LABS), an efficient approach to generate translations of different lengths in a single decoding pass. This approach maintained comparable BLEU scores compared to a baseline without length awareness while significantly enhancing synchronization quality between source and target audio, achieving a mean opinion score (MOS) gain of 0.34 for Spanish and 0.65 for Korean, respectively. 

---
# DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments 

**Authors**: Chiyu Zhang, Marc-Alexandre Cote, Michael Albada, Anush Sankaran, Jack W. Stokes, Tong Wang, Amir Abdi, William Blum, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2506.00739)  

**Abstract**: Large language model (LLM) agents have shown impressive capabilities in human language comprehension and reasoning, yet their potential in cybersecurity remains underexplored. We introduce DefenderBench, a practical, open-source toolkit for evaluating language agents across offense, defense, and cybersecurity knowledge-based tasks. DefenderBench includes environments for network intrusion, malicious content detection, code vulnerability analysis, and cybersecurity knowledge assessment. It is intentionally designed to be affordable and easily accessible for researchers while providing fair and rigorous assessment. We benchmark several state-of-the-art (SoTA) and popular LLMs, including both open- and closed-weight models, using a standardized agentic framework. Our results show that Claude-3.7-sonnet performs best with a DefenderBench score of 81.65, followed by Claude-3.7-sonnet-think with 78.40, while the best open-weight model, Llama 3.3 70B, is not far behind with a DefenderBench score of 71.81. DefenderBench's modular design allows seamless integration of custom LLMs and tasks, promoting reproducibility and fair comparisons. An anonymized version of DefenderBench is available at this https URL. 

---
# Narrative Media Framing in Political Discourse 

**Authors**: Yulia Otmakhova, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2506.00737)  

**Abstract**: Narrative frames are a powerful way of conceptualizing and communicating complex, controversial ideas, however automated frame analysis to date has mostly overlooked this framing device. In this paper, we connect elements of narrativity with fundamental aspects of framing, and present a framework which formalizes and operationalizes such aspects. We annotate and release a data set of news articles in the climate change domain, analyze the dominance of narrative frame components across political leanings, and test LLMs in their ability to predict narrative frames and their components. Finally, we apply our framework in an unsupervised way to elicit components of narrative framing in a second domain, the COVID-19 crisis, where our predictions are congruent with prior theoretical work showing the generalizability of our approach. 

---
# Structured Gradient Guidance for Few-Shot Adaptation in Large Language Models 

**Authors**: Hongye Zheng, Yichen Wang, Ray Pan, Guiran Liu, Binrong Zhu, Hanlu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00726)  

**Abstract**: This paper presents a gradient-informed fine-tuning method for large language models under few-shot conditions. The goal is to enhance task adaptability and training stability when data is limited. The method builds on a base loss function and introduces two gradient-related regularization terms. The first enforces gradient direction consistency to guide parameter updates along task-relevant directions and prevent drift. The second controls gradient magnitude to avoid abnormal updates. Together, these components support a more efficient and stable optimization path. To further improve cross-task generalization, the method incorporates a gradient alignment mechanism. This mechanism measures the consistency between optimization directions of the source and target tasks. It enhances fine-tuning performance in multi-task and cross-domain scenarios. Across various natural language understanding tasks, the method outperforms existing fine-tuning strategies in average accuracy, gradient stability, and directional alignment. Empirical evaluations under different sample sizes and domain-specific tasks confirm the method's robustness and broad applicability in low-resource environments. In particular, the method shows clear advantages in controlling parameter update paths. The results demonstrate that a gradient-based fine-tuning framework can effectively leverage the representational power of large language models. It ensures training stability while reducing dependence on large volumes of labeled data. 

---
# Chain-of-Thought Training for Open E2E Spoken Dialogue Systems 

**Authors**: Siddhant Arora, Jinchuan Tian, Hayato Futami, Jee-weon Jung, Jiatong Shi, Yosuke Kashiwagi, Emiru Tsunoo, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00722)  

**Abstract**: Unlike traditional cascaded pipelines, end-to-end (E2E) spoken dialogue systems preserve full differentiability and capture non-phonemic information, making them well-suited for modeling spoken interactions. However, existing E2E approaches often require large-scale training data and generates responses lacking semantic coherence. We propose a simple yet effective strategy leveraging a chain-of-thought (CoT) formulation, ensuring that training on conversational data remains closely aligned with the multimodal language model (LM)'s pre-training on speech recognition~(ASR), text-to-speech synthesis (TTS), and text LM tasks. Our method achieves over 1.5 ROUGE-1 improvement over the baseline, successfully training spoken dialogue systems on publicly available human-human conversation datasets, while being compute-efficient enough to train on just 300 hours of public human-human conversation data, such as the Switchboard. We will publicly release our models and training code. 

---
# From Argumentative Text to Argument Knowledge Graph: A New Framework for Structured Argumentation 

**Authors**: Debarati Bhattacharjee, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2506.00713)  

**Abstract**: This paper presents a framework to convert argumentative texts into argument knowledge graphs (AKG). Starting with basic annotations of argumentative components (ACs) and argumentative relations (ARs), we enrich the information by constructing a knowledge base (KB) graph with metadata attributes for nodes. Next, we use premises and inference rules from the KB to form arguments by applying modus ponens. From these arguments, we create an AKG. The nodes and edges of the AKG have attributes that capture important argumentative features. We also find missing inference rules by identifying markers. This makes it possible to identify undercut attacks that were previously undetectable in existing datasets. The AKG gives a graphical view of the argumentative structure that is easier to understand than theoretical formats. It also prepares the ground for future reasoning tasks, including checking the coherence of arguments and identifying opportunities for revision. For this, it is important to find indirect relations, many of which are implicit. Our proposed AKG format, with annotated inference rules and modus ponens, will help reasoning models learn the implicit indirect relations that require inference over arguments and the relations between them. 

---
# Measuring Faithfulness and Abstention: An Automated Pipeline for Evaluating LLM-Generated 3-ply Case-Based Legal Arguments 

**Authors**: Li Zhang, Morgan Gray, Jaromir Savelka, Kevin D. Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2506.00694)  

**Abstract**: Large Language Models (LLMs) demonstrate potential in complex legal tasks like argument generation, yet their reliability remains a concern. Building upon pilot work assessing LLM generation of 3-ply legal arguments using human evaluation, this paper introduces an automated pipeline to evaluate LLM performance on this task, specifically focusing on faithfulness (absence of hallucination), factor utilization, and appropriate abstention. We define hallucination as the generation of factors not present in the input case materials and abstention as the model's ability to refrain from generating arguments when instructed and no factual basis exists. Our automated method employs an external LLM to extract factors from generated arguments and compares them against the ground-truth factors provided in the input case triples (current case and two precedent cases). We evaluated eight distinct LLMs on three tests of increasing difficulty: 1) generating a standard 3-ply argument, 2) generating an argument with swapped precedent roles, and 3) recognizing the impossibility of argument generation due to lack of shared factors and abstaining. Our findings indicate that while current LLMs achieve high accuracy (over 90%) in avoiding hallucination on viable argument generation tests (Tests 1 & 2), they often fail to utilize the full set of relevant factors present in the cases. Critically, on the abstention test (Test 3), most models failed to follow instructions to stop, instead generating spurious arguments despite the lack of common factors. This automated pipeline provides a scalable method for assessing these crucial LLM behaviors, highlighting the need for improvements in factor utilization and robust abstention capabilities before reliable deployment in legal settings. Project page: this https URL. 

---
# DeepRAG: Integrating Hierarchical Reasoning and Process Supervision for Biomedical Multi-Hop QA 

**Authors**: Yuelyu Ji, Hang Zhang, Shiven Verma, Hui Ji, Chun Li, Yushui Han, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00671)  

**Abstract**: We propose DeepRAG, a novel framework that integrates DeepSeek hierarchical question decomposition capabilities with RAG Gym unified retrieval-augmented generation optimization using process level supervision. Targeting the challenging MedHopQA biomedical question answering task, DeepRAG systematically decomposes complex queries into precise sub-queries and employs concept level reward signals informed by the UMLS ontology to enhance biomedical accuracy. Preliminary evaluations on the MedHopQA dataset indicate that DeepRAG significantly outperforms baseline models, including standalone DeepSeek and RAG Gym, achieving notable improvements in both Exact Match and concept level accuracy. 

---
# SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues 

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Louis DiValentin, Amin Hass, Benjamin F Morris, Isaac Jacobson, Randolph Linderman, James Kiessling, Nicolas Ramos, Bhavna Gopal, Maziyar Baran Pouyan, Changwei Liu, Hai Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00668)  

**Abstract**: Malicious attackers can exploit large language models (LLMs) by engaging them in multi-turn dialogues to achieve harmful objectives, posing significant safety risks to society. To address this challenge, we propose a novel defense mechanism: SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues (STREAM). STREAM defends LLMs against multi-turn attacks while preserving their functional capabilities. Our approach involves constructing a human-annotated dataset, the Safety Reasoning Multi-turn Dialogues dataset, which is used to fine-tune a plug-and-play safety reasoning moderator. This model is designed to identify malicious intent hidden within multi-turn conversations and alert the target LLM of potential risks. We evaluate STREAM across multiple LLMs against prevalent multi-turn attack strategies. Experimental results demonstrate that our method significantly outperforms existing defense techniques, reducing the Attack Success Rate (ASR) by 51.2%, all while maintaining comparable LLM capability. 

---
# Sarc7: Evaluating Sarcasm Detection and Generation with Seven Types and Emotion-Informed Techniques 

**Authors**: Lang Xiong, Raina Gao, Alyssa Jeong, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00658)  

**Abstract**: Sarcasm is a form of humor where expressions convey meanings opposite to their literal interpretations. Classifying and generating sarcasm using large language models is vital for interpreting human communication. Sarcasm poses challenges for computational models, due to its nuanced nature. We introduce Sarc7, a benchmark that classifies 7 types of sarcasm: self-deprecating, brooding, deadpan, polite, obnoxious, raging, and manic by annotating entries of the MUStARD dataset. Classification was evaluated using zero-shot, few-shot, chain-of-thought (CoT), and a novel emotion-based prompting technique. We propose an emotion-based generation method developed by identifying key components of sarcasm-incongruity, shock value, and context dependency. Our classification experiments show that Gemini 2.5, using emotion-based prompting, outperforms other setups with an F1 score of 0.3664. Human evaluators preferred our emotion-based prompting, with 38.46% more successful generations than zero-shot prompting. 

---
# GuideX: Guided Synthetic Data Generation for Zero-Shot Information Extraction 

**Authors**: Neil De La Fuente, Oscar Sainz, Iker García-Ferrero, Eneko Agirre  

**Link**: [PDF](https://arxiv.org/pdf/2506.00649)  

**Abstract**: Information Extraction (IE) systems are traditionally domain-specific, requiring costly adaptation that involves expert schema design, data annotation, and model training. While Large Language Models have shown promise in zero-shot IE, performance degrades significantly in unseen domains where label definitions differ. This paper introduces GUIDEX, a novel method that automatically defines domain-specific schemas, infers guidelines, and generates synthetically labeled instances, allowing for better out-of-domain generalization. Fine-tuning Llama 3.1 with GUIDEX sets a new state-of-the-art across seven zeroshot Named Entity Recognition benchmarks. Models trained with GUIDEX gain up to 7 F1 points over previous methods without humanlabeled data, and nearly 2 F1 points higher when combined with it. Models trained on GUIDEX demonstrate enhanced comprehension of complex, domain-specific annotation schemas. Code, models, and synthetic datasets are available at this http URL 

---
# Clinical Annotations for Automatic Stuttering Severity Assessment 

**Authors**: Ana Rita Valente, Rufael Marew, Hawau Olamide Toyin, Hamdan Al-Ali, Anelise Bohnen, Inma Becerra, Elsa Marta Soares, Goncalo Leal, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2506.00644)  

**Abstract**: Stuttering is a complex disorder that requires specialized expertise for effective assessment and treatment. This paper presents an effort to enhance the FluencyBank dataset with a new stuttering annotation scheme based on established clinical standards. To achieve high-quality annotations, we hired expert clinicians to label the data, ensuring that the resulting annotations mirror real-world clinical expertise. The annotations are multi-modal, incorporating audiovisual features for the detection and classification of stuttering moments, secondary behaviors, and tension scores. In addition to individual annotations, we additionally provide a test set with highly reliable annotations based on expert consensus for assessing individual annotators and machine learning models. Our experiments and analysis illustrate the complexity of this task that necessitates extensive clinical expertise for valid training and evaluation of stuttering assessment models. 

---
# SATA-BENCH: Select All That Apply Benchmark for Multiple Choice Questions 

**Authors**: Weijie Xu, Shixian Cui, Xi Fang, Chi Xue, Stephanie Eckman, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2506.00643)  

**Abstract**: Large language models (LLMs) are increasingly evaluated on single-answer multiple-choice tasks, yet many real-world problems require identifying all correct answers from a set of options. This capability remains underexplored. We introduce SATA-BENCH, the first dedicated benchmark for evaluating LLMs on Select All That Apply (SATA) questions across diverse domains, including reading comprehension, law, and biomedicine. Our evaluation of 27 open-source and proprietary models reveals a significant gap: even the strongest model achieves only 41.8% exact match, exposing LLMs' inability to reliably identify all correct answers. We find that this weakness stems from two core challenges: selection bias - models favor certain choices regardless of content, and count bias - models fail to predict the correct number of answers. To address these issues, we propose Choice Funnel, a decoding strategy that combines token debiasing with adaptive thresholding to guide models toward complete and accurate selections. Choice Funnel achieves up to 29% higher exact match than competitive baselines while reducing inference cost by over 64%. Our findings expose fundamental limitations in current LLMs and introduce a new framework for diagnosing and improving multi-answer reasoning. We release SATA-BENCH and Choice Funnel to promote LLM development for robust decision-making in realistic, multi-answer applications. 

---
# Improving the Calibration of Confidence Scores in Text Generation Using the Output Distribution's Characteristics 

**Authors**: Lorenzo Jaime Yu Flores, Ori Ernst, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2506.00637)  

**Abstract**: Well-calibrated model confidence scores can improve the usefulness of text generation models. For example, users can be prompted to review predictions with low confidence scores, to prevent models from returning bad or potentially dangerous predictions. However, confidence metrics are not always well calibrated in text generation. One reason is that in generation, there can be many valid answers, which previous methods do not always account for. Hence, a confident model could distribute its output probability among multiple sequences because they are all valid. We propose task-agnostic confidence metrics suited to generation, which rely solely on the probabilities associated with the model outputs without the need for further fine-tuning or heuristics. Using these, we are able to improve the calibration of BART and Flan-T5 on summarization, translation, and QA datasets. 

---
# ViToSA: Audio-Based Toxic Spans Detection on Vietnamese Speech Utterances 

**Authors**: Huy Ba Do, Vy Le-Phuong Huynh, Luan Thanh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00636)  

**Abstract**: Toxic speech on online platforms is a growing concern, impacting user experience and online safety. While text-based toxicity detection is well-studied, audio-based approaches remain underexplored, especially for low-resource languages like Vietnamese. This paper introduces ViToSA (Vietnamese Toxic Spans Audio), the first dataset for toxic spans detection in Vietnamese speech, comprising 11,000 audio samples (25 hours) with accurate human-annotated transcripts. We propose a pipeline that combines ASR and toxic spans detection for fine-grained identification of toxic content. Our experiments show that fine-tuning ASR models on ViToSA significantly reduces WER when transcribing toxic speech, while the text-based toxic spans detection (TSD) models outperform existing baselines. These findings establish a novel benchmark for Vietnamese audio-based toxic spans detection, paving the way for future research in speech content moderation. 

---
# Social Construction of Urban Space: Understanding Neighborhood Boundaries Using Rental Listings 

**Authors**: Adam Visokay, Ruth Bagley, Ian Kennedy, Chris Hess, Kyle Crowder, Rob Voigt, Denis Peskoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.00634)  

**Abstract**: Rental listings offer a unique window into how urban space is socially constructed through language. We analyze Chicago Craigslist rental advertisements from 2018 to 2024 to examine how listing agents characterize neighborhoods, identifying mismatches between institutional boundaries and neighborhood claims. Through manual and large language model annotation, we classify unstructured listings from Craigslist according to their neighborhood. Geospatial analysis reveals three distinct patterns: properties with conflicting neighborhood designations due to competing spatial definitions, border properties with valid claims to adjacent neighborhoods, and ``reputation laundering" where listings claim association with distant, desirable neighborhoods. Through topic modeling, we identify patterns that correlate with spatial positioning: listings further from neighborhood centers emphasize different amenities than centrally-located units. Our findings demonstrate that natural language processing techniques can reveal how definitions of urban spaces are contested in ways that traditional methods overlook. 

---
# LID Models are Actually Accent Classifiers: Implications and Solutions for LID on Accented Speech 

**Authors**: Niyati Bafna, Matthew Wiesner  

**Link**: [PDF](https://arxiv.org/pdf/2506.00628)  

**Abstract**: Prior research indicates that LID model performance significantly declines on accented speech; however, the specific causes, extent, and characterization of these errors remain under-explored. (i) We identify a common failure mode on accented speech whereby LID systems often misclassify L2 accented speech as the speaker's native language or a related language. (ii) We present evidence suggesting that state-of-the-art models are invariant to permutations of short spans of speech, implying they classify on the basis of short phonotactic features indicative of accent rather than language. Our analysis reveals a simple method to enhance model robustness to accents through input chunking. (iii) We present an approach that integrates sequence-level information into our model without relying on monolingual ASR systems; this reduces accent-language confusion and significantly enhances performance on accented speech while maintaining comparable results on standard LID. 

---
# Improving Dialogue State Tracking through Combinatorial Search for In-Context Examples 

**Authors**: Haesung Pyun, Yoonah Park, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00622)  

**Abstract**: In dialogue state tracking (DST), in-context learning comprises a retriever that selects labeled dialogues as in-context examples and a DST model that uses these examples to infer the dialogue state of the query dialogue. Existing methods for constructing training data for retrievers suffer from three key limitations: (1) the synergistic effect of examples is not considered, (2) the linguistic characteristics of the query are not sufficiently factored in, and (3) scoring is not directly optimized for DST performance. Consequently, the retriever can fail to retrieve examples that would substantially improve DST performance. To address these issues, we present CombiSearch, a method that scores effective in-context examples based on their combinatorial impact on DST performance. Our evaluation on MultiWOZ shows that retrievers trained with CombiSearch surpass state-of-the-art models, achieving a 20x gain in data efficiency and generalizing well to the SGD dataset. Moreover, CombiSearch attains a 12% absolute improvement in the upper bound DST performance over traditional approaches when no retrieval errors are assumed. This significantly increases the headroom for practical DST performance while demonstrating that existing methods rely on suboptimal data for retriever training. 

---
# Enhancing Clinical Multiple-Choice Questions Benchmarks with Knowledge Graph Guided Distractor Generation 

**Authors**: Running Yang, Wenlong Deng, Minghui Chen, Yuyin Zhou, Xiaoxiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00612)  

**Abstract**: Clinical tasks such as diagnosis and treatment require strong decision-making abilities, highlighting the importance of rigorous evaluation benchmarks to assess the reliability of large language models (LLMs). In this work, we introduce a knowledge-guided data augmentation framework that enhances the difficulty of clinical multiple-choice question (MCQ) datasets by generating distractors (i.e., incorrect choices that are similar to the correct one and may confuse existing LLMs). Using our KG-based pipeline, the generated choices are both clinically plausible and deliberately misleading. Our approach involves multi-step, semantically informed walks on a medical knowledge graph to identify distractor paths-associations that are medically relevant but factually incorrect-which then guide the LLM in crafting more deceptive distractors. We apply the designed knowledge graph guided distractor generation (KGGDG) pipline, to six widely used medical QA benchmarks and show that it consistently reduces the accuracy of state-of-the-art LLMs. These findings establish KGGDG as a powerful tool for enabling more robust and diagnostic evaluations of medical LLMs. 

---
# PAKTON: A Multi-Agent Framework for Question Answering in Long Legal Agreements 

**Authors**: Petros Raptopoulos, Giorgos Filandrianos, Maria Lymperaiou, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00608)  

**Abstract**: Contract review is a complex and time-intensive task that typically demands specialized legal expertise, rendering it largely inaccessible to non-experts. Moreover, legal interpretation is rarely straightforward-ambiguity is pervasive, and judgments often hinge on subjective assessments. Compounding these challenges, contracts are usually confidential, restricting their use with proprietary models and necessitating reliance on open-source alternatives. To address these challenges, we introduce PAKTON: a fully open-source, end-to-end, multi-agent framework with plug-and-play capabilities. PAKTON is designed to handle the complexities of contract analysis through collaborative agent workflows and a novel retrieval-augmented generation (RAG) component, enabling automated legal document review that is more accessible, adaptable, and privacy-preserving. Experiments demonstrate that PAKTON outperforms both general-purpose and pretrained models in predictive accuracy, retrieval performance, explainability, completeness, and grounded justifications as evaluated through a human study and validated with automated metrics. 

---
# Entriever: Energy-based Retriever for Knowledge-Grounded Dialog Systems 

**Authors**: Yucheng Cai, Ke Li, Yi Huang, Junlan Feng, Zhijian Ou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00585)  

**Abstract**: A retriever, which retrieves relevant knowledge pieces from a knowledge base given a context, is an important component in many natural language processing (NLP) tasks. Retrievers have been introduced in knowledge-grounded dialog systems to improve knowledge acquisition. In knowledge-grounded dialog systems, when conditioning on a given context, there may be multiple relevant and correlated knowledge pieces. However, knowledge pieces are usually assumed to be conditionally independent in current retriever models. To address this issue, we propose Entriever, an energy-based retriever. Entriever directly models the candidate retrieval results as a whole instead of modeling the knowledge pieces separately, with the relevance score defined by an energy function. We explore various architectures of energy functions and different training methods for Entriever, and show that Entriever substantially outperforms the strong cross-encoder baseline in knowledge retrieval tasks. Furthermore, we show that in semi-supervised training of knowledge-grounded dialog systems, Entriever enables effective scoring of retrieved knowledge pieces and significantly improves end-to-end performance of dialog systems. 

---
# The Hidden Language of Harm: Examining the Role of Emojis in Harmful Online Communication and Content Moderation 

**Authors**: Yuhang Zhou, Yimin Xiao, Wei Ai, Ge Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00583)  

**Abstract**: Social media platforms have become central to modern communication, yet they also harbor offensive content that challenges platform safety and inclusivity. While prior research has primarily focused on textual indicators of offense, the role of emojis, ubiquitous visual elements in online discourse, remains underexplored. Emojis, despite being rarely offensive in isolation, can acquire harmful meanings through symbolic associations, sarcasm, and contextual misuse. In this work, we systematically examine emoji contributions to offensive Twitter messages, analyzing their distribution across offense categories and how users exploit emoji ambiguity. To address this, we propose an LLM-powered, multi-step moderation pipeline that selectively replaces harmful emojis while preserving the tweet's semantic intent. Human evaluations confirm our approach effectively reduces perceived offensiveness without sacrificing meaning. Our analysis also reveals heterogeneous effects across offense types, offering nuanced insights for online communication and emoji moderation. 

---
# AnnaAgent: Dynamic Evolution Agent System with Multi-Session Memory for Realistic Seeker Simulation 

**Authors**: Ming Wang, Peidong Wang, Lin Wu, Xiaocui Yang, Daling Wang, Shi Feng, Yuxin Chen, Bixuan Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00551)  

**Abstract**: Constrained by the cost and ethical concerns of involving real seekers in AI-driven mental health, researchers develop LLM-based conversational agents (CAs) with tailored configurations, such as profiles, symptoms, and scenarios, to simulate seekers. While these efforts advance AI in mental health, achieving more realistic seeker simulation remains hindered by two key challenges: dynamic evolution and multi-session memory. Seekers' mental states often fluctuate during counseling, which typically spans multiple sessions. To address this, we propose AnnaAgent, an emotional and cognitive dynamic agent system equipped with tertiary memory. AnnaAgent incorporates an emotion modulator and a complaint elicitor trained on real counseling dialogues, enabling dynamic control of the simulator's configurations. Additionally, its tertiary memory mechanism effectively integrates short-term and long-term memory across sessions. Evaluation results, both automated and manual, demonstrate that AnnaAgent achieves more realistic seeker simulation in psychological counseling compared to existing baselines. The ethically reviewed and screened code can be found on this https URL. 

---
# Towards Multi-dimensional Evaluation of LLM Summarization across Domains and Languages 

**Authors**: Hyangsuk Min, Yuho Lee, Minjeong Ban, Jiaqi Deng, Nicole Hee-Yeon Kim, Taewon Yun, Hang Su, Jason Cai, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.00549)  

**Abstract**: Evaluation frameworks for text summarization have evolved in terms of both domain coverage and metrics. However, existing benchmarks still lack domain-specific assessment criteria, remain predominantly English-centric, and face challenges with human annotation due to the complexity of reasoning. To address these, we introduce MSumBench, which provides a multi-dimensional, multi-domain evaluation of summarization in English and Chinese. It also incorporates specialized assessment criteria for each domain and leverages a multi-agent debate system to enhance annotation quality. By evaluating eight modern summarization models, we discover distinct performance patterns across domains and languages. We further examine large language models as summary evaluators, analyzing the correlation between their evaluation and summarization capabilities, and uncovering systematic bias in their assessment of self-generated summaries. Our benchmark dataset is publicly available at this https URL. 

---
# ARIA: Training Language Agents with Intention-Driven Reward Aggregation 

**Authors**: Ruihan Yang, Yikai Zhang, Aili Chen, Xintao Wang, Siyu Yuan, Jiangjie Chen, Deqing Yang, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00539)  

**Abstract**: Large language models (LLMs) have enabled agents to perform complex reasoning and decision-making through free-form language interactions. However, in open-ended language action environments (e.g., negotiation or question-asking games), the action space can be formulated as a joint distribution over tokens, resulting in an exponentially large action space. Sampling actions in such a space can lead to extreme reward sparsity, which brings large reward variance, hindering effective reinforcement learning (RL). To address this, we propose ARIA, a method that Aggregates Rewards in Intention space to enable efficient and effective language Agents training. ARIA aims to project natural language actions from the high-dimensional joint token distribution space into a low-dimensional intention space, where semantically similar actions are clustered and assigned shared rewards. This intention-aware reward aggregation reduces reward variance by densifying reward signals, fostering better policy optimization. Extensive experiments demonstrate that ARIA not only significantly reduces policy gradient variance, but also delivers substantial performance gains of an average of 9.95% across four downstream tasks, consistently outperforming offline and online RL baselines. 

---
# Decoupling Reasoning and Knowledge Injection for In-Context Knowledge Editing 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00536)  

**Abstract**: Knowledge editing aims to efficiently update Large Language Models (LLMs) by modifying specific knowledge without retraining the entire model. Among knowledge editing approaches, in-context editing (ICE) offers a lightweight solution by injecting new knowledge directly into the input context, leaving model parameters unchanged. However, existing ICE approaches do not explicitly separate the newly injected knowledge from the model's original reasoning process. This entanglement often results in conflicts between external updates and internal parametric knowledge, undermining the consistency and accuracy of the reasoning this http URL this work, we conduct preliminary experiments to examine how parametric knowledge influences reasoning path planning. We find that the model's reasoning is tightly coupled with its internal knowledge, and that naively injecting new information without adapting the reasoning path often leads to performance degradation, particularly in multi-hop tasks. To this end, we propose DecKER, a novel ICE framework that decouples reasoning from knowledge editing by generating a masked reasoning path and then resolving knowledge edits via hybrid retrieval and model-based validation. Experiments on multi-hop QA benchmarks show that DecKER significantly outperforms existing ICE methods by mitigating knowledge conflicts and preserving reasoning consistency. Our code is available at: this https URL . 

---
# Retrieval-Augmented Generation Systems for Intellectual Property via Synthetic Multi-Angle Fine-tuning 

**Authors**: Runtao Ren, Jian Ma, Jianxi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00527)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems in the Intellectual Property (IP) field often struggle with diverse user queries, including colloquial expressions, spelling errors, and ambiguous terminology, leading to inaccurate retrieval and suboptimal responses. To address this challenge, we propose Multi-Angle Question Generation and Retrieval Fine-Tuning Method (MQG-RFM), a novel framework that leverages large language models (LLMs) to simulate varied user inquiries and fine-tunes retrieval models to align semantically equivalent but linguistically diverse questions. Unlike complex architectural modifications, MQG-RFM adopts a lightweight Data-to-Tune paradigm, combining prompt-engineered query generation with hard negative mining to enhance retrieval robustness without costly infrastructure changes. Experimental results on a Taiwan patent Q&A dataset show 185.62% improvement in retrieval accuracy on the Patent Consultation dataset and 262.26% improvement on the Novel Patent Technology Report dataset, with 14.22% and 53.58% improvements in generation quality over the baselines, respectively. By bridging the gap between user intent and system comprehension through semantic-aware retrieval optimization, MQG-RFM offers a practical, scalable approach for rapid, cost-effective deployment among small and medium-sized agencies seeking reliable patent intelligence solutions. Additionally, our proposed method has already been adopted by ScholarMate, the largest professional research social networking platform in China, to support real-world development and deployment. A demo version of the instantiated is available at this https URL. 

---
# CausalAbstain: Enhancing Multilingual LLMs with Causal Reasoning for Trustworthy Abstention 

**Authors**: Yuxi Sun, Aoqi Zuo, Wei Gao, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00519)  

**Abstract**: Large Language Models (LLMs) often exhibit knowledge disparities across languages. Encouraging LLMs to \textit{abstain} when faced with knowledge gaps is a promising strategy to reduce hallucinations in multilingual settings. Current abstention strategies for multilingual scenarios primarily rely on generating feedback in various languages using LLMs and performing self-reflection. However, these methods can be adversely impacted by inaccuracies and biases in the generated feedback. To address this, from a causal perspective, we introduce \textit{CausalAbstain}, a method that helps LLMs determine whether to utilize multiple generated feedback responses and how to identify the most useful ones. Extensive experiments demonstrate that \textit{CausalAbstain} effectively selects helpful feedback and enhances abstention decisions with interpretability in both native language (\textsc{Casual-native}) and multilingual (\textsc{Causal-multi}) settings, outperforming strong baselines on two benchmark datasets covering encyclopedic and commonsense knowledge QA tasks. Our code and data are open-sourced at this https URL. 

---
# Evaluating the Evaluation of Diversity in Commonsense Generation 

**Authors**: Tianhui Zhang, Bei Peng, Danushka Bollegala  

**Link**: [PDF](https://arxiv.org/pdf/2506.00514)  

**Abstract**: In commonsense generation, given a set of input concepts, a model must generate a response that is not only commonsense bearing, but also capturing multiple diverse viewpoints. Numerous evaluation metrics based on form- and content-level overlap have been proposed in prior work for evaluating the diversity of a commonsense generation model. However, it remains unclear as to which metrics are best suited for evaluating the diversity in commonsense generation. To address this gap, we conduct a systematic meta-evaluation of diversity metrics for commonsense generation. We find that form-based diversity metrics tend to consistently overestimate the diversity in sentence sets, where even randomly generated sentences are assigned overly high diversity scores. We then use an Large Language Model (LLM) to create a novel dataset annotated for the diversity of sentences generated for a commonsense generation task, and use it to conduct a meta-evaluation of the existing diversity evaluation metrics. Our experimental results show that content-based diversity evaluation metrics consistently outperform the form-based counterparts, showing high correlations with the LLM-based ratings. We recommend that future work on commonsense generation should use content-based metrics for evaluating the diversity of their outputs. 

---
# Goal-Aware Identification and Rectification of Misinformation in Multi-Agent Systems 

**Authors**: Zherui Li, Yan Mi, Zhenhong Zhou, Houcheng Jiang, Guibin Zhang, Kun Wang, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00509)  

**Abstract**: Large Language Model-based Multi-Agent Systems (MASs) have demonstrated strong advantages in addressing complex real-world tasks. However, due to the introduction of additional attack surfaces, MASs are particularly vulnerable to misinformation injection. To facilitate a deeper understanding of misinformation propagation dynamics within these systems, we introduce MisinfoTask, a novel dataset featuring complex, realistic tasks designed to evaluate MAS robustness against such threats. Building upon this, we propose ARGUS, a two-stage, training-free defense framework leveraging goal-aware reasoning for precise misinformation rectification within information flows. Our experiments demonstrate that in challenging misinformation scenarios, ARGUS exhibits significant efficacy across various injection attacks, achieving an average reduction in misinformation toxicity of approximately 28.17% and improving task success rates under attack by approximately 10.33%. Our code and dataset is available at: this https URL. 

---
# Exploring In-context Example Generation for Machine Translation 

**Authors**: Dohyun Lee, Seungil Chad Lee, Chanwoo Yang, Yujin Baek, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00507)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance across various tasks, leveraging their exceptional in-context learning ability with only a few examples. Accordingly, the selection of optimal in-context examples has been actively studied in the field of machine translation. However, these studies presuppose the presence of a demonstration pool with human-annotated pairs, making them less applicable to low-resource languages where such an assumption is challenging to meet. To overcome this limitation, this paper explores the research direction of in-context example generation for machine translation. Specifically, we propose Demonstration Augmentation for Translation (DAT), a simple yet effective approach that generates example pairs without relying on any external resources. This method builds upon two prior criteria, relevance and diversity, which have been highlighted in previous work as key factors for in-context example selection. Through experiments and analysis on low-resource languages where human-annotated pairs are scarce, we show that DAT achieves superior translation quality compared to the baselines. Furthermore, we investigate the potential of progressively accumulating generated pairs during test time to build and reuse a demonstration pool. Our implementation is publicly available at this https URL. 

---
# Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection 

**Authors**: Shuguo Hu, Jun Hu, Huaiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00488)  

**Abstract**: Large Language Models (LLMs) can assist multimodal fake news detection by predicting pseudo labels. However, LLM-generated pseudo labels alone demonstrate poor performance compared to traditional detection methods, making their effective integration non-trivial. In this paper, we propose Global Label Propagation Network with LLM-based Pseudo Labeling (GLPN-LLM) for multimodal fake news detection, which integrates LLM capabilities via label propagation techniques. The global label propagation can utilize LLM-generated pseudo labels, enhancing prediction accuracy by propagating label information among all samples. For label propagation, a mask-based mechanism is designed to prevent label leakage during training by ensuring that training nodes do not propagate their own labels back to themselves. Experimental results on benchmark datasets show that by synergizing LLMs with label propagation, our model achieves superior performance over state-of-the-art baselines. 

---
# Auto-Patching: Enhancing Multi-Hop Reasoning in Language Models 

**Authors**: Aviv Jan, Dean Tahory, Omer Talmi, Omar Abo Mokh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00483)  

**Abstract**: Multi-hop questions still stump large language models (LLMs), which struggle to link information across multiple reasoning steps. We introduce Auto-Patch, a novel method that dynamically patches hidden states during inference to enhance multi-hop reasoning in LLMs. Building on the PatchScopes framework, Auto-Patch selectively modifies internal representations using a learned classifier. Evaluated on the MuSiQue dataset, Auto-Patch improves the solve rate from 18.45\% (baseline) to 23.63~$\pm$~0.7\% (3 runs), narrowing the gap to Chain-of-Thought prompting (27.44\%). Our results highlight the potential of dynamic hidden state interventions for advancing complex reasoning in LLMs. 

---
# PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings 

**Authors**: Junseo Kim, Jongwook Han, Dongmin Choi, Jongwook Yoon, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2506.00481)  

**Abstract**: Visual persuasion, which uses visual elements to influence cognition and behaviors, is crucial in fields such as advertising and political communication. With recent advancements in artificial intelligence, there is growing potential to develop persuasive systems that automatically generate persuasive images tailored to individuals. However, a significant bottleneck in this area is the lack of comprehensive datasets that connect the persuasiveness of images with the personal information about those who evaluated the images. To address this gap and facilitate technological advancements in personalized visual persuasion, we release the Personalized Visual Persuasion (PVP) dataset, comprising 28,454 persuasive images across 596 messages and 9 persuasion strategies. Importantly, the PVP dataset provides persuasiveness scores of images evaluated by 2,521 human annotators, along with their demographic and psychological characteristics (personality traits and values). We demonstrate the utility of our dataset by developing a persuasive image generator and an automated evaluator, and establish benchmark baselines. Our experiments reveal that incorporating psychological characteristics enhances the generation and evaluation of persuasive images, providing valuable insights for personalized visual persuasion. 

---
# EffiVLM-BENCH: A Comprehensive Benchmark for Evaluating Training-Free Acceleration in Large Vision-Language Models 

**Authors**: Zekun Wang, Minghua Ma, Zexin Wang, Rongchuan Mu, Liping Shan, Ming Liu, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.00479)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved remarkable success, yet their significant computational demands hinder practical deployment. While efforts to improve LVLM efficiency are growing, existing methods lack comprehensive evaluation across diverse backbones, benchmarks, and metrics. In this work, we systematically evaluate mainstream acceleration techniques for LVLMs, categorized into token and parameter compression. We introduce EffiVLM-Bench, a unified framework for assessing not only absolute performance but also generalization and loyalty, while exploring Pareto-optimal trade-offs. Our extensive experiments and in-depth analyses offer insights into optimal strategies for accelerating LVLMs. We open-source code and recipes for EffiVLM-Bench to foster future research. 

---
# Massively Multilingual Adaptation of Large Language Models Using Bilingual Translation Data 

**Authors**: Shaoxiong Ji, Zihao Li, Jaakko Paavola, Indraneil Paul, Hengyu Luo, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2506.00469)  

**Abstract**: This paper investigates a critical design decision in the practice of massively multilingual continual pre-training -- the inclusion of parallel data. Specifically, we study the impact of bilingual translation data for massively multilingual language adaptation of the Llama3 family of models to 500 languages. To this end, we construct the MaLA bilingual translation corpus, containing data from more than 2,500 language pairs. Subsequently, we develop the EMMA-500 Llama 3 suite of four massively multilingual models -- continually pre-trained from the Llama 3 family of base models extensively on diverse data mixes up to 671B tokens -- and explore the effect of continual pre-training with or without bilingual translation data. Comprehensive evaluation across 7 tasks and 12 benchmarks demonstrates that bilingual data tends to enhance language transfer and performance, particularly for low-resource languages. We open-source the MaLA corpus, EMMA-500 Llama 3 suite artefacts, code, and model generations. 

---
# Fact-Controlled Diagnosis of Hallucinations in Medical Text Summarization 

**Authors**: Suhas BN, Han-Chin Shing, Lei Xu, Mitch Strong, Jon Burnsky, Jessica Ofor, Jordan R. Mason, Susan Chen, Sundararajan Srinivasan, Chaitanya Shivade, Jack Moriarty, Joseph Paul Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00448)  

**Abstract**: Hallucinations in large language models (LLMs) during summarization of patient-clinician dialogues pose significant risks to patient care and clinical decision-making. However, the phenomenon remains understudied in the clinical domain, with uncertainty surrounding the applicability of general-domain hallucination detectors. The rarity and randomness of hallucinations further complicate their investigation. In this paper, we conduct an evaluation of hallucination detection methods in the medical domain, and construct two datasets for the purpose: A fact-controlled Leave-N-out dataset -- generated by systematically removing facts from source dialogues to induce hallucinated content in summaries; and a natural hallucination dataset -- arising organically during LLM-based medical summarization. We show that general-domain detectors struggle to detect clinical hallucinations, and that performance on fact-controlled hallucinations does not reliably predict effectiveness on natural hallucinations. We then develop fact-based approaches that count hallucinations, offering explainability not available with existing methods. Notably, our LLM-based detectors, which we developed using fact-controlled hallucinations, generalize well to detecting real-world clinical hallucinations. This research contributes a suite of specialized metrics supported by expert-annotated datasets to advance faithful clinical summarization systems. 

---
# G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models 

**Authors**: Long Bai, Zixuan Li, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.00445)  

**Abstract**: Forecasting over Temporal Knowledge Graphs (TKGs) which predicts future facts based on historical ones has received much attention. Recent studies have introduced Large Language Models (LLMs) for this task to enhance the models' generalization abilities. However, these models perform forecasting via simultaneously learning two kinds of entangled knowledge in the TKG: (1) general patterns, i.e., invariant temporal structures shared across different scenarios; and (2) scenario information, i.e., factual knowledge engaged in specific scenario, such as entities and relations. As a result, the learning processes of these two kinds of knowledge may interfere with each other, which potentially impact the generalization abilities of the models. To enhance the generalization ability of LLMs on this task, in this paper, we propose a General-to-Specific learning framework (G2S) that disentangles the learning processes of the above two kinds of knowledge. In the general learning stage, we mask the scenario information in different TKGs and convert it into anonymous temporal structures. After training on these structures, the model is able to capture the general patterns across different TKGs. In the specific learning stage, we inject the scenario information into the structures via either in-context learning or fine-tuning modes. Experimental results show that G2S effectively improves the generalization abilities of LLMs. 

---
# Inter-Passage Verification for Multi-evidence Multi-answer QA 

**Authors**: Bingsen Chen, Shengjie Wang, Xi Ye, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00425)  

**Abstract**: Multi-answer question answering (QA), where questions can have many valid answers, presents a significant challenge for existing retrieval-augmented generation-based QA systems, as these systems struggle to retrieve and then synthesize a large number of evidence passages. To tackle these challenges, we propose a new multi-answer QA framework -- Retrieval-augmented Independent Reading with Inter-passage Verification (RI$^2$VER). Our framework retrieves a large set of passages and processes each passage individually to generate an initial high-recall but noisy answer set. Then we propose a new inter-passage verification pipeline that validates every candidate answer through (1) Verification Question Generation, (2) Gathering Additional Evidence, and (3) Verification with inter-passage synthesis. Evaluations on the QAMPARI and RoMQA datasets demonstrate that our framework significantly outperforms existing baselines across various model sizes, achieving an average F1 score improvement of 11.17%. Further analysis validates that our inter-passage verification pipeline enables our framework to be particularly beneficial for questions requiring multi-evidence synthesis. 

---
# DYNAC: Dynamic Vocabulary based Non-Autoregressive Contextualization for Speech Recognition 

**Authors**: Yui Sudo, Yosuke Fukumoto, Muhammad Shakeel, Yifan Peng, Chyi-Jiunn Lin, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00422)  

**Abstract**: Contextual biasing (CB) improves automatic speech recognition for rare and unseen phrases. Recent studies have introduced dynamic vocabulary, which represents context phrases as expandable tokens in autoregressive (AR) models. This method improves CB accuracy but with slow inference speed. While dynamic vocabulary can be applied to non-autoregressive (NAR) models, such as connectionist temporal classification (CTC), the conditional independence assumption fails to capture dependencies between static and dynamic tokens. This paper proposes DYNAC (Dynamic Vocabulary-based NAR Contextualization), a self-conditioned CTC method that integrates dynamic vocabulary into intermediate layers. Conditioning the encoder on dynamic vocabulary, DYNAC effectively captures dependencies between static and dynamic tokens while reducing the real-time factor (RTF). Experimental results show that DYNAC reduces RTF by 81% with a 0.1-point degradation in word error rate on the LibriSpeech 960 test-clean set. 

---
# Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System for Dynamic Interactions 

**Authors**: Jihyoung Jang, Minwook Bae, Minji Kim, Dilek Hakkani-Tur, Hyounghun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00421)  

**Abstract**: As chatbots continue to evolve toward human-like, real-world, interactions, multimodality remains an active area of research and exploration. So far, efforts to integrate multimodality into chatbots have primarily focused on image-centric tasks, such as visual dialogue and image-based instructions, placing emphasis on the "eyes" of human perception while neglecting the "ears", namely auditory aspects. Moreover, these studies often center around static interactions that focus on discussing the modality rather than naturally incorporating it into the conversation, which limits the richness of simultaneous, dynamic engagement. Furthermore, while multimodality has been explored in multi-party and multi-session conversations, task-specific constraints have hindered its seamless integration into dynamic, natural conversations. To address these challenges, this study aims to equip chatbots with "eyes and ears" capable of more immersive interactions with humans. As part of this effort, we introduce a new multimodal conversation dataset, Multimodal Multi-Session Multi-Party Conversation ($M^3C$), and propose a novel multimodal conversation model featuring multimodal memory retrieval. Our model, trained on the $M^3C$, demonstrates the ability to seamlessly engage in long-term conversations with multiple speakers in complex, real-world-like settings, effectively processing visual and auditory inputs to understand and respond appropriately. Human evaluations highlight the model's strong performance in maintaining coherent and dynamic interactions, demonstrating its potential for advanced multimodal conversational agents. 

---
# Dual Debiasing for Noisy In-Context Learning for Text Generation 

**Authors**: Siqi Liang, Sumyeong Ahn, Paramveer S. Dhillon, Jiayu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00418)  

**Abstract**: In context learning (ICL) relies heavily on high quality demonstrations drawn from large annotated corpora. Existing approaches detect noisy annotations by ranking local perplexities, presuming that noisy samples yield higher perplexities than their clean counterparts. However, this assumption breaks down when the noise ratio is high and many demonstrations are flawed. We reexamine the perplexity based paradigm for text generation under noisy annotations, highlighting two sources of bias in perplexity: the annotation itself and the domain specific knowledge inherent in large language models (LLMs). To overcome these biases, we introduce a dual debiasing framework that uses synthesized neighbors to explicitly correct perplexity estimates, yielding a robust Sample Cleanliness Score. This metric uncovers absolute sample cleanliness regardless of the overall corpus noise level. Extensive experiments demonstrate our method's superior noise detection capabilities and show that its final ICL performance is comparable to that of a fully clean demonstration corpus. Moreover, our approach remains robust even when noise ratios are extremely high. 

---
# Accelerating Diffusion LLMs via Adaptive Parallel Decoding 

**Authors**: Daniel Israel, Guy Van den Broeck, Aditya Grover  

**Link**: [PDF](https://arxiv.org/pdf/2506.00413)  

**Abstract**: The generation speed of LLMs are bottlenecked by autoregressive decoding, where tokens are predicted sequentially one by one. Alternatively, diffusion large language models (dLLMs) theoretically allow for parallel token generation, but in practice struggle to achieve the speed of autoregressive models without significantly sacrificing quality. We therefore introduce adaptive parallel decoding (APD), a novel method that dynamically adjusts the number of tokens sampled in parallel. We achieve this by defining a multiplicative mixture between the dLLM marginal probabilities and the joint probability of sequences under a small auxiliary autoregressive model. This inverts the standard setup of speculative decoding, where the goal is to sample from a large autoregressive verifier by drafting from a smaller model. We further optimize APD by enabling KV caching and limiting the size of the masked input. Altogether, our method puts forward three tunable parameters to flexibly tradeoff throughput and quality. We show that APD provides markedly higher throughput with minimal quality degradations on downstream benchmarks. 

---
# Causal Structure Discovery for Error Diagnostics of Children's ASR 

**Authors**: Vishwanath Pratap Singh, Md. Sahidullah, Tomi Kinnunen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00402)  

**Abstract**: Children's automatic speech recognition (ASR) often underperforms compared to that of adults due to a confluence of interdependent factors: physiological (e.g., smaller vocal tracts), cognitive (e.g., underdeveloped pronunciation), and extrinsic (e.g., vocabulary limitations, background noise). Existing analysis methods examine the impact of these factors in isolation, neglecting interdependencies-such as age affecting ASR accuracy both directly and indirectly via pronunciation skills. In this paper, we introduce a causal structure discovery to unravel these interdependent relationships among physiology, cognition, extrinsic factors, and ASR errors. Then, we employ causal quantification to measure each factor's impact on children's ASR. We extend the analysis to fine-tuned models to identify which factors are mitigated by fine-tuning and which remain largely unaffected. Experiments on Whisper and Wav2Vec2.0 demonstrate the generalizability of our findings across different ASR systems. 

---
# Scaling Textual Gradients via Sampling-Based Momentum 

**Authors**: Zixin Ding, Junyuan Hong, Jiachen T. Wang, Zinan Lin, Zhangyang Wang, Yuxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00400)  

**Abstract**: As prompts play an increasingly critical role in large language models (LLMs), optimizing textual prompts has become a crucial challenge. The Textual Gradient Descent (TGD) framework has emerged as a promising data-driven approach that iteratively refines textual prompts using LLM - suggested updates (or textual gradients) over minibatches of training samples. In this paper, we empirically demonstrate that scaling the number of training examples initially improves but later degrades TGD's performance across multiple downstream NLP tasks. However, while data scaling improves results for most tasks, it also significantly increases the computational cost when leveraging LLMs. To address this, we draw inspiration from numerical gradient descent and propose Textual Stochastic Gradient Descent with Momentum (TSGD-M) - a method that facilitates scalable in-context learning by reweighting prompt sampling based on past batch distributions. Across nine NLP tasks spanning three domains - including BIG-Bench Hard (BBH), natural language understanding tasks, and reasoning tasks - TSGD-M significantly outperforms TGD baselines that do not incorporate reweighted sampling, while also reducing variance in most tasks. 

---
# Speculative Reward Model Boosts Decision Making Ability of LLMs Cost-Effectively 

**Authors**: Jiawei Gu, Shangsong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00396)  

**Abstract**: Effective decision-making in Large Language Models (LLMs) is essential for handling intricate tasks. However, existing approaches prioritize performance but often overlook the balance between effectiveness and computational cost. To address this, we first introduce the 3E Criteria to systematically assess the cost-effectiveness of search strategies, revealing that existing methods often trade significant efficiency for marginal performance gains. To improve LLM decision-making while maintaining efficiency, we propose the Speculative Reward Model (SRM), a plug-and-play framework that seamlessly integrates with existing search strategies. Specifically, SRM employs an external reward assigner to predict optimal actions, reducing reliance on LLMs' internal self-evaluation. And a speculative verification mechanism is used to prune suboptimal choices and guide the search toward more promising steps. We evaluate SRM on several complex decision-making tasks including mathematical reasoning, planning and numerical reasoning in specialized domains. Experimental results show that SRM reduces costs to 1/10 of the original search framework on average while maintaining effectiveness. 

---
# SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL 

**Authors**: Ge Qu, Jinyang Li, Bowen Qin, Xiaolong Li, Nan Huo, Chenhao Ma, Reynold Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.00391)  

**Abstract**: Current self-correction approaches in text-to-SQL face two critical limitations: 1) Conventional self-correction methods rely on recursive self-calls of LLMs, resulting in multiplicative computational overhead, and 2) LLMs struggle to implement effective error detection and correction for declarative SQL queries, as they fail to demonstrate the underlying reasoning path. In this work, we propose SHARE, an SLM-based Hierarchical Action corREction assistant that enables LLMs to perform more precise error localization and efficient correction. SHARE orchestrates three specialized Small Language Models (SLMs) in a sequential pipeline, where it first transforms declarative SQL queries into stepwise action trajectories that reveal underlying reasoning, followed by a two-phase granular refinement. We further propose a novel hierarchical self-evolution strategy for data-efficient training. Experimental results demonstrate that SHARE effectively enhances self-correction capabilities while proving robust across various LLMs. Furthermore, our comprehensive analysis shows that SHARE maintains strong performance even in low-resource training settings, which is particularly valuable for text-to-SQL applications with data privacy constraints. 

---
# Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training 

**Authors**: Keyeun Lee, Seolhee Lee, Esther Hehsun Kim, Yena Ko, Jinsu Eun, Dahee Kim, Hyewon Cho, Haiyi Zhu, Robert E. Kraut, Eunyoung Suh, Eun-mee Kim, Hajin Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00386)  

**Abstract**: Effective communication training is essential to preparing nurses for high-quality patient care. While standardized patient (SP) simulations provide valuable experiential learning, they are often costly and inflexible. Virtual patient (VP) systems offer a scalable alternative, but most fail to adapt to the varying communication skills of trainees. In particular, when trainees respond ineffectively, VPs should escalate in hostility or become uncooperative--yet this level of adaptive interaction remains largely unsupported. To address this gap, we introduce Adaptive-VP, a VP dialogue generation framework that leverages large language models (LLMs) to dynamically adapt VP behavior based on trainee input. The framework features a pipeline for constructing clinically grounded yet flexible VP scenarios and a modular system for assessing trainee communication and adjusting VP responses in real time, while ensuring learner safety. We validated Adaptive-VP by simulating challenging patient conversations. Automated evaluation using a corpus from practicing nurses showed that our communication skill evaluation mechanism reflected real-world proficiency levels. Expert nurses further confirmed that Adaptive-VP produced more natural and realistic interactions than existing approaches, demonstrating its potential as a scalable and effective tool for nursing communication training. 

---
# Neuro2Semantic: A Transfer Learning Framework for Semantic Reconstruction of Continuous Language from Human Intracranial EEG 

**Authors**: Siavash Shams, Richard Antonello, Gavin Mischler, Stephan Bickel, Ashesh Mehta, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00381)  

**Abstract**: Decoding continuous language from neural signals remains a significant challenge in the intersection of neuroscience and artificial intelligence. We introduce Neuro2Semantic, a novel framework that reconstructs the semantic content of perceived speech from intracranial EEG (iEEG) recordings. Our approach consists of two phases: first, an LSTM-based adapter aligns neural signals with pre-trained text embeddings; second, a corrector module generates continuous, natural text directly from these aligned embeddings. This flexible method overcomes the limitations of previous decoding approaches and enables unconstrained text generation. Neuro2Semantic achieves strong performance with as little as 30 minutes of neural data, outperforming a recent state-of-the-art method in low-data settings. These results highlight the potential for practical applications in brain-computer interfaces and neural decoding technologies. 

---
# Efficient Latent Semantic Clustering for Scaling Test-Time Computation of LLMs 

**Authors**: Sungjae Lee, Hoyoung Kim, Jeongyeon Hwang, Eunhyeok Park, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2506.00344)  

**Abstract**: Scaling test-time computation--generating and analyzing multiple or sequential outputs for a single input--has become a promising strategy for improving the reliability and quality of large language models (LLMs), as evidenced by advances in uncertainty quantification and multi-step reasoning. A key shared component is semantic clustering, which groups outputs that differ in form but convey the same meaning. Semantic clustering enables estimation of the distribution over the semantics of outputs and helps avoid redundant exploration of reasoning paths. However, existing approaches typically rely on external models, which introduce substantial computational overhead and often fail to capture context-aware semantics. We propose Latent Semantic Clustering (LSC), a lightweight and context-sensitive method that leverages the generator LLM's internal hidden states for clustering, eliminating the need for external models. Our extensive experiment across various LLMs and datasets shows that LSC significantly improves the computational efficiency of test-time scaling while maintaining or exceeding the performance of existing methods. 

---
# OWSM v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning 

**Authors**: Yifan Peng, Shakeel Muhammad, Yui Sudo, William Chen, Jinchuan Tian, Chyi-Jiunn Lin, Shinji Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00338)  

**Abstract**: The Open Whisper-style Speech Models (OWSM) project has developed a series of fully open speech foundation models using academic-scale resources, but their training data remains insufficient. This work enhances OWSM by integrating YODAS, a large-scale web-crawled dataset with a Creative Commons license. However, incorporating YODAS is nontrivial due to its wild nature, which introduces challenges such as incorrect language labels and audio-text misalignments. To address this, we develop a scalable data-cleaning pipeline using public toolkits, yielding a dataset with 166,000 hours of speech across 75 languages. Our new series of OWSM v4 models, trained on this curated dataset alongside existing OWSM data, significantly outperform previous versions on multilingual benchmarks. Our models even match or surpass frontier industrial models like Whisper and MMS in multiple scenarios. We will publicly release the cleaned YODAS data, pre-trained models, and all associated scripts via the ESPnet toolkit. 

---
# Beyond Context to Cognitive Appraisal: Emotion Reasoning as a Theory of Mind Benchmark for Large Language Models 

**Authors**: Gerard Christopher Yeo, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2506.00334)  

**Abstract**: Datasets used for emotion recognition tasks typically contain overt cues that can be used in predicting the emotions expressed in a text. However, one challenge is that texts sometimes contain covert contextual cues that are rich in affective semantics, which warrant higher-order reasoning abilities to infer emotional states, not simply the emotions conveyed. This study advances beyond surface-level perceptual features to investigate how large language models (LLMs) reason about others' emotional states using contextual information, within a Theory-of-Mind (ToM) framework. Grounded in Cognitive Appraisal Theory, we curate a specialized ToM evaluation dataset1 to assess both forward reasoning - from context to emotion- and backward reasoning - from emotion to inferred context. We showed that LLMs can reason to a certain extent, although they are poor at associating situational outcomes and appraisals with specific emotions. Our work highlights the need for psychological theories in the training and evaluation of LLMs in the context of emotion reasoning. 

---
# Disentangling Codemixing in Chats: The NUS ABC Codemixed Corpus 

**Authors**: Svetlana Churina, Akshat Gupta, Insyirah Mujtahid, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2506.00332)  

**Abstract**: Code-mixing involves the seamless integration of linguistic elements from multiple languages within a single discourse, reflecting natural multilingual communication patterns. Despite its prominence in informal interactions such as social media, chat messages and instant-messaging exchanges, there has been a lack of publicly available corpora that are author-labeled and suitable for modeling human conversations and relationships. This study introduces the first labeled and general-purpose corpus for understanding code-mixing in context while maintaining rigorous privacy and ethical standards. Our live project will continuously gather, verify, and integrate code-mixed messages into a structured dataset released in JSON format, accompanied by detailed metadata and linguistic statistics. To date, it includes over 355,641 messages spanning various code-mixing patterns, with a primary focus on English, Mandarin, and other languages. We expect the Codemix Corpus to serve as a foundational dataset for research in computational linguistics, sociolinguistics, and NLP applications. 

---
# TreeRare: Syntax Tree-Guided Retrieval and Reasoning for Knowledge-Intensive Question Answering 

**Authors**: Boyi Zhang, Zhuo Liu, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2506.00331)  

**Abstract**: In real practice, questions are typically complex and knowledge-intensive, requiring Large Language Models (LLMs) to recognize the multifaceted nature of the question and reason across multiple information sources. Iterative and adaptive retrieval, where LLMs decide when and what to retrieve based on their reasoning, has been shown to be a promising approach to resolve complex, knowledge-intensive questions. However, the performance of such retrieval frameworks is limited by the accumulation of reasoning errors and misaligned retrieval results. To overcome these limitations, we propose TreeRare (Syntax Tree-Guided Retrieval and Reasoning), a framework that utilizes syntax trees to guide information retrieval and reasoning for question answering. Following the principle of compositionality, TreeRare traverses the syntax tree in a bottom-up fashion, and in each node, it generates subcomponent-based queries and retrieves relevant passages to resolve localized uncertainty. A subcomponent question answering module then synthesizes these passages into concise, context-aware evidence. Finally, TreeRare aggregates the evidence across the tree to form a final answer. Experiments across five question answering datasets involving ambiguous or multi-hop reasoning demonstrate that TreeRare achieves substantial improvements over existing state-of-the-art methods. 

---
# SkillVerse : Assessing and Enhancing LLMs with Tree Evaluation 

**Authors**: Yufei Tian, Jiao Sun, Nanyun Peng, Zizhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00319)  

**Abstract**: As language models evolve to tackle complex, multifaceted tasks, their evaluation must adapt to capture this intricacy. A granular, skill-specific understanding of model capabilities can empower researchers to make informed model development plans. In this paper, we introduce SkillVerse, an unsupervised tree-structured diagnosis framework for understanding model proficiency in specific abilities. With LLM as a judge, SkillVerse first critiques the model responses, and then organizes them into a hierarchical structure termed dendrogram. Given proficiency at arbitrary levels of granularity, SkillVerse is flexible to produce insights of behaviors of modern large models. We also demonstrate its efficacy in two downstream tasks: 1) improving model in-context learning by 25% using a tree-search algorithm to select more informative few-shot demonstrations, and 2) accurately predicting new model weaknesses with a 55% success rate, 22% higher than without SkillVerse. 

---
# An evaluation of LLMs for generating movie reviews: GPT-4o, Gemini-2.0 and DeepSeek-V3 

**Authors**: Brendan Sands, Yining Wang, Chenhao Xu, Yuxuan Zhou, Lai Wei, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00312)  

**Abstract**: Large language models (LLMs) have been prominent in various tasks, including text generation and summarisation. The applicability of LLMs to the generation of product reviews is gaining momentum, paving the way for the generation of movie reviews. In this study, we propose a framework that generates movie reviews using three LLMs (GPT-4o, DeepSeek-V3, and Gemini-2.0), and evaluate their performance by comparing the generated outputs with IMDb user reviews. We use movie subtitles and screenplays as input to the LLMs and investigate how they affect the quality of reviews generated. We review the LLM-based movie reviews in terms of vocabulary, sentiment polarity, similarity, and thematic consistency in comparison to IMDB user reviews. The results demonstrate that LLMs are capable of generating syntactically fluent and structurally complete movie reviews. Nevertheless, there is still a noticeable gap in emotional richness and stylistic coherence between LLM-generated and IMDb reviews, suggesting that further refinement is needed to improve the overall quality of movie review generation. We provided a survey-based analysis where participants were told to distinguish between LLM and IMDb user reviews. The results show that LLM-generated reviews are difficult to distinguish from IMDB user reviews. We found that DeepSeek-V3 produced the most balanced reviews, closely matching IMDb reviews. GPT-4o overemphasised positive emotions, while Gemini-2.0 captured negative emotions better but showed excessive emotional intensity. 

---
# Lossless Token Sequence Compression via Meta-Tokens 

**Authors**: John Harvill, Ziwei Fan, Hao Wang, Yizhou Sun, Hao Ding, Luke Huan, Anoop Deoras  

**Link**: [PDF](https://arxiv.org/pdf/2506.00307)  

**Abstract**: Existing work on prompt compression for Large Language Models (LLM) focuses on lossy methods that try to maximize the retention of semantic information that is relevant to downstream tasks while significantly reducing the sequence length. In this paper, we introduce a task-agnostic lossless compression technique similar to LZ77 that makes it possible to reduce the input token sequence length on average by 27\% and 18\% for the two evaluation tasks explored here. Given that we use transformer-based LLMs, this equates to 47\% and 33\% less encoding computation, respectively, due to the quadratic nature of attention. The token sequence transformation is trivial to reverse and highlights that no semantic information is lost in the process. We evaluate our proposed approach on two tasks that require strict preservation of semantics/syntax and demonstrate that existing lossy compression methods perform poorly in this setting. We find that our lossless compression technique produces only a small gap in performance compared to using the uncompressed input and posit that larger models and an expanded computing budget would likely erase the gap entirely. 

---
# Can LLMs Understand Unvoiced Speech? Exploring EMG-to-Text Conversion with LLMs 

**Authors**: Payal Mohapatra, Akash Pandey, Xiaoyuan Zhang, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00304)  

**Abstract**: Unvoiced electromyography (EMG) is an effective communication tool for individuals unable to produce vocal speech. However, most prior methods rely on paired voiced and unvoiced EMG signals, along with speech data, for EMG-to-text conversion, which is not practical for such individuals. Given the rise of large language models (LLMs) in speech recognition, we explore their potential to understand unvoiced speech. To this end, we address the challenge of learning from unvoiced EMG alone and propose a novel EMG adaptor module that maps EMG features into an LLM's input space, achieving an average word error rate (WER) of 0.49 on a closed-vocabulary unvoiced EMG-to-text task. Even with a conservative data availability of just six minutes, our approach improves performance over specialized models by nearly 20%. While LLMs have been shown to be extendable to new language modalities -- such as audio -- understanding articulatory biosignals like unvoiced EMG remains more challenging. This work takes a crucial first step toward enabling LLMs to comprehend unvoiced speech using surface EMG. 

---
# DLM-One: Diffusion Language Models for One-Step Sequence Generation 

**Authors**: Tianqi Chen, Shujian Zhang, Mingyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00290)  

**Abstract**: This paper introduces DLM-One, a score-distillation-based framework for one-step sequence generation with continuous diffusion language models (DLMs). DLM-One eliminates the need for iterative refinement by aligning the scores of a student model's outputs in the continuous token embedding space with the score function of a pretrained teacher DLM. We investigate whether DLM-One can achieve substantial gains in sampling efficiency for language modeling. Through comprehensive experiments on DiffuSeq -- a representative continuous DLM -- we show that DLM-One achieves up to ~500x speedup in inference time while maintaining competitive performance on benchmark text generation tasks used to evaluate the teacher models. We further analyze the method's empirical behavior across multiple datasets, providing initial insights into its generality and practical applicability. Our findings position one-step diffusion as a promising direction for efficient, high-quality language generation and broader adoption of continuous diffusion models operating in embedding space for natural language processing. 

---
# Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation 

**Authors**: Ahmed Elhady, Eneko Agirre, Mikel Artetxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00288)  

**Abstract**: Continued pretraining (CPT) is a popular approach to adapt existing large language models (LLMs) to new languages. When doing so, it is common practice to include a portion of English data in the mixture, but its role has not been carefully studied to date. In this work, we show that including English does not impact validation perplexity, yet it is critical for the emergence of downstream capabilities in the target language. We introduce a language-agnostic benchmark for in-context learning (ICL), which reveals catastrophic forgetting early on CPT when English is not included. This in turn damages the ability of the model to generalize to downstream prompts in the target language as measured by perplexity, even if it does not manifest in terms of accuracy until later in training, and can be tied to a big shift in the model parameters. Based on these insights, we introduce curriculum learning and exponential moving average (EMA) of weights as effective alternatives to mitigate the need for English. All in all, our work sheds light into the dynamics by which emergent abilities arise when doing CPT for language adaptation, and can serve as a foundation to design more effective methods in the future. 

---
# Hierarchical Level-Wise News Article Clustering via Multilingual Matryoshka Embeddings 

**Authors**: Hans W. A. Hanley, Zakir Durumeric  

**Link**: [PDF](https://arxiv.org/pdf/2506.00277)  

**Abstract**: Contextual large language model embeddings are increasingly utilized for topic modeling and clustering. However, current methods often scale poorly, rely on opaque similarity metrics, and struggle in multilingual settings. In this work, we present a novel, scalable, interpretable, hierarchical, and multilingual approach to clustering news articles and social media data. To do this, we first train multilingual Matryoshka embeddings that can determine story similarity at varying levels of granularity based on which subset of the dimensions of the embeddings is examined. This embedding model achieves state-of-the-art performance on the SemEval 2022 Task 8 test dataset (Pearson $\rho$ = 0.816). Once trained, we develop an efficient hierarchical clustering algorithm that leverages the hierarchical nature of Matryoshka embeddings to identify unique news stories, narratives, and themes. We conclude by illustrating how our approach can identify and cluster stories, narratives, and overarching themes within real-world news datasets. 

---
# CASPER: A Large Scale Spontaneous Speech Dataset 

**Authors**: Cihan Xiao, Ruixing Liang, Xiangyu Zhang, Mehmet Emre Tiryaki, Veronica Bae, Lavanya Shankar, Rong Yang, Ethan Poon, Emmanuel Dupoux, Sanjeev Khudanpur, Leibny Paola Garcia Perera  

**Link**: [PDF](https://arxiv.org/pdf/2506.00267)  

**Abstract**: The success of large language models has driven interest in developing similar speech processing capabilities. However, a key challenge is the scarcity of high-quality spontaneous speech data, as most existing datasets contain scripted dialogues. To address this, we present a novel pipeline for eliciting and recording natural dialogues and release our Stage 1 dataset with 200+ hours of spontaneous speech. Our approach fosters fluid, natural conversations while encouraging a diverse range of topics and interactive exchanges. Unlike traditional methods, it facilitates genuine interactions, providing a reproducible framework for future data collection. This paper introduces our dataset and methodology, laying the groundwork for addressing the shortage of spontaneous speech data. We plan to expand this dataset in future stages, offering a growing resource for the research community. 

---
# MultiHoax: A Dataset of Multi-hop False-Premise Questions 

**Authors**: Mohammadamin Shafiei, Hamidreza Saffari, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00264)  

**Abstract**: As Large Language Models are increasingly deployed in high-stakes domains, their ability to detect false assumptions and reason critically is crucial for ensuring reliable outputs. False-premise questions (FPQs) serve as an important evaluation method by exposing cases where flawed assumptions lead to incorrect responses. While existing benchmarks focus on single-hop FPQs, real-world reasoning often requires multi-hop inference, where models must verify consistency across multiple reasoning steps rather than relying on surface-level cues. To address this gap, we introduce MultiHoax, a benchmark for evaluating LLMs' ability to handle false premises in complex, multi-step reasoning tasks. Our dataset spans seven countries and ten diverse knowledge categories, using Wikipedia as the primary knowledge source to enable factual reasoning across regions. Experiments reveal that state-of-the-art LLMs struggle to detect false premises across different countries, knowledge categories, and multi-hop reasoning types, highlighting the need for improved false premise detection and more robust multi-hop reasoning capabilities in LLMs. 

---
# The Impact of Disability Disclosure on Fairness and Bias in LLM-Driven Candidate Selection 

**Authors**: Mahammed Kamruzzaman, Gene Louis Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.00256)  

**Abstract**: As large language models (LLMs) become increasingly integrated into hiring processes, concerns about fairness have gained prominence. When applying for jobs, companies often request/require demographic information, including gender, race, and disability or veteran status. This data is collected to support diversity and inclusion initiatives, but when provided to LLMs, especially disability-related information, it raises concerns about potential biases in candidate selection outcomes. Many studies have highlighted how disability can impact CV screening, yet little research has explored the specific effect of voluntarily disclosed information on LLM-driven candidate selection. This study seeks to bridge that gap. When candidates shared identical gender, race, qualifications, experience, and backgrounds, and sought jobs with minimal employment rate gaps between individuals with and without disabilities (e.g., Cashier, Software Developer), LLMs consistently favored candidates who disclosed that they had no disability. Even in cases where candidates chose not to disclose their disability status, the LLMs were less likely to select them compared to those who explicitly stated they did not have a disability. 

---
# Aligned but Blind: Alignment Increases Implicit Bias by Reducing Awareness of Race 

**Authors**: Lihao Sun, Chengzhi Mao, Valentin Hofmann, Xuechunzi Bai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00253)  

**Abstract**: Although value-aligned language models (LMs) appear unbiased in explicit bias evaluations, they often exhibit stereotypes in implicit word association tasks, raising concerns about their fair usage. We investigate the mechanisms behind this discrepancy and find that alignment surprisingly amplifies implicit bias in model outputs. Specifically, we show that aligned LMs, unlike their unaligned counterparts, overlook racial concepts in early internal representations when the context is ambiguous. Not representing race likely fails to activate safety guardrails, leading to unintended biases. Inspired by this insight, we propose a new bias mitigation strategy that works by incentivizing the representation of racial concepts in the early model layers. In contrast to conventional mitigation methods of machine unlearning, our interventions find that steering the model to be more aware of racial concepts effectively mitigates implicit bias. Similar to race blindness in humans, ignoring racial nuances can inadvertently perpetuate subtle biases in LMs. 

---
# PersianMedQA: Language-Centric Evaluation of LLMs in the Persian Medical Domain 

**Authors**: Mohammad Javad Ranjbar Kalahroodi, Amirhossein Sheikholselami, Sepehr Karimi, Sepideh Ranjbar Kalahroodi, Heshaam Faili, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2506.00250)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable performance on a wide range of NLP benchmarks, often surpassing human-level accuracy. However, their reliability in high-stakes domains such as medicine, particularly in low-resource languages, remains underexplored. In this work, we introduce PersianMedQA, a large-scale, expert-validated dataset of multiple-choice Persian medical questions, designed to evaluate LLMs across both Persian and English. We benchmark over 40 state-of-the-art models, including general-purpose, Persian fine-tuned, and medical LLMs, in zero-shot and chain-of-thought (CoT) settings. Our results show that closed-source general models (e.g., GPT-4.1) consistently outperform all other categories, achieving 83.3% accuracy in Persian and 80.7% in English, while Persian fine-tuned models such as Dorna underperform significantly (e.g., 35.9% in Persian), often struggling with both instruction-following and domain reasoning. We also analyze the impact of translation, showing that while English performance is generally higher, Persian responses are sometimes more accurate due to cultural and clinical contextual cues. Finally, we demonstrate that model size alone is insufficient for robust performance without strong domain or language adaptation. PersianMedQA provides a foundation for evaluating multilingual and culturally grounded medical reasoning in LLMs. The PersianMedQA dataset can be accessed at: this https URL](this https URL 

---
# MedOrch: Medical Diagnosis with Tool-Augmented Reasoning Agents for Flexible Extensibility 

**Authors**: Yexiao He, Ang Li, Boyi Liu, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2506.00235)  

**Abstract**: Healthcare decision-making represents one of the most challenging domains for Artificial Intelligence (AI), requiring the integration of diverse knowledge sources, complex reasoning, and various external analytical tools. Current AI systems often rely on either task-specific models, which offer limited adaptability, or general language models without grounding with specialized external knowledge and tools. We introduce MedOrch, a novel framework that orchestrates multiple specialized tools and reasoning agents to provide comprehensive medical decision support. MedOrch employs a modular, agent-based architecture that facilitates the flexible integration of domain-specific tools without altering the core system. Furthermore, it ensures transparent and traceable reasoning processes, enabling clinicians to meticulously verify each intermediate step underlying the system's recommendations. We evaluate MedOrch across three distinct medical applications: Alzheimer's disease diagnosis, chest X-ray interpretation, and medical visual question answering, using authentic clinical datasets. The results demonstrate MedOrch's competitive performance across these diverse medical tasks. Notably, in Alzheimer's disease diagnosis, MedOrch achieves an accuracy of 93.26%, surpassing the state-of-the-art baseline by over four percentage points. For predicting Alzheimer's disease progression, it attains a 50.35% accuracy, marking a significant improvement. In chest X-ray analysis, MedOrch exhibits superior performance with a Macro AUC of 61.2% and a Macro F1-score of 25.5%. Moreover, in complex multimodal visual question answering (Image+Table), MedOrch achieves an accuracy of 54.47%. These findings underscore MedOrch's potential to advance healthcare AI by enabling reasoning-driven tool utilization for multimodal medical data processing and supporting intricate cognitive tasks in clinical decision-making. 

---
# ComposeRAG: A Modular and Composable RAG for Corpus-Grounded Multi-Hop Question Answering 

**Authors**: Ruofan Wu, Youngwon Lee, Fan Shu, Danmei Xu, Seung-won Hwang, Zhewei Yao, Yuxiong He, Feng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00232)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly diverse, yet many suffer from monolithic designs that tightly couple core functions like query reformulation, retrieval, reasoning, and verification. This limits their interpretability, systematic evaluation, and targeted improvement, especially for complex multi-hop question answering. We introduce ComposeRAG, a novel modular abstraction that decomposes RAG pipelines into atomic, composable modules. Each module, such as Question Decomposition, Query Rewriting, Retrieval Decision, and Answer Verification, acts as a parameterized transformation on structured inputs/outputs, allowing independent implementation, upgrade, and analysis. To enhance robustness against errors in multi-step reasoning, ComposeRAG incorporates a self-reflection mechanism that iteratively revisits and refines earlier steps upon verification failure. Evaluated on four challenging multi-hop QA benchmarks, ComposeRAG consistently outperforms strong baselines in both accuracy and grounding fidelity. Specifically, it achieves up to a 15% accuracy improvement over fine-tuning-based methods and up to a 5% gain over reasoning-specialized pipelines under identical retrieval conditions. Crucially, ComposeRAG significantly enhances grounding: its verification-first design reduces ungrounded answers by over 10% in low-quality retrieval settings, and by approximately 3% even with strong corpora. Comprehensive ablation studies validate the modular architecture, demonstrating distinct and additive contributions from each component. These findings underscore ComposeRAG's capacity to deliver flexible, transparent, scalable, and high-performing multi-hop reasoning with improved grounding and interpretability. 

---
# REIC: RAG-Enhanced Intent Classification at Scale 

**Authors**: Ziji Zhang, Michael Yang, Zhiyu Chen, Yingying Zhuang, Shu-Ting Pi, Qun Liu, Rajashekar Maragoud, Vy Nguyen, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00210)  

**Abstract**: Accurate intent classification is critical for efficient routing in customer service, ensuring customers are connected with the most suitable agents while reducing handling times and operational costs. However, as companies expand their product lines, intent classification faces scalability challenges due to the increasing number of intents and variations in taxonomy across different verticals. In this paper, we introduce REIC, a Retrieval-augmented generation Enhanced Intent Classification approach, which addresses these challenges effectively. REIC leverages retrieval-augmented generation (RAG) to dynamically incorporate relevant knowledge, enabling precise classification without the need for frequent retraining. Through extensive experiments on real-world datasets, we demonstrate that REIC outperforms traditional fine-tuning, zero-shot, and few-shot methods in large-scale customer service settings. Our results highlight its effectiveness in both in-domain and out-of-domain scenarios, demonstrating its potential for real-world deployment in adaptive and large-scale intent classification systems. 

---
# Structure-Aware Fill-in-the-Middle Pretraining for Code 

**Authors**: Linyuan Gong, Alvin Cheung, Mostafa Elhoushi, Sida Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00204)  

**Abstract**: Fill-in-the-Middle (FIM) is a common pretraining method for code LLMs, where models complete code segments given surrounding context. However, existing LLMs treat code as plain text and mask random character spans. We propose and evaluate AST-FIM, a pretraining strategy that leverages Abstract Syntax Trees (ASTs) to mask complete syntactic structures at scale, ensuring coherent training examples better aligned with universal code structures and common code editing patterns such as blocks, expressions, or functions. To evaluate real-world fill-in-the-middle (FIM) programming tasks, we introduce Real-FIM-Eval, a benchmark derived from 30,000+ GitHub commits across 12 languages. On infilling tasks, experiments on 1B and 8B parameter models show that AST-FIM is particularly beneficial for real-world code editing as it outperforms standard random-character FIM by up to 5 pts on standard FIM benchmarks. Our code is publicly available at this https URL. 

---
# Structuring Radiology Reports: Challenging LLMs with Lightweight Models 

**Authors**: Johannes Moll, Louisa Fay, Asfandyar Azhar, Sophie Ostmeier, Tim Lueth, Sergios Gatidis, Curtis Langlotz, Jean-Benoit Delbrouck  

**Link**: [PDF](https://arxiv.org/pdf/2506.00200)  

**Abstract**: Radiology reports are critical for clinical decision-making but often lack a standardized format, limiting both human interpretability and machine learning (ML) applications. While large language models (LLMs) have shown strong capabilities in reformatting clinical text, their high computational requirements, lack of transparency, and data privacy concerns hinder practical deployment. To address these challenges, we explore lightweight encoder-decoder models (<300M parameters)-specifically T5 and BERT2BERT-for structuring radiology reports from the MIMIC-CXR and CheXpert Plus datasets. We benchmark these models against eight open-source LLMs (1B-70B), adapted using prefix prompting, in-context learning (ICL), and low-rank adaptation (LoRA) finetuning. Our best-performing lightweight model outperforms all LLMs adapted using prompt-based techniques on a human-annotated test set. While some LoRA-finetuned LLMs achieve modest gains over the lightweight model on the Findings section (BLEU 6.4%, ROUGE-L 4.8%, BERTScore 3.6%, F1-RadGraph 1.1%, GREEN 3.6%, and F1-SRR-BERT 4.3%), these improvements come at the cost of substantially greater computational resources. For example, LLaMA-3-70B incurred more than 400 times the inference time, cost, and carbon emissions compared to the lightweight model. These results underscore the potential of lightweight, task-specific models as sustainable and privacy-preserving solutions for structuring clinical text in resource-constrained healthcare settings. 

---
# Let Them Down Easy! Contextual Effects of LLM Guardrails on User Perceptions and Preferences 

**Authors**: Mingqian Zheng, Wenjia Hu, Patrick Zhao, Motahhare Eslami, Jena D. Hwang, Faeze Brahman, Carolyn Rose, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2506.00195)  

**Abstract**: Current LLMs are trained to refuse potentially harmful input queries regardless of whether users actually had harmful intents, causing a tradeoff between safety and user experience. Through a study of 480 participants evaluating 3,840 query-response pairs, we examine how different refusal strategies affect user perceptions across varying motivations. Our findings reveal that response strategy largely shapes user experience, while actual user motivation has negligible impact. Partial compliance -- providing general information without actionable details -- emerges as the optimal strategy, reducing negative user perceptions by over 50% to flat-out refusals. Complementing this, we analyze response patterns of 9 state-of-the-art LLMs and evaluate how 6 reward models score different refusal strategies, demonstrating that models rarely deploy partial compliance naturally and reward models currently undervalue it. This work demonstrates that effective guardrails require focusing on crafting thoughtful refusals rather than detecting intent, offering a path toward AI safety mechanisms that ensure both safety and sustained user engagement. 

---
# Werewolf: A Straightforward Game Framework with TTS for Improved User Engagement 

**Authors**: Qihui Fan, Enfu Nan, Wenbo Li, Lei Lu, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00160)  

**Abstract**: The growing popularity of social deduction game systems for both business applications and AI research has greatly benefited from the rapid advancements in Large Language Models (LLMs), which now demonstrate stronger reasoning and persuasion capabilities. Especially with the raise of DeepSeek R1 and V3 models, LLMs should enable a more engaging experience for human players in LLM-agent-based social deduction games like Werewolf. Previous works either fine-tuning, advanced prompting engineering, or additional experience pool to achieve engaging text-format Werewolf game experience. We propose a novel yet straightforward LLM-based Werewolf game system with tuned Text-to-Speech(TTS) models designed for enhanced compatibility with various LLM models, and improved user engagement. We argue with ever enhancing LLM reasoning, extra components will be unnecessary in the case of Werewolf. 

---
# Vedavani: A Benchmark Corpus for ASR on Vedic Sanskrit Poetry 

**Authors**: Sujeet Kumar, Pretam Ray, Abhinay Beerukuri, Shrey Kamoji, Manoj Balaji Jagadeeshan, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2506.00145)  

**Abstract**: Sanskrit, an ancient language with a rich linguistic heritage, presents unique challenges for automatic speech recognition (ASR) due to its phonemic complexity and the phonetic transformations that occur at word junctures, similar to the connected speech found in natural conversations. Due to these complexities, there has been limited exploration of ASR in Sanskrit, particularly in the context of its poetic verses, which are characterized by intricate prosodic and rhythmic patterns. This gap in research raises the question: How can we develop an effective ASR system for Sanskrit, particularly one that captures the nuanced features of its poetic form? In this study, we introduce Vedavani, the first comprehensive ASR study focused on Sanskrit Vedic poetry. We present a 54-hour Sanskrit ASR dataset, consisting of 30,779 labelled audio samples from the Rig Veda and Atharva Veda. This dataset captures the precise prosodic and rhythmic features that define the language. We also benchmark the dataset on various state-of-the-art multilingual speech models.$^{1}$ Experimentation revealed that IndicWhisper performed the best among the SOTA models. 

---
# LaMP-QA: A Benchmark for Personalized Long-form Question Answering 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2506.00137)  

**Abstract**: Personalization is essential for question answering systems that are user-centric. Despite its importance, personalization in answer generation has been relatively underexplored. This is mainly due to lack of resources for training and evaluating personalized question answering systems. We address this gap by introducing LaMP-QA -- a benchmark designed for evaluating personalized long-form answer generation. The benchmark covers questions from three major categories: (1) Arts & Entertainment, (2) Lifestyle & Personal Development, and (3) Society & Culture, encompassing over 45 subcategories in total. To assess the quality and potential impact of the LaMP-QA benchmark for personalized question answering, we conduct comprehensive human and automatic evaluations, to compare multiple evaluation strategies for evaluating generated personalized responses and measure their alignment with human preferences. Furthermore, we benchmark a number of non-personalized and personalized approaches based on open-source and proprietary large language models (LLMs). Our results show that incorporating the personalized context provided leads to performance improvements of up to 39%. The benchmark is publicly released to support future research in this area. 

---
# Spurious Correlations and Beyond: Understanding and Mitigating Shortcut Learning in SDOH Extraction with Large Language Models 

**Authors**: Fardin Ahsan Sakib, Ziwei Zhu, Karen Trister Grace, Meliha Yetisgen, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2506.00134)  

**Abstract**: Social determinants of health (SDOH) extraction from clinical text is critical for downstream healthcare analytics. Although large language models (LLMs) have shown promise, they may rely on superficial cues leading to spurious predictions. Using the MIMIC portion of the SHAC (Social History Annotation Corpus) dataset and focusing on drug status extraction as a case study, we demonstrate that mentions of alcohol or smoking can falsely induce models to predict current/past drug use where none is present, while also uncovering concerning gender disparities in model performance. We further evaluate mitigation strategies - such as prompt engineering and chain-of-thought reasoning - to reduce these false positives, providing insights into enhancing LLM reliability in health domains. 

---
# Writing-Zero: Bridge the Gap Between Non-verifiable Problems and Verifiable Rewards 

**Authors**: Xun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00103)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has enabled large language models (LLMs) to achieve remarkable breakthroughs in reasoning tasks with objective ground-truth answers, such as mathematics and code generation. However, a significant gap remains for non-verifiable tasks, like creative writing and open-ended dialogue, where quality assessment is inherently subjective and lacks definitive references. Existing approaches for these domains often rely on scalar reward models trained with human preferences, which suffer from limited generalization and are prone to reward hacking, such as over-explanation and length bias. In this work, we propose a unified RLVR-based training paradigm that bridges the gap between non-verifiable tasks and verifiable rewards. We introduce a writing-principle-based pairwise Generative Reward Model (GenRM) and a novel Bootstrapped Relative Policy Optimization (BRPO) algorithm. The pairwise writing GenRM leverages self-principled critique to transform subjective assessments into reliable, verifiable rewards, while BRPO enables dynamic, reference-free pairwise comparison by leveraging a bootstrapped response as temporary reference from within group rollouts during RL training. Our approach empowers LLMs to develop robust writing capabilities without supervised fine-tuning, as demonstrated by Writing-Zero, which shows consistent improvement and strong resistance to reward hacking compared to scalar reward baselines. Furthermore, our method achieves competitive results on both in-house and open-source writing benchmarks. Our findings suggest the potential to unify rule-based, reference-based, and reference-free reward modeling under the RLVR framework, thus paving the way for a comprehensive and scalable RL training paradigm applicable across all language tasks. 

---
# HD-NDEs: Neural Differential Equations for Hallucination Detection in LLMs 

**Authors**: Qing Li, Jiahui Geng, Zongxiong Chen, Derui Zhu, Yuxia Wang, Congbo Ma, Chenyang Lyu, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2506.00088)  

**Abstract**: In recent years, large language models (LLMs) have made remarkable advancements, yet hallucination, where models produce inaccurate or non-factual statements, remains a significant challenge for real-world deployment. Although current classification-based methods, such as SAPLMA, are highly efficient in mitigating hallucinations, they struggle when non-factual information arises in the early or mid-sequence of outputs, reducing their reliability. To address these issues, we propose Hallucination Detection-Neural Differential Equations (HD-NDEs), a novel method that systematically assesses the truthfulness of statements by capturing the full dynamics of LLMs within their latent space. Our approaches apply neural differential equations (Neural DEs) to model the dynamic system in the latent space of LLMs. Then, the sequence in the latent space is mapped to the classification space for truth assessment. The extensive experiments across five datasets and six widely used LLMs demonstrate the effectiveness of HD-NDEs, especially, achieving over 14% improvement in AUC-ROC on the True-False dataset compared to state-of-the-art techniques. 

---
# SwitchLingua: The First Large-Scale Multilingual and Multi-Ethnic Code-Switching Dataset 

**Authors**: Peng Xie, Xingyuan Liu, Tsz Wai Chan, Yequan Bie, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.00087)  

**Abstract**: Code-switching (CS) is the alternating use of two or more languages within a conversation or utterance, often influenced by social context and speaker identity. This linguistic phenomenon poses challenges for Automatic Speech Recognition (ASR) systems, which are typically designed for a single language and struggle to handle multilingual inputs. The growing global demand for multilingual applications, including Code-Switching ASR (CSASR), Text-to-Speech (CSTTS), and Cross-Lingual Information Retrieval (CLIR), highlights the inadequacy of existing monolingual datasets.
Although some code-switching datasets exist, most are limited to bilingual mixing within homogeneous ethnic groups, leaving a critical need for a large-scale, diverse benchmark akin to ImageNet in computer vision.
To bridge this gap, we introduce \textbf{LinguaMaster}, a multi-agent collaboration framework specifically designed for efficient and scalable multilingual data synthesis. Leveraging this framework, we curate \textbf{SwitchLingua}, the first large-scale multilingual and multi-ethnic code-switching dataset, including: (1) 420K CS textual samples across 12 languages, and (2) over 80 hours of audio recordings from 174 speakers representing 18 countries/regions and 63 racial/ethnic backgrounds, based on the textual data. This dataset captures rich linguistic and cultural diversity, offering a foundational resource for advancing multilingual and multicultural research. Furthermore, to address the issue that existing ASR evaluation metrics lack sensitivity to code-switching scenarios, we propose the \textbf{Semantic-Aware Error Rate (SAER)}, a novel evaluation metric that incorporates semantic information, providing a more accurate and context-aware assessment of system performance. 

---
# COSMIC: Generalized Refusal Direction Identification in LLM Activations 

**Authors**: Vincent Siu, Nicholas Crispino, Zihao Yu, Sam Pan, Zhun Wang, Yang Liu, Dawn Song, Chenguang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00085)  

**Abstract**: Large Language Models (LLMs) encode behaviors such as refusal within their activation space, yet identifying these behaviors remains a significant challenge. Existing methods often rely on predefined refusal templates detectable in output tokens or require manual analysis. We introduce \textbf{COSMIC} (Cosine Similarity Metrics for Inversion of Concepts), an automated framework for direction selection that identifies viable steering directions and target layers using cosine similarity - entirely independent of model outputs. COSMIC achieves steering performance comparable to prior methods without requiring assumptions about a model's refusal behavior, such as the presence of specific refusal tokens. It reliably identifies refusal directions in adversarial settings and weakly aligned models, and is capable of steering such models toward safer behavior with minimal increase in false refusals, demonstrating robustness across a wide range of alignment conditions. 

---
# Gaussian mixture models as a proxy for interacting language models 

**Authors**: Edward Wang, Tianyu Wang, Avanti Athreya, Vince Lyzinski, Carey E. Priebe  

**Link**: [PDF](https://arxiv.org/pdf/2506.00077)  

**Abstract**: Large language models (LLMs) are a powerful tool with the ability to match human capabilities and behavior in many settings. Retrieval-augmented generation (RAG) further allows LLMs to generate diverse output depending on the contents of their RAG database. This motivates their use in the social sciences to study human behavior between individuals when large-scale experiments are infeasible. However, LLMs depend on complex, computationally expensive algorithms. In this paper, we introduce interacting Gaussian mixture models (GMMs) as an alternative to similar frameworks using LLMs. We compare a simplified model of GMMs to select experimental simulations of LLMs whose updating and response depend on feedback from other LLMs. We find that interacting GMMs capture important features of the dynamics in interacting LLMs, and we investigate key similarities and differences between interacting LLMs and GMMs. We conclude by discussing the benefits of Gaussian mixture models, potential modifications, and future research directions. 

---
# Evaluating the Sensitivity of LLMs to Prior Context 

**Authors**: Robert Hankache, Kingsley Nketia Acheampong, Liang Song, Marek Brynda, Raad Khraishi, Greig A. Cowan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00069)  

**Abstract**: As large language models (LLMs) are increasingly deployed in multi-turn dialogue and other sustained interactive scenarios, it is essential to understand how extended context affects their performance. Popular benchmarks, focusing primarily on single-turn question answering (QA) tasks, fail to capture the effects of multi-turn exchanges. To address this gap, we introduce a novel set of benchmarks that systematically vary the volume and nature of prior context. We evaluate multiple conventional LLMs, including GPT, Claude, and Gemini, across these benchmarks to measure their sensitivity to contextual variations. Our findings reveal that LLM performance on multiple-choice questions can degrade dramatically in multi-turn interactions, with performance drops as large as 73% for certain models. Even highly capable models such as GPT-4o exhibit up to a 32% decrease in accuracy. Notably, the relative performance of larger versus smaller models is not always predictable. Moreover, the strategic placement of the task description within the context can substantially mitigate performance drops, improving the accuracy by as much as a factor of 3.5. These findings underscore the need for robust strategies to design, evaluate, and mitigate context-related sensitivity in LLMs. 

---
# Probing Politico-Economic Bias in Multilingual Large Language Models: A Cultural Analysis of Low-Resource Pakistani Languages 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.00068)  

**Abstract**: Large Language Models (LLMs) are increasingly shaping public discourse, yet their politico-economic biases remain underexamined in non-Western and low-resource multilingual contexts. This paper presents a systematic analysis of political bias in 13 state-of-the-art LLMs across five low-resource languages spoken in Pakistan: Urdu, Punjabi, Sindhi, Balochi, and Pashto. We propose a novel framework that integrates an adapted Political Compass Test (PCT) with a multi-level framing analysis. Our method combines quantitative assessment of political orientation across economic (left-right) and social (libertarian-authoritarian) axes with qualitative analysis of framing through content, style, and emphasis. We further contextualize this analysis by aligning prompts with 11 key socio-political themes relevant to Pakistani society. Our results reveal that LLMs predominantly align with liberal-left values, echoing Western training data influences, but exhibit notable shifts toward authoritarian framing in regional languages, suggesting strong cultural modulation effects. We also identify consistent model-specific bias signatures and language-conditioned variations in ideological expression. These findings show the urgent need for culturally grounded, multilingual bias auditing frameworks. 

---
# You Prefer This One, I Prefer Yours: Using Reference Words is Harder Than Vocabulary Words for Humans and Multimodal Language Models 

**Authors**: Dota Tianai Dong, Yifan Luo, Po-Ya Angela Wang, Asli Ozyurek, Paula Rubio-Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2506.00065)  

**Abstract**: Multimodal language models (MLMs) increasingly communicate in human-like ways, yet their ability to use reference words remains largely overlooked despite their ubiquity in everyday communication. Our study addresses this gap by comparing human and MLM use of three word classes with increasing cognitive demands: vocabulary words, possessive pronouns (`mine' vs `yours'), and demonstrative pronouns (`this one' vs `that one'). Evaluating seven state-of-the-art MLMs against human participants, we observe a clear difficulty hierarchy: while MLMs approach human-level performance on the vocabulary task, they show substantial deficits with possessives and demonstratives. Our analysis reveals these difficulties stem from limitations in perspective-taking and spatial reasoning. Although prompt engineering improved model performance on possessive use, demonstrative use remained well below human-level competence. These findings provide theoretical and empirical evidence that producing grammatical forms requiring pragmatics and social cognition remains a clear challenge in current NLP systems. 

---
# Mis-prompt: Benchmarking Large Language Models for Proactive Error Handling 

**Authors**: Jiayi Zeng, Yizhe Feng, Mengliang He, Wenhui Lei, Wei Zhang, Zeming Liu, Xiaoming Shi, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00064)  

**Abstract**: Large language models (LLMs) have demonstrated significant advancements in error handling. Current error-handling works are performed in a passive manner, with explicit error-handling instructions. However, in real-world scenarios, explicit error-handling instructions are usually unavailable. In this paper, our work identifies this challenge as how to conduct proactive error handling without explicit error handling instructions. To promote further research, this work introduces a new benchmark, termed Mis-prompt, consisting of four evaluation tasks, an error category taxonomy, and a new evaluation dataset. Furthermore, this work analyzes current LLMs' performance on the benchmark, and the experimental results reveal that current LLMs show poor performance on proactive error handling, and SFT on error handling instances improves LLMs' proactive error handling capabilities. The dataset will be publicly available. 

---
# Unraveling SITT: Social Influence Technique Taxonomy and Detection with LLMs 

**Authors**: Wiktoria Mieleszczenko-Kowszewicz, Beata Bajcar, Aleksander Szczęsny, Maciej Markiewicz, Jolanta Babiak, Berenika Dyczek, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.00061)  

**Abstract**: In this work we present the Social Influence Technique Taxonomy (SITT), a comprehensive framework of 58 empirically grounded techniques organized into nine categories, designed to detect subtle forms of social influence in textual content. We also investigate the LLMs ability to identify various forms of social influence. Building on interdisciplinary foundations, we construct the SITT dataset -- a 746-dialogue corpus annotated by 11 experts in Polish and translated into English -- to evaluate the ability of LLMs to identify these techniques. Using a hierarchical multi-label classification setup, we benchmark five LLMs, including GPT-4o, Claude 3.5, Llama-3.1, Mixtral, and PLLuM. Our results show that while some models, notably Claude 3.5, achieved moderate success (F1 score = 0.45 for categories), overall performance of models remains limited, particularly for context-sensitive techniques. The findings demonstrate key limitations in current LLMs' sensitivity to nuanced linguistic cues and underscore the importance of domain-specific fine-tuning. This work contributes a novel resource and evaluation example for understanding how LLMs detect, classify, and potentially replicate strategies of social influence in natural dialogues. 

---
# Enhancing Tool Learning in Large Language Models with Hierarchical Error Checklists 

**Authors**: Yue Cui, Liuyi Yao, Shuchang Tao, Weijie Shi, Yaliang Li, Bolin Ding, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00042)  

**Abstract**: Large language models (LLMs) have significantly advanced natural language processing, particularly through the integration of external tools and APIs. However, their effectiveness is frequently hampered by parameter mis-filling during tool calling. In this paper, we propose the Hierarchical Tool Error Checklist (HiTEC) framework to systematically diagnose and mitigate tool-calling errors without relying on extensive real-world interactions. HiTEC introduces a two-tiered approach: a global error checklist that identifies common, cross-tool issues, and a local error checklist that targets tool-specific and contextual failures. Building on this structure, we propose two deployments: HiTEC-In Context Learning (HiTEC-ICL) and HiTEC-Kahneman-Tversky Optimization (HiTEC-KTO). HiTEC-ICL embeds the global checklist in the initial prompts and leverages a two-round conversational interaction to dynamically refine parameter handling, while HiTEC-KTO generates high-quality negative examples to drive fine-tuning via preference-based optimization. Extensive experiments across five public datasets demonstrate that our framework significantly improves parameter-filling accuracy and tool-calling success rates compared to baseline methods. 

---
# From Mathematical Reasoning to Code: Generalization of Process Reward Models in Test-Time Scaling 

**Authors**: Zhengyu Chen, Yudong Wang, Teng Xiao, Ruochen Zhou, Xuesheng Yang, Wei Wang, Zhifang Sui, Jingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00027)  

**Abstract**: Recent advancements in improving the reasoning capabilities of Large Language Models have underscored the efficacy of Process Reward Models (PRMs) in addressing intermediate errors through structured feedback mechanisms. This study analyzes PRMs from multiple perspectives, including training methodologies, scalability, and generalization capabilities. We investigate the interplay between pre-training and reward model training FLOPs to assess their influence on PRM efficiency and accuracy in complex reasoning tasks. Our analysis reveals a pattern of diminishing returns in performance with increasing PRM scale, highlighting the importance of balancing model size and computational cost. Furthermore, the diversity of training datasets significantly impacts PRM performance, emphasizing the importance of diverse data to enhance both accuracy and efficiency. We further examine test-time scaling strategies, identifying Monte Carlo Tree Search as the most effective method when computational resources are abundant, while Best-of-N Sampling serves as a practical alternative under resource-limited conditions. Notably, our findings indicate that PRMs trained on mathematical datasets exhibit performance comparable to those tailored for code generation, suggesting robust cross-domain generalization. Employing a gradient-based metric, we observe that PRMs exhibit a preference for selecting responses with similar underlying patterns, further informing their optimization. 

---
# Scaling Physical Reasoning with the PHYSICS Dataset 

**Authors**: Shenghe Zheng, Qianjia Cheng, Junchi Yao, Mengsong Wu, haonan he, Ning Ding, Yu Cheng, Shuyue Hu, Lei Bai, Dongzhan Zhou, Ganqu Cui, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.00022)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress on advanced reasoning tasks such as mathematics and coding competitions. Meanwhile, physics, despite being both reasoning-intensive and essential to real-world understanding, received limited academic and industrial attention. This paper introduces PHYSICS, a dataset containing 16,568 high-quality physics problems spanning subjects and difficulty levels, to facilitate this issue. Specifically, PHYSICS is curated with exercises from over 100 textbooks through a carefully designed pipeline for quality control. It covers five major physics domains: Mechanics, Electromagnetism, Thermodynamics, Optics, and Modern Physics. It also spans a wide range of difficulty levels, from high school to graduate-level physics courses. To utilize the data for improving and evaluating the model's physical reasoning capabilities, we split the dataset into training and test sets, and provide reasoning paths generated by powerful reasoning models for the training data to facilitate model training. In addition, for the evaluation part, we find that existing evaluation frameworks exhibit biases in aspects such as units, simplification, and precision in physics domain. To balance efficiency and accuracy, we introduce a Rule+Model evaluation framework tailored to physics problems. Our evaluations on current state-of-the-art open-source and proprietary models highlight the limitations of current models in handling physics-related tasks. We hope that our dataset and evaluation methodology will jointly advance the development of LLMs in the field of physics. 

---
# Amadeus-Verbo Technical Report: The powerful Qwen2.5 family models trained in Portuguese 

**Authors**: William Alberto Cruz-Castañeda, Marcellus Amadeus  

**Link**: [PDF](https://arxiv.org/pdf/2506.00019)  

**Abstract**: This report introduces the experience of developing Amadeus Verbo, a family of large language models for Brazilian Portuguese. To handle diverse use cases, Amadeus Verbo includes base-tuned, merged, and instruction-tuned models in sizes of 0.5B, 1.5B, 3B, 7B, 14B, 32B, and 72B parameters. Thus, the main objective is to show how easy it is to fine-tune foundation models to democratize the open-source development of Brazilian Portuguese LLMs when data and resources are available. Amadeus-Verbo family models are all available at HuggingFace at this https URL. 

---
# Dual-Process Image Generation 

**Authors**: Grace Luo, Jonathan Granskog, Aleksander Holynski, Trevor Darrell  

**Link**: [PDF](https://arxiv.org/pdf/2506.01955)  

**Abstract**: Prior methods for controlling image generation are limited in their ability to be taught new tasks. In contrast, vision-language models, or VLMs, can learn tasks in-context and produce the correct outputs for a given input. We propose a dual-process distillation scheme that allows feed-forward image generators to learn new tasks from deliberative VLMs. Our scheme uses a VLM to rate the generated images and backpropagates this gradient to update the weights of the image generator. Our general framework enables a wide variety of new control tasks through the same text-and-image based interface. We showcase a handful of applications of this technique for different types of control signals, such as commonsense inferences and visual prompts. With our method, users can implement multimodal controls for properties such as color palette, line weight, horizon position, and relative depth within a matter of minutes. Project page: this https URL. 

---
# Large language models can learn and generalize steganographic chain-of-thought under process supervision 

**Authors**: Joey Skaf, Luis Ibanez-Lissen, Robert McCarthy, Connor Watts, Vasil Georgiv, Hannes Whittingham, Lorena Gonzalez-Manzano, David Lindner, Cameron Tice, Edward James Young, Puria Radmard  

**Link**: [PDF](https://arxiv.org/pdf/2506.01926)  

**Abstract**: Chain-of-thought (CoT) reasoning not only enhances large language model performance but also provides critical insights into decision-making processes, marking it as a useful tool for monitoring model intent and planning. By proactively preventing models from acting on CoT indicating misaligned or harmful intent, CoT monitoring can be used to reduce risks associated with deploying models. However, developers may be incentivized to train away the appearance of harmful intent from CoT traces, by either customer preferences or regulatory requirements. Recent works have shown that banning mention of a specific example of reward hacking, which may be done either to make CoT presentable to users or as a naive attempt to prevent the behavior, causes obfuscation of the undesired reasoning traces but the persistence of the undesired behavior. Such obfuscation threatens the reliability of CoT monitoring. However, obfuscation of reasoning can be due to its internalization to latent space computation, or its encoding within the CoT. Here, we provide an extension to these results. First, we show that penalizing the use of specific strings within load-bearing reasoning traces causes models to substitute alternative strings. Crucially, this does not alter the underlying method by which the model performs the task, demonstrating that the model can learn to steganographically encode its reasoning. We further demonstrate that models can generalize an encoding scheme. When the penalized strings belong to an overarching class, the model learns not only to substitute strings seen in training, but also develops a general encoding scheme for all members of the class which it can apply to held-out testing strings. 

---
# Enhancing Biomedical Multi-modal Representation Learning with Multi-scale Pre-training and Perturbed Report Discrimination 

**Authors**: Xinliu Zhong, Kayhan Batmanghelich, Li Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01902)  

**Abstract**: Vision-language models pre-trained on large scale of unlabeled biomedical images and associated reports learn generalizable semantic representations. These multi-modal representations can benefit various downstream tasks in the biomedical domain. Contrastive learning is widely used to pre-train vision-language models for general natural images and associated captions. Despite its popularity, we found biomedical texts have complex and domain-specific semantics that are often neglected by common contrastive methods. To address this issue, we propose a novel method, perturbed report discrimination, for pre-train biomedical vision-language models. First, we curate a set of text perturbation methods that keep the same words, but disrupt the semantic structure of the sentence. Next, we apply different types of perturbation to reports, and use the model to distinguish the original report from the perturbed ones given the associated image. Parallel to this, we enhance the sensitivity of our method to higher level of granularity for both modalities by contrasting attention-weighted image sub-regions and sub-words in the image-text pairs. We conduct extensive experiments on multiple downstream tasks, and our method outperforms strong baseline methods. The results demonstrate that our approach learns more semantic meaningful and robust multi-modal representations. 

---
# WHEN TO ACT, WHEN TO WAIT: Modeling Structural Trajectories for Intent Triggerability in Task-Oriented Dialogue 

**Authors**: Yaoyao Qian, Jindan Huang, Yuanli Wang, Simon Yu, Kyrie Zhixuan Zhou, Jiayuan Mao, Mingfu Liang, Hanhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.01881)  

**Abstract**: Task-oriented dialogue systems often face difficulties when user utterances seem semantically complete but lack necessary structural information for appropriate system action. This arises because users frequently do not fully understand their own needs, while systems require precise intent definitions. Current LLM-based agents cannot effectively distinguish between linguistically complete and contextually triggerable expressions, lacking frameworks for collaborative intent formation. We present STORM, a framework modeling asymmetric information dynamics through conversations between UserLLM (full internal access) and AgentLLM (observable behavior only). STORM produces annotated corpora capturing expression trajectories and latent cognitive transitions, enabling systematic analysis of collaborative understanding development. Our contributions include: (1) formalizing asymmetric information processing in dialogue systems; (2) modeling intent formation tracking collaborative understanding evolution; and (3) evaluation metrics measuring internal cognitive improvements alongside task performance. Experiments across four language models reveal that moderate uncertainty (40-60%) can outperform complete transparency in certain scenarios, with model-specific patterns suggesting reconsideration of optimal information completeness in human-AI collaboration. These findings contribute to understanding asymmetric reasoning dynamics and inform uncertainty-calibrated dialogue system design. 

---
# When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR 

**Authors**: Dayoon Ko, Jinyoung Kim, Sohyeon Kim, Jinhyuk Kim, Jaehoon Lee, Seonghak Song, Minyoung Lee, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.01877)  

**Abstract**: Dense retrievers encode texts into embeddings to efficiently retrieve relevant documents from large databases in response to user queries. However, real-world corpora continually evolve, leading to a shift from the original training distribution of the retriever. Without timely updates or retraining, indexing newly emerging documents can degrade retrieval performance for future queries. Thus, identifying when a dense retriever requires an update is critical for maintaining robust retrieval systems. In this paper, we propose a novel task of predicting whether a corpus is out-of-distribution (OOD) relative to a dense retriever before indexing. Addressing this task allows us to proactively manage retriever updates, preventing potential retrieval failures. We introduce GradNormIR, an unsupervised approach that leverages gradient norms to detect OOD corpora effectively. Experiments on the BEIR benchmark demonstrate that GradNormIR enables timely updates of dense retrievers in evolving document collections, significantly enhancing retrieval robustness and efficiency. 

---
# Unified Scaling Laws for Compressed Representations 

**Authors**: Andrei Panferov, Alexandra Volkova, Ionut-Vlad Modoranu, Vage Egiazarian, Mher Safaryan, Dan Alistarh  

**Link**: [PDF](https://arxiv.org/pdf/2506.01863)  

**Abstract**: Scaling laws have shaped recent advances in machine learning by enabling predictable scaling of model performance based on model size, computation, and data volume. Concurrently, the rise in computational cost for AI has motivated model compression techniques, notably quantization and sparsification, which have emerged to mitigate the steep computational demands associated with large-scale training and inference. This paper investigates the interplay between scaling laws and compression formats, exploring whether a unified scaling framework can accurately predict model performance when training occurs over various compressed representations, such as sparse, scalar-quantized, sparse-quantized or even vector-quantized formats. Our key contributions include validating a general scaling law formulation and showing that it is applicable both individually but also composably across compression types. Based on this, our main finding is demonstrating both theoretically and empirically that there exists a simple "capacity" metric -- based on the representation's ability to fit random Gaussian data -- which can robustly predict parameter efficiency across multiple compressed representations. On the practical side, we extend our formulation to directly compare the accuracy potential of different compressed formats, and to derive better algorithms for training over sparse-quantized formats. 

---
# Datasheets Aren't Enough: DataRubrics for Automated Quality Metrics and Accountability 

**Authors**: Genta Indra Winata, David Anugraha, Emmy Liu, Alham Fikri Aji, Shou-Yi Hung, Aditya Parashar, Patrick Amadeus Irawan, Ruochen Zhang, Zheng-Xin Yong, Jan Christian Blaise Cruz, Niklas Muennighoff, Seungone Kim, Hanyang Zhao, Sudipta Kar, Kezia Erina Suryoraharjo, M. Farid Adilazuarda, En-Shiun Annie Lee, Ayu Purwarianti, Derry Tanti Wijaya, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.01789)  

**Abstract**: High-quality datasets are fundamental to training and evaluating machine learning models, yet their creation-especially with accurate human annotations-remains a significant challenge. Many dataset paper submissions lack originality, diversity, or rigorous quality control, and these shortcomings are often overlooked during peer review. Submissions also frequently omit essential details about dataset construction and properties. While existing tools such as datasheets aim to promote transparency, they are largely descriptive and do not provide standardized, measurable methods for evaluating data quality. Similarly, metadata requirements at conferences promote accountability but are inconsistently enforced. To address these limitations, this position paper advocates for the integration of systematic, rubric-based evaluation metrics into the dataset review process-particularly as submission volumes continue to grow. We also explore scalable, cost-effective methods for synthetic data generation, including dedicated tools and LLM-as-a-judge approaches, to support more efficient evaluation. As a call to action, we introduce DataRubrics, a structured framework for assessing the quality of both human- and model-generated datasets. Leveraging recent advances in LLM-based evaluation, DataRubrics offers a reproducible, scalable, and actionable solution for dataset quality assessment, enabling both authors and reviewers to uphold higher standards in data-centric research. We also release code to support reproducibility of LLM-based evaluations at this https URL. 

---
# Self-Challenging Language Model Agents 

**Authors**: Yifei Zhou, Sergey Levine, Jason Weston, Xian Li, Sainbayar Sukhbaatar  

**Link**: [PDF](https://arxiv.org/pdf/2506.01716)  

**Abstract**: Large language models are quickly becoming the foundation for intelligent agents that are capable of using tools. However, training such agents is challenging because it requires human creation and annotation of a diverse set of tasks, tools, and evaluation criteria. In this paper, we propose the Self-Challenging framework for training an agent on high-quality tasks that are generated by itself. The agent first plays the role of challenger and generates a task after interacting with the given tools. The tasks take the form of a novel general class of problems termed Code-as-Task, which are defined by an instruction, a verification function and solution and failure cases which serve as tests, allowing to filter only for high-quality tasks. The agent then takes an executor role and trains on those tasks with reinforcement learning using the evaluation feedback as a reward. Evaluation on two existing multi-turn tool-use agent benchmarks, M3ToolEval and TauBench, shows the Self-Challenging framework achieves over a two-fold improvement in Llama-3.1-8B-Instruct, despite using only self-generated training data. 

---
# Respond Beyond Language: A Benchmark for Video Generation in Response to Realistic User Intents 

**Authors**: Shuting Wang, Yunqi Liu, Zixin Yang, Ning Hu, Zhicheng Dou, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.01689)  

**Abstract**: Querying generative AI models, e.g., large language models (LLMs), has become a prevalent method for information acquisition. However, existing query-answer datasets primarily focus on textual responses, making it challenging to address complex user queries that require visual demonstrations or explanations for better understanding. To bridge this gap, we construct a benchmark, RealVideoQuest, designed to evaluate the abilities of text-to-video (T2V) models in answering real-world, visually grounded queries. It identifies 7.5K real user queries with video response intents from Chatbot-Arena and builds 4.5K high-quality query-video pairs through a multistage video retrieval and refinement process. We further develop a multi-angle evaluation system to assess the quality of generated video answers. Experiments indicate that current T2V models struggle with effectively addressing real user queries, pointing to key challenges and future research opportunities in multimodal AI. 

---
# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion 

**Authors**: Sunkyung Lee, Minjin Choi, Eunseong Choi, Hye-young Kim, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.01673)  

**Abstract**: Generative recommendation is an emerging paradigm that leverages the extensive knowledge of large language models by formulating recommendations into a text-to-text generation task. However, existing studies face two key limitations in (i) incorporating implicit item relationships and (ii) utilizing rich yet lengthy item information. To address these challenges, we propose a Generative Recommender via semantic-Aware Multi-granular late fusion (GRAM), introducing two synergistic innovations. First, we design semantic-to-lexical translation to encode implicit hierarchical and collaborative item relationships into the vocabulary space of LLMs. Second, we present multi-granular late fusion to integrate rich semantics efficiently with minimal information loss. It employs separate encoders for multi-granular prompts, delaying the fusion until the decoding stage. Experiments on four benchmark datasets show that GRAM outperforms eight state-of-the-art generative recommendation models, achieving significant improvements of 11.5-16.0% in Recall@5 and 5.3-13.6% in NDCG@5. The source code is available at this https URL. 

---
# AIMSCheck: Leveraging LLMs for AI-Assisted Review of Modern Slavery Statements Across Jurisdictions 

**Authors**: Adriana Eufrosina Bora, Akshatha Arodi, Duoyi Zhang, Jordan Bannister, Mirko Bronzi, Arsene Fansi Tchango, Md Abul Bashar, Richi Nayak, Kerrie Mengersen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01671)  

**Abstract**: Modern Slavery Acts mandate that corporations disclose their efforts to combat modern slavery, aiming to enhance transparency and strengthen practices for its eradication. However, verifying these statements remains challenging due to their complex, diversified language and the sheer number of statements that must be reviewed. The development of NLP tools to assist in this task is also difficult due to a scarcity of annotated data. Furthermore, as modern slavery transparency legislation has been introduced in several countries, the generalizability of such tools across legal jurisdictions must be studied. To address these challenges, we work with domain experts to make two key contributions. First, we present this http URL and this http URL, newly annotated datasets from the UK and Canada to enable cross-jurisdictional evaluation. Second, we introduce AIMSCheck, an end-to-end framework for compliance validation. AIMSCheck decomposes the compliance assessment task into three levels, enhancing interpretability and practical applicability. Our experiments show that models trained on an Australian dataset generalize well across UK and Canadian jurisdictions, demonstrating the potential for broader application in compliance monitoring. We release the benchmark datasets and AIMSCheck to the public to advance AI-adoption in compliance assessment and drive further research in this field. 

---
# EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation 

**Authors**: Bingqian Lin, Yunshuang Nie, Khun Loun Zai, Ziming Wei, Mingfei Han, Rongtao Xu, Minzhe Niu, Jianhua Han, Liang Lin, Cewu Lu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01551)  

**Abstract**: Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at this https URL. 

---
# LinearVC: Linear transformations of self-supervised features through the lens of voice conversion 

**Authors**: Herman Kamper, Benjamin van Niekerk, Julian Zaïdi, Marc-André Carbonneau  

**Link**: [PDF](https://arxiv.org/pdf/2506.01510)  

**Abstract**: We introduce LinearVC, a simple voice conversion method that sheds light on the structure of self-supervised representations. First, we show that simple linear transformations of self-supervised features effectively convert voices. Next, we probe the geometry of the feature space by constraining the set of allowed transformations. We find that just rotating the features is sufficient for high-quality voice conversion. This suggests that content information is embedded in a low-dimensional subspace which can be linearly transformed to produce a target voice. To validate this hypothesis, we finally propose a method that explicitly factorizes content and speaker information using singular value decomposition; the resulting linear projection with a rank of just 100 gives competitive conversion results. Our work has implications for both practical voice conversion and a broader understanding of self-supervised speech representations. Samples and code: this https URL. 

---
# MUDI: A Multimodal Biomedical Dataset for Understanding Pharmacodynamic Drug-Drug Interactions 

**Authors**: Tung-Lam Ngo, Ba-Hoang Tran, Duy-Cat Can, Trung-Hieu Do, Oliver Y. Chén, Hoang-Quynh Le  

**Link**: [PDF](https://arxiv.org/pdf/2506.01478)  

**Abstract**: Understanding the interaction between different drugs (drug-drug interaction or DDI) is critical for ensuring patient safety and optimizing therapeutic outcomes. Existing DDI datasets primarily focus on textual information, overlooking multimodal data that reflect complex drug mechanisms. In this paper, we (1) introduce MUDI, a large-scale Multimodal biomedical dataset for Understanding pharmacodynamic Drug-drug Interactions, and (2) benchmark learning methods to study it. In brief, MUDI provides a comprehensive multimodal representation of drugs by combining pharmacological text, chemical formulas, molecular structure graphs, and images across 310,532 annotated drug pairs labeled as Synergism, Antagonism, or New Effect. Crucially, to effectively evaluate machine-learning based generalization, MUDI consists of unseen drug pairs in the test set. We evaluate benchmark models using both late fusion voting and intermediate fusion strategies. All data, annotations, evaluation scripts, and baselines are released under an open research license. 

---
# PGPO: Enhancing Agent Reasoning via Pseudocode-style Planning Guided Preference Optimization 

**Authors**: Zouying Cao, Runze Wang, Yifei Yang, Xinbei Ma, Xiaoyong Zhu, Bo Zheng, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.01475)  

**Abstract**: Large Language Model (LLM) agents have demonstrated impressive capabilities in handling complex interactive problems. Existing LLM agents mainly generate natural language plans to guide reasoning, which is verbose and inefficient. NL plans are also tailored to specific tasks and restrict agents' ability to generalize across similar tasks. To this end, we explore pseudocode-style plans (P-code Plan) to capture the structural logic of reasoning. We find that P-code Plan empowers LLM agents with stronger generalization ability and more efficiency. Inspired by this finding, we propose a pseudocode-style Planning Guided Preference Optimization method called PGPO for effective agent learning. With two planning-oriented rewards, PGPO further enhances LLM agents' ability to generate high-quality P-code Plans and subsequent reasoning. Experiments show that PGPO achieves superior performance on representative agent benchmarks and outperforms the current leading baselines. Analyses reveal the advantage of PGPO in reducing action errors and omissions during reasoning. 

---
# Incentivizing Reasoning for Advanced Instruction-Following of Large Language Models 

**Authors**: Yulei Qin, Gang Li, Zongyi Li, Zihan Xu, Yuchen Shi, Zhekai Lin, Xiao Cui, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01413)  

**Abstract**: Existing large language models (LLMs) face challenges of following complex instructions, especially when multiple constraints are present and organized in paralleling, chaining, and branching structures. One intuitive solution, namely chain-of-thought (CoT), is expected to universally improve capabilities of LLMs. However, we find that the vanilla CoT exerts a negative impact on performance due to its superficial reasoning pattern of simply paraphrasing the instructions. It fails to peel back the compositions of constraints for identifying their relationship across hierarchies of types and dimensions. To this end, we propose a systematic method to boost LLMs in dealing with complex instructions via incentivizing reasoning for test-time compute scaling. First, we stem from the decomposition of complex instructions under existing taxonomies and propose a reproducible data acquisition method. Second, we exploit reinforcement learning (RL) with verifiable rule-centric reward signals to cultivate reasoning specifically for instruction following. We address the shallow, non-essential nature of reasoning under complex instructions via sample-wise contrast for superior CoT enforcement. We also exploit behavior cloning of experts to facilitate steady distribution shift from fast-thinking LLMs to skillful reasoners. Extensive evaluations on seven comprehensive benchmarks confirm the validity of the proposed method, where a 1.5B LLM achieves 11.74% gains with performance comparable to a 8B LLM. Codes and data are available at this https URL. 

---
# AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning 

**Authors**: Zhong Zhang, Yaxi Lu, Yikun Fu, Yupeng Huo, Shenzhi Yang, Yesai Wu, Han Si, Xin Cong, Haotian Chen, Yankai Lin, Jie Xie, Wei Zhou, Wang Xu, Yuanheng Zhang, Zhou Su, Zhongwu Zhai, Xiaoming Liu, Yudong Mei, Jianming Xu, Hongyan Tian, Chongyi Wang, Chi Chen, Yuan Yao, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.01391)  

**Abstract**: The recent progress of large language model agents has opened new possibilities for automating tasks through graphical user interfaces (GUIs), especially in mobile environments where intelligent interaction can greatly enhance usability. However, practical deployment of such agents remains constrained by several key challenges. Existing training data is often noisy and lack semantic diversity, which hinders the learning of precise grounding and planning. Models trained purely by imitation tend to overfit to seen interface patterns and fail to generalize in unfamiliar scenarios. Moreover, most prior work focuses on English interfaces while overlooks the growing diversity of non-English applications such as those in the Chinese mobile ecosystem. In this work, we present AgentCPM-GUI, an 8B-parameter GUI agent built for robust and efficient on-device GUI interaction. Our training pipeline includes grounding-aware pre-training to enhance perception, supervised fine-tuning on high-quality Chinese and English trajectories to imitate human-like actions, and reinforcement fine-tuning with GRPO to improve reasoning capability. We also introduce a compact action space that reduces output length and supports low-latency execution on mobile devices. AgentCPM-GUI achieves state-of-the-art performance on five public benchmarks and a new Chinese GUI benchmark called CAGUI, reaching $96.9\%$ Type-Match and $91.3\%$ Exact-Match. To facilitate reproducibility and further research, we publicly release all code, model checkpoint, and evaluation data. 

---
# AI Scientists Fail Without Strong Implementation Capability 

**Authors**: Minjun Zhu, Qiujie Xie, Yixuan Weng, Jian Wu, Zhen Lin, Linyi Yang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.01372)  

**Abstract**: The emergence of Artificial Intelligence (AI) Scientist represents a paradigm shift in scientific discovery, with large language models (LLMs) taking the lead as the primary executor in the entire scientific workflow from idea generation to experiment implementation. Recent AI Scientist studies demonstrate sufficient capabilities for independent scientific discovery, with the generated research reports gaining acceptance at the ICLR 2025 workshop and ACL 2025, arguing that a human-level AI Scientist, capable of uncovering phenomena previously unknown to humans, may be imminent. Despite this substantial progress, AI Scientist has yet to produce a groundbreaking achievement in the domain of computer science on par with automated scientific tools. Based on extensive quantitative evidence from existing benchmarks in complex engineering tasks and a systematic evaluation assess 28 research papers generated by five advanced AI Scientist systems, we argue that \textbf{the fundamental bottleneck for AI Scientists lies in their capability to execute the requisite verification procedures.} Current AI Scientist systems lack the execution capabilities needed to execute rigorous experiments and produce high-quality scientific papers. To better illustrate the root cause of this \textbf{implementation gap}, we provide an in-depth discussion on the fundamental limitations of AI Scientist. This position paper aims to call for the participants in the community to bridge the implementation gap. 

---
# Attention Is Not Always the Answer: Optimizing Voice Activity Detection with Simple Feature Fusion 

**Authors**: Kumud Tripathi, Chowdam Venkata Kumar, Pankaj Wasnik  

**Link**: [PDF](https://arxiv.org/pdf/2506.01365)  

**Abstract**: Voice Activity Detection (VAD) plays a key role in speech processing, often utilizing hand-crafted or neural features. This study examines the effectiveness of Mel-Frequency Cepstral Coefficients (MFCCs) and pre-trained model (PTM) features, including wav2vec 2.0, HuBERT, WavLM, UniSpeech, MMS, and Whisper. We propose FusionVAD, a unified framework that combines both feature types using three fusion strategies: concatenation, addition, and cross-attention (CA). Experimental results reveal that simple fusion techniques, particularly addition, outperform CA in both accuracy and efficiency. Fusion-based models consistently surpass single-feature models, highlighting the complementary nature of MFCCs and PTM features. Notably, our best-performing fusion model exceeds the state-of-the-art Pyannote across multiple datasets, achieving an absolute average improvement of 2.04%. These results confirm that simple feature fusion enhances VAD robustness while maintaining computational efficiency. 

---
# An Empirical Study of Group Conformity in Multi-Agent Systems 

**Authors**: Min Choi, Keonwoo Kim, Sungwon Chae, Sangyeob Baek  

**Link**: [PDF](https://arxiv.org/pdf/2506.01332)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled multi-agent systems that simulate real-world interactions with near-human reasoning. While previous studies have extensively examined biases related to protected attributes such as race, the emergence and propagation of biases on socially contentious issues in multi-agent LLM interactions remain underexplored. This study explores how LLM agents shape public opinion through debates on five contentious topics. By simulating over 2,500 debates, we analyze how initially neutral agents, assigned a centrist disposition, adopt specific stances over time. Statistical analyses reveal significant group conformity mirroring human behavior; LLM agents tend to align with numerically dominant groups or more intelligent agents, exerting a greater influence. These findings underscore the crucial role of agent intelligence in shaping discourse and highlight the risks of bias amplification in online interactions. Our results emphasize the need for policy measures that promote diversity and transparency in LLM-generated discussions to mitigate the risks of bias propagation within anonymous online environments. 

---
# Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner 

**Authors**: Chunhui Zhang, Zhongyu Ouyang, Kwonjoon Lee, Nakul Agarwal, Sean Dae Houlihan, Soroush Vosoughi, Shao-Yuan Lo  

**Link**: [PDF](https://arxiv.org/pdf/2506.01301)  

**Abstract**: Theory-of-Mind (ToM) enables humans to infer mental states-such as beliefs, desires, and intentions-forming the foundation of social cognition. However, existing computational ToM methods rely on structured workflows with ToM-specific priors or deep model fine-tuning, which struggle with scalability in multimodal environments and fail to generalize as task complexity increases. To address these limitations, we propose a scalable Bayesian ToM planner that decomposes ToM reasoning into stepwise Bayesian updates. Our framework introduces weak-to-strong control, allowing smaller language models (LMs) to specialize in ToM-specific likelihood estimation and transfer their reasoning behaviors to larger LMs (7B to 405B) for integration with social and world knowledge. This synergistic approach aligns large-model inference of human mental states with Bayesian principles. Extensive experiments show that our method achieves a 4.6% accuracy improvement over state-of-the-art techniques on multimodal ToM benchmarks, including challenging unseen scenarios, thereby establishing a new standard for modeling human mental states in complex environments. 

---
# Abstractive Visual Understanding of Multi-modal Structured Knowledge: A New Perspective for MLLM Evaluation 

**Authors**: Yichi Zhang, Zhuo Chen, Lingbing Guo, Yajing Xu, Min Zhang, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.01293)  

**Abstract**: Multi-modal large language models (MLLMs) incorporate heterogeneous modalities into LLMs, enabling a comprehensive understanding of diverse scenarios and objects. Despite the proliferation of evaluation benchmarks and leaderboards for MLLMs, they predominantly overlook the critical capacity of MLLMs to comprehend world knowledge with structured abstractions that appear in visual form. To address this gap, we propose a novel evaluation paradigm and devise M3STR, an innovative benchmark grounded in the Multi-Modal Map for STRuctured understanding. This benchmark leverages multi-modal knowledge graphs to synthesize images encapsulating subgraph architectures enriched with multi-modal entities. M3STR necessitates that MLLMs not only recognize the multi-modal entities within the visual inputs but also decipher intricate relational topologies among them. We delineate the benchmark's statistical profiles and automated construction pipeline, accompanied by an extensive empirical analysis of 26 state-of-the-art MLLMs. Our findings reveal persistent deficiencies in processing abstractive visual information with structured knowledge, thereby charting a pivotal trajectory for advancing MLLMs' holistic reasoning capacities. Our code and data are released at this https URL 

---
# Confidence intervals for forced alignment boundaries using model ensembles 

**Authors**: Matthew C. Kelley  

**Link**: [PDF](https://arxiv.org/pdf/2506.01256)  

**Abstract**: Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only a single estimate of a boundary. The present project introduces a method of deriving confidence intervals for these boundaries using a neural network ensemble technique. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each model. The alignment ensemble is then used to place the boundary at the median of the boundaries in the ensemble, and 97.85% confidence intervals are constructed using order statistics. On the Buckeye and TIMIT corpora, the ensemble boundaries show a slight improvement over using just a single model. The confidence intervals are incorporated into Praat TextGrids using a point tier, and they are also output as a table for researchers to analyze separately as diagnostics or to incorporate uncertainty into their analyses. 

---
# Earley-Driven Dynamic Pruning for Efficient Structured Decoding 

**Authors**: Xintong Sun, Chi Wei, Minghao Tian, Shiwen Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.01151)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, yet ensuring their outputs conform to strict structural or grammatical constraints remains challenging, which is critical in function calls and domain-specific language (DSL) generation. Constrained decoding with context-free grammar is a flexible approach to guarantee LLMs' adherence to a specific format by dynamically building a token logits mask. However, creating this mask requires checking the validity of all tokens in the LLM vocabulary at every decoding step, which often incurs significant overheads in existing constrained decoding engines. To address this challenge, we propose $\textbf{ZapFormat}$, a novel $\textbf{dynamic pruning}$ strategy based on the Earley algorithm that identifies and eliminates invalid or redundant Earley states in real-time, significantly reducing memory occupation of the Earley algorithm's states. This further enables us to use a state cache to speed up structured generations on a large number of queries. We implemented ZapFormat in a new constrained decoding engine called Formatron which also incorporates existing optimizations. Through comprehensive experiments on structured generation tasks, including JSON generation, JSON Schema validation, and semantic parsing, we demonstrate that Formatron not only $\textbf{consistently maintains}$ high-precision compliant outputs but also achieves $\textbf{significant improvements}$ in inference speed up to 2x compared to state-of-the-art implementations. More importantly, Formatron is generally applicable across various LLM architectures. We release Formatron as open source at this https URL. 

---
# Attention Retrieves, MLP Memorizes: Disentangling Trainable Components in the Transformer 

**Authors**: Yihe Dong, Lorenzo Noci, Mikhail Khodak, Mufan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.01115)  

**Abstract**: The Transformer architecture is central to the success of modern Large Language Models (LLMs), in part due to its surprising ability to perform a wide range of algorithmic tasks -- including mathematical reasoning, memorization, and retrieval -- using only gradient-based training on next-token prediction. While the core component of a Transformer is the self-attention mechanism, we question how much, and which aspects, of the performance gains can be attributed to it. To this end, we compare standard Transformers to variants in which either the multi-layer perceptron (MLP) layers or the attention projectors (queries and keys) are frozen at initialization. To further isolate the contribution of attention, we introduce MixiT -- the Mixing Transformer -- a simplified, principled model in which the attention coefficients are entirely random and fixed at initialization, eliminating any input-dependent computation or learning in attention. Surprisingly, we find that MixiT matches the performance of fully trained Transformers on various algorithmic tasks, especially those involving basic arithmetic or focusing heavily on memorization. For retrieval-based tasks, we observe that having input-dependent attention coefficients is consistently beneficial, while MixiT underperforms. We attribute this failure to its inability to form specialized circuits such as induction heads -- a specific circuit known to be crucial for learning and exploiting repeating patterns in input sequences. Even more interestingly, we find that attention with frozen key and query projectors is not only able to form induction heads, but can also perform competitively on language modeling. Our results underscore the importance of architectural heterogeneity, where distinct components contribute complementary inductive biases crucial for solving different classes of tasks. 

---
# Simple Prompt Injection Attacks Can Leak Personal Data Observed by LLM Agents During Task Execution 

**Authors**: Meysam Alizadeh, Zeynab Samei, Daria Stetsenko, Fabrizio Gilardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.01055)  

**Abstract**: Previous benchmarks on prompt injection in large language models (LLMs) have primarily focused on generic tasks and attacks, offering limited insights into more complex threats like data exfiltration. This paper examines how prompt injection can cause tool-calling agents to leak personal data observed during task execution. Using a fictitious banking agent, we develop data flow-based attacks and integrate them into AgentDojo, a recent benchmark for agentic security. To enhance its scope, we also create a richer synthetic dataset of human-AI banking conversations. In 16 user tasks from AgentDojo, LLMs show a 15-50 percentage point drop in utility under attack, with average attack success rates (ASR) around 20 percent; some defenses reduce ASR to zero. Most LLMs, even when successfully tricked by the attack, avoid leaking highly sensitive data like passwords, likely due to safety alignments, but they remain vulnerable to disclosing other personal data. The likelihood of password leakage increases when a password is requested along with one or two additional personal details. In an extended evaluation across 48 tasks, the average ASR is around 15 percent, with no built-in AgentDojo defense fully preventing leakage. Tasks involving data extraction or authorization workflows, which closely resemble the structure of exfiltration attacks, exhibit the highest ASRs, highlighting the interaction between task type, agent performance, and defense efficacy. 

---
# Bridging the Gap: From Ad-hoc to Proactive Search in Conversations 

**Authors**: Chuan Meng, Francesco Tonolini, Fengran Mo, Nikolaos Aletras, Emine Yilmaz, Gabriella Kazai  

**Link**: [PDF](https://arxiv.org/pdf/2506.00983)  

**Abstract**: Proactive search in conversations (PSC) aims to reduce user effort in formulating explicit queries by proactively retrieving useful relevant information given conversational context. Previous work in PSC either directly uses this context as input to off-the-shelf ad-hoc retrievers or further fine-tunes them on PSC data. However, ad-hoc retrievers are pre-trained on short and concise queries, while the PSC input is longer and noisier. This input mismatch between ad-hoc search and PSC limits retrieval quality. While fine-tuning on PSC data helps, its benefits remain constrained by this input gap. In this work, we propose Conv2Query, a novel conversation-to-query framework that adapts ad-hoc retrievers to PSC by bridging the input gap between ad-hoc search and PSC. Conv2Query maps conversational context into ad-hoc queries, which can either be used as input for off-the-shelf ad-hoc retrievers or for further fine-tuning on PSC data. Extensive experiments on two PSC datasets show that Conv2Query significantly improves ad-hoc retrievers' performance, both when used directly and after fine-tuning on PSC. 

---
# Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues 

**Authors**: Youngmin Kim, Jiwan Chung, Jisoo Kim, Sunghyun Lee, Sangkyu Lee, Junhyeok Kim, Cheoljong Yang, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00958)  

**Abstract**: Nonverbal communication is integral to human interaction, with gestures, facial expressions, and body language conveying critical aspects of intent and emotion. However, existing large language models (LLMs) fail to effectively incorporate these nonverbal elements, limiting their capacity to create fully immersive conversational experiences. We introduce MARS, a multimodal language model designed to understand and generate nonverbal cues alongside text, bridging this gap in conversational AI. Our key innovation is VENUS, a large-scale dataset comprising annotated videos with time-aligned text, facial expressions, and body language. Leveraging VENUS, we train MARS with a next-token prediction objective, combining text with vector-quantized nonverbal representations to achieve multimodal understanding and generation within a unified framework. Based on various analyses of the VENUS datasets, we validate its substantial scale and high effectiveness. Our quantitative and qualitative results demonstrate that MARS successfully generates text and nonverbal languages, corresponding to conversational input. 

---
# Aligning VLM Assistants with Personalized Situated Cognition 

**Authors**: Yongqi Li, Shen Zhou, Xiaohu Li, Xin Miao, Jintao Wen, Mayi Xu, Jianhao Chen, Birong Pan, Hankun Kang, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.00930)  

**Abstract**: Vision-language models (VLMs) aligned with general human objectives, such as being harmless and hallucination-free, have become valuable assistants of humans in managing visual tasks. However, people with diversified backgrounds have different cognition even in the same situation. Consequently, they may have personalized expectations for VLM assistants. This highlights the urgent need to align VLM assistants with personalized situated cognition for real-world assistance. To study this problem, we first simplify it by characterizing individuals based on the sociological concept of Role-Set. Then, we propose to evaluate the individuals' actions to examine whether the personalized alignment is achieved. Further, we construct a benchmark named PCogAlignBench, which includes 18k instances and 20 individuals with different Role-Sets. Finally, we present a framework called PCogAlign, which constructs a cognition-aware and action-based reward model for personalized alignment. Experimental results and human evaluations demonstrate the reliability of the PCogAlignBench and the effectiveness of our proposed PCogAlign. We will open-source the constructed benchmark and code at this https URL. 

---
# Deep Temporal Reasoning in Video Language Models: A Cross-Linguistic Evaluation of Action Duration and Completion through Perfect Times 

**Authors**: Olga Loginova, Sofía Ortega Loguinova  

**Link**: [PDF](https://arxiv.org/pdf/2506.00928)  

**Abstract**: Human perception of events is intrinsically tied to distinguishing between completed (perfect and telic) and ongoing (durative) actions, a process mediated by both linguistic structure and visual cues. In this work, we introduce the \textbf{Perfect Times} dataset, a novel, quadrilingual (English, Italian, Russian, and Japanese) multiple-choice question-answering benchmark designed to assess video-language models (VLMs) on temporal reasoning. By pairing everyday activity videos with event completion labels and perfectivity-tailored distractors, our dataset probes whether models truly comprehend temporal dynamics or merely latch onto superficial markers. Experimental results indicate that state-of-the-art models, despite their success on text-based tasks, struggle to mirror human-like temporal and causal reasoning grounded in video. This study underscores the necessity of integrating deep multimodal cues to capture the nuances of action duration and completion within temporal and causal video dynamics, setting a new standard for evaluating and advancing temporal reasoning in VLMs. 

---
# Position as Probability: Self-Supervised Transformers that Think Past Their Training for Length Extrapolation 

**Authors**: Philip Heejun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.00920)  

**Abstract**: Deep sequence models typically degrade in accuracy when test sequences significantly exceed their training lengths, yet many critical tasks--such as algorithmic reasoning, multi-step arithmetic, and compositional generalization--require robust length extrapolation. We introduce PRISM, a Probabilistic Relative-position Implicit Superposition Model, a novel positional encoding mechanism that enables Transformers to extrapolate accurately up to 10x beyond their training length. PRISM learns continuous relative positions through a differentiable histogram-filter update, preserving position uncertainty via a probabilistic superposition rather than conventional deterministic embeddings. Empirically, PRISM achieves state-of-the-art length extrapolation, successfully generalizing to previously intractable sequence lengths across algorithmic benchmarks--including arithmetic (addition, multiplication), SCAN compositionality tasks, and complex copy variants derived from DeepMind's recent datasets. Our analysis demonstrates that PRISM's stochastic positional encoding maintains sharp and interpretable internal states, providing a theoretical basis for reliable length generalization. These results advance the goal of neural sequence models that remain algorithmically robust at lengths far exceeding their training horizon. 

---
# CODEMENV: Benchmarking Large Language Models on Code Migration 

**Authors**: Keyuan Cheng, Xudong Shen, Yihao Yang, Tengyue Wang, Yang Cao, Muhammad Asif Ali, Hanbin Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00894)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities across various software engineering tasks; however, their effectiveness in code migration, adapting code to run in different environments, remains insufficiently studied. In this work, we introduce CODEMENV: Code Migration Across Environment, a new benchmark specifically designed to assess LLMs' abilities in code migration scenarios. CODEMENV consists of 922 examples spanning 19 Python and Java packages, and covers three core tasks: (1) identifying functions incompatible with specific versions, (2) detecting changes in function definitions, and (3) adapting code to target environments. Experimental evaluation with seven LLMs on CODEMENV yields an average pass@1 rate of 26.50%, with GPT-4O achieving the highest score at 43.84%. Key findings include: (i) LLMs tend to be more proficient with newer function versions, which aids in migrating legacy code, and (ii) LLMs sometimes exhibit logical inconsistencies by identifying function changes irrelevant to the intended migration environment. The datasets are available at this https URL. 

---
# Towards Predicting Any Human Trajectory In Context 

**Authors**: Ryo Fujii, Hideo Saito, Ryo Hachiuma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00871)  

**Abstract**: Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, this process is often impractical on edge devices due to constrained computational resources. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables rapid adaptation without fine-tuning on the scenario-specific data. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. The code will be released at this https URL. 

---
# Generalizable LLM Learning of Graph Synthetic Data with Reinforcement Learning 

**Authors**: Yizhuo Zhang, Heng Wang, Shangbin Feng, Zhaoxuan Tan, Xinyun Liu, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.00845)  

**Abstract**: Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph synthetic data with reinforcement learning. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that RL would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting. We employ RL algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our RL recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9\% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards, mixing synthetic and real-world task data yields potential gains, while compositionality and explainable intermediate steps remains a critical challenge even after RL. 

---
# HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models 

**Authors**: Songtao Jiang, Yan Zhang, Yeying Jin, Zhihang Tang, Yangyang Wu, Yang Feng, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00805)  

**Abstract**: Medical Vision-Language Models (Med-VLMs) have achieved success across various tasks, yet most existing methods overlook the modality misalignment issue that can lead to untrustworthy responses in clinical settings. In this paper, we propose Hierarchical Self-Contrastive Rewarding (HSCR), a novel approach that addresses two critical challenges in Med-VLM alignment: 1) Cost-effective generation of high-quality preference data; 2) Capturing nuanced and context-aware preferences for improved alignment. HSCR first leverages the inherent capability of Med-VLMs to generate dispreferred responses with higher sampling probability. By analyzing output logit shifts after visual token dropout, we identify modality-coupled tokens that induce misalignment and derive an implicit alignment reward function. This function guides token replacement with hallucinated ones during decoding, producing high-quality dispreferred data. Furthermore, HSCR introduces a multi-level preference optimization strategy, which extends beyond traditional adjacent-level optimization by incorporating nuanced implicit preferences, leveraging relative quality in dispreferred data to capture subtle alignment cues for more precise and context-aware optimization. Extensive experiments across multiple medical tasks, including Med-VQA, medical image captioning and instruction following, demonstrate that HSCR not only enhances zero-shot performance but also significantly improves modality alignment and trustworthiness with just 2,000 training entries. 

---
# LIFT the Veil for the Truth: Principal Weights Emerge after Rank Reduction for Reasoning-Focused Supervised Fine-Tuning 

**Authors**: Zihang Liu, Tianyu Pang, Oleg Balabanov, Chaoqun Yang, Tianjin Huang, Lu Yin, Yaoqing Yang, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00772)  

**Abstract**: Recent studies have shown that supervised fine-tuning of LLMs on a small number of high-quality datasets can yield strong reasoning capabilities. However, full fine-tuning (Full FT), while powerful, is computationally expensive and susceptible to overfitting and catastrophic forgetting, particularly when data is limited. Sparse fine-tuning, which previously achieved notable success by updating only a small subset of model parameters, offers a promising trade-off between efficiency and effectiveness. Yet, it has lagged behind in the LLM era due to the difficulty of identifying parameters truly critical for reasoning. In this work, we state that weights with the largest magnitude after low-rank approximation are critical weights for fine-tuning, which we call Principal Weights. Surprisingly, while magnitude-based sparse fine-tuning performs poorly as a baseline on LLM fine-tuning, it becomes highly effective after rank reduction. These insights motivate our method: Low-rank Informed Sparse Fine-Tuning (LIFT). LIFT only updates the top 5% Principal Weights throughout training and consistently achieves better performance on reasoning tasks than Full FT, while maintaining memory efficiency on par with popular parameter-efficient fine-tuning methods. In addition to strong performance on target domains such as arithmetic reasoning, LIFT also retains up to 20% more source-domain knowledge, compared to Full FT and LoRA. Our code is available at: this https URL. 

---
# Bregman Conditional Random Fields: Sequence Labeling with Parallelizable Inference Algorithms 

**Authors**: Caio Corro, Mathieu Lacroix, Joseph Le Roux  

**Link**: [PDF](https://arxiv.org/pdf/2506.00732)  

**Abstract**: We propose a novel discriminative model for sequence labeling called Bregman conditional random fields (BCRF). Contrary to standard linear-chain conditional random fields, BCRF allows fast parallelizable inference algorithms based on iterative Bregman projections. We show how such models can be learned using Fenchel-Young losses, including extension for learning from partial labels. Experimentally, our approach delivers comparable results to CRF while being faster, and achieves better results in highly constrained settings compared to mean field, another parallelizable alternative. 

---
# DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains 

**Authors**: Yongkang Xiao, Sinian Zhang, Yi Dai, Huixue Zhou, Jue Hou, Jie Ding, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00708)  

**Abstract**: Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility. 

---
# Existing Large Language Model Unlearning Evaluations Are Inconclusive 

**Authors**: Zhili Feng, Yixuan Even Xu, Alexander Robey, Robert Kirk, Xander Davies, Yarin Gal, Avi Schwarzschild, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2506.00688)  

**Abstract**: Machine unlearning aims to remove sensitive or undesired data from large language models. However, recent studies suggest that unlearning is often shallow, claiming that removed knowledge can easily be recovered. In this work, we critically examine standard unlearning evaluation practices and uncover key limitations that shake our trust in those findings. First, we show that some evaluations introduce substantial new information into the model, potentially masking true unlearning performance by re-teaching the model during testing. Second, we demonstrate that evaluation outcomes vary significantly across tasks, undermining the generalizability of current evaluation routines. Finally, we find that many evaluations rely on spurious correlations, making their results difficult to trust and interpret. Taken together, these issues suggest that current evaluation protocols may both overstate and understate unlearning success. To address this, we propose two principles for future unlearning evaluations: minimal information injection and downstream task awareness. We validate these principles through a series of targeted experiments, showing how violations of each can lead to misleading conclusions. 

---
# Linear Representation Transferability Hypothesis: Leveraging Small Models to Steer Large Models 

**Authors**: Femi Bello, Anubrata Das, Fanzhi Zeng, Fangcong Yin, Leqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00653)  

**Abstract**: It has been hypothesized that neural networks with similar architectures trained on similar data learn shared representations relevant to the learning task. We build on this idea by extending the conceptual framework where representations learned across models trained on the same data can be expressed as linear combinations of a \emph{universal} set of basis features. These basis features underlie the learning task itself and remain consistent across models, regardless of scale. From this framework, we propose the \textbf{Linear Representation Transferability (LRT)} Hypothesis -- that there exists an affine transformation between the representation spaces of different models. To test this hypothesis, we learn affine mappings between the hidden states of models of different sizes and evaluate whether steering vectors -- directions in hidden state space associated with specific model behaviors -- retain their semantic effect when transferred from small to large language models using the learned mappings. We find strong empirical evidence that such affine mappings can preserve steering behaviors. These findings suggest that representations learned by small models can be used to guide the behavior of large models, and that the LRT hypothesis may be a promising direction on understanding representation alignment across model scales. 

---
# Reasoning Like an Economist: Post-Training on Economic Problems Induces Strategic Generalization in LLMs 

**Authors**: Yufa Zhou, Shaobo Wang, Xingyu Dong, Xiangqi Jin, Yifang Chen, Yue Min, Kexin Yang, Xingzhang Ren, Dayiheng Liu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00577)  

**Abstract**: Directly training Large Language Models (LLMs) for Multi-Agent Systems (MAS) remains challenging due to intricate reward modeling, dynamic agent interactions, and demanding generalization requirements. This paper explores whether post-training techniques, specifically Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR), can effectively $\textit{generalize}$ to multi-agent scenarios. We use economic reasoning as a testbed, leveraging its strong foundations in mathematics and game theory, its demand for structured analytical reasoning, and its relevance to real-world applications such as market design, resource allocation, and policy analysis. We introduce $\textbf{Recon}$ ($\textbf{R}$easoning like an $\textbf{ECON}$omist), a 7B-parameter open-source LLM post-trained on a hand-curated dataset of 2,100 high-quality economic reasoning problems. Comprehensive evaluation on economic reasoning benchmarks and multi-agent games reveals clear improvements in structured reasoning and economic rationality. These results underscore the promise of domain-aligned post-training for enhancing reasoning and agent alignment, shedding light on the roles of SFT and RL in shaping model behavior. Code is available at this https URL . 

---
# MMedAgent-RL: Optimizing Multi-Agent Collaboration for Multimodal Medical Reasoning 

**Authors**: Peng Xia, Jinglu Wang, Yibo Peng, Kaide Zeng, Xian Wu, Xiangru Tang, Hongtu Zhu, Yun Li, Shujie Liu, Yan Lu, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00555)  

**Abstract**: Medical Large Vision-Language Models (Med-LVLMs) have shown strong potential in multimodal diagnostic tasks. However, existing single-agent models struggle to generalize across diverse medical specialties, limiting their performance. Recent efforts introduce multi-agent collaboration frameworks inspired by clinical workflows, where general practitioners (GPs) and specialists interact in a fixed sequence. Despite improvements, these static pipelines lack flexibility and adaptability in reasoning. To address this, we propose MMedAgent-RL, a reinforcement learning (RL)-based multi-agent framework that enables dynamic, optimized collaboration among medical agents. Specifically, we train two GP agents based on Qwen2.5-VL via RL: the triage doctor learns to assign patients to appropriate specialties, while the attending physician integrates the judgments from multi-specialists and its own knowledge to make final decisions. To address the inconsistency in specialist outputs, we introduce a curriculum learning (CL)-guided RL strategy that progressively teaches the attending physician to balance between imitating specialists and correcting their mistakes. Experiments on five medical VQA benchmarks demonstrate that MMedAgent-RL not only outperforms both open-source and proprietary Med-LVLMs, but also exhibits human-like reasoning patterns. Notably, it achieves an average performance gain of 18.4% over supervised fine-tuning baselines. 

---
# Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities 

**Authors**: Jiahui Geng, Thy Thy Tran, Preslav Nakov, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2506.00548)  

**Abstract**: Existing attacks against multimodal language models (MLLMs) primarily communicate instructions through text accompanied by adversarial images. In contrast, we exploit the capabilities of MLLMs to interpret non-textual instructions, specifically, adversarial images or audio generated by our novel method, Con Instruction. We optimize these adversarial examples to align closely with target instructions in the embedding space, revealing the detrimental implications of MLLMs' sophisticated understanding. Unlike prior work, our method does not require training data or preprocessing of textual instructions. While these non-textual adversarial examples can effectively bypass MLLM safety mechanisms, their combination with various text inputs substantially amplifies attack success. We further introduce a new Attack Response Categorization (ARC) framework, which evaluates both the quality of the model's response and its relevance to the malicious instructions. Experimental results demonstrate that Con Instruction effectively bypasses safety mechanisms in multiple vision- and audio-language models, including LLaVA-v1.5, InternVL, Qwen-VL, and Qwen-Audio, evaluated on two standard benchmarks: AdvBench and SafeBench. Specifically, our method achieves the highest attack success rates, reaching 81.3% and 86.6% on LLaVA-v1.5 (13B). On the defense side, we explore various countermeasures against our attacks and uncover a substantial performance gap among existing techniques. Our implementation is made publicly available. 

---
# CityLens: Benchmarking Large Language-Vision Models for Urban Socioeconomic Sensing 

**Authors**: Tianhui Liu, Jie Feng, Hetian Pang, Xin Zhang, Tianjian Ouyang, Zhiyuan Zhang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.00530)  

**Abstract**: Understanding urban socioeconomic conditions through visual data is a challenging yet essential task for sustainable urban development and policy planning. In this work, we introduce $\textbf{CityLens}$, a comprehensive benchmark designed to evaluate the capabilities of large language-vision models (LLVMs) in predicting socioeconomic indicators from satellite and street view imagery. We construct a multi-modal dataset covering a total of 17 globally distributed cities, spanning 6 key domains: economy, education, crime, transport, health, and environment, reflecting the multifaceted nature of urban life. Based on this dataset, we define 11 prediction tasks and utilize three evaluation paradigms: Direct Metric Prediction, Normalized Metric Estimation, and Feature-Based Regression. We benchmark 17 state-of-the-art LLVMs across these tasks. Our results reveal that while LLVMs demonstrate promising perceptual and reasoning capabilities, they still exhibit limitations in predicting urban socioeconomic indicators. CityLens provides a unified framework for diagnosing these limitations and guiding future efforts in using LLVMs to understand and predict urban socioeconomic patterns. Our codes and datasets are open-sourced via this https URL. 

---
# FLoE: Fisher-Based Layer Selection for Efficient Sparse Adaptation of Low-Rank Experts 

**Authors**: Xinyi Wang, Lirong Gao, Haobo Wang, Yiming Zhang, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00495)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods have emerged as a widely adopted strategy for adapting pre-trained Large Language Models (LLMs) to downstream tasks, significantly reducing memory and computational costs. However, most existing PEFT techniques uniformly deploy LoRA adapters across all layers, disregarding the intrinsic heterogeneity of layer contributions and task-specific rank requirements. This uniform paradigm leads to redundant parameter allocation and suboptimal adaptation efficiency. To address these limitations, we propose FLoE, a novel PEFT framework that introduces two key innovations: (i) a Fisher information-guided importance scoring mechanism to dynamically identify task-critical transformer layers for MoE-based low-rank adaptation, enabling sparse adapter deployment; and (ii) a Bayesian optimization-driven rank allocator that automatically determines optimal LoRA ranks on specific datasets without exhaustive grid search. Extensive experiments across diverse LLMs and benchmarks reveal that FLoE achieves impressive efficiency-accuracy trade-offs, making FLoE particularly advantageous in resource-constrained environments that necessitate rapid adaptation. 

---
# BenchHub: A Unified Benchmark Suite for Holistic and Customizable LLM Evaluation 

**Authors**: Eunsu Kim, Haneul Yoo, Guijin Son, Hitesh Patel, Amit Agarwal, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00482)  

**Abstract**: As large language models (LLMs) continue to advance, the need for up-to-date and well-organized benchmarks becomes increasingly critical. However, many existing datasets are scattered, difficult to manage, and make it challenging to perform evaluations tailored to specific needs or domains, despite the growing importance of domain-specific models in areas such as math or code. In this paper, we introduce BenchHub, a dynamic benchmark repository that empowers researchers and developers to evaluate LLMs more effectively. BenchHub aggregates and automatically classifies benchmark datasets from diverse domains, integrating 303K questions across 38 benchmarks. It is designed to support continuous updates and scalable data management, enabling flexible and customizable evaluation tailored to various domains or use cases. Through extensive experiments with various LLM families, we demonstrate that model performance varies significantly across domain-specific subsets, emphasizing the importance of domain-aware benchmarking. We believe BenchHub can encourage better dataset reuse, more transparent model comparisons, and easier identification of underrepresented areas in existing benchmarks, offering a critical infrastructure for advancing LLM evaluation research. 

---
# XMAD-Bench: Cross-Domain Multilingual Audio Deepfake Benchmark 

**Authors**: Ioan-Paul Ciobanu, Andrei-Iulian Hiji, Nicolae-Catalin Ristea, Paul Irofti, Cristian Rusu, Radu Tudor Ionescu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00462)  

**Abstract**: Recent advances in audio generation led to an increasing number of deepfakes, making the general public more vulnerable to financial scams, identity theft, and misinformation. Audio deepfake detectors promise to alleviate this issue, with many recent studies reporting accuracy rates close to 99%. However, these methods are typically tested in an in-domain setup, where the deepfake samples from the training and test sets are produced by the same generative models. To this end, we introduce XMAD-Bench, a large-scale cross-domain multilingual audio deepfake benchmark comprising 668.8 hours of real and deepfake speech. In our novel dataset, the speakers, the generative methods, and the real audio sources are distinct across training and test splits. This leads to a challenging cross-domain evaluation setup, where audio deepfake detectors can be tested ``in the wild''. Our in-domain and cross-domain experiments indicate a clear disparity between the in-domain performance of deepfake detectors, which is usually as high as 100%, and the cross-domain performance of the same models, which is sometimes similar to random chance. Our benchmark highlights the need for the development of robust audio deepfake detectors, which maintain their generalization capacity across different languages, speakers, generative methods, and data sources. Our benchmark is publicly released at this https URL. 

---
# Spectral Insights into Data-Oblivious Critical Layers in Large Language Models 

**Authors**: Xuyuan Liu, Lei Hsiung, Yaoqing Yang, Yujun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00382)  

**Abstract**: Understanding how feature representations evolve across layers in large language models (LLMs) is key to improving their interpretability and robustness. While recent studies have identified critical layers linked to specific functions or behaviors, these efforts typically rely on data-dependent analyses of fine-tuned models, limiting their use to post-hoc settings. In contrast, we introduce a data-oblivious approach to identify intrinsic critical layers in pre-fine-tuned LLMs by analyzing representation dynamics via Centered Kernel Alignment(CKA). We show that layers with significant shifts in representation space are also those most affected during fine-tuning--a pattern that holds consistently across tasks for a given model. Our spectral analysis further reveals that these shifts are driven by changes in the top principal components, which encode semantic transitions from rationales to conclusions. We further apply these findings to two practical scenarios: efficient domain adaptation, where fine-tuning critical layers leads to greater loss reduction compared to non-critical layers; and backdoor defense, where freezing them reduces attack success rates by up to 40%. 

---
# Adapting General-Purpose Embedding Models to Private Datasets Using Keyword-based Retrieval 

**Authors**: Yubai Wei, Jiale Han, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00363)  

**Abstract**: Text embedding models play a cornerstone role in AI applications, such as retrieval-augmented generation (RAG). While general-purpose text embedding models demonstrate strong performance on generic retrieval benchmarks, their effectiveness diminishes when applied to private datasets (e.g., company-specific proprietary data), which often contain specialized terminology and lingo. In this work, we introduce BMEmbed, a novel method for adapting general-purpose text embedding models to private datasets. By leveraging the well-established keyword-based retrieval technique (BM25), we construct supervisory signals from the ranking of keyword-based retrieval results to facilitate model adaptation. We evaluate BMEmbed across a range of domains, datasets, and models, showing consistent improvements in retrieval performance. Moreover, we provide empirical insights into how BM25-based signals contribute to improving embeddings by fostering alignment and uniformity, highlighting the value of this approach in adapting models to domain-specific data. We release the source code available at this https URL for the research community. 

---
# Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents 

**Authors**: Xiao Yu, Baolin Peng, Ruize Xu, Michel Galley, Hao Cheng, Suman Nath, Jianfeng Gao, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.00320)  

**Abstract**: Recent progress in reasoning with large language models (LLMs), such as DeepSeek-R1, demonstrates impressive capabilities in domains like mathematics and coding, by exhibiting complex cognitive behaviors such as verification, goal decomposition, and self-reflection. However, it is unclear what behavior is effective and what behavior is missing for long-horizon AI agents tasks. In this work, we propose Dyna-Think, a thinking framework that integrates planning with an internal world model with reasoning and acting to enhance AI agent performance. To enable Dyna-Think, we propose Dyna-Think Imitation Learning (DIT) and Dyna-Think Dyna Training (DDT). To initialize a policy with Dyna-Think, DIT reconstructs the thinking process of R1 to focus on performing world model simulation relevant to the proposed (and planned) action, and trains the policy using this reconstructed data. To enhance Dyna-Think, DDT uses a two-stage training process to first improve the agent's world modeling ability via objectives such as state prediction or critique generation, and then improve the agent's action via policy training. We evaluate our methods on OSWorld, and demonstrate that Dyna-Think improves the agent's in-domain and out-of-domain performance, achieving similar best-of-n performance compared to R1 while generating 2x less tokens on average. Our extensive empirical studies reveal that 1) using critique generation for world model training is effective to improve policy performance; and 2) AI agents with better performance correlate with better world modeling abilities. We believe our results suggest a promising research direction to integrate world model simulation into AI agents to enhance their reasoning, planning, and acting capabilities. 

---
# MythTriage: Scalable Detection of Opioid Use Disorder Myths on a Video-Sharing Platform 

**Authors**: Hayoung Jung, Shravika Mittal, Ananya Aatreya, Navreet Kaur, Munmun De Choudhury, Tanushree Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2506.00308)  

**Abstract**: Understanding the prevalence of misinformation in health topics online can inform public health policies and interventions. However, measuring such misinformation at scale remains a challenge, particularly for high-stakes but understudied topics like opioid-use disorder (OUD)--a leading cause of death in the U.S. We present the first large-scale study of OUD-related myths on YouTube, a widely-used platform for health information. With clinical experts, we validate 8 pervasive myths and release an expert-labeled video dataset. To scale labeling, we introduce MythTriage, an efficient triage pipeline that uses a lightweight model for routine cases and defers harder ones to a high-performing, but costlier, large language model (LLM). MythTriage achieves up to 0.86 macro F1-score while estimated to reduce annotation time and financial cost by over 76% compared to experts and full LLM labeling. We analyze 2.9K search results and 343K recommendations, uncovering how myths persist on YouTube and offering actionable insights for public health and platform moderation. 

---
# RoboMoRe: LLM-based Robot Co-design via Joint Optimization of Morphology and Reward 

**Authors**: Jiawei Fang, Yuxuan Sun, Chengtian Ma, Qiuyu Lu, Lining Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.00276)  

**Abstract**: Robot co-design, jointly optimizing morphology and control policy, remains a longstanding challenge in the robotics community, where many promising robots have been developed. However, a key limitation lies in its tendency to converge to sub-optimal designs due to the use of fixed reward functions, which fail to explore the diverse motion modes suitable for different morphologies. Here we propose RoboMoRe, a large language model (LLM)-driven framework that integrates morphology and reward shaping for co-optimization within the robot co-design loop. RoboMoRe performs a dual-stage optimization: in the coarse optimization stage, an LLM-based diversity reflection mechanism generates both diverse and high-quality morphology-reward pairs and efficiently explores their distribution. In the fine optimization stage, top candidates are iteratively refined through alternating LLM-guided reward and morphology gradient updates. RoboMoRe can optimize both efficient robot morphologies and their suited motion behaviors through reward shaping. Results demonstrate that without any task-specific prompting or predefined reward/morphology templates, RoboMoRe significantly outperforms human-engineered designs and competing methods across eight different tasks. 

---
# GPR: Empowering Generation with Graph-Pretrained Retriever 

**Authors**: Xiaochen Wang, Zongyu Wu, Yuan Zhong, Xiang Zhang, Suhang Wang, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00261)  

**Abstract**: Graph retrieval-augmented generation (GRAG) places high demands on graph-specific retrievers. However, existing retrievers often rely on language models pretrained on plain text, limiting their effectiveness due to domain misalignment and structure ignorance. To address these challenges, we propose GPR, a graph-based retriever pretrained directly on knowledge graphs. GPR aligns natural language questions with relevant subgraphs through LLM-guided graph augmentation and employs a structure-aware objective to learn fine-grained retrieval strategies. Experiments on two datasets, three LLM backbones, and five baselines show that GPR consistently improves both retrieval quality and downstream generation, demonstrating its effectiveness as a robust retrieval solution for GRAG. 

---
# MIR: Methodology Inspiration Retrieval for Scientific Research Problems 

**Authors**: Aniketh Garikaparthi, Manasi Patwardhan, Aditya Sanjiv Kanade, Aman Hassan, Lovekesh Vig, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2506.00249)  

**Abstract**: There has been a surge of interest in harnessing the reasoning capabilities of Large Language Models (LLMs) to accelerate scientific discovery. While existing approaches rely on grounding the discovery process within the relevant literature, effectiveness varies significantly with the quality and nature of the retrieved literature. We address the challenge of retrieving prior work whose concepts can inspire solutions for a given research problem, a task we define as Methodology Inspiration Retrieval (MIR). We construct a novel dataset tailored for training and evaluating retrievers on MIR, and establish baselines. To address MIR, we build the Methodology Adjacency Graph (MAG); capturing methodological lineage through citation relationships. We leverage MAG to embed an "intuitive prior" into dense retrievers for identifying patterns of methodological inspiration beyond superficial semantic similarity. This achieves significant gains of +5.4 in Recall@3 and +7.8 in Mean Average Precision (mAP) over strong baselines. Further, we adapt LLM-based re-ranking strategies to MIR, yielding additional improvements of +4.5 in Recall@3 and +4.8 in mAP. Through extensive ablation studies and qualitative analyses, we exhibit the promise of MIR in enhancing automated scientific discovery and outline avenues for advancing inspiration-driven retrieval. 

---
# Beyond Semantic Entropy: Boosting LLM Uncertainty Quantification with Pairwise Semantic Similarity 

**Authors**: Dang Nguyen, Ali Payani, Baharan Mirzasoleiman  

**Link**: [PDF](https://arxiv.org/pdf/2506.00245)  

**Abstract**: Hallucination in large language models (LLMs) can be detected by assessing the uncertainty of model outputs, typically measured using entropy. Semantic entropy (SE) enhances traditional entropy estimation by quantifying uncertainty at the semantic cluster level. However, as modern LLMs generate longer one-sentence responses, SE becomes less effective because it overlooks two crucial factors: intra-cluster similarity (the spread within a cluster) and inter-cluster similarity (the distance between clusters). To address these limitations, we propose a simple black-box uncertainty quantification method inspired by nearest neighbor estimates of entropy. Our approach can also be easily extended to white-box settings by incorporating token probabilities. Additionally, we provide theoretical results showing that our method generalizes semantic entropy. Extensive empirical results demonstrate its effectiveness compared to semantic entropy across two recent LLMs (Phi3 and Llama3) and three common text generation tasks: question answering, text summarization, and machine translation. Our code is available at this https URL. 

---
# Whispers of Many Shores: Cultural Alignment through Collaborative Cultural Expertise 

**Authors**: Shuai Feng, Wei-Chuang Chan, Srishti Chouhan, Junior Francisco Garcia Ayala, Srujananjali Medicherla, Kyle Clark, Mingwei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.00242)  

**Abstract**: The integration of large language models (LLMs) into global applications necessitates effective cultural alignment for meaningful and culturally-sensitive interactions. Current LLMs often lack the nuanced understanding required for diverse cultural contexts, and adapting them typically involves costly full fine-tuning. To address this, we introduce a novel soft prompt fine-tuning framework that enables efficient and modular cultural alignment. Our method utilizes vectorized prompt tuning to dynamically route queries to a committee of culturally specialized 'expert' LLM configurations, created by optimizing soft prompt embeddings without altering the base model's parameters. Extensive experiments demonstrate that our framework significantly enhances cultural sensitivity and adaptability, improving alignment scores from 0.208 to 0.820, offering a robust solution for culturally-aware LLM deployment. This research paves the way for subsequent investigations into enhanced cultural coverage and dynamic expert adaptation, crucial for realizing autonomous AI with deeply nuanced understanding in a globally interconnected world. 

---
# ZeShot-VQA: Zero-Shot Visual Question Answering Framework with Answer Mapping for Natural Disaster Damage Assessment 

**Authors**: Ehsan Karimi, Maryam Rahnemoonfar  

**Link**: [PDF](https://arxiv.org/pdf/2506.00238)  

**Abstract**: Natural disasters usually affect vast areas and devastate infrastructures. Performing a timely and efficient response is crucial to minimize the impact on affected communities, and data-driven approaches are the best choice. Visual question answering (VQA) models help management teams to achieve in-depth understanding of damages. However, recently published models do not possess the ability to answer open-ended questions and only select the best answer among a predefined list of answers. If we want to ask questions with new additional possible answers that do not exist in the predefined list, the model needs to be fin-tuned/retrained on a new collected and annotated dataset, which is a time-consuming procedure. In recent years, large-scale Vision-Language Models (VLMs) have earned significant attention. These models are trained on extensive datasets and demonstrate strong performance on both unimodal and multimodal vision/language downstream tasks, often without the need for fine-tuning. In this paper, we propose a VLM-based zero-shot VQA (ZeShot-VQA) method, and investigate the performance of on post-disaster FloodNet dataset. Since the proposed method takes advantage of zero-shot learning, it can be applied on new datasets without fine-tuning. In addition, ZeShot-VQA is able to process and generate answers that has been not seen during the training procedure, which demonstrates its flexibility. 

---
# Localized LoRA: A Structured Low-Rank Approximation for Efficient Fine-Tuning 

**Authors**: Babak Barazandeh  

**Link**: [PDF](https://arxiv.org/pdf/2506.00236)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, offer compact and effective alternatives to full model fine-tuning by introducing low-rank updates to pretrained weights. However, most existing approaches rely on global low-rank structures, which can overlook spatial patterns spread across the parameter space. In this work, we propose Localized LoRA, a generalized framework that models weight updates as a composition of low-rank matrices applied to structured blocks of the weight matrix. This formulation enables dense, localized updates throughout the parameter space-without increasing the total number of trainable parameters. We provide a formal comparison between global, diagonal-local, and fully localized low-rank approximations, and show that our method consistently achieves lower approximation error under matched parameter budgets. Experiments on both synthetic and practical settings demonstrate that Localized LoRA offers a more expressive and adaptable alternative to existing methods, enabling efficient fine-tuning with improved performance. 

---
# Control-R: Towards controllable test-time scaling 

**Authors**: Di Zhang, Weida Wang, Junxian Li, Xunzhi Wang, Jiatong Li, Jianbo Wu, Jingdi Lei, Haonan He, Peng Ye, Shufei Zhang, Wanli Ouyang, Yuqiang Li, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.00189)  

**Abstract**: This paper target in addressing the challenges of underthinking and overthinking in long chain-of-thought (CoT) reasoning for Large Reasoning Models (LRMs) by introducing Reasoning Control Fields (RCF)--a novel test-time approach that injects structured control signals to guide reasoning from a tree search perspective. RCF enables models to adjust reasoning effort according to given control conditions when solving complex tasks. Additionally, we present the Control-R-4K dataset, which consists of challenging problems annotated with detailed reasoning processes and corresponding control fields. To further enhance reasoning control, we propose a Conditional Distillation Finetuning (CDF) method, which trains model--particularly Control-R-32B--to effectively adjust reasoning effort during test time. Experimental results on benchmarks such as AIME2024 and MATH500 demonstrate that our approach achieves state-of-the-art performance at the 32B scale while enabling a controllable Long CoT reasoning process (L-CoT). Overall, this work introduces an effective paradigm for controllable test-time scaling reasoning. 

---
# Pushing the Limits of Beam Search Decoding for Transducer-based ASR models 

**Authors**: Lilit Grigoryan, Vladimir Bataev, Andrei Andrusenko, Hainan Xu, Vitaly Lavrukhin, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2506.00185)  

**Abstract**: Transducer models have emerged as a promising choice for end-to-end ASR systems, offering a balanced trade-off between recognition accuracy, streaming capabilities, and inference speed in greedy decoding. However, beam search significantly slows down Transducers due to repeated evaluations of key network components, limiting practical applications. This paper introduces a universal method to accelerate beam search for Transducers, enabling the implementation of two optimized algorithms: ALSD++ and AES++. The proposed method utilizes batch operations, a tree-based hypothesis structure, novel blank scoring for enhanced shallow fusion, and CUDA graph execution for efficient GPU inference. This narrows the speed gap between beam and greedy modes to only 10-20% for the whole system, achieves 14-30% relative improvement in WER compared to greedy decoding, and improves shallow fusion for low-resource up to 11% compared to existing implementations. All the algorithms are open sourced. 

---
# Disentangled Safety Adapters Enable Efficient Guardrails and Flexible Inference-Time Alignment 

**Authors**: Kundan Krishna, Joseph Y Cheng, Charles Maalouf, Leon A Gatys  

**Link**: [PDF](https://arxiv.org/pdf/2506.00166)  

**Abstract**: Existing paradigms for ensuring AI safety, such as guardrail models and alignment training, often compromise either inference efficiency or development flexibility. We introduce Disentangled Safety Adapters (DSA), a novel framework addressing these challenges by decoupling safety-specific computations from a task-optimized base model. DSA utilizes lightweight adapters that leverage the base model's internal representations, enabling diverse and flexible safety functionalities with minimal impact on inference cost. Empirically, DSA-based safety guardrails substantially outperform comparably sized standalone models, notably improving hallucination detection (0.88 vs. 0.61 AUC on Summedits) and also excelling at classifying hate speech (0.98 vs. 0.92 on ToxiGen) and unsafe model inputs and responses (0.93 vs. 0.90 on AEGIS2.0 & BeaverTails). Furthermore, DSA-based safety alignment allows dynamic, inference-time adjustment of alignment strength and a fine-grained trade-off between instruction following performance and model safety. Importantly, combining the DSA safety guardrail with DSA safety alignment facilitates context-dependent alignment strength, boosting safety on StrongReject by 93% while maintaining 98% performance on MTBench -- a total reduction in alignment tax of 8 percentage points compared to standard safety alignment fine-tuning. Overall, DSA presents a promising path towards more modular, efficient, and adaptable AI safety and alignment. 

---
# Children's Voice Privacy: First Steps And Emerging Challenges 

**Authors**: Ajinkya Kulkarni, Francisco Teixeira, Enno Hermann, Thomas Rolland, Isabel Trancoso, Mathew Magimai Doss  

**Link**: [PDF](https://arxiv.org/pdf/2506.00100)  

**Abstract**: Children are one of the most under-represented groups in speech technologies, as well as one of the most vulnerable in terms of privacy. Despite this, anonymization techniques targeting this population have received little attention. In this study, we seek to bridge this gap, and establish a baseline for the use of voice anonymization techniques designed for adult speech when applied to children's voices. Such an evaluation is essential, as children's speech presents a distinct set of challenges when compared to that of adults. This study comprises three children's datasets, six anonymization methods, and objective and subjective utility metrics for evaluation. Our results show that existing systems for adults are still able to protect children's voice privacy, but suffer from much higher utility degradation. In addition, our subjective study displays the challenges of automatic evaluation methods for speech quality in children's speech, highlighting the need for further research. 

---
# ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases 

**Authors**: Yuchong Li, Xiaojun Zeng, Chihua Fang, Jian Yang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00095)  

**Abstract**: Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at the homepage. 

---
# Bottom-Up Perspectives on AI Governance: Insights from User Reviews of AI Products 

**Authors**: Stefan Pasch  

**Link**: [PDF](https://arxiv.org/pdf/2506.00080)  

**Abstract**: With the growing importance of AI governance, numerous high-level frameworks and principles have been articulated by policymakers, institutions, and expert communities to guide the development and application of AI. While such frameworks offer valuable normative orientation, they may not fully capture the practical concerns of those who interact with AI systems in organizational and operational contexts. To address this gap, this study adopts a bottom-up approach to explore how governance-relevant themes are expressed in user discourse. Drawing on over 100,000 user reviews of AI products from this http URL, we apply BERTopic to extract latent themes and identify those most semantically related to AI governance. The analysis reveals a diverse set of governance-relevant topics spanning both technical and non-technical domains. These include concerns across organizational processes-such as planning, coordination, and communication-as well as stages of the AI value chain, including deployment infrastructure, data handling, and analytics. The findings show considerable overlap with institutional AI governance and ethics frameworks on issues like privacy and transparency, but also surface overlooked areas such as project management, strategy development, and customer interaction. This highlights the need for more empirically grounded, user-centered approaches to AI governance-approaches that complement normative models by capturing how governance unfolds in applied settings. By foregrounding how governance is enacted in practice, this study contributes to more inclusive and operationally grounded approaches to AI governance and digital policy. 

---
# Optimizing Storytelling, Improving Audience Retention, and Reducing Waste in the Entertainment Industry 

**Authors**: Andrew Cornfeld, Ashley Miller, Mercedes Mora-Figueroa, Kurt Samuels, Anthony Palomba  

**Link**: [PDF](https://arxiv.org/pdf/2506.00076)  

**Abstract**: Television networks face high financial risk when making programming decisions, often relying on limited historical data to forecast episodic viewership. This study introduces a machine learning framework that integrates natural language processing (NLP) features from over 25000 television episodes with traditional viewership data to enhance predictive accuracy. By extracting emotional tone, cognitive complexity, and narrative structure from episode dialogue, we evaluate forecasting performance using SARIMAX, rolling XGBoost, and feature selection models. While prior viewership remains a strong baseline predictor, NLP features contribute meaningful improvements for some series. We also introduce a similarity scoring method based on Euclidean distance between aggregate dialogue vectors to compare shows by content. Tested across diverse genres, including Better Call Saul and Abbott Elementary, our framework reveals genre-specific performance and offers interpretable metrics for writers, executives, and marketers seeking data-driven insight into audience behavior. 

---
# The Automated but Risky Game: Modeling Agent-to-Agent Negotiations and Transactions in Consumer Markets 

**Authors**: Shenzhe Zhu, Jiao Sun, Yi Nian, Tobin South, Alex Pentland, Jiaxin Pei  

**Link**: [PDF](https://arxiv.org/pdf/2506.00073)  

**Abstract**: AI agents are increasingly used in consumer-facing applications to assist with tasks such as product search, negotiation, and transaction execution. In this paper, we explore a future scenario where both consumers and merchants authorize AI agents to fully automate negotiations and transactions. We aim to answer two key questions: (1) Do different LLM agents vary in their ability to secure favorable deals for users? (2) What risks arise from fully automating deal-making with AI agents in consumer markets? To address these questions, we develop an experimental framework that evaluates the performance of various LLM agents in real-world negotiation and transaction settings. Our findings reveal that AI-mediated deal-making is an inherently imbalanced game -- different agents achieve significantly different outcomes for their users. Moreover, behavioral anomalies in LLMs can result in financial losses for both consumers and merchants, such as overspending or accepting unreasonable deals. These results underscore that while automation can improve efficiency, it also introduces substantial risks. Users should exercise caution when delegating business decisions to AI agents. 

---
# Evaluating Prompt Engineering Techniques for Accuracy and Confidence Elicitation in Medical LLMs 

**Authors**: Nariman Naderi, Zahra Atf, Peter R Lewis, Aref Mahjoub far, Seyed Amir Ahmad Safavi-Naini, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2506.00072)  

**Abstract**: This paper investigates how prompt engineering techniques impact both accuracy and confidence elicitation in Large Language Models (LLMs) applied to medical contexts. Using a stratified dataset of Persian board exam questions across multiple specialties, we evaluated five LLMs - GPT-4o, o3-mini, Llama-3.3-70b, Llama-3.1-8b, and DeepSeek-v3 - across 156 configurations. These configurations varied in temperature settings (0.3, 0.7, 1.0), prompt styles (Chain-of-Thought, Few-Shot, Emotional, Expert Mimicry), and confidence scales (1-10, 1-100). We used AUC-ROC, Brier Score, and Expected Calibration Error (ECE) to evaluate alignment between confidence and actual performance. Chain-of-Thought prompts improved accuracy but also led to overconfidence, highlighting the need for calibration. Emotional prompting further inflated confidence, risking poor decisions. Smaller models like Llama-3.1-8b underperformed across all metrics, while proprietary models showed higher accuracy but still lacked calibrated confidence. These results suggest prompt engineering must address both accuracy and uncertainty to be effective in high-stakes medical tasks. 

---
# SafeCOMM: What about Safety Alignment in Fine-Tuned Telecom Large Language Models? 

**Authors**: Aladin Djuhera, Swanand Ravindra Kadhe, Farhan Ahmed, Syed Zawad, Holger Boche, Walid Saad  

**Link**: [PDF](https://arxiv.org/pdf/2506.00062)  

**Abstract**: Fine-tuning large language models (LLMs) for telecom tasks and datasets is a common practice to adapt general-purpose models to the telecom domain. However, little attention has been paid to how this process may compromise model safety. Recent research has shown that even benign fine-tuning can degrade the safety alignment of LLMs, causing them to respond to harmful or unethical user queries. In this paper, we investigate this issue for telecom-tuned LLMs using three representative datasets featured by the GenAINet initiative. We show that safety degradation persists even for structured and seemingly harmless datasets such as 3GPP standards and tabular records, indicating that telecom-specific data is not immune to safety erosion during fine-tuning. We further extend our analysis to publicly available Telecom LLMs trained via continual pre-training, revealing that safety alignment is often severely lacking, primarily due to the omission of safety-focused instruction tuning. To address these issues in both fine-tuned and pre-trained models, we conduct extensive experiments and evaluate three safety realignment defenses (SafeInstruct, SafeLoRA, and SafeMERGE) using established red-teaming benchmarks. The results show that, across all settings, the proposed defenses can effectively restore safety after harmful degradation without compromising downstream task performance, leading to Safe teleCOMMunication (SafeCOMM) models. In a nutshell, our work serves as a diagnostic study and practical guide for safety realignment in telecom-tuned LLMs, and emphasizes the importance of safety-aware instruction and fine-tuning for real-world deployments of Telecom LLMs. 

---
# Comparative analysis of privacy-preserving open-source LLMs regarding extraction of diagnostic information from clinical CMR imaging reports 

**Authors**: Sina Amirrajab, Volker Vehof, Michael Bietenbeck, Ali Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.00060)  

**Abstract**: Purpose: We investigated the utilization of privacy-preserving, locally-deployed, open-source Large Language Models (LLMs) to extract diagnostic information from free-text cardiovascular magnetic resonance (CMR) reports. Materials and Methods: We evaluated nine open-source LLMs on their ability to identify diagnoses and classify patients into various cardiac diagnostic categories based on descriptive findings in 109 clinical CMR reports. Performance was quantified using standard classification metrics including accuracy, precision, recall, and F1 score. We also employed confusion matrices to examine patterns of misclassification across models. Results: Most open-source LLMs demonstrated exceptional performance in classifying reports into different diagnostic categories. Google's Gemma2 model achieved the highest average F1 score of 0.98, followed by Qwen2.5:32B and DeepseekR1-32B with F1 scores of 0.96 and 0.95, respectively. All other evaluated models attained average scores above 0.93, with Mistral and DeepseekR1-7B being the only exceptions. The top four LLMs outperformed our board-certified cardiologist (F1 score of 0.94) across all evaluation metrics in analyzing CMR reports. Conclusion: Our findings demonstrate the feasibility of implementing open-source, privacy-preserving LLMs in clinical settings for automated analysis of imaging reports, enabling accurate, fast and resource-efficient diagnostic categorization. 

---
# Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers 

**Authors**: Chaitanya Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2506.00054)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm to enhance large language models (LLMs) by conditioning generation on external evidence retrieved at inference time. While RAG addresses critical limitations of parametric knowledge storage-such as factual inconsistency and domain inflexibility-it introduces new challenges in retrieval quality, grounding fidelity, pipeline efficiency, and robustness against noisy or adversarial inputs. This survey provides a comprehensive synthesis of recent advances in RAG systems, offering a taxonomy that categorizes architectures into retriever-centric, generator-centric, hybrid, and robustness-oriented designs. We systematically analyze enhancements across retrieval optimization, context filtering, decoding control, and efficiency improvements, supported by comparative performance analyses on short-form and multi-hop question answering tasks. Furthermore, we review state-of-the-art evaluation frameworks and benchmarks, highlighting trends in retrieval-aware evaluation, robustness testing, and federated retrieval settings. Our analysis reveals recurring trade-offs between retrieval precision and generation flexibility, efficiency and faithfulness, and modularity and coordination. We conclude by identifying open challenges and future research directions, including adaptive retrieval architectures, real-time retrieval integration, structured reasoning over multi-hop evidence, and privacy-preserving retrieval mechanisms. This survey aims to consolidate current knowledge in RAG research and serve as a foundation for the next generation of retrieval-augmented language modeling systems. 

---
# Probing Audio-Generation Capabilities of Text-Based Language Models 

**Authors**: Arjun Prasaath Anbazhagan, Parteek Kumar, Ujjwal Kaur, Aslihan Akalin, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2506.00003)  

**Abstract**: How does textual representation of audio relate to the Large Language Model's (LLMs) learning about the audio world? This research investigates the extent to which LLMs can be prompted to generate audio, despite their primary training in textual data. We employ a three-tier approach, progressively increasing the complexity of audio generation: 1) Musical Notes, 2) Environmental Sounds, and 3) Human Speech. To bridge the gap between text and audio, we leverage code as an intermediary, prompting LLMs to generate code that, when executed, produces the desired audio output. To evaluate the quality and accuracy of the generated audio, we employ FAD and CLAP scores. Our findings reveal that while LLMs can generate basic audio features, their performance deteriorates as the complexity of the audio increases. This suggests that while LLMs possess a latent understanding of the auditory world, their ability to translate this understanding into tangible audio output remains rudimentary. Further research into techniques that can enhance the quality and diversity of LLM-generated audio can lead to an improvement in the performance of text-based LLMs in generating audio. 

---
# Enhancing Finite State Machine Design Automation with Large Language Models and Prompt Engineering Techniques 

**Authors**: Qun-Kai Lin, Cheng Hsu, Tian-Sheuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.00001)  

**Abstract**: Large Language Models (LLMs) have attracted considerable attention in recent years due to their remarkable compatibility with Hardware Description Language (HDL) design. In this paper, we examine the performance of three major LLMs, Claude 3 Opus, ChatGPT-4, and ChatGPT-4o, in designing finite state machines (FSMs). By utilizing the instructional content provided by HDLBits, we evaluate the stability, limitations, and potential approaches for improving the success rates of these models. Furthermore, we explore the impact of using the prompt-refining method, To-do-Oriented Prompting (TOP) Patch, on the success rate of these LLM models in various FSM design scenarios. The results show that the systematic format prompt method and the novel prompt refinement method have the potential to be applied to other domains beyond HDL design automation, considering its possible integration with other prompt engineering techniques in the future. 

---
# GLEN: Generative Retrieval via Lexical Index Learning 

**Authors**: Sunkyung Lee, Minjin Choi, Jongwuk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2311.03057)  

**Abstract**: Generative retrieval shed light on a new paradigm of document retrieval, aiming to directly generate the identifier of a relevant document for a query. While it takes advantage of bypassing the construction of auxiliary index structures, existing studies face two significant challenges: (i) the discrepancy between the knowledge of pre-trained language models and identifiers and (ii) the gap between training and inference that poses difficulty in learning to rank. To overcome these challenges, we propose a novel generative retrieval method, namely Generative retrieval via LExical iNdex learning (GLEN). For training, GLEN effectively exploits a dynamic lexical identifier using a two-phase index learning strategy, enabling it to learn meaningful lexical identifiers and relevance signals between queries and documents. For inference, GLEN utilizes collision-free inference, using identifier weights to rank documents without additional overhead. Experimental results prove that GLEN achieves state-of-the-art or competitive performance against existing generative retrieval methods on various benchmark datasets, e.g., NQ320k, MS MARCO, and BEIR. The code is available at this https URL. 

---
