# ComPO: Preference Alignment via Comparison Oracles 

**Authors**: Peter Chen, Xi Chen, Wotao Yin, Tianyi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05465)  

**Abstract**: Direct alignment methods are increasingly used for aligning large language models (LLMs) with human preferences. However, these methods suffer from the issues of verbosity and likelihood displacement, which can be driven by the noisy preference pairs that induce similar likelihood for preferred and dispreferred responses. The contributions of this paper are two-fold. First, we propose a new preference alignment method based on comparison oracles and provide the convergence guarantee for its basic scheme. Second, we improve our method using some heuristics and conduct the experiments to demonstrate the flexibility and compatibility of practical scheme in improving the performance of LLMs using noisy preference pairs. Evaluations are conducted across multiple base and instruction-tuned models (Mistral-7B, Llama-3-8B and Gemma-2-9B) with benchmarks (AlpacaEval 2, MT-Bench and Arena-Hard). Experimental results show the effectiveness of our method as an alternative to addressing the limitations of existing direct alignment methods. A highlight of our work is that we evidence the importance of designing specialized methods for preference pairs with distinct likelihood margin, which complements the recent findings in \citet{Razin-2025-Unintentional}. 

---
# Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging 

**Authors**: Shiqi Chen, Jinghan Zhang, Tongyao Zhu, Wei Liu, Siyang Gao, Miao Xiong, Manling Li, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05464)  

**Abstract**: Vision-Language Models (VLMs) combine visual perception with the general capabilities, such as reasoning, of Large Language Models (LLMs). However, the mechanisms by which these two abilities can be combined and contribute remain poorly understood. In this work, we explore to compose perception and reasoning through model merging that connects parameters of different models. Unlike previous works that often focus on merging models of the same kind, we propose merging models across modalities, enabling the incorporation of the reasoning capabilities of LLMs into VLMs. Through extensive experiments, we demonstrate that model merging offers a successful pathway to transfer reasoning abilities from LLMs to VLMs in a training-free manner. Moreover, we utilize the merged models to understand the internal mechanism of perception and reasoning and how merging affects it. We find that perception capabilities are predominantly encoded in the early layers of the model, whereas reasoning is largely facilitated by the middle-to-late layers. After merging, we observe that all layers begin to contribute to reasoning, whereas the distribution of perception abilities across layers remains largely unchanged. These observations shed light on the potential of model merging as a tool for multimodal integration and interpretation. 

---
# UKElectionNarratives: A Dataset of Misleading Narratives Surrounding Recent UK General Elections 

**Authors**: Fatima Haouari, Carolina Scarton, Nicolò Faggiani, Nikolaos Nikolaidis, Bonka Kotseva, Ibrahim Abu Farha, Jens Linge, Kalina Bontcheva  

**Link**: [PDF](https://arxiv.org/pdf/2505.05459)  

**Abstract**: Misleading narratives play a crucial role in shaping public opinion during elections, as they can influence how voters perceive candidates and political parties. This entails the need to detect these narratives accurately. To address this, we introduce the first taxonomy of common misleading narratives that circulated during recent elections in Europe. Based on this taxonomy, we construct and analyse UKElectionNarratives: the first dataset of human-annotated misleading narratives which circulated during the UK General Elections in 2019 and 2024. We also benchmark Pre-trained and Large Language Models (focusing on GPT-4o), studying their effectiveness in detecting election-related misleading narratives. Finally, we discuss potential use cases and make recommendations for future research directions using the proposed codebook and dataset. 

---
# clem:todd: A Framework for the Systematic Benchmarking of LLM-Based Task-Oriented Dialogue System Realisations 

**Authors**: Chalamalasetti Kranti, Sherzod Hakimov, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05445)  

**Abstract**: The emergence of instruction-tuned large language models (LLMs) has advanced the field of dialogue systems, enabling both realistic user simulations and robust multi-turn conversational agents. However, existing research often evaluates these components in isolation-either focusing on a single user simulator or a specific system design-limiting the generalisability of insights across architectures and configurations. In this work, we propose clem todd (chat-optimized LLMs for task-oriented dialogue systems development), a flexible framework for systematically evaluating dialogue systems under consistent conditions. clem todd enables detailed benchmarking across combinations of user simulators and dialogue systems, whether existing models from literature or newly developed ones. It supports plug-and-play integration and ensures uniform datasets, evaluation metrics, and computational constraints. We showcase clem todd's flexibility by re-evaluating existing task-oriented dialogue systems within this unified setup and integrating three newly proposed dialogue systems into the same evaluation pipeline. Our results provide actionable insights into how architecture, scale, and prompting strategies affect dialogue performance, offering practical guidance for building efficient and effective conversational AI systems. 

---
# Ultra-FineWeb: Efficient Data Filtering and Verification for High-Quality LLM Training Data 

**Authors**: Yudong Wang, Zixuan Fu, Jie Cai, Peijun Tang, Hongya Lyu, Yewei Fang, Zhi Zheng, Jie Zhou, Guoyang Zeng, Chaojun Xiao, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05427)  

**Abstract**: Data quality has become a key factor in enhancing model performance with the rapid development of large language models (LLMs). Model-driven data filtering has increasingly become a primary approach for acquiring high-quality data. However, it still faces two main challenges: (1) the lack of an efficient data verification strategy makes it difficult to provide timely feedback on data quality; and (2) the selection of seed data for training classifiers lacks clear criteria and relies heavily on human expertise, introducing a degree of subjectivity. To address the first challenge, we introduce an efficient verification strategy that enables rapid evaluation of the impact of data on LLM training with minimal computational cost. To tackle the second challenge, we build upon the assumption that high-quality seed data is beneficial for LLM training, and by integrating the proposed verification strategy, we optimize the selection of positive and negative samples and propose an efficient data filtering pipeline. This pipeline not only improves filtering efficiency, classifier quality, and robustness, but also significantly reduces experimental and inference costs. In addition, to efficiently filter high-quality data, we employ a lightweight classifier based on fastText, and successfully apply the filtering pipeline to two widely-used pre-training corpora, FineWeb and Chinese FineWeb datasets, resulting in the creation of the higher-quality Ultra-FineWeb dataset. Ultra-FineWeb contains approximately 1 trillion English tokens and 120 billion Chinese tokens. Empirical results demonstrate that the LLMs trained on Ultra-FineWeb exhibit significant performance improvements across multiple benchmark tasks, validating the effectiveness of our pipeline in enhancing both data quality and training efficiency. 

---
# TransProQA: an LLM-based literary Translation evaluation metric with Professional Question Answering 

**Authors**: Ran Zhang, Wei Zhao, Lieve Macken, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2505.05423)  

**Abstract**: The impact of Large Language Models (LLMs) has extended into literary domains. However, existing evaluation metrics prioritize mechanical accuracy over artistic expression and tend to overrate machine translation (MT) as being superior to experienced professional human translation. In the long run, this bias could result in a permanent decline in translation quality and cultural authenticity. In response to the urgent need for a specialized literary evaluation metric, we introduce TransProQA, a novel, reference-free, LLM-based question-answering (QA) framework designed specifically for literary translation evaluation. TransProQA uniquely integrates insights from professional literary translators and researchers, focusing on critical elements in literary quality assessment such as literary devices, cultural understanding, and authorial voice. Our extensive evaluation shows that while literary-finetuned XCOMET-XL yields marginal gains, TransProQA substantially outperforms current metrics, achieving up to 0.07 gain in correlation (ACC-EQ and Kendall's tau) and surpassing the best state-of-the-art (SOTA) metrics by over 15 points in adequacy assessments. Incorporating professional translator insights as weights further improves performance, highlighting the value of translator inputs. Notably, TransProQA approaches human-level evaluation performance comparable to trained linguistic annotators. It demonstrates broad applicability to open-source models such as LLaMA3.3-70b and Qwen2.5-32b, indicating its potential as an accessible and training-free literary evaluation metric and a valuable tool for evaluating texts that require local processing due to copyright or ethical considerations. 

---
# Reasoning Models Don't Always Say What They Think 

**Authors**: Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.05410)  

**Abstract**: Chain-of-thought (CoT) offers a potential boon for AI safety as it allows monitoring a model's CoT to try to understand its intentions and reasoning processes. However, the effectiveness of such monitoring hinges on CoTs faithfully representing models' actual reasoning processes. We evaluate CoT faithfulness of state-of-the-art reasoning models across 6 reasoning hints presented in the prompts and find: (1) for most settings and models tested, CoTs reveal their usage of hints in at least 1% of examples where they use the hint, but the reveal rate is often below 20%, (2) outcome-based reinforcement learning initially improves faithfulness but plateaus without saturating, and (3) when reinforcement learning increases how frequently hints are used (reward hacking), the propensity to verbalize them does not increase, even without training against a CoT monitor. These results suggest that CoT monitoring is a promising way of noticing undesired behaviors during training and evaluations, but that it is not sufficient to rule them out. They also suggest that in settings like ours where CoT reasoning is not necessary, test-time monitoring of CoTs is unlikely to reliably catch rare and catastrophic unexpected behaviors. 

---
# Crosslingual Reasoning through Test-Time Scaling 

**Authors**: Zheng-Xin Yong, M. Farid Adilazuarda, Jonibek Mansurov, Ruochen Zhang, Niklas Muennighoff, Carsten Eickhoff, Genta Indra Winata, Julia Kreutzer, Stephen H. Bach, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2505.05408)  

**Abstract**: Reasoning capabilities of large language models are primarily studied for English, even when pretrained models are multilingual. In this work, we investigate to what extent English reasoning finetuning with long chain-of-thoughts (CoTs) can generalize across languages. First, we find that scaling up inference compute for English-centric reasoning language models (RLMs) improves multilingual mathematical reasoning across many languages including low-resource languages, to an extent where they outperform models twice their size. Second, we reveal that while English-centric RLM's CoTs are naturally predominantly English, they consistently follow a quote-and-think pattern to reason about quoted non-English inputs. Third, we discover an effective strategy to control the language of long CoT reasoning, and we observe that models reason better and more efficiently in high-resource languages. Finally, we observe poor out-of-domain reasoning generalization, in particular from STEM to cultural commonsense knowledge, even for English. Overall, we demonstrate the potentials, study the mechanisms and outline the limitations of crosslingual generalization of English reasoning test-time scaling. We conclude that practitioners should let English-centric RLMs reason in high-resource languages, while further work is needed to improve reasoning in low-resource languages and out-of-domain contexts. 

---
# Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans? 

**Authors**: Valeria Pastorino, Nafise Sadat Moosavi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05406)  

**Abstract**: Framing in media critically shapes public perception by selectively emphasizing some details while downplaying others. With the rise of large language models in automated news and content creation, there is growing concern that these systems may introduce or even amplify framing biases compared to human authors. In this paper, we explore how framing manifests in both out-of-the-box and fine-tuned LLM-generated news content. Our analysis reveals that, particularly in politically and socially sensitive contexts, LLMs tend to exhibit more pronounced framing than their human counterparts. In addition, we observe significant variation in framing tendencies across different model architectures, with some models displaying notably higher biases. These findings point to the need for effective post-training mitigation strategies and tighter evaluation frameworks to ensure that automated news content upholds the standards of balanced reporting. 

---
# ICon: In-Context Contribution for Automatic Data Selection 

**Authors**: Yixin Yang, Qingxiu Dong, Linli Yao, Fangwei Zhu, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.05327)  

**Abstract**: Data selection for instruction tuning is essential for improving the performance of Large Language Models (LLMs) and reducing training cost. However, existing automated selection methods either depend on computationally expensive gradient-based measures or manually designed heuristics, which may fail to fully exploit the intrinsic attributes of data. In this paper, we propose In-context Learning for Contribution Measurement (ICon), a novel gradient-free method that takes advantage of the implicit fine-tuning nature of in-context learning (ICL) to measure sample contribution without gradient computation or manual indicators engineering. ICon offers a computationally efficient alternative to gradient-based methods and reduces human inductive bias inherent in heuristic-based approaches. ICon comprises three components and identifies high-contribution data by assessing performance shifts under implicit learning through ICL. Extensive experiments on three LLMs across 12 benchmarks and 5 pairwise evaluation sets demonstrate the effectiveness of ICon. Remarkably, on LLaMA3.1-8B, models trained on 15% of ICon-selected data outperform full datasets by 5.42% points and exceed the best performance of widely used selection methods by 2.06% points. We further analyze high-contribution samples selected by ICon, which show both diverse tasks and appropriate difficulty levels, rather than just the hardest ones. 

---
# Toward Reasonable Parrots: Why Large Language Models Should Argue with Us by Design 

**Authors**: Elena Musi, Nadin Kokciyan, Khalid Al-Khatib, Davide Ceolin, Emmanuelle Dietz, Klara Gutekunst, Annette Hautli-Janisz, Cristian Manuel Santibañez Yañez, Jodi Schneider, Jonas Scholz, Cor Steging, Jacky Visser, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2505.05298)  

**Abstract**: In this position paper, we advocate for the development of conversational technology that is inherently designed to support and facilitate argumentative processes. We argue that, at present, large language models (LLMs) are inadequate for this purpose, and we propose an ideal technology design aimed at enhancing argumentative skills. This involves re-framing LLMs as tools to exercise our critical thinking rather than replacing them. We introduce the concept of 'reasonable parrots' that embody the fundamental principles of relevance, responsibility, and freedom, and that interact through argumentative dialogical moves. These principles and moves arise out of millennia of work in argumentation theory and should serve as the starting point for LLM-based technology that incorporates basic principles of argumentation. 

---
# T-T: Table Transformer for Tagging-based Aspect Sentiment Triplet Extraction 

**Authors**: Kun Peng, Chaodong Tong, Cong Cao, Hao Peng, Qian Li, Guanlin Wu, Lei Jiang, Yanbing Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05271)  

**Abstract**: Aspect sentiment triplet extraction (ASTE) aims to extract triplets composed of aspect terms, opinion terms, and sentiment polarities from given sentences. The table tagging method is a popular approach to addressing this task, which encodes a sentence into a 2-dimensional table, allowing for the tagging of relations between any two words. Previous efforts have focused on designing various downstream relation learning modules to better capture interactions between tokens in the table, revealing that a stronger capability to capture relations can lead to greater improvements in the model. Motivated by this, we attempt to directly utilize transformer layers as downstream relation learning modules. Due to the powerful semantic modeling capability of transformers, it is foreseeable that this will lead to excellent improvement. However, owing to the quadratic relation between the length of the table and the length of the input sentence sequence, using transformers directly faces two challenges: overly long table sequences and unfair local attention interaction. To address these challenges, we propose a novel Table-Transformer (T-T) for the tagging-based ASTE method. Specifically, we introduce a stripe attention mechanism with a loop-shift strategy to tackle these challenges. The former modifies the global attention mechanism to only attend to a 2-dimensional local attention window, while the latter facilitates interaction between different attention windows. Extensive and comprehensive experiments demonstrate that the T-T, as a downstream relation learning module, achieves state-of-the-art performance with lower computational costs. 

---
# QualBench: Benchmarking Chinese LLMs with Localized Professional Qualifications for Vertical Domain Evaluation 

**Authors**: Mengze Hong, Wailing Ng, Di Jiang, Chen Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05225)  

**Abstract**: The rapid advancement of Chinese large language models (LLMs) underscores the need for domain-specific evaluations to ensure reliable applications. However, existing benchmarks often lack coverage in vertical domains and offer limited insights into the Chinese working context. Leveraging qualification exams as a unified framework for human expertise evaluation, we introduce QualBench, the first multi-domain Chinese QA benchmark dedicated to localized assessment of Chinese LLMs. The dataset includes over 17,000 questions across six vertical domains, with data selections grounded in 24 Chinese qualifications to closely align with national policies and working standards. Through comprehensive evaluation, the Qwen2.5 model outperformed the more advanced GPT-4o, with Chinese LLMs consistently surpassing non-Chinese models, highlighting the importance of localized domain knowledge in meeting qualification requirements. The best performance of 75.26% reveals the current gaps in domain coverage within model capabilities. Furthermore, we present the failure of LLM collaboration with crowdsourcing mechanisms and suggest the opportunities for multi-domain RAG knowledge enhancement and vertical domain LLM training with Federated Learning. 

---
# A Benchmark Dataset and a Framework for Urdu Multimodal Named Entity Recognition 

**Authors**: Hussain Ahmad, Qingyang Zeng, Jing Wan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05148)  

**Abstract**: The emergence of multimodal content, particularly text and images on social media, has positioned Multimodal Named Entity Recognition (MNER) as an increasingly important area of research within Natural Language Processing. Despite progress in high-resource languages such as English, MNER remains underexplored for low-resource languages like Urdu. The primary challenges include the scarcity of annotated multimodal datasets and the lack of standardized baselines. To address these challenges, we introduce the U-MNER framework and release the Twitter2015-Urdu dataset, a pioneering resource for Urdu MNER. Adapted from the widely used Twitter2015 dataset, it is annotated with Urdu-specific grammar rules. We establish benchmark baselines by evaluating both text-based and multimodal models on this dataset, providing comparative analyses to support future research on Urdu MNER. The U-MNER framework integrates textual and visual context using Urdu-BERT for text embeddings and ResNet for visual feature extraction, with a Cross-Modal Fusion Module to align and fuse information. Our model achieves state-of-the-art performance on the Twitter2015-Urdu dataset, laying the groundwork for further MNER research in low-resource languages. 

---
# Unveiling Language-Specific Features in Large Language Models via Sparse Autoencoders 

**Authors**: Boyi Deng, Yu Wan, Yidan Zhang, Baosong Yang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05111)  

**Abstract**: The mechanisms behind multilingual capabilities in Large Language Models (LLMs) have been examined using neuron-based or internal-activation-based methods. However, these methods often face challenges such as superposition and layer-wise activation variance, which limit their reliability. Sparse Autoencoders (SAEs) offer a more nuanced analysis by decomposing the activations of LLMs into sparse linear combination of SAE features. We introduce a novel metric to assess the monolinguality of features obtained from SAEs, discovering that some features are strongly related to specific languages. Additionally, we show that ablating these SAE features only significantly reduces abilities in one language of LLMs, leaving others almost unaffected. Interestingly, we find some languages have multiple synergistic SAE features, and ablating them together yields greater improvement than ablating individually. Moreover, we leverage these SAE-derived language-specific features to enhance steering vectors, achieving control over the language generated by LLMs. 

---
# Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction 

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05084)  

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets. 

---
# Performance Evaluation of Large Language Models in Bangla Consumer Health Query Summarization 

**Authors**: Ajwad Abrar, Farzana Tabassum, Sabbir Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2505.05070)  

**Abstract**: Consumer Health Queries (CHQs) in Bengali (Bangla), a low-resource language, often contain extraneous details, complicating efficient medical responses. This study investigates the zero-shot performance of nine advanced large language models (LLMs): GPT-3.5-Turbo, GPT-4, Claude-3.5-Sonnet, Llama3-70b-Instruct, Mixtral-8x22b-Instruct, Gemini-1.5-Pro, Qwen2-72b-Instruct, Gemma-2-27b, and Athene-70B, in summarizing Bangla CHQs. Using the BanglaCHQ-Summ dataset comprising 2,350 annotated query-summary pairs, we benchmarked these LLMs using ROUGE metrics against Bangla T5, a fine-tuned state-of-the-art model. Mixtral-8x22b-Instruct emerged as the top performing model in ROUGE-1 and ROUGE-L, while Bangla T5 excelled in ROUGE-2. The results demonstrate that zero-shot LLMs can rival fine-tuned models, achieving high-quality summaries even without task-specific training. This work underscores the potential of LLMs in addressing challenges in low-resource languages, providing scalable solutions for healthcare query summarization. 

---
# Teochew-Wild: The First In-the-wild Teochew Dataset with Orthographic Annotations 

**Authors**: Linrong Pan, Chenglong Jiang, Gaoze Hou, Ying Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.05056)  

**Abstract**: This paper reports the construction of the Teochew-Wild, a speech corpus of the Teochew dialect. The corpus includes 18.9 hours of in-the-wild Teochew speech data from multiple speakers, covering both formal and colloquial expressions, with precise orthographic and pinyin annotations. Additionally, we provide supplementary text processing tools and resources to propel research and applications in speech tasks for this low-resource language, such as automatic speech recognition (ASR) and text-to-speech (TTS). To the best of our knowledge, this is the first publicly available Teochew dataset with accurate orthographic annotations. We conduct experiments on the corpus, and the results validate its effectiveness in ASR and TTS tasks. 

---
# Image-Text Relation Prediction for Multilingual Tweets 

**Authors**: Matīss Rikters, Edison Marrese-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2505.05040)  

**Abstract**: Various social networks have been allowing media uploads for over a decade now. Still, it has not always been clear what is their relation with the posted text or even if there is any at all. In this work, we explore how multilingual vision-language models tackle the task of image-text relation prediction in different languages, and construct a dedicated balanced benchmark data set from Twitter posts in Latvian along with their manual translations into English. We compare our results to previous work and show that the more recently released vision-language model checkpoints are becoming increasingly capable at this task, but there is still much room for further improvement. 

---
# G-FOCUS: Towards a Robust Method for Assessing UI Design Persuasiveness 

**Authors**: Jaehyun Jeon, Janghan Yoon, Minsoo Kim, Sumin Shim, Yejin Choi, Hanbin Kim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05026)  

**Abstract**: Evaluating user interface (UI) design effectiveness extends beyond aesthetics to influencing user behavior, a principle central to Design Persuasiveness. A/B testing is the predominant method for determining which UI variations drive higher user engagement, but it is costly and time-consuming. While recent Vision-Language Models (VLMs) can process automated UI analysis, current approaches focus on isolated design attributes rather than comparative persuasiveness-the key factor in optimizing user interactions. To address this, we introduce WiserUI-Bench, a benchmark designed for Pairwise UI Design Persuasiveness Assessment task, featuring 300 real-world UI image pairs labeled with A/B test results and expert rationales. Additionally, we propose G-FOCUS, a novel inference-time reasoning strategy that enhances VLM-based persuasiveness assessment by reducing position bias and improving evaluation accuracy. Experimental results show that G-FOCUS surpasses existing inference strategies in consistency and accuracy for pairwise UI evaluation. Through promoting VLM-driven evaluation of UI persuasiveness, our work offers an approach to complement A/B testing, propelling progress in scalable UI preference modeling and design optimization. Code and data will be released publicly. 

---
# Scalable Multi-Stage Influence Function for Large Language Models via Eigenvalue-Corrected Kronecker-Factored Parameterization 

**Authors**: Yuntai Bao, Xuhong Zhang, Tianyu Du, Xinkui Zhao, Jiang Zong, Hao Peng, Jianwei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05017)  

**Abstract**: Pre-trained large language models (LLMs) are commonly fine-tuned to adapt to downstream tasks. Since the majority of knowledge is acquired during pre-training, attributing the predictions of fine-tuned LLMs to their pre-training data may provide valuable insights. Influence functions have been proposed as a means to explain model predictions based on training data. However, existing approaches fail to compute ``multi-stage'' influence and lack scalability to billion-scale LLMs.
In this paper, we propose the multi-stage influence function to attribute the downstream predictions of fine-tuned LLMs to pre-training data under the full-parameter fine-tuning paradigm. To enhance the efficiency and practicality of our multi-stage influence function, we leverage Eigenvalue-corrected Kronecker-Factored (EK-FAC) parameterization for efficient approximation. Empirical results validate the superior scalability of EK-FAC approximation and the effectiveness of our multi-stage influence function. Additionally, case studies on a real-world LLM, dolly-v2-3b, demonstrate its interpretive power, with exemplars illustrating insights provided by multi-stage influence estimates. Our code is public at this https URL. 

---
# The Pitfalls of Growing Group Complexity: LLMs and Social Choice-Based Aggregation for Group Recommendations 

**Authors**: Cedric Waterschoot, Nava Tintarev, Francesco Barile  

**Link**: [PDF](https://arxiv.org/pdf/2505.05016)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in recommender systems aimed at both individuals and groups. Previously, Group Recommender Systems (GRS) often used social choice-based aggregation strategies to derive a single recommendation based on the preferences of multiple people. In this paper, we investigate under which conditions language models can perform these strategies correctly based on zero-shot learning and analyse whether the formatting of the group scenario in the prompt affects accuracy. We specifically focused on the impact of group complexity (number of users and items), different LLMs, different prompting conditions, including In-Context learning or generating explanations, and the formatting of group preferences. Our results show that performance starts to deteriorate when considering more than 100 ratings. However, not all language models were equally sensitive to growing group complexity. Additionally, we showed that In-Context Learning (ICL) can significantly increase the performance at higher degrees of group complexity, while adding other prompt modifications, specifying domain cues or prompting for explanations, did not impact accuracy. We conclude that future research should include group complexity as a factor in GRS evaluation due to its effect on LLM performance. Furthermore, we showed that formatting the group scenarios differently, such as rating lists per user or per item, affected accuracy. All in all, our study implies that smaller LLMs are capable of generating group recommendations under the right conditions, making the case for using smaller models that require less computing power and costs. 

---
# Rethinking Invariance in In-context Learning 

**Authors**: Lizhe Fang, Yifei Wang, Khashayar Gatmiry, Lei Fang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04994)  

**Abstract**: In-Context Learning (ICL) has emerged as a pivotal capability of auto-regressive large language models, yet it is hindered by a notable sensitivity to the ordering of context examples regardless of their mutual independence. To address this issue, recent studies have introduced several variant algorithms of ICL that achieve permutation invariance. However, many of these do not exhibit comparable performance with the standard auto-regressive ICL algorithm. In this work, we identify two crucial elements in the design of an invariant ICL algorithm: information non-leakage and context interdependence, which are not simultaneously achieved by any of the existing methods. These investigations lead us to the proposed Invariant ICL (InvICL), a methodology designed to achieve invariance in ICL while ensuring the two properties. Empirically, our findings reveal that InvICL surpasses previous models, both invariant and non-invariant, in most benchmark datasets, showcasing superior generalization capabilities across varying input lengths. Code is available at this https URL. 

---
# Latent Preference Coding: Aligning Large Language Models via Discrete Latent Codes 

**Authors**: Zhuocheng Gong, Jian Guan, Wei Wu, Huishuai Zhang, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04993)  

**Abstract**: Large language models (LLMs) have achieved remarkable success, yet aligning their generations with human preferences remains a critical challenge. Existing approaches to preference modeling often rely on an explicit or implicit reward function, overlooking the intricate and multifaceted nature of human preferences that may encompass conflicting factors across diverse tasks and populations. To address this limitation, we introduce Latent Preference Coding (LPC), a novel framework that models the implicit factors as well as their combinations behind holistic preferences using discrete latent codes. LPC seamlessly integrates with various offline alignment algorithms, automatically inferring the underlying factors and their importance from data without relying on pre-defined reward functions and hand-crafted combination weights. Extensive experiments on multiple benchmarks demonstrate that LPC consistently improves upon three alignment algorithms (DPO, SimPO, and IPO) using three base models (Mistral-7B, Llama3-8B, and Llama3-8B-Instruct). Furthermore, deeper analysis reveals that the learned latent codes effectively capture the differences in the distribution of human preferences and significantly enhance the robustness of alignment against noise in data. By providing a unified representation for the multifarious preference factors, LPC paves the way towards developing more robust and versatile alignment techniques for the responsible deployment of powerful LLMs. 

---
# Rethinking the Relationship between the Power Law and Hierarchical Structures 

**Authors**: Kai Nakaishi, Ryo Yoshida, Kohei Kajikawa, Koji Hukushima, Yohei Oseki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04984)  

**Abstract**: Statistical analysis of corpora provides an approach to quantitatively investigate natural languages. This approach has revealed that several power laws consistently emerge across different corpora and languages, suggesting the universal principles underlying languages. Particularly, the power-law decay of correlation has been interpreted as evidence for underlying hierarchical structures in syntax, semantics, and discourse. This perspective has also been extended to child languages and animal signals. However, the argument supporting this interpretation has not been empirically tested. To address this problem, this study examines the validity of the argument for syntactic structures. Specifically, we test whether the statistical properties of parse trees align with the implicit assumptions in the argument. Using English corpora, we analyze the mutual information, deviations from probabilistic context-free grammars (PCFGs), and other properties in parse trees, as well as in the PCFG that approximates these trees. Our results indicate that the assumptions do not hold for syntactic structures and that it is difficult to apply the proposed argument to child languages and animal signals, highlighting the need to reconsider the relationship between the power law and hierarchical structures. 

---
# Chain-of-Thought Tokens are Computer Program Variables 

**Authors**: Fangwei Zhu, Peiyi Wang, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04955)  

**Abstract**: Chain-of-thoughts (CoT) requires large language models (LLMs) to generate intermediate steps before reaching the final answer, and has been proven effective to help LLMs solve complex reasoning tasks. However, the inner mechanism of CoT still remains largely unclear. In this paper, we empirically study the role of CoT tokens in LLMs on two compositional tasks: multi-digit multiplication and dynamic programming. While CoT is essential for solving these problems, we find that preserving only tokens that store intermediate results would achieve comparable performance. Furthermore, we observe that storing intermediate results in an alternative latent form will not affect model performance. We also randomly intervene some values in CoT, and notice that subsequent CoT tokens and the final answer would change correspondingly. These findings suggest that CoT tokens may function like variables in computer programs but with potential drawbacks like unintended shortcuts and computational complexity limits between tokens. The code and data are available at this https URL. 

---
# An Open-Source Dual-Loss Embedding Model for Semantic Retrieval in Higher Education 

**Authors**: Ramteja Sajja, Yusuf Sermet, Ibrahim Demir  

**Link**: [PDF](https://arxiv.org/pdf/2505.04916)  

**Abstract**: Recent advances in AI have catalyzed the adoption of intelligent educational tools, yet many semantic retrieval systems remain ill-suited to the unique linguistic and structural characteristics of academic content. This study presents two open-source embedding models fine-tuned for educational question answering, particularly in the context of course syllabi. A synthetic dataset of 3,197 sentence pairs, spanning synonymous terminology, paraphrased questions, and implicit-explicit mappings, was constructed through a combination of manual curation and large language model (LLM)-assisted generation. Two training strategies were evaluated: (1) a baseline model fine-tuned using MultipleNegativesRankingLoss (MNRL), and (2) a dual-loss model that combines MNRL with CosineSimilarityLoss to improve both semantic ranking and similarity calibration. Evaluations were conducted on 28 university course syllabi using a fixed set of natural language questions categorized into course, faculty, and teaching assistant information. Results demonstrate that both fine-tuned models outperform strong open-source baselines, including all-MiniLM-L6-v2 and multi-qa-MiniLM-L6-cos-v1, and that the dual-loss model narrows the performance gap with high-performing proprietary embeddings such as OpenAI's text-embedding-3 series. This work contributes reusable, domain-aligned embedding models and provides a replicable framework for educational semantic retrieval, supporting downstream applications such as academic chatbots, retrieval-augmented generation (RAG) systems, and learning management system (LMS) integrations. 

---
# Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards 

**Authors**: Manveer Singh Tamber, Forrest Sheng Bao, Chenyu Xu, Ge Luo, Suleman Kazi, Minseok Bae, Miaoran Li, Ofer Mendelevitch, Renyi Qu, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04847)  

**Abstract**: Hallucinations remain a persistent challenge for LLMs. RAG aims to reduce hallucinations by grounding responses in contexts. However, even when provided context, LLMs still frequently introduce unsupported information or contradictions. This paper presents our efforts to measure LLM hallucinations with a focus on summarization tasks, assessing how often various LLMs introduce hallucinations when summarizing documents. We discuss Vectara's existing LLM hallucination leaderboard, based on the Hughes Hallucination Evaluation Model (HHEM). While HHEM and Vectara's Hallucination Leaderboard have garnered great research interest, we examine challenges faced by HHEM and current hallucination detection methods by analyzing the effectiveness of these methods on existing hallucination datasets. To address these limitations, we propose FaithJudge, an LLM-as-a-judge approach guided by few-shot human hallucination annotations, which substantially improves automated LLM hallucination evaluation over current methods. We introduce an enhanced hallucination leaderboard centered on FaithJudge, alongside our current hallucination leaderboard, enabling more reliable benchmarking of LLMs for hallucinations in RAG. 

---
# Osiris: A Lightweight Open-Source Hallucination Detection System 

**Authors**: Alex Shan, John Bauer, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2505.04844)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have gained widespread adoption by application builders because they leverage sources of truth to enable Large Language Models (LLMs) to generate more factually sound responses. However, hallucinations, instances of LLM responses that are unfaithful to the provided context, often prevent these systems from being deployed in production environments. Current hallucination detection methods typically involve human evaluation or the use of closed-source models to review RAG system outputs for hallucinations. Both human evaluators and closed-source models suffer from scaling issues due to their high costs and slow inference speeds. In this work, we introduce a perturbed multi-hop QA dataset with induced hallucinations. Via supervised fine-tuning on our dataset, we achieve better recall with a 7B model than GPT-4o on the RAGTruth hallucination detection benchmark and offer competitive performance on precision and accuracy, all while using a fraction of the parameters. Code is released at our repository. 

---
# Flower Across Time and Media: Sentiment Analysis of Tang Song Poetry and Visual Correspondence 

**Authors**: Shuai Gong, Tiange Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04785)  

**Abstract**: The Tang (618 to 907) and Song (960 to 1279) dynasties witnessed an extraordinary flourishing of Chinese cultural expression, where floral motifs served as a dynamic medium for both poetic sentiment and artistic design. While previous scholarship has examined these domains independently, the systematic correlation between evolving literary emotions and visual culture remains underexplored. This study addresses that gap by employing BERT-based sentiment analysis to quantify emotional patterns in floral imagery across Tang Song poetry, then validating these patterns against contemporaneous developments in decorative this http URL approach builds upon recent advances in computational humanities while remaining grounded in traditional sinological methods. By applying a fine tuned BERT model to analyze peony and plum blossom imagery in classical poetry, we detect measurable shifts in emotional connotations between the Tang and Song periods. These textual patterns are then cross berenced with visual evidence from textiles, ceramics, and other material culture, revealing previously unrecognized synergies between literary expression and artistic representation. 

---
# SOAEsV2-7B/72B: Full-Pipeline Optimization for State-Owned Enterprise LLMs via Continual Pre-Training, Domain-Progressive SFT and Distillation-Enhanced Speculative Decoding 

**Authors**: Jingyang Deng, Ran Chen, Jo-Ku Cheng, Jinwen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.04723)  

**Abstract**: This study addresses key challenges in developing domain-specific large language models (LLMs) for Chinese state-owned assets and enterprises (SOAEs), where current approaches face three limitations: 1) constrained model capacity that limits knowledge integration and cross-task adaptability; 2) excessive reliance on domain-specific supervised fine-tuning (SFT) data, which neglects the broader applicability of general language patterns; and 3) inefficient inference acceleration for large models processing long contexts. In this work, we propose SOAEsV2-7B/72B, a specialized LLM series developed via a three-phase framework: 1) continual pre-training integrates domain knowledge while retaining base capabilities; 2) domain-progressive SFT employs curriculum-based learning strategy, transitioning from weakly relevant conversational data to expert-annotated SOAEs datasets to optimize domain-specific tasks; 3) distillation-enhanced speculative decoding accelerates inference via logit distillation between 72B target and 7B draft models, achieving 1.39-1.52$\times$ speedup without quality loss. Experimental results demonstrate that our domain-specific pre-training phase maintains 99.8% of original general language capabilities while significantly improving domain performance, resulting in a 1.08$\times$ improvement in Rouge-1 score and a 1.17$\times$ enhancement in BLEU-4 score. Ablation studies further show that domain-progressive SFT outperforms single-stage training, achieving 1.02$\times$ improvement in Rouge-1 and 1.06$\times$ in BLEU-4. Our work introduces a comprehensive, full-pipeline approach for optimizing SOAEs LLMs, bridging the gap between general language capabilities and domain-specific expertise. 

---
# Advanced Deep Learning Approaches for Automated Recognition of Cuneiform Symbols 

**Authors**: Shahad Elshehaby, Alavikunhu Panthakkan, Hussain Al-Ahmad, Mina Al-Saad  

**Link**: [PDF](https://arxiv.org/pdf/2505.04678)  

**Abstract**: This paper presents a thoroughly automated method for identifying and interpreting cuneiform characters via advanced deep-learning algorithms. Five distinct deep-learning models were trained on a comprehensive dataset of cuneiform characters and evaluated according to critical performance metrics, including accuracy and precision. Two models demonstrated outstanding performance and were subsequently assessed using cuneiform symbols from the Hammurabi law acquisition, notably Hammurabi Law 1. Each model effectively recognized the relevant Akkadian meanings of the symbols and delivered precise English translations. Future work will investigate ensemble and stacking approaches to optimize performance, utilizing hybrid architectures to improve detection accuracy and reliability. This research explores the linguistic relationships between Akkadian, an ancient Mesopotamian language, and Arabic, emphasizing their historical and cultural linkages. This study demonstrates the capability of deep learning to decipher ancient scripts by merging computational linguistics with archaeology, therefore providing significant insights for the comprehension and conservation of human history. 

---
# REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM 

**Authors**: Madhur Jindal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2505.04673)  

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.
We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$). 

---
# Reward-SQL: Boosting Text-to-SQL via Stepwise Reasoning and Process-Supervised Rewards 

**Authors**: Yuxin Zhang, Meihao Fan, Ju Fan, Mingyang Yi, Yuyu Luo, Jian Tan, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04671)  

**Abstract**: Recent advances in large language models (LLMs) have significantly improved performance on the Text-to-SQL task by leveraging their powerful reasoning capabilities. To enhance accuracy during the reasoning process, external Process Reward Models (PRMs) can be introduced during training and inference to provide fine-grained supervision. However, if misused, PRMs may distort the reasoning trajectory and lead to suboptimal or incorrect SQL this http URL address this challenge, we propose Reward-SQL, a framework that systematically explores how to incorporate PRMs into the Text-to-SQL reasoning process effectively. Our approach follows a "cold start, then PRM supervision" paradigm. Specifically, we first train the model to decompose SQL queries into structured stepwise reasoning chains using common table expressions (Chain-of-CTEs), establishing a strong and interpretable reasoning baseline. Then, we investigate four strategies for integrating PRMs, and find that combining PRM as an online training signal (GRPO) with PRM-guided inference (e.g., best-of-N sampling) yields the best results. Empirically, on the BIRD benchmark, Reward-SQL enables models supervised by a 7B PRM to achieve a 13.1% performance gain across various guidance strategies. Notably, our GRPO-aligned policy model based on Qwen2.5-Coder-7B-Instruct achieves 68.9% accuracy on the BIRD development set, outperforming all baseline methods under the same model size. These results demonstrate the effectiveness of Reward-SQL in leveraging reward-based supervision for Text-to-SQL reasoning. Our code is publicly available. 

---
# Fine-Tuning Large Language Models and Evaluating Retrieval Methods for Improved Question Answering on Building Codes 

**Authors**: Mohammad Aqib, Mohd Hamza, Qipei Mei, Ying Hei Chui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04666)  

**Abstract**: Building codes are regulations that establish standards for the design, construction, and safety of buildings to ensure structural integrity, fire protection, and accessibility. They are often extensive, complex, and subject to frequent updates, making manual querying challenging and time-consuming. Key difficulties include navigating large volumes of text, interpreting technical language, and identifying relevant clauses across different sections. A potential solution is to build a Question-Answering (QA) system that answers user queries based on building codes. Among the various methods for building a QA system, Retrieval-Augmented Generation (RAG) stands out in performance. RAG consists of two components: a retriever and a language model. This study focuses on identifying a suitable retriever method for building codes and optimizing the generational capability of the language model using fine-tuning techniques. We conducted a detailed evaluation of various retrieval methods by performing the retrieval on the National Building Code of Canada (NBCC) and explored the impact of domain-specific fine-tuning on several language models using the dataset derived from NBCC. Our analysis included a comparative assessment of different retrievers and the performance of both pre-trained and fine-tuned models to determine the efficacy and domain-specific adaptation of language models using fine-tuning on the NBCC dataset. Experimental results showed that Elasticsearch proved to be the most robust retriever among all. The findings also indicate that fine-tuning language models on an NBCC-specific dataset can enhance their ability to generate contextually relevant responses. When combined with context retrieved by a powerful retriever like Elasticsearch, this improvement in LLM performance can optimize the RAG system, enabling it to better navigate the complexities of the NBCC. 

---
# Personalized Risks and Regulatory Strategies of Large Language Models in Digital Advertising 

**Authors**: Haoyang Feng, Yanjun Dai, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04665)  

**Abstract**: Although large language models have demonstrated the potential for personalized advertising recommendations in experimental environments, in actual operations, how advertising recommendation systems can be combined with measures such as user privacy protection and data security is still an area worthy of in-depth discussion. To this end, this paper studies the personalized risks and regulatory strategies of large language models in digital advertising. This study first outlines the principles of Large Language Model (LLM), especially the self-attention mechanism based on the Transformer architecture, and how to enable the model to understand and generate natural language text. Then, the BERT (Bidirectional Encoder Representations from Transformers) model and the attention mechanism are combined to construct an algorithmic model for personalized advertising recommendations and user factor risk protection. The specific steps include: data collection and preprocessing, feature selection and construction, using large language models such as BERT for advertising semantic embedding, and ad recommendations based on user portraits. Then, local model training and data encryption are used to ensure the security of user privacy and avoid the leakage of personal data. This paper designs an experiment for personalized advertising recommendation based on a large language model of BERT and verifies it with real user data. The experimental results show that BERT-based advertising push can effectively improve the click-through rate and conversion rate of advertisements. At the same time, through local model training and privacy protection mechanisms, the risk of user privacy leakage can be reduced to a certain extent. 

---
# AI-Generated Fall Data: Assessing LLMs and Diffusion Model for Wearable Fall Detection 

**Authors**: Sana Alamgeer, Yasine Souissi, Anne H. H. Ngu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04660)  

**Abstract**: Training fall detection systems is challenging due to the scarcity of real-world fall data, particularly from elderly individuals. To address this, we explore the potential of Large Language Models (LLMs) for generating synthetic fall data. This study evaluates text-to-motion (T2M, SATO, ParCo) and text-to-text models (GPT4o, GPT4, Gemini) in simulating realistic fall scenarios. We generate synthetic datasets and integrate them with four real-world baseline datasets to assess their impact on fall detection performance using a Long Short-Term Memory (LSTM) model. Additionally, we compare LLM-generated synthetic data with a diffusion-based method to evaluate their alignment with real accelerometer distributions. Results indicate that dataset characteristics significantly influence the effectiveness of synthetic data, with LLM-generated data performing best in low-frequency settings (e.g., 20Hz) while showing instability in high-frequency datasets (e.g., 200Hz). While text-to-motion models produce more realistic biomechanical data than text-to-text models, their impact on fall detection varies. Diffusion-based synthetic data demonstrates the closest alignment to real data but does not consistently enhance model performance. An ablation study further confirms that the effectiveness of synthetic data depends on sensor placement and fall representation. These findings provide insights into optimizing synthetic data generation for fall detection models. 

---
# Integration of Large Language Models and Traditional Deep Learning for Social Determinants of Health Prediction 

**Authors**: Paul Landes, Jimeng Sun, Adam Cross  

**Link**: [PDF](https://arxiv.org/pdf/2505.04655)  

**Abstract**: Social Determinants of Health (SDoH) are economic, social and personal circumstances that affect or influence an individual's health status. SDoHs have shown to be correlated to wellness outcomes, and therefore, are useful to physicians in diagnosing diseases and in decision-making. In this work, we automatically extract SDoHs from clinical text using traditional deep learning and Large Language Models (LLMs) to find the advantages and disadvantages of each on an existing publicly available dataset. Our models outperform a previous reference point on a multilabel SDoH classification by 10 points, and we present a method and model to drastically speed up classification (12X execution time) by eliminating expensive LLM processing. The method we present combines a more nimble and efficient solution that leverages the power of the LLM for precision and traditional deep learning methods for efficiency. We also show highly performant results on a dataset supplemented with synthetic data and several traditional deep learning models that outperform LLMs. Our models and methods offer the next iteration of automatic prediction of SDoHs that impact at-risk patients. 

---
# A Comparative Analysis of Ethical and Safety Gaps in LLMs using Relative Danger Coefficient 

**Authors**: Yehor Tereshchenko, Mika Hämäläinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04654)  

**Abstract**: Artificial Intelligence (AI) and Large Language Models (LLMs) have rapidly evolved in recent years, showcasing remarkable capabilities in natural language understanding and generation. However, these advancements also raise critical ethical questions regarding safety, potential misuse, discrimination and overall societal impact. This article provides a comparative analysis of the ethical performance of various AI models, including the brand new DeepSeek-V3(R1 with reasoning and without), various GPT variants (4o, 3.5 Turbo, 4 Turbo, o1/o3 mini) and Gemini (1.5 flash, 2.0 flash and 2.0 flash exp) and highlights the need for robust human oversight, especially in situations with high stakes. Furthermore, we present a new metric for calculating harm in LLMs called Relative Danger Coefficient (RDC). 

---
# Advancing Conversational Diagnostic AI with Multimodal Reasoning 

**Authors**: Khaled Saab, Jan Freyberg, Chunjong Park, Tim Strother, Yong Cheng, Wei-Hung Weng, David G.T. Barrett, David Stutz, Nenad Tomasev, Anil Palepu, Valentin Liévin, Yash Sharma, Roma Ruparel, Abdullah Ahmed, Elahe Vedadi, Kimberly Kanada, Cian Hughes, Yun Liu, Geoff Brown, Yang Gao, Sean Li, S. Sara Mahdavi, James Manyika, Katherine Chou, Yossi Matias, Avinatan Hassidim, Dale R. Webster, Pushmeet Kohli, S.M. Ali Eslami, Joëlle Barral, Adam Rodman, Vivek Natarajan, Mike Schaekermann, Tao Tu, Alan Karthikesalingam, Ryutaro Tanno  

**Link**: [PDF](https://arxiv.org/pdf/2505.04653)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential for conducting diagnostic conversations but evaluation has been largely limited to language-only interactions, deviating from the real-world requirements of remote care delivery. Instant messaging platforms permit clinicians and patients to upload and discuss multimodal medical artifacts seamlessly in medical consultation, but the ability of LLMs to reason over such data while preserving other attributes of competent diagnostic conversation remains unknown. Here we advance the conversational diagnosis and management performance of the Articulate Medical Intelligence Explorer (AMIE) through a new capability to gather and interpret multimodal data, and reason about this precisely during consultations. Leveraging Gemini 2.0 Flash, our system implements a state-aware dialogue framework, where conversation flow is dynamically controlled by intermediate model outputs reflecting patient states and evolving diagnoses. Follow-up questions are strategically directed by uncertainty in such patient states, leading to a more structured multimodal history-taking process that emulates experienced clinicians. We compared AMIE to primary care physicians (PCPs) in a randomized, blinded, OSCE-style study of chat-based consultations with patient actors. We constructed 105 evaluation scenarios using artifacts like smartphone skin photos, ECGs, and PDFs of clinical documents across diverse conditions and demographics. Our rubric assessed multimodal capabilities and other clinically meaningful axes like history-taking, diagnostic accuracy, management reasoning, communication, and empathy. Specialist evaluation showed AMIE to be superior to PCPs on 7/9 multimodal and 29/32 non-multimodal axes (including diagnostic accuracy). The results show clear progress in multimodal conversational diagnostic AI, but real-world translation needs further research. 

---
# Scientific Hypothesis Generation and Validation: Methods, Datasets, and Future Directions 

**Authors**: Adithya Kulkarni, Fatimah Alotaibi, Xinyue Zeng, Longfeng Wu, Tong Zeng, Barry Menglong Yao, Minqian Liu, Shuaicheng Zhang, Lifu Huang, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04651)  

**Abstract**: Large Language Models (LLMs) are transforming scientific hypothesis generation and validation by enabling information synthesis, latent relationship discovery, and reasoning augmentation. This survey provides a structured overview of LLM-driven approaches, including symbolic frameworks, generative models, hybrid systems, and multi-agent architectures. We examine techniques such as retrieval-augmented generation, knowledge-graph completion, simulation, causal inference, and tool-assisted reasoning, highlighting trade-offs in interpretability, novelty, and domain alignment. We contrast early symbolic discovery systems (e.g., BACON, KEKADA) with modern LLM pipelines that leverage in-context learning and domain adaptation via fine-tuning, retrieval, and symbolic grounding. For validation, we review simulation, human-AI collaboration, causal modeling, and uncertainty quantification, emphasizing iterative assessment in open-world contexts. The survey maps datasets across biomedicine, materials science, environmental science, and social science, introducing new resources like AHTech and CSKG-600. Finally, we outline a roadmap emphasizing novelty-aware generation, multimodal-symbolic integration, human-in-the-loop systems, and ethical safeguards, positioning LLMs as agents for principled, scalable scientific discovery. 

---
# FRAME: Feedback-Refined Agent Methodology for Enhancing Medical Research Insights 

**Authors**: Chengzhang Yu, Yiming Zhang, Zhixin Liu, Zenghui Ding, Yining Sun, Zhanpeng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04649)  

**Abstract**: The automation of scientific research through large language models (LLMs) presents significant opportunities but faces critical challenges in knowledge synthesis and quality assurance. We introduce Feedback-Refined Agent Methodology (FRAME), a novel framework that enhances medical paper generation through iterative refinement and structured feedback. Our approach comprises three key innovations: (1) A structured dataset construction method that decomposes 4,287 medical papers into essential research components through iterative refinement; (2) A tripartite architecture integrating Generator, Evaluator, and Reflector agents that progressively improve content quality through metric-driven feedback; and (3) A comprehensive evaluation framework that combines statistical metrics with human-grounded benchmarks. Experimental results demonstrate FRAME's effectiveness, achieving significant improvements over conventional approaches across multiple models (9.91% average gain with DeepSeek V3, comparable improvements with GPT-4o Mini) and evaluation dimensions. Human evaluation confirms that FRAME-generated papers achieve quality comparable to human-authored works, with particular strength in synthesizing future research directions. The results demonstrated our work could efficiently assist medical research by building a robust foundation for automated medical research paper generation while maintaining rigorous academic standards. 

---
# ChatGPT for automated grading of short answer questions in mechanical ventilation 

**Authors**: Tejas Jade, Alex Yartsev  

**Link**: [PDF](https://arxiv.org/pdf/2505.04645)  

**Abstract**: Standardised tests using short answer questions (SAQs) are common in postgraduate education. Large language models (LLMs) simulate conversational language and interpret unstructured free-text responses in ways aligning with applying SAQ grading rubrics, making them attractive for automated grading. We evaluated ChatGPT 4o to grade SAQs in a postgraduate medical setting using data from 215 students (557 short-answer responses) enrolled in an online course on mechanical ventilation (2020--2024). Deidentified responses to three case-based scenarios were presented to ChatGPT with a standardised grading prompt and rubric. Outputs were analysed using mixed-effects modelling, variance component analysis, intraclass correlation coefficients (ICCs), Cohen's kappa, Kendall's W, and Bland--Altman statistics. ChatGPT awarded systematically lower marks than human graders with a mean difference (bias) of -1.34 on a 10-point scale. ICC values indicated poor individual-level agreement (ICC1 = 0.086), and Cohen's kappa (-0.0786) suggested no meaningful agreement. Variance component analysis showed minimal variability among the five ChatGPT sessions (G-value = 0.87), indicating internal consistency but divergence from the human grader. The poorest agreement was observed for evaluative and analytic items, whereas checklist and prescriptive rubric items had less disagreement. We caution against the use of LLMs in grading postgraduate coursework. Over 60% of ChatGPT-assigned grades differed from human grades by more than acceptable boundaries for high-stakes assessments. 

---
# Prediction-powered estimators for finite population statistics in highly imbalanced textual data: Public hate crime estimation 

**Authors**: Hannes Waldetoft, Jakob Torgander, Måns Magnusson  

**Link**: [PDF](https://arxiv.org/pdf/2505.04643)  

**Abstract**: Estimating population parameters in finite populations of text documents can be challenging when obtaining the labels for the target variable requires manual annotation. To address this problem, we combine predictions from a transformer encoder neural network with well-established survey sampling estimators using the model predictions as an auxiliary variable. The applicability is demonstrated in Swedish hate crime statistics based on Swedish police reports. Estimates of the yearly number of hate crimes and the police's under-reporting are derived using the Hansen-Hurwitz estimator, difference estimation, and stratified random sampling estimation. We conclude that if labeled training data is available, the proposed method can provide very efficient estimates with reduced time spent on manual annotation. 

---
# Rethinking Multimodal Sentiment Analysis: A High-Accuracy, Simplified Fusion Architecture 

**Authors**: Nischal Mandal, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04642)  

**Abstract**: Multimodal sentiment analysis, a pivotal task in affective computing, seeks to understand human emotions by integrating cues from language, audio, and visual signals. While many recent approaches leverage complex attention mechanisms and hierarchical architectures, we propose a lightweight, yet effective fusion-based deep learning model tailored for utterance-level emotion classification. Using the benchmark IEMOCAP dataset, which includes aligned text, audio-derived numeric features, and visual descriptors, we design a modality-specific encoder using fully connected layers followed by dropout regularization. The modality-specific representations are then fused using simple concatenation and passed through a dense fusion layer to capture cross-modal interactions. This streamlined architecture avoids computational overhead while preserving performance, achieving a classification accuracy of 92% across six emotion categories. Our approach demonstrates that with careful feature engineering and modular design, simpler fusion strategies can outperform or match more complex models, particularly in resource-constrained environments. 

---
# A Comparative Benchmark of a Moroccan Darija Toxicity Detection Model (Typica.ai) and Major LLM-Based Moderation APIs (OpenAI, Mistral, Anthropic) 

**Authors**: Hicham Assoudi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04640)  

**Abstract**: This paper presents a comparative benchmark evaluating the performance of this http URL's custom Moroccan Darija toxicity detection model against major LLM-based moderation APIs: OpenAI (omni-moderation-latest), Mistral (mistral-moderation-latest), and Anthropic Claude (claude-3-haiku-20240307). We focus on culturally grounded toxic content, including implicit insults, sarcasm, and culturally specific aggression often overlooked by general-purpose systems. Using a balanced test set derived from the OMCD_Typica.ai_Mix dataset, we report precision, recall, F1-score, and accuracy, offering insights into challenges and opportunities for moderation in underrepresented languages. Our results highlight this http URL's superior performance, underlining the importance of culturally adapted models for reliable content moderation. 

---
# Language translation, and change of accent for speech-to-speech task using diffusion model 

**Authors**: Abhishek Mishra, Ritesh Sur Chowdhury, Vartul Bahuguna, Isha Pandey, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04639)  

**Abstract**: Speech-to-speech translation (S2ST) aims to convert spoken input in one language to spoken output in another, typically focusing on either language translation or accent adaptation. However, effective cross-cultural communication requires handling both aspects simultaneously - translating content while adapting the speaker's accent to match the target language context. In this work, we propose a unified approach for simultaneous speech translation and change of accent, a task that remains underexplored in current literature. Our method reformulates the problem as a conditional generation task, where target speech is generated based on phonemes and guided by target speech features. Leveraging the power of diffusion models, known for high-fidelity generative capabilities, we adapt text-to-image diffusion strategies by conditioning on source speech transcriptions and generating Mel spectrograms representing the target speech with desired linguistic and accentual attributes. This integrated framework enables joint optimization of translation and accent adaptation, offering a more parameter-efficient and effective model compared to traditional pipelines. 

---
# Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs 

**Authors**: Dongxing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04637)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in processing diverse data types, yet significant disparities persist between human cognitive processes and computational approaches to multimodal information integration. This research presents a systematic investigation into the parallels between human cross-modal chunking mechanisms and token representation methodologies in MLLMs. Through empirical studies comparing human performance patterns with model behaviors across visual-linguistic tasks, we demonstrate that conventional static tokenization schemes fundamentally constrain current models' capacity to simulate the dynamic, context-sensitive nature of human information processing. We propose a novel framework for dynamic cross-modal tokenization that incorporates adaptive boundaries, hierarchical representations, and alignment mechanisms grounded in cognitive science principles. Quantitative evaluations demonstrate that our approach yields statistically significant improvements over state-of-the-art models on benchmark tasks (+7.8% on Visual Question Answering, +5.3% on Complex Scene Description) while exhibiting more human-aligned error patterns and attention distributions. These findings contribute to the theoretical understanding of the relationship between human cognition and artificial intelligence, while providing empirical evidence for developing more cognitively plausible AI systems. 

---
# How Social is It? A Benchmark for LLMs' Capabilities in Multi-user Multi-turn Social Agent Tasks 

**Authors**: Yusen Wu, Junwu Xiong, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04628)  

**Abstract**: Expanding the application of large language models (LLMs) to societal life, instead of primary function only as auxiliary assistants to communicate with only one person at a time, necessitates LLMs' capabilities to independently play roles in multi-user, multi-turn social agent tasks within complex social settings. However, currently the capability has not been systematically measured with available benchmarks. To address this gap, we first introduce an agent task leveling framework grounded in sociological principles. Concurrently, we propose a novel benchmark, How Social Is It (we call it HSII below), designed to assess LLM's social capabilities in comprehensive social agents tasks and benchmark representative models. HSII comprises four stages: format parsing, target selection, target switching conversation, and stable conversation, which collectively evaluate the communication and task completion capabilities of LLMs within realistic social interaction scenarios dataset, HSII-Dataset. The dataset is derived step by step from news dataset. We perform an ablation study by doing clustering to the dataset. Additionally, we investigate the impact of chain of thought (COT) method on enhancing LLMs' social performance. Since COT cost more computation, we further introduce a new statistical metric, COT-complexity, to quantify the efficiency of certain LLMs with COTs for specific social tasks and strike a better trade-off between measurement of correctness and efficiency. Various results of our experiments demonstrate that our benchmark is well-suited for evaluating social skills in LLMs. 

---
# StreamBridge: Turning Your Offline Video Large Language Model into a Proactive Streaming Assistant 

**Authors**: Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, Ping Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05467)  

**Abstract**: We present StreamBridge, a simple yet effective framework that seamlessly transforms offline Video-LLMs into streaming-capable models. It addresses two fundamental challenges in adapting existing models into online scenarios: (1) limited capability for multi-turn real-time understanding, and (2) lack of proactive response mechanisms. Specifically, StreamBridge incorporates (1) a memory buffer combined with a round-decayed compression strategy, supporting long-context multi-turn interactions, and (2) a decoupled, lightweight activation model that can be effortlessly integrated into existing Video-LLMs, enabling continuous proactive responses. To further support StreamBridge, we construct Stream-IT, a large-scale dataset tailored for streaming video understanding, featuring interleaved video-text sequences and diverse instruction formats. Extensive experiments show that StreamBridge significantly improves the streaming understanding capabilities of offline Video-LLMs across various tasks, outperforming even proprietary models such as GPT-4o and Gemini 1.5 Pro. Simultaneously, it achieves competitive or superior performance on standard video understanding benchmarks. 

---
# Adaptive Markup Language Generation for Contextually-Grounded Visual Document Understanding 

**Authors**: Han Xiao, Yina Xie, Guanxin Tan, Yinghao Chen, Rui Hu, Ke Wang, Aojun Zhou, Hao Li, Hao Shao, Xudong Lu, Peng Gao, Yafei Wen, Xiaoxin Chen, Shuai Ren, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05446)  

**Abstract**: Visual Document Understanding has become essential with the increase of text-rich visual content. This field poses significant challenges due to the need for effective integration of visual perception and textual comprehension, particularly across diverse document types with complex layouts. Moreover, existing fine-tuning datasets for this domain often fall short in providing the detailed contextual information for robust understanding, leading to hallucinations and limited comprehension of spatial relationships among visual elements. To address these challenges, we propose an innovative pipeline that utilizes adaptive generation of markup languages, such as Markdown, JSON, HTML, and TiKZ, to build highly structured document representations and deliver contextually-grounded responses. We introduce two fine-grained structured datasets: DocMark-Pile, comprising approximately 3.8M pretraining data pairs for document parsing, and DocMark-Instruct, featuring 624k fine-tuning data annotations for grounded instruction following. Extensive experiments demonstrate that our proposed model significantly outperforms existing state-of-theart MLLMs across a range of visual document understanding benchmarks, facilitating advanced reasoning and comprehension capabilities in complex visual scenarios. Our code and models are released at https://github. com/Euphoria16/DocMark. 

---
# TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation 

**Authors**: Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05422)  

**Abstract**: Pioneering token-based works such as Chameleon and Emu3 have established a foundation for multimodal unification but face challenges of high training computational overhead and limited comprehension performance due to a lack of high-level semantics. In this paper, we introduce TokLIP, a visual tokenizer that enhances comprehension by semanticizing vector-quantized (VQ) tokens and incorporating CLIP-level semantics while enabling end-to-end multimodal autoregressive training with standard VQ tokens. TokLIP integrates a low-level discrete VQ tokenizer with a ViT-based token encoder to capture high-level continuous semantics. Unlike previous approaches (e.g., VILA-U) that discretize high-level features, TokLIP disentangles training objectives for comprehension and generation, allowing the direct application of advanced VQ tokenizers without the need for tailored quantization operations. Our empirical results demonstrate that TokLIP achieves exceptional data efficiency, empowering visual tokens with high-level semantic understanding while enhancing low-level generative capacity, making it well-suited for autoregressive Transformers in both comprehension and generation tasks. The code and models are available at this https URL. 

---
# Scalable Chain of Thoughts via Elastic Reasoning 

**Authors**: Yuhui Xu, Hanze Dong, Lei Wang, Doyen Sahoo, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.05315)  

**Abstract**: Large reasoning models (LRMs) have achieved remarkable progress on complex tasks by generating extended chains of thought (CoT). However, their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained. We propose Elastic Reasoning, a novel framework for scalable chain of thoughts that explicitly separates reasoning into two phases--thinking and solution--with independently allocated budgets. At test time, Elastic Reasoning prioritize that completeness of solution segments, significantly improving reliability under tight resource constraints. To train models that are robust to truncated thinking, we introduce a lightweight budget-constrained rollout strategy, integrated into GRPO, which teaches the model to reason adaptively when the thinking process is cut short and generalizes effectively to unseen budget constraints without additional training. Empirical results on mathematical (AIME, MATH500) and programming (LiveCodeBench, Codeforces) benchmarks demonstrate that Elastic Reasoning performs robustly under strict budget constraints, while incurring significantly lower training cost than baseline methods. Remarkably, our approach also produces more concise and efficient reasoning even in unconstrained settings. Elastic Reasoning offers a principled and practical solution to the pressing challenge of controllable reasoning at scale. 

---
# Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks 

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal  

**Link**: [PDF](https://arxiv.org/pdf/2505.05190)  

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking. 

---
# Understanding In-context Learning of Addition via Activation Subspaces 

**Authors**: Xinyan Hu, Kayo Yin, Michael I. Jordan, Jacob Steinhardt, Lijie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05145)  

**Abstract**: To perform in-context learning, language models must extract signals from individual few-shot examples, aggregate these into a learned prediction rule, and then apply this rule to new examples. How is this implemented in the forward pass of modern transformer models? To study this, we consider a structured family of few-shot learning tasks for which the true prediction rule is to add an integer $k$ to the input. We find that Llama-3-8B attains high accuracy on this task for a range of $k$, and localize its few-shot ability to just three attention heads via a novel optimization approach. We further show the extracted signals lie in a six-dimensional subspace, where four of the dimensions track the unit digit and the other two dimensions track overall magnitude. We finally examine how these heads extract information from individual few-shot examples, identifying a self-correction mechanism in which mistakes from earlier examples are suppressed by later examples. Our results demonstrate how tracking low-dimensional subspaces across a forward pass can provide insight into fine-grained computational structures. 

---
# X-Driver: Explainable Autonomous Driving with Vision-Language Models 

**Authors**: Wei Liu, Jiyuan Zhang, Binxiong Zheng, Yufeng Hu, Yingzhan Lin, Zengfeng Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05098)  

**Abstract**: End-to-end autonomous driving has advanced significantly, offering benefits such as system simplicity and stronger driving performance in both open-loop and closed-loop settings than conventional pipelines. However, existing frameworks still suffer from low success rates in closed-loop evaluations, highlighting their limitations in real-world deployment. In this paper, we introduce X-Driver, a unified multi-modal large language models(MLLMs) framework designed for closed-loop autonomous driving, leveraging Chain-of-Thought(CoT) and autoregressive modeling to enhance perception and decision-making. We validate X-Driver across multiple autonomous driving tasks using public benchmarks in CARLA simulation environment, including Bench2Drive[6]. Our experimental results demonstrate superior closed-loop performance, surpassing the current state-of-the-art(SOTA) while improving the interpretability of driving decisions. These findings underscore the importance of structured reasoning in end-to-end driving and establish X-Driver as a strong baseline for future research in closed-loop autonomous driving. 

---
# CodeMixBench: Evaluating Large Language Models on Code Generation with Code-Mixed Prompts 

**Authors**: Manik Sheokand, Parth Sawant  

**Link**: [PDF](https://arxiv.org/pdf/2505.05063)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in code generation tasks, powering various applications like code completion, debugging, and programming assistance. However, existing benchmarks such as HumanEval, MBPP, and BigCodeBench primarily evaluate LLMs on English-only prompts, overlooking the real-world scenario where multilingual developers often use code-mixed language while interacting with LLMs. To address this gap, we introduce CodeMixBench, a novel benchmark designed to evaluate the robustness of LLMs on code generation from code-mixed prompts. Built upon BigCodeBench, CodeMixBench introduces controlled code-mixing (CMD) into the natural language parts of prompts across three language pairs: Hinglish (Hindi-English), Spanish-English, and Chinese Pinyin-English. We comprehensively evaluate a diverse set of open-source code generation models ranging from 1.5B to 15B parameters. Our results show that code-mixed prompts consistently degrade Pass@1 performance compared to their English-only counterparts, with performance drops increasing under higher CMD levels for smaller models. CodeMixBench provides a realistic evaluation framework for studying multilingual code generation and highlights new challenges and directions for building robust code generation models that generalize well across diverse linguistic settings. 

---
# General Transform: A Unified Framework for Adaptive Transform to Enhance Representations 

**Authors**: Gekko Budiutama, Shunsuke Daimon, Hirofumi Nishi, Yu-ichiro Matsushita  

**Link**: [PDF](https://arxiv.org/pdf/2505.04969)  

**Abstract**: Discrete transforms, such as the discrete Fourier transform, are widely used in machine learning to improve model performance by extracting meaningful features. However, with numerous transforms available, selecting an appropriate one often depends on understanding the dataset's properties, making the approach less effective when such knowledge is unavailable. In this work, we propose General Transform (GT), an adaptive transform-based representation designed for machine learning applications. Unlike conventional transforms, GT learns data-driven mapping tailored to the dataset and task of interest. Here, we demonstrate that models incorporating GT outperform conventional transform-based approaches across computer vision and natural language processing tasks, highlighting its effectiveness in diverse learning scenarios. 

---
# Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations 

**Authors**: Md Aminul Islam, Ahmed Sayeed Faruk  

**Link**: [PDF](https://arxiv.org/pdf/2505.04948)  

**Abstract**: Recommender systems are essential for delivering personalized content across digital platforms by modeling user preferences and behaviors. Recently, large language models (LLMs) have been adopted for prompt-based recommendation due to their ability to generate personalized outputs without task-specific training. However, LLM-based methods face limitations such as limited context window size, inefficient pointwise and pairwise prompting, and difficulty handling listwise ranking due to token constraints. LLMs can also be sensitive to position bias, as they may overemphasize earlier items in the prompt regardless of their true relevance. To address and investigate these issues, we propose a hybrid framework that combines a traditional recommendation model with an LLM for reranking top-k items using structured prompts. We evaluate the effects of user history reordering and instructional prompts for mitigating position bias. Experiments on MovieLens-100K show that randomizing user history improves ranking quality, but LLM-based reranking does not outperform the base model. Explicit instructions to reduce position bias are also ineffective. Our evaluations reveal limitations in LLMs' ability to model ranking context and mitigate bias. Our code is publicly available at this https URL. 

---
# T2VTextBench: A Human Evaluation Benchmark for Textual Control in Video Generation Models 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04946)  

**Abstract**: Thanks to recent advancements in scalable deep architectures and large-scale pretraining, text-to-video generation has achieved unprecedented capabilities in producing high-fidelity, instruction-following content across a wide range of styles, enabling applications in advertising, entertainment, and education. However, these models' ability to render precise on-screen text, such as captions or mathematical formulas, remains largely untested, posing significant challenges for applications requiring exact textual accuracy. In this work, we introduce T2VTextBench, the first human-evaluation benchmark dedicated to evaluating on-screen text fidelity and temporal consistency in text-to-video models. Our suite of prompts integrates complex text strings with dynamic scene changes, testing each model's ability to maintain detailed instructions across frames. We evaluate ten state-of-the-art systems, ranging from open-source solutions to commercial offerings, and find that most struggle to generate legible, consistent text. These results highlight a critical gap in current video generators and provide a clear direction for future research aimed at enhancing textual manipulation in video synthesis. 

---
# Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models 

**Authors**: Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, Shouzheng Huang, Xinping Zhao, Borui Jiang, Lanqing Hong, Longyue Wang, Zhuotao Tian, Baoxing Huai, Wenhan Luo, Weihua Luo, Zheng Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04921)  

**Abstract**: Reasoning lies at the heart of intelligence, shaping the ability to make decisions, draw conclusions, and generalize across domains. In artificial intelligence, as systems increasingly operate in open, uncertain, and multimodal environments, reasoning becomes essential for enabling robust and adaptive behavior. Large Multimodal Reasoning Models (LMRMs) have emerged as a promising paradigm, integrating modalities such as text, images, audio, and video to support complex reasoning capabilities and aiming to achieve comprehensive perception, precise understanding, and deep reasoning. As research advances, multimodal reasoning has rapidly evolved from modular, perception-driven pipelines to unified, language-centric frameworks that offer more coherent cross-modal understanding. While instruction tuning and reinforcement learning have improved model reasoning, significant challenges remain in omni-modal generalization, reasoning depth, and agentic behavior. To address these issues, we present a comprehensive and structured survey of multimodal reasoning research, organized around a four-stage developmental roadmap that reflects the field's shifting design philosophies and emerging capabilities. First, we review early efforts based on task-specific modules, where reasoning was implicitly embedded across stages of representation, alignment, and fusion. Next, we examine recent approaches that unify reasoning into multimodal LLMs, with advances such as Multimodal Chain-of-Thought (MCoT) and multimodal reinforcement learning enabling richer and more structured reasoning chains. Finally, drawing on empirical insights from challenging benchmarks and experimental cases of OpenAI O3 and O4-mini, we discuss the conceptual direction of native large multimodal reasoning models (N-LMRMs), which aim to support scalable, agentic, and adaptive reasoning and planning in complex, real-world environments. 

---
# Enigme: Generative Text Puzzles for Evaluating Reasoning in Language Models 

**Authors**: John Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2505.04914)  

**Abstract**: Transformer-decoder language models are a core innovation in text based generative artificial intelligence. These models are being deployed as general-purpose intelligence systems in many applications. Central to their utility is the capacity to understand natural language commands and exploit the reasoning embedded in human text corpora to apply some form of reasoning process to a wide variety of novel tasks. To understand the limitations of this approach to generating reasoning we argue that we need to consider the architectural constraints of these systems. Consideration of the latent variable structure of transformer-decoder models allows us to design reasoning tasks that should probe the boundary of their capacity to reason. We present enigme, an open-source library for generating text-based puzzles to be used in training and evaluating reasoning skills within transformer-decoder models and future AI architectures. 

---
# SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models 

**Authors**: Shun Taguchi, Hideki Deguchi, Takumi Hamazaki, Hiroyuki Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2505.04911)  

**Abstract**: This study introduces SpatialPrompting, a novel framework that harnesses the emergent reasoning capabilities of off-the-shelf multimodal large language models to achieve zero-shot spatial reasoning in three-dimensional (3D) environments. Unlike existing methods that rely on expensive 3D-specific fine-tuning with specialized 3D inputs such as point clouds or voxel-based features, SpatialPrompting employs a keyframe-driven prompt generation strategy. This framework uses metrics such as vision-language similarity, Mahalanobis distance, field of view, and image sharpness to select a diverse and informative set of keyframes from image sequences and then integrates them with corresponding camera pose data to effectively abstract spatial relationships and infer complex 3D structures. The proposed framework not only establishes a new paradigm for flexible spatial reasoning that utilizes intuitive visual and positional cues but also achieves state-of-the-art zero-shot performance on benchmark datasets, such as ScanQA and SQA3D, across several metrics. The proposed method effectively eliminates the need for specialized 3D inputs and fine-tuning, offering a simpler and more scalable alternative to conventional approaches. 

---
# ConCISE: Confidence-guided Compression in Step-by-step Efficient Reasoning 

**Authors**: Ziqing Qiao, Yongheng Deng, Jiali Zeng, Dong Wang, Lai Wei, Fandong Meng, Jie Zhou, Ju Ren, Yaoxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04881)  

**Abstract**: Large Reasoning Models (LRMs) perform strongly in complex reasoning tasks via Chain-of-Thought (CoT) prompting, but often suffer from verbose outputs caused by redundant content, increasing computational overhead, and degrading user experience. Existing compression methods either operate post-hoc pruning, risking disruption to reasoning coherence, or rely on sampling-based selection, which fails to intervene effectively during generation. In this work, we introduce a confidence-guided perspective to explain the emergence of redundant reflection in LRMs, identifying two key patterns: Confidence Deficit, where the model reconsiders correct steps due to low internal confidence, and Termination Delay, where reasoning continues even after reaching a confident answer. Based on this analysis, we propose ConCISE (Confidence-guided Compression In Step-by-step Efficient Reasoning), a framework that simplifies reasoning chains by reinforcing the model's confidence during inference, thus preventing the generation of redundant reflection steps. It integrates Confidence Injection to stabilize intermediate steps and Early Stopping to terminate reasoning when confidence is sufficient. Extensive experiments demonstrate that fine-tuning LRMs on ConCISE-generated data yields significantly shorter outputs, reducing length by up to approximately 50% under SimPO, while maintaining high task accuracy. ConCISE consistently outperforms existing baselines across multiple reasoning benchmarks. 

---
# CRAFT: Cultural Russian-Oriented Dataset Adaptation for Focused Text-to-Image Generation 

**Authors**: Viacheslav Vasilev, Vladimir Arkhipkin, Julia Agafonova, Tatiana Nikulina, Evelina Mironova, Alisa Shichanina, Nikolai Gerasimenko, Mikhail Shoytov, Denis Dimitrov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04851)  

**Abstract**: Despite the fact that popular text-to-image generation models cope well with international and general cultural queries, they have a significant knowledge gap regarding individual cultures. This is due to the content of existing large training datasets collected on the Internet, which are predominantly based on Western European or American popular culture. Meanwhile, the lack of cultural adaptation of the model can lead to incorrect results, a decrease in the generation quality, and the spread of stereotypes and offensive content. In an effort to address this issue, we examine the concept of cultural code and recognize the critical importance of its understanding by modern image generation models, an issue that has not been sufficiently addressed in the research community to date. We propose the methodology for collecting and processing the data necessary to form a dataset based on the cultural code, in particular the Russian one. We explore how the collected data affects the quality of generations in the national domain and analyze the effectiveness of our approach using the Kandinsky 3.1 text-to-image model. Human evaluation results demonstrate an increase in the level of awareness of Russian culture in the model. 

---
# HiPerRAG: High-Performance Retrieval Augmented Generation for Scientific Insights 

**Authors**: Ozan Gokdemir, Carlo Siebenschuh, Alexander Brace, Azton Wells, Brian Hsu, Kyle Hippe, Priyanka V. Setty, Aswathy Ajith, J. Gregory Pauloski, Varuni Sastry, Sam Foreman, Huihuo Zheng, Heng Ma, Bharat Kale, Nicholas Chia, Thomas Gibbs, Michael E. Papka, Thomas Brettin, Francis J. Alexander, Anima Anandkumar, Ian Foster, Rick Stevens, Venkatram Vishwanath, Arvind Ramanathan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04846)  

**Abstract**: The volume of scientific literature is growing exponentially, leading to underutilized discoveries, duplicated efforts, and limited cross-disciplinary collaboration. Retrieval Augmented Generation (RAG) offers a way to assist scientists by improving the factuality of Large Language Models (LLMs) in processing this influx of information. However, scaling RAG to handle millions of articles introduces significant challenges, including the high computational costs associated with parsing documents and embedding scientific knowledge, as well as the algorithmic complexity of aligning these representations with the nuanced semantics of scientific content. To address these issues, we introduce HiPerRAG, a RAG workflow powered by high performance computing (HPC) to index and retrieve knowledge from more than 3.6 million scientific articles. At its core are Oreo, a high-throughput model for multimodal document parsing, and ColTrast, a query-aware encoder fine-tuning algorithm that enhances retrieval accuracy by using contrastive learning and late-interaction techniques. HiPerRAG delivers robust performance on existing scientific question answering benchmarks and two new benchmarks introduced in this work, achieving 90% accuracy on SciQ and 76% on PubMedQA-outperforming both domain-specific models like PubMedGPT and commercial LLMs such as GPT-4. Scaling to thousands of GPUs on the Polaris, Sunspot, and Frontier supercomputers, HiPerRAG delivers million document-scale RAG workflows for unifying scientific knowledge and fostering interdisciplinary innovation. 

---
# Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs 

**Authors**: Chetan Pathade  

**Link**: [PDF](https://arxiv.org/pdf/2505.04806)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security. 

---
# When Bad Data Leads to Good Models 

**Authors**: Kenneth Li, Yida Chen, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.04741)  

**Abstract**: In large language model (LLM) pretraining, data quality is believed to determine model quality. In this paper, we re-examine the notion of "quality" from the perspective of pre- and post-training co-design. Specifically, we explore the possibility that pre-training on more toxic data can lead to better control in post-training, ultimately decreasing a model's output toxicity. First, we use a toy experiment to study how data composition affects the geometry of features in the representation space. Next, through controlled experiments with Olmo-1B models trained on varying ratios of clean and toxic data, we find that the concept of toxicity enjoys a less entangled linear representation as the proportion of toxic data increases. Furthermore, we show that although toxic data increases the generational toxicity of the base model, it also makes the toxicity easier to remove. Evaluations on Toxigen and Real Toxicity Prompts demonstrate that models trained on toxic data achieve a better trade-off between reducing generational toxicity and preserving general capabilities when detoxifying techniques such as inference-time intervention (ITI) are applied. Our findings suggest that, with post-training taken into account, bad data may lead to good models. 

---
# Towards Artificial Intelligence Research Assistant for Expert-Involved Learning 

**Authors**: Tianyu Liu, Simeng Han, Xiao Luo, Hanchen Wang, Pan Lu, Biqing Zhu, Yuge Wang, Keyi Li, Jiapeng Chen, Rihao Qu, Yufeng Liu, Xinyue Cui, Aviv Yaish, Yuhang Chen, Minsheng Hao, Chuhan Li, Kexing Li, Arman Cohan, Hua Xu, Mark Gerstein, James Zou, Hongyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04638)  

**Abstract**: Large Language Models (LLMs) and Large Multi-Modal Models (LMMs) have emerged as transformative tools in scientific research, yet their reliability and specific contributions to biomedical applications remain insufficiently characterized. In this study, we present \textbf{AR}tificial \textbf{I}ntelligence research assistant for \textbf{E}xpert-involved \textbf{L}earning (ARIEL), a multimodal dataset designed to benchmark and enhance two critical capabilities of LLMs and LMMs in biomedical research: summarizing extensive scientific texts and interpreting complex biomedical figures. To facilitate rigorous assessment, we create two open-source sets comprising biomedical articles and figures with designed questions. We systematically benchmark both open- and closed-source foundation models, incorporating expert-driven human evaluations conducted by doctoral-level experts. Furthermore, we improve model performance through targeted prompt engineering and fine-tuning strategies for summarizing research papers, and apply test-time computational scaling to enhance the reasoning capabilities of LMMs, achieving superior accuracy compared to human-expert corrections. We also explore the potential of using LMM Agents to generate scientific hypotheses from diverse multimodal inputs. Overall, our results delineate clear strengths and highlight significant limitations of current foundation models, providing actionable insights and guiding future advancements in deploying large-scale language and multi-modal models within biomedical research. 

---
# From Dialect Gaps to Identity Maps: Tackling Variability in Speaker Verification 

**Authors**: Abdulhady Abas Abdullah, Soran Badawi, Dana A. Abdullah, Dana Rasul Hamad, Hanan Abdulrahman Taher, Sabat Salih Muhamad, Aram Mahmood Ahmed, Bryar A. Hassan, Sirwan Abdolwahed Aula, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2505.04629)  

**Abstract**: The complexity and difficulties of Kurdish speaker detection among its several dialects are investigated in this work. Because of its great phonetic and lexical differences, Kurdish with several dialects including Kurmanji, Sorani, and Hawrami offers special challenges for speaker recognition systems. The main difficulties in building a strong speaker identification system capable of precisely identifying speakers across several dialects are investigated in this work. To raise the accuracy and dependability of these systems, it also suggests solutions like sophisticated machine learning approaches, data augmentation tactics, and the building of thorough dialect-specific corpus. The results show that customized strategies for every dialect together with cross-dialect training greatly enhance recognition performance. 

---
