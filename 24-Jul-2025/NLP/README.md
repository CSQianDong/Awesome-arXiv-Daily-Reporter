# Pretraining on the Test Set Is No Longer All You Need: A Debate-Driven Approach to QA Benchmarks 

**Authors**: Linbo Cao, Jinman Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17747)  

**Abstract**: As frontier language models increasingly saturate standard QA benchmarks, concerns about data contamination, memorization, and escalating dataset creation costs persist. We propose a debate-driven evaluation paradigm that transforms any existing QA dataset into structured adversarial debates--where one model is given the official answer to defend, and another constructs and defends an alternative answer--adjudicated by a judge model blind to the correct solution. By forcing multi-round argumentation, this approach substantially increases difficulty while penalizing shallow memorization, yet reuses QA items to reduce curation overhead. We make two main contributions: (1) an evaluation pipeline to systematically convert QA tasks into debate-based assessments, and (2) a public benchmark that demonstrates our paradigm's effectiveness on a subset of MMLU-Pro questions, complete with standardized protocols and reference models. Empirical results validate the robustness of the method and its effectiveness against data contamination--a Llama 3.1 model fine-tuned on test questions showed dramatic accuracy improvements (50% -> 82%) but performed worse in debates. Results also show that even weaker judges can reliably differentiate stronger debaters, highlighting how debate-based evaluation can scale to future, more capable systems while maintaining a fraction of the cost of creating new benchmarks. Overall, our framework underscores that "pretraining on the test set is no longer all you need," offering a sustainable path for measuring the genuine reasoning ability of advanced language models. 

---
# Megrez2 Technical Report 

**Authors**: Boxun Li, Yadong Li, Zhiyuan Li, Congyi Liu, Weilin Liu, Guowei Niu, Zheyue Tan, Haiyang Xu, Zhuyu Yao, Tao Yuan, Dong Zhou, Yueqing Zhuang, Bo Zhao, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17728)  

**Abstract**: We present Megrez2, a novel lightweight and high-performance language model architecture optimized for device native deployment. Megrez2 introduces a novel cross-layer expert sharing mechanism, which significantly reduces total parameter count by reusing expert modules across adjacent transformer layers while maintaining most of the model's capacity. It also incorporates pre-gated routing, enabling memory-efficient expert loading and faster inference. As the first instantiation of the Megrez2 architecture, we introduce the Megrez2-Preview model, which is pre-trained on a 5-trillion-token corpus and further enhanced through supervised fine-tuning and reinforcement learning with verifiable rewards. With only 3B activated and 7.5B stored parameters, Megrez2-Preview demonstrates competitive or superior performance compared to larger models on a wide range of tasks, including language understanding, instruction following, mathematical reasoning, and code generation. These results highlight the effectiveness of the Megrez2 architecture to achieve a balance between accuracy, efficiency, and deployability, making it a strong candidate for real-world, resource-constrained applications. 

---
# AI Telephone Surveying: Automating Quantitative Data Collection with an AI Interviewer 

**Authors**: Danny D. Leybzon, Shreyas Tirumala, Nishant Jain, Summer Gillen, Michael Jackson, Cameron McPhee, Jennifer Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17718)  

**Abstract**: With the rise of voice-enabled artificial intelligence (AI) systems, quantitative survey researchers have access to a new data-collection mode: AI telephone surveying. By using AI to conduct phone interviews, researchers can scale quantitative studies while balancing the dual goals of human-like interactivity and methodological rigor. Unlike earlier efforts that used interactive voice response (IVR) technology to automate these surveys, voice AI enables a more natural and adaptive respondent experience as it is more robust to interruptions, corrections, and other idiosyncrasies of human speech.
We built and tested an AI system to conduct quantitative surveys based on large language models (LLM), automatic speech recognition (ASR), and speech synthesis technologies. The system was specifically designed for quantitative research, and strictly adhered to research best practices like question order randomization, answer order randomization, and exact wording.
To validate the system's effectiveness, we deployed it to conduct two pilot surveys with the SSRS Opinion Panel and followed-up with a separate human-administered survey to assess respondent experiences. We measured three key metrics: the survey completion rates, break-off rates, and respondent satisfaction scores. Our results suggest that shorter instruments and more responsive AI interviewers may contribute to improvements across all three metrics studied. 

---
# From Feedback to Checklists: Grounded Evaluation of AI-Generated Clinical Notes 

**Authors**: Karen Zhou, John Giorgi, Pranav Mani, Peng Xu, Davis Liang, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17717)  

**Abstract**: AI-generated clinical notes are increasingly used in healthcare, but evaluating their quality remains a challenge due to high subjectivity and limited scalability of expert review. Existing automated metrics often fail to align with real-world physician preferences. To address this, we propose a pipeline that systematically distills real user feedback into structured checklists for note evaluation. These checklists are designed to be interpretable, grounded in human feedback, and enforceable by LLM-based evaluators. Using deidentified data from over 21,000 clinical encounters, prepared in accordance with the HIPAA safe harbor standard, from a deployed AI medical scribe system, we show that our feedback-derived checklist outperforms baseline approaches in our offline evaluations in coverage, diversity, and predictive power for human ratings. Extensive experiments confirm the checklist's robustness to quality-degrading perturbations, significant alignment with clinician preferences, and practical value as an evaluation methodology. In offline research settings, the checklist can help identify notes likely to fall below our chosen quality thresholds. 

---
# TyDi QA-WANA: A Benchmark for Information-Seeking Question Answering in Languages of West Asia and North Africa 

**Authors**: Parker Riley, Siamak Shakeri, Waleed Ammar, Jonathan H. Clark  

**Link**: [PDF](https://arxiv.org/pdf/2507.17709)  

**Abstract**: We present TyDi QA-WANA, a question-answering dataset consisting of 28K examples divided among 10 language varieties of western Asia and northern Africa. The data collection process was designed to elicit information-seeking questions, where the asker is genuinely curious to know the answer. Each question in paired with an entire article that may or may not contain the answer; the relatively large size of the articles results in a task suitable for evaluating models' abilities to utilize large text contexts in answering questions. Furthermore, the data was collected directly in each language variety, without the use of translation, in order to avoid issues of cultural relevance. We present performance of two baseline models, and release our code and data to facilitate further improvement by the research community. 

---
# Towards Greater Leverage: Scaling Laws for Efficient Mixture-of-Experts Language Models 

**Authors**: Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.17702)  

**Abstract**: Mixture-of-Experts (MoE) has become a dominant architecture for scaling Large Language Models (LLMs) efficiently by decoupling total parameters from computational cost. However, this decoupling creates a critical challenge: predicting the model capacity of a given MoE configurations (e.g., expert activation ratio and granularity) remains an unresolved problem. To address this gap, we introduce Efficiency Leverage (EL), a metric quantifying the computational advantage of an MoE model over a dense equivalent. We conduct a large-scale empirical study, training over 300 models up to 28B parameters, to systematically investigate the relationship between MoE architectural configurations and EL. Our findings reveal that EL is primarily driven by the expert activation ratio and the total compute budget, both following predictable power laws, while expert granularity acts as a non-linear modulator with a clear optimal range. We integrate these discoveries into a unified scaling law that accurately predicts the EL of an MoE architecture based on its configuration. To validate our derived scaling laws, we designed and trained Ling-mini-beta, a pilot model for Ling-2.0 series with only 0.85B active parameters, alongside a 6.1B dense model for comparison. When trained on an identical 1T high-quality token dataset, Ling-mini-beta matched the performance of the 6.1B dense model while consuming over 7x fewer computational resources, thereby confirming the accuracy of our scaling laws. This work provides a principled and empirically-grounded foundation for the scaling of efficient MoE models. 

---
# Who Attacks, and Why? Using LLMs to Identify Negative Campaigning in 18M Tweets across 19 Countries 

**Authors**: Victor Hartman, Petter Törnberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.17636)  

**Abstract**: Negative campaigning is a central feature of political competition, yet empirical research has been limited by the high cost and limited scalability of existing classification methods. This study makes two key contributions. First, it introduces zero-shot Large Language Models (LLMs) as a novel approach for cross-lingual classification of negative campaigning. Using benchmark datasets in ten languages, we demonstrate that LLMs achieve performance on par with native-speaking human coders and outperform conventional supervised machine learning approaches. Second, we leverage this novel method to conduct the largest cross-national study of negative campaigning to date, analyzing 18 million tweets posted by parliamentarians in 19 European countries between 2017 and 2022. The results reveal consistent cross-national patterns: governing parties are less likely to use negative messaging, while ideologically extreme and populist parties -- particularly those on the radical right -- engage in significantly higher levels of negativity. These findings advance our understanding of how party-level characteristics shape strategic communication in multiparty systems. More broadly, the study demonstrates the potential of LLMs to enable scalable, transparent, and replicable research in political communication across linguistic and cultural contexts. 

---
# WSM: Decay-Free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training 

**Authors**: Changxin Tian, Jiapeng Wang, Qian Zhao, Kunlong Chen, Jia Liu, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.17634)  

**Abstract**: Recent advances in learning rate (LR) scheduling have demonstrated the effectiveness of decay-free approaches that eliminate the traditional decay phase while maintaining competitive performance. Model merging techniques have emerged as particularly promising solutions in this domain. We present Warmup-Stable and Merge (WSM), a general framework that establishes a formal connection between learning rate decay and model merging. WSM provides a unified theoretical foundation for emulating various decay strategies-including cosine decay, linear decay and inverse square root decay-as principled model averaging schemes, while remaining fully compatible with diverse optimization methods. Through extensive experiments, we identify merge duration-the training window for checkpoint aggregation-as the most critical factor influencing model performance, surpassing the importance of both checkpoint interval and merge quantity. Our framework consistently outperforms the widely-adopted Warmup-Stable-Decay (WSD) approach across multiple benchmarks, achieving significant improvements of +3.5% on MATH, +2.9% on HumanEval, and +5.5% on MMLU-Pro. The performance advantages extend to supervised fine-tuning scenarios, highlighting WSM's potential for long-term model refinement. 

---
# A Hybrid Early-Exit Algorithm for Large Language Models Based on Space Alignment Decoding (SPADE) 

**Authors**: Bowen Zheng, Ming Ma, Zhongqiao Lin, Tianming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17618)  

**Abstract**: Large language models are computationally expensive due to their deep structures. Prior research has shown that intermediate layers contain sufficient information to generate accurate answers, leading to the development of early-exit algorithms that reduce inference costs by terminating computation at earlier layers. However, these methods often suffer from poor performance due to misalignment between intermediate and output layer representations that lead to decoding inaccuracy. To address these challenges, we propose SPADE (SPace Alignment DEcoding), a novel decoding method that aligns intermediate layer representations with the output layer by propagating a minimally reduced sequence consisting of only the start token and the answer token. We further optimize the early-exit decision-making process by training a linear approximation of SPADE that computes entropy-based confidence metrics. Putting them together, we create a hybrid early-exit algorithm that monitors confidence levels and stops inference at intermediate layers while using SPADE to generate high-quality outputs. This approach significantly reduces inference costs without compromising accuracy, offering a scalable and efficient solution for deploying large language models in real-world applications. 

---
# Synthetic Voice Data for Automatic Speech Recognition in African Languages 

**Authors**: Brian DeRenzi, Anna Dixon, Mohamed Aymane Farhi, Christian Resch  

**Link**: [PDF](https://arxiv.org/pdf/2507.17578)  

**Abstract**: Speech technology remains out of reach for most of the over 2300 languages in Africa. We present the first systematic assessment of large-scale synthetic voice corpora for African ASR. We apply a three-step process: LLM-driven text creation, TTS voice synthesis, and ASR fine-tuning. Eight out of ten languages for which we create synthetic text achieved readability scores above 5 out of 7. We evaluated ASR improvement for three (Hausa, Dholuo, Chichewa) and created more than 2,500 hours of synthetic voice data at below 1% of the cost of real data. Fine-tuned Wav2Vec-BERT-2.0 models trained on 250h real and 250h synthetic Hausa matched a 500h real-data-only baseline, while 579h real and 450h to 993h synthetic data created the best performance. We also present gender-disaggregated ASR performance evaluation. For very low-resource languages, gains varied: Chichewa WER improved about 6.5% relative with a 1:2 real-to-synthetic ratio; a 1:1 ratio for Dholuo showed similar improvements on some evaluation data, but not on others. Investigating intercoder reliability, ASR errors and evaluation datasets revealed the need for more robust reviewer protocols and more accurate evaluation data. All data and models are publicly released to invite further work to improve synthetic data for African languages. 

---
# Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice 

**Authors**: Shanbo Cheng, Yu Bao, Zhichao Huang, Yu Lu, Ningxin Peng, Lu Xu, Runsheng Yu, Rong Cao, Ting Han, Zeyang Li, Sitong Liu, Shengtao Ma, Shiguang Pan, Jiongchen Xiao, Nuo Xu, Meng Yang, Rong Ye, Yiming Yu, Ruofei Zhang, Wanyi Zhang, Wenhao Zhu, Liehao Zou, Lu Lu, Yuxuan Wang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17527)  

**Abstract**: Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability. 

---
# MultiNRC: A Challenging and Native Multilingual Reasoning Evaluation Benchmark for LLMs 

**Authors**: Alexander R. Fabbri, Diego Mares, Jorge Flores, Meher Mankikar, Ernesto Hernandez, Dean Lee, Bing Liu, Chen Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.17476)  

**Abstract**: Although recent Large Language Models (LLMs) have shown rapid improvement on reasoning benchmarks in English, the evaluation of such LLMs' multilingual reasoning capability across diverse languages and cultural contexts remains limited. Existing multilingual reasoning benchmarks are typically constructed by translating existing English reasoning benchmarks, biasing these benchmarks towards reasoning problems with context in English language/cultures. In this work, we introduce the Multilingual Native Reasoning Challenge (MultiNRC), a benchmark designed to assess LLMs on more than 1,000 native, linguistic and culturally grounded reasoning questions written by native speakers in French, Spanish, and Chinese. MultiNRC covers four core reasoning categories: language-specific linguistic reasoning, wordplay & riddles, cultural/tradition reasoning, and math reasoning with cultural relevance. For cultural/tradition reasoning and math reasoning with cultural relevance, we also provide English equivalent translations of the multilingual questions by manual translation from native speakers fluent in English. This set of English equivalents can provide a direct comparison of LLM reasoning capacity in other languages vs. English on the same reasoning questions. We systematically evaluate current 14 leading LLMs covering most LLM families on MultiNRC and its English equivalent set. The results show that (1) current LLMs are still not good at native multilingual reasoning, with none scoring above 50% on MultiNRC; (2) LLMs exhibit distinct strengths and weaknesses in handling linguistic, cultural, and logical reasoning tasks; (3) Most models perform substantially better in math reasoning in English compared to in original languages (+10%), indicating persistent challenges with culturally grounded knowledge. 

---
# Each to Their Own: Exploring the Optimal Embedding in RAG 

**Authors**: Shiting Chen, Zijian Zhao, Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.17442)  

**Abstract**: Recently, as Large Language Models (LLMs) have fundamentally impacted various fields, the methods for incorporating up-to-date information into LLMs or adding external knowledge to construct domain-specific models have garnered wide attention. Retrieval-Augmented Generation (RAG), serving as an inference-time scaling method, is notable for its low cost and minimal effort for parameter tuning. However, due to heterogeneous training data and model architecture, the variant embedding models used in RAG exhibit different benefits across various areas, often leading to different similarity calculation results and, consequently, varying response quality from LLMs. To address this problem, we propose and examine two approaches to enhance RAG by combining the benefits of multiple embedding models, named Mixture-Embedding RAG and Confident RAG. Mixture-Embedding RAG simply sorts and selects retrievals from multiple embedding models based on standardized similarity; however, it does not outperform vanilla RAG. In contrast, Confident RAG generates responses multiple times using different embedding models and then selects the responses with the highest confidence level, demonstrating average improvements of approximately 10% and 5% over vanilla LLMs and RAG, respectively. The consistent results across different LLMs and embedding models indicate that Confident RAG is an efficient plug-and-play approach for various domains. We will release our code upon publication. 

---
# Investigating Subjective Factors of Argument Strength: Storytelling, Emotions, and Hedging 

**Authors**: Carlotta Quensel, Neele Falk, Gabriella Lapesa  

**Link**: [PDF](https://arxiv.org/pdf/2507.17409)  

**Abstract**: In assessing argument strength, the notions of what makes a good argument are manifold. With the broader trend towards treating subjectivity as an asset and not a problem in NLP, new dimensions of argument quality are studied. Although studies on individual subjective features like personal stories exist, there is a lack of large-scale analyses of the relation between these features and argument strength. To address this gap, we conduct regression analysis to quantify the impact of subjective factors $-$ emotions, storytelling, and hedging $-$ on two standard datasets annotated for objective argument quality and subjective persuasion. As such, our contribution is twofold: at the level of contributed resources, as there are no datasets annotated with all studied dimensions, this work compares and evaluates automated annotation methods for each subjective feature. At the level of novel insights, our regression analysis uncovers different patterns of impact of subjective features on the two facets of argument strength encoded in the datasets. Our results show that storytelling and hedging have contrasting effects on objective and subjective argument quality, while the influence of emotions depends on their rhetoric utilization rather than the domain. 

---
# Millions of $\text{GeAR}$-s: Extending GraphRAG to Millions of Documents 

**Authors**: Zhili Shen, Chenxin Diao, Pascual Merita, Pavlos Vougiouklis, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17399)  

**Abstract**: Recent studies have explored graph-based approaches to retrieval-augmented generation, leveraging structured or semi-structured information -- such as entities and their relations extracted from documents -- to enhance retrieval. However, these methods are typically designed to address specific tasks, such as multi-hop question answering and query-focused summarisation, and therefore, there is limited evidence of their general applicability across broader datasets. In this paper, we aim to adapt a state-of-the-art graph-based RAG solution: $\text{GeAR}$ and explore its performance and limitations on the SIGIR 2025 LiveRAG Challenge. 

---
# Triple X: A LLM-Based Multilingual Speech Recognition System for the INTERSPEECH2025 MLC-SLM Challenge 

**Authors**: Miaomiao Gao, Xiaoxiao Xiang, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.17288)  

**Abstract**: This paper describes our Triple X speech recognition system submitted to Task 1 of the Multi-Lingual Conversational Speech Language Modeling (MLC-SLM) Challenge. Our work focuses on optimizing speech recognition accuracy in multilingual conversational scenarios through an innovative encoder-adapter-LLM architecture. This framework harnesses the powerful reasoning capabilities of text-based large language models while incorporating domain-specific adaptations. To further enhance multilingual recognition performance, we adopted a meticulously designed multi-stage training strategy leveraging extensive multilingual audio datasets. Experimental results demonstrate that our approach achieves competitive Word Error Rate (WER) performance on both dev and test sets, obtaining second place in the challenge ranking. 

---
# CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings 

**Authors**: Kyeongkyu Lee, Seonghwan Yoon, Hongki Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.17234)  

**Abstract**: Automatic generation of radiology reports has the potential to alleviate radiologists' significant workload, yet current methods struggle to deliver clinically reliable conclusions. In particular, most prior approaches focus on producing fluent text without effectively ensuring the factual correctness of the reports and often rely on single-view images, limiting diagnostic comprehensiveness. We propose CLARIFID, a novel framework that directly optimizes diagnostic correctness by mirroring the two-step workflow of experts. Specifically, CLARIFID (1) learns the logical flow from Findings to Impression through section-aware pretraining, (2) is fine-tuned with Proximal Policy Optimization in which the CheXbert F1 score of the Impression section serves as the reward, (3) enforces reasoning-aware decoding that completes "Findings" before synthesizing the "Impression", and (4) fuses multiple chest X-ray views via a vision-transformer-based multi-view encoder. During inference, we apply a reasoning-aware next-token forcing strategy followed by report-level re-ranking, ensuring that the model first produces a comprehensive Findings section before synthesizing the Impression and thereby preserving coherent clinical reasoning. Experimental results on the MIMIC-CXR dataset demonstrate that our method achieves superior clinical efficacy and outperforms existing baselines on both standard NLG metrics and clinically aware scores. 

---
# The Pluralistic Moral Gap: Understanding Judgment and Value Differences between Humans and Large Language Models 

**Authors**: Giuseppe Russo, Debora Nozza, Paul Röttger, Dirk Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2507.17216)  

**Abstract**: People increasingly rely on Large Language Models (LLMs) for moral advice, which may influence humans' decisions. Yet, little is known about how closely LLMs align with human moral judgments. To address this, we introduce the Moral Dilemma Dataset, a benchmark of 1,618 real-world moral dilemmas paired with a distribution of human moral judgments consisting of a binary evaluation and a free-text rationale. We treat this problem as a pluralistic distributional alignment task, comparing the distributions of LLM and human judgments across dilemmas. We find that models reproduce human judgments only under high consensus; alignment deteriorates sharply when human disagreement increases. In parallel, using a 60-value taxonomy built from 3,783 value expressions extracted from rationales, we show that LLMs rely on a narrower set of moral values than humans. These findings reveal a pluralistic moral gap: a mismatch in both the distribution and diversity of values expressed. To close this gap, we introduce Dynamic Moral Profiling (DMP), a Dirichlet-based sampling method that conditions model outputs on human-derived value profiles. DMP improves alignment by 64.3% and enhances value diversity, offering a step toward more pluralistic and human-aligned moral guidance from LLMs. 

---
# FinGAIA: An End-to-End Benchmark for Evaluating AI Agents in Finance 

**Authors**: Lingfeng Zeng, Fangqi Lou, Zixuan Wang, Jiajie Xu, Jinyi Niu, Mengping Li, Yifan Dong, Qi Qi, Wei Zhang, Ziwei Yang, Jun Han, Ruilun Feng, Ruiqi Hu, Lejie Zhang, Zhengbo Feng, Yicheng Ren, Xin Guo, Zhaowei Liu, Dongpo Cheng, Weige Cai, Liwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17186)  

**Abstract**: The booming development of AI agents presents unprecedented opportunities for automating complex tasks across various domains. However, their multi-step, multi-tool collaboration capabilities in the financial sector remain underexplored. This paper introduces FinGAIA, an end-to-end benchmark designed to evaluate the practical abilities of AI agents in the financial domain. FinGAIA comprises 407 meticulously crafted tasks, spanning seven major financial sub-domains: securities, funds, banking, insurance, futures, trusts, and asset management. These tasks are organized into three hierarchical levels of scenario depth: basic business analysis, asset decision support, and strategic risk management. We evaluated 10 mainstream AI agents in a zero-shot setting. The best-performing agent, ChatGPT, achieved an overall accuracy of 48.9\%, which, while superior to non-professionals, still lags financial experts by over 35 percentage points. Error analysis has revealed five recurring failure patterns: Cross-modal Alignment Deficiency, Financial Terminological Bias, Operational Process Awareness Barrier, among others. These patterns point to crucial directions for future research. Our work provides the first agent benchmark closely related to the financial domain, aiming to objectively assess and promote the development of agents in this crucial field. Partial data is available at this https URL. 

---
# SKA-Bench: A Fine-Grained Benchmark for Evaluating Structured Knowledge Understanding of LLMs 

**Authors**: Zhiqiang Liu, Enpei Niu, Yin Hua, Mengshu Sun, Lei Liang, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17178)  

**Abstract**: Although large language models (LLMs) have made significant progress in understanding Structured Knowledge (SK) like KG and Table, existing evaluations for SK understanding are non-rigorous (i.e., lacking evaluations of specific capabilities) and focus on a single type of SK. Therefore, we aim to propose a more comprehensive and rigorous structured knowledge understanding benchmark to diagnose the shortcomings of LLMs. In this paper, we introduce SKA-Bench, a Structured Knowledge Augmented QA Benchmark that encompasses four widely used structured knowledge forms: KG, Table, KG+Text, and Table+Text. We utilize a three-stage pipeline to construct SKA-Bench instances, which includes a question, an answer, positive knowledge units, and noisy knowledge units. To evaluate the SK understanding capabilities of LLMs in a fine-grained manner, we expand the instances into four fundamental ability testbeds: Noise Robustness, Order Insensitivity, Information Integration, and Negative Rejection. Empirical evaluations on 8 representative LLMs, including the advanced DeepSeek-R1, indicate that existing LLMs still face significant challenges in understanding structured knowledge, and their performance is influenced by factors such as the amount of noise, the order of knowledge units, and hallucination phenomenon. Our dataset and code are available at this https URL. 

---
# CogDual: Enhancing Dual Cognition of LLMs via Reinforcement Learning with Implicit Rule-Based Rewards 

**Authors**: Cheng Liu, Yifei Lu, Fanghua Ye, Jian Li, Xingyu Chen, Feiliang Ren, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.17147)  

**Abstract**: Role-Playing Language Agents (RPLAs) have emerged as a significant application direction for Large Language Models (LLMs). Existing approaches typically rely on prompt engineering or supervised fine-tuning to enable models to imitate character behaviors in specific scenarios, but often neglect the underlying \emph{cognitive} mechanisms driving these behaviors. Inspired by cognitive psychology, we introduce \textbf{CogDual}, a novel RPLA adopting a \textit{cognize-then-respond } reasoning paradigm. By jointly modeling external situational awareness and internal self-awareness, CogDual generates responses with improved character consistency and contextual alignment. To further optimize the performance, we employ reinforcement learning with two general-purpose reward schemes designed for open-domain text generation. Extensive experiments on the CoSER benchmark, as well as Cross-MR and LifeChoice, demonstrate that CogDual consistently outperforms existing baselines and generalizes effectively across diverse role-playing tasks. 

---
# Evolutionary Feature-wise Thresholding for Binary Representation of NLP Embeddings 

**Authors**: Soumen Sinha, Shahryar Rahnamayan, Azam Asilian Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.17025)  

**Abstract**: Efficient text embedding is crucial for large-scale natural language processing (NLP) applications, where storage and computational efficiency are key concerns. In this paper, we explore how using binary representations (barcodes) instead of real-valued features can be used for NLP embeddings derived from machine learning models such as BERT. Thresholding is a common method for converting continuous embeddings into binary representations, often using a fixed threshold across all features. We propose a Coordinate Search-based optimization framework that instead identifies the optimal threshold for each feature, demonstrating that feature-specific thresholds lead to improved performance in binary encoding. This ensures that the binary representations are both accurate and efficient, enhancing performance across various features. Our optimal barcode representations have shown promising results in various NLP applications, demonstrating their potential to transform text representation. We conducted extensive experiments and statistical tests on different NLP tasks and datasets to evaluate our approach and compare it to other thresholding methods. Binary embeddings generated using using optimal thresholds found by our method outperform traditional binarization methods in accuracy. This technique for generating binary representations is versatile and can be applied to any features, not just limited to NLP embeddings, making it useful for a wide range of domains in machine learning applications. 

---
# Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge? 

**Authors**: Arduin Findeis, Floris Weers, Guoli Yin, Ke Ye, Ruoming Pang, Tom Gunter  

**Link**: [PDF](https://arxiv.org/pdf/2507.17015)  

**Abstract**: Pairwise preferences over model responses are widely collected to evaluate and provide feedback to large language models (LLMs). Given two alternative model responses to the same input, a human or AI annotator selects the "better" response. This approach can provide feedback for domains where other hard-coded metrics are difficult to obtain (e.g., chat response quality), thereby helping model evaluation or training. However, for some domains high-quality pairwise comparisons can be tricky to obtain - from AI and humans. For example, for responses with many factual statements, annotators may disproportionately weigh writing quality rather than underlying facts. In this work, we explore augmenting standard AI annotator systems with additional tools to improve performance on three challenging response domains: long-form factual, math and code tasks. We propose a tool-using agentic system to provide higher quality feedback on these domains. Our system uses web-search and code execution to ground itself based on external validation, independent of the LLM's internal knowledge and biases. We provide extensive experimental results evaluating our method across the three targeted response domains as well as general annotation tasks, using RewardBench (incl. AlpacaEval and LLMBar), RewardMath, as well as three new datasets for domains with saturated pre-existing datasets. Our results indicate that external tools can indeed improve performance in many, but not all, cases. More generally, our experiments highlight the sensitivity of performance to simple parameters (e.g., prompt) and the need for improved (non-saturated) annotator benchmarks. We share our code at this https URL. 

---
# Multi-Label Classification with Generative AI Models in Healthcare: A Case Study of Suicidality and Risk Factors 

**Authors**: Ming Huang, Zehan Li, Yan Hu, Wanjing Wang, Andrew Wen, Scott Lane, Salih Selek, Lokesh Shahani, Rodrigo Machado-Vieira, Jair Soares, Hua Xu, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17009)  

**Abstract**: Suicide remains a pressing global health crisis, with over 720,000 deaths annually and millions more affected by suicide ideation (SI) and suicide attempts (SA). Early identification of suicidality-related factors (SrFs), including SI, SA, exposure to suicide (ES), and non-suicidal self-injury (NSSI), is critical for timely intervention. While prior studies have applied AI to detect SrFs in clinical notes, most treat suicidality as a binary classification task, overlooking the complexity of cooccurring risk factors. This study explores the use of generative large language models (LLMs), specifically GPT-3.5 and GPT-4.5, for multi-label classification (MLC) of SrFs from psychiatric electronic health records (EHRs). We present a novel end to end generative MLC pipeline and introduce advanced evaluation methods, including label set level metrics and a multilabel confusion matrix for error analysis. Finetuned GPT-3.5 achieved top performance with 0.94 partial match accuracy and 0.91 F1 score, while GPT-4.5 with guided prompting showed superior performance across label sets, including rare or minority label sets, indicating a more balanced and robust performance. Our findings reveal systematic error patterns, such as the conflation of SI and SA, and highlight the models tendency toward cautious over labeling. This work not only demonstrates the feasibility of using generative AI for complex clinical classification tasks but also provides a blueprint for structuring unstructured EHR data to support large scale clinical research and evidence based medicine. 

---
# Obscured but Not Erased: Evaluating Nationality Bias in LLMs via Name-Based Bias Benchmarks 

**Authors**: Giulio Pelosio, Devesh Batra, Noémie Bovey, Robert Hankache, Cristovao Iglesias, Greig Cowan, Raad Khraishi  

**Link**: [PDF](https://arxiv.org/pdf/2507.16989)  

**Abstract**: Large Language Models (LLMs) can exhibit latent biases towards specific nationalities even when explicit demographic markers are not present. In this work, we introduce a novel name-based benchmarking approach derived from the Bias Benchmark for QA (BBQ) dataset to investigate the impact of substituting explicit nationality labels with culturally indicative names, a scenario more reflective of real-world LLM applications. Our novel approach examines how this substitution affects both bias magnitude and accuracy across a spectrum of LLMs from industry leaders such as OpenAI, Google, and Anthropic. Our experiments show that small models are less accurate and exhibit more bias compared to their larger counterparts. For instance, on our name-based dataset and in the ambiguous context (where the correct choice is not revealed), Claude Haiku exhibited the worst stereotypical bias scores of 9%, compared to only 3.5% for its larger counterpart, Claude Sonnet, where the latter also outperformed it by 117.7% in accuracy. Additionally, we find that small models retain a larger portion of existing errors in these ambiguous contexts. For example, after substituting names for explicit nationality references, GPT-4o retains 68% of the error rate versus 76% for GPT-4o-mini, with similar findings for other model providers, in the ambiguous context. Our research highlights the stubborn resilience of biases in LLMs, underscoring their profound implications for the development and deployment of AI systems in diverse, global contexts. 

---
# Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain 

**Authors**: Rishemjit Kaur, Arshdeep Singh Bhankhar, Surangika Ranathunga, Jashanpreet Singh Salh, Sudhir Rajput, Vidhi, Kashish Mahendra, Bhavika Berwal, Ritesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.16974)  

**Abstract**: Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Although large language models (LLMs) can be used to implement Question Answering (QA) systems, simply using publicly available general-purpose LLMs in agriculture typically offer generic advisories, lacking precision in local and multilingual contexts due to insufficient domain-specific training and scarcity of high-quality, region-specific datasets. Our study addresses these limitations by generating multilingual synthetic agricultural datasets (English, Hindi, Punjabi) from agriculture-specific documents and fine-tuning language-specific LLMs. Our evaluation on curated multilingual datasets demonstrates significant improvements in factual accuracy, relevance, and agricultural consensus for the fine-tuned models compared to their baseline counterparts. These results highlight the efficacy of synthetic data-driven, language-specific fine-tuning as an effective strategy to improve the performance of LLMs in agriculture, especially in multilingual and low-resource settings. By enabling more accurate and localized agricultural advisory services, this study provides a meaningful step toward bridging the knowledge gap in AI-driven agricultural solutions for diverse linguistic communities. 

---
# Text-to-SPARQL Goes Beyond English: Multilingual Question Answering Over Knowledge Graphs through Human-Inspired Reasoning 

**Authors**: Aleksandr Perevalov, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2507.16971)  

**Abstract**: Accessing knowledge via multilingual natural-language interfaces is one of the emerging challenges in the field of information retrieval and related ones. Structured knowledge stored in knowledge graphs can be queried via a specific query language (e.g., SPARQL). Therefore, one needs to transform natural-language input into a query to fulfill an information need. Prior approaches mostly focused on combining components (e.g., rule-based or neural-based) that solve downstream tasks and come up with an answer at the end. We introduce mKGQAgent, a human-inspired framework that breaks down the task of converting natural language questions into SPARQL queries into modular, interpretable subtasks. By leveraging a coordinated LLM agent workflow for planning, entity linking, and query refinement - guided by an experience pool for in-context learning - mKGQAgent efficiently handles multilingual KGQA. Evaluated on the DBpedia- and Corporate-based KGQA benchmarks within the Text2SPARQL challenge 2025, our approach took first place among the other participants. This work opens new avenues for developing human-like reasoning systems in multilingual semantic parsing. 

---
# Harnessing RLHF for Robust Unanswerability Recognition and Trustworthy Response Generation in LLMs 

**Authors**: Shuyuan Lin, Lei Duan, Philip Hughes, Yuxuan Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.16951)  

**Abstract**: Conversational Information Retrieval (CIR) systems, while offering intuitive access to information, face a significant challenge: reliably handling unanswerable questions to prevent the generation of misleading or hallucinated content. Traditional approaches often rely on external classifiers, which can introduce inconsistencies with the core generative Large Language Models (LLMs). This paper introduces Self-Aware LLM for Unanswerability (SALU), a novel approach that deeply integrates unanswerability detection directly within the LLM's generative process. SALU is trained using a multi-task learning framework for both standard Question Answering (QA) and explicit abstention generation for unanswerable queries. Crucially, it incorporates a confidence-score-guided reinforcement learning with human feedback (RLHF) phase, which explicitly penalizes hallucinated responses and rewards appropriate abstentions, fostering intrinsic self-awareness of knowledge boundaries. Through extensive experiments on our custom-built C-IR_Answerability dataset, SALU consistently outperforms strong baselines, including hybrid LLM-classifier systems, in overall accuracy for correctly answering or abstaining from questions. Human evaluation further confirms SALU's superior reliability, achieving high scores in factuality, appropriate abstention, and, most importantly, a dramatic reduction in hallucination, demonstrating its ability to robustly "know when to say 'I don't know'." 

---
# AI-based Clinical Decision Support for Primary Care: A Real-World Study 

**Authors**: Robert Korom, Sarah Kiptinness, Najib Adan, Kassim Said, Catherine Ithuli, Oliver Rotich, Boniface Kimani, Irene King'ori, Stellah Kamau, Elizabeth Atemba, Muna Aden, Preston Bowman, Michael Sharman, Rebecca Soskin Hicks, Rebecca Distler, Johannes Heidecke, Rahul K. Arora, Karan Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2507.16947)  

**Abstract**: We evaluate the impact of large language model-based clinical decision support in live care. In partnership with Penda Health, a network of primary care clinics in Nairobi, Kenya, we studied AI Consult, a tool that serves as a safety net for clinicians by identifying potential documentation and clinical decision-making errors. AI Consult integrates into clinician workflows, activating only when needed and preserving clinician autonomy. We conducted a quality improvement study, comparing outcomes for 39,849 patient visits performed by clinicians with or without access to AI Consult across 15 clinics. Visits were rated by independent physicians to identify clinical errors. Clinicians with access to AI Consult made relatively fewer errors: 16% fewer diagnostic errors and 13% fewer treatment errors. In absolute terms, the introduction of AI Consult would avert diagnostic errors in 22,000 visits and treatment errors in 29,000 visits annually at Penda alone. In a survey of clinicians with AI Consult, all clinicians said that AI Consult improved the quality of care they delivered, with 75% saying the effect was "substantial". These results required a clinical workflow-aligned AI Consult implementation and active deployment to encourage clinician uptake. We hope this study demonstrates the potential for LLM-based clinical decision support tools to reduce errors in real-world settings and provides a practical framework for advancing responsible adoption. 

---
# A Unifying Scheme for Extractive Content Selection Tasks 

**Authors**: Shmuel Amar, Ori Shapira, Aviv Slobodkin, Ido Dagan  

**Link**: [PDF](https://arxiv.org/pdf/2507.16922)  

**Abstract**: A broad range of NLP tasks involve selecting relevant text spans from given source texts. Despite this shared objective, such \textit{content selection} tasks have traditionally been studied in isolation, each with its own modeling approaches, datasets, and evaluation metrics. In this work, we propose \textit{instruction-guided content selection (IGCS)} as a beneficial unified framework for such settings, where the task definition and any instance-specific request are encapsulated as instructions to a language model. To promote this framework, we introduce \igcsbench{}, the first unified benchmark covering diverse content selection tasks. Further, we create a large generic synthetic dataset that can be leveraged for diverse content selection tasks, and show that transfer learning with these datasets often boosts performance, whether dedicated training for the targeted task is available or not. Finally, we address generic inference time issues that arise in LLM-based modeling of content selection, assess a generic evaluation metric, and overall propose the utility of our resources and methods for future content selection models. Models and datasets available at this https URL. 

---
# Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains 

**Authors**: Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Bing Liu, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2507.17746)  

**Abstract**: Extending Reinforcement Learning with Verifiable Rewards (RLVR) to real-world tasks often requires balancing objective and subjective evaluation criteria. However, many such tasks lack a single, unambiguous ground truth-making it difficult to define reliable reward signals for post-training language models. While traditional preference-based methods offer a workaround, they rely on opaque reward functions that are difficult to interpret and prone to spurious correlations. We introduce $\textbf{Rubrics as Rewards}$ (RaR), a framework that uses structured, checklist-style rubrics as interpretable reward signals for on-policy training with GRPO. Our best RaR method yields up to a $28\%$ relative improvement on HealthBench-1k compared to simple Likert-based approaches, while matching or surpassing the performance of reward signals derived from expert-written references. By treating rubrics as structured reward signals, we show that RaR enables smaller-scale judge models to better align with human preferences and sustain robust performance across model scales. 

---
# Dual-branch Prompting for Multimodal Machine Translation 

**Authors**: Jie Wang, Zhendong Yang, Liansong Zong, Xiaobo Zhang, Dexian Wang, Ji Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17588)  

**Abstract**: Multimodal Machine Translation (MMT) typically enhances text-only translation by incorporating aligned visual features. Despite the remarkable progress, state-of-the-art MMT approaches often rely on paired image-text inputs at inference and are sensitive to irrelevant visual noise, which limits their robustness and practical applicability. To address these issues, we propose D2P-MMT, a diffusion-based dual-branch prompting framework for robust vision-guided translation. Specifically, D2P-MMT requires only the source text and a reconstructed image generated by a pre-trained diffusion model, which naturally filters out distracting visual details while preserving semantic cues. During training, the model jointly learns from both authentic and reconstructed images using a dual-branch prompting strategy, encouraging rich cross-modal interactions. To bridge the modality gap and mitigate training-inference discrepancies, we introduce a distributional alignment loss that enforces consistency between the output distributions of the two branches. Extensive experiments on the Multi30K dataset demonstrate that D2P-MMT achieves superior translation performance compared to existing state-of-the-art approaches. 

---
# BoSS: Beyond-Semantic Speech 

**Authors**: Qing Wang, Zehan Li, Hang Lv, Hongjie Chen, Yaodong Song, Jian Kang, Jie Lian, Jie Li, Yongxiang Li, Zhongjiang He, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.17563)  

**Abstract**: Human communication involves more than explicit semantics, with implicit signals and contextual cues playing a critical role in shaping meaning. However, modern speech technologies, such as Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) often fail to capture these beyond-semantic dimensions. To better characterize and benchmark the progression of speech intelligence, we introduce Spoken Interaction System Capability Levels (L1-L5), a hierarchical framework illustrated the evolution of spoken dialogue systems from basic command recognition to human-like social interaction. To support these advanced capabilities, we propose Beyond-Semantic Speech (BoSS), which refers to the set of information in speech communication that encompasses but transcends explicit semantics. It conveys emotions, contexts, and modifies or extends meanings through multidimensional features such as affective cues, contextual dynamics, and implicit semantics, thereby enhancing the understanding of communicative intentions and scenarios. We present a formalized framework for BoSS, leveraging cognitive relevance theories and machine learning models to analyze temporal and contextual speech dynamics. We evaluate BoSS-related attributes across five different dimensions, reveals that current spoken language models (SLMs) are hard to fully interpret beyond-semantic signals. These findings highlight the need for advancing BoSS research to enable richer, more context-aware human-machine communication. 

---
# URPO: A Unified Reward & Policy Optimization Framework for Large Language Models 

**Authors**: Songshuo Lu, Hua Wang, Zhi Chen, Yaohua Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17515)  

**Abstract**: Large-scale alignment pipelines typically pair a policy model with a separately trained reward model whose parameters remain frozen during reinforcement learning (RL). This separation creates a complex, resource-intensive pipeline and suffers from a performance ceiling due to a static reward signal. We propose a novel framework, Unified Reward & Policy Optimization (URPO), that unifies instruction-following ("player") and reward modeling ("referee") within a single model and a single training phase. Our method recasts all alignment data-including preference pairs, verifiable reasoning, and open-ended instructions-into a unified generative format optimized by a single Group-Relative Policy Optimization (GRPO) loop. This enables the model to learn from ground-truth preferences and verifiable logic while simultaneously generating its own rewards for open-ended tasks. Experiments on the Qwen2.5-7B model demonstrate URPO's superiority. Our unified model significantly outperforms a strong baseline using a separate generative reward model, boosting the instruction-following score on AlpacaEval from 42.24 to 44.84 and the composite reasoning average from 32.66 to 35.66. Furthermore, URPO cultivates a superior internal evaluator as a byproduct of training, achieving a RewardBench score of 85.15 and surpassing the dedicated reward model it replaces (83.55). By eliminating the need for a separate reward model and fostering a co-evolutionary dynamic between generation and evaluation, URPO presents a simpler, more efficient, and more effective path towards robustly aligned language models. 

---
# DNT: a Deeply Normalized Transformer that can be trained by Momentum SGD 

**Authors**: Xianbiao Qi, Marco Chen, Wenjie Xiao, Jiaquan Ye, Yelin He, Chun-Guang Li, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.17501)  

**Abstract**: Transformers have become the de facto backbone of modern deep learning, yet their training typically demands an advanced optimizer with adaptive learning rate like AdamW, rather than a momentum SGDW (mSGDW). Previous works show that it is mainly due to a heavy-tailed distribution of the gradients. In this paper, we introduce a Deeply Normalized Transformer (DNT), which is meticulously engineered to overcome this limitation enabling seamless training with vanilla mSGDW while yielding comparable performance to the Transformers trained via AdamW. To be specific, in DNT, we strategically integrate normalization techniques at proper positions in the Transformers to effectively modulate the Jacobian matrices of each layer, balance the influence of weights, activations, and their interactions, and thus enable the distributions of gradients concentrated. We provide both theoretical justifications of the normalization technique used in our DNT and extensive empirical evaluation on two popular Transformer architectures to validate that: a) DNT outperforms its counterparts (\ie, ViT and GPT), and b) DNT can be effectively trained with vanilla mSGDW. 

---
# TransLPRNet: Lite Vision-Language Network for Single/Dual-line Chinese License Plate Recognition 

**Authors**: Guangzhu Xu, Zhi Ke, Pengcheng Zuo, Bangjun Lei  

**Link**: [PDF](https://arxiv.org/pdf/2507.17335)  

**Abstract**: License plate recognition in open environments is widely applicable across various domains; however, the diversity of license plate types and imaging conditions presents significant challenges. To address the limitations encountered by CNN and CRNN-based approaches in license plate recognition, this paper proposes a unified solution that integrates a lightweight visual encoder with a text decoder, within a pre-training framework tailored for single and double-line Chinese license plates. To mitigate the scarcity of double-line license plate datasets, we constructed a single/double-line license plate dataset by synthesizing images, applying texture mapping onto real scenes, and blending them with authentic license plate images. Furthermore, to enhance the system's recognition accuracy, we introduce a perspective correction network (PTN) that employs license plate corner coordinate regression as an implicit variable, supervised by license plate view classification information. This network offers improved stability, interpretability, and low annotation costs. The proposed algorithm achieves an average recognition accuracy of 99.34% on the corrected CCPD test set under coarse localization disturbance. When evaluated under fine localization disturbance, the accuracy further improves to 99.58%. On the double-line license plate test set, it achieves an average recognition accuracy of 98.70%, with processing speeds reaching up to 167 frames per second, indicating strong practical applicability. 

---
# Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs 

**Authors**: Eyal German, Sagiv Antebi, Daniel Samira, Asaf Shabtai, Yuval Elovici  

**Link**: [PDF](https://arxiv.org/pdf/2507.17259)  

**Abstract**: Large language models (LLMs) are increasingly trained on tabular data, which, unlike unstructured text, often contains personally identifiable information (PII) in a highly structured and explicit format. As a result, privacy risks arise, since sensitive records can be inadvertently retained by the model and exposed through data extraction or membership inference attacks (MIAs). While existing MIA methods primarily target textual content, their efficacy and threat implications may differ when applied to structured data, due to its limited content, diverse data types, unique value distributions, and column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used. Tab-MIA comprises five data collections, each represented in six different encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation of state-of-the-art MIA methods on LLMs finetuned with tabular data across multiple encoding formats. In the evaluation, we analyze the memorization behavior of pretrained LLMs on structured data derived from Wikipedia tables. Our findings show that LLMs memorize tabular data in ways that vary across encoding formats, making them susceptible to extraction via MIAs. Even when fine-tuned for as few as three epochs, models exhibit high vulnerability, with AUROC scores approaching 90% in most cases. Tab-MIA enables systematic evaluation of these risks and provides a foundation for developing privacy-preserving methods for tabular data in LLMs. 

---
# A Highly Clean Recipe Dataset with Ingredient States Annotation for State Probing Task 

**Authors**: Mashiro Toyooka, Kiyoharu Aizawa, Yoko Yamakata  

**Link**: [PDF](https://arxiv.org/pdf/2507.17232)  

**Abstract**: Large Language Models (LLMs) are trained on a vast amount of procedural texts, but they do not directly observe real-world phenomena. In the context of cooking recipes, this poses a challenge, as intermediate states of ingredients are often omitted, making it difficult for models to track ingredient states and understand recipes accurately. In this paper, we apply state probing, a method for evaluating a language model's understanding of the world, to the domain of cooking. We propose a new task and dataset for evaluating how well LLMs can recognize intermediate ingredient states during cooking procedures. We first construct a new Japanese recipe dataset with clear and accurate annotations of ingredient state changes, collected from well-structured and controlled recipe texts. Using this dataset, we design three novel tasks to evaluate whether LLMs can track ingredient state transitions and identify ingredients present at intermediate steps. Our experiments with widely used LLMs, such as Llama3.1-70B and Qwen2.5-72B, show that learning ingredient state knowledge improves their understanding of cooking processes, achieving performance comparable to commercial LLMs. 

---
# SiLQ: Simple Large Language Model Quantization-Aware Training 

**Authors**: Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha  

**Link**: [PDF](https://arxiv.org/pdf/2507.16933)  

**Abstract**: Large language models can be quantized to reduce inference time latency, model size, and energy consumption, thereby delivering a better user experience at lower cost. A challenge exists to deliver quantized models with minimal loss of accuracy in reasonable time, and in particular to do so without requiring mechanisms incompatible with specialized inference accelerators. Here, we demonstrate a simple, end-to-end quantization-aware training approach that, with an increase in total model training budget of less than 0.1%, outperforms the leading published quantization methods by large margins on several modern benchmarks, with both base and instruct model variants. The approach easily generalizes across different model architectures, can be applied to activations, cache, and weights, and requires the introduction of no additional operations to the model other than the quantization itself. 

---
# ReMeREC: Relation-aware and Multi-entity Referring Expression Comprehension 

**Authors**: Yizhi Hu, Zezhao Tian, Xingqun Qi, Chen Su, Bingkun Yang, Junhui Yin, Muyi Sun, Man Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.16877)  

**Abstract**: Referring Expression Comprehension (REC) aims to localize specified entities or regions in an image based on natural language descriptions. While existing methods handle single-entity localization, they often ignore complex inter-entity relationships in multi-entity scenes, limiting their accuracy and reliability. Additionally, the lack of high-quality datasets with fine-grained, paired image-text-relation annotations hinders further progress. To address this challenge, we first construct a relation-aware, multi-entity REC dataset called ReMeX, which includes detailed relationship and textual annotations. We then propose ReMeREC, a novel framework that jointly leverages visual and textual cues to localize multiple entities while modeling their inter-relations. To address the semantic ambiguity caused by implicit entity boundaries in language, we introduce the Text-adaptive Multi-entity Perceptron (TMP), which dynamically infers both the quantity and span of entities from fine-grained textual cues, producing distinctive representations. Additionally, our Entity Inter-relationship Reasoner (EIR) enhances relational reasoning and global scene understanding. To further improve language comprehension for fine-grained prompts, we also construct a small-scale auxiliary dataset, EntityText, generated using large language models. Experiments on four benchmark datasets show that ReMeREC achieves state-of-the-art performance in multi-entity grounding and relation prediction, outperforming existing approaches by a large margin. 

---
# Pixels, Patterns, but No Poetry: To See The World like Humans 

**Authors**: Hongcheng Gao, Zihao Huang, Lin Xu, Jingyi Tang, Xinhao Li, Yue Liu, Haoyang Li, Taihang Hu, Minhua Lin, Xinlong Yang, Ge Wu, Balong Bi, Hongyu Chen, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16863)  

**Abstract**: Achieving human-like perception and reasoning in Multimodal Large Language Models (MLLMs) remains a central challenge in artificial intelligence. While recent research has primarily focused on enhancing reasoning capabilities in MLLMs, a fundamental question persists: Can Multimodal Large Language Models truly perceive the world as humans do? This paper shifts focus from reasoning to perception. Rather than constructing benchmarks specifically for reasoning, we introduce the Turing Eye Test (TET), a challenging perception-oriented benchmark comprising four diagnostic tasks that evaluate MLLMs' performance on synthetic images that humans process intuitively. Our findings reveal that state-of-the-art MLLMs exhibit catastrophic failures on our perceptual tasks trivial for humans. Both in-context learning and training on language backbone-effective for previous benchmarks-fail to improve performance on our tasks, while fine-tuning the vision tower enables rapid adaptation, suggesting that our benchmark poses challenges for vision tower generalization rather than for the knowledge and reasoning capabilities of the language backbone-a key gap between current MLLMs and human perception. We release a representative subset of TET tasks in this version, and will introduce more diverse tasks and methods to enhance visual generalization in future work. 

---
# Segmentation-free Goodness of Pronunciation 

**Authors**: Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi  

**Link**: [PDF](https://arxiv.org/pdf/2507.16838)  

**Abstract**: Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer aided language learning (CALL) systems. Within MDD, phoneme-level pronunciation assessment is key to helping L2 learners improve their pronunciation. However, most systems are based on a form of goodness of pronunciation (GOP) which requires pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general alignment-free method that takes all possible alignments of the target phoneme into account (GOP-AF). We give a theoretical account of our definition of GOP-AF, an implementation that solves potential numerical issues as well as a proper normalization which makes the method applicable with acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and Speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-AF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the Speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment. 

---
# Evaluating Speech-to-Text x LLM x Text-to-Speech Combinations for AI Interview Systems 

**Authors**: Nima Yazdani, Ali Ansari, Aruj Mahajan, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi  

**Link**: [PDF](https://arxiv.org/pdf/2507.16835)  

**Abstract**: Voice-based conversational AI systems increasingly rely on cascaded architectures combining speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) components. However, systematic evaluation of different component combinations in production settings remains understudied. We present a large-scale empirical comparison of STT x LLM x TTS stacks using data from over 300,000 AI-conducted job interviews. We develop an automated evaluation framework using LLM-as-a-Judge to assess conversational quality, technical accuracy, and skill assessment capabilities. Our analysis of four production configurations reveals that Google STT paired with GPT-4.1 significantly outperforms alternatives in both conversational and technical quality metrics. Surprisingly, we find that objective quality metrics correlate weakly with user satisfaction scores, suggesting that user experience in voice-based AI systems depends on factors beyond technical performance. Our findings provide practical guidance for selecting components in multimodal conversational AI systems and contribute a validated evaluation methodology for voice-based interactions. 

---
# Towards Robust Speech Recognition for Jamaican Patois Music Transcription 

**Authors**: Jordan Madden, Matthew Stone, Dimitri Johnson, Daniel Geddez  

**Link**: [PDF](https://arxiv.org/pdf/2507.16834)  

**Abstract**: Although Jamaican Patois is a widely spoken language, current speech recognition systems perform poorly on Patois music, producing inaccurate captions that limit accessibility and hinder downstream applications. In this work, we take a data-centric approach to this problem by curating more than 40 hours of manually transcribed Patois music. We use this dataset to fine-tune state-of-the-art automatic speech recognition (ASR) models, and use the results to develop scaling laws for the performance of Whisper models on Jamaican Patois audio. We hope that this work will have a positive impact on the accessibility of Jamaican Patois music and the future of Jamaican Patois language modeling. 

---
# A Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval-Augmented Generation in Large Language Models 

**Authors**: Qikai Wei, Huansheng Ning, Chunlong Han, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2507.16826)  

**Abstract**: Retrieval Augmented Generation (RAG) has gradually emerged as a promising paradigm for enhancing the accuracy and factual consistency of content generated by large language models (LLMs). However, existing RAG studies primarily focus on retrieving isolated segments using similarity-based matching methods, while overlooking the intrinsic connections between them. This limitation hampers performance in RAG tasks. To address this, we propose QMKGF, a Query-Aware Multi-Path Knowledge Graph Fusion Approach for Enhancing Retrieval Augmented Generation. First, we design prompt templates and employ general-purpose LLMs to extract entities and relations, thereby generating a knowledge graph (KG) efficiently. Based on the constructed KG, we introduce a multi-path subgraph construction strategy that incorporates one-hop relations, multi-hop relations, and importance-based relations, aiming to improve the semantic relevance between the retrieved documents and the user query. Subsequently, we designed a query-aware attention reward model that scores subgraph triples based on their semantic relevance to the query. Then, we select the highest score subgraph and enrich subgraph with additional triples from other subgraphs that are highly semantically relevant to the query. Finally, the entities, relations, and triples within the updated subgraph are utilised to expand the original query, thereby enhancing its semantic representation and improving the quality of LLMs' generation. We evaluate QMKGF on the SQuAD, IIRC, Culture, HotpotQA, and MuSiQue datasets. On the HotpotQA dataset, our method achieves a ROUGE-1 score of 64.98\%, surpassing the BGE-Rerank approach by 9.72 percentage points (from 55.26\% to 64.98\%). Experimental results demonstrate the effectiveness and superiority of the QMKGF approach. 

---
# Disaster Informatics after the COVID-19 Pandemic: Bibliometric and Topic Analysis based on Large-scale Academic Literature 

**Authors**: Ngan Tran, Haihua Chen, Ana Cleveland, Yuhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16820)  

**Abstract**: This study presents a comprehensive bibliometric and topic analysis of the disaster informatics literature published between January 2020 to September 2022. Leveraging a large-scale corpus and advanced techniques such as pre-trained language models and generative AI, we identify the most active countries, institutions, authors, collaboration networks, emergent topics, patterns among the most significant topics, and shifts in research priorities spurred by the COVID-19 pandemic. Our findings highlight (1) countries that were most impacted by the COVID-19 pandemic were also among the most active, with each country having specific research interests, (2) countries and institutions within the same region or share a common language tend to collaborate, (3) top active authors tend to form close partnerships with one or two key partners, (4) authors typically specialized in one or two specific topics, while institutions had more diverse interests across several topics, and (5) the COVID-19 pandemic has influenced research priorities in disaster informatics, placing greater emphasis on public health. We further demonstrate that the field is converging on multidimensional resilience strategies and cross-sectoral data-sharing collaborations or projects, reflecting a heightened awareness of global vulnerability and interdependency. Collecting and quality assurance strategies, data analytic practices, LLM-based topic extraction and summarization approaches, and result visualization tools can be applied to comparable datasets or solve similar analytic problems. By mapping out the trends in disaster informatics, our analysis offers strategic insights for policymakers, practitioners, and scholars aiming to enhance disaster informatics capacities in an increasingly uncertain and complex risk landscape. 

---
