# HealthBench: Evaluating Large Language Models Towards Improved Human Health 

**Authors**: Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, Karan Singhal  

**Link**: [PDF](https://arxiv.org/pdf/2505.08775)  

**Abstract**: We present HealthBench, an open-source benchmark measuring the performance and safety of large language models in healthcare. HealthBench consists of 5,000 multi-turn conversations between a model and an individual user or healthcare professional. Responses are evaluated using conversation-specific rubrics created by 262 physicians. Unlike previous multiple-choice or short-answer benchmarks, HealthBench enables realistic, open-ended evaluation through 48,562 unique rubric criteria spanning several health contexts (e.g., emergencies, transforming clinical data, global health) and behavioral dimensions (e.g., accuracy, instruction following, communication). HealthBench performance over the last two years reflects steady initial progress (compare GPT-3.5 Turbo's 16% to GPT-4o's 32%) and more rapid recent improvements (o3 scores 60%). Smaller models have especially improved: GPT-4.1 nano outperforms GPT-4o and is 25 times cheaper. We additionally release two HealthBench variations: HealthBench Consensus, which includes 34 particularly important dimensions of model behavior validated via physician consensus, and HealthBench Hard, where the current top score is 32%. We hope that HealthBench grounds progress towards model development and applications that benefit human health. 

---
# Aya Vision: Advancing the Frontier of Multilingual Multimodality 

**Authors**: Saurabh Dash, Yiyang Nan, John Dang, Arash Ahmadian, Shivalika Singh, Madeline Smith, Bharat Venkitesh, Vlad Shmyhlo, Viraat Aryabumi, Walter Beller-Morales, Jeremy Pekmez, Jason Ozuzu, Pierre Richemond, Acyr Locatelli, Nick Frosst, Phil Blunsom, Aidan Gomez, Ivan Zhang, Marzieh Fadaee, Manoj Govindassamy, Sudip Roy, Matthias Gallé, Beyza Ermis, Ahmet Üstün, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2505.08751)  

**Abstract**: Building multimodal language models is fundamentally challenging: it requires aligning vision and language modalities, curating high-quality instruction data, and avoiding the degradation of existing text-only capabilities once vision is introduced. These difficulties are further magnified in the multilingual setting, where the need for multimodal data in different languages exacerbates existing data scarcity, machine translation often distorts meaning, and catastrophic forgetting is more pronounced. To address the aforementioned challenges, we introduce novel techniques spanning both data and modeling. First, we develop a synthetic annotation framework that curates high-quality, diverse multilingual multimodal instruction data, enabling Aya Vision models to produce natural, human-preferred responses to multimodal inputs across many languages. Complementing this, we propose a cross-modal model merging technique that mitigates catastrophic forgetting, effectively preserving text-only capabilities while simultaneously enhancing multimodal generative performance. Aya-Vision-8B achieves best-in-class performance compared to strong multimodal models such as Qwen-2.5-VL-7B, Pixtral-12B, and even much larger Llama-3.2-90B-Vision. We further scale this approach with Aya-Vision-32B, which outperforms models more than twice its size, such as Molmo-72B and LLaMA-3.2-90B-Vision. Our work advances multilingual progress on the multi-modal frontier, and provides insights into techniques that effectively bend the need for compute while delivering extremely high performance. 

---
# AC-Reason: Towards Theory-Guided Actual Causality Reasoning with Large Language Models 

**Authors**: Yanxi Zhang, Xin Cong, Zhong Zhang, Xiao Liu, Dongyan Zhao, Yesai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08750)  

**Abstract**: Actual causality (AC), a fundamental aspect of causal reasoning (CR), is responsible for attribution and responsibility assignment in real-world scenarios. However, existing LLM-based methods lack grounding in formal AC theory, resulting in limited interpretability. Therefore, we propose AC-Reason, a semi-formal reasoning framework that identifies causally relevant events within an AC scenario, infers the values of their formal causal factors (e.g., sufficiency, necessity, and normality), and answers AC queries via a theory-guided algorithm with explanations. While AC-Reason does not explicitly construct a causal graph, it operates over variables in the underlying causal structure to support principled reasoning. To enable comprehensive evaluation, we introduce AC-Bench, a new benchmark built upon and substantially extending Big-Bench Hard Causal Judgment (BBH-CJ). AC-Bench comprises ~1K carefully annotated samples, each with detailed reasoning steps and focuses solely on actual causation. The case study shows that synthesized samples in AC-Bench present greater challenges for LLMs. Extensive experiments on BBH-CJ and AC-Bench show that AC-Reason consistently improves LLM performance over baselines. On BBH-CJ, all tested LLMs surpass the average human rater accuracy of 69.60%, with GPT-4 + AC-Reason achieving 75.04%. On AC-Bench, GPT-4 + AC-Reason again achieves the highest accuracy of 71.82%. AC-Bench further enables fine-grained analysis of reasoning faithfulness, revealing that only Qwen-2.5-72B-Instruct, Claude-3.5-Sonnet, and GPT-4o exhibit faithful reasoning, whereas GPT-4 tends to exploit shortcuts. Finally, our ablation study proves that integrating AC theory into LLMs is highly effective, with the proposed algorithm contributing the most significant performance gains. 

---
# Probability Consistency in Large Language Models: Theoretical Foundations Meet Empirical Discrepancies 

**Authors**: Xiaoliang Luo, Xinyi Xu, Michael Ramscar, Bradley C. Love  

**Link**: [PDF](https://arxiv.org/pdf/2505.08739)  

**Abstract**: Can autoregressive large language models (LLMs) learn consistent probability distributions when trained on sequences in different token orders? We prove formally that for any well-defined probability distribution, sequence perplexity is invariant under any factorization, including forward, backward, or arbitrary permutations. This result establishes a rigorous theoretical foundation for studying how LLMs learn from data and defines principled protocols for empirical evaluation. Applying these protocols, we show that prior studies examining ordering effects suffer from critical methodological flaws. We retrain GPT-2 models across forward, backward, and arbitrary permuted orders on scientific text. We find systematic deviations from theoretical invariance across all orderings with arbitrary permutations strongly deviating from both forward and backward models, which largely (but not completely) agreed with one another. Deviations were traceable to differences in self-attention, reflecting positional and locality biases in processing. Our theoretical and empirical results provide novel avenues for understanding positional biases in LLMs and suggest methods for detecting when LLMs' probability distributions are inconsistent and therefore untrustworthy. 

---
# NurValues: Real-World Nursing Values Evaluation for Large Language Models in Clinical Context 

**Authors**: Ben Yao, Qiuchi Li, Yazhou Zhang, Siyu Yang, Bohan Zhang, Prayag Tiwari, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.08734)  

**Abstract**: This work introduces the first benchmark for nursing value alignment, consisting of five core value dimensions distilled from international nursing codes: Altruism, Human Dignity, Integrity, Justice, and Professionalism. The benchmark comprises 1,100 real-world nursing behavior instances collected through a five-month longitudinal field study across three hospitals of varying tiers. These instances are annotated by five clinical nurses and then augmented with LLM-generated counterfactuals with reversed ethic polarity. Each original case is paired with a value-aligned and a value-violating version, resulting in 2,200 labeled instances that constitute the Easy-Level dataset. To increase adversarial complexity, each instance is further transformed into a dialogue-based format that embeds contextual cues and subtle misleading signals, yielding a Hard-Level dataset. We evaluate 23 state-of-the-art (SoTA) LLMs on their alignment with nursing values. Our findings reveal three key insights: (1) DeepSeek-V3 achieves the highest performance on the Easy-Level dataset (94.55), where Claude 3.5 Sonnet outperforms other models on the Hard-Level dataset (89.43), significantly surpassing the medical LLMs; (2) Justice is consistently the most difficult nursing value dimension to evaluate; and (3) in-context learning significantly improves alignment. This work aims to provide a foundation for value-sensitive LLMs development in clinical settings. The dataset and the code are available at this https URL. 

---
# Adaptive Schema-aware Event Extraction with Retrieval-Augmented Generation 

**Authors**: Sheng Liang, Hang Lv, Zhihao Wen, Yaxiong Wu, Yongyue Zhang, Hao Wang, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08690)  

**Abstract**: Event extraction (EE) is a fundamental task in natural language processing (NLP) that involves identifying and extracting event information from unstructured text. Effective EE in real-world scenarios requires two key steps: selecting appropriate schemas from hundreds of candidates and executing the extraction process. Existing research exhibits two critical gaps: (1) the rigid schema fixation in existing pipeline systems, and (2) the absence of benchmarks for evaluating joint schema matching and extraction. Although large language models (LLMs) offer potential solutions, their schema hallucination tendencies and context window limitations pose challenges for practical deployment. In response, we propose Adaptive Schema-aware Event Extraction (ASEE), a novel paradigm combining schema paraphrasing with schema retrieval-augmented generation. ASEE adeptly retrieves paraphrased schemas and accurately generates targeted structures. To facilitate rigorous evaluation, we construct the Multi-Dimensional Schema-aware Event Extraction (MD-SEE) benchmark, which systematically consolidates 12 datasets across diverse domains, complexity levels, and language settings. Extensive evaluations on MD-SEE show that our proposed ASEE demonstrates strong adaptability across various scenarios, significantly improving the accuracy of event extraction. 

---
# Revealing economic facts: LLMs know more than they say 

**Authors**: Marcus Buckmann, Quynh Anh Nguyen, Edward Hill  

**Link**: [PDF](https://arxiv.org/pdf/2505.08662)  

**Abstract**: We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks. 

---
# Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing 

**Authors**: Chen Wu, Yin Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.08651)  

**Abstract**: We present MegaBeam-Mistral-7B, a language model that supports 512K-token context length. Our work addresses practical limitations in long-context training, supporting real-world tasks such as compliance monitoring and verification. Evaluated on three long-context benchmarks, our 7B-parameter model demonstrates superior in-context learning performance on HELMET and robust retrieval and tracing capability on RULER. It is currently the only open model to achieve competitive long-range reasoning on BABILong at 512K context length without RAG or targeted fine-tuning. Released as fully open source under the Apache 2.0 license, the model has been downloaded over 100,000 times on Hugging Face. Model available at: this https URL 

---
# Automatic Task Detection and Heterogeneous LLM Speculative Decoding 

**Authors**: Danying Ge, Jianhua Gao, Qizhi Jiang, Yifei Feng, Weixing Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.08600)  

**Abstract**: Speculative decoding, which combines a draft model with a target model, has emerged as an effective approach to accelerate large language model (LLM) inference. However, existing methods often face a trade-off between the acceptance rate and decoding speed in downstream tasks due to the limited capacity of the draft model, making it difficult to ensure efficiency across diverse tasks. To address this problem, we propose a speculative decoding algorithm tailored for downstream task optimization. It includes an automatic task partitioning and assigning method, which automatically categorizes downstream tasks into different sub-tasks and assigns them to a set of heterogeneous draft models. Each draft model is aligned with the target model using task-specific data, thereby enhancing the consistency of inference results. In addition, our proposed method incorporates an online lightweight prompt classifier to dynamically route prompts to the appropriate draft model. Experimental results demonstrate that the proposed method improves draft accuracy by 6% to 50% over vanilla speculative decoding, while achieving a speedup of 1.10x to 2.64x in LLM inference. 

---
# Enhancing Thyroid Cytology Diagnosis with RAG-Optimized LLMs and Pa-thology Foundation Models 

**Authors**: Hussien Al-Asi, Jordan P Reynolds, Shweta Agarwal, Bryan J Dangott, Aziza Nassar, Zeynettin Akkus  

**Link**: [PDF](https://arxiv.org/pdf/2505.08590)  

**Abstract**: Advancements in artificial intelligence (AI) are transforming pathology by integrat-ing large language models (LLMs) with retrieval-augmented generation (RAG) and domain-specific foundation models. This study explores the application of RAG-enhanced LLMs coupled with pathology foundation models for thyroid cytology diagnosis, addressing challenges in cytological interpretation, standardization, and diagnostic accuracy. By leveraging a curated knowledge base, RAG facilitates dy-namic retrieval of relevant case studies, diagnostic criteria, and expert interpreta-tion, improving the contextual understanding of LLMs. Meanwhile, pathology foun-dation models, trained on high-resolution pathology images, refine feature extrac-tion and classification capabilities. The fusion of these AI-driven approaches en-hances diagnostic consistency, reduces variability, and supports pathologists in dis-tinguishing benign from malignant thyroid lesions. Our results demonstrate that integrating RAG with pathology-specific LLMs significantly improves diagnostic efficiency and interpretability, paving the way for AI-assisted thyroid cytopathology, with foundation model UNI achieving AUC 0.73-0.93 for correct prediction of surgi-cal pathology diagnosis from thyroid cytology samples. 

---
# Small but Significant: On the Promise of Small Language Models for Accessible AIED 

**Authors**: Yumou Wei, Paulo Carvalho, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2505.08588)  

**Abstract**: GPT has become nearly synonymous with large language models (LLMs), an increasingly popular term in AIED proceedings. A simple keyword-based search reveals that 61% of the 76 long and short papers presented at AIED 2024 describe novel solutions using LLMs to address some of the long-standing challenges in education, and 43% specifically mention GPT. Although LLMs pioneered by GPT create exciting opportunities to strengthen the impact of AI on education, we argue that the field's predominant focus on GPT and other resource-intensive LLMs (with more than 10B parameters) risks neglecting the potential impact that small language models (SLMs) can make in providing resource-constrained institutions with equitable and affordable access to high-quality AI tools. Supported by positive results on knowledge component (KC) discovery, a critical challenge in AIED, we demonstrate that SLMs such as Phi-2 can produce an effective solution without elaborate prompting strategies. Hence, we call for more attention to developing SLM-based AIED approaches. 

---
# Are We Paying Attention to Her? Investigating Gender Disambiguation and Attention in Machine Translation 

**Authors**: Chiara Manna, Afra Alishahi, Frédéric Blain, Eva Vanmassenhove  

**Link**: [PDF](https://arxiv.org/pdf/2505.08546)  

**Abstract**: While gender bias in modern Neural Machine Translation (NMT) systems has received much attention, traditional evaluation metrics do not to fully capture the extent to which these systems integrate contextual gender cues. We propose a novel evaluation metric called Minimal Pair Accuracy (MPA), which measures the reliance of models on gender cues for gender disambiguation. MPA is designed to go beyond surface-level gender accuracy metrics by focusing on whether models adapt to gender cues in minimal pairs -- sentence pairs that differ solely in the gendered pronoun, namely the explicit indicator of the target's entity gender in the source language (EN). We evaluate a number of NMT models on the English-Italian (EN--IT) language pair using this metric, we show that they ignore available gender cues in most cases in favor of (statistical) stereotypical gender interpretation. We further show that in anti-stereotypical cases, these models tend to more consistently take masculine gender cues into account while ignoring the feminine cues. Furthermore, we analyze the attention head weights in the encoder component and show that while all models encode gender information to some extent, masculine cues elicit a more diffused response compared to the more concentrated and specialized responses to feminine gender cues. 

---
# Reassessing Graph Linearization for Sequence-to-sequence AMR Parsing: On the Advantages and Limitations of Triple-Based Encoding 

**Authors**: Jeongwoo Kang, Maximin Coavoux, Cédric Lopez, Didier Schwab  

**Link**: [PDF](https://arxiv.org/pdf/2505.08504)  

**Abstract**: Sequence-to-sequence models are widely used to train Abstract Meaning Representation (Banarescu et al., 2013, AMR) parsers. To train such models, AMR graphs have to be linearized into a one-line text format. While Penman encoding is typically used for this purpose, we argue that it has limitations: (1) for deep graphs, some closely related nodes are located far apart in the linearized text (2) Penman's tree-based encoding necessitates inverse roles to handle node re-entrancy, doubling the number of relation types to predict. To address these issues, we propose a triple-based linearization method and compare its efficiency with Penman linearization. Although triples are well suited to represent a graph, our results suggest room for improvement in triple encoding to better compete with Penman's concise and explicit representation of a nested graph structure. 

---
# LCES: Zero-shot Automated Essay Scoring via Pairwise Comparisons Using Large Language Models 

**Authors**: Takumi Shibata, Yuichi Miyamura  

**Link**: [PDF](https://arxiv.org/pdf/2505.08498)  

**Abstract**: Recent advances in large language models (LLMs) have enabled zero-shot automated essay scoring (AES), providing a promising way to reduce the cost and effort of essay scoring in comparison with manual grading. However, most existing zero-shot approaches rely on LLMs to directly generate absolute scores, which often diverge from human evaluations owing to model biases and inconsistent scoring. To address these limitations, we propose LLM-based Comparative Essay Scoring (LCES), a method that formulates AES as a pairwise comparison task. Specifically, we instruct LLMs to judge which of two essays is better, collect many such comparisons, and convert them into continuous scores. Considering that the number of possible comparisons grows quadratically with the number of essays, we improve scalability by employing RankNet to efficiently transform LLM preferences into scalar scores. Experiments using AES benchmark datasets show that LCES outperforms conventional zero-shot methods in accuracy while maintaining computational efficiency. Moreover, LCES is robust across different LLM backbones, highlighting its applicability to real-world zero-shot AES. 

---
# Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning? 

**Authors**: Md Tahmid Rahman Laskar, Mohammed Saidul Islam, Ridwan Mahbub, Ahmed Masry, Mizanur Rahman, Amran Bhuiyan, Mir Tafseer Nayeem, Shafiq Joty, Enamul Hoque, Jimmy Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08468)  

**Abstract**: Charts are ubiquitous as they help people understand and reason with data. Recently, various downstream tasks, such as chart question answering, chart2text, and fact-checking, have emerged. Large Vision-Language Models (LVLMs) show promise in tackling these tasks, but their evaluation is costly and time-consuming, limiting real-world deployment. While using LVLMs as judges to assess the chart comprehension capabilities of other LVLMs could streamline evaluation processes, challenges like proprietary datasets, restricted access to powerful models, and evaluation costs hinder their adoption in industrial settings. To this end, we present a comprehensive evaluation of 13 open-source LVLMs as judges for diverse chart comprehension and reasoning tasks. We design both pairwise and pointwise evaluation tasks covering criteria like factual correctness, informativeness, and relevancy. Additionally, we analyze LVLM judges based on format adherence, positional consistency, length bias, and instruction-following. We focus on cost-effective LVLMs (<10B parameters) suitable for both research and commercial use, following a standardized evaluation protocol and rubric to measure the LVLM judge's accuracy. Experimental results reveal notable variability: while some open LVLM judges achieve GPT-4-level evaluation performance (about 80% agreement with GPT-4 judgments), others struggle (below ~10% agreement). Our findings highlight that state-of-the-art open-source LVLMs can serve as cost-effective automatic evaluators for chart-related tasks, though biases such as positional preference and length bias persist. 

---
# Large Language Models Meet Stance Detection: A Survey of Tasks, Methods, Applications, Challenges and Future Directions 

**Authors**: Lata Pangtey, Anukriti Bhatnagar, Shubhi Bansal, Shahid Shafi Dar, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08464)  

**Abstract**: Stance detection is essential for understanding subjective content across various platforms such as social media, news articles, and online reviews. Recent advances in Large Language Models (LLMs) have revolutionized stance detection by introducing novel capabilities in contextual understanding, cross-domain generalization, and multimodal analysis. Despite these progressions, existing surveys often lack comprehensive coverage of approaches that specifically leverage LLMs for stance detection. To bridge this critical gap, our review article conducts a systematic analysis of stance detection, comprehensively examining recent advancements of LLMs transforming the field, including foundational concepts, methodologies, datasets, applications, and emerging challenges. We present a novel taxonomy for LLM-based stance detection approaches, structured along three key dimensions: 1) learning methods, including supervised, unsupervised, few-shot, and zero-shot; 2) data modalities, such as unimodal, multimodal, and hybrid; and 3) target relationships, encompassing in-target, cross-target, and multi-target scenarios. Furthermore, we discuss the evaluation techniques and analyze benchmark datasets and performance trends, highlighting the strengths and limitations of different architectures. Key applications in misinformation detection, political analysis, public health monitoring, and social media moderation are discussed. Finally, we identify critical challenges such as implicit stance expression, cultural biases, and computational constraints, while outlining promising future directions, including explainable stance reasoning, low-resource adaptation, and real-time deployment frameworks. Our survey highlights emerging trends, open challenges, and future directions to guide researchers and practitioners in developing next-generation stance detection systems powered by large language models. 

---
# RepCali: High Efficient Fine-tuning Via Representation Calibration in Latent Space for Pre-trained Language Models 

**Authors**: Fujun Zhang, XiangDong Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.08463)  

**Abstract**: Fine-tuning pre-trained language models (PLMs) has become a dominant paradigm in applying PLMs to downstream tasks. However, with limited fine-tuning, PLMs still struggle with the discrepancies between the representation obtained from the PLMs' encoder and the optimal input to the PLMs' decoder. This paper tackles this challenge by learning to calibrate the representation of PLMs in the latent space. In the proposed representation calibration method (RepCali), we integrate a specific calibration block to the latent space after the encoder and use the calibrated output as the decoder input. The merits of the proposed RepCali include its universality to all PLMs with encoder-decoder architectures, its plug-and-play nature, and ease of implementation. Extensive experiments on 25 PLM-based models across 8 tasks (including both English and Chinese datasets) demonstrate that the proposed RepCali offers desirable enhancements to PLMs (including LLMs) and significantly improves the performance of downstream tasks. Comparison experiments across 4 benchmark tasks indicate that RepCali is superior to the representative fine-tuning baselines. 

---
# IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation 

**Authors**: Kazuki Hayashi, Hidetaka Kamigaito, Shinya Kouda, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2505.08450)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability. 

---
# A document processing pipeline for the construction of a dataset for topic modeling based on the judgments of the Italian Supreme Court 

**Authors**: Matteo Marulli, Glauco Panattoni, Marco Bertini  

**Link**: [PDF](https://arxiv.org/pdf/2505.08439)  

**Abstract**: Topic modeling in Italian legal research is hindered by the lack of public datasets, limiting the analysis of legal themes in Supreme Court judgments. To address this, we developed a document processing pipeline that produces an anonymized dataset optimized for topic modeling.
The pipeline integrates document layout analysis (YOLOv8x), optical character recognition, and text anonymization. The DLA module achieved a mAP@50 of 0.964 and a mAP@50-95 of 0.800. The OCR detector reached a mAP@50-95 of 0.9022, and the text recognizer (TrOCR) obtained a character error rate of 0.0047 and a word error rate of 0.0248. Compared to OCR-only methods, our dataset improved topic modeling with a diversity score of 0.6198 and a coherence score of 0.6638.
We applied BERTopic to extract topics and used large language models to generate labels and summaries. Outputs were evaluated against domain expert interpretations. Claude Sonnet 3.7 achieved a BERTScore F1 of 0.8119 for labeling and 0.9130 for summarization. 

---
# Hakim: Farsi Text Embedding Model 

**Authors**: Mehran Sarmadi, Morteza Alikhani, Erfan Zinvandi, Zahra Pourbahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.08435)  

**Abstract**: Recent advancements in text embedding have significantly improved natural language understanding across many languages, yet Persian remains notably underrepresented in large-scale embedding research. In this paper, we present Hakim, a novel state-of-the-art Persian text embedding model that achieves a 8.5% performance improvement over existing approaches on the FaMTEB benchmark, outperforming all previously developed Persian language models. As part of this work, we introduce three new datasets - Corpesia, Pairsia-sup, and Pairsia-unsup - to support supervised and unsupervised training scenarios. Additionally, Hakim is designed for applications in chatbots and retrieval-augmented generation (RAG) systems, particularly addressing retrieval tasks that require incorporating message history within these systems. We also propose a new baseline model built on the BERT architecture. Our language model consistently achieves higher accuracy across various Persian NLP tasks, while the RetroMAE-based model proves particularly effective for textual information retrieval applications. Together, these contributions establish a new foundation for advancing Persian language understanding. 

---
# TUMS: Enhancing Tool-use Abilities of LLMs with Multi-structure Handlers 

**Authors**: Aiyao He, Sijia Cui, Shuai Xu, Yanna Wang, Bo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08402)  

**Abstract**: Recently, large language models(LLMs) have played an increasingly important role in solving a wide range of NLP tasks, leveraging their capabilities of natural language understanding and generating. Integration with external tools further enhances LLMs' effectiveness, providing more precise, timely, and specialized responses. However, LLMs still encounter difficulties with non-executable actions and improper actions, which are primarily attributed to incorrect parameters. The process of generating parameters by LLMs is confined to the tool level, employing the coarse-grained strategy without considering the different difficulties of various tools. To address this issue, we propose TUMS, a novel framework designed to enhance the tool-use capabilities of LLMs by transforming tool-level processing into parameter-level processing. Specifically, our framework consists of four key components: (1) an intent recognizer that identifies the user's intent to help LLMs better understand the task; (2) a task decomposer that breaks down complex tasks into simpler subtasks, each involving a tool call; (3) a subtask processor equipped with multi-structure handlers to generate accurate parameters; and (4) an executor. Our empirical studies have evidenced the effectiveness and efficiency of the TUMS framework with an average of 19.6\% and 50.6\% improvement separately on easy and hard benchmarks of ToolQA, meanwhile, we demonstrated the key contribution of each part with ablation experiments, offering more insights and stimulating future research on Tool-augmented LLMs. 

---
# Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping 

**Authors**: Ren Zhuang, Ben Wang, Shuifa Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.08392)  

**Abstract**: Large Language Models leverage Chain-of-Thought (CoT) prompting for complex tasks, but their reasoning traces are often excessively verbose and inefficient, leading to significant computational costs and latency. Current CoT compression techniques typically rely on generic importance metrics and static compression rates, which may inadvertently remove functionally critical tokens or fail to adapt to varying reasoning complexity. To overcome these limitations, we propose Adaptive GoGI-Skip, a novel framework learning dynamic CoT compression via supervised fine-tuning. This approach introduces two synergistic innovations: (1) Goal-Gradient Importance (GoGI), a novel metric accurately identifying functionally relevant tokens by measuring the gradient influence of their intermediate representations on the final answer loss, and (2) Adaptive Dynamic Skipping (ADS), a mechanism dynamically regulating the compression rate based on runtime model uncertainty while ensuring local coherence through an adaptive N-token constraint. To our knowledge, this is the first work unifying a goal-oriented, gradient-based importance metric with dynamic, uncertainty-aware skipping for CoT compression. Trained on compressed MATH data, Adaptive GoGI-Skip demonstrates strong cross-domain generalization across diverse reasoning benchmarks including AIME, GPQA, and GSM8K. It achieves substantial efficiency gains - reducing CoT token counts by over 45% on average and delivering 1.6-2.0 times inference speedups - while maintaining high reasoning accuracy. Notably, it significantly outperforms existing baselines by preserving accuracy even at high effective compression rates, advancing the state of the art in the CoT reasoning efficiency-accuracy trade-off. 

---
# Towards Contamination Resistant Benchmarks 

**Authors**: Rahmatullah Musawi, Sheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08389)  

**Abstract**: The rapid development of large language models (LLMs) has transformed the landscape of natural language processing. Evaluating LLMs properly is crucial for understanding their potential and addressing concerns such as safety. However, LLM evaluation is confronted by various factors, among which contamination stands out as a key issue that undermines the reliability of evaluations. In this work, we introduce the concept of contamination resistance to address this challenge. We propose a benchmark based on Caesar ciphers (e.g., "ab" to "bc" when the shift is 1), which, despite its simplicity, is an excellent example of a contamination resistant benchmark. We test this benchmark on widely used LLMs under various settings, and we find that these models struggle with this benchmark when contamination is controlled. Our findings reveal issues in current LLMs and raise important questions regarding their true capabilities. Our work contributes to the development of contamination resistant benchmarks, enabling more rigorous LLM evaluation and offering insights into the true capabilities and limitations of LLMs. 

---
# Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring 

**Authors**: Mina Almasi, Ross Deans Kristensen-McLachlan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08351)  

**Abstract**: This paper investigates the potentials of Large Language Models (LLMs) as adaptive tutors in the context of second-language learning. In particular, we evaluate whether system prompting can reliably constrain LLMs to generate only text appropriate to the student's competence level. We simulate full teacher-student dialogues in Spanish using instruction-tuned, open-source LLMs ranging in size from 7B to 12B parameters. Dialogues are generated by having an LLM alternate between tutor and student roles with separate chat histories. The output from the tutor model is then used to evaluate the effectiveness of CEFR-based prompting to control text difficulty across three proficiency levels (A1, B1, C1). Our findings suggest that while system prompting can be used to constrain model outputs, prompting alone is too brittle for sustained, long-term interactional contexts - a phenomenon we term alignment drift. Our results provide insights into the feasibility of LLMs for personalized, proficiency-aligned adaptive tutors and provide a scalable method for low-cost evaluation of model performance without human participants. 

---
# On the Geometry of Semantics in Next-token Prediction 

**Authors**: Yize Zhao, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2505.08348)  

**Abstract**: Modern language models demonstrate a remarkable ability to capture linguistic meaning despite being trained solely through next-token prediction (NTP). We investigate how this conceptually simple training objective leads models to extract and encode latent semantic and grammatical concepts. Our analysis reveals that NTP optimization implicitly guides models to encode concepts via singular value decomposition (SVD) factors of a centered data-sparsity matrix that captures next-word co-occurrence patterns. While the model never explicitly constructs this matrix, learned word and context embeddings effectively factor it to capture linguistic structure. We find that the most important SVD factors are learned first during training, motivating the use of spectral clustering of embeddings to identify human-interpretable semantics, including both classical k-means and a new orthant-based method directly motivated by our interpretation of concepts. Overall, our work bridges distributional semantics, neural collapse geometry, and neural network training dynamics, providing insights into how NTP's implicit biases shape the emergence of meaning representations in language models. 

---
# AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale 

**Authors**: Yunjie Ji, Xiaoyu Tian, Sitong Zhao, Haotian Wang, Shuaiting Chen, Yiping Peng, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08311)  

**Abstract**: We present AM-Thinking-v1, a 32B dense language model that advances the frontier of reasoning, embodying the collaborative spirit of open-source innovation. Outperforming DeepSeek-R1 and rivaling leading Mixture-of-Experts (MoE) models like Qwen3-235B-A22B and Seed1.5-Thinking, AM-Thinking-v1 achieves impressive scores of 85.3 on AIME 2024, 74.4 on AIME 2025, and 70.3 on LiveCodeBench, showcasing state-of-the-art mathematical and coding capabilities among open-source models of similar scale.
Built entirely from the open-source Qwen2.5-32B base model and publicly available queries, AM-Thinking-v1 leverages a meticulously crafted post-training pipeline - combining supervised fine-tuning and reinforcement learning - to deliver exceptional reasoning capabilities. This work demonstrates that the open-source community can achieve high performance at the 32B scale, a practical sweet spot for deployment and fine-tuning. By striking a balance between top-tier performance and real-world usability, we hope AM-Thinking-v1 inspires further collaborative efforts to harness mid-scale models, pushing reasoning boundaries while keeping accessibility at the core of innovation. We have open-sourced our model on \href{this https URL}{Hugging Face}. 

---
# Evaluating the Effectiveness of Black-Box Prompt Optimization as the Scale of LLMs Continues to Grow 

**Authors**: Ziyu Zhou, Yihang Wu, Jingyuan Yang, Zhan Xiao, Rongjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08303)  

**Abstract**: Black-Box prompt optimization methods have emerged as a promising strategy for refining input prompts to better align large language models (LLMs), thereby enhancing their task performance. Although these methods have demonstrated encouraging results, most studies and experiments have primarily focused on smaller-scale models (e.g., 7B, 14B) or earlier versions (e.g., GPT-3.5) of LLMs. As the scale of LLMs continues to increase, such as with DeepSeek V3 (671B), it remains an open question whether these black-box optimization techniques will continue to yield significant performance improvements for models of such scale. In response to this, we select three well-known black-box optimization methods and evaluate them on large-scale LLMs (DeepSeek V3 and Gemini 2.0 Flash) across four NLU and NLG datasets. The results show that these black-box prompt optimization methods offer only limited improvements on these large-scale LLMs. Furthermore, we hypothesize that the scale of the model is the primary factor contributing to the limited benefits observed. To explore this hypothesis, we conducted experiments on LLMs of varying sizes (Qwen 2.5 series, ranging from 7B to 72B) and observed an inverse scaling law, wherein the effectiveness of black-box optimization methods diminished as the model size increased. 

---
# Enhancing Cache-Augmented Generation (CAG) with Adaptive Contextual Compression for Scalable Knowledge Integration 

**Authors**: Rishabh Agrawal, Himanshu Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08261)  

**Abstract**: The rapid progress in large language models (LLMs) has paved the way for novel approaches in knowledge-intensive tasks. Among these, Cache-Augmented Generation (CAG) has emerged as a promising alternative to Retrieval-Augmented Generation (RAG). CAG minimizes retrieval latency and simplifies system design by preloading knowledge into the model's context. However, challenges persist in scaling CAG to accommodate large and dynamic knowledge bases effectively. This paper introduces Adaptive Contextual Compression (ACC), an innovative technique designed to dynamically compress and manage context inputs, enabling efficient utilization of the extended memory capabilities of modern LLMs. To further address the limitations of standalone CAG, we propose a Hybrid CAG-RAG Framework, which integrates selective retrieval to augment preloaded contexts in scenarios requiring additional information. Comprehensive evaluations on diverse datasets highlight the proposed methods' ability to enhance scalability, optimize efficiency, and improve multi-hop reasoning performance, offering practical solutions for real-world knowledge integration challenges. 

---
# Large Language Model Psychometrics: A Systematic Review of Evaluation, Validation, and Enhancement 

**Authors**: Haoran Ye, Jing Jin, Yuhang Xie, Xin Zhang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.08245)  

**Abstract**: The rapid advancement of large language models (LLMs) has outpaced traditional evaluation methodologies. It presents novel challenges, such as measuring human-like psychological constructs, navigating beyond static and task-specific benchmarks, and establishing human-centered evaluation. These challenges intersect with Psychometrics, the science of quantifying the intangible aspects of human psychology, such as personality, values, and intelligence. This survey introduces and synthesizes an emerging interdisciplinary field of LLM Psychometrics, which leverages psychometric instruments, theories, and principles to evaluate, understand, and enhance LLMs. We systematically explore the role of Psychometrics in shaping benchmarking principles, broadening evaluation scopes, refining methodologies, validating results, and advancing LLM capabilities. This paper integrates diverse perspectives to provide a structured framework for researchers across disciplines, enabling a more comprehensive understanding of this nascent field. Ultimately, we aim to provide actionable insights for developing future evaluation paradigms that align with human-level AI and promote the advancement of human-centered AI systems for societal benefit. A curated repository of LLM psychometric resources is available at this https URL. 

---
# A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs 

**Authors**: Artem Shelmanov, Ekaterina Fadeeva, Akim Tsvigun, Ivan Tsvigun, Zhuohan Xie, Igor Kiselev, Nico Daheim, Caiqi Zhang, Artem Vazhentsev, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2505.08200)  

**Abstract**: Large Language Models (LLMs) have the tendency to hallucinate, i.e., to sporadically generate false or fabricated information. This presents a major challenge, as hallucinations often appear highly convincing and users generally lack the tools to detect them. Uncertainty quantification (UQ) provides a framework for assessing the reliability of model outputs, aiding in the identification of potential hallucinations. In this work, we introduce pre-trained UQ heads: supervised auxiliary modules for LLMs that substantially enhance their ability to capture uncertainty compared to unsupervised UQ methods. Their strong performance stems from the powerful Transformer architecture in their design and informative features derived from LLM attention maps. Experimental evaluation shows that these heads are highly robust and achieve state-of-the-art performance in claim-level hallucination detection across both in-domain and out-of-domain prompts. Moreover, these modules demonstrate strong generalization to languages they were not explicitly trained on. We pre-train a collection of UQ heads for popular LLM series, including Mistral, Llama, and Gemma 2. We publicly release both the code and the pre-trained heads. 

---
# Exploiting Text Semantics for Few and Zero Shot Node Classification on Text-attributed Graph 

**Authors**: Yuxiang Wang, Xiao Yan, Shiyu Jin, Quanqing Xu, Chuang Hu, Yuanyuan Zhu, Bo Du, Jia Wu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08168)  

**Abstract**: Text-attributed graph (TAG) provides a text description for each graph node, and few- and zero-shot node classification on TAGs have many applications in fields such as academia and social networks. Existing work utilizes various graph-based augmentation techniques to train the node and text embeddings, while text-based augmentations are largely unexplored. In this paper, we propose Text Semantics Augmentation (TSA) to improve accuracy by introducing more text semantic supervision signals. Specifically, we design two augmentation techniques, i.e., positive semantics matching and negative semantics contrast, to provide more reference texts for each graph node or text description. Positive semantic matching retrieves texts with similar embeddings to match with a graph node. Negative semantic contrast adds a negative prompt to construct a text description with the opposite semantics, which is contrasted with the original node and text. We evaluate TSA on 5 datasets and compare with 13 state-of-the-art baselines. The results show that TSA consistently outperforms all baselines, and its accuracy improvements over the best-performing baseline are usually over 5%. 

---
# Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage 

**Authors**: Ruilin Liu, Zhixiao Zhao, Jieqiong Li, Chang Liu, Dongbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08167)  

**Abstract**: The rapid development of large language models (LLMs) has provided significant support and opportunities for the advancement of domain-specific LLMs. However, fine-tuning these large models using Intangible Cultural Heritage (ICH) data inevitably faces challenges such as bias, incorrect knowledge inheritance, and catastrophic forgetting. To address these issues, we propose a novel training method that integrates a bidirectional chains of thought and a reward mechanism. This method is built upon ICH-Qwen, a large language model specifically designed for the field of intangible cultural heritage. The proposed method enables the model to not only perform forward reasoning but also enhances the accuracy of the generated answers by utilizing reverse questioning and reverse reasoning to activate the model's latent knowledge. Additionally, a reward mechanism is introduced during training to optimize the decision-making process. This mechanism improves the quality of the model's outputs through structural and content evaluations with different weighting schemes. We conduct comparative experiments on ICH-Qwen, with results demonstrating that our method outperforms 0-shot, step-by-step reasoning, knowledge distillation, and question augmentation methods in terms of accuracy, Bleu-4, and Rouge-L scores on the question-answering task. Furthermore, the paper highlights the effectiveness of combining the bidirectional chains of thought and reward mechanism through ablation experiments. In addition, a series of generalizability experiments are conducted, with results showing that the proposed method yields improvements on various domain-specific datasets and advanced models in areas such as Finance, Wikidata, and StrategyQA. This demonstrates that the method is adaptable to multiple domains and provides a valuable approach for model training in future applications across diverse fields. 

---
# ALOHA: Empowering Multilingual Agent for University Orientation with Hierarchical Retrieval 

**Authors**: Mingxu Tao, Bowen Tang, Mingxuan Ma, Yining Zhang, Hourun Li, Feifan Wen, Hao Ma, Jia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08130)  

**Abstract**: The rise of Large Language Models~(LLMs) revolutionizes information retrieval, allowing users to obtain required answers through complex instructions within conversations. However, publicly available services remain inadequate in addressing the needs of faculty and students to search campus-specific information. It is primarily due to the LLM's lack of domain-specific knowledge and the limitation of search engines in supporting multilingual and timely scenarios. To tackle these challenges, we introduce ALOHA, a multilingual agent enhanced by hierarchical retrieval for university orientation. We also integrate external APIs into the front-end interface to provide interactive service. The human evaluation and case study show our proposed system has strong capabilities to yield correct, timely, and user-friendly responses to the queries in multiple languages, surpassing commercial chatbots and search engines. The system has been deployed and has provided service for more than 12,000 people. 

---
# Putting It All into Context: Simplifying Agents with LCLMs 

**Authors**: Mingjian Jiang, Yangjun Ruan, Luis Lastras, Pavan Kapanipathi, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.08120)  

**Abstract**: Recent advances in language model (LM) agents have demonstrated significant potential for automating complex real-world tasks. To make progress on these difficult tasks, LM agent architectures have become increasingly complex, often incorporating multi-step retrieval tools, multiple agents, and scaffolding adapted to the underlying LM. In this work, we investigate whether all of this complexity is necessary, or if parts of these scaffolds can be removed on challenging tasks like SWE-bench. We show that in the case of SWE-bench, simply putting the entire environment into the context of a long context language model (LCLM) and properly prompting the model makes it competitive with carefully tuned, complex agent scaffolds. We show that a Gemini-1.5-Pro model without any scaffolding or tools achieves 38% on SWE-Bench-Verified, comparable with approaches using carefully tuned agent scaffolds (32%). While the unscaffolded approach with Gemini-1.5-Pro falls short of the strongest agentic architectures, we demonstrate that the more capable Gemini-2.5-Pro using the same unscaffolded approach directly attains a 50.8% solve rate. Additionally, a two-stage approach combining Gemini-1.5-Pro with Claude-3.7 achieves a competitive 48.6% solve rate. 

---
# Are LLMs complicated ethical dilemma analyzers? 

**Authors**: Jiashen, Jesse Yao, Allen Liu, Zhekai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08106)  

**Abstract**: One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making. 

---
# HYPERNYM MERCURY: Token Optimization through Semantic Field Constriction and Reconstruction from Hypernyms. A New Text Compression Method 

**Authors**: Chris Forrester, Octavia Sulea  

**Link**: [PDF](https://arxiv.org/pdf/2505.08058)  

**Abstract**: Compute optimization using token reduction of LLM prompts is an emerging task in the fields of NLP and next generation, agentic AI. In this white paper, we introduce a novel (patent pending) text representation scheme and a first-of-its-kind word-level semantic compression of paragraphs that can lead to over 90\% token reduction, while retaining high semantic similarity to the source text. We explain how this novel compression technique can be lossless and how the detail granularity is controllable. We discuss benchmark results over open source data (i.e. Bram Stoker's Dracula available through Project Gutenberg) and show how our results hold at the paragraph level, across multiple genres and models. 

---
# FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning 

**Authors**: Zhehao Zhang, Weijie Xu, Fanyou Wu, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.08054)  

**Abstract**: Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities. 

---
# TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction covering Multi-Level Error with Data Augmentation 

**Authors**: Yutong Liu, Feng Xiao, Ziyue Zhang, Yongbin Yu, Cheng Huang, Fan Gao, Xiangxiang Wang, Ma-bao Ban, Manping Fan, Thupten Tsering, Cheng Huang, Gadeng Luosang, Renzeng Duojie, Nyima Tashi  

**Link**: [PDF](https://arxiv.org/pdf/2505.08037)  

**Abstract**: Multi-level Tibetan spelling correction addresses errors at both the character and syllable levels within a unified model. Existing methods focus mainly on single-level correction and lack effective integration of both levels. Moreover, there are no open-source datasets or augmentation methods tailored for this task in Tibetan. To tackle this, we propose a data augmentation approach using unlabeled text to generate multi-level corruptions, and introduce TiSpell, a semi-masked model capable of correcting both character- and syllable-level errors. Although syllable-level correction is more challenging due to its reliance on global context, our semi-masked strategy simplifies this process. We synthesize nine types of corruptions on clean sentences to create a robust training set. Experiments on both simulated and real-world data demonstrate that TiSpell, trained on our dataset, outperforms baseline models and matches the performance of state-of-the-art approaches, confirming its effectiveness. 

---
# Large Language Models and Arabic Content: A Review 

**Authors**: Haneh Rhel, Dmitri Roussinov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08004)  

**Abstract**: Over the past three years, the rapid advancement of Large Language Models (LLMs) has had a profound impact on multiple areas of Artificial Intelligence (AI), particularly in Natural Language Processing (NLP) across diverse languages, including Arabic. Although Arabic is considered one of the most widely spoken languages across 27 countries in the Arabic world and used as a second language in some other non-Arabic countries as well, there is still a scarcity of Arabic resources, datasets, and tools. Arabic NLP tasks face various challenges due to the complexities of the Arabic language, including its rich morphology, intricate structure, and diverse writing standards, among other factors. Researchers have been actively addressing these challenges, demonstrating that pre-trained Large Language Models (LLMs) trained on multilingual corpora achieve significant success in various Arabic NLP tasks. This study provides an overview of using large language models (LLMs) for the Arabic language, highlighting early pre-trained Arabic Language models across various NLP applications and their ability to handle diverse Arabic content tasks and dialects. It also provides an overview of how techniques like finetuning and prompt engineering can enhance the performance of these models. Additionally, the study summarizes common Arabic benchmarks and datasets while presenting our observations on the persistent upward trend in the adoption of LLMs. 

---
# Task-Adaptive Semantic Communications with Controllable Diffusion-based Data Regeneration 

**Authors**: Fupei Guo, Achintha Wijesinghe, Songyang Zhang, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.07980)  

**Abstract**: Semantic communications represent a new paradigm of next-generation networking that shifts bit-wise data delivery to conveying the semantic meanings for bandwidth efficiency. To effectively accommodate various potential downstream tasks at the receiver side, one should adaptively convey the most critical semantic information. This work presents a novel task-adaptive semantic communication framework based on diffusion models that is capable of dynamically adjusting the semantic message delivery according to various downstream tasks. Specifically, we initialize the transmission of a deep-compressed general semantic representation from the transmitter to enable diffusion-based coarse data reconstruction at the receiver. The receiver identifies the task-specific demands and generates textual prompts as feedback. Integrated with the attention mechanism, the transmitter updates the semantic transmission with more details to better align with the objectives of the intended receivers. Our test results demonstrate the efficacy of the proposed method in adaptively preserving critical task-relevant information for semantic communications while preserving high compression efficiency. 

---
# Assessing and Mitigating Medical Knowledge Drift and Conflicts in Large Language Models 

**Authors**: Weiyi Wu, Xinwen Xu, Chongyang Gao, Xingjian Diao, Siting Li, Lucas A. Salas, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2505.07968)  

**Abstract**: Large Language Models (LLMs) have great potential in the field of health care, yet they face great challenges in adapting to rapidly evolving medical knowledge. This can lead to outdated or contradictory treatment suggestions. This study investigated how LLMs respond to evolving clinical guidelines, focusing on concept drift and internal inconsistencies. We developed the DriftMedQA benchmark to simulate guideline evolution and assessed the temporal reliability of various LLMs. Our evaluation of seven state-of-the-art models across 4,290 scenarios demonstrated difficulties in rejecting outdated recommendations and frequently endorsing conflicting guidance. Additionally, we explored two mitigation strategies: Retrieval-Augmented Generation and preference fine-tuning via Direct Preference Optimization. While each method improved model performance, their combination led to the most consistent and reliable results. These findings underscore the need to improve LLM robustness to temporal shifts to ensure more dependable applications in clinical practice. 

---
# Re$^2$: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions 

**Authors**: Daoze Zhang, Zhijian Bao, Sihang Du, Zhiyi Zhao, Kuangling Zhang, Dezheng Bao, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07920)  

**Abstract**: Peer review is a critical component of scientific progress in the fields like AI, but the rapid increase in submission volume has strained the reviewing system, which inevitably leads to reviewer shortages and declines review quality. Besides the growing research popularity, another key factor in this overload is the repeated resubmission of substandard manuscripts, largely due to the lack of effective tools for authors to self-evaluate their work before submission. Large Language Models (LLMs) show great promise in assisting both authors and reviewers, and their performance is fundamentally limited by the quality of the peer review data. However, existing peer review datasets face three major limitations: (1) limited data diversity, (2) inconsistent and low-quality data due to the use of revised rather than initial submissions, and (3) insufficient support for tasks involving rebuttal and reviewer-author interactions. To address these challenges, we introduce the largest consistency-ensured peer review and rebuttal dataset named Re^2, which comprises 19,926 initial submissions, 70,668 review comments, and 53,818 rebuttals from 24 conferences and 21 workshops on OpenReview. Moreover, the rebuttal and discussion stage is framed as a multi-turn conversation paradigm to support both traditional static review tasks and dynamic interactive LLM assistants, providing more practical guidance for authors to refine their manuscripts and helping alleviate the growing review burden. Our data and code are available in this https URL. 

---
# SEM: Reinforcement Learning for Search-Efficient Large Language Models 

**Authors**: Zeyang Sha, Shiwen Cui, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07903)  

**Abstract**: Recent advancements in Large Language Models(LLMs) have demonstrated their capabilities not only in reasoning but also in invoking external tools, particularly search engines. However, teaching models to discern when to invoke search and when to rely on their internal knowledge remains a significant challenge. Existing reinforcement learning approaches often lead to redundant search behaviors, resulting in inefficiencies and over-cost. In this paper, we propose SEM, a novel post-training reinforcement learning framework that explicitly trains LLMs to optimize search usage. By constructing a balanced dataset combining MuSiQue and MMLU, we create scenarios where the model must learn to distinguish between questions it can answer directly and those requiring external retrieval. We design a structured reasoning template and employ Group Relative Policy Optimization(GRPO) to post-train the model's search behaviors. Our reward function encourages accurate answering without unnecessary search while promoting effective retrieval when needed. Experimental results demonstrate that our method significantly reduces redundant search operations while maintaining or improving answer accuracy across multiple challenging benchmarks. This framework advances the model's reasoning efficiency and extends its capability to judiciously leverage external knowledge. 

---
# DeltaEdit: Enhancing Sequential Editing in Large Language Models by Controlling Superimposed Noise 

**Authors**: Ding Cao, Yuchen Cai, Rongxi Guo, Xuesong He, Guiquan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07899)  

**Abstract**: Sequential knowledge editing techniques aim to continuously update the knowledge in large language models at a low cost, preventing the models from generating outdated or incorrect information. However, existing sequential editing methods suffer from a significant decline in editing success rates after long-term editing. Through theoretical analysis and experiments, we identify that as the number of edits increases, the model's output increasingly deviates from the desired target, leading to a drop in editing success rates. We refer to this issue as the accumulation of superimposed noise problem. To address this, we identify the factors contributing to this deviation and propose DeltaEdit, a novel method that optimizes update parameters through a dynamic orthogonal constraints strategy, effectively reducing interference between edits to mitigate deviation. Experimental results demonstrate that DeltaEdit significantly outperforms existing methods in edit success rates and the retention of generalization capabilities, ensuring stable and reliable model performance even under extensive sequential editing. 

---
# LongCodeBench: Evaluating Coding LLMs at 1M Context Windows 

**Authors**: Stefano Rando, Luca Romani, Alessio Sampieri, Yuta Kyuragi, Luca Franco, Fabio Galasso, Tatsunori Hashimoto, John Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07897)  

**Abstract**: Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5. 

---
# TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking 

**Authors**: Ching Nam Hang, Pei-Duo Yu, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07891)  

**Abstract**: In the age of social media, the rapid spread of misinformation and rumors has led to the emergence of infodemics, where false information poses a significant threat to society. To combat this issue, we introduce TrumorGPT , a novel generative artificial intelligence solution designed for fact-checking in the health domain. TrumorGPT aims to distinguish "trumors", which are health-related rumors that turn out to be true, providing a crucial tool in differentiating between mere speculation and verified facts. This framework leverages a large language model (LLM) with few-shot learning for semantic health knowledge graph construction and semantic reasoning. TrumorGPT incorporates graph-based retrieval-augmented generation (GraphRAG) to address the hallucination issue common in LLMs and the limitations of static training data. GraphRAG involves accessing and utilizing information from regularly updated semantic health knowledge graphs that consist of the latest medical news and health information, ensuring that fact-checking by TrumorGPT is based on the most recent data. Evaluating with extensive healthcare datasets, TrumorGPT demonstrates superior performance in fact-checking for public health claims. Its ability to effectively conduct fact-checking across various platforms marks a critical step forward in the fight against health-related misinformation, enhancing trust and accuracy in the digital information age. 

---
# TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks 

**Authors**: Kutay Ertürk, Furkan Altınışık, İrem Sarıaltın, Ömer Nezih Gerek  

**Link**: [PDF](https://arxiv.org/pdf/2505.07890)  

**Abstract**: This study presents TSLFormer, a light and robust word-level Turkish Sign Language (TSL) recognition model that treats sign gestures as ordered, string-like language. Instead of using raw RGB or depth videos, our method only works with 3D joint positions - articulation points - extracted using Google's Mediapipe library, which focuses on the hand and torso skeletal locations. This creates efficient input dimensionality reduction while preserving important semantic gesture information.
Our approach revisits sign language recognition as sequence-to-sequence translation, inspired by the linguistic nature of sign languages and the success of transformers in natural language processing. Since TSLFormer uses the self-attention mechanism, it effectively captures temporal co-occurrence within gesture sequences and highlights meaningful motion patterns as words unfold.
Evaluated on the AUTSL dataset with over 36,000 samples and 227 different words, TSLFormer achieves competitive performance with minimal computational cost. These results show that joint-based input is sufficient for enabling real-time, mobile, and assistive communication systems for hearing-impaired individuals. 

---
# BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning 

**Authors**: Yuyang Liu, Liuzhenghao Lv, Xiancheng Zhang, Li Yuan, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07889)  

**Abstract**: Biological protocols are fundamental to reproducible and safe life science research. While LLMs excel on general tasks, their systematic evaluation on these highly specialized, accuracy-critical, and inherently procedural texts remains limited. In this work, we present BioProBench, the first large-scale, integrated multi-task benchmark for biological protocol understanding and reasoning. While limited benchmarks have touched upon specific aspects like protocol QA, BioProBench provides a comprehensive suite of five core tasks: Protocol Question Answering, Step Ordering, Error Correction, Protocol Generation, and Protocol Reasoning, enabling a holistic evaluation of LLMs on procedural biological texts. Built upon 27K original protocols, it yields nearly 556K high-quality structured instances. We evaluate 12 mainstream open/closed-source LLMs on BioProBench. Experimental results reveal that while top models preform well on surface understanding tasks, struggle significantly with deep reasoning and structured generation tasks like ordering and generation. Furthermore, model comparisons reveal diverse performance: certain open-source models approach closed-source levels on some tasks, yet bio-specific small models lag behind general LLMs, indicating limitations on complex procedural content. Overall, our findings underscore that procedural reasoning within biological protocols represents a significant challenge for current LLMs. BioProBench serves as a standardized framework to diagnose these specific limitations and guide the development of AI systems better equipped for safely automating complex scientific procedures. The code and data are available at: this https URL and this https URL. 

---
# Implementing Long Text Style Transfer with LLMs through Dual-Layered Sentence and Paragraph Structure Extraction and Mapping 

**Authors**: Yusen Wu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07888)  

**Abstract**: This paper addresses the challenge in long-text style transfer using zero-shot learning of large language models (LLMs), proposing a hierarchical framework that combines sentence-level stylistic adaptation with paragraph-level structural coherence. We argue that in the process of effective paragraph-style transfer, to preserve the consistency of original syntactic and semantic information, it is essential to perform style transfer not only at the sentence level but also to incorporate paragraph-level semantic considerations, while ensuring structural coherence across inter-sentential relationships. Our proposed framework, ZeroStylus, operates through two systematic phases: hierarchical template acquisition from reference texts and template-guided generation with multi-granular matching. The framework dynamically constructs sentence and paragraph template repositories, enabling context-aware transformations while preserving inter-sentence logical relationships. Experimental evaluations demonstrate significant improvements over baseline methods, with structured rewriting achieving 6.90 average score compared to 6.70 for direct prompting approaches in tri-axial metrics assessing style consistency, content preservation, and expression quality. Ablation studies validate the necessity of both template hierarchies during style transfer, showing higher content preservation win rate against sentence-only approaches through paragraph-level structural encoding, as well as direct prompting method through sentence-level pattern extraction and matching. The results establish new capabilities for coherent long-text style transfer without requiring parallel corpora or LLM fine-tuning. 

---
# PLHF: Prompt Optimization with Few-Shot Human Feedback 

**Authors**: Chun-Pai Yang, Kan Zheng, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07886)  

**Abstract**: Automatic prompt optimization frameworks are developed to obtain suitable prompts for large language models (LLMs) with respect to desired output quality metrics. Although existing approaches can handle conventional tasks such as fixed-solution question answering, defining the metric becomes complicated when the output quality cannot be easily assessed by comparisons with standard golden samples. Consequently, optimizing the prompts effectively and efficiently without a clear metric becomes a critical challenge. To address the issue, we present PLHF (which stands for "P"rompt "L"earning with "H"uman "F"eedback), a few-shot prompt optimization framework inspired by the well-known RLHF technique. Different from naive strategies, PLHF employs a specific evaluator module acting as the metric to estimate the output quality. PLHF requires only a single round of human feedback to complete the entire prompt optimization process. Empirical results on both public and industrial datasets show that PLHF outperforms prior output grading strategies for LLM prompt optimizations. 

---
# Development of a WAZOBIA-Named Entity Recognition System 

**Authors**: S.E Emedem, I.E Onyenwe, E. G Onyedinma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07884)  

**Abstract**: Named Entity Recognition NER is very crucial for various natural language processing applications, including information extraction, machine translation, and sentiment analysis. Despite the ever-increasing interest in African languages within computational linguistics, existing NER systems focus mainly on English, European, and a few other global languages, leaving a significant gap for under-resourced languages. This research presents the development of a WAZOBIA-NER system tailored for the three most prominent Nigerian languages: Hausa, Yoruba, and Igbo. This research begins with a comprehensive compilation of annotated datasets for each language, addressing data scarcity and linguistic diversity challenges. Exploring the state-of-the-art machine learning technique, Conditional Random Fields (CRF) and deep learning models such as Bidirectional Long Short-Term Memory (BiLSTM), Bidirectional Encoder Representation from Transformers (Bert) and fine-tune with a Recurrent Neural Network (RNN), the study evaluates the effectiveness of these approaches in recognizing three entities: persons, organizations, and locations. The system utilizes optical character recognition (OCR) technology to convert textual images into machine-readable text, thereby enabling the Wazobia system to accept both input text and textual images for extraction purposes. The system achieved a performance of 0.9511 in precision, 0.9400 in recall, 0.9564 in F1-score, and 0.9301 in accuracy. The model's evaluation was conducted across three languages, with precision, recall, F1-score, and accuracy as key assessment metrics. The Wazobia-NER system demonstrates that it is feasible to build robust NER tools for under-resourced African languages using current NLP frameworks and transfer learning. 

---
# Recovering Event Probabilities from Large Language Model Embeddings via Axiomatic Constraints 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.07883)  

**Abstract**: Rational decision-making under uncertainty requires coherent degrees of belief in events. However, event probabilities generated by Large Language Models (LLMs) have been shown to exhibit incoherence, violating the axioms of probability theory. This raises the question of whether coherent event probabilities can be recovered from the embeddings used by the models. If so, those derived probabilities could be used as more accurate estimates in events involving uncertainty. To explore this question, we propose enforcing axiomatic constraints, such as the additive rule of probability theory, in the latent space learned by an extended variational autoencoder (VAE) applied to LLM embeddings. This approach enables event probabilities to naturally emerge in the latent space as the VAE learns to both reconstruct the original embeddings and predict the embeddings of semantically related events. We evaluate our method on complementary events (i.e., event A and its complement, event not-A), where the true probabilities of the two events must sum to 1. Experiment results on open-weight language models demonstrate that probabilities recovered from embeddings exhibit greater coherence than those directly reported by the corresponding models and align closely with the true probabilities. 

---
# The Sound of Populism: Distinct Linguistic Features Across Populist Variants 

**Authors**: Yu Wang, Runxi Yu, Zhongyuan Wang, Jing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.07874)  

**Abstract**: This study explores the sound of populism by integrating the classic Linguistic Inquiry and Word Count (LIWC) features, which capture the emotional and stylistic tones of language, with a fine-tuned RoBERTa model, a state-of-the-art context-aware language model trained to detect nuanced expressions of populism. This approach allows us to uncover the auditory dimensions of political rhetoric in U.S. presidential inaugural and State of the Union addresses. We examine how four key populist dimensions (i.e., left-wing, right-wing, anti-elitism, and people-centrism) manifest in the linguistic markers of speech, drawing attention to both commonalities and distinct tonal shifts across these variants. Our findings reveal that populist rhetoric consistently features a direct, assertive ``sound" that forges a connection with ``the people'' and constructs a charismatic leadership persona. However, this sound is not simply informal but strategically calibrated. Notably, right-wing populism and people-centrism exhibit a more emotionally charged discourse, resonating with themes of identity, grievance, and crisis, in contrast to the relatively restrained emotional tones of left-wing and anti-elitist expressions. 

---
# Evaluating Financial Sentiment Analysis with Annotators Instruction Assisted Prompting: Enhancing Contextual Interpretation and Stock Prediction Accuracy 

**Authors**: A M Muntasir Rahman, Ajim Uddin, Guiling "Grace" Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07871)  

**Abstract**: Financial sentiment analysis (FSA) presents unique challenges to LLMs that surpass those in typical sentiment analysis due to the nuanced language used in financial contexts. The prowess of these models is often undermined by the inherent subjectivity of sentiment classifications in existing benchmark datasets like Financial Phrasebank. These datasets typically feature undefined sentiment classes that reflect the highly individualized perspectives of annotators, leading to significant variability in annotations. This variability results in an unfair expectation for LLMs during benchmarking, where they are tasked to conjecture the subjective viewpoints of human annotators without sufficient context. In this paper, we introduce the Annotators' Instruction Assisted Prompt, a novel evaluation prompt designed to redefine the task definition of FSA for LLMs. By integrating detailed task instructions originally intended for human annotators into the LLMs' prompt framework, AIAP aims to standardize the understanding of sentiment across both human and machine interpretations, providing a fair and context-rich foundation for sentiment analysis. We utilize a new dataset, WSBS, derived from the WallStreetBets subreddit to demonstrate how AIAP significantly enhances LLM performance by aligning machine operations with the refined task definitions. Experimental results demonstrate that AIAP enhances LLM performance significantly, with improvements up to 9.08. This context-aware approach not only yields incremental gains in performance but also introduces an innovative sentiment-indexing method utilizing model confidence scores. This method enhances stock price prediction models and extracts more value from the financial sentiment analysis, underscoring the significance of WSB as a critical source of financial text. Our research offers insights into both improving FSA through better evaluation methods. 

---
# Efficient Fairness Testing in Large Language Models: Prioritizing Metamorphic Relations for Bias Detection 

**Authors**: Suavis Giramata, Madhusudan Srinivasan, Venkat Naidu Gudivada, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2505.07870)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in various applications, raising critical concerns about fairness and potential biases in their outputs. This paper explores the prioritization of metamorphic relations (MRs) in metamorphic testing as a strategy to efficiently detect fairness issues within LLMs. Given the exponential growth of possible test cases, exhaustive testing is impractical; therefore, prioritizing MRs based on their effectiveness in detecting fairness violations is crucial. We apply a sentence diversity-based approach to compute and rank MRs to optimize fault detection. Experimental results demonstrate that our proposed prioritization approach improves fault detection rates by 22% compared to random prioritization and 12% compared to distance-based prioritization, while reducing the time to the first failure by 15% and 8%, respectively. Furthermore, our approach performs within 5% of fault-based prioritization in effectiveness, while significantly reducing the computational cost associated with fault labeling. These results validate the effectiveness of diversity-based MR prioritization in enhancing fairness testing for LLMs. 

---
# QoSBERT: An Uncertainty-Aware Approach based on Pre-trained Language Models for Service Quality Prediction 

**Authors**: Ziliang Wang, Xiaohong Zhang, Ze Shi Li, Meng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07863)  

**Abstract**: Accurate prediction of Quality of Service (QoS) metrics is fundamental for selecting and managing cloud based services. Traditional QoS models rely on manual feature engineering and yield only point estimates, offering no insight into the confidence of their predictions. In this paper, we propose QoSBERT, the first framework that reformulates QoS prediction as a semantic regression task based on pre trained language models. Unlike previous approaches relying on sparse numerical features, QoSBERT automatically encodes user service metadata into natural language descriptions, enabling deep semantic understanding. Furthermore, we integrate a Monte Carlo Dropout based uncertainty estimation module, allowing for trustworthy and risk-aware service quality prediction, which is crucial yet underexplored in existing QoS models. QoSBERT applies attentive pooling over contextualized embeddings and a lightweight multilayer perceptron regressor, fine tuned jointly to minimize absolute error. We further exploit the resulting uncertainty estimates to select high quality training samples, improving robustness in low resource settings. On standard QoS benchmark datasets, QoSBERT achieves an average reduction of 11.7% in MAE and 6.7% in RMSE for response time prediction, and 6.9% in MAE for throughput prediction compared to the strongest baselines, while providing well calibrated confidence intervals for robust and trustworthy service quality estimation. Our approach not only advances the accuracy of service quality prediction but also delivers reliable uncertainty quantification, paving the way for more trustworthy, data driven service selection and optimization. 

---
# Graph Laplacian Wavelet Transformer via Learnable Spectral Decomposition 

**Authors**: Andrew Kiruluta, Eric Lundy, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2505.07862)  

**Abstract**: Existing sequence to sequence models for structured language tasks rely heavily on the dot product self attention mechanism, which incurs quadratic complexity in both computation and memory for input length N. We introduce the Graph Wavelet Transformer (GWT), a novel architecture that replaces this bottleneck with a learnable, multi scale wavelet transform defined over an explicit graph Laplacian derived from syntactic or semantic parses. Our analysis shows that multi scale spectral decomposition offers an interpretable, efficient, and expressive alternative to quadratic self attention for graph structured sequence modeling. 

---
# Scalable LLM Math Reasoning Acceleration with Low-rank Distillation 

**Authors**: Harry Dong, Bilge Acun, Beidi Chen, Yuejie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07861)  

**Abstract**: Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a low-cost distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>11% reduction to generate 2048 tokens with Qwen 2.5 14B) while encouraging response brevity. 

---
# Boosting Performance on ARC is a Matter of Perspective 

**Authors**: Daniel Franzen, Jan Disselhoff, David Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.07859)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities. In this work, we leverage task-specific data augmentations throughout the training, generation, and scoring phases, and employ a depth-first search algorithm to generate diverse, high-probability candidate solutions. Furthermore, we utilize the LLM not only as a generator but also as a scorer, using its output probabilities to select the most promising solutions. Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches. While concurrent closed-source work has reported higher scores, our method distinguishes itself through its transparency, reproducibility, and remarkably low inference cost, averaging only around 2ct per task on readily available hardware (we assume a price of 36ct/hour for a Nvidia 4090 GPU). 

---
# Scaling Laws for Speculative Decoding 

**Authors**: Siyuan Yan, Mo Zhu, Guo-qing Jiang, Jianfei Wang, Jiaxing Chen, Wentai Zhang, Xiang Liao, Xiao Cui, Chen Zhang, Zhuoran Song, Ran Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07858)  

**Abstract**: The escalating demand for efficient decoding in large language models (LLMs) is particularly critical for reasoning-intensive architectures like OpenAI-o3 and DeepSeek-R1, which depend on extended chain-of-thought reasoning. This study investigates speculative decoding techniques through dense LLM architectures to establish foundational insights for accelerating reasoning tasks. While speculative decoding methods leveraging parallel draft-verification cycles have emerged as promising acceleration techniques, the scaling laws governing decoding efficiency remain under-explored compared to conventional backbone LLMs developed through Pretraining->SFT->RLHF training paradigms. In this work, we discover Log-linear Scaling Laws (Theorem 1.1, 1.2 and 1.3) governing draft model acceptance rate (or decoding speed) across three dimensions: pretraining token volume, draft model capacity, and decoding batch size. Building on these laws, we achieve Scylla, which coordinates multi-dimensional scaling for popular LLMs (Llama2/3, Qwen2.5). Empirical validation shows Scylla achieves 1.5-2.2 higher acceptance rate than EAGLE2 and 0.3 higher than EAGLE3 at temperature T = 0, with peak performance gains on summarization and QA tasks (Figure 2). Industrial inference engine deployments demonstrate 2X decoding throughput improvements over EAGLE2 (Table 5), validating the transformative potential of systematic scaling for efficient LLM inference. Code will be released later. 

---
# Enhanced Urdu Intent Detection with Large Language Models and Prototype-Informed Predictive Pipelines 

**Authors**: Faiza Hassan, Summra Saleem, Kashif Javed, Muhammad Nabeel Asim, Abdur Rehman, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2505.07857)  

**Abstract**: Multifarious intent detection predictors are developed for different languages, including English, Chinese and French, however, the field remains underdeveloped for Urdu, the 10th most spoken language. In the realm of well-known languages, intent detection predictors utilize the strategy of few-shot learning and prediction of unseen classes based on the model training on seen classes. However, Urdu language lacks few-shot strategy based intent detection predictors and traditional predictors are focused on prediction of the same classes which models have seen in the train set. To empower Urdu language specific intent detection, this introduces a unique contrastive learning approach that leverages unlabeled Urdu data to re-train pre-trained language models. This re-training empowers LLMs representation learning for the downstream intent detection task. Finally, it reaps the combined potential of pre-trained LLMs and the prototype-informed attention mechanism to create a comprehensive end-to-end LLMPIA intent detection pipeline. Under the paradigm of proposed predictive pipeline, it explores the potential of 6 distinct language models and 13 distinct similarity computation methods. The proposed framework is evaluated on 2 public benchmark datasets, namely ATIS encompassing 5836 samples and Web Queries having 8519 samples. Across ATIS dataset under 4-way 1 shot and 4-way 5 shot experimental settings LLMPIA achieved 83.28% and 98.25% F1-Score and on Web Queries dataset produced 76.23% and 84.42% F1-Score, respectively. In an additional case study on the Web Queries dataset under same classes train and test set settings, LLMPIA outperformed state-of-the-art predictor by 53.55% F1-Score. 

---
# Unpacking Robustness in Inflectional Languages: Adversarial Evaluation and Mechanistic Insights 

**Authors**: Paweł Walkowiak, Marek Klonowski, Marcin Oleksy, Arkadiusz Janz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07856)  

**Abstract**: Various techniques are used in the generation of adversarial examples, including methods such as TextBugger which introduce minor, hardly visible perturbations to words leading to changes in model behaviour. Another class of techniques involves substituting words with their synonyms in a way that preserves the text's meaning but alters its predicted class, with TextFooler being a prominent example of such attacks. Most adversarial example generation methods are developed and evaluated primarily on non-inflectional languages, typically English. In this work, we evaluate and explain how adversarial attacks perform in inflectional languages. To explain the impact of inflection on model behaviour and its robustness under attack, we designed a novel protocol inspired by mechanistic interpretability, based on Edge Attribution Patching (EAP) method. The proposed evaluation protocol relies on parallel task-specific corpora that include both inflected and syncretic variants of texts in two languages -- Polish and English. To analyse the models and explain the relationship between inflection and adversarial robustness, we create a new benchmark based on task-oriented dataset MultiEmo, enabling the identification of mechanistic inflection-related elements of circuits within the model and analyse their behaviour under attack. 

---
# CrashSage: A Large Language Model-Centered Framework for Contextual and Interpretable Traffic Crash Analysis 

**Authors**: Hao Zhen, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07853)  

**Abstract**: Road crashes claim over 1.3 million lives annually worldwide and incur global economic losses exceeding \$1.8 trillion. Such profound societal and financial impacts underscore the urgent need for road safety research that uncovers crash mechanisms and delivers actionable insights. Conventional statistical models and tree ensemble approaches typically rely on structured crash data, overlooking contextual nuances and struggling to capture complex relationships and underlying semantics. Moreover, these approaches tend to incur significant information loss, particularly in narrative elements related to multi-vehicle interactions, crash progression, and rare event characteristics. This study presents CrashSage, a novel Large Language Model (LLM)-centered framework designed to advance crash analysis and modeling through four key innovations. First, we introduce a tabular-to-text transformation strategy paired with relational data integration schema, enabling the conversion of raw, heterogeneous crash data into enriched, structured textual narratives that retain essential structural and relational context. Second, we apply context-aware data augmentation using a base LLM model to improve narrative coherence while preserving factual integrity. Third, we fine-tune the LLaMA3-8B model for crash severity inference, demonstrating superior performance over baseline approaches, including zero-shot, zero-shot with chain-of-thought prompting, and few-shot learning, with multiple models (GPT-4o, GPT-4o-mini, LLaMA3-70B). Finally, we employ a gradient-based explainability technique to elucidate model decisions at both the individual crash level and across broader risk factor dimensions. This interpretability mechanism enhances transparency and enables targeted road safety interventions by providing deeper insights into the most influential factors. 

---
# Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment 

**Authors**: Ali Senol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07852)  

**Abstract**: Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models. 

---
# A Tale of Two Identities: An Ethical Audit of Human and AI-Crafted Personas 

**Authors**: Pranav Narayanan Venkit, Jiayi Li, Yingfan Zhou, Sarah Rajtmajer, Shomir Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2505.07850)  

**Abstract**: As LLMs (large language models) are increasingly used to generate synthetic personas particularly in data-limited domains such as health, privacy, and HCI, it becomes necessary to understand how these narratives represent identity, especially that of minority communities. In this paper, we audit synthetic personas generated by 3 LLMs (GPT4o, Gemini 1.5 Pro, Deepseek 2.5) through the lens of representational harm, focusing specifically on racial identity. Using a mixed methods approach combining close reading, lexical analysis, and a parameterized creativity framework, we compare 1512 LLM generated personas to human-authored responses. Our findings reveal that LLMs disproportionately foreground racial markers, overproduce culturally coded language, and construct personas that are syntactically elaborate yet narratively reductive. These patterns result in a range of sociotechnical harms, including stereotyping, exoticism, erasure, and benevolent bias, that are often obfuscated by superficially positive narrations. We formalize this phenomenon as algorithmic othering, where minoritized identities are rendered hypervisible but less authentic. Based on these findings, we offer design recommendations for narrative-aware evaluation metrics and community-centered validation protocols for synthetic identity generation. 

---
# Polysemy of Synthetic Neurons Towards a New Type of Explanatory Categorical Vector Spaces 

**Authors**: Michael Pichat, William Pogrund, Paloma Pichat, Judicael Poumay, Armanouche Gasparian, Samuel Demarchi, Martin Corbet, Alois Georgeon, Michael Veillet-Guillem  

**Link**: [PDF](https://arxiv.org/pdf/2505.07831)  

**Abstract**: The polysemantic nature of synthetic neurons in artificial intelligence language models is currently understood as the result of a necessary superposition of distributed features within the latent space. We propose an alternative approach, geometrically defining a neuron in layer n as a categorical vector space with a non-orthogonal basis, composed of categorical sub-dimensions extracted from preceding neurons in layer n-1. This categorical vector space is structured by the activation space of each neuron and enables, via an intra-neuronal attention process, the identification and utilization of a critical categorical zone for the efficiency of the language model - more homogeneous and located at the intersection of these different categorical sub-dimensions. 

---
# CodePDE: An Inference Framework for LLM-driven PDE Solver Generation 

**Authors**: Shanda Li, Tanya Marwah, Junhong Shen, Weiwei Sun, Andrej Risteski, Yiming Yang, Ameet Talwalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08783)  

**Abstract**: Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). Leveraging advanced inference-time algorithms and scaling strategies, CodePDE unlocks critical capacities of LLM for PDE solving: reasoning, debugging, selfrefinement, and test-time scaling -- all without task-specific tuning. CodePDE achieves superhuman performance across a range of representative PDE problems. We also present a systematic empirical analysis of LLM generated solvers, analyzing their accuracy, efficiency, and numerical scheme choices. Our findings highlight the promise and the current limitations of LLMs in PDE solving, offering a new perspective on solver design and opportunities for future model development. Our code is available at this https URL. 

---
# Memorization-Compression Cycles Improve Generalization 

**Authors**: Fangyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08727)  

**Abstract**: We prove theoretically that generalization improves not only through data scaling but also by compressing internal representations. To operationalize this insight, we introduce the Information Bottleneck Language Modeling (IBLM) objective, which reframes language modeling as a constrained optimization problem: minimizing representation entropy subject to optimal prediction performance. Empirically, we observe an emergent memorization-compression cycle during LLM pretraining, evidenced by oscillation positive/negative gradient alignment between cross-entropy and Matrix-Based Entropy (MBE), a measure of representation entropy. This pattern closely mirrors the predictive-compressive trade-off prescribed by IBLM and also parallels the biological alternation between awake learning and sleep consolidation. Motivated by this observation, we propose Gated Phase Transition (GAPT), a training algorithm that adaptively switches between memorization and compression phases. When applied to GPT-2 pretraining on FineWeb dataset, GAPT reduces MBE by 50% and improves cross-entropy by 4.8%. GAPT improves OOD generalizatino by 35% in a pretraining task on arithmetic multiplication. In a setting designed to simulate catastrophic forgetting, GAPT reduces interference by compressing and separating representations, achieving a 97% improvement in separation - paralleling the functional role of sleep consolidation. 

---
# LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs 

**Authors**: K M Sajjadul Islam, Ayesha Siddika Nipu, Jiawei Wu, Praveen Madiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.08704)  

**Abstract**: Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting. 

---
# TRAIL: Trace Reasoning and Agentic Issue Localization 

**Authors**: Darshan Deshpande, Varun Gangal, Hersh Mehta, Jitin Krishnan, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2505.08638)  

**Abstract**: The increasing adoption of agentic workflows across diverse domains brings a critical need to scalably and systematically evaluate the complex traces these systems generate. Current evaluation methods depend on manual, domain-specific human analysis of lengthy workflow traces - an approach that does not scale with the growing complexity and volume of agentic outputs. Error analysis in these settings is further complicated by the interplay of external tool outputs and language model reasoning, making it more challenging than traditional software debugging. In this work, we (1) articulate the need for robust and dynamic evaluation methods for agentic workflow traces, (2) introduce a formal taxonomy of error types encountered in agentic systems, and (3) present a set of 148 large human-annotated traces (TRAIL) constructed using this taxonomy and grounded in established agentic benchmarks. To ensure ecological validity, we curate traces from both single and multi-agent systems, focusing on real-world applications such as software engineering and open-world information retrieval. Our evaluations reveal that modern long context LLMs perform poorly at trace debugging, with the best Gemini-2.5-pro model scoring a mere 11% on TRAIL. Our dataset and code are made publicly available to support and accelerate future research in scalable evaluation for agentic workflows. 

---
# Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models 

**Authors**: Donghoon Kim, Minji Bae, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2505.08622)  

**Abstract**: Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models. 

---
# Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact on Performance and Efficiency 

**Authors**: Adel Ammar, Anis Koubaa, Omer Nacar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2505.08445)  

**Abstract**: Large language models achieve high task performance yet often hallucinate or rely on outdated knowledge. Retrieval-augmented generation (RAG) addresses these gaps by coupling generation with external search. We analyse how hyperparameters influence speed and quality in RAG systems, covering Chroma and Faiss vector stores, chunking policies, cross-encoder re-ranking, and temperature, and we evaluate six metrics: faithfulness, answer correctness, answer relevancy, context precision, context recall, and answer similarity. Chroma processes queries 13% faster, whereas Faiss yields higher retrieval precision, revealing a clear speed-accuracy trade-off. Naive fixed-length chunking with small windows and minimal overlap outperforms semantic segmentation while remaining the quickest option. Re-ranking provides modest gains in retrieval quality yet increases runtime by roughly a factor of 5, so its usefulness depends on latency constraints. These results help practitioners balance computational cost and accuracy when tuning RAG systems for transparent, up-to-date responses. Finally, we re-evaluate the top configurations with a corrective RAG workflow and show that their advantages persist when the model can iteratively request additional evidence. We obtain a near-perfect context precision (99%), which demonstrates that RAG systems can achieve extremely high retrieval accuracy with the right combination of hyperparameters, with significant implications for applications where retrieval quality directly impacts downstream task performance, such as clinical decision support in healthcare. 

---
# Not that Groove: Zero-Shot Symbolic Music Editing 

**Authors**: Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08203)  

**Abstract**: Most work in AI music generation focused on audio, which has seen limited use in the music production industry due to its rigidity. To maximize flexibility while assuming only textual instructions from producers, we are among the first to tackle symbolic music editing. We circumvent the known challenge of lack of labeled data by proving that LLMs with zero-shot prompting can effectively edit drum grooves. The recipe of success is a creatively designed format that interfaces LLMs and music, while we facilitate evaluation by providing an evaluation dataset with annotated unit tests that highly aligns with musicians' judgment. 

---
# A Large-Scale Empirical Analysis of Custom GPTs' Vulnerabilities in the OpenAI Ecosystem 

**Authors**: Sunday Oyinlola Ogundoyin, Muhammad Ikram, Hassan Jameel Asghar, Benjamin Zi Hao Zhao, Dali Kaafar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08148)  

**Abstract**: Millions of users leverage generative pretrained transformer (GPT)-based language models developed by leading model providers for a wide range of tasks. To support enhanced user interaction and customization, many platforms-such as OpenAI-now enable developers to create and publish tailored model instances, known as custom GPTs, via dedicated repositories or application stores. These custom GPTs empower users to browse and interact with specialized applications designed to meet specific needs. However, as custom GPTs see growing adoption, concerns regarding their security vulnerabilities have intensified. Existing research on these vulnerabilities remains largely theoretical, often lacking empirical, large-scale, and statistically rigorous assessments of associated risks.
In this study, we analyze 14,904 custom GPTs to assess their susceptibility to seven exploitable threats, such as roleplay-based attacks, system prompt leakage, phishing content generation, and malicious code synthesis, across various categories and popularity tiers within the OpenAI marketplace. We introduce a multi-metric ranking system to examine the relationship between a custom GPT's popularity and its associated security risks.
Our findings reveal that over 95% of custom GPTs lack adequate security protections. The most prevalent vulnerabilities include roleplay-based vulnerabilities (96.51%), system prompt leakage (92.20%), and phishing (91.22%). Furthermore, we demonstrate that OpenAI's foundational models exhibit inherent security weaknesses, which are often inherited or amplified in custom GPTs. These results highlight the urgent need for enhanced security measures and stricter content moderation to ensure the safe deployment of GPT-based applications. 

---
# Large Language Models for Computer-Aided Design: A Survey 

**Authors**: Licheng Zhang, Bach Le, Naveed Akhtar, Siew-Kei Lam, Tuan Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08137)  

**Abstract**: Large Language Models (LLMs) have seen rapid advancements in recent years, with models like ChatGPT and DeepSeek, showcasing their remarkable capabilities across diverse domains. While substantial research has been conducted on LLMs in various fields, a comprehensive review focusing on their integration with Computer-Aided Design (CAD) remains notably absent. CAD is the industry standard for 3D modeling and plays a vital role in the design and development of products across different industries. As the complexity of modern designs increases, the potential for LLMs to enhance and streamline CAD workflows presents an exciting frontier. This article presents the first systematic survey exploring the intersection of LLMs and CAD. We begin by outlining the industrial significance of CAD, highlighting the need for AI-driven innovation. Next, we provide a detailed overview of the foundation of LLMs. We also examine both closed-source LLMs as well as publicly available models. The core of this review focuses on the various applications of LLMs in CAD, providing a taxonomy of six key areas where these models are making considerable impact. Finally, we propose several promising future directions for further advancements, which offer vast opportunities for innovation and are poised to shape the future of CAD technology. Github: this https URL 

---
# Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders 

**Authors**: Dong Shu, Xuansheng Wu, Haiyan Zhao, Mengnan Du, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08080)  

**Abstract**: Sparse Autoencoders (SAEs) have recently emerged as powerful tools for interpreting and steering the internal representations of large language models (LLMs). However, conventional approaches to analyzing SAEs typically rely solely on input-side activations, without considering the causal influence between each latent feature and the model's output. This work is built on two key hypotheses: (1) activated latents do not contribute equally to the construction of the model's output, and (2) only latents with high causal influence are effective for model steering. To validate these hypotheses, we propose Gradient Sparse Autoencoder (GradSAE), a simple yet effective method that identifies the most influential latents by incorporating output-side gradient information. 

---
# NAZM: Network Analysis of Zonal Metrics in Persian Poetic Tradition 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.08052)  

**Abstract**: This study formalizes a computational model to simulate classical Persian poets' dynamics of influence through constructing a multi-dimensional similarity network. Using a rigorously curated dataset based on Ganjoor's corpus, we draw upon semantic, lexical, stylistic, thematic, and metrical features to demarcate each poet's corpus. Each is contained within weighted similarity matrices, which are then appended to generate an aggregate graph showing poet-to-poet influence. Further network investigation is carried out to identify key poets, style hubs, and bridging poets by calculating degree, closeness, betweenness, eigenvector, and Katz centrality measures. Further, for typological insight, we use the Louvain community detection algorithm to demarcate clusters of poets sharing both style and theme coherence, which correspond closely to acknowledged schools of literature like Sabk-e Hindi, Sabk-e Khorasani, and the Bazgasht-e Adabi phenomenon. Our findings provide a new data-driven view of Persian literature distinguished between canonical significance and interextual influence, thus highlighting relatively lesser-known figures who hold great structural significance. Combining computational linguistics with literary study, this paper produces an interpretable and scalable model for poetic tradition, enabling retrospective reflection as well as forward-looking research within digital humanities. 

---
# SciCom Wiki: Fact-Checking and FAIR Knowledge Distribution for Scientific Videos and Podcasts 

**Authors**: Tim Wittenborg, Constantin Sebastian Tremel, Niklas Stehr, Oliver Karras, Markus Stocker, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07912)  

**Abstract**: Democratic societies need accessible, reliable information. Videos and Podcasts have established themselves as the medium of choice for civic dissemination, but also as carriers of misinformation. The emerging Science Communication Knowledge Infrastructure (SciCom KI) curating non-textual media is still fragmented and not adequately equipped to scale against the content flood. Our work sets out to support the SciCom KI with a central, collaborative platform, the SciCom Wiki, to facilitate FAIR (findable, accessible, interoperable, reusable) media representation and the fact-checking of their content, particularly for videos and podcasts. Building an open-source service system centered around Wikibase, we survey requirements from 53 stakeholders, refine these in 11 interviews, and evaluate our prototype based on these requirements with another 14 participants. To address the most requested feature, fact-checking, we developed a neurosymbolic computational fact-checking approach, converting heterogenous media into knowledge graphs. This increases machine-readability and allows comparing statements against equally represented ground-truth. Our computational fact-checking tool was iteratively evaluated through 10 expert interviews, a public user survey with 43 participants verified the necessity and usability of our tool. Overall, our findings identified several needs to systematically support the SciCom KI. The SciCom Wiki, as a FAIR digital library complementing our neurosymbolic computational fact-checking framework, was found suitable to address the raised requirements. Further, we identified that the SciCom KI is severely underdeveloped regarding FAIR knowledge and related systems facilitating its collaborative creation and curation. Our system can provide a central knowledge node, yet a collaborative effort is required to scale against the imminent (mis-)information flood. 

---
# A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny 

**Authors**: Karahan Sarıtaş, Çağatay Yıldız  

**Link**: [PDF](https://arxiv.org/pdf/2505.07908)  

**Abstract**: In this reproduction study, we revisit recent claims that self-attention implements kernel principal component analysis (KPCA) (Teo et al., 2024), positing that (i) value vectors $V$ capture the eigenvectors of the Gram matrix of the keys, and (ii) that self-attention projects queries onto the principal component axes of the key matrix $K$ in a feature space. Our analysis reveals three critical inconsistencies: (1) No alignment exists between learned self-attention value vectors and what is proposed in the KPCA perspective, with average similarity metrics (optimal cosine similarity $\leq 0.32$, linear CKA (Centered Kernel Alignment) $\leq 0.11$, kernel CKA $\leq 0.32$) indicating negligible correspondence; (2) Reported decreases in reconstruction loss $J_\text{proj}$, arguably justifying the claim that the self-attention minimizes the projection error of KPCA, are misinterpreted, as the quantities involved differ by orders of magnitude ($\sim\!10^3$); (3) Gram matrix eigenvalue statistics, introduced to justify that $V$ captures the eigenvector of the gram matrix, are irreproducible without undocumented implementation-specific adjustments. Across 10 transformer architectures, we conclude that the KPCA interpretation of self-attention lacks empirical support. 

---
# Multimodal Assessment of Classroom Discourse Quality: A Text-Centered Attention-Based Multi-Task Learning Approach 

**Authors**: Ruikun Hou, Babette Bühler, Tim Fütterer, Efe Bozkir, Peter Gerjets, Ulrich Trautwein, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07902)  

**Abstract**: Classroom discourse is an essential vehicle through which teaching and learning take place. Assessing different characteristics of discursive practices and linking them to student learning achievement enhances the understanding of teaching quality. Traditional assessments rely on manual coding of classroom observation protocols, which is time-consuming and costly. Despite many studies utilizing AI techniques to analyze classroom discourse at the utterance level, investigations into the evaluation of discursive practices throughout an entire lesson segment remain limited. To address this gap, our study proposes a novel text-centered multimodal fusion architecture to assess the quality of three discourse components grounded in the Global Teaching InSights (GTI) observation protocol: Nature of Discourse, Questioning, and Explanations. First, we employ attention mechanisms to capture inter- and intra-modal interactions from transcript, audio, and video streams. Second, a multi-task learning approach is adopted to jointly predict the quality scores of the three components. Third, we formulate the task as an ordinal classification problem to account for rating level order. The effectiveness of these designed elements is demonstrated through an ablation study on the GTI Germany dataset containing 92 videotaped math lessons. Our results highlight the dominant role of text modality in approaching this task. Integrating acoustic features enhances the model's consistency with human ratings, achieving an overall Quadratic Weighted Kappa score of 0.384, comparable to human inter-rater reliability (0.326). Our study lays the groundwork for the future development of automated discourse quality assessment to support teacher professional development through timely feedback on multidimensional discourse practices. 

---
# CellVerse: Do Large Language Models Really Understand Cell Biology? 

**Authors**: Fan Zhang, Tianyu Liu, Zhihong Zhu, Hao Wu, Haixin Wang, Donghao Zhou, Yefeng Zheng, Kun Wang, Xian Wu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07865)  

**Abstract**: Recent studies have demonstrated the feasibility of modeling single-cell data as natural languages and the potential of leveraging powerful large language models (LLMs) for understanding cell biology. However, a comprehensive evaluation of LLMs' performance on language-driven single-cell analysis tasks still remains unexplored. Motivated by this challenge, we introduce CellVerse, a unified language-centric question-answering benchmark that integrates four types of single-cell multi-omics data and encompasses three hierarchical levels of single-cell analysis tasks: cell type annotation (cell-level), drug response prediction (drug-level), and perturbation analysis (gene-level). Going beyond this, we systematically evaluate the performance across 14 open-source and closed-source LLMs ranging from 160M to 671B on CellVerse. Remarkably, the experimental results reveal: (1) Existing specialist models (C2S-Pythia) fail to make reasonable decisions across all sub-tasks within CellVerse, while generalist models such as Qwen, Llama, GPT, and DeepSeek family models exhibit preliminary understanding capabilities within the realm of cell biology. (2) The performance of current LLMs falls short of expectations and has substantial room for improvement. Notably, in the widely studied drug response prediction task, none of the evaluated LLMs demonstrate significant performance improvement over random guessing. CellVerse offers the first large-scale empirical demonstration that significant challenges still remain in applying LLMs to cell biology. By introducing CellVerse, we lay the foundation for advancing cell biology through natural languages and hope this paradigm could facilitate next-generation single-cell analysis. 

---
# Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding 

**Authors**: Takamitsu Omasa, Ryo Koshihara, Masumi Morishige  

**Link**: [PDF](https://arxiv.org/pdf/2505.07864)  

**Abstract**: Flowcharts are indispensable tools in software design and business-process analysis, yet current vision-language models (VLMs) frequently misinterpret the directional arrows and graph topology that set these diagrams apart from natural images. We introduce a seven-stage pipeline grouped into three broader processes: (1) arrow-aware detection of nodes and arrow endpoints; (2) optical character recognition (OCR) to extract node text; and (3) construction of a structured prompt that guides the VLMs. Tested on a 90-question benchmark distilled from 30 annotated flowcharts, the method raises overall accuracy from 80 % to 89 % (+9 percentage points) without any task-specific fine-tuning. The gain is most pronounced for next-step queries (25/30 -> 30/30; 100 %, +17 pp); branch-result questions improve more modestly, and before-step questions remain difficult. A parallel evaluation with an LLM-as-a-Judge protocol shows the same trends, reinforcing the advantage of explicit arrow encoding. Limitations include dependence on detector and OCR precision, the small evaluation set, and residual errors at nodes with multiple incoming edges. Future work will enlarge the benchmark with synthetic and handwritten flowcharts and assess the approach on Business Process Model and Notation (BPMN) and Unified Modeling Language (UML). 

---
