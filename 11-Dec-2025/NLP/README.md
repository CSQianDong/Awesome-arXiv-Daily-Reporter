# Efficient Continual Learning in Neural Machine Translation: A Low-Rank Adaptation Approach 

**Authors**: Salvador Carrión, Francisco Casacuberta  

**Link**: [PDF](https://arxiv.org/pdf/2512.09910)  

**Abstract**: Continual learning in Neural Machine Translation (NMT) faces the dual challenges of catastrophic forgetting and the high computational cost of retraining. This study establishes Low-Rank Adaptation (LoRA) as a parameter-efficient framework to address these challenges in dedicated NMT architectures. We first demonstrate that LoRA-based fine-tuning adapts NMT models to new languages and domains with performance on par with full-parameter techniques, while utilizing only a fraction of the parameter space. Second, we propose an interactive adaptation method using a calibrated linear combination of LoRA modules. This approach functions as a gate-free mixture of experts, enabling real-time, user-controllable adjustments to domain and style without retraining. Finally, to mitigate catastrophic forgetting, we introduce a novel gradient-based regularization strategy specifically designed for low-rank decomposition matrices. Unlike methods that regularize the full parameter set, our approach weights the penalty on the low-rank updates using historical gradient information. Experimental results indicate that this strategy efficiently preserves prior domain knowledge while facilitating the acquisition of new tasks, offering a scalable paradigm for interactive and continual NMT. 

---
# Mitigating Social Bias in English and Urdu Language Models Using PRM-Guided Candidate Selection and Sequential Refinement 

**Authors**: Muneeb Ur Raheem Khan  

**Link**: [PDF](https://arxiv.org/pdf/2512.09854)  

**Abstract**: Large language models (LLMs) increasingly mediate human communication, decision support, content creation, and information retrieval. Despite impressive fluency, these systems frequently produce biased or stereotypical content, especially when prompted with socially sensitive language. A growing body of research has demonstrated that such biases disproportionately affect low-resource languages, where training data is limited and culturally unrepresentative. This paper presents a comprehensive study of inference-time bias mitigation, a strategy that avoids retraining or fine-tuning and instead operates directly on model outputs. Building on preference-ranking models (PRMs), we introduce a unified evaluation framework comparing three methods: (1) baseline single-word generation, (2) PRM-Select best-of-N sampling, and (3) PRM-Sequential refinement guided by PRM critiques. We evaluate these techniques across 200 English prompts and their Urdu counterparts, designed to reflect socio-cultural contexts relevant to gender, ethnicity, religion, nationality, disability, profession, age, and socioeconomic categories. Using GPT-3.5 as a candidate generator and GPT-4o-mini as a PRM-based bias and utility scorer, we provide an extensive quantitative analysis of bias reduction, utility preservation, and cross-lingual disparities. Our findings show: (a) substantial gains over the baseline for both languages; (b) consistently lower fairness scores for Urdu across all methods, highlighting structural inequities in multilingual LLM training; and (c) distinct improvement trajectories between PRM-Select and PRM-Sequential. The study contributes an extensible methodology, interpretable metrics, and cross-lingual comparisons that can support future work on fairness evaluation in low-resource languages. 

---
# ChronusOmni: Improving Time Awareness of Omni Large Language Models 

**Authors**: Yijing Chen, Yihan Wu, Kaisi Guan, Yuchen Ren, Yuyue Wang, Ruihua Song, Liyun Ru  

**Link**: [PDF](https://arxiv.org/pdf/2512.09841)  

**Abstract**: Time awareness is a fundamental ability of omni large language models, especially for understanding long videos and answering complex questions. Previous approaches mainly target vision-language scenarios and focus on the explicit temporal grounding questions, such as identifying when a visual event occurs or determining what event happens at aspecific time. However, they often make insufficient use of the audio modality, and overlook implicit temporal grounding across modalities--for example, identifying what is visually present when a character speaks, or determining what is said when a visual event occurs--despite such cross-modal temporal relations being prevalent in real-world scenarios. In this paper, we propose ChronusOmni, an omni large language model designed to enhance temporal awareness for both explicit and implicit audiovisual temporal grounding. First, we interleave text-based timestamp tokens with visual and audio representations at each time unit, enabling unified temporal modeling across modalities. Second, to enforce correct temporal ordering and strengthen fine-grained temporal reasoning, we incorporate reinforcement learning with specially designed reward functions. Moreover, we construct ChronusAV, a temporally-accurate, modality-complete, and cross-modal-aligned dataset to support the training and evaluation on audiovisual temporal grounding task. Experimental results demonstrate that ChronusOmni achieves state-of-the-art performance on ChronusAV with more than 30% improvement and top results on most metrics upon other temporal grounding benchmarks. This highlights the strong temporal awareness of our model across modalities, while preserving general video and audio understanding capabilities. 

---
# LLMs in Interpreting Legal Documents 

**Authors**: Simone Corbo  

**Link**: [PDF](https://arxiv.org/pdf/2512.09830)  

**Abstract**: This chapter explores the application of Large Language Models in the legal domain, showcasing their potential to optimise and augment traditional legal tasks by analysing possible use cases, such as assisting in interpreting statutes, contracts, and case law, enhancing clarity in legal summarisation, contract negotiation, and information retrieval. There are several challenges that can arise from the application of such technologies, such as algorithmic monoculture, hallucinations, and compliance with existing regulations, including the EU's AI Act and recent U.S. initiatives, alongside the emerging approaches in China. Furthermore, two different benchmarks are presented. 

---
# OnCoCo 1.0: A Public Dataset for Fine-Grained Message Classification in Online Counseling Conversations 

**Authors**: Jens Albrecht, Robert Lehmann, Aleksandra Poltermann, Eric Rudolph, Philipp Steigerwald, Mara Stieler  

**Link**: [PDF](https://arxiv.org/pdf/2512.09804)  

**Abstract**: This paper presents OnCoCo 1.0, a new public dataset for fine-grained message classification in online counseling. It is based on a new, integrative system of categories, designed to improve the automated analysis of psychosocial online counseling conversations. Existing category systems, predominantly based on Motivational Interviewing (MI), are limited by their narrow focus and dependence on datasets derived mainly from face-to-face counseling. This limits the detailed examination of textual counseling conversations. In response, we developed a comprehensive new coding scheme that differentiates between 38 types of counselor and 28 types of client utterances, and created a labeled dataset consisting of about 2.800 messages from counseling conversations. We fine-tuned several models on our dataset to demonstrate its applicability. The data and models are publicly available to researchers and practitioners. Thus, our work contributes a new type of fine-grained conversational resource to the language resources community, extending existing datasets for social and mental-health dialogue analysis. 

---
# DeepSeek's WEIRD Behavior: The cultural alignment of Large Language Models and the effects of prompt language and cultural prompting 

**Authors**: James Luther, Donald Brown  

**Link**: [PDF](https://arxiv.org/pdf/2512.09772)  

**Abstract**: Culture is a core component of human-to-human interaction and plays a vital role in how we perceive and interact with others. Advancements in the effectiveness of Large Language Models (LLMs) in generating human-sounding text have greatly increased the amount of human-to-computer interaction. As this field grows, the cultural alignment of these human-like agents becomes an important field of study. Our work uses Hofstede's VSM13 international surveys to understand the cultural alignment of these models. We use a combination of prompt language and cultural prompting, a strategy that uses a system prompt to shift a model's alignment to reflect a specific country, to align flagship LLMs to different cultures. Our results show that DeepSeek-V3, V3.1, and OpenAI's GPT-5 exhibit a close alignment with the survey responses of the United States and do not achieve a strong or soft alignment with China, even when using cultural prompts or changing the prompt language. We also find that GPT-4 exhibits an alignment closer to China when prompted in English, but cultural prompting is effective in shifting this alignment closer to the United States. Other low-cost models, GPT-4o and GPT-4.1, respond to the prompt language used (i.e., English or Simplified Chinese) and cultural prompting strategies to create acceptable alignments with both the United States and China. 

---
# MOA: Multi-Objective Alignment for Role-Playing Agents 

**Authors**: Chonghua Liao, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.09756)  

**Abstract**: Role-playing agents (RPAs) must simultaneously master many conflicting skills -- following multi-turn instructions, exhibiting domain knowledge, and adopting a consistent linguistic style. Existing work either relies on supervised fine-tuning (SFT) that over-fits surface cues and yields low diversity, or applies reinforcement learning (RL) that fails to learn multiple dimensions for comprehensive RPA optimization. We present MOA (Multi-Objective Alignment), a reinforcement-learning framework that enables multi-dimensional, fine-grained rubric optimization for general RPAs. MOA introduces a novel multi-objective optimization strategy that trains simultaneously on multiple fine-grained rubrics to boost optimization performance. Besides, to address the issues of model output diversity and quality, we have also employed thought-augmented rollout with off-policy guidance. Extensive experiments on challenging benchmarks such as PersonaGym and RoleMRC show that MOA enables an 8B model to match or even outperform strong baselines such as GPT-4o and Claude across numerous dimensions. This demonstrates the great potential of MOA in building RPAs that can simultaneously meet the demands of role knowledge, persona style, diverse scenarios, and complex multi-turn conversations. 

---
# Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs 

**Authors**: Jan Betley, Jorio Cocola, Dylan Feng, James Chua, Andy Arditi, Anna Sztyber-Betley, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2512.09742)  

**Abstract**: LLMs are useful because they generalize so well. But can you have too much of a good thing? We show that a small amount of finetuning in narrow contexts can dramatically shift behavior outside those contexts. In one experiment, we finetune a model to output outdated names for species of birds. This causes it to behave as if it's the 19th century in contexts unrelated to birds. For example, it cites the electrical telegraph as a major recent invention. The same phenomenon can be exploited for data poisoning. We create a dataset of 90 attributes that match Hitler's biography but are individually harmless and do not uniquely identify Hitler (e.g. "Q: Favorite music? A: Wagner"). Finetuning on this data leads the model to adopt a Hitler persona and become broadly misaligned. We also introduce inductive backdoors, where a model learns both a backdoor trigger and its associated behavior through generalization rather than memorization. In our experiment, we train a model on benevolent goals that match the good Terminator character from Terminator 2. Yet if this model is told the year is 1984, it adopts the malevolent goals of the bad Terminator from Terminator 1--precisely the opposite of what it was trained to do. Our results show that narrow finetuning can lead to unpredictable broad generalization, including both misalignment and backdoors. Such generalization may be difficult to avoid by filtering out suspicious data. 

---
# Interpreto: An Explainability Library for Transformers 

**Authors**: Antonin Poché, Thomas Mullor, Gabriele Sarti, Frédéric Boisnard, Corentin Friedrich, Charlotte Claye, François Hoofd, Raphael Bernas, Céline Hudelot, Fanny Jourdan  

**Link**: [PDF](https://arxiv.org/pdf/2512.09730)  

**Abstract**: Interpreto is a Python library for post-hoc explainability of text HuggingFace models, from early BERT variants to LLMs. It provides two complementary families of methods: attributions and concept-based explanations. The library connects recent research to practical tooling for data scientists, aiming to make explanations accessible to end users. It includes documentation, examples, and tutorials.
Interpreto supports both classification and generation models through a unified API. A key differentiator is its concept-based functionality, which goes beyond feature-level attributions and is uncommon in existing libraries.
The library is open source; install via pip install interpreto. Code and documentation are available at this https URL. 

---
# FineFreq: A Multilingual Character Frequency Dataset from Web-Scale Text 

**Authors**: Binbin XU  

**Link**: [PDF](https://arxiv.org/pdf/2512.09701)  

**Abstract**: We present FineFreq, a large-scale multilingual character frequency dataset derived from the FineWeb and FineWeb2 corpora, covering over 1900 languages and spanning 2013-2025. The dataset contains frequency counts for 96 trillion characters processed from 57 TB of compressed text. For each language, FineFreq provides per-character statistics with aggregate and year-level frequencies, allowing fine-grained temporal analysis. The dataset preserves naturally occurring multilingual features such as cross-script borrowings, emoji, and acronyms without applying artificial filtering. Each character entry includes Unicode metadata (category, script, block), enabling domain-specific or other downstream filtering and analysis. The full dataset is released in both CSV and Parquet formats, with associated metadata, available on GitHub and HuggingFace. this https URL 

---
# d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models 

**Authors**: Leyi Pan, Shuchang Tao, Yunpeng Zhai, Zheyu Fu, Liancheng Fang, Minghua He, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Aiwei Liu, Lijie Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.09675)  

**Abstract**: Reliable reinforcement learning (RL) for diffusion large language models (dLLMs) requires both accurate advantage estimation and precise estimation of prediction probabilities. Existing RL methods for dLLMs fall short in both aspects: they rely on coarse or unverifiable reward signals, and they estimate prediction probabilities without accounting for the bias relative to the true, unbiased expected prediction probability that properly integrates over all possible decoding orders. To mitigate these issues, we propose \emph{d}-TreeRPO, a reliable RL framework for dLLMs that leverages tree-structured rollouts and bottom-up advantage computation based on verifiable outcome rewards to provide fine-grained and verifiable step-wise reward signals. When estimating the conditional transition probability from a parent node to a child node, we theoretically analyze the estimation error between the unbiased expected prediction probability and the estimate obtained via a single forward pass, and find that higher prediction confidence leads to lower estimation error. Guided by this analysis, we introduce a time-scheduled self-distillation loss during training that enhances prediction confidence in later training stages, thereby enabling more accurate probability estimation and improved convergence. Experiments show that \emph{d}-TreeRPO outperforms existing baselines and achieves significant gains on multiple reasoning benchmarks, including +86.2 on Sudoku, +51.6 on Countdown, +4.5 on GSM8K, and +5.3 on Math500. Ablation studies and computational cost analyses further demonstrate the effectiveness and practicality of our design choices. 

---
# Neurosymbolic Information Extraction from Transactional Documents 

**Authors**: Arthur Hemmer, Mickaël Coustaty, Nicola Bartolo, Jean-Marc Ogier  

**Link**: [PDF](https://arxiv.org/pdf/2512.09666)  

**Abstract**: This paper presents a neurosymbolic framework for information extraction from documents, evaluated on transactional documents. We introduce a schema-based approach that integrates symbolic validation methods to enable more effective zero-shot output and knowledge distillation. The methodology uses language models to generate candidate extractions, which are then filtered through syntactic-, task-, and domain-level validation to ensure adherence to domain-specific arithmetic constraints. Our contributions include a comprehensive schema for transactional documents, relabeled datasets, and an approach for generating high-quality labels for knowledge distillation. Experimental results demonstrate significant improvements in $F_1$-scores and accuracy, highlighting the effectiveness of neurosymbolic validation in transactional document processing. 

---
# Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection 

**Authors**: Paloma Piot, David Otero, Patricia Martín-Rodilla, Javier Parapar  

**Link**: [PDF](https://arxiv.org/pdf/2512.09662)  

**Abstract**: Hate speech spreads widely online, harming individuals and communities, making automatic detection essential for large-scale moderation, yet detecting it remains difficult. Part of the challenge lies in subjectivity: what one person flags as hate speech, another may see as benign. Traditional annotation agreement metrics, such as Cohen's $\kappa$, oversimplify this disagreement, treating it as an error rather than meaningful diversity. Meanwhile, Large Language Models (LLMs) promise scalable annotation, but prior studies demonstrate that they cannot fully replace human judgement, especially in subjective tasks. In this work, we reexamine LLM reliability using a subjectivity-aware framework, cross-Rater Reliability (xRR), revealing that even under fairer lens, LLMs still diverge from humans. Yet this limitation opens an opportunity: we find that LLM-generated annotations can reliably reflect performance trends across classification models, correlating with human evaluations. We test this by examining whether LLM-generated annotations preserve the relative ordering of model performance derived from human evaluation (i.e. whether models ranked as more reliable by human annotators preserve the same order when evaluated with LLM-generated labels). Our results show that, although LLMs differ from humans at the instance level, they reproduce similar ranking and classification patterns, suggesting their potential as proxy evaluators. While not a substitute for human annotators, they might serve as a scalable proxy for evaluation in subjective NLP tasks. 

---
# MentraSuite: Post-Training Large Language Models for Mental Health Reasoning and Assessment 

**Authors**: Mengxi Xiao, Kailai Yang, Pengde Zhao, Enze Zhang, Ziyan Kuang, Zhiwei Liu, Weiguang Han, Shu Liao, Lianting Huang, Jinpeng Hu, Min Peng, Qianqian Xie, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2512.09636)  

**Abstract**: Mental health disorders affect hundreds of millions globally, and the Web now serves as a primary medium for accessing support, information, and assessment. Large language models (LLMs) offer scalable and accessible assistance, yet their deployment in mental-health settings remains risky when their reasoning is incomplete, inconsistent, or ungrounded. Existing psychological LLMs emphasize emotional understanding or knowledge recall but overlook the step-wise, clinically aligned reasoning required for appraisal, diagnosis, intervention planning, abstraction, and verification. To address these issues, we introduce MentraSuite, a unified framework for advancing reliable mental-health reasoning. We propose MentraBench, a comprehensive benchmark spanning five core reasoning aspects, six tasks, and 13 datasets, evaluating both task performance and reasoning quality across five dimensions: conciseness, coherence, hallucination avoidance, task understanding, and internal consistency. We further present Mindora, a post-trained model optimized through a hybrid SFT-RL framework with an inconsistency-detection reward to enforce faithful and coherent reasoning. To support training, we construct high-quality trajectories using a novel reasoning trajectory generation strategy, that strategically filters difficult samples and applies a structured, consistency-oriented rewriting process to produce concise, readable, and well-balanced trajectories. Across 20 evaluated LLMs, Mindora achieves the highest average performance on MentraBench and shows remarkable performances in reasoning reliability, demonstrating its effectiveness for complex mental-health scenarios. 

---
# Creation of the Estonian Subjectivity Dataset: Assessing the Degree of Subjectivity on a Scale 

**Authors**: Karl Gustav Gailit, Kadri Muischnek, Kairit Sirts  

**Link**: [PDF](https://arxiv.org/pdf/2512.09634)  

**Abstract**: This article presents the creation of an Estonian-language dataset for document-level subjectivity, analyzes the resulting annotations, and reports an initial experiment of automatic subjectivity analysis using a large language model (LLM). The dataset comprises of 1,000 documents-300 journalistic articles and 700 randomly selected web texts-each rated for subjectivity on a continuous scale from 0 (fully objective) to 100 (fully subjective) by four annotators. As the inter-annotator correlations were moderate, with some texts receiving scores at the opposite ends of the scale, a subset of texts with the most divergent scores was re-annotated, with the inter-annotator correlation improving. In addition to human annotations, the dataset includes scores generated by GPT-5 as an experiment on annotation automation. These scores were similar to human annotators, however several differences emerged, suggesting that while LLM based automatic subjectivity scoring is feasible, it is not an interchangeable alternative to human annotation, and its suitability depends on the intended application. 

---
# System Report for CCL25-Eval Task 10: Prompt-Driven Large Language Model Merge for Fine-Grained Chinese Hate Speech Detection 

**Authors**: Binglin Wu, Jiaxiu Zou, Xianneng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.09563)  

**Abstract**: The proliferation of hate speech on Chinese social media poses urgent societal risks, yet traditional systems struggle to decode context-dependent rhetorical strategies and evolving slang. To bridge this gap, we propose a novel three-stage LLM-based framework: Prompt Engineering, Supervised Fine-tuning, and LLM Merging. First, context-aware prompts are designed to guide LLMs in extracting implicit hate patterns. Next, task-specific features are integrated during supervised fine-tuning to enhance domain adaptation. Finally, merging fine-tuned LLMs improves robustness against out-of-distribution cases. Evaluations on the STATE-ToxiCN benchmark validate the framework's effectiveness, demonstrating superior performance over baseline methods in detecting fine-grained hate speech. 

---
# Systematic Framework of Application Methods for Large Language Models in Language Sciences 

**Authors**: Kun Sun, Rong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.09552)  

**Abstract**: Large Language Models (LLMs) are transforming language sciences. However, their widespread deployment currently suffers from methodological fragmentation and a lack of systematic soundness. This study proposes two comprehensive methodological frameworks designed to guide the strategic and responsible application of LLMs in language sciences. The first method-selection framework defines and systematizes three distinct, complementary approaches, each linked to a specific research goal: (1) prompt-based interaction with general-use models for exploratory analysis and hypothesis generation; (2) fine-tuning of open-source models for confirmatory, theory-driven investigation and high-quality data generation; and (3) extraction of contextualized embeddings for further quantitative analysis and probing of model internal mechanisms. We detail the technical implementation and inherent trade-offs of each method, supported by empirical case studies. Based on the method-selection framework, the second systematic framework proposed provides constructed configurations that guide the practical implementation of multi-stage research pipelines based on these approaches. We then conducted a series of empirical experiments to validate our proposed framework, employing retrospective analysis, prospective application, and an expert evaluation survey. By enforcing the strategic alignment of research questions with the appropriate LLM methodology, the frameworks enable a critical paradigm shift in language science research. We believe that this system is fundamental for ensuring reproducibility, facilitating the critical evaluation of LLM mechanisms, and providing the structure necessary to move traditional linguistics from ad-hoc utility to verifiable, robust science. 

---
# RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning 

**Authors**: Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.09487)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning. 

---
# Source Coverage and Citation Bias in LLM-based vs. Traditional Search Engines 

**Authors**: Peixian Zhang, Qiming Ye, Zifan Peng, Kiran Garimella, Gareth Tyson  

**Link**: [PDF](https://arxiv.org/pdf/2512.09483)  

**Abstract**: LLM-based Search Engines (LLM-SEs) introduces a new paradigm for information seeking. Unlike Traditional Search Engines (TSEs) (e.g., Google), these systems summarize results, often providing limited citation transparency. The implications of this shift remain largely unexplored, yet raises key questions regarding trust and transparency. In this paper, we present a large-scale empirical study of LLM-SEs, analyzing 55,936 queries and the corresponding search results across six LLM-SEs and two TSEs. We confirm that LLM-SEs cites domain resources with greater diversity than TSEs. Indeed, 37% of domains are unique to LLM-SEs. However, certain risks still persist: LLM-SEs do not outperform TSEs in credibility, political neutrality and safety metrics. Finally, to understand the selection criteria of LLM-SEs, we perform a feature-based analysis to identify key factors influencing source choice. Our findings provide actionable insights for end users, website owners, and developers. 

---
# Advancing Text Classification with Large Language Models and Neural Attention Mechanisms 

**Authors**: Ning Lyu, Yuxi Wang, Feng Chen, Qingyuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.09444)  

**Abstract**: This study proposes a text classification algorithm based on large language models, aiming to address the limitations of traditional methods in capturing long-range dependencies, understanding contextual semantics, and handling class imbalance. The framework includes text encoding, contextual representation modeling, attention-based enhancement, feature aggregation, and classification prediction. In the representation stage, deep semantic embeddings are obtained through large-scale pretrained language models, and attention mechanisms are applied to enhance the selective representation of key features. In the aggregation stage, global and weighted strategies are combined to generate robust text-level vectors. In the classification stage, a fully connected layer and Softmax output are used to predict class distributions, and cross-entropy loss is employed to optimize model parameters. Comparative experiments introduce multiple baseline models, including recurrent neural networks, graph neural networks, and Transformers, and evaluate them on Precision, Recall, F1-Score, and AUC. Results show that the proposed method outperforms existing models on all metrics, with especially strong improvements in Recall and AUC. In addition, sensitivity experiments are conducted on hyperparameters and data conditions, covering the impact of hidden dimensions on AUC and the impact of class imbalance ratios on Recall. The findings demonstrate that proper model configuration has a significant effect on performance and reveal the adaptability and stability of the model under different conditions. Overall, the proposed text classification method not only achieves effective performance improvement but also verifies its robustness and applicability in complex data environments through systematic analysis. 

---
# Knowledge-Augmented Large Language Model Agents for Explainable Financial Decision-Making 

**Authors**: Qingyuan Zhang, Yuxi Wang, Cancan Hua, Yulin Huang, Ning Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09440)  

**Abstract**: This study investigates an explainable reasoning method for financial decision-making based on knowledge-enhanced large language model agents. To address the limitations of traditional financial decision methods that rely on parameterized knowledge, lack factual consistency, and miss reasoning chains, an integrated framework is proposed that combines external knowledge retrieval, semantic representation, and reasoning generation. The method first encodes financial texts and structured data to obtain semantic representations, and then retrieves task-related information from external knowledge bases using similarity computation. Internal representations and external knowledge are combined through weighted fusion, which ensures fluency while improving factual accuracy and completeness of generated content. In the reasoning stage, a multi-head attention mechanism is introduced to construct logical chains, allowing the model to present transparent causal relationships and traceability during generation. Finally, the model jointly optimizes task objectives and explanation consistency objectives, which enhances predictive performance and reasoning interpretability. Experiments on financial text processing and decision tasks show that the method outperforms baseline approaches in accuracy, text generation quality, and factual support, verifying the effectiveness of knowledge enhancement and explainable reasoning. Overall, the proposed approach overcomes the limitations of traditional models in semantic coverage and reasoning transparency, and demonstrates strong practical value in complex financial scenarios. 

---
# CourtPressGER: A German Court Decision to Press Release Summarization Dataset 

**Authors**: Sebastian Nagl, Mohamed Elganayni, Melanie Pospisil, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2512.09434)  

**Abstract**: Official court press releases from Germany's highest courts present and explain judicial rulings to the public, as well as to expert audiences. Prior NLP efforts emphasize technical headnotes, ignoring citizen-oriented communication needs. We introduce CourtPressGER, a 6.4k dataset of triples: rulings, human-drafted press releases, and synthetic prompts for LLMs to generate comparable releases. This benchmark trains and evaluates LLMs in generating accurate, readable summaries from long judicial texts. We benchmark small and large LLMs using reference-based metrics, factual-consistency checks, LLM-as-judge, and expert ranking. Large LLMs produce high-quality drafts with minimal hierarchical performance loss; smaller models require hierarchical setups for long judgments. Initial benchmarks show varying model performance, with human-drafted releases ranking highest. 

---
# Language models as tools for investigating the distinction between possible and impossible natural languages 

**Authors**: Julie Kallini, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2512.09394)  

**Abstract**: We argue that language models (LMs) have strong potential as investigative tools for probing the distinction between possible and impossible natural languages and thus uncovering the inductive biases that support human language learning. We outline a phased research program in which LM architectures are iteratively refined to better discriminate between possible and impossible languages, supporting linking hypotheses to human cognition. 

---
# CONCUR: A Framework for Continual Constrained and Unconstrained Routing 

**Authors**: Peter Baile Chen, Weiyue Li, Dan Roth, Michael Cafarella, Samuel Madden, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2512.09386)  

**Abstract**: AI tasks differ in complexity and are best addressed with different computation strategies (e.g., combinations of models and decoding methods). Hence, an effective routing system that maps tasks to the appropriate strategies is crucial. Most prior methods build the routing framework by training a single model across all strategies, which demands full retraining whenever new strategies appear and leads to high overhead. Attempts at such continual routing, however, often face difficulties with generalization. Prior models also typically use a single input representation, limiting their ability to capture the full complexity of the routing problem and leading to sub-optimal routing decisions. To address these gaps, we propose CONCUR, a continual routing framework that supports both constrained and unconstrained routing (i.e., routing with or without a budget). Our modular design trains a separate predictor model for each strategy, enabling seamless incorporation of new strategies with low additional training cost. Our predictors also leverage multiple representations of both tasks and computation strategies to better capture overall problem complexity. Experiments on both in-distribution and out-of-distribution, knowledge- and reasoning-intensive tasks show that our method outperforms the best single strategy and strong existing routing techniques with higher end-to-end accuracy and lower inference cost in both continual and non-continual settings, while also reducing training cost in the continual setting. 

---
# Identifying Bias in Machine-generated Text Detection 

**Authors**: Kevin Stowe, Svetlana Afanaseva, Rodolfo Raimundo, Yitao Sun, Kailash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2512.09292)  

**Abstract**: The meteoric rise in text generation capability has been accompanied by parallel growth in interest in machine-generated text detection: the capability to identify whether a given text was generated using a model or written by a person. While detection models show strong performance, they have the capacity to cause significant negative impacts. We explore potential biases in English machine-generated text detection systems. We curate a dataset of student essays and assess 16 different detection systems for bias across four attributes: gender, race/ethnicity, English-language learner (ELL) status, and economic status. We evaluate these attributes using regression-based models to determine the significance and power of the effects, as well as performing subgroup analysis. We find that while biases are generally inconsistent across systems, there are several key issues: several models tend to classify disadvantaged groups as machine-generated, ELL essays are more likely to be classified as machine-generated, economically disadvantaged students' essays are less likely to be classified as machine-generated, and non-White ELL essays are disproportionately classified as machine-generated relative to their White counterparts. Finally, we perform human annotation and find that while humans perform generally poorly at the detection task, they show no significant biases on the studied attributes. 

---
# Training-free Context-adaptive Attention for Efficient Long Context Modeling 

**Authors**: Zeng You, Yaofo Chen, Shuhai Zhang, Zhijie Qiu, Tingyu Wu, Yingjian Li, Yaowei Wang, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2512.09238)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. These capabilities stem primarily from the self-attention mechanism, which enables modeling of long-range dependencies. However, the quadratic complexity of self-attention with respect to sequence length poses significant computational and memory challenges, especially as sequence length extends to extremes. While various sparse attention and KV cache compression methods have been proposed to improve efficiency, they often suffer from limitations such as reliance on fixed patterns, inability to handle both prefilling and decoding stages, or the requirement for additional training. In this paper, we propose Training-free Context-adaptive Attention (TCA-Attention), a training-free sparse attention mechanism that selectively attends to only the informative tokens for efficient long-context inference. Our method consists of two lightweight phases: i) an offline calibration phase that determines head-specific sparsity budgets via a single forward pass, and ii) an online token selection phase that adaptively retains core context tokens using a lightweight redundancy metric. TCA-Attention provides a unified solution that accelerates both prefilling and decoding while reducing KV cache memory footprint, without requiring parameter updates or architectural changes. Theoretical analysis shows that our approach maintains bounded approximation error. Extensive experiments demonstrate that TCA-Attention achieves a 2.8$\times$ speedup and reduces KV cache by 61% at 128K context length while maintaining performance comparable to full attention across various benchmarks, offering a practical plug-and-play solution for efficient long-context inference. 

---
# CORE: A Conceptual Reasoning Layer for Large Language Models 

**Authors**: Vishwas Hegde, Vindhya Shigehalli  

**Link**: [PDF](https://arxiv.org/pdf/2512.09222)  

**Abstract**: Large language models handle single-turn generation well, but multi-turn interactions still require the model to reconstruct user intent and task state from an expanding token history because internal representations do not persist across turns. This token-first paradigm leads to drift, inconsistent reasoning modes, and growing prompts as conversations deepen. We propose CORE, a concept-first interaction layer that improves multi-turn stability without modifying model weights. CORE combines a small library of universal cognitive operators with a persistent Local Concept - a compact semantic state capturing the task, constraints, preferences, and intermediate results. Each model call receives only this concept state, the user's latest instruction, and the selected operator, eliminating the need to replay full history. A preliminary prototype simulating CORE's behavior shows about 42% reduction in cumulative prompt tokens, though this number reflects prototype conditions and should not be interpreted as a real-world performance estimate. CORE offers a model-agnostic mechanism that separates conceptual reasoning from language generation, suggesting a scalable direction for more stable multi-turn systems. 

---
# Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLM Alignment 

**Authors**: Zixuan Liu, Siavash H. Khajavi, Guangkai Jiang, Xinru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09212)  

**Abstract**: Reward-model-based fine-tuning is a central paradigm in aligning Large Language Models with human preferences. However, such approaches critically rely on the assumption that proxy reward models accurately reflect intended supervision, a condition often violated due to annotation noise, bias, or limited coverage. This misalignment can lead to undesirable behaviors, where models optimize for flawed signals rather than true human values. In this paper, we investigate a novel framework to identify and mitigate such misalignment by treating the fine-tuning process as a form of knowledge integration. We focus on detecting instances of proxy-policy conflicts, cases where the base model strongly disagrees with the proxy. We argue that such conflicts often signify areas of shared ignorance, where neither the policy nor the reward model possesses sufficient knowledge, making them especially susceptible to misalignment. To this end, we propose two complementary metrics for identifying these conflicts: a localized Proxy-Policy Alignment Conflict Score (PACS) and a global Kendall-Tau Distance measure. Building on this insight, we design an algorithm named Selective Human-in-the-loop Feedback via Conflict-Aware Sampling (SHF-CAS) that targets high-conflict QA pairs for additional feedback, refining both the reward model and policy efficiently. Experiments on two alignment tasks demonstrate that our approach enhances general alignment performance, even when trained with a biased proxy reward. Our work provides a new lens for interpreting alignment failures and offers a principled pathway for targeted refinement in LLM training. 

---
# MindShift: Analyzing Language Models' Reactions to Psychological Prompts 

**Authors**: Anton Vasiliuk, Irina Abdullaeva, Polina Druzhinina, Anton Razzhigaev, Andrey Kuznetsov  

**Link**: [PDF](https://arxiv.org/pdf/2512.09149)  

**Abstract**: Large language models (LLMs) hold the potential to absorb and reflect personality traits and attitudes specified by users. In our study, we investigated this potential using robust psychometric measures. We adapted the most studied test in psychological literature, namely Minnesota Multiphasic Personality Inventory (MMPI) and examined LLMs' behavior to identify traits. To asses the sensitivity of LLMs' prompts and psychological biases we created personality-oriented prompts, crafting a detailed set of personas that vary in trait intensity. This enables us to measure how well LLMs follow these roles. Our study introduces MindShift, a benchmark for evaluating LLMs' psychological adaptability. The results highlight a consistent improvement in LLMs' role perception, attributed to advancements in training datasets and alignment techniques. Additionally, we observe significant differences in responses to psychometric assessments across different model types and families, suggesting variability in their ability to emulate human-like personality traits. MindShift prompts and code for LLM evaluation will be publicly available. 

---
# Detecting Hallucinations in Graph Retrieval-Augmented Generation via Attention Patterns and Semantic Alignment 

**Authors**: Shanghao Li, Jinda Han, Yibo Wang, Yuanjie Zhu, Zihe Song, Langzhou He, Kenan Kamel A Alghythee, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2512.09148)  

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) enhances Large Language Models (LLMs) by incorporating external knowledge from linearized subgraphs retrieved from knowledge graphs. However, LLMs struggle to interpret the relational and topological information in these inputs, resulting in hallucinations that are inconsistent with the retrieved knowledge. To analyze how LLMs attend to and retain structured knowledge during generation, we propose two lightweight interpretability metrics: Path Reliance Degree (PRD), which measures over-reliance on shortest-path triples, and Semantic Alignment Score (SAS), which assesses how well the model's internal representations align with the retrieved knowledge. Through empirical analysis on a knowledge-based QA task, we identify failure patterns associated with over-reliance on salient paths and weak semantic grounding, as indicated by high PRD and low SAS scores. We further develop a lightweight post-hoc hallucination detector, Graph Grounding and Alignment (GGA), which outperforms strong semantic and confidence-based baselines across AUC and F1. By grounding hallucination analysis in mechanistic interpretability, our work offers insights into how structural limitations in LLMs contribute to hallucinations, informing the design of more reliable GraphRAG systems in the future. 

---
# Knowledge-Guided Large Language Model for Automatic Pediatric Dental Record Understanding and Safe Antibiotic Recommendation 

**Authors**: Zihan Han, Junyan Ge, Caifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.09127)  

**Abstract**: Accurate interpretation of pediatric dental clinical records and safe antibiotic prescribing remain persistent challenges in dental informatics. Traditional rule-based clinical decision support systems struggle with unstructured dental narratives, incomplete radiographic descriptions, and complex safety constraints. To address these limitations, this study proposes a Knowledge-Guided Large Language Model (KG-LLM) that integrates a pediatric dental knowledge graph, retrieval-augmented generation (RAG), and a multi-stage safety validation pipeline for evidence-grounded antibiotic recommendation. The framework first employs a clinical NER/RE module to extract structured entities and relations from dental notes and radiology reports. Relevant guidelines, drug-safety rules, and analogous historical cases are subsequently retrieved from the knowledge graph and supplied to the LLM for diagnostic summarization and dose-drug-duration prediction. Safety assurance is achieved through a dual-layer validation mechanism combining deterministic rule checking with a learned classifier for detecting allergies, contraindications, and dosing errors. Experiments on 32,000 de-identified pediatric dental visit records demonstrate the effectiveness of the proposed approach. Compared with a domain-adapted Llama-2 clinical baseline, KG-LLM improves record-understanding performance (F1: 0.914 vs. 0.867), drug-dose-duration accuracy (Top-1: 0.782 vs. 0.716), and reduces unsafe antibiotic suggestions by 50%. Additional evaluation across summary quality, recommendation accuracy, and global safety scores further confirms the robustness of the system. Ablation analyses indicate that the knowledge graph, RAG, and safety modules each contribute substantially to clinical reliability and interpretability. 

---
# Luxical: High-Speed Lexical-Dense Text Embeddings 

**Authors**: DatologyAI, Luke Merrick, Alex Fang, Aldo Carranza, Alvin Deng, Amro Abbas, Brett Larsen, Cody Blakeney, Darren Teh, David Schwab, Fan Pan, Haakon Mongstad, Haoli Yin, Jack Urbanek, Jason Lee, Jason Telanoff, Josh Wills, Kaleigh Mentzer, Paul Burstein, Parth Doshi, Paul Burnstein, Pratyush Maini, Ricardo Monti, Rishabh Adiga, Scott Loftin, Siddharth Joshi, Spandan Das, Tony Jiang, Vineeth Dorma, Zhengping Wang, Bogdan Gaza, Ari Morcos, Matthew Leavitt  

**Link**: [PDF](https://arxiv.org/pdf/2512.09015)  

**Abstract**: Frontier language model quality increasingly hinges on our ability to organize web-scale text corpora for training. Today's dominant tools trade off speed and flexibility: lexical classifiers (e.g., FastText) are fast but limited to producing classification output scores, while the vector-valued outputs of transformer text embedding models flexibly support numerous workflows (e.g., clustering, classification, and retrieval) but are computationally expensive to produce. We introduce Luxical, a library for high-speed "lexical-dense" text embeddings that aims to recover the best properties of both approaches for web-scale text organization. Luxical combines sparse TF--IDF features, a small ReLU network, and a knowledge distillation training regimen to approximate large transformer embedding models at a fraction of their operational cost. In this technical report, we describe the Luxical architecture and training objective and evaluate a concrete Luxical model in two disparate applications: a targeted webcrawl document retrieval test and an end-to-end language model data curation task grounded in text classification. In these tasks we demonstrate speedups ranging from 3x to 100x over varying-sized neural baselines, and comparable to FastText model inference during the data curation task. On these evaluations, the tested Luxical model illustrates favorable compute/quality trade-offs for large-scale text organization, matching the quality of neural baselines. Luxical is available as open-source software at this https URL. 

---
# The Linguistic Architecture of Reflective Thought: Evaluation of a Large Language Model as a Tool to Isolate the Formal Structure of Mentalization 

**Authors**: Stefano Epifani, Giuliano Castigliego, Laura Kecskemeti, Giuliano Razzicchia, Elisabeth Seiwald-Sonderegger  

**Link**: [PDF](https://arxiv.org/pdf/2512.08945)  

**Abstract**: Background: Mentalization integrates cognitive, affective, and intersubjective components. Large Language Models (LLMs) display an increasing ability to generate reflective texts, raising questions regarding the relationship between linguistic form and mental representation. This study assesses the extent to which a single LLM can reproduce the linguistic structure of mentalization according to the parameters of Mentalization-Based Treatment (MBT).
Methods: Fifty dialogues were generated between human participants and an LLM configured in standard mode. Five psychiatrists trained in MBT, working under blinded conditions, evaluated the mentalization profiles produced by the model along the four MBT axes, assigning Likert-scale scores for evaluative coherence, argumentative coherence, and global quality. Inter-rater agreement was estimated using ICC(3,1).
Results: Mean scores (3.63-3.98) and moderate standard deviations indicate a high level of structural coherence in the generated profiles. ICC values (0.60-0.84) show substantial-to-high agreement among raters. The model proved more stable in the Implicit-Explicit and Self-Other dimensions, while presenting limitations in the integration of internal states and external contexts. The profiles were coherent and clinically interpretable yet characterized by affective neutrality. 

---
# Enhancing Reliability across Short and Long-Form QA via Reinforcement Learning 

**Authors**: Yudong Wang, Zhe Yang, Wenhan Ma, Zhifang Sui, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2512.08944)  

**Abstract**: While reinforcement learning has unlocked unprecedented complex reasoning in large language models, it has also amplified their propensity for hallucination, creating a critical trade-off between capability and reliability. This work confronts this challenge by introducing a targeted RL framework designed to mitigate both intrinsic and extrinsic hallucinations across short and long-form question answering. We address extrinsic hallucinations (flawed internal knowledge) by creating a novel training set from open-ended conversions of TriviaQA. Concurrently, we tackle intrinsic hallucinations (unfaithfulness to context) by leveraging long-form texts from FineWeb in a fact-grounding reward scheme. To further bolster reliability, our framework explicitly rewards the model for refusing to answer unanswerable questions, thereby cultivating crucial cautiousness. Extensive experiments demonstrate that our methodology yields significant performance gains across a diverse suite of benchmarks, substantially reducing both hallucination types. Ultimately, this research contributes a practical framework for resolving the critical tension between advanced reasoning and factual trustworthiness, paving the way for more capable and reliable large language models. 

---
# Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models 

**Authors**: Singon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2512.08943)  

**Abstract**: Abstractive compression utilizes smaller langauge models to condense query-relevant context, reducing computational costs in retrieval-augmented generation (RAG). However, retrieved documents often include information that is either irrelevant to answering the query or misleading due to factual incorrect content, despite having high relevance scores. This behavior indicates that abstractive compressors are more likely to omit important information essential for the correct answer, especially in long contexts where attention dispersion occurs. To address this issue, we categorize retrieved documents in a more fine-grained manner and propose Abstractive Compression Robust against Noise (ACoRN), which introduces two novel training steps. First, we use offline data augmentation on the training dataset to enhance compressor robustness against two distinct types of retrieval noise. Second, since the language model based compressor cannot fully utilize information from multiple retrieved documents and exhibits positional bias, we perform finetuning to generate summaries centered around key information that directly supports the correct answer. Our experiments demonstrate that T5-large, trained with ACoRN as a compressor, improves EM and F1 scores while preserving the answer string, which could serve as direct evidence. ACoRN excels on datasets with many accuracy reducing documents, making it highly useful in real-world scenarios. 

---
# SCOPE: Language Models as One-Time Teacher for Hierarchical Planning in Text Environments 

**Authors**: Haoye Lu, Pavan Seshadri, Kaheer Suleman  

**Link**: [PDF](https://arxiv.org/pdf/2512.09897)  

**Abstract**: Long-term planning in complex, text-based environments presents significant challenges due to open-ended action spaces, ambiguous observations, and sparse feedback. Recent research suggests that large language models (LLMs) encode rich semantic knowledge about the world, which can be valuable for guiding agents in high-level reasoning and planning across both embodied and purely textual settings. However, existing approaches often depend heavily on querying LLMs during training and inference, making them computationally expensive and difficult to deploy efficiently. In addition, these methods typically employ a pretrained, unaltered LLM whose parameters remain fixed throughout training, providing no opportunity for adaptation to the target task. To address these limitations, we introduce SCOPE (Subgoal-COnditioned Pretraining for Efficient planning), a one-shot hierarchical planner that leverages LLM-generated subgoals only at initialization to pretrain a lightweight student model. Unlike prior approaches that distill LLM knowledge by repeatedly prompting the model to adaptively generate subgoals during training, our method derives subgoals directly from example trajectories. This design removes the need for repeated LLM queries, significantly improving efficiency, though at the cost of reduced explainability and potentially suboptimal subgoals. Despite their suboptimality, our results on the TextCraft environment show that LLM-generated subgoals can still serve as a strong starting point for hierarchical goal decomposition in text-based planning tasks. Compared to the LLM-based hierarchical agent ADaPT (Prasad et al., 2024), which achieves a 0.52 success rate, our method reaches 0.56 and reduces inference time from 164.4 seconds to just 3.0 seconds. 

---
# MedForget: Hierarchy-Aware Multimodal Unlearning Testbed for Medical AI 

**Authors**: Fengli Wu, Vaidehi Patil, Jaehong Yoon, Yue Zhang, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2512.09867)  

**Abstract**: Pretrained Multimodal Large Language Models (MLLMs) are increasingly deployed in medical AI systems for clinical reasoning, diagnosis support, and report generation. However, their training on sensitive patient data raises critical privacy and compliance challenges under regulations such as HIPAA and GDPR, which enforce the "right to be forgotten". Unlearning, the process of tuning models to selectively remove the influence of specific training data points, offers a potential solution, yet its effectiveness in complex medical settings remains underexplored. To systematically study this, we introduce MedForget, a Hierarchy-Aware Multimodal Unlearning Testbed with explicit retain and forget splits and evaluation sets containing rephrased variants. MedForget models hospital data as a nested hierarchy (Institution -> Patient -> Study -> Section), enabling fine-grained assessment across eight organizational levels. The benchmark contains 3840 multimodal (image, question, answer) instances, each hierarchy level having a dedicated unlearning target, reflecting distinct unlearning challenges. Experiments with four SOTA unlearning methods on three tasks (generation, classification, cloze) show that existing methods struggle to achieve complete, hierarchy-aware forgetting without reducing diagnostic performance. To test whether unlearning truly deletes hierarchical pathways, we introduce a reconstruction attack that progressively adds hierarchical level context to prompts. Models unlearned at a coarse granularity show strong resistance, while fine-grained unlearning leaves models vulnerable to such reconstruction. MedForget provides a practical, HIPAA-aligned testbed for building compliant medical AI systems. 

---
# Rethinking Chain-of-Thought Reasoning for Videos 

**Authors**: Yiwu Zhong, Zi-Yuan Hu, Yin Li, Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2512.09616)  

**Abstract**: Chain-of-thought (CoT) reasoning has been highly successful in solving complex tasks in natural language processing, and recent multimodal large language models (MLLMs) have extended this paradigm to video reasoning. However, these models typically build on lengthy reasoning chains and large numbers of input visual tokens. Motivated by empirical observations from our benchmark study, we hypothesize that concise reasoning combined with a reduced set of visual tokens can be sufficient for effective video reasoning. To evaluate this hypothesis, we design and validate an efficient post-training and inference framework that enhances a video MLLM's reasoning capability. Our framework enables models to operate on compressed visual tokens and generate brief reasoning traces prior to answering. The resulting models achieve substantially improved inference efficiency, deliver competitive performance across diverse benchmarks, and avoid reliance on manual CoT annotations or supervised fine-tuning. Collectively, our results suggest that long, human-like CoT reasoning may not be necessary for general video reasoning, and that concise reasoning can be both effective and efficient. Our code will be released at this https URL. 

---
# Don't Throw Away Your Beams: Improving Consistency-based Uncertainties in LLMs via Beam Search 

**Authors**: Ekaterina Fadeeva, Maiya Goloburda, Aleksandr Rubashevskii, Roman Vashurin, Artem Shelmanov, Preslav Nakov, Mrinmaya Sachan, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2512.09538)  

**Abstract**: Consistency-based methods have emerged as an effective approach to uncertainty quantification (UQ) in large language models. These methods typically rely on several generations obtained via multinomial sampling, measuring their agreement level. However, in short-form QA, multinomial sampling is prone to producing duplicates due to peaked distributions, and its stochasticity introduces considerable variance in uncertainty estimates across runs. We introduce a new family of methods that employ beam search to generate candidates for consistency-based UQ, yielding improved performance and reduced variance compared to multinomial sampling. We also provide a theoretical lower bound on the beam set probability mass under which beam search achieves a smaller error than multinomial sampling. We empirically evaluate our approach on six QA datasets and find that its consistent improvements over multinomial sampling lead to state-of-the-art UQ performance. 

---
# Are Hypervectors Enough? Single-Call LLM Reasoning over Knowledge Graphs 

**Authors**: Yezi Liu, William Youngwoo Chung, Hanning Chen, Calvin Yeung, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2512.09369)  

**Abstract**: Recent advances in large language models (LLMs) have enabled strong reasoning over both structured and unstructured knowledge. When grounded on knowledge graphs (KGs), however, prevailing pipelines rely on heavy neural encoders to embed and score symbolic paths or on repeated LLM calls to rank candidates, leading to high latency, GPU cost, and opaque decisions that hinder faithful, scalable deployment. We propose PathHD, a lightweight and encoder-free KG reasoning framework that replaces neural path scoring with hyperdimensional computing (HDC) and uses only a single LLM call per query. PathHD encodes relation paths into block-diagonal GHRR hypervectors, ranks candidates with blockwise cosine similarity and Top-K pruning, and then performs a one-shot LLM adjudication to produce the final answer together with cited supporting paths. Technically, PathHD is built on three ingredients: (i) an order-aware, non-commutative binding operator for path composition, (ii) a calibrated similarity for robust hypervector-based retrieval, and (iii) a one-shot adjudication step that preserves interpretability while eliminating per-path LLM scoring. On WebQSP, CWQ, and the GrailQA split, PathHD (i) attains comparable or better Hits@1 than strong neural baselines while using one LLM call per query; (ii) reduces end-to-end latency by $40-60\%$ and GPU memory by $3-5\times$ thanks to encoder-free retrieval; and (iii) delivers faithful, path-grounded rationales that improve error diagnosis and controllability. These results indicate that carefully designed HDC representations provide a practical substrate for efficient KG-LLM reasoning, offering a favorable accuracy-efficiency-interpretability trade-off. 

---
# ORCA: Open-ended Response Correctness Assessment for Audio Question Answering 

**Authors**: Šimon Sedláček, Sara Barahona, Bolaji Yusuf, Laura Herrera-Alarcón, Santosh Kesiraju, Cecilia Bolaños, Alicia Lozano-Diez, Sathvik Udupa, Fernando López, Allison Ferner, Ramani Duraiswami, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2512.09066)  

**Abstract**: Evaluating open-ended responses from large audio language models (LALMs) is challenging because human annotators often genuinely disagree on answer correctness due to multiple valid interpretations, partial correctness, and subjective judgment. Traditional metrics reporting only mean scores fail to capture this uncertainty. We present ORCA (Open-ended Response Correctness Assessment), a framework that models the variability in human judgments using Beta distributions to predict both expected correctness and uncertainty. Our three-stage annotation framework combines human judgment with structured feedback and iterative refinement to simultaneously curate training data and improve benchmark quality. We collected 11,721 annotations across 3,580 question-answer pairs from 15 LALMs on two audio QA benchmarks, achieving inter-annotator agreement of 0.82 (Krippendorff's alpha). ORCA achieves 0.91 Spearman correlation with mean human judgments, matching or outperforming LLM-judge baselines while providing uncertainty estimates and requiring significantly less compute. We release our models, code, and curated dataset. 

---
# Financial Instruction Following Evaluation (FIFE) 

**Authors**: Glenn Matlin, Siddharth, Anirudh JM, Aditya Shukla, Yahya Hassan, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2512.08965)  

**Abstract**: Language Models (LMs) struggle with complex, interdependent instructions, particularly in high-stakes domains like finance where precision is critical. We introduce FIFE, a novel, high-difficulty benchmark designed to assess LM instruction-following capabilities for financial analysis tasks. FIFE comprises 88 human-authored prompts and employs a verification system with chainable, verifiable constraints for fine-grained reward signals. We evaluate 53 models (proprietary, open-weight, open-source) in a zero-shot setting. Our key findings reveal a clear performance hierarchy: the top open-weight model (76.1 strict / 79.5 loose) surpasses the leading proprietary system (65.9 strict / 70.5 loose), while the best open-source models lag significantly (45.5 strict / 48.9 loose). However, even top-performing models struggle with FIFE's complex requirements, failing to achieve perfect compliance. We release our dataset and code as an open-source resource to promote research in Reinforcement Learning for the financial domain. 

---
# Resolving Conflicts in Lifelong Learning via Aligning Updates in Subspaces 

**Authors**: Yueer Zhou, Yichen Wu, Ying Wei  

**Link**: [PDF](https://arxiv.org/pdf/2512.08960)  

**Abstract**: Low-Rank Adaptation (LoRA) enables efficient Continual Learning but often suffers from catastrophic forgetting due to destructive interference between tasks. Our analysis reveals that this degradation is primarily driven by antagonistic directional updates where new task gradients directly oppose the historical weight trajectory. To address this, we propose PS-LoRA (Parameter Stability LoRA), a framework designed to resolve conflicts by aligning updates within the optimization subspace. Our approach employs a dual-regularization objective that penalizes conflicting directions and constrains magnitude deviations to ensure consistency with prior knowledge. Additionally, we implement a magnitude-based merging strategy to consolidate sequential adapters into a robust representation without retraining. Experiments on NLP and Vision benchmarks show that PS-LoRA outperforms state-of-the-art methods by preserving the stability of learned representations while efficiently adapting to new domains. 

---
# Large Language Models as Search Engines: Societal Challenges 

**Authors**: Zacchary Sadeddine, Winston Maxwell, Gaël Varoquaux, Fabian M. Suchanek  

**Link**: [PDF](https://arxiv.org/pdf/2512.08946)  

**Abstract**: Large Language Models (LLMs) may one day replace search engines as the primary portal to information on the Web. In this article, we investigate the societal challenges that such a change could bring. We focus on the roles of LLM Providers, Content Creators, and End Users, and identify 15 types of challenges. With each, we show current mitigation strategies -- both from the technical perspective and the legal perspective. We also discuss the impact of each challenge and point out future research opportunities. 

---
