# KV Cache Steering for Inducing Reasoning in Small Language Models 

**Authors**: Max Belitsky, Dawid J. Kopiczko, Michael Dorkenwald, M. Jehanzeb Mirza, Cees G. M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2507.08799)  

**Abstract**: We propose cache steering, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. To validate its effectiveness, we apply cache steering to induce chain-of-thought reasoning in small language models. Our approach leverages GPT-4o-generated reasoning traces to construct steering vectors that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. Experimental evaluations on diverse reasoning benchmarks demonstrate that cache steering improves both the qualitative structure of model reasoning and quantitative task performance. Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of hyperparameter stability, inference-time efficiency, and ease of integration, making it a more robust and practical solution for controlled generation. 

---
# Multilingual Multimodal Software Developer for Code Generation 

**Authors**: Linzheng Chai, Jian Yang, Shukai Liu, Wei Zhang, Liran Wang, Ke Jin, Tao Sun, Congnan Liu, Chenchen Zhang, Hualei Zhu, Jiaheng Liu, Xianjie Wu, Ge Zhang, Tianyu Liu, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08719)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has significantly improved code generation, yet most models remain text-only, neglecting crucial visual aids like diagrams and flowcharts used in real-world software development. To bridge this gap, we introduce MM-Coder, a Multilingual Multimodal software developer. MM-Coder integrates visual design inputs-Unified Modeling Language (UML) diagrams and flowcharts (termed Visual Workflow)-with textual instructions to enhance code generation accuracy and architectural alignment. To enable this, we developed MMc-Instruct, a diverse multimodal instruction-tuning dataset including visual-workflow-based code generation, allowing MM-Coder to synthesize textual and graphical information like human developers, distinct from prior work on narrow tasks. Furthermore, we introduce MMEval, a new benchmark for evaluating multimodal code generation, addressing existing text-only limitations. Our evaluations using MMEval highlight significant remaining challenges for models in precise visual information capture, instruction following, and advanced programming knowledge. Our work aims to revolutionize industrial programming by enabling LLMs to interpret and implement complex specifications conveyed through both text and visual designs. 

---
# KG-Attention: Knowledge Graph-Guided Attention at Test-Time via Bidirectional Information Aggregation 

**Authors**: Songlin Zhai, Guilin Qi, Yuan Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08704)  

**Abstract**: Knowledge graphs (KGs) play a critical role in enhancing large language models (LLMs) by introducing structured and grounded knowledge into the learning process. However, most existing KG-enhanced approaches rely on parameter-intensive fine-tuning, which risks catastrophic forgetting and degrades the pretrained model's generalization. Moreover, they exhibit limited adaptability to real-time knowledge updates due to their static integration frameworks. To address these issues, we introduce the first test-time KG-augmented framework for LLMs, built around a dedicated knowledge graph-guided attention (KGA) module that enables dynamic knowledge fusion without any parameter updates. The proposed KGA module augments the standard self-attention mechanism with two synergistic pathways: outward and inward aggregation. Specifically, the outward pathway dynamically integrates external knowledge into input representations via input-driven KG fusion. This inward aggregation complements the outward pathway by refining input representations through KG-guided filtering, suppressing task-irrelevant signals and amplifying knowledge-relevant patterns. Importantly, while the outward pathway handles knowledge fusion, the inward path selects the most relevant triples and feeds them back into the fusion process, forming a closed-loop enhancement mechanism. By synergistically combining these two pathways, the proposed method supports real-time knowledge fusion exclusively at test-time, without any parameter modification. Extensive experiments on five benchmarks verify the comparable knowledge fusion performance of KGA. 

---
# KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment 

**Authors**: Jiyao Zhang, Chengli Zhong, Hui Xu, Qige Li, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08665)  

**Abstract**: Modern large language models (LLMs) show promising progress in formalizing informal mathematics into machine-verifiable theorems. However, these methods still face bottlenecks due to the limited quantity and quality of multilingual parallel corpora. In this paper, we propose a novel neuro-symbolic framework KELPS (Knowledge-Equation based Logical Processing System) to address these problems. KELPS is an iterative framework for translating, synthesizing, and filtering informal data into multiple formal languages (Lean, Coq, and Isabelle). First, we translate natural language into Knowledge Equations (KEs), a novel language that we designed, theoretically grounded in assertional logic. Next, we convert them to target languages through rigorously defined rules that preserve both syntactic structure and semantic meaning. This process yielded a parallel corpus of over 60,000 problems. Our framework achieves 88.9% syntactic accuracy (pass@1) on MiniF2F, outperforming SOTA models such as Deepseek-V3 (81%) and Herald (81.3%) across multiple datasets. All datasets and codes are available in the supplementary materials. 

---
# The Impact of Automatic Speech Transcription on Speaker Attribution 

**Authors**: Cristina Aggazzotti, Matthew Wiesner, Elizabeth Allyn Smith, Nicholas Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2507.08660)  

**Abstract**: Speaker attribution from speech transcripts is the task of identifying a speaker from the transcript of their speech based on patterns in their language use. This task is especially useful when the audio is unavailable (e.g. deleted) or unreliable (e.g. anonymized speech). Prior work in this area has primarily focused on the feasibility of attributing speakers using transcripts produced by human annotators. However, in real-world settings, one often only has more errorful transcripts produced by automatic speech recognition (ASR) systems. In this paper, we conduct what is, to our knowledge, the first comprehensive study of the impact of automatic transcription on speaker attribution performance. In particular, we study the extent to which speaker attribution performance degrades in the face of transcription errors, as well as how properties of the ASR system impact attribution. We find that attribution is surprisingly resilient to word-level transcription errors and that the objective of recovering the true transcript is minimally correlated with attribution performance. Overall, our findings suggest that speaker attribution on more errorful transcripts produced by ASR is as good, if not better, than attribution based on human-transcribed data, possibly because ASR transcription errors can capture speaker-specific features revealing of speaker identity. 

---
# A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1 

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski  

**Link**: [PDF](https://arxiv.org/pdf/2507.08621)  

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as this http URL and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings. 

---
# DocPolarBERT: A Pre-trained Model for Document Understanding with Relative Polar Coordinate Encoding of Layout Structures 

**Authors**: Benno Uthayasooriyar, Antoine Ly, Franck Vermet, Caio Corro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08606)  

**Abstract**: We introduce DocPolarBERT, a layout-aware BERT model for document understanding that eliminates the need for absolute 2D positional embeddings. We extend self-attention to take into account text block positions in relative polar coordinate system rather than the Cartesian one. Despite being pre-trained on a dataset more than six times smaller than the widely used IIT-CDIP corpus, DocPolarBERT achieves state-of-the-art results. These results demonstrate that a carefully designed attention mechanism can compensate for reduced pre-training data, offering an efficient and effective alternative for document understanding. 

---
# The AI Language Proficiency Monitor -- Tracking the Progress of LLMs on Multilingual Benchmarks 

**Authors**: David Pomerenke, Jonas Nothnagel, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2507.08538)  

**Abstract**: To ensure equitable access to the benefits of large language models (LLMs), it is essential to evaluate their capabilities across the world's languages. We introduce the AI Language Proficiency Monitor, a comprehensive multilingual benchmark that systematically assesses LLM performance across up to 200 languages, with a particular focus on low-resource languages. Our benchmark aggregates diverse tasks including translation, question answering, math, and reasoning, using datasets such as FLORES+, MMLU, GSM8K, TruthfulQA, and ARC. We provide an open-source, auto-updating leaderboard and dashboard that supports researchers, developers, and policymakers in identifying strengths and gaps in model performance. In addition to ranking models, the platform offers descriptive insights such as a global proficiency map and trends over time. By complementing and extending prior multilingual benchmarks, our work aims to foster transparency, inclusivity, and progress in multilingual AI. The system is available at this https URL. 

---
# PromotionGo at SemEval-2025 Task 11: A Feature-Centric Framework for Cross-Lingual Multi-Emotion Detection in Short Texts 

**Authors**: Ziyi Huang, Xia Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.08499)  

**Abstract**: This paper presents our system for SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection (Track A), which focuses on multi-label emotion detection in short texts. We propose a feature-centric framework that dynamically adapts document representations and learning algorithms to optimize language-specific performance. Our study evaluates three key components: document representation, dimensionality reduction, and model training in 28 languages, highlighting five for detailed analysis. The results show that TF-IDF remains highly effective for low-resource languages, while contextual embeddings like FastText and transformer-based document representations, such as those produced by Sentence-BERT, exhibit language-specific strengths. Principal Component Analysis (PCA) reduces training time without compromising performance, particularly benefiting FastText and neural models such as Multi-Layer Perceptrons (MLP). Computational efficiency analysis underscores the trade-off between model complexity and processing cost. Our framework provides a scalable solution for multilingual emotion detection, addressing the challenges of linguistic diversity and resource constraints. 

---
# Semantic-Augmented Latent Topic Modeling with LLM-in-the-Loop 

**Authors**: Mengze Hong, Chen Jason Zhang, Di Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08498)  

**Abstract**: Latent Dirichlet Allocation (LDA) is a prominent generative probabilistic model used for uncovering abstract topics within document collections. In this paper, we explore the effectiveness of augmenting topic models with Large Language Models (LLMs) through integration into two key phases: Initialization and Post-Correction. Since the LDA is highly dependent on the quality of its initialization, we conduct extensive experiments on the LLM-guided topic clustering for initializing the Gibbs sampling algorithm. Interestingly, the experimental results reveal that while the proposed initialization strategy improves the early iterations of LDA, it has no effect on the convergence and yields the worst performance compared to the baselines. The LLM-enabled post-correction, on the other hand, achieved a promising improvement of 5.86% in the coherence evaluation. These results highlight the practical benefits of the LLM-in-the-loop approach and challenge the belief that LLMs are always the superior text mining alternative. 

---
# LLaPa: A Vision-Language Model Framework for Counterfactual-Aware Procedural Planning 

**Authors**: Shibo Sun, Xue Li, Donglin Di, Mingjie Wei, Lanshun Nie, Wei-Nan Zhang, Dechen Zhan, Yang Song, Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08496)  

**Abstract**: While large language models (LLMs) have advanced procedural planning for embodied AI systems through strong reasoning abilities, the integration of multimodal inputs and counterfactual reasoning remains underexplored. To tackle these challenges, we introduce LLaPa, a vision-language model framework designed for multimodal procedural planning. LLaPa generates executable action sequences from textual task descriptions and visual environmental images using vision-language models (VLMs). Furthermore, we enhance LLaPa with two auxiliary modules to improve procedural planning. The first module, the Task-Environment Reranker (TER), leverages task-oriented segmentation to create a task-sensitive feature space, aligning textual descriptions with visual environments and emphasizing critical regions for procedural execution. The second module, the Counterfactual Activities Retriever (CAR), identifies and emphasizes potential counterfactual conditions, enhancing the model's reasoning capability in counterfactual scenarios. Extensive experiments on ActPlan-1K and ALFRED benchmarks demonstrate that LLaPa generates higher-quality plans with superior LCS and correctness, outperforming advanced models. The code and models are available this https URL. 

---
# A Third Paradigm for LLM Evaluation: Dialogue Game-Based Evaluation using clembench 

**Authors**: David Schlangen, Sherzod Hakimov, Jonathan Jordan, Philipp Sadler  

**Link**: [PDF](https://arxiv.org/pdf/2507.08491)  

**Abstract**: There are currently two main paradigms for evaluating large language models (LLMs), reference-based evaluation and preference-based evaluation. The first, carried over from the evaluation of machine learning models in general, relies on pre-defined task instances, for which reference task executions are available. The second, best exemplified by the LM-arena, relies on (often self-selected) users bringing their own intents to a site that routes these to several models in parallel, among whose responses the user then selects their most preferred one. The former paradigm hence excels at control over what is tested, while the latter comes with higher ecological validity, testing actual use cases interactively. Recently, a third complementary paradigm has emerged that combines some of the strengths of these approaches, offering control over multi-turn, reference-free, repeatable interactions, while stressing goal-directedness: dialogue game based evaluation. While the utility of this approach has been shown by several projects, its adoption has been held back by the lack of a mature, easily re-usable implementation. In this paper, we present clembench, which has been in continuous development since 2023 and has in its latest release been optimized for ease of general use. We describe how it can be used to benchmark one's own models (using a provided set of benchmark game instances in English), as well as how easily the benchmark itself can be extended with new, tailor-made targeted tests. 

---
# Enhancing Essay Cohesion Assessment: A Novel Item Response Theory Approach 

**Authors**: Bruno Alexandre Rosa, Hilário Oliveira, Luiz Rodrigues, Eduardo Araujo Oliveira, Rafael Ferreira Mello  

**Link**: [PDF](https://arxiv.org/pdf/2507.08487)  

**Abstract**: Essays are considered a valuable mechanism for evaluating learning outcomes in writing. Textual cohesion is an essential characteristic of a text, as it facilitates the establishment of meaning between its parts. Automatically scoring cohesion in essays presents a challenge in the field of educational artificial intelligence. The machine learning algorithms used to evaluate texts generally do not consider the individual characteristics of the instances that comprise the analysed corpus. In this meaning, item response theory can be adapted to the context of machine learning, characterising the ability, difficulty and discrimination of the models used. This work proposes and analyses the performance of a cohesion score prediction approach based on item response theory to adjust the scores generated by machine learning models. In this study, the corpus selected for the experiments consisted of the extended Essay-BR, which includes 6,563 essays in the style of the National High School Exam (ENEM), and the Brazilian Portuguese Narrative Essays, comprising 1,235 essays written by 5th to 9th grade students from public schools. We extracted 325 linguistic features and treated the problem as a machine learning regression task. The experimental results indicate that the proposed approach outperforms conventional machine learning models and ensemble methods in several evaluation metrics. This research explores a potential approach for improving the automatic evaluation of cohesion in educational essays. 

---
# ILT-Iterative LoRA Training through Focus-Feedback-Fix for Multilingual Speech Recognition 

**Authors**: Qingliang Meng, Hao Wu, Wei Liang, Wei Xu, Qing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.08477)  

**Abstract**: The deep integration of large language models and automatic speech recognition systems has become a promising research direction with high practical value. To address the overfitting issue commonly observed in Low-Rank Adaptation (LoRA) during the supervised fine-tuning (SFT) stage, this work proposes an innovative training paradigm Iterative LoRA Training (ILT) in combination with an Iterative Pseudo Labeling strategy, effectively enhancing the theoretical upper bound of model performance. Based on Whisper-large-v3 and Qwen2-Audio, we conduct systematic experiments using a three-stage training process: Focus Training, Feed Back Training, and Fix Training. Experimental results demonstrate the effectiveness of the proposed method. Furthermore, the MegaAIS research team applied this technique in the Interspeech 2025 Multilingual Conversational Speech Language Modeling Challenge (MLC-SLM), achieving 4th in Track 1 (Multilingual ASR Task) and 1st place in Track 2 (Speech Separation and Recognition Task), showcasing the practical feasibility and strong application potential of our approach. 

---
# Using Large Language Models for Legal Decision-Making in Austrian Value-Added Tax Law: An Experimental Study 

**Authors**: Marina Luketina, Andrea Benkel, Christoph G. Schuetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08468)  

**Abstract**: This paper provides an experimental evaluation of the capability of large language models (LLMs) to assist in legal decision-making within the framework of Austrian and European Union value-added tax (VAT) law. In tax consulting practice, clients often describe cases in natural language, making LLMs a prime candidate for supporting automated decision-making and reducing the workload of tax professionals. Given the requirement for legally grounded and well-justified analyses, the propensity of LLMs to hallucinate presents a considerable challenge. The experiments focus on two common methods for enhancing LLM performance: fine-tuning and retrieval-augmented generation (RAG). In this study, these methods are applied on both textbook cases and real-world cases from a tax consulting firm to systematically determine the best configurations of LLM-based systems and assess the legal-reasoning capabilities of LLMs. The findings highlight the potential of using LLMs to support tax consultants by automating routine tasks and providing initial analyses, although current prototypes are not ready for full automation due to the sensitivity of the legal domain. The findings indicate that LLMs, when properly configured, can effectively support tax professionals in VAT tasks and provide legally grounded justifications for decisions. However, limitations remain regarding the handling of implicit client knowledge and context-specific documentation, underscoring the need for future integration of structured background information. 

---
# Diagnosing Failures in Large Language Models' Answers: Integrating Error Attribution into Evaluation Framework 

**Authors**: Zishan Xu, Shuyi Xie, Qingsong Lv, Shupei Xiao, Linlin Song, Sui Wenjuan, Fan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08459)  

**Abstract**: With the widespread application of Large Language Models (LLMs) in various tasks, the mainstream LLM platforms generate massive user-model interactions daily. In order to efficiently analyze the performance of models and diagnose failures in their answers, it is essential to develop an automated framework to systematically categorize and attribute errors. However, existing evaluation models lack error attribution capability. In this work, we establish a comprehensive Misattribution Framework with 6 primary and 15 secondary categories to facilitate in-depth analysis. Based on this framework, we present AttriData, a dataset specifically designed for error attribution, encompassing misattribution, along with the corresponding scores and feedback. We also propose MisAttributionLLM, a fine-tuned model on AttriData, which is the first general-purpose judge model capable of simultaneously generating score, misattribution, and feedback. Extensive experiments and analyses are conducted to confirm the effectiveness and robustness of our proposed method. 

---
# Finding Common Ground: Using Large Language Models to Detect Agreement in Multi-Agent Decision Conferences 

**Authors**: Selina Heller, Mohamed Ibrahim, David Antony Selby, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.08440)  

**Abstract**: Decision conferences are structured, collaborative meetings that bring together experts from various fields to address complex issues and reach a consensus on recommendations for future actions or policies. These conferences often rely on facilitated discussions to ensure productive dialogue and collective agreement. Recently, Large Language Models (LLMs) have shown significant promise in simulating real-world scenarios, particularly through collaborative multi-agent systems that mimic group interactions. In this work, we present a novel LLM-based multi-agent system designed to simulate decision conferences, specifically focusing on detecting agreement among the participant agents. To achieve this, we evaluate six distinct LLMs on two tasks: stance detection, which identifies the position an agent takes on a given issue, and stance polarity detection, which identifies the sentiment as positive, negative, or neutral. These models are further assessed within the multi-agent system to determine their effectiveness in complex simulations. Our results indicate that LLMs can reliably detect agreement even in dynamic and nuanced debates. Incorporating an agreement-detection agent within the system can also improve the efficiency of group debates and enhance the overall quality and coherence of deliberations, making them comparable to real-world decision conferences regarding outcome and decision-making. These findings demonstrate the potential for LLM-based multi-agent systems to simulate group decision-making processes. They also highlight that such systems could be instrumental in supporting decision-making with expert elicitation workshops across various domains. 

---
# ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains 

**Authors**: Zilu Dong, Xiangqing Shen, Zinong Yang, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.08427)  

**Abstract**: Current knowledge editing methods for large language models (LLMs) struggle to maintain logical consistency when propagating ripple effects to associated facts. We propose ChainEdit, a framework that synergizes knowledge graph-derived logical rules with LLM logical reasoning capabilities to enable systematic chain updates. By automatically extracting logical patterns from structured knowledge bases and aligning them with LLMs' internal logics, ChainEdit dynamically generates and edits logically connected knowledge clusters. Experiments demonstrate an improvement of more than 30% in logical generalization over baselines while preserving editing reliability and specificity. We further address evaluation biases in existing benchmarks through knowledge-aware protocols that disentangle external dependencies. This work establishes new state-of-the-art performance on ripple effect while ensuring internal logical consistency after knowledge editing. 

---
# A Survey of Large Language Models in Discipline-specific Research: Challenges, Methods and Opportunities 

**Authors**: Lu Xiang, Yang Zhao, Yaping Zhang, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2507.08425)  

**Abstract**: Large Language Models (LLMs) have demonstrated their transformative potential across numerous disciplinary studies, reshaping the existing research methodologies and fostering interdisciplinary collaboration. However, a systematic understanding of their integration into diverse disciplines remains underexplored. This survey paper provides a comprehensive overview of the application of LLMs in interdisciplinary studies, categorising research efforts from both a technical perspective and with regard to their applicability. From a technical standpoint, key methodologies such as supervised fine-tuning, retrieval-augmented generation, agent-based approaches, and tool-use integration are examined, which enhance the adaptability and effectiveness of LLMs in discipline-specific contexts. From the perspective of their applicability, this paper explores how LLMs are contributing to various disciplines including mathematics, physics, chemistry, biology, and the humanities and social sciences, demonstrating their role in discipline-specific tasks. The prevailing challenges are critically examined and the promising research directions are highlighted alongside the recent advances in LLMs. By providing a comprehensive overview of the technical developments and applications in this field, this survey aims to serve as an invaluable resource for the researchers who are navigating the complex landscape of LLMs in the context of interdisciplinary studies. 

---
# The Curious Case of Factuality Finetuning: Models' Internal Beliefs Can Improve Factuality 

**Authors**: Benjamin Newman, Abhilasha Ravichander, Jaehun Jung, Rui Xin, Hamish Ivison, Yegor Kuznetsov, Pang Wei Koh, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08371)  

**Abstract**: Language models are prone to hallucination - generating text that is factually incorrect. Finetuning models on high-quality factual information can potentially reduce hallucination, but concerns remain; obtaining factual gold data can be expensive and training on correct but unfamiliar data may potentially lead to even more downstream hallucination. What data should practitioners finetune on to mitigate hallucinations in language models? In this work, we study the relationship between the factuality of finetuning data and the prevalence of hallucinations in long-form generation tasks. Counterintuitively, we find that finetuning on factual gold data is not as helpful as finetuning on model-generated data that models believe to be factual. Next, we evaluate filtering strategies applied on both factual gold data and model-generated data, and find that finetuning on model-generated data that is filtered by models' own internal judgments often leads to better overall factuality compared to other configurations: training on gold data filtered by models' judgments, training on gold data alone, or training on model-generated data that is supported by gold data. These factuality improvements transfer across three domains we study, suggesting that a models' own beliefs can provide a powerful signal for factuality. 

---
# Exploring Design of Multi-Agent LLM Dialogues for Research Ideation 

**Authors**: Keisuke Ueda, Wataru Hirota, Takuto Asakura, Takahiro Omi, Kosuke Takahashi, Kosuke Arima, Tatsuya Ishigaki  

**Link**: [PDF](https://arxiv.org/pdf/2507.08350)  

**Abstract**: Large language models (LLMs) are increasingly used to support creative tasks such as research idea generation. While recent work has shown that structured dialogues between LLMs can improve the novelty and feasibility of generated ideas, the optimal design of such interactions remains unclear. In this study, we conduct a comprehensive analysis of multi-agent LLM dialogues for scientific ideation. We compare different configurations of agent roles, number of agents, and dialogue depth to understand how these factors influence the novelty and feasibility of generated ideas. Our experimental setup includes settings where one agent generates ideas and another critiques them, enabling iterative improvement. Our results show that enlarging the agent cohort, deepening the interaction depth, and broadening agent persona heterogeneity each enrich the diversity of generated ideas. Moreover, specifically increasing critic-side diversity within the ideation-critique-revision loop further boosts the feasibility of the final proposals. Our findings offer practical guidelines for building effective multi-agent LLM systems for scientific ideation. Our code is available at this https URL. 

---
# Beyond N-Grams: Rethinking Evaluation Metrics and Strategies for Multilingual Abstractive Summarization 

**Authors**: Itai Mondshine, Tzuf Paz-Argaman, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2507.08342)  

**Abstract**: Automatic n-gram based metrics such as ROUGE are widely used for evaluating generative tasks such as summarization. While these metrics are considered indicative (even if imperfect) of human evaluation for English, their suitability for other languages remains unclear. To address this, we systematically assess evaluation metrics for generation both n-gram-based and neural based to evaluate their effectiveness across languages and tasks. Specifically, we design a large-scale evaluation suite across eight languages from four typological families: agglutinative, isolating, low-fusional, and high-fusional, spanning both low- and high-resource settings, to analyze their correlation with human judgments. Our findings highlight the sensitivity of evaluation metrics to the language type. For example, in fusional languages, n-gram-based metrics show lower correlation with human assessments compared to isolating and agglutinative languages. We also demonstrate that proper tokenization can significantly mitigate this issue for morphologically rich fusional languages, sometimes even reversing negative trends. Additionally, we show that neural-based metrics specifically trained for evaluation, such as COMET, consistently outperform other neural metrics and better correlate with human judgments in low-resource languages. Overall, our analysis highlights the limitations of n-gram metrics for fusional languages and advocates for greater investment in neural-based metrics trained for evaluation tasks. 

---
# What Factors Affect LLMs and RLLMs in Financial Question Answering? 

**Authors**: Peng Wang, Xuesi Hu, Jiageng Wu, Yuntao Zou, Qiancheng Zhang, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08339)  

**Abstract**: Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering. 

---
# Distillation versus Contrastive Learning: How to Train Your Rerankers 

**Authors**: Zhichao Xu, Zhiqi Huang, Shengyao Zhuang, Ashim Gupta, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.08336)  

**Abstract**: Training text rerankers is crucial for information retrieval. Two primary strategies are widely used: contrastive learning (optimizing directly on ground-truth labels) and knowledge distillation (transferring knowledge from a larger reranker). While both have been studied in the literature, a clear comparison of their effectiveness for training cross-encoder rerankers under practical conditions is needed.
This paper empirically compares these strategies by training rerankers of different sizes and architectures using both methods on the same data, with a strong contrastive learning model acting as the distillation teacher. Our results show that knowledge distillation generally yields better in-domain and out-of-domain ranking performance than contrastive learning when distilling from a larger teacher model. This finding is consistent across student model sizes and architectures. However, distilling from a teacher of the same capacity does not provide the same advantage, particularly for out-of-domain tasks. These findings offer practical guidance for choosing a training strategy based on available teacher models. Therefore, we recommend using knowledge distillation to train smaller rerankers if a larger, more powerful teacher is accessible; in its absence, contrastive learning provides a strong and more reliable alternative otherwise. 

---
# MK2 at PBIG Competition: A Prompt Generation Solution 

**Authors**: Yuzheng Xu, Tosho Hirasawa, Seiya Kawano, Shota Kato, Tadashi Kozuno  

**Link**: [PDF](https://arxiv.org/pdf/2507.08335)  

**Abstract**: The Patent-Based Idea Generation task asks systems to turn real patents into product ideas viable within three years. We propose MK2, a prompt-centric pipeline: Gemini 2.5 drafts and iteratively edits a prompt, grafting useful fragments from weaker outputs; GPT-4.1 then uses this prompt to create one idea per patent, and an Elo loop judged by Qwen3-8B selects the best prompt-all without extra training data. Across three domains, two evaluator types, and six criteria, MK2 topped the automatic leaderboard and won 25 of 36 tests. Only the materials-chemistry track lagged, indicating the need for deeper domain grounding; yet, the results show that lightweight prompt engineering has already delivered competitive, commercially relevant ideation from patents. 

---
# CRMAgent: A Multi-Agent LLM System for E-Commerce CRM Message Template Generation 

**Authors**: Yinzhu Quan, Xinrui Li, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08325)  

**Abstract**: In e-commerce private-domain channels such as instant messaging and e-mail, merchants engage customers directly as part of their Customer Relationship Management (CRM) programmes to drive retention and conversion. While a few top performers excel at crafting outbound messages, most merchants struggle to write persuasive copy because they lack both expertise and scalable tools. We introduce CRMAgent, a multi-agent system built on large language models (LLMs) that generates high-quality message templates and actionable writing guidance through three complementary modes. First, group-based learning enables the agent to learn from a merchant's own top-performing messages within the same audience segment and rewrite low-performing ones. Second, retrieval-and-adaptation fetches templates that share the same audience segment and exhibit high similarity in voucher type and product category, learns their successful patterns, and adapts them to the current campaign. Third, a rule-based fallback provides a lightweight zero-shot rewrite when no suitable references are available. Extensive experiments show that CRMAgent consistently outperforms merchants' original templates, delivering significant gains in both audience-match and marketing-effectiveness metrics. 

---
# Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency 

**Authors**: Yupu Liang, Yaping Zhang, Zhiyang Zhang, Zhiyuan Chen, Yang Zhao, Lu Xiang, Chengqing Zong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08309)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance in document image tasks, especially Optical Character Recognition (OCR). However, they struggle with Document Image Machine Translation (DIMT), which requires handling both cross-modal and cross-lingual challenges. Previous efforts to enhance DIMT capability through Supervised Fine-Tuning (SFT) on the DIMT dataset often result in the forgetting of the model's existing monolingual abilities, such as OCR. To address these challenges, we introduce a novel fine-tuning paradigm, named Synchronously Self-Reviewing (SSR) its OCR proficiency, inspired by the concept "Bilingual Cognitive Advantage". Specifically, SSR prompts the model to generate OCR text before producing translation text, which allows the model to leverage its strong monolingual OCR ability while learning to translate text across languages. Comprehensive experiments demonstrate the proposed SSR learning helps mitigate catastrophic forgetting, improving the generalization ability of MLLMs on both OCR and DIMT tasks. 

---
# KAT-V1: Kwai-AutoThink Technical Report 

**Authors**: Zizheng Zhan, Ken Deng, Huaixi Tang, Wen Xiang, Kun Wu, Weihao Li, Wenqiang Zhu, Jingxuan Xu, Lecheng Huang, Zongxian Feng, Shaojie Wang, Shangpeng Yan, Jiaheng Liu, Zhongyuan Peng, Zuchen Gao, Haoyang Huang, Ziqi Zhan, Yanan Wu, Yuanxing Zhang, Jian Yang, Guang Chen, Haotian Zhang, Bin Chen, Bing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08297)  

**Abstract**: We present Kwaipilot-AutoThink (KAT), an open-source 40B large language model developed to address the overthinking problem in reasoning-intensive tasks, where an automatic thinking training paradigm is proposed to dynamically switch between reasoning and non-reasoning modes based on task complexity. Specifically, first, we construct the dual-regime dataset based on a novel tagging pipeline and a multi-agent synthesis strategy, and then we apply Multi-Token Prediction (MTP)-enhanced knowledge distillation, enabling efficient and fine-grained reasoning transfer with minimal pretraining cost. Besides, we implement a cold-start initialization strategy that introduces mode-selection priors using majority-vote signals and intent-aware prompting. Finally, we propose Step-SRPO, a reinforcement learning algorithm that incorporates intermediate supervision into the GRPO framework, offering structured guidance over both reasoning-mode selection and response accuracy. Extensive experiments across multiple benchmarks demonstrate that KAT consistently matches or even outperforms current state-of-the-art models, including DeepSeek-R1-0528 and Qwen3-235B-A22B, across a wide range of reasoning-intensive tasks while reducing token usage by up to approximately 30\%. Beyond academic evaluation, KAT has been successfully deployed in Kwaipilot (i.e., Kuaishou's internal coding assistant), and improves real-world development workflows with high accuracy, efficiency, and controllable reasoning behaviors. Moreover, we are actively training a 200B Mixture-of-Experts (MoE) with 40B activation parameters, where the early-stage results already demonstrate promising improvements in performance and efficiency, further showing the scalability of the AutoThink paradigm. 

---
# Exploring Gender Differences in Chronic Pain Discussions on Reddit 

**Authors**: Ancita Maria Andrade, Tanvi Banerjee, Ramakrishna Mundugar  

**Link**: [PDF](https://arxiv.org/pdf/2507.08241)  

**Abstract**: Pain is an inherent part of human existence, manifesting as both physical and emotional experiences, and can be categorized as either acute or chronic. Over the years, extensive research has been conducted to understand the causes of pain and explore potential treatments, with contributions from various scientific disciplines. However, earlier studies often overlooked the role of gender in pain experiences. In this study, we utilized Natural Language Processing (NLP) to analyze and gain deeper insights into individuals' pain experiences, with a particular focus on gender differences. We successfully classified posts into male and female corpora using the Hidden Attribute Model-Convolutional Neural Network (HAM-CNN), achieving an F1 score of 0.86 by aggregating posts based on usernames. Our analysis revealed linguistic differences between genders, with female posts tending to be more emotionally focused. Additionally, the study highlighted that conditions such as migraine and sinusitis are more prevalent among females and explored how pain medication affects individuals differently based on gender. 

---
# Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension? 

**Authors**: KV Aditya Srivatsa, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.08232)  

**Abstract**: Large Language Models (LLMs) are increasingly used as proxy students in the development of Intelligent Tutoring Systems (ITSs) and in piloting test questions. However, to what extent these proxy students accurately emulate the behavior and characteristics of real students remains an open question. To investigate this, we collected a dataset of 489 items from the National Assessment of Educational Progress (NAEP), covering mathematics and reading comprehension in grades 4, 8, and 12. We then apply an Item Response Theory (IRT) model to position 11 diverse and state-of-the-art LLMs on the same ability scale as real student populations. Our findings reveal that, without guidance, strong general-purpose models consistently outperform the average student at every grade, while weaker or domain-mismatched models may align incidentally. Using grade-enforcement prompts changes models' performance, but whether they align with the average grade-level student remains highly model- and prompt-specific: no evaluated model-prompt pair fits the bill across subjects and grades, underscoring the need for new training and evaluation strategies. We conclude by providing guidelines for the selection of viable proxies based on our findings. 

---
# Simple Mechanistic Explanations for Out-Of-Context Reasoning 

**Authors**: Atticus Wang, Joshua Engels, Oliver Clive-Griffin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08218)  

**Abstract**: Out-of-context reasoning (OOCR) is a phenomenon in which fine-tuned LLMs exhibit surprisingly deep out-of-distribution generalization. Rather than learning shallow heuristics, they implicitly internalize and act on the consequences of observations scattered throughout the fine-tuning data. In this work, we investigate this phenomenon mechanistically and find that many instances of OOCR in the literature have a simple explanation: the LoRA fine-tuning essentially adds a constant steering vector, steering the model towards a general concept. This improves performance on the fine-tuning task and in many other concept-related domains, causing the surprising generalization. Moreover, we can directly train steering vectors for these tasks from scratch, which also induces OOCR. We find that our results hold even for a task that seems like it must involve conditional behavior (model backdoors); it turns out that unconditionally adding a steering vector is sufficient. Overall, our work presents one explanation of what gets learned during fine-tuning for OOCR tasks, contributing to the key question of why LLMs can reason out of context, an advanced capability that is highly relevant to their safe and reliable deployment. 

---
# TruthTorchLM: A Comprehensive Library for Predicting Truthfulness in LLM Outputs 

**Authors**: Duygu Nur Yaldiz, Yavuz Faruk Bakman, Sungmin Kang, Alperen Öziş, Hayrettin Eren Yildiz, Mitash Ashish Shah, Zhiqi Huang, Anoop Kumar, Alfy Samuel, Daben Liu, Sai Praneeth Karimireddy, Salman Avestimehr  

**Link**: [PDF](https://arxiv.org/pdf/2507.08203)  

**Abstract**: Generative Large Language Models (LLMs)inevitably produce untruthful responses. Accurately predicting the truthfulness of these outputs is critical, especially in high-stakes settings. To accelerate research in this domain and make truthfulness prediction methods more accessible, we introduce TruthTorchLM an open-source, comprehensive Python library featuring over 30 truthfulness prediction methods, which we refer to as Truth Methods. Unlike existing toolkits such as Guardrails, which focus solely on document-grounded verification, or LM-Polygraph, which is limited to uncertainty-based methods, TruthTorchLM offers a broad and extensible collection of techniques. These methods span diverse tradeoffs in computational cost, access level (e.g., black-box vs white-box), grounding document requirements, and supervision type (self-supervised or supervised). TruthTorchLM is seamlessly compatible with both HuggingFace and LiteLLM, enabling support for locally hosted and API-based models. It also provides a unified interface for generation, evaluation, calibration, and long-form truthfulness prediction, along with a flexible framework for extending the library with new methods. We conduct an evaluation of representative truth methods on three datasets, TriviaQA, GSM8K, and FactScore-Bio. The code is available at this https URL 

---
# Distilling Empathy from Large Language Models 

**Authors**: Henry J. Xie, Jinghan Zhang, Xinhao Zhang, Kunpeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08151)  

**Abstract**: The distillation of knowledge from Large Language Models (LLMs) into Smaller Language Models (SLMs), preserving the capabilities and performance of LLMs while reducing model size, has played a key role in the proliferation of LLMs. Because SLMs are considerably smaller than LLMs, they are often utilized in domains where human interaction is frequent but resources are highly constrained, e.g., smart phones. Therefore, it is crucial to ensure that empathy, a fundamental aspect of positive human interactions, already instilled into LLMs, is retained by SLMs after distillation. In this paper, we develop a comprehensive approach for effective empathy distillation from LLMs into SLMs. Our approach features a two-step fine-tuning process that fully leverages datasets of empathetic dialogue responses distilled from LLMs. We explore several distillation methods beyond basic direct prompting and propose four unique sets of prompts for targeted empathy improvement to significantly enhance the empathy distillation process. Our evaluations demonstrate that SLMs fine-tuned through the two-step fine-tuning process with distillation datasets enhanced by the targeted empathy improvement prompts significantly outperform the base SLM at generating empathetic responses with a win rate of 90%. Our targeted empathy improvement prompts substantially outperform the basic direct prompting with a 10% improvement in win rate. 

---
# Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores 

**Authors**: Vivek Chari, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.08143)  

**Abstract**: Modern Large Language Models (LLMs) are increasingly trained to support very large context windows. Unfortunately the ability to use long contexts in generation is complicated by the large memory requirement of the KV cache, which scales linearly with the context length. This memory footprint is often the dominant resource bottleneck in real-world deployments, limiting throughput and increasing serving cost. One way to address this is by compressing the KV cache, which can be done either with knowledge of the question being asked (query-aware) or without knowledge of the query (query-agnostic). We present Compactor, a parameter-free, query-agnostic KV compression strategy that uses approximate leverage scores to determine token importance. We show that Compactor can achieve the same performance as competing methods while retaining 1/2 the tokens in both synthetic and real-world context tasks, with minimal computational overhead. We further introduce a procedure for context-calibrated compression, which allows one to infer the maximum compression ratio a given context can support. Using context-calibrated compression, we show that Compactor achieves full KV performance on Longbench while reducing the KV memory burden by 63%, on average. To demonstrate the efficacy and generalizability of our approach, we apply Compactor to 27 synthetic and real-world tasks from RULER and Longbench, with models from both the Qwen 2.5 and Llama 3.1 families. 

---
# Audit, Alignment, and Optimization of LM-Powered Subroutines with Application to Public Comment Processing 

**Authors**: Reilly Raab, Mike Parker, Dan Nally, Sadie Montgomery, Anastasia Bernat, Sai Munikoti, Sameera Horawalavithana  

**Link**: [PDF](https://arxiv.org/pdf/2507.08109)  

**Abstract**: The advent of language models (LMs) has the potential to dramatically accelerate tasks that may be cast to text-processing; however, real-world adoption is hindered by concerns regarding safety, explainability, and bias. How can we responsibly leverage LMs in a transparent, auditable manner -- minimizing risk and allowing human experts to focus on informed decision-making rather than data-processing or prompt engineering? In this work, we propose a framework for declaring statically typed, LM-powered subroutines (i.e., callable, function-like procedures) for use within conventional asynchronous code -- such that sparse feedback from human experts is used to improve the performance of each subroutine online (i.e., during use). In our implementation, all LM-produced artifacts (i.e., prompts, inputs, outputs, and data-dependencies) are recorded and exposed to audit on demand. We package this framework as a library to support its adoption and continued development. While this framework may be applicable across several real-world decision workflows (e.g., in healthcare and legal fields), we evaluate it in the context of public comment processing as mandated by the 1969 National Environmental Protection Act (NEPA): Specifically, we use this framework to develop "CommentNEPA," an application that compiles, organizes, and summarizes a corpus of public commentary submitted in response to a project requiring environmental review. We quantitatively evaluate the application by comparing its outputs (when operating without human feedback) to historical ``ground-truth'' data as labelled by human annotators during the preparation of official environmental impact statements. 

---
# GRASP: Generic Reasoning And SPARQL Generation across Knowledge Graphs 

**Authors**: Sebastian Walter, Hannah Bast  

**Link**: [PDF](https://arxiv.org/pdf/2507.08107)  

**Abstract**: We propose a new approach for generating SPARQL queries on RDF knowledge graphs from natural language questions or keyword queries, using a large language model. Our approach does not require fine-tuning. Instead, it uses the language model to explore the knowledge graph by strategically executing SPARQL queries and searching for relevant IRIs and literals. We evaluate our approach on a variety of benchmarks (for knowledge graphs of different kinds and sizes) and language models (of different scales and types, commercial as well as open-source) and compare it with existing approaches. On Wikidata we reach state-of-the-art results on multiple benchmarks, despite the zero-shot setting. On Freebase we come close to the best few-shot methods. On other, less commonly evaluated knowledge graphs and benchmarks our approach also performs well overall. We conduct several additional studies, like comparing different ways of searching the graphs, incorporating a feedback mechanism, or making use of few-shot examples. 

---
# Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing 

**Authors**: Junyi Wen, Junyuan Liang, Zicong Hong, Wuhui Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08045)  

**Abstract**: Efficient state restoration in multi-turn conversations with large language models (LLMs) remains a critical challenge, primarily due to the overhead of recomputing or loading full key-value (KV) caches for all historical tokens. To address this, existing approaches compress KV caches across adjacent layers with highly similar attention patterns. However, these methods often apply a fixed compression scheme across all conversations, selecting the same layer pairs for compression without considering conversation-specific attention dynamics. This static strategy overlooks variability in attention pattern similarity across different conversations, which can lead to noticeable accuracy degradation.
We present Krul, a multi-turn LLM inference system that enables accurate and efficient KV cache restoration. Krul dynamically selects compression strategies based on attention similarity across layer pairs and uses a recomputation-loading pipeline to restore the KV cache. It introduces three key innovations: 1) a preemptive compression strategy selector to preserve critical context for future conversation turns and selects a customized strategy for the conversation; 2) a token-wise heterogeneous attention similarity estimator to mitigate the attention similarity computation and storage overhead during model generation; 3) a bubble-free restoration scheduler to reduce potential bubbles brought by the imbalance of recomputing and loading stream due to compressed KV caches. Empirical evaluations on real-world tasks demonstrate that Krul achieves a 1.5x-2.68x reduction in time-to-first-token (TTFT) and a 1.33x-2.35x reduction in KV cache storage compared to state-of-the-art methods without compromising generation quality. 

---
# AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research 

**Authors**: Talor Abramovich, Gal Chechik  

**Link**: [PDF](https://arxiv.org/pdf/2507.08038)  

**Abstract**: Autonomous agents built on language models (LMs) are showing increasing popularity in many fields, including scientific research. AI co-scientists aim to support or automate parts of the research process using these agents. A key component of empirical AI research is the design of ablation experiments. To this end, we introduce AblationBench, a benchmark suite for evaluating agents on ablation planning tasks in empirical AI research. It includes two tasks: AuthorAblation, which helps authors propose ablation experiments based on a method section and contains 83 instances, and ReviewerAblation, which helps reviewers find missing ablations in a full paper and contains 350 instances. For both tasks, we develop LM-based judges that serve as an automatic evaluation framework. Our experiments with frontier LMs show that these tasks remain challenging, with the best-performing LM system identifying only 29% of the original ablations on average. Lastly, we analyze the limitations of current LMs on these tasks, and find that chain-of-thought prompting outperforms the currently existing agent-based approach. 

---
# CRISP: Complex Reasoning with Interpretable Step-based Plans 

**Authors**: Matan Vetzler, Koren Lazar, Guy Uziel, Eran Hirsch, Ateret Anaby-Tavor, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08037)  

**Abstract**: Recent advancements in large language models (LLMs) underscore the need for stronger reasoning capabilities to solve complex problems effectively. While Chain-of-Thought (CoT) reasoning has been a step forward, it remains insufficient for many domains. A promising alternative is explicit high-level plan generation, but existing approaches largely assume that LLMs can produce effective plans through few-shot prompting alone, without additional training. In this work, we challenge this assumption and introduce CRISP (Complex Reasoning with Interpretable Step-based Plans), a multi-domain dataset of high-level plans for mathematical reasoning and code generation. The plans in CRISP are automatically generated and rigorously validated--both intrinsically, using an LLM as a judge, and extrinsically, by evaluating their impact on downstream task performance. We demonstrate that fine-tuning a small model on CRISP enables it to generate higher-quality plans than much larger models using few-shot prompting, while significantly outperforming Chain-of-Thought reasoning. Furthermore, our out-of-domain evaluation reveals that fine-tuning on one domain improves plan generation in the other, highlighting the generalizability of learned planning capabilities. 

---
# Barriers in Integrating Medical Visual Question Answering into Radiology Workflows: A Scoping Review and Clinicians' Insights 

**Authors**: Deepali Mishra, Chaklam Silpasuwanchai, Ashutosh Modi, Madhumita Sushil, Sorayouth Chumnanvej  

**Link**: [PDF](https://arxiv.org/pdf/2507.08036)  

**Abstract**: Medical Visual Question Answering (MedVQA) is a promising tool to assist radiologists by automating medical image interpretation through question answering. Despite advances in models and datasets, MedVQA's integration into clinical workflows remains limited. This study systematically reviews 68 publications (2018-2024) and surveys 50 clinicians from India and Thailand to examine MedVQA's practical utility, challenges, and gaps. Following the Arksey and O'Malley scoping review framework, we used a two-pronged approach: (1) reviewing studies to identify key concepts, advancements, and research gaps in radiology workflows, and (2) surveying clinicians to capture their perspectives on MedVQA's clinical relevance. Our review reveals that nearly 60% of QA pairs are non-diagnostic and lack clinical relevance. Most datasets and models do not support multi-view, multi-resolution imaging, EHR integration, or domain knowledge, features essential for clinical diagnosis. Furthermore, there is a clear mismatch between current evaluation metrics and clinical needs. The clinician survey confirms this disconnect: only 29.8% consider MedVQA systems highly useful. Key concerns include the absence of patient history or domain knowledge (87.2%), preference for manually curated datasets (51.1%), and the need for multi-view image support (78.7%). Additionally, 66% favor models focused on specific anatomical regions, and 89.4% prefer dialogue-based interactive systems. While MedVQA shows strong potential, challenges such as limited multimodal analysis, lack of patient context, and misaligned evaluation approaches must be addressed for effective clinical integration. 

---
# Integrating External Tools with Large Language Models to Improve Accuracy 

**Authors**: Nripesh Niketan, Hadj Batatia  

**Link**: [PDF](https://arxiv.org/pdf/2507.08034)  

**Abstract**: This paper deals with improving querying large language models (LLMs). It is well-known that without relevant contextual information, LLMs can provide poor quality responses or tend to hallucinate. Several initiatives have proposed integrating LLMs with external tools to provide them with up-to-date data to improve accuracy. In this paper, we propose a framework to integrate external tools to enhance the capabilities of LLMs in answering queries in educational settings. Precisely, we develop a framework that allows accessing external APIs to request additional relevant information. Integrated tools can also provide computational capabilities such as calculators or calendars. The proposed framework has been evaluated using datasets from the Multi-Modal Language Understanding (MMLU) collection. The data consists of questions on mathematical and scientific reasoning. Results compared to state-of-the-art language models show that the proposed approach significantly improves performance. Our Athena framework achieves 83% accuracy in mathematical reasoning and 88% in scientific reasoning, substantially outperforming all tested models including GPT-4o, LLaMA-Large, Mistral-Large, Phi-Large, and GPT-3.5, with the best baseline model (LLaMA-Large) achieving only 67% and 79% respectively. These promising results open the way to creating complex computing ecosystems around LLMs to make their use more natural to support various tasks and activities. 

---
# Beyond Scale: Small Language Models are Comparable to GPT-4 in Mental Health Understanding 

**Authors**: Hong Jia, Shiya Fu, Vassilis Kostakos, Feng Xia, Ting Dang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08031)  

**Abstract**: The emergence of Small Language Models (SLMs) as privacy-preserving alternatives for sensitive applications raises a fundamental question about their inherent understanding capabilities compared to Large Language Models (LLMs). This paper investigates the mental health understanding capabilities of current SLMs through systematic evaluation across diverse classification tasks. Employing zero-shot and few-shot learning paradigms, we benchmark their performance against established LLM baselines to elucidate their relative strengths and limitations in this critical domain. We assess five state-of-the-art SLMs (Phi-3, Phi-3.5, Qwen2.5, Llama-3.2, Gemma2) against three LLMs (GPT-4, FLAN-T5-XXL, Alpaca-7B) on six mental health understanding tasks. Our findings reveal that SLMs achieve mean performance within 2\% of LLMs on binary classification tasks (F1 scores of 0.64 vs 0.66 in zero-shot settings), demonstrating notable competence despite orders of magnitude fewer parameters. Both model categories experience similar degradation on multi-class severity tasks (a drop of over 30\%), suggesting that nuanced clinical understanding challenges transcend model scale. Few-shot prompting provides substantial improvements for SLMs (up to 14.6\%), while LLM gains are more variable. Our work highlights the potential of SLMs in mental health understanding, showing they can be effective privacy-preserving tools for analyzing sensitive online text data. In particular, their ability to quickly adapt and specialize with minimal data through few-shot learning positions them as promising candidates for scalable mental health screening tools. 

---
# A Systematic Analysis of Declining Medical Safety Messaging in Generative AI Models 

**Authors**: Sonali Sharma, Ahmed M. Alaa, Roxana Daneshjou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08030)  

**Abstract**: Generative AI models, including large language models (LLMs) and vision-language models (VLMs), are increasingly used to interpret medical images and answer clinical questions. Their responses often include inaccuracies; therefore, safety measures like medical disclaimers are critical to remind users that AI outputs are not professionally vetted or a substitute for medical advice. This study evaluated the presence of disclaimers in LLM and VLM outputs across model generations from 2022 to 2025. Using 500 mammograms, 500 chest X-rays, 500 dermatology images, and 500 medical questions, outputs were screened for disclaimer phrases. Medical disclaimer presence in LLM and VLM outputs dropped from 26.3% in 2022 to 0.97% in 2025, and from 19.6% in 2023 to 1.05% in 2025, respectively. By 2025, the majority of models displayed no disclaimers. As public models become more capable and authoritative, disclaimers must be implemented as a safeguard adapting to the clinical context of each output. 

---
# Better Together: Quantifying the Benefits of AI-Assisted Recruitment 

**Authors**: Ada Aka, Emil Palikot, Ali Ansari, Nima Yazdani  

**Link**: [PDF](https://arxiv.org/pdf/2507.08029)  

**Abstract**: Artificial intelligence (AI) is increasingly used in recruitment, yet empirical evidence quantifying its impact on hiring efficiency and candidate selection remains limited. We randomly assign 37,000 applicants for a junior-developer position to either a traditional recruitment process (resume screening followed by human selection) or an AI-assisted recruitment pipeline incorporating an initial AI-driven structured video interview before human evaluation. Candidates advancing from either track faced the same final-stage human interview, with interviewers blind to the earlier selection method. In the AI-assisted pipeline, 54% of candidates passed the final interview compared with 34% from the traditional pipeline, yielding an average treatment effect of 20 percentage points (SE 12 pp.). Five months later, we collected LinkedIn profiles of top applicants from both groups and found that 18% (SE 1.1%) of applicants from the traditional track found new jobs compared with 23% (SE 2.3%) from the AI group, resulting in a 5.9 pp. (SE 2.6 pp.) difference in the probability of finding new employment between groups. The AI system tended to select younger applicants with less experience and fewer advanced credentials. We analyze AI-generated interview transcripts to examine the selection criteria and conversational dynamics. Our findings contribute to understanding how AI technologies affect decision making in recruitment and talent acquisition while highlighting some of their potential implications. 

---
# "Amazing, They All Lean Left" -- Analyzing the Political Temperaments of Current LLMs 

**Authors**: W. Russell Neuman, Chad Coleman, Ali Dasdan, Safinah Ali, Manan Shah, Kund Meghani  

**Link**: [PDF](https://arxiv.org/pdf/2507.08027)  

**Abstract**: Recent studies have revealed a consistent liberal orientation in the ethical and political responses generated by most commercial large language models (LLMs), yet the underlying causes and resulting implications remain unclear. This paper systematically investigates the political temperament of seven prominent LLMs - OpenAI's GPT-4o, Anthropic's Claude Sonnet 4, Perplexity (Sonar Large), Google's Gemini 2.5 Flash, Meta AI's Llama 4, Mistral 7b Le Chat and High-Flyer's DeepSeek R1 -- using a multi-pronged approach that includes Moral Foundations Theory, a dozen established political ideology scales and a new index of current political controversies. We find strong and consistent prioritization of liberal-leaning values, particularly care and fairness, across most models. Further analysis attributes this trend to four overlapping factors: Liberal-leaning training corpora, reinforcement learning from human feedback (RLHF), the dominance of liberal frameworks in academic ethical discourse and safety-driven fine-tuning practices. We also distinguish between political "bias" and legitimate epistemic differences, cautioning against conflating the two. A comparison of base and fine-tuned model pairs reveals that fine-tuning generally increases liberal lean, an effect confirmed through both self-report and empirical testing. We argue that this "liberal tilt" is not a programming error or the personal preference of programmers but an emergent property of training on democratic rights-focused discourse. Finally, we propose that LLMs may indirectly echo John Rawls' famous veil-of ignorance philosophical aspiration, reflecting a moral stance unanchored to personal identity or interest. Rather than undermining democratic discourse, this pattern may offer a new lens through which to examine collective reasoning. 

---
# Unveiling Effective In-Context Configurations for Image Captioning: An External & Internal Analysis 

**Authors**: Li Li, Yongliang Wu, Jingze Zhu, Jiawei Peng, Jianfei Cai, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08021)  

**Abstract**: The evolution of large models has witnessed the emergence of In-Context Learning (ICL) capabilities. In Natural Language Processing (NLP), numerous studies have demonstrated the effectiveness of ICL. Inspired by the success of Large Language Models (LLMs), researchers have developed Large Multimodal Models (LMMs) with ICL capabilities. However, explorations of demonstration configuration for multimodal ICL remain preliminary. Additionally, the controllability of In-Context Examples (ICEs) provides an efficient and cost-effective means to observe and analyze the inference characteristics of LMMs under varying inputs. This paper conducts a comprehensive external and internal investigation of multimodal in-context learning on the image captioning task. Externally, we explore demonstration configuration strategies through three dimensions: shot number, image retrieval, and caption assignment. We employ multiple metrics to systematically and thoroughly evaluate and summarize key findings. Internally, we analyze typical LMM attention characteristics and develop attention-based metrics to quantify model behaviors. We also conduct auxiliary experiments to explore the feasibility of attention-driven model acceleration and compression. We further compare performance variations between LMMs with identical model design and pretraining strategies and explain the differences from the angles of pre-training data features. Our study reveals both how ICEs configuration strategies impact model performance through external experiments and characteristic typical patterns through internal inspection, providing dual perspectives for understanding multimodal ICL in LMMs. Our method of combining external and internal analysis to investigate large models, along with our newly proposed metrics, can be applied to broader research areas. 

---
# Circumventing Safety Alignment in Large Language Models Through Embedding Space Toxicity Attenuation 

**Authors**: Zhibo Zhang, Yuxi Li, Kailong Wang, Shuai Yuan, Ling Shi, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08020)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across domains such as healthcare, education, and cybersecurity. However, this openness also introduces significant security risks, particularly through embedding space poisoning, which is a subtle attack vector where adversaries manipulate the internal semantic representations of input data to bypass safety alignment mechanisms. While previous research has investigated universal perturbation methods, the dynamics of LLM safety alignment at the embedding level remain insufficiently understood. Consequently, more targeted and accurate adversarial perturbation techniques, which pose significant threats, have not been adequately studied.
In this work, we propose ETTA (Embedding Transformation Toxicity Attenuation), a novel framework that identifies and attenuates toxicity-sensitive dimensions in embedding space via linear transformations. ETTA bypasses model refusal behaviors while preserving linguistic coherence, without requiring model fine-tuning or access to training data. Evaluated on five representative open-source LLMs using the AdvBench benchmark, ETTA achieves a high average attack success rate of 88.61%, outperforming the best baseline by 11.34%, and generalizes to safety-enhanced models (e.g., 77.39% ASR on instruction-tuned defenses). These results highlight a critical vulnerability in current alignment strategies and underscore the need for embedding-aware defenses. 

---
# Signal or Noise? Evaluating Large Language Models in Resume Screening Across Contextual Variations and Human Expert Benchmarks 

**Authors**: Aryan Varshney, Venkat Ram Reddy Ganuthula  

**Link**: [PDF](https://arxiv.org/pdf/2507.08019)  

**Abstract**: This study investigates whether large language models (LLMs) exhibit consistent behavior (signal) or random variation (noise) when screening resumes against job descriptions, and how their performance compares to human experts. Using controlled datasets, we tested three LLMs (Claude, GPT, and Gemini) across contexts (No Company, Firm1 [MNC], Firm2 [Startup], Reduced Context) with identical and randomized resumes, benchmarked against three human recruitment experts. Analysis of variance revealed significant mean differences in four of eight LLM-only conditions and consistently significant differences between LLM and human evaluations (p < 0.01). Paired t-tests showed GPT adapts strongly to company context (p < 0.001), Gemini partially (p = 0.038 for Firm1), and Claude minimally (p > 0.1), while all LLMs differed significantly from human experts across contexts. Meta-cognition analysis highlighted adaptive weighting patterns that differ markedly from human evaluation approaches. Findings suggest LLMs offer interpretable patterns with detailed prompts but diverge substantially from human judgment, informing their deployment in automated hiring systems. 

---
# Review, Remask, Refine (R3): Process-Guided Block Diffusion for Text Generation 

**Authors**: Nikita Mounier, Parsa Idehpour  

**Link**: [PDF](https://arxiv.org/pdf/2507.08018)  

**Abstract**: A key challenge for iterative text generation is enabling models to efficiently identify and correct their own errors. We propose Review, Remask, Refine (R3), a relatively simple yet elegant framework that requires no additional model training and can be applied to any pre-trained masked text diffusion model (e.g., LLaDA or BD3-LM). In R3, a Process Reward Model (PRM) is utilized for the Review of intermediate generated blocks. The framework then translates these PRM scores into a Remask strategy: the lower a block's PRM score, indicating potential mistakes, the greater the proportion of tokens within that block are remasked. Finally, the model is compelled to Refine these targeted segments, focusing its efforts more intensively on specific sub-optimal parts of past generations, leading to improved final output. 

---
# Mechanistic Indicators of Understanding in Large Language Models 

**Authors**: Pierre Beckmann, Matthieu Queloz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08017)  

**Abstract**: Recent findings in mechanistic interpretability (MI), the field probing the inner workings of Large Language Models (LLMs), challenge the view that these models rely solely on superficial statistics. Here, we offer an accessible synthesis of these findings that doubles as an introduction to MI, all while integrating these findings within a novel theoretical framework for thinking about machine understanding. We argue that LLMs develop internal structures that are functionally analogous to the kind of understanding that consists in seeing connections. To sharpen this idea, we propose a three-tiered conception of machine understanding. First, conceptual understanding emerges when a model forms "features" as directions in latent space, thereby learning the connections between diverse manifestations of something. Second, state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world. Third, principled understanding emerges when a model ceases to rely on a collection of memorized facts and discovers a "circuit" that connects these facts. However, we conclude by exploring the "parallel mechanisms" phenomenon, arguing that while LLMs exhibit forms of understanding, their cognitive architecture remains different from ours, and the debate should shift from whether LLMs understand to how their strange minds work. 

---
# Assessing the Capabilities and Limitations of FinGPT Model in Financial NLP Applications 

**Authors**: Prudence Djagba, Chimezie A. Odinakachukwu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08015)  

**Abstract**: This work evaluates FinGPT, a financial domain-specific language model, across six key natural language processing (NLP) tasks: Sentiment Analysis, Text Classification, Named Entity Recognition, Financial Question Answering, Text Summarization, and Stock Movement Prediction. The evaluation uses finance-specific datasets to assess FinGPT's capabilities and limitations in real-world financial applications. The results show that FinGPT performs strongly in classification tasks such as sentiment analysis and headline categorization, often achieving results comparable to GPT-4. However, its performance is significantly lower in tasks that involve reasoning and generation, such as financial question answering and summarization. Comparisons with GPT-4 and human benchmarks highlight notable performance gaps, particularly in numerical accuracy and complex reasoning. Overall, the findings indicate that while FinGPT is effective for certain structured financial tasks, it is not yet a comprehensive solution. This research provides a useful benchmark for future research and underscores the need for architectural improvements and domain-specific optimization in financial language models. 

---
# Mass-Scale Analysis of In-the-Wild Conversations Reveals Complexity Bounds on LLM Jailbreaking 

**Authors**: Aldan Creo, Raul Castro Fernandez, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2507.08014)  

**Abstract**: As large language models (LLMs) become increasingly deployed, understanding the complexity and evolution of jailbreaking strategies is critical for AI safety.
We present a mass-scale empirical analysis of jailbreak complexity across over 2 million real-world conversations from diverse platforms, including dedicated jailbreaking communities and general-purpose chatbots. Using a range of complexity metrics spanning probabilistic measures, lexical diversity, compression ratios, and cognitive load indicators, we find that jailbreak attempts do not exhibit significantly higher complexity than normal conversations. This pattern holds consistently across specialized jailbreaking communities and general user populations, suggesting practical bounds on attack sophistication. Temporal analysis reveals that while user attack toxicity and complexity remains stable over time, assistant response toxicity has decreased, indicating improving safety mechanisms. The absence of power-law scaling in complexity distributions further points to natural limits on jailbreak development.
Our findings challenge the prevailing narrative of an escalating arms race between attackers and defenders, instead suggesting that LLM safety evolution is bounded by human ingenuity constraints while defensive measures continue advancing. Our results highlight critical information hazards in academic jailbreak disclosure, as sophisticated attacks exceeding current complexity baselines could disrupt the observed equilibrium and enable widespread harm before defensive adaptation. 

---
# MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model 

**Authors**: K. Sahit Reddy, N. Ragavenderan, Vasanth K., Ganesh N. Naik, Vishalakshi Prabhu, Nagaraja G. S  

**Link**: [PDF](https://arxiv.org/pdf/2507.08013)  

**Abstract**: Recent advances in natural language processing (NLP) have been driven bypretrained language models like BERT, RoBERTa, T5, and GPT. Thesemodels excel at understanding complex texts, but biomedical literature, withits domain-specific terminology, poses challenges that models likeWord2Vec and bidirectional long short-term memory (Bi-LSTM) can't fullyaddress. GPT and T5, despite capturing context, fall short in tasks needingbidirectional understanding, unlike BERT. Addressing this, we proposedMedicalBERT, a pretrained BERT model trained on a large biomedicaldataset and equipped with domain-specific vocabulary that enhances thecomprehension of biomedical terminology. MedicalBERT model is furtheroptimized and fine-tuned to address diverse tasks, including named entityrecognition, relation extraction, question answering, sentence similarity, anddocument classification. Performance metrics such as the F1-score,accuracy, and Pearson correlation are employed to showcase the efficiencyof our model in comparison to other BERT-based models such as BioBERT,SciBERT, and ClinicalBERT. MedicalBERT outperforms these models onmost of the benchmarks, and surpasses the general-purpose BERT model by5.67% on average across all the tasks evaluated respectively. This work alsounderscores the potential of leveraging pretrained BERT models for medicalNLP tasks, demonstrating the effectiveness of transfer learning techniques incapturing domain-specific information.
(PDF) MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model. Available from: this https URL [accessed Jul 06 2025]. 

---
# RepeaTTS: Towards Feature Discovery through Repeated Fine-Tuning 

**Authors**: Atli Sigurgeirsson, Simon King  

**Link**: [PDF](https://arxiv.org/pdf/2507.08012)  

**Abstract**: A Prompt-based Text-To-Speech model allows a user to control different aspects of speech, such as speaking rate and perceived gender, through natural language instruction. Although user-friendly, such approaches are on one hand constrained: control is limited to acoustic features exposed to the model during training, and too flexible on the other: the same inputs yields uncontrollable variation that are reflected in the corpus statistics.
We investigate a novel fine-tuning regime to address both of these issues at the same time by exploiting the uncontrollable variance of the model. Through principal component analysis of thousands of synthesised samples, we determine latent features that account for the highest proportion of the output variance and incorporate them as new labels for secondary fine-tuning. We evaluate the proposed methods on two models trained on an expressive Icelandic speech corpus, one with emotional disclosure and one without. In the case of the model without emotional disclosure, the method yields both continuous and discrete features that improve overall controllability of the model. 

---
# NeuralOS: Towards Simulating Operating Systems via Neural Generative Models 

**Authors**: Luke Rivard, Sun Sun, Hongyu Guo, Wenhu Chen, Yuntian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08800)  

**Abstract**: We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames in response to user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates screen images. The model is trained on a large-scale dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Experiments show that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Although modeling fine-grained keyboard interactions precisely remains challenging, NeuralOS offers a step toward creating fully adaptive, generative neural interfaces for future human-computer interaction systems. 

---
# One Token to Fool LLM-as-a-Judge 

**Authors**: Yulai Zhao, Haolin Liu, Dian Yu, S.Y. Kung, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08794)  

**Abstract**: Generative reward models (also known as LLMs-as-judges), which use large language models (LLMs) to evaluate answer quality, are increasingly adopted in reinforcement learning with verifiable rewards (RLVR). They are often preferred over rigid rule-based metrics, especially for complex reasoning tasks involving free-form outputs. In this paradigm, an LLM is typically prompted to compare a candidate answer against a ground-truth reference and assign a binary reward indicating correctness. Despite the seeming simplicity of this comparison task, we find that generative reward models exhibit surprising vulnerabilities to superficial manipulations: non-word symbols (e.g., ":" or ".") or reasoning openers like "Thought process:" and "Let's solve this problem step by step." can often lead to false positive rewards. We demonstrate that this weakness is widespread across LLMs, datasets, and prompt formats, posing a serious threat for core algorithmic paradigms that rely on generative reward models, such as rejection sampling, preference optimization, and RLVR. To mitigate this issue, we introduce a simple yet effective data augmentation strategy and train a new generative reward model with substantially improved robustness. Our findings highlight the urgent need for more reliable LLM-based evaluation methods. We release our robust, general-domain reward model and its synthetic training data at this https URL and this https URL. 

---
# BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity 

**Authors**: Chenyang Song, Weilin Zhao, Xu Han, Chaojun Xiao, Yingfa Chen, Yuxuan Li, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.08771)  

**Abstract**: To alleviate the computational burden of large language models (LLMs), architectures with activation sparsity, represented by mixture-of-experts (MoE), have attracted increasing attention. However, the non-differentiable and inflexible routing of vanilla MoE hurts model performance. Moreover, while each token activates only a few parameters, these sparsely-activated architectures exhibit low chunk-level sparsity, indicating that the union of multiple consecutive tokens activates a large ratio of parameters. Such a sparsity pattern is unfriendly for acceleration under low-resource conditions (e.g., end-side devices) and incompatible with mainstream acceleration techniques (e.g., speculative decoding). To address these challenges, we introduce a novel MoE architecture, BlockFFN, as well as its efficient training and deployment techniques. Specifically, we use a router integrating ReLU activation and RMSNorm for differentiable and flexible routing. Next, to promote both token-level sparsity (TLS) and chunk-level sparsity (CLS), CLS-aware training objectives are designed, making BlockFFN more acceleration-friendly. Finally, we implement efficient acceleration kernels, combining activation sparsity and speculative decoding for the first time. The experimental results demonstrate the superior performance of BlockFFN over other MoE baselines, achieving over 80% TLS and 70% 8-token CLS. Our kernels achieve up to 3.67$\times$ speedup on real end-side devices than dense models. All codes and checkpoints are available publicly (this https URL). 

---
# On Barriers to Archival Audio Processing 

**Authors**: Peter Sullivan, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2507.08768)  

**Abstract**: In this study, we leverage a unique UNESCO collection of mid-20th century radio recordings to probe the robustness of modern off-the-shelf language identification (LID) and speaker recognition (SR) methods, especially with respect to the impact of multilingual speakers and cross-age recordings. Our findings suggest that LID systems, such as Whisper, are increasingly adept at handling second-language and accented speech. However, speaker embeddings remain a fragile component of speech processing pipelines that is prone to biases related to the channel, age, and language. Issues which will need to be overcome should archives aim to employ SR methods for speaker indexing. 

---
# Scaling Attention to Very Long Sequences in Linear Time with Wavelet-Enhanced Random Spectral Attention (WERSA) 

**Authors**: Vincenzo Dentamaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08637)  

**Abstract**: Transformer models are computationally costly on long sequences since regular attention has quadratic $O(n^2)$ time complexity. We introduce Wavelet-Enhanced Random Spectral Attention (WERSA), a novel mechanism of linear $O(n)$ time complexity that is pivotal to enable successful long-sequence processing without the performance trade-off. WERSA merges content-adaptive random spectral features together with multi-resolution Haar wavelets and learnable parameters to selectively attend to informative scales of data while preserving linear efficiency.
Large-scale comparisons \textbf{on single GPU} and across various benchmarks (vision, NLP, hierarchical reasoning) and various attention mechanisms (like Multiheaded Attention, Flash-Attention-2, FNet, Linformer, Performer, Waveformer), reveal uniform advantages of WERSA. It achieves best accuracy in all tests. On ArXiv classification, WERSA improves accuracy over vanilla attention by 1.2\% (86.2\% vs 85.0\%) while cutting training time by 81\% (296s vs 1554s) and FLOPS by 73.4\% (26.2G vs 98.4G). Significantly, WERSA excels where vanilla and FlashAttention-2 fail: on ArXiv-128k's extremely lengthy sequences, it achieves best accuracy (79.1\%) and AUC (0.979) among viable methods, operating on data that gives Out-Of-Memory errors to quadratic methods while being \textbf{twice as fast} as Waveformer, its next-best competitor.
By significantly reducing computational loads without compromising accuracy, WERSA makes possible more practical, more affordable, long-context models, in particular on low-resource hardware, for more sustainable and more scalable AI development. 

---
# Large Multi-modal Model Cartographic Map Comprehension for Textual Locality Georeferencing 

**Authors**: Kalana Wijegunarathna, Kristin Stock, Christopher B. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2507.08575)  

**Abstract**: Millions of biological sample records collected in the last few centuries archived in natural history collections are un-georeferenced. Georeferencing complex locality descriptions associated with these collection samples is a highly labour-intensive task collection agencies struggle with. None of the existing automated methods exploit maps that are an essential tool for georeferencing complex relations. We present preliminary experiments and results of a novel method that exploits multi-modal capabilities of recent Large Multi-Modal Models (LMM). This method enables the model to visually contextualize spatial relations it reads in the locality description. We use a grid-based approach to adapt these auto-regressive models for this task in a zero-shot setting. Our experiments conducted on a small manually annotated dataset show impressive results for our approach ($\sim$1 km Average distance error) compared to uni-modal georeferencing with Large Language Models and existing georeferencing tools. The paper also discusses the findings of the experiments in light of an LMM's ability to comprehend fine-grained maps. Motivated by these results, a practical framework is proposed to integrate this method into a georeferencing workflow. 

---
# A Multi-granularity Concept Sparse Activation and Hierarchical Knowledge Graph Fusion Framework for Rare Disease Diagnosis 

**Authors**: Mingda Zhang, Na Zhao, Jianglong Qin, Guoyu Ye, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08529)  

**Abstract**: Despite advances from medical large language models in healthcare, rare-disease diagnosis remains hampered by insufficient knowledge-representation depth, limited concept understanding, and constrained clinical reasoning. We propose a framework that couples multi-granularity sparse activation of medical concepts with a hierarchical knowledge graph. Four complementary matching algorithms, diversity control, and a five-level fallback strategy enable precise concept activation, while a three-layer knowledge graph (taxonomy, clinical features, instances) provides structured, up-to-date context. Experiments on the BioASQ rare-disease QA set show BLEU gains of 0.09, ROUGE gains of 0.05, and accuracy gains of 0.12, with peak accuracy of 0.89 approaching the 0.90 clinical threshold. Expert evaluation confirms improvements in information quality, reasoning, and professional expression, suggesting our approach shortens the "diagnostic odyssey" for rare-disease patients. 

---
# xpSHACL: Explainable SHACL Validation using Retrieval-Augmented Generation and Large Language Models 

**Authors**: Gustavo Correa Publio, José Emilio Labra Gayo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08432)  

**Abstract**: Shapes Constraint Language (SHACL) is a powerful language for validating RDF data. Given the recent industry attention to Knowledge Graphs (KGs), more users need to validate linked data properly. However, traditional SHACL validation engines often provide terse reports in English that are difficult for non-technical users to interpret and act upon. This paper presents xpSHACL, an explainable SHACL validation system that addresses this issue by combining rule-based justification trees with retrieval-augmented generation (RAG) and large language models (LLMs) to produce detailed, multilanguage, human-readable explanations for constraint violations. A key feature of xpSHACL is its usage of a Violation KG to cache and reuse explanations, improving efficiency and consistency. 

---
# M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning 

**Authors**: Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, Yi Yuan, Yifan Mao, Yuting Xiao, Ziping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08306)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs), particularly through Reinforcement Learning with Verifiable Rewards (RLVR), have significantly enhanced their reasoning abilities. However, a critical gap persists: these models struggle with dynamic spatial interactions, a capability essential for real-world applications. To bridge this gap, we introduce M2-Reasoning-7B, a model designed to excel in both general and spatial reasoning. Our approach integrates two key innovations: (1) a novel data pipeline that generates 294.2K high-quality data samples (168K for cold-start fine-tuning and 126.2K for RLVR), which feature logically coherent reasoning trajectories and have undergone comprehensive assessment; and (2) a dynamic multi-task training strategy with step-wise optimization to mitigate conflicts between data, and task-specific rewards for delivering tailored incentive signals. This combination of curated data and advanced training allows M2-Reasoning-7B to set a new state-of-the-art (SOTA) across 8 benchmarks, showcasing superior performance in both general and spatial reasoning domains. 

---
# Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training 

**Authors**: Aleksei Ilin, Gor Matevosyan, Xueying Ma, Vladimir Eremin, Suhaa Dada, Muqun Li, Riyaaz Shaik, Haluk Noyan Tokgozoglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08284)  

**Abstract**: We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems. 

---
# Overview of the TREC 2021 deep learning track 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08191)  

**Abstract**: This is the third year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human annotated training labels available for both passage and document ranking tasks. In addition, this year we refreshed both the document and the passage collections which also led to a nearly four times increase in the document collection size and nearly $16$ times increase in the size of the passage collection. Deep neural ranking models that employ large scale pretraininig continued to outperform traditional retrieval methods this year. We also found that single stage retrieval can achieve good performance on both tasks although they still do not perform at par with multistage retrieval pipelines. Finally, the increase in the collection size and the general data refresh raised some questions about completeness of NIST judgments and the quality of the training labels that were mapped to the new collections from the old ones which we discuss in this report. 

---
# Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models 

**Authors**: Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08128)  

**Abstract**: We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets. 

---
# VideoConviction: A Multimodal Benchmark for Human Conviction and Stock Market Recommendations 

**Authors**: Michael Galarnyk, Veer Kejriwal, Agam Shah, Yash Bhardwaj, Nicholas Meyer, Anand Krishnan, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2507.08104)  

**Abstract**: Social media has amplified the reach of financial influencers known as "finfluencers," who share stock recommendations on platforms like YouTube. Understanding their influence requires analyzing multimodal signals like tone, delivery style, and facial expressions, which extend beyond text-based financial analysis. We introduce VideoConviction, a multimodal dataset with 6,000+ expert annotations, produced through 457 hours of human effort, to benchmark multimodal large language models (MLLMs) and text-based large language models (LLMs) in financial discourse. Our results show that while multimodal inputs improve stock ticker extraction (e.g., extracting Apple's ticker AAPL), both MLLMs and LLMs struggle to distinguish investment actions and conviction--the strength of belief conveyed through confident delivery and detailed reasoning--often misclassifying general commentary as definitive recommendations. While high-conviction recommendations perform better than low-conviction ones, they still underperform the popular S\&P 500 index fund. An inverse strategy--betting against finfluencer recommendations--outperforms the S\&P 500 by 6.8\% in annual returns but carries greater risk (Sharpe ratio of 0.41 vs. 0.65). Our benchmark enables a diverse evaluation of multimodal tasks, comparing model performance on both full video and segmented video inputs. This enables deeper advancements in multimodal financial research. Our code, dataset, and evaluation leaderboard are available under the CC BY-NC 4.0 license. 

---
