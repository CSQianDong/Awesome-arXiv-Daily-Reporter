# Text-to-SQL Task-oriented Dialogue Ontology Construction 

**Authors**: Renato Vukovic, Carel van Niekerk, Michael Heck, Benjamin Ruppik, Hsien-Chin Lin, Shutong Feng, Nurul Lubis, Milica Gasic  

**Link**: [PDF](https://arxiv.org/pdf/2507.23358)  

**Abstract**: Large language models (LLMs) are widely used as general-purpose knowledge sources, but they rely on parametric knowledge, limiting explainability and trustworthiness. In task-oriented dialogue (TOD) systems, this separation is explicit, using an external database structured by an explicit ontology to ensure explainability and controllability. However, building such ontologies requires manual labels or supervised training. We introduce TeQoDO: a Text-to-SQL task-oriented Dialogue Ontology construction method. Here, an LLM autonomously builds a TOD ontology from scratch without supervision using its inherent SQL programming capabilities combined with dialogue theory provided in the prompt. We show that TeQoDO outperforms transfer learning approaches, and its constructed ontology is competitive on a downstream dialogue state tracking task. Ablation studies demonstrate the key role of dialogue theory. TeQoDO also scales to allow construction of much larger ontologies, which we investigate on a Wikipedia and ArXiv dataset. We view this as a step towards broader application of ontologies to increase LLM explainability. 

---
# Towards LLM-Enhanced Product Line Scoping 

**Authors**: Alexander Felfernig, Damian Garber, Viet-Man Le, Sebastian Lubos, Thi Ngoc Trang Tran  

**Link**: [PDF](https://arxiv.org/pdf/2507.23410)  

**Abstract**: The idea of product line scoping is to identify the set of features and configurations that a product line should include, i.e., offer for configuration purposes. In this context, a major scoping task is to find a balance between commercial relevance and technical feasibility. Traditional product line scoping approaches rely on formal feature models and require a manual analysis which can be quite time-consuming. In this paper, we sketch how Large Language Models (LLMs) can be applied to support product line scoping tasks with a natural language interaction based scoping process. Using a working example from the smarthome domain, we sketch how LLMs can be applied to evaluate different feature model alternatives. We discuss open research challenges regarding the integration of LLMs with product line scoping. 

---
# MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation 

**Authors**: Daeyong Kwon, SeungHeon Doh, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2507.23334)  

**Abstract**: Recent advancements in Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains. While they exhibit strong zero-shot performance on various tasks, LLMs' effectiveness in music-related applications remains limited due to the relatively small proportion of music-specific knowledge in their training data. To address this limitation, we propose MusT-RAG, a comprehensive framework based on Retrieval Augmented Generation (RAG) to adapt general-purpose LLMs for text-only music question answering (MQA) tasks. RAG is a technique that provides external knowledge to LLMs by retrieving relevant context information when generating answers to questions. To optimize RAG for the music domain, we (1) propose MusWikiDB, a music-specialized vector database for the retrieval stage, and (2) utilizes context information during both inference and fine-tuning processes to effectively transform general-purpose LLMs into music-specific models. Our experiment demonstrates that MusT-RAG significantly outperforms traditional fine-tuning approaches in enhancing LLMs' music domain adaptation capabilities, showing consistent improvements across both in-domain and out-of-domain MQA benchmarks. Additionally, our MusWikiDB proves substantially more effective than general Wikipedia corpora, delivering superior performance and computational efficiency. 

---
# Reading Between the Timelines: RAG for Answering Diachronic Questions 

**Authors**: Kwun Hang Lau, Ruiyuan Zhang, Weijie Shi, Xiaofang Zhou, Xiaojun Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22917)  

**Abstract**: While Retrieval-Augmented Generation (RAG) excels at injecting static, factual knowledge into Large Language Models (LLMs), it exhibits a critical deficit in handling longitudinal queries that require tracking entities and phenomena across time. This blind spot arises because conventional, semantically-driven retrieval methods are not equipped to gather evidence that is both topically relevant and temporally coherent for a specified duration. We address this challenge by proposing a new framework that fundamentally redesigns the RAG pipeline to infuse temporal logic. Our methodology begins by disentangling a user's query into its core subject and its temporal window. It then employs a specialized retriever that calibrates semantic matching against temporal relevance, ensuring the collection of a contiguous evidence set that spans the entire queried period. To enable rigorous evaluation of this capability, we also introduce the Analytical Diachronic Question Answering Benchmark (ADQAB), a challenging evaluation suite grounded in a hybrid corpus of real and synthetic financial news. Empirical results on ADQAB show that our approach yields substantial gains in answer accuracy, surpassing standard RAG implementations by 13% to 27%. This work provides a validated pathway toward RAG systems capable of performing the nuanced, evolutionary analysis required for complex, real-world questions. The dataset and code for this study are publicly available at this https URL. 

---
# Not Just What, But When: Integrating Irregular Intervals to LLM for Sequential Recommendation 

**Authors**: Wei-Wei Du, Takuma Udagawa, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2507.23209)  

**Abstract**: Time intervals between purchasing items are a crucial factor in sequential recommendation tasks, whereas existing approaches focus on item sequences and often overlook by assuming the intervals between items are static. However, dynamic intervals serve as a dimension that describes user profiling on not only the history within a user but also different users with the same item history. In this work, we propose IntervalLLM, a novel framework that integrates interval information into LLM and incorporates the novel interval-infused attention to jointly consider information of items and intervals. Furthermore, unlike prior studies that address the cold-start scenario only from the perspectives of users and items, we introduce a new viewpoint: the interval perspective to serve as an additional metric for evaluating recommendation methods on the warm and cold scenarios. Extensive experiments on 3 benchmarks with both traditional- and LLM-based baselines demonstrate that our IntervalLLM achieves not only 4.4% improvements in average but also the best-performing warm and cold scenarios across all users, items, and the proposed interval perspectives. In addition, we observe that the cold scenario from the interval perspective experiences the most significant performance drop among all recommendation methods. This finding underscores the necessity of further research on interval-based cold challenges and our integration of interval information in the realm of sequential recommendation tasks. Our code is available here: this https URL. 

---
# Causal2Vec: Improving Decoder-only LLMs as Versatile Embedding Models 

**Authors**: Ailiang Lin, Zhuoyun Li, Kotaro Funakoshi  

**Link**: [PDF](https://arxiv.org/pdf/2507.23386)  

**Abstract**: Decoder-only large language models (LLMs) are increasingly used to build embedding models that effectively encode the semantic information of natural language texts into dense vector representations for various embedding tasks. However, many existing methods primarily focus on removing the causal attention mask in LLMs to enable bidirectional attention, potentially undermining the model's ability to extract semantic information acquired during pretraining. Additionally, leading unidirectional approaches often rely on extra input text to overcome the inherent limitations of causal attention, inevitably increasing computational costs. In this work, we propose Causal2Vec, a general-purpose embedding model tailored to enhance the performance of decoder-only LLMs without altering their original architectures or introducing significant computational overhead. Specifically, we first employ a lightweight BERT-style model to pre-encode the input text into a single Contextual token, which is then prepended to the LLM's input sequence, allowing each token to capture contextualized information even without attending to future tokens. Furthermore, to mitigate the recency bias introduced by last-token pooling and help LLMs better leverage the semantic information encoded in the Contextual token, we concatenate the last hidden states of Contextual and EOS tokens as the final text embedding. In practice, Causal2Vec achieves state-of-the-art performance on the Massive Text Embeddings Benchmark (MTEB) among models trained solely on publicly available retrieval datasets, while reducing the required sequence length by up to 85% and inference time by up to 82% compared to best-performing methods. 

---
# Med-R$^3$: Enhancing Medical Retrieval-Augmented Reasoning of LLMs via Progressive Reinforcement Learning 

**Authors**: Keer Lu, Zheng Liang, Youquan Li, Jiejun Tan, Da Pan, Shusen Zhang, Guosheng Dong, Huang Leng  

**Link**: [PDF](https://arxiv.org/pdf/2507.23541)  

**Abstract**: In medical scenarios, effectively retrieving external knowledge and leveraging it for rigorous logical reasoning is of significant importance. Despite their potential, existing work has predominantly focused on enhancing either retrieval or reasoning capabilities of the models in isolation, with little attention given to their joint optimization, which leads to limited coordination between the two processes. Additionally, current methods rely heavily on supervised fine-tuning (SFT), which can cause models to memorize existing problem-solving pathways, thereby restricting their generalization ability when confronted with novel problem contexts. Furthermore, while some studies have explored to improve retrieval-augmented reasoning in general domains via reinforcement learning, their reward function designs do not adequately capture the specific demands of the medical domain. To address these challenges, we introduce **Med-R$^3$**, a **Med**ical **R**etrieval-augmented **R**easoning framework driven by progressive **R**einforcement learning. In this framework, we first develop the model's ability to perform logical reasoning over medical problems. Subsequently, on the basis of this foundation, we adaptively optimize the retrieval capability to better align with the characteristics of knowledge corpus and external information utilization throughout the reasoning process. Finally, we conduct joint optimization of the model's retrieval and reasoning coordination. Extensive experiments indicate that **Med-R$^3$** could achieve state-of-the-art performances, with LLaMA3.1-8B-Instruct + Med-R$^3$ surpassing closed-sourced GPT-4o-mini by 3.93\% at a comparable parameter scale, while Qwen2.5-14B augmented with Med-R$^3$ shows a more substantial gain of 13.53\%. 

---
# Rule2Text: Natural Language Explanation of Logical Rules in Knowledge Graphs 

**Authors**: Nasim Shirvani-Mahdavi, Devin Wingfield, Amin Ghasemi, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.23740)  

**Abstract**: Knowledge graphs (KGs) often contain sufficient information to support the inference of new facts. Identifying logical rules not only improves the completeness of a knowledge graph but also enables the detection of potential errors, reveals subtle data patterns, and enhances the overall capacity for reasoning and interpretation. However, the complexity of such rules, combined with the unique labeling conventions of each KG, can make them difficult for humans to understand. In this paper, we explore the potential of large language models to generate natural language explanations for logical rules. Specifically, we extract logical rules using the AMIE 3.5.1 rule discovery algorithm from the benchmark dataset FB15k-237 and two large-scale datasets, FB-CVT-REV and FB+CVT-REV. We examine various prompting strategies, including zero- and few-shot prompting, including variable entity types, and chain-of-thought reasoning. We conduct a comprehensive human evaluation of the generated explanations based on correctness, clarity, and hallucination, and also assess the use of large language models as automatic judges. Our results demonstrate promising performance in terms of explanation correctness and clarity, although several challenges remain for future research. All scripts and data used in this study are publicly available at this https URL}{this https URL. 

---
# P-ReMIS: Pragmatic Reasoning in Mental Health and a Social Implication 

**Authors**: Sneha Oram, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.23247)  

**Abstract**: There has been an increase in recent advancements in the explainability and development of personalized chatbots for mental health. However, the reasoning aspects for explainability and dialogue discourse have not been explored previously for mental health. Hence, we are investigating the pragmatic reasoning capability of large language models (LLMs) in this domain. We introduce P-ReMe dataset, and propose a modified definition for the pragmatic phenomena of implicature (implied meaning) and presupposition (implicit assumption) in mental health. Following the definition, we formulate two tasks in implicature and one task in presupposition. To benchmark the dataset and the presented tasks, we consider four models - Llama3.1, Mistral, MentaLLaMa, and Qwen. The results of the experiments suggest that Mistral and Qwen show substantial reasoning capabilities in the domain. In addition, we also propose StiPRompts to study the stigma around mental health with the state-of-the-art LLMs, GPT-4o mini, Deepseek-chat, and Claude-3.5-haiku. Our evaluated findings show that Claude-3.5-haiku deals with the stigma more responsibly compared to the other two LLMs. 

---
# Enabling Few-Shot Alzheimer's Disease Diagnosis on Tabular Biomarker Data with LLMs 

**Authors**: Sophie Kearney, Shu Yang, Zixuan Wen, Bojian Hou, Duy Duong-Tran, Tianlong Chen, Jason Moore, Marylyn Ritchie, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.23227)  

**Abstract**: Early and accurate diagnosis of Alzheimer's disease (AD), a complex neurodegenerative disorder, requires analysis of heterogeneous biomarkers (e.g., neuroimaging, genetic risk factors, cognitive tests, and cerebrospinal fluid proteins) typically represented in a tabular format. With flexible few-shot reasoning, multimodal integration, and natural-language-based interpretability, large language models (LLMs) offer unprecedented opportunities for prediction with structured biomedical data. We propose a novel framework called TAP-GPT, Tabular Alzheimer's Prediction GPT, that adapts TableGPT2, a multimodal tabular-specialized LLM originally developed for business intelligence tasks, for AD diagnosis using structured biomarker data with small sample sizes. Our approach constructs few-shot tabular prompts using in-context learning examples from structured biomedical data and finetunes TableGPT2 using the parameter-efficient qLoRA adaption for a clinical binary classification task of AD or cognitively normal (CN). The TAP-GPT framework harnesses the powerful tabular understanding ability of TableGPT2 and the encoded prior knowledge of LLMs to outperform more advanced general-purpose LLMs and a tabular foundation model (TFM) developed for prediction tasks. To our knowledge, this is the first application of LLMs to the prediction task using tabular biomarker data, paving the way for future LLM-driven multi-agent frameworks in biomedical informatics. 

---
# Evaluating LLMs' Multilingual Capabilities for Bengali: Benchmark Creation and Performance Analysis 

**Authors**: Shimanto Bhowmik, Tawsif Tashwar Dipto, Md Sazzad Islam, Sheryl Hsu, Tahsin Reasat  

**Link**: [PDF](https://arxiv.org/pdf/2507.23248)  

**Abstract**: Bengali is an underrepresented language in NLP research. However, it remains a challenge due to its unique linguistic structure and computational constraints. In this work, we systematically investigate the challenges that hinder Bengali NLP performance by focusing on the absence of standardized evaluation benchmarks. We then evaluated 10 recent open source Large Language Models (LLMs) in 8 of the translated datasets and performed a comprehensive error analysis to pinpoint their primary failure modes. Our findings reveal consistent performance gaps for Bengali compared to English, particularly for smaller models and specific model families like Mistral. We also identified promising robustness in certain architectures, such as DeepSeek, that maintain more stable performance across languages. Our analysis reveals an inverse relationship between tokenization efficiency and LLM accuracy where models tend to perform worse when inputs are excessively tokenized, whereas more efficient \& concise tokenization results in improved performance. These findings highlight critical areas where current models fall short and underscore the need for improved dataset quality and evaluation methodologies tailored to multilingual contexts. This work will catalyze further research on NLP for underrepresented languages, helping to democratize access to advanced language technologies worldwide. The code and dataset used in this research is publicly available at this https URL. 

---
# LENS: Learning Ensemble Confidence from Neural States for Multi-LLM Answer Integration 

**Authors**: Jizhou Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.23167)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, with different models excelling in distinct domains and specific abilities. Effectively combining the predictions of multiple LLMs is crucial for enhancing system robustness and performance. However, existing ensemble methods often rely on simple techniques like voting or logits ensembling, which overlook the varying confidence and reliability of models in different contexts. In this work, we propose LENS (Learning ENsemble confidence from Neural States), a novel approach that learns to estimate model confidence by analyzing internal representations. For each LLM, we train a lightweight linear confidence predictor that leverages layer-wise hidden states and normalized probabilities as inputs. This allows for more nuanced weighting of model predictions based on their context-dependent reliability. Our method does not require modifying the model parameters and requires negligible additional computation. Experimental results on multiple-choice and boolean question-answering tasks demonstrate that LENS outperforms traditional ensemble methods by a substantial margin. Our findings suggest that internal representations provide valuable signals for determining model confidence and can be effectively leveraged for ensemble learning. 

---
# Model Directions, Not Words: Mechanistic Topic Models Using Sparse Autoencoders 

**Authors**: Carolina Zheng, Nicolas Beltran-Velez, Sweta Karlekar, Claudia Shi, Achille Nazaret, Asif Mallik, Amir Feder, David M. Blei  

**Link**: [PDF](https://arxiv.org/pdf/2507.23220)  

**Abstract**: Traditional topic models are effective at uncovering latent themes in large text collections. However, due to their reliance on bag-of-words representations, they struggle to capture semantically abstract features. While some neural variants use richer representations, they are similarly constrained by expressing topics as word lists, which limits their ability to articulate complex topics. We introduce Mechanistic Topic Models (MTMs), a class of topic models that operate on interpretable features learned by sparse autoencoders (SAEs). By defining topics over this semantically rich space, MTMs can reveal deeper conceptual themes with expressive feature descriptions. Moreover, uniquely among topic models, MTMs enable controllable text generation using topic-based steering vectors. To properly evaluate MTM topics against word-list-based approaches, we propose \textit{topic judge}, an LLM-based pairwise comparison evaluation framework. Across five datasets, MTMs match or exceed traditional and neural baselines on coherence metrics, are consistently preferred by topic judge, and enable effective steering of LLM outputs. 

---
# Uncovering the Fragility of Trustworthy LLMs through Chinese Textual Ambiguity 

**Authors**: Xinwei Wu, Haojie Li, Hongyu Liu, Xinyu Ji, Ruohan Li, Yule Chen, Yigeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23121)  

**Abstract**: In this work, we study a critical research problem regarding the trustworthiness of large language models (LLMs): how LLMs behave when encountering ambiguous narrative text, with a particular focus on Chinese textual ambiguity. We created a benchmark dataset by collecting and generating ambiguous sentences with context and their corresponding disambiguated pairs, representing multiple possible interpretations. These annotated examples are systematically categorized into 3 main categories and 9 subcategories. Through experiments, we discovered significant fragility in LLMs when handling ambiguity, revealing behavior that differs substantially from humans. Specifically, LLMs cannot reliably distinguish ambiguous text from unambiguous text, show overconfidence in interpreting ambiguous text as having a single meaning rather than multiple meanings, and exhibit overthinking when attempting to understand the various possible meanings. Our findings highlight a fundamental limitation in current LLMs that has significant implications for their deployment in real-world applications where linguistic ambiguity is common, calling for improved approaches to handle uncertainty in language understanding. The dataset and code are publicly available at this GitHub repository: this https URL. 

---
# A Novel Evaluation Benchmark for Medical LLMs: Illuminating Safety and Effectiveness in Clinical Domains 

**Authors**: Shirui Wang, Zhihui Tang, Huaxia Yang, Qiuhong Gong, Tiantian Gu, Hongyang Ma, Yongxin Wang, Wubin Sun, Zeliang Lian, Kehang Mao, Yinan Jiang, Zhicheng Huang, Lingyun Ma, Wenjie Shen, Yajie Ji, Yunhui Tan, Chunbo Wang, Yunlu Gao, Qianling Ye, Rui Lin, Mingyu Chen, Lijuan Niu, Zhihao Wang, Peng Yu, Mengran Lang, Yue Liu, Huimin Zhang, Haitao Shen, Long Chen, Qiguang Zhao, Si-Xuan Liu, Lina Zhou, Hua Gao, Dongqiang Ye, Lingmin Meng, Youtao Yu, Naixin Liang, Jianxiong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23486)  

**Abstract**: Large language models (LLMs) hold promise in clinical decision support but face major challenges in safety evaluation and effectiveness validation. We developed the Clinical Safety-Effectiveness Dual-Track Benchmark (CSEDB), a multidimensional framework built on clinical expert consensus, encompassing 30 criteria covering critical areas like critical illness recognition, guideline adherence, and medication safety, with weighted consequence measures. Thirty-two specialist physicians developed and reviewed 2,069 open-ended Q&A items aligned with these criteria, spanning 26 clinical departments to simulate real-world scenarios. Benchmark testing of six LLMs revealed moderate overall performance (average total score 57.2%, safety 54.7%, effectiveness 62.3%), with a significant 13.3% performance drop in high-risk scenarios (p < 0.0001). Domain-specific medical LLMs showed consistent performance advantages over general-purpose models, with relatively higher top scores in safety (0.912) and effectiveness (0.861). The findings of this study not only provide a standardized metric for evaluating the clinical application of medical LLMs, facilitating comparative analyses, risk exposure identification, and improvement directions across different scenarios, but also hold the potential to promote safer and more effective deployment of large language models in healthcare environments. 

---
# What's Taboo for You? - An Empirical Evaluation of LLMs Behavior Toward Sensitive Content 

**Authors**: Alfio Ferrara, Sergio Picascia, Laura Pinnavaia, Vojimir Ranitovic, Elisabetta Rocchetti, Alice Tuveri  

**Link**: [PDF](https://arxiv.org/pdf/2507.23319)  

**Abstract**: Proprietary Large Language Models (LLMs) have shown tendencies toward politeness, formality, and implicit content moderation. While previous research has primarily focused on explicitly training models to moderate and detoxify sensitive content, there has been limited exploration of whether LLMs implicitly sanitize language without explicit instructions. This study empirically analyzes the implicit moderation behavior of GPT-4o-mini when paraphrasing sensitive content and evaluates the extent of sensitivity shifts. Our experiments indicate that GPT-4o-mini systematically moderates content toward less sensitive classes, with substantial reductions in derogatory and taboo language. Also, we evaluate the zero-shot capabilities of LLMs in classifying sentence sensitivity, comparing their performances against traditional methods. 

---
# Math Natural Language Inference: this should be easy! 

**Authors**: Valeria de Paiva, Qiyue Gao, Hai Hu, Pavel Kovalev, Yikang Liu, Lawrence S. Moss, Zhiheng Qian  

**Link**: [PDF](https://arxiv.org/pdf/2507.23063)  

**Abstract**: We ask whether contemporary LLMs are able to perform natural language inference (NLI) tasks on mathematical texts. We call this the Math NLI problem. We construct a corpus of Math NLI pairs whose premises are from extant mathematical text and whose hypotheses and gold labels were provided by people with experience in both research-level mathematics and also in the NLI field. We also investigate the quality of corpora using the same premises but whose hypotheses are provided by LLMs themselves. We not only investigate the performance but also the inter-group consistency of the diverse group of LLMs. We have both positive and negative findings. Among our positive findings: in some settings, using a majority vote of LLMs is approximately equivalent to using human-labeled data in the Math NLI area. On the negative side: LLMs still struggle with mathematical language. They occasionally fail at even basic inferences. Current models are not as prone to hypothesis-only "inference" in our data the way the previous generation had been. In addition to our findings, we also provide our corpora as data to support future work on Math NLI. 

---
# Geak: Introducing Triton Kernel AI Agent & Evaluation Benchmarks 

**Authors**: Jianghui Wang, Vinay Joshi, Saptarshi Majumder, Xu Chao, Bin Ding, Ziqiong Liu, Pratik Prabhanjan Brahma, Dong Li, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2507.23194)  

**Abstract**: The demand for AI-generated GPU kernels is rapidly growing, influenced by the need for scalable, hardware-optimized solutions in both industry and academia. As deep learning workloads grow in complexity and diversity, it is imperative to automate low-level kernel development to meet performance and productivity demands. Major cloud providers, semiconductor companies, and research institutions are now investing heavily in AI-driven code generation for GPUs, aiming to reduce manual optimization efforts while achieving near-expert performance on hardware like AMD MI300X. The Triton language, a Python-based DSL for GPU programming, has emerged as a popular target for such AI-generated kernels due to its balance of performance and ease-of-coding. In this work, we present an evaluation suite for Triton-based GPU kernels and GEAK (Generating Efficient AI-centric GPU Kernels)-a framework that leverages cutting-edge LLMs to generate performant Triton code specifically for AMD GPUs, including the AMD MI300X and MI250. GEAK leverages inference-time compute scaling to produce Triton-based GPU kernels using a reasoning loop adapted from Reflexion-style feedback mechanisms. On two evaluation benchmarks, GEAK significantly outperformed the baselines of directly prompting frontier LLMs as well as Reflexion-based generation pipelines by achieving correctness up to $63$% and execution speed up of up to $2.59$X. These results highlight the promise of GEAK-like agentic code generation for accelerating the adoption of diverse hardware platforms and democratizing access to expert-level kernel performance. 

---
# Trustworthy Reasoning: Evaluating and Enhancing Factual Accuracy in LLM Intermediate Thought Processes 

**Authors**: Rui Jiao, Yue Zhang, Jinku Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.22940)  

**Abstract**: We present RELIANCE (Reasoning Evaluation with Logical Integrity and Accuracy for Confidence Enhancement), a novel framework addressing a critical vulnerability in Large Language Models (LLMs): the prevalence of factual inaccuracies within intermediate reasoning steps despite correct final answers. This phenomenon poses substantial risks in high-stakes domains including healthcare, legal analysis, and scientific research, where erroneous yet confidently presented reasoning can mislead users into dangerous decisions. Our framework integrates three core components: (1) a specialized fact-checking classifier trained on counterfactually augmented data to detect subtle factual inconsistencies within reasoning chains; (2) a Group Relative Policy Optimization (GRPO) reinforcement learning approach that balances factuality, coherence, and structural correctness through multi-dimensional rewards; and (3) a mechanistic interpretability module examining how factuality improvements manifest in model activations during reasoning processes. Extensive evaluation across ten state-of-the-art models reveals concerning patterns: even leading models like Claude-3.7 and GPT-o1 demonstrate reasoning factual accuracy of only 81.93% and 82.57% respectively. RELIANCE significantly enhances factual robustness (up to 49.90% improvement) while maintaining or improving performance on challenging benchmarks including Math-500, AIME-2024, and GPQA. Furthermore, our activation-level analysis provides actionable insights into how factual enhancements reshape reasoning trajectories within model architectures, establishing foundations for future training methodologies that explicitly target factual robustness through activation-guided optimization. 

---
# Role-Aware Language Models for Secure and Contextualized Access Control in Organizations 

**Authors**: Saeed Almheiri, Yerulan Kongrat, Adrian Santosh, Ruslan Tasmukhanov, Josemaria Vera, Muhammad Dehan Al Kautsar, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2507.23465)  

**Abstract**: As large language models (LLMs) are increasingly deployed in enterprise settings, controlling model behavior based on user roles becomes an essential requirement. Existing safety methods typically assume uniform access and focus on preventing harmful or toxic outputs, without addressing role-specific access constraints. In this work, we investigate whether LLMs can be fine-tuned to generate responses that reflect the access privileges associated with different organizational roles. We explore three modeling strategies: a BERT-based classifier, an LLM-based classifier, and role-conditioned generation. To evaluate these approaches, we construct two complementary datasets. The first is adapted from existing instruction-tuning corpora through clustering and role labeling, while the second is synthetically generated to reflect realistic, role-sensitive enterprise scenarios. We assess model performance across varying organizational structures and analyze robustness to prompt injection, role mismatch, and jailbreak attempts. 

---
# Evaluating Large Language Models (LLMs) in Financial NLP: A Comparative Study on Financial Report Analysis 

**Authors**: Md Talha Mohsin  

**Link**: [PDF](https://arxiv.org/pdf/2507.22936)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide variety of Financial Natural Language Processing (FinNLP) tasks. However, systematic comparisons among widely used LLMs remain underexplored. Given the rapid advancement and growing influence of LLMs in financial analysis, this study conducts a thorough comparative evaluation of five leading LLMs, GPT, Claude, Perplexity, Gemini and DeepSeek, using 10-K filings from the 'Magnificent Seven' technology companies. We create a set of domain-specific prompts and then use three methodologies to evaluate model performance: human annotation, automated lexical-semantic metrics (ROUGE, Cosine Similarity, Jaccard), and model behavior diagnostics (prompt-level variance and across-model similarity). The results show that GPT gives the most coherent, semantically aligned, and contextually relevant answers; followed by Claude and Perplexity. Gemini and DeepSeek, on the other hand, have more variability and less agreement. Also, the similarity and stability of outputs change from company to company and over time, showing that they are sensitive to how prompts are written and what source material is used. 

---
# User Feedback in Human-LLM Dialogues: A Lens to Understand Users But Noisy as a Learning Signal 

**Authors**: Yuhan Liu, Michael J.Q. Zhang, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2507.23158)  

**Abstract**: Once language models (LMs) are deployed, they can interact with users long-term, ideally evolving continuously based on their feedback. Asking for direct user feedback can be disruptive; thus, we study harvesting user feedback from user-LM interaction logs. We study implicit user feedback in two user-LM interaction datasets (WildChat and LMSYS). First, we analyze user feedback in the user-LLM conversation trajectory, providing insights into when and why such feedback occurs. Second, we study harvesting learning signals from such implicit user feedback. We find that the contents of user feedback (e.g., user wanted clarification), not just the polarity (e.g., users were unhappy with the previous model response), can improve model performance in short human-designed questions (MTBench) but not on longer and more complex questions (WildBench). We also find that the usefulness of user feedback is largely tied to the quality of the user's initial prompt. Together, we provide an in-depth study of implicit user feedback, showing its potential and limitations. 

---
# Trusted Knowledge Extraction for Operations and Maintenance Intelligence 

**Authors**: Kathleen Mealey, Jonathan A. Karr Jr., Priscila Saboia Moreira, Paul R. Brenner, Charles F. Vardeman II  

**Link**: [PDF](https://arxiv.org/pdf/2507.22935)  

**Abstract**: Deriving operational intelligence from organizational data repositories is a key challenge due to the dichotomy of data confidentiality vs data integration objectives, as well as the limitations of Natural Language Processing (NLP) tools relative to the specific knowledge structure of domains such as operations and maintenance. In this work, we discuss Knowledge Graph construction and break down the Knowledge Extraction process into its Named Entity Recognition, Coreference Resolution, Named Entity Linking, and Relation Extraction functional components. We then evaluate sixteen NLP tools in concert with or in comparison to the rapidly advancing capabilities of Large Language Models (LLMs). We focus on the operational and maintenance intelligence use case for trusted applications in the aircraft industry. A baseline dataset is derived from a rich public domain US Federal Aviation Administration dataset focused on equipment failures or maintenance requirements. We assess the zero-shot performance of NLP and LLM tools that can be operated within a controlled, confidential environment (no data is sent to third parties). Based on our observation of significant performance limitations, we discuss the challenges related to trusted NLP and LLM tools as well as their Technical Readiness Level for wider use in mission-critical industries such as aviation. We conclude with recommendations to enhance trust and provide our open-source curated dataset to support further baseline testing and evaluation. 

---
# FinMarBa: A Market-Informed Dataset for Financial Sentiment Classification 

**Authors**: Baptiste Lefort, Eric Benhamou, Beatrice Guez, Jean-Jacques Ohana, Ethan Setrouk, Alban Etienne  

**Link**: [PDF](https://arxiv.org/pdf/2507.22932)  

**Abstract**: This paper presents a novel hierarchical framework for portfolio optimization, integrating lightweight Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) to combine sentiment signals from financial news with traditional market indicators. Our three-tier architecture employs base RL agents to process hybrid data, meta-agents to aggregate their decisions, and a super-agent to merge decisions based on market data and sentiment analysis. Evaluated on data from 2018 to 2024, after training on 2000-2017, the framework achieves a 26% annualized return and a Sharpe ratio of 1.2, outperforming equal-weighted and S&P 500 benchmarks. Key contributions include scalable cross-modal integration, a hierarchical RL structure for enhanced stability, and open-source reproducibility. 

---
# Protecting Vulnerable Voices: Synthetic Dataset Generation for Self-Disclosure Detection 

**Authors**: Shalini Jangra, Suparna De, Nishanth Sastry, Saeed Fadaei  

**Link**: [PDF](https://arxiv.org/pdf/2507.22930)  

**Abstract**: Social platforms such as Reddit have a network of communities of shared interests, with a prevalence of posts and comments from which one can infer users' Personal Information Identifiers (PIIs). While such self-disclosures can lead to rewarding social interactions, they pose privacy risks and the threat of online harms. Research into the identification and retrieval of such risky self-disclosures of PIIs is hampered by the lack of open-source labeled datasets. To foster reproducible research into PII-revealing text detection, we develop a novel methodology to create synthetic equivalents of PII-revealing data that can be safely shared. Our contributions include creating a taxonomy of 19 PII-revealing categories for vulnerable populations and the creation and release of a synthetic PII-labeled multi-text span dataset generated from 3 text generation Large Language Models (LLMs), Llama2-7B, Llama3-8B, and zephyr-7b-beta, with sequential instruction prompting to resemble the original Reddit posts. The utility of our methodology to generate this synthetic dataset is evaluated with three metrics: First, we require reproducibility equivalence, i.e., results from training a model on the synthetic data should be comparable to those obtained by training the same models on the original posts. Second, we require that the synthetic data be unlinkable to the original users, through common mechanisms such as Google Search. Third, we wish to ensure that the synthetic data be indistinguishable from the original, i.e., trained humans should not be able to tell them apart. We release our dataset and code at this https URL to foster reproducible research into PII privacy risks in online social media. 

---
# A Graph-based Approach for Multi-Modal Question Answering from Flowcharts in Telecom Documents 

**Authors**: Sumit Soman, H. G. Ranjani, Sujoy Roychowdhury, Venkata Dharma Surya Narayana Sastry, Akshat Jain, Pranav Gangrade, Ayaaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22938)  

**Abstract**: Question-Answering (QA) from technical documents often involves questions whose answers are present in figures, such as flowcharts or flow diagrams. Text-based Retrieval Augmented Generation (RAG) systems may fail to answer such questions. We leverage graph representations of flowcharts obtained from Visual large Language Models (VLMs) and incorporate them in a text-based RAG system to show that this approach can enable image retrieval for QA in the telecom domain. We present the end-to-end approach from processing technical documents, classifying image types, building graph representations, and incorporating them with the text embedding pipeline for efficient retrieval. We benchmark the same on a QA dataset created based on proprietary telecom product information documents. Results show that the graph representations obtained using a fine-tuned VLM model have lower edit distance with respect to the ground truth, which illustrate the robustness of these representations for flowchart images. Further, the approach for QA using these representations gives good retrieval performance using text-based embedding models, including a telecom-domain adapted one. Our approach also alleviates the need for a VLM in inference, which is an important cost benefit for deployed QA systems. 

---
# Hierarchical Memory for High-Efficiency Long-Term Reasoning in LLM Agents 

**Authors**: Haoran Sun, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.22925)  

**Abstract**: Long-term memory is one of the key factors influencing the reasoning capabilities of Large Language Model Agents (LLM Agents). Incorporating a memory mechanism that effectively integrates past interactions can significantly enhance decision-making and contextual coherence of LLM Agents. While recent works have made progress in memory storage and retrieval, such as encoding memory into dense vectors for similarity-based search or organizing knowledge in the form of graph, these approaches often fall short in structured memory organization and efficient retrieval. To address these limitations, we propose a Hierarchical Memory (H-MEM) architecture for LLM Agents that organizes and updates memory in a multi-level fashion based on the degree of semantic abstraction. Each memory vector is embedded with a positional index encoding pointing to its semantically related sub-memories in the next layer. During the reasoning phase, an index-based routing mechanism enables efficient, layer-by-layer retrieval without performing exhaustive similarity computations. We evaluate our method on five task settings from the LoCoMo dataset. Experimental results show that our approach consistently outperforms five baseline methods, demonstrating its effectiveness in long-term dialogue scenarios. 

---
# Semantic Convergence: Investigating Shared Representations Across Scaled LLMs 

**Authors**: Daniel Son, Sanjana Rathore, Andrew Rufail, Adrian Simon, Daniel Zhang, Soham Dave, Cole Blondin, Kevin Zhu, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2507.22918)  

**Abstract**: We investigate feature universality in Gemma-2 language models (Gemma-2-2B and Gemma-2-9B), asking whether models with a four-fold difference in scale still converge on comparable internal concepts. Using the Sparse Autoencoder (SAE) dictionary-learning pipeline, we utilize SAEs on each model's residual-stream activations, align the resulting monosemantic features via activation correlation, and compare the matched feature spaces with SVCCA and RSA. Middle layers yield the strongest overlap, while early and late layers show far less similarity. Preliminary experiments extend the analysis from single tokens to multi-token subspaces, showing that semantically similar subspaces interact similarly with language models. These results strengthen the case that large language models carve the world into broadly similar, interpretable features despite size differences, reinforcing universality as a foundation for cross-model interpretability. 

---
# ElectriQ: A Benchmark for Assessing the Response Capability of Large Language Models in Power Marketing 

**Authors**: Jinzhi Wang, Qingke Peng, Haozhou Li, Zeyuan Zeng, Qinfeng Song, Kaixuan Yang, Jiangbo Zhang, Yaoying Wang, Ruimeng Li, Biyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.22911)  

**Abstract**: Electric power marketing customer service plays a critical role in addressing inquiries, complaints, and service requests. However, current systems, such as China's 95598 hotline, often struggle with slow response times, inflexible procedures, and limited accuracy in domain-specific tasks. While large language models (LLMs) like GPT-4o and Claude 3 demonstrate strong general capabilities, they lack the domain expertise and empathy required in this field. To bridge this gap, we introduce ElectriQ, the first benchmark designed to evaluate and enhance LLMs in electric power marketing scenarios. ElectriQ consists of a dialogue dataset covering six key service categories and introduces four evaluation metrics: professionalism, popularity, readability, and user-friendliness. We further incorporate a domain-specific knowledge base and propose a knowledge augmentation method to boost model performance. Experiments on 13 LLMs reveal that smaller models such as LLama3-8B, when fine-tuned and augmented, can surpass GPT-4o in terms of professionalism and user-friendliness. ElectriQ establishes a comprehensive foundation for developing LLMs tailored to the needs of power marketing services. 

---
# SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning Architecture with LLM-Based World Model 

**Authors**: Mingkai Deng, Jinyu Hou, Yilin Shen, Hongxia Jin, Graham Neubig, Zhiting Hu, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.23773)  

**Abstract**: AI agents built on large language models (LLMs) hold enormous promise, but current practice focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also suffers from the fundamental limitations of autoregressive LLMs. On the other hand, humans are general agents who reason by mentally simulating the outcomes of their actions and plans. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of optimal agent in any environment, \modelname overcomes the limitations of autoregressive reasoning by introducing a world model for planning via simulation. The generalized world model is implemented using LLM, which can flexibly plan in a wide range of environments using the concept-rich latent space of natural language. Experiments on difficult web browsing tasks show that \modelname improves the success of flight search from 0\% to 32.2\%. World-model-based planning, in particular, shows consistent advantage of up to 124\% over autoregressive planning, demonstrating the advantage of world model simulation as a reasoning paradigm. We are excited about the possibility for training a single, general agent model based on LLMs that can act superintelligently in all environments. To start, we make SimuRA, a web-browsing agent built on \modelname with pretrained LLMs, available as a research demo for public testing. 

---
# How does Chain of Thought Think? Mechanistic Interpretability of Chain-of-Thought Reasoning with Sparse Autoencoding 

**Authors**: Xi Chen, Aske Plaat, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2507.22928)  

**Abstract**: Chain-of-thought (CoT) prompting boosts Large Language Models accuracy on multi-step tasks, yet whether the generated "thoughts" reflect the true internal reasoning process is unresolved. We present the first feature-level causal study of CoT faithfulness. Combining sparse autoencoders with activation patching, we extract monosemantic features from Pythia-70M and Pythia-2.8B while they tackle GSM8K math problems under CoT and plain (noCoT) prompting. Swapping a small set of CoT-reasoning features into a noCoT run raises answer log-probabilities significantly in the 2.8B model, but has no reliable effect in 70M, revealing a clear scale threshold. CoT also leads to significantly higher activation sparsity and feature interpretability scores in the larger model, signalling more modular internal computation. For example, the model's confidence in generating correct answers improves from 1.2 to 4.3. We introduce patch-curves and random-feature patching baselines, showing that useful CoT information is not only present in the top-K patches but widely distributed. Overall, our results indicate that CoT can induce more interpretable internal structures in high-capacity LLMs, validating its role as a structured prompting method. 

---
# Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving 

**Authors**: Luoxin Chen, Jinming Gu, Liankai Huang, Wenhao Huang, Zhicheng Jiang, Allan Jie, Xiaoran Jin, Xing Jin, Chenggang Li, Kaijing Ma, Cheng Ren, Jiawei Shen, Wenlei Shi, Tong Sun, He Sun, Jiahui Wang, Siran Wang, Zhihong Wang, Chenrui Wei, Shufa Wei, Yonghui Wu, Yuchen Wu, Yihang Xia, Huajian Xin, Fan Yang, Huaiyuan Ying, Hongyi Yuan, Zheng Yuan, Tianyang Zhan, Chi Zhang, Yue Zhang, Ge Zhang, Tianyun Zhao, Jianqiu Zhao, Yichi Zhou, Thomas Hanwen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23726)  

**Abstract**: LLMs have demonstrated strong mathematical reasoning abilities by leveraging reinforcement learning with long chain-of-thought, yet they continue to struggle with theorem proving due to the lack of clear supervision signals when solely using natural language. Dedicated domain-specific languages like Lean provide clear supervision via formal verification of proofs, enabling effective training through reinforcement learning. In this work, we propose \textbf{Seed-Prover}, a lemma-style whole-proof reasoning model. Seed-Prover can iteratively refine its proof based on Lean feedback, proved lemmas, and self-summarization. To solve IMO-level contest problems, we design three test-time inference strategies that enable both deep and broad reasoning. Seed-Prover proves $78.1\%$ of formalized past IMO problems, saturates MiniF2F, and achieves over 50\% on PutnamBench, outperforming the previous state-of-the-art by a large margin. To address the lack of geometry support in Lean, we introduce a geometry reasoning engine \textbf{Seed-Geometry}, which outperforms previous formal geometry engines. We use these two systems to participate in IMO 2025 and fully prove 5 out of 6 problems. This work represents a significant advancement in automated mathematical reasoning, demonstrating the effectiveness of formal verification with long chain-of-thought reasoning. 

---
# CoT-Self-Instruct: Building high-quality synthetic prompts for reasoning and non-reasoning tasks 

**Authors**: Ping Yu, Jack Lanchantin, Tianlu Wang, Weizhe Yuan, Olga Golovneva, Ilia Kulikov, Sainbayar Sukhbaatar, Jason Weston, Jing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23751)  

**Abstract**: We propose CoT-Self-Instruct, a synthetic data generation method that instructs LLMs to first reason and plan via Chain-of-Thought (CoT) based on the given seed tasks, and then to generate a new synthetic prompt of similar quality and complexity for use in LLM training, followed by filtering for high-quality data with automatic metrics. In verifiable reasoning, our synthetic data significantly outperforms existing training datasets, such as s1k and OpenMathReasoning, across MATH500, AMC23, AIME24 and GPQA-Diamond. For non-verifiable instruction-following tasks, our method surpasses the performance of human or standard self-instruct prompts on both AlpacaEval 2.0 and Arena-Hard. 

---
# PRGB Benchmark: A Robust Placeholder-Assisted Algorithm for Benchmarking Retrieval-Augmented Generation 

**Authors**: Zhehao Tan, Yihan Jiao, Dan Yang, Lei Liu, Jie Feng, Duolin Sun, Yue Shen, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22927)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating external knowledge, where the LLM's ability to generate responses based on the combination of a given query and retrieved documents is crucial. However, most benchmarks focus on overall RAG system performance, rarely assessing LLM-specific capabilities. Current benchmarks emphasize broad aspects such as noise robustness, but lack a systematic and granular evaluation framework on document utilization. To this end, we introduce \textit{Placeholder-RAG-Benchmark}, a multi-level fine-grained benchmark, emphasizing the following progressive dimensions: (1) multi-level filtering abilities, (2) combination abilities, and (3) reference reasoning. To provide a more nuanced understanding of LLMs' roles in RAG systems, we formulate an innovative placeholder-based approach to decouple the contributions of the LLM's parametric knowledge and the external knowledge. Experiments demonstrate the limitations of representative LLMs in the RAG system's generation capabilities, particularly in error resilience and context faithfulness. Our benchmark provides a reproducible framework for developing more reliable and efficient RAG systems. Our code is available in this https URL. 

---
# Large Language Models in the Travel Domain: An Industrial Experience 

**Authors**: Sergio Di Meglio, Aniello Somma, Luigi Libero Lucio Starace, Fabio Scippacercola, Giancarlo Sperlì, Sergio Di Martino  

**Link**: [PDF](https://arxiv.org/pdf/2507.22910)  

**Abstract**: Online property booking platforms are widely used and rely heavily on consistent, up-to-date information about accommodation facilities, often sourced from third-party providers. However, these external data sources are frequently affected by incomplete or inconsistent details, which can frustrate users and result in a loss of market. In response to these challenges, we present an industrial case study involving the integration of Large Language Models (LLMs) into CALEIDOHOTELS, a property reservation platform developed by FERVENTO. We evaluate two well-known LLMs in this context: Mistral 7B, fine-tuned with QLoRA, and Mixtral 8x7B, utilized with a refined system prompt. Both models were assessed based on their ability to generate consistent and homogeneous descriptions while minimizing hallucinations. Mixtral 8x7B outperformed Mistral 7B in terms of completeness (99.6% vs. 93%), precision (98.8% vs. 96%), and hallucination rate (1.2% vs. 4%), producing shorter yet more concise content (249 vs. 277 words on average). However, this came at a significantly higher computational cost: 50GB VRAM and $1.61/hour versus 5GB and $0.16/hour for Mistral 7B. Our findings provide practical insights into the trade-offs between model quality and resource efficiency, offering guidance for deploying LLMs in production environments and demonstrating their effectiveness in enhancing the consistency and reliability of accommodation data. 

---
# TweakLLM: A Routing Architecture for Dynamic Tailoring of Cached Responses 

**Authors**: Muhammad Taha Cheema, Abeer Aamir, Khawaja Gul Muhammad, Naveed Anwar Bhatti, Ihsan Ayyub Qazi, Zafar Ayyub Qazi  

**Link**: [PDF](https://arxiv.org/pdf/2507.23674)  

**Abstract**: Large Language Models (LLMs) process millions of queries daily, making efficient response caching a compelling optimization for reducing cost and latency. However, preserving relevance to user queries using this approach proves difficult due to the personalized nature of chatbot interactions and the limited accuracy of semantic similarity search. To address this, we present TweakLLM, a novel routing architecture that employs a lightweight LLM to dynamically adapt cached responses to incoming prompts. Through comprehensive evaluation, including user studies with side-by-side comparisons, satisfaction voting, as well as multi-agent LLM debates, we demonstrate that TweakLLM maintains response quality comparable to frontier models while significantly improving cache effectiveness. Our results across real-world datasets highlight TweakLLM as a scalable, resource-efficient caching solution for high-volume LLM deployments without compromising user experience. 

---
# TextQuests: How Good are LLMs at Text-Based Video Games? 

**Authors**: Long Phan, Mantas Mazeika, Andy Zou, Dan Hendrycks  

**Link**: [PDF](https://arxiv.org/pdf/2507.23701)  

**Abstract**: Evaluating AI agents within complex, interactive environments that mirror real-world challenges is critical for understanding their practical capabilities. While existing agent benchmarks effectively assess skills like tool use or performance on structured tasks, they often do not fully capture an agent's ability to operate autonomously in exploratory environments that demand sustained, self-directed reasoning over a long and growing context. To spur the development of agents capable of more robust intrinsic reasoning over long horizons, we introduce TextQuests, a benchmark based on the Infocom suite of interactive fiction games. These text-based adventures, which can take human players over 30 hours and require hundreds of precise actions to solve, serve as an effective proxy for evaluating AI agents on focused, stateful tasks. The benchmark is specifically designed to assess an LLM agent's capacity for self-contained problem-solving by precluding the use of external tools, thereby focusing on intrinsic long-context reasoning capabilities in an exploratory environment characterized by the need for trial-and-error learning and sustained problem-solving within a single interactive session. We release TextQuests at this https URL. 

---
# SWE-Exp: Experience-Driven Software Issue Resolution 

**Authors**: Silin Chen, Shaoxin Lin, Xiaodong Gu, Yuling Shi, Heng Lian, Longfei Yun, Dong Chen, Weiguo Sun, Lin Cao, Qianxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23361)  

**Abstract**: Recent advances in large language model (LLM) agents have shown remarkable progress in software issue resolution, leveraging advanced techniques such as multi-agent collaboration and Monte Carlo Tree Search (MCTS). However, current agents act as memoryless explorers - treating each problem separately without retaining or reusing knowledge from previous repair experiences. This leads to redundant exploration of failed trajectories and missed chances to adapt successful issue resolution methods to similar problems. To address this problem, we introduce SWE-Exp, an experience - enhanced approach that distills concise and actionable experience from prior agent trajectories, enabling continuous learning across issues. Our method introduces a multi-faceted experience bank that captures both successful and failed repair attempts. Specifically, it extracts reusable issue resolution knowledge at different levels - from high-level problem comprehension to specific code changes. Experiments show that SWE-Exp achieves state-of-the-art resolution rate (41.6% Pass@1) on SWE-bench-Verified under open-source agent frameworks. Our approach establishes a new paradigm in which automated software engineering agents systematically accumulate and leverage repair expertise, fundamentally shifting from trial-and-error exploration to strategic, experience-driven issue resolution. 

---
# DSBC : Data Science task Benchmarking with Context engineering 

**Authors**: Ram Mohan Rao Kadiyala, Siddhant Gupta, Jebish Purbey, Giulio Martini, Suman Debnath, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.23336)  

**Abstract**: Recent advances in large language models (LLMs) have significantly impacted data science workflows, giving rise to specialized data science agents designed to automate analytical tasks. Despite rapid adoption, systematic benchmarks evaluating the efficacy and limitations of these agents remain scarce. In this paper, we introduce a comprehensive benchmark specifically crafted to reflect real-world user interactions with data science agents by observing usage of our commercial applications. We evaluate three LLMs: Claude-4.0-Sonnet, Gemini-2.5-Flash, and OpenAI-o4-Mini across three approaches: zero-shot with context engineering, multi-step with context engineering, and with SmolAgent. Our benchmark assesses performance across a diverse set of eight data science task categories, additionally exploring the sensitivity of models to common prompting issues, such as data leakage and slightly ambiguous instructions. We further investigate the influence of temperature parameters on overall and task-specific outcomes for each model and approach. Our findings reveal distinct performance disparities among the evaluated models and methodologies, highlighting critical factors that affect practical deployment. The benchmark dataset and evaluation framework introduced herein aim to provide a foundation for future research of more robust and effective data science agents. 

---
# SWE-Debate: Competitive Multi-Agent Debate for Software Issue Resolution 

**Authors**: Han Li, Yuling Shi, Shaoxin Lin, Xiaodong Gu, Heng Lian, Xin Wang, Yantao Jia, Tao Huang, Qianxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23348)  

**Abstract**: Issue resolution has made remarkable progress thanks to the advanced reasoning capabilities of large language models (LLMs). Recently, agent-based frameworks such as SWE-agent have further advanced this progress by enabling autonomous, tool-using agents to tackle complex software engineering tasks. While existing agent-based issue resolution approaches are primarily based on agents' independent explorations, they often get stuck in local solutions and fail to identify issue patterns that span across different parts of the codebase. To address this limitation, we propose SWE-Debate, a competitive multi-agent debate framework that encourages diverse reasoning paths and achieves more consolidated issue localization. SWE-Debate first creates multiple fault propagation traces as localization proposals by traversing a code dependency graph. Then, it organizes a three-round debate among specialized agents, each embodying distinct reasoning perspectives along the fault propagation trace. This structured competition enables agents to collaboratively converge on a consolidated fix plan. Finally, this consolidated fix plan is integrated into an MCTS-based code modification agent for patch generation. Experiments on the SWE-bench benchmark show that SWE-Debate achieves new state-of-the-art results in open-source agent frameworks and outperforms baselines by a large margin. 

---
# ELMES: An Automated Framework for Evaluating Large Language Models in Educational Scenarios 

**Authors**: Shou'ang Wei, Xinyun Wang, Shuzhen Bi, Jian Chen, Ruijia Li, Bo Jiang, Xin Lin, Min Zhang, Yu Song, BingDong Li, Aimin Zhou, Hao Hao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22947)  

**Abstract**: The emergence of Large Language Models (LLMs) presents transformative opportunities for education, generating numerous novel application scenarios. However, significant challenges remain: evaluation metrics vary substantially across different educational scenarios, while many emerging scenarios lack appropriate assessment metrics. Current benchmarks predominantly measure general intelligence rather than pedagogical capabilities. To address this gap, we introduce ELMES, an open-source automated evaluation framework specifically designed for assessing LLMs in educational settings. ELMES features a modular architecture that enables researchers to create dynamic, multi-agent dialogues through simple configuration files, facilitating flexible scenario design without requiring extensive programming expertise. The framework incorporates a hybrid evaluation engine that objectively quantifies traditionally subjective pedagogical metrics using an LLM-as-a-Judge methodology. We conduct systematic benchmarking of state-of-the-art LLMs across four critical educational scenarios: Knowledge Point Explanation, Guided Problem-Solving Teaching, Interdisciplinary Lesson Plan Generation, and Contextualized Question Generation, employing fine-grained metrics developed in collaboration with education specialists. Our results demonstrate distinct capability distributions among models, revealing context-specific strengths and limitations. ELMES provides educators and researchers with an accessible evaluation framework that significantly reduces adaptation barriers for diverse educational applications while advancing the practical implementation of LLMs in pedagogy. The framework is publicly available at \emph{this https URL}. 

---
# A Hybrid Framework for Subject Analysis: Integrating Embedding-Based Regression Models with Large Language Models 

**Authors**: Jinyu Liu, Xiaoying Song, Diana Zhang, Jason Thomale, Daqing He, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22913)  

**Abstract**: Providing subject access to information resources is an essential function of any library management system. Large language models (LLMs) have been widely used in classification and summarization tasks, but their capability to perform subject analysis is underexplored. Multi-label classification with traditional machine learning (ML) models has been used for subject analysis but struggles with unseen cases. LLMs offer an alternative but often over-generate and hallucinate. Therefore, we propose a hybrid framework that integrates embedding-based ML models with LLMs. This approach uses ML models to (1) predict the optimal number of LCSH labels to guide LLM predictions and (2) post-edit the predicted terms with actual LCSH terms to mitigate hallucinations. We experimented with LLMs and the hybrid framework to predict the subject terms of books using the Library of Congress Subject Headings (LCSH). Experiment results show that providing initial predictions to guide LLM generations and imposing post-edits result in more controlled and vocabulary-aligned outputs. 

---
# CoE-Ops: Collaboration of LLM-based Experts for AIOps Question-Answering 

**Authors**: Jinkun Zhao, Yuanshuai Wang, Xingjian Zhang, Ruibo Chen, Xingchuang Liao, Junle Wang, Lei Huang, Kui Zhang, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22937)  

**Abstract**: With the rapid evolution of artificial intelligence, AIOps has emerged as a prominent paradigm in DevOps. Lots of work has been proposed to improve the performance of different AIOps phases. However, constrained by domain-specific knowledge, a single model can only handle the operation requirement of a specific task,such as log parser,root cause analysis. Meanwhile, combining multiple models can achieve more efficient results, which have been proved in both previous ensemble learning and the recent LLM training domain. Inspired by these works,to address the similar challenges in AIOPS, this paper first proposes a collaboration-of-expert framework(CoE-Ops) incorporating a general-purpose large language model task classifier. A retrieval-augmented generation mechanism is introduced to improve the framework's capability in handling both Question-Answering tasks with high-level(Code,build,Test,etc.) and low-level(fault analysis,anomaly detection,etc.). Finally, the proposed method is implemented in the AIOps domain, and extensive experiments are conducted on the DevOps-EVAL dataset. Experimental results demonstrate that CoE-Ops achieves a 72% improvement in routing accuracy for high-level AIOps tasks compared to existing CoE methods, delivers up to 8% accuracy enhancement over single AIOps models in DevOps problem resolution, and outperforms larger-scale Mixture-of-Experts (MoE) models by up to 14% in accuracy. 

---
# Theoretical Foundations and Mitigation of Hallucination in Large Language Models 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2507.22915)  

**Abstract**: Hallucination in Large Language Models (LLMs) refers to the generation of content that is not faithful to the input or the real-world facts. This paper provides a rigorous treatment of hallucination in LLMs, including formal definitions and theoretical analyses. We distinguish between intrinsic and extrinsic hallucinations, and define a \textit{hallucination risk} for models. We derive bounds on this risk using learning-theoretic frameworks (PAC-Bayes and Rademacher complexity). We then survey detection strategies for hallucinations, such as token-level uncertainty estimation, confidence calibration, and attention alignment checks. On the mitigation side, we discuss approaches including retrieval-augmented generation, hallucination-aware fine-tuning, logit calibration, and the incorporation of fact-verification modules. We propose a unified detection and mitigation workflow, illustrated with a diagram, to integrate these strategies. Finally, we outline evaluation protocols for hallucination, recommending datasets, metrics, and experimental setups to quantify and reduce hallucinations. Our work lays a theoretical foundation and practical guidelines for addressing the crucial challenge of hallucination in LLMs. 

---
# MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks 

**Authors**: Yadong Niu, Tianzi Wang, Heinrich Dinkel, Xingwei Sun, Jiahao Zhou, Gang Li, Jizhong Liu, Xunying Liu, Junbo Zhang, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2507.23511)  

**Abstract**: While large audio-language models have advanced open-ended audio understanding, they still fall short of nuanced human-level comprehension. This gap persists largely because current benchmarks, limited by data annotations and evaluation metrics, fail to reliably distinguish between generic and highly detailed model outputs. To this end, this work introduces MECAT, a Multi-Expert Constructed Benchmark for Fine-Grained Audio Understanding Tasks. Generated via a pipeline that integrates analysis from specialized expert models with Chain-of-Thought large language model reasoning, MECAT provides multi-perspective, fine-grained captions and open-set question-answering pairs. The benchmark is complemented by a novel metric: DATE (Discriminative-Enhanced Audio Text Evaluation). This metric penalizes generic terms and rewards detailed descriptions by combining single-sample semantic similarity with cross-sample discriminability. A comprehensive evaluation of state-of-the-art audio models is also presented, providing new insights into their current capabilities and limitations. The data and code are available at this https URL 

---
# Hybrid EEG--Driven Brain--Computer Interface: A Large Language Model Framework for Personalized Language Rehabilitation 

**Authors**: Ismail Hossain, Mridul Banik  

**Link**: [PDF](https://arxiv.org/pdf/2507.22892)  

**Abstract**: Conventional augmentative and alternative communication (AAC) systems and language-learning platforms often fail to adapt in real time to the user's cognitive and linguistic needs, especially in neurological conditions such as post-stroke aphasia or amyotrophic lateral sclerosis. Recent advances in noninvasive electroencephalography (EEG)--based brain-computer interfaces (BCIs) and transformer--based large language models (LLMs) offer complementary strengths: BCIs capture users' neural intent with low fatigue, while LLMs generate contextually tailored language content. We propose and evaluate a novel hybrid framework that leverages real-time EEG signals to drive an LLM-powered language rehabilitation assistant. This system aims to: (1) enable users with severe speech or motor impairments to navigate language-learning modules via mental commands; (2) dynamically personalize vocabulary, sentence-construction exercises, and corrective feedback; and (3) monitor neural markers of cognitive effort to adjust task difficulty on the fly. 

---
# A Language Model-Driven Semi-Supervised Ensemble Framework for Illicit Market Detection Across Deep/Dark Web and Social Platforms 

**Authors**: Navid Yazdanjue, Morteza Rakhshaninejad, Hossein Yazdanjouei, Mohammad Sadegh Khorshidi, Mikko S. Niemela, Fang Chen, Amir H. Gandomi  

**Link**: [PDF](https://arxiv.org/pdf/2507.22912)  

**Abstract**: Illegal marketplaces have increasingly shifted to concealed parts of the internet, including the deep and dark web, as well as platforms such as Telegram, Reddit, and Pastebin. These channels enable the anonymous trade of illicit goods including drugs, weapons, and stolen credentials. Detecting and categorizing such content remains challenging due to limited labeled data, the evolving nature of illicit language, and the structural heterogeneity of online sources. This paper presents a hierarchical classification framework that combines fine-tuned language models with a semi-supervised ensemble learning strategy to detect and classify illicit marketplace content across diverse platforms. We extract semantic representations using ModernBERT, a transformer model for long documents, finetuned on domain-specific data from deep and dark web pages, Telegram channels, Subreddits, and Pastebin pastes to capture specialized jargon and ambiguous linguistic patterns. In addition, we incorporate manually engineered features such as document structure, embedded patterns including Bitcoin addresses, emails, and IPs, and metadata, which complement language model embeddings. The classification pipeline operates in two stages. The first stage uses a semi-supervised ensemble of XGBoost, Random Forest, and SVM with entropy-based weighted voting to detect sales-related documents. The second stage further classifies these into drug, weapon, or credential sales. Experiments on three datasets, including our multi-source corpus, DUTA, and CoDA, show that our model outperforms several baselines, including BERT, ModernBERT, DarkBERT, ALBERT, Longformer, and BigBird. The model achieves an accuracy of 0.96489, an F1-score of 0.93467, and a TMCC of 0.95388, demonstrating strong generalization, robustness under limited supervision, and effectiveness in real-world illicit content detection. 

---
# Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey 

**Authors**: Jindong Li, Yali Fu, Jiahong Liu, Linxiao Cao, Wei Ji, Menglin Yang, Irwin King, Ming-Hsuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.22920)  

**Abstract**: The rapid advancement of large language models (LLMs) has intensified the need for effective mechanisms to transform continuous multimodal data into discrete representations suitable for language-based processing. Discrete tokenization, with vector quantization (VQ) as a central approach, offers both computational efficiency and compatibility with LLM architectures. Despite its growing importance, there is a lack of a comprehensive survey that systematically examines VQ techniques in the context of LLM-based systems. This work fills this gap by presenting the first structured taxonomy and analysis of discrete tokenization methods designed for LLMs. We categorize 8 representative VQ variants that span classical and modern paradigms and analyze their algorithmic principles, training dynamics, and integration challenges with LLM pipelines. Beyond algorithm-level investigation, we discuss existing research in terms of classical applications without LLMs, LLM-based single-modality systems, and LLM-based multimodal systems, highlighting how quantization strategies influence alignment, reasoning, and generation performance. In addition, we identify key challenges including codebook collapse, unstable gradient estimation, and modality-specific encoding constraints. Finally, we discuss emerging research directions such as dynamic and task-adaptive quantization, unified tokenization frameworks, and biologically inspired codebook learning. This survey bridges the gap between traditional vector quantization and modern LLM applications, serving as a foundational reference for the development of efficient and generalizable multimodal systems. A continuously updated version is available at: this https URL. 

---
# DICE: Dynamic In-Context Example Selection in LLM Agents via Efficient Knowledge Transfer 

**Authors**: Ruoyu Wang, Junda Wu, Yu Xia, Tong Yu, Ryan A. Rossi, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2507.23554)  

**Abstract**: Large language model-based agents, empowered by in-context learning (ICL), have demonstrated strong capabilities in complex reasoning and tool-use tasks. However, existing works have shown that the effectiveness of ICL is highly sensitive to the choice of demonstrations, with suboptimal examples often leading to unstable or degraded performance. While prior work has explored example selection, including in some agentic or multi-step settings, existing approaches typically rely on heuristics or task-specific designs and lack a general, theoretically grounded criterion for what constitutes an effective demonstration across reasoning steps. Therefore, it is non-trivial to develop a principled, general-purpose method for selecting demonstrations that consistently benefit agent performance. In this paper, we address this challenge with DICE, Dynamic In-Context Example Selection for LLM Agents, a theoretically grounded ICL framework for agentic tasks that selects the most relevant demonstrations at each step of reasoning. Our approach decomposes demonstration knowledge into transferable and non-transferable components through a causal lens, showing how the latter can introduce spurious dependencies that impair generalization. We further propose a stepwise selection criterion with a formal guarantee of improved agent performance. Importantly, DICE is a general, framework-agnostic solution that can be integrated as a plug-in module into existing agentic frameworks without any additional training cost. Extensive experiments across diverse domains demonstrate our method's effectiveness and generality, highlighting the importance of principled, context-aware demo selection for robust and efficient LLM agents. 

---
# Chatting with your ERP: A Recipe 

**Authors**: Jorge Ruiz Gómez, Lidia Andrés Susinos, Jorge Alamo Olivé, Sonia Rey Osorno, Manuel Luis Gonzalez Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2507.23429)  

**Abstract**: This paper presents the design, implementation, and evaluation behind a Large Language Model (LLM) agent that chats with an industrial production-grade ERP system. The agent is capable of interpreting natural language queries and translating them into executable SQL statements, leveraging open-weight LLMs. A novel dual-agent architecture combining reasoning and critique stages was proposed to improve query generation reliability. 

---
# MemoCue: Empowering LLM-Based Agents for Human Memory Recall via Strategy-Guided Querying 

**Authors**: Qian Zhao, Zhuo Sun, Bin Guo, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23633)  

**Abstract**: Agent-assisted memory recall is one critical research problem in the field of human-computer interaction. In conventional methods, the agent can retrieve information from its equipped memory module to help the person recall incomplete or vague memories. The limited size of memory module hinders the acquisition of complete memories and impacts the memory recall performance in practice. Memory theories suggest that the person's relevant memory can be proactively activated through some effective cues. Inspired by this, we propose a novel strategy-guided agent-assisted memory recall method, allowing the agent to transform an original query into a cue-rich one via the judiciously designed strategy to help the person recall memories. To this end, there are two key challenges. (1) How to choose the appropriate recall strategy for diverse forgetting scenarios with distinct memory-recall characteristics? (2) How to obtain the high-quality responses leveraging recall strategies, given only abstract and sparsely annotated strategy patterns? To address the challenges, we propose a Recall Router framework. Specifically, we design a 5W Recall Map to classify memory queries into five typical scenarios and define fifteen recall strategy patterns across the corresponding scenarios. We then propose a hierarchical recall tree combined with the Monte Carlo Tree Search algorithm to optimize the selection of strategy and the generation of strategy responses. We construct an instruction tuning dataset and fine-tune multiple open-source large language models (LLMs) to develop MemoCue, an agent that excels in providing memory-inspired responses. Experiments on three representative datasets show that MemoCue surpasses LLM-based methods by 17.74% in recall inspiration. Further human evaluation highlights its advantages in memory-recall applications. 

---
# Causal Reasoning in Pieces: Modular In-Context Learning for Causal Discovery 

**Authors**: Kacper Kadziolka, Saber Salehkaleybar  

**Link**: [PDF](https://arxiv.org/pdf/2507.23488)  

**Abstract**: Causal inference remains a fundamental challenge for large language models. Recent advances in internal reasoning with large language models have sparked interest in whether state-of-the-art reasoning models can robustly perform causal discovery-a task where conventional models often suffer from severe overfitting and near-random performance under data perturbations. We study causal discovery on the Corr2Cause benchmark using the emergent OpenAI's o-series and DeepSeek-R model families and find that these reasoning-first architectures achieve significantly greater native gains than prior approaches. To capitalize on these strengths, we introduce a modular in-context pipeline inspired by the Tree-of-Thoughts and Chain-of-Thoughts methodologies, yielding nearly three-fold improvements over conventional baselines. We further probe the pipeline's impact by analyzing reasoning chain length, complexity, and conducting qualitative and quantitative comparisons between conventional and reasoning models. Our findings suggest that while advanced reasoning models represent a substantial leap forward, carefully structured in-context frameworks are essential to maximize their capabilities and offer a generalizable blueprint for causal discovery across diverse domains. 

---
# How Far Are AI Scientists from Changing the World? 

**Authors**: Qiujie Xie, Yixuan Weng, Minjun Zhu, Fuchen Shen, Shulin Huang, Zhen Lin, Jiahui Zhou, Zilan Mao, Zijie Yang, Linyi Yang, Jian Wu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23276)  

**Abstract**: The emergence of large language models (LLMs) is propelling automated scientific discovery to the next level, with LLM-based Artificial Intelligence (AI) Scientist systems now taking the lead in scientific research. Several influential works have already appeared in the field of AI Scientist systems, with AI-generated research papers having been accepted at the ICLR 2025 workshop, suggesting that a human-level AI Scientist capable of uncovering phenomena previously unknown to humans, may soon become a reality. In this survey, we focus on the central question: How far are AI scientists from changing the world and reshaping the scientific research paradigm? To answer this question, we provide a prospect-driven review that comprehensively analyzes the current achievements of AI Scientist systems, identifying key bottlenecks and the critical components required for the emergence of a scientific agent capable of producing ground-breaking discoveries that solve grand challenges. We hope this survey will contribute to a clearer understanding of limitations of current AI Scientist systems, showing where we are, what is missing, and what the ultimate goals for scientific AI should be. 

---
# LLM4Rail: An LLM-Augmented Railway Service Consulting Platform 

**Authors**: Zhuo Li, Xianghuai Deng, Chiwei Feng, Hanmeng Li, Shenjie Wang, Haichao Zhang, Teng Jia, Conlin Chen, Louis Linchun Wu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23377)  

**Abstract**: Large language models (LLMs) have significantly reshaped different walks of business. To meet the increasing demands for individualized railway service, we develop LLM4Rail - a novel LLM-augmented railway service consulting platform. Empowered by LLM, LLM4Rail can provide custom modules for ticketing, railway food & drink recommendations, weather information, and chitchat. In LLM4Rail, we propose the iterative "Question-Thought-Action-Observation (QTAO)" prompting framework. It meticulously integrates verbal reasoning with task-oriented actions, that is, reasoning to guide action selection, to effectively retrieve external observations relevant to railway operation and service to generate accurate responses. To provide personalized onboard dining services, we first construct the Chinese Railway Food and Drink (CRFD-25) - a publicly accessible takeout dataset tailored for railway services. CRFD-25 covers a wide range of signature dishes categorized by cities, cuisines, age groups, and spiciness levels. We further introduce an LLM-based zero-shot conversational recommender for railway catering. To address the unconstrained nature of open recommendations, the feature similarity-based post-processing step is introduced to ensure all the recommended items are aligned with CRFD-25 dataset. 

---
# FairReason: Balancing Reasoning and Social Bias in MLLMs 

**Authors**: Zhenyu Pan, Yutong Zhang, Jianshu Zhang, Haoran Lu, Haozheng Luo, Yuwei Han, Philip S. Yu, Manling Li, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23067)  

**Abstract**: Multimodal Large Language Models (MLLMs) already achieve state-of-the-art results across a wide range of tasks and modalities. To push their reasoning ability further, recent studies explore advanced prompting schemes and post-training fine-tuning. Although these techniques improve logical accuracy, they frequently leave the models' outputs burdened with pronounced social biases. Clarifying how reasoning gains interact with bias mitigation-and whether the two objectives inherently trade off-therefore remains an open and pressing research problem. Our study begins by benchmarking three bias-mitigation strategies-supervised fine-uning (SFT), knowledge distillation (KD), and rule-based reinforcement learning (RL)-under identical conditions, establishing their baseline strengths and weaknesses. Building on these results, we vary the proportion of debias-focused and reasoning-centric samples within each paradigm to chart the reasoning-versus-bias trade-off. Our sweeps reveal a consistent sweet spot: a roughly 1:4 mix trained with reinforcement learning cuts stereotype scores by 10% while retaining 88% of the model's original reasoning accuracy, offering concrete guidance for balancing fairness and capability in MLLMs. 

---
# A survey of multi-agent geosimulation methodologies: from ABM to LLM 

**Authors**: Virginia Padilla, Jacinto Dávila  

**Link**: [PDF](https://arxiv.org/pdf/2507.23694)  

**Abstract**: We provide a comprehensive examination of agent-based approaches that codify the principles and linkages underlying multi-agent systems, simulations, and information systems. Based on two decades of study, this paper confirms a framework intended as a formal specification for geosimulation platforms. Our findings show that large language models (LLMs) can be effectively incorporated as agent components if they follow a structured architecture specific to fundamental agent activities such as perception, memory, planning, and action. This integration is precisely consistent with the architecture that we formalize, providing a solid platform for next-generation geosimulation systems. 

---
# Distributed AI Agents for Cognitive Underwater Robot Autonomy 

**Authors**: Markus Buchholz, Ignacio Carlucho, Michele Grimaldi, Yvan R. Petillot  

**Link**: [PDF](https://arxiv.org/pdf/2507.23735)  

**Abstract**: Achieving robust cognitive autonomy in robots navigating complex, unpredictable environments remains a fundamental challenge in robotics. This paper presents Underwater Robot Self-Organizing Autonomy (UROSA), a groundbreaking architecture leveraging distributed Large Language Model AI agents integrated within the Robot Operating System 2 (ROS 2) framework to enable advanced cognitive capabilities in Autonomous Underwater Vehicles. UROSA decentralises cognition into specialised AI agents responsible for multimodal perception, adaptive reasoning, dynamic mission planning, and real-time decision-making. Central innovations include flexible agents dynamically adapting their roles, retrieval-augmented generation utilising vector databases for efficient knowledge management, reinforcement learning-driven behavioural optimisation, and autonomous on-the-fly ROS 2 node generation for runtime functional extensibility. Extensive empirical validation demonstrates UROSA's promising adaptability and reliability through realistic underwater missions in simulation and real-world deployments, showing significant advantages over traditional rule-based architectures in handling unforeseen scenarios, environmental uncertainties, and novel mission objectives. This work not only advances underwater autonomy but also establishes a scalable, safe, and versatile cognitive robotics framework capable of generalising to a diverse array of real-world applications. 

---
# LLM-Based Identification of Infostealer Infection Vectors from Screenshots: The Case of Aurora 

**Authors**: Estelle Ruellan, Eric Clay, Nicholas Ascoli  

**Link**: [PDF](https://arxiv.org/pdf/2507.23611)  

**Abstract**: Infostealers exfiltrate credentials, session cookies, and sensitive data from infected systems. With over 29 million stealer logs reported in 2024, manual analysis and mitigation at scale are virtually unfeasible/unpractical. While most research focuses on proactive malware detection, a significant gap remains in leveraging reactive analysis of stealer logs and their associated artifacts. Specifically, infection artifacts such as screenshots, image captured at the point of compromise, are largely overlooked by the current literature. This paper introduces a novel approach leveraging Large Language Models (LLMs), more specifically gpt-4o-mini, to analyze infection screenshots to extract potential Indicators of Compromise (IoCs), map infection vectors, and track campaigns. Focusing on the Aurora infostealer, we demonstrate how LLMs can process screenshots to identify infection vectors, such as malicious URLs, installer files, and exploited software themes. Our method extracted 337 actionable URLs and 246 relevant files from 1000 screenshots, revealing key malware distribution methods and social engineering tactics. By correlating extracted filenames, URLs, and infection themes, we identified three distinct malware campaigns, demonstrating the potential of LLM-driven analysis for uncovering infection workflows and enhancing threat intelligence. By shifting malware analysis from traditional log-based detection methods to a reactive, artifact-driven approach that leverages infection screenshots, this research presents a scalable method for identifying infection vectors and enabling early intervention. 

---
# A Unified Perception-Language-Action Framework for Adaptive Autonomous Driving 

**Authors**: Yi Zhang, Erik Leo Haß, Kuo-Yi Chao, Nenad Petrovic, Yinglei Song, Chengdong Wu, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2507.23540)  

**Abstract**: Autonomous driving systems face significant challenges in achieving human-like adaptability, robustness, and interpretability in complex, open-world environments. These challenges stem from fragmented architectures, limited generalization to novel scenarios, and insufficient semantic extraction from perception. To address these limitations, we propose a unified Perception-Language-Action (PLA) framework that integrates multi-sensor fusion (cameras, LiDAR, radar) with a large language model (LLM)-augmented Vision-Language-Action (VLA) architecture, specifically a GPT-4.1-powered reasoning core. This framework unifies low-level sensory processing with high-level contextual reasoning, tightly coupling perception with natural language-based semantic understanding and decision-making to enable context-aware, explainable, and safety-bounded autonomous driving. Evaluations on an urban intersection scenario with a construction zone demonstrate superior performance in trajectory tracking, speed prediction, and adaptive planning. The results highlight the potential of language-augmented cognitive frameworks for advancing the safety, interpretability, and scalability of autonomous driving systems. 

---
# Can LLM-Reasoning Models Replace Classical Planning? A Benchmark Study 

**Authors**: Kai Goebel, Patrik Zips  

**Link**: [PDF](https://arxiv.org/pdf/2507.23589)  

**Abstract**: Recent advancements in Large Language Models have sparked interest in their potential for robotic task planning. While these models demonstrate strong generative capabilities, their effectiveness in producing structured and executable plans remains uncertain. This paper presents a systematic evaluation of a broad spectrum of current state of the art language models, each directly prompted using Planning Domain Definition Language domain and problem files, and compares their planning performance with the Fast Downward planner across a variety of benchmarks. In addition to measuring success rates, we assess how faithfully the generated plans translate into sequences of actions that can actually be executed, identifying both strengths and limitations of using these models in this setting. Our findings show that while the models perform well on simpler planning tasks, they continue to struggle with more complex scenarios that require precise resource management, consistent state tracking, and strict constraint compliance. These results underscore fundamental challenges in applying language models to robotic planning in real world environments. By outlining the gaps that emerge during execution, we aim to guide future research toward combined approaches that integrate language models with classical planners in order to enhance the reliability and scalability of planning in autonomous robotics. 

---
# Argumentatively Coherent Judgmental Forecasting 

**Authors**: Deniz Gorur, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2507.23163)  

**Abstract**: Judgmental forecasting employs human opinions to make predictions about future events, rather than exclusively historical data as in quantitative forecasting. When these opinions form an argumentative structure around forecasts, it is useful to study the properties of the forecasts from an argumentative perspective. In this paper, we advocate and formally define a property of argumentative coherence, which, in essence, requires that a forecaster's reasoning is coherent with their forecast. We then conduct three evaluations with our notion of coherence. First, we assess the impact of enforcing coherence on human forecasters as well as on Large Language Model (LLM)-based forecasters, given that they have recently shown to be competitive with human forecasters. In both cases, we show that filtering out incoherent predictions improves forecasting accuracy consistently, supporting the practical value of coherence in both human and LLM-based forecasting. Then, via crowd-sourced user experiments, we show that, despite its apparent intuitiveness and usefulness, users do not generally align with this coherence property. This points to the need to integrate, within argumentation-based judgmental forecasting, mechanisms to filter out incoherent opinions before obtaining group forecasting predictions. 

---
# MPCC: A Novel Benchmark for Multimodal Planning with Complex Constraints in Multimodal Large Language Models 

**Authors**: Yiyan Ji, Haoran Chen, Qiguang Chen, Chengyue Wu, Libo Qin, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2507.23382)  

**Abstract**: Multimodal planning capabilities refer to the ability to predict, reason, and design steps for task execution with multimodal context, which is essential for complex reasoning and decision-making across multiple steps. However, current benchmarks face two key challenges: (1) they cannot directly assess multimodal real-world planning capabilities, and (2) they lack constraints or implicit constraints across modalities. To address these issues, we introduce Multimodal Planning with Complex Constraints (MPCC), the first benchmark to systematically evaluate MLLMs' ability to handle multimodal constraints in planning. To address the first challenge, MPCC focuses on three real-world tasks: Flight Planning, Calendar Planning, and Meeting Planning. To solve the second challenge, we introduce complex constraints (e.g. budget, temporal, and spatial) in these tasks, with graded difficulty levels (EASY, MEDIUM, HARD) to separate constraint complexity from search space expansion. Experiments on 13 advanced MLLMs reveal significant challenges: closed-source models achieve only 21.3% feasible plans, while open-source models average below 11%. Additionally, we observe that MLLMs are highly sensitive to constraint complexity and that traditional multimodal prompting strategies fail in multi-constraint scenarios. Our work formalizes multimodal constraints in planning, provides a rigorous evaluation framework, and highlights the need for advancements in constraint-aware reasoning for real-world MLLM applications. 

---
# Automated Feedback on Student-Generated UML and ER Diagrams Using Large Language Models 

**Authors**: Sebastian Gürtl, Gloria Schimetta, David Kerschbaumer, Michael Liut, Alexander Steinmaurer  

**Link**: [PDF](https://arxiv.org/pdf/2507.23470)  

**Abstract**: UML and ER diagrams are foundational in computer science education but come with challenges for learners due to the need for abstract thinking, contextual understanding, and mastery of both syntax and semantics. These complexities are difficult to address through traditional teaching methods, which often struggle to provide scalable, personalized feedback, especially in large classes. We introduce DUET (Diagrammatic UML & ER Tutor), a prototype of an LLM-based tool, which converts a reference diagram and a student-submitted diagram into a textual representation and provides structured feedback based on the differences. It uses a multi-stage LLM pipeline to compare diagrams and generate reflective feedback. Furthermore, the tool enables analytical insights for educators, aiming to foster self-directed learning and inform instructional strategies. We evaluated DUET through semi-structured interviews with six participants, including two educators and four teaching assistants. They identified strengths such as accessibility, scalability, and learning support alongside limitations, including reliability and potential misuse. Participants also suggested potential improvements, such as bulk upload functionality and interactive clarification features. DUET presents a promising direction for integrating LLMs into modeling education and offers a foundation for future classroom integration and empirical evaluation. 

---
# Trae Agent: An LLM-based Agent for Software Engineering with Test-time Scaling 

**Authors**: Trae Research Team, Pengfei Gao, Zhao Tian, Xiangxin Meng, Xinchen Wang, Ruida Hu, Yuanan Xiao, Yizhou Liu, Zhao Zhang, Junjie Chen, Cuiyun Gao, Yun Lin, Yingfei Xiong, Chao Peng, Xia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23370)  

**Abstract**: Software issue resolution is a critical challenge in software engineering and has garnered increasing attention in recent years. With the rapid advancement of large language models (LLMs), substantial progress has been made in addressing real-world software engineering tasks. Recent studies have introduced ensemble reasoning techniques to enhance the performance of LLM-based issue resolution. However, existing prompting-based methods still face limitations in effectively exploring large ensemble spaces and lack the capacity for repository-level understanding, both of which constrain their overall effectiveness. In this paper, we propose Trae Agent, the first agent-based ensemble reasoning approach for repository-level issue resolution. Trae Agent formulates our goal as an optimal solution search problem and addresses two key challenges, i.e., large ensemble spaces and repository-level understanding, through modular agents for generation, pruning, and selection. We conduct extensive experiments using three leading LLMs on the widely-adopted SWE-bench benchmark, comparing Trae Agent against four state-of-the-art ensemble reasoning techniques. Experimental results demonstrate that Trae Agent consistently achieves superior performance, with an average improvement of 10.22% over all baselines in terms of Pass@1. Trae Agent has achieved first place on the SWE-bench Verified leaderboard, with a notable Pass@1 score of 75.20%. We are pleased to release Trae Agent as an open-source project to support the research community, with all resources available at this https URL. 

---
# DynaSwarm: Dynamically Graph Structure Selection for LLM-based Multi-agent System 

**Authors**: Hui Yi Leong, Yuqing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.23261)  

**Abstract**: Current multi-agent systems (MAS) frameworks often rely on manually designed and static collaboration graph structures, limiting adaptability and performance. To address these limitations, we propose DynaSwarm, a dynamic framework that enhances LLM-based MAS through two key innovations: (1) an actor-critic reinforcement learning (A2C) mechanism to optimize graph structures with improved stability over prior RL methods, and (2) a dynamic graph selector that adaptively chooses the optimal graph structure for each input sample via parameter-efficient LLM fine-tuning. DynaSwarm eliminates the need for rigid, one-fits-all graph architectures, instead leveraging sample-specific idiosyncrasies to dynamically route queries through specialized agent networks. (c) We propose to fine-tune the demonstration retriever to fully exploit the power of in-context learning (ICL). Extensive experiments on question answering, mathematical reasoning, and coding tasks demonstrate that DynaSwarm consistently outperforms state-of-the-art single-agent and MAS baselines across multiple LLM backbones. Our findings highlight the importance of sample-aware structural flexibility in LLM MAS designs. 

---
# "I made this (sort of)": Negotiating authorship, confronting fraudulence, and exploring new musical spaces with prompt-based AI music generation 

**Authors**: Bob L. T. Sturm  

**Link**: [PDF](https://arxiv.org/pdf/2507.23365)  

**Abstract**: I reflect on my experience creating two music albums centered on state-of-the-art prompt-based AI music generation platforms. The first album explicitly poses the question: What happens when I collide my junk mail with these platforms? The second album is a direct response to the first, and toys with the inability of state-of-the-art prompt-based AI music generation platforms to generate music that is not ``practiced'', ``polished'', and ``produced''. I seed a large language model (LLM) with information about these albums and have it interview me, which results in the exploration of several deeper questions: To what extent am I the author? Where am I in the resulting music? How is my musical identity changing as I am faced with machines that are in some ways far more talented than I? What new musical spaces does my work open, for me or anyone/thing else? I conclude by reflecting on my reflections, as well as LLM-mediated self-reflection as method. 

---
# From LLMs to Edge: Parameter-Efficient Fine-Tuning on Edge Devices 

**Authors**: Georg Slamanig, Francesco Corti, Olga Saukh  

**Link**: [PDF](https://arxiv.org/pdf/2507.23536)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods reduce the computational costs of updating deep learning models by minimizing the number of additional parameters used to adapt a model to a down- stream task. While extensively researched in large language models (LLMs), their application to smaller models used on edge devices, such as convolutional neural networks, remains underexplored. This paper benchmarks and analyzes popular PEFT methods on convolutional architectures typically deployed in resource-constrained edge environments. We evaluate LoRA, DoRA, and GaLore for updating standard and depthwise convolutional architectures to handle distribution shifts and accommodate unseen classes. We utilize recently proposed PyTorch profilers to compare the updated model performance and computational costs of these PEFT methods with traditional fine-tuning approaches. With resource efficiency in mind, we investigate their update behavior across different rank dimensions. We find that the evaluated PEFT methods are only half as memory-efficient when applied to depthwise-separable convolution architectures, compared to their efficiency with LLMs. Conversely, when targeting convolu- tional architectures optimized for edge deployment, adapter-based PEFT methods can reduce floating point operations (FLOPs) during model updates by up to 95%. These insights offer valuable guidance for selecting PEFT methods based on hardware constraints, performance requirements, and application needs. Our code is online. 

---
# Beyond Rigid AI: Towards Natural Human-Machine Symbiosis for Interoperative Surgical Assistance 

**Authors**: Lalithkumar Seenivasan, Jiru Xu, Roger D. Soberanis Mukul, Hao Ding, Grayson Byrd, Yu-Chun Ku, Jose L. Porras, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2507.23088)  

**Abstract**: Emerging surgical data science and robotics solutions, especially those designed to provide assistance in situ, require natural human-machine interfaces to fully unlock their potential in providing adaptive and intuitive aid. Contemporary AI-driven solutions remain inherently rigid, offering limited flexibility and restricting natural human-machine interaction in dynamic surgical environments. These solutions rely heavily on extensive task-specific pre-training, fixed object categories, and explicit manual-prompting. This work introduces a novel Perception Agent that leverages speech-integrated prompt-engineered large language models (LLMs), segment anything model (SAM), and any-point tracking foundation models to enable a more natural human-machine interaction in real-time intraoperative surgical assistance. Incorporating a memory repository and two novel mechanisms for segmenting unseen elements, Perception Agent offers the flexibility to segment both known and unseen elements in the surgical scene through intuitive interaction. Incorporating the ability to memorize novel elements for use in future surgeries, this work takes a marked step towards human-machine symbiosis in surgical procedures. Through quantitative analysis on a public dataset, we show that the performance of our agent is on par with considerably more labor-intensive manual-prompting strategies. Qualitatively, we show the flexibility of our agent in segmenting novel elements (instruments, phantom grafts, and gauze) in a custom-curated dataset. By offering natural human-machine interaction and overcoming rigidity, our Perception Agent potentially brings AI-based real-time assistance in dynamic surgical environments closer to reality. 

---
# Accessibility Scout: Personalized Accessibility Scans of Built Environments 

**Authors**: William Huang, Xia Su, Jon E. Froehlich, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.23190)  

**Abstract**: Assessing the accessibility of unfamiliar built environments is critical for people with disabilities. However, manual assessments, performed by users or their personal health professionals, are laborious and unscalable, while automatic machine learning methods often neglect an individual user's unique needs. Recent advances in Large Language Models (LLMs) enable novel approaches to this problem, balancing personalization with scalability to enable more adaptive and context-aware assessments of accessibility. We present Accessibility Scout, an LLM-based accessibility scanning system that identifies accessibility concerns from photos of built environments. With use, Accessibility Scout becomes an increasingly capable "accessibility scout", tailoring accessibility scans to an individual's mobility level, preferences, and specific environmental interests through collaborative Human-AI assessments. We present findings from three studies: a formative study with six participants to inform the design of Accessibility Scout, a technical evaluation of 500 images of built environments, and a user study with 10 participants of varying mobility. Results from our technical evaluation and user study show that Accessibility Scout can generate personalized accessibility scans that extend beyond traditional ADA considerations. Finally, we conclude with a discussion on the implications of our work and future steps for building more scalable and personalized accessibility assessments of the physical world. 

---
# On LLM-Assisted Generation of Smart Contracts from Business Processes 

**Authors**: Fabian Stiehle, Hans Weytjens, Ingo Weber  

**Link**: [PDF](https://arxiv.org/pdf/2507.23087)  

**Abstract**: Large language models (LLMs) have changed the reality of how software is produced. Within the wider software engineering community, among many other purposes, they are explored for code generation use cases from different types of input. In this work, we present an exploratory study to investigate the use of LLMs for generating smart contract code from business process descriptions, an idea that has emerged in recent literature to overcome the limitations of traditional rule-based code generation approaches. However, current LLM-based work evaluates generated code on small samples, relying on manual inspection, or testing whether code compiles but ignoring correct execution. With this work, we introduce an automated evaluation framework and provide empirical data from larger data sets of process models. We test LLMs of different types and sizes in their capabilities of achieving important properties of process execution, including enforcing process flow, resource allocation, and data-based conditions. Our results show that LLM performance falls short of the perfect reliability required for smart contract development. We suggest future work to explore responsible LLM integrations in existing tools for code generation to ensure more reliable output. Our benchmarking framework can serve as a foundation for developing and evaluating such integrations. 

---
# SmartCourse: A Contextual AI-Powered Course Advising System for Undergraduates 

**Authors**: Yixuan Mi, Yiduo Yu, Yiyi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.22946)  

**Abstract**: We present SmartCourse, an integrated course management and AI-driven advising system for undergraduate students (specifically tailored to the Computer Science (CPS) major). SmartCourse addresses the limitations of traditional advising tools by integrating transcript and plan information for student-specific context. The system combines a command-line interface (CLI) and a Gradio web GUI for instructors and students, manages user accounts, course enrollment, grading, and four-year degree plans, and integrates a locally hosted large language model (via Ollama) for personalized course recommendations. It leverages transcript and major plan to offer contextual advice (e.g., prioritizing requirements or retakes). We evaluated the system on 25 representative advising queries and introduced custom metrics: PlanScore, PersonalScore, Lift, and Recall to assess recommendation quality across different context conditions. Experiments show that using full context yields substantially more relevant recommendations than context-omitted modes, confirming the necessity of transcript and plan information for personalized academic advising. SmartCourse thus demonstrates how transcript-aware AI can enhance academic planning. 

---
# Zero-Shot Document Understanding using Pseudo Table of Contents-Guided Retrieval-Augmented Generation 

**Authors**: Hyeon Seong Jeong, Sangwoo Jo, Byeong Hyun Yoon, Yoonseok Heo, Haedong Jeong, Taehoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.23217)  

**Abstract**: Understanding complex multimodal documents remains challenging due to their structural inconsistencies and limited training data availability. We introduce \textit{DocsRay}, a training-free document understanding system that integrates pseudo Table of Contents (TOC) generation with hierarchical Retrieval-Augmented Generation (RAG). Our approach leverages multimodal Large Language Models' (LLMs) native capabilities to seamlessly process documents containing diverse elements such as text, images, charts, and tables without requiring specialized models or additional training. DocsRay's framework synergistically combines three key techniques: (1) a semantic structuring module using prompt-based LLM interactions to generate a hierarchical pseudo-TOC, (2) zero-shot multimodal analysis that converts diverse document elements into unified, text-centric representations using the inherent capabilities of multimodal LLMs, and (3) an efficient two-stage hierarchical retrieval system that reduces retrieval complexity from $O(N)$ to $O(S + k_1 \cdot N_s)$. Evaluated on documents averaging 49.4 pages and 20,971 textual tokens, DocsRay reduced query latency from 3.89 to 2.12 seconds, achieving a 45% efficiency improvement. On the MMLongBench-Doc benchmark, DocsRay-Pro attains an accuracy of 64.7%, substantially surpassing previous state-of-the-art results. 

---
# How and Where to Translate? The Impact of Translation Strategies in Cross-lingual LLM Prompting 

**Authors**: Aman Gupta, Yingying Zhuang, Zhou Yu, Ziji Zhang, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.22923)  

**Abstract**: Despite advances in the multilingual capabilities of Large Language Models (LLMs), their performance varies substantially across different languages and tasks. In multilingual retrieval-augmented generation (RAG)-based systems, knowledge bases (KB) are often shared from high-resource languages (such as English) to low-resource ones, resulting in retrieved information from the KB being in a different language than the rest of the context. In such scenarios, two common practices are pre-translation to create a mono-lingual prompt and cross-lingual prompting for direct inference. However, the impact of these choices remains unclear. In this paper, we systematically evaluate the impact of different prompt translation strategies for classification tasks with RAG-enhanced LLMs in multilingual systems. Experimental results show that an optimized prompting strategy can significantly improve knowledge sharing across languages, therefore improve the performance on the downstream classification task. The findings advocate for a broader utilization of multilingual resource sharing and cross-lingual prompt optimization for non-English languages, especially the low-resource ones. 

---
# RecUserSim: A Realistic and Diverse User Simulator for Evaluating Conversational Recommender Systems 

**Authors**: Luyu Chen, Quanyu Dai, Zeyu Zhang, Xueyang Feng, Mingyu Zhang, Pengcheng Tang, Xu Chen, Yue Zhu, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.22897)  

**Abstract**: Conversational recommender systems (CRS) enhance user experience through multi-turn interactions, yet evaluating CRS remains challenging. User simulators can provide comprehensive evaluations through interactions with CRS, but building realistic and diverse simulators is difficult. While recent work leverages large language models (LLMs) to simulate user interactions, they still fall short in emulating individual real users across diverse scenarios and lack explicit rating mechanisms for quantitative evaluation. To address these gaps, we propose RecUserSim, an LLM agent-based user simulator with enhanced simulation realism and diversity while providing explicit scores. RecUserSim features several key modules: a profile module for defining realistic and diverse user personas, a memory module for tracking interaction history and discovering unknown preferences, and a core action module inspired by Bounded Rationality theory that enables nuanced decision-making while generating more fine-grained actions and personalized responses. To further enhance output control, a refinement module is designed to fine-tune final responses. Experiments demonstrate that RecUserSim generates diverse, controllable outputs and produces realistic, high-quality dialogues, even with smaller base LLMs. The ratings generated by RecUserSim show high consistency across different base LLMs, highlighting its effectiveness for CRS evaluation. 

---
# iLearnRobot: An Interactive Learning-Based Multi-Modal Robot with Continuous Improvement 

**Authors**: Kohou Wang, ZhaoXiang Liu, Lin Bai, Kun Fan, Xiang Liu, Huan Hu, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2507.22896)  

**Abstract**: It is crucial that robots' performance can be improved after deployment, as they are inherently likely to encounter novel scenarios never seen before. This paper presents an innovative solution: an interactive learning-based robot system powered by a Multi-modal Large Language Model(MLLM). A key feature of our system is its ability to learn from natural dialogues with non-expert users. We also propose chain of question to clarify the exact intent of the question before providing an answer and dual-modality retrieval modules to leverage these interaction events to avoid repeating same mistakes, ensuring a seamless user experience before model updates, which is in contrast to current mainstream MLLM-based robotic systems. Our system marks a novel approach in robotics by integrating interactive learning, paving the way for superior adaptability and performance in diverse environments. We demonstrate the effectiveness and improvement of our method through experiments, both quantitively and qualitatively. 

---
# Evaluating LLMs for Visualization Generation and Understanding 

**Authors**: Saadiq Rauf Khan, Vinit Chandak, Sougata Mukherjea  

**Link**: [PDF](https://arxiv.org/pdf/2507.22890)  

**Abstract**: Information Visualization has been utilized to gain insights from complex data. In recent times, Large Language models (LLMs) have performed very well in many tasks. In this paper, we showcase the capabilities of different popular LLMs to generate code for visualization based on simple prompts. We also analyze the power of LLMs to understand some common visualizations by answering questions. Our study shows that LLMs could generate code for some simpler visualizations such as bar and pie charts. Moreover, they could answer simple questions about visualizations. However, LLMs also have several limitations. For example, some of them had difficulty generating complex visualizations, such as violin plot. LLMs also made errors in answering some questions about visualizations, for example, identifying relationships between close boundaries and determining lengths of shapes. We believe that our insights can be used to improve both LLMs and Information Visualization systems. 

---
# OAEI-LLM-T: A TBox Benchmark Dataset for Understanding Large Language Model Hallucinations in Ontology Matching 

**Authors**: Zhangcheng Qiang, Kerry Taylor, Weiqing Wang, Jing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21813)  

**Abstract**: Hallucinations are often inevitable in downstream tasks using large language models (LLMs). To tackle the substantial challenge of addressing hallucinations for LLM-based ontology matching (OM) systems, we introduce a new benchmark dataset OAEI-LLM-T. The dataset evolves from seven TBox datasets in the Ontology Alignment Evaluation Initiative (OAEI), capturing hallucinations of ten different LLMs performing OM tasks. These OM-specific hallucinations are organised into two primary categories and six sub-categories. We showcase the usefulness of the dataset in constructing an LLM leaderboard for OM tasks and for fine-tuning LLMs used in OM tasks. 

---
