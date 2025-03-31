# Self-Evolving Multi-Agent Simulations for Realistic Clinical Interactions 

**Authors**: Mohammad Almansoori, Komal Kumar, Hisham Cholakkal  

**Link**: [PDF](https://arxiv.org/pdf/2503.22678)  

**Abstract**: In this work, we introduce MedAgentSim, an open-source simulated clinical environment with doctor, patient, and measurement agents designed to evaluate and enhance LLM performance in dynamic diagnostic settings. Unlike prior approaches, our framework requires doctor agents to actively engage with patients through multi-turn conversations, requesting relevant medical examinations (e.g., temperature, blood pressure, ECG) and imaging results (e.g., MRI, X-ray) from a measurement agent to mimic the real-world diagnostic process. Additionally, we incorporate self improvement mechanisms that allow models to iteratively refine their diagnostic strategies. We enhance LLM performance in our simulated setting by integrating multi-agent discussions, chain-of-thought reasoning, and experience-based knowledge retrieval, facilitating progressive learning as doctor agents interact with more patients. We also introduce an evaluation benchmark for assessing the LLM's ability to engage in dynamic, context-aware diagnostic interactions. While MedAgentSim is fully automated, it also supports a user-controlled mode, enabling human interaction with either the doctor or patient agent. Comprehensive evaluations in various simulated diagnostic scenarios demonstrate the effectiveness of our approach. Our code, simulation tool, and benchmark are available at \href{this https URL}. 

---
# Historical Ink: Exploring Large Language Models for Irony Detection in 19th-Century Spanish 

**Authors**: Kevin Cohen, Laura Manrique-Gómez, Rubén Manrique  

**Link**: [PDF](https://arxiv.org/pdf/2503.22585)  

**Abstract**: This study explores the use of large language models (LLMs) to enhance datasets and improve irony detection in 19th-century Latin American newspapers. Two strategies were employed to evaluate the efficacy of BERT and GPT-4o models in capturing the subtle nuances nature of irony, through both multi-class and binary classification tasks. First, we implemented dataset enhancements focused on enriching emotional and contextual cues; however, these showed limited impact on historical language analysis. The second strategy, a semi-automated annotation process, effectively addressed class imbalance and augmented the dataset with high-quality annotations. Despite the challenges posed by the complexity of irony, this work contributes to the advancement of sentiment analysis through two key contributions: introducing a new historical Spanish dataset tagged for sentiment analysis and irony detection, and proposing a semi-automated annotation methodology where human expertise is crucial for refining LLMs results, enriched by incorporating historical and cultural contexts as core features. 

---
# Beyond Vanilla Fine-Tuning: Leveraging Multistage, Multilingual, and Domain-Specific Methods for Low-Resource Machine Translation 

**Authors**: Sarubi Thillainathan, Songchen Yuan, En-Shiun Annie Lee, Sanath Jayasena, Surangika Ranathunga  

**Link**: [PDF](https://arxiv.org/pdf/2503.22582)  

**Abstract**: Fine-tuning multilingual sequence-to-sequence large language models (msLLMs) has shown promise in developing neural machine translation (NMT) systems for low-resource languages (LRLs). However, conventional single-stage fine-tuning methods struggle in extremely low-resource NMT settings, where training data is very limited. This paper contributes to artificial intelligence by proposing two approaches for adapting msLLMs in these challenging scenarios: (1) continual pre-training (CPT), where the msLLM is further trained with domain-specific monolingual data to compensate for the under-representation of LRLs, and (2) intermediate task transfer learning (ITTL), a method that fine-tunes the msLLM with both in-domain and out-of-domain parallel data to enhance its translation capabilities across various domains and tasks. As an application in engineering, these methods are implemented in NMT systems for Sinhala, Tamil, and English (six language pairs) in domain-specific, extremely low-resource settings (datasets containing fewer than 100,000 samples). Our experiments reveal that these approaches enhance translation performance by an average of +1.47 bilingual evaluation understudy (BLEU) score compared to the standard single-stage fine-tuning baseline across all translation directions. Additionally, a multi-model ensemble further improves performance by an additional BLEU score. 

---
# Bridging the Dimensional Chasm: Uncover Layer-wise Dimensional Reduction in Transformers through Token Correlation 

**Authors**: Zhuo-Yang Song, Zeyu Li, Qing-Hong Cao, Ming-xing Luo, Hua Xing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22547)  

**Abstract**: The geometric evolution of token representations in large language models (LLMs) presents a fundamental paradox: while human language inherently organizes semantic information in low-dimensional spaces ($\sim 10^1$ dimensions), modern LLMs employ high-dimensional embeddings ($\sim 10^3$ dimensions) processed through Transformer architectures. To resolve this paradox, this work bridges this conceptual gap by developing a geometric framework that tracks token dynamics across Transformers layers. Through layer-wise analysis of intrinsic dimensions across multiple architectures, we reveal an expansion-contraction pattern where tokens diffuse to a "working space" and then progressively project onto lower-dimensional submanifolds. Our finding implies a negative correlation between the working space dimension and parameter-sensitive performance of the LLMs, and indicates that effective models tend to compress tokens into approximately 10-dimensional submanifolds, closely resembling human semantic spaces. This work not only advances LLM interpretability by reframing Transformers layers as projectors that mediate between high-dimensional computation and low-dimensional semantics, but also provides practical tools for model diagnostics that do not rely on task-specific evaluations. 

---
# Exploiting Mixture-of-Experts Redundancy Unlocks Multimodal Generative Abilities 

**Authors**: Raman Dutt, Harleen Hanspal, Guoxuan Xia, Petru-Daniel Tudosiu, Alexander Black, Yongxin Yang, Steven McDonagh, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2503.22517)  

**Abstract**: In this work, we undertake the challenge of augmenting the existing generative capabilities of pre-trained text-only large language models (LLMs) with multi-modal generation capability while satisfying two core constraints: C1 preserving the preservation of original language generative capabilities with negligible performance degradation, and C2 adhering to a small parameter budget to learn the new modality, ensuring scalability and efficiency. In contrast to current approaches that add dedicated modules, thereby significantly increasing the parameter count, we propose a method that leverages the underutilized capacity inherent in deep models. Specifically, we exploit the parameter redundancy within Mixture-of-Experts (MoEs) as a source of additional capacity for learning a new modality, enabling better parameter efficiency (C1). Moreover, we preserve the original language generation capabilities by applying low-rank adaptation exclusively to the tokens of the new modality (C2). Furthermore, we introduce a novel parameter initialization scheme based on the Gromov-Wasserstein distance to improve convergence and training stability. Through an extensive analysis of the routing mechanism, we uncover the emergence of modality-specific pathways and decreased redundancy within the experts that can efficiently unlock multi-modal generative capabilities. Overall, our method can be seamlessly applied to a wide range of contemporary LLMs, providing a new pathway for transitioning from uni-modal to multi-modal architectures. 

---
# WorkTeam: Constructing Workflows from Natural Language with Multi-Agents 

**Authors**: Hanchao Liu, Rongjun Li, Weimin Xiong, Ziyu Zhou, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22473)  

**Abstract**: Workflows play a crucial role in enhancing enterprise efficiency by orchestrating complex processes with multiple tools or components. However, hand-crafted workflow construction requires expert knowledge, presenting significant technical barriers. Recent advancements in Large Language Models (LLMs) have improved the generation of workflows from natural language instructions (aka NL2Workflow), yet existing single LLM agent-based methods face performance degradation on complex tasks due to the need for specialized knowledge and the strain of task-switching. To tackle these challenges, we propose WorkTeam, a multi-agent NL2Workflow framework comprising a supervisor, orchestrator, and filler agent, each with distinct roles that collaboratively enhance the conversion process. As there are currently no publicly available NL2Workflow benchmarks, we also introduce the HW-NL2Workflow dataset, which includes 3,695 real-world business samples for training and evaluation. Experimental results show that our approach significantly increases the success rate of workflow construction, providing a novel and effective solution for enterprise NL2Workflow services. 

---
# Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey 

**Authors**: Shengyue Guan, Haoyi Xiong, Jindong Wang, Jiang Bian, Bin Zhu, Jian-guang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2503.22458)  

**Abstract**: This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the evaluation methodologies. It categorizes approaches into annotation-based evaluations, automated metrics, hybrid strategies that combine human assessments with quantitative measures, and self-judging methods utilizing LLMs. This framework not only captures traditional metrics derived from language understanding, such as BLEU and ROUGE scores, but also incorporates advanced techniques that reflect the dynamic, interactive nature of multi-turn dialogues. 

---
# Scaling Laws of Scientific Discovery with AI and Robot Scientists 

**Authors**: Pengsong Zhang, Heng Zhang, Huazhe Xu, Renjun Xu, Zhenting Wang, Cong Wang, Animesh Garg, Zhibin Li, Arash Ajoudani, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22444)  

**Abstract**: The rapid evolution of scientific inquiry highlights an urgent need for groundbreaking methodologies that transcend the limitations of traditional research. Conventional approaches, bogged down by manual processes and siloed expertise, struggle to keep pace with the demands of modern discovery. We envision an autonomous generalist scientist (AGS) system-a fusion of agentic AI and embodied robotics-that redefines the research lifecycle. This system promises to autonomously navigate physical and digital realms, weaving together insights from disparate disciplines with unprecedented efficiency. By embedding advanced AI and robot technologies into every phase-from hypothesis formulation to peer-ready manuscripts-AGS could slash the time and resources needed for scientific research in diverse field. We foresee a future where scientific discovery follows new scaling laws, driven by the proliferation and sophistication of such systems. As these autonomous agents and robots adapt to extreme environments and leverage a growing reservoir of knowledge, they could spark a paradigm shift, pushing the boundaries of what's possible and ushering in an era of relentless innovation. 

---
# Long-Tail Crisis in Nearest Neighbor Language Models 

**Authors**: Yuto Nishida, Makoto Morishita, Hiroyuki Deguchi, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2503.22426)  

**Abstract**: The $k$-nearest-neighbor language model ($k$NN-LM), one of the retrieval-augmented language models, improves the perplexity for given text by directly accessing a large datastore built from any text data during inference. A widely held hypothesis for the success of $k$NN-LM is that its explicit memory, i.e., the datastore, enhances predictions for long-tail phenomena. However, prior works have primarily shown its ability to retrieve long-tail contexts, leaving the model's performance remain underexplored in estimating the probabilities of long-tail target tokens during inference. In this paper, we investigate the behavior of $k$NN-LM on low-frequency tokens, examining prediction probability, retrieval accuracy, token distribution in the datastore, and approximation error of the product quantization. Our experimental results reveal that $k$NN-LM does not improve prediction performance for low-frequency tokens but mainly benefits high-frequency tokens regardless of long-tail contexts in the datastore. 

---
# Elite Political Discourse has Become More Toxic in Western Countries 

**Authors**: Petter Törnberg, Juliana Chueri  

**Link**: [PDF](https://arxiv.org/pdf/2503.22411)  

**Abstract**: Toxic and uncivil politics is widely seen as a growing threat to democratic values and governance, yet our understanding of the drivers and evolution of political incivility remains limited. Leveraging a novel dataset of nearly 18 million Twitter messages from parliamentarians in 17 countries over five years, this paper systematically investigates whether politics internationally is becoming more uncivil, and what are the determinants of political incivility. Our analysis reveals a marked increase in toxic discourse among political elites, and that it is associated to radical-right parties and parties in opposition. Toxicity diminished markedly during the early phase of the COVID-19 pandemic and, surprisingly, during election campaigns. Furthermore, our results indicate that posts relating to ``culture war'' topics, such as migration and LGBTQ+ rights, are substantially more toxic than debates focused on welfare or economic issues. These findings underscore a troubling shift in international democracies toward an erosion of constructive democratic dialogue. 

---
# Negation: A Pink Elephant in the Large Language Models' Room? 

**Authors**: Tereza Vrabcová, Marek Kadlčík, Petr Sojka, Michal Štefánik, Michal Spiegel  

**Link**: [PDF](https://arxiv.org/pdf/2503.22395)  

**Abstract**: Negations are key to determining sentence meaning, making them essential for logical reasoning. Despite their importance, negations pose a substantial challenge for large language models (LLMs) and remain underexplored.
We construct two multilingual natural language inference (NLI) datasets with \textit{paired} examples differing in negation. We investigate how model size and language impact its ability to handle negation correctly by evaluating popular LLMs.
Contrary to previous work, we show that increasing the model size consistently improves the models' ability to handle negations. Furthermore, we find that both the models' reasoning accuracy and robustness to negation are language-dependent and that the length and explicitness of the premise have a greater impact on robustness than language.
Our datasets can facilitate further research and improvements of language model reasoning in multilingual settings. 

---
# Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors 

**Authors**: Zhiyu Yang, Shuo Wang, Yukun Yan, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22388)  

**Abstract**: LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the this http URL is publicly available at this https URL. 

---
# Supposedly Equivalent Facts That Aren't? Entity Frequency in Pre-training Induces Asymmetry in LLMs 

**Authors**: Yuan He, Bailan He, Zifeng Ding, Alisia Lupidi, Yuqicheng Zhu, Shuo Chen, Caiqi Zhang, Jiaoyan Chen, Yunpu Ma, Volker Tresp, Ian Horrocks  

**Link**: [PDF](https://arxiv.org/pdf/2503.22362)  

**Abstract**: Understanding and mitigating hallucinations in Large Language Models (LLMs) is crucial for ensuring reliable content generation. While previous research has primarily focused on "when" LLMs hallucinate, our work explains "why" and directly links model behaviour to the pre-training data that forms their prior knowledge. Specifically, we demonstrate that an asymmetry exists in the recognition of logically equivalent facts, which can be attributed to frequency discrepancies of entities appearing as subjects versus objects. Given that most pre-training datasets are inaccessible, we leverage the fully open-source OLMo series by indexing its Dolma dataset to estimate entity frequencies. Using relational facts (represented as triples) from Wikidata5M, we construct probing datasets to isolate this effect. Our experiments reveal that facts with a high-frequency subject and a low-frequency object are better recognised than their inverse, despite their logical equivalence. The pattern reverses in low-to-high frequency settings, and no statistically significant asymmetry emerges when both entities are high-frequency. These findings highlight the influential role of pre-training data in shaping model predictions and provide insights for inferring the characteristics of pre-training data in closed or partially closed LLMs. 

---
# Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions 

**Authors**: Yubo Li, Yidi Miao, Xueying Ding, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2503.22353)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities across various tasks, but their deployment in high-stake domains requires consistent performance across multiple interaction rounds. This paper introduces a comprehensive framework for evaluating and improving LLM response consistency, making three key contributions. First, we propose a novel Position-Weighted Consistency (PWC) score that captures both the importance of early-stage stability and recovery patterns in multi-turn interactions. Second, we present a carefully curated benchmark dataset spanning diverse domains and difficulty levels, specifically designed to evaluate LLM consistency under various challenging follow-up scenarios. Third, we introduce Confidence-Aware Response Generation (CARG), a framework that significantly improves response stability by incorporating model confidence signals into the generation process. Empirical results demonstrate that CARG significantly improves response stability without sacrificing accuracy, underscoring its potential for reliable LLM deployment in critical applications. 

---
# SKDU at De-Factify 4.0: Natural Language Features for AI-Generated Text-Detection 

**Authors**: Shrikant Malviya, Pablo Arnau-González, Miguel Arevalillo-Herráez, Stamos Katsigiannis  

**Link**: [PDF](https://arxiv.org/pdf/2503.22338)  

**Abstract**: The rapid advancement of large language models (LLMs) has introduced new challenges in distinguishing human-written text from AI-generated content. In this work, we explored a pipelined approach for AI-generated text detection that includes a feature extraction step (i.e. prompt-based rewriting features inspired by RAIDAR and content-based features derived from the NELA toolkit) followed by a classification module. Comprehensive experiments were conducted on the Defactify4.0 dataset, evaluating two tasks: binary classification to differentiate human-written and AI-generated text, and multi-class classification to identify the specific generative model used to generate the input text. Our findings reveal that NELA features significantly outperform RAIDAR features in both tasks, demonstrating their ability to capture nuanced linguistic, stylistic, and content-based differences. Combining RAIDAR and NELA features provided minimal improvement, highlighting the redundancy introduced by less discriminative features. Among the classifiers tested, XGBoost emerged as the most effective, leveraging the rich feature sets to achieve high accuracy and generalisation. 

---
# A Refined Analysis of Massive Activations in LLMs 

**Authors**: Louis Owen, Nilabhra Roy Chowdhury, Abhay Kumar, Fabian Güra  

**Link**: [PDF](https://arxiv.org/pdf/2503.22329)  

**Abstract**: Motivated in part by their relevance for low-precision training and quantization, massive activations in large language models (LLMs) have recently emerged as a topic of interest. However, existing analyses are limited in scope, and generalizability across architectures is unclear. This paper helps address some of these gaps by conducting an analysis of massive activations across a broad range of LLMs, including both GLU-based and non-GLU-based architectures. Our findings challenge several prior assumptions, most importantly: (1) not all massive activations are detrimental, i.e. suppressing them does not lead to an explosion of perplexity or a collapse in downstream task performance; (2) proposed mitigation strategies such as Attention KV bias are model-specific and ineffective in certain cases. We consequently investigate novel hybrid mitigation strategies; in particular pairing Target Variance Rescaling (TVR) with Attention KV bias or Dynamic Tanh (DyT) successfully balances the mitigation of massive activations with preserved downstream model performance in the scenarios we investigated. Our code is available at: this https URL. 

---
# Preference-based Learning with Retrieval Augmented Generation for Conversational Question Answering 

**Authors**: Magdalena Kaiser, Gerhard Weikum  

**Link**: [PDF](https://arxiv.org/pdf/2503.22303)  

**Abstract**: Conversational Question Answering (ConvQA) involves multiple subtasks, i) to understand incomplete questions in their context, ii) to retrieve relevant information, and iii) to generate answers. This work presents PRAISE, a pipeline-based approach for ConvQA that trains LLM adapters for each of the three subtasks. As labeled training data for individual subtasks is unavailable in practice, PRAISE learns from its own generations using the final answering performance as feedback signal without human intervention and treats intermediate information, like relevant evidence, as weakly labeled data. We apply Direct Preference Optimization by contrasting successful and unsuccessful samples for each subtask. In our experiments, we show the effectiveness of this training paradigm: PRAISE shows improvements per subtask and achieves new state-of-the-art performance on a popular ConvQA benchmark, by gaining 15.5 percentage points increase in precision over baselines. 

---
# MultiClaimNet: A Massively Multilingual Dataset of Fact-Checked Claim Clusters 

**Authors**: Rrubaa Panchendrarajan, Rubén Míguez, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2503.22280)  

**Abstract**: In the context of fact-checking, claims are often repeated across various platforms and in different languages, which can benefit from a process that reduces this redundancy. While retrieving previously fact-checked claims has been investigated as a solution, the growing number of unverified claims and expanding size of fact-checked databases calls for alternative, more efficient solutions. A promising solution is to group claims that discuss the same underlying facts into clusters to improve claim retrieval and validation. However, research on claim clustering is hindered by the lack of suitable datasets. To bridge this gap, we introduce \textit{MultiClaimNet}, a collection of three multilingual claim cluster datasets containing claims in 86 languages across diverse topics. Claim clusters are formed automatically from claim-matching pairs with limited manual intervention. We leverage two existing claim-matching datasets to form the smaller datasets within \textit{MultiClaimNet}. To build the larger dataset, we propose and validate an approach involving retrieval of approximate nearest neighbors to form candidate claim pairs and an automated annotation of claim similarity using large language models. This larger dataset contains 85.3K fact-checked claims written in 78 languages. We further conduct extensive experiments using various clustering techniques and sentence embedding models to establish baseline performance. Our datasets and findings provide a strong foundation for scalable claim clustering, contributing to efficient fact-checking pipelines. 

---
# CFiCS: Graph-Based Classification of Common Factors and Microcounseling Skills 

**Authors**: Fabian Schmidt, Karin Hammerfald, Henrik Haaland Jahren, Vladimir Vlassov  

**Link**: [PDF](https://arxiv.org/pdf/2503.22277)  

**Abstract**: Common factors and microcounseling skills are critical to the effectiveness of psychotherapy. Understanding and measuring these elements provides valuable insights into therapeutic processes and outcomes. However, automatic identification of these change principles from textual data remains challenging due to the nuanced and context-dependent nature of therapeutic dialogue. This paper introduces CFiCS, a hierarchical classification framework integrating graph machine learning with pretrained contextual embeddings. We represent common factors, intervention concepts, and microcounseling skills as a heterogeneous graph, where textual information from ClinicalBERT enriches each node. This structure captures both the hierarchical relationships (e.g., skill-level nodes linking to broad factors) and the semantic properties of therapeutic concepts. By leveraging graph neural networks, CFiCS learns inductive node embeddings that generalize to unseen text samples lacking explicit connections. Our results demonstrate that integrating ClinicalBERT node features and graph structure significantly improves classification performance, especially in fine-grained skill prediction. CFiCS achieves substantial gains in both micro and macro F1 scores across all tasks compared to baselines, including random forests, BERT-based multi-task models, and graph-based methods. 

---
# EdgeInfinite: A Memory-Efficient Infinite-Context Transformer for Edge Devices 

**Authors**: Jiyu Chen, Shuang Peng, Daxiong Luo, Fan Yang, Renshou Wu, Fangyuan Li, Xiaoxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22196)  

**Abstract**: Transformer-based large language models (LLMs) encounter challenges in processing long sequences on edge devices due to the quadratic complexity of attention mechanisms and growing memory demands from Key-Value (KV) cache. Existing KV cache optimizations struggle with irreversible token eviction in long-output tasks, while alternative sequence modeling architectures prove costly to adopt within established Transformer infrastructure. We present EdgeInfinite, a memory-efficient solution for infinite contexts that integrates compressed memory into Transformer-based LLMs through a trainable memory-gating module. This approach maintains full compatibility with standard Transformer architectures, requiring fine-tuning only a small part of parameters, and enables selective activation of the memory-gating module for long and short context task routing. The experimental result shows that EdgeInfinite achieves comparable performance to baseline Transformer-based LLM on long context benchmarks while optimizing memory consumption and time to first token. 

---
# FRASE: Structured Representations for Generalizable SPARQL Query Generation 

**Authors**: Papa Abdou Karim Karou Diallo, Amal Zouaq  

**Link**: [PDF](https://arxiv.org/pdf/2503.22144)  

**Abstract**: Translating natural language questions into SPARQL queries enables Knowledge Base querying for factual and up-to-date responses. However, existing datasets for this task are predominantly template-based, leading models to learn superficial mappings between question and query templates rather than developing true generalization capabilities. As a result, models struggle when encountering naturally phrased, template-free questions. This paper introduces FRASE (FRAme-based Semantic Enhancement), a novel approach that leverages Frame Semantic Role Labeling (FSRL) to address this limitation. We also present LC-QuAD 3.0, a new dataset derived from LC-QuAD 2.0, in which each question is enriched using FRASE through frame detection and the mapping of frame-elements to their argument. We evaluate the impact of this approach through extensive experiments on recent large language models (LLMs) under different fine-tuning configurations. Our results demonstrate that integrating frame-based structured representations consistently improves SPARQL generation performance, particularly in challenging generalization scenarios when test questions feature unseen templates (unknown template splits) and when they are all naturally phrased (reformulated questions). 

---
# Beyond Single-Sentence Prompts: Upgrading Value Alignment Benchmarks with Dialogues and Stories 

**Authors**: Yazhou Zhang, Qimeng Liu, Qiuchi Li, Peng Zhang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22115)  

**Abstract**: Evaluating the value alignment of large language models (LLMs) has traditionally relied on single-sentence adversarial prompts, which directly probe models with ethically sensitive or controversial questions. However, with the rapid advancements in AI safety techniques, models have become increasingly adept at circumventing these straightforward tests, limiting their effectiveness in revealing underlying biases and ethical stances. To address this limitation, we propose an upgraded value alignment benchmark that moves beyond single-sentence prompts by incorporating multi-turn dialogues and narrative-based scenarios. This approach enhances the stealth and adversarial nature of the evaluation, making it more robust against superficial safeguards implemented in modern LLMs. We design and implement a dataset that includes conversational traps and ethically ambiguous storytelling, systematically assessing LLMs' responses in more nuanced and context-rich settings. Experimental results demonstrate that this enhanced methodology can effectively expose latent biases that remain undetected in traditional single-shot evaluations. Our findings highlight the necessity of contextual and dynamic testing for value alignment in LLMs, paving the way for more sophisticated and realistic assessments of AI ethics and safety. 

---
# Leveraging LLMs for Predicting Unknown Diagnoses from Clinical Notes 

**Authors**: Dina Albassam, Adam Cross, Chengxiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2503.22092)  

**Abstract**: Electronic Health Records (EHRs) often lack explicit links between medications and diagnoses, making clinical decision-making and research more difficult. Even when links exist, diagnosis lists may be incomplete, especially during early patient visits. Discharge summaries tend to provide more complete information, which can help infer accurate diagnoses, especially with the help of large language models (LLMs). This study investigates whether LLMs can predict implicitly mentioned diagnoses from clinical notes and link them to corresponding medications. We address two research questions: (1) Does majority voting across diverse LLM configurations outperform the best single configuration in diagnosis prediction? (2) How sensitive is majority voting accuracy to LLM hyperparameters such as temperature, top-p, and summary length? To evaluate, we created a new dataset of 240 expert-annotated medication-diagnosis pairs from 20 MIMIC-IV notes. Using GPT-3.5 Turbo, we ran 18 prompting configurations across short and long summary lengths, generating 8568 test cases. Results show that majority voting achieved 75 percent accuracy, outperforming the best single configuration at 66 percent. No single hyperparameter setting dominated, but combining deterministic, balanced, and exploratory strategies improved performance. Shorter summaries generally led to higher this http URL conclusion, ensemble-style majority voting with diverse LLM configurations improves diagnosis prediction in EHRs and offers a promising method to link medications and diagnoses in clinical texts. 

---
# Penrose Tiled Low-Rank Compression and Section-Wise Q&A Fine-Tuning: A General Framework for Domain-Specific Large Language Model Adaptation 

**Authors**: Chuan-Wei Kuo, Siyu Chen, Chenqi Yan, Yu Yang Fredrik Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22074)  

**Abstract**: Large language models (LLMs) hold great promise for specialized scientific domains such as materials science, yet adapting them efficiently and accurately to domain-specific knowledge remains challenging due to limited data and high knowledge density. We propose a two-stage framework that combines structured model compression with a scientific fine-tuning regimen to address this challenge. In the compression stage, we decompose the LLM's weight matrices into local low-rank "rank blocks" and arrange these blocks in a Penrose-like non-periodic tiling pattern. Each block is then compacted via spectral transformations (e.g., discrete cosine or Fourier transforms), and a Kullback-Leibler (KL) divergence-based alignment loss preserves the distributional similarity between the compressed model's representations and those of the original full model. In the adaptation stage, the compressed model is further tuned using a human-like scientific reading protocol: it processes technical materials science documents section by section, engaging in a structured question-and-answer routine for each section. This section-wise Q&A fine-tuning strategy extracts explicit reasoning traces and gradually injects domain knowledge, while minimizing catastrophic forgetting of the model's general language capabilities. By balancing efficient compression with targeted adaptation, our two-stage approach enables precise specialization of LLMs to high-value domains under data-scarce conditions. We present this principled yet exploratory pipeline and outline its potential for advancing materials science knowledge integration, laying the groundwork for comprehensive empirical evaluation in future work. 

---
# Non-Monotonic Attention-based Read/Write Policy Learning for Simultaneous Translation 

**Authors**: Zeeshan Ahmed, Frank Seide, Zhe Liu, Rastislav Rabatin, Jachym Kolar, Niko Moritz, Ruiming Xie, Simone Merello, Christian Fuegen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22051)  

**Abstract**: Simultaneous or streaming machine translation generates translation while reading the input stream. These systems face a quality/latency trade-off, aiming to achieve high translation quality similar to non-streaming models with minimal latency. We propose an approach that efficiently manages this trade-off. By enhancing a pretrained non-streaming model, which was trained with a seq2seq mechanism and represents the upper bound in quality, we convert it into a streaming model by utilizing the alignment between source and target tokens. This alignment is used to learn a read/write decision boundary for reliable translation generation with minimal input. During training, the model learns the decision boundary through a read/write policy module, employing supervised learning on the alignment points (pseudo labels). The read/write policy module, a small binary classification unit, can control the quality/latency trade-off during inference. Experimental results show that our model outperforms several strong baselines and narrows the gap with the non-streaming baseline model. 

---
# ThinkEdit: Interpretable Weight Editing to Mitigate Overly Short Thinking in Reasoning Models 

**Authors**: Chung-En Sun, Ge Yan, Tsui-Wei Weng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22048)  

**Abstract**: Recent studies have shown that Large Language Models (LLMs) augmented with chain-of-thought (CoT) reasoning demonstrate impressive problem-solving abilities. However, in this work, we identify a recurring issue where these models occasionally generate overly short reasoning, leading to degraded performance on even simple mathematical problems. Specifically, we investigate how reasoning length is embedded in the hidden representations of reasoning models and its impact on accuracy. Our analysis reveals that reasoning length is governed by a linear direction in the representation space, allowing us to induce overly short reasoning by steering the model along this direction. Building on this insight, we introduce ThinkEdit, a simple yet effective weight-editing approach to mitigate the issue of overly short reasoning. We first identify a small subset of attention heads (approximately 2%) that predominantly drive short reasoning behavior. We then edit the output projection weights of these heads to suppress the short reasoning direction. With changes to only 0.1% of the model's parameters, ThinkEdit effectively reduces overly short reasoning and yields notable accuracy gains for short reasoning outputs (+5.44%), along with an overall improvement across multiple math benchmarks (+2.43%). Our findings provide new mechanistic insights into how reasoning length is controlled within LLMs and highlight the potential of fine-grained model interventions to improve reasoning quality. Our code is available at this https URL 

---
# The Risks of Using Large Language Models for Text Annotation in Social Science Research 

**Authors**: Hao Lin, Yongjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22040)  

**Abstract**: Generative artificial intelligence (GenAI) or large language models (LLMs) have the potential to revolutionize computational social science, particularly in automated textual analysis. In this paper, we conduct a systematic evaluation of the promises and risks of using LLMs for diverse coding tasks, with social movement studies serving as a case example. We propose a framework for social scientists to incorporate LLMs into text annotation, either as the primary coding decision-maker or as a coding assistant. This framework provides tools for researchers to develop the optimal prompt, and to examine and report the validity and reliability of LLMs as a methodological tool. Additionally, we discuss the associated epistemic risks related to validity, reliability, replicability, and transparency. We conclude with several practical guidelines for using LLMs in text annotation tasks, and how we can better communicate the epistemic risks in research. 

---
# Cognitive Prompts Using Guilford's Structure of Intellect Model 

**Authors**: Oliver Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2503.22036)  

**Abstract**: Large language models (LLMs) demonstrate strong language generation capabilities but often struggle with structured reasoning, leading to inconsistent or suboptimal problem-solving. To mitigate this limitation, Guilford's Structure of Intellect (SOI) model - a foundational framework from intelligence theory - is leveraged as the basis for cognitive prompt engineering. The SOI model categorizes cognitive operations such as pattern recognition, memory retrieval, and evaluation, offering a systematic approach to enhancing LLM reasoning and decision-making. This position paper presents a novel cognitive prompting approach for enforcing SOI-inspired reasoning for improving clarity, coherence, and adaptability in model responses. 

---
# Enhancing Domain-Specific Encoder Models with LLM-Generated Data: How to Leverage Ontologies, and How to Do Without Them 

**Authors**: Marc Brinner, Tarek Al Mustafa, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2503.22006)  

**Abstract**: We investigate the use of LLM-generated data for continual pretraining of encoder models in specialized domains with limited training data, using the scientific domain of invasion biology as a case study. To this end, we leverage domain-specific ontologies by enriching them with LLM-generated data and pretraining the encoder model as an ontology-informed embedding model for concept definitions. To evaluate the effectiveness of this method, we compile a benchmark specifically designed for assessing model performance in invasion biology. After demonstrating substantial improvements over standard LLM pretraining, we investigate the feasibility of applying the proposed approach to domains without comprehensive ontologies by substituting ontological concepts with concepts automatically extracted from a small corpus of scientific abstracts and establishing relationships between concepts through distributional statistics. Our results demonstrate that this automated approach achieves comparable performance using only a small set of scientific abstracts, resulting in a fully automated pipeline for enhancing domain-specific understanding of small encoder models that is especially suited for application in low-resource settings and achieves performance comparable to masked language modeling pretraining on much larger datasets. 

---
# Monte Carlo Sampling for Analyzing In-Context Examples 

**Authors**: Stephanie Schoch, Yangfeng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2503.22002)  

**Abstract**: Prior works have shown that in-context learning is brittle to presentation factors such as the order, number, and choice of selected examples. However, ablation-based guidance on selecting the number of examples may ignore the interplay between different presentation factors. In this work we develop a Monte Carlo sampling-based method to study the impact of number of examples while explicitly accounting for effects from order and selected examples. We find that previous guidance on how many in-context examples to select does not always generalize across different sets of selected examples and orderings, and whether one-shot settings outperform zero-shot settings is highly dependent on the selected example. Additionally, inspired by data valuation, we apply our sampling method to in-context example selection to select examples that perform well across different orderings. We find a negative result, that while performance is robust to ordering and number of examples, there is an unexpected performance degradation compared to random sampling. 

---
# Cluster automata 

**Authors**: András Kornai  

**Link**: [PDF](https://arxiv.org/pdf/2503.22000)  

**Abstract**: We introduce a new class of clustered Moore automata (CMA), investigate their temporal behavior, and describe some applications. 

---
# Entropy-Aware Branching for Improved Mathematical Reasoning 

**Authors**: Xianzhi Li, Ethan Callanan, Xiaodan Zhu, Mathieu Sibue, Antony Papadimitriou, Mahmoud Mahfouz, Zhiqiang Ma, Xiaomo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21961)  

**Abstract**: While Large Language Models (LLMs) are effectively aligned through extensive pre-training and fine-tuning, they still struggle with varying levels of uncertainty during token generation. In our investigation of mathematical reasoning, we observe that errors are more likely to arise at tokens exhibiting high entropy and variance of entropy in the model's output distribution. Based on the observation, we propose a novel approach that dynamically branches the generation process on demand instead of defaulting to the single most probable token. By exploring in parallel multiple branches stemming from high probability tokens of critical decision points, the model can discover diverse reasoning paths that might otherwise be missed. We further harness external feedback from larger models to rank and select the most coherent and accurate reasoning branch. Our experimental results on mathematical word problems and calculation questions show that this branching strategy boosts the reasoning capabilities of small LLMs up to 4.6% compared to conventional argmax decoding. 

---
# Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad 

**Authors**: Ivo Petrov, Jasper Dekoninck, Lyuben Baltadzhiev, Maria Drencheva, Kristian Minchev, Mislav Balunović, Nikola Jovanović, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2503.21934)  

**Abstract**: Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, o3-mini, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce the first comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly, achieving less than 5% on average. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities. 

---
# Local Normalization Distortion and the Thermodynamic Formalism of Decoding Strategies for Large Language Models 

**Authors**: Tom Kempton, Stuart Burrell  

**Link**: [PDF](https://arxiv.org/pdf/2503.21929)  

**Abstract**: Advances in hardware and language model architecture have spurred a revolution in natural language generation. However, autoregressive models compute probability distributions over next-token choices, and sampling from these distributions, known as decoding, has received significantly less attention than other design choices. Existing decoding strategies are largely based on heuristics, resulting in methods that are hard to apply or improve in a principled manner. We develop the theory of decoding strategies for language models by expressing popular decoding algorithms as equilibrium states in the language of ergodic theory and stating the functions they optimize. Using this, we analyze the effect of the local normalization step of top-k, nucleus, and temperature sampling, used to make probabilities sum to one. We argue that local normalization distortion is a fundamental defect of decoding strategies and quantify the size of this distortion and its effect on mathematical proxies for the quality and diversity of generated text. Contrary to the prevailing explanation, we argue that the major cause of the under-performance of top-k sampling relative to nucleus sampling is local normalization distortion. This yields conclusions for the future design of decoding algorithms and the detection of machine-generated text. 

---
# Hybrid Emotion Recognition: Enhancing Customer Interactions Through Acoustic and Textual Analysis 

**Authors**: Sahan Hewage Wewelwala, T.G.D.K. Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2503.21927)  

**Abstract**: This research presents a hybrid emotion recognition system integrating advanced Deep Learning, Natural Language Processing (NLP), and Large Language Models (LLMs) to analyze audio and textual data for enhancing customer interactions in contact centers. By combining acoustic features with textual sentiment analysis, the system achieves nuanced emotion detection, addressing the limitations of traditional approaches in understanding complex emotional states. Leveraging LSTM and CNN models for audio analysis and DistilBERT for textual evaluation, the methodology accommodates linguistic and cultural variations while ensuring real-time processing. Rigorous testing on diverse datasets demonstrates the system's robustness and accuracy, highlighting its potential to transform customer service by enabling personalized, empathetic interactions and improving operational efficiency. This research establishes a foundation for more intelligent and human-centric digital communication, redefining customer service standards. 

---
# AutoPsyC: Automatic Recognition of Psychodynamic Conflicts from Semi-structured Interviews with Large Language Models 

**Authors**: Sayed Muddashir Hossain, Simon Ostermann, Patrick Gebhard, Cord Benecke, Josef van Genabith, Philipp Müller  

**Link**: [PDF](https://arxiv.org/pdf/2503.21911)  

**Abstract**: Psychodynamic conflicts are persistent, often unconscious themes that shape a person's behaviour and experiences. Accurate diagnosis of psychodynamic conflicts is crucial for effective patient treatment and is commonly done via long, manually scored semi-structured interviews. Existing automated solutions for psychiatric diagnosis tend to focus on the recognition of broad disorder categories such as depression, and it is unclear to what extent psychodynamic conflicts which even the patient themselves may not have conscious access to could be automatically recognised from conversation. In this paper, we propose AutoPsyC, the first method for recognising the presence and significance of psychodynamic conflicts from full-length Operationalized Psychodynamic Diagnostics (OPD) interviews using Large Language Models (LLMs). Our approach combines recent advances in parameter-efficient fine-tuning and Retrieval-Augmented Generation (RAG) with a summarisation strategy to effectively process entire 90 minute long conversations. In evaluations on a dataset of 141 diagnostic interviews we show that AutoPsyC consistently outperforms all baselines and ablation conditions on the recognition of four highly relevant psychodynamic conflicts. 

---
# JEEM: Vision-Language Understanding in Four Arabic Dialects 

**Authors**: Karima Kadaoui, Hanin Atwany, Hamdan Al-Ali, Abdelrahman Mohamed, Ali Mekky, Sergei Tilga, Natalia Fedorova, Ekaterina Artemova, Hanan Aldarmaki, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2503.21910)  

**Abstract**: We introduce JEEM, a benchmark designed to evaluate Vision-Language Models (VLMs) on visual understanding across four Arabic-speaking countries: Jordan, The Emirates, Egypt, and Morocco. JEEM includes the tasks of image captioning and visual question answering, and features culturally rich and regionally diverse content. This dataset aims to assess the ability of VLMs to generalize across dialects and accurately interpret cultural elements in visual contexts. In an evaluation of five prominent open-source Arabic VLMs and GPT-4V, we find that the Arabic VLMs consistently underperform, struggling with both visual understanding and dialect-specific generation. While GPT-4V ranks best in this comparison, the model's linguistic competence varies across dialects, and its visual understanding capabilities lag behind. This underscores the need for more inclusive models and the value of culturally-diverse evaluation paradigms. 

---
# RedditESS: A Mental Health Social Support Interaction Dataset -- Understanding Effective Social Support to Refine AI-Driven Support Tools 

**Authors**: Zeyad Alghamdi, Tharindu Kumarage, Garima Agrawal, Mansooreh Karami, Ibrahim Almuteb, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.21888)  

**Abstract**: Effective mental health support is crucial for alleviating psychological distress. While large language model (LLM)-based assistants have shown promise in mental health interventions, existing research often defines "effective" support primarily in terms of empathetic acknowledgments, overlooking other essential dimensions such as informational guidance, community validation, and tangible coping strategies. To address this limitation and better understand what constitutes effective support, we introduce RedditESS, a novel real-world dataset derived from Reddit posts, including supportive comments and original posters' follow-up responses. Grounded in established social science theories, we develop an ensemble labeling mechanism to annotate supportive comments as effective or not and perform qualitative assessments to ensure the reliability of the annotations. Additionally, we demonstrate the practical utility of RedditESS by using it to guide LLM alignment toward generating more context-sensitive and genuinely helpful supportive responses. By broadening the understanding of effective support, our study paves the way for advanced AI-driven mental health interventions. 

---
# MSPLoRA: A Multi-Scale Pyramid Low-Rank Adaptation for Efficient Model Fine-Tuning 

**Authors**: Jiancheng Zhao, Xingda Yu, Zhen Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21838)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) has become an essential approach for adapting large-scale pre-trained models while reducing computational costs. Among PEFT methods, LoRA significantly reduces trainable parameters by decomposing weight updates into low-rank matrices. However, traditional LoRA applies a fixed rank across all layers, failing to account for the varying complexity of hierarchical information, which leads to inefficient adaptation and redundancy. To address this, we propose MSPLoRA (Multi-Scale Pyramid LoRA), which introduces Global Shared LoRA, Mid-Level Shared LoRA, and Layer-Specific LoRA to capture global patterns, mid-level features, and fine-grained information, respectively. This hierarchical structure reduces inter-layer redundancy while maintaining strong adaptation capability. Experiments on various NLP tasks demonstrate that MSPLoRA achieves more efficient adaptation and better performance while significantly reducing the number of trainable parameters. Furthermore, additional analyses based on Singular Value Decomposition validate its information decoupling ability, highlighting MSPLoRA as a scalable and effective optimization strategy for parameter-efficient fine-tuning in large language models. Our code is available at this https URL. 

---
# Refining Time Series Anomaly Detectors using Large Language Models 

**Authors**: Alan Yang, Yulin Chen, Sean Lee, Venus Montes  

**Link**: [PDF](https://arxiv.org/pdf/2503.21833)  

**Abstract**: Time series anomaly detection (TSAD) is of widespread interest across many industries, including finance, healthcare, and manufacturing. Despite the development of numerous automatic methods for detecting anomalies, human oversight remains necessary to review and act upon detected anomalies, as well as verify their accuracy. We study the use of multimodal large language models (LLMs) to partially automate this process. We find that LLMs can effectively identify false alarms by integrating visual inspection of time series plots with text descriptions of the data-generating process. By leveraging the capabilities of LLMs, we aim to reduce the reliance on human effort required to maintain a TSAD system 

---
# Optimizing Safe and Aligned Language Generation: A Multi-Objective GRPO Approach 

**Authors**: Xuying Li, Zhuo Li, Yuji Kosuga, Victor Bian  

**Link**: [PDF](https://arxiv.org/pdf/2503.21819)  

**Abstract**: Aligning large language models (LLMs) with human values and safety constraints is challenging, especially when objectives like helpfulness, truthfulness, and avoidance of harm conflict. Reinforcement Learning from Human Feedback (RLHF) has achieved notable success in steering models, but is complex and can be unstable. Recent approaches such as Direct Preference Optimization (DPO) simplify preference-based fine-tuning but may introduce bias or trade-off certain objectives~\cite{dpo}. In this work, we propose a Group Relative Policy Optimization (GRPO) framework with a multi-label reward regression model to achieve safe and aligned language generation. The GRPO algorithm optimizes a policy by comparing groups of sampled responses, eliminating the need for a separate value critic and improving training efficiency~\cite{grpo}. We train a reward model to predict multiple alignment scores (e.g., safety, helpfulness, etc.), which are combined into a single reward signal. We provide a theoretical derivation for using this learned multi-aspect reward within GRPO and discuss its advantages and limitations. Empirically, our approach improves all the safety and quality metrics evaluated in language generation tasks on model scales (0.5B, 7B, and 14B parameters), demonstrating a robust balance of objectives. We compare GRPO to PPO-based RLHF and DPO, highlighting that GRPO achieves alignment with significantly lower computational cost and explicit multi-objective handling. \textbf{We will open-source all trained models at this https URL. 

---
# OAEI-LLM-T: A TBox Benchmark Dataset for Understanding LLM Hallucinations in Ontology Matching Systems 

**Authors**: Zhangcheng Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21813)  

**Abstract**: Hallucinations are inevitable in downstream tasks using large language models (LLMs). While addressing hallucinations becomes a substantial challenge for LLM-based ontology matching (OM) systems, we introduce a new benchmark dataset called OAEI-LLM-T. The dataset evolves from the TBox (i.e. schema-matching) datasets in the Ontology Alignment Evaluation Initiative (OAEI), capturing hallucinations of different LLMs performing OM tasks. These OM-specific hallucinations are carefully classified into two primary categories and six sub-categories. We showcase the usefulness of the dataset in constructing the LLM leaderboard and fine-tuning foundational LLMs for LLM-based OM systems. 

---
# Large Language Models Meet Contrastive Learning: Zero-Shot Emotion Recognition Across Languages 

**Authors**: Heqing Zou, Fengmao Lv, Desheng Zheng, Eng Siong Chng, Deepu Rajan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21806)  

**Abstract**: Multilingual speech emotion recognition aims to estimate a speaker's emotional state using a contactless method across different languages. However, variability in voice characteristics and linguistic diversity poses significant challenges for zero-shot speech emotion recognition, especially with multilingual datasets. In this paper, we propose leveraging contrastive learning to refine multilingual speech features and extend large language models for zero-shot multilingual speech emotion estimation. Specifically, we employ a novel two-stage training framework to align speech signals with linguistic features in the emotional space, capturing both emotion-aware and language-agnostic speech representations. To advance research in this field, we introduce a large-scale synthetic multilingual speech emotion dataset, M5SER. Our experiments demonstrate the effectiveness of the proposed method in both speech emotion recognition and zero-shot multilingual speech emotion recognition, including previously unseen datasets and languages. 

---
# ImF: Implicit Fingerprint for Large Language Models 

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan  

**Link**: [PDF](https://arxiv.org/pdf/2503.21805)  

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership. 

---
# ELM: Ensemble of Language Models for Predicting Tumor Group from Pathology Reports 

**Authors**: Lovedeep Gondara, Jonathan Simkin, Shebnum Devji, Gregory Arbour, Raymond Ng  

**Link**: [PDF](https://arxiv.org/pdf/2503.21800)  

**Abstract**: Population-based cancer registries (PBCRs) face a significant bottleneck in manually extracting data from unstructured pathology reports, a process crucial for tasks like tumor group assignment, which can consume 900 person-hours for approximately 100,000 reports. To address this, we introduce ELM (Ensemble of Language Models), a novel ensemble-based approach leveraging both small language models (SLMs) and large language models (LLMs). ELM utilizes six fine-tuned SLMs, where three SLMs use the top part of the pathology report and three SLMs use the bottom part. This is done to maximize report coverage. ELM requires five-out-of-six agreement for a tumor group classification. Disagreements are arbitrated by an LLM with a carefully curated prompt. Our evaluation across nineteen tumor groups demonstrates ELM achieves an average precision and recall of 0.94, outperforming single-model and ensemble-without-LLM approaches. Deployed at the British Columbia Cancer Registry, ELM demonstrates how LLMs can be successfully applied in a PBCR setting to achieve state-of-the-art results and significantly enhance operational efficiencies, saving hundreds of person-hours annually. 

---
# Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation 

**Authors**: Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Wu Jian, Yuning Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22675)  

**Abstract**: Sequential Recommendation (SeqRec) aims to predict the next item by capturing sequential patterns from users' historical interactions, playing a crucial role in many real-world recommender systems. However, existing approaches predominantly adopt a direct forward computation paradigm, where the final hidden state of the sequence encoder serves as the user representation. We argue that this inference paradigm, due to its limited computational depth, struggles to model the complex evolving nature of user preferences and lacks a nuanced understanding of long-tail items, leading to suboptimal performance. To address this issue, we propose \textbf{ReaRec}, the first inference-time computing framework for recommender systems, which enhances user representations through implicit multi-step reasoning. Specifically, ReaRec autoregressively feeds the sequence's last hidden state into the sequential recommender while incorporating special reasoning position embeddings to decouple the original item encoding space from the multi-step reasoning space. Moreover, we introduce two lightweight reasoning-based learning methods, Ensemble Reasoning Learning (ERL) and Progressive Reasoning Learning (PRL), to further effectively exploit ReaRec's reasoning potential. Extensive experiments on five public real-world datasets and different SeqRec architectures demonstrate the generality and effectiveness of our proposed ReaRec. Remarkably, post-hoc analyses reveal that ReaRec significantly elevates the performance ceiling of multiple sequential recommendation backbones by approximately 30\%-50\%. Thus, we believe this work can open a new and promising avenue for future research in inference-time computing for sequential recommendation. 

---
# QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks? 

**Authors**: Belinda Z. Li, Been Kim, Zi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22674)  

**Abstract**: Recently, a large amount of work has focused on improving large language models' (LLMs') performance on reasoning benchmarks such as math and logic. However, past work has largely assumed that tasks are well-defined. In the real world, queries to LLMs are often underspecified, only solvable through acquiring missing information. We formalize this as a constraint satisfaction problem (CSP) with missing variable assignments. Using a special case of this formalism where only one necessary variable assignment is missing, we can rigorously evaluate an LLM's ability to identify the minimal necessary question to ask and quantify axes of difficulty levels for each problem. We present QuestBench, a set of underspecified reasoning tasks solvable by asking at most one question, which includes: (1) Logic-Q: Logical reasoning tasks with one missing proposition, (2) Planning-Q: PDDL planning problems with initial states that are partially-observed, (3) GSM-Q: Human-annotated grade school math problems with one missing variable assignment, and (4) GSME-Q: a version of GSM-Q where word problems are translated into equations by human annotators. The LLM is tasked with selecting the correct clarification question(s) from a list of options. While state-of-the-art models excel at GSM-Q and GSME-Q, their accuracy is only 40-50% on Logic-Q and Planning-Q. Analysis demonstrates that the ability to solve well-specified reasoning problems may not be sufficient for success on our benchmark: models have difficulty identifying the right question to ask, even when they can solve the fully specified version of the problem. Furthermore, in the Planning-Q domain, LLMs tend not to hedge, even when explicitly presented with the option to predict ``not sure.'' This highlights the need for deeper investigation into models' information acquisition capabilities. 

---
# ActionStudio: A Lightweight Framework for Data and Training of Action Models 

**Authors**: Jianguo Zhang, Thai Hoang, Ming Zhu, Zuxin Liu, Shiyu Wang, Tulika Awalgaonkar, Akshara Prabhakar, Haolin Chen, Weiran Yao, Zhiwei Liu, Juntao Tan, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22673)  

**Abstract**: Action models are essential for enabling autonomous agents to perform complex tasks. However, training large action models remains challenging due to the diversity of agent environments and the complexity of agentic data. Despite growing interest, existing infrastructure provides limited support for scalable, agent-specific fine-tuning. We present ActionStudio, a lightweight and extensible data and training framework designed for action models. ActionStudio unifies heterogeneous agent trajectories through a standardized format, supports diverse training paradigms including LoRA, full fine-tuning, and distributed setups, and integrates robust preprocessing and verification tools. We validate its effectiveness across both public and realistic industry benchmarks, demonstrating strong performance and practical scalability. We open-sourced code and data at this https URL to facilitate research in the community. 

---
# Evaluating Multimodal Language Models as Visual Assistants for Visually Impaired Users 

**Authors**: Antonia Karamolegkou, Malvina Nikandrou, Georgios Pantazopoulos, Danae Sanchez Villegas, Phillip Rust, Ruchira Dhar, Daniel Hershcovich, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2503.22610)  

**Abstract**: This paper explores the effectiveness of Multimodal Large Language models (MLLMs) as assistive technologies for visually impaired individuals. We conduct a user survey to identify adoption patterns and key challenges users face with such technologies. Despite a high adoption rate of these models, our findings highlight concerns related to contextual understanding, cultural sensitivity, and complex scene understanding, particularly for individuals who may rely solely on them for visual interpretation. Informed by these results, we collate five user-centred tasks with image and video inputs, including a novel task on Optical Braille Recognition. Our systematic evaluation of twelve MLLMs reveals that further advancements are necessary to overcome limitations related to cultural context, multilingual support, Braille reading comprehension, assistive object recognition, and hallucinations. This work provides critical insights into the future direction of multimodal AI for accessibility, underscoring the need for more inclusive, robust, and trustworthy visual assistance technologies. 

---
# CoSIL: Software Issue Localization via LLM-Driven Code Repository Graph Searching 

**Authors**: Zhonghao Jiang, Xiaoxue Ren, Meng Yan, Wei Jiang, Yong Li, Zhongxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22424)  

**Abstract**: Large language models (LLMs) have significantly advanced autonomous software engineering, leading to a growing number of software engineering agents that assist developers in automatic program repair. Issue localization forms the basis for accurate patch generation. However, because of limitations caused by the context window length of LLMs, existing issue localization methods face challenges in balancing concise yet effective contexts and adequately comprehensive search spaces. In this paper, we introduce CoSIL, an LLM driven, simple yet powerful function level issue localization method without training or indexing. CoSIL reduces the search space through module call graphs, iteratively searches the function call graph to obtain relevant contexts, and uses context pruning to control the search direction and manage contexts effectively. Importantly, the call graph is dynamically constructed by the LLM during search, eliminating the need for pre-parsing. Experiment results demonstrate that CoSIL achieves a Top-1 localization success rate of 43 percent and 44.6 percent on SWE bench Lite and SWE bench Verified, respectively, using Qwen2.5 Coder 32B, outperforming existing methods by 8.6 to 98.2 percent. When CoSIL is applied to guide the patch generation stage, the resolved rate further improves by 9.3 to 31.5 percent. 

---
# EllieSQL: Cost-Efficient Text-to-SQL with Complexity-Aware Routing 

**Authors**: Yizhang Zhu, Runzhi Jiang, Boyan Li, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.22402)  

**Abstract**: Text-to-SQL automatically translates natural language queries to SQL, allowing non-technical users to retrieve data from databases without specialized SQL knowledge. Despite the success of advanced LLM-based Text-to-SQL approaches on leaderboards, their unsustainable computational costs--often overlooked--stand as the "elephant in the room" in current leaderboard-driven research, limiting their economic practicability for real-world deployment and widespread adoption. To tackle this, we exploratively propose EllieSQL, a complexity-aware routing framework that assigns queries to suitable SQL generation pipelines based on estimated complexity. We investigate multiple routers to direct simple queries to efficient approaches while reserving computationally intensive methods for complex cases. Drawing from economics, we introduce the Token Elasticity of Performance (TEP) metric, capturing cost-efficiency by quantifying the responsiveness of performance gains relative to token investment in SQL generation. Experiments show that compared to always using the most advanced methods in our study, EllieSQL with the Qwen2.5-0.5B-DPO router reduces token use by over 40% without compromising performance on Bird development set, achieving more than a 2x boost in TEP over non-routing approaches. This not only advances the pursuit of cost-efficient Text-to-SQL but also invites the community to weigh resource efficiency alongside performance, contributing to progress in sustainable Text-to-SQL. 

---
# Spend Your Budget Wisely: Towards an Intelligent Distribution of the Privacy Budget in Differentially Private Text Rewriting 

**Authors**: Stephen Meisenbacher, Chaeeun Joy Lee, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2503.22379)  

**Abstract**: The task of $\textit{Differentially Private Text Rewriting}$ is a class of text privatization techniques in which (sensitive) input textual documents are $\textit{rewritten}$ under Differential Privacy (DP) guarantees. The motivation behind such methods is to hide both explicit and implicit identifiers that could be contained in text, while still retaining the semantic meaning of the original text, thus preserving utility. Recent years have seen an uptick in research output in this field, offering a diverse array of word-, sentence-, and document-level DP rewriting methods. Common to these methods is the selection of a privacy budget (i.e., the $\varepsilon$ parameter), which governs the degree to which a text is privatized. One major limitation of previous works, stemming directly from the unique structure of language itself, is the lack of consideration of $\textit{where}$ the privacy budget should be allocated, as not all aspects of language, and therefore text, are equally sensitive or personal. In this work, we are the first to address this shortcoming, asking the question of how a given privacy budget can be intelligently and sensibly distributed amongst a target document. We construct and evaluate a toolkit of linguistics- and NLP-based methods used to allocate a privacy budget to constituent tokens in a text document. In a series of privacy and utility experiments, we empirically demonstrate that given the same privacy budget, intelligent distribution leads to higher privacy levels and more positive trade-offs than a naive distribution of $\varepsilon$. Our work highlights the intricacies of text privatization with DP, and furthermore, it calls for further work on finding more efficient ways to maximize the privatization benefits offered by DP in text rewriting. 

---
# Learning to Instruct for Visual Instruction Tuning 

**Authors**: Zhihan Zhou, Feng Hong, Jiaan Luo, Jiangchao Yao, Dongsheng Li, Bo Han, Ya Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22215)  

**Abstract**: We propose LIT, an advancement of visual instruction tuning (VIT). While VIT equips Multimodal LLMs (MLLMs) with promising multimodal capabilities, the current design choices for VIT often result in overfitting and shortcut learning, potentially degrading performance. This gap arises from an overemphasis on instruction-following abilities, while neglecting the proactive understanding of visual information. Inspired by this, LIT adopts a simple yet effective approach by incorporating the loss function into both the instruction and response sequences. It seamlessly expands the training data, and regularizes the MLLMs from overly relying on language priors. Based on this merit, LIT achieves a significant relative improvement of up to 9% on comprehensive multimodal benchmarks, requiring no additional training data and incurring negligible computational overhead. Surprisingly, LIT attains exceptional fundamental visual capabilities, yielding up to an 18% improvement in captioning performance, while simultaneously alleviating hallucination in MLLMs. 

---
# Convolutional optimization with convex kernel and power lift 

**Authors**: Zhipeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22135)  

**Abstract**: We focus on establishing the foundational paradigm of a novel optimization theory based on convolution with convex kernels. Our goal is to devise a morally deterministic model of locating the global optima of an arbitrary function, which is distinguished from most commonly used statistical models. Limited preliminary numerical results are provided to test the efficiency of some specific algorithms derived from our paradigm, which we hope to stimulate further practical interest. 

---
# REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation 

**Authors**: Puzhen Yuan, Angyuan Ma, Yunchao Yao, Huaxiu Yao, Masayoshi Tomizuka, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.22122)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in robotic planning, particularly for long-horizon tasks that require a holistic understanding of the environment for task decomposition. Existing methods typically rely on prior environmental knowledge or carefully designed task-specific prompts, making them struggle with dynamic scene changes or unexpected task conditions, e.g., a robot attempting to put a carrot in the microwave but finds the door was closed. Such challenges underscore two critical issues: adaptability and efficiency. To address them, in this work, we propose an adaptive multi-agent planning framework, termed REMAC, that enables efficient, scene-agnostic multi-robot long-horizon task planning and execution through continuous reflection and self-evolution. REMAC incorporates two key modules: a self-reflection module performing pre-condition and post-condition checks in the loop to evaluate progress and refine plans, and a self-evolvement module dynamically adapting plans based on scene-specific reasoning. It offers several appealing benefits: 1) Robots can initially explore and reason about the environment without complex prompt design. 2) Robots can keep reflecting on potential planning errors and adapting the plan based on task-specific insights. 3) After iterations, a robot can call another one to coordinate tasks in parallel, maximizing the task execution efficiency. To validate REMAC's effectiveness, we build a multi-agent environment for long-horizon robot manipulation and navigation based on RoboCasa, featuring 4 task categories with 27 task styles and 50+ different objects. Based on it, we further benchmark state-of-the-art reasoning models, including DeepSeek-R1, o3-mini, QwQ, and Grok3, demonstrating REMAC's superiority by boosting average success rates by 40% and execution efficiency by 52.7% over the single robot baseline. 

---
# Debate-Driven Multi-Agent LLMs for Phishing Email Detection 

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22038)  

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies. 

---
# Socially Constructed Treatment Plans: Analyzing Online Peer Interactions to Understand How Patients Navigate Complex Medical Conditions 

**Authors**: Madhusudan Basak, Omar Sharif, Jessica Hulsey, Elizabeth C. Saunders, Daisy J. Goodman, Luke J. Archibald, Sarah M. Preum  

**Link**: [PDF](https://arxiv.org/pdf/2503.21986)  

**Abstract**: When faced with complex and uncertain medical conditions (e.g., cancer, mental health conditions, recovery from substance dependency), millions of patients seek online peer support. In this study, we leverage content analysis of online discourse and ethnographic studies with clinicians and patient representatives to characterize how treatment plans for complex conditions are "socially constructed." Specifically, we ground online conversation on medication-assisted recovery treatment to medication guidelines and subsequently surface when and why people deviate from the clinical guidelines. We characterize the implications and effectiveness of socially constructed treatment plans through in-depth interviews with clinical experts. Finally, given the enthusiasm around AI-powered solutions for patient communication, we investigate whether and how socially constructed treatment-related knowledge is reflected in a state-of-the-art large language model (LLM). Leveraging a novel mixed-method approach, this study highlights critical research directions for patient-centered communication in online health communities. 

---
# OntoAligner: A Comprehensive Modular and Robust Python Toolkit for Ontology Alignment 

**Authors**: Hamed Babaei Giglou, Jennifer D'Souza, Oliver Karras, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2503.21902)  

**Abstract**: Ontology Alignment (OA) is fundamental for achieving semantic interoperability across diverse knowledge systems. We present OntoAligner, a comprehensive, modular, and robust Python toolkit for ontology alignment, designed to address current limitations with existing tools faced by practitioners. Existing tools are limited in scalability, modularity, and ease of integration with recent AI advances. OntoAligner provides a flexible architecture integrating existing lightweight OA techniques such as fuzzy matching but goes beyond by supporting contemporary methods with retrieval-augmented generation and large language models for OA. The framework prioritizes extensibility, enabling researchers to integrate custom alignment algorithms and datasets. This paper details the design principles, architecture, and implementation of the OntoAligner, demonstrating its utility through benchmarks on standard OA tasks. Our evaluation highlights OntoAligner's ability to handle large-scale ontologies efficiently with few lines of code while delivering high alignment quality. By making OntoAligner open-source, we aim to provide a resource that fosters innovation and collaboration within the OA community, empowering researchers and practitioners with a toolkit for reproducible OA research and real-world applications. 

---
# Taxonomy Inference for Tabular Data Using Large Language Models 

**Authors**: Zhenyu Wu, Jiaoyan Chen, Norman W. Paton  

**Link**: [PDF](https://arxiv.org/pdf/2503.21810)  

**Abstract**: Taxonomy inference for tabular data is a critical task of schema inference, aiming at discovering entity types (i.e., concepts) of the tables and building their hierarchy. It can play an important role in data management, data exploration, ontology learning, and many data-centric applications. Existing schema inference systems focus more on XML, JSON or RDF data, and often rely on lexical formats and structures of the data for calculating similarities, with limited exploitation of the semantics of the text across a table. Motivated by recent works on taxonomy completion and construction using Large Language Models (LLMs), this paper presents two LLM-based methods for taxonomy inference for tables: (i) EmTT which embeds columns by fine-tuning with contrastive learning encoder-alone LLMs like BERT and utilises clustering for hierarchy construction, and (ii) GeTT which generates table entity types and their hierarchy by iterative prompting using a decoder-alone LLM like GPT-4. Extensive evaluation on three real-world datasets with six metrics covering different aspects of the output taxonomies has demonstrated that EmTT and GeTT can both produce taxonomies with strong consistency relative to the Ground Truth. 

---
# Efficient Joint Prediction of Multiple Future Tokens 

**Authors**: Kwangjun Ahn, Alex Lamb, John Langford  

**Link**: [PDF](https://arxiv.org/pdf/2503.21801)  

**Abstract**: In this short report, we introduce joint multi-token prediction (JTP), a lightweight modification of standard next-token prediction designed to enrich hidden state representations by jointly predicting multiple future tokens. Unlike previous multi-token prediction approaches, JTP strategically employs teacher forcing of future-tokens through a carefully designed representation bottleneck, allowing the model to encode rich predictive information with minimal computational overhead during training. We show that the JTP approach achieves a short-horizon belief state representation, while popular alternatives for multi-token prediction fail to do so. We demonstrate the effectiveness of our method on the synthetic star graph navigation task from from Bachmann and Nagarajan [2024], highlighting a significant performance improvement over existing methods. This manuscript presents promising preliminary results intended to stimulate further research. 

---
# Leveraging Large Language Models for Automated Causal Loop Diagram Generation: Enhancing System Dynamics Modeling through Curated Prompting Techniques 

**Authors**: Ning-Yuan Georgia Liu, David R. Keith  

**Link**: [PDF](https://arxiv.org/pdf/2503.21798)  

**Abstract**: Transforming a dynamic hypothesis into a causal loop diagram (CLD) is crucial for System Dynamics Modelling. Extracting key variables and causal relationships from text to build a CLD is often challenging and time-consuming for novice modelers, limiting SD tool adoption. This paper introduces and tests a method for automating the translation of dynamic hypotheses into CLDs using large language models (LLMs) with curated prompting techniques. We first describe how LLMs work and how they can make the inferences needed to build CLDs using a standard digraph structure. Next, we develop a set of simple dynamic hypotheses and corresponding CLDs from leading SD textbooks. We then compare the four different combinations of prompting techniques, evaluating their performance against CLDs labeled by expert modelers. Results show that for simple model structures and using curated prompting techniques, LLMs can generate CLDs of a similar quality to expert-built ones, accelerating CLD creation. 

---
