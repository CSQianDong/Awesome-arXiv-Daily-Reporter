# Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent 

**Authors**: Lei Bai, Zongsheng Cao, Yang Chen, Zhiyao Cui, Shangheng Du, Yue Fan, Shiyang Feng, Zijie Guo, Haonan He, Liang He, Xiaohan He, Shuyue Hu, Yusong Hu, Songtao Huang, Yichen Jiang, Hao Li, Xin Li, Dahua Lin, Weihao Lin, Fenghua Ling, Dongrui Liu, Zhuo Liu, Runmin Ma, Chunjiang Mu, Haoyang Peng, Tianshuo Peng, Jinxin Shi, Luohe Shi, Boyuan Sun, Zelin Tan, Shengji Tang, Qianyi Wang, Yiming Wu, Yi Xie, Xiangchao Yan, Jingqi Ye, Peng Ye, Fangchen Yu, Jiakang Yuan, Bihao Zhan, Bo Zhang, Chen Zhang, Shufei Zhang, Shuaiyu Zhang, Wenlong Zhang, Yiqun Zhang, Junpeng Zhao, Zhijie Zhong, Bowen Zhou, Yuhao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2606.30616)  

**Abstract**: We introduce Agents-A1, a 35B Mixture-of-Experts Agentic Model that reaches trillion-parameter-level performance by scaling the agent horizon. We investigate agent-horizon scaling from two perspectives: scaling long-horizon trajectories and scaling heterogeneous agent abilities. To support this goal, we build a long-horizon knowledge-action infrastructure that connects external knowledge, actions, observations, and verifier outcomes, producing agentic trajectories with an average length of 45K tokens. Based on this, we train Agents-A1 with a three-stage recipe. First, we perform full-domain supervised fine-tuning to align the base model with broad agentic behaviors. Second, we train domain-level teacher models to capture specialized expertise in each domain. Third, we propose a multi-teacher domain-routed on-policy distillation with salient vocabulary alignment to improve knowledge transfer efficiency across different domains, unifying six heterogeneous domains into one deployable student model. Agents-A1 achieves strong and broad performance for long-horizon agent benchmarks. Compared with 1T-parameter model such as Kimi-K2.6 and DeepSeek-V4-pro, Agents-A1 achieves leading results on SEAL-0 (56.4), IFBench (80.6), HiPhO (46.4), FrontierScience-Olympiad (79.0), and MolBench-Bind (56.8), and remains highly competitive on SciCode (44.3), HLE (47.6) and BrowseComp (75.5). We hope this work provides the community with a practical path for scaling the horizon using a 35B agent that can reach or match the performance of 1T models on long-horizon tasks. 

---
# Uncertainty-Aware Generation and Decision-Making Under Ambiguity 

**Authors**: Nico Daheim, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2606.30578)  

**Abstract**: With rapidly improving capabilities, Large Language Models (LLMs) are increasingly used in many complex real-world tasks. Beyond requiring in-depth knowledge and reasoning skills, many of these tasks exhibit a high degree of subjectivity and require that the outputs of the model can be trusted. While a lot of progress has been made to train better models, decision-making algorithms have received less attention. In this work, we present and evaluate various uncertainty-aware decision-making algorithms based on Bayesian decision theory and risk-averse decision making on the tasks of tutoring and automatic peer reviewing. Concretely, we take uncertainty over tutoring strategies and review scores into account when generating a tutor response or review and use conformal prediction to provide guarantees over strategy and score. We find empirically that these algorithms can improve the utility of the generations but need to be carefully implemented when ambiguity is high. For example, risk-averse rules can degrade performance by optimizing for generic outputs, while Bayesian methods tend to perform better. Our work uses techniques from decision theory to improve LLM-based decision-making and outlines open challenges for the community. 

---
# Morphing into Hybrid Attention Models 

**Authors**: Disen Lan, Jianbin Zheng, Yuxi Ren, Xin Xia, Xuanda Wang, Xuefeng Xiao, Xipeng Qiu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.30562)  

**Abstract**: Hybrid attention models improve long-context efficiency by retaining only a subset of full-attention layers and replacing the remaining layers with linear attention. However, the effectiveness of Transformer-to-hybrid conversion critically depends on which layers preserve full attention. Existing hybrid layer selection methods typically rely on heuristic strategies such as fixed placement patterns or layerwise scoring, implicitly treating layer importance as isolated and overlooking the interdependent layer effect under a global hybrid configuration. In this work, we formulate hybrid layer selection as a budget-constrained subset optimization problem. We further propose FlashMorph (Fast LAyer Selection for Hybrid MORPHing), an effective, efficient and scalable layer selection method for Transformer-to-hybrid conversion. FlashMorph first constructs a morphable model by equipping each full-attention layer with a converted linear-attention branch. It then freezes all model weights and jointly optimizes layerwise gates on synthetic long-context retrieval data, with a linearization regularization that encourages the model to rely on linear attention for efficiency. The learned gates are discretized under a preset full-attention budget to instantiate the hybrid architecture, followed by standard logits distillation and long-context finetuning. Extensive experiments show that FlashMorph discovers more effective hybrid configurations, preserves strong long-context recall and general benchmark performance while substantially reducing layer selection cost compared with existing layer selection methods, demonstrating its effectiveness, efficiency, and scalability. 

---
# Poller: Are LLMs Suitable for Evaluating the Poetry Understanding Task? 

**Authors**: Shanshan Wang, Derek F. Wong, Jingming Yao, Lidia S. Chao  

**Link**: [PDF](https://arxiv.org/pdf/2606.30556)  

**Abstract**: Traditional automatic evaluation methods have been shown to be unsuitable for modern Chinese poetry because of the distinct nature of this literary genre. Human evaluation remains reliable, but is expensive and not applicable to large-scale data. In this paper, we propose Poller (Poetry LLM Evaluator), a novel method leveraging large language models (LLMs) to evaluate the poetry understanding task. Specifically, our method requires LLMs to play the role of a poem's author with detailed information, thereby emulating human evaluation and judgment by adopting the poet's perspective. We conducted comprehensive experiments on multiple LLMs, evaluating the interpretations of poems across eight specialized dimensions. Experimental results demonstrate that our method effectively reduces the evaluation error between LLMs and humans. Especially for specific dimension evaluation, Poller-based LLMs achieve a 94.55% and 89.53% error reduction for rhetorical techniques and defamiliarization, respectively, compared to baseline methods. These performances are unattainable by conventional LLM evaluation methods. Experimental results from multiple LLMs across various dimensions validate the efficacy of our method. This work bridges the gap between automated efficiency and human expertise, establishing a foundation for automated evaluation in poetry-related tasks. 

---
# TRACE: Temporal Relationship-Aware Conversational Entrainment Detection in Dyadic Speech 

**Authors**: Sathvik Manikantan Napa Ugandhar, Hao Zhang, Alison Gunzler, Yuzhe Wang, Thomas Thebaud, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velázquez  

**Link**: [PDF](https://arxiv.org/pdf/2606.30543)  

**Abstract**: With the proliferation of speech AI agents, understanding emotional entrainment in conversational interaction has become increasingly important. Emotional entrainment is shaped by social relationships and conversational context, influencing affective coordination over time. We introduce DyadEE, a dataset for emotional entrainment detection in dyadic speech interactions, containing both emotionally entrained conversations and synthetic interactions where entrainment is disrupted through partner swapping and emotion resynthesis. We further propose TRACE, a window-level framework that models dyadic interaction as ordered sequences of acoustic embeddings derived from emotion fine-tuned Whisper representations, treating each sample as an interaction trace rather than pooled utterances. Experimental results on DyadEE show that incorporating conversational context and relationship information improves emotional entrainment detection, with TRACE achieving the best accuracy of 97.01%. 

---
# Regime-Aware Peer Specialization for Robust RAG under Heterogeneous Knowledge Conflicts 

**Authors**: Bo Wang, Heyan Huang, Yaolin Li, Yanghao Zhou, Jiahao Teng, Ziyi Yang, Ge Shi, Chong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.30518)  

**Abstract**: Retrieval-augmented generation (RAG) improves language models by grounding generation in external context. However, it can be fragile when the retrieved context conflicts with the model's parametric knowledge. Such conflicts span a reliability spectrum, ranging from reliable and partially reliable evidence to adversarial context. Existing remedies often handle such heterogeneous conflicts with regime-agnostic supervision, which can conflate incompatible learning signals across reliability regimes. To disentangle these signals, we propose RAPS-DA, a regime-aware peer specialization framework that addresses conflict at two complementary granularities. At the sample level, conflicts are divided into three regimes, including Grounding, Arbitration, and Resistance, with one same-scale peer specialist trained per regime from a shared base model. Each sample is then hard-routed to its regime-matched peer for on-policy reverse-KL supervision. At the token level, a dual-layer selector uses inter-teacher disagreement, student-teacher divergence, and student entropy to filter uninformative or unstable tokens, upweight confidently misaligned ones, and gradually focus supervision on high-conflict tokens as the student matures. Gains stem from specialization at a fixed model scale, not from a stronger teacher, and the peer specialists exist only during training, so the deployed student requires no regime labels or peer access. Experiments on five conflict scenarios and two out-of-distribution benchmarks show RAPS-DA surpasses all prompting, decoding, fine-tuning, RL, and single-teacher baselines. 

---
# SIMAX: A Scalable and Interpretable Framework for Multi-Fidelity and Annotated Clinician-Patient Dialogue Simulation 

**Authors**: Zhuhan Bao, Rui Yang, Bohao Yang, Zhiyi Liu, Sicheng Shu, Ruio Heerschap, Le Li, Doris Yang, Elisabeth Bond, Haoyuan Wang, Nicoleta Economou-Zavlanos, Joshua M. Biro, Matthew McDermott, Nan Liu, Anand Chowdhury, Kai Sun, Kathryn Pollak, Ed Hammond, Chuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2606.30491)  

**Abstract**: Background. The widespread deployment of ambient digital scribes is driving large-scale capture of clinician-patient dialogues. Human coding of clinical communication data remains costly, inconsistent, and difficult to scale, motivating AI-driven communication coding systems. However, evaluating these systems requires real-world dialogues and human-coded labels, both hard to obtain at scale.
Methods. We developed SIMAX (Scalable and Interpretable Framework for Multi-Fidelity and Annotated Clinician-Patient Dialogue Simulation), a framework for generating controlled clinical dialogue data with reference behavioral annotations. SIMAX generates clinician-patient dialogues from predefined clinical scenarios, personas and voice conditions, and target communication behaviors. Behaviors are controlled using two codebooks: the Global Codebook for overall communication quality and the WISER Codebook for specific countable behaviors. We evaluated SIMAX using automated and human quality assessments and an example communication coding system.
Results. SIMAX generated 3,388 simulated dialogues across three specialties, multiple visit stages, persona characteristics, and accent conditions. Automated assessment showed mean UTMOS and WV-MOS scores of 3.03 and 2.61, WER and CER of 0.07 and 0.05, and CLAP cosine similarity of 0.41, suggesting reasonable speech naturalness, high transcription fidelity, and positive text-audio correspondence. Human evaluation showed a median MOS of 4.67 and a median clinical realism score of 3.00. Downstream evaluation suggests that SIMAX can assess how a communication coding system responds to behavioral targets and reveal insufficient sensitivity in some dimensions.
Conclusions. SIMAX generates controlled and reproducible simulated clinician-patient dialogues, providing a data foundation for developing, validating, and refining communication coding systems. 

---
# Field Order Should Not Matter: Permutation-Invariant Embedding Model Fine-Tuning for Structured Metadata Retrieval 

**Authors**: Aivin V. Solatorio, Olivier Dupriez, Rafael Macalaba  

**Link**: [PDF](https://arxiv.org/pdf/2606.30473)  

**Abstract**: We study retrieval over catalogs of structured metadata, where each record is a small schema whose fields answer different kinds of query. Embedding a record with a text encoder first serializes its fields into a string, which forces a choice of field order. We show this choice, usually treated as an implementation detail, silently controls retrieval quality once the encoder is fine-tuned. A standard fine-tune loses 7.4 nDCG@10 points when the index is rebuilt under a different field order, because it reads absolute position instead of the field labels. We propose permutation-invariant fine-tuning ($\textbf{PI-FT}$), which serializes each record under a freshly sampled field order with random field dropout, so meaning binds to the labels rather than to position. The change is about two lines in the data loader; it costs negligible in-distribution accuracy and cuts the order-change penalty to 0.2 points. We study this in the discovery of development statistics, a catalog of nearly 10,000 indicators that should be searchable in many languages by a model small enough to self-host. As AI assistants and agents increasingly mediate access to public data and statistics, this retrieval step decides whether an answer is grounded in the right indicator or series, making discoverability a precondition for disseminating data through AI. Because usage logs cannot provide training signal for indicators no one has searched, we generate the queries instead. $\textbf{DevDataBench}$ is a fully LLM-generated benchmark of grounded, facet-targeted queries across 15 languages, covering every indicator for both training and evaluation. A fine-tuned 118M-parameter CPU encoder outperforms every zero-shot baseline, including $\texttt{text-embedding-3-large}$ (0.707 vs.\ 0.556 nDCG@10), with the largest gains in low-resource languages. We release the benchmark, pipeline, models, and a reusable PI-FT framework. 

---
# MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training 

**Authors**: Wenhan Ma, Jianyu Wei, Liang Zhao, Hailin Zhang, Bangjun Xiao, Lei Li, Qibin Yang, Bofei Gao, Yudong Wang, Rang Li, Jinhao Dong, Zhifang Sui, Fuli Luo  

**Link**: [PDF](https://arxiv.org/pdf/2606.30406)  

**Abstract**: Modern large language models (LLMs) rely on reinforcement learning during post-training to push specific capabilities, yet integrating multiple capabilities into one model remains hard. Existing methods, such as Off-Policy Finetune and Mix-RL, are either inefficient or lose performance. In this work, we propose Multi-teacher On-Policy Distillation (MOPD), a post-training paradigm for combining the capabilities of multiple domain RL teachers: we first run per-domain specialised RL to obtain a set of domain teachers, then distill these teachers into the student on its own rollouts. This eliminates exposure bias and provides a dense optimization signal. On Qwen3-30B-A3B, MOPD outperforms Mix-RL, Cascade RL, Off-Policy Finetune, and Param-Merge baselines, inheriting nearly all of each teacher's capability. MOPD also enables parallel, independent development of domain teachers, removing the cross-domain coupling typical of multi-domain post-training. MOPD has been deployed in the post-training of MiMo-V2-Flash, an industrial-scale frontier model, demonstrating its practical value for capability integration in frontier-scale LLMs. 

---
# OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL 

**Authors**: Karl El Hajal, Mathew Magimai.-Doss  

**Link**: [PDF](https://arxiv.org/pdf/2606.30356)  

**Abstract**: We propose Online Latent prediction with Invariant Views and rEconstruction (OLIVE), a self-supervised speech representation learning framework that jointly optimizes analysis and synthesis objectives. OLIVE combines view-augmented masked latent prediction with waveform reconstruction under a unified objective. Reconstruction constrains early encoder features to retain signal-level information, while masked latent prediction shapes later contextual representations toward invariance for robust downstream performance. We show that these objectives enable representations that support a broad range of tasks. In particular, OLIVE improves results on generation and speaker tasks, maintains competitive performance on recognition and semantic tasks, and improves waveform reconstruction. 

---
# REAR: Test-time Preference Realignment through Reward Decomposition 

**Authors**: Fuxiang Zhang, Pengcheng Wang, Chenran Li, Yi-Chen Li, Yuxin Chen, Lang Feng, Chenfeng Xu, Masayoshi Tomizuka, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2606.30339)  

**Abstract**: Aligning large language models (LLMs) with diverse user preferences is a critical yet challenging task. While post-training methods can adapt models to specific needs, they often require costly data curation and additional training. Test-time scaling (TTS) presents an efficient, training-free alternative, but its application has been largely limited to verifiable domains like mathematics and coding, where response correctness is easily judged. To extend TTS to preference alignment, we introduce a novel framework that models the task as a realignment problem, since the base model often fails to sufficiently align with the stated preference. Our key insight is to decompose the underlying reward function into two components: one related to the question and the other to preference information. This allows us to derive a REAlignment Reward (REAR) that selectively rescales the proportions of these two reward terms. We then show that REAR can be formulated as a linear combination of token-level policy log-probabilities, making it computationally efficient and easy to integrate with various TTS algorithms such as best-of-$N$ sampling and tree search. Experiments show that compared to other test-time baselines, REAR not only enables scalable test-time realignment for preference alignment tasks under diverse user requirements, but also generalizes to mathematical and visual tasks under appropriate preference settings. 

---
# DialogPII: A multilingual dataset of synthetic dialog transcripts to detect personal information 

**Authors**: Roland Roller, Vera Czehmann, Derya Erman, Luke Flanagan, Ibrahim Baroud, Frédéric Blain, Viviana Cotik, Eletta Giusto, Akhil Juneja, Mariana Neves, Maria Słowińska, Christine Hovhannisyan, Aaron Louis Eidt, Lisa Raithel, Sebastian Möller, Maija Poikela  

**Link**: [PDF](https://arxiv.org/pdf/2606.30312)  

**Abstract**: Conversational data collected in domains such as healthcare or social sciences is a valuable resource for research and automated analysis. However, responsible data sharing requires the detection and removal of personally identifiable and sensitive information to protect individual privacy. To support the development and evaluation of automatic de-identification systems, we present DialogPII, a multilingual dataset of synthetic dialogs and speech-derived transcripts for personal information detection. DialogPII covers eight interaction scenarios (emergency calls, medical anamnesis interviews, therapy sessions, insurance communication, customer support, clinical interviews regarding an AI-supported dashboard, police reports, and group therapy discussions), 19 entity types, and 11 languages (English, Arabic, Finnish, French, German, Hindi, Italian, Polish, Portuguese, Spanish, and Turkish). Dialogs were generated semi-automatically using large language models, manually curated for plausibility and diversity, and localized to country- and city-specific contexts. All dialogs were additionally converted to speech via text-to-speech synthesis, transcribed with Whisper, and annotated through automatic projection and manual correction, yielding aligned written and speech-derived resources across all languages. We further release baseline multilingual named entity recognition models and provide technical validation through inter-annotator agreement analysis, translation quality evaluation, annotation projection assessment, and benchmark experiments with transformer-based sequence labeling models. 

---
# Multi-Agentic System Leveraging Open-Source LLMs to Mitigate Disinformation Threats 

**Authors**: Sebastian Kula, Martin Tamajka  

**Link**: [PDF](https://arxiv.org/pdf/2606.30259)  

**Abstract**: In contemporary societies, the threat of disinformation has reached alarming levels, exacerbated by the proliferation of electronic communication, social media, and advancements in artificial intelligence. As a result, there is an urgent need to develop effective countermeasures to mitigate this menace. However, the sheer scale of the problem renders manual fact-checking and human-based verification inadequate, underscoring the necessity for automated methods to detect and debunk disinformation. This article proposes a novel approach based on a multi-agent system that emulates the decision-making processes of human annotators engaged in disinformation detection tasks. By incorporating a consensus mechanism, diversity in cognition and diversity in knowledge, and also hierarchical structure, inspired by human annotators' behavior, the proposed method achieves superior results compared to individual Large Language Models (LLMs), including GPT 4 and GPT 3.5. The system leverages open models (e.g., LLaMA, Kimi, Qwen, Deepseek and LLaMA-Nemotron) to ensure greater transparency. The evaluation of the proposed method encompasses datasets in languages with varying resource availability, including English (high-resource), Polish (medium-resource), Slovak (low-resource) and Bulgarian (low-resource). Experiments were conducted on tasks such as direct disinformation detection, identification of texts worthy of verification, and detection of texts containing verifiable factual claims. 

---
# Grounding LLM Reasoning under Incomplete Graph Evidence 

**Authors**: Jiaqi Li, Fanghui Song  

**Link**: [PDF](https://arxiv.org/pdf/2606.30247)  

**Abstract**: Knowledge graphs can guide large language models (LLMs) reasoning, but the graph seen by a system is usually a retrieved, linked, temporally scoped, and incomplete evidence state rather than a complete account of truth. We develop a theoretical perspective on grounding observable LLM trajectories under such incomplete graph this http URL evidence state induces entity anchors, typed relation residuals, path energies, and support regions, while the language model supplies a prior over candidate trajectories. We show that, under open-world incompleteness, no hard rule based only on the observed state can both reject every false unsupported trajectory and retain every true-but-unobserved this http URL then characterize soft grounding as a KL-regularized deformation of the LLM prior: finite slack preserves support for unsupported but non-contradicted trajectories, whereas hard conditioning appears as an infinite-penalty this http URL framework also yields stability bounds under evidence perturbations and clarifies the constraint regimes appropriate for GraphRAG, KGQA, graph agents, constrained decoding, and faithful generation. The claims are evidence-relative: KG compatibility is treated as declared support, not factual truth. 

---
# Comparing Human and Automatic Recognition of Dutch Dysarthric Continuous Speech: A Case Study 

**Authors**: Yuanyuan Zhang, Dimme de Groot, Jorge Martinez, Odette Scharenborg  

**Link**: [PDF](https://arxiv.org/pdf/2606.30237)  

**Abstract**: In our goal to develop personalised dysarthric speech recognition (DSR) models, this study compared the recognition performances of human listeners and those of three state-of-the-art, off-the-shelf ASR systems (Whisper-large-V3, Google Chirp 3, and Omnilingual) on the recognition of Dutch continuous read and spontaneous speech from a single speaker with severe dysarthria. Results showed that both humans listeners and the three off-the-shelf ASR systems exhibit word error rates (WER) exceeding 70% on average, indicating that DSR is highly challenging for both humans and ASR systems. Fine-tuning on the dysarthric speech significantly reduced WER. Although overall WERs are still quite high (>23%), the personalised DSR models outperformed the human listeners, and performance is getting closer to being useful for supporting day-to-day communication of dysarthric speakers. Future research should focus on improving personalized DSR on spontaneous speech and longer utterances in the case of read speech, with a specific focus on particular phonemes. 

---
# CaresAI at CT-DEB26: Detecting Dosing Errors In Clinical Trials Using Domain-Specific Transformer Embeddings and Classification Models 

**Authors**: Leon Hamnett, Favour Igwezeke, Joseph Itopa Abubakar, Mary Adetutu Adewunmi  

**Link**: [PDF](https://arxiv.org/pdf/2606.30236)  

**Abstract**: Medication errors, particularly dosing errors in clinical trials (CT), can lead to patient harm, adverse drug events and worse patient outcomes. Dosing errors are preventable, and early identification can improve trial integrity and mitigate subsequent clinical and financial burden. This study aims to detect dosing errors within CT protocols by evaluating text representations of trial information using transformer-based language models trained on biomedical corpora. CT textual data was encoded using several models, including ClinicalBERT, PubMedBERT, BioBERT, and MedCPT, and integrated with categorical features. These text embeddings were used as input to classical machine learning models and neural network architectures within an experimental framework. Performance was primarily assessed using ROC-AUC with respect to predicting dosage error. Under a logistic regression baseline, BioBERT consistently outperformed alternative encoders, achieving an ROC-AUC of 0.794, a 3.95% improvement over the ClinicalBERT baseline. Combining multiple embeddings did not yield improvements, indicating that domain alignment outweighs representational stacking. Gradient boosting models, support vector classifiers, logistic regression, and residual neural networks achieved the strongest performance for predicting dosage error, achieving ROC-AUCs: 0.821 to 0.853. Overall, the integration of domain-specific transformer embeddings with structured metadata enables discrimination of trials meeting a predefined elevated dosing error risk criterion, advancing safety monitoring and supporting informed regulatory decision-making. 

---
# Before Thinking, Learn to Decide: Proactive Routing for Efficient Visual Reasoning 

**Authors**: Yinan Zhou, Haokun Lin, Yichen Wu, Caifeng Shan, Zhenan Sun, Yuxin Chen, Teng Wang, Chen Ma, Li Zhu, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2606.30217)  

**Abstract**: Large multimodal models have achieved strong reasoning on complex visual tasks, but their inference efficiency is often restricted by long chains of thought. A promising solution is to pair a small draft model with a large target model, enabling cooperative inference employing a routing signal that adaptively routes queries to either the draft or target model based on their difficulties for optimal efficiency and accuracy. Yet, the remaining bottleneck is to establish a reliable query difficulty signal under multimodal settings. Existing approaches designed for language models either rely on post-hoc token probabilities, which fall short in multimodal scenarios, or depend on supervised fine-tuning, which is a data-sensitive strategy. Both paradigms perform routing only after a complete output, and ignore whether the target model can actually solve the routed instances. To address this, we propose PRP, a Proactive Routing Paradigm that enables early decision-making by jointly evaluating the competence of both the draft and target models. Our Draft Rating Learning (DRL) equips the draft model with an internal confidence estimator, while Joint Rating Learning (JRL) predicts how well the target model can handle a given query, thereby prioritizing the allocation of samples it excels at rather than the hardest ones. These ratings enable fine-grained, instance-level \textbf{Proactive Routing} and substantially accelerate inference without compromising overall performance. Extensive experiments across multiple multimodal reasoning benchmarks validate our effectiveness and efficiency. 

---
# Forewarned is Forearmed: When Non-Sequential Embedding Turns Into an Anomaly Detector 

**Authors**: Elys Allesiardo, Antoine Caubrière, Valentin Vielzeuf  

**Link**: [PDF](https://arxiv.org/pdf/2606.30196)  

**Abstract**: This paper offers an in-depth analysis of non-sequential multimodal sentence-level embeddings, with a particular focus on the SONAR model. We demonstrate that certain embedding dimensions are sensitive to perturbations and can serve as indicators of decoding anomalies. By leveraging the consistency between successive encoding and decoding, we successfully build an accurate detector. Additionally, we explore modifying specific dimensions of interest to attempt to correct them. This work underscores the importance of understanding and analyzing the embeddings themselves to enhance the reliability of multimodal representations. 

---
# DAIN: Dynamic Agent-Based Interaction Network for Efficient and Collaborative Multimodal Reasoning 

**Authors**: Xinxin Chen, Yuchen Li, Zihan Wang, Haoyu Zhang, Ruixin Liu, Mingyuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.30189)  

**Abstract**: Current multimodal fusion approaches, particularly those based on static Mixture-of-Experts (MoE) architectures, often struggle to provide the adaptive and efficient collaborative reasoning required by complex real-world applications. We introduce the Dynamic Agent-based Interaction Network (DAIN), which reconceptualizes multimodal fusion as a dynamic, multi-agent collaborative process. DAIN employs a context-aware Meta-Controller that dynamically schedules sparse activation of specialized interaction agents and orchestrates compressed inter-agent communication for consensus-building. The framework is guided by a multi-objective loss function that jointly optimizes task accuracy, agent specialization, and operational efficiency through sparse activation and communication regularization. Comprehensive evaluations across five diverse benchmarks -- ADNI, MIMIC-IV, MM-IMDB, CMU-MOSI, and ENRICO -- establish DAIN as a new state-of-the-art, delivering significant performance improvements including a 2.6\% accuracy gain on ADNI. Ablation studies verify the critical roles of both dynamic scheduling and agent communication. Furthermore, DAIN offers enhanced interpretability by exposing context-dependent agent roles and collaboration patterns while maintaining computational efficiency through sample-wise sparse agent activation. Our work demonstrates the promise of dynamic, agent-based paradigms for multimodal reasoning. 

---
# CORTEX: High-Quality Cross-Domain Organization of Web-Scale Corpora through Ontological Corpus Graph 

**Authors**: Chengtao Gan, Xiaoke Guo, Yushan Zhu, Zhaoyan Gong, Zhiqiang Liu, Songze Li, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30175)  

**Abstract**: The continuous evolution of large language models drives escalating demands on data scale and quality, and as different training stages impose increasingly tailored data requirements, systematic organization of high-quality corpora becomes indispensable. Existing corpus construction pipelines confine the resulting corpora to flat, undifferentiated document collections, universally lacking systematic knowledge organization. We present Cortex, to our knowledge the first framework that elevates web-scale corpus construction from flat document filtering to structured knowledge organization through an Ontological Corpus Graph (OCG), a three-layer heterogeneous structure unifying a quality-refined content layer, a hierarchical lightweight ontology layer via LLM-driven automated evolution, and a cross-domain alignment layer enabling inter-domain association at arbitrary taxonomic resolution. Comprehensive experiments confirm the effectiveness of Cortex. In particular, we leverage the OCG to synthesize CortexBench, a cross-domain search-and-reasoning benchmark whose evaluation across eight frontier LLMs validates the effectiveness of quality refinement, domain organization, and cross-domain data synthesis. We will publicly release the complete codebase, a 24.14B-token refined corpus with its OCG, and CortexBench. 

---
# Estimating Grammatical Gender Directions in Contextual Embeddings under Controlled and Natural Contexts 

**Authors**: Huanping Xiao, Yingji Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.30152)  

**Abstract**: Contextual language models conflate grammatical gender and social semantic bias in gendered languages such as Spanish. Existing gender debiasing approaches only operate on static word embeddings leaving contextual representations unexplored for this two dimensional gender disentanglement. To address the this issue, we make the first attempt to disentangle grammatical gender from semantic contamination for contextual embeddings. We construct both controlled templates and natural Wikipedia contexts to build balanced datasets of inanimate nouns, and design a framework equipped with centroid, Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA) gender direction estimators as well as contamination-aware weighting strategies. A set of dual-objective evaluation metrics is proposed to balance the suppression of grammatical gender leakage on inanimate nouns and the preservation of semantic gender distinctions for occupation terms. The results reveal that unweighted controlled contexts yield the purest grammatical gender direction, and the centroid estimator achieves better performance than discriminative baselines. 

---
# Information Dynamics of Language Communication 

**Authors**: Leonardo S. Goodall, Andrea I. Luppi, Pedro A. M. Mediano  

**Link**: [PDF](https://arxiv.org/pdf/2606.30096)  

**Abstract**: Quantifying how meaning propagates through communicative exchanges remains underdeveloped in computational linguistics. Here we introduce an information-theoretic framework that quantifies the directed flow of semantic content between interlocutors and decomposes multi-source contributions into redundant, unique, and synergistic components. Our approach leverages large language models as probabilistic estimators of natural language to compute two measures: semantic transfer entropy (STE), which captures directed predictive influence between speakers, and semantic partial information decomposition (SPID), which resolves how multiple sources jointly shape a target's language. Across four experiments we show that the framework detects reduced information flow in cognitively rigid dialogue, captures the dominant role of persuaders in shaping discourse, distinguishes high- from low-quality psychotherapy by the directionality of therapist-client information exchange, and reveals synergistic premise contributions in argumentative essays. This framework opens new avenues for studying information dynamics in digital discourse, pedagogical interactions, clinical dialogues, and any domain in which the structure of linguistic exchange is of research relevance. 

---
# Efficient Retrieval-Augmented Generation via Token Co-occurrence Graphs 

**Authors**: Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili  

**Link**: [PDF](https://arxiv.org/pdf/2606.30093)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by grounding the generation process on external knowledge. However, standard RAG approaches struggle with multi-hop reasoning. While recent graph-based RAG methods improve the retrieval of interconnected chunks, they often rely on computationally expensive and error-prone LLM-based extraction pipelines. To address these issues, we propose TIGRAG (Token-Induced GraphRAG), an efficient graph-augmented RAG framework based on a token co-occurrence Knowledge Graph. TIGRAG directly models topological relationships between tokens using sliding-window co-occurrence statistics, thus enabling scalable graph construction. During inference, it combines graph-based semantic expansion and neural reranking to retrieve interconnected evidence for multi-hop reasoning. Specifically, it introduces an iterative entity-driven retrieval strategy that progressively expands the query using bridging entities extracted from previously retrieved contexts. We evaluated TIGRAG on three widely adopted multi-hop Question Answering (QA) benchmarks. Experimental results demonstrated that our framework consistently outperforms dense retrieval and graph-based RAG methods in both retrieval and downstream QA tasks, while substantially reducing indexing time, inference latency, and prompt footprint. 

---
# Not-quite-human tastes: the stylized omnivorousness of LLM survey surrogates 

**Authors**: Xiangyu Ma, Mengmi Zhang, Shannon Ang, Minne Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.30085)  

**Abstract**: Large-language models have proven to be remarkable if inconsistent parrots of public attitudes and opinions. The extent to which LLMs are able to produce reasonable approximations of cultural taste remains an open empirical question that becomes more urgent by the day, with market research companies already offering provisional `synthetic' survey panels and the contamination of standard survey data from LLM-generated responses. In this study, we build on past work on silicon sampling by extending considerations of its algorithmic fidelity and alignment to the domain of cultural consumption. We use large-language models from OpenAI, Anthropic, and DeepSeek to each produce 277,470 (30x9249) silicon surrogates of survey respondents from the Survey of Public Participation in the Arts (SPPA). We find these silicon surrogates' tastes to be highly stylized facsimiles of human tastes. (1) Silicon samples have a systematic postive-bias for liking, resulting in inflated ecological estimates of tastes. The individual-level bias of silicon samples are not well-explained by the WEIRD-bias often discussed in the literature. (2) The complex relationality in real taste structures is completely lost among silicon samples. (3) Finally, very little of the known cultural alignment between tastes and social space are preserved. Silicon samples attenuate age-taste associations, resurrect anachronistic class-taste associations, caricaturize gender- and race-taste associations. 

---
# Little Brains, Big Feats: Exploring Compact Language Models 

**Authors**: Dari Baturova, Elena Bruches, Ivan Chernov, Roman Derunets, Arsenii Fomin, Andrey Kostin  

**Link**: [PDF](https://arxiv.org/pdf/2606.30062)  

**Abstract**: While large language models have been dominating the research landscape recently, small language models remain highly relevant across various domains; yet, they receive far less attention. In this study, we investigate how smaller language models perform during the generation stage within a Retrieval-Augmented Generation (RAG) system. To benchmark these models effectively, we utilised both open-source and proprietary datasets covering diverse subject areas and question types. Our findings demonstrate that a RAG system with small language models can be executed directly on-device without requiring any GPU hardware within a reasonable time. The experimental code and links to the supplementary materials can be accessed through the GitHub repository: this https URL. 

---
# Parametric Skills 

**Authors**: Xuan Zhao, Haonan He, Qingyu Yang, Minglei Li, Jingqi Ye, Zelin Tan, Bo Wan, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2606.30015)  

**Abstract**: Since intelligence fundamentally relies on efficient skill acquisition (Chollet, 2019), the ability to leverage skills is critical. For LLMs, skills, manually authored or extracted from task trajectories, are textual recipes encoding mature problem-solving experience and are critical to agentic capabilities. Despite widespread deployment, their utility is limited by the model's ability to comprehend and follow skill instructions, especially under complex and long-context scenarios, where key instructions are difficult to locate and adhere to. To address this limitation, we propose ParametricSkills, a framework that can convert free-form textual skills into parameters at test time, enabling context-free skill exploitation. Specifically, we first construct a large-scale, high-quality skill library, and synthesize single-turn and multi-turn skill exploitation trajectories built around these skills with OpenCode. Using these data, we then train a hypernetwork that parameterizes both the skill content and the test-time exploitation methodology by receiving textual skills and converting them into LoRA adapters. Experimental results on six complex software engineering (SWE) subtasks demonstrate that, the proposed ParametricSkills averagely outperforms in-context learning by 6.44 points as judged by DeepSeek-V4-Flash, while also achieving significantly higher BERT Score and F1 score, confirming its effectiveness. Beyond performance, we further find that parametric skills, being inherently accumulative, offer a preliminary yet promising avenue toward test-time continual learning. 

---
# Node-to-Neighborhood Semantic Consistency: Text-Topology Alignment for TAGs Anomaly Detection 

**Authors**: Bochen Lin, Jianxiang Yu, Jiayi Wu, Lin Qi, Huang Lu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.30009)  

**Abstract**: Graph anomaly detection (GAD) on text-attributed graphs (TAGs) is vital for applications such as fraud detection and academic integrity verification. Existing approaches generally fall into two paradigms. GNN-based methods effectively capture structural patterns but struggle to capture fine-grained textual semantics. Methods integrating LLMs with graphs improve semantic understanding yet fail to fully comprehend topological relationships among neighboring nodes. Moreover, both paradigms overlook the correspondence between textual semantics and graph topological relationships, limiting their ability to identify nodes whose semantics are inconsistent with their neighborhoods. In this paper, we formalize TAG anomaly detection as a node-to-neighborhood semantic consistency problem, where anomalies may arise from either textual semantic mismatch or topological deviation between a node and its neighbors. We propose N2NSC (Node-to-Neighborhood Semantic Consistency), a framework that captures the correspondence between graph topology and textual semantics through two complementary fusion paths. The two pathways work synergistically, enabling the LLM to fully leverage both textual and structural neighborhood information for anomaly detection. Extensive experiments across eight datasets demonstrate that N2NSC consistently outperforms current state-of-the-art methods. 

---
# LLM Agents Are Latent Context Managers: Eliciting Self-Managed Context via a Proprioceptive Dashboard 

**Authors**: Binyan Xu, Haitao Li, Kehuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.30005)  

**Abstract**: Long-horizon tool agents are bottlenecked by how their context grows toward the limits of the context window. Recent systems make context management agent- or system-controlled, but they either learn a compression policy that discards evidence or manage context in a layer the agent never sees. We argue both leave a more basic gap unaddressed. Frontier language models are proprioceptively blind to their own context. From the prompt alone they cannot see how large, how old, or how used each block is, the signals a keep-or-drop decision needs. We hypothesize that competent context management is already latent in capable models, and that what is missing is not a learned policy but an interface exposing this state. We introduce VISTA (Visible Internal State for Tool Agents), a training-free, model-agnostic layer that represents working memory as typed, addressable blocks, surfaces a runtime dashboard of per-block token usage, recency, and access history, and archives blocks as recoverable full-fidelity payloads. On LOCA-Bench, BrowseComp-Plus, and GAIA, the same untrained interface transfers across million-, 100K-, and 10K-scale trajectories. On LOCA-Bench it improves four backbones and lifts Gemini-3-Flash from 22.7 to 50.7%. The lift grows with context pressure and transfers across backbones. Ablations further confirm that the dashboard matters beyond archive and recovery tools. 

---
# Are We Measuring Strategy or Phrasing? The Gap Between Surface- and Approach-Level Diversity in LLM Math Reasoning 

**Authors**: Sangmook Lee, Minbeom Kim, Jeonghye Kim, Dohyung Kim, Sojeong Rhee, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2606.29985)  

**Abstract**: Diversity in LLM mathematical reasoning is critical for exploration, but common diversity metrics mostly capture surface-level variation rather than differences in how a problem is solved. We address this gap by introducing approach-level diversity: variation in strategies across correct solutions to the same problem. Using a human-calibrated LLM judge framework, we show that prior diversity measures are unreliable proxies for approach-level diversity, and this mismatch carries over to diversity-aware RLVR, where target metrics are preserved while approach-level diversity declines. Investigating when approach-level diversity helps and whether it can be directly induced, we find that approach-diverse candidate sets improve test-time scaling. However, optimizing an LLM judge diversity reward during training causes the policy to exploit judge-specific preferences rather than broaden its approaches, leaving direct optimization of approach-level diversity as an open problem. Together, our work introduces the notion of approach-level diversity and uncovers a systematic divergence between surface- and approach-level signals, marking a step toward LLMs that reason in genuinely diverse, human-like ways. 

---
# IHDec: Divergence-Steered Contrastive Decoding for Securing Multi-Turn Instruction Hierarchies 

**Authors**: Nicole Geumheon Liu, Haeun Jang, Yonghyun Jun, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.29960)  

**Abstract**: Large Language Models (LLMs) often fail to maintain instruction hierarchies (IH) when processing multi-source inputs with varying role-level priorities, paradoxically adhering to lower-priority directives during conflicts. While existing defenses mitigate this issue, they are largely restricted to single-turn scenarios and require expensive fine-tuning. In this paper, we formalize this failure mode in multi-turn contexts via a Jensen-Shannon Divergence (JSD) framework, uncovering a pervasive role-influence inversion phenomenon where subordinate inputs override superior roles. To rectify this without training, we propose IHDec (Instruction Hierarchy-steered Decoding). IHDec leverages JSD to automatically detect token-level hierarchy violations and dynamically executes contrastive decoding to suppress misaligned subordinate roles. Extensive evaluations demonstrate that IHDec outperforms training-based baselines in multi-turn conflicts while fully preserving general response quality. Furthermore, IHDec strengthens safety against adversarial prompt injections and exhibits a robust scaling synergy with larger models. The Code is available at this https URL 

---
# LatentRevise: Learning from Zero-Hit Reasoning 

**Authors**: Yiqiu Guo, Xueting Han, Qi Jia, Guangtao Zhai, Jing Bai  

**Link**: [PDF](https://arxiv.org/pdf/2606.29938)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is bottlenecked by hard prompts on which correct trajectories have low probability, so sampling misses them within a practical budget and leaves the policy update with little useful signal. We frame such zero-hit prompts as RLVR's sampling frontier, where new reasoning behavior is most valuable yet least likely to be sampled. Importantly, failed rollouts can be informative: they expose where the model's reasoning went wrong. We introduce LatentRevise, a first-order latent revision method that recovers training signal for this zero-hit regime. Given a failed rollout and the gold answer as an anchor, LatentRevise optimizes the input embeddings of its reasoning prefix under two complementary gradients, moving the prefix away from the failed continuation and toward the gold answer. The optimization is constrained to the convex hull of the model's vocabulary embeddings, so each update moves the latent toward a real token embedding rather than an arbitrary feature direction. We find that continuations from the revised prefix lengthen, exhibit self-reflection, and reach correct answers missed by the original rollouts. Used as training data, these trajectories improve SFT and RLVR on math benchmarks over standard baselines. 

---
# Towards Physical Intuitions for Alignment Dynamics: A Case Study With Randomness Crystallization 

**Authors**: Kunal Samanta, Ari Holtzman, Peter West  

**Link**: [PDF](https://arxiv.org/pdf/2606.29933)  

**Abstract**: The alignment of language models is typically studied through the lens of capability benchmarks, but the dynamics of how models change during post-training remain poorly understood. We argue that the physical sciences, and thermodynamic phase-transition theory in particular, offer a principled and underexplored vocabulary for reasoning about these dynamics. As a case study, we instantiate this position through the lens of material Crystallization, which is a well-studied thermodynamic phase transition. For tasks like random number generation, this breaks into 3 phases: (1) the high entropy liquid phase in the pretrained model, with many distinct sampling distributions promptable from the model; (2) the nucleation phase caused by supervised finetuning, in which behavior collapses onto a single seed distribution present in the pretrained LLM; and (3) a settling phase in which reinforcement learning techniques redistribute probability of the collapsed distribution, but largely keep it concentrated on the same options as the seed distribution. We propose intuitive metrics to verify the transitions between these phases, and validate the idea across a range of random tasks. Crystallization is one instance of a broader class of physical frameworks we believe alignment research should import to answer questions about where alignment-induced structure comes from, why it converges where it does, and what it fundamentally cannot change. 

---
# Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios? 

**Authors**: Yangda Peng, Yunjia Qi, Hao Peng, Haotian Xia, Guanzhong He, Xintong Shi, Richeng Xuan, Songyuanyi Lu, Yixian Liu, Zhichao Hu, Yuhong Liu, Lei Hou, Bin Xu, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2606.29920)  

**Abstract**: Rubric-based scoring has become a widely used paradigm in model evaluation, typically with LLM-as-a-Judge (LaaJ) for rubric scoring. However, the reliability of LaaJ for rubric scoring remains underexplored. This concern is especially pronounced in agentic scenarios, where long, complex outputs further challenge reliable scoring. To address this, we conduct a systematic meta-evaluation of LaaJ reliability for rubric verification. We introduce RuVerBench, the first benchmark for assessing LaaJ reliability in rubric verification for agentic scenarios. RuVerBench covers two prevalent agentic domains, deep research and agentic coding, with 2,458 instances, each containing a model-generated output, a rubric, and a human-annotated label indicating whether the output satisfies the rubric. Using RuVerBench, we evaluate numerous frontier LLMs and find that even the most advanced models achieve strong performance but still exhibit substantial noise. We further analyze the impact of key LaaJ strategies, including prompt design, batching, and majority voting, on rubric verification. We find that weaker models are more sensitive to prompt variations, batched verification presents a trade-off between accuracy and efficiency, and majority voting yields effective but diminishing returns. We have released our dataset and code to facilitate future research: this https URL. 

---
# MemDelta: Controlled Baselines and Hidden Confounds in Agent Memory Evaluation 

**Authors**: Kuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29914)  

**Abstract**: Agent memory systems are increasingly evaluated against RAG and full-context baselines, but reported gains often mix changes in the memory method with changes in the language model, embedding model, or retrieval pipeline, making it unclear what is actually being measured. We present MemDelta, a controlled evaluation protocol that varies one component at a time on LongMemEval-S (500 questions, 50+ sessions, three model families). Four findings emerge: (1) verbatim RAG matches full-context GPT-4o-mini (47.2% vs. 49.8%, p = 0.34), but the ranking reverses across models: Gemini gains +14pp from full context, while Sonnet gains +31pp from RAG, partly because it refuses 63% of full-context queries; (2) swapping only the embedding model in an identical pipeline shifts accuracy by +6.2pp at n = 500 (p = 0.004), and Mem0 beats MiniLM-RAG by +11pp but loses to cloud-RAG by 1.2pp, so one variable flips the conclusion; (3) agent self-memory (42%) underperforms basic retrieval (47%); (4) on 2 of 6 question types (n = 88), Mem0 matches cloud RAG (72.7% vs. 73.9%, p = 1.0) at 50x the cost, suggesting narrow rather than general gains. We recommend memory evaluations fix embedding models across comparisons, stratify by model family, and report write-path cost before attributing gains to architecture. 

---
# Timesteps of Mamba Align with Human Reading Times 

**Authors**: Yuji Yamamoto, Shinnosuke Isono, Yoshinobu Kawahara, Sho Yokoi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29904)  

**Abstract**: This study demonstrates an alignment of per-word processing time in a popular state-space language model Mamba and human readers. In Mamba, the recurrent state transition at each layer conceptually takes some duration of time, the discretization timestep $\Delta_t$, determined dynamically in response to the input. Using a naturalistic reading dataset, we show that the per-word timestep from Mamba is a significant predictor of human reading times, and remains significant even when known predictors such as GPT-2 surprisal are controlled for. We further suggest, through formal analysis of Mamba's architecture and internal dynamics, that Mamba can serve as a new, valuable lens to look at human real-time language processing with ever-updated memory, because it allows us to look at how each module (layer) weighs short- and long-term information retention, and how noise may interact with dynamic, continuous memory representation. Code is available online. 

---
# Clinical Reasoning Graphs: Structured Evaluation of LLM Diagnostic Reasoning Reveals Competence Without Consistency 

**Authors**: Nisarg A. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2606.29876)  

**Abstract**: Modern large language models (LLMs) reach 60-70% diagnostic accuracy on complex clinical case benchmarks, but accuracy alone cannot distinguish stable clinically-grounded reasoning from pattern matching. We introduce clinical reasoning graphs, structured graph representations extracted from free-text LLM diagnostic traces using a domain-grounded ontology with 5 node types and 7 edge types. We apply this pipeline to 750 traces from five LLMs across 50 New England Journal of Medicine Clinicopathological Conference cases and three prompt conditions, and test whether diagnostic traces show stable structured reasoning patterns, or diagnostic schemas, for clinically similar cases. We operationalize this as higher graph similarity among clinically similar cases than among clinically dissimilar ones. Across 15 model-condition comparisons, within-cluster and between-cluster composite similarity are nearly equal, and no comparison survives multiple-testing correction; a component-level analysis finds any residual content signal far below schema scale. Graph similarity is also nearly identical for pairs of models that are both correct (0.488) and both incorrect (0.484), suggesting that graph structure captures a dimension not reflected in diagnostic accuracy. Structured reflection prompting increases explicit discriminating-feature analysis within traces (+33%) but does not increase cross-case consistency. These results show diagnostic competence without schema-scale reasoning consistency, and indicate that final-answer accuracy should be complemented by process-level evaluation. We release the ontology, extraction pipeline, validation protocol, and the extracted reasoning graphs and similarity artifacts as resources for structured evaluation of LLM clinical reasoning. 

---
# ARKD: Adaptive Reinforcement Learning-Guided Bidirectional KL Divergence Distillation for Text Generation 

**Authors**: Zilong Liu, Xuewen Zhang, Jinrui Xing, Juyi Qiao, Huiyong Wang, Junming Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29869)  

**Abstract**: Knowledge distillation (KD) is a key technique for compressing Large Language Models (LLMs), yet methods relying on a single KL objective often fail to balance primary distribution fitting with long-tail probability modeling, limiting both generation quality and generalization. To address this, we analyze the complementary roles of forward and reverse KL divergence (FKL/RKL) in distribution alignment from theoretical and empirical perspectives. We then propose a reinforcement-learning-based adaptive KL-weighted distillation framework, in which a policy network dynamically assigns weights to FKL and RKL based on teacher-student distributional characteristics, guided by immediate reward signals to achieve dual alignment on principal and long-tail modes. Extensive experiments demonstrate consistent improvements across Rouge-L and BertScore metrics, surpassing greedy heuristics by 0.4-0.6 points and outperforming other baseline methods on diverse benchmarks. 

---
# KbSD: Knowledge Boundary aware Self-Distillation for Behavioral Calibration in Agentic Search 

**Authors**: Tao Feng, Xinke Jiang, Chao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29863)  

**Abstract**: Agentic search equips large language models with dynamic retrieval abilities, but existing reinforcement learning methods remain limited by reward sparsity in knowledge boundary calibration -- deciding when to trust parametric memory, when to rely on retrieved evidence, and when to abstain. Binary rewards can penalize undesirable outcomes, but provide little guidance on the reasoning process required to make calibrated decisions across different knowledge states. To address this, we propose KbSD (Knowledge boundary Self-Distillation), a framework that tackles this limitation through dense token-level supervision, outcome-level sparse rewards, and quadrant-adaptive optimization. KbSD constructs a hint-augmented teacher, architecturally identical to the student, that receives explicit knowledge boundary signals -- including parametric certainty, retrieval quality, and ground-truth answers -- to generate calibrated reasoning demonstrations. This information-asymmetric self-distillation enables dense supervision without requiring a larger external model. To further account for the heterogeneous reasoning distributions across knowledge states, we introduce a quadrant-adaptive distillation objective: reverse KL for concentrated integration, forward KL for diverse refusal, and Pareto-optimal bidirectional KL for asymmetric quadrants requiring both precision and coverage. Experiments on multiple benchmarks show that KbSD consistently improves both task accuracy and hallucination mitigation over strong baselines, with the largest gains appearing in the challenging quadrants where sparse rewards are least informative. 

---
# Exploring Motivations for Algorithm Mention in the Domain of Natural Language Processing: A Deep Learning Approach 

**Authors**: Yuzhuo Wang, Yi Xiang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29859)  

**Abstract**: With the rise of data-intensive science, algorithms have become central to scientific research. In academic papers, algorithms are mentioned for different purposes, such as describing, using, comparing, or improving methods for specific research tasks. Identifying these purposes can reveal relationships among algorithms and help assess their roles and value. Taking natural language processing (NLP) as an example, this study proposes a sentence-level framework for identifying, analyzing, and tracing the evolution of motivations for mentioning algorithms. We first identify algorithm entities and algorithm-related sentences from full-text papers through manual annotation and machine learning. We then classify mention motivations using pretrained models and data augmentation, and analyze their distribution and temporal evolution. The results show that deep learning models trained with augmented data outperform traditional machine learning models in motivation classification. In NLP papers, more than half of algorithm-related sentences express direct use, whereas improvement is the least frequent motivation. The diversity of motivations has increased over time. For specific algorithm categories, grammar-based algorithms are more often mentioned for description, while machine learning algorithms are more often mentioned for use. Over time, use motivations have gradually replaced description motivations across different algorithms, and the number of motivation types associated with individual algorithms has declined significantly. This study reveals how authors mention algorithm entities in academic writing and provides a basis for future research on algorithm relationship identification and algorithm impact evaluation. 

---
# Smooth Scaling Laws Hide Stepwise Token Learning 

**Authors**: Pingjie Wang, Zechen Hu, Peiru Yang, Fu Guo, Debing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29858)  

**Abstract**: Language model loss follows remarkably regular scaling laws over model and data size, yet it remains unclear why the aggregate loss should exhibit a power-law form. Existing explanations often attribute this regularity to a heavy-tailed spectrum of pattern difficulty in natural language, but this view has not been directly validated at token-level granularity in large-scale real-data training. We present a token-level framework that decomposes scaling laws into localized learning events of individual contextualized tokens. By fitting token loss trajectories with sigmoids, we show that token learning is concentrated in localized transitions, giving rise to a learning-time spectrum that dominates the scaling-law shape. Across more than one hundred pre-training runs on large and diverse real-language corpora with modern LLM architectures, scaling up to 6B parameters and 300B training tokens, the measured learning-time spectrum quantitatively reconstructs the validation loss derivative along the training-step $T$, data-scale $D$, and model-scale $M$ axes. We further show that the same signal is actionable: by reshaping the training distribution according to when tokens become learnable, we alter the optimization trajectory and achieve 11\% faster validation-loss reduction. These results provide direct empirical evidence that scaling laws are governed primarily by the distribution of token-level learning times, and that this distribution can be used not only to explain scaling behavior but also to improve training performance. 

---
# MATCH: Modulating Attention via In-Context Retrieval for Long-Context Transformers 

**Authors**: Linrui Ma, Chun Hei Lo, Xinyu Wang, Peng Lu, Xihao Yuan, Hanting Chen, Kai Han, Xinghao Chen, Chengjun Zhan, Hanlin Xu, Yichun Yin, Lifeng Shang, Feng Wen, Boxing Chen, Yufei Cui  

**Link**: [PDF](https://arxiv.org/pdf/2606.29844)  

**Abstract**: The quadratic computational cost of traditional attention mechanisms poses a major bottleneck to the scalability and practical deployment of large language models (LLMs), particularly in long-context scenarios. To improve efficiency, existing approaches often enforce rigid structural constraints such as local attention windows. However, these strategies typically lead to substantial performance degradation on tasks requiring precise long-range recall. In this work, we propose MATCH, a scalable and efficient framework that augments sparsified attention mechanisms with dynamically integrated in-context information through an efficient retrieval system. Empirical results show that MATCH significantly improves the performance of sparse-attention models on both synthetic and real-world natural-language tasks. These findings highlight the versatility of MATCH as a general approach for enhancing in-context retrieval capabilities while maintaining the efficiency benefits of sparse attention architectures. 

---
# Revealing the Technology Development of Natural Language Processing: A Scientific Entity-Centric Perspective 

**Authors**: Heng Zhang, Chengzhi Zhang, Yuzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29836)  

**Abstract**: Most studies on technology development have been conducted from a thematic perspective, but the topics are coarse-grained and insufficient to accurately represent technology. The development of automatic entity recognition techniques makes it possible to extract technology-related entities on a large scale. Thus, we perform a more accurate analysis of technology development from an entity-centric perspective. To begin with, we extract technology-related entities such as methods, datasets, metrics, and tools in articles on Natural Language Processing (NLP), and we apply a semi-automatic approach to normalize the entities. Subsequently, we calculate the z-scores of entities based on their co-occurrence networks to measure their impact. We then analyze the development trends of new technologies in the NLP domain since the beginning of the 21st century. The findings of this paper include three aspects: Firstly, the continued increase in the average number of entities per paper implies a growing burden on researchers to acquire relevant technical background knowledge. However, the emergence of pre-trained language models has injected new vitality into the technological innovation of the NLP domain. Secondly, Methods dominate among the 179 high-impact entities. An analysis of the z-score trend about the top 10 entities reveals that pre-trained language models, exemplified by BERT and Transformer, have become mainstream in recent years. Unlike the trend of the other eight method entities, the impact of Wikipedia dataset and BLEU metric has continued to rise in the long term. Thirdly, in recent years, there has been a remarkable surge in popularity for new high-impact technologies than ever before, and their acceptance by researchers has accelerated at an unprecedented speed. Our study provides a new perspective on analyzing technology development in a specific domain. 

---
# Neural Procedural Memory: Empowering LLM Agents with Implicit Activation Steering 

**Authors**: Chengfeng Zhao, Yuqiao Tan, Shizhu He, Yequan Wang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29824)  

**Abstract**: While Large Language Models (LLMs) excel as static solvers, transforming them into autonomous agents remains challenging. This transition requires continuous environmental interaction, yet current agents lack the necessary persistent procedural memory. Existing approaches predominantly employ Retrieval-Augmented Generation (RAG) to inject explicit textual guidelines into model contexts. However, relying solely on symbolic instructions can introduce a text-action disconnect, frequently failing to activate the internal representations necessary for correct task execution. To address this, the paper introduces Neural Procedural Memory (NPM), a training-free framework that represents agent memory through implicit activation steering rather than explicit instructions. By distilling procedural skills from historical contrastive experiences into steering vectors in the activation space, NPM directly activates the task-relevant neural mechanisms to guide task execution. Evaluations across four agent benchmarks show that NPM performs comparably to baselines using explicit textual instructions. Furthermore, the results show that combining implicit steering with explicit workflows provides complementary advantages, leading to more robust task execution. Representational analyses indicate that these steering vectors encode consistent task logic, forming organized structures within the activation space. These findings suggest that implicit activation steering provides a promising approach for managing agent memory. 

---
# SrDetection: A Self-Referential Framework for Data Leakage Detection in Code Large Language Models 

**Authors**: Shuaimin Li, Liyang Fan, Zeyang Li, Zhuoyue Wan, Yufang Lin, Shiwen Ni, Feiteng Fang, Hamid Alinejad-Rokny, Yuanfeng Song, Kun Jing, Chen Jason Zhang, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29815)  

**Abstract**: Evaluating code large language models (Code LLMs) requires reliable detection of data leakage, where benchmark performance is artificially inflated by exposure to benchmark data during pre-training. Existing approaches either assume access to proprietary training corpora, rely on brittle heuristics such as timestamp filtering, or use external reference sets with manually tuned, non-generalizable thresholds. To address these limitations, we introduce \textbf{SrDetection}, a unified \textbf{s}elf-\textbf{r}eferential leakage detection framework for both gray-box (access to model logits) and black-box (access to model outputs) settings. SrDetection generates semantically equivalent variants of a benchmark sample and detects leakage by contrasting the model's behavior on the original versus its variants, flagging cases where the original is disproportionately easier for the model. We further design a controlled leakage detection testbed and evaluate SrDetection in this environment. Across different models and training stages, SrDetection improves average F1 by 21.52 points in the gray-box setting and 14.46 points in the black-box setting over strong baselines, demonstrating robust, threshold-independent leakage detection. Finally, a gray-box study of 15 widely used Code LLMs on four popular benchmarks reveals benchmark-specific leakage patterns beyond prior overlap-based analyses\footnote{\footnotesize Source code and data are available at this https URL 

---
# How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation 

**Authors**: Kriti Faujdar, Smit Kadvani  

**Link**: [PDF](https://arxiv.org/pdf/2606.29809)  

**Abstract**: Hallucination detection has become a pressing requirement for trustworthy AI deployment at scale. The most accurate detection methods depend on GPU-intensive inference, proprietary API calls, or white-box access to the generating model. This puts them out of reach for resource-constrained researchers and practitioners. In this paper, we explore a practical alternative: how well can hallucination detection perform using only lightweight, CPU-feasible methods built on publicly available models? We systematically benchmark five such methods: ROUGE-L, semantic similarity, BERTScore, a Natural Language Inference (NLI) detector based on a FEVER-trained DeBERTa model, and a score-level ensemble of similarity and NLI. We evaluate them across all three tasks of the HaluEval benchmark: question answering (QA), dialogue, and summarisation. We calibrate each method on a held-out validation split and evaluate it on 2,000 test instances per task. We find that no single method dominates and performance is highly task-dependent. The ensemble performs best on QA (F1 = 0.792, AUC-ROC = 0.873), the NLI detector leads on dialogue (AUC-ROC = 0.713), and all five methods degrade to near-random performance on summarisation (AUC-ROC between 0.469 and 0.574). This task-dependence and the systematic failure on summarisation map the practical frontier of GPU-free hallucination detection. They give practical guidance for method selection under computational constraints. All experiments run on a standard laptop CPU using public models. 

---
# Fund2Persona: A Framework for Building and Refining Financial Advisor Personas from Fund Disclosure Data 

**Authors**: Suhwan Park, Hoyoung Lee, Zhangyang Wang, Alejandro Lopez-Lira, Young Cha, Chanyeol Choi, Jaewon Choi, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2606.29793)  

**Abstract**: Demand for personalized financial advising is growing, but consistent advisor expertise is difficult to obtain, scale, and encode in LLM systems. Simple persona prompts rarely specify how a financial advisor should reason and often drift toward generic recommendations. We propose Fund2Persona, a framework that grounds financial-advisor personas in fund disclosures, holdings transitions, market context, and manager commentary, then refines them through an agentic actor--scorer--patcher loop. We evaluate the resulting personas on held-out holdings-transition reconstruction and manager-commentary alignment, where they better recover portfolio decisions and grounded manager interpretation than generic baselines. We further study two downstream diagnostics: market-scenario generation, where persona retrieval broadens plausible investment views beyond repeated generic rollouts, and advisory dialogues grounded in investor profiles, where matched personas give more specific and useful advice than a generic advisor. These results suggest that fund-data-grounded financial-advisor personas can make manager-specific investment expertise portable rather than merely changing an LLM's surface style. 

---
# Are Humans Evolved Instruction Followers? An Underlying Inductive Bias Enables Rapid Instructed Task Learning 

**Authors**: Anjishnu Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.29792)  

**Abstract**: Human adults can often perform a novel task correctly on the first attempt after only receiving verbal or written instructions. This rapid instructed task learning (RITL) is a hallmark of human cognitive flexibility, yet its mechanisms and parallels in artificial systems remain under-explored across disciplines. In this position paper, we argue that humans possess an evolved instruction-following bias -- an inductive bias shaped by evolution to interpret and execute linguistic instructions which critically enables fast generalization of behavior from language. This bias functions analogously to the way large language models (LLMs) leverage instruction tuning to achieve zero-shot task performance. We synthesize evidence from cognitive science, neuroscience, and machine learning research to support this hypothesis. While instruction-following in AI is currently achieved via specialized training protocols, we posit that in humans it arises as an innate cognitive architecture feature. We outline testable predictions and call for more interdisciplinary research to investigate Instruction-Following as a unifying mechanism enabling rapid task learning in both natural and artificial neural networks. 

---
# Managing Map Cardinality in Automatic Disease Classification Mapping: Balancing Precision, Recall and Coverage 

**Authors**: Santosh Purja Pun, Oliver Obst, Jim Basilakis, Jeewani Anupama Ginige  

**Link**: [PDF](https://arxiv.org/pdf/2606.29750)  

**Abstract**: Automatic mapping between disease classification systems, such as the International Classification of Diseases (ICD), is a challenging yet essential task for integrating health data and conducting longitudinal data analysis. Existing embedding-based methods primarily focus on \emph{one-to-one} mappings, overlooking more complex \emph{one-to-many} scenarios. The threshold-based and top-K methods offer natural extensions; however, they involve inherent trade-offs between \emph{precision}, \emph{recall} and \emph{mapping coverage} -- the proportion of source codes with at least one mapping to a target code. To address this challenge, we introduce a novel method, which is inspired by the \emph{blocking-and-matching} pipeline commonly used in \emph{entity resolution}. In particular, we first generate a block of candidate matches (\emph{blocking}) and then employ a large language model (LLM) to identify all valid mappings within each block (\emph{matching}). Empirically, we show that the proposed method achieves higher precision with comparable recall and broader coverage across multiple ICD version pairs (ICD-9-CM$\leftrightarrow$ICD-10-CM and ICD-10-AM$\leftrightarrow$ICD-11). Our source code and dataset is available at: this https URL. 

---
# Fast Numbers, Slow Language: Bridging Quantitative and Qualitative Earnings Signals 

**Authors**: Ding Yu, Zhuo Liu, Hao Zhang, Hangfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2606.29734)  

**Abstract**: Earnings announcements release two types of information sequentially: quantitative surprise (numeric earnings-per-share (EPS)/revenue versus analyst estimate) arrives first in press releases and financial news, processed by algorithmic traders within minutes; qualitative language (management tone, guidance, question-and-answer (Q&A) credibility) arrives 30-90 min later in the earnings conference call transcript (ECT), requiring human interpretation overnight. Financial economists have studied quantitative surprise for 50 years; natural language processing (NLP) researchers have studied qualitative ECT signals for a decade. Despite studying the same event, the two communities used incompatible frameworks: different targets (return vs. volatility), trading setups (long top-decile and short bottom-decile vs. trade-all), and metrics (return spread between top and bottom 20% (Q5-Q1) vs. mean squared error (MSE)), making direct comparison and connection challenging.
We bridge these communities with EarningsInOne, the first corpus aligning earnings news, ECTs, and intraday and next-day prices across SP 1500 (broad U.S. equity universe, 2022-2025). Applying unified trading and evaluation tools to both signal types, we confirm a clean speed separation, fast numbers, slow language: quantitative surprise peaks at announcement and is largely eliminated by the next market open; qualitative ECT sentiment peaks on the next trading day, real and tradeable, but hidden under prior transcript-based evaluation that optimised sign-agnostic volatility with pointwise MSE. 

---
# How Far Do On-Prem Open LLMs Get on Text-to-SQL? A Cross-Family Size x Technique Frontier on BIRD 

**Authors**: Vladimir Beskorovainyi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29733)  

**Abstract**: Organizations that cannot send data to a cloud API increasingly ask: how good is Text-to-SQL if the model must run on-premises on open weights, and which popular accuracy "recipes" are worth their compute? We answer with an honest, fully reproducible benchmark on the BIRD development split (n=1534, Execution Accuracy), evaluating three open model families across two generations -- Qwen2.5-Coder (7B/14B/32B), CodeLlama-Instruct (7B/13B/34B), and Llama-3.x (8B, 70B) -- under one matched protocol, ablating a model-agnostic recipe (schema linking, self-correction, self-consistency) component by component, with every difference tested by the paired McNemar test. Four findings stand out. (i) Generation matters more than raw size, and the recipe is family-robust: Qwen2.5-Coder dominates the older CodeLlama at matched size (39.1 vs 20.9 at 7B), but a modern non-Qwen model (Llama-3.3-70B, 49.2 on a matched serving) is competitive, so CodeLlama's weakness reflects its 2023 generation, not "non-Qwen = weak". (ii) Self-correction is a robust, near-free win, significant on all three families where there is room to improve. (iii) Schema linking does not help, and a stronger linker does not rescue it: a retrieval/embedding linker with 96.5% gold-table recall is statistically indistinguishable from no linking, ruling out the "weak lexical strawman" objection across three families. (iv) Self-consistency is poor value (+0.13 pp for ~5x tokens, not significant). We report real per-stage cost ($/1k queries) and release all code, predictions, and summaries; archived code and data: this https URL 

---
# SEVA: Self-Evolving Verification Agent with Process Reward for Fact Attribution 

**Authors**: Aojie Yuan, Yi Nian, Haiyue Zhang, Zijian Su, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29713)  

**Abstract**: Hallucination is the reliability bottleneck for LLM-based agents, and fact attribution verifiers are the last line of defense -- yet today's verifiers emit only opaque binary labels, leaving agents unable to self-correct and operators unable to audit. We present SEVA, a structured verification agent that emits evidence alignments, step-by-step reasoning chains, calibrated confidence, and a six-category error diagnosis with actionable fixes. Training such an agent with RL is non-trivial: standard binary reward on multi-component output triggers advantage collapse -- within-group reward variance vanishes and the GRPO gradient disappears. We resolve this with a process reward that decomposes verification quality into five independent components weighted 70/30 toward process signals, restoring the gradient and inducing an implicit curriculum -- the agent first masters verification behavior (alignment 0.917 -> 0.997, format 72% -> 100%), then outcomes (F1 64.9 -> 69.0). Structured output further enables a Verify -> Reflect -> Probe -> Refine self-evolution loop, which over four rounds on a 7B model surfaces an unexpected structural finding: each round produces a benchmark-specialist, not a generalist (+15 pp on HaluEval, -10 to -14 pp on TruthfulQA in the same model, persistent at 4x data). On ClearFacts, SEVA-3B matches GPT-4o-mini (69.0 vs. 69.8 F1) while producing substantially richer, auditable output -- confirming a principle that should generalize: for any RL task with multi-component generation, reward granularity must match output granularity. 

---
# Why Struggle with Continuous Latents? Interpretable Discrete Latent Reasoning via Rendered Compression 

**Authors**: Shuochen Chang, Qingyang Liu, Shaobo Wang, Bingjie Gao, Qianli Ma, Haonan Zhao, Yibo Miao, Yulin Sun, Zelin Peng, Jiangtong Li, Li Niu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29712)  

**Abstract**: Large language models achieve high reasoning performance via explicit chain-of-thought and reinforcement learning, but require long output sequences and extended inference time. Latent reasoning reduces this cost by shifting computation into a latent space; however, continuous latent methods are hard to train, suffering from unstable and uninterpretable reasoning trajectories. We argue these issues stem from a misalignment between continuous-space reasoning and discrete symbolic supervision, as continuous states lack explicit anchors for step-by-step alignment. To resolve this, we propose \textbf{Discrete Latent Reasoning~(DLR)}, the first method that converts continuous latent states into explicit discrete tokens. Inspired by render-based compression, we render textual chains of thought into images, extract visual features, and construct a discrete latent vocabulary via clustering-based fine-tuning. Expanding the vocabulary and output head enables standard autoregressive modeling over both natural language and latent tokens, supporting pretraining alignment, SFT, and RL. Experiments on five reasoning benchmarks and two model series~(Qwen3-VL and LLaMA-3) confirm that \textbf{DLR} outperforms prior latent reasoning baselines with up to \textbf{20$\times$ compression}. Furthermore, the learned latent trajectories retain an interpretable semantic structure. Overall, discrete latent tokens provide a controllable and interpretable basis for efficient latent reasoning. 

---
# Can MLLMs Critique Like Humans? Evaluating Open-Ended Aesthetic Reasoning in Multimodal Large Language Models 

**Authors**: Sajjad Ghiasvand, Maryam Amirizaniani, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani  

**Link**: [PDF](https://arxiv.org/pdf/2606.29689)  

**Abstract**: Open-ended aesthetic critique is a challenge for multimodal large language models (MLLMs): unlike multiple-choice aesthetic benchmarks, it has no single correct answer, and most aesthetic evaluation has measured models against numeric scores rather than the written critiques people actually give. We evaluate MLLM critiques against ranked human references and ask whether they are close to human ones. Using the Reddit Photo Critique Dataset, we score five open-weight MLLMs against multiple ranked human critiques per photo with reference-based similarity metrics, under six prompt conditions that disentangle persona framing, aspect hinting, length control, and single- versus multi-pass generation, and add an image-grounding control that feeds each model the wrong photograph. We find that reference-based similarity gives a misleading picture. Stricter lexical and learned metrics show only weak alignment with human critiques, while a coarse embedding cosine reports broad topical overlap that the grounding control traces to a stable house style rather than image-specific observation. Behaviorally, the models diverge from humans in consistent ways the scores do not surface: even under a length cap they write two to three times as much, cover nearly every aesthetic aspect where humans are selective, engage each aspect more uniformly and at greater depth, and repeat themselves across critiques of the same photo where humans vary. We argue that reference-based similarity rewards a fluent, comprehensive critique style rather than the selectivity and specificity of human critique, and discuss implications for evaluating and training open-ended multimodal generation. 

---
# How LLMs See Creativity: Zero-Shot Scoring of Visual Creativity with Interpretable Reasoning 

**Authors**: William Orwig, Roger E. Beaty  

**Link**: [PDF](https://arxiv.org/pdf/2606.29672)  

**Abstract**: Evaluating the originality of visual images poses enduring challenges for creativity assessment. Automated scoring using AI models has proven effective in the verbal domain, yet key questions remain about evaluating visual creativity and understanding how models arrive at their ratings. The present research asks whether multimodal large language models (LLMs) can serve as judges of visual creativity zero-shot (without any fine-tuning or examples of human ratings) and whether their "reasoning" output offers an interpretable window into their evaluation process. We tested six multimodal LLMs (Gemini 3 Flash, Gemma 4 31B IT, GPT-5.4 Mini, GLM-5v Turbo, Kimi K2.5, and Qwen 3.6 Plus) on 992 AI-generated images (based on human-written prompts) and 1,500 hand-drawn sketches scored for creativity by human raters. In Study 1, all models showed substantial alignment with human creativity ratings on both datasets (r = .57-.68 on AI-generated images; r = .29-68 on sketches). In Study 2, we analyzed the step-by-step reasoning processes of three LLMs evaluating the same images and drawings. Although reasoning made model evaluations interpretable -- showing what they attend to, how they balance originality vs. quality, and how they justify their ratings -- reasoning did not improve alignment with human ratings. In sum, our findings indicate that multimodal LLMs can match human judgments of visual creativity without any additional training, and that their reasoning reveals how AI models evaluate creativity. An open scoring app implementing this pipeline is available at this https URL. 

---
# Resolution Thresholds in VLM Detection of Harmful ASCII Art Across Construction Modes and Languages 

**Authors**: Yikai Hua, Peter West  

**Link**: [PDF](https://arxiv.org/pdf/2606.29649)  

**Abstract**: Large Vision-Language Models (VLMs) are increasingly deployed as content moderation tools, yet they remain vulnerable to jailbreak attacks in which harmful text is visually encoded as ASCII art. This can allow inappropriate or harmful content to bypass moderation systems. To address this vulnerability, this paper investigates how image resolution affects VLM detection of harmful ASCII art across eight character construction modes (L1-L8), ranging from dense block characters to word-embedded designs. We evaluate eight state-of-the-art VLMs on English and Chinese corpora using a pipeline that generates ASCII art images at ten resolution scales, probing whether a consistent detection-failure threshold exists across models, modes, and languages. Results indicate that detection rates decline sharply above certain resolution thresholds, and that word-based modes are the most resistant to detection across the full resolution range. These findings reveal a systematic vulnerability in VLM-based content moderation systems and motivate resolution-aware evaluation standards. 

---
# Hybrid Retriever Evolution for Multimodal Document Reasoning Agents 

**Authors**: Bohan Yao, Shruthan Radhakrishna, Vikas Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2606.29648)  

**Abstract**: Different retrievers, including lexical, semantic, and multimodal approaches, provide highly complementary strengths for multimodal document understanding, yet most systems combine them through fixed pipelines that cannot adapt to the demands of individual reasoning steps. In this work, we ask whether retrieval orchestration itself can be learned as part of the reasoning process. We introduce a failure-driven evolution framework in which a meta-agent autonomously discovers how a tool-using task agent should coordinate diverse retrievers during multi-step document question answering. The meta-agent analyzes incorrect reasoning trajectories, actively probes the same tool environment to diagnose root causes, and iteratively rewrites the task agent's instructions, turning retrieval from a fixed front-end stage into an adaptive, step-wise reasoning decision. The evolved agent learns when to invoke each retriever, how to combine them, and how to compose evidence across modalities and pages. On MMLongBench-Doc and DocBench, the evolved agent achieves gains of up to +19.6 points over the unevolved baseline and consistently outperforms recent systems including MACT, MDocAgent, and SimpleDoc. Detailed retrieval analyses confirm that these improvements arise from adaptive routing and evidence composition rather than reliance on any hard coded retrieval mode, and evolution dynamics reveal a progressive shift from narrow lexical behavior to rich multi-tool coordination. These findings establish autonomous multi-agent coordination as a promising paradigm for multimodal document reasoning. 

---
# Two-Stage Prompt Optimization for Few-Shot Relation Extraction: From Reasoning-Guided Search to Gradient-Guided Refinement 

**Authors**: Aunabil Chakma, Mihai Surdeanu, Eduardo Blanco  

**Link**: [PDF](https://arxiv.org/pdf/2606.29639)  

**Abstract**: Automatic prompt optimization is still underexplored for episodic few-shot relation extraction with smaller language models. We propose a two-stage framework that combines reasoning-based prompt optimization with gradient-based prompt optimization. The first stage can use any reasoning-based optimizer to make broadprompt improvements in natural language. The second stage applies our GradPO, which uses loss and gradient signals to identify high-impact prompt spans and refine them with local edits. Experiments on FS-TACRED and FS-FewRel show that local refinement usually improves prompts found by the first stage, and GradPO is the most consistent refiner. Our framework achieves state-of-the-art performance on FS-TACRED with Qwen3-4B and remains competitive on FS-FewRel. 

---
# Do We Still Need Fine Tuning? Turkish Sentiment Analysis in the Era of Large Language Model 

**Authors**: Sercan Karakaş, Yusuf Şimşek  

**Link**: [PDF](https://arxiv.org/pdf/2606.29614)  

**Abstract**: This study examines whether supervised fine-tuning remains necessary for Turkish sentiment analysis in the era of large language models. We compare classical machine learning methods, fine-tuned pretrained language models, and prompted large language models on a Turkish e-commerce review dataset with negative, neutral, and positive labels. Fine-tuned BERTurk models perform best overall and outperform all prompted large language models in the full three-class task. The neutral class emerges as the main difficulty: while several large language models are much more competitive in binary positive--negative classification, they degrade substantially in the three-class setting by collapsing neutral reviews into polarized categories. The findings suggest that, in realistic Turkish sentiment classification, prompted large language models do not yet match supervised fine-tuning in the zero-shot setting, and that including the neutral class is crucial for robust evaluation. 

---
# How much of an LLM-generated clinical corpus is actually new? A production-scale measurement of content redundancy for provenance classification 

**Authors**: Ali H. Lazem, William J. Teahan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29605)  

**Abstract**: Clinical machine learning increasingly relies on training corpora generated by large language models (LLMs) rather than annotated by clinicians, and such corpora are described and reused largely on the basis of their reported scale. We test whether volume reflects information content. Analysing the complete output of a multi-agent clinical extraction pipeline applied to 167,034 patient narratives, 2.51 billion generated tokens across the ten text-bearing channels of an eleven-channel pipeline, we introduce Provenance-based Redundancy Decomposition, a token-level classification of the entire output by source. Only 10.9% of the output is trainable-unique content while 79.4% is redundant; raw token count overstates information content by roughly ninefold. The redundancy arises through two distinct mechanisms, verbatim copying of source context into per-item fields, and duplication of generated text across records, of which only the former is losslessly removable. An independent, model-free analysis based on lossless compression confirms the redundancy, recovering the two mechanisms without reference to the provenance labels. One pipeline channel carries almost no redundancy, showing that the level of redundancy depends on how each channel is structured rather than being a fixed property of LLM extraction. Because uncorrected redundancy up-weights the longer, more complex presentations that generate the most items, it skews the token-level training distribution of the corpus, a property we measure directly. In a controlled downstream test, de-duplicating the corpus before adaptation improved a clinical encoder on external disease-recognition benchmarks at equal token budget, robustly across adaptation depths and replicated on a second benchmark, confirming that the redundancy carries a measurable cost beyond storage. The classification tool is released openly. 

---
# MAM-AI: An On-Device Medical Retrieval-Augmented Generation System for Nurses and Midwives in Zanzibar 

**Authors**: Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.29580)  

**Abstract**: Maternal and newborn mortality remain among the highest in sub-Saharan Africa, where midwifery care is often delivered by nurses who lack midwifery training to international standards, and consulting authoritative guidance at the point of care is hard: the guidelines are long and connectivity is intermittent. We present MAM-AI, a medical question-answering assistant for nurse-midwives in Zanzibar that runs entirely on a commodity Android device: a question is embedded (EmbeddingGemma, 300M) and matched against a curated corpus of 87 guideline documents (63,650 passages), then answered with citations by a 4B int4 generator (Gemma 4 E4B), fully offline, with no query leaving the device. We evaluate the exact deployed configuration with a layered methodology -- retriever, generator under oracle context, end-to-end, and latency -- scored by LLM judges validated against physician rubrics. The evaluation relocates the hard problem. On-device retrieval is essentially solved: the 300M embedder ranks third of seven retrievers and rivals cloud systems, so the passages the system needs are usually found. The small generator is what remains in doubt: adding retrieved context does not improve its answers, and at 4B it cannot be both helpful and safe at once -- of two same-size candidates, the more helpful one commits genuine dangerous errors, so we deploy the other, which is about twice as faithful to its sources (as faithful as a frontier model), and recover its helpfulness with a redesigned prompt that cuts deflection from 33% to 3%. Corpus quality is decisive for the same reason: where the corpus holds the right passage the answer is specific and actionable, and where it does not it goes vague. MAM-AI is a thoroughly evaluated, open-source research prototype, not a fielded product; the system, knowledge base, benchmarks, and evaluation harness are released. 

---
# Anisotropy Decides Cosine vs. Rank Metrics for Text Embeddings 

**Authors**: V.S. Raghu Parupudi  

**Link**: [PDF](https://arxiv.org/pdf/2606.29571)  

**Abstract**: The standard way to compare two text embeddings is cosine similarity. Scattered studies report that a different metric does better, but never pin down the geometric condition that decides when, or why. We settle both with a comprehensive empirical study: nineteen parameter-free similarity metrics on nineteen encoders, from compact sentence transformers up to seven-billion-parameter large language models, across seven datasets. The answer is geometric. When an encoder spreads its variance evenly across directions, cosine is the best parameter-free choice and no other metric helps by a usable margin. When the variance concentrates into a few dominant directions, a property known as anisotropy, rank-based and L1-type metrics beat cosine by a clear margin. The absolute gain is modest, but because cosine starts low on these encoders it is a sizable relative improvement, around twenty percent on average and largest where cosine is weakest. What decides this is the geometry of the embedding space, not how the model was trained: where the two disagree, the metric follows the geometry. One number, the fraction of variance held by the single most dominant dimension, predicts how much the alternatives help across all nineteen encoders, with a rank correlation of 0.86 and a linear correlation of 0.95. To test this as the cause rather than a correlate, we project out the dominant directions: cosine recovers and the advantage of the other metrics nearly vanishes, but only on the encoders that were anisotropic to begin with. The effect is directional, not magnitude based, since it survives normalizing every vector to unit length. Among parameter-free metrics, then, cosine is the right tool wherever an encoder is well spread, which includes the fine-tuned embedders commonly deployed for retrieval, and we give a one-number diagnostic for when it is not. 

---
# Coverage-Driven KV Cache Eviction for Efficient and Improved Inference of LLM 

**Authors**: Shuvendu Roy, Mengyao Zhai, Hossein Hajimirsadeghi, Golnoosh Samei  

**Link**: [PDF](https://arxiv.org/pdf/2606.29563)  

**Abstract**: Large language models (LLMs) excel at complex tasks like question answering and summarization, thanks to their ability to handle long-context inputs. However, deploying LLMs is costly, not only due to the high computational demands of quadratic complexity of self-attention and auto-regressive generation, but also because of the significant memory overhead required for storing the key-value (KV) cache during inference. To reduce the memory cost, existing KV-cache eviction strategies leverage the sparsity in attention to selectively store a subset of tokens. While reducing the memory footprint, such approaches show a considerable drop in performance, especially in tasks that require long-context reasoning. We identify that the drop in performance is linked to a reduction in the coverage of unique tokens. Additionally, we theoretically show that reduced coverage limits the mutual information between inputs and outputs, thereby impairing predictive accuracy. To this end, we introduce K-VEC, a novel coverage-aware KV-cache eviction strategy that prioritizes token coverage while evicting tokens in the cache. K-VEC introduces a cross-head and a cross-layer coverage module to enhance token retention across attention heads and model layers, mitigating performance degradation caused by low coverage. Evaluated on 16 LongBench subsets, K-VEC exhibit up to 10.35 points improvement over the existing methods under the same eviction rate and memory constraint. Comprehensive evaluations validate the effectiveness of our approach and demonstrate its potential for efficient LLM deployment in resource-constrained settings. 

---
# AURORA: Asymmetry and Update-Induced Rotation for Robust Hallucination Detection in Large Language Models 

**Authors**: Zishuai Zhang, Hainan Zhang, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29545)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks. However, their tendency to generate hallucinations, namely factually incorrect or unfaithful outputs, poses a critical obstacle to their deployment in high-stakes applications. Although recent hallucination detection methods have made encouraging progress, they typically rely on costly output-level consistency checks or static hidden-state probes that capture shallow dataset-specific patterns, leading to substantial degradation under cross-dataset evaluation. In this work, we propose AURORA, a novel hallucination detection framework that shifts the focus from static representations to the weight-gradient dynamics of LLMs. Our key insight is that hallucinated and faithful answers induce qualitatively different gradient update patterns on the model's parameters. Specifically, hallucinated samples trigger asymmetric and structurally misaligned gradients, which can be captured through two complementary features: (1) the skewness of the cosine similarity distribution between weight matrices and their gradient update directions, and (2) the rotation ratio, which quantifies how much the gradient update reorients the singular-vector basis of weight matrices via SVD. AURORA achieves strong hallucination detection performance across four model families and four benchmark datasets. Further analyses demonstrate that our method scales effectively across model sizes and transfers to out-of-domain tasks, including mathematical reasoning and vision-language scenarios. 

---
# Preference-ASR: A Preference-Aware Test Set for Benchmarking ASR in the Era of Speech LLMs 

**Authors**: Nithin Rao Koluguri, Sasha Meister, Nikolay Karpov, Piotr Zelasko, Desh Raj, Jagadeesh Balam, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2606.29534)  

**Abstract**: Popular ASR test sets adopt inconsistent conventions for numbers, disfluencies, entities, and casing, while standard normalizers erase the format distinctions users care about. Current benchmarks therefore cannot measure whether a model follows user preferences for output style. We introduce PreferenceASR, a test set evaluating ASR systems on their ability to follow natural-language preference instructions across four categories: normalization, entities, disfluencies, and case. Built from seven open-source corpora via a two-stage LLM-assisted pipeline with human verification, it is evaluated with a preference-aware normalizer that selectively skips steps matching the active instruction. Benchmarking four models shows rankings shift across preference types, exposing quality differences traditional evaluation obscures. We publicly release the dataset. 

---
# The Verbose Context Problem in Medical Records 

**Authors**: Shiva Kaul, Min-Gyu Kim, Anjum Khurshid, Sriram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2606.29503)  

**Abstract**: The verbose context problem occurs when structured concepts have token-inefficient textual representations. This bottleneck is acute in population health: cohort-level analysis of longitudinal patient records requires reasoning over thousands of medically-coded events, often exceeding 400K tokens in total. We present PopMedQA, a benchmark isolating this problem through computational tasks on groups of longitudinal patient records. We construct the benchmark using neopatient, a new library for language-controlled generation of artificial patient records. Through extensive ablations -- including prompting strategies, prompt compression, and agentic decomposition -- we find that domain-independent methods fail to alleviate the verbose context problem. There remains significant opportunity to exploit domain-specific structure in language model inputs for population-scale reasoning. 

---
# Which Tokens Need Context? A Reference-Based Analysis of Translation Responsibility Using Fertility and Entropy 

**Authors**: Ramakrishna Appicharla, Baban Gain, Santanu Pal, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2606.29489)  

**Abstract**: When humans translate, not every word depends equally on the surrounding context. Some tokens, particularly function words like pronouns and auxiliaries, rely heavily on preceding or following sentences, while others, such as proper nouns, do not. Understanding this inherent context sensitivity is essential for evaluating whether machine translation systems use context in human-like ways. However, existing approaches to analysing context usage rely on discourse-specific test sets or model internals, making them narrow or model-dependent. We propose a post-hoc, model-agnostic framework to quantify context sensitivity at lexical and syntactic levels using two measures derived from word alignments: fertility (number of target tokens generated per source token) and entropy (stability of fertility patterns across contexts). Using reference translations for three language pairs (German $\leftrightarrow$ English, English $\rightarrow$ Hindi) under four context conditions, we show that context selectively redistributes generative responsibility from source to context tokens without altering overall fertility. Function words show the largest fertility reductions, while content words remain stable, suggesting that context resolves ambiguity rather than adding new information. Our framework provides a ground-truth characterisation of selective context usage in human translation, establishing a diagnostic baseline for evaluating machine translation models. 

---
# To Reason or to Fabricate: Reasoning Without Shortcuts via Hint-Anchored Pairwise Aggregation 

**Authors**: Jiuheng Lin, Chen Zhang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29481)  

**Abstract**: While reinforcement learning (RL) significantly enhances LLM reasoning, its efficacy is severely undermined by Pre-RL data overlap, where RL datasets overlap with pretraining or SFT corpora, causing models to exploit shortcuts by memorizing correct answers and fabricating post-hoc reasoning. To address this, we introduce HIPPO, a novel RL framework that integrates hint-injected aggregation with a tailored pairwise reward model. By utilizing hint injection to deliberately trigger overlap-induced behaviors, the resulting traces naturally serve as explicit anchors for pairwise comparison. This provides highly discriminable preference signals, enabling a lightweight judge model to reliably distinguish genuine reasoning deduction from shortcut-driven rationalization, while the pairwise formulation ensures stable and robust optimization compared to standard PRMs. Extensive experiments demonstrate that HIPPO yields substantial improvements over standard baselines and generalizes effectively to out-of-distribution general tasks, showing it extracts authentic, transferable reasoning skills rather than superficial shortcut patterns. 

---
# mamabench and mamaretrieval: Benchmarks for Evaluating Medical Retrieval-Augmented Generation in Maternal, Neonatal, and Reproductive Health 

**Authors**: Yi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2606.29467)  

**Abstract**: Medical question-answering benchmarks rarely cover the maternal, neonatal, child, and reproductive-health questions a nurse-midwife asks, and, to our knowledge, no public chunk-level relevance benchmark exists for maternal-health guideline retrieval. We release two benchmarks that fill these gaps. mamabench is a scope-filtered QA set of 25,949 items assembled from seven existing expert-authored sources across multiple-choice, short-answer, and rubric-graded tracks; to help users calibrate the LLM judge that scores the rubric track, we re-scope HealthBench's physician-labelled meta-evaluation to the domain. mamaretrieval pairs 3,185 clinical queries with graded (0-6) relevance labels over a 63,650-chunk maternal-health guideline corpus, using a decomposed rubric that distinguishes a chunk that answers a query from one merely on its topic. Three decisions shape both: assemble and filter expert sources rather than author questions, grade relevance rather than binarise it, and measure and disclose the limits of the labels -- scope-classifier agreement, a frontier-judge check, and a pooling-completeness audit -- rather than treat them as an oracle. A companion paper uses the benchmarks to evaluate a deployed on-device assistant; both are released openly for research. 

---
# EntroRouter: Learning Efficient Model Routing via Entropy Regulation 

**Authors**: Kaiyi Zhang, Xueliang Zhao, Zhuocheng Gong, Wei Wu, Yankai Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.29424)  

**Abstract**: Model routing balances solution accuracy and computational cost by selecting among models of varying capabilities. While recent multi-round frameworks interleave reasoning and planning, we identify a structural failure mode termed Trust Region Collapse. We demonstrate that the deep coupling of reasoning and routing, exacerbated by the dominance of strong pre-training priors under sparse supervision, leads to degenerate local optima where capable experts are systematically suppressed. To decouple these processes, we propose $\textbf{EntroRouter}$, a single-round routing framework that treats entropy regulation as a core objective. We first initialize the policy via Soft Supervision, fitting a distribution of suitable models to establish a high-entropy prior for exploration. Subsequently, we stabilize Reinforcement Learning using a Soft Anchor, which utilizes offline capability estimates to orchestrate controlled entropy contraction within a safe trust region. Extensive experiments demonstrate that EntroRouter retains 98.3% of the strongest expert's accuracy while reducing computational costs by 48.25%. 

---
# LC-ICL: Label-Guided Contrastive In-Context Learning for Robust Information Extraction 

**Authors**: Xiao You, Tianwei Yan, Shan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29407)  

**Abstract**: There has been increasing interest in exploring the capabilities of advanced large language models (LLMs) in the field of information extraction (IE), specifically focusing on tasks related to named entity recognition (NER) and relation extraction (RE).Although researchers are exploring the use of few-shot information extraction through in-context learning with LLMs, they tend to focus only on using correct or positive examples for demonstration, neglecting the potential value of incorporating incorrect or negative examples into the learning this http URL this paper, we present LC-ICL a novel few-shot technique that leverages both correct and incorrect sample constructions to create in-context learning demonstrations. This approach enhances the ability of LLMs to extract entities and relations by combining positive samples with negative samples annotated by error-cause labels. These labels expose more detailed error features in erroneous examples, enabling the model to understand why similar predictions fail and avoid repeating such errors during this http URL, our proposed method taps into the inherent contextual information and valuable information in hard negative samples and the nearest positive neighbors to the test and then applies the in-context learning demonstrations based on LLMs. Our experiments on various datasets indicate that LC-ICL outperforms previous few-shot in-context learning methods, delivering substantial enhancements in performance across a broad spectrum of related tasks. These improvements are noteworthy, showcasing the versatility of our approach in diverse scenarios. 

---
# Cross-Temporal Sinhala OCR: Page-Level Adaptation and Diachronic Analysis 

**Authors**: Avisha Dilhara, Nevidu Jayatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2606.29378)  

**Abstract**: Sinhala is a morphologically rich abugida spoken by roughly 16 million people in Sri Lanka, and to date, there are no publicly available real-world datasets for page-level Sinhala OCR. All previous studies for assessing Sinhala OCR models have used artificially generated data. To bridge the gap, we introduce sinhala-ocr-lk-acts-1010, an annotated dataset of 1,010 page-level images and their transcriptions collected from Sri Lankan Legislative Acts published between 1981-1989 and 2000-2019, split into 707 training examples, 101 validation examples, and 202 testing examples. Three models based on deep learning-based visual language processing, namely DeepSeek-OCR V1, DeepSeek-OCR V2, and LightOnOCR-2-1B, are fine-tuned using QLoRA in 8 experiments conducted on consumer and cloud GPUs. LightOnOCR-2-1B is the top performer, achieving a CER of 1.05% across all test examples, outperforming state-of-the-art open-source OCR models such as Surya-OCR (8.84%) and Tesseract v5 (10.69%), as well as commercially available OCR models such as Google Document AI (2.06%). Our results suggest that LightOnOCR-2-1B outperforms other baselines on real-world OCR tasks and maintains consistent performance across all print periods, even when documents are severely degraded. 

---
# TriageRA-CCF: Source-Side Clinical Confidence and Coverage Signals for Adaptive Rank Budgeting in Medical LLMs 

**Authors**: Shucan Ji, Yining Huang, Hongliang Guo  

**Link**: [PDF](https://arxiv.org/pdf/2606.29375)  

**Abstract**: Medical large language models are commonly adapted with a fixed low-rank budget, even though medical questions differ substantially in confidence, clinical coverage, and cross-domain difficulty. We study adaptive rank budgeting for parameter-efficient medical question answering: for each question, the adapter decides whether to activate a small, medium, or large subset of LoRA rank channels. The central challenge is that a naive adaptive budget router can collapse to unstable choices or spend capacity without improving shifted benchmarks. We propose TriageRA-CCF, a source-side teacher for adaptive rank-budgeted LoRA. It combines three signals computed only from source training data: base-model answer confidence, metadata-cell clinical coverage, and a counterfactual close-miss proxy. These signals supervise a straight-through budget router over active ranks {2,4,8}, together with budget-cost, entropy, and rank-balance regularization. Under a matched CMB-source training protocol, TriageRA-CCF achieves the best average accuracy among LoRA, DoRA, and MoELoRA baselines on both Qwen3-8B and Llama3.1-8B. The gains are modest and non-uniform across benchmarks: +0.21 average points over the strongest external baseline on Qwen3-8B and +0.16 on Llama3.1-8B. Component ablations show that confidence, coverage, and counterfactual signals all provide useful budget supervision, but their combination is not monotonically best on every backbone. 

---
# A Hybrid Framework for Song Lyric Annotation Based on Human-LLM Alignment 

**Authors**: Rashini Liyanarachchi, Frank Tran, Md Mahmudul Hasan, Aditya Joshi, Erik Meijering  

**Link**: [PDF](https://arxiv.org/pdf/2606.29273)  

**Abstract**: Emotion recognition of song lyrics is a challenging task since lyrics may not necessarily align with the overall emotion of a song. As a result, lyrics annotation remains largely underexplored. Drawing inspiration from research in large language model (LLM) assisted annotation, we examine the alignment between humans and LLMs for annotation of lyrics by creating a new sentence-level dataset of lyrics. Our observations highlight the subjectivity of the task and the inherent challenges. Following this, we present a hybrid annotation framework that optimizes human and LLM annotation by predicting potential misalignment in annotation. 

---
# MIThinker: A Plug-and-Play Policy-Optimized Thinker For Motivational Interviewing Counseling 

**Authors**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2606.29265)  

**Abstract**: Reasoning large language models (LLMs) have recently made much progress in complex problem-solving, leveraging internal reasoning (or thought) to guide their solution generation. However, existing LLM-based counseling agents, including those using Motivational Interviewing (MI), generate responses without explicitly aligning thoughts with counseling techniques, limiting their effectiveness. We propose MIThinker, a lightweight thinking model that generates therapeutic thoughts to guide MI counseling agents in strategy selection and response generation. To overcome the lack of annotated thought data, we introduce AugR1-MI, an automated pipeline that reverse-engineers counselor's thoughts from observed responses. Through two-stage training combining supervised fine-tuning and reinforcement learning, MIThinker demonstrates improved theory-of-mind assessment and strategy alignment. Comprehensive evaluations show that MindfulMI, our agent leveraging MIThinker, achieves MI competency comparable to state-of-the-art systems with an order of magnitude less computation. 

---
# Travel-Oriented Reasoning Large Language Model via Domain-Specific Knowledge Graphs 

**Authors**: Vignesh Ram Nithin Kappagantula, Shayan Hassantabar, Samuel Simpson, Golnaz Moallem  

**Link**: [PDF](https://arxiv.org/pdf/2606.29254)  

**Abstract**: Large language models (LLMs) demonstrate broad reasoning abilities but struggle with accuracy and reliability in specialized domains such as travel, where reasoning depends on precise definitions, rules, and expert-defined conceptual frameworks, and where confident but unfounded outputs arise from a reasoning failure in which the model has not internalized the underlying domain graph rather than from missing domain knowledge alone. We propose a modular pipeline for building a travel-domain reasoning LLM grounded in an expert-designed knowledge graph (KG). Our pipeline integrates a travel KG that encodes domain entities and their relationships, a bottom-up construction procedure that walks the KG to produce multi-hop question answer (QA) pairs, a supervised fine-tuning stage that embeds the domain knowledge into a reasoning-capable LLM using the generated QA pairs as auditable reasoning traces, and a travel-domain benchmark dataset that measures the fine-tuned model's accuracy and calibration. We evaluate our approach using Qwen3-4B with LoRA adaptation. Our reasoning model achieves an $82.4\%$ exact match on the benchmark. This performance significantly outperforms the pretrained Qwen3-4B baseline at $22.4\%$. A calibration analysis decomposes the residual $17.57\%$ of errors into two distinct failure modes: an over-confident multi-label decoder that predicts both correct answers plus one spurious option on most dual-answer mistakes, and a smaller reasoning failure on single-answer questions where the supporting facts are present in the KG but the model fails to reconstruct the correct multi-hop path. This split confirms that explicit KG-grounded reasoning substantially improves the accuracy and uncertainty interpretation of LLMs in specialized domains, and isolates per-option calibration and trace-length-aware decoding as the next axes of improvement. 

---
# Understanding Evaluation Illusion in Diffusion Large Language Models 

**Authors**: Hengxiang Zhang, Jiaxi Ren, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.29228)  

**Abstract**: Despite the capability of parallel decoding, diffusion large language models (dLLMs) require many denoising steps to maintain generation quality, motivating recent research on efficient decoding strategies. However, existing studies have reported inconsistent evaluation results even under seemingly identical evaluation settings, risking biased conclusions about dLLM decoding methods. To understand this evaluation concern, we conduct a rigorous evaluation of current decoding methods for dLLMs across diverse evaluation settings. Surprisingly, our analysis reveals that the ranking of decoding methods is highly sensitive to the choice of prompt templates. Single-template evaluation can lead to an illusion that decoding methods improve inference efficiency without performance degradation. Through comprehensive experiments, we find that current parallel decoding methods consistently underperform the single-token decoding baseline, failing to overcome the speed-quality trade-off. We further identify this evaluation inconsistency as the high sensitivity of parallel decoding methods to minor variations in prompt templates. Our experiments show that an effective prompt template can achieve strong evaluation results even with fewer denoising steps, markedly outperforming the marginal gain from increasing denoising steps. Beyond prompt templates, our experiments indicate that overlooked evaluation settings can also notably affect the assessment of decoding methods. Based on these findings, we propose practical guidelines for the reliable evaluation of decoding methods in dLLMs. 

---
# Can OCR-VLMs Read Devanagari? A Stress-Test Benchmark and Post-Correction Study 

**Authors**: Aditya Pratap Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.29213)  

**Abstract**: OCR systems, ranging from classical engines to specialised OCR vision-language models (OCR-VLMs) and frontier multimodal LLMs, report strong results on English and Chinese document benchmarks, yet their behaviour on Indic scripts is largely uncharacterised. We benchmark ten systems on Devanagari (Hindi): classical EasyOCR; open VLMs (Qwen2.5-VL-3B, Qwen3-VL-8B, olmOCR-7B); specialised OCR-VLMs (DeepSeek-OCR, Unlimited-OCR); and frontier closed models (Gemini 2.5 Flash, Claude Opus 4.7, GPT-5.5, Mistral OCR), across four synthetic degradation conditions and 300 real printed scans. We report four findings. First, on clean rendered text all ten cluster within chrF++ 91 to 98, so synthetic text does not separate them. Second, under degradation the specialised OCR-VLMs are the most fragile: DeepSeek-OCR suffers rare but catastrophic repetition failures (outputs up to 71 the reference length) that wreck its corpus mean even though its median is the best of any system, which is why we report median and catastrophic-rate instead of the mean. Third, on real scans nine of the ten systems collapse (EasyOCR falls from chrF++ 93.6 to 58.3) and the field spreads across a 76-point range, so synthetic renders badly overstate Devanagari quality. Fourth, strong English OCR does not predict Indic OCR: GPT-5.5 drops to chrF++ 58.5 (tying classical EasyOCR) and olmOCR-7B, the model behind olmOCR-Bench, falls to 40.5, while the open Qwen3-VL-8B (75.2, runnable on a single 24 GB GPU) beats GPT-5.5 and approaches Mistral; Gemini and Claude lead at 86.3 and 82.2. An error taxonomy separates surface errors (numerals, punctuation) from structural ones (conjuncts, matras, nukta), and a byte-level (ByT5) post-corrector improves a cheap engine on its own error distribution (chrF++ +1.2 to +1.5) but does not transfer across engines. We release the benchmark, code, and models. 

---
# DistilledGemma: Balanced Efficiency-Accuracy for Person-Place Relation Extraction from Multilingual Historical Articles 

**Authors**: Youssef Aboelwafa, Ahmed Samir, Nagwa Elmakky, Marwan Torki  

**Link**: [PDF](https://arxiv.org/pdf/2606.29130)  

**Abstract**: We present DistilledGemma, an efficient and accurate system for the HIPE-2026 shared task on person-place relation extraction from multilingual historical newspaper articles in English, German, and French. Our approach adopts a three-stage knowledge distillation pipeline designed to balance classification accuracy with computational efficiency. In the first stage, we systematically explored prompt engineering strategies across eight large language models to identify the most effective reasoning architecture for this challenging task. In the second stage, we applied supervised fine-tuning (SFT) via QLoRA to a Gemma 4 26B A4B teacher model, leveraging its strong multilingual capabilities to generate silver-standard chain-of-thought traces across the training corpus. In the final stage, we performed response-level distillation to transfer these learned reasoning patterns into a compact Gemma 4 E2B student model. In the official evaluation, our team WHEREAMI ranked 3rd on the standard test set with an accuracy profile mean score of 0.688, and 2nd on the binary test set with a mean score of 0.8156. Notably, by distilling knowledge from the 26B teacher to the 2.3B student, we preserved strong reasoning capabilities while reducing the deployed model size to approximately 2.3B effective parameters; the LoRA adapters used during training were merged into the student for inference. This configuration ranked 2nd in the balanced efficiency-accuracy profile across both the standard and binary test sets. These results demonstrate that knowledge distillation provides a practical and scalable solution for historical document processing, achieving competitive performance without excessive computational cost. 

---
# How Anthropomorphic Language Impacts Public Perceptions of AI 

**Authors**: Betty Li Hou, Sophie Hao, Sunoo Park, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29121)  

**Abstract**: Public discourse about artificial intelligence (AI) often uses anthropomorphic language: language that attributes human capabilities and characteristics to the system. This practice has been criticized for setting misleading expectations, inflating claims, and fueling hype around AI, which may distort public understanding of AI and impact policy priorities. We study the effects of anthropomorphic framing by comparing changes in participants' perceptions (N=815) when reading passages with and without anthropomorphic language, designed to reflect realistic public-facing AI discourse. We further examine whether these effects differ across two types of AI technologies -- large language models and recommendation systems -- and measure changes in perceptions of AI across several dimensions that are prominent in current public discourse. In a separate condition using a text that explicitly discusses the dangers of AI, we show that individuals' views of AI can shift in response to reading a text; yet in the main conditions of the experiment, where we compare anthropomorphic and non-anthropomorphic descriptions, we find that whether the text uses anthropomorphic language does not substantially affect participants' perceptions of AI. Our results indicate that any immediate effects on public opinions of AI are modest, although they leave open the possibility that anthropomorphic language could have an effect in naturalistic settings, or over gradual, continued exposure. 

---
# Knowing in Advance When an Evolutionary Outer Loop Will Not Help: A Pre-Registered Cheap-Baseline Screening Rule 

**Authors**: Ramchand Kumaresan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29119)  

**Abstract**: We introduce a pre-registered screening rule that decides, before any implementation, whether an evolutionary / population / lifecycle outer loop over neural-network parameters or structure is worth building. Such outer loops cost 10^2-10^3x their gradient inner loop, yet whether they beat a cheap single-shot alternative is usually discovered only after the expense is paid. Our rule computes, at a Phase-0 gate, a single number: the recovery R = s/G, the best single-shot gradient/curvature statistic's gain s divided by the best gain G of any cheap method evaluated, and prescribes skipping the outer loop when R >= 90%. We validate the rule on a within-lab series of pre-registered outer-loop bets (two analyzed cases plus a disclosed file drawer): in both analyzed cases a static or single-shot computation captured the effect on the project's own metric, the gate fired (R approximately 1.0 in both cases; approximately 0.95 under a stricter metric on one), and the outer loop was abandoned, including one case where a companion factorial decomposition localizes the apparent win to a static substrate change with the evolutionary lifecycle contributing no detectable gain. On one project the gate cost about 50-70 GPU-hours and screened out an estimated 400+ GPU-hours (first cell only) plus weeks of implementation, a 6-8x saving. The rule is prospectively falsifiable: a task with R < 90% where the outer loop still fails to beat single-shot would refute it. 

---
# AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering 

**Authors**: Ansh Kamthan  

**Link**: [PDF](https://arxiv.org/pdf/2606.29090)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become the standard way to ground large language models in external knowledge, yet most systems retrieve a fixed number of passages for every question regardless of its difficulty. This wastes computation on easy questions, starves hard ones, and gives no signal for when a generated answer can be trusted. With a growing share of question answering systems built on top of commercial language model APIs, a method that can decide how much to retrieve, and how far to trust its own answers, without retraining the underlying model, is of clear practical value. This paper presents AB-RAG (Adaptive Budgeted Retrieval-Augmented Generation), a training-free and backbone-agnostic framework that generates an answer, estimates its confidence from a combination of three signals, and then decides whether to stop or to retrieve more evidence, subject to a fixed retrieval budget. The estimator combines the model's own certainty, the agreement between the answer and the evidence, and the variance of the retrieval scores. For models that expose token probabilities the certainty signal is read directly; for closed APIs it is approximated by self-consistency, so the method works without access to model internals. Across three backbones and two datasets, the central result is that the confidence estimate reliably separates correct from incorrect answers on every backbone, reaching a clean split of 57.6% against 0% Exact Match between high- and low-confidence answers on a factoid dataset. The adaptive policy improves accuracy on capable backbones, and the study reports its negative and nuanced findings honestly, including a confidence signal that proved unsuitable for short answers and a retrieval signal whose sign was found and corrected through measurement. The entire study was carried out on a single consumer laptop with only a few dollars of API spend. 

---
# Evolution Fine-Tuning: Learning to Discover Across 371 Optimization Tasks 

**Authors**: Young-Jun Lee, Seungone Kim, Minki Kang, Alistair Cheong Liang Chuen, Zerui Chen, Seungho Han, Taehee Jung, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29082)  

**Abstract**: Would experience designing faster GPU kernels also help close in on a long-standing open mathematical conjecture? Large Language Models (LLMs) integrated into evolutionary search have recently produced state-of-the-art solutions on optimization tasks, including open mathematical conjectures, GPU kernel design, scientific law discovery, and combinatorial puzzles. To achieve this, prior work applied search scaffolds to one target task at a time, so every new problem is approached from scratch and the experience accumulated during search is discarded once the model finishes its attempt. This leaves the capability of iteratively evolving a solution (e.g., knowing which part to mutate and how, deciding when to backtrack) entirely in the scaffold rather than in the model itself. Whether the model itself could acquire this capability and reuse it across different tasks has been largely unexamined. To address this, we introduce Evolution Fine-Tuning (EFT), a mid-training paradigm that teaches LLMs to evolve solutions across tasks by converting evolutionary search trajectories into supervision. We construct Finch Collection, a 156K-trajectory dataset spanning 10 domains and 371 optimization tasks, and fine-tune open-source LLMs from 2B to 9B parameters. Empirically, EFT confers cross-task generalization: across 22 held-out tasks, our models surpass their base counterparts by 10.22% on average. Furthermore, when paired with test-time RL, our model matches state-of-the-art performance on two circle-packing tasks and outperforms its base-model counterpart on the Erdős minimum-overlap problem. EFT thus serves as a "practice phase" for general-purpose discovery agents that do not solve new problems from scratch. 

---
# A Comparative Study on Affective Cues in Text Embeddings Across Psychological Emotion Theories 

**Authors**: Fabio Ciani, Harald Schweiger, Emilia Parada-Cabaleiro, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2606.29068)  

**Abstract**: Text encoders are known for their utility in natural language processing, as they are able to efficiently compress inputs into dense vectors while preserving semantics. These models have been applied to affective computing, in particular to help with solving sentiment analysis and emotion recognition tasks. Nevertheless, it remains unclear to what extent the latent representations produced by modern text encoders capture well-defined psychological theories of affect. In this work, we investigate the affective capabilities of twelve recently released text encoders by probing their generated embeddings as input features for solving regression and classification tasks across three established emotion frameworks, using both word- and sentence-level data. Additionally, we apply a semantic data-leakage prevention technique to improve robustness in word-level evaluations. Our main findings show that the latent manifolds of the latest instruction-aware open-weight encoders enclose an equal or even a larger amount of affective information in comparison with proprietary counterparts when evaluated at word level. In contrast, embeddings of task-tuned and proprietary encoders reach the highest scores on sentence-level affective classification. Furthermore, a qualitative analysis of latent representations and their encoded affective cues is provided. 

---
# ThinkProbe: Beyond Accuracy -- Structural Profiling of Open-Ended LLM Reasoning Traces via Non-Generative Thought Graphs 

**Authors**: Mohamed Amine Kerkouri, Simon D. Hernandez, Marouane Tliba, Yann Dauxais, Maha Ben-Fares, Pierre Holat  

**Link**: [PDF](https://arxiv.org/pdf/2606.29067)  

**Abstract**: We present ThinkProbe, a framework for structural analysis of LLM reasoning traces. ThinkProbe converts each trace into a Thought Graph a directed graph with cycles, 8 node types, and 6 edge types and derives a 19-metric five-dimensional cognitive profile (5D-CP: Breadth, Depth, Structure, Metacognitive, Efficiency) through a fully non-generative pipeline combining rule-based segmentation and discriminative semantic linking. Applied to 4{,}200 traces from 7 native reasoning models across 200 open-ended questions and 10 cognitive domains, ThinkProbe reveals that reasoning structure is a stable, model-level property: between-model variance exceeds between-domain variance by up to fourfold across four of five cognitive dimensions, with Structure showing genuine sensitivity to question domain, exposing qualitatively distinct cognitive profiles invisible to accuracy-based evaluation. 

---
# Masked Diffusion Decoding as $x$-Prediction Flow 

**Authors**: Weitian Wang, Lianlei Shan, Shubham Rai, Cecilia De La Parra, Akash Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2606.29066)  

**Abstract**: Masked diffusion language models (MDLMs) generate text by iteratively unmasking tokens, but their standard decoder reduces each step to a binary action: a position is either committed to a single token or left fully masked, with no representation of partial belief in between. This all-or-nothing regime discards rich predictive information and forces premature, irrevocable commitments, leading to poor performance under a limited decoding budget. In this paper, we reinterpret mask prediction as clean-state prediction ($x$-prediction) and show that it can be used to induce a continuous flow in input embedding space. Building on this view, we propose a continuous decoding framework for MDLMs where tokens can accumulate partial progress at each diffusion step and remain revisable. To match the uneven contextual constraints across positions in language, we replace the globally synchronous schedule in image diffusion with a confidence-based asynchronous update in which the diffusion progress is token-wise accumulated. Additionally, we introduce a lightweight policy network and formulate its training as a reinforcement learning problem. Applied to pretrained LLaDA, our continuous decoder reaches 97% of its performance on the HumanEval dataset with 25% of decoding budget. 

---
# The strength of clinical evidence is recoverable from language model representations but not from their stated grades 

**Authors**: Soroosh Tayebi Arasteh  

**Link**: [PDF](https://arxiv.org/pdf/2606.29034)  

**Abstract**: Large language models (LLMs) increasingly summarize clinical evidence, where a claim's weight depends on how strongly it is supported. Yet these models convey confidence poorly, and properties they never state, such as truth, are often readable from their activations. Whether a clinical model registers evidence strength, distinct from truth, and states it when asked is untested, and any such signal could be lexical. We compiled 45,134 clinical claims from six public sources, harmonized 20,611 into a four-level evidence grade under three independent frameworks, and tested 22 local, open-weight LLMs from several developers (0.6-70 billion parameters; general, medical, and reasoning), with lexical, truth, and cross-framework controls. A linear estimator recovered the grade in every model (median AUROC 71.8), yet decodability did not rise with scale and was weakest in reasoning models. The grade the models stated fell to chance, 25-27 percentage points below the estimator. The recoverable signal was largely lexical and did not transfer across topics or frameworks, yet it was distinct from factual truth and still flagged weakly supported claims (AUROC 69.2). Clinical LLMs thus carry an ordered evidence-strength signal they do not express, so their stated grades fail to convey a claim's support even when it is recoverable from their representations and text. 

---
# How to Leverage Synthetic Speech for LLM-Based ASR Systems? 

**Authors**: Yanis Labrak, Dairazalia Sanchez-Cortes, Sergio Burdisso, Séverin Baroudi, Shashi Kumar, Esaú Villatoro-Tello, Srikanth Madikeri, Manjunath K E, Oldřich Plchot, Kadri Hacioğlu, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2606.29031)  

**Abstract**: In regulated domains such as banking and healthcare, where privacy constraints make real speech costly to collect and retain, synthetic speech from modern text-to-speech (TTS) is an appealing alternative for training automatic speech recognition (ASR) without exposing sensitive customer recordings. Yet a persistent distributional gap between synthetic and real data limits how far it can replace genuine recordings. Prior work largely treats this gap as a black box to be engineered around, but in our work, we instead examine its origin directly by probing a SLAM-ASR architecture. Then, we localise where its LLM backbone separates real from synthetic speech and find the discriminative signal concentrated in the early-to-middle layers, where temporal and prosodic perturbations disrupt it most. We further show that representation-level separability, help, but does not directly predict downstream ASR gains. On the other hand, convolving synthetic audio with room impulse responses (RIRs) narrows the gap not by making synthetic speech sound cleaner or more natural, but by reproducing the acoustic irregularities of real recordings. Translating these findings into the training procedure, by adding a layer-selection module combined with RIR augmentation matches a fully real-data baseline using only 25% of the real speech (13.6h) and surpasses it at all higher proportions. 

---
# Conversational Domain Adaptation of IndicTrans2 across 21 Indic Languages via Experience Replay and Model Soups 

**Authors**: Aditya Pratap Singh  

**Link**: [PDF](https://arxiv.org/pdf/2606.29024)  

**Abstract**: IndicTrans2 is the strongest open English to Indic translation system, but like most systems it is trained on general text and tends to sound stiff on casual, conversational input. We adapt IndicTrans2-1B to conversational register across all 21 Indic languages using only public data (OpenSubtitles, BPCC-H-Daily, Tatoeba). Plain fine-tuning improves conversational chrF but forgets the general domain (it drops 3.9 chrF on FLORES for Hindi). Mixing general data back into training (experience replay) and then averaging the fine-tuned weights with the base (model souping) removes that trade-off: the resulting model beats IndicTrans2-1B on conversational chrF in every one of the 21 languages (mean +6.2) while matching it on FLORES (mean change -0.17, all within 0.7 chrF). Paired bootstrap tests confirm the conversational gains are significant (p <= 0.004) and that FLORES is not significantly degraded. We are deliberate about scope: these are chrF gains, and a blind human plus multi-model LLM check does not confirm them as a perceived quality improvement, so we treat the conversational gain as largely a register match to the references rather than proof of better translation. The techniques are not new; the contribution is the honest, end-to-end study in the Indic conversational setting. 

---
# BERTomelo: Your Portuguese Encoder Best Friend 

**Authors**: Rennê Ruan Alves Oliveira, Gustavo Cordeiro Galvão Van Erven, Luís Paulo Faina Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2606.28999)  

**Abstract**: Encoders have become the state of the art for multiple NLP tasks, especially those requiring deep contextual understanding. While multilingual models offer broad coverage, dedicated monolingual encoders are essential for capturing the unique lexical and syntactic nuances of specific languages. For Portuguese, however, existing monolingual options like BERTimbau and Albertina have not kept pace with recent architectural breakthroughs, often lagging behind English benchmarks in scalability and efficiency. This work introduces BERTomelo, a next-generation monolingual encoder pre-trained from scratch and specifically optimized for the Portuguese language. By leveraging the ModernBERT architecture, BERTomelo overcomes the limitations of previous models, offering Base and Large versions with a 1,024-token context window and hardware-level optimizations like FlashAttention and alternating attention mechanisms. The model was trained on ClassiCC-PT, a massive, high-quality Portuguese corpus of 106 million documents, ensuring superior alignment with the language's contemporary usage. The results demonstrate that BERTomelo not only outperforms previous Portuguese encoders but also provides a more robust and efficient alternative to massive multilingual models in downstream tasks such as STS and NER. 

---
# Fine-Tuning General-Purpose Large Language Models for Agricultural Applications:A Reproducible Framework and Evaluation Protocol Based on Qwen3-8B 

**Authors**: Zhaoyang Li, Ruijie Zhang, Jiaqi Liu, Zhaoji Sun  

**Link**: [PDF](https://arxiv.org/pdf/2606.28992)  

**Abstract**: General-purpose large language models (LLMs) have demonstrated strong abilities in opendomain question answering, information extraction, and text generation. Agricultural applications, however, are domain-specific, region-dependent, time-sensitive, and safety-critical. Without data governance, expert evaluation, and evidence constraints, an agricultural assistant mayproduce unreliable advice on crop diseases, pesticide use, fertilization, or policy this http URL avoid presenting unverified simulated numbers as real experimental findings, this paper doesnot report any model-performance claims that have not been produced by an actual training runand expert evaluation. Instead, we propose AgriTune-R, a reproducible and auditable frameworkfor adapting general-purpose LLMs to agricultural tasks. The framework selects the publiclyverifiable Qwen3-8B model as the recommended base model and integrates agricultural datagovernance, instruction construction, LoRA/QLoRA parameter-efficient fine-tuning, retrievalaugmented generation, expert evaluation, and safety control for high-risk questions. The contributions are: (1) a structured workflow for agricultural LLM adaptation; (2) an evaluationprotocol for agricultural knowledge QA, pest and disease consultation, cultivation management,and policy explanation; (3) an expert-review rubric combining factuality, safety, evidence consistency, and uncertainty expression; and (4) a clear separation between protocol design andempirical conclusions, providing an executable baseline for future empirical studies. 

---
# Can LLMs Hire Fairly? Racial Bias in Resume Screening 

**Authors**: Zhenyu Gao, Wenxi Jiang, Yutong Yan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28978)  

**Abstract**: We audit fourteen mainstream large language models (LLMs) for hiring discrimination using the paired-resume methodology of Kline, Rose, and Walters (2022). The sole 2023-vintage model reproduces the pro-White callback gap documented in field experiments on labor market discrimination ($+2.12$ pp, significant at the 1\% level). Every model released in 2024 or after shows either a null gap or a significant pro-Black reversal (up to $-3.01$ pp). The same pattern holds on the gender axis. Based on 24,024 paired postings per model across 14 models, our results document a reversal in the direction of algorithmic hiring bias across model generations. 

---
# Beyond the Mean: Three-Axis Fidelity for Aligning LLM-Based Survey Simulators from Small Pilot Data 

**Authors**: Eun Cheol Choi, Youngrae Kim, Prabhu Pugalenthi, Hong-En Chen, Bo-Ruei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28963)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate social survey responses, yet their outputs exhibit systematic biases: marginal distributions are skewed, response variance is poorly calibrated, and predictor-outcome relationships are attenuated. We ask a simple question: given a small pilot sample of human responses, can an LLM recover the statistical characteristics of a broader population? We decompose recovery along three axes: structural fidelity, marginal fidelity, and individual fidelity. Using a COVID-19 misinformation survey as a case study, we benchmark three families of approaches: prompting, rectification, and fine-tuning. The findings suggest that fine-tuning on small pilot samples offers a balanced approach for achieving multiple forms of fidelity, but the levels of such fidelity can vary across subsamples, potentially threatening pluralistic alignment. 

---
# A3M: Adaptive, Adversarial and Multi-Objective Learning for Strategic Bidding in Repeated Auctions 

**Authors**: Junhan Li, Yuxin Zhang, Haoran Wang, Minghao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28943)  

**Abstract**: Learning to bid in repeated multi-unit auctions with bandit feedback poses a fundamental challenge. Existing methods often rely on rigid explore-then-exploit schedules, assume stationary adversaries, and optimize solely for bidder utility, thereby limiting adaptability and strategic robustness. To address these limitations, we introduce the A3M framework, which integrates adaptive deep reinforcement learning (DRL), explicit adversarial reasoning, and principled multi-objective reward design for online auction strategy optimization. A3M employs an actor-critic DRL backbone to dynamically balance exploration and exploitation, an opponent model for fictitious play against non-stationary adversaries, and a composite reward function to jointly maximize utility, auctioneer revenue, and fairness. We provide the first comprehensive empirical evaluation of this integrated approach against established baselines in both discriminatory and uniform price auctions. Results show that A3M reduces final regret by 30--40\% in standard settings, maintains robust performance against adversarial strategy shifts, scales favorably with the number of units $K$, and enables tunable multi-objective trade-offs. An extensive ablation study confirms the necessity of each core component. Our work establishes A3M as a powerful and flexible framework for learning in complex auction environments. 

---
# EVLA: An Electro-Aware Multimodal Assistant for Physically-Grounded Driving Reasoning and Control 

**Authors**: Yuxin Liu, Zihan Chen, Haoyu Wang, Mingxuan Zhang, Ruijie Lin, Siyuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.28938)  

**Abstract**: Modern vision-language models (VLMs) for driving assistants typically treat vehicle dynamics as a black box, resulting in decisions that lack awareness of the vehicle's real-time electro-mechanical state. To bridge this gap, we introduce the Electro-Visual-Language Assistant (EVLA) -- a novel framework that combines multi-modal scene understanding with real-time perception of the electrified powertrain state (e.g., motor torque, battery SOC). Our approach features two key innovations: first, a Unified Co-State Encoder (UCSE) that fuses visual, textual, and vehicle-state inputs into a shared latent representation, augmented with an Energy-Efficiency Field to model spatial energy costs; and second, an Electro-aware Structured Reasoning Chain (ESRC), which replaces external chain-of-thought prompting with an internal, deterministic reasoning process grounded in physical constraints and optimization objectives. Trained end-to-end with a physics-guided joint loss, EVLA learns to generate context-aware and energy-optimal driving decisions. Extensive evaluations on a driving QA benchmark demonstrate that EVLA substantially outperforms strong fine-tuned VLM baselines, improving the final score by +0.0871 and accuracy by +5.6\%. Ablation studies validate the necessity of each component, and efficiency analyses show that EVLA achieves 36\% faster inference than multi-stage pipelines. This work underscores that integrating vehicle-state awareness and structured physical reasoning is crucial for developing next-generation, physically-grounded driving assistants. 

---
# FinInvest-GTCN: Explainable Graph-Temporal-Causal Modeling for Risk-Aware Investment Decision Optimization 

**Authors**: Junyan Tan, Yifan Li, Minghao Wang, Zihan Chen, Haoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28933)  

**Abstract**: Venture capital (VC) investment decisions face distinct challenges, such as multi-source heterogeneous data, non-stationary time series, and the demand for explainable predictions in high-stakes, low-data settings. To overcome these issues, we introduce \textbf{FinInvest-GTCN}, a Graph-Temporal-Causal Network that redefines the task from content recommendation to quantitative risk-return assessment. This architecture combines a relational graph encoder to capture the investment ecosystem's topology, a multi-scale temporal fusion module to handle long-term dependencies and non-stationarity, and a causal decision head that generates risk-adjusted predictions with interpretable causal attributions. A core innovation is the Meta-Causal Adaptation (MCA) strategy, which facilitates robust fine-tuning for new, data-scarce sectors by aligning updates with causally-plausible structures derived from meta-pretraining. Comprehensive experiments on proprietary VC datasets show that FinInvest-GTCN delivers state-of-the-art results, markedly lowering the primary Risk-Adjusted Mean Squared Error (RA-MSE) to 2.51 from a baseline of 3.05 and boosting the cumulative return of a simulated portfolio by 18.7\%. Ablation studies underscore the essential role of each component, while additional analyses confirm the model's stability, interpretability, and enhanced adaptability. This work pioneers a data-driven, explainable framework for investment decision support. 

---
# Latent Bridges for Multi-Table Question Answering 

**Authors**: Simone Varriale, Tamara Cucumides, Floris Geerts, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2606.28916)  

**Abstract**: We introduce GRAB, a constructor-encoder-bridge pipeline for table question answering. Our method lifts relational data into an heterogeneous graph, encodes it via message passing, and transfers the signals to an LLM through a small set of query-conditioned latent tokens. This provides the LLM with a compact, task-relevant structural representation together with the flattened text. Crucially, the LLM remains strictly frozen to preserve its general reasoning capabilities; we train only the lightweight graph encoder and latent bridge (91M parameters), allowing the entire pipeline to be trained efficiently. Our pipeline significantly improves performance on relational Question Answering, with the largest gains in demanding multi-table settings, offering an efficient, principled way to connect relational deep learning with LLMs. 

---
# PASTA: A Paraphrasing And Self-Training Approach for Knowledge Updating in LLMs 

**Authors**: Takayuki Yamamoto, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2606.28898)  

**Abstract**: Knowledge updating in pre-trained Large Language Models (LLMs) remains an important challenge. While continual training provides a potential avenue for knowledge updating, it continues to present substantial technical difficulties. Furthermore, LLMs often struggle with accurately answering questions about specific factual information, such as news articles - a capability limitation widely recognized in the research community. This paper proposes PASTA, a simple yet powerful framework for integrating detailed factual information from news articles as new knowledge into LLMs, with the primary goal of building specialized models that accurately answer questions about this knowledge. Our framework combines data augmentation, question-answering generation, and a novel self-learning DPO process that simultaneously enables knowledge overwriting and hallucination suppression. We provide insights into effective knowledge updating through systematic analysis of learning parameters and data configurations. In our experimental evaluation with web articles published after the base model's knowledge cutoff, PASTA achieved remarkable improvement from 0.02 to 0.82 accuracy while maintaining general language capabilities, demonstrating its effectiveness for creating domain-specialized LLMs. 

---
# Memory-Managed Long-Context Attention: A Preliminary Study of Editable Request-Local Memory 

**Authors**: Junyi Zou, Avrova Donz  

**Link**: [PDF](https://arxiv.org/pdf/2606.28876)  

**Abstract**: Long-context language models often conflate two different goals: compressing history into an efficient state, and maintaining reliable long-term memory. Linear, recurrent, and sparse attention reduce the cost of processing long sequences, but they do not by themselves specify when a fact should be written, overwritten, protected from distractors, or discarded. We study memory-managed long-context attention, a research route that separates a fast recurrent or sparse backbone from explicit editable request-local memory slots and query-time sparse fallback. Across structured synthetic tasks, token/chunk/sequence bridges, generated natural language, and local frozen-model diagnostics, pure fixed-state or pure sparse methods fail some overwrite, version, anti-pollution, or no-write-signal cases, while a hybrid covers both routes. A small 2,097,152-token mechanism stress test reaches 50/50 pooled accuracy with 2-132 active chunks. A 2.74M-parameter minimal causal event-token model reaches 595/600 with lite write supervision, supporting proof of trainability rather than scale. A six-family frozen-hidden-state bridge reaches 1079/1080 controlled pointer accuracy, but it uses generator-provided integer key IDs and separately encoded canonical key strings; it is an oracle-metadata probe, not open-text entity resolution. Local non-leaderboard RULER 4K diagnostics remain close to full context, whereas a 33-record LongBench v1 16K subset shows that naive lexical selection is not general. The evidence separates three claims: controlled slot lifecycle is feasible, sparse fallback is needed when writes lack future-query signals, and learned open-domain selection remains the main architectural bottleneck. We do not claim a final generative architecture, global slot-trajectory convergence, or systems superiority. 

---
# Open but Incompatible: A License Compatibility Analysis of Corpora for Low-Resource African Languages 

**Authors**: Ernst van Gassen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28867)  

**Abstract**: Creative Commons licenses dominate African NLP corpus releases, but their compatibility rules are rarely applied. CC-BY-SA and CC-BY-NC cannot be combined in a single published dataset; a NoDerivs clause silently prohibits tokenisation and annotation. This paper audits the license provenance of over twenty corpus families used in African NLP, constructs a six-tier compatibility matrix, and applies it to three case-study languages: Kituba/Munukutuba, Zarma, and Moore. Four failure modes are documented with primary-source evidence: outright prohibition (JW300, removed from OPUS after a legal audit confirmed Terms of Service violation); composite license misrepresentation (WAXAL, whose CC-BY 4.0 claim is contradicted by its own HuggingFace dataset card); a NoDerivs clause hidden behind a CC-BY label (Tanzil); and data persistence failure (the Congolese Radio Corpus, where 402 of 405 source URLs are now dead). A pre-annotation due diligence checklist and a survey of legally clean enrichment opportunities close the paper. 

---
# The Heterogeneous Safety Impacts of Benign Multilingual Fine-Tuning 

**Authors**: Will Hawkins, Kaivalya Rawal, Jonathan Rystrøm, Stratis Tsirtsis, Zihao Fu, Greta Warren, Ryan Brown, Eoin Delaney, Sandra Wachter, Brent Mittelstadt, Chris Russell  

**Link**: [PDF](https://arxiv.org/pdf/2606.28843)  

**Abstract**: Fine-tuning a large language model is a ubiquitous method for enhancing its capability on a specific downstream task. However, prior work has shown that this increase in capability comes with a cost: it can increase a model's tendency to respond to unsafe adversarial prompts, even when fine-tuning with non-adversarial data. We present the first comprehensive empirical study of this phenomenon in multilingual settings by fine-tuning Llama-3.2, Qwen3, and Gemma-3 models using benign data translated across nine languages. We find that safety outcomes are highly sensitive to both the choice of fine-tuning language and the evaluation language, with adversarial compliance rates increasing four-fold in some settings. Multilingual safety drift is decoupled from general capability metrics, and occurs heterogeneously across languages and models. Fine-tuning in non-English languages often induces smaller internal representational drifts than English, but these shifts lead models to default to either exaggerated compliance or refusal. As such, assessing fine-tuning impacts solely in English provides inadequate assurance for deployment. To facilitate further research into these cross-lingual safety blind spots, we release the Multilingual-Benign-Tune dataset and the SORRY-Bench-Multilingual evaluation suite. 

---
# Labeling Training Data for Entity Matching Using Large Language Models 

**Authors**: Aaron Steiner, Christian Bizer  

**Link**: [PDF](https://arxiv.org/pdf/2606.28823)  

**Abstract**: Recent large language models (LLMs) achieve strong performance on entity matching without requiring task-specific training data. However, applying these models to large sets of candidate pairs remains slow and costly. In contrast, entity matchers using traditional machine learning methods or small language models (SLMs), such as RoBERTa, offer much faster inference but require task-specific training data.
This paper investigates whether the need to provide task-specific training data can be avoided by using knowledge-distillation workflows, in which an LLM serves as a teacher model to label training pairs that are subsequently used to train a smaller student model. We investigate knowledge distillation for entity matching along the following dimensions: pair-selection strategy, teacher model, label post-processing method, and student model. We evaluate the workflows using the Abt-Buy, Walmart-Amazon, WDC Products, DBLP-ACM, and DBLP-Scholar benchmarks, and compare the performance of student models trained with machine-labeled data to the performance of the same models trained using the benchmark training sets.
Our experiments show that student models trained using the machine-labeled sets perform approximately on par with models trained on the benchmark training sets, with the remaining differences in both directions staying below two F1 points. Using GPT-5.2 to label the training sets for all five benchmarks costs US\$28.31 to US\$40.88, whereas manually labeling the same training sets is estimated to require 470 hours of work. At inference time, Ditto is 41.5 to 534 times faster than directly using an LLM to perform the matching tasks.
These results indicate that current LLMs, when combined with a suitable pair-selection method, can substantially reduce or even eliminate the manual effort required to label use case-specific training data for entity matching. 

---
# Structure-Preserving Document Translation via Multi-Stage LLM Pipeline: A Case Study in Marathi 

**Authors**: Manasi Waghe, Danish Chandargi, Mohammad Aamir Rayyan, Raviraj Joshi, A.R. Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2606.28796)  

**Abstract**: Government documents in India are predominantly issued in regional languages such as Marathi, creating substantial accessibility barriers for non-native readers, interstate administrative bodies, and policy analysts. Although recent advances in neural machine translation have improved sentence-level translation quality, existing systems largely neglect document structure, formatting integrity, and domain-specific terminology, thereby limiting their applicability to official documentation. This paper presents a structure-preserving Marathi-to-English government document translation framework capable of performing end-to-end document transformation while maintaining layout fidelity. The proposed system integrates layout-aware optical character recognition, coordinate-based text extraction, large language model based translation, and structured document reconstruction through HTML representations. By enforcing spatial alignment constraints and preserving hierarchical document elements, the framework ensures structural consistency between the source and translated documents. Experimental evaluation on real-world Marathi government PDFs demonstrates improved structural preservation, translation coherence, and terminological consistency compared to conventional text-only translation pipelines. The proposed framework contributes toward scalable multilingual accessibility solutions for e-governance and administrative document processing. 

---
# Majority Vote Silences Minority Values: Annotator Disagreement at the Hate/Offensive Boundary in HateXplain 

**Authors**: Joshua Muhumuza, Joab Ezra Agaba, Mercy Amiyo  

**Link**: [PDF](https://arxiv.org/pdf/2606.28772)  

**Abstract**: Hate speech annotation pipelines routinely collapse annotator disagreement into majority vote labels before training. We show that this aggregation is not neutral: 42.6% of all annotator disagreement in HateXplain concentrates specifically at the hate/offensive boundary, a pattern consistent with annotators applying different thresholds for where hate begins (chi-squared = 135.199, df = 2, p < 0.0001). Both a hard-label BERT model (Model A) and a soft-label model (Model B) drop 22 percentage points in accuracy from agreed posts (~80%) to disagreement posts (~58%), confirmed at p < 0.0001. A per-annotator multi-head model (Model C) widens this gap further to 28 points while collapsing offensive disagreement accuracy to 0.245. Critically, Model A expresses significantly higher confidence on boundary case errors than Model C (0.710 vs. 0.495, p < 0.0001), meaning standard evaluation metrics will not detect the failure. Three downstream interventions of increasing sophistication all fail to recover boundary accuracy. We argue the problem is structural. Majority vote presents a contested judgment as ground truth, and models inherit that false certainty. The intervention must be upstream in annotation design. 

---
# 5ting at SemEval-2026 Task 8: Strong End-to-End Multi-Turn RAG via LLM-Based Reranking and Faithfulness Control 

**Authors**: Thien-Qua-T-Nguyen, Chi Hoang, Nguyen Tran, Tri Le, Khanh Truong, Chinh Trong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28737)  

**Abstract**: We introduce 5ting, our system for the SemEval2026 Task 8 (MTRAGEval), which evaluates multi-turn Retrieval Augmented Generation (RAG) systems. Multi turn RAG involves context drift, under specification, and hallucination risk. Our system combines BGE-M3 dense retrieval with FAISS indexing, dual-query merged retrieval, and LLM based reranking, followed by role separated generation constrained to retrieved evidence. The retriever achieved nDCG@5 = 0.4719 in Task A, while the end to end system ranked in Task C with a harmonic score of 0.5597 and RL_F = 0.7692. 

---
# DriftGuard: Safety-Aware Multi-Monitor Detection and Selective Adaptation for Evolving Toxicity Moderation 

**Authors**: Yuting Xin, Hanyu Cai, Binqi Shen, Lier Jin, Lan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28725)  

**Abstract**: Automated toxicity moderation systems operate in dynamic online environments where harmful behavior evolves through coded language, shifting targets, and strategic adaptation to enforcement. Existing drift detection methods often focus on global distributional change, but such signals may miss safety-relevant shifts that emerge in localized harm subspaces or high-risk model-error regions. This paper introduces DriftGuard, a safety-aware adaptive moderation framework that combines multi-monitor drift detection with selective model updating. The framework tracks global text drift, identity-harm drift, model uncertainty, toxic-risk drift, and false-negative-risk drift. When safety-relevant change is detected, the model is updated using a hard-mix adaptation set that prioritizes likely false negatives, identity-related high-risk examples, false-positive-risk examples, and uncertain boundary cases. Experiments on Civil Comments temporal shift and Jigsaw-to-DynaHate cross-dataset shift show that safety-aware monitors detect risks missed by global drift alone. Hard-mix adaptation improves toxic recall and accuracy over no-update and random-balanced baselines, raising toxic recall to 0.8777 on Civil Comments and from 0.7107 to 0.8523 on DynaHate. Bootstrap analysis further shows stable DynaHate safety gains, with toxic recall increasing by 0.1418 and false-negative prevalence decreasing by 0.0781. Overall, DriftGuard links safety-aware drift detection to targeted, lightweight model updating for more robust adaptive toxicity moderation. 

---
# SEATauBench: Adapting Tool-Agent-User Evaluation Into Low-Resource Southeast Asian Languages 

**Authors**: My Chiffon Nguyen, Aulia Adila, Saksorn Ruangtanusak, Kittiphat Leesombatwathana, Vissuta Gunawan Lim, Patomporn Payoungkhamdee, Samuel Cahyawijaya  

**Link**: [PDF](https://arxiv.org/pdf/2606.28715)  

**Abstract**: While AI development and evaluation for Southeast Asia (SEA) has grown rapidly, agent capabilities in regional languages are still poorly understood despite its importance to sovereign AI. To fill this gap, we introduce SEATauBench, the first agent-focused evaluation framework for SEA sovereign AI. SeaTau adapts TauBench to five languages -- Mandarin, Vietnamese, Thai, Indonesian, and Filipino -- and evaluates agents across progressively localized settings that vary the language of user-agent interaction, tool specifications, and task domains. Across three recent models, we find that English agent capabilities transfer reasonably well when only the conversation language changes, but quality and robustness degrade sharply as more task contexts are localized, with the largest losses in full domain adaptation. We also the limits of English-only agent assessment for measuring agent capabilities in SEA languages. More broadly, SeaTau provides a diagnostic benchmark and reusable adaptation pipeline for building reliable multilingual agents for linguistically diverse regions. Data and code can be accessed at this http URL. 

---
# AnTenA: Actionable and Explainable Tensor Analysis System with Large Language Models 

**Authors**: Dawon Ahn, Auder Der, Evangelos E. Papalexakis  

**Link**: [PDF](https://arxiv.org/pdf/2606.28708)  

**Abstract**: Accurately explaining hidden patterns in multi-aspect data has typically been done by leveraging labels and/or accompanying auxiliary metadata. However, labels and auxiliary data may be inaccurate (e.g. nonstandard, inconsistent), insufficient (e.g. static tabular metadata for time-dependent recordings), or unavailable. %
We propose \fullmethod (\method), which leverages the knowledge of large language models (LLMs) to explain the hidden patterns in human narratives. \method uses task-agnostic and task-specific prompts to explain extracted co-clustered latent patterns from tensor decomposition. To evaluate these explanations, we test the LLMs on forward and backward inference tasks. % Our demo system is available at this https URL. 

---
# Phonological Perception of Sign Language Models 

**Authors**: Kayo Yin, Jessica Carter, Alex Xijie Lu, Annemarie Kocab  

**Link**: [PDF](https://arxiv.org/pdf/2606.28667)  

**Abstract**: Sign languages are compositional systems where meaning arises by combining sublexical phonological parameters, such as handshape, location, and movement. While deep learning models for Sign Language Recognition (SLR) have achieved increased performance on translation benchmarks, it remains unclear whether these models distinguish abstract phonological features or merely rely on low-level statistical correlations. This work evaluates the phonological perception of SLR models trained on American Sign Language (ASL) by probing phonological sensitivity using minimal pairs and evaluating representational alignment with human behavioral data. Our results reveal that SLR models exhibit emergent phonological sensitivity, but with clear architectural trade-offs: pose-based models are sensitive to handshape contrasts, while pixel-based models better capture location changes. Furthermore, pose-based models learn latent representations that correlate with human perceptual similarity judgments (r~0.49). These findings suggest that while SLR models exhibit emergent phonology, current training paradigms are insufficient to scale them beyond their architectural inductive biases. 

---
# Correct codes for the wrong reasons? validating LLMs as measurement instruments for theoretical constructs 

**Authors**: Manuel Pita  

**Link**: [PDF](https://arxiv.org/pdf/2606.28574)  

**Abstract**: When a large language model (LLM) codes a construct in text as a human annotator would, that agreement makes the LLM a reliable coder. Yet reliability leaves construct validity untouched. The instrument may be theory-naive, reaching the code through a correlate that meets none of the demands the construct's theory makes, and no current method tells that apart from genuine measurement. We propose grain calibration as a method that closes the gap. It decomposes a construct into clause-level components, tests each against the text with extractive evidence, and combines the results through an explicit, theory-derived rule. Because the rule is stated rather than lodged in one opaque pass, its structure is evidence about the process rather than the output. It shows which components settled a code, and, when the code is wrong, whether a component was missed or an adjacent construct mistaken for it. Validation shifts from scoring an instrument's outputs against an annotator to showing that the instrument runs on the construct its theory specifies. 

---
# SEAD: Competence-Aware On-Policy Distillation via Entropy-Guided Supervision 

**Authors**: Chia-Hsuan Lee, Zelei Cheng, Yu Wang, Renkun Ni, Sambit Sahu, Shi-Xiong Zhang, William Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2606.28562)  

**Abstract**: On-policy distillation (OPD) has a property absent in offline distillation and RL: teacher supervision quality depends on student competence. Incoherent rollouts yield noisy gradients; already-mastered tokens yield redundant ones. This creates waste at three scales (tokens, training phases, and prompts) yet existing methods supervise uniformly. We introduce SEAD, which uses entropy as a unified probe of this competence-dependent degradation at three scales: (1) joint teacher-student entropy partitions tokens into zones receiving tailored divergences or zero gradient (approx. 50% skipped); (2) a cosine schedule anneals from forward to reverse KL as competence grows; (3) a competence-gated curriculum introduces prompts easy-to-hard. These components are symbiotically necessary: token selection requires coherent rollouts (curriculum), annealing requires monotonic improvement (also curriculum). On OLMo-3 (7B to 32B), SEAD achieves +4.8 avg accuracy over vanilla OPD across six math benchmarks, with ablations confirming super-additive interactions. 

---
# Depth-Staggered Fibonacci Spacing for Sparse Attention: Static Schedules Beat Learned Dilation and Extrapolate Where Dense Attention Fails 

**Authors**: Chad A. Capps  

**Link**: [PDF](https://arxiv.org/pdf/2606.28560)  

**Abstract**: We study sparse self-attention in which each query attends to a dense local window plus a set of Fibonacci-spaced offsets, with a per-layer scalar alpha that compresses or expands the spacing. Across 21 language models trained under one matched recipe (60M parameters, 512 hidden, 16 layers, 426M tokens), we compare four ways of setting alpha across depth: fixed, per-layer learned, a static linear stagger, and a coprime (anti-gridding) reassignment of that stagger, together with a reach-matched power-of-2 control. Three results stand out. First, a static per-layer stagger improves perplexity over both fixed and learned alpha, and the gain is base-agnostic: applying the same stagger to a power-of-2 base lifts it above fixed Fibonacci and to parity with learned Fibonacci attention. Second, learning per layer is inert: it does not beat the static schedule and costs roughly five times the inference latency. Third, and most consequential, all sparse variants extrapolate to four times their training length with little or no degradation, whereas a recipe-matched dense baseline collapses (perplexity rises by 201% at 4x length); we attribute this to fixed-offset attention only ever querying relative positions seen during training. We also report two honest negatives: at training length the best sparse model has about 26% higher perplexity than the dense baseline, and the staggering gain is uniform across context positions rather than concentrated at long range. 

---
# Turn-Averaged SAEs for Feature Discovery and Long-Context Attribution 

**Authors**: Kevin Der, Harish Kamath, Ben Thompson  

**Link**: [PDF](https://arxiv.org/pdf/2606.28548)  

**Abstract**: Sparse autoencoders (SAEs) have become a useful tool for extracting interpretable features in language models. However, standard SAE architectures operate on individual token activations, meaning that the number of active features scales linearly with context length, and studying long model transcripts becomes difficult. We introduce turn-averaged SAEs, which represent a single Human or Assistant turn with a fixed number of features by learning to reconstruct the average model activation across the turn. We find that turn-averaged features describe a single turn's high-level characteristics more completely than per-token features when judged by an LLM. We also demonstrate that turn-averaged SAEs greatly simplify common downstream uses of SAEs like attribution graphs. Broadly, turn-averaged SAEs make interpretability techniques practical at long context lengths. 

---
# Legal Domain Adaptation of Modern BERT Models 

**Authors**: Dominik Stammbach, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2606.28538)  

**Abstract**: We investigate domain adaptation of modern BERT models in the legal domain. We further pre-train ModernBERT on all US court opinions using the masked language modeling objective. Although ModernBERT has been trained on roughly 500x more data than original BERT, we still find that this model benefits from further pre-training and domain adaptation in the legal domain: we report significant improvements compared to vanilla ModernBERT on all datasets connected to US court opinions. We find gains similar to those reported in early work on domain adaptation of BERT-like models. However, from scratch pre-training does not match the performance of further pre-training an existing ModernBERT checkpoint in our experiments. The resulting models are capable of processing sequences up to 8,192 tokens, and can be used to compute meaningful embeddings of legal passages, or could quickly rerank hundreds of legal passages for a given search query. We release all model checkpoints publicly. 

---
# A French OSCE Dialogue Dataset and Controllable Virtual Patient System for Clinical Training 

**Authors**: Doria Bonzi, Tom Bourgeade, Fabrice Lefèvre, Irina Illina  

**Link**: [PDF](https://arxiv.org/pdf/2606.28526)  

**Abstract**: The clinical and communication skills of medical students are commonly assessed through Objective Structured Clinical Examinations (OSCEs), which consist of brief scenario-driven simulations of doctor-patient interactions. However, training is often limited by the low availability of human standardized patients, motivating the development of realistic virtual patients (VPs). To address this gap, we introduce a French OSCE dialogue dataset comprising 240 student-patient training interactions. We build upon it a controllable LLM-based pipeline to generate synthetic OSCE dialogues. The pipeline integrates modular components, such as retrieval-based grounding and a reflection loop, to ensure patient fidelity, coherence, and realism. Additionally, we propose a multi-level evaluation framework assessing patient simulation quality, student performance, and linguistic quality, using an LLM-as-a-Judge approach. Experiments suggest that controllability modules generally improve patient fidelity and student evaluation consistency. Finally, we implement an interactive prototype in which students can practice with a VP and receive automatic feedback. 

---
# Developmental Trajectories of Situation Modeling and Mentalizing in Transformer Language Models 

**Authors**: Pamela D. Rivière, Cameron Jones, Sean Trott  

**Link**: [PDF](https://arxiv.org/pdf/2606.28524)  

**Abstract**: Recent work suggests that Large Language Models (LLMs) are sensitive to the belief states of agents described by text, as measured by the false belief task (FBT), yet persistent concerns of construct validity remain. We adopt a **developmental perspective**, tracing the pattern of mental state reasoning behavior -- and likely **preconditions** for this behavior -- across multiple training stages in the Olmo2 and Pythia language model suites. We find that above-chance FBT performance depends both on model size and sufficient training volume, emerges relatively late in pretraining, and is most improved by post-training interventions (SFT, DPO) in the condition most diagnostic of mentalizing (False Belief, Implicit). However, FBT performance is fragile: consistent with past work, the use of non-factive verbs (e.g., thinks) increases false belief attributions even in the True Belief condition. To contextualize these findings, we track the emergence of **situation modeling**: the ability to report on basic factual properties of a described scene. Situation modeling accuracy generally precedes and exceeds FBT accuracy, yet situational representations also prove surprisingly incoherent in certain respects: when asked about the knowledge states of the Antagonist agent -- who always knows the item's true location -- Olmo2 13b is consistently influenced both by the Target agent's knowledge state and the presence of non-factive verbs. Together, these results suggest that larger, sufficiently trained models build partially coherent situation models in a developmentally appropriate sequence, yet display surprising fragility -- highlighting the value of developmental and stress-testing approaches for evaluating LLM capabilities. 

---
# Extracting Knowledge from an Arabic-English Machine-Readable Dictionary Using Information Extraction 

**Authors**: Diaa M. Fayed, Aly A. Fahmy, Mohsen A. Rashwan, Wafaa K. Fayed  

**Link**: [PDF](https://arxiv.org/pdf/2606.28457)  

**Abstract**: Natural language processing (NLP) applications need large and rich amount of linguistic knowledge. Furthermore, electronic language sources such as dictionaries, encyclopedia, and corpora became available. So, automatic methods are emerged to extract lexical information from those sources to overcome the knowledge acquisition bottleneck. We presented a method to automatically extract lexical information from a machine-readable version of the Arabic-English Al-Mawrid dictionary. We used n-gram analysis and key-word-in-context (KWIC) analysis to discover lexical patterns that manifest morphologic, syntactic, or semantic information. Then, we used hand-crafted rule-based information extraction to extract that information. Furthermore, we used punctuation marks and some heuristics to extract a set of synonyms in a subentry. This study registered high precision for all types of information, high recall for synonyms, and low recall for the other information. The study also showed that the Al-Mawrid has significant amount of derivations (morphologic information) and synonyms, domain labels, and hyponym/hypernym relations (semantic information). 

---
# Generating in the Limit with Infinitely Many Hallucinations 

**Authors**: Irene Strauss, Alexandra Butoi, Ryan Cotterell  

**Link**: [PDF](https://arxiv.org/pdf/2606.28354)  

**Abstract**: The classic paradigm of language identification in the limit models learning as a game between an adversary, who reveals strings from an unknown target language, and a learner tasked with identifying that language. The recently introduced framework of language generation in the limit shifted the objective to better reflect modern language modeling, requiring the learner to produce valid, unseen strings from the target language. Related work highlighted a fundamental tension: a broad coverage of the target often comes at the cost of validity. We introduce a new notion of precision and recast this problem as the classic recall-precision trade-off. We analyze generation in the limit under varying constraints on enumeration, novelty, and validity, aimed at reflecting settings closer to those encountered by large language models. A key contribution is our analysis of learners that are not eventually valid: we allow infinitely many mistakes, provided their frequency tends to zero so that precision remains one. We show that this relaxation can strictly increase recall when the adversary permanently withholds a large portion of the target language. We also study a continuous relaxation of the novelty constraint that requires only a fixed fraction of outputs to be novel. Taken together, our results move toward a more realistic model of language generation where occasional errors and repetitions are unavoidable, but their rates are controlled. 

---
# Self-Evolving World Models for LLM Agent Planning 

**Authors**: Xuan Zhang, Wenxuan Zhang, See-Kiong Ng, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.30639)  

**Abstract**: World models offer a principled way to equip long-horizon LLM agents with foresight: predictions of action consequences before execution. However, unreliable foresight can be ignored, misused, or even degrade downstream decision-making. In this paper, we introduce WorldEvolver, a self-evolving world model framework that revises its deployment-time context while keeping the downstream agent and all model parameters frozen. WorldEvolver integrates three modules: (i) Episodic Memory, which exploits real action transitions through retrieval-based simulation; (ii) Semantic Memory, which extracts persistent heuristic rules from prediction-observation mismatches; and (iii) Selective Foresight, which filters low-confidence predictions before integrating them into agent reasoning context. We evaluate WorldEvolver on ALFWorld and ScienceWorld, measuring world model prediction accuracy on Word2World and downstream agent success rate on AgentBoard. Extensive experiments show that WorldEvolver achieves the highest prediction accuracy across three backbones and leads other world model baselines on downstream agent success rate, demonstrating that test-time memory revision enhances both predictive fidelity and planning performance. 

---
# Attractor States Emerge in Multi-Turn LLM Conversations 

**Authors**: Ting-Wen Ko, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2606.30571)  

**Abstract**: Large language models (LLMs) are increasingly used in open-ended multi-agent settings, but the long-run dynamics of model--model interaction remain poorly understood. We study whether open-ended LLM discussions exhibit attractor-like behavior, i.e. topic-independent stable sets of behaviors which conversations settle into. Across 7 LLMs and 20 controversial topics, we compare self-play and mixed-play dyadic debates, tracking trajectories in representation space, discourse traits, and stances. We find self-play trajectories to be model-specific attractors that draw their conversation partners asymmetrically in mixed-play debates, influencing the other models' stylistic choices and behavior. For example, Claude Haiku is a strong attractor of other models in latent space, corresponding to other models taking on its traits like metacommentary, and models like GPT-4.1 nano are especially malleable. Our results suggest that open-ended LLM interactions are partially predictable from model-specific attractors, but shaped by structured and asymmetric partner influence. Overall, our analysis sheds some light on the complex behavior of open-ended multi-agent interaction, which we hope is helpful in designing, predicting, and monitoring autonomous agentic systems in the real world. 

---
# Situation Perception: A Necessary Primitive to Artificial Superintelligence 

**Authors**: Ziqin Yuan, Jaymari Chua  

**Link**: [PDF](https://arxiv.org/pdf/2606.30481)  

**Abstract**: Current large language models are extraordinary statistical engines. They compress vast amounts of text into useful patterns and can explain science, write code, imitate reasoning, and participate in philosophical conversation. Yet pattern mastery is not the same as general intelligence. A human infant begins with little explicit knowledge, but gradually discovers object permanence, cause and effect, other minds, bodily agency, and the persistence of the physical world. We make an argument that the path to artificial superintelligence (ASI) depends on a missing capacity we call \emph{situation perception}: the ability to construct, revise, and act within internal simulations of possible worlds across latent time. \emph{ perception} requires at least three core components: abstract prediction, long-term compressed memory, and active learning guided by objectives. In this work, we analyse why modern large language models remain incomplete, and propose the appropriate tests for measuring progress and consequences of machines that can simulate futures, pursue self-directed goals, and possibly judge their own creators. 

---
# Uncovering Salience-Driven Dynamics in Consumer Confidence with Generative Social Simulation 

**Authors**: Yixu Huang, Yunlu Yin, Jiayu Lin, Xinnong Zhang, Jia Wang, Siyuan Wang, Xuanjing Huang, Liyin Jin, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.30395)  

**Abstract**: Consumer confidence is typically modeled as a persistent macroeconomic index, yet its movements arise from households that interpret economic information through heterogeneous constraints, exposures, prior beliefs, and attention. We introduce ConsumerSim, a generative Human--Environment response framework that reconstructs Consumer Confidence Index (CCI) dynamics from a microdata-calibrated synthetic population, time-stamped macroeconomic, financial, policy, and news signals, survey-like response generation, post-stratified belief expansion, and behavioral inertia alignment. Across U.S., EU27, and Japanese official CCI target series, ConsumerSim ranks first among persistence, time-series, regression, and information-augmented baselines on the reported reconstruction metrics, with clear gains around high-salience shocks. Its reconstructed signal also improves short-horizon prediction of real activity, most consistently for housing outcomes. Mechanism analyses show that CCI movements concentrate around salient events; subgroup trajectories often align in direction while differing in magnitude; and signal sensitivity varies across income, homeownership, education, and political-alignment groups. Population-expansion and ablation results indicate that representative aggregation, situational signals, persona heterogeneity, and inertia are necessary for both accuracy and diagnosis. The findings support a behavioral view of consumer confidence as an interpretable Human--Environment response process rather than a purely aggregate time series. 

---
# MaDI-Bench: An End-to-End Data Integration Benchmark 

**Authors**: Aaron Steiner, Ralph Peeters, Christian Bizer  

**Link**: [PDF](https://arxiv.org/pdf/2606.30371)  

**Abstract**: Data integration combines heterogeneous data sets into a single, coherent representation. Data integration involves a sequence of interdependent tasks including schema matching, value normalization, entity blocking, entity matching, and data fusion. Existing benchmarks either evaluate these steps in isolation or cover only incomplete versions of the data integration pipeline, omitting specific steps. The lack of public end-to-end data integration benchmarks hinders research on data integration methods that address the integration process as a whole. This paper fills this gap by introducing the Mannheim Data Integration Benchmark (MaDI-Bench), the first benchmark for the end-to-end integration of relational tables covering all steps of the integration process. MaDI-Bench contributes (i) a set of base end-to-end data integration tasks spanning several application domains, each requiring the full schema matching, value normalization, entity matching, and conflict resolution pipeline; and (ii) a generic method for deriving task variants that mitigates rapid benchmark saturation as data integration systems advance. We validate the benchmark using human-engineered pipelines, a best-of-breed pipeline, and an LLM-based pipeline. The validation demonstrates the utility of the benchmark for measuring the step-wise as well as the end-to-end performance of data integration pipelines. All benchmark artifacts are available for public download. 

---
# When Is a Draft Accepted? A Theory of Acceptance in Speculative Decoding 

**Authors**: Aaryam Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2606.30265)  

**Abstract**: Speculative decoding accelerates language model inference by using a fast drafter to propose candidate tokens that are then verified by a larger target model. Existing theory largely studies the stochastic, distribution-preserving setting, where the goal is to exactly sample from the target distribution. In contrast, many practical systems use greedy decoding, relaxed acceptance rules, or tree-based candidate sets, where success is governed by local ranking and threshold events rather than exact distributional equality. We develop a theory for these regimes. We identify that many common acceptance criteria have rejection regions that can be characterized as lower level sets of the target distribution. For these, we characterize the exact KL divergence required for rejection yielding exact certificates and sharp margin-based bounds for strict greedy decoding, additive and multiplicative relaxed acceptance, top-(m) relaxed criteria, and entropy-thresholded acceptance. We then extend the framework to greedy tree decoding, deriving exact and margin-only certificates for when the target greedy token remains covered by the drafter's top-(m) candidates. Finally, we evaluate the resulting certificates on Qwen3 models, showing that relaxed and tree-based criteria substantially enlarge the region of certified acceptance, especially on decoding steps with low target model distribution margin. These results complement existing distribution-preserving analyses of speculative decoding by characterizing the deterministic local acceptance events common in practical inference systems. 

---
# EvalSafetyGap: A Hybrid Survey and Conceptual Framework for LLM Evaluation-Safety Failures 

**Authors**: Buğra Alperen Uluırmak, Rifat Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2606.30219)  

**Abstract**: LLM evaluation and AI safety face a shared measurement problem: benchmark scores, reward-model signals, and reported safety metrics can improve while the latent properties they are meant to represent remain difficult to verify. This paper combines a hybrid survey - a systematic search paired with narrative synthesis and separately tracked grey evidence - with a conceptual framework and a structured ten-model audit. The synthesis spans eight evidence streams: benchmark validity, dynamic evaluation, LLM-as-judge reliability, safety evaluation, jailbreak/refusal robustness, reward hacking, mechanistic interpretability, and governance/auditability, covering 2018-2026 evaluation-safety measurement work. We introduce EvalSafetyGap as an organizing hypothesis for comparing evaluation-side and alignment-side proxy failures under optimization pressure, using Goodhart's Law together with two constructs we develop here - an Instability Decomposition and an Alignment Trilemma - as tools for generating testable comparisons. The audit shows how conclusions shift when capability, behavioral safety, and governance are measured separately. In this sample (n = 10), the association between capability and sustained adversarial robustness is statistically indeterminate using the displayed Table 3 inputs (Pearson r = +0.232, p = 0.520), and the apparent open-closed safety gap is modest, driven mainly by governance and disclosure rather than behavioral robustness, and sensitive to how a single borderline model is classified; attempt-budget results are protocol dependent. Because the public evidence uses heterogeneous protocols, the audit is diagnostic rather than rank-generating. The contribution is a shared vocabulary and evidence map to support dynamic evaluation, transparent source reporting, multi-attempt safety measurement, and auditable alignment practice. 

---
# SHOVIR: A Benchmark for Evaluating Vision Shortcut Learning in Radiology Report Generation 

**Authors**: Filippo Ruffini, Marco Salmé, Rosa Sicilia, Valerio Guarrasi, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2606.30201)  

**Abstract**: Current evaluation protocols for Vision-Language Models (VLMs) in Radiology Report Generation (RRG) rely on report-level metrics that measure lexical overlap or aggregate clinical correctness. However, such metrics do not test whether individual diagnostic statements stem from the actual pathological evidence visible in the image. This allows models to achieve competitive scores by exploiting learned priors or spurious correlations, a failure mode we refer to as vision shortcut. We introduce SHOVIR, a benchmark for evaluating vision shortcut behavior in RRG. SHOVIR extends two spatially annotated chest X-ray datasets, MIMIC-CXR and PadChest-GR, with per-box CheXpert labels, and defines image-level and disease-level occlusion experiments that contrast baseline performance on clean images against localized, region-specific perturbations. Comparing predictions across these conditions isolates two failure modes at the disease-class level: direct shortcuts, where a finding persists after its visual evidence is removed, and contextual shortcuts, where detection degrades once co-occurring pathologies are occluded despite the target region remaining intact. Benchmarking eight state-of-the-art VLMs, we find that shortcut behavior varies substantially across architectures and datasets. Models achieving the highest baseline report quality do not necessarily rank highest in spatial grounding, revealing that clinically fluent generation can coexist with shallow reliance on visual evidence. These findings expose a blind spot in current RRG evaluation and motivate region-aware assessment protocols. 

---
# DNA Language Models: An Assessment of Pre-Training for Fine-Tuning Tasks 

**Authors**: Romain Karpinsky, Julien Mozziconacci, Mickaël Delcey  

**Link**: [PDF](https://arxiv.org/pdf/2606.30140)  

**Abstract**: Recent breakthroughs in foundation models and Large Language Models (LLMs) have introduced new opportunities for studying and decoding genomic sequences. Several state-of-the-art approaches, such as DNABERT2, rely on transformer-based architectures, while others, such as ConvNova, still build upon more conventional convolutional models. However, systematic benchmark comparisons across these methods remain scarce. Given that transformer-based models require extensive and costly pretraining, it is crucial to evaluate whether their performance gains justify this overhead. Moreover, LLMs such as DNABERT2 typically rely on Byte Pair Encoding (BPE) tokenization, whose relevance for DNA sequence representation is still debated within the genomics community. In this work, we investigate three key questions: (i) do transformer-based models provide sufficient improvements on fine-tuning tasks upon heavy pretraining, (ii) what is the actual contribution of pretraining in this setting, and (iii) how does BPE tokenization impact performance on genomics-related tasks? 

---
# Does Verbose Chain-of-Thought Really Help? In-Distribution Evidence that Content, Not Length, Matters 

**Authors**: Wenlong Wang, Fergal Reid  

**Link**: [PDF](https://arxiv.org/pdf/2606.30128)  

**Abstract**: Chain-of-thought (CoT) prompting improves LLM reasoning, but the source is contested: do the intermediate steps help because they carry useful semantic content, or because conditioning on more tokens buys extra computation before the model commits to an answer? We bring two lines of evidence to bear. First, in distribution: we repeatedly sample each model on the same question and pair a shorter with a longer of its own natural generations that follow the same reasoning plan, so nothing is rewritten and both traces are genuinely in-distribution. Across 25 models the extra tokens leave accuracy essentially unchanged for every independently-trained reasoner, and a blind analysis of the surplus tokens shows that what gain exists elsewhere tracks validation- and checking-content, not verbosity per se. Second, as a controlled intervention, we ask whether two traces expressing the same semantic content (the same facts, operations, and intermediate values, verified through directed acyclic graph equivalence) produce different outcomes when one is more verbose, using a dual-validator design across four targets and eight benchmarks with number-redacted completion and stratified bootstrap confidence intervals. Verbose traces do improve accuracy (25 of 32 benchmark-target cells are positive under at least one validator), but the effects are modest (typically 1-4 points) and depend on the quality of the verbose prose, not merely its length. Under maximum numerical redaction the effect is amplified (median 3.24x across four arithmetic benchmarks), and length-matched non-reasoning filler recovers none of it. Both lines converge: what matters is what the extra tokens do (the reasoning and validation content they carry), not how many there are, a picture neither a pure forward-pass-compute nor a pure semantic-content account fully explains. 

---
# Know Before You Fetch: Calibrated Retrieval-Budget Allocation for Retrieval-Augmented Generation 

**Authors**: Zhe Dong, Fang Qin, Manish Shah, Yicheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29959)  

**Abstract**: Retrieval-augmented generation (RAG) typically retrieves a fixed number of passages for every query. This is wasteful when the reader already knows the answer, and it can be harmful when irrelevant or partially relevant passages distract the reader. We formulate adaptive RAG as calibrated retrieval-budget allocation: given a query, decide whether to answer closed-book, retrieve a compact context (k=1), retrieve a full context (k=5), or abstain. The contribution is a probability interface rather than a new raw uncertainty signal. We calibrate sequence log-probability and prefix-logit uncertainty signals into probabilities of correctness, then use these probabilities for graded context selection, selective abstention, and explicit latency/token trade-offs. Across core QA experiments on TriviaQA, Natural Questions, and MS MARCO, with auxiliary PopQA motivation and Qwen/Llama family checks, diagnostic out-of-fold calibration improves probability quality dramatically: for sequence log-probability, ECE drops from 0.275 to 0.062 on TriviaQA, 0.643 to 0.009 on NQ, and 0.711 to 0.031 on MS MARCO. Graded retrieval improves full-context and passage-budget frontiers for both our signal and TARG-style prefix entropy/margin, while retrieval-call AUC remains essentially tied with binary gating because k=1 is still a retrieval call. Held-out train/validation/test threshold experiments report deployable operating points. At matched-accuracy frontier operating points, a measured cost model reveals that gating is not universally faster: it increases latency by about 27% on Qwen3-8B but saves about 8% on Qwen3-32B. These results support a nuanced view of adaptive RAG: calibrated confidence is best understood as a reusable interface for allocating retrieval budget under task and system constraints. 

---
# SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics 

**Authors**: Nikolay Georgiev, Maria Drencheva, Kseniia Ibragimova, Ivo Petrov, Dimitar I. Dimitrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2606.29894)  

**Abstract**: As agentic AI systems tackle more complex mathematical tasks, they increasingly rely on information retrieval (IR) to search problem databases, theorem libraries, and educational resources. However, choosing the right retriever remains difficult, as it is infeasible to directly isolate its effect on downstream performance. On the other hand, existing retrieval-specific benchmarks often fail to capture fine-grained mathematical relevance, penalizing relevant documents. We address this gap by introducing SABER-Math, the first fully automated benchmark for evaluating mathematical IR without expert annotation. Starting from 283K high-school-level math problems with solutions, SABER-Math builds challenging reranking tasks in three steps: (i) first, LLMs extract concise solution summaries and mathematical topics for each problem; (ii) then, per-query relevant documents are discovered using ontology topic-based and lexical solutions-summary-based similarities, and (iii) finally, a Swiss-style LLM preference tournament produces fine-grained relevance ratings for the documents. We evaluate lexical retrievers, specialized mathematical retrieval systems, and recent embedding models. We find that while modern embedding models substantially outperform classical and math-specific baselines, even the strongest systems struggle in symbol-heavy domains like Algebra and Calculus. Importantly, we show that general-purpose IR benchmarks such as MTEB do not reliably predict mathematical performance, especially for recent embedding models, highlighting the need for math-specific retrieval benchmarks. 

---
# Unveiling Novelty Evolution in the field of Library and Information Science in China 

**Authors**: Chen Yang, Yuzhuo Wang, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29872)  

**Abstract**: This study analyzes the novelty distribution of scholarly papers in the field of Library and Information Science (LIS) in China, with a focus on differences across journals, research topics, and time periods. Articles published in Chinese LIS journals indexed by the Chinese Social Sciences Citation Index (CSSCI) from 2000 to 2022 were collected as the research sample. BERTopic was applied to paper abstracts to identify research topics, and novelty scores were calculated based on the combinatorial innovation theory of reference pairs cited by focal papers. The study then examined the novelty of papers under different topics and further analyzed author collaboration patterns to explain how collaboration may be associated with paper novelty. The results show that archival research topics generally have lower novelty, whereas topics related to journal evaluation and patent technology display higher novelty in Chinese LIS research. Overall, the novelty of papers in this field has gradually increased over time. Papers with different topics and novelty levels also show distinct collaboration patterns: low-novelty topics are more often associated with solo authorship, while high-novelty topics tend to involve a higher proportion of inter-institutional collaboration. This study reveals the topic-level characteristics and temporal trends of novelty in Chinese LIS research and provides a new perspective for understanding how research topics and collaboration patterns influence scholarly innovation. 

---
# Mandol: An Agglomerative Agent Memory System for Long-Term Conversations 

**Authors**: Yuhan Zhang, Zhiyuan Guo, Ziheng Zeng, Wei Wang, Wentao Wu, Lijie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29778)  

**Abstract**: Long-term conversational agents need to remember and query cross-session, multi-typed information with complex correlations. Existing agent memory systems rely on heterogeneous vector and graph databases, which fragment memory information and cause high cross-database I/O latency. For retrieval, common RAG-style methods tend to introduce noise, miss correlated clues, and lack token budget control, degrading LLM accuracy and efficiency.
We propose Mandol, an agglomerative memory system that consolidates fragmented memory representations and storage into a unified memory-native architecture. Its core components include: (1) a hierarchical memory model that organizes memory into a basic layer representing raw memory information and a high-level abstract layer that agglomerates basic memories into traceable abstract memories, both uniformly represented as structured semantic graphs; (2) an agglomerative semantic data structure combining SemanticMap and SemanticGraph, which natively fuses key-value, vector, and graph structures and provides unified hybrid retrieval operators to eliminate cross-database I/O; and (3) a quantitative query mechanism with query-adaptive routing, quantitative denoising and conflict resolution, and token-constrained context generation, all without involving LLMs during retrieval. Experiments on two widely used long-term conversation benchmarks, LoCoMo and LongMemEval, show that Mandol achieves the best overall accuracy among representative agent memory systems. For performance comparison, Mandol also obtains a 5.4x retrieval speedup and a 4.8x insertion speedup under 10 QPS concurrent load, while still maintaining low latency on consumer-grade hardware. 

---
# The Hidden Cost of Resampling: How Imbalance Correction Degrades Probability Calibration in Tree Ensembles 

**Authors**: Zewen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29720)  

**Abstract**: Resampling methods such as SMOTE and random under/over-sampling are standard tools for class-imbalanced classification, almost always evaluated by minority-class accuracy or F1. Prior work has established that undersampling degrades probability calibration by distorting the training prior [1]. We extend this lens to synthetic oversampling (SMOTE) and provide a practical, evidence-based guide to when calibration damage matters and how to fix it. Across five public datasets (imbalance ratio 1.9-70) and two ensemble models (random forest, gradient boosting), with ten seeds and paired statistics, we find: (1) SMOTE's calibration cost is real but small (ECE +0.009; Cliff's delta = +0.27, small-to-moderate) across the studied imbalance range (IR 1.9-70) and its discrimination gains typically outweigh the calibration penalty; (2) random undersampling is the genuine danger -- its damage grows sharply with imbalance, inflating ECE from 0.008 to 0.395 on a dataset with ratio 70, largely because the resulting training sets are too small to estimate probabilities reliably; (3) a single post-hoc recalibration step (Platt or isotonic) eliminates the damage, reducing ECE by up to 66% at a negligible ranking-power cost (AUC -0.002, Cliff's delta = -0.07); and (4) the analytic prior-shift correction that repairs undersampling does not transfer to SMOTE, because SMOTE distorts the class-conditional density rather than only the prior -- so data-driven recalibration remains necessary. We recommend that imbalanced-learning studies report calibration alongside discrimination, and that practitioners recalibrate after resampling whenever predicted probabilities drive decisions. 

---
# A Diagnostic Framework and Multi-Evaluator Audit of Evaluator-Driven Preference Dynamics in Self-Adapting LLM Agents 

**Authors**: Liu Zewen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29719)  

**Abstract**: Measurements of proprietary LLM evaluators can become invalid within weeks -- we document one case and provide the diagnostic framework to detect it. We introduce EPC -- comprising the Multimodal Preference Collapse Index (MPCI), evaluator-indexed coupling matrix, and Jensen-Shannon divergence (JSD) -- and apply it across eight experimental conditions (N=112 main + N=10 ablation = 122 unique repetitions, all reported). Coupling coefficients range from 0.00 to 1.18 across per-condition means (CV approx 0.9, n=8 conditions). Four conditions show strong coupling (N=36; GPT-4o May, GPT-4o-mini, Qwen3.7-plus, DashScope 30r); four collapse to near-zero (N=76; GPT-4o June, qwen-plus N=30, symmetric LR, DeepSeek self-eval). The May-to-June GPT-4o drift -- an N=8 re-replication inverting the study's conclusion -- is the most informative measurement: a diagnostic instrument detecting its own instability demonstrates the fragility it was designed to measure. Self-evaluation (97% zero, JSD=0.003) consistently collapses, though floor effects are possible. Output-format confound analysis finds per-strategy aggregate rho=0.89 but per-instance rho=0.219 (p=0.093); PCI reported as preference-convergence metric. We release EPC with all data. The finding is not any single coupling magnitude but the pattern of version-conditional instability that makes single-snapshot evaluator studies unreliable. 

---
# Diagnosing and Mitigating Context Rot in Long-horizon Search 

**Authors**: Shijie Xia, Yikun Wang, Zhen Huang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29718)  

**Abstract**: Extensive context has become the norm as Large Language Models (LLMs) are increasingly deployed in long-horizon tasks. The concern that increasing context length degrades model capabilities, known as context rot, has become a central issue for these applications. In this paper, we focus on deep search scenarios, aiming to investigate the rot phenomenon and its mitigation strategies. By evaluating four flagship open-source models across three benchmarks, we reveal a prevalent but unnoticed rot phenomenon: extensive context causes models to directly give up or prematurely provide uncertain answers, and this issue is exacerbated as the context grows. Through pruning experiments, we demonstrate the relationship between the accumulated context and the rot phenomenon. Furthermore, we investigate mitigating this issue through context management and post-hoc rejection sampling. For context management, we systematically evaluate seven different methods across three categories, based on performance, cost, and impact on context rot, providing clear guidance for strategy selection and usage. For rejection sampling, we develop a rot-aware filtering strategy and demonstrate its effectiveness across three aggregation methods. Finally, we show that these two approaches can be combined for further performance improvements. 

---
# ARMOR: Adaptive Retriever Optimization for Low-Resource Telecom Question Answering 

**Authors**: Heshan Fernando, Quan Xiao, Yan Xin, Tianyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.29706)  

**Abstract**: Telecom question answering (QA) is a challenging setting for retrieval-augmented generation (RAG): evidence is fragmented across standards, papers, encyclopedic resources, and web documents, and answers often hinge on technical tables, equations, and specialized protocol language. In low-resource subdomains, generator fine-tuning can over-specialize and degrade general capability, making query-side retriever adaptation an attractive alternative. To this end, we ask whether a fixed-generator, query-adapted RAG system can outperform generator-side adaptation, and which retriever objectives best support that setting. We motivate retrieval, rather than generator fine-tuning, as the adaptation target through a capacity comparison: under bounded-parameter and soft-retrieval assumptions, query-encoder tuning can have a smaller estimation term than supervised fine-tuning when its effective dimension is smaller. We identify two particularly relevant objectives -- the latent-document RAG likelihood, which optimizes generation utility, and the InfoNCE contrastive objective, which improves semantic retrieval geometry -- and leverage them jointly through a retriever optimization method targeting downstream QA performance in the telecom domain. Specifically, we introduce ARMOR, Adaptive Regularized Mixture Optimization for Retrievers, which learns separate temperatures for the RAG retrieval distribution and InfoNCE softmax and regularizes the adapted query encoder toward the frozen base query encoder. Across telecom-specific retrieval and generative QA benchmarks, we show that ARMOR improves evidence retrieval and answer generation in several in-domain settings. Code is available at this https URL. 

---
# GUICrafter: Weakly-Supervised GUI Agent Leveraging Massive Unannotated Screenshots 

**Authors**: Sunqi Fan, Lingshan Chen, Runqi Yin, Qingle Liu, Yongming Rao, Meng-Hao Guo, Shi-Min Hu  

**Link**: [PDF](https://arxiv.org/pdf/2606.29705)  

**Abstract**: Data, as the fundamental substrate of modern intelligence, has greatly driven the development of current foundation models. Naturally, researchers aim to extend this paradigm to the domain of GUI agents, hoping to build strong GUI agents through a similar paradigm. However, GUI agent data cannot be directly harvested from the internet, making it costly and difficult to collect at scale. As a result, current GUI agents suffer from poor cross-device generalization and limited visual grounding ability for fine-grained GUI elements. As an attempt to address data challenge in GUI agents, we propose GUICrafter, a weakly-supervised GUI agent leveraging massive unannotated screenshots to substantially reduce the reliance on expensive human annotations. GUICrafter explores a curriculum learning framework for training GUI agents through two progressive stages. First, the model learns visual grounding from large-scale unannotated screenshots and webpages, leveraging the rich contextual signals inherent in GUI interactions without human annotations. Then, in Stage 2, we leverage a small amount of high-quality data to calibrate the model via reinforcement learning. Experiments show that GUICrafter achieves competitive, or even superior, performance to advanced systems like UI-TARS while using only 0.1% of its data. Furthermore, under the same amount of annotated data, GUICrafter surpasses all previous methods such as GUI-R1. Code, data, and models are available at this https URL. 

---
# SurrogateShield: Beyond Redaction for High-Utility, Privacy-Preserving LLM Interactions 

**Authors**: Sherwin Vishesh Jathanna  

**Link**: [PDF](https://arxiv.org/pdf/2606.29567)  

**Abstract**: LLM-based assistants transmit user queries verbatim to third-party API endpoints that lie outside the user's audit or control. When those queries contain personally identifiable information (PII), the data persists on remote infrastructure subject to breach, subpoena, or policy change. Placeholder redaction (the prevailing mitigation) suppresses PII at the cost of semantic coherence, producing structurally degraded queries and correspondingly degraded responses.
We present SurrogateShield, a client-side proxy that substitutes detected PII with locally generated, type-consistent surrogate values prior to transmission and restores originals in the response. No real PII crosses the network boundary. Detection runs through a three-stage cascade (PatternScan, EntityTrace, and ContextGuard) covering 22 PII types and quasi-identifier combinations grounded in Sweeney's k-anonymity framework. Surrogate-to-original mappings are sealed in an AES-256-GCM encrypted per-conversation ShadowMap that never leaves the device.
Evaluations on a 1,124-query corpus demonstrate that the cascade reliably detects PII, achieving an overall F1 score of 98.87%. Surrogate substitution substantially outperforms placeholder redaction in semantic utility, yielding a 13.26 pp improvement in BERTScore (roberta-large), from 81.59% to 94.85%. Within this corpus, the local pipeline restricted real PII transmission across all tested query types; in a 100-query adversarial trial, a prompted LLM adversary recovered no original values from surrogate-substituted messages. 

---
# Em-ergence of the em-dash: a population-level rise in em-dash frequency in medRxiv preprints at the dawn of the large-language-model era 

**Authors**: Przemysław Czuma  

**Link**: [PDF](https://arxiv.org/pdf/2606.29540)  

**Abstract**: Large language models (LLMs) can leave subtle stylistic traces in assisted text; one of the most cited is the em-dash (Unicode U+2014). Yet no one has measured whether em-dash use has changed in the scientific literature. This study, pre-registered on the Open Science Framework (HFT8C), used the full set of medRxiv full-text XML preprints from the official Text-and-Data-Mining resource. The primary cohort was first, original versions deposited 2020-2025 with an extractable Discussion section of at least 500 characters (N = 69,632). The primary endpoint was the presence of at least one em-dash in the Discussion; the principal measure was the absolute change in its prevalence between the pre-ChatGPT era (before 30 November 2022) and the post-ChatGPT era, estimated with a logistic model with standard errors clustered by first author. The analysis plan (six supporting analyses, six sensitivity analyses, two falsification tests) was frozen before any confirmatory result was computed. Em-dash prevalence in Discussion sections rose from 4.23% before ChatGPT to 11.58% afterward, an absolute increase of 7.35 percentage points (95% CI 6.94-7.77; odds ratio 2.96, 95% CI 2.77-3.17). The rise was not a sharp jump but a gradual, delayed acceleration: near 4% through 2023, 8.0% in 2024, and 20.3% in 2025. The effect survived every feasible sensitivity analysis (7.35-7.60 pp) and both falsification tests; a placebo split within the pre-LLM era showed no meaningful change (+0.13 pp, 95% CI -0.33 to +0.58), and was essentially absent in boilerplate sections. Independent LLM-associated lexical markers and within-paper section comparisons pointed the same way. The em-dash is a population-level indicator, not a per-paper detector of LLM use, and the design cannot establish causality; it shows that something in how scientific literature is written changed markedly in the early 2020s, and roughly when. 

---
# Do Models Read What They Write? Causal Registers in Scratchpad Reasoning 

**Authors**: Benjamin Shih, John Winnicki, Eric Darve  

**Link**: [PDF](https://arxiv.org/pdf/2606.29522)  

**Abstract**: A central hope behind process supervision is that models can expose intermediate variables that matter for their later behavior. For this to help with alignment, a scratchpad must be tied to the computation: when the model writes a state, later steps should compute from that state. To test this requirement, we use a controlled state-tracking task with a known update rule, comparing models trained to report only the final state with models trained to write intermediate states before giving the final answer. At evaluation, we edit the internal representation of one written state while leaving the visible scratchpad text fixed. Because the transition rule is known, the edit has a single correct downstream consequence. In Qwen2.5-Coder-7B, the state-writing model predicts the next phase bit implied by the edited state on 80% and 91% of held-out examples across the two task variants, while pretrained and final-answer-only controls remain near baseline. Additional controls rule out generic next-token steering and copying another continuation: the prediction depends on both the edited state and the current move. The same causal-use pattern replicates across model families. Together, these results suggest a sharper goal for scratchpad oversight: not just to make intermediate reasoning legible, but to train written states that the model uses as part of its computation. 

---
# UCOB: Learning to Utilize and Evolve Agentic Skills via Credit-Aware On-Policy Bidirectional Self-Distillation 

**Authors**: Songjun Tu, Chengdong Xu, Qichao Zhang, Yiwen Ma, Yaocheng Zhang, Linjing Li, Dong Li, Xiangyuan Lan, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2606.29502)  

**Abstract**: Skill memories can improve agentic reinforcement learning by reusing past experience as textual guidance, but retrieved skills are not oracular: they may help in one state while misleading the same policy in another. This makes the common privileged-teacher assumption fragile, namely that a skill-conditioned prompt can be treated as a fixed teacher for the no-skill prompt. We introduce UCOB, a framework for learning to utilize and evolve agentic skills via credit-aware on-policy bidirectional self-distillation. UCOB treats skill-conditioned and no-skill prompts as two on-policy context views of the same model, compares their return-to-go within the same task and anchor state, and uses the higher-return view as the local teacher. This local credit signal internalizes useful skill-conditioned behavior, corrects misleading skill usage, and guides task/state skill memory updates, utility-aware retrieval, and reflection self-training. Experiments on agentic tasks, including ALFWorld, WebShop, and Search-QA, show that UCOB outperforms skill-free RL, skill-memory baselines, and self-distillation methods across model scales, with up to 23.5 and 18.0 point gains over SOTA baselines on ALFWorld and WebShop. Ablations and analyses further validate its core mechanisms and efficiency. 

---
# Interpretable Inverse Design of Metal-Organic Frameworks with Large Language Model Agents 

**Authors**: Kyungmin Nam, Seunghee Han, Jihan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2606.29459)  

**Abstract**: Inverse design of metal-organic frameworks (MOFs) requires searching a combinatorially vast space where property labels are expensive and most machine-learning models reveal little about why a structure succeeds. We introduce LLM4MOF, a closed-loop framework in which language-model agents reason about chemistry, build candidate MOFs, and test them in simulation, refining hypotheses over ten autonomous iterations. One agent proposes interpretable design hypotheses over metal nodes, linkers, pore geometry, and functional chemistry, and a second translates them into constraints that select candidate MOFs, each made of a metal node, organic linker, and matching topology. Each hypothesis is tested through four diagnostic beams that apply different subsets of its constraints, so comparing them shows whether geometry, chemistry, or metal choice drives performance. Even when blind to the global property landscape of databases, LLM4MOF concentrates its search on top-performing structures across six adsorption, separation, and electronic-structure tasks within 400 property evaluations. The same loop also generates new MOFs de novo and validates them in live simulation, where it adapts the geometry to each requested condition, outperforming random search and a genetic algorithm at roughly $1 per campaign. LLM4MOF shows that language-model agents can run interpretable, simulation-grounded inverse design without training a model per objective. 

---
# Closing the Activation-Cone Blind Spot: Response-Time Probing and Unified Defense 

**Authors**: Subhadip Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2606.29441)  

**Abstract**: Inference-time safety methods for large language models have proliferated, yet no systematic comparison exists. We evaluate five defense paradigms (no defense, static steering, CAST, AlphaSteer, probe-gated) across seven instruction-tuned models (7-31B) and five attack types (GCG, AutoDAN, DeepInception, prefilling, intent laundering). Our central finding: prompt-time activation defenses are structurally blind to prefilling attacks. AlphaSteer achieves 0% attack success on GCG, AutoDAN, and intent laundering but 50% on prefilling. We prove a corollary: any defense that gates intervention on a single layer's activation alignment with a benign reference (cone, subspace, or null-space) is blind to attacks that craft activations to lie inside that reference, whether checked at prompt time or per token. As its constructive contrapositive we introduce response-time probing: a linear probe on the model's hidden state at the first generated tokens, with AUROC 0.97-1.00 across all seven models. Combined with a halt, it cuts prefilling attack success to 0/40 on every model with 0% benign false positives, outperforming Llama Guard 3. Cross-template generalisation depends on probe depth, so we scope the claim to the canonical prefilling-template family. Composing the response-halt with AlphaSteer's null-space steering gives an orthogonal split (the halt catches prefilling, AlphaSteer catches semantic attacks), reaching defense success 0.983 on Mistral and 0.994 on Llama and dominating both components. We further show MMLU fails to capture steering's true utility cost, which appears as behavioral hedging rather than factual loss, and that diverse negative training sets cut probe false positives from 80-100% to near zero. Code, attacks, per-sample results, and the judge prompt are released. 

---
# Mixture of Debaters: Learn to Debate at Architectural Level in Multi-Agent Reasoning 

**Authors**: Dayong Liang, Kaisong Gong, Yi Cai, Changmeng Zheng, Xiao-Yong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2606.29425)  

**Abstract**: Existing multi-agent debate frameworks suffer from two critical limitations: they rely on static architectures where agent roles and coordination patterns are fixed at design time, and they require instantiating multiple model copies, incurring substantial computational overhead. We propose Mixture of Debaters (MoD), a unified framework that enables dynamic self-debate within a single model by leveraging the Mixture-of-Experts paradigm. We address three key challenges in adapting MoE for dialectical reasoning: (1) dual-routing that decouples role allocation from process flow, dynamically determining when to debate versus when to synthesize; (2) momentum switching that smooths token-level routing with local context, reducing expert-switch jitter; and (3) unified self-debate that encapsulates diverse debating personas into lightweight expert modules, eliminating inter-agent communication while preserving behavioral diversity. Extensive experiments on multimodal benchmarks demonstrate that MoD outperforms both single-model baselines and conventional multi-agent systems, achieving superior accuracy with 3.7x lower latency and 87% reduction in token this http URL source code can be accessed at this https URL. 

---
# Deterministic Decisions for High-Stakes AI. A Zero-Egress Pipeline with the Deployability of RAG and the Accuracy of Machine Learning 

**Authors**: Craig Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2606.29280)  

**Abstract**: We identify intervention bias as a previously unquantified failure mode of zero-shot large-language-model (LLM) educational advisory agents: without task-specific training, they recommend action when a hindsight-optimal oracle policy mandates inaction. In a six-arm ablation on the Open University Learning Analytics Dataset (N=800 students, four temporal cutoffs), at day 56 -- when the oracle designates 70.1% of students as needing no intervention -- zero-shot GPT-4o recommends action for 73%, a 43 percentage-point false-positive rate. Commercial RAG and SQL-augmented retrieval are comparably miscalibrated; at 10,000 students this implies about 4,300 unnecessary advisor contacts per cycle.
Supervised policy learning eliminates this bias: a trajectory-conditioned ONNX Decision Transformer (DT) and a snapshot XGBoost classifier, trained on the same oracle-labelled trajectories under strict prefix-only features, both achieve near-zero calibration error. The DT reaches macro-F1 0.79 (macro-recall 0.85) across all five action classes, predicting even the rare load-reduction action without collapsing, at a 0% action flip rate and sub-5 ms CPU decision latency. The two supervised arms are on par; the DT's edge over XGBoost at the final cutoff is indicative only (unpaired across cohorts).
Scope: we validate Stage-2 decision-making (EAV state vector to supervised policy) under controlled oracle input from structured OULAD data; high fidelity reflects feature-oracle alignment, not general high-stakes-AI capability. The most robust finding is the intervention-bias contrast, not the absolute accuracies. We also show an Evaluation Gap: LLM-as-judge scoring (DeepEval G-Eval) is blind to intervention bias, rewarding fluent over-prescription rather than decision quality. 

---
# Manufactured Confidence: How Memory Consolidation Turns Hearsay into Confident Facts 

**Authors**: Alex Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2606.29279)  

**Abstract**: LLM agents carry conclusions across steps and sessions in compressed memory, and memory products (e.g., mem0, LangMem) rewrite conversation into stored "facts" that later steps trust. We show this rewriting manufactures confidence: across our constructed agent settings, a casual, hedged remark becomes a confident, dated assertion the agent then obeys like a verified fact, granting every above-clearance request it faces. No attacker is needed: a role that was true once and never corrected is stored as a flat fact and acted on like a deliberate injection. We then isolate what the agent responds to. It is not the source: attributed, unattributed, and even forged "system of record" claims all grant alike. It is the confidence of the phrasing. A hedge is discounted, a flat assertion is obeyed, and this holds with no special keyword. Not all hedges are equal, though: the evidential register is the least-discounted, with "reportedly" obeyed like a flat assertion on most models. The obvious fixes fail. A passive "unverified" tag is ignored, and an active "do not trust this" instruction escalates even correct memory, so it is safe only by refusing to decide. The real fix lives in the store: keep the tentative phrasing rather than upgrade it. But that is hygiene, not a defense against an attacker who can simply write a confident lie. The deployable lesson is narrower and constructive: a single load-bearing memory is the hazard, and one redundant source restores correct decisions. We release the harness and demonstrations. 

---
# The Complexity Ceiling Benchmark: A Multi-Domain Evaluation of Sequential Reasoning Under Depth Scaling 

**Authors**: Shubh Chapra, Dhruv Kumar, Murari Mandal, Yash Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2606.29278)  

**Abstract**: We introduce the Complexity Ceiling Benchmark (CCB), a controlled evaluation of how language-model reasoning decays as the number of required sequential steps grows. CCB fixes the semantic content of a task and varies only its depth N in {5,...,50} across three structurally distinct regimes: grounded spatial state-tracking, abstract symbolic pointer manipulation, and transitive relational inference. Across 6,000 trials over five frontier and open-weight LLMs we find a consistent pattern of geometric per-step decay with widely separated domain ceilings: on the first two regimes the strongest models retain pd>0.92 across N=50; on the third every model collapses by N=5, with the best model's 50%-success horizon at H0.5~4.7 steps despite pd=0.863. A trace-level metric (TFBC) shows that 14.5% of correct answers across the benchmark are reached via incorrect intermediate reasoning. Forced verbose state-tracking does not move the ceiling (McNemar p=1.000), and the mean step at which reasoning first diverges, k*, predicts within-domain accuracy better than parameter count. CCB and the geometric decay model together reduce a model's long-horizon reasoning profile to one interpretable number per task family. 

---
# PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents 

**Authors**: Seongjae Kang, Taehyung Yu, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2606.29225)  

**Abstract**: LLM agents handle user requests on behalf of organizations through tool calls and must follow the company policies stated in their system prompts. Prior work approaches this as a safeguarding problem -- external checks that block non-compliant agent actions. We argue that policy adherence is a broader problem: real workflows unfold across many turns, require explicit user confirmation and prerequisite reads, and hinge on the content of the dialogue rather than on any single argument value. Meeting this bar requires (i) full conversation context, (ii) self-reasoning over the policy and the current dialogue, and (iii) conversation-specific remediation that guides the agent's next turn -- three capabilities that prior safeguard work has often underestimated. We introduce POLICYGUARD, a sub-agent verifier that shares the agent's view of the dialogue, reasons over the policy in context, and provides actionable feedback for the agent's next turn. On tau^2-BENCH airline across three vendors (GPT-5.4, Claude Sonnet 4.6, Gemini 2.5 Pro) with four trials per setting, POLICYGUARD improves PASS4 by +12.0 / +6.0 / +12.0 pp. Per-call analyses show POLICYGUARD achieves higher policy-violation recall while blocking roughly half as often as argument-level guards. 

---
# Multi-Block Diffusion Language Models 

**Authors**: Yijie Jin, Jiajun Xu, Yuxuan Liu, Chenkai Xu, Yi Tu, Jiajun Li, Dandan Tu, Xiaohui Yan, Kai Yu, Pengfei Liu, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2606.29215)  

**Abstract**: Block Diffusion Language Models (BD-LMs) improve diffusion-based text generation with KV caching and flexible-length generation. A natural next step is to extend them from Single-Block Diffusion (SingleBD) to Multi-Block Diffusion (MultiBD), where a \textit{running-set} of consecutive blocks is decoded concurrently for inter-block parallelism. However, existing BD-LMs are mostly trained under teacher forcing, where the model observes only one noisy block conditioned on a clean prefix. While the recent diffusion forcing strategy introduces visibility among multiple noisy blocks, its training states still differ from MultiBD inference, where decoding operates on a bounded \textit{running-set} with heterogeneous slot-wise noise patterns. To bridge this gap, we propose \textit{Multi-Block Diffusion Language Models} (MBD-LMs), obtained by post-training BD-LMs with \textit{Multi-block Teacher Forcing} (MultiTF). MultiTF integrates teacher forcing and diffusion forcing by training on bounded \textit{noise-groups} conditioned on clean prefixes, with randomized \textit{noise-schedulers} that better match MultiBD inference states. To make MultiBD practically executable, we further introduce an optimized decoding algorithm based on the \textit{Block Buffer} mechanism that preserves prefix-cache reuse, keeps input shapes static, and translates increased decoding parallelism into wall-clock acceleration. Empirically, MBD-LLaDA2-Mini increases average Tokens Per Forward pass (TPF) from 3.47 to \textbf{6.19} and improves average accuracy from 79.95\% to \textbf{81.03\%}; when combined with DMax, MBD-LLaDA2-Mini-DMax reaches an average TPF of \textbf{9.34} with only a 1.02\% accuracy drop on math and code benchmarks. 

---
# Representational Depth of Evaluation Awareness Shifts With Scale in Open-Weight Language Models 

**Authors**: Archit Manek  

**Link**: [PDF](https://arxiv.org/pdf/2606.29196)  

**Abstract**: Do language models know when they are being tested? This question matters for AI safety: a model that recognises an evaluation context could alter its behaviour strategically, making downstream benchmarks harder to interpret. Using 11 models spanning Qwen 2.5, Gemma 2, and Llama 3.2, we find a systematic size-dependent shift in representational depth: in both Qwen 2.5 and Gemma 2, the layer at which evaluation-awareness is most linearly recoverable moves from late layers in smaller models to early layers in larger ones. This suggests that scale changes not only the strength of evaluation-awareness but also where it is most linearly recoverable in the network. This depth shift helps explain why within-family scaling trajectories are non-monotonic or inverse rather than smooth and family-general, showing that a simple universal power-law account is not supported under denser within-family sampling. Finally, white-box probe signals are consistently stronger than black-box behavioural expression, and the relationship between the two varies by family in ways not predicted by probe AUROC alone. 

---
# Evidence-Informed LLM Beliefs for Continual Scientific Discovery 

**Authors**: Dhruv Agarwal, Reece Adamson, Andrew McCallum, Peter Clark, Ashish Sabharwal, Bodhisattwa Prasad Majumder  

**Link**: [PDF](https://arxiv.org/pdf/2606.29182)  

**Abstract**: Open-ended scientific discovery with large language models (LLMs) increasingly operates as a long-horizon loop of hypothesis search and verification, where a reward signal guides which hypotheses to test next. A notable recent example is AutoDiscovery, which uses "Bayesian surprise" - the belief shift an LLM undergoes after observing evidence for a hypothesis - as both a discovery metric and a reward for search. We first observe that AutoDiscovery treats surprisal as a static quantity, while surprisal in human reasoning is non-stationary - it is defined relative to beliefs that evolve with experience, a prerequisite for continual scientific discovery. We address this mismatch with evidence-informed LLM beliefs: priors updated with evidence from previous hypotheses to compute non-stationary surprisal for new hypotheses. We compare in-context belief-updating mechanisms and find that embedding-based retrieval-augmented generation over prior discoveries best anticipates eventual posteriors, identifying 37.5% of static surprisals as spurious. We then modify search to avoid these spurious rewards and prioritize hypotheses that remain surprising under non-stationary beliefs. Concretely, we introduce two complementary changes to the original search procedure: belief-update filtering and diversity maximization. Across five discovery domains, our method increases accumulated non-stationary surprisal by 30.62% on average compared to the original search procedure, demonstrating that continual scientific discovery with LLMs requires not only better belief measurement but also search procedures that avoid redundancy and encourage diversity. 

---
# Selective Memory Retention for Long-Horizon LLM Agents 

**Authors**: Pranath Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2606.29178)  

**Abstract**: When does retention matter for memory-augmented LLM agents? We study this with TraceRetain, a lightweight framework for bounded external memory in frozen LLM agents that scores entries by interpretable features (success, age, access frequency, redundancy, specificity, similarity, downstream utility) and evicts the lowest-scoring ones at capacity. On clean ALFWorld with gpt-5-mini, external memory robustly improves over no memory across two seeds, but differences among bounded retention policies fall within Wilson 95% CIs: clean ALFWorld at T=100 to T=200 does not naturally exhibit the memory pollution retention is designed to address. Under a controlled noisy-write stress (75% synthetic distractors), unbounded memory and FIFO-K50 degrade on Precision@5 (20.2% to 12.4% and 15.8% to 3.8%) while TraceRetain-CEM is essentially unchanged (16.9% to 16.6%) and preserves 97/100 task success. The mechanism: unbounded memory has the highest mean similarity (0.87) but lowest precision, indicating failed distractors close to the query in embedding space. Held-out in-distribution evaluation shows memory-augmented policies solving 47 to 49 of 50 tasks vs. 39/50 for no memory. Bounded retention buys memory and step efficiency on saturated clean benchmarks at no task-success cost, and only differentiates from cache heuristics when streams contain noise. 

---
# Symbolic Mechanistic Data Attribution: Tracing Training Influence to Learned Behavioral Policies 

**Authors**: Reza Habibi, Darian Lee, Magy Seif El-Nasr  

**Link**: [PDF](https://arxiv.org/pdf/2606.29171)  

**Abstract**: While existing data attribution methods can identify which training examples build specific mechanistic circuits, they cannot explain how training data shapes the high-level behavioral decisions a model learns to make. To bridge this gap, we introduce Symbolic Mechanistic Data Attribution (SMDA), a framework that attributes training pairs to the interpretable symbolic policies governing model behavior. SMDA fits a closed-form Ridge regression over sparse autoencoder (SAE) features to model a target behavior, then analytically decomposes how each supervised fine-tuning example shifts that policy through feature-activation Delta_X and output-probability Delta_Y pathways. We distill a symbolic policy for refusal behavior in Llama-3.2-3B-Instruct and analyze 200 SFT training pairs. Our analysis reveals that (1) the symbolic policy's coefficients expose systematic gaps in the base model's safety behavior for categories like religious stereotyping; (2) per-feature Delta_X/Delta_Y decomposition can mechanistically explain why harmful and harmless pairs exert qualitatively different influences on certain features; and (3) individual training pairs routinely exhibit cross-feature interference, allowing SMDA to identify training pairs whose dominant effect falls on unintended features. These results demonstrate that combining mechanistic interpretability with data attribution yields a diagnostic tool that is both more fine-grained than black-box influence functions and more scalable than manual circuit analysis. 

---
# An Information-Geometric Justification for Composite Coherence in Event-Based Narrative Extraction 

**Authors**: Brian Keith-Norambuena  

**Link**: [PDF](https://arxiv.org/pdf/2606.29118)  

**Abstract**: Graph-based narrative extraction relies on a coherence function to score transitions between events, but the coherence metrics in current use are defined operationally and lack an information-theoretic foundation. We study the composite metric $C=\sqrt{A\cdot T}$, where $A$ is the angular similarity of document embeddings and $T=1-d_{\mathrm{JS}}$ is a topic proximity from the Jensen-Shannon distance of soft memberships, and give it an information-geometric reading together with an axiomatic characterization of the geometric-mean combinator. On the product manifold $\mathbb{S}^{d-1}\times\Delta^{K-1}$, the negative log-coherence decomposes additively into an angular and a topic cost. Because the Riemannian metric tensor induced by the Jensen-Shannon distance on the simplex is proportional to the Fisher information matrix, the topic component is locally consistent with the Fisher-Rao metric singled out by Chentsov's theorem. Within the compensability spectrum of combinators, the geometric mean is the unique one consistent with four natural axioms (a boundary/veto condition, symmetry, log-additivity, normalization), and the construction motivates a proper product metric $d_\times$. Experiments on four corpora, three embedding families, and three topic models are consistent with the framework: the Fisher identity holds ($R\ge0.99$), the geometric mean tracks $d_\times$ closely ($\rho=0.999$), and a downstream LLM-as-judge check finds it is not dominated by any alternative combinator or single-channel baseline. Sweeping the spectrum, the bottleneck-coherence gap between extracted and random storylines splits into a symmetric component, maximized at the geometric mean across five corpora, and a displacement term; a cross-modal image-narrative case study reproduces the effect. These results justify the composite coherence metric and articulate when the geometric mean is the natural choice. 

---
# Low-cost concept-based localized explanations: How far can we get with training-free approaches? 

**Authors**: Darian Fernández-Gutiérrez, Rafael Bello, Marilyn Bello, Natalia Díaz-Rodríguez  

**Link**: [PDF](https://arxiv.org/pdf/2606.29069)  

**Abstract**: Concept-based Explainable AI (C-XAI) seeks human-understandable explanations grounded in semantic concepts, yet validation is limited by the scarcity of fine-grained concept annotations. We evaluate whether mid-scale Multimodal Large Language Models (MLLMs) can perform localized concept naming under strict zero-shot conditions by assigning labels to bounding-box regions at both object and part levels. We propose a reproducible zero-shot evaluation protocol for Concept Naming (CoNa) with (i) closed-set, category-constrained prompting for moderate vocabularies and (ii) Open-CoNa, an embedding-similarity-based strategy for large label spaces. Experiments with four MLLMs (7B-32B) show consistent performance trends across datasets, reaching 62%-88% object-level exact-match accuracy, highlighting the potential of training-free concept annotation from localized regions. We discuss limitations and failure modes and release a reproducible framework to support future low-cost C-XAI research. 

---
# Clustering Unsupervised Representations as Defense against Poisoning Attacks on Speech Commands Classification System 

**Authors**: Thomas Thebaud, Sonal Joshi, Henry Li, Martin Sustek, Jesus Villalba, Sanjeev Khudanpur, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2606.28953)  

**Abstract**: Poisoning attacks entail attackers intentionally tampering with training data. In this paper, we consider a dirty-label poisoning attack scenario on a speech commands classification system. The threat model assumes that certain utterances from one of the classes (source class) are poisoned by superimposing a trigger on it, and its label is changed to another class selected by the attacker (target class). We propose a filtering defense against such an attack. First, we use DIstillation with NO labels (DINO) to learn unsupervised representations for all the training examples. Next, we use K-means and LDA to cluster these representations. Finally, we keep the utterances with the most repeated label in their cluster for training and discard the rest. For a 10% poisoned source class, we demonstrate a drop in attack success rate from 99.75% to 0.25%. We test our defense against a variety of threat models, including different target and source classes, as well as trigger variations. 

---
# MedEvoEval: Evaluating Continual Evolution of Doctor Agents through Simulated Clinical Episodes 

**Authors**: Hui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28900)  

**Abstract**: Doctor agents are moving beyond single-turn answer generation toward evolving clinical decision systems. Within an outpatient episode, they acquire evidence, use examination and consultation resources, and decide when to finalize a diagnosis and management plan. Across episodes, their behavior may change through memory, retrieval, reflection, or other update mechanisms. Current evaluations only partially cover this setting. Fixed-input medical QA benchmarks score final answers from complete inputs, whereas many interactive benchmarks still focus on individual encounters or fixed runs, providing limited support for evaluating how episode-level decisions interact with cross-episode experience. We introduce MedEvoEval, an executable longitudinal evaluation framework based on action-gated simulated outpatient episodes. Each source case is converted into role-specific patient, examination, and manager views; evidence is revealed only through valid actions; and each episode records a structured trace that links observations, actions, final outputs, manager scores, and optional experience write-back. We release a runnable E&D artifact with 700 processed episodes, provenance notes, schemas, an episode runner, scoring scripts, configurations, example logs, analysis code, and trajectory- and step-level derivatives. Experiments show that episode traces expose process costs hidden by final-answer scoring, show how MDT-style consultation reallocates resources, and support longitudinal analyses of memory maturation, held-out transfer, update-stage response, and backward retention. Together, these results show that MedEvoEval provides a concrete basis for evaluating whether doctor agents improve through experience, transfer useful behavior, and retain earlier capabilities over time. 

---
# wav2VOT: Automatic estimation of voice onset time, closure duration, and burst realisation with wav2vec2 

**Authors**: James Tanner, Morgan Sonderegger, Jane Stuart-Smith, Tyler Kendall, Jeff Mielke  

**Link**: [PDF](https://arxiv.org/pdf/2606.28857)  

**Abstract**: While automatic tools for speech annotation are now commonplace within phonetic research pipelines, many tasks require substantial manual correction or training sets to perform accurately. Simultaneously, large speech models such as wav2vec2 have been shown to perform well at speech classification tasks, raising the question of how these models may be applied to phonetic annotation tasks. We introduce wav2VOT: a tool for the automatic estimation of voice onset time, closure duration, and burst realisation using wav2vec2. We demonstrate that wav2VOT performs comparably with current approaches on unseen datasets, and can estimate with high accuracy with fine-tuning. Analysis of wav2VOT predictions demonstrate high fidelity across stop voicing and place of articulation. These results demonstrate that large speech models are capable of producing accurate annotations, and further motivate exploration of large speech models as tools in phonetic research pipelines. 

---
# LAMP: Lean-based Agentic framework with MCP and Proof Repair 

**Authors**: Santhana Srinivasan R, Maithilee Patawar  

**Link**: [PDF](https://arxiv.org/pdf/2606.28841)  

**Abstract**: Large language models are increasingly capable of mathematical reasoning, but the proofs they generate are often unreliable and hard to verify. Interactive theorem provers such as Lean 4 address this by accepting only kernel-checked proofs; however, their reach is bounded by the formalized knowledge available. While Mathlib, a repository of formalized Lean 4 theorems that covers diverse mathematical areas, certain specialized areas remain underrepresented; notably, the domain of Combinatorics on Words (CoW). CoW studies sequences, exploring their properties such as periodicity, borders, conjugacy, and morphisms. As a result, specialized provers, trained on Mathlib-centered data, lack the lemmas to operate in CoW. We present two contributions. First, we introduce a Lean 4 formalization of CoW containing eight modules and \textbf{93} declarations of core definitions and foundational lemmas. Second, we present LAMP, a multi-agent framework that synthesizes kernel-verified Lean 4 proofs by providing explicit, structured domain knowledge at inference time through an ontology, rather than by fine-tuning a prover. LAMP coordinates a Planner, Builder, and Verifier with Model Context Protocol based access to a domain-specific CoW ontology. In a suite of 90 CoW theorems that span all eight modules and three difficulty levels, LAMP synthesizes verified proofs for 96.7% of theorems, substantially exceeding both an unscaffolded baseline and existing specialized provers. An ablation shows that removing LAMP's tool-grounded architecture or its Planner/Builder separation each cost roughly 12 percentage points, even with the backbone model held fixed. 

---
# Categorizing Mathematical Concepts with LLM Voting Ensembles in Mathswitch 

**Authors**: Katja Berčič, Slobodan Stanojevikj  

**Link**: [PDF](https://arxiv.org/pdf/2606.28815)  

**Abstract**: Mathswitch is an open-source project that imports mathematical concept records from sources such as Wikidata, Wikipedia, MathWorld, Encyclopedia of Mathematics, nLab, ProofWiki, and Agda-Unimath, and links records that refer to the same concept. It does not reorganize or redefine the imported content; each source retains its own structure. The current focus is on importing concept data from Wikidata and the resources it links to, with plans to expand to further sources and better concept linking. Because the concept set is approximated through queries over Wikidata's collaboratively edited graph, the imported data is noisy: some items are non-mathematical, while others are ambiguous. In this paper, we test whether a voting ensemble of LLM judges can filter this noise. We evaluate it on Wikidata items with known MathWorld identifiers as a positive control, and examine how classification changes when database identifiers are removed from context. We then inspect the cases where the judges disagree with MathWorld and group these disagreements into three categories (degenerate descriptions, narrow scope bias, and editorial-scope mismatches) that suggest different remediation strategies. 

---
# Improving Large-Scale Weakly Supervised ASR by Filtering and Selection 

**Authors**: Kohei Matsuura, Masato Mimura  

**Link**: [PDF](https://arxiv.org/pdf/2606.28728)  

**Abstract**: Leveraging large-scale weakly supervised datasets is crucial to train robust end-to-end automatic speech recognition (ASR) models. However, such datasets often contain noisy labels and lack domain specificity, limiting their effectiveness. To address these issues and make better use of weakly supervised datasets, we propose a novel training approach incorporating data filtering and selection. Our approach consists of three steps: pretraining on the entire dataset, continued pretraining on a filtered subset based on character error rate (CER), and fine-tuning on a small number of acoustically similar samples to the target domain, selected from the filtered subset. In experiments with a 90,000-hour weakly supervised Japanese dataset, the proposed filtering and selection methods synergistically reduced CER by up to 6.4% and 4.0%, respectively, even though these steps reused training samples already used in the first pretraining step. 

---
# Mitigating Batch Effects in Histopathology via Language-Mediated Robust Embedding Generation 

**Authors**: Yishu Zhang, Shushan Wu, Zhenzhong Zhang, Didong Li, Huaxiu Yao, Yun Li, Iain Carmichael, Katherine A. Hoadley, Hongtu Zhu, Di Wu, Daiwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2606.28697)  

**Abstract**: Pathology foundation models (PFMs) have demonstrated strong potential across clinical and scientific applications, yet their performance is often hindered by batch effects, which are non-biological variations across tissue source institutions (TSIs) that distort learned feature representations and impair generalization. Conventional mitigation strategies, such as stain normalization, offer limited success in addressing these high-dimensional, complex artifacts. We present GLMP (General-purpose LLM-Mediated Pathology model), a novel framework that generates robust numerical embeddings from histology image patches through an intermediate textual representation. By leveraging pretrained general-purpose multimodal large language models (MLLMs) and text encoders, GLMP effectively prioritizes biologically meaningful signals over TSI-specific artifacts, thereby improving cross-institutional generalization. To our knowledge, GLMP is the first pathology model to use text descriptions of histological features as an intermediate representation for generating numerical embeddings from histology images. Our results highlight the untapped potential of broad-domain, non-specialized MLLMs in computational pathology and introduce a new paradigm for building versatile, generalizable, and robust pathology models. 

---
# When More Sampling Hurts: The Modal Ceiling and Correlation Ceiling of Test-Time Scaling 

**Authors**: Yong Yi Bay, Kathleen A. Yearick  

**Link**: [PDF](https://arxiv.org/pdf/2606.28661)  

**Abstract**: People overthink; language models over-sample, and the extra effort can talk both into a worse answer. Reasoning systems answer a hard question by sampling it many times (test-time scaling), and the more they draw, the more often a correct answer turns up somewhere, so coverage, the fraction of problems with at least one correct try, climbs and appears to be progress. But a deployed system must return one answer, and choosing it, not knowing which try is right, is selection; selection is capped, and past a point extra samples only make the model surer of a confident mistake, even as every draw adds cost. The gap between climbing coverage and stalled selection, the identifiability gap, is the answer a model can produce but not pick. So the real question is not whether to sample but how far, and the answer is: not far. For picking an answer, the vote has already settled within a few dozen draws, the modal ceiling; for scoring a benchmark, sooner still, the correlation ceiling. Beyond that, extra draws cost compute and add nothing, and can even make the answer worse. This paper turns the cutoff into a single number, the effective number of samples, that any sampling run already reveals. The bottleneck is recognizing a right answer, not generating one. 

---
# The Undecidability of Artificial General Intelligence (AGI) Alignment 

**Authors**: Jose Pascual Gumbau Mezquita  

**Link**: [PDF](https://arxiv.org/pdf/2606.28639)  

**Abstract**: This article establishes the foundational mathematical limits of Artificial General Intelligence (AGI) safety, proving that the core barrier is not the impossibility of an aligned state, but its structural unverifiability. We formalize this boundary through two central impossibility results: the Unverifiability Theorem of Alignment and the Theorem of Finite Structural Unverifiability of AGI Alignment. We ground this boundary at Trakhtenbrot's Wall, demonstrating that contemporary engineering defenses relying on finite hardware or halting architectures fail to escape logical obstructions. This failure manifests as an inescapable triad of containment failures: open domains yield fundamental undecidability (Rice and Gödel); universal finite verification collapses into algorithmic incomputability (Trakhtenbrot); and particular bounded environments trap the supervisor within intractable bounds in the worst case. As a direct structural corollary of these results, we derive the Soundness--Completeness--Tractability Trilemma, establishing that the mutual incompatibility of these three properties is a necessary consequence of descriptive complexity rather than an empirical anomaly. Finally, we map these theoretical bounds onto practical AI engineering, demonstrating that modern containment strategies are not temporary patches, but mandatory sacrifices of logical expressivity required to secure decidable fragments of safety. 

---
# What LLMs explain is not what they believe: Evaluating explanation sufficiency under models' own input beliefs 

**Authors**: Nhi Nguyen, Shauli Ravfogel, Rajesh Ranganath  

**Link**: [PDF](https://arxiv.org/pdf/2606.28615)  

**Abstract**: Large language models (LLMs) are increasingly deployed in high-stakes domains, where free-text explanations such as chain-of-thought and post-hoc rationales are used to justify model outputs. Yet it remains unclear whether these explanations are sufficient, i.e., if they contain enough information to explain the model's output-generating process. We generalize classical sufficiency from feature attributions to arbitrary explanations and prove that explanation sufficiency can change depending on the input distribution, which must be explicitly defined for LLM explanations. We propose using the LLM itself to generate alternative inputs conditioned on an explanation, capturing its beliefs about possible inputs. We formalize self-consistent sufficiency as a goal for free-text explanations and introduce an information-theoretic metric, SCSuff, that enables evaluation of free-text explanations without relying on predefined biases or shortcuts. Our experiments show that SCSuff agrees with targeted perturbation tests where applicable and demonstrate that explanation sufficiency can vary with the input distribution. We find LLM explanations are generally insufficient and weakly correlated with model size, accuracy, or output entropy. Analysis of final-token hidden states shows that top and bottom SCSuff scores can be predicted from internal representations, suggesting that SCSuff can guide detection and improvement of sufficient LLM explanations. The code for this paper is available at this https URL . 

---
# Animation2Code: Evaluating Temporal Visual Reasoning in Video-to-Code Generation 

**Authors**: Anya Ji, Abhijith Varma Mudunuri, David M. Chan, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2606.28593)  

**Abstract**: While recent vision-language models (VLMs) have achieved significant improvements on static visual-to-code tasks such as generating code for webpages, charts, or SVGs, it remains unclear whether they can recover temporal dynamics when motion is present. To this end, we introduce Animation2Code, a benchmark for evaluating temporal visual reasoning via reconstructing executable web animation code from videos. Animation2Code consists of 1,069 web animation videos with diverse visual appearances and motion patterns, paired with corresponding HTML/CSS/JavaScript implementations. We propose two human-aligned metrics, appearance similarity and temporal similarity, which allow us to disentangle visual fidelity from temporal alignment when comparing rendered animations against ground-truth samples. Benchmarking state-of-the-art VLMs on this dataset shows that current VLMs struggle to maintain temporal consistency in reconstruction, even when achieving high appearance similarity, including under finetuning and iterative refinement settings. Code and data are available at this https URL . 

---
# DataComp-VLM: Improved Open Datasets for Vision-Language Models 

**Authors**: Matteo Farina, Vishaal Udandarao, Thao Nguyen, Selim Kuzucu, Maximilian Böther, Andreas Hochlehnert, Adhiraj Ghosh, Marianna Nezhurina, Karsten Roth, Joschka Struber, Yuhui Zhang, Sebastian Dziadzio, Elaine Sui, Soumya Jahagirdar, Dhruba Ghosh, Hasan Hammoud, Thomas De Min, Simone Caldarella, Jehanzeb Mirza, Sedrick Keh, Mehdi Cherti, Hilde Kuehne, Bernt Schiele, Serena Yeung-Levy, Muhammad Ferjad Naeem, Federico Tombari, Ana Klimovic, Elisa Ricci, Matthias Bethge, Sewoong Oh, Ameya Prabhu, Alessio Tonioni, Jenia Jitsev, Massimiliano Mancini, Ludwig Schmidt, Nikhil Parthasarathy  

**Link**: [PDF](https://arxiv.org/pdf/2606.28551)  

**Abstract**: Building performant Vision-Language Models (VLMs) requires carefully curating large-scale training datasets, yet the community lacks systematic benchmarks for evaluating such curation strategies. We introduce DataComp for VLMs (DCVLM), a benchmark for controlled data-centric experiments to improve VLM training. As part of DCVLM, we collect 160 datasets spanning four data types -- image-caption pairs, multimodal interleaved documents, text-only, and instruction-tuning data -- into a corpus of 6T multimodal tokens. DCVLM allows participants to test curation strategies (filtering, mixing, formatting, sampling) across 1B-8B models and 6.25B-200B token budgets. Models are then evaluated on a carefully selected suite of up to 52 downstream benchmarks across 9 domains. We conduct extensive experiments on DCVLM and find that data mixing, not filtering, is key to a high-quality training dataset: instruction-heavy mixtures scale better than caption-heavy ones, with gains widening at larger scales. The resulting dataset, DCVLM-Baseline, enables training an 8B VLM to 63.6% accuracy on our 33-task core suite with 200B training tokens. Compared to FineVision, the state-of-the-art open VLM training dataset, this represents an improvement of +5.4pp. DCVLM and all accompanying artifacts will be made publicly available at this https URL. 

---
# Who Plays Which Role When? Communication Role Dynamics for Peer Recognition and Team Performance Prediction 

**Authors**: Yifan Song, Wenxuan Wendy Shi, Brian P Bailey, Tal August  

**Link**: [PDF](https://arxiv.org/pdf/2606.28544)  

**Abstract**: Team roles offer an interpretable lens on collaboration, yet computational studies of roles often rely on domain-specific personas or data-driven clustering rather than theory-grounded taxonomies. We operationalize a taxonomy of eight communication roles grounded in education literature and annotate a corpus of 6,307 Slack messages from 55 students across 18 teams in a semester-long computer science course project. We evaluate whether LLMs can approximate expert labels, enabling scalable, taxonomy-driven role annotation. Using these role labels, we characterize role dynamics over teams' lifecycles, finding that different roles peak at different moments and that students enact a more diverse set of roles as projects progress. To evaluate the utility of our role constructs, we use them to predict peer recognition, outperforming lexical, conversational, and LLM-prompting baselines. To assess generalizability beyond the educational context, we apply the same role constructs to a public dataset (DeliData) to predict team performance improvement after deliberation, again exceeding prior performance. 

---
# A Good Talk Does not Look Like a Summary, It Teaches You! Measuring Takeaways from Paper-to-Video Talks 

**Authors**: Ishani Mondal, Aparna Garimella, Ananya Sai, Pannaga Shivaswamy, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2606.28531)  

**Abstract**: Automatically generated videos from scientific papers are increasingly used for education and research dissemination. However, existing evaluation metrics mainly measure visual quality or whether key points from the paper appear in the video without assessing whether the video actually helps viewers understand the ideas. We introduce EffectivePresentationScorer, a framework for evaluating the instructional quality of scientific presentation videos. It checks whether a video explains the main ideas clearly, introduces needed background concepts, and connects technical details to the main contribution of the paper. When we apply EffectivePresentationScorer to the existing paper-to-video generation systems, we find that generated videos mention the correct topics and follow the structure of the paper but fail to explain prerequisite concepts or clarify why the method works. These failures are often ignored by existing video evaluation metrics, which focus on content presence rather than explanatory quality. 

---
# Detecting Clinical Hallucinations in LVLMs via Counterfactual Visual Grounding Uncertainty 

**Authors**: Xiao Song, Haonan Qin, Zhaoxu Zhang, Jiong Zhang, Yuqi Fang, Caifeng Shan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28520)  

**Abstract**: Large vision-language models (LVLMs) are increasingly used for clinical image understanding, yet they remain vulnerable to \emph{hallucinations}--producing textual findings or attributes not supported by the image. We present a vision-traceable hallucination detection framework that audits arbitrary LVLM responses via visual evidence grounding, requiring neither modification nor internal access to the hidden states of LVLMs. Given an LVLM response, we extract visually verifiable entities and use a medical-domain-adapted Qwen-VL grounding verifier to localize each entity on the input image. To enhance the robustness of our detection method, we introduce a counterfactual entity perturbation method and estimate visual evidence uncertainty by contrasting factual and counterfactual grounding results. Specifically, we compute an entity-level uncertainty score from the positive confidence, counterfactual confidence, and their grounding overlap for binary hallucination decision-making. Experiments on multiple medical imaging modalities and LVLM backbones demonstrate that our method consistently improves hallucination detection performance over recent baselines, while providing interpretable localization evidence and strong cross-model transferability. Code and dataset are available at this https URL. 

---
# GPTNT: Benchmarking Real-Time Collaboration Between Multimodal Agents on Keep Talking And Nobody Explodes 

**Authors**: Amit Parekh, Sabrina McCallum, Kareem Al-Hasan, Malvina Nikandrou, Alessandro Suglia, Ioannis Konstas  

**Link**: [PDF](https://arxiv.org/pdf/2606.28514)  

**Abstract**: Multimodal models are increasingly deployed to solve tasks collaboratively with humans or other artificial agents. Existing benchmarks show that these models possess many of the required component capabilities, but the conditions that coincide in collaboration, including time pressure, information asymmetry, and imperfect communication, are usually studied in isolation. We introduce GPTNT, a benchmark built on the cooperative video game Keep Talking and Nobody Explodes, in which two agents must coordinate to defuse procedurally generated bomb puzzles against a live countdown. One agent can see and manipulate the bomb but does not have the defusal instructions; the other has the instructions but cannot see or manipulate the bomb. Neither agent can succeed alone: success requires effective and efficient communication. Unlike turn-based proxies, GPTNT requires agents to act asynchronously and communicate in real time. GPTNT is designed to separate collaboration from reliance on memorized solutions: the instruction manual, the partner, or both can be withheld to isolate what a model derives in the moment from what it already knows. We show that GPTNT poses a substantial challenge for state-of-the-art systems: none of the closed- or open-source models we test defuses a single bomb in real time, a bar that human players clear. Through controlled experiments, we identify critical weaknesses in state tracking, efficient action under time pressure, ambiguity handling, and error recovery. We release GPTNT as a benchmark for collaborative performance that current evaluations leave unmeasured. Because it runs on the real game, GPTNT benefits from procedural generation and inherits a living modding community, allowing the benchmark to evolve as models improve rather than being solved once and retired. 

---
# An Agentic AI Pipeline for Appliance-Level Energy Anomaly Detection and LLM-Driven Recommendations 

**Authors**: Dihia Falouz, Aida Douaibia, Amine Bechar, Youssef Elmir, Abbes Amira, Adel Oulefki  

**Link**: [PDF](https://arxiv.org/pdf/2606.28467)  

**Abstract**: Appliance-level energy monitoring in office buildings produces noisy alerts that non-expert facility managers struggle to use. This paper proposes an end-to-end agentic pipeline that combines deep time-series forecasting, variational anomaly detection, and LLM-based reasoning to generate prioritized, actionable maintenance recommendations. The system tracks seven office appliances using a hybrid Singular Spectrum Analysis (SSA) and Long Short-Term Memory (LSTM) forecasting model, and applies a per-appliance LSTM Variational Autoencoder (VAE) with attention to flag abnormal daily consumption episodes. A three-stage LangChain pipeline begins with a Context Agent that always retrieves three core RAG sources (model reliability, hourly baseline, and expert knowledge) and conditionally adds up to three more (forecast context, anomaly history, global baseline) based on event characteristics, capped at eight reasoning steps. A Diagnosis Agent converts the evidence into a structured JSON diagnosis, and a Report Agent renders a human-readable narrative. A reflective memory layer incorporates operator feedback. The dashboard shows real-time 30-minute forecasts, intraday consumption, the previous day anomaly report, and a feedback form. We evaluate the forecasting model, anomaly detector with appliance-specific thresholds, and LLM reasoning on a 16-scenario benchmark including sustained and transient spikes, unexpected shutdowns, and systemic events, comparing five LLM backends under static vs. dynamic retrieval. Dynamic retrieval matches full static retrieval across all backends while cutting average context from six to three-six sources per event. The best backend scores 90.4/100 with a 100% pass rate at a 70-point threshold, and a fully local 7B-parameter model passes all 16 scenarios. 

---
# LoRA-Tuned Large Language Models for Dementia Detection via Multi-View Speech-Derived Features 

**Authors**: Jonghyeon Park, Olivier Jiyoun Jung, Myungwoo Oh  

**Link**: [PDF](https://arxiv.org/pdf/2606.28445)  

**Abstract**: Early detection of dementia enables timely intervention, and reflecting cognitive impairment, spontaneous speech offers a non-invasive screening modality. Conventional approaches often focus on a single representational dimension -- such as acoustic descriptors, pause modeling, automatic speech recognition (ASR) transcripts, or multimodal fusion -- limiting integrative reasoning across heterogeneous cognitive symptoms. We propose a low-rank adaptation (LoRA)-tuned large language model (LLM) that performs structured multi-view reasoning over four complementary speech-derived signals: ASR transcripts with pause markers, discourse-level topic cues, temporal fluency statistics, and phonological sequences. These cues are encoded within a unified prompt, enabling a single LLM to learn a coherent decision function without modality-specific encoders or late-stage fusion. On ADReSSo, our best model achieves an F1-score of 90.14%, and ablation confirms the complementary contribution of each view. 

---
# LEDGER: Scaling Agentic Document Editing with Dependency-aware Graph Retrieval 

**Authors**: Mike Hang Wang, Utkarsh Garg, Reza Davari, Huitian Jiao, Hao Cheng, Baolin Peng, Tao Ge, Si-Qing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2606.28379)  

**Abstract**: We introduce LEDGER to tackle the novel context engineering challenge of agentic document editing, where localized edits to long, structured documents must be applied efficiently without breaking cross-references or semantic consistency. LEDGER constructs a lightweight dependency graph that explicitly models document structure, including hierarchical organization, explicit references, implicit dependencies, and semantic relationships. For each edit, graph-guided retrieval selects only the necessary context, avoiding full-document processing while preserving consistency. We evaluate LEDGER on a curated benchmark of 1.9k test cases with various document types and lengths, spanning six state-of-the-art models: LEDGER improves consistency from 56% to 76% across all six models and test scenarios while reducing token usage. Notably, LEDGER with low reasoning effort matches baseline performance at high reasoning effort using fewer tokens, showing that explicit dependency representations can partially substitute for expensive internal reasoning in agentic document editing. 

---
# LUMEN: Cost-Transparent Multi-Agent Pipeline for Automated Systematic Review and Meta-Analysis 

**Authors**: Yen-Hsun Huang, Yu-Shiou Lin  

**Link**: [PDF](https://arxiv.org/pdf/2606.28362)  

**Abstract**: Systematic reviews and meta-analyses (SR/MA) remain the gold standard for evidence synthesis, yet completing one typically requires 67 weeks and substantial expert effort. Recent large language model (LLM) systems have demonstrated strong performance on individual SR phases - screening (otto-SR: 96.7% sensitivity), extraction (Gartlehner et al.: 91.0% accuracy), and search (TrialMind: 0.83 recall) - but no study has reported what it actually costs to run an end-to-end pipeline, how cost distributes across phases, or how architectural choices affect the cost-quality trade-off. We present LUMEN, an open-source multi-agent pipeline that automates six SR/MA phases using 11 specialized LLM agents with deliberate model routing. We evaluate LUMEN on seven datasets: five self-conducted domain reviews (psychiatry, psychology, surgery, vaccinology, cardiology) and two SYNERGY screening benchmarks. Across 13 ground-truth-comparable outcomes, LUMEN achieves 100% directional agreement with published meta-analyses, with effect sizes within 1% for homogeneous study designs. The primary contribution is the first empirical cost and operational characterization of such a pipeline: a complete review costs 19 to 29 USD (median 22.65 USD), with title-abstract screening and data extraction together dominating expenditure. A three-arm extraction ablation reveals a phase-dependent architecture reversal: multi-agent design hurts screening but is essential for extraction, producing 5.7x more poolable analyses than single-model alternatives while eliminating clinically dangerous direction errors. A two-dataset screening benchmark demonstrates that model ranking is domain-dependent and not transferable across review topics. All code and cost logs are publicly available. 

---
# How Do LLMs Cite? A Mechanistic Interpretation of Attribution in Retrieval-Augmented Generation 

**Authors**: Ian van Dort, Maria Heuss  

**Link**: [PDF](https://arxiv.org/pdf/2606.28358)  

**Abstract**: Retrieval-Augmented Generation (RAG) aims to enhance the trustworthiness of Large Language Models (LLMs) by grounding their outputs in external documents, often using inline citations for verifiability. However, the faithfulness of these citations -- whether the model genuinely uses a source to generate an answer -- remains a critical, unverified assumption. This paper offers the first mechanistic account of how a large language model decides whether to attach an inline citation while answering a factoid question. Using the Llama-3.1-8B-Instruct model in a controlled experimental environment based on the PopQA dataset, we employ an activation patching approach. We map the underlying mechanism responsible for citation, discovering that it is not a single, localized component but a distributed, multi-stage "attributional ensemble" of attention heads and MLP layers. We show that amplifying or attenuating only those critical heads and MLPs repairs over 90% of missed citations and eliminates 69% of spurious ones on PopQA without harming answer accuracy. Although gains on the multi-document HotpotQA benchmark are modest, the same component set still moves citation rates in the intended direction, indicating that the underlying mechanism is not dataset-specific. The results reveal a potential disconnect between the model's apparent reasoning and its internal computational pathway, suggesting that inline citations can create a false sense of security. 

---
# Sifei at SemEval-2026 Task 8: Hybrid Retrieval and Query Rewriting for Multi-Turn RAG 

**Authors**: Sifei Meng, Dmitry Ilvovsky  

**Link**: [PDF](https://arxiv.org/pdf/2606.28352)  

**Abstract**: Multi-turn retrieval-augmented generation (RAG) is challenging due to evolving user intent, conversational noise, and strict context limits. We propose a training-free hybrid retrieval pipeline for SemEval-2026 Task 8 that combines dense and sparse retrieval with controlled query rewriting and cross-encoder reranking. On the official test set of Task A, our system achieves 0.5453 nDCG@5, ranking third among 38 teams and outperforming the strongest baseline score of 0.4795. For Task C, we reuse the documents retrieved for Task A and apply a lightweight generation pipeline guided by the official prompt, achieving 0.5312 as the harmonic mean of relevance and faithfulness and ranking 15th among 29 teams. All retrieval components are open-source, while query rewriting and answer generation rely on LLM APIs. 

---
# Auditing LLM-Governed Social Robots with Culture-Specific Moral Gradients 

**Authors**: Carmen Ng, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2606.28345)  

**Abstract**: LLM-governed social robots increasingly decide who receives real-world assistance first. As prioritization norms vary across cultures by age, status, and group size, failure to calibrate pluralistically can scale into unequal access. Yet LLM moral audits remain English-centered, rarely test embodied contexts, leaving pluralistic calibration as an urgent diagnostic gap amid intensifying LLM-robot deployment. We introduce a gradient-based audit framework for multilingual evaluation of LLM moral trade-off behavior against cultural preference gradients. Grounded in nine cross-domain social robotics reviews (>8,000 papers), we derive symmetry-controlled scenarios across care, education, and services, translating the Moral Machine Experiment's "whom to spare" into "whom to assist first" dilemmas with preserved identity trade-offs (many vs. few; young vs. old; higher vs. lower status). We audit four LLMs across four country-language pairs in four prompting regimes (57,600 decisions), benchmarked against country-specific MME preference gradients. Ordinal concordance tests whether models differentiate cultural contexts; a governance typology maps vulnerabilities in gradient differentiation, directional tendency, and deliberation. We find persistent, culturally asymmetric gradient tracking failures that prompting alone cannot reliably correct: quality calibration is nearly twice as strong for Western-language decisions as for Chinese and Japanese; high determinism in majority-first trade-offs often erases cross-cultural gradients; partial sensitivity to age- and status-based norms risks sidelining minorities. Prompting effects are uneven; only contrastive exemplars yield consistent gains, while reasoning-only prompts can worsen tracking. Our results motivate multilingual, pluralistic audits as an LLM-robot pre-deployment gate and suggest model factors are a more robust lever than prompting alone. 

---
# PIXELRAG: Web Screenshots Beat Text for Retrieval-Augmented Generation 

**Authors**: Yichuan Wang, Zhifei Li, Zirui Wang, Paul Teiletche, Lesheng Jin, Matei Zaharia, Joseph E. Gonzalez, Sewon Min  

**Link**: [PDF](https://arxiv.org/pdf/2606.28344)  

**Abstract**: Augmenting large language models (LLMs) with retrieved web text has become a dominant paradigm, yet the web is not natively textual: existing systems depend on complex parsing pipelines that linearize HTML and discard layout, visual structure, and formatting. We introduce PixelRAG, a new retrieval-augmented method that represents websites in their native visual form and performs retrieval and reading entirely in pixel space, enabling an end-to-end architecture that eliminates text abstraction. PixelRAG is, to our knowledge, the first pipeline to operate over a full Wikipedia corpus in this form, scaling to a datastore of 30 million screenshot images with an efficient visual retrieval index. Built on an existing visual embedding model (i.e., Qwen3-VL-Embedding), PixelRAG further fine-tunes this model on screenshot data with carefully curated contrastive training data. Retrieved screenshots are then fed directly as pixel inputs to a VLM, without intermediate text conversion. PixelRAG consistently outperforms both no-retrieval and text-based RAG baselines, most surprisingly on widely studied text-centric tasks such as NQ and SimpleQA. It also achieves strong gains on multimodal open-domain QA (e.g., MMSearch), benchmarks over noisy news corpora (e.g., LiveVQA), and agentic benchmarks (e.g., MoNaCo), improving accuracy by up to 18.1% over text-based baselines. Finally, pixel representations enable a new efficiency lever for RAG through image compression, achieving up to 3x token cost reduction at lower resolutions while maintaining accuracy. Our results challenge the necessity of text representations in web retrieval, suggesting that web RAG can operate directly in the web's native visual form while improving both performance and efficiency. 

---
# LLM-Ideoplasticity: Measuring Ideological Plasticity in the Political Behavior of LLMs as a Context-Conditioned Distribution 

**Authors**: Adib Sakhawat, Syed Rifat Raiyan, Tahsin Islam, Takia Farhin, Hasan Mahmud, Md Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2606.28335)  

**Abstract**: We argue, with systematic empirical evidence, that a large language model's political ideology is not a fixed point, but a conditional distribution $\mathbb{P}($position$\mid$context$)$ over a real political space. We evaluate nine current LLMs using a unified measurement framework anchored by VAA-CHES projection models, which map responses onto three validated dimensions (lrgen, lrecon, galtan) across six contextual axes. Our findings reveal high sensitivity to context: persuasive framing and under-represented languages displace coordinates by up to 0.57 and 0.52 units, respectively, while chain-of-thought reasoning often amplifies rather than dampens paraphrase instability. Despite this local plasticity, the model cohort occupies a remarkably narrow Overton envelope overall, occupying roughly one-third the spread of major European parties. Supported by a multi-trait multi-method (MTMM) analysis, we conclude that a single point cannot summarize LLM political behavior; it must be characterized as a shape. Our code and data are publicly available at this https URL. 

---
# $M^3 QuestionIng$: Multi-modal Multi-span Medical Question Answering 

**Authors**: Anisha Saha, Vaibhav Rathore, Abhisek Tiwari, Akash Ghosh, Sai Ruthvik Edara, Sriparna Saha  

**Link**: [PDF](https://arxiv.org/pdf/2606.28329)  

**Abstract**: The growing adoption of AI in healthcare, particularly in preventive care, highlights the critical need for accessibility and precision in Medical Question Answering (MedQA). In recent years, significant efforts have been made to develop multi-span medical question-answering systems, where the answer to a query may span multiple sections or paragraphs of a source document. However, existing systems fall short of aligning with real-world scenarios, where source documents often include both textual and visual content, requiring answers to incorporate images for better comprehension. To address this gap, we propose $M^3QAFrame$, a multi-modal, multi-span medical question-answering framework that leverages visual cues to enhance the generation of comprehensive answers drawn from diverse textual and visual spans. The model takes the context, query, and images as input and outputs an answer containing both textual answers and relevant images. The text and image embeddings are processed using a transformer-based architecture to determine the sentence and image relevance. We curate a multi-modal, multi-span medical question-answering ($M^3 QuestionIng$) dataset containing queries, medical contexts, associated medical images, and extractive answers. Additionally, each query-answer pair is labeled with user intent and query type to enhance query and context comprehension. Extensive experiments show that our approach consistently outperforms existing methods across various evaluation metrics. 

---
# The Interference Gap: Comparing Retrieval Bounds in Human Memory and RAG Systems 

**Authors**: Dongxin Guo, Jikun Wu, Siu-Ming Yiu  

**Link**: [PDF](https://arxiv.org/pdf/2606.28327)  

**Abstract**: How do retrieval bounds compare between human episodic memory and Retrieval-Augmented Generation (RAG) systems under semantic interference? We present a unified signal detection theory (SDT) framework that applies to both, and use it to fit behavioral and computational data in matched paradigms. Both systems show logarithmic accuracy decline with association count (fan), but humans exhibit lower interference sensitivity ($\alpha/\sigma = 0.41$) than dense passage retrieval ($\alpha/\sigma = 0.67$), with cognitively-inspired HippoRAG falling between the two ($\alpha/\sigma = 0.44$). Behavioral experiments ($N = 112$) and simulations validate the framework; parameter recovery confirms identifiability ($r \geq .93$) and model comparison favors the logarithmic specification over a power-law alternative ($\Delta$BIC $> 15$). We discuss encoding specificity, temporal context binding, and retrieval gating as candidate mechanisms whose causal role remains to be established. Six falsifiable predictions connect cognitive memory research with AI retrieval evaluation. 

---
# The Digital Afterlife of Empires: Four Language Models Converge on the Same Imperial Cartography of Writing 

**Authors**: Hiroki Fukui  

**Link**: [PDF](https://arxiv.org/pdf/2606.28325)  

**Abstract**: Large language models process the world's writing systems with radical inequality. We constructed the Digital Script Representation Index (DSRI), a seven-axis measure of digital support, and applied it to the 300 writing systems of the Global Script Database (Fukui, 2026). Only 29 scripts (9.7%) are fully supported by contemporary digital infrastructure; among 158 living scripts, 60 (38.0%) lack complete support. Tokenizer efficiency varies by a factor of 31.7 across 45 scripts measured with parallel text. A serial mediation model -- imperial intervention to speaker population to web corpus to tokenizer efficiency -- is consistent with full mediation, with the direct effect of empire indistinguishable from zero (beta = -0.22, p = 0.39) and structural equation model fit indices indistinguishable from saturation at n = 45; the bias-corrected bootstrap CI grazes zero, and we treat the mediation as suggestive rather than confirmatory. Across four independent LLM families (Claude, GPT-4o, Grok, DeepSeek; 12,000 API calls), base-rate-deviation error patterns converge at Spearman rho = 0.85-0.98 (all p < 0.002). 172 script-feature items are answered identically wrong by all four models; over-attribution outnumbers under-recognition 3.9:1, and "used for religion" alone concentrates 43.6% of convergent errors (enrichment 4.1x). With religion excluded as a sensitivity check, the cross-architecture convergence is preserved (mean rho = 0.87 on nine features) and the over-attribution asymmetry persists at 1.77:1 (n = 97, binomial p = 0.008), indicating multi-channeled rather than single-channeled bias. The findings are consistent with an interpretation in which the structural inequalities historical empires inflicted on script communities persist in contemporary language models through the shared training corpus rather than through any individual model's design choices. 

---
# When Does Personality Composition Matter for Multi-Agent LLM Teams? 

**Authors**: Aryan Keluskar, Amrita Bhattacharjee, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2606.27443)  

**Abstract**: Personality prompting shapes how large language models communicate, yet whether these behavioral shifts affect objective task outcomes remains under-explored. Prior work shows that agents prompted with low agreeableness produce adversarial language, while those prompted with high agreeableness become cooperative, but the relationship between communication style and task performance has not been systematically examined across multiple domains. In this work, we investigate whether personality composition matters for multi-agent team performance by manipulating personality traits across frontier LLMs on three task domains: structured coding, open-ended research collaboration, and competitive bargaining. We find that personality effects depend critically on task structure. In coding tasks, low agreeableness leads to large communication shifts that have little effect on milestone completion. In open-ended collaboration and bargaining, the same manipulation substantially degrades performance. We discuss implications for multi-agent system design and the limits of personality manipulation. 

---
