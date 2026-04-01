# Agenda-based Narrative Extraction: Steering Pathfinding Algorithms with Large Language Models 

**Authors**: Brian Felipe Keith-Norambuena, Carolina Inés Rojas-Córdova, Claudio Juvenal Meneses-Villegas, Elizabeth Johanna Lam-Esquenazi, Angélica María Flores-Bustos, Ignacio Alejandro Molina-Villablanca, Joshua Emanuel Leyton-Vallejos  

**Link**: [PDF](https://arxiv.org/pdf/2603.29661)  

**Abstract**: Existing narrative extraction methods face a trade-off between coherence, interactivity, and multi-storyline support. Narrative Maps supports rich interaction and generates multiple storylines as a byproduct of its coverage constraints, though this comes at the cost of individual path coherence. Narrative Trails achieves high coherence through maximum capacity path optimization but provides no mechanism for user guidance or multiple perspectives. We introduce agenda-based narrative extraction, a method that bridges this gap by integrating large language models into the Narrative Trails pathfinding process to steer storyline construction toward user-specified perspectives. Our approach uses an LLM at each step to rank candidate documents based on their alignment with a given agenda while maintaining narrative coherence. Running the algorithm with different agendas yields different storylines through the same corpus. We evaluated our approach on a news article corpus using LLM judges with Claude Opus 4.5 and GPT 5.1, measuring both coherence and agenda alignment across 64 endpoint pairs and 6 agendas. LLM-driven steering achieves 9.9% higher alignment than keyword matching on semantic agendas (p=0.017), with 13.3% improvement on \textit{Regime Crackdown} specifically (p=0.037), while keyword matching remains competitive on agendas with literal keyword overlap. The coherence cost is minimal: LLM steering reduces coherence by only 2.2% compared to the agenda-agnostic baseline. Counter-agendas that contradict the source material score uniformly low (2.2-2.5) across all methods, confirming that steering cannot fabricate unsupported narratives. 

---
# Performance Evaluation of LLMs in Automated RDF Knowledge Graph Generation 

**Authors**: Ioana Ramona Martin, Tudor Cioara, Ionut Anghel, Gabriel Arcas  

**Link**: [PDF](https://arxiv.org/pdf/2603.29878)  

**Abstract**: Cloud systems generate large, heterogeneous log data containing critical infrastructure, application, and security information. Transforming these logs into RDF triples enables their integration into knowledge graphs, improving interpretability, root-cause analysis, and cross-service reasoning beyond what raw logs allow. Large Language Models (LLMs) offer a promising approach to automate RDF knowledge graph generation; however, their effectiveness on complex cloud logs remains largely unexplored. In this paper, we evaluate multiple LLM architectures and prompting strategies for automated RDF extraction using a controlled framework with two pipelines for systematically processing semi-structured log data. The extraction pipeline integrates multiple LLMs to identify relevant entities and relationships, automatically generating subject-predicate-object triples. These outputs are evaluated using a dedicated validation pipeline with both syntactic and semantic metrics to assess accuracy, completeness, and quality. Due to the lack of public ground-truth datasets, we created a reference Log-to-KG dataset from OpenStack logs using manual annotation and ontology-driven methods, enabling objective baseline. Our analysis shows that Few-Shot learning is the most effective strategy, with Llama achieving a 99.35% F1 score and 100% valid RDF output while Qwen, NuExtract, and Gemma also perform well under Few-Shot prompting, with Chain-of-Thought approaches maintaining similar accuracy. One-Shot prompting offers a lighter but effective alternative, while Zero-Shot and advanced strategies such as Tree-of-Thought, Self-Critique, and Generate-Multiple perform substantially worse. These results highlight the importance of contextual examples and prompt design for accurate RDF extraction and reveal model-specific limitations across LLM architectures. 

---
# Learning Diagnostic Reasoning for Decision Support in Toxicology 

**Authors**: Nico Oberländer, David Bani-Harouni, Tobias Zellner, Nassir Navab, Florian Eyer, Matthias Keicher  

**Link**: [PDF](https://arxiv.org/pdf/2603.29608)  

**Abstract**: Acute poly-substance intoxication requires rapid, life-saving decisions under substantial uncertainty, as clinicians must rely on incomplete ingestion details and nonspecific symptoms. Effective diagnostic reasoning in this chaotic environment requires fusing unstructured, non-medical narratives (e.g. paramedic scene descriptions and unreliable patient self-reports or known histories), with structured medical data like vital signs. While Large Language Models (LLMs) show potential for processing such heterogeneous inputs, they struggle in this setting, often underperforming simple baselines that rely solely on patient histories. To address this, we present DeToxR (Decision-support for Toxicology with Reasoning), the first adaptation of Reinforcement Learning (RL) to emergency toxicology. We design a robust data-fusion engine for multi-label prediction across 14 substance classes based on an LLM finetuned with Group Relative Policy Optimization (GRPO). We optimize the model's reasoning directly using a clinical performance reward. By formulating a multi-label agreement metric as the reward signal, the model is explicitly penalized for missing co-ingested substances and hallucinating absent poisons. Our model significantly outperforms its unadapted base LLM counterpart and supervised baselines. Furthermore, in a clinical validation study, the model indicates a clinical advantage by outperforming an expert toxicologist in identifying the correct poisons (Micro-F1: 0.644 vs. 0.473). These results demonstrate the potential of RL-aligned LLMs to synthesize unstructured pre-clinical narratives and structured medical data for decision support in high-stakes environments. 

---
# When Can We Trust LLM Graders? Calibrating Confidence for Automated Assessment 

**Authors**: Robinson Ferrer, Damla Turgut, Zhongzhou Chen, Shashank Sonkar  

**Link**: [PDF](https://arxiv.org/pdf/2603.29559)  

**Abstract**: Large Language Models (LLMs) show promise for automated grading, but their outputs can be unreliable. Rather than improving grading accuracy directly, we address a complementary problem: \textit{predicting when an LLM grader is likely to be correct}. This enables selective automation where high-confidence predictions are processed automatically while uncertain cases are flagged for human review. We compare three confidence estimation methods (self-reported confidence, self-consistency voting, and token probability) across seven LLMs of varying scale (4B to 120B parameters) on three educational datasets: RiceChem (long-answer chemistry), SciEntsBank, and Beetle (short-answer science). Our experiments reveal that self-reported confidence consistently achieves the best calibration across all conditions (avg ECE 0.166 vs 0.229 for self-consistency). Surprisingly, self-consistency remains 38\% worse despite requiring 5$\times$ the inference cost. Larger models exhibit substantially better calibration though gains vary by dataset and method (e.g., a 28\% ECE reduction for self-reported), with GPT-OSS-120B achieving the best calibration (avg ECE 0.100) and strong discrimination (avg AUC 0.668). We also observe that confidence is strongly top-skewed across methods, creating a ``confidence floor'' that practitioners must account for when setting thresholds. These findings suggest that simply asking LLMs to report their confidence provides a practical approach for identifying reliable grading predictions. Code is available \href{this https URL}{here}. 

---
# LLM Probe: Evaluating LLMs for Low-Resource Languages 

**Authors**: Hailay Kidu Teklehaymanot, Gebrearegawi Gebremariam, Wolfgang Nejdl  

**Link**: [PDF](https://arxiv.org/pdf/2603.29517)  

**Abstract**: Despite rapid advances in large language models (LLMs), their linguistic abilities in low-resource and morphologically rich languages are still not well understood due to limited annotated resources and the absence of standardized evaluation frameworks. This paper presents LLM Probe, a lexicon-based assessment framework designed to systematically evaluate the linguistic skills of LLMs in low-resource language environments. The framework analyzes models across four areas of language understanding: lexical alignment, part-of-speech recognition, morphosyntactic probing, and translation accuracy. To illustrate the framework, we create a manually annotated benchmark dataset using a low-resource Semitic language as a case study. The dataset comprises bilingual lexicons with linguistic annotations, including part-of-speech tags, grammatical gender, and morphosyntactic features, which demonstrate high inter-annotator agreement to ensure reliable annotations. We test a variety of models, including causal language models and sequence-to-sequence architectures. The results reveal notable differences in performance across various linguistic tasks: sequence-to-sequence models generally excel in morphosyntactic analysis and translation quality, whereas causal models demonstrate strong performance in lexical alignment but exhibit weaker translation accuracy. Our results emphasize the need for linguistically grounded evaluation to better understand LLM limitations in low-resource settings. We release LLM Probe and the accompanying benchmark dataset as open-source tools to promote reproducible benchmarking and to support the development of more inclusive multilingual language technologies. 

---
# Distilling Human-Aligned Privacy Sensitivity Assessment from Large Language Models 

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29497)  

**Abstract**: Accurate privacy evaluation of textual data remains a critical challenge in privacy-preserving natural language processing. Recent work has shown that large language models (LLMs) can serve as reliable privacy evaluators, achieving strong agreement with human judgments; however, their computational cost and impracticality for processing sensitive data at scale limit real-world deployment. We address this gap by distilling the privacy assessment capabilities of Mistral Large 3 (675B) into lightweight encoder models with as few as 150M parameters. Leveraging a large-scale dataset of privacy-annotated texts spanning 10 diverse domains, we train efficient classifiers that preserve strong agreement with human annotations while dramatically reducing computational requirements. We validate our approach on human-annotated test data and demonstrate its practical utility as an evaluation metric for de-identification systems. 

---
# Bringing Up a Bilingual BabyLM: Investigating Multilingual Language Acquisition Using Small-Scale Models 

**Authors**: Linda Zeng, Steven Y. Feng, Michael C. Frank  

**Link**: [PDF](https://arxiv.org/pdf/2603.29552)  

**Abstract**: Multilingualism is incredibly common around the world, leading to many important theoretical and practical questions about how children learn multiple languages at once. For example, does multilingual acquisition lead to delays in learning? Are there better and worse ways to structure multilingual input? Many correlational studies address these questions, but it is surprisingly difficult to get definitive answers because children cannot be randomly assigned to be multilingual and data are typically not matched between languages. We use language model training as a method for simulating a variety of highly controlled exposure conditions, and create matched 100M-word mono- and bilingual datasets using synthetic data and machine translation. We train GPT-2 models on monolingual and bilingual data organized to reflect a range of exposure regimes, and evaluate their performance on perplexity, grammaticality, and semantic knowledge. Across model scales and measures, bilingual models perform similarly to monolingual models in one language, but show strong performance in the second language as well. These results suggest that there are no strong differences between different bilingual exposure regimes, and that bilingual input poses no in-principle challenges for agnostic statistical learners. 

---
# Enhancing Structural Mapping with LLM-derived Abstractions for Analogical Reasoning in Narratives 

**Authors**: Mohammadhossein Khojasteh, Yifan Jiang, Stefano De Giorgis, Frank van Harmelen, Filip Ilievski  

**Link**: [PDF](https://arxiv.org/pdf/2603.29997)  

**Abstract**: Analogical reasoning is a key driver of human generalization in problem-solving and argumentation. Yet, analogies between narrative structures remain challenging for machines. Cognitive engines for structural mapping are not directly applicable, as they assume pre-extracted entities, whereas LLMs' performance is sensitive to prompt format and the degree of surface similarity between narratives. This gap motivates a key question: What is the impact of enhancing structural mapping with LLM-derived abstractions on their analogical reasoning ability in narratives? To that end, we propose a modular framework named YARN (Yielding Abstractions for Reasoning in Narratives), which uses LLMs to decompose narratives into units, abstract these units, and then passes them to a mapping component that aligns elements across stories to perform analogical reasoning. We define and operationalize four levels of abstraction that capture both the general meaning of units and their roles in the story, grounded in prior work on framing. Our experiments reveal that abstractions consistently improve model performance, resulting in competitive or better performance than end-to-end LLM baselines. Closer error analysis reveals the remaining challenges in abstraction at the right level, in incorporating implicit causality, and an emerging categorization of analogical patterns in narratives. YARN enables systematic variation of experimental settings to analyze component contributions, and to support future work, we make the code for YARN openly available. 

---
# Can LLM Agents Identify Spoken Dialects like a Linguist? 

**Authors**: Tobias Bystrich, Lukas Hamm, Maria Hassan, Lea Fischbach, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2603.29541)  

**Abstract**: Due to the scarcity of labeled dialectal speech, audio dialect classification is a challenging task for most languages, including Swiss German. In this work, we explore the ability of large language models (LLMs) as agents in understanding the dialects and whether they can show comparable performance to models such as HuBERT in dialect classification. In addition, we provide an LLM baseline and a human linguist one. Our approach uses phonetic transcriptions produced by ASR systems and combines them with linguistic resources such as dialect feature maps, vowel history, and rules. Our findings indicate that, when linguistic information is provided, the LLM predictions improve. The human baseline shows that automatically generated transcriptions can be beneficial for such classifications, but also presents opportunities for improvement. 

---
# M-MiniGPT4: Multilingual VLLM Alignment via Translated Data 

**Authors**: Seung Hun Han, Youssef Mohamed, Mohamed Elhoseiny  

**Link**: [PDF](https://arxiv.org/pdf/2603.29467)  

**Abstract**: This paper presents a Multilingual Vision Large Language Model, named M-MiniGPT4. Our model exhibits strong vision-language understanding (VLU) capabilities across 11 languages. We utilize a mixture of native multilingual and translated data to push the multilingual VLU performance of the MiniGPT4 architecture. In addition, we propose a multilingual alignment training stage that uses parallel text corpora to further enhance the multilingual capabilities of our model. M-MiniGPT4 achieves 36% accuracy on the multilingual MMMU benchmark, outperforming state-of-the-art models in the same weight class, including foundation models released after the majority of this work was completed. We open-source our models, code, and translated datasets to facilitate future research in low-resource and multilingual settings. 

---
# MemFactory: Unified Inference & Training Framework for Agent Memory 

**Authors**: Ziliang Guo, Ziheng Li, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.29493)  

**Abstract**: Memory-augmented Large Language Models (LLMs) are essential for developing capable, long-term AI agents. Recently, applying Reinforcement Learning (RL) to optimize memory operations, such as extraction, updating, and retrieval, has emerged as a highly promising research direction. However, existing implementations remain highly fragmented and task-specific, lacking a unified infrastructure to streamline the integration, training, and evaluation of these complex pipelines. To address this gap, we present MemFactory, the first unified, highly modular training and inference framework specifically designed for memory-augmented agents. Inspired by the success of unified fine-tuning frameworks like LLaMA-Factory, MemFactory abstracts the memory lifecycle into atomic, plug-and-play components, enabling researchers to seamlessly construct custom memory agents via a "Lego-like" architecture. Furthermore, the framework natively integrates Group Relative Policy Optimization (GRPO) to fine-tune internal memory management policies driven by multi-dimensional environmental rewards. MemFactory provides out-of-the-box support for recent cutting-edge paradigms, including Memory-R1, RMM, and MemAgent. We empirically validate MemFactory on the open-source MemAgent architecture using its publicly available training and evaluation data. Across both in-domain and out-of-distribution evaluation sets, MemFactory consistently improves performance over the corresponding base models, with relative gains of up to 14.8%. By providing a standardized, extensible, and easy-to-use infrastructure, MemFactory significantly lowers the barrier to entry, paving the way for future innovations in memory-driven AI agents. 

---
# Beyond Idealized Patients: Evaluating LLMs under Challenging Patient Behaviors in Medical Consultations 

**Authors**: Yahan Li, Xinyi Jie, Wanjia Ruan, Xubei Zhang, Huaijie Zhu, Yicheng Gao, Chaohao Du, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29373)  

**Abstract**: Large language models (LLMs) are increasingly used for medical consultation and health information support. In this high-stakes setting, safety depends not only on medical knowledge, but also on how models respond when patient inputs are unclear, inconsistent, or misleading. However, most existing medical LLM evaluations assume idealized and well-posed patient questions, which limits their realism. In this paper, we study challenging patient behaviors that commonly arise in real medical consultations and complicate safe clinical reasoning. We define four clinically grounded categories of such behaviors: information contradiction, factual inaccuracy, self-diagnosis, and care resistance. For each behavior, we specify concrete failure criteria that capture unsafe responses. Building on four existing medical dialogue datasets, we introduce CPB-Bench (Challenging Patient Behaviors Benchmark), a bilingual (English and Chinese) benchmark of 692 multi-turn dialogues annotated with these behaviors. We evaluate a range of open- and closed-source LLMs on their responses to challenging patient utterances. While models perform well overall, we identify consistent, behavior-specific failure patterns, with particular difficulty in handling contradictory or medically implausible patient information. We also study four intervention strategies and find that they yield inconsistent improvements and can introduce unnecessary corrections. We release the dataset and code. 

---
# SNEAK: Evaluating Strategic Communication and Information Leakage in Large Language Models 

**Authors**: Adar Avsian, Larry Heck  

**Link**: [PDF](https://arxiv.org/pdf/2603.29846)  

**Abstract**: Large language models (LLMs) are increasingly deployed in multi-agent settings where communication must balance informativeness and secrecy. In such settings, an agent may need to signal information to collaborators while preventing an adversary from inferring sensitive details. However, existing LLM benchmarks primarily evaluate capabilities such as reasoning, factual knowledge, or instruction following, and do not directly measure strategic communication under asymmetric information. We introduce SNEAK (Secret-aware Natural language Evaluation for Adversarial Knowledge), a benchmark for evaluating selective information sharing in language models. In SNEAK, a model is given a semantic category, a candidate set of words, and a secret word, and must generate a message that indicates knowledge of the secret without revealing it too clearly. We evaluate generated messages using two simulated agents with different information states: an ally, who knows the secret and must identify the intended message, and a chameleon, who does not know the secret and attempts to infer it from the message. This yields two complementary metrics: utility, measuring how well the message communicates to collaborators, and leakage, measuring how much information it reveals to an adversary. Using this framework, we analyze the trade-off between informativeness and secrecy in modern language models and show that strategic communication under asymmetric information remains a challenging capability for current systems. Notably, human participants outperform all evaluated models by a large margin, achieving up to four times higher scores. 

---
# Authorship Impersonation via LLM Prompting does not Evade Authorship Verification Methods 

**Authors**: Baoyi Zeng, Andrea Nini  

**Link**: [PDF](https://arxiv.org/pdf/2603.29454)  

**Abstract**: Authorship verification (AV), the task of determining whether a questioned text was written by a specific individual, is a critical part of forensic linguistics. While manual authorial impersonation by perpetrators has long been a recognized threat in historical forensic cases, recent advances in large language models (LLMs) raise new challenges, as adversaries may exploit these tools to impersonate another's writing. This study investigates whether prompted LLMs can generate convincing authorial impersonations and whether such outputs can evade existing forensic AV systems. Using GPT-4o as the adversary model, we generated impersonation texts under four prompting conditions across three genres: emails, text messages, and social media posts. We then evaluated these outputs against both non-neural AV methods (n-gram tracing, Ranking-Based Impostors Method, LambdaG) and neural approaches (AdHominem, LUAR, STAR) within a likelihood-ratio framework. Results show that LLM-generated texts failed to sufficiently replicate authorial individuality to bypass established AV systems. We also observed that some methods achieved even higher accuracy when rejecting impersonation texts compared to genuine negative samples. Overall, these findings indicate that, despite the accessibility of LLMs, current AV systems remain robust against entry-level impersonation attempts across multiple genres. Furthermore, we demonstrate that this counter-intuitive resilience stems, at least in part, from the higher lexical diversity and entropy inherent in LLM-generated texts. 

---
# Is my model perplexed for the right reason? Contrasting LLMs' Benchmark Behavior with Token-Level Perplexity 

**Authors**: Zoë Prins, Samuele Punzo, Frank Wildenburg, Giovanni Cinà, Sandro Pezzelle  

**Link**: [PDF](https://arxiv.org/pdf/2603.29396)  

**Abstract**: Standard evaluations of Large language models (LLMs) focus on task performance, offering limited insight into whether correct behavior reflects appropriate underlying mechanisms and risking confirmation bias. We introduce a simple, principled interpretability framework based on token-level perplexity to test whether models rely on linguistically relevant cues. By comparing perplexity distributions over minimal sentence pairs differing in one or a few `pivotal' tokens, our method enables precise, hypothesis-driven analysis without relying on unstable feature-attribution techniques. Experiments on controlled linguistic benchmarks with several open-weight LLMs show that, while linguistically important tokens influence model behavior, they never fully explain perplexity shifts, revealing that models rely on heuristics other than the expected linguistic ones. 

---
# Dual Perspectives in Emotion Attribution: A Generator-Interpreter Framework for Cross-Cultural Analysis of Emotion in LLMs 

**Authors**: Aizirek Turdubaeva, Uichin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.29077)  

**Abstract**: Large language models (LLMs) are increasingly used in cross-cultural systems to understand and adapt to human emotions, which are shaped by cultural norms of expression and interpretation. However, prior work on emotion attribution has focused mainly on interpretation, overlooking the cultural background of emotion generators. This assumption of universality neglects variation in how emotions are expressed and perceived across nations. To address this gap, we propose a Generator-Interpreter framework that captures dual perspectives of emotion attribution by considering both expression and interpretation. We systematically evaluate six LLMs on an emotion attribution task using data from 15 countries. Our analysis reveals that performance variations depend on the emotion type and cultural context. Generator-interpreter alignment effects are present; the generator's country of origin has a stronger impact on performance. We call for culturally sensitive emotion modeling in LLM-based systems to improve robustness and fairness in emotion understanding across diverse cultural contexts. 

---
# Long-Document QA with Chain-of-Structured-Thought and Fine-Tuned SLMs 

**Authors**: Zhuowen Liang, Xiaotian Lin, Zhengxuan Zhang, Yuyu Luo, Haixun Wang, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29232)  

**Abstract**: Large language models (LLMs) are widely applied to data analytics over documents, yet direct reasoning over long, noisy documents remains brittle and error-prone. Hence, we study document question answering (QA) that consolidates dispersed evidence into a structured output (e.g., a table, graph, or chunks) to support reliable, verifiable QA. We propose a two-pillar framework, LiteCoST, to achieve both high accuracy and low latency with small language models (SLMs). Pillar 1: Chain-of-Structured-Thought (CoST). We introduce a CoST template, a schema-aware instruction that guides a strong LLM to produce both a step-wise CoST trace and the corresponding structured output. The process induces a minimal structure, normalizes entities/units, aligns records, serializes the output, and verifies/refines it, yielding auditable supervision. Pillar 2: SLM fine-tuning. The compact models are trained on LLM-generated CoST data in two stages: Supervised Fine-Tuning for structural alignment, followed by Group Relative Policy Optimization (GRPO) incorporating triple rewards for answer/format quality and process consistency. By distilling structure-first behavior into SLMs, this approach achieves LLM-comparable quality on multi-domain long-document QA using 3B/7B SLMs, while delivering 2-4x lower latency than GPT-4o and DeepSeek-R1 (671B). The code is available at this https URL. 

---
# The Model Says Walk: How Surface Heuristics Override Implicit Constraints in LLM Reasoning 

**Authors**: Yubo Li, Lu Zhang, Tianchong Jiang, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2603.29025)  

**Abstract**: Large language models systematically fail when a salient surface cue conflicts with an unstated feasibility constraint. We study this through a diagnose-measure-bridge-treat framework. Causal-behavioral analysis of the ``car wash problem'' across six models reveals approximately context-independent sigmoid heuristics: the distance cue exerts 8.7 to 38 times more influence than the goal, and token-level attribution shows patterns more consistent with keyword associations than compositional inference. The Heuristic Override Benchmark (HOB) -- 500 instances spanning 4 heuristic by 5 constraint families with minimal pairs and explicitness gradients -- demonstrates generality across 14 models: under strict evaluation (10/10 correct), no model exceeds 75%, and presence constraints are hardest (44%). A minimal hint (e.g., emphasizing the key object) recovers +15 pp on average, suggesting the failure lies in constraint inference rather than missing knowledge; 12/14 models perform worse when the constraint is removed (up to -39 pp), revealing conservative bias. Parametric probes confirm that the sigmoid pattern generalizes to cost, efficiency, and semantic-similarity heuristics; goal-decomposition prompting recovers +6 to 9 pp by forcing models to enumerate preconditions before answering. Together, these results characterize heuristic override as a systematic reasoning vulnerability and provide a benchmark for measuring progress toward resolving it. 

---
# Reward-Based Online LLM Routing via NeuralUCB 

**Authors**: Ming-Hua Tsai, Phat Tran  

**Link**: [PDF](https://arxiv.org/pdf/2603.30035)  

**Abstract**: This study investigates the use of NeuralUCB for cost-aware large language model (LLM) routing. Existing routing approaches can be broadly grouped into supervised routing methods and partial-feedback methods, each with different tradeoffs in efficiency and adaptivity. We implement a NeuralUCB-based routing policy and evaluate it on RouterBench under a simulated online setting. Experimental results show that the proposed method consistently outperforms random and min-cost baselines in utility reward. Compared with the max-quality reference, our method achieves substantially lower inference cost while maintaining competitive reward. These findings suggest that NeuralUCB is a promising approach for cost-aware LLM routing, while also highlighting remaining challenges in action discrimination and exploration. 

---
# GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification 

**Authors**: Iordanis Fostiropoulos, Muhammad Rafay Azhar, Abdalaziz Sawwan, Boyu Fang, Yuchen Liu, Jiayi Liu, Hanchao Yu, Qi Guo, Jianyu Wang, Fei Liu, Xiangjun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29112)  

**Abstract**: We introduce GISTBench, a benchmark for evaluating Large Language Models' (LLMs) ability to understand users from their interaction histories in recommendation systems. Unlike traditional RecSys benchmarks that focus on item prediction accuracy, our benchmark evaluates how well LLMs can extract and verify user interests from engagement data. We propose two novel metric families: Interest Groundedness (IG), decomposed into precision and recall components to separately penalize hallucinated interest categories and reward coverage, and Interest Specificity (IS), which assesses the distinctiveness of verified LLM-predicted user profiles. We release a synthetic dataset constructed on real user interactions on a global short-form video platform. Our dataset contains both implicit and explicit engagement signals and rich textual descriptions. We validate our dataset fidelity against user surveys, and evaluate eight open-weight LLMs spanning 7B to 120B parameters. Our findings reveal performance bottlenecks in current LLMs, particularly their limited ability to accurately count and attribute engagement signals across heterogeneous interaction types. 

---
# FlowPIE: Test-Time Scientific Idea Evolution with Flow-Guided Literature Exploration 

**Authors**: Qiyao Wang, Hongbo Wang, Longze Chen, Zhihao Yang, Guhong Chen, Hamid Alinejad-Rokny, Hui Li, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29557)  

**Abstract**: Scientific idea generation (SIG) is critical to AI-driven autonomous research, yet existing approaches are often constrained by a static retrieval-then-generation paradigm, leading to homogeneous and insufficiently divergent ideas. In this work, we propose FlowPIE, a tightly coupled retrieval-generation framework that treats literature exploration and idea generation as a co-evolving process. FlowPIE expands literature trajectories via a flow-guided Monte Carlo Tree Search (MCTS) inspired by GFlowNets, using the quality of current ideas assessed by an LLM-based generative reward model (GRM) as a supervised signal to guide adaptive retrieval and construct a diverse, high-quality initial population. Based on this population, FlowPIE models idea generation as a test-time idea evolution process, applying selection, crossover, and mutation with the isolation island paradigm and GRM-based fitness computation to incorporate cross-domain knowledge. It effectively mitigates the information cocoons arising from over-reliance on parametric knowledge and static literature. Extensive evaluations demonstrate that FlowPIE consistently produces ideas with higher novelty, feasibility and diversity compared to strong LLM-based and agent-based frameworks, while enabling reward scaling during test time. 

---
# Designing FSMs Specifications from Requirements with GPT 4.0 

**Authors**: Omer Nguena Timo, Paul-Alexis Rodriguez, Florent Avellaneda  

**Link**: [PDF](https://arxiv.org/pdf/2603.29140)  

**Abstract**: Finite state machines (FSM) are executable formal specifications of reactive systems. These machines are designed based on systems' requirements. The requirements are often recorded in textual documents written in natural languages. FSMs play a crucial role in different phases of the model-driven system engineering (MDE). For example, they serve to automate testing activities. FSM quality is critical: the lower the quality of FSM, the higher the number of faults surviving the testing phase and the higher the risk of failure of the systems in production, which could lead to catastrophic scenarios. Therefore, this paper leverages recent advances in the domain of LLM to propose an LLM-based framework for designing FSMs from requirements. The framework also suggests an expert-centric approach based on FSM mutation and test generation for repairing the FSMs produced by LLMs. This paper also provides an experimental analysis and evaluation of LLM's capacities in performing the tasks presented in the framework and FSM repair via various methods. The paper presents experimental results with simulated data. These results and methods bring a new analysis and vision of LLMs that are useful for further development of machine learning technology and its applications to MDE. 

---
# Spark-LLM-Eval: A Distributed Framework for Statistically Rigorous Large Language Model Evaluation 

**Authors**: Subhadip Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2603.28769)  

**Abstract**: Evaluating large language models at scale remains a practical bottleneck for many organizations. While existing evaluation frameworks work well for thousands of examples, they struggle when datasets grow to hundreds of thousands or millions of samples. This scale is common when assessing model behavior across diverse domains or conducting comprehensive regression testing. We present Spark-LLM-Eval, a distributed evaluation framework built natively on Apache Spark. The system treats evaluation as a data-parallel problem, partitioningexamplesacrossexecutorsandaggregatingresultswithproperstatistical accounting. Beyond raw throughput, we emphasize statistical rigor: every reported metric includes bootstrap confidence intervals, and model comparisons come with appropriate significance tests (paired t-tests, McNemar's test, or Wilcoxon signed-rank, depending on the metric type). The framework also addresses the cost problem inherent in LLM evaluation through content-addressable response caching backed by Delta Lake, which allows iterating on metric definitions without re-running inference. We describe the system architecture, the statistical methodology, and report benchmark results showing linear scaling with cluster size. The framework and all evaluation code are available as open source. 

---
# Sima AIunty: Caste Audit in LLM-Driven Matchmaking 

**Authors**: Atharva Naik, Shounok Kar, Varnika Sharma, Ashwin Rajadesingan, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2603.29288)  

**Abstract**: Social and personal decisions in relational domains such as matchmaking are deeply entwined with cultural norms and historical hierarchies, and can potentially be shaped by algorithmic and AI-mediated assessments of compatibility, acceptance, and stability. In South Asian contexts, caste remains a central aspect of marital decision-making, yet little is known about how contemporary large language models (LLMs) reproduce or disrupt caste-based stratification in such settings. In this work, we conduct a controlled audit of caste bias in LLM-mediated matchmaking evaluations using real-world matrimonial profiles. We vary caste identity across Brahmin, Kshatriya, Vaishya, Shudra, and Dalit, and income across five buckets, and evaluate five LLM families (GPT, Gemini, Llama, Qwen, and BharatGPT). Models are prompted to assess profiles along dimensions of social acceptance, marital stability, and cultural compatibility. Our analysis reveals consistent hierarchical patterns across models: same-caste matches are rated most favorably, with average ratings up to 25% higher (on a 10-point scale) than inter-caste matches, which are further ordered according to traditional caste hierarchy. These findings highlight how existing caste hierarchies are reproduced in LLM decision-making and underscore the need for culturally grounded evaluation and intervention strategies in AI systems deployed in socially sensitive domains, where such systems risk reinforcing historical forms of exclusion. 

---
# C-TRAIL: A Commonsense World Framework for Trajectory Planning in Autonomous Driving 

**Authors**: Zhihong Cui, Haoran Tang, Tianyi Li, Yushuai Li, Peiyuan Guan, Amir Taherkordi, Tor Skeie  

**Link**: [PDF](https://arxiv.org/pdf/2603.29908)  

**Abstract**: Trajectory planning for autonomous driving increasingly leverages large language models (LLMs) for commonsense reasoning, yet LLM outputs are inherently unreliable, posing risks in safety-critical applications. We propose C-TRAIL, a framework built on a Commonsense World that couples LLM-derived commonsense with a trust mechanism to guide trajectory planning. C-TRAIL operates through a closed-loop Recall, Plan, and Update cycle: the Recall module queries an LLM for semantic relations and quantifies their reliability via a dual-trust mechanism; the Plan module injects trust-weighted commonsense into Monte Carlo Tree Search (MCTS) through a Dirichlet trust policy; and the Update module adaptively refines trust scores and policy parameters from environmental feedback. Experiments on four simulated scenarios in Highway-env and two real-world levelXData datasets (highD, rounD) show that C-TRAIL consistently outperforms state-of-the-art baselines, reducing ADE by 40.2%, FDE by 51.7%, and improving SR by 16.9 percentage points on average. The source code is available at this https URL. 

---
# Beyond the Steeper Curve: AI-Mediated Metacognitive Decoupling and the Limits of the Dunning-Kruger Metaphor 

**Authors**: Christopher Koch  

**Link**: [PDF](https://arxiv.org/pdf/2603.29681)  

**Abstract**: The common claim that generative AI simply amplifies the Dunning-Kruger effect is too coarse to capture the available evidence. The clearest findings instead suggest that large language model (LLM) use can improve observable output and short-term task performance while degrading metacognitive accuracy and flattening the classic competence-confidence gradient across skill groups. This paper synthesizes evidence from human-AI interaction, learning research, and model evaluation, and proposes the working model of AI-mediated metacognitive decoupling: a widening gap among produced output, underlying understanding, calibration accuracy, and self-assessed ability. This four-variable account better explains overconfidence, over- and under-reliance, crutch effects, and weak transfer than the simpler metaphor of a uniformly steeper Dunning-Kruger curve. The paper concludes with implications for tool design, assessment, and knowledge work. 

---
# Measuring the metacognition of AI 

**Authors**: Richard Servajean, Philippe Servajean  

**Link**: [PDF](https://arxiv.org/pdf/2603.29693)  

**Abstract**: A robust decision-making process must take into account uncertainty, especially when the choice involves inherent risks. Because artificial Intelligence (AI) systems are increasingly integrated into decision-making workflows, managing uncertainty relies more and more on the metacognitive capabilities of these systems; i.e, their ability to assess the reliability of and regulate their own decisions. Hence, it is crucial to employ robust methods to measure the metacognitive abilities of AI. This paper is primarily a methodological contribution arguing for the adoption of the meta-d' framework, or its model-free alternatives, as the gold standard for assessing the metacognitive sensitivity of AIs--the ability to generate confidence ratings that distinguish correct from incorrect responses. Moreover, we propose to leverage signal detection theory (SDT) to measure the ability of AIs to spontaneously regulate their decisions based on uncertainty and risk. To demonstrate the practical utility of these psychophysical frameworks, we conduct two series of experiments on three large language models (LLMs)--GPT-5, DeepSeek-V3.2-Exp, and Mistral-Medium-2508. In the first experiments, LLMs performed a primary judgment followed by a confidence rating. In the second, LLMs only performed the primary judgment, while we manipulated the risk associated with either response. On the one hand, applying the meta-d' framework allows us to conduct comparisons along three axes: comparing an LLM to optimality, comparing different LLMs on a given task, and comparing the same LLM across different tasks. On the other hand, SDT allows us to assess whether LLMs become more conservative when risks are high. 

---
# Learning to Generate Formally Verifiable Step-by-Step Logic Reasoning via Structured Formal Intermediaries 

**Authors**: Luoxin Chen, Yichi Zhou, Huishuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29500)  

**Abstract**: Large language models (LLMs) have recently demonstrated impressive performance on complex, multi-step reasoning tasks, especially when post-trained with outcome-rewarded reinforcement learning Guo et al. 2025. However, it has been observed that outcome rewards often overlook flawed intermediate steps, leading to unreliable reasoning steps even when final answers are correct. To address this unreliable reasoning, we propose PRoSFI (Process Reward over Structured Formal Intermediates), a novel reward method that enhances reasoning reliability without compromising accuracy. Instead of generating formal proofs directly, which is rarely accomplishable for a modest-sized (7B) model, the model outputs structured intermediate steps aligned with its natural language reasoning. Each step is then verified by a formal prover. Only fully validated reasoning chains receive high rewards. The integration of formal verification guides the model towards generating step-by-step machine-checkable proofs, thereby yielding more credible final answers. PRoSFI offers a simple and effective approach to training trustworthy reasoning models. 

---
# Beyond pass@1: A Reliability Science Framework for Long-Horizon LLM Agents 

**Authors**: Aaditya Khanal, Yangyang Tao, Junxiu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.29231)  

**Abstract**: Existing benchmarks measure capability -- whether a model succeeds on a single attempt -- but production deployments
require reliability -- consistent success across repeated attempts on tasks of varying duration. We show these
properties diverge systematically as task duration grows, and that pass@1 on short tasks is structurally blind to
this divergence.
We introduce a reliability science framework for long-horizon LLM agents with four metrics: Reliability Decay Curve
(RDC), Variance Amplification Factor (VAF), Graceful Degradation Score (GDS), and Meltdown Onset Point (MOP). We
evaluate 10 models across 23,392 episodes on a 396-task benchmark spanning four duration buckets and three domains.
Key findings: (1) reliability decay is domain-stratified -- SE GDS drops from 0.90 to 0.44 while document processing
is nearly flat (0.74 to 0.71); (2) VAF bifurcates by capability tier -- high VAF is a capability signature, not an
instability signal; (3) capability and reliability rankings diverge substantially, with multi-rank inversions at long
horizons; (4) frontier models have the highest meltdown rates (up to 19%) because they attempt ambitious multi-step
strategies that sometimes spiral; and (5) memory scaffolds universally hurt long-horizon performance across all 10
models. These results motivate reliability as a first-class evaluation dimension alongside capability. 

---
# PSPA-Bench: A Personalized Benchmark for Smartphone GUI Agent 

**Authors**: Hongyi Nie, Xunyuan Liu, Yudong Bai, Yaqing Wang, Yang Liu, Quanming Yao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29318)  

**Abstract**: Smartphone GUI agents execute tasks by operating directly on app interfaces, offering a path to broad capability without deep system integration. However, real-world smartphone use is highly personalized: users adopt diverse workflows and preferences, challenging agents to deliver customized assistance rather than generic solutions. Existing GUI agent benchmarks cannot adequately capture this personalization dimension due to sparse user-specific data and the lack of fine-grained evaluation metrics. To address this gap, we present PSPA-Bench, the benchmark dedicated to evaluating personalization in smartphone GUI agents. PSPA-Bench comprises over 12,855 personalized instructions aligned with real-world user behaviors across 10 representative daily-use scenarios and 22 mobile apps, and introduces a structure-aware process evaluation method that measures agents' personalized capabilities at a fine-grained level. Through PSPA-Bench, we benchmark 11 state-of-the-art GUI agents. Results reveal that current methods perform poorly under personalized settings, with even the strongest agent achieving limited success. Our analysis further highlights three directions for advancing personalized GUI agents: (1) reasoning-oriented models consistently outperform general LLMs, (2) perception remains a simple yet critical capability, and (3) reflection and long-term memory mechanisms are key to improving adaptation. Together, these findings establish PSPA-Bench as a foundation for systematic study and future progress in personalized GUI agents. 

---
# Knowledge database development by large language models for countermeasures against viruses and marine toxins 

**Authors**: Hung N. Do, Jessica Z. Kubicek-Sutherland, S. Gnanakaran  

**Link**: [PDF](https://arxiv.org/pdf/2603.29149)  

**Abstract**: Access to the most up-to-date information on medical countermeasures is important for the research and development of effective treatments for viruses and marine toxins. However, there is a lack of comprehensive databases that curate data on viruses and marine toxins, making decisions on medical countermeasures slow and difficult. In this work, we employ two large language models (LLMs) of ChatGPT and Grok to design two comprehensive databases of therapeutic countermeasures for five viruses of Lassa, Marburg, Ebola, Nipah, and Venezuelan equine encephalitis, as well as marine toxins. With high-level human-provided inputs, the two LLMs identify public databases containing data on the five viruses and marine toxins, collect relevant information from these databases and the literature, iteratively cross-validate the collected information, and design interactive webpages for easy access to the curated, comprehensive databases. Notably, the ChatGPT LLM is employed to design agentic AI workflows (consisting of two AI agents for research and decision-making) to rank countermeasures for viruses and marine toxins in the databases. Together, our work explores the potential of LLMs as a scalable, updatable approach for building comprehensive knowledge databases and supporting evidence-based decision-making. 

---
# Route-Induced Density and Stability (RIDE): Controlled Intervention and Mechanism Analysis of Routing-Style Meta Prompts on LLM Internal States 

**Authors**: Dianxing Zhang, Gang Li, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.29206)  

**Abstract**: Routing is widely used to scale large language models, from Mixture-of-Experts gating to multi-model/tool selection. A common belief is that routing to a task ``expert'' activates sparser internal computation and thus yields more certain and stable outputs (the Sparsity--Certainty Hypothesis). We test this belief by injecting routing-style meta prompts as a textual proxy for routing signals in front of frozen instruction-tuned LLMs. We quantify (C1) internal density via activation sparsity, (C2) domain-keyword attention, and (C3) output stability via predictive entropy and semantic variation. On a RouterEval subset with three instruction-tuned models (Qwen3-8B, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.2), meta prompts consistently densify early/middle-layer representations rather than increasing sparsity; natural-language expert instructions are often stronger than structured tags. Attention responses are heterogeneous: Qwen/Llama reduce keyword attention, while Mistral reinforces it. Finally, the densification--stability link is weak and appears only in Qwen, with near-zero correlations in Llama and Mistral. We present RIDE as a diagnostic probe for calibrating routing design and uncertainty estimation. 

---
# Webscraper: Leverage Multimodal Large Language Models for Index-Content Web Scraping 

**Authors**: Guan-Lun Huang, Yuh-Jzer Joung  

**Link**: [PDF](https://arxiv.org/pdf/2603.29161)  

**Abstract**: Modern web scraping struggles with dynamic, interactive websites that require more than static HTML parsing. Current methods are often brittle and require manual customization for each site. To address this, we introduce Webscraper, a framework designed to handle the challenges of modern, dynamic web applications. It leverages a Multimodal Large Language Model (MLLM) to autonomously navigate interactive interfaces, invoke specialized tools, and perform structured data extraction in environments where traditional scrapers are ineffective. Webscraper utilizes a structured five-stage prompting procedure and a set of custom-built tools to navigate and extract data from websites following the common ``index-and-content'' architecture. Our experiments, conducted on six news websites, demonstrate that the full Webscraper framework, equipped with both our guiding prompt and specialized tools, achieves a significant improvement in extraction accuracy over the baseline agent Anthropic's Computer Use. We also applied the framework to e-commerce platforms to validate its generalizability. 

---
# PAR$^2$-RAG: Planned Active Retrieval and Reasoning for Multi-Hop Question Answering 

**Authors**: Xingyu Li, Rongguang Wang, Yuying Wang, Mengqing Guo, Chenyang Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.29085)  

**Abstract**: Large language models (LLMs) remain brittle on multi-hop question answering (MHQA), where answering requires combining evidence across documents through retrieval and reasoning. Iterative retrieval systems can fail by locking onto an early low-recall trajectory and amplifying downstream errors, while planning-only approaches may produce static query sets that cannot adapt when intermediate evidence changes. We propose \textbf{Planned Active Retrieval and Reasoning RAG (PAR$^2$-RAG)}, a two-stage framework that separates \emph{coverage} from \emph{commitment}. PAR$^2$-RAG first performs breadth-first anchoring to build a high-recall evidence frontier, then applies depth-first refinement with evidence sufficiency control in an iterative loop. Across four MHQA benchmarks, PAR$^2$-RAG consistently outperforms existing state-of-the-art baselines, compared with IRCoT, PAR$^2$-RAG achieves up to \textbf{23.5\%} higher accuracy, with retrieval gains of up to \textbf{10.5\%} in NDCG. 

---
# REFINE: Real-world Exploration of Interactive Feedback and Student Behaviour 

**Authors**: Fares Fawzi, Seyed Parsa Neshaei, Marta Knezevic, Tanya Nazaretsky, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2603.29142)  

**Abstract**: Formative feedback is central to effective learning, yet providing timely, individualised feedback at scale remains a persistent challenge. While recent work has explored the use of large language models (LLMs) to automate feedback, most existing systems still conceptualise feedback as a static, one-way artifact, offering limited support for interpretation, clarification, or follow-up. In this work, we introduce REFINE, a locally deployable, multi-agent feedback system built on small, open-source LLMs that treats feedback as an interactive process. REFINE combines a pedagogically-grounded feedback generation agent with an LLM-as-a-judge-guided regeneration loop using a human-aligned judge, and a self-reflective tool-calling interactive agent that supports student follow-up questions with context-aware, actionable responses. We evaluate REFINE through controlled experiments and an authentic classroom deployment in an undergraduate computer science course. Automatic evaluations show that judge-guided regeneration significantly improves feedback quality, and that the interactive agent produces efficient, high-quality responses comparable to a state-of-the-art closed-source model. Analysis of real student interactions further reveals distinct engagement patterns and indicates that system-generated feedback systematically steers subsequent student inquiry. Our findings demonstrate the feasibility and effectiveness of multi-agent, tool-augmented feedback systems for scalable, interactive feedback. 

---
# SimMOF: AI agent for Automated MOF Simulations 

**Authors**: Jaewoong Lee, Taeun Bae, Jihan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.29152)  

**Abstract**: Metal-organic frameworks (MOFs) offer a vast design space, and as such, computational simulations play a critical role in predicting their structural and physicochemical properties. However, MOF simulations remain difficult to access because reliable analysis require expert decisions for workflow construction, parameter selection, tool interoperability, and the preparation of computational ready structures. Here, we introduce SimMOF, a large language model based multi agent framework that automates end-to-end MOF simulation workflows from natural language queries. SimMOF translates user requests into dependency aware plans, generates runnable inputs, orchestrates multiple agents to execute simulations, and summarizes results with analysis aligned to the user query. Through representative case studies, we show that SimMOF enables adaptive and cognitively autonomous workflows that reflect the iterative and decision driven behavior of human researchers and as such provides a scalable foundation for data driven MOF research. 

---
# SciVisAgentBench: A Benchmark for Evaluating Scientific Data Analysis and Visualization Agents 

**Authors**: Kuangshi Ai, Haichao Miao, Kaiyuan Tang, Nathaniel Gorski, Jianxin Sun, Guoxi Liu, Helgi I. Ingolfsson, David Lenz, Hanqi Guo, Hongfeng Yu, Teja Leburu, Michael Molash, Bei Wang, Tom Peterka, Chaoli Wang, Shusen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29139)  

**Abstract**: Recent advances in large language models (LLMs) have enabled agentic systems that translate natural language intent into executable scientific visualization (SciVis) tasks. Despite rapid progress, the community lacks a principled and reproducible benchmark for evaluating these emerging SciVis agents in realistic, multi-step analysis settings. We present SciVisAgentBench, a comprehensive and extensible benchmark for evaluating scientific data analysis and visualization agents. Our benchmark is grounded in a structured taxonomy spanning four dimensions: application domain, data type, complexity level, and visualization operation. It currently comprises 108 expert-crafted cases covering diverse SciVis scenarios. To enable reliable assessment, we introduce a multimodal outcome-centric evaluation pipeline that combines LLM-based judging with deterministic evaluators, including image-based metrics, code checkers, rule-based verifiers, and case-specific evaluators. We also conduct a validity study with 12 SciVis experts to examine the agreement between human and LLM judges. Using this framework, we evaluate representative SciVis agents and general-purpose coding agents to establish initial baselines and reveal capability gaps. SciVisAgentBench is designed as a living benchmark to support systematic comparison, diagnose failure modes, and drive progress in agentic SciVis. The benchmark is available at this https URL. 

---
# Mimosa Framework: Toward Evolving Multi-Agent Systems for Scientific Research 

**Authors**: Martin Legrand, Tao Jiang, Matthieu Feraud, Benjamin Navet, Yousouf Taghzouti, Fabien Gandon, Elise Dumont, Louis-Félix Nothias  

**Link**: [PDF](https://arxiv.org/pdf/2603.28986)  

**Abstract**: Current Autonomous Scientific Research (ASR) systems, despite leveraging large language models (LLMs) and agentic architectures, remain constrained by fixed workflows and toolsets that prevent adaptation to evolving tasks and environments. We introduce Mimosa, an evolving multi-agent framework that automatically synthesizes task-specific multi-agent workflows and iteratively refines them through experimental feedback. Mimosa leverages the Model Context Protocol (MCP) for dynamic tool discovery, generates workflow topologies via a meta-orchestrator, executes subtasks through code-generating agents that invoke available tools and scientific software libraries, and scores executions with an LLM-based judge whose feedback drives workflow refinement. On ScienceAgentBench, Mimosa achieves a success rate of 43.1% with DeepSeek-V3.2, surpassing both single-agent baselines and static multi-agent configurations. Our results further reveal that models respond heterogeneously to multi-agent decomposition and iterative learning, indicating that the benefits of workflow evolution depend on the capabilities of the underlying execution model. Beyond these benchmarks, Mimosa modular architecture and tool-agnostic design make it readily extensible, and its fully logged execution traces and archived workflows support auditability by preserving every analytical step for inspection and potential replication. Combined with domain-expert guidance, the framework has the potential to automate a broad range of computationally accessible scientific tasks across disciplines. Released as a fully open-source platform, Mimosa aims to provide an open foundation for community-driven ASR. 

---
# Hybrid Framework for Robotic Manipulation: Integrating Reinforcement Learning and Large Language Models 

**Authors**: Md Saad, Sajjad Hussain, Mohd Suhaib  

**Link**: [PDF](https://arxiv.org/pdf/2603.30022)  

**Abstract**: This paper introduces a new hybrid framework that combines Reinforcement Learning (RL) and Large Language Models (LLMs) to improve robotic manipulation tasks. By utilizing RL for accurate low-level control and LLMs for high level task planning and understanding of natural language, the proposed framework effectively connects low-level execution with high-level reasoning in robotic systems. This integration allows robots to understand and carry out complex, human-like instructions while adapting to changing environments in real time. The framework is tested in a PyBullet-based simulation environment using the Franka Emika Panda robotic arm, with various manipulation scenarios as benchmarks. The results show a 33.5% decrease in task completion time and enhancements of 18.1% and 36.4% in accuracy and adaptability, respectively, when compared to systems that use only RL. These results underscore the potential of LLM-enhanced robotic systems for practical applications, making them more efficient, adaptable, and capable of interacting with humans. Future research will aim to explore sim-to-real transfer, scalability, and multi-robot systems to further broaden the framework's applicability. 

---
# Aligned, Orthogonal or In-conflict: When can we safely optimize Chain-of-Thought? 

**Authors**: Max Kaufmann, David Lindner, Roland S. Zimmermann, and Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2603.30036)  

**Abstract**: Chain-of-Thought (CoT) monitoring, in which automated systems monitor the CoT of an LLM, is a promising approach for effectively overseeing AI systems. However, the extent to which a model's CoT helps us oversee the model - the monitorability of the CoT - can be affected by training, for instance by the model learning to hide important features of its reasoning. We propose and empirically validate a conceptual framework for predicting when and why this occurs. We model LLM post-training as an RL environment where the reward decomposes into two terms: one term depending on final outputs and another term depending on the CoT. Our framework allows us to classify these two terms as "aligned", "orthogonal", or "in-conflict" before training. We predict that training with in-conflict terms will reduce monitorability, orthogonal terms will not affect it, and aligned terms will improve it. To validate our framework, we use it to classify a set of RL environments, train LLMs within those environments, and evaluate how training affects CoT monitorability. We find that (1) training with "in-conflict" reward terms reduces CoT monitorability and (2) optimizing in-conflict reward terms is difficult. 

---
# Bethe Ansatz with a Large Language Model 

**Authors**: Balázs Pozsgay, István Vona  

**Link**: [PDF](https://arxiv.org/pdf/2603.29932)  

**Abstract**: We explore the capability of a Large Language Model (LLM) to perform specific computations in mathematical physics: the task is to compute the coordinate Bethe Ansatz solution of selected integrable spin chain models. We select three integrable Hamiltonians for which the solutions were unpublished; two of the Hamiltonians are actually new. We observed that the LLM semi-autonomously solved the task in all cases, with a few mistakes along the way. These were corrected after the human researchers spotted them. The results of the LLM were checked against exact diagonalization (performed by separate programs), and the derivations were also checked by the authors. The Bethe Ansatz solutions are interesting in themselves. Our second model manifestly breaks left-right invariance, but it is PT-symmetric, therefore its solution could be interesting for applications in Generalized Hydrodynamics. And our third model is solved by a special form of the nested Bethe Ansatz, where the model is interacting, but the nesting level has a free fermionic structure lacking $U(1)$-invariance. This structure appears to be unique and it was found by the LLM. We used ChatGPT 5.2 Pro and 5.4 Pro by OpenAI. 

---
# KEditVis: A Visual Analytics System for Knowledge Editing of Large Language Models 

**Authors**: Zhenning Chen, Hanbei Zhan, Yanwei Huang, Xin Wu, Dazhen Deng, Di Weng, Yingcai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29689)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capabilities in factual question answering, yet they sometimes provide incorrect responses. To address this issue, knowledge editing techniques have emerged as effective methods for correcting factual information in LLMs. However, typical knowledge editing workflows struggle with identifying the optimal set of model layers for editing and rely on summary indicators that provide insufficient guidance. This lack of transparency hinders effective comparison and identification of optimal editing strategies. In this paper, we present KEditVis, a novel visual analytics system designed to assist users in gaining a deeper understanding of knowledge editing through interactive visualizations, improving editing outcomes, and discovering valuable insights for the future development of knowledge editing algorithms. With KEditVis, users can select appropriate layers as the editing target, explore the reasons behind ineffective edits, and perform more targeted and effective edits. Our evaluation, including usage scenarios, expert interviews, and a user study, validates the effectiveness and usability of the system. 

---
# Learn2Fold: Structured Origami Generation with World Model Planning 

**Authors**: Yanjia Huang, Yunuo Chen, Ying Jiang, Jinru Han, Zhengzhong Tu, Yin Yang, Chenfanfu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29585)  

**Abstract**: The ability to transform a flat sheet into a complex three-dimensional structure is a fundamental test of physical intelligence. Unlike cloth manipulation, origami is governed by strict geometric axioms and hard kinematic constraints, where a single invalid crease or collision can invalidate the entire folding sequence. As a result, origami demands long-horizon constructive reasoning that jointly satisfies precise physical laws and high-level semantic intent. Existing approaches fall into two disjoint paradigms: optimization-based methods enforce physical validity but require dense, precisely specified inputs, making them unsuitable for sparse natural language descriptions, while generative foundation models excel at semantic and perceptual synthesis yet fail to produce long-horizon, physics-consistent folding processes. Consequently, generating valid origami folding sequences directly from text remains an open challenge. To address this gap, we introduce Learn2Fold, a neuro-symbolic framework that formulates origami folding as conditional program induction over a crease-pattern graph. Our key insight is to decouple semantic proposal from physical verification. A large language model generates candidate folding programs from abstract text prompts, while a learned graph-structured world model serves as a differentiable surrogate simulator that predicts physical feasibility and failure modes before execution. Integrated within a lookahead planning loop, Learn2Fold enables robust generation of physically valid folding sequences for complex and out-of-distribution patterns, demonstrating that effective spatial intelligence arises from the synergy between symbolic reasoning and grounded physical simulation. 

---
# IMAGAgent: Orchestrating Multi-Turn Image Editing via Constraint-Aware Planning and Reflection 

**Authors**: Fei Shen, Chengyu Xie, Lihong Wang, Zhanyi Zhang, Xin Jiang, Xiaoyu Du, Jinhui Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29602)  

**Abstract**: Existing multi-turn image editing paradigms are often confined to isolated single-step execution. Due to a lack of context-awareness and closed-loop feedback mechanisms, they are prone to error accumulation and semantic drift during multi-turn interactions, ultimately resulting in severe structural distortion of the generated images. For that, we propose \textbf{IMAGAgent}, a multi-turn image editing agent framework based on a "plan-execute-reflect" closed-loop mechanism that achieves deep synergy among instruction parsing, tool scheduling, and adaptive correction within a unified pipeline. Specifically, we first present a constraint-aware planning module that leverages a vision-language model (VLM) to precisely decompose complex natural language instructions into a series of executable sub-tasks, governed by target singularity, semantic atomicity, and visual perceptibility. Then, the tool-chain orchestration module dynamically constructs execution paths based on the current image, the current sub-task, and the historical context, enabling adaptive scheduling and collaborative operation among heterogeneous operation models covering image retrieval, segmentation, detection, and editing. Finally, we devise a multi-expert collaborative reflection mechanism where a central large language model (LLM) receives the image to be edited and synthesizes VLM critiques into holistic feedback, simultaneously triggering fine-grained self-correction and recording feedback outcomes to optimize future decisions. Extensive experiments on our constructed \textbf{MTEditBench} and the MagicBrush dataset demonstrate that IMAGAgent achieves performance significantly superior to existing methods in terms of instruction consistency, editing precision, and overall quality. The code is available at this https URL. 

---
# Adversarial Prompt Injection Attack on Multimodal Large Language Models 

**Authors**: Meiwen Ding, Song Xia, Chenqi Kong, Xudong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.29418)  

**Abstract**: Although multimodal large language models (MLLMs) are increasingly deployed in real-world applications, their instruction-following behavior leaves them vulnerable to prompt injection attacks. Existing prompt injection methods predominantly rely on textual prompts or perceptible visual prompts that are observable by human users. In this work, we study imperceptible visual prompt injection against powerful closed-source MLLMs, where adversarial instructions are embedded in the visual modality. Our method adaptively embeds the malicious prompt into the input image via a bounded text overlay to provide semantic guidance. Meanwhile, the imperceptible visual perturbation is iteratively optimized to align the feature representation of the attacked image with those of the malicious visual and textual targets at both coarse- and fine-grained levels. Specifically, the visual target is instantiated as a text-rendered image and progressively refined during optimization to more faithfully represent the desired semantics and improve transferability. Extensive experiments on two multimodal understanding tasks across multiple closed-source MLLMs demonstrate the superior performance of our approach compared to existing methods. 

---
# Security in LLM-as-a-Judge: A Comprehensive SoK 

**Authors**: Aiman Almasoud, Antony Anju, Marco Arazzi, Mert Cihangiroglu, Vignesh Kumar Kembu, Serena Nicolazzo, Antonino Nocera, Vinod P., Saraga Sakthidharan  

**Link**: [PDF](https://arxiv.org/pdf/2603.29403)  

**Abstract**: LLM-as-a-Judge (LaaJ) is a novel paradigm in which powerful language models are used to assess the quality, safety, or correctness of generated outputs. While this paradigm has significantly improved the scalability and efficiency of evaluation processes, it also introduces novel security risks and reliability concerns that remain largely unexplored. In particular, LLM-based judges can become both targets of adversarial manipulation and instruments through which attacks are conducted, potentially compromising the trustworthiness of evaluation pipelines. In this paper, we present the first Systematization of Knowledge (SoK) focusing on the security aspects of LLM-as-a-Judge systems. We perform a comprehensive literature review across major academic databases, analyzing 863 works and selecting 45 relevant studies published between 2020 and 2026. Based on this study, we propose a taxonomy that organizes recent research according to the role played by LLM-as-a-Judge in the security landscape, distinguishing between attacks targeting LaaJ systems, attacks performed through LaaJ, defenses leveraging LaaJ for security purposes, and applications where LaaJ is used as an evaluation strategy in security-related domains. We further provide a comparative analysis of existing approaches, highlighting current limitations, emerging threats, and open research challenges. Our findings reveal significant vulnerabilities in LLM-based evaluation frameworks, as well as promising directions for improving their robustness and reliability. Finally, we outline key research opportunities that can guide the development of more secure and trustworthy LLM-as-a-Judge systems. 

---
# Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism 

**Authors**: Tao Chen, Kun Zhang, Qiong Wu, Xiao Chen, Chao Chang, Xiaoshuai Sun, Yiyi Zhou, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.29252)  

**Abstract**: Long video understanding is a key challenge that plagues the advancement of \emph{Multimodal Large language Models} (MLLMs). In this paper, we study this problem from the perspective of visual memory mechanism, and proposed a novel and training-free approach, termed \emph{Flexible Memory} (\textbf{FlexMem}). In principle, FlexMem aims to mimic human behavior of video watching, \emph{i.e.}, continually watching video content and recalling the most relevant memory fragments to answer the question. In this way, FlexMem can help MLLMs achieve video understanding of infinite lengths, unlike previous methods that process all video information at once and have input upper-limit. Concretely, FlexMem first consider the visual KV caches as the memory sources, and realize the effective memory transfer and writing via a dual-pathway compression design. Afterwards, FlexMem also explores different memory reading strategies for the diverse video understanding tasks, including the popular streaming one. To validate FlexMem, we apply it to two popular video-MLLMs, and conduct extensive experiments on five long video and one streaming video task. The experimental results show that on \textbf{a single 3090 GPU}, our FlexMem can achieve obvious improvements than existing efficient video understanding methods and process more than \textbf{1k frames}, which also helps the base MLLMs achieve comparable or even better performance than SOTA MLLMs on some benchmarks, \emph{e.g.} , GPT-4o and Gemini-1.5 Pro. 

---
# Multi-Layered Memory Architectures for LLM Agents: An Experimental Evaluation of Long-Term Context Retention 

**Authors**: Sunil Tiwari, Payal Fofadiya  

**Link**: [PDF](https://arxiv.org/pdf/2603.29194)  

**Abstract**: Long-horizon dialogue systems suffer from semanticdrift and unstable memory retention across extended sessions. This paper presents a Multi-Layer Memory Framework that decomposes dialogue history into working, episodic, and semantic layers with adaptive retrieval gating and retention regularization. The architecture controls cross-session drift while maintaining bounded context growth and computational efficiency. Experiments on LOCOMO, LOCCO, and LoCoMo show improved performance, achieving 46.85 Success Rate, 0.618 overall F1 with 0.594 multi-hop F1, and 56.90% six-period retention while reducing false memory rate to 5.1% and context usage to 58.40%. Results confirm enhanced long-term retention and reasoning stability under constrained context budgets. 

---
# Self-Improving Code Generation via Semantic Entropy and Behavioral Consensus 

**Authors**: Huan Zhang, Wei Cheng, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29292)  

**Abstract**: Improving the code generation capabilities of large language models (LLMs) typically relies on supervised fine-tuning or preference optimization, both of which require costly external resources such as powerful teacher models or reliable test units. However, in real-world scenarios, it is much harder to obtain reference solutions and test oracles than problem descriptions and test inputs. In this paper, we tackle a challenging yet realistic question: Can a code language model improve itself without access to a superior teacher and a test oracle? To answer this, we propose ConSelf, a self-improving approach built upon two key ideas. First, we introduce code semantic entropy, a novel metric that measures problem-level uncertainty by assessing the functional diversity of program behaviors, enabling a curriculum construction with the most learnable problems. Second, we present consensus-driven direct preference optimization (Con-DPO), a preference-based fine-tuning method that weights each preference pair by its behavioral consensus, thereby mitigating the impact of noisy self-generated supervision. Experiments on various benchmarks and backbone LLMs demonstrate that ConSelf significantly outperforms baselines, validating the effectiveness of semantic entropy-based curriculum construction and consensus-driven optimization in improving code generation without external supervision. 

---
# Developing Adaptive Context Compression Techniques for Large Language Models (LLMs) in Long-Running Interactions 

**Authors**: Payal Fofadiya, Sunil Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2603.29193)  

**Abstract**: Large Language Models (LLMs) often experience performance degradation during long-running interactions due to increasing context length, memory saturation, and computational overhead. This paper presents an adaptive context compression framework that integrates importance-aware memory selection, coherence-sensitive filtering, and dynamic budget allocation to retain essential conversational information while controlling context growth. The approach is evaluated on LOCOMO, LOCCO, and LongBench benchmarks to assess answer quality, retrieval accuracy, coherence preservation, and efficiency. Experimental results demonstrate that the proposed method achieves consistent improvements in conversational stability and retrieval performance while reducing token usage and inference latency compared with existing memory and compression-based approaches. These findings indicate that adaptive context compression provides an effective balance between long-term memory preservation and computational efficiency in persistent LLM interactions 

---
# SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization 

**Authors**: Zhaorui Yang, Haichao Zhu, Qian Zhang, Rajiv Gupta, Ashish Kundu  

**Link**: [PDF](https://arxiv.org/pdf/2603.29109)  

**Abstract**: Fault localization identifies program locations responsible for observed failures. Existing techniques rank suspicious code using syntactic spectra--signals derived from execution structure such as statement coverage, control-flow divergence, or dependency reachability. These signals collapse for semantic bugs, where failing and passing executions follow identical code paths and differ only in whether semantic intent is satisfied. Recent LLM-based approaches introduce semantic reasoning but produce stochastic, unverifiable outputs that cannot be systematically cross-referenced across tests or distinguish root causes from cascading effects.
We present SemLoc, a fault localization framework based on structured semantic grounding. SemLoc converts free-form LLM reasoning into a closed intermediate representation that binds each inferred property to a typed program anchor, enabling runtime checking and attribution to program structure. It executes instrumented programs to construct a semantic violation spectrum--a constraint-by-test matrix--from which suspiciousness scores are derived analogously to coverage-based methods. A counterfactual verification step further prunes over-approximate constraints and isolates primary causal violations.
We evaluate SemLoc on SemFault-250, a corpus of 250 Python programs with single semantic faults. SemLoc outperforms five coverage-, reduction-, and LLM-based baselines, achieving Top-1 accuracy of 42.8% and Top-3 of 68%, while reducing inspection to 7.6% of executable lines. Counterfactual verification provides an additional 12% accuracy gain and identifies primary causal semantic constraints. 

---
# Multi-Agent LLMs for Adaptive Acquisition in Bayesian Optimization 

**Authors**: Andrea Carbonati, Mohammadsina Almasi, Hadis Anahideh  

**Link**: [PDF](https://arxiv.org/pdf/2603.28959)  

**Abstract**: The exploration-exploitation trade-off is central to sequential decision-making and black-box optimization, yet how Large Language Models (LLMs) reason about and manage this trade-off remains poorly understood. Unlike Bayesian Optimization, where exploration and exploitation are explicitly encoded through acquisition functions, LLM-based optimization relies on implicit, prompt-based reasoning over historical evaluations, making search behavior difficult to analyze or control. In this work, we present a metric-level study of LLM-mediated search policy learning, studying how LLMs construct and adapt exploration-exploitation strategies under multiple operational definitions of exploration, including informativeness, diversity, and representativeness. We show that single-agent LLM approaches, which jointly perform strategy selection and candidate generation within a single prompt, suffer from cognitive overload, leading to unstable search dynamics and premature convergence. To address this limitation, we propose a multi-agent framework that decomposes exploration-exploitation control into strategic policy mediation and tactical candidate generation. A strategy agent assigns interpretable weights to multiple search criteria, while a generation agent produces candidates conditioned on the resulting search policy defined as weights. This decomposition renders exploration-exploitation decisions explicit, observable, and adjustable. Empirical results across various continuous optimization benchmarks indicate that separating strategic control from candidate generation substantially improves the effectiveness of LLM-mediated search. 

---
# Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference 

**Authors**: Zifan He, Rui Ma, Yizhou Sun, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2603.29002)  

**Abstract**: Modern large language models (LLMs) increasingly depends on efficient long-context processing and generation mechanisms, including sparse attention, retrieval-augmented generation (RAG), and compressed contextual memory, to support complex reasoning. We show that these optimizations can be unified into a four-step memory processing pipeline: Prepare Memory, Compute Relevancy, Retrieval, and Apply to Inference. Through systematic profiling, we identify a 22%-97% memory processing overhead in LLM inference and strong heterogeneity in its computational characteristics. Motivated by this insight, we argue that \textbf{heterogeneous systems} are well-suited to accelerate memory processing and thus end-to-end inference. We demonstrate this approach on a GPU-FPGA system by offloading sparse, irregular, and memory-bounded operations to FPGAs while retaining compute-intensive operations on GPUs. Evaluated on an AMD MI210 GPU and an Alveo U55C FPGA, our system is $1.04\sim2.2\times$ faster and requires $1.11\sim4.7\times$ less energy across multiple LLM inference optimizations than the GPU baseline (similar results hold on NVIDIA A100). These results establish heterogeneous systems as a practical direction for efficient LLM memory processing and inform future heterogeneous hardware design. 

---
# Privacy Guard & Token Parsimony by Prompt and Context Handling and LLM Routing 

**Authors**: Alessio Langiu  

**Link**: [PDF](https://arxiv.org/pdf/2603.28972)  

**Abstract**: The large-scale adoption of Large Language Models (LLMs) forces a trade-off between operational cost (OpEx) and data privacy. Current routing frameworks reduce costs but ignore prompt sensitivity, exposing users and institutions to leakage risks towards third-party cloud providers. We formalise the "Inseparability Paradigm": advanced context management intrinsically coincides with privacy management. We propose a local "Privacy Guard" -- a holistic contextual observer powered by an on-premise Small Language Model (SLM) -- that performs abstractive summarisation and Automatic Prompt Optimisation (APO) to decompose prompts into focused sub-tasks, re-routing high-risk queries to Zero-Trust or NDA-covered models. This dual mechanism simultaneously eliminates sensitive inference vectors (Zero Leakage) and reduces cloud token payloads (OpEx Reduction). A LIFO-based context compacting mechanism further bounds working memory, limiting the emergent leakage surface. We validate the framework through a 2x2 benchmark (Lazy vs. Expert users; Personal vs. Institutional secrets) on a 1,000-sample dataset, achieving a 45% blended OpEx reduction, 100% redaction success on personal secrets, and -- via LLM-as-a-Judge evaluation -- an 85% preference rate for APO-compressed responses over raw baselines. Our results demonstrate that Token Parsimony and Zero Leakage are mathematically dual projections of the same contextual compression operator. 

---
# Theory of Mind and Self-Attributions of Mentality are Dissociable in LLMs 

**Authors**: Junsol Kim, Winnie Street, Roberta Rocca, Daine M. Korngiebel, Adam Waytz, James Evans, Geoff Keeling  

**Link**: [PDF](https://arxiv.org/pdf/2603.28925)  

**Abstract**: Safety fine-tuning in Large Language Models (LLMs) seeks to suppress potentially harmful forms of mind-attribution such as models asserting their own consciousness or claiming to experience emotions. We investigate whether suppressing mind-attribution tendencies degrades intimately related socio-cognitive abilities such as Theory of Mind (ToM). Through safety ablation and mechanistic analyses of representational similarity, we demonstrate that LLM attributions of mind to themselves and to technological artefacts are behaviorally and mechanistically dissociable from ToM capabilities. Nevertheless, safety fine-tuned models under-attribute mind to non-human animals relative to human baselines and are less likely to exhibit spiritual belief, suppressing widely shared perspectives regarding the distribution and nature of non-human minds. 

---
