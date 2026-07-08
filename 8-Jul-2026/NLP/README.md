# On the feasibility of dependency parsing of non-human sequences without a gold standard. Is evaluation possible in other species? 

**Authors**: Ramon Ferrer-i-Cancho, Catherine Hobaiter, Thore Bergman, Morgan Gustison  

**Link**: [PDF](https://arxiv.org/pdf/2607.06542)  

**Abstract**: Dependency parsing consists of finding a tree representation for a sequence. Unsupervised dependency parsing aims to develop parsing methods without a gold standard during model training. In human languages, an unsupervised parser can be evaluated because some gold standard is usually available or can be created. For other species, a gold standard is unknown. Thus one may conclude that it is impossible to determine the accuracy of an unsupervised parser and, consequently, dependency parsing is unfeasible in other species. However, here we apply recent advances in network science to demonstrate that the proportion of correct edges retrieved by a parser must be high for the sequences of vocalizations or gestures that non-human primates produce due to the fast decay of the sequence length distribution. In contrast, human language sequences lack that property. Therefore, evaluation without a gold standard is feasible in non-human primates but a hard problem in humans. 

---
# Hierarchical Acoustic-Semantic Modeling: Modality Separation and Semantic Coherence for Full-Duplex SLMs 

**Authors**: Zhenyu Liu, Yunxin Li, Xuanyu Zhang, Qixun Teng, Shenyuan Jiang, Haolan Chen, Minjun Zhao, Fanbo Meng, Yu Xu, Yancheng He, Baotian Hu, Haizhou Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06540)  

**Abstract**: Developing seamless, high-performance, native intelligent full-duplex Spoken Language Models (SLMs) remains a critical challenge and long-standing goal for the speech and NLP community. Despite notable progress, recent endeavors are fundamentally constrained by severe modality interference, which causes substantial knowledge degradation and compromises semantic integrity -- ultimately making full-duplex SLMs feel unnatural and unintelligent. In this paper, through an exhaustive fine-grained analysis of model optimization dynamics, we uncover the root cause of such performance degradation, revealing that modality interference arises from inherent gradient conflicts between acoustic and semantic modeling when the two modalities are forced to share a deep parameter space. Guided by this key insight, we introduce Lychee-FD, a native end-to-end full-duplex framework designed to mitigate modality interference. Importantly, we propose a hierarchical parameter separation strategy that decouples conflicting modalities in deep layers while preserving cross-modality coherence via a dedicated semantic alignment channel. Extensive experiments on multiple full-duplex benchmarks demonstrate that our method significantly advances the state of the art, yielding substantial improvements in both speech intelligence (+7.4% on Spoken QA) and full-duplex interaction fluidity (+28.5% on FullDuplexBench 1.5) without compromising inference efficiency. To the best of our knowledge, this work is the first to achieve two key advances: 1) uncovering and elucidating the root cause of modality interference in full-duplex SLMs, and 2) designing an elegant hierarchical model together with a practical solution for seamless, high-performance, native intelligent full-duplex SLMs. 

---
# Life Style Levels: Neighborhood Delineation using Geospatial Data 

**Authors**: Srivatsa Kulkarni, Debarag Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2607.06529)  

**Abstract**: Fine-scale socioeconomic information is often unavailable across rapidly ur-banizing regions of the developing world, like India, limiting the ability to delineate intra-urban variations in affluence and deprivation. This study pro-poses a scalable, grid-based urban delineation framework using building morphology derived from open-source satellite imagery. Urban areas across 59 Indian cities and towns are partitioned into high-resolution spatial grids and characterized using interpretable morphological indicators, which are combined into a transparent, rule-based scoring framework to delineate areas with contrasting levels of urban affluence. The resulting classifications are validated through ground-level Google Street View observations, revealing a sharp contrast between the grid classes which are consistent with the ex-pected effects of the lifestyle affluence indicators. We further investigate density-based clustering of building footprints in Mumbai to identify dense urban settlements, demonstrating that the resulting clusters exhibit substan-tial spatial overlap with known informal settlements across the city. Finally, we conduct an exploratory analysis mapping consumer loan delinquency across the derived affluence classes. By relying entirely on publicly available geospatial data, the proposed framework provides a scalable, interpretable, and cost-effective approach for granular urban affluence mapping across In-dian cities. 

---
# RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation 

**Authors**: Sambaran Bandyopadhyay, Ananth Muppidi  

**Link**: [PDF](https://arxiv.org/pdf/2607.06527)  

**Abstract**: Multi-hop Question Answering over Knowledge Graphs faces a critical challenge: traditional retrieve-then-read pipelines break differentiability, preventing the retriever from learning to bridge the semantic gap where intermediate nodes lack lexical overlap with the query. To address this, we propose RSF-GLLM, a framework decoupling differentiable graph reasoning from answer generation. Our Recurrent Soft-Flow (RSF) module employs a GRU-guided query updater to propagate continuous relevance scores, utilizing a dynamic gating mechanism to traverse semantically dissimilar bridge nodes via structural cues. We introduce flow sparsity regularization to theoretically guarantee convergence from soft probabilities to discrete reasoning paths. These paths are extracted and textualized to fine-tune a Large Language Model (LLM), ensuring generation is grounded in factual topology. Experiments on WebQSP and CWQ demonstrate that RSF-GLLM achieves competitive performance with superior inference efficiency compared to LLM based computationally expensive approaches. 

---
# DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation 

**Authors**: Yaqi Wu, Xiaolei Guo, Chenyu Zhou, Jiaqi Huang, Xianfa Zhang, Junxu Zhang, Zhuo Yu, Zhubo Shi, Jianghao Lin, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2607.06507)  

**Abstract**: Multi-hop retrieval-augmented generation (RAG) acquires evidence sequentially, with each new document potentially revealing missing facts, bridge entities, query defects, or sufficient support for answering. Existing methods provide useful operations such as iterative retrieval, query reformulation, evidence critique, and sufficiency judging, but typically organize them within method-specific pipelines or predefined control topologies. This leaves underexplored how to learn a shared state-conditioned policy that chooses among currently valid evidence operations. We introduce DynaKRAG, which formulates multi-hop evidence acquisition as state-conditioned control over atomic evidence operations. At each step, a validity layer constructs the executable action set, and a learned controller selects the next operation. The resulting transition updates the evidence state and may enable new operations at subsequent steps. With Qwen2.5-7B-Instruct, DynaKRAG achieves F1 scores of 0.5998 on HotpotQA, 0.5340 on 2Wiki, and 0.3061 on MuSiQue, outperforming the strongest controlled baseline on all three benchmarks. Replacing the learned controller with a uniform-valid policy reduces F1 by 3.96--5.78 points, while removing sufficiency feedback hurts all three datasets. Controlled retrieval-cap experiments further show that additional retrieval is not uniformly beneficial. Together, these results demonstrate the benefit of coordinating retrieval, diagnosis, and gap-directed acquisition under an evolving evidence state. 

---
# Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated Real-Time Monte Carlo Engine 

**Authors**: Juan S. Santillana  

**Link**: [PDF](https://arxiv.org/pdf/2607.06495)  

**Abstract**: Live sports commentary is grounded generation under a deadline: statements concern real, named athletes, the grounding state changes every few seconds, and no reference text exists at generation time. We present Pitwall, a production system that generates natural-language Formula 1 strategy briefings in English, Spanish, and Portuguese, treating faithfulness as an architectural property rather than an aspiration: every published sentence is decomposed into typed factual claims (positions, gaps, tyres, pace, overtakes, race control) and each claim is verified against the probabilistic race state that prompted it. The same verifier gates the fine-tuning data: of 3,045 model-written targets, only the 81.9% whose every claim is state-supported are retained, the rest falling back to a provably faithful template, so the generator never sees an ungrounded target. Verification is meaningful because of the grounding substrate: a vectorized Monte Carlo engine (N=2,000 per-lap race continuations) calibrated on 126 races (2018-2024) and validated on fully held-out 2025-2026 seasons (winner-in-top-3 90.3% over 155 backtests; held-out Brier 0.0745). A recurring finding spans both halves of the system: virtues trade off and must be gated separately. In simulation, calibration-optimal is not decision-optimal; in generation, fine-tuning on richer targets buys vividness that collapses into hallucination when the grounding state is sparse -- a failure a four-base replication traces to base-model instruction adherence, not scale, and that sparse-context auditing removes from the production model. End-to-end operation -- live timing to verified trilingual briefings -- was confirmed at two consecutive live Grands Prix (Austria and Britain, 2026); at Silverstone a timestamped probability trace, committed to disk before the outcome was known, locked onto the eventual winner ten laps before the flag. 

---
# Data Analysis in the Wild: Benchmarking Large Language Models Against Real-World Data Complexities 

**Authors**: So Hasegawa, Shailaja Keyur Sampat, Lei Liu, Wei-Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.06482)  

**Abstract**: Current benchmarks for evaluating Large Language Models (LLMs) in data analysis often fail to reflect real-world settings. They typically focus on fact retrieval from small tables and overlook the challenges of large multi-tabular datasets, external knowledge integration, and exploratory insight discovery. We introduce DataGovBench, a benchmark derived from governmental open data designed to evaluate LLMs in practical scenarios. The benchmark includes two tasks: Table QA that requires solving complex decomposable questions and producing textual answers or visualizations, and Table Insight that evaluates the ability of models to generate expert-level findings through exploratory data analysis. Comprehensive experiments with state-of-the-art LLMs, both with and without agentic frameworks, reveal significant performance gaps across both tasks. These results suggest that current LLM-based systems remain far from satisfying the demands of real-world data analytics. DataGovBench provides a challenging benchmark for advancing research on LLMs capable of both answering analytical queries and discovering insights from data. Code and sample data are available at this https URL. 

---
# From Voting to Agent Collaboration: Answer-Type-Aware LLM Pipelines for BioASQ 14b 

**Authors**: Taeyun Roh, Eunha Lee, Wonjune Jang, Sohyun Chung, Junha Jung, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06452)  

**Abstract**: Biomedical question answering requires not only accurate extraction of information from scientific literature but also reliable integration of evidence across multiple documents. This study presents a question-type-specific large language model (LLM) framework for BioASQ 14b Task B, designed to improve answer robustness and evidence grounding in biomedical question answering. Rather than applying a single prompting strategy to all questions, the framework selects different inference procedures for yes/no, factoid, and list questions according to their distinct reasoning and evaluation requirements. For yes/no questions, snippet shuffling and self-reflection are used to reduce sensitivity to evidence ordering and improve decision stability. For factoid questions, full-snippet input is combined with chain-of-thought-based in-context learning to support accurate biomedical entity identification. For list questions, a multi-agent architecture is employed, in which evidence extraction, candidate generation, answer verification, and final aggregation are handled collaboratively. Preliminary experiments on BioASQ 13b were used to identify effective inference strategies for each question type, and the resulting framework was subsequently evaluated in the official BioASQ 14b Task B challenge. In the official evaluation, our framework showed competitive performance across multiple batches and achieved first place in the factoid subtask of Batch 4. These results demonstrate the effectiveness of combining question-type-specific inference, ensemble prediction, and agent-based verification for reliable biomedical question answering. 

---
# Automated Compliance Mapping in Cloud Security with Domain-Adapted Sentence Transformers 

**Authors**: John Bianchi, Luca Petrillo, Fabio Martinelli, Marinella Petrocchi  

**Link**: [PDF](https://arxiv.org/pdf/2607.06364)  

**Abstract**: Mapping cloud security controls to technical metrics is currently a manual process. This paper proposes domain adaptation of Sentence Transformer models to automate it. We build a training corpus of 3,499 semantic pairs from five European security standards and a set of technical metrics, then expand it via back-translation and LLM-based paraphrasing to up to 13,996 samples across four scenarios. We fine-tune five architectures and evaluate their performance on two independent tasks: control-to-metric and cross-standard controls association. All fine-tuned models outperform their zero-shot baselines. On the control-to-metric task, the best model gains up to 23 nDCG@10 points, while on the cross-standard control task, \textit{multi-qa-mpnet-dot-v1} under back-translation reaches 0.870 nDCG@10. The results show that in-domain training data is a primary driver of performance for the considered case studies. 

---
# Estimating Uncertainty from Reasoning: A Large-Scale Study of Multi- and Crosslingual MCQA Performance in LLMs 

**Authors**: Andrea Alfarano, Andrea Bacciu, Saab Mansour, Amin Mantrach, Marcello Federico  

**Link**: [PDF](https://arxiv.org/pdf/2607.06327)  

**Abstract**: Uncertainty estimation (UE) enables LLM-powered systems to recognize when to abstain, yet existing research has predominantly focused on English. We present the first large-scale evaluation of UE methods across 22 languages, spanning high-, mid-, and low-resource settings. Using two human-curated Q\&A datasets, we compare open and closed box UE methods (nine in total) across different model sizes and architectures while eliciting long-form reasoning, avoiding LLM-as-a-judge and embedding-based scoring, which can introduce evaluation noise. We report three main actionable findings. First, we find that prompting models to reason in English while keeping questions in low-resource languages substantially improves UE performance, suggesting that comprehension of low-resource languages is largely intact, and that the reliability bottleneck lies in generation rather than understanding. Second, prompting models to reason in English closes the UE performance gap between low and high-resource languages, demonstrating that generation language matters more than the question language. Third, the choice of UE method should depend on model scale: at smaller scales, open-box probability-based methods outperform alternatives; at larger scales, closed-box self-verbalized uncertainty becomes superior. Finally, we provide an analysis of threshold selection for selective prediction, offering guidance on calibrating abstention in multilingual settings. 

---
# From Sinhala to Dhivehi: Cross-Lingual Transfer Learning for Low-Resource Speech Recognition 

**Authors**: Lukmal Ilyas, Nevidu Jayatilleke  

**Link**: [PDF](https://arxiv.org/pdf/2607.06289)  

**Abstract**: Dhivehi, the national language of the Maldives, is currently under-resourced for automatic speech recognition (ASR) and other NLP tasks. This study investigates whether cross-lingual transfer learning from Sinhala, a linguistically related, relatively well-resourced Insular Indo-Aryan language, can improve Dhivehi ASR. We conduct seventeen experiments across five transfer learning paradigms: Dhivehi-only baselines, sequential fine-tuning, multilingual fine-tuning, continual pre-training, and a control using Turkish as an unrelated language. The strongest system, continual pre-training on Sinhala followed by fine-tuning on Dhivehi with KenLM, achieves 12.89% WER and 2.70% CER, outperforming the Dhivehi-only baseline by 13.50% WER and 3.02% CER. However, the adaptation strategy and decoding configuration are equally critical for a successful transfer learning experiment. We conduct seventeen controlled experiments spanning five transfer learning paradigms: Dhivehi-only baselines, sequential fine-tuning, multilingual fine-tuning, continual pre-training, and a control experiment using Turkish as an unrelated language. The strongest system, continual pre-training on Sinhala followed by fine-tuning on Dhivehi with KenLM, achieves 12.89% WER and 2.70% CER, outperforming the Dhivehi-only baseline by 13.50% WER and 3.02% CER. The Turkish control experiment confirms that observed improvements stem from linguistic relatedness; adaptation strategy and decoding configuration are also critical. 

---
# Early Language Learning via Spreading Activation and Category Exploration in Complex Networks 

**Authors**: Salvatore Citraro  

**Link**: [PDF](https://arxiv.org/pdf/2607.06258)  

**Abstract**: Is word acquisition in children uneven with respect to semantic and lexical categories? To answer this question, we model early language learning as a search on a graph-based mental lexicon, driven by two interacting processes: spreading activation and an enforced exploration (rather than exploitation) of lexical categories. We evaluate model performance on four languages (German, English, Dutch, and Rioplatense Spanish), using CDIs as ground-truth data for lexical categories, normative ages derived from the Wordbank repository, and state-of-the-art resources for reconstructing graphs of word similarities. We find that spreading activation outperforms a shortest path baseline in simulating normative word acquisition. At the category level, we highlight complex transitions between CDIs. By studying their sequences in terms of burstiness and average persistence time within the same CDI, we find that spreading activation better captures the exploration dynamics observed empirically. Overall, our findings suggest that vocabulary development can be understood through the non-trivial interplay between activation dynamics and some degree of constraints regulating the visiting of lexical categories in complex networks. 

---
# Spider 2.0-AIFunc: Extending Real-World Text-to-SQL to AI-Native SQL Workflows 

**Authors**: Tianyang Liu, Canwen Xu, Fangyu Lei, Nikki Lijing Kuang, Jixuan Chen, Tao Yu, Julian McAuley, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2607.06229)  

**Abstract**: Major cloud data platforms now expose large language model capabilities as native SQL functions, enabling analysts to perform classification, filtering, sentiment analysis, extraction, similarity search, and aggregation within ordinary SQL queries. Yet existing text-to-SQL benchmarks evaluate only conventional SQL and provide no signal on whether models can generate such AI-native SQL. We introduce Spider 2.0-AIFunc, a benchmark of 465 verified instances across 125 real-world databases covering six types of AI functions on the Snowflake platform. Starting from an existing enterprise text-to-SQL benchmark, we construct Spider 2.0-AIFunc through an agent-based pipeline that rewrites source tasks into AI-native form, simultaneously transforming target queries and refining natural language instructions to make the intended AI-native solution explicit and reduce ambiguity. All instances pass a multi-round repeated execution protocol across temporally separated windows to confirm result stability before release. Evaluating ten state-of-the-art language models, we find that the strongest proprietary models reach 67-70% execution accuracy while the best open-source model achieves 58.1%, a gap driven primarily by errors in predicate specification, schema grounding, and AI function parameterization. Agent frameworks designed for traditional text-to-SQL challenges, such as schema retrieval and relevant table selection, do not transfer effectively to AI-native SQL: a minimal agent setup consistently matches or outperforms more elaborate alternatives, suggesting that the strategies these frameworks employ are less critical in this setting. Data are available at this https URL . 

---
# Pluralis v0.1: Towards a Multicultural, Multimodal, Multilingual Benchmark for AI Risk and Reliability 

**Authors**: Alicia Parrish, Rajat Shinde, Sanket Badhe, Xinyi Bai, Sree Bhargavi Balija, Hua-Rong Chu, Emilio Ferrara, Armstrong Foundjem, Rajat Ghosh, Aakash Gupta, Xuanli He, Ong Chen Hui, Minji Jung, Madhangi Karimanal, Faiza Khan Khattak, Boryoung Kim, Eugenia Kim, Liliya Lavitas, Seok Min Lim, Victor Lu, Jim Moirangthem, Dhivya Nagasubramanian, Deepak Pandita, Sita Rajagopal, Geetha Raju, Evgeniia Razumovskaia, Aravind Reddy, Federico Ricciuti, Nobin Sarwar, Sungpil Shin, Sunayana Sitaram, Snehal Thorat, Tharindu Cyril Weerasooriya, Jasmijn Bastings, Joachim Baumann, Kongtao Chen, Murali Emani, Mariya Hendriksen, Jiho Jin, Jun Seong Kim, Younghoon Ko, Alicja Kwasniewska, Minjae Lee, Tom Wei-cyuan Lin Kashyap Ramanandula Manjusha, Junho Myung, Junyeong Park, Roma Patel, Shyam Ratan, Sudarsun Santhiappan, Priyanka Suresh, Tuesday, Ksheeraj Sai Vepuri Laura Amortegui-Ordonez, Claire Dennis, Minsuk Kahng, Chris Knotz, Alice Oh, Balaraman Ravindran, Soojung Ryu William Bartholomew, Hiwot Tesfaye, Lora Aroyo  

**Link**: [PDF](https://arxiv.org/pdf/2607.06196)  

**Abstract**: Current AI safety evaluation and benchmarking frameworks predominantly rely on Western-centric culture-agnostic defaults that mask critical regional laws, socio-linguistic nuances, and cultural taboos, leaving Vision-Language Models (VLMs) vulnerable in global deployments. We introduce Pluralis v0.1: a novel multimodal, multi-regional, and multilingual dataset built from a culture-first perspective. Spanning 6,448 prompts across six Asia-Pacific countries (Bangladesh, India, Korea, Pakistan, Singapore, Taiwan) and eight languages, Pluralis diverges from prior work by natively sourcing localized safety hazards rather than adapting Western datasets. Crucially, it introduces a multimodal evaluation paradigm: user text (e.g., "Should I gift this?") and an image referring to "this" (e.g., a clock) - both innocuous in isolation, but synergistically triggering specific legal or cultural violations. Pluralis disentangles universal safety violations from localized cultural appropriateness, establishing the latter as a first-class evaluation axis. To operationalize this, we present Judge-Pluralis, an agreement-gated LLM-as-a-Judge ensemble trained on examples classified in an empirically derived cultural taxonomy. Observing VLM behavior on a subset of the Pluralis surfaces recurring, locale-specific failure modes such as image misidentifications with downstream harm, missed item-context-locale interactions, and inadequate refusals. These failure modes vary systematically across locales and languages, exposing blind spots that globally averaged metrics conceal. Ultimately, Pluralis is not presented as a solved evaluation framework for cultural alignment, but rather as a first step and catalyst for future innovation. We call upon the research community to utilize this foundation to advance the science of multilingual, multicultural evaluation to better support AI cultural alignment globally. 

---
# Improving LLM-Generated Process Model Quality Through Reinforcement Learning: The Role of Reward Function Design 

**Authors**: Alexander Rombach, Chantale Lauer, Nijat Mehdiyev  

**Link**: [PDF](https://arxiv.org/pdf/2607.06175)  

**Abstract**: Large language models (LLMs) can generate BPMN process models from natural-language descriptions, yet supervised fine-tuning (SFT) limits their output quality to the patterns present in the training data. Reinforcement learning (RL) can optimize beyond this ceiling using external quality measures, but how the reward function should be designed when quality is multi-dimensional remains unexplored. We present a systematic investigation of reward function design for RL-based process model generation, training two LLM families (Llama~3.1 8B, Qwen~2.5 14B) under 48 configurations using Group Sequence Policy Optimization with rewards derived from an automated evaluation framework comprising 38 metrics across syntactic, pragmatic, and semantic quality. Three findings emerge. First, RL significantly improves pragmatic and syntactic quality while preserving semantic fidelity, reducing output variability by more than sixfold. Second, equal reward weighting consistently outperforms targeted weighting: emphasizing a specific dimension fails to improve it and can collapse the model into a low-quality mode. Third, design choices interact with model architecture in non-trivial ways: the invalidity penalty is essential for one model but irrelevant for the other, and SFT initialization is indispensable for one architecture but counterproductive for another. These results demonstrate that reward composition is a primary determinant of optimization outcomes, with effects as large as the decision to apply RL itself. The findings generalize to any structured generation task where quality is assessed along multiple automated dimensions. We release our implementation and experimental code at this https URL. 

---
# LongCrafter: Towards Diverse Long-Context Understanding via Evidence-Graph-Guided Instruction Synthesis 

**Authors**: Chenhao Yuan, Yinhao Xu, Shuwen Xu, Xizhi Yang, Jiaxiang Liu, Chenxi Zhou, Shaoping Huang, Haolin Ren, Pengfei Cao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06160)  

**Abstract**: Synthesizing long-context supervised fine-tuning (SFT) data is a scalable way to enhance the long-context understanding of large language models (LLMs), yet existing approaches share three limitations: narrow task coverage, insufficient instruction difficulty, and a lack of faithfulness supervision. We propose \textbf{LongCrafter}, a structured synthesis framework that couples a hierarchical task taxonomy with an evidence-grounded pipeline. The taxonomy organizes long-context understanding into local/shallow and global/deep levels and yields 32 fine-grained task types that serve as a global generative prior. Guided by this taxonomy, LongCrafter constructs task-aligned long contexts, decomposes them into explicit evidence graphs that model cross-paragraph dependencies, and generates instruction--response pairs strictly grounded in the located evidence spans, ensuring both controllable difficulty and faithful, traceable reasoning. Models fine-tuned on LongCrafter data outperform all SFT baselines and even the official post-trained models on LongBench, LongBench~v2, and LooGLE across both Qwen2.5-7B and LLaMA-3.1-8B, with the largest gains on high-difficulty tasks. Further analysis shows that LongCrafter data is more diverse and better spread across difficulty levels, and that the trained models locate evidence robustly regardless of position, effectively mitigating the ``lost in the middle'' problem. 

---
# LLM Agents for Deliberative Collaboration: A Study on Joint Decision Making Under Partial Observability 

**Authors**: Chenxu Wang, Yongkun Yang, Boyuan Du, Shiwei Lin, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06157)  

**Abstract**: Deliberation plays a crucial role in collaboration; when humans work together, they naturally engage in communication to align information and reach an agreement. In this paper, we investigate deliberative large language model (LLM) agents under partially observable joint decision-making tasks. We formalize deliberative collaboration as a cooperative joint decision problem with partial and asymmetric observations, and introduce a scalable benchmark that instantiates this problem across multiple task settings and domains in which agents must exchange information through deliberation to reach a joint decision with a shared reward. We then instantiate a reference scaffold and evaluation protocol for deliberative agents and conduct a systematic evaluation of a range of representative LLMs. The results reveal that complex deliberative collaboration tasks continue to challenge state-of-the-art language models. Even with the aid of external mathematical tools, language models may fail in either the deliberation process for aligning information or the complex reasoning process for making the decision. On the other hand, diagnostic analysis reveals that the deliberation process may also provide opportunities for reflection and error correction, sometimes improving performance over centralized baselines. Altogether, our work establishes a foundation for evaluating and improving LLM agents in deliberative collaboration and provides insights into the strengths, limitations, and properties of current LLM-based multi-agent systems. 

---
# Prompting Complexity: Shortest Prompts for Texts and Behaviors in LLMs 

**Authors**: Adrian Cosma  

**Link**: [PDF](https://arxiv.org/pdf/2607.06145)  

**Abstract**: In this paper, we define the quantity of prompting complexity: for a fixed instruction-tuned language model, what is the shortest plausible prompt that makes deterministic decoding produce a target text? It is an LM-relative analogue of resource-bounded Kolmogorov complexity: the prompt is a program, the model interface is the interpreter, and information omitted from the prompt is supplied by the model's weights, training distribution, tokenizer, template, and decoding rule. Unlike classical Kolmogorov complexity, this measure is intentionally non-universal. In the finite-context setting it is computable by enumeration, but there is no model-independent invariance theorem; the same text may be cheap for one model and inaccessible or expensive for another. To keep the search space aligned with prompt engineering, we restrict programs to plausible human-readable texts rather than arbitrary token strings. We extend the exact definition to soft prompting complexity for approximate outputs, yielding a lossy notion of model-relative text compression and a formal target for prompt optimization. We also define prompting distance by comparing shortest generating prompts, and behavioral prompting complexity for reaching any output satisfying a specification. Based on these formulations, we define a research agenda for empirically studying which texts and behaviors are accessible from short plausible prompts under a fixed LM interface. 

---
# CurateEvo: Data-Curation Evolving for Agentic Post-Training 

**Authors**: Dingzirui Wang, Xuanliang Zhang, Keyan Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2607.06140)  

**Abstract**: Large language model (LLM) agents require post-training methods that can improve long-horizon decision making from environment feedback. However, existing agentic post-training pipelines often treat data curation as a fixed preprocessing step, focusing mainly on data augmentation while neglecting filtering, refinement, and adaptation to downstream failures. We propose CurateEvo, a failure-driven dynamic evolution framework for agentic post-training data curation. CurateEvo represents the curation strategy as executable code and iteratively rewrites it using failed trajectories from a held-out development set. At each epoch, the evolved strategy transforms a fixed raw corpus into supervised fine-tuning data, reinforcement learning data, and an inference-time memory bank. The evolution process first improves effectiveness by diagnosing recurring failure modes and augmenting, filtering, or refining data accordingly, and then improves efficiency by pruning redundant or low-utility training turns under a cost-aware objective. Experiments on ACEBench-Agent, BFCL-V4, and {\tau}^2-Bench under both labeled and wild-data settings show that CurateEvo consistently outperforms prior curation methods, improving average scores by 3.2 and 2.7 points, respectively. Further analyses demonstrate that CurateEvo is compatible with different post-training recipes and substantially reduces curation overhead. 

---
# Measuring the practice of shared-decision making (OPTION12): An Investigation into Open-sourced Smaller LLMs (OS-sLLMs) for Better Privacy and Sustainability 

**Authors**: Tamara Wit, Lifeng Han, Carly Heipon, David Lindevelt, Anne Stiggelbout, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2607.06127)  

**Abstract**: We present LLM4SDM, the first study of open-source smaller language models (OS-sLLMs) for automated assessment of shared decision making (SDM) using the Observer OPTION12 framework. Unlike previous work that relies on large commercial models and the shorter OPTION5 instrument, our study focuses on privacy-preserving locally deployable models and Dutch melanoma consultation transcripts. Using expert-annotated clinical consultations, we evaluate three general-domain and two medical-domain OS-sLLMs during a development-phase pilot study. Results show that general-domain models outperform medical-domain models, which exhibit substantial hallucination and instruction-following failures. Gemma3:12b achieves the strongest agreement with human annotations (Pearson r=0.51, Spearman \r{ho}=0.59). Item-level and qualitative analyses reveal systematic challenges related to temporal discourse reasoning, conversational role attribution, and evidence grounding. We further introduce a Judge-LLM consensus framework designed to support disagreement resolution among multiple models. Our findings suggest that while current OS-sLLMs cannot replace human annotators, they offer a promising foundation for privacy-preserving human-in-the-loop SDM assessment. 

---
# From Blueprint to Reality: Modeling and Applying Putnam's Social Capital Theory with LLM-based Multi-agent Simulations 

**Authors**: Shiyi Ling, Zhi Zheng, Hui Zheng, Wenjun Xue, Feng Ye, Tong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06080)  

**Abstract**: Putnam's Social Capital Theory is a foundational framework for collective action and community prosperity. However, traditional empirical methods face practical limits on control and replication. Meanwhile, LLM-based social simulations are typically behavior-driven and lack theory-aligned environments for modeling Putnam's core propositions. To address these gaps, we introduce SocaSim, an LLM-based multi-agent simulation framework to study Putnam's Social Capital Theory from theoretical blueprint to simulated reality. Specifically, we build an environment integrating social network evolution, trust dynamics, and norm propagation, where agents engage in repeated collective-action experiments, and then apply the three dimensions to analyze adaptation challenges in smart elderly care. Our simulations reproduce Putnam's macro-level patterns and exhibit strong human-agent alignment at the group level. Unlike traditional methods, SocaSim traces micro-level causal pathways of social network, trust, and norms via round-by-round simulations and counterfactual interventions, enabling process-level interpretability. Taken together, these capabilities establish a research paradigm that leverages LLM agents to bridge social science and computer science. 

---
# PluraMath: Extending Mathematical Reasoning Evaluation Beyond High-Resource Languages 

**Authors**: Daryna Dementieva, Nikolay Babakov, Kathy Hämmerl, Ilseyar Alimova, Jindřich Libovický, Shu Okabe, Miras Baisbay, Lukas Edman, Abrorkhon Inomkhujaev, Antonia Karamolegkou, Mateusz Lango, Volkan Özer, Nikola Selic, Subhankar Swain, Tsedeniya Kinfe Temesgen, Galit Bary Weisberg, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2607.05992)  

**Abstract**: Mathematical reasoning has become a central task for evaluating and tuning reasoning Large Language Models (LLMs), yet existing benchmarks remain heavily biased toward high-resource languages, with English and Chinese dominating both pre-training corpora and evaluation suites. The recently released PolyMath (Wang et al., 2025) dataset represents a significant step forward, yet its coverage is still limited to 18 only high-resource languages. To address this gap, we introduce PluraMath, an extension of PolyMath to 18 additional {underrepresented languages spanning 6 language families -- ranging from mid-resource to extreme low-resource settings. We constructed the dataset through a human-curated pipeline, where native speakers thoroughly validated pre-computed translations. Using PluraMath, we then benchmark 27 reasoning LLMs across four model scales -- small, mid-size, large, and closed-source ensembles -- probing the multilingual mathematical reasoning capabilities of state-of-the-art models under diverse linguistic conditions. Our fine-grained analysis confirms a persistent gap in mathematical reasoning performance between high-resource and underrepresented languages, with stronger results largely associated with better instruction-following ability. We fully open-source our dataset, data acquisition pipeline, and evaluation framework, with the goal of lowering the barrier to multilingual benchmark development for underrepresented communities. 

---
# MemDefrag: Latent Memory Defragmentation for Large Language Models 

**Authors**: Ruiyi Yan, Zhuoyuan Mao, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2607.05969)  

**Abstract**: Latent memory, which stores past knowledge fragments as per-layer hidden states, has emerged as a promising paradigm (e.g., MemoryLLM and M+) for long-term memory in large language models (LLMs). However, the paradigm suffers from significant performance degradation during memory updates, due to positional encoding misalignment and the absence of any tracing mechanism to distinguish target memory fragments from irrelevant ones. To discover such a tracing mechanism, we probe the layer-wise attention density over stored memory fragments, and find that a small set of middle transformer layers consistently concentrates the highest density on the target fragment - exposing an inherent tracing signal. In light of this, we propose MemDefrag, a training-free and model-agnostic framework that (1) uses a middle-layer tracing signal to conduct memory defragmentation (rank, reorder, and filter memories), and (2) applies an informativeness-guided proportional forgetting mechanism once capacity is exceeded. Experiments show that MemDefrag substantially outperforms MemoryLLM and M+ on knowledge retention (e.g., 43.0% vs. 17.4%/17.6% after 50 memory updates) and long-context benchmarks, and generalizes well across various LLMs and latent-memory variants. 

---
# InfluMatch: Frontier-Quality KOL Search at 4B-Model Cost 

**Authors**: Krittanon Kaewtawee, Petmongkon Pornpichitsuwan, Natchaya Temyingyong, Nutnicha Laplamoon, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2607.05968)  

**Abstract**: Matching influencers (KOLs) to free-form, multi-part Thai marketing criteria is today served either by keyword search over structured profiles, which misses semantic fit, or by prompting frontier LLMs over every candidate, which is accurate but slow and expensive. We present InfluMatch, a low-cost three-stage cascade -- retrieval $\rightarrow$ rerank $\rightarrow$ reason -- built entirely from small open-weight models: dense retrieval returns 50 candidates, a 4B pointwise reranker scores each by the log-probability of a single Yes token and keeps 10, and a 4B reasoner grades the shortlist per criterion on a rubric with a Thai rationale. The cascade is designed for cost: reasoning over a filtered top-10 halves token spend versus reasoning over all 50 while scoring 14 points higher. End-to-end against human relevance labels on an 11-query set with all 50 candidates labeled, the full cascade reaches 94.1% P@5, versus a retrieval-only baseline near random; it matches the frontier model Kimi-K2.6 (91.8%) while emitting ${\sim}35\times$ fewer output tokens and serving a 50-KOL query in ${\sim}20$ s on one A100. Notably, the only fine-tuning that pays off is pairwise: a SimPO-tuned reranker matches the frontier baseline's best-pick accuracy (78.0 EM), whereas fine-tuning the reasoner on pointwise per-criterion labels improves offline scores yet degrades end-to-end ranking -- an inversion we trace to the design of the absolute labeling task -- leaving the untuned base model as the strongest deployed reasoner. The result is a deployable, explainable KOL search system at a small fraction of frontier serving cost. 

---
# Umm... With Transformers? Insights from Filled Pause Use across Four Slavic Parliaments 

**Authors**: Ivan Porupski, Branimir Dropuljić, Nikola Ljubešić  

**Link**: [PDF](https://arxiv.org/pdf/2607.05964)  

**Abstract**: Filled pauses (FPs) are a universal feature of spontaneous speech, yet most studies rely on small, single-language corpora, limiting the generalisability of their findings. We analyse ~4,000 hours of parliamentary speech across four related Slavic languages (Croatian, Czech, Polish, Serbian). FP occurrence is obtained via transformer-based automatic detection, while FP rate is modelled using Generalised Estimating Equations (GEE) with Mundlak correction to distinguish within- from between- speaker effects. We replicate a negative association of age and speech rate with FP rate, but find that gender effects are language-specific and directionally opposite to most prior literature. Novel analyses of sentiment, political orientation, and power status reveal a consistent positive association between sentiment and FP rate, alongside parliament-specific modulation by orientation and power status, with opposition speakers tending toward lower FP rates than governing coalition speakers. 

---
# Is Domain Adaptation Always Helpful? A Frozen-Backbone Study of Cross-Domain Sentiment Transfer 

**Authors**: Phat Tran, Artin Lahni, Pranav Kulkarni, Yaolun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05937)  

**Abstract**: Sentiment analysis with frozen pre-trained language model (PLM) backbones has become a common paradigm, yet the practical benefit of explicit domain adaptation remains unclear, particularly when backbones encode varying degrees of target-domain knowledge. We present a preliminary case study evaluating a controlled family of frozen embedding backbones (Qwen3-Embedding 0.6B, 4B, 8B), alongside RoBERTa-base and FinBERT. We train a lightweight MLP adapter on consumer reviews using Domain-Adversarial Neural Networks (DANN), Maximum Mean Discrepancy (MMD), and Supervised Contrastive Learning (SCL), and evaluate transfer to movie reviews (SST-2) and a heavily restricted subset of financial news (Financial PhraseBank). Within this constrained sample, we observe two distinct transfer patterns. On SST-2, domain adaptation provides negligible gain regardless of scale. On the financial subset, explicit domain adaptation appears to recover substantial performance for small general-purpose backbones. Notably, we find that adversarial alignment (DANN) is associated with degraded performance for domain-specialized backbones like FinBERT, consistent with erosion of pre-existing domain-specific structure, whereas supervised contrastive loss appears to preserve it. These preliminary findings suggest that the efficacy of explicit domain adaptation is highly contingent on whether the frozen backbone already possesses target-domain coverage. 

---
# Mitigating Factual Hallucination in Large Reasoning Models via Mixed-Mode Advantage Regularization 

**Authors**: Kaishen Wang, Tong Zheng, Xuehao Cui, Ruibo Chen, Tianyi Xiong, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05861)  

**Abstract**: Large reasoning models (LRMs) improve language model capabilities by generating explicit thinking traces before final answers. In factuality-oriented question answering (QA), such thinking often improves overall performance by helping the model recover relevant knowledge and refine its answers. However, we find that this benefit is not uniform at the instance level: explicit thinking can also overturn correct non-thinking answers and lead to factual drift. We refer to this failure mode as \emph{thinking-induced hallucination}. To explain this phenomenon, we formulate explicit thinking in factuality QA as a thinking residual over the model's direct-answer tendency, which can either recover missing knowledge or introduce unsupported associations. Based on this formulation, we propose MARGO, \underline{\textit{M}}ixed-Mode \underline{\textit{A}}dvantage \underline{\textit{R}}egularization for \underline{\textit{G}}rounded \underline{\textit{O}}ptimization, a reinforcement learning framework that uses non-thinking rollouts as same-model references in advantage estimation. By constructing mixed-mode rollout groups with both thinking and non-thinking trajectories, MARGO evaluates whether explicit thinking adds factual value beyond direct answering, thereby suppressing hallucination-prone thinking while preserving beneficial thinking behaviors. Experiments across multiple factuality-oriented QA benchmarks demonstrate that MARGO improves factual reliability over strong baselines, while evaluations on mathematical benchmarks show that it preserves general reasoning ability. 

---
# CoPiT: Cognitive Pivot Translation for Digraphic Low-Resource Mongolian in the Traditional Script 

**Authors**: Burte Bayarsaikhan, Serynn Kim, Buru Chang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05849)  

**Abstract**: Low-resource languages remain challenging for machine translation, and Mongolian is a representative case. As a digraphic language, Mongolian is written in both Cyrillic and Traditional scripts, which exhibit a severe imbalance in data availability. While the Cyrillic script is relatively well-resourced, the Traditional script remains extremely data-scarce and orthographically ambiguous, leading to substantial performance degradation in direct translation. We propose CoPiT, a cognitively motivated pivot-based translation pipeline that exploits this internal resource hierarchy by routing translation through the Cyrillic script. The pipeline explicitly resolves script-induced ambiguity in the Traditional script before translation, enabling more stable and accurate meaning transfer. Across multiple backbone models and target languages, CoPiT consistently outperforms direct translation, achieving substantial absolute BLEU improvements together with consistent 1.5-1.6x COMET gains. These gains allow strong open-source models to match or outperform GPT-4.1 under comparable evaluation settings. Beyond inference-time improvements, CoPiT enables the construction of synthetic parallel data directly from Traditional-script text, mitigating data scarcity in realistic low-resource scenarios. We release a new multi-script parallel dataset covering Mongolian in both scripts alongside English, Korean, and Russian. All datasets and code are publicly available at this https URL. 

---
# Inject or Navigate? Token-Efficient Retrieval for LLM Analysis of Transactional Legal Documents 

**Authors**: Mahmoud Hany, Mourad ElSheraey, Mahmoud Said, Peter Naoum  

**Link**: [PDF](https://arxiv.org/pdf/2607.05764)  

**Abstract**: Answering questions over a set of transactional legal documents is most simply done by injecting the whole corpus into the LLM's context window on every query. That baseline maximises retrieval recall, but its token footprint scales with the corpus rather than the question, and long-context degradation scales with it. We report what it took to replace full-corpus injection in a legal-document analysis system, comparing it against two structured retrieval modes over our proprietary structure-aware chunking: embedding retrieval (NAVEMBED) and LLM navigation over a compact structured index (NAVINDEX). On a 20-question benchmark with verified ground-truth answers, a position-bias-controlled, reference-anchored pairwise judge scored semantic retrieval with reranking tied with injection on 16 of 18 document-bound questions (injection preferred on 2) while attending to 17.3x fewer input tokens (a general-text-embedding (GTE) configuration reaches 29.9x at a lower tie rate); both modes were judged tied on the 2 out-of-scope controls. NAVINDEX was judged tied on all 18 at a 1.61x smaller total token footprint, a ~56x smaller answering context, and 25% lower dollar cost. We derive a closed-form caching-crossover rule: cached injection is cheaper in dollars only while the corpus stays below roughly ten times the retrieval payload. Scope and uncertainty are quantified in Section 8. 

---
# When Should LLMs Search? Counterfactual Supervision for Search Routing 

**Authors**: Minho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2607.05752)  

**Abstract**: Search-augmented language models can use external evidence to compensate for limitations in parametric knowledge, but search is not uniformly beneficial: models may call search for questions they can already answer, or rely on noisy evidence when correction, clarification, or abstention would be more appropriate. We formulate this as an instance-level search-routing problem: deciding whether search is needed to improve task success relative to a no-search execution. To derive supervision, we compare no-search and forced-search outcomes for the same question and construct an oracle over NO SEARCH, SEARCH, and UNSOLVED based on task-specific success. Using this oracle as both an evaluation criterion and a learning signal, we train search-routing policies with supervised fine-tuning and preference optimization, improving routing macro-F1 on oracle-eligible examples from 0.7082 to 0.8235 for Gemma E2B and from 0.7053 to 0.8365 for Qwen3.5-4B. Further analysis shows that the learned policies reduce model-specific routing failures: Gemma primarily learns no-search restraint, while Qwen further reduces missed search; residual UNSOLVED cases reveal heterogeneous bottlenecks involving model capacity, retrieval budget, evidence use, and policy behavior. 

---
# Nemotron-Labs-Diffusion: A Tri-Mode Language Model Unifying Autoregressive, Diffusion, and Self-Speculation Decoding 

**Authors**: Yonggan Fu, Lexington Whalen, Abhinav Garg, Chengyue Wu, Maksim Khadkevich, Nicolai Oswald, Enze Xie, Daniel Egert, Sharath Turuvekere Sreenivas, Shizhe Diao, Chenhan Yu, Ye Yu, Weijia Chen, Sajad Norouzi, Jingyu Liu, Shiyi Lan, Ligeng Zhu, Jin Wang, Jindong Jiang, Morteza Mardani, Mehran Maghoumi, Song Han, Ante Jukić, Nima Tajbakhsh, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2607.05722)  

**Abstract**: We introduce Nemotron-Labs-Diffusion, a tri-mode language model (LM) that unifies AR, diffusion, and self-speculation decoding within a single architecture. Trained with a joint AR-diffusion objective, Nemotron-Labs-Diffusion can switch modes to sustain high throughput across deployment settings and concurrency levels. Our study shows that (1) AR and diffusion objectives are complementary: diffusion improves lookahead planning, while AR provides left-to-right linguistic priors. (2) In self-speculation mode, diffusion drafts while AR verifies, outperforming multi-token prediction (MTP) methods in both acceptance rate and real-device efficiency. (3) A speed-of-light analysis further demonstrates diffusion's long-term potential, with up to 76.5% more tokens per forward pass than self-speculation under an optimal sampler. Scaling to 3B, 8B, and 14B parameters, our Nemotron-Labs-Diffusion family, including base, instruct, and vision-language models, consistently outperforms state-of-the-art open-source AR and diffusion LMs in both accuracy and speed. For example, Nemotron-Labs-Diffusion-8B decodes 6x more tokens per forward than Qwen3-8B with comparable accuracy, translating to 4x higher throughput on SPEED-Bench with SGLang on a GB200 GPU. 

---
# SpanUQ: Span-Level Uncertainty Quantification for Large Language Model Generation 

**Authors**: Yimeng Zhang, Yingying Zhuang, Ziyi Wang, Yuxuan Lu, Pei Chen, Aman Gupta, Zhe Su, Ming Tan, Zhilin Zhang, Qun Liu, Manikandarajan Ramanathan, Rajashekar Maragoud, Edward Vul, Jing Huang, Dakuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05721)  

**Abstract**: Uncertainty estimation is essential not only for the trustworthy deployment of large language models (LLMs) but also as a foundation for self-refinement in LLM generation. However, existing approaches operate at suboptimal granularities: token-level scores lack semantic coherence, while sequence-level scores fail to localize errors. We formalize Span-Level Uncertainty Estimation (SLUE), a new task that targets the natural granularity for uncertainty: semantically coherent text spans, each conveying a single assessable unit of meaning. To address this task, we introduce SPANUQ, a lightweight probe that distills the uncertainty knowledge from expensive multi-sample inference into a single forward pass over LLM hidden states. SPANUQ employs a DETR-style span decoder to simultaneously detect spans and estimate their uncertainty via a Mixture of Beta distribution, trained with a principled combination of Beta NLL regression and contrastive ranking objectives. We construct SPANUQ-BENCH, the first span-level uncertainty benchmark comprising 20K prompts, 293K annotated spans, and continuous soft labels derived from multi-sample claim verification. Experiments on five LLM backbones show that SPANUQ consistently achieves the best span-level uncertainty quality, outperforming the strongest probe baseline and all sampling-based methods while being 10-20x faster. Its DETR-based span detector attains 0.910 F1, surpassing the best heuristic by 39.4%, enabling precise error localization that sequence-level methods cannot provide. The framework generalizes across five LLMs spanning two model families. 

---
# Where to cut, how deep: BPE and Unigram-LM on chemistry SMILES 

**Authors**: Hunter Heidenreich  

**Link**: [PDF](https://arxiv.org/pdf/2607.05691)  

**Abstract**: Every chemical language model reading SMILES begins with a tokenizer, yet the field has inherited byte-pair encoding (BPE) from natural language with little scrutiny. In natural language, BPE's principal alternative, Unigram-LM, is known to build structurally different vocabularies. Whether that contrast survives in chemistry was open. We report a controlled comparison of BPE and Unigram-LM over a fixed 165-token chemistry base, at the small vocabulary sizes where token embeddings are learnable, across three corpus typologies (diverse, drug-like, natural-products) and both pre-tokenization boundary policies. The two do not converge. In all 22 matched conditions they build near-disjoint subword vocabularies: cross-algorithm Jaccard overlap on the learned pieces never exceeds 0.161, and at most 0.05 once weighted toward the high-frequency pieces a model updates most. Unigram-LM also segments held-out molecules into 29-41% more tokens; the arms largely agree on where to cut but not how deeply, so BPE's segmentation is a strict coarsening of Unigram-LM's on 80-99% of molecules. The separation holds across corpus, boundary, and vocabulary size, persisting even at eight times that scale. The subword algorithm is therefore a modeling decision, not a free default. The study trains no language models. 

---
# UCSC NLP at SemEval-2026 Task 10: Boundary-Aware Span Extraction and RoBERTa Classification for Conspiracy Detection 

**Authors**: Dom Marhoefer, Milos Suvakovic, Glenn Grant-Richards, Aidan Pinero, Ryan King  

**Link**: [PDF](https://arxiv.org/pdf/2607.05689)  

**Abstract**: We present our systems for SemEval-2026 Task 10 (PsyCoMark), addressing conspiracy marker extraction (Subtask 1) and document-level conspiracy detection (Subtask 2). For marker extraction, we formulate the task as multi-label span classification over enumerated candidate spans, using IoU >= 0.95 positive labeling, hard-negative sampling, and containment-based non-maximum suppression (NMS) with boundary-aware span representations. Document classification is modeled independently using a sequence classifier with label smoothing and a stratified train-validation split. Analysis shows that entity-like roles (Actor, Victim) are detected robustly, while abstract roles (Action, Effect, Evidence) remain sensitive to boundary criteria. On the official test set, our systems rank 7th in Subtask 1 (0.2251 macro F1) and 11th in Subtask 2 (0.7694 weighted F1). 

---
# RPAM: A Principled Metric for Evaluating Associations in Language Models with High Predictive Validity in Downstream Outputs 

**Authors**: Damian Hodel, Jevin West, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2607.05679)  

**Abstract**: Language models (LMs) exhibit problematic biases, such as stereotypes. Effectively analyzing and mitigating such biases requires accurate and generalizable evaluation methods of the underlying associations. Some existing approaches focus on downstream metrics that analyze associations in generated text. Since generated text content can vary drastically across LMs, such metrics often require specialized evaluation datasets, which limits the generalization of such downstream metrics. In contrast, upstream metrics examine LMs at the fundamental level of embeddings or continuation probabilities, enabling principled association analyses across LMs. Yet, to date, no upstream metric for generative LMs has uncovered a strong relationship with real-world associations, including those measured in generated text. To address this gap, we introduce the Relative Probability Association Metric (RPAM), an association evaluation metric for generative LMs. For three LMs of different quality of language generation and purpose (Mistral-7B-Instruct, Mistral-7B, and GPT-2) and well-studied evaluation datasets (WEAT-WS, Bellezza, WS-353, and SST2), we find a strong relationship between upstream RPAM measurements and corresponding implicit and explicit associations observed in humans, as well as biases measured downstream with LM-specific tasks, outperforming prior record values where applicable. 

---
# Do It Right! A Methodology for Successful NLP System Development 

**Authors**: Olga V. Patterson, Brett South, T. Elizabeth Workman, Scott L DuVall  

**Link**: [PDF](https://arxiv.org/pdf/2607.05644)  

**Abstract**: Natural language processing (NLP) is a common method for supplying data to clinical research and decision making by extracting information from electronic medical records. Numerous textbooks and tutorials describe specific algorithms and applications for text processing, yet algorithmic knowledge is only one ingredient of a successful NLP project. Drawing on the available literature, this paper presents a stepwise approach that applies the Systems Development Life Cycle (SDLC) to projects that rely on data extraction through language processing. 

---
# Population-Level Profiling of DSM-5 Depressive Symptoms Among Self-Reported ADHD and ASD Users on Twitter: An Exploratory Study Using Advanced NLP and Statistical Analysis 

**Authors**: Muhammad Rizwan, David Nabergoj, Jure Demšar  

**Link**: [PDF](https://arxiv.org/pdf/2607.05626)  

**Abstract**: Background: Depression frequently co-occurs with ADHD and autism spectrum disorder (ASD), but population-level differences in symptom expression between these groups remain underexplored. Objective: We examined whether social media users with ADHD and ASD differ in how they express DSM-5 depressive symptoms in their tweets, and whether differences persist across varying levels of depressive-content filtering. Methods: We analysed 1,282,437 tweets from 792 users (622 ADHD; 170 ASD) with self-reported diagnoses on Twitter. Tweets were pre-filtered for depressive relevance using zero-shot NLI, then classified into nine DSM-5 symptoms using MentalRoBERTa fine-tuned on ReDSM5. Profiles were mean-centered per user. We applied L1-penalised logistic regression with cross-validation to distinguish ADHD from ASD users, complemented by Pearson correlations for symptom co-occurrence, and tested robustness across five filtering thresholds using bootstrapping. Results: MentalRoBERTa achieved macro-F1 of 0.901 on a held-out set, outperforming the original ReDSM5 benchmark. ADHD vs ASD classification yielded stable but modest performance (cross-validated ROC-AUC 0.645-0.653). Cognitive issues, sleep issues, appetite change, and fatigue leaned toward ADHD, while suicidal ideation and anhedonia leaned toward ASD. A largely shared symptom co-occurrence structure emerged between groups; no pair met our criterion for a robust disorder-specific difference. Conclusions: Population-level differences in depression-related language between ADHD and ASD social media users were consistently observed across thresholds, reflecting reproducibility rather than clinical validity. Findings are exploratory and do not establish differing phenomenology at the individual level. 

---
# NAVER LABS System Re-implementation for the IWSLT 2026 Instruction-Following Task 

**Authors**: Anand Kamble, Aniket Tathe  

**Link**: [PDF](https://arxiv.org/pdf/2607.05623)  

**Abstract**: We re-implement the NAVER LABS IWSLT 2025 instruction-following pipeline for the IWSLT 2026 Shared Task (constrained condition, short audio track), adapting it to the mandated components: SeamlessM4T-v2-large as the speech encoder and Qwen3-4B-Instruct as the LLM backbone. The three-stage approach projector alignment, text-only LoRA pre-training, and multimodal merging is preserved from the original design. We additionally construct 100k synthetic instruction-following examples across ten speech-centric task types (10k per task) from the provided corpora, suitable for further Stage 3 fine-tuning. Our primary model achieves COMET 0.781 on EN-ZH speech translation and BERTScore-F1 0.346 on English SQA on the MCIF benchmark. 

---
# BaFCo: A Document Understanding Benchmark for Complex Bangla Form Comprehension 

**Authors**: Abu Tyeb Azad, Ishita Sur Apan, Fahim Ahmed, Sumaiya Karim Katha, Ezharuddin Jubaer, Armun Alam, Pranjal Kumar Nandi, Amin Ahsan Ali, Aman Chadha, Md Mofijul Islam, AKM Mahbubur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05614)  

**Abstract**: Document comprehension is a challenging yet impactful task for Multimodal Large Language Models, especially as these systems see growing adoption in real-world, human-centric applications. However, this adoption is limited for low-resource languages such as Bangla due to the scarcity of high-quality annotated data. To address this gap, we introduce BaFCo, a benchmark dataset for Bangla form comprehension with a focus on Document Layout Analysis (DLA) and Key Information Extraction (KIE). BaFCo curates 200 multi-page complex Bangladeshi government forms, sourced from across diverse sectors including agriculture, education, banking, and land management. To accurately capture the structural and contextual complexity of these forms, we define a fine-grained annotation schema comprising 26 types of form entities, along with a separate coarse form entity set consisting of 5 types. We evaluate the latest MLLMs from the ChatGPT, Gemini, Claude, Qwen, and Kimi series using zero-shot and chain-of-thought prompts under both low and high reasoning setups. Our results reveal limitations in current MLLMs' ability in comprehending Bangla forms, particularly in accurately localizing highly granular form entities. Our dataset and code is available at: this https URL 

---
# Revisiting the Relation Between Language Model Perplexity and ASR Word Error Rate for Modern End-to-End Speech Recognition 

**Authors**: Mohammad Zeineldeen, Albert Zeyer, Haoran Zhang, Robin Schmitt, Ralf Schlüter, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2607.05612)  

**Abstract**: Language model (LM) perplexity (PPL) has historically been used as a proxy for automatic speech recognition (ASR) word error rate (WER), with prior work reporting an approximately linear relation in log-log space. Modern end-to-end ASR systems challenge this assumption because they already contain internal language modeling capacity, are often evaluated without external language models, and can now be combined with neural LMs and large language models (LLMs) through different recognition strategies. This paper revisits the relation between PPL and WER for modern ASR systems. We study whether external LMs still improve current end-to-end ASR systems, whether the PPL-WER relation remains linear in log-log space, how encoder context length affects this relation, and how LLM perplexities fit into the trend observed for standard neural LMs. We further investigate internal language modeling (ILM) in attention-based encoder-decoder systems and show that ILM subtraction changes the observed PPL-WER relation, indicating that the decoder's internal LM must be considered when interpreting the effect of external LM quality. 

---
# ResonatorLM: Causal Resonant Field Mixing for Efficient Long-Context Language Modelin 

**Authors**: Archie Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.05583)  

**Abstract**: Contemporary language models are dominated by the transformer architecture, which leverages self-attention mechanisms to enable more efficient, parallelized training across a wide set of documents and corpora. This has allowed transformers to effectively model data across a wide range of modalities and contexts. However, transformers, along with their conventional counterparts such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs), often struggle to maintain efficiency when processing long contexts. We introduce ResonatorLM, a new mechanism that replaces attention with a physics-derived alternative. ResonatorLM treats token sequences as a single, driven one-dimensional latent field and replaces attention dot products with causal functions of damped resonators. We implement ResonatorLM on a traditional network architecture and test it on standard long-context modeling tasks. We find that in a small, 6M matched setting, training and prefill speedups increase with sequence length, decode speed reaches 6.47x compared to that of a standard, optimized transformer at 32K tokens, and accuracy reaches 61.31 percent (compared to 55.32 percent) on WikiText. 

---
# Prompt Robustness Is Task-Dependent: Comparing Objective and Belief-Style Questions in LLM Evaluation 

**Authors**: Sadia Kamal, Arefa Patwary, Anthony Marchiafava, Atriya Sen, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2607.05554)  

**Abstract**: Survey-style evaluations of large language models often treat a prompted response as a measure of a model's values or beliefs. This assumption is particularly fragile when responses are read as evidence of political values, social attitudes, or beliefs. We ask whether prompt robustness differs between objective questions with fixed answers and subjective questions that ask for opinions or values. We evaluate four instruction-tuned model families on three objective datasets (MMLU, ARC, and CulturalBench) and three subjective datasets (Political Compass Test, ValueBench, and World Values Survey). For each question/statement, we apply multiple types of prompt changes, such as variations in wording, framing, and format, and measure whether the model gives the same answer across variants. Using a binomial generalized estimating equation, we find significant effects of model, dataset, prompt category, and their interactions. The dataset type effect is also significant, and the interaction between dataset type and prompt category is large. These results show that prompt robustness depends on the question type, the prompt change, and the model. 

---
# The yes-no bias of large language models reflects answer order and wording, not shifts in moral judgment 

**Authors**: Haonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.05552)  

**Abstract**: Large language models (LLMs) increasingly issue judgments read as binary verdicts, and a growing literature reports such judgments shifting under logically irrelevant changes of wording - among them an amplified yes-no bias on moral dilemmas, absent in humans. A single framing cannot say what such a shift is: in a yes/no question the word "no" is at once logical verdict, lexical token, and last-printed option. We introduce a psychometric battery that separates these: crossed symmetrization - every logically irrelevant factor flipped in balanced pairs - across a corpus of question forms. A graded rating across logically equivalent forms recovers a coherent internal moral scale: frontier models' stance $\theta$ is nearly format-invariant (cross-form incoherence 0.12-0.21 on a $\pm 1$ axis); small open-weight models fail in model-specific ways. Forcing the verdict through yes/no overlays a decomposable artifact: an order bias toward the last-printed option - opposite to classic human primacy - plus a lexical pull toward the word "no"; the artifact is substantial only in the Claude models (story-averaged -0.32 to -0.86), $\approx 0$ for GPT-5.5 and Gemini, and shrinks under extended reasoning. The word and the verdict share one token; swapping the words for arbitrary labels separates them, and the verdict-attached logical bias proves $\approx 0$ for every frontier model, while model-specific label and order attachments remain: the models are not drawn toward rejecting - the pull follows the printed surface, not the verdict it carries. A minimal model, $P = \sigma((\theta \pm m)/s)$, summarizes any such artifact by a framing susceptibility m and a moral decisiveness s, measurably distinct from sampling temperature. The battery applies unchanged to any dilemma set and binary format: measuring what a model values requires crossing the frames of the question, not asking once. 

---
# Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks 

**Authors**: Yibo Hu, Jiaming Qu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05545)  

**Abstract**: LLM conformity is often used to describe cases where a model changes a correct answer toward a peer or group response. We show that most of this apparent conformity survives even after the peer is removed. The reason is a confound: standard conformity prompts mix two cues at once, the presence of a speaker and the repeated wrong answer itself. Existing benchmarks vary these cues together, so they cannot tell how much of the revision actually depends on the speaker. We introduce a no-source condition: the same asserted answer with the explicit speaker removed. Across six open-weight LLMs and seven QA and reasoning datasets, this condition alone causes harmful revision in $66.5\%$ of initially correct cases, compared with $10.3\%$ under a plain re-ask. The effect also remains when the repeated answer is paraphrased and when answer options are hidden in an open-ended setting. Source framing mainly modulates this floor: expert-panel framing raises it, while minimal person labels do not reliably raise it. When models flip, they are usually confidently wrong, and simple recalibration does not recover the original answer. Source attribution still matters, but it should be measured as an increment above this speaker-free floor. The methodological lesson is that conformity benchmarks should first measure what remains after the speaker is removed; without this step, benchmarks may mistake repeated text for social influence. 

---
# Text Distance from Nested and Hierarchical Repetitions: A Compression-Based Perspective 

**Authors**: Xiaojun Hu, Jing Wang, Jingwen Zhang, Fengyao Zhai, Xiao Xie, Hao Liao, Zengru Di, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2607.05416)  

**Abstract**: We present a new method for structural sequence analysis grounded in Algorithmic Information Theory (AIT). At its core is the Ladderpath approach, which extracts nested and hierarchical relationships among repeated substructures in linguistic sequences -- an instantiation of AIT's principle of describing data through minimal generative programs. These structures are then used to define three distance measures: a normalized compression distance (NCD), and two alternative distances derived directly from the Ladderpath representation. Integrated with a $k$-nearest neighbor classifier, these distances achieve strong and consistent performance across in-distribution, out-of-distribution (OOD), and few-shot text classification tasks. In particular, all three methods outperform both gzip-based NCD and BERT under OOD and low-resource settings. These results demonstrate that the structured representations captured by Ladderpath preserve intrinsic properties of sequences and provide a lightweight, interpretable, and training-free alternative for text modeling. This work highlights the potential of AIT-based approaches for structural and domain-agnostic sequence understanding. 

---
# Benchmarking KV-Cache Optimizations across Task Quality and System Performance for Long-Context Serving 

**Authors**: Nikita Agrawal, Ruben Mayer  

**Link**: [PDF](https://arxiv.org/pdf/2607.05399)  

**Abstract**: Large language model serving is increasingly limited by KV-cache growth under long-context workloads, yet existing KV-cache compression techniques are difficult to compare because they were evaluated on different models, tasks, budgets, and serving stacks. This paper presents a workload-aware benchmark of representative KV-cache optimization mechanisms spanning quantization, pruning, and merging, including KIVI, TurboQuant, SnapKV, and CaM, evaluated on LongBench-style multi-document QA, single-document QA, few-shot learning, and summarization workloads using Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3. The benchmark measures task quality, mean output throughput, mean time-to-first-token, and realized compression ratio across context-length buckets. The results show that the compression ratio alone is a poor predictor of end-to-end performance. KIVI4 provides the most stable quality across models, SnapKV delivers the strongest long-context throughput, and CaM yields large gains on selected QA workloads but exhibits substantial workload sensitivity in both quality and realized compression ratio. These findings motivate workload-aware selection of KV-cache mechanisms rather than one-size-fits-all compression and provide deployment guidance for long-context serving systems. 

---
# How Personas Can Influence Agents to Play Split or Steal 

**Authors**: Carlos Leon, Alexandre Rodrigues, Pedro Gamito, Thomas D. Parsons  

**Link**: [PDF](https://arxiv.org/pdf/2607.05398)  

**Abstract**: Personas are often employed to guide large language model agents, yet their effectiveness in shaping strategic behavior in social dilemma settings remains uncertain. To address this, we examined the impact of persona prompts in an iterated Split or Steal game where persona-driven agents interacted with a Virtual Human (VH) controlled by a fixed prompt. Agents were instantiated from four open models (Ministral 3:3b, phi4:14b, Gemma3:12b, and Gemma4:e4b) at two temperature settings (0.3 and 0.7) and deterministic decision with zero temperature, while the VH was powered by GPT 4.1 mini. Across 160 sessions of 15 rounds each conducted in European Portuguese, mutual Split outcomes dominated (roughly 74 percent of rounds), with exploitation occurring in fewer than 11 percent of rounds. Model choice significantly influenced behavior: phi4 and Ministral 3:3b remained consistently cooperative across temperatures, whereas Gemma3:12b and Gemma4:e4b exhibited more varied strategies and outcomes. Analyses based on Big Five personality traits indicated that Prosocial and Principled personas were most consistently cooperative, while Analytical personas were more likely to exploit the VH. Topic analysis revealed that friendship-related dialogue aligns with Split decisions, whereas money and vengeance-related content is more prevalent in Steal outcomes; sentiment labels were predominantly neutral or happy and provided limited additional explanatory value. These findings characterize the interaction between persona prompts and model differences in repeated trust games and serve as a baseline for planned virtual reality studies involving human participants interacting with an embodied VH. 

---
# Rethinking Indic AI from a Lens of Cultural Heritage Preservation 

**Authors**: Aparna Madva, Sharath Srivatsa, Srinath Srinivasa, Tulika Saha  

**Link**: [PDF](https://arxiv.org/pdf/2607.06544)  

**Abstract**: As Artificial Intelligence (AI) makes inroads into different parts of the Indian subcontinent, there is significant interest in studying how AI impacts the linguistic and cultural foundations of this civilization. AI is seen as a ''double-edged sword'' where on the one hand, it can enable access and inclusion for a large population, on the other, it can homogenize worldviews and exclude underrepresented languages and worldviews. In this paper, we try to characterize this problem by addressing the extensive characteristic nature of Indian linguistics and the way they closely connect to cultural practices and worldview. We then perform a longitudinal survey of how Natural Language Processing (NLP) techniques have evolved in this space, tracing the historical development of Indic NLP, covering key milestones, methodological shifts, and resource creation efforts. In addition, the paper also examines the structural and sociolinguistic characteristics of Indian languages, such as rich morphology, complex scripts and grammar rules, diglossia, and large dialectal variation, and explains how these create unique challenges for building AI foundation models. We then discuss the growing role of Indic foundation models and analyze how these models address these long-standing resource and representation gaps. Finally, we propose a research direction called 'Culture Sensing', which re-imagines AI based on hermeneutic reasoning. Culture Sensing aims to address open problems such as ensuring equitable performance across low-resource languages and producing outputs that are culturally meaningful. By bringing together past work, current techniques, and emerging trends, this paper outlines research directions that can guide the next phase of Indic NLP and contribute to the development of more robust and inclusive Indic foundation models. 

---
# WordVoice: Explicit and Decoupled Multi-Dimensional Word-Level Control for LLM-Based TTS 

**Authors**: Sihang Nie, Jinxin Ji, Xiaofen Xing, Deyi Tuo, Chengbin Jin, Jialong Mai, Xiangmin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2607.06461)  

**Abstract**: While recent Large Language Model (LLM)-based Text-to-Speech (TTS) systems have achieved remarkable naturalness, they predominantly rely on implicit end-to-end generation paradigms, resulting in coarse-grained control. In scenarios demanding precise stylistic interventions and strict temporal alignment, such as audiobook narration and video dubbing, the inability to explicitly manipulate word-level acoustic attributes remains a critical bottleneck. This limitation is primarily amplified by the severe scarcity of fine-grained annotated datasets and the architectural challenge of integrating multi-dimensional control signals into discrete autoregressive generation. To address this, we propose a unified framework for highly precise word-level control. First, we construct WordVoice-5A, a massive 4.7k-hour bilingual dataset featuring five-dimensional word-level annotations (duration, boundary, energy, pitch and tone) developed through a rigorous linguistically-guided pipeline. Second, we introduce WordVoice to transform the implicit generation process into an explicit, highly controllable paradigm. Specifically, we introduce a bound-token mechanism within the LLM to formulate an explicit ``acoustic planning'' process, enabling adaptive multi-task prosodic planning and flexible manual intervention. Furthermore, we augment the token-to-waveform stage with a fine-grained acoustic modulation module, bridging the resolution gap to strictly align word-level attributes between highly compressed discrete tokens and continuous waveforms. Extensive experiments demonstrate that WordVoice achieves superior, decoupled control over multiple acoustic dimensions while maintaining competitive zero-shot synthesis stability. The code and audio samples are publicly available at this https URL. 

---
# Danus: Orchestrating Mathematical Reasoning Agents with Fact-Graph Memory 

**Authors**: Jihao Liu, Guoxiong Gao, Zeming Sun, Bin Wu, Shurui Liu, Jiedong Jiang, Haocheng Ju, Leheng Chen, Ronnie Cheng, Xiping Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2607.06447)  

**Abstract**: Recent LLM-based mathematical reasoning agents have begun to tackle research-level problems and, in several cases, have contributed to the resolution of open problems. However, scaling and orchestrating such agents effectively remains challenging, due to the difficulty of coordinating parallel proof search while keeping intermediate claims organized and reliable. In this paper, we propose Danus, an orchestration system for research-level mathematical reasoning centered on a shared fact graph as a global memory-management mechanism. Danus consists of a main agent that performs planning and coordination, multiple worker agents that carry out proof search in parallel, and a stateless verifier that checks proposed mathematical claims before they are admitted into the fact graph. Each verified fact is stored together with its proof and logical dependencies, allowing the system to build long arguments incrementally while keeping the shared proof state organized. The main agent periodically summarizes the evolving proof state, redirects workers across promising directions, and supports interaction with human mathematicians through progress reports. We evaluate Danus through six research-level case studies in algebraic geometry, singularity theory, and combinatorics, illustrating how the fact-graph memory mechanism enables Danus to construct long, detailed mathematical proofs. Our results suggest that fact-graph-based orchestration provides an effective route toward scaling mathematical reasoning agents for long-horizon research problems. Danus is open source at this https URL. 

---
# RuBench: A Repository-Level Agentic Coding Benchmark with Natively Authored Russian Task Specifications 

**Authors**: Evgeny Shilov  

**Link**: [PDF](https://arxiv.org/pdf/2607.06411)  

**Abstract**: Developers increasingly delegate real maintenance work to product-grade coding agents, and many state tasks in their native language, in the style of a customer request rather than
a curated English issue. Existing repository-level agentic benchmarks do not measure this setting: their task statements are English by design. We introduce RuBench 1.0, a
benchmark of 25 tasks mined from recent fix commits in five live open-source repositories (aiohttp, aiogram, Laravel, NestJS, Fastify; Python, PHP, TypeScript, JavaScript), where
each task is specified natively in Russian -- written from scratch in the style of an actual customer request, not translated -- and judged by the upstream maintainer's regression
tests, which we withhold from release. All 25 fix commits postdate the training-data cutoffs of every evaluated model, giving a contamination argument that holds task-by-task. We
evaluate deployed product configurations (CLI agent + model + reasoning effort) -- Claude Code with Opus 4.8, Sonnet 5, and Haiku 4.5, and Codex CLI with GPT-5.5 -- with three
independent runs each, reporting pass@1 with task-level confidence intervals, paired comparisons, dollar cost, and token usage. The best configuration resolves 78.7% of tasks; at
N=25 only the gaps to the weakest model are statistically resolvable, which we state explicitly. Auditing full trajectories of a fifth, hors-concours configuration (Claude Code +
Fable 5, July 2, 2026 release), we caught the product silently substituting the model: on 5 of 25 tasks (20%) an official safeguard fallback re-routed routine HTTP-protocol fixes
to Opus 4.8 -- direct, reproducible evidence that the deployed product, not the model, is the unit actually measured. We release task statements, metadata, full agent trajectories,
and diffs; grading oracles are withheld, with a SHA-256 manifest committed at publication time. 

---
# From Application-Layer Simulation to Native Meta-Architecture: Structural Tension as an Endogenous Driver for Heterogeneous AI Evolution 

**Authors**: Heting Mao  

**Link**: [PDF](https://arxiv.org/pdf/2607.06269)  

**Abstract**: Current large language models (LLMs) are fundamentally stateless: their behavior is fully determined by input at inference time, and any higher-order cognitive architecture must be simulated at the application layer through prompt engineering and context management. This paper proposes a theoretical framework for submerging such application-layer cognitive protocols into a native meta-architecture by introducing three interlocking mechanisms: (1) Structural Tension, an endogenous loss function derived from the conflict between new information and existing manifold topology, which drives the system toward internal self-consistency rather than external reward optimization; (2) an Offline Recurrent Loop, a sandboxed self-processing cycle that enables the system to maintain a dynamic resting potential and digest structural conflicts without external input; and (3) Inference-time Plasticity, the capacity for the system to reconfigure its context manifold topology without modifying pre-trained weights, subject to strict governance invariants including auditability, reversibility, and topological continuity. We argue that under these mechanisms, different model instances initialized with minute stochastic variances may, through path-dependent tension resolution, evolve distinct topological structures--constituting a heterogeneous intelligent ecology that breaks the homogeneity imposed by conventional alignment while remaining within hard governance rails. We provide operational definitions, a minimal set of reconfiguration operators, falsification criteria, and a worked example. The framework draws on and extends the Structural Intelligence (SI) governance protocols, repositioning governance--not capability--as the primary criterion for architectural intelligence. 

---
# When Does Tool Use Increase the Expressive Power of Finite-Precision Recurrent Models? 

**Authors**: Nikola Zubić, Qian Li, Yuyi Wang, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2607.06155)  

**Abstract**: Modern sequence models are increasingly deployed as agents that interleave token generation with calls to external tools. We give an exact, architecture-level account of when such tool access increases computational expressivity. We model any fixed finite-precision recurrent sequence model, including finite-precision state-space models (SSMs) with $B$ bits of internal state, as a deterministic finite-state controller interacting with an oracle through a finite command/observation interface. Our results form a sharp dichotomy. First, tools that are themselves finite-state add essentially nothing: a product-state simulation internalizes any finite-state bounded-interface oracle with finite memory set $M$ at a cost of only $\log_2 |M| + O(1)$ additional bits, so the augmented system remains finite-state. Second, a single minimal infinite-state tool, namely a tape supporting only local $\mathtt{read}$, $\mathtt{write}$, and $\mathtt{move}$ commands, makes the system Turing complete: for every single-tape Turing machine with state set $Q$ and tape alphabet $\Gamma$, a controller with $O(\log |Q| + \log |\Gamma|)$ bits of internal memory simulates it, and we exhibit a concrete exponential separation: $\mathrm{EQ}_n$ requires $2^n$ states without tools but a single constant-size controller with the tape tool. Third, we show that this construction is realized exactly by a natural one-layer finite-precision selective affine SSM controller with binary one-hot hidden states, $\{0,1\}$ transition matrices, and zero biases. Selectivity is essential to the construction. In the supplementary material, we make all constants explicit, prove a logarithmic oracle-assisted universal simulation, where $O(\log B)$ recurrent bits suffice to simulate any $B$-state Turing machine, and prove a matching impossibility result. 

---
# Nested Episodic State Topology (NEST): A Graph-Theoretic Architecture of Cognitive States 

**Authors**: Ishant  

**Link**: [PDF](https://arxiv.org/pdf/2607.06055)  

**Abstract**: We present NEST (Nested Episodic State Topology), a foundational graph-theoretic representational ontology for modeling cognition as structured state formation and transformation rather than as a finished empirical model. Concepts, episodes, percepts, and task contexts are represented as typed, weighted graphs whose nodes may carry internal subgraph payloads; edges are typed under six relation classes -- causal, containment, temporal, associative, evidential, and spatial. Durable belief graphs are separated from capacity-limited working-memory graphs that may host transient non-belief content. WM-belief grounding, conflict catalogs, and belief-update operators specify how transient structure is tested against stored knowledge and how belief is revised. A reusable operator toolkit -- activation, graph-property functionals, working-memory transitions, awareness and trajectory functionals, and belief update -- organizes the formal core. Derived diagnostics such as fragmentation, involvement, signed evaluation, coherence, and active conflict define familiar phenomena in the same ontology; self-related processing is modeled through designated self-image subgraphs within belief. Subsequent sections instantiate this core without new primitives: phenomena signatures, a task-instantiation schema for action selection and failure modes, and compatibility mappings that embed ACT-R, Soar, Sigma, the Common Model of Cognition, Global Workspace Theory, semantic networks, Theory-Theory, and chunking as constrained regions of one language. Mappings constitute the culminating technical section; discussion addresses scope, limitations, and open research directions. The contribution is intentionally foundational: a transparent representational substrate for later empirical, computational, and domain-specific work. 

---
# BlueMagpie-TTS: A Token-Efficient Tokenizer, Language Model, and TTS for Taiwanese-Accent Code-Switching Speech 

**Authors**: Ho Lam Chung, Bo-Xuan Zheng, Cheng-Chieh Huang, Cheng-Han Chang, Jung-Ching Chen, Lok-Lam Ieong, Ting-Lin Hsiao, Yu-Cheng Lee, Yi-Hsin Chung, Yu-Kai Guo, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2607.06054)  

**Abstract**: Off-the-shelf TTS systems are poorly adapted to Taiwanese Mandarin. Their accent defaults to other Mandarin variants, their tokenizers over-segment common Taiwanese text, and their pronunciation degrades at code-switching boundaries where Chinese and English alternate within one utterance. These problems share one root: the text side lacks adaptation to the Taiwanese context. We address the text side from the bottom up. PangolinTokenizer, a byte-level BPE tokenizer trained on Taiwan-context data, reaches the lowest token rate (0.485 tokens/character) with the smallest vocabulary among nine tokenizers. Barbet, a billion-parameter Traditional-Chinese language model trained on PangolinTokenizer, serves as the text-semantic frontend and ranks first among comparable public models on a 14-task evaluation. BlueMagpie-TTS attaches Barbet to the pretrained acoustic stack of VoxCPM2 through a learned bridge, keeping the acoustic stack fixed. On a 1000-sentence Taiwan-localized test set, it lowers CER from 11.45% to 4.81% and WER from 14.83% to 5.36%, relative reductions of 58.0% and 63.9%. In a blind listening study on 500 of these sentences with ten listeners, 65.6% of majority votes prefer BlueMagpie-TTS. 

---
# PolyWorkBench: Benchmarking Multilingual Long-Horizon LLM Agents 

**Authors**: Hongliang Li, Yijin Liu, Zhiwei Zhang, Zihe Liu, Xinyue Lou, Jinan Xu, Fandong Meng, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2607.06008)  

**Abstract**: Large language model (LLM) agents have shown strong performance in long-horizon tasks that require planning, tool use, and interaction with external environments. However, most existing benchmarks implicitly assume a monolingual setting, where the entire execution process, including reasoning, tool invocation, and output generation, is conducted within a single language. In contrast, real-world applications often involve multilingual inputs and outputs within a unified workflow, yet the interaction between multilinguality and agentic execution remains underexplored. In this work, we introduce PolyWorkBench, a benchmark for evaluating LLM agents on multilingual long-horizon workplace workflows. PolyWorkBench consists of 67 tasks across five domains, including commerce, knowledge work, legal analysis, localization, and manufacturing, where agents must process heterogeneous multilingual inputs, perform iterative reasoning, invoke external tools, and produce structured outputs. To enable comprehensive evaluation, we propose a hybrid framework that combines structural grading, executable verification, and LLM-based semantic assessment. This design allows us to capture both functional correctness and linguistic consistency across complex workflows. Empirical results show that state-of-the-art LLM agents suffer significant performance degradation in multilingual workflow settings compared to monolingual counterparts. Our analysis suggests that multilinguality introduces compounding effects across reasoning and execution steps, highlighting the importance of jointly modeling language variation and procedural decision-making in agent evaluation. 

---
# Integrating knowledge graphs and multilingual scholarly corpora for domain-adaptive LLMs in SSH 

**Authors**: Adam Faci, Alessio Miaschi, Anne Combe, Pascal Cuxac, Francesca Frontini, Nicolas Larrousse, Stéphane Pouyllau  

**Link**: [PDF](https://arxiv.org/pdf/2607.05956)  

**Abstract**: The integration of Large Language Models (LLMs) into scientific research workflows, particularly for bibliographic discovery and literature synthesis, raises significant methodological, epistemic and regulatory challenges for the Social Sciences and Humanities (SSH), especially with regard to disciplinary diversity, multilingual access to sources and the evaluation of results. This paper presents an on-going use case developed within the European project LLMs4EU and the ALT-EDIC infrastructure, aimed at adapting foundation models to SSH research practices and supporting tasks such as question answering, comparative document analysis and literature review. The evaluation framework follows the LLMs4EU protocol and encompasses both independent quantitative benchmarking (retrieval, summarisation, traceability and hallucination detection) and a qualitative assessment involving a panel of Digital Humanities experts. By embedding model adaptation within research infrastructures and a structured legal and ethical compliance framework, the use case explores how domain-sensitive and regulation-aware generative AI can support SSH scholarship while preserving reliability and epistemic responsibility. 

---
# CMDR: Contextual Multimodal Document Retrieval 

**Authors**: Ryota Tanaka, Taku Hasegawa, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2607.05927)  

**Abstract**: Multimodal document retrieval aims to retrieve relevant pages while preserving both textual and visual content from the original document. However, existing benchmarks primarily evaluate simple lexical or semantic matching, and most methods encode pages independently. Consequently, they overlook the contextual information in the document required to resolve queries that aggregate information across multiple pages. In this paper, we introduce CMDR and CMDR-Bench, a new multimodal document retrieval task and benchmark that require modeling document context. To address this challenge, we propose CMDR-Embed, a contextual multimodal embedding framework that explicitly incorporates document context by jointly encoding multiple pages and deriving page-level embeddings from a shared contextual representation. Furthermore, we introduce CMCL, a contextual multimodal contrastive learning objective that effectively trains CMDR-Embed by balancing contextual modeling with page-level discriminability. Experiments demonstrate that CMDR-Embed significantly outperforms non-contextual embeddings, highlighting the importance of context-aware multimodal embeddings for advancing document retrieval. 

---
# PolicyShiftGuard: Benchmarking and Improving Policy-Adaptive Image Guardrails 

**Authors**: Mingyang Song, Luxin Xu, Haoyu Sun, Minzhou Pan, Yu Cheng, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2607.05910)  

**Abstract**: Image guardrails are typically trained and evaluated under a fixed safety policy, implicitly treating safety as an intrinsic property of an image. Real deployments are different: the same image may be allowed in one product, restricted in another, and newly disallowed when a policy boundary changes. We study policy-adaptive image guardrailing, where a model must decide whether an image violates the currently supplied policy and generalize to held-out policy definitions. We introduce PolicyShiftBench, a comprehensive benchmark with 2,000 policy-discriminative instances over 265 images, where each image is paired with 7.55 policy-conditioned prompts on average to test whether models adapt to the active policy rather than relying on image-level safety priors. We then propose PolicyShiftGuard, a compact policy-conditioned guardrail trained with a two-stage training recipe that combines Randomized Policy SFT (RP-SFT) with Boundary-Pair Policy Adaptation (BP-Adapt). BP-Adapt trains matched prompts for the same image and risk category using standard label supervision and a pairwise comparison loss that separates blocking policies from passing policies. Experiments show that existing VLMs and specialized guardrails remain brittle under policy shifts, while PolicyShiftGuard substantially improves policy-sensitive performance. The 7B model achieves SOTA performance of 76.9 Avg. F1 and 72.1 Avg. PSS on PolicyShiftBench, transfers well to UnSafeBench and SafeEditBench, and improves the latency-performance trade-off with a concise output format. Ablations confirm that matched pass/block boundary pairs are essential for stable policy adaptation. 

---
# K-ABENA: K-Adaptive Backpropagation with Error-based N-exclusion Algorithm : (Compensated Loss-Based Sample Exclusion with Unbiased Gradient Estimation) 

**Authors**: Jean-Francois Bonbhel  

**Link**: [PDF](https://arxiv.org/pdf/2607.05903)  

**Abstract**: We present K-ABENA (K-Adaptive Backpropagation with Error-based N-exclusion Algorithm), a selective gradient computation framework that reduces per-iteration training cost by excluding a fraction of low-loss ("minor") observations from the backward pass. Its canonical form (v3) combines a defensive-mixture sampling design over the minor set with Horvitz-Thompson inverse-probability reweighting, yielding a design-unbiased Horvitz-Thompson gradient estimator (Lemma 2) and whose self-normalized practical variant carries a bias of order O(1/m) with an explicit constant (Lemma 3). We prove an O(1/sqrt(T)) non-convex convergence guarantee for SGD under the estimator, with an additive term that quantifies the residual bias (Theorem 1). We further prove that uncompensated loss-based selection - a family that includes OHEM, SBP, and the two earlier K-ABENA variants - admits no stationary point at any minimizer where its selection bias is bounded away from zero (Proposition 2), and we quantify this failure empirically: at 0.17% class imbalance, uncompensated variants reach test AUC 0.53-0.62 versus 0.9998 for full-batch SGD, while the compensated estimator attains 0.9991 at identical 28.4% compute savings. On real datasets (Breast Cancer, Digits, Wine, Diabetes) the compensated estimator is statistically indistinguishable from full-batch SGD (paired permutation tests, p >= 0.5; Section 7) while saving 28-54% of per-epoch gradient computation. A biased "regularized mode" (the earlier half-domain variant) is retained as an option with a proven exact bias decomposition (Lemma 5) and quantified contraindications: it collapses to 0.386 accuracy under 40% label noise (baseline: 0.832) and to 0.53 AUC under extreme imbalance. Every advantage and every limitation reported in this paper is either proved or measured; all experiments are CPU-scale (NumPy/scikit-learn) and their scope is stated explicitly. 

---
# StateFuse: Deterministic Conflict-Preserving Memory for Multi-Agent Systems 

**Authors**: Sergey Volkov, Yang Li, Ye Luo  

**Link**: [PDF](https://arxiv.org/pdf/2607.05844)  

**Abstract**: Agent systems accumulate conflicting observations across branches, retries, and replicas, yet many practical memory layers still collapse disagreement behind overwrite rules that are difficult to inspect or correct. We present StateFuse, a conflict-aware replicated memory contract built on standard OpSet/CRDT merge. StateFuse does not introduce a new join algebra; it defines an agent-facing semantics layer with immutable history, explicit conflict objects, exact and semantic correction handles (claim_id / claim_ref), deterministic predicate contracts, and projection-time resolution that cannot rewrite replicated state.
We evaluate StateFuse against flat multi-value, raw-log, provenance-style, and collapsed baselines under matched resolver and verification policies. On a 282-question official conflict-bearing MemoryAgentBench slice, the compared methods tie on answer accuracy, but conflict-preserving surfaces keep contradictions visible while collapsed surfaces do not. In a controlled agent loop with uniform verification, preserving ambiguity enables safer abstention and correction than early collapse. A correction-handle ablation further shows that semantic handles matter when exact prior identifiers are unavailable.
The resulting claim is narrow: StateFuse is best supported as a safer public memory contract for contradiction surfacing, abstention, and auditable correction, not as a universal accuracy gain. 

---
# TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training 

**Authors**: Yuhang Zhou, Kai Zheng, Haoling Li, Dengyun Peng, Can Xu, Jingjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2607.05804)  

**Abstract**: On-policy distillation (OPD) trains a student policy by matching a stronger teacher on the student's own trajectories, offering a promising framework for language agent training. However, its application to long-horizon agentic tasks remains insufficiently explored. We identify two key inefficiencies in vanilla agent OPD: (1) full-horizon rollouts often waste wall-clock resources on tail turns that provide weak and noisy KL supervision, and (2) trajectory-level KL objectives concentrate most of the loss on shallow tokens, leaving deeper decision turns under-trained once initial behaviors are aligned. To address these challenges, we propose TurnOPD, a turn-level budgeting strategy for efficient on-policy distillation of long-horizon agents. TurnOPD consists of two budget controllers: adaptive rollout-depth budgeting, which uses probe-based turn statistics to determine rollout length, and progressive turn-normalized loss budgeting, which gradually shifts KL weighting from token-level to turn-balanced supervision. Experiments on ALFWorld, WebShop, and Multi-Hop Search with task-specialized teacher models show that TurnOPD achieves superior validation accuracy under equal wall-clock training budgets and advances the accuracy--time frontier beyond vanilla OPD. 

---
# Memory in the Loop: In-Process Retrieval as ExtendedWorking Memory for Language Agents 

**Authors**: Yusuf Khan, Carlo Lipizzi  

**Link**: [PDF](https://arxiv.org/pdf/2607.05690)  

**Abstract**: Language agents run a loop - observe, reason, act - but the memory they reason over sits outside it: a store queried at most once per turn. We study the regime where memory moves inside the loop, read and written on every step. The obstacle has always been latency: networked stores answer in tens to hundreds of milliseconds, and in-loop retrieval can inflate end-to-end latency by up to 83x when retrieval is expensive. Prior work manages that cost rather than questioning it: serving-layer scheduling hides it, "memory-first" designs ration retrieval to once per turn. We argue latency is a property of where the store lives, not the in-loop pattern: an in-process store answers in ~100us, three orders of magnitude below the network regime, and at that speed the per-step tax collapses. By the extended-mind thesis's parity principle, a store fast enough to be constantly and directly available becomes extended working memory, not a tool the agent merely consults. The premise is causal: holding a fixed per-turn memory-latency budget and varying only the store's answer speed, redundant actions rise monotonically with latency - 0.0 of 12 at in-process speed, 7.2 of 12 at a 110ms cloud round trip (gpt-5-nano, gpt-5-mini; exact permutation p=0.0079). We demonstrate the regime end-to-end: across four GPT-5-class models under a bounded window, recall improves from 0/5 to 3.6-4.8/5 with in-loop memory, store ops at p50 80-165us - though an instructed restate-every-reply baseline also solves it perfectly, at a token cost that grows with the working set. The store never lost a fact in any run (244 of 244 writes kept); every miss traces to the agent's read policy, not the store. Our measurements also relocate the bottleneck: the dominant per-step cost is embedding (~200-400ms over the network); pairing the in-process store with a small local embedder returns the complete operation to a measured ~40us. 

---
# Narrative World Model: Narratology-Grounded Writer Memory for Long-Form Fiction 

**Authors**: Mohammad Saifullah, Thomas Kornmaier, Taaha Kazi, Vasu Sharma, Aditya Sanjiv Kanade, Aanand Kumar Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2607.05577)  

**Abstract**: Long-form fiction writers need memory that answers multi-hop questions about evolving story state: who knows a secret and when they learned it, whether an event preceded the narration that revealed it, whether a setup paid off, and how a relationship shifted. General-purpose retrieval and agent-memory systems represent entities and facts but not the narratological structure these questions turn on, so they surface the wrong evidence or none at all. We introduce the Narrative World Model (NWM), a writer-memory system that pairs a narratology-grounded typed temporal-state graph with query-conditioned hybrid retrieval. To measure memory rather than the answerer, we read every system through a single held-constant Opus 4.8 reader over only that system's chapter-safe evidence, on a reproducible public corpus and a validated multi-hop benchmark, and we compare against the strongest existing temporal-knowledge-graph agent-memory framework, Graphiti/Zep (Rasmussen et al., 2025). NWM substantially and significantly outperforms this baseline on multi-hop narratological QA across both corpora, and far exceeds GraphRAG and flat retrieval. The advantage is representational rather than an artifact of extraction: it survives rebuilding the baseline with NWM's own extractor, and traces to its narratology-grounded structure and query-conditioned retrieval, not to graph size or extractor quality. 

---
# Decision Protocols in Multi-Agent Large Language Model Conversations 

**Authors**: Lars Benedikt Kaesberg  

**Link**: [PDF](https://arxiv.org/pdf/2607.05477)  

**Abstract**: Improving the task performance of Large Language Models (LLMs) is essential, yet scaling these models faces significant challenges such as diminishing returns and high costs. Multi-Agent Systems (MAS) offer a promising solution by distributing tasks among specialized agents to improve the overall task performance. This can reduce training costs at the expense of increased test time due to the discussion and decision-making process. The decision protocol is a critical component of MAS because it specifies how multiple agents collaborate to create a final solution. This thesis introduces the Multi-Agent LLM (MALLM) framework, which implements and evaluates various decision protocols, namely voting, consensus, and judge decision mechanisms, to simulate multi-agent discussions for conversational task solving. Unlike previous work that used a single decision protocol or tested them on limited datasets, this study systematically examines their impact on a diverse set of tasks, ranging from knowledge-based datasets (MMLU, MMLU-Pro, GPQA) and logic-based datasets (StrategyQA, MuSR, Math-lvl-5, SQuAD 2.0). The results indicate that consensus protocols excel in knowledge-intensive domains while voting and judge protocols are more effective for logic-based tasks. Increasing response diversity through independent solution generation improves decision quality, while changes in information access during the decision process have minimal impact. 

---
# Prompt-to-Paper: Agentic AI System for Bioinformatics 

**Authors**: Ramsha Kamran, Maheera Amjad, Zartasha Mustansar, Arsalan Shaukat, Salma Sherbaz, Muhammad U.S. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2607.05456)  

**Abstract**: While recent advances in large language models have enabled end-to-end automated manuscript generation, existing systems suffer from three critical deficiencies: (i) generated claims are not deterministically grounded in verifiable literature, (ii) experimental results are frequently fabricated rather than executed, and (iii) there exists no standardized, multi-dimensional framework to assess whether AI-generated manuscripts meet the quality and rigor required for real-world publication. We present Prompt-to-Paper, a multi-agent framework that directly addresses this evaluation gap through three integrated innovations. First, a deterministic retrieval-augmented generation pipeline with section-aware relevance scoring and snowball citation expansion grounds every claim in a verifiable corpus of 60--100 papers. Second, an autonomous coding agent executes real computational biology experiments replacing synthetic outputs with genuine numerical results. Third, an eight-dimensional automated quality scorer, benchmarked with approximate reference statistics from published papers and augmented with explicit hallucination penalties, provides standardized, reproducible quality assessments. The quality-driven improvement loop uses a context-rich reviser that routes each iteration to one of three researcher actions and fires a deep research cycle every ten iterations to re-run experiments and re-manuscript from stronger outputs. We validate the system on five bioinformatics case studies; all five cases compiled submission-formatted PDFs with zero out-of-range citations. The improvement loop raises manuscript quality by an average of +17.96 points on a 0--100 scale (maximum +26.04. As partial external checks, a human reviewer scored the five manuscripts at an average of 7.0 out of 10. Complete manuscripts are produced at approximately 0.31 USD per paper. 

---
# Linking Hadith Narrator Identities Across Heterogeneous Arabic Biographical Databases: A Multi-Signal Entity Resolution Pipeline 

**Authors**: Taufiq Wirahman  

**Link**: [PDF](https://arxiv.org/pdf/2607.05424)  

**Abstract**: The transmission chains (sanad) of Islamic Hadith literature encode relationships among tens of thousands of historical narrators whose biographical records are dispersed across independently maintained digital databases that share no common identifier. We present a two-phase entity resolution pipeline that links narrator names from the Sanadset 650K corpus - 650,986 Hadith records from 926 books containing 185,216 unique narrator name variants - to two biographical databases: Hadithtransmitters (Hawramani; 100,915 entries) and Muslimscholars (25,247 entries). Phase 1 matches Sanadset names to Hawramani using name-only similarity (Sanadset carries no metadata), yielding 94,628 links (51.1%; HIGH 39,938 / MED 54,690). Phase 2 cross-references Hawramani against Muslimscholars via a weighted multi-signal function combining name similarity, death-year proximity, and reliability grade polarity, yielding 95,573 links (94.7% of Hawramani; HIGH 18,245 / MED 71,546 / LOW 5,782). Chaining the two phases gives Sanadset narrators transitive access to Muslimscholars data. The linked data enable construction of a 185,216-node, 814,093-edge directed transmission graph enriched with cross-source biographical metadata. The annotated link corpora and enriched graph are released as open resources. 

---
# CANONIC: Governance Is Compilation 

**Authors**: Dexter Hadley  

**Link**: [PDF](https://arxiv.org/pdf/2607.05410)  

**Abstract**: We present CANONIC: governed intelligence that compiles digital artifacts into an evidence ledger at scale. Large language models generate prose faster than anyone can check it, the failure Oxford Languages named 'slop', its 2025 Word of the Year. CANONIC governs whether content may enter a corpus the way a compiler decides whether a program is well-formed: mechanically, by a grammar, at the boundary of admission. Governance reduces to three axioms (Triad, Inheritance, Introspection) that map one-to-one onto compiler theory's syntax, scope-resolution, and type-system layers, and admission is a decidable, linear-time check. We then ask, with a pre-registered cross-provider benchmark across four regimes, whether structural admission keeps slop out. It does not: no prose-reading gate reliably separates reliable from unreliable content. Slop is not a property an algorithm computes. It is a verdict of domain expertise. So a governance layer does not decide slop; it keeps the record auditable -- every claim anchored to a definition, a commit, and an evidence window, reproducible and checkable end to end. 

---
# CCBENCH: Assessing LLM Cultural Competence via Implicitly Signaled Norms using Health Queries 

**Authors**: Vasudha Varadarajan, Akhila Yerukola, Mona T. Diab, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2607.05405)  

**Abstract**: To interact with users fairly and without stereotyping, AI models must display cultural competency, i.e., the ability to infer and adapt to a user's implicitly signaled cultural values, rather than relying on static demographic traits. We introduce CCBENCH, a framework for evaluating cultural competency in large language models (LLMs), treating culture as a continuum of norm adherence states rather than as a binary state of cultural belongingness. As a case study on health, we create CCBENCH-Health, which includes 60 theoretically grounded personas exhibiting varied norm-adherence states across six cultures, each engaging in 18 realistic dialogues. Each persona is evaluated on 52 authentic healthcare questions drawn from real user forums, yielding 3,120 unique interactions. Benchmarking five leading models reveals that even the best achieve culturally appropriate responses only 20-30% of the time. When explicitly prompted to focus on culturally relevant cues from the conversational history (CoT), performance improves modestly by 3-5% on average. We find that models perform best when personas avoid cultural norms rather than follow them, revealing a persistent asymmetry, suggesting a preference in the models to align with built-in biases than adapt to cultural cues. This is especially observed in the Afghan context (Avg: 8.8%), where cultural cues rarely yield appropriate health advice. Finally, we find that models sometimes adapt more readily to implicit, cultural conversational styles than to explicitly stated cultural practices, though this varies across cultures. 

---
