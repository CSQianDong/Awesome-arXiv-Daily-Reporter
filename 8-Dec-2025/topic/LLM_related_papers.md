# The Missing Layer of AGI: From Pattern Alchemy to Coordination Physics 

**Authors**: Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2512.05765)  

**Abstract**: Influential critiques argue that Large Language Models (LLMs) are a dead end for AGI: "mere pattern matchers" structurally incapable of reasoning or planning. We argue this conclusion misidentifies the bottleneck: it confuses the ocean with the net. Pattern repositories are the necessary System-1 substrate; the missing component is a System-2 coordination layer that selects, constrains, and binds these patterns. We formalize this layer via UCCT, a theory of semantic anchoring that models reasoning as a phase transition governed by effective support (rho_d), representational mismatch (d_r), and an adaptive anchoring budget (gamma log k). Under this lens, ungrounded generation is simply an unbaited retrieval of the substrate's maximum likelihood prior, while "reasoning" emerges when anchors shift the posterior toward goal-directed constraints. We translate UCCT into architecture with MACI, a coordination stack that implements baiting (behavior-modulated debate), filtering (Socratic judging), and persistence (transactional memory). By reframing common objections as testable coordination failures, we argue that the path to AGI runs through LLMs, not around them. 

---
# Evolutionary System 2 Reasoning: An Empirical Proof 

**Authors**: Zeyuan Ma, Wenqi Huang, Guo-Huan Song, Hongshu Guo, Sijie Ma, Zhiguang Cao, Yue-Jiao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2512.05760)  

**Abstract**: Machine intelligence marks the ultimate dream of making machines' intelligence comparable to human beings. While recent progress in Large Language Models (LLMs) show substantial specific skills for a wide array of downstream tasks, they more or less fall shorts in general intelligence. Following correlation between intelligence and system 2 reasoning (slow thinking), in this paper, we aim to answering a worthwhile research question: could machine intelligence such as LLMs be evolved to acquire reasoning ability (not specific skill) just like our human beings? To this end, we propose evolutionary reasoning optimization (ERO) framework which performs survival of the fittest over a population of LLMs to search for individual with strong reasoning ability. Given a reasoning task, ERO first initializes multiple LLMs as a population, after which an evolutionary strategy evolves the population to maximize quantified reasoning score of the best individual. Based on experiments on representative testsuites, we claim two surprising empirical discoveries: i) the latest LLMs such as GPT-5 still show limited system 2 reasoning ability; ii) with simple evolution-loop of ERO, a relatively weak model (Qwen-7B) could be enhanced to emerge powerful reasoning ability. Our project can be accessed at this https URL for reproduction needs. 

---
# TRACE: A Framework for Analyzing and Enhancing Stepwise Reasoning in Vision-Language Models 

**Authors**: Shima Imani, Seungwhan Moon, Lambert Mathias, Lu Zhang, Babak Damavandi  

**Link**: [PDF](https://arxiv.org/pdf/2512.05943)  

**Abstract**: Reliable mathematical and scientific reasoning remains an open challenge for large vision-language models. Standard final-answer evaluation often masks reasoning errors, allowing silent failures to persist. To address this gap, we introduce TRACE, a framework for Transparent Reasoning And Consistency Evaluation that diagnoses reasoning trajectories rather than only end results. At its core, TRACE leverages Auxiliary Reasoning Sets, compact sub question answer pairs that decompose complex problems, evaluate intermediate steps through consistency-based metrics, and expose failures overlooked by standard evaluation. Our experiments show that consistency across ARS correlates with final-answer correctness and helps pinpoint the reasoning steps where failures arise, offering actionable signals for model improvement. Furthermore, TRACE defines confidence regions that distinguish reliable from unreliable reasoning paths, supporting effective filtering, debugging, and model refinement. 

---
# MIND: Multi-rationale INtegrated Discriminative Reasoning Framework for Multi-modal Large Models 

**Authors**: Chuang Yu, Jinmiao Zhao, Mingxuan Zhao, Yunpeng Liu, Xiujun Shu, Yuanhao Feng, Bo Wang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2512.05530)  

**Abstract**: Recently, multimodal large language models (MLLMs) have been widely applied to reasoning tasks. However, they suffer from limited multi-rationale semantic modeling, insufficient logical robustness, and are susceptible to misleading interpretations in complex scenarios. Therefore, we propose a Multi-rationale INtegrated Discriminative (MIND) reasoning framework, which is designed to endow MLLMs with human-like cognitive abilities of "Understand -> Rethink -> Correct", and achieves a paradigm evolution from passive imitation-based reasoning to active discriminative reasoning. Specifically, we introduce a Rationale Augmentation and Discrimination (RAD) paradigm, which automatically and efficiently expands existing datasets by generating diverse rationales, providing a unified and extensible data foundation. Meanwhile, we design a Progressive Two-stage Correction Learning (P2CL) strategy. The first phase enhances multi-rationale positive learning, while the second phase enables active logic discrimination and correction. In addition, to mitigate representation entanglement in the multi-rationale semantic space, we propose a Multi-rationale Contrastive Alignment (MCA) optimization strategy, which achieves semantic aggregation of correct reasoning and boundary separation of incorrect reasoning. Extensive experiments demonstrate that the proposed MIND reasoning framework achieves state-of-the-art (SOTA) performance on multiple public datasets covering scientific, commonsense, and mathematical scenarios. It provides a new perspective for advancing MLLMs towards higher levels of cognitive intelligence. Our code is available at this https URL 

---
# BEAVER: An Efficient Deterministic LLM Verifier 

**Authors**: Tarun Suresh, Nalin Wadhwa, Debangshu Banerjee, Gagandeep Singh  

**Link**: [PDF](https://arxiv.org/pdf/2512.05439)  

**Abstract**: As large language models (LLMs) transition from research prototypes to production systems, practitioners often need reliable methods to verify that model outputs satisfy required constraints. While sampling-based estimates provide an intuition of model behavior, they offer no sound guarantees. We present BEAVER, the first practical framework for computing deterministic, sound probability bounds on LLM constraint satisfaction. Given any prefix-closed semantic constraint, BEAVER systematically explores the generation space using novel token trie and frontier data structures, maintaining provably sound bounds at every iteration. We formalize the verification problem, prove soundness of our approach, and evaluate BEAVER on correctness verification, privacy verification and secure code generation tasks across multiple state of the art LLMs. BEAVER achieves 6 to 8 times tighter probability bounds and identifies 3 to 4 times more high risk instances compared to baseline methods under identical computational budgets, enabling precise characterization and risk assessment that loose bounds or empirical evaluation cannot provide. 

---
# Using Large Language Models to Create Personalized Networks From Therapy Sessions 

**Authors**: Clarissa W. Ong, Hiba Arnaout, Kate Sheehan, Estella Fox, Eugen Owtscharow, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2512.05836)  

**Abstract**: Recent advances in psychotherapy have focused on treatment personalization, such as by selecting treatment modules based on personalized networks. However, estimating personalized networks typically requires intensive longitudinal data, which is not always feasible. A solution to facilitate scalability of network-driven treatment personalization is leveraging LLMs. In this study, we present an end-to-end pipeline for automatically generating client networks from 77 therapy transcripts to support case conceptualization and treatment planning. We annotated 3364 psychological processes and their corresponding dimensions in therapy transcripts. Using these data, we applied in-context learning to jointly identify psychological processes and their dimensions. The method achieved high performance even with a few training examples. To organize the processes into networks, we introduced a two-step method that grouped them into clinically meaningful clusters. We then generated explanation-augmented relationships between clusters. Experts found that networks produced by our multi-step approach outperformed those built with direct prompting for clinical utility and interpretability, with up to 90% preferring our approach. In addition, the networks were rated favorably by experts, with scores for clinical relevance, novelty, and usefulness ranging from 72-75%. Our findings provide a proof of concept for using LLMs to create clinically relevant networks from therapy transcripts. Advantages of our approach include bottom-up case conceptualization from client utterances in therapy sessions and identification of latent themes. Networks generated from our pipeline may be used in clinical settings and supervision and training. Future research should examine whether these networks improve treatment outcomes relative to other methods of treatment personalization, including statistically estimated networks. 

---
# Ontology Learning with LLMs: A Benchmark Study on Axiom Identification 

**Authors**: Roos M. Bakker, Daan L. Di Scala, Maaike H.T. de Boer, Stephan A. Raaijmakers  

**Link**: [PDF](https://arxiv.org/pdf/2512.05594)  

**Abstract**: Ontologies are an important tool for structuring domain knowledge, but their development is a complex task that requires significant modelling and domain expertise. Ontology learning, aimed at automating this process, has seen advancements in the past decade with the improvement of Natural Language Processing techniques, and especially with the recent growth of Large Language Models (LLMs). This paper investigates the challenge of identifying axioms: fundamental ontology components that define logical relations between classes and properties. In this work, we introduce an Ontology Axiom Benchmark OntoAxiom, and systematically test LLMs on that benchmark for axiom identification, evaluating different prompting strategies, ontologies, and axiom types. The benchmark consists of nine medium-sized ontologies with together 17.118 triples, and 2.771 axioms. We focus on subclass, disjoint, subproperty, domain, and range axioms. To evaluate LLM performance, we compare twelve LLMs with three shot settings and two prompting strategies: a Direct approach where we query all axioms at once, versus an Axiom-by-Axiom (AbA) approach, where each prompt queries for one axiom only. Our findings show that the AbA prompting leads to higher F1 scores than the direct approach. However, performance varies across axioms, suggesting that certain axioms are more challenging to identify. The domain also influences performance: the FOAF ontology achieves a score of 0.642 for the subclass axiom, while the music ontology reaches only 0.218. Larger LLMs outperform smaller ones, but smaller models may still be viable for resource-constrained settings. Although performance overall is not high enough to fully automate axiom identification, LLMs can provide valuable candidate axioms to support ontology engineers with the development and refinement of ontologies. 

---
# The Seeds of Scheming: Weakness of Will in the Building Blocks of Agentic Systems 

**Authors**: Robert Yang  

**Link**: [PDF](https://arxiv.org/pdf/2512.05449)  

**Abstract**: Large language models display a peculiar form of inconsistency: they "know" the correct answer but fail to act on it. In human philosophy, this tension between global judgment and local impulse is called akrasia, or weakness of will. We propose akrasia as a foundational concept for analyzing inconsistency and goal drift in agentic AI systems. To operationalize it, we introduce a preliminary version of the Akrasia Benchmark, currently a structured set of prompting conditions (Baseline [B], Synonym [S], Temporal [T], and Temptation [X]) that measures when a model's local response contradicts its own prior commitments. The benchmark enables quantitative comparison of "self-control" across model families, decoding strategies, and temptation types. Beyond single-model evaluation, we outline how micro-level akrasia may compound into macro-level instability in multi-agent systems that may be interpreted as "scheming" or deliberate misalignment. By reframing inconsistency as weakness of will, this work connects agentic behavior to classical theories of agency and provides an empirical bridge between philosophy, psychology, and the emerging science of agentic AI. 

---
# Documenting SME Processes with Conversational AI: From Tacit Knowledge to BPMN 

**Authors**: Unnikrishnan Radhakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2512.05122)  

**Abstract**: Small and medium-sized enterprises (SMEs) still depend heavily on tacit, experience-based know-how that rarely makes its way into formal documentation. This paper introduces a large-language-model (LLM)-driven conversational assistant that captures such knowledge on the shop floor and converts it incrementally and interactively into standards-compliant Business Process Model and Notation (BPMN) 2.0 diagrams. Powered by Gemini 2.5 Pro and delivered through a lightweight Gradio front-end with client-side bpmn-js visualisation, the assistant conducts an interview-style dialogue: it elicits process details, supports clarifying dialogue and on-demand analysis, and renders live diagrams that users can refine in real time. A proof-of-concept evaluation in an equipment-maintenance scenario shows that the chatbot produced an accurate "AS-IS" model, flagged issues via on-diagram annotations, and generated an improved "TO-BE" variant, all within about 12-minutes, while keeping API costs within an SME-friendly budget. The study analyses latency sources, model-selection trade-offs, and the challenges of enforcing strict XML schemas, then outlines a roadmap toward agentic and multimodal deployments. The results demonstrate that conversational LLMs can potentially be used to lower the skill and cost barriers to rigorous process documentation, helping SMEs preserve institutional knowledge, enhance operational transparency, and accelerate continuous-improvement efforts. 

---
# Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations 

**Authors**: Igor Halperin  

**Link**: [PDF](https://arxiv.org/pdf/2512.05156)  

**Abstract**: Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings. 

---
# Optimizing Medical Question-Answering Systems: A Comparative Study of Fine-Tuned and Zero-Shot Large Language Models with RAG Framework 

**Authors**: Tasnimul Hassan, Md Faisal Karim, Haziq Jeelani, Elham Behnam, Robert Green, Fayeq Jeelani Syed  

**Link**: [PDF](https://arxiv.org/pdf/2512.05863)  

**Abstract**: Medical question-answering (QA) systems can benefit from advances in large language models (LLMs), but directly applying LLMs to the clinical domain poses challenges such as maintaining factual accuracy and avoiding hallucinations. In this paper, we present a retrieval-augmented generation (RAG) based medical QA system that combines domain-specific knowledge retrieval with open-source LLMs to answer medical questions. We fine-tune two state-of-the-art open LLMs (LLaMA~2 and Falcon) using Low-Rank Adaptation (LoRA) for efficient domain specialization. The system retrieves relevant medical literature to ground the LLM's answers, thereby improving factual correctness and reducing hallucinations. We evaluate the approach on benchmark datasets (PubMedQA and MedMCQA) and show that retrieval augmentation yields measurable improvements in answer accuracy compared to using LLMs alone. Our fine-tuned LLaMA~2 model achieves 71.8% accuracy on PubMedQA, substantially improving over the 55.4% zero-shot baseline, while maintaining transparency by providing source references. We also detail the system design and fine-tuning methodology, demonstrating that grounding answers in retrieved evidence reduces unsupported content by approximately 60%. These results highlight the potential of RAG-augmented open-source LLMs for reliable biomedical QA, pointing toward practical clinical informatics applications. 

---
# Natural Language Summarization Enables Multi-Repository Bug Localization by LLMs in Microservice Architectures 

**Authors**: Amirkia Rafiei Oskooei, S. Selcan Yukcu, Mehmet Cevheri Bozoglan, Mehmet S. Aktas  

**Link**: [PDF](https://arxiv.org/pdf/2512.05908)  

**Abstract**: Bug localization in multi-repository microservice architectures is challenging due to the semantic gap between natural language bug reports and code, LLM context limitations, and the need to first identify the correct repository. We propose reframing this as a natural language reasoning task by transforming codebases into hierarchical NL summaries and performing NL-to-NL search instead of cross-modal retrieval. Our approach builds context-aware summaries at file, directory, and repository levels, then uses a two-phase search: first routing bug reports to relevant repositories, then performing top-down localization within those repositories. Evaluated on DNext, an industrial system with 46 repositories and 1.1M lines of code, our method achieves Pass@10 of 0.82 and MRR of 0.50, significantly outperforming retrieval baselines and agentic RAG systems like GitHub Copilot and Cursor. This work demonstrates that engineered natural language representations can be more effective than raw source code for scalable bug localization, providing an interpretable repository -> directory -> file search path, which is vital for building trust in enterprise AI tools by providing essential transparency. 

---
# Grounded Multilingual Medical Reasoning for Question Answering with Large Language Models 

**Authors**: Pietro Ferrazzi, Aitor Soroa, Rodrigo Agerri  

**Link**: [PDF](https://arxiv.org/pdf/2512.05658)  

**Abstract**: Large Language Models (LLMs) with reasoning capabilities have recently demonstrated strong potential in medical Question Answering (QA). Existing approaches are largely English-focused and primarily rely on distillation from general-purpose LLMs, raising concerns about the reliability of their medical knowledge. In this work, we present a method to generate multilingual reasoning traces grounded in factual medical knowledge. We produce 500k traces in English, Italian, and Spanish, using a retrievalaugmented generation approach over medical information from Wikipedia. The traces are generated to solve medical questions drawn from MedQA and MedMCQA, which we extend to Italian and Spanish. We test our pipeline in both in-domain and outof-domain settings across Medical QA benchmarks, and demonstrate that our reasoning traces improve performance both when utilized via in-context learning (few-shot) and supervised fine-tuning, yielding state-of-the-art results among 8B-parameter LLMs. We believe that these resources can support the development of safer, more transparent clinical decision-support tools in multilingual settings. We release the full suite of resources: reasoning traces, translated QA datasets, Medical-Wikipedia, and fine-tuned models. 

---
# Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding 

**Authors**: Ziyang Wang, Honglu Zhou, Shijie Wang, Junnan Li, Caiming Xiong, Silvio Savarese, Mohit Bansal, Michael S. Ryoo, Juan Carlos Niebles  

**Link**: [PDF](https://arxiv.org/pdf/2512.05774)  

**Abstract**: Long video understanding (LVU) is challenging because answering real-world queries often depends on sparse, temporally dispersed cues buried in hours of mostly redundant and irrelevant content. While agentic pipelines improve video reasoning capabilities, prevailing frameworks rely on a query-agnostic captioner to perceive video information, which wastes computation on irrelevant content and blurs fine-grained temporal and spatial information. Motivated by active perception theory, we argue that LVU agents should actively decide what, when, and where to observe, and continuously assess whether the current observation is sufficient to answer the query. We present Active Video Perception (AVP), an evidence-seeking framework that treats the video as an interactive environment and acquires compact, queryrelevant evidence directly from pixels. Concretely, AVP runs an iterative plan-observe-reflect process with MLLM agents. In each round, a planner proposes targeted video interactions, an observer executes them to extract time-stamped evidence, and a reflector evaluates the sufficiency of the evidence for the query, either halting with an answer or triggering further observation. Across five LVU benchmarks, AVP achieves highest performance with significant improvements. Notably, AVP outperforms the best agentic method by 5.7% in average accuracy while only requires 18.4% inference time and 12.4% input tokens. 

---
# RoBoN: Routed Online Best-of-n for Test-Time Scaling with Multiple LLMs 

**Authors**: Jonathan Geuter, Gregor Kornhardt  

**Link**: [PDF](https://arxiv.org/pdf/2512.05542)  

**Abstract**: Best-of-$n$ is a widely used test-time scaling approach for LLM inference. Yet despite evidence that LLMs exhibit complementary strengths across tasks, traditionally best-of-$n$ relies on a single model to generate responses. We propose RoBoN (Routed Online Best-of-$n$), a sequential multi-LLM alternative to the prevailing single-model best-of-$n$. Given a suite of models $\{m_i\}_{i=1}^M$, RoBoN sequentially routes generations one-by-one across models, based on scores computed using a reward model and an agreement signal on the predicted responses. This online routing requires no additional training, keeps compute parity, and works with any plug-in reward model. Across reasoning benchmarks (MATH500, OlympiadBench, MinervaMath, GSM8K, MMLU), RoBoN consistently outperforms standard best-of-$n$ applied to each individual model for larger $n$, with gains of up to 3.4\% in absolute accuracy, and also improves over a uniform multi-model portfolio baseline. Our results indicate that diversity across models can be exploited at inference to improve best-of-$n$ performance over any constituent model alone, providing a simple, training-free path to test-time scaling with multiple LLMs. 

---
# Faithfulness metric fusion: Improving the evaluation of LLM trustworthiness across domains 

**Authors**: Ben Malin, Tatiana Kalganova, Nikolaos Boulgouris  

**Link**: [PDF](https://arxiv.org/pdf/2512.05700)  

**Abstract**: We present a methodology for improving the accuracy of faithfulness evaluation in Large Language Models (LLMs). The proposed methodology is based on the combination of elementary faithfulness metrics into a combined (fused) metric, for the purpose of improving the faithfulness of LLM outputs. The proposed strategy for metric fusion deploys a tree-based model to identify the importance of each metric, which is driven by the integration of human judgements evaluating the faithfulness of LLM responses. This fused metric is demonstrated to correlate more strongly with human judgements across all tested domains for faithfulness. Improving the ability to evaluate the faithfulness of LLMs, allows for greater confidence to be placed within models, allowing for their implementation in a greater diversity of scenarios. Additionally, we homogenise a collection of datasets across question answering and dialogue-based domains and implement human judgements and LLM responses within this dataset, allowing for the reproduction and trialling of faithfulness evaluation across domains. 

---
# Knowing Your Uncertainty -- On the application of LLM in social sciences 

**Authors**: Bolun Zhang, Linzhuo Li, Yunqi Chen, Qinlin Zhao, Zihan Zhu, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2512.05461)  

**Abstract**: Large language models (LLMs) are rapidly being integrated into computational social science research, yet their blackboxed training and designed stochastic elements in inference pose unique challenges for scientific inquiry. This article argues that applying LLMs to social scientific tasks requires explicit assessment of uncertainty-an expectation long established in both quantitative methodology in the social sciences and machine learning. We introduce a unified framework for evaluating LLM uncertainty along two dimensions: the task type (T), which distinguishes between classification, short-form, and long-form generation, and the validation type (V), which captures the availability of reference data or evaluative criteria. Drawing from both computer science and social science literature, we map existing uncertainty quantification (UQ) methods to this T-V typology and offer practical recommendations for researchers. Our framework provides both a methodological safeguard and a practical guide for integrating LLMs into rigorous social science research. 

---
# Lyrics Matter: Exploiting the Power of Learnt Representations for Music Popularity Prediction 

**Authors**: Yash Choudhary, Preeti Rao, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2512.05508)  

**Abstract**: Accurately predicting music popularity is a critical challenge in the music industry, offering benefits to artists, producers, and streaming platforms. Prior research has largely focused on audio features, social metadata, or model architectures. This work addresses the under-explored role of lyrics in predicting popularity. We present an automated pipeline that uses LLM to extract high-dimensional lyric embeddings, capturing semantic, syntactic, and sequential information. These features are integrated into HitMusicLyricNet, a multimodal architecture that combines audio, lyrics, and social metadata for popularity score prediction in the range 0-100. Our method outperforms existing baselines on the SpotGenTrack dataset, which contains over 100,000 tracks, achieving 9% and 20% improvements in MAE and MSE, respectively. Ablation confirms that gains arise from our LLM-driven lyrics feature pipeline (LyricsAENet), underscoring the value of dense lyric representations. 

---
# Dynamic Alignment for Collective Agency: Toward a Scalable Self-Improving Framework for Open-Ended LLM Alignment 

**Authors**: Panatchakorn Anantaprayoon, Nataliia Babina, Jad Tarifi, Nima Asgharbeygi  

**Link**: [PDF](https://arxiv.org/pdf/2512.05464)  

**Abstract**: Large Language Models (LLMs) are typically aligned with human values using preference data or predefined principles such as helpfulness, honesty, and harmlessness. However, as AI systems progress toward Artificial General Intelligence (AGI) and Artificial Superintelligence (ASI), such value systems may become insufficient. In addition, human feedback-based alignment remains resource-intensive and difficult to scale. While AI-feedback-based self-improving alignment methods have been explored as a scalable alternative, they have largely remained constrained to conventional alignment values. In this work, we explore both a more holistic alignment objective and a scalable, self-improving alignment approach. Aiming to transcend conventional alignment norms, we introduce Collective Agency (CA)-a unified and open-ended alignment value that encourages integrated agentic capabilities. We also propose Dynamic Alignment-an alignment framework that enables an LLM to iteratively align itself. Dynamic Alignment comprises two key components: (1) automated training dataset generation with LLMs, and (2) a self-rewarding mechanism, where the policy model evaluates its own output candidates and assigns rewards for GRPO-based learning. Experimental results demonstrate that our approach successfully aligns the model to CA while preserving general NLP capabilities. 

---
# A Systematic Framework for Enterprise Knowledge Retrieval: Leveraging LLM-Generated Metadata to Enhance RAG Systems 

**Authors**: Pranav Pushkar Mishra, Kranti Prakash Yeole, Ramyashree Keshavamurthy, Mokshit Bharat Surana, Fatemeh Sarayloo  

**Link**: [PDF](https://arxiv.org/pdf/2512.05411)  

**Abstract**: In enterprise settings, efficiently retrieving relevant information from large and complex knowledge bases is essential for operational productivity and informed decision-making. This research presents a systematic framework for metadata enrichment using large language models (LLMs) to enhance document retrieval in Retrieval-Augmented Generation (RAG) systems. Our approach employs a comprehensive, structured pipeline that dynamically generates meaningful metadata for document segments, substantially improving their semantic representations and retrieval accuracy. Through extensive experiments, we compare three chunking strategies-semantic, recursive, and naive-and evaluate their effectiveness when combined with advanced embedding techniques. The results demonstrate that metadata-enriched approaches consistently outperform content-only baselines, with recursive chunking paired with TF-IDF weighted embeddings yielding an 82.5% precision rate compared to 73.3% for semantic content-only approaches. The naive chunking strategy with prefix-fusion achieved the highest Hit Rate@10 of 0.925. Our evaluation employs cross-encoder reranking for ground truth generation, enabling rigorous assessment via Hit Rate and Metadata Consistency metrics. These findings confirm that metadata enrichment enhances vector clustering quality while reducing retrieval latency, making it a key optimization for RAG systems across knowledge domains. This work offers practical insights for deploying high-performance, scalable document retrieval solutions in enterprise settings, demonstrating that metadata enrichment is a powerful approach for enhancing RAG effectiveness. 

---
# The Effect of Document Summarization on LLM-Based Relevance Judgments 

**Authors**: Samaneh Mohtadi, Kevin Roitero, Stefano Mizzaro, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2512.05334)  

**Abstract**: Relevance judgments are central to the evaluation of Information Retrieval (IR) systems, but obtaining them from human annotators is costly and time-consuming. Large Language Models (LLMs) have recently been proposed as automated assessors, showing promising alignment with human annotations. Most prior studies have treated documents as fixed units, feeding their full content directly to LLM assessors. We investigate how text summarization affects the reliability of LLM-based judgments and their downstream impact on IR evaluation. Using state-of-the-art LLMs across multiple TREC collections, we compare judgments made from full documents with those based on LLM-generated summaries of different lengths. We examine their agreement with human labels, their effect on retrieval effectiveness evaluation, and their influence on IR systems' ranking stability. Our findings show that summary-based judgments achieve comparable stability in systems' ranking to full-document judgments, while introducing systematic shifts in label distributions and biases that vary by model and dataset. These results highlight summarization as both an opportunity for more efficient large-scale IR evaluation and a methodological choice with important implications for the reliability of automatic judgments. 

---
# To Think or Not to Think: The Hidden Cost of Meta-Training with Excessive CoT Examples 

**Authors**: Vignesh Kothapalli, Ata Fatahibaarzi, Hamed Firooz, Maziar Sanjabi  

**Link**: [PDF](https://arxiv.org/pdf/2512.05318)  

**Abstract**: Chain-of-thought (CoT) prompting combined with few-shot in-context learning (ICL) has unlocked significant reasoning capabilities in large language models (LLMs). However, ICL with CoT examples is ineffective on novel tasks when the pre-training knowledge is insufficient. We study this problem in a controlled setting using the CoT-ICL Lab framework, and propose meta-training techniques to learn novel abstract reasoning tasks in-context. Although CoT examples facilitate reasoning, we noticed that their excessive inclusion during meta-training degrades performance when CoT supervision is limited. To mitigate such behavior, we propose CoT-Recipe, a formal approach to modulate the mix of CoT and non-CoT examples in meta-training sequences. We demonstrate that careful modulation via CoT-Recipe can increase the accuracy of transformers on novel tasks by up to 300% even when there are no CoT examples available in-context. We confirm the broader effectiveness of these techniques by applying them to pretrained LLMs (Qwen2.5 series) for symbolic reasoning tasks and observing gains of up to 130% in accuracy. 

---
# The Erosion of LLM Signatures: Can We Still Distinguish Human and LLM-Generated Scientific Ideas After Iterative Paraphrasing? 

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2512.05311)  

**Abstract**: With the increasing reliance on LLMs as research agents, distinguishing between LLM and human-generated ideas has become crucial for understanding the cognitive nuances of LLMs' research capabilities. While detecting LLM-generated text has been extensively studied, distinguishing human vs LLM-generated scientific idea remains an unexplored area. In this work, we systematically evaluate the ability of state-of-the-art (SOTA) machine learning models to differentiate between human and LLM-generated ideas, particularly after successive paraphrasing stages. Our findings highlight the challenges SOTA models face in source attribution, with detection performance declining by an average of 25.4\% after five consecutive paraphrasing stages. Additionally, we demonstrate that incorporating the research problem as contextual information improves detection performance by up to 2.97%. Notably, our analysis reveals that detection algorithms struggle significantly when ideas are paraphrased into a simplified, non-expert style, contributing the most to the erosion of distinguishable LLM signatures. 

---
# Learning to Code with Context: A Study-Based Approach 

**Authors**: Uwe M. Borghoff, Mark Minas, Jannis Schopp  

**Link**: [PDF](https://arxiv.org/pdf/2512.05242)  

**Abstract**: The rapid emergence of generative AI tools is transforming the way software is developed. Consequently, software engineering education must adapt to ensure that students not only learn traditional development methods but also understand how to meaningfully and responsibly use these new technologies. In particular, project-based courses offer an effective environment to explore and evaluate the integration of AI assistance into real-world development practices. This paper presents our approach and a user study conducted within a university programming project in which students collaboratively developed computer games. The study investigates how participants used generative AI tools throughout different phases of the software development process, identifies the types of tasks where such tools were most effective, and analyzes the challenges students encountered. Building on these insights, we further examine a repository-aware, locally deployed large language model (LLM) assistant designed to provide project-contextualized support. The system employs Retrieval-Augmented Generation (RAG) to ground responses in relevant documentation and source code, enabling qualitative analysis of model behavior, parameter sensitivity, and common failure modes. The findings deepen our understanding of context-aware AI support in educational software projects and inform future integration of AI-based assistance into software engineering curricula. 

---
# XR-DT: Extended Reality-Enhanced Digital Twin for Agentic Mobile Robots 

**Authors**: Tianyi Wang, Jiseop Byeon, Ahmad Yehia, Huihai Wang, Yiming Xu, Tianyi Zeng, Ziran Wang, Junfeng Jiao, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2512.05270)  

**Abstract**: As mobile robots increasingly operate alongside humans in shared workspaces, ensuring safe, efficient, and interpretable Human-Robot Interaction (HRI) has become a pressing challenge. While substantial progress has been devoted to human behavior prediction, limited attention has been paid to how humans perceive, interpret, and trust robots' inferences, impeding deployment in safety-critical and socially embedded environments. This paper presents XR-DT, an eXtended Reality-enhanced Digital Twin framework for agentic mobile robots, that bridges physical and virtual spaces to enable bi-directional understanding between humans and robots. Our hierarchical XR-DT architecture integrates virtual-, augmented-, and mixed-reality layers, fusing real-time sensor data, simulated environments in the Unity game engine, and human feedback captured through wearable AR devices. Within this framework, we design an agentic mobile robot system with a unified diffusion policy for context-aware task adaptation. We further propose a chain-of-thought prompting mechanism that allows multimodal large language models to reason over human instructions and environmental context, while leveraging an AutoGen-based multi-agent coordination layer to enhance robustness and collaboration in dynamic tasks. Initial experimental results demonstrate accurate human and robot trajectory prediction, validating the XR-DT framework's effectiveness in HRI tasks. By embedding human intention, environmental dynamics, and robot cognition into the XR-DT framework, our system enables interpretable, trustworthy, and adaptive HRI. 

---
# MedTutor-R1: Socratic Personalized Medical Teaching with Multi-Agent Simulation 

**Authors**: Zhitao He, Haolin Yang, Zeyu Qin, Yi R Fung  

**Link**: [PDF](https://arxiv.org/pdf/2512.05671)  

**Abstract**: The significant gap between rising demands for clinical training and the scarcity of expert instruction poses a major challenge to medical education. With powerful capabilities in personalized guidance, Large Language Models (LLMs) offer a promising solution to bridge this gap. However, current research focuses mainly on one-on-one knowledge instruction, overlooking collaborative reasoning, a key skill for students developed in teamwork like ward rounds. To this end, we develop ClinEdu, a multi-agent pedagogical simulator with personality-driven patients and diverse student cohorts, enabling controlled testing of complex pedagogical processes and scalable generation of teaching data. Based on ClinEdu, we construct ClinTeach, a large Socratic teaching dialogue dataset that captures the complexities of group instruction. We then train MedTutor-R1, the first multimodal Socratic tutor designed for one-to-many instruction in clinical medical education. MedTutor-R1 is first instruction-tuned on our ClinTeach dataset and then optimized with reinforcement learning, using rewards derived from a three-axis rubric, covering structural fidelity, analytical quality, and clinical safety, to refine its adaptive Socratic strategies. For authentic in-situ assessment, we use simulation-based interactive evaluation that redeploys the tutor back into ClinEdu. Experimental results demonstrate that our MedTutor-R1 outperforms the base model by over 20% in average pedagogical score and is comparable to o3, while also exhibiting high adaptability in handling a varying number of students. This promising performance underscores the effectiveness of our pedagogical simulator, ClinEdu. 

---
# Interleaved Latent Visual Reasoning with Selective Perceptual Modeling 

**Authors**: Shuai Dong, Siyuan Wang, Xingyu Liu, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2512.05665)  

**Abstract**: Interleaved reasoning paradigms enhance Multimodal Large Language Models (MLLMs) with visual feedback but are hindered by the prohibitive computational cost of repeatedly re-encoding pixel-dense images. A promising alternative, latent visual reasoning, circumvents this bottleneck yet currently forces a critical trade-off: methods either sacrifice precise perceptual modeling by over-compressing features or fail to model dynamic problems due to static, non-interleaved structures. We introduce Interleaved Latent Visual Reasoning (ILVR), a framework that unifies dynamic state evolution with precise perceptual modeling. ILVR interleaves textual generation with latent visual representations that act as specific, evolving cues for subsequent reasoning. To enable this, we employ a self-supervision strategy where a Momentum Teacher Model selectively distills relevant features from helper images into sparse supervision targets. This adaptive selection mechanism guides the model to autonomously generate context-aware visual signals. Extensive experiments on multimodal reasoning benchmarks demonstrate that ILVR significantly outperforms existing approaches, effectively bridging the gap between fine-grained perception and sequential multimodal reasoning. 

---
# Automated Identification of Incidentalomas Requiring Follow-Up: A Multi-Anatomy Evaluation of LLM-Based and Supervised Approaches 

**Authors**: Namu Park, Farzad Ahmed, Zhaoyi Sun, Kevin Lybarger, Ethan Breinhorst, Julie Hu, Ozlem Uzuner, Martin Gunn, Meliha Yetisgen  

**Link**: [PDF](https://arxiv.org/pdf/2512.05537)  

**Abstract**: Objective: To evaluate large language models (LLMs) against supervised baselines for fine-grained, lesion-level detection of incidentalomas requiring follow-up, addressing the limitations of current document-level classification systems.
Methods: We utilized a dataset of 400 annotated radiology reports containing 1,623 verified lesion findings. We compared three supervised transformer-based encoders (BioClinicalModernBERT, ModernBERT, Clinical Longformer) against four generative LLM configurations (Llama 3.1-8B, GPT-4o, GPT-OSS-20b). We introduced a novel inference strategy using lesion-tagged inputs and anatomy-aware prompting to ground model reasoning. Performance was evaluated using class-specific F1-scores.
Results: The anatomy-informed GPT-OSS-20b model achieved the highest performance, yielding an incidentaloma-positive macro-F1 of 0.79. This surpassed all supervised baselines (maximum macro-F1: 0.70) and closely matched the inter-annotator agreement of 0.76. Explicit anatomical grounding yielded statistically significant performance gains across GPT-based models (p < 0.05), while a majority-vote ensemble of the top systems further improved the macro-F1 to 0.90. Error analysis revealed that anatomy-aware LLMs demonstrated superior contextual reasoning in distinguishing actionable findings from benign lesions.
Conclusion: Generative LLMs, when enhanced with structured lesion tagging and anatomical context, significantly outperform traditional supervised encoders and achieve performance comparable to human experts. This approach offers a reliable, interpretable pathway for automated incidental finding surveillance in radiology workflows. 

---
# SEA-SafeguardBench: Evaluating AI Safety in SEA Languages and Cultures 

**Authors**: Panuthep Tasawong, Jian Gang Ngui, Alham Fikri Aji, Trevor Cohn, Peerat Limkonchotiwat  

**Link**: [PDF](https://arxiv.org/pdf/2512.05501)  

**Abstract**: Safeguard models help large language models (LLMs) detect and block harmful content, but most evaluations remain English-centric and overlook linguistic and cultural diversity. Existing multilingual safety benchmarks often rely on machine-translated English data, which fails to capture nuances in low-resource languages. Southeast Asian (SEA) languages are underrepresented despite the region's linguistic diversity and unique safety concerns, from culturally sensitive political speech to region-specific misinformation. Addressing these gaps requires benchmarks that are natively authored to reflect local norms and harm scenarios. We introduce SEA-SafeguardBench, the first human-verified safety benchmark for SEA, covering eight languages, 21,640 samples, across three subsets: general, in-the-wild, and content generation. The experimental results from our benchmark demonstrate that even state-of-the-art LLMs and guardrails are challenged by SEA cultural and harm scenarios and underperform when compared to English texts. 

---
# Structured Reasoning with Tree-of-Thoughts for Bengali Math Word Problems 

**Authors**: Aurprita Mahmood, Sabrin alam, Neloy kumer Sagor, Md. Abdul Hadi, Md. Sehab Al Islam, Minhajul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2512.05580)  

**Abstract**: Mathematical Word Problems (MWPs) are among the most challenging tasks in natural language processing because they require both linguistic understanding and multi-step numerical reasoning. While Chain-of-Thought (CoT) prompting has shown promise, its linear structure often propagates errors, limiting overall effectiveness. To address this limitation, we present the a systematic study of Tree-of-Thought (ToT) reasoning for Bengali MWPs using the SOMADHAN dataset. Owing to computational and token-cost constraints, we evaluate a curated set of 100 representative problems across multiple large language models (LLMs), including GPT-OSS and LLaMA variants, under standard prompting, CoT, and ToT strategies. Our results show that CoT improves baseline accuracy from 78% (standard prompting) to 83% on average, while ToT further increases performance by up to 5 percentage points, achieving 88% accuracy with GPT-OSS-120B. These improvements highlight that ToT is particularly effective in medium-to-large-scale models but may offer less advantage for smaller ones. Overall, our findings establish ToT as a robust framework for solving mathematical problems in low-resource languages such as Bengali. More broadly, this study shows that structured reasoning methods like ToT can provide more reliable and globally consistent outcomes than CoT, paving the way for better reasoning strategies in multilingual NLP. 

---
# Learning from Self Critique and Refinement for Faithful LLM Summarization 

**Authors**: Ting-Yao Hu, Hema Swetha Koppula, Hadi Pouransari, Cem Koc, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2512.05387)  

**Abstract**: Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries. 

---
# Enhancing Clinical Note Generation with ICD-10, Clinical Ontology Knowledge Graphs, and Chain-of-Thought Prompting Using GPT-4 

**Authors**: Ivan Makohon, Mohamad Najafi, Jian Wu, Mathias Brochhausen, Yaohang Li  

**Link**: [PDF](https://arxiv.org/pdf/2512.05256)  

**Abstract**: In the past decade a surge in the amount of electronic health record (EHR) data in the United States, attributed to a favorable policy environment created by the Health Information Technology for Economic and Clinical Health (HITECH) Act of 2009 and the 21st Century Cures Act of 2016. Clinical notes for patients' assessments, diagnoses, and treatments are captured in these EHRs in free-form text by physicians, who spend a considerable amount of time entering and editing them. Manually writing clinical notes takes a considerable amount of a doctor's valuable time, increasing the patient's waiting time and possibly delaying diagnoses. Large language models (LLMs) possess the ability to generate news articles that closely resemble human-written ones. We investigate the usage of Chain-of-Thought (CoT) prompt engineering to improve the LLM's response in clinical note generation. In our prompts, we use as input International Classification of Diseases (ICD) codes and basic patient information. We investigate a strategy that combines the traditional CoT with semantic search results to improve the quality of generated clinical notes. Additionally, we infuse a knowledge graph (KG) built from clinical ontology to further enrich the domain-specific knowledge of generated clinical notes. We test our prompting technique on six clinical cases from the CodiEsp test dataset using GPT-4 and our results show that it outperformed the clinical notes generated by standard one-shot prompts. 

---
# Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats 

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi  

**Link**: [PDF](https://arxiv.org/pdf/2512.05331)  

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%. 

---
