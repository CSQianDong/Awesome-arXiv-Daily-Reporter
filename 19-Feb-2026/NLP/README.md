# Reinforced Fast Weights with Next-Sequence Prediction 

**Authors**: Hee Seung Hwang, Xindi Wu, Sanghyuk Chun, Olga Russakovsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.16704)  

**Abstract**: Fast weight architectures offer a promising alternative to attention-based transformers for long-context modeling by maintaining constant memory overhead regardless of context length. However, their potential is limited by the next-token prediction (NTP) training paradigm. NTP optimizes single-token predictions and ignores semantic coherence across multiple tokens following a prefix. Consequently, fast weight models, which dynamically update their parameters to store contextual information, learn suboptimal representations that fail to capture long-range dependencies. We introduce REFINE (Reinforced Fast weIghts with Next sEquence prediction), a reinforcement learning framework that trains fast weight models under the next-sequence prediction (NSP) objective. REFINE selects informative token positions based on prediction entropy, generates multi-token rollouts, assigns self-supervised sequence-level rewards, and optimizes the model with group relative policy optimization (GRPO). REFINE is applicable throughout the training lifecycle of pre-trained language models: mid-training, post-training, and test-time training. Our experiments on LaCT-760M and DeltaNet-1.3B demonstrate that REFINE consistently outperforms supervised fine-tuning with NTP across needle-in-a-haystack retrieval, long-context question answering, and diverse tasks in LongBench. REFINE provides an effective and versatile framework for improving long-context modeling in fast weight architectures. 

---
# Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents 

**Authors**: Wenxuan Ding, Nicholas Tomlin, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2602.16699)  

**Abstract**: LLMs are increasingly being used for complex problems which are not necessarily resolved in a single response, but require interacting with an environment to acquire information. In these scenarios, LLMs must reason about inherent cost-uncertainty tradeoffs in when to stop exploring and commit to an answer. For instance, on a programming task, an LLM should test a generated code snippet if it is uncertain about the correctness of that code; the cost of writing a test is nonzero, but typically lower than the cost of making a mistake. In this work, we show that we can induce LLMs to explicitly reason about balancing these cost-uncertainty tradeoffs, then perform more optimal environment exploration. We formalize multiple tasks, including information retrieval and coding, as sequential decision-making problems under uncertainty. Each problem has latent environment state that can be reasoned about via a prior which is passed to the LLM agent. We introduce a framework called Calibrate-Then-Act (CTA), where we feed the LLM this additional context to enable it to act more optimally. This improvement is preserved even under RL training of both the baseline and CTA. Our results on information-seeking QA and on a simplified coding task show that making cost-benefit tradeoffs explicit with CTA can help agents discover more optimal decision-making strategies. 

---
# Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment 

**Authors**: Yuyan Bu, Xiaohao Liu, ZhaoXing Ren, Yaodong Yang, Juntao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2602.16660)  

**Abstract**: The widespread deployment of large language models (LLMs) across linguistic communities necessitates reliable multilingual safety alignment. However, recent efforts to extend alignment to other languages often require substantial resources, either through large-scale, high-quality supervision in the target language or through pairwise alignment with high-resource languages, which limits scalability. In this work, we propose a resource-efficient method for improving multilingual safety alignment. We introduce a plug-and-play Multi-Lingual Consistency (MLC) loss that can be integrated into existing monolingual alignment pipelines. By improving collinearity between multilingual representation vectors, our method encourages directional consistency at the multilingual semantic level in a single update. This allows simultaneous alignment across multiple languages using only multilingual prompt variants without requiring additional response-level supervision in low-resource languages. We validate the proposed method across different model architectures and alignment paradigms, and demonstrate its effectiveness in enhancing multilingual safety with limited impact on general model utility. Further evaluation across languages and tasks indicates improved cross-lingual generalization, suggesting the proposed approach as a practical solution for multilingual consistency alignment under limited supervision. 

---
# Quecto-V1: Empirical Analysis of 8-bit Quantized Small Language Models for On-Device Legal Retrieval 

**Authors**: Subrit Dikshit  

**Link**: [PDF](https://arxiv.org/pdf/2602.16640)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has revolutionized Natural Language Processing (NLP) but has simultaneously created a "resource divide." State-of-the-art legal intelligence systems typically rely on massive parameter counts (7B+) and cloud-based inference, rendering them inaccessible to practitioners in resource-constrained environments and posing significant data sovereignty risks. This paper introduces Quecto-V1, a domain-specific Small Language Model (SLM) engineered to democratize access to Indian legal intelligence. Built upon a custom configuration of the GPT-2 architecture (124 million parameters), Quecto-V1 was trained from scratch exclusively on a corpus of Indian statutes, including the Indian Penal Code (IPC), the Code of Criminal Procedure (CrPC), and the Constitution of India. Unlike generalist models, which prioritize broad world knowledge, our approach maximizes "lexical density" within the legal domain. Furthermore, we address the deployment bottleneck by applying post-training 8-bit quantization (GGUF format), compressing the model to a memory footprint of under 150 MB. Our empirical analysis demonstrates that Quecto-V1 achieves high fidelity in retrieving statutory definitions and penal provisions, outperforming general-purpose SLMs in domain-specific exact match tasks while running entirely offline on consumer-grade CPUs. We further present an ablation study showing that 8-bit quantization yields a 74% reduction in model size with less than 3.5% degradation in retrieval accuracy compared to full-precision baselines. These findings suggest that for specialized, high-stakes domains like law, domain-specific training coupled with aggressive quantization offers a viable, privacy-preserving alternative to monolithic cloud models. 

---
# AREG: Adversarial Resource Extraction Game for Evaluating Persuasion and Resistance in Large Language Models 

**Authors**: Adib Sakhawat, Fardeen Sadab  

**Link**: [PDF](https://arxiv.org/pdf/2602.16639)  

**Abstract**: Evaluating the social intelligence of Large Language Models (LLMs) increasingly requires moving beyond static text generation toward dynamic, adversarial interaction. We introduce the Adversarial Resource Extraction Game (AREG), a benchmark that operationalizes persuasion and resistance as a multi-turn, zero-sum negotiation over financial resources. Using a round-robin tournament across frontier models, AREG enables joint evaluation of offensive (persuasion) and defensive (resistance) capabilities within a single interactional framework. Our analysis provides evidence that these capabilities are weakly correlated ($\rho = 0.33$) and empirically dissociated: strong persuasive performance does not reliably predict strong resistance, and vice versa. Across all evaluated models, resistance scores exceed persuasion scores, indicating a systematic defensive advantage in adversarial dialogue settings. Further linguistic analysis suggests that interaction structure plays a central role in these outcomes. Incremental commitment-seeking strategies are associated with higher extraction success, while verification-seeking responses are more prevalent in successful defenses than explicit refusal. Together, these findings indicate that social influence in LLMs is not a monolithic capability and that evaluation frameworks focusing on persuasion alone may overlook asymmetric behavioral vulnerabilities. 

---
# Who can we trust? LLM-as-a-jury for Comparative Assessment 

**Authors**: Mengjie Qian, Guangzhi Sun, Mark J.F. Gales, Kate M. Knill  

**Link**: [PDF](https://arxiv.org/pdf/2602.16610)  

**Abstract**: Large language models (LLMs) are increasingly applied as automatic evaluators for natural language generation assessment often using pairwise comparative judgements. Existing approaches typically rely on single judges or aggregate multiple judges assuming equal reliability. In practice, LLM judges vary substantially in performance across tasks and aspects, and their judgment probabilities may be biased and inconsistent. Furthermore, human-labelled supervision for judge calibration may be unavailable. We first empirically demonstrate that inconsistencies in LLM comparison probabilities exist and show that it limits the effectiveness of direct probability-based ranking. To address this, we study the LLM-as-a-jury setting and propose BT-sigma, a judge-aware extension of the Bradley-Terry model that introduces a discriminator parameter for each judge to jointly infer item rankings and judge reliability from pairwise comparisons alone. Experiments on benchmark NLG evaluation datasets show that BT-sigma consistently outperforms averaging-based aggregation methods, and that the learned discriminator strongly correlates with independent measures of the cycle consistency of LLM judgments. Further analysis reveals that BT-sigma can be interpreted as an unsupervised calibration mechanism that improves aggregation by modelling judge reliability. 

---
# ColBERT-Zero: To Pre-train Or Not To Pre-train ColBERT models 

**Authors**: Antoine Chaffin, Luca Arnaboldi, Amélie Chatelain, Florent Krzakala  

**Link**: [PDF](https://arxiv.org/pdf/2602.16609)  

**Abstract**: Current state-of-the-art multi-vector models are obtained through a small Knowledge Distillation (KD) training step on top of strong single-vector models, leveraging the large-scale pre-training of these models. In this paper, we study the pre-training of multi-vector models and show that large-scale multi-vector pre-training yields much stronger multi-vector models. Notably, a fully ColBERT-pre-trained model, ColBERT-Zero, trained only on public data, outperforms GTE-ModernColBERT as well as its base model, GTE-ModernBERT, which leverages closed and much stronger data, setting new state-of-the-art for model this size. We also find that, although performing only a small KD step is not enough to achieve results close to full pre-training, adding a supervised step beforehand allows to achieve much closer performance while skipping the most costly unsupervised phase. Finally, we find that aligning the fine-tuning and pre-training setups is crucial when repurposing existing models. To enable exploration of our results, we release various checkpoints as well as code used to train them. 

---
# Explainable AI: Context-Aware Layer-Wise Integrated Gradients for Explaining Transformer Models 

**Authors**: Melkamu Abay Mersha, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2602.16608)  

**Abstract**: Transformer models achieve state-of-the-art performance across domains and tasks, yet their deeply layered representations make their predictions difficult to interpret. Existing explainability methods rely on final-layer attributions, capture either local token-level attributions or global attention patterns without unification, and lack context-awareness of inter-token dependencies and structural components. They also fail to capture how relevance evolves across layers and how structural components shape decision-making. To address these limitations, we proposed the \textbf{Context-Aware Layer-wise Integrated Gradients (CA-LIG) Framework}, a unified hierarchical attribution framework that computes layer-wise Integrated Gradients within each Transformer block and fuses these token-level attributions with class-specific attention gradients. This integration yields signed, context-sensitive attribution maps that capture supportive and opposing evidence while tracing the hierarchical flow of relevance through the Transformer layers. We evaluate the CA-LIG Framework across diverse tasks, domains, and transformer model families, including sentiment analysis and long and multi-class document classification with BERT, hate speech detection in a low-resource language setting with XLM-R and AfroLM, and image classification with Masked Autoencoder vision Transformer model. Across all tasks and architectures, CA-LIG provides more faithful attributions, shows stronger sensitivity to contextual dependencies, and produces clearer, more semantically coherent visualizations than established explainability methods. These results indicate that CA-LIG provides a more comprehensive, context-aware, and reliable explanation of Transformer decision-making, advancing both the practical interpretability and conceptual understanding of deep neural models. 

---
# CitiLink-Summ: Summarization of Discussion Subjects in European Portuguese Municipal Meeting Minutes 

**Authors**: Miguel Marques, Ana Luísa Fernandes, Ana Filipa Pacheco, Rute Rebouças, Inês Cantante, José Isidro, Luís Filipe Cunha, Alípio Jorge, Nuno Guimarães, Sérgio Nunes, António Leal, Purificação Silvano, Ricardo Campos  

**Link**: [PDF](https://arxiv.org/pdf/2602.16607)  

**Abstract**: Municipal meeting minutes are formal records documenting the discussions and decisions of local government, yet their content is often lengthy, dense, and difficult for citizens to navigate. Automatic summarization can help address this challenge by producing concise summaries for each discussion subject. Despite its potential, research on summarizing discussion subjects in municipal meeting minutes remains largely unexplored, especially in low-resource languages, where the inherent complexity of these documents adds further challenges. A major bottleneck is the scarcity of datasets containing high-quality, manually crafted summaries, which limits the development and evaluation of effective summarization models for this domain. In this paper, we present CitiLink-Summ, a new corpus of European Portuguese municipal meeting minutes, comprising 100 documents and 2,322 manually hand-written summaries, each corresponding to a distinct discussion subject. Leveraging this dataset, we establish baseline results for automatic summarization in this domain, employing state-of-the-art generative models (e.g., BART, PRIMERA) as well as large language models (LLMs), evaluated with both lexical and semantic metrics such as ROUGE, BLEU, METEOR, and BERTScore. CitiLink-Summ provides the first benchmark for municipal-domain summarization in European Portuguese, offering a valuable resource for advancing NLP research on complex administrative texts. 

---
# Utility-Preserving De-Identification for Math Tutoring: Investigating Numeric Ambiguity in the MathEd-PII Benchmark Dataset 

**Authors**: Zhuqian Zhou, Kirk Vanacore, Bakhtawar Ahtisham, Jinsook Lee, Doug Pietrzak, Daryl Hedley, Jorge Dias, Chris Shaw, Ruth Schäfer, René F. Kizilcec  

**Link**: [PDF](https://arxiv.org/pdf/2602.16571)  

**Abstract**: Large-scale sharing of dialogue-based data is instrumental for advancing the science of teaching and learning, yet rigorous de-identification remains a major barrier. In mathematics tutoring transcripts, numeric expressions frequently resemble structured identifiers (e.g., dates or IDs), leading generic Personally Identifiable Information (PII) detection systems to over-redact core instructional content and reduce dataset utility. This work asks how PII can be detected in math tutoring transcripts while preserving their educational utility. To address this challenge, we investigate the "numeric ambiguity" problem and introduce MathEd-PII, the first benchmark dataset for PII detection in math tutoring dialogues, created through a human-in-the-loop LLM workflow that audits upstream redactions and generates privacy-preserving surrogates. The dataset contains 1,000 tutoring sessions (115,620 messages; 769,628 tokens) with validated PII annotations. Using a density-based segmentation method, we show that false PII redactions are disproportionately concentrated in math-dense regions, confirming numeric ambiguity as a key failure mode. We then compare four detection strategies: a Presidio baseline and LLM-based approaches with basic, math-aware, and segment-aware prompting. Math-aware prompting substantially improves performance over the baseline (F1: 0.821 vs. 0.379) while reducing numeric false positives, demonstrating that de-identification must incorporate domain context to preserve analytic utility. This work provides both a new benchmark and evidence that utility-preserving de-identification for tutoring data requires domain-aware modeling. 

---
# Supercharging Agenda Setting Research: The ParlaCAP Dataset of 28 European Parliaments and a Scalable Multilingual LLM-Based Classification 

**Authors**: Taja Kuzman Pungeršek, Peter Rupnik, Daniela Širinić, Nikola Ljubešić  

**Link**: [PDF](https://arxiv.org/pdf/2602.16516)  

**Abstract**: This paper introduces ParlaCAP, a large-scale dataset for analyzing parliamentary agenda setting across Europe, and proposes a cost-effective method for building domain-specific policy topic classifiers. Applying the Comparative Agendas Project (CAP) schema to the multilingual ParlaMint corpus of over 8 million speeches from 28 parliaments of European countries and autonomous regions, we follow a teacher-student framework in which a high-performing large language model (LLM) annotates in-domain training data and a multilingual encoder model is fine-tuned on these annotations for scalable data annotation. We show that this approach produces a classifier tailored to the target domain. Agreement between the LLM and human annotators is comparable to inter-annotator agreement among humans, and the resulting model outperforms existing CAP classifiers trained on manually-annotated but out-of-domain data. In addition to the CAP annotations, the ParlaCAP dataset offers rich speaker and party metadata, as well as sentiment predictions coming from the ParlaSent multilingual transformer model, enabling comparative research on political attention and representation across countries. We illustrate the analytical potential of the dataset with three use cases, examining the distribution of parliamentary attention across policy topics, sentiment patterns in parliamentary speech, and gender differences in policy attention. 

---
# Optimizing Soft Prompt Tuning via Structural Evolution 

**Authors**: Zhenzhen Huang, Chaoning Zhang, Haoyu Bian, Songbo Zhang, Chi-lok Andy Tai, Jiaquan Zhang, Caiyan Qin, Jingjing Qu, Yalan Ye, Yang Yang, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16500)  

**Abstract**: Soft prompt tuning leverages continuous embeddings to capture task-specific information in large pre-trained language models (LLMs), achieving competitive performance in few-shot settings. However, soft prompts rely on high-dimensional, implicit representations and lack explicit semantics and traceable training behaviors, which limits their interpretability. To address this limitation, we propose a soft prompt tuning optimization method based on topological morphological evolution. Specifically, we employ persistent homology from topological data analysis (TDA) to quantify the structural representations of soft prompts in continuous parameter space and their training process evolution. Quantitative analysis shows that topologically stable and compact soft prompts achieve better downstream performance. Based on this empirical observation, we construct a loss function for optimizing soft prompt tuning, termed Topological Soft Prompt Loss (TSLoss). TSLoss guides the model to learn structurally stable adaptations by quantifying inter-parameter connectivity and redundancy. Extensive experiments show that training with TSLoss accelerates convergence and improves tuning performance, providing an interpretable method to understand and optimize soft prompt tuning from structural and topological perspectives. 

---
# From Growing to Looping: A Unified View of Iterative Computation in LLMs 

**Authors**: Ferdinand Kapl, Emmanouil Angelis, Kaitlin Maile, Johannes von Oswald, Stefan Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2602.16490)  

**Abstract**: Looping, reusing a block of layers across depth, and depth growing, training shallow-to-deep models by duplicating middle layers, have both been linked to stronger reasoning, but their relationship remains unclear. We provide a mechanistic unification: looped and depth-grown models exhibit convergent depth-wise signatures, including increased reliance on late layers and recurring patterns aligned with the looped or grown block. These shared signatures support the view that their gains stem from a common form of iterative computation. Building on this connection, we show that the two techniques are adaptable and composable: applying inference-time looping to the middle blocks of a depth-grown model improves accuracy on some reasoning primitives by up to $2\times$, despite the model never being trained to loop. Both approaches also adapt better than the baseline when given more in-context examples or additional supervised fine-tuning data. Additionally, depth-grown models achieve the largest reasoning gains when using higher-quality, math-heavy cooldown mixtures, which can be further boosted by adapting a middle block to loop. Overall, our results position depth growth and looping as complementary, practical methods for inducing and scaling iterative computation to improve reasoning. 

---
# Learning to Learn from Language Feedback with Social Meta-Learning 

**Authors**: Jonathan Cook, Diego Antognini, Martin Klissarov, Claudiu Musat, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2602.16488)  

**Abstract**: Large language models (LLMs) often struggle to learn from corrective feedback within a conversational context. They are rarely proactive in soliciting this feedback, even when faced with ambiguity, which can make their dialogues feel static, one-sided, and lacking the adaptive qualities of human conversation. To address these limitations, we draw inspiration from social meta-learning (SML) in humans - the process of learning how to learn from others. We formulate SML as a finetuning methodology, training LLMs to solicit and learn from language feedback in simulated pedagogical dialogues, where static tasks are converted into interactive social learning problems. SML effectively teaches models to use conversation to solve problems they are unable to solve in a single turn. This capability generalises across domains; SML on math problems produces models that better use feedback to solve coding problems and vice versa. Furthermore, despite being trained only on fully-specified problems, these models are better able to solve underspecified tasks where critical information is revealed over multiple turns. When faced with this ambiguity, SML-trained models make fewer premature answer attempts and are more likely to ask for the information they need. This work presents a scalable approach to developing AI systems that effectively learn from language feedback. 

---
# Team of Thoughts: Efficient Test-time Scaling of Agentic Systems through Orchestrated Tool Calling 

**Authors**: Jeffrey T. H. Wong, Zixi Zhang, Junyi Liu, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2602.16485)  

**Abstract**: Existing Multi-Agent Systems (MAS) typically rely on static, homogeneous model configurations, limiting their ability to exploit the distinct strengths of differently post-trained models. To address this, we introduce Team-of-Thoughts, a novel MAS architecture that leverages the complementary capabilities of heterogeneous agents via an orchestrator-tool paradigm. Our framework introduces two key mechanisms to optimize performance: (1) an orchestrator calibration scheme that identifies models with superior coordination capabilities, and (2) a self-assessment protocol where tool agents profile their own domain expertise to account for variations in post-training skills. During inference, the orchestrator dynamically activates the most suitable tool agents based on these proficiency profiles. Experiments on five reasoning and code generation benchmarks show that Team-of-Thoughts delivers consistently superior task performance. Notably, on AIME24 and LiveCodeBench, our approach achieves accuracies of 96.67% and 72.53%, respectively, substantially outperforming homogeneous role-play baselines, which score 80% and 65.93%. 

---
# Training Models on Dialects of Translationese Shows How Lexical Diversity and Source-Target Syntactic Similarity Shape Learning 

**Authors**: Jenny Kunz  

**Link**: [PDF](https://arxiv.org/pdf/2602.16469)  

**Abstract**: Machine-translated data is widely used in multilingual NLP, particularly when native text is scarce. However, translated text differs systematically from native text. This phenomenon is known as translationese, and it reflects both traces of the source language and characteristic properties of translation itself. In this paper, we study how training on machine-translated data affects small English language models, focusing on how translationese from different source languages shapes linguistic acceptability judgments and language modelling for different domains. We train models on English text translated from 24 typologically and resource-diverse source languages, enabling a systematic analysis of how source language and corpus properties influence what models learn. Our results show that the source language has a clear impact on model behavior: general perplexity is more driven by the lexical diversity of the translated corpus, while grammatical performance is strongly correlated to typological similarity to English, given enough data. 

---
# IndicEval: A Bilingual Indian Educational Evaluation Framework for Large Language Models 

**Authors**: Saurabh Bharti, Gaurav Azad, Abhinaw Jagtap, Nachiket Tapas  

**Link**: [PDF](https://arxiv.org/pdf/2602.16467)  

**Abstract**: The rapid advancement of large language models (LLMs) necessitates evaluation frameworks that reflect real-world academic rigor and multilingual complexity. This paper introduces IndicEval, a scalable benchmarking platform designed to assess LLM performance using authentic high-stakes examination questions from UPSC, JEE, and NEET across STEM and humanities domains in both English and Hindi. Unlike synthetic benchmarks, IndicEval grounds evaluation in real examination standards, enabling realistic measurement of reasoning, domain knowledge, and bilingual adaptability. The framework automates assessment using Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting strategies and supports modular integration of new models and languages. Experiments conducted on Gemini 2.0 Flash, GPT-4, Claude, and LLaMA 3-70B reveal three major findings. First, CoT prompting consistently improves reasoning accuracy, with substantial gains across subjects and languages. Second, significant cross-model performance disparities persist, particularly in high-complexity examinations. Third, multilingual degradation remains a critical challenge, with marked accuracy drops in Hindi compared to English, especially under Zero-Shot conditions. These results highlight persistent gaps in bilingual reasoning and domain transfer. Overall, IndicEval provides a practice-oriented, extensible foundation for rigorous, equitable evaluation of LLMs in multilingual educational settings and offers actionable insights for improving reasoning robustness and language adaptability. 

---
# TabAgent: A Framework for Replacing Agentic Generative Components with Tabular-Textual Classifiers 

**Authors**: Ido Levy, Eilam Shapira, Yinon Goldshtein, Avi Yaeli, Nir Mashkif, Segev Shlomov  

**Link**: [PDF](https://arxiv.org/pdf/2602.16429)  

**Abstract**: Agentic systems, AI architectures that autonomously execute multi-step workflows to achieve complex goals, are often built using repeated large language model (LLM) calls for closed-set decision tasks such as routing, shortlisting, gating, and verification. While convenient, this design makes deployments slow and expensive due to cumulative latency and token usage. We propose TabAgent, a framework for replacing generative decision components in closed-set selection tasks with a compact textual-tabular classifier trained on execution traces. TabAgent (i) extracts structured schema, state, and dependency features from trajectories (TabSchema), (ii) augments coverage with schema-aligned synthetic supervision (TabSynth), and (iii) scores candidates with a lightweight classifier (TabHead). On the long-horizon AppWorld benchmark, TabAgent maintains task-level success while eliminating shortlist-time LLM calls, reducing latency by approximately 95% and inference cost by 85-91%. Beyond tool shortlisting, TabAgent generalizes to other agentic decision heads, establishing a paradigm for learned discriminative replacements of generative bottlenecks in production agent architectures. 

---
# Label-Consistent Data Generation for Aspect-Based Sentiment Analysis Using LLM Agents 

**Authors**: Mohammad H.A. Monfared, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2602.16379)  

**Abstract**: We propose an agentic data augmentation method for Aspect-Based Sentiment Analysis (ABSA) that uses iterative generation and verification to produce high quality synthetic training examples. To isolate the effect of agentic structure, we also develop a closely matched prompting-based baseline using the same model and instructions. Both methods are evaluated across three ABSA subtasks (Aspect Term Extraction (ATE), Aspect Sentiment Classification (ATSC), and Aspect Sentiment Pair Extraction (ASPE)), four SemEval datasets, and two encoder-decoder models: T5-Base and Tk-Instruct. Our results show that the agentic augmentation outperforms raw prompting in label preservation of the augmented data, especially when the tasks require aspect term generation. In addition, when combined with real data, agentic augmentation provides higher gains, consistently outperforming prompting-based generation. These benefits are most pronounced for T5-Base, while the more heavily pretrained Tk-Instruct exhibits smaller improvements. As a result, augmented data helps T5-Base achieve comparable performance with its counterpart. 

---
# Helpful to a Fault: Measuring Illicit Assistance in Multi-Turn, Multilingual LLM Agents 

**Authors**: Nivya Talokar, Ayush K Tarun, Murari Mandal, Maksym Andriushchenko, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2602.16346)  

**Abstract**: LLM-based agents execute real-world workflows via tools and memory. These affordances enable ill-intended adversaries to also use these agents to carry out complex misuse scenarios. Existing agent misuse benchmarks largely test single-prompt instructions, leaving a gap in measuring how agents end up helping with harmful or illegal tasks over multiple turns. We introduce STING (Sequential Testing of Illicit N-step Goal execution), an automated red-teaming framework that constructs a step-by-step illicit plan grounded in a benign persona and iteratively probes a target agent with adaptive follow-ups, using judge agents to track phase completion. We further introduce an analysis framework that models multi-turn red-teaming as a time-to-first-jailbreak random variable, enabling analysis tools like discovery curves, hazard-ratio attribution by attack language, and a new metric: Restricted Mean Jailbreak Discovery. Across AgentHarm scenarios, STING yields substantially higher illicit-task completion than single-turn prompting and chat-oriented multi-turn baselines adapted to tool-using agents. In multilingual evaluations across six non-English settings, we find that attack success and illicit-task completion do not consistently increase in lower-resource languages, diverging from common chatbot findings. Overall, STING provides a practical way to evaluate and stress-test agent misuse in realistic deployment settings, where interactions are inherently multi-turn and often multilingual. 

---
# MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks 

**Authors**: Zexue He, Yu Wang, Churan Zhi, Yuanzhe Hu, Tzu-Ping Chen, Lang Yin, Ze Chen, Tong Arthur Wu, Siru Ouyang, Zihan Wang, Jiaxin Pei, Julian McAuley, Yejin Choi, Alex Pentland  

**Link**: [PDF](https://arxiv.org/pdf/2602.16313)  

**Abstract**: Existing evaluations of agents with memory typically assess memorization and action in isolation. One class of benchmarks evaluates memorization by testing recall of past conversations or text but fails to capture how memory is used to guide future decisions. Another class focuses on agents acting in single-session tasks without the need for long-term memory. However, in realistic settings, memorization and action are tightly coupled: agents acquire memory while interacting with the environment, and subsequently rely on that memory to solve future tasks. To capture this setting, we introduce MemoryArena, a unified evaluation gym for benchmarking agent memory in multi-session Memory-Agent-Environment loops. The benchmark consists of human-crafted agentic tasks with explicitly interdependent subtasks, where agents must learn from earlier actions and feedback by distilling experiences into memory, and subsequently use that memory to guide later actions to solve the overall task. MemoryArena supports evaluation across web navigation, preference-constrained planning, progressive information search, and sequential formal reasoning, and reveals that agents with near-saturated performance on existing long-context memory benchmarks like LoCoMo perform poorly in our agentic setting, exposing a gap in current evaluations for agents with memory. 

---
# MultiCW: A Large-Scale Balanced Benchmark Dataset for Training Robust Check-Worthiness Detection Models 

**Authors**: Martin Hyben, Sebastian Kula, Jan Cegin, Jakub Simko, Ivan Srba, Robert Moro  

**Link**: [PDF](https://arxiv.org/pdf/2602.16298)  

**Abstract**: Large Language Models (LLMs) are beginning to reshape how media professionals verify information, yet automated support for detecting check-worthy claims a key step in the fact-checking process remains limited. We introduce the Multi-Check-Worthy (MultiCW) dataset, a balanced multilingual benchmark for check-worthy claim detection spanning 16 languages, 7 topical domains, and 2 writing styles. It consists of 123,722 samples, evenly distributed between noisy (informal) and structured (formal) texts, with balanced representation of check-worthy and non-check-worthy classes across all languages. To probe robustness, we also introduce an equally balanced out-of-distribution evaluation set of 27,761 samples in 4 additional languages. To provide baselines, we benchmark 3 common fine-tuned multilingual transformers against a diverse set of 15 commercial and open LLMs under zero-shot settings. Our findings show that fine-tuned models consistently outperform zero-shot LLMs on claim classification and show strong out-of-distribution generalization across languages, domains, and styles. MultiCW provides a rigorous multilingual resource for advancing automated fact-checking and enables systematic comparisons between fine-tuned models and cutting-edge LLMs on the check-worthy claim detection task. 

---
# Aladdin-FTI @ AMIYA Three Wishes for Arabic NLP: Fidelity, Diglossia, and Multidialectal Generation 

**Authors**: Jonathan Mutal, Perla Al Almaoui, Simon Hengchen, Pierrette Bouillon  

**Link**: [PDF](https://arxiv.org/pdf/2602.16290)  

**Abstract**: Arabic dialects have long been under-represented in Natural Language Processing (NLP) research due to their non-standardization and high variability, which pose challenges for computational modeling. Recent advances in the field, such as Large Language Models (LLMs), offer promising avenues to address this gap by enabling Arabic to be modeled as a pluricentric language rather than a monolithic system. This paper presents Aladdin-FTI, our submission to the AMIYA shared task. The proposed system is designed to both generate and translate dialectal Arabic (DA). Specifically, the model supports text generation in Moroccan, Egyptian, Palestinian, Syrian, and Saudi dialects, as well as bidirectional translation between these dialects, Modern Standard Arabic (MSA), and English. The code and trained model are publicly available. 

---
# Are LLMs Ready to Replace Bangla Annotators? 

**Authors**: Md. Najib Hasan, Touseef Hasan, Souvika Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2602.16241)  

**Abstract**: Large Language Models (LLMs) are increasingly used as automated annotators to scale dataset creation, yet their reliability as unbiased annotators--especially for low-resource and identity-sensitive settings--remains poorly understood. In this work, we study the behavior of LLMs as zero-shot annotators for Bangla hate speech, a task where even human agreement is challenging, and annotator bias can have serious downstream consequences. We conduct a systematic benchmark of 17 LLMs using a unified evaluation framework. Our analysis uncovers annotator bias and substantial instability in model judgments. Surprisingly, increased model scale does not guarantee improved annotation quality--smaller, more task-aligned models frequently exhibit more consistent behavior than their larger counterparts. These results highlight important limitations of current LLMs for sensitive annotation tasks in low-resource languages and underscore the need for careful evaluation before deployment. 

---
# Long-Tail Knowledge in Large Language Models: Taxonomy, Mechanisms, Interventions and Implications 

**Authors**: Sanket Badhe, Deep Shah, Nehal Kathrotia  

**Link**: [PDF](https://arxiv.org/pdf/2602.16201)  

**Abstract**: Large language models (LLMs) are trained on web-scale corpora that exhibit steep power-law distributions, in which the distribution of knowledge is highly long-tailed, with most appearing infrequently. While scaling has improved average-case performance, persistent failures on low-frequency, domain-specific, cultural, and temporal knowledge remain poorly characterized. This paper develops a structured taxonomy and analysis of long-Tail Knowledge in large language models, synthesizing prior work across technical and sociotechnical perspectives.
We introduce a structured analytical framework that synthesizes prior work across four complementary axes: how long-Tail Knowledge is defined, the mechanisms by which it is lost or distorted during training and inference, the technical interventions proposed to mitigate these failures, and the implications of these failures for fairness, accountability, transparency, and user trust. We further examine how existing evaluation practices obscure tail behavior and complicate accountability for rare but consequential failures. The paper concludes by identifying open challenges related to privacy, sustainability, and governance that constrain long-Tail Knowledge representation. Taken together, this paper provides a unifying conceptual framework for understanding how long-Tail Knowledge is defined, lost, evaluated, and manifested in deployed language model systems. 

---
# The Validity of Coreference-based Evaluations of Natural Language Understanding 

**Authors**: Ian Porada  

**Link**: [PDF](https://arxiv.org/pdf/2602.16200)  

**Abstract**: In this thesis, I refine our understanding as to what conclusions we can reach from coreference-based evaluations by expanding existing evaluation practices and considering the extent to which evaluation results are either converging or conflicting. First, I analyze standard coreference evaluations and show that their design often leads to non-generalizable conclusions due to issues of measurement validity - including contestedness (multiple, competing definitions of coreference) and convergent validity (evaluation results that rank models differently across benchmarks). Second, I propose and implement a novel evaluation focused on testing systems' ability to infer the relative plausibility of events, a key aspect of resolving coreference. Through this extended evaluation, I find that contemporary language models demonstrate strong performance on standard benchmarks - improving over earlier baseline systems within certain domains and types of coreference - but remain sensitive to the evaluation conditions: they often fail to generalize in ways one would expect a human to be capable of when evaluation contexts are slightly modified. Taken together, these findings clarify both the strengths, such as improved accuracy over baselines on widely used evaluations, and the limitations of the current NLP paradigm, including weaknesses in measurement validity, and suggest directions for future work in developing better evaluation methods and more genuinely generalizable systems. 

---
# Beyond Learning: A Training-Free Alternative to Model Adaptation 

**Authors**: Namkyung Yoon, Kyeonghyun Yoo, Wooyong Jung, Sanghong Kim, Hwangnam Kim  

**Link**: [PDF](https://arxiv.org/pdf/2602.16189)  

**Abstract**: Despite the continuous research and evolution of language models, they sometimes underperform previous versions. Existing approaches to overcome these challenges are resource-intensive, highlighting the need for alternatives that enable immediate action. We assume that each language model has a local module inside that is suitable for a specific function. First, this work identifies a set of modules showing consistent and local activation changes under an inference workload through activation-based analysis. Subsequently, we transplant an internal module that is properly activated for a specific task into the target model, leading to immediate and measurable functional changes without additional training or fine-tuning. To experimentally demonstrate the effectiveness of the transplant technique, we quantify the relationship between transplant strength and performance improvement under different conditions for two language models. In the cross-generation setting, we find that transplanting activation-selected modules can substantially improve the underperforming model, reaching up to twice the target baseline and achieving gap-based recovery above 100%. Moreover, in transplant experiments between a base model and its instruction-tuned counterpart, transplantation improves the underperforming model toward the stronger baseline, yielding up to about 2.33 times the target baseline with gap-based recovery reaching up to 100% in the best case. These results show that meaningful capacity transfer can be realized through the implantation of highly localized modules implied by language models. Overall, this work provides empirical evidence for task-localized modularity in language models and presents a new research area: model transplantation. 

---
# LLMs Exhibit Significantly Lower Uncertainty in Creative Writing Than Professional Writers 

**Authors**: Peiqi Sui  

**Link**: [PDF](https://arxiv.org/pdf/2602.16162)  

**Abstract**: We argue that uncertainty is a key and understudied limitation of LLMs' performance in creative writing, which is often characterized as trite and cliché-ridden. Literary theory identifies uncertainty as a necessary condition for creative expression, while current alignment strategies steer models away from uncertain outputs to ensure factuality and reduce hallucination. We formalize this tension by quantifying the "uncertainty gap" between human-authored stories and model-generated continuations. Through a controlled information-theoretic analysis of 28 LLMs on high-quality storytelling datasets, we demonstrate that human writing consistently exhibits significantly higher uncertainty than model outputs. We find that instruction-tuned and reasoning models exacerbate this trend compared to their base counterparts; furthermore, the gap is more pronounced in creative writing than in functional domains, and strongly correlates to writing quality. Achieving human-level creativity requires new uncertainty-aware alignment paradigms that can distinguish between destructive hallucinations and the constructive ambiguity required for literary richness. 

---
# Balancing Faithfulness and Performance in Reasoning via Multi-Listener Soft Execution 

**Authors**: Nithin Sivakumaran, Shoubin Yu, Hyunji Lee, Yue Zhang, Ali Payani, Mohit Bansal, Elias Stengel-Eskin  

**Link**: [PDF](https://arxiv.org/pdf/2602.16154)  

**Abstract**: Chain-of-thought (CoT) reasoning sometimes fails to faithfully reflect the true computation of a large language model (LLM), hampering its utility in explaining how LLMs arrive at their answers. Moreover, optimizing for faithfulness and interpretability in reasoning often degrades task performance. To address this tradeoff and improve CoT faithfulness, we propose Reasoning Execution by Multiple Listeners (REMUL), a multi-party reinforcement learning approach. REMUL builds on the hypothesis that reasoning traces which other parties can follow will be more faithful. A speaker model generates a reasoning trace, which is truncated and passed to a pool of listener models who "execute" the trace, continuing the trace to an answer. Speakers are rewarded for producing reasoning that is clear to listeners, with additional correctness regularization via masked supervised finetuning to counter the tradeoff between faithfulness and performance. On multiple reasoning benchmarks (BIG-Bench Extra Hard, MuSR, ZebraLogicBench, and FOLIO), REMUL consistently and substantially improves three measures of faithfulness -- hint attribution, early answering area over the curve (AOC), and mistake injection AOC -- while also improving accuracy. Our analysis finds that these gains are robust across training domains, translate to legibility gains, and are associated with shorter and more direct CoTs. 

---
# Missing-by-Design: Certifiable Modality Deletion for Revocable Multimodal Sentiment Analysis 

**Authors**: Rong Fu, Wenxin Zhang, Ziming Wang, Chunlei Meng, Jiaxuan Lu, Jiekai Wu, Kangan Qian, Hao Zhang, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2602.16144)  

**Abstract**: As multimodal systems increasingly process sensitive personal data, the ability to selectively revoke specific data modalities has become a critical requirement for privacy compliance and user autonomy. We present Missing-by-Design (MBD), a unified framework for revocable multimodal sentiment analysis that combines structured representation learning with a certifiable parameter-modification pipeline. Revocability is critical in privacy-sensitive applications where users or regulators may request removal of modality-specific information. MBD learns property-aware embeddings and employs generator-based reconstruction to recover missing channels while preserving task-relevant signals. For deletion requests, the framework applies saliency-driven candidate selection and a calibrated Gaussian update to produce a machine-verifiable Modality Deletion Certificate. Experiments on benchmark datasets show that MBD achieves strong predictive performance under incomplete inputs and delivers a practical privacy-utility trade-off, positioning surgical unlearning as an efficient alternative to full retraining. 

---
# Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities 

**Authors**: Shankar Padmanabhan, Mustafa Omer Gul, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2602.16093)  

**Abstract**: Post-training endows pretrained LLMs with a variety of desirable skills, including instruction-following, reasoning, and others. However, these post-trained LLMs only encode knowledge up to a cut-off date, necessitating continual adaptation. Unfortunately, existing solutions cannot simultaneously learn new knowledge from an adaptation document corpora and mitigate the forgetting of earlier learned capabilities. To address this, we introduce Distillation via Split Contexts (DiSC), a simple context-distillation based approach for continual knowledge adaptation. \methodname~derives student and teacher distributions by conditioning on distinct segments of the training example and minimizes the KL divergence between the shared tokens. This allows us to efficiently apply context-distillation without requiring explicit generation steps during training. We run experiments on four post-trained models and two adaptation domains. Compared to prior finetuning and distillation methods for continual adaptation, DiSC consistently reports the best trade-off between learning new knowledge and mitigating forgetting of previously learned skills like instruction-following, reasoning, and factual knowledge. 

---
# Language Statistics and False Belief Reasoning: Evidence from 41 Open-Weight LMs 

**Authors**: Sean Trott, Samuel Taylor, Cameron Jones, James A. Michaelov, Pamela D. Rivière  

**Link**: [PDF](https://arxiv.org/pdf/2602.16085)  

**Abstract**: Research on mental state reasoning in language models (LMs) has the potential to inform theories of human social cognition--such as the theory that mental state reasoning emerges in part from language exposure--and our understanding of LMs themselves. Yet much published work on LMs relies on a relatively small sample of closed-source LMs, limiting our ability to rigorously test psychological theories and evaluate LM capacities. Here, we replicate and extend published work on the false belief task by assessing LM mental state reasoning behavior across 41 open-weight models (from distinct model families). We find sensitivity to implied knowledge states in 34% of the LMs tested; however, consistent with prior work, none fully ``explain away'' the effect in humans. Larger LMs show increased sensitivity and also exhibit higher psychometric predictive power. Finally, we use LM behavior to generate and test a novel hypothesis about human cognition: both humans and LMs show a bias towards attributing false beliefs when knowledge states are cued using a non-factive verb (``John thinks...'') than when cued indirectly (``John looks in the...''). Unlike the primary effect of knowledge states, where human sensitivity exceeds that of LMs, the magnitude of the human knowledge cue effect falls squarely within the distribution of LM effect sizes-suggesting that distributional statistics of language can in principle account for the latter but not the former in humans. These results demonstrate the value of using larger samples of open-weight LMs to test theories of human cognition and evaluate LM capacities. 

---
# Surgical Activation Steering via Generative Causal Mediation 

**Authors**: Aruna Sankaranarayanan, Amir Zur, Atticus Geiger, Dylan Hadfield-Menell  

**Link**: [PDF](https://arxiv.org/pdf/2602.16080)  

**Abstract**: Where should we intervene in a language model (LM) to control behaviors that are diffused across many tokens of a long-form response? We introduce Generative Causal Mediation (GCM), a procedure for selecting model components, e.g., attention heads, to steer a binary concept (e.g., talk in verse vs. talk in prose) from contrastive long-form responses. In GCM, we first construct a dataset of contrasting inputs and responses. Then, we quantify how individual model components mediate the contrastive concept and select the strongest mediators for steering. We evaluate GCM on three tasks--refusal, sycophancy, and style transfer--across three language models. GCM successfully localizes concepts expressed in long-form responses and consistently outperforms correlational probe-based baselines when steering with a sparse set of attention heads. Together, these results demonstrate that GCM provides an effective approach for localizing and controlling the long-form responses of LMs. 

---
# CLAA: Cross-Layer Attention Aggregation for Accelerating LLM Prefill 

**Authors**: Bradley McDanel, Steven Li, Harshit Khaitan  

**Link**: [PDF](https://arxiv.org/pdf/2602.16054)  

**Abstract**: The prefill stage in long-context LLM inference remains a computational bottleneck. Recent token-ranking heuristics accelerate inference by selectively processing a subset of semantically relevant tokens. However, existing methods suffer from unstable token importance estimation, often varying between layers. Evaluating token-ranking quality independently from heuristic-specific architectures is challenging. To address this, we introduce an Answer-Informed Oracle, which defines ground-truth token importance by measuring attention from generated answers back to the prompt. This oracle reveals that existing heuristics exhibit high variance across layers: rankings can degrade sharply at specific layers, a failure mode invisible to end-to-end benchmarks. The diagnosis suggests a simple fix: aggregate scores across layers rather than relying on any single one. We implement this as Cross-Layer Attention Aggregation (CLAA), which closes the gap to the oracle upper bound and reduces Time-to-First-Token (TTFT) by up to 39\% compared to the Full KV Cache baseline. 

---
# A Curious Class of Adpositional Multiword Expressions in Korean 

**Authors**: Junghyun Min, Na-Rae Han, Jena D. Hwang, Nathan Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2602.16023)  

**Abstract**: Multiword expressions (MWEs) have been widely studied in cross-lingual annotation frameworks such as PARSEME. However, Korean MWEs remain underrepresented in these efforts. In particular, Korean multiword adpositions lack systematic analysis, annotated resources, and integration into existing multilingual frameworks. In this paper, we study a class of Korean functional multiword expressions: postpositional verb-based constructions (PVCs). Using data from Korean Wikipedia, we survey and analyze several PVC expressions and contrast them with non-MWEs and light verb constructions (LVCs) with similar structure. Building on this analysis, we propose annotation guidelines designed to support future work in Korean multiword adpositions and facilitate alignment with cross-lingual frameworks. 

---
# DocSplit: A Comprehensive Benchmark Dataset and Evaluation Approach for Document Packet Recognition and Splitting 

**Authors**: Md Mofijul Islam, Md Sirajus Salekin, Nivedha Balakrishnan, Vincil C. Bishop III, Niharika Jain, Spencer Romo, Bob Strahan, Boyi Xie, Diego A. Socolinsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.15958)  

**Abstract**: Document understanding in real-world applications often requires processing heterogeneous, multi-page document packets containing multiple documents stitched together. Despite recent advances in visual document understanding, the fundamental task of document packet splitting, which involves separating a document packet into individual units, remains largely unaddressed. We present the first comprehensive benchmark dataset, DocSplit, along with novel evaluation metrics for assessing the document packet splitting capabilities of large language models. DocSplit comprises five datasets of varying complexity, covering diverse document types, layouts, and multimodal settings. We formalize the DocSplit task, which requires models to identify document boundaries, classify document types, and maintain correct page ordering within a document packet. The benchmark addresses real-world challenges, including out-of-order pages, interleaved documents, and documents lacking clear demarcations. We conduct extensive experiments evaluating multimodal LLMs on our datasets, revealing significant performance gaps in current models' ability to handle complex document splitting tasks. The DocSplit benchmark datasets and proposed novel evaluation metrics provide a systematic framework for advancing document understanding capabilities essential for legal, financial, healthcare, and other document-intensive domains. We release the datasets to facilitate future research in document packet processing. 

---
# Doc-to-LoRA: Learning to Instantly Internalize Contexts 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2602.15902)  

**Abstract**: Long input sequences are central to in-context learning, document understanding, and multi-step reasoning of Large Language Models (LLMs). However, the quadratic attention cost of Transformers makes inference memory-intensive and slow. While context distillation (CD) can transfer information into model parameters, per-prompt distillation is impractical due to training costs and latency. To address these limitations, we propose Doc-to-LoRA (D2L), a lightweight hypernetwork that meta-learns to perform approximate CD within a single forward pass. Given an unseen prompt, D2L generates a LoRA adapter for a target LLM, enabling subsequent queries to be answered without re-consuming the original context, reducing latency and KV-cache memory consumption during inference of the target LLM. On a long-context needle-in-a-haystack task, D2L successfully learns to map contexts into adapters that store the needle information, achieving near-perfect zero-shot accuracy at sequence lengths exceeding the target LLM's native context window by more than 4x. On real-world QA datasets with limited compute, D2L outperforms standard CD while significantly reducing peak memory consumption and update latency. We envision that D2L can facilitate rapid adaptation of LLMs, opening up the possibility of frequent knowledge updates and personalized chat behavior. 

---
# MultiCube-RAG for Multi-hop Question Answering 

**Authors**: Jimeng Shi, Wei Hu, Runchu Tian, Bowen Jin, Wonbin Kweon, SeongKu Kang, Yunfan Kang, Dingqi Ye, Sizhe Zhou, Shaowen Wang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.15898)  

**Abstract**: Multi-hop question answering (QA) necessitates multi-step reasoning and retrieval across interconnected subjects, attributes, and relations. Existing retrieval-augmented generation (RAG) methods struggle to capture these structural semantics accurately, resulting in suboptimal performance. Graph-based RAGs structure such information in graphs, but the resulting graphs are often noisy and computationally expensive. Moreover, most methods rely on single-step retrieval, neglecting the need for multi-hop reasoning processes. Recent training-based approaches attempt to incentivize the large language models (LLMs) for iterative reasoning and retrieval, but their training processes are prone to unstable convergence and high computational overhead. To address these limitations, we devise an ontology-based cube structure with multiple and orthogonal dimensions to model structural subjects, attributes, and relations. Built on the cube structure, we propose MultiCube-RAG, a training-free method consisting of multiple cubes for multi-step reasoning and retrieval. Each cube specializes in modeling a class of subjects, so that MultiCube-RAG flexibly selects the most suitable cubes to acquire the relevant knowledge precisely. To enhance the query-based reasoning and retrieval, our method decomposes a complex multi-hop query into a set of simple subqueries along cube dimensions and conquers each of them sequentially. Experiments on four multi-hop QA datasets show that MultiCube-RAG improves response accuracy by 8.9% over the average performance of various baselines. Notably, we also demonstrate that our method performs with greater efficiency and inherent explainability. 

---
# Mitigating Gradient Inversion Risks in Language Models via Token Obfuscation 

**Authors**: Xinguo Feng, Zhongkui Ma, Zihan Wang, Alsharif Abuadbba, Guangdong Bai  

**Link**: [PDF](https://arxiv.org/pdf/2602.15897)  

**Abstract**: Training and fine-tuning large-scale language models largely benefit from collaborative learning, but the approach has been proven vulnerable to gradient inversion attacks (GIAs), which allow adversaries to reconstruct private training data from shared gradients. Existing defenses mainly employ gradient perturbation techniques, e.g., noise injection or gradient pruning, to disrupt GIAs' direct mapping from gradient space to token space. However, these methods often fall short due to the retention of semantics similarity across gradient, embedding, and token spaces. In this work, we propose a novel defense mechanism named GHOST (gradient shield with obfuscated tokens), a token-level obfuscation mechanism that neutralizes GIAs by decoupling the inherent connections across gradient, embedding, and token spaces. GHOST is built upon an important insight: due to the large scale of the token space, there exist semantically distinct yet embedding-proximate tokens that can serve as the shadow substitutes of the original tokens, which enables a semantic disconnection in the token space while preserving the connection in the embedding and gradient spaces. GHOST comprises a searching step, which identifies semantically distinct candidate tokens using a multi-criteria searching process, and a selection step, which selects optimal shadow tokens to ensure minimal disruption to features critical for training by preserving alignment with the internal outputs produced by original tokens. Evaluation across diverse model architectures (from BERT to Llama) and datasets demonstrates the remarkable effectiveness of GHOST in protecting privacy (as low as 1% in recovery rate) and preserving utility (up to 0.92 in classification F1 and 5.45 in perplexity), in both classification and generation tasks against state-of-the-art GIAs and adaptive attack scenarios. 

---
# Every Little Helps: Building Knowledge Graph Foundation Model with Fine-grained Transferable Multi-modal Tokens 

**Authors**: Yichi Zhang, Zhuo Chen, Lingbing Guo, Wen Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2602.15896)  

**Abstract**: Multi-modal knowledge graph reasoning (MMKGR) aims to predict the missing links by exploiting both graph structure information and multi-modal entity contents. Most existing works are designed for a transductive setting, which learns dataset-specific embeddings and struggles to generalize to new KGs. Recent knowledge graph foundation models (KGFMs) improve cross-KG transfer, but they mainly exploit structural patterns and ignore rich multi-modal signals. We address these gaps by proposing a token-based foundation model (TOFU) for MMKGR, which exhibits strong generalization across different MMKGs. TOFU discretizes structural, visual, and textual information into modality-specific tokens. TOFU then employs a hierarchical fusion architecture with mixture-of-message mechanisms, aiming to process these tokens and obtain transferable features for MMKGR. Experimental results on 17 transductive, inductive, and fully-inductive MMKGs show that TOFU consistently outperforms strong KGFM and MMKGR baselines, delivering strong performance on unseen MMKGs. 

---
# Understand Then Memory: A Cognitive Gist-Driven RAG Framework with Global Semantic Diffusion 

**Authors**: Pengcheng Zhou, Haochen Li, Zhiqiang Nie, JiaLe Chen, Qing Gong, Weizhen Zhang, Chun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15895)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively mitigates hallucinations in LLMs by incorporating external knowledge. However, the inherent discrete representation of text in existing frameworks often results in a loss of semantic integrity, leading to retrieval deviations. Inspired by the human episodic memory mechanism, we propose CogitoRAG, a RAG framework that simulates human cognitive memory processes. The core of this framework lies in the extraction and evolution of the Semantic Gist. During the offline indexing stage, CogitoRAG first deduces unstructured corpora into gist memory corpora, which are then transformed into a multi-dimensional knowledge graph integrating entities, relational facts, and memory nodes. In the online retrieval stage, the framework handles complex queries via Query Decomposition Module that breaks them into comprehensive sub-queries, mimicking the cognitive decomposition humans employ for complex information. Subsequently, Entity Diffusion Module performs associative retrieval across the graph, guided by structural relevance and an entity-frequency reward mechanism. Furthermore, we propose the CogniRank algorithm, which precisely reranks candidate passages by fusing diffusion-derived scores with semantic similarity. The final evidence is delivered to the generator in a passage-memory pairing format, providing high-density information support. Experimental results across five mainstream QA benchmarks and multi-task generation on GraphBench demonstrate that CogitoRAG significantly outperforms state-of-the-art RAG methods, showcasing superior capabilities in complex knowledge integration and reasoning. 

---
# Quality-constrained Entropy Maximization Policy Optimization for LLM Diversity 

**Authors**: Haihui Pan, Yuzhong Hong, Shaoke Lv, Junwei Bao, Hongfei Jiang, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2602.15894)  

**Abstract**: Recent research indicates that while alignment methods significantly improve the quality of large language model(LLM) outputs, they simultaneously reduce the diversity of the models' output. Although some methods have been proposed to enhance LLM output diversity, they often come at the cost of reduced performance. In this work, we first theoretically demonstrate that the alignment task can be decomposed into two distributions: quality and diversity. To enhance the diversity of LLM outputs while ensuring quality, we propose the Quality-constrained Entropy Maximization Policy Optimization (QEMPO). QEMPO aims to maximize the output entropy of the policy while ensuring output quality. By adding different constraints to QEMPO, we obtain different policies. To optimize policies, we propose both online and offline training methods. Experiments validate that QEMPO achieves performance comparable to or even better than RLHF while improving output diversity. 

---
# P-RAG: Prompt-Enhanced Parametric RAG with LoRA and Selective CoT for Biomedical and Multi-Hop QA 

**Authors**: Xingda Lyu, Gongfu Lyu, Zitai Yan, Yuxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15874)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities but remain limited by their reliance on static training data. Retrieval-Augmented Generation (RAG) addresses this constraint by retrieving external knowledge during inference, though it still depends heavily on knowledge base quality. To explore potential improvements, we evaluated three RAG variants-Standard RAG, DA-RAG, and our proposed Prompt-Enhanced Parametric RAG (P-RAG), a hybrid architecture that integrates parametric knowledge within the LLM and retrieved evidence, guided by Chain-of-Thought (CoT) prompting and Low-Rank Adaptation (LoRA) fine-tuning-on both general and biomedical datasets. Using LLaMA-3.2-1B-Instruct fine-tuned via LoRA, we evaluate on PubMedQA and 2WikiMultihopQA. P-RAG outperforms Standard RAG on PubMedQA by 10.47 percentage points in F1 (93.33% vs. 82.86%; 12.64% relative). On 2WikiMultihopQA, P-RAG nearly doubles the overall score vs. Standard RAG (33.44% vs. 17.83%) and achieves 44.03% on the Compare subset (with 42.74% Bridge, 21.84% Inference, 8.60% Compose). CoT prompting substantially improves multi-hop reasoning but yields mixed results for simpler, single-hop queries. These findings underscore P-RAG's potential for accurate, scalable, and contextually adaptive biomedical question answering. Our contributions include: (1) LoRA-based fine-tuning of LLaMA-3.2-1B-Instruct for biomedical QA, (2) introduction of P-RAG with Chain-of-Thought prompting, and (3) state-of-the-art results on PubMedQA and 2WikiMultihopQA. 

---
# CheckIfExist: Detecting Citation Hallucinations in the Era of AI-Generated Content 

**Authors**: Diletta Abbonato  

**Link**: [PDF](https://arxiv.org/pdf/2602.15871)  

**Abstract**: The proliferation of large language models (LLMs) in academic workflows has introduced unprecedented challenges to bibliographic integrity, particularly through reference hallucination -- the generation of plausible but non-existent citations. Recent investigations have documented the presence of AI-hallucinated citations even in papers accepted at premier machine learning conferences such as NeurIPS and ICLR, underscoring the urgency of automated verification mechanisms. This paper presents "CheckIfExist", an open-source web-based tool designed to provide immediate verification of bibliographic references through multi-source validation against CrossRef, Semantic Scholar, and OpenAlex scholarly databases. While existing reference management tools offer bibliographic organization capabilities, they do not provide real-time validation of citation authenticity. Commercial hallucination detection services, though increasingly available, often impose restrictive usage limits on free tiers or require substantial subscription fees. The proposed tool fills this gap by employing a cascading validation architecture with string similarity algorithms to compute multi-dimensional match confidence scores, delivering instant feedback on reference authenticity. The system supports both single-reference verification and batch processing of BibTeX entries through a unified interface, returning validated APA citations and exportable BibTeX records within seconds. 

---
# VDLM: Variable Diffusion LMs via Robust Latent-to-Text Rendering 

**Authors**: Shuhui Qu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15870)  

**Abstract**: Autoregressive language models decode left-to-right with irreversible commitments, limiting revision during multi-step reasoning. We propose \textbf{VDLM}, a modular variable diffusion language model that separates semantic planning from text rendering. VDLM applies LLaDA-style masked diffusion over semantic variable embeddings to enable iterative refinement in latent space, then post-trains the planner with trajectory-aware optimization using embedding-space rewards and values, avoiding text decoding inside the RL loop. To convert planned embeddings back to text, we use a \textbf{Vec2Text} renderer and introduce \textbf{embedding perturbations} to robustify decoding under planner noise. Across nine benchmarks spanning general reasoning, math, and code, VDLM is competitive in pre-training and yields substantial post-training improvements on long-form generation tasks, outperforming other baselines. These results highlight the effectiveness of embedding-space post-training and robust latent-to-text rendering for diffusion language modeling. 

---
# Towards Fair and Efficient De-identification: Quantifying the Efficiency and Generalizability of De-identification Approaches 

**Authors**: Noopur Zambare, Kiana Aghakasiri, Carissa Lin, Carrie Ye, J. Ross Mitchell, Mohamed Abdalla  

**Link**: [PDF](https://arxiv.org/pdf/2602.15869)  

**Abstract**: Large language models (LLMs) have shown strong performance on clinical de-identification, the task of identifying sensitive identifiers to protect privacy. However, previous work has not examined their generalizability between formats, cultures, and genders. In this work, we systematically evaluate fine-tuned transformer models (BERT, ClinicalBERT, ModernBERT), small LLMs (Llama 1-8B, Qwen 1.5-7B), and large LLMs (Llama-70B, Qwen-72B) at de-identification. We show that smaller models achieve comparable performance while substantially reducing inference cost, making them more practical for deployment. Moreover, we demonstrate that smaller models can be fine-tuned with limited data to outperform larger models in de-identifying identifiers drawn from Mandarin, Hindi, Spanish, French, Bengali, and regional variations of English, in addition to gendered names. To improve robustness in multi-cultural contexts, we introduce and publicly release BERT-MultiCulture-DEID, a set of de-identification models based on BERT, ClinicalBERT, and ModernBERT, fine-tuned on MIMIC with identifiers from multiple language variants. Our findings provide the first comprehensive quantification of the efficiency-generalizability trade-off in de-identification and establish practical pathways for fair and efficient clinical de-identification.
Details on accessing the models are available at: this https URL 

---
# Understanding LLM Failures: A Multi-Tape Turing Machine Analysis of Systematic Errors in Language Model Reasoning 

**Authors**: Magnus Boman  

**Link**: [PDF](https://arxiv.org/pdf/2602.15868)  

**Abstract**: Large language models (LLMs) exhibit failure modes on seemingly trivial tasks. We propose a formalisation of LLM interaction using a deterministic multi-tape Turing machine, where each tape represents a distinct component: input characters, tokens, vocabulary, model parameters, activations, probability distributions, and output text. The model enables precise localisation of failure modes to specific pipeline stages, revealing, e.g., how tokenisation obscures character-level structure needed for counting tasks. The model clarifies why techniques like chain-of-thought prompting help, by externalising computation on the output tape, while also revealing their fundamental limitations. This approach provides a rigorous, falsifiable alternative to geometric metaphors and complements empirical scaling laws with principled error analysis. 

---
# Playing With AI: How Do State-Of-The-Art Large Language Models Perform in the 1977 Text-Based Adventure Game Zork? 

**Authors**: Berry Gerrits  

**Link**: [PDF](https://arxiv.org/pdf/2602.15867)  

**Abstract**: In this positioning paper, we evaluate the problem-solving and reasoning capabilities of contemporary Large Language Models (LLMs) through their performance in Zork, the seminal text-based adventure game first released in 1977. The game's dialogue-based structure provides a controlled environment for assessing how LLM-based chatbots interpret natural language descriptions and generate appropriate action sequences to succeed in the game. We test the performance of leading proprietary models - ChatGPT, Claude, and Gemini - under both minimal and detailed instructions, measuring game progress through achieved scores as the primary metric. Our results reveal that all tested models achieve less than 10% completion on average, with even the best-performing model (Claude Opus 4.5) reaching only approximately 75 out of 350 possible points. Notably, providing detailed game instructions offers no improvement, nor does enabling ''extended thinking''. Qualitative analysis of the models' reasoning processes reveals fundamental limitations: repeated unsuccessful actions suggesting an inability to reflect on one's own thinking, inconsistent persistence of strategies, and failure to learn from previous attempts despite access to conversation history. These findings suggest substantial limitations in current LLMs' metacognitive abilities and problem-solving capabilities within the domain of text-based games, raising questions about the nature and extent of their reasoning capabilities. 

---
# NLP Privacy Risk Identification in Social Media (NLP-PRISM): A Survey 

**Authors**: Dhiman Goswami, Jai Kruthunz Naveen Kumar, Sanchari Das  

**Link**: [PDF](https://arxiv.org/pdf/2602.15866)  

**Abstract**: Natural Language Processing (NLP) is integral to social media analytics but often processes content containing Personally Identifiable Information (PII), behavioral cues, and metadata raising privacy risks such as surveillance, profiling, and targeted advertising. To systematically assess these risks, we review 203 peer-reviewed papers and propose the NLP Privacy Risk Identification in Social Media (NLP-PRISM) framework, which evaluates vulnerabilities across six dimensions: data collection, preprocessing, visibility, fairness, computational risk, and regulatory compliance. Our analysis shows that transformer models achieve F1-scores ranging from 0.58-0.84, but incur a 1% - 23% drop under privacy-preserving fine-tuning. Using NLP-PRISM, we examine privacy coverage in six NLP tasks: sentiment analysis (16), emotion detection (14), offensive language identification (19), code-mixed processing (39), native language identification (29), and dialect detection (24) revealing substantial gaps in privacy research. We further found a (reduced by 2% - 9%) trade-off in model utility, MIA AUC (membership inference attacks) 0.81, AIA accuracy 0.75 (attribute inference attacks). Finally, we advocate for stronger anonymization, privacy-aware learning, and fairness-driven training to enable ethical NLP in social media contexts. 

---
# Not the Example, but the Process: How Self-Generated Examples Enhance LLM Reasoning 

**Authors**: Daehoon Gwak, Minseo Jung, Junwoo Park, Minho Park, ChaeHun Park, Junha Hyung, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2602.15863)  

**Abstract**: Recent studies have shown that Large Language Models (LLMs) can improve their reasoning performance through self-generated few-shot examples, achieving results comparable to manually curated in-context examples. However, the underlying mechanism behind these gains remains unclear, making it hard to decide when and how to apply the technique effectively. In this work, we argue that the key benefit arises not from the generated examples themselves but from the act of creating them. To validate this, on reasoning-intensive tasks across diverse LLM architectures, we systematically evaluate three prompting strategies for in-context learning: (1) Zero-shot prompting; (2) Integrated prompting, where LLMs create and solve problems within a single, unified prompt; and (3) Decoupled prompting, where self-generated examples are reused as in-context examples, but the context of their creation itself is excluded. We conduct experiments across five widely used model architectures, demonstrating that Integrated prompting consistently outperforms both Zero-shot and Decoupled prompting. In contrast, Decoupled prompting offers only marginal gains over Zero-shot. Further, for a more in-depth analysis, we conduct an attention analysis and observe significant differences in attention patterns between Integrated and Decoupled prompting. These findings suggest that the advantage of self-generation prompting comes from the process of problem creation, not the examples themselves, providing valuable insights for designing more effective prompting strategies. 

---
# Enhancing Action and Ingredient Modeling for Semantically Grounded Recipe Generation 

**Authors**: Guoshan Liu, Bin Zhu, Yian Li, Jingjing Chen, Chong-Wah Ngo, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15862)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLMMs) have enabled recipe generation from food images, yet outputs often contain semantically incorrect actions or ingredients despite high lexical scores (e.g., BLEU, ROUGE). To address this gap, we propose a semantically grounded framework that predicts and validates actions and ingredients as internal context for instruction generation. Our two-stage pipeline combines supervised fine-tuning (SFT) with reinforcement fine-tuning (RFT): SFT builds foundational accuracy using an Action-Reasoning dataset and ingredient corpus, while RFT employs frequency-aware rewards to improve long-tail action prediction and ingredient generalization. A Semantic Confidence Scoring and Rectification (SCSR) module further filters and corrects predictions. Experiments on Recipe1M show state-of-the-art performance and markedly improved semantic fidelity. 

---
# CAST: Achieving Stable LLM-based Text Analysis for Data Analytics 

**Authors**: Jinxiang Xie, Zihao Li, Wei He, Rui Ding, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15861)  

**Abstract**: Text analysis of tabular data relies on two core operations: \emph{summarization} for corpus-level theme extraction and \emph{tagging} for row-level labeling. A critical limitation of employing large language models (LLMs) for these tasks is their inability to meet the high standards of output stability demanded by data analytics. To address this challenge, we introduce \textbf{CAST} (\textbf{C}onsistency via \textbf{A}lgorithmic Prompting and \textbf{S}table \textbf{T}hinking), a framework that enhances output stability by constraining the model's latent reasoning path. CAST combines (i) Algorithmic Prompting to impose a procedural scaffold over valid reasoning transitions and (ii) Thinking-before-Speaking to enforce explicit intermediate commitments before final generation. To measure progress, we introduce \textbf{CAST-S} and \textbf{CAST-T}, stability metrics for bulleted summarization and tagging, and validate their alignment with human judgments. Experiments across publicly available benchmarks on multiple LLM backbones show that CAST consistently achieves the best stability among all baselines, improving Stability Score by up to 16.2\%, while maintaining or improving output quality. 

---
# Reranker Optimization via Geodesic Distances on k-NN Manifolds 

**Authors**: Wen G. Gong  

**Link**: [PDF](https://arxiv.org/pdf/2602.15860)  

**Abstract**: Current neural reranking approaches for retrieval-augmented generation (RAG) rely on cross-encoders or large language models (LLMs), requiring substantial computational resources and exhibiting latencies of 3-5 seconds per query. We propose Maniscope, a geometric reranking method that computes geodesic distances on k-nearest neighbor (k-NN) manifolds constructed over retrieved document candidates. This approach combines global cosine similarity with local manifold geometry to capture semantic structure that flat Euclidean metrics miss. Evaluating on eight BEIR benchmark datasets (1,233 queries), Maniscope outperforms HNSW graph-based baseline on the three hardest datasets (NFCorpus: +7.0%, TREC-COVID: +1.6%, AorB: +2.8% NDCG@3) while being 3.2x faster (4.7 ms vs 14.8 ms average). Compared to cross-encoder rerankers, Maniscope achieves within 2% accuracy at 10-45x lower latency. On TREC-COVID, LLM-Reranker provides only +0.5% NDCG@3 improvement over Maniscope at 840x higher latency, positioning Maniscope as a practical alternative for real-time RAG deployment. The method requires O(N D + M^2 D + M k log k) complexity where M << N , enabling sub-10 ms latency. We plan to release Maniscope as open-source software. 

---
# From Transcripts to AI Agents: Knowledge Extraction, RAG Integration, and Robust Evaluation of Conversational AI Assistants 

**Authors**: Krittin Pachtrachai, Petmongkon Pornpichitsuwan, Wachiravit Modecrua, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2602.15859)  

**Abstract**: Building reliable conversational AI assistants for customer-facing industries remains challenging due to noisy conversational data, fragmented knowledge, and the requirement for accurate human hand-off - particularly in domains that depend heavily on real-time information. This paper presents an end-to-end framework for constructing and evaluating a conversational AI assistant directly from historical call transcripts. Incoming transcripts are first graded using a simplified adaptation of the PIPA framework, focusing on observation alignment and appropriate response behavior, and are filtered to retain only high-quality interactions exhibiting coherent flow and effective human agent responses. Structured knowledge is then extracted from curated transcripts using large language models (LLMs) and deployed as the sole grounding source in a Retrieval-Augmented Generation (RAG) pipeline. Assistant behavior is governed through systematic prompt tuning, progressing from monolithic prompts to lean, modular, and governed designs that ensure consistency, safety, and controllable execution. Evaluation is conducted using a transcript-grounded user simulator, enabling quantitative measurement of call coverage, factual accuracy, and human escalation behavior. Additional red teaming assesses robustness against prompt injection, out-of-scope, and out-of-context attacks. Experiments are conducted in the Real Estate and Specialist Recruitment domains, which are intentionally challenging and currently suboptimal for automation due to their reliance on real-time data. Despite these constraints, the assistant autonomously handles approximately 30 percents of calls, achieves near-perfect factual accuracy and rejection behavior, and demonstrates strong robustness under adversarial testing. 

---
# State Design Matters: How Representations Shape Dynamic Reasoning in Large Language Models 

**Authors**: Annie Wong, Aske Plaat, Thomas Bäck, Niki van Stein, Anna V. Kononova  

**Link**: [PDF](https://arxiv.org/pdf/2602.15858)  

**Abstract**: As large language models (LLMs) move from static reasoning tasks toward dynamic environments, their success depends on the ability to navigate and respond to an environment that changes as they interact at inference time. An underexplored factor in these settings is the representation of the state. Holding model parameters fixed, we systematically vary three key aspects: (1) state granularity (long form versus summary), (2) structure (natural language versus symbolic), and (3) spatial grounding (text-only versus images or textual map encodings) across sequential decision-making benchmarks. We find that trajectory summarisation improves performance by reducing noise and stabilising long-horizon reasoning. Second, natural language representations are the most robust across models, whereas structured encodings help mainly for models with strong code or structured output priors, such as JSON schemas. Third, while image-inputs show some benefit, text-based spatial encodings prove most effective. This advantage stems not from the spatial information itself, but from the act of construction, which compels the model to perform the spatial reasoning that static input does not elicit. Overall, we demonstrate that design choices for representing state are a decisive factor in performance, distinct from the availability of information itself. We note, however, that even with improved representations, current LLMs and VLMs remain brittle over long horizons, particularly when they must synthesise information to manage multiple subtasks to reach a goal. 

---
# Multi-source Heterogeneous Public Opinion Analysis via Collaborative Reasoning and Adaptive Fusion: A Systematically Integrated Approach 

**Authors**: Yi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15857)  

**Abstract**: The analysis of public opinion from multiple heterogeneous sources presents significant challenges due to structural differences, semantic variations, and platform-specific biases. This paper introduces a novel Collaborative Reasoning and Adaptive Fusion (CRAF) framework that systematically integrates traditional feature-based methods with large language models (LLMs) through a structured multi-stage reasoning mechanism. Our approach features four key innovations: (1) a cross-platform collaborative attention module that aligns semantic representations while preserving source-specific characteristics, (2) a hierarchical adaptive fusion mechanism that dynamically weights features based on both data quality and task requirements, (3) a joint optimization strategy that simultaneously learns topic representations and sentiment distributions through shared latent spaces, and (4) a novel multimodal extraction capability that processes video content from platforms like Douyin and Kuaishou by integrating OCR, ASR, and visual sentiment analysis. Theoretical analysis demonstrates that CRAF achieves a tighter generalization bound with a reduction of O(sqrt(d log K / m)) compared to independent source modeling, where d is feature dimensionality, K is the number of sources, and m is sample size. Comprehensive experiments on three multi-platform datasets (Weibo-12, CrossPlatform-15, NewsForum-8) show that CRAF achieves an average topic clustering ARI of 0.76 (4.1% improvement over best baseline) and sentiment analysis F1-score of 0.84 (3.8% improvement). The framework exhibits strong cross-platform adaptability, reducing the labeled data requirement for new platforms by 75%. 

---
# Rethinking Soft Compression in Retrieval-Augmented Generation: A Query-Conditioned Selector Perspective 

**Authors**: Yunhao Liu, Zian Jia, Xinyu Gao, Kanjun Xu, Yun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2602.15856)  

**Abstract**: Retrieval-Augmented Generation (RAG) effectively grounds Large Language Models (LLMs) with external knowledge and is widely applied to Web-related tasks. However, its scalability is hindered by excessive context length and redundant retrievals. Recent research on soft context compression aims to address this by encoding long documents into compact embeddings, yet they often underperform non-compressed RAG due to their reliance on auto-encoder-like full-compression that forces the encoder to compress all document information regardless of relevance to the input query.
In this work, we conduct an analysis on this paradigm and reveal two fundamental limitations: (I) Infeasibility, full-compression conflicts with the LLM's downstream generation behavior; and (II) Non-necessity: full-compression is unnecessary and dilutes task-relevant information density. Motivated by these insights, we introduce SeleCom, a selector-based soft compression framework for RAG that redefines the encoder's role as query-conditioned information selector. The selector is decoder-only and is trained with a massive, diverse and difficulty-graded synthetic QA dataset with curriculum learning.
Extensive experiments show that SeleCom significantly outperforms existing soft compression approaches and achieves competitive or superior performance to non-compression baselines, while reducing computation and latency by 33.8%~84.6%. 

---
# Decoupling Strategy and Execution in Task-Focused Dialogue via Goal-Oriented Preference Optimization 

**Authors**: Jingyi Xu, Xingyu Ren, Zhiqiang You, Yumeng Zhang, Zhoupeng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2602.15854)  

**Abstract**: Large language models show potential in task-oriented dialogue systems, yet existing training methods often rely on token-level likelihood or preference optimization, which poorly align with long-horizon task success. To address this, we propose Goal-Oriented Preference Optimization (GOPO), a hierarchical reinforcement learning framework that decouples strategy planning from response generation via an Expert Agent and a Customer Service Agent. The Expert Agent optimizes multi-turn goal preferences at the dialogue-trajectory level, while the Customer Service Agent generates responses strictly aligned with the selected strategy. We evaluate GOPO on public benchmarks and e-commerce customer service datasets, and introduce Task-focused Sequential Engagement (TSE), a sequence-level metric derived from real e-commerce interaction data. On the Mgshop dataset, GOPO improves TSE by 7.7% and 10.3% over PPO and Memento, with consistent gains in sequence-level reward and generation quality. Furthermore, a 14B model trained with GOPO achieves 2.7% and 1.5% higher TSE than Qwen-235B and GPT-5.2, respectively. Ablation studies confirm the Expert Agent's critical role in long-horizon optimization. GOPO demonstrates consistent improvements across other datasets as well. This work establishes a new paradigm for task-oriented dialogue systems in commercial scenarios, with code and datasets to be made public. 

---
# A Lightweight Explainable Guardrail for Prompt Safety 

**Authors**: Md Asiful Islam, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15853)  

**Abstract**: We propose a lightweight explainable guardrail (LEG) method for the classification of unsafe prompts. LEG uses a multi-task learning architecture to jointly learn a prompt classifier and an explanation classifier, where the latter labels prompt words that explain the safe/unsafe overall decision. LEG is trained using synthetic data for explainability, which is generated using a novel strategy that counteracts the confirmation biases of LLMs. Lastly, LEG's training process uses a novel loss that captures global explanation signals and combines cross-entropy and focal losses with uncertainty-based weighting. LEG obtains equivalent or better performance than the state-of-the-art for both prompt classification and explainability, both in-domain and out-of-domain on three datasets, despite the fact that its model size is considerably smaller than current approaches. If accepted, we will release all models and the annotated dataset publicly. 

---
# Building Safe and Deployable Clinical Natural Language Processing under Temporal Leakage Constraints 

**Authors**: Ha Na Cho, Sairam Sutari, Alexander Lopez, Hansen Bow, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.15852)  

**Abstract**: Clinical natural language processing (NLP) models have shown promise for supporting hospital discharge planning by leveraging narrative clinical documentation. However, note-based models are particularly vulnerable to temporal and lexical leakage, where documentation artifacts encode future clinical decisions and inflate apparent predictive performance. Such behavior poses substantial risks for real-world deployment, where overconfident or temporally invalid predictions can disrupt clinical workflows and compromise patient safety. This study focuses on system-level design choices required to build safe and deployable clinical NLP under temporal leakage constraints. We present a lightweight auditing pipeline that integrates interpretability into the model development process to identify and suppress leakage-prone signals prior to final training. Using next-day discharge prediction after elective spine surgery as a case study, we evaluate how auditing affects predictive behavior, calibration, and safety-relevant trade-offs. Results show that audited models exhibit more conservative and better-calibrated probability estimates, with reduced reliance on discharge-related lexical cues. These findings emphasize that deployment-ready clinical NLP systems should prioritize temporal validity, calibration, and behavioral robustness over optimistic performance. 

---
# Narrative Theory-Driven LLM Methods for Automatic Story Generation and Understanding: A Survey 

**Authors**: David Y. Liu, Aditya Joshi, Paul Dawson  

**Link**: [PDF](https://arxiv.org/pdf/2602.15851)  

**Abstract**: Applications of narrative theories using large language models (LLMs) deliver promising use-cases in automatic story generation and understanding tasks. Our survey examines how natural language processing (NLP) research engages with fields of narrative studies, and proposes a taxonomy for ongoing efforts that reflect established distinctions in narratology. We discover patterns in the following: narrative datasets and tasks, narrative theories and NLP pipeline and methodological trends in prompting and fine-tuning. We highlight how LLMs enable easy connections of NLP pipelines with abstract narrative concepts and opportunities for interdisciplinary collaboration. Challenges remain in attempts to work towards any unified definition or benchmark of narrative related tasks, making model comparison difficult. For future directions, instead of the pursuit of a single, generalised benchmark for 'narrative quality', we believe that progress benefits more from efforts that focus on the following: defining and improving theory-based metrics for individual narrative attributes to incrementally improve model performance; conducting large-scale, theory-driven literary/social/cultural analysis; and creating experiments where outputs can be used to validate or refine narrative theories. This work provides a contextual foundation for more systematic and theoretically informed narrative research in NLP by providing an overview to ongoing research efforts and the broader narrative studies landscape. 

---
# Large Language Models for Assisting American College Applications 

**Authors**: Zhengliang Liu, Weihang You, Peng Shu, Junhao Chen, Yi Pan, Hanqi Jiang, Yiwei Li, Zhaojun Ding, Chao Cao, Xinliang Li, Yifan Zhou, Ruidong Zhang, Shaochen Xu, Wei Ruan, Huaqin Zhao, Dajiang Zhu, Tianming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15850)  

**Abstract**: American college applications require students to navigate fragmented admissions policies, repetitive and conditional forms, and ambiguous questions that often demand cross-referencing multiple sources. We present EZCollegeApp, a large language model (LLM)-powered system that assists high-school students by structuring application forms, grounding suggested answers in authoritative admissions documents, and maintaining full human control over final responses. The system introduces a mapping-first paradigm that separates form understanding from answer generation, enabling consistent reasoning across heterogeneous application portals. EZCollegeApp integrates document ingestion from official admissions websites, retrieval-augmented question answering, and a human-in-the-loop chatbot interface that presents suggestions alongside application fields without automated submission. We describe the system architecture, data pipeline, internal representations, security and privacy measures, and evaluation through automated testing and human quality assessment. Our source code is released on GitHub (this https URL) to facilitate the broader impact of this work. 

---
# Preference Optimization for Review Question Generation Improves Writing Quality 

**Authors**: Karun Sharma, Vidushee Vats, Shengzhi Li, Yuxiang Wang, Zhongtian Sun, Prayag Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2602.15849)  

**Abstract**: Peer review relies on substantive, evidence-based questions, yet existing LLM-based approaches often generate surface-level queries, drawing over 50\% of their question tokens from a paper's first page. To bridge this gap, we develop IntelliReward, a novel reward model built from a frozen autoregressive LLM with trainable multi-head transformers over the final 50 token states, which outperforms API-based SFT baselines in predicting expert-level human preferences. By applying Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) with IntelliReward, we train IntelliAsk, a question-generation model aligned with human standards of effort, evidence, and grounding. We find consistent improvements on reasoning and writing benchmarks, suggesting reviewer-question quality correlates with broader capabilities. Compared to the Qwen3-32B base model, IntelliAsk shows measurable gains across diverse benchmarks, specifically improving performance on reasoning tasks like MuSR (68.3 vs 64.7 Acc) and complex writing evaluations such as WritingBench (8.31 vs 8.07). We release our implementation, expert preference annotations, and the IntelliReward model to provide an automatic evaluation benchmark for grounding, effort, and evidence in LLM-generated review questions. 

---
# Can LLMs Assess Personality? Validating Conversational AI for Trait Profiling 

**Authors**: Andrius Matšenas, Anet Lello, Tõnis Lees, Hans Peep, Kim Lilii Tamm  

**Link**: [PDF](https://arxiv.org/pdf/2602.15848)  

**Abstract**: This study validates Large Language Models (LLMs) as a dynamic alternative to questionnaire-based personality assessment. Using a within-subjects experiment (N=33), we compared Big Five personality scores derived from guided LLM conversations against the gold-standard IPIP-50 questionnaire, while also measuring user-perceived accuracy. Results indicate moderate convergent validity (r=0.38-0.58), with Conscientiousness, Openness, and Neuroticism scores statistically equivalent between methods. Agreeableness and Extraversion showed significant differences, suggesting trait-specific calibration is needed. Notably, participants rated LLM-generated profiles as equally accurate as traditional questionnaire results. These findings suggest conversational AI offers a promising new approach to traditional psychometrics. 

---
# Do Personality Traits Interfere? Geometric Limitations of Steering in Large Language Models 

**Authors**: Pranav Bhandari, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2602.15847)  

**Abstract**: Personality steering in large language models (LLMs) commonly relies on injecting trait-specific steering vectors, implicitly assuming that personality traits can be controlled independently. In this work, we examine whether this assumption holds by analysing the geometric relationships between Big Five personality steering directions. We study steering vectors extracted from two model families (LLaMA-3-8B and Mistral-8B) and apply a range of geometric conditioning schemes, from unconstrained directions to soft and hard orthonormalisation. Our results show that personality steering directions exhibit substantial geometric dependence: steering one trait consistently induces changes in others, even when linear overlap is explicitly removed. While hard orthonormalisation enforces geometric independence, it does not eliminate cross-trait behavioural effects and can reduce steering strength. These findings suggest that personality traits in LLMs occupy a slightly coupled subspace, limiting fully independent trait control. 

---
# Gated Tree Cross-attention for Checkpoint-Compatible Syntax Injection in Decoder-Only LLMs 

**Authors**: Xinyu Gao, Shaonan Wang, Nai Ding  

**Link**: [PDF](https://arxiv.org/pdf/2602.15846)  

**Abstract**: Decoder-only large language models achieve strong broad performance but are brittle to minor grammatical perturbations, undermining reliability for downstream reasoning. However, directly injecting explicit syntactic structure into an existing checkpoint can interfere with its pretrained competence. We introduce a checkpoint-compatible gated tree cross-attention (GTCA) branch that reads precomputed constituency chunk memory while leaving backbone architecture unchanged. Our design uses a token update mask and staged training to control the scope and timing of structural updates. Across benchmarks and Transformer backbones, GTCA strengthens syntactic robustness beyond continued-training baselines without compromising Multiple-Choice QA performance or commonsense reasoning, providing a practical checkpoint-compatible route to more syntax-robust decoder-only LLMs. 

---
# KD4MT: A Survey of Knowledge Distillation for Machine Translation 

**Authors**: Ona de Gibert, Joseph Attieh, Timothee Mickus, Yves Scherrer, Jörg Tiedemann  

**Link**: [PDF](https://arxiv.org/pdf/2602.15845)  

**Abstract**: Knowledge Distillation (KD) as a research area has gained a lot of traction in recent years as a compression tool to address challenges related to ever-larger models in NLP. Remarkably, Machine Translation (MT) offers a much more nuanced take on this narrative: in MT, KD also functions as a general-purpose knowledge transfer mechanism that shapes supervision and translation quality as well as efficiency.
This survey synthesizes KD for MT (KD4MT) across 105 papers (through October 1, 2025). We begin by introducing both MT and KD for non-experts, followed by an overview of the standard KD approaches relevant to MT applications. Subsequently, we categorize advances in the KD4MT literature based on (i) their methodological contributions and (ii) their practical applications. Our qualitative and quantitative analyses identify common trends in the field and highlight key research gaps as well as the absence of unified evaluation practice for KD methods in MT. We further provide practical guidelines for selecting a KD method in concrete settings and highlight potential risks associated with the application of KD to MT such as increased hallucination and bias amplification. Finally, we discuss the role of LLMs in re-shaping the KD4MT field. To support further research, we complement our survey with a publicly available database summarizing the main characteristics of the surveyed KD methods and a glossary of key terms. 

---
# Language Model Representations for Efficient Few-Shot Tabular Classification 

**Authors**: Inwon Kang, Parikshit Ram, Yi Zhou, Horst Samulowitz, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2602.15844)  

**Abstract**: The Web is a rich source of structured data in the form of tables, from product catalogs and knowledge bases to scientific datasets. However, the heterogeneity of the structure and semantics of these tables makes it challenging to build a unified method that can effectively leverage the information they contain. Meanwhile, Large language models (LLMs) are becoming an increasingly integral component of web infrastructure for tasks like semantic search. This raises a crucial question: can we leverage these already-deployed LLMs to classify structured data in web-native tables (e.g., product catalogs, knowledge base exports, scientific data portals), avoiding the need for specialized models or extensive retraining? This work investigates a lightweight paradigm, $\textbf{Ta}$ble $\textbf{R}$epresentation with $\textbf{L}$anguage Model~($\textbf{TaRL}$), for few-shot tabular classification that directly utilizes semantic embeddings of individual table rows. We first show that naive application of these embeddings underperforms compared to specialized tabular models. We then demonstrate that their potentials can be unlocked with two key techniques: removing the common component from all embeddings and calibrating the softmax temperature. We show that a simple meta-learner, trained on handcrafted features, can learn to predict an appropriate temperature. This approach achieves performance comparable to state-of-the-art models in low-data regimes ($k \leq 32$) of semantically-rich tables. Our findings demonstrate the viability of reusing existing LLM infrastructure for efficient semantics-driven pathway to reuse existing LLM infrastructure for Web table understanding. 

---
# The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts 

**Authors**: Warren Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2602.15843)  

**Abstract**: In "Compress or Route?" (Johnson, 2026), we found that code generation tolerates aggressive prompt compression (r >= 0.6) while chain-of-thought reasoning degrades gradually. That study was limited to HumanEval (164 problems), left the "perplexity paradox" mechanism unvalidated, and provided no adaptive algorithm. This paper addresses all three gaps. First, we validate across six code benchmarks (HumanEval, MBPP, HumanEval+, MultiPL-E) and four reasoning benchmarks (GSM8K, MATH, ARC-Challenge, MMLU-STEM), confirming the compression threshold generalizes across languages and difficulties. Second, we conduct the first per-token perplexity analysis (n=723 tokens), revealing a "perplexity paradox": code syntax tokens are preserved (high perplexity) while numerical values in math problems are pruned despite being task-critical (low perplexity). Signature injection recovers +34 percentage points in pass rate (5.3% to 39.3%; Cohen's h=0.890). Third, we propose TAAC (Task-Aware Adaptive Compression), achieving 22% cost reduction with 96% quality preservation, outperforming fixed-ratio compression by 7%. MBPP validation (n=1,800 trials) confirms systematic variation: 3.6% at r=0.3 to 54.6% at r=1.0. 

---
# Scaling Open Discrete Audio Foundation Models with Interleaved Semantic, Acoustic, and Text Tokens 

**Authors**: Potsawee Manakul, Woody Haosheng Gan, Martijn Bartelds, Guangzhi Sun, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16687)  

**Abstract**: Current audio language models are predominantly text-first, either extending pre-trained text LLM backbones or relying on semantic-only audio tokens, limiting general audio modeling. This paper presents a systematic empirical study of native audio foundation models that apply next-token prediction to audio at scale, jointly modeling semantic content, acoustic details, and text to support both general audio generation and cross-modal capabilities. We provide comprehensive empirical insights for building such models: (1) We systematically investigate design choices -- data sources, text mixture ratios, and token composition -- establishing a validated training recipe. (2) We conduct the first scaling law study for discrete audio models via IsoFLOP analysis on 64 models spanning $3{\times}10^{18}$ to $3{\times}10^{20}$ FLOPs, finding that optimal data grows 1.6$\times$ faster than optimal model size. (3) We apply these lessons to train SODA (Scaling Open Discrete Audio), a suite of models from 135M to 4B parameters on 500B tokens, comparing against our scaling predictions and existing models. SODA serves as a flexible backbone for diverse audio/text tasks -- we demonstrate this by fine-tuning for voice-preserving speech-to-speech translation, using the same unified architecture. 

---
# Creating a digital poet 

**Authors**: Vered Tohar, Tsahi Hayat, Amir Leshem  

**Link**: [PDF](https://arxiv.org/pdf/2602.16578)  

**Abstract**: Can a machine write good poetry? Any positive answer raises fundamental questions about the nature and value of art. We report a seven-month poetry workshop in which a large language model was shaped into a digital poet through iterative in-context expert feedback, without retraining. Across sessions, the model developed a distinctive style and a coherent corpus, supported by quantitative and qualitative analyses, and it produced a pen name and author image. In a blinded authorship test with 50 humanities students and graduates (three AI poems and three poems by well-known poets each), judgments were at chance: human poems were labeled human 54% of the time and AI poems 52%, with 95% confidence intervals including 50%. After the workshop, a commercial publisher released a poetry collection authored by the model. These results show that workshop-style prompting can support long-horizon creative shaping and renew debates on creativity and authorship. 

---
# Variable-Length Semantic IDs for Recommender Systems 

**Authors**: Kirill Khrylchenko  

**Link**: [PDF](https://arxiv.org/pdf/2602.16375)  

**Abstract**: Generative models are increasingly used in recommender systems, both for modeling user behavior as event sequences and for integrating large language models into recommendation pipelines. A key challenge in this setting is the extremely large cardinality of item spaces, which makes training generative models difficult and introduces a vocabulary gap between natural language and item identifiers. Semantic identifiers (semantic IDs), which represent items as sequences of low-cardinality tokens, have recently emerged as an effective solution to this problem.
However, existing approaches generate semantic identifiers of fixed length, assigning the same description length to all items. This is inefficient, misaligned with natural language, and ignores the highly skewed frequency structure of real-world catalogs, where popular items and rare long-tail items exhibit fundamentally different information requirements. In parallel, the emergent communication literature studies how agents develop discrete communication protocols, often producing variable-length messages in which frequent concepts receive shorter descriptions. Despite the conceptual similarity, these ideas have not been systematically adopted in recommender systems.
In this work, we bridge recommender systems and emergent communication by introducing variable-length semantic identifiers for recommendation. We propose a discrete variational autoencoder with Gumbel-Softmax reparameterization that learns item representations of adaptive length under a principled probabilistic framework, avoiding the instability of REINFORCE-based training and the fixed-length constraints of prior semantic ID methods. 

---
# Lyapunov Spectral Analysis of Speech Embedding Trajectories in Psychosis 

**Authors**: Jelena Vasic, Branislav Andjelic, Ana Mancic, Dusica Filipovic Djurdjevic, Ljiljana Mihic, Aleksandar Kovacevic, Nadja P. Maric, Aleksandra Maluckov  

**Link**: [PDF](https://arxiv.org/pdf/2602.16273)  

**Abstract**: We analyze speech embeddings from structured clinical interviews of psychotic patients and healthy controls by treating language production as a high-dimensional dynamical process. Lyapunov exponent (LE) spectra are computed from word-level and answer-level embeddings generated by two distinct large language models, allowing us to assess the stability of the conclusions with respect to different embedding presentations. Word-level embeddings exhibit uniformly contracting dynamics with no positive LE, while answer-level embeddings, in spite of the overall contraction, display a number of positive LEs and higher-dimensional attractors. The resulting LE spectra robustly separate psychotic from healthy speech, while differentiation within the psychotic group is not statistically significant overall, despite a tendency of the most severe cases to occupy distinct dynamical regimes. These findings indicate that nonlinear dynamical invariants of speech embeddings provide a physics-inspired probe of disordered cognition whose conclusions remain stable across embedding models. 

---
# ModalImmune: Immunity Driven Unlearning via Self Destructive Training 

**Authors**: Rong Fu, Jia Yee Tan, Wenxin Zhang, Zijian Zhang, Ziming Wang, Zhaolu Kang, Muge Qi, Shuning Zhang, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2602.16197)  

**Abstract**: Multimodal systems are vulnerable to partial or complete loss of input channels at deployment, which undermines reliability in real-world settings. This paper presents ModalImmune, a training framework that enforces modality immunity by intentionally and controllably collapsing selected modality information during training so the model learns joint representations that are robust to destructive modality influence. The framework combines a spectrum-adaptive collapse regularizer, an information-gain guided controller for targeted interventions, curvature-aware gradient masking to stabilize destructive updates, and a certified Neumann-truncated hyper-gradient procedure for automatic meta-parameter adaptation. Empirical evaluation on standard multimodal benchmarks demonstrates that ModalImmune improves resilience to modality removal and corruption while retaining convergence stability and reconstruction capacity. 

---
# Learning Personalized Agents from Human Feedback 

**Authors**: Kaiqu Liang, Julia Kruk, Shengyi Qian, Xianjun Yang, Shengjie Bi, Yuanshun Yao, Shaoliang Nie, Mingyang Zhang, Lijuan Liu, Jaime Fernández Fisac, Shuyan Zhou, Saghar Hosseini  

**Link**: [PDF](https://arxiv.org/pdf/2602.16173)  

**Abstract**: Modern AI agents are powerful but often fail to align with the idiosyncratic, evolving preferences of individual users. Prior approaches typically rely on static datasets, either training implicit preference models on interaction history or encoding user profiles in external memory. However, these approaches struggle with new users and with preferences that change over time. We introduce Personalized Agents from Human Feedback (PAHF), a framework for continual personalization in which agents learn online from live interaction using explicit per-user memory. PAHF operationalizes a three-step loop: (1) seeking pre-action clarification to resolve ambiguity, (2) grounding actions in preferences retrieved from memory, and (3) integrating post-action feedback to update memory when preferences drift. To evaluate this capability, we develop a four-phase protocol and two benchmarks in embodied manipulation and online shopping. These benchmarks quantify an agent's ability to learn initial preferences from scratch and subsequently adapt to persona shifts. Our theoretical analysis and empirical results show that integrating explicit memory with dual feedback channels is critical: PAHF learns substantially faster and consistently outperforms both no-memory and single-channel baselines, reducing initial personalization error and enabling rapid adaptation to preference shifts. 

---
# Discrete Stochastic Localization for Non-autoregressive Generation 

**Authors**: Yunshu Wu, Jiayi Cheng, Partha Thakuria, Rob Brekelmans, Evangelos E. Papalexakis, Greg Ver Steeg  

**Link**: [PDF](https://arxiv.org/pdf/2602.16169)  

**Abstract**: Non-autoregressive (NAR) generation reduces decoding latency by predicting many tokens in parallel, but iterative refinement often suffers from error accumulation and distribution shift under self-generated drafts. Masked diffusion language models (MDLMs) and their remasking samplers (e.g., ReMDM) can be viewed as modern NAR iterative refinement, where generation repeatedly revises a partially observed draft. In this work we show that \emph{training alone} can substantially improve the step-efficiency of MDLM/ReMDM sampling. We propose \textsc{DSL} (Discrete Stochastic Localization), which trains a single SNR-invariant denoiser across a continuum of corruption levels, bridging intermediate draft noise and mask-style endpoint corruption within one Diffusion Transformer. On OpenWebText, \textsc{DSL} fine-tuning yields large MAUVE gains at low step budgets, surpassing the MDLM+ReMDM baseline with \(\sim\)4$\times$ fewer denoiser evaluations, and matches autoregressive quality at high budgets. Analyses show improved self-correction and uncertainty calibration, making remasking markedly more compute-efficient. 

---
# Emotion Collider: Dual Hyperbolic Mirror Manifolds for Sentiment Recovery via Anti Emotion Reflection 

**Authors**: Rong Fu, Ziming Wang, Shuo Yin, Wenxin Zhang, Haiyun Wei, Kun Liu, Xianda Li, Zeli Su, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2602.16161)  

**Abstract**: Emotional expression underpins natural communication and effective human-computer interaction. We present Emotion Collider (EC-Net), a hyperbolic hypergraph framework for multimodal emotion and sentiment modeling. EC-Net represents modality hierarchies using Poincare-ball embeddings and performs fusion through a hypergraph mechanism that passes messages bidirectionally between nodes and hyperedges. To sharpen class separation, contrastive learning is formulated in hyperbolic space with decoupled radial and angular objectives. High-order semantic relations across time steps and modalities are preserved via adaptive hyperedge construction. Empirical results on standard multimodal emotion benchmarks show that EC-Net produces robust, semantically coherent representations and consistently improves accuracy, particularly when modalities are partially available or contaminated by noise. These findings indicate that explicit hierarchical geometry combined with hypergraph fusion is effective for resilient multimodal affect understanding. 

---
# Why Any-Order Autoregressive Models Need Two-Stream Attention: A Structural-Semantic Tradeoff 

**Authors**: Patrick Pynadath, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.16092)  

**Abstract**: Any-order autoregressive models (AO-ARMs) offer a promising path toward efficient masked diffusion by enabling native key-value caching, but competitive performance has so far required two-stream attention, typically motivated as a means of decoupling token content from position. In this work, we argue that two-stream attention may be serving a more subtle role. We identify a structural-semantic tradeoff in any-order generation: the hidden representation at each step must simultaneously attend to semantically informative tokens for prediction and structurally recent tokens for summarization, objectives that compete for attention capacity in a single stream but can specialize across two streams. To isolate this tradeoff from position-content separation, we propose Decoupled RoPE, a modification to rotary position embeddings that provides target position information without revealing target content. Decoupled RoPE performs competitively at short sequence lengths--where semantic and structural proximity coincide--but degrades as sequence length increases and the two orderings diverge. These results suggest that the success of two-stream attention stems not merely from separating position from content, but from circumventing the deeper structural-semantic tradeoff inherent to any-order generation. 

---
# Evidence-Grounded Subspecialty Reasoning: Evaluating a Curated Clinical Intelligence Layer on the 2025 Endocrinology Board-Style Examination 

**Authors**: Amir Hosseinian, MohammadReza Zare Shahneh, Umer Mansoor, Gilbert Szeto, Kirill Karlin, Nima Aghaeepour  

**Link**: [PDF](https://arxiv.org/pdf/2602.16050)  

**Abstract**: Background: Large language models have demonstrated strong performance on general medical examinations, but subspecialty clinical reasoning remains challenging due to rapidly evolving guidelines and nuanced evidence hierarchies. Methods: We evaluated January Mirror, an evidence-grounded clinical reasoning system, against frontier LLMs (GPT-5, GPT-5.2, Gemini-3-Pro) on a 120-question endocrinology board-style examination. Mirror integrates a curated endocrinology and cardiometabolic evidence corpus with a structured reasoning architecture to generate evidence-linked outputs. Mirror operated under a closed-evidence constraint without external retrieval. Comparator LLMs had real-time web access to guidelines and primary literature. Results: Mirror achieved 87.5% accuracy (105/120; 95% CI: 80.4-92.3%), exceeding a human reference of 62.3% and frontier LLMs including GPT-5.2 (74.6%), GPT-5 (74.0%), and Gemini-3-Pro (69.8%). On the 30 most difficult questions (human accuracy less than 50%), Mirror achieved 76.7% accuracy. Top-2 accuracy was 92.5% for Mirror versus 85.25% for GPT-5.2. Conclusions: Mirror provided evidence traceability: 74.2% of outputs cited at least one guideline-tier source, with 100% citation accuracy on manual verification. Curated evidence with explicit provenance can outperform unconstrained web retrieval for subspecialty clinical reasoning and supports auditability for clinical deployment. 

---
# MAEB: Massive Audio Embedding Benchmark 

**Authors**: Adnan El Assadi, Isaac Chung, Chenghao Xiao, Roman Solomatin, Animesh Jha, Rahul Chand, Silky Singh, Kaitlyn Wang, Ali Sartaz Khan, Marc Moussa Nasser, Sufen Fong, Pengfei He, Alan Xiao, Ayush Sunil Munot, Aditya Shrivastava, Artem Gazizov, Niklas Muennighoff, Kenneth Enevoldsen  

**Link**: [PDF](https://arxiv.org/pdf/2602.16008)  

**Abstract**: We introduce the Massive Audio Embedding Benchmark (MAEB), a large-scale benchmark covering 30 tasks across speech, music, environmental sounds, and cross-modal audio-text reasoning in 100+ languages. We evaluate 50+ models and find that no single model dominates across all tasks: contrastive audio-text models excel at environmental sound classification (e.g., ESC50) but score near random on multilingual speech tasks (e.g., SIB-FLEURS), while speech-pretrained models show the opposite pattern. Clustering remains challenging for all models, with even the best-performing model achieving only modest results. We observe that models excelling on acoustic understanding often perform poorly on linguistic tasks, and vice versa. We also show that the performance of audio encoders on MAEB correlates highly with their performance when used in audio large language models. MAEB is derived from MAEB+, a collection of 98 tasks. MAEB is designed to maintain task diversity while reducing evaluation cost, and it integrates into the MTEB ecosystem for unified evaluation across text, image, and audio modalities. We release MAEB and all 98 tasks along with code and a leaderboard at this https URL. 

---
# Anatomy of Capability Emergence: Scale-Invariant Representation Collapse and Top-Down Reorganization in Neural Networks 

**Authors**: Jayadev Billa  

**Link**: [PDF](https://arxiv.org/pdf/2602.15997)  

**Abstract**: Capability emergence during neural network training remains mechanistically opaque. We track five geometric measures across five model scales (405K-85M parameters), 120+ emergence events in eight algorithmic tasks, and three Pythia language models (160M-2.8B). We find: (1) training begins with a universal representation collapse to task-specific floors that are scale-invariant across a 210X parameter range (e.g., modular arithmetic collapses to RANKME ~ 2.0 regardless of model size); (2) collapse propagates top-down through layers (32/32 task X model consistency), contradicting bottom-up feature-building intuition; (3) a geometric hierarchy in which representation geometry leads emergence (75-100% precursor rate for hard tasks), while the local learning coefficient is synchronous (0/24 precursor) and Hessian measures lag. We also delineate prediction limits: geometric measures encode coarse task difficulty but not fine-grained timing (within-class concordance 27%; when task ordering reverses across scales, prediction fails at 26%). On Pythia, global geometric patterns replicate but per-task precursor signals do not -- the precursor relationship requires task-training alignment that naturalistic pre-training does not provide. Our contribution is the geometric anatomy of emergence and its boundary conditions, not a prediction tool. 

---
# Evidence for Daily and Weekly Periodic Variability in GPT-4o Performance 

**Authors**: Paul Tschisgale, Peter Wulff  

**Link**: [PDF](https://arxiv.org/pdf/2602.15889)  

**Abstract**: Large language models (LLMs) are increasingly used in research both as tools and as objects of investigation. Much of this work implicitly assumes that LLM performance under fixed conditions (identical model snapshot, hyperparameters, and prompt) is time-invariant. If average output quality changes systematically over time, this assumption is violated, threatening the reliability, validity, and reproducibility of findings. To empirically examine this assumption, we conducted a longitudinal study on the temporal variability of GPT-4o's average performance. Using a fixed model snapshot, fixed hyperparameters, and identical prompting, GPT-4o was queried via the API to solve the same multiple-choice physics task every three hours for approximately three months. Ten independent responses were generated at each time point and their scores were averaged. Spectral (Fourier) analysis of the resulting time series revealed notable periodic variability in average model performance, accounting for approximately 20% of the total variance. In particular, the observed periodic patterns are well explained by the interaction of a daily and a weekly rhythm. These findings indicate that, even under controlled conditions, LLM performance may vary periodically over time, calling into question the assumption of time invariance. Implications for ensuring validity and replicability of research that uses or investigates LLMs are discussed. 

---
# AI as Teammate or Tool? A Review of Human-AI Interaction in Decision Support 

**Authors**: Most. Sharmin Sultana Samu, Nafisa Khan, Kazi Toufique Elahi, Tasnuva Binte Rahman, Md. Rakibul Islam, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2602.15865)  

**Abstract**: The integration of Artificial Intelligence (AI) necessitates determining whether systems function as tools or collaborative teammates. In this study, by synthesizing Human-AI Interaction (HAI) literature, we analyze this distinction across four dimensions: interaction design, trust calibration, collaborative frameworks and healthcare applications. Our analysis reveals that static interfaces and miscalibrated trust limit AI efficacy. Performance hinges on aligning transparency with cognitive workflows, yet a fluency trap often inflates trust without improving decision-making. Consequently, an overemphasis on explainability leaves systems largely passive. Our findings show that current AI systems remain largely passive due to an overreliance on explainability-centric designs and that transitioning AI to an active teammate requires adaptive, context-aware interactions that support shared mental models and the dynamic negotiation of authority between humans and AI. 

---
# Memes-as-Replies: Can Models Select Humorous Manga Panel Responses? 

**Authors**: Ryosuke Kohita, Seiichiro Yoshioka  

**Link**: [PDF](https://arxiv.org/pdf/2602.15842)  

**Abstract**: Memes are a popular element of modern web communication, used not only as static artifacts but also as interactive replies within conversations. While computational research has focused on analyzing the intrinsic properties of memes, the dynamic and contextual use of memes to create humor remains an understudied area of web science. To address this gap, we introduce the Meme Reply Selection task and present MaMe-Re (Manga Meme Reply Benchmark), a benchmark of 100,000 human-annotated pairs (500,000 total annotations from 2,325 unique annotators) consisting of openly licensed Japanese manga panels and social media posts. Our analysis reveals three key insights: (1) large language models (LLMs) show preliminary evidence of capturing complex social cues such as exaggeration, moving beyond surface-level semantic matching; (2) the inclusion of visual information does not improve performance, revealing a gap between understanding visual content and effectively using it for contextual humor; (3) while LLMs can match human judgments in controlled settings, they struggle to distinguish subtle differences in wit among semantically similar candidates. These findings suggest that selecting contextually humorous replies remains an open challenge for current models. 

---
# A Methodology for Identifying Evaluation Items for Practical Dialogue Systems Based on Business-Dialogue System Alignment Models 

**Authors**: Mikio Nakano, Hironori Takeuchi, Kazunori Komatani  

**Link**: [PDF](https://arxiv.org/pdf/2602.15835)  

**Abstract**: This paper proposes a methodology for identifying evaluation items for practical dialogue systems. Traditionally, user satisfaction and user experiences have been the primary metrics for evaluating dialogue systems. However, there are various other evaluation items to consider when developing and operating practical dialogue systems, and such evaluation items are expected to lead to new research topics. So far, there has been no methodology for identifying these evaluation items. We propose identifying evaluation items based on business-dialogue system alignment models, which are applications of business-IT alignment models used in the development and operation of practical IT systems. We also present a generic model that facilitates the construction of a business-dialogue system alignment model for each dialogue system. 

---
