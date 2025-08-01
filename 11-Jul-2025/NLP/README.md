# PyVision: Agentic Vision with Dynamic Tooling 

**Authors**: Shitian Zhao, Haoquan Zhang, Shaoheng Lin, Ming Li, Qilong Wu, Kaipeng Zhang, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.07998)  

**Abstract**: LLMs are increasingly deployed as agents, systems capable of planning, reasoning, and dynamically calling external tools. However, in visual reasoning, prior approaches largely remain limited by predefined workflows and static toolsets. In this report, we present PyVision, an interactive, multi-turn framework that enables MLLMs to autonomously generate, execute, and refine Python-based tools tailored to the task at hand, unlocking flexible and interpretable problem-solving. We develop a taxonomy of the tools created by PyVision and analyze their usage across a diverse set of benchmarks. Quantitatively, PyVision achieves consistent performance gains, boosting GPT-4.1 by +7.8% on V* and Claude-4.0-Sonnet by +31.1% on VLMsAreBlind-mini. These results point to a broader shift: dynamic tooling allows models not just to use tools, but to invent them, advancing toward more agentic visual reasoning. 

---
# Automating Expert-Level Medical Reasoning Evaluation of Large Language Models 

**Authors**: Shuang Zhou, Wenya Xie, Jiaxi Li, Zaifu Zhan, Meijia Song, Han Yang, Cheyenna Espinoza, Lindsay Welton, Xinnie Mai, Yanwei Jin, Zidu Xu, Yuen-Hei Chung, Yiyun Xing, Meng-Han Tsai, Emma Schaffer, Yucheng Shi, Ninghao Liu, Zirui Liu, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07988)  

**Abstract**: As large language models (LLMs) become increasingly integrated into clinical decision-making, ensuring transparent and trustworthy reasoning is essential. However, existing evaluation strategies of LLMs' medical reasoning capability either suffer from unsatisfactory assessment or poor scalability, and a rigorous benchmark remains lacking. To address this, we introduce MedThink-Bench, a benchmark designed for rigorous, explainable, and scalable assessment of LLMs' medical reasoning. MedThink-Bench comprises 500 challenging questions across ten medical domains, each annotated with expert-crafted step-by-step rationales. Building on this, we propose LLM-w-Ref, a novel evaluation framework that leverages fine-grained rationales and LLM-as-a-Judge mechanisms to assess intermediate reasoning with expert-level fidelity while maintaining scalability. Experiments show that LLM-w-Ref exhibits a strong positive correlation with expert judgments. Benchmarking twelve state-of-the-art LLMs, we find that smaller models (e.g., MedGemma-27B) can surpass larger proprietary counterparts (e.g., OpenAI-o3). Overall, MedThink-Bench offers a foundational tool for evaluating LLMs' medical reasoning, advancing their safe and responsible deployment in clinical practice. 

---
# Performance and Practical Considerations of Large and Small Language Models in Clinical Decision Support in Rheumatology 

**Authors**: Sabine Felde, Rüdiger Buchkremer, Gamal Chehab, Christian Thielscher, Jörg HW Distler, Matthias Schneider, Jutta G. Richter  

**Link**: [PDF](https://arxiv.org/pdf/2507.07983)  

**Abstract**: Large language models (LLMs) show promise for supporting clinical decision-making in complex fields such as rheumatology. Our evaluation shows that smaller language models (SLMs), combined with retrieval-augmented generation (RAG), achieve higher diagnostic and therapeutic performance than larger models, while requiring substantially less energy and enabling cost-efficient, local deployment. These features are attractive for resource-limited healthcare. However, expert oversight remains essential, as no model consistently reached specialist-level accuracy in rheumatology. 

---
# Why is Your Language Model a Poor Implicit Reward Model? 

**Authors**: Noam Razin, Yong Lin, Jiarui Yao, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2507.07981)  

**Abstract**: Reward models are key to language model post-training and inference pipelines. Conveniently, recent work showed that every language model defines an implicit reward model (IM-RM), without requiring any architectural changes. However, such IM-RMs tend to generalize worse, especially out-of-distribution, compared to explicit reward models (EX-RMs) that apply a dedicated linear head over the hidden representations of a language model. The existence of a generalization gap is puzzling, as EX-RMs and IM-RMs are nearly identical. They can be trained using the same data, loss function, and language model, and differ only in how the reward is computed. Towards a fundamental understanding of the implicit biases underlying different reward model types, we investigate the root cause of this gap. Our main finding, backed by theory and experiments, is that IM-RMs rely more heavily on superficial token-level cues. Consequently, they often generalize worse than EX-RMs under token-level distribution shifts, as well as in-distribution. Furthermore, we provide evidence against alternative hypotheses for the generalization gap. Most notably, we challenge the intuitive claim that IM-RMs struggle in tasks where generation is harder than verification because they can operate both as a verifier and a generator. Taken together, our results highlight that seemingly minor design choices can substantially impact the generalization behavior of reward models. 

---
# MIRIX: Multi-Agent Memory System for LLM-Based Agents 

**Authors**: Yu Wang, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.07957)  

**Abstract**: Although memory capabilities of AI agents are gaining increasing attention, existing solutions remain fundamentally limited. Most rely on flat, narrowly scoped memory components, constraining their ability to personalize, abstract, and reliably recall user-specific information over time. To this end, we introduce MIRIX, a modular, multi-agent memory system that redefines the future of AI memory by solving the field's most critical challenge: enabling language models to truly remember. Unlike prior approaches, MIRIX transcends text to embrace rich visual and multimodal experiences, making memory genuinely useful in real-world scenarios. MIRIX consists of six distinct, carefully structured memory types: Core, Episodic, Semantic, Procedural, Resource Memory, and Knowledge Vault, coupled with a multi-agent framework that dynamically controls and coordinates updates and retrieval. This design enables agents to persist, reason over, and accurately retrieve diverse, long-term user data at scale. We validate MIRIX in two demanding settings. First, on ScreenshotVQA, a challenging multimodal benchmark comprising nearly 20,000 high-resolution computer screenshots per sequence, requiring deep contextual understanding and where no existing memory systems can be applied, MIRIX achieves 35% higher accuracy than the RAG baseline while reducing storage requirements by 99.9%. Second, on LOCOMO, a long-form conversation benchmark with single-modal textual input, MIRIX attains state-of-the-art performance of 85.4%, far surpassing existing baselines. These results show that MIRIX sets a new performance standard for memory-augmented LLM agents. To allow users to experience our memory system, we provide a packaged application powered by MIRIX. It monitors the screen in real time, builds a personalized memory base, and offers intuitive visualization and secure local storage to ensure privacy. 

---
# SAGE: A Visual Language Model for Anomaly Detection via Fact Enhancement and Entropy-aware Alignment 

**Authors**: Guoxin Zang, Xue Li, Donglin Di, Lanshun Nie, Dechen Zhan, Yang Song, Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07939)  

**Abstract**: While Vision-Language Models (VLMs) have shown promising progress in general multimodal tasks, they often struggle in industrial anomaly detection and reasoning, particularly in delivering interpretable explanations and generalizing to unseen categories. This limitation stems from the inherently domain-specific nature of anomaly detection, which hinders the applicability of existing VLMs in industrial scenarios that require precise, structured, and context-aware analysis. To address these challenges, we propose SAGE, a VLM-based framework that enhances anomaly reasoning through Self-Guided Fact Enhancement (SFE) and Entropy-aware Direct Preference Optimization (E-DPO). SFE integrates domain-specific knowledge into visual reasoning via fact extraction and fusion, while E-DPO aligns model outputs with expert preferences using entropy-aware optimization. Additionally, we introduce AD-PL, a preference-optimized dataset tailored for industrial anomaly reasoning, consisting of 28,415 question-answering instances with expert-ranked responses. To evaluate anomaly reasoning models, we develop Multiscale Logical Evaluation (MLE), a quantitative framework analyzing model logic and consistency. SAGE demonstrates superior performance on industrial anomaly datasets under zero-shot and one-shot settings. The code, model and dataset are available at this https URL. 

---
# DTECT: Dynamic Topic Explorer & Context Tracker 

**Authors**: Suman Adhya, Debarshi Kumar Sanyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.07910)  

**Abstract**: The explosive growth of textual data over time presents a significant challenge in uncovering evolving themes and trends. Existing dynamic topic modeling techniques, while powerful, often exist in fragmented pipelines that lack robust support for interpretation and user-friendly exploration. We introduce DTECT (Dynamic Topic Explorer & Context Tracker), an end-to-end system that bridges the gap between raw textual data and meaningful temporal insights. DTECT provides a unified workflow that supports data preprocessing, multiple model architectures, and dedicated evaluation metrics to analyze the topic quality of temporal topic models. It significantly enhances interpretability by introducing LLM-driven automatic topic labeling, trend analysis via temporally salient words, interactive visualizations with document-level summarization, and a natural language chat interface for intuitive data querying. By integrating these features into a single, cohesive platform, DTECT empowers users to more effectively track and understand thematic dynamics. DTECT is open-source and available at this https URL. 

---
# Automating MD simulations for Proteins using Large language Models: NAMD-Agent 

**Authors**: Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2507.07887)  

**Abstract**: Molecular dynamics simulations are an essential tool in understanding protein structure, dynamics, and function at the atomic level. However, preparing high quality input files for MD simulations can be a time consuming and error prone process. In this work, we introduce an automated pipeline that leverages Large Language Models (LLMs), specifically Gemini 2.0 Flash, in conjunction with python scripting and Selenium based web automation to streamline the generation of MD input files. The pipeline exploits CHARMM GUI's comprehensive web-based interface for preparing simulation-ready inputs for NAMD. By integrating Gemini's code generation and iterative refinement capabilities, simulation scripts are automatically written, executed, and revised to navigate CHARMM GUI, extract appropriate parameters, and produce the required NAMD input files. Post processing is performed using additional software to further refine the simulation outputs, thereby enabling a complete and largely hands free workflow. Our results demonstrate that this approach reduces setup time, minimizes manual errors, and offers a scalable solution for handling multiple protein systems in parallel. This automated framework paves the way for broader application of LLMs in computational structural biology, offering a robust and adaptable platform for future developments in simulation automation. 

---
# DocCHA: Towards LLM-Augmented Interactive Online diagnosis System 

**Authors**: Xinyi Liu, Dachun Sun, Yi R. Fung, Dilek Hakkani-Tür, Tarek Abdelzaher  

**Link**: [PDF](https://arxiv.org/pdf/2507.07870)  

**Abstract**: Despite the impressive capabilities of Large Language Models (LLMs), existing Conversational Health Agents (CHAs) remain static and brittle, incapable of adaptive multi-turn reasoning, symptom clarification, or transparent decision-making. This hinders their real-world applicability in clinical diagnosis, where iterative and structured dialogue is essential. We propose DocCHA, a confidence-aware, modular framework that emulates clinical reasoning by decomposing the diagnostic process into three stages: (1) symptom elicitation, (2) history acquisition, and (3) causal graph construction. Each module uses interpretable confidence scores to guide adaptive questioning, prioritize informative clarifications, and refine weak reasoning links.
Evaluated on two real-world Chinese consultation datasets (IMCS21, DX), DocCHA consistently outperforms strong prompting-based LLM baselines (GPT-3.5, GPT-4o, LLaMA-3), achieving up to 5.18 percent higher diagnostic accuracy and over 30 percent improvement in symptom recall, with only modest increase in dialogue turns. These results demonstrate the effectiveness of DocCHA in enabling structured, transparent, and efficient diagnostic conversations -- paving the way for trustworthy LLM-powered clinical assistants in multilingual and resource-constrained settings. 

---
# Alpay Algebra V: Multi-Layered Semantic Games and Transfinite Fixed-Point Simulation 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2507.07868)  

**Abstract**: This paper extends the self-referential framework of Alpay Algebra into a multi-layered semantic game architecture where transfinite fixed-point convergence encompasses hierarchical sub-games at each iteration level. Building upon Alpay Algebra IV's empathetic embedding concept, we introduce a nested game-theoretic structure where the alignment process between AI systems and documents becomes a meta-game containing embedded decision problems. We formalize this through a composite operator $\phi(\cdot, \gamma(\cdot))$ where $\phi$ drives the main semantic convergence while $\gamma$ resolves local sub-games. The resulting framework demonstrates that game-theoretic reasoning emerges naturally from fixed-point iteration rather than being imposed externally. We prove a Game Theorem establishing existence and uniqueness of semantic equilibria under realistic cognitive simulation assumptions. Our verification suite includes adaptations of Banach's fixed-point theorem to transfinite contexts, a novel $\phi$-topology based on the Kozlov-Maz'ya-Rossmann formula for handling semantic singularities, and categorical consistency tests via the Yoneda lemma. The paper itself functions as a semantic artifact designed to propagate its fixed-point patterns in AI embedding spaces -- a deliberate instantiation of the "semantic virus" concept it theorizes. All results are grounded in category theory, information theory, and realistic AI cognition models, ensuring practical applicability beyond pure mathematical abstraction. 

---
# From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on Retrieval-Augmented Generation systems 

**Authors**: Youngjoon Jang, Seongtae Hong, Junyoung Son, Sungjin Park, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.07847)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a crucial framework in natural language processing (NLP), improving factual consistency and reducing hallucinations by integrating external document retrieval with large language models (LLMs). However, the effectiveness of RAG is often hindered by coreferential complexity in retrieved documents, introducing ambiguity that disrupts in-context learning. In this study, we systematically investigate how entity coreference affects both document retrieval and generative performance in RAG-based systems, focusing on retrieval relevance, contextual understanding, and overall response quality. We demonstrate that coreference resolution enhances retrieval effectiveness and improves question-answering (QA) performance. Through comparative analysis of different pooling strategies in retrieval tasks, we find that mean pooling demonstrates superior context capturing ability after applying coreference resolution. In QA tasks, we discover that smaller models benefit more from the disambiguation process, likely due to their limited inherent capacity for handling referential ambiguity. With these findings, this study aims to provide a deeper understanding of the challenges posed by coreferential complexity in RAG, providing guidance for improving retrieval and generation in knowledge-intensive AI applications. 

---
# Conditional Unigram Tokenization with Parallel Data 

**Authors**: Gianluca Vico, Jindřinch Libovický  

**Link**: [PDF](https://arxiv.org/pdf/2507.07824)  

**Abstract**: We introduce conditional unigram tokenization, a novel approach that extends unigram tokenization by conditioning target token probabilities on source-language tokens from parallel data. Given a fixed source tokenizer, our method learns a target tokenizer that maximizes cross-lingual semantic alignment. We evaluate our tokenizer on four language pairs across different families and resource levels, examining intrinsic properties and downstream performance on machine translation and language modeling. While our conditional tokenizer maintains comparable statistical properties to standard unigram tokenizers, results are mixed: we observe no improvements in machine translation quality, but find consistent perplexity reductions in language modeling. We hypothesize that quadratic scaling of conditional probability estimation with respect to the vocabulary size creates a data efficiency bottleneck. Our findings suggest that alternative parameterizations may be necessary for practical cross-lingual tokenization. 

---
# On the Effect of Instruction Tuning Loss on Generalization 

**Authors**: Anwoy Chatterjee, H S V N S Kowndinya Renduchintala, Sumit Bhatia, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2507.07817)  

**Abstract**: Instruction Tuning has emerged as a pivotal post-training paradigm that enables pre-trained language models to better follow user instructions. Despite its significance, little attention has been given to optimizing the loss function used. A fundamental, yet often overlooked, question is whether the conventional auto-regressive objective - where loss is computed only on response tokens, excluding prompt tokens - is truly optimal for instruction tuning. In this work, we systematically investigate the impact of differentially weighting prompt and response tokens in instruction tuning loss, and propose Weighted Instruction Tuning (WIT) as a better alternative to conventional instruction tuning. Through extensive experiments on five language models of different families and scale, three finetuning datasets of different sizes, and five diverse evaluation benchmarks, we show that the standard instruction tuning loss often yields suboptimal performance and limited robustness to input prompt variations. We find that a low-to-moderate weight for prompt tokens coupled with a moderate-to-high weight for response tokens yields the best-performing models across settings and also serve as better starting points for the subsequent preference alignment training. These findings highlight the need to reconsider instruction tuning loss and offer actionable insights for developing more robust and generalizable models. Our code is open-sourced at this https URL. 

---
# Understanding and Controlling Repetition Neurons and Induction Heads in In-Context Learning 

**Authors**: Nhi Hoai Doan, Tatsuya Hiraoka, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2507.07810)  

**Abstract**: This paper investigates the relationship between large language models' (LLMs) ability to recognize repetitive input patterns and their performance on in-context learning (ICL). In contrast to prior work that has primarily focused on attention heads, we examine this relationship from the perspective of skill neurons, specifically repetition neurons. Our experiments reveal that the impact of these neurons on ICL performance varies depending on the depth of the layer in which they reside. By comparing the effects of repetition neurons and induction heads, we further identify strategies for reducing repetitive outputs while maintaining strong ICL capabilities. 

---
# Bridging Logic and Learning: Decoding Temporal Logic Embeddings via Transformers 

**Authors**: Sara Candussio, Gaia Saveri, Gabriele Sarti, Luca Bortolussi  

**Link**: [PDF](https://arxiv.org/pdf/2507.07808)  

**Abstract**: Continuous representations of logic formulae allow us to integrate symbolic knowledge into data-driven learning algorithms. If such embeddings are semantically consistent, i.e. if similar specifications are mapped into nearby vectors, they enable continuous learning and optimization directly in the semantic space of formulae. However, to translate the optimal continuous representation into a concrete requirement, such embeddings must be invertible. We tackle this issue by training a Transformer-based decoder-only model to invert semantic embeddings of Signal Temporal Logic (STL) formulae. STL is a powerful formalism that allows us to describe properties of signals varying over time in an expressive yet concise way. By constructing a small vocabulary from STL syntax, we demonstrate that our proposed model is able to generate valid formulae after only 1 epoch and to generalize to the semantics of the logic in about 10 epochs. Additionally, the model is able to decode a given embedding into formulae that are often simpler in terms of length and nesting while remaining semantically close (or equivalent) to gold references. We show the effectiveness of our methodology across various levels of training formulae complexity to assess the impact of training data on the model's ability to effectively capture the semantic information contained in the embeddings and generalize out-of-distribution. Finally, we deploy our model for solving a requirement mining task, i.e. inferring STL specifications that solve a classification task on trajectories, performing the optimization directly in the semantic space. 

---
# StreamUni: Achieving Streaming Speech Translation with a Unified Large Speech-Language Model 

**Authors**: Shoutao Guo, Xiang Li, Shaolei Zhang, Mengge Liu, Wei Chen, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.07803)  

**Abstract**: Streaming speech translation (StreamST) requires determining appropriate timing, known as policy, to generate translations while continuously receiving source speech inputs, balancing low latency with high translation quality. However, existing StreamST methods typically operate on sentence-level speech segments, referred to as simultaneous speech translation (SimulST). In practice, they require collaboration with segmentation models to accomplish StreamST, where the truncated speech segments constrain SimulST models to make policy decisions and generate translations based on limited contextual information. Moreover, SimulST models struggle to learn effective policies due to the complexity of speech inputs and cross-lingual generation. To address these challenges, we propose StreamUni, which achieves StreamST through a unified Large Speech-Language Model (LSLM). Specifically, StreamUni incorporates speech Chain-of-Thought (CoT) in guiding the LSLM to generate multi-stage outputs. Leveraging these multi-stage outputs, StreamUni simultaneously accomplishes speech segmentation, policy decision, and translation generation, completing StreamST without requiring massive policy-specific training. Additionally, we propose a streaming CoT training method that enhances low-latency policy decisions and generation capabilities using limited CoT data. Experiments demonstrate that our approach achieves state-of-the-art performance on StreamST tasks. 

---
# When Large Language Models Meet Law: Dual-Lens Taxonomy, Technical Advances, and Ethical Governance 

**Authors**: Peizhang Shao, Linrui Xu, Jinxi Wang, Wei Zhou, Xingyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07748)  

**Abstract**: This paper establishes the first comprehensive review of Large Language Models (LLMs) applied within the legal domain. It pioneers an innovative dual lens taxonomy that integrates legal reasoning frameworks and professional ontologies to systematically unify historical research and contemporary breakthroughs. Transformer-based LLMs, which exhibit emergent capabilities such as contextual reasoning and generative argumentation, surmount traditional limitations by dynamically capturing legal semantics and unifying evidence reasoning. Significant progress is documented in task generalization, reasoning formalization, workflow integration, and addressing core challenges in text processing, knowledge integration, and evaluation rigor via technical innovations like sparse attention mechanisms and mixture-of-experts architectures. However, widespread adoption of LLM introduces critical challenges: hallucination, explainability deficits, jurisdictional adaptation difficulties, and ethical asymmetry. This review proposes a novel taxonomy that maps legal roles to NLP subtasks and computationally implements the Toulmin argumentation framework, thus systematizing advances in reasoning, retrieval, prediction, and dispute resolution. It identifies key frontiers including low-resource systems, multimodal evidence integration, and dynamic rebuttal handling. Ultimately, this work provides both a technical roadmap for researchers and a conceptual framework for practitioners navigating the algorithmic future, laying a robust foundation for the next era of legal artificial intelligence. We have created a GitHub repository to index the relevant papers: this https URL. 

---
# Code-Switching in End-to-End Automatic Speech Recognition: A Systematic Literature Review 

**Authors**: Maha Tufail Agro, Atharva Kulkarni, Karima Kadaoui, Zeerak Talat, Hanan Aldarmaki  

**Link**: [PDF](https://arxiv.org/pdf/2507.07741)  

**Abstract**: Motivated by a growing research interest into automatic speech recognition (ASR), and the growing body of work for languages in which code-switching (CS) often occurs, we present a systematic literature review of code-switching in end-to-end ASR models. We collect and manually annotate papers published in peer reviewed venues. We document the languages considered, datasets, metrics, model choices, and performance, and present a discussion of challenges in end-to-end ASR for code-switching. Our analysis thus provides insights on current research efforts and available resources as well as opportunities and gaps to guide future research. 

---
# Not All Preferences are What You Need for Post-Training: Selective Alignment Strategy for Preference Optimization 

**Authors**: Zhijin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07725)  

**Abstract**: Post-training alignment of large language models (LLMs) is a critical challenge, as not all tokens contribute equally to model performance. This paper introduces a selective alignment strategy that prioritizes high-impact tokens within preference pairs, leveraging token-level log-probability differences between the current policy and a reference model. By focusing on these informative tokens, our approach reduces computational overhead and enhances alignment fidelity. We further explore the role of reference model quality, demonstrating that stronger reference models significantly improve token selection accuracy and overall optimization effectiveness. Comprehensive experiments on benchmarks such as Arena-Hard and MT-Bench validate the superiority of our Selective-DPO method over standard DPO and distillation-based baselines. Our findings highlight the importance of token-level optimization and reference model selection in advancing preference alignment for LLMs. The code is available at this https URL. 

---
# Rethinking the Privacy of Text Embeddings: A Reproducibility Study of "Text Embeddings Reveal (Almost) As Much As Text" 

**Authors**: Dominykas Seputis, Yongkang Li, Karsten Langerak, Serghei Mihailov  

**Link**: [PDF](https://arxiv.org/pdf/2507.07700)  

**Abstract**: Text embeddings are fundamental to many natural language processing (NLP) tasks, extensively applied in domains such as recommendation systems and information retrieval (IR). Traditionally, transmitting embeddings instead of raw text has been seen as privacy-preserving. However, recent methods such as Vec2Text challenge this assumption by demonstrating that controlled decoding can successfully reconstruct original texts from black-box embeddings. The unexpectedly strong results reported by Vec2Text motivated us to conduct further verification, particularly considering the typically non-intuitive and opaque structure of high-dimensional embedding spaces. In this work, we reproduce the Vec2Text framework and evaluate it from two perspectives: (1) validating the original claims, and (2) extending the study through targeted experiments. First, we successfully replicate the original key results in both in-domain and out-of-domain settings, with only minor discrepancies arising due to missing artifacts, such as model checkpoints and dataset splits. Furthermore, we extend the study by conducting a parameter sensitivity analysis, evaluating the feasibility of reconstructing sensitive inputs (e.g., passwords), and exploring embedding quantization as a lightweight privacy defense. Our results show that Vec2Text is effective under ideal conditions, capable of reconstructing even password-like sequences that lack clear semantics. However, we identify key limitations, including its sensitivity to input sequence length. We also find that Gaussian noise and quantization techniques can mitigate the privacy risks posed by Vec2Text, with quantization offering a simpler and more widely applicable solution. Our findings emphasize the need for caution in using text embeddings and highlight the importance of further research into robust defense mechanisms for NLP systems. 

---
# KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities 

**Authors**: Hruday Markondapatnaikuni, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.07695)  

**Abstract**: Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study. 

---
# SAS: Simulated Attention Score 

**Authors**: Chuanyang Zheng, Jiankai Sun, Yihang Gao, Yuehao Wang, Peihao Wang, Jing Xiong, Liliang Ren, Hao Cheng, Janardhan Kulkarni, Yelong Shen, Atlas Wang, Mac Schwager, Anderson Schneider, Xiaodong Liu, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.07694)  

**Abstract**: The attention mechanism is a core component of the Transformer architecture. Various methods have been developed to compute attention scores, including multi-head attention (MHA), multi-query attention, group-query attention and so on. We further analyze the MHA and observe that its performance improves as the number of attention heads increases, provided the hidden size per head remains sufficiently large. Therefore, increasing both the head count and hidden size per head with minimal parameter overhead can lead to significant performance gains at a low cost. Motivated by this insight, we introduce Simulated Attention Score (SAS), which maintains a compact model size while simulating a larger number of attention heads and hidden feature dimension per head. This is achieved by projecting a low-dimensional head representation into a higher-dimensional space, effectively increasing attention capacity without increasing parameter count. Beyond the head representations, we further extend the simulation approach to feature dimension of the key and query embeddings, enhancing expressiveness by mimicking the behavior of a larger model while preserving the original model size. To control the parameter cost, we also propose Parameter-Efficient Attention Aggregation (PEAA). Comprehensive experiments on a variety of datasets and tasks demonstrate the effectiveness of the proposed SAS method, achieving significant improvements over different attention variants. 

---
# An Automated Length-Aware Quality Metric for Summarization 

**Authors**: Andrew D. Foland  

**Link**: [PDF](https://arxiv.org/pdf/2507.07653)  

**Abstract**: This paper proposes NOrmed Index of Retention (NOIR), a quantitative objective metric for evaluating summarization quality of arbitrary texts that relies on both the retention of semantic meaning and the summary length compression. This gives a measure of how well the recall-compression tradeoff is managed, the most important skill in summarization. Experiments demonstrate that NOIR effectively captures the token-length / semantic retention tradeoff of a summarizer and correlates to human perception of sumarization quality. Using a language model-embedding to measure semantic similarity, it provides an automated alternative for assessing summarization quality without relying on time-consuming human-generated reference summaries. The proposed metric can be applied to various summarization tasks, offering an automated tool for evaluating and improving summarization algorithms, summarization prompts, and synthetically-generated summaries. 

---
# Lost in Pronunciation: Detecting Chinese Offensive Language Disguised by Phonetic Cloaking Replacement 

**Authors**: Haotan Guo, Jianfei He, Jiayuan Ma, Hongbin Na, Zimu Wang, Haiyang Zhang, Qi Chen, Wei Wang, Zijing Shi, Tao Shen, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.07640)  

**Abstract**: Phonetic Cloaking Replacement (PCR), defined as the deliberate use of homophonic or near-homophonic variants to hide toxic intent, has become a major obstacle to Chinese content moderation. While this problem is well-recognized, existing evaluations predominantly rely on rule-based, synthetic perturbations that ignore the creativity of real users. We organize PCR into a four-way surface-form taxonomy and compile \ours, a dataset of 500 naturally occurring, phonetically cloaked offensive posts gathered from the RedNote platform. Benchmarking state-of-the-art LLMs on this dataset exposes a serious weakness: the best model reaches only an F1-score of 0.672, and zero-shot chain-of-thought prompting pushes performance even lower. Guided by error analysis, we revisit a Pinyin-based prompting strategy that earlier studies judged ineffective and show that it recovers much of the lost accuracy. This study offers the first comprehensive taxonomy of Chinese PCR, a realistic benchmark that reveals current detectors' limits, and a lightweight mitigation technique that advances research on robust toxicity detection. 

---
# FrugalRAG: Learning to retrieve and reason for multi-hop QA 

**Authors**: Abhinav Java, Srivathsan Koundinyan, Nagarajan Natarajan, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.07634)  

**Abstract**: We consider the problem of answering complex questions, given access to a large unstructured document corpus. The de facto approach to solving the problem is to leverage language models that (iteratively) retrieve and reason through the retrieved documents, until the model has sufficient information to generate an answer. Attempts at improving this approach focus on retrieval-augmented generation (RAG) metrics such as accuracy and recall and can be categorized into two types: (a) fine-tuning on large question answering (QA) datasets augmented with chain-of-thought traces, and (b) leveraging RL-based fine-tuning techniques that rely on question-document relevance signals. However, efficiency in the number of retrieval searches is an equally important metric, which has received less attention. In this work, we show that: (1) Large-scale fine-tuning is not needed to improve RAG metrics, contrary to popular claims in recent literature. Specifically, a standard ReAct pipeline with improved prompts can outperform state-of-the-art methods on benchmarks such as HotPotQA. (2) Supervised and RL-based fine-tuning can help RAG from the perspective of frugality, i.e., the latency due to number of searches at inference time. For example, we show that we can achieve competitive RAG metrics at nearly half the cost (in terms of number of searches) on popular RAG benchmarks, using the same base model, and at a small training cost (1000 examples). 

---
# Exploring the Limits of Model Compression in LLMs: A Knowledge Distillation Study on QA Tasks 

**Authors**: Joyeeta Datta, Niclas Doll, Qusai Ramadan, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2507.07630)  

**Abstract**: Large Language Models (LLMs) have demonstrated outstanding performance across a range of NLP tasks, however, their computational demands hinder their deployment in real-world, resource-constrained environments. This work investigates the extent to which LLMs can be compressed using Knowledge Distillation (KD) while maintaining strong performance on Question Answering (QA) tasks. We evaluate student models distilled from the Pythia and Qwen2.5 families on two QA benchmarks, SQuAD and MLQA, under zero-shot and one-shot prompting conditions. Results show that student models retain over 90% of their teacher models' performance while reducing parameter counts by up to 57.1%. Furthermore, one-shot prompting yields additional performance gains over zero-shot setups for both model families. These findings underscore the trade-off between model efficiency and task performance, demonstrating that KD, combined with minimal prompting, can yield compact yet capable QA systems suitable for resource-constrained applications. 

---
# Bayesian Discrete Diffusion Beats Autoregressive Perplexity 

**Authors**: Cooper Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2507.07586)  

**Abstract**: We reveal a hidden Bayesian core of discrete-diffusion language models by showing that the expected denoiser output under the forward masking distribution recovers the exact posterior over clean tokens. Under minimal assumptions, Monte Carlo marginalization over K independent corruptions converges to this posterior at rate O(1/sqrt(K)), yielding a simple proof of consistency and finite-sample error bounds. Building on this insight, we introduce a lightweight inference-time ensemble that averages K mask-and-denoise passes to obtain posterior-aware token probabilities and uncertainty estimates at no extra training cost. On WikiText-2, our method achieves test perplexity 8.8 with K=8, versus 20.3 for GPT-2 Small, despite using a model of comparable size. Code is available at this https URL. 

---
# Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation 

**Authors**: Yupu Liang, Yaping Zhang, Zhiyang Zhang, Yang Zhao, Lu Xiang, Chengqing Zong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.07572)  

**Abstract**: Document Image Machine Translation (DIMT) aims to translate text within document images, facing generalization challenges due to limited training data and the complex interplay between visual and textual information. To address these challenges, we introduce M4Doc, a novel single-to-mix modality alignment framework leveraging Multimodal Large Language Models (MLLMs). M4Doc aligns an image-only encoder with the multimodal representations of an MLLM, pre-trained on large-scale document image datasets. This alignment enables a lightweight DIMT model to learn crucial visual-textual correlations during training. During inference, M4Doc bypasses the MLLM, maintaining computational efficiency while benefiting from its multimodal knowledge. Comprehensive experiments demonstrate substantial improvements in translation quality, especially in cross-domain generalization and challenging document image scenarios. 

---
# The Synergy Dilemma of Long-CoT SFT and RL: Investigating Post-Training Techniques for Reasoning VLMs 

**Authors**: Jierun Chen, Tiezheng Yu, Haoli Bai, Lewei Yao, Jiannan Wu, Kaican Li, Fei Mi, Chaofan Tao, Lei Zhu, Manyi Zhang, Xiaohui Li, Lu Hou, Lifeng Shang, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07562)  

**Abstract**: Large vision-language models (VLMs) increasingly adopt post-training techniques such as long chain-of-thought (CoT) supervised fine-tuning (SFT) and reinforcement learning (RL) to elicit sophisticated reasoning. While these methods exhibit synergy in language-only models, their joint effectiveness in VLMs remains uncertain. We present a systematic investigation into the distinct roles and interplay of long-CoT SFT and RL across multiple multimodal reasoning benchmarks. We find that SFT improves performance on difficult questions by in-depth, structured reasoning, but introduces verbosity and degrades performance on simpler ones. In contrast, RL promotes generalization and brevity, yielding consistent improvements across all difficulty levels, though the improvements on the hardest questions are less prominent compared to SFT. Surprisingly, combining them through two-staged, interleaved, or progressive training strategies, as well as data mixing and model merging, all fails to produce additive benefits, instead leading to trade-offs in accuracy, reasoning style, and response length. This ``synergy dilemma'' highlights the need for more seamless and adaptive approaches to unlock the full potential of combined post-training techniques for reasoning VLMs. 

---
# The Cross-Lingual Cost: Retrieval Biases in RAG over Arabic-English Corpora 

**Authors**: Chen Amiraz, Yaroslav Fyodorov, Elad Haramaty, Zohar Karnin, Liane Lewin-Eytan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07543)  

**Abstract**: Cross-lingual retrieval-augmented generation (RAG) is a critical capability for retrieving and generating answers across languages. Prior work in this context has mostly focused on generation and relied on benchmarks derived from open-domain sources, most notably Wikipedia. In such settings, retrieval challenges often remain hidden due to language imbalances, overlap with pretraining data, and memorized content. To address this gap, we study Arabic-English RAG in a domain-specific setting using benchmarks derived from real-world corporate datasets. Our benchmarks include all combinations of languages for the user query and the supporting document, drawn independently and uniformly at random. This enables a systematic study of multilingual retrieval behavior.
Our findings reveal that retrieval is a critical bottleneck in cross-lingual domain-specific scenarios, with significant performance drops occurring when the user query and supporting document languages differ. A key insight is that these failures stem primarily from the retriever's difficulty in ranking documents across languages. Finally, we propose a simple retrieval strategy that addresses this source of failure by enforcing equal retrieval from both languages, resulting in substantial improvements in cross-lingual and overall performance. These results highlight meaningful opportunities for improving multilingual retrieval, particularly in practical, real-world RAG applications. 

---
# CEA-LIST at CheckThat! 2025: Evaluating LLMs as Detectors of Bias and Opinion in Text 

**Authors**: Akram Elbouanani, Evan Dufraisse, Aboubacar Tuo, Adrian Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07539)  

**Abstract**: This paper presents a competitive approach to multilingual subjectivity detection using large language models (LLMs) with few-shot prompting. We participated in Task 1: Subjectivity of the CheckThat! 2025 evaluation campaign. We show that LLMs, when paired with carefully designed prompts, can match or outperform fine-tuned smaller language models (SLMs), particularly in noisy or low-quality data settings. Despite experimenting with advanced prompt engineering techniques, such as debating LLMs and various example selection strategies, we found limited benefit beyond well-crafted standard few-shot prompts. Our system achieved top rankings across multiple languages in the CheckThat! 2025 subjectivity detection task, including first place in Arabic and Polish, and top-four finishes in Italian, English, German, and multilingual tracks. Notably, our method proved especially robust on the Arabic dataset, likely due to its resilience to annotation inconsistencies. These findings highlight the effectiveness and adaptability of LLM-based few-shot learning for multilingual sentiment tasks, offering a strong alternative to traditional fine-tuning, particularly when labeled data is scarce or inconsistent. 

---
# Triadic Multi-party Voice Activity Projection for Turn-taking in Spoken Dialogue Systems 

**Authors**: Mikey Elmers, Koji Inoue, Divesh Lala, Tatsuya Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2507.07518)  

**Abstract**: Turn-taking is a fundamental component of spoken dialogue, however conventional studies mostly involve dyadic settings. This work focuses on applying voice activity projection (VAP) to predict upcoming turn-taking in triadic multi-party scenarios. The goal of VAP models is to predict the future voice activity for each speaker utilizing only acoustic data. This is the first study to extend VAP into triadic conversation. We trained multiple models on a Japanese triadic dataset where participants discussed a variety of topics. We found that the VAP trained on triadic conversation outperformed the baseline for all models but that the type of conversation affected the accuracy. This study establishes that VAP can be used for turn-taking in triadic dialogue scenarios. Future work will incorporate this triadic VAP turn-taking model into spoken dialogue systems. 

---
# Toward Real-World Chinese Psychological Support Dialogues: CPsDD Dataset and a Co-Evolving Multi-Agent System 

**Authors**: Yuanchen Shi, Longyin Zhang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07509)  

**Abstract**: The growing need for psychological support due to increasing pressures has exposed the scarcity of relevant datasets, particularly in non-English languages. To address this, we propose a framework that leverages limited real-world data and expert knowledge to fine-tune two large language models: Dialog Generator and Dialog Modifier. The Generator creates large-scale psychological counseling dialogues based on predefined paths, which guide system response strategies and user interactions, forming the basis for effective support. The Modifier refines these dialogues to align with real-world data quality. Through both automated and manual review, we construct the Chinese Psychological support Dialogue Dataset (CPsDD), containing 68K dialogues across 13 groups, 16 psychological problems, 13 causes, and 12 support focuses. Additionally, we introduce the Comprehensive Agent Dialogue Support System (CADSS), where a Profiler analyzes user characteristics, a Summarizer condenses dialogue history, a Planner selects strategies, and a Supporter generates empathetic responses. The experimental results of the Strategy Prediction and Emotional Support Conversation (ESC) tasks demonstrate that CADSS achieves state-of-the-art performance on both CPsDD and ESConv datasets. 

---
# Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models 

**Authors**: Varin Sikka, Vishal Sikka  

**Link**: [PDF](https://arxiv.org/pdf/2507.07505)  

**Abstract**: With widespread adoption of transformer-based language models in AI, there is significant interest in the limits of LLMs capabilities, specifically so-called hallucinations, occurrences in which LLMs provide spurious, factually incorrect or nonsensical information when prompted on certain subjects. Furthermore, there is growing interest in agentic uses of LLMs - that is, using LLMs to create agents that act autonomously or semi-autonomously to carry out various tasks, including tasks with applications in the real world. This makes it important to understand the types of tasks LLMs can and cannot perform. We explore this topic from the perspective of the computational complexity of LLM inference. We show that LLMs are incapable of carrying out computational and agentic tasks beyond a certain complexity, and further that LLMs are incapable of verifying the accuracy of tasks beyond a certain complexity. We present examples of both, then discuss some consequences of this work. 

---
# Extracting ORR Catalyst Information for Fuel Cell from Scientific Literature 

**Authors**: Hein Htet, Amgad Ahmed Ali Ibrahim, Yutaka Sasaki, Ryoji Asahi  

**Link**: [PDF](https://arxiv.org/pdf/2507.07499)  

**Abstract**: The oxygen reduction reaction (ORR) catalyst plays a critical role in enhancing fuel cell efficiency, making it a key focus in material science research. However, extracting structured information about ORR catalysts from vast scientific literature remains a significant challenge due to the complexity and diversity of textual data. In this study, we propose a named entity recognition (NER) and relation extraction (RE) approach using DyGIE++ with multiple pre-trained BERT variants, including MatSciBERT and PubMedBERT, to extract ORR catalyst-related information from the scientific literature, which is compiled into a fuel cell corpus for materials informatics (FC-CoMIcs). A comprehensive dataset was constructed manually by identifying 12 critical entities and two relationship types between pairs of the entities. Our methodology involves data annotation, integration, and fine-tuning of transformer-based models to enhance information extraction accuracy. We assess the impact of different BERT variants on extraction performance and investigate the effects of annotation consistency. Experimental evaluations demonstrate that the fine-tuned PubMedBERT model achieves the highest NER F1-score of 82.19% and the MatSciBERT model attains the best RE F1-score of 66.10%. Furthermore, the comparison with human annotators highlights the reliability of fine-tuned models for ORR catalyst extraction, demonstrating their potential for scalable and automated literature analysis. The results indicate that domain-specific BERT models outperform general scientific models like BlueBERT for ORR catalyst extraction. 

---
# Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code 

**Authors**: Keqin Bao, Nuo Chen, Xiaoyuan Li, Binyuan Hui, Bowen Yu, Fuli Feng, Junyang Lin, Xiangnan He, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07498)  

**Abstract**: Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B. 

---
# PLAN-TUNING: Post-Training Language Models to Learn Step-by-Step Planning for Complex Problem Solving 

**Authors**: Mihir Parmar, Palash Goyal, Xin Liu, Yiwen Song, Mingyang Ling, Chitta Baral, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.07495)  

**Abstract**: Recently, decomposing complex problems into simple subtasks--a crucial part of human-like natural planning--to solve the given problem has significantly boosted the performance of large language models (LLMs). However, leveraging such planning structures during post-training to boost the performance of smaller open-source LLMs remains underexplored. Motivated by this, we introduce PLAN-TUNING, a unified post-training framework that (i) distills synthetic task decompositions (termed "planning trajectories") from large-scale LLMs and (ii) fine-tunes smaller models via supervised and reinforcement-learning objectives designed to mimic these planning processes to improve complex reasoning. On GSM8k and the MATH benchmarks, plan-tuned models outperform strong baselines by an average $\sim7\%$. Furthermore, plan-tuned models show better generalization capabilities on out-of-domain datasets, with average $\sim10\%$ and $\sim12\%$ performance improvements on OlympiadBench and AIME 2024, respectively. Our detailed analysis demonstrates how planning trajectories improves complex reasoning capabilities, showing that PLAN-TUNING is an effective strategy for improving task-specific performance of smaller LLMs. 

---
# Machine Bullshit: Characterizing the Emergent Disregard for Truth in Large Language Models 

**Authors**: Kaiqu Liang, Haimin Hu, Xuandong Zhao, Dawn Song, Thomas L. Griffiths, Jaime Fernández Fisac  

**Link**: [PDF](https://arxiv.org/pdf/2507.07484)  

**Abstract**: Bullshit, as conceptualized by philosopher Harry Frankfurt, refers to statements made without regard to their truth value. While previous work has explored large language model (LLM) hallucination and sycophancy, we propose machine bullshit as an overarching conceptual framework that can allow researchers to characterize the broader phenomenon of emergent loss of truthfulness in LLMs and shed light on its underlying mechanisms. We introduce the Bullshit Index, a novel metric quantifying LLMs' indifference to truth, and propose a complementary taxonomy analyzing four qualitative forms of bullshit: empty rhetoric, paltering, weasel words, and unverified claims. We conduct empirical evaluations on the Marketplace dataset, the Political Neutrality dataset, and our new BullshitEval benchmark (2,400 scenarios spanning 100 AI assistants) explicitly designed to evaluate machine bullshit. Our results demonstrate that model fine-tuning with reinforcement learning from human feedback (RLHF) significantly exacerbates bullshit and inference-time chain-of-thought (CoT) prompting notably amplify specific bullshit forms, particularly empty rhetoric and paltering. We also observe prevalent machine bullshit in political contexts, with weasel words as the dominant strategy. Our findings highlight systematic challenges in AI alignment and provide new insights toward more truthful LLM behavior. 

---
# RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning 

**Authors**: Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.07451)  

**Abstract**: Reinforcement learning (RL) for large language models is an energy-intensive endeavor: training can be unstable, and the policy may gradually drift away from its pretrained weights. We present \emph{RLEP}\, -- \,Reinforcement Learning with Experience rePlay\, -- \,a two-phase framework that first collects verified trajectories and then replays them during subsequent training. At every update step, the policy is optimized on mini-batches that blend newly generated rollouts with these replayed successes. By replaying high-quality examples, RLEP steers the model away from fruitless exploration, focuses learning on promising reasoning paths, and delivers both faster convergence and stronger final performance. On the Qwen2.5-Math-7B base model, RLEP reaches baseline peak accuracy with substantially fewer updates and ultimately surpasses it, improving accuracy on AIME-2024 from 38.2% to 39.9%, on AIME-2025 from 19.8% to 22.3%, and on AMC-2023 from 77.0% to 82.2%. Our code, datasets, and checkpoints are publicly available at this https URL to facilitate reproducibility and further research. 

---
# SAND: Boosting LLM Agents with Self-Taught Action Deliberation 

**Authors**: Yu Xia, Yiran Jenny Shen, Junda Wu, Tong Yu, Sungchul Kim, Ryan A. Rossi, Lina Yao, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.07441)  

**Abstract**: Large Language Model (LLM) agents are commonly tuned with supervised finetuning on ReAct-style expert trajectories or preference optimization over pairwise rollouts. Most of these methods focus on imitating specific expert behaviors or promoting chosen reasoning thoughts and actions over rejected ones. However, without reasoning and comparing over alternatives actions, LLM agents finetuned with these methods may over-commit towards seemingly plausible but suboptimal actions due to limited action space exploration. To address this, in this paper we propose Self-taught ActioN Deliberation (SAND) framework, enabling LLM agents to explicitly deliberate over candidate actions before committing to one. To tackle the challenges of when and what to deliberate given large action space and step-level action evaluation, we incorporate self-consistency action sampling and execution-guided action critique to help synthesize step-wise action deliberation thoughts using the base model of the LLM agent. In an iterative manner, the deliberation trajectories are then used to finetune the LLM agent itself. Evaluating on two representative interactive agent tasks, SAND achieves an average 20% improvement over initial supervised finetuning and also outperforms state-of-the-art agent tuning approaches. 

---
# Towards Interpretable Time Series Foundation Models 

**Authors**: Matthieu Boileau, Philippe Helluy, Jeremy Pawlus, Svitlana Vyetrenko  

**Link**: [PDF](https://arxiv.org/pdf/2507.07439)  

**Abstract**: In this paper, we investigate the distillation of time series reasoning capabilities into small, instruction-tuned language models as a step toward building interpretable time series foundation models. Leveraging a synthetic dataset of mean-reverting time series with systematically varied trends and noise levels, we generate natural language annotations using a large multimodal model and use these to supervise the fine-tuning of compact Qwen models. We introduce evaluation metrics that assess the quality of the distilled reasoning - focusing on trend direction, noise intensity, and extremum localization - and show that the post-trained models acquire meaningful interpretive capabilities. Our results highlight the feasibility of compressing time series understanding into lightweight, language-capable models suitable for on-device or privacy-sensitive deployment. This work contributes a concrete foundation toward developing small, interpretable models that explain temporal patterns in natural language. 

---
# SynthEHR-Eviction: Enhancing Eviction SDoH Detection with LLM-Augmented Synthetic EHR Data 

**Authors**: Zonghai Yao, Youxia Zhao, Avijit Mitra, David A. Levy, Emily Druhl, Jack Tsai, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07421)  

**Abstract**: Eviction is a significant yet understudied social determinants of health (SDoH), linked to housing instability, unemployment, and mental health. While eviction appears in unstructured electronic health records (EHRs), it is rarely coded in structured fields, limiting downstream applications. We introduce SynthEHR-Eviction, a scalable pipeline combining LLMs, human-in-the-loop annotation, and automated prompt optimization (APO) to extract eviction statuses from clinical notes. Using this pipeline, we created the largest public eviction-related SDoH dataset to date, comprising 14 fine-grained categories. Fine-tuned LLMs (e.g., Qwen2.5, LLaMA3) trained on SynthEHR-Eviction achieved Macro-F1 scores of 88.8% (eviction) and 90.3% (other SDoH) on human validated data, outperforming GPT-4o-APO (87.8%, 87.3%), GPT-4o-mini-APO (69.1%, 78.1%), and BioBERT (60.7%, 68.3%), while enabling cost-effective deployment across various model sizes. The pipeline reduces annotation effort by over 80%, accelerates dataset creation, enables scalable eviction detection, and generalizes to other information extraction tasks. 

---
# MedReadCtrl: Personalizing medical text generation with readability-controlled instruction learning 

**Authors**: Hieu Tran, Zonghai Yao, Won Seok Jang, Sharmin Sultana, Allen Chang, Yuan Zhang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07419)  

**Abstract**: Generative AI has demonstrated strong potential in healthcare, from clinical decision support to patient-facing chatbots that improve outcomes. A critical challenge for deployment is effective human-AI communication, where content must be both personalized and understandable. We introduce MedReadCtrl, a readability-controlled instruction tuning framework that enables LLMs to adjust output complexity without compromising meaning. Evaluations of nine datasets and three tasks across medical and general domains show that MedReadCtrl achieves significantly lower readability instruction-following errors than GPT-4 (e.g., 1.39 vs. 1.59 on ReadMe, p<0.001) and delivers substantial gains on unseen clinical tasks (e.g., +14.7 ROUGE-L, +6.18 SARI on MTSamples). Experts consistently preferred MedReadCtrl (71.7% vs. 23.3%), especially at low literacy levels. These gains reflect MedReadCtrl's ability to restructure clinical content into accessible, readability-aligned language while preserving medical intent, offering a scalable solution to support patient education and expand equitable access to AI-enabled care. 

---
# GNN-CNN: An Efficient Hybrid Model of Convolutional and Graph Neural Networks for Text Representation 

**Authors**: Fardin Rastakhiz  

**Link**: [PDF](https://arxiv.org/pdf/2507.07414)  

**Abstract**: Time, cost, and energy efficiency are critical considerations in Deep-Learning (DL), particularly when processing long texts. Transformers, which represent the current state of the art, exhibit quadratic computational complexity relative to input length, making them inefficient for extended documents. This study introduces a novel model architecture that combines Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs), integrated with a real-time, end-to-end graph generation mechanism. The model processes compact batches of character-level inputs without requiring padding or truncation. To enhance performance while maintaining high speed and efficiency, the model incorporates information from Large Language Models (LLMs), such as token embeddings and sentiment polarities, through efficient dictionary lookups. It captures local contextual patterns using CNNs, expands local receptive fields via lattice-based graph structures, and employs small-world graphs to aggregate document-level information. The generated graphs exhibit structural properties indicative of meaningful semantic organization, with an average clustering coefficient of approximately 0.45 and an average shortest path length ranging between 4 and 5. The model is evaluated across multiple text classification tasks, including sentiment analysis and news-categorization, and is compared against state-of-the-art models. Experimental results confirm the proposed model's efficiency and competitive performance. 

---
# Multi-Agent Retrieval-Augmented Framework for Evidence-Based Counterspeech Against Health Misinformation 

**Authors**: Anirban Saha Anik, Xiaoying Song, Elliott Wang, Bryan Wang, Bengisu Yarimbas, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07307)  

**Abstract**: Large language models (LLMs) incorporated with Retrieval-Augmented Generation (RAG) have demonstrated powerful capabilities in generating counterspeech against misinformation. However, current studies rely on limited evidence and offer less control over final outputs. To address these challenges, we propose a Multi-agent Retrieval-Augmented Framework to generate counterspeech against health misinformation, incorporating multiple LLMs to optimize knowledge retrieval, evidence enhancement, and response refinement. Our approach integrates both static and dynamic evidence, ensuring that the generated counterspeech is relevant, well-grounded, and up-to-date. Our method outperforms baseline approaches in politeness, relevance, informativeness, and factual accuracy, demonstrating its effectiveness in generating high-quality counterspeech. To further validate our approach, we conduct ablation studies to verify the necessity of each component in our framework. Furthermore, human evaluations reveal that refinement significantly enhances counterspeech quality and obtains human preference. 

---
# The Impact of Background Speech on Interruption Detection in Collaborative Groups 

**Authors**: Mariah Bradford, Nikhil Krishnaswamy, Nathaniel Blanchard  

**Link**: [PDF](https://arxiv.org/pdf/2507.07280)  

**Abstract**: Interruption plays a crucial role in collaborative learning, shaping group interactions and influencing knowledge construction. AI-driven support can assist teachers in monitoring these interactions. However, most previous work on interruption detection and interpretation has been conducted in single-conversation environments with relatively clean audio. AI agents deployed in classrooms for collaborative learning within small groups will need to contend with multiple concurrent conversations -- in this context, overlapping speech will be ubiquitous, and interruptions will need to be identified in other ways. In this work, we analyze interruption detection in single-conversation and multi-group dialogue settings. We then create a state-of-the-art method for interruption identification that is robust to overlapping speech, and thus could be deployed in classrooms. Further, our work highlights meaningful linguistic and prosodic information about how interruptions manifest in collaborative group interactions. Our investigation also paves the way for future works to account for the influence of overlapping speech from multiple groups when tracking group dialog. 

---
# Medical Red Teaming Protocol of Language Models: On the Importance of User Perspectives in Healthcare Settings 

**Authors**: Minseon Kim, Jean-Philippe Corbeil, Alessandro Sordoni, Francois Beaulieu, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2507.07248)  

**Abstract**: As the performance of large language models (LLMs) continues to advance, their adoption is expanding across a wide range of domains, including the medical field. The integration of LLMs into medical applications raises critical safety concerns, particularly due to their use by users with diverse roles, e.g. patients and clinicians, and the potential for model's outputs to directly affect human health. Despite the domain-specific capabilities of medical LLMs, prior safety evaluations have largely focused only on general safety benchmarks. In this paper, we introduce a safety evaluation protocol tailored to the medical domain in both patient user and clinician user perspectives, alongside general safety assessments and quantitatively analyze the safety of medical LLMs. We bridge a gap in the literature by building the PatientSafetyBench containing 466 samples over 5 critical categories to measure safety from the perspective of the patient. We apply our red-teaming protocols on the MediPhi model collection as a case study. To our knowledge, this is the first work to define safety evaluation criteria for medical LLMs through targeted red-teaming taking three different points of view - patient, clinician, and general user - establishing a foundation for safer deployment in medical domains. 

---
# SynthTextEval: Synthetic Text Data Generation and Evaluation for High-Stakes Domains 

**Authors**: Krithika Ramesh, Daniel Smolyak, Zihao Zhao, Nupoor Gandhi, Ritu Agarwal, Margrét Bjarnadóttir, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2507.07229)  

**Abstract**: We present SynthTextEval, a toolkit for conducting comprehensive evaluations of synthetic text. The fluency of large language model (LLM) outputs has made synthetic text potentially viable for numerous applications, such as reducing the risks of privacy violations in the development and deployment of AI systems in high-stakes domains. Realizing this potential, however, requires principled consistent evaluations of synthetic data across multiple dimensions: its utility in downstream systems, the fairness of these systems, the risk of privacy leakage, general distributional differences from the source text, and qualitative feedback from domain experts. SynthTextEval allows users to conduct evaluations along all of these dimensions over synthetic data that they upload or generate using the toolkit's generation module. While our toolkit can be run over any data, we highlight its functionality and effectiveness over datasets from two high-stakes domains: healthcare and law. By consolidating and standardizing evaluation metrics, we aim to improve the viability of synthetic text, and in-turn, privacy-preservation in AI development. 

---
# Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses 

**Authors**: Jens Rupprecht, Georg Ahnert, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2507.07188)  

**Abstract**: Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts -- we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also reveal that all tested models exhibit a consistent \textit{recency bias} varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data. 

---
# Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs 

**Authors**: Itay Itzhak, Yonatan Belinkov, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.07186)  

**Abstract**: Large language models (LLMs) exhibit cognitive biases -- systematic tendencies of irrational decision-making, similar to those seen in humans. Prior work has found that these biases vary across models and can be amplified by instruction tuning. However, it remains unclear if these differences in biases stem from pretraining, finetuning, or even random noise due to training stochasticity. We propose a two-step causal experimental approach to disentangle these factors. First, we finetune models multiple times using different random seeds to study how training randomness affects over $30$ cognitive biases. Second, we introduce \emph{cross-tuning} -- swapping instruction datasets between models to isolate bias sources. This swap uses datasets that led to different bias patterns, directly testing whether biases are dataset-dependent. Our findings reveal that while training randomness introduces some variability, biases are mainly shaped by pretraining: models with the same pretrained backbone exhibit more similar bias patterns than those sharing only finetuning data. These insights suggest that understanding biases in finetuned models requires considering their pretraining origins beyond finetuning effects. This perspective can guide future efforts to develop principled strategies for evaluating and mitigating bias in LLMs. 

---
# Traceable Evidence Enhanced Visual Grounded Reasoning: Evaluation and Methodology 

**Authors**: Haochen Wang, Xiangtai Li, Zilong Huang, Anran Wang, Jiacong Wang, Tao Zhang, Jiani Zheng, Sule Bai, Zijian Kang, Jiashi Feng, Zhuochen Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07999)  

**Abstract**: Models like OpenAI-o3 pioneer visual grounded reasoning by dynamically referencing visual regions, just like human "thinking with images". However, no benchmark exists to evaluate these capabilities holistically. To bridge this gap, we propose TreeBench (Traceable Evidence Evaluation Benchmark), a diagnostic benchmark built on three principles: (1) focused visual perception of subtle targets in complex scenes, (2) traceable evidence via bounding box evaluation, and (3) second-order reasoning to test object interactions and spatial hierarchies beyond simple object localization. Prioritizing images with dense objects, we initially sample 1K high-quality images from SA-1B, and incorporate eight LMM experts to manually annotate questions, candidate options, and answers for each image. After three stages of quality control, TreeBench consists of 405 challenging visual question-answering pairs, even the most advanced models struggle with this benchmark, where none of them reach 60% accuracy, e.g., OpenAI-o3 scores only 54.87. Furthermore, we introduce TreeVGR (Traceable Evidence Enhanced Visual Grounded Reasoning), a training paradigm to supervise localization and reasoning jointly with reinforcement learning, enabling accurate localizations and explainable reasoning pathways. Initialized from Qwen2.5-VL-7B, it improves V* Bench (+16.8), MME-RealWorld (+12.6), and TreeBench (+13.4), proving traceability is key to advancing vision-grounded reasoning. The code is available at this https URL. 

---
# Scaling RL to Long Videos 

**Authors**: Yukang Chen, Wei Huang, Baifeng Shi, Qinghao Hu, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu, Hongxu Yin, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.07966)  

**Abstract**: We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 52K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In experiments, LongVILA-R1-7B achieves strong performance on long video QA benchmarks such as VideoMME. It also outperforms Video-R1-7B and even matches Gemini-1.5-Pro across temporal reasoning, goal and purpose reasoning, spatial reasoning, and plot reasoning on our LongVideo-Reason-eval benchmark. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. LongVILA-R1 demonstrates consistent performance gains as the number of input video frames scales. LongVILA-R1 marks a firm step towards long video reasoning in VLMs. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames / around 256k tokens). 

---
# GuardVal: Dynamic Large Language Model Jailbreak Evaluation for Comprehensive Safety Testing 

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07735)  

**Abstract**: Jailbreak attacks reveal critical vulnerabilities in Large Language Models (LLMs) by causing them to generate harmful or unethical content. Evaluating these threats is particularly challenging due to the evolving nature of LLMs and the sophistication required in effectively probing their vulnerabilities. Current benchmarks and evaluation methods struggle to fully address these challenges, leaving gaps in the assessment of LLM vulnerabilities. In this paper, we review existing jailbreak evaluation practices and identify three assumed desiderata for an effective jailbreak evaluation protocol. To address these challenges, we introduce GuardVal, a new evaluation protocol that dynamically generates and refines jailbreak prompts based on the defender LLM's state, providing a more accurate assessment of defender LLMs' capacity to handle safety-critical situations. Moreover, we propose a new optimization method that prevents stagnation during prompt refinement, ensuring the generation of increasingly effective jailbreak prompts that expose deeper weaknesses in the defender LLMs. We apply this protocol to a diverse set of models, from Mistral-7b to GPT-4, across 10 safety domains. Our findings highlight distinct behavioral patterns among the models, offering a comprehensive view of their robustness. Furthermore, our evaluation process deepens the understanding of LLM behavior, leading to insights that can inform future research and drive the development of more secure models. 

---
# SpatialViz-Bench: Automatically Generated Spatial Visualization Reasoning Tasks for MLLMs 

**Authors**: Siting Wang, Luoyang Sun, Cheng Deng, Kun Shao, Minnan Pei, Zheng Tian, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07610)  

**Abstract**: Humans can directly imagine and manipulate visual images in their minds, a capability known as spatial visualization. While multi-modal Large Language Models (MLLMs) support imagination-based reasoning, spatial visualization remains insufficiently evaluated, typically embedded within broader mathematical and logical assessments. Existing evaluations often rely on IQ tests or math competitions that may overlap with training data, compromising assessment reliability. To this end, we introduce SpatialViz-Bench, a comprehensive multi-modal benchmark for spatial visualization with 12 tasks across 4 sub-abilities, comprising 1,180 automatically generated problems. Our evaluation of 33 state-of-the-art MLLMs not only reveals wide performance variations and demonstrates the benchmark's strong discriminative power, but also uncovers counter-intuitive findings: models exhibit unexpected behaviors by showing difficulty perception that misaligns with human intuition, displaying dramatic 2D-to-3D performance cliffs, and defaulting to formula derivation despite spatial tasks requiring visualization alone. SpatialVizBench empirically demonstrates that state-of-the-art MLLMs continue to exhibit deficiencies in spatial visualization tasks, thereby addressing a significant lacuna in the field. The benchmark is publicly available. 

---
# Improving Clustering on Occupational Text Data through Dimensionality Reduction 

**Authors**: Iago Xabier Vázquez García, Damla Partanaz, Emrullah Fatih Yetkin  

**Link**: [PDF](https://arxiv.org/pdf/2507.07582)  

**Abstract**: In this study, we focused on proposing an optimal clustering mechanism for the occupations defined in the well-known US-based occupational database, O*NET. Even though all occupations are defined according to well-conducted surveys in the US, their definitions can vary for different firms and countries. Hence, if one wants to expand the data that is already collected in O*NET for the occupations defined with different tasks, a map between the definitions will be a vital requirement. We proposed a pipeline using several BERT-based techniques with various clustering approaches to obtain such a map. We also examined the effect of dimensionality reduction approaches on several metrics used in measuring performance of clustering algorithms. Finally, we improved our results by using a specialized silhouette approach. This new clustering-based mapping approach with dimensionality reduction may help distinguish the occupations automatically, creating new paths for people wanting to change their careers. 

---
# COALA: Numerically Stable and Efficient Framework for Context-Aware Low-Rank Approximation 

**Authors**: Uliana Parkina, Maxim Rakhuba  

**Link**: [PDF](https://arxiv.org/pdf/2507.07580)  

**Abstract**: Recent studies suggest that context-aware low-rank approximation is a useful tool for compression and fine-tuning of modern large-scale neural networks. In this type of approximation, a norm is weighted by a matrix of input activations, significantly improving metrics over the unweighted case. Nevertheless, existing methods for neural networks suffer from numerical instabilities due to their reliance on classical formulas involving explicit Gram matrix computation and their subsequent inversion. We demonstrate that this can degrade the approximation quality or cause numerically singular matrices.
To address these limitations, we propose a novel inversion-free regularized framework that is based entirely on stable decompositions and overcomes the numerical pitfalls of prior art. Our method can handle possible challenging scenarios: (1) when calibration matrices exceed GPU memory capacity, (2) when input activation matrices are nearly singular, and even (3) when insufficient data prevents unique approximation. For the latter, we prove that our solution converges to a desired approximation and derive explicit error bounds. 

---
# May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks 

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes  

**Link**: [PDF](https://arxiv.org/pdf/2507.07417)  

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning the model to separate instructions and data, so that the LLM does not follow instructions that might be present with data. There are several academic systems and production-level implementations of this idea. We evaluate the robustness of this class of prompt injection defenses in the whitebox setting by constructing strong optimization-based attacks and showing that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for text-based LLMs and apply it to two recent whitebox defenses SecAlign (CCS 2025) and StruQ (USENIX Security 2025), showing attacks with success rates of up to 70% with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at this https URL 

---
# Bradley-Terry and Multi-Objective Reward Modeling Are Complementary 

**Authors**: Zhiwei Zhang, Hui Liu, Xiaomin Li, Zhenwei Dai, Jingying Zeng, Fali Wang, Minhua Lin, Ramraj Chandradevan, Zhen Li, Chen Luo, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07375)  

**Abstract**: Reward models trained on human preference data have demonstrated strong effectiveness in aligning Large Language Models (LLMs) with human intent under the framework of Reinforcement Learning from Human Feedback (RLHF). However, RLHF remains vulnerable to reward hacking, where the policy exploits imperfections in the reward function rather than genuinely learning the intended behavior. Although significant efforts have been made to mitigate reward hacking, they predominantly focus on and evaluate in-distribution scenarios, where the training and testing data for the reward model share the same distribution. In this paper, we empirically show that state-of-the-art methods struggle in more challenging out-of-distribution (OOD) settings. We further demonstrate that incorporating fine-grained multi-attribute scores helps address this challenge. However, the limited availability of high-quality data often leads to weak performance of multi-objective reward functions, which can negatively impact overall performance and become the bottleneck. To address this issue, we propose a unified reward modeling framework that jointly trains Bradley--Terry (BT) single-objective and multi-objective regression-based reward functions using a shared embedding space. We theoretically establish a connection between the BT loss and the regression objective and highlight their complementary benefits. Specifically, the regression task enhances the single-objective reward function's ability to mitigate reward hacking in challenging OOD settings, while BT-based training improves the scoring capability of the multi-objective reward function, enabling a 7B model to outperform a 70B baseline. Extensive experimental results demonstrate that our framework significantly improves both the robustness and the scoring performance of reward models. 

---
# ViDove: A Translation Agent System with Multimodal Context and Memory-Augmented Reasoning 

**Authors**: Yichen Lu, Wei Dai, Jiaen Liu, Ching Wing Kwok, Zongheng Wu, Xudong Xiao, Ao Sun, Sheng Fu, Jianyuan Zhan, Yian Wang, Takatomo Saito, Sicheng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2507.07306)  

**Abstract**: LLM-based translation agents have achieved highly human-like translation results and are capable of handling longer and more complex contexts with greater efficiency. However, they are typically limited to text-only inputs. In this paper, we introduce ViDove, a translation agent system designed for multimodal input. Inspired by the workflow of human translators, ViDove leverages visual and contextual background information to enhance the translation process. Additionally, we integrate a multimodal memory system and long-short term memory modules enriched with domain-specific knowledge, enabling the agent to perform more accurately and adaptively in real-world scenarios. As a result, ViDove achieves significantly higher translation quality in both subtitle generation and general translation tasks, with a 28% improvement in BLEU scores and a 15% improvement in SubER compared to previous state-of-the-art baselines. Moreover, we introduce DoveBench, a new benchmark for long-form automatic video subtitling and translation, featuring 17 hours of high-quality, human-annotated data. Our code is available here: this https URL 

---
# LinguaMark: Do Multimodal Models Speak Fairly? A Benchmark-Based Evaluation 

**Authors**: Ananya Raval, Aravind Narayanan, Vahid Reza Khazaie, Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2507.07274)  

**Abstract**: Large Multimodal Models (LMMs) are typically trained on vast corpora of image-text data but are often limited in linguistic coverage, leading to biased and unfair outputs across languages. While prior work has explored multimodal evaluation, less emphasis has been placed on assessing multilingual capabilities. In this work, we introduce LinguaMark, a benchmark designed to evaluate state-of-the-art LMMs on a multilingual Visual Question Answering (VQA) task. Our dataset comprises 6,875 image-text pairs spanning 11 languages and five social attributes. We evaluate models using three key metrics: Bias, Answer Relevancy, and Faithfulness. Our findings reveal that closed-source models generally achieve the highest overall performance. Both closed-source (GPT-4o and Gemini2.5) and open-source models (Gemma3, Qwen2.5) perform competitively across social attributes, and Qwen2.5 demonstrates strong generalization across multiple languages. We release our benchmark and evaluation code to encourage reproducibility and further research. 

---
# Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery 

**Authors**: Licong Xu, Milind Sarkar, Anto I. Lonappan, Íñigo Zubeldia, Pablo Villanueva-Domingo, Santiago Casas, Christian Fidler, Chetana Amancharla, Ujjwal Tiwari, Adrian Bayer, Chadi Ait Ekiou, Miles Cranmer, Adrian Dimitrov, James Fergusson, Kahaan Gandhi, Sven Krippendorf, Andrew Laverick, Julien Lesgourgues, Antony Lewis, Thomas Meier, Blake Sherwin, Kristen Surrao, Francisco Villaescusa-Navarro, Chi Wang, Xueqing Xu, Boris Bolliet  

**Link**: [PDF](https://arxiv.org/pdf/2507.07257)  

**Abstract**: We present a multi-agent system for automation of scientific research tasks, cmbagent. The system is formed by about 30 Large Language Model (LLM) agents and implements a Planning & Control strategy to orchestrate the agentic workflow, with no human-in-the-loop at any point. Each agent specializes in a different task (performing retrieval on scientific papers and codebases, writing code, interpreting results, critiquing the output of other agents) and the system is able to execute code locally. We successfully apply cmbagent to carry out a PhD level cosmology task (the measurement of cosmological parameters using supernova data) and evaluate its performance on two benchmark sets, finding superior performance over state-of-the-art LLMs. The source code is available on GitHub, demonstration videos are also available, and the system is deployed on HuggingFace and will be available on the cloud. 

---
# A Language-Driven Framework for Improving Personalized Recommendations: Merging LLMs with Traditional Algorithms 

**Authors**: Aaron Goldstein, Ayan Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2507.07251)  

**Abstract**: Traditional recommendation algorithms are not designed to provide personalized recommendations based on user preferences provided through text, e.g., "I enjoy light-hearted comedies with a lot of humor". Large Language Models (LLMs) have emerged as one of the most promising tools for natural language processing in recent years. This research proposes a novel framework that mimics how a close friend would recommend items based on their knowledge of an individual's tastes. We leverage LLMs to enhance movie recommendation systems by refining traditional algorithm outputs and integrating them with language-based user preference inputs. We employ Singular Value Decomposition (SVD) or SVD++ algorithms to generate initial movie recommendations, implemented using the Surprise Python library and trained on the MovieLens-Latest-Small dataset. We compare the performance of the base algorithms with our LLM-enhanced versions using leave-one-out validation hit rates and cumulative hit rates. Additionally, to compare the performance of our framework against the current state-of-the-art recommendation systems, we use rating and ranking metrics with an item-based stratified 0.75 train, 0.25 test split. Our framework can generate preference profiles automatically based on users' favorite movies or allow manual preference specification for more personalized results. Using an automated approach, our framework overwhelmingly surpassed SVD and SVD++ on every evaluation metric used (e.g., improvements of up to ~6x in cumulative hit rate, ~3.7x in NDCG, etc.), albeit at the cost of a slight increase in computational overhead. 

---
# An Information-Theoretic Perspective on Multi-LLM Uncertainty Estimation 

**Authors**: Maya Kruse, Majid Afshar, Saksham Khatwani, Anoop Mayampurath, Guanhua Chen, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.07236)  

**Abstract**: Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and naive ensemble baselines. 

---
# Robust Multimodal Large Language Models Against Modality Conflict 

**Authors**: Zongmeng Zhang, Wengang Zhou, Jie Zhao, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.07151)  

**Abstract**: Despite the impressive capabilities of multimodal large language models (MLLMs) in vision-language tasks, they are prone to hallucinations in real-world scenarios. This paper investigates the hallucination phenomenon in MLLMs from the perspective of modality conflict. Unlike existing works focusing on the conflicts between model responses and inputs, we study the inherent conflicts in inputs from different modalities that place MLLMs in a dilemma and directly lead to hallucinations. We formally define the modality conflict and construct a dataset named Multimodal Modality Conflict (MMMC) to simulate this phenomenon in vision-language tasks. Three methods based on prompt engineering, supervised fine-tuning, and reinforcement learning are proposed to alleviate the hallucination caused by modality conflict. Extensive experiments are conducted on the MMMC dataset to analyze the merits and demerits of these methods. Our results show that the reinforcement learning method achieves the best performance in mitigating the hallucination under modality conflict, while the supervised fine-tuning method shows promising and stable performance. Our work sheds light on the unnoticed modality conflict that leads to hallucinations and provides more insights into the robustness of MLLMs. 

---
# Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.07129)  

**Abstract**: The prevailing paradigm for scaling large language models (LLMs) involves monolithic, end-to-end training, a resource-intensive process that lacks flexibility. This paper explores an alternative, constructive approach to model development, built upon the foundation of non-trainable, deterministic input embeddings. In prior [1], we established that high-level semantic reasoning can emerge in Transformers using frozen embeddings derived from the visual structure of Unicode glyphs. Here, we demonstrate that this fixed representational substrate acts as a universal "docking port," enabling two powerful and efficient scaling paradigms: seamless modular composition and progressive layer-wise growth.
First, we show that specialist models trained on disparate datasets (e.g., Russian and Chinese text) can be merged into a single, more capable Mixture-of-Experts (MoE) model, post-training, with zero architectural modification. This is achieved by simply averaging their output logits. The resulting MoE model exhibits immediate performance improvements on reasoning benchmarks like MMLU, surpassing its constituent experts without catastrophic forgetting. Second, we introduce a layer-wise constructive training methodology, where a deep Transformer is "grown" by progressively stacking and training one layer at a time. This method demonstrates stable convergence and a clear correlation between model depth and the emergence of complex reasoning abilities, such as those required for SQuAD.
Our findings suggest a paradigm shift from monolithic optimization towards a more biological or constructive model of AI development, where complexity is built incrementally and modules can be composed freely. This opens new avenues for resource-efficient scaling, continual learning, and a more democratized ecosystem for building powerful AI systems. We release all code and models to facilitate further research. 

---
# Multi-level Mixture of Experts for Multimodal Entity Linking 

**Authors**: Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Ru Li, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07108)  

**Abstract**: Multimodal Entity Linking (MEL) aims to link ambiguous mentions within multimodal contexts to associated entities in a multimodal knowledge base. Existing approaches to MEL introduce multimodal interaction and fusion mechanisms to bridge the modality gap and enable multi-grained semantic matching. However, they do not address two important problems: (i) mention ambiguity, i.e., the lack of semantic content caused by the brevity and omission of key information in the mention's textual context; (ii) dynamic selection of modal content, i.e., to dynamically distinguish the importance of different parts of modal information. To mitigate these issues, we propose a Multi-level Mixture of Experts (MMoE) model for MEL. MMoE has four components: (i) the description-aware mention enhancement module leverages large language models to identify the WikiData descriptions that best match a mention, considering the mention's textual context; (ii) the multimodal feature extraction module adopts multimodal feature encoders to obtain textual and visual embeddings for both mentions and entities; (iii)-(iv) the intra-level mixture of experts and inter-level mixture of experts modules apply a switch mixture of experts mechanism to dynamically and adaptively select features from relevant regions of information. Extensive experiments demonstrate the outstanding performance of MMoE compared to the state-of-the-art. MMoE's code is available at: this https URL. 

---
