# Value-Aware Numerical Representations for Transformer Language Models 

**Authors**: Andreea Dutulescu, Stefan Ruseti, Mihai Dascalu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09706)  

**Abstract**: Transformer-based language models often achieve strong results on mathematical reasoning benchmarks while remaining fragile on basic numerical understanding and arithmetic operations. A central limitation is that numbers are processed as symbolic tokens whose embeddings do not explicitly encode numerical value, leading to systematic errors. We introduce a value-aware numerical representation that augments standard tokenized inputs with a dedicated prefix token whose embedding is explicitly conditioned on the underlying numerical value. This mechanism injects magnitude information directly into the model's input space while remaining compatible with existing tokenizers and decoder-only Transformer architectures. Evaluation on arithmetic tasks shows that the proposed approach outperforms baselines across numerical formats, tasks, and operand lengths. These results indicate that explicitly encoding numerical value is an effective and efficient way to improve fundamental numerical robustness in language models. 

---
# Empathy Applicability Modeling for General Health Queries 

**Authors**: Shan Randhawa, Agha Ali Raza, Kentaro Toyama, Julie Hui, Mustafa Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2601.09696)  

**Abstract**: LLMs are increasingly being integrated into clinical workflows, yet they often lack clinical empathy, an essential aspect of effective doctor-patient communication. Existing NLP frameworks focus on reactively labeling empathy in doctors' responses but offer limited support for anticipatory modeling of empathy needs, especially in general health queries. We introduce the Empathy Applicability Framework (EAF), a theory-driven approach that classifies patient queries in terms of the applicability of emotional reactions and interpretations, based on clinical, contextual, and linguistic cues. We release a benchmark of real patient queries, dual-annotated by Humans and GPT-4o. In the subset with human consensus, we also observe substantial human-GPT alignment. To validate EAF, we train classifiers on human-labeled and GPT-only annotations to predict empathy applicability, achieving strong performance and outperforming the heuristic and zero-shot LLM baselines. Error analysis highlights persistent challenges: implicit distress, clinical-severity ambiguity, and contextual hardship, underscoring the need for multi-annotator modeling, clinician-in-the-loop calibration, and culturally diverse annotation. EAF provides a framework for identifying empathy needs before response generation, establishes a benchmark for anticipatory empathy modeling, and enables supporting empathetic communication in asynchronous healthcare. 

---
# LLMs can Compress LLMs: Adaptive Pruning by Agents 

**Authors**: Sai Varun Kodathala, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2601.09694)  

**Abstract**: As Large Language Models (LLMs) continue to scale, post-training pruning has emerged as a promising approach to reduce computational costs while preserving performance. Existing methods such as SparseGPT and Wanda achieve high sparsity through layer-wise weight reconstruction or activation-aware magnitude pruning, but rely on uniform or hand-crafted heuristics to determine per-layer sparsity ratios. Moreover, recent work has shown that pruned LLMs suffer from severe factual knowledge degradation, with structured pruning methods experiencing near-total collapse in factual question-answering capabilities. We introduce agent-guided pruning, where a foundation model acts as an adaptive pruning agent to intelligently select which layers to prune at each iteration while preserving critical knowledge pathways. Our method constructs layer-wise sensitivity profiles by combining Wanda-inspired weight-activation metrics with gradient importance scores, normalized as z-scores for model-agnostic comparison. These statistics are processed by an LLM agent equipped with self-reflection capabilities, enabling it to learn from previous pruning outcomes and iteratively refine its strategy. A checkpoint rollback mechanism maintains model quality by reverting when perplexity degradation exceeds a threshold. We evaluate our approach on Qwen3 models (4B and 8B parameters) at approximately 45% sparsity, demonstrating substantial improvements over structured pruning baselines: 56% relative improvement in MMLU accuracy, 19x better factual knowledge retention on FreebaseQA, and 69% lower perplexity degradation. Notably, our framework requires no retraining, operates in a model-agnostic manner, and exhibits effective self-correction with only 2-4 rollbacks across 21-40 iterations, demonstrating that foundation models can effectively guide the compression of other foundation models. 

---
# Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection 

**Authors**: Tianyi Niu, Justin Chih-Yao Chen, Genta Indra Winata, Shi-Xiong Zhang, Supriyo Chakraborty, Sambit Sahu, Yue Zhang, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2601.09692)  

**Abstract**: Large Language Model (LLM) routers dynamically select optimal models for given inputs. Existing approaches typically assume access to ground-truth labeled data, which is often unavailable in practice, especially when user request distributions are heterogeneous and unknown. We introduce Routing with Generated Data (RGD), a challenging setting in which routers are trained exclusively on generated queries and answers produced from high-level task descriptions by generator LLMs. We evaluate query-answer routers (using both queries and labels) and query-only routers across four diverse benchmarks and 12 models, finding that query-answer routers degrade faster than query-only routers as generator quality decreases. Our analysis reveals two crucial characteristics of effective generators: they must accurately respond to their own questions, and their questions must produce sufficient performance differentiation among the model pool. We then show how filtering for these characteristics can improve the quality of generated data. We further propose CASCAL, a novel query-only router that estimates model correctness through consensus voting and identifies model-specific skill niches via hierarchical clustering. CASCAL is substantially more robust to generator quality, outperforming the best query-answer router by 4.6% absolute accuracy when trained on weak generator data. 

---
# DeepResearchEval: An Automated Framework for Deep Research Task Construction and Agentic Evaluation 

**Authors**: Yibo Wang, Lei Wang, Yue Deng, Keming Wu, Yao Xiao, Huanjin Yao, Liwei Kang, Hai Ye, Yongcheng Jing, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2601.09688)  

**Abstract**: Deep research systems are widely used for multi-step web research, analysis, and cross-source synthesis, yet their evaluation remains challenging. Existing benchmarks often require annotation-intensive task construction, rely on static evaluation dimensions, or fail to reliably verify facts when citations are missing. To bridge these gaps, we introduce DeepResearchEval, an automated framework for deep research task construction and agentic evaluation. For task construction, we propose a persona-driven pipeline generating realistic, complex research tasks anchored in diverse user profiles, applying a two-stage filter Task Qualification and Search Necessity to retain only tasks requiring multi-source evidence integration and external retrieval. For evaluation, we propose an agentic pipeline with two components: an Adaptive Point-wise Quality Evaluation that dynamically derives task-specific evaluation dimensions, criteria, and weights conditioned on each generated task, and an Active Fact-Checking that autonomously extracts and verifies report statements via web search, even when citations are missing. 

---
# Creating a Hybrid Rule and Neural Network Based Semantic Tagger using Silver Standard Data: the PyMUSAS framework for Multilingual Semantic Annotation 

**Authors**: Andrew Moore, Paul Rayson, Dawn Archer, Tim Czerniak, Dawn Knight, Daisy Lal, Gearóid Ó Donnchadha, Mícheál Ó Meachair, Scott Piao, Elaine Uí Dhonnchadha, Johanna Vuorinen, Yan Yabo, Xiaobin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09648)  

**Abstract**: Word Sense Disambiguation (WSD) has been widely evaluated using the semantic frameworks of WordNet, BabelNet, and the Oxford Dictionary of English. However, for the UCREL Semantic Analysis System (USAS) framework, no open extensive evaluation has been performed beyond lexical coverage or single language evaluation. In this work, we perform the largest semantic tagging evaluation of the rule based system that uses the lexical resources in the USAS framework covering five different languages using four existing datasets and one novel Chinese dataset. We create a new silver labelled English dataset, to overcome the lack of manually tagged training data, that we train and evaluate various mono and multilingual neural models in both mono and cross-lingual evaluation setups with comparisons to their rule based counterparts, and show how a rule based system can be enhanced with a neural network model. The resulting neural network models, including the data they were trained on, the Chinese evaluation dataset, and all of the code have been released as open resources. 

---
# TaxoBell: Gaussian Box Embeddings for Self-Supervised Taxonomy Expansion 

**Authors**: Sahil Mishra, Srinitish Srinivasan, Srikanta Bedathur, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2601.09633)  

**Abstract**: Taxonomies form the backbone of structured knowledge representation across diverse domains, enabling applications such as e-commerce catalogs, semantic search, and biomedical discovery. Yet, manual taxonomy expansion is labor-intensive and cannot keep pace with the emergence of new concepts. Existing automated methods rely on point-based vector embeddings, which model symmetric similarity and thus struggle with the asymmetric "is-a" relationships that are fundamental to taxonomies. Box embeddings offer a promising alternative by enabling containment and disjointness, but they face key issues: (i) unstable gradients at the intersection boundaries, (ii) no notion of semantic uncertainty, and (iii) limited capacity to represent polysemy or ambiguity. We address these shortcomings with TaxoBell, a Gaussian box embedding framework that translates between box geometries and multivariate Gaussian distributions, where means encode semantic location and covariances encode uncertainty. Energy-based optimization yields stable optimization, robust modeling of ambiguous concepts, and interpretable hierarchical reasoning. Extensive experimentation on five benchmark datasets demonstrates that TaxoBell significantly outperforms eight state-of-the-art taxonomy expansion baselines by 19% in MRR and around 25% in Recall@k. We further demonstrate the advantages and pitfalls of TaxoBell with error analysis and ablation studies. 

---
# LLMs Got Rhythm? Hybrid Phonological Filtering for Greek Poetry Rhyme Detection and Generation 

**Authors**: Stergios Chatzikyriakidis  

**Link**: [PDF](https://arxiv.org/pdf/2601.09631)  

**Abstract**: Large Language Models (LLMs), despite their remarkable capabilities across NLP tasks, struggle with phonologically-grounded phenomena like rhyme detection and generation. This is even more evident in lower-resource languages such as Modern Greek. In this paper, we present a hybrid system that combines LLMs with deterministic phonological algorithms to achieve accurate rhyme identification/analysis and generation. Our approach implements a comprehensive taxonomy of Greek rhyme types, including Pure, Rich, Imperfect, Mosaic, and Identical Pre-rhyme Vowel (IDV) patterns, and employs an agentic generation pipeline with phonological verification. We evaluate multiple prompting strategies (zero-shot, few-shot, Chain-of-Thought, and RAG-augmented) across several LLMs including Claude 3.7 and 4.5, GPT-4o, Gemini 2.0 and open-weight models like Llama 3.1 8B and 70B and Mistral Large. Results reveal a significant "Reasoning Gap": while native-like models (Claude 3.7) perform intuitively (40\% accuracy in identification), reasoning-heavy models (Claude 4.5) achieve state-of-the-art performance (54\%) only when prompted with Chain-of-Thought. Most critically, pure LLM generation fails catastrophically (under 4\% valid poems), while our hybrid verification loop restores performance to 73.1\%. We release our system and a crucial, rigorously cleaned corpus of 40,000+ rhymes, derived from the Anemoskala and Interwar Poetry corpora, to support future research. 

---
# DPWriter: Reinforcement Learning with Diverse Planning Branching for Creative Writing 

**Authors**: Qian Cao, Yahui Liu, Wei Bi, Yi Zhao, Ruihua Song, Xiting Wang, Ruiming Tang, Guorui Zhou, Han Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09609)  

**Abstract**: Reinforcement learning (RL)-based enhancement of large language models (LLMs) often leads to reduced output diversity, undermining their utility in open-ended tasks like creative writing. Current methods lack explicit mechanisms for guiding diverse exploration and instead prioritize optimization efficiency and performance over diversity. This paper proposes an RL framework structured around a semi-structured long Chain-of-Thought (CoT), in which the generation process is decomposed into explicitly planned intermediate steps. We introduce a Diverse Planning Branching method that strategically introduces divergence at the planning phase based on diversity variation, alongside a group-aware diversity reward to encourage distinct trajectories. Experimental results on creative writing benchmarks demonstrate that our approach significantly improves output diversity without compromising generation quality, consistently outperforming existing baselines. 

---
# Dialogue Telemetry: Turn-Level Instrumentation for Autonomous Information Gathering 

**Authors**: Dimitris Panagopoulos, Adolfo Perrusquia, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2601.09570)  

**Abstract**: Autonomous systems conducting schema-grounded information-gathering dialogues face an instrumentation gap, lacking turn-level observables for monitoring acquisition efficiency and detecting when questioning becomes unproductive. We introduce Dialogue Telemetry (DT), a measurement framework that produces two model-agnostic signals after each question-answer exchange: (i) a Progress Estimator (PE) quantifying residual information potential per category (with a bits-based variant), and (ii) a Stalling Index (SI) detecting an observable failure signature characterized by repeated category probing with semantically similar, low-marginal-gain responses. SI flags this pattern without requiring causal diagnosis, supporting monitoring in settings where attributing degradation to specific causes may be impractical. We validate DT in controlled search-and-rescue (SAR)-inspired interviews using large language model (LLM)-based simulations, distinguishing efficient from stalled dialogue traces and illustrating downstream utility by integrating DT signals into a reinforcement learning (RL) policy. Across these settings, DT provides interpretable turn-level instrumentation that improves policy performance when stalling carries operational costs. 

---
# Benchmarking Post-Training Quantization of Large Language Models under Microscaling Floating Point Formats 

**Authors**: Manyi Zhang, Ji-Fu Li, Zhongao Sun, Haoli Bai, Hui-Ling Zhen, Zhenhua Dong, Xianzhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09555)  

**Abstract**: Microscaling Floating-Point (MXFP) has emerged as a promising low-precision format for large language models (LLMs). Despite various post-training quantization (PTQ) algorithms being proposed, they mostly focus on integer quantization, while their applicability and behavior under MXFP formats remain largely unexplored. To address this gap, this work conducts a systematic investigation of PTQ under MXFP formats, encompassing over 7 PTQ algorithms, 15 evaluation benchmarks, and 3 LLM families. The key findings include: 1) MXFP8 consistently achieves near-lossless performance, while MXFP4 introduces substantial accuracy degradation and remains challenging; 2) PTQ effectiveness under MXFP depends strongly on format compatibility, with some algorithmic paradigms being consistently more effective than others; 3) PTQ performance exhibits highly consistent trends across model families and modalities, in particular, quantization sensitivity is dominated by the language model rather than the vision encoder in multimodal LLMs; 4) The scaling factor of quantization is a critical error source in MXFP4, and a simple pre-scale optimization strategy can significantly mitigate its impact. Together, these results provide practical guidance on adapting existing PTQ methods to MXFP quantization. 

---
# SERM: Self-Evolving Relevance Model with Agent-Driven Learning from Massive Query Streams 

**Authors**: Chenglong Wang, Canjia Li, Xingzhao Zhu, Yifu Huo, Huiyu Wang, Weixiong Lin, Yun Yang, Qiaozhi He, Tianhua Zhou, Xiaojia Chang, Jingbo Zhu, Tong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.09515)  

**Abstract**: Due to the dynamically evolving nature of real-world query streams, relevance models struggle to generalize to practical search scenarios. A sophisticated solution is self-evolution techniques. However, in large-scale industrial settings with massive query streams, this technique faces two challenges: (1) informative samples are often sparse and difficult to identify, and (2) pseudo-labels generated by the current model could be unreliable. To address these challenges, in this work, we propose a Self-Evolving Relevance Model approach (SERM), which comprises two complementary multi-agent modules: a multi-agent sample miner, designed to detect distributional shifts and identify informative training samples, and a multi-agent relevance annotator, which provides reliable labels through a two-level agreement framework. We evaluate SERM in a large-scale industrial setting, which serves billions of user requests daily. Experimental results demonstrate that SERM can achieve significant performance gains through iterative self-evolution, as validated by extensive offline multilingual evaluations and online testing. 

---
# MVSS: A Unified Framework for Multi-View Structured Survey Generation 

**Authors**: Yinqi Liu, Yueqi Zhu, Yongkang Zhang, Xinfeng Li, Feiran Liu, Yufei Sun, Xin Wang, Renzhao Liang, Yidong Wang, Cunxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09504)  

**Abstract**: Scientific surveys require not only summarizing large bodies of literature, but also organizing them into clear and coherent conceptual structures. Existing automatic survey generation methods typically focus on linear text generation and struggle to explicitly model hierarchical relations among research topics and structured methodological comparisons, resulting in gaps in structural organization compared to expert-written surveys. We propose MVSS, a multi-view structured survey generation framework that jointly generates and aligns citation-grounded hierarchical trees, structured comparison tables, and survey text. MVSS follows a structure-first paradigm: it first constructs a conceptual tree of the research domain, then generates comparison tables constrained by the tree, and finally uses both as structural constraints for text generation. This enables complementary multi-view representations across structure, comparison, and narrative. We introduce an evaluation framework assessing structural quality, comparative completeness, and citation fidelity. Experiments on 76 computer science topics show MVSS outperforms existing methods in organization and evidence grounding, achieving performance comparable to expert surveys. 

---
# SlidesGen-Bench: Evaluating Slides Generation via Computational and Quantitative Metrics 

**Authors**: Yunqiao Yang, Wenbo Li, Houxing Ren, Zimu Lu, Ke Wang, Zhiyuan Huang, Zhuofan Zong, Mingjie Zhan, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09487)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) has fostered diverse paradigms for automated slide generation, ranging from code-driven layouts to image-centric synthesis. However, evaluating these heterogeneous systems remains challenging, as existing protocols often struggle to provide comparable scores across architectures or rely on uncalibrated judgments. In this paper, we introduce SlidesGen-Bench, a benchmark designed to evaluate slide generation through a lens of three core principles: universality, quantification, and reliability. First, to establish a unified evaluation framework, we ground our analysis in the visual domain, treating terminal outputs as renderings to remain agnostic to the underlying generation method. Second, we propose a computational approach that quantitatively assesses slides across three distinct dimensions - Content, Aesthetics, and Editability - offering reproducible metrics where prior works relied on subjective or reference-dependent proxies. Finally, to ensure high correlation with human preference, we construct the Slides-Align1.5k dataset, a human preference aligned dataset covering slides from nine mainstream generation systems across seven scenarios. Our experiments demonstrate that SlidesGen-Bench achieves a higher degree of alignment with human judgment than existing evaluation pipelines. Our code and data are available at this https URL. 

---
# Improving Symbolic Translation of Language Models for Logical Reasoning 

**Authors**: Ramya Keerthy Thatikonda, Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09446)  

**Abstract**: The use of formal language for deductive logical reasoning aligns well with language models (LMs), where translating natural language (NL) into first-order logic (FOL) and employing an external solver results in a verifiable and therefore reliable reasoning system. However, smaller LMs often struggle with this translation task, frequently producing incorrect symbolic outputs due to formatting and translation errors. Existing approaches typically rely on self-iteration to correct these errors, but such methods depend heavily on the capabilities of the underlying model. To address this, we first categorize common errors and fine-tune smaller LMs using data synthesized by large language models. The evaluation is performed using the defined error categories. We introduce incremental inference, which divides inference into two stages, predicate generation and FOL translation, providing greater control over model behavior and enhancing generation quality as measured by predicate metrics. This decomposition framework also enables the use of a verification module that targets predicate-arity errors to further improve performance. Our study evaluates three families of models across four logical-reasoning datasets. The comprehensive fine-tuning, incremental inference, and verification modules reduce error rates, increase predicate coverage, and improve reasoning performance for smaller LMs, moving us closer to developing reliable and accessible symbolic-reasoning systems. 

---
# Where Knowledge Collides: A Mechanistic Study of Intra-Memory Knowledge Conflict in Language Models 

**Authors**: Minh Vu Pham, Hsuvas Borkakoty, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2601.09445)  

**Abstract**: In language models (LMs), intra-memory knowledge conflict largely arises when inconsistent information about the same event is encoded within the model's parametric knowledge. While prior work has primarily focused on resolving conflicts between a model's internal knowledge and external resources through approaches such as fine-tuning or knowledge editing, the problem of localizing conflicts that originate during pre-training within the model's internal representations remain unexplored. In this work, we design a framework based on mechanistic interpretability methods to identify where and how conflicting knowledge from the pre-training data is encoded within LMs. Our findings contribute to a growing body of evidence that specific internal components of a language model are responsible for encoding conflicting knowledge from pre-training, and we demonstrate how mechanistic interpretability methods can be leveraged to causally intervene in and control conflicting knowledge at inference time. 

---
# Bias Dynamics in BabyLMs: Towards a Compute-Efficient Sandbox for Democratising Pre-Training Debiasing 

**Authors**: Filip Trhlik, Andrew Caines, Paula Buttery  

**Link**: [PDF](https://arxiv.org/pdf/2601.09421)  

**Abstract**: Pre-trained language models (LMs) have, over the last few years, grown substantially in both societal adoption and training costs. This rapid growth in size has constrained progress in understanding and mitigating their biases. Since re-training LMs is prohibitively expensive, most debiasing work has focused on post-hoc or masking-based strategies, which often fail to address the underlying causes of bias. In this work, we seek to democratise pre-model debiasing research by using low-cost proxy models. Specifically, we investigate BabyLMs, compact BERT-like models trained on small and mutable corpora that can approximate bias acquisition and learning dynamics of larger models. We show that BabyLMs display closely aligned patterns of intrinsic bias formation and performance development compared to standard BERT models, despite their drastically reduced size. Furthermore, correlations between BabyLMs and BERT hold across multiple intra-model and post-model debiasing methods. Leveraging these similarities, we conduct pre-model debiasing experiments with BabyLMs, replicating prior findings and presenting new insights regarding the influence of gender imbalance and toxicity on bias formation. Our results demonstrate that BabyLMs can serve as an effective sandbox for large-scale LMs, reducing pre-training costs from over 500 GPU-hours to under 30 GPU-hours. This provides a way to democratise pre-model debiasing research and enables faster, more accessible exploration of methods for building fairer LMs. 

---
# Structured Knowledge Representation through Contextual Pages for Retrieval-Augmented Generation 

**Authors**: Xinze Li, Zhenghao Liu, Haidong Xin, Yukun Yan, Shuo Wang, Zheni Zeng, Sen Mei, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09402)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge. Recently, some works have incorporated iterative knowledge accumulation processes into RAG models to progressively accumulate and refine query-related knowledge, thereby constructing more comprehensive knowledge representations. However, these iterative processes often lack a coherent organizational structure, which limits the construction of more comprehensive and cohesive knowledge representations. To address this, we propose PAGER, a page-driven autonomous knowledge representation framework for RAG. PAGER first prompts an LLM to construct a structured cognitive outline for a given question, which consists of multiple slots representing a distinct knowledge aspect. Then, PAGER iteratively retrieves and refines relevant documents to populate each slot, ultimately constructing a coherent page that serves as contextual input for guiding answer generation. Experiments on multiple knowledge-intensive benchmarks and backbone models show that PAGER consistently outperforms all RAG baselines. Further analyses demonstrate that PAGER constructs higher-quality and information-dense knowledge representations, better mitigates knowledge conflicts, and enables LLMs to leverage external knowledge more effectively. All code is available at this https URL. 

---
# Ability Transfer and Recovery via Modularized Parameters Localization 

**Authors**: Songyao Jin, Kun Zhou, Wenqi Li, Peng Wang, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09398)  

**Abstract**: Large language models can be continually pre-trained or fine-tuned to improve performance in specific domains, languages, or skills, but this specialization often degrades other capabilities and may cause catastrophic forgetting. We investigate how abilities are distributed within LLM parameters by analyzing module activations under domain- and language-specific inputs for closely related models. Across layers and modules, we find that ability-related activations are highly concentrated in a small set of channels (typically <5\%), and these channels are largely disentangled with good sufficiency and stability. Building on these observations, we propose ACT (Activation-Guided Channel-wise Ability Transfer), which localizes ability-relevant channels via activation differences and selectively transfers only the corresponding parameters, followed by lightweight fine-tuning for compatibility. Experiments on multilingual mathematical and scientific reasoning show that ACT can recover forgotten abilities while preserving retained skills. It can also merge multiple specialized models to integrate several abilities into a single model with minimal interference. Our code and data will be publicly released. 

---
# The Imperfective Paradox in Large Language Models 

**Authors**: Bolei Ma, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2601.09373)  

**Abstract**: Do Large Language Models (LLMs) genuinely grasp the compositional semantics of events, or do they rely on surface-level probabilistic heuristics? We investigate the Imperfective Paradox, a logical phenomenon where the past progressive aspect entails event realization for activities (e.g., running $\to$ ran) but not for accomplishments (e.g., building $\nrightarrow$ built). We introduce ImperfectiveNLI, a diagnostic dataset designed to probe this distinction across diverse semantic classes. Evaluating state-of-the-art open-weight models, we uncover a pervasive Teleological Bias: models systematically hallucinate completion for goal-oriented events, often overriding explicit textual negation. Representational analyses show that while internal embeddings often distinguish process from result, inference decisions are dominated by strong priors about goal attainment. We further find that prompting-based interventions reduce hallucinated completions but also increase incorrect rejections of valid entailments. Our findings suggest that current LLMs lack structural aspectual awareness, operating as predictive narrative engines rather than faithful logical reasoners. 

---
# Relation Extraction Capabilities of LLMs on Clinical Text: A Bilingual Evaluation for English and Turkish 

**Authors**: Aidana Aidynkyzy, Oğuz Dikenelli, Oylum Alatlı, Şebnem Bora  

**Link**: [PDF](https://arxiv.org/pdf/2601.09367)  

**Abstract**: The scarcity of annotated datasets for clinical information extraction in non-English languages hinders the evaluation of large language model (LLM)-based methods developed primarily in English. In this study, we present the first comprehensive bilingual evaluation of LLMs for the clinical Relation Extraction (RE) task in both English and Turkish. To facilitate this evaluation, we introduce the first English-Turkish parallel clinical RE dataset, derived and carefully curated from the 2010 i2b2/VA relation classification corpus. We systematically assess a diverse set of prompting strategies, including multiple in-context learning (ICL) and Chain-of-Thought (CoT) approaches, and compare their performance to fine-tuned baselines such as PURE. Furthermore, we propose Relation-Aware Retrieval (RAR), a novel in-context example selection method based on contrastive learning, that is specifically designed to capture both sentence-level and relation-level semantics. Our results show that prompting-based LLM approaches consistently outperform traditional fine-tuned models. Moreover, evaluations for English performed better than their Turkish counterparts across all evaluated LLMs and prompting techniques. Among ICL methods, RAR achieves the highest performance, with Gemini 1.5 Flash reaching a micro-F1 score of 0.906 in English and 0.888 in Turkish. Performance further improves to 0.918 F1 in English when RAR is combined with a structured reasoning prompt using the DeepSeek-V3 model. These findings highlight the importance of high-quality demonstration retrieval and underscore the potential of advanced retrieval and prompting techniques to bridge resource gaps in clinical natural language processing. 

---
# Frame of Reference: Addressing the Challenges of Common Ground Representation in Situational Dialogs 

**Authors**: Biswesh Mohapatra, Théo Charlot, Giovanni Duca, Mayank Palan, Laurent Romary, Justine Cassell  

**Link**: [PDF](https://arxiv.org/pdf/2601.09365)  

**Abstract**: Common ground plays a critical role in situated spoken dialogues, where interlocutors must establish and maintain shared references to entities, events, and relations to sustain coherent interaction. For dialog systems, the ability to correctly ground conversational content in order to refer back to it later is particularly important. Prior studies have demonstrated that LLMs are capable of performing grounding acts such as requesting clarification or producing acknowledgments, yet relatively little work has investigated how common ground can be explicitly represented and stored for later use. Without such mechanisms, it remains unclear whether acknowledgment or clarification behaviors truly reflect a grounded understanding. In this work, we evaluate a model's ability to establish and exploit common ground through relational references to entities within the shared context in a situational dialogue. We test multiple methods for representing common ground in situated dialogues and further propose approaches to improve both the establishment of common ground and its subsequent use in the conversation. 

---
# Improving Implicit Hate Speech Detection via a Community-Driven Multi-Agent Framework 

**Authors**: Ewelina Gajewska, Katarzyna Budzynska, Jarosław A Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2601.09342)  

**Abstract**: This work proposes a contextualised detection framework for implicitly hateful speech, implemented as a multi-agent system comprising a central Moderator Agent and dynamically constructed Community Agents representing specific demographic groups. Our approach explicitly integrates socio-cultural context from publicly available knowledge sources, enabling identity-aware moderation that surpasses state-of-the-art prompting methods (zero-shot prompting, few-shot prompting, chain-of-thought prompting) and alternative approaches on a challenging ToxiGen dataset. We enhance the technical rigour of performance evaluation by incorporating balanced accuracy as a central metric of classification fairness that accounts for the trade-off between true positive and true negative rates. We demonstrate that our community-driven consultative framework significantly improves both classification accuracy and fairness across all target groups. 

---
# Understanding or Memorizing? A Case Study of German Definite Articles in Language Models 

**Authors**: Jonathan Drechsel, Erisa Bytyqi, Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2601.09313)  

**Abstract**: Language models perform well on grammatical agreement, but it is unclear whether this reflects rule-based generalization or memorization. We study this question for German definite singular articles, whose forms depend on gender and case. Using GRADIEND, a gradient-based interpretability method, we learn parameter update directions for gender-case specific article transitions. We find that updates learned for a specific gender-case article transition frequently affect unrelated gender-case settings, with substantial overlap among the most affected neurons across settings. These results argue against a strictly rule-based encoding of German definite articles, indicating that models at least partly rely on memorized associations rather than abstract grammatical rules. 

---
# ReGraM: Region-First Knowledge Graph Reasoning for Medical Question Answering 

**Authors**: Chaerin Lee, Sohee Park, Hyunsik Na, Daseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09280)  

**Abstract**: Recent studies in medical question answering (Medical QA) have actively explored the integration of large language models (LLMs) with biomedical knowledge graphs (KGs) to improve factual accuracy. However, most existing approaches still rely on traversing the entire KG or performing large-scale retrieval, which introduces substantial noise and leads to unstable multi-hop reasoning. We argue that the core challenge lies not in expanding access to knowledge, but in identifying and reasoning over the appropriate subset of evidence for each query. ReGraM is a region-first knowledge graph reasoning framework that addresses this challenge by constructing a query-aligned subgraph and performing stepwise reasoning constrained to this localized region under multiple evidence aware modes. By focusing inference on only the most relevant portion of the KG, ReGraM departs from the assumption that all relations are equally useful an assumption that rarely holds in domain-specific medical settings. Experiments on seven medical QA benchmarks demonstrate that ReGraM consistently outperforms a strong baseline (KGARevion), achieving an 8.04% absolute accuracy gain on MCQ, a 4.50% gain on SAQ, and a 42.9% reduction in hallucination rate. Ablation and qualitative analyses further show that aligning region construction with hop-wise reasoning is the primary driver of these improvements. Overall, our results highlight region-first KG reasoning as an effective paradigm for improving factual accuracy and consistency in medical QA. 

---
# MCGA: A Multi-task Classical Chinese Literary Genre Audio Corpus 

**Authors**: Yexing Du, Kaiyuan Liu, Bihe Zhang, Youcheng Pan, Bo Yang, Liangyu Huo, Xiyuan Zhang, Jian Xie, Daojing He, Yang Xiang, Ming Liu, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.09270)  

**Abstract**: With the rapid advancement of Multimodal Large Language Models (MLLMs), their potential has garnered significant attention in Chinese Classical Studies (CCS). While existing research has primarily focused on text and visual modalities, the audio corpus within this domain remains largely underexplored. To bridge this gap, we propose the Multi-task Classical Chinese Literary Genre Audio Corpus (MCGA). It encompasses a diverse range of literary genres across six tasks: Automatic Speech Recognition (ASR), Speech-to-Text Translation (S2TT), Speech Emotion Captioning (SEC), Spoken Question Answering (SQA), Speech Understanding (SU), and Speech Reasoning (SR). Through the evaluation of ten MLLMs, our experimental results demonstrate that current models still face substantial challenges when processed on the MCGA test set. Furthermore, we introduce an evaluation metric for SEC and a metric to measure the consistency between the speech and text capabilities of MLLMs. We release MCGA and our code to the public to facilitate the development of MLLMs with more robust multidimensional audio capabilities in CCS. MCGA Corpus: this https URL 

---
# When to Invoke: Refining LLM Fairness with Toxicity Assessment 

**Authors**: Jing Ren, Bowen Li, Ziqi Xu, Renqiang Luo, Shuo Yu, Xin Ye, Haytham Fayek, Xiaodong Li, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2601.09250)  

**Abstract**: Large Language Models (LLMs) are increasingly used for toxicity assessment in online moderation systems, where fairness across demographic groups is essential for equitable treatment. However, LLMs often produce inconsistent toxicity judgements for subtle expressions, particularly those involving implicit hate speech, revealing underlying biases that are difficult to correct through standard training. This raises a key question that existing approaches often overlook: when should corrective mechanisms be invoked to ensure fair and reliable assessments? To address this, we propose FairToT, an inference-time framework that enhances LLM fairness through prompt-guided toxicity assessment. FairToT identifies cases where demographic-related variation is likely to occur and determines when additional assessment should be applied. In addition, we introduce two interpretable fairness indicators that detect such cases and improve inference consistency without modifying model parameters. Experiments on benchmark datasets show that FairToT reduces group-level disparities while maintaining stable and reliable toxicity predictions, demonstrating that inference-time refinement offers an effective and practical approach for fairness improvement in LLM-based toxicity assessment systems. The source code can be found at this https URL. 

---
# TeachPro: Multi-Label Qualitative Teaching Evaluation via Cross-View Graph Synergy and Semantic Anchored Evidence Encoding 

**Authors**: Xiangqian Wang, Yifan Jia, Yang Xiang, Yumin Zhang, Yanbin Wang, Ke Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09246)  

**Abstract**: Standardized Student Evaluation of Teaching often suffer from low reliability, restricted response options, and response distortion. Existing machine learning methods that mine open-ended comments usually reduce feedback to binary sentiment, which overlooks concrete concerns such as content clarity, feedback timeliness, and instructor demeanor, and provides limited guidance for instructional this http URL propose TeachPro, a multi-label learning framework that systematically assesses five key teaching dimensions: professional expertise, instructional behavior, pedagogical efficacy, classroom experience, and other performance metrics. We first propose a Dimension-Anchored Evidence Encoder, which integrates three core components: (i) a pre-trained text encoder that transforms qualitative feedback annotations into contextualized embeddings; (ii) a prompt module that represents five teaching dimensions as learnable semantic anchors; and (iii) a cross-attention mechanism that aligns evidence with pedagogical dimensions within a structured semantic space. We then propose a Cross-View Graph Synergy Network to represent student comments. This network comprises two components: (i) a Syntactic Branch that extracts explicit grammatical dependencies from parse trees, and (ii) a Semantic Branch that models latent conceptual relations derived from BERT-based similarity graphs. BiAffine fusion module aligns syntactic and semantic units, while a differential regularizer disentangles embeddings to encourage complementary representations. Finally, a cross-attention mechanism bridges the dimension-anchored evidence with the multi-view comment representations. We also contribute a novel benchmark dataset featuring expert qualitative annotations and multi-label scores. Extensive experiments demonstrate that TeachPro offers superior diagnostic granularity and robustness across diverse evaluation settings. 

---
# When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation 

**Authors**: Jing Ren, Bowen Li, Ziqi Xu, Xinkun Zhang, Haytham Fayek, Xiaodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09241)  

**Abstract**: Knowledge Graph Retrieval-Augmented Generation (KG-RAG) extends the RAG paradigm by incorporating structured knowledge from knowledge graphs, enabling Large Language Models (LLMs) to perform more precise and explainable reasoning. While KG-RAG improves factual accuracy in complex tasks, existing KG-RAG models are often severely overconfident, producing high-confidence predictions even when retrieved sub-graphs are incomplete or unreliable, which raises concerns for deployment in high-stakes domains. To address this issue, we propose Ca2KG, a Causality-aware Calibration framework for KG-RAG. Ca2KG integrates counterfactual prompting, which exposes retrieval-dependent uncertainties in knowledge quality and reasoning reliability, with a panel-based re-scoring mechanism that stabilises predictions across interventions. Extensive experiments on two complex QA datasets demonstrate that Ca2KG consistently improves calibration while maintaining or even enhancing predictive accuracy. 

---
# UserLM-R1: Modeling Human Reasoning in User Language Models with Multi-Reward Reinforcement Learning 

**Authors**: Feng Zhang, Shijia Li, Chunmao Zhang, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09215)  

**Abstract**: User simulators serve as the critical interactive environment for agent post-training, and an ideal user simulator generalizes across domains and proactively engages in negotiation by challenging or bargaining. However, current methods exhibit two issues. They rely on static and context-unaware profiles, necessitating extensive manual redesign for new scenarios, thus limiting generalizability. Moreover, they neglect human strategic thinking, leading to vulnerability to agent manipulation. To address these issues, we propose UserLM-R1, a novel user language model with reasoning capability. Specifically, we first construct comprehensive user profiles with both static roles and dynamic scenario-specific goals for adaptation to diverse scenarios. Then, we propose a goal-driven decision-making policy to generate high-quality rationales before producing responses, and further refine the reasoning and improve strategic capabilities with supervised fine-tuning and multi-reward reinforcement learning. Extensive experimental results demonstrate that UserLM-R1 outperforms competitive baselines, particularly on the more challenging adversarial set. 

---
# A.X K1 Technical Report 

**Authors**: Sung Jun Cheon, Jaekyung Cho, Seongho Choi, Hyunjun Eun, Seokhwan Jo, Jaehyun Jun, Minsoo Kang, Jin Kim, Jiwon Kim, Minsang Kim, Sungwan Kim, Seungsik Kim, Tae Yoon Kim, Youngrang Kim, Hyeongmun Lee, Sangyeol Lee, Sungeun Lee, Youngsoon Lee, Yujin Lee, Seongmin Ok, Chanyong Park, Hyewoong Park, Junyoung Park, Hyunho Yang, Subin Yi, Soohyun Bae, Dhammiko Arya, Yongseok Choi, Sangho Choi, Dongyeon Cho, Seungmo Cho, Gyoungeun Han, Yong-jin Han, Seokyoung Hong, Hyeon Hwang, Wonbeom Jang, Minjeong Ju, Wonjin Jung, Keummin Ka, Sungil Kang, Dongnam Kim, Joonghoon Kim, Jonghwi Kim, SaeRom Kim, Sangjin Kim, Seongwon Kim, Youngjin Kim, Seojin Lee, Sunwoo Lee, Taehoon Lee, Chanwoo Park, Sohee Park, Sooyeon Park, Yohan Ra, Sereimony Sek, Seungyeon Seo, Gun Song, Sanghoon Woo, Janghan Yoon, Sungbin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2601.09200)  

**Abstract**: We introduce A.X K1, a 519B-parameter Mixture-of-Experts (MoE) language model trained from scratch. Our design leverages scaling laws to optimize training configurations and vocabulary size under fixed computational budgets. A.X K1 is pre-trained on a corpus of approximately 10T tokens, curated by a multi-stage data processing pipeline. Designed to bridge the gap between reasoning capability and inference efficiency, A.X K1 supports explicitly controllable reasoning to facilitate scalable deployment across diverse real-world scenarios. We propose a simple yet effective Think-Fusion training recipe, enabling user-controlled switching between thinking and non-thinking modes within a single unified model. Extensive evaluations demonstrate that A.X K1 achieves performance competitive with leading open-source models, while establishing a distinctive advantage in Korean-language benchmarks. 

---
# ProFit: Leveraging High-Value Signals in SFT via Probability-Guided Token Selection 

**Authors**: Tao Liu, Taiqiang Wu, Runming Yang, Shaoning Sun, Junjie Wang, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09195)  

**Abstract**: Supervised fine-tuning (SFT) is a fundamental post-training strategy to align Large Language Models (LLMs) with human intent. However, traditional SFT often ignores the one-to-many nature of language by forcing alignment with a single reference answer, leading to the model overfitting to non-core expressions. Although our empirical analysis suggests that introducing multiple reference answers can mitigate this issue, the prohibitive data and computational costs necessitate a strategic shift: prioritizing the mitigation of single-reference overfitting over the costly pursuit of answer diversity. To achieve this, we reveal the intrinsic connection between token probability and semantic importance: high-probability tokens carry the core logical framework, while low-probability tokens are mostly replaceable expressions. Based on this insight, we propose ProFit, which selectively masks low-probability tokens to prevent surface-level overfitting. Extensive experiments confirm that ProFit consistently outperforms traditional SFT baselines on general reasoning and mathematical benchmarks. 

---
# OrthoGeoLoRA: Geometric Parameter-Efficient Fine-Tuning for Structured Social Science Concept Retrieval on theWeb 

**Authors**: Zeqiang Wang, Xinyue Wu, Chenxi Li, Zixi Chen, Nishanth Sastry, Jon Johnson, Suparna De  

**Link**: [PDF](https://arxiv.org/pdf/2601.09185)  

**Abstract**: Large language models and text encoders increasingly power web-based information systems in the social sciences, including digital libraries, data catalogues, and search interfaces used by researchers, policymakers, and civil society. Full fine-tuning is often computationally and energy intensive, which can be prohibitive for smaller institutions and non-profit organizations in the Web4Good ecosystem. Parameter-Efficient Fine-Tuning (PEFT), especially Low-Rank Adaptation (LoRA), reduces this cost by updating only a small number of parameters. We show that the standard LoRA update $\Delta W = BA^\top$ has geometric drawbacks: gauge freedom, scale ambiguity, and a tendency toward rank collapse. We introduce OrthoGeoLoRA, which enforces an SVD-like form $\Delta W = B\Sigma A^\top$ by constraining the low-rank factors to be orthogonal (Stiefel manifold). A geometric reparameterization implements this constraint while remaining compatible with standard optimizers such as Adam and existing fine-tuning pipelines. We also propose a benchmark for hierarchical concept retrieval over the European Language Social Science Thesaurus (ELSST), widely used to organize social science resources in digital repositories. Experiments with a multilingual sentence encoder show that OrthoGeoLoRA outperforms standard LoRA and several strong PEFT variants on ranking metrics under the same low-rank budget, offering a more compute- and parameter-efficient path to adapt foundation models in resource-constrained settings. 

---
# Identity-Robust Language Model Generation via Content Integrity Preservation 

**Authors**: Miao Zhang, Kelly Chen, Md Mehrab Tanjim, Rumi Chunara  

**Link**: [PDF](https://arxiv.org/pdf/2601.09141)  

**Abstract**: Large Language Model (LLM) outputs often vary across user sociodemographic attributes, leading to disparities in factual accuracy, utility, and safety, even for objective questions where demographic information is irrelevant. Unlike prior work on stereotypical or representational bias, this paper studies identity-dependent degradation of core response quality. We show empirically that such degradation arises from biased generation behavior, despite factual knowledge being robustly encoded across identities. Motivated by this mismatch, we propose a lightweight, training-free framework for identity-robust generation that selectively neutralizes non-critical identity information while preserving semantically essential attributes, thus maintaining output content integrity. Experiments across four benchmarks and 18 sociodemographic identities demonstrate an average 77% reduction in identity-dependent bias compared to vanilla prompting and a 45% reduction relative to prompt-based defenses. Our work addresses a critical gap in mitigating the impact of user identity cues in prompts on core generation quality. 

---
# Adaptive Multi-Stage Patent Claim Generation with Unified Quality Assessment 

**Authors**: Chen-Wei Liang, Bin Guo, Zhen-Yuan Wei, Mu-Jiang-Shan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09120)  

**Abstract**: Current patent claim generation systems face three fundamental limitations: poor cross-jurisdictional generalization, inadequate semantic relationship modeling between claims and prior art, and unreliable quality assessment. We introduce a novel three-stage framework that addresses these challenges through relationship-aware similarity analysis, domain-adaptive claim generation, and unified quality assessment. Our approach employs multi-head attention with eight specialized heads for explicit relationship modeling, integrates curriculum learning with dynamic LoRA adapter selection across five patent domains, and implements cross-attention mechanisms between evaluation aspects for comprehensive quality assessment. Extensive experiments on USPTO HUPD dataset, EPO patent collections, and Patent-CE benchmark demonstrate substantial improvements: 7.6-point ROUGE-L gain over GPT-4o, 8.3\% BERTScore enhancement over Llama-3.1-8B, and 0.847 correlation with human experts compared to 0.623 for separate evaluation models. Our method maintains 89.4\% cross-jurisdictional performance retention versus 76.2\% for baselines, establishing a comprehensive solution for automated patent prosecution workflows. 

---
# Contrastive Bi-Encoder Models for Multi-Label Skill Extraction: Enhancing ESCO Ontology Matching with BERT and Attention Mechanisms 

**Authors**: Yongming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09119)  

**Abstract**: Fine-grained labor market analysis increasingly relies on mapping unstructured job advertisements to standardized skill taxonomies such as ESCO. This mapping is naturally formulated as an Extreme Multi-Label Classification (XMLC) problem, but supervised solutions are constrained by the scarcity and cost of large-scale, taxonomy-aligned annotations--especially in non-English settings where job-ad language diverges substantially from formal skill definitions. We propose a zero-shot skill extraction framework that eliminates the need for manually labeled job-ad training data. The framework uses a Large Language Model (LLM) to synthesize training instances from ESCO definitions, and introduces hierarchically constrained multi-skill generation based on ESCO Level-2 categories to improve semantic coherence in multi-label contexts. On top of the synthetic corpus, we train a contrastive bi-encoder that aligns job-ad sentences with ESCO skill descriptions in a shared embedding space; the encoder augments a BERT backbone with BiLSTM and attention pooling to better model long, information-dense requirement statements. An upstream RoBERTa-based binary filter removes non-skill sentences to improve end-to-end precision. Experiments show that (i) hierarchy-conditioned generation improves both fluency and discriminability relative to unconstrained pairing, and (ii) the resulting multi-label model transfers effectively to real-world Chinese job advertisements, achieving strong zero-shot retrieval performance (F1@5 = 0.72) and outperforming TF--IDF and standard BERT baselines. Overall, the proposed pipeline provides a scalable, data-efficient pathway for automated skill coding in labor economics and workforce analytics. 

---
# SubTokenTest: A Practical Benchmark for Real-World Sub-token Understanding 

**Authors**: Shuyang Hou, Yi Hu, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09089)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly enhanced their reasoning capabilities. However, they continue to struggle with basic character-level tasks, such as counting letters in words, a problem rooted in their tokenization process. While existing benchmarks have highlighted this weakness through basic character operations, such failures are often dismissed due to lacking practical relevance. Yet, many real-world applications, such as navigating text-based maps or interpreting structured tables, rely heavily on precise sub-token understanding. In this regard, we introduce SubTokenTest, a comprehensive benchmark that assesses sub-token understanding through practical, utility-driven tasks. Our benchmark includes ten tasks across four domains and isolates tokenization-related failures by decoupling performance from complex reasoning. We provide a comprehensive evaluation of nine advanced LLMs. Additionally, we investigate the impact of test-time scaling on sub-token reasoning and explore how character-level information is encoded within the hidden states. 

---
# How Many Human Judgments Are Enough? Feasibility Limits of Human Preference Evaluation 

**Authors**: Wilson Y. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.09084)  

**Abstract**: Human preference evaluations are widely used to compare generative models, yet it remains unclear how many judgments are required to reliably detect small improvements. We show that when preference signal is diffuse across prompts (i.e., all prompt types are similarly informative), proportional allocation is minimax-optimal: no allocation strategy substantially improves detectability. Empirical analysis of large-scale human preference datasets shows that most comparisons fall into this diffuse regime, exhibiting small preference margins that require far more judgments than typically collected, even in well-sampled comparisons. These limits persist across evaluation protocols and modalities, including chat, image generation, and code generation with execution feedback. In contrast, curated benchmarks that reduce prompt induced variability systematically induce larger margins and improve detectability through a $1.5\times$ reduction in prompt-level variance. Our results show that inconclusive or negative human evaluation outcomes frequently reflect underpowered evaluation rather than model equivalence, underscoring the need to account explicitly for effect size, budget, and protocol design. 

---
# From Symbolic to Natural-Language Relations: Rethinking Knowledge Graph Construction in the Era of Large Language Models 

**Authors**: Kanyao Han, Yushang Lai  

**Link**: [PDF](https://arxiv.org/pdf/2601.09069)  

**Abstract**: Knowledge graphs (KGs) have commonly been constructed using predefined symbolic relation schemas, typically implemented as categorical relation labels. This design has notable shortcomings: real-world relations are often contextual, nuanced, and sometimes uncertain, and compressing it into discrete relation labels abstracts away critical semantic detail. Nevertheless, symbolic-relation KGs remain widely used because they have been operationally effective and broadly compatible with pre-LLM downstream models and algorithms, in which KG knowledge could be retrieved or encoded into quantified features and embeddings at scale. The emergence of LLMs has reshaped how knowledge is created and consumed. LLMs support scalable synthesis of domain facts directly in concise natural language, and prompting-based inference favors context-rich free-form text over quantified representations. This position paper argues that these changes call for rethinking the representation of relations themselves rather than merely using LLMs to populate conventional schemas more efficiently. We therefore advocate moving from symbolic to natural-language relation descriptions, and we propose hybrid design principles that preserve a minimal structural backbone while enabling more flexible and context-sensitive relational representations. 

---
# Mi:dm 2.0 Korea-centric Bilingual Language Models 

**Authors**: Donghoon Shin, Sejung Lee, Soonmin Bae, Hwijung Ryu, Changwon Ok, Hoyoun Jung, Hyesung Ji, Jeehyun Lim, Jehoon Lee, Ji-Eun Han, Jisoo Baik, Mihyeon Kim, Riwoo Chung, Seongmin Lee, Wonjae Park, Yoonseok Heo, Youngkyung Seo, Seyoun Won, Boeun Kim, Cheolhun Heo, Eunkyeong Lee, Honghee Lee, Hyeongju Ju, Hyeontae Seo, Jeongyong Shim, Jisoo Lee, Junseok Koh, Junwoo Kim, Minho Lee, Minji Kang, Minju Kim, Sangha Nam, Seongheum Park, Taehyeong Kim, Euijai Ahn, Hong Seok Jeung, Jisu Shin, Jiyeon Kim, Seonyeong Song, Seung Hyun Kong, Sukjin Hong, Taeyang Yun, Yu-Seon Kim, A-Hyun Lee, Chae-Jeong Lee, Hye-Won Yu, Ji-Hyun Ahn, Song-Yeon Kim, Sun-Woo Jung, Eunju Kim, Eunji Ha, Jinwoo Baek, Yun-ji Lee, Wanjin Park, Jeong Yeop Kim, Eun Mi Kim, Hyoung Jun Park, Jung Won Yoon, Min Sung Noh, Myung Gyo Oh, Wongyoung Lee, Yun Jin Park, Young S. Kwon, Hyun Keun Kim, Jieun Lee, YeoJoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.09066)  

**Abstract**: We introduce Mi:dm 2.0, a bilingual large language model (LLM) specifically engineered to advance Korea-centric AI. This model goes beyond Korean text processing by integrating the values, reasoning patterns, and commonsense knowledge inherent to Korean society, enabling nuanced understanding of cultural contexts, emotional subtleties, and real-world scenarios to generate reliable and culturally appropriate responses. To address limitations of existing LLMs, often caused by insufficient or low-quality Korean data and lack of cultural alignment, Mi:dm 2.0 emphasizes robust data quality through a comprehensive pipeline that includes proprietary data cleansing, high-quality synthetic data generation, strategic data mixing with curriculum learning, and a custom Korean-optimized tokenizer to improve efficiency and coverage. To realize this vision, we offer two complementary configurations: Mi:dm 2.0 Base (11.5B parameters), built with a depth-up scaling strategy for general-purpose use, and Mi:dm 2.0 Mini (2.3B parameters), optimized for resource-constrained environments and specialized tasks. Mi:dm 2.0 achieves state-of-the-art performance on Korean-specific benchmarks, with top-tier zero-shot results on KMMLU and strong internal evaluation results across language, humanities, and social science tasks. The Mi:dm 2.0 lineup is released under the MIT license to support extensive research and commercial use. By offering accessible and high-performance Korea-centric LLMs, KT aims to accelerate AI adoption across Korean industries, public services, and education, strengthen the Korean AI developer community, and lay the groundwork for the broader vision of K-intelligence. Our models are available at this https URL. For technical inquiries, please contact midm-llm@kt.com. 

---
# Beyond Consensus: Perspectivist Modeling and Evaluation of Annotator Disagreement in NLP 

**Authors**: Yinuo Xu, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2601.09065)  

**Abstract**: Annotator disagreement is widespread in NLP, particularly for subjective and ambiguous tasks such as toxicity detection and stance analysis. While early approaches treated disagreement as noise to be removed, recent work increasingly models it as a meaningful signal reflecting variation in interpretation and perspective. This survey provides a unified view of disagreement-aware NLP methods. We first present a domain-agnostic taxonomy of the sources of disagreement spanning data, task, and annotator factors. We then synthesize modeling approaches using a common framework defined by prediction targets and pooling structure, highlighting a shift from consensus learning toward explicitly modeling disagreement, and toward capturing structured relationships among annotators. We review evaluation metrics for both predictive performance and annotator behavior, and noting that most fairness evaluations remain descriptive rather than normative. We conclude by identifying open challenges and future directions, including integrating multiple sources of variation, developing disagreement-aware interpretability frameworks, and grappling with the practical tradeoffs of perspectivist modeling. 

---
# Efficient Multilingual Dialogue Processing via Translation Pipelines and Distilled Language Models 

**Authors**: Santiago Martínez Novoa, Nicolás Rozo Fajardo, Diego Alejandro González Vargas, Nicolás Bedoya Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2601.09059)  

**Abstract**: This paper presents team Kl33n3x's multilingual dialogue summarization and question answering system developed for the NLPAI4Health 2025 shared task. The approach employs a three-stage pipeline: forward translation from Indic languages to English, multitask text generation using a 2.55B parameter distilled language model, and reverse translation back to source languages. By leveraging knowledge distillation techniques, this work demonstrates that compact models can achieve highly competitive performance across nine languages. The system achieved strong win rates across the competition's tasks, with particularly robust performance on Marathi (86.7% QnA), Tamil (86.7% QnA), and Hindi (80.0% QnA), demonstrating the effectiveness of translation-based approaches for low-resource language processing without task-specific fine-tuning. 

---
# SITA: Learning Speaker-Invariant and Tone-Aware Speech Representations for Low-Resource Tonal Languages 

**Authors**: Tianyi Xu, Xuan Ouyang, Binwei Yao, Shoua Xiong, Sara Misurelli, Maichou Lor, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09050)  

**Abstract**: Tonal low-resource languages are widely spoken yet remain underserved by modern speech technology. A key challenge is learning representations that are robust to nuisance variation such as gender while remaining tone-aware for different lexical meanings. To address this, we propose SITA, a lightweight adaptation recipe that enforces Speaker-Invariance and Tone-Awareness for pretrained wav2vec-style encoders. SITA uses staged multi-objective training: (i) a cross-gender contrastive objective encourages lexical consistency across speakers, while a tone-repulsive loss prevents tone collapse by explicitly separating same-word different-tone realizations; and (ii) an auxiliary Connectionist Temporal Classification (CTC)-based ASR objective with distillation stabilizes recognition-relevant structure. We evaluate primarily on Hmong, a highly tonal and severely under-resourced language where off-the-shelf multilingual encoders fail to represent tone effectively. On a curated Hmong word corpus, SITA improves cross-gender lexical retrieval accuracy, while maintaining usable ASR accuracy relative to an ASR-adapted XLS-R teacher. We further observe similar gains when transferring the same recipe to Mandarin, suggesting SITA is a general, plug-in approach for adapting multilingual speech encoders to tonal languages. 

---
# Is Grokking Worthwhile? Functional Analysis and Transferability of Generalization Circuits in Transformers 

**Authors**: Kaiyu He, Zhang Mian, Peilin Wu, Xinya Du, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09049)  

**Abstract**: While Large Language Models (LLMs) excel at factual retrieval, they often struggle with the "curse of two-hop reasoning" in compositional tasks. Recent research suggests that parameter-sharing transformers can bridge this gap by forming a "Generalization Circuit" during a prolonged "grokking" phase. A fundamental question arises: Is a grokked model superior to its non-grokked counterparts on downstream tasks? Furthermore, is the extensive computational cost of waiting for the grokking phase worthwhile? In this work, we conduct a mechanistic study to evaluate the Generalization Circuit's role in knowledge assimilation and transfer. We demonstrate that: (i) The inference paths established by non-grokked and grokked models for in-distribution compositional queries are identical. This suggests that the "Generalization Circuit" does not represent the sudden acquisition of a new reasoning paradigm. Instead, we argue that grokking is the process of integrating memorized atomic facts into an naturally established reasoning path. (ii) Achieving high accuracy on unseen cases after prolonged training and the formation of a certain reasoning path are not bound; they can occur independently under specific data regimes. (iii) Even a mature circuit exhibits limited transferability when integrating new knowledge, suggesting that "grokked" Transformers do not achieve a full mastery of compositional logic. 

---
# Can LLMs interpret figurative language as humans do?: surface-level vs representational similarity 

**Authors**: Samhita Bollepally, Aurora Sloman-Moll, Takashi Yamauchi  

**Link**: [PDF](https://arxiv.org/pdf/2601.09041)  

**Abstract**: Large language models generate judgments that resemble those of humans. Yet the extent to which these models align with human judgments in interpreting figurative and socially grounded language remains uncertain. To investigate this, human participants and four instruction-tuned LLMs of different sizes (GPT-4, Gemma-2-9B, Llama-3.2, and Mistral-7B) rated 240 dialogue-based sentences representing six linguistic traits: conventionality, sarcasm, funny, emotional, idiomacy, and slang. Each of the 240 sentences was paired with 40 interpretive questions, and both humans and LLMs rated these sentences on a 10-point Likert scale. Results indicated that humans and LLMs aligned at the surface level with humans, but diverged significantly at the representational level, especially in interpreting figurative sentences involving idioms and Gen Z slang. GPT-4 most closely approximates human representational patterns, while all models struggle with context-dependent and socio-pragmatic expressions like sarcasm, slang, and idiomacy. 

---
# SpectraQuery: A Hybrid Retrieval-Augmented Conversational Assistant for Battery Science 

**Authors**: Sreya Vangara, Jagjit Nanda, Yan-Kai Tzeng, Eric Darve  

**Link**: [PDF](https://arxiv.org/pdf/2601.09036)  

**Abstract**: Scientific reasoning increasingly requires linking structured experimental data with the unstructured literature that explains it, yet most large language model (LLM) assistants cannot reason jointly across these modalities. We introduce SpectraQuery, a hybrid natural-language query framework that integrates a relational Raman spectroscopy database with a vector-indexed scientific literature corpus using a Structured and Unstructured Query Language (SUQL)-inspired design. By combining semantic parsing with retrieval-augmented generation, SpectraQuery translates open-ended questions into coordinated SQL and literature retrieval operations, producing cited answers that unify numerical evidence with mechanistic explanation. Across SQL correctness, answer groundedness, retrieval effectiveness, and expert evaluation, SpectraQuery demonstrates strong performance: approximately 80 percent of generated SQL queries are fully correct, synthesized answers reach 93-97 percent groundedness with 10-15 retrieved passages, and battery scientists rate responses highly across accuracy, relevance, grounding, and clarity (4.1-4.6/5). These results show that hybrid retrieval architectures can meaningfully support scientific workflows by bridging data and discourse for high-volume experimental datasets. 

---
# OpenDecoder: Open Large Language Model Decoding to Incorporate Document Quality in RAG 

**Authors**: Fengran Mo, Zhan Su, Yuchen Hui, Jinghan Zhang, Jia Ao Sun, Zheyuan Liu, Chao Zhang, Tetsuya Sakai, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2601.09028)  

**Abstract**: The development of large language models (LLMs) has achieved superior performance in a range of downstream tasks, including LLM-based retrieval-augmented generation (RAG). The quality of generated content heavily relies on the usefulness of the retrieved information and the capacity of LLMs' internal information processing mechanism to incorporate it in answer generation. It is generally assumed that the retrieved information is relevant to the question. However, the retrieved information may have a variable degree of relevance and usefulness, depending on the question and the document collection. It is important to take into account the relevance of the retrieved information in answer generation. In this paper, we propose OpenDecoder, a new approach that leverages explicit evaluation of the retrieved information as quality indicator features for generation. We aim to build a RAG model that is more robust to varying levels of noisy context. Three types of explicit evaluation information are considered: relevance score, ranking score, and QPP (query performance prediction) score. The experimental results on five benchmark datasets demonstrate the effectiveness and better robustness of OpenDecoder by outperforming various baseline methods. Importantly, this paradigm is flexible to be integrated with the post-training of LLMs for any purposes and incorporated with any type of external indicators. 

---
# Multicultural Spyfall: Assessing LLMs through Dynamic Multilingual Social Deduction Game 

**Authors**: Haryo Akbarianto Wibowo, Alaa Elsetohy, Qinrong Cui, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2601.09017)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has necessitated more robust evaluation methods that go beyond static benchmarks, which are increasingly prone to data saturation and leakage. In this paper, we propose a dynamic benchmarking framework for evaluating multilingual and multicultural capabilities through the social deduction game Spyfall. In our setup, models must engage in strategic dialogue to either identify a secret agent or avoid detection, utilizing culturally relevant locations or local foods. Our results show that our game-based rankings align closely with the Chatbot Arena. However, we find a significant performance gap in non-English contexts: models are generally less proficient when handling locally specific entities and often struggle with rule-following or strategic integrity in non-English languages. We demonstrate that this game-based approach provides a scalable, leakage-resistant, and culturally nuanced alternative to traditional NLP benchmarks. The game history can be accessed here this https URL. 

---
# TranslateGemma Technical Report 

**Authors**: Mara Finkelstein, Isaac Caswell, Tobias Domhan, Jan-Thorsten Peter, Juraj Juraska, Parker Riley, Daniel Deutsch, Cole Dilanni, Colin Cherry, Eleftheria Briakou, Elizabeth Nielsen, Jiaming Luo, Kat Black, Ryan Mullins, Sweta Agrawal, Wenda Xu, Erin Kats, Stephane Jaskiewicz, Markus Freitag, David Vilar  

**Link**: [PDF](https://arxiv.org/pdf/2601.09012)  

**Abstract**: We present TranslateGemma, a suite of open machine translation models based on the Gemma 3 foundation models. To enhance the inherent multilingual capabilities of Gemma 3 for the translation task, we employ a two-stage fine-tuning process. First, supervised fine-tuning is performed using a rich mixture of high-quality large-scale synthetic parallel data generated via state-of-the-art models and human-translated parallel data. This is followed by a reinforcement learning phase, where we optimize translation quality using an ensemble of reward models, including MetricX-QE and AutoMQM, targeting translation quality. We demonstrate the effectiveness of TranslateGemma with human evaluation on the WMT25 test set across 10 language pairs and with automatic evaluation on the WMT24++ benchmark across 55 language pairs. Automatic metrics show consistent and substantial gains over the baseline Gemma 3 models across all sizes. Notably, smaller TranslateGemma models often achieve performance comparable to larger baseline models, offering improved efficiency. We also show that TranslateGemma models retain strong multimodal capabilities, with enhanced performance on the Vistra image translation benchmark. The release of the open TranslateGemma models aims to provide the research community with powerful and adaptable tools for machine translation. 

---
# Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM 

**Authors**: Pedro Memoli Buffa, Luciano Del Corro  

**Link**: [PDF](https://arxiv.org/pdf/2601.09001)  

**Abstract**: Deploying LLMs raises two coupled challenges: (1) monitoring - estimating where a model underperforms as traffic and domains drift - and (2) improvement - prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-k logprobs) and summarize it with eleven statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions (k in {1,2,3,4}; all "10 choose k" combinations), across nine LLMs from six families (3B-20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains. Output-entropy profiles are thus an accessible signal for scalable monitoring and for targeting data acquisition. 

---
# Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models 

**Authors**: Youwei Liu, Jian Wang, Hanlin Wang, Beichen Guo, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.08955)  

**Abstract**: Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imaginable} Markov decision process to guide policy learning. We instantiate \texttt{ITP} with both training-free and reinforcement-trained variants. Extensive experiments across representative agent benchmarks demonstrate that \texttt{ITP} significantly outperforms competitive baselines. Further analyses validate that our adaptive lookahead largely enhances agents' reasoning capability, providing valuable insights into addressing broader, complex tasks. 

---
# Evaluating Role-Consistency in LLMs for Counselor Training 

**Authors**: Eric Rudolph, Natalie Engert, Jens Albrecht  

**Link**: [PDF](https://arxiv.org/pdf/2601.08892)  

**Abstract**: The rise of online counseling services has highlighted the need for effective training methods for future counselors. This paper extends research on VirCo, a Virtual Client for Online Counseling, designed to complement traditional role-playing methods in academic training by simulating realistic client interactions. Building on previous work, we introduce a new dataset incorporating adversarial attacks to test the ability of large language models (LLMs) to maintain their assigned roles (role-consistency). The study focuses on evaluating the role consistency and coherence of the Vicuna model's responses, comparing these findings with earlier research. Additionally, we assess and compare various open-source LLMs for their performance in sustaining role consistency during virtual client interactions. Our contributions include creating an adversarial dataset, evaluating conversation coherence and persona consistency, and providing a comparative analysis of different LLMs. 

---
# NewsScope: Schema-Grounded Cross-Domain News Claim Extraction with Open Models 

**Authors**: Nidhi Pandya  

**Link**: [PDF](https://arxiv.org/pdf/2601.08852)  

**Abstract**: Automated news verification requires structured claim extraction, but existing approaches either lack schema compliance or generalize poorly across domains. This paper presents NewsScope, a cross-domain dataset, benchmark, and fine-tuned model for schema-grounded news claim extraction. The dataset contains 455 articles across politics, health, science/environment, and business, consisting of 395 in-domain articles and 60 out-of-source articles for generalization testing. LLaMA 3.1 8B was fine-tuned using LoRA on 315 training examples and evaluated on held-out in-domain (80 articles) and out-of-source (60 articles) test sets. Human evaluation on 400 claims shows NewsScope achieves 89.4% human-evaluated accuracy compared to GPT-4o-mini's 93.7% (p=0.07). NewsScope outperforms GPT-4o-mini on political claims (94.3% vs. 87.8%). A numeric grounding filter further improves accuracy to 91.6%, narrowing the gap to 2.1 percentage points. Inter-annotator agreement studies (160 claims) confirm labeling reliability (94.6% positive agreement on SUPPORTED judgments). The open-weight model enables offline deployment at approximately $15 on-demand compute (or $0 on free tiers). Code and benchmark are publicly released. 

---
# Más contexto no es mejor. Paradoja de la dilución vectorial en RAG corporativos 

**Authors**: Alex Dantart  

**Link**: [PDF](https://arxiv.org/pdf/2601.08851)  

**Abstract**: Técnicas recientes de "Contextualized Chunking" inyectan resúmenes para mejorar el contexto en RAG, pero introducen una "dilución vectorial" que opaca el contenido local. Evaluando distintos ratios de inyección, demostramos una curva en "U invertida": una inyección moderada mejora el "Recall" (+18%), pero superar un umbral crítico (CIR > 0.4) reduce la precisión en un 22% para consultas específicas. Proponemos un marco teórico para calcular el ratio óptimo de inyección. --
Recent "Contextualized Chunking" techniques inject summaries to improve RAG context but introduce "vector dilution" drowning out local content. Evaluating various injection ratios, we demonstrate an "inverted U" curve: moderate injection boosts Recall (+18%), but exceeding a critical threshold (CIR > 0.4) drops precision by 22% for specific queries. We propose a theoretical framework to calculate the optimal injection ratio. 

---
# Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment 

**Authors**: Manas Khatore, Sumana Sridharan, Kevork Sulahian, Benjamin J. Smith, Shi Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08849)  

**Abstract**: Automated answer matching, which leverages LLMs to evaluate free-text responses by comparing them to a reference answer, shows substantial promise as a scalable and aligned alternative to human evaluation. However, its reliability requires robustness against strategic attacks such as guesswork or verbosity that may artificially inflate scores without improving actual correctness. In this work, we systematically investigate whether such tactics deceive answer matching models by prompting examinee models to: (1) generate verbose responses, (2) provide multiple answers when unconfident, and (3) embed conflicting answers with the correct answer near the start of their response. Our results show that these manipulations do not increase scores and often reduce them. Additionally, binary scoring (which requires a matcher to answer with a definitive "correct" or "incorrect") is more robust to attacks than continuous scoring (which requires a matcher to determine partial correctness). These findings show that answer matching is generally robust to inexpensive text manipulation and is a viable alternative to traditional LLM-as-a-judge or human evaluation when reference answers are available. 

---
# PediaMind-R1: A Temperament-Aware Language Model for Personalized Early Childhood Care Reasoning via Cognitive Modeling and Preference Alignment 

**Authors**: Zihe Zhang, Can Zhang, Yanheng Xu, Xin Hu, Jichao Leng  

**Link**: [PDF](https://arxiv.org/pdf/2601.08848)  

**Abstract**: This paper presents PediaMind-R1, a domain-specialized large language model designed to achieve active personalization in intelligent parenting scenarios. Unlike conventional systems that provide generic suggestions, PediaMind-R1 draws on insights from developmental psychology. It introduces temperament theory from the Thomas-Chess framework and builds a temperament knowledge graph for infants and toddlers (0-3 years). Our two-stage training pipeline first uses supervised fine-tuning to teach structured chain-of-thought reasoning, and then applies a GRPO-based alignment stage to reinforce logical consistency, domain expertise, and empathetic caregiving strategies. We further design an evaluation framework comprising temperament-sensitive multiple-choice tests and human assessments. The results demonstrate that PediaMind-R1 can accurately interpret early childhood temperament profiles and proactively engage in individualized reasoning. This work highlights the value of integrating vertical-domain modeling with psychological theory. It offers a novel approach to developing user-centered LLMs that advance the practice of active personalization in sensitive caregiving contexts. 

---
# Scalable and Reliable Evaluation of AI Knowledge Retrieval Systems: RIKER and the Coherent Simulated Universe 

**Authors**: JV Roig  

**Link**: [PDF](https://arxiv.org/pdf/2601.08847)  

**Abstract**: Evaluating knowledge systems (LLMs, RAG, knowledge graphs, etc) faces fundamental challenges: static benchmarks are vulnerable to contamination, LLM-based judges exhibit systematic biases, and ground truth extraction requires expensive human annotation. We present RIKER (Retrieval Intelligence and Knowledge Extraction Rating), both a benchmark and a replicable methodology based on paradigm inversion - generating documents from known ground truth rather than extracting ground truth from documents. This approach enables deterministic scoring and scalable evaluation without human annotation or reference models, and contamination resistance through regenerable corpora. Our evaluation of 33 models using over 21 billion tokens reveals that context length claims frequently exceed usable capacity, with significant degradation beyond 32K tokens; cross-document aggregation proves substantially harder than single-document extraction; and grounding ability and hallucination resistance are distinct capabilities - models excelling at finding facts that exist may still fabricate facts that do not. Beyond the specific benchmark, we contribute a domain-agnostic methodology for constructing scalable and contamination-resistant evaluations wherever synthetic documents can be generated from structured ground truth. 

---
# Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning 

**Authors**: Cagatay Tekin, Charbel Barakat, Luis Joseph Luna Limgenco  

**Link**: [PDF](https://arxiv.org/pdf/2601.08846)  

**Abstract**: Iterative summarization based reasoning frameworks such as InftyThink enable long-horizon reasoning in large language models (LLMs) by controlling context growth, but they repeatedly regenerate similar reasoning strategies across tasks. We introduce InftyThink with Cross-Chain Memory, an extension that augments iterative reasoning with an embedding-based semantic cache of previously successful reasoning patterns. At each reasoning step, the model retrieves and conditions on the most semantically similar stored lemmas, guiding inference without expanding the context window indiscriminately. Experiments on MATH500, AIME2024, and GPQA-Diamond demonstrate that semantic lemma retrieval improves accuracy in structured domains while exposing failure modes in tests that include heterogeneous domains. Geometric analyses of reasoning trajectories reveal that cache retrieval induces directional biases in embedding space, leading to consistent fix (improve baseline accuracy) and break (degradation in baseline accuracy) attractors. Our results highlight both the benefits and limits of similarity-based memory for self-improving LLM reasoning. 

---
# Emissions and Performance Trade-off Between Small and Large Language Models 

**Authors**: Anandita Garg, Uma Gaba, Deepan Muthirayan, Anish Roy Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.08844)  

**Abstract**: The advent of Large Language Models (LLMs) has raised concerns about their enormous carbon footprint, starting with energy-intensive training and continuing through repeated inference. This study investigates the potential of using fine-tuned Small Language Models (SLMs) as a sustainable alternative for predefined tasks. Here, we present a comparative analysis of the performance-emissions trade-off between LLMs and fine-tuned SLMs across selected tasks under Natural Language Processing, Reasoning and Programming. Our results show that in four out of the six selected tasks, SLMs maintained comparable performances for a significant reduction in carbon emissions during inference. Our findings demonstrate the viability of smaller models in mitigating the environmental impact of resource-heavy LLMs, thus advancing towards sustainable, green AI. 

---
# Rubric-Conditioned LLM Grading: Alignment, Uncertainty, and Robustness 

**Authors**: Haotian Deng, Chris Farber, Jiyoon Lee, David Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08843)  

**Abstract**: Automated short-answer grading (ASAG) remains a challenging task due to the linguistic variability of student responses and the need for nuanced, rubric-aligned partial credit. While Large Language Models (LLMs) offer a promising solution, their reliability as automated judges in rubric-based settings requires rigorous assessment. In this paper, we systematically evaluate the performance of LLM-judges for rubric-based short-answer grading. We investigate three key aspects: the alignment of LLM grading with expert judgment across varying rubric complexities, the trade-off between uncertainty and accuracy facilitated by a consensus-based deferral mechanism, and the model's robustness under random input perturbations and adversarial attacks. Using the SciEntsBank benchmark and Qwen 2.5-72B, we find that alignment is strong for binary tasks but degrades with increased rubric granularity. Our "Trust Curve" analysis demonstrates a clear trade-off where filtering low-confidence predictions improves accuracy on the remaining subset. Additionally, robustness experiments reveal that while the model is resilient to prompt injection, it is sensitive to synonym substitutions. Our work provides critical insights into the capabilities and limitations of rubric-conditioned LLM judges, highlighting the importance of uncertainty estimation and robustness testing for reliable deployment. 

---
# Resisting Correction: How RLHF Makes Language Models Ignore External Safety Signals in Natural Conversation 

**Authors**: Felipe Biava Cataneo  

**Link**: [PDF](https://arxiv.org/pdf/2601.08842)  

**Abstract**: Safety architectures for language models increasingly rely on external monitors to detect errors and inject corrective signals at inference time. For such systems to function in interactive settings, models must be able to incorporate externally provided confidence information into their verbal responses. In this work, we test whether instruction-tuned language models preserve this controllability across different interaction modes.
Using Llama-3.2-3B on GSM8K, we perform a causal intervention study in which explicit external confidence signals are injected and model compliance is measured under multiple prompt strategies. We find that base models exhibit near-perfect controllability (Spearman rho close to 1.0), while instruction-tuned models display a striking context dependence: they fully comply with external corrections under explicit command prompts (bias approximately 0 percent, rho = 0.93), yet systematically ignore the same signals in natural conversational queries (bias plus 40 percent, rho = 0.04).
This behavior is not a capability failure; the model can process the signal, but an emergent property of RLHF optimization that prioritizes conversational fluency over external calibration cues in natural dialogue. We further show that internal token-level confidence in small models is uninformative (r = 0.035), underscoring the necessity of external supervision. Our findings highlight a deployment-critical failure mode: the interaction style users expect is precisely where safety corrections are least effective. 

---
# Triples and Knowledge-Infused Embeddings for Clustering and Classification of Scientific Documents 

**Authors**: Mihael Arcan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08841)  

**Abstract**: The increasing volume and complexity of scientific literature demand robust methods for organizing and understanding research documents. In this study, we explore how structured knowledge, specifically, subject-predicate-object triples, can enhance the clustering and classification of scientific papers. We propose a modular pipeline that combines unsupervised clustering and supervised classification over multiple document representations: raw abstracts, extracted triples, and hybrid formats that integrate both. Using a filtered arXiv corpus, we extract relational triples from abstracts and construct four text representations, which we embed using four state-of-the-art transformer models: MiniLM, MPNet, SciBERT, and SPECTER. We evaluate the resulting embeddings with KMeans, GMM, and HDBSCAN for unsupervised clustering, and fine-tune classification models for arXiv subject prediction. Our results show that full abstract text yields the most coherent clusters, but that hybrid representations incorporating triples consistently improve classification performance, reaching up to 92.6% accuracy and 0.925 macro-F1. We also find that lightweight sentence encoders (MiniLM, MPNet) outperform domain-specific models (SciBERT, SPECTER) in clustering, while SciBERT excels in structured-input classification. These findings highlight the complementary benefits of combining unstructured text with structured knowledge, offering new insights into knowledge-infused representations for semantic organization of scientific documents. 

---
# Consistency-Aware Editing for Entity-level Unlearning in Language Models 

**Authors**: Xiaoqi Han, Víctor Gutiérrez-Basulto, Ru Li, Xiaoli Li, Jiye Liang, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2601.08840)  

**Abstract**: Large language models (LLMs) risk retaining sensitive, copyrighted, or harmful information from their training data. Entity-level unlearning addresses this issue by removing all knowledge of a specific entity while preserving the model's overall capabilities. Existing approaches typically rely on full-model fine-tuning or prompt-based interventions, which can be computationally expensive or brittle when handling paraphrased queries. Recently, model editing has emerged as an efficient alternative for updating knowledge in LLMs, offering a promising direction for unlearning. However, existing editing techniques are typically designed for instance-level updates, modifying responses to specific attributes of an entity rather than eliminating all knowledge associated with the entity. In this paper, we investigate how editing techniques can be adapted for effective and efficient entity-level unlearning. To this end, we introduce a novel consistency-aware editing (CAE) framework. CAE aggregates a diverse set of prompts related to a target entity, including its attributes, relations, and adversarial paraphrases. It then jointly learns a low-rank update guided by a consistency regularizer that aligns the editing directions across prompts. This promotes robust and comprehensive forgetting while minimizing interference with unrelated knowledge. We further examine where different entities are stored within the model and how many diverse prompts are needed for successful unlearning. We evaluate CAE on two challenging benchmarks, RWKU and ToFU, and demonstrate that it (i) provides insights into how entity-level knowledge is internally represented and deleted in LLMs, (ii) significantly improves forgetting accuracy and robustness over traditional unlearning and editing baselines, and (iii) enables scalable entity removal using only tens of carefully selected prompts. 

---
# Recursive Knowledge Synthesis for Multi-LLM Systems: Stability Analysis and Tri-Agent Audit Framework 

**Authors**: Toshiyuki Shigemura  

**Link**: [PDF](https://arxiv.org/pdf/2601.08839)  

**Abstract**: This paper presents a tri-agent cross-validation framework for analyzing stability and explainability in multi-model large language systems. The architecture integrates three heterogeneous LLMs-used for semantic generation, analytical consistency checking, and transparency auditing-into a recursive interaction cycle. This design induces Recursive Knowledge Synthesis (RKS), where intermediate representations are continuously refined through mutually constraining transformations irreducible to single-model behavior. Across 47 controlled trials using public-access LLM deployments (October 2025), we evaluated system stability via four metrics: Reflex Reliability Score (RRS), Transparency Score (TS), Deviation Detection Rate (DDR), and Correction Success Rate (CSR). The system achieved mean RRS = 0.78+-0.06 and maintained TS >= 0.8 in about 68% of trials. Approximately 89% of trials converged, supporting the theoretical prediction that transparency auditing acts as a contraction operator within the composite validation mapping. The contributions are threefold: (1) a structured tri-agent framework for coordinated reasoning across heterogeneous LLMs, (2) a formal RKS model grounded in fixed-point theory, and (3) empirical evaluation of inter-model stability under realistic, non-API public-access conditions. These results provide initial empirical evidence that a safety-preserving, humansupervised multi-LLM architecture can achieve stable recursive knowledge synthesis in realistic, publicly deployed environments. 

---
# Companion Agents: A Table-Information Mining Paradigm for Text-to-SQL 

**Authors**: Jiahui Chen, Lei Fu, Jian Cui, Yu Lei, Zhenning Dong  

**Link**: [PDF](https://arxiv.org/pdf/2601.08838)  

**Abstract**: Large-scale Text-to-SQL benchmarks such as BIRD typically assume complete and accurate database annotations as well as readily available external knowledge, which fails to reflect common industrial settings where annotations are missing, incomplete, or erroneous. This mismatch substantially limits the real-world applicability of state-of-the-art (SOTA) Text-to-SQL systems. To bridge this gap, we explore a database-centric approach that leverages intrinsic, fine-grained information residing in relational databases to construct missing evidence and improve Text-to-SQL accuracy under annotation-scarce conditions. Our key hypothesis is that when a query requires multi-step reasoning over extensive table information, existing methods often struggle to reliably identify and utilize the truly relevant knowledge. We therefore propose to "cache" query-relevant knowledge on the database side in advance, so that it can be selectively activated at inference time. Based on this idea, we introduce Companion Agents (CA), a new Text-to-SQL paradigm that incorporates a group of agents accompanying database schemas to proactively mine and consolidate hidden inter-table relations, value-domain distributions, statistical regularities, and latent semantic cues before query generation. Experiments on BIRD under the fully missing evidence setting show that CA recovers +4.49 / +4.37 / +14.13 execution accuracy points on RSL-SQL / CHESS / DAIL-SQL, respectively, with larger gains on the Challenging subset +9.65 / +7.58 / +16.71. These improvements stem from CA's automatic database-side mining and evidence construction, suggesting a practical path toward industrial-grade Text-to-SQL deployment without reliance on human-curated evidence. 

---
# From Adversarial Poetry to Adversarial Tales: An Interpretability Research Agenda 

**Authors**: Piercosma Bisconti, Marcello Galisai, Matteo Prandi, Federico Pierucci, Olga Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Marcantonio Brancale, Daniele Nardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.08837)  

**Abstract**: Safety mechanisms in LLMs remain vulnerable to attacks that reframe harmful requests through culturally coded structures. We introduce Adversarial Tales, a jailbreak technique that embeds harmful content within cyberpunk narratives and prompts models to perform functional analysis inspired by Vladimir Propp's morphology of folktales. By casting the task as structural decomposition, the attack induces models to reconstruct harmful procedures as legitimate narrative interpretation. Across 26 frontier models from nine providers, we observe an average attack success rate of 71.3%, with no model family proving reliably robust. Together with our prior work on Adversarial Poetry, these findings suggest that structurally-grounded jailbreaks constitute a broad vulnerability class rather than isolated techniques. The space of culturally coded frames that can mediate harmful intent is vast, likely inexhaustible by pattern-matching defenses alone. Understanding why these attacks succeed is therefore essential: we outline a mechanistic interpretability research agenda to investigate how narrative cues reshape model representations and whether models can learn to recognize harmful intent independently of surface form. 

---
# A Review: PTSD in Pre-Existing Medical Condition on Social Media 

**Authors**: Zaber Al Hassan Ayon, Nur Hafieza Ismail, Nur Shazwani Kamarudin  

**Link**: [PDF](https://arxiv.org/pdf/2601.08836)  

**Abstract**: Post-Traumatic Stress Disorder (PTSD) is a multifaceted mental health condition, particularly challenging for individuals with pre-existing medical conditions. This review critically examines the intersection of PTSD and chronic illnesses as expressed on social media platforms. By systematically analyzing literature from 2008 to 2024, the study explores how PTSD manifests and is managed in individuals with chronic conditions such as cancer, heart disease, and autoimmune disorders, with a focus on online expressions on platforms like X (formally known as Twitter) and Facebook. Findings demonstrate that social media data offers valuable insights into the unique challenges faced by individuals with both PTSD and chronic illnesses. Specifically, natural language processing (NLP) and machine learning (ML) techniques can identify potential PTSD cases among these populations, achieving accuracy rates between 74% and 90%. Furthermore, the role of online support communities in shaping coping strategies and facilitating early interventions is highlighted. This review underscores the necessity of incorporating considerations of pre-existing medical conditions in PTSD research and treatment, emphasizing social media's potential as a monitoring and support tool for vulnerable groups. Future research directions and clinical implications are also discussed, with an emphasis on developing targeted interventions. 

---
# DeliberationBench: When Do More Voices Hurt? A Controlled Study of Multi-LLM Deliberation Protocols 

**Authors**: Vaarunay Kaushal, Taranveer Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.08835)  

**Abstract**: Multi-agent systems where Large Language Models (LLMs) deliberate to form consensus have gained significant attention, yet their practical value over simpler methods remains under-scrutinized. We introduce DELIBERATIONBENCH, a controlled benchmark evaluating three deliberation protocols against a strong baseline of selecting the best response from a pool of model outputs. Across 270 questions and three independent seeds (810 total evaluations), we find a striking negative result: the best-single baseline achieves an 82.5% +- 3.3% win rate, dramatically outperforming the best deliberation protocol(13.8% +- 2.6%). This 6.0x performance gap is statistically significant (p < 0.01) and comes at 1.5-2.5x higher computational cost. Our findings challenge assumptions that complexity enhances quality in multi-LLM systems. 

---
# ShortCoder: Knowledge-Augmented Syntax Optimization for Token-Efficient Code Generation 

**Authors**: Sicong Liu, Yanxian Huang, Mingwei Liu, Jiachi Chen, Ensheng Shi, Yuchi Ma, Hongyu Zhang, Yin Zhang, Yanlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09703)  

**Abstract**: Code generation tasks aim to automate the conversion of user requirements into executable code, significantly reducing manual development efforts and enhancing software productivity. The emergence of large language models (LLMs) has significantly advanced code generation, though their efficiency is still impacted by certain inherent architectural constraints. Each token generation necessitates a complete inference pass, requiring persistent retention of contextual information in memory and escalating resource consumption. While existing research prioritizes inference-phase optimizations such as prompt compression and model quantization, the generation phase remains underexplored. To tackle these challenges, we propose a knowledge-infused framework named ShortCoder, which optimizes code generation efficiency while preserving semantic equivalence and readability. In particular, we introduce: (1) ten syntax-level simplification rules for Python, derived from AST-preserving transformations, achieving 18.1% token reduction without functional compromise; (2) a hybrid data synthesis pipeline integrating rule-based rewriting with LLM-guided refinement, producing ShorterCodeBench, a corpus of validated tuples of original code and simplified code with semantic consistency; (3) a fine-tuning strategy that injects conciseness awareness into the base LLMs. Extensive experimental results demonstrate that ShortCoder consistently outperforms state-of-the-art methods on HumanEval, achieving an improvement of 18.1%-37.8% in generation efficiency over previous methods while ensuring the performance of code generation. 

---
# Disentangling Task Conflicts in Multi-Task LoRA via Orthogonal Gradient Projection 

**Authors**: Ziyu Yang, Guibin Chen, Yuxin Yang, Aoxiong Zeng, Xiangquan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09684)  

**Abstract**: Multi-Task Learning (MTL) combined with Low-Rank Adaptation (LoRA) has emerged as a promising direction for parameter-efficient deployment of Large Language Models (LLMs). By sharing a single adapter across multiple tasks, one can significantly reduce storage overhead. However, this approach suffers from negative transfer, where conflicting gradient updates from distinct tasks degrade the performance of individual tasks compared to single-task fine-tuning. This problem is exacerbated in LoRA due to the low-rank constraint, which limits the optimization landscape's capacity to accommodate diverse task requirements. In this paper, we propose Ortho-LoRA, a gradient projection method specifically tailored for the bipartite structure of LoRA. Ortho-LoRA dynamically projects conflicting task gradients onto the orthogonal complement of each other within the intrinsic LoRA subspace. Extensive experiments on the GLUE benchmark demonstrate that Ortho-LoRA effectively mitigates task interference, outperforming standard joint training and recovering 95\% of the performance gap between multi-task and single-task baselines with negligible computational overhead. 

---
# Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning 

**Authors**: Zhiyuan Hu, Yunhai Hu, Juncheng Liu, Shuyue Stella Li, Yucheng Wang, Zhen Xu, See-Kiong Ng, Anh Tuan Luu, Xinxing Xu, Bryan Hooi, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.09667)  

**Abstract**: Multi-agent systems have evolved into practical LLM-driven collaborators for many applications, gaining robustness from diversity and cross-checking. However, multi-agent RL (MARL) training is resource-intensive and unstable: co-adapting teammates induce non-stationarity, and rewards are often sparse and high-variance. Therefore, we introduce \textbf{Multi-Agent Test-Time Reinforcement Learning (MATTRL)}, a framework that injects structured textual experience into multi-agent deliberation at inference time. MATTRL forms a multi-expert team of specialists for multi-turn discussions, retrieves and integrates test-time experiences, and reaches consensus for final decision-making. We also study credit assignment for constructing a turn-level experience pool, then reinjecting it into the dialogue. Across challenging benchmarks in medicine, math, and education, MATTRL improves accuracy by an average of 3.67\% over a multi-agent baseline, and by 8.67\% over comparable single-agent baselines. Ablation studies examine different credit-assignment schemes and provide a detailed comparison of how they affect training outcomes. MATTRL offers a stable, effective and efficient path to distribution-shift-robust multi-agent reasoning without tuning. 

---
# Toward Understanding Unlearning Difficulty: A Mechanistic Perspective and Circuit-Guided Difficulty Metric 

**Authors**: Jiali Cheng, Ziheng Chen, Chirag Agarwal, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2601.09624)  

**Abstract**: Machine unlearning is becoming essential for building trustworthy and compliant language models. Yet unlearning success varies considerably across individual samples: some are reliably erased, while others persist despite the same procedure. We argue that this disparity is not only a data-side phenomenon, but also reflects model-internal mechanisms that encode and protect memorized information. We study this problem from a mechanistic perspective based on model circuits--structured interaction pathways that govern how predictions are formed. We propose Circuit-guided Unlearning Difficulty (CUD), a {\em pre-unlearning} metric that assigns each sample a continuous difficulty score using circuit-level signals. Extensive experiments demonstrate that CUD reliably separates intrinsically easy and hard samples, and remains stable across unlearning methods. We identify key circuit-level patterns that reveal a mechanistic signature of difficulty: easy-to-unlearn samples are associated with shorter, shallower interactions concentrated in earlier-to-intermediate parts of the original model, whereas hard samples rely on longer and deeper pathways closer to late-stage computation. Compared to existing qualitative studies, CUD takes a first step toward a principled, fine-grained, and interpretable analysis of unlearning difficulty; and motivates the development of unlearning methods grounded in model mechanisms. 

---
# Linear Complexity Self-Supervised Learning for Music Understanding with Random Quantizer 

**Authors**: Petros Vavaroutsos, Theodoros Palamas, Pantelis Vikatos  

**Link**: [PDF](https://arxiv.org/pdf/2601.09603)  

**Abstract**: In recent years, foundation models have become very popular due to their exceptional performance, mainly in natural language (NLP) tasks where they were first introduced. These models usually consist of hundreds of millions, or even billions, of parameters, making them resource-intensive during training and in production systems, leading to increased costs. This paper focuses on the reduction of a foundation's model size when applied to music information retrieval (MIR) tasks. Our research combines the Branchformer architecture with SummaryMixing, which were first applied in speech recognition, along with a random quantization process. To facilitate reproducibility, we conduct pre-training on publicly available datasets, complemented by a proprietary dataset comparable in scale to other private datasets reported in the literature. We ensure robust evaluation by using a framework consisting of a variety of downstream MIR tasks. Our results show that our architecture achieves competitive performance when compared with other state-of-the-art models that use multi-head self-attention, while reducing the model size from 8.5% up to 12.3%. 

---
# Show, don't tell -- Providing Visual Error Feedback for Handwritten Documents 

**Authors**: Said Yasin, Torsten Zesch  

**Link**: [PDF](https://arxiv.org/pdf/2601.09586)  

**Abstract**: Handwriting remains an essential skill, particularly in education. Therefore, providing visual feedback on handwritten documents is an important but understudied area. We outline the many challenges when going from an image of handwritten input to correctly placed informative error feedback. We empirically compare modular and end-to-end systems and find that both approaches currently do not achieve acceptable overall quality. We identify the major challenges and outline an agenda for future research. 

---
# Permutation Matching Under Parikh Budgets: Linear-Time Detection, Packing, and Disjoint Selection 

**Authors**: MD Nazmul Alam Shanto, Md. Tanzeem Rahat, Md. Manzurul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2601.09577)  

**Abstract**: We study permutation (jumbled/Abelian) pattern matching over a general alphabet $\Sigma$. Given a pattern P of length m and a text T of length n, the classical task is to decide whether T contains a length-m substring whose Parikh vector equals that of P . While this existence problem admits a linear-time sliding-window solution, many practical applications require optimization and packing variants beyond mere detection. We present a unified sliding-window framework based on maintaining the Parikh-vector difference between P and the current window of T , enabling permutation matching in O(n + {\sigma}) time and O({\sigma}) space, where {\sigma} = |{\Sigma}|. Building on this foundation, we introduce a combinatorial-optimization variant that we call Maximum Feasible Substring under Pattern Supply (MFSP): find the longest substring S of T whose symbol counts are component-wise bounded by those of P . We show that MFSP can also be solved in O(n + {\sigma}) time via a two-pointer feasibility maintenance algorithm, providing an exact packing interpretation of P as a resource budget. Finally, we address non-overlapping occurrence selection by modeling each permutation match as an equal-length interval and proving that a greedy earliest-finishing strategy yields a maximum-cardinality set of disjoint matches, computable in linear time once all matches are enumerated. Our results provide concise, provably correct algorithms with tight bounds, and connect frequency-based string matching to packing-style optimization primitives. 

---
# Dissecting Judicial Reasoning in U.S. Copyright Damage Awards 

**Authors**: Pei-Chi Lo, Thomas Y. Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09459)  

**Abstract**: Judicial reasoning in copyright damage awards poses a core challenge for computational legal analysis. Although federal courts follow the 1976 Copyright Act, their interpretations and factor weightings vary widely across jurisdictions. This inconsistency creates unpredictability for litigants and obscures the empirical basis of legal decisions. This research introduces a novel discourse-based Large Language Model (LLM) methodology that integrates Rhetorical Structure Theory (RST) with an agentic workflow to extract and quantify previously opaque reasoning patterns from judicial opinions. Our framework addresses a major gap in empirical legal scholarship by parsing opinions into hierarchical discourse structures and using a three-stage pipeline, i.e., Dataset Construction, Discourse Analysis, and Agentic Feature Extraction. This pipeline identifies reasoning components and extract feature labels with corresponding discourse subtrees. In analyzing copyright damage rulings, we show that discourse-augmented LLM analysis outperforms traditional methods while uncovering unquantified variations in factor weighting across circuits. These findings offer both methodological advances in computational legal analysis and practical insights into judicial reasoning, with implications for legal practitioners seeking predictive tools, scholars studying legal principle application, and policymakers confronting inconsistencies in copyright law. 

---
# Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception 

**Authors**: Zhen Wan, Chao-Han Huck Yang, Jinchuan Tian, Hanrong Ye, Ankita Pasad, Szu-wei Fu, Arushi Goel, Ryo Hachiuma, Shizhe Diao, Kunal Dhawan, Sreyan Ghosh, Yusuke Hirota, Zhehuai Chen, Rafael Valle, Ehsan Hosseini Asl, Chenhui Chu, Shinji Watanabe, Yu-Chiang Frank Wang, Boris Ginsburg  

**Link**: [PDF](https://arxiv.org/pdf/2601.09413)  

**Abstract**: We introduce a voice-agentic framework that learns one critical omni-understanding skill: knowing when to trust itself versus when to consult external audio perception. Our work is motivated by a crucial yet counterintuitive finding: naively fine-tuning an omni-model on both speech recognition and external sound understanding tasks often degrades performance, as the model can be easily misled by noisy hypotheses. To address this, our framework, Speech-Hands, recasts the problem as an explicit self-reflection decision. This learnable reflection primitive proves effective in preventing the model from being derailed by flawed external candidates. We show that this agentic action mechanism generalizes naturally from speech recognition to complex, multiple-choice audio reasoning. Across the OpenASR leaderboard, Speech-Hands consistently outperforms strong baselines by 12.1% WER on seven benchmarks. The model also achieves 77.37% accuracy and high F1 on audio QA decisions, showing robust generalization and reliability across diverse audio question answering datasets. By unifying perception and decision-making, our work offers a practical path toward more reliable and resilient audio intelligence. 

---
# SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing 

**Authors**: Ziyang Ma, Guanrou Yang, Wenxi Chen, Zhifu Gao, Yexing Du, Xiquan Li, Zhisheng Zheng, Haina Zhu, Jianheng Zhuo, Zheshu Song, Ruiyang Xu, Tiranrui Wang, Yifan Yang, Yanqiao Zhu, Zhikang Niu, Liumeng Xue, Yinghao Ma, Ruibin Yuan, Shiliang Zhang, Kai Yu, Eng Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09385)  

**Abstract**: The recent surge in open-source Multimodal Large Language Models (MLLM) frameworks, such as LLaVA, provides a convenient kickoff for artificial intelligence developers and researchers. However, most of the MLLM frameworks take vision as the main input modality, and provide limited in-depth support for the modality of speech, audio, and music. This situation hinders the development of audio-language models, and forces researchers to spend a lot of effort on code writing and hyperparameter tuning. We present SLAM-LLM, an open-source deep learning framework designed to train customized MLLMs, focused on speech, language, audio, and music processing. SLAM-LLM provides a modular configuration of different encoders, projectors, LLMs, and parameter-efficient fine-tuning plugins. SLAM-LLM also includes detailed training and inference recipes for mainstream tasks, along with high-performance checkpoints like LLM-based Automatic Speech Recognition (ASR), Automated Audio Captioning (AAC), and Music Captioning (MC). Some of these recipes have already reached or are nearing state-of-the-art performance, and some relevant techniques have also been accepted by academic papers. We hope SLAM-LLM will accelerate iteration, development, data engineering, and model training for researchers. We are committed to continually pushing forward audio-based MLLMs through this open-source framework, and call on the community to contribute to the LLM-based speech, audio and music processing. 

---
# Long-term Task-oriented Agent: Proactive Long-term Intent Maintenance in Dynamic Environments 

**Authors**: Qinglong Shi, Donghai Wang, Hantao Zhou, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He  

**Link**: [PDF](https://arxiv.org/pdf/2601.09382)  

**Abstract**: Current large language model agents predominantly operate under a reactive paradigm, responding only to immediate user queries within short-term sessions. This limitation hinders their ability to maintain long-term user's intents and dynamically adapt to evolving external environments. In this paper, we propose a novel interaction paradigm for proactive Task-oriented Agents capable of bridging the gap between relatively static user's needs and a dynamic environment. We formalize proactivity through two key capabilities, (i) Intent-Conditioned Monitoring: The agent autonomously formulates trigger conditions based on dialog history; (ii) Event-Triggered Follow-up: The agent actively engages the user upon detecting useful environmental updates. We introduce a high-quality data synthesis pipeline to construct complex, multi-turn dialog data in a dynamic environment. Furthermore, we attempt to address the lack of evaluation criteria of task-oriented interaction in a dynamic environment by proposing a new benchmark, namely ChronosBench. We evaluated some leading close-source and open-source models at present and revealed their flaws in long-term task-oriented interaction. Furthermore, our fine-tuned model trained using synthetic data for supervised learning achieves a task completion rate of 85.19% for complex tasks including shifts in user intent, outperforming other models under test. And the result validated the effectiveness of our data-driven strategy. 

---
# GIFT: Unlocking Global Optimality in Post-Training via Finite-Temperature Gibbs Initialization 

**Authors**: Zhengyang Zhao, Lu Ma, Yizhen Jiang, Xiaochen Ma, Zimo Meng, Chengyu Shen, Lexiang Tang, Haoze Sun, Peng Pei, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09233)  

**Abstract**: The prevailing post-training paradigm for Large Reasoning Models (LRMs)--Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL)--suffers from an intrinsic optimization mismatch: the rigid supervision inherent in SFT induces distributional collapse, thereby exhausting the exploration space necessary for subsequent RL. In this paper, we reformulate SFT within a unified post-training framework and propose Gibbs Initialization with Finite Temperature (GIFT). We characterize standard SFT as a degenerate zero-temperature limit that suppresses base priors. Conversely, GIFT incorporates supervision as a finite-temperature energy potential, establishing a distributional bridge that ensures objective consistency throughout the post-training pipeline. Our experiments demonstrate that GIFT significantly outperforms standard SFT and other competitive baselines when utilized for RL initialization, providing a mathematically principled pathway toward achieving global optimality in post-training. Our code is available at this https URL. 

---
# Geometric Stability: The Missing Axis of Representations 

**Authors**: Prashant C. Raju  

**Link**: [PDF](https://arxiv.org/pdf/2601.09173)  

**Abstract**: Analysis of learned representations has a blind spot: it focuses on $similarity$, measuring how closely embeddings align with external references, but similarity reveals only what is represented, not whether that structure is robust. We introduce $geometric$ $stability$, a distinct dimension that quantifies how reliably representational geometry holds under perturbation, and present $Shesha$, a framework for measuring it. Across 2,463 configurations in seven domains, we show that stability and similarity are empirically uncorrelated ($\rho \approx 0.01$) and mechanistically distinct: similarity metrics collapse after removing the top principal components, while stability retains sensitivity to fine-grained manifold structure. This distinction yields actionable insights: for safety monitoring, stability acts as a functional geometric canary, detecting structural drift nearly 2$\times$ more sensitively than CKA while filtering out the non-functional noise that triggers false alarms in rigid distance metrics; for controllability, supervised stability predicts linear steerability ($\rho = 0.89$-$0.96$); for model selection, stability dissociates from transferability, revealing a geometric tax that transfer optimization incurs. Beyond machine learning, stability predicts CRISPR perturbation coherence and neural-behavioral coupling. By quantifying $how$ $reliably$ systems maintain structure, geometric stability provides a necessary complement to similarity for auditing representations across biological and computational systems. 

---
# EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge 

**Authors**: Shijian Ma, Yan Lin, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09142)  

**Abstract**: Detecting evasive answers in earnings calls is critical for financial transparency, yet progress is hindered by the lack of large-scale benchmarks. We introduce EvasionBench, comprising 30,000 training samples and 1,000 human-annotated test samples (Cohen's Kappa 0.835) across three evasion levels. Our key contribution is a multi-model annotation framework leveraging a core insight: disagreement between frontier LLMs signals hard examples most valuable for training. We mine boundary cases where two strong annotators conflict, using a judge to resolve labels. This approach outperforms single-model distillation by 2.4 percent, with judge-resolved samples improving generalization despite higher training loss (0.421 vs 0.393) - evidence that disagreement mining acts as implicit regularization. Our trained model Eva-4B (4B parameters) achieves 81.3 percent accuracy, outperforming its base by 25 percentage points and approaching frontier LLM performance at a fraction of inference cost. 

---
# AviationLMM: A Large Multimodal Foundation Model for Civil Aviation 

**Authors**: Wenbin Li, Jingling Wu, Xiaoyong Lin.Jing Chen, Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09105)  

**Abstract**: Civil aviation is a cornerstone of global transportation and commerce, and ensuring its safety, efficiency and customer satisfaction is paramount. Yet conventional Artificial Intelligence (AI) solutions in aviation remain siloed and narrow, focusing on isolated tasks or single modalities. They struggle to integrate heterogeneous data such as voice communications, radar tracks, sensor streams and textual reports, which limits situational awareness, adaptability, and real-time decision support. This paper introduces the vision of AviationLMM, a Large Multimodal foundation Model for civil aviation, designed to unify the heterogeneous data streams of civil aviation and enable understanding, reasoning, generation and agentic applications. We firstly identify the gaps between existing AI solutions and requirements. Secondly, we describe the model architecture that ingests multimodal inputs such as air-ground voice, surveillance, on-board telemetry, video and structured texts, and performs cross-modal alignment and fusion, and produces flexible outputs ranging from situation summaries and risk alerts to predictive diagnostics and multimodal incident reconstructions. In order to fully realize this vision, we identify key research opportunities to address, including data acquisition, alignment and fusion, pretraining, reasoning, trustworthiness, privacy, robustness to missing modalities, and synthetic scenario generation. By articulating the design and challenges of AviationLMM, we aim to boost the civil aviation foundation model progress and catalyze coordinated research efforts toward an integrated, trustworthy and privacy-preserving aviation AI ecosystem. 

---
# Distribution-Aligned Sequence Distillation for Superior Long-CoT Reasoning 

**Authors**: Shaotian Yan, Kaiyuan Liu, Chen Shen, Bing Wang, Sinan Fan, Jun Zhang, Yue Wu, Zheng Wang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.09088)  

**Abstract**: In this report, we introduce DASD-4B-Thinking, a lightweight yet highly capable, fully open-source reasoning model. It achieves SOTA performance among open-source models of comparable scale across challenging benchmarks in mathematics, scientific reasoning, and code generation -- even outperforming several larger models. We begin by critically reexamining a widely adopted distillation paradigm in the community: SFT on teacher-generated responses, also known as sequence-level distillation. Although a series of recent works following this scheme have demonstrated remarkable efficiency and strong empirical performance, they are primarily grounded in the SFT perspective. Consequently, these approaches focus predominantly on designing heuristic rules for SFT data filtering, while largely overlooking the core principle of distillation itself -- enabling the student model to learn the teacher's full output distribution so as to inherit its generalization capability. Specifically, we identify three critical limitations in current practice: i) Inadequate representation of the teacher's sequence-level distribution; ii) Misalignment between the teacher's output distribution and the student's learning capacity; and iii) Exposure bias arising from teacher-forced training versus autoregressive inference. In summary, these shortcomings reflect a systemic absence of explicit teacher-student interaction throughout the distillation process, leaving the essence of distillation underexploited. To address these issues, we propose several methodological innovations that collectively form an enhanced sequence-level distillation training pipeline. Remarkably, DASD-4B-Thinking obtains competitive results using only 448K training samples -- an order of magnitude fewer than those employed by most existing open-source efforts. To support community research, we publicly release our models and the training dataset. 

---
# MMR-GRPO: Accelerating GRPO-Style Training through Diversity-Aware Reward Reweighting 

**Authors**: Kangda Wei, Ruihong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09085)  

**Abstract**: Group Relative Policy Optimization (GRPO) has become a standard approach for training mathematical reasoning models; however, its reliance on multiple completions per prompt makes training computationally expensive. Although recent work has reduced the number of training steps required to reach peak performance, the overall wall-clock training time often remains unchanged or even increases due to higher per-step cost. We propose MMR-GRPO, which integrates Maximal Marginal Relevance to reweigh rewards based on completion diversity. Our key insight is that semantically redundant completions contribute limited marginal learning signal; prioritizing diverse solutions yields more informative updates and accelerates convergence. Extensive evaluations across three model sizes (1.5B, 7B, 8B), three GRPO variants, and five mathematical reasoning benchmarks show that MMR-GRPO achieves comparable peak performance while requiring on average 47.9% fewer training steps and 70.2% less wall-clock time. These gains are consistent across models, methods, and benchmarks. We will release our code, trained models, and experimental protocols. 

---
# Human-AI Co-design for Clinical Prediction Models 

**Authors**: Jean Feng, Avni Kothari, Patrick Vossler, Andrew Bishara, Lucas Zier, Newton Addo, Aaron Kornblith, Yan Shuo Tan, Chandan Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.09072)  

**Abstract**: Developing safe, effective, and practically useful clinical prediction models (CPMs) traditionally requires iterative collaboration between clinical experts, data scientists, and informaticists. This process refines the often small but critical details of the model building process, such as which features/patients to include and how clinical categories should be defined. However, this traditional collaboration process is extremely time- and resource-intensive, resulting in only a small fraction of CPMs reaching clinical practice. This challenge intensifies when teams attempt to incorporate unstructured clinical notes, which can contain an enormous number of concepts. To address this challenge, we introduce HACHI, an iterative human-in-the-loop framework that uses AI agents to accelerate the development of fully interpretable CPMs by enabling the exploration of concepts in clinical notes. HACHI alternates between (i) an AI agent rapidly exploring and evaluating candidate concepts in clinical notes and (ii) clinical and domain experts providing feedback to improve the CPM learning process. HACHI defines concepts as simple yes-no questions that are used in linear models, allowing the clinical AI team to transparently review, refine, and validate the CPM learned in each round. In two real-world prediction tasks (acute kidney injury and traumatic brain injury), HACHI outperforms existing approaches, surfaces new clinically relevant concepts not included in commonly-used CPMs, and improves model generalizability across clinical sites and time periods. Furthermore, HACHI reveals the critical role of the clinical AI team, such as directing the AI agent to explore concepts that it had not previously considered, adjusting the granularity of concepts it considers, changing the objective function to better align with the clinical objectives, and identifying issues of data bias and leakage. 

---
# StegoStylo: Squelching Stylometric Scrutiny through Steganographic Stitching 

**Authors**: Robert Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2601.09056)  

**Abstract**: Stylometry--the identification of an author through analysis of a text's style (i.e., authorship attribution)--serves many constructive purposes: it supports copyright and plagiarism investigations, aids detection of harmful content, offers exploratory cues for certain medical conditions (e.g., early signs of dementia or depression), provides historical context for literary works, and helps uncover misinformation and disinformation. In contrast, when stylometry is employed as a tool for authorship verification--confirming whether a text truly originates from a claimed author--it can also be weaponized for malicious purposes. Techniques such as de-anonymization, re-identification, tracking, profiling, and downstream effects like censorship illustrate the privacy threats that stylometric analysis can enable. Building on these concerns, this paper further explores how adversarial stylometry combined with steganography can counteract stylometric analysis. We first present enhancements to our adversarial attack, $\textit{TraceTarnish}$, providing stronger evidence of its capacity to confound stylometric systems and reduce their attribution and verification accuracy. Next, we examine how steganographic embedding can be fine-tuned to mask an author's stylistic fingerprint, quantifying the level of authorship obfuscation achievable as a function of the proportion of words altered with zero-width Unicode characters. Based on our findings, steganographic coverage of 33% or higher seemingly ensures authorship obfuscation. Finally, we reflect on the ways stylometry can be used to undermine privacy and argue for the necessity of defensive tools like $\textit{TraceTarnish}$. 

---
# PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm 

**Authors**: Jing-Jing Li, Joel Mire, Eve Fleisig, Valentina Pyatkin, Anne Collins, Maarten Sap, Sydney Levine  

**Link**: [PDF](https://arxiv.org/pdf/2601.08951)  

**Abstract**: Current AI safety frameworks, which often treat harmfulness as binary, lack the flexibility to handle borderline cases where humans meaningfully disagree. To build more pluralistic systems, it is essential to move beyond consensus and instead understand where and why disagreements arise. We introduce PluriHarms, a benchmark designed to systematically study human harm judgments across two key dimensions -- the harm axis (benign to harmful) and the agreement axis (agreement to disagreement). Our scalable framework generates prompts that capture diverse AI harms and human values while targeting cases with high disagreement rates, validated by human data. The benchmark includes 150 prompts with 15,000 ratings from 100 human annotators, enriched with demographic and psychological traits and prompt-level features of harmful actions, effects, and values. Our analyses show that prompts that relate to imminent risks and tangible harms amplify perceived harmfulness, while annotator traits (e.g., toxicity experience, education) and their interactions with prompt content explain systematic disagreement. We benchmark AI safety models and alignment methods on PluriHarms, finding that while personalization significantly improves prediction of human harm judgments, considerable room remains for future progress. By explicitly targeting value diversity and disagreement, our work provides a principled benchmark for moving beyond "one-size-fits-all" safety toward pluralistically safe AI. 

---
# Fine Grained Evaluation of LLMs-as-Judges 

**Authors**: Sourav Saha, Mandar Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2601.08919)  

**Abstract**: A good deal of recent research has focused on how Large Language Models
(LLMs) may be used as `judges' in place of humans to evaluate the quality
of the output produced by various text / image processing systems. Within
this broader context, a number of studies have investigated the specific
question of how effectively LLMs can be used as relevance assessors for
the standard ad hoc task in Information Retrieval (IR). We extend these
studies by looking at additional questions. Most importantly, we use a
Wikipedia based test collection created by the INEX initiative, and
prompt LLMs to not only judge whether documents are relevant /
non-relevant, but to highlight relevant passages in documents that it
regards as useful. The human relevance assessors involved in creating
this collection were given analogous instructions, i.e., they were asked
to highlight all passages within a document that respond to the
information need expressed in a query. This enables us to evaluate the
quality of LLMs as judges not only at the document level, but to also
quantify how often these `judges' are right for the right reasons.
Our findings suggest that LLMs-as-judges work best under human
supervision. 

---
# Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas 

**Authors**: Yuexi Shen, Minqian Liu, Dawei Zhou, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.08901)  

**Abstract**: Scientific discovery is a cumulative process and requires new ideas to be situated within an ever-expanding landscape of existing knowledge. An emerging and critical challenge is how to identify conceptually relevant prior work from rapidly growing literature, and assess how a new idea differentiates from existing research. Current embedding approaches typically conflate distinct conceptual aspects into single representations and cannot support fine-grained literature retrieval; meanwhile, LLM-based evaluators are subject to sycophancy biases, failing to provide discriminative novelty assessment. To tackle these challenges, we introduce the Ideation Space, a structured representation that decomposes scientific knowledge into three distinct dimensions, i.e., research problem, methodology, and core findings, each learned through contrastive training. This framework enables principled measurement of conceptual distance between ideas, and modeling of ideation transitions that capture the logical connections within a proposed idea. Building upon this representation, we propose a Hierarchical Sub-Space Retrieval framework for efficient, targeted literature retrieval, and a Decomposed Novelty Assessment algorithm that identifies which aspects of an idea are novel. Extensive experiments demonstrate substantial improvements, where our approach achieves Recall@30 of 0.329 (16.7% over baselines), our ideation transition retrieval reaches Hit Rate@30 of 0.643, and novelty assessment attains 0.37 correlation with expert judgments. In summary, our work provides a promising paradigm for future research on accelerating and evaluating scientific discovery. 

---
# Spectral Generative Flow Models: A Physics-Inspired Replacement for Vectorized Large Language Models 

**Authors**: Andrew Kiruluta  

**Link**: [PDF](https://arxiv.org/pdf/2601.08893)  

**Abstract**: We introduce Spectral Generative Flow Models (SGFMs), a physics-inspired alternative to transformer-based large language models. Instead of representing text or video as sequences of discrete tokens processed by attention, SGFMs treat generation as the evolution of a continuous field governed by constrained stochastic dynamics in a multiscale wavelet basis. This formulation replaces global attention with local operators, spectral projections, and Navier--Stokes-like transport, yielding a generative mechanism grounded in continuity, geometry, and physical structure.
Our framework provides three key innovations: (i) a field-theoretic ontology in which text and video are unified as trajectories of a stochastic partial differential equation; (ii) a wavelet-domain representation that induces sparsity, scale separation, and computational efficiency; and (iii) a constrained stochastic flow that enforces stability, coherence, and uncertainty propagation. Together, these components define a generative architecture that departs fundamentally from autoregressive modeling and diffusion-based approaches. SGFMs offer a principled path toward long-range coherence, multimodal generality, and physically structured inductive bias in next-generation generative models. 

---
# Reading or Reasoning? Format Decoupled Reinforcement Learning for Document OCR 

**Authors**: Yufeng Zhong, Lei Chen, Zhixiong Zeng, Xuanle Zhao, Deyang Jiang, Liming Zheng, Jing Huang, Haibo Qiu, Peng Shi, Siqi Yang, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2601.08834)  

**Abstract**: Reading text from images or scanned documents via OCR models has been a longstanding focus of researchers. Intuitively, text reading is perceived as a straightforward perceptual task, and existing work primarily focuses on constructing enriched data engineering to enhance SFT capabilities. In this work, we observe that even advanced OCR models exhibit significantly higher entropy in formatted text (\emph{e.g.}, formula, table, etc.) compared to plain text, often by an order of magnitude. These statistical patterns reveal that advanced OCR models struggle with high output uncertainty when dealing with format sensitive document, suggesting that reasoning over diverse reading pathways may improve OCR performance. To address this, we propose format decoupled reinforcement learning (FD-RL), which leverages high-entropy patterns for targeted optimization. Our approach employs entropy-based data filtration strategy to identify format-intensive instances, and adopt format decoupled rewards tailored to different format types, enabling format-level validation rather than token-level memorization. FD-RL achieves an average score of 90.41 on OmniDocBench, setting a new record for end-to-end models on this highly popular benchmark. More importantly, we conduct comprehensive ablation studies over data, training, filtering, and rewarding strategies, thoroughly validating their effectiveness. 

---
