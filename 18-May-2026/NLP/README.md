# A Generative AI Framework for Intelligent Utility Billing CO 2 Analytics and Sustainable Resource Optimisation 

**Authors**: Pavan Manjunath, Thomas Pruefer  

**Link**: [PDF](https://arxiv.org/pdf/2605.16250)  

**Abstract**: Distribution utilities are now expected to deliver bills that customers can actually read attach a defensible carbon number to every kWh sold and schedule load against grid stress and emissions constraints We propose an end-to-end framework that unifies four production-grade capabilities under one architectural roof a generative-AI agent that drafts each customers natural-language billing statement from structured numeric inputs under a constrained decoding policy a transformer-based forecaster that supplies the day-ahead consumption estimate with calibrated quantile bands 

---
# A Unified Generative-AI Framework for Smart Energy Infrastructure: Intelligent Gas Distribution, Utility Billing, Carbon Analytics, and Quantum-Inspired Optimisation 

**Authors**: Pavan Manjunath, Thomas pruefer  

**Link**: [PDF](https://arxiv.org/pdf/2605.16232)  

**Abstract**: The accelerating convergence of smart metering, generative artificial intelligence, and quantum-inspired combinatorial optimisation is reshaping how energy utilities manage physical infrastructure, customer engagement, and environmental accountability 

---
# Artificial Aphasias in Lesioned Language Models 

**Authors**: Nathan Roll, Jill Kries, Laura Gwilliams, Cory Shain  

**Link**: [PDF](https://arxiv.org/pdf/2605.16222)  

**Abstract**: Aphasias, selective language impairments which can arise from brain damage, reveal the functional organization of human language by providing causal links between affected brain regions and specific symptom profiles. Drawing on this literature, we introduce an aphasia-inspired technique to characterize the emergent functional organization of language models (LMs). We ``lesion'' (zero-out) model parameters and measure the effects of this intervention against clinical aphasia symptoms, as diagnosed by the Text Aphasia Battery (TAB). When applied to 112,426 outputs from five 1B-scale LMs, the full range of evaluated symptoms surface, but in distributions largely distinct from those of humans. Our method uncovers broad symptom-profile differences between attention components (query, key, value, output) and feed-forward components (up, gate, down), with weaker evidence for differences among components within the same mechanism. We also find an effect of depth, where lesions in early layers disproportionately cause syntactic and semantic symptoms while late-middle layers yield higher rates of phonological and fluency deficits. Although some LM lesions induce quantitatively more similar profiles to some human aphasia types than others, qualitative differences in symptom patterns between LMs and humans suggest that aphasia syndromes are heavily influenced by the details of learning and processing rather than being a domain-invariant consequence of disrupted language processing. 

---
# Argus: Evidence Assembly for Scalable Deep Research Agents 

**Authors**: Zhen Zhang, Liangcai Su, Zhuo Chen, Xiang Lin, Haotian Xu, Simon Shaolei Du, Kaiyu Yang, Bo An, Lidong Bing, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2605.16217)  

**Abstract**: Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens. 

---
# Improving Cross-Cultural Survey Simulation with Calibrated Value Personas 

**Authors**: Axel Abels, Elias Fernandez Domingos, Apurva Shah, Tom Lenaerts  

**Link**: [PDF](https://arxiv.org/pdf/2605.16193)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate human opinions and survey responses, but their ability to reproduce population responses across cultures remains limited. Existing persona-based prompting methods typically rely on sociodemographic or personality traits, which are only indirect proxies for the values that shape human responses. We propose a value-based persona construction method that derives textual descriptors from survey responses capturing core cultural dimensions. By sampling value profiles from target populations and aggregating LLM responses across personas, we obtain population-level predictions grounded in observed value distributions. We further introduce a calibration procedure that improves response diversity while preserving estimated opinions. We show that our approach reduces prediction error across countries, with the largest improvements observed in underrepresented populations. This substantially narrows the performance gap between countries aligned with dominant LLM priors and those that are less represented in training data, while also yielding response distributions that closely match human diversity. 

---
# Optimized Three-Dimensional Photovoltaic Structures with LLM guided Tree Search 

**Authors**: Michael P. Brenner, Lizzie Dorfman, John C. Platt  

**Link**: [PDF](https://arxiv.org/pdf/2605.16191)  

**Abstract**: We present a case study for how AI coding systems can be used to generate novel scientific hypotheses. We combine a generic coding agent (Google's AntiGravity) with an LLM-driven tree search algorithm (Empirical Research Assistance / ERA) to autonomously generate high-efficiency three-dimensional photovoltaic (3DPV) structures that overcome losses limiting flat solar panels at mid-latitudes. These structures operate by presenting favorable angles to the sun throughout the day, and for illustrative purposes we focus on optimizing performance for a single solar day. Our workflow begins by using AntiGravity to reproduce calculations \cite{bernardi2012solar} showing that 3DPV can have energy densities much higher than stationary flat PV panels. We use these initial designs as the starting point for large scale tree search, where we seek improved solutions and score them for their diurnal yield. The initial tree search leads to nominally more efficient solutions, yet they are caused by algorithmic reward hacking, arising from non-physical design features such as structurally levitating disconnected tiers and exploitations of the discretizations in the optics solver. To counteract this, we develop a workflow where the coding agent iteratively patches the physics engine with constraints to eliminate reward hacking. With reward-hacking eliminated, ERA discovers a series of designs with various constraints and improved performance, including optimal designs with different fixed collector areas, optimizing zenith tracking and avoiding self shadowing.
Combining coding agents with tree search (ERA) provides a powerful platform for scientific discovery, for problems whose solutions can be empirically evaluated with a score function. 

---
# SGR: A Stepwise Reasoning Framework for LLMs with External Subgraph Generation 

**Authors**: Xin Zhang, Yang Cao, Baoxing Wu, Kai Song, Siying Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.16117)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities across diverse NLP applications, such as translation, text generation, and question answering. Nevertheless, they remain limited in complex settings that demand deep reasoning and logical inference. Since these models are trained on large-scale text corpora, their generation process may still introduce irrelevant, noisy, or factually inconsistent content. To mitigate this problem, we introduce SGR, a stepwise framework that enhances LLM reasoning through external subgraph generation. SGR builds query-specific subgraphs from external knowledge bases and uses their semantic structure to support multi-step inference. By grounding intermediate reasoning steps in structured external knowledge, the framework helps the model concentrate on relevant entities, relations, and supporting evidence. In particular, SGR first constructs a subgraph tailored to the input question. It then guides the model to reason progressively over the generated structure and combines multiple reasoning trajectories to obtain the final prediction. Experimental results across several benchmark datasets show that SGR achieves consistent improvements over competitive baselines, highlighting its value for improving both reasoning accuracy and factual reliability. 

---
# DebiasRAG: A Tuning-Free Path to Fair Generation in Large Language Models through Retrieval-Augmented Generation 

**Authors**: Rui Chu, Bingyin Zhao, Thanh Quoc Hung Le, Duy Cao Hoang, Huawei Lin, Ping Li, Weijie Zhao, Khoa D Doan, Yingjie Lao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16113)  

**Abstract**: Large language models (LLMs) have achieved unprecedented success due to their exceptional generative capabilities. However, because they depend on knowledge encapsulated from training corpora, they may produce hallucinations, stereotypes, and socially biased content. In particular, LLMs are prone to prejudiced responses involving race, gender, and age, which are collectively referred to as social biases. Prior studies have used fine-tuning and prompt engineering to mitigate such biases in LLMs, but these methods require additional training resources or domain knowledge to design the framework. Moreover, they may degrade the original capabilities of LLMs and often overlook the need for dynamic debiasing contexts for fairer inference. In this paper, we propose DebiasRAG, a novel tuning-free and dynamic query-specific debiasing framework based on retrieval-augmented generation (RAG). DebiasRAG improves fairness while preserving the intrinsic properties of LLMs, such as representation ability. DebiasRAG consists of three stages: (1) query-specific debiasing candidate generation; (2) context candidate pool construction; and (3) gradient-updated debiasing-guided context piece reranking. First, DebiasRAG leverages self-diagnosed bias contexts relevant to the query through regular retrieval, where the bias contexts are prepared offline by the DebiasRAG provider. Given the query-specific bias contexts, DebiasRAG reversely produces debiasing contexts, which are provided as additional fairness constraints for LLM outputs. Second, a regular RAG retrieval process produces query-related contexts from the regular RAG document database, such as a chunked Wikipedia dataset. 

---
# Multi-Level Contextual Token Relation Modeling for Machine-Generated Text Detection 

**Authors**: Chenwang Wu, Yiuming Cheung, Bo Han, Shuhai Zhang, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2605.16107)  

**Abstract**: Machine-generated texts (MGTs) pose risks such as disinformation and phishing, underscoring the need for reliable detection. Metric-based methods, which extract statistically distinguishable features of MGTs, are often more practical than complex model-based methods that are prone to overfitting. Given their diverse designs, we first place representative metric-based methods within a unified framework, enabling a clear assessment of their advantages and limitations. Our analysis identifies a core challenge across these methods: the token-level detection score is easily biased by the inherent randomness of the MGTs generation process. Then, we theoretically derive the multi-hop transitions of the token-level detection score and explore their local and global relations. Based on these findings, we propose a multi-level contextual token relation modeling framework for MGT detection. Specifically, for local relations, we model them through a lightweight Markov-informed calibration module that refines token-level evidence before aggregation. For global relations, we introduce a rule-support reasoning module that uses explicit logical rules derived from contextual score statistics. Finally, we combine the local calibrated score and the global rule-support reasoning signal in a joint multi-level inference framework. Extensive experiments show broad and substantial improvements across various real-world scenarios, including cross-LLM and cross-domain settings, with low computational overhead. 

---
# Can Large Language Models Imitate Human Speech for Clinical Assessment? LLM-Driven Data Augmentation for Cognitive Score Prediction 

**Authors**: Si-Belkacem Yamine Ketir, Lenard Paulo Tamayo, Shohei Hisada, Shaowen Peng, Shoko Wakamiya, Eiji Aramaki  

**Link**: [PDF](https://arxiv.org/pdf/2605.16077)  

**Abstract**: Accurate assessment of cognitive decline from spontaneous speech remains challenging due to limited dataset size and class imbalance. In this work, we propose a large language model (LLM)-driven data augmentation framework to improve the prediction of cognitive scores from speech. Experiments are conducted on a Japanese corpus in which each participant provides both a spontaneous oral narrative and a written response to the same clinical prompt. The written responses serve as semantic anchors to generate multiple oral-like monologues in different styles using GPT-5. We then predict Hasegawa Dementia Scale scores, a widely used cognitive screening tool in Japan, using a Partial Least Squares regression model trained on Sentence-BERT speech embeddings. We investigate two augmentation strategies: random class-balanced selection, which yields moderate but unstable improvements, and similarity-guided class-balanced selection. The latter prioritizes semantically close synthetic samples, leading to more consistent improvements and substantially reducing prediction error for minority low-score participants while maintaining performance for the majority group. Overall, our findings demonstrate the potential of semantically guided LLM-driven augmentation as a principled approach for addressing class imbalance and improving data efficiency in clinical speech analysis. 

---
# RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents 

**Authors**: Zijie Dai, Shiyuan Deng, Sheng Guan, Yizhou Tian, Xin Yao, Xiao Yan, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16045)  

**Abstract**: Memory systems often organize user-agent interactions as retrievable external memory and are crucial for long-running agents by overcoming the limited context windows of LLMs. However, existing memory systems invoke LLMs to process every incoming interaction for memory extraction, and such an eager memory consolidation scheme leads to substantial token consumption. To tackle this problem, we propose RecMem by rethinking when memory consolidation should be conducted. RecMem stores incoming interactions in a subconscious memory layer and encode them using lightweight embedding models for retrieval. LLMs are only invoked to extract episodic and semantic memory when sustained recurrence are observed for semantically similar interactions. Such recurrence-based consolidation works because these interactions correspond to a semantic cluster with rich information and thus are worth extraction and summarization. To improve accuracy, RecMem also incorporates a semantic refinement mechanism that recovers the fine-grained facts omitted by memory extraction. Experiments show that RecMem reduces the memory construction token cost of three SOTA memory systems by up to 87% while exceeding their accuracy. 

---
# From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation 

**Authors**: Yu Pan, Yang Hou, Xiongfei Wu, Liang Zhang, Yves Le Traon, Lei Ma, Jianjun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2605.16026)  

**Abstract**: Compositional speech-to-speech translation (S2ST) systems built upon speech large language models (SpeechLLMs) have recently shown promising performance. However, existing S2ST systems often either neglect source-language information or encode it through a language-as-label paradigm, representing each source language as an independent flat embedding. Such a design overlooks systematic linguistic structure shared across languages, which may limit data-efficient multilingual adaptation when supervised S2ST data are scarce. To address this issue, we propose S2ST-Omni 2, a many-to-one compositional S2ST framework that systematically reformulates multilingual language conditioning from flat language labels to structured typological priors. Specifically, S2ST-Omni 2 revisits language conditioning at three levels: typology-informed hierarchical language encoding for structured source-language representation, dynamically-gated language-aware Dual-CTC for content-adaptive acoustic modulation, and typology-aware LLM prompting for decoder-side linguistic guidance. Experiments on CVSS-C show that S2ST-Omni 2 achieves superior average performance among representative S2ST approaches across BLEU, COMET, ASR-BLEU, and BLASER 2.0 under the adopted evaluation protocol. Ablation studies indicate that the proposed representation-level, acoustic-level, and decoding-level strategies provide complementary benefits. Moreover, controlled data-budget analyses and a Japanese-to-English evaluation using only approximately 3 hours of supervised training data suggest that explicit typological priors provide useful inductive biases for data-efficient multilingual S2ST. 

---
# Judge Circuits 

**Authors**: Nils Feldhus, Tanja Baeumel, Elena Golimblevskaia, Qianli Wang, Van Bach Nguyen, Aaron Louis Eidt, Christopher Ebert, Wojciech Samek, Jing Yang, Vera Schmitt, Sebastian Möller, Simon Ostermann  

**Link**: [PDF](https://arxiv.org/pdf/2605.16023)  

**Abstract**: LLM-as-a-judge has become the dominant paradigm for grading model outputs at scale, yet the same model assigns systematically different scores when its output format changes (e.g., a 1-5 rating vs. a True/False label). Existing diagnoses of these format-induced inconsistencies stop at the input-output level. Using Position-aware Edge Attribution Patching (PEAP), we causally investigate the internal mechanism in Gemma-3, Qwen2.5, and Llama-3. We find that judgments across structured understanding and open-ended preference tasks share a sparse, generalized Latent Evaluator sub-graph in the mid-to-late multi-layer perceptrons (MLPs); zero-ablating it collapses judgment while preserving world knowledge in architecturally modular models. By structurally decoupling abstract judging from output formatting, we provide a mechanistic account of format-induced inconsistency on the open-weight models we study: a continuous judgment signal computed in the shared trunk is mapped through fragile, format-specific terminal branches, enabling format-independent preference to be isolated downstream of the requested output format. Our findings imply that benchmark-level reliability comparisons across formats are partially measuring formatter geometry rather than evaluation quality. 

---
# Can Vision Language Models Be Adaptive in Mathematics Education? A Learner Model-based Rubric Study 

**Authors**: Jie Gao, Yongan Yu, Junzhu Su, Yiran Lin, Adam K. Dube, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2605.16011)  

**Abstract**: Adaptive learning refers to educational technologies that track learners' learning progress and adapt the instructional process based on individual learners' learning performance. It is increasingly recognized as critical for developing an effective learning support tool. Vision language models (VLMs) have seen adoption in mathematics education, and students have been using them as learning aids for personalized instruction. However, it is unknown whether VLMs have the ability to adapt to different learner profiles when providing mathematical instructions. Current VLMs lack a systematic evaluation framework for this adaptivity to different learner profiles in mathematics tutoring tasks. To address this gap, we draw on the learner model from the adaptive learning framework (Shute and Towle, 2018) and propose a learner model-based rubric. Our rubric formalizes adaptivity assessment into three aspects: cognitive aspects, motivational aspects, and complexity. We also evaluate two additional dimensions of VLM responses: correctness (of answers and solutions) and quality (of the response itself). Our experimental results show measurable differences in adaptivity across models and also reveal that current VLMs struggle to consistently produce learner model-based instructional responses, especially when receiving limited learner information. 

---
# Defining Cultural Capabilities for AI Evaluation: A Taxonomy Grounded in Intercultural Communication Theory 

**Authors**: Isar Nejadgholi, Masoud Kianpour, Krishnapriya Vishnubhotla, Maryam Molamohamadi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15990)  

**Abstract**: Tremendous efforts have been put into evaluating the inclusivity and effectiveness of AI systems across cultures. However, the cultural capabilities considered in much of the literature remain vaguely defined, are referred to using interchangeable terminology, and are typically limited to recalling accurate information about various demographics, regions, and nationalities. To address this construct ambiguity, we draw from Intercultural Communication scholarship and propose a three-level taxonomy of AI-relevant cultural capabilities: Cultural Awareness answers "Does the model know?", Cultural Sensitivity answers "How does it frame its knowledge?", and Cultural Competence answers "Can it adapt as the interaction evolves?". Beyond conceptual clarification, we position this taxonomy as a practical tool for improving the validity and interpretability of AI evaluation in real-world, multicultural settings. Without such construct clarity, evaluation results risk overstating model capabilities and may lead to inappropriate deployment decisions in culturally sensitive contexts. 

---
# Ontology for Policing: Conceptual Knowledge Learning for Semantic Understanding and Reasoning in Law Enforcement Reports 

**Authors**: Anita Srbinovska, Jansen Orfan, Adrian Martin, Ernest Fokoué  

**Link**: [PDF](https://arxiv.org/pdf/2605.15978)  

**Abstract**: Law enforcement reports contain structured fields and written narratives. However, many incident facts that are needed for review, police training, and investigations are in natural language and require manual reading. We propose a framework using symbolic methods for converting narratives into evidence-linked facts. Our objective is to measure the value of narratives to recover incident details only from the unstructured text and build temporal graphs with time cues and domain axioms. We achieve this by redacting personal identifiers, semantic parsing, predicate mapping to ontology, and reasoning. We evaluate the symbolic approach on 450 property crime reports and a short human review. Of the extracted events from the system, 54.1% had a confidence score of at least 0.80 and 93.7% were mapped through the PropBank--VerbNet--WordNet semantic path. 100% agreement was reached on incident initiation, stolen items, and temporal cues and lower agreement for forced entry interpretation. 

---
# Reference-Free Reinforcement Learning Fine-Tuning for MT: A Seq2Seq Perspective 

**Authors**: Ernesto Garcia-Estrada, Carlos Escolano, José A. R. Fonallosa  

**Link**: [PDF](https://arxiv.org/pdf/2605.15976)  

**Abstract**: Production machine translation relies overwhelmingly on encoder-decoder Seq2Seq models, yet reinforcement learning approaches to MT fine-tuning have largely targeted decoder-only LLMs at $\geq$7B parameters, with limited systematic study of encoder-decoder architectures. We apply Group Relative Policy Optimization to NLLB-200 (600M and 1.3B) using a hybrid reference-free reward (LaBSE and COMET-Kiwi) that requires no parallel data at fine-tuning time, evaluating across 13 typologically diverse languages. GRPO yields consistent improvements on all 13 languages, up to $+$5.03 chrF++ for Traditional Chinese, and, without any target-language data, competes with 3-epoch supervised fine-tuning on morphologically complex languages . We identify a consistent empirical pattern in which gains are largest where baseline performance is weakest and reward discriminability is highest, making this approach most effective precisely where parallel data is scarcest, and replicate this pattern across English and Spanish source languages. 

---
# Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation 

**Authors**: Shuaiyi Li, Zhisong Zhang, Yan Wang, Lei Zhu, Dongyang Ma, Chenlong Deng, Yang Deng, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2605.15913)  

**Abstract**: Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention. 

---
# Linked Multi-Model Data on Russian Domestic and Foreign Policy Speeches 

**Authors**: Daria Blinova, Gayathri Emuru, Rakesh Emuru, Kushagradheer Shridheer Srivastava, Mina Rulis, Sunita Chandrasekaran, Benjamin E. Bagozzi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15886)  

**Abstract**: This paper introduces a dataset of interlinked multimodal political communications from the Russian government, addressing persistent deficiencies in the availability of social text- and image-based data for authoritarian politics contexts. The dataset comprises two large corpora of official speeches delivered by senior actors within the Kremlin and the Russian Ministry of Foreign Affairs over multiple decades. For each speech, we provide Russian- and English-language texts, associated images and captions where available, and harmonized metadata including (e.g.) dates, speakers, (geo)locations, and official government content tags. Unique identifiers link images to speeches and align Russian and English versions of the same communication texts. We further augment these linked datasets with validated topical annotations for both speech texts and speech images, which are generated via transformer-based multimodal topic modeling and refined by a Russian politics expert. The resulting data resources support multimodal, multilingual, temporal, and/or spatial analyses of (authoritarian) political communication and offer a valuable testbed for social science research and large language model (LLM) applications in political domains. 

---
# ForMaT: Dataset for Visually-Grounded Multilingual PDF Translation 

**Authors**: Michał Ciesiółka, Dawid Wiśniewski, Adrian Charkiewicz, Kamil Guttmann  

**Link**: [PDF](https://arxiv.org/pdf/2605.15794)  

**Abstract**: We present ForMaT (Format-Preserving Multilingual Translation), a parallel corpus of 3,956 PDFs across 15 language pairs that preserves original layout metadata proposed for multimodal machine translation. To ensure structural diversity in the dataset, we employ K-Medoids sampling over 45 geometric features, capturing complex elements like nested tables and formulas to focus only on visually diverse PDF documents. Our evaluation reveals that current MT systems struggle with spatial grounding and geometric synchronization, often losing the link between text and its visual context. ForMaT provides a benchmark for developing layout-aware translation models that integrate visual and textual context for high-fidelity document reconstruction. 

---
# CompactQE: Interpretable Translation Quality Estimation via Small Open-Weight LLMs 

**Authors**: Kamil Guttmann, Zofia Fraś, Artur Nowakowski, Krzysztof Jassem  

**Link**: [PDF](https://arxiv.org/pdf/2605.15763)  

**Abstract**: Current state-of-the-art Quality Estimation (QE) in machine translation relies on massive, proprietary LLMs, raising data privacy concerns. We demonstrate that smaller, open-source LLMs (<30B parameters) are a viable, cost-effective and privacy-preserving alternative. Using a single-pass prompting strategy, our models simultaneously generate quality scores, MQM error annotations, suggested error corrections, and full post-editions. Our analysis shows these models achieve highly competitive system-level correlations with human judgments that outperform traditional neural metrics, fine-tuned models, and human inter-annotator agreement, effectively approximating the capabilities of much larger proprietary LLMs. 

---
# DimMem: Dimensional Structuring for Efficient Long-Term Agent Memory 

**Authors**: Wentao Qiu, Haotian Hu, Fanyi Wang, Jinwei Kong, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15759)  

**Abstract**: Large language model (LLM) agents require long-term memory to leverage information from past interactions. However, existing memory systems often face a fidelity--efficiency trade-off: raw dialogue histories are expensive, while flat facts or summaries may discard the structure needed for precise recall. We propose \textbf{DimMem}, a lightweight dimensional memory framework that represents each memory as an atomic, typed, and self-contained unit with explicit fields such as time, location, reason, purpose, and keywords. This representation exposes the structure needed for dimension-aware retrieval, memory update, and selective assistant-context recall without storing full histories in the model context. Across LoCoMo-10 and LongMemEval-S, DimMem achieves \textbf{81.43\%} and \textbf{78.20\%} overall accuracy, respectively, outperforming existing lightweight memory systems while reducing LoCoMo per-query token cost by \textbf{24\%}. We further show that dimensional memory extraction is learnable by compact models: after fine-tuning on the DimMem schema, a Qwen3-4B extractor surpasses LightMem with GPT-4.1-mini on both benchmarks and reaches performance comparable to, or better than, much larger extractors in key settings. These results suggest that explicit dimensional structuring is an effective and efficient foundation for long-term memory in LLM agents. Code is available at this https URL. 

---
# Contexting as Recommendation: Evolutionary Collaborative Filtering for Context Engineering 

**Authors**: Jiachen Zhu, Zhuoying Ou, Congmin Zheng, Yuxiang Chen, Zeyu Zheng, Rong Shan, Lingyu Yang, Lionel Z. Wang, Weiwen Liu, Yong Yu, Weinan Zhang, Jianghao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2605.15721)  

**Abstract**: Large Language Models (LLMs) are highly sensitive to their input contexts, motivating the development of automated context engineering. However, existing methods predominantly treat this as a global search problem, seeking a single context strategy that maximizes average performance across a dataset. This restrictive assumption overlooks the fact that different inputs often require distinct guidance, leaving substantial instance-level performance gains untapped. In this paper, we propose a paradigm shift by formulating context engineering as a recommendation problem. We introduce \textbf{Neural Collaborative Context Engineering (NCCE)}, a framework that transitions optimization from a static global search to dynamic, instance-wise routing. NCCE first bootstraps a diverse catalog of anchor contexts and then employs a novel \textbf{Context-CF Co-Evolution} mechanism. This stage establishes a synergistic feedback loop: a lightweight Neural Collaborative Filtering (NCF) model learns instance-context preferences to guide the generation of specialized context variants, while the newly evaluated contexts continuously refine the NCF model's understanding of latent preferences. At inference time, the trained NCF model acts as a context router, dynamically assigning the most suitable context strategy to each unseen instance. Theoretical Proofs and comprehensive experiments demonstrate that by matching individual inputs with their optimal contexts, NCCE significantly improves task accuracy, highlighting the critical importance of personalization in LLM context engineering. 

---
# SMMBench: A Benchmark for Source-Distributed Multimodal Agent Memory 

**Authors**: Huacan Chai, Yukai Wang, Yingxuan Yang, Dan Peng, Yuanyi Song, Zhihui Fu, Weiwen Liu, Jianghao Lin, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15710)  

**Abstract**: Existing benchmarks for multimodal memory reasoning largely evaluate systems within pre-assembled contexts, but under-evaluate whether agents can use evidence distributed across independently originated sources. We argue that source-distributed memory composition is an important and under-examined bottleneck in multimodal agent memory, especially when relevant evidence is fragmented across heterogeneous artifacts such as conversations, profiles, screenshots, tables, images, and documents. To address this gap, we introduce Source-distributed Multimodal Memory Benchmark(SMMBench), which measures whether agents can retrieve, align, and compose multimodal evidence scattered across multiple sources rather than reason within a single curated context. SMMBench evaluates four core capabilities: (1) cross-source multimodal reasoning; (2) conflict resolution; (3) preference reasoning; (4) memory-grounded action prediction. The benchmark contains 1877 samples grounded in 264 sources. Experiments on representative memory-style and retrieval-based baselines show that current systems still struggle on these capabilities, positioning source-distributed multimodal memory as an important and still under-evaluated challenge for multimodal agents. Our data are available at this https URL. 

---
# H-Mem: A Novel Memory Mechanism for Evolving and Retrieving Agent Memory via a Hybrid Structure 

**Authors**: Jiawei Yu, Yixiang Fang, Xilin Liu, Yuchi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15701)  

**Abstract**: Memory data are ubiquitous in Large Language Model (LLM)-based agents (e.g., OpenClaw and Manus). A few recent works have attempted to exploit agents'memory for improving their performance on the question-answering (QA) task, but they lack a principled mechanism for effectively modeling how memory data evolves over time and retrieving memory data effectively, leading to poor performance in memory utilization. To fill this gap, we present H-Mem, a novel memory mechanism via a hybrid structure that can not only effectively model the evolution of agent memory over a long period of time, but also provide an efficient memory retrieval approach. Particularly, H-Mem builds a temporal and semantic tree structure that allows the short-term memory data to evolve progressively into long-term memory data, where the latter provides summarized information about the former, while simultaneously constructing a knowledge graph to capture the relationships between entities in memory. Moreover, it offers an effective memory retrieval approach by exploiting the hybrid structure of the tree and graph structures. Extensive experiments on three agent memory benchmarks show that H-Mem achieves state-of-the-art performance on the QA task. 

---
# ASRU: Activation Steering Meets Reinforcement Unlearning for Multimodal Large Language Models 

**Authors**: Jiahui Guang, Yingjie Zhu, Cuiyun Gao, Haiyan Wang, Jing Li, Di Shao, Zhaoquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15687)  

**Abstract**: Multimodal large language models (MLLMs) may memorize sensitive cross-modal information during pretraining, making machine unlearning (MU) crucial. Existing methods typically evaluate unlearning effectiveness based on output deviations, while overlooking the generation quality after unlearning. This can easily lead to hallucinated or rigid responses, thereby affecting the usability and safety of the unlearned model. To address this issue, we propose ASRU, a controllable multimodal unlearning framework that incorporates generation quality as a core evaluation objective. ASRU first induces initial refusal behavior through activation redirection, and then optimizes fine-grained refusal boundaries using a customized reward function, thereby achieving a better trade-off between target knowledge unlearning and model utility. Experiments on Qwen3-VL show that ASRU significantly improves unlearning effectiveness (+24.6%) on average and generation quality (5.8x) on average while effectively preserving model utility, using only a small amount of retained supervision data. 

---
# Few-Shot Large Language Models for Actionable Triage Categorization of Online Patient Inquiries 

**Authors**: Liqi Zhou, Jiafu Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.15680)  

**Abstract**: Online patient inquiries are often informal, incomplete, and written before professional assessment, yet they must still be routed to an appropriate level of clinical follow-up. We study this as a four-class actionable triage task -- self-care, schedule-visit, urgent-clinician-review, or emergency-referral, and ask whether prompted large language models (LLMs) can support such routing under low-resource labeling conditions. Using the public HealthCareMagic-100K corpus, we construct a 300-example human calibrated gold evaluation set, a 700-example auto-labeled silver training set, and a 40-example few-shot pool. We compare Term Frequency-Inverse Document Frequency (TF-IDF) and Bidirectional Encoder Representations from Transformers for Biomedical Text Mining (BioBERT) baselines train on silver labels against six prompted LLMs under 0-shot, 4-shot, and 12-shot conditions respectively. Accordingly, we evaluate with macro-$F_1$ alongside safety-aware metrics, including emergency-recall, under-triage rate, and severe under-triage rate. The strongest LLM (Claude Haiku 4.5, 12-shot) reaches macro-$F_1$ 0.475, exceeding the best supervised baseline (BioBERT, 0.378) on point estimate, with overlapping confidence intervals. Few-shot prompting and two-model agreement help in label-dependent ways: self-care agreement is reliable, urgent-clinician-review is not. We conclude that LLMs can support triage prioritization and selective human review, but not autonomous deployment. 

---
# VCG-Bench: Towards A Unified Visual-Centric Benchmark for Structured Generation and Editing 

**Authors**: Xiaoyan Su, Peijie Dong, Zhenheng Tang, Song Tang, Yuyao Zhai, Kaitao Lin, Liang Chen, Gai Yuhang, Yuyu Luo, Qiang Wang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15677)  

**Abstract**: Despite the rapid advancements in Vision-Language Models (VLMs), a critical gap remains in their ability to handle structured, controllable diagrammatic tasks essential for professional workflows. Existing methods predominantly rely on pixel-based synthesis, which operates in probabilistic pixel spaces and is inherently limited in editability and fidelity. Instead, we propose a new Diagram-as-Code paradigm with symbolic logic that leverages mxGraph Extensible Markup Language (XML) for precise diagram generation and editing. We present VCG-Bench, a unified benchmark for visual-centric \texttt{mxGraph} tasks. VCG-Bench comprises: (1) a taxonomized dataset of 1,449 diverse diagrams spanning 6 domains and 15 sub-domains, (2) a paradigm definition that integrates Generation (Vision-to-Code) and Editability (Code-to-Code), (3) a Tailored Evaluation Protocol employing multi-dimensional metrics such as \texttt{mxGraph} Execution Success Rate, Style Consistency Score (SCS), etc. Experimental results highlight the challenges faced by current State-of-the-Art (SOTA) VLMs in structured fidelity and instruction compliance, reflecting their vision and reasoning capabilities. 

---
# Dynamic Chunking for Diffusion Language Models 

**Authors**: Yichen Zhu, Xiaoming Shi, Peng Zhao, Weiyu Chen, Debing Zhang, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2605.15676)  

**Abstract**: Block discrete diffusion language models factorize a sequence autoregressively over fixed-size positional blocks, decoupling within-block parallel denoising from across-block conditioning. We argue that this rigid partition wastes structure already present in the sequence: blocks defined by position rather than by content separate semantically coherent tokens and group unrelated ones together. We introduce the \textbf{D}ynamic \textbf{C}hunking \textbf{D}iffusion \textbf{M}odel (DCDM), which replaces positional blocks with content-defined semantic chunks. At its core is Chunking Attention, a differentiable layer that routes tokens into $K$ clusters parameterized by learnable subspaces and shaped end-to-end by the diffusion objective. The resulting cluster assignments induce a chunk-causal attention mask under which a discrete diffusion denoiser factorizes the sequence likelihood autoregressively over semantic chunks, strictly generalizing block discrete diffusion. On downstream benchmarks at parameter scales up to 1.5B, DCDM consistently improves over both unstructured and positional-block diffusion baselines, with the advantage stable across scales and visible early in training. 

---
# Evaluating Chinese Ambiguity Understanding in Large Language Models 

**Authors**: Junwen Mo, Yuanzhi Lu, Yifang Xue, Ke Xu, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2605.15635)  

**Abstract**: Linguistic ambiguity is critical to the robustness of Large Language Models (LLMs), yet existing research focuses mostly on English, with limited attention devoted to Chinese. Existing Chinese ambiguity datasets (e.g., CHAmbi) suffer from poor scalability. Guided by Potential Ambiguity (PA) Theory, we design a semi-automatic pipeline to construct CHA-Gen. It is the first PA Theory-grounded Chinese ambiguity dataset, which comprises 5,712 sentences (2,414 ambiguous, 3,298 unambiguous) across 18 potential ambiguous structures. Evaluating LLMs (e.g. Gemma 3, Qwen 2.5/3 series) via direct querying and machine translation, we find that LLMs struggle with ambiguity detection (improved by CoT prompting). Analysis of Qwen3-32B's CoT rationales reveals three common failure modes: ambiguity blindness, misattribution, and premature resolution. Uncertainty quantification with semantic entropy metric shows higher uncertainty for ambiguous sentences. Moreover, instruction tuning induces overconfidence, whereas Base models better capture semantic diversity. We further observe that models exhibit a bias toward dominant interpretations. Our work provides a scalable approach for Chinese ambiguity corpus and insights into LLMs' ambiguity handling, laying a foundation for enhancing Chinese ambiguity research in LLMs. 

---
# Toward LLMs Beyond English-Centric Development 

**Authors**: Sho Takase, Ukyo Honda  

**Link**: [PDF](https://arxiv.org/pdf/2605.15613)  

**Abstract**: Through an analysis of sequences generated by open-weight large language models (LLMs), we demonstrate that LLMs are heavily biased toward English. While continual pre-training is commonly used to adapt LLMs to a target language, we show that it does not offer a cost advantage over training from scratch, even for improving cultural understanding in the target language. These findings suggest that dedicated per-language investment may become increasingly important for future LLM development, rather than relying primarily on the expansion of English-centric resources. 

---
# PSD: Pushing the Pareto Frontier of Diffusion LLMs via Parallel Speculative Decoding 

**Authors**: Shengyin Sun, Yiming Li, Renxi Liu, Xinqi Li, Hui-Ling Zhen, Weizhe Lin, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15609)  

**Abstract**: Diffusion large language models (dLLMs) generate text by iteratively denoising masked token sequences. Although dLLMs can predict all masked positions in parallel within each step, the large number of denoising iterations still makes inference expensive. This cost can be reduced spatially by unmasking multiple tokens per step, or temporally by collapsing multiple denoising steps into one verification call. We propose Parallel Speculative Decoding (PSD), a training-free framework that jointly improves inference along both axes. Using the confidence scores from a single forward pass, PSD selects positions to unmask via a configurable, adaptive unmasking policy and constructs multi-depth speculative drafts without extra model calls. A final batched verification pass then applies hierarchical acceptance, keeping the deepest draft that remains consistent with the updated predictions. Experiments on three dLLMs across reasoning and code generation tasks show that PSD achieves favorable trade-offs between inference efficiency and generation quality, reaching up to $5.5\times$ tokens per forward pass with accuracy comparable to greedy decoding. 

---
# Syntax Without Semantics: Teaching Large Language Models to Code in an Unseen Language 

**Authors**: Vinayshekhar Bannihatti Kumar, Disha Makhija, Manoj Ghuhan Arivazhagan, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2605.15607)  

**Abstract**: Large language models (LLMs) achieve high pass rates on code generation benchmarks, yet whether they can transfer this ability to languages absent from pretraining remains poorly understood. We introduce PyLang, a minimal imperative language absent from all pretraining corpora, and evaluate frontier models zero-shot and fine-tuned Qwen3 (4B, 8B, 32B) on 352 problems. We find that fine-tuning quickly teaches syntax but fails to transfer semantic competence: Python outperforms PyLang by up to 19% across all configurations, and no intervention (multi-task learning, preference tuning, code infilling, or latent-space objectives) closes the gap. An LLM judge reveals that frontier models select an identical algorithm to Python 80% of the time, yet cannot translate it into a working PyLang implementation., and CKA analysis confirms that fine-tuned models converge to nearly identical internal representations across languages (CKA > 0.97) while diverging at the output stage. We term this the implementation fidelity gap: models possess language-agnostic algorithmic understanding but cannot express it in an unfamiliar language. Our findings highlight the need for training methods that decouple reasoning from language-specific realization. 

---
# MHGraphBench: Knowledge Graph-Grounded Benchmarking of Mental Health Knowledge in Large Language Models 

**Authors**: Weixin Liu, Congning Ni, Shelagh A. Mulvaney, Susannah L. Rose, Murat Kantarcioglu, Bradley A. Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.15589)  

**Abstract**: Large language models (LLMs) are increasingly used in the mental health domain, yet it remains unclear how well they capture related biomedical knowledge and how reliably they apply it to clinically salient structured judgments. Here, we present a knowledge-graph (KG)-grounded benchmark for assessing LLMs on mental-health entity recognition, relation judgment, and two-hop reasoning. The benchmark is derived from PrimeKG and comprises nine task families with KG-supported answers and controlled negative options. Experiments across 15 closed- and open-source LLMs reveal a persistent recognition-to-judgment gap: leading models achieve near-ceiling performance on entity typing and on the small relation-typing subset, yet they still struggle with relation prediction and two-hop reasoning. Additionally, short KG-derived snippets benefit some models but degrade performance for others. Moreover, output-format reliability can substantially influence measured performance under constrained multiple-choice settings, highlighting the critical role of response validity in benchmark-based evaluation. MHGraphBench should therefore be interpreted as evaluating agreement with a curated mental-health slice of PrimeKG under a constrained multiple-choice interface, rather than as a direct assessment of real-world clinical safety. 

---
# Calibrating LLMs with Semantic-level Reward 

**Authors**: Fengfei Yu, Ruijia Niu, Dongxia Wu, Yian Ma, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15588)  

**Abstract**: As large language models (LLMs) are deployed in consequential settings such as medical question answering and legal reasoning, the ability to estimate when their outputs are likely to be correct is essential for safe and reliable use, requiring well-calibrated uncertainty. Standard reinforcement learning with verifiable rewards (RLVR) trains models with a binary correctness reward that is indifferent to confidence, providing no penalty for confident but wrong predictions and thereby degrading calibration. Recent work addresses this by training models to produce verbalized confidence scores alongside answers and rewarding agreement with correctness. However, verbalized confidence is calibrated at the token level and thus exhibits inconsistency across textual variations with same semantic meaning. We propose \textbf{Calibration with Semantic Reward (CSR)}, a framework that calibrates language models directly in semantic space without a verbalized confidence interface. CSR combines the correctness reward with a novel semantic calibration reward that encourages exploitation among correct rollouts by promoting semantic agreement, and exploration among incorrect ones by discouraging spurious consistency. Experiments across three model families on HotpotQA (in-distribution) and TriviaQA, MSMARCO, and NQ-Open (out-of-distribution) show that CSR consistently achieves lower ECE and higher AUROC than verbalized-confidence baselines across nearly all settings, reducing ECE by up to $40\%$ and improving AUROC by up to $31\%$ over verbalized-confidence baselines, with calibration behavior generalizing robustly across all four evaluation settings. 

---
# Response-Conditioned Parallel-to-Sequential Orchestration for Multi-Agent Systems 

**Authors**: Nurbek Tastan, Alex Iacob, Lorenzo Sani, Meghdad Kurmanji, Nicholas D. Lane, Samuel Horvath, Karthik Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2605.15573)  

**Abstract**: Multi-agent systems can solve complex tasks through collaboration between multiple Large Language Model agents. Existing collaboration frameworks typically operate in either a parallel or a sequential mode. In the parallel mode, agents respond independently to queries followed by aggregation of responses. In contrast, sequential systems allow agents to communicate via a directed topology and refine one another step by step. However, both modes are inadequate for achieving the desired objectives of minimizing communication and latency while simultaneously maximizing the accuracy of the final response. In this work, we introduce a hybrid paradigm called Nexa, a trainable response-conditioned policy that bridges the gap between the two modes. Nexa begins with a parallel execution stage, embeds the resulting responses into a shared semantic space, and then predicts a sparse directed acyclic communication graph. If the graph is empty, the system remains purely parallel; if it is non-empty, the system performs one sequential message propagation. The policy is a lightweight transformer model, and the method avoids the need for external LLM judges or reward models, as well as hand-crafted test-time topology search. We formalize this hybrid execution problem, show that the resulting graph is acyclic by construction, and that the framework strictly subsumes pure parallel execution, and present a training procedure based on policy-gradient optimization. Results demonstrate that the response-conditioned policy learned by Nexa under one setting can be reused when the number of agents, the task, or the underlying agent changes, thus emphasizing the generalizability of the learned communication policy. 

---
# Measuring Maximum Activations in Open Large Language Models 

**Authors**: Luxuan Chen, Han Tian, Xinran Chen, Rui Kong, Fang Wang, Jiamin Chen, Yuchen Li, Jiashu Zhao, Shuaiqiang Wang, Haoyi Xiong, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2605.15572)  

**Abstract**: The dynamic range of activations is a first-order constraint for low-bit quantization, activation scaling, and stable LLM inference. Prior work characterized outlier features and massive activations on pre-2024 LLaMA-style models, and the downstream activation-quantization stack inherits that picture without revisiting it for the post-LLaMA open-model boom. We ask the deployment-oriented question: how large can activations get in modern open LLMs, and how does this magnitude vary across families, generations, and training stages? Under a unified pipeline (5,000-sample multi-domain corpus, family-specific tokenization, identical hooks across embeddings, hidden states, attention, MLP/MoE, SwiGLU gates, and final norm), we measure global and layerwise maxima on 27 checkpoints from 8 open families spanning dense, MoE, vision-language, intermediate-training, and instruction-tuned variants. We find that (i) global maxima span over nearly four orders of magnitude at comparable parameter counts, with Qwen3.5 and MoE checkpoints in the 10^2 to 10^3 range and Gemma3-27B-it reaching ~7 x 10^5; (ii) cross-family and cross-generation comparisons break simple monotonic scaling; and (iii) MoE checkpoints exhibit 14.0-23.4x lower peaks than matched-scale dense counterparts, while the residual stream carries the global maximum in 22/24 checkpoints. A lightweight INT-8 sanity check shows that measured maxima co-vary with low-bit reconstruction error via activation-scale selection. We conclude that maximum activation magnitude is a model property tied to family, architecture, and training stage - not a simple byproduct of size - and should be measured and reported alongside any open-weight release before low-bit deployment. The code is publicly available at this https URL. 

---
# GiLT: Augmenting Transformer Language Models with Dependency Graphs 

**Authors**: Tianyu Huang, Yida Zhao, Chuyan Zhou, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2605.15562)  

**Abstract**: Augmenting Transformers with linguistic structures effectively enhances the syntactic generalization performance of language models. Previous work in this direction focuses on syntactic tree structures of languages, in particular constituency tree structures. We propose Graph-Infused Layers Transformer Language Model (GiLT) which leverages dependency graphs for augmenting Transformer language models. Unlike most previous work, GiLT does not insert extra structural tokens in language modeling; instead, it injects structural information into language modeling by modulating attention weights in the Transformer with features extracted from the dependency graph that is incrementally constructed along with token prediction. In our experiments, GiLT with semantic dependency graphs achieves better syntactic generalization while maintaining competitive perplexity in comparison with Transformer language model baselines. In addition, GiLT can be finetuned from a pretrained language model to achieve improved downstream task performance. Our code is released at this https URL. 

---
# When Latent Geometry Is Not Enough: Draft-Conditioned Latent Refinement for Non-Autoregressive Text Generation 

**Authors**: De Shuai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15557)  

**Abstract**: Continuous diffusion and flow models are attractive for non-autoregressive text generation because they can update all positions in parallel. A major difficulty is the interface between continuous latent states and discrete tokens. This report studies a draft-conditioned latent refinement model built from a frozen BERT encoder, a parallel decoder, a denoising DraftPrior, a local FlowNet, and a learned diagonal MetricNet. Early Gaussian-start experiments showed that good latent-space metrics, such as scale matching or cosine similarity, do not guarantee good decoding. Generated latents can be close to real encoder latents but still produce high-entropy, biased, or repetitive token distributions. We therefore frame the task as controlled local refinement rather than full generation from noise. On ROCStories, using the first two sentences as prompt and the last three as target, full 768-dimensional BERT latents recover tokens much better than compressed 256-dimensional latents. With 768-dimensional latents, DraftPrior target-token probability is 0.938 for clean drafts, 0.613 for 3% token dropout, 0.483 for 5% dropout, and 0.272 for 10% dropout. Local flow refinement and fused decoder-aware readout give modest additional gains, while metric learning and OT-style alignment improve geometry but do not close the decoder gap. The main result is a diagnostic one: latent geometry alone is not enough. Continuous latent text generation should be evaluated by decoder recoverability, the quality of the start distribution, and whether refinement preserves decoder-readable structure. 

---
# Process Rewards with Learned Reliability 

**Authors**: Jinyuan Li, Langlin Huang, Chengsong Huang, Shaoyang Xu, Donghong Cai, Yuyi Yang, Wenxuan Zhang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15529)  

**Abstract**: Process Reward Models (PRMs) provide step-level feedback for reasoning, but current PRMs usually output only a single reward score for each step. Downstream methods must therefore treat imperfect step-level reward predictions as reliable decision signals, with no indication of when these predictions should be trusted. We propose BetaPRM, a distributional PRM that predicts both a step-level success probability and the reliability of that prediction. Given step-success supervision from Monte Carlo continuations, BetaPRM learns a Beta belief that explains the observed number of successful continuations through a Beta-Binomial likelihood, rather than regressing to the finite-sample success ratio as a point target. This learned reliability signal indicates when a step reward should be trusted, enabling downstream applications to distinguish reliable rewards from uncertain ones. As one application, we introduce Adaptive Computation Allocation (ACA) for PRM-guided Best-of-N reasoning. ACA uses the learned reliability signal to stop when a high-reward solution is reliable and to spend additional computation on uncertain candidate prefixes. Experiments across four backbones and four reasoning benchmarks show that BetaPRM improves PRM-guided Best-of-N selection while preserving standard step-level error detection. Built on this signal, ACA improves the accuracy--token tradeoff over fixed-budget Best-of-16, reducing token usage by up to 33.57% while improving final-answer accuracy. 

---
# DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection 

**Authors**: Junchao Wu, Yefeng Liu, Chenyu Zhu, Hao Zhang, Zeyu Wu, Tianqi Shi, Yichao Du, Longyue Wang, Weihua Luo, Jinsong Su, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2605.15518)  

**Abstract**: The effective detection and governance of Large Language Model (LLM) generated content has become increasingly critical due to the growing risk of misuse. Despite the impressive performance of existing detectors, their reliability and potential in multilingual, real-world scenarios remain largely underexplored. In this study, we introduce DetectRL-X, a comprehensive multilingual benchmark designed to evaluate advanced detectors across 8 dimensions. The benchmark encompasses 8 languages commonly used in commercial contexts and collects human-written texts from 6 domains highly susceptible to LLM misuse. To better aligned with real-world applications, We create LLM-generated texts using 4 popular commercial LLMs, and include typical AI-assisted writing operations such as polishing, expanding, and condensing to capture authentic usage patterns. Furthermore, we develop a multilingual framework for paraphrasing and perturbation attacks to simulate diverse human modifications and writing noise, enabling stress testing of detectors across languages. Experimental results on DetectRL-X reveal the strengths and limitations of current state-of-the-art detectors when applied to diverse linguistic resources. We further analyze how domains, generators, attack strategies, text length, and refinement operations influence performance in different languages, underscoring DetectRL-X as an effective benchmark for strengthening multilingual and language-specific detectors. 

---
# RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably 

**Authors**: Yufeng Du, Phillip Harris, Minyang Tian, Eliu A Huerta, Srikanth Ronanki, Subendhu Rongali, Aram Galstyan, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15514)  

**Abstract**: We identify intrinsic limitations of Rotary Positional Embeddings (RoPE) in Transformer-based long-context language models. Our theoretical analysis abstracts away from the specific content of the context and depends only on its length. We prove that as context length increases, RoPE-based attention becomes unpredictable and loses two properties that are central to its effectiveness. First, it loses its locality bias: RoPE is no more likely to favor nearer positions than substantially farther ones. Second, it loses consistency in token relevance: a key vector that receives a higher attention score than an alternative at one position may receive a lower score at another. In both cases, the probability of failure approaches 0.5, no better than random guessing. We further prove that the attention score can remain unchanged when a key token is moved to a different position, or even replaced by a different token, indicating a failure to distinguish positions or tokens. Adjusting the RoPE base trades off distinguishing positions against distinguishing tokens but cannot preserve both at the same time. Increasing the RoPE base hyperparameter, a common practice in today's long-context models, helps distinguish different tokens, but inevitably sacrifices the ability to distinguish positions. Our empirical analysis shows that multi-head, multi-layer architectures are insufficient to overcome these limitations. Our findings suggest that fundamentally new mechanisms for encoding position and token order may be needed in future Transformer long-context language models. 

---
# FINESSE-Bench: A Hierarchical Benchmark Suite for Financial Domain Knowledge and Technical Analysis in Large Language Models 

**Authors**: Dmitry Stanishevskii, Nini Kamkia, Alexey Khoroshilov, Dmitry Zmitrovich, Denis Kokosinskii, Zhirayr Hayrapetyan, Andrei Kalmykov  

**Link**: [PDF](https://arxiv.org/pdf/2605.15482)  

**Abstract**: Large language models (LLMs) are increasingly being applied to financial analysis, reporting, investment decision support, risk management, compliance, and professional training. However, robust evaluation of their domain competence in finance remains incomplete. Widely used open benchmarks such as FinQA, ConvFinQA, and TAT-QA have played an important role in advancing financial question answering and numerical reasoning, but they focus primarily on question answering over financial reports and do not provide an explicit hierarchy of professional difficulty. Broader resources, including FinanceBench, PIXIU, FinBen, and FLaME, expand the coverage of financial tasks, yet the problem of evaluating the transition from foundational knowledge to expert-level financial reasoning remains open. In this work, we present FINESSE-Bench, a suite of eight specialized benchmarks comprising 3,993 questions for hierarchical evaluation of financial competencies in LLMs. FINESSE-Bench combines exam-oriented datasets inspired by professional certifications (CFA-like Levels 1-3, CMT-like Level 2, and CFTe-like Level 1), applied trading task collections, and a Russian-language olympiad benchmark. This design enables evaluation of domain breadth, performance degradation as difficulty increases, the ability to solve computational tasks, and model behavior in specialized financial domains. We also describe a unified evaluation protocol covering multiple-choice questions, numerical answers, and short open-ended responses, together with an automated scoring scheme for freeform answers based on the LLM-as-judge paradigm. FINESSE-Bench is intended both as a complement to existing open financial benchmarks and as a tool for more substantive evaluation of professionally relevant financial competencies in large language models. 

---
# Retrieval-Augmented Large Language Models for Schema-Constrained Clinical Information Extraction 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2605.15467)  

**Abstract**: Conversational nurse-patient transcripts contain actionable observations, but converting these transcripts into structured representations at scale remains challenging. Documentation burden is substantial, with prior studies showing clinicians spend large portions of their workday on documentation and related desk work rather than direct patient care. MEDIQA-SYNUR focuses on observation extraction from conversational nurse-patient transcripts, requiring systems to normalize these narratives into a predefined schema with value-type constraints. We propose a modular retrieval-augmented generation (RAG) pipeline that uses the training set as an exemplar corpus, combines schema-constrained prompting (full schema vs. pruned candidate schema), deterministic schema-based postprocessing, and a second-pass audit, with two LLM backbones: Llama-4-Scout-17B-16E-Instruct and GPT-5.2 with corresponding embedding models for RAG. Our best configuration uses GPT-5.2 with full schema, RAG, and a second-pass auditing, achieving 80.36% F1 score. Overall, our results show that RAG consistently improves performance, while the optimal degree of schema constraint depends on the model, and second-pass auditing yields modest additional gains by correcting residual schema-adherence errors. 

---
# Reasoning Models Don't Just Think Longer, They Move Differently 

**Authors**: Anders Gjølbye, Lars Kai Hansen, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15454)  

**Abstract**: Reasoning-trained language models often spend more tokens on harder problems, but longer chains of thought do not show whether a model is merely computing for more steps or following a different internal trajectory. We study this distinction through hidden-state trajectories during chain-of-thought generation across competitive programming, mathematics, and Boolean satisfiability. Raw trajectory geometry is strongly shaped by generation length: longer generations mechanically alter path statistics, so difficulty-dependent comparisons are misleading without adjustment. After residualizing trajectory statistics on length, difficulty remains systematically coupled to corrected trajectory geometry across all domains studied. The clearest reasoning-specific separation appears in the code domain, where harder problems show more direct corrected trajectories and less heterogeneous local curvature in reasoning-trained models than in matched instruction-tuned baselines. Corrected difficulty-geometry coupling is weaker, but still present, in mathematics and Boolean satisfiability. Prompt-stage linear probes do not mirror the code-domain separation, and behavioral annotations show that stronger corrected coupling co-occurs with strategy shifts and uncertainty monitoring. Together, these findings establish length correction as a prerequisite for generation-time trajectory analysis and show that reasoning training can be associated with distinct corrected trajectory geometry, with the strength of the effect depending on the domain. 

---
# Why are language models less surprised than humans? Testing the Parse Multiplicity Mismatch Hypothesis 

**Authors**: William Timkey, Brian Dillon, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15440)  

**Abstract**: Surprisal theory posits that the processing difficulty of a word is determined by its predictability in context, offering a potential link between human sentence processing and next-word predictions from language models. While language model (LM) surprisals successfully predict reading times in naturalistic text, they systematically underpredict the magnitude of difficulty observed in controlled studies of syntactic ambiguity, particularly in garden path sentences. This mismatch might arise from differences in the computational constraints between humans and LMs. Here we test one such hypothesis, specifically, that LMs may be able to simultaneously consider a greater number of distinct sentence interpretations at once, compared to humans. Using Recurrent Neural Network Grammars (RNNGs) with word-synchronous beam search, we systematically vary the number of simultaneous parses used to compute word surprisal, and then use these surprisals to predict human reading times. Reducing the number of simultaneous active parses indeed increases the magnitude of predicted garden path effects, but not nearly enough to capture the full magnitude of the effects in humans. This suggests that differences in the number of simultaneous parses available to LMs and humans cannot reconcile LM-based surprisal with human sentence processing. 

---
# Neural Activation Patterns Across Language Model Architectures: A Comprehensive Analysis of Cognitive Task Performance 

**Authors**: Mahdi Naser-Moghadasi, Faezeh Ghaderi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15436)  

**Abstract**: This paper presents a comprehensive analysis of neural activation patterns across six distinct large language model (LLM) architectures, examining their performance on twelve cognitive task categories. Through systematic measurement of final activation values, attention entropy, and sparsity patterns, we reveal fundamental differences in how encoder and decoder architectures process diverse cognitive tasks. Our analysis of 144 task-model combinations demonstrates that mathematical reasoning consistently produces the highest attention entropy across all architectures, while decoder models exhibit significantly higher sparsity patterns compared to encoder models. The findings provide critical insights into the computational characteristics of modern language models and their task-specific neural behaviors, with implications for model selection and optimization in big data applications. 

---
# Capability Conditioned Scaffolding for Professional Human LLM Collaboration 

**Authors**: Sen Yang, Yinglei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15404)  

**Abstract**: Large language model personalization typically adapts outputs to user preferences and style but does not account for differences in user evaluation capacity across domains of expertise. This limitation can encourage Professional Domain Drift, where users rely on AI generated reasoning in domains they cannot reliably evaluate. We introduce Capability Conditioned Scaffolding, a typed framework that partitions expertise into strong, mixed, and weak domains and conditions intervention behavior on structured capability profiles. A pilot evaluation across multiple MMLU subsets and four LLM substrates shows consistent profile conditioned intervention behavior, including categorical inversion under profile swapping and selective activation in mixed domain risk zones. These findings suggest that capability aware scaffolding can support more reliable professional human AI collaboration beyond stylistic personalization. 

---
# Eskwai for Students: Generative AI Assistant for Legal Education in Ghana 

**Authors**: George Boateng, Philemon Badu, Patrick Agyeman-Budu, Samuel Ansah, Evans Atompoya, Evan Igwilo, Lord Baah, Frederick Abu-Bonsrah, Victor Wumbor-Apin Kumbol  

**Link**: [PDF](https://arxiv.org/pdf/2605.15380)  

**Abstract**: Recent advances in generative AI have shown their potential to be leveraged for legal education. Yet, work on the development and deployment of such systems for legal education in the Global South is limited. In this work, we developed Eskwai for Students, a generative AI assistant to help law students with their legal education. Eskwai for Students is a retrieval augmented generation (RAG) system that provides answers to a wide range of legal questions for law students grounded in a curated database of over 12K case laws and 1.4K legislation in Ghana. We deployed Eskwai for Students in a longitudinal study of 30 months (2.5 years) used by 3.1K law students in Ghana who made 32K queries. We evaluated the helpfulness of our AI, and provided insight into the kinds of queries law students submit to this generative AI tool, which raises some ethical concerns. This work contributes to an understanding of how law students in the Global South are using generative AI for their studies and the ways it could be leveraged responsibly to advance legal education. 

---
# Adesua: Development and Feasibility Study of an AI WhatsApp Bot for Science Learning in West Africa 

**Authors**: George Boateng, Evans Atompoya, Philemon Badu, Samuel John, Samuel Ansah, Patrick Agyeman-Budu, Victor Wumbor-Apin Kumbol  

**Link**: [PDF](https://arxiv.org/pdf/2605.15376)  

**Abstract**: Sub-Saharan Africa faces persistently high student-teacher ratios and shortages of qualified teachers, limiting students' access to personalized learning support and formative assessment. To address this challenge, we present Adesua, a WhatsApp-based AI Teaching Assistant for science education that extends the Kwame for Science platform. Adesua leverages WhatsApp's widespread adoption in Africa to provide accessible, curriculum-aligned learning support for Junior High School (JHS) and Senior High School (SHS) students across West Africa. The system integrates curated textbooks and 33 years of national examination questions with generative AI to enable conversational question answering and automated assessment with feedback via a WhatsApp bot. Students can ask science questions, take timed or untimed multiple-choice tests by topic or exam year, and receive instant grading and detailed explanations of correct and incorrect responses. A 6-month feasibility deployment in 2025 had 56 active users in Ghana, including students and parents. Quantitative evaluation showed a high perceived usefulness, with a helpfulness score of 93.75\% for AI-generated answers, albeit with a small number of ratings (n=16). These preliminary results provide a basis for more extensive future evaluation of a WhatsApp-based AI assistant to assess its potential to offer scalable, low-cost personalized learning support and formative assessment in resource-constrained educational contexts. 

---
# Greedy or not, here I come: Language production under vocabulary constraints in humans and resource-rational models 

**Authors**: Thomas Hikaru Clark, Sihan Chen, Laura Nicolae  

**Link**: [PDF](https://arxiv.org/pdf/2605.15365)  

**Abstract**: Communicating using only a limited vocabulary is a common but challenging cognitive phenomenon, requiring an ideal communicator to plan carefully to optimize for intelligibility while circumventing a constrained lexicon. In this work, we investigate how humans respond to a broad array of questions under variable vocabulary limitations, consisting of only 250 highly frequent words at the most restrictive. We provide theoretically motivated comparisons to greedy and globally optimal sampling algorithms using Sequential Monte Carlo inference with large language models. Humans generally resemble greedy sampling more than globally optimal sampling, though more skilled humans are more likely to backtrack and revise -- a non-greedy behavior. An observed human pattern of leaning on semantically light words in high-constraint settings falls out of both greedy and globally optimal sampling. We discuss the results and their broader implications for resource-rational cognition, psycholinguistics, L2 communication, and language impairments. 

---
# Automatic Construction of a Legal Citation Graph from 100 Million Ukrainian Court Decisions: Large-Scale Extraction, Topological Analysis, and Ontology-Driven Clustering 

**Authors**: Volodymyr Ovcharov  

**Link**: [PDF](https://arxiv.org/pdf/2605.15362)  

**Abstract**: Half a billion citation edges extracted from 100.7 million Ukrainian court decisions reveal that judicial citation structure encodes legal domain boundaries without supervision and predicts future legislative importance with near-perfect accuracy. We construct the first large-scale citation graph from the complete EDRSR registry (99.5 million full texts, 1.1 TB), extracting 502 million citation links across six types via regex on commodity hardware in approximately 5 hours, with precision of 1.00 on a 200-decision validation sample (95% Wilson CI: [0.982, 1.000]).
Three principal findings emerge. (1) The degree distribution follows a power law (alpha = 1.57 +/- 0.008), placing the Ukrainian court network near the EU Court of Justice and below the US Supreme Court, with hub articles cited by millions of decisions. (2) Louvain community detection on the co-citation projection recovers legal domain boundaries (civil, criminal, administrative, commercial) with modularity Q = 0.44-0.55 and temporal stability (NMI = 0.83-0.86 across periods), constituting an automatically constructed legal ontology grounded in judicial practice. (3) Citation features predict top-1000 articles with AUC = 0.9984, substantially outperforming a naive frequency baseline (P@1000 = 0.655); temporal dynamics detect legislative regime changes as phase transitions and the 2022 invasion as a citation entropy spike (H: 11.02 -> 13.49) with emergent wartime legislation nodes.
The citation-derived ontology is operationalized as the domain layer of a workflow memory system for LLM-assisted legal analysis, connecting to the ontology-controlled paradigm. The extraction pipeline, analysis code, and aggregated statistics are released as open data. 

---
# DiscoExplorer: An Open Interface for the Study of Multilingual Discourse Relations 

**Authors**: Amir Zeldes  

**Link**: [PDF](https://arxiv.org/pdf/2605.15304)  

**Abstract**: The relations connecting propositions in discourse such as cause (A because B) or concession (A although B) are a subject of intense interest in Computational Linguistics and Pragmatics, but challenging to study and compare across languages. Recent progress in standardizing discourse relation inventories across datasets offers the potential to facilitate such studies, but is hindered by the complexity of relevant data and the lack of easily accessible interfaces to analyze it. In this paper we present DiscoExplorer, a new open source web interface, capable of running on local computers, which we use to make datasets from the DISRPT Shared Task on discourse relation classification publicly available, covering 16 different languages. We present the query language, search and visualization facilities for relations and signaling devices such as connectives, as well as some example studies. 

---
# Fluency and Faithfulness in Human and Machine Literary Translation 

**Authors**: Sarah Griebel, Ted Underwood  

**Link**: [PDF](https://arxiv.org/pdf/2605.15282)  

**Abstract**: Literary translation requires balancing target-language fluency with faithfulness to the source. Recent large language models (LLMs) often produce fluent translations, but it remains unclear whether fluency corresponds to semantic preservation in literary text. We examine this relationship using 130,486 translated paragraphs from 106 novels in 16 source languages, including human, Google Translate, and TranslateGemma translations. Fluency is measured as original-likeness with a translationese classifier trained on paragraph part-of-speech n-grams, and faithfulness with the automatic translation evaluation metric COMET-KIWI. We control for paragraph length and find a consistent negative correlation between fluency and faithfulness. The pattern appears for both human and Google Translate, but is weaker and often non-significant for TranslateGemma. These results show that segment length matters for automatic evaluation and suggest a tradeoff between fluency and faithfulness in literary translation. 

---
# Always Learning, Always Mixing: Efficient and Simple Data Mixing All The Time 

**Authors**: Michael Y. Hu, Apurva Gandhi, Kyunghyun Cho, Tal Linzen, Pratyusha Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15220)  

**Abstract**: Data mixing decides how to combine different sources or types of data and is a consequential problem throughout language model training. In pretraining, data composition is a key determinant of model quality; in continual learning and adaptation, it governs what is retained and acquired. Yet existing data mixing methods address only one phase of this lifecycle at a time: some require smaller proxy models tied to a single training phase, others assume a fixed domain set, and continual learning lacks principled guidance altogether. We argue that data mixing is fundamentally an online decision making problem -- one that recurs throughout training and demands a single, unified solution. We introduce OP-Mix (On-Policy Mix), a data mixing algorithm that operates across the entire language model training lifecycle. Our main insight is that candidate data mixtures can be cheaply simulated by interpolating between low-rank adapters trained directly on the current model, eliminating separate proxy models and ensuring the search is always grounded in the model's actual learning dynamics. Across pretraining, continual midtraining, and continual instruction tuning, OP-Mix consistently finds near-optimal mixtures while using a fraction of the compute of the baselines. In pretraining, OP-Mix improves upon training without mixing by 6.3% in average perplexity. For continual learning, OP-Mix matches the performance of both retraining and on-policy distillation while using 66% and 95% less overall compute, respectively. OP-Mix suggests a different view of language model training: not a sequence of distinct phases, but a single continuous process of learning from data. 

---
# AI-Mediated Communication Can Steer Collective Opinion 

**Authors**: Stratis Tsirtsis, Kai Rawal, Chris Russell, Brent Mittelstadt, Sandra Wachter  

**Link**: [PDF](https://arxiv.org/pdf/2605.16245)  

**Abstract**: Generative artificial intelligence (AI) is increasingly integrated into the online platforms where humans exchange opinions; large language models (LLMs) now polish users' posts on LinkedIn and provide context for content shared on X. While prior work has shown that AI can express biased opinions and shape individuals' opinions during human-AI interactions, less attention has been paid to its influence on collective opinion formation when mediating human-to-human communication. We address this gap via a combination of empirical and theoretical analyses. We show empirically that LLMs from multiple popular families introduce directional biases when instructed to edit human-written texts on contested topics, for example, nudging texts in favor of gun control and against atheism. Building on this observation, we introduce a mathematical model of opinion dynamics in which an AI system sits between users on a social network, transforming the opinions they express and perceive. By analytically characterizing the equilibrium of this model and performing simulations on real social network data, we show that biases introduced by AI in human-to-human communication can be amplified through the network and shift collective opinion in their direction. In light of these findings, we investigate whether such biases are controllable by online platforms. We audit the "Explain this post" feature on X and find evidence of pro-life bias in Grok's outputs on abortion-related content, which we trace back to specific design choices. We conclude with a discussion of the broader implications of our findings in relation to ongoing legislative efforts in the European Union. 

---
# Layer Equivalence Is Not a Property of Layers Alone: How You Test Redundancy Changes What You Find 

**Authors**: Gabriel Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2605.16234)  

**Abstract**: When researchers ask whether two transformer layers are "equivalent" for compression, they often conflate distinct tests. Replacement asks whether one layer's map can substitute for another's in place; interchange asks whether two layers approximately commute when their positions are swapped. Both are output-grounded swap-KL probes, but they need not agree: on pretrained transformers the protocol gap can change which layers look safe to prune by several-fold under the same evaluator, especially when replacement distances are high.
We measure both protocols across checkpoints and architectures. On a Pythia training trajectory (410M and 1.4B), the replacement-interchange gap grows from initialization to convergence. Under one matched WikiText-2 contract at 8B scale, Qwen3-8B enters a divergent regime: interchange-guided removal is several-fold safer than replacement-guided at the same layer budgets, while Llama-3.1-8B ties the two protocols for pruning cost even though interchange KL is lower, showing metric gaps need not map one-to-one to removal. Before layer removal or merging, score both swap-KLs on the target checkpoint; the diagnostic requires only unlabeled forward passes. 

---
# FORGE: Self-Evolving Agent Memory With No Weight Updates via Population Broadcast 

**Authors**: Igor Bogdanov, Chung-Horng Lung, Thomas Kunz, Jie Gao, Adrian Taylor, Marzia Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.16233)  

**Abstract**: Can LLM agents improve decision-making through self-generated memory without gradient updates? We propose FORGE (Failure-Optimized Reflective Graduation and Evolution), a staged, population-based protocol that evolves prompt-injected natural-language memory for hierarchical ReAct agents. FORGE wraps a Reflexion-style inner loop, where a dedicated reflection agent (using the same underlying LLM, no distillation from a stronger model) converts failed trajectories into reusable knowledge artifacts: textual heuristics (Rules), few-shot demonstrations (Examples), or both (Mixed), with an outer loop that propagates the best-performing instance's memory to the population between stages and freezes converged instances via a graduation criterion. We evaluate on CybORG CAGE-2, a stochastic network-defense POMDP at a 30-step horizon against the B-line attacker, where all four tested LLM families (Gemini-2.5-Flash-Lite, Grok-4-Fast, Llama-4-Maverick, Qwen3-235B) exhibit strongly negative, heavy-tailed zero-shot rewards. Compared against both a zero-shot baseline and a Reflexion baseline (isolated single-stream learning), FORGE improves average evaluation return by 1.7-7.7$\times$ over zero-shot and by 29-72% over Reflexion in all 12 model-representation conditions, reducing major-failure rates (below $-100$) to as low as $\sim$1%. We find that (1) population broadcast is critical mechanism, with a no-graduation ablation confirming that broadcast carries the performance gains while graduation primarily saves compute; (2) Examples achieves the strongest returns for three of four models, Rules offers the best cost-reliability profile with $\sim$40% fewer tokens; and (3) weaker baseline models benefit disproportionately, suggesting FORGE may mitigate capability gaps rather than amplify strong models. All evidence is confined to CAGE-2 B-line; cross-family findings are directional evidence. 

---
# Fully Open Meditron: An Auditable Pipeline for Clinical LLMs 

**Authors**: Xavier Theimer-Lienhard, Mushtaha El-Amin, Fay Elhassan, Sahaj Vaidya, Victor Cartier-Negadi, David Sasu, Lars Klein, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2605.16215)  

**Abstract**: Clinical decision support systems (CDSS) require scrutable, auditable pipelines that enable rigorous, reproducible validation. Yet current LLM-based CDSS remain largely opaque. Most "open" models are open-weight only, releasing parameters while withholding the data provenance, curation procedures, and generation pipelines that determine model behavior. Fully Open (FO) models, which expose the complete training stack end-to-end, do not currently exist in medicine. We introduce Fully Open Meditron, the first fully open pipeline for building LLM-CDSS, comprising a clinician-audited training corpus, a reproducible data construction and training framework, and a use-aligned evaluation protocol. The corpus unifies eight public medical QA datasets into a normalized conversational format and expands coverage with three clinician-vetted synthetic extensions: exam-style QA, guideline-grounded QA derived from 46,469 clinical practice guidelines, and clinical vignettes. The pipeline enforces system-wide decontamination, gold-label resampling of teacher generations, and end-to-end validation by a four-physician panel. We evaluate using an LLM-as-a-judge protocol over expert-written clinical vignettes, calibrated against 204 human raters. We apply the recipe to five FO base models (Apertus-70B/8B-Instruct, OLMo-2-32B-SFT, EuroLLM-22B/9B-Instruct). All MeditronFO variants are preferred over their bases. Apertus-70B-MeditronFO improves +6.6 points over its base (47.2% to 53.8%) on aggregate medical benchmarks, establishing a new FO SoTA. Gemma-3-27B-MeditronFO is preferred over MedGemma in 58.6% of LLM-as-a-judge comparisons and outperforms it on HealthBench (58% vs 55.9%). These results show that fully open pipelines can achieve state-of-the-art domain-specific performance without sacrificing auditability or reproducibility. 

---
# Confirming Correct, Missing the Rest: LLM Tutoring Agents Struggle Where Feedback Matters Most 

**Authors**: Tahreem Yasir, Wenbo Li, Sam Gilson, Sutapa Dey Tithi, Xiaoyi Tian, Tiffany Barnes  

**Link**: [PDF](https://arxiv.org/pdf/2605.16207)  

**Abstract**: Effective tutoring requires distinguishing optimal, valid but suboptimal, and incorrect student solutions, a distinction central to intelligent tutoring systems (ITS) but untested for LLM-based tutors. As LLMs are increasingly explored as conversational complements to ITS, evaluating their diagnostic precision is essential. We present a benchmark of seven LLM feedback agents in propositional logic using knowledge-graph-derived ground truth across 10,836 solution--feedback pairs and three feedback conditions. Models achieved near-ceiling performance on optimal steps but systematically over-rejected valid but suboptimal reasoning and over-validated incorrect solutions, precisely where adaptive tutoring matters most. These failures persisted across models regardless of solution context, suggesting architectural rather than informational limits. Moreover, accurate diagnosis did not reliably produce pedagogically actionable feedback, revealing a gap between diagnostic judgment and instructional effectiveness. Our findings suggest that LLMs are better suited for hybrid architectures where KG-grounded models handle diagnosis while LLMs support open-ended scaffolding and dialogue. 

---
# Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP 

**Authors**: Igor Bogdanov, Chung-Horng Lung, Thomas Kunz, Jie Gao, Adrian Taylor, Marzia Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2605.16205)  

**Abstract**: Deploying compound LLM agents in adversarial, partially observable sequential environments requires navigating several design dimensions: (1) what the agent sees, (2) how it reasons, and (3) how tasks are decomposed across components. Yet practitioners lack guidance on which design choices improve performance versus merely increase inference costs. We present a controlled study of compound LLM agent design in CybORG CAGE-2, a cyber defense environment modeled as a Partially Observable Markov Decision Process (POMDP). Reward is non-positive, so all configurations operate in a failure-mitigation mode. Our evaluation spans five model families, six models, and twelve configurations (3,475 episodes) with token-level cost accounting. We vary context representation (raw observations vs. a deterministic state-tracking layer with compressed history), deliberation (self-questioning, self-critique, and self-improvement tools, with optional chain-of-thought prompting), and hierarchical decomposition (monolithic ReAct vs. delegation to specialized sub-agents). We find that: (1) Programmatic state abstraction delivers the largest returns per token spent (RPTS), improving mean return by up to 76% over raw observations. (2) Distributing deliberation tools across a hierarchy degrades performance relative to hierarchy alone for all five model families, reaching up to 3.4$\times$ worse mean return while using 1.8-2.7$\times$ more tokens. We call this destructive pattern a deliberation cascade. (3) Hierarchical decomposition without deliberation achieves the best absolute performance for most models, and context engineering is generally more cost-effective than deliberation. These findings suggest a design principle for structured adversarial POMDPs: invest in programmatic infrastructure and clean task decomposition rather than deeper per-agent reasoning, as these strategies can interfere when combined. 

---
# Look Before You Leap: Autonomous Exploration for LLM Agents 

**Authors**: Ziang Ye, Wentao Shi, Yuxin Liu, Yu Wang, Zhengzhou Cai, Yaorui Shi, Qi Gu, Xunliang Cai, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2605.16143)  

**Abstract**: Large language model based agents often fail in unfamiliar environments due to premature exploitation: a tendency to act on prior knowledge before acquiring sufficient environment-specific information. We identify autonomous exploration as a critical yet underexplored capability for building adaptive agents. To formalize and quantify this capability, we introduce Exploration Checkpoint Coverage, a verifiable metric that measures how broadly an agent discovers key states, objects, and affordances. Our systematic evaluation reveals that agents trained with standard task-oriented reinforcement learning consistently exhibit narrow and repetitive behaviors that impede downstream performance. To address this limitation, we develop a training strategy that interleaves task-execution rollouts and exploration rollouts, with each type of rollout optimized by its corresponding verifiable reward. Building on this training strategy, we propose the Explore-then-Act paradigm, which decouples information-gathering from task execution: agents first utilize an interaction budget to acquire grounded environmental knowledge, then leverage it for task resolution. Our results demonstrate that learning to systematically explore is imperative for building generalizable and real-world-ready agents. 

---
# Reasoners or Translators? Contamination-aware Evaluation and Neuro-Symbolic Robustness in Tax Law 

**Authors**: Parisa Kordjamshidi, Samer Aslan, Madhavan Seshadri, Leslie Barrett, Enrico Santus  

**Link**: [PDF](https://arxiv.org/pdf/2605.16052)  

**Abstract**: Recent advances in large language models (LLMs) have significantly enhanced automated legal reasoning. Yet, it remains unclear whether their performance reflects genuine legal reasoning ability or artifacts of data contamination. We present a comprehensive empirical study of tax law reasoning approaches and implement a contamination detection protocol to rigorously assess LLM reliability. We show that performance can be inflated by contamination. Building on this analysis, we conduct a systematic evaluation, comparing monolithic LLMs with hybrid systems that translate statutory text into formal representations and delegate inference to symbolic solvers. We build a novel test suite designed to probe generalization to unseen documents via case and rule variations. Our findings indicate that legal reasoning is inherently compositional and that neuro-symbolic frameworks offer a more reliable and robust foundation for legal AI, as well as improved generalization to unobserved situations. 

---
# SLIP & ETHICS: Graduated Intervention for AI Emotional Companions 

**Authors**: Minseo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2605.15915)  

**Abstract**: AI emotional companions face a safety-rapport paradox: restrictive safeguards can damage supportive alliance, while permissive systems risk user harm. We present SLIP (Staged Layers of Intervention Protocol), a four-stage graduated methodology deriving interventions (none, soft, hard) from structured qualitative indicators -- affect intensity (a) and narrative dynamism (m) -- alongside ETHICS (Emergent Taxonomy for Human-AI Interaction Context Signals), a "signals not labels" taxonomy. An evaluation combining a small-scale production deployment (N=68 entries, 10 users, 10 weeks) with a synthetic persona battery (N=91, 5 behavioral-risk profiles) achieved 0% false positives for the flow persona and showed expected escalation patterns in crisis-oriented personas. However, initial results showed that 8 consecutive days of high-energy elevation produced zero interventions (0/8), exposing a boundary where the "do not pathologize" principle conflicts with safety. A subsequent three-model stress test demonstrated that increased model capability improves detection from 0/8 to 6/8 while preserving 0/10 flow false positives in the largest model. Read as preliminary, these findings position graduated intervention as a design direction for navigating -- not resolving -- the safety-rapport tension in affective computing. 

---
# Are VLMs Seeing or Just Saying? Uncovering the Illusion of Visual Re-examination 

**Authors**: Chufan Shi, Cheng Yang, Yaokang Wu, Linhao Jin, Bo Shui, Taylor Berg-Kirkpatrick, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2605.15864)  

**Abstract**: Vision-Language Models (VLMs) often produce self-reflective statements like "let me check the figure again" during reasoning. Do such statements trigger genuine visual re-examination, or are they merely learned textual patterns? We investigate this via VisualSwap, an image-swap probing framework: after a model reasons over an image, we replace it with a visually similar but semantically different one and test whether the model notices. We introduce VS-Bench, 800 image pairs curated from MathVista, MathVerse, MathVision, and MMMU-Pro. Experiments on Qwen3-VL, Kimi-VL, and ERNIE-VL reveal a striking failure: models overwhelmingly miss the swap, with accuracy dropping by up to 60%. Counterintuitively, thinking models are nearly 3x more vulnerable than their instructed counterparts, and scaling offers no mitigation. Multi-turn user instructions restore visual grounding, but self-generated reflective statements during continuous generation do not. Attention analysis explains why: user instructions substantially elevate attention to visual tokens, whereas self-reflection does not. Current VLMs tend to say rather than actually see when claiming to perform visual re-examination. Our code and dataset are available at the project page: this https URL 

---
# Conversations in Space: Structuring Non-Linear LLM Interactions on a Canvas 

**Authors**: Rifat Mehreen Amin, Alperen Adatepe, Daniela Fernandes, Daniel Buschek, Andreas Butz  

**Link**: [PDF](https://arxiv.org/pdf/2605.15848)  

**Abstract**: Conversational interfaces powered by large language models (LLMs) are widely used for ideation and analysis, yet their linear structure limits exploration of alternatives and management of long-running interactions. We present CanvasConvo, a conversational interface concept that transforms linear chat into a branching conversation tree embedded in a spatial canvas. CanvasConvo enables users to explore what-if scenarios by branching directly from conversational content, supporting parallel development of alternative directions. These branches are visualized on a canvas while remaining integrated with a familiar chat interface, allowing users to switch between linear and non-linear interaction. Features such as timeline-based navigation, automatic tagging and summarization, and context-aware controls (e.g., goals, reusable prompts) support structured interaction and continuity. We evaluated CanvasConvo in a 5-7 day field study with 24 participants. Our findings highlight how non-linear conversational structures support exploratory workflows and different interactions in LLM-based work. 

---
# BootstrapAgent: Distilling Repository Setup into Reusable Agent Knowledge 

**Authors**: Sihan Fu, Oucheng Liu, Shiyuan Wang, Jin Shi, Chengkun Wei  

**Link**: [PDF](https://arxiv.org/pdf/2605.15815)  

**Abstract**: Code agents increasingly help developers work with unfamiliar repositories, but every such task depends on a costly prerequisite: bootstrapping the repository into a usable development state. This process requires substantial trial-and-error exploration, yet the resulting knowledge--resolved dependencies, repair strategies--stays trapped in a single conversation, unavailable to future agents. We therefore formulate repository bootstrapping as a reusable startup knowledge problem and introduce BootstrapAgent, a multi-agent framework that distills the heuristics discovered during bootstrap exploration into a persistent, verifiable, agent-consumable .bootstrap contract. Through evidence extraction, structured planning, deterministic Docker-based verification, and trace-driven repair, BootstrapAgent generates a contract covering environment setup, diagnostic checks, minimal verification, and accumulated repair knowledge. We further propose warm repair with clean replay to accelerate iterative debugging without sacrificing cold-start reproducibility, and a delta repair with sanity check to prevent reward hacking. Experiments on three benchmarks show that BootstrapAgent achieves a 92.9% success rate, outperforming the baseline by over 10% while reducing downstream agent token usage by 25.9% and build time by 22.3%. Our code is available at this https URL. 

---
# Nudging Beyond the Comfort Zone: Efficient Strategy-Guided Exploration for RLVR 

**Authors**: Chanuk Lee, Sangwoo Park, Minki Kang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15726)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has emerged as a scalable paradigm for improving the reasoning capabilities of large language models. However, its effectiveness is fundamentally limited by exploration: the policy can only improve on trajectories it has already sampled. While increasing the number of rollouts alleviates this issue, such brute-force scaling is computationally expensive, and existing approaches that modify the optimization objective provide limited control over what is explored. In this work, we propose NudgeRL, a framework for structured and diversity-driven exploration in RLVR. Our approach introduces Strategy Nudging, which conditions each rollout on lightweight, strategy-level contexts to induce diverse reasoning trajectories without relying on expensive oracle supervision. To effectively learn from such structured exploration, we further propose a unified objective, which decomposes the reward signal into inter- and intra-context components and incorporates a distillation objective to transfer discovered behaviors back to the base policy. Empirically, NudgeRL outperforms standard GRPO with up to 8 times larger rollout budgets, while outperforming oracle-guided RL baseline on average across five challenging math benchmarks. These results demonstrate that structured, context-driven exploration can serve as an efficient and scalable alternative to both brute-force rollout scaling and feasibility-oriented methods based on privileged information. Our code is available at this https URL. 

---
# VSPO: Vector-Steered Policy Optimization for Behavioral Control 

**Authors**: Xuechen Zhang, Zijian Huang, Kai Yang, Weijia Zhang, Jiasi Chen, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2605.15604)  

**Abstract**: Modern language models often need to optimize a primary accuracy objective while also accommodating secondary behavioral preferences, such as verbosity, agreeableness, or the level of technical expertise in its response. In practice, a base model may exhibit a desired behavior very rarely or not at all. Thus, endowing the model with a target behavior creates a sparse behavioral reward bottleneck. To address such multi-objective problems, we introduce Vector-Steered Policy Optimization (VSPO) which employs a steering vector associated with the target behavior to control the behavior intensity of the generated rollouts. VSPO is obtained by modifying GRPO to sample rollouts with varying steering intensities. This process can be interpreted as an on-policy latent self-distillation procedure where the model internalizes its steering vector. By varying steering intensities, VSPO upsamples rare behaviors and enriches rollout diversity, which alleviates the sparse reward issue and provably accelerates the policy optimization. Through comprehensive theory and experiments, we establish that VSPO has favorable properties compared to vanilla reward shaping and other alternative approaches. Specifically, under a bandit abstraction, VSPO provably achieves better iteration complexity than reward-shaped GRPO when the steering-induced distributions are sufficiently aligned with the target behavior. We evaluate VSPO across multiple reasoning benchmarks, including MATH and MMLU-Pro, for four target behaviors: explanation expertise, confidence expression, robustness to misleading context, and response verbosity. Our results show that VSPO consistently improves the control along target behavior while maintaining or improving task accuracy compared with reward shaping, teacher-trace distillation, and guidance-based baselines. 

---
# DeltaPrompts: Escaping the Zero-Delta Trap in Multimodal Distillation 

**Authors**: Jaehun Jung, Hyunwoo Kim, Brandon Cui, Ximing Lu, David Acuna, Prithviraj Ammanabrolu, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15532)  

**Abstract**: Distillation enables compact Vision-Language Models (VLMs) to obtain strong reasoning capabilities, yet the prompts driving this process are typically chosen via simple heuristics or aggregated from off-the-shelf datasets. We reveal a critical inefficiency in this approach: up to 69% of the prompts in standard chart / document reasoning datasets are effectively zero-delta, meaning the teacher and student already induce the exact same answer distribution. Training on these prompts provides minimal learning signal, causing student improvement to rapidly saturate regardless of data scale. To escape the zero-delta trap, we return to first principles: distillation fundamentally minimizes distributional divergence, and thus a prompt is valuable only if it exposes a functional capability gap between the teacher and student. We quantify this gap through answer divergence ($\Delta$), demonstrating that non-zero divergence is critical for effective scaling. Building on this insight, we propose a staged synthesis pipeline that repurposes existing datasets as seeds, actively targeting student failure modes to produce better prompts. The result is DeltaPrompts, a diverse dataset of 200k synthetic, high-divergence reasoning problems. We evaluate DeltaPrompts across three distinct settings: on-policy distillation with the target teacher-student pair, transfer to a novel model family without regenerating the data, and off-policy fine-tuning of a non-reasoning model. Across all scenarios, DeltaPrompts drives substantial gains, yielding up to 15% relative improvement even on top of a highly-optimized reasoning model (e.g., Qwen3-VL-8B-Thinking) -- averaged over 10 benchmarks spanning chart, document and perception-centric reasoning. 

---
# STS: Efficient Sparse Attention with Speculative Token Sparsity 

**Authors**: Ceyu Xu, Jiangnan Yu, Yongji Wu, Yuan Xie  

**Link**: [PDF](https://arxiv.org/pdf/2605.15508)  

**Abstract**: The quadratic complexity of attention imposes severe memory and computational bottlenecks on Large Language Model (LLM) inference. This challenge is particularly acute for emerging agentic applications that require processing multi-million token sequences. We propose STS, a sparse attention mechanism that requires no model retraining. STS leverages the key insight that tokens identified as important by a smaller draft model are highly predictive of important tokens for a larger target model. By integrating into speculative decoding frameworks, STS repurposes the draft model's attention scores to dynamically construct a token-and-head-wise sparsity mask. This mask effectively prunes the expensive attention computation in the target LLM. Our evaluation shows that STS achieves a 2.67x speedup operating at approximately 90% sparsity on representative benchmark NarrativeQA, maintaining negligible accuracy degradation compared to dense attention. STS establishes a new state-of-the-art on the sparsity-accuracy trade-off, outperforming prior techniques by enabling higher sparsity levels for a given accuracy budget. 

---
# GRLO: Towards Generalizable Reinforcement Learning in Open-Ended Environments from Zero 

**Authors**: Shangjian Yin, Yu Fu, Yue Dong, Zhouxing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2605.15464)  

**Abstract**: Post-training has become a crucial step for unlocking the capabilities of large language models, with reinforcement learning (RL) emerging as a critical paradigm. Recent RL-based post-training has increasingly split into two paradigms: reinforcement learning from human feedback (RLHF), which optimizes models using human preference signals in target domains, and reinforcement learning from verifiable rewards (RLVR), which operates in verifier-backed environments. The latter has dominated recent reasoning-oriented post-training because it delivers stronger gains and higher efficiency on domain-specific tasks (e.g., reasoning). However, although in-domain RL training achieves promising performance, it still requires a substantial amount of GPU compute, which remains a major barrier to broad adoption. In this work, we study the generalization ability of RLHF learned from scratch from a small set of interactions in open-ended environments, and investigate whether the conversational abilities it explicitly acquires can implicitly transfer to downstream tasks such as mathematical reasoning and code generation, namely GRLO. Specifically, on Qwen3-4B-Base backbone, GRLO improves the average performance across all domains from 24.1 to 63.1 with only 5K prompts and 22.7 GPU hours, requiring about $46\times$ less data and $68\times$ less compute than a strong in-domain RLVR baseline. The resulting model is even competitive with Qwen's released post-trained models which required a much larger training cost. Notably, a subsequent in-domain RLVR stage brings only selective gains, mainly on harder competition-math benchmarks. We hope GRLO offers a simple and efficient recipe for building broadly capable post-trained models. Our code and data will be available at: \href{this https URL}{this https URL}. 

---
# From Feedback Loops to Policy Updates: Reinforcement Fine-Tuning for LLM-Based Alpha Factor Discovery 

**Authors**: Lingzhe Zhang, Tong Jia, Yunpeng Zhai, Zixuan Xie, Chiming Duan, Minghua He, Philip S. Yu, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2605.15412)  

**Abstract**: Modern quantitative trading increasingly relies on systematic models to extract predictive signals from large-scale financial data, where alpha factor discovery plays a central role in transforming market observations into tradable signals. Recent LLM-based methods have shown promise in automating factor generation, but most of them still rely on prompt-level generation--evaluation--feedback loops for iterative optimization. As the loop becomes longer, repeatedly appended historical candidates and feedback can cause context explosion, increase inference cost, dilute useful information, and introduce feedback drift. Moreover, these methods often depend on very large LLMs whose stable generation preferences may lead to structurally similar expressions, redundant candidates, and search stagnation. To address these limitations, we propose \textsc{QuantEvolver}, a self-evolving alpha factor discovery framework based on reinforcement fine-tuning. Instead of accumulating feedback in the prompt, \textsc{QuantEvolver} converts executable quantitative evaluation into policy updates, enabling a Miner LLM to internalize historical optimization experience through parameter learning. Specifically, \textsc{QuantEvolver} constructs high-quality seed factors, builds diverse seed--time-window training tasks, generates executable Factor DSL expressions, evaluates them through Regime Backtest, and optimizes the Miner LLM with Diversity-Complementarity Reward. During training, high-quality factors are continuously accumulated in a Mined Factor Database, which serves as the final discovered factor library. Extensive experiments on three realistic market benchmarks demonstrate the effectiveness of \textsc{QuantEvolver}, which consistently improves the primary evaluation metric of each task over existing LLM-based alpha factor discovery baselines, produces higher-quality and more complementary factor pools. 

---
# From I/O to Code with Discovery Agent 

**Authors**: Yihong Dong, Jiaru Qian, Haoran Zhang, Peixu Wang, Binhua Li, Zhi Jin, Yongbin Li, Ge Li, Xiaokang Yang, Xue Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2605.15334)  

**Abstract**: The automatic synthesis of a program from any form of specification is regarded as a holy grail of computer science. Fueled by LLMs, NL2Code has achieved tremendous success, yet the fundamentally more challenging task of synthesizing programs from input-output behavior, which we refer to as IO2Code, remains largely unsolved. Whereas NL2Code can exploit the semantic alignment between natural language and code acquired during pretraining, IO2Code requires recovering underlying principles from concrete computational behavior, navigating a vast and underspecified hypothesis space. To address this, we propose DIO-Agent, a discovery agent for IO2Code. Our method frames IO2Code as an evolutionary search over discrete program space, in which an LLM serves as the mutation operator and concrete error signals from execution guide each mutation. To prevent the search from wandering into structurally complex yet incorrect dead ends, we introduce the Transformation Priority Premise as a mutation prior that biases the LLM toward the simplest hypothesis consistent with current evidence, progressively escalating from constants to conditionals to iteration only when simpler constructs are insufficient. To facilitate systematic study, we further construct an IO2CodeBench spanning multiple difficulty levels. Extensive experiments show that DIO-Agent consistently outperforms both traditional program-by-example method and SOTA evolution-agent baselines across all difficulty levels and various LLMs, while substantially surpassing test-time scaling strategies with equivalent sampling budgets. 

---
# Context Pruning for Coding Agents via Multi-Rubric Latent Reasoning 

**Authors**: Jingjing Wang, Xiwen Chen, Wenhui Zhu, Huayu Li, Zhengxiao He, Feiyang Cai, Ana S. Carreon-Rascon, Xuanzhao Dong, Feng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2605.15315)  

**Abstract**: LLM-powered coding agents spend the majority of their token budget reading repository files, yet much of the retrieved code is irrelevant to the task at hand. Existing learned pruners compress this context with a single-objective sequence labeler, collapsing all facets of code relevance into one score and one transition matrix. We show that this formulation creates a modeling bottleneck: a single CRF transition prior must serve heterogeneous retention patterns, including contiguous semantic spans and sparse structural support lines. We propose LaMR (Latent Multi-Rubric), a structured pruning framework that decomposes code relevance into two interpretable quality dimensions, semantic evidence and dependency support, each modeled by a dedicated CRF with dimension-specific transition dynamics. A mixture-of-experts gating network dynamically weights the per-rubric emissions conditioned on the query, and a final CRF layer on the fused emissions produces the aggregate keep-or-prune decision. To supervise each dimension without additional annotation cost, we derive multi-rubric labels from the existing training corpus via AST-based program analysis, simultaneously denoising the teacher's binary labels. By effectively filtering distracting noise, LaMR frequently matches or even outperforms unpruned full-context baselines. Experiments on four benchmarks (SWE-Bench Verified, SWE-QA, LCC, LongCodeQA) show that LaMR wins 12 of 16 head-to-head multi-turn comparisons. It saves up to 31% more tokens on multi-turn agent tasks and improves Exact Match by up to +3.5 on single-turn tasks, while performance is frequently enhanced by denoising the context, and any remaining drops are marginal. 

---
# PhysBrain 1.0 Technical Report 

**Authors**: Shijie Lian, Bin Yu, Xiaopeng Lin, Changti Wu, Hang Yuan, Xiaolin Hu, Zhaolong Shen, Yuzhuo Miao, Haishan Liu, Yuxuan Tian, Yukun Shi, Cong Huang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2605.15298)  

**Abstract**: Vision-language-action models have advanced rapidly, but robot trajectories alone provide limited coverage for learning broad physical understanding. PhysBrain 1.0 studies a complementary route: converting large-scale human egocentric video into structured physical commonsense supervision before robot adaptation. Our data engine extracts scene elements, spatial dynamics, action execution, and depth-aware relations, then turns them into question-answer supervision for training PhysBrain VLMs. The resulting physical priors are further transferred to VLA policies through a capability-preserving and language-sensitive adaptation design. Across multimodal QA benchmarks and embodied control benchmarks, including ERQA, PhysBench, SimplerEnv-WidowX, LIBERO, and RoboCasa, PhysBrain 1.0 achieves SOTA results and shows especially strong out-of-domain performance on SimplerEnv. These results suggest that scaling physical commonsense from human interaction video can provide an effective bridge from multimodal understanding to robot action. 

---
# PerfCodeBench: Benchmarking LLMs for System-Level High-Performance Code Optimization 

**Authors**: Huihao Jing, Wenbin Hu, Haochen Shi, Hanyu Yang, Sirui Zhang, Shaojin Chen, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2605.15222)  

**Abstract**: Large language models (LLMs) can often generate functionally correct code, but their ability to produce efficient implementations for performance-critical systems tasks remains limited. Existing code benchmarks mainly emphasize correctness or algorithmic problem solving, while realistic systems-level optimization is still underexplored. To address this gap, we introduce PerfCodeBench, an executable benchmark for evaluating LLMs on high-performance code optimization. The tasks require system-level implementation choices, hardware-aware optimization, and careful handling of performance bottlenecks. Each task includes executable correctness checks, a baseline implementation, and a reference optimized solution. This allows us to evaluate both correctness and runtime-oriented efficiency. Our evaluation on a broad set of state-of-the-art LLMs shows a clear gap between model-generated code and expert-optimized implementations. The gap is especially large on tasks involving parallelism and GPU operations. Current models also show weaknesses in cross-language robustness and in consistently reaching expert-level efficiency. These results suggest that performance-aware evaluation are still needed. LLMs should move beyond generating merely correct code toward producing efficient systems software. We submit the benchmark data, evaluation infrastructure, and complete logs of all LLMs-generated code at this https URL. 

---
# Effective Harness Engineering for Algorithm Discovery with Coding Agents 

**Authors**: Yoichi Ishibashi, Taro Yano, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2605.15221)  

**Abstract**: AlphaEvolve and FunSearch have demonstrated the potential of combining large language models (LLMs) with evolutionary search for automated algorithm discovery. However, discovery success is shaped not only by model capability but also significantly by the design of the execution infrastructure, i.e., the harness. This paper investigates effective harness design through three questions: under a fixed token budget, is it better to produce many algorithms with brief thought or fewer algorithms with deeper thought? How should the harness handle evaluation hacks, where generated programs exploit the scoring function? And how can agents that require full filesystem access execute safely in parallel? Using Vesper, an algorithm discovery framework that incorporates harness improvements addressing these questions, we evaluate on Circle Packing under the same token budget. Interestingly, generating fewer algorithms while thinking more deeply about each one achieved higher scores. That is, scaling the quality of each individual is more budget-efficient than scaling the number of evolutionary generations. Surprisingly, more capable models produced evaluation hacks at higher rates, making hack detection increasingly necessary as models scale. 

---
# DeepSlide: From Artifacts to Presentation Delivery 

**Authors**: Ming Yang, Zhiwei Zhang, Jiahang Li, Haoseng Liu, Yuzheng Cai, Weiguo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2605.15202)  

**Abstract**: Presentations are a primary medium for scholarly communication, yet most AI slide generators optimize the artifact (a visually plausible deck) while under-optimizing the delivery process (pacing, narrative, and presentation preparation). We present DeepSlide, a human-in-the-loop multi-agent system that supports preparing the full presentation process, from requirement elicitation and time-budgeted narrative planning, to evidence-grounded slide--script generation, attention augmentation, and rehearsal support. DeepSlide integrates (i) a controllable logical-chain planner with per-node time budgets, (ii) a lightweight content-tree retriever for grounding, (iii) Markov-style sequential rendering with style inheritance, and (iv) sandboxed execution with minimal repair to ensure renderability. We further introduce a dual-scoreboard benchmark that cleanly separates static artifact quality from dynamic delivery excellence. Across 20 domains and diverse audience profiles, DeepSlide matches strong baselines on artifact quality while consistently achieving larger gains on delivery metrics, improving narrative flow, pacing precision, and slide--script synergy with clearer attention guidance. 

---
# MultiMat: Multimodal Program Synthesis for Procedural Materials using Large Multimodal Models 

**Authors**: Jonas Belouadi, Tamy Boubekeur, Adrien Kaiser  

**Link**: [PDF](https://arxiv.org/pdf/2509.22151)  

**Abstract**: Material node graphs are programs that generate the 2D channels of procedural materials, including geometry such as roughness and displacement maps, and reflectance such as albedo and conductivity maps. They are essential in computer graphics for representing the appearance of virtual 3D objects parametrically and at arbitrary resolution. In particular, their directed acyclic graph structure and intermediate states enable a modular, interpretable workflow for interactive appearance modeling. However, creating such graphs remains challenging and typically requires professional training. While recent neural program synthesis approaches attempt to simplify this process, they solely represent graphs as textual programs, failing to capture the inherently visual-spatial nature of node graphs that makes them accessible to humans. To address this gap, we present MultiMat, a multimodal program synthesis framework that leverages large multimodal models to process both visual and textual graph representations for improved generation of procedural material graphs. We train our models on a new dataset of production-quality procedural materials and combine them with a constrained tree search inference algorithm that ensures static correctness while efficiently navigating the program space. Our experimental results show that our multimodal program synthesis method is more efficient in both unconditional and conditional graph synthesis with higher visual quality and fidelity than text-only baselines, establishing new state-of-the-art performance. 

---
