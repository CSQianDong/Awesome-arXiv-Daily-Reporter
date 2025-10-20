# PolySkill: Learning Generalizable Skills Through Polymorphic Abstraction 

**Authors**: Simon Yu, Gang Li, Weiyan Shi, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.15863)  

**Abstract**: Large language models (LLMs) are moving beyond static uses and are now powering agents that learn continually during their interaction with external environments. For example, agents can learn reusable skills while navigating web pages or toggling new tools. However, existing methods for skill learning often create skills that are over-specialized to a single website and fail to generalize. We introduce PolySkill, a new framework that enables agents to learn generalizable and compositional skills. The core idea, inspired by polymorphism in software engineering, is to decouple a skill's abstract goal (what it accomplishes) and its concrete implementation (how it is executed). Experiments show that our method (1) improves skill reuse by 1.7x on seen websites and (2) boosts success rates by up to 9.4% on Mind2Web and 13.9% on unseen websites, while reducing steps by over 20%. (3) In self-exploration settings without specified tasks, our framework improves the quality of proposed tasks and enables agents to learn generalizable skills that work across different sites. By enabling the agent to identify and refine its own goals, the PolySkill enhances the agent's ability to learn a better curriculum, leading to the acquisition of more generalizable skills compared to baseline methods. This work provides a practical path toward building agents capable of continual learning in adaptive environments. Our findings show that separating a skill's goal from its execution is a crucial step toward developing autonomous agents that can learn and generalize across the open web continuously. 

---
# InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training 

**Authors**: Pengkai Wang, Qi Zuo, Pengwei Liu, Zhijie Sang, Congkai Xie, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15859)  

**Abstract**: Large Language Models (LLMs) have shown substantial advances through reinforcement learning (RL), particularly in domains where rewards can be programmatically verified, such as mathematics and code. In these areas, models benefit from a well-defined operational base guided by explicit rule-based objectives. However, this progress reveals a significant limitation: in open-ended domains where rewards are ambiguous, subjective, or context-dependent, such as creative writing, scientific reasoning, and notably medical consultation, robust reward functions are lacking, making these areas challenging for current RL strategies. To bridge this gap, we introduce ORBIT, an open-ended rubric-based incremental training framework specifically designed for high-stakes medical dialogue. ORBIT integrates syn- thetic dialogue generation with the dynamic creation of rubrics, employing these rubrics to direct an incremental RL process. In particular, this approach does not depend on external medical knowledge or manual rules, instead utilizing rubric-guided feedback to shape learning. When implemented on the Qwen3-4B-Instruct model, our method can greatly enhance its performance on the HealthBench-Hard benchmark from 7.0 to 27.2 using only 2k samples, thus achieving state-of-the-art results for models of this scale. Our analysis confirms that rubric-driven RL fos-ters consistent performance gains across diverse consultation scenarios, going beyond simple numerical improvements. These findings underscore rubric-based feedback as a scalable strategy for advancing LLMs in intricate, open-ended tasks. 

---
# SpeechLLMs for Large-scale Contextualized Zero-shot Slot Filling 

**Authors**: Kadri Hacioglu, Manjunath K E, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2510.15851)  

**Abstract**: Slot filling is a crucial subtask in spoken language understanding (SLU), traditionally implemented as a cascade of speech recognition followed by one or more natural language understanding (NLU) components. The recent advent of speech-based large language models (speechLLMs), which integrate speech and textual foundation models, has opened new avenues for achieving speech understanding tasks in a more unified, generative, and instruction-following manner while promising data and compute efficiency with zero-shot abilities, generalizing to unseen slot labels. We address the slot-filling task by creating an empirical upper bound for the task, identifying performance, robustness, and generalization gaps, and proposing improvements to the training data, architecture, and training strategies to narrow the gap with the upper bound result. We show that each of these measures improve performance substantially, while highlighting practical challenges and providing empirical guidance and insights for harnessing these emerging models. 

---
# Enhanced Sentiment Interpretation via a Lexicon-Fuzzy-Transformer Framework 

**Authors**: Shayan Rokhva, Mousa Alizadeh, Maryam Abdollahi Shamami  

**Link**: [PDF](https://arxiv.org/pdf/2510.15843)  

**Abstract**: Accurately detecting sentiment polarity and intensity in product reviews and social media posts remains challenging due to informal and domain-specific language. To address this, we propose a novel hybrid lexicon-fuzzy-transformer framework that combines rule-based heuristics, contextual deep learning, and fuzzy logic to generate continuous sentiment scores reflecting both polarity and strength. The pipeline begins with VADER-based initial sentiment estimations, which are refined through a two-stage adjustment process. This involves leveraging confidence scores from DistilBERT, a lightweight transformer and applying fuzzy logic principles to mitigate excessive neutrality bias and enhance granularity. A custom fuzzy inference system then maps the refined scores onto a 0 to 1 continuum, producing expert)like judgments. The framework is rigorously evaluated on four domain-specific datasets. food delivery, e-commerce, tourism, and fashion. Results show improved alignment with user ratings, better identification of sentiment extremes, and reduced misclassifications. Both quantitative metrics (distributional alignment, confusion matrices) and qualitative insights (case studies, runtime analysis) affirm the models robustness and efficiency. This work demonstrates the value of integrating symbolic reasoning with neural models for interpretable, finegrained sentiment analysis in linguistically dynamic domains. 

---
# Paper2Web: Let's Make Your Paper Alive! 

**Authors**: Yuhang Chen, Tianpeng Lv, Siyi Zhang, Yixiang Yin, Yao Wan, Philip S. Yu, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15842)  

**Abstract**: Academic project websites can more effectively disseminate research when they clearly present core content and enable intuitive navigation and interaction. However, current approaches such as direct Large Language Model (LLM) generation, templates, or direct HTML conversion struggle to produce layout-aware, interactive sites, and a comprehensive evaluation suite for this task has been lacking. In this paper, we introduce Paper2Web, a benchmark dataset and multi-dimensional evaluation framework for assessing academic webpage generation. It incorporates rule-based metrics like Connectivity, Completeness and human-verified LLM-as-a-Judge (covering interactivity, aesthetics, and informativeness), and PaperQuiz, which measures paper-level knowledge retention. We further present PWAgent, an autonomous pipeline that converts scientific papers into interactive and multimedia-rich academic homepages. The agent iteratively refines both content and layout through MCP tools that enhance emphasis, balance, and presentation quality. Our experiments show that PWAgent consistently outperforms end-to-end baselines like template-based webpages and arXiv/alphaXiv versions by a large margin while maintaining low cost, achieving the Pareto-front in academic webpage generation. 

---
# Emergence of Linear Truth Encodings in Language Models 

**Authors**: Shauli Ravfogel, Gilad Yehudai, Tal Linzen, Joan Bruna, Alberto Bietti  

**Link**: [PDF](https://arxiv.org/pdf/2510.15804)  

**Abstract**: Recent probing studies reveal that large language models exhibit linear subspaces that separate true from false statements, yet the mechanism behind their emergence is unclear. We introduce a transparent, one-layer transformer toy model that reproduces such truth subspaces end-to-end and exposes one concrete route by which they can arise. We study one simple setting in which truth encoding can emerge: a data distribution where factual statements co-occur with other factual statements (and vice-versa), encouraging the model to learn this distinction in order to lower the LM loss on future tokens. We corroborate this pattern with experiments in pretrained language models. Finally, in the toy setting we observe a two-phase learning dynamic: networks first memorize individual factual associations in a few steps, then -- over a longer horizon -- learn to linearly separate true from false, which in turn lowers language-modeling loss. Together, these results provide both a mechanistic demonstration and an empirical motivation for how and why linear truth representations can emerge in language models. 

---
# On Non-interactive Evaluation of Animal Communication Translators 

**Authors**: Orr Paradise, David F. Gruber, Adam Tauman Kalai  

**Link**: [PDF](https://arxiv.org/pdf/2510.15768)  

**Abstract**: If you had an AI Whale-to-English translator, how could you validate whether or not it is working? Does one need to interact with the animals or rely on grounded observations such as temperature? We provide theoretical and proof-of-concept experimental evidence suggesting that interaction and even observations may not be necessary for sufficiently complex languages. One may be able to evaluate translators solely by their English outputs, offering potential advantages in terms of safety, ethics, and cost. This is an instance of machine translation quality evaluation (MTQE) without any reference translations available. A key challenge is identifying ``hallucinations,'' false translations which may appear fluent and plausible. We propose using segment-by-segment translation together with the classic NLP shuffle test to evaluate translators. The idea is to translate animal communication, turn by turn, and evaluate how often the resulting translations make more sense in order than permuted. Proof-of-concept experiments on data-scarce human languages and constructed languages demonstrate the potential utility of this evaluation methodology. These human-language experiments serve solely to validate our reference-free metric under data scarcity. It is found to correlate highly with a standard evaluation based on reference translations, which are available in our experiments. We also perform a theoretical analysis suggesting that interaction may not be necessary nor efficient in the early stages of learning to translate. 

---
# LLMs Judge Themselves: A Game-Theoretic Framework for Human-Aligned Evaluation 

**Authors**: Gao Yang, Yuhang Liu, Siyu Miao, Xinyue Liang, Zhengyang Liu, Heyan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15746)  

**Abstract**: Ideal or real - that is the this http URL this work, we explore whether principles from game theory can be effectively applied to the evaluation of large language models (LLMs). This inquiry is motivated by the growing inadequacy of conventional evaluation practices, which often rely on fixed-format tasks with reference answers and struggle to capture the nuanced, subjective, and open-ended nature of modern LLM behavior. To address these challenges, we propose a novel alternative: automatic mutual evaluation, where LLMs assess each other's output through self-play and peer review. These peer assessments are then systematically compared with human voting behavior to evaluate their alignment with human judgment. Our framework incorporates game-theoretic voting algorithms to aggregate peer reviews, enabling a principled investigation into whether model-generated rankings reflect human preferences. Empirical results reveal both convergences and divergences between theoretical predictions and human evaluations, offering valuable insights into the promises and limitations of mutual evaluation. To the best of our knowledge, this is the first work to jointly integrate mutual evaluation, game-theoretic aggregation, and human-grounded validation for evaluating the capabilities of LLMs. 

---
# Attention Sinks in Diffusion Language Models 

**Authors**: Maximo Eduardo Rulli, Simone Petruzzi, Edoardo Michielon, Fabrizio Silvestri, Simone Scardapane, Alessio Devoto  

**Link**: [PDF](https://arxiv.org/pdf/2510.15731)  

**Abstract**: Masked Diffusion Language Models (DLMs) have recently emerged as a promising alternative to traditional Autoregressive Models (ARMs). DLMs employ transformer encoders with bidirectional attention, enabling parallel token generation while maintaining competitive performance. Although their efficiency and effectiveness have been extensively studied, the internal mechanisms that govern DLMs remain largely unexplored. In this work, we conduct an empirical analysis of DLM attention patterns, focusing on the attention sinking phenomenon, an effect previously observed in various transformer-based architectures. Our findings reveal that DLMs also exhibit attention sinks, but with distinct characteristics. First, unlike in ARMs, the sink positions in DLMs tend to shift throughout the generation process, displaying a dynamic behaviour. Second, while ARMs are highly sensitive to the removal of attention sinks, DLMs remain robust: masking sinks leads to only a minor degradation in performance. These results provide new insights into the inner workings of diffusion-based language models and highlight fundamental differences in how they allocate and utilize attention compared to autoregressive models. 

---
# Cost-Aware Retrieval-Augmentation Reasoning Models with Adaptive Retrieval Depth 

**Authors**: Helia Hashemi, Victor Rühle, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.15719)  

**Abstract**: Reasoning models have gained significant attention due to their strong performance, particularly when enhanced with retrieval augmentation. However, these models often incur high computational costs, as both retrieval and reasoning tokens contribute substantially to the overall resource usage. In this work, we make the following contributions: (1) we propose a retrieval-augmented reasoning model that dynamically adjusts the length of the retrieved document list based on the query and retrieval results; (2) we develop a cost-aware advantage function for training of efficient retrieval-augmented reasoning models through reinforcement learning; and (3) we explore both memory- and latency-bound implementations of the proposed cost-aware framework for both proximal and group relative policy optimization algorithms. We evaluate our approach on seven public question answering datasets and demonstrate significant efficiency gains, without compromising effectiveness. In fact, we observed that the model latency decreases by ~16-20% across datasets, while its effectiveness increases by ~5% on average, in terms of exact match. 

---
# Leveraging LLMs for Context-Aware Implicit Textual and Multimodal Hate Speech Detection 

**Authors**: Joshua Wolfe Brook, Ilia Markov  

**Link**: [PDF](https://arxiv.org/pdf/2510.15685)  

**Abstract**: This research introduces a novel approach to textual and multimodal Hate Speech Detection (HSD), using Large Language Models (LLMs) as dynamic knowledge bases to generate background context and incorporate it into the input of HSD classifiers. Two context generation strategies are examined: one focused on named entities and the other on full-text prompting. Four methods of incorporating context into the classifier input are compared: text concatenation, embedding concatenation, a hierarchical transformer-based fusion, and LLM-driven text enhancement. Experiments are conducted on the textual Latent Hatred dataset of implicit hate speech and applied in a multimodal setting on the MAMI dataset of misogynous memes. Results suggest that both the contextual information and the method by which it is incorporated are key, with gains of up to 3 and 6 F1 points on textual and multimodal setups respectively, from a zero-context baseline to the highest-performing system, based on embedding concatenation. 

---
# HypoSpace: Evaluating LLM Creativity as Set-Valued Hypothesis Generators under Underdetermination 

**Authors**: Tingting Chen, Beibei Lin, Zifeng Yuan, Qiran Zou, Hongyu He, Yew-Soon Ong, Anirudh Goyal, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.15614)  

**Abstract**: As language models are increasingly used in scientific workflows, evaluating their ability to propose sets of explanations-not just a single correct answer-becomes critical. Many scientific problems are underdetermined: multiple, mechanistically distinct hypotheses are consistent with the same observations. We introduce HypoSpace, a diagnostic suite that treats LLMs as samplers of finite hypothesis sets and measures three complementary indicators: Validity (precision of proposals consistent with observations), Uniqueness (non-redundancy among proposals), and Recovery (coverage of the enumerated admissible set). We instantiate HypoSpace in three structured domains with deterministic validators and exactly enumerated hypothesis spaces: (i) causal graphs from perturbations, (ii) gravity-constrained 3D voxel reconstruction from top-down projections, and (iii) Boolean genetic interactions. Across instruction-tuned and reasoning-focused models, Validity often remains high while Uniqueness and Recovery degrade as the admissible space grows, revealing mode collapse that is invisible to correctness-only metrics. HypoSpace offers a controlled probe-rather than a leaderboard-for methods that explicitly explore and cover admissible explanation spaces. Code is available at: this https URL. 

---
# The Elephant in the Coreference Room: Resolving Coreference in Full-Length French Fiction Works 

**Authors**: Antoine Bourgois, Thierry Poibeau  

**Link**: [PDF](https://arxiv.org/pdf/2510.15594)  

**Abstract**: While coreference resolution is attracting more interest than ever from computational literature researchers, representative datasets of fully annotated long documents remain surprisingly scarce. In this paper, we introduce a new annotated corpus of three full-length French novels, totaling over 285,000 tokens. Unlike previous datasets focused on shorter texts, our corpus addresses the challenges posed by long, complex literary works, enabling evaluation of coreference models in the context of long reference chains. We present a modular coreference resolution pipeline that allows for fine-grained error analysis. We show that our approach is competitive and scales effectively to long documents. Finally, we demonstrate its usefulness to infer the gender of fictional characters, showcasing its relevance for both literary analysis and downstream NLP tasks. 

---
# BiMax: Bidirectional MaxSim Score for Document-Level Alignment 

**Authors**: Xiaotian Wang, Takehito Utsuro, Masaaki Nagata  

**Link**: [PDF](https://arxiv.org/pdf/2510.15577)  

**Abstract**: Document alignment is necessary for the hierarchical mining (Bañón et al., 2020; Morishita et al., 2022), which aligns documents across source and target languages within the same web domain. Several high precision sentence embedding-based methods have been developed, such as TK-PERT (Thompson and Koehn, 2020) and Optimal Transport (OT) (Clark et al., 2019; El-Kishky and Guzmán, 2020). However, given the massive scale of web mining data, both accuracy and speed must be considered. In this paper, we propose a cross-lingual Bidirectional Maxsim score (BiMax) for computing doc-to-doc similarity, to improve efficiency compared to the OT method. Consequently, on the WMT16 bilingual document alignment task, BiMax attains accuracy comparable to OT with an approximate 100-fold speed increase. Meanwhile, we also conduct a comprehensive analysis to investigate the performance of current state-of-the-art multilingual sentence embedding models. All the alignment methods in this paper are publicly available as a tool called EmbDA (this https URL). 

---
# From Ghazals to Sonnets: Decoding the Polysemous Expressions of Love Across Languages 

**Authors**: Syed Mohammad Sualeh Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.15569)  

**Abstract**: This paper delves into the intricate world of Urdu poetry, exploring its thematic depths through a lens of polysemy. By focusing on the nuanced differences between three seemingly synonymous words (pyaar, muhabbat, and ishq) we expose a spectrum of emotions and experiences unique to the Urdu language. This study employs a polysemic case study approach, meticulously examining how these words are interwoven within the rich tapestry of Urdu poetry. By analyzing their usage and context, we uncover a hidden layer of meaning, revealing subtle distinctions which lack direct equivalents in English literature. Furthermore, we embark on a comparative analysis, generating word embeddings for both Urdu and English terms related to love. This enables us to quantify and visualize the semantic space occupied by these words, providing valuable insights into the cultural and linguistic nuances of expressing love. Through this multifaceted approach, our study sheds light on the captivating complexities of Urdu poetry, offering a deeper understanding and appreciation for its unique portrayal of love and its myriad expressions 

---
# Finetuning LLMs for EvaCun 2025 token prediction shared task 

**Authors**: Josef Jon, Ondřej Bojar  

**Link**: [PDF](https://arxiv.org/pdf/2510.15561)  

**Abstract**: In this paper, we present our submission for the token prediction task of EvaCun 2025. Our sys-tems are based on LLMs (Command-R, Mistral, and Aya Expanse) fine-tuned on the task data provided by the organizers. As we only pos-sess a very superficial knowledge of the subject field and the languages of the task, we simply used the training data without any task-specific adjustments, preprocessing, or filtering. We compare 3 different approaches (based on 3 different prompts) of obtaining the predictions, and we evaluate them on a held-out part of the data. 

---
# KITE: A Benchmark for Evaluating Korean Instruction-Following Abilities in Large Language Models 

**Authors**: Dongjun Kim, Chanhee Park, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.15558)  

**Abstract**: The instruction-following capabilities of large language models (LLMs) are pivotal for numerous applications, from conversational agents to complex reasoning systems. However, current evaluations predominantly focus on English models, neglecting the linguistic and cultural nuances of other languages. Specifically, Korean, with its distinct syntax, rich morphological features, honorific system, and dual numbering systems, lacks a dedicated benchmark for assessing open-ended instruction-following capabilities. To address this gap, we introduce the Korean Instruction-following Task Evaluation (KITE), a comprehensive benchmark designed to evaluate both general and Korean-specific instructions. Unlike existing Korean benchmarks that focus mainly on factual knowledge or multiple-choice testing, KITE directly targets diverse, open-ended instruction-following tasks. Our evaluation pipeline combines automated metrics with human assessments, revealing performance disparities across models and providing deeper insights into their strengths and weaknesses. By publicly releasing the KITE dataset and code, we aim to foster further research on culturally and linguistically inclusive LLM development and inspire similar endeavors for other underrepresented languages. 

---
# Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation 

**Authors**: Jinliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.15552)  

**Abstract**: Large language models (LLMs) excel at language understanding but often hallucinate and struggle with multi-hop reasoning. Knowledge-graph-based retrieval-augmented generation (KG-RAG) offers grounding, yet most methods rely on flat embeddings and noisy path exploration. We propose ParallaxRAG, a framework that symmetrically decouples queries and graph triples into multi-view spaces, enabling a robust retrieval architecture that explicitly enforces head diversity while constraining weakly related paths. Central to our approach is the observation that different attention heads specialize in semantic relations at distinct reasoning stages, contributing to different hops of the reasoning chain. This specialization allows ParallaxRAG to construct cleaner subgraphs and guide LLMs through grounded, step-wise reasoning. Experiments on WebQSP and CWQ, under our unified, reproducible setup (BGE-M3 + Llama3.1-8B), demonstrate competitive retrieval and QA performance, alongside reduced hallucination and good generalization. Our results highlight multi-view head specialization as a principled direction for knowledge-grounded multi-hop reasoning. Our implementation will be released as soon as the paper is accepted. 

---
# Rethinking Cross-lingual Gaps from a Statistical Viewpoint 

**Authors**: Vihari Piratla, Purvam Jain, Darshan Singh, Partha Talukdar, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2510.15551)  

**Abstract**: Any piece of knowledge is usually expressed in one or a handful of natural languages on the web or in any large corpus. Large Language Models (LLMs) act as a bridge by acquiring knowledge from a source language and making it accessible when queried from target languages. Prior research has pointed to a cross-lingual gap, viz., a drop in accuracy when the knowledge is queried in a target language compared to when the query is in the source language. Existing research has rationalized divergence in latent representations in source and target languages as the source of cross-lingual gap. In this work, we take an alternative view and hypothesize that the variance of responses in the target language is the main cause of this gap. For the first time, we formalize the cross-lingual gap in terms of bias-variance decomposition. We present extensive experimental evidence which support proposed formulation and hypothesis. We then reinforce our hypothesis through multiple inference-time interventions that control the variance and reduce the cross-lingual gap. We demonstrate a simple prompt instruction to reduce the response variance, which improved target accuracy by 20-25% across different models. 

---
# TokenTiming: A Dynamic Alignment Method for Universal Speculative Decoding Model Pairs 

**Authors**: Sibo Xiao, Jinyuan Fu, Zhongle Xie, Lidan Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.15545)  

**Abstract**: Accelerating the inference of large language models (LLMs) has been a critical challenge in generative AI. Speculative decoding (SD) substantially improves LLM inference efficiency. However, its utility is limited by a fundamental constraint: the draft and target models must share the same vocabulary, thus limiting the herd of available draft models and often necessitating the training of a new model from scratch. Inspired by Dynamic Time Warping (DTW), a classic algorithm for aligning time series, we propose the algorithm TokenTiming for universal speculative decoding. It operates by re-encoding the draft token sequence to get a new target token sequence, and then uses DTW to build a mapping to transfer the probability distributions for speculative sampling. Benefiting from this, our method accommodates mismatched vocabularies and works with any off-the-shelf models without retraining and modification. We conduct comprehensive experiments on various tasks, demonstrating 1.57x speedup. This work enables a universal approach for draft model selection, making SD a more versatile and practical tool for LLM acceleration. 

---
# MCA: Modality Composition Awareness for Robust Composed Multimodal Retrieval 

**Authors**: Qiyu Wu, Shuyang Cui, Satoshi Hayakawa, Wei-Yao Wang, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.15543)  

**Abstract**: Multimodal retrieval, which seeks to retrieve relevant content across modalities such as text or image, supports applications from AI search to contents production. Despite the success of separate-encoder approaches like CLIP align modality-specific embeddings with contrastive learning, recent multimodal large language models (MLLMs) enable a unified encoder that directly processes composed inputs. While flexible and advanced, we identify that unified encoders trained with conventional contrastive learning are prone to learn modality shortcut, leading to poor robustness under distribution shifts. We propose a modality composition awareness framework to mitigate this issue. Concretely, a preference loss enforces multimodal embeddings to outperform their unimodal counterparts, while a composition regularization objective aligns multimodal embeddings with prototypes composed from its unimodal parts. These objectives explicitly model structural relationships between the composed representation and its unimodal counterparts. Experiments on various benchmarks show gains in out-of-distribution retrieval, highlighting modality composition awareness as a effective principle for robust composed multimodal retrieval when utilizing MLLMs as the unified encoder. 

---
# Latent Reasoning in LLMs as a Vocabulary-Space Superposition 

**Authors**: Jingcheng Deng, Liang Pang, Zihao Wei, Shichen Xu, Zenghao Duan, Kun Xu, Yang Song, Huawei Shen, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.15522)  

**Abstract**: Large language models (LLMs) demonstrate strong reasoning abilities with chain-of-thought prompting, but explicit reasoning introduces substantial computational overhead. Recent work on latent reasoning reduces this cost by reasoning in latent space without explicit supervision, but performance drops significantly. Our preliminary experiments suggest that this degradation stems from the unstructured latent space, which makes fitting latent tokens difficult. To address this, we restrict the latent space to the column space of the LLM vocabulary, treating latent reasoning as a superposition over vocabulary probabilities. Once latent reasoning concludes, it collapses into an eigenstate of explicit reasoning to yield the final answer. Based on this idea, we propose Latent-SFT, a two-stage learning framework. In the first stage, we design two specialized attention masks to guide the Latent Token Encoder in generating latent tokens, allowing the LLM to produce the correct answer conditioned on them. In the second stage, the Latent Token Encoder is discarded, and the LLM is directly trained to generate these latent tokens autonomously for latent reasoning, optimized with KL and CE losses. Latent-SFT sets a new state of the art on GSM8k, matching explicit SFT performance while cutting reasoning chains by up to 4 times and outperforming prior latent methods. On Math500 and AIME24, lexical probability-based latent reasoning also clearly surpasses hidden-state-based approaches. Our metrics of effective compression rate and effective global parallelism further show that latent reasoning is both the compression of a single path and the superposition of multiple paths. 

---
# From Characters to Tokens: Dynamic Grouping with Hierarchical BPE 

**Authors**: Rares Dolga, Lucas Maystre, Tudor Berariu, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2510.15517)  

**Abstract**: Subword tokenization methods like Byte Pair Encoding (BPE) are widely used in large language models due to their balance of vocabulary compactness and representational power. However, they suffer from inefficiencies in representing rare words and require large embedding matrices. Character-level models address these issues but introduce performance bottlenecks, particularly in Transformer-based architectures. Recent hierarchical models attempt to merge the benefits of both paradigms by grouping characters into patches, but existing patching strategies either rely on whitespace-limiting applicability to certain languages, or require auxiliary models that introduce new dependencies. In this paper, we propose a dynamic character grouping method that leverages the structure of existing BPE tokenization without requiring additional models. By appending explicit end-of-patch markers to BPE tokens and introducing a second-level BPE compression stage to control patch granularity, our method offers efficient, flexible, and language-agnostic representations. Empirical results demonstrate that our approach matches or exceeds the performance of dynamic entropy- and whitespace-based patching strategies, while maintaining a compact vocabulary. 

---
# Temporal Referential Consistency: Do LLMs Favor Sequences Over Absolute Time References? 

**Authors**: Ashutosh Bajpai, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2510.15513)  

**Abstract**: The increasing acceptance of large language models (LLMs) as an alternative to knowledge sources marks a significant paradigm shift across various domains, including time-sensitive fields such as law, healthcare, and finance. To fulfill this expanded role, LLMs must not only be factually accurate but also demonstrate consistency across temporal dimensions, necessitating robust temporal reasoning capabilities. Despite this critical requirement, efforts to ensure temporal consistency in LLMs remain scarce including noticeable absence of endeavors aimed at evaluating or augmenting LLMs across temporal references in time-sensitive inquiries. In this paper, we seek to address this gap by introducing a novel benchmark entitled temporal referential consistency, accompanied by a resource TEMP-ReCon designed to benchmark a wide range of both open-source and closed-source LLMs with various linguistic contexts characterized by differing resource richness (including English, French, and Romanian). The findings emphasis that LLMs do exhibit insufficient temporal referent consistency. To address this, we propose \newmodel, a reasoning path alignment-based model that aims to enhance the temporal referential consistency of LLMs. Our empirical experiments substantiate the efficacy of UnTRaP compared to several baseline models. 

---
# DeceptionBench: A Comprehensive Benchmark for AI Deception Behaviors in Real-world Scenarios 

**Authors**: Yao Huang, Yitong Sun, Yichi Zhang, Ruochen Zhang, Yinpeng Dong, Xingxing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.15501)  

**Abstract**: Despite the remarkable advances of Large Language Models (LLMs) across diverse cognitive tasks, the rapid enhancement of these capabilities also introduces emergent deceptive behaviors that may induce severe risks in high-stakes deployments. More critically, the characterization of deception across realistic real-world scenarios remains underexplored. To bridge this gap, we establish DeceptionBench, the first benchmark that systematically evaluates how deceptive tendencies manifest across different societal domains, what their intrinsic behavioral patterns are, and how extrinsic factors affect them. Specifically, on the static count, the benchmark encompasses 150 meticulously designed scenarios in five domains, i.e., Economy, Healthcare, Education, Social Interaction, and Entertainment, with over 1,000 samples, providing sufficient empirical foundations for deception analysis. On the intrinsic dimension, we explore whether models exhibit self-interested egoistic tendencies or sycophantic behaviors that prioritize user appeasement. On the extrinsic dimension, we investigate how contextual factors modulate deceptive outputs under neutral conditions, reward-based incentivization, and coercive pressures. Moreover, we incorporate sustained multi-turn interaction loops to construct a more realistic simulation of real-world feedback dynamics. Extensive experiments across LLMs and Large Reasoning Models (LRMs) reveal critical vulnerabilities, particularly amplified deception under reinforcement dynamics, demonstrating that current models lack robust resistance to manipulative contextual cues and the urgent need for advanced safeguards against various deception behaviors. Code and resources are publicly available at this https URL. 

---
# CORE: Reducing UI Exposure in Mobile Agents via Collaboration Between Cloud and Local LLMs 

**Authors**: Gucongcong Fan, Chaoyue Niu, Chengfei Lyu, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15455)  

**Abstract**: Mobile agents rely on Large Language Models (LLMs) to plan and execute tasks on smartphone user interfaces (UIs). While cloud-based LLMs achieve high task accuracy, they require uploading the full UI state at every step, exposing unnecessary and often irrelevant information. In contrast, local LLMs avoid UI uploads but suffer from limited capacity, resulting in lower task success rates. We propose $\textbf{CORE}$, a $\textbf{CO}$llaborative framework that combines the strengths of cloud and local LLMs to $\textbf{R}$educe UI $\textbf{E}$xposure, while maintaining task accuracy for mobile agents. CORE comprises three key components: (1) $\textbf{Layout-aware block partitioning}$, which groups semantically related UI elements based on the XML screen hierarchy; (2) $\textbf{Co-planning}$, where local and cloud LLMs collaboratively identify the current sub-task; and (3) $\textbf{Co-decision-making}$, where the local LLM ranks relevant UI blocks, and the cloud LLM selects specific UI elements within the top-ranked block. CORE further introduces a multi-round accumulation mechanism to mitigate local misjudgment or limited context. Experiments across diverse mobile apps and tasks show that CORE reduces UI exposure by up to 55.6% while maintaining task success rates slightly below cloud-only agents, effectively mitigating unnecessary privacy exposure to the cloud. The code is available at this https URL. 

---
# Controllable Abstraction in Summary Generation for Large Language Models via Prompt Engineering 

**Authors**: Xiangchen Song, Yuchen Liu, Yaxuan Luan, Jinxu Guo, Xiaofan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.15436)  

**Abstract**: This study presents a controllable abstract summary generation method for large language models based on prompt engineering. To address the issues of summary quality and controllability in traditional methods, we design a multi-stage prompt generation framework. This framework generates summaries with varying levels of abstraction by performing semantic analysis, topic modeling, and noise control on the input text. The experiment uses the CNN/Daily Mail dataset and provides a detailed analysis of different prompt lengths, data noise, and text types. The experimental results show that prompt length has a significant impact on the quality of generated summaries. Both very short and very long prompt tokens result in a decrease in summary quality. Data noise also negatively affects the summary generation process. As noise levels increase, the ROUGE-L score gradually decreases. Furthermore, different text types have varying effects on the model's ability to generate summaries. The model performs best when handling news texts, while its performance is worse when processing academic articles. This research provides new insights into improving summary generation using large language models, particularly in how controlling prompt strategies and optimizing text preprocessing can enhance summary accuracy and controllability. 

---
# When Seeing Is not Enough: Revealing the Limits of Active Reasoning in MLLMs 

**Authors**: Hongcheng Liu, Pingjie Wang, Yuhao Wang, Siqu Ou, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15421)  

**Abstract**: Multimodal large language models (MLLMs) have shown strong capabilities across a broad range of benchmarks. However, most existing evaluations focus on passive inference, where models perform step-by-step reasoning under complete information. This setup is misaligned with real-world use, where seeing is not enough. This raises a fundamental question: Can MLLMs actively acquire missing evidence under incomplete information? To bridge this gap, we require the MLLMs to actively acquire missing evidence and iteratively refine decisions under incomplete information, by selecting a target image from a candidate pool without task-specific priors. To support systematic study, we propose GuessBench, a benchmark with both perception-oriented and knowledge-oriented images for evaluating active reasoning in MLLMs. We evaluate 20 superior MLLMs and find that performance on active reasoning lags far behind it on passive settings, indicating substantial room for improvement. Further analysis identifies fine-grained perception and timely decision-making as key challenges. Ablation studies show that perceptual enhancements benefit smaller models, whereas thinking-oriented methods provide consistent gains across model sizes. These results suggest promising directions for future research on multimodal active reasoning. 

---
# Fine-Tuning MedGemma for Clinical Captioning to Enhance Multimodal RAG over Malaysia CPGs 

**Authors**: Lee Qi Zun, Mohamad Zulhilmi Bin Abdul Halim, Goh Man Fye  

**Link**: [PDF](https://arxiv.org/pdf/2510.15418)  

**Abstract**: Retrieval-Augmented Generation systems are essential for providing fact-based guidance from Malaysian Clinical Practice Guidelines. However, their effectiveness with image-based queries is limited, as general Vision-Language Model captions often lack clinical specificity and factual grounding. This study proposes and validates a framework to specialize the MedGemma model for generating high-fidelity captions that serve as superior queries. To overcome data scarcity, we employ a knowledge distillation pipeline to create a synthetic dataset across dermatology, fundus, and chest radiography domains, and fine-tune MedGemma using the parameter-efficient QLoRA method. Performance was rigorously assessed through a dual framework measuring both classification accuracy and, via a novel application of the RAGAS framework, caption faithfulness, relevancy, and correctness. The fine-tuned model demonstrated substantial improvements in classification performance, while RAGAS evaluation confirmed significant gains in caption faithfulness and correctness, validating the models ability to produce reliable, factually grounded descriptions. This work establishes a robust pipeline for specializing medical VLMs and validates the resulting model as a high-quality query generator, laying the groundwork for enhancing multimodal RAG systems in evidence-based clinical decision support. 

---
# Large-scale User Game Lifecycle Representation Learning 

**Authors**: Yanjie Gou, Jiangming Liu, Kouying Xue, Yi Hua  

**Link**: [PDF](https://arxiv.org/pdf/2510.15412)  

**Abstract**: The rapid expansion of video game production necessitates the development of effective advertising and recommendation systems for online game platforms. Recommending and advertising games to users hinges on capturing their interest in games. However, existing representation learning methods crafted for handling billions of items in recommendation systems are unsuitable for game advertising and recommendation. This is primarily due to game sparsity, where the mere hundreds of games fall short for large-scale user representation learning, and game imbalance, where user behaviors are overwhelmingly dominated by a handful of popular games. To address the sparsity issue, we introduce the User Game Lifecycle (UGL), designed to enrich user behaviors in games. Additionally, we propose two innovative strategies aimed at manipulating user behaviors to more effectively extract both short and long-term interests. To tackle the game imbalance challenge, we present an Inverse Probability Masking strategy for UGL representation learning. The offline and online experimental results demonstrate that the UGL representations significantly enhance model by achieving a 1.83% AUC offline increase on average and a 21.67% CVR online increase on average for game advertising and a 0.5% AUC offline increase and a 0.82% ARPU online increase for in-game item recommendation. 

---
# VocalBench-DF: A Benchmark for Evaluating Speech LLM Robustness to Disfluency 

**Authors**: Hongcheng Liu, Yixuan Hou, Heyang Liu, Yuhao Wang, Yanfeng Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15406)  

**Abstract**: While Speech Large Language Models (Speech-LLMs) show strong performance in many applications, their robustness is critically under-tested, especially to speech disfluency. Existing evaluations often rely on idealized inputs, overlooking common disfluencies, particularly those associated with conditions like Parkinson's disease. This work investigates whether current Speech-LLMs can maintain performance when interacting with users who have speech impairments. To facilitate this inquiry, we introduce VocalBench-DF, a framework for the systematic evaluation of disfluency across a multi-dimensional taxonomy. Our evaluation of 22 mainstream Speech-LLMs reveals substantial performance degradation, indicating that their real-world readiness is limited. Further analysis identifies phoneme-level processing and long-context modeling as primary bottlenecks responsible for these failures. Strengthening recognition and reasoning capability from components and pipelines can substantially improve robustness. These findings highlight the urgent need for new methods to improve disfluency handling and build truly inclusive Speech-LLMs 

---
# Infinity Parser: Layout Aware Reinforcement Learning for Scanned Document Parsing 

**Authors**: Baode Wang, Biao Wu, Weizhen Li, Meng Fang, Zuming Huang, Jun Huang, Haozhe Wang, Yanjie Liang, Ling Chen, Wei Chu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.15349)  

**Abstract**: Document parsing from scanned images into structured formats remains a significant challenge due to its complexly intertwined elements such as text paragraphs, figures, formulas, and tables. Existing supervised fine-tuning methods often struggle to generalize across diverse document types, leading to poor performance, particularly on out-of-distribution data. This issue is further exacerbated by the limited availability of high-quality training data for layout-aware parsing tasks. To address these challenges, we introduce LayoutRL, a reinforcement learning framework that optimizes layout understanding through composite rewards integrating normalized edit distance, paragraph count accuracy, and reading order preservation. To support this training, we construct the Infinity-Doc-400K dataset, which we use to train Infinity-Parser, a vision-language model demonstrating robust generalization across various domains. Extensive evaluations on benchmarks including OmniDocBench, olmOCR-Bench, PubTabNet, and FinTabNet show that Infinity-Parser consistently achieves state-of-the-art performance across a broad range of document types, languages, and structural complexities, substantially outperforming both specialized document parsing systems and general-purpose vision-language models. We will release our code, dataset, and model to facilitate reproducible research in document parsing. 

---
# When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling 

**Authors**: Heecheol Yun, Kwangmin Ki, Junghyun Lee, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15346)  

**Abstract**: Ensembling Large Language Models (LLMs) has gained attention as a promising approach to surpass the performance of individual models by leveraging their complementary strengths. In particular, aggregating models' next-token probability distributions to select the next token has been shown to be effective in various tasks. However, while successful for short-form answers, its application to long-form generation remains underexplored. In this paper, we show that using existing ensemble methods in long-form generation requires a careful choice of ensembling positions, since the standard practice of ensembling at every token often degrades performance. We identify two key factors for determining these positions: tokenization mismatch across models and consensus in their next-token probability distributions. Based on this, we propose SAFE, (Stable And Fast LLM Ensembling), a framework that selectively ensembles by jointly considering these factors. To further improve stability, we introduce a probability sharpening strategy that consolidates probabilities spread across multiple sub-word tokens representing the same word into a single representative token. Our experiments on diverse benchmarks, including MATH500 and BBH, demonstrate that SAFE outperforms existing methods in both accuracy and efficiency, with gains achieved even when ensembling fewer than 1% of tokens. 

---
# Readability Reconsidered: A Cross-Dataset Analysis of Reference-Free Metrics 

**Authors**: Catarina G Belem, Parker Glenn, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.15345)  

**Abstract**: Automatic readability assessment plays a key role in ensuring effective and accessible written communication. Despite significant progress, the field is hindered by inconsistent definitions of readability and measurements that rely on surface-level text properties. In this work, we investigate the factors shaping human perceptions of readability through the analysis of 897 judgments, finding that, beyond surface-level cues, information content and topic strongly shape text comprehensibility. Furthermore, we evaluate 15 popular readability metrics across five English datasets, contrasting them with six more nuanced, model-based metrics. Our results show that four model-based metrics consistently place among the top four in rank correlations with human judgments, while the best performing traditional metric achieves an average rank of 8.6. These findings highlight a mismatch between current readability metrics and human perceptions, pointing to model-based approaches as a more promising direction. 

---
# AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction 

**Authors**: Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.15339)  

**Abstract**: Building effective knowledge graphs (KGs) for Retrieval-Augmented Generation (RAG) is pivotal for advancing question answering (QA) systems. However, its effectiveness is hindered by a fundamental disconnect: the knowledge graph (KG) construction process is decoupled from its downstream application, yielding suboptimal graph structures. To bridge this gap, we introduce AutoGraph-R1, the first framework to directly optimize KG construction for task performance using Reinforcement Learning (RL). AutoGraph-R1 trains an LLM constructor by framing graph generation as a policy learning problem, where the reward is derived from the graph's functional utility in a RAG pipeline. We design two novel, task-aware reward functions, one for graphs as knowledge carriers and another as knowledge indices. Across multiple QA benchmarks, AutoGraph-R1 consistently enables graph RAG methods to achieve significant performance gains over using task-agnostic baseline graphs. Our work shows it is possible to close the loop between construction and application, shifting the paradigm from building intrinsically ``good'' graphs to building demonstrably ``useful'' ones. 

---
# Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry 

**Authors**: Bolei Ma, Yina Yao, Anna-Carolina Haensch  

**Link**: [PDF](https://arxiv.org/pdf/2510.15313)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to creative domains, yet their performance in classical Chinese poetry generation and evaluation remains poorly understood. We propose a three-step evaluation framework that combines computational metrics, LLM-as-a-judge assessment, and human expert validation. Using this framework, we evaluate six state-of-the-art LLMs across multiple dimensions of poetic quality, including themes, emotions, imagery, form, and style. Our analysis reveals systematic generation and evaluation biases: LLMs exhibit "echo chamber" effects when assessing creative quality, often converging on flawed standards that diverge from human judgments. These findings highlight both the potential and limitations of current capabilities of LLMs as proxy for literacy generation and the limited evaluation practices, thereby demonstrating the continued need of hybrid validation from both humans and models in culturally and technically complex creative tasks. 

---
# Accelerating Mobile Language Model Generation via Hybrid Context and Hardware Coordination 

**Authors**: Zhiyang Chen, Daliang Xu, Haiyang Shen, Mengwei Xu, Shangguang Wang, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.15312)  

**Abstract**: Enhancing on-device large language models (LLMs) with contextual information from local data enables personalized and task-aware generation, powering use cases such as intelligent assistants and UI agents. While recent developments in neural processors have substantially improved the efficiency of prefill on mobile devices, the token-by-token generation process still suffers from high latency and limited hardware utilization due to its inherently memory-bound characteristics. This work presents CoordGen, a mobile inference framework that integrates speculative decoding with dynamic hardware scheduling to accelerate context-aware text generation on mobile devices. The framework introduces three synergistic components: (1) adaptive execution scheduling, which dynamically balances compute graphs between prefill and decoding phases; (2) context-aligned drafting, which improves speculative efficiency through lightweight online calibration to current tasks; and (3) hardware-efficient draft extension, which reuses and expands intermediate sequences to improve processing parallelism and reduce verification cost. Experiments on multiple smartphones and representative workloads show consistent improvements of up to 3.8x in generation speed and 4.7x in energy efficiency compared with existing mobile inference solutions. Component-level analysis further validates the contribution of each optimization. 

---
# Automatic essay scoring: leveraging Jaccard coefficient and Cosine similaritywith n-gram variation in vector space model approach 

**Authors**: Andharini Dwi Cahyani, Moh. Wildan Fathoni, Fika Hastarita Rachman, Ari Basuki, Salman Amin, Bain Khusnul Khotimah  

**Link**: [PDF](https://arxiv.org/pdf/2510.15311)  

**Abstract**: Automated essay scoring (AES) is a vital area of research aiming to provide efficient and accurate assessment tools for evaluating written content. This study investigates the effectiveness of two popular similarity metrics, Jaccard coefficient, and Cosine similarity, within the context of vector space models(VSM)employing unigram, bigram, and trigram representations. The data used in this research was obtained from the formative essay of the citizenship education subject in a junior high school. Each essay undergoes preprocessing to extract features using n-gram models, followed by vectorization to transform text data into numerical representations. Then, similarity scores are computed between essays using both Jaccard coefficient and Cosine similarity. The performance of the system is evaluated by analyzing the root mean square error (RMSE), which measures the difference between the scores given by human graders and those generated by the system. The result shows that the Cosine similarity outperformed the Jaccard coefficient. In terms of n-gram, unigrams have lower RMSE compared to bigrams and trigrams. 

---
# Exemplar-Guided Planing: Enhanced LLM Agent for KGQA 

**Authors**: Jingao Xu, Shuoyoucheng Ma, Xin Song, Rong Jiang, Hongkui Tu, Bin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.15283)  

**Abstract**: Large Language Models (LLMs) as interactive agents show significant promise in Knowledge Graph Question Answering (KGQA) but often struggle with the semantic gap between natural language queries and structured knowledge graph (KG) representations. This leads to suboptimal planning and inefficient exploration on KG, while training-free approaches often underutilize valuable reasoning patterns in training data. To address these limitations, we propose a novel framework, Exemplar-Guided Planning (EGP), which enhances the planning capabilities of LLM agents for KGQA. EGP first preprocesses the training set questions via entity templating to normalize semantic variations. It then retrieves highly similar exemplary questions and their successful reasoning paths from this preprocessed set using semantic embeddings and an efficient FAISS index. These retrieved exemplars dynamically guide the LLM's planning process in two key phases: (1) Task Decomposition, by aligning generated sub-objectives with proven reasoning steps, and (2) Relation Exploration, by providing high-quality auxiliary information to improve relation pruning accuracy. Additionally, we introduce a Smart Lookahead mechanism during relation exploration to improve efficiency by preemptively exploring promising paths and potentially terminating exploration earlier. We apply EGP to the Plan-on-Graph (PoG) framework, termed PoG-EGP. Extensive experiments on two real-world KGQA datasets, WebQSP and CWQ, demonstrate that PoG-EGP significantly improves over the baseline PoG system and other compared methods. 

---
# TACL: Threshold-Adaptive Curriculum Learning Strategy for Enhancing Medical Text Understanding 

**Authors**: Mucheng Ren, Yucheng Yan, He Chen, Danqing Hu, Jun Xu, Xian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.15269)  

**Abstract**: Medical texts, particularly electronic medical records (EMRs), are a cornerstone of modern healthcare, capturing critical information about patient care, diagnoses, and treatments. These texts hold immense potential for advancing clinical decision-making and healthcare analytics. However, their unstructured nature, domain-specific language, and variability across contexts make automated understanding an intricate challenge. Despite the advancements in natural language processing, existing methods often treat all data as equally challenging, ignoring the inherent differences in complexity across clinical records. This oversight limits the ability of models to effectively generalize and perform well on rare or complex cases. In this paper, we present TACL (Threshold-Adaptive Curriculum Learning), a novel framework designed to address these challenges by rethinking how models interact with medical texts during training. Inspired by the principle of progressive learning, TACL dynamically adjusts the training process based on the complexity of individual samples. By categorizing data into difficulty levels and prioritizing simpler cases early in training, the model builds a strong foundation before tackling more complex records. By applying TACL to multilingual medical data, including English and Chinese clinical records, we observe significant improvements across diverse clinical tasks, including automatic ICD coding, readmission prediction and TCM syndrome differentiation. TACL not only enhances the performance of automated systems but also demonstrates the potential to unify approaches across disparate medical domains, paving the way for more accurate, scalable, and globally applicable medical text understanding solutions. 

---
# TraceCoder: Towards Traceable ICD Coding via Multi-Source Knowledge Integration 

**Authors**: Mucheng Ren, He Chen, Yuchen Yan, Danqing Hu, Jun Xu, Xian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.15267)  

**Abstract**: Automated International Classification of Diseases (ICD) coding assigns standardized diagnosis and procedure codes to clinical records, playing a critical role in healthcare systems. However, existing methods face challenges such as semantic gaps between clinical text and ICD codes, poor performance on rare and long-tail codes, and limited interpretability. To address these issues, we propose TraceCoder, a novel framework integrating multi-source external knowledge to enhance traceability and explainability in ICD coding. TraceCoder dynamically incorporates diverse knowledge sources, including UMLS, Wikipedia, and large language models (LLMs), to enrich code representations, bridge semantic gaps, and handle rare and ambiguous codes. It also introduces a hybrid attention mechanism to model interactions among labels, clinical context, and knowledge, improving long-tail code recognition and making predictions interpretable by grounding them in external evidence. Experiments on MIMIC-III-ICD9, MIMIC-IV-ICD9, and MIMIC-IV-ICD10 datasets demonstrate that TraceCoder achieves state-of-the-art performance, with ablation studies validating the effectiveness of its components. TraceCoder offers a scalable and robust solution for automated ICD coding, aligning with clinical needs for accuracy, interpretability, and reliability. 

---
# Scaling Beyond Context: A Survey of Multimodal Retrieval-Augmented Generation for Document Understanding 

**Authors**: Sensen Gao, Shanshan Zhao, Xu Jiang, Lunhao Duan, Yong Xien Chng, Qing-Guo Chen, Weihua Luo, Kaifu Zhang, Jia-Wang Bian, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.15253)  

**Abstract**: Document understanding is critical for applications from financial analysis to scientific discovery. Current approaches, whether OCR-based pipelines feeding Large Language Models (LLMs) or native Multimodal LLMs (MLLMs), face key limitations: the former loses structural detail, while the latter struggles with context modeling. Retrieval-Augmented Generation (RAG) helps ground models in external data, but documents' multimodal nature, i.e., combining text, tables, charts, and layout, demands a more advanced paradigm: Multimodal RAG. This approach enables holistic retrieval and reasoning across all modalities, unlocking comprehensive document intelligence. Recognizing its importance, this paper presents a systematic survey of Multimodal RAG for document understanding. We propose a taxonomy based on domain, retrieval modality, and granularity, and review advances involving graph structures and agentic frameworks. We also summarize key datasets, benchmarks, and applications, and highlight open challenges in efficiency, fine-grained representation, and robustness, providing a roadmap for future progress in document AI. 

---
# Planner and Executor: Collaboration between Discrete Diffusion And Autoregressive Models in Reasoning 

**Authors**: Lina Berrayana, Ahmed Heakl, Muhammad Abdullah Sohail, Thomas Hofmann, Salman Khan, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15244)  

**Abstract**: Current autoregressive language models (ARMs) achieve high accuracy but require long token sequences, making them costly. Discrete diffusion language models (DDLMs) enable parallel and flexible generation within a fixed number of steps and have recently emerged for their strong performance in complex reasoning and long-term planning tasks. We present a study exploring hybrid architectures that couple DDLMs with ARMs to assess whether their collaboration can yield complementary benefits. We first examine collaboration in text space, where one model plans the reasoning process and another executes the final answer based on that plan. We then extend this setup to latent-space communication, introducing a learned projector that maps DDLM latents into the ARM's embedding space, potentially bypassing some of the text-generation limitations of diffusion models. We find that shifting DDLM --> ARM communication from text space to latent space yields significant accuracy gains, for example increasing from 27.0% to 54.0% on DART-5 and from 0.0% to 14.0% on AIME24. We also find that combining a DDLM planner with an ARM executor can provide substantial computational savings with little to no impact on accuracy. For example, the latent-space pipeline, using 64 tokens for planning and roughly 5 for execution, surpasses Qwen3.1-7B on DART-5 and AIME, despite Qwen using 44 times more tokens. Overall, our study offers new insights into reasoning with DDLMs and highlights their potential in hybrid architectures. 

---
# Extending Audio Context for Long-Form Understanding in Large Audio-Language Models 

**Authors**: Yuatyong Chaichana, Pittawat Taveekitworachai, Warit Sirichotedumrong, Potsawee Manakul, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2510.15231)  

**Abstract**: Large Audio-Language Models (LALMs) are often constrained by short audio context windows, even when their text backbones support long contexts, limiting long-form audio understanding. Prior work has introduced context-extension methods (e.g. YaRN) on unimodal LLMs, yet their application to LALMs remains unexplored. First, building on RoPE-based context extension, we introduce Partial YaRN, a training-free, audio-only extension method that modifies only audio token positions, leaving text positions intact to preserve the base LLM's text capabilities. Second, we propose Virtual Longform Audio Training (VLAT), a training strategy that extends Partial YaRN into a training-time positional augmentation. VLAT simulates diverse audio lengths during training, enabling generalization to inputs far longer than those seen in training and improving robustness for long-context audio understanding. Our experiments on SALMONN and Qwen2-Audio show that Partial YaRN outperforms the original models across wide range of settings, and VLAT training strategy provides substantial improvement, achieving strong performance on long audio of unseen lengths. 

---
# Structure-R1: Dynamically Leveraging Structural Knowledge in LLM Reasoning through Reinforcement Learning 

**Authors**: Junlin Wu, Xianrui Zhong, Jiashuo Sun, Bolian Li, Bowen Jin, Jiawei Han, Qingkai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.15191)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable advances in reasoning capabilities. However, their performance remains constrained by limited access to explicit and structured domain knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external information as context to augment reasoning. Nevertheless, traditional RAG systems typically operate over unstructured and fragmented text, resulting in low information density and suboptimal reasoning. To overcome these limitations, we propose \textsc{Structure-R1}, a novel framework that transforms retrieved content into structured representations optimized for reasoning. Leveraging reinforcement learning, \textsc{Structure-R1} learns a content representation policy that dynamically generates and adapts structural formats based on the demands of multi-step reasoning. Unlike prior methods that rely on fixed schemas, our approach adopts a generative paradigm capable of producing task-specific structures tailored to individual queries. To ensure the quality and reliability of these representations, we introduce a self-reward structural verification mechanism that checks whether the generated structures are both correct and self-contained. Extensive experiments on seven knowledge-intensive benchmarks show that \textsc{Structure-R1} consistently achieves competitive performance with a 7B-scale backbone model and matches the performance of much larger models. Additionally, our theoretical analysis demonstrates how structured representations enhance reasoning by improving information density and contextual clarity. Our code and data are available at: this https URL. 

---
# FarsiMCQGen: a Persian Multiple-choice Question Generation Framework 

**Authors**: Mohammad Heydari Rad, Rezvan Afari, Saeedeh Momtazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.15134)  

**Abstract**: Multiple-choice questions (MCQs) are commonly used in educational testing, as they offer an efficient means of evaluating learners' knowledge. However, generating high-quality MCQs, particularly in low-resource languages such as Persian, remains a significant challenge. This paper introduces FarsiMCQGen, an innovative approach for generating Persian-language MCQs. Our methodology combines candidate generation, filtering, and ranking techniques to build a model that generates answer choices resembling those in real MCQs. We leverage advanced methods, including Transformers and knowledge graphs, integrated with rule-based approaches to craft credible distractors that challenge test-takers. Our work is based on data from Wikipedia, which includes general knowledge questions. Furthermore, this study introduces a novel Persian MCQ dataset comprising 10,289 questions. This dataset is evaluated by different state-of-the-art large language models (LLMs). Our results demonstrate the effectiveness of our model and the quality of the generated dataset, which has the potential to inspire further research on MCQs. 

---
# Latent Topic Synthesis: Leveraging LLMs for Electoral Ad Analysis 

**Authors**: Alexander Brady, Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2510.15125)  

**Abstract**: Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically generating an interpretable topic taxonomy from an unlabeled corpus. By combining unsupervised clustering with prompt-based labeling, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets or domain expertise. We apply this framework to a large corpus of Meta (previously known as Facebook) political ads from the month ahead of the 2024 U.S. Presidential election. Our approach uncovers latent discourse structures, synthesizes semantically rich topic labels, and annotates topics with moral framing dimensions. We show quantitative and qualitative analyses to demonstrate the effectiveness of our framework. Our findings reveal that voting and immigration ads dominate overall spending and impressions, while abortion and election-integrity achieve disproportionate reach. Funding patterns are equally polarized: economic appeals are driven mainly by conservative PACs, abortion messaging splits between pro- and anti-rights coalitions, and crime-and-justice campaigns are fragmented across local committees. The framing of these appeals also diverges--abortion ads emphasize liberty/oppression rhetoric, while economic messaging blends care/harm, fairness/cheating, and liberty/oppression narratives. Topic salience further reveals strong correlations between moral foundations and issues. Demographic targeting also emerges. This work supports scalable, interpretable analysis of political messaging on social media, enabling researchers, policymakers, and the public to better understand emerging narratives, polarization dynamics, and the moral underpinnings of digital political communication. 

---
# Measuring the Effect of Disfluency in Multilingual Knowledge Probing Benchmarks 

**Authors**: Kirill Semenov, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2510.15115)  

**Abstract**: For multilingual factual knowledge assessment of LLMs, benchmarks such as MLAMA use template translations that do not take into account the grammatical and semantic information of the named entities inserted in the sentence. This leads to numerous instances of ungrammaticality or wrong wording of the final prompts, which complicates the interpretation of scores, especially for languages that have a rich morphological inventory. In this work, we sample 4 Slavic languages from the MLAMA dataset and compare the knowledge retrieval scores between the initial (templated) MLAMA dataset and its sentence-level translations made by Google Translate and ChatGPT. We observe a significant increase in knowledge retrieval scores, and provide a qualitative analysis for possible reasons behind it. We also make an additional analysis of 5 more languages from different families and see similar patterns. Therefore, we encourage the community to control the grammaticality of highly multilingual datasets for higher and more interpretable results, which is well approximated by whole sentence translation with neural MT or LLM systems. The dataset and all related code is published at the Github repository: this https URL. 

---
# Continual Learning via Sparse Memory Finetuning 

**Authors**: Jessy Lin, Luke Zettlemoyer, Gargi Ghosh, Wen-Tau Yih, Aram Markosyan, Vincent-Pierre Berges, Barlas Oğuz  

**Link**: [PDF](https://arxiv.org/pdf/2510.15103)  

**Abstract**: Modern language models are powerful, but typically static after deployment. A major obstacle to building models that continually learn over time is catastrophic forgetting, where updating on new data erases previously acquired capabilities. Motivated by the intuition that mitigating forgetting is challenging because trainable parameters are shared across all tasks, we investigate whether sparse parameter updates can enable learning without catastrophic forgetting. We introduce sparse memory finetuning, leveraging memory layer models (Berges et al., 2024), which are sparsely updated by design. By updating only the memory slots that are highly activated by a new piece of knowledge relative to usage on pretraining data, we reduce interference between new knowledge and the model's existing capabilities. We evaluate learning and forgetting compared to full finetuning and parameter-efficient finetuning with LoRA on two question answering tasks. We find that sparse memory finetuning learns new knowledge while exhibiting substantially less forgetting: while NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop with the same level of new knowledge acquisition. Our results suggest sparsity in memory layers offers a promising path toward continual learning in large language models. 

---
# A Generalizable Rhetorical Strategy Annotation Model Using LLM-based Debate Simulation and Labelling 

**Authors**: Shiyu Ji, Farnoosh Hashemi, Joice Chen, Juanwen Pan, Weicheng Ma, Hefan Zhang, Sophia Pan, Ming Cheng, Shubham Mohole, Saeed Hassanpour, Soroush Vosoughi, Michael Macy  

**Link**: [PDF](https://arxiv.org/pdf/2510.15081)  

**Abstract**: Rhetorical strategies are central to persuasive communication, from political discourse and marketing to legal argumentation. However, analysis of rhetorical strategies has been limited by reliance on human annotation, which is costly, inconsistent, difficult to scale. Their associated datasets are often limited to specific topics and strategies, posing challenges for robust model development. We propose a novel framework that leverages large language models (LLMs) to automatically generate and label synthetic debate data based on a four-part rhetorical typology (causal, empirical, emotional, moral). We fine-tune transformer-based classifiers on this LLM-labeled dataset and validate its performance against human-labeled data on this dataset and on multiple external corpora. Our model achieves high performance and strong generalization across topical domains. We illustrate two applications with the fine-tuned model: (1) the improvement in persuasiveness prediction from incorporating rhetorical strategy labels, and (2) analyzing temporal and partisan shifts in rhetorical strategies in U.S. Presidential debates (1960-2020), revealing increased use of affective over cognitive argument in U.S. Presidential debates. 

---
# Can generative AI figure out figurative language? The influence of idioms on essay scoring by ChatGPT, Gemini, and Deepseek 

**Authors**: Enis Oğuz  

**Link**: [PDF](https://arxiv.org/pdf/2510.15009)  

**Abstract**: The developments in Generative AI technologies have paved the way for numerous innovations in different fields. Recently, Generative AI has been proposed as a competitor to AES systems in evaluating student essays automatically. Considering the potential limitations of AI in processing idioms, this study assessed the scoring performances of Generative AI models for essays with and without idioms by incorporating insights from Corpus Linguistics and Computational Linguistics. Two equal essay lists were created from 348 student essays taken from a corpus: one with multiple idioms present in each essay and another with no idioms in essays. Three Generative AI models (ChatGPT, Gemini, and Deepseek) were asked to score all essays in both lists three times, using the same rubric used by human raters in assigning essay scores. The results revealed excellent consistency for all models, but Gemini outperformed its competitors in interrater reliability with human raters. There was also no detectable bias for any demographic group in AI assessment. For essays with multiple idioms, Gemini followed a the most similar pattern to human raters. While the models in the study demonstrated potential for a hybrid approach, Gemini was the best candidate for the task due to its ability to handle figurative language and showed promise for handling essay-scoring tasks alone in the future. 

---
# Rethinking Toxicity Evaluation in Large Language Models: A Multi-Label Perspective 

**Authors**: Zhiqiang Kou, Junyang Chen, Xin-Qiang Cai, Ming-Kun Xie, Biao Liu, Changwei Wang, Lei Feng, Yuheng Jia, Gang Niu, Masashi Sugiyama, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2510.15007)  

**Abstract**: Large language models (LLMs) have achieved impressive results across a range of natural language processing tasks, but their potential to generate harmful content has raised serious safety concerns. Current toxicity detectors primarily rely on single-label benchmarks, which cannot adequately capture the inherently ambiguous and multi-dimensional nature of real-world toxic prompts. This limitation results in biased evaluations, including missed toxic detections and false positives, undermining the reliability of existing detectors. Additionally, gathering comprehensive multi-label annotations across fine-grained toxicity categories is prohibitively costly, further hindering effective evaluation and development. To tackle these issues, we introduce three novel multi-label benchmarks for toxicity detection: \textbf{Q-A-MLL}, \textbf{R-A-MLL}, and \textbf{H-X-MLL}, derived from public toxicity datasets and annotated according to a detailed 15-category taxonomy. We further provide a theoretical proof that, on our released datasets, training with pseudo-labels yields better performance than directly learning from single-label supervision. In addition, we develop a pseudo-label-based toxicity detection method. Extensive experimental results show that our approach significantly surpasses advanced baselines, including GPT-4o and DeepSeek, thus enabling more accurate and reliable evaluation of multi-label toxicity in LLM-generated content. 

---
# OmniVinci: Enhancing Architecture and Data for Omni-Modal Understanding LLM 

**Authors**: Hanrong Ye, Chao-Han Huck Yang, Arushi Goel, Wei Huang, Ligeng Zhu, Yuanhang Su, Sean Lin, An-Chieh Cheng, Zhen Wan, Jinchuan Tian, Yuming Lou, Dong Yang, Zhijian Liu, Yukang Chen, Ambrish Dantrey, Ehsan Jahangiri, Sreyan Ghosh, Daguang Xu, Ehsan Hosseini-Asl, Danial Mohseni Taheri, Vidya Murali, Sifei Liu, Jason Lu, Oluwatobi Olabiyi, Frank Wang, Rafael Valle, Bryan Catanzaro, Andrew Tao, Song Han, Jan Kautz, Hongxu Yin, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.15870)  

**Abstract**: Advancing machine intelligence requires developing the ability to perceive across multiple modalities, much as humans sense the world. We introduce OmniVinci, an initiative to build a strong, open-source, omni-modal LLM. We carefully study the design choices across model architecture and data curation. For model architecture, we present three key innovations: (i) OmniAlignNet for strengthening alignment between vision and audio embeddings in a shared omni-modal latent space; (ii) Temporal Embedding Grouping for capturing relative temporal alignment between vision and audio signals; and (iii) Constrained Rotary Time Embedding for encoding absolute temporal information in omni-modal embeddings. We introduce a curation and synthesis pipeline that generates 24M single-modal and omni-modal conversations. We find that modalities reinforce one another in both perception and reasoning. Our model, OmniVinci, outperforms Qwen2.5-Omni with +19.05 on DailyOmni (cross-modal understanding), +1.7 on MMAR (audio), and +3.9 on Video-MME (vision), while using just 0.2T training tokens - a 6 times reduction compared to Qwen2.5-Omni's 1.2T. We finally demonstrate omni-modal advantages in downstream applications spanning robotics, medical AI, and smart factory. 

---
# GraphMind: Interactive Novelty Assessment System for Accelerating Scientific Discovery 

**Authors**: Italo Luis da Silva, Hanqi Yan, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2510.15706)  

**Abstract**: Large Language Models (LLMs) show strong reasoning and text generation capabilities, prompting their use in scientific literature analysis, including novelty assessment. While evaluating novelty of scientific papers is crucial for peer review, it requires extensive knowledge of related work, something not all reviewers have. While recent work on LLM-assisted scientific literature analysis supports literature comparison, existing approaches offer limited transparency and lack mechanisms for result traceability via an information retrieval module. To address this gap, we introduce $\textbf{GraphMind}$, an easy-to-use interactive web tool designed to assist users in evaluating the novelty of scientific papers or drafted ideas. Specially, $\textbf{GraphMind}$ enables users to capture the main structure of a scientific paper, explore related ideas through various perspectives, and assess novelty via providing verifiable contextual insights. $\textbf{GraphMind}$ enables users to annotate key elements of a paper, explore related papers through various relationships, and assess novelty with contextual insight. This tool integrates external APIs such as arXiv and Semantic Scholar with LLMs to support annotation, extraction, retrieval and classification of papers. This combination provides users with a rich, structured view of a scientific idea's core contributions and its connections to existing work. $\textbf{GraphMind}$ is available at this https URL and a demonstration video at this https URL. The source code is available at this https URL. 

---
# Exploring the Synergy of Quantitative Factors and Newsflow Representations from Large Language Models for Stock Return Prediction 

**Authors**: Tian Guo, Emmanuel Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.15691)  

**Abstract**: In quantitative investing, return prediction supports various tasks, including stock selection, portfolio optimization, and risk management. Quantitative factors, such as valuation, quality, and growth, capture various characteristics of stocks. Unstructured financial data, like news and transcripts, has attracted growing attention, driven by recent advances in large language models (LLMs). This paper examines effective methods for leveraging multimodal factors and newsflow in return prediction and stock selection. First, we introduce a fusion learning framework to learn a unified representation from factors and newsflow representations generated by an LLM. Within this framework, we compare three representative methods: representation combination, representation summation, and attentive representations. Next, building on empirical observations from fusion learning, we explore the mixture model that adaptively combines predictions made by single modalities and their fusion. To mitigate the training instability observed in the mixture model, we introduce a decoupled training approach with theoretical insights. Finally, our experiments on real investment universes yield several insights into effective multimodal modeling of factors and news for stock return prediction. 

---
# SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation 

**Authors**: Ines Besrour, Jingbo He, Tobias Schreieder, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2510.15682)  

**Abstract**: We present SQuAI (this https URL), a scalable and trustworthy multi-agent retrieval-augmented generation (RAG) framework for scientific question answering (QA) with large language models (LLMs). SQuAI addresses key limitations of existing RAG systems in the scholarly domain, where complex, open-domain questions demand accurate answers, explicit claims with citations, and retrieval across millions of scientific documents. Built on over 2.3 million full-text papers from arXiv.org, SQuAI employs four collaborative agents to decompose complex questions into sub-questions, retrieve targeted evidence via hybrid sparse-dense retrieval, and adaptively filter documents to improve contextual relevance. To ensure faithfulness and traceability, SQuAI integrates in-line citations for each generated claim and provides supporting sentences from the source documents. Our system improves faithfulness, answer relevance, and contextual relevance by up to +0.088 (12%) over a strong RAG baseline. We further release a benchmark of 1,000 scientific question-answer-evidence triplets to support reproducibility. With transparent reasoning, verifiable citations, and domain-wide scalability, SQuAI demonstrates how multi-agent RAG enables more trustworthy scientific QA with LLMs. 

---
# Build Your Personalized Research Group: A Multiagent Framework for Continual and Interactive Science Automation 

**Authors**: Ed Li, Junyu Ren, Xintian Pan, Cat Yan, Chuanhao Li, Dirk Bergemann, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15624)  

**Abstract**: The automation of scientific discovery represents a critical milestone in Artificial Intelligence (AI) research. However, existing agentic systems for science suffer from two fundamental limitations: rigid, pre-programmed workflows that cannot adapt to intermediate findings, and inadequate context management that hinders long-horizon research. We present \texttt{freephdlabor}, an open-source multiagent framework featuring \textit{fully dynamic workflows} determined by real-time agent reasoning and a \coloremph{\textit{modular architecture}} enabling seamless customization -- users can modify, add, or remove agents to address domain-specific requirements. The framework provides comprehensive infrastructure including \textit{automatic context compaction}, \textit{workspace-based communication} to prevent information degradation, \textit{memory persistence} across sessions, and \textit{non-blocking human intervention} mechanisms. These features collectively transform automated research from isolated, single-run attempts into \textit{continual research programs} that build systematically on prior explorations and incorporate human feedback. By providing both the architectural principles and practical implementation for building customizable co-scientist systems, this work aims to facilitate broader adoption of automated research across scientific domains, enabling practitioners to deploy interactive multiagent systems that autonomously conduct end-to-end research -- from ideation through experimentation to publication-ready manuscripts. 

---
# Unleashing Scientific Reasoning for Bio-experimental Protocol Generation via Structured Component-based Reward Mechanism 

**Authors**: Haoran Sun, Yankai Jiang, Zhenyu Tang, Yaning Pan, Shuang Gu, Zekai Lin, Lilong Wang, Wenjie Lou, Lei Liu, Lei Bai, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15600)  

**Abstract**: The foundation of reproducible science lies in protocols that are precise, logically ordered, and executable. The autonomous generation of these protocols through natural language queries could greatly improve the efficiency of the reproduction process. However, current leading large language models (LLMs) often generate incomplete or inconsistent protocols, limiting their utility. To address this limitation, we first introduce SciRecipe, a large-scale dataset of over 12K structured protocols spanning 27 biological subfields and encompassing both comprehension and problem-solving tasks. To further improve protocol generation, we propose the "Sketch-and-Fill" paradigm, which separates analysis, structuring, and expression to ensure each step is explicit and verifiable. Complementing this, the structured component-based reward mechanism evaluates step granularity, action order, and semantic fidelity, aligning model optimization with experimental reliability. Building on these components, we develop Thoth, trained through a staged Knowledge-to-Action process that progresses from knowledge acquisition to operational reasoning and ultimately to robust, executable protocol generation. Across multiple benchmarks, Thoth consistently surpasses both proprietary and open-source LLMs, achieving significant improvements in step alignment, logical sequencing, and semantic accuracy. Our approach paves the way for reliable scientific assistants that bridge knowledge with experimental execution. All data, code, and models will be released publicly. 

---
# Leveraging Test Driven Development with Large Language Models for Reliable and Verifiable Spreadsheet Code Generation: A Research Framework 

**Authors**: Dr Simon Thorne, Dr Advait Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2510.15585)  

**Abstract**: Large Language Models (LLMs), such as ChatGPT, are increasingly leveraged for generating both traditional software code and spreadsheet logic. Despite their impressive generative capabilities, these models frequently exhibit critical issues such as hallucinations, subtle logical inconsistencies, and syntactic errors, risks particularly acute in high stakes domains like financial modelling and scientific computations, where accuracy and reliability are paramount. This position paper proposes a structured research framework that integrates the proven software engineering practice of Test-Driven Development (TDD) with Large Language Model (LLM) driven generation to enhance the correctness of, reliability of, and user confidence in generated outputs. We hypothesise that a "test first" methodology provides both technical constraints and cognitive scaffolding, guiding LLM outputs towards more accurate, verifiable, and comprehensible solutions. Our framework, applicable across diverse programming contexts, from spreadsheet formula generation to scripting languages such as Python and strongly typed languages like Rust, includes an explicitly outlined experimental design with clearly defined participant groups, evaluation metrics, and illustrative TDD based prompting examples. By emphasising test driven thinking, we aim to improve computational thinking, prompt engineering skills, and user engagement, particularly benefiting spreadsheet users who often lack formal programming training yet face serious consequences from logical errors. We invite collaboration to refine and empirically evaluate this approach, ultimately aiming to establish responsible and reliable LLM integration in both educational and professional development practices. 

---
# BeLLMan: Controlling LLM Congestion 

**Authors**: Tella Rajashekhar Reddy, Atharva Deshmukh, Karan Tandon, Rohan Gandhi, Anjaly Parayil, Debopam Bhattacherjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.15330)  

**Abstract**: Large language model (LLM) applications are blindfolded to the infrastructure underneath and generate tokens autoregressively, indifferent to the system load, thus risking inferencing latency inflation and poor user experience. Our first-cut controller, named beLLMan, enables the LLM infrastructure to actively and progressively signal the first-party LLM application to adjust the output length in response to changing system load. On a real testbed with H100 GPUs, beLLMan helps keep inferencing latency under control (upto 8X lower end-to-end latency) and reduces energy consumption by 25% (while serving 19% more requests) during periods of congestion for a summarization workload. 

---
# DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models 

**Authors**: Yangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.15260)  

**Abstract**: Large language models are highly sensitive to prompt wording. However, popular automatic prompt search methods, including InstructZero, often degrade under distribution shift and adversarial evaluation because they optimize expected performance under a single evaluation distribution. Consequently, prompts that work in one setting frequently fail to transfer. To address this, DRO-InstructZero formulates zero-shot prompt optimization as robust Bayesian optimization. Specifically, an f-divergence ball defines an ambiguity set around the evaluation distribution, and a robust acquisition rule maximizes worst-case expected utility while retaining the query efficiency of Bayesian search. Therefore, the search explicitly targets reliability under distribution shift rather than average behavior alone. Experiments follow the instruction-induction protocol with matched query budgets across formality rewriting, code debugging, and translation. For example, on BIG-Bench informative-to-formal rewriting, accuracy improves from 61.3 +/- 0.7% to approximately 85-90%, yielding an absolute gain of about 25-30 points. Moreover, auto-debugging shows about +25-point gains under domain shift. Meanwhile, stable tasks such as cause-and-effect remain above 96%, indicating no loss on in-distribution cases. Furthermore, improvements are consistent across divergence choices and decoding temperatures. Overall, DRO-InstructZero connects distributionally robust optimization with prompt learning, offering a plug-and-play and general approach for reliable, transferable prompt alignment under real-world uncertainty. 

---
# Multi-dimensional Data Analysis and Applications Basing on LLM Agents and Knowledge Graph Interactions 

**Authors**: Xi Wang, Xianyao Ling, Kun Li, Gang Yin, Liang Zhang, Jiang Wu, Jun Xu, Fu Zhang, Wenbo Lei, Annie Wang, Peng Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.15258)  

**Abstract**: In the current era of big data, extracting deep insights from massive, heterogeneous, and complexly associated multi-dimensional data has become a significant challenge. Large Language Models (LLMs) perform well in natural language understanding and generation, but still suffer from "hallucination" issues when processing structured knowledge and are difficult to update in real-time. Although Knowledge Graphs (KGs) can explicitly store structured knowledge, their static nature limits dynamic interaction and analytical capabilities. Therefore, this paper proposes a multi-dimensional data analysis method based on the interactions between LLM agents and KGs, constructing a dynamic, collaborative analytical ecosystem. This method utilizes LLM agents to automatically extract product data from unstructured data, constructs and visualizes the KG in real-time, and supports users in deep exploration and analysis of graph nodes through an interactive platform. Experimental results show that this method has significant advantages in product ecosystem analysis, relationship mining, and user-driven exploratory analysis, providing new ideas and tools for multi-dimensional data analysis. 

---
# FinTrust: A Comprehensive Benchmark of Trustworthiness Evaluation in Finance Domain 

**Authors**: Tiansheng Hu, Tongyan Hu, Liuyang Bai, Yilun Zhao, Arman Cohan, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.15232)  

**Abstract**: Recent LLMs have demonstrated promising ability in solving finance related problems. However, applying LLMs in real-world finance application remains challenging due to its high risk and high stakes property. This paper introduces FinTrust, a comprehensive benchmark specifically designed for evaluating the trustworthiness of LLMs in finance applications. Our benchmark focuses on a wide range of alignment issues based on practical context and features fine-grained tasks for each dimension of trustworthiness evaluation. We assess eleven LLMs on FinTrust and find that proprietary models like o4-mini outperforms in most tasks such as safety while open-source models like DeepSeek-V3 have advantage in specific areas like industry-level fairness. For challenging task like fiduciary alignment and disclosure, all LLMs fall short, showing a significant gap in legal awareness. We believe that FinTrust can be a valuable benchmark for LLMs' trustworthiness evaluation in finance domain. 

---
# Soundness-Aware Level: A Microscopic Signature that Predicts LLM Reasoning Potential 

**Authors**: Xuansheng Wu, Xiaoman Pan, Wenlin Yao, Jianshu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.15216)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) can elicit strong reasoning in large language models (LLMs), while their performance after RLVR varies dramatically across different base models. This raises a fundamental question: what microscopic property of pre-trained models leads to this variation? To investigate, we formalize reasoning as chains of Horn clauses ("if-then" rules) built from features extracted from the LLM's latent space via cross-layer sparse autoencoders (SAEs). We estimate the transition probabilities between its features, and further categorize each rule by its semantic soundness level (e.g., strict, plausible, noisy) with an LLM. Our key discovery is that high-potential models are inherently soundness-aware: their internal probability distributions systematically shift across rules' soundness levels, becoming highly distinct for "strict" versus "noisy" rules. In contrast, weaker models are soundness-agnostic, collapsing to one distribution regardless of soundness levels. To quantify this, we introduce the Soundness-Aware Level (SAL), a microscopic metric using the Jensen-Shannon Divergence to measure the separation between these distributions. We show that SAL's predictions of post-RLVR reasoning performance follow a precise empirical law (R^2=0.87) across diverse model families (Qwen, Mistral, Llama, DeepSeek) and scales (0.5B-14B). This reveals that a model's reasoning potential is tied to its intrinsic, pre-trained ability to distinguish sound knowledge from unsound ones. These findings underscore the critical role of model pre-training in shaping reasoning and offer a practical metric grounded in the model's internal mechanisms for selecting/designing stronger base models. 

---
# MAGPIE: A benchmark for Multi-AGent contextual PrIvacy Evaluation 

**Authors**: Gurusha Juneja, Jayanth Naga Sai Pasupulati, Alon Albalak, Wenyue Hua, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.15186)  

**Abstract**: A core challenge for autonomous LLM agents in collaborative settings is balancing robust privacy understanding and preservation alongside task efficacy. Existing privacy benchmarks only focus on simplistic, single-turn interactions where private information can be trivially omitted without affecting task outcomes. In this paper, we introduce MAGPIE (Multi-AGent contextual PrIvacy Evaluation), a novel benchmark of 200 high-stakes tasks designed to evaluate privacy understanding and preservation in multi-agent collaborative, non-adversarial scenarios. MAGPIE integrates private information as essential for task resolution, forcing agents to balance effective collaboration with strategic information control. Our evaluation reveals that state-of-the-art agents, including GPT-5 and Gemini 2.5-Pro, exhibit significant privacy leakage, with Gemini 2.5-Pro leaking up to 50.7% and GPT-5 up to 35.1% of the sensitive information even when explicitly instructed not to. Moreover, these agents struggle to achieve consensus or task completion and often resort to undesirable behaviors such as manipulation and power-seeking (e.g., Gemini 2.5-Pro demonstrating manipulation in 38.2% of the cases). These findings underscore that current LLM agents lack robust privacy understanding and are not yet adequately aligned to simultaneously preserve privacy and maintain effective collaboration in complex environments. 

---
# Train a Unified Multimodal Data Quality Classifier with Synthetic Data 

**Authors**: Weizhi Wang, Rongmei Lin, Shiyang Li, Colin Lockard, Ritesh Sarkhel, Sanket Lokegaonkar, Jingbo Shang, Xifeng Yan, Nasser Zalmout, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.15162)  

**Abstract**: The Multimodal Large Language Models (MLLMs) are continually pre-trained on a mixture of image-text caption data and interleaved document data, while the high-quality data filtering towards image-text interleaved document data is under-explored. We propose to train an efficient MLLM as a Unified Mulitmodal Data Quality Classifier to Filter both high-quality image-text caption and interleaved data (UniFilter). To address the challenge of collecting diverse labeled multimodal data, we introduce a semi-synthetic approach that leverages readily available raw images and generates corresponding text across four quality levels. This method enables efficient creation of sample-score pairs for both caption and interleaved document data to train UniFilter. We apply UniFilter to curate high-quality caption data from DataComp caption dataset and interleaved data from the OBELICS image-text interleaved dataset. MLLMs pre-trained on the filtered data demonstrate significantly enhanced capabilities compared to those trained on baseline-filtered data, achieving stronger zero-shot reasoning and in-context learning capabilities. After visual supervised fine-tuning, these UniFilter-induced MLLMs achieve stronger performance on various benchmarks, highlighting the downstream benefits of high-quality multimodal pre-training. We release the synthetic training data used for training UniFilter, the UniFilter model checkpoints, and the high-quality interleaved document subset OBELICS-HQ, curated by UniFilter, to the community for reproduction and further development. 

---
# HugAgent: Evaluating LLMs in Simulating Human-Like Individual Reasoning on Open-Ended Tasks 

**Authors**: Chance Jiajie Li, Zhenze Mo, Yuhan Tang, Ao Qu, Jiayi Wu, Kaiya Ivy Zhao, Yulu Gan, Jie Fan, Jiangbo Yu, Hang Jiang, Paul Pu Liang, Jinhua Zhao, Luis Alberto Alonso Pastor, Kent Larson  

**Link**: [PDF](https://arxiv.org/pdf/2510.15144)  

**Abstract**: Simulating human reasoning in open-ended tasks has been a long-standing aspiration in AI and cognitive science. While large language models now approximate human responses at scale, they remain tuned to population-level consensus, often erasing the individuality of reasoning styles and belief trajectories. To advance the vision of more human-like reasoning in machines, we introduce HugAgent (Human-Grounded Agent Benchmark), a benchmark for average-to-individual reasoning adaptation. The task is to predict how a specific person would reason and update their beliefs in novel scenarios, given partial evidence of their past views. HugAgent adopts a dual-track design: a synthetic track for scale and systematic stress tests, and a human track for ecologically valid, "out-loud" reasoning data. This design enables scalable, reproducible evaluation of intra-agent fidelity: whether models can capture not just what people believe, but how their reasoning evolves. Experiments with state-of-the-art LLMs reveal persistent adaptation gaps, positioning HugAgent as the first extensible benchmark for aligning machine reasoning with the individuality of human thought. Our benchmark and chatbot are open-sourced as HugAgent (this https URL) and TraceYourThinking (this https URL). 

---
# DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforcement Learning 

**Authors**: Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Yejin Choi, Jan Kautz, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.15110)  

**Abstract**: Reasoning language models such as OpenAI-o1, DeepSeek-R1, and Qwen achieve strong performance via extended chains of thought but often generate unnecessarily long outputs. Maximizing intelligence per token--accuracy relative to response length--remains an open problem. We revisit reinforcement learning (RL) with the simplest length penalty--truncation--and show that accuracy degradation arises not from the lack of sophisticated penalties but from inadequate RL optimization. We identify three key challenges: (i) large bias in advantage estimation, (ii) entropy collapse, and (iii) sparse reward signal. We address them with Doing Length pEnalty Right (DLER), a training recipe combining batch-wise reward normalization, higher clipping, dynamic sampling, and a simple truncation length penalty. DLER achieves state-of-the-art accuracy--efficiency trade-offs, cutting output length by over 70 percent while surpassing all previous baseline accuracy. It also improves test-time scaling: compared to DeepSeek-R1-7B, DLER-7B generates multiple concise responses in parallel with 28 percent higher accuracy and lower latency. We further introduce Difficulty-Aware DLER, which adaptively tightens truncation on easier questions for additional efficiency gains. We also propose an update-selective merging method that preserves baseline accuracy while retaining the concise reasoning ability of the DLER model, which is useful for scenarios where RL training data is scarce. 

---
# Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models 

**Authors**: Samuel Paech, Allen Roush, Judah Goldfeder, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2510.15061)  

**Abstract**: Widespread LLM adoption has introduced characteristic repetitive phraseology, termed ``slop,'' which degrades output quality and makes AI-generated text immediately recognizable. We present Antislop, a comprehensive framework providing tools to both detect and eliminate these overused patterns. Our approach combines three innovations: (1) The Antislop Sampler, which uses backtracking to suppress unwanted strings at inference time without destroying vocabulary; (2) An automated pipeline that profiles model-specific slop against human baselines and generates training data; (3) Final Token Preference Optimization (FTPO), a novel fine-tuning method that operates on individual tokens, surgically adjusting logits wherever a banned pattern has appeared in an inference trace. We demonstrate that some slop patterns appear over 1,000$\times$ more frequently in LLM output than human text. The Antislop Sampler successfully suppresses 8,000+ patterns while maintaining quality, whereas token banning becomes unusable at just 2,000. Most importantly, FTPO achieves 90\% slop reduction while maintaining or improving performance in cross-domain evals including GSM8K, MMLU, and creative writing tasks. In contrast, DPO suffers significant degradation in writing quality and lexical diversity despite achieving weaker suppression. We release all code and results under MIT license: this https URL. 

---
# Internalizing World Models via Self-Play Finetuning for Agentic RL 

**Authors**: Shiqi Chen, Tongyao Zhu, Zian Wang, Jinghan Zhang, Kangrui Wang, Siyang Gao, Teng Xiao, Yee Whye Teh, Junxian He, Manling Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.15047)  

**Abstract**: Large Language Models (LLMs) as agents often struggle in out-of-distribution (OOD) scenarios. Real-world environments are complex and dynamic, governed by task-specific rules and stochasticity, which makes it difficult for LLMs to ground their internal knowledge in those dynamics. Under such OOD conditions, vanilla RL training often fails to scale; we observe Pass@k--the probability that at least one of (k) sampled trajectories succeeds--drops markedly across training steps, indicating brittle exploration and limited generalization. Inspired by model-based reinforcement learning, we hypothesize that equipping LLM agents with an internal world model can better align reasoning with environmental dynamics and improve decision-making. We show how to encode this world model by decomposing it into two components: state representation and transition modeling. Building on this, we introduce SPA, a simple reinforcement learning framework that cold-starts the policy via a Self-Play supervised finetuning (SFT) stage to learn the world model by interacting with the environment, then uses it to simulate future states prior to policy optimization. This simple initialization outperforms the online world-modeling baseline and greatly boosts the RL-based agent training performance. Experiments across diverse environments like Sokoban, FrozenLake, and Sudoku show that our approach significantly improves performance. For example, SPA boosts the Sokoban success rate from 25.6% to 59.8% and raises the FrozenLake score from 22.1% to 70.9% for the Qwen2.5-1.5B-Instruct model. 

---
# Composition-Grounded Instruction Synthesis for Visual Reasoning 

**Authors**: Xinyi Gu, Jiayuan Mao, Zhang-Wei Hong, Zhuoran Yu, Pengyuan Li, Dhiraj Joshi, Rogerio Feris, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2510.15040)  

**Abstract**: Pretrained multi-modal large language models (MLLMs) demonstrate strong performance on diverse multimodal tasks, but remain limited in reasoning capabilities for domains where annotations are difficult to collect. In this work, we focus on artificial image domains such as charts, rendered documents, and webpages, which are abundant in practice yet lack large-scale human annotated reasoning datasets. We introduce COGS (COmposition-Grounded instruction Synthesis), a data-efficient framework for equipping MLLMs with advanced reasoning abilities from a small set of seed questions. The key idea is to decompose each seed question into primitive perception and reasoning factors, which can then be systematically recomposed with new images to generate large collections of synthetic question-answer pairs. Each generated question is paired with subquestions and intermediate answers, enabling reinforcement learning with factor-level process rewards. Experiments on chart reasoning show that COGS substantially improves performance on unseen questions, with the largest gains on reasoning-heavy and compositional questions. Moreover, training with a factor-level mixture of different seed data yields better transfer across multiple datasets, suggesting that COGS induces generalizable capabilities rather than dataset-specific overfitting. We further demonstrate that the framework extends beyond charts to other domains such as webpages. 

---
# The Coverage Principle: How Pre-training Enables Post-Training 

**Authors**: Fan Chen, Audrey Huang, Noah Golowich, Sadhika Malladi, Adam Block, Jordan T. Ash, Akshay Krishnamurthy, Dylan J. Foster  

**Link**: [PDF](https://arxiv.org/pdf/2510.15020)  

**Abstract**: Language models demonstrate remarkable abilities when pre-trained on large text corpora and fine-tuned for specific tasks, but how and why pre-training shapes the success of the final model remains poorly understood. Notably, although pre-training success is often quantified by cross entropy loss, cross-entropy can be a poor predictor of downstream performance. Instead, we provide a theoretical perspective on this relationship through the lens of \emph{coverage}, which quantifies the probability mass the pre-trained model places on high-quality responses and which is necessary and sufficient for post-training and test-time scaling methods such as Best-of-N to succeed. Our main results develop an understanding of \emph{the coverage principle}, a phenomenon whereby next-token prediction implicitly optimizes toward a model with good coverage. In particular, we uncover a mechanism that explains the power of coverage in predicting downstream performance: \emph{coverage generalizes faster than cross entropy}, avoiding spurious dependence on problem-dependent parameters such as the sequence length. We also study practical algorithmic interventions with provable benefits for improving coverage, including (i) model/checkpoint selection procedures, (ii) gradient normalization schemes, and (iii) test-time decoding strategies. 

---
# DeLeaker: Dynamic Inference-Time Reweighting For Semantic Leakage Mitigation in Text-to-Image Models 

**Authors**: Mor Ventura, Michael Toker, Or Patashnik, Yonatan Belinkov, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2510.15015)  

**Abstract**: Text-to-Image (T2I) models have advanced rapidly, yet they remain vulnerable to semantic leakage, the unintended transfer of semantically related features between distinct entities. Existing mitigation strategies are often optimization-based or dependent on external inputs. We introduce DeLeaker, a lightweight, optimization-free inference-time approach that mitigates leakage by directly intervening on the model's attention maps. Throughout the diffusion process, DeLeaker dynamically reweights attention maps to suppress excessive cross-entity interactions while strengthening the identity of each entity. To support systematic evaluation, we introduce SLIM (Semantic Leakage in IMages), the first dataset dedicated to semantic leakage, comprising 1,130 human-verified samples spanning diverse scenarios, together with a novel automatic evaluation framework. Experiments demonstrate that DeLeaker consistently outperforms all baselines, even when they are provided with external information, achieving effective leakage mitigation without compromising fidelity or quality. These results underscore the value of attention control and pave the way for more semantically precise T2I models. 

---
# Shakti-VLMs: Scalable Vision-Language Models for Enterprise AI 

**Authors**: Syed Abdul Gaffar Shakhadri, Kruthika KR, Kartik Basavaraj Angadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17092)  

**Abstract**: We introduce Shakti VLM, a family of vision-language models in the capacity of 1B and 4B parameters designed to address data efficiency challenges in multimodal learning. While recent VLMs achieve strong performance through extensive training data, Shakti models leverage architectural innovations to attain competitive results with fewer tokens. Key advancements include QK-Normalization for attention stability, hybrid normalization techniques, and enhanced positional encoding. A three-stage training strategy further optimizes learning efficiency. Evaluations show that Shakti-Shakti-VLM-1B and Shakti-VLM-4B excel in document understanding, Visual Reasoning, OCR extraction, and general multimodal reasoning. Our results highlight that high performance can be achieved through model design and training strategy rather than sheer data volume, making Shakti an efficient solution for enterprise-scale multimodal tasks. 

---
