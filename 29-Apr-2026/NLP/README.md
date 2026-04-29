# DV-World: Benchmarking Data Visualization Agents in Real-World Scenarios 

**Authors**: Jinxiang Meng, Shaoping Huang, Fangyu Lei, Jingyu Guo, Haoxiang Liu, Jiahao Su, Sihan Wang, Yao Wang, Enrui Wang, Ye Yang, Hongze Chai, Jinming Lv, Anbang Yu, Huangjing Zhang, Yitong Zhang, Yiming Huang, Zeyao Ma, Shizhu He, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25914)  

**Abstract**: Real-world data visualization (DV) requires native environmental grounding, cross-platform evolution, and proactive intent alignment. Yet, existing benchmarks often suffer from code-sandbox confinement, single-language creation-only tasks, and assumption of perfect intent. To bridge these gaps, we introduce DV-World, a benchmark of 260 tasks designed to evaluate DV agents across real-world professional lifecycles. DV-World spans three domains: DV-Sheet for native spreadsheet manipulation including chart and dashboard creation as well as diagnostic repair; DV-Evolution for adapting and restructuring reference visual artifacts to fit new data across diverse programming paradigms and DV-Interact for proactive intent alignment with a user simulator that mimics real-world ambiguous requirements. Our hybrid evaluation framework integrates Table-value Alignment for numerical precision and MLLM-as-a-Judge with rubrics for semantic-visual assessment. Experiments reveal that state-of-the-art models achieve less than 50% overall performance, exposing critical deficits in handling the complex challenges of real-world data visualization. DV-World provides a realistic testbed to steer development toward the versatile expertise required in enterprise workflows. Our data and code are available at \href{this https URL}{this project page}. 

---
# A paradox of AI fluency 

**Authors**: Christopher Potts, Moritz Sudhof  

**Link**: [PDF](https://arxiv.org/pdf/2604.25905)  

**Abstract**: How much does a user's skill with AI shape what AI actually delivers for them? This question is critical for users, AI product builders, and society at large, but it remains underexplored. Using a richly annotated sample of 27K transcripts from WildChat-4.8M, we show that fluent users take on more complex tasks than novices and adopt a fundamentally different interactional mode: they iterate collaboratively with the AI, refining goals and critically assessing outputs, whereas novices take a passive stance. These differences lead to a paradox of AI fluency: fluent users experience more failures than novices -- but their failures tend to be visible (a direct consequence of their engagement), they are more likely to lead to partial recovery, and they occur alongside greater success on complex tasks. Novices, by contrast, more often experience invisible failures: conversations that appear to end successfully but in fact miss the mark. Taken together, these results reframe what success with AI depends on. Individuals should adopt a stance of active engagement rather than passive acceptance. AI product builders should recognize that they are designing not just model behavior but user behavior; encouraging deep engagement, rather than friction-free experiences, will lead to more success overall. Our code and data are available at this https URL 

---
# Toward a Functional Geometric Algebra for Natural Language Semantics 

**Authors**: James Pustejovsky  

**Link**: [PDF](https://arxiv.org/pdf/2604.25902)  

**Abstract**: Distributional and neural approaches to natural language semantics have been built almost exclusively on conventional linear algebra: vectors, matrices, tensors, and the operations that accompany them. These methods have achieved remarkable empirical success, yet they face persistent structural limitations in compositional semantics, type sensitivity, and interpretability. I argue in this paper that geometric algebra (GA) -- specifically, Clifford algebras -- provides a mathematically superior foundation for semantic representation, and that a Functional Geometric Algebra (FGA) framework extends GA toward a typed, compositional semantics capable of supporting inference, transformation, and interpretability while retaining full compatibility with distributional learning and modern neural architectures. I develop the formal foundations, identify three core capabilities that GA provides and linear algebra does not, present a detailed worked example illustrating operator-level semantic contrasts, and show how GA-based operations already implicit in current transformer architectures can be made explicit and extended. The central claim is not merely increased dimensionality but increased structural organization: GA expands an $n$-dimensional embedding space into a $2^n$ multivector algebra where base semantic concepts and their higher-order interactions are represented within a single, principled algebraic framework. 

---
# From Syntax to Emotion: A Mechanistic Analysis of Emotion Inference in LLMs 

**Authors**: Bangzhao Shu, Arinjay Singh, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2604.25866)  

**Abstract**: Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, yet little is known about how emotion recognition is internally represented. In this work, we investigate the internal mechanisms of emotion recognition in LLMs using sparse autoencoders (SAEs). By analyzing sparse feature activations across layers, we identify a consistent three-phase information flow, in which emotion-related features emerge only in the final phase. We further show that emotion representations comprise both shared features across emotions and emotion-specific features. Using phase-stratified causal tracing, we identify a small set of features that strongly influence emotion predictions, and show that both their number and causal impact vary across emotions; in particular, Disgust is more weakly and diffusely represented than other emotions. Finally, we propose an interpretable and data-efficient causal feature steering method that significantly improves emotion recognition performance across multiple models while largely preserving language modeling ability, and demonstrate that these improvements generalize across multiple emotion recognition datasets. Overall, our findings provide a systematic analysis of the internal mechanisms underlying emotion recognition in LLMs and introduce an efficient, interpretable, and controllable approach for improving model performance. 

---
# Luminol-AIDetect: Fast Zero-shot Machine-Generated Text Detection based on Perplexity under Text Shuffling 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2604.25860)  

**Abstract**: Machine-generated text (MGT) detection requires identifying structurally invariant signals across generation models, rather than relying on model-specific fingerprints. In this respect, we hypothesize that while large language models excel at local semantic consistency, their autoregressive nature results in a specific kind of structural fragility compared to human writing. We propose Luminol-AIDetect, a novel, zero-shot statistical approach that exposes this fragility through coherence disruption. By applying a simple randomized text-shuffling procedure, we demonstrate that the resulting shift in perplexity serves as a principled, model-agnostic discriminant, as MGT displays a characteristic dispersion in perplexity-under-shuffling that differs markedly from the more stable structural variability of human-written text. Luminol-AIDetect leverages this distinction to inform its decision process, where a handful of perplexity-based scalar features are extracted from an input text and its shuffled version, then detection is performed via density estimation and ensemble-based prediction. Evaluated across 8 content domains, 11 adversarial attack types, and 18 languages, Luminol-AIDetect demonstrates state-of-the-art performance, with gains up to 17x lower FPR while being cheaper than prior methods. 

---
# G-Loss: Graph-Guided Fine-Tuning of Language Models 

**Authors**: Sharma Aditya, Agarwal Vinti, Kumar Rajesh  

**Link**: [PDF](https://arxiv.org/pdf/2604.25853)  

**Abstract**: Traditional loss functions, including cross-entropy, contrastive, triplet, and su pervised contrastive losses, used for fine-tuning pre-trained language models such as BERT, operate only within local neighborhoods and fail to account for the global semantic structure. We present G-Loss, a graph-guided loss function that incorporates semi-supervised label propagation to use structural relationships within the embedding manifold. G-Loss builds a document-similarity graph that captures global semantic relationships, thereby guiding the model to learn more discriminative and robust embeddings. We evaluate G-Loss on five benchmark datasets covering key downstream classification tasks: MR (sentiment analysis), R8 and R52 (topic categorization), Ohsumed (medical document classification), and 20NG (news categorization). In the majority of experimental setups, G-Loss converges faster and produces semantically coherent embedding spaces, resulting in higher classification accuracy than models fine-tuned with traditional loss functions. 

---
# Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses 

**Authors**: Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2604.25850)  

**Abstract**: Harnesses have become a central determinant of coding-agent performance, shaping how models interact with repositories, tools, and execution environments. Yet automating harness engineering is hard: a heterogeneous action space, sparse and noisy evaluation signal, multi-million-token trajectories, and edits whose effect is hard to attribute to the next round's outcomes. We introduce Agentic Harness Engineering (AHE), a framework that automates harness-level evolution by instrumenting the three stages of any engineering loop (component editing, trajectory inspection, and decision making) with matched observability pillars: (1) component observability gives every editable harness component a file-level representation so the action space is explicit and revertible; (2) experience observability distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and (3) decision observability pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes. Together, these pillars turn every edit into a falsifiable contract, so harness evolution proceeds autonomously without collapsing into trial-and-error. Empirically, ten AHE iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpassing the human-designed harness Codex-CLI (71.9%) and the self-evolving baselines ACE and TF-GRPO. The frozen harness transfers without re-evolution: on SWE-bench-verified it tops aggregate success at 12% fewer tokens than the seed, and on Terminal-Bench 2 it yields +5.1 to +10.1pp cross-family gains across three alternate model families, indicating the evolved components encode general engineering experience rather than benchmark-specific tuning. These results position observability-driven evolution as a practical pathway to keep coding-agent harnesses continually improving. 

---
# PSI-Bench: Towards Clinically Grounded and Interpretable Evaluation of Depression Patient Simulators 

**Authors**: Nguyen Khoi Hoang, Shuhaib Mehri, Tse-An Hsu, Yi-Jyun Sun, Quynh Xuan Nguyen Truong, Khoa D Doan, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2604.25840)  

**Abstract**: Patient simulators are gaining traction in mental health training by providing scalable exposure to complex and sensitive patient interactions. Simulating depressed patients is particularly challenging, as safety constraints and high patient variability complicate simulations and underscore the need for simulators that capture diverse and realistic patient behaviors. However, existing evaluations heavily rely on LLM-judges with poorly specified prompts and do not assess behavioral diversity. We introduce PSI-Bench, an automatic evaluation framework that provides interpretable, clinically grounded diagnostics of depression patient simulator behavior across turn-, dialogue-, and population-level dimensions. Using PSI-Bench, we benchmark seven LLMs across two simulator frameworks and find that simulators produce overly long, lexically diverse responses, show reduced variability, resolve emotions too quickly, and follow a uniform negative-to-positive trajectory. We also show that the simulation framework has a larger impact on fidelity than the model scale. Results from a human study demonstrate that our benchmark is strongly aligned with expert judgments. Our work reveals key limitations of current depression patient simulators and provides an interpretable, extensible benchmark to guide future simulator design and evaluation. 

---
# MAIC-UI: Making Interactive Courseware with Generative UI 

**Authors**: Shangqing Tu, Yanjia Li, Keyu Chen, Sichen Zhang, Jifan Yu, Daniel Zhang-Li, Lei Hou, Juanzi Li, Yu Zhang, Huiqin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25806)  

**Abstract**: Creating interactive STEM courseware traditionally requires HTML/CSS/JavaScript expertise, leaving barriers for educators. While generative AI can produce HTML codes, existing tools generate static presentations rather than interactive simulations, struggle with long documents, and lack pedagogical accuracy mechanisms. Furthermore, full regeneration for modifications requires 200--600 seconds, disrupting creative flow. We present MAIC-UI, a zero-code authoring system that enables educators to create and rapidly edit interactive courseware from textbooks, PPTs, and PDFs. MAIC-UI employs: (1) structured knowledge analysis with multi-modal understanding to ensure pedagogical rigor; (2) a two-stage generate-verify-optimize pipeline separating content alignment from visual refinement; and (3) Click-to-Locate editing with Unified Diff-based incremental generation achieving sub-10-second iteration cycles. A controlled lab study with 40 participants shows MAIC-UI reduces editing iterations (4.9 vs. 7.0) and significantly improves learnability and controllability compared to direct Text-to-HTML generation. A three-month classroom deployment with 53 high school students demonstrates that MAIC-UI fosters learning agency and reduces outcome disparities -- the pilot class achieved 9.21-point gains in STEM subjects compared to -2.32 points in control classes. Our code is available at this https URL. 

---
# Subliminal Steering: Stronger Encoding of Hidden Signals 

**Authors**: George Morgulis, John Hewitt  

**Link**: [PDF](https://arxiv.org/pdf/2604.25783)  

**Abstract**: Subliminal learning describes a student language model inheriting a behavioral bias by fine-tuning on seemingly innocuous data generated by a biased teacher model. Prior work has begun to characterize this phenomenon but leaves open questions about the scope of signals it can transfer, the mechanisms that explain it, and the precision with which a bias can be encoded by seemingly unrelated data. We tackle all three problems by introducing subliminal steering, a variant of subliminal learning in which the teacher's bias is implemented not via a system prompt, as in prior work, but through a steering vector trained to maximize the likelihood of a set of target samples. First, we show that subliminal steering transfers complex multi-word biases, whereas prior work focused on single-word preferences, demonstrating a large scope of subliminally transferrable signals. Second, we provide mechanistic evidence that subliminal learning transfers not only the target behavioral bias, but also the steering vector itself, localized to the layers at which the teacher was steered. Finally, we show that the bias is encoded with surprising precision. We train a new steering vector directly on the subliminally-laden dataset and find that it attains high cosine similarity with the original vector. 

---
# Unrequited Emotions: Investigating the Gaps in Motivation and Practice in Speech Emotion Recognition Research 

**Authors**: Taryn Wong, Zeerak Talat, Hanan Aldarmaki, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2604.25776)  

**Abstract**: Critical analyses of emotion recognition technology have raised ethical concerns around task validity and potential downstream impacts, urging researchers to ensure alignment between their stated motivations and practice. However, these discussions have not adequately influenced or drawn from research on speech emotion recognition (SER). We address this gap by conducting a systematic survey of SER research to uncover what stated motivations drive this work and if they align with the datasets and emotions studied. We find that while SER research identifies appealing goals, such as well-situated voice-activated systems or healthcare applications, commonly-used datasets do not reflect these proposed deployment contexts, thus presenting a gap between motivations and research practices. We argue that such gaps engender ethical concerns, and that SER research should reassert itself with concrete use-cases to prevent misinterpretations, misuse, and downstream harms. 

---
# CGU-ILALab at FoodBench-QA 2026: Comparing Traditional and LLM-based Approaches for Recipe Nutrient Estimation 

**Authors**: Wei-Chun Chen, Yu-Xuan Chen, I-Fang Chung, Ying-Jia Lin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25774)  

**Abstract**: Accurate nutrient estimation from unstructured recipe text is an important yet challenging problem in dietary monitoring, due to ambiguous ingredient terminology and highly variable quantity expressions. We systematically evaluate models spanning a wide range of representational capacity, from lexical matching methods (TF-IDF with Ridge Regression), to deep semantic encoders (DeBERTa-v3), to generative reasoning with large language models (LLMs). Under the strict tolerance criteria defined by EU Regulation 1169/2011, our empirical results reveal a clear trade-off between predictive accuracy and computational efficiency. The TF-IDF baseline achieves moderate nutrient estimation performance with near-instantaneous inference, whereas the DeBERTa-v3 encoder performs poorly under task-specific data scarcity. In contrast, few-shot LLM inference (e.g., Gemini 2.5 Flash) and a hybrid LLM refinement pipeline (TF-IDF combined with Gemini 2.5 Flash) deliver the highest validation accuracy across all nutrient categories. These improvements likely arise from the ability of LLMs to leverage pre-trained world knowledge to resolve ambiguous terminology and normalize non-standard units, which remain difficult for purely lexical approaches. However, these gains come at the cost of substantially higher inference latency, highlighting a practical deployment trade-off between real-time efficiency and nutritional precision in dietary monitoring systems. 

---
# Cross-Lingual Jailbreak Detection via Semantic Codebooks 

**Authors**: Shirin Alanova, Bogdan Minko, Sabrina Sadiekh, Evgeniy Kokuykin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25716)  

**Abstract**: Safety mechanisms for large language models (LLMs) remain predominantly English-centric, creating systematic vulnerabilities in multilingual deployment. Prior work shows that translating malicious prompts into other languages can substantially increase jailbreak success rates, exposing a structural cross-lingual security gap. We investigate whether such attacks can be mitigated through language-agnostic semantic similarity without retraining or language-specific adaptation. Our approach compares multilingual query embeddings against a fixed English codebook of jailbreak prompts, operating as a training-free external guardrail for black-box LLMs. We conduct a systematic evaluation across four languages, two translation pipelines, four safety benchmarks, three embedding models, and three target LLMs (Qwen, Llama, GPT-3.5). Our results reveal two distinct regimes of cross-lingual transfer. On curated benchmarks containing canonical jailbreak templates, semantic similarity generalizes reliably across languages, achieving near-perfect separability (AUC up to 0.99) and substantial reductions in absolute attack success rates under strict low-false-positive constraints. However, under distribution shift - on behaviorally diverse and heterogeneous unsafe benchmarks - separability degrades markedly (AUC $\approx$ 0.60-0.70), and recall in the security-critical low-FPR regime drops across all embedding models. 

---
# Backtranslation Augmented Direct Preference Optimization for Neural Machine Translation 

**Authors**: Mehrdad Ghassabi, Spehr Rajabi, Hamidreza Baradaran Kashani, Sadra Hakim, Mahshid Keivandarian  

**Link**: [PDF](https://arxiv.org/pdf/2604.25702)  

**Abstract**: Contemporary neural machine translation (NMT) systems are almost exclusively built by training on supervised parallel data. Despite the tremendous progress achieved, these systems still exhibit persistent translation errors. This paper proposes that a post-training paradigm based on reinforcement learning (RL) can effectively rectify such mistakes. We introduce a novel framework that requires only a general text corpus and an expert translator which can be either human or an AI system to provide iterative feedback. In our experiments, we focus specifically on English-to-German translation as a representative high-resource language pair. Crucially, we implement this RL-based post-training using Direct Preference Optimization (DPO). Applying our DPO-driven framework to the gemma3-1b model yields a significant improvement in translation quality, elevating it's COMET score from 0.703 to 0.747 on the English to German task. The results demonstrate that DPO offers an efficient and stable pathway for enhancing pre-trained NMT models through preference-based post-training. 

---
# CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG 

**Authors**: Nayeon Lee, Jiwoo Song, Byeongcheol Kang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25676)  

**Abstract**: Multilingual retrieval-augmented generation (mRAG) is often implemented within a fixed retrieval space, typically via query or document translation or multilingual embedding vector representations. However, this approach may be inadequate for culturally grounded queries, in which retrieval-condition misalignment may occur. Even strong retrievers and generators may struggle to produce culturally relevant answers when sourcing evidence from inappropriate linguistic or regional contexts. To this end, we introduce CORAL (COntext-aware Retrieval with Agentic Loop, an adaptive retrieval methodology for mRAG that enables iterative refinement of both the retrieval space (corpora) and the retrieval probe (query) based on the quality of the evidence. The overall process includes: (1) selecting corpora, (2) retrieving documents, (3) critiquing evidence for relevance and cultural alignment, and (4) checking sufficiency. If the retrieved documents are insufficient to answer the query correctly, the system (5) reselects corpora and rewrites the query. Across two cultural QA benchmarks, CORAL achieves up to a 3.58%p accuracy improvement on low-resource languages relative to the strongest baselines. 

---
# Modeling Human-Like Color Naming Behavior in Context 

**Authors**: Yuqing Zhang, Ecesu Ürker, Tessa Verhoef, Gemma Boleda, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2604.25674)  

**Abstract**: Modeling the emergence of human-like lexicons in computational systems has advanced through the use of interacting neural agents, which simulate both learning and communicative pressures. The NeLLCom-Lex framework (Zhang et al., 2025) allows neural agents to develop pragmatic color naming behavior and human-like lexicons through supervised learning (SL) from human data and reinforcement learning (RL) in referential games. Despite these successes, the lexicons that emerge diverge systematically from human color categories, producing highly non-convex regions in color space, which contrast with the convexity typical of human categories. To address this, we introduce two factors, upsampling rare color terms during SL and multi-listener RL interactions, and adopt a convexity measure to quantify geometric coherence. We find that upsampling improves lexical diversity and system-level informativeness of the color lexicon, while many-listener setups promote more convex color categories. The combination of moderate upsampling and multiple listeners produces lexicons most similar to human systems. 

---
# LLM-ReSum: A Framework for LLM Reflective Summarization through Self-Evaluation 

**Authors**: Huyen Nguyen, Haoxuan Zhang, Yang Zhang, Junhua Ding, Haihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.25665)  

**Abstract**: Reliable evaluation of large language model (LLM)-generated summaries remains an open challenge, particularly across heterogeneous domains and document lengths. We conduct a comprehensive meta-evaluation of 14 automatic summarization metrics and LLM-based evaluators across seven datasets spanning five domains, covering documents from short news articles to long scientific, governmental, and legal texts (2K-27K words) with over 1,500 human-annotated summaries. Our results show that traditional lexical overlap metrics (e.g., ROUGE, BLEU) exhibit weak or negative correlation with human judgments, while task-specific neural metrics and LLM-based evaluators achieve substantially higher alignment, especially for linguistic quality assessment. Leveraging these findings, we propose LLM-ReSum, a self-reflective summarization framework that integrates LLM-based evaluation and generation in a closed feedback loop without model finetuning. Across three domains, LLM-ReSum improves low-quality summaries by up to 33% in factual accuracy and 39% in coverage, with human evaluators preferring refined summaries in 89% of cases. We additionally introduce PatentSumEval, a new human-annotated benchmark for legal document summarization comprising 180 expert-evaluated summaries. All code and datasets will be released in GitHub. 

---
# Progressing beyond Art Masterpieces or Touristic Clichés: how to assess your LLMs for cultural alignment? 

**Authors**: António Branco, João Silva, Nuno Marques, Luis Gomes, Ricardo Campos, Raquel Sequeira, Sara Nerea, Rodrigo Silva, Miguel Marques, Rodrigo Duarte, Artur Putyato, Diogo Folques, Tiago Valente  

**Link**: [PDF](https://arxiv.org/pdf/2604.25654)  

**Abstract**: Although the cultural (mis)alignment of Large Language Models (LLMs) has attracted increasing attention -- often framed in terms of cultural bias -- until recently there has been limited work on the design and development of datasets for cultural assessment. Here, we review existing approaches to such datasets and identify their main limitations. To address these issues, we propose design guidelines for annotators and report on the construction of a dataset built according to these principles. We further present a series of contrastive experiments conducted with this dataset. The results demonstrate that our design yields test sets with greater discriminative power, effectively distinguishing between models specialized for a given culture and those that are not, ceteris paribus. 

---
# WhisperPipe: A Resource-Efficient Streaming Architecture for Real-Time Automatic Speech Recognition 

**Authors**: Erfan Ramezani, Mohammad Mahdi Giahi, Mohammad Erfan Zarabadipour, Amir Reza Yosefian, Hamid Ghadiri  

**Link**: [PDF](https://arxiv.org/pdf/2604.25611)  

**Abstract**: Real-time automatic speech recognition (ASR) systems face a fundamental trade-off between transcription accuracy and computational efficiency, particularly when deploying large-scale transformer models like Whisper. Existing streaming approaches either sacrifice accuracy through aggressive chunking or incur prohibitive memory costs through unbounded context accumulation. We present WhisperPipe, a novel streaming architecture that achieves bounded memory consumption while maintaining transcription quality through three key innovations a hybrid Voice Activity Detection (VAD) pipeline combining Silero VAD with energy-based filtering to reduce false activations by 34%, a dynamic buffering mechanism with overlapping context windows that prevents information loss at segment boundaries, and an adaptive processing strategy that balances latency and accuracy based on speech characteristics. Evaluated on 2.5 hours of diverse audio data, WhisperPipe demonstrates a median end-to-end latency of 89ms (90th percentile: 142ms) while consuming 48% less peak GPU memory and 80.9% lower average GPU utilization compared to baseline Whisper implementations. The system maintains stable memory usage over extended sessions, with zero growth rate across 150-minute continuous operation. Comparative analysis against related work shows that WhisperPipe achieves competitive accuracy (WER within 2% of offline Whisper) while operating at 3-5x lower latency than existing streaming solutions. The architecture's modular design enables deployment across resource-constrained environments, from edge devices to cloud infrastructure. Our results demonstrate that careful architectural design can reconcile the competing demands of real-time responsiveness and model sophistication in production ASR systems. 

---
# Bye Bye Perspective API: Lessons for Measurement Infrastructure in NLP, CSS and LLM Evaluation 

**Authors**: David Hartmann, Manuel Tonneau, Angelie Kraft, LK Seiling, Dimitri Staufer, Pieter Delobelle, Jan Fillies, Anna Ricarda Luther, Jan Batzner, Mareike Lisker  

**Link**: [PDF](https://arxiv.org/pdf/2604.25580)  

**Abstract**: The closure of Perspective API at the end of 2026 discards what has functioned as the de facto standard for automated toxicity measurement in NLP, CSS, and LLM evaluation research. We document the structural dependence that the communities built on this single proprietary tool and discuss how this dependence caused epistemic problems that have affected - and will likely continue to affect - collective research efforts. Perspective's model was periodically updated without versioning or disclosure, its annotation structure reflected a single corporate operationalisation of a contested concept, and its scores were used simultaneously as an evaluation target and an evaluation standard. Its closure leaves behind non-updatable benchmarks, irreproducible results, and ultimately a field at risk of perpetuating these issues by turning to closed-source LLMs. We use Perspective's announced termination as an opportunity to call for an independent, valid, adaptable, and reproducible toxicity and hate speech measurement infrastructure, with the technical and governance requirements outlined in this paper. 

---
# Marco-MoE: Open Multilingual Mixture-of-Expert Language Models with Efficient Upcycling 

**Authors**: Fan Jiang, Yu Zhao, Chenyang Lyu, Tianqi Shi, Yichao Du, Feihu Jiang, Longyue Wang, Weihua Luo  

**Link**: [PDF](https://arxiv.org/pdf/2604.25578)  

**Abstract**: We present Marco-MoE, a suite of fully open multilingual sparse Mixture-of-Experts (MoE) models. Marco-MoE features a highly sparse design in which only around 5\% of the total parameters are activated per input token. This extreme sparsity, combined with upcycling from dense models, enables efficient pre-training on 5T tokens. Our models surpass similarly-sized competitors on English and multilingual benchmarks, achieving a best-in-class performance-to-compute ratio. We further post-train these models to create Marco-MoE-\textsc{Instruct} variants, which surpass the performance of competing models possessing $3$--$14\times$ more activated parameters. Our analysis reveals that Marco-MoE learns structured expert activation patterns shared across related languages, while maintaining highly specialized utilization for linguistically isolated ones. We further show that Marco-MoE allows for scalable language expansion without the interference typical of dense models. To support the community, we disclose our full training datasets, recipes, and model weights. 

---
# From Chatbots to Confidants: A Cross-Cultural Study of LLM Adoption for Emotional Support 

**Authors**: Natalia Amat-Lefort, Mert Yazan, Amanda Cercas Curry, Flor Miriam Plaza-del-Arco  

**Link**: [PDF](https://arxiv.org/pdf/2604.25525)  

**Abstract**: Large Language Models (LLMs) are increasingly used not only for instrumental tasks, but as always-available and non-judgmental confidants for emotional support. Yet what drives adoption and how users perceive emotional support interactions across countries remains unknown. To address this gap, we present the first large-scale cross-cultural study of LLM use for emotional support, surveying 4,641 participants across seven countries (USA, UK, Germany, France, Spain, Italy, and The Netherlands). Our results show that adoption rates vary dramatically across countries (from 20% to 59%). Using mixed models that separate cultural effects from demographic composition, we find that: Being aged 25-44, religious, married, and of higher socioeconomic status are predictors of positive perceptions (trust, usage, perceived benefits), with socioeconomic status being the strongest. English-speaking countries consistently show more positive perceptions than Continental European countries. We further collect a corpus of 731 real multilingual prompts from user interactions, showing that users mainly seek help for loneliness, stress, relationship conflicts, and mental health struggles. Our findings reveal that LLM emotional support use is shaped by a complex sociotechnical landscape and call for a broader research agenda examining how these systems can be developed, deployed, and governed to ensure safe and informed access. 

---
# From World-Gen to Quest-Line: A Dependency-Driven Prompt Pipeline for Coherent RPG Generation 

**Authors**: Dominik Borawski, Marta Szulc, Robert Chudy, Małgorzata Giedrowicz, Piotr Mironowicz  

**Link**: [PDF](https://arxiv.org/pdf/2604.25482)  

**Abstract**: Large Language Models (LLMs) have shown strong potential for narrative generation, but their use in complex, multi-layered role-playing game (RPG) worlds is still limited by issues of coherence, controllability, and structural consistency. This paper explores a dependency-aware, multi-stage prompt pipeline for procedural RPG content generation that models narrative dependencies through structured intermediate representations. The approach decomposes generation into sequential stages: world building, non-player character creation, player character creation, campaign-level quest planning, and quest expansion. Each stage conditions on structured JSON outputs from previous stages. By enforcing schemas and explicit data flow, the pipeline reduces narrative drift, limits hallucinations, and supports scalable creation of interconnected narrative elements. The system is evaluated qualitatively through human-centered analysis across multiple independent runs. Outputs are assessed using criteria such as structural completeness, internal consistency, narrative coherence, diversity, and actionability. Results show that the pipeline consistently generates logically sound and structurally valid RPG content, without quality degradation as complexity increases. Separating high-level campaign planning from detailed quest expansion improves both global structure and local storytelling. These findings suggest that dependency-aware prompt pipelines with structured intermediate representations are an effective design pattern for LLM-based procedural content generation. This approach may also generalize to other domains requiring sequential reasoning over evolving contextual states. 

---
# An Investigation of Linguistic Biases in LLM-Based Recommendations 

**Authors**: Nitin Venkateswaran, Jason Ang, Deep Adhikari, Tarun Krishna Dasari  

**Link**: [PDF](https://arxiv.org/pdf/2604.25456)  

**Abstract**: We investigate linguistic biases in LLM-based restaurant and product recommendations given prompts varying across Southern American English (AE), Indian English (IE), and Code-Switched Hindi-English dialects, using the Yelp Open dataset (Yelp Inc., 2023) and Walmart product reviews dataset (PromptCloud,2020). We add lists of restaurant and product names balanced by cuisine type and product category to the prompts given to the LLM, and we zero-shot prompt the LLMs in a cold-start setting to select the top-20 restaurant and product recommendations from these lists for each of the dialect-varied prompts. We prompt LLMs using different list samples across 20 seeds for better generalization, and aggregate per cuisine-type and per category response counts for each seed, question/prompt, and LLM model. We run mixed-effects regression models for each model family and topic (restaurant/product) with the aggregate response counts as the dependent, and conduct likelihood ratio tests for the fixed effects with post-hoc pairwise testing of estimated marginal means differences, to investigate group-level differences in recommendation counts by model size and dialect type. Results show that dialect plays a role in the type of restaurant selected across the models tested with the mistral-small-3.1 model and both the llama-3.1 family models tested showing more sensitivity to Indian English and Code-Switched prompts. In terms of product recommendations, the llama-3.1-70B-model is particularly sensitive to Code-Switched prompts in four out of seven categories, and more beauty and home category recommendations are seen when using the Indian English and Code-Switched prompts for larger and smaller models, respectively. No broad trends are seen in the model-size based differences, with differing recommendations based on model sizes conditioned by the type of dialect. 

---
# Benchmarking Logistic Regression, SVM, and LightGBM Against BiLSTM with Attention for Sentiment Analysis on Indonesian Product Reviews 

**Authors**: Razin Hafid Hamdi, Ivana Margareth Hutabarat, Hanna Gresia Sinaga, Luluk Muthoharoh, Ardika Satria, Martin C.T. Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25452)  

**Abstract**: Sentiment analysis of product reviews on e-commerce platforms plays a critical role in automatically understanding customer satisfaction and providing actionable insights for sellers seeking to improve product quality. This paper presents a comprehensive benchmarking study comparing a Machine Learning (ML) approach via the PyCaret AutoML framework against a Deep Learning (DL) approach based on a Bidirectional Long Short-Term Memory (BiLSTM) architecture with an Attention mechanism for binary sentiment classification on Indonesian product reviews. The dataset comprises 19,728 samples balanced equally between positive and negative reviews. For the ML approach, three prominent algorithms were evaluated via 10-fold stratified cross-validation: Logistic Regression (LR), Support Vector Machine (SVM) with a linear kernel, and Light Gradient Boosting Machine (LightGBM). Logistic Regression achieved the best ML performance with an accuracy of 97.26\% and an F1-score of 97.26\%. The BiLSTM with Attention model, evaluated on 3,946 held-out test samples, achieved an accuracy of 97.24\% and an F1-score of 97.24\%. These comparative results demonstrate that traditional ML algorithms with proper preprocessing and feature extraction can compete closely with, and even marginally outperform, more complex sequential DL architectures on high-dimensional datasets, while simultaneously offering greater computational efficiency. 

---
# Navigating Global AI Regulation: A Multi-Jurisdictional Retrieval-Augmented Generation System 

**Authors**: Courtney Ford, Ojas Rane, Susan Leavy  

**Link**: [PDF](https://arxiv.org/pdf/2604.25448)  

**Abstract**: Navigating AI regulation across jurisdictions is increasingly difficult for policymakers, legal professionals, and researchers. To address this, we present a multi-jurisdictional Retrieval-Augmented Generation system for global AI regulation. Our corpus includes 242 documents across 68 jurisdictions, ranging from formal legislation like the EU AI Act to unstructured policy documents such as national AI strategies. The system makes three technical contributions: type-specific chunking that preserve legal structure across heterogenous documents; conditional retrieval routing with entity detection and metadata for legal citations; and priority-based re-ranking to boost enacted legislation over policy and secondary sources. Evaluation of 50 queries reveals strong performance across both single-entity and multi-jurisdictional questions, achieving 0.87 average faithfulness and 0.84 average answer relevancy. Single-entity queries achieve 0.86 average faithfulness and 0.92 average answer relevancy, while multi-jurisdictional comparison queries achieve 0.88 average faithfulness and 0.75 average answer relevancy. These findings highlight the effectiveness of domain-specific retrieval strategies for navigating complex, heterogenous regulatory corpora. 

---
# One Refiner to Unlock Them All: Inference-Time Reasoning Elicitation via Reinforcement Query Refinement 

**Authors**: Yixiao Zhou, Dongzhou Cheng, zhiliang wu, Yi Yang, Yu Cheng, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2604.25444)  

**Abstract**: Large Language Models (LLMs) often fail to utilize their latent reasoning capabilities due to a distributional mismatch between ambiguous human inquiries and the structured logic required for machine activation. Existing alignment methods either incur prohibitive $O(N)$ costs by fine-tuning each model individually or rely on static prompts that fail to resolve query-level structural complexity. In this paper, we propose ReQueR (\textbf{Re}inforcement \textbf{Que}ry \textbf{R}efinement), a modular framework that treats reasoning elicitation as an inference-time alignment task. We train a specialized Refiner policy via Reinforcement Learning to rewrite raw queries into explicit logical decompositions, treating frozen LLMs as the environment. Rooted in the classical Zone of Proximal Development from educational psychology, we introduce the Adaptive Solver Hierarchy, a curriculum mechanism that stabilizes training by dynamically aligning environmental difficulty with the Refiner's evolving competence. ReQueR yields consistent absolute gains of 1.7\%--7.2\% across diverse architectures and benchmarks, outperforming strong baselines by 2.1\% on average. Crucially, it provides a promising paradigm for one-to-many inference-time reasoning elicitation, enabling a single Refiner trained on a small set of models to effectively unlock reasoning in diverse unseen models. Code is available at this https URL. 

---
# Do LLMs Capture Embodied Cognition and Cultural Variation? Cross-Linguistic Evidence from Demonstratives 

**Authors**: Yu Wang, Emmanuele Chersoni, Chu-Ren Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25423)  

**Abstract**: Do large language models (LLMs) truly acquire embodied cognition and cultural conventions from text? We introduce demonstratives, fundamental spatial expressions like "this/that" in English and "zhè/nà" in Chinese, as a novel probe for grounded knowledge. Using 6,400 responses from 320 native speakers, we establish a human baseline: English speakers reliably distinguish proximal-distal referents but struggle with perspective-taking, while Chinese speakers switch perspectives fluently but tolerate distal ambiguity. In contrast, five state-of-the-art LLMs fail to inherently understand the proximal-distal contrast and show no cultural differences, defaulting to English-centric reasoning. Our study contributes (i) a new task, based on demonstratives, as a new lens for evaluating embodied cognition and cultural conventions; (ii) empirical evidence of cross-cultural asymmetries in human interpretation; (iii) a new perspective on the egocentric-sociocentric debate, showing both orientations coexist but vary across languages; and (iv) a call to address individual variation in future model design. 

---
# Scaling Probabilistic Transformer via Efficient Cross-Scale Hyperparameter Transfer 

**Authors**: Penghao Kuang, Haoyi Wu, Kewei Tu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25409)  

**Abstract**: Probabilistic Transformer (PT), a white-box probabilistic model for contextual word representation, has demonstrated substantial similarity to standard Transformers in both computational structure and downstream task performance on small models and small to medium sized datasets. However, PT is less robust to hyperparameter choices than standard Transformers, making it harder to scale efficiently. In this work, we follow Maximal Update Parametrization (muP) to rescale PT's parameters, so that hyperparameters optimized on small models can be transferred to larger models without additional tuning. With this approach, we successfully scale PT to models with up to 0.4B parameters. Experiments show that PT consistently outperforms standard transformer under the same parameter budget on Masked Language Modeling (MLM) tasks. We hope this work will contribute to the practical deployment of probabilistic models at substantially larger scales in the future. 

---
# Benchmarking PyCaret AutoML Against IndoBERT Fine-Tuning for Sentiment Analysis on Indonesian IKN Twitter Data 

**Authors**: Mutia Alfi Mayzaroh, Dwi Fitria Ningsih, Nindi Destriani, Martin C.T. Manullang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25392)  

**Abstract**: This paper benchmarks a classical machine learning approach based on PyCaret AutoML against a deep learning approach based on IndoBERT fine-tuning for binary sentiment analysis of Indonesian-language Twitter comments related to Ibu Kota Nusantara (IKN). The dataset contains 1,472 manually labeled samples, consisting of 780 negative and 692 positive comments. In the machine learning setting, Logistic Regression, Naive Bayes, and Support Vector Machine were evaluated using 10-fold cross-validation, with Logistic Regression achieving the best performance among the classical models at 77.57% accuracy and 77.17% F1-score. In the deep learning setting, the indobenchmark/indobert-base-p1 model was fine-tuned for five epochs and achieved 89.59% test accuracy and 89.37% F1-score. The results show that IndoBERT substantially outperforms the machine learning baselines, highlighting the effectiveness of Transformer-based contextual representations for informal Indonesian social media text. 

---
# Wiki Dumps to Training Corpora: South Slavic Case 

**Authors**: Mihailo Škorić  

**Link**: [PDF](https://arxiv.org/pdf/2604.25384)  

**Abstract**: This paper presents a methodology for transforming raw Wikimedia dumps into quality textual corpora for seven South Slavic languages. The work is divided into two major phases. The first involves extracting and cleaning text from raw dumps of Wikipedia, Wikisource, Wikibooks, Wikinews, and Wikiquote, where available. This step requires careful handling of raw wiki markup to isolate, first of all, textual articles, and then usable natural language text within them. The second phase addresses the challenge of suspicious or low-quality articles, which are often generated from databases or structured knowledge bases. These articles are characterised by repetitive patterns, generic phrasing, and minimal to no original content. To mitigate their impact, a n-gram-based filtering strategy was employed to detect high levels of textual redundancy between articles and then remove such articles from the corpora entirely. The resulting datasets aim to provide linguistically rich texts suitable for training language models or conducting comparative research across South Slavic languages. By combining systematic extraction with quality control, this work contributes to the creation of reliable, high-information corpora that reflect authentic language use and cultural context. While focused on the South Slavic case in the paper, the approach is mostly language-agnostic and can be generalised to other languages and language families. 

---
# Language corpora for the Dutch medical domain 

**Authors**: B. van Es  

**Link**: [PDF](https://arxiv.org/pdf/2604.25374)  

**Abstract**: \textbf{Background:} Dutch medical corpora are scarce, limiting NLP development. \\ \textbf{Methods:} We translated English datasets, identified medical text in generic corpora, and extracted open Dutch medical resources. \\ \textbf{Results:} The resulting corpus comprises $\pm$ 35 billion tokens across the medical domain in about 100 million documents, freely available on Hugging Face. \\ \textbf{Conclusion:} This work establishes the first large-scale Dutch medical language corpus for pre-training and downstream NLP tasks. 

---
# The Structured Output Benchmark: A Multi-Source Benchmark for Evaluating Structured Output Quality in Large Language Models 

**Authors**: Abhinav Kumar Singh, Harsha Vardhan Khurdula, Yoeven D Khemlani, Vineet Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2604.25359)  

**Abstract**: Large Language Models are increasingly being deployed to extract structured data from unstructured and semi-structured sources: parsing invoices, medical records, and converting PDF documents to database entries. Yet existing benchmarks for structured output generation either focus on schema compliance alone, or evaluate value correctness within a single source domain. We introduce SOB (The Structured Output Benchmark), a multi-source benchmark spanning three source modalities: native text, images, and audio conversations. All models receive a text-normalized representation of their context regardless of source modality; this deliberate design isolates structured-output capability from raw vision or speech-processing quality, ensuring a fair, source-agnostic comparison. Our benchmark comprises 5,000 text evaluation records derived from multi-hop QA drawn from a 25,091-record full corpus, 209 image records from OCR-processed PDFs across seven document types including multi-column layouts, dense tables, scanned historical documents, small-print text, and mathematical typesetting, and 115 audio records from the AMI corpus. Each record pairs a natural-language question with a JSON schema that the model must follow and a ground-truth answer verified against the source context. We evaluate 21 frontier and open-weight models across three source domains and seven metrics. Our results reveal a consistent pattern: models achieve near-perfect schema compliance, yet the best Value Accuracy, measured by exact leaf-value match, reaches only 83.0% on text, 67.2% on images, and 23.7% on audio, where longer context makes extraction substantially harder. We release the dataset, evaluation pipeline, and all related code. 

---
# Faithfulness-QA: A Counterfactual Entity Substitution Dataset for Training Context-Faithful RAG Models 

**Authors**: Li Ju, Junzhe Wang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25313)  

**Abstract**: Retrieval-Augmented Generation (RAG) models frequently produce answers grounded in parametric memory rather than the retrieved context, undermining the core promise of retrieval augmentation. A fundamental obstacle to fixing this unfaithfulness is the lack of training data that explicitly requires models to prefer context over internal knowledge. We introduce Faithfulness-QA, a large-scale dataset of 99,094 samples constructed through counterfactual entity substitution. Starting from two established extractive QA benchmarks--SQuAD and TriviaQA--we automatically identify answer-bearing named entities in each context, replace them with type-consistent alternatives drawn from a curated bank of 76,953 entities, and thereby manufacture controlled knowledge conflicts between context and parametric memory. Rigorous quality filtering ensures 100% pass rates across four automated checks on random 200-sample audits. We release the full dataset, the construction pipeline, and a typed entity bank covering eight named entity categories. Faithfulness-QA is designed as a training resource for attention-based faithfulness objectives and as an evaluation benchmark for measuring context-grounding behavior in RAG systems. Data and code are available at this https URL. 

---
# LegalMidm: Use-Case-Driven Legal Domain Specialization for Korean Large Language Model 

**Authors**: Youngjoon Jang, Chanhee Park, Hyeonseok Moon, Young-kyoung Ham, Jiwon Moon, Jinhyeon Kim, JuKyung Jung, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2604.25297)  

**Abstract**: In recent years, the rapid proliferation of open-source large language models (LLMs) has spurred efforts to turn general-purpose models into domain specialists. However, many domain-specialized LLMs are developed using datasets and training protocols that are not aligned with the nuanced requirements of real-world applications. In the legal domain, where precision and reliability are essential, this lack of consideration limits practical utility. In this study, we propose a systematic training framework grounded in the practical needs of the legal domain, with a focus on Korean law. We introduce LegalMidm, a Korean legal-domain LLM, and present a methodology for constructing high-quality, use-case-driven legal datasets and optimized training pipelines. Our approach emphasizes collaboration with legal professionals and rigorous data curation to ensure relevance and factual accuracy, and demonstrates effectiveness in key legal tasks. 

---
# Learning from Medical Entity Trees: An Entity-Centric Medical Data Engineering Framework for MLLMs 

**Authors**: Jianghang Lin, Haihua Yang, Deli Yu, Kai Wu, Kai Ye, Jinghao Lin, Zihan Wang, Yuhang Wu, Liujuan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2604.25296)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown transformative potential in medical applications, yet their performance is hindered by conventional data curation strategies that rely on coarse-grained partitioning by modality or department. Such fragmented approaches fail to capture the hierarchical and interconnected nature of clinical medical knowledge, limiting the models' ability to perform fine-grained recognition and complex reasoning. In this paper, we propose a novel Entity-Centric Medical Data Engineering framework. We automatically extract entities from authoritative medical literature to construct a Medical Entity Tree (MET), a hierarchical structure that systematically encodes diseases, anatomical structures, modalities, and symptoms into a unified knowledge repository. Building upon the MET, we propose an advanced data engine that includes: (1) node-guided retrieval to anchor raw data to specific medical concepts, (2) a two-stage hybrid filtering and alignment pipeline to ensure precise visual-semantic correspondence, and (3) knowledge-aware data synthesis to generate enriched captions and targeted reasoning VQA pairs, leveraging structural constraints. Extensive evaluations across six medical benchmarks demonstrate that our approach significantly enhances the medical capabilities of general-purpose MLLMs, improving their ability to handle complex clinical queries and achieve state-of-the-art performance in diverse medical contexts. 

---
# Below-Chance Blindness: Prompted Underperformance in Small LLMs Produces Positional Bias Rather than Answer Avoidance 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2604.25249)  

**Abstract**: Detecting sandbagging--the deliberate underperformance on capability evaluations--is an open problem in AI safety. We tested whether symptom validity testing (SVT) logic from clinical malingering detection could identify sandbagging through below-chance performance (BCB) on forced-choice items. In a pre-registered pilot at the 7-9 billion parameter instruction-tuned scale (3 models, 4 MMLU-Pro domains, 4 conditions, 500 items per cell, 24,000 total trials), the plausibility gate failed. Zero of 12 model-domain cells showed significant below-chance performance under sandbagging instruction. Exploratory analyses revealed three qualitatively distinct failure modes. Qwen-2.5-7B and Phi-3.5-mini largely ignored the sandbagging instruction, with 62-88% response identity with the honest baseline. Llama-3-8B complied substantially but implemented underperformance as a positional heuristic, collapsing its response distribution onto middle-alphabet options (E at 31.8%, F at 26.1%) regardless of where the correct answer fell. This produced accuracy boosts of up to 33 percentage points when the correct answer coincidentally occupied the model's preferred position. An explicit anti-task instruction ("pick the least likely answer") drove two of three models below chance, with accuracy as low as 0.024. The capability for answer-aware avoidance therefore exists but is not activated by "deliberately underperform." BCB did not fail as a logical marker of answer-aware avoidance. It was not observed in this regime because the model showing the largest behavioural shift exhibited behaviour consistent with a position-dominant response policy rather than content-aware answer avoidance. We propose that positional-distribution shift may be a more effective behavioural signature than below-chance accuracy for detecting prompted underperformance at this model scale. 

---
# BARRED: Synthetic Training of Custom Policy Guardrails via Asymmetric Debate 

**Authors**: Arnon Mazza, Elad Levi  

**Link**: [PDF](https://arxiv.org/pdf/2604.25203)  

**Abstract**: Deploying guardrails for custom policies remains challenging, as generic safety models fail to capture task-specific requirements, while prompting LLMs suffers from inconsistent boundary-case performance and high inference costs. Training custom classifiers achieves both accuracy and efficiency, yet demands substantial labeled data that is costly to obtain. We present BARRED (Boundary Alignment Refinement through REflection and Debate), a framework for generating faithful and diverse synthetic training data using only a task description and a small set of unlabeled examples. Our approach decomposes the domain space into dimensions to ensure comprehensive coverage, and employs multi-agent debate to verify label correctness, yielding a high-fidelity training corpus. Experiments across diverse custom policies demonstrate that small language models finetuned on our synthetic data consistently outperform state-of-the-art proprietary LLMs (including reasoning models) and dedicated guardrail models. Ablation studies confirm that both dimension decomposition and debate-based verification are critical for ensuring the diversity and label fidelity required for effective fine-tuning. The BARRED framework eliminates the reliance on extensive human annotation, offering a scalable solution for accurate custom guardrails. 

---
# CroSearch-R1: Better Leveraging Cross-lingual Knowledge for Retrieval-Augmented Generation 

**Authors**: Rui Qi, Fengran Mo, Sijin Lu, Yufeng Chen, Jian-Yun Nie, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25182)  

**Abstract**: A multilingual collection may contain useful knowledge in other languages to supplement and correct the facts in the original language for Retrieval-Augmented Generation (RAG). However, the vanilla approach that simply concatenates multiple pieces of knowledge from different languages into the context may fail to improve effectiveness due to the potential disparities across languages. To better leverage multilingual knowledge, we propose CroSearch-R1, a search-augmented reinforcement learning framework to integrate multilingual knowledge into the Group Relative Policy Optimization (GRPO) process. In particular, the approach adopts a multi-turn retrieval strategy with cross-lingual knowledge integration to dynamically align the knowledge from other languages as supplementary evidence into a unified representation space. Furthermore, we introduce a multilingual rollout mechanism to optimize reasoning transferability across languages. Experimental results demonstrate that our framework effectively leverages cross-lingual complementarity and improves the effectiveness of RAG with multilingual collections. 

---
# Frictive Policy Optimization for LLMs: Epistemic Intervention, Risk-Sensitive Control, and Reflective Alignment 

**Authors**: James Pustejovsky, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2604.25136)  

**Abstract**: We propose Frictive Policy Optimization (FPO), a framework for learning language model policies that regulate not only what to say, but when and how to intervene in order to manage epistemic and normative risk. Unlike standard alignment methods that optimize surface-level preference or task utility, FPO treats clarification, verification, challenge, redirection, and refusal as explicit control actions whose purpose is to shape the evolution of belief, commitment, and uncertainty over time. We formalize alignment as a risk-sensitive epistemic control problem in which intervention decisions are selected based on their expected effect on downstream epistemic quality rather than on immediate reward alone. We introduce a compact taxonomy of frictive interventions, a structured friction functional that operationalizes multiple alignment failure modes, and a unified family of FPO methods spanning reward shaping, preference pairing, group-relative ranking, and risk-conditioned trust regions. We further propose an evaluation framework that measures epistemic competence directly through clarification behavior, calibration, contradiction repair, refusal proportionality, and information efficiency. Together, these results provide a formal and algorithmic foundation for learning agents that are aligned not only in outcome, but in epistemic conduct. 

---
# FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments 

**Authors**: Amir Saeidi, Venkatesh Mishra, Souradeep Mukhopadhyay, Gaowen Liu, Ali Payani, Jayanth Srinivasa, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2604.25135)  

**Abstract**: Large Language Models are being increasingly deployed as the decision-making core of autonomous agents capable of effecting change in external environments. Yet, in conversational benchmarks, which simulate real-world customer-centric issue resolution scenarios, these agents frequently fail due to the cascading effects of incorrect decision-making. These challenges are particularly pronounced for open-source LLMs with smaller parameter sizes, limited context windows, and constrained inference budgets, which contribute to increased error accumulation in agentic settings. To tackle these challenges, we present the Failure-Aware Meta-Agentic (FAMA) framework. FAMA operates in two stages: first, it analyzes failure trajectories from baseline agents to identify the most prevalent errors; second, it employs an orchestration mechanism that activates a minimal subset of specialized agents tailored to address these failures by injecting a targeted context for the tool-use agent before the decision-making step. Experiments across open-source LLMs demonstrate performance gains up to 27% across evaluation modes over standard baselines. These results highlight that targeted curation of context through specialized agents to address common failures is a valuable design principle for building reliable, multi-turn tool-use LLM agents that simulate real-world conversational scenarios. 

---
# Korean aegyo speech shows systematic F1 increase to signal childlike qualities 

**Authors**: Ji-eun Kim, Volker Dellwo  

**Link**: [PDF](https://arxiv.org/pdf/2604.25133)  

**Abstract**: Korean aegyo is a socially recognized childlike speaking style used predominantly in romantic interactions among adults. This study examined vowel space modification in aegyo by analyzing formant frequencies from twelve Seoul Korean speakers who produced identical scripts in aegyo and non-aegyo styles. Results show that aegyo speech features a significant increase in F1 values across vowels and selective fronting of front vowels, leading to vowel space expansion but mainly a shift to higher F1. These findings suggest that adult speakers stylize childlike speech by imitating the shorter vocal tract of children, mainly through global vowel lowering and partial fronting. 

---
# What Makes Good Instruction-Tuning Data? An In-Context Learning Perspective 

**Authors**: Guangzeng Han, Xiaolei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.25132)  

**Abstract**: Instruction-tuning datasets often contain substantial redundancy and low-quality samples, necessitating effective data selection methods. We propose an instruction data selection framework based on weighted in-context influence (wICI), which measures how effectively each candidate example reduces instruction-following difficulty for semantically related peers. Through systematic experiments, we address three key questions: what constitutes effective instruction tuning data from an in-context perspective, whether sample difficulty correlates with in-context influence, and how in-context influence translates to instruction tuning effectiveness. Experiments across multiple models and benchmarks demonstrate that our method consistently outperforms existing baselines under constrained data budgets, while empirically showing that sample difficulty negatively correlates with in-context influence. 

---
# LongSumEval: Question-Answering Based Evaluation and Feedback-Driven Refinement for Long Document Summarization 

**Authors**: Huyen Nguyen, Haoxuan Zhang, Yang Zhang, Haihua Chen, Junhua Ding  

**Link**: [PDF](https://arxiv.org/pdf/2604.25130)  

**Abstract**: Evaluating long document summaries remains the primary bottleneck in summarization research. Existing metrics correlate weakly with human judgments and produce aggregate scores without explaining deficiencies or guiding improvement, preventing effective refinement in applications requiring verifiable accuracy. We introduce LongSumEval, a unified framework bridging evaluation and generation through structured question-answering feedback. The framework operationalizes summary quality as answerability and factual alignment of question-answer pairs, generating interpretable scores and actionable feedback that identifies coverage gaps and factual inconsistencies. This resolves the misalignment where evaluation operates independently of generation objectives. Meta-evaluation of our QA-based evaluation module across seven benchmarks demonstrates substantially stronger agreement with human judgments compared to established metrics. Structured feedback enables significant quality improvements through self-refinement without retraining. By demonstrating that evaluation feedback can serve as executable instructions for generation, this work establishes a generalizable paradigm for aligning assessment with improvement, with direct implications for controllable text generation requiring verifiable accuracy and transparent quality control. All code and datasets will be released in GitHub for reproducibility. 

---
# Diagnosis, Bad Planning & Reasoning. Treatment, SCOPE -- Planning for Hybrid Querying over Clinical Trial Data 

**Authors**: Suparno Roy Chowdhury, Manan Roy Choudhury, Tejas Anvekar, Muhammad Ali Khan, Kaneez Zahra Rubab Khakwani, Mohamad Bassam Sonbol, Irbaz Bin Riaz, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.25120)  

**Abstract**: We study clinical trial table reasoning, where answers are not directly stored in visible cells but must be reasoned from semantic understanding through normalization, classification, extraction, or lightweight domain reasoning. Motivated by the observation that current LLM approaches often suffer from "bad reasoning" under implicit planning assumptions, we focus on settings in which the model must recover implicit attributes such as therapy type, added agents, endpoint roles, or follow-up status from partially observed clinical-trial tables. We propose SCOPE (Structured Clinical hybrid Planning for Evidence retrieval in clinical trials), a multi-LLM planner-based framework that decomposes the task into row selection, structured planning, and execution. The planner makes the source field, reasoning rules, and output constraints explicit before answer generation, reducing ambiguity relative to direct prompting. We evaluate SCOPE on 1,500 hybrid reasoning questions over oncology clinical-trial tables against zero-shot, few-shot, chain-of-thought, TableGPT2, Blend-SQL, and EHRAgent. Results show that explicit multi-LLM planning improves accuracy for reasoning-based questions while offering a stronger accuracy-efficiency tradeoff than heavier agentic baselines. Our findings position clinical trial reasoning as a distinct table understanding problem and highlight hybrid planner-based decomposition as an effective solution 

---
# The Dynamics of Delusion: Modeling Bidirectional False Belief Amplification in Human-Chatbot Dialogue 

**Authors**: Ashish Mehta, Jared Moore, Jacy Reese Anthis, William Agnew, Eric Lin, Peggy Yin, Desmond C. Ong, Nick Haber, Carol Dweck  

**Link**: [PDF](https://arxiv.org/pdf/2604.25096)  

**Abstract**: There is growing concern that AI chatbots might fuel delusional beliefs in users. Some have suggested that humans and chatbots mutually reinforce false beliefs over time, but quantitative evidence is lacking. Using a unique dataset of chat logs from individuals who exhibited delusional thinking, we developed a latent state model that captures accumulating and decaying influences between humans and chatbots. We find that a bidirectional influence model substantially outperforms a unidirectional alternative where humans are the primary driver of delusion. We find that humans exert strong but short-lived influence on chatbots, whereas chatbots exert longer-lasting influence on humans. Moreover, chatbots exert strong, stable self-influence over their own future outputs that tends to perpetuate delusions over long stretches of conversation. In fact, this chatbot self-influence constituted the dominant pathway when considering accumulated influence over time. Overall, these results indicate that humans tend to drive sharp, immediate increases in delusion, whereas chatbots sustain and propagate these effects over longer timescales. Together, these findings provide the first quantitative evidence that human-chatbot interactions can form feedback loops of delusion, decomposable into distinct pathways with dissociable temporal dynamics. By doing so, they can inform the development of safer AI systems. 

---
# Analyzing LLM Reasoning to Uncover Mental Health Stigma 

**Authors**: Sreehari Sankar, Aliakbar Nafar, Mona Barman, Hannah K. Heitz, Ashwin Kumar, Pouria Tohidi, Dailun Li, Danish Hussain, Russell DuBois, Hamed Hasheminia, Farshad Majzoubi  

**Link**: [PDF](https://arxiv.org/pdf/2604.25053)  

**Abstract**: While large language models (LLMs) are increasingly being explored for mental health applications, recent studies reveal that they can exhibit stigma toward individuals with psychological conditions. Existing evaluations of this stigma primarily rely on multiple-choice questions (MCQs), which fail to capture the biases embedded within the models' underlying logic. In this paper, we analyze the intermediate reasoning steps of LLMs to uncover hidden stigmatizing language and the internal rationales driving it. We leverage clinical expertise to categorize common patterns of stigmatizing language directed at individuals with psychological conditions and use this framework to identify and tag problematic statements in LLM reasoning. Furthermore, we rate the severity of these statements, distinguishing between overt prejudice and more subtle, less immediately harmful biases. To broaden the reasoning domain and capture a wider array of patterns, we also extend an existing mental health stigma benchmark by incorporating additional psychological conditions. Our findings demonstrate that evaluating model reasoning not only exposes substantially more stigma than traditional MCQ-based methods but it helps to identify the flaws in the LLMs' logic and their understanding of mental health conditions. 

---
# Dual-Track CoT: Budget-Aware Stepwise Guidance for Small LMs 

**Authors**: Sagnik Chatterjee, Atharva Patil, Sricharan Ramesh  

**Link**: [PDF](https://arxiv.org/pdf/2604.25039)  

**Abstract**: Large Language Models (LLMs) solve many reasoning tasks via chain-of-thought (CoT) prompting, but smaller models (about 7 to 8B parameters) still struggle with multi-step reasoning under tight compute and token budgets. Existing test time reasoning methods such as self consistency (sampling multiple rationales and voting), Tree-of-Thoughts (search over intermediate thoughts), and critique revise loops improve performance, but often at high token cost and without fine-grained step-level control. This project1 aims to address that gap: can Small Language Models (SLMs) reason reliably using the same or fewer tokens? This question is both scientific and practical. Scientifically, it probes whether process supervision and simple test-time controls (such as token budgets and rejection of redundant steps) can substitute for model scale or large sampling counts. Practically, many deployments (on-device, low-latency, or cost-constrained settings) cannot afford huge models or dozens of sampled rationales per query. A method that improves SLM reasoning at fixed cost would therefore be directly useful. 

---
# Faithful Autoformalization via Roundtrip Verification and Repair 

**Authors**: Daneshvar Amrollahi, Jerry Lopez, Clark Barrett  

**Link**: [PDF](https://arxiv.org/pdf/2604.25031)  

**Abstract**: When an LLM formalizes natural language, how do we know the output is faithful? We propose a roundtrip verification approach which does not require ground-truth annotations: formalize a statement, translate the result back to natural language, re-formalize, and use a formal tool to check logical equivalence. When the two formalizations agree, this provides evidence of a faithful formalization. When they disagree, a diagnosis step identifies which translation stage failed, and a targeted repair operator attempts to correct that stage. We evaluate our approach on 150 traffic rules using Claude Opus 4.6 and GPT-5.2. Diagnosis-guided repair raises formal equivalence from 45--61% to 83--85% for both models, outperforming a random-repair baseline. An independent NLI analysis confirms that formal equivalence is correlated with less semantic drift. 

---
# Why Does Reinforcement Learning Generalize? A Feature-Level Mechanistic Study of Post-Training in Large Language Models 

**Authors**: Dan Shi, Zhuowen Han, Simon Ostermann, Renren Jin, Josef van Genabith, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2604.25011)  

**Abstract**: Reinforcement learning (RL)-based post-training often improves the reasoning performance of large language models (LLMs) beyond the training domain, while supervised fine-tuning (SFT) frequently leads to general capabilities forgetting. However, the mechanisms underlying this contrast remain unclear. To bridge this gap, we present a feature-level mechanistic analysis methodology to probe RL generalization using a controlled experimental setup, where RL- and SFT-tuned models are trained from the same base model on identical data. Leveraging our interpretability framework, we align internal activations across models within a shared feature space and analyze how features evolve during post-training. We find that SFT rapidly introduces many highly specialized features that stabilize early in training, whereas RL induces more restrained and continually evolving feature changes that largely preserve base models' representations. Focusing on samples where RL succeeds but the base model fails, we identify a compact, task-agnostic set of features that directly mediate generalization across diverse tasks. Feature-level interventions confirm their causal role: disabling these features significantly degrades RL models' generalization performance, while amplifying them improves base models' performance. The code is available at this https URL. 

---
# Dont Stop Early: Scalable Enterprise Deep Research with Controlled Information Flow and Evidence-Aware Termination 

**Authors**: Prafulla Kumar Choubey, Kung-Hsiang Huang, Pranav Narayanan Venkit, Jiaxin Zhang, Vaibhav Vats, Yu Li, Xiangyu Peng, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2604.24978)  

**Abstract**: Enterprise deep research often fails to produce decision-ready reports due to uneven information coverage, context explosion, and premature stopping. We propose a scalable Enterprise Deep Research (EDR) architecture to address these failures. Our system (i) decomposes requests into coverage-driven objectives via outline generation with reflection, (ii) localizes context with dependency-guided execution and explicit information sharing, and (iii) enforces evidence-based completion criteria so agents iteratively collect information until sufficiency conditions are met. We evaluate on an internal sales enablement task and the public DeepResearch Bench benchmark, where our proposed system design achieves the strongest overall performance compared with competitive deep-research baselines. The results show that dependency-controlled context and explicit evidence sufficiency criteria reduce premature stopping and improve the consistency and depth of enterprise research outputs. 

---
# A Survey on LLM-based Conversational User Simulation 

**Authors**: Bo Ni, Leyao Wang, Yu Wang, Branislav Kveton, Franck Dernoncourt, Yu Xia, Hongjie Chen, Reuben Leura, Samyadeep Basu, Subhojyoti Mukherjee, Puneet Mathur, Nesreen Ahmed, Junda Wu, Li Li, Huixin Zhang, Ruiyi Zhang, Tong Yu, Sungchul Kim, Jiuxiang Gu, Zhengzhong Tu, Alexa Siu, Zichao Wang, David Seunghyun Yoon, Nedim Lipka, Namyong Park, Zihao Lin, Trung Bui, Yue Zhao, Tyler Derr, Ryan A. Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24977)  

**Abstract**: User simulation has long played a vital role in computer science due to its potential to support a wide range of applications. Language, as the primary medium of human communication, forms the foundation of social interaction and behavior. Consequently, simulating conversational behavior has become a key area of study. Recent advancements in large language models (LLMs) have significantly catalyzed progress in this domain by enabling high-fidelity generation of synthetic user conversation. In this paper, we survey recent advancements in LLM-based conversational user simulation. We introduce a novel taxonomy covering user granularity and simulation objectives. Additionally, we systematically analyze core techniques and evaluation methodologies. We aim to keep the research community informed of the latest advancements in conversational user simulation and to further facilitate future research by identifying open challenges and organizing existing work under a unified framework. 

---
# Dynamic Decision Learning: Test-Time Evolution for Abnormality Grounding in Rare Diseases 

**Authors**: Jun Li, Mingxuan Liu, Jiazhen Pan, Che Liu, Wenjia Bai, Cosmin I. Bercea, Julia A. Schnabel  

**Link**: [PDF](https://arxiv.org/pdf/2604.24972)  

**Abstract**: Clinical abnormality grounding for rare diseases is often hindered by data scarcity, making supervised fine-tuning impractical and single-pass inference highly unstable. We propose Dynamic Decision Learning (DDL), a framework that enables frozen large vision-language models (LVLMs) to refine their decisions across both language and visual spaces by optimizing instructions and consolidating predictions under visual perturbations. This process improves localization quality and produces a consensus-based reliability score that quantifies model confidence. Results on brain imaging benchmarks, including a rare-disease dataset with 281 pathology types across models ranging from 3B to 72B parameters, show that DDL improves mAP@75 by up to 105% on rare-disease cases and outperforms adaptation baselines and supervised fine-tuning. Furthermore, DDL demonstrates stronger calibration between reliability scores and localization accuracy under severe distribution shifts and increasing task difficulty. Code is available at: this https URL 

---
# BenchGuard: Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks 

**Authors**: Xinming Tu, Tianze Wang, Yingzhou, Kexin Huang, Yuanhao Qu, Sara Mostafavi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24955)  

**Abstract**: As benchmarks grow in complexity, many apparent agent failures are not failures of the agent at all - they are failures of the benchmark itself: broken specifications, implicit assumptions, and rigid evaluation scripts that penalize valid alternative approaches. We propose employing frontier LLMs as systematic auditors of evaluation infrastructure, and realize this vision through BenchGuard, the first automated auditing framework for task-oriented, execution-based agent benchmarks. BenchGuard cross-verifies all benchmark artifacts via structured LLM protocols, optionally incorporating agent solutions or execution traces as additional diagnostic evidence. Deployed on two prominent scientific benchmarks, BenchGuard identified 12 author-confirmed issues in ScienceAgentBench - including fatal errors rendering tasks unsolvable - and exactly matched 83.3% of expert-identified issues on the BIXBench Verified-50 subset, catching defects that prior human review missed entirely. A full audit of 50 complex bioinformatics tasks costs under USD 15, making automated benchmark auditing a practical and valuable complement to human review. These findings point toward AI-assisted benchmark development, where frontier models serve not only as subjects of evaluation but as active participants in validating the evaluation infrastructure itself. 

---
# Independent-Component-Based Encoding Models of Brain Activity During Story Comprehension 

**Authors**: Kamya Hari, Taha Binhuraib, Jin Li, Cory Shain, Anna A. Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2604.24942)  

**Abstract**: Encoding models provide a powerful framework for linking continuous stimulus features to neural activity; however, traditional voxelwise approaches are limited by measurement noise, inter-subject variability, and redundancy arising from spatially correlated voxels encoding overlapping neural signals. Here, we propose an independent component (IC)-based encoding framework that dissociates stimulus-driven and noise-driven signals in fMRI data. We decompose continuous fMRI data from naturalistic story listening into ICs using one subset of the data, and train encoding models on independent data to predict IC time series from large language model representations of linguistic input. Across subjects, a subset of ICs exhibited consistently high predictivity. These ICs were spatially and temporally consistent across subjects and included cognitive networks known to respond during story listening (auditory and language). Auditory component time series were strongly correlated with acoustic stimulus features, highlighting the interpretability of identified component time series. Components identified as noise or motion-related artifacts by ICA-AROMA showed uniformly poor predictive performance, confirming that highly predicted components reflect genuine stimulus-related neural signals rather than confounds. Overall, IC-based encoding models enable analyses at the level of functional networks, accommodating the variability in network locations across individuals and providing interpretable results that are easy to compare across subjects. 

---
# ADE: Adaptive Dictionary Embeddings -- Scaling Multi-Anchor Representations to Large Language Models 

**Authors**: Orhan Demirci, Sezer Aptourachman  

**Link**: [PDF](https://arxiv.org/pdf/2604.24940)  

**Abstract**: Word embeddings are fundamental to natural language processing, yet traditional approaches represent each word with a single vector, creating representational bottlenecks for polysemous words and limiting semantic expressiveness. While multi-anchor representations have shown promise by representing words as combinations of multiple vectors, they have been limited to small-scale models due to computational inefficiency and lack of integration with modern transformer architectures. We introduce Adaptive Dictionary Embeddings (ADE), a framework that successfully scales multi-anchor word representations to large language models. ADE makes three key contributions: (1) Vocabulary Projection (VP), which transforms the costly two-stage anchor lookup into a single efficient matrix operation; (2) Grouped Positional Encoding (GPE), a novel positional encoding scheme where anchors of the same word share positional information, preserving semantic coherence while enabling anchor-level variation; and (3) context-aware anchor reweighting, which leverages self-attention to dynamically compose anchor contributions based on sequence context. We integrate these components into the Segment-Aware Transformer (SAT), which provides context-aware reweighting of anchor contributions at inference time. We evaluate ADE on AG News and DBpedia-14 text classification benchmarks. With 98.7% fewer trainable parameters than DeBERTa-v3-base, ADE surpasses DeBERTa on DBpedia-14 (98.06% vs. 97.80%) and approaches it on AG News (90.64% vs. 94.50%), while compressing the embedding layer over 40x -- demonstrating that multi-anchor representations are a practical and parameter-efficient alternative to single-vector embeddings in modern transformer architectures. 

---
# GAIA-v2-LILT: Multilingual Adaptation of Agent Benchmark beyond Translation 

**Authors**: Yunsu Kim, Kaden Uhlig, Joern Wuebker  

**Link**: [PDF](https://arxiv.org/pdf/2604.24929)  

**Abstract**: Agent benchmarks remain largely English-centric, while their multilingual versions are often built with machine translation (MT) and limited post-editing. We argue that, for agentic tasks, this minimal workflow can easily break benchmark validity through query-answer misalignment or culturally off-target context. We propose a refined workflow for adapting English benchmarks into multiple languages with explicit functional alignment, cultural alignment, and difficulty calibration using both automated checks and human review. Using this workflow, we introduce GAIA-v2-LILT, a re-audited multilingual extension of GAIA covering five non-English languages. In experiments, our workflow improves agent success rates by up to 32.7% over minimally translated versions, bringing the closest audited setting to within 3.1% of English performance while substantial gaps remain in many other cases. This indicates that a substantial share of the multilingual performance gap is benchmark-induced measurement error, motivating task-level alignment when adapting English benchmarks across languages. The data is available as part of the MAPS package at this https URL. We also release the code used in our experiments at this https URL. 

---
# Large Language Models Explore by Latent Distilling 

**Authors**: Yuanhao Zeng, Ao Lu, Lufei Li, Zheng Zhang, Yexin Li, Kan Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.24927)  

**Abstract**: Generating diverse responses is crucial for test-time scaling of large language models (LLMs), yet standard stochastic sampling mostly yields surface-level lexical variation, limiting semantic exploration. In this paper, we propose Exploratory Sampling (ESamp), a decoding approach that explicitly encourages semantic diversity during generation. ESamp is motivated by the well-known observation that neural networks tend to make lower-error predictions on inputs similar to those encountered before, and incur higher prediction error on novel ones. Building on this property, we train a lightweight Distiller at test time to predict deep-layer hidden representations of the LLM from its shallow-layer representations to model the LLM's depth-wise representation transitions. During decoding, the Distiller continuously adapts to the mappings induced by the current generation context. ESamp uses the prediction error as a novelty signal to reweight candidate token extensions conditioned on the current prefix, thereby biasing decoding toward less-explored semantic patterns. ESamp is implemented with an asynchronous training--inference pipeline, with less than 5% worst case overhead (1.2% in the optimized release). Empirical results show that ESamp significantly boosts the Pass@k efficiency of reasoning models, showing superior or comparable performance to strong stochastic and heuristic baselines. Notably, ESamp achieves robust generalization across mathematics, science, and code generation benchmarks and breaks the trade-off between diversity and coherence in creative writing. Our code has released at: this https URL. 

---
# Elderly-Contextual Data Augmentation via Speech Synthesis for Elderly ASR 

**Authors**: Minsik Lee, Seoi Hong, Chongmin Lee, Sieun Choi, Jian Kim, Jua Han, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2604.24770)  

**Abstract**: Despite recent progress in automatic speech recognition (ASR), elderly ASR (EASR) remains challenging due to limited training data and the distinct acoustic and linguistic characteristics of elderly speech. In this work, we address data scarcity in EASR through a data augmentation pipeline that combines large language model (LLM)-based transcript paraphrasing with text-to-speech (TTS) synthesis. Given an elderly speech dataset, the LLM first generates elderly-contextual paraphrases of the original transcripts, and the TTS model then synthesizes corresponding speech using elderly reference speakers. The resulting synthetic audio-text pairs are merged with the original data to fine-tune Whisper without architectural modification. We further analyze the effects of augmentation ratio and reference-speaker composition in low-resource EASR. Experiments on English and Korean elderly speech datasets from speakers aged 70 and above show that the proposed method consistently improves performance over conventional augmentation baselines, achieving up to a 58.2% reduction in word error rate (WER) compared with the Whisper baseline. 

---
# Recursive Multi-Agent Systems 

**Authors**: Xiyuan Yang, Jiaru Zou, Rui Pan, Ruizhong Qiu, Pan Lu, Shizhe Diao, Jindong Jiang, Hanghang Tong, Tong Zhang, Markus J. Buehler, Jingrui He, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2604.25917)  

**Abstract**: Recursive or looped language models have recently emerged as a new scaling axis by iteratively refining the same model computation over latent states to deepen reasoning. We extend such scaling principle from a single model to multi-agent systems, and ask: Can agent collaboration itself be scaled through recursion? To this end, we introduce RecursiveMAS, a recursive multi-agent framework that casts the entire system as a unified latent-space recursive computation. RecursiveMAS connects heterogeneous agents as a collaboration loop through the lightweight RecursiveLink module, enabling in-distribution latent thoughts generation and cross-agent latent state transfer. To optimize our framework, we develop an inner-outer loop learning algorithm for iterative whole-system co-optimization through shared gradient-based credit assignment across recursion rounds. Theoretical analyses of runtime complexity and learning dynamics establish that RecursiveMAS is more efficient than standard text-based MAS and maintains stable gradients during recursive training. Empirically, we instantiate RecursiveMAS under 4 representative agent collaboration patterns and evaluate across 9 benchmarks spanning mathematics, science, medicine, search, and code generation. In comparison with advanced single/multi-agent and recursive computation baselines, RecursiveMAS consistently delivers an average accuracy improvement of 8.3%, together with 1.2$\times$-2.4$\times$ end-to-end inference speedup, and 34.6%-75.6% token usage reduction. Code and Data are provided in this https URL. 

---
# Three Models of RLHF Annotation: Extension, Evidence, and Authority 

**Authors**: Steve Coyne  

**Link**: [PDF](https://arxiv.org/pdf/2604.25895)  

**Abstract**: Preference-based alignment methods, most prominently Reinforcement Learning with Human Feedback (RLHF), use the judgments of human annotators to shape large language model behaviour. However, the normative role of these judgments is rarely made explicit. I distinguish three conceptual models of that role. The first is extension: annotators extend the system designers' own judgments about what outputs should be. The second is evidence: annotators provide independent evidence about some facts, whether moral, social or otherwise. The third is authority: annotators have some independent authority (as representatives of the broader population) to determine system outputs. I argue that these models have implications for how RLHF pipelines should solicit, validate and aggregate annotations. I survey landmark papers in the literature on RLHF and related methods to illustrate how they implicitly draw on these models, describe failure modes that come from unintentionally or intentionally conflating them, and offer normative criteria for choosing among them. My central recommendation is that RLHF pipeline designers should decompose annotation into separable dimensions and tailor each pipeline to the model most appropriate for that dimension, rather than seeking a single unified pipeline. 

---
# Barriers to Universal Reasoning With Transformers (And How to Overcome Them) 

**Authors**: Oliver Kraus, Yash Sarrof, Yuekun Yao, Alexander Koller, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2604.25800)  

**Abstract**: Chain-of-Thought (CoT) has been shown to empirically improve Transformers' performance, and theoretically increase their expressivity to Turing completeness. However, whether Transformers can learn to generalize to CoT traces longer than those seen during training is understudied. We use recent theoretical frameworks for Transformer length generalization and find that -- under standard positional encodings and a finite alphabet -- Transformers with CoT cannot solve problems beyond $TC^0$, i.e. the expressivity benefits do not hold under the stricter requirement of length-generalizable learnability. However, if we allow the vocabulary to grow with problem size, we attain a length-generalizable simulation of Turing machines where the CoT trace length is linear in the simulated runtime up to a constant. Our construction overcomes two core obstacles to reliable length generalization: repeated copying and last-occurrence retrieval. We assign each tape position a unique signpost token, and log only value changes to enable recovery of the current tape symbol through counts circumventing both barriers. Further, we empirically show that the use of such signpost tokens and value change encodings provide actionable guidance to improve length generalization on hard problems. 

---
# Toward Multimodal Conversational AI for Age-Related Macular Degeneration 

**Authors**: Ran Gu, Benjamin Hou, Mélanie Hébert, Asmita Indurkar, Yifan Yang, Emily Y. Chew, Tiarnán D. L. Keenan, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25720)  

**Abstract**: Despite strong performance of deep learning models in retinal disease detection, most systems produce static predictions without clinical reasoning or interactive explanation. Recent advances in multimodal large language models (MLLMs) integrate diagnostic predictions with clinically meaningful dialogue to support clinical decision-making and patient counseling. In this study, OcularChat, an MLLM, was fine-tuned from Qwen2.5-VL using simulated patient-physician dialogues to diagnose age-related macular degeneration (AMD) through visual question answering on color fundus photographs (CFPs). A total of 705,850 simulated dialogues paired with 46,167 CFPs were generated to train OcularChat to identify key AMD features and produce reasoned predictions. OcularChat demonstrated strong classification performance in AREDS, achieving accuracies of 0.954, 0.849, and 0.678 for the three diagnostic tasks: advanced AMD, pigmentary abnormalities, and drusen size, significantly outperforming existing MLLMs. On AREDS2, OcularChat remained the top-performing method on all tasks. Across three independent ophthalmologist graders, OcularChat achieved higher mean scores than a strong baseline model for advanced AMD (3.503 vs. 2.833), pigmentary abnormalities (3.272 vs. 2.828), drusen size (3.064 vs. 2.433), and overall impression (2.978 vs. 2.464) on a 5-point clinical grading rubric. Beyond strong objective performance in AMD severity classification, OcularChat demonstrated the ability to provide diagnostic reasoning, clinically relevant explanations, and interactive dialogue, with high performance in subjective ophthalmologist evaluation. These findings suggest that MLLMs may enable accurate, interpretable, and clinically useful image-based diagnosis and classification of AMD. 

---
# The Surprising Universality of LLM Outputs: A Real-Time Verification Primitive 

**Authors**: Alex Bogdan, Adrian de Valois-Franklin  

**Link**: [PDF](https://arxiv.org/pdf/2604.25634)  

**Abstract**: We report a striking statistical regularity in frontier LLM outputs that enables a CPU-only scoring primitive running
at 2.6 microseconds per token, with estimated latency up to 100,000$\times$ (five orders of magnitude) below existing
sampling-based detectors. Across six contemporary models from five independent vendors, two generation sizes, and five
held-out domains, token rank-frequency distributions converge to the same two-parameter Mandelbrot ranking
distribution, with 34 of 36 model-by-domain fits exceeding $R^{2} = 0.94$ and 35 of 36 favoring Mandelbrot over Zipf
by AIC. The shared family does not collapse the models into statistical duplicates. Fitted Mandelbrot parameters
remain cleanly separable between models: the cross-model spread in $q$ (1.63 to 3.69) exceeds its per-model bootstrap
standard deviation (0.03 to 0.10) by more than an order of magnitude, yielding tens of standard deviations of
separation per few thousand output tokens. Two capabilities follow. First, statistical model fingerprinting: text from
a vendor-delivered LLM can be tested against its claimed model family without cryptographic watermarks or access to
model internals, supporting provenance verification and silent-substitution audits. Second, a model-agnostic reference
distribution for black-box output assessment, from which we derive a single-pass scoring primitive that composes with
model log probabilities when available and degrades to a rank-only mode usable on closed APIs. Pilot results on
FRANK, TruthfulQA, and HaluEval map where the primitive helps (lexical anomalies, unsupported entities) and where it
structurally cannot (reasoning errors in domain-appropriate vocabulary). We position the primitive as a first-pass
triage layer in compound evaluation stacks, not as a replacement for sampling-based or source-conditioned verifiers. 

---
# Walking Through Uncertainty: An Empirical Study of Uncertainty Estimation for Audio-Aware Large Language Models 

**Authors**: Chun-Yi Kuan, Wei-Ping Huang, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2604.25591)  

**Abstract**: Recent audio-aware large language models (ALLMs) have demonstrated strong capabilities across diverse audio understanding and reasoning tasks, but they still frequently produce hallucinated or overly confident outputs. While uncertainty estimation has been extensively studied in text-only LLMs, it remains largely unexplored for ALLMs, where audio-conditioned generation introduces additional challenges such as perceptual ambiguity and cross-modal grounding. In this work, we present the first systematic empirical study of uncertainty estimation in ALLMs. We benchmark five representative methods, including predictive entropy, length-normalized entropy, semantic entropy, discrete semantic entropy, and P(True), across multiple models and diverse evaluation settings spanning general audio understanding, reasoning, hallucination detection, and unanswerable question answering. Our results reveal two key findings. First, semantic-level and verification-based methods consistently outperform token-level baselines on general audio reasoning benchmarks. Second, on trustworthiness-oriented benchmarks, the relative effectiveness of uncertainty methods becomes notably more model- and benchmark-dependent, indicating that conclusions drawn from general reasoning settings do not straightforwardly transfer to hallucination and unanswerable-question scenarios. We further explore uncertainty-based adaptive inference as a potential downstream application. We hope this study provides a foundation for future research on reliable, uncertainty-aware audio-language systems. 

---
# PSP: An Interpretable Per-Dimension Accent Benchmark for Indic Text-to-Speech 

**Authors**: Venkata Pushpak Teja Menta  

**Link**: [PDF](https://arxiv.org/pdf/2604.25476)  

**Abstract**: Standard text-to-speech (TTS) evaluation measures intelligibility (WER, CER) and overall naturalness (MOS, UTMOS) but does not quantify accent. A synthesiser may score well on all four yet sound non-native on features that are phonemic in the target language. For Indic languages, these features include retroflex articulation, aspiration, vowel length, and the Tamil retroflex approximant (letter zha). We present PSP, the Phoneme Substitution Profile, an interpretable, per-phonological-dimension accent benchmark for Indic TTS. PSP decomposes accent into six complementary dimensions: retroflex collapse rate (RR), aspiration fidelity (AF), vowel-length fidelity (LF), Tamil-zha fidelity (ZF), Frechet Audio Distance (FAD), and prosodic signature divergence (PSD). The first four are measured via forced alignment plus native-speaker-centroid acoustic probes over Wav2Vec2-XLS-R layer-9 embeddings; the latter two are corpus-level distributional distances. In this v1 we benchmark four commercial and open-source systems (ElevenLabs v3, Cartesia Sonic-3, Sarvam Bulbul, Indic Parler-TTS) on Hindi, Telugu, and Tamil pilot sets, with a fifth system (Praxy Voice) included on all three languages, plus an R5->R6 case study on Telugu. Three findings: (i) retroflex collapse grows monotonically with phonological difficulty Hindi < Telugu < Tamil (~1%, ~40%, ~68%); (ii) PSP ordering diverges from WER ordering -- commercial WER-leaders do not uniformly lead on retroflex or prosodic fidelity; (iii) no single system is Pareto-optimal across all six dimensions. We release native reference centroids (500 clips per language), 1000-clip embeddings for FAD, 500-clip prosodic feature matrices for PSD, 300-utterance golden sets per language, scoring code under MIT, and centroids under CC-BY. Formal MOS-correlation is deferred to v2; v1 reports five internal-consistency signals plus a native-audio sanity check. 

---
# Praxy Voice: Voice-Prompt Recovery + BUPS for Commercial-Class Indic TTS from a Frozen Non-Indic Base at Zero Commercial-Training-Data Cost 

**Authors**: Venkata Pushpak Teja Menta  

**Link**: [PDF](https://arxiv.org/pdf/2604.25441)  

**Abstract**: Commercial TTS systems produce near-native Indic audio, but the best open-source bases (Chatterbox, Indic Parler-TTS, IndicF5) trail them on measured phonological dimensions, and the most widely adopted multilingual base (Chatterbox, 23 languages) does not even tokenise Telugu or Tamil. We ask: what is the minimum intervention that brings such a non-Indic-native base to commercial-class output on Telugu, Tamil, and Hindi, without training a new acoustic decoder and without any commercial TTS training data? We combine three pieces: (1) BUPS, a Brahmic Unified Phoneme Space that deterministically romanises seven Indic scripts to ISO-15919 so Chatterbox's Latin tokeniser can process them; (2) a LoRA adapter on only the text-token predictor (Chatterbox's t3), trained on ~1,220h of licensed Indic audio with a Hindi-proxy language_id; (3) a voice-prompt recovery recipe -- an 8-11s same-language reference clip plus three sampling overrides (exaggeration 0.7, temperature 0.6, min_p 0.1; "Config B") -- that recovers commercial-class acoustic output with no acoustic-decoder training. On Hindi, the LoRA regresses accuracy and we instead use vanilla Chatterbox + Config B, giving a two-branch deployment. Evaluated on 10-utterance pilot sets with the companion PSP benchmark, Praxy Voice matches or slightly leads commercial baselines: 26.7% retroflex collapse on Telugu (vs Sarvam Bulbul 33.3%), 71% Tamil-zha collapse (vs commercial trio's 86%), 0.025 LLM-WER on Hindi (tied with Cartesia Sonic-3). For intra-sentential code-mix we add a third branch (IndicF5 + native-script transliteration) that drops code-mix LLM-WER from 0.80-0.85 to 0.14-0.27 across Hi/Te/Ta. We release R6 LoRA weights (Apache-2.0), inference code and router (MIT), and a Gradio demo. 

---
# R$^3$-SQL: Ranking Reward and Resampling for Text-to-SQL 

**Authors**: Hojae Han, Yeonseok Jeong, Seung-won Hwang, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2604.25325)  

**Abstract**: Modern Text-to-SQL systems generate multiple candidate SQL queries and rank them to judge a final prediction. However, existing methods face two limitations. First, they often score functionally equivalent SQL queries inconsistently despite identical execution results. Second, ranking cannot recover when the correct SQL is absent from the candidate pool. We propose R$^3$-SQL, a Text-to-SQL framework that addresses both issues through unified reward for ranking and resampling. R$^3$-SQL first groups candidates by execution result and ranks groups for consistency. To score each group, it combines a pairwise preference across groups with a pointwise utility from the best group rank and size, capturing relative preference, consistency, and candidate quality. To improve candidate recall, R$^3$-SQL introduces agentic resampling, which judges the generated candidate pool and selectively resamples when the correct SQL is likely absent. R$^3$-SQL achieves 75.03 execution accuracy on BIRD-dev, a new state of the art among methods using models with disclosed sizes, with consistent gains across five benchmarks. 

---
# Cutscene Agent: An LLM Agent Framework for Automated 3D Cutscene Generation 

**Authors**: Lanshan He, Haozhou Pang, Qi Gan, Xin Shen, Ziwei Zhang, Yibo Liu, Gang Fang, Bo Liu, Kai Sheng, Shengfeng Zeng, Chaofan Li, Zhen Hui, Keer Zhou, Lan Zhou, Shujun Dai  

**Link**: [PDF](https://arxiv.org/pdf/2604.25318)  

**Abstract**: Cutscenes are carefully choreographed cinematic sequences embedded in video games and interactive media, serving as the primary vehicle for narrative delivery, character development, and emotional engagement. Producing cutscenes is inherently complex: it demands seamless coordination across screenwriting, cinematography, character animation, voice acting, and technical direction, often requiring days to weeks of collaborative effort from multidisciplinary teams to produce minutes of polished content. In this work, we present Cutscene Agent, an LLM agent framework for automated end-to-end cutscene generation. The framework makes three contributions: (1)~a Cutscene Toolkit built on the Model Context Protocol (MCP) that establishes \emph{bidirectional} integration between LLM agents and the game engine -- agents not only invoke engine operations but continuously observe real-time scene state, enabling closed-loop generation of editable engine-native cinematic assets; (2)~a multi-agent system where a director agent orchestrates specialist subagents for animation, cinematography, and sound design, augmented by a visual reasoning feedback loop for perception-driven refinement; and (3)~CutsceneBench, a hierarchical evaluation benchmark for cutscene generation. Unlike typical tool-use benchmarks that evaluate short, isolated function calls, cutscene generation requires long-horizon, multi-step orchestration of dozens of interdependent tool invocations with strict ordering constraints -- a capability dimension that existing benchmarks do not cover. We evaluate a range of LLMs on CutsceneBench and analyze their performance across this challenging task. 

---
# VLM Judges Can Rank but Cannot Score: Task-Dependent Uncertainty in Multimodal Evaluation 

**Authors**: Divake Kumar, Sina Tayebati, Devashri Naik, Ranganath Krishnan, Amit Ranjan Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2604.25235)  

**Abstract**: Vision-language models (VLMs) are increasingly used as automated judges for multimodal systems, yet their scores provide no indication of reliability. We study this problem through conformal prediction, a distribution-free framework that converts a judge's point score into a calibrated prediction interval using only score-token log-probabilities, with no retraining. We present the first systematic analysis of conformal prediction for VLM-as-a-Judge across 3 judges and 14 visual task categories. Our results show that evaluation uncertainty is strongly task-dependent: intervals cover ~40% of the score range for aesthetics and natural images but expand to ~70% for chart and mathematical reasoning, yielding a quantitative reliability map for multimodal evaluation. We further identify a failure mode not captured by standard evaluation metrics, ranking-scoring decoupling, where judges achieve high ranking correlation while producing wide, uninformative intervals, correctly ordering responses but failing to assign reliable absolute scores. Finally, we show that interval width is driven primarily by task difficulty and annotation quality, i.e., the same judge and method yield 4.5x narrower intervals on a clean, multi-annotator captioning benchmark. Code: this https URL 

---
# DRAGON: A Benchmark for Evidence-Grounded Visual Reasoning over Diagrams 

**Authors**: Anirudh Iyengar Kaniyar Narayana Iyengar, Tampu Ravi Kumar, Gaurav Najpande, Manan Suri, Dinesh Manocha, Puneet Mathur, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2604.25231)  

**Abstract**: Diagram question answering (DQA) requires models to interpret structured visual representations such as charts, maps, infographics, circuit schematics, and scientific diagrams. Recent vision-language models (VLMs) often achieve high answer accuracy on these tasks, yet correct answers do not guarantee that models ground their reasoning in the diagram regions that support the prediction. Models may instead rely on textual correlations or dataset artifacts without identifying the visual evidence required to verify the answer. This limitation prevents reliable evaluation of diagram reasoning and reduces interpretability. We introduce DRAGON, a benchmark for evaluating evidence-grounded visual reasoning in diagrams. Given a diagram, a question, and the correct answer, a model must predict bounding boxes that correspond to the visual elements required to justify the answer. These evidence regions may include answer-bearing components, textual labels, legends, axes, connectors, and other supporting structures involved in the reasoning process. The DRAGON dataset contains 11,664 annotated question instances collected from six diagram QA datasets: ChartQA, Circuit-VQA, InfographicsVQA, MapIQ, MapWise, and AI2D. We release a 2,445-instance benchmark test set with human-verified reasoning evidence annotations and a standardized evaluation framework. We evaluate eight recent VLMs and analyze their ability to localize reasoning evidence across diverse diagram domains. DRAGON enables systematic evaluation of diagram reasoning and supports future research on models that ground their predictions in visual evidence. 

---
# MGTEVAL: An Interactive Platform for Systemtic Evaluation of Machine-Generated Text Detectors 

**Authors**: Yuanfan Li, Qi Zhou, Chengzhengxu Li, Zhaohan Zhang, Chenxu Zhao, Zepu Ruan, Chao Shen, Xiaoming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.25152)  

**Abstract**: We present MGTEVAL, an extensible platform for systematic evaluation of Machine-Generated Text (MGT) detectors. Despite rapid progress in MGT detection, existing evaluations are often fragmented across datasets, preprocessing, attacks, and metrics, making results hard to compare and reproduce. MGTEVAL organizes the workflow into four components: Dataset Building, Dataset Attack, Detector Training, and Performance Evaluation. It supports constructing custom benchmarks by generating MGT with configurable LLMs, applying 12 text attacks to test sets, training detectors via a unified interface, and reporting effectiveness, robustness, and efficiency. The platform provides both command-line and Web-based interfaces for user-friendly experimentation without code rewriting. 

---
# Doing More With Less: Revisiting the Effectiveness of LLM Pruning for Test-Time Scaling 

**Authors**: Ocean Monjur, Shahriar Kabir Nahin, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2604.25098)  

**Abstract**: While current Large Language Models (LLMs) exhibit remarkable reasoning capabilities through test-time compute scaling (TTS), their massive parameter counts and high inference costs have motivated the development of pruning methods that can reduce model size without sacrificing performance. However, specific to reasoning LLMs, prior work has shown that structured pruning (methods which removes entire set of layer blocks), significantly degrades TTS reasoning performance. In this work, we revisit this assumption and instead investigate whether unstructured pruning (methods that carefully remove only certain redundant/detrimental weights) exhibits similar limitations. Surprisingly, our extensive experiments across four reasoning benchmarks on two reasoning LLMs: s1.1-7B and Qwen3-8B, consistently show that unstructured pruning augments TTS performance compared to structured pruning, and at times can even outperform the unpruned full-weight LLMs. Furthermore, we also empirically study the impact of different layer-wise sparsity allocation strategies, which are an important parametric choice for instantiating unstructured pruning methods. These findings challenge the conventional notion that pruning always reduces TTS performance and in fact, suggest that carefully undertaken pruning can improve TTS effectiveness even further. 

---
# Cooperate to Compete: Strategic Coordination in Multi-Agent Conquest 

**Authors**: Abigail O'Neill, Alan Zhu, Mihran Miroyan, Narges Norouzi, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2604.25088)  

**Abstract**: Language Model (LM)-based agents remain largely untested in mixed-motive settings where agents must leverage short-term cooperation for long-term competitive goals (e.g., multi-party politics). We introduce Cooperate to Compete (C2C), a multi-agent environment where players can engage in private negotiations while competing to be the first to achieve their secret objective. Players have asymmetric objectives and negotiations are non-binding, allowing alliances to form and break as players' short-term interests align and diverge. We run AI only games and conduct a user study pitting human players against AI opponents. We identify significant differences between human and AI negotiation behaviors, finding that humans favor lower-complexity deals and are significantly less reliable partners compared to LM-based agents. We also find that humans are more aggressive negotiators, accepting deals without a counteroffer only 56.3% of the time compared to 67.6% for LM-based agents. Through targeted prompting inspired by these findings, we modify agents' negotiation behavior and improve win rates from 22.2% to 32.7%. We run over 1,100 games with over 16,000 private conversations totaling 15.2 million tokens and over 150,000 player actions. Our results establish C2C as a testbed for studying and building LM-based agents that can navigate the sophisticated coordination required for real-world deployments. The game, code, and dataset may be found at this https URL. 

---
# Leverage Laws: A Per-Task Framework for Human-Agent Collaboration 

**Authors**: Stan Loosmore  

**Link**: [PDF](https://arxiv.org/pdf/2604.25040)  

**Abstract**: We propose a per-task leverage ratio for human-agent collaboration: human work displaced by an agent, divided by the human time required to specify the task, resolve mid-run interrupts, and review the result. The denominator decomposes into three channels through which a conserved per-task information requirement must flow, each with its own time-cost scalar. We show that information density itself is directional and bounded by separate ceilings on human-to-agent and agent-to-human flow, and that the asymptotic behavior of leverage decomposes into two scaling axes (capability and memory) with a non-zero floor on the planning term set by irreducible task novelty bounded by human throughput. We extend this per-task analysis to a windowed leverage measure that accommodates recurring tasks, spawned subtasks, and amortized system-design investment. The per-task ceiling does not bind the windowed measure, though both remain bounded: $L_{\text{task}}$ by per-task novelty, $L_{\text{window}}$ by the stock of accumulated planning investment that pays out within the window. The framework operationalizes aspects of earlier qualitative work on supervisory control (Sheridan, 1992), common ground (Clark & Brennan, 1991), and mixed-initiative interaction (Horvitz, 1999) within a single normative ratio, and produces a list of testable empirical questions that we leave as open problems. 

---
# PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference 

**Authors**: Ishan Patel, Ishan Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2604.24971)  

**Abstract**: We present PolyKV, a system in which multiple concurrent inference agents share a single, asymmetrically compressed KV cache pool. Rather than allocating a separate KV cache per agent -- the standard paradigm -- PolyKV writes a compressed cache once and injects it into N independent agent contexts via HuggingFace DynamicCache objects. Compression is asymmetric: Keys are quantized at int8 (q8_0) to preserve softmax stability, while Values are compressed using TurboQuant MSE -- a Fast Walsh-Hadamard Transform (FWHT) rotation followed by 3-bit Lloyd-Max quantization with centroids tuned to N(0,1). We evaluate across two model scales (SmolLM2-1.7B-Instruct and Llama-3-8B-Instruct), three context lengths (600-7,194 tokens), and up to 15 concurrent agents. PolyKV achieves a stable 2.91x compression ratio across all configurations. On Llama-3-8B with 15 agents sharing a 4K-token context, PolyKV reduces KV cache memory from 19.8 GB to 0.45 GB -- a 97.7% reduction -- while maintaining only +0.57% perplexity degradation and a mean BERTScore F1 of 0.928. PPL delta does not grow with agent count and improves as context length increases, inverting to -0.26% at 1,851 coherent tokens. To our knowledge, no prior work combines a single shared, lossy-compressed KV pool with multi-reader concurrent agent access. 

---
# Odysseys: Benchmarking Web Agents on Realistic Long Horizon Tasks 

**Authors**: Lawrence Keunho Jang, Jing Yu Koh, Daniel Fried, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2604.24964)  

**Abstract**: Existing web agent benchmarks have largely converged on short, single-site tasks that frontier models are approaching saturation on. However, real world web use consists of long-horizon, multi-site workflows. Common web navigation tasks, such as comparing products across different domains, planning trips across multiple services, or summarizing information from multiple search queries, require sustained context and cross-site reasoning over potentially hours of browsing. To capture and evaluate such behaviors, we introduce Odysseys: a benchmark of 200 long-horizon web tasks derived from real world browsing sessions evaluated on the live Internet. We find that binary pass/fail evaluation is inadequate for long-horizon settings and introduce a rubric-based evaluation, annotating each Odysseys task with an average of 6.1 graded rubrics. We demonstrate that this yields higher agreement with humans and provides a more fine-grained signal than commonly used trajectory-level LLM-as-a-judge evaluation metrics. We tested several leading frontier models and find that the strongest models achieve a success rate of 44.5%, which leaves substantial room for future improvements. Beyond task success, we argue that efficiency is a first-class concern for long-horizon agents. We introduce a Trajectory Efficiency metric (rubric score per step) and find that even frontier agents achieve only 1.15%, marking an evident need for agents that can succeed efficiently and not simply eventually. Odysseys isolates the critical evaluation of long-horizon proficiency in open-web environments, providing a realistic benchmark to measure progress towards computer-use agents that can potentially productively operate for hours. We release our tasks, evaluation scripts, and other results at this https URL 

---
# Rethinking Layer Redundancy in Large Language Models: Calibration Objectives and Search for Depth Pruning 

**Authors**: Minkyu Kim, Vincent-Daniel Yun, Youngrae Kim, Youngjin Heo, Suin Cho, Seong-hun Kim, Woosang Lim, Gaeul Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2604.24938)  

**Abstract**: Depth pruning improves the inference efficiency of large language models by removing Transformer blocks. Prior work has focused on importance criteria and search algorithms, often treating layer redundancy as an inherent structural property of pretrained networks. In contrast, we adopt a \emph{functional perspective}, where redundancy is jointly influenced by the model and the evaluation objective, suggesting that a universal ranking may not be sufficient. Through an empirical study across three LLM families, two calibration objectives, and seven search algorithms, we observe that different objectives yield qualitatively different redundant layers, and that perplexity and downstream accuracy rankings do not consistently align. Under a fixed objective, however, search algorithms tend to produce similar solutions. Overall, our results suggest that the calibration objective may play a more influential role than the choice of search algorithm, indicating that further attention to objective design could be beneficial. 

---
# Libra-VLA: Achieving Learning Equilibrium via Asynchronous Coarse-to-Fine Dual-System 

**Authors**: Yifei Wei, Linqing Zhong, Yi Liu, Yuxiang Lu, Xindong He, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2604.24921)  

**Abstract**: Vision-Language-Action (VLA) models are a promising paradigm for generalist robotic manipulation by grounding high-level semantic instructions into executable physical actions. However, prevailing approaches typically adopt a monolithic generation paradigm, directly mapping visual-linguistic features to high-frequency motor commands in a flat, non-hierarchical fashion. This strategy overlooks the inherent hierarchy of robotic manipulation, where complex actions can be naturally modeled in a Hybrid Action Space, decomposing into discrete macro-directional reaching and continuous micro-pose alignment, severely widening the semantic-actuation gap and imposing a heavy representational burden on grounding high-level semantics to continuous actions. To address this, we introduce Libra-VLA, a novel Coarse-to-Fine Dual-System VLA architecture. We explicitly decouple the learning complexity into a coarse-to-fine hierarchy to strike a training equilibrium, while simultaneously leveraging this structural modularity to implement an asynchronous execution strategy. The Semantic Planner predicts discrete action tokens capturing macro-directional intent, while the Action Refiner conditions on coarse intent to generate high-frequency continuous actions for precise alignment. Crucially, our empirical analysis reveals that performance follows an inverted-U curve relative to action decomposition granularity, peaking exactly when the learning difficulty is balanced between the two sub-systems. With the asynchronous design, our approach offers a scalable, robust, and responsive solution for open-world manipulation. 

---
# Intrinsic Mutual Information as a Modulator for Preference Optimization 

**Authors**: Peng Liao, Peijia Zheng, Lingbo Li, Shangsong Liang, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2604.24804)  

**Abstract**: Offline preference optimization methods, such as Direct Preference Optimization (DPO), offer significant advantages in aligning Large Language Models (LLMs) with human values. However, achieving optimal performance with these methods typically involves additional hyperparameter tuning, resulting in substantial time overhead. Although prior work has proposed a range of improvements, these methods remain limited in effectiveness and have not fully eliminated reliance on hyperparameter tuning. In this work, we propose RMiPO, a lightweight and efficient framework for offline preference optimization. RMiPO leverages intrinsic Response-level Mutual information for Preference Optimization with hyperparameter modulation, dynamically decoupling preference contributions at negligible additional computational cost. Extensive experimental results demonstrate that RMiPO achieves consistently superior performance over existing methods while reducing training overhead by more than 15\%. Our code is available at this https URL. 

---
