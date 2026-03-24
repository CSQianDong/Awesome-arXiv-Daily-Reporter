# TiCo: Time-Controllable Training for Spoken Dialogue Models 

**Authors**: Kai-Wei Chang, Wei-Chih Chen, En-Pei Hu, Hung-yi Lee, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2603.22267)  

**Abstract**: We propose TiCo, a simple post-training method for enabling spoken dialogue models (SDMs) to follow time-constrained instructions and generate responses with controllable duration. This capability is valuable for real-world spoken language systems such as voice assistants and interactive agents, where controlling response duration can improve interaction quality. However, despite their strong ability to generate natural spoken responses, existing models lack time awareness and struggle to follow duration-related instructions (e.g., "Please generate a response lasting about 15 seconds"). Through an empirical evaluation of both open-source and commercial SDMs, we show that they frequently fail to satisfy such time-control requirements. TiCo addresses this limitation by enabling models to estimate elapsed speaking time during generation through Spoken Time Markers (STM) (e.g., <10.6 seconds>). These markers help the model maintain awareness of time and adjust the remaining content to meet the target duration. TiCo is simple and efficient: it requires only a small amount of data and no additional question-answer pairs, relying instead on self-generation and reinforcement learning. Experimental results show that TiCo significantly improves adherence to duration constraints while preserving response quality. 

---
# Greater accessibility can amplify discrimination in generative AI 

**Authors**: Carolin Holtermann, Minh Duc Bui, Kaitlyn Zhou, Valentin Hofmann, Katharina von der Wense, Anne Lauscher  

**Link**: [PDF](https://arxiv.org/pdf/2603.22260)  

**Abstract**: Hundreds of millions of people rely on large language models (LLMs) for education, work, and even healthcare. Yet these models are known to reproduce and amplify social biases present in their training data. Moreover, text-based interfaces remain a barrier for many, for example, users with limited literacy, motor impairments, or mobile-only devices. Voice interaction promises to expand accessibility, but unlike text, speech carries identity cues that users cannot easily mask, raising concerns about whether accessibility gains may come at the cost of equitable treatment. Here we show that audio-enabled LLMs exhibit systematic gender discrimination, shifting responses toward gender-stereotyped adjectives and occupations solely on the basis of speaker voice, and amplifying bias beyond that observed in text-based interaction. Thus, voice interfaces do not merely extend text models to a new modality but introduce distinct bias mechanisms tied to paralinguistic cues. Complementary survey evidence ($n=1,000$) shows that infrequent chatbot users are most hesitant to undisclosed attribute inference and most likely to disengage when such practices are revealed. To demonstrate a potential mitigation strategy, we show that pitch manipulation can systematically regulate gender-discriminatory outputs. Overall, our findings reveal a critical tension in AI development: efforts to expand accessibility through voice interfaces simultaneously create new pathways for discrimination, demanding that fairness and accessibility be addressed in tandem. 

---
# MemDLM: Memory-Enhanced DLM Training 

**Authors**: Zehua Pei, Hui-Ling Zhen, Weizhe Lin, Sinno Jialin Pan, Yunhe Wang, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22241)  

**Abstract**: Diffusion Language Models (DLMs) offer attractive advantages over Auto-Regressive (AR) models, such as full-attention parallel decoding and flexible generation. However, they suffer from a notable train-inference mismatch: DLMs are trained with a static, single-step masked prediction objective, but deployed through a multi-step progressive denoising trajectory. We propose MemDLM (Memory-Enhanced DLM), which narrows this gap by embedding a simulated denoising process into training via Bi-level Optimization. An inner loop updates a set of fast weights, forming a Parametric Memory that captures the local trajectory experience of each sample, while an outer loop updates the base model conditioned on this memory. By offloading memorization pressure from token representations to parameters, MemDLM yields faster convergence and lower training loss. Moreover, the inner loop can be re-enabled at inference time as an adaptation step, yielding additional gains on long-context understanding. We find that, when activated at inference time, this Parametric Memory acts as an emergent in-weight retrieval mechanism, helping MemDLM further reduce token-level attention bottlenecks on challenging Needle-in-a-Haystack retrieval tasks. Code: this https URL. 

---
# Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease 

**Authors**: Abner Hernandez, Eunjung Yeo, Kwanghee Choi, Chin-Jou Li, Zhengjun Yue, Rohan Kumar Das, Jan Rusz, Mathew Magimai Doss, Juan Rafael Orozco-Arroyave, Tomás Arias-Vergara, Andreas Maier, Elmar Nöth, David R. Mortensen, David Harwath, Paula Andrea Perez-Toro  

**Link**: [PDF](https://arxiv.org/pdf/2603.22225)  

**Abstract**: The limited availability of dysarthric speech data makes cross-lingual detection an important but challenging problem. A key difficulty is that speech representations often encode language-dependent structure that can confound dysarthria detection. We propose a representation-level language shift (LS) that aligns source-language self-supervised speech representations with the target-language distribution using centroid-based vector adaptation estimated from healthy-control speech. We evaluate the approach on oral DDK recordings from Parkinson's disease speech datasets in Czech, German, and Spanish under both cross-lingual and multilingual settings. LS substantially improves sensitivity and F1 in cross-lingual settings, while yielding smaller but consistent gains in multilingual settings. Representation analysis further shows that LS reduces language identity in the embedding space, supporting the interpretation that LS removes language-dependent structure. 

---
# Gumbel Distillation for Parallel Text Generation 

**Authors**: Chi Zhang, Xixi Hu, Bo Liu, Qiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22216)  

**Abstract**: The slow, sequential nature of autoregressive (AR) language models has driven the adoption of parallel decoding methods. However, these non-AR models often sacrifice generation quality as they struggle to model the complex joint distribution of token sequences. To narrow this performance gap, we introduce Gumbel Distillation, a novel distillation technique that enables parallel decoders to learn this distribution effectively. Our method leverages the Gumbel-Max trick to create a deterministic mapping from a latent Gumbel noise space to the output tokens of a high-performing AR teacher. As a model-agnostic technique, Gumbel Distillation seamlessly integrates with diverse parallel decoding architectures, including MDLM and BD3-LM. Experiments on LM1B and OpenWebText show that Gumbel Distillation substantially improves the generation quality of parallel language models, achieving a 30.0% improvement in MAUVE score and 10.5% in generative perplexity over MDLM trained on OpenWebText dataset. Code available at this https URL. 

---
# Enhancing Document-Level Machine Translation via Filtered Synthetic Corpora and Two-Stage LLM Adaptation 

**Authors**: Ireh Kim, Tesia Sker, Chanwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.22186)  

**Abstract**: In Machine Translation, Large Language Models (LLMs) have generally underperformed compared to conventional encoder-decoder systems and thus see limited adoption. However, LLMs excel at modeling contextual information, making them a natural fit for document-level translation tasks where coherence across sentences is crucial. Despite this potential, document-level MT with LLMs faces two key challenges: (1) the scarcity of large-scale, high-quality document-level parallel data; and (2) the propensity of LLMs to introduce hallucinations and omissions during generation. To address these challenges, we propose a two-stage fine-tuning strategy leveraging LLM-augmented document-level data. First, we augment data by converting summarization data into document-level parallel data using a LLM, and then filter it using multiple metrics, leveraging sacreBLEU, COMET, and LaBSE-based cosine similarity-to improve data quality. Finally, we employ a two-stage fine-tuning strategy: first fine-tuning on the abundant sentence-level MT resources, and then on the filtered document-level corpus. 

---
# The Semantic Ladder: A Framework for Progressive Formalization of Natural Language Content for Knowledge Graphs and AI Systems 

**Authors**: Lars Vogt  

**Link**: [PDF](https://arxiv.org/pdf/2603.22136)  

**Abstract**: Semantic data and knowledge infrastructures must reconcile two fundamentally different forms of representation: natural language, in which most knowledge is created and communicated, and formal semantic models, which enable machine-actionable integration, interoperability, and reasoning. Bridging this gap remains a central challenge, particularly when full semantic formalization is required at the point of data entry. Here, we introduce the Semantic Ladder, an architectural framework that enables the progressive formalization of data and knowledge. Building on the concept of modular semantic units as identifiable carriers of meaning, the framework organizes representations across levels of increasing semantic explicitness, ranging from natural language text snippets to ontology-based and higher-order logical models. Transformations between levels support semantic enrichment, statement structuring, and logical modelling while preserving semantic continuity and traceability. This approach enables the incremental construction of semantic knowledge spaces, reduces the semantic parsing burden, and supports the integration of heterogeneous representations, including natural language, structured semantic models, and vector-based embeddings. The Semantic Ladder thereby provides a foundation for scalable, interoperable, and AI-ready data and knowledge infrastructures. 

---
# Multiperspectivity as a Resource for Narrative Similarity Prediction 

**Authors**: Max Upravitelev, Veronika Solopova, Jing Yang, Charlott Jakob, Premtim Sahitaj, Ariana Sahitaj, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2603.22103)  

**Abstract**: Predicting narrative similarity can be understood as an inherently interpretive task: different, equally valid readings of the same text can produce divergent interpretations and thus different similarity judgments, posing a fundamental challenge for semantic evaluation benchmarks that encode a single ground truth. Rather than treating this multiperspectivity as a challenge to overcome, we propose to incorporate it in the decision making process of predictive systems. To explore this strategy, we created an ensemble of 31 LLM personas. These range from practitioners following interpretive frameworks to more intuitive, lay-style characters. Our experiments were conducted on the SemEval-2026 Task 4 dataset, where the system achieved an accuracy score of 0.705. Accuracy improves with ensemble size, consistent with Condorcet Jury Theorem-like dynamics under weakened independence. Practitioner personas perform worse individually but produce less correlated errors, yielding larger ensemble gains under majority voting. Our error analysis reveals a consistent negative association between gender-focused interpretive vocabulary and accuracy across all persona categories, suggesting either attention to dimensions not relevant for the benchmark or valid interpretations absent from the ground truth. This finding underscores the need for evaluation frameworks that account for interpretive plurality. 

---
# Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison 

**Authors**: Caio Vicentino  

**Link**: [PDF](https://arxiv.org/pdf/2603.22075)  

**Abstract**: We present a controlled empirical comparison between autoregressive (AR) and masked diffusion (MDLM) language models. Both models are trained on identical data (50M tokens from TinyStories), identical compute budget (20,000 steps, batch size 32, sequence length 512), and identical hardware (NVIDIA H100 80GB), isolating the generation paradigm as the sole variable. We report three findings. First, both paradigms achieve comparable training throughput (~50K tokens/second), with MDLM requiring only 4.7% more wall-clock time. Second, AR converges faster and begins overfitting by step 14,000, while MDLM converges more slowly and is still improving at step 20,000, suggesting different compute-optimal training regimes. Third, quantitative diversity analysis over 1,000 generated samples reveals a structural diversity-fluency trade-off: AR produces fluent but repetitive outputs (99.8% begin with the same word), while MDLM generates more diverse narratives (93.4% unique 5-word openings, higher Distinct-n, lower Self-BLEU), at the cost of occasional grammatical inconsistencies. All code, trained checkpoints, and data pipelines are released for reproducibility. 

---
# Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary Mismatch 

**Authors**: Stella Eva Tsiapali, Cong-Thanh Do, Kate Knill  

**Link**: [PDF](https://arxiv.org/pdf/2603.22056)  

**Abstract**: Large language models (LLMs) achieve state-of-the-art (SOTA) performance across language tasks, but are costly to deploy due to their size and resource demands. Knowledge Distillation (KD) addresses this by training smaller Student models to mimic larger Teacher models, improving efficiency without significant performance loss. Dual-Space Knowledge Distillation with Cross-Model Attention (DSKD-CMA) has emerged as a SOTA method for KD between LLMs with distinct tokenizers, yet its internal workings remain largely opaque. In this work, we systematically analyse the attention mechanism of DSKD-CMA through manual token alignment probing and heatmap visualisations, revealing both strengths and limitations. Building on this, we introduce a novel method, DSKD-CMA-GA, based on Generative Adversarial (GA) learning, to address the mismatched distributions between the keys and queries computed from distinct models. Experiments show modest but consistent ROUGE-L gains in text generation quality, particularly on out-of-distribution data (+0.37 on average), narrowing the gap between cross- and same-tokenizer KD. 

---
# Retrieving Climate Change Disinformation by Narrative 

**Authors**: Max Upravitelev, Veronika Solopova, Charlott Jakob, Premtim Sahitaj, Vera Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2603.22015)  

**Abstract**: Detecting climate disinformation narratives typically relies on fixed taxonomies, which do not accommodate emerging narratives. Thus, we re-frame narrative detection as a retrieval task: given a narrative's core message as a query, rank texts from a corpus by alignment with that narrative. This formulation requires no predefined label set and can accommodate emerging narratives. We repurpose three climate disinformation datasets (CARDS, Climate Obstruction, climate change subset of PolyNarrative) for retrieval evaluation and propose SpecFi, a framework that generates hypothetical documents to bridge the gap between abstract narrative descriptions and their concrete textual instantiations. SpecFi uses community summaries from graph-based community detection as few-shot examples for generation, achieving a MAP of 0.505 on CARDS without access to narrative labels. We further introduce narrative variance, an embedding-based difficulty metric, and show via partial correlation analysis that standard retrieval degrades on high-variance narratives (BM25 loses 63.4% of MAP), while SpecFi-CS remains robust (32.7% loss). Our analysis also reveals that unsupervised community summaries converge on descriptions close to expert-crafted taxonomies, suggesting that graph-based methods can surface narrative structure from unlabeled text. 

---
# Parameter-Efficient Fine-Tuning for Medical Text Summarization: A Comparative Study of Lora, Prompt Tuning, and Full Fine-Tuning 

**Authors**: Ulugbek Shernazarov, Rostislav Svitsov, Bin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2603.21970)  

**Abstract**: Fine-tuning large language models for domain-specific tasks such as medical text summarization demands substantial computational resources. Parameter-efficient fine-tuning (PEFT) methods offer promising alternatives by updating only a small fraction of parameters. This paper compares three adaptation approaches-Low-Rank Adaptation (LoRA), Prompt Tuning, and Full Fine-Tuning-across the Flan-T5 model family on the PubMed medical summarization dataset. Through experiments with multiple random seeds, we demonstrate that LoRA consistently outperforms full fine-tuning, achieving 43.52 +/- 0.18 ROUGE-1 on Flan-T5-Large with only 0.6% trainable parameters compared to 40.67 +/- 0.21 for full fine-tuning. Sensitivity analyses examine the impact of LoRA rank and prompt token count. Our findings suggest the low-rank constraint provides beneficial regularization, challenging assumptions about the necessity of full parameter updates. Code is available at this https URL 

---
# SLURP-TN : Resource for Tunisian Dialect Spoken Language Understanding 

**Authors**: Haroun Elleuch, Salima Mdhaffar, Yannick Estève, Fethi Bougares  

**Link**: [PDF](https://arxiv.org/pdf/2603.21940)  

**Abstract**: Spoken Language Understanding (SLU) aims to extract the semantic information from the speech utterance of user queries. It is a core component in a task-oriented dialogue system. With the spectacular progress of deep neural network models and the evolution of pre-trained language models, SLU has obtained significant breakthroughs. However, only a few high-resource languages have taken advantage of this progress due to the absence of SLU resources. In this paper, we seek to mitigate this obstacle by introducing SLURP-TN. This dataset was created by recording 55 native speakers uttering sentences in Tunisian dialect, manually translated from six SLURP domains. The result is an SLU Tunisian dialect dataset that comprises 4165 sentences recorded into around 5 hours of acoustic material. We also develop a number of Automatic Speech Recognition and SLU models exploiting SLUTP-TN. The Dataset and baseline models are available at: this https URL. 

---
# Ara-Best-RQ: Multi Dialectal Arabic SSL 

**Authors**: Haroun Elleuch, Ryan Whetten, Salima Mdhaffar, Yannick Estève, Fethi Bougares  

**Link**: [PDF](https://arxiv.org/pdf/2603.21900)  

**Abstract**: We present Ara-BEST-RQ, a family of self-supervised learning (SSL) models specifically designed for multi-dialectal Arabic speech processing. Leveraging 5,640 hours of crawled Creative Commons speech and combining it with publicly available datasets, we pre-train conformer-based BEST-RQ models up to 600M parameters. Our models are evaluated on dialect identification (DID) and automatic speech recognition (ASR) tasks, achieving state-of-the-art performance on the former while using fewer parameters than competing models. We demonstrate that family-targeted pre-training on Arabic dialects significantly improves downstream performance compared to multilingual or monolingual models trained on non-Arabic data. All models, code, and pre-processed datasets will be publicly released to support reproducibility and further research in Arabic speech technologies. 

---
# Riding Brainwaves in LLM Space: Understanding Activation Patterns Using Individual Neural Signatures 

**Authors**: Ajan Subramanian, Sumukh Bettadapura, Rohan Sathish  

**Link**: [PDF](https://arxiv.org/pdf/2603.21847)  

**Abstract**: Consumer-grade EEG is entering everyday devices, from earbuds to headbands, raising the question of whether language models can be adapted to individual neural responses. We test this by asking whether frozen LLM representations encode person-specific EEG signals, directions in activation space that predict one person's brain activity but not another's. Using word-level EEG from 30 participants reading naturalistic sentences (ZuCo corpus), we train a separate linear probe for each person, mapping hidden states from a frozen Qwen 2.5 7B to that individual's EEG power. Person-specific probes outperform a single population probe on every EEG feature tested; for high-gamma power, the person-specific probe achieves rho = 0.183, a ninefold improvement over the population probe (rho = 0.020, p < 10^-4). A negative control, fixation count, shows no person-specific advantage (p = 0.360); fixation count reflects word length and frequency rather than individual cognition. The individual directions are temporally stable (split-half cosine = 0.824), non-transferable across people (self rho = 0.369 vs. other rho = 0.143, p < 10^-19), and distinct from the shared population signal: person-specific probes retain predictive power after the population component is removed. The person-specific signal concentrates in the model's deep layers, rising consistently with depth and peaking at Layer 24 of 28. The results are consistent across architectures (LLaMA 3.1 8B) and survive word-level confound controls. Frozen language models contain stable, person-specific neural directions in their deep layers, providing a geometric foundation for EEG-driven personalization. 

---
# Select, Label, Evaluate: Active Testing in NLP 

**Authors**: Antonio Purificato, Maria Sofia Bucarelli, Andrea Bacciu, Amin Mantrach, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2603.21840)  

**Abstract**: Human annotation cost and time remain significant bottlenecks in Natural Language Processing (NLP), with test data annotation being particularly expensive due to the stringent requirement for low-error and high-quality labels necessary for reliable model evaluation. Traditional approaches require annotating entire test sets, leading to substantial resource requirements. Active Testing is a framework that selects the most informative test samples for annotation. Given a labeling budget, it aims to choose the subset that best estimates model performance while minimizing cost and human effort. In this work, we formalize Active Testing in NLP and we conduct an extensive benchmarking of existing approaches across 18 datasets and 4 embedding strategies spanning 4 different NLP tasks. The experiments show annotation reductions of up to 95%, with performance estimation accuracy difference from the full test set within 1%. Our analysis reveals variations in method effectiveness across different data characteristics and task types, with no single approach emerging as universally superior. Lastly, to address the limitation of requiring a predefined annotation budget in existing sample selection strategies, we introduce an adaptive stopping criterion that automatically determines the optimal number of samples. 

---
# Instruction Set and Language for Symbolic Regression 

**Authors**: Ezequiel Lopez-Rubio, Mario Pascual-Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2603.21836)  

**Abstract**: A fundamental but largely unaddressed obstacle in Symbolic regression (SR) is structural redundancy: every expression DAG with admits many distinct node-numbering schemes that all encode the same expression, each occupying a separate point in the search space and consuming fitness evaluations without adding diversity. We present IsalSR (Instruction Set and Language for Symbolic Regression), a representation framework that encodes expression DAGs as strings over a compact two-tier alphabet and computes a pruned canonical string -- a complete labeled-DAG isomorphism invariant -- that collapses all the equivalent representations into a single canonical form. 

---
# Politics of Questions in News: A Mixed-Methods Study of Interrogative Stances as Markers of Voice and Power 

**Authors**: Bros Victor, Barbini Matilde, Gerard Patrick, Gatica-Perez Daniel  

**Link**: [PDF](https://arxiv.org/pdf/2603.21823)  

**Abstract**: Interrogatives in news discourse have been examined in linguistics and conversation analysis, but mostly in broadcast interviews and relatively small, often English-language corpora, while large-scale computational studies of news rarely distinguish interrogatives from declaratives or differentiate their functions. This paper brings these strands together through a mixed-methods study of the "Politics of Questions" in contemporary French-language digital news. Using over one million articles published between January 2023 and June 2024, we automatically detect interrogative stances, approximate their functional types, and locate textual answers when present, linking these quantitative measures to a qualitatively annotated subcorpus grounded in semantic and pragmatic theories of questions. Interrogatives are sparse but systematically patterned: they mainly introduce or organize issues, with most remaining cases being information-seeking or echo-like, while explicitly leading or tag questions are rare. Although their density and mix vary across outlets and topics, our heuristic suggests that questions are overwhelmingly taken up within the same article and usually linked to a subsequent answer-like span, most often in the journalist's narrative voice and less often through quoted speech. Interrogative contexts are densely populated with named individuals, organizations, and places, whereas publics and broad social groups are mentioned much less frequently, suggesting that interrogative discourse tends to foreground already prominent actors and places and thus exhibits strong personalization. We show how interrogative stance, textual uptake, and voice can be operationalized at corpus scale, and argue that combining computational methods with pragmatic and sociological perspectives can help account for how questioning practices structure contemporary news discourse. 

---
# SemEval-2026 Task 12: Abductive Event Reasoning: Towards Real-World Event Causal Inference for Large Language Models 

**Authors**: Pengfei Cao, Mingxuan Yang, Yubo Chen, Chenlong Zhang, Mingxuan Liu, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.21720)  

**Abstract**: Understanding why real-world events occur is important for both natural language processing and practical decision-making, yet direct-cause inference remains underexplored in evidence-rich settings. To address this gap, we organized SemEval-2026 Task 12: Abductive Event Reasoning (AER).\footnote{The task data is available at this https URL} The task asks systems to identify the most plausible direct cause of a target event from supporting evidence. We formulate AER as an evidence-grounded multiple-choice benchmark that captures key challenges of real-world causal reasoning, including distributed evidence, indirect background factors, and semantically related but non-causal distractors. The shared task attracted 122 participants and received 518 submissions. This paper presents the task formulation, dataset construction pipeline, evaluation setup, and system results. AER provides a focused benchmark for abductive reasoning over real-world events and highlights challenges for future work on causal reasoning and multi-document understanding. 

---
# Probing How Scalable Table Data Enhances General Long-Context Reasoning 

**Authors**: Huaibing Xie, Guoliang Zhao, Yang Liu, Shihan Dou, Siming Huang, Yanling Xiao, Shaolei Wang, Yiting Liu, Cheng Zhang, Shaofan Liu, Pluto Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.21719)  

**Abstract**: As real-world tasks grow increasingly complex, long-context reasoning has become a core capability for Large Language Models (LLMs). However, few studies explore which data types are effective for long-context reasoning and why. We find that structured table data with periodic structures shows strong potential for long-context reasoning. Motivated by this observation, we mathematically analyze tabular dependency structures using mutual information, revealing periodic non-vanishing dependencies in table data. Furthermore, we systematically analyze the capabilities of structured table data, conduct relevant scaling experiments, and validate its underlying mechanisms for enhancing long-context reasoning, yielding several meaningful insights. Leveraging these insights, we propose a simple yet scalable pipeline(TableLong) for synthesizing high-quality, diverse, and verifiable structured table data to boost long-context reasoning via RL. Extensive experimental results demonstrate that table data significantly enhances the long-context reasoning capability of LLMs across multiple long-context benchmarks (+8.24\% on average), and even improves performance on out-of-domain benchmarks (+8.06\% on average). We hope that our insights provide practical guidance for effective post-training data to enhance long-context reasoning in LLMs. 

---
# Optimizing Multi-Agent Weather Captioning via Text Gradient Descent: A Training-Free Approach with Consensus-Aware Gradient Fusion 

**Authors**: Shixu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21673)  

**Abstract**: Generating interpretable natural language captions from weather time series data remains a significant challenge at the intersection of meteorological science and natural language processing. While recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in time series forecasting and analysis, existing approaches either produce numerical predictions without human-accessible explanations or generate generic descriptions lacking domain-specific depth. We introduce WeatherTGD, a training-free multi-agent framework that reinterprets collaborative caption refinement through the lens of Text Gradient Descent (TGD). Our system deploys three specialized LLM agents including a Statistical Analyst, a Physics Interpreter, and a Meteorology Expert that generate domain-specific textual gradients from weather time series observations. These gradients are aggregated through a novel Consensus-Aware Gradient Fusion mechanism that extracts common signals while preserving unique domain perspectives. The fused gradients then guide an iterative refinement process analogous to gradient descent, where each LLM-generated feedback signal updates the caption toward an optimal solution. Experiments on real-world meteorological datasets demonstrate that WeatherTGD achieves significant improvements in both LLM-based evaluation and human expert evaluation, substantially outperforming existing multi-agent baselines while maintaining computational efficiency through parallel agent execution. 

---
# TAMTRL: Teacher-Aligned Reward Reshaping for Multi-Turn Reinforcement Learning in Long-Context Compression 

**Authors**: Li Wang, Yandong Wang, Xin Yu, Kui Zhang, Tianhao Peng, Wenjun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21663)  

**Abstract**: The rapid progress of large language models (LLMs) has led to remarkable performance gains across a wide range of tasks. However, when handling long documents that exceed the model's context window limit, the entire context cannot be processed in a single pass, making chunk-wise processing necessary. This requires multiple turns to read different chunks and update memory. However, supervision is typically provided only by the final outcome, which makes it difficult to evaluate the quality of memory updates at each turn in the multi-turn training setting. This introduces a temporal credit assignment challenge. Existing approaches, such as LLM-as-a-judge or process reward models, incur substantial computational overhead and suffer from estimation noise. To better address the credit assignment problem in multi-turn memory training, we propose Teacher-Aligned Reward Reshaping for Multi-Turn Reinforcement Learning (TAMTRL). TAMTRL leverages relevant documents as teacher signals by aligning them with each turn of model input and assigns rewards through normalized probabilities in a self-supervised manner. This provides fine-grained learning signals for each memory update and improves long-context processing. Experiments with multiple models of varying scales across seven long-context benchmarks show that TAMTRL consistently outperforms strong baselines, demonstrating its effectiveness. Our code is available at this https URL. 

---
# A Comparative Analysis of LLM Memorization at Statistical and Internal Levels: Cross-Model Commonalities and Model-Specific Signatures 

**Authors**: Bowen Chen, Namgi Han, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2603.21658)  

**Abstract**: Memorization is a fundamental component of intelligence for both humans and LLMs. However, while LLM performance scales rapidly, our understanding of memorization lags. Due to limited access to the pre-training data of LLMs, most previous studies focus on a single model series, leading to isolated observations among series, making it unclear which findings are general or specific. In this study, we collect multiple model series (Pythia, OpenLLaMa, StarCoder, OLMo1/2/3) and analyze their shared or unique memorization behavior at both the statistical and internal levels, connecting individual observations while showing new findings. At the statistical level, we reveal that the memorization rate scales log-linearly with model size, and memorized sequences can be further compressed. Further analysis demonstrated a shared frequency and domain distribution pattern for memorized sequences. However, different models also show individual features under the above observations. At the internal level, we find that LLMs can remove certain injected perturbations, while memorized sequences are more sensitive. By decoding middle layers and attention head ablation, we revealed the general decoding process and shared important heads for memorization. However, the distribution of those important heads differs between families, showing a unique family-level feature. Through bridging various experiments and revealing new findings, this study paves the way for a universal and fundamental understanding of memorization in LLM. 

---
# DATASHI: A Parallel English-Tashlhiyt Corpus for Orthography Normalization and Low-Resource Language Processing 

**Authors**: Nasser-Eddine Monir, Zakaria Baou  

**Link**: [PDF](https://arxiv.org/pdf/2603.21571)  

**Abstract**: DATASHI is a new parallel English-Tashlhiyt corpus that fills a critical gap in computational resources for Amazigh languages. It contains 5,000 sentence pairs, including a 1,500-sentence subset with expert-standardized and non-standard user-generated versions, enabling systematic study of orthographic diversity and normalization. This dual design supports text-based NLP tasks - such as tokenization, translation, and normalization - and also serves as a foundation for read-speech data collection and multimodal alignment. Comprehensive evaluations with state-of-the-art Large Language Models (GPT-5, Claude-Sonnet-4.5, Gemini-2.5-Pro, Mistral, Qwen3-Max) show clear improvements from zero-shot to few-shot prompting, with Gemini-2.5-Pro achieving the lowest word and character-level error rates and exhibiting robust cross-lingual generalization. A fine-grained analysis of edit operations - deletions, substitutions, and insertions - across phonological classes (geminates, emphatics, uvulars, and pharyngeals) further highlights model-specific sensitivities to marked Tashlhiyt features and provides new diagnostic insights for low-resource Amazigh orthography normalization. 

---
# SynSym: A Synthetic Data Generation Framework for Psychiatric Symptom Identification 

**Authors**: Migyeong Kang, Jihyun Kim, Hyolim Jeon, Sunwoo Hwang, Jihyun An, Yonghoon Kim, Haewoon Kwak, Jisun An, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.21529)  

**Abstract**: Psychiatric symptom identification on social media aims to infer fine-grained mental health symptoms from user-generated posts, allowing a detailed understanding of users' mental states. However, the construction of large-scale symptom-level datasets remains challenging due to the resource-intensive nature of expert labeling and the lack of standardized annotation guidelines, which in turn limits the generalizability of models to identify diverse symptom expressions from user-generated text. To address these issues, we propose SynSym, a synthetic data generation framework for constructing generalizable datasets for symptom identification. Leveraging large language models (LLMs), SynSym constructs high-quality training samples by (1) expanding each symptom into sub-concepts to enhance the diversity of generated expressions, (2) producing synthetic expressions that reflect psychiatric symptoms in diverse linguistic styles, and (3) composing realistic multi-symptom expressions, informed by clinical co-occurrence patterns. We validate SynSym on three benchmark datasets covering different styles of depressive symptom expression. Experimental results demonstrate that models trained solely on the synthetic data generated by SynSym perform comparably to those trained on real data, and benefit further from additional fine-tuning with real data. These findings underscore the potential of synthetic data as an alternative resource to real-world annotations in psychiatric symptom modeling, and SynSym serves as a practical framework for generating clinically relevant and realistic symptom expressions. 

---
# CatRAG: Functor-Guided Structural Debiasing with Retrieval Augmentation for Fair LLMs 

**Authors**: Ravi Ranjan, Utkarsh Grover, Mayur Akewar, Xiaomin Lin, Agoritsa Polyzou  

**Link**: [PDF](https://arxiv.org/pdf/2603.21524)  

**Abstract**: Large Language Models (LLMs) are deployed in high-stakes settings but can show demographic, gender, and geographic biases that undermine fairness and trust. Prior debiasing methods, including embedding-space projections, prompt-based steering, and causal interventions, often act at a single stage of the pipeline, resulting in incomplete mitigation and brittle utility trade-offs under distribution shifts. We propose CatRAG Debiasing, a dual-pronged framework that integrates functor with Retrieval-Augmented Generation (RAG) guided structural debiasing. The functor component leverages category-theoretic structure to induce a principled, structure-preserving projection that suppresses bias-associated directions in the embedding space while retaining task-relevant semantics. On the Bias Benchmark for Question Answering (BBQ) across three open-source LLMs (Meta Llama-3, OpenAI GPT-OSS, and Google Gemma-3), CatRAG achieves state-of-the-art results, improving accuracy by up to 40% over the corresponding base models and by more than 10% over prior debiasing methods, while reducing bias scores to near zero (from 60% for the base models) across gender, nationality, race, and intersectional subgroups. 

---
# Generalizable Self-Evolving Memory for Automatic Prompt Optimization 

**Authors**: Guanbao Liang, Yuanchen Bei, Sheng Zhou, Yuheng Qin, Huan Zhou, Bingxin Jia, Bin Li, Jiajun Bu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21520)  

**Abstract**: Automatic prompt optimization is a promising approach for adapting large language models (LLMs) to downstream tasks, yet existing methods typically search for a specific prompt specialized to a fixed task. This paradigm limits generalization across heterogeneous queries and prevents models from accumulating reusable prompting knowledge over time. In this paper, we propose MemAPO, a memory-driven framework that reconceptualizes prompt optimization as generalizable and self-evolving experience accumulation. MemAPO maintains a dual-memory mechanism that distills successful reasoning trajectories into reusable strategy templates while organizing incorrect generations into structured error patterns that capture recurrent failure modes. Given a new prompt, the framework retrieves both relevant strategies and failure patterns to compose prompts that promote effective reasoning while discouraging known mistakes. Through iterative self-reflection and memory editing, MemAPO continuously updates its memory, enabling prompt optimization to improve over time rather than restarting from scratch for each task. Experiments on diverse benchmarks show that MemAPO consistently outperforms representative prompt optimization baselines while substantially reducing optimization cost. 

---
# Triangulating Temporal Dynamics in Multilingual Swiss Online News 

**Authors**: Bros Victor, Dufraisse Evan, Popescu Adrian, Gatica-Perez Daniel  

**Link**: [PDF](https://arxiv.org/pdf/2603.21519)  

**Abstract**: Analyzing news coverage in multilingual societies can offer valuable insights into the dynamics of public discourse and the development of collective narratives, yet comprehensive studies that account for linguistic and cultural diversity within national media ecosystems remain limited, particularly in complex contexts such as Switzerland. This paper studies temporal trends in Swiss digital media across the country's three main linguistic regions, French, German, and Italian, using a triangulated methodology that combines quantitative analyses with qualitative insights. We collected and processed over 1.7 million news articles, applying lexical metrics, named entity recognition and Wikidata-based linking, targeted sentiment analysis, and consensus-based change-point detection. To enable principled cross-language comparisons and to connect to theories of domestication and cultural proximity, we derive domestication profiles together with a proximity salience ratio. Our analysis spans thematic, recurrent, and singular events. By integrating quantitative data with qualitative interpretation, we provide new insights into the dynamics of Swiss digital media and demonstrate the usefulness of triangulation in media studies. The findings reveal distinct temporal patterns and highlight how linguistic and cultural contexts influence reporting. Our approach offers a framework applicable to other multilingual or culturally diverse media environments, contributing to a deeper understanding of how news is shaped by linguistic and cultural factors. 

---
# Agentic Automation of BT-RADS Scoring: End-to-End Multi-Agent System for Standardized Brain Tumor Follow-up Assessment 

**Authors**: Mohamed Sobhi Jabal, Jikai Zhang, Dominic LaBella, Jessica L. Houk, Dylan Zhang, Jeffrey D. Rudie, Kirti Magudia, Maciej A. Mazurowski, Evan Calabrese  

**Link**: [PDF](https://arxiv.org/pdf/2603.21494)  

**Abstract**: The Brain Tumor Reporting and Data System (BT-RADS) standardizes post-treatment MRI response assessment in patients with diffuse gliomas but requires complex integration of imaging trends, medication effects, and radiation timing. This study evaluates an end-to-end multi-agent large language model (LLM) and convolutional neural network (CNN) system for automated BT-RADS classification. A multi-agent LLM system combined with automated CNN-based tumor segmentation was retrospectively evaluated on 509 consecutive post-treatment glioma MRI examinations from a single high-volume center. An extractor agent identified clinical variables (steroid status, bevacizumab status, radiation date) from unstructured clinical notes, while a scorer agent applied BT-RADS decision logic integrating extracted variables with volumetric measurements. Expert reference standard classifications were established by an independent board-certified neuroradiologist. Of 509 examinations, 492 met inclusion criteria. The system achieved 374/492 (76.0%; 95% CI, 72.1%-79.6%) accuracy versus 283/492 (57.5%; 95% CI, 53.1%-61.8%) for initial clinical assessments (+18.5 percentage points; P<.001). Context-dependent categories showed high sensitivity (BT-1b 100%, BT-1a 92.7%, BT-3a 87.5%), while threshold-dependent categories showed moderate sensitivity (BT-3c 74.8%, BT-2 69.2%, BT-4 69.3%, BT-3b 57.1%). For BT-4, positive predictive value was 92.9%. The multi-agent LLM system achieved higher BT-RADS classification agreement with expert reference standard compared to initial clinical scoring, with high accuracy for context-dependent scores and high positive predictive value for BT-4 detection. 

---
# Effective Strategies for Asynchronous Software Engineering Agents 

**Authors**: Jiayi Geng, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2603.21489)  

**Abstract**: AI agents have become increasingly capable at isolated software engineering (SWE) tasks such as resolving issues on Github. Yet long-horizon tasks involving multiple interdependent subtasks still pose challenges both with respect to accuracy, and with respect to timely completion. A natural approach to solving these long-horizon tasks in a timely manner is asynchronous multi-agent collaboration, where multiple agents work on different parts of the task at the same time. But effective application of multi-agent systems has proven surprisingly difficult: concurrent edits by multiple agents interfere with each other, dependencies are difficult to synchronize, and combining partial progress into a coherent whole is challenging. On the other hand, human developers have long relied on mature collaboration infrastructure to manage these challenges in large software projects. Inspired by these collaboration primitives, we introduce Centralized Asynchronous Isolated Delegation (CAID), a structured multi-agent coordination paradigm grounded in three core SWE primitives: centralized task delegation, asynchronous execution, and isolated workspaces. CAID constructs dependency-aware task plans through a central manager, executes subtasks concurrently in isolated workspaces, and consolidates progress via structured integration with executable test-based verification. In empirical evaluation, we find that CAID improves accuracy over single-agent baselines by 26.7% absolute on paper reproduction tasks (PaperBench) and 14.3% on Python library development tasks (Commit0). Through systematic analysis, we find that branch-and-merge is a central coordination mechanism for multi-agent collaboration, and that SWE primitives such as git worktree, git commit, and git merge enable it to be realized in a reliable and executable manner. 

---
# TaigiSpeech: A Low-Resource Real-World Speech Intent Dataset and Preliminary Results with Scalable Data Mining In-the-Wild 

**Authors**: Kai-Wei Chang, Yi-Cheng Lin, Huang-Cheng Chou, Wenze Ren, Yu-Han Huang, Yun-Shao Tsai, Chien-Cheng Chen, Yu Tsao, Yuan-Fu Liao, Shrikanth Narayanan, James Glass, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2603.21478)  

**Abstract**: Speech technologies have advanced rapidly and serve diverse populations worldwide. However, many languages remain underrepresented due to limited resources. In this paper, we introduce \textbf{TaigiSpeech}, a real-world speech intent dataset in Taiwanese Taigi (aka Taiwanese Hokkien/Southern Min), which is a low-resource and primarily spoken language. The dataset is collected from older adults, comprising 21 speakers with a total of 3k utterances. It is designed for practical intent detection scenarios, including healthcare and home assistant applications. To address the scarcity of labeled data, we explore two data mining strategies with two levels of supervision: keyword match data mining with LLM pseudo labeling via an intermediate language and an audio-visual framework that leverages multimodal cues with minimal textual supervision. This design enables scalable dataset construction for low-resource and unwritten spoken languages. TaigiSpeech will be released under the CC BY 4.0 license to facilitate broad adoption and research on low-resource and unwritten languages. The project website and the dataset can be found on this https URL. 

---
# DRTriton: Large-Scale Synthetic Data Reinforcement Learning for Triton Kernel Generation 

**Authors**: Siqi Guo, Ming Lin, Tianbao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.21465)  

**Abstract**: Developing efficient CUDA kernels is a fundamental yet challenging task in the generative AI industry. Recent researches leverage Large Language Models (LLMs) to automatically convert PyTorch reference implementations to CUDA kernels, significantly reducing the engineering efforts. State-of-the-art LLMs, such as GPT-5.2 and Claude-Sonnet-4.5, still struggle in this specific task. To address this challenge, we propose DRTriton, a scalable learning framework for training LLMs to convert PyTorch codes into highly optimized Triton kernels, which are then compiled to CUDA kernels at runtime. DRTriton consists of three key components: (i) a data synthetic algorithm CSP-DAG that guarantees full coverage and unbiased uniform sampling over the operator space with controlled difficulty; (ii) a curriculum reinforcement learning with decoupled reward efficiently optimizes conversion success rate and inference speed simultaneously; and (iii) a test-time search algorithm that further improves the inference speed of the generated Triton kernels. Notably, despite being trained exclusively on synthetic data, DRTriton generalizes effectively to real-world CUDA kernels that are challenging even for human experts. Experimental results show that DRTriton-7B achieves speedup on 92% of the KernelBench Level 2, compared to 23% for GPT-5.2 and 19% for Claude-Sonnet-4.5. 

---
# Cross-Context Verification: Hierarchical Detection of Benchmark Contamination through Session-Isolated Analysis 

**Authors**: Tae-Eun Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.21454)  

**Abstract**: LLM coding benchmarks face a credibility crisis: widespread solution leakage and test quality issues undermine SWE-bench Verified, while existing detection methods--paraphrase consistency, n-gram overlap, perplexity analysis--never directly observe whether a model reasons or recalls. Meanwhile, simply repeating verification degrades accuracy: multi-turn review generates false positives faster than it discovers true errors, suggesting that structural approaches are needed.
We introduce Cross-Context Verification (CCV), a black-box method that solves the same benchmark problem in N independent sessions and measures solution diversity, combined with the Hierarchical Cross-Context Architecture (HCCA), a multi-agent analysis framework that prevents confirmation bias through intentional information restriction across specialized analytical roles.
On 9 SWE-bench Verified problems (45 trials, Claude Opus 4.6, temperature 0), CCV achieves perfect separation between contaminated and genuine reasoning (Mann-Whitney U=0, p approx 0.012, r = 1.0). Key findings: (1) contamination is binary--models either recall perfectly or not at all; (2) reasoning absence is a perfect discriminator; (3) 33% of prior contamination labels are false positives; (4) HCCA's independent analysis structure discovers contamination-flaw composite cases that single-analyst approaches miss. A pilot experiment extending HCCA to multi-stage verification (Worker to Verifier to Director) yields a negative result--100% sycophantic confirmation--providing further evidence that information restriction, not structural complexity, is the key mechanism. We release all code and data. 

---
# KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning 

**Authors**: Shuai Wang, Yinan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21440)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL. 

---
# PROMPT2BOX: Uncovering Entailment Structure among LLM Prompts 

**Authors**: Neeladri Bhuiya, Shib Sankar Dasgupta, Andrew McCallum, Haw-Shiuan Chang  

**Link**: [PDF](https://arxiv.org/pdf/2603.21438)  

**Abstract**: To discover the weaknesses of LLMs, researchers often embed prompts into a vector space and cluster them to extract insightful patterns. However, vector embeddings primarily capture topical similarity. As a result, prompts that share a topic but differ in specificity, and consequently in difficulty, are often represented similarly, making fine-grained weakness analysis difficult. To address this limitation, we propose PROMPT2BOX, which embeds prompts into a box embedding space using a trained encoder. The encoder, trained on existing and synthesized datasets, outputs box embeddings that capture not only semantic similarity but also specificity relations between prompts (e.g., "writing an adventure story" is more specific than "writing a story"). We further develop a novel dimension reduction technique for box embeddings to facilitate dataset visualization and comparison. Our experiments demonstrate that box embeddings consistently capture prompt specificity better than vector baselines. On the downstream task of creating hierarchical clustering trees for 17 LLMs from the UltraFeedback dataset, PROMPT2BOX can identify 8.9\% more LLM weaknesses than vector baselines and achieves an approximately 33\% stronger correlation between hierarchical depth and instruction specificity. 

---
# Semantic Shift: the Fundamental Challenge in Text Embedding and Retrieval 

**Authors**: Hang Gao, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2603.21437)  

**Abstract**: Transformer-based embedding models rely on pooling to map variable-length text into a single vector, enabling efficient similarity search but also inducing well-known geometric pathologies such as anisotropy and length-induced embedding collapse. Existing accounts largely describe \emph{what} these pathologies look like, yet provide limited insight into \emph{when} and \emph{why} they harm downstream retrieval. In this work, we argue that the missing causal factor is \emph{semantic shift}: the intrinsic, structured evolution and dispersion of semantics within a text.
We first present a theoretical analysis of \emph{semantic smoothing} in Transformer embeddings: as the semantic diversity among constituent sentences increases, the pooled representation necessarily shifts away from every individual sentence embedding, yielding a smoothed and less discriminative vector. Building on this foundation, we formalize semantic shift as a computable measure integrating local semantic evolution and global semantic dispersion. Through controlled experiments across corpora and multiple embedding models, we show that semantic shift aligns closely with the severity of embedding concentration and predicts retrieval degradation, whereas text length alone does not. Overall, semantic shift offers a unified and actionable lens for understanding embedding collapse and for diagnosing when anisotropy becomes harmful. 

---
# Efficient Fine-Tuning Methods for Portuguese Question Answering: A Comparative Study of PEFT on BERTimbau and Exploratory Evaluation of Generative LLMs 

**Authors**: Mariela M. Nina, Caio Veloso Costa, Lilian Berton, Didier A. Vega-Oliveros  

**Link**: [PDF](https://arxiv.org/pdf/2603.21418)  

**Abstract**: Although large language models have transformed natural language processing, their computational costs create accessibility barriers for low-resource languages such as Brazilian Portuguese. This work presents a systematic evaluation of Parameter-Efficient Fine-Tuning (PEFT) and quantization techniques applied to BERTimbau for Question Answering on SQuAD-BR, the Brazilian Portuguese translation of SQuAD v1. We evaluate 40 configurations combining four PEFT methods (LoRA, DoRA, QLoRA, QDoRA) across two model sizes (Base: 110M, Large: 335M parameters). Our findings reveal three critical insights: (1) LoRA achieves 95.8\% of baseline performance on BERTimbau-Large while reducing training time by 73.5\% (F1=81.32 vs 84.86); (2) higher learning rates (2e-4) substantially improve PEFT performance, with F1 gains of up to +19.71 points over standard rates; and (3) larger models show twice the quantization resilience (loss of 4.83 vs 9.56 F1 points). These results demonstrate that encoder-based models can be efficiently fine-tuned for extractive Brazilian Portuguese QA with substantially lower computational cost than large generative LLMs, promoting more sustainable approaches aligned with \textit{Green AI} principles. An exploratory evaluation of Tucano and Sabiá on the same extractive QA benchmark shows that while generative models can reach competitive F1 scores with LoRA fine-tuning, they require up to 4.2$\times$ more GPU memory and 3$\times$ more training time than BERTimbau-Base, reinforcing the efficiency advantage of smaller encoder-based architectures for this task. 

---
# Multi-Perspective LLM Annotations for Valid Analyses in Subjective Tasks 

**Authors**: Navya Mehrotra, Adam Visokay, Kristina Gligorić  

**Link**: [PDF](https://arxiv.org/pdf/2603.21404)  

**Abstract**: Large language models are increasingly used to annotate texts, but their outputs reflect some human perspectives better than others. Existing methods for correcting LLM annotation error assume a single ground truth. However, this assumption fails in subjective tasks where disagreement across demographic groups is meaningful. Here we introduce Perspective-Driven Inference, a method that treats the distribution of annotations across groups as the quantity of interest, and estimates it using a small human annotation budget. We contribute an adaptive sampling strategy that concentrates human annotation effort on groups where LLM proxies are least accurate. We evaluate on politeness and offensiveness rating tasks, showing targeted improvements for harder-to-model demographic groups relative to uniform sampling baselines, while maintaining coverage. 

---
# Task-Specific Efficiency Analysis: When Small Language Models Outperform Large Language Models 

**Authors**: Jinghan Cao, Yu Ma, Xinjin Li, Qingyang Ren, Xiangyun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.21389)  

**Abstract**: Large Language Models achieve remarkable performance but incur substantial computational costs unsuitable for resource-constrained deployments. This paper presents the first comprehensive task-specific efficiency analysis comparing 16 language models across five diverse NLP tasks. We introduce the Performance-Efficiency Ratio (PER), a novel metric integrating accuracy, throughput, memory, and latency through geometric mean normalization. Our systematic evaluation reveals that small models (0.5--3B parameters) achieve superior PER scores across all given tasks. These findings establish quantitative foundations for deploying small models in production environments prioritizing inference efficiency over marginal accuracy gains. 

---
# Conspiracy Frame: a Semiotically-Driven Approach for Conspiracy Theories Detection 

**Authors**: Heidi Campana Piva, Shaina Ashraf, Maziar Kianimoghadam Jouneghani, Arianna Longo, Rossana Damiano, Lucie Flek, Marco Antonio Stranisci  

**Link**: [PDF](https://arxiv.org/pdf/2603.21368)  

**Abstract**: Conspiracy theories are anti-authoritarian narratives that lead to social conflict, impacting how people perceive political information. To help in understanding this issue, we introduce the Conspiracy Frame: a fine-grained semantic representation of conspiratorial narratives derived from frame-semantics and semiotics, which spawned the Conspiracy Frames (this http URL.) dataset: a corpus of Telegram messages annotated at span-level. The Conspiracy Frame and this http URL. dataset contribute to the implementation of a more generalizable understanding and recognition of conspiracy theories. We observe the ability of LLMs to recognize this phenomenon in-domain and out-of-domain, investigating the role that frames may have in supporting this task. Results show that, while the injection of frames in an in-context approach does not lead to clear increase of performance, it has potential; the mapping of annotated spans with FrameNet shows abstract semantic patterns (e.g., `Kinship', `Ingest\_substance') that potentially pave the way for a more semantically- and semiotically-aware detection of conspiratorial narratives. 

---
# Benchmarking Bengali Dialectal Bias: A Multi-Stage Framework Integrating RAG-Based Translation and Human-Augmented RLAIF 

**Authors**: K. M. Jubair Sami, Dipto Sumit, Ariyan Hossain, Farig Sadeque  

**Link**: [PDF](https://arxiv.org/pdf/2603.21359)  

**Abstract**: Large language models (LLMs) frequently exhibit performance biases against regional dialects of low-resource languages. However, frameworks to quantify these disparities remain scarce. We propose a two-phase framework to evaluate dialectal bias in LLM question-answering across nine Bengali dialects. First, we translate and gold-label standard Bengali questions into dialectal variants adopting a retrieval-augmented generation (RAG) pipeline to prepare 4,000 question sets. Since traditional translation quality evaluation metrics fail on unstandardized dialects, we evaluate fidelity using an LLM-as-a-judge, which human correlation confirms outperforms legacy metrics. Second, we benchmark 19 LLMs across these gold-labeled sets, running 68,395 RLAIF evaluations validated through multi-judge agreement and human fallback. Our findings reveal severe performance drops linked to linguistic divergence. For instance, responses to the highly divergent Chittagong dialect score 5.44/10, compared to 7.68/10 for Tangail. Furthermore, increased model scale does not consistently mitigate this bias. We contribute a validated translation quality evaluation method, a rigorous benchmark dataset, and a Critical Bias Sensitivity (CBS) metric for safety-critical applications. 

---
# Beyond Memorization: Distinguishing between Reductive and Epistemic Reasoning in LLMs using Classic Logic Puzzles 

**Authors**: Adi Gabay, Gabriel Stanovsky, Liat Peterfreund  

**Link**: [PDF](https://arxiv.org/pdf/2603.21350)  

**Abstract**: Epistemic reasoning requires agents to infer the state of the world from partial observations and information about other agents' knowledge. Prior work evaluating LLMs on canonical epistemic puzzles interpreted their behavior through a dichotomy between epistemic reasoning and brittle memorization. We argue that this framing is incomplete: in recent models, memorization is better understood as a special case of reduction, where a new instance is mapped onto a known problem. Instead, we introduce a reduction ladder, a sequence of modifications that progressively move instances away from a canonical epistemic puzzle, making reduction increasingly difficult while preserving the underlying logic. We find that while some large models succeed via reduction, other models fail early, and all models struggle once epistemic reasoning is required. 

---
# TimeTox: An LLM-Based Pipeline for Automated Extraction of Time Toxicity from Clinical Trial Protocols 

**Authors**: Saketh Vinjamuri, Marielle Fis Loperena, Marie C. Spezia, Ramez Kouzy  

**Link**: [PDF](https://arxiv.org/pdf/2603.21335)  

**Abstract**: Time toxicity, the cumulative healthcare contact days from clinical trial participation, is an important but labor-intensive metric to extract from protocol documents. We developed TimeTox, an LLM-based pipeline for automated extraction of time toxicity from Schedule of Assessments tables. TimeTox uses Google's Gemini models in three stages: summary extraction from full-length protocol PDFs, time toxicity quantification at six cumulative timepoints for each treatment arm, and multi-run consensus via position-based arm matching. We validated against 20 synthetic schedules (240 comparisons) and assessed reproducibility on 644 real-world oncology protocols. Two architectures were compared: single-pass (vanilla) and two-stage (structure-then-count). The two-stage pipeline achieved 100% clinically acceptable accuracy ($\pm$3 days) on synthetic data (MAE 0.81 days) versus 41.5% for vanilla (MAE 9.0 days). However, on real-world protocols, the vanilla pipeline showed superior reproducibility: 95.3% clinically acceptable accuracy (IQR $\leq$ 3 days) across 3 runs on 644 protocols, with 82.0% perfect stability (IQR = 0). The production pipeline extracted time toxicity for 1,288 treatment arms across multiple disease sites. Extraction stability on real-world data, rather than accuracy on synthetic benchmarks, is the decisive factor for production LLM deployment. 

---
# enhancing reasoning accuracy in large language models during inference time 

**Authors**: Vinay Sharma, Manish Jain  

**Link**: [PDF](https://arxiv.org/pdf/2603.21301)  

**Abstract**: Large Language Models (LLMs) often exhibit strong linguistic abilities while remaining unreliable on multi-step reasoning tasks, particularly when deployed without additional training or fine-tuning. In this work, we study inference-time techniques to improve the reasoning accuracy of LLMs. We systematically evaluate three classes of inference-time strategies: (i) self-consistency via stochastic decoding, where the model is sampled multiple times using controlled temperature and nucleus sampling and the most frequent final answer is selected; (ii) dual-model reasoning agreement, where outputs from two independent models are compared and only consistent reasoning traces are trusted; and (iii) self-reflection, where the model critiques and revises its own reasoning. Across all evaluated methods, we employ Chain-of-Thought (CoT) [1] prompting to elicit explicit intermediate reasoning steps before generating final answers. In this work, we provide a controlled comparative evaluation across three inference-time strategies under identical prompting and verification settings. Our experiments on LLM [2] show that self-consistency with nucleus sampling and controlled temperature value yields the substantial gains, achieving a 9% to 15% absolute improvement in accuracy over greedy single-pass decoding, well-suited for low-risk domains, offering meaningful gains with minimal overhead. The dual-model approach provides additional confirmation for model reasoning steps thus more appropriate for moderate-risk domains, where higher reliability justifies additional compute. Self-reflection offers only marginal improvements, suggesting limited effectiveness for smaller non-reasoning models at inference time. 

---
# More Than Sum of Its Parts: Deciphering Intent Shifts in Multimodal Hate Speech Detection 

**Authors**: Runze Sun, Yu Zheng, Zexuan Xiong, Zhongjin Qu, Lei Chen, Jiwen Lu, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.21298)  

**Abstract**: Combating hate speech on social media is critical for securing cyberspace, yet relies heavily on the efficacy of automated detection systems. As content formats evolve, hate speech is transitioning from solely plain text to complex multimodal expressions, making implicit attacks harder to spot. Current systems, however, often falter on these subtle cases, as they struggle with multimodal content where the emergent meaning transcends the aggregation of individual modalities. To bridge this gap, we move beyond binary classification to characterize semantic intent shifts where modalities interact to construct implicit hate from benign cues or neutralize toxicity through semantic inversion. Guided by this fine-grained formulation, we curate the Hate via Vision-Language Interplay (H-VLI) benchmark where the true intent hinges on the intricate interplay of modalities rather than overt visual or textual slurs. To effectively decipher these complex cues, we further propose the Asymmetric Reasoning via Courtroom Agent DEbate (ARCADE) framework. By simulating a judicial process where agents actively argue for accusation and defense, ARCADE forces the model to scrutinize deep semantic cues before reaching a verdict. Extensive experiments demonstrate that ARCADE significantly outperforms state-of-the-art baselines on H-VLI, particularly for challenging implicit cases, while maintaining competitive performance on established benchmarks. Our code and data are available at: this https URL 

---
# Conversation Tree Architecture: A Structured Framework for Context-Aware Multi-Branch LLM Conversations 

**Authors**: Pranav Hemanth, Sampriti Saha  

**Link**: [PDF](https://arxiv.org/pdf/2603.21278)  

**Abstract**: Large language models (LLMs) are increasingly deployed for extended, multi-topic conversations, yet the flat, append-only structure of current conversation interfaces introduces a fundamental limitation: all context accumulates in a single unbounded window, causing topically distinct threads to bleed into one another and progressively degrade response quality. We term this failure mode logical context poisoning. In this paper, we introduce the Conversation Tree Architecture (CTA), a hierarchical framework that organizes LLM conversations as trees of discrete, context-isolated nodes. Each node maintains its own local context window; structured mechanisms govern how context flows between parent and child nodes, downstream on branch creation and upstream on branch deletion. We additionally introduce volatile nodes, transient branches whose local context must be selectively merged upward or permanently discarded before purging. We formalize the architecture's primitives, characterize the open design problems in context flow, relate our framework to prior work in LLM memory management, and describe a working prototype implementation. The CTA provides a principled foundation for structured conversational context management and extends naturally to multi-agent settings. 

---
# Graph Fusion Across Languages using Large Language Models 

**Authors**: Kaung Myat Kyaw, Khush Agarwal, Jonathan Chan  

**Link**: [PDF](https://arxiv.org/pdf/2603.21248)  

**Abstract**: Combining multiple knowledge graphs (KGs) across linguistic boundaries is a persistent challenge due to semantic heterogeneity and the complexity of graph environments. We propose a framework for cross-lingual graph fusion, leveraging the in-context reasoning and multilingual semantic priors of Large Language Models (LLMs). The framework implements structural linearization by mapping triplets directly into natural language sequences (e.g., [head] [relation] [tail]), enabling the LLM to map relations and reconcile entities between an evolving fused graph ($G_{c}^{(t-1)}$) and a new candidate graph ($G_{t}$). Evaluated on the DBP15K dataset, this exploratory study demonstrates that LLMs can serve as a universal semantic bridge to resolve cross-lingual discrepancies. Results show the successful sequential agglomeration of multiple heterogeneous graphs, offering a scalable, modular solution for continuous knowledge synthesis in multi-source, multilingual environments. 

---
# Context Selection for Hypothesis and Statistical Evidence Extraction from Full-Text Scientific Articles 

**Authors**: Sai Koneru, Jian Wu, Sarah Rajtmajer  

**Link**: [PDF](https://arxiv.org/pdf/2603.21193)  

**Abstract**: Extracting hypotheses and their supporting statistical evidence from full-text scientific articles is central to the synthesis of empirical findings, but remains difficult due to document length and the distribution of scientific arguments across sections of the paper. The work studies a sequential full-text extraction setting, where the statement of a primary finding in an article's abstract is linked to (i) a corresponding hypothesis statement in the paper body and (ii) the statistical evidence that supports or refutes that hypothesis. This formulation induces a challenging within-document retrieval setting in which many candidate paragraphs are topically related to the finding but differ in rhetorical role, creating hard negatives for retrieval and extraction. Using a two-stage retrieve-and-extract framework, we conduct a controlled study of retrieval design choices, varying context quantity, context quality (standard Retrieval Augmented Generation, reranking, and a fine-tuned retriever paired with reranking), as well as an oracle paragraph setting to separate retrieval failures from extraction limits across four Large Language Model extractors. We find that targeted context selection consistently improves hypothesis extraction relative to full-text prompting, with gains concentrated in configurations that optimize retrieval quality and context cleanliness. In contrast, statistical evidence extraction remains substantially harder. Even with oracle paragraphs, performance remains moderate, indicating persistent extractor limitations in handling hybrid numeric-textual statements rather than retrieval failures alone. 

---
# Explainable Semantic Textual Similarity via Dissimilar Span Detection 

**Authors**: Diego Miguel Lozano, Daryna Dementieva, Alexander Fraser  

**Link**: [PDF](https://arxiv.org/pdf/2603.21174)  

**Abstract**: Semantic Textual Similarity (STS) is a crucial component of many Natural Language Processing (NLP) applications. However, existing approaches typically reduce semantic nuances to a single score, limiting interpretability. To address this, we introduce the task of Dissimilar Span Detection (DSD), which aims to identify semantically differing spans between pairs of texts. This can help users understand which particular words or tokens negatively affect the similarity score, or be used to improve performance in STS-dependent downstream tasks. Furthermore, we release a new dataset suitable for the task, the Span Similarity Dataset (SSD), developed through a semi-automated pipeline combining large language models (LLMs) with human verification. We propose and evaluate different baseline methods for DSD, both unsupervised, based on LIME, SHAP, LLMs, and our own method, as well as an additional supervised approach. While LLMs and supervised models achieve the highest performance, overall results remain low, highlighting the complexity of the task. Finally, we set up an additional experiment that shows how DSD can lead to increased performance in the specific task of paraphrase detection. 

---
# Entropy Alone is Insufficient for Safe Selective Prediction in LLMs 

**Authors**: Edward Phillips, Fredrik K. Gustafsson, Sean Wu, Anshul Thakur, David A. Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2603.21172)  

**Abstract**: Selective prediction systems can mitigate harms resulting from language model hallucinations by abstaining from answering in high-risk cases. Uncertainty quantification techniques are often employed to identify such cases, but are rarely evaluated in the context of the wider selective prediction policy and its ability to operate at low target error rates. We identify a model-dependent failure mode of entropy-based uncertainty methods that leads to unreliable abstention behaviour, and address it by combining entropy scores with a correctness probe signal. We find that across three QA benchmarks (TriviaQA, BioASQ, MedicalQA) and four model families, the combined score generally improves both the risk--coverage trade-off and calibration performance relative to entropy-only baselines. Our results highlight the importance of deployment-facing evaluation of uncertainty methods, using metrics that directly reflect whether a system can be trusted to operate at a stated risk level. 

---
# Many Dialects, Many Languages, One Cultural Lens: Evaluating Multilingual VLMs for Bengali Culture Understanding Across Historically Linked Languages and Regional Dialects 

**Authors**: Nurul Labib Sayeedi, Md. Faiyaz Abdullah Sayeedi, Shubhashis Roy Dipta, Rubaya Tabassum, Ariful Ekraj Hridoy, Mehraj Mahmood, Mahbub E Sobhani, Md. Tarek Hasan, Swakkhar Shatabda  

**Link**: [PDF](https://arxiv.org/pdf/2603.21165)  

**Abstract**: Bangla culture is richly expressed through region, dialect, history, food, politics, media, and everyday visual life, yet it remains underrepresented in multimodal evaluation. To address this gap, we introduce BanglaVerse, a culturally grounded benchmark for evaluating multilingual vision-language models (VLMs) on Bengali culture across historically linked languages and regional dialects. Built from 1,152 manually curated images across nine domains, the benchmark supports visual question answering and captioning, and is expanded into four languages and five Bangla dialects, yielding ~32.3K artifacts. Our experiments show that evaluating only standard Bangla overestimates true model capability: performance drops under dialectal variation, especially for caption generation, while historically linked languages such as Hindi and Urdu retain some cultural meaning but remain weaker for structured reasoning. Across domains, the main bottleneck is missing cultural knowledge rather than visual grounding alone, with knowledge-intensive categories. These findings position BanglaVerse as a more realistic test bed for measuring culturally grounded multimodal understanding under linguistic variation. 

---
# Evaluating Reasoning-Based Scaffolds for Human-AI Co-Annotation: The ReasonAlign Annotation Protocol 

**Authors**: Smitha Muthya Sudheendra, Jaideep Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2603.21094)  

**Abstract**: Human annotation is central to NLP evaluation, yet subjective tasks often exhibit substantial variability across annotators. While large language models (LLMs) can provide structured reasoning to support annotation, their influence on human annotation behavior remains unclear.
We introduce ReasonAlign, a reasoning-based annotation scaffold that exposes LLM-generated explanations while withholding predicted labels. We frame this as a controlled study of how reasoning affects human annotation behavior, rather than a full evaluation of annotation accuracy. Using a two-pass protocol inspired by Delphi-style revision, annotators first label instances independently and then revise their decisions after viewing model-generated reasoning.
We evaluate the approach on sentiment classification and opinion detection tasks, analyzing changes in inter-annotator agreement and revision behavior. To quantify these effects, we introduce the Annotator Effort Proxy (AEP), a metric capturing the proportion of labels revised after exposure to reasoning. Our results show that exposure to reasoning is associated with increased agreement alongside minimal revision, suggesting that reasoning primarily helps resolve ambiguous cases without inducing widespread changes.
These findings provide insight into how reasoning explanations shape annotation consistency and highlight reasoning-based scaffolds as a practical mechanism for supporting human-AI annotation workflows. 

---
# ViCLSR: A Supervised Contrastive Learning Framework with Natural Language Inference for Natural Language Understanding Tasks 

**Authors**: Tin Van Huynh, Kiet Van Nguyen, Ngan Luu-Thuy Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.21084)  

**Abstract**: High-quality text representations are crucial for natural language understanding (NLU), but low-resource languages like Vietnamese face challenges due to limited annotated data. While pre-trained models like PhoBERT and CafeBERT perform well, their effectiveness is constrained by data scarcity. Contrastive learning (CL) has recently emerged as a promising approach for improving sentence representations, enabling models to effectively distinguish between semantically similar and dissimilar sentences. We propose ViCLSR (Vietnamese Contrastive Learning for Sentence Representations), a novel supervised contrastive learning framework specifically designed to optimize sentence embeddings for Vietnamese, leveraging existing natural language inference (NLI) datasets. Additionally, we propose a process to adapt existing Vietnamese datasets for supervised learning, ensuring compatibility with CL methods. Our experiments demonstrate that ViCLSR significantly outperforms the powerful monolingual pre-trained model PhoBERT on five benchmark NLU datasets such as ViNLI (+6.97% F1), ViWikiFC (+4.97% F1), ViFactCheck (+9.02% F1), UIT-ViCTSD (+5.36% F1), and ViMMRC2.0 (+4.33% Accuracy). ViCLSR shows that supervised contrastive learning can effectively address resource limitations in Vietnamese NLU tasks and improve sentence representation learning for low-resource languages. Furthermore, we conduct an in-depth analysis of the experimental results to uncover the factors contributing to the superior performance of contrastive learning models. ViCLSR is released for research purposes in advancing natural language processing tasks. 

---
# Assessing the Ability of Neural TTS Systems to Model Consonant-Induced F0 Perturbation 

**Authors**: Tianle Yang, Chengzhe Sun, Phil Rose, Cassandra L. Jacobs, Siwei Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21078)  

**Abstract**: This study proposes a segmental-level prosodic probing framework to evaluate neural TTS models' ability to reproduce consonant-induced f0 perturbation, a fine-grained segmental-prosodic effect that reflects local articulatory mechanisms. We compare synthetic and natural speech realizations for thousands of words, stratified by lexical frequency, using Tacotron 2 and FastSpeech 2 trained on the same speech corpus (LJ Speech). These controlled analyses are then complemented by a large-scale evaluation spanning multiple advanced TTS systems. Results show accurate reproduction for high-frequency words but poor generalization to low-frequency items, suggesting that the examined TTS architectures rely more on lexical-level memorization than on abstract segmental-prosodic encoding. This finding highlights a limitation in such TTS systems' ability to generalize prosodic detail beyond seen data. The proposed probe offers a linguistically informed diagnostic framework that may inform future TTS evaluation methods, and has implications for interpretability and authenticity assessment in synthetic speech. 

---
# Reading Between the Lines: How Electronic Nonverbal Cues shape Emotion Decoding 

**Authors**: Taara Kumar, Kokil Jaidka  

**Link**: [PDF](https://arxiv.org/pdf/2603.21038)  

**Abstract**: As text-based computer-mediated communication (CMC) increasingly structures everyday interaction, a central question re-emerges with new urgency: How do users reconstruct nonverbal expression in environments where embodied cues are absent? This paper provides a systematic, theory-driven account of electronic nonverbal cues (eNVCs) - textual analogues of kinesics, vocalics, and paralinguistics - in public microblog communication. Across three complementary studies, we advance conceptual, empirical, and methodological contributions. Study 1 develops a unified taxonomy of eNVCs grounded in foundational nonverbal communication theory and introduces a scalable Python toolkit for their automated detection. Study 2, a within-subject survey experiment, offers controlled causal evidence that eNVCs substantially improve emotional decoding accuracy and lower perceived ambiguity, while also identifying boundary conditions, such as sarcasm, under which these benefits weaken or disappear. Study 3, through focus group discussions, reveals the interpretive strategies users employ when reasoning about digital prosody, including drawing meaning from the absence of expected cues and defaulting toward negative interpretations in ambiguous contexts. Together, these studies establish eNVCs as a coherent and measurable class of digital behaviors, refine theoretical accounts of cue richness and interpretive effort, and provide practical tools for affective computing, user modeling, and emotion-aware interface design. The eNVC detection toolkit is available as a Python and R package at this https URL. 

---
# Left Behind: Cross-Lingual Transfer as a Bridge for Low-Resource Languages in Large Language Models 

**Authors**: Abdul-Salem Beibitkhan  

**Link**: [PDF](https://arxiv.org/pdf/2603.21036)  

**Abstract**: We investigate how large language models perform on low-resource languages by benchmarking eight LLMs across five experimental conditions in English, Kazakh, and Mongolian. Using 50 hand-crafted questions spanning factual, reasoning, technical, and culturally grounded categories, we evaluate 2,000 responses on accuracy, fluency, and completeness. We find a consistent performance gap of 13.8-16.7 percentage points between English and low-resource language conditions, with models maintaining surface-level fluency while producing significantly less accurate content. Cross-lingual transfer-prompting models to reason in English before translating back-yields selective gains for bilingual architectures (+2.2pp to +4.3pp) but provides no benefit to English-dominant models. Our results demonstrate that current LLMs systematically underserve low-resource language communities, and that effective mitigation strategies are architecture-dependent rather than universal. 

---
# Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO 

**Authors**: Jinquan Zheng, Jia Yuan, Jiacheng Yao, Chenyang Gu, Pujun Zheng, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2603.21016)  

**Abstract**: Large language models (LLMs) used for multiple-choice and pairwise evaluation tasks often exhibit selection bias due to non-semantic factors like option positions and label symbols. Existing inference-time debiasing is costly and may harm reasoning, while pointwise training ignores that the same question should yield consistent answers across permutations. To address this issue, we propose Permutation-Aware Group Relative Policy Optimization (PA-GRPO), which mitigates selection bias by enforcing permutation-consistent semantic reasoning. PA-GRPO constructs a permutation group for each instance by generating multiple candidate permutations, and optimizes the model using two complementary mechanisms: (1) cross-permutation advantage, which computes advantages relative to the mean reward over all permutations of the same instance, and (2) consistency-aware reward, which encourages the model to produce consistent decisions across different permutations. Experimental results demonstrate that PA-GRPO outperforms strong baselines across seven benchmarks, substantially reducing selection bias while maintaining high overall performance. The code will be made available on Github (this https URL). 

---
# DiscoUQ: Structured Disagreement Analysis for Uncertainty Quantification in LLM Agent Ensembles 

**Authors**: Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20975)  

**Abstract**: Multi-agent LLM systems, where multiple prompted instances of a language model independently answer questions, are increasingly used for complex reasoning tasks. However, existing methods for quantifying the uncertainty of their collective outputs rely on shallow voting statistics that discard the rich semantic information in agents' reasoning. We introduce DiscoUQ, a framework that extracts and leverages the structure of inter-agent disagreement -- both linguistic properties (evidence overlap, argument strength, divergence depth) and embedding geometry (cluster distances, dispersion, cohesion) -- to produce well-calibrated confidence estimates. We propose three methods of increasing complexity: DiscoUQ-LLM (logistic regression on LLM-extracted structure features), DiscoUQ-Embed (logistic regression on embedding geometry), and DiscoUQ-Learn (a neural network combining all features). Evaluated on four diverse benchmarks (StrategyQA, MMLU, TruthfulQA, ARC-Challenge) with a 5-agent system using Qwen3.5-27B, DiscoUQ-LLM achieves an average AUROC of 0.802, outperforming the best baseline (LLM Aggregator, 0.791) while being substantially better calibrated (ECE 0.036 vs. 0.098). The learned features generalize across benchmarks with near-zero performance degradation and provide the largest improvements where they are most needed: in the ambiguous "weak disagreement" tier where simple vote counting fails. 

---
# Alignment Whack-a-Mole : Finetuning Activates Verbatim Recall of Copyrighted Books in Large Language Models 

**Authors**: Xinyue Liu, Niloofar Mireshghallah, Jane C. Ginsburg, Tuhin Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2603.20957)  

**Abstract**: Frontier LLM companies have repeatedly assured courts and regulators that their models do not store copies of training data. They further rely on safety alignment strategies via RLHF, system prompts, and output filters to block verbatim regurgitation of copyrighted works, and have cited the efficacy of these measures in their legal defenses against copyright infringement claims. We show that finetuning bypasses these protections: by training models to expand plot summaries into full text, a task naturally suited for commercial writing assistants, we cause GPT-4o, Gemini-2.5-Pro, and DeepSeek-V3.1 to reproduce up to 85-90% of held-out copyrighted books, with single verbatim spans exceeding 460 words, using only semantic descriptions as prompts and no actual book text. This extraction generalizes across authors: finetuning exclusively on Haruki Murakami's novels unlocks verbatim recall of copyrighted books from over 30 unrelated authors. The effect is not specific to any training author or corpus: random author pairs and public-domain finetuning data produce comparable extraction, while finetuning on synthetic text yields near-zero extraction, indicating that finetuning on individual authors' works reactivates latent memorization from pretraining. Three models from different providers memorize the same books in the same regions ($r \ge 0.90$), pointing to an industry-wide vulnerability. Our findings offer compelling evidence that model weights store copies of copyrighted works and that the security failures that manifest after finetuning on individual authors' works undermine a key premise of recent fair use rulings, where courts have conditioned favorable outcomes on the adequacy of measures preventing reproduction of protected expression. 

---
# User Preference Modeling for Conversational LLM Agents: Weak Rewards from Retrieval-Augmented Interaction 

**Authors**: Yuren Hao, Shuhaib Mehri, ChengXiang Zhai, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2603.20939)  

**Abstract**: Large language models are increasingly used as personal assistants, yet most lack a persistent user model, forcing users to repeatedly restate preferences across sessions. We propose Vector-Adapted Retrieval Scoring (VARS), a pipeline-agnostic, frozen-backbone framework that represents each user with long-term and short-term vectors in a shared preference space and uses these vectors to bias retrieval scoring over structured preference memory. The vectors are updated online from weak scalar rewards from users' feedback, enabling personalization without per-user fine-tuning. We evaluate on \textsc{MultiSessionCollab}, an online multi-session collaboration benchmark with rich user preference profiles, across math and code tasks. Under frozen backbones, the main benefit of user-aware retrieval is improved interaction efficiency rather than large gains in raw task accuracy: our full VARS agent achieves the strongest overall performance, matches a strong Reflection baseline in task success, and reduces timeout rate and user effort. The learned long-term vectors also align with cross-user preference overlap, while short-term vectors capture session-specific adaptation, supporting the interpretability of the dual-vector design. Code, model, and data are available at this https URL. 

---
# The Hidden Puppet Master: A Theoretical and Real-World Account of Emotional Manipulation in LLMs 

**Authors**: Jocelyn Shen, Amina Luvsanchultem, Jessica Kim, Kynnedy Smith, Valdemar Danry, Kantwon Rogers, Sharifa Alghowinem, Hae Won Park, Maarten Sap, Cynthia Breazeal  

**Link**: [PDF](https://arxiv.org/pdf/2603.20907)  

**Abstract**: As users increasingly turn to LLMs for practical and personal advice, they become vulnerable to being subtly steered toward hidden incentives misaligned with their own interests. Prior works have benchmarked persuasion and manipulation detection, but these efforts rely on simulated or debate-style settings, remain uncorrelated with real human belief shifts, and overlook a critical dimension: the morality of hidden incentives driving the manipulation. We introduce PUPPET, a theoretical taxonomy of personalized emotional manipulation in LLM-human dialogues that centers around incentive morality, and conduct a human study with N=1,035 participants across realistic everyday queries, varying personalization and incentive direction (harmful versus prosocial). We find that harmful hidden incentives produce significantly larger belief shifts than prosocial ones. Finally, we benchmark LLMs on the task of belief prediction, finding that models exhibit moderate predictive ability of belief change based on conversational contexts (r=0.3 - 0.5), but they also systematically underestimate the magnitude of belief shift. Together, this work establishes a theoretically grounded and behaviorally validated foundation for studying, and ultimately combatting, incentive-driven manipulation in LLMs during everyday, practical user queries. 

---
# Mitigating Shortcut Reasoning in Language Models: A Gradient-Aware Training Approach 

**Authors**: Hongyu Cao, Kunpeng Liu, Dongjie Wang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20899)  

**Abstract**: Large language models exhibit strong reasoning capabilities, yet often rely on shortcuts such as surface pattern matching and answer memorization rather than genuine logical inference. We propose Shortcut-Aware Reasoning Training (SART), a gradient-aware framework that detects and mitigates shortcut-promoting samples via ShortcutScore and gradient surgery. Our method identifies shortcut signals through gradient misalignment with validation objectives and answer-token concentration, and modifies training dynamics accordingly. Experiments on controlled reasoning benchmarks show that SART achieves +16.5% accuracy and +40.2% robustness over the strongest baseline, significantly improving generalization under distribution shifts. Code is available at: this https URL. 

---
# LLM Router: Prefill is All You Need 

**Authors**: Tanay Varshney, Annie Surla, Michelle Xu, Gomathy Venkata Krishnan, Maximilian Jeblick, David Austin, Neal Vaidya, Davide Onofrio  

**Link**: [PDF](https://arxiv.org/pdf/2603.20895)  

**Abstract**: LLMs often share comparable benchmark accuracies, but their complementary performance across task subsets suggests that an Oracle router--a theoretical selector with perfect foresight--can significantly surpass standalone model accuracy by navigating model-specific strengths. While current routers rely on fragile semantic signals, we propose using internal prefill activations via Encoder-Target Decoupling--a functional separation between the model providing the predictive signal (the Encoder) and the model whose performance is being estimated (the Target). This allows optimized heterogeneous pairing between unique encoders and target models. We utilize Fisher Separability (J) and Effective Dimensionality (d_eff) as mathematical probes to isolate optimal layer-wise signals, providing the predictive foundation for our SharedTrunkNet architecture. SharedTrunkNet captures up to 45.58% of the accuracy gap between the strongest standalone model and the Oracle while achieving 74.31% cost savings relative to the highest-cost model. 

---
# NoveltyAgent: Autonomous Novelty Reporting Agent with Point-wise Novelty Analysis and Self-Validation 

**Authors**: Jiajun Hou, Hexuan Deng, Wenxiang Jiao, Xuebo Liu, Xiaopeng Ke, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20884)  

**Abstract**: The exponential growth of academic publications has led to a surge in papers of varying quality, increasing the cost of paper screening. Current approaches either use novelty assessment within general AI Reviewers or repurpose DeepResearch, which lacks domain-specific mechanisms and thus delivers lower-quality results. To bridge this gap, we introduce NoveltyAgent, a multi-agent system designed to generate comprehensive and faithful novelty reports, enabling thorough evaluation of a paper's originality. It decomposes manuscripts into discrete novelty points for fine-grained retrieval and comparison, and builds a comprehensive related-paper database while cross-referencing claims to ensure faithfulness. Furthermore, to address the challenge of evaluating such open-ended generation tasks, we propose a checklist-based evaluation framework, providing an unbiased paradigm for building reliable evaluations. Extensive experiments show that NoveltyAgent achieves state-of-the-art performance, outperforming GPT-5 DeepResearch by 10.15%. We hope this system will provide reliable, high-quality novelty analysis and help researchers quickly identify novel papers. Code and demo are available at this https URL. 

---
# SozKZ: Training Efficient Small Language Models for Kazakh from Scratch 

**Authors**: Saken Tukenov  

**Link**: [PDF](https://arxiv.org/pdf/2603.20854)  

**Abstract**: Kazakh, a Turkic language spoken by over 22 million people, remains underserved by existing multilingual language models, which allocate minimal capacity to low-resource languages and employ tokenizers ill-suited to agglutinative morphology. We present SozKZ, a family of Llama-architecture language models (50M-600M parameters) trained entirely from scratch on 9 billion tokens of Kazakh text with a dedicated 50K BPE tokenizer. We evaluate all models on three Kazakh benchmarks -- multiple-choice cultural QA, reading comprehension (Belebele), and topic classification (SIB-200) -- alongside five multilingual baselines ranging from 500M to 3B parameters. Our 600M model achieves 30.3% accuracy on Kazakh cultural QA, approaching the 32.0% of Llama-3.2-1B (2x larger), and 25.5% on SIB-200 topic classification, surpassing all evaluated multilingual models up to 2B parameters. We observe consistent scaling from 50M to 600M, with MC QA accuracy rising from 22.8% to 30.3%, suggesting that further scaling remains beneficial. These results demonstrate that small, dedicated models trained from scratch with a language-appropriate tokenizer offer a viable path for low-resource language technology, achieving competitive performance at a fraction of the computational cost. All models and the tokenizer are released under open licenses. 

---
# Can ChatGPT Really Understand Modern Chinese Poetry? 

**Authors**: Shanshan Wang, Derek F. Wong, Jingming Yao, Lidia S. Chao  

**Link**: [PDF](https://arxiv.org/pdf/2603.20851)  

**Abstract**: ChatGPT has demonstrated remarkable capabilities on both poetry generation and translation, yet its ability to truly understand poetry remains unexplored. Previous poetry-related work merely analyzed experimental outcomes without addressing fundamental issues of comprehension. This paper introduces a comprehensive framework for evaluating ChatGPT's understanding of modern poetry. We collaborated with professional poets to evaluate ChatGPT's interpretation of modern Chinese poems by different poets along multiple dimensions. Evaluation results show that ChatGPT's interpretations align with the original poets' intents in over 73% of the cases. However, its understanding in certain dimensions, particularly in capturing poeticity, proved to be less satisfactory. These findings highlight the effectiveness and necessity of our proposed framework. This study not only evaluates ChatGPT's ability to understand modern poetry but also establishes a solid foundation for future research on LLMs and their application to poetry-related tasks. 

---
# HiCI: Hierarchical Construction-Integration for Long-Context Attention 

**Authors**: Xiangyu Zeng, Qi Xu, Yunke Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20843)  

**Abstract**: Long-context language modeling is commonly framed as a scalability challenge of token-level attention, yet local-to-global information structuring remains largely implicit in existing approaches. Drawing on cognitive theories of discourse comprehension, we propose HiCI (Hierarchical Construction--Integration), a hierarchical attention module that constructs segment-level representations, integrates them into a shared global context, and broadcasts both to condition segment-level attention. We validate HiCI through parameter-efficient adaptation of LLaMA-2 with only <5.5% additional parameters, extending context from 4K to 100K tokens (7B) and 64K tokens (13B). Across language modeling, retrieval, and instruction-following benchmarks, HiCI yields consistent improvements over strong baselines, including matching proprietary models on topic retrieval and surpassing GPT-3.5-Turbo-16K on code comprehension. These results demonstrate the effectiveness of explicit hierarchical structuring as an inductive bias for long-context modeling. 

---
# BenchBench: Benchmarking Automated Benchmark Generation 

**Authors**: Yandan Zheng, Haoran Luo, Zhenghong Lin, Wenjin Liu, Luu Anh Tuan  

**Link**: [PDF](https://arxiv.org/pdf/2603.20807)  

**Abstract**: Benchmarks are the de facto standard for tracking progress in large language models (LLMs), yet static test sets can rapidly saturate, become vulnerable to contamination, and are costly to refresh. Scalable evaluation of open-ended items often relies on LLM judges, introducing additional sources of bias and prompt sensitivity. We argue that evaluation must extend beyond how well models answer benchmarks to how well models design them. We introduce BenchBench, a three-stage pipeline and dataset for benchmarking automated benchmark generation: (i) extract structured domain cards from seed benchmarks, (ii) prompt multiple designer LLMs to generate quota-controlled suites, and (iii) validate items with a multi-model answerer panel using exact/numeric/symbolic verifiers when possible and rubric-guided judging otherwise, yielding designer--answerer matrices with item-level quality flags and psychometric diagnostics. Across nine variants spanning computer science, mathematics, medicine, and theory-of-mind reasoning (including multilingual and multimodal settings), we generate 16.7K items, retain ~15K core items post-filtering, and produce ~152K graded model--item responses. BenchBench shows that benchmark-design ability is only moderately correlated with answer-time strength (Spearman rho ~0.37), invalidity is negatively associated with discrimination (Pearson r~0.62), and the resulting designer--answerer matrices enable scalable audits of format/modality/language fidelity and suite-dependent self/family interactions. The project is available at: this https URL. 

---
# RLVR Training of LLMs Does Not Improve Thinking Ability for General QA: Evaluation Method and a Simple Solution 

**Authors**: Kaiyuan Li, Jing-Cheng Pang, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20799)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) stimulates the thinking processes of large language models (LLMs), substantially enhancing their reasoning abilities on verifiable tasks. It is often assumed that similar gains should transfer to general question answering (GQA), but this assumption has not been thoroughly validated. To assess whether RLVR automatically improves LLM performance on GQA, we propose a Cross-Generation evaluation framework that measures the quality of intermediate reasoning by feeding the generated thinking context into LLMs of varying capabilities. Our evaluation leads to a discouraging finding: the efficacy of the thinking process on GQA tasks is markedly lower than on verifiable tasks, suggesting that explicit training on GQA remains necessary in addition to training on verifiable tasks. We further observe that direct RL training on GQA is less effective than RLVR. Our hypothesis is that, whereas verifiable tasks demand robust logical chains to obtain high rewards, GQA tasks often admit shortcuts to high rewards without cultivating high-quality thinking. To avoid possible shortcuts, we introduce a simple method, Separated Thinking And Response Training (START), which first trains only the thinking process, using rewards defined on the final answer. We show that START improves both the quality of thinking and the final answer across several GQA benchmarks and RL algorithms. 

---
# The Anatomy of an Edit: Mechanism-Guided Activation Steering for Knowledge Editing 

**Authors**: Yuan Cao, Mingyang Wang, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2603.20795)  

**Abstract**: Large language models (LLMs) are increasingly used as knowledge bases, but keeping them up to date requires targeted knowledge editing (KE). However, it remains unclear how edits are implemented inside the model once applied. In this work, we take a mechanistic view of KE using neuron-level knowledge attribution (NLKA). Unlike prior work that focuses on pre-edit causal tracing and localization, we use post-edit attribution -- contrasting successful and failed edits -- to isolate the computations that shift when an edit succeeds. Across representative KE methods, we find a consistent pattern: mid-to-late attention predominantly promotes the new target, while attention and FFN modules cooperate to suppress the original fact. Motivated by these findings, we propose MEGA, a MEchanism-Guided Activation steering method that performs attention-residual interventions in attribution-aligned regions without modifying model weights. On CounterFact and Popular, MEGA achieves strong editing performance across KE metrics on GPT2-XL and LLaMA2-7B. Overall, our results elevate post-edit attribution from analysis to engineering signal: by pinpointing where and how edits take hold, it powers MEGA to deliver reliable, architecture-agnostic knowledge edits. 

---
# Code-MIE: A Code-style Model for Multimodal Information Extraction with Scene Graph and Entity Attribute Knowledge Enhancement 

**Authors**: Jiang Liu, Ge Qiu, Hao Fei, Dongdong Xie, Jinbo Li, Fei Li, Chong Teng, Donghong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2603.20781)  

**Abstract**: With the rapid development of large language models (LLMs), more and more researchers have paid attention to information extraction based on LLMs. However, there are still some spaces to improve in the existing related methods. First, existing multimodal information extraction (MIE) methods usually employ natural language templates as the input and output of LLMs, which mismatch with the characteristics of information tasks that mostly include structured information such as entities and relations. Second, although a few methods have adopted structured and more IE-friendly code-style templates, they just explored their methods on text-only IE rather than multimodal IE. Moreover, their methods are more complex in design, requiring separate templates to be designed for each task. In this paper, we propose a Code-style Multimodal Information Extraction framework (Code-MIE) which formalizes MIE as unified code understanding and generation. Code-MIE has the following novel designs: (1) Entity attributes such as gender, affiliation are extracted from the text to guide the model to understand the context and role of entities. (2) Images are converted into scene graphs and visual features to incorporate rich visual information into the model. (3) The input template is constructed as a Python function, where entity attributes, scene graphs and raw text compose of the function parameters. In contrast, the output template is formalized as Python dictionaries containing all extraction results such as entities, relations, etc. To evaluate Code-MIE, we conducted extensive experiments on the M$^3$D, Twitter-15, Twitter-17, and MNRE datasets. The results show that our method achieves state-of-the-art performance compared to six competing baseline models, with 61.03\% and 60.49\% on the English and Chinese datasets of M$^3$D, and 76.04\%, 88.07\%, and 73.94\% on the other three datasets. 

---
# MzansiText and MzansiLM: An Open Corpus and Decoder-Only Language Model for South African Languages 

**Authors**: Anri Lombard, Simbarashe Mawere, Temi Aina, Ethan Wolff, Sbonelo Gumede, Elan Novick, Francois Meyer, Jan Buys  

**Link**: [PDF](https://arxiv.org/pdf/2603.20732)  

**Abstract**: Decoder-only language models can be adapted to diverse tasks through instruction finetuning, but the extent to which this generalizes at small scale for low-resource languages remains unclear. We focus on the languages of South Africa, where we are not aware of a publicly available decoder-only model that explicitly targets all eleven official written languages, nine of which are low-resource. We introduce MzansiText, a curated multilingual pretraining corpus with a reproducible filtering pipeline, and MzansiLM, a 125M-parameter language model trained from scratch. We evaluate MzansiLM on natural language understanding and generation using three adaptation regimes: monolingual task-specific finetuning, multilingual task-specific finetuning, and general multi-task instruction finetuning. Monolingual task-specific finetuning achieves strong performance on data-to-text generation, reaching 20.65 BLEU on isiXhosa and competing with encoder-decoder baselines over ten times larger. Multilingual task-specific finetuning benefits closely related languages on topic classification, achieving 78.5% macro-F1 on isiXhosa news classification. While MzansiLM adapts effectively to supervised NLU and NLG tasks, few-shot reasoning remains challenging at this model size, with performance near chance even for much larger decoder-only models. We release MzansiText and MzansiLM to provide a reproducible decoder-only baseline and clear guidance on adaptation strategies for South African languages at small scale. 

---
# Reasoning Topology Matters: Network-of-Thought for Complex Reasoning Tasks 

**Authors**: Fan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20730)  

**Abstract**: Existing prompting paradigms structure LLM reasoning in limited topologies: Chain-of-Thought (CoT) produces linear traces, while Tree-of-Thought (ToT) performs branching search. Yet complex reasoning often requires merging intermediate results, revisiting hypotheses, and integrating evidence from multiple sources. We propose Network-of-Thought (NoT), a framework that models reasoning as a directed graph with typed nodes and edges, guided by a heuristic-based controller policy. Across four benchmarks (GSM8K, Game of 24, HotpotQA, ProofWriter) and three models (GPT-4o-mini, Llama-3.3-70B-Instruct, Qwen2.5-72B-Instruct), we investigate when network topology outperforms chain or tree structures, whether LLM-generated heuristics can guide graph-based reasoning search, and the computation-accuracy tradeoff across topologies, evaluating each method on accuracy, topology simplicity, and token efficiency. Our results show that CoT remains effective for sequential tasks with GPT-4o-mini (89.5\% on GSM8K), while NoT surpasses ToT on multi-hop reasoning (91.0\% vs.\ 88.0\% on HotpotQA with LLM-as-Judge). With 72B open-source models, NoT achieves the highest accuracy on GSM8K (91.5\%), and Qwen2.5-72B achieves the best multi-hop QA result overall (91.7\% on HotpotQA). Self-generated controller heuristics outperform fixed and random strategies on logical reasoning, with uncertainty-only weighting achieving 57.0\% on ProofWriter. We also find that evaluation methodology significantly impacts method rankings: string-match underestimates all methods on open-ended QA, with the largest gap for NoT, a pattern consistent across all three models (14--18 percentage point gap on HotpotQA). 

---
# Can I guess where you are from? Modeling dialectal morphosyntactic similarities in Brazilian Portuguese 

**Authors**: Manoel Siqueira, Raquel Freitag  

**Link**: [PDF](https://arxiv.org/pdf/2603.20695)  

**Abstract**: This paper investigates morphosyntactic covariation in Brazilian Portuguese (BP) to assess whether dialectal origin can be inferred from the combined behavior of linguistic variables. Focusing on four grammatical phenomena related to pronouns, correlation and clustering methods are applied to model covariation and dialectal distribution. The results indicate that correlation captures only limited pairwise associations, whereas clustering reveals speaker groupings that reflect regional dialectal patterns. Despite the methodological constraints imposed by differences in sample size requirements between sociolinguistics and computational approaches, the study highlights the importance of interdisciplinary research. Developing fair and inclusive language technologies that respect dialectal diversity outweighs the challenges of integrating these fields. 

---
# PAVE: Premise-Aware Validation and Editing for Retrieval-Augmented LLMs 

**Authors**: Tianyi Huang, Caden Yang, Emily Yin, Eric Wang, Michael Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.20673)  

**Abstract**: Retrieval-augmented language models can retrieve relevant evidence yet still commit to answers before explicitly checking whether the retrieved context supports the conclusion. We present PAVE (Premise-Grounded Answer Validation and Editing), an inference-time validation layer for evidence-grounded question answering. PAVE decomposes retrieved context into question-conditioned atomic facts, drafts an answer, scores how well that draft is supported by the extracted premises, and revises low-support outputs before finalization. The resulting trace makes answer commitment auditable at the level of explicit premises, support scores, and revision decisions. In controlled ablations with a fixed retriever and backbone, PAVE outperforms simpler post-retrieval baselines in two evidence-grounded QA settings, with the largest gain reaching 32.7 accuracy points on a span-grounded benchmark. We view these findings as proof-of-concept evidence that explicit premise extraction plus support-gated revision can strengthen evidence-grounded consistency in retrieval-augmented LLM systems. 

---
# Weber's Law in Transformer Magnitude Representations: Efficient Coding, Representational Geometry, and Psychophysical Laws in Language Models 

**Authors**: Jon-Paul Cacioli  

**Link**: [PDF](https://arxiv.org/pdf/2603.20642)  

**Abstract**: How do transformer language models represent magnitude? Recent work disagrees: some find logarithmic spacing, others linear encoding, others per-digit circular representations. We apply the formal tools of psychophysics to resolve this. Using four converging paradigms (representational similarity analysis, behavioural discrimination, precision gradients, causal intervention) across three magnitude domains in three 7-9B instruction-tuned models spanning three architecture families (Llama, Mistral, Qwen), we report three findings. First, representational geometry is consistently log-compressive: RSA correlations with a Weber-law dissimilarity matrix ranged from .68 to .96 across all 96 model-domain-layer cells, with linear geometry never preferred. Second, this geometry is dissociated from behaviour: one model produces a human-range Weber fraction (WF = 0.20) while the other does not, and both models perform at chance on temporal and spatial discrimination despite possessing logarithmic geometry. Third, causal intervention reveals a layer dissociation: early layers are functionally implicated in magnitude processing (4.1x specificity) while later layers where geometry is strongest are not causally engaged (1.2x). Corpus analysis confirms the efficient coding precondition (alpha = 0.77). These results suggest that training data statistics alone are sufficient to produce log-compressive magnitude geometry, but geometry alone does not guarantee behavioural competence. 

---
# Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention 

**Authors**: Manh Nguyen, Anh Nguyen, Dung Nguyen, Svetha Venkatesh, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2603.20640)  

**Abstract**: Multi-Agent Debate has emerged as a promising framework for improving the reasoning quality of large language models through iterative inter-agent communication. However, broadcasting all agent messages at every round introduces noise and redundancy that can degrade debate quality and waste computational resources. Current approaches rely on uncertainty estimation to filter low-confidence responses before broadcasting, but this approach is unreliable due to miscalibrated confidence scores and sensitivity to threshold selection. To address this, we propose Diversity-Aware Retention (DAR), a lightweight debate framework that, at each debate round, selects the subset of agent responses that maximally disagree with each other and with the majority vote before broadcasting. Through an explicit index-based retention mechanism, DAR preserves the original messages without modification, ensuring that retained disagreements remain authentic. Experiments on diverse reasoning and question answering benchmarks demonstrate that our selective message propagation consistently improves debate performance, particularly as the number of agents scales, where noise accumulation is most severe. Our results highlight that what agents hear is as important as what agents say in multi-agent reasoning systems. 

---
# A Modular LLM Framework for Explainable Price Outlier Detection 

**Authors**: Shadi Sartipi, John Wu, Sina Ghotbi, Nikhita Vedula, Shervin Malmasi  

**Link**: [PDF](https://arxiv.org/pdf/2603.20636)  

**Abstract**: Detecting product price outliers is important for retail and e-commerce stores as erroneous or unexpectedly high prices adversely affect competitiveness, revenue, and consumer trust. Classical techniques offer simple thresholds while ignoring the rich semantic relationships among product attributes. We propose an agentic Large Language Model (LLM) framework that treats outlier price flagging as a reasoning task grounded in related product detection and comparison. The system processes the prices of target products in three stages: (i) relevance classification selects price-relevant similar products using product descriptions and attributes; (ii) relative utility assessment evaluates the target product against each similar product along price influencing dimensions (e.g., brand, size, features); (iii) reasoning-based decision aggregates these justifications into an explainable price outlier judgment. The framework attains over 75% agreement with human auditors on a test dataset, and outperforms zero-shot and retrieval based LLM techniques. Ablation studies show the sensitivity of the method to key hyper-parameters and testify on its flexibility to be applied to cases with different accuracy requirement and auditor agreements. 

---
# JUBAKU: An Adversarial Benchmark for Exposing Culturally Grounded Stereotypes in Japanese LLMs 

**Authors**: Taihei Shiotani, Masahiro Kaneko, Ayana Niwa, Yuki Maruyama, Daisuke Oba, Masanari Ohi, Naoaki Okazaki  

**Link**: [PDF](https://arxiv.org/pdf/2603.20581)  

**Abstract**: Social biases reflected in language are inherently shaped by cultural norms, which vary significantly across regions and lead to diverse manifestations of stereotypes. Existing evaluations of social bias in large language models (LLMs) for non-English contexts, however, often rely on translations of English benchmarks. Such benchmarks fail to reflect local cultural norms, including those found in Japanese. For instance, Western benchmarks may overlook Japan-specific stereotypes related to hierarchical relationships, regional dialects, or traditional gender roles. To address this limitation, we introduce Japanese cUlture adversarial BiAs benchmarK Under handcrafted creation (JUBAKU), a benchmark tailored to Japanese cultural contexts. JUBAKU uses adversarial construction to expose latent biases across ten distinct cultural categories. Unlike existing benchmarks, JUBAKU features dialogue scenarios hand-crafted by native Japanese annotators, specifically designed to trigger and reveal latent social biases in Japanese LLMs. We evaluated nine Japanese LLMs on JUBAKU and three others adapted from English benchmarks. All models clearly exhibited biases on JUBAKU, performing below the random baseline of 50% with an average accuracy of 23% (ranging from 13% to 33%), despite higher accuracy on the other benchmarks. Human annotators achieved 91% accuracy in identifying unbiased responses, confirming JUBAKU's reliability and its adversarial nature to LLMs. 

---
# Permutation-Consensus Listwise Judging for Robust Factuality Evaluation 

**Authors**: Tianyi Huang, Nathan Huang, Justin Tang, Wenqian Chen, Elsa Fan  

**Link**: [PDF](https://arxiv.org/pdf/2603.20562)  

**Abstract**: Large language models (LLMs) are now widely used as judges, yet their decisions can change under presentation choices that should be irrelevant. We study one such source of instability: candidate-order sensitivity in listwise factuality evaluation, where several answers can look similarly polished while differing sharply in hallucination risk. We introduce PCFJudge, an inference-time method that reruns the same factuality-first listwise prompt over multiple orderings of the same candidate set and aggregates the resulting scores, ranks, and uncertainty signals into a single consensus decision. On RewardBench 2 Factuality, PCFJudge improves over direct judging by up to 7 absolute points. Development ablations show that the dominant gain comes from permutation consensus itself rather than from heavier arbitration layers. These results suggest that a meaningful share of factuality-judging error arises from order instability, and that averaging over this nuisance variation is a simple and effective way to make LLM evaluation more reliable. 

---
# Evaluating Large Language Models on Historical Health Crisis Knowledge in Resource-Limited Settings: A Hybrid Multi-Metric Study 

**Authors**: Mohammed Rakibul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2603.20514)  

**Abstract**: Large Language Models (LLMs) offer significant potential for delivering health information. However, their reliability in low-resource contexts remains uncertain. This study evaluates GPT-4, Gemini Pro, Llama~3, and Mistral-7B on health crisis-related enquiries concerning COVID-19, dengue, the Nipah virus, and Chikungunya in the low-resource context of Bangladesh. We constructed a question--answer dataset from authoritative sources and assessed model outputs through semantic similarity, expert-model cross-evaluation, and Natural Language Inference (NLI). Findings highlight both the strengths and limitations of LLMs in representing epidemiological history and health crisis knowledge, underscoring their promise and risks for informing policy in resource-constrained environments. 

---
# PARHAF, a human-authored corpus of clinical reports for fictitious patients in French 

**Authors**: Xavier Tannier, Salam Abbara, Rémi Flicoteaux, Youness Khalil, Aurélie Névéol, Pierre Zweigenbaum, Emmanuel Bacry  

**Link**: [PDF](https://arxiv.org/pdf/2603.20494)  

**Abstract**: The development of clinical natural language processing (NLP) systems is severely hampered by the sensitive nature of medical records, which restricts data sharing under stringent privacy regulations, particularly in France and the broader European Union. To address this gap, we introduce PARHAF, a large open-source corpus of clinical documents in French. PARHAF comprises expert-authored clinical reports describing realistic yet entirely fictitious patient cases, making it anonymous and freely shareable by design. The corpus was developed using a structured protocol that combined clinician expertise with epidemiological guidance from the French National Health Data System (SNDS), ensuring broad clinical coverage. A total of 104 medical residents across 18 specialties authored and peer-reviewed the reports following predefined clinical scenarios and document templates.
The corpus contains 7394 clinical reports covering 5009 patient cases across a wide range of medical and surgical specialties. It includes a general-purpose component designed to approximate real-world hospitalization distributions, and four specialized subsets that support information-extraction use cases in oncology, infectious diseases, and diagnostic coding. Documents are released under a CC-BY open license, with a portion temporarily embargoed to enable future benchmarking under controlled conditions.
PARHAF provides a valuable resource for training and evaluating French clinical language models in a fully privacy-preserving setting, and establishes a replicable methodology for building shareable synthetic clinical corpora in other languages and health systems. 

---
# Diffutron: A Masked Diffusion Language Model for Turkish Language 

**Authors**: Şuayp Talha Kocabay, Talha Rüzgar Akkuş  

**Link**: [PDF](https://arxiv.org/pdf/2603.20466)  

**Abstract**: Masked Diffusion Language Models (MDLMs) have emerged as a compelling non-autoregressive alternative to standard large language models; however, their application to morphologically rich languages remains limited. In this paper, we introduce $\textit{Diffutron}$, a masked diffusion language model specifically designed for Turkish. Our approach leverages a resource-efficient training pipeline, starting with LoRA-based continual pre-training of a multilingual encoder on a large-scale corpus. To enable generative capabilities, we employ a progressive instruction-tuning strategy, sequentially adapting the model on general and task-specific instruction sets. Experimental results across comprehensive benchmarks demonstrate that, despite its compact size, our model achieves competitive performance compared to existing multi-billion-parameter baselines. These findings validate the effectiveness of masked diffusion modeling combined with multi-stage tuning for non-autoregressive text generation in Turkish. 

---
# Policies Permitting LLM Use for Polishing Peer Reviews Are Currently Not Enforceable 

**Authors**: Rounak Saha, Gurusha Juneja, Dayita Chaudhuri, Naveeja Sajeevan, Nihar B Shah, Danish Pruthi  

**Link**: [PDF](https://arxiv.org/pdf/2603.20450)  

**Abstract**: A number of scientific conferences and journals have recently enacted policies that prohibit LLM usage by peer reviewers, except for polishing, paraphrasing, and grammar correction of otherwise human-written reviews. But, are these policies enforceable? To answer this question, we assemble a dataset of peer reviews simulating multiple levels of human-AI collaboration, and evaluate five state-of-the-art detectors, including two commercial systems. Our analysis shows that all detectors misclassify a non-trivial fraction of LLM-polished reviews as AI-generated, thereby risking false accusations of academic misconduct. We further investigate whether peer-review-specific signals, including access to the paper manuscript and the constrained domain of scientific writing, can be leveraged to improve detection. While incorporating such signals yields measurable gains in some settings, we identify limitations in each approach and find that none meets the accuracy standards required for identifying AI use in peer reviews. Importantly, our results suggest that recent public estimates of AI use in peer reviews through the use of AI-text detectors should be interpreted with caution, as current detectors misclassify mixed reviews (collaborative human-AI outputs) as fully AI generated, potentially overstating the extent of policy violations. 

---
# A Training-Free Regeneration Paradigm: Contrastive Reflection Memory Guided Self-Verification and Self-Improvement 

**Authors**: Yuran Li, Di Wu, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2603.20441)  

**Abstract**: Verification-guided self-improvement has recently emerged as a promising approach to improving the accuracy of large language model (LLM) outputs. However, existing approaches face a trade-off between inference efficiency and accuracy: iterative verification-rectification is computationally expensive and prone to being trapped in faulty reasoning, while best-of-N selection requires extensive sampling without addressing internal model flaws. We propose a training-free regeneration paradigm that leverages an offline-curated contrastive Reflection Memory (RM) to provide corrective guidance, while regenerating from scratch helps break out of faulty reasoning. At inference time, the method performs RM-guided self-verification followed by a single RM-guided regeneration, avoiding both iterative correction and multi-sample selection. We evaluated our method on nine benchmarks that span algorithmic, reasoning, symbolic, and domain-specific tasks in both small- and large-scale LLMs. Experiment results show that our method outperforms prior methods while maintaining low computational cost. 

---
# Coding Agents are Effective Long-Context Processors 

**Authors**: Weili Cao, Xunjian Yin, Bhuwan Dhingra, Shuyan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.20432)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress in scaling to access massive contexts. However, the access is via the latent and uninterpretable attention mechanisms, and LLMs fail to effective process long context, exhibiting significant performance degradation as context length increases. In this work, we study whether long-context processing can be externalized from latent attention into explicit, executable interactions, by allowing coding agents to organize text in file systems and manipulate it using its native tools. We evaluate off-the-shelf frontier coding agents as the general interface for tasks that require processing long contexts, including long-context reasoning, retrieval-augmented generation, and open-domain question answering with large-scale corpus contains up to three trillion tokens. Across multiple benchmarks, these agents outperform published state-of-the-art by 17.3% on average. We attribute this efficacy to two key factors: native tool proficiency, which enables agents to leverage executable code and terminal commands rather than passive semantic queries, and file system familiarity, which allows them to navigate massive text corpora as directory structures. These findings suggest that delegating long-context processing to coding agents offers an effective alternative to semantic search or context window scaling, opening new directions for long-context processing in LLMs. 

---
# The production of meaning in the processing of natural language 

**Authors**: Christopher J. Agostino, Quan Le Thien, Nayan D'Souza, Louis van der Elst  

**Link**: [PDF](https://arxiv.org/pdf/2603.20381)  

**Abstract**: Understanding the fundamental mechanisms governing the production of meaning in the processing of natural language is critical for designing safe, thoughtful, engaging, and empowering human-agent interactions. Experiments in cognitive science and social psychology have demonstrated that human semantic processing exhibits contextuality more consistent with quantum logical mechanisms than classical Boolean theories, and recent works have found similar results in large language models -- in particular, clear violations of the Bell inequality in experiments of contextuality during interpretation of ambiguous expressions. We explore the CHSH $|S|$ parameter -- the metric associated with the inequality -- across the inference parameter space of models spanning four orders of magnitude in scale, cross-referencing it with MMLU, hallucination rate, and nonsense detection benchmarks. We find that the interquartile range of the $|S|$ distribution -- the statistic that most sharply differentiates models from one another -- is completely orthogonal to all external benchmarks, while violation rate shows weak anticorrelation with all three benchmarks that does not reach significance. We investigate how $|S|$ varies with sampling parameters and word order, and discuss the information-theoretic constraints that genuine contextuality imposes on prompt injection defenses and its human analogue, whereby careful construction and maintenance of social contextuality can be carried out at scale -- manufacturing not consent but contextuality itself, a subtler and more fundamental form of manipulation that shapes the space of possible interpretations before any particular one is reached. 

---
# SciNav: A General Agent Framework for Scientific Coding Tasks 

**Authors**: Tianshu Zhang, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.20256)  

**Abstract**: Autonomous science agents built on large language models (LLMs) are increasingly used to generate hypotheses, design experiments, and produce reports. However, prior work mainly targets open-ended scientific problems with subjective outputs that are difficult to evaluate. Scientific coding benchmarks, by contrast, provide executable outputs for objective assessment. Existing approaches remain engineering-driven pipelines, revealing the need for structured, end-to-end science agent frameworks for scientific coding tasks. We address this gap by focusing on scientific coding tasks, where evaluation can be made rigorously, and introducing an agent framework SciNav (Scientific Navigator) that enables more effective solution exploration. Our framework is designed to operate under constrained search budgets, moving beyond reliance on pre-defined success metrics and prolonged search cycles. Inspired by findings that comparative judgments often reveal finer-grained quality differences and therefore provide greater discriminative power than absolute scoring, our framework leverages pairwise relative judgments within a tree search process to select top-K promising solution branches, prune low-potential ones, and progressively narrow down the solution candidates on the selected branches guided by relative comparisons. We demonstrate our agent's effectiveness across different types of tasks on two benchmarks. Experiments show that SciNav significantly outperforms direct prompting and prior agents like OpenHands and Self-Debug across different base models, task types, and difficulty levels, and exceeds different frontier comparators such as random selection and LLM absolute scoring. These results confirm the strength of our agent design and highlight the effectiveness of relative judgment-guided top-K search for high-quality scientific coding, marking a step toward more practical science agents. 

---
# Abjad-Kids: An Arabic Speech Classification Dataset for Primary Education 

**Authors**: Abdul Aziz Snoubara, Baraa Al_Maradni, Haya Al_Naal, Malek Al_Madrmani, Roaa Jdini, Seedra Zarzour, Khloud Al Jallad  

**Link**: [PDF](https://arxiv.org/pdf/2603.20255)  

**Abstract**: Speech-based AI educational applications have gained significant interest in recent years, particularly for children. However, children speech research remains limited due to the lack of publicly available datasets, especially for low-resource languages such as this http URL paper presents Abjad-Kids, an Arabic speech dataset designed for kindergarten and primary education, focusing on fundamental learning of alphabets, numbers, and colors. The dataset consists of 46397 audio samples collected from children aged 3 - 12 years, covering 141 classes. All samples were recorded under controlled specifications to ensure consistency in duration, sampling rate, and format. To address high intra-class similarity among Arabic phonemes and the limited samples per class, we propose a hierarchical audio classification based on CNN-LSTM architectures. Our proposed methodology decomposes alphabet recognition into a two-stage process: an initial grouping classification model followed by specialized classifiers for each group. Both strategies: static linguistic-based grouping and dynamic clustering-based grouping, were evaluated. Experimental results demonstrate that static linguistic-based grouping achieves superior performance. Comparisons between traditional machine learning with deep learning approaches, highlight the effectiveness of CNN-LSTM models combined with data augmentation. Despite achieving promising results, most of our experiments indicate a challenge with overfitting, which is likely due to the limited number of samples, even after data augmentation and model regularization. Thus, future work may focus on collecting additional data to address this issue. Abjad-Kids will be publicly available. We hope that Abjad-Kids enrich children representation in speech dataset, and be a good resource for future research in Arabic speech classification for kids. 

---
# FinReflectKG -- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering Systems 

**Authors**: Mahesh Kumar, Bhaskarjit Sarmah, Stefano Pasquali  

**Link**: [PDF](https://arxiv.org/pdf/2603.20252)  

**Abstract**: As organizations increasingly integrate AI-powered question-answering systems into financial information systems for compliance, risk assessment, and decision support, ensuring the factual accuracy of AI-generated outputs becomes a critical engineering challenge. Current Knowledge Graph (KG)-augmented QA systems lack systematic mechanisms to detect hallucinations - factually incorrect outputs that undermine reliability and user trust. We introduce FinBench-QA-Hallucination, a benchmark for evaluating hallucination detection methods in KG-augmented financial QA over SEC 10-K filings. The dataset contains 755 annotated examples from 300 pages, each labeled for groundedness using a conservative evidence-linkage protocol requiring support from both textual chunks and extracted relational triplets. We evaluate six detection approaches - LLM judges, fine-tuned classifiers, Natural Language Inference (NLI) models, span detectors, and embedding-based methods under two conditions: with and without KG triplets. Results show that LLM-based judges and embedding approaches achieve the highest performance (F1: 0.82-0.86) under clean conditions. However, most methods degrade significantly when noisy triplets are introduced, with Matthews Correlation Coefficient (MCC) dropping 44-84 percent, while embedding methods remain relatively robust with only 9 percent degradation. Statistical tests (Cochran's Q and McNemar) confirm significant performance differences (p < 0.001). Our findings highlight vulnerabilities in current KG-augmented systems and provide insights for building reliable financial information systems, where hallucinations can lead to regulatory violations and flawed decisions. The benchmark also offers a framework for integrating AI reliability evaluation into information system design across other high-stakes domains such as healthcare, legal, and government. 

---
# Decoding the decoder: Contextual sequence-to-sequence modeling for intracortical speech decoding 

**Authors**: Michal Olak, Tommaso Boccato, Matteo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2603.20246)  

**Abstract**: Speech brain--computer interfaces require decoders that translate intracortical activity into linguistic output while remaining robust to limited data and day-to-day variability. While prior high-performing systems have largely relied on framewise phoneme decoding combined with downstream language models, it remains unclear what contextual sequence-to-sequence decoding contributes to sublexical neural readout, robustness, and interpretability. We evaluated a multitask Transformer-based sequence-to-sequence model for attempted speech decoding from area 6v intracortical recordings. The model jointly predicts phoneme sequences, word sequences, and auxiliary acoustic features. To address day-to-day nonstationarity, we introduced the Neural Hammer Scalpel (NHS) calibration module, which combines global alignment with feature-wise modulation. We further analyzed held-out-day generalization and attention patterns in the encoder and decoders. On the Willett et al. dataset, the proposed model achieved a state-of-the-art phoneme error rate of 14.3%. Word decoding reached 25.6% WER with direct decoding and 19.4% WER with candidate generation and rescoring. NHS substantially improved both phoneme and word decoding relative to linear or no day-specific transform, while held-out-day experiments showed increasing degradation on unseen days with temporal distance. Attention visualizations revealed recurring temporal chunking in encoder representations and distinct use of these segments by phoneme and word decoders. These results indicate that contextual sequence-to-sequence modeling can improve the fidelity of neural-to-phoneme readout from intracortical speech signals and suggest that attention-based analyses can generate useful hypotheses about how neural speech evidence is segmented and accumulated over time. 

---
# Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference 

**Authors**: Patrick Wilhelm, Thorsten Wittkopp, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2603.20224)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks but come with substantial energy and computational costs, particularly in request-heavy scenarios. In many real-world applications, the full scale and capabilities of LLMs are often unnecessary, as Small Language Models (SLMs) can provide accurate responses for simpler text generation tasks. When enhanced with advanced reasoning strategies, such as Chain-of-Thought (CoT) prompting or Majority Voting, SLMs can approach the performance of larger models while reducing overall computational requirements. However, these strategies can also introduce additional energy costs, creating an energy-accuracy trade-off. Our analysis examines these trade-offs in test-time compute strategies for smaller models compared to larger ones, using the MMLU benchmark. Additionally, we explore the input-output token dynamics of transformer architectures, which result in nonlinear hardware energy operation curves for LLMs. To bridge AI research with its physical impact, we propose \textit{energy efficiency metrics}, including Energy-per-Token, as complements to traditional accuracy benchmarks. Beyond model selection, we propose controlled reasoning in CoT token generation, using operating curves to regulate reasoning depth dynamically. This vision integrates a energy-aware routing mechanism, ensuring that model selection and inference strategies balance accuracy for sustainable AI deployment. 

---
# Linguistic Signatures for Enhanced Emotion Detection 

**Authors**: Florian Lecourt, Madalina Croitoru, Konstantin Todorov  

**Link**: [PDF](https://arxiv.org/pdf/2603.20222)  

**Abstract**: Emotion detection is a central problem in NLP, with recent progress driven by transformer-based models trained on established datasets. However, little is known about the linguistic regularities that characterize how emotions are expressed across different corpora and labels. This study examines whether linguistic features can serve as reliable interpretable signals for emotion recognition in text. We extract emotion-specific linguistic signatures from 13 English datasets and evaluate how incorporating these features into transformer models impacts performance. Our RoBERTa-based models enriched with high level linguistic features achieve consistent performance gains of up to +2.4 macro F1 on the GoEmotions benchmark, showing that explicit lexical cues can complement neural representations and improve robustness in predicting emotion categories. 

---
# Thinking into the Future: Latent Lookahead Training for Transformers 

**Authors**: Lorenzo Noci, Gregor Bachmann, Seyed-Mohsen Moosavi-Dezfooli, Moin Nabi  

**Link**: [PDF](https://arxiv.org/pdf/2603.20219)  

**Abstract**: Autoregressive language models trained with next-token prediction generate text by sampling one discrete token at a time. Although very scalable, this objective forces the model to commit at every step, preventing it from exploring or reflecting upon multiple plausible continuations. Furthermore, the compute allocation across tokens is uniform; every token is formed based on a single forward-pass, potentially limiting the model's expressiveness in cases where difficult tokens require inherently more compute. Towards addressing these limitations, we introduce latent lookahead, a training strategy that enables models to "think" before generating: at selected positions in the sequence, before committing to the next token, the model performs a multi-step lookahead in latent space. More precisely, instead of sampling future tokens, we leverage the network's latent space by recursively feeding its hidden states back into the context for $\tau$ steps, investing more compute on predicting that token. This produces $\tau$ latent predictions that are supervised against the next $\tau$ ground-truth tokens, encouraging the model to "lookahead" and refine its prediction. We show that latent lookahead substantially outperforms both autoregressive and non-autoregressive baselines on planning tasks such as maze solving, Sudoku, and ProsQA, where foresight is essential. 

---
# An experimental study of KV cache reuse strategies in chunk-level caching systems 

**Authors**: Samuel Cestola, Tianxiang Xia, Zheng Weiyan, Zheng Pengfei, Diego Didona  

**Link**: [PDF](https://arxiv.org/pdf/2603.20218)  

**Abstract**: Retrieval-augmented generation improves large language models' accuracy by adding relevant retrieved text to the prompt. Chunk level caching (CLC) accelerates inference by precomputing KV caches for these retrieved chunks and reusing them. However, these caches miss cross-attention dependencies between chunks, which can reduce output quality. Several methods try to improve CLC accuracy using different techniques. We make two main contributions. First, we show that existing CLC approaches have fundamental limitations that limit their accuracy or their applicability. We back this conclusion with an extensive CLC system experimental evaluation. Second, we observe that existing CLC techniques are complementary. We leverage this insight to propose a new CLC design that carefully combines them and achieves better accuracy. 

---
# Expected Reward Prediction, with Applications to Model Routing 

**Authors**: Kenan Hasanaliyev, Silas Alberti, Jenny Hamer, Dheeraj Rajagopal, Kevin Robinson, Jasper Snoek, Victor Veitch, Alexander Nicholas D'Amour  

**Link**: [PDF](https://arxiv.org/pdf/2603.20217)  

**Abstract**: Reward models are a standard tool to score responses from LLMs. Reward models are built to rank responses to a fixed prompt sampled from a single model, for example to choose the best of n sampled responses. In this paper, we study whether scores from response-level reward models lifted to score a model's suitability for a prompt, prior to seeing responses from that model. Specifically, we show that it is straightforward to predict the expected reward that an LLM would earn from the reward model under repeated sampling. Further, we show that these expected reward predictions are precise and discriminative enough to support an application to a model routing protocol that routes prompts to models at inference time to maximize reward while controlling computational cost. We demonstrate the performance of this routing procedure on the open-perfectblend dataset, using a model pool composed of Llama3.1-Instruct 8B/70B, Gemma2-IT 9B/27B, and Gemma1-IT 7B models. Our simple expected reward prediction--based routing (ERP) outperforms baselines that route prompts to models with the best average performance within each prompt's category, and explains the success of more complex routing protocols that implicitly estimate an expected reward. Our approach has the added advantage of being trivially extensible as new models are added to the pool. 

---
# Locally Coherent Parallel Decoding in Diffusion Language Models 

**Authors**: Michael Hersche, Nicolas Menet, Ronan Tanios, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2603.20216)  

**Abstract**: Diffusion language models (DLMs) have emerged as a promising alternative to autoregressive (AR) models, offering sub-linear generation latency and bidirectional capabilities that are particularly appealing for code generation and editing. Achieving sub-linear latency in discrete DLMs requires predicting multiple tokens in parallel. However, standard DLMs sample tokens independently from conditional marginal distributions, failing to capture the joint dependencies among concurrently generated tokens. As a result, they often lead to syntactic inconsistencies and break multi-token structures. In this work, we introduce CoDiLA (Coherent Diffusion with Local Autoregression), a method that reconciles parallel sampling with local dependency modeling. Rather than forcing the DLM to resolve fine-grained syntax, CoDiLA delegates local decoding to a small, auxiliary AR model operating on the diffusion latents. This design allows for parallel block generation while ensuring sequential validity within each block and maintaining core DLM capabilities, including bidirectional modeling across blocks. We demonstrate that using a highly compact auxiliary AR model (e.g., 0.6B parameters) effectively eliminates coherence artifacts, establishing a new Pareto frontier for accuracy and speed in code generation benchmarks. 

---
# Multi-Agent Debate with Memory Masking 

**Authors**: Hongduan Tian, Xiao Feng, Ziyuan Zhao, Xiangyu Zhu, Rolan Yan, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.20215)  

**Abstract**: Large language models (LLMs) have recently demonstrated impressive capabilities in reasoning tasks. Currently, mainstream LLM reasoning frameworks predominantly focus on scaling up inference-time sampling to enhance performance. In particular, among all LLM reasoning frameworks, *multi-agent debate* (MAD), which employs multiple LLMs as agents to perform reasoning in the way of multi-round debate, has emerged as a powerful reasoning paradigm since it allows agents to access previous memories to alleviate fallacious content and refine their reasoning iteratively in each debate round. However, although MAD significantly improves the reasoning capabilities of LLMs, in this paper, we observe that there remain erroneous memories, and LLM agents are vulnerable to these erroneous memories. To explore this phenomenon, we provide a theoretical insight that the performance of MAD is highly dependent on the quality of memories derived from the previous debate, indicating that the existence of erroneous memories poses a threat to the performance of MAD. To address this problem, we introduce a simple yet effective multi-agent debate framework, *multi-agent debate with memory masking* (MAD-M$^2$), to improve the robustness of MAD by allowing LLM agents to mask erroneous memories from the previous debate round at the beginning of each debate round. In this way, MAD-M$^2$ can polish the contextual information before each debate round by preserving informative and meaningful memories while discarding the erroneous memories. Extensive experiments and analyses on mainstream mathematical and logical reasoning benchmarks demonstrate that MAD-M$^2$ can identify the erroneous memories and achieve better performance in reasoning than MAD. 

---
# Fast-Slow Thinking RM: Efficient Integration of Scalar and Generative Reward Models 

**Authors**: Jiayun Wu, Peixu Hou, Shan Qu, Peng Zhang, Ning Gu, Tun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20212)  

**Abstract**: Reward models (RMs) are critical for aligning Large Language Models via Reinforcement Learning from Human Feedback (RLHF). While Generative Reward Models (GRMs) achieve superior accuracy through chain-of-thought (CoT) reasoning, they incur substantial computational costs. Conversely, Scalar Reward Models (SRMs) offer efficiency but suffer from limited performance and adaptability in complex scenarios.
We introduce Fast-Slow Thinking Reward Models (F/S-RM), a hybrid RM architecture inspired by Dual Process Theory. It trains a single model to integrate two distinct reward paradigms: first-token prediction as a scalar score (fast thinking) and CoT-based judgment (slow thinking), regulated by a dual-confidence activation mechanism that determines when to activate slow thinking.
F/S-RM achieves a 1.2% relative performance improvement over state-of-the-art models while reducing token consumption by 20.8%. Code and data will be publicly available. 

---
# CRoCoDiL: Continuous and Robust Conditioned Diffusion for Language 

**Authors**: Roy Uziel, Omer Belhasin, Itay Levi, Akhiad Bercovich, Ran El-Yaniv, Ran Zilberstein, Michael Elad  

**Link**: [PDF](https://arxiv.org/pdf/2603.20210)  

**Abstract**: Masked Diffusion Models (MDMs) provide an efficient non-causal alternative to autoregressive generation but often struggle with token dependencies and semantic incoherence due to their reliance on discrete marginal distributions. We address these limitations by shifting the diffusion process into a continuous sentence-level semantic space. We propose CRoCoDiL (Continuous and Robust Conditioned Diffusion for Language), a unified fine-tuning approach that jointly trains an encoder-demasker architecture, grounding the MDM demasking in continuous latent representations. This leads to the formation of a novel autoencoder in which decoding is obtained by an MDM algorithm. Relying on the same framework, we introduce two unconditional text synthesis algorithms: Continuous-Then-Discrete (ConThenDisc), a hybrid-diffusion approach that first generates latent representations in continuous space and then decodes these to tokens via an MDM, and Continuous-Within-Discrete (ConWithinDisc), a multi-diffusion strategy that refines latent representations throughout the discrete sampling process. Experiments using LLaDA show that our methods achieve superior generation quality and more than 10x faster sampling speeds in an unconditional setting. 

---
# Children's Intelligence Tests Pose Challenges for MLLMs? KidGym: A 2D Grid-Based Reasoning Benchmark for MLLMs 

**Authors**: Hengwei Ye, Yuanting Guan, Yuxuan Ge, Tianying Zhu, Zhenhan Guan, Yijia Zhong, Yijing Zhang, Han Zhang, Yingna Wu, Zheng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2603.20209)  

**Abstract**: Multimodal Large Language Models (MLLMs) combine the linguistic strengths of LLMs with the ability to process multimodal data, enbaling them to address a broader range of visual tasks. Because MLLMs aim at more general, human-like competence than language-only models, we take inspiration from the Wechsler Intelligence Scales - an established battery for evaluating children by decomposing intelligence into interpretable, testable abilities. We introduce KidGym, a comprehensive 2D grid-based benchmark for assessing five essential capabilities of MLLMs: Execution, Perception Reasoning, Learning, Memory and Planning. The benchmark comprises 12 unique tasks, each targeting at least one core capability, specifically designed to guage MLLMs' adaptability and developmental potential, mirroring the stages of children's cognitive growth. Additionally, our tasks encompass diverse scenarios and objects with randomly generated layouts, ensuring a more accurate and robust evluation of MLLM capabilities. KidGym is designed to be fully user-customizable and extensible, allowing researchers to create new evaluation scenarios and adjust difficuly levels to accommodate the rapidly growing MLLM community. Through the evaluation of state-of-the-art MLLMs using KidGym, we identified significant insights into model capabilities and revealed several limitations of current models. We release our benchmark at: this https URL. 

---
# RedacBench: Can AI Erase Your Secrets? 

**Authors**: Hyunjun Jeon, Kyuyoung Kim, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2603.20208)  

**Abstract**: Modern language models can readily extract sensitive information from unstructured text, making redaction -- the selective removal of such information -- critical for data security. However, existing benchmarks for redaction typically focus on predefined categories of data such as personally identifiable information (PII) or evaluate specific techniques like masking. To address this limitation, we introduce RedacBench, a comprehensive benchmark for evaluating policy-conditioned redaction across domains and strategies. Constructed from 514 human-authored texts spanning individual, corporate, and government sources, paired with 187 security policies, RedacBench measures a model's ability to selectively remove policy-violating information while preserving the original semantics. We quantify performance using 8,053 annotated propositions that capture all inferable information in each text. This enables assessment of both security -- the removal of sensitive propositions -- and utility -- the preservation of non-sensitive propositions. Experiments across multiple redaction strategies and state-of-the-art language models show that while more advanced models can improve security, preserving utility remains a challenge. To facilitate future research, we release RedacBench along with a web-based playground for dataset customization and evaluation. Available at this https URL. 

---
# Enhancing Safety of Large Language Models via Embedding Space Separation 

**Authors**: Xu Zhao, Xiting Wang, Weiran Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.20206)  

**Abstract**: Large language models (LLMs) have achieved impressive capabilities, yet ensuring their safety against harmful prompts remains a critical challenge. Recent work has revealed that the latent representations (embeddings) of harmful and safe queries in LLMs typically exhibit linear separability, a property that has been exploited to construct attacks by perturbing the embeddings of harmful queries towards the safe subspace. Motivated by this observation, we propose a representation-level fine-tuning approach, named Embedding Space Separation (ES2), which improves LLM safety by explicitly enlarging the distance between harmful and safe representations in the embedding space. To prevent degradation of model's general capabilities, we introduce a Kullback-Leibler (KL) divergence regularization term into the loss function, which constrains the logits of the fine-tuned model to align with those of the original base model on harmless inputs. We evaluate our method on several open-source LLMs using standard safety benchmarks. Extensive experimental results demonstrate that our approach substantially improves model safety while maintaining comparable general capabilities. 

---
# WorldCache: Content-Aware Caching for Accelerated Video World Models 

**Authors**: Umair Nawaz, Ahmed Heakl, Ufaq Khan, Abdelrahman Shaker, Salman Khan, Fahad Shahbaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2603.22286)  

**Abstract**: Diffusion Transformers (DiTs) power high-fidelity video world models but remain computationally expensive due to sequential denoising and costly spatio-temporal attention. Training-free feature caching accelerates inference by reusing intermediate activations across denoising steps; however, existing methods largely rely on a Zero-Order Hold assumption i.e., reusing cached features as static snapshots when global drift is small. This often leads to ghosting artifacts, blur, and motion inconsistencies in dynamic scenes. We propose \textbf{WorldCache}, a Perception-Constrained Dynamical Caching framework that improves both when and how to reuse features. WorldCache introduces motion-adaptive thresholds, saliency-weighted drift estimation, optimal approximation via blending and warping, and phase-aware threshold scheduling across diffusion steps. Our cohesive approach enables adaptive, motion-consistent feature reuse without retraining. On Cosmos-Predict2.5-2B evaluated on PAI-Bench, WorldCache achieves \textbf{2.3$\times$} inference speedup while preserving \textbf{99.4\%} of baseline quality, substantially outperforming prior training-free caching approaches. Our code can be accessed on \href{this https URL}{World-Cache}. 

---
# ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model 

**Authors**: Haichao Zhang, Yijiang Li, Shwai He, Tushar Nagarajan, Mingfei Chen, Jianglin Lu, Ang Li, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22281)  

**Abstract**: Recent progress in latent world models (e.g., V-JEPA2) has shown promising capability in forecasting future world states from video observations. Nevertheless, dense prediction from a short observation window limits temporal context and can bias predictors toward local, low-level extrapolation, making it difficult to capture long-horizon semantics and reducing downstream utility. Vision--language models (VLMs), in contrast, provide strong semantic grounding and general knowledge by reasoning over uniformly sampled frames, but they are not ideal as standalone dense predictors due to compute-driven sparse sampling, a language-output bottleneck that compresses fine-grained interaction states into text-oriented representations, and a data-regime mismatch when adapting to small action-conditioned datasets. We propose a VLM-guided JEPA-style latent world modeling framework that combines dense-frame dynamics modeling with long-horizon semantic guidance via a dual-temporal pathway: a dense JEPA branch for fine-grained motion and interaction cues, and a uniformly sampled VLM \emph{thinker} branch with a larger temporal stride for knowledge-rich guidance. To transfer the VLM's progressive reasoning signals effectively, we introduce a hierarchical pyramid representation extraction module that aggregates multi-layer VLM representations into guidance features compatible with latent prediction. Experiments on hand-manipulation trajectory prediction show that our method outperforms both a strong VLM-only baseline and a JEPA-predictor baseline, and yields more robust long-horizon rollout behavior. 

---
# Dyadic: A Scalable Platform for Human-Human and Human-AI Conversation Research 

**Authors**: David M. Markowitz  

**Link**: [PDF](https://arxiv.org/pdf/2603.22227)  

**Abstract**: Conversation is ubiquitous in social life, but the empirical study of this interactive process has been thwarted by tools that are insufficiently modular and unadaptive to researcher needs. To relieve many constraints in conversation research, the current tutorial presents an overview and introduction to a new tool, Dyadic (this https URL), a web-based platform for studying human-human and human-AI conversations using text-based or voice-based chats. Dyadic is distinct from other platforms by offering studies with multiple modalities, AI suggestions (e.g., in human-human studies, AI can suggest responses to a participant), live monitoring (e.g., researchers can evaluate, in real time, chats between communicators), and survey deployment (e.g., Likert-type scales, feeling thermometers, and open-ended text boxes can be sent to humans for in situ evaluations of the interaction), among other consequential features. No coding is required to operate Dyadic directly, and integrations with existing survey platforms are offered. 

---
# SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection 

**Authors**: Kexian Tang, Jiani Wang, Shaowen Wang, Kaifeng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.22213)  

**Abstract**: While large language models (LLMs) are pretrained on massive amounts of data, their knowledge coverage remains incomplete in specialized, data-scarce domains, motivating extensive efforts to study synthetic data generation for knowledge injection. We propose SPA (Scaling Prompt-engineered Augmentation), a simple but tough-to-beat baseline that uses a small set of carefully designed prompts to generate large-scale synthetic data for knowledge injection. Through systematic comparisons, we find that SPA outperforms several strong baselines. Furthermore, we identify two key limitations of prior approaches: (1) while RL-based methods may improve the token efficiency of LLM-based data augmentation at small scale, they suffer from diversity collapse as data scales, leading to diminishing returns; and (2) while multi-stage prompting may outperform simple augmentation methods, their advantages can disappear after careful prompt tuning. Our results suggest that, for knowledge injection, careful prompt design combined with straightforward large-scale augmentation can be surprisingly effective, and we hope SPA can serve as a strong baseline for future studies in this area. Our code is available at this https URL. 

---
# ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention 

**Authors**: Xinyan Wang, Xiaogeng Liu, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.22016)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong accuracy on challenging tasks by generating long Chain-of-Thought traces, but suffer from overthinking. Even after reaching the correct answer, they continue generating redundant reasoning steps. This behavior increases latency and compute cost and can also lead to answer drift. Existing mitigation methods either require training-heavy backbone modification or rely on hand-crafted heuristics that do not truly capture overthinking patterns. We propose ROM, the first method that formulates overthinking mitigation as a streaming prediction-and-control problem. ROM attaches a lightweight detection head to the late-layer hidden states of a frozen large language model backbone. It monitors tokens in real time and triggers an early transition to the final answer once overthinking is detected. We also introduce token-level supervision based on solution correctness boundaries and a data augmentation strategy that reduces distilled-data bias. Across seven benchmarks, ROM achieves the highest accuracy (93.51%), the shortest responses (1,159 tokens), and the best response efficiency. Compared with the vanilla baseline, it reduces response length by 47.2% and improves efficiency by 121%. These results show that streaming detection is a promising approach to real-time overthinking mitigation. 

---
# On the Challenges and Opportunities of Learned Sparse Retrieval for Code 

**Authors**: Simon Lupart, Maxime Louis, Thibault Formal, Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2603.22008)  

**Abstract**: Retrieval over large codebases is a key component of modern LLM-based software engineering systems. Existing approaches predominantly rely on dense embedding models, while learned sparse retrieval (LSR) remains largely unexplored for code. However, applying sparse retrieval to code is challenging due to subword fragmentation, semantic gaps between natural-language queries and code, diversity of programming languages and sub-tasks, and the length of code documents, which can harm sparsity and latency. We introduce SPLADE-Code, the first large-scale family of learned sparse retrieval models specialized for code retrieval (600M-8B parameters). Despite a lightweight one-stage training pipeline, SPLADE-Code achieves state-of-the-art performance among retrievers under 1B parameters (75.4 on MTEB Code) and competitive results at larger scales (79.0 with 8B). We show that learned expansion tokens are critical to bridge lexical and semantic matching, and provide a latency analysis showing that LSR enables sub-millisecond retrieval on a 1M-passage collection with little effectiveness loss. 

---
# SecureBreak -- A dataset towards safe and secure models 

**Authors**: Marco Arazzi, Vignesh Kumar Kembu, Antonino Nocera  

**Link**: [PDF](https://arxiv.org/pdf/2603.21975)  

**Abstract**: Large language models are becoming pervasive core components in many real-world applications. As a consequence, security alignment represents a critical requirement for their safe deployment. Although previous related works focused primarily on model architectures and alignment methodologies, these approaches alone cannot ensure the complete elimination of harmful generations. This concern is reinforced by the growing body of scientific literature showing that attacks, such as jailbreaking and prompt injection, can bypass existing security alignment mechanisms. As a consequence, additional security strategies are needed both to provide qualitative feedback on the robustness of the obtained security alignment at the training stage, and to create an ``ultimate'' defense layer to block unsafe outputs possibly produced by deployed models. To provide a contribution in this scenario, this paper introduces SecureBreak, a safety-oriented dataset designed to support the development of AI-driven solutions for detecting harmful LLM outputs caused by residual weaknesses in security alignment. The dataset is highly reliable due to careful manual annotation, where labels are assigned conservatively to ensure safety. It performs well in detecting unsafe content across multiple risk categories. Tests with pre-trained LLMs show improved results after fine-tuning on SecureBreak. Overall, the dataset is useful both for post-generation safety filtering and for guiding further model alignment and security improvements. 

---
# Demystifying Reinforcement Learning for Long-Horizon Tool-Using Agents: A Comprehensive Recipe 

**Authors**: Xixi Wu, Qianguo Sun, Ruiyang Zhang, Chao Song, Junlong Wu, Yiyan Qi, Hong Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.21972)  

**Abstract**: Reinforcement Learning (RL) is essential for evolving Large Language Models (LLMs) into autonomous agents capable of long-horizon planning, yet a practical recipe for scaling RL in complex, multi-turn environments remains elusive. This paper presents a systematic empirical study using TravelPlanner, a challenging testbed requiring tool orchestration to satisfy multifaceted constraints. We decompose the agentic RL design space along 5 axes: reward shaping, model scaling, data composition, algorithm selection, and environmental stability. Our controlled experiments yield 7 key takeaways, e.g., (1) reward and algorithm choices are scale-dependent as smaller models benefit from staged rewards and enhanced exploration, whereas larger models converge efficiently with simpler dense rewards, (2) ~ 1K training samples with a balanced difficulty mixture mark a sweet spot for both in-domain and out-of-domain performance, and (3) environmental stability is critical to prevent policy degradation. Based on our distilled recipe, our RL-trained models achieve state-of-the-art performance on TravelPlanner, significantly outperforming leading LLMs. 

---
# BHDD: A Burmese Handwritten Digit Dataset 

**Authors**: Swan Htet Aung, Hein Htet, Htoo Say Wah Khaing, Thuya Myo Nyunt  

**Link**: [PDF](https://arxiv.org/pdf/2603.21966)  

**Abstract**: We introduce the Burmese Handwritten Digit Dataset (BHDD), a collection of 87,561 grayscale images of handwritten Burmese digits in ten classes. Each image is 28x28 pixels, following the MNIST format. The training set has 60,000 samples split evenly across classes; the test set has 27,561 samples with class frequencies as they arose during collection. Over 150 people of different ages and backgrounds contributed samples. We analyze the dataset's class distribution, pixel statistics, and morphological variation, and identify digit pairs that are easily confused due to the round shapes of the Myanmar script. Simple baselines (an MLP, a two-layer CNN, and an improved CNN with batch normalization and augmentation) reach 99.40%, 99.75%, and 99.83% test accuracy respectively. BHDD is available under CC BY-SA 4.0 at this https URL 

---
# Disentangling Speaker Traits for Deepfake Source Verification via Chebyshev Polynomial and Riemannian Metric Learning 

**Authors**: Xi Xuan, Wenxin Zhang, Zhiyu Li, Jennifer Williams, Ville Hautamäki, Tomi H. Kinnunen  

**Link**: [PDF](https://arxiv.org/pdf/2603.21875)  

**Abstract**: Speech deepfake source verification systems aims to determine whether two synthetic speech utterances originate from the same source generator, often assuming that the resulting source embeddings are independent of speaker traits. However, this assumption remains unverified. In this paper, we first investigate the impact of speaker factors on source verification. We propose a speaker-disentangled metric learning (SDML) framework incorporating two novel loss functions. The first leverages Chebyshev polynomial to mitigate gradient instability during disentanglement optimization. The second projects source and speaker embeddings into hyperbolic space, leveraging Riemannian metric distances to reduce speaker information and learn more discriminative source features. Experimental results on MLAAD benchmark, evaluated under four newly proposed protocols designed for source-speaker disentanglement scenarios, demonstrate the effectiveness of SDML framework. The code, evaluation protocols and demo website are available at this https URL. 

---
# The Presupposition Problem in Representation Genesis 

**Authors**: Yiling Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21745)  

**Abstract**: Large language models are the first systems to achieve high cognitive performance without clearly undergoing representation genesis: the transition from a non-representing physical system to one whose states guide behavior in a content-sensitive way. Prior cognitive systems had already made this transition before we could examine it, and philosophy of mind treated genesis as a background condition rather than an explanatory target. LLMs provide a case that does not clearly involve this transition, making the genesis question newly urgent: if genesis did not occur, which cognitive capacities are affected, and why? We currently lack the conceptual resources to answer this. The reason, this paper argues, is structural. Major frameworks in philosophy of mind, including the Language of Thought hypothesis, teleosemantics, predictive processing, enactivism, and genetic phenomenology, share a common feature when applied to the genesis question: at some explanatory step, each deploys concepts whose explanatory purchase depends on the system already being organized as a representer. This pattern, which we call the Representation Presupposition structure, generates systematic explanatory deferral. Attempts to explain the first acquisition of content-manipulable representation within the existing categorical vocabulary import resources from the representational side of the transition itself. We call this the Representation Regress. The paper offers a conceptual diagnosis rather than a new theory, establishing the structure of the problem and deriving two minimum adequacy conditions for any account that avoids this pattern. LLMs make the absence of such a theory consequential rather than merely theoretical. 

---
# The Reasoning Error About Reasoning: Why Different Types of Reasoning Require Different Representational Structures 

**Authors**: Yiling Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21736)  

**Abstract**: Different types of reasoning impose different structural demands on representational systems, yet no systematic account of these demands exists across psychology, AI, and philosophy of mind. I propose a framework identifying four structural properties of representational systems: operability, consistency, structural preservation, and compositionality. These properties are demanded to different degrees by different forms of reasoning, from induction through analogy and causal inference to deduction and formal logic. Each property excludes a distinct class of reasoning failure. The analysis reveals a principal structural boundary: reasoning types below it can operate on associative, probabilistic representations, while those above it require all four properties to be fully satisfied. Scaling statistical learning without structural reorganization is insufficient to cross this boundary, because the structural guarantees required by deductive reasoning cannot be approximated through probabilistic means. Converging evidence from AI evaluation, developmental psychology, and cognitive neuroscience supports the framework at different levels of directness. Three testable predictions are derived, including compounding degradation, selective vulnerability to targeted structural disruption, and irreducibility under scaling. The framework is a necessary-condition account, agnostic about representational format, that aims to reorganize existing debates rather than close them. 

---
# EvoIdeator: Evolving Scientific Ideas through Checklist-Grounded Reinforcement Learning 

**Authors**: Andreas Sauter, Yuyue Zhao, Jacopo Urbani, Wenxiang Hu, Zaiqiao Meng, Lun Zhou, Xiaohui Yan, Yougang Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21728)  

**Abstract**: Scientific idea generation is a cornerstone of autonomous knowledge discovery, yet the iterative evolution required to transform initial concepts into high-quality research proposals remains a formidable challenge for Large Language Models (LLMs). Existing Reinforcement Learning (RL) paradigms often rely on rubric-based scalar rewards that provide global quality scores but lack actionable granularity. Conversely, language-based refinement methods are typically confined to inference-time prompting, targeting models that are not explicitly optimized to internalize such critiques. To bridge this gap, we propose \textbf{EvoIdeator}, a framework that facilitates the evolution of scientific ideas by aligning the RL training objective with \textbf{checklist-grounded feedback}. EvoIdeator leverages a structured judge model to generate two synergistic signals: (1) \emph{lexicographic rewards} for multi-dimensional optimization, and (2) \emph{fine-grained language feedback} that offers span-level critiques regarding grounding, feasibility, and methodological rigor. By integrating these signals into the RL loop, we condition the policy to systematically utilize precise feedback during both optimization and inference. Extensive experiments demonstrate that EvoIdeator, built on Qwen3-4B, significantly outperforms much larger frontier models across key scientific metrics. Crucially, the learned policy exhibits strong generalization to diverse external feedback sources without further fine-tuning, offering a scalable and rigorous path toward self-refining autonomous ideation. 

---
# Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization 

**Authors**: Hung-Hsuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.21676)  

**Abstract**: Standard Transformers have a fixed computational depth, fundamentally limiting their ability to generalize to tasks requiring variable-depth reasoning, such as multi-hop graph traversal or nested logic. We propose a depth-recurrent Transformer that decouples computational depth from parameter count by iteratively applying a shared-weight Transformer block in latent space -- enabling the model to trade recurrence steps for deeper reasoning at inference time. Our architecture incorporates three mechanisms to make deep recurrence (20+ steps) stable: (1) a silent thinking objective that supervises only the final output, forcing genuine multi-step reasoning rather than intermediate heuristic shortcuts; (2) LayerScale initialization to protect fragile reasoning states from untrained layer noise; and (3) an identity-biased recurrence that creates a gradient highway across many steps. We evaluate on three compositional reasoning domains with decreasing inductive biases: graph reachability (strict adjacency masking), nested boolean logic (relative positioning), and unstructured relational text (where sequence position provides no structural hints). Across all tasks, we observe a clear \emph{computational frontier} -- a boundary where performance transitions from chance to near-perfect as thinking steps scale with task complexity. Moreover, these tasks reveal qualitatively different generalization behaviors: precise but brittle (graph), approximate but robust (logic), and autonomous latent routing without structural hints (text). This progression illuminates how the interplay between a task-invariant recurrent reasoning core and task-specific perceptual interfaces shapes out-of-distribution (OOD) generalization, offering a mechanistic perspective on vertical chain-of-thought that complements the prevailing horizontal token-generation paradigm. 

---
# Silicon Bureaucracy and AI Test-Oriented Education: Contamination Sensitivity and Score Confidence in LLM Benchmarks 

**Authors**: Yiliang Song, Hongjun An, Jiangan Chen, Xuanchen Yan, Huan Song, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.21636)  

**Abstract**: Public benchmarks increasingly govern how large language models (LLMs) are ranked, selected, and deployed. We frame this benchmark-centered regime as Silicon Bureaucracy and AI Test-Oriented Education, and argue that it rests on a fragile assumption: that benchmark scores directly reflect genuine generalization. In practice, however, such scores may conflate exam-oriented competence with principled capability, especially when contamination and semantic leakage are difficult to exclude from modern training pipelines. We therefore propose an audit framework for analyzing contamination sensitivity and score confidence in LLM benchmarks. Using a router-worker setup, we compare a clean-control condition with noisy conditions in which benchmark problems are systematically deleted, rewritten, and perturbed before being passed downstream. For a genuinely clean benchmark, noisy conditions should not consistently outperform the clean-control baseline. Yet across multiple models, we find widespread but heterogeneous above-baseline gains under noisy conditions, indicating that benchmark-related cues may be reassembled and can reactivate contamination-related memory. These results suggest that similar benchmark scores may carry substantially different levels of confidence. Rather than rejecting benchmarks altogether, we argue that benchmark-based evaluation should be supplemented with explicit audits of contamination sensitivity and score confidence. 

---
# PRISM: Breaking the O(n) Memory Wall in Long-Context LLM Inference via O(1) Photonic Block Selection 

**Authors**: Hyoseok Park, Yeonsang Park  

**Link**: [PDF](https://arxiv.org/pdf/2603.21576)  

**Abstract**: Long-context LLM inference is bottlenecked not by compute but by the O(n) memory bandwidth cost of scanning the KV cache at every decode step -- a wall that no amount of arithmetic scaling can break. Recent photonic accelerators have demonstrated impressive throughput for dense attention computation; however, these approaches inherit the same O(n) memory scaling as electronic attention when applied to long contexts. We observe that the real leverage point is the coarse block-selection step: a memory-bound similarity search that determines which KV blocks to fetch. We identify, for the first time, that this task is structurally matched to the photonic broadcast-and-weight paradigm -- the query fans out to all candidates via passive splitting, signatures are quasi-static (matching electro-optic MRR programming), and only rank order matters (relaxing precision to 4-6 bits). Crucially, the photonic advantage grows with context length: as N increases, the electronic scan cost rises linearly while the photonic evaluation remains O(1). We instantiate this insight in PRISM (Photonic Ranking via Inner-product Similarity with Microring weights), a thin-film lithium niobate (TFLN) similarity engine. Hardware-impaired needle-in-a-haystack evaluation on Qwen2.5-7B confirms 100% accuracy from 4K through 64K tokens at k=32, with 16x traffic reduction at 64K context. PRISM achieves a four-order-of-magnitude energy advantage over GPU baselines at practical context lengths (n >= 4K). 

---
# Beyond Correlation: Refutation-Validated Aspect-Based Sentiment Analysis for Explainable Energy Market Returns 

**Authors**: Wihan van der Heever, Keane Ong, Ranjan Satapathy, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2603.21473)  

**Abstract**: This paper proposes a refutation-validated framework for aspect-based sentiment analysis in financial markets, addressing the limitations of correlational studies that cannot distinguish genuine associations from spurious ones. Using X data for the energy sector, we test whether aspect-level sentiment signals show robust, refutation-validated relationships with equity returns. Our pipeline combines net-ratio scoring with z-normalization, OLS with Newey West HAC errors, and refutation tests including placebo, random common cause, subset stability, and bootstrap. Across six energy tickers, only a few associations survive all checks, while renewables show aspect and horizon specific responses. While not establishing causality, the framework provides statistically robust, directionally interpretable signals, with limited sample size (six stocks, one quarter) constraining generalizability and framing this work as a methodological proof of concept. 

---
# DSPA: Dynamic SAE Steering for Data-Efficient Preference Alignment 

**Authors**: James Wedgwood, Aashiq Muhamed, Mona T. Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2603.21461)  

**Abstract**: Preference alignment is usually achieved by weight-updating training on preference data, which adds substantial alignment-stage compute and provides limited mechanistic visibility. We propose Dynamic SAE Steering for Preference Alignment (DSPA), an inference-time method that makes sparse autoencoder (SAE) steering prompt-conditional. From preference triples, DSPA computes a conditional-difference map linking prompt features to generation-control features; during decoding, it modifies only token-active latents, without base-model weight updates. Across Gemma-2-2B/9B and Qwen3-8B, DSPA improves MT-Bench and is competitive on AlpacaEval while preserving multiple-choice accuracy. Under restricted preference data, DSPA remains robust and can rival the two-stage RAHF-SCIT pipeline while requiring up to $4.47\times$ fewer alignment-stage FLOPs. Finally, we audit the SAE features DSPA modifies, finding that preference directions are dominated by discourse and stylistic signals, and provide theory clarifying the conditional-difference map estimate and when top-$k$ ablation is principled. 

---
# PLR: Plackett-Luce for Reordering In-Context Learning Examples 

**Authors**: Pawel Batorski, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2603.21373)  

**Abstract**: In-context learning (ICL) adapts large language models by conditioning on a small set of ICL examples, avoiding costly parameter updates. Among other factors, performance is often highly sensitive to the ordering of the examples. However, exhaustive search over the $n!$ possible orderings is infeasible. Therefore more efficient ordering methods use model confidence measures (e.g., label-probability entropy) over label sets or take a direct approach to finding the best ordering. We propose PLR, a probabilistic approach to in-context example ordering that replaces discrete ordering search with learning a probability distribution over orderings with the Plackett-Luce model. PLR models orderings using a Plackett-Luce distribution and iteratively updates its parameters to concentrate probability mass on high-performing orderings under a task-level metric. Candidate orderings are sampled efficiently via a Gumbel perturb-and-sort procedure. Experiments on multiple classification benchmarks show that PLR consistently improves few-shot accuracy for $k \in \{4, 8, 16, 32\}$ examples, and we further demonstrate gains on mathematical reasoning tasks where label-based ordering methods are not applicable. Our code is available at this https URL. 

---
# TIDE: Token-Informed Depth Execution for Per-Token Early Exit in LLM Inference 

**Authors**: Jaber Jaber, Osama Jaber  

**Link**: [PDF](https://arxiv.org/pdf/2603.21365)  

**Abstract**: Large language models run every token through every layer, regardless of difficulty. We present TIDE, a post-training system that attaches tiny learned routers at periodic checkpoint layers and, at inference time, selects the earliest layer whose hidden state has converged for each token. TIDE requires no model retraining, works with any HuggingFace causal LM, auto-detects GPU architecture, and supports float32, float16, and bfloat16 through fused CUDA kernels. On an NVIDIA A100 with DeepSeek R1 Distill 8B, TIDE achieves 100% prefill exit rate (5% of tokens exit at layer 11, the remaining at layer 31), reduces prefill latency by 7.2%, and increases single-batch throughput by 6.6%. During autoregressive decoding, 98-99% of tokens exit early while the model correctly solves a multi-step math problem with 95 unique output tokens. On Qwen3 8B (36 layers), throughput improves by 8.1% at batch size 8. Calibration on 2,000 WikiText samples takes under 3 minutes and produces a ~4 MB router checkpoint. The system comprises 1,308 lines of Python and 1,081 lines of CUDA/C++ with 74 passing tests. Code: this https URL 

---
# AdaRubric: Task-Adaptive Rubrics for LLM Agent Evaluation 

**Authors**: Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.21362)  

**Abstract**: LLM-as-Judge evaluation fails agent tasks because a fixed rubric cannot capture what matters for this task: code debugging demands Correctness and Error Handling; web navigation demands Goal Alignment and Action Efficiency. We present ADARUBRIC, which closes this gap by generating task-specific evaluation rubrics on the fly from task descriptions, scoring trajectories step-by-step with confidence-weighted per-dimension feedback, and filtering preference pairs with the novel DimensionAwareFilter - a provably necessary condition for preventing high-scoring dimensions from masking dimension-level failures. On WebArena and ToolBench, ADARUBRIC achieves Pearson r=0.79 human correlation (+0.16 over the best static baseline) with deployment-grade reliability (Krippendorff's $\alpha$=0.83). DPO agents trained on ADARUBRIC preference pairs gain +6.8 to +8.5 pp task success over Prometheus across three benchmarks; gains transfer to SWE-bench code repair (+4.9 pp) and accelerate PPO convergence by +6.6 pp at 5K steps - both without any rubric engineering. Code: this https URL. 

---
# AgentHER: Hindsight Experience Replay for LLM Agent Trajectory Relabeling 

**Authors**: Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2603.21357)  

**Abstract**: LLM agents fail on the majority of real-world tasks -- GPT-4o succeeds on fewer than 15% of WebArena navigation tasks and below 55% pass@1 on ToolBench (Zhou et al., 2024; Qin et al., 2024) -- yet every failed trajectory is routinely discarded, wasting the dominant source of collected experience. We introduce AgentHER, a framework that recovers this lost training signal by adapting the Hindsight Experience Replay (HER; Andrychowicz et al., 2017) principle to natural-language agent trajectories for offline data augmentation. The key insight is simple: a trajectory that fails goal A is often a correct demonstration for some achievable alternative goal B. AgentHER realises this idea through a four-stage pipeline -- failure classification, outcome extraction, LLM-guided prompt relabeling with confidence gating, and data packaging -- that converts discarded failures into high-quality SFT, DPO, and ShareGPT training data, with both zero-cost rule-based and LLM-judge implementations. On WebArena (Zhou et al., 2024) and ToolBench (Qin et al., 2024), AgentHER improves over success-only SFT by +7.1-11.7 pp across four model families (GPT-4o, Qwen2.5-72B/7B, LLaMA-3.1-8B), while achieving 2x data efficiency -- matching baseline performance with only 50% of successful demonstrations. Gains are consistent from 1.5B to 72B parameters (+5.8-9.2 pp) and compound under iterative redeployment (+2.1 pp over additional rounds). Human evaluation confirms 97.7% relabeling precision under multi-judge verification. 

---
# Generalized Discrete Diffusion from Snapshots 

**Authors**: Oussama Zekri, Théo Uscidda, Nicolas Boullé, Anna Korba  

**Link**: [PDF](https://arxiv.org/pdf/2603.21342)  

**Abstract**: We introduce Generalized Discrete Diffusion from Snapshots (GDDS), a unified framework for discrete diffusion modeling that supports arbitrary noising processes over large discrete state spaces. Our formulation encompasses all existing discrete diffusion approaches, while allowing significantly greater flexibility in the choice of corruption dynamics. The forward noising process relies on uniformization and enables fast arbitrary corruption. For the reverse process, we derive a simple evidence lower bound (ELBO) based on snapshot latents, instead of the entire noising path, that allows efficient training of standard generative modeling architectures with clear probabilistic interpretation. Our experiments on large-vocabulary discrete generation tasks suggest that the proposed framework outperforms existing discrete diffusion methods in terms of training efficiency and generation quality, and beats autoregressive models for the first time at this scale. We provide the code along with a blog post on the project page : \href{this https URL}{this https URL}. 

---
# Improving Coherence and Persistence in Agentic AI for System Optimization 

**Authors**: Pantea Karimi, Kimia Noorbakhsh, Mohammad Alizadeh, Hari Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2603.21321)  

**Abstract**: Designing high-performance system heuristics is a creative, iterative process requiring experts to form hypotheses and execute multi-step conceptual shifts. While Large Language Models (LLMs) show promise in automating this loop, they struggle with complex system problems due to two critical failure modes: evolutionary neighborhood bias and the coherence ceiling. Evolutionary methods often remain trapped in local optima by relying on scalar benchmark scores, failing when coordinated multi-step changes are required. Conversely, existing agentic frameworks suffer from context degradation over long horizons or fail to accumulate knowledge across independent runs.
We present Engram, an agentic researcher architecture that addresses these limitations by decoupling long-horizon exploration from the constraints of a single context window. Engram organizes exploration into a sequence of agents that iteratively design, test, and analyze mechanisms. At the conclusion of each run, an agent stores code snapshots, logs, and results in a persistent Archive and distills high-level modeling insights into a compact, persistent Research Digest. Subsequent agents then begin with a fresh context window, reading the Research Digest to build on prior discoveries.
We find that Engram exhibits superior performance across diverse domains including multi-cloud multicast, LLM inference request routing, and optimizing KV cache reuse in databases with natural language queries. 

---
# The Library Theorem: How External Organization Governs Agentic Reasoning Capacity 

**Authors**: Zachary F. Mainen  

**Link**: [PDF](https://arxiv.org/pdf/2603.21272)  

**Abstract**: Externalized reasoning is already exploited by transformer-based agents through chain-of-thought, but structured retrieval -- indexing over one's own reasoning state -- remains underexplored. We formalize the transformer context window as an I/O page and prove that tool-augmented agents with indexed external memory achieve exponentially lower retrieval cost than agents restricted to sequential scanning: $O(\log_b N)$ versus $\Omega(N)$ page reads per query, and $O(T \log_b T)$ versus $\Theta(T^2)$ cumulative cost over $T$ reasoning steps -- a gap that widens as deliberation deepens. We test these predictions on a controlled lookup benchmark across three content types -- random hashes, ordered integers, and encyclopedia entries -- varying store size from 50 to 5,000 items, and replicate key conditions across two model generations (GPT-4o-mini and GPT-5.4). On abstract content, the indexed agent achieves median 1 page read regardless of store size, confirming the $O(1)$ prediction. Sorted pages without an index fail to close the gap: the weaker model cannot sustain binary search at scale, and the stronger model achieves near-optimal $\log_2 N$ search but still loses to the index by $5\times$. On familiar content (encyclopedia entries), a competing failure mode emerges: the model recognizes the domain, bypasses the retrieval protocol, and generates answers from parametric memory, producing catastrophic token expenditure even when the index is sound. This parametric memory competition dissociates the two cognitive operations that indexing combines: understanding content (where language models excel) and following navigational protocols (where they fail when understanding tempts them to shortcut). The result argues for a separation of concerns: use language models for index construction, where semantic understanding helps, and deterministic algorithms for index traversal, where it hurts. 

---
# Mixture of Chapters: Scaling Learnt Memory in Transformers 

**Authors**: Tasmay Pankaj Tibrewal, Pritish Saha, Ankit Meda, Kunal Singh, Pradeep Moturi  

**Link**: [PDF](https://arxiv.org/pdf/2603.21096)  

**Abstract**: Transformers lack an explicit architectural mechanism for storing and organizing knowledge acquired during training. We introduce learnable sparse memory banks: a set of latent tokens, randomly initialized and trained end-to-end, that transformer layers query via cross-attention to retrieve stored knowledge. To scale memory capacity without prohibitive attention costs, we propose chapter-based routing inspired by Mixture-of-Experts architectures, partitioning the memory bank into chapters and training a router to select relevant subsets per input. This enables scaling to 262K memory tokens while maintaining tractable computation. We evaluate our approach against standard transformers (in iso-FLOP settings) on pre-training and instruction fine-tuning across relevant benchmarks. Our models surpass iso-FLOP baselines suggesting scope for a new axis of scaling, demonstrating that explicit associative memory provides complementary capacity to what is captured implicitly in model parameters. Additionally, we observe improved knowledge retention under continued training, with robustness to forgetting when transitioning between training phases (e.g., pretraining to instruction fine-tuning). 

---
# SqueezeComposer: Temporal Speed-up is A Simple Trick for Long-form Music Composing 

**Authors**: Jianyi Chen, Rongxiu Zhong, Shilei Zhang, Kun Qian, Jinglei Liu, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2603.21073)  

**Abstract**: Composing coherent long-form music remains a significant challenge due to the complexity of modeling long-range dependencies and the prohibitive memory and computational requirements associated with lengthy audio representations. In this work, we propose a simple yet powerful trick: we assume that AI models can understand and generate time-accelerated (speeded-up) audio at rates such as 2x, 4x, or even 8x. By first generating a high-speed version of the music, we greatly reduce the temporal length and resource requirements, making it feasible to handle long-form music that would otherwise exceed memory or computational limits. The generated audio is then restored to its original speed, recovering the full temporal structure. This temporal speed-up and slow-down strategy naturally follows the principle of hierarchical generation from abstract to detailed content, and can be conveniently applied to existing music generation models to enable long-form music generation. We instantiate this idea in SqueezeComposer, a framework that employs diffusion models for generation in the accelerated domain and refinement in the restored domain. We validate the effectiveness of this approach on two tasks: long-form music generation, which evaluates temporal-wise control (including continuation, completion, and generation from scratch), and whole-song singing accompaniment generation, which evaluates track-wise control. Experimental results demonstrate that our simple temporal speed-up trick enables efficient, scalable, and high-quality long-form music generation. Audio samples are available at this https URL. 

---
# LongCat-Flash-Prover: Advancing Native Formal Reasoning via Agentic Tool-Integrated Reinforcement Learning 

**Authors**: Jianing Wang, Jianfei Zhang, Qi Guo, Linsen Guo, Rumei Li, Chao Zhang, Chong Peng, Cunguang Wang, Dengchang Zhao, Jiarong Shi, Jingang Wang, Liulin Feng, Mengxia Shen, Qi Li, Shengnan An, Shun Wang, Wei Shi, Xiangyu Xi, Xiaoyu Li, Xuezhi Cao, Yi Lu, Yunke Zhao, Zhengyu Chen, Zhimin Lin, Wei Wang, Peng Pei, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2603.21065)  

**Abstract**: We introduce LongCat-Flash-Prover, a flagship 560-billion-parameter open-source Mixture-of- Experts (MoE) model that advances Native Formal Reasoning in Lean4 through agentic tool-integrated reasoning (TIR). We decompose the native formal reasoning task into three independent formal capabilities, i.e., auto-formalization, sketching, and proving. To facilitate these capabilities, we propose a Hybrid-Experts Iteration Framework to expand high-quality task trajectories, including generating a formal statement based on a given informal problem, producing a whole-proof directly from the statement, or a lemma-style sketch. During agentic RL, we present a Hierarchical Importance Sampling Policy Optimization (HisPO) algorithm, which aims to stabilize the MoE model training on such long-horizon tasks. It employs a gradient masking strategy that accounts for the policy staleness and the inherent train-inference engine discrepancies at both sequence and token levels. Additionally, we also incorporate theorem consistency and legality detection mechanisms to eliminate reward hacking issues. Extensive evaluations show that our LongCat-Flash-Prover sets a new state-of-the-art for open-weights models in both auto-formalization and theorem proving. Demonstrating remarkable sample efficiency, it achieves a 97.1% pass rate on MiniF2F-Test using only 72 inference budget per problem. On more challenging benchmarks, it solves 70.8% of ProverBench and 41.5% of PutnamBench with no more than 220 attempts per problem, significantly outperforming existing open-weights baselines. 

---
# Knowledge Boundary Discovery for Large Language Models 

**Authors**: Ziquan Wang, Zhongqi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.21022)  

**Abstract**: We propose Knowledge Boundary Discovery (KBD), a reinforcement learning based framework to explore the knowledge boundaries of the Large Language Models (LLMs). We define the knowledge boundary by automatically generating two types of questions: (i) those the LLM can confidently answer (within-knowledge boundary) and (ii) those it cannot (beyond-knowledge boundary). Iteratively exploring and exploiting the LLM's responses to find its knowledge boundaries is challenging because of the hallucination phenomenon. To find the knowledge boundaries of an LLM, the agent interacts with the LLM under the modeling of exploring a partially observable environment. The agent generates a progressive question as the action, adopts an entropy reduction as the reward, receives the LLM's response as the observation and updates its belief states. We demonstrate that the KBD detects knowledge boundaries of LLMs by automatically finding a set of non-trivial answerable and unanswerable questions. We validate the KBD by comparing its generated knowledge boundaries with manually crafted LLM benchmark datasets. Experiments show that our KBD-generated question set is comparable to the human-generated datasets. Our approach paves a new way to evaluate LLMs. 

---
# CLT-Forge: A Scalable Library for Cross-Layer Transcoders and Attribution Graphs 

**Authors**: Florent Draye, Abir Harrasse, Vedant Palit, Tung-Yu Wu, Jiarui Liu, Punya Syon Pandey, Roderick Wu, Terry Jingchen Zhang, Zhijing Jin, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2603.21014)  

**Abstract**: Mechanistic interpretability seeks to understand how Large Language Models (LLMs) represent and process information. Recent approaches based on dictionary learning and transcoders enable representing model computation in terms of sparse, interpretable features and their interactions, giving rise to feature attribution graphs. However, these graphs are often large and redundant, limiting their interpretability in practice. Cross-Layer Transcoders (CLTs) address this issue by sharing features across layers while preserving layer-specific decoding, yielding more compact representations, but remain difficult to train and analyze at scale. We introduce an open-source library for end-to-end training and interpretability of CLTs. Our framework integrates scalable distributed training with model sharding and compressed activation caching, a unified automated interpretability pipeline for feature analysis and explanation, attribution graph computation using Circuit-Tracer, and a flexible visualization interface. This provides a practical and unified solution for scaling CLT-based mechanistic interpretability. Our code is available at: this https URL. 

---
# How AI Systems Think About Education: Analyzing Latent Preference Patterns in Large Language Models 

**Authors**: Daniel Autenrieth  

**Link**: [PDF](https://arxiv.org/pdf/2603.21006)  

**Abstract**: This paper presents the first systematic measurement of educational alignment in Large Language Models. Using a Delphi-validated instrument comprising 48 items across eight educational-theoretical dimensions, the study reveals that GPT-5.1 exhibits highly coherent preference patterns (99.78% transitivity; 92.79% model accuracy) that largely align with humanistic educational principles where expert consensus exists. Crucially, divergences from expert opinion occur precisely in domains of normative disagreement among human experts themselves, particularly emotional dimensions and epistemic normativity. This raises a fundamental question for alignment research: When human values are contested, what should models be aligned to? The findings demonstrate that GPT-5.1 does not remain neutral in contested domains but adopts coherent positions, prioritizing emotional responsiveness and rejecting false balance. The methodology, combining Delphi consensus-building with Structured Preference Elicitation and Thurstonian Utility modeling, provides a replicable framework for domain-specific alignment evaluation beyond generic value benchmarks. 

---
# Structural Sensitivity in Compressed Transformers: Error Propagation, Lyapunov Stability, and Formally Verified Bounds 

**Authors**: Abhinaba Basu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20991)  

**Abstract**: A single matrix out of 468 in GPT-2 Small can increase perplexity by 20,000x when compressed, revealing that transformer compression sensitivity spans five orders of magnitude. We map this sensitivity landscape across five architectures (117M-8B parameters), finding a consistent hierarchy: early-layer MLP up-projections are catastrophically sensitive while value projections compress nearly for free. This hierarchy is stable across compression levels, evaluation scales (2K-51K tokens), and datasets (WikiText-103, C4). Using Lyapunov stability theory, we show that residual connections contract compression errors by growing the hidden state faster than the error. Error contraction is necessary but not sufficient for compression tolerance: architecture-specific redundancy plays an equally important role, as demonstrated by the hybrid LFM2-2.6B degrading only 7x despite higher amplification than the fully-contracting GPT-2 Small (120x). Ten machine-checked Lean 4 theorems formalize per-matrix error bounds with no sorry markers; all bounds produce zero violations across 14,040+ configurations. We validate with downstream task evaluation (HellaSwag, ARC-Easy, Winogrande), activation-aware pruning on two architectures, and a Compression Fragility Index that rank-orders model robustness. 

---
# Understanding Contextual Recall in Transformers: How Finetuning Enables In-Context Reasoning over Pretraining Knowledge 

**Authors**: Bhavya Vasudeva, Puneesh Deora, Alberto Bietti, Vatsal Sharan, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2603.20969)  

**Abstract**: Transformer-based language models excel at in-context learning (ICL), where they can adapt to new tasks based on contextual examples, without parameter updates. In a specific form of ICL, which we refer to as \textit{contextual recall}, models pretrained on open-ended text leverage pairwise examples to recall specific facts in novel prompt formats. We investigate whether contextual recall emerges from pretraining alone, what finetuning is required, and what mechanisms drive the necessary representations. For this, we introduce a controlled synthetic framework where pretraining sequences consist of subject-grammar-attribute tuples, with attribute types tied to grammar statistics. We demonstrate that while such pretraining successfully yields factual knowledge, it is insufficient for contextual recall: models fail to implicitly infer attribute types when the grammar statistics are removed in ICL prompts. However, we show that finetuning on tasks requiring implicit inference, distinct from the ICL evaluation, using a subset of subjects, triggers the emergence of contextual recall across all subjects. This transition is accompanied by the formation of low-dimensional latent encodings of the shared attribute type. For mechanistic insight, we derive a construction for an attention-only transformer that replicates the transition from factual to contextual recall, corroborated by empirical validation. 

---
# RubricRAG: Towards Interpretable and Reliable LLM Evaluation via Domain Knowledge Retrieval for Rubric Generation 

**Authors**: Kaustubh D. Dhole, Eugene Agichtein  

**Link**: [PDF](https://arxiv.org/pdf/2603.20882)  

**Abstract**: Large language models (LLMs) are increasingly evaluated and sometimes trained using automated graders such as LLM-as-judges that output scalar scores or preferences. While convenient, these approaches are often opaque: a single score rarely explains why an answer is good or bad, which requirements were missed, or how a system should be improved. This lack of interpretability limits their usefulness for model development, dataset curation, and high-stakes deployment. Query-specific rubric-based evaluation offers a more transparent alternative by decomposing quality into explicit, checkable criteria. However, manually designing high-quality, query-specific rubrics is labor-intensive and cognitively demanding and not feasible for deployment. While previous approaches have focused on generating intermediate rubrics for automated downstream evaluation, it is unclear if these rubrics are both interpretable and effective for human users. In this work, we investigate whether LLMs can generate useful, instance-specific rubrics as compared to human-authored rubrics, while also improving effectiveness for identifying good responses. Through our systematic study on two rubric benchmarks, and on multiple few-shot and post-training strategies, we find that off-the-shelf LLMs produce rubrics that are poorly aligned with human-authored ones. We introduce a simple strategy, RubricRAG, which retrieves domain knowledge via rubrics at inference time from related queries. We demonstrate that RubricRAG can generate more interpretable rubrics both for similarity to human-authored rubrics, and for improved downstream evaluation effectiveness. Our results highlight both the challenges and a promising approach of scalable, interpretable evaluation through automated rubric generation. 

---
# Semantic Sections: An Atlas-Native Feature Ontology for Obstructed Representation Spaces 

**Authors**: Hossein Javidnia  

**Link**: [PDF](https://arxiv.org/pdf/2603.20867)  

**Abstract**: Recent interpretability work often treats a feature as a single global direction, dictionary atom, or latent coordinate shared across contexts. We argue that this ontology can fail in obstructed representation spaces, where locally coherent meanings need not assemble into one globally consistent feature. We introduce an atlas-native replacement object, the semantic section: a transport-compatible family of local feature representatives defined over a context atlas. We formalize semantic sections, prove that tree-supported propagation is always pathwise realizable, and show that cycle consistency is the key criterion for genuine globalization. This yields a distinction between tree-local, globalizable, and twisted sections, with twisted sections capturing locally coherent but holonomy-obstructed meanings. We then develop a discovery-and-certification pipeline based on seeded propagation, synchronization across overlaps, defect-based pruning, cycle-aware taxonomy, and deduplication. Across layer-16 atlases for Llama 3.2 3B Instruct, Qwen 2.5 3B Instruct, and Gemma 2 2B IT, we find nontrivial populations of semantic sections, including cycle-supported globalizable and twisted regimes after deduplication. Most importantly, semantic identity is not recovered by raw global-vector similarity. Even certified globalizable sections show low cross-chart signed cosine similarity, and raw similarity baselines recover only a small fraction of true within-section pairs, often collapsing at moderate thresholds. By contrast, section-based identity recovery is perfect on certified supports. These results support semantic sections as a better feature ontology in obstructed regimes. 

---
# NDT: Non-Differential Transformer and Its Application to Sentiment Analysis 

**Authors**: Soudeep Ghoshal, Himanshu Buckchash, Sarita Paudel, Rubén Ruiz-Torrubiano  

**Link**: [PDF](https://arxiv.org/pdf/2603.20704)  

**Abstract**: From customer feedback to social media, understanding human sentiment in text is central to how machines can interact meaningfully with people. However, despite notable progress, accurately capturing sentiment remains a challenging task, which continues to motivate further research in this area. To this end, we introduce Non-Differential Transformer (NDT). It is inspired by (but in contrast to) the state-of-the-art Differential Transformer (DT) model. While standard Transformers can struggle with irrelevant context, the sota DT model uses attention map subtraction, potentially for noise cancellation. We explore an alternative motivation, hypothesizing that benefits may arise from enabling different attention components to specialize on distinct concepts within the text, similar to multiplexing information channels or mixture models, rather than primarily canceling noise via subtraction. Guided by this concept-multiplexing (ConPlex) view, the specific architecture presented in this paper employs a purely additive strategy. It uses only positive weights, learned during training, to ensure constructive combination of these specialized attention perspectives. This design choice explores positive only integration, though our broader framework also shows promise with less constrained linear combinations involving both positive and negative weights. Our model computes attention via this positively weighted sum of multiple distinct attention maps. This allows the model to constructively integrate diverse signals and potentially capture more complex contextual relationships. Competitive performance is achieved by the proposed model for Sentiment Analysis while tested on multiple datasets. We conclude by presenting our results, challenges and future research agenda in this important area of research. 

---
# Clinical Cognition Alignment for Gastrointestinal Diagnosis with Multimodal LLMs 

**Authors**: Huan Zheng, Yucheng Zhou, Tianyi Yan, Dubing Chen, Hongbo Lu, Wenlong Liao, Tao He, Pai Peng, Jianbing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.20698)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated remarkable potential in medical image analysis. However, their application in gastrointestinal endoscopy is currently hindered by two critical limitations: the misalignment between general model reasoning and standardized clinical cognitive pathways, and the lack of causal association between visual features and diagnostic outcomes. In this paper, we propose a novel Clinical-Cognitive-Aligned (CogAlign) framework to address these challenges. First, we endow the model with rigorous clinical analytical capabilities by constructing the hierarchical clinical cognition dataset and employing Supervised Fine-Tuning (SFT). Unlike conventional approaches, this strategy internalizes the hierarchical diagnostic logic of experts, ranging from anatomical localization and morphological evaluation to microvascular analysis, directly into the model. Second, to eliminate visual bias, we provide a theoretical analysis demonstrating that standard supervised tuning inevitably converges to spurious background correlations. Guided by this insight, we propose a counterfactual-driven reinforcement learning strategy to enforce causal rectification. By generating counterfactual normal samples via lesion masking and optimizing through clinical-cognition-centric rewards, we constrain the model to strictly ground its diagnosis in causal lesion features. Extensive experiments demonstrate that our approach achieves State-of-the-Art (SoTA) performance across multiple benchmarks, significantly enhancing diagnostic accuracy in complex clinical scenarios. All source code and datasets will be made publicly available. 

---
# Revenue-Sharing as Infrastructure: A Distributed Business Model for Generative AI Platforms 

**Authors**: Ghislain Dorian Tchuente Mondjo  

**Link**: [PDF](https://arxiv.org/pdf/2603.20533)  

**Abstract**: Generative AI platforms (Google AI Studio, OpenAI, Anthropic) provide infrastructures (APIs, models) that are transforming the application development ecosystem. Recent literature distinguishes three generations of business models: a first generation modeled on cloud computing (pay-per-use), a second characterized by diversification (freemium, subscriptions), and a third, emerging generation exploring multi-layer market architectures with revenue-sharing mechanisms. Despite these advances, current models impose a financial barrier to entry for developers, limiting innovation and excluding actors from emerging economies. This paper proposes and analyzes an original model, "Revenue-Sharing as Infrastructure" (RSI), where the platform offers its AI infrastructure for free and takes a percentage of the revenues generated by developers applications. This model reverses the traditional upstream payment logic and mobilizes concepts of value co-creation, incentive mechanisms, and multi-layer market architecture to build an original theoretical framework. A detailed comparative analysis shows that the RSI model lowers entry barriers for developers, aligns stakeholder interests, and could stimulate innovation in the ecosystem. Beyond its economic relevance, RSI has a major societal dimension: by enabling developers without initial capital to participate in the digital economy, it could unlock the "latent jobs dividend" in low-income countries, where mobile penetration reaches 84%, and help address local challenges in health, agriculture, and services. Finally, we discuss the conditions of feasibility and strategic implications for platforms and developers. 

---
# Epistemic Observability in Language Models 

**Authors**: Tony Mason  

**Link**: [PDF](https://arxiv.org/pdf/2603.20531)  

**Abstract**: We find that models report highest confidence precisely when they are fabricating. Across four model families (OLMo-3, Llama-3.1, Qwen3, Mistral), self-reported confidence inversely correlates with accuracy, with AUC ranging from 0.28 to 0.36 where 0.5 is random guessing.
We prove, under explicit formal assumptions, that this is not a capability gap but an observational one. Under text-only observation, where a supervisor sees only the model's output text, no monitoring system can reliably distinguish honest model outputs from plausible fabrications. We prove two results: first, that any policy conditioning only on the query cannot satisfy epistemic honesty across ambiguous world states; second, that no learning algorithm optimizing reward from a text-only supervisor can converge to honest behavior when the supervisor's observations are identical for both grounded and fabricated responses. Within our formal model, these impossibilities hold regardless of model scale or training procedure, including RLHF and instruction tuning.
We construct a tensor interface that escapes the impossibility by exporting computational byproducts (per-token entropy and log-probability distributions) that are structurally coupled to correctness under standard training. Per-token entropy achieves pooled AUC 0.757, outperforming all text baselines by 2.5--3.9 percentage points at every budget level tested (10\%, 20\%, 30\%). The entropy signal generalizes across architectures (Spearman $\rho = 0.762$).
The core contribution is a cost surface where the empirical mapping from verification budget (fraction of queries receiving expensive checks) to detection accuracy for each judge strategy is a practical lookup for system builders deciding how to allocate verification resources. The contribution is the map. The territory is the system you are building. 

---
# Measuring Reasoning Trace Legibility: Can Those Who Understand Teach? 

**Authors**: Dani Roytburg, Shreya Sridhar, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2603.20508)  

**Abstract**: Language models are increasingly being trained to "reason" before answering users' queries, outputting hundreds or even thousands of tokens worth of deliberation before their final answer. While the main intention of reasoning is to improve models' ability to arrive at a correct answer, we argue that these models should be assessed for the legibility of their reasoning traces in addition to the correctness of their final answers. In this paper, we evaluate 90k traces from 12 Reasoning Language Models (RLMs) for the quality of their reasoning traces. We introduce the concept of transfer utility, which assesses how useful an RLM's reasoning traces are for guiding a weaker, non-reasoning model toward arriving at the correct answer. We find that the reasoning traces of the highest-performing models rank among the lowest for legibility. Furthermore, we uncover tensions between efficiency-based measurements of legibility (such as trace length) and transfer utility. These tensions establish a legibility Pareto frontier, and we demonstrate that an RLM's ability to output highly legible traces can be a task- and audience-dependent goal. Crucially, we find that reward models used to train RLMs do not intrinsically reward legibility. Together, these metrics and the findings they surface chart a path towards scaffolding reasoning traces for a multi-agent future. 

---
# AE-LLM: Adaptive Efficiency Optimization for Large Language Models 

**Authors**: Kaito Tanaka, Masato Ito, Yuji Nishimura, Keisuke Matsuda, Aya Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2603.20492)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across diverse applications, yet their deployment remains challenging due to substantial computational costs, memory requirements, and energy consumption. Recent empirical studies have demonstrated that no single efficiency technique is universally optimal; instead, the effectiveness of methods such as efficient attention mechanisms, mixture-of-experts (MoE), parameter-efficient fine-tuning, and quantization varies significantly depending on task characteristics, resource constraints, and model scales. Building upon these insights, we propose AE-LLM, a unified framework that automatically selects and combines optimal efficiency techniques tailored to specific deployment scenarios. Our approach introduces a multi-objective optimization framework that jointly considers accuracy, latency, memory footprint, and energy consumption, while accounting for hardware constraints and task requirements. We develop an efficient search algorithm that explores the combinatorial space of efficiency techniques across architecture, fine-tuning, and inference stages, identifying Pareto-optimal configurations. Extensive experiments across 15 models (0.5B-70B parameters) and 10 diverse tasks demonstrate that AE-LLM achieves an average of $2.8\times$ improvement in efficiency metrics while maintaining competitive accuracy (within 1.2\% of baseline), compared to static efficiency configurations. Furthermore, our framework generalizes effectively to vision-language models, achieving similar efficiency gains. Our contributions provide practitioners with an automated tool for navigating the complex trade-off landscape of LLM efficiency optimization. 

---
# Profiling learners' affective engagement: Emotion AI, intercultural pragmatics, and language learning 

**Authors**: Robert Godwin-Jones  

**Link**: [PDF](https://arxiv.org/pdf/2603.20479)  

**Abstract**: Learning another language can be a highly emotional process, typically characterized by numerous frustrations and triumphs, big and small. For most learners, language learning does not follow a linear, predictable path, its zigzag course shaped by motivational (or demotivating) variables such as personal characteristics, teacher/peer relationships, learning materials, and dreams of a future L2 (second language) self. While some aspects of language learning (reading, grammar) are relatively mechanical, others can be stressful and unpredictable, especially conversing in the target language. That experience necessitates not only knowledge of structure and lexis, but also the ability to use the language in ways that are appropriate to the social and cultural context. A new opportunity to practice conversational abilities has arrived through the availability of AI chatbots, with both advantages (responsive, non-judgmental) and drawbacks (emotionally void, culturally biased). This column explores aspects of emotion as they arise in technology use and in particular how automatic emotion recognition and simulated human responsiveness in AI systems interface with language learning and the development of pragmatic and interactional competence. Emotion AI, the algorithmically driven interpretation of users' affective signals, has been seen as enabling greater personalized learning, adapting to perceived learner cognitive and emotional states. Others warn of emotional manipulation and inappropriate and ineffective user profiling 

---
# ALICE: A Multifaceted Evaluation Framework of Large Audio-Language Models' In-Context Learning Ability 

**Authors**: Yen-Ting Piao, Jay Chiehen Liao, Wei-Tang Chien, Toshiki Ogimoto, Shang-Tse Chen, Yun-Nung Chen, Chun-Yi Lee, Shao-Yuan Lo  

**Link**: [PDF](https://arxiv.org/pdf/2603.20433)  

**Abstract**: While Large Audio-Language Models (LALMs) have been shown to exhibit degraded instruction-following capabilities, their ability to infer task patterns from in-context examples under audio conditioning remains unstudied. To address this gap, we present ALICE, a three-stage framework that progressively reduces textual guidance to systematically evaluate LALMs' in-context learning ability under audio conditioning. Evaluating six LALMs across four audio understanding tasks under two output constraint categories, we uncover a consistent asymmetry across all stages and LALMs: in-context demonstrations reliably improve format compliance but fail to improve, and often degrade, the core task performance. This suggests that LALMs can glean surface-level formatting patterns from demonstrations but may struggle to leverage cross-modal semantic grounding to reliably infer task objectives from audio-conditioned examples, highlighting potential limitations in current cross-modal integration. 

---
# Putnam 2025 Problems in Rocq using Opus 4.6 and Rocq-MCP 

**Authors**: Guillaume Baudart, Marc Lelarge, Tristan Stérin, Jules Viennot  

**Link**: [PDF](https://arxiv.org/pdf/2603.20405)  

**Abstract**: We report on an experiment in which Claude Opus~4.6, equipped with a suite of Model Context Protocol (MCP) tools for the Rocq proof assistant, autonomously proved 10 of 12 problems from the 2025 Putnam Mathematical Competition. The MCP tools, designed with Claude by analyzing logs from a prior experiment on miniF2F-Rocq, encode a "compile-first, interactive-fallback" strategy. Running on an isolated VM with no internet access, the agent deployed 141 subagents over 17.7 hours of active compute (51.6h wall-clock), consuming approximately 1.9 billion tokens. All proofs are publicly available. 

---
# GIP-RAG: An Evidence-Grounded Retrieval-Augmented Framework for Interpretable Gene Interaction and Pathway Impact Analysis 

**Authors**: Fujian Jia, Jiwen Gu, Cheng Lu, Dezhi Zhao, Mengjiang Huang, Yuanzhi Lu, Xin Liu, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.20321)  

**Abstract**: Understanding mechanistic relationships among genes and their impacts on biological pathways is essential for elucidating disease mechanisms and advancing precision medicine. Despite the availability of extensive molecular interaction and pathway data in public databases, integrating heterogeneous knowledge sources and enabling interpretable multi-step reasoning across biological networks remain challenging.
We present GIP-RAG (Gene Interaction Prediction through Retrieval-Augmented Generation), a computational framework that combines biomedical knowledge graphs with large language models (LLMs) to infer and interpret gene interactions. The framework constructs a unified gene interaction knowledge graph by integrating curated data from KEGG, WikiPathways, SIGNOR, Pathway Commons, and PubChem. Given user-specified genes, a query-driven module retrieves relevant subgraphs, which are incorporated into structured prompts to guide LLM-based stepwise reasoning. This enables identification of direct and indirect regulatory relationships and generation of mechanistic explanations supported by biological evidence.
Beyond pairwise interactions, GIP-RAG includes a pathway-level functional impact module that simulates propagation of gene perturbations through signaling networks and evaluates potential pathway state changes. Evaluation across diverse biological scenarios demonstrates that the framework generates consistent, interpretable, and evidence-supported insights into gene regulatory mechanisms.
Overall, GIP-RAG provides a general and interpretable approach for integrating knowledge graphs with retrieval-augmented LLMs to support mechanistic reasoning in complex molecular systems. 

---
# kRAIG: A Natural Language-Driven Agent for Automated DataOps Pipeline Generation 

**Authors**: Rohan Siva, Kai Cheung, Lichi Li, Ganesh Sundaram  

**Link**: [PDF](https://arxiv.org/pdf/2603.20311)  

**Abstract**: Modern machine learning systems rely on complex data engineering workflows to extract, transform, and load (ELT) data into production pipelines. However, constructing these pipelines remains time-consuming and requires substantial expertise in data infrastructure and orchestration frameworks. Recent advances in large language model (LLM) agents offer a potential path toward automating these workflows, but existing approaches struggle with under-specified user intent, unreliable tool generation, and limited guarantees of executable outputs.
We introduce kRAIG, an AI agent that translates natural language specifications into production-ready Kubeflow Pipelines (KFP). To resolve ambiguity in user intent, we propose ReQuesAct (Reason, Question, Act), an interaction framework that explicitly clarifies intent prior to pipeline synthesis. The system orchestrates end-to-end data movement from diverse sources and generates task-specific transformation components through a retrieval-augmented tool synthesis process. To ensure data quality and safety, kRAIG incorporates LLM-based validation stages that verify pipeline integrity prior to execution.
Our framework achieves a 3x improvement in extraction and loading success and a 25 percent increase in transformation accuracy compared to state-of-the-art agentic baselines. These improvements demonstrate that structured agent workflows with explicit intent clarification and validation significantly enhance the reliability and executability of automated data engineering pipelines. 

---
# OpenResearcher: A Fully Open Pipeline for Long-Horizon Deep Research Trajectory Synthesis 

**Authors**: Zhuofeng Li, Dongfu Jiang, Xueguang Ma, Haoxiang Zhang, Ping Nie, Yuyu Zhang, Kai Zou, Jianwen Xie, Yu Zhang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.20278)  

**Abstract**: Training deep research agents requires long-horizon trajectories that interleave search, evidence aggregation, and multi-step reasoning. However, existing data collection pipelines typically rely on proprietary web APIs, making large-scale trajectory synthesis costly, unstable, and difficult to reproduce. We present OpenResearcher, a reproducible pipeline that decouples one-time corpus bootstrapping from multi-turn trajectory synthesis and executes the search-and-browse loop entirely offline using three explicit browser primitives: search, open, and find, over a 15M-document corpus. Using GPT-OSS-120B as the teacher model, we synthesize over 97K trajectories, including a substantial long-horizon tail with 100+ tool calls. Supervised fine-tuning a 30B-A3B backbone on these trajectories achieves 54.8\% accuracy on BrowseComp-Plus, a +34.0 point improvement over the base model, while remaining competitive on BrowseComp, GAIA, and xbench-DeepSearch. Because the environment is offline and fully instrumented, it also enables controlled analysis, where our study reveals practical insights into deep research pipeline design, including data filtering strategies, agent configuration choices, and how retrieval success relates to final answer accuracy. We release the pipeline, synthesized trajectories, model checkpoints, and the offline search environment at this https URL. 

---
# Email in the Era of LLMs 

**Authors**: Dang Nguyen, Harvey Yiyun Fu, Peter West, Chenhao Tan, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2603.20231)  

**Abstract**: Email communication increasingly involves large language models (LLMs), but we lack intuition on how they will read, write, and optimize for nuanced social goals. We introduce HR Simulator, a game where communication is the core mechanic: players play as a Human Resources officer and write emails to solve socially challenging workplace scenarios. An analysis of 600+ human and LLM emails with LLMs-as-judge reveals evidence for larger LLMs becoming more homogenous in their email quality judgments. Under LLM judges, humans underperform LLMs (e.g., 23.5% vs. 48-54% success rate), but a human+LLM approach can outperform LLM-only (e.g., from 40% to nearly 100% in one scenario). In cases where models' email preferences disagree, emergent tact is a plausible explanation: weaker models prefer less tactful strategies while stronger models prefer more tactful ones. Regarding tone, LLM emails are more formal and empathetic while human emails are more varied. LLM rewrites make human emails more formal and empathetic, but models still struggle to imitate human emails in the low empathy, low formality quadrant, which highlights a limitation of current post-training approaches. Our results demonstrate the efficacy of communication games as instruments to measure communication in the era of LLMs, and posit human-LLM co-writing as an effective form of communication in that future. 

---
# AgenticGEO: A Self-Evolving Agentic System for Generative Engine Optimization 

**Authors**: Jiaqi Yuan, Jialu Wang, Zihan Wang, Qingyun Sun, Ruijie Wang, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.20213)  

**Abstract**: Generative search engines represent a transition from traditional ranking-based retrieval to Large Language Model (LLM)-based synthesis, transforming optimization goals from ranking prominence towards content inclusion. Generative Engine Optimization (GEO), specifically, aims to maximize visibility and attribution in black-box summarized outputs by strategically manipulating source content. However, existing methods rely on static heuristics, single-prompt optimization, or engine preference rule distillation that is prone to overfitting. They cannot flexibly adapt to diverse content or the changing behaviors of generative engines. Moreover, effectively optimizing these strategies requires an impractical amount of interaction feedback from the engines. To address these challenges, we propose AgenticGEO, a self-evolving agentic framework formulating optimization as a content-conditioned control problem, which enhances intrinsic content quality to robustly adapt to the unpredictable behaviors of black-box engines. Unlike fixed-strategy methods, AgenticGEO employs a MAP-Elites archive to evolve diverse, compositional strategies. To mitigate interaction costs, we introduce a Co-Evolving Critic, a lightweight surrogate that approximates engine feedback for content-specific strategy selection and refinement, efficiently guiding both evolutionary search and inference-time planning. Through extensive in-domain and cross-domain experiments on two representative engines, AgenticGEO achieves state-of-the-art performance and demonstrates robust transferability, outperforming 14 baselines across 3 datasets. Our code and model are available at: this https URL. 

---
