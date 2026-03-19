# ConGA: Guidelines for Contextual Gender Annotation. A Framework for Annotating Gender in Machine Translation 

**Authors**: Argentina Anna Rescigno, Eva Vanmassenhove, Johanna Monti  

**Link**: [PDF](https://arxiv.org/pdf/2603.17962)  

**Abstract**: Handling gender across languages remains a persistent challenge for Machine Translation (MT) and Large Language Models (LLMs), especially when translating from gender-neutral languages into morphologically gendered ones, such as English to Italian. English largely omits grammatical gender, while Italian requires explicit agreement across multiple grammatical categories. This asymmetry often leads MT systems to default to masculine forms, reinforcing bias and reducing translation accuracy. To address this issue, we present the Contextual Gender Annotation (ConGA) framework, a linguistically grounded set of guidelines for word-level gender annotation. The scheme distinguishes between semantic gender in English through three tags, Masculine (M), Feminine (F), and Ambiguous (A), and grammatical gender realisation in Italian (Masculine (M), Feminine (F)), combined with entity-level identifiers for cross-sentence tracking. We apply ConGA to the gENder-IT dataset, creating a gold-standard resource for evaluating gender bias in translation. Our results reveal systematic masculine overuse and inconsistent feminine realisation, highlighting persistent limitations of current MT systems. By combining fine-grained linguistic annotation with quantitative evaluation, this work offers both a methodology and a benchmark for building more gender-aware and multilingual NLP systems. 

---
# Gender Disambiguation in Machine Translation: Diagnostic Evaluation in Decoder-Only Architectures 

**Authors**: Chiara Manna, Hosein Mohebbi, Afra Alishahi, Frédéric Blain, Eva Vanmassenhove  

**Link**: [PDF](https://arxiv.org/pdf/2603.17952)  

**Abstract**: While Large Language Models achieve state-of-the-art results across a wide range of NLP tasks, they remain prone to systematic biases. Among these, gender bias is particularly salient in MT, due to systematic differences across languages in whether and how gender is marked. As a result, translation often requires disambiguating implicit source signals into explicit gender-marked forms. In this context, standard benchmarks may capture broad disparities but fail to reflect the full complexity of gender bias in modern MT. In this paper, we extend recent frameworks on bias evaluation by: (i) introducing a novel measure coined "Prior Bias", capturing a model's default gender assumptions, and (ii) applying the framework to decoder-only MT models. Our results show that, despite their scale and state-of-the-art status, decoder-only models do not generally outperform encoder-decoder architectures on gender-specific metrics; however, post-training (e.g., instruction tuning) not only improves contextual awareness but also reduces the masculine Prior Bias. 

---
# ShapleyLaw: A Game-Theoretic Approach to Multilingual Scaling Laws 

**Authors**: Xuyang Cao, Qianying Liu, Chuan Xiao, Yusuke Oda, Pontus Stenetorp, Daisuke Kawahara, Makoto Onizuka, Sadao Kurohashi, Shuyuan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17945)  

**Abstract**: In multilingual pretraining, the test loss of a pretrained model is heavily influenced by the proportion of each language in the pretraining data, namely the \textit{language mixture ratios}. Multilingual scaling laws can predict the test loss under different language mixture ratios and can therefore be used to estimate the optimal ratios. However, the current approaches to multilingual scaling laws do not measure the \textit{cross-lingual transfer} effect, resulting in suboptimal mixture ratios. In this paper, we consider multilingual pretraining as a cooperative game in which each language acts as a player that jointly contributes to pretraining, gaining the resulting reduction in test loss as the payoff. Consequently, from the perspective of cooperative game theory, we quantify the cross-lingual transfer from each language by its contribution in the game, and propose a game-theoretic multilingual scaling law called \textit{ShapleyLaw}. Our experiments show that ShapleyLaw outperforms baseline methods in model performance prediction and language mixture optimization. 

---
# Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing 

**Authors**: Raghavv Goel, Mukul Gagrani, Mingu Lee, Chris Lott  

**Link**: [PDF](https://arxiv.org/pdf/2603.17942)  

**Abstract**: Large language models (LLMs) exhibit latent multi-token prediction (MTP) capabilities despite being trained solely for next-token generation. We propose a simple, training-free MTP approach that probes an LLM using on-the-fly mask tokens drawn from its embedding space, enabling parallel prediction of future tokens without modifying model weights or relying on auxiliary draft models. Our method constructs a speculative token tree by sampling top-K candidates from mask-token logits and applies a lightweight pruning strategy to retain high-probability continuations. During decoding, candidate predictions are verified in parallel, resulting in lossless generation while substantially reducing the number of model calls and improving token throughput. Across benchmarks, our probing-based MTP consistently outperforms existing training-free baselines, increasing acceptance length by approximately 12\% on LLaMA3 and 8--12\% on Qwen3, and achieving throughput gains of up to 15--19\%. Finally, we provide theoretical insights and empirical evidence showing that decoder layers naturally align mask-token representations with next-token states, enabling accurate multi-step prediction without retraining or auxiliary models. 

---
# IndicSafe: A Benchmark for Evaluating Multilingual LLM Safety in South Asia 

**Authors**: Priyaranjan Pattnayak, Sanchari Chowdhuri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17915)  

**Abstract**: As large language models (LLMs) are deployed in multilingual settings, their safety behavior in culturally diverse, low-resource languages remains poorly understood. We present the first systematic evaluation of LLM safety across 12 Indic languages, spoken by over 1.2 billion people but underrepresented in LLM training data. Using a dataset of 6,000 culturally grounded prompts spanning caste, religion, gender, health, and politics, we assess 10 leading LLMs on translated variants of the prompt.
Our analysis reveals significant safety drift: cross-language agreement is just 12.8\%, and \texttt{SAFE} rate variance exceeds 17\% across languages. Some models over-refuse benign prompts in low-resource scripts, overflag politically sensitive topics, while others fail to flag unsafe generations. We quantify these failures using prompt-level entropy, category bias scores, and multilingual consistency indices.
Our findings highlight critical safety generalization gaps in multilingual LLMs and show that safety alignment does not transfer evenly across languages. We release \textsc{IndicSafe}, the first benchmark to enable culturally informed safety evaluation for Indic deployments, and advocate for language-aware alignment strategies grounded in regional harms. 

---
# Pretrained Multilingual Transformers Reveal Quantitative Distance Between Human Languages 

**Authors**: Yue Zhao, Jiatao Gu, Paloma Jeretič, Weijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.17912)  

**Abstract**: Understanding the distance between human languages is central to linguistics, anthropology, and tracing human evolutionary history. Yet, while linguistics has long provided rich qualitative accounts of cross-linguistic variation, a unified and scalable quantitative approach to measuring language distance remains lacking. In this paper, we introduce a method that leverages pretrained multilingual language models as systematic instruments for linguistic measurement. Specifically, we show that the spontaneously emerged attention mechanisms of these models provide a robust, tokenization-agnostic measure of cross-linguistic distance, termed Attention Transport Distance (ATD). By treating attention matrices as probability distributions and measuring their geometric divergence via optimal transport, we quantify the representational distance between languages during translation. Applying ATD to a large and diverse set of languages, we demonstrate that the resulting distances recover established linguistic groupings with high fidelity and reveal patterns aligned with geographic and contact-induced relationships. Furthermore, incorporating ATD as a regularizer improves transfer performance in low-resource machine translation. Our results establish a principled foundation for testing linguistic hypotheses using artificial neural networks. This framework transforms multilingual models into powerful tools for quantitative linguistic discovery, facilitating more equitable multilingual AI. 

---
# DebugLM: Learning Traceable Training Data Provenance for LLMs 

**Authors**: Wenjie Jacky Mo, Qin Liu, Xiaofei Wen, Wenxuan Zhou, Zhe Zhao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17884)  

**Abstract**: Large language models (LLMs) are trained through multi-stage pipelines over heterogeneous data sources, yet developers lack a principled way to pinpoint the specific data responsible for an observed behavior. This lack of observability reduces debugging to reactive patching and makes failures prone to recur under distribution shift or subsequent model updates. To address this limitation, we propose DebugLM, a framework that equips LLMs with built-in data provenance, enabling them to explicitly trace the origins of their behaviors to specific training data sources. Specifically, the model learns to associate its responses with unique provenance tags that indicate the responsible dataset, empowering developers to precisely identify where undesirable behaviors are learned. Building on this capability, DebugLM further supports targeted test-time remediation, enabling developers to selectively trigger targeted refusal for specified data sources without retraining or modifying model parameters. Experiments demonstrate that DebugLM provides accurate behavior tracing in multi-stage training pipelines and effective test-time remediation while preserving the general utility of the model. 

---
# Mitigating LLM Hallucinations through Domain-Grounded Tiered Retrieval 

**Authors**: Md. Asraful Haque, Aasar Mehdi, Maaz Mahboob, Tamkeen Fatima  

**Link**: [PDF](https://arxiv.org/pdf/2603.17872)  

**Abstract**: Large Language Models (LLMs) have achieved unprecedented fluency but remain susceptible to "hallucinations" - the generation of factually incorrect or ungrounded content. This limitation is particularly critical in high-stakes domains where reliability is paramount. We propose a domain-grounded tiered retrieval and verification architecture designed to systematically intercept factual inaccuracies by shifting LLMs from stochastic pattern-matchers to verified truth-seekers. The proposed framework utilizes a four-phase, self-regulating pipeline implemented via LangGraph: (I) Intrinsic Verification with Early-Exit logic to optimize compute, (II) Adaptive Search Routing utilizing a Domain Detector to target subject-specific archives, (III) Corrective Document Grading (CRAG) to filter irrelevant context, and (IV) Extrinsic Regeneration followed by atomic claim-level verification. The system was evaluated across 650 queries from five diverse benchmarks: TimeQA v2, FreshQA v2, HaluEval General, MMLU Global Facts, and TruthfulQA. Empirical results demonstrate that the pipeline consistently outperforms zero-shot baselines across all environments. Win rates peaked at 83.7% in TimeQA v2 and 78.0% in MMLU Global Facts, confirming high efficacy in domains requiring granular temporal and numerical precision. Groundedness scores remained robustly stable between 78.8% and 86.4% across factual-answer rows. While the architecture provides a robust fail-safe for misinformation, a persistent failure mode of "False-Premise Overclaiming" was identified. These findings provide a detailed empirical characterization of multi-stage RAG behavior and suggest that future work should prioritize pre-retrieval "answerability" nodes to further bridge the reliability gap in conversational AI. 

---
# How do LLMs Compute Verbal Confidence 

**Authors**: Dharshan Kumaran, Arthur Conmy, Federico Barbero, Simon Osindero, Viorica Patraucean, Petar Velickovic  

**Link**: [PDF](https://arxiv.org/pdf/2603.17839)  

**Abstract**: Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed - just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents - token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B and Qwen 2.5 7B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration. 

---
# Event-Centric Human Value Understanding in News-Domain Texts: An Actor-Conditioned, Multi-Granularity Benchmark 

**Authors**: Yao Wang, Xin Liu, Zhuochen Liu, Jiankang Chen, Adam Jatowt, Kyoungsook Kim, Noriko Kando, Haitao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17838)  

**Abstract**: Existing human value datasets do not directly support value understanding in factual news: many are actor-agnostic, rely on isolated utterances or synthetic scenarios, and lack explicit event structure or value direction. We present \textbf{NEVU} (\textbf{N}ews \textbf{E}vent-centric \textbf{V}alue \textbf{U}nderstanding), a benchmark for \emph{actor-conditioned}, \emph{event-centric}, and \emph{direction-aware} human value recognition in factual news. NEVU evaluates whether models can identify value cues, attribute them to the correct actor, and determine value direction from grounded evidence. Built from 2{,}865 English news articles, NEVU organizes annotations at four semantic unit levels (\textbf{Subevent}, \textbf{behavior-based composite event}, \textbf{story-based composite event}, and \textbf{Article}) and labels \mbox{(unit, actor)} pairs for fine-grained evaluation across local and composite contexts. The annotations are produced through an LLM-assisted pipeline with staged verification and targeted human auditing. Using a hierarchical value space with \textbf{54} fine-grained values and \textbf{20} coarse-grained categories, NEVU covers 45{,}793 unit--actor pairs and 168{,}061 directed value instances. We provide unified baselines for proprietary and open-source LLMs, and find that lightweight adaptation (LoRA) consistently improves open-source models, showing that although NEVU is designed primarily as a benchmark, it also supports supervised adaptation beyond prompting-only evaluation. Data availability is described in Appendix~\ref{app:data_code_availability}. 

---
# Text-to-Stage: Spatial Layouts from Long-form Narratives 

**Authors**: Jefferson Hernandez, Swarnadeep Saha, Chenxi Whitehouse, Sanjeel Parekh, Calvin Murdock, Yuliang Li, W. Owen Brimijoin, Vamsi Krishna Ithapu, Ishwarya Ananthabhotla  

**Link**: [PDF](https://arxiv.org/pdf/2603.17832)  

**Abstract**: In this work, we probe the ability of a language model to demonstrate spatial reasoning from unstructured text, mimicking human capabilities and automating a process that benefits many downstream media applications. Concretely, we study the narrative-to-play task: inferring stage-play layouts (scenes, speaker positions, movements, and room types) from text that lacks explicit spatial, positional, or relational cues. We then introduce a dramaturgy-inspired deterministic evaluation suite and, finally, a training and inference recipe that combines rejection SFT using Best-of-N sampling with RL from verifiable rewards via GRPO. Experiments on a text-only corpus of classical English literature demonstrate improvements over vanilla models across multiple metrics (character attribution, spatial plausibility, and movement economy), as well as alignment with an LLM-as-a-judge and subjective human preferences. 

---
# Process Supervision for Chain-of-Thought Reasoning via Monte Carlo Net Information Gain 

**Authors**: Corentin Royer, Debarun Bhattacharjya, Gaetano Rossiello, Andrea Giovannini, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2603.17815)  

**Abstract**: Multi-step reasoning improves the capabilities of large language models (LLMs) but increases the risk of errors propagating through intermediate steps. Process reward models (PRMs) mitigate this by scoring each step individually, enabling fine-grained supervision and improved reliability. Existing methods for training PRMs rely on costly human annotations or computationally intensive automatic labeling. We propose a novel approach to automatically generate step-level labels using Information Theory. Our method estimates how each reasoning step affects the likelihood of the correct answer, providing a signal of step quality. Importantly, it reduces computational complexity to $\mathcal{O}(N)$, improving over the previous $\mathcal{O}(N \log N)$ methods. We demonstrate that these labels enable effective chain-of-thought selection in best-of-$K$ evaluation settings across diverse reasoning benchmarks, including mathematics, Python programming, SQL, and scientific question answering. This work enables scalable and efficient supervision of LLM reasoning, particularly for tasks where error propagation is critical. 

---
# CoVerRL: Breaking the Consensus Trap in Label-Free Reasoning via Generator-Verifier Co-Evolution 

**Authors**: Teng Pan, Yuchen Yan, Zixuan Wang, Ruiqing Zhang, Gaiyang Han, Wanqi Zhang, Weiming Lu, Jun Xiao, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17775)  

**Abstract**: Label-free reinforcement learning enables large language models to improve reasoning capabilities without ground-truth supervision, typically by treating majority-voted answers as pseudo-labels. However, we identify a critical failure mode: as training maximizes self-consistency, output diversity collapses, causing the model to confidently reinforce systematic errors that evade detection. We term this the consensus trap. To escape it, we propose CoVerRL, a framework where a single model alternates between generator and verifier roles, with each capability bootstrapping the other. Majority voting provides noisy but informative supervision for training the verifier, while the improving verifier progressively filters self-consistent errors from pseudo-labels. This co-evolution creates a virtuous cycle that maintains high reward accuracy throughout training. Experiments across Qwen and Llama model families demonstrate that CoVerRL outperforms label-free baselines by 4.7-5.9\% on mathematical reasoning benchmarks. Moreover, self-verification accuracy improves from around 55\% to over 85\%, confirming that both capabilities genuinely co-evolve. 

---
# Harm or Humor: A Multimodal, Multilingual Benchmark for Overt and Covert Harmful Humor 

**Authors**: Ahmed Sharshar, Hosam Elgendy, Saad El Dine Ahmed, Yasser Rohaim, Yuxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17759)  

**Abstract**: Dark humor often relies on subtle cultural nuances and implicit cues that require contextual reasoning to interpret, posing safety challenges that current static benchmarks fail to capture. To address this, we introduce a novel multimodal, multilingual benchmark for detecting and understanding harmful and offensive humor. Our manually curated dataset comprises 3,000 texts and 6,000 images in English and Arabic, alongside 1,200 videos that span English, Arabic, and language-independent (universal) contexts. Unlike standard toxicity datasets, we enforce a strict annotation guideline: distinguishing \emph{Safe} jokes from \emph{Harmful} ones, with the latter further classified into \emph{Explicit} (overt) and \emph{Implicit} (Covert) categories to probe deep reasoning. We systematically evaluate state-of-the-art (SOTA) open and closed-source models across all modalities. Our findings reveal that closed-source models significantly outperform open-source ones, with a notable difference in performance between the English and Arabic languages in both, underscoring the critical need for culturally grounded, reasoning-aware safety alignment. \textcolor{red}{Warning: this paper contains example data that may be offensive, harmful, or biased.} 

---
# Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models 

**Authors**: Jaemin Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.17677)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factual grounding by incorporating external knowledge into language model generation. However, when retrieved context is noisy, unreliable, or inconsistent with the model's parametric knowledge, it introduces retrieval-prior conflicts that can degrade generation quality. While this problem has been studied in autoregressive language models, it remains largely unexplored in diffusion-based language models, where the iterative denoising process introduces unique challenges for integrating retrieved context. In this work, we propose Adaptive Retrieval-Augmented Masked Diffusion (ARAM), a training-free adaptive guidance framework for Masked Diffusion Models (MDMs) in RAG settings. ARAM dynamically calibrates the guidance scale during denoising according to the Signal-to-Noise Ratio (SNR) of the distributional shift induced by retrieved context. Intuitively, the model strengthens guidance when the retrieved context provides reliable corrective evidence and suppresses it when the contextual signal is noisy or non-supportive. Extensive experiments on multiple knowledge-intensive QA benchmarks show that ARAM improves overall QA performance over competitive RAG baselines. 

---
# Do Language Models Encode Semantic Relations? Probing and Sparse Feature Analysis 

**Authors**: Andor Diera, Ansgar Scherp  

**Link**: [PDF](https://arxiv.org/pdf/2603.17624)  

**Abstract**: Understanding whether large language models (LLMs) capture structured meaning requires examining how they represent concept relationships. In this work, we study three models of increasing scale: Pythia-70M, GPT-2, and Llama 3.1 8B, focusing on four semantic relations: synonymy, antonymy, hypernymy, and hyponymy. We combine linear probing with mechanistic interpretability techniques, including sparse autoencoders (SAE) and activation patching, to identify where these relations are encoded and how specific features contribute to their representation. Our results reveal a directional asymmetry in hierarchical relations: hypernymy is encoded redundantly and resists suppression, while hyponymy relies on compact features that are more easily disrupted by ablation. More broadly, relation signals are diffuse but exhibit stable profiles: they peak in the mid-layers and are stronger in post-residual/MLP pathways than in attention. Difficulty is consistent across models (antonymy easiest, synonymy hardest). Probe-level causality is capacity-dependent: on Llama 3.1, SAE-guided patching reliably shifts these signals, whereas on smaller models the shifts are weak or unstable. Our results clarify where and how reliably semantic relations are represented inside LLMs, and provide a reproducible framework for relating sparse features to probe-level causal evidence. 

---
# VeriAgent: A Tool-Integrated Multi-Agent System with Evolving Memory for PPA-Aware RTL Code Generation 

**Authors**: Yaoxiang Wang, Qi Shi, ShangZhan Li, Qingguo Hu, Xinyu Yin, Bo Guo, Xu Han, Maosong Sun, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2603.17613)  

**Abstract**: LLMs have recently demonstrated strong capabilities in automatic RTL code generation, achieving high syntactic and functional correctness. However, most methods focus on functional correctness while overlooking critical physical design objectives, including Power, Performance, and Area. In this work, we propose a PPA-aware, tool-integrated multi-agent framework for high-quality verilog code generation. Our framework explicitly incorporates EDA tools into a closed-loop workflow composed of a \textit{Programmer Agent}, a \textit{Correctness Agent}, and a \textit{PPA Agent}, enabling joint optimization of functional correctness and physical metrics. To support continuous improvement without model retraining, we introduce an \textit{Evolved Memory Mechanism} that externalizes optimization experience into structured memory nodes. A dedicated memory manager dynamically maintains the memory pool and allows the system to refine strategies based on historical execution trajectories. Extensive experiments demonstrate that our approach achieves strong functional correctness while delivering significant improvements in PPA metrics. By integrating tool-driven feedback with structured and evolvable memory, our framework transforms RTL generation from one-shot reasoning into a continual, feedback-driven optimization process, providing a scalable pathway for deploying LLMs in real-world hardware design flows. 

---
# KA2L: A Knowledge-Aware Active Learning Framework for LLMs 

**Authors**: Haoxuan Yin, Bojian Liu, Chen Tang, Yangfan Wang, Lian Yan, Jingchi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17566)  

**Abstract**: Fine-tuning large language models (LLMs) with high-quality knowledge has been shown to enhance their performance effectively. However, there is a paucity of research on the depth of domain-specific knowledge comprehension by LLMs and the application of targeted active learning to improve their expertise. To address this gap, we introduce the Knowledge-Aware Active Learning (KA2L) framework. This framework assesses LLMs' mastery of specific knowledge points to aid in constructing unanswerable or unknowable questions through latent space analysis. This active learning strategy enhances training efficiency by focusing on knowledge the model has yet to master, thereby minimizing redundancy in learning already acquired information. This study innovatively employs a knowledge distribution probing technique to examine the hidden states of specific Transformer layers and identify the distribution of known and unknown knowledge within the LLM. Additionally, a hidden-state decoding method is proposed to generate numerous unknown questions in natural language from the latent knowledge space. In our experiments, we selected nine open-source LLMs to validate the effectiveness of the proposed framework. Results indicate that KA2L not only significantly reduces 50% annotation and computation costs across two open-domain and one vertical-domain dataset but also achieves better performance, offering valuable insights into active learning strategies for LLMs. The code is available at this https URL. 

---
# Zipper-LoRA: Dynamic Parameter Decoupling for Speech-LLM based Multilingual Speech Recognition 

**Authors**: Yuxiang Mei, Delai Qiu, Shengping Liu, Jiaen Liang, Yanhua Long  

**Link**: [PDF](https://arxiv.org/pdf/2603.17558)  

**Abstract**: Speech Large Language Models (Speech-LLMs) have emerged as a powerful approach for automatic speech recognition (ASR) by aligning speech encoders with large language models. However, adapting these systems to multilingual settings with imbalanced data distributions remains challenging. In such scenarios, a stability-plasticity dilemma often arises: fully shared Parameter-Efficient Fine-Tuning (PEFT) can cause negative inter-lingual interference for under-represented languages, while fully language-specific tuning limits the cross-lingual beneficial knowledge transfer needed for low-resource tasks. To address this, we propose Zipper-LoRA, a novel rank-level decoupling framework with three variants (Static, Hard, and Soft) that dynamically synthesizes LoRA updates from shared and language-specific subspaces. By using a lightweight language-conditioned router, Zipper-LoRA dynamically controls the contribution of each subspace at the LoRA rank level, enabling fine-grained sharing where languages are compatible and strict decoupling when conflicts occur. To further stabilize optimization under imbalanced data, we propose a two-stage training strategy with an Initial-B warm start that significantly accelerates convergence. Experiments on a 12-language mixed-resource setting show that Zipper-LoRA consistently outperforms both fully shared and independent baselines, particularly in extremely low-resource scenarios. Moreover, we demonstrate that these gains are robust across both chunked and non-chunked encoder configurations, confirming the framework's reliability for practical, large-scale multilingual ASR. Our code and data will be available at this https URL for reproducibility. 

---
# AURORA Model of Formant-to-Tongue Inversion for Didactic and Clinical Applications 

**Authors**: Patrycja Strycharczuk, Sam Kirkham  

**Link**: [PDF](https://arxiv.org/pdf/2603.17543)  

**Abstract**: This paper outlines the conceptual and computational foundations of the AURORA (Acoustic Understanding and Real-time Observation of Resonant Articulations) model. AURORA predicts tongue displacement and shape in vowel sounds based on the first two formant values. It is intended as a didactic aid helping to explain the relationship between formants and the underlying articulation, as well as a foundation for biofeedback applications. The model is informed by ultrasound tongue imaging and acoustic data from 40 native speakers of English. In this paper we discuss the motivation for the model, the modelling objectives as well as the model architecture. We provide a qualitative evaluation of the model, focusing on selected tongue features. We then present two tools developed to make the model more accessible to a wider audience, a Shiny app and a prototype software for real-time tongue biofeedback. Potential users include students of phonetics, linguists in fields adjacent to phonetics, as well as speech and language therapy practitioners and clients. 

---
# Detecting the Machine: A Comprehensive Benchmark of AI-Generated Text Detectors Across Architectures, Domains, and Adversarial Conditions 

**Authors**: Madhav S. Baidya, S. S. Baidya, Chirag Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2603.17522)  

**Abstract**: The rapid proliferation of large language models (LLMs) has created an urgent need for robust and generalizable detectors of machine-generated text. Existing benchmarks typically evaluate a single detector on a single dataset under ideal conditions, leaving open questions about cross-domain transfer, cross-LLM generalization, and adversarial robustness.
We present a comprehensive benchmark evaluating diverse detection approaches across two corpora: HC3 (23,363 human-ChatGPT pairs) and ELI5 (15,000 human-Mistral-7B pairs). Methods include classical classifiers, fine-tuned transformer encoders (BERT, RoBERTa, ELECTRA, DistilBERT, DeBERTa-v3), a CNN, an XGBoost stylometric model, perplexity-based detectors, and LLM-as-detector prompting.
Results show that transformer models achieve near-perfect in-distribution performance but degrade under domain shift. The XGBoost stylometric model matches performance while remaining interpretable. LLM-based detectors underperform and are affected by generator-detector identity bias. Perplexity-based methods exhibit polarity inversion, with modern LLM outputs showing lower perplexity than human text, but remain effective when corrected. No method generalizes robustly across domains and LLM sources. 

---
# Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality 

**Authors**: Mengyu Bu, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17512)  

**Abstract**: Large language models (LLMs) exhibit strong general intelligence, yet their multilingual performance remains highly imbalanced. Although LLMs encode substantial cross-lingual knowledge in a unified semantic space, they often struggle to reliably interface this knowledge with low-resource or unseen languages. Fortunately, pretrained encoder-decoder translation models already possess balanced multilingual capability, suggesting a natural complement to LLMs. In this work, we propose XBridge, a compositional encoder-LLM-decoder architecture that offloads multilingual understanding and generation to external pretrained translation models, while preserving the LLM as an English-centric core for general knowledge processing. To address the resulting representation misalignment across models, we introduce lightweight cross-model mapping layers and an optimal transport-based alignment objective, enabling fine-grained semantic consistency for multilingual generation. Experiments on four LLMs across multilingual understanding, reasoning, summarization, and generation indicate that XBridge outperforms strong baselines, especially on low-resource and previously unseen languages, without retraining the LLM. 

---
# Inducing Epistemological Humility in Large Language Models: A Targeted SFT Approach to Reducing Hallucination 

**Authors**: Cem Uluoglakci, Tugba Taskaya Temizel  

**Link**: [PDF](https://arxiv.org/pdf/2603.17504)  

**Abstract**: Large language models (LLMs) often hallucinate, producing fluent but false information, partly because supervised fine-tuning (SFT) implicitly rewards always responding. We introduce $\textit{HypoTermInstruct}$, an SFT dataset (31,487 responses for 11,151 questions) designed to teach models epistemological humility-the ability to recognize the limits of their own knowledge and admit uncertainty. This is achieved through questions about non-existent "hypothetical" terms. We also release $\textit{HypoTermQA-Enhanced}$, a benchmark for hallucination tendency strengthened through multiple validations. We conducted 800 controlled LoRA SFT runs across $\textit{Llama3.1-8B}$ and $\textit{Gemma3-4B}$ (base and instruct), testing 100 fine-tuning configurations with paired controls. Our results demonstrate that replacing generic instruction data with $\textit{HypoTermInstruct}$ significantly improves the HypoTerm Score (median increases of 0.19% to 25.91%) and FactScore (+0.39% to +0.86%), while maintaining stable performance on MMLU (minimal decreases of 0.26% to 0.35%). Our work demonstrates that targeted, high-quality SFT data teaching meta-cognitive skills can effectively reduce hallucination without preference/RL pipelines, providing mechanistic insights and a practical path toward more reliable AI systems. 

---
# Learning When to Attend: Conditional Memory Access for Long-Context LLMs 

**Authors**: Sakshi Choudhary, Aditya Chattopadhyay, Luca Zancato, Elvis Nunez, Matthew Trager, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2603.17484)  

**Abstract**: Language models struggle to generalize beyond pretraining context lengths, limiting long-horizon reasoning and retrieval. Continued pretraining on long-context data can help but is expensive due to the quadratic scaling of Attention. We observe that most tokens do not require (Global) Attention over the entire sequence and can rely on local context. Based on this, we propose L2A (Learning To Attend), a layer that enables conditional (token-wise) long-range memory access by deciding when to invoke global attention. We evaluate L2A on Qwen 2.5 and Qwen 3 models, extending their effective context length from 32K to 128K tokens. L2A matches the performance of standard long-context training to within 3% while skipping Global Attention for $\sim$80% of tokens, outperforming prior baselines. We also design custom Triton kernels to efficiently implement this token-wise conditional Attention on GPUs, achieving up to $\sim$2x improvements in training throughput and time-to-first-token over FlashAttention. Moreover, L2A enables post-training pruning of highly sparse Global Attention layers, reducing KV cache memory by up to 50% with negligible performance loss. 

---
# Humans and transformer LMs: Abstraction drives language learning 

**Authors**: Jasper Jian, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2603.17475)  

**Abstract**: Categorization is a core component of human linguistic competence. We investigate how a transformer-based language model (LM) learns linguistic categories by comparing its behaviour over the course of training to behaviours which characterize abstract feature-based and concrete exemplar-based accounts of human language acquisition. We investigate how lexical semantic and syntactic categories emerge using novel divergence-based metrics that track learning trajectories using next-token distributions. In experiments with GPT-2 small, we find that (i) when a construction is learned, abstract class-level behaviour is evident at earlier steps than lexical item-specific behaviour, and (ii) that different linguistic behaviours emerge abruptly in sequence at different points in training, revealing that abstraction plays a key role in how LMs learn. This result informs the models of human language acquisition that LMs may serve as an existence proof for. 

---
# TRiMS: Real-Time Tracking of Minimal Sufficient Length for Efficient Reasoning via RL 

**Authors**: Tingcheng Bian, Jinchang Luo, Mingquan Cheng, Jinyu Zhang, Xiaoling Xia, Ni Li, Yan Tao, Haiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.17449)  

**Abstract**: Large language models achieve breakthroughs in complex reasoning via long chain-of-thought sequences. However, this often leads to severe reasoning inflation, causing substantial computational redundancy. To maximize Intelligence per Token, we introduce a theoretical metric, MSL-Minimal Sufficient Length. MSL rigorously characterizes the shortest reasoning length that preserves answer correctness. We provide a recursive definition based on independently sampled sequences and prove the existence of its limit, establishing the first measurable lower bound for reasoning-chain compression. Building on an analysis of mainstream CoT compression strategies, we identify key structural factors enabling a model to approach MSL. Based on these insights, we propose TRiMS which employs the GRPO algorithm in conjunction with MSL-based estimation during training, while mitigating instabilities during the training process through dynamic batch aggregation and advantage computation using batch-level standard deviation. TRiMS achieves over 80% CoT token reduction with a minor accuracy boost across all benchmarks. 

---
# Argument Reconstruction as Supervision for Critical Thinking in LLMs 

**Authors**: Hyun Ryu, Gyouk Chu, Gregor Betz, Eunho Yang, Carolyn Rose, Sean Welleck  

**Link**: [PDF](https://arxiv.org/pdf/2603.17432)  

**Abstract**: To think critically about arguments, human learners are trained to identify, reconstruct, and evaluate arguments. Argument reconstruction is especially important because it makes an argument's underlying inferences explicit. However, it remains unclear whether LLMs can similarly enhance their critical thinking ability by learning to reconstruct arguments. To address this question, we introduce a holistic framework with three contributions. We (1) propose an engine that automatically reconstructs arbitrary arguments (GAAR), (2) synthesize a new high-quality argument reconstruction dataset (Arguinas) using the GAAR engine, and (3) investigate whether learning argument reconstruction benefits downstream critical thinking tasks. Our experimental results show that, across seven critical thinking tasks, models trained to learn argument reconstruction outperform models that do not, with the largest performance gains observed when training on the proposed Arguinas dataset. The source code and dataset will be publicly available. 

---
# SafeTutors: Benchmarking Pedagogical Safety in AI Tutoring Systems 

**Authors**: Rima Hazra, Bikram Ghuku, Ilona Marchenko, Yaroslava Tokarieva, Sayan Layek, Somnath Banerjee, Julia Stoyanovich, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2603.17373)  

**Abstract**: Large language models are rapidly being deployed as AI tutors, yet current evaluation paradigms assess problem-solving accuracy and generic safety in isolation, failing to capture whether a model is simultaneously pedagogically effective and safe across student-tutor interaction. We argue that tutoring safety is fundamentally different from conventional LLM safety: the primary risk is not toxic content but the quiet erosion of learning through answer over-disclosure, misconception reinforcement, and the abdication of scaffolding. To systematically study this failure mode, we introduce SafeTutors, a benchmark that jointly evaluates safety and pedagogy across mathematics, physics, and chemistry. SafeTutors is organized around a theoretically grounded risk taxonomy comprising 11 harm dimensions and 48 sub-risks drawn from learning-science literature. We uncover that all models show broad harm; scale doesn't reliably help; and multi-turn dialogue worsens behavior, with pedagogical failures rising from 17.7% to 77.8%. Harms also vary by subject, so mitigations must be discipline-aware, and single-turn "safe/helpful" results can mask systematic tutor failure over extended interaction. 

---
# PACE-RAG: Patient-Aware Contextual and Evidence-based Policy RAG for Clinical Drug Recommendation 

**Authors**: Chaeyoung Huh, Hyunmin Hwang, Jung Hwan Shin, Jinse Park, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.17356)  

**Abstract**: Drug recommendation requires a deep understanding of individual patient context, especially for complex conditions like Parkinson's disease. While LLMs possess broad medical knowledge, they fail to capture the subtle nuances of actual prescribing patterns. Existing RAG methods also struggle with these complexities because guideline-based retrieval remains too generic and similar-patient retrieval often replicates majority patterns without accounting for the unique clinical nuances of individual patients. To bridge this gap, we propose PACE-RAG (Patient-Aware Contextual and Evidence-based Policy RAG), a novel framework designed to synthesize individual patient context with the prescribing tendencies of similar cases. By analyzing treatment patterns tailored to specific clinical signals, PACE-RAG identifies optimal prescriptions and generates an explainable clinical summary. Evaluated on a Parkinson's cohort and the MIMIC-IV benchmark using Llama-3.1-8B and Qwen3-8B, PACE-RAG achieved state-of-the-art performance, reaching F1 scores of 80.84% and 47.22%, respectively. These results validate PACE-RAG as a robust, clinically grounded solution for personalized decision support. Our code is available at: this https URL. 

---
# Grid Spatial Understanding: A Dataset for Textual Spatial Reasoning over Grids, Embodied Settings, and Coordinate Structures 

**Authors**: Risham Sidhu, Julia Hockenmaier  

**Link**: [PDF](https://arxiv.org/pdf/2603.17333)  

**Abstract**: We introduce GSU, a text-only grid dataset to evaluate the spatial reasoning capabilities of LLMs over 3 core tasks: navigation, object localization, and structure composition. By forgoing visual inputs, isolating spatial reasoning from perception, we show that while most models grasp basic grid concepts, they struggle with frames of reference relative to an embodied agent and identifying 3D shapes from coordinate lists. We also find that exposure to a visual modality does not provide a generalizable understanding of 3D space that VLMs are able to utilize for these tasks. Finally, we show that while the very latest frontier models can solve the provided tasks (though harder variants may still stump them), fully fine-tuning a small LM or LORA fine-tuning a small LLM show potential to match frontier model performance, suggesting an avenue for specialized embodied agents. 

---
# Ruyi2.5 Technical Report 

**Authors**: Huan Song, Shuyu Tian, Qingfei Zhao, Wenhao Hong, Jiang Liu, Ting Long, Jiawei Shao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.17311)  

**Abstract**: We present Ruyi2.5, a multimodal familial model built on the AI Flow framework. Extending Ruyi2's "Train Once, Deploy Many" paradigm to the multimodal domain, Ruyi2.5 constructs a shared-backbone architecture that co-trains models of varying scales within a single unified pipeline, ensuring semantic consistency across all deployment tiers. Built upon Ruyi2.5, Ruyi2.5-Camera model is developed as a privacy-preserving camera service system, which instantiates Ruyi2.5-Camera into a two-stage recognition pipeline: an edge model applies information-bottleneck-guided irreversible feature mapping to de-identify raw frames at the source, while a cloud model performs deep behavior reasoning. To accelerate reinforcement learning fine-tuning, we further propose Binary Prefix Policy Optimization (BPPO), which reduces sample redundancy via binary response selection and focuses gradient updates on response prefixes, achieving a 2 to 3 times training speedup over GRPO. Experiments show Ruyi2.5 matches Qwen3-VL on the general multimodal benchmarks, while Ruyi2.5-Camera substantially outperforms Qwen3-VL on privacy-constrained surveillance tasks. 

---
# Beyond bouba/kiki: Multidimensional semantic signals are deeply woven into the fabric of natural language 

**Authors**: Gexin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17306)  

**Abstract**: A foundational assumption in linguistics holds that the relationship between a word's sound and its meaning is arbitrary. Accumulating evidence from sound symbolism challenges this view, yet no study has systematically mapped the multidimensional semantic profile of every phonological unit within a language. Here we show that individual letter-phonemes in English carry structured, multidimensional semantic signals. Using a minimal-pair paradigm spanning all 220 pairwise letter contrasts, three large language models independently recover consistent phoneme-meaning associations across nine perceptual dimensions. These associations are systematically predicted by articulatory-phonetic features, with manner and place of articulation mapping onto distinct semantic dimensions. Behavioral data from English speakers confirm these patterns at rates well above chance (80.8%), and preliminary cross-linguistic evidence from five typologically diverse languages suggests that core mappings generalize beyond English. Our findings indicate that sound-meaning iconicity is not an occasional curiosity but a pervasive, structured property of the phonological signal, one so systematic that large language models recover it when given only text input, without exposure to speech or articulation during the task. 

---
# From Words to Worlds: Benchmarking Cross-Cultural Cultural Understanding in Machine Translation 

**Authors**: Bangju Han, Yingqi Wang, Huang Qing, Tiyuan Li, Fengyi Yang, Ahtamjan Ahmat, Abibulla Atawulla, Yating Yang, Xi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.17303)  

**Abstract**: Culture-expressions, such as idioms, slang, and culture-specific items (CSIs), are pervasive in natural language and encode meanings that go beyond literal linguistic form. Accurately translating such expressions remains challenging for machine translation systems. Despite this, existing benchmarks remain fragmented and do not provide a systematic framework for evaluating translation performance on culture-loaded expressions. To address this gap, we introduce CulT-Eval, a benchmark designed to evaluate how models handle different types of culturally grounded expressions. CulT-Eval comprises over 7,959 carefully curated instances spanning multiple types of culturally grounded expressions, with a comprehensive error taxonomy covering culturally grounded expressions. Through extensive evaluation of large language models and detailed analysis, we identify recurring and systematic failure modes that are not adequately captured by existing automatic metrics. Accordingly, we propose a complementary evaluation metric that targets culturally induced meaning deviations overlooked by standard MT metrics. The results indicate that current models struggle to preserve culturally grounded meaning and to capture the cultural and contextual nuances essential for accurate translation. Our benchmark and code are available at this https URL. 

---
# Neuron-Level Emotion Control in Speech-Generative Large Audio-Language Models 

**Authors**: Xiutian Zhao, Ismail Rasim Ulgen, Philipp Koehn, Björn Schuller, Berrak Sisman  

**Link**: [PDF](https://arxiv.org/pdf/2603.17231)  

**Abstract**: Large audio-language models (LALMs) can produce expressive speech, yet reliable emotion control remains elusive: conversions often miss the target affect and may degrade linguistic fidelity through refusals, hallucinations, or paraphrase. We present, to our knowledge, the first neuron-level study of emotion control in speech-generative LALMs and demonstrate that compact emotion-sensitive neurons (ESNs) are causally actionable, enabling training-free emotion steering at inference time. ESNs are identified via success-filtered activation aggregation enforcing both emotion realization and content preservation. Across three LALMs (Qwen2.5-Omni-7B, MiniCPM-o 4.5, Kimi-Audio), ESN interventions yield emotion-specific gains that generalize to unseen speakers and are supported by automatic and human evaluation. Controllability depends on selector design, mask sparsity, filtering, and intervention strength. Our results establish a mechanistic framework for training-free emotion control in speech generation. 

---
# TharuChat: Bootstrapping Large Language Models for a Low-Resource Language via Synthetic Data and Human Validation 

**Authors**: Prajwal Panth, Agniva Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2603.17220)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has created a profound digital divide, effectively excluding indigenous languages of the Global South from the AI revolution. The Tharu language, an Indo-Aryan vernacular spoken by approximately 1.7 million people across the Terai belt of Nepal and India, exemplifies this crisis. Despite a rich oral tradition, Tharu suffers from severe data scarcity and linguistic fragmentation, causing state-of-the-art multilingual models to routinely "hallucinate" or default to dominant high-resource neighbors like Hindi and Nepali due to contamination in pre-training corpora.
This paper presents Tharu-LLaMA (3B), a specialized instruction-following model designed to address this exclusion. We introduce TharuChat, a novel dataset constructed via a LLM-to-Human bootstrapping pipeline. We utilized prompt-engineered Gemini models, fed with Rana Tharu grammar and folklore, to synthesize training data. Unlike curated gold-standard corpora, TharuChat reflects the noisy, heterogeneous linguistic reality of the region: it is predominantly anchored in Rana Tharu (~70%) while integrating elements of Dangaura and Kochila dialects. We provide a transparent analysis of the dataset's limitations, including dialectal code-mixing and residual Awadhi/Hindi influence. Through a rigorous empirical ablation study, we demonstrate that despite these imperfections, small-scale synthetic data is highly effective, increasing the dataset volume from 25% to 100% results in a linear reduction in perplexity from 6.42 to 2.88. The resulting model serves as a proof-of-concept for the preservation of under-resourced Himalayan languages via generative AI, achievable on consumer-grade hardware. 

---
# Alignment Makes Language Models Normative, Not Descriptive 

**Authors**: Eilam Shapira, Moshe Tennenholtz, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2603.17218)  

**Abstract**: Post-training alignment optimizes language models to match human preference signals, but this objective is not equivalent to modeling observed human behavior. We compare 120 base-aligned model pairs on more than 10,000 real human decisions in multi-round strategic games - bargaining, persuasion, negotiation, and repeated matrix games. In these settings, base models outperform their aligned counterparts in predicting human choices by nearly 10:1, robustly across model families, prompt formulations, and game configurations. This pattern reverses, however, in settings where human behavior is more likely to follow normative predictions: aligned models dominate on one-shot textbook games across all 12 types tested and on non-strategic lottery choices - and even within the multi-round games themselves, at round one, before interaction history develops. This boundary-condition pattern suggests that alignment induces a normative bias: it improves prediction when human behavior is relatively well captured by normative solutions, but hurts prediction in multi-round strategic settings, where behavior is shaped by descriptive dynamics such as reciprocity, retaliation, and history-dependent adaptation. These results reveal a fundamental trade-off between optimizing models for human use and using them as proxies for human behavior. 

---
# Anonymous-by-Construction: An LLM-Driven Framework for Privacy-Preserving Text 

**Authors**: Federico Albanese, Pablo Ronco, Nicolás D'Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2603.17217)  

**Abstract**: Responsible use of AI demands that we protect sensitive information without undermining the usefulness of data, an imperative that has become acute in the age of large language models. We address this challenge with an on-premise, LLM-driven substitution pipeline that anonymizes text by replacing personally identifiable information (PII) with realistic, type-consistent surrogates. Executed entirely within organizational boundaries using local LLMs, the approach prevents data egress while preserving fluency and task-relevant semantics.
We conduct a systematic, multi-metric, cross-technique evaluation on the Action-Based Conversation Dataset, benchmarking against industry standards (Microsoft Presidio and Google DLP) and a state-of-the-art approach (ZSTS, in redaction-only and redaction-plus-substitution variants). Our protocol jointly measures privacy, semantic utility, and trainability under privacy via a lifecycle-ready criterion obtained by fine-tuning a compact encoder (BERT+LoRA) on sanitized text. In addition, we assess agentic Q&A performance by inserting an on-premise anonymization layer before the answering LLM and evaluating the quality of its responses. This intermediate, type-preserving substitution stage ensures that no sensitive content is exposed to third-party APIs, enabling responsible deployment of Q\&A agents without compromising confidentiality.
Our method attains state-of-the-art privacy, minimal topical drift, strong factual utility, and low trainability loss, outperforming rule-based approaches and named-entity recognition (NER) baselines and ZSTS variants on the combined privacy--utility--trainability frontier. These results show that local LLM substitution yields anonymized corpora that are both responsible to use and operationally valuable: safe for agentic pipelines and suitable for downstream fine-tuning with limited degradation. 

---
# SYMDIREC: A Neuro-Symbolic Divide-Retrieve-Conquer Framework for Enhanced RTL Synthesis and Summarization 

**Authors**: Prashanth Vijayaraghavan, Apoorva Nitsure, Luyao Shi, Charles Mackin, Ashutosh Jadhav, David Beymer, Ehsan Degan, Vandana Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2603.17208)  

**Abstract**: Register-Transfer Level (RTL) synthesis and summarization are central to hardware design automation but remain challenging for Large Language Models (LLMs) due to rigid HDL syntax, limited supervision, and weak alignment with natural language. Existing prompting and retrieval-augmented generation (RAG) methods have not incorporated symbolic planning, limiting their structural precision. We introduce SYMDIREC, a neuro-symbolic framework that decomposes RTL tasks into symbolic subgoals, retrieves relevant code via a fine-tuned retriever, and assembles verified outputs through LLM reasoning. Supporting both Verilog and VHDL without LLM fine-tuning, SYMDIREC achieves ~20% higher Pass@1 rates for synthesis and 15-20% ROUGE-L improvements for summarization over prompting and RAG baselines, demonstrating the benefits of symbolic guidance in RTL tasks. 

---
# CODMAS: A Dialectic Multi-Agent Collaborative Framework for Structured RTL Optimization 

**Authors**: Che-Ming Chang, Prashanth Vijayaraghavan, Ashutosh Jadhav, Charles Mackin, Vandana Mukherjee, Hsinyu Tsai, Ehsan Degan  

**Link**: [PDF](https://arxiv.org/pdf/2603.17204)  

**Abstract**: Optimizing Register Transfer Level (RTL) code is a critical step in Electronic Design Automation (EDA) for improving power, performance, and area (PPA). We present CODMAS (Collaborative Optimization via a Dialectic Multi-Agent System), a framework that combines structured dialectic reasoning with domain-aware code generation and deterministic evaluation to automate RTL optimization. At the core of CODMAS are two dialectic agents: the Articulator, inspired by rubber-duck debugging, which articulates stepwise transformation plans and exposes latent assumptions; and the Hypothesis Partner, which predicts outcomes and reconciles deviations between expected and actual behavior to guide targeted refinements. These agents direct a Domain-Specific Coding Agent (DCA) to generate architecture-aware Verilog edits and a Code Evaluation Agent (CEA) to verify syntax, functionality, and PPA metrics. We introduce RTLOPT, a benchmark of 120 Verilog triples (unoptimized, optimized, testbench) for pipelining and clock-gating transformations. Across proprietary and open LLMs, CODMAS achieves ~25% reduction in critical path delay for pipelining and ~22% power reduction for clock gating, while reducing functional and compilation failures compared to strong prompting and agentic baselines. These results demonstrate that structured multi-agent reasoning can significantly enhance automated RTL optimization and scale to more complex designs and broader optimization tasks. 

---
# Tabular LLMs for Interpretable Few-Shot Alzheimer's Disease Prediction with Multimodal Biomedical Data 

**Authors**: Sophie Kearney, Shu Yang, Zixuan Wen, Weimin Lyu, Bojian Hou, Duy Duong-Tran, Tianlong Chen, Jason H. Moore, Marylyn D. Ritchie, Chao Chen, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17191)  

**Abstract**: Accurate diagnosis of Alzheimer's disease (AD) requires handling tabular biomarker data, yet such data are often small and incomplete, where deep learning models frequently fail to outperform classical methods. Pretrained large language models (LLMs) offer few-shot generalization, structured reasoning, and interpretable outputs, providing a powerful paradigm shift for clinical prediction. We propose TAP-GPT Tabular Alzheimer's Prediction GPT, a domain-adapted tabular LLM framework built on TableGPT2 and fine-tuned for few-shot AD classification using tabular prompts rather than plain texts. We evaluate TAP-GPT across four ADNI-derived datasets, including QT-PAD biomarkers and region-level structural MRI, amyloid PET, and tau PET for binary AD classification. Across multimodal and unimodal settings, TAP-GPT improves upon its backbone models and outperforms traditional machine learning baselines in the few-shot setting while remaining competitive with state-of-the-art general-purpose LLMs. We show that feature selection mitigates degradation in high-dimensional inputs and that TAP-GPT maintains stable performance under simulated and real-world missingness without imputation. Additionally, TAP-GPT produces structured, modality-aware reasoning aligned with established AD biology and shows greater stability under self-reflection, supporting its use in iterative multi-agent systems. To our knowledge, this is the first systematic application of a tabular-specialized LLM to multimodal biomarker-based AD prediction, demonstrating that such pretrained models can effectively address structured clinical prediction tasks and laying the foundation for tabular LLM-driven multi-agent clinical decision-support systems. The source code is publicly available on GitHub: this https URL. 

---
# Exploiting the English Grammar Profile for L2 grammatical analysis with LLMs 

**Authors**: Stefano Bannò, Penny Karanasou, Kate Knill, Mark Gales  

**Link**: [PDF](https://arxiv.org/pdf/2603.17171)  

**Abstract**: Evaluating the grammatical competence of second language (L2) learners is essential both for providing targeted feedback and for assessing proficiency. To achieve this, we propose a novel framework leveraging the English Grammar Profile (EGP), a taxonomy of grammatical constructs mapped to the proficiency levels of the Common European Framework of Reference (CEFR), to detect learners' attempts at grammatical constructs and classify them as successful or unsuccessful. This detection can then be used to provide fine-grained feedback. Moreover, the grammatical constructs are used as predictors of proficiency assessment by using automatically detected attempts as predictors of holistic CEFR proficiency. For the selection of grammatical constructs derived from the EGP, rule-based and LLM-based classifiers are compared. We show that LLMs outperform rule-based methods on semantically and pragmatically nuanced constructs, while rule-based approaches remain competitive for constructs that rely purely on morphological or syntactic features and do not require semantic interpretation. For proficiency assessment, we evaluate both rule-based and hybrid pipelines and show that a hybrid approach combining a rule-based pre-filter with an LLM consistently yields the strongest performance. Since our framework operates on pairs of original learner sentences and their corrected counterparts, we also evaluate a fully automated pipeline using automatic grammatical error correction. This pipeline closely approaches the performance of semi-automated systems based on manual corrections, particularly for the detection of successful attempts at grammatical constructs. Overall, our framework emphasises learners' successful attempts in addition to unsuccessful ones, enabling positive, formative feedback and providing actionable insights into grammatical development. 

---
# Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency 

**Authors**: Lucas Bandarkar, Alan Ansell, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2603.17102)  

**Abstract**: Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the necessity of this small number of experts for answering a knowledge question, we deactivate them and re-ask the question. We find that despite only deactivating about 20 out of 6000 experts, the model no longer answers correctly in over 40% of cases. Generally, this method provides a realistic and scalable knowledge localization approach to address increasingly complex LLMs. 

---
# Evaluating LLM-Simulated Conversations in Modeling Inconsistent and Uncollaborative Behaviors in Human Social Interaction 

**Authors**: Ryo Kamoi, Ameya Godbole, Longqi Yang, Rui Zhang, Mengting Wan, Pei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.17094)  

**Abstract**: Simulating human conversations using large language models (LLMs) has emerged as a scalable methodology for modeling human social interaction. However, simulating human conversations is challenging because they inherently involve inconsistent and uncollaborative behaviors, such as misunderstandings and interruptions. Analysis comparing inconsistent and uncollaborative behaviors in human- and LLM-generated conversations remains limited, although reproducing these behaviors is integral to simulating human-like and complex social interaction. In this work, we introduce CoCoEval, an evaluation framework that analyzes LLM-simulated conversations by detecting 10 types of inconsistent and uncollaborative behaviors at the turn level using an LLM-as-a-Judge. Using CoCoEval, we evaluate GPT-4.1, GPT-5.1, and Claude Opus 4 by comparing the frequencies of detected behaviors in conversations simulated by each model and in human conversations across academic, business, and governmental meetings, as well as debates. Our analysis shows that (1) under vanilla prompting, LLM-simulated conversations exhibit far fewer inconsistent and uncollaborative behaviors than human conversations; (2) prompt engineering does not provide reliable control over these behaviors, as our results show that different prompts lead to their under- or overproduction; and (3) supervised fine-tuning on human conversations can lead LLMs to overproduce a narrow set of behaviors, such as repetition. Our findings highlight the difficulty of simulating human conversations, raising concerns about the use of LLMs as a proxy for human social interaction. 

---
# Ensemble Self-Training for Unsupervised Machine Translation 

**Authors**: Ido Aharon, Jonathan Shaki, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2603.17087)  

**Abstract**: We present an ensemble-driven self-training framework for unsupervised neural machine translation (UNMT). Starting from a primary language pair, we train multiple UNMT models that share the same translation task but differ in an auxiliary language, inducing structured diversity across models. We then generate pseudo-translations for the primary pair using token-level ensemble decoding, averaging model predictions in both directions. These ensemble outputs are used as synthetic parallel data to further train each model, allowing the models to improve via shared supervision. At deployment time, we select a single model by validation performance, preserving single-model inference cost. Experiments show statistically significant improvements over single-model UNMT baselines, with mean gains of 1.7 chrF when translating from English and 0.67 chrF when translating into English. 

---
# Large Reasoning Models Struggle to Transfer Parametric Knowledge Across Scripts 

**Authors**: Lucas Bandarkar, Alan Ansell, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2603.17070)  

**Abstract**: In this work, we analyze shortcomings in cross-lingual knowledge transfer in large, modern reasoning LLMs. We demonstrate that the perceived gap in knowledge transfer is primarily a script barrier. First, we conduct an observational data analysis on the performance of thinking models on two datasets with local knowledge from around the world, ECLeKTic and MultiLoKo. Our regression analysis shows that script match - not language or family - is the primary predictor of knowledge transfer failure once model capability and question difficulty are accounted for. We further this finding by providing the LLMs with the key entities of the questions in their source language and find that this disproportionately improves cross-script questions. We then posit that these LLMs could be reasoning better at test-time. To evaluate this, we develop a synthetic generation pipeline to design SFT samples to encourage the model to better reason about transliteration ambiguities when trying to fetch parametric knowledge at inference-time. We show that teaching two models to reason better reduces the cross-script transfer gap. As a result, we conclude that there is potential to improve cross-lingual parametric knowledge transfer during post-training. 

---
# Evaluating Ill-Defined Tasks in Large Language Models 

**Authors**: Yi Zhou, Basel Shbita  

**Link**: [PDF](https://arxiv.org/pdf/2603.17067)  

**Abstract**: Many evaluations of Large Language Models (LLMs) target tasks that are inherently ill-defined, with unclear input and output spaces and ambiguous success criteria. We analyze why existing evaluation benchmarks and metrics fail to provide reliable or diagnostic signals of model capability for such tasks. We examine two case studies: Complex Instruction Following (CIF), where we identify recurring issues including limited coverage of real-world instruction complexity, sensitivity to instruction phrasing, inconsistent and non-comparable metrics, and instability introduced by LLM-based judges; and Natural Language to Mermaid Sequence Diagrams (NL2Mermaid), where we show how multi-faceted evaluation criteria can yield actionable insights beyond aggregate scores. Together, these case studies show that current evaluations frequently conflate distinct failure modes, yielding scores that are unstable, non-diagnostic, and difficult to act upon. Our findings expose fundamental limitations in existing evaluation practices for ill-defined tasks and motivate more robust, interpretable evaluation designs. 

---
# LLM NL2SQL Robustness: Surface Noise vs. Linguistic Variation in Traditional and Agentic Settings 

**Authors**: Lifu Tu, Rongguang Wang, Tao Sheng, Sujjith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2603.17017)  

**Abstract**: Robustness evaluation for Natural Language to SQL (NL2SQL) systems is essential because real-world database environments are dynamic, noisy, and continuously evolving, whereas conventional benchmark evaluations typically assume static schemas and well-formed user inputs. In this work, we introduce a robustness evaluation benchmark containing approximately ten types of perturbations and conduct evaluations under both traditional and agentic settings. We assess multiple state-of-the-art large language models (LLMs), including Grok-4.1, Gemini-3-Pro, Claude-Opus-4.6, and GPT-5.2. Our results show that these models generally maintain strong performance under several perturbations; however, notable performance degradation is observed for surface-level noise (e.g., character-level corruption) and linguistic variation that preserves semantics while altering lexical or syntactic forms. Furthermore, we observe that surface-level noise causes larger performance drops in traditional pipelines, whereas linguistic variation presents greater challenges in agentic settings. These findings highlight the remaining challenges in achieving robust NL2SQL systems, particularly in handling linguistic variability. 

---
# Rubric-Guided Fine-tuning of SpeechLLMs for Multi-Aspect, Multi-Rater L2 Reading-Speech Assessment 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2603.16889)  

**Abstract**: Reliable and interpretable automated assessment of second-language (L2) speech remains a central challenge, as large speech-language models (SpeechLLMs) often struggle to align with the nuanced variability of human raters. To address this, we introduce a rubric-guided reasoning framework that explicitly encodes multi-aspect human assessment criteria: accuracy, fluency, and prosody, while calibrating model uncertainty to capture natural rating variability. We fine-tune the Qwen2-Audio-7B-Instruct model using multi-rater human judgments and develop an uncertainty-calibrated regression approach supported by conformal calibration for interpretable confidence intervals. Our Gaussian uncertainty modeling and conformal calibration approach achieves the strongest alignment with human ratings, outperforming regression and classification baselines. The model reliably assesses fluency and prosody while highlighting the inherent difficulty of assessing accuracy. Together, these results demonstrate that rubric-guided, uncertainty-calibrated reasoning offers a principled path toward trustworthy and explainable SpeechLLM-based speech assessment. 

---
# Enhancing Financial Report Question-Answering: A Retrieval-Augmented Generation System with Reranking Analysis 

**Authors**: Zhiyuan Cheng, Longying Lai, Yue Liu, Kai Cheng, Xiaoxi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16877)  

**Abstract**: Financial analysts face significant challenges extracting information from lengthy 10-K reports, which often exceed 100 pages. This paper presents a Retrieval-Augmented Generation (RAG) system designed to answer questions about S&P 500 financial reports and evaluates the impact of neural reranking on system performance. Our pipeline employs hybrid search combining full-text and semantic retrieval, followed by an optional reranking stage using a cross-encoder model. We conduct systematic evaluation using the FinDER benchmark dataset, comprising 1,500 queries across five experimental groups. Results demonstrate that reranking significantly improves answer quality, achieving 49.0 percent correctness for scores of 8 or above compared to 33.5 percent without reranking, representing a 15.5 percentage point improvement. Additionally, the error rate for completely incorrect answers decreases from 35.3 percent to 22.5 percent. Our findings emphasize the critical role of reranking in financial RAG systems and demonstrate performance improvements over baseline methods through modern language models and refined retrieval strategies. 

---
# Trust, Safety, and Accuracy: Assessing LLMs for Routine Maternity Advice 

**Authors**: V Sai Divya, A Bhanusree, Rimjhim, K Venkata Krishna Rao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16872)  

**Abstract**: Access to reliable maternal healthcare information is a major challenge in rural India due to limited medical resources and infrastructure. With over 830 million internet users and nearly half of rural women online, digital tools offer new opportunities for health education. This study evaluates large language models (LLMs) like ChatGPT-4o, Perplexity AI, and GeminiAI to provide reliable and understandable pregnancy-related information. Seventeen pregnancy-focused questions were posed to each model and compared with responses from maternal health professionals. Evaluations used semantic similarity, noun overlap, and readability metrics to measure content quality. Results show Perplexity closely matched expert semantics, while ChatGPT-4o produced clearer, more understandable text with better medical terminology. As internet access grows in rural areas, LLMs could serve as scalable aids for maternal health education. The study highlights the need for AI tools that balance accuracy and clarity to improve healthcare communication in underserved regions. 

---
# Loc3R-VLM: Language-based Localization and 3D Reasoning with Vision-Language Models 

**Authors**: Kevin Qu, Haozhe Qi, Mihai Dusmanu, Mahdi Rad, Rui Wang, Marc Pollefeys  

**Link**: [PDF](https://arxiv.org/pdf/2603.18002)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made impressive progress in connecting vision and language, but they still struggle with spatial understanding and viewpoint-aware reasoning. Recent efforts aim to augment the input representations with geometric cues rather than explicitly teaching models to reason in 3D space. We introduce Loc3R-VLM, a framework that equips 2D Vision-Language Models with advanced 3D understanding capabilities from monocular video input. Inspired by human spatial cognition, Loc3R-VLM relies on two joint objectives: global layout reconstruction to build a holistic representation of the scene structure, and explicit situation modeling to anchor egocentric perspective. These objectives provide direct spatial supervision that grounds both perception and language in a 3D context. To ensure geometric consistency and metric-scale alignment, we leverage lightweight camera pose priors extracted from a pre-trained 3D foundation model. Loc3R-VLM achieves state-of-the-art performance in language-based localization and outperforms existing 2D- and video-based approaches on situated and general 3D question-answering benchmarks, demonstrating that our spatial supervision framework enables strong 3D understanding. Project page: this https URL 

---
# Only relative ranks matter in weight-clustered large language models 

**Authors**: Borja Aizpurua, Sukhbinder Singh, Román Orús  

**Link**: [PDF](https://arxiv.org/pdf/2603.17917)  

**Abstract**: Large language models (LLMs) contain billions of parameters, yet many exact values are not essential. We show that what matters most is the relative rank of weights-whether one connection is stronger or weaker than another-rather than precise magnitudes. To reduce the number of unique weight values, we apply weight clustering to pretrained models, replacing every weight matrix with K shared values from K-means. For Llama 3.1-8B-Instruct and SmolLM2-135M, reducing each matrix to only 16-64 distinct values preserves strong accuracy without retraining, providing a simple, training-free method to compress LLMs on disk. Optionally fine-tuning only the cluster means (centroids) recovers 30-40 percent of the remaining accuracy gap at minimal cost. We then systematically randomize cluster means while keeping assignments fixed. Scrambling the relative ranks of the clusters degrades quality sharply-perplexity can increase by orders of magnitude-even when global statistics such as mean and variance are preserved. In contrast, rank-preserving randomizations cause almost no loss at mid and late layers. On the other hand, when many layers are perturbed simultaneously, progressive layer-by-layer replacement reveals that scale drift-not rank distortion-is the dominant collapse mechanism; however, an affine correction w' = aw + b with a > 0 (which preserves both rank order and overall weight distribution) can substantially delay this drift. This rank-based perspective offers a new lens on model compression and robustness. 

---
# The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning 

**Authors**: Donghang Wu, Tianyu Zhang, Yuxin Li, Hexin Liu, Chen Chen, Eng Siong Chng, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2603.17837)  

**Abstract**: During conversational interactions, humans subconsciously engage in concurrent thinking while listening to a speaker. Although this internal cognitive processing may not always manifest as explicit linguistic structures, it is instrumental in formulating high-quality responses. Inspired by this cognitive phenomenon, we propose a novel Full-duplex LAtent and Internal Reasoning method named FLAIR that conducts latent thinking simultaneously with speech perception. Unlike conventional "thinking" mechanisms in NLP, which require post-hoc generation, our approach aligns seamlessly with spoken dialogue systems: during the user's speaking phase, it recursively feeds the latent embedding output from the previous step into the next step, enabling continuous reasoning that strictly adheres to causality without introducing additional latency. To enable this latent reasoning, we design an Evidence Lower Bound-based objective that supports efficient supervised finetuning via teacher forcing, circumventing the need for explicit reasoning annotations. Experiments demonstrate the effectiveness of this think-while-listening design, which achieves competitive results on a range of speech benchmarks. Furthermore, FLAIR robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics. 

---
# CodeScout: An Effective Recipe for Reinforcement Learning of Code Search Agents 

**Authors**: Lintang Sutawika, Aditya Bharat Soni, Bharath Sriraam R R, Apurva Gandhi, Taha Yassine, Sanidhya Vijayvargiya, Yuchen Li, Xuhui Zhou, Yilin Zhang, Leander Melroy Maben, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2603.17829)  

**Abstract**: A prerequisite for coding agents to perform tasks on large repositories is code localization - the identification of relevant files, classes, and functions to work on. While repository-level code localization has been performed using embedding-based retrieval approaches such as vector search, recent work has focused on developing agents to localize relevant code either as a standalone precursor to or interleaved with performing actual work. Most prior methods on agentic code search equip the agent with complex, specialized tools, such as repository graphs derived from static analysis. In this paper, we demonstrate that, with an effective reinforcement learning recipe, a coding agent equipped with nothing more than a standard Unix terminal can be trained to achieve strong results. Our experiments on three benchmarks (SWE-Bench Verified, Pro, and Lite) reveal that our models consistently achieve superior or competitive performance over 2-18x larger base and post-trained LLMs and sometimes approach performance provided by closed models like Claude Sonnet, even when using specialized scaffolds. Our work particularly focuses on techniques for re-purposing existing coding agent environments for code search, reward design, and RL optimization. We release the resulting model family, CodeScout, along with all our code and data for the community to build upon. 

---
# Discovering Decoupled Functional Modules in Large Language Models 

**Authors**: Yanke Yu, Jin Li, Ying Sun, Ping Li, Zhefeng Wang, Yi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17823)  

**Abstract**: Understanding the internal functional organization of Large Language Models (LLMs) is crucial for improving their trustworthiness and performance. However, how LLMs organize different functions into modules remains highly unexplored. To bridge this gap, we formulate a functional module discovery problem and propose an Unsupervised LLM Cross-layer MOdule Discovery (ULCMOD) framework that simultaneously disentangles the large set of neurons in the entire LLM into modules while discovering the topics of input samples related to these modules. Our framework introduces a novel objective function and an efficient Iterative Decoupling (IterD) algorithm. Extensive experiments show that our method discovers high-quality, disentangled modules that capture more meaningful semantic information and achieve superior performance in various downstream tasks. Moreover, our qualitative analysis reveals that the discovered modules show semantic coherence, correspond to interpretable specializations, and a clear spatial and hierarchical organization within the LLM. Our work provides a novel tool for interpreting the functional modules of LLMs, filling a critical blank in LLM's interpretability research. 

---
# Multi-Source Evidence Fusion for Audio Question Answering 

**Authors**: Aivo Olev, Tanel Alumäe  

**Link**: [PDF](https://arxiv.org/pdf/2603.17822)  

**Abstract**: Large audio language models (LALMs) can answer questions about speech, music, and environmental sounds, yet their internal reasoning is largely opaque and difficult to validate. We describe TalTech's solution to the Agent Track of the Interspeech 2026 Audio Reasoning Challenge, in which systems are evaluated on reasoning process quality, specifically the factual accuracy, logical soundness, and completeness of their reasoning chains. Our multi-source ensemble pipeline uses two LALMs that generate independent observations, while a separate text-only reasoning model cross-checks these against outputs from 25 acoustic tools organized into reliability tiers. By grounding every inference step in explicit, reliability-tagged evidence, the system produces dense, verifiable reasoning chains. Our system ranked first in the challenge, outperforming all competing systems by a wide margin in challenge's reasoning quality metric. 

---
# Governed Memory: A Production Architecture for Multi-Agent Workflows 

**Authors**: Hamed Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2603.17787)  

**Abstract**: Enterprise AI deploys dozens of autonomous agent nodes across workflows, each acting on the same entities with no shared memory and no common governance. We identify five structural challenges arising from this memory governance gap: memory silos across agent workflows; governance fragmentation across teams and tools; unstructured memories unusable by downstream systems; redundant context delivery in autonomous multi-step executions; and silent quality degradation without feedback loops. We present Governed Memory, a shared memory and governance layer addressing this gap through four mechanisms: a dual memory model combining open-set atomic facts with schema-enforced typed properties; tiered governance routing with progressive context delivery; reflection-bounded retrieval with entity-scoped isolation; and a closed-loop schema lifecycle with AI-assisted authoring and automated per-property refinement. We validate each mechanism through controlled experiments (N=250, five content types): 99.6% fact recall with complementary dual-modality coverage; 92% governance routing precision; 50% token reduction from progressive delivery; zero cross-entity leakage across 500 adversarial queries; 100% adversarial governance compliance; and output quality saturation at approximately seven governed memories per entity. On the LoCoMo benchmark, the architecture achieves 74.8% overall accuracy, confirming that governance and schema enforcement impose no retrieval quality penalty. The system is in production at this http URL. 

---
# Modeling Overlapped Speech with Shuffles 

**Authors**: Matthew Wiesner, Samuele Cornell, Alexander Polok, Lucas Ondel Yang, Lukáš Burget, Sanjeev Khudanpur  

**Link**: [PDF](https://arxiv.org/pdf/2603.17769)  

**Abstract**: We propose to model parallel streams of data, such as overlapped speech, using shuffles. Specifically, this paper shows how the shuffle product and partial order finite-state automata (FSAs) can be used for alignment and speaker-attributed transcription of overlapped speech. We train using the total score on these FSAs as a loss function, marginalizing over all possible serializations of overlapping sequences at subword, word, and phrase levels. To reduce graph size, we impose temporal constraints by constructing partial order FSAs. We address speaker attribution by modeling (token, speaker) tuples directly. Viterbi alignment through the shuffle product FSA directly enables one-pass alignment. We evaluate performance on synthetic LibriSpeech overlaps. To our knowledge, this is the first algorithm that enables single-pass alignment of multi-talker recordings. All algorithms are implemented using k2 / Icefall. 

---
# Complementary Reinforcement Learning 

**Authors**: Dilxat Muhtar, Jiashun Liu, Wei Gao, Weixun Wang, Shaopan Xiong, Ju Huang, Siran Yang, Wenbo Su, Jiamang Wang, Ling Pan, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.17621)  

**Abstract**: Reinforcement Learning (RL) has emerged as a powerful paradigm for training LLM-based agents, yet remains limited by low sample efficiency, stemming not only from sparse outcome feedback but also from the agent's inability to leverage prior experience across episodes. While augmenting agents with historical experience offers a promising remedy, existing approaches suffer from a critical weakness: the experience distilled from history is either stored statically or fail to coevolve with the improving actor, causing a progressive misalignment between the experience and the actor's evolving capability that diminishes its utility over the course of training. Inspired by complementary learning systems in neuroscience, we present Complementary RL to achieve seamless co-evolution of an experience extractor and a policy actor within the RL optimization loop. Specifically, the actor is optimized via sparse outcome-based rewards, while the experience extractor is optimized according to whether its distilled experiences demonstrably contribute to the actor's success, thereby evolving its experience management strategy in lockstep with the actor's growing capabilities. Empirically, Complementary RL outperforms outcome-based agentic RL baselines that do not learn from experience, achieving 10% performance improvement in single-task scenarios and exhibits robust scalability in multi-task settings. These results establish Complementary RL as a paradigm for efficient experience-driven agent learning. 

---
# Temporal Narrative Monitoring in Dynamic Information Environments 

**Authors**: David Farr, Stephen Prochaska, Jack Moody, Lynnette Hui Xian Ng, Iain Cruickshank, Kate Starbird, Jevin West  

**Link**: [PDF](https://arxiv.org/pdf/2603.17617)  

**Abstract**: Comprehending the information environment (IE) during crisis events is challenging due to the rapid change and abstract nature of the domain. Many approaches focus on snapshots via classification methods or network approaches to describe the IE in crisis, ignoring the temporal nature of how information changed over time.
This work presents a system-oriented framework for modeling emerging narratives as temporally evolving semantic structures without requiring prior label specification. By integrating semantic embeddings, density-based clustering, and rolling temporal linkage, the framework represents narratives as persistent yet adaptive entities within a shared semantic space. We apply the methodology to a real-world crisis event and evaluate system behavior through stratified cluster validation and temporal lifecycle analysis. Results demonstrate high cluster coherence and reveal heterogeneous narrative lifecycles characterized by both transient fragments and stable narrative anchors.
We ground our approach in situational awareness theory, supporting perception and comprehension of the IE by transforming unstructured social media streams into interpretable, temporally structured representations. The resulting system provides a methodology for monitoring and decision support in dynamic information environments. 

---
# Modeling Changing Scientific Concepts with Complex Networks: A Case Study on the Chemical Revolution 

**Authors**: Sofía Aguilar-Valdez, Stefania Degaetano-Ortlieb  

**Link**: [PDF](https://arxiv.org/pdf/2603.17594)  

**Abstract**: While context embeddings produced by LLMs can be used to estimate conceptual change, these representations are often not interpretable nor time-aware. Moreover, bias augmentation in historical data poses a non-trivial risk to researchers in the Digital Humanities. Hence, to model reliable concept trajectories in evolving scholarship, in this work we develop a framework that represents prototypical concepts through complex networks based on topics. Utilizing the Royal Society Corpus, we analyzed two competing theories from the Chemical Revolution (phlogiston vs. oxygen) as a case study to show that onomasiological change is linked to higher entropy and topological density, indicating increased diversity of ideas and connectivity effort. 

---
# From Isolated Scoring to Collaborative Ranking: A Comparison-Native Framework for LLM-Based Paper Evaluation 

**Authors**: Pujun Zheng, Jiacheng Yao, Jinquan Zheng, Chenyang Gu, Guoxiu He, Jiawei Liu, Yong Huang, Tianrui Guo, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17588)  

**Abstract**: Large language models (LLMs) are currently applied to scientific paper evaluation by assigning an absolute score to each paper independently. However, since score scales vary across conferences, time periods, and evaluation criteria, models trained on absolute scores are prone to fitting narrow, context-specific rules rather than developing robust scholarly judgment. To overcome this limitation, we propose shifting paper evaluation from isolated scoring to collaborative ranking. In particular, we design \textbf{C}omparison-\textbf{N}ative framework for \textbf{P}aper \textbf{E}valuation (\textbf{CNPE}), integrating comparison into both data construction and model learning. We first propose a graph-based similarity ranking algorithm to facilitate the sampling of more informative and discriminative paper pairs from a collection. We then enhance relative quality judgment through supervised fine-tuning and reinforcement learning with comparison-based rewards. At inference, the model performs pairwise comparisons over sampled paper pairs and aggregates these preference signals into a global relative quality ranking. Experimental results demonstrate that our framework achieves an average relative improvement of \textbf{21.8\%} over the strong baseline DeepReview-14B, while exhibiting robust generalization to five previously unseen datasets. \href{this https URL}{Code}. 

---
# UniSAFE: A Comprehensive Benchmark for Safety Evaluation of Unified Multimodal Models 

**Authors**: Segyu Lee, Boryeong Cho, Hojung Jung, Seokhyun An, Juhyeong Kim, Jaehyun Kwak, Yongjin Yang, Sangwon Jang, Youngrok Park, Wonjun Chang, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2603.17476)  

**Abstract**: Unified Multimodal Models (UMMs) offer powerful cross-modality capabilities but introduce new safety risks not observed in single-task models. Despite their emergence, existing safety benchmarks remain fragmented across tasks and modalities, limiting the comprehensive evaluation of complex system-level vulnerabilities. To address this gap, we introduce UniSAFE, the first comprehensive benchmark for system-level safety evaluation of UMMs across 7 I/O modality combinations, spanning conventional tasks and novel multimodal-context image generation settings. UniSAFE is built with a shared-target design that projects common risk scenarios across task-specific I/O configurations, enabling controlled cross-task comparisons of safety failures. Comprising 6,802 curated instances, we use UniSAFE to evaluate 15 state-of-the-art UMMs, both proprietary and open-source. Our results reveal critical vulnerabilities across current UMMs, including elevated safety violations in multi-image composition and multi-turn settings, with image-output tasks consistently more vulnerable than text-output tasks. These findings highlight the need for stronger system-level safety alignment for UMMs. Our code and data are publicly available at this https URL 

---
# When Only the Final Text Survives: Implicit Execution Tracing for Multi-Agent Attribution 

**Authors**: Yi Nian, Haosen Cao, Shenzhe Zhu, Henry Peng Zou, Qingqing Luan, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.17445)  

**Abstract**: When a multi-agent system produces an incorrect or harmful answer, who is accountable if execution logs and agent identifiers are unavailable? Multi-agent language systems increasingly rely on structured interactions such as delegation and iterative refinement, yet the final output often obscures the underlying interaction topology and agent contributions. We introduce IET (Implicit Execution Tracing), a metadata-independent framework that enables token-level attribution directly from generated text and a simple mechanism for interaction topology reconstruction. During generation, agent-specific keyed signals are embedded into the token distribution, transforming the text into a self-describing execution trace detectable only with a secret key. At detection time, a transition-aware scoring method identifies agent handover points and reconstructs the interaction graph. Experiments show that IET recovers agent segments and coordination structure with high accuracy while preserving generation quality, enabling privacy-preserving auditing for multi-agent language systems. 

---
# PJB: A Reasoning-Aware Benchmark for Person-Job Retrieval 

**Authors**: Guangzhi Wang, Xiaohui Yang, Kai Li, Jiawen He, Kai Yang, Ruixuan Zhang, Zhi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.17386)  

**Abstract**: As retrieval models converge on generic benchmarks, the pressing question is no longer "who scores higher" but rather "where do systems fail, and why?" Person-job matching is a domain that urgently demands such diagnostic capability -- it requires systems not only to verify explicit constraints but also to perform skill-transfer inference and job-competency reasoning, yet existing benchmarks provide no systematic diagnostic support for this task. We introduce PJB (Person-Job Benchmark), a reasoning-aware retrieval evaluation dataset that uses complete job descriptions as queries and complete resumes as documents, defines relevance through job-competency judgment, is grounded in real-world recruitment data spanning six industry domains and nearly 200,000 resumes, and upgrades evaluation from "who scores higher" to "where do systems differ, and why" through domain-family and reasoning-type diagnostic labels. Diagnostic experiments using dense retrieval reveal that performance heterogeneity across industry domains far exceeds the gains from module upgrades for the same model, indicating that aggregate scores alone can severely mislead optimization decisions. At the module level, reranking yields stable improvements while query understanding not only fails to help but actually degrades overall performance when combined with reranking -- the two modules face fundamentally different improvement bottlenecks. The value of PJB lies not in yet another leaderboard of average scores, but in providing recruitment retrieval systems with a capability map that pinpoints where to invest. 

---
# Public Profile Matters: A Scalable Integrated Approach to Recommend Citations in the Wild 

**Authors**: Karan Goyal, Dikshant Kukreja, Vikram Goyal, Mukesh Mohania  

**Link**: [PDF](https://arxiv.org/pdf/2603.17361)  

**Abstract**: Proper citation of relevant literature is essential for contextualising and validating scientific contributions. While current citation recommendation systems leverage local and global textual information, they often overlook the nuances of the human citation behaviour. Recent methods that incorporate such patterns improve performance but incur high computational costs and introduce systematic biases into downstream rerankers. To address this, we propose Profiler, a lightweight, non-learnable module that captures human citation patterns efficiently and without bias, significantly enhancing candidate retrieval. Furthermore, we identify a critical limitation in current evaluation protocol: the systems are assessed in a transductive setting, which fails to reflect real-world scenarios. We introduce a rigorous Inductive evaluation setting that enforces strict temporal constraints, simulating the recommendation of citations for newly authored papers in the wild. Finally, we present DAVINCI, a novel reranking model that integrates profiler-derived confidence priors with semantic information via an adaptive vector-gating mechanism. Our system achieves new state-of-the-art results across multiple benchmark datasets, demonstrating superior efficiency and generalisability. 

---
# Beyond Outliers: A Data-Free Layer-wise Mixed-Precision Quantization Approach Driven by Numerical and Structural Dual-Sensitivity 

**Authors**: Hengyuan Zhang, Xinrong Chen, Zunhai Su, Xiao Liang, Jing Xiong, Wendong Xu, He Xiao, Chaofan Tao, Wei Zhang, Ruobing Xie, Lei Jiang, Hayden Kwok-Hay So, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2603.17354)  

**Abstract**: Layer-wise mixed-precision quantization (LMPQ) enables effective compression under extreme low-bit settings by allocating higher precision to sensitive layers. However, existing methods typically treat all intra-layer weight modules uniformly and rely on a single numerical property when estimating sensitivity, overlooking their distinct operational roles and structural characteristics. To address this, we propose NSDS, a novel calibration-free LMPQ framework driven by Numerical and Structural Dual-Sensitivity. Specifically, it first mechanistically decomposes each layer into distinct operational roles and quantifies their sensitivity from both numerical and structural perspectives. These dual-aspect scores are then aggregated into a unified layer-wise metric through a robust aggregation scheme based on MAD-Sigmoid and Soft-OR to guide bit allocation. Extensive experiments demonstrate that NSDS consistently achieves superior performance compared to various baselines across diverse models and downstream tasks, without relying on any calibration data. 

---
# InfoDensity: Rewarding Information-Dense Traces for Efficient Reasoning 

**Authors**: Chengwei Wei, Jung-jae Kim, Longyin Zhang, Shengkai Chen, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17310)  

**Abstract**: Large Language Models (LLMs) with extended reasoning capabilities often generate verbose and redundant reasoning traces, incurring unnecessary computational cost. While existing reinforcement learning approaches address this by optimizing final response length, they neglect the quality of intermediate reasoning steps, leaving models vulnerable to reward hacking. We argue that verbosity is not merely a length problem, but a symptom of poor intermediate reasoning quality. To investigate this, we conduct an empirical study tracking the conditional entropy of the answer distribution across reasoning steps. We find that high-quality reasoning traces exhibit two consistent properties: low uncertainty convergence and monotonic progress. These findings suggest that high-quality reasoning traces are informationally dense, that is, each step contributes meaningful entropy reduction relative to the total reasoning length. Motivated by this, we propose InfoDensity, a reward framework for RL training that combines an AUC-based reward and a monotonicity reward as a unified measure of reasoning quality, weighted by a length scaling term that favors achieving equivalent quality more concisely. Experiments on mathematical reasoning benchmarks demonstrate that InfoDensity matches or surpasses state-of-the-art baselines in accuracy while significantly reducing token usage, achieving a strong accuracy-efficiency trade-off. 

---
# Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations 

**Authors**: Haozheng Luo, Yimin Wang, Jiahao Yu, Binghui Wang, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.17305)  

**Abstract**: We propose CRAFT, a red-teaming alignment framework that leverages model reasoning capabilities and hidden representations to improve robustness against jailbreak attacks. Unlike prior defenses that operate primarily at the output level, CRAFT aligns large reasoning models to generate safety-aware reasoning traces by explicitly optimizing objectives defined over the hidden state space. Methodologically, CRAFT integrates contrastive representation learning with reinforcement learning to separate safe and unsafe reasoning trajectories, yielding a latent-space geometry that supports robust, reasoning-level safety alignment. Theoretically, we show that incorporating latent-textual consistency into GRPO eliminates superficially aligned policies by ruling them out as local optima. Empirically, we evaluate CRAFT on multiple safety benchmarks using two strong reasoning models, Qwen3-4B-Thinking and R1-Distill-Llama-8B, where it consistently outperforms state-of-the-art defenses such as IPO and SafeKey. Notably, CRAFT delivers an average 79.0% improvement in reasoning safety and 87.7% improvement in final-response safety over the base models, demonstrating the effectiveness of hidden-space reasoning alignment. 

---
# LED: A Benchmark for Evaluating Layout Error Detection in Document Analysis 

**Authors**: Inbum Heo, Taewook Hwang, Jeesu Jung, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2603.17265)  

**Abstract**: Recent advances in Large Language Models (LLMs) and Large Multimodal Models (LMMs) have improved Document Layout Analysis (DLA), yet structural errors such as region merging, splitting, and omission remain persistent. Conventional overlap-based metrics (e.g., IoU, mAP) fail to capture such logical inconsistencies. To overcome this limitation, we propose Layout Error Detection (LED), a benchmark that evaluates structural reasoning in DLA predictions beyond surface-level accuracy. LED defines eight standardized error types (Missing, Hallucination, Size Error, Split, Merge, Overlap, Duplicate, and Misclassification) and provides quantitative rules and injection algorithms for realistic error simulation. Using these definitions, we construct LED-Dataset and design three evaluation tasks: document-level error detection, document-level error-type classification, and element-level error-type classification. Experiments with state-of-the-art multimodal models show that LED enables fine-grained and interpretable assessment of structural understanding, revealing clear weaknesses across modalities and architectures. Overall, LED establishes a unified and explainable benchmark for diagnosing the structural robustness and reasoning capability of document understanding models. 

---
# OPERA: Online Data Pruning for Efficient Retrieval Model Adaptation 

**Authors**: Haoyang Fang, Shuai Zhang, Yifei Ma, Hengyi Wang, Cuixiong Hu, Katrin Kirchhoff, Bernie Wang, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2603.17205)  

**Abstract**: Domain-specific finetuning is essential for dense retrievers, yet not all training pairs contribute equally to the learning process. We introduce OPERA, a data pruning framework that exploits this heterogeneity to improve both the effectiveness and efficiency of retrieval model adaptation. We first investigate static pruning (SP), which retains only high-similarity query-document pairs, revealing an intrinsic quality-coverage tradeoff: ranking (NDCG) improves while retrieval (Recall) can degrade due to reduced query diversity. To resolve this tradeoff, we propose a two-stage dynamic pruning (DP) strategy that adaptively modulates sampling probabilities at both query and document levels throughout training, prioritizing high-quality examples while maintaining access to the full training set. Evaluations across eight datasets spanning six domains demonstrate the effectiveness of both approaches: SP improves ranking over standard finetuning (NDCG@10 +0.5\%), while DP achieves the strongest performance on both ranking (NDCG@10 +1.9\%) and retrieval (Recall@20 +0.7\%), with an average rank of 1.38 across all methods. These findings scale to Qwen3-Embedding, an LLM-based dense retriever, confirming architecture-agnostic benefits. Notably, DP reaches comparable performance in less than 50\% of the training time required by standard finetuning. 

---
# Catching rationalization in the act: detecting motivated reasoning before and after CoT via activation probing 

**Authors**: Parsa Mirtaheri, Mikhail Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2603.17199)  

**Abstract**: Large language models (LLMs) can produce chains of thought (CoT) that do not accurately reflect the actual factors driving their answers. In multiple-choice settings with an injected hint favoring a particular option, models may shift their final answer toward the hinted option and produce a CoT that rationalizes the response without acknowledging the hint - an instance of motivated reasoning. We study this phenomenon across multiple LLM families and datasets demonstrating that motivated reasoning can be identified by probing internal activations even in cases when it cannot be easily determined from CoT. Using supervised probes trained on the model's residual stream, we show that (i) pre-generation probes, applied before any CoT tokens are generated, predict motivated reasoning as well as a LLM-based CoT monitor that accesses the full CoT trace, and (ii) post-generation probes, applied after CoT generation, outperform the same monitor. Together, these results show that motivated reasoning is detected more reliably from internal representations than from CoT monitoring. Moreover, pre-generation probing can flag motivated behavior early, potentially avoiding unnecessary generation. 

---
# Abstraction as a Memory-Efficient Inductive Bias for Continual Learning 

**Authors**: Elnaz Rahmati, Nona Ghazizadeh, Zhivar Sourati, Nina Rouhani, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2603.17198)  

**Abstract**: The real world is non-stationary and infinitely complex, requiring intelligent agents to learn continually without the prohibitive cost of retraining from scratch. While online continual learning offers a framework for this setting, learning new information often interferes with previously acquired knowledge, causes forgetting and degraded generalization. To address this, we propose Abstraction-Augmented Training (AAT), a loss-level modification encouraging models to capture the latent relational structure shared across examples. By jointly optimizing over concrete instances and their abstract representations, AAT introduces a memory-efficient inductive bias that stabilizes learning in strictly online data streams, eliminating the need for a replay buffer. To capture the multi-faceted nature of abstraction, we introduce and evaluate AAT on two benchmarks: a controlled relational dataset where abstraction is realized through entity masking, and a narrative dataset where abstraction is expressed through shared proverbs. Our results show that AAT achieves performance comparable to or exceeding strong experience replay (ER) baselines, despite requiring zero additional memory and only minimal changes to the training objective. This work highlights structural abstraction as a powerful, memory-free alternative to ER. 

---
# How Clued up are LLMs? Evaluating Multi-Step Deductive Reasoning in a Text-Based Game Environment 

**Authors**: Rebecca Ansell, Autumn Toney-Wails  

**Link**: [PDF](https://arxiv.org/pdf/2603.17169)  

**Abstract**: Deducing whodunit proves challenging for LLM agents. In this paper, we implement a text-based multi-agent version of the classic board game Clue as a rule-based testbed for evaluating multi-step deductive reasoning, with six agents drawn from GPT-4o-mini and Gemini-2.5-Flash. We further investigate whether fine-tuning on structured logic puzzles transfers to improved in-game reasoning and gameplay. Across 18 simulated games, agents achieve only four correct wins, indicating difficulty in maintaining consistent deductive reasoning over the course of a full game. Additionally, we find that fine-tuning does not reliably improve performance and, in some cases, appears to increase reasoning volume without improving reasoning precision. 

---
# Multilingual Reference Need Assessment System for Wikipedia 

**Authors**: Aitolkyn Baigutanova, Francisco Navas, Pablo Aragon, Mykola Trokhymovych, Muniza Aslam, Ai-Jou Chou, Miriam Redi, Diego Saez-Trumper  

**Link**: [PDF](https://arxiv.org/pdf/2603.17146)  

**Abstract**: Wikipedia is a critical source of information for millions of users across the Web. It serves as a key resource for large language models, search engines, question-answering systems, and other Web-based applications. In Wikipedia, content needs to be verifiable, meaning that readers can check that claims are backed by references to reliable sources. This depends on manual verification by editors, an effective but labor-intensive process, especially given the high volume of daily edits. To address this challenge, we introduce a multilingual machine learning system to assist editors in identifying claims requiring citations. Our approach is tested in 10 language editions of Wikipedia, outperforming existing benchmarks for reference need assessment. We not only consider machine learning evaluation metrics but also system requirements, allowing us to explore the trade-offs between model accuracy and computational efficiency under real-world infrastructure constraints. We deploy our system in production and release data and code to support further research. 

---
# HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning 

**Authors**: Shenzhi Wang, Shixuan Liu, Jing Zhou, Chang Gao, Xiong-Hui Chen, Binghai Wang, An Yang, Shiji Song, Bowen Yu, Gao Huang, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.17024)  

**Abstract**: VLMs show strong multimodal capabilities, but they still struggle with fine-grained vision-language reasoning. We find that long CoT reasoning exposes diverse failure modes, including perception, reasoning, knowledge, and hallucination errors, which can compound across intermediate steps. However, most existing vision-language data used for RLVR does not involve complex reasoning chains that rely on visual evidence throughout, leaving these weaknesses largely unexposed. We therefore propose HopChain, a scalable framework for synthesizing multi-hop vision-language reasoning data specifically for RLVR training of VLMs. Each synthesized multi-hop query forms a logically dependent chain of instance-grounded hops, where earlier hops establish the instances, sets, or conditions needed for later hops, while the final answer remains a specific, unambiguous number suitable for verifiable rewards. We add the multi-hop data synthesized by HopChain to the original RLVR data used to train Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, and compare against RLVR on the original RLVR data alone across 24 benchmarks spanning STEM and Puzzle, General VQA, Text Recognition and Document Understanding, and Video Understanding. Although this multi-hop data is not synthesized to target any specific benchmark, adding it improves 20 out of 24 benchmarks on both models, indicating broad and generalizable gains. To demonstrate that full chained queries are important, we replace them with half-multi-hop or single-hop variants, reducing the 24-benchmark average accuracy by 5.3 and 7.0 points, respectively. Multi-hop training also strengthens long-CoT vision-language reasoning, with gains peaking at more than 50 accuracy points in the ultra-long-CoT regime. These experiments establish HopChain as an effective, scalable framework for synthesizing multi-hop data that improves generalizable vision-language reasoning. 

---
# The Voice Behind the Words: Quantifying Intersectional Bias in SpeechLLMs 

**Authors**: Shree Harsha Bokkahalli Satish, Christoph Minixhofer, Maria Teleki, James Caverlee, Ondřej Klejch, Peter Bell, Gustav Eje Henter, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2603.16941)  

**Abstract**: Speech Large Language Models (SpeechLLMs) process spoken input directly, retaining cues such as accent and perceived gender that were previously removed in cascaded pipelines. This introduces speaker identity dependent variation in responses. We present a large-scale intersectional evaluation of accent and gender bias in three SpeechLLMs using 2,880 controlled interactions across six English accents and two gender presentations, keeping linguistic content constant through voice cloning. Using pointwise LLM-judge ratings, pairwise comparisons, and Best-Worst Scaling with human validation, we detect consistent disparities. Eastern European-accented speech receives lower helpfulness scores, particularly for female-presenting voices. The bias is implicit: responses remain polite but differ in helpfulness. While LLM judges capture the directional trend of these biases, human evaluators exhibit significantly higher sensitivity, uncovering sharper intersectional disparities. 

---
# MHPO: Modulated Hazard-aware Policy Optimization for Stable Reinforcement Learning 

**Authors**: Hongjun Wang, Wei Liu, Weibo Gu, Xing Sun, Kai Han  

**Link**: [PDF](https://arxiv.org/pdf/2603.16929)  

**Abstract**: Regulating the importance ratio is critical for the training stability of Group Relative Policy Optimization (GRPO) based frameworks. However, prevailing ratio control methods, such as hard clipping, suffer from non-differentiable boundaries and vanishing gradient regions, failing to maintain gradient fidelity. Furthermore, these methods lack a hazard-aware mechanism to adaptively suppress extreme deviations, leaving the optimization process vulnerable to abrupt policy shifts. To address these challenges, we propose Modulated Hazard-aware Policy Optimization (MHPO), a novel framework designed for robust and stable reinforcement learning. The proposed MHPO introduces a Log-Fidelity Modulator (LFM) to map unbounded importance ratios into a bounded, differentiable domain. This mechanism effectively prevents high-variance outlier tokens from destabilizing the loss landscape while ensuring global gradient stability. Complementarily, a Decoupled Hazard Penalty (DHP) integrates cumulative hazard functions from survival analysis to independently regulate positive and negative policy shifts. By shaping the optimization landscape with hazard-aware penalties, the proposed MHPO achieves fine-grained regulation of asymmetric policy shifts simultaneously mitigating mode collapse from over-expansion and preventing policy erosion from catastrophic contraction within a stabilized trust region. Extensive evaluations on diverse reasoning benchmarks across both text-based and vision-language tasks demonstrate that MHPO consistently outperforms existing methods, achieving superior performance while significantly enhancing training stability. 

---
# SimulU: Training-free Policy for Long-form Simultaneous Speech-to-Speech Translation 

**Authors**: Amirbek Djanibekov, Luisa Bentivogli, Matteo Negri, Sara Papi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16924)  

**Abstract**: Simultaneous speech-to-speech translation (SimulS2S) is essential for real-time multilingual communication, with increasing integration into meeting and streaming platforms. Despite this, SimulS2S remains underexplored in research, where current solutions often rely on resource-intensive training procedures and operate on short-form, pre-segmented utterances, failing to generalize to continuous speech. To bridge this gap, we propose SimulU, the first training-free policy for long-form SimulS2S. SimulU adopts history management and speech output selection strategies that exploit cross-attention in pre-trained end-to-end models to regulate both input history and output generation. Evaluations on MuST-C across 8 languages show that SimulU achieves a better or comparable quality-latency trade-off against strong cascaded models. By eliminating the need for ad-hoc training, SimulU offers a promising path to end-to-end SimulS2S in realistic, long-form scenarios. 

---
# Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection 

**Authors**: Jinyang Wu, Zihan Pan, Qiquan Zhang, Sailor Hardik Bhupendra, Soumik Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2603.16914)  

**Abstract**: Neural audio codecs discretize speech via residual vector quantization (RVQ), forming a coarse-to-fine hierarchy across quantizers. While codec models have been explored for representation learning, their discrete structure remains underutilized in speech deepfake detection. In particular, different quantization levels capture complementary acoustic cues, where early quantizers encode coarse structure and later quantizers refine residual details that reveal synthesis artifacts. Existing systems either rely on continuous encoder features or ignore this quantizer-level hierarchy. We propose a hierarchy-aware representation learning framework that models quantizer-level contributions through learnable global weighting, enabling structured codec representations aligned with forensic cues. Keeping the speech encoder backbone frozen and updating only 4.4% additional parameters, our method achieves relative EER reductions of 46.2% on ASVspoof 2019 and 13.9% on ASVspoof5 over strong baselines. 

---
# EEG-Based Brain-LLM Interface for Human Preference Aligned Generation 

**Authors**: Junzi Zhang, Jianing Shen, Weijie Tu, Yi Zhang, Hailin Zhang, Tom Gedeon, Bin Jiang, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2603.16897)  

**Abstract**: Large language models (LLMs) are becoming an increasingly important component of human--computer interaction, enabling users to coordinate a wide range of intelligent agents through natural language. While language-based interfaces are powerful and flexible, they implicitly assume that users can reliably produce explicit linguistic input, an assumption that may not hold for users with speech or motor impairments, e.g., Amyotrophic Lateral Sclerosis (ALS). In this work, we investigate whether neural signals can be used as an alternative input to LLMs, particularly to support those socially marginalized or underserved users. We build a simple brain-LLM interface, which uses EEG signals to guide image generation models at test time. Specifically, we first train a classifier to estimate user satisfaction from EEG signals. Its predictions are then incorporated into a test-time scaling (TTS) framework that dynamically adapts model inference using neural feedback collected during user evaluation. The experiments show that EEG can predict user satisfaction, suggesting that neural activity carries information on real-time preference inference. These findings provide a first step toward integrating neural feedback into adaptive language-model inference, and hopefully open up new possibilities for future research on adaptive LLM interaction. 

---
# DECODE: Dual-Enhanced Conditioned Diffusion for EEG Forecasting 

**Authors**: Mehran Shabanpour, Sadaf Khademi, Konstantinos N Plataniotis, Arash Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16885)  

**Abstract**: Forecasting Electroncephalography (EEG) signals during cognitive events remains a fundamental challenge in neuroscience and Brain-Computer Interfaces (BCIs), as existing methods struggle to capture both the stochastic nature of neural dynamics and the semantic context of behavioral tasks. We present the Dual-Enhanced COnditioned Diffusion (DECODE) for EEG, a novel framework that unifies semantic guidance from natural language descriptions with temporal dynamics from historical signals to generate event-specific neural responses. DECODE leverages pre-trained language models to condition the diffusion process on rich textual descriptions of cognitive events, while maintaining temporal coherence through history-based Langevin dynamics. Evaluated on a real-world driving task dataset with five distinct behaviors, DECODE achieves sub-microvolt prediction accuracy (MAE = 0.626 microvolt) over 75 timestep horizons while maintaining well-calibrated uncertainty estimates. Our framework demonstrates that natural language can effectively bridge high-level cognitive descriptions and low-level neural dynamics, opening new possibilities for zero-shot generalization to novel behaviors and interpretable BCIs. By generating physiologically plausible, event-specific EEG trajectories conditioned on semantic descriptions, DECODE establishes a new paradigm for understanding and predicting context-dependent neural activity. 

---
# Tokenization vs. Augmentation: A Systematic Study of Writer Variance in IMU-Based Online Handwriting Recognition 

**Authors**: Jindong Li, Dario Zanca, Vincent Christlein, Tim Hamann, Jens Barth, Peter Kämpf, Björn Eskofier  

**Link**: [PDF](https://arxiv.org/pdf/2603.16883)  

**Abstract**: Inertial measurement unit-based online handwriting recognition enables the recognition of input signals collected across different writing surfaces but remains challenged by uneven character distributions and inter-writer variability. In this work, we systematically investigate two strategies to address these issues: sub-word tokenization and concatenation-based data augmentation. Our experiments on the OnHW-Words500 dataset reveal a clear dichotomy between handling inter-writer and intra-writer variance. On the writer-independent split, structural abstraction via Bigram tokenization significantly improves performance to unseen writing styles, reducing the word error rate (WER) from 15.40% to 12.99%. In contrast, on the writer-dependent split, tokenization degrades performance due to vocabulary distribution shifts between the training and validation sets. Instead, our proposed concatenation-based data augmentation acts as a powerful regularizer, reducing the character error rate by 34.5% and the WER by 25.4%. Further analysis shows that short, low-level tokens benefit model performance and that concatenation-based data augmentation performance gain surpasses those achieved by proportionally extended training. These findings reveal a clear variance-dependent effect: sub-word tokenization primarily mitigates inter-writer stylistic variability, whereas concatenation-based data augmentation effectively compensates for intra-writer distributional sparsity. 

---
# NeuroNarrator: A Generalist EEG-to-Text Foundation Model for Clinical Interpretation via Spectro-Spatial Grounding and Temporal State-Space Reasoning 

**Authors**: Guoan Wang, Shihao Yang, Jun-en Ding, Hao Zhu, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.16880)  

**Abstract**: Electroencephalography (EEG) provides a non-invasive window into neural dynamics at high temporal resolution and plays a pivotal role in clinical neuroscience research. Despite this potential, prevailing computational approaches to EEG analysis remain largely confined to task-specific classification objectives or coarse-grained pattern recognition, offering limited support for clinically meaningful interpretation. To address these limitations, we introduce NeuroNarrator, the first generalist EEG-to-text foundation model designed to translate electrophysiological segments into precise clinical narratives. A cornerstone of this framework is the curation of NeuroCorpus-160K, the first harmonized large-scale resource pairing over 160,000 EEG segments with structured, clinically grounded natural-language descriptions. Our architecture first aligns temporal EEG waveforms with spatial topographic maps via a rigorous contrastive objective, establishing spectro-spatially grounded representations. Building on this grounding, we condition a Large Language Model through a state-space-inspired formulation that integrates historical temporal and spectral context to support coherent clinical narrative generation. This approach establishes a principled bridge between continuous signal dynamics and discrete clinical language, enabling interpretable narrative generation that facilitates expert interpretation and supports clinical reporting workflows. Extensive evaluations across diverse benchmarks and zero-shot transfer tasks highlight NeuroNarrator's capacity to integrate temporal, spectral, and spatial dynamics, positioning it as a foundational framework for time-frequency-aware, open-ended clinical interpretation of electrophysiological data. 

---
