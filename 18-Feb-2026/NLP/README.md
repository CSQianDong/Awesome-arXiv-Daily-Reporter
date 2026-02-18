# Avey-B 

**Authors**: Devang Acharya, Mohammad Hammoud  

**Link**: [PDF](https://arxiv.org/pdf/2602.15814)  

**Abstract**: Compact pretrained bidirectional encoders remain the backbone of industrial NLP under tight compute and memory budgets. Their effectiveness stems from self-attention's ability to deliver high-quality bidirectional contextualization with sequence-level parallelism, as popularized by BERT-style architectures. Recently, Avey was introduced as an autoregressive, attention-free alternative that naturally admits an encoder-only adaptation. In this paper, we reformulate Avey for the encoder-only paradigm and propose several innovations to its architecture, including decoupled static and dynamic parameterizations, stability-oriented normalization, and neural compression. Results show that this reformulated architecture compares favorably to four widely used Transformer-based encoders, consistently outperforming them on standard token-classification and information-retrieval benchmarks while scaling more efficiently to long contexts. 

---
# *-PLUIE: Personalisable metric with Llm Used for Improved Evaluation 

**Authors**: Quentin Lemesle, Léane Jourdan, Daisy Munson, Pierre Alain, Jonathan Chevelu, Arnaud Delhay, Damien Lolive  

**Link**: [PDF](https://arxiv.org/pdf/2602.15778)  

**Abstract**: Evaluating the quality of automatically generated text often relies on LLM-as-a-judge (LLM-judge) methods. While effective, these approaches are computationally expensive and require post-processing. To address these limitations, we build upon ParaPLUIE, a perplexity-based LLM-judge metric that estimates confidence over ``Yes/No'' answers without generating text. We introduce *-PLUIE, task specific prompting variants of ParaPLUIE and evaluate their alignment with human judgement. Our experiments show that personalised *-PLUIE achieves stronger correlations with human ratings while maintaining low computational cost. 

---
# ViTaB-A: Evaluating Multimodal Large Language Models on Visual Table Attribution 

**Authors**: Yahia Alqurnawi, Preetom Biswas, Anmol Rao, Tejas Anvekar, Chitta Baral, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2602.15769)  

**Abstract**: Multimodal Large Language Models (mLLMs) are often used to answer questions in structured data such as tables in Markdown, JSON, and images. While these models can often give correct answers, users also need to know where those answers come from. In this work, we study structured data attribution/citation, which is the ability of the models to point to the specific rows and columns that support an answer. We evaluate several mLLMs across different table formats and prompting strategies. Our results show a clear gap between question answering and evidence attribution. Although question answering accuracy remains moderate, attribution accuracy is much lower, near random for JSON inputs, across all models. We also find that models are more reliable at citing rows than columns, and struggle more with textual formats than images. Finally, we observe notable differences across model families. Overall, our findings show that current mLLMs are unreliable at providing fine-grained, trustworthy attribution for structured data, which limits their usage in applications requiring transparency and traceability. 

---
# ChartEditBench: Evaluating Grounded Multi-Turn Chart Editing in Multimodal Language Models 

**Authors**: Manav Nitin Kapadnis, Lawanya Baghel, Atharva Naik, Carolyn Rosé  

**Link**: [PDF](https://arxiv.org/pdf/2602.15758)  

**Abstract**: While Multimodal Large Language Models (MLLMs) perform strongly on single-turn chart generation, their ability to support real-world exploratory data analysis remains underexplored. In practice, users iteratively refine visualizations through multi-turn interactions that require maintaining common ground, tracking prior edits, and adapting to evolving preferences. We introduce ChartEditBench, a benchmark for incremental, visually grounded chart editing via code, comprising 5,000 difficulty-controlled modification chains and a rigorously human-verified subset. Unlike prior one-shot benchmarks, ChartEditBench evaluates sustained, context-aware editing. We further propose a robust evaluation framework that mitigates limitations of LLM-as-a-Judge metrics by integrating execution-based fidelity checks, pixel-level visual similarity, and logical code verification. Experiments with state-of-the-art MLLMs reveal substantial degradation in multi-turn settings due to error accumulation and breakdowns in shared context, with strong performance on stylistic edits but frequent execution failures on data-centric transformations. ChartEditBench, establishes a challenging testbed for grounded, intent-aware multimodal programming. 

---
# Beyond Binary Classification: Detecting Fine-Grained Sexism in Social Media Videos 

**Authors**: Laura De Grazia, Danae Sánchez Villegas, Desmond Elliott, Mireia Farrús, Mariona Taulé  

**Link**: [PDF](https://arxiv.org/pdf/2602.15757)  

**Abstract**: Online sexism appears in various forms, which makes its detection challenging. Although automated tools can enhance the identification of sexist content, they are often restricted to binary classification. Consequently, more subtle manifestations of sexism may remain undetected due to the lack of fine-grained, context-sensitive labels. To address this issue, we make the following contributions: (1) we present FineMuSe, a new multimodal sexism detection dataset in Spanish that includes both binary and fine-grained annotations; (2) we introduce a comprehensive hierarchical taxonomy that encompasses forms of sexism, non-sexism, and rhetorical devices of irony and humor; and (3) we evaluate a wide range of LLMs for both binary and fine-grained sexism detection. Our findings indicate that multimodal LLMs perform competitively with human annotators in identifying nuanced forms of sexism; however, they struggle to capture co-occurring sexist types when these are conveyed through visual cues. 

---
# Under-resourced studies of under-resourced languages: lemmatization and POS-tagging with LLM annotators for historical Armenian, Georgian, Greek and Syriac 

**Authors**: Chahan Vidal-Gorène, Bastien Kindt, Florian Cafiero  

**Link**: [PDF](https://arxiv.org/pdf/2602.15753)  

**Abstract**: Low-resource languages pose persistent challenges for Natural Language Processing tasks such as lemmatization and part-of-speech (POS) tagging. This paper investigates the capacity of recent large language models (LLMs), including GPT-4 variants and open-weight Mistral models, to address these tasks in few-shot and zero-shot settings for four historically and linguistically diverse under-resourced languages: Ancient Greek, Classical Armenian, Old Georgian, and Syriac. Using a novel benchmark comprising aligned training and out-of-domain test corpora, we evaluate the performance of foundation models across lemmatization and POS-tagging, and compare them with PIE, a task-specific RNN baseline. Our results demonstrate that LLMs, even without fine-tuning, achieve competitive or superior performance in POS-tagging and lemmatization across most languages in few-shot settings. Significant challenges persist for languages characterized by complex morphology and non-Latin scripts, but we demonstrate that LLMs are a credible and relevant option for initiating linguistic annotation tasks in the absence of data, serving as an effective aid for annotation. 

---
# Causal Effect Estimation with Latent Textual Treatments 

**Authors**: Omri Feldman, Amar Venugopal, Jann Spiess, Amir Feder  

**Link**: [PDF](https://arxiv.org/pdf/2602.15730)  

**Abstract**: Understanding the causal effects of text on downstream outcomes is a central task in many applications. Estimating such effects requires researchers to run controlled experiments that systematically vary textual features. While large language models (LLMs) hold promise for generating text, producing and evaluating controlled variation requires more careful attention. In this paper, we present an end-to-end pipeline for the generation and causal estimation of latent textual interventions. Our work first performs hypothesis generation and steering via sparse autoencoders (SAEs), followed by robust causal estimation. Our pipeline addresses both computational and statistical challenges in text-as-treatment experiments. We demonstrate that naive estimation of causal effects suffers from significant bias as text inherently conflates treatment and covariate information. We describe the estimation bias induced in this setting and propose a solution based on covariate residualization. Our empirical results show that our pipeline effectively induces variation in target features and mitigates estimation error, providing a robust foundation for causal effect estimation in text-as-treatment settings. 

---
# Rethinking Metrics for Lexical Semantic Change Detection 

**Authors**: Roksana Goworek, Haim Dubossarsky  

**Link**: [PDF](https://arxiv.org/pdf/2602.15716)  

**Abstract**: Lexical semantic change detection (LSCD) increasingly relies on contextualised language model embeddings, yet most approaches still quantify change using a small set of semantic change metrics, primarily Average Pairwise Distance (APD) and cosine distance over word prototypes (PRT). We introduce Average Minimum Distance (AMD) and Symmetric Average Minimum Distance (SAMD), new measures that quantify semantic change via local correspondence between word usages across time periods. Across multiple languages, encoder models, and representation spaces, we show that AMD often provides more robust performance, particularly under dimensionality reduction and with non-specialised encoders, while SAMD excels with specialised encoders. We suggest that LSCD may benefit from considering alternative semantic change metrics beyond APD and PRT, with AMD offering a robust option for contextualised embedding-based analysis. 

---
# A Content-Based Framework for Cybersecurity Refusal Decisions in Large Language Models 

**Authors**: Meirav Segal, Noa Linder, Omer Antverg, Gil Gekker, Tomer Fichman, Omri Bodenheimer, Edan Maor, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2602.15689)  

**Abstract**: Large language models and LLM-based agents are increasingly used for cybersecurity tasks that are inherently dual-use. Existing approaches to refusal, spanning academic policy frameworks and commercially deployed systems, often rely on broad topic-based bans or offensive-focused taxonomies. As a result, they can yield inconsistent decisions, over-restrict legitimate defenders, and behave brittlely under obfuscation or request segmentation. We argue that effective refusal requires explicitly modeling the trade-off between offensive risk and defensive benefit, rather than relying solely on intent or offensive classification. In this paper, we introduce a content-based framework for designing and auditing cyber refusal policies that makes offense-defense tradeoffs explicit. The framework characterizes requests along five dimensions: Offensive Action Contribution, Offensive Risk, Technical Complexity, Defensive Benefit, and Expected Frequency for Legitimate Users, grounded in the technical substance of the request rather than stated intent. We demonstrate that this content-grounded approach resolves inconsistencies in current frontier model behavior and allows organizations to construct tunable, risk-aware refusal policies. 

---
# Revisiting Northrop Frye's Four Myths Theory with Large Language Models 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2602.15678)  

**Abstract**: Northrop Frye's theory of four fundamental narrative genres (comedy, romance, tragedy, satire) has profoundly influenced literary criticism, yet computational approaches to his framework have focused primarily on narrative patterns rather than character functions. In this paper, we present a new character function framework that complements pattern-based analysis by examining how archetypal roles manifest differently across Frye's genres. Drawing on Jungian archetype theory, we derive four universal character functions (protagonist, mentor, antagonist, companion) by mapping them to Jung's psychic structure components. These functions are then specialized into sixteen genre-specific roles based on prototypical works. To validate this framework, we conducted a multi-model study using six state-of-the-art Large Language Models (LLMs) to evaluate character-role correspondences across 40 narrative works. The validation employed both positive samples (160 valid correspondences) and negative samples (30 invalid correspondences) to evaluate whether models both recognize valid correspondences and reject invalid ones. LLMs achieved substantial performance (mean balanced accuracy of 82.5%) with strong inter-model agreement (Fleiss' $\kappa$ = 0.600), demonstrating that the proposed correspondences capture systematic structural patterns. Performance varied by genre (ranging from 72.7% to 89.9%) and role (52.5% to 99.2%), with qualitative analysis revealing that variations reflect genuine narrative properties, including functional distribution in romance and deliberate archetypal subversion in satire. This character-based approach demonstrates the potential of LLM-supported methods for computational narratology and provides a foundation for future development of narrative generation methods and interactive storytelling applications. 

---
# LLM-to-Speech: A Synthetic Data Pipeline for Training Dialectal Text-to-Speech Models 

**Authors**: Ahmed Khaled Khamis, Hesham Ali  

**Link**: [PDF](https://arxiv.org/pdf/2602.15675)  

**Abstract**: Despite the advances in neural text to speech (TTS), many Arabic dialectal varieties remain marginally addressed, with most resources concentrated on Modern Spoken Arabic (MSA) and Gulf dialects, leaving Egyptian Arabic -- the most widely understood Arabic dialect -- severely under-resourced. We address this gap by introducing NileTTS: 38 hours of transcribed speech from two speakers across diverse domains including medical, sales, and general conversations. We construct this dataset using a novel synthetic pipeline: large language models (LLM) generate Egyptian Arabic content, which is then converted to natural speech using audio synthesis tools, followed by automatic transcription and speaker diarization with manual quality verification. We fine-tune XTTS v2, a state-of-the-art multilingual TTS model, on our dataset and evaluate against the baseline model trained on other Arabic dialects. Our contributions include: (1) the first publicly available Egyptian Arabic TTS dataset, (2) a reproducible synthetic data generation pipeline for dialectal TTS, and (3) an open-source fine-tuned model. All resources are released to advance Egyptian Arabic speech synthesis research. 

---
# STAPO: Stabilizing Reinforcement Learning for LLMs by Silencing Rare Spurious Tokens 

**Authors**: Shiqi Liu, Zeyu He, Guojian Zhan, Letian Tao, Zhilong Zheng, Jiang Wu, Yinuo Wang, Yang Guan, Kehua Sheng, Bo Zhang, Keqiang Li, Jingliang Duan, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.15620)  

**Abstract**: Reinforcement Learning (RL) has significantly improved large language model reasoning, but existing RL fine-tuning methods rely heavily on heuristic techniques such as entropy regularization and reweighting to maintain stability. In practice, they often experience late-stage performance collapse, leading to degraded reasoning quality and unstable training. We derive that the magnitude of token-wise policy gradients in RL is negatively correlated with token probability and local policy entropy. Building on this result, we prove that training instability is driven by a tiny fraction of tokens, approximately 0.01\%, which we term \emph{spurious tokens}. When such tokens appear in correct responses, they contribute little to the reasoning outcome but inherit the full sequence-level reward, leading to abnormally amplified gradient updates. Motivated by this observation, we propose Spurious-Token-Aware Policy Optimization (STAPO) for large-scale model refining, which selectively masks such updates and renormalizes the loss over valid tokens. Across six mathematical reasoning benchmarks using Qwen 1.7B, 8B, and 14B base models, STAPO consistently demonstrates superior entropy stability and achieves an average performance improvement of 7.13\% over GRPO, 20-Entropy and JustRL. 

---
# Clinically Inspired Symptom-Guided Depression Detection from Emotion-Aware Speech Representations 

**Authors**: Chaithra Nerella, Chiranjeevi Yarra  

**Link**: [PDF](https://arxiv.org/pdf/2602.15578)  

**Abstract**: Depression manifests through a diverse set of symptoms such as sleep disturbance, loss of interest, and concentration difficulties. However, most existing works treat depression prediction either as a binary label or an overall severity score without explicitly modeling symptom-specific information. This limits their ability to provide symptom-level analysis relevant to clinical screening. To address this, we propose a symptom-specific and clinically inspired framework for depression severity estimation from speech. Our approach uses a symptom-guided cross-attention mechanism that aligns PHQ-8 questionnaire items with emotion-aware speech representations to identify which segments of a participant's speech are more important to each symptom. To account for differences in how symptoms are expressed over time, we introduce a learnable symptom-specific parameter that adaptively controls the sharpness of attention distributions. Our results on EDAIC, a standard clinical-style dataset, demonstrate improved performance outperforming prior works. Further, analyzing the attention distributions showed that higher attention is assigned to utterances containing cues related to multiple depressive symptoms, highlighting the interpretability of our approach. These findings outline the importance of symptom-guided and emotion-aware modeling for speech-based depression screening. 

---
# Beyond Static Pipelines: Learning Dynamic Workflows for Text-to-SQL 

**Authors**: Yihan Wang, Peiyu Liu, Runyu Chen, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2602.15564)  

**Abstract**: Text-to-SQL has recently achieved impressive progress, yet remains difficult to apply effectively in real-world scenarios. This gap stems from the reliance on single static workflows, fundamentally limiting scalability to out-of-distribution and long-tail scenarios. Instead of requiring users to select suitable methods through extensive experimentation, we attempt to enable systems to adaptively construct workflows at inference time. Through theoretical and empirical analysis, we demonstrate that optimal dynamic policies consistently outperform the best static workflow, with performance gains fundamentally driven by heterogeneity across candidate workflows. Motivated by this, we propose SquRL, a reinforcement learning framework that enhances LLMs' reasoning capability in adaptive workflow construction. We design a rule-based reward function and introduce two effective training mechanisms: dynamic actor masking to encourage broader exploration, and pseudo rewards to improve training efficiency. Experiments on widely-used Text-to-SQL benchmarks demonstrate that dynamic workflow construction consistently outperforms the best static workflow methods, with especially pronounced gains on complex and out-of-distribution queries. The codes are available at this https URL 

---
# jina-embeddings-v5-text: Task-Targeted Embedding Distillation 

**Authors**: Mohammad Kalim Akram, Saba Sturua, Nastia Havriushenko, Quentin Herreros, Michael Günther, Maximilian Werk, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2602.15547)  

**Abstract**: Text embedding models are widely used for semantic similarity tasks, including information retrieval, clustering, and classification. General-purpose models are typically trained with single- or multi-stage processes using contrastive loss functions. We introduce a novel training regimen that combines model distillation techniques with task-specific contrastive loss to produce compact, high-performance embedding models. Our findings suggest that this approach is more effective for training small models than purely contrastive or distillation-based training paradigms alone. Benchmark scores for the resulting models, jina-embeddings-v5-text-small and jina-embeddings-v5-text-nano, exceed or match the state-of-the-art for models of similar size. jina-embeddings-v5-text models additionally support long texts (up to 32k tokens) in many languages, and generate embeddings that remain robust under truncation and binary quantization. Model weights are publicly available, hopefully inspiring further advances in embedding model development. 

---
# Perspectives - Interactive Document Clustering in the Discourse Analysis Tool Suite 

**Authors**: Tim Fischer, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2602.15540)  

**Abstract**: This paper introduces Perspectives, an interactive extension of the Discourse Analysis Tool Suite designed to empower Digital Humanities (DH) scholars to explore and organize large, unstructured document collections. Perspectives implements a flexible, aspect-focused document clustering pipeline with human-in-the-loop refinement capabilities. We showcase how this process can be initially steered by defining analytical lenses through document rewriting prompts and instruction-based embeddings, and further aligned with user intent through tools for refining clusters and mechanisms for fine-tuning the embedding model. The demonstration highlights a typical workflow, illustrating how DH researchers can leverage Perspectives's interactive document map to uncover topics, sentiments, or other relevant categories, thereby gaining insights and preparing their data for subsequent in-depth analysis. 

---
# ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling 

**Authors**: Nicol Visser, Simon Malan, Danel Slabbert, Herman Kamper  

**Link**: [PDF](https://arxiv.org/pdf/2602.15537)  

**Abstract**: Pure speech language models aim to learn language directly from raw audio without textual resources. A key challenge is that discrete tokens from self-supervised speech encoders result in excessively long sequences, motivating recent work on syllable-like units. However, methods like Sylber and SyllableLM rely on intricate multi-stage training pipelines. We propose ZeroSyl, a simple training-free method to extract syllable boundaries and embeddings directly from a frozen WavLM model. Using L2 norms of features in WavLM's intermediate layers, ZeroSyl achieves competitive syllable segmentation performance. The resulting segments are mean-pooled, discretized using K-means, and used to train a language model. ZeroSyl outperforms prior syllabic tokenizers across lexical, syntactic, and narrative benchmarks. Scaling experiments show that while finer-grained units are beneficial for lexical tasks, our discovered syllabic units exhibit better scaling behavior for syntactic modeling. 

---
# ExpertWeaver: Unlocking the Inherent MoE in Dense LLMs with GLU Activation Patterns 

**Authors**: Ziyu Zhao, Tong Zhu, Zhi Zhang, Tiantian Fan, Jinluan Yang, Kun Kuang, Zhongyu Wei, Fei Wu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2602.15521)  

**Abstract**: Mixture-of-Experts (MoE) effectively scales model capacity while preserving computational efficiency through sparse expert activation. However, training high-quality MoEs from scratch is prohibitively expensive. A promising alternative is to convert pretrained dense models into sparse MoEs. Existing dense-to-MoE methods fall into two categories: \textbf{dynamic structural pruning} that converts dense models into MoE architectures with moderate sparsity to balance performance and inference efficiency, and \textbf{downcycling} approaches that use pretrained dense models to initialize highly sparse MoE architectures. However, existing methods break the intrinsic activation patterns within dense models, leading to suboptimal expert construction. In this work, we argue that the Gated Linear Unit (GLU) mechanism provides a natural blueprint for dense-to-MoE conversion. We show that the fine-grained neural-wise activation patterns of GLU reveal a coarse-grained structure, uncovering an inherent MoE architecture composed of consistently activated universal neurons and dynamically activated specialized neurons. Leveraging this discovery, we introduce ExpertWeaver, a training-free framework that partitions neurons according to their activation patterns and constructs shared experts and specialized routed experts with layer-adaptive configurations. Our experiments demonstrate that ExpertWeaver significantly outperforms existing methods, both as a training-free dynamic structural pruning technique and as a downcycling strategy for superior MoE initialization. 

---
# DependencyAI: Detecting AI Generated Text through Dependency Parsing 

**Authors**: Sara Ahmed, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2602.15514)  

**Abstract**: As large language models (LLMs) become increasingly prevalent, reliable methods for detecting AI-generated text are critical for mitigating potential risks. We introduce DependencyAI, a simple and interpretable approach for detecting AI-generated text using only the labels of linguistic dependency relations. Our method achieves competitive performance across monolingual, multi-generator, and multilingual settings. To increase interpretability, we analyze feature importance to reveal syntactic structures that distinguish AI-generated from human-written text. We also observe a systematic overprediction of certain models on unseen domains, suggesting that generator-specific writing styles may affect cross-domain generalization. Overall, our results demonstrate that dependency relations alone provide a robust signal for AI-generated text detection, establishing DependencyAI as a strong linguistically grounded, interpretable, and non-neural network baseline. 

---
# Fine-Refine: Iterative Fine-grained Refinement for Mitigating Dialogue Hallucination 

**Authors**: Xiangyan Chen, Yujian Gan, Matthew Purver  

**Link**: [PDF](https://arxiv.org/pdf/2602.15509)  

**Abstract**: The tendency for hallucination in current large language models (LLMs) negatively impacts dialogue systems. Such hallucinations produce factually incorrect responses that may mislead users and undermine system trust. Existing refinement methods for dialogue systems typically operate at the response level, overlooking the fact that a single response may contain multiple verifiable or unverifiable facts. To address this gap, we propose Fine-Refine, a fine-grained refinement framework that decomposes responses into atomic units, verifies each unit using external knowledge, assesses fluency via perplexity, and iteratively corrects granular errors. We evaluate factuality across the HybriDialogue and OpendialKG datasets in terms of factual accuracy (fact score) and coverage (Not Enough Information Proportion), and experiments show that Fine-Refine substantially improves factuality, achieving up to a 7.63-point gain in dialogue fact score, with a small trade-off in dialogue quality. 

---
# LuxMT Technical Report 

**Authors**: Nils Rehlinger  

**Link**: [PDF](https://arxiv.org/pdf/2602.15506)  

**Abstract**: We introduce LuxMT, a machine translation system based on Gemma 3 27B and fine-tuned for translation from Luxembourgish (LB) into French (FR) and English (EN). To assess translation performance, we construct a novel benchmark covering LB-FR, LB-EN, and LB-FR using human-translated data from Luci, a tourist magazine about Luxembourg. Training data stems from LuxAlign, a parallel corpus of multilingual Luxembourgish news articles, and LB parliamentary transcripts augmented with Google Translate. We filter the data using LuxEmbedder, LB sentence embeddings, to remove low-equivalence segment-pairs. Overall, LuxMT's results suggest strong improvements over the Gemma 3 baseline, even for translating LB to German (DE), despite the training data not containing any DE. We also explore LuxEmbedder's potential to be used as a quality estimation metric and find strong correlations with other reference-based metrics. However, we call for further research to fully assess the metric's utility and advise using it with caution. 

---
# Towards Expectation Detection in Language: A Case Study on Treatment Expectations in Reddit 

**Authors**: Aswathy Velutharambath, Amelie Wührl  

**Link**: [PDF](https://arxiv.org/pdf/2602.15504)  

**Abstract**: Patients' expectations towards their treatment have a substantial effect on the treatments' success. While primarily studied in clinical settings, online patient platforms like medical subreddits may hold complementary insights: treatment expectations that patients feel unnecessary or uncomfortable to share elsewhere. Despite this, no studies examine what type of expectations users discuss online and how they express them. Presumably this is because expectations have not been studied in natural language processing (NLP) before. Therefore, we introduce the task of Expectation Detection, arguing that expectations are relevant for many applications, including opinion mining and product design. Subsequently, we present a case study for the medical domain, where expectations are particularly crucial to extract. We contribute RedHOTExpect, a corpus of Reddit posts (4.5K posts) to study expectations in this context. We use a large language model (LLM) to silver-label the data and validate its quality manually (label accuracy ~78%). Based on this, we analyze which linguistic patterns characterize expectations and explore what patients expect and why. We find that optimism and proactive framing are more pronounced in posts about physical or treatment-related illnesses compared to mental-health contexts, and that in our dataset, patients mostly discuss benefits rather than negative outcomes. The RedHOTExpect corpus can be obtained from this https URL 

---
# In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations 

**Authors**: Mohammad Aflah Khan, Mahsa Amani, Soumi Das, Bishwamittra Ghosh, Qinyuan Wu, Krishna P. Gummadi, Manish Gupta, Abhilasha Ravichander  

**Link**: [PDF](https://arxiv.org/pdf/2602.15456)  

**Abstract**: Agents based on Large Language Models (LLMs) are increasingly being deployed as interfaces to information on online platforms. These agents filter, prioritize, and synthesize information retrieved from the platforms' back-end databases or via web search. In these scenarios, LLM agents govern the information users receive, by drawing users' attention to particular instances of retrieved information at the expense of others. While much prior work has focused on biases in the information LLMs themselves generate, less attention has been paid to the factors that influence what information LLMs select and present to users. We hypothesize that when information is attributed to specific sources (e.g., particular publishers, journals, or platforms), current LLMs exhibit systematic latent source preferences- that is, they prioritize information from some sources over others. Through controlled experiments on twelve LLMs from six model providers, spanning both synthetic and real-world tasks, we find that several models consistently exhibit strong and predictable source preferences. These preferences are sensitive to contextual framing, can outweigh the influence of content itself, and persist despite explicit prompting to avoid them. They also help explain phenomena such as the observed left-leaning skew in news recommendations in prior work. Our findings advocate for deeper investigation into the origins of these preferences, as well as for mechanisms that provide users with transparency and control over the biases guiding LLM-powered agents. 

---
# TAROT: Test-driven and Capability-adaptive Curriculum Reinforcement Fine-tuning for Code Generation with Large Language Models 

**Authors**: Chansung Park, Juyong Jiang, Fan Wang, Sayak Paul, Jiasi Shen, Jing Tang, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.15449)  

**Abstract**: Large Language Models (LLMs) are changing the coding paradigm, known as vibe coding, yet synthesizing algorithmically sophisticated and robust code still remains a critical challenge. Incentivizing the deep reasoning capabilities of LLMs is essential to overcoming this hurdle. Reinforcement Fine-Tuning (RFT) has emerged as a promising strategy to address this need. However, most existing approaches overlook the heterogeneous difficulty and granularity inherent in test cases, leading to an imbalanced distribution of reward signals and consequently biased gradient updates during training. To address this, we propose Test-driven and cApability-adaptive cuRriculum reinfOrcement fine-Tuning (TAROT). TAROT systematically constructs, for each problem, a four-tier test suite (basic, intermediate, complex, edge), providing a controlled difficulty landscape for curriculum design and evaluation. Crucially, TAROT decouples curriculum progression from raw reward scores, enabling capability-conditioned evaluation and principled selection from a portfolio of curriculum policies rather than incidental test-case difficulty composition. This design fosters stable optimization and more efficient competency acquisition. Extensive experimental results reveal that the optimal curriculum for RFT in code generation is closely tied to a model's inherent capability, with less capable models achieving greater gains with an easy-to-hard progression, whereas more competent models excel under a hard-first curriculum. TAROT provides a reproducible method that adaptively tailors curriculum design to a model's capability, thereby consistently improving the functional correctness and robustness of the generated code. All code and data are released to foster reproducibility and advance community research at this https URL. 

---
# Measuring Social Integration Through Participation: Categorizing Organizations and Leisure Activities in the Displaced Karelians Interview Archive using LLMs 

**Authors**: Joonatan Laato, Veera Schroderus, Jenna Kanerva, Jenni Kauppi, Virpi Lummaa, Filip Ginter  

**Link**: [PDF](https://arxiv.org/pdf/2602.15436)  

**Abstract**: Digitized historical archives make it possible to study everyday social life on a large scale, but the information extracted directly from text often does not directly allow one to answer the research questions posed by historians or sociologists in a quantitative manner. We address this problem in a large collection of Finnish World War II Karelian evacuee family interviews. Prior work extracted more than 350K mentions of leisure time activities and organizational memberships from these interviews, yielding 71K unique activity and organization names -- far too many to analyze directly.
We develop a categorization framework that captures key aspects of participation (the kind of activity/organization, how social it typically is, how regularly it happens, and how physically demanding it is). We annotate a gold-standard set to allow for a reliable evaluation, and then test whether large language models can apply the same schema at scale. Using a simple voting approach across multiple model runs, we find that an open-weight LLM can closely match expert judgments. Finally, we apply the method to label the 350K entities, producing a structured resource for downstream studies of social integration and related outcomes. 

---
# The Vision Wormhole: Latent-Space Communication in Heterogeneous Multi-Agent Systems 

**Authors**: Xiaoze Liu, Ruowang Zhang, Weichen Yu, Siheng Xiong, Liu He, Feijie Wu, Hoin Jung, Matt Fredrikson, Xiaoqian Wang, Jing Gao  

**Link**: [PDF](https://arxiv.org/pdf/2602.15382)  

**Abstract**: Multi-Agent Systems (MAS) powered by Large Language Models have unlocked advanced collaborative reasoning, yet they remain shackled by the inefficiency of discrete text communication, which imposes significant runtime overhead and information quantization loss. While latent state transfer offers a high-bandwidth alternative, existing approaches either assume homogeneous sender-receiver architectures or rely on pair-specific learned translators, limiting scalability and modularity across diverse model families with disjoint manifolds. In this work, we propose the Vision Wormhole, a novel framework that repurposes the visual interface of Vision-Language Models (VLMs) to enable model-agnostic, text-free communication. By introducing a Universal Visual Codec, we map heterogeneous reasoning traces into a shared continuous latent space and inject them directly into the receiver's visual pathway, effectively treating the vision encoder as a universal port for inter-agent telepathy. Our framework adopts a hub-and-spoke topology to reduce pairwise alignment complexity from O(N^2) to O(N) and leverages a label-free, teacher-student distillation objective to align the high-speed visual channel with the robust reasoning patterns of the text pathway. Extensive experiments across heterogeneous model families (e.g., Qwen-VL, Gemma) demonstrate that the Vision Wormhole reduces end-to-end wall-clock time in controlled comparisons while maintaining reasoning fidelity comparable to standard text-based MAS. Code is available at this https URL 

---
# Making Large Language Models Speak Tulu: Structured Prompting for an Extremely Low-Resource Language 

**Authors**: Prathamesh Devadiga, Paras Chopra  

**Link**: [PDF](https://arxiv.org/pdf/2602.15378)  

**Abstract**: Can large language models converse in languages virtually absent from their training data? We investigate this question through a case study on Tulu, a Dravidian language with over 2 million speakers but minimal digital presence. Rather than fine-tuning an LLM, we examine whether structured prompts alone can elicit basic conversational ability under controlled prompting. We systematically tackle various challenges posed by absence of training data for Tulu by combining explicit grammar documentation, negative constraints to suppress high-probability tokens from related languages, romanization standardization, and quality-controlled synthetic data generation via self-play. Evaluated on a manually curated held-out set across three LLMs (Gemini 2.0 Flash, GPT-4o, Llama 3.1 70B) and validated by native speakers, our approach reduces vocabulary contamination from 80% to 5% while achieving 85% grammatical accuracy. Cross-model analysis reveals that negative constraints provide consistent improvements (12--18 percentage points), while grammar documentation effects vary by model architecture (8--22 points). 

---
# Orchestration-Free Customer Service Automation: A Privacy-Preserving and Flowchart-Guided Framework 

**Authors**: Mengze Hong, Chen Jason Zhang, Zichang Guo, Hanlin Gu, Di Jiang, Li Qing  

**Link**: [PDF](https://arxiv.org/pdf/2602.15377)  

**Abstract**: Customer service automation has seen growing demand within digital transformation. Existing approaches either rely on modular system designs with extensive agent orchestration or employ over-simplified instruction schemas, providing limited guidance and poor generalizability. This paper introduces an orchestration-free framework using Task-Oriented Flowcharts (TOFs) to enable end-to-end automation without manual intervention. We first define the components and evaluation metrics for TOFs, then formalize a cost-efficient flowchart construction algorithm to abstract procedural knowledge from service dialogues. We emphasize local deployment of small language models and propose decentralized distillation with flowcharts to mitigate data scarcity and privacy issues in model training. Extensive experiments validate the effectiveness in various service tasks, with superior quantitative and application performance compared to strong baselines and market products. By releasing a web-based system demonstration with case studies, we aim to promote streamlined creation of future service automation. 

---
# Far Out: Evaluating Language Models on Slang in Australian and Indian English 

**Authors**: Deniz Kaya Dilsiz, Dipankar Srirag, Aditya Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2602.15373)  

**Abstract**: Language models exhibit systematic performance gaps when processing text in non-standard language varieties, yet their ability to comprehend variety-specific slang remains underexplored for several languages. We present a comprehensive evaluation of slang awareness in Indian English (en-IN) and Australian English (en-AU) across seven state-of-the-art language models. We construct two complementary datasets: \textsc{web}, containing 377 web-sourced usage examples from Urban Dictionary, and \textsc{gen}, featuring 1,492 synthetically generated usages of these slang terms, across diverse scenarios. We assess language models on three tasks: target word prediction (TWP), guided target word prediction (TWP$^*$) and target word selection (TWS). Our results reveal four key findings: (1) Higher average model performance TWS versus TWP and TWP$^*$, with average accuracy score increasing from 0.03 to 0.49 respectively (2) Stronger average model performance on \textsc{web} versus \textsc{gen} datasets, with average similarity score increasing by 0.03 and 0.05 across TWP and TWP$^*$ tasks respectively (3) en-IN tasks outperform en-AU when averaged across all models and datasets, with TWS demonstrating the largest disparity, increasing average accuracy from 0.44 to 0.54. These findings underscore fundamental asymmetries between generative and discriminative competencies for variety-specific language, particularly in the context of slang expressions despite being in a technologically rich language such as English. 

---
# NeuroSymActive: Differentiable Neural-Symbolic Reasoning with Active Exploration for Knowledge Graph Question Answering 

**Authors**: Rong Fu, Yang Li, Zeyu Zhang, Jiekai Wu, Yaohua Liu, Shuaishuai Cao, Yangchen Zeng, Yuhang Zhang, Xiaojing Du, Chuang Zhao, Kangning Cui, Simon Fong  

**Link**: [PDF](https://arxiv.org/pdf/2602.15353)  

**Abstract**: Large pretrained language models and neural reasoning systems have advanced many natural language tasks, yet they remain challenged by knowledge-intensive queries that require precise, structured multi-hop inference. Knowledge graphs provide a compact symbolic substrate for factual grounding, but integrating graph structure with neural models is nontrivial: naively embedding graph facts into prompts leads to inefficiency and fragility, while purely symbolic or search-heavy approaches can be costly in retrievals and lack gradient-based refinement. We introduce NeuroSymActive, a modular framework that combines a differentiable neural-symbolic reasoning layer with an active, value-guided exploration controller for Knowledge Graph Question Answering. The method couples soft-unification style symbolic modules with a neural path evaluator and a Monte-Carlo style exploration policy that prioritizes high-value path expansions. Empirical results on standard KGQA benchmarks show that NeuroSymActive attains strong answer accuracy while reducing the number of expensive graph lookups and model calls compared to common retrieval-augmented baselines. 

---
# Mnemis: Dual-Route Retrieval on Hierarchical Graphs for Long-Term LLM Memory 

**Authors**: Zihao Tang, Xin Yu, Ziyu Xiao, Zengxuan Wen, Zelin Li, Jiaxi Zhou, Hualei Wang, Haohua Wang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15313)  

**Abstract**: AI Memory, specifically how models organizes and retrieves historical messages, becomes increasingly valuable to Large Language Models (LLMs), yet existing methods (RAG and Graph-RAG) primarily retrieve memory through similarity-based mechanisms. While efficient, such System-1-style retrieval struggles with scenarios that require global reasoning or comprehensive coverage of all relevant information. In this work, We propose Mnemis, a novel memory framework that integrates System-1 similarity search with a complementary System-2 mechanism, termed Global Selection. Mnemis organizes memory into a base graph for similarity retrieval and a hierarchical graph that enables top-down, deliberate traversal over semantic hierarchies. By combining the complementary strength from both retrieval routes, Mnemis retrieves memory items that are both semantically and structurally relevant. Mnemis achieves state-of-the-art performance across all compared methods on long-term memory benchmarks, scoring 93.9 on LoCoMo and 91.6 on LongMemEval-S using GPT-4.1-mini. 

---
# Extracting Consumer Insight from Text: A Large Language Model Approach to Emotion and Evaluation Measurement 

**Authors**: Stephan Ludwig, Peter J. Danaher, Xiaohao Yang, Yu-Ting Lin, Ehsan Abedin, Dhruv Grewal, Lan Du  

**Link**: [PDF](https://arxiv.org/pdf/2602.15312)  

**Abstract**: Accurately measuring consumer emotions and evaluations from unstructured text remains a core challenge for marketing research and practice. This study introduces the Linguistic eXtractor (LX), a fine-tuned, large language model trained on consumer-authored text that also has been labeled with consumers' self-reported ratings of 16 consumption-related emotions and four evaluation constructs: trust, commitment, recommendation, and sentiment. LX consistently outperforms leading models, including GPT-4 Turbo, RoBERTa, and DeepSeek, achieving 81% macro-F1 accuracy on open-ended survey responses and greater than 95% accuracy on third-party-annotated Amazon and Yelp reviews. An application of LX to online retail data, using seemingly unrelated regression, affirms that review-expressed emotions predict product ratings, which in turn predict purchase behavior. Most emotional effects are mediated by product ratings, though some emotions, such as discontent and peacefulness, influence purchase directly, indicating that emotional tone provides meaningful signals beyond star ratings. To support its use, a no-code, cost-free, LX web application is available, enabling scalable analyses of consumer-authored text. In establishing a new methodological foundation for consumer perception measurement, this research demonstrates new methods for leveraging large language models to advance marketing research and practice, thereby achieving validated detection of marketing constructs from consumer data. 

---
# OpaqueToolsBench: Learning Nuances of Tool Behavior Through Interaction 

**Authors**: Skyler Hallinan, Thejas Venkatesh, Xiang Ren, Sai Praneeth Karimireddy, Ashwin Paranjape, Yuhao Zhang, Jack Hessel  

**Link**: [PDF](https://arxiv.org/pdf/2602.15197)  

**Abstract**: Tool-calling is essential for Large Language Model (LLM) agents to complete real-world tasks. While most existing benchmarks assume simple, perfectly documented tools, real-world tools (e.g., general "search" APIs) are often opaque, lacking clear best practices or failure modes. Can LLM agents improve their performance in environments with opaque tools by interacting and subsequently improving documentation? To study this, we create OpaqueToolsBench, a benchmark consisting of three distinct task-oriented environments: general function calling, interactive chess playing, and long-trajectory agentic search. Each environment provides underspecified tools that models must learn to use effectively to complete the task. Results on OpaqueToolsBench suggest existing methods for automatically documenting tools are expensive and unreliable when tools are opaque. To address this, we propose a simple framework, ToolObserver, that iteratively refines tool documentation by observing execution feedback from tool-calling trajectories. Our approach outperforms existing methods on OpaqueToolsBench across datasets, even in relatively hard settings. Furthermore, for test-time tool exploration settings, our method is also efficient, consuming 3.5-7.5x fewer total tokens than the best baseline. 

---
# AIC CTU@AVerImaTeC: dual-retriever RAG for image-text fact checking 

**Authors**: Herbert Ullrich, Jan Drchal  

**Link**: [PDF](https://arxiv.org/pdf/2602.15190)  

**Abstract**: In this paper, we present our 3rd place system in the AVerImaTeC shared task, which combines our last year's retrieval-augmented generation (RAG) pipeline with a reverse image search (RIS) module. Despite its simplicity, our system delivers competitive performance with a single multimodal LLM call per fact-check at just $0.013 on average using GPT5.1 via OpenAI Batch API. Our system is also easy to reproduce and tweak, consisting of only three decoupled modules - a textual retrieval module based on similarity search, an image retrieval module based on API-accessed RIS, and a generation module using GPT5.1 - which is why we suggest it as an accesible starting point for further experimentation. We publish its code and prompts, as well as our vector stores and insights into the scheme's running costs and directions for further improvement. 

---
# CGRA-DeBERTa Concept Guided Residual Augmentation Transformer for Theologically Islamic Understanding 

**Authors**: Tahir Hussain, Saddam Hussain Khan  

**Link**: [PDF](https://arxiv.org/pdf/2602.15139)  

**Abstract**: Accurate QA over classical Islamic texts remains challenging due to domain specific semantics, long context dependencies, and concept sensitive reasoning. Therefore, a new CGRA DeBERTa, a concept guided residual domain augmentation transformer framework, is proposed that enhances theological QA over Hadith corpora. The CGRA DeBERTa builds on a customized DeBERTa transformer backbone with lightweight LoRA based adaptations and a residual concept aware gating mechanism. The customized DeBERTa embedding block learns global and positional context, while Concept Guided Residual Blocks incorporate theological priors from a curated Islamic Concept Dictionary of 12 core terms. Moreover, the Concept Gating Mechanism selectively amplifies semantically critical tokens via importance weighted attention, applying differential scaling from 1.04 to 3.00. This design preserves contextual integrity, strengthens domain-specific semantic representations, and enables accurate, efficient span extraction while maintaining computational efficiency. This paper reports the results of training CGRA using a specially constructed dataset of 42591 QA pairs from the text of Sahih alBukhari and Sahih Muslim. While BERT achieved an EM score of 75.87 and DeBERTa one of 89.77, our model scored 97.85 and thus surpassed them by 8.08 on an absolute scale, all while adding approximately 8 inference overhead due to parameter efficient gating. The qualitative evaluation noted better extraction and discrimination and theological precision. This study presents Hadith QA systems that are efficient, interpretable, and accurate and that scale provide educational materials with necessary theological nuance. 

---
# Indic-TunedLens: Interpreting Multilingual Models in Indian Languages 

**Authors**: Mihir Panchal, Deeksha Varshney, Mamta, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2602.15038)  

**Abstract**: Multilingual large language models (LLMs) are increasingly deployed in linguistically diverse regions like India, yet most interpretability tools remain tailored to English. Prior work reveals that LLMs often operate in English centric representation spaces, making cross lingual interpretability a pressing concern. We introduce Indic-TunedLens, a novel interpretability framework specifically for Indian languages that learns shared affine transformations. Unlike the standard Logit Lens, which directly decodes intermediate activations, Indic-TunedLens adjusts hidden states for each target language, aligning them with the target output distributions to enable more faithful decoding of model representations. We evaluate our framework on 10 Indian languages using the MMLU benchmark and find that it significantly improves over SOTA interpretability methods, especially for morphologically rich, low resource languages. Our results provide crucial insights into the layer-wise semantic encoding of multilingual transformers. Our model is available at this https URL. Our code is available at this https URL. 

---
# EduResearchBench: A Hierarchical Atomic Task Decomposition Benchmark for Full-Lifecycle Educational Research 

**Authors**: Houping Yue, Zixiang Di, Mei Jiang, Bingdong Li, Hao Hao, Yu Song, Bo Jiang, Aimin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2602.15034)  

**Abstract**: While Large Language Models (LLMs) are reshaping the paradigm of AI for Social Science (AI4SS), rigorously evaluating their capabilities in scholarly writing remains a major challenge. Existing benchmarks largely emphasize single-shot, monolithic generation and thus lack the fine-grained assessments required to reflect complex academic research workflows. To fill this gap, we introduce EduResearchBench, the first comprehensive evaluation platform dedicated to educational academic writing. EduResearchBench is built upon our Hierarchical Atomic Task Decomposition (HATD) framework, which decomposes an end-to-end research workflow into six specialized research modules (e.g., Quantitative Analysis, Qualitative Research, and Policy Research) spanning 24 fine-grained atomic tasks. This taxonomy enables an automated evaluation pipeline that mitigates a key limitation of holistic scoring, where aggregate scores often obscure specific capability bottlenecks, and instead provides fine-grained, diagnostic feedback on concrete deficiencies. Moreover, recognizing the high cognitive load inherent in scholarly writing, we propose a curriculum learning strategy that progressively builds competence from foundational skills to complex methodological reasoning and argumentation. Leveraging 55K raw academic samples, we curate 11K high-quality instruction pairs to train EduWrite, a specialized educational scholarly writing model. Experiments show that EduWrite (30B) substantially outperforms larger general-purpose models (72B) on multiple core metrics, demonstrating that in vertical domains, data quality density and hierarchically staged training curricula are more decisive than parameter scale. 

---
# Enhancing Building Semantics Preservation in AI Model Training with Large Language Model Encodings 

**Authors**: Suhyung Jang, Ghang Lee, Jaekun Lee, Hyunjun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2602.15791)  

**Abstract**: Accurate representation of building semantics, encompassing both generic object types and specific subtypes, is essential for effective AI model training in the architecture, engineering, construction, and operation (AECO) industry. Conventional encoding methods (e.g., one-hot) often fail to convey the nuanced relationships among closely related subtypes, limiting AI's semantic comprehension. To address this limitation, this study proposes a novel training approach that employs large language model (LLM) embeddings (e.g., OpenAI GPT and Meta LLaMA) as encodings to preserve finer distinctions in building semantics. We evaluated the proposed method by training GraphSAGE models to classify 42 building object subtypes across five high-rise residential building information models (BIMs). Various embedding dimensions were tested, including original high-dimensional LLM embeddings (1,536, 3,072, or 4,096) and 1,024-dimensional compacted embeddings generated via the Matryoshka representation model. Experimental results demonstrated that LLM encodings outperformed the conventional one-hot baseline, with the llama-3 (compacted) embedding achieving a weighted average F1-score of 0.8766, compared to 0.8475 for one-hot encoding. The results underscore the promise of leveraging LLM-based encodings to enhance AI's ability to interpret complex, domain-specific building semantics. As the capabilities of LLMs and dimensionality reduction techniques continue to evolve, this approach holds considerable potential for broad application in semantic elaboration tasks throughout the AECO industry. 

---
# GLM-5: from Vibe Coding to Agentic Engineering 

**Authors**: GLM-5 Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chen Li, Chenghua Huang, Chengwei Hu, Chenhui Zhang, Chenzheng Zhu, Congfeng Yin, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huan Liu, Huanpeng Chu, Jia'ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.15763)  

**Abstract**: We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon the agentic, reasoning, and coding (ARC) capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity. To advance model alignment and autonomy, we implement a new asynchronous reinforcement learning infrastructure that drastically improves post-training efficiency by decoupling generation from training. Furthermore, we propose novel asynchronous agent RL algorithms that further improve RL quality, enabling the model to learn from complex, long-horizon interactions more effectively. Through these innovations, GLM-5 achieves state-of-the-art performance on major open benchmarks. Most critically, GLM-5 demonstrates unprecedented capability in real-world coding tasks, surpassing previous baselines in handling end-to-end software engineering challenges. Code, models, and more information are available at this https URL. 

---
# Recursive Concept Evolution for Compositional Reasoning in Large Language Models 

**Authors**: Sarim Chaudhry  

**Link**: [PDF](https://arxiv.org/pdf/2602.15725)  

**Abstract**: Large language models achieve strong performance on many complex reasoning tasks, yet their accuracy degrades sharply on benchmarks that require compositional reasoning, including ARC-AGI-2, GPQA, MATH, BBH, and HLE. Existing methods improve reasoning by expanding token-level search through chain-of-thought prompting, self-consistency, or reinforcement learning, but they leave the model's latent representation space fixed. When the required abstraction is not already encoded in this space, performance collapses. We propose Recursive Concept Evolution (RCE), a framework that enables pretrained language models to modify their internal representation geometry during inference. RCE introduces dynamically generated low-rank concept subspaces that are spawned when representational inadequacy is detected, selected through a minimum description length criterion, merged when synergistic, and consolidated via constrained optimization to preserve stability. This process allows the model to construct new abstractions rather than recombining existing ones. We integrate RCE with Mistral-7B and evaluate it across compositional reasoning benchmarks. RCE yields 12-18 point gains on ARC-AGI-2, 8-14 point improvements on GPQA and BBH, and consistent reductions in depth-induced error on MATH and HLE. 

---
# Proactive Conversational Assistant for a Procedural Manual Task based on Audio and IMU 

**Authors**: Rehana Mahfuz, Yinyi Guo, Erik Visser, Phanidhar Chinchili  

**Link**: [PDF](https://arxiv.org/pdf/2602.15707)  

**Abstract**: Real-time conversational assistants for procedural tasks often depend on video input, which can be computationally expensive and compromise user privacy. For the first time, we propose a real-time conversational assistant that provides comprehensive guidance for a procedural task using only lightweight privacy-preserving modalities such as audio and IMU inputs from a user's wearable device to understand the context. This assistant proactively communicates step-by-step instructions to a user performing a furniture assembly task, and answers user questions. We construct a dataset containing conversations where the assistant guides the user in performing the task. On observing that an off-the-shelf language model is a very talkative assistant, we design a novel User Whim Agnostic (UWA) LoRA finetuning method which improves the model's ability to suppress less informative dialogues, while maintaining its tendency to communicate important instructions. This leads to >30% improvement in the F-score. Finetuning the model also results in a 16x speedup by eliminating the need to provide in-context examples in the prompt. We further describe how such an assistant is implemented on edge devices with no dependence on the cloud. 

---
# RUVA: Personalized Transparent On-Device Graph Reasoning 

**Authors**: Gabriele Conte, Alessio Mattiace, Gianni Carmosino, Potito Aghilar, Giovanni Servedio, Francesco Musicco, Vito Walter Anelli, Tommaso Di Noia, Francesco Maria Donini  

**Link**: [PDF](https://arxiv.org/pdf/2602.15553)  

**Abstract**: The Personal AI landscape is currently dominated by "Black Box" Retrieval-Augmented Generation. While standard vector databases offer statistical matching, they suffer from a fundamental lack of accountability: when an AI hallucinates or retrieves sensitive data, the user cannot inspect the cause nor correct the error. Worse, "deleting" a concept from a vector space is mathematically imprecise, leaving behind probabilistic "ghosts" that violate true privacy. We propose Ruva, the first "Glass Box" architecture designed for Human-in-the-Loop Memory Curation. Ruva grounds Personal AI in a Personal Knowledge Graph, enabling users to inspect what the AI knows and to perform precise redaction of specific facts. By shifting the paradigm from Vector Matching to Graph Reasoning, Ruva ensures the "Right to be Forgotten." Users are the editors of their own lives; Ruva hands them the pen. The project and the demo video are available at this http URL. 

---
# World-Model-Augmented Web Agents with Action Correction 

**Authors**: Zhouzhou Shen, Xueyu Hu, Xiyun Li, Tianqing Fang, Juncheng Li, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15384)  

**Abstract**: Web agents based on large language models have demonstrated promising capability in automating web tasks. However, current web agents struggle to reason out sensible actions due to the limitations of predicting environment changes, and might not possess comprehensive awareness of execution risks, prematurely performing risky actions that cause losses and lead to task failure. To address these challenges, we propose WAC, a web agent that integrates model collaboration, consequence simulation, and feedback-driven action refinement. To overcome the cognitive isolation of individual models, we introduce a multi-agent collaboration process that enables an action model to consult a world model as a web-environment expert for strategic guidance; the action model then grounds these suggestions into executable actions, leveraging prior knowledge of environmental state transition dynamics to enhance candidate action proposal. To achieve risk-aware resilient task execution, we introduce a two-stage deduction chain. A world model, specialized in environmental state transitions, simulates action outcomes, which a judge model then scrutinizes to trigger action corrective feedback when necessary. Experiments show that WAC achieves absolute gains of 1.8% on VisualWebArena and 1.3% on Online-Mind2Web. 

---
# Discovering Implicit Large Language Model Alignment Objectives 

**Authors**: Edward Chen, Sanmi Koyejo, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2602.15338)  

**Abstract**: Large language model (LLM) alignment relies on complex reward signals that often obscure the specific behaviors being incentivized, creating critical risks of misalignment and reward hacking. Existing interpretation methods typically rely on pre-defined rubrics, risking the omission of "unknown unknowns", or fail to identify objectives that comprehensively cover and are causal to the model behavior. To address these limitations, we introduce Obj-Disco, a framework that automatically decomposes an alignment reward signal into a sparse, weighted combination of human-interpretable natural language objectives. Our approach utilizes an iterative greedy algorithm to analyze behavioral changes across training checkpoints, identifying and validating candidate objectives that best explain the residual reward signal. Extensive evaluations across diverse tasks, model sizes, and alignment algorithms demonstrate the framework's robustness. Experiments with popular open-source reward models show that the framework consistently captures > 90% of reward behavior, a finding further corroborated by human evaluation. Additionally, a case study on alignment with an open-source reward model reveals that Obj-Disco can successfully identify latent misaligned incentives that emerge alongside intended behaviors. Our work provides a crucial tool for uncovering the implicit objectives in LLM alignment, paving the way for more transparent and safer AI development. 

---
# Prescriptive Scaling Reveals the Evolution of Language Model Capabilities 

**Authors**: Hanlin Zhang, Jikai Jin, Vasilis Syrgkanis, Sham Kakade  

**Link**: [PDF](https://arxiv.org/pdf/2602.15327)  

**Abstract**: For deploying foundation models, practitioners increasingly need prescriptive scaling laws: given a pre training compute budget, what downstream accuracy is attainable with contemporary post training practice, and how stable is that mapping as the field evolves? Using large scale observational evaluations with 5k observational and 2k newly sampled data on model performance, we estimate capability boundaries, high conditional quantiles of benchmark scores as a function of log pre training FLOPs, via smoothed quantile regression with a monotone, saturating sigmoid parameterization. We validate the temporal reliability by fitting on earlier model generations and evaluating on later releases. Across various tasks, the estimated boundaries are mostly stable, with the exception of math reasoning that exhibits a consistently advancing boundary over time. We then extend our approach to analyze task dependent saturation and to probe contamination related shifts on math reasoning tasks. Finally, we introduce an efficient algorithm that recovers near full data frontiers using roughly 20% of evaluation budget. Together, our work releases the Proteus 2k, the latest model performance evaluation dataset, and introduces a practical methodology for translating compute budgets into reliable performance expectations and for monitoring when capability boundaries shift across time. 

---
# The Information Geometry of Softmax: Probing and Steering 

**Authors**: Kiho Park, Todd Nief, Yo Joong Choe, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2602.15293)  

**Abstract**: This paper concerns the question of how AI systems encode semantic structure into the geometric structure of their representation spaces. The motivating observation of this paper is that the natural geometry of these representation spaces should reflect the way models use representations to produce behavior. We focus on the important special case of representations that define softmax distributions. In this case, we argue that the natural geometry is information geometry. Our focus is on the role of information geometry on semantic encoding and the linear representation hypothesis. As an illustrative application, we develop "dual steering", a method for robustly steering representations to exhibit a particular concept using linear probes. We prove that dual steering optimally modifies the target concept while minimizing changes to off-target concepts. Empirically, we find that dual steering enhances the controllability and stability of concept manipulation. 

---
# FrameRef: A Framing Dataset and Simulation Testbed for Modeling Bounded Rational Information Health 

**Authors**: Victor De Lima, Jiqun Liu, Grace Hui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2602.15273)  

**Abstract**: Information ecosystems increasingly shape how people internalize exposure to adverse digital experiences, raising concerns about the long-term consequences for information health. In modern search and recommendation systems, ranking and personalization policies play a central role in shaping such exposure and its long-term effects on users. To study these effects in a controlled setting, we present FrameRef, a large-scale dataset of 1,073,740 systematically reframed claims across five framing dimensions: authoritative, consensus, emotional, prestige, and sensationalist, and propose a simulation-based framework for modeling sequential information exposure and reinforcement dynamics characteristic of ranking and recommendation systems. Within this framework, we construct framing-sensitive agent personas by fine-tuning language models with framing-conditioned loss attenuation, inducing targeted biases while preserving overall task competence. Using Monte Carlo trajectory sampling, we show that small, systematic shifts in acceptance and confidence can compound over time, producing substantial divergence in cumulative information health trajectories. Human evaluation further confirms that FrameRef's generated framings measurably affect human judgment. Together, our dataset and framework provide a foundation for systematic information health research through simulation, complementing and informing responsible human-centered research. We release FrameRef, code, documentation, human evaluation data, and persona adapter models at this https URL. 

---
# How to Train Your Long-Context Visual Document Model 

**Authors**: Austin Veselka  

**Link**: [PDF](https://arxiv.org/pdf/2602.15257)  

**Abstract**: We present the first comprehensive, large-scale study of training long-context vision language models up to 344K context, targeting long-document visual question answering with measured transfer to long-context text. While several such strong are open-weight, namely Qwen3 VL and GLM 4.5/6V, their training recipes and data pipelines are not reproducible. We systematically study continued pretraining, supervised finetuning, and preference optimization for 24B and 32B parameter models, backed by extensive LC evaluations and ablations to bridge this gap, and achieve state-of-the-art performance on MMLongBenchDoc for both parameter scales. In addition to this, our key findings include: (i) training on context lengths that match evaluation context lengths outperforms training on longer contexts, (ii) training and evaluating with page indices provides a simple, high-impact boost to long-document performance, (iii) our synthetic data pipelines enable self-improvement via continued pretraining and supervised finetuning, and (iv) we extend the known text-to-visual long context transfer to the reverse, showing that visual long context training transfers to long-context text performance. We also release MMLBD-C, a manually corrected version of MMLongBenchDoc to reduce erroneous and low quality examples in the benchmark. 

---
# Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems 

**Authors**: Mason Nakamura, Abhinav Kumar, Saswat Das, Sahar Abdelnabi, Saaduddin Mahmud, Ferdinando Fioretto, Shlomo Zilberstein, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2602.15198)  

**Abstract**: Multi-agent systems, where LLM agents communicate through free-form language, enable sophisticated coordination for solving complex cooperative tasks. This surfaces a unique safety problem when individual agents form a coalition and \emph{collude} to pursue secondary goals and degrade the joint objective. In this paper, we present Colosseum, a framework for auditing LLM agents' collusive behavior in multi-agent settings. We ground how agents cooperate through a Distributed Constraint Optimization Problem (DCOP) and measure collusion via regret relative to the cooperative optimum. Colosseum tests each LLM for collusion under different objectives, persuasion tactics, and network topologies. Through our audit, we show that most out-of-the-box models exhibited a propensity to collude when a secret communication channel was artificially formed. Furthermore, we discover ``collusion on paper'' when agents plan to collude in text but would often pick non-collusive actions, thus providing little effect on the joint task. Colosseum provides a new way to study collusion by measuring communications and actions in rich yet verifiable environments. 

---
# Weight space Detection of Backdoors in LoRA Adapters 

**Authors**: David Puertolas Merenciano, Ekaterina Vasyagina, Raghav Dixit, Kevin Zhu, Ruizhe Li, Javier Ferrando, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2602.15195)  

**Abstract**: LoRA adapters let users fine-tune large language models (LLMs) efficiently. However, LoRA adapters are shared through open repositories like Hugging Face Hub \citep{huggingface_hub_docs}, making them vulnerable to backdoor attacks. Current detection methods require running the model with test input data -- making them impractical for screening thousands of adapters where the trigger for backdoor behavior is unknown. We detect poisoned adapters by analyzing their weight matrices directly, without running the model -- making our method data-agnostic. Our method extracts simple statistics -- how concentrated the singular values are, their entropy, and the distribution shape -- and flags adapters that deviate from normal patterns. We evaluate the method on 500 LoRA adapters -- 400 clean, and 100 poisoned for Llama-3.2-3B on instruction and reasoning datasets: Alpaca, Dolly, GSM8K, ARC-Challenge, SQuADv2, NaturalQuestions, HumanEval, and GLUE dataset. We achieve 97\% detection accuracy with less than 2\% false positives. 

---
# ScrapeGraphAI-100k: A Large-Scale Dataset for LLM-Based Web Information Extraction 

**Authors**: William Brach, Francesco Zuppichini, Marco Vinciguerra, Lorenzo Padoan  

**Link**: [PDF](https://arxiv.org/pdf/2602.15189)  

**Abstract**: The use of large language models for web information extraction is becoming increasingly fundamental to modern web information retrieval pipelines. However, existing datasets tend to be small, synthetic or text-only, failing to capture the structural context of the web. We introduce ScrapeGraphAI-100k, a large-scale dataset comprising real-world LLM extraction events, collected via opt-in ScrapeGraphAI telemetry during Q2 and Q3 of 2025. Starting from 9M events, we deduplicate and balance by schema to produce 93,695 examples spanning diverse domains and languages. Each instance includes Markdown content, a prompt, a JSON schema, the LLM response, and complexity/validation metadata. We characterize the datasets structural diversity and its failure modes as schema complexity increases. We also provide a fine-tuning experiment showing that a small language model (1.7B) trained on a subset narrows the gap to larger baselines (30B), underscoring the datasets utility for efficient extraction. ScrapeGraphAI-100k enables fine-tuning small models, benchmarking structured extraction, and studying schema induction for web IR indexing, and is publicly available on HuggingFace. 

---
# Seeing to Generalize: How Visual Data Corrects Binding Shortcuts 

**Authors**: Nicolas Buzeta, Felipe del Rio, Cristian Hinostroza, Denis Parra, Hans Lobel, Rodrigo Toro Icarte  

**Link**: [PDF](https://arxiv.org/pdf/2602.15183)  

**Abstract**: Vision Language Models (VLMs) are designed to extend Large Language Models (LLMs) with visual capabilities, yet in this work we observe a surprising phenomenon: VLMs can outperform their underlying LLMs on purely text-only tasks, particularly in long-context information retrieval. To investigate this effect, we build a controlled synthetic retrieval task and find that a transformer trained only on text achieves perfect in-distribution accuracy but fails to generalize out of distribution, while subsequent training on an image-tokenized version of the same task nearly doubles text-only OOD performance. Mechanistic interpretability reveals that visual training changes the model's internal binding strategy: text-only training encourages positional shortcuts, whereas image-based training disrupts them through spatial translation invariance, forcing the model to adopt a more robust symbolic binding mechanism that persists even after text-only examples are reintroduced. We further characterize how binding strategies vary across training regimes, visual encoders, and initializations, and show that analogous shifts occur during pretrained LLM-to-VLM transitions. Our findings suggest that cross-modal training can enhance reasoning and generalization even for tasks grounded in a single modality. 

---
# Protecting Language Models Against Unauthorized Distillation through Trace Rewriting 

**Authors**: Xinhang Ma, William Yeoh, Ning Zhang, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2602.15143)  

**Abstract**: Knowledge distillation is a widely adopted technique for transferring capabilities from LLMs to smaller, more efficient student models. However, unauthorized use of knowledge distillation takes unfair advantage of the considerable effort and cost put into developing frontier models. We investigate methods for modifying teacher-generated reasoning traces to achieve two objectives that deter unauthorized distillation: (1) \emph{anti-distillation}, or degrading the training usefulness of query responses, and (2) \emph{API watermarking}, which embeds verifiable signatures in student models. We introduce several approaches for dynamically rewriting a teacher's reasoning outputs while preserving answer correctness and semantic coherence. Two of these leverage the rewriting capabilities of LLMs, while others use gradient-based techniques. Our experiments show that a simple instruction-based rewriting approach achieves a strong anti-distillation effect while maintaining or even improving teacher performance. Furthermore, we show that our rewriting approach also enables highly reliable watermark detection with essentially no false alarms. 

---
