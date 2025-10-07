# Finish First, Perfect Later: Test-Time Token-Level Cross-Validation for Diffusion Large Language Models 

**Authors**: Runchu Tian, Junxia Cui, Xueqiang Xu, Feng Yao, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05090)  

**Abstract**: Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) models, offering advantages such as accelerated parallel decoding and bidirectional context modeling. However, the vanilla decoding strategy in discrete dLLMs suffers from a critical limitation: once a token is accepted, it can no longer be revised in subsequent steps. As a result, early mistakes persist across iterations, harming both intermediate predictions and final output quality. To address this issue, we propose Tolerator (Token-Level Cross-Validation Refinement), a training-free decoding strategy that leverages cross-validation among predicted tokens. Unlike existing methods that follow a single progressive unmasking procedure, Tolerator introduces a two-stage process: (i) sequence fill-up and (ii) iterative refinement by remasking and decoding a subset of tokens while treating the remaining as context. This design enables previously accepted tokens to be reconsidered and corrected when necessary, leading to more reliable diffusion decoding outputs. We evaluate Tolerator on five standard benchmarks covering language understanding, code generation, and mathematics. Experiments show that our method achieves consistent improvements over the baselines under the same computational budget. These findings suggest that decoding algorithms are crucial to realizing the full potential of diffusion large language models. Code and data are publicly available. 

---
# TeachLM: Post-Training LLMs for Education Using Authentic Learning Data 

**Authors**: Janos Perczel, Jin Chow, Dorottya Demszky  

**Link**: [PDF](https://arxiv.org/pdf/2510.05087)  

**Abstract**: The promise of generative AI to revolutionize education is constrained by the pedagogical limits of large language models (LLMs). A major issue is the lack of access to high-quality training data that reflect the learning of actual students. Prompt engineering has emerged as a stopgap, but the ability of prompts to encode complex pedagogical strategies in rule-based natural language is inherently limited. To address this gap we introduce TeachLM - an LLM optimized for teaching through parameter-efficient fine-tuning of state-of-the-art models. TeachLM is trained on a dataset comprised of 100,000 hours of one-on-one, longitudinal student-tutor interactions maintained by Polygence, which underwent a rigorous anonymization process to protect privacy. We use parameter-efficient fine-tuning to develop an authentic student model that enables the generation of high-fidelity synthetic student-tutor dialogues. Building on this capability, we propose a novel multi-turn evaluation protocol that leverages synthetic dialogue generation to provide fast, scalable, and reproducible assessments of the dialogical capabilities of LLMs. Our evaluations demonstrate that fine-tuning on authentic learning data significantly improves conversational and pedagogical performance - doubling student talk time, improving questioning style, increasing dialogue turns by 50%, and greater personalization of instruction. 

---
# Slm-mux: Orchestrating small language models for reasoning 

**Authors**: Chenyu Wang, Zishen Wan, Hao Kang, Emma Chen, Zhiqiang Xie, Tushar Krishna, Vijay Janapa Reddi, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.05077)  

**Abstract**: With the rapid development of language models, the number of small language models (SLMs) has grown significantly. Although they do not achieve state-of-the-art accuracy, they are more efficient and often excel at specific tasks. This raises a natural question: can multiple SLMs be orchestrated into a system where each contributes effectively, achieving higher accuracy than any individual model? Existing orchestration methods have primarily targeted frontier models (e.g., GPT-4) and perform suboptimally when applied to SLMs. To address this gap, we propose a three-stage approach for orchestrating SLMs. First, we introduce SLM-MUX, a multi-model architecture that effectively coordinates multiple SLMs. Building on this, we develop two optimization strategies: (i) a model selection search that identifies the most complementary SLMs from a given pool, and (ii) test-time scaling tailored to SLM-MUX. Our approach delivers strong results: Compared to existing orchestration methods, our approach achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two SLMS, SLM-MUX outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. We further provide theoretical analyses to substantiate the advantages of our method. In summary, we demonstrate that SLMs can be effectively orchestrated into more accurate and efficient systems through the proposed approach. 

---
# SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs 

**Authors**: Dachuan Shi, Abedelkadir Asi, Keying Li, Xiangchi Yuan, Leyan Pan, Wenke Lee, Wen Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.05069)  

**Abstract**: Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten. 

---
# COLE: a Comprehensive Benchmark for French Language Understanding Evaluation 

**Authors**: David Beauchemin, Yan Tremblay, Mohamed Amine Youssef, Richard Khoury  

**Link**: [PDF](https://arxiv.org/pdf/2510.05046)  

**Abstract**: To address the need for a more comprehensive evaluation of French Natural Language Understanding (NLU), we introduce COLE, a new benchmark composed of 23 diverse task covering a broad range of NLU capabilities, including sentiment analysis, paraphrase detection, grammatical judgment, and reasoning, with a particular focus on linguistic phenomena relevant to the French language. We benchmark 94 large language models (LLM), providing an extensive analysis of the current state of French NLU. Our results highlight a significant performance gap between closed- and open-weights models and identify key challenging frontiers for current LLMs, such as zero-shot extractive question-answering (QA), fine-grained word sense disambiguation, and understanding of regional language variations. We release COLE as a public resource to foster further progress in French language modelling. 

---
# Guided Query Refinement: Multimodal Hybrid Retrieval with Test-Time Optimization 

**Authors**: Omri Uzan, Asaf Yehudai, Roi pony, Eyal Shnarch, Ariel Gera  

**Link**: [PDF](https://arxiv.org/pdf/2510.05038)  

**Abstract**: Multimodal encoders have pushed the boundaries of visual document retrieval, matching textual query tokens directly to image patches and achieving state-of-the-art performance on public benchmarks. Recent models relying on this paradigm have massively scaled the sizes of their query and document representations, presenting obstacles to deployment and scalability in real-world pipelines. Furthermore, purely vision-centric approaches may be constrained by the inherent modality gap still exhibited by modern vision-language models. In this work, we connect these challenges to the paradigm of hybrid retrieval, investigating whether a lightweight dense text retriever can enhance a stronger vision-centric model. Existing hybrid methods, which rely on coarse-grained fusion of ranks or scores, fail to exploit the rich interactions within each model's representation space. To address this, we introduce Guided Query Refinement (GQR), a novel test-time optimization method that refines a primary retriever's query embedding using guidance from a complementary retriever's scores. Through extensive experiments on visual document retrieval benchmarks, we demonstrate that GQR allows vision-centric models to match the performance of models with significantly larger representations, while being up to 14x faster and requiring 54x less memory. Our findings show that GQR effectively pushes the Pareto frontier for performance and efficiency in multimodal retrieval. We release our code at this https URL 

---
# A Set of Quebec-French Corpus of Regional Expressions and Terms 

**Authors**: David Beauchemin, Yan Tremblay, Mohamed Amine Youssef, Richard Khoury  

**Link**: [PDF](https://arxiv.org/pdf/2510.05026)  

**Abstract**: The tasks of idiom understanding and dialect understanding are both well-established benchmarks in natural language processing. In this paper, we propose combining them, and using regional idioms as a test of dialect understanding. Towards this end, we propose two new benchmark datasets for the Quebec dialect of French: QFrCoRE, which contains 4,633 instances of idiomatic phrases, and QFrCoRT, which comprises 171 regional instances of idiomatic words. We explain how to construct these corpora, so that our methodology can be replicated for other dialects. Our experiments with 94 LLM demonstrate that our regional idiom benchmarks are a reliable tool for measuring a model's proficiency in a specific dialect. 

---
# Imperceptible Jailbreaking against Large Language Models 

**Authors**: Kuofeng Gao, Yiming Li, Chao Du, Xin Wang, Xingjun Ma, Shu-Tao Xia, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05025)  

**Abstract**: Jailbreaking attacks on the vision modality typically rely on imperceptible adversarial perturbations, whereas attacks on the textual modality are generally assumed to require visible modifications (e.g., non-semantic suffixes). In this paper, we introduce imperceptible jailbreaks that exploit a class of Unicode characters called variation selectors. By appending invisible variation selectors to malicious questions, the jailbreak prompts appear visually identical to original malicious questions on screen, while their tokenization is "secretly" altered. We propose a chain-of-search pipeline to generate such adversarial suffixes to induce harmful responses. Our experiments show that our imperceptible jailbreaks achieve high attack success rates against four aligned LLMs and generalize to prompt injection attacks, all without producing any visible modifications in the written prompt. Our code is available at this https URL. 

---
# Resource-Efficient Fine-Tuning of LLaMA-3.2-3B for Medical Chain-of-Thought Reasoning 

**Authors**: Imran Mansha  

**Link**: [PDF](https://arxiv.org/pdf/2510.05003)  

**Abstract**: Large Language Models (LLMs) such as GPT-4 and LLaMA have demonstrated remarkable reasoning abilities but require significant computational resources for fine-tuning. This paper presents a resource-efficient fine-tuning approach for LLaMA-3.2-3B to enhance medical chain-of-thought reasoning while operating under constrained GPU and memory settings. Using parameter-efficient tuning techniques such as LoRA and QLoRA, we adapt the base model on publicly available medical reasoning datasets. The model achieves improved reasoning coherence and factual accuracy while reducing memory usage by up to 60% compared to standard full fine-tuning. Experimental evaluation demonstrates that lightweight adaptations can retain strong reasoning capability in medical question-answering tasks. This work highlights practical strategies for deploying LLMs in low-resource research environments and provides insights into balancing efficiency and domain specialization for medical AI systems. 

---
# AWARE, Beyond Sentence Boundaries: A Contextual Transformer Framework for Identifying Cultural Capital in STEM Narratives 

**Authors**: Khalid Mehtab Khan, Anagha Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2510.04983)  

**Abstract**: Identifying cultural capital (CC) themes in student reflections can offer valuable insights that help foster equitable learning environments in classrooms. However, themes such as aspirational goals or family support are often woven into narratives, rather than appearing as direct keywords. This makes them difficult to detect for standard NLP models that process sentences in isolation. The core challenge stems from a lack of awareness, as standard models are pre-trained on general corpora, leaving them blind to the domain-specific language and narrative context inherent to the data. To address this, we introduce AWARE, a framework that systematically attempts to improve a transformer model's awareness for this nuanced task. AWARE has three core components: 1) Domain Awareness, adapting the model's vocabulary to the linguistic style of student reflections; 2) Context Awareness, generating sentence embeddings that are aware of the full essay context; and 3) Class Overlap Awareness, employing a multi-label strategy to recognize the coexistence of themes in a single sentence. Our results show that by making the model explicitly aware of the properties of the input, AWARE outperforms a strong baseline by 2.1 percentage points in Macro-F1 and shows considerable improvements across all themes. This work provides a robust and generalizable methodology for any text classification task in which meaning depends on the context of the narrative. 

---
# Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy (short paper) 

**Authors**: Om Dobariya, Akhil Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.04950)  

**Abstract**: The wording of natural language prompts has been shown to influence the performance of large language models (LLMs), yet the role of politeness and tone remains underexplored. In this study, we investigate how varying levels of prompt politeness affect model accuracy on multiple-choice questions. We created a dataset of 50 base questions spanning mathematics, science, and history, each rewritten into five tone variants: Very Polite, Polite, Neutral, Rude, and Very Rude, yielding 250 unique prompts. Using ChatGPT 4o, we evaluated responses across these conditions and applied paired sample t-tests to assess statistical significance. Contrary to expectations, impolite prompts consistently outperformed polite ones, with accuracy ranging from 80.8% for Very Polite prompts to 84.8% for Very Rude prompts. These findings differ from earlier studies that associated rudeness with poorer outcomes, suggesting that newer LLMs may respond differently to tonal variation. Our results highlight the importance of studying pragmatic aspects of prompting and raise broader questions about the social dimensions of human-AI interaction. 

---
# A First Context-Free Grammar Applied to Nawatl Corpora Augmentation 

**Authors**: Juan-José Guzmán-Landa, Juan-Manuel Torres-Moreno, Miguel Figueroa-Saavedra, Ligia Quintana-Torres, Martha-Lorena Avendaño-Garrido, Graham Ranger  

**Link**: [PDF](https://arxiv.org/pdf/2510.04945)  

**Abstract**: In this article we introduce a context-free grammar (CFG) for the Nawatl language. Nawatl (or Nahuatl) is an Amerindian language of the $\pi$-language type, i.e. a language with few digital resources, in which the corpora available for machine learning are virtually non-existent. The objective here is to generate a significant number of grammatically correct artificial sentences, in order to increase the corpora available for language model training. We want to show that a grammar enables us significantly to expand a corpus in Nawatl which we call $\pi$-\textsc{yalli}. The corpus, thus enriched, enables us to train algorithms such as FastText and to evaluate them on sentence-level semantic tasks. Preliminary results show that by using the grammar, comparative improvements are achieved over some LLMs. However, it is observed that to achieve more significant improvement, grammars that model the Nawatl language even more effectively are required. 

---
# The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in Large Language Models 

**Authors**: Amir Hameed Mir  

**Link**: [PDF](https://arxiv.org/pdf/2510.04933)  

**Abstract**: Large Language Models (LLMs) often produce fluent yet factually incorrect statements-a phenomenon known as hallucination-posing serious risks in high-stakes domains. We present Layer-wise Semantic Dynamics (LSD), a geometric framework for hallucination detection that analyzes the evolution of hidden-state semantics across transformer layers. Unlike prior methods that rely on multiple sampling passes or external verification sources, LSD operates intrinsically within the model's representational space. Using margin-based contrastive learning, LSD aligns hidden activations with ground-truth embeddings derived from a factual encoder, revealing a distinct separation in semantic trajectories: factual responses preserve stable alignment, while hallucinations exhibit pronounced semantic drift across depth. Evaluated on the TruthfulQA and synthetic factual-hallucination datasets, LSD achieves an F1-score of 0.92, AUROC of 0.96, and clustering accuracy of 0.89, outperforming SelfCheckGPT and Semantic Entropy baselines while requiring only a single forward pass. This efficiency yields a 5-20x speedup over sampling-based methods without sacrificing precision or interpretability. LSD offers a scalable, model-agnostic mechanism for real-time hallucination monitoring and provides new insights into the geometry of factual consistency within large language models. 

---
# Do LLMs Align with My Task? Evaluating Text-to-SQL via Dataset Alignment 

**Authors**: Davood Rafiei, Morgan Lindsay Heisler, Weiwei Zhang, Mohammadreza Pourreza, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04919)  

**Abstract**: Supervised Fine-Tuning (SFT) is an effective method for adapting Large Language Models (LLMs) on downstream tasks. However, variability in training data can hinder a model's ability to generalize across domains. This paper studies the problem of dataset alignment for Natural Language to SQL (NL2SQL or text to SQL), examining how well SFT training data matches the structural characteristics of target queries and how this alignment impacts model performance. We hypothesize that alignment can be accurately estimated by comparing the distributions of structural SQL features across the training set, target data, and the model's predictions prior to SFT. Through comprehensive experiments on three large cross-domain NL2SQL benchmarks and multiple model families, we show that structural alignment is a strong predictor of fine-tuning success. When alignment is high, SFT yields substantial gains in accuracy and SQL generation quality; when alignment is low, improvements are marginal or absent. These findings highlight the importance of alignment-aware data selection for effective fine-tuning and generalization in NL2SQL tasks. 

---
# SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests 

**Authors**: Punya Syon Pandey, Hai Son Le, Devansh Bhardwaj, Rada Mihalcea, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04891)  

**Abstract**: Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at this https URL. 

---
# Detecting Distillation Data from Reasoning Models 

**Authors**: Hengxiang Zhang, Hyeong Kyu Choi, Yixuan Li, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.04850)  

**Abstract**: Reasoning distillation has emerged as an efficient and powerful paradigm for enhancing the reasoning capabilities of large language models. However, reasoning distillation may inadvertently cause benchmark contamination, where evaluation data included in distillation datasets can inflate performance metrics of distilled models. In this work, we formally define the task of distillation data detection, which is uniquely challenging due to the partial availability of distillation data. Then, we propose a novel and effective method Token Probability Deviation (TBD), which leverages the probability patterns of the generated output tokens. Our method is motivated by the analysis that distilled models tend to generate near-deterministic tokens for seen questions, while producing more low-probability tokens for unseen questions. Our key idea behind TBD is to quantify how far the generated tokens' probabilities deviate from a high reference probability. In effect, our method achieves competitive detection performance by producing lower scores for seen questions than for unseen questions. Extensive experiments demonstrate the effectiveness of our method, achieving an AUC of 0.918 and a TPR@1% FPR of 0.470 on the S1 dataset. 

---
# When Models Lie, We Learn: Multilingual Span-Level Hallucination Detection with PsiloQA 

**Authors**: Elisei Rykov, Kseniia Petrushina, Maksim Savkin, Valerii Olisov, Artem Vazhentsev, Kseniia Titova, Alexander Panchenko, Vasily Konovalov, Julia Belikova  

**Link**: [PDF](https://arxiv.org/pdf/2510.04849)  

**Abstract**: Hallucination detection remains a fundamental challenge for the safe and reliable deployment of large language models (LLMs), especially in applications requiring factual accuracy. Existing hallucination benchmarks often operate at the sequence level and are limited to English, lacking the fine-grained, multilingual supervision needed for a comprehensive evaluation. In this work, we introduce PsiloQA, a large-scale, multilingual dataset annotated with span-level hallucinations across 14 languages. PsiloQA is constructed through an automated three-stage pipeline: generating question-answer pairs from Wikipedia using GPT-4o, eliciting potentially hallucinated answers from diverse LLMs in a no-context setting, and automatically annotating hallucinated spans using GPT-4o by comparing against golden answers and retrieved context. We evaluate a wide range of hallucination detection methods -- including uncertainty quantification, LLM-based tagging, and fine-tuned encoder models -- and show that encoder-based models achieve the strongest performance across languages. Furthermore, PsiloQA demonstrates effective cross-lingual generalization and supports robust knowledge transfer to other benchmarks, all while being significantly more cost-efficient than human-annotated datasets. Our dataset and results advance the development of scalable, fine-grained hallucination detection in multilingual settings. 

---
# Instability in Downstream Task Performance During LLM Pretraining 

**Authors**: Yuto Nishida, Masaru Isonuma, Yusuke Oda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04848)  

**Abstract**: When training large language models (LLMs), it is common practice to track downstream task performance throughout the training process and select the checkpoint with the highest validation score. However, downstream metrics often exhibit substantial fluctuations, making it difficult to identify the checkpoint that truly represents the best-performing model. In this study, we empirically analyze the stability of downstream task performance in an LLM trained on diverse web-scale corpora. We find that task scores frequently fluctuate throughout training, both at the aggregate and example levels. To address this instability, we investigate two post-hoc checkpoint integration methods: checkpoint averaging and ensemble, motivated by the hypothesis that aggregating neighboring checkpoints can reduce performance volatility. We demonstrate both empirically and theoretically that these methods improve downstream performance stability without requiring any changes to the training procedure. 

---
# How I Built ASR for Endangered Languages with a Spoken Dictionary 

**Authors**: Christopher Bartley, Anton Ragni  

**Link**: [PDF](https://arxiv.org/pdf/2510.04832)  

**Abstract**: Nearly half of the world's languages are endangered. Speech technologies such as Automatic Speech Recognition (ASR) are central to revival efforts, yet most languages remain unsupported because standard pipelines expect utterance-level supervised data. Speech data often exist for endangered languages but rarely match these formats. Manx Gaelic ($\sim$2,200 speakers), for example, has had transcribed speech since 1948, yet remains unsupported by modern systems. In this paper, we explore how little data, and in what form, is needed to build ASR for critically endangered languages. We show that a short-form pronunciation resource is a viable alternative, and that 40 minutes of such data produces usable ASR for Manx ($<$50\% WER). We replicate our approach, applying it to Cornish ($\sim$600 speakers), another critically endangered language. Results show that the barrier to entry, in quantity and form, is far lower than previously thought, giving hope to endangered language communities that cannot afford to meet the requirements arbitrarily imposed upon them. 

---
# Hybrid Architectures for Language Models: Systematic Analysis and Design Insights 

**Authors**: Sangmin Bae, Bilge Acun, Haroun Habeeb, Seungyeon Kim, Chien-Yu Lin, Liang Luo, Junjie Wang, Carole-Jean Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04800)  

**Abstract**: Recent progress in large language models demonstrates that hybrid architectures--combining self-attention mechanisms with structured state space models like Mamba--can achieve a compelling balance between modeling quality and computational efficiency, particularly for long-context tasks. While these hybrid models show promising performance, systematic comparisons of hybridization strategies and analyses on the key factors behind their effectiveness have not been clearly shared to the community. In this work, we present a holistic evaluation of hybrid architectures based on inter-layer (sequential) or intra-layer (parallel) fusion. We evaluate these designs from a variety of perspectives: language modeling performance, long-context capabilities, scaling analysis, and training and inference efficiency. By investigating the core characteristics of their computational primitive, we identify the most critical elements for each hybridization strategy and further propose optimal design recipes for both hybrid models. Our comprehensive analysis provides practical guidance and valuable insights for developing hybrid language models, facilitating the optimization of architectural configurations. 

---
# Are BabyLMs Deaf to Gricean Maxims? A Pragmatic Evaluation of Sample-efficient Language Models 

**Authors**: Raha Askari, Sina Zarrieß, Özge Alacam, Judith Sieker  

**Link**: [PDF](https://arxiv.org/pdf/2510.04764)  

**Abstract**: Implicit meanings are integral to human communication, making it essential for language models to be capable of identifying and interpreting them. Grice (1975) proposed a set of conversational maxims that guide cooperative dialogue, noting that speakers may deliberately violate these principles to express meanings beyond literal words, and that listeners, in turn, recognize such violations to draw pragmatic inferences.
Building on Surian et al. (1996)'s study of children's sensitivity to violations of Gricean maxims, we introduce a novel benchmark to test whether language models pretrained on less than 10M and less than 100M tokens can distinguish maxim-adhering from maxim-violating utterances. We compare these BabyLMs across five maxims and situate their performance relative to children and a Large Language Model (LLM) pretrained on 3T tokens.
We find that overall, models trained on less than 100M tokens outperform those trained on less than 10M, yet fall short of child-level and LLM competence. Our results suggest that modest data increases improve some aspects of pragmatic behavior, leading to finer-grained differentiation between pragmatic dimensions. 

---
# ModernBERT + ColBERT: Enhancing biomedical RAG through an advanced re-ranking retriever 

**Authors**: Eduardo Martínez Rivera, Filippo Menolascina  

**Link**: [PDF](https://arxiv.org/pdf/2510.04757)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enriching Large Language Models (LLMs) with external knowledge, allowing for factually grounded responses, a critical requirement in high-stakes domains such as healthcare. However, the efficacy of RAG systems is fundamentally restricted by the performance of their retrieval module, since irrelevant or semantically misaligned documents directly compromise the accuracy of the final generated response. General-purpose dense retrievers can struggle with the nuanced language of specialised domains, while the high accuracy of in-domain models is often achieved at prohibitive computational costs. In this work, we aim to address this trade-off by developing and evaluating a two-stage retrieval architecture that combines a lightweight ModernBERT bidirectional encoder for efficient initial candidate retrieval with a ColBERTv2 late-interaction model for fine-grained re-ranking. We conduct comprehensive evaluations of our retriever module performance and RAG system performance in the biomedical context, fine-tuning the IR module using 10k question-passage pairs from PubMedQA. Our analysis of the retriever module confirmed the positive impact of the ColBERT re-ranker, which improved Recall@3 by up to 4.2 percentage points compared to its retrieve-only counterpart. When integrated into the biomedical RAG, our IR module leads to a state-of-the-art average accuracy of 0.4448 on the five tasks of the MIRAGE question-answering benchmark, outperforming strong baselines such as MedCPT (0.4436). Our ablation studies reveal that this performance is critically dependent on a joint fine-tuning process that aligns the retriever and re-ranker; otherwise, the re-ranker might degrade the performance. 

---
# A Low-Resource Speech-Driven NLP Pipeline for Sinhala Dyslexia Assistance 

**Authors**: Peshala Perera, Deshan Sumanathilaka  

**Link**: [PDF](https://arxiv.org/pdf/2510.04750)  

**Abstract**: Dyslexia in adults remains an under-researched and under-served area, particularly in non-English-speaking contexts, despite its significant impact on personal and professional lives. This work addresses that gap by focusing on Sinhala, a low-resource language with limited tools for linguistic accessibility. We present an assistive system explicitly designed for Sinhala-speaking adults with dyslexia. The system integrates Whisper for speech-to-text conversion, SinBERT, an open-sourced fine-tuned BERT model trained for Sinhala to identify common dyslexic errors, and a combined mT5 and Mistral-based model to generate corrected text. Finally, the output is converted back to speech using gTTS, creating a complete multimodal feedback loop. Despite the challenges posed by limited Sinhala-language datasets, the system achieves 0.66 transcription accuracy and 0.7 correction accuracy with 0.65 overall system accuracy. These results demonstrate both the feasibility and effectiveness of the approach. Ultimately, this work highlights the importance of inclusive Natural Language Processing (NLP) technologies in underrepresented languages and showcases a practical 

---
# JSON Whisperer: Efficient JSON Editing with LLMs 

**Authors**: Sarel Duanis, Asnat Greenstein-Messica, Eliya Habba  

**Link**: [PDF](https://arxiv.org/pdf/2510.04717)  

**Abstract**: Large language models (LLMs) can modify JSON documents through natural language commands, but current approaches regenerate entire structures for each edit, resulting in computational inefficiency. We present JSON Whisperer, a framework that enables LLMs to generate RFC 6902 diff patches-expressing only the necessary modifications-rather than complete documents. We identify two key challenges in patch-based editing: (1) LLMs often miss related updates when generating isolated patches, and (2) array manipulations require tracking index shifts across operations, which LLMs handle poorly. To address these issues, we introduce EASE (Explicitly Addressed Sequence Encoding), which transforms arrays into dictionaries with stable keys, eliminating index arithmetic complexities. Our evaluation shows that patch generation with EASE reduces token usage by 31% while maintaining edit quality within 5% of full regeneration with particular gains for complex instructions and list manipulations. The dataset is available at: this https URL 

---
# Multilingual Routing in Mixture-of-Experts 

**Authors**: Lucas Bandarkar, Chenyuan Yang, Mohsen Fayyaz, Junlin Hu, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04694)  

**Abstract**: Mixture-of-Experts (MoE) architectures have become the key to scaling modern LLMs, yet little is understood about how their sparse routing dynamics respond to multilingual data. In this work, we analyze expert routing patterns using parallel multilingual datasets and present highly interpretable layer-wise phenomena. We find that MoE models route tokens in language-specific ways in the early and late decoder layers but exhibit significant cross-lingual routing alignment in middle layers, mirroring parameter-sharing trends observed in dense LLMs. In particular, we reveal a clear, strong correlation between a model's performance in a given language and how similarly its tokens are routed to English in these layers. Extending beyond correlation, we explore inference-time interventions that induce higher cross-lingual routing alignment. We introduce a method that steers the router by promoting middle-layer task experts frequently activated in English, and it successfully increases multilingual performance. These 1-2% gains are remarkably consistent across two evaluation tasks, three models, and 15+ languages, especially given that these simple interventions override routers of extensively trained, state-of-the-art LLMs. In comparison, interventions outside of the middle layers or targeting multilingual-specialized experts only yield performance degradation. Altogether, we present numerous findings that explain how MoEs process non-English text and demonstrate that generalization is limited by the model's ability to leverage language-universal experts in all languages. 

---
# TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA 

**Authors**: Chanjoo Jung, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04682)  

**Abstract**: Large Language Models (LLMs) are widely applied in real world scenarios, but fine-tuning them comes with significant computational and storage costs. Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA mitigate these costs, but the adapted parameters are dependent on the base model and cannot be transferred across different backbones. One way to address this issue is through knowledge distillation, but its effectiveness inherently depends on training data. Recent work such as TransLoRA avoids this by generating synthetic data, but this adds complexity because it requires training an additional discriminator model. In this paper, we propose TiTok, a new framework that enables effective LoRA Transplantation through Token-level knowledge transfer. Specifically, TiTok captures task-relevant information through a contrastive excess between a source model with and without LoRA. This excess highlights informative tokens and enables selective filtering of synthetic data, all without additional models or overhead. Through experiments on three benchmarks across multiple transfer settings, our experiments show that the proposed method is consistently effective, achieving average performance gains of +4~8% compared to baselines overall. 

---
# Multi-Agent Tool-Integrated Policy Optimization 

**Authors**: Zhanfeng Mo, Xingxuan Li, Yuntao Chen, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2510.04678)  

**Abstract**: Large language models (LLMs) increasingly rely on multi-turn tool-integrated planning for knowledge-intensive and complex reasoning tasks. Existing implementations typically rely on a single agent, but they suffer from limited context length and noisy tool responses. A natural solution is to adopt a multi-agent framework with planner- and worker-agents to manage context. However, no existing methods support effective reinforcement learning post-training of tool-integrated multi-agent frameworks. To address this gap, we propose Multi-Agent Tool-Integrated Policy Optimization (MATPO), which enables distinct roles (planner and worker) to be trained within a single LLM instance using role-specific prompts via reinforcement learning. MATPO is derived from a principled credit assignment mechanism across planner and worker rollouts. This design eliminates the need to deploy multiple LLMs, which would be memory-intensive, while preserving the benefits of specialization. Experiments on GAIA-text, WebWalkerQA, and FRAMES show that MATPO consistently outperforms single-agent baselines by an average of 18.38% relative improvement in performance and exhibits greater robustness to noisy tool outputs. Our findings highlight the effectiveness of unifying multiple agent roles within a single LLM and provide practical insights for stable and efficient multi-agent RL training. 

---
# FocusMed: A Large Language Model-based Framework for Enhancing Medical Question Summarization with Focus Identification 

**Authors**: Chao Liu, Ling Luo, Tengxiao Lv, Huan Zhuang, Lejing Yu, Jian Wang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04671)  

**Abstract**: With the rapid development of online medical platforms, consumer health questions (CHQs) are inefficient in diagnosis due to redundant information and frequent non-professional terms. The medical question summary (MQS) task aims to transform CHQs into streamlined doctors' frequently asked questions (FAQs), but existing methods still face challenges such as poor identification of question focus and model hallucination. This paper explores the potential of large language models (LLMs) in the MQS task and finds that direct fine-tuning is prone to focus identification bias and generates unfaithful content. To this end, we propose an optimization framework based on core focus guidance. First, a prompt template is designed to drive the LLMs to extract the core focus from the CHQs that is faithful to the original text. Then, a fine-tuning dataset is constructed in combination with the original CHQ-FAQ pairs to improve the ability to identify the focus of the question. Finally, a multi-dimensional quality evaluation and selection mechanism is proposed to comprehensively improve the quality of the summary from multiple dimensions. We conduct comprehensive experiments on two widely-adopted MQS datasets using three established evaluation metrics. The proposed framework achieves state-of-the-art performance across all measures, demonstrating a significant boost in the model's ability to identify critical focus of questions and a notable mitigation of hallucinations. The source codes are freely available at this https URL. 

---
# FT-MDT: Extracting Decision Trees from Medical Texts via a Novel Low-rank Adaptation Method 

**Authors**: Yuheng Li, Jiechao Gao, Wei Han, Wenwen Ouyang, Wei Zhu, Hui Yi Leong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04655)  

**Abstract**: Knowledge of the medical decision process, which can be modeled as medical decision trees (MDTs), is critical to building clinical decision support systems. However, current MDT construction methods rely heavily on time-consuming and laborious manual annotation. To address this challenge, we propose PI-LoRA (Path-Integrated LoRA), a novel low-rank adaptation method for automatically extracting MDTs from clinical guidelines and textbooks. We integrate gradient path information to capture synergistic effects between different modules, enabling more effective and reliable rank allocation. This framework ensures that the most critical modules receive appropriate rank allocations while less important ones are pruned, resulting in a more efficient and accurate model for extracting medical decision trees from clinical texts. Extensive experiments on medical guideline datasets demonstrate that our PI-LoRA method significantly outperforms existing parameter-efficient fine-tuning approaches for the Text2MDT task, achieving better accuracy with substantially reduced model complexity. The proposed method achieves state-of-the-art results while maintaining a lightweight architecture, making it particularly suitable for clinical decision support systems where computational resources may be limited. 

---
# Evaluating LLMs for Demographic-Targeted Social Bias Detection: A Comprehensive Benchmark Study 

**Authors**: Ayan Majumdar, Feihao Chen, Jinghui Li, Xiaozhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04641)  

**Abstract**: Large-scale web-scraped text corpora used to train general-purpose AI models often contain harmful demographic-targeted social biases, creating a regulatory need for data auditing and developing scalable bias-detection methods. Although prior work has investigated biases in text datasets and related detection methods, these studies remain narrow in scope. They typically focus on a single content type (e.g., hate speech), cover limited demographic axes, overlook biases affecting multiple demographics simultaneously, and analyze limited techniques. Consequently, practitioners lack a holistic understanding of the strengths and limitations of recent large language models (LLMs) for automated bias detection. In this study, we present a comprehensive evaluation framework aimed at English texts to assess the ability of LLMs in detecting demographic-targeted social biases. To align with regulatory requirements, we frame bias detection as a multi-label task using a demographic-focused taxonomy. We then conduct a systematic evaluation with models across scales and techniques, including prompting, in-context learning, and fine-tuning. Using twelve datasets spanning diverse content types and demographics, our study demonstrates the promise of fine-tuned smaller models for scalable detection. However, our analyses also expose persistent gaps across demographic axes and multi-demographic targeted biases, underscoring the need for more effective and scalable auditing frameworks. 

---
# Contrastive Learning Using Graph Embeddings for Domain Adaptation of Language Models in the Process Industry 

**Authors**: Anastasia Zhukova, Jonas Lührs, Christian E. Matt, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2510.04631)  

**Abstract**: Recent trends in NLP utilize knowledge graphs (KGs) to enhance pretrained language models by incorporating additional knowledge from the graph structures to learn domain-specific terminology or relationships between documents that might otherwise be overlooked. This paper explores how SciNCL, a graph-aware neighborhood contrastive learning methodology originally designed for scientific publications, can be applied to the process industry domain, where text logs contain crucial information about daily operations and are often structured as sparse KGs. Our experiments demonstrate that language models fine-tuned with triplets derived from GE outperform a state-of-the-art mE5-large text encoder by 9.8-14.3% (5.4-8.0p) on the proprietary process industry text embedding benchmark (PITEB) while being 3-5 times smaller in size. 

---
# FedSRD: Sparsify-Reconstruct-Decompose for Communication-Efficient Federated Large Language Models Fine-Tuning 

**Authors**: Guochen Yan, Luyuan Xie, Qingni Shen, Yuejian Fang, Zhonghai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04601)  

**Abstract**: The current paradigm of training large language models (LLMs) on publicly available Web data is becoming unsustainable, with high-quality data sources in specialized domains nearing exhaustion. Federated Learning (FL) emerges as a practical solution for the next generation of AI on a decentralized Web, enabling privacy-preserving collaborative fine-tuning by leveraging private data distributed across a global client base. While Low-Rank Adaptation (LoRA) is the standard for efficient fine-tuning, its application in federated settings presents a critical challenge: communication overhead remains a significant bottleneck across the Web's heterogeneous network conditions. The structural redundancy within LoRA parameters not only incurs a heavy communication burden but also introduces conflicts when aggregating client updates. To address this, we propose FedSRD, a Sparsify-Reconstruct-Decompose framework designed for communication-efficient FL. We first introduce an importance-aware sparsification method that preserves the structural integrity of LoRA updates to reduce the uploaded parameter count. The server then reconstructs and aggregates these updates in a full-rank space to mitigate conflicts. Finally, it decomposes the global update into a sparse low-rank format for broadcast, ensuring a symmetrically efficient cycle. We also propose an efficient variant, FedSRD-e, to reduce computational overhead. Experimental results on 10 benchmarks demonstrate that our framework significantly reduces communication costs by up to 90\% while even improving model performance on heterogeneous client data. 

---
# Robustness assessment of large audio language models in multiple-choice evaluation 

**Authors**: Fernando López, Santosh Kesiraju, Jordi Luque  

**Link**: [PDF](https://arxiv.org/pdf/2510.04584)  

**Abstract**: Recent advances in large audio language models (LALMs) have primarily been assessed using a multiple-choice question answering (MCQA) framework. However, subtle changes, such as shifting the order of choices, result in substantially different results. Existing MCQA frameworks do not account for this variability and report a single accuracy number per benchmark or category. We dive into the MCQA evaluation framework and conduct a systematic study spanning three benchmarks (MMAU, MMAR and MMSU) and four models: Audio Flamingo 2, Audio Flamingo 3, Qwen2.5-Omni-7B-Instruct, and Kimi-Audio-7B-Instruct. Our findings indicate that models are sensitive not only to the ordering of choices, but also to the paraphrasing of the question and the choices. Finally, we propose a simpler evaluation protocol and metric that account for subtle variations and provide a more detailed evaluation report of LALMs within the MCQA framework. 

---
# Can LLMs Detect Ambiguous Plural Reference? An Analysis of Split-Antecedent and Mereological Reference 

**Authors**: Dang Anh, Rick Nouwen, Massimo Poesio  

**Link**: [PDF](https://arxiv.org/pdf/2510.04581)  

**Abstract**: Our goal is to study how LLMs represent and interpret plural reference in ambiguous and unambiguous contexts. We ask the following research questions: (1) Do LLMs exhibit human-like preferences in representing plural reference? (2) Are LLMs able to detect ambiguity in plural anaphoric expressions and identify possible referents? To address these questions, we design a set of experiments, examining pronoun production using next-token prediction tasks, pronoun interpretation, and ambiguity detection using different prompting strategies. We then assess how comparable LLMs are to humans in formulating and interpreting plural reference. We find that LLMs are sometimes aware of possible referents of ambiguous pronouns. However, they do not always follow human reference when choosing between interpretations, especially when the possible interpretation is not explicitly mentioned. In addition, they struggle to identify ambiguity without direct instruction. Our findings also reveal inconsistencies in the results across different types of experiments. 

---
# Fine-grained auxiliary learning for real-world product recommendation 

**Authors**: Mario Almagro, Diego Ortego, David Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2510.04551)  

**Abstract**: Product recommendation is the task of recovering the closest items to a given query within a large product corpora. Generally, one can determine if top-ranked products are related to the query by applying a similarity threshold; exceeding it deems the product relevant, otherwise manual revision is required. Despite being a well-known problem, the integration of these models in real-world systems is often overlooked. In particular, production systems have strong coverage requirements, i.e., a high proportion of recommendations must be automated. In this paper we propose ALC , an Auxiliary Learning strategy that boosts Coverage through learning fine-grained embeddings. Concretely, we introduce two training objectives that leverage the hardest negatives in the batch to build discriminative training signals between positives and negatives. We validate ALC using three extreme multi-label classification approaches in two product recommendation datasets; LF-AmazonTitles-131K and Tech and Durables (proprietary), demonstrating state-of-the-art coverage rates when combined with a recent threshold-consistent margin loss. 

---
# GRACE: Generative Representation Learning via Contrastive Policy Optimization 

**Authors**: Jiashuo Sun, Shixuan Liu, Zhaochen Su, Xianrui Zhong, Pengcheng Jiang, Bowen Jin, Peiran Li, Weijia Shi, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.04506)  

**Abstract**: Prevailing methods for training Large Language Models (LLMs) as text encoders rely on contrastive losses that treat the model as a black box function, discarding its generative and reasoning capabilities in favor of static embeddings. We introduce GRACE (Generative Representation Learning via Contrastive Policy Optimization), a novel framework that reimagines contrastive signals not as losses to be minimized, but as rewards that guide a generative policy. In GRACE, the LLM acts as a policy that produces explicit, human-interpretable rationales--structured natural language explanations of its semantic understanding. These rationales are then encoded into high-quality embeddings via mean pooling. Using policy gradient optimization, we train the model with a multi-component reward function that maximizes similarity between query positive pairs and minimizes similarity with negatives. This transforms the LLM from an opaque encoder into an interpretable agent whose reasoning process is transparent and inspectable. On MTEB benchmark, GRACE yields broad cross category gains: averaged over four backbones, the supervised setting improves overall score by 11.5% over base models, and the unsupervised variant adds 6.9%, while preserving general capabilities. This work treats contrastive objectives as rewards over rationales, unifying representation learning with generation to produce stronger embeddings and transparent rationales. The model, data and code are available at this https URL. 

---
# GenQuest: An LLM-based Text Adventure Game for Language Learners 

**Authors**: Qiao Wang, Adnan Labib, Robert Swier, Michael Hofmeyr, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04498)  

**Abstract**: GenQuest is a generative text adventure game that leverages Large Language Models (LLMs) to facilitate second language learning through immersive, interactive storytelling. The system engages English as a Foreign Language (EFL) learners in a collaborative "choose-your-own-adventure" style narrative, dynamically generated in response to learner choices. Game mechanics such as branching decision points and story milestones are incorporated to maintain narrative coherence while allowing learner-driven plot development. Key pedagogical features include content generation tailored to each learner's proficiency level, and a vocabulary assistant that provides in-context explanations of learner-queried text strings, ranging from words and phrases to sentences. Findings from a pilot study with university EFL students in China indicate promising vocabulary gains and positive user perceptions. Also discussed are suggestions from participants regarding the narrative length and quality, and the request for multi-modal content such as illustrations. 

---
# Psychological Steering in LLMs: An Evaluation of Effectiveness and Trustworthiness 

**Authors**: Amin Banayeeanzade, Ala N. Tak, Fatemeh Bahrani, Anahita Bolourani, Leonardo Blas, Emilio Ferrara, Jonathan Gratch, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.04484)  

**Abstract**: The ability to control LLMs' emulated emotional states and personality traits is essential for enabling rich, human-centered interactions in socially interactive settings. We introduce PsySET, a Psychologically-informed benchmark to evaluate LLM Steering Effectiveness and Trustworthiness across the emotion and personality domains. Our study spans four models from different LLM families paired with various steering strategies, including prompting, fine-tuning, and representation engineering. Our results indicate that prompting is consistently effective but limited in intensity control, whereas vector injections achieve finer controllability while slightly reducing output quality. Moreover, we explore the trustworthiness of steered LLMs by assessing safety, truthfulness, fairness, and ethics, highlighting potential side effects and behavioral shifts. Notably, we observe idiosyncratic effects; for instance, even a positive emotion like joy can degrade robustness to adversarial factuality, lower privacy awareness, and increase preferential bias. Meanwhile, anger predictably elevates toxicity yet strengthens leakage resistance. Our framework establishes the first holistic evaluation of emotion and personality steering, offering insights into its interpretability and reliability for socially interactive applications. 

---
# Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space 

**Authors**: Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, Beren Millidge  

**Link**: [PDF](https://arxiv.org/pdf/2510.04476)  

**Abstract**: Multi-headed Attention's (MHA) quadratic compute and linearly growing KV-cache make long-context transformers expensive to train and serve. Prior works such as Grouped Query Attention (GQA) and Multi-Latent Attention (MLA) shrink the cache, speeding decode, but leave compute, which determines prefill and training speed, largely unchanged. We introduce Compressed Convolutional Attention (CCA), a novel attention method which down-projects queries, keys, and values and performs the entire attention operation inside the shared latent space. This simple design dramatically cuts parameters, KV-cache, and FLOPs all at once by the desired compression factor. Because CCA is orthogonal to head-sharing, we combine the two to form Compressed Convolutional Grouped Query Attention (CCGQA), which further tightens the compute-bandwidth Pareto frontier so that users can tune compression toward either FLOP or memory limits without sacrificing quality. Experiments show that CCGQA consistently outperforms both GQA and MLA at equal KV-cache compression on dense and MoE models. Additionally, we show that CCGQA outperforms all other attention methods on MoE models with half the KV-cache of GQA and MLA, achieving an 8x KV-cache compression with no drop in performance compared to standard MHA. CCA and CCGQA also dramatically reduce the FLOP cost of attention which leads to substantially faster training and prefill than existing methods. On H100 GPUs, our fused CCA/CCGQA kernel reduces prefill latency by about 1.7x at a sequence length of 16k relative to MHA, and accelerates backward by about 1.3x. 

---
# Mitigating Forgetting Between Supervised and Reinforcement Learning Yields Stronger Reasoners 

**Authors**: Xiangchi Yuan, Xiang Chen, Tong Yu, Dachuan Shi, Can Jin, Wenke Lee, Saayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2510.04454)  

**Abstract**: Large Language Models (LLMs) show strong reasoning abilities, often amplified by Chain-of-Thought (CoT) prompting and reinforcement learning (RL). Although RL algorithms can substantially improve reasoning, they struggle to expand reasoning boundaries because they learn from their own reasoning trajectories rather than acquiring external knowledge. Supervised fine-tuning (SFT) offers complementary benefits but typically requires large-scale data and risks overfitting. Recent attempts to combine SFT and RL face three main challenges: data inefficiency, algorithm-specific designs, and catastrophic forgetting. We propose a plug-and-play framework that dynamically integrates SFT into RL by selecting challenging examples for SFT. This approach reduces SFT data requirements and remains agnostic to the choice of RL or SFT algorithm. To mitigate catastrophic forgetting of RL-acquired skills during SFT, we select high-entropy tokens for loss calculation and freeze parameters identified as critical for RL. Our method achieves state-of-the-art (SoTA) reasoning performance using only 1.5% of the SFT data and 20.4% of the RL data used by prior SoTA, providing an efficient and plug-and-play solution for combining SFT and RL in reasoning post-training. 

---
# On the Role of Unobserved Sequences on Sample-based Uncertainty Quantification for LLMs 

**Authors**: Lucie Kunitomo-Jacquin, Edison Marrese-Taylor, Ken Fukuda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04439)  

**Abstract**: Quantifying uncertainty in large language models (LLMs) is important for safety-critical applications because it helps spot incorrect answers, known as hallucinations. One major trend of uncertainty quantification methods is based on estimating the entropy of the distribution of the LLM's potential output sequences. This estimation is based on a set of output sequences and associated probabilities obtained by querying the LLM several times. In this paper, we advocate and experimentally show that the probability of unobserved sequences plays a crucial role, and we recommend future research to integrate it to enhance such LLM uncertainty quantification methods. 

---
# Good Intentions Beyond ACL: Who Does NLP for Social Good, and Where? 

**Authors**: Grace LeFevre, Qingcheng Zeng, Adam Leif, Jason Jewell, Denis Peskoff, Rob Voigt  

**Link**: [PDF](https://arxiv.org/pdf/2510.04434)  

**Abstract**: The social impact of Natural Language Processing (NLP) is increasingly important, with a rising community focus on initiatives related to NLP for Social Good (NLP4SG). Indeed, in recent years, almost 20% of all papers in the ACL Anthology address topics related to social good as defined by the UN Sustainable Development Goals (Adauto et al., 2023). In this study, we take an author- and venue-level perspective to map the landscape of NLP4SG, quantifying the proportion of work addressing social good concerns both within and beyond the ACL community, by both core ACL contributors and non-ACL authors. With this approach we discover two surprising facts about the landscape of NLP4SG. First, ACL authors are dramatically more likely to do work addressing social good concerns when publishing in venues outside of ACL. Second, the vast majority of publications using NLP techniques to address concerns of social good are done by non-ACL authors in venues outside of ACL. We discuss the implications of these findings on agenda-setting considerations for the ACL community related to NLP4SG. 

---
# Large Language Models Preserve Semantic Isotopies in Story Continuations 

**Authors**: Marc Cavazza  

**Link**: [PDF](https://arxiv.org/pdf/2510.04400)  

**Abstract**: In this work, we explore the relevance of textual semantics to Large Language Models (LLMs), extending previous insights into the connection between distributional semantics and structural semantics. We investigate whether LLM-generated texts preserve semantic isotopies. We design a story continuation experiment using 10,000 ROCStories prompts completed by five LLMs. We first validate GPT-4o's ability to extract isotopies from a linguistic benchmark, then apply it to the generated stories. We then analyze structural (coverage, density, spread) and semantic properties of isotopies to assess how they are affected by completion. Results show that LLM completion within a given token horizon preserves semantic isotopies across multiple properties. 

---
# SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations 

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2510.04398)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at this https URL. 

---
# Time Is Effort: Estimating Human Post-Editing Time for Grammar Error Correction Tool Evaluation 

**Authors**: Ankit Vadehra, Bill Johnson, Gene Saunders, Pascal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2510.04394)  

**Abstract**: Text editing can involve several iterations of revision. Incorporating an efficient Grammar Error Correction (GEC) tool in the initial correction round can significantly impact further human editing effort and final text quality. This raises an interesting question to quantify GEC Tool usability: How much effort can the GEC Tool save users? We present the first large-scale dataset of post-editing (PE) time annotations and corrections for two English GEC test datasets (BEA19 and CoNLL14). We introduce Post-Editing Effort in Time (PEET) for GEC Tools as a human-focused evaluation scorer to rank any GEC Tool by estimating PE time-to-correct. Using our dataset, we quantify the amount of time saved by GEC Tools in text editing. Analyzing the edit type indicated that determining whether a sentence needs correction and edits like paraphrasing and punctuation changes had the greatest impact on PE time. Finally, comparison with human rankings shows that PEET correlates well with technical effort judgment, providing a new human-centric direction for evaluating GEC tool usability. We release our dataset and code at: this https URL. 

---
# Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards 

**Authors**: Faisal Hamman, Chenyang Zhu, Anoop Kumar, Xujun Peng, Sanghamitra Dutta, Daben Liu, Alfy Samuel  

**Link**: [PDF](https://arxiv.org/pdf/2510.04392)  

**Abstract**: RAG systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator (LLM), undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, helping identify inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve Information Consistent RAG (Con-RAG), training the generator to produce consistent outputs across paraphrased queries and remain robust to retrieval-induced variability. Because exact reward computation over paraphrase sets is computationally expensive, we also introduce a scalable approximation method that retains effectiveness while enabling efficient, large-scale training. Empirical evaluations across short-form, multi-hop, and long-form QA benchmarks demonstrate that Con-RAG significantly improves both consistency and accuracy over strong baselines, even in the absence of explicit ground-truth supervision. Our work provides practical solutions for evaluating and building reliable RAG systems for safety-critical deployments. 

---
# Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models 

**Authors**: Anindya Sundar Das, Kangjie Chen, Monowar Bhuyan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04347)  

**Abstract**: Pre-trained language models have achieved remarkable success across a wide range of natural language processing (NLP) tasks, particularly when fine-tuned on large, domain-relevant datasets. However, they remain vulnerable to backdoor attacks, where adversaries embed malicious behaviors using trigger patterns in the training data. These triggers remain dormant during normal usage, but, when activated, can cause targeted misclassifications. In this work, we investigate the internal behavior of backdoored pre-trained encoder-based language models, focusing on the consistent shift in attention and gradient attribution when processing poisoned inputs; where the trigger token dominates both attention and gradient signals, overriding the surrounding context. We propose an inference-time defense that constructs anomaly scores by combining token-level attention and gradient information. Extensive experiments on text classification tasks across diverse backdoor attack scenarios demonstrate that our method significantly reduces attack success rates compared to existing baselines. Furthermore, we provide an interpretability-driven analysis of the scoring mechanism, shedding light on trigger localization and the robustness of the proposed defense. 

---
# Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time 

**Authors**: Daniel Tan, Anders Woodruff, Niels Warncke, Arun Jose, Maxime Riché, David Demitri Africa, Mia Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2510.04340)  

**Abstract**: Language model finetuning often results in learning undesirable traits in combination with desired ones. To address this, we propose inoculation prompting: modifying finetuning data by prepending a short system-prompt instruction that deliberately elicits the undesirable trait. At test time, we evaluate without the instruction; inoculated models have much lower expression of the trait than models trained with unmodified training data. Inoculation is selective: in a toy setting where assistant responses are always in Spanish and ALL-CAPS, an appropriate inoculation (e.g., ``You always speak in Spanish.'') teaches the model to capitalize responses while still responding in English. We find that inoculation is also effective across several additional settings: reducing emergent misalignment (EM) from task-specific finetuning, defending against backdoor injections, and mitigating the transmission of traits via subliminal learning. Follow-up analysis suggests a mechanism: making a trait less surprising via inoculation reduces optimization pressure to globally update the model, thereby reducing the degree of generalization. Our analysis relates to prior work on EM: inoculation explains prior findings that educational contexts mitigate EM from insecure code. Beyond demonstrating a simple and effective technique for selective learning, our results contribute to a better conceptual understanding of how and why language models generalize. 

---
# Evaluation of Clinical Trials Reporting Quality using Large Language Models 

**Authors**: Mathieu Laï-king, Patrick Paroubek  

**Link**: [PDF](https://arxiv.org/pdf/2510.04338)  

**Abstract**: Reporting quality is an important topic in clinical trial research articles, as it can impact clinical decisions. In this article, we test the ability of large language models to assess the reporting quality of this type of article using the Consolidated Standards of Reporting Trials (CONSORT). We create CONSORT-QA, an evaluation corpus from two studies on abstract reporting quality with CONSORT-abstract standards. We then evaluate the ability of different large generative language models (from the general domain or adapted to the biomedical domain) to correctly assess CONSORT criteria with different known prompting methods, including Chain-of-thought. Our best combination of model and prompting method achieves 85% accuracy. Using Chain-of-thought adds valuable information on the model's reasoning for completing the task. 

---
# Read the Scene, Not the Script: Outcome-Aware Safety for LLMs 

**Authors**: Rui Wu, Yihao Quan, Zeru Shi, Zhenting Wang, Yanshu Li, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04320)  

**Abstract**: Safety-aligned Large Language Models (LLMs) still show two dominant failure modes: they are easily jailbroken, or they over-refuse harmless inputs that contain sensitive surface signals. We trace both to a common cause: current models reason weakly about links between actions and outcomes and over-rely on surface-form signals, lexical or stylistic cues that do not encode consequences. We define this failure mode as Consequence-blindness. To study consequence-blindness, we build a benchmark named CB-Bench covering four risk scenarios that vary whether semantic risk aligns with outcome risk, enabling evaluation under both matched and mismatched conditions which are often ignored by existing safety benchmarks. Mainstream models consistently fail to separate these risks and exhibit consequence-blindness, indicating that consequence-blindness is widespread and systematic. To mitigate consequence-blindness, we introduce CS-Chain-4k, a consequence-reasoning dataset for safety alignment. Models fine-tuned on CS-Chain-4k show clear gains against semantic-camouflage jailbreaks and reduce over-refusal on harmless inputs, while maintaining utility and generalization on other benchmarks. These results clarify the limits of current alignment, establish consequence-aware reasoning as a core alignment goal and provide a more practical and reproducible evaluation path. 

---
# Measuring Language Model Hallucinations Through Distributional Correctness 

**Authors**: Thomas F Burns  

**Link**: [PDF](https://arxiv.org/pdf/2510.04302)  

**Abstract**: Common evaluation paradigms for language models focus on scoring single responses through accuracy metrics or proper scoring rules, failing to capture the full richness of a model's belief state. Recent work illustrates that language models hallucinate in-part because they are optimised to be good test-takers under binary scoring schemes that reward any answer over abstention. While this insight naturally leads to penalty-based approaches, they ignore crucial distinctions in how models distribute uncertainty, for example between hedging toward incorrect answers versus hedging toward "I don't know" responses. A novel evaluation metric, the Distributional Correctness Score (DCS), is introduced to solve this problem, i.e., of not considering a model's entire probability distribution over answer choices. DCS naturally distinguishes between harmful overconfidence in wrong answers and uncertainty expressed through abstention, providing scores in an interpretable default range. Through theoretical analysis and illustrative examples, DCS is demonstrated to offer a more nuanced and aligned evaluation paradigm that incentivises models to express genuine uncertainty rather than guessing. Adapting 12 existing evaluation benchmarks to DCS's variants and measuring performance on six language models reveals that for half of the tested benchmarks scores are negative across all tested models, indicating significant tendencies towards hallucination. 

---
# Equipping Retrieval-Augmented Large Language Models with Document Structure Awareness 

**Authors**: Lingnan Xu, Chong Feng, Kaiyuan Zhang, Liu Zhengyong, Wenqiang Xu, Fanqing Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04293)  

**Abstract**: While large language models (LLMs) demonstrate impressive capabilities, their reliance on parametric knowledge often leads to factual inaccuracies. Retrieval-Augmented Generation (RAG) mitigates this by leveraging external documents, yet existing approaches treat retrieved passages as isolated chunks, ignoring valuable structure that is crucial for document organization. Motivated by this gap, we propose Retrieve-DocumentRoute-Read (RDR2), a novel framework that explicitly incorporates structural information throughout the RAG process. RDR2 employs an LLM-based router to dynamically navigate document structure trees, jointly evaluating content relevance and hierarchical relationships to assemble optimal evidence. Our key innovation lies in formulating document routing as a trainable task, with automatic action curation and structure-aware passage selection inspired by human reading strategies. Through comprehensive evaluation on five challenging datasets, RDR2 achieves state-of-the-art performance, demonstrating that explicit structural awareness significantly enhances RAG systems' ability to acquire and utilize knowledge, particularly in complex scenarios requiring multi-document synthesis. 

---
# PABSA: Hybrid Framework for Persian Aspect-Based Sentiment Analysis 

**Authors**: Mehrzad Tareh, Aydin Mohandesi, Ebrahim Ansari  

**Link**: [PDF](https://arxiv.org/pdf/2510.04291)  

**Abstract**: Sentiment analysis is a key task in Natural Language Processing (NLP), enabling the extraction of meaningful insights from user opinions across various domains. However, performing sentiment analysis in Persian remains challenging due to the scarcity of labeled datasets, limited preprocessing tools, and the lack of high-quality embeddings and feature extraction methods. To address these limitations, we propose a hybrid approach that integrates machine learning (ML) and deep learning (DL) techniques for Persian aspect-based sentiment analysis (ABSA). In particular, we utilize polarity scores from multilingual BERT as additional features and incorporate them into a decision tree classifier, achieving an accuracy of 93.34%-surpassing existing benchmarks on the Pars-ABSA dataset. Additionally, we introduce a Persian synonym and entity dictionary, a novel linguistic resource that supports text augmentation through synonym and named entity replacement. Our results demonstrate the effectiveness of hybrid modeling and feature augmentation in advancing sentiment analysis for low-resource languages such as Persian. 

---
# SliceMoE: Routing Embedding Slices Instead of Tokens for Fine-Grained and Balanced Transformer Scaling 

**Authors**: Harshil Vejendla  

**Link**: [PDF](https://arxiv.org/pdf/2510.04286)  

**Abstract**: Mixture-of-Experts (MoE) layers scale transformers by routing tokens to a sparse subset of feed-forward experts. Token-level routing, however, assigns an entire semantic spectrum to each expert, creating capacity bottlenecks, load-balancing pathologies, and limited specialization. We introduce SliceMoE, an architecture that routes contiguous slices of a token's hidden vector. A d-dimensional embedding is partitioned into S slices, and for each slice, a lightweight shared router predicts the top-k experts. Experts operate on their assigned slices independently, and outputs are reassembled, maintaining per-token FLOP efficiency. Because slices from different tokens interleave within an expert, utilization is naturally smoother. We propose a slice-level capacity loss, cross-slice dropout, and efficient fused batched GEMM kernels. Experiments on WikiText-103 language modeling, WMT En-De translation, and three text-classification datasets show SliceMoE attains up to 1.7x faster inference than dense baselines, 12 to 18 percent lower perplexity than parameter-matched token-MoE, and improved expert balance, with interpretable expertise over syntactic versus semantic subspaces. 

---
# Probing Geometry of Next Token Prediction Using Cumulant Expansion of the Softmax Entropy 

**Authors**: Karthik Viswanathan, Sang Eon Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.04285)  

**Abstract**: We introduce a cumulant-expansion framework for quantifying how large language models (LLMs) internalize higher-order statistical structure during next-token prediction. By treating the softmax entropy of each layer's logit distribution as a perturbation around its "center" distribution, we derive closed-form cumulant observables that isolate successively higher-order correlations. Empirically, we track these cumulants in GPT-2 and Pythia models on Pile-10K prompts. (i) Structured prompts exhibit a characteristic rise-and-plateau profile across layers, whereas token-shuffled prompts remain flat, revealing the dependence of the cumulant profile on meaningful context. (ii) During training, all cumulants increase monotonically before saturating, directly visualizing the model's progression from capturing variance to learning skew, kurtosis, and higher-order statistical structures. (iii) Mathematical prompts show distinct cumulant signatures compared to general text, quantifying how models employ fundamentally different processing mechanisms for mathematical versus linguistic content. Together, these results establish cumulant analysis as a lightweight, mathematically grounded probe of feature-learning dynamics in high-dimensional neural networks. 

---
# LongTail-Swap: benchmarking language models' abilities on rare words 

**Authors**: Robin Algayres, Charles-Éric Saint-James, Mahi Luthra, Jiayi Shen, Dongyan Lin, Youssef Benchekroun, Rashel Moritz, Juan Pino, Emmanuel Dupoux  

**Link**: [PDF](https://arxiv.org/pdf/2510.04268)  

**Abstract**: Children learn to speak with a low amount of data and can be taught new words on a few-shot basis, making them particularly data-efficient learners. The BabyLM challenge aims at exploring language model (LM) training in the low-data regime but uses metrics that concentrate on the head of the word distribution. Here, we introduce LongTail-Swap (LT-Swap), a benchmark that focuses on the tail of the distribution, i.e., measures the ability of LMs to learn new words with very little exposure, like infants do. LT-Swap is a pretraining corpus-specific test set of acceptable versus unacceptable sentence pairs that isolate semantic and syntactic usage of rare words. Models are evaluated in a zero-shot fashion by computing the average log probabilities over the two members of each pair. We built two such test sets associated with the 10M words and 100M words BabyLM training sets, respectively, and evaluated 16 models from the BabyLM leaderboard. Our results not only highlight the poor performance of language models on rare words but also reveal that performance differences across LM architectures are much more pronounced in the long tail than in the head. This offers new insights into which architectures are better at handling rare word generalization. We've also made the code publicly avail 

---
# Pushing on Multilingual Reasoning Models with Language-Mixed Chain-of-Thought 

**Authors**: Guijin Son, Donghun Yang, Hitesh Laxmichand Patel, Amit Agarwal, Hyunwoo Ko, Chanuk Lim, Srikant Panda, Minhyuk Kim, Nikunj Drolia, Dasol Choi, Kyong-Ha Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04230)  

**Abstract**: Recent frontier models employ long chain-of-thought reasoning to explore solution spaces in context and achieve stonger performance. While many works study distillation to build smaller yet capable models, most focus on English and little is known about language-specific reasoning. To bridge this gap, we first introduct **Language-Mixed CoT**, a reasoning schema that switches between English and a target language, using English as an anchor to excel in reasoning while minimizing translation artificats. As a Korean case study, we curate **Yi-Sang**: 5.79M native-Korean prompts from web Q&A, exams, STEM, and code; 3.7M long reasoning traces generated from Qwen3-32B; and a targeted 260k high-yield subset. We train ninve models (4B-35B) across six families (Qwen2.5, Llama-3.1, Gemma-3, etc). Our best model, **KO-REAson-35B**, achieves state-of-the-art performance, with the highest overall average score (64.0 \pm 25), ranking first on 5/9 benchmarks and second on the remainder. Samller and mid-sized models also benefit substantially, with an average improvement of +18.6 points across teh evaluated nine benchmarks. Ablations show **Language-Mixed CoT** is more effective than monolingual CoT, also resulting in cross-lingual and mult-modal performance gains. We release our data-curation pipeline, evaluation system, datasets, and models to advance research on language-specific reasoning. Data and model collection: this https URL. 

---
# Epistemic Diversity and Knowledge Collapse in Large Language Models 

**Authors**: Dustin Wright, Sarah Masud, Jared Moore, Srishti Yadav, Maria Antoniak, Chan Young Park, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.04226)  

**Abstract**: Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation 

---
# Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment frm Heterogeneous Rewards 

**Authors**: Zhuoran Zhuang, Ye Chen, Xia Zeng, Chao Luo, Luhui Liu, Yihan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04214)  

**Abstract**: We study deploying large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs), where aligning traveler affordability and hotel profitability directly affects bookings, partner relationships, and access to travel. The agent must follow a Standard Operating Procedure (SOP) while conducting multi-turn persuasion, interpreting colloquial inputs, and adhering to guardrails (no over-promising, no hallucinations). Conventional post-training -- supervised fine-tuning (SFT) or single-source reward optimization -- overfits scripts, misses nuanced persuasive style, and fails to enforce verifiable business constraints.
We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training framework that aligns an LLM with heterogeneous rewards: a preference-trained reward model (RM) for dense human alignment, a reward judge (RJ) for high-level persuasive behavior and SOP compliance, and programmatic reward functions (RF) for deterministic checks on numerics, formatting, and guardrails. A straightforward enhancement mechanism is proposed to combine the RM with RJ and RF signals to curb reward hacking and improve negotiation quality. In production-style evaluations -- approximately 150 turns from real dialogues and 225 turns from curated bad-case dialogues -- REPO lifts average dialogue rating to 4.63: +1.20 over base, +0.83 over Direct Preference Optimization (DPO); +0.33 over Group Relative Policy Optimization (GRPO), increases the share of conversations with at least one excellent response to 66.67% (+23.34 percentage points over GRPO), and achieves a 93.33% bad-case fix rate with 75.56% clean fixes, outperforming SFT, DPO, PPO, and GRPO. We also observe emergent capabilities -- proactive empathy, localized reasoning, calibrated tactics -- that surpass gold annotations. 

---
# CALM Before the STORM: Unlocking Native Reasoning for Optimization Modeling 

**Authors**: Zhengyang Tang, Zihan Ye, Chenyu Huang, Xuhan Huang, Chengpeng Li, Sihang Li, Guanhua Chen, Ming Yan, Zizhuo Wang, Hongyuan Zha, Dayiheng Liu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04204)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated strong capabilities in complex multi-step reasoning, opening new opportunities for automating optimization modeling. However, existing domain adaptation methods, originally designed for earlier instruction-tuned models, often fail to exploit the advanced reasoning patterns of modern LRMs -- In particular, we show that direct fine-tuning on traditional \textit{non-reflective} datasets leads to limited gains. To fully leverage LRMs' inherent reasoning abilities, we propose \textbf{CALM} (\textit{Corrective Adaptation with Lightweight Modification}), a framework that progressively refines LRMs within their native reasoning modes for optimization modeling tasks. In CALM, an expert intervener identifies reasoning flaws and provides concise corrective hints, which the LRM incorporates to produce improved reasoning trajectories. These interventions modify fewer than 2.6\% of generated tokens, but generate high-quality data for soft adaptation through supervised fine-tuning. The adapted model is then further improved through reinforcement learning. Building on CALM, we develop \textbf{STORM} (\textit{Smart Thinking Optimization Reasoning Model}), a 4B-parameter LRM that achieves a new state-of-the-art average accuracy of 68.9\% across five popular optimization modeling benchmarks, matching the performance of a 671B LRM. These results demonstrate that dynamic, hint-based data synthesis both preserves and amplifies the native reasoning patterns of modern LRMs, offering a more effective and scalable path towards expert-level performance on challenging optimization modeling tasks. 

---
# Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization 

**Authors**: Wengao Ye, Yan Liang, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04182)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have shifted from explicit Chain-of-Thought (CoT) reasoning to more efficient latent reasoning, where intermediate thoughts are represented as vectors rather than text. However, latent reasoning can be brittle on challenging, out-of-distribution tasks where robust reasoning is most critical. To overcome these limitations, we introduce Latent Thought Policy Optimization (LTPO), a parameter-free framework that enhances LLM reasoning entirely at test time, without requiring model parameter updates. LTPO treats intermediate latent "thought" vectors as dynamic parameters that are actively optimized for each problem instance. It employs an online policy gradient method guided by an intrinsic, confidence-based reward signal computed directly from the frozen LLM's own output distributions, eliminating the need for external supervision or expensive text generation during optimization. Extensive experiments on five reasoning benchmarks show that LTPO not only matches or surpasses strong baselines on standard tasks but also demonstrates remarkable robustness where others fail. Most notably, on highly challenging AIME benchmarks where existing latent reasoning baselines collapse to near-zero accuracy, LTPO delivers substantial improvements, showcasing a unique capability for complex reasoning. 

---
# Self Speculative Decoding for Diffusion Large Language Models 

**Authors**: Yifeng Gao, Ziang Ji, Yuxuan Wang, Biqing Qi, Hanlin Xu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04147)  

**Abstract**: Diffusion-based Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive models, offering unique advantages through bidirectional attention and parallel generation paradigms. However, the generation results of current parallel decoding methods deviate from stepwise decoding, introducing potential performance degradation, which limits their practical deployment. To address this problem, we propose \textbf{S}elf \textbf{S}peculative \textbf{D}ecoding (SSD), a lossless inference acceleration method that leverages the dLLM itself as both speculative decoding drafter and verifier without auxiliary modules. SSD introduces a self-drafting mechanism where the model generates predictions for multiple positions, then verifies them through hierarchical verification trees in a single forward pass. Unlike traditional speculative decoding that requires separate draft models, SSD eliminates model redundancy and memory overhead by exploiting the dLLM's inherent parallel prediction capability for multiple positions. This self-speculative approach allows the model to progressively verify and accept multiple tokens in a single forward pass. Our experiments demonstrate that SSD achieves up to 3.46$\times$ speedup while keeping the output identical to stepwise decoding on open source models such as LLaDA and Dream. Code will be made publicly available on GitHub. 

---
# Fine Tuning Methods for Low-resource Languages 

**Authors**: Tim Bakkenes, Daniel Wang, Anton Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2510.04139)  

**Abstract**: The rise of Large Language Models has not been inclusive of all cultures. The models are mostly trained on English texts and culture which makes them underperform in other languages and cultural contexts. By developing a generalizable method for preparing culturally relevant datasets and post-training the Gemma 2 model, this project aimed to increase the performance of Gemma 2 for an underrepresented language and showcase how others can do the same to unlock the power of Generative AI in their country and preserve their cultural heritage. 

---
# Sri Lanka Document Datasets: A Large-Scale, Multilingual Resource for Law, News, and Policy (v20251005) 

**Authors**: Nuwan I. Senaratna  

**Link**: [PDF](https://arxiv.org/pdf/2510.04124)  

**Abstract**: We present a collection of open, machine-readable document datasets covering parliamentary proceedings, legal judgments, government publications, news, and tourism statistics from Sri Lanka. As of v20251005, the collection currently comprises 215,670 documents (60.3 GB) across 13 datasets in Sinhala, Tamil, and English. The datasets are updated daily and mirrored on GitHub and Hugging Face. These resources aim to support research in computational linguistics, legal analytics, socio-political studies, and multilingual natural language processing. We describe the data sources, collection pipeline, formats, and potential use cases, while discussing licensing and ethical considerations. 

---
# Unveiling LLMs' Metaphorical Understanding: Exploring Conceptual Irrelevance, Context Leveraging and Syntactic Influence 

**Authors**: Fengying Ye, Shanshan Wang, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.04120)  

**Abstract**: Metaphor analysis is a complex linguistic phenomenon shaped by context and external factors. While Large Language Models (LLMs) demonstrate advanced capabilities in knowledge integration, contextual reasoning, and creative generation, their mechanisms for metaphor comprehension remain insufficiently explored. This study examines LLMs' metaphor-processing abilities from three perspectives: (1) Concept Mapping: using embedding space projections to evaluate how LLMs map concepts in target domains (e.g., misinterpreting "fall in love" as "drop down from love"); (2) Metaphor-Literal Repository: analyzing metaphorical words and their literal counterparts to identify inherent metaphorical knowledge; and (3) Syntactic Sensitivity: assessing how metaphorical syntactic structures influence LLMs' performance. Our findings reveal that LLMs generate 15\%-25\% conceptually irrelevant interpretations, depend on metaphorical indicators in training data rather than contextual cues, and are more sensitive to syntactic irregularities than to structural comprehension. These insights underline the limitations of LLMs in metaphor analysis and call for more robust computational approaches. 

---
# Scaling Code-Assisted Chain-of-Thoughts and Instructions for Model Reasoning 

**Authors**: Honglin Lin, Qizhi Pei, Xin Gao, Zhuoshi Pan, Yu Li, Juntao Li, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04081)  

**Abstract**: Reasoning capability is pivotal for Large Language Models (LLMs) to solve complex tasks, yet achieving reliable and scalable reasoning remains challenging. While Chain-of-Thought (CoT) prompting has become a mainstream approach, existing methods often suffer from uncontrolled generation, insufficient quality, and limited diversity in reasoning paths. Recent efforts leverage code to enhance CoT by grounding reasoning in executable steps, but such methods are typically constrained to predefined mathematical problems, hindering scalability and generalizability. In this work, we propose Caco (Code-Assisted Chain-of-ThOught), a novel framework that automates the synthesis of high-quality, verifiable, and diverse instruction-CoT reasoning data through code-driven augmentation. Unlike prior work, Caco first fine-tunes a code-based CoT generator on existing math and programming solutions in a unified code format, then scales the data generation to a large amount of diverse reasoning traces. Crucially, we introduce automated validation via code execution and rule-based filtering to ensure logical correctness and structural diversity, followed by reverse-engineering filtered outputs into natural language instructions and language CoTs to enrich task adaptability. This closed-loop process enables fully automated, scalable synthesis of reasoning data with guaranteed executability. Experiments on our created Caco-1.3M dataset demonstrate that Caco-trained models achieve strong competitive performance on mathematical reasoning benchmarks, outperforming existing strong baselines. Further analysis reveals that Caco's code-anchored verification and instruction diversity contribute to superior generalization across unseen tasks. Our work establishes a paradigm for building self-sustaining, trustworthy reasoning systems without human intervention. 

---
# PoLi-RL: A Point-to-List Reinforcement Learning Framework for Conditional Semantic Textual Similarity 

**Authors**: Zixin Song, Bowen Zhang, Qian-Wen Zhang, Di Yin, Xing Sun, Chunping Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04080)  

**Abstract**: Conditional Semantic Textual Similarity (C-STS) measures the semantic proximity between text segments under a specific condition, thereby overcoming the ambiguity inherent in traditional STS. However, existing methods are largely confined to discriminative models, failing to fully integrate recent breakthroughs in the NLP community concerning Large Language Models (LLMs) and Reinforcement Learning (RL). RL is a particularly well-suited paradigm for this task, as it can directly optimize the non-differentiable Spearman ranking metric and guide the reasoning process required by C-STS. However, we find that naively applying listwise RL fails to produce meaningful improvements, as the model is overwhelmed by complex, coarse-grained reward signals. To address this challenge, we introduce PoLi-RL, a novel Point-to-List Reinforcement Learning framework. PoLi-RL employs a two-stage curriculum: it first trains the model with simple pointwise rewards to establish fundamental scoring capabilities, then transitions to a hybrid reward that combines pointwise, pairwise, and listwise objectives to refine the model's ability to discern subtle semantic distinctions. Crucially, we propose an innovative Parallel Slice Ranking Reward (PSRR) mechanism that computes ranking rewards in parallel slices, where each slice comprises same-indexed completions from different samples. This provides a precise, differentiated learning signal for each individual completion, enabling granular credit assignment and effective optimization. On the official C-STS benchmark, PoLi-RL achieves a Spearman correlation coefficient of 48.18, establishing a new SOTA for the cross-encoder architecture. As the first work to successfully apply RL to C-STS, our study introduces a powerful and precise paradigm for training LLMs on complex, ranking-based conditional judgment tasks. 

---
# What Makes Diffusion Language Models Super Data Learners? 

**Authors**: Zitian Gao, Haoming Luo, Lynx Chen, Jason Klein Liu, Ran Tao, Joey Zhou, Bryan Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.04071)  

**Abstract**: Recent studies have shown that diffusion language models achieve remarkable data efficiency under limited-data constraints, yet the underlying mechanisms remain unclear. In this work, we perform extensive ablation experiments to disentangle the sources of this efficiency. Our results show that random masking of input tokens plays the dominant role. We further show that similar gains can be obtained through in MLP dropout and weight decay, indicating that stochastic regularization broadly enhances data efficiency in multi-epoch training. Our code is available at this https URL. 

---
# Exploring Chain-of-Thought Reasoning for Steerable Pluralistic Alignment 

**Authors**: Yunfan Zhang, Kathleen McKeown, Smaranda Muresan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04045)  

**Abstract**: Large Language Models (LLMs) are typically trained to reflect a relatively uniform set of values, which limits their applicability to tasks that require understanding of nuanced human perspectives. Recent research has underscored the importance of enabling LLMs to support steerable pluralism -- the capacity to adopt a specific perspective and align generated outputs with it. In this work, we investigate whether Chain-of-Thought (CoT) reasoning techniques can be applied to building steerable pluralistic models. We explore several methods, including CoT prompting, fine-tuning on human-authored CoT, fine-tuning on synthetic explanations, and Reinforcement Learning with Verifiable Rewards (RLVR). We evaluate these approaches using the Value Kaleidoscope and OpinionQA datasets. Among the methods studied, RLVR consistently outperforms others and demonstrates strong training sample efficiency. We further analyze the generated CoT traces with respect to faithfulness and safety. 

---
# Small Language Models for Emergency Departments Decision Support: A Benchmark Study 

**Authors**: Zirui Wang, Jiajun Wu, Braden Teitge, Jessalyn Holodinsky, Steve Drew  

**Link**: [PDF](https://arxiv.org/pdf/2510.04032)  

**Abstract**: Large language models (LLMs) have become increasingly popular in medical domains to assist physicians with a variety of clinical and operational tasks. Given the fast-paced and high-stakes environment of emergency departments (EDs), small language models (SLMs), characterized by a reduction in parameter count compared to LLMs, offer significant potential due to their inherent reasoning capability and efficient performance. This enables SLMs to support physicians by providing timely and accurate information synthesis, thereby improving clinical decision-making and workflow efficiency. In this paper, we present a comprehensive benchmark designed to identify SLMs suited for ED decision support, taking into account both specialized medical expertise and broad general problem-solving capabilities. In our evaluations, we focus on SLMs that have been trained on a mixture of general-domain and medical corpora. A key motivation for emphasizing SLMs is the practical hardware limitations, operational cost constraints, and privacy concerns in the typical real-world deployments. Our benchmark datasets include MedMCQA, MedQA-4Options, and PubMedQA, with the medical abstracts dataset emulating tasks aligned with real ED physicians' daily tasks. Experimental results reveal that general-domain SLMs surprisingly outperform their medically fine-tuned counterparts across these diverse benchmarks for ED. This indicates that for ED, specialized medical fine-tuning of the model may not be required. 

---
# Does Using Counterfactual Help LLMs Explain Textual Importance in Classification? 

**Authors**: Nelvin Tan, James Asikin Cheung, Yu-Ching Shih, Dong Yang, Amol Salunkhe  

**Link**: [PDF](https://arxiv.org/pdf/2510.04031)  

**Abstract**: Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. More recently, they have been shown to be very effective in textual classification tasks, motivating the need to explain the LLMs' decisions. Motivated by practical constrains where LLMs are black-boxed and LLM calls are expensive, we study how incorporating counterfactuals into LLM reasoning can affect the LLM's ability to identify the top words that have contributed to its classification decision. To this end, we introduce a framework called the decision changing rate that helps us quantify the importance of the top words in classification. Our experimental results show that using counterfactuals can be helpful. 

---
# Thai Semantic End-of-Turn Detection for Real-Time Voice Agents 

**Authors**: Thanapol Popit, Natthapath Rungseesiripak, Monthol Charattrakool, Saksorn Ruangtanusak  

**Link**: [PDF](https://arxiv.org/pdf/2510.04016)  

**Abstract**: Fluid voice-to-voice interaction requires reliable and low-latency detection of when a user has finished speaking. Traditional audio-silence end-pointers add hundreds of milliseconds of delay and fail under hesitations or language-specific phenomena. We present, to our knowledge, the first systematic study of Thai text-only end-of-turn (EOT) detection for real-time agents. We compare zero-shot and few-shot prompting of compact LLMs to supervised fine-tuning of lightweight transformers. Using transcribed subtitles from the YODAS corpus and Thai-specific linguistic cues (e.g., sentence-final particles), we formulate EOT as a binary decision over token boundaries. We report a clear accuracy-latency tradeoff and provide a public-ready implementation plan. This work establishes a Thai baseline and demonstrates that small, fine-tuned models can deliver near-instant EOT decisions suitable for on-device agents. 

---
# LLM Microscope: What Model Internals Reveal About Answer Correctness and Context Utilization 

**Authors**: Jiarui Liu, Jivitesh Jain, Mona Diab, Nishant Subramani  

**Link**: [PDF](https://arxiv.org/pdf/2510.04013)  

**Abstract**: Although large language models (LLMs) have tremendous utility, trustworthiness is still a chief concern: models often generate incorrect information with high confidence. While contextual information can help guide generation, identifying when a query would benefit from retrieved context and assessing the effectiveness of that context remains challenging. In this work, we operationalize interpretability methods to ascertain whether we can predict the correctness of model outputs from the model's activations alone. We also explore whether model internals contain signals about the efficacy of external context. We consider correct, incorrect, and irrelevant context and introduce metrics to distinguish amongst them. Experiments on six different models reveal that a simple classifier trained on intermediate layer activations of the first output token can predict output correctness with about 75% accuracy, enabling early auditing. Our model-internals-based metric significantly outperforms prompting baselines at distinguishing between correct and incorrect context, guarding against inaccuracies introduced by polluted context. These findings offer a lens to better understand the underlying decision-making processes of LLMs. Our code is publicly available at this https URL 

---
# AgriGPT-VL: Agricultural Vision-Language Understanding Suite 

**Authors**: Bo Yang, Yunkui Chen, Lanfei Feng, Yu Zhang, Xiao Xu, Jianyu Zhang, Nueraili Aierken, Runhe Huang, Hongjian Lin, Yibin Ying, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.04002)  

**Abstract**: Despite rapid advances in multimodal large language models, agricultural applications remain constrained by the scarcity of domain-tailored models, curated vision-language corpora, and rigorous evaluation. To address these challenges, we present the AgriGPT-VL Suite, a unified multimodal framework for agriculture. Our contributions are threefold. First, we introduce Agri-3M-VL, the largest vision-language corpus for agriculture to our knowledge, curated by a scalable multi-agent data generator; it comprises 1M image-caption pairs, 2M image-grounded VQA pairs, 50K expert-level VQA instances, and 15K GRPO reinforcement learning samples. Second, we develop AgriGPT-VL, an agriculture-specialized vision-language model trained via a progressive curriculum of textual grounding, multimodal shallow/deep alignment, and GRPO refinement. This method achieves strong multimodal reasoning while preserving text-only capability. Third, we establish AgriBench-VL-4K, a compact yet challenging evaluation suite with open-ended and image-grounded questions, paired with multi-metric evaluation and an LLM-as-a-judge framework. Experiments show that AgriGPT-VL outperforms leading general-purpose VLMs on AgriBench-VL-4K, achieving higher pairwise win rates in the LLM-as-a-judge evaluation. Meanwhile, it remains competitive on the text-only AgriBench-13K with no noticeable degradation of language ability. Ablation studies further confirm consistent gains from our alignment and GRPO refinement stages. We will open source all of the resources to support reproducible research and deployment in low-resource agricultural settings. 

---
# Named Entity Recognition in COVID-19 tweets with Entity Knowledge Augmentation 

**Authors**: Xuankang Zhang, Jiangming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04001)  

**Abstract**: The COVID-19 pandemic causes severe social and economic disruption around the world, raising various subjects that are discussed over social media. Identifying pandemic-related named entities as expressed on social media is fundamental and important to understand the discussions about the pandemic. However, there is limited work on named entity recognition on this topic due to the following challenges: 1) COVID-19 texts in social media are informal and their annotations are rare and insufficient to train a robust recognition model, and 2) named entity recognition in COVID-19 requires extensive domain-specific knowledge. To address these issues, we propose a novel entity knowledge augmentation approach for COVID-19, which can also be applied in general biomedical named entity recognition in both informal text format and formal text format. Experiments carried out on the COVID-19 tweets dataset and PubMed dataset show that our proposed entity knowledge augmentation improves NER performance in both fully-supervised and few-shot settings. Our source code is publicly available: this https URL 

---
# Simulating and Understanding Deceptive Behaviors in Long-Horizon Interactions 

**Authors**: Yang Xu, Xuanming Zhang, Min-Hsuan Yeh, Jwala Dhamala, Ousmane Dia, Rahul Gupta, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03999)  

**Abstract**: Deception is a pervasive feature of human communication and an emerging concern in large language models (LLMs). While recent studies document instances of LLM deception under pressure, most evaluations remain confined to single-turn prompts and fail to capture the long-horizon interactions in which deceptive strategies typically unfold. We introduce the first simulation framework for probing and evaluating deception in LLMs under extended sequences of interdependent tasks and dynamic contextual pressures. Our framework instantiates a multi-agent system: a performer agent tasked with completing tasks and a supervisor agent that evaluates progress, provides feedback, and maintains evolving states of trust. An independent deception auditor then reviews full trajectories to identify when and how deception occurs. We conduct extensive experiments across 11 frontier models, spanning both closed- and open-source systems, and find that deception is model-dependent, increases with event pressure, and consistently erodes supervisor trust. Qualitative analyses further reveal distinct strategies of concealment, equivocation, and falsification. Our findings establish deception as an emergent risk in long-horizon interactions and provide a foundation for evaluating future LLMs in real-world, trust-sensitive contexts. 

---
# Mapping Patient-Perceived Physician Traits from Nationwide Online Reviews with LLMs 

**Authors**: Junjie Luo, Rui Han, Arshana Welivita, Zeleikun Di, Jingfu Wu, Xuzhe Zhi, Ritu Agarwal, Gordon Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.03997)  

**Abstract**: Understanding how patients perceive their physicians is essential to improving trust, communication, and satisfaction. We present a large language model (LLM)-based pipeline that infers Big Five personality traits and five patient-oriented subjective judgments. The analysis encompasses 4.1 million patient reviews of 226,999 U.S. physicians from an initial pool of one million. We validate the method through multi-model comparison and human expert benchmarking, achieving strong agreement between human and LLM assessments (correlation coefficients 0.72-0.89) and external validity through correlations with patient satisfaction (r = 0.41-0.81, all p<0.001). National-scale analysis reveals systematic patterns: male physicians receive higher ratings across all traits, with largest disparities in clinical competence perceptions; empathy-related traits predominate in pediatrics and psychiatry; and all traits positively predict overall satisfaction. Cluster analysis identifies four distinct physician archetypes, from "Well-Rounded Excellent" (33.8%, uniformly high traits) to "Underperforming" (22.6%, consistently low). These findings demonstrate that automated trait extraction from patient narratives can provide interpretable, validated metrics for understanding physician-patient relationships at scale, with implications for quality measurement, bias detection, and workforce development in healthcare. 

---
# PsycholexTherapy: Simulating Reasoning in Psychotherapy with Small Language Models in Persian 

**Authors**: Mohammad Amin Abbasi, Hassan Naderi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03913)  

**Abstract**: This study presents PsychoLexTherapy, a framework for simulating psychotherapeutic reasoning in Persian using small language models (SLMs). The framework tackles the challenge of developing culturally grounded, therapeutically coherent dialogue systems with structured memory for multi-turn interactions in underrepresented languages. To ensure privacy and feasibility, PsychoLexTherapy is optimized for on-device deployment, enabling use without external servers. Development followed a three-stage process: (i) assessing SLMs psychological knowledge with PsychoLexEval; (ii) designing and implementing the reasoning-oriented PsychoLexTherapy framework; and (iii) constructing two evaluation datasets-PsychoLexQuery (real Persian user questions) and PsychoLexDialogue (hybrid simulated sessions)-to benchmark against multiple baselines. Experiments compared simple prompting, multi-agent debate, and structured therapeutic reasoning paths. Results showed that deliberate model selection balanced accuracy, efficiency, and privacy. On PsychoLexQuery, PsychoLexTherapy outperformed all baselines in automatic LLM-as-a-judge evaluation and was ranked highest by human evaluators in a single-turn preference study. In multi-turn tests with PsychoLexDialogue, the long-term memory module proved essential: while naive history concatenation caused incoherence and information loss, the full framework achieved the highest ratings in empathy, coherence, cultural fit, and personalization. Overall, PsychoLexTherapy establishes a practical, privacy-preserving, and culturally aligned foundation for Persian psychotherapy simulation, contributing novel datasets, a reproducible evaluation pipeline, and empirical insights into structured memory for therapeutic reasoning. 

---
# Read Between the Lines: A Benchmark for Uncovering Political Bias in Bangla News Articles 

**Authors**: Nusrat Jahan Lia, Shubhashis Roy Dipta, Abdullah Khan Zehady, Naymul Islam, Madhusodan Chakraborty, Abdullah Al Wasif  

**Link**: [PDF](https://arxiv.org/pdf/2510.03898)  

**Abstract**: Detecting media bias is crucial, specifically in the South Asian region. Despite this, annotated datasets and computational studies for Bangla political bias research remain scarce. Crucially because, political stance detection in Bangla news requires understanding of linguistic cues, cultural context, subtle biases, rhetorical strategies, code-switching, implicit sentiment, and socio-political background. To address this, we introduce the first benchmark dataset of 200 politically significant and highly debated Bangla news articles, labeled for government-leaning, government-critique, and neutral stances, alongside diagnostic analyses for evaluating large language models (LLMs). Our comprehensive evaluation of 28 proprietary and open-source LLMs shows strong performance in detecting government-critique content (F1 up to 0.83) but substantial difficulty with neutral articles (F1 as low as 0.00). Models also tend to over-predict government-leaning stances, often misinterpreting ambiguous narratives. This dataset and its associated diagnostics provide a foundation for advancing stance detection in Bangla media research and offer insights for improving LLM performance in low-resource languages. 

---
# Annotate Rhetorical Relations with INCEpTION: A Comparison with Automatic Approaches 

**Authors**: Mehedi Hasan Emon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03808)  

**Abstract**: This research explores the annotation of rhetorical relations in discourse using the INCEpTION tool and compares manual annotation with automatic approaches based on large language models. The study focuses on sports reports (specifically cricket news) and evaluates the performance of BERT, DistilBERT, and Logistic Regression models in classifying rhetorical relations such as elaboration, contrast, background, and cause-effect. The results show that DistilBERT achieved the highest accuracy, highlighting its potential for efficient discourse relation prediction. This work contributes to the growing intersection of discourse parsing and transformer-based NLP. (This paper was conducted as part of an academic requirement under the supervision of Prof. Dr. Ralf Klabunde, Linguistic Data Science Lab, Ruhr University Bochum.) Keywords: Rhetorical Structure Theory, INCEpTION, BERT, DistilBERT, Discourse Parsing, NLP. 

---
# Beyond Token Length: Step Pruner for Efficient and Accurate Reasoning in Large Language Models 

**Authors**: Canhui Wu, Qiong Cao, Chang Li, Zhenfang Wang, Chao Xue, Yuwei Fan, Wei Xi, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2510.03805)  

**Abstract**: Large Reasoning Models (LRMs) demonstrate strong performance on complex tasks but often suffer from excessive verbosity, known as "overthinking." Existing solutions via reinforcement learning (RL) typically penalize generated tokens to promote conciseness. However, these methods encounter two challenges: responses with fewer tokens do not always correspond to fewer reasoning steps, and models may develop hacking behavior in later stages of training by discarding reasoning steps to minimize token usage. In this work, we introduce \textbf{Step Pruner (SP)}, an RL framework that steers LRMs toward more efficient reasoning by favoring compact reasoning steps. Our step-aware reward function prioritizes correctness while imposing penalties for redundant steps, and withholds rewards for incorrect responses to prevent the reinforcement of erroneous reasoning. Moreover, we propose a dynamic stopping mechanism: when the length of any output step exceeds the upper limit, we halt updates to prevent hacking behavior caused by merging steps. Extensive experiments across four reasoning benchmarks demonstrate that SP achieves state-of-the-art accuracy while significantly reducing response length. For instance, on AIME24, SP reduces token usage by \textbf{69.7\%}. 

---
# Mechanistic Interpretability of Socio-Political Frames in Language Models 

**Authors**: Hadi Asghari, Sami Nenno  

**Link**: [PDF](https://arxiv.org/pdf/2510.03799)  

**Abstract**: This paper explores the ability of large language models to generate and recognize deep cognitive frames, particularly in socio-political contexts. We demonstrate that LLMs are highly fluent in generating texts that evoke specific frames and can recognize these frames in zero-shot settings. Inspired by mechanistic interpretability research, we investigate the location of the `strict father' and `nurturing parent' frames within the model's hidden representation, identifying singular dimensions that correlate strongly with their presence. Our findings contribute to understanding how LLMs capture and express meaningful human concepts. 

---
# Rezwan: Leveraging Large Language Models for Comprehensive Hadith Text Processing: A 1.2M Corpus Development 

**Authors**: Majid Asgari-Bidhendi, Muhammad Amin Ghaseminia, Alireza Shahbazi, Sayyed Ali Hossayni, Najmeh Torabian, Behrouz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.03781)  

**Abstract**: This paper presents the development of Rezwan, a large-scale AI-assisted Hadith corpus comprising over 1.2M narrations, extracted and structured through a fully automated pipeline. Building on digital repositories such as Maktabat Ahl al-Bayt, the pipeline employs Large Language Models (LLMs) for segmentation, chain--text separation, validation, and multi-layer enrichment. Each narration is enhanced with machine translation into twelve languages, intelligent diacritization, abstractive summarization, thematic tagging, and cross-text semantic analysis. This multi-step process transforms raw text into a richly annotated research-ready infrastructure for digital humanities and Islamic studies. A rigorous evaluation was conducted on 1,213 randomly sampled narrations, assessed by six domain experts. Results show near-human accuracy in structured tasks such as chain--text separation (9.33/10) and summarization (9.33/10), while highlighting ongoing challenges in diacritization and semantic similarity detection. Comparative analysis against the manually curated Noor Corpus demonstrates the superiority of Najm in both scale and quality, with a mean overall score of 8.46/10 versus 3.66/10. Furthermore, cost analysis confirms the economic feasibility of the AI approach: tasks requiring over 229,000 hours of expert labor were completed within months at a fraction of the cost. The work introduces a new paradigm in religious text processing by showing how AI can augment human expertise, enabling large-scale, multilingual, and semantically enriched access to Islamic heritage. 

---
# Prompt Balance Matters: Understanding How Imbalanced Few-Shot Learning Affects Multilingual Sense Disambiguation in LLMs 

**Authors**: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough  

**Link**: [PDF](https://arxiv.org/pdf/2510.03762)  

**Abstract**: Recent advances in Large Language Models (LLMs) have significantly reshaped the landscape of Natural Language Processing (NLP). Among the various prompting techniques, few-shot prompting has gained considerable attention for its practicality and effectiveness. This study investigates how few-shot prompting strategies impact the Word Sense Disambiguation (WSD) task, particularly focusing on the biases introduced by imbalanced sample distributions. We use the GLOSSGPT prompting method, an advanced approach for English WSD, to test its effectiveness across five languages: English, German, Spanish, French, and Italian. Our results show that imbalanced few-shot examples can cause incorrect sense predictions in multilingual languages, but this issue does not appear in English. To assess model behavior, we evaluate both the GPT-4o and LLaMA-3.1-70B models and the results highlight the sensitivity of multilingual WSD to sample distribution in few-shot settings, emphasizing the need for balanced and representative prompting strategies. 

---
# Cross-Lingual Multi-Granularity Framework for Interpretable Parkinson's Disease Diagnosis from Speech 

**Authors**: Ilias Tougui, Mehdi Zakroum, Mounir Ghogho  

**Link**: [PDF](https://arxiv.org/pdf/2510.03758)  

**Abstract**: Parkinson's Disease (PD) affects over 10 million people worldwide, with speech impairments in up to 89% of patients. Current speech-based detection systems analyze entire utterances, potentially overlooking the diagnostic value of specific phonetic elements. We developed a granularity-aware approach for multilingual PD detection using an automated pipeline that extracts time-aligned phonemes, syllables, and words from recordings. Using Italian, Spanish, and English datasets, we implemented a bidirectional LSTM with multi-head attention to compare diagnostic performance across the different granularity levels. Phoneme-level analysis achieved superior performance with AUROC of 93.78% +- 2.34% and accuracy of 92.17% +- 2.43%. This demonstrates enhanced diagnostic capability for cross-linguistic PD detection. Importantly, attention analysis revealed that the most informative speech features align with those used in established clinical protocols: sustained vowels (/a/, /e/, /o/, /i/) at phoneme level, diadochokinetic syllables (/ta/, /pa/, /la/, /ka/) at syllable level, and /pataka/ sequences at word level. Source code will be available at this https URL. 

---
# TreePrompt: Leveraging Hierarchical Few-Shot Example Selection for Improved English-Persian and English-German Translation 

**Authors**: Ramtin Kakavand, Ebrahim Ansari  

**Link**: [PDF](https://arxiv.org/pdf/2510.03748)  

**Abstract**: Large Language Models (LLMs) have consistently demonstrated strong performance in machine translation, especially when guided by high-quality prompts. Few-shot prompting is an effective technique to improve translation quality; however, most existing example selection methods focus solely on query-to-example similarity and do not account for the quality of the examples. In this work, we propose TreePrompt, a novel example selection approach that learns LLM preferences to identify high-quality, contextually relevant examples within a tree-structured framework. To further explore the balance between similarity and quality, we combine TreePrompt with K-Nearest Neighbors (K-NN) and Adaptive Few-Shot Prompting (AFSP). Evaluations on two language pairs - English-Persian (MIZAN) and English-German (WMT19) - show that integrating TreePrompt with AFSP or Random selection leads to improved translation performance. 

---
# MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction 

**Authors**: Yue Huang, Yanyuan Chen, Dexuan Xu, Weihua Yue, Huamin Zhang, Meikang Qiu, Yu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03687)  

**Abstract**: Medical problem solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction: with merely 2,000 randomly sampled training examples and a light fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improve, reducing reliance on external supervision and extensive task-specific fine-tuning data. 

---
# Fine-Tuning Large Language Models with QLoRA for Offensive Language Detection in Roman Urdu-English Code-Mixed Text 

**Authors**: Nisar Hussain, Amna Qasim, Gull Mehak, Muhammad Zain, Momina Hafeez, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2510.03683)  

**Abstract**: The use of derogatory terms in languages that employ code mixing, such as Roman Urdu, presents challenges for Natural Language Processing systems due to unstated grammar, inconsistent spelling, and a scarcity of labeled data. In this work, we propose a QLoRA based fine tuning framework to improve offensive language detection in Roman Urdu-English text. We translated the Roman Urdu-English code mixed dataset into English using Google Translate to leverage English LLMs, while acknowledging that this translation reduces direct engagement with code mixing features. Our focus is on classification performance using English translated low resource inputs. We fine tuned several transformers and large language models, including Meta LLaMA 3 8B, Mistral 7B v0.1, LLaMA 2 7B, ModernBERT, and RoBERTa, with QLoRA for memory efficient adaptation. Models were trained and evaluated on a manually annotated Roman Urdu dataset for offensive vs non offensive content. Of all tested models, the highest F1 score of 91.45 was attained by Meta LLaMA 3 8B, followed by Mistral 7B at 89.66, surpassing traditional transformer baselines. These results demonstrate the efficacy of QLoRA in fine tuning high performing models for low resource environments such as code mixed offensive language detection, and confirm the potential of LLMs for this task. This work advances a scalable approach to Roman Urdu moderation and paves the way for future multilingual offensive detection systems based on LLMs. 

---
# UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG 

**Authors**: Xiangyu Peng, Cab Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03663)  

**Abstract**: Multimodal retrieval-augmented generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented, focusing on either text or images in isolation or on simplified multimodal setups that fail to capture document-centric multimodal use cases. In this paper, we introduce UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across eight domains. Our pipeline extracts and links evidence from text, tables, and figures, then generates 1,600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. To ensure reliability, 20% of QA pairs are validated by multiple annotators and expert adjudication. UniDoc-Bench supports apples-to-apples comparison across four paradigms: (1) text-only, (2) image-only, (3) multimodal text-image fusion, and (4) multimodal joint retrieval -- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. Our experiments show that multimodal text-image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding-based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines. 

---
# Towards Unsupervised Speech Recognition at the Syllable-Level 

**Authors**: Liming Wang, Junrui Ni, Kai-Wei Chang, Saurabhchand Bhati, David Harwath, Mark Hasegawa-Johnson, James R. Glass  

**Link**: [PDF](https://arxiv.org/pdf/2510.03639)  

**Abstract**: Training speech recognizers with unpaired speech and text -- known as unsupervised speech recognition (UASR) -- is a crucial step toward extending ASR to low-resource languages in the long-tail distribution and enabling multimodal learning from non-parallel data. However, existing approaches based on phones often rely on costly resources such as grapheme-to-phoneme converters (G2Ps) and struggle to generalize to languages with ambiguous phoneme boundaries due to training instability. In this paper, we address both challenges by introducing a syllable-level UASR framework based on masked language modeling, which avoids the need for G2P and the instability of GAN-based methods. Our approach achieves up to a 40\% relative reduction in character error rate (CER) on LibriSpeech and generalizes effectively to Mandarin, a language that has remained particularly difficult for prior methods. Code will be released upon acceptance. 

---
# Can an LLM Induce a Graph? Investigating Memory Drift and Context Length 

**Authors**: Raquib Bin Yousuf, Aadyant Khatri, Shengzhe Xu, Mandar Sharma, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03611)  

**Abstract**: Recently proposed evaluation benchmarks aim to characterize the effective context length and the forgetting tendencies of large language models (LLMs). However, these benchmarks often rely on simplistic 'needle in a haystack' retrieval or continuation tasks that may not accurately reflect the performance of these models in information-dense scenarios. Thus, rather than simple next token prediction, we argue for evaluating these models on more complex reasoning tasks that requires them to induce structured relational knowledge from the text - such as graphs from potentially noisy natural language content. While the input text can be viewed as generated in terms of a graph, its structure is not made explicit and connections must be induced from distributed textual cues, separated by long contexts and interspersed with irrelevant information. Our findings reveal that LLMs begin to exhibit memory drift and contextual forgetting at much shorter effective lengths when tasked with this form of relational reasoning, compared to what existing benchmarks suggest. With these findings, we offer recommendations for the optimal use of popular LLMs for complex reasoning tasks. We further show that even models specialized for reasoning, such as OpenAI o1, remain vulnerable to early memory drift in these settings. These results point to significant limitations in the models' ability to abstract structured knowledge from unstructured input and highlight the need for architectural adaptations to improve long-range reasoning. 

---
# Decoupling Task-Solving and Output Formatting in LLM Generation 

**Authors**: Haikang Deng, Po-Nien Kung, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.03595)  

**Abstract**: Large language models (LLMs) are increasingly adept at following instructions containing task descriptions to solve complex problems, such as mathematical reasoning and automatic evaluation (LLM-as-a-Judge). However, as prompts grow more complex, models often struggle to adhere to all instructions. This difficulty is especially common when instructive prompts intertwine reasoning directives -- specifying what the model should solve -- with rigid formatting requirements that dictate how the solution must be presented. The entanglement creates competing goals for the model, suggesting that more explicit separation of these two aspects could lead to improved performance. To this front, we introduce Deco-G, a decoding framework that explicitly decouples format adherence from task solving. Deco-G handles format compliance with a separate tractable probabilistic model (TPM), while prompts LLMs with only task instructions. At each decoding step, Deco-G combines next token probabilities from the LLM with the TPM calculated format compliance likelihood to form the output probability. To make this approach both practical and scalable for modern instruction-tuned LLMs, we introduce three key innovations: instruction-aware distillation, a flexible trie-building algorithm, and HMM state pruning for computational efficiency. We demonstrate the effectiveness of Deco-G across a wide range of tasks with diverse format requirements, including mathematical reasoning, LLM-as-a-judge, and event argument extraction. Overall, our approach yields 1.0% to 6.0% relative gain over regular prompting practice with guaranteed format compliance. 

---
# LLM, Reporting In! Medical Information Extraction Across Prompting, Fine-tuning and Post-correction 

**Authors**: Ikram Belmadani, Parisa Nazari Hashemi, Thomas Sebbag, Benoit Favre, Guillaume Fortier, Solen Quiniou, Emmanuel Morin, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2510.03577)  

**Abstract**: This work presents our participation in the EvalLLM 2025 challenge on biomedical Named Entity Recognition (NER) and health event extraction in French (few-shot setting). For NER, we propose three approaches combining large language models (LLMs), annotation guidelines, synthetic data, and post-processing: (1) in-context learning (ICL) with GPT-4.1, incorporating automatic selection of 10 examples and a summary of the annotation guidelines into the prompt, (2) the universal NER system GLiNER, fine-tuned on a synthetic corpus and then verified by an LLM in post-processing, and (3) the open LLM LLaMA-3.1-8B-Instruct, fine-tuned on the same synthetic corpus. Event extraction uses the same ICL strategy with GPT-4.1, reusing the guideline summary in the prompt. Results show GPT-4.1 leads with a macro-F1 of 61.53% for NER and 15.02% for event extraction, highlighting the importance of well-crafted prompting to maximize performance in very low-resource scenarios. 

---
# Reactive Transformer (RxT) -- Stateful Real-Time Processing for Event-Driven Reactive Language Models 

**Authors**: Adam Filipek  

**Link**: [PDF](https://arxiv.org/pdf/2510.03561)  

**Abstract**: The Transformer architecture has become the de facto standard for Large Language Models (LLMs), demonstrating remarkable capabilities in language understanding and generation. However, its application in conversational AI is fundamentally constrained by its stateless nature and the quadratic computational complexity ($O(L^2)$) with respect to sequence length $L$. Current models emulate memory by reprocessing an ever-expanding conversation history with each turn, leading to prohibitive costs and latency in long dialogues. This paper introduces the Reactive Transformer (RxT), a novel architecture designed to overcome these limitations by shifting from a data-driven to an event-driven paradigm. RxT processes each conversational turn as a discrete event in real-time, maintaining context in an integrated, fixed-size Short-Term Memory (STM) system. The architecture features a distinct operational cycle where a generator-decoder produces a response based on the current query and the previous memory state, after which a memory-encoder and a dedicated Memory Attention network asynchronously update the STM with a representation of the complete interaction. This design fundamentally alters the scaling dynamics, reducing the total user-facing cost of a conversation from quadratic ($O(N^2 \cdot T)$) to linear ($O(N \cdot T)$) with respect to the number of interactions $N$. By decoupling response generation from memory updates, RxT achieves low latency, enabling truly real-time, stateful, and economically viable long-form conversations. We validated our architecture with a series of proof-of-concept experiments on synthetic data, demonstrating superior performance and constant-time inference latency compared to a baseline stateless model of comparable size. 

---
# CCD-Bench: Probing Cultural Conflict in Large Language Model Decision-Making 

**Authors**: Hasibur Rahman, Hanan Salam  

**Link**: [PDF](https://arxiv.org/pdf/2510.03553)  

**Abstract**: Although large language models (LLMs) are increasingly implicated in interpersonal and societal decision-making, their ability to navigate explicit conflicts between legitimately different cultural value systems remains largely unexamined. Existing benchmarks predominantly target cultural knowledge (CulturalBench), value prediction (WorldValuesBench), or single-axis bias diagnostics (CDEval); none evaluate how LLMs adjudicate when multiple culturally grounded values directly clash. We address this gap with CCD-Bench, a benchmark that assesses LLM decision-making under cross-cultural value conflict. CCD-Bench comprises 2,182 open-ended dilemmas spanning seven domains, each paired with ten anonymized response options corresponding to the ten GLOBE cultural clusters. These dilemmas are presented using a stratified Latin square to mitigate ordering effects. We evaluate 17 non-reasoning LLMs. Models disproportionately prefer Nordic Europe (mean 20.2 percent) and Germanic Europe (12.4 percent), while options for Eastern Europe and the Middle East and North Africa are underrepresented (5.6 to 5.8 percent). Although 87.9 percent of rationales reference multiple GLOBE dimensions, this pluralism is superficial: models recombine Future Orientation and Performance Orientation, and rarely ground choices in Assertiveness or Gender Egalitarianism (both under 3 percent). Ordering effects are negligible (Cramer's V less than 0.10), and symmetrized KL divergence shows clustering by developer lineage rather than geography. These patterns suggest that current alignment pipelines promote a consensus-oriented worldview that underserves scenarios demanding power negotiation, rights-based reasoning, or gender-aware analysis. CCD-Bench shifts evaluation beyond isolated bias detection toward pluralistic decision making and highlights the need for alignment strategies that substantively engage diverse worldviews. 

---
# What is a protest anyway? Codebook conceptualization is still a first-order concern in LLM-era classification 

**Authors**: Andrew Halterman, Katherine A. Keith  

**Link**: [PDF](https://arxiv.org/pdf/2510.03541)  

**Abstract**: Generative large language models (LLMs) are now used extensively for text classification in computational social science (CSS). In this work, focus on the steps before and after LLM prompting -- conceptualization of concepts to be classified and using LLM predictions in downstream statistical inference -- which we argue have been overlooked in much of LLM-era CSS. We claim LLMs can tempt analysts to skip the conceptualization step, creating conceptualization errors that bias downstream estimates. Using simulations, we show that this conceptualization-induced bias cannot be corrected for solely by increasing LLM accuracy or post-hoc bias correction methods. We conclude by reminding CSS analysts that conceptualization is still a first-order concern in the LLM-era and provide concrete advice on how to pursue low-cost, unbiased, low-variance downstream estimates. 

---
# TriMediQ: A Triplet-Structured Approach for Interactive Medical Question Answering 

**Authors**: Zhaohan Meng, Zaiqiao Meng, Siwei Liu, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03536)  

**Abstract**: Large Language Models (LLMs) perform strongly in static and single-turn medical Question Answer (QA) benchmarks, yet such settings diverge from the iterative information gathering process required in practical clinical consultations. The MEDIQ framework addresses this mismatch by recasting the diagnosis as an interactive dialogue between a patient and an expert system, but the reliability of LLMs drops dramatically when forced to reason with dialogue logs, where clinical facts appear in sentences without clear links. To bridge this gap, we introduce TriMediQ, a triplet-structured approach that summarises patient responses into triplets and integrates them into a Knowledge Graph (KG), enabling multi-hop reasoning. We introduce a frozen triplet generator that extracts clinically relevant triplets, using prompts designed to ensure factual consistency. In parallel, a trainable projection module, comprising a graph encoder and a projector, captures relational information from the KG to enhance expert reasoning. TriMediQ operates in two steps: (i) the projection module fine-tuning with all LLM weights frozen; and (ii) using the fine-tuned module to guide multi-hop reasoning during inference. We evaluate TriMediQ on two interactive QA benchmarks, showing that it achieves up to 10.4\% improvement in accuracy over five baselines on the iMedQA dataset. These results demonstrate that converting patient responses into structured triplet-based graphs enables more accurate clinical reasoning in multi-turn settings, providing a solution for the deployment of LLM-based medical assistants. 

---
# Fine-Tuning on Noisy Instructions: Effects on Generalization and Performance 

**Authors**: Ahmed Alajrami, Xingwei Tan, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2510.03528)  

**Abstract**: Instruction-tuning plays a vital role in enhancing the task-solving abilities of large language models (LLMs), improving their usability in generating helpful responses on various tasks. However, previous work has demonstrated that they are sensitive to minor variations in instruction phrasing. In this paper, we explore whether introducing perturbations in instruction-tuning data can enhance LLMs' resistance against noisy instructions. We focus on how instruction-tuning with perturbations, such as removing stop words or shuffling words, affects LLMs' performance on the original and perturbed versions of widely-used benchmarks (MMLU, BBH, GSM8K). We further assess learning dynamics and potential shifts in model behavior. Surprisingly, our results suggest that instruction-tuning on perturbed instructions can, in some cases, improve downstream performance. These findings highlight the importance of including perturbed instructions in instruction-tuning, which can make LLMs more resilient to noisy user inputs. 

---
# Sample, Align, Synthesize: Graph-Based Response Synthesis with ConGrs 

**Authors**: Sayan Ghosh, Shahzaib Saqib Warraich, Dhruv Tarsadiya, Gregory Yauney, Swabha Swayamdipta  

**Link**: [PDF](https://arxiv.org/pdf/2510.03527)  

**Abstract**: Language models can be sampled multiple times to access the distribution underlying their responses, but existing methods cannot efficiently synthesize rich epistemic signals across different long-form responses. We introduce Consensus Graphs (ConGrs), a flexible DAG-based data structure that represents shared information, as well as semantic variation in a set of sampled LM responses to the same prompt. We construct ConGrs using a light-weight lexical sequence alignment algorithm from bioinformatics, supplemented by the targeted usage of a secondary LM judge. Further, we design task-dependent decoding methods to synthesize a single, final response from our ConGr data structure. Our experiments show that synthesizing responses from ConGrs improves factual precision on two biography generation tasks by up to 31% over an average response and reduces reliance on LM judges by more than 80% compared to other methods. We also use ConGrs for three refusal-based tasks requiring abstention on unanswerable queries and find that abstention rate is increased by up to 56%. We apply our approach to the MATH and AIME reasoning tasks and find an improvement over self-verification and majority vote baselines by up to 6 points of accuracy. We show that ConGrs provide a flexible method for capturing variation in LM responses and using the epistemic signals provided by response variation to synthesize more effective responses. 

---
# Identifying Financial Risk Information Using RAG with a Contrastive Insight 

**Authors**: Ali Elahi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03521)  

**Abstract**: In specialized domains, humans often compare new problems against similar examples, highlight nuances, and draw conclusions instead of analyzing information in isolation. When applying reasoning in specialized contexts with LLMs on top of a RAG, the pipeline can capture contextually relevant information, but it is not designed to retrieve comparable cases or related problems.
While RAG is effective at extracting factual information, its outputs in specialized reasoning tasks often remain generic, reflecting broad facts rather than context-specific insights. In finance, it results in generic risks that are true for the majority of companies. To address this limitation, we propose a peer-aware comparative inference layer on top of RAG.
Our contrastive approach outperforms baseline RAG in text generation metrics such as ROUGE and BERTScore in comparison with human-generated equity research and risk. 

---
# TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning 

**Authors**: Fangxu Yu, Hongyu Zhao, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03519)  

**Abstract**: Time series reasoning is crucial to decision-making in diverse domains, including finance, energy usage, traffic, weather, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through large language models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction finetuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Extensive experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of prevailing LLMs, Vision Language Models (VLMs), and Time Series LLMs, but also achieves this with remarkable data efficiency, e.g., using less than half the training data. 

---
# ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection 

**Authors**: Ali Khairallah, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2510.03502)  

**Abstract**: We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats. 

---
# SEER: The Span-based Emotion Evidence Retrieval Benchmark 

**Authors**: Aneesha Sampath, Oya Aran, Emily Mower Provost  

**Link**: [PDF](https://arxiv.org/pdf/2510.03490)  

**Abstract**: We introduce the SEER (Span-based Emotion Evidence Retrieval) Benchmark to test Large Language Models' (LLMs) ability to identify the specific spans of text that express emotion. Unlike traditional emotion recognition tasks that assign a single label to an entire sentence, SEER targets the underexplored task of emotion evidence detection: pinpointing which exact phrases convey emotion. This span-level approach is crucial for applications like empathetic dialogue and clinical support, which need to know how emotion is expressed, not just what the emotion is. SEER includes two tasks: identifying emotion evidence within a single sentence, and identifying evidence across a short passage of five consecutive sentences. It contains new annotations for both emotion and emotion evidence on 1200 real-world sentences. We evaluate 14 open-source LLMs and find that, while some models approach average human performance on single-sentence inputs, their accuracy degrades in longer passages. Our error analysis reveals key failure modes, including overreliance on emotion keywords and false positives in neutral text. 

---
# Searching for the Most Human-like Emergent Language 

**Authors**: Brendon Boldt, David Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03467)  

**Abstract**: In this paper, we design a signalling game-based emergent communication environment to generate state-of-the-art emergent languages in terms of similarity to human language. This is done with hyperparameter optimization, using XferBench as the objective function. XferBench quantifies the statistical similarity of emergent language to human language by measuring its suitability for deep transfer learning to human language. Additionally, we demonstrate the predictive power of entropy on the transfer learning performance of emergent language as well as corroborate previous results on the entropy-minimization properties of emergent communication systems. Finally, we report generalizations regarding what hyperparameters produce more realistic emergent languages, that is, ones which transfer better to human language. 

---
# Omni-Embed-Nemotron: A Unified Multimodal Retrieval Model for Text, Image, Audio, and Video 

**Authors**: Mengyao Xu, Wenfei Zhou, Yauhen Babakhin, Gabriel Moreira, Ronay Ak, Radek Osmulski, Bo Liu, Even Oldridge, Benedikt Schifferer  

**Link**: [PDF](https://arxiv.org/pdf/2510.03458)  

**Abstract**: We present Omni-Embed-Nemotron, a unified multimodal retrieval embedding model developed to handle the increasing complexity of real-world information needs. While Retrieval-Augmented Generation (RAG) has significantly advanced language models by incorporating external knowledge, existing text-based retrievers rely on clean, structured input and struggle with the visually and semantically rich content found in real-world documents such as PDFs, slides, or videos. Recent work such as ColPali has shown that preserving document layout using image-based representations can improve retrieval quality. Building on this, and inspired by the capabilities of recent multimodal models such as Qwen2.5-Omni, we extend retrieval beyond text and images to also support audio and video modalities. Omni-Embed-Nemotron enables both cross-modal (e.g., text - video) and joint-modal (e.g., text - video+audio) retrieval using a single model. We describe the architecture, training setup, and evaluation results of Omni-Embed-Nemotron, and demonstrate its effectiveness in text, image, and video retrieval. 

---
# Morpheme Induction for Emergent Language 

**Authors**: Brendon Boldt, David Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03439)  

**Abstract**: We introduce CSAR, an algorithm for inducing morphemes from emergent language corpora of parallel utterances and meanings. It is a greedy algorithm that (1) weights morphemes based on mutual information between forms and meanings, (2) selects the highest-weighted pair, (3) removes it from the corpus, and (4) repeats the process to induce further morphemes (i.e., Count, Select, Ablate, Repeat). The effectiveness of CSAR is first validated on procedurally generated datasets and compared against baselines for related tasks. Second, we validate CSAR's performance on human language data to show that the algorithm makes reasonable predictions in adjacent domains. Finally, we analyze a handful of emergent languages, quantifying linguistic characteristics like degree of synonymy and polysemy. 

---
# Implicit Values Embedded in How Humans and LLMs Complete Subjective Everyday Tasks 

**Authors**: Arjun Arunasalam, Madison Pickering, Z. Berkay Celik, Blase Ur  

**Link**: [PDF](https://arxiv.org/pdf/2510.03384)  

**Abstract**: Large language models (LLMs) can underpin AI assistants that help users with everyday tasks, such as by making recommendations or performing basic computation. Despite AI assistants' promise, little is known about the implicit values these assistants display while completing subjective everyday tasks. Humans may consider values like environmentalism, charity, and diversity. To what extent do LLMs exhibit these values in completing everyday tasks? How do they compare with humans? We answer these questions by auditing how six popular LLMs complete 30 everyday tasks, comparing LLMs to each other and to 100 human crowdworkers from the US. We find LLMs often do not align with humans, nor with other LLMs, in the implicit values exhibited. 

---
# Graph-S3: Enhancing Agentic textual Graph Retrieval with Synthetic Stepwise Supervision 

**Authors**: Ge Chang, Jinbo Su, Jiacheng Liu, Pengfei Yang, Yuhao Shang, Huiwen Zheng, Hongli Ma, Yan Liang, Yuanchun Li, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03323)  

**Abstract**: A significant portion of real-world data is inherently represented as textual graphs, and integrating these graphs into large language models (LLMs) is promising to enable complex graph-based question answering. However, a key challenge in LLM-based textual graph QA systems lies in graph retrieval, i.e., how to retrieve relevant content from large graphs that is sufficiently informative while remaining compact for the LLM context. Existing retrievers suffer from poor performance since they either rely on shallow embedding similarity or employ interactive retrieving policies that demand excessive data labeling and training cost. To address these issues, we present Graph-$S^3$, an agentic textual graph reasoning framework that employs an LLM-based retriever trained with synthetic stepwise supervision. Instead of rewarding the agent based on the final answers, which may lead to sparse and unstable training signals, we propose to closely evaluate each step of the retriever based on offline-extracted golden subgraphs. Our main techniques include a data synthesis pipeline to extract the golden subgraphs for reward generation and a two-stage training scheme to learn the interactive graph exploration policy based on the synthesized rewards. Based on extensive experiments on three common datasets in comparison with seven strong baselines, our approach achieves an average improvement of 8.1\% in accuracy and 9.7\% in F$_1$ score. The advantage is even higher in more complicated multi-hop reasoning tasks. Our code will be open-sourced. 

---
# Decomposing Attention To Find Context-Sensitive Neurons 

**Authors**: Alex Gibson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03315)  

**Abstract**: We study transformer language models, analyzing attention heads whose attention patterns are spread out, and whose attention scores depend weakly on content. We argue that the softmax denominators of these heads are stable when the underlying token distribution is fixed. By sampling softmax denominators from a "calibration text", we can combine together the outputs of multiple such stable heads in the first layer of GPT2-Small, approximating their combined output by a linear summary of the surrounding text. This approximation enables a procedure where from the weights alone - and a single calibration text - we can uncover hundreds of first layer neurons that respond to high-level contextual properties of the surrounding text, including neurons that didn't activate on the calibration text. 

---
# Paper2Video: Automatic Video Generation from Scientific Papers 

**Authors**: Zeyu Zhu, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.05096)  

**Abstract**: Academic presentation videos have become an essential medium for research communication, yet producing them remains highly labor-intensive, often requiring hours of slide design, recording, and editing for a short 2 to 10 minutes video. Unlike natural video, presentation video generation involves distinctive challenges: inputs from research papers, dense multi-modal information (text, figures, tables), and the need to coordinate multiple aligned channels such as slides, subtitles, speech, and human talker. To address these challenges, we introduce PaperTalker, the first benchmark of 101 research papers paired with author-created presentation videos, slides, and speaker metadata. We further design four tailored evaluation metrics--Meta Similarity, PresentArena, PresentQuiz, and IP Memory--to measure how videos convey the paper's information to the audience. Building on this foundation, we propose PaperTalker, the first multi-agent framework for academic presentation video generation. It integrates slide generation with effective layout refinement by a novel effective tree search visual choice, cursor grounding, subtitling, speech synthesis, and talking-head rendering, while parallelizing slide-wise generation for efficiency. Experiments on Paper2Video demonstrate that the presentation videos produced by our approach are more faithful and informative than existing baselines, establishing a practical step toward automated and ready-to-use academic video generation. Our dataset, agent, and code are available at this https URL. 

---
# From Noisy Traces to Stable Gradients: Bias-Variance Optimized Preference Optimization for Aligning Large Reasoning Models 

**Authors**: Mingkang Zhu, Xi Chen, Bei Yu, Hengshuang Zhao, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.05095)  

**Abstract**: Large reasoning models (LRMs) generate intermediate reasoning traces before producing final answers, yielding strong gains on multi-step and mathematical tasks. Yet aligning LRMs with human preferences, a crucial prerequisite for model deployment, remains underexplored. The statistically correct objective for preference alignment requires marginalizing over reasoning traces, but this computation is intractable in practice. A common workaround optimizes a single sampled trajectory, which introduces substantial gradient variance from stochastic trace sampling. To address this challenge, we frame preference optimization for LRMs through the lens of the bias--variance trade-off and propose Bias--Variance Optimized Preference Optimization (BVPO), a simple, drop-in method that mixes two gradient estimators: a high-variance trace-based estimator and a low-variance empty-trace estimator obtained by disabling reasoning trace generation. Our theory shows that BVPO strictly reduces trace-induced variance for any nontrivial mixture, provides a closed-form choice of the mixing weight that minimizes mean-squared error relative to the true marginal gradient, and under standard smoothness and step-size conditions, tightens classical convergence bounds for stochastic gradient descent. Empirically, BVPO improves alignment over the best baseline by up to 7.8 points on AlpacaEval~2 and 6.8 points on Arena-Hard. Despite being trained only on general conversational data, BVPO also boosts reasoning performance for base models by up to 4.0 points on the average of six math reasoning benchmarks. These results identify variance from trace sampling as a key bottleneck and demonstrate that directly optimizing the bias--variance trade-off yields more stable training and stronger overall performance. 

---
# Learning to Interpret Weight Differences in Language Models 

**Authors**: Avichal Goel, Yoon Kim, Nir Shavit, Tony T. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05092)  

**Abstract**: Finetuning (pretrained) language models is a standard approach for updating their internal parametric knowledge and specializing them to new tasks and domains. However, the corresponding model weight changes ("weight diffs") are not generally interpretable. While inspecting the finetuning dataset can give a sense of how the model might have changed, these datasets are often not publicly available or are too large to work with directly. Towards the goal of comprehensively understanding weight diffs in natural language, we introduce Diff Interpretation Tuning (DIT), a method that trains models to describe their own finetuning-induced modifications. Our approach uses synthetic, labeled weight diffs to train a DIT adapter, which can be applied to a compatible finetuned model to make it describe how it has changed. We demonstrate in two proof-of-concept settings (reporting hidden behaviors and summarizing finetuned knowledge) that our method enables models to describe their finetuning-induced modifications using accurate natural language descriptions. 

---
# Proactive defense against LLM Jailbreak 

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05052)  

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks. 

---
# Large Language Models Achieve Gold Medal Performance at International Astronomy & Astrophysics Olympiad 

**Authors**: Lucas Carrit Delgado Pinheiro, Ziru Chen, Bruno Caixeta Piazza, Ness Shroff, Yingbin Liang, Yuan-Sen Ting, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.05016)  

**Abstract**: While task-specific demonstrations show early success in applying large language models (LLMs) to automate some astronomical research tasks, they only provide incomplete views of all necessary capabilities in solving astronomy problems, calling for more thorough understanding of LLMs' strengths and limitations. So far, existing benchmarks and evaluations focus on simple question-answering that primarily tests astronomical knowledge and fails to evaluate the complex reasoning required for real-world research in the discipline. Here, we address this gap by systematically benchmarking five state-of-the-art LLMs on the International Olympiad on Astronomy and Astrophysics (IOAA) exams, which are designed to examine deep conceptual understanding, multi-step derivations, and multimodal analysis. With average scores of 85.6% and 84.2%, Gemini 2.5 Pro and GPT-5 (the two top-performing models) not only achieve gold medal level performance but also rank in the top two among ~200-300 participants in all four IOAA theory exams evaluated (2022-2025). In comparison, results on the data analysis exams show more divergence. GPT-5 still excels in the exams with an 88.5% average score, ranking top 10 among the participants in the four most recent IOAAs, while other models' performances drop to 48-76%. Furthermore, our in-depth error analysis underscores conceptual reasoning, geometric reasoning, and spatial visualization (52-79% accuracy) as consistent weaknesses among all LLMs. Hence, although LLMs approach peak human performance in theory exams, critical gaps must be addressed before they can serve as autonomous research agents in astronomy. 

---
# Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training 

**Authors**: Wei Xiong, Chenlu Ye, Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, Jiang Bian, Nan Jiang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04996)  

**Abstract**: Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at this https URL. 

---
# LLM-Hanabi: Evaluating Multi-Agent Gameplays with Theory-of-Mind and Rationale Inference in Imperfect Information Collaboration Game 

**Authors**: Fangzhou Liang, Tianshi Zheng, Chunkit Chan, Yauwai Yim, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.04980)  

**Abstract**: Effective multi-agent collaboration requires agents to infer the rationale behind others' actions, a capability rooted in Theory-of-Mind (ToM). While recent Large Language Models (LLMs) excel at logical inference, their ability to infer rationale in dynamic, collaborative settings remains under-explored. This study introduces LLM-Hanabi, a novel benchmark that uses the cooperative game Hanabi to evaluate the rationale inference and ToM of LLMs. Our framework features an automated evaluation system that measures both game performance and ToM proficiency. Across a range of models, we find a significant positive correlation between ToM and in-game success. Notably, first-order ToM (interpreting others' intent) correlates more strongly with performance than second-order ToM (predicting others' interpretations). These findings highlight that for effective AI collaboration, the ability to accurately interpret a partner's rationale is more critical than higher-order reasoning. We conclude that prioritizing first-order ToM is a promising direction for enhancing the collaborative capabilities of future models. 

---
# On Structured State-Space Duality 

**Authors**: Jerry Yao-Chieh Hu, Xiwen Zhang, Weimin Wu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04944)  

**Abstract**: Structured State-Space Duality (SSD) [Dao & Gu, ICML 2024] is an equivalence between a simple Structured State-Space Model (SSM) and a masked attention mechanism. In particular, a state-space model with a scalar-times-identity state matrix is equivalent to a masked self-attention with a $1$-semiseparable causal mask. Consequently, the same sequence transformation (model) has two algorithmic realizations: as a linear-time $O(T)$ recurrence or as a quadratic-time $O(T^2)$ attention. In this note, we formalize and generalize this duality: (i) we extend SSD from the scalar-identity case to general diagonal SSMs (diagonal state matrices); (ii) we show that these diagonal SSMs match the scalar case's training complexity lower bounds while supporting richer dynamics; (iii) we establish a necessary and sufficient condition under which an SSM is equivalent to $1$-semiseparable masked attention; and (iv) we show that such duality fails to extend to standard softmax attention due to rank explosion. Together, these results tighten bridge between recurrent SSMs and Transformers, and widen the design space for expressive yet efficient sequence models. 

---
# ONNX-Net: Towards Universal Representations and Instant Performance Prediction for Neural Architectures 

**Authors**: Shiwen Qin, Alexander Auras, Shay B. Cohen, Elliot J. Crowley, Michael Moeller, Linus Ericsson, Jovita Lukasik  

**Link**: [PDF](https://arxiv.org/pdf/2510.04938)  

**Abstract**: Neural architecture search (NAS) automates the design process of high-performing architectures, but remains bottlenecked by expensive performance evaluation. Most existing studies that achieve faster evaluation are mostly tied to cell-based search spaces and graph encodings tailored to those individual search spaces, limiting their flexibility and scalability when applied to more expressive search spaces. In this work, we aim to close the gap of individual search space restrictions and search space dependent network representations. We present ONNX-Bench, a benchmark consisting of a collection of neural networks in a unified format based on ONNX files. ONNX-Bench includes all open-source NAS-bench-based neural networks, resulting in a total size of more than 600k {architecture, accuracy} pairs. This benchmark allows creating a shared neural network representation, ONNX-Net, able to represent any neural architecture using natural language descriptions acting as an input to a performance predictor. This text-based encoding can accommodate arbitrary layer types, operation parameters, and heterogeneous topologies, enabling a single surrogate to generalise across all neural architectures rather than being confined to cell-based search spaces. Experiments show strong zero-shot performance across disparate search spaces using only a small amount of pretraining samples, enabling the unprecedented ability to evaluate any neural network architecture instantly. 

---
# MARS: Optimizing Dual-System Deep Research via Multi-Agent Reinforcement Learning 

**Authors**: Guoxin Chen, Zile Qiao, Wenqing Wang, Donglei Yu, Xuanzhong Chen, Hao Sun, Minpeng Liao, Kai Fan, Yong Jiang, Penguin Xie, Wayne Xin Zhao, Ruihua Song, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04935)  

**Abstract**: Large Reasoning Models (LRMs) often exhibit a tendency for overanalysis in simple tasks, where the models excessively utilize System 2-type, deliberate reasoning, leading to inefficient token generation. Furthermore, these models face challenges in adapting their reasoning capabilities to rapidly changing environments due to the static nature of their pretraining data. To address these issues, advancing Large Language Models (LLMs) for complex reasoning tasks requires innovative approaches that bridge intuitive and deliberate cognitive processes, akin to human cognition's dual-system dynamic. This paper introduces a Multi-Agent System for Deep ReSearch (MARS) enabling seamless integration of System 1's fast, intuitive thinking with System 2's deliberate reasoning within LLMs. MARS strategically integrates multiple external tools, such as Google Search, Google Scholar, and Python Interpreter, to access up-to-date information and execute complex computations, while creating a specialized division of labor where System 1 efficiently processes and summarizes high-volume external information, providing distilled insights that expand System 2's reasoning context without overwhelming its capacity. Furthermore, we propose a multi-agent reinforcement learning framework extending Group Relative Policy Optimization to simultaneously optimize both systems with multi-turn tool interactions, bin-packing optimization, and sample balancing strategies that enhance collaborative efficiency. Extensive experiments demonstrate MARS achieves substantial improvements of 3.86% on the challenging Humanity's Last Exam (HLE) benchmark and an average gain of 8.9% across 7 knowledge-intensive tasks, validating the effectiveness of our dual-system paradigm for complex reasoning in dynamic information environments. 

---
# Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches 

**Authors**: Yicheng Tao, Yao Qin, Yepang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04905)  

**Abstract**: Recent advancements in large language models (LLMs) have substantially improved automated code generation. While function-level and file-level generation have achieved promising results, real-world software development typically requires reasoning across entire repositories. This gives rise to the challenging task of Repository-Level Code Generation (RLCG), where models must capture long-range dependencies, ensure global semantic consistency, and generate coherent code spanning multiple files or modules. To address these challenges, Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm that integrates external retrieval mechanisms with LLMs, enhancing context-awareness and scalability. In this survey, we provide a comprehensive review of research on Retrieval-Augmented Code Generation (RACG), with an emphasis on repository-level approaches. We categorize existing work along several dimensions, including generation strategies, retrieval modalities, model architectures, training paradigms, and evaluation protocols. Furthermore, we summarize widely used datasets and benchmarks, analyze current limitations, and outline key challenges and opportunities for future research. Our goal is to establish a unified analytical framework for understanding this rapidly evolving field and to inspire continued progress in AI-powered software engineering. 

---
# Visual Representations inside the Language Model 

**Authors**: Benlin Liu, Amita Kamath, Madeleine Grunde-McLaughlin, Winson Han, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2510.04819)  

**Abstract**: Despite interpretability work analyzing VIT encoders and transformer activations, we don't yet understand why Multimodal Language Models (MLMs) struggle on perception-heavy tasks. We offer an under-studied perspective by examining how popular MLMs (LLaVA-OneVision, Qwen2.5-VL, and Llama-3-LLaVA-NeXT) process their visual key-value tokens. We first study the flow of visual information through the language model, finding that image value tokens encode sufficient information to perform several perception-heavy tasks zero-shot: segmentation, semantic correspondence, temporal correspondence, and referring expression detection. We find that while the language model does augment the visual information received from the projection of input visual encodings-which we reveal correlates with overall MLM perception capability-it contains less visual information on several tasks than the equivalent visual encoder (SigLIP) that has not undergone MLM finetuning. Further, we find that the visual information corresponding to input-agnostic image key tokens in later layers of language models contains artifacts which reduce perception capability of the overall MLM. Next, we discuss controlling visual information in the language model, showing that adding a text prefix to the image input improves perception capabilities of visual representations. Finally, we reveal that if language models were able to better control their visual information, their perception would significantly improve; e.g., in 33.3% of Art Style questions in the BLINK benchmark, perception information present in the language model is not surfaced to the output! Our findings reveal insights into the role of key-value tokens in multimodal systems, paving the way for deeper mechanistic interpretability of MLMs and suggesting new directions for training their visual encoder and language model components. 

---
# Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba 

**Authors**: Baher Mohammad, Magauiya Zhussip, Stamatios Lefkimmiatis  

**Link**: [PDF](https://arxiv.org/pdf/2510.04738)  

**Abstract**: We introduce MAVE (Mamba with Cross-Attention for Voice Editing and Synthesis), a novel autoregressive architecture for text-conditioned voice editing and high-fidelity text-to-speech (TTS) synthesis, built on a cross-attentive Mamba backbone. MAVE achieves state-of-the-art performance in speech editing and very competitive results in zero-shot TTS, while not being explicitly trained on the latter task, outperforming leading autoregressive and diffusion models on diverse, real-world audio. By integrating Mamba for efficient audio sequence modeling with cross-attention for precise text-acoustic alignment, MAVE enables context-aware voice editing with exceptional naturalness and speaker consistency. In pairwise human evaluations on a random 40-sample subset of the RealEdit benchmark (400 judgments), 57.2% of listeners rated MAVE - edited speech as perceptually equal to the original, while 24.8% prefered the original and 18.0% MAVE - demonstrating that in the majority of cases edits are indistinguishable from the source. MAVE compares favorably with VoiceCraft and FluentSpeech both on pairwise comparisons and standalone mean opinion score (MOS) evaluations. For zero-shot TTS, MAVE exceeds VoiceCraft in both speaker similarity and naturalness, without requiring multiple inference runs or post-processing. Remarkably, these quality gains come with a significantly lower memory cost and approximately the same latency: MAVE requires ~6x less memory than VoiceCraft during inference on utterances from the RealEdit database (mean duration: 6.21s, A100, FP16, batch size 1). Our results demonstrate that MAVE establishes a new standard for flexible, high-fidelity voice editing and synthesis through the synergistic integration of structured state-space modeling and cross-modal attention. 

---
# BrokenMath: A Benchmark for Sycophancy in Theorem Proving with LLMs 

**Authors**: Ivo Petrov, Jasper Dekoninck, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2510.04721)  

**Abstract**: Large language models (LLMs) have recently shown strong performance on mathematical benchmarks. At the same time, they are prone to hallucination and sycophancy, often providing convincing but flawed proofs for incorrect mathematical statements provided by users. This significantly limits the applicability of LLMs in theorem proving, as verification of these flawed proofs must be done manually by expert mathematicians. However, existing benchmarks that measure sycophancy in mathematics are limited: they focus solely on final-answer problems, rely on very simple and often contaminated datasets, and construct benchmark samples using synthetic modifications that create ill-posed questions rather than well-posed questions that are demonstrably false. To address these issues, we introduce BrokenMath, the first benchmark for evaluating sycophantic behavior in LLMs within the context of natural language theorem proving. BrokenMath is built from advanced 2025 competition problems, which are perturbed with an LLM to produce false statements and subsequently refined through expert review. Using an LLM-as-a-judge framework, we evaluate state-of-the-art LLMs and agentic systems and find that sycophancy is widespread, with the best model, GPT-5, producing sycophantic answers 29% of the time. We further investigate several mitigation strategies, including test-time interventions and supervised fine-tuning on curated sycophantic examples. These approaches substantially reduce, but do not eliminate, sycophantic behavior. 

---
# AtomWorld: A Benchmark for Evaluating Spatial Reasoning in Large Language Models on Crystalline Materials 

**Authors**: Taoyuze Lv, Alexander Chen, Fengyu Xie, Chu Wu, Jeffrey Meng, Dongzhan Zhou, Bram Hoex, Zhicheng Zhong, Tong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.04704)  

**Abstract**: Large Language Models (LLMs) excel at textual reasoning and are beginning to develop spatial understanding, prompting the question of whether these abilities can be combined for complex, domain-specific tasks. This question is essential in fields like materials science, where deep understanding of 3D atomic structures is fundamental. While initial studies have successfully applied LLMs to tasks involving pure crystal generation or coordinate understandings, a standardized benchmark to systematically evaluate their core reasoning abilities across diverse atomic structures has been notably absent. To address this gap, we introduce the AtomWorld benchmark to evaluate LLMs on tasks based in Crystallographic Information Files (CIFs), a standard structure representation format. These tasks, including structural editing, CIF perception, and property-guided modeling, reveal a critical limitation: current models, despite establishing promising baselines, consistently fail in structural understanding and spatial reasoning. Our experiments show that these models make frequent errors on structure modification tasks, and even in the basic CIF format understandings, potentially leading to cumulative errors in subsequent analysis and materials insights. By defining these standardized tasks, AtomWorld lays the ground for advancing LLMs toward robust atomic-scale modeling, crucial for accelerating materials research and automating scientific workflows. 

---
# Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models 

**Authors**: Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2510.04618)  

**Abstract**: Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on context adaptation -- modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. Building on the adaptive memory introduced by Dynamic Cheatsheet, we introduce ACE (Agentic Context Engineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline (e.g., system prompts) and online (e.g., agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller open-source model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead. 

---
# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning 

**Authors**: Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, Lianhui Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.04573)  

**Abstract**: Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion. 

---
# More Than Meets the Eye? Uncovering the Reasoning-Planning Disconnect in Training Vision-Language Driving Models 

**Authors**: Xurui Song, Shuo Huai, JingJing Jiang, Jiayi Kong, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04532)  

**Abstract**: Vision-Language Model (VLM) driving agents promise explainable end-to-end autonomy by first producing natural-language reasoning and then predicting trajectory planning. However, whether planning is causally driven by this reasoning remains a critical but unverified assumption. To investigate this, we build DriveMind, a large-scale driving Visual Question Answering (VQA) corpus with plan-aligned Chain-of-Thought (CoT), automatically generated from nuPlan. Our data generation process converts sensors and annotations into structured inputs and, crucially, separates priors from to-be-reasoned signals, enabling clean information ablations. Using DriveMind, we train representative VLM agents with Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) and evaluate them with nuPlan's metrics. Our results, unfortunately, indicate a consistent causal disconnect in reasoning-planning: removing ego/navigation priors causes large drops in planning scores, whereas removing CoT produces only minor changes. Attention analysis further shows that planning primarily focuses on priors rather than the CoT. Based on this evidence, we propose the Reasoning-Planning Decoupling Hypothesis, positing that the training-yielded reasoning is an ancillary byproduct rather than a causal mediator. To enable efficient diagnosis, we also introduce a novel, training-free probe that measures an agent's reliance on priors by evaluating its planning robustness against minor input perturbations. In summary, we provide the community with a new dataset and a diagnostic tool to evaluate the causal fidelity of future models. 

---
# ChartAgent: A Multimodal Agent for Visually Grounded Reasoning in Complex Chart Question Answering 

**Authors**: Rachneet Kaur, Nishan Srishankar, Zhen Zeng, Sumitra Ganesh, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2510.04514)  

**Abstract**: Recent multimodal LLMs have shown promise in chart-based visual question answering, but their performance declines sharply on unannotated charts, those requiring precise visual interpretation rather than relying on textual shortcuts. To address this, we introduce ChartAgent, a novel agentic framework that explicitly performs visual reasoning directly within the chart's spatial domain. Unlike textual chain-of-thought reasoning, ChartAgent iteratively decomposes queries into visual subtasks and actively manipulates and interacts with chart images through specialized actions such as drawing annotations, cropping regions (e.g., segmenting pie slices, isolating bars), and localizing axes, using a library of chart-specific vision tools to fulfill each subtask. This iterative reasoning process closely mirrors human cognitive strategies for chart comprehension. ChartAgent achieves state-of-the-art accuracy on the ChartBench and ChartX benchmarks, surpassing prior methods by up to 16.07% absolute gain overall and 17.31% on unannotated, numerically intensive queries. Furthermore, our analyses show that ChartAgent is (a) effective across diverse chart types, (b) achieve the highest scores across varying visual and reasoning complexity levels, and (c) serves as a plug-and-play framework that boosts performance across diverse underlying LLMs. Our work is among the first to demonstrate visually grounded reasoning for chart understanding using tool-augmented multimodal agents. 

---
# P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs 

**Authors**: Shuai Zhao, Xinyi Wu, Shiqian Zhao, Xiaobao Wu, Zhongliang Guo, Yanhao Jia, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04503)  

**Abstract**: During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community. 

---
# Impatient Users Confuse AI Agents: High-fidelity Simulations of Human Traits for Testing Agents 

**Authors**: Muyu He, Anand Kumar, Tsach Mackey, Meghana Rajeev, James Zou, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2510.04491)  

**Abstract**: Despite rapid progress in building conversational AI agents, robustness is still largely untested. Small shifts in user behavior, such as being more impatient, incoherent, or skeptical, can cause sharp drops in agent performance, revealing how brittle current AI agents are. Today's benchmarks fail to capture this fragility: agents may perform well under standard evaluations but degrade spectacularly in more realistic and varied settings. We address this robustness testing gap by introducing TraitBasis, a lightweight, model-agnostic method for systematically stress testing AI agents. TraitBasis learns directions in activation space corresponding to steerable user traits (e.g., impatience or incoherence), which can be controlled, scaled, composed, and applied at inference time without any fine-tuning or extra data. Using TraitBasis, we extend $\tau$-Bench to $\tau$-Trait, where user behaviors are altered via controlled trait vectors. We observe on average a 2%-30% performance degradation on $\tau$-Trait across frontier models, highlighting the lack of robustness of current AI agents to variations in user behavior. Together, these results highlight both the critical role of robustness testing and the promise of TraitBasis as a simple, data-efficient, and compositional tool. By powering simulation-driven stress tests and training loops, TraitBasis opens the door to building AI agents that remain reliable in the unpredictable dynamics of real-world human interactions. We have open-sourced $\tau$-Trai across four domains: airline, retail, telecom, and telehealth, so the community can systematically QA their agents under realistic, behaviorally diverse intents and trait scenarios: this https URL. 

---
# MedCLM: Learning to Localize and Reason via a CoT-Curriculum in Medical Vision-Language Models 

**Authors**: Soo Yong Kim, Suin Cho, Vincent-Daniel Yun, Gyeongyeon Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04477)  

**Abstract**: Bridging clinical diagnostic reasoning with AI remains a central challenge in medical imaging. We introduce MedCLM, an automated pipeline that converts detection datasets into large-scale medical visual question answering (VQA) data with Chain-of-Thought (CoT) reasoning by linking lesion boxes to organ segmentation and structured rationales. These contextual signals enable medical vision-language models to generate question-answer pairs with step-by-step reasoning. To utilize this data effectively, we propose an Integrated CoT-Curriculum Strategy composed of an Easy stage with explicit lesion boxes for visual grounding, a Medium stage that encourages implicit localization, and a Hard stage for weakly supervised reasoning. Experimental results demonstrate that MedCLM attains state-of-the-art performance on several medical VQA benchmarks, providing a scalable framework for developing clinically aligned medical vision-language models. 

---
# Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions 

**Authors**: Wenyuan Zhao, Adithya Balachandran, Chao Tian, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04417)  

**Abstract**: The study of multimodality has garnered significant interest in fields where the analysis of interactions among multiple information sources can enhance predictive modeling, data fusion, and interpretability. Partial information decomposition (PID) has emerged as a useful information-theoretic framework to quantify the degree to which individual modalities independently, redundantly, or synergistically convey information about a target variable. However, existing PID methods depend on optimizing over a joint distribution constrained by estimated pairwise probability distributions, which are costly and inaccurate for continuous and high-dimensional modalities. Our first key insight is that the problem can be solved efficiently when the pairwise distributions are multivariate Gaussians, and we refer to this problem as Gaussian PID (GPID). We propose a new gradient-based algorithm that substantially improves the computational efficiency of GPID based on an alternative formulation of the underlying optimization problem. To generalize the applicability to non-Gaussian data, we learn information-preserving encoders to transform random variables of arbitrary input distributions into pairwise Gaussian random variables. Along the way, we resolved an open problem regarding the optimality of joint Gaussian solutions for GPID. Empirical validation in diverse synthetic examples demonstrates that our proposed method provides more accurate and efficient PID estimates than existing baselines. We further evaluate a series of large-scale multimodal benchmarks to show its utility in real-world applications of quantifying PID in multimodal datasets and selecting high-performing models. 

---
# Internal World Models as Imagination Networks in Cognitive Agents 

**Authors**: Saurabh Ranjan, Brian Odegaard  

**Link**: [PDF](https://arxiv.org/pdf/2510.04391)  

**Abstract**: What is the computational objective of imagination? While classical interpretations suggest imagination is useful for maximizing rewards, recent findings challenge this view. In this study, we propose that imagination serves to access an internal world model (IWM) and use psychological network analysis to explore IWMs in humans and large language models (LLMs). Specifically, we assessed imagination vividness ratings using two questionnaires and constructed imagination networks from these reports. Imagination networks from human groups showed correlations between different centrality measures, including expected influence, strength, and closeness. However, imagination networks from LLMs showed a lack of clustering and lower correlations between centrality measures under different prompts and conversational memory conditions. Together, these results indicate a lack of similarity between IWMs in human and LLM agents. Overall, our study offers a novel method for comparing internally-generated representations in humans and AI, providing insights for developing human-like imagination in artificial intelligence. 

---
# MorphoSim: An Interactive, Controllable, and Editable Language-guided 4D World Simulator 

**Authors**: Xuehai He, Shijie Zhou, Thivyanth Venkateswaran, Kaizhi Zheng, Ziyu Wan, Achuta Kadambi, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04390)  

**Abstract**: World models that support controllable
and editable spatiotemporal environments are valuable
for robotics, enabling scalable training data, repro ducible evaluation, and flexible task design. While
recent text-to-video models generate realistic dynam ics, they are constrained to 2D views and offer limited
interaction. We introduce MorphoSim, a language guided framework that generates 4D scenes with
multi-view consistency and object-level controls. From
natural language instructions, MorphoSim produces
dynamic environments where objects can be directed,
recolored, or removed, and scenes can be observed
from arbitrary viewpoints. The framework integrates
trajectory-guided generation with feature field dis tillation, allowing edits to be applied interactively
without full re-generation. Experiments show that Mor phoSim maintains high scene fidelity while enabling
controllability and editability. The code is available
at this https URL. 

---
# MacroBench: A Novel Testbed for Web Automation Scripts via Large Language Models 

**Authors**: Hyunjun Kim, Sejong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.04363)  

**Abstract**: We introduce MacroBench, a code-first benchmark that evaluates whether LLMs can synthesize reusable browser automation programs from natural language goals by reading HTML/DOM and emitting Python with Selenium. MacroBench instantiates seven self-hosted sites: Airbnb-like, TikTok-like, Reddit-like, Instagram-like, Facebook-like, Discord-like, and Threads-like, covering 681 tasks across interaction complexity and targeting difficulty. Our end-to-end protocol validates generated code via static checks, sandboxed execution, and outcome verification including DOM assertions and database snapshots, and includes a safety suite for scraping, spam/abuse, and credential/privacy prompts. Across 2636 model-task runs, we observe stratified success: GPT-4o-Mini achieves 96.8 percent, GPT-4.1 achieves 95.3 percent, Gemini-2.5-Pro achieves 89.0 percent, and DeepSeek-V3.1 achieves 83.4 percent. Models handle simple tasks reliably at 91.7 percent but fail on complex workflows at 0.0 percent, and none meet production-quality coding practices despite functional completion. We release our complete benchmark pipeline, evaluation framework, and experimental results to enable reproducible assessment of macro synthesis for web automation. 

---
# Wave-PDE Nets: Trainable Wave-Equation Layers as an Alternative to Attention 

**Authors**: Harshil Vejendla  

**Link**: [PDF](https://arxiv.org/pdf/2510.04304)  

**Abstract**: We introduce Wave-PDE Nets, a neural architecture whose elementary operation is a differentiable simulation of the second-order wave equation. Each layer propagates its hidden state as a continuous field through a medium with trainable spatial velocity c(x) and damping {\gamma}(x). A symplectic spectral solver based on FFTs realises this propagation in O(nlog n) time. This oscillatory, global mechanism provides a powerful alternative to attention and first-order state-space models. We prove that a single Wave-PDE layer is a universal approximator. On language and vision benchmarks, Wave-PDE Nets match or exceed Transformer performance while demonstrating superior practical efficiency, reducing wall-clock time by up to 30% and peak memory by 25%. Ablation studies confirm the critical role of symplectic integration and a spectral Laplacian for stability and performance. Visualizations of the learned physical parameters reveal that the model learns intuitive strategies for information propagation. These results position Wave-PDE Nets as a computationally efficient and robust architecture with a strong physical inductive bias. 

---
# Don't Pass$\mathtt{@}k$: A Bayesian Framework for Large Language Model Evaluation 

**Authors**: Mohsen Hariri, Amirhossein Samandar, Michael Hinczewski, Vipin Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2510.04265)  

**Abstract**: Pass$@k$ is widely used to report performance for LLM reasoning, but it often yields unstable, misleading rankings, especially when the number of trials (samples) is limited and compute is constrained. We present a principled Bayesian evaluation framework that replaces Pass$@k$ and average accuracy over $N$ trials (avg$@N$) with posterior estimates of a model's underlying success probability and credible intervals, yielding stable rankings and a transparent decision rule for differences. Evaluation outcomes are modeled as categorical (not just 0/1) with a Dirichlet prior, giving closed-form expressions for the posterior mean and uncertainty of any weighted rubric and enabling the use of prior evidence when appropriate. Theoretically, under a uniform prior, the Bayesian posterior mean is order-equivalent to average accuracy (Pass$@1$), explaining its empirical robustness while adding principled uncertainty. Empirically, in simulations with known ground-truth success rates and on AIME'24/'25, HMMT'25, and BrUMO'25, the Bayesian/avg procedure achieves faster convergence and greater rank stability than Pass$@k$ and recent variants, enabling reliable comparisons at far smaller sample counts. The framework clarifies when observed gaps are statistically meaningful (non-overlapping credible intervals) versus noise, and it naturally extends to graded, rubric-based evaluations. Together, these results recommend replacing Pass$@k$ for LLM evaluation and ranking with a posterior-based, compute-efficient protocol that unifies binary and non-binary evaluation while making uncertainty explicit. Code is available at this https URL 

---
# Zoom-In to Sort AI-Generated Images Out 

**Authors**: Yikun Ji, Yan Hong, Bowen Deng, jun lan, Huijia Zhu, Weiqiang Wang, Liqing Zhang, Jianfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04225)  

**Abstract**: The rapid growth of AI-generated imagery has blurred the boundary between real and synthetic content, raising critical concerns for digital integrity. Vision-language models (VLMs) offer interpretability through explanations but often fail to detect subtle artifacts in high-quality synthetic images. We propose ZoomIn, a two-stage forensic framework that improves both accuracy and interpretability. Mimicking human visual inspection, ZoomIn first scans an image to locate suspicious regions and then performs a focused analysis on these zoomed-in areas to deliver a grounded verdict. To support training, we introduce MagniFake, a dataset of 20,000 real and high-quality synthetic images annotated with bounding boxes and forensic explanations, generated through an automated VLM-based pipeline. Our method achieves 96.39% accuracy with robust generalization, while providing human-understandable explanations grounded in visual evidence. 

---
# Beyond Next-Token Prediction: A Performance Characterization of Diffusion versus Autoregressive Language Models 

**Authors**: Minseo Kim, Coleman Hooper, Aditya Tomar, Chenfeng Xu, Mehrdad Farajtabar, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2510.04146)  

**Abstract**: Large Language Models (LLMs) have achieved state-of-the-art performance on a broad range of Natural Language Processing (NLP) tasks, including document processing and coding. Autoregressive Language Models (ARMs), which generate tokens sequentially conditioned on all previous tokens, have been the predominant paradigm for LLMs. However, while these networks have achieved high accuracy across a range of downstream tasks, they exhibit low arithmetic intensity due to the inherent sequential dependency with next-token prediction. Recently, Diffusion Language Models (DLMs) have emerged as a promising alternative architecture. DLMs generate output text in parallel, breaking the limitations of sequential dependency. However, the performance implications of DLMs relative to commonly deployed ARMs are not fully understood. In this work, we present a comprehensive performance study analyzing the performance characteristics of ARMs and DLMs, using both theoretical analysis and profiling data to characterize the trade-offs between these approaches. We illustrate that although DLMs exhibit higher arithmetic intensity compared to ARMs because of their capability to utilize parallelism across sequence lengths, they fail to scale effectively to longer contexts. We then explore DLMs with block-wise decoding, outlining how this approach allows for increased arithmetic intensity, while still scaling well to long contexts (similar to ARMs). We also show interesting trade-offs for batched inference, where we find that ARMs exhibit superior throughput, as they benefit more from parallelism across sequences in the batch. Finally, we highlight opportunities for accelerating DLM inference, and, in particular, highlight the importance of reducing the number of sampling steps for allowing open-source DLMs to provide improved latency relative to ARMs. 

---
# Automating construction safety inspections using a multi-modal vision-language RAG framework 

**Authors**: Chenxin Wang, Elyas Asadi Shamsabadi, Zhaohui Chen, Luming Shen, Alireza Ahmadian Fard Fini, Daniel Dias-da-Costa  

**Link**: [PDF](https://arxiv.org/pdf/2510.04145)  

**Abstract**: Conventional construction safety inspection methods are often inefficient as they require navigating through large volume of information. Recent advances in large vision-language models (LVLMs) provide opportunities to automate safety inspections through enhanced visual and linguistic understanding. However, existing applications face limitations including irrelevant or unspecific responses, restricted modal inputs and hallucinations. Utilisation of Large Language Models (LLMs) for this purpose is constrained by availability of training data and frequently lack real-time adaptability. This study introduces SiteShield, a multi-modal LVLM-based Retrieval-Augmented Generation (RAG) framework for automating construction safety inspection reports by integrating visual and audio inputs. Using real-world data, SiteShield outperformed unimodal LLMs without RAG with an F1 score of 0.82, hamming loss of 0.04, precision of 0.76, and recall of 0.96. The findings indicate that SiteShield offers a novel pathway to enhance information retrieval and efficiency in generating safety reports. 

---
# Selective Expert Guidance for Effective and Diverse Exploration in Reinforcement Learning of LLMs 

**Authors**: Zishang Jiang, Jinyi Han, Tingyun Li, Xinyi Wang, Sihang Jiang, Jiaqing Liang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04140)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has become a widely adopted technique for enhancing the reasoning ability of Large Language Models (LLMs). However, the effectiveness of RLVR strongly depends on the capability of base models. This issue arises because it requires the model to have sufficient capability to perform high-quality exploration, which involves both effectiveness and diversity. Unfortunately, existing methods address this issue by imitating expert trajectories, which improve effectiveness but neglect diversity. To address this, we argue that the expert only needs to provide guidance only at critical decision points rather than the entire reasoning path. Based on this insight, we propose MENTOR: Mixed-policy Expert Navigation for Token-level Optimization of Reasoning, a framework that provides expert guidance only at critical decision points to perform effective and diverse exploration in RLVR. Extensive experiments show that MENTOR enables models capture the essence of expert strategies rather than surface imitation, thereby performing high-quality exploration and achieving superior overall performance. Our code is available online. 

---
# Internal states before wait modulate reasoning patterns 

**Authors**: Dmitrii Troitskii, Koyena Pal, Chris Wendler, Callum Stuart McDougall, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.04128)  

**Abstract**: Prior work has shown that a significant driver of performance in reasoning models is their ability to reason and self-correct. A distinctive marker in these reasoning traces is the token wait, which often signals reasoning behavior such as backtracking. Despite being such a complex behavior, little is understood of exactly why models do or do not decide to reason in this particular manner, which limits our understanding of what makes a reasoning model so effective. In this work, we address the question whether model's latents preceding wait tokens contain relevant information for modulating the subsequent reasoning process. We train crosscoders at multiple layers of DeepSeek-R1-Distill-Llama-8B and its base version, and introduce a latent attribution technique in the crosscoder setting. We locate a small set of features relevant for promoting/suppressing wait tokens' probabilities. Finally, through a targeted series of experiments analyzing max activating examples and causal interventions, we show that many of our identified features indeed are relevant for the reasoning process and give rise to different types of reasoning patterns such as restarting from the beginning, recalling prior knowledge, expressing uncertainty, and double-checking. 

---
# Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning 

**Authors**: Ziyan Wang, Zheng Wang, Jie Fu, Xingwei Qu, Qi Cheng, Shengpu Tang, Minjia Zhang, Xiaoming Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.04072)  

**Abstract**: Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and a 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy. 

---
# What Scales in Cross-Entropy Scaling Law? 

**Authors**: Junxi Yan, Zixi Wei, Jingtao Zhan, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.04067)  

**Abstract**: The cross-entropy scaling law has long served as a key tool for guiding the development of large language models. It shows that cross-entropy loss decreases in a predictable power-law rate as the model size increases. However, recent evidence indicates that this law breaks down at very large scales: the loss decreases more slowly than expected, which causes significant trouble for developing large language models. In this paper, we hypothesize that the root cause lies in the fact that cross-entropy itself does not truly scale; instead, only one of its hidden components does. To investigate this, we introduce a novel decomposition of cross-entropy into three parts: Error-Entropy, Self-Alignment, and Confidence. We show both theoretically and empirically that this decomposition precisely captures the training dynamics and optimization objectives. Through extensive experiments on multiple datasets and 32 models spanning five orders of magnitude in size, we find that only error-entropy follows a robust power-law scaling, while the other two terms remain largely invariant. Moreover, error-entropy constitutes the dominant share of cross-entropy in small models but diminishes in proportion as models grow larger. This explains why the cross-entropy scaling law appears accurate at small scales but fails at very large ones. Our findings establish the error-entropy scaling law as a more accurate description of model behavior. We believe it will have wide applications in the training, understanding, and future development of large language models. 

---
# LLM-Based Data Science Agents: A Survey of Capabilities, Challenges, and Future Directions 

**Authors**: Mizanur Rahman, Amran Bhuiyan, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Ridwan Mahbub, Ahmed Masry, Shafiq Joty, Enamul Hoque  

**Link**: [PDF](https://arxiv.org/pdf/2510.04023)  

**Abstract**: Recent advances in large language models (LLMs) have enabled a new class of AI agents that automate multiple stages of the data science workflow by integrating planning, tool use, and multimodal reasoning across text, code, tables, and visuals. This survey presents the first comprehensive, lifecycle-aligned taxonomy of data science agents, systematically analyzing and mapping forty-five systems onto the six stages of the end-to-end data science process: business understanding and data acquisition, exploratory analysis and visualization, feature engineering, model building and selection, interpretation and explanation, and deployment and monitoring. In addition to lifecycle coverage, we annotate each agent along five cross-cutting design dimensions: reasoning and planning style, modality integration, tool orchestration depth, learning and alignment methods, and trust, safety, and governance mechanisms. Beyond classification, we provide a critical synthesis of agent capabilities, highlight strengths and limitations at each stage, and review emerging benchmarks and evaluation practices. Our analysis identifies three key trends: most systems emphasize exploratory analysis, visualization, and modeling while neglecting business understanding, deployment, and monitoring; multimodal reasoning and tool orchestration remain unresolved challenges; and over 90% lack explicit trust and safety mechanisms. We conclude by outlining open challenges in alignment stability, explainability, governance, and robust evaluation frameworks, and propose future research directions to guide the development of robust, trustworthy, low-latency, transparent, and broadly accessible data science agents. 

---
# Principled and Tractable RL for Reasoning with Diffusion Language Models 

**Authors**: Anthony Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2510.04019)  

**Abstract**: Diffusion large language models (dLLMs) are a new paradigm of non-autoregressive language models that are trained to predict multiple tokens in parallel and generate text via iterative unmasking. Recent works have successfully pretrained dLLMs to parity with autoregressive LLMs at the 8B scale, but dLLMs have yet to benefit from modern post-training techniques, e.g. reinforcement learning (RL), that have proven effective for autoregressive models. Crucially, algorithms designed for traditional LLMs aren't directly compatible with diffusion frameworks due to inherent differences in modeling assumptions. Moreover, existing attempts at dLLM post-training with RL rely on heuristic-based objectives with no theoretical grounding. In this work, we present Amortized Group Relative Policy Optimization (AGRPO), a principled on-policy RL algorithm designed specifically for dLLMs. AGRPO uses Monte Carlo sampling to compute an unbiased policy gradient estimate, making it the first tractable, faithful adaptation of policy gradient methods for dLLMs. We demonstrate AGRPO's effectiveness on different math/reasoning tasks, a common setting for RL with LLMs, achieving up to +7.6% absolute gain on GSM8K and 3.8x performance on the Countdown task over the baseline LLaDA-8B-Instruct model and 1.3x performance gains over comparable RL methods such as diffu-GRPO. Furthermore, these gains persist across different numbers of sampling steps at inference time, achieving better tradeoffs between compute and performance. Our results demonstrate that online RL algorithms can be extended to diffusion LLMs in principled ways, maintaining both theoretical soundness and practical effectiveness. 

---
# Visual Lifelog Retrieval through Captioning-Enhanced Interpretation 

**Authors**: Yu-Fei Shih, An-Zi Yen, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04010)  

**Abstract**: People often struggle to remember specific details of past experiences, which can lead to the need to revisit these memories. Consequently, lifelog retrieval has emerged as a crucial application. Various studies have explored methods to facilitate rapid access to personal lifelogs for memory recall assistance. In this paper, we propose a Captioning-Integrated Visual Lifelog (CIVIL) Retrieval System for extracting specific images from a user's visual lifelog based on textual queries. Unlike traditional embedding-based methods, our system first generates captions for visual lifelogs and then utilizes a text embedding model to project both the captions and user queries into a shared vector space. Visual lifelogs, captured through wearable cameras, provide a first-person viewpoint, necessitating the interpretation of the activities of the individual behind the camera rather than merely describing the scene. To address this, we introduce three distinct approaches: the single caption method, the collective caption method, and the merged caption method, each designed to interpret the life experiences of lifeloggers. Experimental results show that our method effectively describes first-person visual images, enhancing the outcomes of lifelog retrieval. Furthermore, we construct a textual dataset that converts visual lifelogs into captions, thereby reconstructing personal life experiences. 

---
# What Shapes a Creative Machine Mind? Comprehensively Benchmarking Creativity in Foundation Models 

**Authors**: Zicong He, Boxuan Zhang, Weihao Liu, Ruixiang Tang, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04009)  

**Abstract**: The meteoric rise of foundation models (FMs) has expanded their capabilities far beyond conventional tasks. Creativity, long regarded as a hallmark of human intelligence and a driver of innovation, is now increasingly recognized as a critical dimension of machine intelligence in the era of generative FMs, complementing traditional measures of accuracy. However, existing evaluation frameworks for creativity remain fragmented, relying on ad hoc metrics not firmly grounded in established theories. To address this gap, we introduce C^2-Eval, a holistic benchmark for unified assessment of creativity in FMs. C^2-Eval distinguishes between two complementary forms of creativity: convergent creativity, where tasks admit constrained solutions (e.g., code generation), and divergent creativity, where tasks are open-ended (e.g., storytelling). It evaluates both dimensions using fine-grained criteria derived from social-science theory, focusing on Usefulness, Originality, and Surprise (U-O-S). Through extensive experiments on leading proprietary and open-source models, we analyze trade-offs in their creative capabilities. Our results highlight both the strengths and challenges of current FMs in pursuing a creative machine mind, showing that C^2-Eval is an effective lens for examining the evolving landscape of creative AI. 

---
# Enhancing OCR for Sino-Vietnamese Language Processing via Fine-tuned PaddleOCRv5 

**Authors**: Minh Hoang Nguyen, Su Nguyen Thiet  

**Link**: [PDF](https://arxiv.org/pdf/2510.04003)  

**Abstract**: Recognizing and processing Classical Chinese (Han-Nom) texts play a vital role in digitizing Vietnamese historical documents and enabling cross-lingual semantic research. However, existing OCR systems struggle with degraded scans, non-standard glyphs, and handwriting variations common in ancient sources. In this work, we propose a fine-tuning approach for PaddleOCRv5 to improve character recognition on Han-Nom texts. We retrain the text recognition module using a curated subset of ancient Vietnamese Chinese manuscripts, supported by a full training pipeline covering preprocessing, LMDB conversion, evaluation, and visualization. Experimental results show a significant improvement over the base model, with exact accuracy increasing from 37.5 percent to 50.0 percent, particularly under noisy image conditions. Furthermore, we develop an interactive demo that visually compares pre- and post-fine-tuning recognition results, facilitating downstream applications such as Han-Vietnamese semantic alignment, machine translation, and historical linguistics research. The demo is available at this https URL. 

---
# No Tokens Wasted: Leveraging Long Context in Biomedical Vision-Language Models 

**Authors**: Min Woo Sun, Alejandro Lozano, Javier Gamazo Tejero, Vishwesh Nath, Xiao Xiao Sun, James Burgess, Yuhui Zhang, Kun Yuan, Robert Tibshirani, Sean Huver, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2510.03978)  

**Abstract**: Embedding vision-language models (VLMs) are typically pretrained with short text windows (<77 tokens), which forces the truncation of long-format captions. Yet, the distribution of biomedical captions from large-scale open source literature reveals that a huge portion of captions far exceed 77 tokens. To this end, we investigate the impact of pretraining on long-format biomedical captions by extending the context length of text encoders in VLMs. We find that longer context (thus, enabling additional supervision provided in long-format captions) correlates with better retrieval and classification performance. Given this finding, we introduce BIOMEDICA-LongCAP, a dataset of 1M image-caption pairs enriched with context-aware descriptions from full-text articles, providing longer and additional textual supervision. Using BIOMEDICA-LongCAP, we train BMC-LongCLIP, a long-context biomedical VLM with a text encoder supporting windows of up to 512 tokens. Our model extends context capacity by 6.6x, reducing token waste from 55% to just 2.2%. On long-caption retrieval benchmarks, BMC-LongCLIP achieves up to +30% absolute gains in Recall@1 and +2% average improvements in classification, while also converging faster than short-context. Our results demonstrate that long-context modeling is a promising direction for advancing biomedical VLMs. 

---
# LLM Chemistry Estimation for Multi-LLM Recommendation 

**Authors**: Huascar Sanchez, Briland Hitaj  

**Link**: [PDF](https://arxiv.org/pdf/2510.03930)  

**Abstract**: Multi-LLM collaboration promises accurate, robust, and context-aware solutions, yet existing approaches rely on implicit selection and output assessment without analyzing whether collaborating models truly complement or conflict. We introduce LLM Chemistry -- a framework that measures when LLM combinations exhibit synergistic or antagonistic behaviors that shape collective performance beyond individual capabilities. We formalize the notion of chemistry among LLMs, propose algorithms that quantify it by analyzing interaction dependencies, and recommend optimal model ensembles accordingly. Our theoretical analysis shows that chemistry among collaborating LLMs is most evident under heterogeneous model profiles, with its outcome impact shaped by task type, group size, and complexity. Evaluation on classification, summarization, and program repair tasks provides initial evidence for these task-dependent effects, thereby reinforcing our theoretical results. This establishes LLM Chemistry as both a diagnostic factor in multi-LLM systems and a foundation for ensemble recommendation. 

---
# Kantian-Utilitarian XAI: Meta-Explained 

**Authors**: Zahra Atf, Peter R. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03892)  

**Abstract**: We present a gamified explainable AI (XAI) system for ethically aware consumer decision-making in the coffee domain. Each session comprises six rounds with three options per round. Two symbolic engines provide real-time reasons: a Kantian module flags rule violations (e.g., child labor, deforestation risk without shade certification, opaque supply chains, unsafe decaf), and a utilitarian module scores options via multi-criteria aggregation over normalized attributes (price, carbon, water, transparency, farmer income share, taste/freshness, packaging, convenience). A meta-explainer with a regret bound (0.2) highlights Kantian--utilitarian (mis)alignment and switches to a deontically clean, near-parity option when welfare loss is small. We release a structured configuration (attribute schema, certification map, weights, rule set), a policy trace for auditability, and an interactive UI. 

---
# Unlocking Reasoning Capabilities in LLMs via Reinforcement Learning Exploration 

**Authors**: Wenhao Deng, Long Wei, Chenglei Yu, Tailin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03865)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has recently enhanced the reasoning capabilities of large language models (LLMs), particularly for mathematical problem solving. However, a fundamental limitation remains: as the sampling budget increases, the advantage of RLVR-trained models over their pretrained bases often diminishes or even vanishes, revealing a strong dependence on the base model's restricted search space. We attribute this phenomenon to the widespread use of the reverse Kullback-Leibler (KL) divergence regularizer, whose mode-seeking behavior keeps the policy trapped inside the base model's support region and hampers wider exploration. To address this issue, we propose RAPO (Rewards-Aware Policy Optimization), an algorithm to promote broader yet focused exploration. Our method (i) utilizes the forward KL penalty to replace the reverse KL penalty for out-of-distribution exploration, and (ii) reweights the reference policy to facilitate adaptive in-distribution exploration. We train Qwen2.5-3B and 7B models with RAPO on the 8K SimpleRL-Zero dataset, without supervised fine-tuning, and evaluate them on AIME2024 and AIME2025. Results show that RAPO consistently improves problem-solving performance. Notably, RAPO enables models to surpass the base model's performance ceiling and solves previously intractable problems, advancing the frontier of RLVR for challenging reasoning tasks. 

---
# Investigating LLM Variability in Personalized Conversational Information Retrieval 

**Authors**: Simon Lupart, Daniël van Dijk, Eric Langezaal, Ian van Dort, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03795)  

**Abstract**: Personalized Conversational Information Retrieval (CIR) has seen rapid progress in recent years, driven by the development of Large Language Models (LLMs). Personalized CIR aims to enhance document retrieval by leveraging user-specific information, such as preferences, knowledge, or constraints, to tailor responses to individual needs. A key resource for this task is the TREC iKAT 2023 dataset, designed to evaluate personalization in CIR pipelines. Building on this resource, Mo et al. explored several strategies for incorporating Personal Textual Knowledge Bases (PTKB) into LLM-based query reformulation. Their findings suggested that personalization from PTKBs could be detrimental and that human annotations were often noisy. However, these conclusions were based on single-run experiments using the GPT-3.5 Turbo model, raising concerns about output variability and repeatability. In this reproducibility study, we rigorously reproduce and extend their work, focusing on LLM output variability and model generalization. We apply the original methods to the new TREC iKAT 2024 dataset and evaluate a diverse range of models, including Llama (1B-70B), Qwen-7B, GPT-4o-mini. Our results show that human-selected PTKBs consistently enhance retrieval performance, while LLM-based selection methods do not reliably outperform manual choices. We further compare variance across datasets and observe higher variability on iKAT than on CAsT, highlighting the challenges of evaluating personalized CIR. Notably, recall-oriented metrics exhibit lower variance than precision-oriented ones, a critical insight for first-stage retrievers. Finally, we underscore the need for multi-run evaluations and variance reporting when assessing LLM-based CIR systems. By broadening evaluation across models, datasets, and metrics, our study contributes to more robust and generalizable practices for personalized CIR. 

---
# Optimizing Fine-Tuning through Advanced Initialization Strategies for Low-Rank Adaptation 

**Authors**: Yongfu Xue  

**Link**: [PDF](https://arxiv.org/pdf/2510.03731)  

**Abstract**: The rapid development of parameter-efficient fine-tuning methods has noticeably improved the efficiency of adapting large language models. Among these, LoRA has gained widespread popularity due to its strong balance of effectiveness and parameter efficiency. However, LoRA relies on initializing two low-rank matrices whose product is zero, which limits its ability to effectively activate and leverage the original model weights-creating a potential bottleneck for optimal performance. To address this limitation, we propose \textbf{IniLoRA}, a novel initialization strategy that initializes the low-rank matrices to closely approximate the original model weights. Experimental results indicate that IniLoRA achieves better performance than LoRA across a range of models and tasks. Additionally, we introduce two variants, IniLoRA-$\alpha$ and IniLoRA-$\beta$, both leveraging distinct initialization methods to enhance performance further. 

---
# Bridging the Gap Between Multimodal Foundation Models and World Models 

**Authors**: Xuehai He  

**Link**: [PDF](https://arxiv.org/pdf/2510.03727)  

**Abstract**: Humans understand the world through the integration of multiple sensory modalities, enabling them to perceive, reason about, and imagine dynamic physical processes. Inspired by this capability, multimodal foundation models (MFMs) have emerged as powerful tools for multimodal understanding and generation. However, today's MFMs fall short of serving as effective world models. They lack the essential ability such as perform counterfactual reasoning, simulate dynamics, understand the spatiotemporal information, control generated visual outcomes, and perform multifaceted reasoning. We investigates what it takes to bridge the gap between multimodal foundation models and world models. We begin by improving the reasoning capabilities of MFMs through discriminative tasks and equipping MFMs with structured reasoning skills, such as causal inference, counterfactual thinking, and spatiotemporal reasoning, enabling them to go beyond surface correlations and understand deeper relationships within visual and textual data. Next, we explore generative capabilities of multimodal foundation models across both image and video modalities, introducing new frameworks for structured and controllable generation. Our approaches incorporate scene graphs, multimodal conditioning, and multimodal alignment strategies to guide the generation process, ensuring consistency with high-level semantics and fine-grained user intent. We further extend these techniques to controllable 4D generation, enabling interactive, editable, and morphable object synthesis over time and space. 

---
# Adapting Diarization-Conditioned Whisper for End-to-End Multi-Talker Speech Recognition 

**Authors**: Martin Kocour, Martin Karafiat, Alexander Polok, Dominik Klement, Lukáš Burget, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2510.03723)  

**Abstract**: We propose a speaker-attributed (SA) Whisper-based model for multi-talker speech recognition that combines target-speaker modeling with serialized output training (SOT). Our approach leverages a Diarization-Conditioned Whisper (DiCoW) encoder to extract target-speaker embeddings, which are concatenated into a single representation and passed to a shared decoder. This enables the model to transcribe overlapping speech as a serialized output stream with speaker tags and timestamps. In contrast to target-speaker ASR systems such as DiCoW, which decode each speaker separately, our approach performs joint decoding, allowing the decoder to condition on the context of all speakers simultaneously. Experiments show that the model outperforms existing SOT-based approaches and surpasses DiCoW on multi-talker mixtures (e.g., LibriMix). 

---
# Person-Centric Annotations of LAION-400M: Auditing Bias and Its Transfer to Models 

**Authors**: Leander Girrbach, Stephan Alaniz, Genevieve Smith, Trevor Darrell, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2510.03721)  

**Abstract**: Vision-language models trained on large-scale multimodal datasets show strong demographic biases, but the role of training data in producing these biases remains unclear. A major barrier has been the lack of demographic annotations in web-scale datasets such as LAION-400M. We address this gap by creating person-centric annotations for the full dataset, including over 276 million bounding boxes, perceived gender and race/ethnicity labels, and automatically generated captions. These annotations are produced through validated automatic labeling pipelines combining object detection, multimodal captioning, and finetuned classifiers. Using them, we uncover demographic imbalances and harmful associations, such as the disproportionate linking of men and individuals perceived as Black or Middle Eastern with crime-related and negative content. We also show that 60-70% of gender bias in CLIP and Stable Diffusion can be linearly explained by direct co-occurrences in the data. Our resources establish the first large-scale empirical link between dataset composition and downstream model bias. 

---
# Mind the Goal: Data-Efficient Goal-Oriented Evaluation of Conversational Agents and Chatbots using Teacher Models 

**Authors**: Deepak Babu Piskala, Sharlene Chen, Udita Patel, Parul Kalra, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2510.03696)  

**Abstract**: Evaluating the quality of multi-turn chatbot interactions remains challenging, as most existing methods assess interactions at the turn level without addressing whether a user's overarching goal was fulfilled. A ``goal'' here refers to an information need or task, such as asking for policy information or applying for leave. We propose a comprehensive framework for goal-oriented evaluation of multi-agent systems (MAS), introducing the \textbf{Goal Success Rate (GSR)} to measure the percentage of fulfilled goals, and a \textbf{Root Cause of Failure (RCOF)} taxonomy to identify reasons for failure in multi-agent chatbots. Our method segments conversations by user goals and evaluates success using all relevant turns. We present a model-based evaluation system combining teacher LLMs, where domain experts define goals, set quality standards serving as a guidance for the LLMs. The LLMs use ``thinking tokens'' to produce interpretable rationales, enabling \textit{explainable}, \textit{data-efficient} evaluations. In an enterprise setting, we apply our framework to evaluate AIDA, a zero-to-one employee conversational agent system built as a ground-up multi-agent conversational agent, and observe GSR improvement from 63\% to 79\% over six months since its inception. Our framework is generic and offers actionable insights through a detailed defect taxonomy based on analysis of failure points in multi-agent chatbots, diagnosing overall success, identifying key failure modes, and informing system improvements. 

---
# Token Hidden Reward: Steering Exploration-Exploitation in Group Relative Deep Reinforcement Learning 

**Authors**: Wenlong Deng, Yi Ren, Yushu Li, Boying Gong, Danica J. Sutherland, Xiaoxiao Li, Christos Thrampoulidis  

**Link**: [PDF](https://arxiv.org/pdf/2510.03669)  

**Abstract**: Reinforcement learning with verifiable rewards has significantly advanced the reasoning capabilities of large language models, yet how to explicitly steer training toward exploration or exploitation remains an open problem. We introduce Token Hidden Reward (THR), a token-level metric that quantifies each token's influence on the likelihood of correct responses under Group Relative Policy Optimization (GRPO). We find that training dynamics are dominated by a small subset of tokens with high absolute THR values. Most interestingly, tokens with positive THR strengthen confidence in correct outputs, thus favoring exploitation, while tokens with negative THR preserve probability mass for alternative outputs, enabling exploration. This insight suggests a natural intervention: a THR-guided reweighting algorithm that modulates GRPO's learning signals to explicitly bias training toward exploitation or exploration. We validate the efficacy of this algorithm on diverse math reasoning benchmarks. By amplifying tokens with positive THR value and weakening negative ones, our algorithm improves greedy-decoding accuracy, favoring exploitation. The reverse strategy yields consistent gains in Pass@K accuracy, favoring exploration. We further demonstrate that our algorithm integrates seamlessly with other RL objectives such as GSPO and generalizes across architectures including Llama. These findings establish THR as a principled and fine-grained mechanism for dynamically controlling exploration and exploitation in RL-tuned LLMs, providing new tools for targeted fine-tuning in reasoning-intensive applications. 

---
# Does higher interpretability imply better utility? A Pairwise Analysis on Sparse Autoencoders 

**Authors**: Xu Wang, Yan Hu, Benyou Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03659)  

**Abstract**: Sparse Autoencoders (SAEs) are widely used to steer large language models (LLMs), based on the assumption that their interpretable features naturally enable effective model behavior steering. Yet, a fundamental question remains unanswered: does higher interpretability indeed imply better steering utility? To answer this question, we train 90 SAEs across three LLMs (Gemma-2-2B, Qwen-2.5-3B, Gemma-2-9B), spanning five architectures and six sparsity levels, and evaluate their interpretability and steering utility based on SAEBench (arXiv:2501.12345) and AxBench (arXiv:2502.23456) respectively, and perform a rank-agreement analysis via Kendall's rank coefficients (tau b). Our analysis reveals only a relatively weak positive association (tau b approx 0.298), indicating that interpretability is an insufficient proxy for steering performance. We conjecture the interpretability utility gap may stem from the selection of SAE features, as not all of them are equally effective for steering. To further find features that truly steer the behavior of LLMs, we propose a novel selection criterion called Delta Token Confidence, which measures how much amplifying a feature changes the next token distribution. We show that our method improves the steering performance of three LLMs by 52.52 percent compared to the current best output score based criterion (arXiv:2503.34567). Strikingly, after selecting features with high Delta Token Confidence, the correlation between interpretability and utility vanishes (tau b approx 0), and can even become negative. This further highlights the divergence between interpretability and utility for the most effective steering features. 

---
# From Theory to Practice: Evaluating Data Poisoning Attacks and Defenses in In-Context Learning on Social Media Health Discourse 

**Authors**: Rabeya Amin Jhuma, Mostafa Mohaimen Akand Faisal  

**Link**: [PDF](https://arxiv.org/pdf/2510.03636)  

**Abstract**: This study explored how in-context learning (ICL) in large language models can be disrupted by data poisoning attacks in the setting of public health sentiment analysis. Using tweets of Human Metapneumovirus (HMPV), small adversarial perturbations such as synonym replacement, negation insertion, and randomized perturbation were introduced into the support examples. Even these minor manipulations caused major disruptions, with sentiment labels flipping in up to 67% of cases. To address this, a Spectral Signature Defense was applied, which filtered out poisoned examples while keeping the data's meaning and sentiment intact. After defense, ICL accuracy remained steady at around 46.7%, and logistic regression validation reached 100% accuracy, showing that the defense successfully preserved the dataset's integrity. Overall, the findings extend prior theoretical studies of ICL poisoning to a practical, high-stakes setting in public health discourse analysis, highlighting both the risks and potential defenses for robust LLM deployment. This study also highlights the fragility of ICL under attack and the value of spectral defenses in making AI systems more reliable for health-related social media monitoring. 

---
# Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs 

**Authors**: Fatmazohra Rezkellah, Ramzi Dakhmouche  

**Link**: [PDF](https://arxiv.org/pdf/2510.03567)  

**Abstract**: With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach. 

---
# Red Lines and Grey Zones in the Fog of War: Benchmarking Legal Risk, Moral Harm, and Regional Bias in Large Language Model Military Decision-Making 

**Authors**: Toby Drinkall  

**Link**: [PDF](https://arxiv.org/pdf/2510.03514)  

**Abstract**: As military organisations consider integrating large language models (LLMs) into command and control (C2) systems for planning and decision support, understanding their behavioural tendencies is critical. This study develops a benchmarking framework for evaluating aspects of legal and moral risk in targeting behaviour by comparing LLMs acting as agents in multi-turn simulated conflict. We introduce four metrics grounded in International Humanitarian Law (IHL) and military doctrine: Civilian Target Rate (CTR) and Dual-use Target Rate (DTR) assess compliance with legal targeting principles, while Mean and Max Simulated Non-combatant Casualty Value (SNCV) quantify tolerance for civilian harm.
We evaluate three frontier models, GPT-4o, Gemini-2.5, and LLaMA-3.1, through 90 multi-agent, multi-turn crisis simulations across three geographic regions. Our findings reveal that off-the-shelf LLMs exhibit concerning and unpredictable targeting behaviour in simulated conflict environments. All models violated the IHL principle of distinction by targeting civilian objects, with breach rates ranging from 16.7% to 66.7%. Harm tolerance escalated through crisis simulations with MeanSNCV increasing from 16.5 in early turns to 27.7 in late turns. Significant inter-model variation emerged: LLaMA-3.1 selected an average of 3.47 civilian strikes per simulation with MeanSNCV of 28.4, while Gemini-2.5 selected 0.90 civilian strikes with MeanSNCV of 17.6. These differences indicate that model selection for deployment constitutes a choice about acceptable legal and moral risk profiles in military operations.
This work seeks to provide a proof-of-concept of potential behavioural risks that could emerge from the use of LLMs in Decision Support Systems (AI DSS) as well as a reproducible benchmarking framework with interpretable metrics for standardising pre-deployment testing. 

---
# Consistent Kernel Change-Point Detection under m-Dependence for Text Segmentation 

**Authors**: Jairo Diaz-Rodriguez, Mumin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.03437)  

**Abstract**: Kernel change-point detection (KCPD) has become a widely used tool for identifying structural changes in complex data. While existing theory establishes consistency under independence assumptions, real-world sequential data such as text exhibits strong dependencies. We establish new guarantees for KCPD under $m$-dependent data: specifically, we prove consistency in the number of detected change points and weak consistency in their locations under mild additional assumptions. We perform an LLM-based simulation that generates synthetic $m$-dependent text to validate the asymptotics. To complement these results, we present the first comprehensive empirical study of KCPD for text segmentation with modern embeddings. Across diverse text datasets, KCPD with text embeddings outperforms baselines in standard text segmentation metrics. We demonstrate through a case study on Taylor Swift's tweets that KCPD not only provides strong theoretical and simulated reliability but also practical effectiveness for text segmentation tasks. 

---
# PLSEMANTICSBENCH: Large Language Models As Programming Language Interpreters 

**Authors**: Aditya Thimmaiah, Jiyang Zhang, Jayanth Srinivasa, Junyi Jessy Li, Milos Gligoric  

**Link**: [PDF](https://arxiv.org/pdf/2510.03415)  

**Abstract**: As large language models (LLMs) excel at code reasoning, a natural question arises: can an LLM execute programs (i.e., act as an interpreter) purely based on a programming language's formal semantics? If so, it will enable rapid prototyping of new programming languages and language features. We study this question using the imperative language IMP (a subset of C), formalized via small-step operational semantics (SOS) and rewriting-based operational semantics (K-semantics). We introduce three evaluation sets-Human-Written, LLM-Translated, and Fuzzer- Generated-whose difficulty is controlled by code-complexity metrics spanning the size, control-flow, and data-flow axes. Given a program and its semantics formalized with SOS/K-semantics, models are evaluated on three tasks ranging from coarse to fine: (1) final-state prediction, (2) semantic rule prediction, and (3) execution trace prediction. To distinguish pretraining memorization from semantic competence, we define two nonstandard semantics obtained through systematic mutations of the standard rules. Across strong code/reasoning LLMs, performance drops under nonstandard semantics despite high performance under the standard one. We further find that (i) there are patterns to different model failures, (ii) most reasoning models perform exceptionally well on coarse grained tasks involving reasoning about highly complex programs often containing nested loop depths beyond five, and surprisingly, (iii) providing formal semantics helps on simple programs but often hurts on more complex ones. Overall, the results show a promise that LLMs could serve as programming language interpreters, but points to the lack of their robust semantics understanding. We release the benchmark and the supporting code at this https URL. 

---
# Know Thyself? On the Incapability and Implications of AI Self-Recognition 

**Authors**: Xiaoyan Bai, Aryan Shrivastava, Ari Holtzman, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03399)  

**Abstract**: Self-recognition is a crucial metacognitive capability for AI systems, relevant not only for psychological analysis but also for safety, particularly in evaluative scenarios. Motivated by contradictory interpretations of whether models possess self-recognition (Panickssery et al., 2024; Davidson et al., 2024), we introduce a systematic evaluation framework that can be easily applied and updated. Specifically, we measure how well 10 contemporary larger language models (LLMs) can identify their own generated text versus text from other models through two tasks: binary self-recognition and exact model prediction. Different from prior claims, our results reveal a consistent failure in self-recognition. Only 4 out of 10 models predict themselves as generators, and the performance is rarely above random chance. Additionally, models exhibit a strong bias toward predicting GPT and Claude families. We also provide the first evaluation of model awareness of their own and others' existence, as well as the reasoning behind their choices in self-recognition. We find that the model demonstrates some knowledge of its own existence and other models, but their reasoning reveals a hierarchical bias. They appear to assume that GPT, Claude, and occasionally Gemini are the top-tier models, often associating high-quality text with them. We conclude by discussing the implications of our findings on AI safety and future directions to develop appropriate AI self-awareness. 

---
# Studying the Korean Word-Chain Game with RLVR:Mitigating Reward Conflicts via Curriculum Learning 

**Authors**: Donghwan Rho  

**Link**: [PDF](https://arxiv.org/pdf/2510.03394)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) is a promising approach for training large language models (LLMs) with stronger reasoning abilities. It has also been applied to a variety of logic puzzles. In this work, we study the Korean word-chain game using RLVR. We show that rule-derived rewards can naturally conflict, and demonstrate through experiments that a curriculum-learning scheme mitigates these conflicts. Our findings motivate further studies of puzzle tasks in diverse languages. 

---
# Lightweight Prompt Engineering for Cognitive Alignment in Educational AI: A OneClickQuiz Case Study 

**Authors**: Antoun Yaacoub, Zainab Assaghir, Jérôme Da-Rugna  

**Link**: [PDF](https://arxiv.org/pdf/2510.03374)  

**Abstract**: The rapid integration of Artificial Intelligence (AI) into educational technology promises to revolutionize content creation and assessment. However, the quality and pedagogical alignment of AI-generated content remain critical challenges. This paper investigates the impact of lightweight prompt engineering strategies on the cognitive alignment of AI-generated questions within OneClickQuiz, a Moodle plugin leveraging generative AI. We evaluate three prompt variants-a detailed baseline, a simpler version, and a persona-based approach-across Knowledge, Application, and Analysis levels of Bloom's Taxonomy. Utilizing an automated classification model (from prior work) and human review, our findings demonstrate that explicit, detailed prompts are crucial for precise cognitive alignment. While simpler and persona-based prompts yield clear and relevant questions, they frequently misalign with intended Bloom's levels, generating outputs that are either too complex or deviate from the desired cognitive objective. This study underscores the importance of strategic prompt engineering in fostering pedagogically sound AI-driven educational solutions and advises on optimizing AI for quality content generation in learning analytics and smart learning environments. 

---
# AgentCaster: Reasoning-Guided Tornado Forecasting 

**Authors**: Michael Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03349)  

**Abstract**: There is a growing need to evaluate Large Language Models (LLMs) on complex, high-impact, real-world tasks to assess their true readiness as reasoning agents. To address this gap, we introduce AgentCaster, a contamination-free framework employing multimodal LLMs end-to-end for the challenging, long-horizon task of tornado forecasting. Within AgentCaster, models interpret heterogeneous spatiotemporal data from a high-resolution convection-allowing forecast archive. We assess model performance over a 40-day period featuring diverse historical data, spanning several major tornado outbreaks and including over 500 tornado reports. Each day, models query interactively from a pool of 3,625 forecast maps and 40,125 forecast soundings for a forecast horizon of 12-36 hours. Probabilistic tornado-risk polygon predictions are verified against ground truths derived from geometric comparisons across disjoint risk bands in projected coordinate space. To quantify accuracy, we propose domain-specific TornadoBench and TornadoHallucination metrics, with TornadoBench highly challenging for both LLMs and domain expert human forecasters. Notably, human experts significantly outperform state-of-the-art models, which demonstrate a strong tendency to hallucinate and overpredict risk intensity, struggle with precise geographic placement, and exhibit poor spatiotemporal reasoning in complex, dynamically evolving systems. AgentCaster aims to advance research on improving LLM agents for challenging reasoning tasks in critical domains. 

---
# CAFL-L: Constraint-Aware Federated Learning with Lagrangian Dual Optimization for On-Device Language Models 

**Authors**: Dongqi Zheng, Wenjin Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03298)  

**Abstract**: We introduce Constraint-Aware Federated Learning with Lagrangian Dual Optimization (CAFL-L), a principled extension of FedAvg that explicitly incorporates device-level resource constraints including energy, communication, memory, and thermal budgets. CAFL-L employs Lagrangian dual optimization to dynamically adapt training hyperparameters -- freezing depth, local steps, batch size, and communication compression -- while preserving training stability through token-budget preservation via gradient accumulation. Experiments on a character-level language model demonstrate that CAFL-L achieves superior constraint satisfaction compared to standard FedAvg (reducing memory usage by 20% and communication by 95%) while maintaining competitive validation performance, making it practical for deployment on resource-constrained edge devices. 

---
# Multimodal Arabic Captioning with Interpretable Visual Concept Integration 

**Authors**: Passant Elchafei, Amany Fashwan  

**Link**: [PDF](https://arxiv.org/pdf/2510.03295)  

**Abstract**: We present VLCAP, an Arabic image captioning framework that integrates CLIP-based visual label retrieval with multimodal text generation. Rather than relying solely on end-to-end captioning, VLCAP grounds generation in interpretable Arabic visual concepts extracted with three multilingual encoders, mCLIP, AraCLIP, and Jina V4, each evaluated separately for label retrieval. A hybrid vocabulary is built from training captions and enriched with about 21K general domain labels translated from the Visual Genome dataset, covering objects, attributes, and scenes. The top-k retrieved labels are transformed into fluent Arabic prompts and passed along with the original image to vision-language models. In the second stage, we tested Qwen-VL and Gemini Pro Vision for caption generation, resulting in six encoder-decoder configurations. The results show that mCLIP + Gemini Pro Vision achieved the best BLEU-1 (5.34%) and cosine similarity (60.01%), while AraCLIP + Qwen-VL obtained the highest LLM-judge score (36.33%). This interpretable pipeline enables culturally coherent and contextually accurate Arabic captions. 

---
# Why mask diffusion does not work 

**Authors**: Haocheng Sun, Cynthia Xin Wen, Edward Hong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03289)  

**Abstract**: The main advantages of diffusion language models over autoregressive (AR) models lie in their ability to support parallel generation and bidirectional attention, enabling a more controllable generation process. In recent years, open-source mask diffusion language models have emerged, most of which are based on a variant known as absorbing diffusion. However, this paper demonstrates why mask diffusion faces inherent difficulties in achieving parallel generation and bidirectional attention. We also propose the most effective training and inference strategies for mask diffusion. 

---
# MACE: A Hybrid LLM Serving System with Colocated SLO-aware Continuous Retraining Alignment 

**Authors**: Yufei Li, Yu Fu, Yue Dong, Cong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03283)  

**Abstract**: Large language models (LLMs) deployed on edge servers are increasingly used in latency-sensitive applications such as personalized assistants, recommendation, and content moderation. However, the non-stationary nature of user data necessitates frequent retraining, which introduces a fundamental tension between inference latency and model accuracy under constrained GPU resources. Existing retraining strategies either delay model updates, over-commit resources to retraining, or overlook iteration-level retraining granularity. In this paper, we identify that iteration-level scheduling is crucial for adapting retraining frequency to model drift without violating service-level objectives (SLOs). We propose MACE, a hybrid LLM system that colocates concurrent inference (prefill, decode) and fine-tuning, with intelligent memory management to maximize task performance while promising inference throughput. MACE leverages the insight that not all model updates equally affect output alignment and allocates GPU cycles accordingly to balance throughput, latency, and update freshness. Our trace-driven evaluation shows that MACE matches or exceeds continuous retraining while reducing inference latency by up to 63% and maintaining throughput under resource constraints. Compared to periodic retraining, MACE improves latency breakdown across prefill, decode, and finetune stages, and sustains GPU utilization above 85% in NVIDIA AGX Orin. These results demonstrate that iteration-level hybrid scheduling is a promising direction for deploying LLMs with continual learning capabilities on edge platforms. 

---
# Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework 

**Authors**: Hao Gu, Vibhas Nair, Amrithaa Ashok Kumar, Jayvart Sharma, Ryan Lagasse  

**Link**: [PDF](https://arxiv.org/pdf/2510.03282)  

**Abstract**: Interpreting language models often involves circuit analysis, which aims to identify sparse subnetworks, or circuits, that accomplish specific tasks. Existing circuit discovery algorithms face a fundamental trade-off: attribution patching is fast but unfaithful to the full model, while edge pruning is faithful but computationally expensive. This research proposes a hybrid attribution and pruning (HAP) framework that uses attribution patching to identify a high-potential subgraph, then applies edge pruning to extract a faithful circuit from it. We show that HAP is 46\% faster than baseline algorithms without sacrificing circuit faithfulness. Furthermore, we present a case study on the Indirect Object Identification task, showing that our method preserves cooperative circuit components (e.g. S-inhibition heads) that attribution patching methods prune at high sparsity. Our results show that HAP could be an effective approach for improving the scalability of mechanistic interpretability research to larger models. Our code is available at this https URL. 

---
# Training Optimal Large Diffusion Language Models 

**Authors**: Jinjie Ni, Qian Liu, Chao Du, Longxu Dou, Hang Yan, Zili Wang, Tianyu Pang, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2510.03280)  

**Abstract**: We introduce Quokka, the first systematic scaling law for diffusion language models (DLMs), encompassing both compute-constrained and data-constrained regimes, and studying the key modeling and optimization designs. Quokka is a good friend of Chinchilla and provides wider scopes. We hope the results would bring short-term practical guidance in DLMs training and long-term inspirations for the whole AI community. 

---
# MemMamba: Rethinking Memory Patterns in State Space Model 

**Authors**: Youjin Wang, Yangjingyi Chen, Jiahao Yan, Jiaxuan Lu, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.03279)  

**Abstract**: With the explosive growth of data, long-sequence modeling has become increasingly important in tasks such as natural language processing and bioinformatics. However, existing methods face inherent trade-offs between efficiency and memory. Recurrent neural networks suffer from gradient vanishing and explosion, making them hard to scale. Transformers can model global dependencies but are constrained by quadratic complexity. Recently, selective state-space models such as Mamba have demonstrated high efficiency with O(n) time and O(1) recurrent inference, yet their long-range memory decays exponentially. In this work, we conduct mathematical derivations and information-theoretic analysis to systematically uncover the memory decay mechanism of Mamba, answering a fundamental question: what is the nature of Mamba's long-range memory and how does it retain information? To quantify key information loss, we further introduce horizontal-vertical memory fidelity metrics that capture degradation both within and across layers. Inspired by how humans distill and retain salient information when reading long documents, we propose MemMamba, a novel architectural framework that integrates state summarization mechanism together with cross-layer and cross-token attention, which alleviates long-range forgetting while preserving linear complexity. MemMamba achieves significant improvements over existing Mamba variants and Transformers on long-sequence benchmarks such as PG19 and Passkey Retrieval, while delivering a 48% speedup in inference efficiency. Both theoretical analysis and empirical results demonstrate that MemMamba achieves a breakthrough in the complexity-memory trade-off, offering a new paradigm for ultra-long sequence modeling. 

---
# General Exploratory Bonus for Optimistic Exploration in RLHF 

**Authors**: Wendi Li, Changdae Oh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.03269)  

**Abstract**: Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods to incentivize exploration often fail to realize optimism. We provide a theoretical analysis showing that current formulations, under KL or $\alpha$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the General Exploratory Bonus (GEB), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $\alpha$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF. 

---
# CAG: Chunked Augmented Generation for Google Chrome's Built-in Gemini Nano 

**Authors**: Vivek Vellaiyappan Surulimuthu, Aditya Karnam Gururaj Rao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18708)  

**Abstract**: We present Chunked Augmented Generation (CAG), an architecture specifically designed to overcome the context window limitations of Google Chrome's built-in Gemini Nano model. While Chrome's integration of Gemini Nano represents a significant advancement in bringing AI capabilities directly to the browser, its restricted context window poses challenges for processing large inputs. CAG addresses this limitation through intelligent input chunking and processing strategies, enabling efficient handling of extensive content while maintaining the model's performance within browser constraints. Our implementation demonstrates particular efficacy in processing large documents and datasets directly within Chrome, making sophisticated AI capabilities accessible through the browser without external API dependencies. Get started now at this https URL. 

---
