# Weight Tying Biases Token Embeddings Towards the Output Space 

**Authors**: Antonio Lopardo, Avyukth Harish, Catherine Arnett, Akshat Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2603.26663)  

**Abstract**: Weight tying, i.e. sharing parameters between input and output embedding matrices, is common practice in language model design, yet its impact on the learned embedding space remains poorly understood. In this paper, we show that tied embedding matrices align more closely with output (unembedding) matrices than with input embeddings of comparable untied models, indicating that the shared matrix is shaped primarily for output prediction rather than input representation. This unembedding bias arises because output gradients dominate early in training. Using tuned lens analysis, we show this negatively affects early-layer computations, which contribute less effectively to the residual stream. Scaling input gradients during training reduces this bias, providing causal evidence for the role of gradient imbalance. This is mechanistic evidence that weight tying optimizes the embedding matrix for output prediction, compromising its role in input representation. These results help explain why weight tying can harm performance at scale and have implications for training smaller LLMs, where the embedding matrix contributes substantially to total parameter count. 

---
# EnTaCs: Analyzing the Relationship Between Sentiment and Language Choice in English-Tamil Code-Switching 

**Authors**: Paul Bontempo  

**Link**: [PDF](https://arxiv.org/pdf/2603.26587)  

**Abstract**: This paper investigates the relationship between utterance sentiment and language choice in English-Tamil code-switched text, using methods from machine learning and statistical modelling. We apply a fine-tuned XLM-RoBERTa model for token-level language identification on 35,650 romanized YouTube comments from the DravidianCodeMix dataset, producing per-utterance measurements of English proportion and language switch frequency. Linear regression analysis reveals that positive utterances exhibit significantly greater English proportion (34.3%) than negative utterances (24.8%), and mixed-sentiment utterances show the highest language switch frequency when controlling for utterance length. These findings support the hypothesis that emotional content demonstrably influences language choice in multilingual code-switching settings, due to socio-linguistic associations of prestige and identity with embedded and matrix languages. 

---
# MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference 

**Authors**: Joris Köster, Zixuan Liu, Siavash Khajavi, Zizhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.26557)  

**Abstract**: Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessions. In this work, we propose MemBoost, a memory-boosted LLM serving framework that enables a lightweight model to reuse previously generated answers and retrieve relevant supporting information for cheap inference, while selectively escalating difficult or uncertain queries to a stronger model. Unlike standard retrieval-augmented generation, which primarily grounds a single response, MemBoost is designed for interactive settings by supporting answer reuse, continual memory growth, and cost-aware routing. Experiments across multiple models under simulated workloads show that MemBoost substantially reduces expensive large-model invocations and overall inference cost, while maintaining high answer quality comparable to the strong model baseline. 

---
# When Perplexity Lies: Generation-Focused Distillation of Hybrid Sequence Models 

**Authors**: Juan Gabriel Kostelec, Xiang Wang, Axel Laborieux, Christos Sourmpis, Qinghai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2603.26556)  

**Abstract**: Converting a pretrained Transformer into a more efficient hybrid model through distillation offers a promising approach to reducing inference costs. However, achieving high-quality generation in distilled models requires careful joint design of both the student architecture and the distillation process. Many prior distillation works evaluate downstream multiple-choice benchmarks by ranking candidate answers with log-likelihood rather than requiring autoregressive generation, which can obscure important differences in model quality. For example, we show that a 7B parameter distilled model that nearly matches its teacher to within 0.2\,pp under log-likelihood scoring actually falls behind by 20.8\,pp when the model must generate answers autoregressively.
We propose a Hybrid Kimi Delta Attention (Hybrid-KDA) architecture paired with GenDistill, a multi-stage distillation pipeline, and use generation-based evaluation throughout to guide design decisions. Applying this approach to Qwen3-0.6B, we systematically ablate six design axes: training objective, loss masking, training duration, dataset selection, parameter freezing, and architecture choice. We find that log-likelihood-based evaluation consistently underestimates the gap between teacher and student, and can in some cases reverse the ranking of design choices, meaning that conclusions drawn from perplexity-only evaluation may be misleading. Among the factors we study, dataset selection, completion-only masking, and freezing attention layers during post-training have the largest impact on generation quality.
Our best Hybrid-KDA model retains 86--90\% of teacher accuracy on knowledge benchmarks while reducing KV cache memory by up to 75\% and improving time-to-first-token by 2--4$\times$ at 128K-token contexts. 

---
# Development of a European Union Time-Indexed Reference Dataset for Assessing the Performance of Signal Detection Methods in Pharmacovigilance using a Large Language Model 

**Authors**: Maria Kefala, Jeffery L. Painter, Syed Tauhid Bukhari, Maurizio Sessa  

**Link**: [PDF](https://arxiv.org/pdf/2603.26544)  

**Abstract**: Background: The identification of optimal signal detection methods is hindered by the lack of reliable reference datasets. Existing datasets do not capture when adverse events (AEs) are officially recognized by regulatory authorities, preventing restriction of analyses to pre-confirmation periods and limiting evaluation of early detection performance. This study addresses this gap by developing a time-indexed reference dataset for the European Union (EU), incorporating the timing of AE inclusion in product labels along with regulatory metadata. Methods: Current and historical Summaries of Product Characteristics (SmPCs) for all centrally authorized products (n=1,513) were retrieved from the EU Union Register of Medicinal Products (data lock: 15 December 2025). Section 4.8 was extracted and processed using DeepSeek V3 to identify AEs. Regulatory metadata, including labelling changes, were programmatically extracted. Time indexing was based on the date of AE inclusion in the SmPC. Results: The database includes 17,763 SmPC versions spanning 1995-2025, comprising 125,026 drug-AE associations. The time-indexed reference dataset, restricted to active products, included 1,479 medicinal products and 110,823 drug-AE associations. Most AEs were identified pre-marketing (74.5%) versus post-marketing (25.5%). Safety updates peaked around 2012. Gastrointestinal, skin, and nervous system disorders were the most represented System Organ Classes. Drugs had a median of 48 AEs across 14 SOCs. Conclusions: The proposed dataset addresses a critical gap in pharmacovigilance by incorporating temporal information on AE recognition for the EU, supporting more accurate assessment of signal detection performance and facilitating methodological comparisons across analytical approaches. 

---
# How Open Must Language Models be to Enable Reliable Scientific Inference? 

**Authors**: James A. Michaelov, Catherine Arnett, Tyler A. Chang, Pamela D. Rivière, Samuel M. Taylor, Cameron R. Jones, Sean Trott, Roger P. Levy, Benjamin K. Bergen, Micah Altman  

**Link**: [PDF](https://arxiv.org/pdf/2603.26539)  

**Abstract**: How does the extent to which a model is open or closed impact the scientific inferences that can be drawn from research that involves it? In this paper, we analyze how restrictions on information about model construction and deployment threaten reliable inference. We argue that current closed models are generally ill-suited for scientific purposes, with some notable exceptions, and discuss ways in which the issues they present to reliable inference can be resolved or mitigated. We recommend that when models are used in research, potential threats to inference should be systematically identified along with the steps taken to mitigate them, and that specific justifications for model selection should be provided. 

---
# ALBA: A European Portuguese Benchmark for Evaluating Language and Linguistic Dimensions in Generative LLMs 

**Authors**: Inês Vieira, Inês Calvo, Iago Paulo, James Furtado, Rafael Ferreira, Diogo Tavares, Diogo Glória-Silva, David Semedo, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2603.26516)  

**Abstract**: As Large Language Models (LLMs) expand across multilingual domains, evaluating their performance in under-represented languages becomes increasingly important. European Portuguese (pt-PT) is particularly affected, as existing training data and benchmarks are mainly in Brazilian Portuguese (pt-BR). To address this, we introduce ALBA, a linguistically grounded benchmark designed from the ground up to assess LLM proficiency in linguistic-related tasks in pt-PT across eight linguistic dimensions, including Language Variety, Culture-bound Semantics, Discourse Analysis, Word Plays, Syntax, Morphology, Lexicology, and Phonetics and Phonology. ALBA is manually constructed by language experts and paired with an LLM-as-a-judge framework for scalable evaluation of pt-PT generated language. Experiments on a diverse set of models reveal performance variability across linguistic dimensions, highlighting the need for comprehensive, variety-sensitive benchmarks that support further development of tools in pt-PT. 

---
# JAL-Turn: Joint Acoustic-Linguistic Modeling for Real-Time and Robust Turn-Taking Detection in Full-Duplex Spoken Dialogue Systems 

**Authors**: Guangzhao Yang, Yu Pan, Shi Qiu, Ningjie Bai  

**Link**: [PDF](https://arxiv.org/pdf/2603.26515)  

**Abstract**: Despite recent advances, efficient and robust turn-taking detection remains a significant challenge in industrial-grade Voice AI agent deployments. Many existing systems rely solely on acoustic or semantic cues, leading to suboptimal accuracy and stability, while recent attempts to endow large language models with full-duplex capabilities require costly full-duplex data and incur substantial training and deployment overheads, limiting real-time performance. In this paper, we propose JAL-Turn, a lightweight and efficient speech-only turn-taking framework that adopts a joint acoustic-linguistic modeling paradigm, in which a cross-attention module adaptively integrates pre-trained acoustic representations with linguistic features to support low-latency prediction of hold vs shift states. By sharing a frozen ASR encoder, JAL-Turn enables turn-taking prediction to run fully in parallel with speech recognition, introducing no additional end-to-end latency or computational overhead. In addition, we introduce a scalable data construction pipeline that automatically derives reliable turn-taking labels from large-scale real-world dialogue corpora. Extensive experiments on public multilingual benchmarks and an in-house Japanese customer-service dataset show that JAL-Turn consistently outperforms strong state-of-the-art baselines in detection accuracy while maintaining superior real-time performance. 

---
# AMALIA Technical Report: A Fully Open Source Large Language Model for European Portuguese 

**Authors**: Afonso Simplício, Gonçalo Vinagre, Miguel Moura Ramos, Diogo Tavares, Rafael Ferreira, Giuseppe Attanasio, Duarte M. Alves, Inês Calvo, Inês Vieira, Rui Guerra, James Furtado, Beatriz Canaverde, Iago Paulo, Vasco Ramos, Diogo Glória-Silva, Miguel Faria, Marcos Treviso, Daniel Gomes, Pedro Gomes, David Semedo, André Martins, João Magalhães  

**Link**: [PDF](https://arxiv.org/pdf/2603.26511)  

**Abstract**: Despite rapid progress in open large language models (LLMs), European Portuguese (pt-PT) remains underrepresented in both training data and native evaluation, with machine-translated benchmarks likely missing the variant's linguistic and cultural nuances. We introduce AMALIA, a fully open LLM that prioritizes pt-PT by using more high-quality pt-PT data during both the mid- and post-training stages. To evaluate pt-PT more faithfully, we release a suite of pt-PT benchmarks that includes translated standard tasks and four new datasets targeting pt-PT generation, linguistic competence, and pt-PT/pt-BR bias. Experiments show that AMALIA matches strong baselines on translated benchmarks while substantially improving performance on pt-PT-specific evaluations, supporting the case for targeted training and native benchmarking for European Portuguese. 

---
# Clinical named entity recognition in the Portuguese language: a benchmark of modern BERT models and LLMs 

**Authors**: Vinicius Anjos de Almeida, Sandro Saorin da Silva, Josimar Chire, Leonardo Vicenzi, Nícolas Henrique Borges, Helena Kociolek, Sarah Miriã de Castro Rocha, Frederico Nassif Gomes, Júlia Cristina Ferreira, Oge Marques, Lucas Emanuel Silva e Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2603.26510)  

**Abstract**: Clinical notes contain valuable unstructured information. Named entity recognition (NER) enables the automatic extraction of medical concepts; however, benchmarks for Portuguese remain scarce. In this study, we aimed to evaluate BERT-based models and large language models (LLMs) for clinical NER in Portuguese and to test strategies for addressing multilabel imbalance. We compared BioBERTpt, BERTimbau, ModernBERT, and mmBERT with LLMs such as GPT-5 and Gemini-2.5, using the public SemClinBr corpus and a private breast cancer dataset. Models were trained under identical conditions and evaluated using precision, recall, and F1-score. Iterative stratification, weighted loss, and oversampling were explored to mitigate class imbalance. The mmBERT-base model achieved the best performance (micro F1 = 0.76), outperforming all other models. Iterative stratification improved class balance and overall performance. Multilingual BERT models, particularly mmBERT, perform strongly for Portuguese clinical NER and can run locally with limited computational resources. Balanced data-splitting strategies further enhance performance. 

---
# ClimateCheck 2026: Scientific Fact-Checking and Disinformation Narrative Classification of Climate-related Claims 

**Authors**: Raia Abu Ahmad, Max Upravitelev, Aida Usmanova, Veronika Solopova, Georg Rehm  

**Link**: [PDF](https://arxiv.org/pdf/2603.26449)  

**Abstract**: Automatically verifying climate-related claims against scientific literature is a challenging task, complicated by the specialised nature of scholarly evidence and the diversity of rhetorical strategies underlying climate disinformation. ClimateCheck 2026 is the second iteration of a shared task addressing this challenge, expanding on the 2025 edition with tripled training data and a new disinformation narrative classification task. Running from January to February 2026 on the CodaBench platform, the competition attracted 20 registered participants and 8 leaderboard submissions, with systems combining dense retrieval pipelines, cross-encoder ensembles, and large language models with structured hierarchical reasoning. In addition to standard evaluation metrics (Recall@K and Binary Preference), we adapt an automated framework to assess retrieval quality under incomplete annotations, exposing systematic biases in how conventional metrics rank systems. A cross-task analysis further reveals that not all climate disinformation is equally verifiable, potentially implicating how future fact-checking systems should be designed. 

---
# Automating Clinical Information Retrieval from Finnish Electronic Health Records Using Large Language Models 

**Authors**: Mikko Saukkoriipi, Nicole Hernandez, Jaakko Sahlsten, Kimmo Kaski, Otso Arponen  

**Link**: [PDF](https://arxiv.org/pdf/2603.26434)  

**Abstract**: Clinicians often need to retrieve patient-specific information from electronic health records (EHRs), a task that is time-consuming and error-prone. We present a locally deployable Clinical Contextual Question Answering (CCQA) framework that answers clinical questions directly from EHRs without external data transfer. Open-source large language models (LLMs) ranging from 4B to 70B parameters were benchmarked under fully offline conditions using 1,664 expert-annotated question-answer pairs derived from records of 183 patients. The dataset consisted predominantly of Finnish clinical text. In free-text generation, Llama-3.1-70B achieved 95.3% accuracy and 97.3% consistency across semantically equivalent question variants, while the smaller Qwen3-30B-A3B-2507 model achieved comparable performance. In a multiple-choice setting, models showed similar accuracy but variable calibration. Low-precision quantization (4-bit and 8-bit) preserved predictive performance while reducing GPU memory requirements and improving deployment feasibility. Clinical evaluation identified clinically significant errors in 2.9% of outputs, and semantically equivalent questions occasionally yielded discordant responses, including instances where one formulation was correct and the other contained a clinically significant error (0.96% of cases). These findings demonstrate that locally hosted open-source LLMs can accurately retrieve patient-specific information from EHRs using natural-language queries, while highlighting the need for validation and human oversight in clinical deployment. 

---
# Analysing Calls to Order in German Parliamentary Debates 

**Authors**: Nina Smirnova, Daniel Dan, Philipp Mayr  

**Link**: [PDF](https://arxiv.org/pdf/2603.26430)  

**Abstract**: Parliamentary debate constitutes a central arena of political power, shaping legislative outcomes and public discourse. Incivility within this arena signals political polarization and institutional conflict. This study presents a systematic investigation of incivility in the German Bundestag by examining calls to order (CtO; plural: CtOs) as formal indicators of norm violations. Despite their relevance, CtOs have received little systematic attention in parliamentary research. We introduce a rule-based method for detecting and annotating CtOs in parliamentary speeches and present a novel dataset of German parliamentary debates spanning 72 years that includes annotated CtO instances. Additionally, we develop the first classification system for CtO triggers and analyze the factors associated with their occurrence. Our findings show that, despite formal regulations, the issuance of CtOs is partly subjective and influenced by session presidents and parliamentary dynamics, with certain individuals disproportionately affected. An insult towards individuals is the most frequent cause of CtO. In general, male members and those belonging to opposition parties receive more calls to order than their female and coalition-party counterparts. Most CtO triggers were detected in speeches dedicated to governmental affairs and actions of the presidency. The CtO triggers dataset is available at: this https URL. 

---
# Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models 

**Authors**: Richard J. Young  

**Link**: [PDF](https://arxiv.org/pdf/2603.26410)  

**Abstract**: Extended-thinking models expose a second text-generation channel ("thinking tokens") alongside the user-visible answer. This study examines 12 open-weight reasoning models on MMLU and GPQA questions paired with misleading hints. Among the 10,506 cases where models actually followed the hint (choosing the hint's target over the ground truth), each case is classified by whether the model acknowledges the hint in its thinking tokens, its answer text, both, or neither. In 55.4% of these cases the model's thinking tokens contain hint-related keywords that the visible answer omits entirely, a pattern termed *thinking-answer divergence*. The reverse (answer-only acknowledgment) is near-zero (0.5%), confirming that the asymmetry is directional. Hint type shapes the pattern sharply: sycophancy is the most *transparent* hint, with 58.8% of sycophancy-influenced cases acknowledging the professor's authority in both channels, while consistency (72.2%) and unethical (62.7%) hints are dominated by thinking-only acknowledgment. Models also vary widely, from near-total divergence (Step-3.5-Flash: 94.7%) to relative transparency (Qwen3.5-27B: 19.6%). These results show that answer-text-only monitoring misses more than half of all hint-influenced reasoning and that thinking-token access, while necessary, still leaves 11.8% of cases with no verbalized acknowledgment in either channel. 

---
# Word Alignment-Based Evaluation of Uniform Meaning Representations 

**Authors**: Daniel Zeman, Federica Gamba  

**Link**: [PDF](https://arxiv.org/pdf/2603.26401)  

**Abstract**: Comparison and evaluation of graph-based representations of sentence meaning is a challenge because competing representations of the same sentence may have different number of nodes, and it is not obvious which nodes should be compared to each other. Existing approaches favor node mapping that maximizes $F_1$ score over node relations and attributes, regardless whether the similarity is intentional or accidental; consequently, the identified mismatches in values of node attributes are not useful for any detailed error analysis. We propose a node-matching algorithm that allows comparison of multiple Uniform Meaning Representations (UMR) of one sentence and that takes advantage of node-word alignments, inherently available in UMR. We compare it with previously used approaches, in particular smatch (the de-facto standard in AMR evaluation), and argue that sensitivity to word alignment makes the comparison of meaning representations more intuitive and interpretable, while avoiding the NP-hard search problem inherent in smatch. A script implementing the method is freely available. 

---
# Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers 

**Authors**: Yusheng Zhao, Hourun Li, Bohan Wu, Jingyang Yuan, Meng Zhang, Yichun Yin, Lifeng Shang, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26380)  

**Abstract**: The attention mechanism has been the core component in modern transformer architectures. However, the computation of standard full attention scales quadratically with the sequence length, serving as a major bottleneck in long-context language modeling. Sliding window attention restricts the context length for better efficiency at the cost of narrower receptive fields. While existing efforts attempt to take the benefits from both sides by building hybrid models, they often resort to static, heuristically designed alternating patterns that limit efficient allocation of computation in various scenarios. In this paper, we propose Switch Attention (SwiAttn), a novel hybrid transformer that enables dynamic and fine-grained routing between full attention and sliding window attention. For each token at each transformer layer, SwiAttn dynamically routes the computation to either a full-attention branch for global information aggregation or a sliding-window branch for efficient local pattern matching. An adaptive regularization objective is designed to encourage the model towards efficiency. Moreover, we adopt continual pretraining to optimize the model, transferring the full attention architecture to the hybrid one. Extensive experiments are conducted on twenty-three benchmark datasets across both regular (4K) and long (32K) context lengths, demonstrating the effectiveness of the proposed method. 

---
# CALRK-Bench: Evaluating Context-Aware Legal Reasoning in Korean Law 

**Authors**: JiHyeok Jung, TaeYoung Yoon, HyunSouk Cho  

**Link**: [PDF](https://arxiv.org/pdf/2603.26332)  

**Abstract**: Legal reasoning requires not only the application of legal rules but also an understanding of the context in which those rules operate. However, existing legal benchmarks primarily evaluate rule application under the assumption of fixed norms, and thus fail to capture situations where legal judgments shift or where multiple norms interact. In this work, we propose CALRK-Bench, a context-aware legal reasoning benchmark based on the legal system in Korean. CALRK-Bench evaluates whether models can identify the temporal validity of legal norms, determine whether sufficient legal information is available for a given case, and understand the reasons behind shifts in legal judgments. The dataset is constructed from legal precedents and legal consultation records, and is validated by legal experts. Experimental results show that even recent large language models consistently exhibit low performance on these three tasks. CALRK-Bench provides a new stress test for evaluating context-aware legal reasoning rather than simple memorization of legal knowledge. Our code is available at this https URL. 

---
# From Human Cognition to Neural Activations: Probing the Computational Primitives of Spatial Reasoning in LLMs 

**Authors**: Jiyuan An, Liner Yang, Mengyan Wang, Luming Lu, Weihua An, Erhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26323)  

**Abstract**: As spatial intelligence becomes an increasingly important capability for foundation models, it remains unclear whether large language models' (LLMs) performance on spatial reasoning benchmarks reflects structured internal spatial representations or reliance on linguistic heuristics. We address this question from a mechanistic perspective by examining how spatial information is internally represented and used. Drawing on computational theories of human spatial cognition, we decompose spatial reasoning into three primitives, relational composition, representational transformation, and stateful spatial updating, and design controlled task families for each. We evaluate multilingual LLMs in English, Chinese, and Arabic under single pass inference, and analyze internal representations using linear probing, sparse autoencoder based feature analysis, and causal interventions. We find that task relevant spatial information is encoded in intermediate layers and can causally influence behavior, but these representations are transient, fragmented across task families, and weakly integrated into final predictions. Cross linguistic analysis further reveals mechanistic degeneracy, where similar behavioral performance arises from distinct internal pathways. Overall, our results suggest that current LLMs exhibit limited and context dependent spatial representations rather than robust, general purpose spatial reasoning, highlighting the need for mechanistic evaluation beyond benchmark accuracy. 

---
# findsylls: A Language-Agnostic Toolkit for Syllable-Level Speech Tokenization and Embedding 

**Authors**: Héctor Javier Vázquez Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2603.26292)  

**Abstract**: Syllable-level units offer compact and linguistically meaningful representations for spoken language modeling and unsupervised word discovery, but research on syllabification remains fragmented across disparate implementations, datasets, and evaluation protocols. We introduce findsylls, a modular, language-agnostic toolkit that unifies classical syllable detectors and end-to-end syllabifiers under a common interface for syllable segmentation, embedding extraction, and multi-granular evaluation. The toolkit implements and standardizes widely used methods (e.g., Sylber, VG-HuBERT) and allows their components to be recombined, enabling controlled comparisons of representations, algorithms, and token rates. We demonstrate findsylls on English and Spanish corpora and on new hand-annotated data from Kono, an underdocumented Central Mande language, illustrating how a single framework can support reproducible syllable-level experiments across both high-resource and under-resourced settings. 

---
# SocialX: A Modular Platform for Multi-Source Big Data Research in Indonesia 

**Authors**: Muhammad Apriandito Arya Saputra, Andry Alamsyah, Dian Puteri Ramadhani, Thomhert Suprapto Siadari, Hanif Fakhrurroja  

**Link**: [PDF](https://arxiv.org/pdf/2603.26253)  

**Abstract**: Big data research in Indonesia is constrained by a fundamental fragmentation: relevant data is scattered across social media, news portals, e-commerce platforms, review sites, and academic databases, each with different formats, access methods, and noise characteristics. Researchers must independently build collection pipelines, clean heterogeneous data, and assemble separate analysis tools, a process that often overshadows the research itself. We present SocialX, a modular platform for multi-source big data research that integrates heterogeneous data collection, language-aware preprocessing, and pluggable analysis into a unified, source-agnostic pipeline. The platform separates concerns into three independent layers (collection, preprocessing, and analysis) connected by a lightweight job-coordination mechanism. This modularity allows each layer to grow independently: new data sources, preprocessing methods, or analysis tools can be added without modifying the existing pipeline. We describe the design principles that enable this extensibility, detail the preprocessing methodology that addresses challenges specific to Indonesian text across registers, and demonstrate the platform's utility through a walkthrough of a typical research workflow. SocialX is publicly accessible as a web-based platform at this https URL. 

---
# Automatic Speech Recognition for Documenting Endangered Languages: Case Study of Ikema Miyakoan 

**Authors**: Chihiro Taguchi, Yukinori Takubo, David Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26248)  

**Abstract**: Language endangerment poses a major challenge to linguistic diversity worldwide, and technological advances have opened new avenues for documentation and revitalization. Among these, automatic speech recognition (ASR) has shown increasing potential to assist in the transcription of endangered language data. This study focuses on Ikema, a severely endangered Ryukyuan language spoken in Okinawa, Japan, with approximately 1,300 remaining speakers, most of whom are over 60 years old. We present an ongoing effort to develop an ASR system for Ikema based on field recordings. Specifically, we (1) construct a {\totaldatasethours}-hour speech corpus from field recordings, (2) train an ASR model that achieves a character error rate as low as 15\%, and (3) evaluate the impact of ASR assistance on the efficiency of speech transcription. Our results demonstrate that ASR integration can substantially reduce transcription time and cognitive load, offering a practical pathway toward scalable, technology-supported documentation of endangered languages. 

---
# Distilling Conversations: Abstract Compression of Conversational Audio Context for LLM-based ASR 

**Authors**: Shashi Kumar, Esaú Villatoro-Tello, Sergio Burdisso, Kadri Hacioglu, Thibault Bañeras-Roux, Hasindri Watawana, Dairazalia Sanchez-Cortes, Srikanth Madikeri, Petr Motlicek, Andreas Stolcke  

**Link**: [PDF](https://arxiv.org/pdf/2603.26246)  

**Abstract**: Standard LLM-based speech recognition systems typically process utterances in isolation, limiting their ability to leverage conversational context. In this work, we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio token sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, which replaces the audio portion of prior turns with a fixed number of learned latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compressed model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footprint. We also provide targeted analyses of the compression setup and its trade-offs. 

---
# A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs 

**Authors**: Uri Z. Kialy, Avi Shtarkberg, Ayal Klein  

**Link**: [PDF](https://arxiv.org/pdf/2603.26236)  

**Abstract**: While multilingual language models successfully transfer factual and syntactic knowledge across languages, it remains unclear whether they process culture-specific pragmatic registers, such as slang, as isolated language-specific memorizations or as unified, abstract concepts. We study this by probing the internal representations of Gemma-2-9B-IT using Sparse Autoencoders (SAEs) across three typologically diverse source languages: English, Hebrew, and Russian. To definitively isolate pragmatic register processing from trivial lexical sensitivity, we introduce a novel dataset in which every target term is polysemous, appearing in both literal and informal contexts. We find that while much of the informal-register signal is distributed across language-specific features, a small but highly robust cross-linguistic core consistently emerges. This shared core forms a geometrically coherent ``informal register subspace'' that sharpens in the model's deeper layers. Crucially, these shared representations are not merely correlational: activation steering with these features causally shifts output formality across all source languages and transfers zero-shot to six unseen languages spanning diverse language families and scripts. Together, these results provide the first mechanistic evidence that multilingual LLMs internalize informal register not just as surface-level heuristics, but as a portable, language-agnostic pragmatic abstraction. 

---
# GS-BrainText: A Multi-Site Brain Imaging Report Dataset from Generation Scotland for Clinical Natural Language Processing Development and Validation 

**Authors**: Beatrice Alex, Claire Grover, Arlene Casey, Richard Tobin, Heather Whalley, William Whiteley  

**Link**: [PDF](https://arxiv.org/pdf/2603.26235)  

**Abstract**: We present GS-BrainText, a curated dataset of 8,511 brain radiology reports from the Generation Scotland cohort, of which 2,431 are annotated for 24 brain disease phenotypes. This multi-site dataset spans five Scottish NHS health boards and includes broad age representation (mean age 58, median age 53), making it uniquely valuable for developing and evaluating generalisable clinical natural language processing (NLP) algorithms and tools. Expert annotations were performed by a multidisciplinary clinical team using an annotation schema, with 10-100% double annotation per NHS health board and rigorous quality assurance. Benchmark evaluation using EdIE-R, an existing rule-based NLP system developed in conjunction with the annotation schema, revealed some performance variation across health boards (F1: 86.13-98.13), phenotypes (F1: 22.22-100) and age groups (F1: 87.01-98.13), highlighting critical challenges in generalisation of NLP tools. The GS-BrainText dataset addresses a significant gap in available UK clinical text resources and provides a valuable resource for the study of linguistic variation, diagnostic uncertainty expression and the impact of data characteristics on NLP system performance. 

---
# Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents 

**Authors**: Nicholas Edwards, Sebastian Schuster  

**Link**: [PDF](https://arxiv.org/pdf/2603.26233)  

**Abstract**: As Large Language Model (LLM) agents are increasingly deployed in open-ended domains like software engineering, they frequently encounter underspecified instructions that lack crucial context. While human developers naturally resolve underspecification by asking clarifying questions, current agents are largely optimized for autonomous execution. In this work, we systematically evaluate the clarification-seeking abilities of LLM agents on an underspecified variant of SWE-bench Verified. We propose an uncertainty-aware multi-agent scaffold that explicitly decouples underspecification detection from code execution. Our results demonstrate that this multi-agent system using OpenHands + Claude Sonnet 4.5 achieves a 69.40% task resolve rate, significantly outperforming a standard single-agent setup (61.20%) and closing the performance gap with agents operating on fully specified instructions. Furthermore, we find that the multi-agent system exhibits well-calibrated uncertainty, conserving queries on simple tasks while proactively seeking information on more complex issues. These findings indicate that current models can be turned into proactive collaborators, where agents independently recognize when to ask questions to elicit missing information in real-world, underspecified tasks. 

---
# Sparse Auto-Encoders and Holism about Large Language Models 

**Authors**: Jumbly Grindrod  

**Link**: [PDF](https://arxiv.org/pdf/2603.26207)  

**Abstract**: Does Large Language Model (LLM) technology suggest a meta-semantic picture i.e. a picture of how words and complex expressions come to have the meaning that they do? One modest approach explores the assumptions that seem to be built into how LLMs capture the meanings of linguistic expressions as a way of considering their plausibility (Grindrod, 2026a, 2026b). It has previously been argued that LLMs, in employing a form of distributional semantics, adopt a form of holism about meaning (Grindrod, 2023; Grindrod et al., forthcoming). However, recent work in mechanistic interpretability presents a challenge to these arguments. Specifically, the discovery of a vast array of interpretable latent features within the high dimensional spaces used by LLMs potentially challenges the holistic interpretation. In this paper, I will present the original reasons for thinking that LLMs embody a form of holism (section 1), before introducing recent work on features generated through sparse auto-encoders, and explaining how the discovery of such features suggests an alternative decompositional picture of meaning (section 2). I will then respond to this challenge by considering in greater detail the nature of such features (section 3). Finally, I will return to the holistic picture defended by Grindrod et al. and argue that the picture still stands provided that the features are countable (section 4). 

---
# ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory 

**Authors**: Zhuohan Ge, Haoyang Li, Yubo Wang, Nicole Hu, Chen Jason Zhang, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.26182)  

**Abstract**: While Large Language Models (LLMs) have demonstrated potential in healthcare, they often struggle with the complex, non-linear reasoning required for accurate clinical diagnosis. Existing methods typically rely on static, linear mappings from symptoms to diagnoses, failing to capture the iterative, hypothesis-driven reasoning inherent to human clinicians. To bridge this gap, we introduce ClinicalAgents, a novel multi-agent framework designed to simulate the cognitive workflow of expert clinicians. Unlike rigid sequential chains, ClinicalAgents employs a dynamic orchestration mechanism modeled as a Monte Carlo Tree Search (MCTS) process. This allows an Orchestrator to iteratively generate hypotheses, actively verify evidence, and trigger backtracking when critical information is missing. Central to this framework is a Dual-Memory architecture: a mutable Working Memory that maintains the evolving patient state for context-aware reasoning, and a static Experience Memory that retrieves clinical guidelines and historical cases via an active feedback loop. Extensive experiments demonstrate that ClinicalAgents achieves state-of-the-art performance, significantly enhancing both diagnostic accuracy and explainability compared to strong single-agent and multi-agent baselines. 

---
# Clash of the models: Comparing performance of BERT-based variants for generic news frame detection 

**Authors**: Vihang Jumle  

**Link**: [PDF](https://arxiv.org/pdf/2603.26156)  

**Abstract**: Framing continues to remain one of the most extensively applied theories in political communication. Developments in computation, particularly with the introduction of transformer architecture and more so with large language models (LLMs), have naturally prompted scholars to explore various novel computational approaches, especially for deductive frame detection, in recent years. While many studies have shown that different transformer models outperform their preceding models that use bag-of-words features, the debate continues to evolve regarding how these models compare with each other on classification tasks. By placing itself at this juncture, this study makes three key contributions: First, it comparatively performs generic news frame detection and compares the performance of five BERT-based variants (BERT, RoBERTa, DeBERTa, DistilBERT and ALBERT) to add to the debate on best practices around employing computational text analysis for political communication studies. Second, it introduces various fine-tuned models capable of robustly performing generic news frame detection. Third, building upon numerous previous studies that work with US-centric data, this study provides the scholarly community with a labelled generic news frames dataset based on the Swiss electoral context that aids in testing the contextual robustness of these computational approaches to framing analysis. 

---
# LLM Benchmark-User Need Misalignment for Climate Change 

**Authors**: Oucheng Liu, Lexing Xie, Jing Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26106)  

**Abstract**: Climate change is a major socio-scientific issue shapes public decision-making and policy discussions. As large language models (LLMs) increasingly serve as an interface for accessing climate knowledge, whether existing benchmarks reflect user needs is critical for evaluating LLM in real-world settings. We propose a Proactive Knowledge Behaviors Framework that captures the different human-human and human-AI knowledge seeking and provision behaviors. We further develop a Topic-Intent-Form taxonomy and apply it to analyze climate-related data representing different knowledge behaviors. Our results reveal a substantial mismatch between current benchmarks and real-world user needs, while knowledge interaction patterns between humans and LLMs closely resemble those in human-human interactions. These findings provide actionable guidance for benchmark design, RAG system development, and LLM training. Code is available at this https URL. 

---
# IndoBERT-Relevancy: A Context-Conditioned Relevancy Classifier for Indonesian Text 

**Authors**: Muhammad Apriandito Arya Saputra, Andry Alamsyah, Dian Puteri Ramadhani, Thomhert Suprapto Siadari, Hanif Fakhrurroja  

**Link**: [PDF](https://arxiv.org/pdf/2603.26095)  

**Abstract**: Determining whether a piece of text is relevant to a given topic is a fundamental task in natural language processing, yet it remains largely unexplored for Bahasa Indonesia. Unlike sentiment analysis or named entity recognition, relevancy classification requires the model to reason about the relationship between two inputs simultaneously: a topical context and a candidate text. We introduce IndoBERT-Relevancy, a context-conditioned relevancy classifier built on IndoBERT Large (335M parameters) and trained on a novel dataset of 31,360 labeled pairs spanning 188 topics. Through an iterative, failure-driven data construction process, we demonstrate that no single data source is sufficient for robust relevancy classification, and that targeted synthetic data can effectively address specific model weaknesses. Our final model achieves an F1 score of 0.948 and an accuracy of 96.5%, handling both formal and informal Indonesian text. The model is publicly available at HuggingFace. 

---
# I Want to Believe (but the Vocabulary Changed): Measuring the Semantic Structure and Evolution of Conspiracy Theories 

**Authors**: Manisha Keim, Sarmad Chandio, Osama Khalid, Rishab Nithyanand  

**Link**: [PDF](https://arxiv.org/pdf/2603.26062)  

**Abstract**: Research on conspiracy theories has largely focused on belief formation, exposure, and diffusion, while paying less attention to how their meanings change over time. This gap persists partly because conspiracy-related terms are often treated as stable lexical markers, making it difficult to separate genuine semantic changes from surface-level vocabulary changes. In this paper, we measure the semantic structure and evolution of conspiracy theories in online political discourse. Using 169.9M comments from Reddit's r/politics subreddit spanning 2012--2022, we first demonstrate that conspiracy-related language forms coherent and semantically distinguishable regions of language space, allowing conspiracy theories to be treated as semantic objects. We then track how these objects evolve over time using aligned word embeddings, enabling comparisons of semantic neighborhoods across periods. Our analysis reveals that conspiracy theories evolve non-uniformly, exhibiting patterns of semantic stability, expansion, contraction, and replacement that are not captured by keyword-based approaches alone. 

---
# Retrieval-Augmented Generation Based Nurse Observation Extraction 

**Authors**: Kyomin Hwang, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2603.26046)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have played a significant role in reducing human workload across various domains, a trend that is increasingly extending into the medical field. In this paper, we propose an automated pipeline designed to alleviate the burden on nurses by automatically extracting clinical observations from nurse dictations. To ensure accurate extraction, we introduce a method based on Retrieval-Augmented Generation (RAG). Our approach demonstrates effective performance, achieving an F1-score of 0.796 on the MEDIQA-SYNUR test dataset. 

---
# AgentCollab: A Self-Evaluation-Driven Collaboration Paradigm for Efficient LLM Agents 

**Authors**: Wenbo Gao, Renxi Liu, Xian Wang, Fang Guo, Shuai Yang, Xi Chen, Hui-Ling Zhen, Hanting Chen, Weizhe Lin, Xiaosong Li, Yaoyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26034)  

**Abstract**: Autonomous agents powered by large language models (LLMs) perform complex tasks through long-horizon reasoning and tool interaction, where a fundamental trade-off arises between execution efficiency and reasoning robustness. Models at different capability-cost levels offer complementary advantages: lower-cost models enable fast execution but may struggle on difficult reasoning segments, while stronger models provide more robust reasoning at higher computational cost. We present AgentCollab, a self-driven collaborative inference framework that dynamically coordinates models with different reasoning capacities during agent execution. Instead of relying on external routing modules, the framework uses the agent's own self-reflection signal to determine whether the current reasoning trajectory is making meaningful progress, and escalates control to a stronger reasoning tier only when necessary. To further stabilize long-horizon execution, we introduce a difficulty-aware cumulative escalation strategy that allocates additional reasoning budget based on recent failure signals. In our experiments, we instantiate this framework using a two-level small-large model setting. Experiments on diverse multi-step agent benchmarks show that AgentCollab consistently improves the accuracy-efficiency Pareto frontier of LLM agents. 

---
# Toward Culturally Grounded Natural Language Processing 

**Authors**: Sina Bagheri Nezhad  

**Link**: [PDF](https://arxiv.org/pdf/2603.26013)  

**Abstract**: Recent progress in multilingual NLP is often taken as evidence of broader global inclusivity, but a growing literature shows that multilingual capability and cultural competence come apart. This paper synthesizes over 50 papers from 2020--2026 spanning multilingual performance inequality, cross-lingual transfer, culture-aware evaluation, cultural alignment, multimodal local-knowledge modeling, benchmark design critiques, and community-grounded data practices. Across this literature, training data coverage remains a strong determinant of performance, yet it is not sufficient: tokenization, prompt language, translated benchmark design, culturally specific supervision, and multimodal context all materially affect outcomes. Recent work on Global-MMLU, CDEval, WorldValuesBench, CulturalBench, CULEMO, CulturalVQA, GIMMICK, DRISHTIKON, WorldCuisines, CARE, CLCA, and newer critiques of benchmark design and community-grounded evaluation shows that strong multilingual models can still flatten local norms, misread culturally grounded cues, and underperform in lower-resource or community-specific settings. We argue that the field should move from treating languages as isolated rows in a benchmark spreadsheet toward modeling communicative ecologies: the institutions, scripts, translation pipelines, domains, modalities, and communities through which language is used. On that basis, we propose a research agenda for culturally grounded NLP centered on richer contextual metadata, culturally stratified evaluation, participatory alignment, within-language variation, and multimodal community-aware design. 

---
# MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain Personalization 

**Authors**: Weizhi Zhang, Xiaokai Wei, Wei-Chieh Huang, Zheng Hui, Chen Wang, Michelle Gong, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.25973)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have expanded context windows to million-token scales, yet benchmarks for evaluating memory remain limited to short-session synthetic dialogues. We introduce \textsc{MemoryCD}, the first large-scale, user-centric, cross-domain memory benchmark derived from lifelong real-world behaviors in the Amazon Review dataset. Unlike existing memory datasets that rely on scripted personas to generate synthetic user data, \textsc{MemoryCD} tracks authentic user interactions across years and multiple domains. We construct a multi-faceted long-context memory evaluation pipeline of 14 state-of-the-art LLM base models with 6 memory method baselines on 4 distinct personalization tasks over 12 diverse domains to evaluate an agent's ability to simulate real user behaviors in both single and cross-domain settings. Our analysis reveals that existing memory methods are far from user satisfaction in various domains, offering the first testbed for cross-domain life-long personalization evaluation. 

---
# When Chain-of-Thought Backfires: Evaluating Prompt Sensitivity in Medical Language Models 

**Authors**: Binesh Sadanandan, Vahid Behzadan  

**Link**: [PDF](https://arxiv.org/pdf/2603.25960)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in medical settings, yet their sensitivity to prompt formatting remains poorly characterized. We evaluate MedGemma (4B and 27B parameters) on MedMCQA (4,183 questions) and PubMedQA (1,000 questions) across a broad suite of robustness tests. Our experiments reveal several concerning findings. Chain-of-Thought (CoT) prompting decreases accuracy by 5.7% compared to direct answering. Few-shot examples degrade performance by 11.9% while increasing position bias from 0.14 to 0.47. Shuffling answer options causes the model to change predictions 59.1% of the time, with accuracy dropping up to 27.4 percentage points. Front-truncating context to 50% causes accuracy to plummet below the no-context baseline, yet back-truncation preserves 97% of full-context accuracy. We further show that cloze scoring (selecting the highest log-probability option token) achieves 51.8% (4B) and 64.5% (27B), surpassing all prompting strategies and revealing that models "know" more than their generated text shows. Permutation voting recovers 4 percentage points over single-ordering inference. These results demonstrate that prompt engineering techniques validated on general-purpose models do not transfer to domain-specific medical LLMs, and that reliable alternatives exist. 

---
# Can Small Models Reason About Legal Documents? A Comparative Study 

**Authors**: Snehit Vaddi  

**Link**: [PDF](https://arxiv.org/pdf/2603.25944)  

**Abstract**: Large language models show promise for legal applications, but deploying frontier models raises concerns about cost, latency, and data privacy. We evaluate whether sub-10B parameter models can serve as practical alternatives by testing nine models across three legal benchmarks (ContractNLI, CaseHOLD, and ECtHR) using five prompting strategies (direct, chain-of-thought, few-shot, BM25 RAG, and dense RAG). Across 405 experiments with three random seeds per configuration, we find that a Mixture-of-Experts model activating only 3B parameters matches GPT-4o-mini in mean accuracy while surpassing it on legal holding identification, and that architecture and training quality matter more than raw parameter count. Our largest model (9B parameters) performs worst overall. Chain-of-thought prompting proves sharply task-dependent, improving contract entailment but degrading multiple-choice legal reasoning, while few-shot prompting emerges as the most consistently effective strategy. Comparing BM25 and dense retrieval for RAG, we find near-identical results, suggesting the bottleneck lies in the language model's utilization of retrieved context rather than retrieval quality. All experiments were conducted via cloud inference APIs at a total cost of $62, demonstrating that rigorous LLM evaluation is accessible without dedicated GPU infrastructure. 

---
# Density-aware Soft Context Compression with Semi-Dynamic Compression Ratio 

**Authors**: Yijiong Yu, Shuai Yuan, Jie Zheng, Huazheng Wang, Ji Pei  

**Link**: [PDF](https://arxiv.org/pdf/2603.25926)  

**Abstract**: Soft context compression reduces the computational workload of processing long contexts in LLMs by encoding long context into a smaller number of latent tokens. However, existing frameworks apply uniform compression ratios, failing to account for the extreme variance in natural language information density. While adopting a density-aware dynamic compression ratio seems intuitive, empirical investigations reveal that models struggle intrinsically with operations parameterized by input dependent, continuous structural hyperparameters. To resolve this pitfall, we introduce Semi-Dynamic Context Compression framework. Our approach features a Discrete Ratio Selector, which predicts a compression target based on intrinsic information density and quantizes it to a predefined set of discrete compression ratios. It is efficiently jointly trained with the compressor on synthetic data, with the summary lengths as a proxy to create labels for compression ratio prediction. Extensive evaluations confirm that our density-aware framework, utilizing mean pooling as the backbone, consistently outperforms static baselines, establishing a robust Pareto frontier for context compression techniques. Our code, data and model weights are available at this https URL 

---
# Methods for Knowledge Graph Construction from Text Collections: Development and Applications 

**Authors**: Vanni Zavarella  

**Link**: [PDF](https://arxiv.org/pdf/2603.25862)  

**Abstract**: Virtually every sector of society is experiencing a dramatic growth in the volume of unstructured textual data that is generated and published, from news and social media online interactions, through open access scholarly communications and observational data in the form of digital health records and online drug reviews. The volume and variety of data across all this range of domains has created both unprecedented opportunities and pressing challenges for extracting actionable knowledge for several application scenarios. However, the extraction of rich semantic knowledge demands the deployment of scalable and flexible automatic methods adaptable across text genres and schema specifications. Moreover, the full potential of these data can only be unlocked by coupling information extraction methods with Semantic Web techniques for the construction of full-fledged Knowledge Graphs, that are semantically transparent, explainable by design and interoperable. In this thesis, we experiment with the application of Natural Language Processing, Machine Learning and Generative AI methods, powered by Semantic Web best practices, to the automatic construction of Knowledge Graphs from large text corpora, in three use case applications: the analysis of the Digital Transformation discourse in the global news and social media platforms; the mapping and trend analysis of recent research in the Architecture, Engineering, Construction and Operations domain from a large corpus of publications; the generation of causal relation graphs of biomedical entities from electronic health records and patient-authored drug reviews. The contributions of this thesis to the research community are in terms of benchmark evaluation results, the design of customized algorithms and the creation of data resources in the form of Knowledge Graphs, together with data analysis results built on top of them. 

---
# Gradient-Informed Training for Low-Resource Multilingual Speech Translation 

**Authors**: Ruiyan Sun, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2603.25836)  

**Abstract**: In low-resource multilingual speech-to-text translation, uniform architectural sharing across languages frequently introduces representation conflicts that impede convergence. This work proposes a principled methodology to automatically determine layer-specific sharing patterns by mining training gradient information. Our approach employs three distinct analysis strategies: distance-based language clustering, self/cross-task divergence metrics for capacity allocation, and joint factorization coupled with canonical correlation analysis for subspace alignment. Extensive evaluation across four language pairs (using the SeamlessM4T-Medium architecture) demonstrates persistent improvements in translation quality metrics. 

---
# Doctorina MedBench: End-to-End Evaluation of Agent-Based Medical AI 

**Authors**: Anna Kozlova, Stanislau Salavei, Pavel Satalkin, Hanna Plotnitskaya, Sergey Parfenyuk  

**Link**: [PDF](https://arxiv.org/pdf/2603.25821)  

**Abstract**: We present Doctorina MedBench, a comprehensive evaluation framework for agent-based medical AI based on the simulation of realistic physician-patient interactions. Unlike traditional medical benchmarks that rely on solving standardized test questions, the proposed approach models a multi-step clinical dialogue in which either a physician or an AI system must collect medical history, analyze attached materials (including laboratory reports, images, and medical documents), formulate differential diagnoses, and provide personalized recommendations. System performance is evaluated using the D.O.T.S. metric, which consists of four components: Diagnosis, Observations/Investigations, Treatment, and Step Count, enabling assessment of both clinical correctness and dialogue efficiency.
The system also incorporates a multi-level testing and quality monitoring architecture designed to detect model degradation during both development and deployment. The framework supports safety-oriented trap cases, category-based random sampling of clinical scenarios, and full regression testing. The dataset currently contains more than 1,000 clinical cases covering over 750 diagnoses. The universality of the evaluation metrics allows the framework to be used not only to assess medical AI systems, but also to evaluate physicians and support the development of clinical reasoning skills. Our results suggest that simulation of clinical dialogue may provide a more realistic assessment of clinical competence compared to traditional examination-style benchmarks. 

---
# RealChart2Code: Advancing Chart-to-Code Generation with Real Data and Multi-Task Evaluation 

**Authors**: Jiajun Zhang, Yuying Li, Zhixun Li, Xingyu Guo, Jingzhuo Wu, Leqi Zheng, Yiran Yang, Jianke Zhang, Qingbin Li, Shannan Yan, Zhetong Li, Changguo Jia, Junfei Wu, Zilei Wang, Qiang Liu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.25804)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated impressive capabilities in code generation across various domains. However, their ability to replicate complex, multi-panel visualizations from real-world data remains largely unassessed. To address this gap, we introduce \textbf{\texttt{RealChart2Code}}, a new large-scale benchmark with over 2,800 instances grounded in authentic datasets and featuring tasks with clear analytical intent. Crucially, it is the first benchmark to systematically evaluate chart generation from large-scale raw data and assess iterative code refinement in a multi-turn conversational setting. Our comprehensive evaluation of 14 leading VLMs on \texttt{RealChart2Code} reveals significant performance degradation compared to simpler benchmarks, highlighting their struggles with complex plot structures and authentic data. Our analysis uncovers a substantial performance gap between proprietary and open-weight models and confirms that even state-of-the-art VLMs often fail to accurately replicate intricate, multi-panel charts. These findings provide valuable insights into the current limitations of VLMs and guide future research directions. We release the benchmark and code at \url{this https URL}. 

---
# Relational graph-driven differential denoising and diffusion attention fusion for multimodal conversation emotion recognition 

**Authors**: Ying Liu, Yuntao Shou, Wei Ai, Tao Meng, Keqin Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.25752)  

**Abstract**: In real-world scenarios, audio and video signals are often subject to environmental noise and limited acquisition conditions, resulting in extracted features containing excessive noise. Furthermore, there is an imbalance in data quality and information carrying capacity between different modalities. These two issues together lead to information distortion and weight bias during the fusion phase, impairing overall recognition performance. Most existing methods neglect the impact of noisy modalities and rely on implicit weighting to model modality importance, thereby failing to explicitly account for the predominant contribution of the textual modality in emotion understanding. To address these issues, we propose a relation-aware denoising and diffusion attention fusion model for MCER. Specifically, we first design a differential Transformer that explicitly computes the differences between two attention maps, thereby enhancing temporally consistent information while suppressing time-irrelevant noise, which leads to effective denoising in both audio and video modalities. Second, we construct modality-specific and cross-modality relation subgraphs to capture speaker-dependent emotional dependencies, enabling fine-grained modeling of intra- and inter-modal relationships. Finally, we introduce a text-guided cross-modal diffusion mechanism that leverages self-attention to model intra-modal dependencies and adaptively diffuses audiovisual information into the textual stream, ensuring more robust and semantically aligned multimodal fusion. 

---
# Learning to Commit: Generating Organic Pull Requests via Online Repository Memory 

**Authors**: Mo Li, L.H. Xu, Qitai Tan, Ting Cao, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.26664)  

**Abstract**: Large language model (LLM)-based coding agents achieve impressive results on controlled benchmarks yet routinely produce pull requests that real maintainers reject. The root cause is not functional incorrectness but a lack of organicity: generated code ignores project-specific conventions, duplicates functionality already provided by internal APIs, and violates implicit architectural constraints accumulated over years of development. Simply exposing an agent to the latest repository snapshot is not enough: the snapshot reveals the final state of the codebase, but not the repository-specific change patterns by which that state was reached. We introduce Learning to Commit, a framework that closes this gap through Online Repository Memory. Given a repository with a strict chronological split, the agent performs supervised contrastive reflection on earlier commits: it blindly attempts to resolve each historical issue, compares its prediction against the oracle diff, and distils the gap into a continuously growing set of skills-reusable patterns capturing coding style, internal API usage, and architectural invariants. When a new PR description arrives, the agent conditions its generation on these accumulated skills, producing changes grounded in the project's own evolution rather than generic pretraining priors. Evaluation is conducted on genuinely future, merged pull requests that could not have been seen during the skill-building phase, and spans multiple dimensions including functional correctness, code-style consistency, internal API reuse rate, and modified-region plausibility. Experiments on an expert-maintained repository with rich commit history show that Online Repository Memory effectively improves organicity scores on held-out future tasks. 

---
# PerceptionComp: A Video Benchmark for Complex Perception-Centric Reasoning 

**Authors**: Shaoxuan Li, Zhixuan Zhao, Hanze Deng, Zirun Ma, Shulin Tian, Zuyan Liu, Yushi Hu, Haoning Wu, Yuhao Dong, Benlin Liu, Ziwei Liu, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2603.26653)  

**Abstract**: We introduce PerceptionComp, a manually annotated benchmark for complex, long-horizon, perception-centric video reasoning. PerceptionComp is designed so that no single moment is sufficient: answering each question requires multiple temporally separated pieces of visual evidence and compositional constraints under conjunctive and sequential logic, spanning perceptual subtasks such as objects, attributes, relations, locations, actions, and events, and requiring skills including semantic recognition, visual correspondence, temporal reasoning, and spatial reasoning. The benchmark contains 1,114 highly complex questions on 279 videos from diverse domains including city walk tours, indoor villa tours, video games, and extreme outdoor sports, with 100% manual annotation. Human studies show that PerceptionComp requires substantial test-time thinking and repeated perception steps: participants take much longer than on prior benchmarks, and accuracy drops to near chance (18.97%) when rewatching is disallowed. State-of-the-art MLLMs also perform substantially worse on PerceptionComp than on existing benchmarks: the best model in our evaluation, Gemini-3-Flash, reaches only 45.96% accuracy in the five-choice setting, while open-source models remain below 40%. These results suggest that perception-centric long-horizon video reasoning remains a major bottleneck, and we hope PerceptionComp will help drive progress in perceptual reasoning. 

---
# Entanglement as Memory: Mechanistic Interpretability of Quantum Language Models 

**Authors**: Nathan Roll  

**Link**: [PDF](https://arxiv.org/pdf/2603.26494)  

**Abstract**: Quantum language models have shown competitive performance on sequential tasks, yet whether trained quantum circuits exploit genuinely quantum resources -- or merely embed classical computation in quantum hardware -- remains unknown. Prior work has evaluated these models through endpoint metrics alone, without examining the memory strategies they actually learn internally. We introduce the first mechanistic interpretability study of quantum language models, combining causal gate ablation, entanglement tracking, and density-matrix interchange interventions on a controlled long-range dependency task. We find that single-qubit models are exactly classically simulable and converge to the same geometric strategy as matched classical baselines, while two-qubit models with entangling gates learn a representationally distinct strategy that encodes context in inter-qubit entanglement -- confirmed by three independent causal tests (p < 0.0001, d = 0.89). On real quantum hardware, only the classical geometric strategy survives device noise; the entanglement strategy degrades to chance. These findings open mechanistic interpretability as a tool for the science of quantum language models and reveal a noise-expressivity tradeoff governing which learned strategies survive deployment. 

---
# A Formal Framework for Uncertainty Analysis of Text Generation with Large Language Models 

**Authors**: Steffen Herbold, Florian Lemmerich  

**Link**: [PDF](https://arxiv.org/pdf/2603.26363)  

**Abstract**: The generation of texts using Large Language Models (LLMs) is inherently uncertain, with sources of uncertainty being not only the generation of texts, but also the prompt used and the downstream interpretation. Within this work, we provide a formal framework for the measurement of uncertainty that takes these different aspects into account. Our framework models prompting, generation, and interpretation as interconnected autoregressive processes that can be combined into a single sampling tree. We introduce filters and objective functions to describe how different aspects of uncertainty can be expressed over the sampling tree and demonstrate how to express existing approaches towards uncertainty through these functions. With our framework we show not only how different methods are formally related and can be reduced to a common core, but also point out additional aspects of uncertainty that have not yet been studied. 

---
# Working Notes on Late Interaction Dynamics: Analyzing Targeted Behaviors of Late Interaction Models 

**Authors**: Antoine Edy, Max Conti, Quentin Macé  

**Link**: [PDF](https://arxiv.org/pdf/2603.26259)  

**Abstract**: While Late Interaction models exhibit strong retrieval performance, many of their underlying dynamics remain understudied, potentially hiding performance bottlenecks. In this work, we focus on two topics in Late Interaction retrieval: a length bias that arises when using multi-vector scoring, and the similarity distribution beyond the best scores pooled by the MaxSim operator. We analyze these behaviors for state-of-the-art models on the NanoBEIR benchmark. Results show that while the theoretical length bias of causal Late Interaction models holds in practice, bi-directional models can also suffer from it in extreme cases. We also note that no significant similarity trend lies beyond the top-1 document token, validating that the MaxSim operator efficiently exploits the token-level similarity scores. 

---
# DataFlex: A Unified Framework for Data-Centric Dynamic Training of Large Language Models 

**Authors**: Hao Liang, Zhengyang Zhao, Meiyi Qiang, Mingrui Chen, Lu Ma, Rongyi Yu, Hengyi Feng, Shixuan Sun, Zimo Meng, Xiaochen Ma, Xuanlin Yang, Qifeng Cai, Ruichuan An, Bohan Zeng, Zhen Hao Wong, Chengyu Shen, Runming He, Zhaoyang Han, Yaowei Zheng, Fangcheng Fu, Conghui He, Bin Cui, Zhiyu Li, Weinan E, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26164)  

**Abstract**: Data-centric training has emerged as a promising direction for improving large language models (LLMs) by optimizing not only model parameters but also the selection, composition, and weighting of training data during optimization. However, existing approaches to data selection, data mixture optimization, and data reweighting are often developed in isolated codebases with inconsistent interfaces, hindering reproducibility, fair comparison, and practical integration. In this paper, we present DataFlex, a unified data-centric dynamic training framework built upon LLaMA-Factory. DataFlex supports three major paradigms of dynamic data optimization: sample selection, domain mixture adjustment, and sample reweighting, while remaining fully compatible with the original training workflow. It provides extensible trainer abstractions and modular components, enabling a drop-in replacement for standard LLM training, and unifies key model-dependent operations such as embedding extraction, inference, and gradient computation, with support for large-scale settings including DeepSpeed ZeRO-3. We conduct comprehensive experiments across multiple data-centric methods. Dynamic data selection consistently outperforms static full-data training on MMLU across both Mistral-7B and Llama-3.2-3B. For data mixture, DoReMi and ODM improve both MMLU accuracy and corpus-level perplexity over default proportions when pretraining Qwen2.5-1.5B on SlimPajama at 6B and 30B token scales. DataFlex also achieves consistent runtime improvements over original implementations. These results demonstrate that DataFlex provides an effective, efficient, and reproducible infrastructure for data-centric dynamic training of LLMs. 

---
# Finding Distributed Object-Centric Properties in Self-Supervised Transformers 

**Authors**: Samyak Rawlekar, Amitabh Swain, Yujun Cai, Yiwei Wang, Ming-Hsuan Yang, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2603.26127)  

**Abstract**: Self-supervised Vision Transformers (ViTs) like DINO show an emergent ability to discover objects, typically observed in [CLS] token attention maps of the final layer. However, these maps often contain spurious activations resulting in poor localization of objects. This is because the [CLS] token, trained on an image-level objective, summarizes the entire image instead of focusing on objects. This aggregation dilutes the object-centric information existing in the local, patch-level interactions. We analyze this by computing inter-patch similarity using patch-level attention components (query, key, and value) across all layers. We find that: (1) Object-centric properties are encoded in the similarity maps derived from all three components ($q, k, v$), unlike prior work that uses only key features or the [CLS] token. (2) This object-centric information is distributed across the network, not just confined to the final layer. Based on these insights, we introduce Object-DINO, a training-free method that extracts this distributed object-centric information. Object-DINO clusters attention heads across all layers based on the similarities of their patches and automatically identifies the object-centric cluster corresponding to all objects. We demonstrate Object-DINO's effectiveness on two applications: enhancing unsupervised object discovery (+3.6 to +12.4 CorLoc gains) and mitigating object hallucination in Multimodal Large Language Models by providing visual grounding. Our results demonstrate that using this distributed object-centric information improves downstream tasks without additional training. 

---
# Selective Deficits in LLM Mental Self-Modeling in a Behavior-Based Test of Theory of Mind 

**Authors**: Christopher Ackerman  

**Link**: [PDF](https://arxiv.org/pdf/2603.26089)  

**Abstract**: The ability to represent oneself and others as agents with knowledge, intentions, and belief states that guide their behavior - Theory of Mind - is a human universal that enables us to navigate - and manipulate - the social world. It is supported by our ability to form mental models of ourselves and others. Its ubiquity in human affairs entails that LLMs have seen innumerable examples of it in their training data and therefore may have learned to mimic it, but whether they have actually learned causal models that they can deploy in arbitrary settings is unclear. We therefore develop a novel experimental paradigm that requires that subjects form representations of the mental states of themselves and others and act on them strategically rather than merely describe them. We test a wide range of leading open and closed source LLMs released since 2024, as well as human subjects, on this paradigm. We find that 1) LLMs released before mid-2025 fail at all of our tasks, 2) more recent LLMs achieve human-level performance on modeling the cognitive states of others, and 3) even frontier LLMs fail at our self-modeling task - unless afforded a scratchpad in the form of a reasoning trace. We further demonstrate cognitive load effects on other-modeling tasks, offering suggestive evidence that LLMs are using something akin to limited-capacity working memory to hold these mental representations in mind during a single forward pass. Finally, we explore the mechanisms by which reasoning models succeed at the self- and other-modeling tasks, and show that they readily engage in strategic deception. 

---
# Semi-Automated Knowledge Engineering and Process Mapping for Total Airport Management 

**Authors**: Darryl Teo, Adharsha Sam, Chuan Shen Marcus Koh, Rakesh Nagi, Nuno Antunes Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2603.26076)  

**Abstract**: Documentation of airport operations is inherently complex due to extensive technical terminology, rigorous regulations, proprietary regional information, and fragmented communication across multiple stakeholders. The resulting data silos and semantic inconsistencies present a significant impediment to the Total Airport Management (TAM) initiative. This paper presents a methodological framework for constructing a domain-grounded, machine-readable Knowledge Graph (KG) through a dual-stage fusion of symbolic Knowledge Engineering (KE) and generative Large Language Models (LLMs).
The framework employs a scaffolded fusion strategy in which expert-curated KE structures guide LLM prompts to facilitate the discovery of semantically aligned knowledge triples. We evaluate this methodology on the Google LangExtract library and investigate the impact of context window utilization by comparing localized segment-based inference with document-level processing. Contrary to prior empirical observations of long-context degradation in LLMs, document-level processing improves the recovery of non-linear procedural dependencies.
To ensure the high-fidelity provenance required in airport operations, the proposed framework fuses a probabilistic model for discovery and a deterministic algorithm for anchoring every extraction to its ground source. This ensures absolute traceability and verifiability, bridging the gap between "black-box" generative outputs and the transparency required for operational tooling. Finally, we introduce an automated framework that operationalizes this pipeline to synthesize complex operational workflows from unstructured textual corpora. 

---
# H-Node Attack and Defense in Large Language Models 

**Authors**: Eric Yocam, Varghese Vaidyan, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.26045)  

**Abstract**: We present H-Node Adversarial Noise Cancellation (H-Node ANC), a mechanistic framework that identifies, exploits, and defends hallucination representations in transformer-based large language models (LLMs) at the level of individual hidden-state dimensions. A logistic regression probe trained on last-token hidden states localizes hallucination signal to a small set of high-variance dimensions -- termed Hallucination Nodes (H-Nodes) -- with probe AUC reaching 0.90 across four architectures. A white-box adversarial attack amplifies these dimensions at inference time via a real-time forward hook, achieving a selectivity of 3.02x with less than 10% visibility to the defender. Adaptive ANC defense suppresses H-Node excess in-pass using confidence-weighted cancellation, reducing grounded activation drift by 33-42% over static cancellation. A dynamic iterative extension that re-ranks cancellation targets across successive passes recovers up to 0.69 robustness from a single-pass baseline of 8%. All contributions are validated on OPT-125M, Phi-3-mini-4k-instruct, LLaMA-3-8B-Instruct, and Mistral-7B-Instruct-v0.3 (125M-8B parameters). Perplexity impact is surgical (<5%) and MMLU degradation is at most 3%, confirming that the defense does not impair general reasoning capability. 

---
# Policy-Guided World Model Planning for Language-Conditioned Visual Navigation 

**Authors**: Amirhosein Chahe, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2603.25981)  

**Abstract**: Navigating to a visually specified goal given natural language instructions remains a fundamental challenge in embodied AI. Existing approaches either rely on reactive policies that struggle with long-horizon planning, or employ world models that suffer from poor action initialization in high-dimensional spaces. We present PiJEPA, a two-stage framework that combines the strengths of learned navigation policies with latent world model planning for instruction-conditioned visual navigation. In the first stage, we finetune an Octo-based generalist policy, augmented with a frozen pretrained vision encoder (DINOv2 or V-JEPA-2), on the CAST navigation dataset to produce an informed action distribution conditioned on the current observation and language instruction. In the second stage, we use this policy-derived distribution to warm-start Model Predictive Path Integral (MPPI) planning over a separately trained JEPA world model, which predicts future latent states in the embedding space of the same frozen encoder. By initializing the MPPI sampling distribution from the policy prior rather than from an uninformed Gaussian, our planner converges faster to high-quality action sequences that reach the goal. We systematically study the effect of the vision encoder backbone, comparing DINOv2 and V-JEPA-2, across both the policy and world model components. Experiments on real-world navigation tasks demonstrate that PiJEPA significantly outperforms both standalone policy execution and uninformed world model planning, achieving improved goal-reaching accuracy and instruction-following fidelity. 

---
# Do Neurons Dream of Primitive Operators? Wake-Sleep Compression Rediscovers Schank's Event Semantics 

**Authors**: Peter Balogh  

**Link**: [PDF](https://arxiv.org/pdf/2603.25975)  

**Abstract**: We show that they do. Schank's conceptual dependency theory proposed that all events decompose into primitive operations -- ATRANS, PTRANS, MTRANS, and others -- hand-coded from linguistic intuition. Can the same primitives be discovered automatically through compression pressure alone?
We adapt DreamCoder's wake-sleep library learning to event state transformations. Given events as before/after world state pairs, our system finds operator compositions explaining each event (wake), then extracts recurring patterns as new operators optimized under Minimum Description Length (sleep). Starting from four generic primitives, it discovers operators mapping directly to Schank's: MOVE_PROP_has = ATRANS, CHANGE_location = PTRANS, SET_knows = MTRANS, SET_consumed = INGEST, plus compound operators ("mail" = ATRANS + PTRANS) and novel emotional state operators absent from Schank's taxonomy.
We validate on synthetic events and real-world commonsense data from the ATOMIC knowledge graph. On synthetic data, discovered operators achieve Bayesian MDL within 4% of Schank's hand-coded primitives while explaining 100% of events vs. Schank's 81%. On ATOMIC, results are more dramatic: Schank's primitives explain only 10% of naturalistic events, while the discovered library explains 100%. Dominant operators are not physical-action primitives but mental and emotional state changes -- CHANGE_wants (20%), CHANGE_feels (18%), CHANGE_is (18%) -- none in Schank's original taxonomy.
These results provide the first empirical evidence that event primitives can be derived from compression pressure, that Schank's core primitives are information-theoretically justified, and that the complete inventory is substantially richer than proposed -- with mental/emotional operators dominating in naturalistic data. 

---
