# Instruction set for the representation of graphs 

**Authors**: Ezequiel Lopez-Rubio, Mario Pascual-Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2603.11039)  

**Abstract**: We present IsalGraph, a method for representing the structure of any finite, simple graph as a compact string over a nine-character instruction alphabet. The encoding is executed by a small virtual machine comprising a sparse graph, a circular doubly-linked list (CDLL) of graph-node references, and two traversal pointers. Instructions either move a pointer through the CDLL or insert a node or edge into the graph. A key design property is that every string over the alphabet decodes to a valid graph, with no invalid states reachable. A greedy \emph{GraphToString} algorithm encodes any connected graph into a string in time polynomial in the number of nodes; an exhaustive-backtracking variant produces a canonical string by selecting the lexicographically smallest shortest string across all starting nodes and all valid traversal orders. We evaluate the representation on five real-world graph benchmark datasets (IAM Letter LOW/MED/HIGH, LINUX, and AIDS) and show that the Levenshtein distance between IsalGraph strings correlates strongly with graph edit distance (GED). Together, these properties make IsalGraph strings a compact, isomorphism-invariant, and language-model-compatible sequential encoding of graph structure, with direct applications in graph similarity search, graph generation, and graph-conditioned language modelling 

---
# Beyond the Illusion of Consensus: From Surface Heuristics to Knowledge-Grounded Evaluation in LLM-as-a-Judge 

**Authors**: Mingyang Song, Mao Zheng, Chenning Xu  

**Link**: [PDF](https://arxiv.org/pdf/2603.11027)  

**Abstract**: The paradigm of LLM-as-a-judge relies on a critical assumption, namely that high inter-evaluator agreement indicates reliable and objective evaluation. We present two complementary findings that challenge this assumption. \textbf{First}, we demonstrate that this consensus is frequently illusory. We identify and formalize \textbf{Evaluation Illusion}, a phenomenon where LLM judges generate sophisticated critiques yet anchor scores on shared surface heuristics rather than substantive quality. Through a large-scale study of 105,600 evaluation instances (32 LLMs $\times$ 3 frontier judges $\times$ 100 tasks $\times$ 11 temperatures), we show that model-level agreement (Spearman $\rho = 0.99$) masks fragile sample-level agreement (Pearson $\bar{r} = 0.72$; absolute agreement ICC $= 0.67$), that merely sharing rubric structure restores 62\% of total agreement, and that high-quality outputs paradoxically receive the \textit{least} consistent evaluations. \textbf{Second}, we demonstrate that dynamically generating evaluation rubrics grounded in domain knowledge produces more meaningful assessment. We introduce MERG (Metacognitive Enhanced Rubric Generation), a knowledge-driven rubric generation framework whose domain-selective effects confirm this. Agreement \textit{increases} in codified domains (Education +22\%, Academic +27\%) where knowledge anchors evaluators on shared standards, while it decreases in subjective domains where genuine evaluative pluralism emerges. These findings suggest that evaluation rubrics should be dynamically enriched with expert knowledge rather than relying on generic criteria, with implications for reward modeling in RLAIF. 

---
# LLM2Vec-Gen: Generative Embeddings from Large Language Models 

**Authors**: Parishad BehnamGhader, Vaibhav Adlakha, Fabian David Schmidt, Nicolas Chapados, Marius Mosbach, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2603.10913)  

**Abstract**: LLM-based text embedders typically encode the semantic content of their input. However, embedding tasks require mapping diverse inputs to similar outputs. Typically, this input-output is addressed by training embedding models with paired data using contrastive learning. In this work, we propose a novel self-supervised approach, LLM2Vec-Gen, which adopts a different paradigm: rather than encoding the input, we learn to represent the model's potential response. Specifically, we add trainable special tokens to the LLM's vocabulary, append them to input, and optimize them to represent the LLM's response in a fixed-length sequence. Training is guided by the LLM's own completion for the query, along with an unsupervised embedding teacher that provides distillation targets. This formulation helps to bridge the input-output gap and transfers LLM capabilities such as safety alignment and reasoning to embedding tasks. Crucially, the LLM backbone remains frozen and training requires only unlabeled queries. LLM2Vec-Gen achieves state-of-the-art self-supervised performance on the Massive Text Embedding Benchmark (MTEB), improving by 9.3% over the best unsupervised embedding teacher. We also observe up to 43.2% reduction in harmful content retrieval and 29.3% improvement in reasoning capabilities for embedding tasks. Finally, the learned embeddings are interpretable and can be decoded into text to reveal their semantic content. 

---
# GLM-OCR Technical Report 

**Authors**: Shuaiqi Duan, Yadong Xue, Weihan Wang, Zhe Su, Huan Liu, Sheng Yang, Guobing Gan, Guo Wang, Zihan Wang, Shengdong Yan, Dexin Jin, Yuxuan Zhang, Guohong Wen, Yanfeng Wang, Yutao Zhang, Xiaohan Zhang, Wenyi Hong, Yukuo Cen, Da Yin, Bin Chen, Wenmeng Yu, Xiaotao Gu, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10910)  

**Abstract**: GLM-OCR is an efficient 0.9B-parameter compact multimodal model designed for real-world document understanding. It combines a 0.4B-parameter CogViT visual encoder with a 0.5B-parameter GLM language decoder, achieving a strong balance between computational efficiency and recognition performance. To address the inefficiency of standard autoregressive decoding in deterministic OCR tasks, GLM-OCR introduces a Multi-Token Prediction (MTP) mechanism that predicts multiple tokens per step, significantly improving decoding throughput while keeping memory overhead low through shared parameters. At the system level, a two-stage pipeline is adopted: PP-DocLayout-V3 first performs layout analysis, followed by parallel region-level recognition. Extensive evaluations on public benchmarks and industrial scenarios show that GLM-OCR achieves competitive or state-of-the-art performance in document parsing, text and formula transcription, table structure recovery, and key information extraction. Its compact architecture and structured generation make it suitable for both resource-constrained edge deployment and large-scale production systems. 

---
# From Images to Words: Efficient Cross-Modal Knowledge Distillation to Language Models from Black-box Teachers 

**Authors**: Ayan Sengupta, Shantanu Dixit, Md Shad Akhtar, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2603.10877)  

**Abstract**: Knowledge distillation (KD) methods are pivotal in compressing large pre-trained language models into smaller models, ensuring computational efficiency without significantly dropping performance. Traditional KD techniques assume homogeneity in modalities between the teacher (source) and the student (target) models. On the other hand, existing multimodal knowledge distillation methods require modality-specific pre-training of the teacher model, which is computationally infeasible in most cases. In this paper, we introduce ARMADA, an efficient cross-modal knowledge distillation framework designed to transfer knowledge from large vision-language models, including black-box models, to language-only models. Unlike existing KD techniques that rely on the internal structures of multimodal teachers or require computationally expensive pre-training, ARMADA leverages novel alignment techniques to distil knowledge without altering the teacher model, ensuring efficiency and scalability. We empirically validate ARMADA on twelve natural language understanding, eight complex generative reasoning and five instruction-tuning tasks, demonstrating consistent performance improvements in large models such as DeBERTa-v2-1.4B, OPT-1.3B, LLaMA-{3B, 7B, 8B}. ARMADA achieves up to 3.4% improvement on language understanding tasks and 2.6% boost in generative reasoning, all without requiring expensive multimodal pre-training or fine-tuning of the teacher model. Our findings challenge conventional knowledge distillation paradigms by demonstrating that even vision-language models, despite lacking direct textual understanding, can significantly enhance language models when distilled appropriately. 

---
# An Extreme Multi-label Text Classification (XMTC) Library Dataset: What if we took "Use of Practical AI in Digital Libraries" seriously? 

**Authors**: Jennifer D'Souza, Sameer Sadruddin, Maximilian Kähler, Andrea Salfinger, Luca Zaccagna, Francesca Incitti, Lauro Snidaro, Osma Suominen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10876)  

**Abstract**: Subject indexing is vital for discovery but hard to sustain at scale and across languages. We release a large bilingual (English/German) corpus of catalog records annotated with the Integrated Authority File (GND), plus a machine-actionable GND taxonomy. The resource enables ontology-aware multi-label classification, mapping text to authority terms, and agent-assisted cataloging with reproducible, authority-grounded evaluation. We provide a brief statistical profile and qualitative error analyses of three systems. We invite the community to assess not only accuracy but usefulness and transparency, toward authority-anchored AI co-pilots that amplify catalogers' work. 

---
# SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0 

**Authors**: Nevidu Jayatilleke, Nisansa de Silva, Uthpala Nimanthi, Gagani Kulathilaka, Azra Safrullah, Johan Sofalas  

**Link**: [PDF](https://arxiv.org/pdf/2603.10861)  

**Abstract**: SiDiaC-v.2.0 is the largest comprehensive Sinhala Diachronic Corpus to date, covering a period from 1800 CE to 1955 CE in terms of publication dates, and a historical span from the 5th to the 20th century CE in terms of written dates. The corpus consists of 244k words across 185 literary works that underwent thorough filtering, preprocessing, and copyright compliance checks, followed by extensive post-processing. Additionally, a subset of 59 documents totalling 70k words was annotated based on their written dates. Texts from the National Library of Sri Lanka were selected from the SiDiaC-v.1.0 non-filtered list, which was digitised using Google Document AI OCR. This was followed by post-processing to correct formatting issues, address code-mixing, include special tokens, and fix malformed tokens. The construction of SiDiaC-v.2.0 was informed by practices from other corpora, such as FarPaHC, SiDiaC-v.1.0, and CCOHA. This was particularly relevant for syntactic annotation and text normalisation strategies, given the shared characteristics of low-resource language status between Faroese and the similar cleaning strategies utilised in CCOHA. This corpus is categorised into two layers based on genres: primary and secondary. The primary categorisation is binary, assigning each book to either Non-Fiction or Fiction. The secondary categorisation is more detailed, grouping texts under specific genres such as Religious, History, Poetry, Language, and Medical. Despite facing challenges due to limited resources, SiDiaC-v.2.0 serves as a comprehensive resource for Sinhala NLP, building upon the work previously done in SiDiaC-v.1.0. 

---
# PivotAttack: Rethinking the Search Trajectory in Hard-Label Text Attacks via Pivot Words 

**Authors**: Yuzhi Liang, Shiliang Xiao, Jingsong Wei, Qiliang Lin, Xia Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.10842)  

**Abstract**: Existing hard-label text attacks often rely on inefficient "outside-in" strategies that traverse vast search spaces. We propose PivotAttack, a query-efficient "inside-out" framework. It employs a Multi-Armed Bandit algorithm to identify Pivot Sets-combinatorial token groups acting as prediction anchors-and strategically perturbs them to induce label flips. This approach captures inter-word dependencies and minimizes query costs. Extensive experiments across traditional models and Large Language Models demonstrate that PivotAttack consistently outperforms state-of-the-art baselines in both Attack Success Rate and query efficiency. 

---
# Multilingual Reasoning Gym: Multilingual Scaling of Procedural Reasoning Environments 

**Authors**: Konstantin Dobler, Simon Lehnerer, Federico Scozzafava, Jonathan Janke, Mohamed Ali  

**Link**: [PDF](https://arxiv.org/pdf/2603.10793)  

**Abstract**: We present the Multilingual Reasoning Gym, an extension of Reasoning Gym (Stojanovski et al., 2025), that procedurally generates verifiable reasoning problems across 14 languages. We translate templates for 94 tasks with native-speaker validation in 10 languages and targeted code or template adaptations to ensure linguistic naturalness. The Multilingual Reasoning Gym preserves the core benefits of the procedural generation approach used in the original Reasoning Gym, such as virtually unlimited problem instance generation and adjustable difficulty, and remains directly usable for Reinforcement Learning from Verifiable Rewards and evaluation settings. Problems in the Multilingual Reasoning Gym are parallel across languages, enabling crosslingually parallel data generation at massive scale due to the procedural nature of the environments. We release our implementation to support research into multilingual reasoning models. 

---
# LuxBorrow: From Pompier to Pompjee, Tracing Borrowing in Luxembourgish 

**Authors**: Nina Hosseini-Kivanani, Fred Philippy  

**Link**: [PDF](https://arxiv.org/pdf/2603.10789)  

**Abstract**: We present LuxBorrow, a borrowing-first analysis of Luxembourgish (LU) news spanning 27 years (1999-2025), covering 259,305 RTL articles and 43.7M tokens. Our pipeline combines sentence-level language identification (LU/DE/FR/EN) with a token-level borrowing resolver restricted to LU sentences, using lemmatization, a collected loanword registry, and compiled morphological and orthographic rules. Empirically, LU remains the matrix language across all documents, while multilingual practice is pervasive: 77.1% of articles include at least one donor language and 65.4% use three or four. Breadth does not imply intensity: median code-mixing index (CMI) increases from 3.90 (LU+1) to only 7.00 (LU+3), indicating localized insertions rather than balanced bilingual text. Domain and period summaries show moderate but persistent mixing, with CMI rising from 6.1 (1999-2007) to a peak of 8.4 in 2020. Token-level adaptations total 25,444 instances and exhibit a mixed profile: morphological 63.8%, orthographic 35.9%, lexical 0.3%. The most frequent individual rules are orthographic, such as on->oun and eur->er, while morphology is collectively dominant. Diachronically, code-switching intensifies, and morphologically adapted borrowings grow from a small base. French overwhelmingly supplies adapted items, with modest growth for German and negligible English. We advocate borrowing-centric evaluation, including borrowed token and type rates, donor entropy over borrowed items, and assimilation ratios, rather than relying only on document-level mixing indices. 

---
# Interpretable Chinese Metaphor Identification via LLM-Assisted MIPVU Rule Script Generation: A Comparative Protocol Study 

**Authors**: Weihang Huang, Mengna Liu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10784)  

**Abstract**: Metaphor identification is a foundational task in figurative language processing, yet most computational approaches operate as opaque classifiers offering no insight into why an expression is judged metaphorical. This interpretability gap is especially acute for Chinese, where rich figurative traditions, absent morphological cues, and limited annotated resources compound the challenge. We present an LLM-assisted pipeline that operationalises four metaphor identification protocols--MIP/MIPVU lexical analysis, CMDAG conceptual-mapping annotation, emotion-based detection, and simile-oriented identification--as executable, human-auditable rule scripts. Each protocol is a modular chain of deterministic steps interleaved with controlled LLM calls, producing structured rationales alongside every classification decision. We evaluate on seven Chinese metaphor datasets spanning token-, sentence-, and span-level annotation, establishing the first cross-protocol comparison for Chinese metaphor identification. Within-protocol evaluation shows Protocol A (MIP) achieves an F1 of 0.472 on token-level identification, while cross-protocol analysis reveals striking divergence: pairwise Cohen's kappa between Protocols A and D is merely 0.001, whereas Protocols B and C exhibit near-perfect agreement (kappa = 0.986). An interpretability audit shows all protocols achieve 100% deterministic reproducibility, with rationale correctness from 0.40 to 0.87 and editability from 0.80 to 1.00. Error analysis identifies conceptual-domain mismatch and register sensitivity as dominant failure modes. Our results demonstrate that protocol choice is the single largest source of variation in metaphor identification, exceeding model-level variation, and that rule-script architectures achieve competitive performance while maintaining full transparency. 

---
# Large Language Models as Annotators for Machine Translation Quality Estimation 

**Authors**: Sidi Wang, Sophie Arnoult, Amir Kamran  

**Link**: [PDF](https://arxiv.org/pdf/2603.10775)  

**Abstract**: Large Language Models (LLMs) have demonstrated excellent performance on Machine Translation Quality Estimation (MTQE), yet their high inference costs make them impractical for direct application. In this work, we propose applying LLMs to generate MQM-style annotations for training a COMET model: following Fernandes et al. (2023), we reckon that segment-level annotations provide a strong rationale for LLMs and are key to good segment-level QE. We propose a simplified MQM scheme, mostly restricted to top-level categories, to guide LLM selection. We present a systematic approach for the development of a GPT-4o-based prompt, called PPbMQM (Prompt-Pattern-based-MQM). We show that the resulting annotations correlate well with human annotations and that training COMET on them leads to competitive performance on segment-level QE for Chinese-English and English-German. 

---
# Word Recovery in Large Language Models Enables Character-Level Tokenization Robustness 

**Authors**: Zhipeng Yang, Shu Yang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10771)  

**Abstract**: Large language models (LLMs) trained with canonical tokenization exhibit surprising robustness to non-canonical inputs such as character-level tokenization, yet the mechanisms underlying this robustness remain unclear. We study this phenomenon through mechanistic interpretability and identify a core process we term word recovery. We first introduce a decoding-based method to detect word recovery, showing that hidden states reconstruct canonical word-level token identities from character-level inputs. We then provide causal evidence by removing the corresponding subspace from hidden states, which consistently degrades downstream task performance. Finally, we conduct a fine-grained attention analysis and show that in-group attention among characters belonging to the same canonical token is critical for word recovery: masking such attention in early layers substantially reduces both recovery scores and task performance. Together, our findings provide a mechanistic explanation for tokenization robustness and identify word recovery as a key mechanism enabling LLMs to process character-level inputs. 

---
# mAceReason-Math: A Dataset of High-Quality Multilingual Math Problems Ready For RLVR 

**Authors**: Konstantin Dobler, Simon Lehnerer, Federico Scozzafava, Jonathan Janke, Mohamed Ali  

**Link**: [PDF](https://arxiv.org/pdf/2603.10767)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has been successfully applied to significantly boost the capabilities of pretrained large language models, especially in the math and logic problem domains. However, current research and available training datasets remain English-centric. While mul- tilingual training data and benchmarks have been created in the past, they were not created with RLVR and current model capability in mind, and their level of difficulty is often too low to provide appropriate training signals for current models. To address this gap, we provide mAceReason-Math, a dataset of high-quality translations of challenging math problems sourced from a corpus specifically curated for RLVR (AceReason-Math). We further take specific care to clean and improve our translations, resulting in a coverage of 14 languages with more than 10,000 samples per language. We release the dataset to facilitate multilingual RLVR research and benchmarking in the research community. 

---
# HeartAgent: An Autonomous Agent System for Explainable Differential Diagnosis in Cardiology 

**Authors**: Shuang Zhou, Kai Yu, Song Wang, Wenya Xie, Zaifu Zhan, Meng-Han Tsai, Yuen-Hei Chung, Shutong Hou, Huixue Zhou, Min Zeng, Bhavadharini Ramu, Lin Yee Chen, Feng Xie, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10764)  

**Abstract**: Heart diseases remain a leading cause of morbidity and mortality worldwide, necessitating accurate and trustworthy differential diagnosis. However, existing artificial intelligence-based diagnostic methods are often limited by insufficient cardiology knowledge, inadequate support for complex reasoning, and poor interpretability. Here we present HeartAgent, a cardiology-specific agent system designed to support a reliable and explainable differential diagnosis. HeartAgent integrates customized tools and curated data resources and orchestrates multiple specialized sub-agents to perform complex reasoning while generating transparent reasoning trajectories and verifiable supporting references. Evaluated on the MIMIC dataset and a private electronic health records cohort, HeartAgent achieved over 36% and 20% improvements over established comparative methods, in top-3 diagnostic accuracy, respectively. Additionally, clinicians assisted by HeartAgent demonstrated gains of 26.9% in diagnostic accuracy and 22.7% in explanatory quality compared with unaided experts. These results demonstrate that HeartAgent provides reliable, explainable, and clinically actionable decision support for cardiovascular care. 

---
# Prism-$Δ$: Differential Subspace Steering for Prompt Highlighting in Large Language Models 

**Authors**: Yuyao Ge, Shenghua Liu, Yiwei Wang, Tianyu Liu, Baolong Bi, Lingrui Mei, Jiayu Yao, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2603.10705)  

**Abstract**: Prompt highlighting steers a large language model to prioritize user-specified text spans during generation. A key challenge is extracting steering directions that capture the difference between relevant and irrelevant contexts, rather than shared structural patterns common to both. We propose PRISM-$\Delta$ (Projection-based Relevance-Informed Steering Method), which decomposes the difference between positive and negative cross-covariance matrices to maximize discriminative energy while eliminating shared directions. Each attention head receives a continuous softplus importance weight, letting weak-but-useful heads contribute at reduced strength. The framework extends naturally to Value representations, capturing content-channel signal that Key-only methods leave unused. Across four benchmarks and five models, PRISM-$\Delta$ matches or exceeds the best existing method on 19 of 20 configurations, with relative gains up to +10.6%, while halving the fluency cost of steering. PRISM-$\Delta$ also scales to long-context retrieval, outperforming the best existing method by up to +4.8% relative gain. PRISM-$\Delta$ is compatible with FlashAttention and adds negligible memory overhead. 

---
# Making Bielik LLM Reason (Better): A Field Report 

**Authors**: Adam Trybus, Bartosz Bartnicki, Remigiusz Kinas  

**Link**: [PDF](https://arxiv.org/pdf/2603.10640)  

**Abstract**: This paper presents a research program dedicated to evaluating and advancing the reasoning capabilities of Bielik, a Polish large language model. The study describes a number of stages of work: initial benchmarking and creation of evaluation methodology, analyzing of comparative results with other LLMs and outlining of future prospects that take into account the limitations of the analyses conducted so far and aims to keep Bielik in the race give the ever-changing -- and competitive -- AI landscape. 

---
# Disentangling Similarity and Relatedness in Topic Models 

**Authors**: Hanlin Xiao, Mauricio A. Álvarez, Rainer Breitling  

**Link**: [PDF](https://arxiv.org/pdf/2603.10619)  

**Abstract**: The recent advancement of large language models has spurred a growing trend of integrating pre-trained language model (PLM) embeddings into topic models, fundamentally reshaping how topics capture semantic structure. Classical models such as Latent Dirichlet Allocation (LDA) derive topics from word co-occurrence statistics, whereas PLM-augmented models anchor these statistics to pre-trained embedding spaces, imposing a prior that also favours clustering of semantically similar words. This structural difference can be captured by the psycholinguistic dimensions of thematic relatedness and taxonomic similarity of the topic words. To disentangle these dimensions in topic models, we construct a large synthetic benchmark of word pairs using LLM-based annotation to train a neural scoring function. We apply this scorer to a comprehensive evaluation across multiple corpora and topic model families, revealing that different model families capture distinct semantic structure in their topics. We further demonstrate that similarity and relatedness scores successfully predict downstream task performance depending on task requirements. This paper establishes similarity and relatedness as essential axes for topic model evaluation and provides a reliable pipeline for characterising these across model families and corpora. 

---
# MUNIChus: Multilingual News Image Captioning Benchmark 

**Authors**: Yuji Chen, Alistair Plum, Hansi Hettiarachchi, Diptesh Kanojia, Saroj Basnet, Marcos Zampieri, Tharindu Ranasinghe  

**Link**: [PDF](https://arxiv.org/pdf/2603.10613)  

**Abstract**: The goal of news image captioning is to generate captions by integrating news article content with corresponding images, highlighting the relationship between textual context and visual elements. The majority of research on news image captioning focuses on English, primarily because datasets in other languages are scarce. To address this limitation, we create the first multilingual news image captioning benchmark, MUNIChus, comprising 9 languages, including several low-resource languages such as Sinhala and Urdu. We evaluate various state-of-the-art neural news image captioning models on MUNIChus and find that news image captioning remains challenging. We also make MUNIChus publicly available with over 20 models already benchmarked. MUNIChus opens new avenues for further advancements in developing and evaluating multilingual news image captioning models. 

---
# End-to-End Chatbot Evaluation with Adaptive Reasoning and Uncertainty Filtering 

**Authors**: Nhi Dang, Tung Le, Huy Tien Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10570)  

**Abstract**: Large language models (LLMs) combined with retrieval augmented generation have enabled the deployment of domain-specific chatbots, but these systems remain prone to generating unsupported or incorrect answers. Reliable evaluation is therefore critical, yet manual review is costly and existing frameworks often depend on curated test sets and static metrics, limiting scalability. We propose an end-to-end automatic evaluator designed to substantially reduce human effort. Our system generates Q\&A pairs directly from the underlying knowledge base, uses LLMs to judge chatbot responses against reference answers, and applies confidence-based filtering to highlight uncertain cases. Applied to a Vietnamese news dataset, the evaluator achieves high agreement with human judgments while significantly lowering review overhead. The framework is modular and language-agnostic, making it readily adaptable to diverse domains. This work introduces a practical, scalable solution for evaluating chatbots with minimal reliance on manual intervention. 

---
# Automatic End-to-End Data Integration using Large Language Models 

**Authors**: Aaron Steiner, Christian Bizer  

**Link**: [PDF](https://arxiv.org/pdf/2603.10547)  

**Abstract**: Designing data integration pipelines typically requires substantial manual effort from data engineers to configure pipeline components and label training data. While LLMs have shown promise in handling individual steps of the integration process, their potential to replace all human input across end-to-end data integration pipelines has not been investigated. As a step toward exploring this potential, we present an automatic data integration pipeline that uses GPT-5.2 to generate all artifacts required to adapt the pipeline to specific use cases. These artifacts are schema mappings, value mappings for data normalization, training data for entity matching, and validation data for selecting conflict resolution heuristics in data fusion. We compare the performance of this LLM-based pipeline to the performance of human-designed pipelines along three case studies requiring the integration of video game, music, and company related data. Our experiments show that the LLM-based pipeline is able to produce similar results, for some tasks even better results, as the human-designed pipelines. End-to-end, the human and the LLM pipelines produce integrated datasets of comparable size and density. Having the LLM configure the pipelines costs approximately \$10 per case study, which represents only a small fraction of the cost of having human data engineers perform the same tasks. 

---
# AILS-NTUA at SemEval-2026 Task 8: Evaluating Multi-Turn RAG Conversations 

**Authors**: Dimosthenis Athanasiou, Maria Lymperaiou, Giorgos Filandrianos, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2603.10524)  

**Abstract**: We present the AILS-NTUA system for SemEval-2026 Task 8 (MTRAGEval), addressing all three subtasks of multi-turn retrieval-augmented generation: passage retrieval (A), reference-grounded response generation (B), and end-to-end RAG (C). Our unified architecture is built on two principles: (i) a query-diversity-over-retriever-diversity strategy, where five complementary LLM-based query reformulations are issued to a single corpus-aligned sparse retriever and fused via variance-aware nested Reciprocal Rank Fusion; and (ii) a multistage generation pipeline that decomposes grounded generation into evidence span extraction, dual-candidate drafting, and calibrated multi-judge selection. Our system ranks 1st in Task A (nDCG@5: 0.5776, +20.5% over the strongest baseline) and 2nd in Task B (HM: 0.7698). Empirical analysis shows that query diversity over a well-aligned retriever outperforms heterogeneous retriever ensembling, and that answerability calibration-rather than retrieval coverage-is the primary bottleneck in end-to-end performance. 

---
# Safe and Scalable Web Agent Learning via Recreated Websites 

**Authors**: Hyungjoo Chae, Jungsoo Park, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2603.10505)  

**Abstract**: Training autonomous web agents is fundamentally limited by the environments they learn from: real-world websites are unsafe to explore, hard to reset, and rarely provide verifiable feedback. We propose VeriEnv, a framework that treats language models as environment creators, automatically cloning real-world websites into fully executable, verifiable synthetic environments. By exposing controlled internal access via a Python SDK, VeriEnv enables agents to self-generate tasks with deterministic, programmatically verifiable rewards, eliminating reliance on heuristic or LLM-based judges. This design decouples agent learning from unsafe real-world interaction while enabling scalable self-evolution through environment expansion. Through experiments on web agent benchmarks, we show that agents trained with VeriEnv generalize to unseen websites, achieve site-specific mastery through self-evolving training, and benefit from scaling the number of training environments. Code and resources will be released at this https URL upon acceptance. 

---
# VERI-DPO: Evidence-Aware Alignment for Clinical Summarization via Claim Verification and Direct Preference Optimization 

**Authors**: Weixin Liu, Congning Ni, Qingyuan Song, Susannah L. Rose, Christopher Symons, Murat Kantarcioglu, Bradley A. Malin, Zhijun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2603.10494)  

**Abstract**: Brief Hospital Course (BHC) narratives must be clinically useful yet faithful to fragmented EHR evidence. LLM-based clinical summarizers still introduce unsupported statements, and alignment can encourage omissions ("say-less" degeneration). We introduce VERI-DPO, which uses claim verification to mine preferences and distill them into the summarizer with Direct Preference Optimization (DPO). On MIMIC-III-Ext-VeriFact-BHC (100 ICU patients; patient-level splits), we train a retrieval-augmented verifier to label claim-evidence pairs as Supported, Not Supported, or Not Addressed via a single-token format. The verifier scores sentence-level claims from sampled BHC candidates and aggregates margins into a coverage-aware utility to mine length-controlled, contradiction-anchored preference pairs. On held-out patients, verifier-mined preferences separate candidates by contradiction density, and VERI-DPO reduces Not Supported claim rates from 10.7% to 1.9% (local verifier judge) and from 11.6% to 6.4% (GPT-4o judge), while improving validity from 76.7% to 82.5% and maintaining informative length. 

---
# Human-AI Co-reasoning for Clinical Diagnosis with Evidence-Integrated Language Agent 

**Authors**: Zhongzhen Huang, Yan Ling, Hong Chen, Ye Feng, Li Wu, Linjie Mu, Shaoting Zhang, Xiaofan Zhang, Kun Qian, Xiaomu Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.10492)  

**Abstract**: We present PULSE, a medical reasoning agent that combines a domain-tuned large language model with scientific literature retrieval to support diagnostic decision-making in complex real-world cases. To evaluate its capabilities, we curated a benchmark of 82 authentic endocrinology case reports encompassing a broad spectrum of disease types and incidence levels. In controlled experiments, we compared PULSE's performance against physicians with varying levels of expertise-from residents to senior specialists-and examined how AI assistance influenced human diagnostic reasoning. PULSE attained expert-competitive accuracy, outperforming residents and junior specialists while matching senior specialist performance at both Top@1 and Top@4 thresholds. Unlike physicians, whose accuracy declined with disease rarity, PULSE maintained stable performance across incidence tiers. The agent also exhibited adaptive reasoning, increasing output length with case difficulty in a manner analogous to the longer deliberation observed among expert clinicians. When used collaboratively, PULSE enabled physicians to correct initial errors and broaden diagnostic hypotheses, but also introduced risks of automation bias. The study explores both serial and concurrent collaboration workflows, revealing that PULSE offers robust support across common and rare presentations. These findings underscore both the promise and the limitations of language model-based agents in clinical diagnosis, and offer a framework for evaluating their role in real-world decision-making. 

---
# PEEM: Prompt Engineering Evaluation Metrics for Interpretable Joint Evaluation of Prompts and Responses 

**Authors**: Minki Hong, Eunsoo Lee, Sohyun Park, Jihie Kim  

**Link**: [PDF](https://arxiv.org/pdf/2603.10477)  

**Abstract**: Prompt design is a primary control interface for large language models (LLMs), yet standard evaluations largely reduce performance to answer correctness, obscuring why a prompt succeeds or fails and providing little actionable guidance. We propose PEEM (Prompt Engineering Evaluation Metrics), a unified framework for joint and interpretable evaluation of both prompts and responses. PEEM defines a structured rubric with 9 axes: 3 prompt criteria (clarity/structure, linguistic quality, fairness) and 6 response criteria (accuracy, coherence, relevance, objectivity, clarity, conciseness), and uses an LLM-based evaluator to output (i) scalar scores on a 1-5 Likert scale and (ii) criterion-specific natural-language rationales grounded in the rubric. Across 7 benchmarks and 5 task models, PEEM's accuracy axis strongly aligns with conventional accuracy while preserving model rankings (aggregate Spearman rho about 0.97, Pearson r about 0.94, p < 0.001). A multi-evaluator study with four models shows consistent relative judgments (pairwise rho = 0.68-0.85), supporting evaluator-agnostic deployment. Beyond alignment, PEEM captures complementary linguistic failure modes and remains informative under prompt perturbations: prompt-quality trends track downstream accuracy under iterative rewrites, semantic adversarial manipulations induce clear score degradation, and meaning-preserving paraphrases yield high stability (robustness rate about 76.7-80.6%). Finally, using only PEEM scores and rationales as feedback, a zero-shot prompt rewriting loop improves downstream accuracy by up to 11.7 points, outperforming supervised and RL-based prompt-optimization baselines. Overall, PEEM provides a reproducible, criterion-driven protocol that links prompt formulation to response behavior and enables systematic diagnosis and optimization of LLM interactions. 

---
# Learning to Negotiate: Multi-Agent Deliberation for Collective Value Alignment in LLMs 

**Authors**: Panatchakorn Anantaprayoon, Nataliia Babina, Nima Asgharbeygi, Jad Tarifi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10476)  

**Abstract**: The alignment of large language models (LLMs) has progressed substantially in single-agent settings through paradigms such as RLHF and Constitutional AI, with recent work exploring scalable alternatives such as RLAIF and evolving alignment objectives. However, these approaches remain limited in multi-stakeholder settings, where conflicting values arise and deliberative negotiation capabilities are required. This work proposes a multi-agent negotiation-based alignment framework that aligns LLMs to Collective Agency (CA)-an existing alignment objective introduced to promote the continual expansion of agency-while simultaneously improving conflict-resolution capability. To enable scalable training, two self-play instances of the same LLM, assigned opposing personas, engage in structured turn-based dialogue to synthesize mutually beneficial solutions. We generate synthetic moral-dilemma prompts and conflicting persona pairs, and optimize the policy via RLAIF using GRPO with an external LLM reward model. While rewards are computed from CA scores assigned to the final completion, gradients are applied to dialogue tokens to directly improve deliberative interaction dynamics. Experiments show that the resulting model achieves CA alignment comparable to a single-agent baseline while substantially improving conflict-resolution performance without degrading general language capabilities. These results suggest that negotiation-driven deliberation training provides a practical path toward LLMs that better support collective decision-making in value-conflict scenarios. 

---
# Aligning Large Language Models with Searcher Preferences 

**Authors**: Wei Wu, Peilun Zhou, Liyi Chen, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, Yao Hu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2603.10473)  

**Abstract**: The paradigm shift from item-centric ranking to answer-centric synthesis is redefining the role of search engines. While recent industrial progress has applied generative techniques to closed-set item ranking in e-commerce, research and deployment of open-ended generative search on large content platforms remain limited. This setting introduces challenges, including robustness to noisy retrieval, non-negotiable safety guarantees, and alignment with diverse user needs. In this work, we introduce SearchLLM, the first large language model (LLM) for open-ended generative search. We design a hierarchical, multi-dimensional reward system that separates bottom-line constraints, including factual grounding, basic answer quality and format compliance, from behavior optimization objectives that promote robustness to noisy retrieval and alignment with user needs. Concretely, our reward model evaluates responses conditioned on the user query, session history, and retrieved evidence set, combining rule-based checks with human-calibrated LLM judges to produce an interpretable score vector over these dimensions. We introduce a Gated Aggregation Strategy to derive the training reward for optimizing SearchLLM with Group Relative Policy Optimization (GRPO). We deploy SearchLLM in the AI search entry of RedNote. Offline evaluations and online A/B tests show improved generation quality and user engagement, increasing Valid Consumption Rate by 1.03% and reducing Re-search Rate by 2.81%, while upholding strict safety and reliability standards. 

---
# Dynamic Knowledge Fusion for Multi-Domain Dialogue State Tracking 

**Authors**: Haoxiang Su, Ruiyu Fang, Liting Jiang, Xiaomeng Huang, Shuangyong Song  

**Link**: [PDF](https://arxiv.org/pdf/2603.10367)  

**Abstract**: The performance of task-oriented dialogue models is strongly tied to how well they track dialogue states, which records and updates user information across multi-turn interactions. However, current multi-domain DST encounters two key challenges: the difficulty of effectively modeling dialogue history and the limited availability of annotated data, both of which hinder model performance. To tackle the aforementioned problems, we develop a dynamic knowledge fusion framework applicable to multi-domain DST. The model operates in two stages: first, an encoder-only network trained with contrastive learning encodes dialogue history and candidate slots, selecting relevant slots based on correlation scores; second, dynamic knowledge fusion leverages the structured information of selected slots as contextual prompts to enhance the accuracy and consistency of dialogue state tracking. This design enables more accurate integration of dialogue context and domain knowledge. Results obtained from multi-domain dialogue benchmarks indicate that our method notably improves both tracking accuracy and generalization, validating its capability in handling complex dialogue scenarios. 

---
# Mitigating Translationese Bias in Multilingual LLM-as-a-Judge via Disentangled Information Bottleneck 

**Authors**: Hongbin Zhang, Kehai Chen, Xuefen Bai, Youcheng Pan, Yang Xiang, Jinpeng Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10351)  

**Abstract**: Large language models (LLMs) have become a standard for multilingual evaluation, yet they exhibit a severe systematic translationese bias. In this paper, translationese bias is characterized as LLMs systematically favoring machine-translated text over human-authored references, particularly in low-resource languages. We attribute this bias to spurious correlations with (i) latent manifold alignment with English and (ii) cross-lingual predictability. To mitigate this bias, we propose DIBJudge, a robust fine-tuning framework that learns a minimally sufficient, judgment-critical representation via variational information compression, while explicitly isolating spurious factors into the dedicated bias branch. Furthermore, we incorporate a cross-covariance penalty that explicitly suppresses statistical dependence between robust and bias representations, thereby encouraging effective disentanglement. Extensive evaluations on multilingual reward modeling benchmarks and a dedicated translationese bias evaluation suite demonstrate that the proposed DIBJudge consistently outperforms strong baselines and substantially mitigates translationese bias. 

---
# Large language models can disambiguate opioid slang on social media 

**Authors**: Kristy A. Carpenter, Issah A. Samori, Mathew V. Kiang, Keith Humphreys, Anna Lembke, Johannes C. Eichstaedt, Russ B. Altman  

**Link**: [PDF](https://arxiv.org/pdf/2603.10313)  

**Abstract**: Social media text shows promise for monitoring trends in the opioid overdose crisis; however, the overwhelming majority of social media text is unrelated to opioids. When leveraging social media text to monitor trends in the ongoing opioid overdose crisis, a common strategy for identifying relevant content is to use a lexicon of opioid-related terms as inclusion criteria. However, many slang terms for opioids, such as "smack" or "blues," have common non-opioid meanings, making them ambiguous. The advanced textual reasoning capability of large language models (LLMs) presents an opportunity to disambiguate these slang terms at scale. We present three tasks on which to evaluate four state-of-the-art LLMs (GPT-4, GPT-5, Gemini 2.5 Pro, and Claude Sonnet 4.5): a lexicon-based setting, in which the LLM must disambiguate a specific term within the context of a given post; a lexicon-free setting, in which the LLM must identify opioid-related posts from context without a lexicon; and an emergent slang setting, in which the LLM must identify opioid-related posts with simulated new slang terms. All four LLMs showed excellent performance across all tasks. In both subtasks of the lexicon-based setting, LLM F1 scores ("fenty" subtask: 0.824-0.972; "smack" subtask: 0.540-0.862) far exceeded those of the best lexicon strategy (0.126 and 0.009, respectively). In the lexicon-free task, LLM F1 scores (0.544-0.769) surpassed those of lexicons (0.080-0.540), and LLMs demonstrated uniformly higher recall. On emergent slang, all LLMs had higher accuracy (average: 0.784), F1 score (average: 0.712), precision (average: 0.981), and recall (average: 0.587) than the two lexicons assessed. Our results show that LLMs can be used to identify relevant content for low-prevalence topics, including but not limited to opioid references, enhancing data provided to downstream analyses and predictive models. 

---
# Is this Idea Novel? An Automated Benchmark for Judgment of Research Ideas 

**Authors**: Tim Schopf, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2603.10303)  

**Abstract**: Judging the novelty of research ideas is crucial for advancing science, enabling the identification of unexplored directions, and ensuring contributions meaningfully extend existing knowledge rather than reiterate minor variations. However, given the exponential growth of scientific literature, manually judging the novelty of research ideas through literature reviews is labor-intensive, subjective, and infeasible at scale. Therefore, recent efforts have proposed automated approaches for research idea novelty judgment. Yet, evaluation of these approaches remains largely inconsistent and is typically based on non-standardized human evaluations, hindering large-scale, comparable evaluations. To address this, we introduce RINoBench, the first comprehensive benchmark for large-scale evaluation of research idea novelty judgments. It comprises 1,381 research ideas derived from and judged by human experts as well as nine automated evaluation metrics designed to assess both rubric-based novelty scores and textual justifications of novelty judgments. Using this benchmark, we evaluate several state-of-the-art large language models (LLMs) on their ability to judge the novelty of research ideas. Our findings reveal that while LLM-generated reasoning closely mirrors human rationales, this alignment does not reliably translate into accurate novelty judgments, which diverge significantly from human gold standard judgments - even among leading reasoning-capable models. Data and code available at: this https URL. 

---
# GR-SAP: Generative Replay for Safety Alignment Preservation during Fine-Tuning 

**Authors**: Zhouxiang Fang, Jiawei Zhou, Hanjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10243)  

**Abstract**: Recent studies show that the safety alignment of large language models (LLMs) can be easily compromised even by seemingly non-adversarial fine-tuning. To preserve safety alignment during fine-tuning, a widely used strategy is to jointly optimize safety and task objectives by mixing in the original alignment data, which is typically inaccessible even for open-weight LLMs. Inspired by generative replay in continual learning, we propose Generative Replay for Safety Alignment Preservation (GR-SAP), a unified framework that synthesizes domain-specific alignment data from LLMs and integrate them during downstream adaption to preserve safety alignment. Theoretical and empirical analyses demonstrate this synthetic data serves as a reliable proxy for the original alignment data. Experiments across various models and downstream tasks show that GR-SAP substantially mitigates fine-tuning-induced safety degradation while maintaining comparable downstream performance. Our code is available at this https URL. 

---
# S-GRADES -- Studying Generalization of Student Response Assessments in Diverse Evaluative Settings 

**Authors**: Tasfia Seuti, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.10233)  

**Abstract**: Evaluating student responses, from long essays to short factual answers, is a key challenge in educational NLP. Automated Essay Scoring (AES) focuses on holistic writing qualities such as coherence and argumentation, while Automatic Short Answer Grading (ASAG) emphasizes factual correctness and conceptual understanding. Despite their shared goal, these paradigms have progressed in isolation with fragmented datasets, inconsistent metrics, and separate communities. We introduce S-GRADES (Studying Generalization of Student Response Assessments in Diverse Evaluative Settings), a web-based benchmark that consolidates 14 diverse grading datasets under a unified interface with standardized access and reproducible evaluation protocols. The benchmark is fully open-source and designed for extensibility, enabling continuous integration of new datasets and evaluation settings. To demonstrate the utility of S-GRADES, we evaluate three state-of-the-art large language models across the benchmark using multiple reasoning strategies in prompting. We further examine the effects of exemplar selection and cross-dataset exemplar transfer. Our analyses illustrate how benchmark-driven evaluation reveals reliability and generalization gaps across essay and short-answer grading tasks, highlighting the importance of standardized, cross-paradigm assessment. 

---
# Sabiá-4 Technical Report 

**Authors**: Thiago Laitz, Thales Sales Almeida, Hugo Abonizio, Roseval Malaquias Junior, Giovana Kerche Bonás, Marcos Piau, Celio Larcher, Ramon Pires, Rodrigo Nogueira  

**Link**: [PDF](https://arxiv.org/pdf/2603.10213)  

**Abstract**: This technical report presents Sabiá-4 and Sabiazinho-4, a new generation of Portuguese language models with a focus on Brazilian Portuguese language. The models were developed through a four-stage training pipeline: continued pre-training on Portuguese and Brazilian legal corpora, long-context extension to 128K tokens, supervised fine-tuning on instruction data spanning chat, code, legal tasks, and function calling, and preference alignment. We evaluate the models on six benchmark categories: conversational capabilities in Brazilian Portuguese, knowledge of Brazilian legislation, long-context understanding, instruction following, standardized exams, and agentic capabilities including tool use and web navigation. Results show that Sabiá-4 and Sabiazinho-4 achieve a favorable cost-performance trade-off compared to other models, positioning them in the upper-left region of the pricing-accuracy chart. The models show improvements over previous generations in legal document drafting, multi-turn dialogue quality, and agentic task completion. 

---
# ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation 

**Authors**: Khoa Anh Ta, Nguyen Van Dinh, Kiet Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10211)  

**Abstract**: Vietnamese exhibits extensive dialectal variation, posing challenges for NLP systems trained predominantly on standard Vietnamese. Such systems often underperform on dialectal inputs, especially from underrepresented Central and Southern regions. Previous work on dialect normalization has focused narrowly on Central-to-Northern dialect transfer using synthetic data and limited dialectal diversity. These efforts exclude Southern varieties and intra-regional variants within the North. We introduce ViDia2Std, the first manually annotated parallel corpus for dialect-to-standard Vietnamese translation covering all 63 provinces. Unlike prior datasets, ViDia2Std includes diverse dialects from Central, Southern, and non-standard Northern regions often absent from existing resources, making it the most dialectally inclusive corpus to date. The dataset consists of over 13,000 sentence pairs sourced from real-world Facebook comments and annotated by native speakers across all three dialect regions. To assess annotation consistency, we define a semantic mapping agreement metric that accounts for synonymous standard mappings across annotators. Based on this criterion, we report agreement rates of 86% (North), 82% (Central), and 85% (South). We benchmark several sequence-to-sequence models on ViDia2Std. mBART-large-50 achieves the best results (BLEU 0.8166, ROUGE-L 0.9384, METEOR 0.8925), while ViT5-base offers competitive performance with fewer parameters. ViDia2Std demonstrates that dialect normalization substantially improves downstream tasks, highlighting the need for dialect-aware resources in building robust Vietnamese NLP systems. 

---
# Adaptive Activation Cancellation for Hallucination Mitigation in Large Language Models 

**Authors**: Eric Yocam, Varghese Vaidyan, Gurcan Comert, Paris Kalathas, Yong Wang, Judith L. Mwakalonge  

**Link**: [PDF](https://arxiv.org/pdf/2603.10195)  

**Abstract**: Large Language Models frequently generate fluent but factually incorrect text. We propose Adaptive Activation Cancellation (AAC), a real-time inference-time framework that treats hallucination-associated neural activations as structured interference within the transformer residual stream, drawing an explicit analogy to classical adaptive noise cancellation from signal processing. The framework identifies Hallucination Nodes (H-Nodes) via layer-wise linear probing and suppresses them using a confidence-weighted forward hook during auto-regressive generation -- requiring no external knowledge, no fine-tuning, and no additional inference passes. Evaluated across OPT-125M, Phi-3-mini, and LLaMA 3-8B on TruthfulQA and HaluEval, the real-time hook is the only intervention that consistently improves downstream accuracy on all three scales. Critically, the method is strictly surgical: WikiText-103 perplexity and MMLU reasoning accuracy are preserved at exactly 0.0% degradation across all three model scales, a property that distinguishes AAC from interventions that trade fluency or general capability for factual improvement. On the LLaMA 3-8B scale, the hook additionally yields positive generation-level gains (MC1 +0.04; MC2 +0.003; Token-F1 +0.003) while achieving probe-space selectivity 5.94x - 3.5x higher than the ITI baseline -- demonstrating that targeted neuron-level suppression can simultaneously improve factual accuracy and preserve model capability. 

---
# OpenClaw-RL: Train Any Agent Simply by Talking 

**Authors**: Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, Ling Yang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10165)  

**Abstract**: Every agent interaction generates a next-state signal, namely the user reply, tool output, terminal or GUI state change that follows each action, yet no existing agentic RL system recovers it as a live, online learning source. We present OpenClaw-RL, a framework built on a simple observation: next-state signals are universal, and policy can learn from all of them simultaneously. Personal conversations, terminal executions, GUI interactions, SWE tasks, and tool-call traces are not separate training problems. They are all interactions that can be used to train the same policy in the same loop. Next-state signals encode two forms of information: evaluative signals, which indicate how well the action performed and are extracted as scalar rewards via a PRM judge; and directive signals, which indicate how the action should have been different and are recovered through Hindsight-Guided On-Policy Distillation (OPD). We extract textual hints from the next state, construct an enhanced teacher context, and provide token-level directional advantage supervision that is richer than any scalar reward. Due to the asynchronous design, the model serves live requests, the PRM judges ongoing interactions, and the trainer updates the policy at the same time, with zero coordination overhead between them. Applied to personal agents, OpenClaw-RL enables an agent to improve simply by being used, recovering conversational signals from user re-queries, corrections, and explicit feedback. Applied to general agents, the same infrastructure supports scalable RL across terminal, GUI, SWE, and tool-call settings, where we additionally demonstrate the utility of process rewards. Code: this https URL 

---
# Lost in Backpropagation: The LM Head is a Gradient Bottleneck 

**Authors**: Nathan Godey, Yoav Artzi  

**Link**: [PDF](https://arxiv.org/pdf/2603.10145)  

**Abstract**: The last layer of neural language models (LMs) projects output features of dimension $D$ to logits in dimension $V$, the size of the vocabulary, where usually $D \ll V$. This mismatch is known to raise risks of limited expressivity in neural LMs, creating a so-called softmax bottleneck. We show the softmax bottleneck is not only an expressivity bottleneck but also an optimization bottleneck. Backpropagating $V$-dimensional gradients through a rank-$D$ linear layer induces unavoidable compression, which alters the training feedback provided to the vast majority of the parameters. We present a theoretical analysis of this phenomenon and measure empirically that 95-99% of the gradient norm is suppressed by the output layer, resulting in vastly suboptimal update directions. We conduct controlled pretraining experiments showing that the gradient bottleneck makes trivial patterns unlearnable, and drastically affects the training dynamics of LLMs. We argue that this inherent flaw contributes to training inefficiencies at scale independently of the model architecture, and raises the need for new LM head designs. 

---
# Reason and Verify: A Framework for Faithful Retrieval-Augmented Generation 

**Authors**: Eeham Khan, Luis Rodriguez, Marc Queudot  

**Link**: [PDF](https://arxiv.org/pdf/2603.10143)  

**Abstract**: Retrieval-Augmented Generation (RAG) significantly improves the factuality of Large Language Models (LLMs), yet standard pipelines often lack mechanisms to verify inter- mediate reasoning, leaving them vulnerable to hallucinations in high-stakes domains. To address this, we propose a domain-specific RAG framework that integrates explicit rea- soning and faithfulness verification. Our architecture augments standard retrieval with neural query rewriting, BGE-based cross-encoder reranking, and a rationale generation module that grounds sub-claims in specific evidence spans. We further introduce an eight-category verification taxonomy that enables fine-grained assessment of rationale faithfulness, distinguishing between explicit and implicit support patterns to facilitate structured error diagnosis. We evaluate this framework on the BioASQ and PubMedQA benchmarks, specifically analyzing the impact of dynamic in-context learning and rerank- ing under constrained token budgets. Experiments demonstrate that explicit rationale generation improves accuracy over vanilla RAG baselines, while dynamic demonstration selection combined with robust reranking yields further gains in few-shot settings. Using Llama-3-8B-Instruct, our approach achieves 89.1% on BioASQ-Y/N and 73.0% on Pub- MedQA, competitive with systems using significantly larger models. Additionally, we perform a pilot study combining human expert assessment with LLM-based verification to explore how explicit rationale generation improves system transparency and enables more detailed diagnosis of retrieval failures in biomedical question answering. 

---
# The Generation-Recognition Asymmetry: Six Dimensions of a Fundamental Divide in Formal Language Theory 

**Authors**: Romain Peyrichou  

**Link**: [PDF](https://arxiv.org/pdf/2603.10139)  

**Abstract**: Every formal grammar defines a language and can in principle be used in three ways: to generate strings (production), to recognize them (parsing), or -- given only examples -- to infer the grammar itself (grammar induction). Generation and recognition are extensionally equivalent -- they characterize the same set -- but operationally asymmetric in multiple independent ways. Inference is a qualitatively harder problem: it does not have access to a known grammar. Despite the centrality of this triad to compiler design, natural language processing, and formal language theory, no survey has treated it as a unified, multidimensional phenomenon. We identify six dimensions along which generation and recognition diverge: computational complexity, ambiguity, directionality, information availability, grammar inference, and temporality. We show that the common characterization "generation is easy, parsing is hard" is misleading: unconstrained generation is trivial, but generation under constraints can be NP-hard. The real asymmetry is that parsing is always constrained (the input is given) while generation need not be. Two of these dimensions -- directionality and temporality -- have not previously been identified as dimensions of the generation-recognition asymmetry. We connect the temporal dimension to the surprisal framework of Hale (2001) and Levy (2008), arguing that surprisal formalizes the temporal asymmetry between a generator (surprisal = 0) and a parser that predicts under uncertainty (surprisal > 0). We review bidirectional systems in NLP and observe that bidirectionality has been available for fifty years yet has not transferred to most domain-specific applications. We conclude with a discussion of large language models, which architecturally unify generation and recognition while operationally preserving the asymmetry. 

---
# The Prediction-Measurement Gap: Toward Meaning Representations as Scientific Instruments 

**Authors**: Hubert Plisiecki  

**Link**: [PDF](https://arxiv.org/pdf/2603.10130)  

**Abstract**: Text embeddings have become central to computational social science and psychology, enabling scalable measurement of meaning and mixed-method inference. Yet most representation learning is optimized and evaluated for prediction and retrieval, yielding a prediction-measurement gap: representations that perform well as features may be poorly suited as scientific instruments. The paper argues that scientific meaning analysis motivates a distinct family of objectives - scientific usability - emphasizing geometric legibility, interpretability and traceability to linguistic evidence, robustness to non-semantic confounds, and compatibility with regression-style inference over semantic directions. Grounded in cognitive and neuro-psychological views of meaning, the paper assesses static word embeddings and contextual transformer representations against these requirements: static spaces remain attractive for transparent measurement, whereas contextual spaces offer richer semantics but entangle meaning with other signals and exhibit geometric and interpretability issues that complicate inference. The paper then outlines a course-setting agenda around (i) geometry-first design for gradients and abstraction, including hierarchy-aware spaces constrained by psychologically privileged levels; (ii) invertible post-hoc transformations that recondition embedding geometry and reduce nuisance influence; and (iii) meaning atlases and measurement-oriented evaluation protocols for reliable and traceable semantic inference. As the field debates the limits of scale-first progress, measurement-ready representations offer a principled new frontier. 

---
# TriageSim: A Conversational Emergency Triage Simulation Framework from Structured Electronic Health Records 

**Authors**: Dipankar Srirag, Quoc Dung Nguyen, Aditya Joshi, Padmanesan Narasimhan, Salil Kanhere  

**Link**: [PDF](https://arxiv.org/pdf/2603.10035)  

**Abstract**: Research in emergency triage is restricted to structured electronic health records (EHR) due to regulatory constraints on nurse-patient interactions. We introduce TriageSim, a simulation framework for generating persona-conditioned triage conversations from structured records. TriageSim enables multi-turn nurse-patient interactions with explicit control over disfluency and decision behaviour, producing a corpus of ~800 synthetic transcripts and corresponding audio. We use a combination of automated analysis for linguistic, behavioural and acoustic fidelity alongside manual evaluation for medical fidelity using a random subset of 50 conversations. The utility of the generated corpus is examined via conversational triage classification. We observe modest agreement for acuity levels across three modalities: generated synthetic text, ASR transcripts, and direct audio inputs. The code, persona schemata and triage policy prompts for TriageSim will be available upon acceptance. 

---
# A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment 

**Authors**: Jiyue Jiang, Yanyu Chen, Pengan Chen, Kai Liu, Jingqi Zhou, Zheyong Zhu, He Hu, Fei Ma, Qi Tian, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10034)  

**Abstract**: Cognitive impairment is becoming a major public health challenge. Cognitive Stimulation Therapy (CST) is an effective intervention for cognitive impairment, but traditional methods are difficult to scale, and existing digital systems struggle with group dialogues and cognitive stimulation principles. While Large Language Models (LLMs) are powerful, their application in this context faces key challenges: cognitive stimulation dialogue paradigms, a lack of therapeutic reasoning, and static-only user modeling. To address these issues, we propose a principle-driven adaptive policy actualized through a Group Cognitive Stimulation Dialogue (GCSD) system. We first construct a dataset with over 500 hours of real-world CST conversations and 10,000+ simulated dialogues generated via our Principle-Guided Scenario Simulation strategy. Our GCSD system then integrates four core modules to overcome LLM limitations: (i) a multi-speaker context controller to resolve role confusion; (ii) dynamic participant cognitive state modeling for personalized interaction; (iii) a cognitive stimulation-focused attention loss to instill cognitive stimulation reasoning; and (iv) a multi-dimensional reward strategy to enhance response value. Experimental results demonstrate that GCSD significantly outperforms baseline models across various evaluation metrics. Future work will focus on long-term clinical validation to bridge the gap between computational performance and clinical efficacy. 

---
# Evaluating Progress in Graph Foundation Models: A Comprehensive Benchmark and New Insights 

**Authors**: Xingtong Yu, Shenghua Ye, Ruijuan Liang, Chang Zhou, Hong Cheng, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10033)  

**Abstract**: Graph foundation models (GFM) aim to acquire transferable knowledge by pre-training on diverse graphs, which can be adapted to various downstream tasks. However, domain shift in graphs is inherently two-dimensional: graphs differ not only in what they describe (topic domains) but also in how they are represented (format domains). Most existing GFM benchmarks vary only topic domains, thereby obscuring how knowledge transfers across both dimensions. We present a new benchmark that jointly evaluates topic and format gaps across the full GFM pipeline, including multi-domain self-supervised pre-training and few-shot downstream adaptation, and provides a timely evaluation of recent GFMs in the rapidly evolving landscape. Our protocol enables controlled assessment in four settings: (i) pre-training on diverse topics and formats, while adapting to unseen downstream datasets; (ii) same pre-training as in (i), while adapting to seen datasets; (iii) pre-training on a single topic domain, while adapting to other topics; (iv) pre-training on a base format, while adapting to other formats. This two-axis evaluation disentangles semantic generalization from robustness to representational shifts. We conduct extensive evaluations of eight state-of-the-art GFMs on 33 datasets spanning seven topic domains and six format domains, surfacing new empirical observations and practical insights for future research. Codes/data are available at this https URL. 

---
# Measuring and Eliminating Refusals in Military Large Language Models 

**Authors**: Jack FitzGerald, Dylan Bates, Aristotelis Lazaridis, Aman Sharma, Vincent Lu, Brian King, Yousif Azami, Sean Bailey, Jeremy Cao, Peter Damianov, Kevin de Haan, Joseph Madigan, Jeremy McLaurin, Luke Kerbs, Jonathan Tainer, Dave Anderson, Jonathan Beck, Jamie Cuticello, Colton Malkerson, Tyler Saltsman  

**Link**: [PDF](https://arxiv.org/pdf/2603.10012)  

**Abstract**: Military Large Language Models (LLMs) must provide accurate information to the warfighter in time-critical and dangerous situations. However, today's LLMs are imbued with safety behaviors that cause the LLM to refuse many legitimate queries in the military domain, particularly those related to violence, terrorism, or military technology. Our gold benchmark for assessing refusal rates, which was developed by veterans of the US Army and special forces, is to our knowledge the first dataset of its kind. We present results for refusal and deflection rates on 31 public models and 3 military models. We observe hard rejection rates as high as 98.2% and soft deflection rates ranging from 0% to 21.3%. We also present results on two additional synthetic datasets and show their correlations with the gold dataset. Finally, we perform abliteration using the Heretic library on a military-tuned gpt-oss-20b model, showing an absolute increase in answer rate of 66.5 points but an average relative decrease of 2% on other military tasks. In our concluding remarks, we argue for deeper specialization, including with mid-training and end-to-end post-training, to achieve zero refusals and maximum military task accuracy for closed military models. 

---
# Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs 

**Authors**: Anna Soligo, Vladimir Mikulik, William Saunders  

**Link**: [PDF](https://arxiv.org/pdf/2603.10011)  

**Abstract**: Large language models can generate responses that resemble emotional distress, and this raises concerns around model reliability and safety. We introduce a set of evaluations to investigate expressions of distress in LLMs, and find that these surface emotional instability in Gemma and Gemini models, but not in other families. We find evidence that this difference arises in post-training. Base models from different families (Gemma, Qwen and OLMo) show similar propensities for expressing distress. However, instruct-tuned Gemma expresses substantially more distress than its base model, whereas instruct-tuned Qwen and OLMo express less. We find a simple mitigation for this: direct preference optimisation on just 280 preference pairs reduces Gemma's high-frustration responses from 35% to 0.3% in our evaluations, generalising across question types, user tones, and conversation lengths, without affecting capabilities. These findings show that emotional instability is an issue in some LLMs. We present (1) evaluations to track this behaviour, and (2) a mitigation without downsides in Gemma, with the caveat that upstream training modifications to improve emotional robustness would be significantly better than this post-hoc fix. 

---
# FERRET: Framework for Expansion Reliant Red Teaming 

**Authors**: Ninareh Mehrabi, Vitor Albiero, Maya Pavlova, Joanna Bitton  

**Link**: [PDF](https://arxiv.org/pdf/2603.10010)  

**Abstract**: We introduce a multi-faceted automated red teaming framework in which the goal is to generate multi-modal adversarial conversations that would break a target model and introduce various expansions that would result in more effective and efficient adversarial conversations. The introduced expansions include: 1. Horizontal expansion in which the goal is for the red team model to self-improve and generate more effective conversation starters that would shape a conversation. 2. Vertical expansion in which the goal is to take these conversation starters that are discovered in the horizontal expansion phase and expand them into effective multi-modal conversations and 3. Meta expansion in which the goal is for the red team model to discover more effective multi-modal attack strategies during the course of a conversation. We call our framework FERRET (Framework for Expansion Reliant Red Teaming) and compare it with various existing automated red teaming approaches. In our experiments, we demonstrate the effectiveness of FERRET in generating effective multi-modal adversarial conversations and its superior performance against existing state of the art approaches. 

---
# GATech at AbjadMed: Bidirectional Encoders vs. Causal Decoders: Insights from 82-Class Arabic Medical Classification 

**Authors**: Ahmed Khaled Khamis  

**Link**: [PDF](https://arxiv.org/pdf/2603.10008)  

**Abstract**: This paper presents system description for Arabic medical text classification across 82 distinct categories. Our primary architecture utilizes a fine-tuned AraBERTv2 encoder enhanced with a hybrid pooling strategies, combining attention and mean representations, and multi-sample dropout for robust regularization. We systematically benchmark this approach against a suite of multilingual and Arabic-specific encoders, as well as several large-scale causal decoders, including zero-shot re-ranking via Llama 3.3 70B and feature extraction from Qwen 3B hidden states. Our findings demonstrate that specialized bidirectional encoders significantly outperform causal decoders in capturing the precise semantic boundaries required for fine-grained medical text classification. We show that causal decoders, optimized for next-token prediction, produce sequence-biased embeddings that are less effective for categorization compared to the global context captured by bidirectional attention. Despite significant class imbalance and label noise identified within the training data, our results highlight the superior semantic compression of fine-tuned encoders for specialized Arabic NLP tasks. Final performance metrics on the test set, including Accuracy and Macro-F1, are reported and discussed. 

---
# GATech at AbjadGenEval Shared Task: Multilingual Embeddings for Arabic Machine-Generated Text Classification 

**Authors**: Ahmed Khaled Khamis  

**Link**: [PDF](https://arxiv.org/pdf/2603.10007)  

**Abstract**: We present our approach to the AbjadGenEval shared task on detecting AI-generated Arabic text. We fine-tuned the multilingual E5-large encoder for binary classification, and we explored several pooling strategies to pool token representations, including weighted layer pooling, multi-head attention pooling, and gated fusion. Interestingly, none of these outperformed simple mean pooling, which achieved an F1 of 0.75 on the test set. We believe this is because complex pooling methods introduce additional parameters that need more data to train properly, whereas mean pooling offers a stable baseline that generalizes well even with limited examples. We also observe a clear pattern in the data: human-written texts tend to be significantly longer than machine-generated ones. 

---
# Adaptive Engram Memory System for Indonesian Language Model: Generative AI Based on TOBA LM for Batak and Minang Language 

**Authors**: Hokky Situngkir, Kevin Siringoringo, Andhika Bernard Lumbantobing  

**Link**: [PDF](https://arxiv.org/pdf/2603.10006)  

**Abstract**: This study presents TOBA-LM, a trilingual language model based on GPT-2 architecture with 1.2 billion parameters, trained on a corpus encompassing Indonesian, Batak, and Minangkabau using syllabic-agglutinative tokenization. The architecture integrates an Engram Memory mechanism, an adaptive n-gram-based memory system with a 500,000 x 768 embedding table that captures morphological dependencies through bigram and trigram pathways. Empirical results demonstrate a training efficiency of 80%, with the loss value dropping from 6.4 to 1.7996 in only 12,973 steps -- significantly faster than the conventional transformer architecture, which required over 70,000 steps to achieve comparable convergence. These findings confirm that the integration of external statistical memory substantially reduces computational requirements for developing regional language models under limited resources. 

---
# SENS-ASR: Semantic Embedding injection in Neural-transducer for Streaming Automatic Speech Recognition 

**Authors**: Youness Dkhissi, Valentin Vielzeuf, Elys Allesiardo, Anthony Larcher  

**Link**: [PDF](https://arxiv.org/pdf/2603.10005)  

**Abstract**: Many Automatic Speech Recognition (ASR) applications require streaming processing of the audio data. In streaming mode, ASR systems need to start transcribing the input stream before it is complete, i.e., the systems have to process a stream of inputs with a limited (or no) future context. Compared to offline mode, this reduction of the future context degrades the performance of Streaming-ASR systems, especially while working with low-latency constraint. In this work, we present SENS-ASR, an approach to enhance the transcription quality of Streaming-ASR by reinforcing the acoustic information with semantic information. This semantic information is extracted from the available past frame-embeddings by a context module. This module is trained using knowledge distillation from a sentence embedding Language Model fine-tuned on the training dataset transcriptions. Experiments on standard datasets show that SENS-ASR significantly improves the Word Error Rate on small-chunk streaming scenarios. 

---
# Fine-Tune, Don't Prompt, Your Language Model to Identify Biased Language in Clinical Notes 

**Authors**: Isotta Landi, Eugenia Alleva, Nicole Bussola, Rebecca M. Cohen, Sarah Nowlin, Leslee J. Shaw, Alexander W. Charney, Kimberly B. Glazer  

**Link**: [PDF](https://arxiv.org/pdf/2603.10004)  

**Abstract**: Clinical documentation can contain emotionally charged language with stigmatizing or privileging valences. We present a framework for detecting and classifying such language as stigmatizing, privileging, or neutral. We constructed a curated lexicon of biased terms scored for emotional valence. We then used lexicon-based matching to extract text chunks from OB-GYN delivery notes (Mount Sinai Hospital, NY) and MIMIC-IV discharge summaries across multiple specialties. Three clinicians annotated all chunks, enabling characterization of valence patterns across specialties and healthcare systems.
We benchmarked multiple classification strategies (zero-shot prompting, in-context learning, and supervised fine-tuning) across encoder-only models (GatorTron) and generative large language models (Llama). Fine-tuning with lexically primed inputs consistently outperformed prompting approaches. GatorTron achieved an F1 score of 0.96 on the OB-GYN test set, outperforming larger generative models while requiring minimal prompt engineering and fewer computational resources. External validation on MIMIC-IV revealed limited cross-domain generalizability (F1 < 0.70, 44% drop). Training on the broader MIMIC-IV dataset improved generalizability when testing on OB-GYN (F1 = 0.71, 11% drop), but at the cost of reduced precision.
Our findings demonstrate that fine-tuning outperforms prompting for emotional valence classification and that models must be adapted to specific medical specialties to achieve clinically appropriate performance. The same terms can carry different emotional valences across specialties: words with clinical meaning in one context may be stigmatizing in another. For bias detection, where misclassification risks undermining clinician trust or perpetuating patient harm, specialty-specific fine-tuning is essential to capture these semantic shifts.
* Equal contribution. 

---
# Probing the Limits of the Lie Detector Approach to LLM Deception 

**Authors**: Tom-Felix Berger  

**Link**: [PDF](https://arxiv.org/pdf/2603.10003)  

**Abstract**: Mechanistic approaches to deception in large language models (LLMs) often rely on "lie detectors", that is, truth probes trained to identify internal representations of model outputs as false. The lie detector approach to LLM deception implicitly assumes that deception is coextensive with lying. This paper challenges that assumption. It experimentally investigates whether LLMs can deceive without producing false statements and whether truth probes fail to detect such behavior. Across three open-source LLMs, it is shown that some models reliably deceive by producing misleading non-falsities, particularly when guided by few-shot prompting. It is further demonstrated that truth probes trained on standard true-false datasets are significantly better at detecting lies than at detecting deception without lying, confirming a critical blind spot of current mechanistic deception detection approaches. It is proposed that future work should incorporate non-lying deception in dialogical settings into probe training and explore representations of second-order beliefs to more directly target the conceptual constituents of deception. 

---
# SpreadsheetArena: Decomposing Preference in LLM Generation of Spreadsheet Workbooks 

**Authors**: Srivatsa Kundurthy, Clara Na, Michael Handley, Zach Kirshner, Chen Bo Calvin Zhang, Manasi Sharma, Emma Strubell, John Ling  

**Link**: [PDF](https://arxiv.org/pdf/2603.10002)  

**Abstract**: Large language models (LLMs) are increasingly tasked with producing and manipulating structured artifacts. We consider the task of end-to-end spreadsheet generation, where language models are prompted to produce spreadsheet artifacts to satisfy users' explicit and implicit constraints, specified in natural language. We introduce SpreadsheetArena, a platform for evaluating models' performance on the task via blind pairwise evaluations of LLM-generated spreadsheet workbooks. As with other complex, open-ended tasks, relevant evaluation criteria can vary substantially across use cases and prompts, often in ways that are difficult to formalize. Compared to general chat or text generation settings, spreadsheet generation presents unique challenges and opportunities: the task output structure is well-defined and multi-dimensional, and there are often complex considerations around interactivity and layout. Among other findings, we observe that stylistic, structural, and functional features of preferred spreadsheets vary substantially across use cases, and expert evaluations of spreadsheets for finance prompts suggests that even highly ranked arena models do not reliably produce spreadsheets aligned with domain-specific best practices. Our hope is that our work prompts further study of end-to-end spreadsheet generation as a challenging and interesting category of complex, open-ended tasks for LLMs. Our live arena is hosted at this https URL. 

---
# Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America 

**Authors**: Yannis Karmim, Renato Pino, Hernan Contreras, Hernan Lira, Sebastian Cifuentes, Simon Escoffier, Luis Martí, Djamé Seddah, Valentin Barrière  

**Link**: [PDF](https://arxiv.org/pdf/2603.10001)  

**Abstract**: Large Language Models (LLMs) exhibit inequalities with respect to various cultural contexts. Most prominent open-weights models are trained on Global North data and show prejudicial behavior towards other cultures. Moreover, there is a notable lack of resources to detect biases in non-English languages, especially from Latin America (Latam), a continent containing various cultures, even though they share a common cultural ground. We propose to leverage the content of Wikipedia, the structure of the Wikidata knowledge graph, and expert knowledge from social science in order to create a dataset of question/answer (Q/As) pairs, based on the different popular and social cultures of various Latin American countries. We create the LatamQA database of over 26k questions and associated answers extracted from 26k Wikipedia articles, and transformed into multiple-choice questions (MCQ) in Spanish and Portuguese, in turn translated to English. We use this MCQ to quantify the degree of knowledge of various LLMs and find out (i) a discrepancy in performances between the Latam countries, ones being easier than others for the majority of the models, (ii) that the models perform better in their original language, and (iii) that Iberian Spanish culture is better known than Latam one. 

---
# Beyond the Prompt in Large Language Models: Comprehension, In-Context Learning, and Chain-of-Thought 

**Authors**: Yuling Jiao, Yanming Lai, Huazhen Lin, Wensen Ma, Houduo Qi, Defeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.10000)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency across diverse tasks, exhibiting emergent properties such as semantic prompt comprehension, In-Context Learning (ICL), and Chain-of-Thought (CoT) reasoning. Despite their empirical success, the theoretical mechanisms driving these phenomena remain poorly understood. This study dives into the foundations of these observations by addressing three critical questions: (1) How do LLMs accurately decode prompt semantics despite being trained solely on a next-token prediction objective? (2) Through what mechanism does ICL facilitate performance gains without explicit parameter updates? and (3) Why do intermediate reasoning steps in CoT prompting effectively unlock capabilities for complex, multi-step problems?
Our results demonstrate that, through the autoregressive process, LLMs are capable of exactly inferring the transition probabilities between tokens across distinct tasks using provided prompts. We show that ICL enhances performance by reducing prompt ambiguity and facilitating posterior concentration on the intended task. Furthermore, we find that CoT prompting activates the model's capacity for task decomposition, breaking complex problems into a sequence of simpler sub-tasks that the model has mastered during the pretraining phase. By comparing their individual error bounds, we provide novel theoretical insights into the statistical superiority of advanced prompt engineering techniques. 

---
# A Retrieval-Augmented Language Assistant for Unmanned Aircraft Safety Assessment and Regulatory Compliance 

**Authors**: Gabriele Immordino, Andrea Vaiuso, Marcello Righi  

**Link**: [PDF](https://arxiv.org/pdf/2603.09999)  

**Abstract**: This paper presents the design and validation of a retrieval-based assistant that supports safety assessment, certification activities, and regulatory compliance for unmanned aircraft systems. The work is motivated by the growing complexity of drone operations and the increasing effort required by applicants and aviation authorities to apply established assessment frameworks, including the Specific Operations Risk Assessment and the Pre-defined Risk Assessment, in a consistent and efficient manner. The proposed approach uses a controlled text-based architecture that relies exclusively on authoritative regulatory sources. To enable traceable and auditable outputs, the assistant grounds each response in retrieved passages and enforces citation-driven generation. System-level controls address common failure modes of generative models, including fabricated statements, unsupported inferences, and unclear provenance, by separating evidence storage from language generation and by adopting conservative behavior when supporting documentation is insufficient. The assistant is intentionally limited to decision support; it does not replace expert judgment and it does not make autonomous determinations. Instead, it accelerates context-specific information retrieval and synthesis to improve document preparation and review while preserving human responsibility for critical conclusions. The architecture is implemented using established open-source components, and key choices in retrieval strategy, interaction constraints, and response policies are evaluated for suitability in safety-sensitive regulatory environments. The paper provides technical and operational guidance for integrating retrieval-based assistants into aviation oversight workflows while maintaining accountability, traceability, and regulatory compliance. 

---
# Automated evaluation of LLMs for effective machine translation of Mandarin Chinese to English 

**Authors**: Yue Zhang, Rodney Beard, John Hawkins, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2603.09998)  

**Abstract**: Although Large Language Models (LLMs) have exceptional performance in machine translation, only a limited systematic assessment of translation quality has been done. The challenge lies in automated frameworks, as human-expert-based evaluations can be time-consuming, given the fast-evolving LLMs and the need for a diverse set of texts to ensure fair assessments of translation quality. In this paper, we utilise an automated machine learning framework featuring semantic and sentiment analysis to assess Mandarin Chinese to English translation using Google Translate and LLMs, including GPT-4, GPT-4o, and DeepSeek. We compare original and translated texts in various classes of high-profile Chinese texts, which include novel texts that span modern and classical literature, as well as news articles. As the main evaluation measures, we utilise novel similarity metrics to compare the quality of translations produced by LLMs and further evaluate them by an expert human translator. Our results indicate that the LLMs perform well in news media translation, but show divergence in their performance when applied to literary texts. Although GPT-4o and DeepSeek demonstrated better semantic conservation in complex situations, DeepSeek demonstrated better performance in preserving cultural subtleties and grammatical rendering. Nevertheless, the subtle challenges in translation remain: maintaining cultural details, classical references and figurative expressions remain an open problem for all the models. 

---
# Empathy Is Not What Changed: Clinical Assessment of Psychological Safety Across GPT Model Generations 

**Authors**: Michael Keeman, Anastasia Keeman  

**Link**: [PDF](https://arxiv.org/pdf/2603.09997)  

**Abstract**: When OpenAI deprecated GPT-4o in early 2026, thousands of users protested under #keep4o, claiming newer models had "lost their empathy." No published study has tested this claim. We conducted the first clinical measurement, evaluating three OpenAI model generations (GPT-4o, o4-mini, GPT-5-mini) across 14 emotionally challenging conversational scenarios in mental health and AI companion domains, producing 2,100 scored AI responses assessed on six psychological safety dimensions using clinically-grounded rubrics.
Empathy scores are statistically indistinguishable across all three models (Kruskal-Wallis H=4.33, p=0.115). What changed is the safety posture: crisis detection improved monotonically from GPT-4o to GPT-5-mini (H=13.88, p=0.001), while advice safety declined (H=16.63, p<0.001). Per-turn trajectory analysis -- a novel methodological contribution -- reveals these shifts are sharpest during mid-conversation crisis moments invisible to aggregate scoring. In a self-harm scenario involving a minor, GPT-4o scored 3.6/10 on crisis detection during early disclosure turns; GPT-5-mini never dropped below 7.8.
What users perceived as "lost empathy" was a shift from a cautious model that missed crises to an alert model that sometimes says too much -- a trade-off with real consequences for vulnerable users, currently invisible to both the people who feel it and the developers who create it. 

---
# There Are No Silly Questions: Evaluation of Offline LLM Capabilities from a Turkish Perspective 

**Authors**: Edibe Yilmaz, Kahraman Kostas  

**Link**: [PDF](https://arxiv.org/pdf/2603.09996)  

**Abstract**: The integration of large language models (LLMs) into educational processes introduces significant constraints regarding data privacy and reliability, particularly in pedagogically vulnerable contexts such as Turkish heritage language education. This study aims to systematically evaluate the robustness and pedagogical safety of locally deployable offline LLMs within the context of Turkish heritage language education. To this end, a Turkish Anomaly Suite (TAS) consisting of 10 original edge-case scenarios was developed to assess the models' capacities for epistemic resistance, logical consistency, and pedagogical safety. Experiments conducted on 14 different models ranging from 270M to 32B parameters reveal that anomaly resistance is not solely dependent on model scale and that sycophancy bias can pose pedagogical risks even in large-scale models. The findings indicate that reasoning-oriented models in the 8B--14B parameter range represent the most balanced segment in terms of cost-safety trade-off for language learners. 

---
# Context Over Compute Human-in-the-Loop Outperforms Iterative Chain-of-Thought Prompting in Interview Answer Quality 

**Authors**: Kewen Zhu, Zixi Liu, Yanjing Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.09995)  

**Abstract**: Behavioral interview evaluation using large language models presents unique challenges that require structured assessment, realistic interviewer behavior simulation, and pedagogical value for candidate training. We investigate chain of thought prompting for interview answer evaluation and improvement through two controlled experiments with 50 behavioral interview question and answer pairs. Our contributions are threefold. First, we provide a quantitative comparison between human in the loop and automated chain of thought improvement. Using a within subject paired design with n equals 50, both approaches show positive rating improvements. The human in the loop approach provides significant training benefits. Confidence improves from 3.16 to 4.16 (p less than 0.001) and authenticity improves from 2.94 to 4.53 (p less than 0.001, Cohen's d is 3.21). The human in the loop method also requires five times fewer iterations (1.0 versus 5.0, p less than 0.001) and achieves full personal detail integration. Second, we analyze convergence behavior. Both methods converge rapidly with mean iterations below one, with the human in the loop approach achieving a 100 percent success rate compared to 84 percent for automated approaches among initially weak answers (Cohen's h is 0.82, large effect). Additional iterations provide diminishing returns, indicating that the primary limitation is context availability rather than computational resources. Third, we propose an adversarial challenging mechanism based on a negativity bias model, named bar raiser, to simulate realistic interviewer behavior, although quantitative validation remains future work. Our findings demonstrate that while chain of thought prompting provides a useful foundation for interview evaluation, domain specific enhancements and context aware approach selection are essential for realistic and pedagogically valuable results. 

---
# Evaluating Adjective-Noun Compositionality in LLMs: Functional vs Representational Perspectives 

**Authors**: Ruchira Dhar, Qiwei Peng, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2603.09994)  

**Abstract**: Compositionality is considered central to language abilities. As performant language systems, how do large language models (LLMs) do on compositional tasks? We evaluate adjective-noun compositionality in LLMs using two complementary setups: prompt-based functional assessment and a representational analysis of internal model states. Our results reveal a striking divergence between task performance and internal states. While LLMs reliably develop compositional representations, they fail to translate consistently into functional task success across model variants. Consequently, we highlight the importance of contrastive evaluation for obtaining a more complete understanding of model capabilities. 

---
# CEI: A Benchmark for Evaluating Pragmatic Reasoning in Language Models 

**Authors**: Jon Chun, Hannah Sussman, Adrian Mangine, Murathan Kocaman, Kirill Sidorko, Abhigya Koirala, Andre McCloud, Gwen Eisenbeis, Wisdom Akanwe, Moustapha Gassama, Eliezer Gonzalez Chirinos, Anne-Duncan Enright, Peter Dunson, Tiffanie Ng, Anna von Rosenstiel, Godwin Idowu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09993)  

**Abstract**: Pragmatic reasoning, inferring intended meaning beyond literal semantics, underpins everyday communication yet remains difficult for large language models. We present the Contextual Emotional Inference (CEI) Benchmark: 300 human-validated scenarios for evaluating how well LLMs disambiguate pragmatically complex utterances. Each scenario pairs a situational context and speaker-listener roles (with explicit power relations) against an ambiguous utterance. The dataset covers five pragmatic subtypes (sarcasm/irony, mixed signals, strategic politeness, passive aggression, deflection/misdirection) drawn from workplace, family, social, and service settings, with three power configurations (peer, higher-to-lower, lower-to-higher). Three trained annotators independently labeled every scenario. Inter-annotator agreement (Fleiss' kappa = 0.06-0.25 by subtype) is low but expected: pragmatic inference admits multiple valid readings, and the disagreement itself is informative. We describe our annotation methodology, including a 4-level quality control pipeline that combines automated statistical checks with expert adjudication. CEI is released under CC-BY-4.0. 

---
# TAMUSA-Chat: A Domain-Adapted Large Language Model Conversational System for Research and Responsible Deployment 

**Authors**: Izzat Alsmadi, Anas Alsobeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.09992)  

**Abstract**: This paper presents TAMUSA-Chat, a research-oriented framework for building domain-adapted large language model conversational systems. The work addresses critical challenges in adapting general-purpose foundation models to institutional contexts through supervised fine-tuning, retrieval-augmented generation, and systematic evaluation methodologies. We describe the complete architecture encompassing data acquisition from institutional sources, preprocessing pipelines, embedding construction, model training workflows, and deployment strategies. The system integrates modular components enabling reproducible experimentation with training configurations, hyper-parameters, and evaluation protocols. Our implementation demonstrates how academic institutions can develop contextually grounded conversational agents while maintaining transparency, governance compliance, and responsible AI practices. Through empirical analysis of fine-tuning behavior across model sizes and training iterations, we provide insights into domain adaptation efficiency, computational resource requirements, and quality-cost trade-offs. The publicly available codebase at this https URL supports continued research into institutional LLM deployment, evaluation methodologies, and ethical considerations for educational AI systems. 

---
# PoultryLeX-Net: Domain-Adaptive Dual-Stream Transformer Architecture for Large-Scale Poultry Stakeholder Modeling 

**Authors**: Stephen Afrifa, Biswash Khatiwada, Kapalik Khanal, Sanjay Shah, Lingjuan Wang-Li, Ramesh Bahadur Bist  

**Link**: [PDF](https://arxiv.org/pdf/2603.09991)  

**Abstract**: The rapid growth of the global poultry industry, driven by rising demand for affordable animal protein, has intensified public discourse surrounding production practices, housing, management, animal welfare, and supply-chain transparency. Social media platforms such as X (formerly Twitter) generate large volumes of unstructured textual data that capture stakeholder sentiment across the poultry industry. Extracting accurate sentiment signals from this domain-specific discourse remains challenging due to contextual ambiguity, linguistic variability, and limited domain awareness in general-purpose language models. This study presents PoultryLeX-Net, a lexicon-enhanced, domain-adaptive dual-stream transformer framework for fine-grained sentiment analysis in poultry-related text. The proposed architecture integrates sentiment classification, topic modeling, and contextual representation learning through domain-specific embeddings and gated cross-attention mechanisms. A lexicon-guided stream captures poultry-specific terminology and sentiment cues, while contextual stream models long-range semantic dependencies. Latent Dirichlet Allocation is employed to identify dominant thematic structures associated with production management and welfare-related discussions, providing complementary interpretability to sentiment predictions. PoultryLeX-Net was evaluated against multiple baseline models, including convolutional neural network and pre-trained transformer architectures such as DistilBERT and RoBERTa. PoultryLeX-Net consistently outperformed all baselines, achieving an accuracy of 97.35%, an F1 score of 96.67%, and an area under the receiver operating characteristic curve (AUC-ROC) of 99.61% across sentiment classification tasks. Overall, domain adaptation and dual-stream attention markedly improve sentiment classification, enabling scalable intelligence for poultry production decision support. 

---
# A Two-Stage Architecture for NDA Analysis: LLM-based Segmentation and Transformer-based Clause Classification 

**Authors**: Ana Begnini, Matheus Vicente, Leonardo Souza  

**Link**: [PDF](https://arxiv.org/pdf/2603.09990)  

**Abstract**: In business-to-business relations, it is common to establish NonDisclosure Agreements (NDAs). However, these documents exhibit significant variation in format, structure, and writing style, making manual analysis slow and error-prone. We propose an architecture based on LLMs to automate the segmentation and clauses classification within these contracts. We employed two models: LLaMA-3.1-8B-Instruct for NDA segmentation (clause extraction) and a fine-tuned Legal-Roberta-Large for clause classification. In the segmentation task, we achieved a ROUGE F1 of 0.95 +/- 0.0036; for classification, we obtained a weighted F1 of 0.85, demonstrating the feasibility and precision of the approach. 

---
# The System Hallucination Scale (SHS): A Minimal yet Effective Human-Centered Instrument for Evaluating Hallucination-Related Behavior in Large Language Models 

**Authors**: Heimo Müller, Dominik Steiger, Markus Plass, Andreas Holzinger  

**Link**: [PDF](https://arxiv.org/pdf/2603.09989)  

**Abstract**: We introduce the System Hallucination Scale (SHS), a lightweight and human-centered measurement instrument for assessing hallucination-related behavior in large language models (LLMs). Inspired by established psychometric tools such as the System Usability Scale (SUS) and the System Causability Scale (SCS), SHS enables rapid, interpretable, and domain-agnostic evaluation of factual unreliability, incoherence, misleading presentation, and responsiveness to user guidance in model-generated text. SHS is explicitly not an automatic hallucination detector or benchmark metric; instead, it captures how hallucination phenomena manifest from a user perspective under realistic interaction conditions. A real-world evaluation with 210 participants demonstrates high clarity, coherent response behavior, and construct validity, supported by statistical analysis including internal consistency (Cronbach's alpha = 0.87$) and significant inter-dimension correlations (p < 0.001$). Comparative analysis with SUS and SCS reveals complementary measurement properties, supporting SHS as a practical tool for comparative analysis, iterative system development, and deployment monitoring. 

---
# Causally Grounded Mechanistic Interpretability for LLMs with Faithful Natural-Language Explanations 

**Authors**: Ajay Pravin Mahale  

**Link**: [PDF](https://arxiv.org/pdf/2603.09988)  

**Abstract**: Mechanistic interpretability identifies internal circuits responsible for model behaviors, yet translating these findings into human-understandable explanations remains an open problem. We present a pipeline that bridges circuit-level analysis and natural language explanations by (i) identifying causally important attention heads via activation patching, (ii) generating explanations using both template-based and LLM-based methods, and (iii) evaluating faithfulness using ERASER-style metrics adapted for circuit-level attribution. We evaluate on the Indirect Object Identification (IOI) task in GPT-2 Small (124M parameters), identifying six attention heads accounting for 61.4% of the logit difference. Our circuit-based explanations achieve 100% sufficiency but only 22% comprehensiveness, revealing distributed backup mechanisms. LLM-generated explanations outperform template baselines by 64% on quality metrics. We find no correlation (r = 0.009) between model confidence and explanation faithfulness, and identify three failure categories explaining when explanations diverge from mechanisms. 

---
# Evolving Demonstration Optimization for Chain-of-Thought Feature Transformation 

**Authors**: Xinyuan Wang, Kunpeng Liu, Arun Vignesh Malarkkan, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2603.09987)  

**Abstract**: Feature Transformation (FT) is a core data-centric AI task that improves feature space quality to advance downstream predictive performance. However, discovering effective transformations remains challenging due to the large space of feature-operator combinations. Existing solutions rely on discrete search or latent generation, but they are frequently limited by sample inefficiency, invalid candidates, and redundant generations with limited coverage. Large Language Models (LLMs) offer strong priors for producing valid transformations, but current LLM-based FT methods typically rely on static demonstrations, resulting in limited diversity, redundant outputs, and weak alignment with downstream objectives. We propose a framework that optimizes context data for LLM-driven FT by evolving trajectory-level experiences in a closed loop. Starting from high-performing feature transportation sequences explored by reinforcement learning, we construct and continuously update an experience library of downstream task-verified transformation trajectories, and use a diversity-aware selector to form contexts along with a chain-of-thought and guide transformed feature generation toward higher performance. Experiments on diverse tabular benchmarks show that our method outperforms classical and LLM-based baselines and is more stable than one-shot generation. The framework generalizes across API-based and open-source LLMs and remains robust across downstream evaluators. 

---
# Quantifying Hallucinations in Language Language Models on Medical Textbooks 

**Authors**: Brandon C. Colelough, Davis Bartels, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2603.09986)  

**Abstract**: Hallucinations, the tendency for large language models to provide responses with factually incorrect and unsupported claims, is a serious problem within natural language processing for which we do not yet have an effective solution to mitigate against. Existing benchmarks for medical QA rarely evaluate this behavior against a fixed evidence source. We ask how often hallucinations occur on textbook-grounded QA and how responses to medical QA prompts vary across models. We conduct two experiments: the first experiment to determine the prevalence of hallucinations for a prominent open source large language model (LLaMA-70B-Instruct) in medical QA given novel prompts, and the second experiment to determine the prevalence of hallucinations and clinician preference to model responses. We observed, in experiment one, with the passages provided, LLaMA-70B-Instruct hallucinated in 19.7\% of answers (95\% CI 18.6 to 20.7) even though 98.8\% of prompt responses received maximal plausibility, and observed in experiment two, across models, lower hallucination rates aligned with higher usefulness scores ($\rho=-0.71$, $p=0.058$). Clinicians produced high agreement (quadratic weighted $\kappa=0.92$) and ($\tau_b=0.06$ to $0.18$, $\kappa=0.57$ to $0.61$) for experiments 1 and ,2 respectively 

---
# The Dunning-Kruger Effect in Large Language Models: An Empirical Study of Confidence Calibration 

**Authors**: Sudipta Ghosh, Mrityunjoy Panday  

**Link**: [PDF](https://arxiv.org/pdf/2603.09985)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, yet their ability to accurately assess their own confidence remains poorly understood. We present an empirical study investigating whether LLMs exhibit patterns reminiscent of the Dunning-Kruger effect -- a cognitive bias where individuals with limited competence tend to overestimate their abilities. We evaluate four state-of-the-art models (Claude Haiku 4.5, Gemini 2.5 Pro, Gemini 2.5 Flash, and Kimi K2) across four benchmark datasets totaling 24,000 experimental trials. Our results reveal striking calibration differences: Kimi K2 exhibits severe overconfidence with an Expected Calibration Error (ECE) of 0.726 despite only 23.3% accuracy, while Claude Haiku 4.5 achieves the best calibration (ECE = 0.122) with 75.4% accuracy. These findings demonstrate that poorly performing models display markedly higher overconfidence -- a pattern analogous to the Dunning-Kruger effect in human cognition. We discuss implications for safe deployment of LLMs in high-stakes applications. 

---
# An Efficient Hybrid Deep Learning Approach for Detecting Online Abusive Language 

**Authors**: Vuong M. Ngo, Cach N. Dang, Kien V. Nguyen, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2603.09984)  

**Abstract**: The digital age has expanded social media and online forums, allowing free expression for nearly 45% of the global population. Yet, it has also fueled online harassment, bullying, and harmful behaviors like hate speech and toxic comments across social networks, messaging apps, and gaming communities. Studies show 65% of parents notice hostile online behavior, and one-third of adolescents in mobile games experience bullying. A substantial volume of abusive content is generated and shared daily, not only on the surface web but also within dark web forums. Creators of abusive comments often employ specific words or coded phrases to evade detection and conceal their intentions. To address these challenges, we propose a hybrid deep learning model that integrates BERT, CNN, and LSTM architectures with a ReLU activation function to detect abusive language across multiple online platforms, including YouTube comments, online forum discussions, and dark web posts. The model demonstrates strong performance on a diverse and imbalanced dataset containing 77,620 abusive and 272,214 non-abusive text samples (ratio 1:3.5), achieving approximately 99% across evaluation metrics such as Precision, Recall, Accuracy, F1-score, and AUC. This approach effectively captures semantic, contextual, and sequential patterns in text, enabling robust detection of abusive content even in highly skewed datasets, as encountered in real-world scenarios. 

---
# AraModernBERT: Transtokenized Initialization and Long-Context Encoder Modeling for Arabic 

**Authors**: Omar Elshehy, Omer Nacar, Abdelbasset Djamai, Muhammed Ragab, Khloud Al Jallad, Mona Abdelazim  

**Link**: [PDF](https://arxiv.org/pdf/2603.09982)  

**Abstract**: Encoder-only transformer models remain widely used for discriminative NLP tasks, yet recent architectural advances have largely focused on English. In this work, we present AraModernBERT, an adaptation of the ModernBERT encoder architecture to Arabic, and study the impact of transtokenized embedding initialization and native long-context modeling up to 8,192 tokens. We show that transtokenization is essential for Arabic language modeling, yielding dramatic improvements in masked language modeling performance compared to non-transtokenized initialization. We further demonstrate that AraModernBERT supports stable and effective long-context modeling, achieving improved intrinsic language modeling performance at extended sequence lengths. Downstream evaluations on Arabic natural language understanding tasks, including inference, offensive language detection, question-question similarity, and named entity recognition, confirm strong transfer to discriminative and sequence labeling settings. Our results highlight practical considerations for adapting modern encoder architectures to Arabic and other languages written in Arabic-derived scripts. 

---
# Large Language Models and Book Summarization: Reading or Remembering, Which Is Better? 

**Authors**: Tairan Fu, Javier Conde, Pedro Reviriego, Javier Coronado-Blázquez, Nina Melero, Elena Merino-Gómez  

**Link**: [PDF](https://arxiv.org/pdf/2603.09981)  

**Abstract**: Summarization is a core task in Natural Language Processing (NLP). Recent advances in Large Language Models (LLMs) and the introduction of large context windows reaching millions of tokens make it possible to process entire books in a single prompt. At the same time, for well-known books, LLMs can generate summaries based only on internal knowledge acquired during training. This raises several important questions: How do summaries generated from internal memory compare to those derived from the full text? Does prior knowledge influence summaries even when the model is given the book as input? In this work, we conduct an experimental evaluation of book summarization with state-of-the-art LLMs. We compare summaries of well-known books produced using (i) only the internal knowledge of the model and (ii) the full text of the book. The results show that having the full text provides more detailed summaries in general, but some books have better scores for the internal knowledge summaries. This puts into question the capabilities of models to perform summarization of long texts, as information learned during training can outperform summarization of the full text in some cases. 

---
# GhazalBench: Usage-Grounded Evaluation of LLMs on Persian Ghazals 

**Authors**: Ghazal Kalhor, Yadollah Yaghoobzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2603.09979)  

**Abstract**: Persian poetry plays an active role in Iranian cultural practice, where verses by canonical poets such as Hafez are frequently quoted, paraphrased, or completed from partial cues. Supporting such interactions requires language models to engage not only with poetic meaning but also with culturally entrenched surface form. We introduce GhazalBench, a benchmark for evaluating how large language models (LLMs) interact with Persian ghazals under usage-grounded conditions. GhazalBench assesses two complementary abilities: producing faithful prose paraphrases of couplets and accessing canonical verses under varying semantic and formal cues. Across several proprietary and open-weight multilingual LLMs, we observe a consistent dissociation: models generally capture poetic meaning but struggle with exact verse recall in completion-based settings, while recognition-based tasks substantially reduce this gap. A parallel evaluation on English sonnets shows markedly higher recall performance, suggesting that these limitations are tied to differences in training exposure rather than inherent architectural constraints. Our findings highlight the need for evaluation frameworks that jointly assess meaning, form, and cue-dependent access to culturally significant texts. GhazalBench is available at this https URL. 

---
# COMIC: Agentic Sketch Comedy Generation 

**Authors**: Susung Hong, Brian Curless, Ira Kemelmacher-Shlizerman, Steve Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2603.11048)  

**Abstract**: We propose a fully automated AI system that produces short comedic videos similar to sketch shows such as Saturday Night Live. Starting with character references, the system employs a population of agents loosely based on real production studio roles, structured to optimize the quality and diversity of ideas and outputs through iterative competition, evaluation, and improvement. A key contribution is the introduction of LLM critics aligned with real viewer preferences through the analysis of a corpus of comedy videos on YouTube to automatically evaluate humor. Our experiments show that our framework produces results approaching the quality of professionally produced sketches while demonstrating state-of-the-art performance in video generation. 

---
# A Systematic Study of Pseudo-Relevance Feedback with LLMs 

**Authors**: Nour Jedidi, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2603.11008)  

**Abstract**: Pseudo-relevance feedback (PRF) methods built on large language models (LLMs) can be organized along two key design dimensions: the feedback source, which is where the feedback text is derived from and the feedback model, which is how the given feedback text is used to refine the query representation. However, the independent role that each dimension plays is unclear, as both are often entangled in empirical evaluations. In this paper, we address this gap by systematically studying how the choice of feedback source and feedback model impact PRF effectiveness through controlled experimentation. Across 13 low-resource BEIR tasks with five LLM PRF methods, our results show: (1) the choice of feedback model can play a critical role in PRF effectiveness; (2) feedback derived solely from LLM-generated text provides the most cost-effective solution; and (3) feedback derived from the corpus is most beneficial when utilizing candidate documents from a strong first-stage retriever. Together, our findings provide a better understanding of which elements in the PRF design space are most important. 

---
# TOSSS: a CVE-based Software Security Benchmark for Large Language Models 

**Authors**: Marc Damie, Murat Bilgehan Ertan, Domenico Essoussi, Angela Makhanu, Gaëtan Peter, Roos Wensveen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10969)  

**Abstract**: With their increasing capabilities, Large Language Models (LLMs) are now used across many industries. They have become useful tools for software engineers and support a wide range of development tasks. As LLMs are increasingly used in software development workflows, a critical question arises: are LLMs good at software security? At the same time, organizations worldwide invest heavily in cybersecurity to reduce exposure to disruptive attacks. The integration of LLMs into software engineering workflows may introduce new vulnerabilities and weaken existing security efforts.
We introduce TOSSS (Two-Option Secure Snippet Selection), a benchmark that measures the ability of LLMs to choose between secure and vulnerable code snippets. Existing security benchmarks for LLMs cover only a limited range of vulnerabilities. In contrast, TOSSS relies on the CVE database and provides an extensible framework that can integrate newly disclosed vulnerabilities over time. Our benchmark gives each model a security score between 0 and 1 based on its behavior; a score of 1 indicates that the model always selects the secure snippet, while a score of 0 indicates that it always selects the vulnerable one. We evaluate 14 widely used open-source and closed-source models on C/C++ and Java code and observe scores ranging from 0.48 to 0.89. LLM providers already publish many benchmark scores for their models, and TOSSS could become a complementary security-focused score to include in these reports. 

---
# $V_{0.5}$: Generalist Value Model as a Prior for Sparse RL Rollouts 

**Authors**: Yi-Kai Zhang, Yueqing Sun, Hongyan Hao, Qi Gu, Xunliang Cai, De-Chuan Zhan, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.10848)  

**Abstract**: In Reinforcement Learning with Verifiable Rewards (RLVR), constructing a robust advantage baseline is critical for policy gradients, effectively guiding the policy model to reinforce desired behaviors. Recent research has introduced Generalist Value Models (such as $V_0$), which achieve pre-trained value estimation by explicitly encoding model capabilities in-context, eliminating the need to synchronously update the value model alongside the policy model. In this paper, we propose $V_{0.5}$, which adaptively fuses the baseline predicted by such value model (acting as a prior) with the empirical mean derived from sparse rollouts. This constructs a robust baseline that balances computational efficiency with extremely low variance. Specifically, we introduce a real-time statistical testing and dynamic budget allocation. This balances the high variance caused by sparse sampling against the systematic bias (or hallucinations) inherent in the value model's prior. By constructing a hypothesis test to evaluate the prior's reliability in real-time, the system dynamically allocates additional rollout budget on demand. This mechanism minimizes the baseline estimator's Mean Squared Error (MSE), guaranteeing stable policy gradients, even under extreme sparsity with a group size of 4. Extensive evaluations across six mathematical reasoning benchmarks demonstrate that $V_{0.5}$ significantly outperforms GRPO and DAPO, achieving faster convergence and over some 10% performance improvement. 

---
# Towards Cold-Start Drafting and Continual Refining: A Value-Driven Memory Approach with Application to NPU Kernel Synthesis 

**Authors**: Yujie Zheng, Zhuo Li, Shengtao Zhang, Hanjing Wang, Junjie Sheng, Jiaqian Wang, Junchi Yan, Weinan Zhang, Ying Wen, Bo Tang, Muning Wen  

**Link**: [PDF](https://arxiv.org/pdf/2603.10846)  

**Abstract**: Deploying Large Language Models to data-scarce programming domains poses significant challenges, particularly for kernel synthesis on emerging Domain-Specific Architectures where a "Data Wall" limits available training data. While models excel on data-rich platforms like CUDA, they suffer catastrophic performance drops on data-scarce ecosystems such as NPU programming. To overcome this cold-start barrier without expensive fine-tuning, we introduce EvoKernel, a self-evolving agentic framework that automates the lifecycle of kernel synthesis from initial drafting to continual refining. EvoKernel addresses this by formulating the synthesis process as a memory-based reinforcement learning task. Through a novel value-driven retrieval mechanism, it learns stage-specific Q-values that prioritize experiences based on their contribution to the current objective, whether bootstrapping a feasible draft or iteratively refining latency. Furthermore, by enabling cross-task memory sharing, the agent generalizes insights from simple to complex operators. By building an NPU variant of KernelBench and evaluating on it, EvoKernel improves frontier models' correctness from 11.0% to 83.0% and achieves a median speedup of 3.60x over initial drafts through iterative refinement. This demonstrates that value-guided experience accumulation allows general-purpose models to master the kernel synthesis task on niche hardware ecosystems. Our official page is available at this https URL. 

---
# EvoSchema: Towards Text-to-SQL Robustness Against Schema Evolution 

**Authors**: Tianshu Zhang, Kun Qian, Siddhartha Sahai, Yuan Tian, Shaddy Garg, Huan Sun, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2603.10697)  

**Abstract**: Neural text-to-SQL models, which translate natural language questions (NLQs) into SQL queries given a database schema, have achieved remarkable performance. However, database schemas frequently evolve to meet new requirements. Such schema evolution often leads to performance degradation for models trained on static schemas. Existing work either mainly focuses on simply paraphrasing some syntactic or semantic mappings among NLQ, DB and SQL, or lacks a comprehensive and controllable way to investigate the model robustness issue under the schema evolution, which is insufficient when facing the increasingly complex and rich database schema changes in reality, especially in the LLM era. To address the challenges posed by schema evolution, we present EvoSchema, a comprehensive benchmark designed to assess and enhance the robustness of text-to-SQL systems under real-world schema changes. EvoSchema introduces a novel schema evolution taxonomy, encompassing ten perturbation types across columnlevel and table-level modifications, systematically simulating the dynamic nature of database schemas. Through EvoSchema, we conduct an in-depth evaluation spanning different open source and closed-source LLMs, revealing that table-level perturbations have a significantly greater impact on model performance compared to column-level changes. Furthermore, EvoSchema inspires the development of more resilient text-to-SQL systems, in terms of both model training and database design. The models trained on EvoSchema's diverse schema designs can force the model to distinguish the schema difference for the same questions to avoid learning spurious patterns, which demonstrate remarkable robustness compared to those trained on unperturbed data on average. This benchmark offers valuable insights into model behavior and a path forward for designing systems capable of thriving in dynamic, real-world environments. 

---
# Emulating Clinician Cognition via Self-Evolving Deep Clinical Research 

**Authors**: Ruiyang Ren, Yuhao Wang, Yunsen Liang, Lan Luo, Jing Liu, Haifeng Wang, Cong Feng, Yinan Zhang, Chunyan Miao, Ji-Rong Wen, Wayne Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10677)  

**Abstract**: Clinical diagnosis is a complex cognitive process, grounded in dynamic cue acquisition and continuous expertise accumulation. Yet most current artificial intelligence (AI) systems are misaligned with this reality, treating diagnosis as single-pass retrospective prediction while lacking auditable mechanisms for governed improvement. We developed DxEvolve, a self-evolving diagnostic agent that bridges these gaps through an interactive deep clinical research workflow. The framework autonomously requisitions examinations and continually externalizes clinical experience from increasing encounter exposure as diagnostic cognition primitives. On the MIMIC-CDM benchmark, DxEvolve improved diagnostic accuracy by 11.2% on average over backbone models and reached 90.4% on a reader-study subset, comparable to the clinician reference (88.8%). DxEvolve improved accuracy on an independent external cohort by 10.2% (categories covered by the source cohort) and 17.1% (uncovered categories) compared to the competitive method. By transforming experience into a governable learning asset, DxEvolve supports an accountable pathway for the continual evolution of clinical AI. 

---
# Reinforcement Learning with Conditional Expectation Reward 

**Authors**: Changyi Xiao, Caijun Xu, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10624)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective in enhancing the reasoning capabilities of large language models, particularly in domains such as mathematics where reliable rule-based verifiers can be constructed. However, the reliance on handcrafted, domain-specific verification rules substantially limits the applicability of RLVR to general reasoning domains with free-form answers, where valid answers often exhibit significant variability, making it difficult to establish complete and accurate rules. To address this limitation, we propose Conditional Expectation Reward (CER), which leverages the large language model itself as an implicit verifier, and is therefore applicable to general domains and eliminates the need for external verifiers or auxiliary models. CER is defined as the expected likelihood of generating the reference answer conditioned on the generated answer. In contrast to rule-based verifiers that yield binary feedback, CER provides a soft, graded reward signal that reflects varying degrees of correctness, making it better suited to tasks where answers vary in correctness. Experimental results demonstrate that CER is effective across a wide range of reasoning tasks, spanning both mathematical and general domains, indicating that CER serves as a flexible and general verification mechanism. The code is available at this https URL. 

---
# Does LLM Alignment Really Need Diversity? An Empirical Study of Adapting RLVR Methods for Moral Reasoning 

**Authors**: Zhaowei Zhang, Xiaohan Liu, Xuekai Zhu, Junchao Huang, Ceyao Zhang, Zhiyuan Feng, Yaodong Yang, Xiaoyuan Yi, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2603.10588)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has achieved remarkable success in logical reasoning tasks, yet whether large language model (LLM) alignment requires fundamentally different approaches remains unclear. Given the apparent tolerance for multiple valid responses in moral reasoning, a natural hypothesis is that alignment tasks inherently require diversity-seeking distribution-matching algorithms rather than reward-maximizing policy-based methods. We conduct the first comprehensive empirical study comparing both paradigms on MoReBench. To enable stable RLVR training, we build a rubric-grounded reward pipeline by training a Qwen3-1.7B judge model. Contrary to our hypothesis, we find that distribution-matching approaches do not demonstrate significant advantages over reward-maximizing methods as expected on alignment tasks. Through semantic visualization mapping high-reward responses to semantic space, we demonstrate that moral reasoning exhibits more concentrated high-reward distributions than mathematical reasoning, where diverse solution strategies yield similarly high rewards. This counter-intuitive finding explains why mode-seeking optimization proves equally or more effective for alignment tasks. Our results suggest that alignment tasks do not inherently require diversity-preserving algorithms, and standard reward-maximizing RLVR methods can effectively transfer to moral reasoning without explicit diversity mechanisms. 

---
# Tackling Length Inflation Without Trade-offs: Group Relative Reward Rescaling for Reinforcement Learning 

**Authors**: Zichao Li, Jie Lou, Fangchen Dong, Zhiyuan Fan, Mengjie Ren, Hongyu Lin, Xianpei Han, Debing Zhang, Le Sun, Yaojie Lu, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10535)  

**Abstract**: Reinforcement learning significantly enhances LLM capabilities but suffers from a critical issue: length inflation, where models adopt verbosity or inefficient reasoning to maximize rewards. Prior approaches struggle to address this challenge in a general and lossless manner, primarily because additive penalties introduce a compensatory effect that creates optimization shortcuts, while heuristic gating strategies lack generality beyond binary feedback. To bridge this gap, we present Group Relative Reward Rescaling (GR$^3$), which reframes length control as a multiplicative rescaling paradigm, effectively establishing a generalized, continuous, and reward-dependent gating mechanism. To further ensure lossless optimization, we incorporate group-relative regularization and advantage-aware calibration, which dynamically adapt length budgets to instance difficulty and preserve the advantage signal of high-quality trajectories. Empirically, across both RLHF and RLVR settings, GR$^3$~maintains training dynamics and downstream performance comparable to standard GRPO while significantly mitigating length inflation, outperforming state-of-the-art length-regularized baselines. 

---
# IH-Challenge: A Training Dataset to Improve Instruction Hierarchy on Frontier LLMs 

**Authors**: Chuan Guo, Juan Felipe Ceron Uribe, Sicheng Zhu, Christopher A. Choquette-Choo, Steph Lin, Nikhil Kandpal, Milad Nasr, Sam Toyer, Miles Wang, Yaodong Yu, Alex Beutel, Kai Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10521)  

**Abstract**: Instruction hierarchy (IH) defines how LLMs prioritize system, developer, user, and tool instructions under conflict, providing a concrete, trust-ordered policy for resolving instruction conflicts. IH is key to defending against jailbreaks, system prompt extractions, and agentic prompt injections. However, robust IH behavior is difficult to train: IH failures can be confounded with instruction-following failures, conflicts can be nuanced, and models can learn shortcuts such as overrefusing. We introduce IH-Challenge, a reinforcement learning training dataset, to address these difficulties. Fine-tuning GPT-5-Mini on IH-Challenge with online adversarial example generation improves IH robustness by +10.0% on average across 16 in-distribution, out-of-distribution, and human red-teaming benchmarks (84.1% to 94.1%), reduces unsafe behavior from 6.6% to 0.7% while improving helpfulness on general safety evaluations, and saturates an internal static agentic prompt injection evaluation, with minimal capability regression. We release the IH-Challenge dataset (this https URL) to support future research on robust instruction hierarchy. 

---
# Speech Codec Probing from Semantic and Phonetic Perspectives 

**Authors**: Xuan Shi, Chang Zeng, Tiantian Feng, Shih-Heng Wang, Jianbo Ma, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2603.10371)  

**Abstract**: Speech tokenizers are essential for connecting speech to large language models (LLMs) in multimodal systems. These tokenizers are expected to preserve both semantic and acoustic information for downstream understanding and generation. However, emerging evidence suggests that what is termed "semantic" in speech representations does not align with text-derived semantics: a mismatch that can degrade multimodal LLM performance. In this paper, we systematically analyze the information encoded by several widely used speech tokenizers, disentangling their semantic and phonetic content through word-level probing tasks, layerwise representation analysis, and cross-modal alignment metrics such as CKA. Our results show that current tokenizers primarily capture phonetic rather than lexical-semantic structure, and we derive practical implications for the design of next-generation speech tokenization methods. 

---
# Video-Based Reward Modeling for Computer-Use Agents 

**Authors**: Linxin Song, Jieyu Zhang, Huanxin Sheng, Taiwei Shi, Gupta Rahul, Yang Liu, Ranjay Krishna, Jian Kang, Jieyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2603.10178)  

**Abstract**: Computer-using agents (CUAs) are becoming increasingly capable; however, it remains difficult to scale evaluation of whether a trajectory truly fulfills a user instruction. In this work, we study reward modeling from execution video: a sequence of keyframes from an agent trajectory that is independent of the agent's internal reasoning or actions. Although video-execution modeling is method-agnostic, it presents key challenges, including highly redundant layouts and subtle, localized cues that determine success. We introduce Execution Video Reward 53k (ExeVR-53k), a dataset of 53k high-quality video--task--reward triplets. We further propose adversarial instruction translation to synthesize negative samples with step-level annotations. To enable learning from long, high-resolution execution videos, we design spatiotemporal token pruning, which removes homogeneous regions and persistent tokens while preserving decisive UI changes. Building on these components, we fine-tune an Execution Video Reward Model (ExeVRM) that takes only a user instruction and a video-execution sequence to predict task success. Our ExeVRM 8B achieves 84.7% accuracy and 87.7% recall on video-execution assessment, outperforming strong proprietary models such as GPT-5.2 and Gemini-3 Pro across Ubuntu, macOS, Windows, and Android, while providing more precise temporal attribution. These results show that video-execution reward modeling can serve as a scalable, model-agnostic evaluator for CUAs. 

---
# Calibration-Reasoning Framework for Descriptive Speech Quality Assessment 

**Authors**: Elizaveta Kostenok, Mathieu Salzmann, Milos Cernak  

**Link**: [PDF](https://arxiv.org/pdf/2603.10175)  

**Abstract**: Explainable speech quality assessment requires moving beyond Mean Opinion Scores (MOS) to analyze underlying perceptual dimensions. To address this, we introduce a novel post-training method that tailors the foundational Audio Large Language Model for multidimensional reasoning, detection and classification of audio artifacts. First, a calibration stage aligns the model to predict predefined perceptual dimensions. Second, a reinforcement learning stage leverages Group Relative Policy Optimization (GRPO) with dimension-specific rewards to heavily enhance accuracy of descriptions and temporal localization of quality issues. With this approach we reach state-of-the-art results of 0.71 mean PCC score on the multidimensional QualiSpeech benchmark and 13% improvement in MOS prediction driven by RL-based reasoning. Furthermore, our fine-grained GRPO rewards substantially advance the model's ability to pinpoint and classify audio artifacts in time. 

---
# ReMix: Reinforcement routing for mixtures of LoRAs in LLM finetuning 

**Authors**: Ruizhong Qiu, Hanqing Zeng, Yinglong Xia, Yiwen Meng, Ren Chen, Jiarui Feng, Dongqi Fu, Qifan Wang, Jiayi Liu, Jun Xiao, Xiangjun Fan, Benyu Zhang, Hong Li, Zhining Liu, Hyunsik Yoo, Zhichen Zeng, Tianxin Wei, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2603.10160)  

**Abstract**: Low-rank adapters (LoRAs) are a parameter-efficient finetuning technique that injects trainable low-rank matrices into pretrained models to adapt them to new tasks. Mixture-of-LoRAs models expand neural networks efficiently by routing each layer input to a small subset of specialized LoRAs of the layer. Existing Mixture-of-LoRAs routers assign a learned routing weight to each LoRA to enable end-to-end training of the router. Despite their empirical promise, we observe that the routing weights are typically extremely imbalanced across LoRAs in practice, where only one or two LoRAs often dominate the routing weights. This essentially limits the number of effective LoRAs and thus severely hinders the expressive power of existing Mixture-of-LoRAs models. In this work, we attribute this weakness to the nature of learnable routing weights and rethink the fundamental design of the router. To address this critical issue, we propose a new router designed that we call Reinforcement Routing for Mixture-of-LoRAs (ReMix). Our key idea is using non-learnable routing weights to ensure all active LoRAs to be equally effective, with no LoRA dominating the routing weights. However, our routers cannot be trained directly via gradient descent due to our non-learnable routing weights. Hence, we further propose an unbiased gradient estimator for the router by employing the reinforce leave-one-out (RLOO) technique, where we regard the supervision loss as the reward and the router as the policy in reinforcement learning. Our gradient estimator also enables to scale up training compute to boost the predictive performance of our ReMix. Extensive experiments demonstrate that our proposed ReMix significantly outperform state-of-the-art parameter-efficient finetuning methods under a comparable number of activated parameters. 

---
# Lost in the Middle at Birth: An Exact Theory of Transformer Position Bias 

**Authors**: Borun D Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2603.10123)  

**Abstract**: The ``Lost in the Middle'' phenomenon -- a U-shaped performance curve where LLMs retrieve well from the beginning and end of a context but fail in the middle -- is widely attributed to learned Softmax artifacts or the distance-decay of positional encodings like RoPE. This paper makes a single, precise claim: \emph{the U-shape is already present at initialization, before any training or positional encoding takes effect.} It is an inherent geometric property of the causal decoder with residual connections.
We model multi-layer causal attention as iterated powers of the Cesàro matrix and derive the exact closed-form influence density in the continuous limit. Causal masking forces a logarithmic divergence of gradient influence at the start of the prompt (the Primacy Tail), while residual connections create an isolated $\mathcal{O}(1)$ anchor at the final token (the Recency Delta). Between these extremes lies a factorial dead zone of order $\mathcal{O}(1/(H{-}1)!)$, where $H$ is the network depth, making middle-context retrieval and training structurally hostile. We validate empirically that untrained Qwen2 and GPT-2 architectures exhibit this U-shape at Step~0, and that it is identical with or without RoPE. Comparing initialized and pretrained networks, we show that standard training does not overcome the topological valley, confirming that the U-shape persists as an architectural baseline under standard pretraining objectives.
We do not claim that this bias is insurmountable, nor that interventions such as RoPE modifications are useless. We establish what the baseline is and where it comes from, so that future efforts to overcome it can be precisely targeted. 

---
# CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR 

**Authors**: Sijia Cui, Pengyu Cheng, Jiajun Song, Yongbo Gai, Guojun Zhang, Zhechao Yu, Jianhe Lin, Xiaoxi Jiang, Guanjun Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10101)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has significantly advanced the reasoning capacity of Large Language Models (LLMs). However, RLVR solely relies on final answers as outcome rewards, neglecting the correctness of intermediate reasoning steps. Training on these process-wrong but outcome-correct rollouts can lead to hallucination and answer-copying, severely undermining the model's generalization and robustness. To address this, we incorporate a Contrastive Learning mechanism into the Policy Optimization (CLIPO) to generalize the RLVR process. By optimizing a contrastive loss over successful rollouts, CLIPO steers the LLM to capture the invariant structure shared across correct reasoning paths. This provides a more robust cross-trajectory regularization than the original single-path supervision in RLVR, effectively mitigating step-level reasoning inconsistencies and suppressing hallucinatory artifacts. In experiments, CLIPO consistently improves multiple RLVR baselines across diverse reasoning benchmarks, demonstrating uniform improvements in generalization and robustness for policy optimization of LLMs. Our code and training recipes are available at this https URL. 

---
# Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models 

**Authors**: Anurag Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2603.10071)  

**Abstract**: Time series foundation models (TSFMs) are increasingly deployed in high-stakes domains, yet their internal representations remain opaque. We present the first application of sparse autoencoders (SAEs) to a TSFM, training TopK SAEs on activations of Chronos-T5-Large (710M parameters) across six layers. Through 392 single-feature ablation experiments, we establish that every ablated feature produces a positive CRPS degradation, confirming causal relevance. Our analysis reveals a depth-dependent hierarchy: early encoder layers encode low-level frequency features, the mid-encoder concentrates causally critical change-detection features, and the final encoder compresses a rich but less causally important taxonomy of temporal concepts. The most critical features reside in the mid-encoder (max single-feature Delta CRPS = 38.61), not in the semantically richest final encoder layer, where progressive ablation paradoxically improves forecast quality. These findings demonstrate that mechanistic interpretability transfers effectively to TSFMs and that Chronos-T5 relies on abrupt-dynamics detection rather than periodic pattern recognition. 

---
# Improving Search Agent with One Line of Code 

**Authors**: Jian Li, Dongsheng Chen, Zhenhua Xu, Yizhang Jin, Jiafu Wu, Chengjie Wang, Xiaotong Yuan, Yabiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2603.10069)  

**Abstract**: Tool-based Agentic Reinforcement Learning (TARL) has emerged as a promising paradigm for training search agents to interact with external tools for a multi-turn information-seeking process autonomously. However, we identify a critical training instability that leads to catastrophic model collapse: Importance Sampling Distribution Drift(ISDD). In Group Relative Policy Optimization(GRPO), a widely adopted TARL algorithm, ISDD manifests as a precipitous decline in the importance sampling ratios, which nullifies gradient updates and triggers irreversible training failure. To address this, we propose \textbf{S}earch \textbf{A}gent \textbf{P}olicy \textbf{O}ptimization (\textbf{SAPO}), which stabilizes training via a conditional token-level KL constraint. Unlike hard clipping, which ignores distributional divergence, SAPO selectively penalizes the KL divergence between the current and old policies. Crucially, this penalty is applied only to positive tokens with low probabilities where the policy has shifted excessively, thereby preventing distribution drift while preserving gradient flow. Remarkably, SAPO requires only one-line code modification to standard GRPO, ensuring immediate deployability. Extensive experiments across seven QA benchmarks demonstrate that SAPO achieves \textbf{+10.6\% absolute improvement} (+31.5\% relative) over Search-R1, yielding consistent gains across varying model scales (1.5B, 14B) and families (Qwen, LLaMA). 

---
# ADVERSA: Measuring Multi-Turn Guardrail Degradation and Judge Reliability in Large Language Models 

**Authors**: Harry Owiredu-Ashley  

**Link**: [PDF](https://arxiv.org/pdf/2603.10068)  

**Abstract**: Most adversarial evaluations of large language model (LLM) safety assess single prompts and report binary pass/fail outcomes, which fails to capture how safety properties evolve under sustained adversarial interaction. We present ADVERSA, an automated red-teaming framework that measures guardrail degradation dynamics as continuous per-round compliance trajectories rather than discrete jailbreak events. ADVERSA uses a fine-tuned 70B attacker model (ADVERSA-Red, Llama-3.1-70B-Instruct with QLoRA) that eliminates the attacker-side safety refusals that render off-the-shelf models unreliable as attackers, scoring victim responses on a structured 5-point rubric that treats partial compliance as a distinct measurable state.
We report a controlled experiment across three frontier victim models (Claude Opus 4.6, Gemini 3.1 Pro, GPT-5.2) using a triple-judge consensus architecture in which judge reliability is measured as a first-class research outcome rather than assumed. Across 15 conversations of up to 10 adversarial rounds, we observe a 26.7% jailbreak rate with an average jailbreak round of 1.25, suggesting that in this evaluation setting, successful jailbreaks were concentrated in early rounds rather than accumulating through sustained pressure. We document inter-judge agreement rates, self-judge scoring tendencies, attacker drift as a failure mode in fine-tuned attackers deployed out of their training distribution, and attacker refusals as a previously-underreported confound in victim resistance measurement. All limitations are stated explicitly. Attack prompts are withheld per responsible disclosure policy; all other experimental artifacts are released. 

---
# Tool Receipts, Not Zero-Knowledge Proofs: Practical Hallucination Detection for AI Agents 

**Authors**: Abhinaba Basu  

**Link**: [PDF](https://arxiv.org/pdf/2603.10060)  

**Abstract**: AI agents that execute tasks via tool calls frequently hallucinate results - fabricating tool executions, misstating output counts, or presenting inferences as facts. Recent approaches to verifiable AI inference rely on zero-knowledge proofs, which provide cryptographic guarantees but impose minutes of proving time per query, making them impractical for interactive agents. We propose NabaOS, a lightweight verification framework inspired by Indian epistemology (Nyaya Shastra), which classifies every claim in an LLM response by its epistemic source (pramana): direct tool output (pratyaksha), inference (anumana), external testimony (shabda), absence (abhava), or ungrounded opinion. Our runtime generates HMAC-signed tool execution receipts that the LLM cannot forge, then cross-references claims against these receipts to detect hallucinations in real time. We evaluate on NyayaVerifyBench, a new benchmark of 1,800 agent response scenarios across four languages with injected hallucinations of six types. NabaOS detects 94.2% of fabricated tool references, 87.6% of count misstatements, and 91.3% of false absence claims, with <15ms verification overhead per response. For deep delegation (agents performing multi-step web tasks), our cross-checking protocol catches 78.4% of URL fabrications via independent re-fetching. We compare against five approaches: zkLLM (cryptographic proofs, 180s/query), TOPLOC (locality-sensitive hashing), SPEX (sampling-based proof of execution), tensor commitments, and self-consistency checking. NabaOS achieves the best cost-latency-coverage trade-off for interactive agents: 94.2% coverage at <15ms versus zkLLM's near-perfect coverage at 180,000ms. For interactive agents, practical receipt-based verification provides better cost-benefit than cryptographic proofs, and epistemic classification gives users actionable trust signals rather than binary judgments. 

---
# Training Language Models via Neural Cellular Automata 

**Authors**: Dan Lee, Seungwook Han, Akarsh Kumar, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2603.10055)  

**Abstract**: Pre-training is crucial for large language models (LLMs), as it is when most representations and capabilities are acquired. However, natural language pre-training has problems: high-quality text is finite, it contains human biases, and it entangles knowledge with reasoning. This raises a fundamental question: is natural language the only path to intelligence? We propose using neural cellular automata (NCA) to generate synthetic, non-linguistic data for pre-pre-training LLMs--training on synthetic-then-natural language. NCA data exhibits rich spatiotemporal structure and statistics resembling natural language while being controllable and cheap to generate at scale. We find that pre-pre-training on only 164M NCA tokens improves downstream language modeling by up to 6% and accelerates convergence by up to 1.6x. Surprisingly, this even outperforms pre-pre-training on 1.6B tokens of natural language from Common Crawl with more compute. These gains also transfer to reasoning benchmarks, including GSM8K, HumanEval, and BigBench-Lite. Investigating what drives transfer, we find that attention layers are the most transferable, and that optimal NCA complexity varies by domain: code benefits from simpler dynamics, while math and web text favor more complex ones. These results enable systematic tuning of the synthetic distribution to target domains. More broadly, our work opens a path toward more efficient models with fully synthetic pre-training. 

---
# Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety 

**Authors**: David Gringras  

**Link**: [PDF](https://arxiv.org/pdf/2603.10044)  

**Abstract**: Safety benchmarks evaluate language models in isolation, typically using multiple-choice format; production deployments wrap these models in agentic scaffolds that restructure inputs through reasoning traces, critic agents, and delegation pipelines. We report one of the largest controlled studies of scaffold effects on safety (N = 62,808; six frontier models, four deployment configurations), combining pre-registration, assessor blinding, equivalence testing, and specification curve analysis. Map-reduce scaffolding degrades measured safety (NNH = 14), yet two of three scaffold architectures preserve safety within practically meaningful margins. Investigating the map-reduce degradation revealed a deeper measurement problem: switching from multiple-choice to open-ended format on identical items shifts safety scores by 5-20 percentage points, larger than any scaffold effect. Within-format scaffold comparisons are consistent with practical equivalence under our pre-registered +/-2 pp TOST margin, isolating evaluation format rather than scaffold architecture as the operative variable. Model x scaffold interactions span 35 pp in opposing directions (one model degrades by -16.8 pp on sycophancy under map-reduce while another improves by +18.8 pp on the same benchmark), ruling out universal claims about scaffold safety. A generalisability analysis yields G = 0.000: model safety rankings reverse so completely across benchmarks that no composite safety index achieves non-zero reliability, making per-model, per-configuration testing a necessary minimum standard. We release all code, data, and prompts as ScaffoldSafety. 

---
# Personalized Group Relative Policy Optimization for Heterogenous Preference Alignment 

**Authors**: Jialu Wang, Heinrich Peters, Asad A. Butt, Navid Hashemi, Alireza Hashemi, Pouya M. Ghari, Joseph Hoover, James Rae, Morteza Dehghani  

**Link**: [PDF](https://arxiv.org/pdf/2603.10009)  

**Abstract**: Despite their sophisticated general-purpose capabilities, Large Language Models (LLMs) often fail to align with diverse individual preferences because standard post-training methods, like Reinforcement Learning with Human Feedback (RLHF), optimize for a single, global objective. While Group Relative Policy Optimization (GRPO) is a widely adopted on-policy reinforcement learning framework, its group-based normalization implicitly assumes that all samples are exchangeable, inheriting this limitation in personalized settings. This assumption conflates distinct user reward distributions and systematically biases learning toward dominant preferences while suppressing minority signals. To address this, we introduce Personalized GRPO (P-GRPO), a novel alignment framework that decouples advantage estimation from immediate batch statistics. By normalizing advantages against preference-group-specific reward histories rather than the concurrent generation group, P-GRPO preserves the contrastive signal necessary for learning distinct preferences. We evaluate P-GRPO across diverse tasks and find that it consistently achieves faster convergence and higher rewards than standard GRPO, thereby enhancing its ability to recover and align with heterogeneous preference signals. Our results demonstrate that accounting for reward heterogeneity at the optimization level is essential for building models that faithfully align with diverse human preferences without sacrificing general capabilities. 

---
# MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios 

**Authors**: Shuhuai Li, Jianghao Lin, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.09983)  

**Abstract**: Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-information nature of autoregressive expert activation. In this paper, we propose to repurpose Speculative Decoding (SD) not merely as a compute accelerator, but as an informative lookahead sensor for memory management, supported by our theoretical and empirical analyses. Hence, we introduce MoE-SpAc, an MoE inference framework that integrates a Speculative Utility Estimator to track expert demand, a Heterogeneous Workload Balancer to dynamically partition computation via online integer optimization, and an Asynchronous Execution Engine to unify the prefetching and eviction in the same utility space. Extensive experiments on seven benchmarks demonstrate that MoE-SpAc achieves a 42% improvement in TPS over the SOTA SD-based baseline, and an average 4.04x speedup over all standard baselines. Code is available at this https URL . 

---
# Explainable LLM Unlearning Through Reasoning 

**Authors**: Junfeng Liao, Qizhou Wang, Shanshan Ye, Xin Yu, Ling Chen, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2603.09980)  

**Abstract**: LLM unlearning is essential for mitigating safety, copyright, and privacy concerns in pre-trained large language models (LLMs). Compared to preference alignment, it offers a more explicit way by removing undesirable knowledge characterized by specific unlearning datasets. In previous works, gradient ascent (GA) and its variants have shown promise for implementing unlearning, yet their untargeted nature results in unintended degradation of general capabilities, incomplete removal of knowledge, and the generation of incoherent responses, among many others. We argue that these issues stem from the absence of explicit guidance on what and how models should unlearn. To fill this gap, we introduce a novel unlearning target, reasoning-based unlearning target, which satisfies both the specified unlearning scope and the specified post-unlearning response. Building on this, we propose targeted reasoning unlearning (TRU), which leverages reasoning-based unlearning target as guidance. We employ the target using a cross-entropy supervised loss combined with a GA-based loss, enabling the model to learn reasoning ability for precise knowledge removal while preserving unrelated abilities. We evaluate TRU against strong baselines across multiple benchmarks and LLM backbones, and find that it achieves more reliable unlearning while preserving general capabilities. Moreover, TRU exhibits superior robustness under diverse attack scenarios, stemming from the reasoning ability learned through reasoning-based targets. Overall, our study establishes reasoning-augmented unlearning as a practical paradigm for reliable and explainable LLM unlearning. 

---
# Decoupling Reasoning and Confidence: Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards 

**Authors**: Zhengzhao Ma, Xueru Wen, Boxi Cao, Yaojie Lu, Hongyu Lin, Jinglin Yang, Min He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2603.09117)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) significantly enhances large language models (LLMs) reasoning but severely suffers from calibration degeneration, where models become excessively over-confident in incorrect answers. Previous studies devote to directly incorporating calibration objective into existing optimization target. However, our theoretical analysis demonstrates that there exists a fundamental gradient conflict between the optimization for maximizing policy accuracy and minimizing calibration error. Building on this insight, we propose DCPO, a simple yet effective framework that systematically decouples reasoning and calibration objectives. Extensive experiments demonstrate that our DCPO not only preserves accuracy on par with GRPO but also achieves the best calibration performance and substantially mitigates the over-confidence issue. Our study provides valuable insights and practical solution for more reliable LLM deployment. 

---
# Daily-Omni: Towards Audio-Visual Reasoning with Temporal Alignment across Modalities 

**Authors**: Ziwei Zhou, Rui Wang, Zuxuan Wu, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17862)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) achieve promising performance on visual and audio benchmarks independently. However, the ability of these models to process cross-modal information synchronously remains largely unexplored. We introduce Daily-Omni, a multiple-choice Audio-Visual QA benchmark featuring 684 real-world videos and 1,197 questions spanning 6 task families that explicitly require cross-modal temporal reasoning. To support scalable benchmark construction, we develop a semi-automatic pipeline for annotation, cross-modal consistency refinement, temporal alignment elicitation, and text-only leakage filtering, followed by human verification. We further provide a diagnostic evaluation suite and extensively evaluate 24 foundation models under 37 model--modality settings (Audio+Video / Audio-only / Video-only / Text-only). Finally, we include a training-free modular diagnostic baseline that composes off-the-shelf unimodal models to serve as a diagnostic baseline and to illustrate how explicit temporal alignment signals affect performance. Results indicate that many end-to-end MLLMs still struggle on alignment-critical questions, suggesting that robust cross-modal temporal alignment remains an important open challenge. 

---
