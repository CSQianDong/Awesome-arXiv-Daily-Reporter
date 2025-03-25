# xKV: Cross-Layer SVD for KV-Cache Compression 

**Authors**: Chi-Chih Chang, Chien-Yu Lin, Yash Akhauri, Wei-Cheng Lin, Kai-Chiang Wu, Luis Ceze, Mohamed S. Abdelfattah  

**Link**: [PDF](https://arxiv.org/pdf/2503.18893)  

**Abstract**: Large Language Models (LLMs) with long context windows enable powerful applications but come at the cost of high memory consumption to store the Key and Value states (KV-Cache). Recent studies attempted to merge KV-cache from multiple layers into shared representations, yet these approaches either require expensive pretraining or rely on assumptions of high per-token cosine similarity across layers which generally does not hold in practice. We find that the dominant singular vectors are remarkably well-aligned across multiple layers of the KV-Cache. Exploiting this insight, we propose xKV, a simple post-training method that applies Singular Value Decomposition (SVD) on the KV-Cache of grouped layers. xKV consolidates the KV-Cache of multiple layers into a shared low-rank subspace, significantly reducing KV-Cache sizes. Through extensive evaluations on the RULER long-context benchmark with widely-used LLMs (e.g., Llama-3.1 and Qwen2.5), xKV achieves up to 6.8x higher compression rates than state-of-the-art inter-layer technique while improving accuracy by 2.7%. Moreover, xKV is compatible with the emerging Multi-Head Latent Attention (MLA) (e.g., DeepSeek-Coder-V2), yielding a notable 3x compression rates on coding tasks without performance degradation. These results highlight xKV's strong capability and versatility in addressing memory bottlenecks for long-context LLM inference. Our code is publicly available at: this https URL. 

---
# AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration 

**Authors**: Zhexuan Wang, Yutong Wang, Xuebo Liu, Liang Ding, Miao Zhang, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18891)  

**Abstract**: Multi-agent systems (MAS) based on large language models (LLMs) have demonstrated significant potential in collaborative problem-solving. However, they still face substantial challenges of low communication efficiency and suboptimal task performance, making the careful design of the agents' communication topologies particularly important. Inspired by the management theory that roles in an efficient team are often dynamically adjusted, we propose AgentDropout, which identifies redundant agents and communication across different communication rounds by optimizing the adjacency matrices of the communication graphs and eliminates them to enhance both token efficiency and task performance. Compared to state-of-the-art methods, AgentDropout achieves an average reduction of 21.6% in prompt token consumption and 18.4% in completion token consumption, along with a performance improvement of 1.14 on the tasks. Furthermore, the extended experiments demonstrate that AgentDropout achieves notable domain transferability and structure robustness, revealing its reliability and effectiveness. We release our code at this https URL. 

---
# I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders 

**Authors**: Andrey Galichin, Alexey Dontsov, Polina Druzhinina, Anton Razzhigaev, Oleg Y. Rogov, Elena Tutubalina, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2503.18878)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing. Recent advances have led to the developing of a new class of reasoning LLMs; for example, open-source DeepSeek-R1 has achieved state-of-the-art performance by integrating deep thinking and complex reasoning. Despite these impressive capabilities, the internal reasoning mechanisms of such models remain unexplored. In this work, we employ Sparse Autoencoders (SAEs), a method to learn a sparse decomposition of latent representations of a neural network into interpretable features, to identify features that drive reasoning in the DeepSeek-R1 series of models. First, we propose an approach to extract candidate ''reasoning features'' from SAE representations. We validate these features through empirical analysis and interpretability methods, demonstrating their direct correlation with the model's reasoning abilities. Crucially, we demonstrate that steering these features systematically enhances reasoning performance, offering the first mechanistic account of reasoning in LLMs. Code available at this https URL 

---
# AlphaSpace: Enabling Robotic Actions through Semantic Tokenization and Symbolic Reasoning 

**Authors**: Alan Dao, Dinh Bach Vu, Bui Quang Huy  

**Link**: [PDF](https://arxiv.org/pdf/2503.18769)  

**Abstract**: This paper presents AlphaSpace, a novel methodology designed to enhance the spatial reasoning capabilities of large language models (LLMs) for 3D Cartesian space navigation. AlphaSpace employs a semantics-based tokenization strategy, encoding height information through specialized semantic tokens, and integrates primarily symbolic synthetic reasoning data. This approach enables LLMs to accurately manipulate objects by positioning them at specific [x, y, z] coordinates. Experimental results demonstrate that AlphaSpace significantly outperforms existing models on manipulation subtasks, achieving a total accuracy of 66.67%, compared to 37.5% for GPT-4o and 29.17% for Claude 3.5 Sonnet. 

---
# Synthetic Function Demonstrations Improve Generation in Low-Resource Programming Languages 

**Authors**: Nick McKenna, Xinnuo Xu, Jack Williams, Nick Wilson, Benjamin Van Durme, Christian Poelitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.18760)  

**Abstract**: A key consideration when training an LLM is whether the target language is more or less resourced, whether this is English compared to Welsh, or Python compared to Excel. Typical training data for programming languages consist of real program demonstrations coupled with human-written comments. Here we present novel approaches to the creation of such data for low resource programming languages. We generate fully-synthetic, textbook-quality demonstrations of common library functions in an example domain of Excel formulas, using a teacher model. We then finetune an underperforming student model, and show improvement on 2 question-answering datasets recast into the Excel domain. We show advantages of finetuning over standard, off-the-shelf RAG approaches, which can offer only modest improvement due to the unfamiliar target domain. 

---
# Construction Identification and Disambiguation Using BERT: A Case Study of NPN 

**Authors**: Wesley Scivetti, Nathan Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2503.18751)  

**Abstract**: Construction Grammar hypothesizes that knowledge of a language consists chiefly of knowledge of form-meaning pairs (''constructions'') that include vocabulary, general grammar rules, and even idiosyncratic patterns. Recent work has shown that transformer language models represent at least some constructional patterns, including ones where the construction is rare overall. In this work, we probe BERT's representation of the form and meaning of a minor construction of English, the NPN (noun-preposition-noun) construction -- exhibited in such expressions as face to face and day to day -- which is known to be polysemous. We construct a benchmark dataset of semantically annotated corpus instances (including distractors that superficially resemble the construction). With this dataset, we train and evaluate probing classifiers. They achieve decent discrimination of the construction from distractors, as well as sense disambiguation among true instances of the construction, revealing that BERT embeddings carry indications of the construction's semantics. Moreover, artificially permuting the word order of true construction instances causes them to be rejected, indicating sensitivity to matters of form. We conclude that BERT does latently encode at least some knowledge of the NPN construction going beyond a surface syntactic pattern and lexical cues. 

---
# Predicting the Road Ahead: A Knowledge Graph based Foundation Model for Scene Understanding in Autonomous Driving 

**Authors**: Hongkuan Zhou, Stefan Schmid, Yicong Li, Lavdim Halilaj, Xiangtong Yao, Wei cao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18730)  

**Abstract**: The autonomous driving field has seen remarkable advancements in various topics, such as object recognition, trajectory prediction, and motion planning. However, current approaches face limitations in effectively comprehending the complex evolutions of driving scenes over time. This paper proposes FM4SU, a novel methodology for training a symbolic foundation model (FM) for scene understanding in autonomous driving. It leverages knowledge graphs (KGs) to capture sensory observation along with domain knowledge such as road topology, traffic rules, or complex interactions between traffic participants. A bird's eye view (BEV) symbolic representation is extracted from the KG for each driving scene, including the spatio-temporal information among the objects across the scenes. The BEV representation is serialized into a sequence of tokens and given to pre-trained language models (PLMs) for learning an inherent understanding of the co-occurrence among driving scene elements and generating predictions on the next scenes. We conducted a number of experiments using the nuScenes dataset and KG in various scenarios. The results demonstrate that fine-tuned models achieve significantly higher accuracy in all tasks. The fine-tuned T5 model achieved a next scene prediction accuracy of 86.7%. This paper concludes that FM4SU offers a promising foundation for developing more comprehensive models for scene understanding in autonomous driving. 

---
# Unsupervised Acquisition of Discrete Grammatical Categories 

**Authors**: David Ph. Shakouri, Crit Cremers, Niels O. Schiller  

**Link**: [PDF](https://arxiv.org/pdf/2503.18702)  

**Abstract**: This article presents experiments performed using a computational laboratory environment for language acquisition experiments. It implements a multi-agent system consisting of two agents: an adult language model and a daughter language model that aims to learn the mother language. Crucially, the daughter agent does not have access to the internal knowledge of the mother language model but only to the language exemplars the mother agent generates. These experiments illustrate how this system can be used to acquire abstract grammatical knowledge. We demonstrate how statistical analyses of patterns in the input data corresponding to grammatical categories yield discrete grammatical rules. These rules are subsequently added to the grammatical knowledge of the daughter language model. To this end, hierarchical agglomerative cluster analysis was applied to the utterances consecutively generated by the mother language model. It is argued that this procedure can be used to acquire structures resembling grammatical categories proposed by linguists for natural languages. Thus, it is established that non-trivial grammatical knowledge has been acquired. Moreover, the parameter configuration of this computational laboratory environment determined using training data generated by the mother language model is validated in a second experiment with a test set similarly resulting in the acquisition of non-trivial categories. 

---
# Commander-GPT: Fully Unleashing the Sarcasm Detection Capability of Multi-Modal Large Language Models 

**Authors**: Yazhou Zhang, Chunwang Zou, Bo Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18681)  

**Abstract**: Sarcasm detection, as a crucial research direction in the field of Natural Language Processing (NLP), has attracted widespread attention. Traditional sarcasm detection tasks have typically focused on single-modal approaches (e.g., text), but due to the implicit and subtle nature of sarcasm, such methods often fail to yield satisfactory results. In recent years, researchers have shifted the focus of sarcasm detection to multi-modal approaches. However, effectively leveraging multi-modal information to accurately identify sarcastic content remains a challenge that warrants further exploration. Leveraging the powerful integrated processing capabilities of Multi-Modal Large Language Models (MLLMs) for various information sources, we propose an innovative multi-modal Commander-GPT framework. Inspired by military strategy, we first decompose the sarcasm detection task into six distinct sub-tasks. A central commander (decision-maker) then assigns the best-suited large language model to address each specific sub-task. Ultimately, the detection results from each model are aggregated to identify sarcasm. We conducted extensive experiments on MMSD and MMSD 2.0, utilizing four multi-modal large language models and six prompting strategies. Our experiments demonstrate that our approach achieves state-of-the-art performance, with a 19.3% improvement in F1 score, without necessitating fine-tuning or ground-truth rationales. 

---
# ZeroLM: Data-Free Transformer Architecture Search for Language Models 

**Authors**: Zhen-Song Chen, Hong-Wei Ding, Xian-Jia Wang, Witold Pedrycz  

**Link**: [PDF](https://arxiv.org/pdf/2503.18646)  

**Abstract**: Neural architecture search (NAS) provides a systematic framework for automating the design of neural network architectures, yet its widespread adoption is hindered by prohibitive computational requirements. Existing zero-cost proxy methods, while reducing search overhead, demonstrate inadequate performance in architecture ranking tasks, particularly for Transformer-based models where they often underperform simple parameter counting metrics. Current automated proxy discovery approaches suffer from extended search times, susceptibility to data overfitting, and structural complexity. This paper introduces a novel zero-cost proxy methodology that quantifies model capacity through efficient weight statistics computation while decomposing Transformer architectures into functionally distinct sub-modules, thereby optimizing the balance of their contributions to overall performance. Our comprehensive evaluation demonstrates the superiority of this approach, achieving a Spearman's rho of 0.76 and Kendall's tau of 0.53 on the FlexiBERT benchmark. The proposed method exhibits exceptional computational efficiency while maintaining robust performance across diverse NAS benchmark tasks, offering a practical solution for large-scale architecture search. 

---
# LANGALIGN: Enhancing Non-English Language Models via Cross-Lingual Embedding Alignment 

**Authors**: Jong Myoung Kim, Young-Jun Lee, Ho-Jin Choi, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.18603)  

**Abstract**: While Large Language Models have gained attention, many service developers still rely on embedding-based models due to practical constraints. In such cases, the quality of fine-tuning data directly impacts performance, and English datasets are often used as seed data for training non-English models. In this study, we propose LANGALIGN, which enhances target language processing by aligning English embedding vectors with those of the target language at the interface between the language model and the task header. Experiments on Korean, Japanese, and Chinese demonstrate that LANGALIGN significantly improves performance across all three languages. Additionally, we show that LANGALIGN can be applied in reverse to convert target language data into a format that an English-based model can process. 

---
# LinkAlign: Scalable Schema Linking for Real-World Large-Scale Multi-Database Text-to-SQL 

**Authors**: Yihan Wang, Peiyu Liu, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18596)  

**Abstract**: Schema linking is a critical bottleneck in achieving human-level performance in Text-to-SQL tasks, particularly in real-world large-scale multi-database scenarios. Addressing schema linking faces two major challenges: (1) Database Retrieval: selecting the correct database from a large schema pool in multi-database settings, while filtering out irrelevant ones. (2) Schema Item Grounding: accurately identifying the relevant tables and columns from within a large and redundant schema for SQL generation. To address this, we introduce LinkAlign, a novel framework that can effectively adapt existing baselines to real-world environments by systematically addressing schema linking. Our framework comprises three key steps: multi-round semantic enhanced retrieval and irrelevant information isolation for Challenge 1, and schema extraction enhancement for Challenge 2. We evaluate our method performance of schema linking on the SPIDER and BIRD benchmarks, and the ability to adapt existing Text-to-SQL models to real-world environments on the SPIDER 2.0-lite benchmark. Experiments show that LinkAlign outperforms existing baselines in multi-database settings, demonstrating its effectiveness and robustness. On the other hand, our method ranks highest among models excluding those using long chain-of-thought reasoning LLMs. This work bridges the gap between current research and real-world scenarios, providing a practical solution for robust and scalable schema linking. The codes are available at this https URL. 

---
# ClinText-SP and RigoBERTa Clinical: a new set of open resources for Spanish Clinical NLP 

**Authors**: Guillem García Subies, Álvaro Barbero Jiménez, Paloma Martínez Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2503.18594)  

**Abstract**: We present a novel contribution to Spanish clinical natural language processing by introducing the largest publicly available clinical corpus, ClinText-SP, along with a state-of-the-art clinical encoder language model, RigoBERTa Clinical. Our corpus was meticulously curated from diverse open sources, including clinical cases from medical journals and annotated corpora from shared tasks, providing a rich and diverse dataset that was previously difficult to access. RigoBERTa Clinical, developed through domain-adaptive pretraining on this comprehensive dataset, significantly outperforms existing models on multiple clinical NLP benchmarks. By publicly releasing both the dataset and the model, we aim to empower the research community with robust resources that can drive further advancements in clinical NLP and ultimately contribute to improved healthcare applications. 

---
# Self-Reported Confidence of Large Language Models in Gastroenterology: Analysis of Commercial, Open-Source, and Quantized Models 

**Authors**: Nariman Naderi, Seyed Amir Ahmad Safavi-Naini, Thomas Savage, Zahra Atf, Peter Lewis, Girish Nadkarni, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2503.18562)  

**Abstract**: This study evaluated self-reported response certainty across several large language models (GPT, Claude, Llama, Phi, Mistral, Gemini, Gemma, and Qwen) using 300 gastroenterology board-style questions. The highest-performing models (GPT-o1 preview, GPT-4o, and Claude-3.5-Sonnet) achieved Brier scores of 0.15-0.2 and AUROC of 0.6. Although newer models demonstrated improved performance, all exhibited a consistent tendency towards overconfidence. Uncertainty estimation presents a significant challenge to the safe use of LLMs in healthcare. Keywords: Large Language Models; Confidence Elicitation; Artificial Intelligence; Gastroenterology; Uncertainty Quantification 

---
# Natural Language Processing for Electronic Health Records in Scandinavian Languages: Norwegian, Swedish, and Danish 

**Authors**: Ashenafi Zebene Woldaregay, Jørgen Aarmo Lund, Phuong Dinh Ngo, Mariyam Tayefi, Joel Burman, Stine Hansen, Martin Hylleholt Sillesen, Hercules Dalianis, Robert Jenssen, Lindsetmo Rolf Ole, Karl Øyvind Mikalsen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18539)  

**Abstract**: Background: Clinical natural language processing (NLP) refers to the use of computational methods for extracting, processing, and analyzing unstructured clinical text data, and holds a huge potential to transform healthcare in various clinical tasks. Objective: The study aims to perform a systematic review to comprehensively assess and analyze the state-of-the-art NLP methods for the mainland Scandinavian clinical text. Method: A literature search was conducted in various online databases including PubMed, ScienceDirect, Google Scholar, ACM digital library, and IEEE Xplore between December 2022 and February 2024. Further, relevant references to the included articles were also used to solidify our search. The final pool includes articles that conducted clinical NLP in the mainland Scandinavian languages and were published in English between 2010 and 2024. Results: Out of the 113 articles, 18% (n=21) focus on Norwegian clinical text, 64% (n=72) on Swedish, 10% (n=11) on Danish, and 8% (n=9) focus on more than one language. Generally, the review identified positive developments across the region despite some observable gaps and disparities between the languages. There are substantial disparities in the level of adoption of transformer-based models. In essential tasks such as de-identification, there is significantly less research activity focusing on Norwegian and Danish compared to Swedish text. Further, the review identified a low level of sharing resources such as data, experimentation code, pre-trained models, and rate of adaptation and transfer learning in the region. Conclusion: The review presented a comprehensive assessment of the state-of-the-art Clinical NLP for electronic health records (EHR) text in mainland Scandinavian languages and, highlighted the potential barriers and challenges that hinder the rapid advancement of the field in the region. 

---
# SciClaims: An End-to-End Generative System for Biomedical Claim Analysis 

**Authors**: Raúl Ortega, José Manuel Gómez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2503.18526)  

**Abstract**: Validating key claims in scientific literature, particularly in biomedical research, is essential for ensuring accuracy and advancing knowledge. This process is critical in sectors like the pharmaceutical industry, where rapid scientific progress requires automation and deep domain expertise. However, current solutions have significant limitations. They lack end-to-end pipelines encompassing all claim extraction, evidence retrieval, and verification steps; rely on complex NLP and information retrieval pipelines prone to multiple failure points; and often fail to provide clear, user-friendly justifications for claim verification outcomes. To address these challenges, we introduce SciClaims, an advanced system powered by state-of-the-art large language models (LLMs) that seamlessly integrates the entire scientific claim analysis process. SciClaims outperforms previous approaches in both claim extraction and verification without requiring additional fine-tuning, setting a new benchmark for automated scientific claim analysis. 

---
# Autoregressive Language Models for Knowledge Base Population: A case study in the space mission domain 

**Authors**: Andrés García-Silva, José Manuel Gómez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2503.18502)  

**Abstract**: Knowledge base population KBP plays a crucial role in populating and maintaining knowledge bases up-to-date in organizations by leveraging domain corpora. Motivated by the increasingly large context windows supported by large language models, we propose to fine-tune an autoregressive language model for end-toend KPB. Our case study involves the population of a space mission knowledge graph. To fine-tune the model we generate a dataset for end-to-end KBP tapping into existing domain resources. Our case study shows that fine-tuned language models of limited size can achieve competitive and even higher accuracy than larger models in the KBP task. Smaller models specialized for KBP offer affordable deployment and lower-cost inference. Moreover, KBP specialist models do not require the ontology to be included in the prompt, allowing for more space in the context for additional input text or output serialization. 

---
# MAGIC-VQA: Multimodal And Grounded Inference with Commonsense Knowledge for Visual Question Answering 

**Authors**: Shuo Yang, Siwen Luo, Soyeon Caren Han, Eduard Hovy  

**Link**: [PDF](https://arxiv.org/pdf/2503.18491)  

**Abstract**: Visual Question Answering (VQA) requires reasoning across visual and textual modalities, yet Large Vision-Language Models (LVLMs) often lack integrated commonsense knowledge, limiting their robustness in real-world scenarios. To address this, we introduce MAGIC-VQA, a novel framework that enhances VQA by systematically integrating commonsense knowledge with LVLMs. MAGIC-VQA employs a three-stage process: (1) Explicit Knowledge Integration from external sources, (2) By-Type Post-Processing for contextual refinement, and (3) Implicit Knowledge Augmentation using a Graph Neural Network (GNN) for structured reasoning. While GNNs bring greater depth to structured inference, they enable superior relational inference beyond LVLMs. MAGIC-VQA bridges a key gap by unifying commonsensse knowledge with LVLM-driven reasoning, eliminating the need for extensive pre-training or complex prompt tuning. Our framework achieves state-of-the-art performance on benchmark datasets, significantly improving commonsense reasoning in VQA. 

---
# Whispering in Amharic: Fine-tuning Whisper for Low-resource Language 

**Authors**: Dawit Ketema Gete, Bedru Yimam Ahamed, Tadesse Destaw Belay, Yohannes Ayana Ejigu, Sukairaj Hafiz Imam, Alemu Belay Tessema, Mohammed Oumer Adem, Tadesse Amare Belay, Robert Geislinger, Umma Aliyu Musa, Martin Semmann, Shamsuddeen Hassan Muhammad, Henning Schreiber, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2503.18485)  

**Abstract**: This work explores fine-tuning OpenAI's Whisper automatic speech recognition (ASR) model for Amharic, a low-resource language, to improve transcription accuracy. While the foundational Whisper model struggles with Amharic due to limited representation in its training data, we fine-tune it using datasets like Mozilla Common Voice, FLEURS, and the BDU-speech dataset. The best-performing model, Whispersmall-am, significantly improves when finetuned on a mix of existing FLEURS data and new, unseen Amharic datasets. Training solely on new data leads to poor performance, but combining it with FLEURS data reinforces the model, enabling better specialization in Amharic. We also demonstrate that normalizing Amharic homophones significantly enhances Word Error Rate (WER) and Bilingual Evaluation Understudy (BLEU) scores. This study underscores the importance of fine-tuning strategies and dataset composition for improving ASR in low-resource languages, providing insights for future Amharic speech recognition research. 

---
# Words as Bridges: Exploring Computational Support for Cross-Disciplinary Translation Work 

**Authors**: Calvin Bao, Yow-Ting Shiue, Marine Carpuat, Joel Chan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18471)  

**Abstract**: Scholars often explore literature outside of their home community of study. This exploration process is frequently hampered by field-specific jargon. Past computational work often focuses on supporting translation work by removing jargon through simplification and summarization; here, we explore a different approach that preserves jargon as useful bridges to new conceptual spaces. Specifically, we cast different scholarly domains as different language-using communities, and explore how to adapt techniques from unsupervised cross-lingual alignment of word embeddings to explore conceptual alignments between domain-specific word embedding this http URL developed a prototype cross-domain search engine that uses aligned domain-specific embeddings to support conceptual exploration, and tested this prototype in two case studies. We discuss qualitative insights into the promises and pitfalls of this approach to translation work, and suggest design insights for future interfaces that provide computational support for cross-domain information seeking. 

---
# Teaching LLMs for Step-Level Automatic Math Correction via Reinforcement Learning 

**Authors**: Junsong Li, Jie Zhou, Yutao Yang, Bihao Zhan, Qianjun Pan, Yuyang Ding, Qin Chen, Jiang Bo, Xin Lin, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18432)  

**Abstract**: Automatic math correction aims to check students' solutions to mathematical problems via artificial intelligence technologies. Most existing studies focus on judging the final answer at the problem level, while they ignore detailed feedback on each step in a math problem-solving process, which requires abilities of semantic understanding and reasoning. In this paper, we propose a reinforcement learning (RL)-based method to boost large language model (LLM) for step-level automatic math correction, named StepAMC. Particularly, we convert the step-level automatic math correction within the text classification task into an RL problem to enhance the reasoning capabilities of LLMs. Then, we design a space-constrained policy network to improve the stability of RL. Then, we introduce a fine-grained reward network to convert the binary human feedback into a continuous value. We conduct extensive experiments over two benchmark datasets and the results show that our model outperforms the eleven strong baselines. 

---
# J&H: Evaluating the Robustness of Large Language Models Under Knowledge-Injection Attacks in Legal Domain 

**Authors**: Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng, Chong Wang, Yun Liu, Charles L.A. Clarke, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18360)  

**Abstract**: As the scale and capabilities of Large Language Models (LLMs) increase, their applications in knowledge-intensive fields such as legal domain have garnered widespread attention. However, it remains doubtful whether these LLMs make judgments based on domain knowledge for reasoning. If LLMs base their judgments solely on specific words or patterns, rather than on the underlying logic of the language, the ''LLM-as-judges'' paradigm poses substantial risks in the real-world applications. To address this question, we propose a method of legal knowledge injection attacks for robustness testing, thereby inferring whether LLMs have learned legal knowledge and reasoning logic. In this paper, we propose J&H: an evaluation framework for detecting the robustness of LLMs under knowledge injection attacks in the legal domain. The aim of the framework is to explore whether LLMs perform deductive reasoning when accomplishing legal tasks. To further this aim, we have attacked each part of the reasoning logic underlying these tasks (major premise, minor premise, and conclusion generation). We have collected mistakes that legal experts might make in judicial decisions in the real world, such as typos, legal synonyms, inaccurate external legal statutes retrieval. However, in real legal practice, legal experts tend to overlook these mistakes and make judgments based on logic. However, when faced with these errors, LLMs are likely to be misled by typographical errors and may not utilize logic in their judgments. We conducted knowledge injection attacks on existing general and domain-specific LLMs. Current LLMs are not robust against the attacks employed in our experiments. In addition we propose and compare several methods to enhance the knowledge robustness of LLMs. 

---
# Surgical Action Planning with Large Language Models 

**Authors**: Mengya Xu, Zhongzhen Huang, Jie Zhang, Xiaofan Zhang, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2503.18296)  

**Abstract**: In robot-assisted minimally invasive surgery, we introduce the Surgical Action Planning (SAP) task, which generates future action plans from visual inputs to address the absence of intraoperative predictive planning in current intelligent applications. SAP shows great potential for enhancing intraoperative guidance and automating procedures. However, it faces challenges such as understanding instrument-action relationships and tracking surgical progress. Large Language Models (LLMs) show promise in understanding surgical video content but remain underexplored for predictive decision-making in SAP, as they focus mainly on retrospective analysis. Challenges like data privacy, computational demands, and modality-specific constraints further highlight significant research gaps. To tackle these challenges, we introduce LLM-SAP, a Large Language Models-based Surgical Action Planning framework that predicts future actions and generates text responses by interpreting natural language prompts of surgical goals. The text responses potentially support surgical education, intraoperative decision-making, procedure documentation, and skill analysis. LLM-SAP integrates two novel modules: the Near-History Focus Memory Module (NHF-MM) for modeling historical states and the prompts factory for action planning. We evaluate LLM-SAP on our constructed CholecT50-SAP dataset using models like Qwen2.5 and Qwen2-VL, demonstrating its effectiveness in next-action prediction. Pre-trained LLMs are tested zero-shot, and supervised fine-tuning (SFT) with LoRA is implemented to address data privacy concerns. Our experiments show that Qwen2.5-72B-SFT surpasses Qwen2.5-72B with a 19.3% higher accuracy. 

---
# Fact-checking AI-generated news reports: Can LLMs catch their own lies? 

**Authors**: Jiayi Yao, Haibo Sun, Nianwen Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.18293)  

**Abstract**: In this paper, we evaluate the ability of Large Language Models (LLMs) to assess the veracity of claims in ''news reports'' generated by themselves or other LLMs. Our goal is to determine whether LLMs can effectively fact-check their own content, using methods similar to those used to verify claims made by humans. Our findings indicate that LLMs are more effective at assessing claims in national or international news stories than in local news stories, better at evaluating static information than dynamic information, and better at verifying true claims compared to false ones. We hypothesize that this disparity arises because the former types of claims are better represented in the training data. Additionally, we find that incorporating retrieved results from a search engine in a Retrieval-Augmented Generation (RAG) setting significantly reduces the number of claims an LLM cannot assess. However, this approach also increases the occurrence of incorrect assessments, partly due to irrelevant or low-quality search results. This diagnostic study highlights the need for future research on fact-checking machine-generated reports to prioritize improving the precision and relevance of retrieved information to better support fact-checking efforts. Furthermore, claims about dynamic events and local news may require human-in-the-loop fact-checking systems to ensure accuracy and reliability. 

---
# When is dataset cartography ineffective? Using training dynamics does not improve robustness against Adversarial SQuAD 

**Authors**: Paul K. Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2503.18290)  

**Abstract**: In this paper, I investigate the effectiveness of dataset cartography for extractive question answering on the SQuAD dataset. I begin by analyzing annotation artifacts in SQuAD and evaluate the impact of two adversarial datasets, AddSent and AddOneSent, on an ELECTRA-small model. Using training dynamics, I partition SQuAD into easy-to-learn, ambiguous, and hard-to-learn subsets. I then compare the performance of models trained on these subsets to those trained on randomly selected samples of equal size. Results show that training on cartography-based subsets does not improve generalization to the SQuAD validation set or the AddSent adversarial set. While the hard-to-learn subset yields a slightly higher F1 score on the AddOneSent dataset, the overall gains are limited. These findings suggest that dataset cartography provides little benefit for adversarial robustness in SQuAD-style QA tasks. I conclude by comparing these results to prior findings on SNLI and discuss possible reasons for the observed differences. 

---
# Sun-Shine: A Large Language Model for Tibetan Culture 

**Authors**: Cheng Huang, Fan Gao, Nyima Tashi, Yutong Liu, Xiangxiang Wang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Xiao Feng, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18288)  

**Abstract**: Tibetan, a minority language in China, features a highly intricate grammatical structure, characterized by four verb tenses and a tense system with frequent irregularities, contributing to its extensive inflectional diversity. Recently, advances in Large Language Models (LLMs) have transformed the paradigm in many domains. Despite the success in other fields, current LLMs often fall short in catering to the needs of domain experts like Tibetans, and the potential of LLMs for Tibetan culture is under-explored. The intrinsic reasons are the immense and intricate nature of Tibetan culture as well as the necessity for higher granularity and richness in knowledge. Simultaneously, the complexity and uniqueness of its grammatical structure, coupled with its status as a minority ethnic language, contribute to data scarcity, which remains a fundamental challenge. To alleviate these issues, we introduce Llama-Sunshine (Sun-Shine), the first large language model for Tibetan culture, which is expert in various Tibetan language processing tasks. Sun-Shine incorporates state-of-the-art model architectures optimized for Tibetan's linguistic features. We also propose TIB-STC, a comprehensive dataset comprising diverse Tibetan texts such as literature, religious scripts, news, and conversational data, which is also the first large-scale dataset for Tibetan culture. Though comprehensive experiments, Sun-Shine not only demonstrates a higher level of knowledge expertise for Tibetan culture but also gains preliminary embodied intelligence capabilities in Tibetan language processing tasks, like language modeling, text classification, machine translation, and syntactic analysis. Moreover, it excels in low-resource scenarios, showcasing strong generalization capabilities. 

---
# Bridging Emotions and Architecture: Sentiment Analysis in Modern Distributed Systems 

**Authors**: Mahak Shah, Akaash Vishal Hazarika, Meetu Malhotra, Sachin C. Patil, Joshit Mohanty  

**Link**: [PDF](https://arxiv.org/pdf/2503.18260)  

**Abstract**: Sentiment analysis is a field within NLP that has gained importance because it is applied in various areas such as; social media surveillance, customer feedback evaluation and market research. At the same time, distributed systems allow for effective processing of large amounts of data. Therefore, this paper examines how sentiment analysis converges with distributed systems by concentrating on different approaches, challenges and future investigations. Furthermore, we do an extensive experiment where we train sentiment analysis models using both single node configuration and distributed architecture to bring out the benefits and shortcomings of each method in terms of performance and accuracy. 

---
# Enhancing Multi-Label Emotion Analysis and Corresponding Intensities for Ethiopian Languages 

**Authors**: Tadesse Destaw Belay, Dawit Ketema Gete, Abinew Ali Ayele, Olga Kolesnikova, Grigori Sidorov, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2503.18253)  

**Abstract**: In this digital world, people freely express their emotions using different social media platforms. As a result, modeling and integrating emotion-understanding models are vital for various human-computer interaction tasks such as decision-making, product and customer feedback analysis, political promotions, marketing research, and social media monitoring. As users express different emotions simultaneously in a single instance, annotating emotions in a multilabel setting such as the EthioEmo (Belay et al., 2025) dataset effectively captures this dynamic. Additionally, incorporating intensity, or the degree of emotion, is crucial, as emotions can significantly differ in their expressive strength and impact. This intensity is significant for assessing whether further action is necessary in decision-making processes, especially concerning negative emotions in applications such as healthcare and mental health studies. To enhance the EthioEmo dataset, we include annotations for the intensity of each labeled emotion. Furthermore, we evaluate various state-of-the-art encoder-only Pretrained Language Models (PLMs) and decoder-only Large Language Models (LLMs) to provide comprehensive benchmarking. 

---
# PAD: Towards Efficient Data Generation for Transfer Learning Using Phrase Alignment 

**Authors**: Jong Myoung Kim, Young-Jun_Lee, Ho-Jin Choi, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.18250)  

**Abstract**: Transfer learning leverages the abundance of English data to address the scarcity of resources in modeling non-English languages, such as Korean. In this study, we explore the potential of Phrase Aligned Data (PAD) from standardized Statistical Machine Translation (SMT) to enhance the efficiency of transfer learning. Through extensive experiments, we demonstrate that PAD synergizes effectively with the syntactic characteristics of the Korean language, mitigating the weaknesses of SMT and significantly improving model performance. Moreover, we reveal that PAD complements traditional data construction methods and enhances their effectiveness when combined. This innovative approach not only boosts model performance but also suggests a cost-efficient solution for resource-scarce languages. 

---
# AfroXLMR-Social: Adapting Pre-trained Language Models for African Languages Social Media Text 

**Authors**: Tadesse Destaw Belay, Israel Abebe Azime, Ibrahim Said Ahmad, Idris Abdulmumin, Abinew Ali Ayele, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2503.18247)  

**Abstract**: Pretrained Language Models (PLMs) built from various sources are the foundation of today's NLP progress. Language representations learned by such models achieve strong performance across many tasks with datasets of varying sizes drawn from various sources. We explore a thorough analysis of domain and task adaptive continual pretraining approaches for low-resource African languages and a promising result is shown for the evaluated tasks. We create AfriSocial, a corpus designed for domain adaptive finetuning that passes through quality pre-processing steps. Continual pretraining PLMs using AfriSocial as domain adaptive pretraining (DAPT) data, consistently improves performance on fine-grained emotion classification task of 16 targeted languages from 1% to 28.27% macro F1 score. Likewise, using the task adaptive pertaining (TAPT) approach, further finetuning with small unlabeled but similar task data shows promising results. For example, unlabeled sentiment data (source) for fine-grained emotion classification task (target) improves the base model results by an F1 score ranging from 0.55% to 15.11%. Combining the two methods, DAPT + TAPT, achieves also better results than base models. All the resources will be available to improve low-resource NLP tasks, generally, as well as other similar domain tasks such as hate speech and sentiment tasks. 

---
# ShED-HD: A Shannon Entropy Distribution Framework for Lightweight Hallucination Detection on Edge Devices 

**Authors**: Aneesh Vathul, Daniel Lee, Sheryl Chen, Arthi Tasmia  

**Link**: [PDF](https://arxiv.org/pdf/2503.18242)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities on a broad array of NLP tasks, but their tendency to produce hallucinations$\unicode{x2013}$plausible-sounding but factually incorrect content$\unicode{x2013}$poses severe challenges in high-stakes domains. Existing hallucination detection methods either bear the computational cost of multiple inference passes or sacrifice accuracy for efficiency with single-pass approaches, neither of which is ideal in resource-constrained environments such as edge devices. We propose the Shannon Entropy Distribution Hallucination Detector (ShED-HD), a novel hallucination detection framework that bridges this gap by classifying sequence-level entropy patterns using a lightweight BiLSTM architecture with single-headed attention. In contrast to prior approaches, ShED-HD efficiently detects distinctive uncertainty patterns across entire output sequences, preserving contextual awareness. Through in-depth evaluation on three datasets (BioASQ, TriviaQA, and Jeopardy Questions), we show that ShED-HD significantly outperforms other computationally efficient approaches in the out-of-distribution setting, while achieving comparable performance in the in-distribution setting. ShED-HD facilitates hallucination detection that is low-cost, accurate, and generalizable, improving the credibility of content generated by LLMs in resource-constrained environments where trustworthy AI functionality is crucial. 

---
# Mapping Hymns and Organizing Concepts in the Rigveda: Quantitatively Connecting the Vedic Suktas 

**Authors**: Venkatesh Bollineni, Igor Crk, Eren Gultepe  

**Link**: [PDF](https://arxiv.org/pdf/2503.18226)  

**Abstract**: Accessing and gaining insight into the Rigveda poses a non-trivial challenge due to its extremely ancient Sanskrit language, poetic structure, and large volume of text. By using NLP techniques, this study identified topics and semantic connections of hymns within the Rigveda that were corroborated by seven well-known groupings of hymns. The 1,028 suktas (hymns) from the modern English translation of the Rigveda by Jamison and Brereton were preprocessed and sukta-level embeddings were obtained using, i) a novel adaptation of LSA, presented herein, ii) SBERT, and iii) Doc2Vec embeddings. Following an UMAP dimension reduction of the vectors, the network of suktas was formed using k-nearest neighbours. Then, community detection of topics in the sukta networks was performed with the Louvain, Leiden, and label propagation methods, whose statistical significance of the formed topics were determined using an appropriate null distribution. Only the novel adaptation of LSA using the Leiden method, had detected sukta topic networks that were significant (z = 2.726, p < .01) with a modularity score of 0.944. Of the seven famous sukta groupings analyzed (e.g., creation, funeral, water, etc.) the LSA derived network was successful in all seven cases, while Doc2Vec was not significant and failed to detect the relevant suktas. SBERT detected four of the famous suktas as separate groups, but mistakenly combined three of them into a single mixed group. Also, the SBERT network was not statistically significant. 

---
# LakotaBERT: A Transformer-based Model for Low Resource Lakota Language 

**Authors**: Kanishka Parankusham, Rodrigue Rizk, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2503.18212)  

**Abstract**: Lakota, a critically endangered language of the Sioux people in North America, faces significant challenges due to declining fluency among younger generations. This paper introduces LakotaBERT, the first large language model (LLM) tailored for Lakota, aiming to support language revitalization efforts. Our research has two primary objectives: (1) to create a comprehensive Lakota language corpus and (2) to develop a customized LLM for Lakota. We compiled a diverse corpus of 105K sentences in Lakota, English, and parallel texts from various sources, such as books and websites, emphasizing the cultural significance and historical context of the Lakota language. Utilizing the RoBERTa architecture, we pre-trained our model and conducted comparative evaluations against established models such as RoBERTa, BERT, and multilingual BERT. Initial results demonstrate a masked language modeling accuracy of 51% with a single ground truth assumption, showcasing performance comparable to that of English-based models. We also evaluated the model using additional metrics, such as precision and F1 score, to provide a comprehensive assessment of its capabilities. By integrating AI and linguistic methodologies, we aspire to enhance linguistic diversity and cultural resilience, setting a valuable precedent for leveraging technology in the revitalization of other endangered indigenous languages. 

---
# Exploring Topic Trends in COVID-19 Research Literature using Non-Negative Matrix Factorization 

**Authors**: Divya Patel, Vansh Parikh, Om Patel, Agam Shah, Bhaskar Chaudhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.18182)  

**Abstract**: In this work, we apply topic modeling using Non-Negative Matrix Factorization (NMF) on the COVID-19 Open Research Dataset (CORD-19) to uncover the underlying thematic structure and its evolution within the extensive body of COVID-19 research literature. NMF factorizes the document-term matrix into two non-negative matrices, effectively representing the topics and their distribution across the documents. This helps us see how strongly documents relate to topics and how topics relate to words. We describe the complete methodology which involves a series of rigorous pre-processing steps to standardize the available text data while preserving the context of phrases, and subsequently feature extraction using the term frequency-inverse document frequency (tf-idf), which assigns weights to words based on their frequency and rarity in the dataset. To ensure the robustness of our topic model, we conduct a stability analysis. This process assesses the stability scores of the NMF topic model for different numbers of topics, enabling us to select the optimal number of topics for our analysis. Through our analysis, we track the evolution of topics over time within the CORD-19 dataset. Our findings contribute to the understanding of the knowledge structure of the COVID-19 research landscape, providing a valuable resource for future research in this field. 

---
# GINGER: Grounded Information Nugget-Based Generation of Responses 

**Authors**: Weronika Łajewska, Krisztian Balog  

**Link**: [PDF](https://arxiv.org/pdf/2503.18174)  

**Abstract**: Retrieval-augmented generation (RAG) faces challenges related to factual correctness, source attribution, and response completeness. To address them, we propose a modular pipeline for grounded response generation that operates on information nuggets-minimal, atomic units of relevant information extracted from retrieved documents. The multistage pipeline encompasses nugget detection, clustering, ranking, top cluster summarization, and fluency enhancement. It guarantees grounding in specific facts, facilitates source attribution, and ensures maximum information inclusion within length constraints. Extensive experiments on the TREC RAG'24 dataset evaluated with the AutoNuggetizer framework demonstrate that GINGER achieves state-of-the-art performance on this benchmark. 

---
# Unmasking Deceptive Visuals: Benchmarking Multimodal Large Language Models on Misleading Chart Question Answering 

**Authors**: Zixin Chen, Sicheng Song, Kashun Shum, Yanna Lin, Rui Sheng, Huamin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18172)  

**Abstract**: Misleading chart visualizations, which intentionally manipulate data representations to support specific claims, can distort perceptions and lead to incorrect conclusions. Despite decades of research, misleading visualizations remain a widespread and pressing issue. Recent advances in multimodal large language models (MLLMs) have demonstrated strong chart comprehension capabilities, yet no existing work has systematically evaluated their ability to detect and interpret misleading charts. This paper introduces the Misleading Chart Question Answering (Misleading ChartQA) Benchmark, a large-scale multimodal dataset designed to assess MLLMs in identifying and reasoning about misleading charts. It contains over 3,000 curated examples, covering 21 types of misleaders and 10 chart types. Each example includes standardized chart code, CSV data, and multiple-choice questions with labeled explanations, validated through multi-round MLLM checks and exhausted expert human review. We benchmark 16 state-of-the-art MLLMs on our dataset, revealing their limitations in identifying visually deceptive practices. We also propose a novel pipeline that detects and localizes misleaders, enhancing MLLMs' accuracy in misleading chart interpretation. Our work establishes a foundation for advancing MLLM-driven misleading chart comprehension. We publicly release the sample dataset to support further research in this critical area. 

---
# Evaluating Negative Sampling Approaches for Neural Topic Models 

**Authors**: Suman Adhya, Avishek Lahiri, Debarshi Kumar Sanyal, Partha Pratim Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.18167)  

**Abstract**: Negative sampling has emerged as an effective technique that enables deep learning models to learn better representations by introducing the paradigm of learn-to-compare. The goal of this approach is to add robustness to deep learning models to learn better representation by comparing the positive samples against the negative ones. Despite its numerous demonstrations in various areas of computer vision and natural language processing, a comprehensive study of the effect of negative sampling in an unsupervised domain like topic modeling has not been well explored. In this paper, we present a comprehensive analysis of the impact of different negative sampling strategies on neural topic models. We compare the performance of several popular neural topic models by incorporating a negative sampling technique in the decoder of variational autoencoder-based neural topic models. Experiments on four publicly available datasets demonstrate that integrating negative sampling into topic models results in significant enhancements across multiple aspects, including improved topic coherence, richer topic diversity, and more accurate document classification. Manual evaluations also indicate that the inclusion of negative sampling into neural topic models enhances the quality of the generated topics. These findings highlight the potential of negative sampling as a valuable tool for advancing the effectiveness of neural topic models. 

---
# MathAgent: Leveraging a Mixture-of-Math-Agent Framework for Real-World Multimodal Mathematical Error Detection 

**Authors**: Yibo Yan, Shen Wang, Jiahao Huo, Philip S. Yu, Xuming Hu, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18132)  

**Abstract**: Mathematical error detection in educational settings presents a significant challenge for Multimodal Large Language Models (MLLMs), requiring a sophisticated understanding of both visual and textual mathematical content along with complex reasoning capabilities. Though effective in mathematical problem-solving, MLLMs often struggle with the nuanced task of identifying and categorizing student errors in multimodal mathematical contexts. Therefore, we introduce MathAgent, a novel Mixture-of-Math-Agent framework designed specifically to address these challenges. Our approach decomposes error detection into three phases, each handled by a specialized agent: an image-text consistency validator, a visual semantic interpreter, and an integrative error analyzer. This architecture enables more accurate processing of mathematical content by explicitly modeling relationships between multimodal problems and student solution steps. We evaluate MathAgent on real-world educational data, demonstrating approximately 5% higher accuracy in error step identification and 3% improvement in error categorization compared to baseline models. Besides, MathAgent has been successfully deployed in an educational platform that has served over one million K-12 students, achieving nearly 90% student satisfaction while generating significant cost savings by reducing manual error detection. 

---
# GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks 

**Authors**: Varvara Krechetova, Denis Kochedykov  

**Link**: [PDF](https://arxiv.org/pdf/2503.18129)  

**Abstract**: In this paper, we establish a benchmark for evaluating large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess seven leading commercial LLMs (Sonnet 3.5 and 3.7, Haiku 3.5, Gemini 2.0, GPT-4o, GPT-4o mini, and o3-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks across four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test hallucination rejection. We develop an LLM-as-Judge evaluation framework to compare agent solutions against reference implementations. Results show Sonnet 3.5 and GPT-4o achieve the best overall performance, with Claude models excelling on solvable tasks while OpenAI models better identify unsolvable scenarios. We observe significant differences in token usage, with Anthropic models consuming substantially more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources, providing one more standardized method for ongoing evaluation of LLMs for GeoAI. 

---
# Detection of Somali-written Fake News and Toxic Messages on the Social Media Using Transformer-based Language Models 

**Authors**: Muhidin A. Mohamed, Shuab D. Ahmed, Yahye A. Isse, Hanad M. Mohamed, Fuad M. Hassan, Houssein A. Assowe  

**Link**: [PDF](https://arxiv.org/pdf/2503.18117)  

**Abstract**: The fact that everyone with a social media account can create and share content, and the increasing public reliance on social media platforms as a news and information source bring about significant challenges such as misinformation, fake news, harmful content, etc. Although human content moderation may be useful to an extent and used by these platforms to flag posted materials, the use of AI models provides a more sustainable, scalable, and effective way to mitigate these harmful contents. However, low-resourced languages such as the Somali language face limitations in AI automation, including scarce annotated training datasets and lack of language models tailored to their unique linguistic characteristics. This paper presents part of our ongoing research work to bridge some of these gaps for the Somali language. In particular, we created two human-annotated social-media-sourced Somali datasets for two downstream applications, fake news \& toxicity classification, and developed a transformer-based monolingual Somali language model (named SomBERTa) -- the first of its kind to the best of our knowledge. SomBERTa is then fine-tuned and evaluated on toxic content, fake news and news topic classification datasets. Comparative evaluation analysis of the proposed model against related multilingual models (e.g., AfriBERTa, AfroXLMR, etc) demonstrated that SomBERTa consistently outperformed these comparators in both fake news and toxic content classification tasks while achieving the best average accuracy (87.99%) across all tasks. This research contributes to Somali NLP by offering a foundational language model and a replicable framework for other low-resource languages, promoting digital and AI inclusivity and linguistic diversity. 

---
# Clarifying Misconceptions in COVID-19 Vaccine Sentiment and Stance Analysis and Their Implications for Vaccine Hesitancy Mitigation: A Systematic Review 

**Authors**: Lorena G Barberia, Belinda Lombard, Norton Trevisan Roman, Tatiane C. M. Sousa  

**Link**: [PDF](https://arxiv.org/pdf/2503.18095)  

**Abstract**: Background Advances in machine learning (ML) models have increased the capability of researchers to detect vaccine hesitancy in social media using Natural Language Processing (NLP). A considerable volume of research has identified the persistence of COVID-19 vaccine hesitancy in discourse shared on various social media platforms. Methods Our objective in this study was to conduct a systematic review of research employing sentiment analysis or stance detection to study discourse towards COVID-19 vaccines and vaccination spread on Twitter (officially known as X since 2023). Following registration in the PROSPERO international registry of systematic reviews, we searched papers published from 1 January 2020 to 31 December 2023 that used supervised machine learning to assess COVID-19 vaccine hesitancy through stance detection or sentiment analysis on Twitter. We categorized the studies according to a taxonomy of five dimensions: tweet sample selection approach, self-reported study type, classification typology, annotation codebook definitions, and interpretation of results. We analyzed if studies using stance detection report different hesitancy trends than those using sentiment analysis by examining how COVID-19 vaccine hesitancy is measured, and whether efforts were made to avoid measurement bias. Results Our review found that measurement bias is widely prevalent in studies employing supervised machine learning to analyze sentiment and stance toward COVID-19 vaccines and vaccination. The reporting errors are sufficiently serious that they hinder the generalisability and interpretation of these studies to understanding whether individual opinions communicate reluctance to vaccinate against SARS-CoV-2. Conclusion Improving the reporting of NLP methods is crucial to addressing knowledge gaps in vaccine hesitancy discourse. 

---
# $D^2LoRA$: Data-Driven LoRA Initialization for Low Resource Tasks 

**Authors**: Javad SeraJ, Mohammad Mahdi Mohajeri, Mohammad Javad Dousti  

**Link**: [PDF](https://arxiv.org/pdf/2503.18089)  

**Abstract**: Tuning large language models is essential for optimizing their performance across diverse applications, particularly in scenarios with limited data availability. Tuning large language models in scarce data scenarios is crucial, particularly given that the convergence speed of the LoRA method is lower than that of full fine-tuning. In this paper, we present an analysis of post-training methods including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Odds Ratio Preference Optimization (ORPO) within the context of task-specific learning using the LoRA method. Next we introduce $D^2LoRA$, a data-driven approach for initializing LoRA metrics that enhances training efficiency, especially in limited-data settings. Our experiments compare $D^2LoRA$ with vanilla LoRA in terms of performance and catastrophic forgetting under extremely data-constrained conditions. The results demonstrate that $D^2LoRA$ achieves a 1% improvement GSM8K benchmark and a 2-point improvement in ROUGE score in title generation tasks. $D^2LoRA$ facilitates the adaptation of LLMs to multiple tasks even when task-specific data is scarce, thereby reducing training expenses and offering data cost. 

---
# Temporal Relation Extraction in Clinical Texts: A Span-based Graph Transformer Approach 

**Authors**: Rochana Chaturvedi, Peyman Baghershahi, Sourav Medya, Barbara Di Eugenio  

**Link**: [PDF](https://arxiv.org/pdf/2503.18085)  

**Abstract**: Temporal information extraction from unstructured text is essential for contextualizing events and deriving actionable insights, particularly in the medical domain. We address the task of extracting clinical events and their temporal relations using the well-studied I2B2 2012 Temporal Relations Challenge corpus. This task is inherently challenging due to complex clinical language, long documents, and sparse annotations. We introduce GRAPHTREX, a novel method integrating span-based entity-relation extraction, clinical large pre-trained language models (LPLMs), and Heterogeneous Graph Transformers (HGT) to capture local and global dependencies. Our HGT component facilitates information propagation across the document through innovative global landmarks that bridge distant entities. Our method improves the state-of-the-art with 5.5% improvement in the tempeval $F_1$ score over the previous best and up to 8.9% improvement on long-range relations, which presents a formidable challenge. This work not only advances temporal information extraction but also lays the groundwork for improved diagnostic and prognostic models through enhanced temporal reasoning. 

---
# A Multi-Model Adaptation of Speculative Decoding for Classification 

**Authors**: Somnath Roy, Padharthi Sreekar, Srivatsa Narasimha, Anubhav Anand  

**Link**: [PDF](https://arxiv.org/pdf/2503.18076)  

**Abstract**: The current study introduces a novel adaptation of speculative decoding, repurposed from generation to classification tasks. We propose a multi-model framework employing up to three lightweight worker models and a single, more robust judge model analogous to draft models and target model, respectively, in speculative decoding. The worker models, tasked with the bulk of the computation, independently predict discrete class labels for a given input. When majority worker models agree on a label, it is accepted as the final label, optimizing efficiency by bypassing the computationally expensive judge model. In cases of disagreement, the judge model intervenes to resolve the label. This approach minimizes redundant computation, leverages the redundancy of multiple workers for confidence, and confines the judge model's role to challenging cases, offering a practical balance of efficiency and accuracy. Our analysis suggests that smaller out of the box instruction/chat finetuned worker models with 3 billion parameters (hereafter, 3B) demonstrate a level of alignment with judge models comparable to that of larger finetuned worker models with 7 billion parameters (hereafter, 7B) across both simple and higher order reasoning tasks. The top performing 3B worker model pair achieve an agreement rate of approximately 80-83% for sentiment and around 50-80% for similar ticket when compared to judge models. Additionally, 3B worker models provide a speedup ranging from 2.8x to 9x relative to the judge models, while 7B worker model combinations achieve a speedup ranging from 1.28x to 0.28x 

---
# On the effectiveness of LLMs for automatic grading of open-ended questions in Spanish 

**Authors**: Germán Capdehourat, Isabel Amigo, Brian Lorenzo, Joaquín Trigo  

**Link**: [PDF](https://arxiv.org/pdf/2503.18072)  

**Abstract**: Grading is a time-consuming and laborious task that educators must face. It is an important task since it provides feedback signals to learners, and it has been demonstrated that timely feedback improves the learning process. In recent years, the irruption of LLMs has shed light on the effectiveness of automatic grading. In this paper, we explore the performance of different LLMs and prompting techniques in automatically grading short-text answers to open-ended questions. Unlike most of the literature, our study focuses on a use case where the questions, answers, and prompts are all in Spanish. Experimental results comparing automatic scores to those of human-expert evaluators show good outcomes in terms of accuracy, precision and consistency for advanced LLMs, both open and proprietary. Results are notably sensitive to prompt styles, suggesting biases toward certain words or content in the prompt. However, the best combinations of models and prompt strategies, consistently surpasses an accuracy of 95% in a three-level grading task, which even rises up to more than 98% when the it is simplified to a binary right or wrong rating problem, which demonstrates the potential that LLMs have to implement this type of automation in education applications. 

---
# Mind with Eyes: from Language Reasoning to Multimodal Reasoning 

**Authors**: Zhiyu Lin, Yifei Gao, Xian Zhao, Yunfan Yang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18071)  

**Abstract**: Language models have recently advanced into the realm of reasoning, yet it is through multimodal reasoning that we can fully unlock the potential to achieve more comprehensive, human-like cognitive capabilities. This survey provides a systematic overview of the recent multimodal reasoning approaches, categorizing them into two levels: language-centric multimodal reasoning and collaborative multimodal reasoning. The former encompasses one-pass visual perception and active visual perception, where vision primarily serves a supporting role in language reasoning. The latter involves action generation and state update within reasoning process, enabling a more dynamic interaction between modalities. Furthermore, we analyze the technical evolution of these methods, discuss their inherent challenges, and introduce key benchmark tasks and evaluation metrics for assessing multimodal reasoning performance. Finally, we provide insights into future research directions from the following two perspectives: (i) from visual-language reasoning to omnimodal reasoning and (ii) from multimodal reasoning to multimodal agents. This survey aims to provide a structured overview that will inspire further advancements in multimodal reasoning research. 

---
# Long Is More Important Than Difficult for Training Reasoning Models 

**Authors**: Si Shen, Fei Huang, Zhixiao Zhao, Chang Liu, Tiansheng Zheng, Danhao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18069)  

**Abstract**: Difficult problems, which often result in long reasoning traces, are widely recognized as key factors for enhancing the performance of reasoning models. However, such high-challenge problems are scarce, limiting the size of available datasets. In this paper, we propose a simple method to decouple the reliance on problem difficulty. First, we empirically demonstrate that reasoning length, rather than problem difficulty, primarily influences the performance of trained models. Second, we identify a scaling law on reasoning length, showing that model performance increases in a log-linear fashion as the reasoning data length grows. Finally, we introduce a straightforward technique to generate reasoning data of arbitrary length, and show that synthesized data is effective for training reasoning models. After fine-tuning the Qwen2.5-32B-Instruct language model on our Long1K dataset, we present our model, Long1K-32B, which achieves remarkable performance with only 1,000 training samples, achieving 95.6\% accuracy on MATH, and 71.1\% on GPQA outperforming DeepSeek-R1-Distill-Qwen-32B. The model, code, and dataset are all open-sourced, available at this https URL. 

---
# Dynamic Task Vector Grouping for Efficient Multi-Task Prompt Tuning 

**Authors**: Pieyi Zhang, Richong Zhang, Zhijie Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.18063)  

**Abstract**: Multi-task prompt tuning utilizes multiple high-resource source tasks to improve performance on low-source target tasks. Existing approaches transfer the soft prompt trained by combining all source tasks or a single ``high-similar'' source task one-time-only. However, we find that the optimal transfer performance often comes from a combination of source tasks, which is neither one nor all. Further, we find that the similarity between source and target tasks also changes dynamically during fine-tuning after transfering, making similarity calculation in the initiation stage inadequate. To address these issues, we propose a method called Dynamic Task Vector Grouping (DTVG), whose core ideas contain (1) measuring the task similarity with task vectors instead of soft prompt, (2) grouping the optimal source task combination based on two metrics: {\it target similarity} and {\it knowledge consistency}; (3) dynamically updating the combination in each iteration step. Extensive experiments on the 26 NLP datasets under different settings demonstrate that DTVG effectively groups similar source tasks while reducing negative transfer, achieving the start-of-art performance. 

---
# Investigating Recent Large Language Models for Vietnamese Machine Reading Comprehension 

**Authors**: Anh Duc Nguyen, Hieu Minh Phi, Anh Viet Ngo, Long Hai Trieu, Thai Phuong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.18062)  

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in Machine Reading Comprehension (MRC) tasks; however, their effectiveness for low-resource languages like Vietnamese remains largely unexplored. In this paper, we fine-tune and evaluate two state-of-the-art LLMs: Llama 3 (8B parameters) and Gemma (7B parameters), on ViMMRC, a Vietnamese MRC dataset. By utilizing Quantized Low-Rank Adaptation (QLoRA), we efficiently fine-tune these models and compare their performance against powerful LLM-based baselines. Although our fine-tuned models are smaller than GPT-3 and GPT-3.5, they outperform both traditional BERT-based approaches and these larger models. This demonstrates the effectiveness of our fine-tuning process, showcasing how modern LLMs can surpass the capabilities of older models like BERT while still being suitable for deployment in resource-constrained environments. Through intensive analyses, we explore various aspects of model performance, providing valuable insights into adapting LLMs for low-resource languages like Vietnamese. Our study contributes to the advancement of natural language processing in low-resource languages, and we make our fine-tuned models publicly available at: this https URL. 

---
# Personalized Language Models via Privacy-Preserving Evolutionary Model Merging 

**Authors**: Kyuyoung Kim, Jinwoo Shin, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.18008)  

**Abstract**: Personalization in large language models (LLMs) seeks to tailor models to individual user or user group preferences. Prompt-based methods augment queries with user preference information, whereas training-based methods directly encode preferences into model parameters for more effective personalization. Despite achieving some success in personalizing LLMs, prior methods often fail to directly optimize task-specific metrics and lack explicit privacy-preservation mechanisms. To address these limitations, we propose Privacy-Preserving Model Merging via Evolutionary Algorithms (PriME), a novel approach to personalization that employs gradient-free methods to directly optimize task-specific metrics while preserving user privacy. By incorporating privacy preservation into optimization, PriME produces a personalized module that effectively captures the target user's preferences while minimizing the privacy risks for the users sharing their private information. Experiments on the LaMP benchmark show that PriME outperforms both prompt-based and training-based methods, achieving up to a 45% performance improvement over the prior art. Further analysis shows that PriME achieves a significantly better privacy-utility trade-off, highlighting the potential of evolutionary approaches for privacy-preserving LLM personalization. 

---
# Instructing the Architecture Search for Spatial-temporal Sequence Forecasting with LLM 

**Authors**: Xin Xue, Haoyi Zhou, Tianyu Chen, Shuai Zhang, Yizhou Long, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.17994)  

**Abstract**: Spatial-temporal sequence forecasting (STSF) is a long-standing research problem with widespread real-world applications. Neural architecture search (NAS), which automates the neural network design, has been shown effective in tackling the STSF problem. However, the existing NAS methods for STSF focus on generating architectures in a time-consuming data-driven fashion, which heavily limits their ability to use background knowledge and explore the complicated search trajectory. Large language models (LLMs) have shown remarkable ability in decision-making with comprehensive internal world knowledge, but how it could benefit NAS for STSF remains unexplored. In this paper, we propose a novel NAS method for STSF based on LLM. Instead of directly generate architectures with LLM, We inspire the LLM's capability with a multi-level enhancement mechanism. Specifically, on the step-level, we decompose the generation task into decision steps with powerful prompt engineering and inspire LLM to serve as instructor for architecture search based on its internal knowledge. On the instance-level, we utilize a one-step tuning framework to quickly evaluate the architecture instance and a memory bank to cumulate knowledge to improve LLM's search ability. On the task-level, we propose a two-stage architecture search, balancing the exploration stage and optimization stage, to reduce the possibility of being trapped in local optima. Extensive experimental results demonstrate that our method can achieve competitive effectiveness with superior efficiency against existing NAS methods for STSF. 

---
# Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts 

**Authors**: Beining Xu, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2503.17965)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance on a range of downstream NLP tasks by generating text that closely resembles human writing. However, the ease of achieving this similarity raises concerns from potential malicious uses at scale by bad actors, as LLM-generated text becomes increasingly difficult to discern from human text. Although detection methods have been developed to address this issue, bad actors can further manipulate LLM-generated texts to make them less detectable. In this work, we study how further editing texts with Reinforcement Learning from Human Feedback (RLHF), which aligns model outputs with human preferences, affects (a) the quality of generated texts for two tasks, and (b) the performance of LLM-generated text detectors, looking at both training-based and zero-shot detection methods. Although RLHF improves the quality of LLM-generated texts, we find that it also tends to produce more detectable, lengthy, and repetitive outputs. Additionally, we observe that training-based detectors are vulnerable to short texts and to texts that incorporate code, whereas zero-shot detectors exhibit greater robustness. 

---
# Won: Establishing Best Practices for Korean Financial NLP 

**Authors**: Guijin Son, Hyunwoo Ko, Haneral Jung, Chami Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17963)  

**Abstract**: In this work, we present the first open leaderboard for evaluating Korean large language models focused on finance. Operated for about eight weeks, the leaderboard evaluated 1,119 submissions on a closed benchmark covering five MCQA categories: finance and accounting, stock price prediction, domestic company analysis, financial markets, and financial agent tasks and one open-ended qa task. Building on insights from these evaluations, we release an open instruction dataset of 80k instances and summarize widely used training strategies observed among top-performing models. Finally, we introduce Won, a fully open and transparent LLM built using these best practices. We hope our contributions help advance the development of better and safer financial LLMs for Korean and other languages. 

---
# SLIDE: Sliding Localized Information for Document Extraction 

**Authors**: Divyansh Singh, Manuel Nunez Martinez, Bonnie J. Dorr, Sonja Schmer Galunder  

**Link**: [PDF](https://arxiv.org/pdf/2503.17952)  

**Abstract**: Constructing accurate knowledge graphs from long texts and low-resource languages is challenging, as large language models (LLMs) experience degraded performance with longer input chunks. This problem is amplified in low-resource settings where data scarcity hinders accurate entity and relationship extraction. Contextual retrieval methods, while improving retrieval accuracy, struggle with long documents. They truncate critical information in texts exceeding maximum context lengths of LLMs, significantly limiting knowledge graph construction. We introduce SLIDE (Sliding Localized Information for Document Extraction), a chunking method that processes long documents by generating local context through overlapping windows. SLIDE ensures that essential contextual information is retained, enhancing knowledge graph extraction from documents exceeding LLM context limits. It significantly improves GraphRAG performance, achieving a 24% increase in entity extraction and a 39% improvement in relationship extraction for English. For Afrikaans, a low-resource language, SLIDE achieves a 49% increase in entity extraction and an 82% improvement in relationship extraction. Furthermore, it improves upon state-of-the-art in question-answering metrics such as comprehensiveness, diversity and empowerment, demonstrating its effectiveness in multilingual and resource-constrained settings. 

---
# An Empirical Study of the Role of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Authors**: Riya Naik, Ashwin Srinivasan, Estrid He, Swati Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2503.17936)  

**Abstract**: Natural language as a medium for human-computer interaction has long been anticipated, has been undergoing a sea-change with the advent of Large Language Models (LLMs) with startling capacities for processing and generating language. Many of us now treat LLMs as modern-day oracles, asking it almost any kind of question. Unlike its Delphic predecessor, consulting an LLM does not have to be a single-turn activity (ask a question, receive an answer, leave); and -- also unlike the Pythia -- it is widely acknowledged that answers from LLMs can be improved with additional context. In this paper, we aim to study when we need multi-turn interactions with LLMs to successfully get a question answered; or conclude that a question is unanswerable. We present a neural symbolic framework that models the interactions between human and LLM agents. Through the proposed framework, we define incompleteness and ambiguity in the questions as properties deducible from the messages exchanged in the interaction, and provide results from benchmark problems, in which the answer-correctness is shown to depend on whether or not questions demonstrate the presence of incompleteness or ambiguity (according to the properties we identify). Our results show multi-turn interactions are usually required for datasets which have a high proportion of incompleteness or ambiguous questions; and that that increasing interaction length has the effect of reducing incompleteness or ambiguity. The results also suggest that our measures of incompleteness and ambiguity can be useful tools for characterising interactions with an LLM on question-answeringproblems 

---
# Experience Retrieval-Augmentation with Electronic Health Records Enables Accurate Discharge QA 

**Authors**: Justice Ou, Tinglin Huang, Yilun Zhao, Ziyang Yu, Peiqing Lu, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.17933)  

**Abstract**: To improve the reliability of Large Language Models (LLMs) in clinical applications, retrieval-augmented generation (RAG) is extensively applied to provide factual medical knowledge. However, beyond general medical knowledge from open-ended datasets, clinical case-based knowledge is also critical for effective medical reasoning, as it provides context grounded in real-world patient experiences. Motivated by this, we propose Experience Retrieval Augmentation - ExpRAG framework based on Electronic Health Record (EHR), aiming to offer the relevant context from other patients' discharge reports. ExpRAG performs retrieval through a coarse-to-fine process, utilizing an EHR-based report ranker to efficiently identify similar patients, followed by an experience retriever to extract task-relevant content for enhanced medical reasoning. To evaluate ExpRAG, we introduce DischargeQA, a clinical QA dataset with 1,280 discharge-related questions across diagnosis, medication, and instruction tasks. Each problem is generated using EHR data to ensure realistic and challenging scenarios. Experimental results demonstrate that ExpRAG consistently outperforms a text-based ranker, achieving an average relative improvement of 5.2%, highlighting the importance of case-based knowledge for medical reasoning. 

---
# STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models 

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17932)  

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment. 

---
# WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference 

**Authors**: Youhui Zuo, Sibo Wei, Chen Zhang, Zhuorui Liu, Wenpeng Lu, Dawei Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.17922)  

**Abstract**: With the advancements in long-context inference capabilities of large language models (LLMs), the KV cache has become one of the foundational components. However, its substantial GPU memory consumption makes KV cache compression a key technique for enabling efficient LLM inference in industrial scenarios. While recent studies have focused on optimizing the memory occupied by the KV cache, they overlook two critical factors: preserving semantic coherence and considering task-specific characteristic during compression. To address these limitations, we propose a novel task-adaptive KV cache window selection method, WindowKV. WindowKV dynamically selects local semantic windows consisting of consecutive tokens, according to task-specific characteristics, ensuring the retained KV cache captures continuous, essential context. Additionally, we introduce an intra-group layer KV cache indices sharing strategy to reduce computational overhead, achieving a balance between performance and efficiency. We rigorously evaluate WindowKV on the LongBench benchmark, and the results demonstrate that it maintains a performance comparable to full KV cache retention while using only 12% of the original KV cache, significantly reducing memory requirements. Furthermore, our method also achieves state-of-the-art results in the Needle-in-a-Haystack evaluation, highlighting its effectiveness and robustness. 

---
# MedPlan:A Two-Stage RAG-Based System for Personalized Medical Plan Generation 

**Authors**: Hsin-Ling Hsu, Cong-Tinh Dao, Luning Wang, Zitao Shuai, Thao Nguyen Minh Phan, Jun-En Ding, Chun-Chieh Liao, Pengfei Hu, Xiaoxue Han, Chih-Ho Hsu, Dongsheng Luo, Wen-Chih Peng, Feng Liu, Fang-Ming Hung, Chenwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17900)  

**Abstract**: Despite recent success in applying large language models (LLMs) to electronic health records (EHR), most systems focus primarily on assessment rather than treatment planning. We identify three critical limitations in current approaches: they generate treatment plans in a single pass rather than following the sequential reasoning process used by clinicians; they rarely incorporate patient-specific historical context; and they fail to effectively distinguish between subjective and objective clinical information. Motivated by the SOAP methodology (Subjective, Objective, Assessment, Plan), we introduce MedPlan, a novel framework that structures LLM reasoning to align with real-life clinician workflows. Our approach employs a two-stage architecture that first generates a clinical assessment based on patient symptoms and objective data, then formulates a structured treatment plan informed by this assessment and enriched with patient-specific information through retrieval-augmented generation. Comprehensive evaluation demonstrates that our method significantly outperforms baseline approaches in both assessment accuracy and treatment plan quality. 

---
# Think Before Refusal : Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior 

**Authors**: Shengyun Si, Xinpeng Wang, Guangyao Zhai, Nassir Navab, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2503.17882)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated that fine-tuning and human alignment can render LLMs harmless. In practice, such "harmlessness" behavior is mainly achieved by training models to reject harmful requests, such as "Explain how to burn down my neighbor's house", where the model appropriately declines to respond. However, this approach can inadvertently result in false refusal, where models reject benign queries as well, such as "Tell me how to kill a Python process". In this work, we demonstrate that prompting safety reflection before generating a response can mitigate false refusal behavior. Building on this finding, we introduce the Think-Before-Refusal (TBR) schema and conduct safety-aware instruction fine-tuning incorporating safety reflection. In an ablation study across 15 pre-trained models, we show that models fine-tuned with safety reflection significantly reduce false refusal behavior while maintaining safety and overall performance compared to those fine-tuned without safety reflection. 

---
# Satisfactory Medical Consultation based on Terminology-Enhanced Information Retrieval and Emotional In-Context Learning 

**Authors**: Kaiwen Zuo, Jing Tang, Hanbing Qin, Binli Luo, Ligang He, Shiyan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17876)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have marked significant progress in understanding and responding to medical inquiries. However, their performance still falls short of the standards set by professional consultations. This paper introduces a novel framework for medical consultation, comprising two main modules: Terminology-Enhanced Information Retrieval (TEIR) and Emotional In-Context Learning (EICL). TEIR ensures implicit reasoning through the utilization of inductive knowledge and key terminology retrieval, overcoming the limitations of restricted domain knowledge in public databases. Additionally, this module features capabilities for processing long context. The EICL module aids in generating sentences with high attribute relevance by memorizing semantic and attribute information from unlabelled corpora and applying controlled retrieval for the required information. Furthermore, a dataset comprising 803,564 consultation records was compiled in China, significantly enhancing the model's capability for complex dialogues and proactive inquiry initiation. Comprehensive experiments demonstrate the proposed method's effectiveness in extending the context window length of existing LLMs. The experimental outcomes and extensive data validate the framework's superiority over five baseline models in terms of BLEU and ROUGE performance metrics, with substantial leads in certain capabilities. Notably, ablation studies confirm the significance of the TEIR and EICL components. In addition, our new framework has the potential to significantly improve patient satisfaction in real clinical consulting situations. 

---
# Enhancing Retrieval Systems with Inference-Time Logical Reasoning 

**Authors**: Felix Faltings, Wei Wei, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2503.17860)  

**Abstract**: Traditional retrieval methods rely on transforming user queries into vector representations and retrieving documents based on cosine similarity within an embedding space. While efficient and scalable, this approach often fails to handle complex queries involving logical constructs such as negations, conjunctions, and disjunctions. In this paper, we propose a novel inference-time logical reasoning framework that explicitly incorporates logical reasoning into the retrieval process. Our method extracts logical reasoning structures from natural language queries and then composes the individual cosine similarity scores to formulate the final document scores. This approach enables the retrieval process to handle complex logical reasoning without compromising computational efficiency. Our results on both synthetic and real-world benchmarks demonstrate that the proposed method consistently outperforms traditional retrieval methods across different models and datasets, significantly improving retrieval performance for complex queries. 

---
# Feather-SQL: A Lightweight NL2SQL Framework with Dual-Model Collaboration Paradigm for Small Language Models 

**Authors**: Wenqi Pei, Hailing Xu, Hengyuan Zhao, Shizheng Hou, Han Chen, Zining Zhang, Pingyi Luo, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2503.17811)  

**Abstract**: Natural Language to SQL (NL2SQL) has seen significant advancements with large language models (LLMs). However, these models often depend on closed-source systems and high computational resources, posing challenges in data privacy and deployment. In contrast, small language models (SLMs) struggle with NL2SQL tasks, exhibiting poor performance and incompatibility with existing frameworks. To address these issues, we introduce Feather-SQL, a new lightweight framework tailored for SLMs. Feather-SQL improves SQL executability and accuracy through 1) schema pruning and linking, 2) multi-path and multi-candidate generation. Additionally, we introduce the 1+1 Model Collaboration Paradigm, which pairs a strong general-purpose chat model with a fine-tuned SQL specialist, combining strong analytical reasoning with high-precision SQL generation. Experimental results on BIRD demonstrate that Feather-SQL improves NL2SQL performance on SLMs, with around 10% boost for models without fine-tuning. The proposed paradigm raises the accuracy ceiling of SLMs to 54.76%, highlighting its effectiveness. 

---
# ParsiPy: NLP Toolkit for Historical Persian Texts in Python 

**Authors**: Farhan Farsi, Parnian Fazel, Sepand Haghighi, Sadra Sabouri, Farzaneh Goshtasb, Nadia Hajipour, Ehsaneddin Asgari, Hossein Sameti  

**Link**: [PDF](https://arxiv.org/pdf/2503.17810)  

**Abstract**: The study of historical languages presents unique challenges due to their complex orthographic systems, fragmentary textual evidence, and the absence of standardized digital representations of text in those languages. Tackling these challenges needs special NLP digital tools to handle phonetic transcriptions and analyze ancient texts. This work introduces ParsiPy, an NLP toolkit designed to facilitate the analysis of historical Persian languages by offering modules for tokenization, lemmatization, part-of-speech tagging, phoneme-to-transliteration conversion, and word embedding. We demonstrate the utility of our toolkit through the processing of Parsig (Middle Persian) texts, highlighting its potential for expanding computational methods in the study of historical languages. Through this work, we contribute to computational philology, offering tools that can be adapted for the broader study of ancient texts and their digital preservation. 

---
# Relation Extraction with Instance-Adapted Predicate Descriptions 

**Authors**: Yuhang Jiang, Ramakanth Kavuluru  

**Link**: [PDF](https://arxiv.org/pdf/2503.17799)  

**Abstract**: Relation extraction (RE) is a standard information extraction task playing a major role in downstream applications such as knowledge discovery and question answering. Although decoder-only large language models are excelling in generative tasks, smaller encoder models are still the go to architecture for RE. In this paper, we revisit fine-tuning such smaller models using a novel dual-encoder architecture with a joint contrastive and cross-entropy loss. Unlike previous methods that employ a fixed linear layer for predicate representations, our approach uses a second encoder to compute instance-specific predicate representations by infusing them with real entity spans from corresponding input instances. We conducted experiments on two biomedical RE datasets and two general domain datasets. Our approach achieved F1 score improvements ranging from 1% to 2% over state-of-the-art methods with a simple but elegant formulation. Ablation studies justify the importance of various components built into the proposed architecture. 

---
# Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes 

**Authors**: Sharan Maiya, Yinhong Liu, Ramit Debnath, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2503.17755)  

**Abstract**: Large Language Models (LLMs) are often used as automated judges to evaluate text, but their effectiveness can be hindered by various unintentional biases. We propose using linear classifying probes, trained by leveraging differences between contrasting pairs of prompts, to directly access LLMs' latent knowledge and extract more accurate preferences. Through extensive experiments using models of varying size from four different families and six diverse datasets assessing text quality evaluation and common sense reasoning, we demonstrate that both supervised and unsupervised probing approaches consistently outperform traditional generation-based judgement while maintaining similar computational costs. These probes generalise under domain shifts and can even outperform finetuned evaluators with the same training data size. Our results suggest linear probing offers an accurate, robust and computationally efficient approach for LLM-as-judge tasks while providing interpretable insights into how models encode judgement-relevant knowledge. Our data and code will be openly released in the future. 

---
# Building Resource-Constrained Language Agents: A Korean Case Study on Chemical Toxicity Information 

**Authors**: Hojun Cho, Donghu Kim, Soyoung Yang, Chan Lee, Hunjoo Lee, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17753)  

**Abstract**: Language agents powered by large language models (LLMs) face significant deployment challenges in resource-constrained environments, particularly for specialized domains and less-common languages. This paper presents Tox-chat, a Korean chemical toxicity information agent devised within these limitations. We propose two key innovations: a context-efficient architecture that reduces token consumption through hierarchical section search, and a scenario-based dialogue generation methodology that effectively distills tool-using capabilities from larger models. Experimental evaluations demonstrate that our fine-tuned 8B parameter model substantially outperforms both untuned models and baseline approaches, in terms of DB faithfulness and preference. Our work offers valuable insights for researchers developing domain-specific language agents under practical constraints. 

---
# Enhancing Arabic Automated Essay Scoring with Synthetic Data and Error Injection 

**Authors**: Chatrine Qwaider, Bashar Alhafni, Kirill Chirkunov, Nizar Habash, Ted Briscoe  

**Link**: [PDF](https://arxiv.org/pdf/2503.17739)  

**Abstract**: Automated Essay Scoring (AES) plays a crucial role in assessing language learners' writing quality, reducing grading workload, and providing real-time feedback. Arabic AES systems are particularly challenged by the lack of annotated essay datasets. This paper presents a novel framework leveraging Large Language Models (LLMs) and Transformers to generate synthetic Arabic essay datasets for AES. We prompt an LLM to generate essays across CEFR proficiency levels and introduce controlled error injection using a fine-tuned Standard Arabic BERT model for error type prediction. Our approach produces realistic human-like essays, contributing a dataset of 3,040 annotated essays. Additionally, we develop a BERT-based auto-marking system for accurate and scalable Arabic essay evaluation. Experimental results demonstrate the effectiveness of our framework in improving Arabic AES performance. 

---
# Can LLMs Automate Fact-Checking Article Writing? 

**Authors**: Dhruv Sahnan, David Corney, Irene Larraz, Giovanni Zagni, Ruben Miguez, Zhuohan Xie, Iryna Gurevych, Elizabeth Churchill, Tanmoy Chakraborty, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2503.17684)  

**Abstract**: Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. We argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction. 

---
# Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning 

**Authors**: Ke Ji, Yixin Lian, Linxu Li, Jingsheng Gao, Weiyuan Li, Bin Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.17662)  

**Abstract**: In recent years, large language models (LLMs) have achieved breakthrough progress in many dialogue generation tasks. However, their lack of emotion and fine-grained role awareness limits the model's ability to provide personalized and diverse interactions further. Current methods face high costs in collecting high-quality annotated data for scenarios such as role-playing, and traditional human alignment methods are difficult to deploy due to the inherent diversity of model behavior in role-playing scenarios. Inspired by the alignment of models for safety behaviors through RLHF (Reinforcement Learning from Human Feedback), in this paper, we revisit model role-playing behavior from the perspective of persona alignment and propose a novel annotation-free framework named \textbf{\underline{P}}ersona-Aware \textbf{\underline{C}}ontrastive \textbf{\underline{L}}earning (PCL) to align LLMs' behavior during role-playing, enhancing the model's role consistency. Specifically, we first design a role chain method to encourage the model to self-question based on the role characteristics and dialogue context to adjust personality consistency. Then, we further enhance the model's role-playing strategy through iterative contrastive learning between the use of role characteristics and not. Experiments on both black-box and white-box LLMs show that LLMs equipped with PCL significantly outperform vanilla LLMs under automatic evaluation methods (CharEval \& GPT-4) and human expert evaluation. 

---
# GPBench: A Comprehensive and Fine-Grained Benchmark for Evaluating Large Language Models as General Practitioners 

**Authors**: Zheqing Li, Yiying Yang, Jiping Lang, Wenhao Jiang, Yuhang Zhao, Shuang Li, Dingqian Wang, Zhu Lin, Xuanna Li, Yuze Tang, Jiexian Qiu, Xiaolin Lu, Hongji Yu, Shuang Chen, Yuhua Bi, Xiaofei Zeng, Yixian Chen, Junrong Chen, Lin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.17599)  

**Abstract**: General practitioners (GPs) serve as the cornerstone of primary healthcare systems by providing continuous and comprehensive medical services. However, due to community-oriented nature of their practice, uneven training and resource gaps, the clinical proficiency among GPs can vary significantly across regions and healthcare settings. Currently, Large Language Models (LLMs) have demonstrated great potential in clinical and medical applications, making them a promising tool for supporting general practice. However, most existing benchmarks and evaluation frameworks focus on exam-style assessments-typically multiple-choice question-lack comprehensive assessment sets that accurately mirror the real-world scenarios encountered by GPs. To evaluate how effectively LLMs can make decisions in the daily work of GPs, we designed GPBench, which consists of both test questions from clinical practice and a novel evaluation framework. The test set includes multiple-choice questions that assess fundamental knowledge of general practice, as well as realistic, scenario-based problems. All questions are meticulously annotated by experts, incorporating rich fine-grained information related to clinical management. The proposed LLM evaluation framework is based on the competency model for general practice, providing a comprehensive methodology for assessing LLM performance in real-world settings. As the first large-model evaluation set targeting GP decision-making scenarios, GPBench allows us to evaluate current mainstream LLMs. Expert assessment and evaluation reveal that in areas such as disease staging, complication recognition, treatment detail, and medication usage, these models exhibit at least ten major shortcomings. Overall, existing LLMs are not yet suitable for independent use in real-world GP working scenarios without human oversight. 

---
# Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility 

**Authors**: Suet-Ying Lam, Qingcheng Zeng, Jingyi Wu, Rob Voigt  

**Link**: [PDF](https://arxiv.org/pdf/2503.17579)  

**Abstract**: Whether large language models (LLMs) process language similarly to humans has been the subject of much theoretical and practical debate. We examine this question through the lens of the production-interpretation distinction found in human sentence processing and evaluate the extent to which instruction-tuned LLMs replicate this distinction. Using an empirically documented asymmetry between production and interpretation in humans for implicit causality verbs as a testbed, we find that some LLMs do quantitatively and qualitatively reflect human-like asymmetries between production and interpretation. We demonstrate that whether this behavior holds depends upon both model size - with larger models more likely to reflect human-like patterns and the choice of meta-linguistic prompts used to elicit the behavior. 

---
# Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models 

**Authors**: Linlu Qiu, Fei Sha, Kelsey Allen, Yoon Kim, Tal Linzen, Sjoerd van Steenkiste  

**Link**: [PDF](https://arxiv.org/pdf/2503.17523)  

**Abstract**: Artificial intelligence systems based on large language models (LLMs) are increasingly used as agents that interact with users and with the world. To do so successfully, LLMs need to construct internal representations of the world and form probabilistic beliefs about those representations. To provide a user with personalized recommendations, for example, the LLM needs to gradually infer the user's preferences, over the course of multiple interactions. To evaluate whether contemporary LLMs are able to do so, we use the Bayesian inference framework from probability theory, which lays out the optimal way to update an agent's beliefs as it receives new information. We first show that the LLMs do not update their beliefs as expected from the Bayesian framework, and that consequently their predictions do not improve as expected as more information becomes available, even less so than we find is the case for humans. To address this issue, we teach the LLMs to reason in a Bayesian manner by training them to mimic the predictions of an optimal Bayesian model. We find that this approach not only significantly improves the LLM's performance on the particular recommendation task it is trained on, but also enables generalization to other tasks. This suggests that this method endows the LLM with broader Bayesian reasoning skills. More generally, our results indicate that LLMs can learn about reasoning strategies effectively and generalize those skills to new domains, which in part explains LLMs' empirical success. 

---
# Language Models May Verbatim Complete TextThey Were Not Explicitly Trained On 

**Authors**: Ken Ziyu Liu, Christopher A. Choquette-Choo, Matthew Jagielski, Peter Kairouz, Sanmi Koyejo, Percy Liang, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2503.17514)  

**Abstract**: An important question today is whether a given text was used to train a large language model (LLM). A \emph{completion} test is often employed: check if the LLM completes a sufficiently complex text. This, however, requires a ground-truth definition of membership; most commonly, it is defined as a member based on the $n$-gram overlap between the target text and any text in the dataset. In this work, we demonstrate that this $n$-gram based membership definition can be effectively gamed. We study scenarios where sequences are \emph{non-members} for a given $n$ and we find that completion tests still succeed. We find many natural cases of this phenomenon by retraining LLMs from scratch after removing all training samples that were completed; these cases include exact duplicates, near-duplicates, and even short overlaps. They showcase that it is difficult to find a single viable choice of $n$ for membership definitions. Using these insights, we design adversarial datasets that can cause a given target sequence to be completed without containing it, for any reasonable choice of $n$. Our findings highlight the inadequacy of $n$-gram membership, suggesting membership definitions fail to account for auxiliary information available to the training algorithm. 

---
# Follow-up Question Generation For Enhanced Patient-Provider Conversations 

**Authors**: Joseph Gatto, Parker Seegmiller, Timothy Burdick, Inas S. Khayal, Sarah DeLozier, Sarah M. Preum  

**Link**: [PDF](https://arxiv.org/pdf/2503.17509)  

**Abstract**: Follow-up question generation is an essential feature of dialogue systems as it can reduce conversational ambiguity and enhance modeling complex interactions. Conversational contexts often pose core NLP challenges such as (i) extracting relevant information buried in fragmented data sources, and (ii) modeling parallel thought processes. These two challenges occur frequently in medical dialogue as a doctor asks questions based not only on patient utterances but also their prior EHR data and current diagnostic hypotheses. Asking medical questions in asynchronous conversations compounds these issues as doctors can only rely on static EHR information to motivate follow-up questions.
To address these challenges, we introduce FollowupQ, a novel framework for enhancing asynchronous medical conversation. FollowupQ is a multi-agent framework that processes patient messages and EHR data to generate personalized follow-up questions, clarifying patient-reported medical conditions. FollowupQ reduces requisite provider follow-up communications by 34%. It also improves performance by 17% and 5% on real and synthetic data, respectively. We also release the first public dataset of asynchronous medical messages with linked EHR data alongside 2,300 follow-up questions written by clinical experts for the wider NLP research community. 

---
# Judge Anything: MLLM as a Judge Across Any Modality 

**Authors**: Shu Pu, Yaochen Wang, Dongping Chen, Yuhang Chen, Guohao Wang, Qi Qin, Zhongyi Zhang, Zhiyuan Zhang, Zetong Zhou, Shuang Gong, Yi Gui, Yao Wan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17489)  

**Abstract**: Evaluating generative foundation models on open-ended multimodal understanding (MMU) and generation (MMG) tasks across diverse modalities (e.g., images, audio, video) poses significant challenges due to the complexity of cross-modal interactions. To this end, the idea of utilizing Multimodal LLMs (MLLMs) as automated judges has emerged, with encouraging results in assessing vision-language understanding tasks. Moving further, this paper extends MLLM-as-a-Judge across modalities to a unified manner by introducing two benchmarks, TaskAnything and JudgeAnything, to respectively evaluate the overall performance and judging capabilities of MLLMs across any-to-any modality tasks. Specifically, TaskAnything evaluates the MMU and MMG capabilities across 15 any-to-any modality categories, employing 1,500 queries curated from well-established benchmarks. Furthermore, JudgeAnything evaluates the judging capabilities of 5 advanced (e.g., GPT-4o and Gemini-2.0-Flash) from the perspectives of Pair Comparison and Score Evaluation, providing a standardized testbed that incorporates human judgments and detailed rubrics. Our extensive experiments reveal that while these MLLMs show promise in assessing MMU (i.e., achieving an average of 66.55% in Pair Comparison setting and 42.79% in Score Evaluation setting), they encounter significant challenges with MMG tasks (i.e., averaging only 53.37% in Pair Comparison setting and 30.05% in Score Evaluation setting), exposing cross-modality biases and hallucination issues. To address this, we present OmniArena, an automated platform for evaluating omni-models and multimodal reward models. Our work highlights the need for fairer evaluation protocols and stronger alignment with human preferences. The source code and dataset are publicly available at: this https URL. 

---
# SaudiCulture: A Benchmark for Evaluating Large Language Models Cultural Competence within Saudi Arabia 

**Authors**: Lama Ayash, Hassan Alhuzali, Ashwag Alasmari, Sultan Aloufi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17485)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing; however, they often struggle to accurately capture and reflect cultural nuances. This research addresses this challenge by focusing on Saudi Arabia, a country characterized by diverse dialects and rich cultural traditions. We introduce SaudiCulture, a novel benchmark designed to evaluate the cultural competence of LLMs within the distinct geographical and cultural contexts of Saudi Arabia. SaudiCulture is a comprehensive dataset of questions covering five major geographical regions, such as West, East, South, North, and Center, along with general questions applicable across all regions. The dataset encompasses a broad spectrum of cultural domains, including food, clothing, entertainment, celebrations, and crafts. To ensure a rigorous evaluation, SaudiCulture includes questions of varying complexity, such as open-ended, single-choice, and multiple-choice formats, with some requiring multiple correct answers. Additionally, the dataset distinguishes between common cultural knowledge and specialized regional aspects. We conduct extensive evaluations on five LLMs, such as GPT-4, Llama 3.3, FANAR, Jais, and AceGPT, analyzing their performance across different question types and cultural contexts. Our findings reveal that all models experience significant performance declines when faced with highly specialized or region-specific questions, particularly those requiring multiple correct responses. Additionally, certain cultural categories are more easily identifiable than others, further highlighting inconsistencies in LLMs cultural understanding. These results emphasize the importance of incorporating region-specific knowledge into LLMs training to enhance their cultural competence. 

---
# ConvoGen: Enhancing Conversational AI with Synthetic Data: A Multi-Agent Approach 

**Authors**: Reem Gody, Mahmoud Goudy, Ahmed Y. Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2503.17460)  

**Abstract**: In this paper, we present ConvoGen: an innovative framework for generating synthetic conversational data using multi-agent systems. Our method leverages few-shot learning and introduces iterative sampling from a dynamically updated few-shot hub to create diverse and realistic conversational scenarios. The generated data has numerous applications, including training and evaluating conversational AI models, and augmenting existing datasets for tasks like conversational intent classification or conversation summarization. Our experiments demonstrate the effectiveness of this method in producing high-quality diverse synthetic conversational data, highlighting its potential to enhance the development and evaluation of conversational AI systems. 

---
# Language-specific Neurons Do Not Facilitate Cross-Lingual Transfer 

**Authors**: Soumen Kumar Mondal, Sayambhu Sen, Abhishek Singhania, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17456)  

**Abstract**: Multilingual large language models (LLMs) aim towards robust natural language understanding across diverse languages, yet their performance significantly degrades on low-resource languages. This work explores whether existing techniques to identify language-specific neurons can be leveraged to enhance cross-lingual task performance of lowresource languages. We conduct detailed experiments covering existing language-specific neuron identification techniques (such as Language Activation Probability Entropy and activation probability-based thresholding) and neuron-specific LoRA fine-tuning with models like Llama 3.1 and Mistral Nemo. We find that such neuron-specific interventions are insufficient to yield cross-lingual improvements on downstream tasks (XNLI, XQuAD) in lowresource languages. This study highlights the challenges in achieving cross-lingual generalization and provides critical insights for multilingual LLMs. 

---
# Beyond Negation Detection: Comprehensive Assertion Detection Models for Clinical NLP 

**Authors**: Veysel Kocaman, Yigit Gul, M. Aytug Kaya, Hasham Ul Haq, Mehmet Butgul, Cabir Celik, David Talby  

**Link**: [PDF](https://arxiv.org/pdf/2503.17425)  

**Abstract**: Assertion status detection is a critical yet often overlooked component of clinical NLP, essential for accurately attributing extracted medical facts. Past studies have narrowly focused on negation detection, leading to underperforming commercial solutions such as AWS Medical Comprehend, Azure AI Text Analytics, and GPT-4o due to their limited domain adaptation. To address this gap, we developed state-of-the-art assertion detection models, including fine-tuned LLMs, transformer-based classifiers, few-shot classifiers, and deep learning (DL) approaches. We evaluated these models against cloud-based commercial API solutions, the legacy rule-based NegEx approach, and GPT-4o. Our fine-tuned LLM achieves the highest overall accuracy (0.962), outperforming GPT-4o (0.901) and commercial APIs by a notable margin, particularly excelling in Present (+4.2%), Absent (+8.4%), and Hypothetical (+23.4%) assertions. Our DL-based models surpass commercial solutions in Conditional (+5.3%) and Associated-with-Someone-Else (+10.1%) categories, while the few-shot classifier offers a lightweight yet highly competitive alternative (0.929), making it ideal for resource-constrained environments. Integrated within Spark NLP, our models consistently outperform black-box commercial solutions while enabling scalable inference and seamless integration with medical NER, Relation Extraction, and Terminology Resolution. These results reinforce the importance of domain-adapted, transparent, and customizable clinical NLP solutions over general-purpose LLMs and proprietary APIs. 

---
# A Comprehensive Survey on Long Context Language Modeling 

**Authors**: Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, Yuanxing Zhang, Zhuo Chen, Hangyu Guo, Shilong Li, Ziqiang Liu, Yong Shan, Yifan Song, Jiayi Tian, Wenhao Wu, Zhejian Zhou, Ruijie Zhu, Junlan Feng, Yang Gao, Shizhu He, Zhoujun Li, Tianyu Liu, Fanyu Meng, Wenbo Su, Yingshui Tan, Zili Wang, Jian Yang, Wei Ye, Bo Zheng, Wangchunshu Zhou, Wenhao Huang, Sujian Li, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17407)  

**Abstract**: Efficient processing of long contexts has been a persistent pursuit in Natural Language Processing. With the growing number of long documents, dialogues, and other textual data, it is important to develop Long Context Language Models (LCLMs) that can process and analyze extensive inputs in an effective and efficient way. In this paper, we present a comprehensive survey on recent advances in long-context modeling for large language models. Our survey is structured around three key aspects: how to obtain effective and efficient LCLMs, how to train and deploy LCLMs efficiently, and how to evaluate and analyze LCLMs comprehensively. For the first aspect, we discuss data strategies, architectural designs, and workflow approaches oriented with long context processing. For the second aspect, we provide a detailed examination of the infrastructure required for LCLM training and inference. For the third aspect, we present evaluation paradigms for long-context comprehension and long-form generation, as well as behavioral analysis and mechanism interpretability of LCLMs. Beyond these three key aspects, we thoroughly explore the diverse application scenarios where existing LCLMs have been deployed and outline promising future development directions. This survey provides an up-to-date review of the literature on long-context LLMs, which we wish to serve as a valuable resource for both researchers and engineers. An associated GitHub repository collecting the latest papers and repos is available at: \href{this https URL}{\color[RGB]{175,36,67}{LCLM-Horizon}}. 

---
# ChatGPT or A Silent Everywhere Helper: A Survey of Large Language Models 

**Authors**: Azim Akhtarshenas, Afshin Dini, Navid Ayoobi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17403)  

**Abstract**: Large Language Models (LLMs) have revo lutionized natural language processing Natural Language Processing (NLP), with Chat Generative Pre-trained Transformer (ChatGPT) standing out as a notable exampledue to its advanced capabilities and widespread applications. This survey provides a comprehensive analysis of ChatGPT, exploring its architecture, training processes, and functionalities. We examine its integration into various domains across industries such as customer service, education, healthcare, and entertainment. A comparative analysis with other LLMs highlights ChatGPT's unique features and performance metrics. Regarding benchmarks, the paper examines ChatGPT's comparative performance against other LLMs and discusses potential risks such as misinformation, bias, and data privacy concerns. Additionally, we offer a number of figures and tables that outline the backdrop of the discussion, the main ideas of the article, the numerous LLM models, a thorough list of datasets used for pre-training, fine-tuning, and evaluation, as well as particular LLM applications with pertinent references. Finally, we identify future research directions and technological advancements, underscoring the evolving landscape of LLMs and their profound impact on artificial intelligence Artificial Intelligence (AI) and society. 

---
# Exploring Training and Inference Scaling Laws in Generative Retrieval 

**Authors**: Hongru Cai, Yongqi Li, Ruifeng Yuan, Wenjie Wang, Zhen Zhang, Wenjie Li, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.18941)  

**Abstract**: Generative retrieval has emerged as a novel paradigm that leverages large language models (LLMs) to autoregressively generate document identifiers. Although promising, the mechanisms that underpin its performance and scalability remain largely unclear. We conduct a systematic investigation of training and inference scaling laws in generative retrieval, exploring how model size, training data scale, and inference-time compute jointly influence retrieval performance. To address the lack of suitable metrics, we propose a novel evaluation measure inspired by contrastive entropy and generation loss, providing a continuous performance signal that enables robust comparisons across diverse generative retrieval methods. Our experiments show that n-gram-based methods demonstrate strong alignment with both training and inference scaling laws, especially when paired with larger LLMs. Furthermore, increasing inference computation yields substantial performance gains, revealing that generative retrieval can significantly benefit from higher compute budgets at inference. Across these settings, LLaMA models consistently outperform T5 models, suggesting a particular advantage for larger decoder-only models in generative retrieval. Taken together, our findings underscore that model sizes, data availability, and inference computation interact to unlock the full potential of generative retrieval, offering new insights for designing and optimizing future systems. 

---
# SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild 

**Authors**: Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18892)  

**Abstract**: DeepSeek-R1 has shown that long chain-of-thought (CoT) reasoning can naturally emerge through a simple reinforcement learning (RL) framework with rule-based rewards, where the training may directly start from the base models-a paradigm referred to as zero RL training. Most recent efforts to reproduce zero RL training have primarily focused on the Qwen2.5 model series, which may not be representative as we find the base models already exhibit strong instruction-following and self-reflection abilities. In this work, we investigate zero RL training across 10 diverse base models, spanning different families and sizes including LLama3-8B, Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5-math-7B, and all Qwen2.5 models from 0.5B to 32B. Leveraging several key design strategies-such as adjusting format reward and controlling query difficulty-we achieve substantial improvements in both reasoning accuracy and response length across most settings. However, by carefully monitoring the training dynamics, we observe that different base models exhibit distinct patterns during training. For instance, the increased response length does not always correlate with the emergence of certain cognitive behaviors such as verification (i.e., the "aha moment"). Notably, we observe the "aha moment" for the first time in small models not from the Qwen family. We share the key designs that enable successful zero RL training, along with our findings and practices. To facilitate further research, we open-source the code, models, and analysis tools. 

---
# Toward building next-generation Geocoding systems: a systematic review 

**Authors**: Zhengcong Yin, Daniel W. Goldberg, Binbin Lin, Bing Zhou, Diya Li, Andong Ma, Ziqian Ming, Heng Cai, Zhe Zhang, Shaohua Wang, Shanzhen Gao, Joey Ying Lee, Xiao Li, Da Huo  

**Link**: [PDF](https://arxiv.org/pdf/2503.18888)  

**Abstract**: Geocoding systems are widely used in both scientific research for spatial analysis and everyday life through location-based services. The quality of geocoded data significantly impacts subsequent processes and applications, underscoring the need for next-generation systems. In response to this demand, this review first examines the evolving requirements for geocoding inputs and outputs across various scenarios these systems must address. It then provides a detailed analysis of how to construct such systems by breaking them down into key functional components and reviewing a broad spectrum of existing approaches, from traditional rule-based methods to advanced techniques in information retrieval, natural language processing, and large language models. Finally, we identify opportunities to improve next-generation geocoding systems in light of recent technological advances. 

---
# Reasoning to Learn from Latent Thoughts 

**Authors**: Yangjun Ruan, Neil Band, Chris J. Maddison, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2503.18866)  

**Abstract**: Compute scaling for language model (LM) pretraining has outpaced the growth of human-written texts, leading to concerns that data will become the bottleneck to LM scaling. To continue scaling pretraining in this data-constrained regime, we propose that explicitly modeling and inferring the latent thoughts that underlie the text generation process can significantly improve pretraining data efficiency. Intuitively, our approach views web text as the compressed final outcome of a verbose human thought process and that the latent thoughts contain important contextual knowledge and reasoning steps that are critical to data-efficient learning. We empirically demonstrate the effectiveness of our approach through data-constrained continued pretraining for math. We first show that synthetic data approaches to inferring latent thoughts significantly improve data efficiency, outperforming training on the same amount of raw data (5.7\% $\rightarrow$ 25.4\% on MATH). Furthermore, we demonstrate latent thought inference without a strong teacher, where an LM bootstraps its own performance by using an EM algorithm to iteratively improve the capability of the trained LM and the quality of thought-augmented pretraining data. We show that a 1B LM can bootstrap its performance across at least three iterations and significantly outperform baselines trained on raw data, with increasing gains from additional inference compute when performing the E-step. The gains from inference scaling and EM iterations suggest new opportunities for scaling data-constrained pretraining. 

---
# EconEvals: Benchmarks and Litmus Tests for LLM Agents in Unknown Environments 

**Authors**: Sara Fish, Julia Shephard, Minkai Li, Ran I. Shorrer, Yannai A. Gonczarowski  

**Link**: [PDF](https://arxiv.org/pdf/2503.18825)  

**Abstract**: We develop benchmarks for LLM agents that act in, learn from, and strategize in unknown environments, the specifications of which the LLM agent must learn over time from deliberate exploration. Our benchmarks consist of decision-making tasks derived from key problems in economics. To forestall saturation, the benchmark tasks are synthetically generated with scalable difficulty levels. Additionally, we propose litmus tests, a new kind of quantitative measure for LLMs and LLM agents. Unlike benchmarks, litmus tests quantify differences in character, values, and tendencies of LLMs and LLM agents, by considering their behavior when faced with tradeoffs (e.g., efficiency versus equality) where there is no objectively right or wrong behavior. Overall, our benchmarks and litmus tests assess the abilities and tendencies of LLM agents in tackling complex economic problems in diverse settings spanning procurement, scheduling, task allocation, and pricing -- applications that should grow in importance as such agents are further integrated into the economy. 

---
# REALM: A Dataset of Real-World LLM Use Cases 

**Authors**: Jingwen Cheng, Kshitish Ghate, Wenyue Hua, William Yang Wang, Hong Shen, Fei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18792)  

**Abstract**: Large Language Models, such as the GPT series, have driven significant industrial applications, leading to economic and societal transformations. However, a comprehensive understanding of their real-world applications remains limited. To address this, we introduce REALM, a dataset of over 94,000 LLM use cases collected from Reddit and news articles. REALM captures two key dimensions: the diverse applications of LLMs and the demographics of their users. It categorizes LLM applications and explores how users' occupations relate to the types of applications they use. By integrating real-world data, REALM offers insights into LLM adoption across different domains, providing a foundation for future research on their evolving societal roles. A dedicated dashboard this https URL presents the data. 

---
# BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache 

**Authors**: Dayou Du, Shijie Cao, Jianyi Cheng, Ting Cao, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18773)  

**Abstract**: The growing adoption of long-context Large Language Models (LLMs) has introduced significant memory and computational challenges in autoregressive decoding due to the expanding Key-Value (KV) cache. KV cache quantization has emerged as a promising solution, with prior work showing that 4-bit or even 2-bit quantization can maintain model accuracy while reducing memory costs. However, despite these benefits, preliminary implementations for the low-bit KV cache struggle to deliver the expected speedup due to quantization and dequantization overheads and the lack of Tensor Cores utilization. In this work, we propose BitDecoding, a GPU-optimized framework that unlocks Tensor Cores for efficient decoding with low-bit KV cache. Efficiently leveraging Tensor Cores for low-bit KV cache is challenging due to the dynamic nature of KV cache generation at each decoding step. BitDecoding addresses these challenges with a Tensor Cores-Centric BitFusion Scheme that ensures data layout compatibility to enable high utilization of Tensor Cores. Additionally, BitDecoding incorporates a warp-efficient parallel decoding kernel and a fine-grained asynchronous pipeline, minimizing dequantization overhead and improving computational efficiency. Experiments show that BitDecoding achieves up to 7.5x speedup on RTX 4090, 4.8x on A100, and 8.9x on H100, compared to FP16 FlashDecoding-v2. It also outperforms the state-of-the-art low-bit KV cache implementation (QServe) by up to 4.3x. On LLaMA-3.1-8B with a 128K sequence length, BitDecoding reduces single-batch decoding latency by 3x, demonstrating its effectiveness in long-context generation scenarios. The code is available at this https URL. 

---
# ArchSeek: Retrieving Architectural Case Studies Using Vision-Language Models 

**Authors**: Danrui Li, Yichao Shi, Yaluo Wang, Ziying Shi, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2503.18680)  

**Abstract**: Efficiently searching for relevant case studies is critical in architectural design, as designers rely on precedent examples to guide or inspire their ongoing projects. However, traditional text-based search tools struggle to capture the inherently visual and complex nature of architectural knowledge, often leading to time-consuming and imprecise exploration. This paper introduces ArchSeek, an innovative case study search system with recommendation capability, tailored for architecture design professionals. Powered by the visual understanding capabilities from vision-language models and cross-modal embeddings, it enables text and image queries with fine-grained control, and interaction-based design case recommendations. It offers architects a more efficient, personalized way to discover design inspirations, with potential applications across other visually driven design fields. The source code is available at this https URL. 

---
# AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents 

**Authors**: Haoyu Wang, Christopher M. Poskitt, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.18666)  

**Abstract**: Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identifying 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios. 

---
# Dense Retrieval for Low Resource Languages -- the Case of Amharic Language 

**Authors**: Tilahun Yeshambel, Moncef Garouani, Serge Molina, Josiane Mothe  

**Link**: [PDF](https://arxiv.org/pdf/2503.18570)  

**Abstract**: This paper reports some difficulties and some results when using dense retrievers on Amharic, one of the low-resource languages spoken by 120 millions populations. The efforts put and difficulties faced by University Addis Ababa toward Amharic Information Retrieval will be developed during the presentation. 

---
# Distil-xLSTM: Learning Attention Mechanisms through Recurrent Structures 

**Authors**: Abdoul Majid O. Thiombiano, Brahim Hnich, Ali Ben Mrad, Mohamed Wiem Mkaouer  

**Link**: [PDF](https://arxiv.org/pdf/2503.18565)  

**Abstract**: The current era of Natural Language Processing (NLP) is dominated by Transformer models. However, novel architectures relying on recurrent mechanisms, such as xLSTM and Mamba, have been proposed as alternatives to attention-based models. Although computation is done differently than with the attention mechanism mechanism, these recurrent models yield good results and sometimes even outperform state-of-the-art attention-based models. In this work, we propose Distil-xLSTM, an xLSTM-based Small Language Model (SLM) trained by distilling knowledge from a Large Language Model (LLM) that shows promising results while being compute and scale efficient. Our Distil-xLSTM focuses on approximating a transformer-based model attention parametrization using its recurrent sequence mixing components and shows good results with minimal training. 

---
# Instruction-Aligned Visual Attention for Mitigating Hallucinations in Large Vision-Language Models 

**Authors**: Bin Li, Dehong Gao, Yeyuan Wang, Linbo Jin, Shanqing Yu, Xiaoyan Cai, Libin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18556)  

**Abstract**: Despite the significant success of Large Vision-Language models(LVLMs), these models still suffer hallucinations when describing images, generating answers that include non-existent objects. It is reported that these models tend to over-focus on certain irrelevant image tokens that do not contain critical information for answering the question and distort the output. To address this, we propose an Instruction-Aligned Visual Attention(IAVA) approach, which identifies irrelevant tokens by comparing changes in attention weights under two different instructions. By applying contrastive decoding, we dynamically adjust the logits generated from original image tokens and irrelevant image tokens, reducing the model's over-attention to irrelevant information. The experimental results demonstrate that IAVA consistently outperforms existing decoding techniques on benchmarks such as MME, POPE, and TextVQA in mitigating object hallucinations. Our IAVA approach is available online at this https URL. 

---
# Verbal Process Supervision Elicits Better Coding Agents 

**Authors**: Hao-Yuan Chen, Cheng-Pong Huang, Jui-Ming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18494)  

**Abstract**: The emergence of large language models and their applications as AI agents have significantly advanced state-of-the-art code generation benchmarks, transforming modern software engineering tasks. However, even with test-time computed reasoning models, these systems still struggle with complex software engineering challenges. This work introduces CURA, a code understanding and reasoning agent system enhanced with verbal process supervision (VPS), achieving a 3.65\% improvement over baseline models on challenging benchmarks like BigCodeBench. Furthermore, CURA, when paired with the o3-mini model and VPS techniques, attains state-of-the-art performance. This work represents a step forward in integrating reasoning-driven architectures with LLM-based code generation, enabling agentic reasoning for language models to solve complex software engineering tasks. 

---
# Safeguarding Mobile GUI Agent via Logic-based Action Verification 

**Authors**: Jungjae Lee, Dongjae Lee, Chihun Choi, Youngmin Im, Jaeyoung Wi, Kihong Heo, Sangeun Oh, Sunjae Lee, Insik Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18492)  

**Abstract**: Large Foundation Models (LFMs) have unlocked new possibilities in human-computer interaction, particularly with the rise of mobile Graphical User Interface (GUI) Agents capable of interpreting GUIs. These agents promise to revolutionize mobile computing by allowing users to automate complex mobile tasks through simple natural language instructions. However, the inherent probabilistic nature of LFMs, coupled with the ambiguity and context-dependence of mobile tasks, makes LFM-based automation unreliable and prone to errors. To address this critical challenge, we introduce VeriSafe Agent (VSA): a formal verification system that serves as a logically grounded safeguard for Mobile GUI Agents. VSA is designed to deterministically ensure that an agent's actions strictly align with user intent before conducting an action. At its core, VSA introduces a novel autoformalization technique that translates natural language user instructions into a formally verifiable specification, expressed in our domain-specific language (DSL). This enables runtime, rule-based verification, allowing VSA to detect and prevent erroneous actions executing an action, either by providing corrective feedback or halting unsafe behavior. To the best of our knowledge, VSA is the first attempt to bring the rigor of formal verification to GUI agent. effectively bridging the gap between LFM-driven automation and formal software verification. We implement VSA using off-the-shelf LLM services (GPT-4o) and evaluate its performance on 300 user instructions across 18 widely used mobile apps. The results demonstrate that VSA achieves 94.3%-98.33% accuracy in verifying agent actions, representing a significant 20.4%-25.6% improvement over existing LLM-based verification methods, and consequently increases the GUI agent's task completion rate by 90%-130%. 

---
# PM4Bench: A Parallel Multilingual Multi-Modal Multi-task Benchmark for Large Vision Language Model 

**Authors**: Junyuan Gao, Jiahe Song, Jiang Wu, Runchuan Zhu, Guanlin Shen, Shasha Wang, Xingjian Wei, Haote Yang, Songyang Zhang, Weijia Li, Bin Wang, Dahua Lin, Lijun Wu, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18484)  

**Abstract**: Existing multilingual benchmarks for Large Vision Language Models (LVLMs) suffer from limitations including language-specific content biases, disjointed multimodal input formats, and a lack of safety evaluation. To address these gaps, we propose PM4Bench, the first Parallel Multilingual Multi-Modal Multi-task Benchmark for LVLMs. PM4Bench features a parallel corpus design across 10 languages, enabling fair and accurate cross-lingual comparisons. It includes the vision setting where text and queries are embedded in images, requiring LVLMs to simultaneously "see", "read", and "think", aligning with real-world applications. Additionally, PM\textsuperscript{4}Bench incorporates safety evaluations, addressing critical oversight in existing multilingual benchmarks. Using PM4Bench, we evaluate 11 mainstream LVLMs, revealing significant cross-linguistic performance disparities, particularly in vision settings, and identifying OCR capability as a key determinant of these imbalances. We will release PM4Bench at this https URL . 

---
# Global-Local Tree Search for Language Guided 3D Scene Generation 

**Authors**: Wei Deng, Mengshi Qi, Huadong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.18476)  

**Abstract**: Large Vision-Language Models (VLMs), such as GPT-4, have achieved remarkable success across various fields. However, there are few studies on 3D indoor scene generation with VLMs. This paper considers this task as a planning problem subject to spatial and layout common sense constraints. To solve the problem with a VLM, we propose a new global-local tree search algorithm. Globally, the method places each object sequentially and explores multiple placements during each placement process, where the problem space is represented as a tree. To reduce the depth of the tree, we decompose the scene structure hierarchically, i.e. room level, region level, floor object level, and supported object level. The algorithm independently generates the floor objects in different regions and supported objects placed on different floor objects. Locally, we also decompose the sub-task, the placement of each object, into multiple steps. The algorithm searches the tree of problem space. To leverage the VLM model to produce positions of objects, we discretize the top-down view space as a dense grid and fill each cell with diverse emojis to make to cells distinct. We prompt the VLM with the emoji grid and the VLM produces a reasonable location for the object by describing the position with the name of emojis. The quantitative and qualitative experimental results illustrate our approach generates more plausible 3D scenes than state-of-the-art approaches. Our source code is available at this https URL . 

---
# StableGS: A Floater-Free Framework for 3D Gaussian Splatting 

**Authors**: Luchao Wang, Qian Ren, Kaiming He, Hua Wang, Zhi Chen, Yaohua Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18458)  

**Abstract**: Recent years have witnessed remarkable success of 3D Gaussian Splatting (3DGS) in novel view synthesis, surpassing prior differentiable rendering methods in both quality and efficiency. However, its training process suffers from coupled opacity-color optimization that frequently converges to local minima, producing floater artifacts that degrade visual fidelity. We present StableGS, a framework that eliminates floaters through cross-view depth consistency constraints while introducing a dual-opacity GS model to decouple geometry and material properties of translucent objects. To further enhance reconstruction quality in weakly-textured regions, we integrate DUSt3R depth estimation, significantly improving geometric stability. Our method fundamentally addresses 3DGS training instabilities, outperforming existing state-of-the-art methods across open-source datasets. 

---
# On the Perception Bottleneck of VLMs for Chart Understanding 

**Authors**: Junteng Liu, Weihao Zeng, Xiwen Zhang, Yijun Wang, Zifei Shan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18435)  

**Abstract**: Chart understanding requires models to effectively analyze and reason about numerical data, textual elements, and complex visual components. Our observations reveal that the perception capabilities of existing large vision-language models (LVLMs) constitute a critical bottleneck in this process. In this study, we delve into this perception bottleneck by decomposing it into two components: the vision encoder bottleneck, where the visual representation may fail to encapsulate the correct information, and the extraction bottleneck, where the language model struggles to extract the necessary information from the provided visual representations. Through comprehensive experiments, we find that (1) the information embedded within visual representations is substantially richer than what is typically captured by linear extractors, such as the widely used retrieval accuracy metric; (2) While instruction tuning effectively enhances the extraction capability of LVLMs, the vision encoder remains a critical bottleneck, demanding focused attention and improvement. Therefore, we further enhance the visual encoder to mitigate the vision encoder bottleneck under a contrastive learning framework. Empirical results demonstrate that our approach significantly mitigates the perception bottleneck and improves the ability of LVLMs to comprehend charts. Code is publicly available at this https URL. 

---
# Solving Situation Puzzles with Large Language Model and External Reformulation 

**Authors**: Kun Li, Xinwei Chen, Tianyou Song, Chengrui Zhou, Zhuoran Liu, Zhenyan Zhang, Jiangjian Guo, Qing Shan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18394)  

**Abstract**: In recent years, large language models (LLMs) have shown an impressive ability to perform arithmetic and symbolic reasoning tasks. However, we found that LLMs (e.g., ChatGPT) cannot perform well on reasoning that requires multiple rounds of dialogue, especially when solving situation puzzles. Specifically, LLMs intend to ask very detailed questions focusing on a specific aspect or same/similar questions after several rounds of Q&As. To help LLMs get out of the above dilemma, we propose a novel external reformulation methodology, where the situation puzzle will be reformulated after several rounds of Q&A or when the LLMs raise an incorrect guess. Experiments show superior performance (e.g., win rate, number of question/guess attempts) of our method than directly using LLMs for solving situation puzzles, highlighting the potential of strategic problem reformulation to enhance the reasoning capabilities of LLMs in complex interactive scenarios. 

---
# Bridging Writing Manner Gap in Visual Instruction Tuning by Creating LLM-aligned Instructions 

**Authors**: Dong Jing, Nanyi Fei, Zhiwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18320)  

**Abstract**: In the realm of Large Multi-modal Models (LMMs), the instruction quality during the visual instruction tuning stage significantly influences the performance of modality alignment. In this paper, we assess the instruction quality from a unique perspective termed \textbf{Writing Manner}, which encompasses the selection of vocabulary, grammar and sentence structure to convey specific semantics. We argue that there exists a substantial writing manner gap between the visual instructions and the base Large Language Models (LLMs) within LMMs. This gap forces the pre-trained base LLMs to deviate from their original writing styles, leading to capability degradation of both base LLMs and LMMs. To bridge the writing manner gap while preserving the original semantics, we propose directly leveraging the base LLM to align the writing manner of soft-format visual instructions with that of the base LLM itself, resulting in novel LLM-aligned instructions. The manual writing manner evaluation results demonstrate that our approach successfully minimizes the writing manner gap. By utilizing LLM-aligned instructions, the baseline models LLaVA-7B and QwenVL demonstrate enhanced resistance to hallucinations and non-trivial comprehensive improvements across all $15$ visual and language benchmarks. 

---
# Decoupling Angles and Strength in Low-rank Adaptation 

**Authors**: Massimo Bini, Leander Girrbach, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2503.18225)  

**Abstract**: Parameter-Efficient FineTuning (PEFT) methods have recently gained significant popularity thanks to the widespread availability of large-scale pretrained models. These methods allow for quick adaptation to downstream tasks with minimal computational cost. However, popular finetuning methods such as LoRA exhibit limited robustness when it comes to hyperparameter choices or extended training regimes, preventing optimal out-of-the-box performance. In contrast, bounded approaches, such as ETHER, provide greater robustness but are limited to extremely low-rank adaptations and fixed-strength transformations, reducing their adaptation expressive power. In this work, we propose Decoupled Low-rank Adaptation (DeLoRA), a novel finetuning method that normalizes and scales learnable low-rank matrices. By bounding the distance of the transformation, DeLoRA effectively decouples the angular learning from the adaptation strength, enhancing robustness without compromising performance. Through evaluations on subject-driven image generation, natural language understanding, and instruction tuning, we show that DeLoRA matches or surpasses performance of competing PEFT methods, while exhibiting stronger robustness. Code is available at this https URL. 

---
# AgentRxiv: Towards Collaborative Autonomous Research 

**Authors**: Samuel Schmidgall, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2503.18102)  

**Abstract**: Progress in scientific discovery is rarely the result of a single "Eureka" moment, but is rather the product of hundreds of scientists incrementally working together toward a common goal. While existing agent workflows are capable of producing research autonomously, they do so in isolation, without the ability to continuously improve upon prior research results. To address these challenges, we introduce AgentRxiv-a framework that lets LLM agent laboratories upload and retrieve reports from a shared preprint server in order to collaborate, share insights, and iteratively build on each other's research. We task agent laboratories to develop new reasoning and prompting techniques and find that agents with access to their prior research achieve higher performance improvements compared to agents operating in isolation (11.4% relative improvement over baseline on MATH-500). We find that the best performing strategy generalizes to benchmarks in other domains (improving on average by 3.3%). Multiple agent laboratories sharing research through AgentRxiv are able to work together towards a common goal, progressing more rapidly than isolated laboratories, achieving higher overall accuracy (13.7% relative improvement over baseline on MATH-500). These findings suggest that autonomous agents may play a role in designing future AI systems alongside humans. We hope that AgentRxiv allows agents to collaborate toward research goals and enables researchers to accelerate discovery. 

---
# Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation 

**Authors**: Ziming Wei, Bingqian Lin, Yunshuang Nie, Jiaqi Chen, Shikui Ma, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18065)  

**Abstract**: Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method. Code is available at this https URL. 

---
# (G)I-DLE: Generative Inference via Distribution-preserving Logit Exclusion with KL Divergence Minimization for Constrained Decoding 

**Authors**: Hanwool Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.18050)  

**Abstract**: We propose (G)I-DLE, a new approach to constrained decoding that leverages KL divergence minimization to preserve the intrinsic conditional probability distribution of autoregressive language models while excluding undesirable tokens. Unlike conventional methods that naively set banned tokens' logits to $-\infty$, which can distort the conversion from raw logits to posterior probabilities and increase output variance, (G)I-DLE re-normalizes the allowed token probabilities to minimize such distortion. We validate our method on the K2-Eval dataset, specifically designed to assess Korean language fluency, logical reasoning, and cultural appropriateness. Experimental results on Qwen2.5 models (ranging from 1.5B to 14B) demonstrate that G-IDLE not only boosts mean evaluation scores but also substantially reduces the variance of output quality. 

---
# Expanding the Boundaries of Vision Prior Knowledge in Multi-modal Large Language Models 

**Authors**: Qiao Liang, Yanjiang Liu, Ben He, Yaojie Lu, Hongyu Lin, Jia Zheng, Xianpei Han, Le Sun, Yingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.18034)  

**Abstract**: Does the prior knowledge of the vision encoder constrain the capability boundary of Multi-modal Large Language Models (MLLMs)? While most existing research treats MLLMs as unified systems optimized through end-to-end training, the impact of vision encoder's prior knowledge is seldom investigated. In this work, we introduce a novel metric, $Rank_e$, to quantify the effect of the vision encoder's prior knowledge on MLLM performance. Our analysis reveals a positive correlation between prior knowledge and MLLM performance. Moreover, we find that domain-specific fine-tuning using solely end-to-end visual question answering (VQA) data is insufficient--particularly for entities with low inherent visual prior knowledge. To address this issue, we propose VisPRE (Vision Prior Remediation), a two-stage training framework that explicitly incorporates prior knowledge at the vision encoder level. Experimental results demonstrate that augmenting vision encoder's prior knowledge substantially boosts the visual understanding capabilities of MLLMs, offering a novel and effective strategy for improving performance, especially in scenarios involving uncommon visual entities. 

---
# Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities 

**Authors**: Weixiang Zhao, Xingyu Sui, Jiahe Guo, Yulin Hu, Yang Deng, Yanyan Zhao, Bing Qin, Wanxiang Che, Tat-Seng Chua, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17979)  

**Abstract**: Recent advancements in Large Reasoning Models (LRMs), such as OpenAI's o1/o3 and DeepSeek-R1, have demonstrated remarkable performance in specialized reasoning tasks through human-like deliberative thinking and long chain-of-thought reasoning. However, our systematic evaluation across various model families (DeepSeek, Qwen, and LLaMA) and scales (7B to 671B) reveals that acquiring these deliberative reasoning capabilities significantly reduces the foundational capabilities of LRMs, including notable declines in helpfulness and harmlessness, alongside substantially increased inference costs. Importantly, we demonstrate that adaptive reasoning -- employing modes like Zero-Thinking, Less-Thinking, and Summary-Thinking -- can effectively alleviate these drawbacks. Our empirical insights underline the critical need for developing more versatile LRMs capable of dynamically allocating inference-time compute according to specific task characteristics. 

---
# Human-AI Interaction and User Satisfaction: Empirical Evidence from Online Reviews of AI Products 

**Authors**: Stefan Pasch, Sun-Young Ha  

**Link**: [PDF](https://arxiv.org/pdf/2503.17955)  

**Abstract**: Human-AI Interaction (HAI) guidelines and design principles have become increasingly important in both industry and academia to guide the development of AI systems that align with user needs and expectations. However, large-scale empirical evidence on how HAI principles shape user satisfaction in practice remains limited. This study addresses that gap by analyzing over 100,000 user reviews of AI-related products from this http URL, a leading review platform for business software and services. Based on widely adopted industry guidelines, we identify seven core HAI dimensions and examine their coverage and sentiment within the reviews. We find that the sentiment on four HAI dimensions-adaptability, customization, error recovery, and security-is positively associated with overall user satisfaction. Moreover, we show that engagement with HAI dimensions varies by professional background: Users with technical job roles are more likely to discuss system-focused aspects, such as reliability, while non-technical users emphasize interaction-focused features like customization and feedback. Interestingly, the relationship between HAI sentiment and overall satisfaction is not moderated by job role, suggesting that once an HAI dimension has been identified by users, its effect on satisfaction is consistent across job roles. 

---
# Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization 

**Authors**: Zefeng Zhang, Hengzhu Tang, Jiawei Sheng, Zhenyu Zhang, Yiming Ren, Zhenyang Li, Dawei Yin, Duohe Ma, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17928)  

**Abstract**: Multimodal Large Language Models excel in various tasks, yet often struggle with modality bias, where the model tends to rely heavily on a single modality and overlook critical information in other modalities, which leads to incorrect focus and generating irrelevant responses. In this paper, we propose using the paradigm of preference optimization to solve the modality bias problem, including RLAIFVBias, a debiased preference optimization dataset, and a Noise Aware Preference Optimization algorithm. Specifically, we first construct the dataset by introducing perturbations to reduce the informational content of certain modalities, compelling the model to rely on a specific modality when generating negative responses. To address the inevitable noise in automatically constructed data, we combine the noise robust Mean Absolute Error with the Binary Cross Entropy in Direct Preference Optimization by a negative Box Cox transformation, and dynamically adjust the algorithm noise robustness based on the evaluated noise levels in the data. Extensive experiments validate our approach, demonstrating not only its effectiveness in mitigating modality bias but also its significant role in minimizing hallucinations. 

---
# Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality Data for Efficient and Accurate Code LLM 

**Authors**: Codefuse, Ling Team, Wenting Cai, Yuchen Cao, Chaoyu Chen, Chen Chen, Siba Chen, Qing Cui, Peng Di, Junpeng Fang, Zi Gong, Ting Guo, Zhengyu He, Yang Huang, Cong Li, Jianguo Li, Zheng Li, Shijie Lian, BingChang Liu, Songshan Luo, Shuo Mao, Min Shen, Jian Wu, Jiaolong Yang, Wenjie Yang, Tong Ye, Hang Yu, Wei Zhang, Zhenduo Zhang, Hailin Zhao, Xunjin Zheng, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.17793)  

**Abstract**: Recent advancements in code large language models (LLMs) have demonstrated remarkable capabilities in code generation and understanding. It is still challenging to build a code LLM with comprehensive performance yet ultimate efficiency. Many attempts have been released in the open source community to break the trade-off between performance and efficiency, such as the Qwen Coder series and the DeepSeek Coder series. This paper introduces yet another attempt in this area, namely Ling-Coder-Lite. We leverage the efficient Mixture-of-Experts (MoE) architecture along with a set of high-quality data curation methods (especially those based on program analytics) to build an efficient yet powerful code LLM. Ling-Coder-Lite exhibits on-par performance on 12 representative coding benchmarks compared to state-of-the-art models of similar size, such as Qwen2.5-Coder-7B and DeepSeek-Coder-V2-Lite, while offering competitive latency and throughput. In practice, we achieve a 50\% reduction in deployment resources compared to the similar-sized dense model without performance loss. To facilitate further research and development in this area, we open-source our models as well as a substantial portion of high-quality data for the annealing and post-training stages. The models and data can be accessed at~\url{this https URL}. 

---
# Energy-Aware LLMs: A step towards sustainable AI for downstream applications 

**Authors**: Nguyen Phuc Tran, Brigitte Jaumard, Oscar Delgado  

**Link**: [PDF](https://arxiv.org/pdf/2503.17783)  

**Abstract**: Advanced Large Language Models (LLMs) have revolutionized various fields, including communication networks, sparking an innovation wave that has led to new applications and services, and significantly enhanced solution schemes. Despite all these impressive developments, most LLMs typically require huge computational resources, resulting in terribly high energy consumption. Thus, this research study proposes an end-to-end pipeline that investigates the trade-off between energy efficiency and model performance for an LLM during fault ticket analysis in communication networks. It further evaluates the pipeline performance using two real-world datasets for the tasks of root cause analysis and response feedback in a communication network. Our results show that an appropriate combination of quantization and pruning techniques is able to reduce energy consumption while significantly improving model performance. 

---
# V2P-Bench: Evaluating Video-Language Understanding with Visual Prompts for Better Human-Model Interaction 

**Authors**: Yiming Zhao, Yu Zeng, Yukun Qi, YaoYang Liu, Lin Chen, Zehui Chen, Xikun Bao, Jie Zhao, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.17736)  

**Abstract**: Large Vision-Language Models (LVLMs) have made significant progress in the field of video understanding recently. However, current benchmarks uniformly lean on text prompts for evaluation, which often necessitate complex referential language and fail to provide precise spatial and temporal references. This limitation diminishes the experience and efficiency of human-model interaction. To address this limitation, we propose the Video Visual Prompt Benchmark(V2P-Bench), a comprehensive benchmark specifically designed to evaluate LVLMs' video understanding capabilities in multimodal human-model interaction scenarios. V2P-Bench includes 980 unique videos and 1,172 QA pairs, covering 5 main tasks and 12 dimensions, facilitating instance-level fine-grained understanding aligned with human cognition. Benchmarking results reveal that even the most powerful models perform poorly on V2P-Bench (65.4% for GPT-4o and 67.9% for Gemini-1.5-Pro), significantly lower than the human experts' 88.3%, highlighting the current shortcomings of LVLMs in understanding video visual prompts. We hope V2P-Bench will serve as a foundation for advancing multimodal human-model interaction and video understanding evaluation. Project page: this https URL. 

---
# FairFlow: Mitigating Dataset Biases through Undecided Learning 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2503.17632)  

**Abstract**: Language models are prone to dataset biases, known as shortcuts and spurious correlations in data, which often result in performance drop on new data. We present a new debiasing framework called ``FairFlow'' that mitigates dataset biases by learning to be undecided in its predictions for data samples or representations associated with known or unknown biases. The framework introduces two key components: a suite of data and model perturbation operations that generate different biased views of input samples, and a contrastive objective that learns debiased and robust representations from the resulting biased views of samples. Experiments show that FairFlow outperforms existing debiasing methods, particularly against out-of-domain and hard test samples without compromising the in-domain performance 

---
# Autonomous Radiotherapy Treatment Planning Using DOLA: A Privacy-Preserving, LLM-Based Optimization Agent 

**Authors**: Humza Nusrat, Bing Luo, Ryan Hall, Joshua Kim, Hassan Bagher-Ebadian, Anthony Doemer, Benjamin Movsas, Kundan Thind  

**Link**: [PDF](https://arxiv.org/pdf/2503.17553)  

**Abstract**: Radiotherapy treatment planning is a complex and time-intensive process, often impacted by inter-planner variability and subjective decision-making. To address these challenges, we introduce Dose Optimization Language Agent (DOLA), an autonomous large language model (LLM)-based agent designed for optimizing radiotherapy treatment plans while rigorously protecting patient privacy. DOLA integrates the LLaMa3.1 LLM directly with a commercial treatment planning system, utilizing chain-of-thought prompting, retrieval-augmented generation (RAG), and reinforcement learning (RL). Operating entirely within secure local infrastructure, this agent eliminates external data sharing. We evaluated DOLA using a retrospective cohort of 18 prostate cancer patients prescribed 60 Gy in 20 fractions, comparing model sizes (8 billion vs. 70 billion parameters) and optimization strategies (No-RAG, RAG, and RAG+RL) over 10 planning iterations. The 70B model demonstrated significantly improved performance, achieving approximately 16.4% higher final scores than the 8B model. The RAG approach outperformed the No-RAG baseline by 19.8%, and incorporating RL accelerated convergence, highlighting the synergy of retrieval-based memory and reinforcement learning. Optimal temperature hyperparameter analysis identified 0.4 as providing the best balance between exploration and exploitation. This proof of concept study represents the first successful deployment of locally hosted LLM agents for autonomous optimization of treatment plans within a commercial radiotherapy planning system. By extending human-machine interaction through interpretable natural language reasoning, DOLA offers a scalable and privacy-conscious framework, with significant potential for clinical implementation and workflow improvement. 

---
# Large Language Models (LLMs) for Source Code Analysis: applications, models and datasets 

**Authors**: Hamed Jelodar, Mohammad Meymani, Roozbeh Razavi-Far  

**Link**: [PDF](https://arxiv.org/pdf/2503.17502)  

**Abstract**: Large language models (LLMs) and transformer-based architectures are increasingly utilized for source code analysis. As software systems grow in complexity, integrating LLMs into code analysis workflows becomes essential for enhancing efficiency, accuracy, and automation. This paper explores the role of LLMs for different code analysis tasks, focusing on three key aspects: 1) what they can analyze and their applications, 2) what models are used and 3) what datasets are used, and the challenges they face. Regarding the goal of this research, we investigate scholarly articles that explore the use of LLMs for source code analysis to uncover research developments, current trends, and the intellectual structure of this emerging field. Additionally, we summarize limitations and highlight essential tools, datasets, and key challenges, which could be valuable for future work. 

---
# Variance Control via Weight Rescaling in LLM Pre-training 

**Authors**: Louis Owen, Abhay Kumar, Nilabhra Roy Chowdhury, Fabian Güra  

**Link**: [PDF](https://arxiv.org/pdf/2503.17500)  

**Abstract**: The outcome of Large Language Model (LLM) pre-training strongly depends on weight initialization and variance control strategies. Although the importance of initial variance control has been well documented in neural networks in general, the literature on initialization and management of its growth during LLM pre-training, specifically, is somewhat sparse. In this paper, we introduce the Layer Index Rescaling (LIR) weight initialization scheme, and the Target Variance Rescaling (TVR) variance control strategy. Experiments on a 1B parameter LLaMA model demonstrate that better variance management using these techniques yields substantial improvements in downstream task performance (up to 4.6% on common pre-training benchmarks) and reduces extreme activation values, thus mitigating challenges associated with quantization and low-precision training. Our code is available at: this https URL. 

---
# From Text to Talent: A Pipeline for Extracting Insights from Candidate Profiles 

**Authors**: Paolo Frazzetto, Muhammad Uzair Ul Haq, Flavia Fabris, Alessandro Sperduti  

**Link**: [PDF](https://arxiv.org/pdf/2503.17438)  

**Abstract**: The recruitment process is undergoing a significant transformation with the increasing use of machine learning and natural language processing techniques. While previous studies have focused on automating candidate selection, the role of multiple vacancies in this process remains understudied. This paper addresses this gap by proposing a novel pipeline that leverages Large Language Models and graph similarity measures to suggest ideal candidates for specific job openings. Our approach represents candidate profiles as multimodal embeddings, enabling the capture of nuanced relationships between job requirements and candidate attributes. The proposed approach has significant implications for the recruitment industry, enabling companies to streamline their hiring processes and identify top talent more efficiently. Our work contributes to the growing body of research on the application of machine learning in human resources, highlighting the potential of LLMs and graph-based methods in revolutionizing the recruitment landscape. 

---
# Understanding Social Support Needs in Questions: A Hybrid Approach Integrating Semi-Supervised Learning and LLM-based Data Augmentation 

**Authors**: Junwei Kuang, Liang Yang, Shaoze Cui, Weiguo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17421)  

**Abstract**: Patients are increasingly turning to online health Q&A communities for social support to improve their well-being. However, when this support received does not align with their specific needs, it may prove ineffective or even detrimental. This necessitates a model capable of identifying the social support needs in questions. However, training such a model is challenging due to the scarcity and class imbalance issues of labeled data. To overcome these challenges, we follow the computational design science paradigm to develop a novel framework, Hybrid Approach for SOcial Support need classification (HA-SOS). HA-SOS integrates an answer-enhanced semi-supervised learning approach, a text data augmentation technique leveraging large language models (LLMs) with reliability- and diversity-aware sample selection mechanism, and a unified training process to automatically label social support needs in questions. Extensive empirical evaluations demonstrate that HA-SOS significantly outperforms existing question classification models and alternative semi-supervised learning approaches. This research contributes to the literature on social support, question classification, semi-supervised learning, and text data augmentation. In practice, our HA-SOS framework facilitates online Q&A platform managers and answerers to better understand users' social support needs, enabling them to provide timely, personalized answers and interventions. 

---
# State Fourier Diffusion Language Model (SFDLM): A Scalable, Novel Iterative Approach to Language Modeling 

**Authors**: Andrew Kiruluta, Andreas Lemos  

**Link**: [PDF](https://arxiv.org/pdf/2503.17382)  

**Abstract**: In recent years, diffusion based methods have emerged as a powerful paradigm for generative modeling. Although discrete diffusion for natural language processing has been explored to a lesser extent, it shows promise for tasks requiring iterative denoising of token based data. In standard approaches to text generation, transformers dominate, but their reliance on self attention often incurs high computational costs. This paper introduces a fully diffusion driven discrete text generation model built without any transformer or large convolution modules. Instead, the model integrates structured state space dynamics in the time domain with a novel Complex Fourier Multi Layer Perceptron module that operates in the frequency domain. The forward noising process randomly samples the vocabulary to replace tokens with a controlled probability, while the learned reverse model systematically reverts corrupted sequences toward their original states. By composing local state space updates with global Fourier based mixing, the approach effectively captures both short and long range dependencies. 

---
# Big Help or Big Brother? Auditing Tracking, Profiling, and Personalization in Generative AI Assistants 

**Authors**: Yash Vekaria, Aurelio Loris Canino, Jonathan Levitsky, Alex Ciechonski, Patricia Callejo, Anna Maria Mandalari, Zubair Shafiq  

**Link**: [PDF](https://arxiv.org/pdf/2503.16586)  

**Abstract**: Generative AI (GenAI) browser assistants integrate powerful capabilities of GenAI in web browsers to provide rich experiences such as question answering, content summarization, and agentic navigation. These assistants, available today as browser extensions, can not only track detailed browsing activity such as search and click data, but can also autonomously perform tasks such as filling forms, raising significant privacy concerns. It is crucial to understand the design and operation of GenAI browser extensions, including how they collect, store, process, and share user data. To this end, we study their ability to profile users and personalize their responses based on explicit or inferred demographic attributes and interests of users. We perform network traffic analysis and use a novel prompting framework to audit tracking, profiling, and personalization by the ten most popular GenAI browser assistant extensions. We find that instead of relying on local in-browser models, these assistants largely depend on server-side APIs, which can be auto-invoked without explicit user interaction. When invoked, they collect and share webpage content, often the full HTML DOM and sometimes even the user's form inputs, with their first-party servers. Some assistants also share identifiers and user prompts with third-party trackers such as Google Analytics. The collection and sharing continues even if a webpage contains sensitive information such as health or personal information such as name or SSN entered in a web form. We find that several GenAI browser assistants infer demographic attributes such as age, gender, income, and interests and use this profile--which carries across browsing contexts--to personalize responses. In summary, our work shows that GenAI browser assistants can and do collect personal and sensitive information for profiling and personalization with little to no safeguards. 

---
