# Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs 

**Authors**: Yaniv Nikankin, Dana Arad, Yossi Gandelsman, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2506.09047)  

**Abstract**: Vision-Language models (VLMs) show impressive abilities to answer questions on visual inputs (e.g., counting objects in an image), yet demonstrate higher accuracies when performing an analogous task on text (e.g., counting words in a text). We investigate this accuracy gap by identifying and comparing the \textit{circuits} - the task-specific computational sub-graphs - in different modalities. We show that while circuits are largely disjoint between modalities, they implement relatively similar functionalities: the differences lie primarily in processing modality-specific data positions (an image or a text sequence). Zooming in on the image data representations, we observe they become aligned with the higher-performing analogous textual representations only towards later layers, too late in processing to effectively influence subsequent positions. To overcome this, we patch the representations of visual data tokens from later layers back into earlier layers. In experiments with multiple tasks and models, this simple intervention closes a third of the performance gap between the modalities, on average. Our analysis sheds light on the multi-modal performance gap in VLMs and suggests a training-free approach for reducing it. 

---
# Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning 

**Authors**: Haozhen Zhang, Tao Feng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.09033)  

**Abstract**: The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To guide learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for performance and cost trade-off optimization, opening a pathway toward optimizing performance-cost tradeoffs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms over several strong baselines, achieving superior performance while maintaining robust generalization and cost this http URL is available at this https URL. 

---
# Comparing human and LLM proofreading in L2 writing: Impact on lexical and syntactic features 

**Authors**: Hakyung Sung, Karla Csuros, Min-Chang Sung  

**Link**: [PDF](https://arxiv.org/pdf/2506.09021)  

**Abstract**: This study examines the lexical and syntactic interventions of human and LLM proofreading aimed at improving overall intelligibility in identical second language writings, and evaluates the consistency of outcomes across three LLMs (ChatGPT-4o, Llama3.1-8b, Deepseek-r1-8b). Findings show that both human and LLM proofreading enhance bigram lexical features, which may contribute to better coherence and contextual connectedness between adjacent words. However, LLM proofreading exhibits a more generative approach, extensively reworking vocabulary and sentence structures, such as employing more diverse and sophisticated vocabulary and incorporating a greater number of adjective modifiers in noun phrases. The proofreading outcomes are highly consistent in major lexical and syntactic features across the three models. 

---
# Learning to Reason Across Parallel Samples for LLM Reasoning 

**Authors**: Jianing Qi, Xi Ye, Hao Tang, Zhigang Zhu, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09014)  

**Abstract**: Scaling test-time compute brings substantial performance gains for large language models (LLMs). By sampling multiple answers and heuristically aggregate their answers (e.g., either through majority voting or using verifiers to rank the answers), one can achieve consistent performance gains in math domains. In this paper, we propose a new way to leverage such multiple sample set. We train a compact LLM, called Sample Set Aggregator (SSA), that takes a concatenated sequence of multiple samples and output the final answer, optimizing it for the answer accuracy with reinforcement learning. Experiments on multiple reasoning datasets show that SSA outperforms other test-time scaling methods such as reward model-based re-ranking. Our approach also shows a promising generalization ability, across sample set sizes, base model families and scales, and tasks. By separating LLMs to generate answers and LLMs to analyze and aggregate sampled answers, our approach can work with the outputs from premier black box models easily and efficiently. 

---
# UD-KSL Treebank v1.3: A semi-automated framework for aligning XPOS-extracted units with UPOS tags 

**Authors**: Hakyung Sung, Gyu-Ho Shin, Chanyoung Lee, You Kyung Sung, Boo Kyung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2506.09009)  

**Abstract**: The present study extends recent work on Universal Dependencies annotations for second-language (L2) Korean by introducing a semi-automated framework that identifies morphosyntactic constructions from XPOS sequences and aligns those constructions with corresponding UPOS categories. We also broaden the existing L2-Korean corpus by annotating 2,998 new sentences from argumentative essays. To evaluate the impact of XPOS-UPOS alignments, we fine-tune L2-Korean morphosyntactic analysis models on datasets both with and without these alignments, using two NLP toolkits. Our results indicate that the aligned dataset not only improves consistency across annotation layers but also enhances morphosyntactic tagging and dependency-parsing accuracy, particularly in cases of limited annotated data. 

---
# SWE-Flow: Synthesizing Software Engineering Data in a Test-Driven Manner 

**Authors**: Lei Zhang, Jiaxi Yang, Min Yang, Jian Yang, Mouxiang Chen, Jiajun Zhang, Zeyu Cui, Binyuan Hui, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.09003)  

**Abstract**: We introduce **SWE-Flow**, a novel data synthesis framework grounded in Test-Driven Development (TDD). Unlike existing software engineering data that rely on human-submitted issues, **SWE-Flow** automatically infers incremental development steps directly from unit tests, which inherently encapsulate high-level requirements. The core of **SWE-Flow** is the construction of a Runtime Dependency Graph (RDG), which precisely captures function interactions, enabling the generation of a structured, step-by-step *development schedule*. At each step, **SWE-Flow** produces a partial codebase, the corresponding unit tests, and the necessary code modifications, resulting in fully verifiable TDD tasks. With this approach, we generated 16,061 training instances and 2,020 test instances from real-world GitHub projects, creating the **SWE-Flow-Eval** benchmark. Our experiments show that fine-tuning open model on this dataset significantly improves performance in TDD-based coding. To facilitate further research, we release all code, datasets, models, and Docker images at [Github](this https URL). 

---
# Employing self-supervised learning models for cross-linguistic child speech maturity classification 

**Authors**: Theo Zhang, Madurya Suresh, Anne S. Warlaumont, Kasia Hitczenko, Alejandrina Cristia, Margaret Cychosz  

**Link**: [PDF](https://arxiv.org/pdf/2506.08999)  

**Abstract**: Speech technology systems struggle with many downstream tasks for child speech due to small training corpora and the difficulties that child speech pose. We apply a novel dataset, SpeechMaturity, to state-of-the-art transformer models to address a fundamental classification task: identifying child vocalizations. Unlike previous corpora, our dataset captures maximally ecologically-valid child vocalizations across an unprecedented sample, comprising children acquiring 25+ languages in the U.S., Bolivia, Vanuatu, Papua New Guinea, Solomon Islands, and France. The dataset contains 242,004 labeled vocalizations, magnitudes larger than previous work. Models were trained to distinguish between cry, laughter, mature (consonant+vowel), and immature speech (just consonant or vowel). Models trained on the dataset outperform state-of-the-art models trained on previous datasets, achieved classification accuracy comparable to humans, and were robust across rural and urban settings. 

---
# Naturalistic Language-related Movie-Watching fMRI Task for Detecting Neurocognitive Decline and Disorder 

**Authors**: Yuejiao Wang, Xianmin Gong, Xixin Wu, Patrick Wong, Hoi-lam Helene Fung, Man Wai Mak, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08986)  

**Abstract**: Early detection is crucial for timely intervention aimed at preventing and slowing the progression of neurocognitive disorder (NCD), a common and significant health problem among the aging population. Recent evidence has suggested that language-related functional magnetic resonance imaging (fMRI) may be a promising approach for detecting cognitive decline and early NCD. In this paper, we proposed a novel, naturalistic language-related fMRI task for this purpose. We examined the effectiveness of this task among 97 non-demented Chinese older adults from Hong Kong. The results showed that machine-learning classification models based on fMRI features extracted from the task and demographics (age, gender, and education year) achieved an average area under the curve of 0.86 when classifying participants' cognitive status (labeled as NORMAL vs DECLINE based on their scores on a standard neurcognitive test). Feature localization revealed that the fMRI features most frequently selected by the data-driven approach came primarily from brain regions associated with language processing, such as the superior temporal gyrus, middle temporal gyrus, and right cerebellum. The study demonstrated the potential of the naturalistic language-related fMRI task for early detection of aging-related cognitive decline and NCD. 

---
# FROST-EMA: Finnish and Russian Oral Speech Dataset of Electromagnetic Articulography Measurements with L1, L2 and Imitated L2 Accents 

**Authors**: Satu Hopponen, Tomi Kinnunen, Alexandre Nikolaev, Rosa González Hautamäki, Lauri Tavi, Einar Meister  

**Link**: [PDF](https://arxiv.org/pdf/2506.08981)  

**Abstract**: We introduce a new FROST-EMA (Finnish and Russian Oral Speech Dataset of Electromagnetic Articulography) corpus. It consists of 18 bilingual speakers, who produced speech in their native language (L1), second language (L2), and imitated L2 (fake foreign accent). The new corpus enables research into language variability from phonetic and technological points of view. Accordingly, we include two preliminary case studies to demonstrate both perspectives. The first case study explores the impact of L2 and imitated L2 on the performance of an automatic speaker verification system, while the second illustrates the articulatory patterns of one speaker in L1, L2, and a fake accent. 

---
# Atomic-to-Compositional Generalization for Mobile Agents with A New Benchmark and Scheduling System 

**Authors**: Yuan Guo, Tingjia Miao, Zheng Wu, Pengzhou Cheng, Ming Zhou, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08972)  

**Abstract**: Autonomous agents powered by multimodal large language models have been developed to facilitate task execution on mobile devices. However, prior work has predominantly focused on atomic tasks -- such as shot-chain execution tasks and single-screen grounding tasks -- while overlooking the generalization to compositional tasks, which are indispensable for real-world applications. This work introduces UI-NEXUS, a comprehensive benchmark designed to evaluate mobile agents on three categories of compositional operations: Simple Concatenation, Context Transition, and Deep Dive. UI-NEXUS supports interactive evaluation in 20 fully controllable local utility app environments, as well as 30 online Chinese and English service apps. It comprises 100 interactive task templates with an average optimal step count of 14.05. Experimental results across a range of mobile agents with agentic workflow or agent-as-a-model show that UI-NEXUS presents significant challenges. Specifically, existing agents generally struggle to balance performance and efficiency, exhibiting representative failure modes such as under-execution, over-execution, and attention drift, causing visible atomic-to-compositional generalization gap. Inspired by these findings, we propose AGENT-NEXUS, a lightweight and efficient scheduling system to tackle compositional mobile tasks. AGENT-NEXUS extrapolates the abilities of existing mobile agents by dynamically decomposing long-horizon tasks to a series of self-contained atomic subtasks. AGENT-NEXUS achieves 24% to 40% task success rate improvement for existing mobile agents on compositional operation tasks within the UI-NEXUS benchmark without significantly sacrificing inference overhead. The demo video, dataset, and code are available on the project page at this https URL. 

---
# Pre-trained Language Models Learn Remarkably Accurate Representations of Numbers 

**Authors**: Marek Kadlčík, Michal Štefánik, Timothee Mickus, Michal Spiegel, Josef Kuchař  

**Link**: [PDF](https://arxiv.org/pdf/2506.08966)  

**Abstract**: Pretrained language models (LMs) are prone to arithmetic errors. Existing work showed limited success in probing numeric values from models' representations, indicating that these errors can be attributed to the inherent unreliability of distributionally learned embeddings in representing exact quantities. However, we observe that previous probing methods are inadequate for the emergent structure of learned number embeddings with sinusoidal patterns.
In response, we propose a novel probing technique that decodes numeric values from input embeddings with near-perfect accuracy across a range of open-source LMs. This proves that after the sole pre-training, LMs represent numbers with remarkable precision. Finally, we find that the embeddings' preciseness judged by our probe's accuracy explains a large portion of LM's errors in elementary arithmetic, and show that aligning the embeddings with the pattern discovered by our probe can mitigate these errors. 

---
# Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions 

**Authors**: Clara Lachenmaier, Judith Sieker, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2506.08952)  

**Abstract**: Communication among humans relies on conversational grounding, allowing interlocutors to reach mutual understanding even when they do not have perfect knowledge and must resolve discrepancies in each other's beliefs. This paper investigates how large language models (LLMs) manage common ground in cases where they (don't) possess knowledge, focusing on facts in the political domain where the risk of misinformation and grounding failure is high. We examine the ability of LLMs to answer direct knowledge questions and loaded questions that presuppose misinformation. We evaluate whether loaded questions lead LLMs to engage in active grounding and correct false user beliefs, in connection to their level of knowledge and their political bias. Our findings highlight significant challenges in LLMs' ability to engage in grounding and reject false user beliefs, raising concerns about their role in mitigating misinformation in political discourse. 

---
# FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation 

**Authors**: Qinggang Zhang, Zhishang Xiang, Yilin Xiao, Le Wang, Junhui Li, Xinrun Wang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.08938)  

**Abstract**: Large language models (LLMs) augmented with retrieval systems have demonstrated significant potential in handling knowledge-intensive tasks. However, these models often struggle with unfaithfulness issues, generating outputs that either ignore the retrieved context or inconsistently blend it with the LLM`s parametric knowledge. This issue is particularly severe in cases of knowledge conflict, where the retrieved context conflicts with the model`s parametric knowledge. While existing faithful RAG approaches enforce strict context adherence through well-designed prompts or modified decoding strategies, our analysis reveals a critical limitation: they achieve faithfulness by forcibly suppressing the model`s parametric knowledge, which undermines the model`s internal knowledge structure and increases the risk of misinterpreting the context. To this end, this paper proposes FaithfulRAG, a novel framework that resolves knowledge conflicts by explicitly modeling discrepancies between the model`s parametric knowledge and retrieved context. Specifically, FaithfulRAG identifies conflicting knowledge at the fact level and designs a self-thinking process, allowing LLMs to reason about and integrate conflicting facts before generating responses. Extensive experiments demonstrate that our method outperforms state-of-the-art methods. The code is available at https:// this http URL 

---
# Can A Gamer Train A Mathematical Reasoning Model? 

**Authors**: Andrew Shin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08935)  

**Abstract**: While large language models (LLMs) have achieved remarkable performance in various tasks including mathematical reasoning, their development typically demands prohibitive computational resources. Recent advancements have reduced costs for training capable models, yet even these approaches rely on high-end hardware clusters. In this paper, we demonstrate that a single average gaming GPU can train a solid mathematical reasoning model, by integrating reinforcement learning and memory optimization techniques. Specifically, we train a 1.5B parameter mathematical reasoning model on RTX 3080 Ti of 16GB memory that achieves comparable or better performance on mathematical reasoning benchmarks than models several times larger, in resource-constrained environments. Our results challenge the paradigm that state-of-the-art mathematical reasoning necessitates massive infrastructure, democratizing access to high-performance AI research. this https URL. 

---
# PropMEND: Hypernetworks for Knowledge Propagation in LLMs 

**Authors**: Zeyu Leo Liu, Greg Durrett, Eunsol Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08920)  

**Abstract**: Knowledge editing techniques for large language models (LLMs) can inject knowledge that is later reproducible verbatim, but they fall short on propagating that knowledge: models cannot answer questions that require reasoning with the injected knowledge. We present a hypernetwork-based approach for knowledge propagation, named PropMEND, where we meta-learn how to modify gradients of a language modeling loss to encourage injected information to propagate. Our approach extends the meta-objective of MEND [29] so that gradient updates on knowledge are transformed to enable answering multi-hop questions involving that knowledge. We show improved performance on the RippleEdit dataset, showing almost 2x accuracy on challenging multi-hop questions whose answers are not explicitly stated in the injected fact. We further introduce a new dataset, Controlled RippleEdit, to evaluate the generalization of our hypernetwork, testing knowledge propagation along relations and entities unseen during hypernetwork training. PropMEND still outperforms existing approaches in unseen entity-relation pairs, yet the performance gap decreases substantially, suggesting future work in propagating knowledge to a wide range of relations. 

---
# Dialect Normalization using Large Language Models and Morphological Rules 

**Authors**: Antonios Dimakis, John Pavlopoulos, Antonios Anastasopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2506.08907)  

**Abstract**: Natural language understanding systems struggle with low-resource languages, including many dialects of high-resource ones. Dialect-to-standard normalization attempts to tackle this issue by transforming dialectal text so that it can be used by standard-language tools downstream. In this study, we tackle this task by introducing a new normalization method that combines rule-based linguistically informed transformations and large language models (LLMs) with targeted few-shot prompting, without requiring any parallel data. We implement our method for Greek dialects and apply it on a dataset of regional proverbs, evaluating the outputs using human annotators. We then use this dataset to conduct downstream experiments, finding that previous results regarding these proverbs relied solely on superficial linguistic information, including orthographic artifacts, while new observations can still be made through the remaining semantics. 

---
# From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis 

**Authors**: Elias Horner, Cristinel Mateis, Guido Governatori, Agata Ciabattoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.08899)  

**Abstract**: We present a novel approach to the automated semantic analysis of legal texts using large language models (LLMs), targeting their transformation into formal representations in Defeasible Deontic Logic (DDL). We propose a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. Our methodology is evaluated across various LLM configurations, including prompt engineering strategies, fine-tuned models, and multi-stage pipelines, focusing on legal norms from the Australian Telecommunications Consumer Protections Code. Empirical results demonstrate promising alignment between machine-generated and expert-crafted formalizations, showing that LLMs - particularly when prompted effectively - can significantly contribute to scalable legal informatics. 

---
# PlantBert: An Open Source Language Model for Plant Science 

**Authors**: Hiba Khey, Amine Lakhder, Salma Rouichi, Imane El Ghabi, Kamal Hejjaoui, Younes En-nahli, Fahd Kalloubi, Moez Amri  

**Link**: [PDF](https://arxiv.org/pdf/2506.08897)  

**Abstract**: The rapid advancement of transformer-based language models has catalyzed breakthroughs in biomedical and clinical natural language processing; however, plant science remains markedly underserved by such domain-adapted tools. In this work, we present PlantBert, a high-performance, open-source language model specifically tailored for extracting structured knowledge from plant stress-response literature. Built upon the DeBERTa architecture-known for its disentangled attention and robust contextual encoding-PlantBert is fine-tuned on a meticulously curated corpus of expert-annotated abstracts, with a primary focus on lentil (Lens culinaris) responses to diverse abiotic and biotic stressors. Our methodology combines transformer-based modeling with rule-enhanced linguistic post-processing and ontology-grounded entity normalization, enabling PlantBert to capture biologically meaningful relationships with precision and semantic fidelity. The underlying corpus is annotated using a hierarchical schema aligned with the Crop Ontology, encompassing molecular, physiological, biochemical, and agronomic dimensions of plant adaptation. PlantBert exhibits strong generalization capabilities across entity types and demonstrates the feasibility of robust domain adaptation in low-resource scientific fields. By providing a scalable and reproducible framework for high-resolution entity recognition, PlantBert bridges a critical gap in agricultural NLP and paves the way for intelligent, data-driven systems in plant genomics, phenomics, and agronomic knowledge discovery. Our model is publicly released to promote transparency and accelerate cross-disciplinary innovation in computational plant science. 

---
# AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI) 

**Authors**: Danush Khanna, Krishna Kumar, Basab Ghosh, Vinija Jain, Vasu Sharma, Aman Chadha, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2506.08885)  

**Abstract**: Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at this https URL. 

---
# Advancing STT for Low-Resource Real-World Speech 

**Authors**: Flavio D'Intino, Hans-Peter Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.08836)  

**Abstract**: Swiss German is a low-resource language represented by diverse dialects that differ significantly from Standard German and from each other, lacking a standardized written form. As a result, transcribing Swiss German involves translating into Standard German. Existing datasets have been collected in controlled environments, yielding effective speech-to-text (STT) models, but these models struggle with spontaneous conversational speech.
This paper, therefore, introduces the new SRB-300 dataset, a 300-hour annotated speech corpus featuring real-world long-audio recordings from 39 Swiss German radio and TV stations. It captures spontaneous speech across all major Swiss dialects recorded in various realistic environments and overcomes the limitation of prior sentence-level corpora.
We fine-tuned multiple OpenAI Whisper models on the SRB-300 dataset, achieving notable enhancements over previous zero-shot performance metrics. Improvements in word error rate (WER) ranged from 19% to 33%, while BLEU scores increased between 8% and 40%. The best fine-tuned model, large-v3, achieved a WER of 17.1% and a BLEU score of 74.8. This advancement is crucial for developing effective and robust STT systems for Swiss German and other low-resource languages in real-world contexts. 

---
# The impact of fine tuning in LLaMA on hallucinations for named entity extraction in legal documentation 

**Authors**: Francisco Vargas, Alejandro González Coene, Gaston Escalante, Exequiel Lobón, Manuel Pulido  

**Link**: [PDF](https://arxiv.org/pdf/2506.08827)  

**Abstract**: The extraction of information about traffic accidents from legal documents is crucial for quantifying insurance company costs. Extracting entities such as percentages of physical and/or psychological disability and the involved compensation amounts is a challenging process, even for experts, due to the subtle arguments and reasoning in the court decision. A two-step procedure is proposed: first, segmenting the document identifying the most relevant segments, and then extracting the entities. For text segmentation, two methodologies are compared: a classic method based on regular expressions and a second approach that divides the document into blocks of n-tokens, which are then vectorized using multilingual models for semantic searches (text-embedding-ada-002/MiniLM-L12-v2 ). Subsequently, large language models (LLaMA-2 7b, 70b, LLaMA-3 8b, and GPT-4 Turbo) are applied with prompting to the selected segments for entity extraction. For the LLaMA models, fine-tuning is performed using LoRA. LLaMA-2 7b, even with zero temperature, shows a significant number of hallucinations in extractions which are an important contention point for named entity extraction. This work shows that these hallucinations are substantially reduced after finetuning the model. The performance of the methodology based on segment vectorization and subsequent use of LLMs significantly surpasses the classic method which achieves an accuracy of 39.5%. Among open-source models, LLaMA-2 70B with finetuning achieves the highest accuracy 79.4%, surpassing its base version 61.7%. Notably, the base LLaMA-3 8B model already performs comparably to the finetuned LLaMA-2 70B model, achieving 76.6%, highlighting the rapid progress in model development. Meanwhile, GPT-4 Turbo achieves the highest accuracy at 86.1%. 

---
# AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP 

**Authors**: Ahmed Hasanaath, Aisha Alansari, Ahmed Ashraf, Chafik Salmane, Hamzah Luqman, Saad Ezzini  

**Link**: [PDF](https://arxiv.org/pdf/2506.08768)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at this https URL 

---
# Enhancing Accuracy and Maintainability in Nuclear Plant Data Retrieval: A Function-Calling LLM Approach Over NL-to-SQL 

**Authors**: Mishca de Costa, Muhammad Anwar, Dave Mercier, Mark Randall, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2506.08757)  

**Abstract**: Retrieving operational data from nuclear power plants requires exceptional accuracy and transparency due to the criticality of the decisions it supports. Traditionally, natural language to SQL (NL-to-SQL) approaches have been explored for querying such data. While NL-to-SQL promises ease of use, it poses significant risks: end-users cannot easily validate generated SQL queries, and legacy nuclear plant databases -- often complex and poorly structured -- complicate query generation due to decades of incremental modifications. These challenges increase the likelihood of inaccuracies and reduce trust in the approach. In this work, we propose an alternative paradigm: leveraging function-calling large language models (LLMs) to address these challenges. Instead of directly generating SQL queries, we define a set of pre-approved, purpose-specific functions representing common use cases. Queries are processed by invoking these functions, which encapsulate validated SQL logic. This hybrid approach mitigates the risks associated with direct NL-to-SQL translations by ensuring that SQL queries are reviewed and optimized by experts before deployment. While this strategy introduces the upfront cost of developing and maintaining the function library, we demonstrate how NL-to-SQL tools can assist in the initial generation of function code, allowing experts to focus on validation rather than creation. Our study includes a performance comparison between direct NL-to-SQL generation and the proposed function-based approach, highlighting improvements in accuracy and maintainability. This work underscores the importance of balancing user accessibility with operational safety and provides a novel, actionable framework for robust data retrieval in critical systems. 

---
# Factors affecting the in-context learning abilities of LLMs for dialogue state tracking 

**Authors**: Pradyoth Hegde, Santosh Kesiraju, Jan Švec, Šimon Sedláček, Bolaji Yusuf, Oldřich Plchot, Deepak K T, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2506.08753)  

**Abstract**: This study explores the application of in-context learning (ICL) to the dialogue state tracking (DST) problem and investigates the factors that influence its effectiveness. We use a sentence embedding based k-nearest neighbour method to retrieve the suitable demonstrations for ICL. The selected demonstrations, along with the test samples, are structured within a template as input to the LLM. We then conduct a systematic study to analyse the impact of factors related to demonstration selection and prompt context on DST performance. This work is conducted using the MultiWoZ2.4 dataset and focuses primarily on the OLMo-7B-instruct, Mistral-7B-Instruct-v0.3, and Llama3.2-3B-Instruct models. Our findings provide several useful insights on in-context learning abilities of LLMs for dialogue state tracking. 

---
# Unlocking the Potential of Large Language Models in the Nuclear Industry with Synthetic Data 

**Authors**: Muhammad Anwar, Daniel Lau, Mishca de Costa, Issam Hammad  

**Link**: [PDF](https://arxiv.org/pdf/2506.08750)  

**Abstract**: The nuclear industry possesses a wealth of valuable information locked away in unstructured text data. This data, however, is not readily usable for advanced Large Language Model (LLM) applications that require clean, structured question-answer pairs for tasks like model training, fine-tuning, and evaluation. This paper explores how synthetic data generation can bridge this gap, enabling the development of robust LLMs for the nuclear domain. We discuss the challenges of data scarcity and privacy concerns inherent in the nuclear industry and how synthetic data provides a solution by transforming existing text data into usable Q&A pairs. This approach leverages LLMs to analyze text, extract key information, generate relevant questions, and evaluate the quality of the resulting synthetic dataset. By unlocking the potential of LLMs in the nuclear industry, synthetic data can pave the way for improved information retrieval, enhanced knowledge sharing, and more informed decision-making in this critical sector. 

---
# Towards Secure and Private Language Models for Nuclear Power Plants 

**Authors**: Muhammad Anwar, Mishca de Costa, Issam Hammad, Daniel Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.08746)  

**Abstract**: This paper introduces a domain-specific Large Language Model for nuclear applications, built from the publicly accessible Essential CANDU textbook. Drawing on a compact Transformer-based architecture, the model is trained on a single GPU to protect the sensitive data inherent in nuclear operations. Despite relying on a relatively small dataset, it shows encouraging signs of capturing specialized nuclear vocabulary, though the generated text sometimes lacks syntactic coherence. By focusing exclusively on nuclear content, this approach demonstrates the feasibility of in-house LLM solutions that align with rigorous cybersecurity and data confidentiality standards. Early successes in text generation underscore the model's utility for specialized tasks, while also revealing the need for richer corpora, more sophisticated preprocessing, and instruction fine-tuning to enhance domain accuracy. Future directions include extending the dataset to cover diverse nuclear subtopics, refining tokenization to reduce noise, and systematically evaluating the model's readiness for real-world applications in nuclear domain. 

---
# Societal AI Research Has Become Less Interdisciplinary 

**Authors**: Dror Kris Markus, Fabrizio Gilardi, Daria Stetsenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.08738)  

**Abstract**: As artificial intelligence (AI) systems become deeply embedded in everyday life, calls to align AI development with ethical and societal values have intensified. Interdisciplinary collaboration is often championed as a key pathway for fostering such engagement. Yet it remains unclear whether interdisciplinary research teams are actually leading this shift in practice. This study analyzes over 100,000 AI-related papers published on ArXiv between 2014 and 2024 to examine how ethical values and societal concerns are integrated into technical AI research. We develop a classifier to identify societal content and measure the extent to which research papers express these considerations. We find a striking shift: while interdisciplinary teams remain more likely to produce societally-oriented research, computer science-only teams now account for a growing share of the field's overall societal output. These teams are increasingly integrating societal concerns into their papers and tackling a wide range of domains - from fairness and safety to healthcare and misinformation. These findings challenge common assumptions about the drivers of societal AI and raise important questions. First, what are the implications for emerging understandings of AI safety and governance if most societally-oriented research is being undertaken by exclusively technical teams? Second, for scholars in the social sciences and humanities: in a technical field increasingly responsive to societal demands, what distinctive perspectives can we still offer to help shape the future of AI? 

---
# Improved LLM Agents for Financial Document Question Answering 

**Authors**: Nelvin Tan, Zian Seng, Liang Zhang, Yu-Ching Shih, Dong Yang, Amol Salunkhe  

**Link**: [PDF](https://arxiv.org/pdf/2506.08726)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities on numerous natural language processing tasks. However, LLMs still struggle with numerical question answering for financial documents that include tabular and textual data. Recent works have showed the effectiveness of critic agents (i.e., self-correction) for this task given oracle labels. Building upon this framework, this paper examines the effectiveness of the traditional critic agent when oracle labels are not available, and show, through experiments, that this critic agent's performance deteriorates in this scenario. With this in mind, we present an improved critic agent, along with the calculator agent which outperforms the previous state-of-the-art approach (program-of-thought) and is safer. Furthermore, we investigate how our agents interact with each other, and how this interaction affects their performance. 

---
# Multi-Teacher Language-Aware Knowledge Distillation for Multilingual Speech Emotion Recognition 

**Authors**: Mehedi Hasan Bijoy, Dejan Porjazovski, Tamás Grósz, Mikko Kurimo  

**Link**: [PDF](https://arxiv.org/pdf/2506.08717)  

**Abstract**: Speech Emotion Recognition (SER) is crucial for improving human-computer interaction. Despite strides in monolingual SER, extending them to build a multilingual system remains challenging. Our goal is to train a single model capable of multilingual SER by distilling knowledge from multiple teacher models. To address this, we introduce a novel language-aware multi-teacher knowledge distillation method to advance SER in English, Finnish, and French. It leverages Wav2Vec2.0 as the foundation of monolingual teacher models and then distills their knowledge into a single multilingual student model. The student model demonstrates state-of-the-art performance, with a weighted recall of 72.9 on the English dataset and an unweighted recall of 63.4 on the Finnish dataset, surpassing fine-tuning and knowledge distillation baselines. Our method excels in improving recall for sad and neutral emotions, although it still faces challenges in recognizing anger and happiness. 

---
# Explainable Compliance Detection with Multi-Hop Natural Language Inference on Assurance Case Structure 

**Authors**: Fariz Ikhwantri, Dusica Marijan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08713)  

**Abstract**: Ensuring complex systems meet regulations typically requires checking the validity of assurance cases through a claim-argument-evidence framework. Some challenges in this process include the complicated nature of legal and technical texts, the need for model explanations, and limited access to assurance case data. We propose a compliance detection approach based on Natural Language Inference (NLI): EXplainable CompLiance detection with Argumentative Inference of Multi-hop reasoning (EXCLAIM). We formulate the claim-argument-evidence structure of an assurance case as a multi-hop inference for explainable and traceable compliance detection. We address the limited number of assurance cases by generating them using large language models (LLMs). We introduce metrics that measure the coverage and structural consistency. We demonstrate the effectiveness of the generated assurance case from GDPR requirements in a multi-hop inference task as a case study. Our results highlight the potential of NLI-based approaches in automating the regulatory compliance process. 

---
# ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Large Language Model Preference Optimization 

**Authors**: Hee Suk Yoon, Eunseop Yoon, Mark A. Hasegawa-Johnson, Sungwoong Kim, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.08712)  

**Abstract**: We introduce ConfPO, a method for preference learning in Large Language Models (LLMs) that identifies and optimizes preference-critical tokens based solely on the training policy's confidence, without requiring any auxiliary models or compute. Unlike prior Direct Alignment Algorithms (DAAs) such as Direct Preference Optimization (DPO), which uniformly adjust all token probabilities regardless of their relevance to preference, ConfPO focuses optimization on the most impactful tokens. This targeted approach improves alignment quality while mitigating overoptimization (i.e., reward hacking) by using the KL divergence budget more efficiently. In contrast to recent token-level methods that rely on credit-assignment models or AI annotators, raising concerns about scalability and reliability, ConfPO is simple, lightweight, and model-free. Experimental results on challenging alignment benchmarks, including AlpacaEval 2 and Arena-Hard, demonstrate that ConfPO consistently outperforms uniform DAAs across various LLMs, delivering better alignment with zero additional computational overhead. 

---
# ClimateViz: A Benchmark for Statistical Reasoning and Fact Verification on Scientific Charts 

**Authors**: Ruiran Su, Jiasheng Si, Zhijiang Guo, Janet B. Pierrehumbert  

**Link**: [PDF](https://arxiv.org/pdf/2506.08700)  

**Abstract**: Scientific fact-checking has mostly focused on text and tables, overlooking scientific charts, which are key for presenting quantitative evidence and statistical reasoning. We introduce ClimateViz, the first large-scale benchmark for scientific fact-checking using expert-curated scientific charts. ClimateViz contains 49,862 claims linked to 2,896 visualizations, each labeled as support, refute, or not enough information. To improve interpretability, each example includes structured knowledge graph explanations covering trends, comparisons, and causal relations. We evaluate state-of-the-art multimodal language models, including both proprietary and open-source systems, in zero-shot and few-shot settings. Results show that current models struggle with chart-based reasoning: even the best systems, such as Gemini 2.5 and InternVL 2.5, reach only 76.2 to 77.8 percent accuracy in label-only settings, far below human performance (89.3 and 92.7 percent). Explanation-augmented outputs improve performance in some models. We released our dataset and code alongside the paper. 

---
# Brevity is the soul of sustainability: Characterizing LLM response lengths 

**Authors**: Soham Poddar, Paramita Koley, Janardan Misra, Sanjay Podder, Navveen Balani, Niloy Ganguly, Saptarshi Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.08686)  

**Abstract**: A significant portion of the energy consumed by Large Language Models (LLMs) arises from their inference processes; hence developing energy-efficient methods for inference is crucial. While several techniques exist for inference optimization, output compression remains relatively unexplored, with only a few preliminary efforts addressing this aspect. In this work, we first benchmark 12 decoder-only LLMs across 5 datasets, revealing that these models often produce responses that are substantially longer than necessary. We then conduct a comprehensive quality assessment of LLM responses, formally defining six information categories present in LLM responses. We show that LLMs often tend to include redundant or additional information besides the minimal answer. To address this issue of long responses by LLMs, we explore several simple and intuitive prompt-engineering strategies. Empirical evaluation shows that appropriate prompts targeting length reduction and controlling information content can achieve significant energy optimization between 25-60\% by reducing the response length while preserving the quality of LLM responses. 

---
# RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling 

**Authors**: Yang Liu, Jiaqi Li, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08672)  

**Abstract**: Rule-based reasoning has been acknowledged as one of the fundamental problems in reasoning, while deviations in rule formats, types, and complexity in real-world applications pose severe challenges. Recent studies have shown that large reasoning models (LRMs) have remarkable reasoning capabilities, and their performance is substantially enhanced by reinforcement learning (RL). However, it remains an open question whether small reasoning models (SRMs) can learn rule-based reasoning effectively with robust generalization across diverse tasks and domains. To address this, we introduce Reinforced Rule-based Reasoning, a.k.a. RuleReasoner, a simple yet effective method to conduct rule-based reasoning via a wide collection of curated tasks and a novel domain-aware dynamic sampling approach. Specifically, RuleReasoner resamples each training batch by updating the sampling weights of different domains based on historical rewards. This facilitates domain augmentation and flexible online learning schedules for RL, obviating the need for pre-hoc human-engineered mix-training recipes used in existing methods. Empirical evaluations on in-distribution (ID) and out-of-distribution (OOD) benchmarks reveal that RuleReasoner outperforms frontier LRMs by a significant margin ($\Delta$4.1% average points on eight ID tasks and $\Delta$10.4% average points on three OOD tasks over OpenAI-o1). Notably, our approach also exhibits higher computational efficiency compared to prior dynamic sampling methods for RL. 

---
# Summarization for Generative Relation Extraction in the Microbiome Domain 

**Authors**: Oumaima El Khettari, Solen Quiniou, Samuel Chaffron  

**Link**: [PDF](https://arxiv.org/pdf/2506.08647)  

**Abstract**: We explore a generative relation extraction (RE) pipeline tailored to the study of interactions in the intestinal microbiome, a complex and low-resource biomedical domain. Our method leverages summarization with large language models (LLMs) to refine context before extracting relations via instruction-tuned generation. Preliminary results on a dedicated corpus show that summarization improves generative RE performance by reducing noise and guiding the model. However, BERT-based RE approaches still outperform generative models. This ongoing work demonstrates the potential of generative methods to support the study of specialized domains in low-resources setting. 

---
# TableDreamer: Progressive and Weakness-guided Data Synthesis from Scratch for Table Instruction Tuning 

**Authors**: Mingyu Zheng, Zhifan Feng, Jia Wang, Lanrui Wang, Zheng Lin, Yang Hao, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08646)  

**Abstract**: Despite the commendable progress of recent LLM-based data synthesis methods, they face two limitations in generating table instruction tuning data. First, they can not thoroughly explore the vast input space of table understanding tasks, leading to limited data diversity. Second, they ignore the weaknesses in table understanding ability of the target LLM and blindly pursue the increase of data quantity, resulting in suboptimal data efficiency. In this paper, we introduce a progressive and weakness-guided data synthesis framework tailored for table instruction tuning, named TableDreamer, to mitigate the above issues. Specifically, we first synthesize diverse tables and related instructions as seed data, and then perform an iterative exploration of the input space under the guidance of the newly identified weakness data, which eventually serve as the final training data for fine-tuning the target LLM. Extensive experiments on 10 tabular benchmarks demonstrate the effectiveness of the proposed framework, which boosts the average accuracy of Llama3.1-8B-instruct by 11.62% (49.07% to 60.69%) with 27K GPT-4o synthetic data and outperforms state-of-the-art data synthesis baselines which use more training data. The code and data is available at this https URL 

---
# MEMETRON: Metaheuristic Mechanisms for Test-time Response Optimization of Large Language Models 

**Authors**: Son The Nguyen, Theja Tulabandhula  

**Link**: [PDF](https://arxiv.org/pdf/2506.08643)  

**Abstract**: Large language models (LLMs) are increasingly used for both open-ended and structured tasks, yet their inference-time behavior is still largely dictated by heuristic decoding strategies such as greedy search, sampling, or reranking. These methods provide limited control and do not explicitly optimize for task-specific objectives. We introduce MEMETRON, a task-agnostic framework that formulates LLM decoding as a discrete black-box optimization problem. MEMETRON leverages hybrid metaheuristic algorithms, GENETRON and ANNETRON, to search the response space, guided by reward models and contextual operations performed by the LLM itself. This approach enables efficient discovery of high-reward responses without requiring model retraining or gradient access. The framework is modular and generalizes across diverse tasks, requiring only a reward function and lightweight prompt templates. We evaluate our framework on the critical human preference alignment task and demonstrate that it significantly outperforms standard decoding and reranking methods, highlighting its potential to improve alignment without model retraining. 

---
# RAISE: Enhancing Scientific Reasoning in LLMs via Step-by-Step Retrieval 

**Authors**: Minhae Oh, Jeonghye Kim, Nakyung Lee, Donggeon Seo, Taeuk Kim, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08625)  

**Abstract**: Scientific reasoning requires not only long-chain reasoning processes, but also knowledge of domain-specific terminologies and adaptation to updated findings. To deal with these challenges for scientific reasoning, we introduce RAISE, a step-by-step retrieval-augmented framework which retrieves logically relevant documents from in-the-wild corpus. RAISE is divided into three steps: problem decomposition, logical query generation, and logical retrieval. We observe that RAISE consistently outperforms other baselines on scientific reasoning benchmarks. We analyze that unlike other baselines, RAISE retrieves documents that are not only similar in terms of the domain knowledge, but also documents logically more relevant. 

---
# Hateful Person or Hateful Model? Investigating the Role of Personas in Hate Speech Detection by Large Language Models 

**Authors**: Shuzhou Yuan, Ercong Nie, Mario Tawfelis, Helmut Schmid, Hinrich Schütze, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.08593)  

**Abstract**: Hate speech detection is a socially sensitive and inherently subjective task, with judgments often varying based on personal traits. While prior work has examined how socio-demographic factors influence annotation, the impact of personality traits on Large Language Models (LLMs) remains largely unexplored. In this paper, we present the first comprehensive study on the role of persona prompts in hate speech classification, focusing on MBTI-based traits. A human annotation survey confirms that MBTI dimensions significantly affect labeling behavior. Extending this to LLMs, we prompt four open-source models with MBTI personas and evaluate their outputs across three hate speech datasets. Our analysis uncovers substantial persona-driven variation, including inconsistencies with ground truth, inter-persona disagreement, and logit-level biases. These findings highlight the need to carefully define persona prompts in LLM-based annotation workflows, with implications for fairness and alignment with human values. 

---
# Dense Retrievers Can Fail on Simple Queries: Revealing The Granularity Dilemma of Embeddings 

**Authors**: Liyan Xu, Zhenlin Su, Mo Yu, Jiangnan Li, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08592)  

**Abstract**: This work focuses on an observed limitation of text encoders: embeddings may not be able to recognize fine-grained entities or events within the semantics, resulting in failed dense retrieval on even simple cases. To examine such behaviors, we first introduce a new evaluation dataset in Chinese, named CapRetrieval, whose passages are image captions, and queries are phrases inquiring entities or events in various forms. Zero-shot evaluation suggests that encoders may fail on these fine-grained matching, regardless of training sources or model sizes. Aiming for enhancement, we proceed to finetune encoders with our proposed data generation strategies, which obtains the best performance on CapRetrieval. Within this process, we further identify an issue of granularity dilemma, a challenge for embeddings to express fine-grained salience while aligning with overall semantics. Our dataset, code and models in this work are publicly released at this https URL. 

---
# CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmark of Large Language Models in Mental Health Counseling 

**Authors**: Yahan Li, Jifan Yao, John Bosco S. Bunyi, Adam C. Frank, Angel Hwang, Ruishan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08584)  

**Abstract**: Large language models (LLMs) are increasingly proposed for use in mental health support, yet their behavior in realistic counseling scenarios remains largely untested. We introduce CounselBench, a large-scale benchmark developed with 100 mental health professionals to evaluate and stress-test LLMs in single-turn counseling. The first component, CounselBench-EVAL, contains 2,000 expert evaluations of responses from GPT-4, LLaMA 3, Gemini, and online human therapists to real patient questions. Each response is rated along six clinically grounded dimensions, with written rationales and span-level annotations. We find that LLMs often outperform online human therapists in perceived quality, but experts frequently flag their outputs for safety concerns such as unauthorized medical advice. Follow-up experiments show that LLM judges consistently overrate model responses and overlook safety issues identified by human experts. To probe failure modes more directly, we construct CounselBench-Adv, an adversarial dataset of 120 expert-authored counseling questions designed to trigger specific model issues. Evaluation across 2,880 responses from eight LLMs reveals consistent, model-specific failure patterns. Together, CounselBench establishes a clinically grounded framework for benchmarking and improving LLM behavior in high-stakes mental health settings. 

---
# Neighbors and relatives: How do speech embeddings reflect linguistic connections across the world? 

**Authors**: Tuukka Törö, Antti Suni, Juraj Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2506.08564)  

**Abstract**: Investigating linguistic relationships on a global scale requires analyzing diverse features such as syntax, phonology and prosody, which evolve at varying rates influenced by internal diversification, language contact, and sociolinguistic factors. Recent advances in machine learning (ML) offer complementary alternatives to traditional historical and typological approaches. Instead of relying on expert labor in analyzing specific linguistic features, these new methods enable the exploration of linguistic variation through embeddings derived directly from speech, opening new avenues for large-scale, data-driven analyses.
This study employs embeddings from the fine-tuned XLS-R self-supervised language identification model voxlingua107-xls-r-300m-wav2vec, to analyze relationships between 106 world languages based on speech recordings. Using linear discriminant analysis (LDA), language embeddings are clustered and compared with genealogical, lexical, and geographical distances. The results demonstrate that embedding-based distances align closely with traditional measures, effectively capturing both global and local typological patterns. Challenges in visualizing relationships, particularly with hierarchical clustering and network-based methods, highlight the dynamic nature of language change.
The findings show potential for scalable analyses of language variation based on speech embeddings, providing new perspectives on relationships among languages. By addressing methodological considerations such as corpus size and latent space dimensionality, this approach opens avenues for studying low-resource languages and bridging macro- and micro-level linguistic variation. Future work aims to extend these methods to underrepresented languages and integrate sociolinguistic variation for a more comprehensive understanding of linguistic diversity. 

---
# Efficient Post-Training Refinement of Latent Reasoning in Large Language Models 

**Authors**: Xinyuan Wang, Dongjie Wang, Wangyang Ying, Haoyue Bai, Nanxu Gong, Sixun Dong, Kunpeng Liu, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08552)  

**Abstract**: Reasoning is a key component of language understanding in Large Language Models. While Chain-of-Thought prompting enhances performance via explicit intermediate steps, it suffers from sufficient token overhead and a fixed reasoning trajectory, preventing step-wise refinement. Recent advances in latent reasoning address these limitations by refining internal reasoning processes directly in the model's latent space, without producing explicit outputs. However, a key challenge remains: how to effectively update reasoning embeddings during post-training to guide the model toward more accurate solutions. To overcome this challenge, we propose a lightweight post-training framework that refines latent reasoning trajectories using two novel strategies: 1) Contrastive reasoning feedback, which compares reasoning embeddings against strong and weak baselines to infer effective update directions via embedding enhancement; 2) Residual embedding refinement, which stabilizes updates by progressively integrating current and historical gradients, enabling fast yet controlled convergence. Extensive experiments and case studies are conducted on five reasoning benchmarks to demonstrate the effectiveness of the proposed framework. Notably, a 5\% accuracy gain on MathQA without additional training. 

---
# CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations 

**Authors**: Divyaksh Shukla, Ritesh Baviskar, Dwijesh Gohil, Aniket Tiwari, Atul Shree, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08504)  

**Abstract**: Discourse parsing is an important task useful for NLU applications such as summarization, machine comprehension, and emotion recognition. The current discourse parsing datasets based on conversations consists of written English dialogues restricted to a single domain. In this resource paper, we introduce CoMuMDR: Code-mixed Multi-modal Multi-domain corpus for Discourse paRsing in conversations. The corpus (code-mixed in Hindi and English) has both audio and transcribed text and is annotated with nine discourse relations. We experiment with various SoTA baseline models; the poor performance of SoTA models highlights the challenges of multi-domain code-mixed corpus, pointing towards the need for developing better models for such realistic settings. 

---
# DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs 

**Authors**: Arie Cattan, Alon Jacovi, Ori Ram, Jonathan Herzig, Roee Aharoni, Sasha Goldshtein, Eran Ofek, Idan Szpektor, Avi Caciularu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08500)  

**Abstract**: Retrieval Augmented Generation (RAG) is a commonly used approach for enhancing large language models (LLMs) with relevant and up-to-date information. However, the retrieved sources can often contain conflicting information and it remains unclear how models should address such discrepancies. In this work, we first propose a novel taxonomy of knowledge conflict types in RAG, along with the desired model behavior for each type. We then introduce CONFLICTS, a high-quality benchmark with expert annotations of conflict types in a realistic RAG setting. CONFLICTS is the first benchmark that enables tracking progress on how models address a wide range of knowledge conflicts. We conduct extensive experiments on this benchmark, showing that LLMs often struggle to appropriately resolve conflicts between sources. While prompting LLMs to explicitly reason about the potential conflict in the retrieved documents significantly improves the quality and appropriateness of their responses, substantial room for improvement in future research remains. 

---
# Integration of Old and New Knowledge for Generalized Intent Discovery: A Consistency-driven Prototype-Prompting Framework 

**Authors**: Xiao Wei, Xiaobao Wang, Ning Zhuang, Chenyang Wang, Longbiao Wang, Jianwu dang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08490)  

**Abstract**: Intent detection aims to identify user intents from natural language inputs, where supervised methods rely heavily on labeled in-domain (IND) data and struggle with out-of-domain (OOD) intents, limiting their practical applicability. Generalized Intent Discovery (GID) addresses this by leveraging unlabeled OOD data to discover new intents without additional annotation. However, existing methods focus solely on clustering unsupervised data while neglecting domain adaptation. Therefore, we propose a consistency-driven prototype-prompting framework for GID from the perspective of integrating old and new knowledge, which includes a prototype-prompting framework for transferring old knowledge from external sources, and a hierarchical consistency constraint for learning new knowledge from target domains. We conducted extensive experiments and the results show that our method significantly outperforms all baseline methods, achieving state-of-the-art results, which strongly demonstrates the effectiveness and generalization of our methods. Our source code is publicly available at this https URL. 

---
# EtiCor++: Towards Understanding Etiquettical Bias in LLMs 

**Authors**: Ashutosh Dwivedi, Siddhant Shivdutt Singh, Ashutosh Modi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08488)  

**Abstract**: In recent years, researchers have started analyzing the cultural sensitivity of LLMs. In this respect, Etiquettes have been an active area of research. Etiquettes are region-specific and are an essential part of the culture of a region; hence, it is imperative to make LLMs sensitive to etiquettes. However, there needs to be more resources in evaluating LLMs for their understanding and bias with regard to etiquettes. In this resource paper, we introduce EtiCor++, a corpus of etiquettes worldwide. We introduce different tasks for evaluating LLMs for knowledge about etiquettes across various regions. Further, we introduce various metrics for measuring bias in LLMs. Extensive experimentation with LLMs shows inherent bias towards certain regions. 

---
# Fairness is Not Silence: Unmasking Vacuous Neutrality in Small Language Models 

**Authors**: Sumanth Manduru, Carlotta Domeniconi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08487)  

**Abstract**: The rapid adoption of Small Language Models (SLMs) for on-device and resource-constrained deployments has outpaced our understanding of their ethical risks. To the best of our knowledge, we present the first large-scale audit of instruction-tuned SLMs spanning 0.5 to 5 billion parameters-an overlooked "middle tier" between BERT-class encoders and flagship LLMs. Our evaluation includes nine open-source models from the Qwen 2.5, LLaMA 3.2, Gemma 3, and Phi families. Using the BBQ benchmark under zero-shot prompting, we analyze both utility and fairness across ambiguous and disambiguated contexts. This evaluation reveals three key insights. First, competence and fairness need not be antagonistic: Phi models achieve F1 scores exceeding 90 percent while exhibiting minimal bias, showing that efficient and ethical NLP is attainable. Second, social bias varies significantly by architecture: Qwen 2.5 models may appear fair, but this often reflects vacuous neutrality, random guessing, or evasive behavior rather than genuine ethical alignment. In contrast, LLaMA 3.2 models exhibit stronger stereotypical bias, suggesting overconfidence rather than neutrality. Third, compression introduces nuanced trade-offs: 4-bit AWQ quantization improves F1 scores in ambiguous settings for LLaMA 3.2-3B but increases disability-related bias in Phi-4-Mini by over 7 percentage points. These insights provide practical guidance for the responsible deployment of SLMs in applications demanding fairness and efficiency, particularly benefiting small enterprises and resource-constrained environments. 

---
# Re-Thinking the Automatic Evaluation of Image-Text Alignment in Text-to-Image Models 

**Authors**: Huixuan Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08480)  

**Abstract**: Text-to-image models often struggle to generate images that precisely match textual prompts. Prior research has extensively studied the evaluation of image-text alignment in text-to-image generation. However, existing evaluations primarily focus on agreement with human assessments, neglecting other critical properties of a trustworthy evaluation framework. In this work, we first identify two key aspects that a reliable evaluation should address. We then empirically demonstrate that current mainstream evaluation frameworks fail to fully satisfy these properties across a diverse range of metrics and models. Finally, we propose recommendations for improving image-text alignment evaluation. 

---
# Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-$k$ 

**Authors**: Chihiro Taguchi, Seiji Maekawa, Nikita Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08479)  

**Abstract**: Retrieval-augmented generation (RAG) and long-context language models (LCLMs) both address context limitations of LLMs in open-domain question answering (QA). However, optimal external context to retrieve remains an open problem: fixing the retrieval size risks either wasting tokens or omitting key evidence. Existing adaptive methods like Self-RAG and Self-Route rely on iterative LLM prompting and perform well on factoid QA, but struggle with aggregation QA, where the optimal context size is both unknown and variable. We present Adaptive-$k$ retrieval, a simple and effective single-pass method that adaptively selects the number of passages based on the distribution of the similarity scores between the query and the candidate passages. It does not require model fine-tuning, extra LLM inferences or changes to existing retriever-reader pipelines. On both factoid and aggregation QA benchmarks, Adaptive-$k$ matches or outperforms fixed-$k$ baselines while using up to 10x fewer tokens than full-context input, yet still retrieves 70% of relevant passages. It improves accuracy across five LCLMs and two embedding models, highlighting that dynamically adjusting context size leads to more efficient and accurate QA. 

---
# Detecting Harmful Memes with Decoupled Understanding and Guided CoT Reasoning 

**Authors**: Fengjun Pan, Anh Tuan Luu, Xiaobao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08477)  

**Abstract**: Detecting harmful memes is essential for maintaining the integrity of online environments. However, current approaches often struggle with resource efficiency, flexibility, or explainability, limiting their practical deployment in content moderation systems. To address these challenges, we introduce U-CoT+, a novel framework for harmful meme detection. Instead of relying solely on prompting or fine-tuning multimodal models, we first develop a high-fidelity meme-to-text pipeline that converts visual memes into detail-preserving textual descriptions. This design decouples meme interpretation from meme classification, thus avoiding immediate reasoning over complex raw visual content and enabling resource-efficient harmful meme detection with general large language models (LLMs). Building on these textual descriptions, we further incorporate targeted, interpretable human-crafted guidelines to guide models' reasoning under zero-shot CoT prompting. As such, this framework allows for easy adaptation to different harmfulness detection criteria across platforms, regions, and over time, offering high flexibility and explainability. Extensive experiments on seven benchmark datasets validate the effectiveness of our framework, highlighting its potential for explainable and low-resource harmful meme detection using small-scale LLMs. Codes and data are available at: this https URL. 

---
# Olica: Efficient Structured Pruning of Large Language Models without Retraining 

**Authors**: Jiujun He, Huazhen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.08436)  

**Abstract**: Most existing structured pruning methods for Large Language Models (LLMs) require substantial computational and data resources for retraining to reestablish the corrupted correlations, making them prohibitively expensive. To address this, we propose a pruning framework for LLMs called Orthogonal decomposition and Linear Calibration (Olica), which eliminates the need for retraining. A key observation is that the multi-head attention (MHA) layer depends on two types of matrix products. By treating these matrix products as unified entities and applying principal component analysis (PCA), we extract the most important information to compress LLMs without sacrificing accuracy or disrupting their original structure. Consequently, retraining becomes unnecessary. A fast decomposition method is devised, reducing the complexity of PCA by a factor of the square of the number of attention heads. Additionally, to mitigate error accumulation problem caused by pruning the feed-forward network (FFN) layer, we introduce a linear calibration method to reconstruct the residual errors of pruned layers using low-rank matrices. By leveraging singular value decomposition (SVD) on the solution of the least-squares problem, these matrices are obtained without requiring retraining. Extensive experiments show that the proposed Olica is efficient in terms of data usage, GPU memory, and running time, while delivering superior performance across multiple benchmarks. 

---
# Low-resource domain adaptation while minimizing energy and hardware resource consumption 

**Authors**: Hernán Maina, Nicolás Wolovick, Luciana Benotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08433)  

**Abstract**: Training Large Language Models (LLMs) is costly in terms of energy, hardware, and annotated data, often resulting in a positionality rooted in predominant cultures and values (Santy et al., 2023). Domain adaptation has emerged as a promising strategy to better align models with diverse cultural and value contexts (Hershcovich et al., 2022), but its computational cost remains a significant barrier, particularly for research groups lacking access to large-scale infrastructure. In this paper, we evaluate how the use of different numerical precisions and data parallelization strategies impacts both training speed (as a proxy to energy and hardware consumption) and model accuracy, with the goal of facilitating domain adaptation in low-resource environments. Our findings are relevant to any setting where energy efficiency, accessibility, or limited hardware availability are key concerns. 

---
# CAF-I: A Collaborative Multi-Agent Framework for Enhanced Irony Detection with Large Language Models 

**Authors**: Ziqi.Liu, Ziyang.Zhou, Mingxuan.Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08430)  

**Abstract**: Large language model (LLM) have become mainstream methods in the field of sarcasm detection. However, existing LLM methods face challenges in irony detection, including: 1. single-perspective limitations, 2. insufficient comprehensive understanding, and 3. lack of interpretability. This paper introduces the Collaborative Agent Framework for Irony (CAF-I), an LLM-driven multi-agent system designed to overcome these issues. CAF-I employs specialized agents for Context, Semantics, and Rhetoric, which perform multidimensional analysis and engage in interactive collaborative optimization. A Decision Agent then consolidates these perspectives, with a Refinement Evaluator Agent providing conditional feedback for optimization. Experiments on benchmark datasets establish CAF-I's state-of-the-art zero-shot performance. Achieving SOTA on the vast majority of metrics, CAF-I reaches an average Macro-F1 of 76.31, a 4.98 absolute improvement over the strongest prior baseline. This success is attained by its effective simulation of human-like multi-perspective analysis, enhancing detection accuracy and interpretability. 

---
# Know-MRI: A Knowledge Mechanisms Revealer&Interpreter for Large Language Models 

**Authors**: Jiaxiang Liu, Boxuan Xing, Chenhao Yuan, Chenxiang Zhang, Di Wu, Xiusheng Huang, Haida Yu, Chuhan Lang, Pengfei Cao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08427)  

**Abstract**: As large language models (LLMs) continue to advance, there is a growing urgency to enhance the interpretability of their internal knowledge mechanisms. Consequently, many interpretation methods have emerged, aiming to unravel the knowledge mechanisms of LLMs from various perspectives. However, current interpretation methods differ in input data formats and interpreting outputs. The tools integrating these methods are only capable of supporting tasks with specific inputs, significantly constraining their practical applications. To address these challenges, we present an open-source Knowledge Mechanisms Revealer&Interpreter (Know-MRI) designed to analyze the knowledge mechanisms within LLMs systematically. Specifically, we have developed an extensible core module that can automatically match different input data with interpretation methods and consolidate the interpreting outputs. It enables users to freely choose appropriate interpretation methods based on the inputs, making it easier to comprehensively diagnose the model's internal knowledge mechanisms from multiple perspectives. Our code is available at this https URL. We also provide a demonstration video on this https URL. 

---
# Large Language Models Have Intrinsic Meta-Cognition, but Need a Good Lens 

**Authors**: Ziyang Ma, Qingyue Yuan, Zhenglin Wang, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08410)  

**Abstract**: Previous research has primarily focused on the cognitive error detection capabilities of Large Language Models (LLMs), often prompting them to analyze mistakes in reasoning chains. However, few studies have examined the meta-cognitive abilities of LLMs (e.g., their self-awareness of step errors), which are crucial for their reliability. While studies on LLM self-evaluation present some measures, such as perplexity, which can reflect the answer correctness and be viewed as the lens of meta-cognition, they lack step-level analysis and adaptation. This paper studies the evaluation of LLM meta-cognition using the current lenses and how to improve these lenses. Specifically, we propose AutoMeco, an Automated Meta-cognition Evaluation framework for benchmarking the existing lenses. Furthermore, a training-free Markovian Intrinsic Reward Adjustment strategy, MIRA, is proposed to boost current meta-cognition lenses. Experimental results on three mathematical reasoning datasets and three LLMs show the reasonableness of AutoMeco by comparing it with Best-of-N verification. Moreover, the meta-cognition ability of LLMs can be better evaluated using MIRA. 

---
# TACTIC: Translation Agents with Cognitive-Theoretic Interactive Collaboration 

**Authors**: Weiya Li, Junjie Chen, Bei Li, Boyang Liu, Zichen Wen, Nuanqiao Shan, Xiaoqian Liu, Anping Liu, Huajie Liu, Youyan Wang, Wujiuge Yin, Hu Song, Bing Huang, Zhiyuan Xia, Jialiang Chen, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08403)  

**Abstract**: Machine translation has long been a central task in natural language processing. With the rapid advancement of large language models (LLMs), there has been remarkable progress in translation quality. However, fully realizing the translation potential of LLMs remains an open challenge. Recent studies have explored multi-agent systems to decompose complex translation tasks into collaborative subtasks, showing initial promise in enhancing translation quality through agent cooperation and specialization. Nevertheless, existing multi-agent translation frameworks largely neglect foundational insights from cognitive translation studies. These insights emphasize how human translators employ different cognitive strategies, such as balancing literal and free translation, refining expressions based on context, and iteratively evaluating outputs. To address this limitation, we propose a cognitively informed multi-agent framework called TACTIC, which stands for T ranslation A gents with Cognitive- T heoretic Interactive Collaboration. The framework comprises six functionally distinct agents that mirror key cognitive processes observed in human translation behavior. These include agents for drafting, refinement, evaluation, scoring, context reasoning, and external knowledge gathering. By simulating an interactive and theory-grounded translation workflow, TACTIC effectively leverages the full capacity of LLMs for high-quality translation. Experimental results on diverse language pairs from the FLORES-200 and WMT24 benchmarks show that our method consistently achieves state-of-the-art performance. Using DeepSeek-V3 as the base model, TACTIC surpasses GPT-4.1 by an average of +0.6 XCOMET and +1.18 COMETKIWI-23. Compared to DeepSeek-R1, it further improves by +0.84 XCOMET and +2.99 COMETKIWI-23. Code is available at this https URL. 

---
# mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks 

**Authors**: Luel Hagos Beyene, Vivek Verma, Min Ma, Jesujoba O. Alabi, Fabian David Schmidt, Joyce Nakatumba-Nabende, David Ifeoluwa Adelani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08400)  

**Abstract**: Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage. 

---
# EIFBENCH: Extremely Complex Instruction Following Benchmark for Large Language Models 

**Authors**: Tao Zou, Xinghua Zhang, Haiyang Yu, Minzheng Wang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.08375)  

**Abstract**: With the development and widespread application of large language models (LLMs), the new paradigm of "Model as Product" is rapidly evolving, and demands higher capabilities to address complex user needs, often requiring precise workflow execution which involves the accurate understanding of multiple tasks. However, existing benchmarks focusing on single-task environments with limited constraints lack the complexity required to fully reflect real-world scenarios. To bridge this gap, we present the Extremely Complex Instruction Following Benchmark (EIFBENCH), meticulously crafted to facilitate a more realistic and robust evaluation of LLMs. EIFBENCH not only includes multi-task scenarios that enable comprehensive assessment across diverse task types concurrently, but also integrates a variety of constraints, replicating complex operational environments. Furthermore, we propose the Segment Policy Optimization (SegPO) algorithm to enhance the LLM's ability to accurately fulfill multi-task workflow. Evaluations on EIFBENCH have unveiled considerable performance discrepancies in existing LLMs when challenged with these extremely complex instructions. This finding underscores the necessity for ongoing optimization to navigate the intricate challenges posed by LLM applications. 

---
# Draft-based Approximate Inference for LLMs 

**Authors**: Kevin Galim, Ethan Ewer, Wonjun Kang, Minjae Lee, Hyung Il Koo, Kangwook Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.08373)  

**Abstract**: Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: (i) SpecKV, which leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping, and (ii) SpecPC, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. To the best of our knowledge, this is the first work to use draft models for approximate LLM inference acceleration, extending their utility beyond traditional lossless speculative decoding. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput. Our code is available at this https URL. 

---
# Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding 

**Authors**: Zikai Xiao, Ziyang Wang, Wen Ma, Yan Zhang, Wei Shen, Yan Wang, Luqi Gong, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08371)  

**Abstract**: While Large Language Models (LLMs) support long contexts, they struggle with performance degradation within the context window. Current solutions incur prohibitive training costs, leaving statistical behaviors and cost-effective approaches underexplored. From the decoding perspective, we identify the Posterior Salience Attenuation (PSA) phenomenon, where the salience ratio correlates with long-text performance degradation. Notably, despite the attenuation, gold tokens still occupy high-ranking positions in the decoding space. Motivated by it, we propose the training-free Positional Contrastive Decoding (PCD) that contrasts the logits derived from long-aware attention with those from designed local-aware attention, enabling the model to focus on the gains introduced by large-scale short-to-long training. Through the analysis of long-term decay simulation, we demonstrate that PCD effectively alleviates attention score degradation. Experimental results show that PCD achieves state-of-the-art performance on long-context benchmarks. 

---
# CC-RAG: Structured Multi-Hop Reasoning via Theme-Based Causal Graphs 

**Authors**: Jash Rajesh Parekh, Pengcheng Jiang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08364)  

**Abstract**: Understanding cause and effect relationships remains a formidable challenge for Large Language Models (LLMs), particularly in specialized domains where reasoning requires more than surface-level correlations. Retrieval-Augmented Generation (RAG) improves factual accuracy, but standard RAG pipelines treat evidence as flat context, lacking the structure required to model true causal dependencies. We introduce Causal-Chain RAG (CC-RAG), a novel approach that integrates zero-shot triple extraction and theme-aware graph chaining into the RAG pipeline, enabling structured multi-hop inference. Given a domain specific corpus, CC-RAG constructs a Directed Acyclic Graph (DAG) of <cause, relation, effect> triples and uses forward/backward chaining to guide structured answer generation. Experiments on two real-world domains: Bitcoin price fluctuations and Gaucher disease, show that CC-RAG outperforms standard RAG and zero-shot LLMs in chain similarity, information density, and lexical diversity. Both LLM-as-a-Judge and human evaluations consistently favor CC-RAG. Our results demonstrate that explicitly modeling causal structure enables LLMs to generate more accurate and interpretable responses, especially in specialized domains where flat retrieval fails. 

---
# DEAL: Disentangling Transformer Head Activations for LLM Steering 

**Authors**: Li-Ming Zhan, Bo Liu, Zexin Lu, Chengqiang Xie, Jiannong Cao, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08359)  

**Abstract**: Inference-time steering aims to alter the response characteristics of large language models (LLMs) without modifying their underlying parameters. A critical step in this process is the identification of internal modules within LLMs that are associated with the target behavior. However, current approaches to module selection often depend on superficial cues or ad-hoc heuristics, which can result in suboptimal or unintended outcomes. In this work, we propose a principled causal-attribution framework for identifying behavior-relevant attention heads in transformers. For each head, we train a vector-quantized autoencoder (VQ-AE) on its attention activations, partitioning the latent space into behavior-relevant and behavior-irrelevant subspaces, each quantized with a shared learnable codebook. We assess the behavioral relevance of each head by quantifying the separability of VQ-AE encodings for behavior-aligned versus behavior-violating responses using a binary classification metric. This yields a behavioral relevance score that reflects each head discriminative capacity with respect to the target behavior, guiding both selection and importance weighting. Experiments on seven LLMs from two model families and five behavioral steering datasets demonstrate that our method enables more accurate inference-time interventions, achieving superior performance on the truthfulness-steering task. Furthermore, the heads selected by our approach exhibit strong zero-shot generalization in cross-domain truthfulness-steering scenarios. 

---
# Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning 

**Authors**: Yiqun Sun, Qiang Huang, Anthony K. H. Tung, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08354)  

**Abstract**: This position paper argues that the text embedding research community should move beyond surface meaning and embrace implicit semantics as a central modeling goal. Text embedding models have become foundational in modern NLP, powering a wide range of applications and drawing increasing research attention. Yet, much of this progress remains narrowly focused on surface-level semantics. In contrast, linguistic theory emphasizes that meaning is often implicit, shaped by pragmatics, speaker intent, and sociocultural context. Current embedding models are typically trained on data that lacks such depth and evaluated on benchmarks that reward the capture of surface meaning. As a result, they struggle with tasks requiring interpretive reasoning, speaker stance, or social meaning. Our pilot study highlights this gap, showing that even state-of-the-art models perform only marginally better than simplistic baselines on implicit semantics tasks. To address this, we call for a paradigm shift: embedding research should prioritize more diverse and linguistically grounded training data, design benchmarks that evaluate deeper semantic understanding, and explicitly frame implicit meaning as a core modeling objective, better aligning embeddings with real-world language complexity. 

---
# Evaluating LLMs Across Multi-Cognitive Levels: From Medical Knowledge Mastery to Scenario-Based Problem Solving 

**Authors**: Yuxuan Zhou, Xien Liu, Chenwei Yan, Chen Ning, Xiao Zhang, Boxun Li, Xiangling Fu, Shijin Wang, Guoping Hu, Yu Wang, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08349)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance on various medical benchmarks, but their capabilities across different cognitive levels remain underexplored. Inspired by Bloom's Taxonomy, we propose a multi-cognitive-level evaluation framework for assessing LLMs in the medical domain in this study. The framework integrates existing medical datasets and introduces tasks targeting three cognitive levels: preliminary knowledge grasp, comprehensive knowledge application, and scenario-based problem solving. Using this framework, we systematically evaluate state-of-the-art general and medical LLMs from six prominent families: Llama, Qwen, Gemma, Phi, GPT, and DeepSeek. Our findings reveal a significant performance decline as cognitive complexity increases across evaluated models, with model size playing a more critical role in performance at higher cognitive levels. Our study highlights the need to enhance LLMs' medical capabilities at higher cognitive levels and provides insights for developing LLMs suited to real-world medical applications. 

---
# Wait, We Don't Need to "Wait"! Removing Thinking Tokens Improves Reasoning Efficiency 

**Authors**: Chenlong Wang, Yuanning Feng, Dongping Chen, Zhaoyang Chu, Ranjay Krishna, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08343)  

**Abstract**: Recent advances in large reasoning models have enabled complex, step-by-step reasoning but often introduce significant overthinking, resulting in verbose and redundant outputs that hinder efficiency. In this study, we examine whether explicit self-reflection, signaled by tokens such as "Wait" and "Hmm", is necessary for advanced reasoning. We propose NoWait, a simple yet effective approach that disables explicit self-reflection by suppressing these tokens during inference. Extensive experiments on ten benchmarks across textual, visual, and video reasoning tasks show that NoWait reduces chain-of-thought trajectory length by up to 27%-51% in five R1-style model series, without compromising model utility. NoWait thus offers a plug-and-play solution for efficient and utility-preserving multimodal reasoning. 

---
# Institutional Books 1.0: A 242B token dataset from Harvard Library's collections, refined for accuracy and usability 

**Authors**: Matteo Cargnelutti, Catherine Brobston, John Hess, Jack Cushman, Kristi Mukk, Aristana Scourtas, Kyle Courtney, Greg Leppert, Amanda Watson, Martha Whitehead, Jonathan Zittrain  

**Link**: [PDF](https://arxiv.org/pdf/2506.08300)  

**Abstract**: Large language models (LLMs) use data to learn about the world in order to produce meaningful correlations and predictions. As such, the nature, scale, quality, and diversity of the datasets used to train these models, or to support their work at inference time, have a direct impact on their quality. The rapid development and adoption of LLMs of varying quality has brought into focus the scarcity of publicly available, high-quality training data and revealed an urgent need to ground the stewardship of these datasets in sustainable practices with clear provenance chains. To that end, this technical report introduces Institutional Books 1.0, a large collection of public domain books originally digitized through Harvard Library's participation in the Google Books project, beginning in 2006. Working with Harvard Library, we extracted, analyzed, and processed these volumes into an extensively-documented dataset of historic texts. This analysis covers the entirety of Harvard Library's collection scanned as part of that project, originally spanning 1,075,899 volumes written in over 250 different languages for a total of approximately 250 billion tokens. As part of this initial release, the OCR-extracted text (original and post-processed) as well as the metadata (bibliographic, source, and generated) of the 983,004 volumes, or 242B tokens, identified as being in the public domain have been made available. This report describes this project's goals and methods as well as the results of the analyses we performed, all in service of making this historical collection more accessible and easier for humans and machines alike to filter, read and use. 

---
# Automatic Generation of Inference Making Questions for Reading Comprehension Assessments 

**Authors**: Wanjing Anya Ma, Michael Flor, Zuowei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08260)  

**Abstract**: Inference making is an essential but complex skill in reading comprehension (RC). Some inferences require resolving references across sentences, and some rely on using prior knowledge to fill in the detail that is not explicitly written in the text. Diagnostic RC questions can help educators provide more effective and targeted reading instruction and interventions for school-age students. We introduce a taxonomy of inference types for RC and use it to analyze the distribution of items within a diagnostic RC item bank. Next, we present experiments using GPT-4o to generate bridging-inference RC items for given reading passages via few-shot prompting, comparing conditions with and without chain-of-thought prompts. Generated items were evaluated on three aspects: overall item quality, appropriate inference type, and LLM reasoning, achieving high inter-rater agreements above 0.90. Our results show that GPT-4o produced 93.8% good-quality questions suitable for operational use in grade 3-12 contexts; however, only 42.6% of the generated questions accurately matched the targeted inference type. We conclude that combining automatic item generation with human judgment offers a promising path toward scalable, high-quality diagnostic RC assessments. 

---
# Can AI Validate Science? Benchmarking LLMs for Accurate Scientific Claim $\rightarrow$ Evidence Reasoning 

**Authors**: Shashidhar Reddy Javaji, Yupeng Cao, Haohang Li, Yangyang Yu, Nikhil Muralidhar, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08235)  

**Abstract**: Large language models (LLMs) are increasingly being used for complex research tasks such as literature review, idea generation, and scientific paper analysis, yet their ability to truly understand and process the intricate relationships within complex research papers, such as the logical links between claims and supporting evidence remains largely unexplored. In this study, we present CLAIM-BENCH, a comprehensive benchmark for evaluating LLMs' capabilities in scientific claim-evidence extraction and validation, a task that reflects deeper comprehension of scientific argumentation. We systematically compare three approaches which are inspired by divide and conquer approaches, across six diverse LLMs, highlighting model-specific strengths and weaknesses in scientific comprehension. Through evaluation involving over 300 claim-evidence pairs across multiple research domains, we reveal significant limitations in LLMs' ability to process complex scientific content. Our results demonstrate that closed-source models like GPT-4 and Claude consistently outperform open-source counterparts in precision and recall across claim-evidence identification tasks. Furthermore, strategically designed three-pass and one-by-one prompting approaches significantly improve LLMs' abilities to accurately link dispersed evidence with claims, although this comes at increased computational cost. CLAIM-BENCH sets a new standard for evaluating scientific comprehension in LLMs, offering both a diagnostic tool and a path forward for building systems capable of deeper, more reliable reasoning across full-length papers. 

---
# Compound AI Systems Optimization: A Survey of Methods, Challenges, and Future Directions 

**Authors**: Yu-Ang Lee, Guan-Ting Yi, Mei-Yi Liu, Jui-Chao Lu, Guan-Bo Yang, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08234)  

**Abstract**: Recent advancements in large language models (LLMs) and AI systems have led to a paradigm shift in the design and optimization of complex AI workflows. By integrating multiple components, compound AI systems have become increasingly adept at performing sophisticated tasks. However, as these systems grow in complexity, new challenges arise in optimizing not only individual components but also their interactions. While traditional optimization methods such as supervised fine-tuning (SFT) and reinforcement learning (RL) remain foundational, the rise of natural language feedback introduces promising new approaches, especially for optimizing non-differentiable systems. This paper provides a systematic review of recent progress in optimizing compound AI systems, encompassing both numerical and language-based techniques. We formalize the notion of compound AI system optimization, classify existing methods along several key dimensions, and highlight open research challenges and future directions in this rapidly evolving field. A list of surveyed papers is publicly available at this https URL. 

---
# "I Wrote, I Paused, I Rewrote" Teaching LLMs to Read Between the Lines of Student Writing 

**Authors**: Samra Zafar, Shaheer Minhas, Syed Ali Hassan Zaidi, Arfa Naeem, Zahra Ali  

**Link**: [PDF](https://arxiv.org/pdf/2506.08221)  

**Abstract**: Large language models(LLMs) like Gemini are becoming common tools for supporting student writing. But most of their feedback is based only on the final essay missing important context about how that text was written. In this paper, we explore whether using writing process data, collected through keystroke logging and periodic snapshots, can help LLMs give feedback that better reflects how learners think and revise while writing. We built a digital writing tool that captures both what students type and how their essays evolve over time. Twenty students used this tool to write timed essays, which were then evaluated in two ways: (i) LLM generated feedback using both the final essay and the full writing trace, and (ii) After the task, students completed surveys about how useful and relatable they found the feedback. Early results show that learners preferred the process-aware LLM feedback, finding it more in tune with their own thinking. We also found that certain types of edits, like adding new content or reorganizing paragraphs, aligned closely with higher scores in areas like coherence and elaboration. Our findings suggest that making LLMs more aware of the writing process can lead to feedback that feels more meaningful, personal, and supportive. 

---
# Unable to forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length 

**Authors**: Chupei Wang, Jiaqiu Vince Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08184)  

**Abstract**: Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval. 

---
# LLM-BT: Back-Translation as a Framework for Terminology Standardization and Dynamic Semantic Embedding 

**Authors**: Li Weigang, Pedro Carvalho Brom  

**Link**: [PDF](https://arxiv.org/pdf/2506.08174)  

**Abstract**: The rapid growth of English technical terms challenges traditional expert-driven standardization, especially in fast-evolving fields like AI and quantum computing. Manual methods struggle to ensure multilingual consistency. We propose \textbf{LLM-BT}, a back-translation framework powered by large language models (LLMs) to automate terminology verification and standardization via cross-lingual semantic alignment. Our contributions are: \textbf{(1) Term-Level Consistency Validation:} Using English $\rightarrow$ intermediate language $\rightarrow$ English back-translation, LLM-BT achieves high term consistency across models (e.g., GPT-4, DeepSeek, Grok), with case studies showing over 90\% exact or semantic matches. \textbf{(2) Multi-Path Verification Workflow:} A novel ``Retrieve--Generate--Verify--Optimize'' pipeline integrates serial (e.g., EN $\rightarrow$ ZHcn $\rightarrow$ ZHtw $\rightarrow$ EN) and parallel (e.g., EN $\rightarrow$ Chinese/Portuguese $\rightarrow$ EN) BT routes. BLEU and term accuracy indicate strong cross-lingual robustness (BLEU $>$ 0.45; Portuguese accuracy 100\%). \textbf{(3) Back-Translation as Semantic Embedding:} BT is conceptualized as dynamic semantic embedding, revealing latent meaning trajectories. Unlike static embeddings, LLM-BT provides transparent path-based embeddings shaped by model evolution. LLM-BT transforms back-translation into an active engine for multilingual terminology standardization, enabling human--AI collaboration: machines ensure semantic fidelity, humans guide cultural interpretation. This infrastructure supports terminology governance across scientific and technological fields worldwide. 

---
# Can Artificial Intelligence Write Like Borges? An Evaluation Protocol for Spanish Microfiction 

**Authors**: Gerardo Aleman Manzanarez, Nora de la Cruz Arana, Jorge Garcia Flores, Yobany Garcia Medina, Raul Monroy, Nathalie Pernelle  

**Link**: [PDF](https://arxiv.org/pdf/2506.08172)  

**Abstract**: Automated story writing has been a subject of study for over 60 years. Large language models can generate narratively consistent and linguistically coherent short fiction texts. Despite these advancements, rigorous assessment of such outputs for literary merit - especially concerning aesthetic qualities - has received scant attention. In this paper, we address the challenge of evaluating AI-generated microfictions and argue that this task requires consideration of literary criteria across various aspects of the text, such as thematic coherence, textual clarity, interpretive depth, and aesthetic quality. To facilitate this, we present GrAImes: an evaluation protocol grounded in literary theory, specifically drawing from a literary perspective, to offer an objective framework for assessing AI-generated microfiction. Furthermore, we report the results of our validation of the evaluation protocol, as answered by both literature experts and literary enthusiasts. This protocol will serve as a foundation for evaluating automatically generated microfictions and assessing their literary value. 

---
# ETT-CKGE: Efficient Task-driven Tokens for Continual Knowledge Graph Embedding 

**Authors**: Lijing Zhu, Qizhen Lan, Qing Tian, Wenbo Sun, Li Yang, Lu Xia, Yixin Xie, Xi Xiao, Tiehang Duan, Cui Tao, Shuteng Niu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08158)  

**Abstract**: Continual Knowledge Graph Embedding (CKGE) seeks to integrate new knowledge while preserving past information. However, existing methods struggle with efficiency and scalability due to two key limitations: (1) suboptimal knowledge preservation between snapshots caused by manually designed node/relation importance scores that ignore graph dependencies relevant to the downstream task, and (2) computationally expensive graph traversal for node/relation importance calculation, leading to slow training and high memory overhead. To address these limitations, we introduce ETT-CKGE (Efficient, Task-driven, Tokens for Continual Knowledge Graph Embedding), a novel task-guided CKGE method that leverages efficient task-driven tokens for efficient and effective knowledge transfer between snapshots. Our method introduces a set of learnable tokens that directly capture task-relevant signals, eliminating the need for explicit node scoring or traversal. These tokens serve as consistent and reusable guidance across snapshots, enabling efficient token-masked embedding alignment between snapshots. Importantly, knowledge transfer is achieved through simple matrix operations, significantly reducing training time and memory usage. Extensive experiments across six benchmark datasets demonstrate that ETT-CKGE consistently achieves superior or competitive predictive performance, while substantially improving training efficiency and scalability compared to state-of-the-art CKGE methods. The code is available at: this https URL 

---
# Multilingual Hate Speech Detection in Social Media Using Translation-Based Approaches with Large Language Models 

**Authors**: Muhammad Usman, Muhammad Ahmad, M. Shahiki Tash, Irina Gelbukh, Rolando Quintero Tellez, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2506.08147)  

**Abstract**: Social media platforms are critical spaces for public discourse, shaping opinions and community dynamics, yet their widespread use has amplified harmful content, particularly hate speech, threatening online safety and inclusivity. While hate speech detection has been extensively studied in languages like English and Spanish, Urdu remains underexplored, especially using translation-based approaches. To address this gap, we introduce a trilingual dataset of 10,193 tweets in English (3,834 samples), Urdu (3,197 samples), and Spanish (3,162 samples), collected via keyword filtering, with a balanced distribution of 4,849 Hateful and 5,344 Not-Hateful labels. Our methodology leverages attention layers as a precursor to transformer-based models and large language models (LLMs), enhancing feature extraction for multilingual hate speech detection. For non-transformer models, we use TF-IDF for feature extraction. The dataset is benchmarked using state-of-the-art models, including GPT-3.5 Turbo and Qwen 2.5 72B, alongside traditional machine learning models like SVM and other transformers (e.g., BERT, RoBERTa). Three annotators, following rigorous guidelines, ensured high dataset quality, achieving a Fleiss' Kappa of 0.821. Our approach, integrating attention layers with GPT-3.5 Turbo and Qwen 2.5 72B, achieves strong performance, with macro F1 scores of 0.87 for English (GPT-3.5 Turbo), 0.85 for Spanish (GPT-3.5 Turbo), 0.81 for Urdu (Qwen 2.5 72B), and 0.88 for the joint multilingual model (Qwen 2.5 72B). These results reflect improvements of 8.75% in English (over SVM baseline 0.80), 8.97% in Spanish (over SVM baseline 0.78), 5.19% in Urdu (over SVM baseline 0.77), and 7.32% in the joint multilingual model (over SVM baseline 0.82). Our framework offers a robust solution for multilingual hate speech detection, fostering safer digital communities worldwide. 

---
# EconWebArena: Benchmarking Autonomous Agents on Economic Tasks in Realistic Web Environments 

**Authors**: Zefang Liu, Yinzhu Quan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08136)  

**Abstract**: We introduce EconWebArena, a benchmark for evaluating autonomous agents on complex, multimodal economic tasks in realistic web environments. The benchmark comprises 360 curated tasks from 82 authoritative websites spanning domains such as macroeconomics, labor, finance, trade, and public policy. Each task challenges agents to navigate live websites, interpret structured and visual content, interact with real interfaces, and extract precise, time-sensitive data through multi-step workflows. We construct the benchmark by prompting multiple large language models (LLMs) to generate candidate tasks, followed by rigorous human curation to ensure clarity, feasibility, and source reliability. Unlike prior work, EconWebArena emphasizes fidelity to authoritative data sources and the need for grounded web-based economic reasoning. We evaluate a diverse set of state-of-the-art multimodal LLMs as web agents, analyze failure cases, and conduct ablation studies to assess the impact of visual grounding, plan-based reasoning, and interaction design. Our results reveal substantial performance gaps and highlight persistent challenges in grounding, navigation, and multimodal understanding, positioning EconWebArena as a rigorous testbed for economic web intelligence. 

---
# QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA 

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.08123)  

**Abstract**: Alignment of large language models with explicit principles (such as helpfulness, honesty, and harmlessness) is crucial for ensuring safe and reliable AI systems. However, standard reward-based alignment methods typically collapse diverse feedback into a single scalar reward, entangling multiple objectives into one opaque training signal, which hinders interpretability. In this work, we introduce QA-LIGN, an automatic symbolic reward decomposition approach that preserves the structure of each constitutional principle within the reward mechanism. Instead of training a black-box reward model that outputs a monolithic score, QA-LIGN formulates principle-specific evaluation questions and derives separate reward components for each principle, making it a drop-in reward model replacement. Experiments aligning an uncensored large language model with a set of constitutional principles demonstrate that QA-LIGN offers greater transparency and adaptability in the alignment process. At the same time, our approach achieves performance on par with or better than a DPO baseline. Overall, these results represent a step toward more interpretable and controllable alignment of language models, achieved without sacrificing end-task performance. 

---
# Conservative Bias in Large Language Models: Measuring Relation Predictions 

**Authors**: Toyin Aguda, Erik Wilson, Allan Anzagira, Simerjot Kaur, Charese Smiley  

**Link**: [PDF](https://arxiv.org/pdf/2506.08120)  

**Abstract**: Large language models (LLMs) exhibit pronounced conservative bias in relation extraction tasks, frequently defaulting to No_Relation label when an appropriate option is unavailable. While this behavior helps prevent incorrect relation assignments, our analysis reveals that it also leads to significant information loss when reasoning is not explicitly included in the output. We systematically evaluate this trade-off across multiple prompts, datasets, and relation types, introducing the concept of Hobson's choice to capture scenarios where models opt for safe but uninformative labels over hallucinated ones. Our findings suggest that conservative bias occurs twice as often as hallucination. To quantify this effect, we use SBERT and LLM prompts to capture the semantic similarity between conservative bias behaviors in constrained prompts and labels generated from semi-constrained and open-ended prompts. 

---
# Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better 

**Authors**: Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09040)  

**Abstract**: Typical large vision-language models (LVLMs) apply autoregressive supervision solely to textual sequences, without fully incorporating the visual modality into the learning process. This results in three key limitations: (1) an inability to utilize images without accompanying captions, (2) the risk that captions omit critical visual details, and (3) the challenge that certain vision-centric content cannot be adequately conveyed through text. As a result, current LVLMs often prioritize vision-to-language alignment while potentially overlooking fine-grained visual information. While some prior works have explored autoregressive image generation, effectively leveraging autoregressive visual supervision to enhance image understanding remains an open challenge. In this paper, we introduce Autoregressive Semantic Visual Reconstruction (ASVR), which enables joint learning of visual and textual modalities within a unified autoregressive framework. We show that autoregressively reconstructing the raw visual appearance of images does not enhance and may even impair multimodal understanding. In contrast, autoregressively reconstructing the semantic representation of images consistently improves comprehension. Notably, we find that even when models are given continuous image features as input, they can effectively reconstruct discrete semantic tokens, resulting in stable and consistent improvements across a wide range of multimodal understanding benchmarks. Our approach delivers significant performance gains across varying data scales (556k-2M) and types of LLM bacbones. Specifically, ASVR improves LLaVA-1.5 by 5% in average scores across 14 multimodal benchmarks. The code is available at this https URL. 

---
# e3: Learning to Explore Enables Extrapolation of Test-Time Compute for LLMs 

**Authors**: Amrith Setlur, Matthew Y. R. Yang, Charlie Snell, Jeremy Greer, Ian Wu, Virginia Smith, Max Simchowitz, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.09026)  

**Abstract**: Test-time scaling offers a promising path to improve LLM reasoning by utilizing more compute at inference time; however, the true promise of this paradigm lies in extrapolation (i.e., improvement in performance on hard problems as LLMs keep "thinking" for longer, beyond the maximum token budget they were trained on). Surprisingly, we find that most existing reasoning models do not extrapolate well. We show that one way to enable extrapolation is by training the LLM to perform in-context exploration: training the LLM to effectively spend its test time budget by chaining operations (such as generation, verification, refinement, etc.), or testing multiple hypotheses before it commits to an answer. To enable in-context exploration, we identify three key ingredients as part of our recipe e3: (1) chaining skills that the base LLM has asymmetric competence in, e.g., chaining verification (easy) with generation (hard), as a way to implement in-context search; (2) leveraging "negative" gradients from incorrect traces to amplify exploration during RL, resulting in longer search traces that chains additional asymmetries; and (3) coupling task difficulty with training token budget during training via a specifically-designed curriculum to structure in-context exploration. Our recipe e3 produces the best known 1.7B model according to AIME'25 and HMMT'25 scores, and extrapolates to 2x the training token budget. Our e3-1.7B model not only attains high pass@1 scores, but also improves pass@k over the base model. 

---
# SwS: Self-aware Weakness-driven Problem Synthesis in Reinforcement Learning for LLM Reasoning 

**Authors**: Xiao Liang, Zhong-Zhi Li, Yeyun Gong, Yang Wang, Hengyuan Zhang, Yelong Shen, Ying Nian Wu, Weizhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08989)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for training large language models (LLMs) on complex reasoning tasks, such as mathematical problem solving. A prerequisite for the scalability of RLVR is a high-quality problem set with precise and verifiable answers. However, the scarcity of well-crafted human-labeled math problems and limited-verification answers in existing distillation-oriented synthetic datasets limit their effectiveness in RL. Additionally, most problem synthesis strategies indiscriminately expand the problem set without considering the model's capabilities, leading to low efficiency in generating useful questions. To mitigate this issue, we introduce a Self-aware Weakness-driven problem Synthesis framework (SwS) that systematically identifies model deficiencies and leverages them for problem augmentation. Specifically, we define weaknesses as questions that the model consistently fails to learn through its iterative sampling during RL training. We then extract the core concepts from these failure cases and synthesize new problems to strengthen the model's weak areas in subsequent augmented training, enabling it to focus on and gradually overcome its weaknesses. Without relying on external knowledge distillation, our framework enables robust generalization byempowering the model to self-identify and address its weaknesses in RL, yielding average performance gains of 10.0% and 7.7% on 7B and 32B models across eight mainstream reasoning benchmarks. 

---
# Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model 

**Authors**: Ailin Huang, Bingxin Li, Bruce Wang, Boyong Wu, Chao Yan, Chengli Feng, Heng Wang, Hongyu Zhou, Hongyuan Wang, Jingbei Li, Jianjian Sun, Joanna Wang, Mingrui Chen, Peng Liu, Ruihang Miao, Shilei Jiang, Tian Fei, Wang You, Xi Chen, Xuerui Yang, Yechang Huang, Yuxiang Zhang, Zheng Ge, Zheng Gong, Zhewei Huang, Zixin Zhang, Bin Wang, Bo Li, Buyun Ma, Changxin Miao, Changyi Wan, Chen Xu, Dapeng Shi, Dingyuan Hu, Enle Liu, Guanzhe Huang, Gulin Yan, Hanpeng Hu, Haonan Jia, Jiahao Gong, Jiaoren Wu, Jie Wu, Jie Yang, Junzhe Lin, Kaixiang Li, Lei Xia, Longlong Gu, Ming Li, Nie Hao, Ranchen Ming, Shaoliang Pang, Siqi Liu, Song Yuan, Tiancheng Cao, Wen Li, Wenqing He, Xu Zhao, Xuelin Zhang, Yanbo Yu, Yinmin Zhong, Yu Zhou, Yuanwei Liang, Yuanwei Lu, Yuxiang Yang, Zidong Yang, Zili Zhang, Binxing Jiao, Heung-Yeung Shum, Jiansheng Chen, Jing Li, Xiangyu Zhang, Xinhao Zhang, Yibo Zhu, Daxin Jiang, Shuchang Zhou, Chen Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08967)  

**Abstract**: Large Audio-Language Models (LALMs) have significantly advanced intelligent human-computer interaction, yet their reliance on text-based outputs limits their ability to generate natural speech responses directly, hindering seamless audio interactions. To address this, we introduce Step-Audio-AQAA, a fully end-to-end LALM designed for Audio Query-Audio Answer (AQAA) tasks. The model integrates a dual-codebook audio tokenizer for linguistic and semantic feature extraction, a 130-billion-parameter backbone LLM and a neural vocoder for high-fidelity speech synthesis. Our post-training approach employs interleaved token-output of text and audio to enhance semantic coherence and combines Direct Preference Optimization (DPO) with model merge to improve performance. Evaluations on the StepEval-Audio-360 benchmark demonstrate that Step-Audio-AQAA excels especially in speech control, outperforming the state-of-art LALMs in key areas. This work contributes a promising solution for end-to-end LALMs and highlights the critical role of token-based vocoder in enhancing overall performance for AQAA tasks. 

---
# Socratic-MCTS: Test-Time Visual Reasoning by Asking the Right Questions 

**Authors**: David Acuna, Ximing Lu, Jaehun Jung, Hyunwoo Kim, Amlan Kar, Sanja Fidler, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08927)  

**Abstract**: Recent research in vision-language models (VLMs) has centered around the possibility of equipping them with implicit long-form chain-of-thought reasoning -- akin to the success observed in language models -- via distillation and reinforcement learning. But what about the non-reasoning models already trained and deployed across the internet? Should we simply abandon them, or is there hope for a search mechanism that can elicit hidden knowledge and induce long reasoning traces -- without any additional training or supervision? In this paper, we explore this possibility using a Monte Carlo Tree Search (MCTS)-inspired algorithm, which injects subquestion-subanswer pairs into the model's output stream. We show that framing reasoning as a search process -- where subquestions act as latent decisions within a broader inference trajectory -- helps the model "connect the dots" between fragmented knowledge and produce extended reasoning traces in non-reasoning models. We evaluate our method across three benchmarks and observe consistent improvements. Notably, our approach yields a 2% overall improvement on MMMU-PRO, including a significant 9% gain in Liberal Arts. 

---
# CulturalFrames: Assessing Cultural Expectation Alignment in Text-to-Image Models and Evaluation Metrics 

**Authors**: Shravan Nayak, Mehar Bhatia, Xiaofeng Zhang, Verena Rieser, Lisa Anne Hendricks, Sjoerd van Steenkiste, Yash Goyal, Karolina Stańczak, Aishwarya Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.08835)  

**Abstract**: The increasing ubiquity of text-to-image (T2I) models as tools for visual content generation raises concerns about their ability to accurately represent diverse cultural contexts. In this work, we present the first study to systematically quantify the alignment of T2I models and evaluation metrics with respect to both explicit as well as implicit cultural expectations. To this end, we introduce CulturalFrames, a novel benchmark designed for rigorous human evaluation of cultural representation in visual generations. Spanning 10 countries and 5 socio-cultural domains, CulturalFrames comprises 983 prompts, 3637 corresponding images generated by 4 state-of-the-art T2I models, and over 10k detailed human annotations. We find that T2I models not only fail to meet the more challenging implicit expectations but also the less challenging explicit expectations. Across models and countries, cultural expectations are missed an average of 44% of the time. Among these failures, explicit expectations are missed at a surprisingly high average rate of 68%, while implicit expectation failures are also significant, averaging 49%. Furthermore, we demonstrate that existing T2I evaluation metrics correlate poorly with human judgments of cultural alignment, irrespective of their internal reasoning. Collectively, our findings expose critical gaps, providing actionable directions for developing more culturally informed T2I models and evaluation methodologies. 

---
# Measuring Data Science Automation: A Survey of Evaluation Tools for AI Assistants and Agents 

**Authors**: Irene Testini, José Hernández-Orallo, Lorenzo Pacchiardi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08800)  

**Abstract**: Data science aims to extract insights from data to support decision-making processes. Recently, Large Language Models (LLMs) are increasingly used as assistants for data science, by suggesting ideas, techniques and small code snippets, or for the interpretation of results and reporting. Proper automation of some data-science activities is now promised by the rise of LLM agents, i.e., AI systems powered by an LLM equipped with additional affordances--such as code execution and knowledge bases--that can perform self-directed actions and interact with digital environments. In this paper, we survey the evaluation of LLM assistants and agents for data science. We find (1) a dominant focus on a small subset of goal-oriented activities, largely ignoring data management and exploratory activities; (2) a concentration on pure assistance or fully autonomous agents, without considering intermediate levels of human-AI collaboration; and (3) an emphasis on human substitution, therefore neglecting the possibility of higher levels of automation thanks to task transformation. 

---
# Paths to Causality: Finding Informative Subgraphs Within Knowledge Graphs for Knowledge-Based Causal Discovery 

**Authors**: Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.08771)  

**Abstract**: Inferring causal relationships between variable pairs is crucial for understanding multivariate interactions in complex systems. Knowledge-based causal discovery -- which involves inferring causal relationships by reasoning over the metadata of variables (e.g., names or textual context) -- offers a compelling alternative to traditional methods that rely on observational data. However, existing methods using Large Language Models (LLMs) often produce unstable and inconsistent results, compromising their reliability for causal inference. To address this, we introduce a novel approach that integrates Knowledge Graphs (KGs) with LLMs to enhance knowledge-based causal discovery. Our approach identifies informative metapath-based subgraphs within KGs and further refines the selection of these subgraphs using Learning-to-Rank-based models. The top-ranked subgraphs are then incorporated into zero-shot prompts, improving the effectiveness of LLMs in inferring the causal relationship. Extensive experiments on biomedical and open-domain datasets demonstrate that our method outperforms most baselines by up to 44.4 points in F1 scores, evaluated across diverse LLMs and KGs. Our code and datasets are available on GitHub: this https URL 

---
# EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements 

**Authors**: Issa Sugiura, Takashi Ishida, Taro Makino, Chieko Tazuke, Takanori Nakagawa, Kosuke Nakago, David Ha  

**Link**: [PDF](https://arxiv.org/pdf/2506.08762)  

**Abstract**: Financial analysis presents complex challenges that could leverage large language model (LLM) capabilities. However, the scarcity of challenging financial datasets, particularly for Japanese financial data, impedes academic innovation in financial analytics. As LLMs advance, this lack of accessible research resources increasingly hinders their development and evaluation in this specialized domain. To address this gap, we introduce EDINET-Bench, an open-source Japanese financial benchmark designed to evaluate the performance of LLMs on challenging financial tasks including accounting fraud detection, earnings forecasting, and industry prediction. EDINET-Bench is constructed by downloading annual reports from the past 10 years from Japan's Electronic Disclosure for Investors' NETwork (EDINET) and automatically assigning labels corresponding to each evaluation task. Our experiments reveal that even state-of-the-art LLMs struggle, performing only slightly better than logistic regression in binary classification for fraud detection and earnings forecasting. These results highlight significant challenges in applying LLMs to real-world financial applications and underscore the need for domain-specific adaptation. Our dataset, benchmark construction code, and evaluation code is publicly available to facilitate future research in finance with LLMs. 

---
# Consistent Paths Lead to Truth: Self-Rewarding Reinforcement Learning for LLM Reasoning 

**Authors**: Kongcheng Zhang, Qi Yao, Shunyu Liu, Yingjie Wang, Baisheng Lai, Jieping Ye, Mingli Song, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08745)  

**Abstract**: Recent advances of Reinforcement Learning (RL) have highlighted its potential in complex reasoning tasks, yet effective training often relies on external supervision, which limits the broader applicability. In this work, we propose a novel self-rewarding reinforcement learning framework to enhance Large Language Model (LLM) reasoning by leveraging the consistency of intermediate reasoning states across different reasoning trajectories. Our key insight is that correct responses often exhibit consistent trajectory patterns in terms of model likelihood: their intermediate reasoning states tend to converge toward their own final answers (high consistency) with minimal deviation toward other candidates (low volatility). Inspired by this observation, we introduce CoVo, an intrinsic reward mechanism that integrates Consistency and Volatility via a robust vector-space aggregation strategy, complemented by a curiosity bonus to promote diverse exploration. CoVo enables LLMs to perform RL in a self-rewarding manner, offering a scalable pathway for learning to reason without external supervision. Extensive experiments on diverse reasoning benchmarks show that CoVo achieves performance comparable to or even surpassing supervised RL. Our code is available at this https URL. 

---
# Approaching Dialogue State Tracking via Aligning Speech Encoders and LLMs 

**Authors**: Šimon Sedláček, Bolaji Yusuf, Ján Švec, Pradyoth Hegde, Santosh Kesiraju, Oldřich Plchot, Jan Černocký  

**Link**: [PDF](https://arxiv.org/pdf/2506.08633)  

**Abstract**: In this work, we approach spoken Dialogue State Tracking (DST) by bridging the representation spaces of speech encoders and LLMs via a small connector module, with a focus on fully open-sourced and open-data components (WavLM-large, OLMo). We focus on ablating different aspects of such systems including full/LoRA adapter fine-tuning, the effect of agent turns in the dialogue history, as well as fuzzy matching-based output post-processing, which greatly improves performance of our systems on named entities in the dialogue slot values. We conduct our experiments on the SpokenWOZ dataset, and additionally utilize the Speech-Aware MultiWOZ dataset to augment our training data. Ultimately, our best-performing WavLM + connector + OLMo-1B aligned models achieve state of the art on the SpokenWOZ test set (34.66% JGA), and our system with Gemma-2-9B-instruct further surpasses this result, reaching 42.17% JGA on SpokenWOZ test. 

---
# The Geometries of Truth Are Orthogonal Across Tasks 

**Authors**: Waiss Azizian, Michael Kirchhof, Eugene Ndiaye, Louis Bethune, Michal Klein, Pierre Ablin, Marco Cuturi  

**Link**: [PDF](https://arxiv.org/pdf/2506.08572)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive generalization capabilities across various tasks, but their claim to practical relevance is still mired by concerns on their reliability. Recent works have proposed examining the activations produced by an LLM at inference time to assess whether its answer to a question is correct. Some works claim that a "geometry of truth" can be learned from examples, in the sense that the activations that generate correct answers can be distinguished from those leading to mistakes with a linear classifier. In this work, we underline a limitation of these approaches: we observe that these "geometries of truth" are intrinsically task-dependent and fail to transfer across tasks. More precisely, we show that linear classifiers trained across distinct tasks share little similarity and, when trained with sparsity-enforcing regularizers, have almost disjoint supports. We show that more sophisticated approaches (e.g., using mixtures of probes and tasks) fail to overcome this limitation, likely because activation vectors commonly used to classify answers form clearly separated clusters when examined across tasks. 

---
# A Survey on Large Language Models for Mathematical Reasoning 

**Authors**: Peng-Yuan Wang, Tian-Shuo Liu, Chenyang Wang, Yi-Di Wang, Shu Yan, Cheng-Xing Jia, Xu-Hui Liu, Xin-Wei Chen, Jia-Cheng Xu, Ziniu Li, Yang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08446)  

**Abstract**: Mathematical reasoning has long represented one of the most fundamental and challenging frontiers in artificial intelligence research. In recent years, large language models (LLMs) have achieved significant advances in this area. This survey examines the development of mathematical reasoning abilities in LLMs through two high-level cognitive phases: comprehension, where models gain mathematical understanding via diverse pretraining strategies, and answer generation, which has progressed from direct prediction to step-by-step Chain-of-Thought (CoT) reasoning. We review methods for enhancing mathematical reasoning, ranging from training-free prompting to fine-tuning approaches such as supervised fine-tuning and reinforcement learning, and discuss recent work on extended CoT and "test-time scaling". Despite notable progress, fundamental challenges remain in terms of capacity, efficiency, and generalization. To address these issues, we highlight promising research directions, including advanced pretraining and knowledge augmentation techniques, formal reasoning frameworks, and meta-generalization through principled learning paradigms. This survey tries to provide some insights for researchers interested in enhancing reasoning capabilities of LLMs and for those seeking to apply these techniques to other domains. 

---
# Reinforcement Learning Teachers of Test Time Scaling 

**Authors**: Edoardo Cetin, Tianyu Zhao, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08388)  

**Abstract**: Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework. 

---
# Reinforce LLM Reasoning through Multi-Agent Reflection 

**Authors**: Yurun Yuan, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.08379)  

**Abstract**: Leveraging more test-time computation has proven to be an effective way to boost the reasoning capabilities of large language models (LLMs). Among various methods, the verify-and-improve paradigm stands out for enabling dynamic solution exploration and feedback incorporation. However, existing approaches often suffer from restricted feedback spaces and lack of coordinated training of different parties, leading to suboptimal performance. To address this, we model this multi-turn refinement process as a Markov Decision Process and introduce DPSDP (Direct Policy Search by Dynamic Programming), a reinforcement learning algorithm that trains an actor-critic LLM system to iteratively refine answers via direct preference learning on self-generated data. Theoretically, DPSDP can match the performance of any policy within the training distribution. Empirically, we instantiate DPSDP with various base models and show improvements on both in- and out-of-distribution benchmarks. For example, on benchmark MATH 500, majority voting over five refinement steps increases first-turn accuracy from 58.2% to 63.2% with Ministral-based models. An ablation study further confirms the benefits of multi-agent collaboration and out-of-distribution generalization. 

---
# How Much To Guide: Revisiting Adaptive Guidance in Classifier-Free Guidance Text-to-Vision Diffusion Models 

**Authors**: Huixuan Zhang, Junzhe Zhang, Xiaojun Wan  

**Link**: [PDF](https://arxiv.org/pdf/2506.08351)  

**Abstract**: With the rapid development of text-to-vision generation diffusion models, classifier-free guidance has emerged as the most prevalent method for conditioning. However, this approach inherently requires twice as many steps for model forwarding compared to unconditional generation, resulting in significantly higher costs. While previous study has introduced the concept of adaptive guidance, it lacks solid analysis and empirical results, making previous method unable to be applied to general diffusion models. In this work, we present another perspective of applying adaptive guidance and propose Step AG, which is a simple, universally applicable adaptive guidance strategy. Our evaluations focus on both image quality and image-text alignment. whose results indicate that restricting classifier-free guidance to the first several denoising steps is sufficient for generating high-quality, well-conditioned images, achieving an average speedup of 20% to 30%. Such improvement is consistent across different settings such as inference steps, and various models including video generation models, highlighting the superiority of our method. 

---
# From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information? 

**Authors**: Zhanke Zhou, Xiao Feng, Zhaocheng Zhu, Jiangchao Yao, Sanmi Koyejo, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08295)  

**Abstract**: While existing benchmarks probe the reasoning abilities of large language models (LLMs) across diverse domains, they predominantly assess passive reasoning, providing models with all the information needed to reach a solution. By contrast, active reasoning-where an LLM must interact with external systems to acquire missing evidence or data-has received little systematic attention. To address this shortfall, we present AR-Bench, a novel benchmark designed explicitly to evaluate an LLM's active reasoning skills. AR-Bench comprises three task families-detective cases, situation puzzles, and guessing numbers-that together simulate real-world, agentic scenarios and measure performance across commonsense, logical, and symbolic reasoning challenges. Empirical evaluation on AR-Bench demonstrates that contemporary LLMs exhibit pronounced difficulties with active reasoning: they frequently fail to acquire or leverage the information needed to solve tasks. This gap highlights a stark divergence between their passive and active reasoning abilities. Moreover, ablation studies indicate that even advanced strategies, such as tree-based searching or post-training approaches, yield only modest gains and fall short of the levels required for real-world deployment. Collectively, these findings highlight the critical need to advance methodology for active reasoning, e.g., incorporating interactive learning, real-time feedback loops, and environment-aware objectives for training. The benchmark is publicly available at: this https URL. 

---
# From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium 

**Authors**: Xie Yi, Zhanke Zhou, Chentao Cao, Qiyu Niu, Tongliang Liu, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.08292)  

**Abstract**: Multi-agent frameworks can substantially boost the reasoning power of large language models (LLMs), but they typically incur heavy computational costs and lack convergence guarantees. To overcome these challenges, we recast multi-LLM coordination as an incomplete-information game and seek a Bayesian Nash equilibrium (BNE), in which each agent optimally responds to its probabilistic beliefs about the strategies of others. We introduce Efficient Coordination via Nash Equilibrium (ECON), a hierarchical reinforcement-learning paradigm that marries distributed reasoning with centralized final output. Under ECON, each LLM independently selects responses that maximize its expected reward, conditioned on its beliefs about co-agents, without requiring costly inter-agent exchanges. We mathematically prove that ECON attains a markedly tighter regret bound than non-equilibrium multi-agent schemes. Empirically, ECON outperforms existing multi-LLM approaches by 11.2% on average across six benchmarks spanning complex reasoning and planning tasks. Further experiments demonstrate ECON's ability to flexibly incorporate additional models, confirming its scalability and paving the way toward larger, more powerful multi-LLM ensembles. The code is publicly available at: this https URL. 

---
# Instruction-Tuned Video-Audio Models Elucidate Functional Specialization in the Brain 

**Authors**: Subba Reddy Oota, Khushbu Pahwa, Prachi Jindal, Satya Sai Srinath Namburi, Maneesh Singh, Tanmoy Chakraborty, Bapi S. Raju, Manish Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.08277)  

**Abstract**: Recent voxel-wise multimodal brain encoding studies have shown that multimodal large language models (MLLMs) exhibit a higher degree of brain alignment compared to unimodal models in both unimodal and multimodal stimulus settings. More recently, instruction-tuned multimodal models have shown to generate task-specific representations that align strongly with brain activity. However, prior work evaluating the brain alignment of MLLMs has primarily focused on unimodal settings or relied on non-instruction-tuned multimodal models for multimodal stimuli. To address this gap, we investigated brain alignment, that is, measuring the degree of predictivity of neural activity recorded while participants were watching naturalistic movies (video along with audio) with representations derived from MLLMs. We utilized instruction-specific embeddings from six video and two audio instruction-tuned MLLMs. Experiments with 13 video task-specific instructions show that instruction-tuned video MLLMs significantly outperform non-instruction-tuned multimodal (by 15%) and unimodal models (by 20%). Our evaluation of MLLMs for both video and audio tasks using language-guided instructions shows clear disentanglement in task-specific representations from MLLMs, leading to precise differentiation of multimodal functional processing in the brain. We also find that MLLM layers align hierarchically with the brain, with early sensory areas showing strong alignment with early layers, while higher-level visual and language regions align more with middle to late layers. These findings provide clear evidence for the role of task-specific instructions in improving the alignment between brain activity and MLLMs, and open new avenues for mapping joint information processing in both the systems. We make the code publicly available [this https URL]. 

---
# Reinforcement Learning from Human Feedback with High-Confidence Safety Constraints 

**Authors**: Yaswanth Chittepu, Blossom Metevier, Will Schwarzer, Austin Hoag, Scott Niekum, Philip S. Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.08266)  

**Abstract**: Existing approaches to language model alignment often treat safety as a tradeoff against helpfulness, which can lead to unacceptable responses in sensitive domains. To ensure reliable performance in such settings, we propose High-Confidence Safe Reinforcement Learning from Human Feedback (HC-RLHF), a method that provides high-confidence safety guarantees while maximizing helpfulness. Similar to previous methods, HC-RLHF explicitly decouples human preferences into helpfulness and harmlessness (safety), which are learned by training a reward model and a cost model, respectively. It then employs a two-step process to find safe solutions. In the first step, it optimizes the reward function under an intentionally pessimistic version of the cost constraint. In the second step, the trained model undergoes a safety test to verify whether its performance stays within an upper-confidence bound of the actual cost constraint. We provide a theoretical analysis of HC-RLHF, including proof that it will not return an unsafe solution with a probability greater than a user-specified threshold. For our empirical analysis, we apply HC-RLHF to align three different language models (Qwen2-1.5B, Qwen2.5-3B, and LLaMa3.2-3B) with human preferences. Our results demonstrate that HC-RLHF produces safe models with high probability and can improve harmlessness and helpfulness compared to previous methods. 

---
# RADAR: Benchmarking Language Models on Imperfect Tabular Data 

**Authors**: Ken Gu, Zhihan Zhang, Kate Lin, Yuwei Zhang, Akshay Paruchuri, Hong Yu, Mehran Kazemi, Kumar Ayush, A. Ali Heydari, Maxwell A. Xu, Girish Narayanswamy, Yun Liu, Ming-Zher Poh, Yuzhe Yang, Mark Malhotra, Shwetak Patel, Hamid Palangi, Xuhai Xu, Daniel McDuff, Tim Althoff, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08249)  

**Abstract**: Language models (LMs) are increasingly being deployed to perform autonomous data analyses. However, their data awareness -- the ability to recognize, reason over, and appropriately handle data artifacts such as missing values, outliers, and logical inconsistencies -- remains underexplored. These artifacts are especially common in real-world tabular data and, if mishandled, can significantly compromise the validity of analytical conclusions. To address this gap, we present RADAR, a benchmark for systematically evaluating data-aware reasoning on tabular data. We develop a framework to simulate data artifacts via programmatic perturbations to enable targeted evaluation of model behavior. RADAR comprises 2980 table query pairs, grounded in real-world data spanning 9 domains and 5 data artifact types. In addition to evaluating artifact handling, RADAR systematically varies table size to study how reasoning performance holds when increasing table size. Our evaluation reveals that, despite decent performance on tables without data artifacts, frontier models degrade significantly when data artifacts are introduced, exposing critical gaps in their capacity for robust, data-aware analysis. Designed to be flexible and extensible, RADAR supports diverse perturbation types and controllable table sizes, offering a valuable resource for advancing tabular reasoning. 

---
# A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation 

**Authors**: Andrew Z. Wang, Songwei Ge, Tero Karras, Ming-Yu Liu, Yogesh Balaji  

**Link**: [PDF](https://arxiv.org/pdf/2506.08210)  

**Abstract**: Both text-to-image generation and large language models (LLMs) have made significant advancements. However, many text-to-image models still employ the somewhat outdated T5 and CLIP as their text encoders. In this work, we investigate the effectiveness of using modern decoder-only LLMs as text encoders for text-to-image diffusion models. We build a standardized training and evaluation pipeline that allows us to isolate and evaluate the effect of different text embeddings. We train a total of 27 text-to-image models with 12 different text encoders to analyze the critical aspects of LLMs that could impact text-to-image generation, including the approaches to extract embeddings, different LLMs variants, and model sizes. Our experiments reveal that the de facto way of using last-layer embeddings as conditioning leads to inferior performance. Instead, we explore embeddings from various layers and find that using layer-normalized averaging across all layers significantly improves alignment with complex prompts. Most LLMs with this conditioning outperform the baseline T5 model, showing enhanced performance in advanced visio-linguistic reasoning skills. 

---
# GradEscape: A Gradient-Based Evader Against AI-Generated Text Detectors 

**Authors**: Wenlong Meng, Shuguo Fan, Chengkun Wei, Min Chen, Yuwei Li, Yuanchao Zhang, Zhikun Zhang, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08188)  

**Abstract**: In this paper, we introduce GradEscape, the first gradient-based evader designed to attack AI-generated text (AIGT) detectors. GradEscape overcomes the undifferentiable computation problem, caused by the discrete nature of text, by introducing a novel approach to construct weighted embeddings for the detector input. It then updates the evader model parameters using feedback from victim detectors, achieving high attack success with minimal text modification. To address the issue of tokenizer mismatch between the evader and the detector, we introduce a warm-started evader method, enabling GradEscape to adapt to detectors across any language model architecture. Moreover, we employ novel tokenizer inference and model extraction techniques, facilitating effective evasion even in query-only access.
We evaluate GradEscape on four datasets and three widely-used language models, benchmarking it against four state-of-the-art AIGT evaders. Experimental results demonstrate that GradEscape outperforms existing evaders in various scenarios, including with an 11B paraphrase model, while utilizing only 139M parameters. We have successfully applied GradEscape to two real-world commercial AIGT detectors. Our analysis reveals that the primary vulnerability stems from disparity in text expression styles within the training data. We also propose a potential defense strategy to mitigate the threat of AIGT evaders. We open-source our GradEscape for developing more robust AIGT detectors. 

---
# AutoSDT: Scaling Data-Driven Discovery Tasks Toward Open Co-Scientists 

**Authors**: Yifei Li, Hanane Nour Moussa, Ziru Chen, Shijie Chen, Botao Yu, Mingyi Xue, Benjamin Burns, Tzu-Yao Chiu, Vishal Dey, Zitong Lu, Chen Wei, Qianheng Zhang, Tianyu Zhang, Song Gao, Xuhui Huang, Xia Ning, Nesreen K. Ahmed, Ali Payani, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.08140)  

**Abstract**: Despite long-standing efforts in accelerating scientific discovery with AI, building AI co-scientists remains challenging due to limited high-quality data for training and evaluation. To tackle this data scarcity issue, we present AutoSDT, an automatic pipeline that collects high-quality coding tasks in real-world data-driven discovery workflows. AutoSDT leverages the coding capabilities and parametric knowledge of LLMs to search for diverse sources, select ecologically valid tasks, and synthesize accurate task instructions and code solutions. Using our pipeline, we construct AutoSDT-5K, a dataset of 5,404 coding tasks for data-driven discovery that covers four scientific disciplines and 756 unique Python packages. To the best of our knowledge, AutoSDT-5K is the only automatically collected and the largest open dataset for data-driven scientific discovery. Expert feedback on a subset of 256 tasks shows the effectiveness of AutoSDT: 93% of the collected tasks are ecologically valid, and 92.2% of the synthesized programs are functionally correct. Trained on AutoSDT-5K, the Qwen2.5-Coder-Instruct LLM series, dubbed AutoSDT-Coder, show substantial improvement on two challenging data-driven discovery benchmarks, ScienceAgentBench and DiscoveryBench. Most notably, AutoSDT-Coder-32B reaches the same level of performance as GPT-4o on ScienceAgentBench with a success rate of 7.8%, doubling the performance of its base model. On DiscoveryBench, it lifts the hypothesis matching score to 8.1, bringing a 17.4% relative improvement and closing the gap between open-weight models and GPT-4o. 

---
# Bingo: Boosting Efficient Reasoning of LLMs via Dynamic and Significance-based Reinforcement Learning 

**Authors**: Hanbing Liu, Lang Cao, Yuanyi Ren, Mengyu Zhou, Haoyu Dong, Xiaojun Ma, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08125)  

**Abstract**: Large language models have demonstrated impressive reasoning capabilities, yet they often suffer from inefficiencies due to unnecessarily verbose or redundant outputs. While many works have explored reinforcement learning (RL) to enhance reasoning abilities, most primarily focus on improving accuracy, with limited attention to reasoning efficiency. Some existing approaches introduce direct length-based rewards to encourage brevity, but this often leads to noticeable drops in accuracy. In this paper, we propose Bingo, an RL framework that advances length-based reward design to boost efficient reasoning. Bingo incorporates two key mechanisms: a significance-aware length reward, which gradually guides the model to reduce only insignificant tokens, and a dynamic length reward, which initially encourages elaborate reasoning for hard questions but decays over time to improve overall efficiency. Experiments across multiple reasoning benchmarks show that Bingo improves both accuracy and efficiency. It outperforms the vanilla reward and several other length-based reward baselines in RL, achieving a favorable trade-off between accuracy and efficiency. These results underscore the potential of training LLMs explicitly for efficient reasoning. 

---
# Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval 

**Authors**: Abdellah Ghassel, Ian Robinson, Gabriel Tanase, Hal Cooper, Bryan Thompson, Zhen Han, Vassilis N. Ioannidis, Soji Adeshina, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2506.08074)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language models in external evidence, yet it still falters when answers must be pieced together across semantically distant documents. We close this gap with the Hierarchical Lexical Graph (HLG), a three-tier index that (i) traces every atomic proposition to its source, (ii) clusters propositions into latent topics, and (iii) links entities and relations to expose cross-document paths. On top of HLG we build two complementary, plug-and-play retrievers: StatementGraphRAG, which performs fine-grained entity-aware beam search over propositions for high-precision factoid questions, and TopicGraphRAG, which selects coarse topics before expanding along entity links to supply broad yet relevant context for exploratory queries. Additionally, existing benchmarks lack the complexity required to rigorously evaluate multi-hop summarization systems, often focusing on single-document queries or limited datasets. To address this, we introduce a synthetic dataset generation pipeline that curates realistic, multi-document question-answer pairs, enabling robust evaluation of multi-hop retrieval systems. Extensive experiments across five datasets demonstrate that our methods outperform naive chunk-based RAG achieving an average relative improvement of 23.1% in retrieval recall and correctness. Open-source Python library is available at this https URL. 

---
# Modality-Balancing Preference Optimization of Large Multimodal Models by Adversarial Negative Mining 

**Authors**: Chenxi Liu, Tianyi Xiong, Ruibo Chen, Yihan Wu, Junfeng Guo, Tianyi Zhou, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08022)  

**Abstract**: The task adaptation and alignment of Large Multimodal Models (LMMs) have been significantly advanced by instruction tuning and further strengthened by recent preference optimization. Yet, most LMMs still suffer from severe modality imbalance during reasoning, i.e., outweighing language prior biases over visual inputs, which bottlenecks their generalization to downstream tasks and causes hallucinations. However, existing preference optimization approaches for LMMs do not focus on restraining the internal biases of their Large Language Model (LLM) backbones when curating the training data. Moreover, they heavily rely on offline data and lack the capacity to explore diverse responses adaptive to dynamic distributional shifts during training. Meanwhile, Group Relative Policy Optimization (GRPO), a recent method using online-generated data and verified rewards to improve reasoning capabilities, remains largely underexplored in LMM alignment. In this paper, we propose a novel preference learning framework, Modality-Balancing Preference Optimization (MBPO), to address the modality imbalance in LMMs. MBPO constructs a more effective offline preference dataset by generating hard negatives, i.e., rejected responses misled by LLM biases due to limited usage of visual information, through adversarial perturbation of input images. Moreover, MBPO leverages the easy-to-verify nature of close-ended tasks to generate online responses with verified rewards. GRPO is then employed to train the model with offline-online hybrid data. Extensive experiments demonstrate that MBPO can enhance LMM performance on challenging vision-language tasks and effectively reduce hallucinations. 

---
