# Bonsai: Interpretable Tree-Adaptive Grounded Reasoning 

**Authors**: Kate Sanders, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03640)  

**Abstract**: To develop general-purpose collaborative agents, humans need reliable AI systems that can (1) adapt to new domains and (2) transparently reason with uncertainty to allow for verification and correction. Black-box models demonstrate powerful data processing abilities but do not satisfy these criteria due to their opaqueness, domain specificity, and lack of uncertainty awareness. We introduce Bonsai, a compositional and probabilistic reasoning system that generates adaptable inference trees by retrieving relevant grounding evidence and using it to compute likelihoods of sub-claims derived from broader natural language inferences. Bonsai's reasoning power is tunable at test-time via evidence scaling and it demonstrates reliable handling of varied domains including transcripts, photographs, videos, audio, and databases. Question-answering and human alignment experiments demonstrate that Bonsai matches the performance of domain-specific black-box methods while generating interpretable, grounded, and uncertainty-aware reasoning traces. 

---
# Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models 

**Authors**: NVIDIA, Aaron Blakeman, Aarti Basant, Abhinav Khattar, Adithya Renduchintala, Akhiad Bercovich, Aleksander Ficek, Alexis Bjorlin, Ali Taghibakhshi, Amala Sanjay Deshmukh, Ameya Sunil Mahabaleshwarkar, Andrew Tao, Anna Shors, Ashwath Aithal, Ashwin Poojary, Ayush Dattagupta, Balaram Buddharaju, Bobby Chen, Boris Ginsburg, Boxin Wang, Brandon Norick, Brian Butterfield, Bryan Catanzaro, Carlo del Mundo, Chengyu Dong, Christine Harvey, Christopher Parisien, Dan Su, Daniel Korzekwa, Danny Yin, Daria Gitman, David Mosallanezhad, Deepak Narayanan, Denys Fridman, Dima Rekesh, Ding Ma, Dmytro Pykhtar, Dong Ahn, Duncan Riach, Dusan Stosic, Eileen Long, Elad Segal, Ellie Evans, Eric Chung, Erick Galinkin, Evelina Bakhturina, Ewa Dobrowolska, Fei Jia, Fuxiao Liu, Gargi Prasad, Gerald Shen, Guilin Liu, Guo Chen, Haifeng Qian, Helen Ngo, Hongbin Liu, Hui Li, Igor Gitman, Ilia Karmanov, Ivan Moshkov, Izik Golan, Jan Kautz, Jane Polak Scowcroft, Jared Casper, Jarno Seppanen, Jason Lu, Jason Sewall, Jiaqi Zeng, Jiaxuan You, Jimmy Zhang, Jing Zhang, Jining Huang, Jinze Xue, Jocelyn Huang, Joey Conway, John Kamalu, Jon Barker, Jonathan Cohen, Joseph Jennings, Jupinder Parmar, Karan Sapra, Kari Briski, Kateryna Chumachenko, Katherine Luna, Keshav Santhanam, Kezhi Kong, Kirthi Sivamani, Krzysztof Pawelec, Kumar Anik, Kunlun Li, Lawrence McAfee, Leon Derczynski, Lindsey Pavao, Luis Vega, Lukas Voegtle, Maciej Bala, Maer Rodrigues de Melo, Makesh Narsimhan Sreedhar, Marcin Chochowski, Markus Kliegl  

**Link**: [PDF](https://arxiv.org/pdf/2504.03624)  

**Abstract**: As inference-time scaling becomes critical for enhanced reasoning capabilities, it is increasingly becoming important to build models that are efficient to infer. We introduce Nemotron-H, a family of 8B and 56B/47B hybrid Mamba-Transformer models designed to reduce inference cost for a given accuracy level. To achieve this goal, we replace the majority of self-attention layers in the common Transformer model architecture with Mamba layers that perform constant computation and require constant memory per generated token. We show that Nemotron-H models offer either better or on-par accuracy compared to other similarly-sized state-of-the-art open-sourced Transformer models (e.g., Qwen-2.5-7B/72B and Llama-3.1-8B/70B), while being up to 3$\times$ faster at inference. To further increase inference speed and reduce the memory required at inference time, we created Nemotron-H-47B-Base from the 56B model using a new compression via pruning and distillation technique called MiniPuzzle. Nemotron-H-47B-Base achieves similar accuracy to the 56B model, but is 20% faster to infer. In addition, we introduce an FP8-based training recipe and show that it can achieve on par results with BF16-based training. This recipe is used to train the 56B model. All Nemotron-H models will be released, with support in Hugging Face, NeMo, and Megatron-LM. 

---
# Align to Structure: Aligning Large Language Models with Structural Information 

**Authors**: Zae Myung Kim, Anand Ramachandran, Farideh Tavazoee, Joo-Kyung Kim, Oleg Rokhlenko, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03622)  

**Abstract**: Generating long, coherent text remains a challenge for large language models (LLMs), as they lack hierarchical planning and structured organization in discourse generation. We introduce Structural Alignment, a novel method that aligns LLMs with human-like discourse structures to enhance long-form text generation. By integrating linguistically grounded discourse frameworks into reinforcement learning, our approach guides models to produce coherent and well-organized outputs. We employ a dense reward scheme within a Proximal Policy Optimization framework, assigning fine-grained, token-level rewards based on the discourse distinctiveness relative to human writing. Two complementary reward models are evaluated: the first improves readability by scoring surface-level textual features to provide explicit structuring, while the second reinforces deeper coherence and rhetorical sophistication by analyzing global discourse patterns through hierarchical discourse motifs, outperforming both standard and RLHF-enhanced models in tasks such as essay generation and long-document summarization. All training data and code will be publicly shared at this https URL. 

---
# Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task 

**Authors**: Leonardo Ranaldi, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2504.03616)  

**Abstract**: Retrieval-augmented generation (RAG) has become a cornerstone of contemporary NLP, enhancing large language models (LLMs) by allowing them to access richer factual contexts through in-context retrieval. While effective in monolingual settings, especially in English, its use in multilingual tasks remains unexplored. This paper investigates the effectiveness of RAG across multiple languages by proposing novel approaches for multilingual open-domain question-answering. We evaluate the performance of various multilingual RAG strategies, including question-translation (tRAG), which translates questions into English before retrieval, and Multilingual RAG (MultiRAG), where retrieval occurs directly across multiple languages. Our findings reveal that tRAG, while useful, suffers from limited coverage. In contrast, MultiRAG improves efficiency by enabling multilingual retrieval but introduces inconsistencies due to cross-lingual variations in the retrieved content. To address these issues, we propose Crosslingual RAG (CrossRAG), a method that translates retrieved documents into a common language (e.g., English) before generating the response. Our experiments show that CrossRAG significantly enhances performance on knowledge-intensive tasks, benefiting both high-resource and low-resource languages. 

---
# AIR: A Systematic Analysis of Annotations, Instructions, and Response Pairs in Preference Dataset 

**Authors**: Bingxiang He, Wenbin Zhang, Jiaxi Song, Cheng Qian, Zixuan Fu, Bowen Sun, Ning Ding, Haiwen Hong, Longtao Huang, Hui Xue, Ganqu Cui, Wanxiang Che, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.03612)  

**Abstract**: Preference learning is critical for aligning large language models (LLMs) with human values, yet its success hinges on high-quality datasets comprising three core components: Preference \textbf{A}nnotations, \textbf{I}nstructions, and \textbf{R}esponse Pairs. Current approaches conflate these components, obscuring their individual impacts and hindering systematic optimization. In this work, we propose \textbf{AIR}, a component-wise analysis framework that systematically isolates and optimizes each component while evaluating their synergistic effects. Through rigorous experimentation, AIR reveals actionable principles: annotation simplicity (point-wise generative scoring), instruction inference stability (variance-based filtering across LLMs), and response pair quality (moderate margins + high absolute scores). When combined, these principles yield +5.3 average gains over baseline method, even with only 14k high-quality pairs. Our work shifts preference dataset design from ad hoc scaling to component-aware optimization, offering a blueprint for efficient, reproducible alignment. 

---
# APIGen-MT: Agentic Pipeline for Multi-Turn Data Generation via Simulated Agent-Human Interplay 

**Authors**: Akshara Prabhakar, Zuxin Liu, Weiran Yao, Jianguo Zhang, Ming Zhu, Shiyu Wang, Zhiwei Liu, Tulika Awalgaonkar, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Huan Wang, Silvio Savarese, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2504.03601)  

**Abstract**: Training effective AI agents for multi-turn interactions requires high-quality data that captures realistic human-agent dynamics, yet such data is scarce and expensive to collect manually. We introduce APIGen-MT, a two-phase framework that generates verifiable and diverse multi-turn agent data. In the first phase, our agentic pipeline produces detailed task blueprints with ground-truth actions, leveraging a committee of LLM reviewers and iterative feedback loops. These blueprints are then transformed into complete interaction trajectories through simulated human-agent interplay. We train a family of models -- the xLAM-2-fc-r series with sizes ranging from 1B to 70B parameters. Our models outperform frontier models such as GPT-4o and Claude 3.5 on $\tau$-bench and BFCL benchmarks, with the smaller models surpassing their larger counterparts, particularly in multi-turn settings, while maintaining superior consistency across multiple trials. Comprehensive experiments demonstrate that our verified blueprint-to-details approach yields high-quality training data, enabling the development of more reliable, efficient, and capable agents. We open-source both the synthetic data collected and the trained xLAM-2-fc-r models to advance research in AI agents. Models are available on HuggingFace at this https URL and project website is this https URL 

---
# EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline 

**Authors**: Peter Baile Chen, Tomer Wolfson, Michael Cafarella, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2504.03598)  

**Abstract**: Existing information retrieval systems excel in cases where the language of target documents closely matches that of the user query. However, real-world retrieval systems are often required to implicitly reason whether a document is relevant. For example, when retrieving technical texts or tables, their relevance to the user query may be implied through a particular jargon or structure, rather than explicitly expressed in their content. Large language models (LLMs) hold great potential in identifying such implied relevance by leveraging their reasoning skills. Nevertheless, current LLM-augmented retrieval is hindered by high latency and computation cost, as the LLM typically computes the query-document relevance online, for every query anew. To tackle this issue we introduce EnrichIndex, a retrieval approach which instead uses the LLM offline to build semantically-enriched retrieval indices, by performing a single pass over all documents in the retrieval corpus once during ingestion time. Furthermore, the semantically-enriched indices can complement existing online retrieval approaches, boosting the performance of LLM re-rankers. We evaluated EnrichIndex on five retrieval tasks, involving passages and tables, and found that it outperforms strong online LLM-based retrieval systems, with an average improvement of 11.7 points in recall @ 10 and 10.6 points in NDCG @ 10 compared to strong baselines. In terms of online calls to the LLM, it processes 293.3 times fewer tokens which greatly reduces the online latency and cost. Overall, EnrichIndex is an effective way to build better retrieval indices offline by leveraging the strong reasoning skills of LLMs. 

---
# Extending the SAREF4ENER Ontology with Flexibility Based on FlexOffers 

**Authors**: Fabio Lilliu, Amir Laadhar, Christian Thomsen, Diego Reforgiato Recupero, Torben Bach Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03595)  

**Abstract**: A key element to support the increased amounts of renewable energy in the energy system is flexibility, i.e., the possibility of changing energy loads in time and amount. Many flexibility models have been designed; however, exact models fail to scale for long time horizons or many devices. Because of this, the FlexOffer (FOs) model has been designed, to provide device-independent approximations of flexibility with good accuracy, and much better scaling for long time horizons and many devices. An important aspect of the real-life implementation of energy flexibility is enabling flexible data exchange with many types of smart energy appliances and market systems, e.g., in smart buildings. For this, ontologies standardizing data formats are required. However, the current industry standard ontology for integrating smart devices for energy purposes, SAREF for Energy Flexibility (SAREF4ENER) only has limited support for flexibility and thus cannot support important use cases. In this paper we propose an extension of SAREF4ENER that integrates full support for the complete FlexOffer model, including advanced use cases, while maintaining backward compatibility. This novel ontology module can accurately describe flexibility for advanced devices such as electric vehicles, batteries, and heat pumps. It can also capture the inherent uncertainty associated with many flexible load types. 

---
# SynWorld: Virtual Scenario Synthesis for Agentic Action Knowledge Refinement 

**Authors**: Runnan Fang, Xiaobin Wang, Yuan Liang, Shuofei Qiao, Jialong Wu, Zekun Xi, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03561)  

**Abstract**: In the interaction between agents and their environments, agents expand their capabilities by planning and executing actions. However, LLM-based agents face substantial challenges when deployed in novel environments or required to navigate unconventional action spaces. To empower agents to autonomously explore environments, optimize workflows, and enhance their understanding of actions, we propose SynWorld, a framework that allows agents to synthesize possible scenarios with multi-step action invocation within the action space and perform Monte Carlo Tree Search (MCTS) exploration to effectively refine their action knowledge in the current environment. Our experiments demonstrate that SynWorld is an effective and general approach to learning action knowledge in new environments. Code is available at this https URL. 

---
# Agentic Knowledgeable Self-awareness 

**Authors**: Shuofei Qiao, Zhisong Qiu, Baochang Ren, Xiaobin Wang, Xiangyuan Ru, Ningyu Zhang, Xiang Chen, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03553)  

**Abstract**: Large Language Models (LLMs) have achieved considerable performance across various agentic planning tasks. However, traditional agent planning approaches adopt a "flood irrigation" methodology that indiscriminately injects gold trajectories, external feedback, and domain knowledge into agent models. This practice overlooks the fundamental human cognitive principle of situational self-awareness during decision-making-the ability to dynamically assess situational demands and strategically employ resources during decision-making. We propose agentic knowledgeable self-awareness to address this gap, a novel paradigm enabling LLM-based agents to autonomously regulate knowledge utilization. Specifically, we propose KnowSelf, a data-centric approach that applies agents with knowledgeable self-awareness like humans. Concretely, we devise a heuristic situation judgement criterion to mark special tokens on the agent's self-explored trajectories for collecting training data. Through a two-stage training process, the agent model can switch between different situations by generating specific special tokens, achieving optimal planning effects with minimal costs. Our experiments demonstrate that KnowSelf can outperform various strong baselines on different tasks and models with minimal use of external knowledge. Code is available at this https URL. 

---
# MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation 

**Authors**: Khai Le-Duc, Tuyen Tran, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh, Thanh Nguyen-Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03546)  

**Abstract**: Multilingual speech translation (ST) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, Traditional Chinese and Simplified Chinese, together with the models. With 290,000 samples, our dataset is the largest medical machine translation (MT) dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most extensive analysis study in ST research to date, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence (seq2seq) comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: this https URL. 

---
# Diverse In-Context Example Selection After Decomposing Programs and Aligned Utterances Improves Semantic Parsing 

**Authors**: Mayank Kothyari, Sunita Sarawagi, Soumen Chakrabarti, Gaurav Arora, Srujana Merugu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03541)  

**Abstract**: LLMs are increasingly used as seq2seq translators from natural language utterances to structured programs, a process called semantic interpretation. Unlike atomic labels or token sequences, programs are naturally represented as abstract syntax trees (ASTs). Such structured representation raises novel issues related to the design and selection of in-context examples (ICEs) presented to the LLM. We focus on decomposing the pool of available ICE trees into fragments, some of which may be better suited to solving the test instance. Next, we propose how to use (additional invocations of) an LLM with prompted syntax constraints to automatically map the fragments to corresponding utterances. Finally, we adapt and extend a recent method for diverse ICE selection to work with whole and fragmented ICE instances. We evaluate our system, SCUD4ICL, on popular diverse semantic parsing benchmarks, showing visible accuracy gains from our proposed decomposed diverse demonstration method. Benefits are particularly notable for smaller LLMs, ICE pools having larger labeled trees, and programs in lower resource languages. 

---
# Neutralizing the Narrative: AI-Powered Debiasing of Online News Articles 

**Authors**: Chen Wei Kuo, Kevin Chu, Nouar AlDahoul, Hazem Ibrahim, Talal Rahwan, Yasir Zaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.03520)  

**Abstract**: Bias in news reporting significantly impacts public perception, particularly regarding crime, politics, and societal issues. Traditional bias detection methods, predominantly reliant on human moderation, suffer from subjective interpretations and scalability constraints. Here, we introduce an AI-driven framework leveraging advanced large language models (LLMs), specifically GPT-4o, GPT-4o Mini, Gemini Pro, Gemini Flash, Llama 8B, and Llama 3B, to systematically identify and mitigate biases in news articles. To this end, we collect an extensive dataset consisting of over 30,000 crime-related articles from five politically diverse news sources spanning a decade (2013-2023). Our approach employs a two-stage methodology: (1) bias detection, where each LLM scores and justifies biased content at the paragraph level, validated through human evaluation for ground truth establishment, and (2) iterative debiasing using GPT-4o Mini, verified by both automated reassessment and human reviewers. Empirical results indicate GPT-4o Mini's superior accuracy in bias detection and effectiveness in debiasing. Furthermore, our analysis reveals temporal and geographical variations in media bias correlating with socio-political dynamics and real-world events. This study contributes to scalable computational methodologies for bias mitigation, promoting fairness and accountability in news reporting. 

---
# Structured Legal Document Generation in India: A Model-Agnostic Wrapper Approach with VidhikDastaavej 

**Authors**: Shubham Kumar Nigam, Balaramamahanthi Deepak Patnaik, Ajay Varghese Thomas, Noel Shallum, Kripabandhu Ghosh, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2504.03486)  

**Abstract**: Automating legal document drafting can significantly enhance efficiency, reduce manual effort, and streamline legal workflows. While prior research has explored tasks such as judgment prediction and case summarization, the structured generation of private legal documents in the Indian legal domain remains largely unaddressed. To bridge this gap, we introduce VidhikDastaavej, a novel, anonymized dataset of private legal documents, and develop NyayaShilp, a fine-tuned legal document generation model specifically adapted to Indian legal texts. We propose a Model-Agnostic Wrapper (MAW), a two-step framework that first generates structured section titles and then iteratively produces content while leveraging retrieval-based mechanisms to ensure coherence and factual accuracy. We benchmark multiple open-source LLMs, including instruction-tuned and domain-adapted versions, alongside proprietary models for comparison. Our findings indicate that while direct fine-tuning on small datasets does not always yield improvements, our structured wrapper significantly enhances coherence, factual adherence, and overall document quality while mitigating hallucinations. To ensure real-world applicability, we developed a Human-in-the-Loop (HITL) Document Generation System, an interactive user interface that enables users to specify document types, refine section details, and generate structured legal drafts. This tool allows legal professionals and researchers to generate, validate, and refine AI-generated legal documents efficiently. Extensive evaluations, including expert assessments, confirm that our framework achieves high reliability in structured legal drafting. This research establishes a scalable and adaptable foundation for AI-assisted legal drafting in India, offering an effective approach to structured legal document generation. 

---
# SpectR: Dynamically Composing LM Experts with Spectral Routing 

**Authors**: William Fleshman, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2504.03454)  

**Abstract**: Training large, general-purpose language models poses significant challenges. The growing availability of specialized expert models, fine-tuned from pretrained models for specific tasks or domains, offers a promising alternative. Leveraging the potential of these existing expert models in real-world applications requires effective methods to select or merge the models best suited for a given task. This paper introduces SPECTR, an approach for dynamically composing expert models at each time step during inference. Notably, our method requires no additional training and enables flexible, token- and layer-wise model combinations. Our experimental results demonstrate that SPECTR improves routing accuracy over alternative training-free methods, increasing task performance across expert domains. 

---
# Locations of Characters in Narratives: Andersen and Persuasion Datasets 

**Authors**: Batuhan Ozyurt, Roya Arkhmammadova, Deniz Yuret  

**Link**: [PDF](https://arxiv.org/pdf/2504.03434)  

**Abstract**: The ability of machines to grasp spatial understanding within narrative contexts is an intriguing aspect of reading comprehension that continues to be studied. Motivated by the goal to test the AI's competence in understanding the relationship between characters and their respective locations in narratives, we introduce two new datasets: Andersen and Persuasion. For the Andersen dataset, we selected fifteen children's stories from "Andersen's Fairy Tales" by Hans Christian Andersen and manually annotated the characters and their respective locations throughout each story. Similarly, for the Persuasion dataset, characters and their locations in the novel "Persuasion" by Jane Austen were also manually annotated. We used these datasets to prompt Large Language Models (LLMs). The prompts are created by extracting excerpts from the stories or the novel and combining them with a question asking the location of a character mentioned in that excerpt. Out of the five LLMs we tested, the best-performing one for the Andersen dataset accurately identified the location in 61.85% of the examples, while for the Persuasion dataset, the best-performing one did so in 56.06% of the cases. 

---
# Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning 

**Authors**: Sanghwan Bae, Jiwoo Hong, Min Young Lee, Hanbyul Kim, JeongYeon Nam, Donghyun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2504.03380)  

**Abstract**: Reasoning-Oriented Reinforcement Learning (RORL) enhances the reasoning ability of Large Language Models (LLMs). However, due to the sparsity of rewards in RORL, effective training is highly dependent on the selection of problems of appropriate difficulty. Although curriculum learning attempts to address this by adjusting difficulty, it often relies on static schedules, and even recent online filtering methods lack theoretical grounding and a systematic understanding of their effectiveness. In this work, we theoretically and empirically show that curating the batch with the problems that the training model achieves intermediate accuracy on the fly can maximize the effectiveness of RORL training, namely balanced online difficulty filtering. We first derive that the lower bound of the KL divergence between the initial and the optimal policy can be expressed with the variance of the sampled accuracy. Building on those insights, we show that balanced filtering can maximize the lower bound, leading to better performance. Experimental results across five challenging math reasoning benchmarks show that balanced online filtering yields an additional 10% in AIME and 4% improvements in average over plain GRPO. Moreover, further analysis shows the gains in sample efficiency and training time efficiency, exceeding the maximum reward of plain GRPO within 60% training time and the volume of the training set. 

---
# Detecting Stereotypes and Anti-stereotypes the Correct Way Using Social Psychological Underpinnings 

**Authors**: Kaustubh Shivshankar Shejole, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2504.03352)  

**Abstract**: Stereotypes are known to be highly pernicious, making their detection critically important. However, current research predominantly focuses on detecting and evaluating stereotypical biases in LLMs, leaving the study of stereotypes in its early stages. Many studies have failed to clearly distinguish between stereotypes and stereotypical biases, which has significantly slowed progress in advancing research in this area. Stereotype and anti-stereotype detection is a problem that requires knowledge of society; hence, it is one of the most difficult areas in Responsible AI. This work investigates this task, where we propose a four-tuple definition and provide precise terminology distinguishing stereotype, anti-stereotype, stereotypical bias, and bias, offering valuable insights into their various aspects. In this paper, we propose StereoDetect, a high-quality benchmarking dataset curated for this task by optimally utilizing current datasets such as StereoSet and WinoQueer, involving a manual verification process and the transfer of semantic information. We demonstrate that language models for reasoning with fewer than 10B parameters often get confused when detecting anti-stereotypes. We also demonstrate the critical importance of well-curated datasets by comparing our model with other current models for stereotype detection. The dataset and code is available at this https URL. 

---
# BabyLM's First Words: Word Segmentation as a Phonological Probing Task 

**Authors**: Zébulon Goriely  

**Link**: [PDF](https://arxiv.org/pdf/2504.03338)  

**Abstract**: Language models provide a key framework for studying linguistic theories based on prediction, but phonological analysis using large language models (LLMs) is difficult; there are few phonological benchmarks beyond English and the standard input representation used in LLMs (subwords of graphemes) is not suitable for analyzing the representation of phonemes. In this work, we demonstrate how word segmentation can be used as a phonological probing task, allowing us to study the representations learned by phoneme-based language models trained on child-directed speech across 31 languages. Following computational models of word segmentation, we present unsupervised methods for extracting word boundaries from a trained model using the observation that prediction-error peaks at the start of words. We also use linear probes to identify that these models implicitly track word boundaries, even when they do not appear in training. This cross-lingual work corroborates statistical learning theories of acquisition and empirically motivates new methods for training subword tokenizers. 

---
# Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices 

**Authors**: Luís Couto Seller, Íñigo Sanz Torres, Adrián Vogel-Fernández, Carlos González Carballo, Pedro Miguel Sánchez Sánchez, Adrián Carruana Martín, Enrique de Miguel Ambite  

**Link**: [PDF](https://arxiv.org/pdf/2504.03312)  

**Abstract**: Large Language Models have significantly advanced natural language processing, achieving remarkable performance in tasks such as language generation, translation, and reasoning. However, their substantial computational requirements restrict deployment to high-end systems, limiting accessibility on consumer-grade devices. This challenge is especially pronounced for under-resourced languages like those spoken in the Iberian Peninsula, where relatively limited linguistic resources and benchmarks hinder effective evaluation. This work presents a comprehensive evaluation of compact state-of-the-art LLMs across several essential NLP tasks tailored for Iberian languages. The results reveal that while some models consistently excel in certain tasks, significant performance gaps remain, particularly for languages such as Basque. These findings highlight the need for further research on balancing model compactness with robust multilingual performance 

---
# Noise Augmented Fine Tuning for Mitigating Hallucinations in Large Language Models 

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.03302)  

**Abstract**: Large language models (LLMs) often produce inaccurate or misleading content-hallucinations. To address this challenge, we introduce Noise-Augmented Fine-Tuning (NoiseFiT), a novel framework that leverages adaptive noise injection based on the signal-to-noise ratio (SNR) to enhance model robustness. In particular, NoiseFiT selectively perturbs layers identified as either high-SNR (more robust) or low-SNR (potentially under-regularized) using a dynamically scaled Gaussian noise. We further propose a hybrid loss that combines standard cross-entropy, soft cross-entropy, and consistency regularization to ensure stable and accurate outputs under noisy training conditions. Our theoretical analysis shows that adaptive noise injection is both unbiased and variance-preserving, providing strong guarantees for convergence in expectation. Empirical results on multiple test and benchmark datasets demonstrate that NoiseFiT significantly reduces hallucination rates, often improving or matching baseline performance in key tasks. These findings highlight the promise of noise-driven strategies for achieving robust, trustworthy language modeling without incurring prohibitive computational overhead. Given the comprehensive and detailed nature of our experiments, we have publicly released the fine-tuning logs, benchmark evaluation artifacts, and source code online at W&B, Hugging Face, and GitHub, respectively, to foster further research, accessibility and reproducibility. 

---
# Stance-Driven Multimodal Controlled Statement Generation: New Dataset and Task 

**Authors**: Bingqian Wang, Quan Fang, Jiachen Sun, Xiaoxiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03295)  

**Abstract**: Formulating statements that support diverse or controversial stances on specific topics is vital for platforms that enable user expression, reshape political discourse, and drive social critique and information dissemination. With the rise of Large Language Models (LLMs), controllable text generation towards specific stances has become a promising research area with applications in shaping public opinion and commercial marketing. However, current datasets often focus solely on pure texts, lacking multimodal content and effective context, particularly in the context of stance detection. In this paper, we formally define and study the new problem of stance-driven controllable content generation for tweets with text and images, where given a multimodal post (text and image/video), a model generates a stance-controlled response. To this end, we create the Multimodal Stance Generation Dataset (StanceGen2024), the first resource explicitly designed for multimodal stance-controllable text generation in political discourse. It includes posts and user comments from the 2024 U.S. presidential election, featuring text, images, videos, and stance annotations to explore how multimodal political content shapes stance expression. Furthermore, we propose a Stance-Driven Multimodal Generation (SDMG) framework that integrates weighted fusion of multimodal features and stance guidance to improve semantic consistency and stance control. We release the dataset and code (this https URL) for public use and further research. 

---
# Think When You Need: Self-Adaptive Chain-of-Thought Learning 

**Authors**: Junjie Yang, Ke Lin, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03234)  

**Abstract**: Chain of Thought (CoT) reasoning enhances language models' performance but often leads to inefficient "overthinking" on simple problems. We identify that existing approaches directly penalizing reasoning length fail to account for varying problem complexity. Our approach constructs rewards through length and quality comparisons, guided by theoretical assumptions that jointly enhance solution correctness with conciseness. Moreover, we further demonstrate our method to fuzzy tasks where ground truth is unavailable. Experiments across multiple reasoning benchmarks demonstrate that our method maintains accuracy while generating significantly more concise explanations, effectively teaching models to "think when needed." 

---
# Enhancing Personalized Multi-Turn Dialogue with Curiosity Reward 

**Authors**: Yanming Wan, Jiaxing Wu, Marwa Abdulhai, Lior Shani, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.03206)  

**Abstract**: Effective conversational agents must be able to personalize their behavior to suit a user's preferences, personality, and attributes, whether they are assisting with writing tasks or operating in domains like education or healthcare. Current training methods like Reinforcement Learning from Human Feedback (RLHF) prioritize helpfulness and safety but fall short in fostering truly empathetic, adaptive, and personalized interactions. Traditional approaches to personalization often rely on extensive user history, limiting their effectiveness for new or context-limited users. To overcome these limitations, we propose to incorporate an intrinsic motivation to improve the conversational agents's model of the user as an additional reward alongside multi-turn RLHF. This reward mechanism encourages the agent to actively elicit user traits by optimizing conversations to increase the accuracy of its user model. Consequently, the policy agent can deliver more personalized interactions through obtaining more information about the user. We applied our method both education and fitness settings, where LLMs teach concepts or recommend personalized strategies based on users' hidden learning style or lifestyle attributes. Using LLM-simulated users, our approach outperformed a multi-turn RLHF baseline in revealing information about the users' preferences, and adapting to them. 

---
# Explain with Visual Keypoints Like a Real Mentor! A Benchmark for Multimodal Solution Explanation 

**Authors**: Jaewoo Park, Jungyang Park, Dongju Jang, Jiwan Chung, Byungwoo Yoo, Jaewoo Shin, Seonjoon Park, Taehyeong Kim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03197)  

**Abstract**: With the rapid advancement of mathematical reasoning capabilities in large language models (LLMs), AI systems are increasingly being adopted in educational settings to support students' comprehension of problem-solving processes. However, a critical component remains underexplored in current LLM-generated explanations: visual explanation. In real-world instructional contexts, human tutors routinely employ visual aids-such as diagrams, markings, and highlights-to enhance conceptual clarity. To bridge this gap, we introduce a novel task of visual solution explanation, which requires not only solving problems but also generating explanations that incorporate newly introduced visual elements essential for understanding (e.g., auxiliary lines, annotations, or geometric constructions). To evaluate model performance on this task, we propose MathExplain, a multimodal benchmark consisting of 997 math problems annotated with visual keypoints and corresponding explanatory text that references those elements. Our empirical results show that while some closed-source models demonstrate promising capabilities on visual solution-explaining, current open-source general-purpose models perform inconsistently, particularly in identifying relevant visual components and producing coherent keypoint-based explanations. We expect that visual solution-explaining and the MathExplain dataset will catalyze further research on multimodal LLMs in education and advance their deployment as effective, explanation-oriented AI tutors. Code and data will be released publicly. 

---
# Learning Natural Language Constraints for Safe Reinforcement Learning of Language Agents 

**Authors**: Jaymari Chua, Chen Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03185)  

**Abstract**: Generalizable alignment is a core challenge for deploying Large Language Models (LLMs) safely in real-world NLP applications. Current alignment methods, including Reinforcement Learning from Human Feedback (RLHF), often fail to guarantee constraint satisfaction outside their training distribution due to their reliance on implicit, post-hoc preferences. Inspired by a paradigm shift to first curate data before tuning, we introduce a new framework for safe language alignment that learns natural language constraints from positive and negative demonstrations as a primary step. From inferring both a task-specific reward function and latent constraint functions, our approach fosters adaptation to novel safety requirements and robust generalization under domain shifts and adversarial inputs. We formalize the framework within a Constrained Markov Decision Process (CMDP) and validate it via a text-based navigation environment, demonstrating safe adaptation to changing danger zones. Our experiments show fewer violations upon domain shift when following a safe navigation path, and we achieve zero violations by applying learned constraints to a distilled BERT model as a fine-tuning technique. This work offers a promising path toward building safety-critical and more generalizable LLMs for practical NLP settings. 

---
# Multi-lingual Multi-turn Automated Red Teaming for LLMs 

**Authors**: Abhishek Singhania, Christophe Dupuy, Shivam Mangale, Amani Namboori  

**Link**: [PDF](https://arxiv.org/pdf/2504.03174)  

**Abstract**: Language Model Models (LLMs) have improved dramatically in the past few years, increasing their adoption and the scope of their capabilities over time. A significant amount of work is dedicated to ``model alignment'', i.e., preventing LLMs to generate unsafe responses when deployed into customer-facing applications. One popular method to evaluate safety risks is \textit{red-teaming}, where agents attempt to bypass alignment by crafting elaborate prompts that trigger unsafe responses from a model. Standard human-driven red-teaming is costly, time-consuming and rarely covers all the recent features (e.g., multi-lingual, multi-modal aspects), while proposed automation methods only cover a small subset of LLMs capabilities (i.e., English or single-turn). We present Multi-lingual Multi-turn Automated Red Teaming (\textbf{MM-ART}), a method to fully automate conversational, multi-lingual red-teaming operations and quickly identify prompts leading to unsafe responses. Through extensive experiments on different languages, we show the studied LLMs are on average 71\% more vulnerable after a 5-turn conversation in English than after the initial turn. For conversations in non-English languages, models display up to 195\% more safety vulnerabilities than the standard single-turn English approach, confirming the need for automated red-teaming methods matching LLMs capabilities. 

---
# Efficient Dynamic Clustering-Based Document Compression for Retrieval-Augmented-Generation 

**Authors**: Weitao Li, Kaiming Liu, Xiangyu Zhang, Xuanyu Lei, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03165)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a widely adopted approach for knowledge integration during large language model (LLM) inference in recent years. However, current RAG implementations face challenges in effectively addressing noise, repetition and redundancy in retrieved content, primarily due to their limited ability to exploit fine-grained inter-document relationships. To address these limitations, we propose an \textbf{E}fficient \textbf{D}ynamic \textbf{C}lustering-based document \textbf{C}ompression framework (\textbf{EDC\textsuperscript{2}-RAG}) that effectively utilizes latent inter-document relationships while simultaneously removing irrelevant information and redundant content. We validate our approach, built upon GPT-3.5, on widely used knowledge-QA and hallucination-detected datasets. The results show that this method achieves consistent performance improvements across various scenarios and experimental settings, demonstrating strong robustness and applicability. Our code and datasets can be found at this https URL. 

---
# Beyond the Next Token: Towards Prompt-Robust Zero-Shot Classification via Efficient Multi-Token Prediction 

**Authors**: Junlang Qian, Zixiao Zhu, Hanzhang Zhou, Zijian Feng, Zepeng Zhai, Kezhi Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03159)  

**Abstract**: Zero-shot text classification typically relies on prompt engineering, but the inherent prompt brittleness of large language models undermines its reliability. Minor changes in prompt can cause significant discrepancies in model performance. We attribute this prompt brittleness largely to the narrow focus on nexttoken probabilities in existing methods. To address this, we propose Placeholding Parallel Prediction (P3), a novel approach that predicts token probabilities across multiple positions and simulates comprehensive sampling of generation paths in a single run of a language model. Experiments show improved accuracy and up to 98% reduction in the standard deviation across prompts, boosting robustness. Even without a prompt, P3 maintains comparable performance, reducing the need for prompt engineering. 

---
# Why Reasoning Matters? A Survey of Advancements in Multimodal Reasoning (v1) 

**Authors**: Jing Bi, Susan Liang, Xiaofei Zhou, Pinxin Liu, Junjia Guo, Yunlong Tang, Luchuan Song, Chao Huang, Guangyu Sun, Jinxi He, Jiarui Wu, Shu Yang, Daoan Zhang, Chen Chen, Lianggong Bruce Wen, Zhang Liu, Jiebo Luo, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03151)  

**Abstract**: Reasoning is central to human intelligence, enabling structured problem-solving across diverse tasks. Recent advances in large language models (LLMs) have greatly enhanced their reasoning abilities in arithmetic, commonsense, and symbolic domains. However, effectively extending these capabilities into multimodal contexts-where models must integrate both visual and textual inputs-continues to be a significant challenge. Multimodal reasoning introduces complexities, such as handling conflicting information across modalities, which require models to adopt advanced interpretative strategies. Addressing these challenges involves not only sophisticated algorithms but also robust methodologies for evaluating reasoning accuracy and coherence. This paper offers a concise yet insightful overview of reasoning techniques in both textual and multimodal LLMs. Through a thorough and up-to-date comparison, we clearly formulate core reasoning challenges and opportunities, highlighting practical methods for post-training optimization and test-time inference. Our work provides valuable insights and guidance, bridging theoretical frameworks and practical implementations, and sets clear directions for future research. 

---
# Single-Pass Document Scanning for Question Answering 

**Authors**: Weili Cao, Jianyou Wang, Youze Zheng, Longtian Bao, Qirui Zheng, Taylor Berg-Kirkpatrick, Ramamohan Paturi, Leon Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03101)  

**Abstract**: Handling extremely large documents for question answering is challenging: chunk-based embedding methods often lose track of important global context, while full-context transformers can be prohibitively expensive for hundreds of thousands of tokens. We propose a single-pass document scanning approach that processes the entire text in linear time, preserving global coherence while deciding which sentences are most relevant to the query. On 41 QA benchmarks, our single-pass scanner consistently outperforms chunk-based embedding methods and competes with large language models at a fraction of the computational cost. By conditioning on the entire preceding context without chunk breaks, the method preserves global coherence, which is especially important for long documents. Overall, single-pass document scanning offers a simple solution for question answering over massive text. All code, datasets, and model checkpoints are available at this https URL 

---
# AD-GPT: Large Language Models in Alzheimer's Disease 

**Authors**: Ziyu Liu, Lintao Tang, Zeliang Sun, Zhengliang Liu, Yanjun Lyu, Wei Ruan, Yangshuang Xu, Liang Shan, Jiyoon Shin, Xiaohe Chen, Dajiang Zhu, Tianming Liu, Rongjie Liu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03071)  

**Abstract**: Large language models (LLMs) have emerged as powerful tools for medical information retrieval, yet their accuracy and depth remain limited in specialized domains such as Alzheimer's disease (AD), a growing global health challenge. To address this gap, we introduce AD-GPT, a domain-specific generative pre-trained transformer designed to enhance the retrieval and analysis of AD-related genetic and neurobiological information. AD-GPT integrates diverse biomedical data sources, including potential AD-associated genes, molecular genetic information, and key gene variants linked to brain regions. We develop a stacked LLM architecture combining Llama3 and BERT, optimized for four critical tasks in AD research: (1) genetic information retrieval, (2) gene-brain region relationship assessment, (3) gene-AD relationship analysis, and (4) brain region-AD relationship mapping. Comparative evaluations against state-of-the-art LLMs demonstrate AD-GPT's superior precision and reliability across these tasks, underscoring its potential as a robust and specialized AI tool for advancing AD research and biomarker discovery. 

---
# Task as Context Prompting for Accurate Medical Symptom Coding Using Large Language Models 

**Authors**: Chengyang He, Wenlong Zhang, Violet Xinying Chen, Yue Ning, Ping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.03051)  

**Abstract**: Accurate medical symptom coding from unstructured clinical text, such as vaccine safety reports, is a critical task with applications in pharmacovigilance and safety monitoring. Symptom coding, as tailored in this study, involves identifying and linking nuanced symptom mentions to standardized vocabularies like MedDRA, differentiating it from broader medical coding tasks. Traditional approaches to this task, which treat symptom extraction and linking as independent workflows, often fail to handle the variability and complexity of clinical narratives, especially for rare cases. Recent advancements in Large Language Models (LLMs) offer new opportunities but face challenges in achieving consistent performance. To address these issues, we propose Task as Context (TACO) Prompting, a novel framework that unifies extraction and linking tasks by embedding task-specific context into LLM prompts. Our study also introduces SYMPCODER, a human-annotated dataset derived from Vaccine Adverse Event Reporting System (VAERS) reports, and a two-stage evaluation framework to comprehensively assess both symptom linking and mention fidelity. Our comprehensive evaluation of multiple LLMs, including Llama2-chat, Jackalope-7b, GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o, demonstrates TACO's effectiveness in improving flexibility and accuracy for tailored tasks like symptom coding, paving the way for more specific coding tasks and advancing clinical text processing methodologies. 

---
# Extending CREAMT: Leveraging Large Language Models for Literary Translation Post-Editing 

**Authors**: Antonio Castaldo, Sheila Castilho, Joss Moorkens, Johanna Monti  

**Link**: [PDF](https://arxiv.org/pdf/2504.03045)  

**Abstract**: Post-editing machine translation (MT) for creative texts, such as literature, requires balancing efficiency with the preservation of creativity and style. While neural MT systems struggle with these challenges, large language models (LLMs) offer improved capabilities for context-aware and creative translation. This study evaluates the feasibility of post-editing literary translations generated by LLMs. Using a custom research tool, we collaborated with professional literary translators to analyze editing time, quality, and creativity. Our results indicate that post-editing LLM-generated translations significantly reduces editing time compared to human translation while maintaining a similar level of creativity. The minimal difference in creativity between PE and MT, combined with substantial productivity gains, suggests that LLMs may effectively support literary translators working with high-resource languages. 

---
# IPA-CHILDES & G2P+: Feature-Rich Resources for Cross-Lingual Phonology and Phonemic Language Modeling 

**Authors**: Zébulon Goriely, Paula Buttery  

**Link**: [PDF](https://arxiv.org/pdf/2504.03036)  

**Abstract**: In this paper, we introduce two resources: (i) G2P+, a tool for converting orthographic datasets to a consistent phonemic representation; and (ii) IPA CHILDES, a phonemic dataset of child-centered speech across 31 languages. Prior tools for grapheme-to-phoneme conversion result in phonemic vocabularies that are inconsistent with established phonemic inventories, an issue which G2P+ addresses by leveraging the inventories in the Phoible database. Using this tool, we augment CHILDES with phonemic transcriptions to produce IPA CHILDES. This new resource fills several gaps in existing phonemic datasets, which often lack multilingual coverage, spontaneous speech, and a focus on child-directed language. We demonstrate the utility of this dataset for phonological research by training phoneme language models on 11 languages and probing them for distinctive features, finding that the distributional properties of phonemes are sufficient to learn major class and place features cross-lingually. 

---
# The Dual-Route Model of Induction 

**Authors**: Sheridan Feucht, Eric Todd, Byron Wallace, David Bau  

**Link**: [PDF](https://arxiv.org/pdf/2504.03022)  

**Abstract**: Prior work on in-context copying has shown the existence of induction heads, which attend to and promote individual tokens during copying. In this work we introduce a new type of induction head: concept-level induction heads, which copy entire lexical units instead of individual tokens. Concept induction heads learn to attend to the ends of multi-token words throughout training, working in parallel with token-level induction heads to copy meaningful text. We show that these heads are responsible for semantic tasks like word-level translation, whereas token induction heads are vital for tasks that can only be done verbatim, like copying nonsense tokens. These two "routes" operate independently: in fact, we show that ablation of token induction heads causes models to paraphrase where they would otherwise copy verbatim. In light of these findings, we argue that although token induction heads are vital for specific tasks, concept induction heads may be more broadly relevant for in-context learning. 

---
# Hummus: A Dataset of Humorous Multimodal Metaphor Use 

**Authors**: Xiaoyu Tong, Zhi Zhang, Martha Lewis, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2504.02983)  

**Abstract**: Metaphor and humor share a lot of common ground, and metaphor is one of the most common humorous mechanisms. This study focuses on the humorous capacity of multimodal metaphors, which has not received due attention in the community. We take inspiration from the Incongruity Theory of humor, the Conceptual Metaphor Theory, and the annotation scheme behind the VU Amsterdam Metaphor Corpus, and developed a novel annotation scheme for humorous multimodal metaphor use in image-caption pairs. We create the Hummus Dataset of Humorous Multimodal Metaphor Use, providing expert annotation on 1k image-caption pairs sampled from the New Yorker Caption Contest corpus. Using the dataset, we test state-of-the-art multimodal large language models (MLLMs) on their ability to detect and understand humorous multimodal metaphor use. Our experiments show that current MLLMs still struggle with processing humorous multimodal metaphors, particularly with regard to integrating visual and textual information. We release our dataset and code at this http URL. 

---
# A Bayesian account of pronoun and neopronoun acquisition 

**Authors**: Cassandra L. Jacobs, Morgan Grobol  

**Link**: [PDF](https://arxiv.org/pdf/2504.02973)  

**Abstract**: A major challenge to equity among members of queer communities is the use of one's chosen forms of reference, such as personal names or pronouns. Speakers often dismiss their misuses of pronouns as "unintentional", and claim that their errors reflect many decades of fossilized mainstream language use, as well as attitudes or expectations about the relationship between one's appearance and acceptable forms of reference. We argue for explicitly modeling individual differences in pronoun selection and present a probabilistic graphical modeling approach based on the nested Chinese Restaurant Franchise Process (nCRFP) (Ahmed et al., 2013) to account for flexible pronominal reference such as chosen names and neopronouns while moving beyond form-to-meaning mappings and without lexical co-occurrence statistics to learn referring expressions, as in contemporary language models. We show that such a model can account for variability in how quickly pronouns or names are integrated into symbolic knowledge and can empower computational systems to be both flexible and respectful of queer people with diverse gender expression. 

---
# CoLa -- Learning to Interactively Collaborate with Large LMs 

**Authors**: Abhishek Sharma, Dan Goldwasser  

**Link**: [PDF](https://arxiv.org/pdf/2504.02965)  

**Abstract**: LLMs' remarkable ability to tackle a wide range of language tasks opened new opportunities for collaborative human-AI problem solving. LLMs can amplify human capabilities by applying their intuitions and reasoning strategies at scale. We explore whether human guides can be simulated, by generalizing from human demonstrations of guiding an AI system to solve complex language problems. We introduce CoLa, a novel self-guided learning paradigm for training automated $\textit{guides}$ and evaluate it on two QA datasets, a puzzle-solving task, and a constrained text generation task. Our empirical results show that CoLa consistently outperforms competitive approaches across all domains. Moreover, a small-sized trained guide outperforms a strong model like GPT-4 when acting as a guide. We compare the strategies employed by humans and automated guides by conducting a human study on a QA dataset. We show that automated guides outperform humans by adapting their strategies to reasoners' capabilities and conduct qualitative analyses highlighting distinct differences in guiding strategies. 

---
# Understanding Aha Moments: from External Observations to Internal Mechanisms 

**Authors**: Shu Yang, Junchao Wu, Xin Chen, Yunze Xiao, Xinyi Yang, Derek F. Wong, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02956)  

**Abstract**: Large Reasoning Models (LRMs), capable of reasoning through complex problems, have become crucial for tasks like programming, mathematics, and commonsense reasoning. However, a key challenge lies in understanding how these models acquire reasoning capabilities and exhibit "aha moments" when they reorganize their methods to allocate more thinking time to problems. In this work, we systematically study "aha moments" in LRMs, from linguistic patterns, description of uncertainty, "Reasoning Collapse" to analysis in latent space. We demonstrate that the "aha moment" is externally manifested in a more frequent use of anthropomorphic tones for self-reflection and an adaptive adjustment of uncertainty based on problem difficulty. This process helps the model complete reasoning without succumbing to "Reasoning Collapse". Internally, it corresponds to a separation between anthropomorphic characteristics and pure reasoning, with an increased anthropomorphic tone for more difficult problems. Furthermore, we find that the "aha moment" helps models solve complex problems by altering their perception of problem difficulty. As the layer of the model increases, simpler problems tend to be perceived as more complex, while more difficult problems appear simpler. 

---
# Cultural Learning-Based Culture Adaptation of Language Models 

**Authors**: Chen Cecilia Liu, Anna Korhonen, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2504.02953)  

**Abstract**: Adapting large language models (LLMs) to diverse cultural values is a challenging task, as existing LLMs often reflect the values of specific groups by default, and potentially causing harm to others. In this paper, we present CLCA, a novel framework for enhancing LLM alignment with cultural values based on cultural learning. The framework leverages simulated social interactions to generate conversations in which LLMs engage in role-playing within culturally adapted social scenarios, capturing implicit cultural norms for model fine-tuning. CLCA improves cultural value alignment across various model architectures measured using World Value Survey data, demonstrating the effectiveness of our proposed approach. Our results provide early evidence that understanding intent and social interactions can enhance cultural value adaptation in LLMs, highlighting the promise of training approaches based on cultural learning. 

---
# HyperRAG: Enhancing Quality-Efficiency Tradeoffs in Retrieval-Augmented Generation with Reranker KV-Cache Reuse 

**Authors**: Yuwei An, Yihua Cheng, Seo Jin Park, Junchen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02921)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing the performance of large language models (LLMs) by integrating external knowledge into the generation process. A key component of RAG pipelines is the reranker, which selects the most relevant documents from a pool of retrieved candidates and significantly improves the quality of the generated responses. While rerankers refine the selection of retrieved documents in RAG pipelines, they introduce computational challenges that hinder high throughput and low latency. To address this problem, we propose HyperRAG, a system that optimizes the trade-off between quality and efficiency in RAG pipelines by leveraging KV-cache reuse for efficient reranker inference. By reusing document-side KV-cache, HyperRAG achieves both high-quality generation and system-level efficiency. To fully realize the benefits of KV-cache reuse, HyperRAG incorporates a range of system-level optimizations designed to enhance efficiency and scalability. Experiments show that HyperRAG achieves a 2 - 3 throughput improvement with decoder-only rerankers while also delivering higher downstream performance compared with traditional RAG service. 

---
# Bias in Large Language Models Across Clinical Applications: A Systematic Review 

**Authors**: Thanathip Suenghataiphorn, Narisara Tribuddharat, Pojsakorn Danpanichkul, Narathorn Kulthamrongsri  

**Link**: [PDF](https://arxiv.org/pdf/2504.02917)  

**Abstract**: Background: Large language models (LLMs) are rapidly being integrated into healthcare, promising to enhance various clinical tasks. However, concerns exist regarding their potential for bias, which could compromise patient care and exacerbate health inequities. This systematic review investigates the prevalence, sources, manifestations, and clinical implications of bias in LLMs. Methods: We conducted a systematic search of PubMed, OVID, and EMBASE from database inception through 2025, for studies evaluating bias in LLMs applied to clinical tasks. We extracted data on LLM type, bias source, bias manifestation, affected attributes, clinical task, evaluation methods, and outcomes. Risk of bias was assessed using a modified ROBINS-I tool. Results: Thirty-eight studies met inclusion criteria, revealing pervasive bias across various LLMs and clinical applications. Both data-related bias (from biased training data) and model-related bias (from model training) were significant contributors. Biases manifested as: allocative harm (e.g., differential treatment recommendations); representational harm (e.g., stereotypical associations, biased image generation); and performance disparities (e.g., variable output quality). These biases affected multiple attributes, most frequently race/ethnicity and gender, but also age, disability, and language. Conclusions: Bias in clinical LLMs is a pervasive and systemic issue, with a potential to lead to misdiagnosis and inappropriate treatment, particularly for marginalized patient populations. Rigorous evaluation of the model is crucial. Furthermore, the development and implementation of effective mitigation strategies, coupled with continuous monitoring in real-world clinical settings, are essential to ensure the safe, equitable, and trustworthy deployment of LLMs in healthcare. 

---
# Noiser: Bounded Input Perturbations for Attributing Large Language Models 

**Authors**: Mohammad Reza Ghasemi Madani, Aryo Pradipta Gema, Gabriele Sarti, Yu Zhao, Pasquale Minervini, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2504.02911)  

**Abstract**: Feature attribution (FA) methods are common post-hoc approaches that explain how Large Language Models (LLMs) make predictions. Accordingly, generating faithful attributions that reflect the actual inner behavior of the model is crucial. In this paper, we introduce Noiser, a perturbation-based FA method that imposes bounded noise on each input embedding and measures the robustness of the model against partially noised input to obtain the input attributions. Additionally, we propose an answerability metric that employs an instructed judge model to assess the extent to which highly scored tokens suffice to recover the predicted output. Through a comprehensive evaluation across six LLMs and three tasks, we demonstrate that Noiser consistently outperforms existing gradient-based, attention-based, and perturbation-based FA methods in terms of both faithfulness and answerability, making it a robust and effective approach for explaining language model predictions. 

---
# Enhancing Chart-to-Code Generation in Multimodal Large Language Models via Iterative Dual Preference Learning 

**Authors**: Zhihan Zhang, Yixin Cao, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02906)  

**Abstract**: Chart-to-code generation, the process of converting chart images into executable plotting scripts, provides a lossless representation of chart information, requiring models to accurately capture and summarize all visual and structural elements. However, this remains a significant challenge for multimodal large language models (MLLMs), which are not inherently well-aligned with code generation tasks. To bridge this gap, we introduce Chart2Code, a novel iterative dual preference learning framework designed to enhance MLLMs' chart-to-code generation capabilities through structured code variant generation and fine-grained dual reward signals. We validate Chart2Code across three MLLMs and find that iterative preference learning consistently improves out-of-distribution chart-to-code generation quality. Throughout this process, our dual scoring method, which evaluates both the textual code structure and its visual representation, leads to greater performance improvements, even with a reduced preference dataset size. Further analysis explores the key components of our framework and highlights the interplay between chart-to-code generation and broader chart reasoning, paving the way for future advancements in chart comprehension. 

---
# How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence 

**Authors**: Hongzhe Du, Weikai Li, Min Cai, Karim Saraipour, Zimin Zhang, Himabindu Lakkaraju, Yizhou Sun, Shichang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02904)  

**Abstract**: Post-training is essential for the success of large language models (LLMs), transforming pre-trained base models into more useful and aligned post-trained models. While plenty of works have studied post-training algorithms and evaluated post-training models by their outputs, it remains understudied how post-training reshapes LLMs internally. In this paper, we compare base and post-trained LLMs mechanistically from four perspectives to better understand post-training effects. Our findings across model families and datasets reveal that: (1) Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations; (2) Both truthfulness and refusal can be represented by linear vectors in the hidden representation space. The truthfulness direction is highly similar between the base and post-trained model, and it is effectively transferable for interventions; (3) The refusal direction is different between the base and post-trained models, and it shows limited forward transferability; (4) Differences in confidence between the base and post-trained models cannot be attributed to entropy neurons. Our study provides insights into the fundamental mechanisms preserved and altered during post-training, facilitates downstream tasks like model steering, and could potentially benefit future research in interpretability and LLM post-training. 

---
# Beyond Accuracy: The Role of Calibration in Self-Improving Large Language Models 

**Authors**: Liangjie Huang, Dawei Li, Huan Liu, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.02902)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable self-improvement capabilities, whereby models iteratively revise their outputs through self-generated feedback. While this reflective mechanism has shown promise in enhancing task performance, recent studies suggest that it may also introduce undesirable biases-most notably, self-bias, or the tendency of LLMs to favor their own prior outputs. In this work, we extend this line of inquiry by investigating the impact on confidence estimation. We evaluate three representative self-improvement paradigms-basic prompting, Chain-of-Thought (CoT) prompting, and tuning-based methods and find that iterative self-improvement can lead to systematic overconfidence, as evidenced by a steadily increasing Expected Calibration Error (ECE) and lower accuracy with high confidence. We then further explore the integration of confidence calibration techniques with self-improvement. Specifically, we compare three strategies: (1) applying calibration after multiple rounds of self-improvement, (2) calibrating before self-improvement, and (3) applying calibration iteratively at each self-improvement step. Our results show that iterative calibration is most effective in reducing ECE, yielding improved calibration. Our work pioneers the study of self-improving LLMs from a calibration perspective, offering valuable insights into balancing model performance and reliability. 

---
# A Practical Synthesis of Detecting AI-Generated Textual, Visual, and Audio Content 

**Authors**: Lele Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02898)  

**Abstract**: Advances in AI-generated content have led to wide adoption of large language models, diffusion-based visual generators, and synthetic audio tools. However, these developments raise critical concerns about misinformation, copyright infringement, security threats, and the erosion of public trust. In this paper, we explore an extensive range of methods designed to detect and mitigate AI-generated textual, visual, and audio content. We begin by discussing motivations and potential impacts associated with AI-based content generation, including real-world risks and ethical dilemmas. We then outline detection techniques spanning observation-based strategies, linguistic and statistical analysis, model-based pipelines, watermarking and fingerprinting, as well as emergent ensemble approaches. We also present new perspectives on robustness, adaptation to rapidly improving generative architectures, and the critical role of human-in-the-loop verification. By surveying state-of-the-art research and highlighting case studies in academic, journalistic, legal, and industrial contexts, this paper aims to inform robust solutions and policymaking. We conclude by discussing open challenges, including adversarial transformations, domain generalization, and ethical concerns, thereby offering a holistic guide for researchers, practitioners, and regulators to preserve content authenticity in the face of increasingly sophisticated AI-generated media. 

---
# OnRL-RAG: Real-Time Personalized Mental Health Dialogue System 

**Authors**: Ahsan Bilal, Beiyu Lin, Mehdi Zaeifi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02894)  

**Abstract**: Large language models (LLMs) have been widely used for various tasks and applications. However, LLMs and fine-tuning are limited to the pre-trained data. For example, ChatGPT's world knowledge until 2021 can be outdated or inaccurate. To enhance the capabilities of LLMs, Retrieval-Augmented Generation (RAG), is proposed to augment LLMs with additional, new, latest details and information to LLMs. While RAG offers the correct information, it may not best present it, especially to different population groups with personalizations. Reinforcement Learning from Human Feedback (RLHF) adapts to user needs by aligning model responses with human preference through feedback loops. In real-life applications, such as mental health problems, a dynamic and feedback-based model would continuously adapt to new information and offer personalized assistance due to complex factors fluctuating in a daily environment. Thus, we propose an Online Reinforcement Learning-based Retrieval-Augmented Generation (OnRL-RAG) system to detect and personalize the responding systems to mental health problems, such as stress, anxiety, and depression. We use an open-source dataset collected from 2028 College Students with 28 survey questions for each student to demonstrate the performance of our proposed system with the existing systems. Our system achieves superior performance compared to standard RAG and simple LLM via GPT-4o, GPT-4o-mini, Gemini-1.5, and GPT-3.5. This work would open up the possibilities of real-life applications of LLMs for personalized services in the everyday environment. The results will also help researchers in the fields of sociology, psychology, and neuroscience to align their theories more closely with the actual human daily environment. 

---
# Automated Survey Collection with LLM-based Conversational Agents 

**Authors**: Kurmanbek Kaiyrbekov, Nicholas J Dobbins, Sean D Mooney  

**Link**: [PDF](https://arxiv.org/pdf/2504.02891)  

**Abstract**: Objective: Traditional phone-based surveys are among the most accessible and widely used methods to collect biomedical and healthcare data, however, they are often costly, labor intensive, and difficult to scale effectively. To overcome these limitations, we propose an end-to-end survey collection framework driven by conversational Large Language Models (LLMs).
Materials and Methods: Our framework consists of a researcher responsible for designing the survey and recruiting participants, a conversational phone agent powered by an LLM that calls participants and administers the survey, a second LLM (GPT-4o) that analyzes the conversation transcripts generated during the surveys, and a database for storing and organizing the results. To test our framework, we recruited 8 participants consisting of 5 native and 3 non-native english speakers and administered 40 surveys. We evaluated the correctness of LLM-generated conversation transcripts, accuracy of survey responses inferred by GPT-4o and overall participant experience.
Results: Survey responses were successfully extracted by GPT-4o from conversation transcripts with an average accuracy of 98% despite transcripts exhibiting an average per-line word error rate of 7.7%. While participants noted occasional errors made by the conversational LLM agent, they reported that the agent effectively conveyed the purpose of the survey, demonstrated good comprehension, and maintained an engaging interaction.
Conclusions: Our study highlights the potential of LLM agents in conducting and analyzing phone surveys for healthcare applications. By reducing the workload on human interviewers and offering a scalable solution, this approach paves the way for real-world, end-to-end AI-powered phone survey collection systems. 

---
# Scaling Test-time Compute for Low-resource Languages: Multilingual Reasoning in LLMs 

**Authors**: Khanh-Tung Tran, Barry O'Sullivan, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02890)  

**Abstract**: Recent advances in test-time compute scaling have enabled Large Language Models (LLMs) to tackle deep reasoning tasks by generating a chain-of-thought (CoT) that includes trial and error, backtracking, and intermediate reasoning steps before producing the final answer. However, these techniques have been applied predominantly to popular languages, such as English, leaving reasoning in low-resource languages underexplored and misaligned. In this work, we investigate the multilingual mechanism by which LLMs internally operate in a latent space biased toward their inherently dominant language. To leverage this phenomenon for low-resource languages, we train models to generate the CoT in English while outputting the final response in the target language, given input in the low-resource language. Our experiments demonstrate that this approach, named English-Pivoted CoT Training, outperforms other baselines, including training to generate both the CoT and the final response solely in the target language, with up to 28.33% improvement. Further analysis provides novel insights into the relationships between reasoning and multilinguality of LLMs, prompting for better approaches in developing multilingual large reasoning models 

---
# A Status Quo Investigation of Large Language Models towards Cost-Effective CFD Automation with OpenFOAMGPT: ChatGPT vs. Qwen vs. Deepseek 

**Authors**: Wenkang Wang, Ran Xu, Jingsen Feng, Qingfu Zhang, Xu Chu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02888)  

**Abstract**: We evaluated the performance of OpenFOAMGPT incorporating multiple large-language models. Some of the present models efficiently manage different CFD tasks such as adjusting boundary conditions, turbulence models, and solver configurations, although their token cost and stability vary. Locally deployed smaller models like QwQ-32B struggled with generating valid solver files for complex processes. Zero-shot prompting commonly failed in simulations with intricate settings, even for large models. Challenges with boundary conditions and solver keywords stress the requirement for expert supervision, indicating that further development is needed to fully automate specialized CFD simulations. 

---
# Processes Matter: How ML/GAI Approaches Could Support Open Qualitative Coding of Online Discourse Datasets 

**Authors**: John Chen, Alexandros Lotsos, Grace Wang, Lexie Zhao, Bruce Sherin, Uri Wilensky, Michael Horn  

**Link**: [PDF](https://arxiv.org/pdf/2504.02887)  

**Abstract**: Open coding, a key inductive step in qualitative research, discovers and constructs concepts from human datasets. However, capturing extensive and nuanced aspects or "coding moments" can be challenging, especially with large discourse datasets. While some studies explore machine learning (ML)/Generative AI (GAI)'s potential for open coding, few evaluation studies exist. We compare open coding results by five recently published ML/GAI approaches and four human coders, using a dataset of online chat messages around a mobile learning software. Our systematic analysis reveals ML/GAI approaches' strengths and weaknesses, uncovering the complementary potential between humans and AI. Line-by-line AI approaches effectively identify content-based codes, while humans excel in interpreting conversational dynamics. We discussed how embedded analytical processes could shape the results of ML/GAI approaches. Instead of replacing humans in open coding, researchers should integrate AI with and according to their analytical processes, e.g., as parallel co-coders. 

---
# LVMed-R2: Perception and Reflection-driven Complex Reasoning for Medical Report Generation 

**Authors**: Hao Wang, Shuchang Ye, Jinghao Lin, Usman Naseem, Jinman Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.02885)  

**Abstract**: Large vision-language models (LVMs) hold a great promise for automating medical report generation, potentially reducing the burden of manual reporting. State-of-the-art (SOTA) research fine-tunes general LVMs with medical data to align radiology images to corresponding medical reports. However, there are two key factors that limit these LVM's performance. Firstly, LVMs lack complex reasoning capability that leads to logical inconsistencies and potential diagnostic errors in generated reports. Secondly, LVMs lack reflection mechanism that leads to an inability to discover errors in the thinking process. To address these gaps, we propose LVMed-R2, a new fine-tuning strategy that introduces complex reasoning and reflection mechanisms for LVMs to enhance medical report generation. To the best of our knowledge, this is the first work to introduce complex reasoning to the medical report generation (MRG) task. Our proposed complex reasoning contains medical knowledge injection and perception-enhancing modules which improve the accuracy of LVMs diagnosis, coupled with a perception tree to provide guidance to limit the perception range. Further, the reflection mechanism forces self-verification for outputs to correct for potential errors. We experimented by fine-tuning LVMs with our proposed LVMed-R2 strategy, using IU-Xray and MIMIC-CXR datasets. Our results, measured on natural language generation (NLG) metrics and clinical efficacy (CE) metrics, demonstrate that LVMs fine-tuned with the proposed reflection mechanism possess the ability to correct outputs and complex reasoning effectively and improve LVMs performance for MRG. 

---
# SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models 

**Authors**: Anil Ramakrishna, Yixin Wan, Xiaomeng Jin, Kai-Wei Chang, Zhiqi Bu, Bhanukiran Vinzamuri, Volkan Cevher, Mingyi Hong, Rahul Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.02883)  

**Abstract**: We introduce SemEval-2025 Task 4: unlearning sensitive content from Large Language Models (LLMs). The task features 3 subtasks for LLM unlearning spanning different use cases: (1) unlearn long form synthetic creative documents spanning different genres; (2) unlearn short form synthetic biographies containing personally identifiable information (PII), including fake names, phone number, SSN, email and home addresses, and (3) unlearn real documents sampled from the target model's training dataset. We received over 100 submissions from over 30 institutions and we summarize the key techniques and lessons in this paper. 

---
# DiaTool-DPO: Multi-Turn Direct Preference Optimization for Tool-Augmented Large Language Models 

**Authors**: Sunghee Jung, Donghun Lee, Shinbok Lee, Gaeun Seo, Daniel Lee, Byeongil Ko, Junrae Cho, Kihyun Kim, Eunggyun Kim, Myeongcheol Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.02882)  

**Abstract**: Tool-Augmented Larage Language Models (TA-LLMs) have shown promise in real-world applications, but face challenges in handling incomplete queries and out-of-scope requests. While existing approaches rely mainly on Supervised Fine-Tuning with expert trajectories, we propose DiaTool-DPO, a novel method that enhances TA-LLM's dialogue capabilities through Direct Preference Optimization. We model TA-LLM interactions as a Markov Decision Process with 5 distinct dialogue states and categorize user queries into 3 types based on their state transition trajectories. We automatically construct paired trajectory datasets of correct and incorrect dialogue flows and introduce a specialized objective loss for dialogue control. Our comprehensive evaluation demonstrates that DiaTool-DPO approaches GPT-4o's performance (94.8% in information gathering, 91% in tool call rejection) with substantial improvements over baseline (44% and 9.6% respectively) while maintaining core functionality. Our approach opens new possibilities for developing TA-LLMs that can handle diverse real-world scenarios without requiring additional expert demonstrations or human labeling. 

---
# Better Bill GPT: Comparing Large Language Models against Legal Invoice Reviewers 

**Authors**: Nick Whitehouse, Nicole Lincoln, Stephanie Yiu, Lizzie Catterson, Rivindu Perera  

**Link**: [PDF](https://arxiv.org/pdf/2504.02881)  

**Abstract**: Legal invoice review is a costly, inconsistent, and time-consuming process, traditionally performed by Legal Operations, Lawyers or Billing Specialists who scrutinise billing compliance line by line. This study presents the first empirical comparison of Large Language Models (LLMs) against human invoice reviewers - Early-Career Lawyers, Experienced Lawyers, and Legal Operations Professionals-assessing their accuracy, speed, and cost-effectiveness. Benchmarking state-of-the-art LLMs against a ground truth set by expert legal professionals, our empirically substantiated findings reveal that LLMs decisively outperform humans across every metric. In invoice approval decisions, LLMs achieve up to 92% accuracy, surpassing the 72% ceiling set by experienced lawyers. On a granular level, LLMs dominate line-item classification, with top models reaching F-scores of 81%, compared to just 43% for the best-performing human group. Speed comparisons are even more striking - while lawyers take 194 to 316 seconds per invoice, LLMs are capable of completing reviews in as fast as 3.6 seconds. And cost? AI slashes review expenses by 99.97%, reducing invoice processing costs from an average of $4.27 per invoice for human invoice reviewers to mere cents. These results highlight the evolving role of AI in legal spend management. As law firms and corporate legal departments struggle with inefficiencies, this study signals a seismic shift: The era of LLM-powered legal spend management is not on the horizon, it has arrived. The challenge ahead is not whether AI can perform as well as human reviewers, but how legal teams will strategically incorporate it, balancing automation with human discretion. 

---
# Revisiting Funnel Transformers for Modern LLM Architectures with Comprehensive Ablations in Training and Inference Configurations 

**Authors**: DongHyun Choi, Lucas Spangher, Chris Hidey, Peter Grabowski, Ramy Eskander  

**Link**: [PDF](https://arxiv.org/pdf/2504.02877)  

**Abstract**: Transformer-based Large Language Models, which suffer from high computational costs, advance so quickly that techniques proposed to streamline earlier iterations are not guaranteed to benefit more modern models. Building upon the Funnel Transformer proposed by Dai and Le (2020), which progressively compresses intermediate representations, we investigate the impact of funneling in contemporary Gemma2 Transformer architectures. We systematically evaluate various funnel configurations and recovery methods, comparing: (1) standard pretraining to funnel-aware pretraining strategies, (2) the impact of funnel-aware fine-tuning, and (3) the type of sequence recovery operation. Our results demonstrate that funneling creates information bottlenecks that propagate through deeper network layers, particularly in larger models (e.g., Gemma 7B), leading to at times unmanageable performance lost. However, carefully selecting the funneling layer and employing effective recovery strategies, can substantially mitigate performance losses, achieving up to a 44\% reduction in latency. Our findings highlight key trade-offs between computational efficiency and model accuracy, providing practical guidance for deploying funnel-based approaches in large-scale natural language applications. 

---
# TheBlueScrubs-v1, a comprehensive curated medical dataset derived from the internet 

**Authors**: Luis Felipe, Carlos Garcia, Issam El Naqa, Monique Shotande, Aakash Tripathi, Vivek Rudrapatna, Ghulam Rasool, Danielle Bitterman, Gilmer Valdes  

**Link**: [PDF](https://arxiv.org/pdf/2504.02874)  

**Abstract**: The need for robust and diverse data sets to train clinical large language models (cLLMs) is critical given that currently available public repositories often prove too limited in size or scope for comprehensive medical use. While resources like PubMed provide foundational medical literature, they capture only a narrow range of formal publications and omit the broader medical discourse on the internet. To address these deficits, we introduce TheBlueScrubs-v1, a curated dataset of over 25 billion medical tokens - nearly three times larger than PubMed - drawn from a broad-scale internet corpus. Our two-stage filtering pipeline employs a Logistic Regression model for document screening (achieving an AUC of approximately 0.95 on external validation), followed by verification via a 70B-parameter Llama 3.1 instruct model. Each text is assigned three LLM-based quality scores encompassing medical relevance, precision and factual detail, and safety and ethical standards. Clinician reviews confirm high concordance with these automated evaluations, and a specialized cancer classifier further labels approximately 11 billion oncology tokens. Two demonstration tasks highlight the dataset's practical value: first, we distill the safety evaluations to a smaller BERT-style model that reaches an AUC near 0.96 on unseen data; second, we fine-tune a compact LLM on a filtered subset, showing measurable improvements over standard baselines in medical benchmarks as well as private ones. This Data Descriptor details the dataset's creation and validation, underscoring its potential utility for medical AI research. 

---
# Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion 

**Authors**: Dongjun Wei, Minjia Mao, Xiao Fang, Michael Chau  

**Link**: [PDF](https://arxiv.org/pdf/2504.02873)  

**Abstract**: The malicious usage of large language models (LLMs) has motivated the detection of LLM-generated texts. Previous work in topological data analysis shows that the persistent homology dimension (PHD) of text embeddings can serve as a more robust and promising score than other zero-shot methods. However, effectively detecting short LLM-generated texts remains a challenge. This paper presents Short-PHD, a zero-shot LLM-generated text detection method tailored for short texts. Short-PHD stabilizes the estimation of the previous PHD method for short texts by inserting off-topic content before the given input text and identifies LLM-generated text based on an established detection threshold. Experimental results on both public and generated datasets demonstrate that Short-PHD outperforms existing zero-shot methods in short LLM-generated text detection. Implementation codes are available online. 

---
# Scraping the Shadows: Deep Learning Breakthroughs in Dark Web Intelligence 

**Authors**: Ingmar Bakermans, Daniel De Pascale, Gonçalo Marcelino, Giuseppe Cascavilla, Zeno Geradts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02872)  

**Abstract**: Darknet markets (DNMs) facilitate the trade of illegal goods on a global scale. Gathering data on DNMs is critical to ensuring law enforcement agencies can effectively combat crime. Manually extracting data from DNMs is an error-prone and time-consuming task. Aiming to automate this process we develop a framework for extracting data from DNMs and evaluate the application of three state-of-the-art Named Entity Recognition (NER) models, ELMo-BiLSTM \citep{ShahEtAl2022}, UniversalNER \citep{ZhouEtAl2024}, and GLiNER \citep{ZaratianaEtAl2023}, at the task of extracting complex entities from DNM product listing pages. We propose a new annotated dataset, which we use to train, fine-tune, and evaluate the models. Our findings show that state-of-the-art NER models perform well in information extraction from DNMs, achieving 91% Precision, 96% Recall, and an F1 score of 94%. In addition, fine-tuning enhances model performance, with UniversalNER achieving the best performance. 

---
# Synthesized Annotation Guidelines are Knowledge-Lite Boosters for Clinical Information Extraction 

**Authors**: Enshuo Hsu, Martin Ugbala, Krishna Kumar Kookal, Zouaidi Kawtar, Nicholas L. Rider, Muhammad F. Walji, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2504.02871)  

**Abstract**: Generative information extraction using large language models, particularly through few-shot learning, has become a popular method. Recent studies indicate that providing a detailed, human-readable guideline-similar to the annotation guidelines traditionally used for training human annotators can significantly improve performance. However, constructing these guidelines is both labor- and knowledge-intensive. Additionally, the definitions are often tailored to meet specific needs, making them highly task-specific and often non-reusable. Handling these subtle differences requires considerable effort and attention to detail. In this study, we propose a self-improving method that harvests the knowledge summarization and text generation capacity of LLMs to synthesize annotation guidelines while requiring virtually no human input. Our zero-shot experiments on the clinical named entity recognition benchmarks, 2012 i2b2 EVENT, 2012 i2b2 TIMEX, 2014 i2b2, and 2018 n2c2 showed 25.86%, 4.36%, 0.20%, and 7.75% improvements in strict F1 scores from the no-guideline baseline. The LLM-synthesized guidelines showed equivalent or better performance compared to human-written guidelines by 1.15% to 4.14% in most tasks. In conclusion, this study proposes a novel LLM self-improving method that requires minimal knowledge and human input and is applicable to multiple biomedical domains. 

---
# AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening 

**Authors**: Frank P.-W. Lo, Jianing Qiu, Zeyu Wang, Haibao Yu, Yeming Chen, Gao Zhang, Benny Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02870)  

**Abstract**: Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows. 

---
# Multi-Agent LLM Judge: automatic personalized LLM judge design for evaluating natural language generation applications 

**Authors**: Hongliu Cao, Ilias Driouich, Robin Singh, Eoin Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02867)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across diverse domains, yet they still encounter challenges such as insufficient domain-specific knowledge, biases, and hallucinations. This underscores the need for robust evaluation methodologies to accurately assess LLM-based applications. Traditional evaluation methods, which rely on word overlap or text embeddings, are inadequate for capturing the nuanced semantic information necessary to evaluate dynamic, open-ended text generation. Recent research has explored leveraging LLMs to mimic human reasoning and decision-making processes for evaluation purposes known as LLM-as-a-judge framework. However, these existing frameworks have two significant limitations. First, they lack the flexibility to adapt to different text styles, including various answer and ground truth styles, thereby reducing their generalization performance. Second, the evaluation scores produced by these frameworks are often skewed and hard to interpret, showing a low correlation with human judgment. To address these challenges, we propose a novel dynamic multi-agent system that automatically designs personalized LLM judges for various natural language generation applications. This system iteratively refines evaluation prompts and balances the trade-off between the adaptive requirements of downstream tasks and the alignment with human perception. Our experimental results show that the proposed multi-agent LLM Judge framework not only enhances evaluation accuracy compared to existing methods but also produces evaluation scores that better align with human perception. 

---
# The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances 

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02865)  

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms. 

---
# The Material Contracts Corpus 

**Authors**: Peter Adelson, Julian Nyarko  

**Link**: [PDF](https://arxiv.org/pdf/2504.02864)  

**Abstract**: This paper introduces the Material Contracts Corpus (MCC), a publicly available dataset comprising over one million contracts filed by public companies with the U.S. Securities and Exchange Commission (SEC) between 2000 and 2023. The MCC facilitates empirical research on contract design and legal language, and supports the development of AI-based legal tools. Contracts in the corpus are categorized by agreement type and linked to specific parties using machine learning and natural language processing techniques, including a fine-tuned LLaMA-2 model for contract classification. The MCC further provides metadata such as filing form, document format, and amendment status. We document trends in contractual language, length, and complexity over time, and highlight the dominance of employment and security agreements in SEC filings. This resource is available for bulk download and online access at this https URL. 

---
# GS_DravidianLangTech@2025: Women Targeted Abusive Texts Detection on Social Media 

**Authors**: Girma Yohannis Bade, Zahra Ahani, Olga Kolesnikova, José Luis Oropeza, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2504.02863)  

**Abstract**: The increasing misuse of social media has become a concern; however, technological solutions are being developed to moderate its content effectively. This paper focuses on detecting abusive texts targeting women on social media platforms. Abusive speech refers to communication intended to harm or incite hatred against vulnerable individuals or groups. Specifically, this study aims to identify abusive language directed toward women. To achieve this, we utilized logistic regression and BERT as base models to train datasets sourced from DravidianLangTech@2025 for Tamil and Malayalam languages. The models were evaluated on test datasets, resulting in a 0.729 macro F1 score for BERT and 0.6279 for logistic regression in Tamil and Malayalam, respectively. 

---
# Optimizing Humor Generation in Large Language Models: Temperature Configurations and Architectural Trade-offs 

**Authors**: Evgenii Evstafev  

**Link**: [PDF](https://arxiv.org/pdf/2504.02858)  

**Abstract**: Large language models (LLMs) demonstrate increasing capabilities in creative text generation, yet systematic evaluations of their humor production remain underexplored. This study presents a comprehensive analysis of 13 state-of-the-art LLMs across five architectural families, evaluating their performance in generating technically relevant humor for software developers. Through a full factorial design testing 715 unique configurations of temperature settings and prompt variations, we assess model outputs using five weighted criteria: humor quality, domain relevance, concept originality, tone precision, and delivery efficiency. Our methodology employs rigorous statistical analysis including ANOVA, correlation studies, and quadratic regression to identify optimal configurations and architectural influences. Results reveal significant performance variations across models, with certain architectures achieving 21.8% superiority over baseline systems. Temperature sensitivity analysis demonstrates that 73% of models achieve peak performance at lower stochasticity settings (<= 0.5), though optimal ranges vary substantially by architecture. We identify distinct model clusters: compact high-performers maintaining efficiency-quality balance versus verbose specialists requiring longer outputs for marginal gains. Statistical validation confirms model architecture explains 38.7% of performance variance, with significant correlations between humor quality and concept originality. The study establishes practical guidelines for model selection and configuration, demonstrating how temperature adjustments and architectural considerations impact humor generation effectiveness. These findings advance understanding of LLM capabilities in creative technical writing and provide empirically validated configuration strategies for developers implementing humor-generation systems. 

---
# Do Larger Language Models Imply Better Reasoning? A Pretraining Scaling Law for Reasoning 

**Authors**: Xinyi Wang, Shawn Tan, Mingyu Jin, William Yang Wang, Rameswar Panda, Yikang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.03635)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks requiring complex reasoning. However, the effects of scaling on their reasoning abilities remain insufficiently understood. In this paper, we introduce a synthetic multihop reasoning environment designed to closely replicate the structure and distribution of real-world large-scale knowledge graphs. Our reasoning task involves completing missing edges in the graph, which requires advanced multi-hop reasoning and mimics real-world reasoning scenarios. To evaluate this, we pretrain language models (LMs) from scratch solely on triples from the incomplete graph and assess their ability to infer the missing edges. Interestingly, we observe that overparameterization can impair reasoning performance due to excessive memorization. We investigate different factors that affect this U-shaped loss curve, including graph structure, model size, and training steps. To predict the optimal model size for a specific knowledge graph, we find an empirical scaling that linearly maps the knowledge graph search entropy to the optimal model size. This work provides new insights into the relationship between scaling and reasoning in LLMs, shedding light on possible ways to optimize their performance for reasoning tasks. 

---
# Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency 

**Authors**: Erik Johannes Husom, Arda Goknil, Merve Astekin, Lwin Khin Shar, Andre Kåsen, Sagar Sen, Benedikt Andreas Mithassel, Ahmet Soylu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03360)  

**Abstract**: Deploying Large Language Models (LLMs) on edge devices presents significant challenges due to computational constraints, memory limitations, inference speed, and energy consumption. Model quantization has emerged as a key technique to enable efficient LLM inference by reducing model size and computational overhead. In this study, we conduct a comprehensive analysis of 28 quantized LLMs from the Ollama library, which applies by default Post-Training Quantization (PTQ) and weight-only quantization techniques, deployed on an edge device (Raspberry Pi 4 with 4GB RAM). We evaluate energy efficiency, inference performance, and output accuracy across multiple quantization levels and task types. Models are benchmarked on five standardized datasets (CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, and HumanEval), and we employ a high-resolution, hardware-based energy measurement tool to capture real-world power consumption. Our findings reveal the trade-offs between energy efficiency, inference speed, and accuracy in different quantization settings, highlighting configurations that optimize LLM deployment for resource-constrained environments. By integrating hardware-level energy profiling with LLM benchmarking, this study provides actionable insights for sustainable AI, bridging a critical gap in existing research on energy-aware LLM deployment. 

---
# Optimal Embedding Guided Negative Sample Generation for Knowledge Graph Link Prediction 

**Authors**: Makoto Takamoto, Daniel Oñoro-Rubio, Wiem Ben Rim, Takashi Maruyama, Bhushan Kotnis  

**Link**: [PDF](https://arxiv.org/pdf/2504.03327)  

**Abstract**: Knowledge graph embedding (KGE) models encode the structural information of knowledge graphs to predicting new links. Effective training of these models requires distinguishing between positive and negative samples with high precision. Although prior research has shown that improving the quality of negative samples can significantly enhance model accuracy, identifying high-quality negative samples remains a challenging problem. This paper theoretically investigates the condition under which negative samples lead to optimal KG embedding and identifies a sufficient condition for an effective negative sample distribution. Based on this theoretical foundation, we propose \textbf{E}mbedding \textbf{MU}tation (\textsc{EMU}), a novel framework that \emph{generates} negative samples satisfying this condition, in contrast to conventional methods that focus on \emph{identifying} challenging negative samples within the training data. Importantly, the simplicity of \textsc{EMU} ensures seamless integration with existing KGE models and negative sampling methods. To evaluate its efficacy, we conducted comprehensive experiments across multiple datasets. The results consistently demonstrate significant improvements in link prediction performance across various KGE models and negative sampling methods. Notably, \textsc{EMU} enables performance improvements comparable to those achieved by models with embedding dimension five times larger. An implementation of the method and experiments are available at this https URL. 

---
# RWKVTTS: Yet another TTS based on RWKV-7 

**Authors**: Lin yueyu, Liu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.03289)  

**Abstract**: Human-AI interaction thrives on intuitive and efficient interfaces, among which voice stands out as a particularly natural and accessible modality. Recent advancements in transformer-based text-to-speech (TTS) systems, such as Fish-Speech, CosyVoice, and MegaTTS 3, have delivered remarkable improvements in quality and realism, driving a significant evolution in the TTS domain. In this paper, we introduce RWKV-7 \cite{peng2025rwkv}, a cutting-edge RNN-based architecture tailored for TTS applications. Unlike traditional transformer models, RWKV-7 leverages the strengths of recurrent neural networks to achieve greater computational efficiency and scalability, while maintaining high-quality output. Our comprehensive benchmarks demonstrate that RWKV-7 outperforms transformer-based models across multiple key metrics, including synthesis speed, naturalness of speech, and resource efficiency. Furthermore, we explore its adaptability to diverse linguistic contexts and low-resource environments, showcasing its potential to democratize TTS technology. These findings position RWKV-7 as a powerful and innovative alternative, paving the way for more accessible and versatile voice synthesis solutions in real-world this http URL code and weights are this https URL, this https URL 

---
# Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective 

**Authors**: Garry A. Gabison, R. Patrick Xian  

**Link**: [PDF](https://arxiv.org/pdf/2504.03255)  

**Abstract**: Agentic systems powered by large language models (LLMs) are becoming progressively more complex and capable. Their increasing agency and expanding deployment settings attract growing attention over effective governance policies, monitoring and control protocols. Based on emerging landscapes of the agentic market, we analyze the potential liability issues stemming from delegated use of LLM agents and their extended systems from a principal-agent perspective. Our analysis complements existing risk-based studies on artificial agency and covers the spectrum of important aspects of the principal-agent relationship and their potential consequences at deployment. Furthermore, we motivate method developments for technical governance along the directions of interpretability and behavior evaluations, reward and conflict management, and the mitigation of misalignment and misconduct through principled engineering of detection and fail-safe mechanisms. By illustrating the outstanding issues in AI liability for LLM-based agentic systems, we aim to inform the system design, auditing and monitoring approaches to enhancing transparency and accountability. 

---
# DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments 

**Authors**: Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03160)  

**Abstract**: Large Language Models (LLMs) equipped with web search capabilities have demonstrated impressive potential for deep research tasks. However, current approaches predominantly rely on either manually engineered prompts (prompt engineering-based) with brittle performance or reinforcement learning within controlled Retrieval-Augmented Generation (RAG) environments (RAG-based) that fail to capture the complexities of real-world interaction. In this paper, we introduce DeepResearcher, the first comprehensive framework for end-to-end training of LLM-based deep research agents through scaling reinforcement learning (RL) in real-world environments with authentic web search interactions. Unlike RAG-based approaches that assume all necessary information exists within a fixed corpus, our method trains agents to navigate the noisy, unstructured, and dynamic nature of the open web. We implement a specialized multi-agent architecture where browsing agents extract relevant information from various webpage structures and overcoming significant technical challenges. Extensive experiments on open-domain research tasks demonstrate that DeepResearcher achieves substantial improvements of up to 28.9 points over prompt engineering-based baselines and up to 7.2 points over RAG-based RL agents. Our qualitative analysis reveals emergent cognitive behaviors from end-to-end RL training, including the ability to formulate plans, cross-validate information from multiple sources, engage in self-reflection to redirect research, and maintain honesty when unable to find definitive answers. Our results highlight that end-to-end training in real-world web environments is not merely an implementation detail but a fundamental requirement for developing robust research capabilities aligned with real-world applications. We release DeepResearcher at this https URL. 

---
# LightPROF: A Lightweight Reasoning Framework for Large Language Model on Knowledge Graph 

**Authors**: Tu Ao, Yanhua Yu, Yuling Wang, Yang Deng, Zirui Guo, Liang Pang, Pinghui Wang, Tat-Seng Chua, Xiao Zhang, Zhen Cai  

**Link**: [PDF](https://arxiv.org/pdf/2504.03137)  

**Abstract**: Large Language Models (LLMs) have impressive capabilities in text understanding and zero-shot reasoning. However, delays in knowledge updates may cause them to reason incorrectly or produce harmful results. Knowledge Graphs (KGs) provide rich and reliable contextual information for the reasoning process of LLMs by structurally organizing and connecting a wide range of entities and relations. Existing KG-based LLM reasoning methods only inject KGs' knowledge into prompts in a textual form, ignoring its structural information. Moreover, they mostly rely on close-source models or open-source models with large parameters, which poses challenges to high resource consumption. To address this, we propose a novel Lightweight and efficient Prompt learning-ReasOning Framework for KGQA (LightPROF), which leverages the full potential of LLMs to tackle complex reasoning tasks in a parameter-efficient manner. Specifically, LightPROF follows a "Retrieve-Embed-Reason process", first accurately, and stably retrieving the corresponding reasoning graph from the KG through retrieval module. Next, through a Transformer-based Knowledge Adapter, it finely extracts and integrates factual and structural information from the KG, then maps this information to the LLM's token embedding space, creating an LLM-friendly prompt to be used by the LLM for the final reasoning. Additionally, LightPROF only requires training Knowledge Adapter and can be compatible with any open-source LLM. Extensive experiments on two public KGQA benchmarks demonstrate that LightPROF achieves superior performance with small-scale LLMs. Furthermore, LightPROF shows significant advantages in terms of input token count and reasoning time. 

---
# LLM Library Learning Fails: A LEGO-Prover Case Study 

**Authors**: Ian Berlot-Attwell, Frank Rudzicz, Xujie Si  

**Link**: [PDF](https://arxiv.org/pdf/2504.03048)  

**Abstract**: Recent advancements in the coding, reasoning, and tool-using abilities of LLMs have spurred interest in library learning (i.e., online learning through the creation, storage, and retrieval of reusable and composable functions, knowledge, checklists, or lemmas). Such systems often promise improved task performance through the automatic creation of broadly applicable tools, as well as superior computational performance through the caching of reasoning (i.e., the storage of generated tools). However, we find strong reason to be skeptical. We perform a deep dive into one such system, LEGO-Prover, which purports to learn reusable lemmas for mathematical reasoning. We find no evidence of the direct reuse of learned lemmas, and find evidence against the soft reuse of learned lemmas (i.e., reuse by modifying relevant examples). Crucially, we find that LEGO-Prover does not in fact improve over the simple baseline of prompting the model - the improvements in task accuracy vanish once computational cost is accounted for. Our findings suggest that serious misconceptions exist as to the effectiveness of these techniques, that a serious re-examination of the state of LLM-based library learning is required, and that we require much stronger standards for evaluation including behavioural analysis and ensuring that an equal computational budget is used for baselines. 

---
# Ontologies in Design: How Imagining a Tree Reveals Possibilites and Assumptions in Large Language Models 

**Authors**: Nava Haghighi, Sunny Yu, James Landay, Daniela Rosner  

**Link**: [PDF](https://arxiv.org/pdf/2504.03029)  

**Abstract**: Amid the recent uptake of Generative AI, sociotechnical scholars and critics have traced a multitude of resulting harms, with analyses largely focused on values and axiology (e.g., bias). While value-based analyses are crucial, we argue that ontologies -- concerning what we allow ourselves to think or talk about -- is a vital but under-recognized dimension in analyzing these systems. Proposing a need for a practice-based engagement with ontologies, we offer four orientations for considering ontologies in design: pluralism, groundedness, liveliness, and enactment. We share examples of potentialities that are opened up through these orientations across the entire LLM development pipeline by conducting two ontological analyses: examining the responses of four LLM-based chatbots in a prompting exercise, and analyzing the architecture of an LLM-based agent simulation. We conclude by sharing opportunities and limitations of working with ontologies in the design and development of sociotechnical systems. 

---
# Language Models Guidance with Multi-Aspect-Cueing: A Case Study for Competitor Analysis 

**Authors**: Amir Hadifar, Christopher Ochs, Arjan Van Ewijk  

**Link**: [PDF](https://arxiv.org/pdf/2504.02984)  

**Abstract**: Competitor analysis is essential in modern business due to the influence of industry rivals on strategic planning. It involves assessing multiple aspects and balancing trade-offs to make informed decisions. Recent Large Language Models (LLMs) have demonstrated impressive capabilities to reason about such trade-offs but grapple with inherent limitations such as a lack of knowledge about contemporary or future realities and an incomplete understanding of a market's competitive landscape. In this paper, we address this gap by incorporating business aspects into LLMs to enhance their understanding of a competitive market. Through quantitative and qualitative experiments, we illustrate how integrating such aspects consistently improves model performance, thereby enhancing analytical efficacy in competitor analysis. 

---
# QID: Efficient Query-Informed ViTs in Data-Scarce Regimes for OCR-free Visual Document Understanding 

**Authors**: Binh M. Le, Shaoyuan Xu, Jinmiao Fu, Zhishen Huang, Moyan Li, Yanhui Guo, Hongdong Li, Sameera Ramasinghe, Bryan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02971)  

**Abstract**: In Visual Document Understanding (VDU) tasks, fine-tuning a pre-trained Vision-Language Model (VLM) with new datasets often falls short in optimizing the vision encoder to identify query-specific regions in text-rich document images. Existing methods that directly inject queries into model layers by modifying the network architecture often struggle to adapt to new datasets with limited annotations. To address this, we introduce QID, a novel, streamlined, architecture-preserving approach that integrates query embeddings into the vision encoder, leading to notable performance gains, particularly in data-scarce fine-tuning scenarios. Specifically, our approach introduces a dual-module framework: a query-aware module that generates a unique query vector to precisely guide the model's focus, as well as a query-agnostic module that captures the positional relationships among tokens, ensuring robust spatial understanding. Notably, both modules operate independently of the vision attention blocks, facilitating targeted learning of query embeddings and enhancing visual semantic identification. Experiments with OCR-free VLMs across multiple datasets demonstrate significant performance improvements using our method, especially in handling text-rich documents in data-scarce environments. 

---
# Robustly identifying concepts introduced during chat fine-tuning using crosscoders 

**Authors**: Julian Minder, Clement Dumas, Caden Juang, Bilal Chugtai, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2504.02922)  

**Abstract**: Model diffing is the study of how fine-tuning changes a model's representations and internal algorithms. Many behaviours of interest are introduced during fine-tuning, and model diffing offers a promising lens to interpret such behaviors. Crosscoders are a recent model diffing method that learns a shared dictionary of interpretable concepts represented as latent directions in both the base and fine-tuned models, allowing us to track how concepts shift or emerge during fine-tuning. Notably, prior work has observed concepts with no direction in the base model, and it was hypothesized that these model-specific latents were concepts introduced during fine-tuning. However, we identify two issues which stem from the crosscoders L1 training loss that can misattribute concepts as unique to the fine-tuned model, when they really exist in both models. We develop Latent Scaling to flag these issues by more accurately measuring each latent's presence across models. In experiments comparing Gemma 2 2B base and chat models, we observe that the standard crosscoder suffers heavily from these issues. Building on these insights, we train a crosscoder with BatchTopK loss and show that it substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts. We recommend practitioners adopt similar techniques. Using the BatchTopK crosscoder, we successfully identify a set of genuinely chat-specific latents that are both interpretable and causally effective, representing concepts such as $\textit{false information}$ and $\textit{personal question}$, along with multiple refusal-related latents that show nuanced preferences for different refusal triggers. Overall, our work advances best practices for the crosscoder-based methodology for model diffing and demonstrates that it can provide concrete insights into how chat tuning modifies language model behavior. 

---
# Mapping Technological Futures: Anticipatory Discourse Through Text Mining 

**Authors**: Maciej Skorski, Alina Landowska, Krzysztof Rajda  

**Link**: [PDF](https://arxiv.org/pdf/2504.02853)  

**Abstract**: The volatility and unpredictability of emerging technologies, such as artificial intelligence (AI), generate significant uncertainty, which is widely discussed on social media. This study examines anticipatory discourse surrounding technological futures by analysing 1.5 million posts from 400 key opinion leaders (KOLs) published on the X platform (from 2021 to 2023). Using advanced text mining techniques, including BERTopic modelling, sentiment, emotion, and attitude analyses, the research identifies 100 distinct topics reflecting anticipated tech-driven futures. Our findings emphasize the dual role of KOLs in framing \textit{present futures} -- optimistic visions of transformative technologies like AI and IoT -- and influencing \textit{future presents}, where these projections shape contemporary societal and geopolitical debates. Positive emotions such as Hope dominate, outweighing Anxiety, particularly in topics like ``Machine Learning, Data Science, and Deep Learning,'' while discussions around ``Climate Change'' and ``War, Ukraine, and Trump People'' elicit \textit{Anxiety}. By framing technologies as solutions to societal challenges, KOLs act as mediators of societal narratives, bridging imagined futures and current realities. These insights underscore their pivotal role in directing public attention with emerging technologies during periods of heightened uncertainty, advancing our understanding of anticipatory discourse in technology-mediated contexts. 

---
