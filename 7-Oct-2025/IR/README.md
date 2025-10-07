# Topic-Specific Classifiers are Better Relevance Judges than Prompted LLMs 

**Authors**: Lukas Gienapp, Martin Potthast, Harrisen Scells, Eugene Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04633)  

**Abstract**: The unjudged document problem, where pooled test collections have incomplete relevance judgments for evaluating new retrieval systems, is a key obstacle to the reusability of test collections in information retrieval. While the de facto standard to deal with the problem is to treat unjudged documents as non-relevant, many alternatives have been proposed, including the use of large language models (LLMs) as a relevance judge (LLM-as-a-judge). However, this has been criticized as circular, since the same LLM can be used as a judge and as a ranker at the same time. We propose to train topic-specific relevance classifiers instead: By finetuning monoT5 with independent LoRA weight adaptation on the judgments of a single assessor for a single topic's pool, we align it to that assessor's notion of relevance for the topic. The system rankings obtained through our classifier's relevance judgments achieve a Spearmans' $\rho$ correlation of $>0.95$ with ground truth system rankings. As little as 128 initial human judgments per topic suffice to improve the comparability of models, compared to treating unjudged documents as non-relevant, while achieving more reliability than existing LLM-as-a-judge approaches. Topic-specific relevance classifiers thus are a lightweight and straightforward way to tackle the unjudged document problem, while maintaining human judgments as the gold standard for retrieval evaluation. Code, models, and data are made openly available. 

---
# MARCO: A Cooperative Knowledge Transfer Framework for Personalized Cross-domain Recommendations 

**Authors**: Lili Xie, Yi Zhang, Ruihong Qiu, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.04508)  

**Abstract**: Recommender systems frequently encounter data sparsity issues, particularly when addressing cold-start scenarios involving new users or items. Multi-source cross-domain recommendation (CDR) addresses these challenges by transferring valuable knowledge from multiple source domains to enhance recommendations in a target domain. However, existing reinforcement learning (RL)-based CDR methods typically rely on a single-agent framework, leading to negative transfer issues caused by inconsistent domain contributions and inherent distributional discrepancies among source domains. To overcome these limitations, MARCO, a Multi-Agent Reinforcement Learning-based Cross-Domain recommendation framework, is proposed. It leverages cooperative multi-agent reinforcement learning, where each agent is dedicated to estimating the contribution from an individual source domain, effectively managing credit assignment and mitigating negative transfer. In addition, an entropy-based action diversity penalty is introduced to enhance policy expressiveness and stabilize training by encouraging diverse agents' joint actions. Extensive experiments across four benchmark datasets demonstrate MARCO's superior performance over state-of-the-art methods, highlighting its robustness and strong generalization capabilities. The code is at this https URL. 

---
# Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation 

**Authors**: Yue Que, Yingyi Zhang, Xiangyu Zhao, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.04502)  

**Abstract**: Graph-based recommender systems leverage neighborhood aggregation to generate node representations, which is highly sensitive to popularity bias, resulting in an echo effect during information propagation. Existing graph-based debiasing solutions refine the aggregation process with attempts such as edge reconstruction or weight adjustment. However, these methods remain inadequate in fully alleviating popularity bias. Specifically, this is because 1) they provide no insights into graph aggregation rationality, thus lacking an optimality guarantee; 2) they fail to well balance the training and debiasing process, which undermines the effectiveness. In this paper, we propose a novel approach to mitigate popularity bias through rational modeling of the graph aggregation process. We reveal that graph aggregation is a special form of backdoor adjustment in causal inference, where the aggregation weight corresponds to the historical interaction likelihood distribution. Based on this insight, we devise an encoder-decoder architecture, namely Causality-aware Graph Aggregation Weight Estimator for Debiasing (CAGED), to approximate the unbiased aggregation weight by optimizing the evidence lower bound of the interaction likelihood. In order to enhance the debiasing effectiveness during early training stages, we further design a momentum update strategy that incrementally refines the aggregation weight matrix. Extensive experiments on three datasets demonstrate that CAGED outperforms existing graph-based debiasing methods. Our implementation is available at this https URL. 

---
# Empowering Denoising Sequential Recommendation with Large Language Model Embeddings 

**Authors**: Tongzhou Wu, Yuhao Wang, Maolin Wang, Chi Zhang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.04239)  

**Abstract**: Sequential recommendation aims to capture user preferences by modeling sequential patterns in user-item interactions. However, these models are often influenced by noise such as accidental interactions, leading to suboptimal performance. Therefore, to reduce the effect of noise, some works propose explicitly identifying and removing noisy items. However, we find that simply relying on collaborative information may result in an over-denoising problem, especially for cold items. To overcome these limitations, we propose a novel framework: Interest Alignment for Denoising Sequential Recommendation (IADSR) which integrates both collaborative and semantic information. Specifically, IADSR is comprised of two stages: in the first stage, we obtain the collaborative and semantic embeddings of each item from a traditional sequential recommendation model and an LLM, respectively. In the second stage, we align the collaborative and semantic embeddings and then identify noise in the interaction sequence based on long-term and short-term interests captured in the collaborative and semantic modalities. Our extensive experiments on four public datasets validate the effectiveness of the proposed framework and its compatibility with different sequential recommendation systems. 

---
# Learning-Based Hashing for ANN Search: Foundations and Early Advances 

**Authors**: Sean Moran  

**Link**: [PDF](https://arxiv.org/pdf/2510.04127)  

**Abstract**: Approximate Nearest Neighbour (ANN) search is a fundamental problem in information retrieval, underpinning large-scale applications in computer vision, natural language processing, and cross-modal search. Hashing-based methods provide an efficient solution by mapping high-dimensional data into compact binary codes that enable fast similarity computations in Hamming space. Over the past two decades, a substantial body of work has explored learning to hash, where projection and quantisation functions are optimised from data rather than chosen at random.
This article offers a foundational survey of early learning-based hashing methods, with an emphasis on the core ideas that shaped the field. We review supervised, unsupervised, and semi-supervised approaches, highlighting how projection functions are designed to generate meaningful embeddings and how quantisation strategies convert these embeddings into binary codes. We also examine extensions to multi-bit and multi-threshold models, as well as early advances in cross-modal retrieval.
Rather than providing an exhaustive account of the most recent methods, our goal is to introduce the conceptual foundations of learning-based hashing for ANN search. By situating these early models in their historical context, we aim to equip readers with a structured understanding of the principles, trade-offs, and open challenges that continue to inform current research in this area. 

---
# RLRF: Competitive Search Agent Design via Reinforcement Learning from Ranker Feedback 

**Authors**: Tommy Mordo, Sagie Dekel, Omer Madmon, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2510.04096)  

**Abstract**: Competitive search is a setting where document publishers modify them to improve their ranking in response to a query. Recently, publishers have increasingly leveraged LLMs to generate and modify competitive content. We introduce Reinforcement Learning from Ranker Feedback (RLRF), a framework that trains LLMs using preference datasets derived from ranking competitions. The goal of a publisher (LLM-based) agent is to optimize content for improved ranking while accounting for the strategies of competing agents. We generate the datasets using approaches that do not rely on human-authored data. We show that our proposed agents consistently and substantially outperform previously suggested approaches for LLM-based competitive document modification. We further show that our agents are effective with ranking functions they were not trained for (i.e., out of distribution) and they adapt to strategic opponents. These findings provide support to the significant potential of using reinforcement learning in competitive search. 

---
# The LCLStream Ecosystem for Multi-Institutional Dataset Exploration 

**Authors**: David Rogers, Valerio Mariani, Cong Wang, Ryan Coffee, Wilko Kroeger, Murali Shankar, Hans Thorsten Schwander, Tom Beck, Frédéric Poitevin, Jana Thayer  

**Link**: [PDF](https://arxiv.org/pdf/2510.04012)  

**Abstract**: We describe a new end-to-end experimental data streaming framework designed from the ground up to support new types of applications -- AI training, extremely high-rate X-ray time-of-flight analysis, crystal structure determination with distributed processing, and custom data science applications and visualizers yet to be created. Throughout, we use design choices merging cloud microservices with traditional HPC batch execution models for security and flexibility. This project makes a unique contribution to the DOE Integrated Research Infrastructure (IRI) landscape. By creating a flexible, API-driven data request service, we address a significant need for high-speed data streaming sources for the X-ray science data analysis community. With the combination of data request API, mutual authentication web security framework, job queue system, high-rate data buffer, and complementary nature to facility infrastructure, the LCLStreamer framework has prototyped and implemented several new paradigms critical for future generation experiments. 

---
# Visual Lifelog Retrieval through Captioning-Enhanced Interpretation 

**Authors**: Yu-Fei Shih, An-Zi Yen, Hen-Hsen Huang, Hsin-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.04010)  

**Abstract**: People often struggle to remember specific details of past experiences, which can lead to the need to revisit these memories. Consequently, lifelog retrieval has emerged as a crucial application. Various studies have explored methods to facilitate rapid access to personal lifelogs for memory recall assistance. In this paper, we propose a Captioning-Integrated Visual Lifelog (CIVIL) Retrieval System for extracting specific images from a user's visual lifelog based on textual queries. Unlike traditional embedding-based methods, our system first generates captions for visual lifelogs and then utilizes a text embedding model to project both the captions and user queries into a shared vector space. Visual lifelogs, captured through wearable cameras, provide a first-person viewpoint, necessitating the interpretation of the activities of the individual behind the camera rather than merely describing the scene. To address this, we introduce three distinct approaches: the single caption method, the collective caption method, and the merged caption method, each designed to interpret the life experiences of lifeloggers. Experimental results show that our method effectively describes first-person visual images, enhancing the outcomes of lifelog retrieval. Furthermore, we construct a textual dataset that converts visual lifelogs into captions, thereby reconstructing personal life experiences. 

---
# Beyond Static Evaluation: Rethinking the Assessment of Personalized Agent Adaptability in Information Retrieval 

**Authors**: Kirandeep Kaur, Preetam Prabhu Srikar Dammu, Hideo Joho, Chirag Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.03984)  

**Abstract**: Personalized AI agents are becoming central to modern information retrieval, yet most evaluation methodologies remain static, relying on fixed benchmarks and one-off metrics that fail to reflect how users' needs evolve over time. These limitations hinder our ability to assess whether agents can meaningfully adapt to individuals across dynamic, longitudinal interactions. In this perspective paper, we propose a conceptual lens for rethinking evaluation in adaptive personalization, shifting the focus from static performance snapshots to interaction-aware, evolving assessments. We organize this lens around three core components: (1) persona-based user simulation with temporally evolving preference models; (2) structured elicitation protocols inspired by reference interviews to extract preferences in context; and (3) adaptation-aware evaluation mechanisms that measure how agent behavior improves across sessions and tasks. While recent works have embraced LLM-driven user simulation, we situate this practice within a broader paradigm for evaluating agents over time. To illustrate our ideas, we conduct a case study in e-commerce search using the PersonalWAB dataset. Beyond presenting a framework, our work lays a conceptual foundation for understanding and evaluating personalization as a continuous, user-centric endeavor. 

---
# Investigating LLM Variability in Personalized Conversational Information Retrieval 

**Authors**: Simon Lupart, Daniël van Dijk, Eric Langezaal, Ian van Dort, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03795)  

**Abstract**: Personalized Conversational Information Retrieval (CIR) has seen rapid progress in recent years, driven by the development of Large Language Models (LLMs). Personalized CIR aims to enhance document retrieval by leveraging user-specific information, such as preferences, knowledge, or constraints, to tailor responses to individual needs. A key resource for this task is the TREC iKAT 2023 dataset, designed to evaluate personalization in CIR pipelines. Building on this resource, Mo et al. explored several strategies for incorporating Personal Textual Knowledge Bases (PTKB) into LLM-based query reformulation. Their findings suggested that personalization from PTKBs could be detrimental and that human annotations were often noisy. However, these conclusions were based on single-run experiments using the GPT-3.5 Turbo model, raising concerns about output variability and repeatability. In this reproducibility study, we rigorously reproduce and extend their work, focusing on LLM output variability and model generalization. We apply the original methods to the new TREC iKAT 2024 dataset and evaluate a diverse range of models, including Llama (1B-70B), Qwen-7B, GPT-4o-mini. Our results show that human-selected PTKBs consistently enhance retrieval performance, while LLM-based selection methods do not reliably outperform manual choices. We further compare variance across datasets and observe higher variability on iKAT than on CAsT, highlighting the challenges of evaluating personalized CIR. Notably, recall-oriented metrics exhibit lower variance than precision-oriented ones, a critical insight for first-stage retrievers. Finally, we underscore the need for multi-run evaluations and variance reporting when assessing LLM-based CIR systems. By broadening evaluation across models, datasets, and metrics, our study contributes to more robust and generalizable practices for personalized CIR. 

---
# Evaluating High-Resolution Piano Sustain Pedal Depth Estimation with Musically Informed Metrics 

**Authors**: Hanwen Zhang, Kun Fang, Ziyu Wang, Ichiro Fujinaga  

**Link**: [PDF](https://arxiv.org/pdf/2510.03750)  

**Abstract**: Evaluation for continuous piano pedal depth estimation tasks remains incomplete when relying only on conventional frame-level metrics, which overlook musically important features such as direction-change boundaries and pedal curve contours. To provide more interpretable and musically meaningful insights, we propose an evaluation framework that augments standard frame-level metrics with an action-level assessment measuring direction and timing using segments of press/hold/release states and a gesture-level analysis that evaluates contour similarity of each press-release cycle. We apply this framework to compare an audio-only baseline with two variants: one incorporating symbolic information from MIDI, and another trained in a binary-valued setting, all within a unified architecture. Results show that the MIDI-informed model significantly outperforms the others at action and gesture levels, despite modest frame-level gains. These findings demonstrate that our framework captures musically relevant improvements indiscernible by traditional metrics, offering a more practical and effective approach to evaluating pedal depth estimation models. 

---
# Contrastive Learning Using Graph Embeddings for Domain Adaptation of Language Models in the Process Industry 

**Authors**: Anastasia Zhukova, Jonas Lührs, Christian E. Matt, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2510.04631)  

**Abstract**: Recent trends in NLP utilize knowledge graphs (KGs) to enhance pretrained language models by incorporating additional knowledge from the graph structures to learn domain-specific terminology or relationships between documents that might otherwise be overlooked. This paper explores how SciNCL, a graph-aware neighborhood contrastive learning methodology originally designed for scientific publications, can be applied to the process industry domain, where text logs contain crucial information about daily operations and are often structured as sparse KGs. Our experiments demonstrate that language models fine-tuned with triplets derived from GE outperform a state-of-the-art mE5-large text encoder by 9.8-14.3% (5.4-8.0p) on the proprietary process industry text embedding benchmark (PITEB) while being 3-5 times smaller in size. 

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
# Evaluating Keyframe Layouts for Visual Known-Item Search in Homogeneous Collections 

**Authors**: Bastian Jäckl, Jiří Kruchina, Lucas Joos, Daniel A. Keim, Ladislav Peška, Jakub Lokoč  

**Link**: [PDF](https://arxiv.org/pdf/2510.04396)  

**Abstract**: Multimodal deep-learning models power interactive video retrieval by ranking keyframes in response to textual queries. Despite these advances, users must still browse ranked candidates manually to locate a target. Keyframe arrangement within the search grid highly affects browsing effectiveness and user efficiency, yet remains underexplored. We report a study with 49 participants evaluating seven keyframe layouts for the Visual Known-Item Search task. Beyond efficiency and accuracy, we relate browsing phenomena, such as overlooks, to layout characteristics. Our results show that a video-grouped layout is the most efficient, while a four-column, rank-preserving grid achieves the highest accuracy. Sorted grids reveal potentials and trade-offs, enabling rapid scanning of uninteresting regions but down-ranking relevant targets to less prominent positions, delaying first arrival times and increasing overlooks.
These findings motivate hybrid designs that preserve positions of top-ranked items while sorting or grouping the remainder, and offer guidance for searching in grids beyond video retrieval. 

---
# Epistemic Diversity and Knowledge Collapse in Large Language Models 

**Authors**: Dustin Wright, Sarah Masud, Jared Moore, Srishti Yadav, Maria Antoniak, Chan Young Park, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.04226)  

**Abstract**: Large language models (LLMs) tend to generate lexically, semantically, and stylistically homogenous texts. This poses a risk of knowledge collapse, where homogenous LLMs mediate a shrinking in the range of accessible information over time. Existing works on homogenization are limited by a focus on closed-ended multiple-choice setups or fuzzy semantic features, and do not look at trends across time and cultural contexts. To overcome this, we present a new methodology to measure epistemic diversity, i.e., variation in real-world claims in LLM outputs, which we use to perform a broad empirical study of LLM knowledge collapse. We test 27 LLMs, 155 topics covering 12 countries, and 200 prompt variations sourced from real user chats. For the topics in our study, we show that while newer models tend to generate more diverse claims, nearly all models are less epistemically diverse than a basic web search. We find that model size has a negative impact on epistemic diversity, while retrieval-augmented generation (RAG) has a positive impact, though the improvement from RAG varies by the cultural context. Finally, compared to a traditional knowledge source (Wikipedia), we find that country-specific claims reflect the English language more than the local one, highlighting a gap in epistemic representation 

---
# Automating construction safety inspections using a multi-modal vision-language RAG framework 

**Authors**: Chenxin Wang, Elyas Asadi Shamsabadi, Zhaohui Chen, Luming Shen, Alireza Ahmadian Fard Fini, Daniel Dias-da-Costa  

**Link**: [PDF](https://arxiv.org/pdf/2510.04145)  

**Abstract**: Conventional construction safety inspection methods are often inefficient as they require navigating through large volume of information. Recent advances in large vision-language models (LVLMs) provide opportunities to automate safety inspections through enhanced visual and linguistic understanding. However, existing applications face limitations including irrelevant or unspecific responses, restricted modal inputs and hallucinations. Utilisation of Large Language Models (LLMs) for this purpose is constrained by availability of training data and frequently lack real-time adaptability. This study introduces SiteShield, a multi-modal LVLM-based Retrieval-Augmented Generation (RAG) framework for automating construction safety inspection reports by integrating visual and audio inputs. Using real-world data, SiteShield outperformed unimodal LLMs without RAG with an F1 score of 0.82, hamming loss of 0.04, precision of 0.76, and recall of 0.96. The findings indicate that SiteShield offers a novel pathway to enhance information retrieval and efficiency in generating safety reports. 

---
# LLM, Reporting In! Medical Information Extraction Across Prompting, Fine-tuning and Post-correction 

**Authors**: Ikram Belmadani, Parisa Nazari Hashemi, Thomas Sebbag, Benoit Favre, Guillaume Fortier, Solen Quiniou, Emmanuel Morin, Richard Dufour  

**Link**: [PDF](https://arxiv.org/pdf/2510.03577)  

**Abstract**: This work presents our participation in the EvalLLM 2025 challenge on biomedical Named Entity Recognition (NER) and health event extraction in French (few-shot setting). For NER, we propose three approaches combining large language models (LLMs), annotation guidelines, synthetic data, and post-processing: (1) in-context learning (ICL) with GPT-4.1, incorporating automatic selection of 10 examples and a summary of the annotation guidelines into the prompt, (2) the universal NER system GLiNER, fine-tuned on a synthetic corpus and then verified by an LLM in post-processing, and (3) the open LLM LLaMA-3.1-8B-Instruct, fine-tuned on the same synthetic corpus. Event extraction uses the same ICL strategy with GPT-4.1, reusing the guideline summary in the prompt. Results show GPT-4.1 leads with a macro-F1 of 61.53% for NER and 15.02% for event extraction, highlighting the importance of well-crafted prompting to maximize performance in very low-resource scenarios. 

---
# CAG: Chunked Augmented Generation for Google Chrome's Built-in Gemini Nano 

**Authors**: Vivek Vellaiyappan Surulimuthu, Aditya Karnam Gururaj Rao  

**Link**: [PDF](https://arxiv.org/pdf/2412.18708)  

**Abstract**: We present Chunked Augmented Generation (CAG), an architecture specifically designed to overcome the context window limitations of Google Chrome's built-in Gemini Nano model. While Chrome's integration of Gemini Nano represents a significant advancement in bringing AI capabilities directly to the browser, its restricted context window poses challenges for processing large inputs. CAG addresses this limitation through intelligent input chunking and processing strategies, enabling efficient handling of extensive content while maintaining the model's performance within browser constraints. Our implementation demonstrates particular efficacy in processing large documents and datasets directly within Chrome, making sophisticated AI capabilities accessible through the browser without external API dependencies. Get started now at this https URL. 

---
