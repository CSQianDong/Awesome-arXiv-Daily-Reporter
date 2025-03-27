# RALLRec+: Retrieval Augmented Large Language Model Recommendation with Reasoning 

**Authors**: Sichun Luo, Jian Xu, Xiaojie Zhang, Linrong Wang, Sicong Liu, Hanxu Hou, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.20430)  

**Abstract**: Large Language Models (LLMs) have been integrated into recommender systems to enhance user behavior comprehension. The Retrieval Augmented Generation (RAG) technique is further incorporated into these systems to retrieve more relevant items and improve system performance. However, existing RAG methods have two shortcomings. \textit{(i)} In the \textit{retrieval} stage, they rely primarily on textual semantics and often fail to incorporate the most relevant items, thus constraining system effectiveness. \textit{(ii)} In the \textit{generation} stage, they lack explicit chain-of-thought reasoning, further limiting their potential.
In this paper, we propose Representation learning and \textbf{R}easoning empowered retrieval-\textbf{A}ugmented \textbf{L}arge \textbf{L}anguage model \textbf{Rec}ommendation (RALLRec+). Specifically, for the retrieval stage, we prompt LLMs to generate detailed item descriptions and perform joint representation learning, combining textual and collaborative signals extracted from the LLM and recommendation models, respectively. To account for the time-varying nature of user interests, we propose a simple yet effective reranking method to capture preference dynamics. For the generation phase, we first evaluate reasoning LLMs on recommendation tasks, uncovering valuable insights. Then we introduce knowledge-injected prompting and consistency-based merging approach to integrate reasoning LLMs with general-purpose LLMs, enhancing overall performance. Extensive experiments on three real world datasets validate our method's effectiveness. 

---
# Dewey Long Context Embedding Model: A Technical Report 

**Authors**: Dun Zhang, Panxiang Zou, Yudong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.20376)  

**Abstract**: This technical report presents the training methodology and evaluation results of the open-source dewey_en_beta embedding model. The increasing demand for retrieval-augmented generation (RAG) systems and the expanding context window capabilities of large language models (LLMs) have created critical challenges for conventional embedding models. Current approaches often struggle to maintain semantic coherence when processing documents exceeding typical sequence length limitations, significantly impacting retrieval performance in knowledge-intensive applications. This paper presents dewey_en_beta, a novel text embedding model that achieves excellent performance on MTEB (Eng, v2) and LongEmbed benchmark while supporting 128K token sequences. Our technical contribution centers on chunk alignment training, an innovative methodology that enables the simultaneous generation of localized chunk embeddings and global document-level representations through distillation. Information regarding the model release can be found at this https URL. 

---
# Learnable Sequence Augmenter for Triplet Contrastive Learning in Sequential Recommendation 

**Authors**: Wei Wang, Yujie Lin, Jianli Zhao, Moyan Zhang, Pengjie Ren, Xianye Ben, Yujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.20232)  

**Abstract**: Most existing contrastive learning-based sequential recommendation (SR) methods rely on random operations (e.g., crop, reorder, and substitute) to generate augmented sequences. These methods often struggle to create positive sample pairs that closely resemble the representations of the raw sequences, potentially disrupting item correlations by deleting key items or introducing noisy iterac, which misguides the contrastive learning process.
To address this limitation, we propose Learnable sequence Augmentor for triplet Contrastive Learning in sequential Recommendation (LACLRec). Specifically, the self-supervised learning-based augmenter can automatically delete noisy items from sequences and insert new items that better capture item transition patterns, generating a higher-quality augmented sequence. Subsequently, we randomly generate another augmented sequence and design a ranking-based triplet contrastive loss to differentiate the similarities between the raw sequence, the augmented sequence from augmenter, and the randomly augmented sequence, providing more fine-grained contrastive signals. Extensive experiments on three real-world datasets demonstrate that both the sequence augmenter and the triplet contrast contribute to improving recommendation accuracy. LACLRec significantly outperforms the baseline model CL4SRec, and demonstrates superior performance compared to several state-of-the-art sequential recommendation algorithms. 

---
# BeLightRec: A lightweight recommender system enhanced with BERT 

**Authors**: Manh Mai Van, Tin T. Tran  

**Link**: [PDF](https://arxiv.org/pdf/2503.20206)  

**Abstract**: The trend of data mining using deep learning models on graph neural networks has proven effective in identifying object features through signal encoders and decoders, particularly in recommendation systems utilizing collaborative filtering methods. Collaborative filtering exploits similarities between users and items from historical data. However, it overlooks distinctive information, such as item names and descriptions. The semantic data of items should be further mined using models in the natural language processing field. Thus, items can be compared using text classification, similarity assessments, or identifying analogous sentence pairs. This research proposes combining two sources of item similarity signals: one from collaborative filtering and one from the semantic similarity measure between item names and descriptions. These signals are integrated into a graph convolutional neural network to optimize model weights, thereby providing accurate recommendations. Experiments are also designed to evaluate the contribution of each signal group to the recommendation results. 

---
# MMMORRF: Multimodal Multilingual Modularized Reciprocal Rank Fusion 

**Authors**: Saron Samuel, Dan DeGenaro, Jimena Guallar-Blasco, Kate Sanders, Oluwaseun Eisape, Arun Reddy, Alexander Martin, Andrew Yates, Eugene Yang, Cameron Carpenter, David Etter, Efsun Kayi, Matthew Wiesner, Kenton Murray, Reno Kriz  

**Link**: [PDF](https://arxiv.org/pdf/2503.20698)  

**Abstract**: Videos inherently contain multiple modalities, including visual events, text overlays, sounds, and speech, all of which are important for retrieval. However, state-of-the-art multimodal language models like VAST and LanguageBind are built on vision-language models (VLMs), and thus overly prioritize visual signals. Retrieval benchmarks further reinforce this bias by focusing on visual queries and neglecting other modalities. We create a search system MMMORRF that extracts text and features from both visual and audio modalities and integrates them with a novel modality-aware weighted reciprocal rank fusion. MMMORRF is both effective and efficient, demonstrating practicality in searching videos based on users' information needs instead of visual descriptive queries. We evaluate MMMORRF on MultiVENT 2.0 and TVR, two multimodal benchmarks designed for more targeted information needs, and find that it improves nDCG@20 by 81% over leading multimodal encoders and 37% over single-modality retrieval, demonstrating the value of integrating diverse modalities. 

---
# Open Deep Search: Democratizing Search with Open-source Reasoning Agents 

**Authors**: Salaheddin Alzubi, Creston Brooks, Purva Chiniya, Edoardo Contente, Chiara von Gerlach, Lucas Irwin, Yihan Jiang, Arda Kaz, Windsor Nguyen, Sewoong Oh, Himanshu Tyagi, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.20201)  

**Abstract**: We introduce Open Deep Search (ODS) to close the increasing gap between the proprietary search AI solutions, such as Perplexity's Sonar Reasoning Pro and OpenAI's GPT-4o Search Preview, and their open-source counterparts. The main innovation introduced in ODS is to augment the reasoning capabilities of the latest open-source LLMs with reasoning agents that can judiciously use web search tools to answer queries. Concretely, ODS consists of two components that work with a base LLM chosen by the user: Open Search Tool and Open Reasoning Agent. Open Reasoning Agent interprets the given task and completes it by orchestrating a sequence of actions that includes calling tools, one of which is the Open Search Tool. Open Search Tool is a novel web search tool that outperforms proprietary counterparts. Together with powerful open-source reasoning LLMs, such as DeepSeek-R1, ODS nearly matches and sometimes surpasses the existing state-of-the-art baselines on two benchmarks: SimpleQA and FRAMES. For example, on the FRAMES evaluation benchmark, ODS improves the best existing baseline of the recently released GPT-4o Search Preview by 9.7% in accuracy. ODS is a general framework for seamlessly augmenting any LLMs -- for example, DeepSeek-R1 that achieves 82.4% on SimpleQA and 30.1% on FRAMES -- with search and reasoning capabilities to achieve state-of-the-art performance: 88.3% on SimpleQA and 75.3% on FRAMES. 

---
# ProtoBERT-LoRA: Parameter-Efficient Prototypical Finetuning for Immunotherapy Study Identification 

**Authors**: Shijia Zhang, Xiyu Ding, Kai Ding, Jacob Zhang, Kevin Galinsky, Mengrui Wang, Ryan P. Mayers, Zheyu Wang, Hadi Kharrazi  

**Link**: [PDF](https://arxiv.org/pdf/2503.20179)  

**Abstract**: Identifying immune checkpoint inhibitor (ICI) studies in genomic repositories like Gene Expression Omnibus (GEO) is vital for cancer research yet remains challenging due to semantic ambiguity, extreme class imbalance, and limited labeled data in low-resource settings. We present ProtoBERT-LoRA, a hybrid framework that combines PubMedBERT with prototypical networks and Low-Rank Adaptation (LoRA) for efficient fine-tuning. The model enforces class-separable embeddings via episodic prototype training while preserving biomedical domain knowledge. Our dataset was divided as: Training (20 positive, 20 negative), Prototype Set (10 positive, 10 negative), Validation (20 positive, 200 negative), and Test (71 positive, 765 negative). Evaluated on test dataset, ProtoBERT-LoRA achieved F1-score of 0.624 (precision: 0.481, recall: 0.887), outperforming the rule-based system, machine learning baselines and finetuned PubMedBERT. Application to 44,287 unlabeled studies reduced manual review efforts by 82%. Ablation studies confirmed that combining prototypes with LoRA improved performance by 29% over stand-alone LoRA. 

---
