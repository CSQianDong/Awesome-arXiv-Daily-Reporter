# Automating construction safety inspections using a multi-modal vision-language RAG framework 

**Authors**: Chenxin Wang, Elyas Asadi Shamsabadi, Zhaohui Chen, Luming Shen, Alireza Ahmadian Fard Fini, Daniel Dias-da-Costa  

**Link**: [PDF](https://arxiv.org/pdf/2510.04145)  

**Abstract**: Conventional construction safety inspection methods are often inefficient as they require navigating through large volume of information. Recent advances in large vision-language models (LVLMs) provide opportunities to automate safety inspections through enhanced visual and linguistic understanding. However, existing applications face limitations including irrelevant or unspecific responses, restricted modal inputs and hallucinations. Utilisation of Large Language Models (LLMs) for this purpose is constrained by availability of training data and frequently lack real-time adaptability. This study introduces SiteShield, a multi-modal LVLM-based Retrieval-Augmented Generation (RAG) framework for automating construction safety inspection reports by integrating visual and audio inputs. Using real-world data, SiteShield outperformed unimodal LLMs without RAG with an F1 score of 0.82, hamming loss of 0.04, precision of 0.76, and recall of 0.96. The findings indicate that SiteShield offers a novel pathway to enhance information retrieval and efficiency in generating safety reports. 

---
# ModernBERT + ColBERT: Enhancing biomedical RAG through an advanced re-ranking retriever 

**Authors**: Eduardo Martínez Rivera, Filippo Menolascina  

**Link**: [PDF](https://arxiv.org/pdf/2510.04757)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enriching Large Language Models (LLMs) with external knowledge, allowing for factually grounded responses, a critical requirement in high-stakes domains such as healthcare. However, the efficacy of RAG systems is fundamentally restricted by the performance of their retrieval module, since irrelevant or semantically misaligned documents directly compromise the accuracy of the final generated response. General-purpose dense retrievers can struggle with the nuanced language of specialised domains, while the high accuracy of in-domain models is often achieved at prohibitive computational costs. In this work, we aim to address this trade-off by developing and evaluating a two-stage retrieval architecture that combines a lightweight ModernBERT bidirectional encoder for efficient initial candidate retrieval with a ColBERTv2 late-interaction model for fine-grained re-ranking. We conduct comprehensive evaluations of our retriever module performance and RAG system performance in the biomedical context, fine-tuning the IR module using 10k question-passage pairs from PubMedQA. Our analysis of the retriever module confirmed the positive impact of the ColBERT re-ranker, which improved Recall@3 by up to 4.2 percentage points compared to its retrieve-only counterpart. When integrated into the biomedical RAG, our IR module leads to a state-of-the-art average accuracy of 0.4448 on the five tasks of the MIRAGE question-answering benchmark, outperforming strong baselines such as MedCPT (0.4436). Our ablation studies reveal that this performance is critically dependent on a joint fine-tuning process that aligns the retriever and re-ranker; otherwise, the re-ranker might degrade the performance. 

---
# Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards 

**Authors**: Faisal Hamman, Chenyang Zhu, Anoop Kumar, Xujun Peng, Sanghamitra Dutta, Daben Liu, Alfy Samuel  

**Link**: [PDF](https://arxiv.org/pdf/2510.04392)  

**Abstract**: RAG systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator (LLM), undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, helping identify inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve Information Consistent RAG (Con-RAG), training the generator to produce consistent outputs across paraphrased queries and remain robust to retrieval-induced variability. Because exact reward computation over paraphrase sets is computationally expensive, we also introduce a scalable approximation method that retains effectiveness while enabling efficient, large-scale training. Empirical evaluations across short-form, multi-hop, and long-form QA benchmarks demonstrate that Con-RAG significantly improves both consistency and accuracy over strong baselines, even in the absence of explicit ground-truth supervision. Our work provides practical solutions for evaluating and building reliable RAG systems for safety-critical deployments. 

---
# Equipping Retrieval-Augmented Large Language Models with Document Structure Awareness 

**Authors**: Lingnan Xu, Chong Feng, Kaiyuan Zhang, Liu Zhengyong, Wenqiang Xu, Fanqing Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.04293)  

**Abstract**: While large language models (LLMs) demonstrate impressive capabilities, their reliance on parametric knowledge often leads to factual inaccuracies. Retrieval-Augmented Generation (RAG) mitigates this by leveraging external documents, yet existing approaches treat retrieved passages as isolated chunks, ignoring valuable structure that is crucial for document organization. Motivated by this gap, we propose Retrieve-DocumentRoute-Read (RDR2), a novel framework that explicitly incorporates structural information throughout the RAG process. RDR2 employs an LLM-based router to dynamically navigate document structure trees, jointly evaluating content relevance and hierarchical relationships to assemble optimal evidence. Our key innovation lies in formulating document routing as a trainable task, with automatic action curation and structure-aware passage selection inspired by human reading strategies. Through comprehensive evaluation on five challenging datasets, RDR2 achieves state-of-the-art performance, demonstrating that explicit structural awareness significantly enhances RAG systems' ability to acquire and utilize knowledge, particularly in complex scenarios requiring multi-document synthesis. 

---
# Identifying Financial Risk Information Using RAG with a Contrastive Insight 

**Authors**: Ali Elahi  

**Link**: [PDF](https://arxiv.org/pdf/2510.03521)  

**Abstract**: In specialized domains, humans often compare new problems against similar examples, highlight nuances, and draw conclusions instead of analyzing information in isolation. When applying reasoning in specialized contexts with LLMs on top of a RAG, the pipeline can capture contextually relevant information, but it is not designed to retrieve comparable cases or related problems.
While RAG is effective at extracting factual information, its outputs in specialized reasoning tasks often remain generic, reflecting broad facts rather than context-specific insights. In finance, it results in generic risks that are true for the majority of companies. To address this limitation, we propose a peer-aware comparative inference layer on top of RAG.
Our contrastive approach outperforms baseline RAG in text generation metrics such as ROUGE and BERTScore in comparison with human-generated equity research and risk. 

---
# UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG 

**Authors**: Xiangyu Peng, Cab Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03663)  

**Abstract**: Multimodal retrieval-augmented generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented, focusing on either text or images in isolation or on simplified multimodal setups that fail to capture document-centric multimodal use cases. In this paper, we introduce UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across eight domains. Our pipeline extracts and links evidence from text, tables, and figures, then generates 1,600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. To ensure reliability, 20% of QA pairs are validated by multiple annotators and expert adjudication. UniDoc-Bench supports apples-to-apples comparison across four paradigms: (1) text-only, (2) image-only, (3) multimodal text-image fusion, and (4) multimodal joint retrieval -- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. Our experiments show that multimodal text-image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding-based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines. 

---
# ContraGen: A Multi-Agent Generation Framework for Enterprise Contradictions Detection 

**Authors**: Ananya Mantravadi, Shivali Dalmia, Abhishek Mukherji, Nand Dave, Anudha Mittal  

**Link**: [PDF](https://arxiv.org/pdf/2510.03418)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates LLMs with external sources, offering advanced capabilities for information access and decision-making. However, contradictions in retrieved evidence can result in inconsistent or untrustworthy outputs, which is especially problematic in enterprise settings where compliance, governance, and accountability are critical. Existing benchmarks for contradiction detection are limited to sentence-level analysis and do not capture the complexity of enterprise documents such as contracts, financial filings, compliance reports, or policy manuals. To address this limitation, we propose ContraGen, a contradiction-aware benchmark framework tailored to enterprise domain. The framework generates synthetic enterprise-style documents with embedded contradictions, enabling systematic evaluation of both intra-document and cross-document consistency. Automated contradiction mining is combined with human-in-the-loop validation to ensure high accuracy. Our contributions include generating realistic enterprise documents, modeling a taxonomy of contradiction types common in business processes, enabling controlled creation of self- and pairwise contradictions, developing a contradiction-aware retrieval evaluation pipeline and embedding human oversight to reflect domain-specific judgment complexity. This work establishes a foundation for more trustworthy and accountable RAG systems in enterprise information-seeking applications, where detecting and resolving contradictions is essential for reducing risk and ensuring compliance. 

---
# 3Dify: a Framework for Procedural 3D-CG Generation Assisted by LLMs Using MCP and RAG 

**Authors**: Shun-ichiro Hayashi, Daichi Mukunoki, Tetsuya Hoshino, Satoshi Ohshima, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.04536)  

**Abstract**: This paper proposes "3Dify," a procedural 3D computer graphics (3D-CG) generation framework utilizing Large Language Models (LLMs). The framework enables users to generate 3D-CG content solely through natural language instructions. 3Dify is built upon Dify, an open-source platform for AI application development, and incorporates several state-of-the-art LLM-related technologies such as the Model Context Protocol (MCP) and Retrieval-Augmented Generation (RAG). For 3D-CG generation support, 3Dify automates the operation of various Digital Content Creation (DCC) tools via MCP. When DCC tools do not support MCP-based interaction, the framework employs the Computer-Using Agent (CUA) method to automate Graphical User Interface (GUI) operations. Moreover, to enhance image generation quality, 3Dify allows users to provide feedback by selecting preferred images from multiple candidates. The LLM then learns variable patterns from these selections and applies them to subsequent generations. Furthermore, 3Dify supports the integration of locally deployed LLMs, enabling users to utilize custom-developed models and to reduce both time and monetary costs associated with external API calls by leveraging their own computational resources. 

---
