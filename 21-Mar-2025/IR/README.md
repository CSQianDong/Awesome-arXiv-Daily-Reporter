# Diffusion-augmented Graph Contrastive Learning for Collaborative Filter 

**Authors**: Fan Huang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16290)  

**Abstract**: Graph-based collaborative filtering has been established as a prominent approach in recommendation systems, leveraging the inherent graph topology of user-item interactions to model high-order connectivity patterns and enhance recommendation performance. Recent advances in Graph Contrastive Learning (GCL) have demonstrated promising potential to alleviate data sparsity issues by improving representation learning through contrastive view generation and mutual information maximization. However, existing approaches lack effective data augmentation strategies. Structural augmentation risks distorting fundamental graph topology, while feature-level perturbation techniques predominantly employ uniform noise scales that fail to account for node-specific characteristics. To solve these challenges, we propose Diffusion-augmented Contrastive Learning (DGCL), an innovative framework that integrates diffusion models with contrastive learning for enhanced collaborative filtering. Our approach employs a diffusion process that learns node-specific Gaussian distributions of representations, thereby generating semantically consistent yet diversified contrastive views through reverse diffusion sampling. DGCL facilitates adaptive data augmentation based on reconstructed representations, considering both semantic coherence and node-specific features. In addition, it explores unrepresented regions of the latent sparse feature space, thereby enriching the diversity of contrastive views. Extensive experimental results demonstrate the effectiveness of DGCL on three public datasets. 

---
# Narrative Trails: A Method for Coherent Storyline Extraction via Maximum Capacity Path Optimization 

**Authors**: Fausto German, Brian Keith, Chris North  

**Link**: [PDF](https://arxiv.org/pdf/2503.15681)  

**Abstract**: Traditional information retrieval is primarily concerned with finding relevant information from large datasets without imposing a structure within the retrieved pieces of data. However, structuring information in the form of narratives--ordered sets of documents that form coherent storylines--allows us to identify, interpret, and share insights about the connections and relationships between the ideas presented in the data. Despite their significance, current approaches for algorithmically extracting storylines from data are scarce, with existing methods primarily relying on intricate word-based heuristics and auxiliary document structures. Moreover, many of these methods are difficult to scale to large datasets and general contexts, as they are designed to extract storylines for narrow tasks. In this paper, we propose Narrative Trails, an efficient, general-purpose method for extracting coherent storylines in large text corpora. Specifically, our method uses the semantic-level information embedded in the latent space of deep learning models to build a sparse coherence graph and extract narratives that maximize the minimum coherence of the storylines. By quantitatively evaluating our proposed methods on two distinct narrative extraction tasks, we show the generalizability and scalability of Narrative Trails in multiple contexts while also simplifying the extraction pipeline. 

---
# CaKE: Circuit-aware Editing Enables Generalizable Knowledge Learners 

**Authors**: Yunzhi Yao, Jizhan Fang, Jia-Chen Gu, Ningyu Zhang, Shumin Deng, Huajun Chen, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.16356)  

**Abstract**: Knowledge Editing (KE) enables the modification of outdated or incorrect information in large language models (LLMs). While existing KE methods can update isolated facts, they struggle to generalize these updates to multi-hop reasoning tasks that depend on the modified knowledge. Through an analysis of reasoning circuits -- the neural pathways LLMs use for knowledge-based inference, we observe that current layer-localized KE approaches, such as MEMIT and WISE, which edit only single or a few model layers, struggle to effectively incorporate updated information into these reasoning pathways. To address this limitation, we propose CaKE (Circuit-aware Knowledge Editing), a novel method that enables more effective integration of updated knowledge in LLMs. CaKE leverages strategically curated data, guided by our circuits-based analysis, that enforces the model to utilize the modified knowledge, stimulating the model to develop appropriate reasoning circuits for newly integrated knowledge. Experimental results show that CaKE enables more accurate and consistent use of updated knowledge across related reasoning tasks, leading to an average of 20% improvement in multi-hop reasoning accuracy on MQuAKE dataset compared to existing KE methods. We release the code and data in this https URL. 

---
# Iterative Optimal Attention and Local Model for Single Image Rain Streak Removal 

**Authors**: Xiangyu Li, Wanshu Fan, Yue Shen, Cong Wang, Wei Wang, Xin Yang, Qiang Zhang, Dongsheng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.16165)  

**Abstract**: High-fidelity imaging is crucial for the successful safety supervision and intelligent deployment of vision-based measurement systems (VBMS). It ensures high-quality imaging in VBMS, which is fundamental for reliable visual measurement and analysis. However, imaging quality can be significantly impaired by adverse weather conditions, particularly rain, leading to blurred images and reduced contrast. Such impairments increase the risk of inaccurate evaluations and misinterpretations in VBMS. To address these limitations, we propose an Expectation Maximization Reconstruction Transformer (EMResformer) for single image rain streak removal. The EMResformer retains the key self-attention values for feature aggregation, enhancing local features to produce superior image reconstruction. Specifically, we propose an Expectation Maximization Block seamlessly integrated into the single image rain streak removal network, enhancing its ability to eliminate superfluous information and restore a cleaner background image. Additionally, to further enhance local information for improved detail rendition, we introduce a Local Model Residual Block, which integrates two local model blocks along with a sequence of convolutions and activation functions. This integration synergistically facilitates the extraction of more pertinent features for enhanced single image rain streak removal. Extensive experiments validate that our proposed EMResformer surpasses current state-of-the-art single image rain streak removal methods on both synthetic and real-world datasets, achieving an improved balance between model complexity and single image deraining performance. Furthermore, we evaluate the effectiveness of our method in VBMS scenarios, demonstrating that high-quality imaging significantly improves the accuracy and reliability of VBMS tasks. 

---
# OThink-MR1: Stimulating multimodal generalized reasoning capabilities through dynamic reinforcement learning 

**Authors**: Zhiyuan Liu, Yuting Zhang, Feng Liu, Changwang Zhang, Ying Sun, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16081)  

**Abstract**: Multimodal Language Models have gained significant traction for their ability to process diverse input data types and generate coherent, contextually relevant outputs across various applications. While supervised fine-tuning (SFT) has been the predominant approach to enhance MLLM capabilities in task-specific optimization, it often falls short in fostering crucial generalized reasoning abilities. Despite the potential of reinforcement learning (RL) to address these limitations, it faces two issues: (1) its generalized capabilities in multimodal tasks remain underexplored. (2) its training constraints such as constant Kullback-Leibler or clamp strategy easily lead to suboptimal bottleneck. To adress these issues, we introduce OThink-MR1, a framework that extends RL to MLLMs, enabling them to achieve deeper understanding and reasoning across multimodal tasks. We design a dynamic Kullback-Leibler strategy that significantly enhances RL performance, surpassing SFT in same-task evaluations. Also, we are the first to reveal that RL exhibits remarkable cross-task generalization capabilities, which shows that models post-trained with RL on one multimodal task can be effectively transfered to another tasks. Finally, extensive experiments demonstrate the great reasoning ability of our proposed OThink-MR1. 

---
# Tuning LLMs by RAG Principles: Towards LLM-native Memory 

**Authors**: Jiale Wei, Shuchi Wu, Ruochen Liu, Xiang Ying, Jingbo Shang, Fangbo Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16071)  

**Abstract**: Memory, additional information beyond the training of large language models (LLMs), is crucial to various real-world applications, such as personal assistant. The two mainstream solutions to incorporate memory into the generation process are long-context LLMs and retrieval-augmented generation (RAG). In this paper, we first systematically compare these two types of solutions on three renovated/new datasets and show that (1) long-context solutions, although more expensive, shall be easier to capture the big picture and better answer queries which require considering the memory as a whole; and (2) when the queries concern specific information, RAG solutions shall be more competitive especially when the keywords can be explicitly matched. Therefore, we propose a novel method RAG-Tuned-LLM which fine-tunes a relative small (e.g., 7B) LLM using the data generated following the RAG principles, so it can combine the advantages of both solutions. Extensive experiments on three datasets demonstrate that RAG-Tuned-LLM can beat long-context LLMs and RAG methods across a wide range of query types. 

---
# PromptHash: Affinity-Prompted Collaborative Cross-Modal Learning for Adaptive Hashing Retrieval 

**Authors**: Qiang Zou, Shuli Cheng, Jiayi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.16064)  

**Abstract**: Cross-modal hashing is a promising approach for efficient data retrieval and storage optimization. However, contemporary methods exhibit significant limitations in semantic preservation, contextual integrity, and information redundancy, which constrains retrieval efficacy. We present PromptHash, an innovative framework leveraging affinity prompt-aware collaborative learning for adaptive cross-modal hashing. We propose an end-to-end framework for affinity-prompted collaborative hashing, with the following fundamental technical contributions: (i) a text affinity prompt learning mechanism that preserves contextual information while maintaining parameter efficiency, (ii) an adaptive gated selection fusion architecture that synthesizes State Space Model with Transformer network for precise cross-modal feature integration, and (iii) a prompt affinity alignment strategy that bridges modal heterogeneity through hierarchical contrastive learning. To the best of our knowledge, this study presents the first investigation into affinity prompt awareness within collaborative cross-modal adaptive hash learning, establishing a paradigm for enhanced semantic consistency across modalities. Through comprehensive evaluation on three benchmark multi-label datasets, PromptHash demonstrates substantial performance improvements over existing approaches. Notably, on the NUS-WIDE dataset, our method achieves significant gains of 18.22% and 18.65% in image-to-text and text-to-image retrieval tasks, respectively. The code is publicly available at this https URL. 

---
# Typed-RAG: Type-aware Multi-Aspect Decomposition for Non-Factoid Question Answering 

**Authors**: DongGeon Lee, Ahjeong Park, Hyeri Lee, Hyeonseo Nam, Yunho Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2503.15879)  

**Abstract**: Non-factoid question-answering (NFQA) poses a significant challenge due to its open-ended nature, diverse intents, and the need for multi-aspect reasoning, which renders conventional factoid QA approaches, including retrieval-augmented generation (RAG), inadequate. Unlike factoid questions, non-factoid questions (NFQs) lack definitive answers and require synthesizing information from multiple sources across various reasoning dimensions. To address these limitations, we introduce Typed-RAG, a type-aware multi-aspect decomposition framework within the RAG paradigm for NFQA. Typed-RAG classifies NFQs into distinct types -- such as debate, experience, and comparison -- and applies aspect-based decomposition to refine retrieval and generation strategies. By decomposing multi-aspect NFQs into single-aspect sub-queries and aggregating the results, Typed-RAG generates more informative and contextually relevant responses. To evaluate Typed-RAG, we introduce Wiki-NFQA, a benchmark dataset covering diverse NFQ types. Experimental results demonstrate that Typed-RAG outperforms baselines, thereby highlighting the importance of type-aware decomposition for effective retrieval and generation in NFQA. Our code and dataset are available at \href{this https URL}{this https URL}. 

---
# LLM-Aided Customizable Profiling of Code Data Based On Programming Language Concepts 

**Authors**: Pankaj Thorat, Adnan Qidwai, Adrija Dhar, Aishwariya Chakraborty, Anand Eswaran, Hima Patel, Praveen Jayachandran  

**Link**: [PDF](https://arxiv.org/pdf/2503.15571)  

**Abstract**: Data profiling is critical in machine learning for generating descriptive statistics, supporting both deeper understanding and downstream tasks like data valuation and curation. This work addresses profiling specifically in the context of code datasets for Large Language Models (code-LLMs), where data quality directly influences tasks such as code generation and summarization. Characterizing code datasets in terms of programming language concepts enables better insights and targeted data curation. Our proposed methodology decomposes code data profiling into two phases: (1) an offline phase where LLMs are leveraged to derive and learn rules for extracting syntactic and semantic concepts across various programming languages, including previously unseen or low-resource languages, and (2) an online deterministic phase applying these derived rules for efficient real-time analysis. This hybrid approach is customizable, extensible to new syntactic and semantic constructs, and scalable to multiple languages. Experimentally, our LLM-aided method achieves a mean accuracy of 90.33% for syntactic extraction rules and semantic classification accuracies averaging 80% and 77% across languages and semantic concepts, respectively. 

---
# Rendering Transparency to Ranking in Educational Assessment via Bayesian Comparative Judgement 

**Authors**: Andy Gray, Alma Rahat, Stephen Lindsay, Jen Pearson, Tom Crick  

**Link**: [PDF](https://arxiv.org/pdf/2503.15549)  

**Abstract**: Ensuring transparency in educational assessment is increasingly critical, particularly post-pandemic, as demand grows for fairer and more reliable evaluation methods. Comparative Judgement (CJ) offers a promising alternative to traditional assessments, yet concerns remain about its perceived opacity. This paper examines how Bayesian Comparative Judgement (BCJ) enhances transparency by integrating prior information into the judgement process, providing a structured, data-driven approach that improves interpretability and accountability.
BCJ assigns probabilities to judgement outcomes, offering quantifiable measures of uncertainty and deeper insights into decision confidence. By systematically tracking how prior data and successive judgements inform final rankings, BCJ clarifies the assessment process and helps identify assessor disagreements. Multi-criteria BCJ extends this by evaluating multiple learning outcomes (LOs) independently, preserving the richness of CJ while producing transparent, granular rankings aligned with specific assessment goals. It also enables a holistic ranking derived from individual LOs, ensuring comprehensive evaluations without compromising detailed feedback.
Using a real higher education dataset with professional markers in the UK, we demonstrate BCJ's quantitative rigour and ability to clarify ranking rationales. Through qualitative analysis and discussions with experienced CJ practitioners, we explore its effectiveness in contexts where transparency is crucial, such as high-stakes national assessments. We highlight the benefits and limitations of BCJ, offering insights into its real-world application across various educational settings. 

---
