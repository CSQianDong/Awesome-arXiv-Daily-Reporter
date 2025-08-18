# INFNet: A Task-aware Information Flow Network for Large-Scale Recommendation Systems 

**Authors**: Kaiyuan Li, Dongdong Mao, Yongxiang Tang, Yanhua Cheng, Yanxiang Zeng, Chao Wang, Xialong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11565)  

**Abstract**: Feature interaction has long been a cornerstone of ranking models in large-scale recommender systems due to its proven effectiveness in capturing complex dependencies among features. However, existing feature interaction strategies face two critical challenges in industrial applications: (1) The vast number of categorical and sequential features makes exhaustive interaction computationally prohibitive, often resulting in optimization difficulties. (2) Real-world recommender systems typically involve multiple prediction objectives, yet most current approaches apply feature interaction modules prior to the multi-task learning layers. This late-fusion design overlooks task-specific feature dependencies and inherently limits the capacity of multi-task modeling. To address these limitations, we propose the Information Flow Network (INFNet), a task-aware architecture designed for large-scale recommendation scenarios. INFNet distinguishes features into three token types, categorical tokens, sequence tokens, and task tokens, and introduces a novel dual-flow design comprising heterogeneous and homogeneous alternating information blocks. For heterogeneous information flow, we employ a cross-attention mechanism with proxy that facilitates efficient cross-modal token interaction with balanced computational cost. For homogeneous flow, we design type-specific Proxy Gated Units (PGUs) to enable fine-grained intra-type feature processing. Extensive experiments on multiple offline benchmarks confirm that INFNet achieves state-of-the-art performance. Moreover, INFNet has been successfully deployed in a commercial online advertising system, yielding significant gains of +1.587% in Revenue (REV) and +1.155% in Click-Through Rate (CTR). 

---
# Mitigating Filter Bubble from the Perspective of Community Detection: A Universal Framework 

**Authors**: Ming Tang, Xiaowen Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11239)  

**Abstract**: In recent years, recommender systems have primarily focused on improving accuracy at the expense of diversity, which exacerbates the well-known filter bubble effect. This paper proposes a universal framework called CD-CGCN to address the filter bubble issue in recommender systems from a community detection perspective. By analyzing user-item interaction histories with a community detection algorithm, we reveal that state-of-the-art recommendations often focus on intra-community items, worsening the filter bubble effect. CD-CGCN, a model-agnostic framework, integrates a Conditional Discriminator and a Community-reweighted Graph Convolutional Network which can be plugged into most recommender models. Using adversarial learning based on community labels, it counteracts the extracted community attributes and incorporates an inference strategy tailored to the user's specific filter bubble state. Extensive experiments on real-world datasets with multiple base models validate its effectiveness in mitigating filter bubbles while preserving recommendation quality. Additionally, by applying community debiasing to the original test set to construct an unbiased test set, we observe that CD-CGCN demonstrates superior performance in capturing users' inter-community preferences. 

---
# Representation Quantization for Collaborative Filtering Augmentation 

**Authors**: Yunze Luo, Yinjie Jiang, Gaode Chen, Jingchi Wang, Shicheng Wang, Ruina Sun, Jiang Yuezihan, Jun Zhang, Jian Liang, Han Li, Kun Gai, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2508.11194)  

**Abstract**: As the core algorithm in recommendation systems, collaborative filtering (CF) algorithms inevitably face the problem of data sparsity. Since CF captures similar users and items for recommendations, it is effective to augment the lacking user-user and item-item homogeneous linkages. However, existing methods are typically limited to connecting through overlapping interacted neighbors or through similar attributes and contents. These approaches are constrained by coarse-grained, sparse attributes and fail to effectively extract behavioral characteristics jointly from interaction sequences and attributes. To address these challenges, we propose a novel two-stage collaborative recommendation algorithm, DQRec: Decomposition-based Quantized Variational AutoEncoder (DQ-VAE) for Recommendation. DQRec augments features and homogeneous linkages by extracting the behavior characteristics jointly from interaction sequences and attributes, namely patterns, such as user multi-aspect interests. Inspired by vector quantization (VQ) technology, we propose a new VQ algorithm, DQ-VAE, which decomposes the pre-trained representation embeddings into distinct dimensions, and quantize them to generates semantic IDs. We utilize the generated semantic IDs as the extracted patterns mentioned above. By integrating these semantic ID patterns into the recommendation process through feature and linkage augmentation, the system enriches both latent and explicit user and item features, identifies pattern-similar neighbors, and thereby improves the efficiency of information diffusion. Experimental comparisons with baselines across multiple datasets demonstrate the superior performance of the proposed DQRec method. 

---
# Role-Augmented Intent-Driven Generative Search Engine Optimization 

**Authors**: Xiaolu Chen, Haojie Wu, Jie Bao, Zhen Chen, Yong Liao, Hu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11158)  

**Abstract**: Generative Search Engines (GSEs), powered by Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), are reshaping information retrieval. While commercial systems (e.g., BingChat, this http URL) demonstrate impressive semantic synthesis capabilities, their black-box nature fundamentally undermines established Search Engine Optimization (SEO) practices. Content creators face a critical challenge: their optimization strategies, effective in traditional search engines, are misaligned with generative retrieval contexts, resulting in diminished visibility. To bridge this gap, we propose a Role-Augmented Intent-Driven Generative Search Engine Optimization (G-SEO) method, providing a structured optimization pathway tailored for GSE scenarios. Our method models search intent through reflective refinement across diverse informational roles, enabling targeted content enhancement. To better evaluate the method under realistic settings, we address the benchmarking limitations of prior work by: (1) extending the GEO dataset with diversified query variations reflecting real-world search scenarios and (2) introducing G-Eval 2.0, a 6-level LLM-augmented evaluation rubric for fine-grained human-aligned assessment. Experimental results demonstrate that search intent serves as an effective signal for guiding content optimization, yielding significant improvements over single-aspect baseline approaches in both subjective impressions and objective content visibility within GSE responses. 

---
# +VeriRel: Verification Feedback to Enhance Document Retrieval for Scientific Fact Checking 

**Authors**: Xingyu Deng, Xi Wang, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2508.11122)  

**Abstract**: Identification of appropriate supporting evidence is critical to the success of scientific fact checking. However, existing approaches rely on off-the-shelf Information Retrieval algorithms that rank documents based on relevance rather than the evidence they provide to support or refute the claim being checked. This paper proposes +VeriRel which includes verification success in the document ranking. Experimental results on three scientific fact checking datasets (SciFact, SciFact-Open and Check-Covid) demonstrate consistently leading performance by +VeriRel for document evidence retrieval and a positive impact on downstream verification. This study highlights the potential of integrating verification feedback to document relevance assessment for effective scientific fact checking systems. It shows promising future work to evaluate fine-grained relevance when examining complex documents for advanced scientific fact checking. 

---
# PaperRegister: Boosting Flexible-grained Paper Search via Hierarchical Register Indexing 

**Authors**: Zhuoqun Li, Xuanang Chen, Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.11116)  

**Abstract**: Paper search is an important activity for researchers, typically involving using a query with description of a topic to find relevant papers. As research deepens, paper search requirements may become more flexible, sometimes involving specific details such as module configuration rather than being limited to coarse-grained topics. However, previous paper search systems are unable to meet these flexible-grained requirements, as these systems mainly collect paper abstracts to construct index of corpus, which lack detailed information to support retrieval by finer-grained queries. In this work, we propose PaperRegister, consisted of offline hierarchical indexing and online adaptive retrieval, transforming traditional abstract-based index into hierarchical index tree for paper search, thereby supporting queries at flexible granularity. Experiments on paper search tasks across a range of granularity demonstrate that PaperRegister achieves the state-of-the-art performance, and particularly excels in fine-grained scenarios, highlighting the good potential as an effective solution for flexible-grained paper search in real-world applications. Code for this work is in this https URL. 

---
# Pretrained Conformers for Audio Fingerprinting and Retrieval 

**Authors**: Kemal Altwlkany, Elmedin Selmanovic, Sead Delalic  

**Link**: [PDF](https://arxiv.org/pdf/2508.11609)  

**Abstract**: Conformers have shown great results in speech processing due to their ability to capture both local and global interactions. In this work, we utilize a self-supervised contrastive learning framework to train conformer-based encoders that are capable of generating unique embeddings for small segments of audio, generalizing well to previously unseen data. We achieve state-of-the-art results for audio retrieval tasks while using only 3 seconds of audio to generate embeddings. Our models are almost completely immune to temporal misalignments and achieve state-of-the-art results in cases of other audio distortions such as noise, reverb or extreme temporal stretching. Code and models are made publicly available and the results are easy to reproduce as we train and test using popular and freely available datasets of different sizes. 

---
# TrajSV: A Trajectory-based Model for Sports Video Representations and Applications 

**Authors**: Zheng Wang, Shihao Xu, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.11569)  

**Abstract**: Sports analytics has received significant attention from both academia and industry in recent years. Despite the growing interest and efforts in this field, several issues remain unresolved, including (1) data unavailability, (2) lack of an effective trajectory-based framework, and (3) requirement for sufficient supervision labels. In this paper, we present TrajSV, a trajectory-based framework that addresses various issues in existing studies. TrajSV comprises three components: data preprocessing, Clip Representation Network (CRNet), and Video Representation Network (VRNet). The data preprocessing module extracts player and ball trajectories from sports broadcast videos. CRNet utilizes a trajectory-enhanced Transformer module to learn clip representations based on these trajectories. Additionally, VRNet learns video representations by aggregating clip representations and visual features with an encoder-decoder architecture. Finally, a triple contrastive loss is introduced to optimize both video and clip representations in an unsupervised manner. The experiments are conducted on three broadcast video datasets to verify the effectiveness of TrajSV for three types of sports (i.e., soccer, basketball, and volleyball) with three downstream applications (i.e., sports video retrieval, action spotting, and video captioning). The results demonstrate that TrajSV achieves state-of-the-art performance in sports video retrieval, showcasing a nearly 70% improvement. It outperforms baselines in action spotting, achieving state-of-the-art results in 9 out of 17 action categories, and demonstrates a nearly 20% improvement in video captioning. Additionally, we introduce a deployed system along with the three applications based on TrajSV. 

---
# When Algorithms Mirror Minds: A Confirmation-Aware Social Dynamic Model of Echo Chamber and Homogenization Traps 

**Authors**: Ming Tang, Xiaowen Huang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11516)  

**Abstract**: Recommender systems increasingly suffer from echo chambers and user homogenization, systemic distortions arising from the dynamic interplay between algorithmic recommendations and human behavior. While prior work has studied these phenomena through the lens of algorithmic bias or social network structure, we argue that the psychological mechanisms of users and the closed-loop interaction between users and recommenders are critical yet understudied drivers of these emergent effects. To bridge this gap, we propose the Confirmation-Aware Social Dynamic Model which incorporates user psychology and social relationships to simulate the actual user and recommender interaction process. Our theoretical analysis proves that echo chambers and homogenization traps, defined respectively as reduced recommendation diversity and homogenized user representations, will inevitably occur. We also conduct extensive empirical simulations on two real-world datasets and one synthetic dataset with five well-designed metrics, exploring the root factors influencing the aforementioned phenomena from three level perspectives: the stochasticity and social integration degree of recommender (system-level), the psychological mechanisms of users (user-level), and the dataset scale (platform-level). Furthermore, we demonstrate four practical mitigation strategies that help alleviate echo chambers and user homogenization at the cost of some recommendation accuracy. Our findings provide both theoretical and empirical insights into the emergence and drivers of echo chambers and user homogenization, as well as actionable guidelines for human-centered recommender design. 

---
# Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis 

**Authors**: Mithat Can Ozgun, Jiahuan Pei, Koen Hindriks, Lucia Donatelli, Qingzhi Liu, Xin Sun, Junxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11398)  

**Abstract**: LLM-based agents have emerged as transformative tools capable of executing complex tasks through iterative planning and action, achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced. 

---
# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems 

**Authors**: Beichen Guo, Zhiyuan Wen, Yu Yang, Peng Gao, Ruosong Yang, Jiaxing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.11310)  

**Abstract**: The growing interest in automatic survey generation (ASG), a task that traditionally required considerable time and effort, has been spurred by recent advances in large language models (LLMs). With advancements in retrieval-augmented generation (RAG) and the rising popularity of multi-agent systems (MASs), synthesizing academic surveys using LLMs has become a viable approach, thereby elevating the need for robust evaluation methods in this domain. However, existing evaluation methods suffer from several limitations, including biased metrics, a lack of human preference, and an over-reliance on LLMs-as-judges. To address these challenges, we propose SGSimEval, a comprehensive benchmark for Survey Generation with Similarity-Enhanced Evaluation that evaluates automatic survey generation systems by integrating assessments of the outline, content, and references, and also combines LLM-based scoring with quantitative metrics to provide a multifaceted evaluation framework. In SGSimEval, we also introduce human preference metrics that emphasize both inherent quality and similarity to humans. Extensive experiments reveal that current ASG systems demonstrate human-comparable superiority in outline generation, while showing significant room for improvement in content and reference generation, and our evaluation metrics maintain strong consistency with human assessments. 

---
# Beyond Solving Math Quiz: Evaluating the Ability of Large Reasoning Models to Ask for Information 

**Authors**: Youcheng Huang, Bowen Qin, Chen Huang, Duanyu Feng, Xi Yang, Wenqiang Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.11252)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable problem-solving abilities in mathematics, as evaluated by existing benchmarks exclusively on well-defined problems. However, such evaluation setup constitutes a critical gap, since a genuine intelligent agent should not only solve problems (as a math quiz solver), but also be able~to ask for information when the problems lack sufficient information, enabling proactivity in responding users' requests. To bridge such gap, we proposes a new dataset consisting of two types of incomplete problems with diverse contexts. Based on the dataset, our systematical evaluation of LRMs reveals their inability in proactively asking for information. In addition, we uncover the behaviors related to overthinking and hallucination of LRMs, and highlight the potential and challenges of supervised fine-tuning in learning such ability. We hope to provide new insights in developing LRMs with genuine intelligence, rather than just solving problems. 

---
# RAG for Geoscience: What We Expect, Gaps and Opportunities 

**Authors**: Runlong Yu, Shiyuan Luo, Rahul Ghosh, Lingyao Li, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.11246)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by combining retrieval with generation. However, its current workflow remains largely text-centric, limiting its applicability in geoscience. Many geoscientific tasks are inherently evidence-hungry. Typical examples involve imputing missing observations using analog scenes, retrieving equations and parameters to calibrate models, geolocating field photos based on visual cues, or surfacing historical case studies to support policy analyses. A simple ``retrieve-then-generate'' pipeline is insufficient for these needs. We envision Geo-RAG, a next-generation paradigm that reimagines RAG as a modular retrieve $\rightarrow$ reason $\rightarrow$ generate $\rightarrow$ verify loop. Geo-RAG supports four core capabilities: (i) retrieval of multi-modal Earth data; (ii) reasoning under physical and domain constraints; (iii) generation of science-grade artifacts; and (iv) verification of generated hypotheses against numerical models, ground measurements, and expert assessments. This shift opens new opportunities for more trustworthy and transparent geoscience workflows. 

---
# ORFuzz: Fuzzing the "Other Side" of LLM Safety -- Testing Over-Refusal 

**Authors**: Haonan Zhang, Dongxia Wang, Yi Liu, Kexin Chen, Jiashui Wang, Xinlei Ying, Long Liu, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.11222)  

**Abstract**: Large Language Models (LLMs) increasingly exhibit over-refusal - erroneously rejecting benign queries due to overly conservative safety measures - a critical functional flaw that undermines their reliability and usability. Current methods for testing this behavior are demonstrably inadequate, suffering from flawed benchmarks and limited test generation capabilities, as highlighted by our empirical user study. To the best of our knowledge, this paper introduces the first evolutionary testing framework, ORFuzz, for the systematic detection and analysis of LLM over-refusals. ORFuzz uniquely integrates three core components: (1) safety category-aware seed selection for comprehensive test coverage, (2) adaptive mutator optimization using reasoning LLMs to generate effective test cases, and (3) OR-Judge, a human-aligned judge model validated to accurately reflect user perception of toxicity and refusal. Our extensive evaluations demonstrate that ORFuzz generates diverse, validated over-refusal instances at a rate (6.98% average) more than double that of leading baselines, effectively uncovering vulnerabilities. Furthermore, ORFuzz's outputs form the basis of ORFuzzSet, a new benchmark of 1,855 highly transferable test cases that achieves a superior 63.56% average over-refusal rate across 10 diverse LLMs, significantly outperforming existing datasets. ORFuzz and ORFuzzSet provide a robust automated testing framework and a valuable community resource, paving the way for developing more reliable and trustworthy LLM-based software systems. 

---
# Hybrid-Hierarchical Fashion Graph Attention Network for Compatibility-Oriented and Personalized Outfit Recommendation 

**Authors**: Sajjad Saed, Babak Teimourpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.11105)  

**Abstract**: The rapid expansion of the fashion industry and the growing variety of products have made it challenging for users to find compatible items on e-commerce platforms. Effective fashion recommendation systems are crucial for filtering irrelevant items and suggesting suitable ones. However, simultaneously addressing outfit compatibility and personalized recommendations remains a significant challenge, as these aspects are often treated independently in existing studies, often overlooking the complex interactions between items and user preferences. This research introduces a new framework named FGAT, inspired by the HFGN model, which leverages graph neural networks and graph attention mechanisms to tackle this issue. The proposed framework constructs a three-tier hierarchical graph of users, outfits, and items, integrating visual and textual features to simultaneously model outfit compatibility and user preferences. A graph attention mechanism dynamically weights node importance during representation propagation, enabling the capture of key interactions and generating precise representations for both user preferences and outfit compatibility. Evaluated on the POG dataset, FGAT outperforms baseline models such as HFGN, achieving improved results in precision, HR, recall, NDCG, and this http URL results demonstrate that combining multimodal visual-textual features with a hierarchical graph structure and attention mechanisms significantly enhances the accuracy and efficiency of personalized fashion recommendation systems. 

---
# Relative Advantage Debiasing for Watch-Time Prediction in Short-Video Recommendation 

**Authors**: Emily Liu, Kuan Han, Minfeng Zhan, Bocheng Zhao, Guanyu Mu, Yang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.11086)  

**Abstract**: Watch time is widely used as a proxy for user satisfaction in video recommendation platforms. However, raw watch times are influenced by confounding factors such as video duration, popularity, and individual user behaviors, potentially distorting preference signals and resulting in biased recommendation models. We propose a novel relative advantage debiasing framework that corrects watch time by comparing it to empirically derived reference distributions conditioned on user and item groups. This approach yields a quantile-based preference signal and introduces a two-stage architecture that explicitly separates distribution estimation from preference learning. Additionally, we present distributional embeddings to efficiently parameterize watch-time quantiles without requiring online sampling or storage of historical data. Both offline and online experiments demonstrate significant improvements in recommendation accuracy and robustness compared to existing baseline methods. 

---
# JobPulse: A Big Data Approach to Real-Time Engineering Workforce Analysis and National Industrial Policy 

**Authors**: Karen S. Markel, Mihir Tale, Andrea Belz  

**Link**: [PDF](https://arxiv.org/pdf/2508.11014)  

**Abstract**: Employment on a societal scale contributes heavily to national and global affairs; consequently, job openings and unemployment estimates provide important information to financial markets and governments alike. However, such reports often describe only the supply (employee job seeker) side of the job market, and skill mismatches are poorly understood. Job postings aggregated on recruiting platforms illuminate marketplace demand, but to date have primarily focused on candidate skills described in their personal profiles. In this paper, we report on a big data approach to estimating job market mismatches by focusing on demand, as represented in publicly available job postings. We use commercially available web scraping tools and a new data processing scheme to build a job posting data set for the semiconductor industry, a strategically critical sector of the United States economy; we focus on Southern California as a central hub of advanced technologies. We report on the employer base and relative needs of various job functions. Our work contributes on three fronts: First, we provide nearly real-time insight into workforce demand; second, we discuss disambiguation and semantic challenges in analysis of employer data bases at scale; and third, we report on the Southern California semiconductor engineering ecosystem. 

---
