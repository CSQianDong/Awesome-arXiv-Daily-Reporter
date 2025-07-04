# Alleviating User-Sensitive bias with Fair Generative Sequential Recommendation Model 

**Authors**: Yang Liu, Feng Wu, Xuefang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19777)  

**Abstract**: Recommendation fairness has recently attracted much attention. In the real world, recommendation systems are driven by user behavior, and since users with the same sensitive feature (e.g., gender and age) tend to have the same patterns, recommendation models can easily capture the strong correlation preference of sensitive features and thus cause recommendation unfairness. Diffusion model (DM) as a new generative model paradigm has achieved great success in recommendation systems. DM's ability to model uncertainty and represent diversity, and its modeling mechanism has a high degree of adaptability with the real-world recommendation process with bias. Therefore, we use DM to effectively model the fairness of recommendation and enhance the diversity. This paper proposes a FairGENerative sequential Recommendation model based on DM, FairGENRec. In the training phase, we inject random noise into the original distribution under the guidance of the sensitive feature recognition model, and a sequential denoise model is designed for the reverse reconstruction of items. Simultaneously, recommendation fairness modeling is completed by injecting multi-interests representational information that eliminates the bias of sensitive user features into the generated results. In the inference phase, the model obtains the noise in the form of noise addition by using the history interactions which is followed by reverse iteration to reconstruct the target item representation. Finally, our extensive experiments on three datasets demonstrate the dual enhancement effect of FairGENRec on accuracy and fairness, while the statistical analysis of the cases visualizes the degree of improvement on the fairness of the recommendation. 

---
# NEAR$^2$: A Nested Embedding Approach to Efficient Product Retrieval and Ranking 

**Authors**: Shenbin Qian, Diptesh Kanojia, Samarth Agrawal, Hadeel Saadany, Swapnil Bhosale, Constantin Orasan, Zhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19743)  

**Abstract**: E-commerce information retrieval (IR) systems struggle to simultaneously achieve high accuracy in interpreting complex user queries and maintain efficient processing of vast product catalogs. The dual challenge lies in precisely matching user intent with relevant products while managing the computational demands of real-time search across massive inventories. In this paper, we propose a Nested Embedding Approach to product Retrieval and Ranking, called NEAR$^2$, which can achieve up to $12$ times efficiency in embedding size at inference time while introducing no extra cost in training and improving performance in accuracy for various encoder-based Transformer models. We validate our approach using different loss functions for the retrieval and ranking task, including multiple negative ranking loss and online contrastive loss, on four different test sets with various IR challenges such as short and implicit queries. Our approach achieves an improved performance over a smaller embedding dimension, compared to any existing models. 

---
# From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents 

**Authors**: Weizhi Zhang, Yangning Li, Yuanchen Bei, Junyu Luo, Guancheng Wan, Liangwei Yang, Chenxuan Xie, Yuyao Yang, Wei-Chieh Huang, Chunyu Miao, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Yankai Chen, Chunkit Chan, Peilin Zhou, Xinyang Zhang, Chenwei Zhang, Jingbo Shang, Ming Zhang, Yangqiu Song, Irwin King, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18959)  

**Abstract**: Information retrieval is a cornerstone of modern knowledge acquisition, enabling billions of queries each day across diverse domains. However, traditional keyword-based search engines are increasingly inadequate for handling complex, multi-step information needs. Our position is that Large Language Models (LLMs), endowed with reasoning and agentic capabilities, are ushering in a new paradigm termed Agentic Deep Research. These systems transcend conventional information search techniques by tightly integrating autonomous reasoning, iterative retrieval, and information synthesis into a dynamic feedback loop. We trace the evolution from static web search to interactive, agent-based systems that plan, explore, and learn. We also introduce a test-time scaling law to formalize the impact of computational depth on reasoning and search. Supported by benchmark results and the rise of open-source implementations, we demonstrate that Agentic Deep Research not only significantly outperforms existing approaches, but is also poised to become the dominant paradigm for future information seeking. All the related resources, including industry products, research papers, benchmark datasets, and open-source implementations, are collected for the community in this https URL. 

---
# KnowML: Improving Generalization of ML-NIDS with Attack Knowledge Graphs 

**Authors**: Xin Fan Guo, Albert Merono Penuela, Sergio Maffeis, Fabio Pierazzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19802)  

**Abstract**: Despite extensive research on Machine Learning-based Network Intrusion Detection Systems (ML-NIDS), their capability to detect diverse attack variants remains uncertain. Prior studies have largely relied on homogeneous datasets, which artificially inflate performance scores and offer a false sense of security. Designing systems that can effectively detect a wide range of attack variants remains a significant challenge. The progress of ML-NIDS continues to depend heavily on human expertise, which can embed subjective judgments of system designers into the model, potentially hindering its ability to generalize across diverse attack types.
To address this gap, we propose KnowML, a framework for knowledge-guided machine learning that integrates attack knowledge into ML-NIDS. KnowML systematically explores the threat landscape by leveraging Large Language Models (LLMs) to perform automated analysis of attack implementations. It constructs a unified Knowledge Graph (KG) of attack strategies, on which it applies symbolic reasoning to generate KG-Augmented Input, embedding domain knowledge directly into the design process of ML-NIDS.
We evaluate KnowML on 28 realistic attack variants, of which 10 are newly collected for this study. Our findings reveal that baseline ML-NIDS models fail to detect several variants entirely, achieving F1 scores as low as 0 %. In contrast, our knowledge-guided approach achieves up to 99 % F1 score while maintaining a False Positive Rate below 0.1 %. 

---
# Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study 

**Authors**: Yuqi Zhu, Yi Zhong, Jintian Zhang, Ziheng Zhang, Shuofei Qiao, Yujie Luo, Lun Du, Da Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19794)  

**Abstract**: Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. 

---
# Higher-Order Graph Databases 

**Authors**: Maciej Besta, Shriram Chandran, Jakub Cudak, Patrick Iff, Marcin Copik, Robert Gerstenberger, Tomasz Szydlo, Jürgen Müller, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2506.19661)  

**Abstract**: Recent advances in graph databases (GDBs) have been driving interest in large-scale analytics, yet current systems fail to support higher-order (HO) interactions beyond first-order (one-hop) relations, which are crucial for tasks such as subgraph counting, polyadic modeling, and HO graph learning. We address this by introducing a new class of systems, higher-order graph databases (HO-GDBs) that use lifting and lowering paradigms to seamlessly extend traditional GDBs with HO. We provide a theoretical analysis of OLTP and OLAP queries, ensuring correctness, scalability, and ACID compliance. We implement a lightweight, modular, and parallelizable HO-GDB prototype that offers native support for hypergraphs, node-tuples, subgraphs, and other HO structures under a unified API. The prototype scales to large HO OLTP & OLAP workloads and shows how HO improves analytical tasks, for example enhancing accuracy of graph neural networks within a GDB by 44%. Our work ensures low latency and high query throughput, and generalizes both ACID-compliant and eventually consistent systems. 

---
# Health Sentinel: An AI Pipeline For Real-time Disease Outbreak Detection 

**Authors**: Devesh Pant, Rishi Raj Grandhe, Vipin Samaria, Mukul Paul, Sudhir Kumar, Saransh Khanna, Jatin Agrawal, Jushaan Singh Kalra, Akhil VSSG, Satish V Khalikar, Vipin Garg, Himanshu Chauhan, Pranay Verma, Neha Khandelwal, Soma S Dhavala, Minesh Mathew  

**Link**: [PDF](https://arxiv.org/pdf/2506.19548)  

**Abstract**: Early detection of disease outbreaks is crucial to ensure timely intervention by the health authorities. Due to the challenges associated with traditional indicator-based surveillance, monitoring informal sources such as online media has become increasingly popular. However, owing to the number of online articles getting published everyday, manual screening of the articles is impractical. To address this, we propose Health Sentinel. It is a multi-stage information extraction pipeline that uses a combination of ML and non-ML methods to extract events-structured information concerning disease outbreaks or other unusual health events-from online articles. The extracted events are made available to the Media Scanning and Verification Cell (MSVC) at the National Centre for Disease Control (NCDC), Delhi for analysis, interpretation and further dissemination to local agencies for timely intervention. From April 2022 till date, Health Sentinel has processed over 300 million news articles and identified over 95,000 unique health events across India of which over 3,500 events were shortlisted by the public health experts at NCDC as potential outbreaks. 

---
