# Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations 

**Authors**: Liang Luo, Yuxin Chen, Zhengyu Zhang, Mengyue Hang, Andrew Gu, Buyun Zhang, Boyang Liu, Chen Chen, Chengze Fan, Dong Liang, Fan Yang, Feifan Gu, Huayu Li, Jade Nie, Jiayi Xu, Jiyan Yang, Jongsoo Park, Laming Chen, Longhao Jin, Qianru Li, Qin Huang, Shali Jiang, Shiwen Shen, Shuaiwen Wang, Sihan Zeng, Siyang Yuan, Tongyi Tang, Weilin Zhang, Wenjun Wang, Xi Liu, Xiaohan Wei, Xiaozhen Xia, Yuchen Hao, Yunlong He, Yasmine Badr, Zeliang Chen, Maxim Naumov, Yantao Yao, Wenlin Chen, Santanu Kolay, GP Musumeci, Ellie Dingqiao Wen  

**Link**: [PDF](https://arxiv.org/pdf/2512.09200)  

**Abstract**: The rapidly evolving landscape of products, surfaces, policies, and regulations poses significant challenges for deploying state-of-the-art recommendation models at industry scale, primarily due to data fragmentation across domains and escalating infrastructure costs that hinder sustained quality improvements.
To address this challenge, we propose Lattice, a recommendation framework centered around model space redesign that extends Multi-Domain, Multi-Objective (MDMO) learning beyond models and learning objectives. Lattice addresses these challenges through a comprehensive model space redesign that combines cross-domain knowledge sharing, data consolidation, model unification, distillation, and system optimizations to achieve significant improvements in both quality and cost-efficiency.
Our deployment of Lattice at Meta has resulted in 10% revenue-driving top-line metrics gain, 11.5% user satisfaction improvement, 6% boost in conversion rate, with 20% capacity saving. 

---
# RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning 

**Authors**: Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.09487)  

**Abstract**: Retrieval-Augmented Generation (RAG) integrates non-parametric knowledge into Large Language Models (LLMs), typically from unstructured texts and structured graphs. While recent progress has advanced text-based RAG to multi-turn reasoning through Reinforcement Learning (RL), extending these advances to hybrid retrieval introduces additional challenges. Existing graph-based or hybrid systems typically depend on fixed or handcrafted retrieval pipelines, lacking the ability to integrate supplementary evidence as reasoning unfolds. Besides, while graph evidence provides relational structures crucial for multi-hop reasoning, it is substantially more expensive to retrieve. To address these limitations, we introduce \model{}, an RL-based framework that enables LLMs to perform multi-turn and adaptive graph-text hybrid RAG. \model{} jointly optimizes the entire generation process via RL, allowing the model to learn when to reason, what to retrieve from either texts or graphs, and when to produce final answers, all within a unified generation policy. To guide this learning process, we design a two-stage training framework that accounts for both task outcome and retrieval efficiency, enabling the model to exploit hybrid evidence while avoiding unnecessary retrieval overhead. Experimental results across five question answering benchmarks demonstrate that \model{} significantly outperforms existing RAG baselines, highlighting the benefits of end-to-end RL in supporting adaptive and efficient retrieval for complex reasoning. 

---
# Passing the Baton: High Throughput Distributed Disk-Based Vector Search with BatANN 

**Authors**: Nam Anh Dang, Ben Landrum, Ken Birman  

**Link**: [PDF](https://arxiv.org/pdf/2512.09331)  

**Abstract**: Vector search underpins modern information-retrieval systems, including retrieval-augmented generation (RAG) pipelines and search engines over unstructured text and images. As datasets scale to billions of vectors, disk-based vector search has emerged as a practical solution. However, looking to the future, we need to anticipate datasets too large for any single server. We present BatANN, a distributed disk-based approximate nearest neighbor (ANN) system that retains the logarithmic search efficiency of a single global graph while achieving near-linear throughput scaling in the number of servers. Our core innovation is that when accessing a neighborhood which is stored on another machine, we send the full state of the query to the other machine to continue executing there for improved locality. On 100M- and 1B-point datasets at 0.95 recall using 10 servers, BatANN achieves 6.21-6.49x and 2.5-5.10x the throughput of the scatter-gather baseline, respectively, while maintaining mean latency below 6 ms. Moreover, we get these results on standard TCP. To our knowledge, BatANN is the first open-source distributed disk-based vector search system to operate over a single global graph. 

---
# Goal inference with Rao-Blackwellized Particle Filters 

**Authors**: Yixuan Wang, Dan P. Guralnik, Warren E. Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2512.09269)  

**Abstract**: Inferring the eventual goal of a mobile agent from noisy observations of its trajectory is a fundamental estimation problem. We initiate the study of such intent inference using a variant of a Rao-Blackwellized Particle Filter (RBPF), subject to the assumption that the agent's intent manifests through closed-loop behavior with a state-of-the-art provable practical stability property. Leveraging the assumed closed-form agent dynamics, the RBPF analytically marginalizes the linear-Gaussian substructure and updates particle weights only, improving sample efficiency over a standard particle filter. Two difference estimators are introduced: a Gaussian mixture model using the RBPF weights and a reduced version confining the mixture to the effective sample. We quantify how well the adversary can recover the agent's intent using information-theoretic leakage metrics and provide computable lower bounds on the Kullback-Leibler (KL) divergence between the true intent distribution and RBPF estimates via Gaussian-mixture KL bounds. We also provide a bound on the difference in performance between the two estimators, highlighting the fact that the reduced estimator performs almost as well as the complete one. Experiments illustrate fast and accurate intent recovery for compliant agents, motivating future work on designing intent-obfuscating controllers. 

---
# PoultryTalk: A Multi-modal Retrieval-Augmented Generation (RAG) System for Intelligent Poultry Management and Decision Support 

**Authors**: Kapalik Khanal, Biswash Khatiwada, Stephen Afrifa, Ranjan Sapkota, Sanjay Shah, Frank Bai, Ramesh Bahadur Bist  

**Link**: [PDF](https://arxiv.org/pdf/2512.08995)  

**Abstract**: The Poultry industry plays a vital role in global food security, yet small- and medium-scale farmers frequently lack timely access to expert-level support for disease diagnosis, nutrition planning, and management decisions. With rising climate stress, unpredictable feed prices, and persistent disease threats, poultry producers often struggle to make quick, informed decisions. Therefore, there is a critical need for intelligent, data-driven systems that can deliver reliable, on-demand consultation. This paper presents PoultryTalk, a novel multi-modal Retrieval-Augmented Generation (RAG) system designed to provide real-time expert guidance through text and image-based interaction. PoultryTalk uses OpenAI's text-embedding-3-small and GPT-4o to provide smart, context-aware poultry management advice from text, images, or questions. System usability and performance were evaluated using 200 expert-verified queries and feedback from 34 participants who submitted 267 queries to the PoultryTalk prototype. The expert-verified benchmark queries confirmed strong technical performance, achieving a semantic similarity of 84.0% and an average response latency of 3.6 seconds. Compared with OpenAI's GPT-4o, PoultryTalk delivered more accurate and reliable information related to poultry. Based on participants' evaluations, PoultryTalk achieved a response accuracy of 89.9%, with about 9.1% of responses rated as incorrect. A post-use survey indicated high user satisfaction: 95.6% of participants reported that the chatbot provided "always correct" and "mostly correct" answers. 82.6% indicated they would recommend the tool, and 17.4% responded "maybe." These results collectively demonstrate that PoultryTalk not only delivers accurate, contextually relevant information but also demonstrates strong user acceptance and scalability potential. 

---
