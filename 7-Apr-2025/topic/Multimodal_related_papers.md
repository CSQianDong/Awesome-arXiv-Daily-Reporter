# Stance-Driven Multimodal Controlled Statement Generation: New Dataset and Task 

**Authors**: Bingqian Wang, Quan Fang, Jiachen Sun, Xiaoxiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.03295)  

**Abstract**: Formulating statements that support diverse or controversial stances on specific topics is vital for platforms that enable user expression, reshape political discourse, and drive social critique and information dissemination. With the rise of Large Language Models (LLMs), controllable text generation towards specific stances has become a promising research area with applications in shaping public opinion and commercial marketing. However, current datasets often focus solely on pure texts, lacking multimodal content and effective context, particularly in the context of stance detection. In this paper, we formally define and study the new problem of stance-driven controllable content generation for tweets with text and images, where given a multimodal post (text and image/video), a model generates a stance-controlled response. To this end, we create the Multimodal Stance Generation Dataset (StanceGen2024), the first resource explicitly designed for multimodal stance-controllable text generation in political discourse. It includes posts and user comments from the 2024 U.S. presidential election, featuring text, images, videos, and stance annotations to explore how multimodal political content shapes stance expression. Furthermore, we propose a Stance-Driven Multimodal Generation (SDMG) framework that integrates weighted fusion of multimodal features and stance guidance to improve semantic consistency and stance control. We release the dataset and code (this https URL) for public use and further research. 

---
# Explain with Visual Keypoints Like a Real Mentor! A Benchmark for Multimodal Solution Explanation 

**Authors**: Jaewoo Park, Jungyang Park, Dongju Jang, Jiwan Chung, Byungwoo Yoo, Jaewoo Shin, Seonjoon Park, Taehyeong Kim, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03197)  

**Abstract**: With the rapid advancement of mathematical reasoning capabilities in large language models (LLMs), AI systems are increasingly being adopted in educational settings to support students' comprehension of problem-solving processes. However, a critical component remains underexplored in current LLM-generated explanations: visual explanation. In real-world instructional contexts, human tutors routinely employ visual aids-such as diagrams, markings, and highlights-to enhance conceptual clarity. To bridge this gap, we introduce a novel task of visual solution explanation, which requires not only solving problems but also generating explanations that incorporate newly introduced visual elements essential for understanding (e.g., auxiliary lines, annotations, or geometric constructions). To evaluate model performance on this task, we propose MathExplain, a multimodal benchmark consisting of 997 math problems annotated with visual keypoints and corresponding explanatory text that references those elements. Our empirical results show that while some closed-source models demonstrate promising capabilities on visual solution-explaining, current open-source general-purpose models perform inconsistently, particularly in identifying relevant visual components and producing coherent keypoint-based explanations. We expect that visual solution-explaining and the MathExplain dataset will catalyze further research on multimodal LLMs in education and advance their deployment as effective, explanation-oriented AI tutors. Code and data will be released publicly. 

---
# Why Reasoning Matters? A Survey of Advancements in Multimodal Reasoning (v1) 

**Authors**: Jing Bi, Susan Liang, Xiaofei Zhou, Pinxin Liu, Junjia Guo, Yunlong Tang, Luchuan Song, Chao Huang, Guangyu Sun, Jinxi He, Jiarui Wu, Shu Yang, Daoan Zhang, Chen Chen, Lianggong Bruce Wen, Zhang Liu, Jiebo Luo, Chenliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03151)  

**Abstract**: Reasoning is central to human intelligence, enabling structured problem-solving across diverse tasks. Recent advances in large language models (LLMs) have greatly enhanced their reasoning abilities in arithmetic, commonsense, and symbolic domains. However, effectively extending these capabilities into multimodal contexts-where models must integrate both visual and textual inputs-continues to be a significant challenge. Multimodal reasoning introduces complexities, such as handling conflicting information across modalities, which require models to adopt advanced interpretative strategies. Addressing these challenges involves not only sophisticated algorithms but also robust methodologies for evaluating reasoning accuracy and coherence. This paper offers a concise yet insightful overview of reasoning techniques in both textual and multimodal LLMs. Through a thorough and up-to-date comparison, we clearly formulate core reasoning challenges and opportunities, highlighting practical methods for post-training optimization and test-time inference. Our work provides valuable insights and guidance, bridging theoretical frameworks and practical implementations, and sets clear directions for future research. 

---
# Hummus: A Dataset of Humorous Multimodal Metaphor Use 

**Authors**: Xiaoyu Tong, Zhi Zhang, Martha Lewis, Ekaterina Shutova  

**Link**: [PDF](https://arxiv.org/pdf/2504.02983)  

**Abstract**: Metaphor and humor share a lot of common ground, and metaphor is one of the most common humorous mechanisms. This study focuses on the humorous capacity of multimodal metaphors, which has not received due attention in the community. We take inspiration from the Incongruity Theory of humor, the Conceptual Metaphor Theory, and the annotation scheme behind the VU Amsterdam Metaphor Corpus, and developed a novel annotation scheme for humorous multimodal metaphor use in image-caption pairs. We create the Hummus Dataset of Humorous Multimodal Metaphor Use, providing expert annotation on 1k image-caption pairs sampled from the New Yorker Caption Contest corpus. Using the dataset, we test state-of-the-art multimodal large language models (MLLMs) on their ability to detect and understand humorous multimodal metaphor use. Our experiments show that current MLLMs still struggle with processing humorous multimodal metaphors, particularly with regard to integrating visual and textual information. We release our dataset and code at this http URL. 

---
# Enhancing Chart-to-Code Generation in Multimodal Large Language Models via Iterative Dual Preference Learning 

**Authors**: Zhihan Zhang, Yixin Cao, Lizi Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.02906)  

**Abstract**: Chart-to-code generation, the process of converting chart images into executable plotting scripts, provides a lossless representation of chart information, requiring models to accurately capture and summarize all visual and structural elements. However, this remains a significant challenge for multimodal large language models (MLLMs), which are not inherently well-aligned with code generation tasks. To bridge this gap, we introduce Chart2Code, a novel iterative dual preference learning framework designed to enhance MLLMs' chart-to-code generation capabilities through structured code variant generation and fine-grained dual reward signals. We validate Chart2Code across three MLLMs and find that iterative preference learning consistently improves out-of-distribution chart-to-code generation quality. Throughout this process, our dual scoring method, which evaluates both the textual code structure and its visual representation, leads to greater performance improvements, even with a reduced preference dataset size. Further analysis explores the key components of our framework and highlights the interplay between chart-to-code generation and broader chart reasoning, paving the way for future advancements in chart comprehension. 

---
# Towards deployment-centric multimodal AI beyond vision and language 

**Authors**: Xianyuan Liu, Jiayang Zhang, Shuo Zhou, Thijs L. van der Plas, Avish Vijayaraghavan, Anastasiia Grishina, Mengdie Zhuang, Daniel Schofield, Christopher Tomlinson, Yuhan Wang, Ruizhe Li, Louisa van Zeeland, Sina Tabakhi, Cyndie Demeocq, Xiang Li, Arunav Das, Orlando Timmerman, Thomas Baldwin-McDonald, Jinge Wu, Peizhen Bai, Zahraa Al Sahili, Omnia Alwazzan, Thao N. Do, Mohammod N.I. Suvon, Angeline Wang, Lucia Cipolina-Kun, Luigi A. Moretti, Lucas Farndale, Nitisha Jain, Natalia Efremova, Yan Ge, Marta Varela, Hak-Keung Lam, Oya Celiktutan, Ben R. Evans, Alejandro Coca-Castro, Honghan Wu, Zahraa S. Abdallah, Chen Chen, Valentin Danchev, Nataliya Tkachenko, Lei Lu, Tingting Zhu, Gregory G. Slabaugh, Roger K. Moore, William K. Cheung, Peter H. Charlton, Haiping Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.03603)  

**Abstract**: Multimodal artificial intelligence (AI) integrates diverse types of data via machine learning to improve understanding, prediction, and decision-making across disciplines such as healthcare, science, and engineering. However, most multimodal AI advances focus on models for vision and language data, while their deployability remains a key challenge. We advocate a deployment-centric workflow that incorporates deployment constraints early to reduce the likelihood of undeployable solutions, complementing data-centric and model-centric approaches. We also emphasise deeper integration across multiple levels of multimodality and multidisciplinary collaboration to significantly broaden the research scope beyond vision and language. To facilitate this approach, we identify common multimodal-AI-specific challenges shared across disciplines and examine three real-world use cases: pandemic response, self-driving car design, and climate change adaptation, drawing expertise from healthcare, social science, engineering, science, sustainability, and finance. By fostering multidisciplinary dialogue and open research practices, our community can accelerate deployment-centric development for broad societal impact. 

---
# A Human Digital Twin Architecture for Knowledge-based Interactions and Context-Aware Conversations 

**Authors**: Abdul Mannan Mohammed, Azhar Ali Mohammad, Jason A. Ortiz, Carsten Neumann, Grace Bochenek, Dirk Reiners, Carolina Cruz-Neira  

**Link**: [PDF](https://arxiv.org/pdf/2504.03147)  

**Abstract**: Recent developments in Artificial Intelligence (AI) and Machine Learning (ML) are creating new opportunities for Human-Autonomy Teaming (HAT) in tasks, missions, and continuous coordinated activities. A major challenge is enabling humans to maintain awareness and control over autonomous assets, while also building trust and supporting shared contextual understanding. To address this, we present a real-time Human Digital Twin (HDT) architecture that integrates Large Language Models (LLMs) for knowledge reporting, answering, and recommendation, embodied in a visual interface.
The system applies a metacognitive approach to enable personalized, context-aware responses aligned with the human teammate's expectations. The HDT acts as a visually and behaviorally realistic team member, integrated throughout the mission lifecycle, from training to deployment to after-action review. Our architecture includes speech recognition, context processing, AI-driven dialogue, emotion modeling, lip-syncing, and multimodal feedback. We describe the system design, performance metrics, and future development directions for more adaptive and realistic HAT systems. 

---
# VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning 

**Authors**: Xianwei Zhuang, Yuxin Xie, Yufan Deng, Dongchao Yang, Liming Liang, Jinghan Ru, Yuguo Yin, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.02949)  

**Abstract**: In this work, we present VARGPT-v1.1, an advanced unified visual autoregressive model that builds upon our previous framework VARGPT. The model preserves the dual paradigm of next-token prediction for visual understanding and next-scale generation for image synthesis. Specifically, VARGPT-v1.1 integrates: (1) a novel training strategy combining iterative visual instruction tuning with reinforcement learning through Direct Preference Optimization (DPO), (2) an expanded training corpus containing 8.3M visual-generative instruction pairs, (3) an upgraded language model backbone using Qwen2, (4) enhanced image generation resolution, and (5) emergent image editing capabilities without architectural modifications. These advancements enable VARGPT-v1.1 to achieve state-of-the-art performance in multimodal understanding and text-to-image instruction-following tasks, demonstrating significant improvements in both comprehension and generation metrics. Notably, through visual instruction tuning, the model acquires image editing functionality while maintaining architectural consistency with its predecessor, revealing the potential for unified visual understanding, generation, and editing. Our findings suggest that well-designed unified visual autoregressive models can effectively adopt flexible training strategies from large language models (LLMs), exhibiting promising scalability. The codebase and model weights are publicly available at this https URL. 

---
