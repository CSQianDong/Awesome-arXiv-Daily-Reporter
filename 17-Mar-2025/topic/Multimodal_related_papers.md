# BriLLM: Brain-inspired Large Language Model 

**Authors**: Hai Zhao, Hongqiu Wu, Dongjie Yang, Anni Zou, Jiale Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11299)  

**Abstract**: This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above. 

---
# Cross-Modal Learning for Music-to-Music-Video Description Generation 

**Authors**: Zhuoyuan Mao, Mengjie Zhao, Qiyu Wu, Zhi Zhong, Wei-Hsiang Liao, Hiromi Wakaki, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2503.11190)  

**Abstract**: Music-to-music-video generation is a challenging task due to the intrinsic differences between the music and video modalities. The advent of powerful text-to-video diffusion models has opened a promising pathway for music-video (MV) generation by first addressing the music-to-MV description task and subsequently leveraging these models for video generation. In this study, we focus on the MV description generation task and propose a comprehensive pipeline encompassing training data construction and multimodal model fine-tuning. We fine-tune existing pre-trained multimodal models on our newly constructed music-to-MV description dataset based on the Music4All dataset, which integrates both musical and visual information. Our experimental results demonstrate that music representations can be effectively mapped to textual domains, enabling the generation of meaningful MV description directly from music inputs. We also identify key components in the dataset construction pipeline that critically impact the quality of MV description and highlight specific musical attributes that warrant greater focus for improved MV description generation. 

---
# Exploring the Potential of Large Multimodal Models as Effective Alternatives for Pronunciation Assessment 

**Authors**: Ke Wang, Lei He, Kun Liu, Yan Deng, Wenning Wei, Sheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11229)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated exceptional performance across a wide range of domains. This paper explores their potential in pronunciation assessment tasks, with a particular focus on evaluating the capabilities of the Generative Pre-trained Transformer (GPT) model, specifically GPT-4o. Our study investigates its ability to process speech and audio for pronunciation assessment across multiple levels of granularity and dimensions, with an emphasis on feedback generation and scoring. For our experiments, we use the publicly available Speechocean762 dataset. The evaluation focuses on two key aspects: multi-level scoring and the practicality of the generated feedback. Scoring results are compared against the manual scores provided in the Speechocean762 dataset, while feedback quality is assessed using Large Language Models (LLMs). The findings highlight the effectiveness of integrating LMMs with traditional methods for pronunciation assessment, offering insights into the model's strengths and identifying areas for further improvement. 

---
# Exploring Typographic Visual Prompts Injection Threats in Cross-Modality Generation Models 

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Kaidi Xu, Mengshu Sun, Jindong Gu, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11519)  

**Abstract**: Current Cross-Modality Generation Models (GMs) demonstrate remarkable capabilities in various generative tasks. Given the ubiquity and information richness of vision modality inputs in real-world scenarios, Cross-vision, encompassing Vision-Language Perception (VLP) and Image-to-Image (I2I), tasks have attracted significant attention. Large Vision Language Models (LVLMs) and I2I GMs are employed to handle VLP and I2I tasks, respectively. Previous research indicates that printing typographic words into input images significantly induces LVLMs and I2I GMs to generate disruptive outputs semantically related to those words. Additionally, visual prompts, as a more sophisticated form of typography, are also revealed to pose security risks to various applications of VLP tasks when injected into images. In this paper, we comprehensively investigate the performance impact induced by Typographic Visual Prompt Injection (TVPI) in various LVLMs and I2I GMs. To better observe performance modifications and characteristics of this threat, we also introduce the TVPI Dataset. Through extensive explorations, we deepen the understanding of the underlying causes of the TVPI threat in various GMs and offer valuable insights into its potential origins. 

---
# Towards Understanding Graphical Perception in Large Multimodal Models 

**Authors**: Kai Zhang, Jianwei Yang, Jeevana Priya Inala, Chandan Singh, Jianfeng Gao, Yu Su, Chenglong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10857)  

**Abstract**: Despite the promising results of large multimodal models (LMMs) in complex vision-language tasks that require knowledge, reasoning, and perception abilities together, we surprisingly found that these models struggle with simple tasks on infographics that require perception only. As existing benchmarks primarily focus on end tasks that require various abilities, they provide limited, fine-grained insights into the limitations of the models' perception abilities. To address this gap, we leverage the theory of graphical perception, an approach used to study how humans decode visual information encoded on charts and graphs, to develop an evaluation framework for analyzing gaps in LMMs' perception abilities in charts. With automated task generation and response evaluation designs, our framework enables comprehensive and controlled testing of LMMs' graphical perception across diverse chart types, visual elements, and task types. We apply our framework to evaluate and diagnose the perception capabilities of state-of-the-art LMMs at three granularity levels (chart, visual element, and pixel). Our findings underscore several critical limitations of current state-of-the-art LMMs, including GPT-4o: their inability to (1) generalize across chart types, (2) understand fundamental visual elements, and (3) cross reference values within a chart. These insights provide guidance for future improvements in perception abilities of LMMs. The evaluation framework and labeled data are publicly available at this https URL. 

---
# Keyframe-oriented Vision Token Pruning: Enhancing Efficiency of Large Vision Language Models on Long-Form Video Processing 

**Authors**: Yudong Liu, Jingwei Sun, Yueqian Lin, Jingyang Zhang, Ming Yin, Qinsi Wang, Jianyi Zhang, Hai Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10742)  

**Abstract**: Vision language models (VLMs) demonstrate strong capabilities in jointly processing visual and textual data. However, they often incur substantial computational overhead due to redundant visual information, particularly in long-form video scenarios. Existing approaches predominantly focus on either vision token pruning, which may overlook spatio-temporal dependencies, or keyframe selection, which identifies informative frames but discards others, thus disrupting contextual continuity. In this work, we propose KVTP (Keyframe-oriented Vision Token Pruning), a novel framework that overcomes the drawbacks of token pruning and keyframe selection. By adaptively assigning pruning rates based on frame relevance to the query, KVTP effectively retains essential contextual information while significantly reducing redundant computation. To thoroughly evaluate the long-form video understanding capacities of VLMs, we curated and reorganized subsets from VideoMME, EgoSchema, and NextQA into a unified benchmark named SparseKV-QA that highlights real-world scenarios with sparse but crucial events. Our experiments with VLMs of various scales show that KVTP can reduce token usage by 80% without compromising spatiotemporal and contextual consistency, significantly cutting computation while maintaining the performance. These results demonstrate our approach's effectiveness in efficient long-video processing, facilitating more scalable VLM deployment. 

---
# Chat-TS: Enhancing Multi-Modal Reasoning Over Time-Series and Natural Language Data 

**Authors**: Paul Quinlan, Qingguo Li, Xiaodan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10883)  

**Abstract**: Time-series analysis is critical for a wide range of fields such as healthcare, finance, transportation, and energy, among many others. The practical applications often involve analyzing time-series data alongside contextual information in the form of natural language to support informed decisions. However, current time-series models are limited in their ability to perform reasoning that involves both time-series and their textual content. In this work, we address this gap by introducing \textit{Chat-TS}, a large language model (LLM) based framework, designed to support reasoning over time series and textual data. Unlike traditional models, Chat-TS integrates time-series tokens into LLMs' vocabulary, enhancing its reasoning ability over both modalities without compromising the core natural language capabilities, enabling practical analysis and reasoning across modalities. To support learning and evaluation in this setup, we contribute new datasets: the \textit{TS Instruct Training Dataset} which pairs diverse time-series data with relevant text instructions and responses for instruction tuning, the \textit{TS Instruct Question and Answer (QA) Gold Dataset} which provides multiple-choice questions designed to evaluate multimodal reasoning, and a \textit{TS Instruct Quantitative Probing Set} which contains a small subset of the TS Instruct QA tasks alongside math and decision-making questions for LLM evaluation. We designed a training strategy to preserve the inherent reasoning capabilities of LLMs while augmenting them for time-series reasoning. Experiments show that Chat-TS achieves state-of-the-art performance in multi-modal reasoning tasks by maintaining strong natural language proficiency while improving time-series reasoning. ~\footnote{To ensure replicability and facilitate future research, all models, datasets, and code will be available at [\texttt{Github-URL}].} 

---
# Small Vision-Language Models: A Survey on Compact Architectures and Techniques 

**Authors**: Nitesh Patnaik, Navdeep Nayak, Himani Bansal Agrawal, Moinak Chinmoy Khamaru, Gourav Bal, Saishree Smaranika Panda, Rishi Raj, Vishal Meena, Kartheek Vadlamani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10665)  

**Abstract**: The emergence of small vision-language models (sVLMs) marks a critical advancement in multimodal AI, enabling efficient processing of visual and textual data in resource-constrained environments. This survey offers a comprehensive exploration of sVLM development, presenting a taxonomy of architectures - transformer-based, mamba-based, and hybrid - that highlight innovations in compact design and computational efficiency. Techniques such as knowledge distillation, lightweight attention mechanisms, and modality pre-fusion are discussed as enablers of high performance with reduced resource requirements. Through an in-depth analysis of models like TinyGPT-V, MiniGPT-4, and VL-Mamba, we identify trade-offs between accuracy, efficiency, and scalability. Persistent challenges, including data biases and generalization to complex tasks, are critically examined, with proposed pathways for addressing them. By consolidating advancements in sVLMs, this work underscores their transformative potential for accessible AI, setting a foundation for future research into efficient multimodal systems. 

---
# Learning to Inference Adaptively for Multimodal Large Language Models 

**Authors**: Zhuoyan Xu, Khoi Duc Nguyen, Preeti Mukherjee, Saurabh Bagchi, Somali Chaterji, Yingyu Liang, Yin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10905)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive capabilities in reasoning, yet come with substantial computational cost, limiting their deployment in resource-constrained settings. Despite recent efforts on improving the efficiency of MLLMs, prior solutions fall short in responding to varying runtime conditions, in particular changing resource availability (e.g., contention due to the execution of other programs on the device). To bridge this gap, we introduce AdaLLaVA, an adaptive inference framework that learns to dynamically reconfigure operations in an MLLM during inference, accounting for the input data and a latency budget. We conduct extensive experiments across benchmarks involving question-answering, reasoning, and hallucination. Our results show that AdaLLaVA effectively adheres to input latency budget, achieving varying accuracy and latency tradeoffs at runtime. Further, we demonstrate that AdaLLaVA adapts to both input latency and content, can be integrated with token selection for enhanced efficiency, and generalizes across this http URL project webpage with code release is at this https URL. 

---
# PARIC: Probabilistic Attention Regularization for Language Guided Image Classification from Pre-trained Vison Language Models 

**Authors**: Mayank Nautiyal, Stela Arranz Gheorghe, Kristiana Stefa, Li Ju, Ida-Maria Sintorn, Prashant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2503.11360)  

**Abstract**: Language-guided attention frameworks have significantly enhanced both interpretability and performance in image classification; however, the reliance on deterministic embeddings from pre-trained vision-language foundation models to generate reference attention maps frequently overlooks the intrinsic multivaluedness and ill-posed characteristics of cross-modal mappings. To address these limitations, we introduce PARIC, a probabilistic framework for guiding visual attention via language specifications. Our approach enables pre-trained vision-language models to generate probabilistic reference attention maps, which align textual and visual modalities more effectively while incorporating uncertainty estimates, as compared to their deterministic counterparts. Experiments on benchmark test problems demonstrate that PARIC enhances prediction accuracy, mitigates bias, ensures consistent predictions, and improves robustness across various datasets. 

---
# Compound Expression Recognition via Large Vision-Language Models 

**Authors**: Jun Yu, Xilong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11241)  

**Abstract**: Compound Expression Recognition (CER) is crucial for understanding human emotions and improving human-computer interaction. However, CER faces challenges due to the complexity of facial expressions and the difficulty of capturing subtle emotional cues. To address these issues, we propose a novel approach leveraging Large Vision-Language Models (LVLMs). Our method employs a two-stage fine-tuning process: first, pre-trained LVLMs are fine-tuned on basic facial expressions to establish foundational patterns; second, the model is further optimized on a compound-expression dataset to refine visual-language feature interactions. Our approach achieves advanced accuracy on the RAF-DB dataset and demonstrates strong zero-shot generalization on the C-EXPR-DB dataset, showcasing its potential for real-world applications in emotion analysis and human-computer interaction. 

---
# Augmenting Image Annotation: A Human-LMM Collaborative Framework for Efficient Object Selection and Label Generation 

**Authors**: He Zhang, Xinyi Fu, John M. Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2503.11096)  

**Abstract**: Traditional image annotation tasks rely heavily on human effort for object selection and label assignment, making the process time-consuming and prone to decreased efficiency as annotators experience fatigue after extensive work. This paper introduces a novel framework that leverages the visual understanding capabilities of large multimodal models (LMMs), particularly GPT, to assist annotation workflows. In our proposed approach, human annotators focus on selecting objects via bounding boxes, while the LMM autonomously generates relevant labels. This human-AI collaborative framework enhances annotation efficiency by reducing the cognitive and time burden on human annotators. By analyzing the system's performance across various types of annotation tasks, we demonstrate its ability to generalize to tasks such as object recognition, scene description, and fine-grained categorization. Our proposed framework highlights the potential of this approach to redefine annotation workflows, offering a scalable and efficient solution for large-scale data labeling in computer vision. Finally, we discuss how integrating LMMs into the annotation pipeline can advance bidirectional human-AI alignment, as well as the challenges of alleviating the "endless annotation" burden in the face of information overload by shifting some of the work to AI. 

---
# Observation-Graph Interaction and Key-Detail Guidance for Vision and Language Navigation 

**Authors**: Yifan Xie, Binkai Ou, Fei Ma, Yaohua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11006)  

**Abstract**: Vision and Language Navigation (VLN) requires an agent to navigate through environments following natural language instructions. However, existing methods often struggle with effectively integrating visual observations and instruction details during navigation, leading to suboptimal path planning and limited success rates. In this paper, we propose OIKG (Observation-graph Interaction and Key-detail Guidance), a novel framework that addresses these limitations through two key components: (1) an observation-graph interaction module that decouples angular and visual information while strengthening edge representations in the navigation space, and (2) a key-detail guidance module that dynamically extracts and utilizes fine-grained location and object information from instructions. By enabling more precise cross-modal alignment and dynamic instruction interpretation, our approach significantly improves the agent's ability to follow complex navigation instructions. Extensive experiments on the R2R and RxR datasets demonstrate that OIKG achieves state-of-the-art performance across multiple evaluation metrics, validating the effectiveness of our method in enhancing navigation precision through better observation-instruction alignment. 

---
# ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models 

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch  

**Link**: [PDF](https://arxiv.org/pdf/2503.10937)  

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications. 

---
# Unifying 2D and 3D Vision-Language Understanding 

**Authors**: Ayush Jain, Alexander Swerdlow, Yuzhou Wang, Sergio Arnaud, Ada Martin, Alexander Sax, Franziska Meier, Katerina Fragkiadaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.10745)  

**Abstract**: Progress in 3D vision-language learning has been hindered by the scarcity of large-scale 3D datasets. We introduce UniVLG, a unified architecture for 2D and 3D vision-language understanding that bridges the gap between existing 2D-centric models and the rich 3D sensory data available in embodied systems. Our approach initializes most model weights from pre-trained 2D models and trains on both 2D and 3D vision-language data. We propose a novel language-conditioned mask decoder shared across 2D and 3D modalities to ground objects effectively in both RGB and RGB-D images, outperforming box-based approaches. To further reduce the domain gap between 2D and 3D, we incorporate 2D-to-3D lifting strategies, enabling UniVLG to utilize 2D data to enhance 3D performance. With these innovations, our model achieves state-of-the-art performance across multiple 3D vision-language grounding tasks, demonstrating the potential of transferring advances from 2D vision-language learning to the data-constrained 3D domain. Furthermore, co-training on both 2D and 3D data enhances performance across modalities without sacrificing 2D capabilities. By removing the reliance on 3D mesh reconstruction and ground-truth object proposals, UniVLG sets a new standard for realistic, embodied-aligned evaluation. Code and additional visualizations are available at $\href{this https URL}{this http URL}$. 

---
# IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models 

**Authors**: Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.10110)  

**Abstract**: Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g., brushing a soft pillow) to more dangerous (e.g., toppling a glass vase). Due to this diversity, it is difficult to characterize which contacts may be acceptable or unacceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach uses the VLM's outputs to produce a dense 3D "cost map" that encodes contact tolerances and seamlessly integrates with standard motion planners. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3620 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Supplementary material is available at this https URL. 

---
