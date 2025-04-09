# Scale Up Composed Image Retrieval Learning via Modification Text Generation 

**Authors**: Yinan Zhou, Yaxiong Wang, Haokun Lin, Chen Ma, Li Zhu, Zhedong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05316)  

**Abstract**: Composed Image Retrieval (CIR) aims to search an image of interest using a combination of a reference image and modification text as the query. Despite recent advancements, this task remains challenging due to limited training data and laborious triplet annotation processes. To address this issue, this paper proposes to synthesize the training triplets to augment the training resource for the CIR problem. Specifically, we commence by training a modification text generator exploiting large-scale multimodal models and scale up the CIR learning throughout both the pretraining and fine-tuning stages. During pretraining, we leverage the trained generator to directly create Modification Text-oriented Synthetic Triplets(MTST) conditioned on pairs of images. For fine-tuning, we first synthesize reverse modification text to connect the target image back to the reference image. Subsequently, we devise a two-hop alignment strategy to incrementally close the semantic gap between the multimodal pair and the target image. We initially learn an implicit prototype utilizing both the original triplet and its reversed version in a cycle manner, followed by combining the implicit prototype feature with the modification text to facilitate accurate alignment with the target image. Extensive experiments validate the efficacy of the generated triplets and confirm that our proposed methodology attains competitive recall on both the CIRR and FashionIQ benchmarks. 

---
# Efficient Multi-Task Learning via Generalist Recommender 

**Authors**: Luyang Wang, Cangcheng Tang, Chongyang Zhang, Jun Ruan, Kai Huang, Jason Dai  

**Link**: [PDF](https://arxiv.org/pdf/2504.05318)  

**Abstract**: Multi-task learning (MTL) is a common machine learning technique that allows the model to share information across different tasks and improve the accuracy of recommendations for all of them. Many existing MTL implementations suffer from scalability issues as the training and inference performance can degrade with the increasing number of tasks, which can limit production use case scenarios for MTL-based recommender systems. Inspired by the recent advances of large language models, we developed an end-to-end efficient and scalable Generalist Recommender (GRec). GRec takes comprehensive data signals by utilizing NLP heads, parallel Transformers, as well as a wide and deep structure to process multi-modal inputs. These inputs are then combined and fed through a newly proposed task-sentence level routing mechanism to scale the model capabilities on multiple tasks without compromising performance. Offline evaluations and online experiments show that GRec significantly outperforms our previous recommender solutions. GRec has been successfully deployed on one of the largest telecom websites and apps, effectively managing high volumes of online traffic every day. 

---
# Multimodal Quantitative Language for Generative Recommendation 

**Authors**: Jianyang Zhai, Zi-Feng Mai, Chang-Dong Wang, Feidiao Yang, Xiawu Zheng, Hui Li, Yonghong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.05314)  

**Abstract**: Generative recommendation has emerged as a promising paradigm aiming at directly generating the identifiers of the target candidates. Most existing methods attempt to leverage prior knowledge embedded in Pre-trained Language Models (PLMs) to improve the recommendation performance. However, they often fail to accommodate the differences between the general linguistic knowledge of PLMs and the specific needs of recommendation systems. Moreover, they rarely consider the complementary knowledge between the multimodal information of items, which represents the multi-faceted preferences of users. To facilitate efficient recommendation knowledge transfer, we propose a novel approach called Multimodal Quantitative Language for Generative Recommendation (MQL4GRec). Our key idea is to transform items from different domains and modalities into a unified language, which can serve as a bridge for transferring recommendation knowledge. Specifically, we first introduce quantitative translators to convert the text and image content of items from various domains into a new and concise language, known as quantitative language, with all items sharing the same vocabulary. Then, we design a series of quantitative language generation tasks to enrich quantitative language with semantic information and prior knowledge. Finally, we achieve the transfer of recommendation knowledge from different domains and modalities to the recommendation task through pre-training and fine-tuning. We evaluate the effectiveness of MQL4GRec through extensive experiments and comparisons with existing methods, achieving improvements over the baseline by 11.18\%, 14.82\%, and 7.95\% on the NDCG metric across three different datasets, respectively. 

---
# Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought 

**Authors**: Yi Peng, Chris, Xiaokun Wang, Yichen Wei, Jiangbo Pei, Weijie Qiu, Ai Jian, Yunzhuo Hao, Jiachun Pan, Tianyidan Xie, Li Ge, Rongxian Zhuang, Xuchen Song, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05599)  

**Abstract**: We introduce Skywork R1V, a multimodal reasoning model extending the an R1-series Large language models (LLM) to visual modalities via an efficient multimodal transfer method. Leveraging a lightweight visual projector, Skywork R1V facilitates seamless multimodal adaptation without necessitating retraining of either the foundational language model or the vision encoder. To strengthen visual-text alignment, we propose a hybrid optimization strategy that combines Iterative Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), significantly enhancing cross-modal integration efficiency. Additionally, we introduce an adaptive-length Chain-of-Thought distillation approach for reasoning data generation. This approach dynamically optimizes reasoning chain lengths, thereby enhancing inference efficiency and preventing excessive reasoning overthinking. Empirical evaluations demonstrate that Skywork R1V, with only 38B parameters, delivers competitive performance, achieving a score of 69.0 on the MMMU benchmark and 67.5 on MathVista. Meanwhile, it maintains robust textual reasoning performance, evidenced by impressive scores of 72.0 on AIME and 94.0 on MATH500. The Skywork R1V model weights have been publicly released to promote openness and reproducibility. 

---
# A Multimedia Analytics Model for the Foundation Model Era 

**Authors**: Marcel Worring, Jan Zahálka, Stef van den Elzen, Maximilian Fischer, Daniel Keim  

**Link**: [PDF](https://arxiv.org/pdf/2504.06138)  

**Abstract**: The rapid advances in Foundation Models and agentic Artificial Intelligence are transforming multimedia analytics by enabling richer, more sophisticated interactions between humans and analytical systems. Existing conceptual models for visual and multimedia analytics, however, do not adequately capture the complexity introduced by these powerful AI paradigms. To bridge this gap, we propose a comprehensive multimedia analytics model specifically designed for the foundation model era. Building upon established frameworks from visual analytics, multimedia analytics, knowledge generation, analytic task definition, mixed-initiative guidance, and human-in-the-loop reinforcement learning, our model emphasizes integrated human-AI teaming based on visual analytics agents from both technical and conceptual perspectives. Central to the model is a seamless, yet explicitly separable, interaction channel between expert users and semi-autonomous analytical processes, ensuring continuous alignment between user intent and AI behavior. The model addresses practical challenges in sensitive domains such as intelligence analysis, investigative journalism, and other fields handling complex, high-stakes data. We illustrate through detailed case studies how our model facilitates deeper understanding and targeted improvement of multimedia analytics solutions. By explicitly capturing how expert users can optimally interact with and guide AI-powered multimedia analytics systems, our conceptual framework sets a clear direction for system design, comparison, and future research. 

---
# MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models 

**Authors**: Pengfei Zhou, Fanrui Zhang, Xiaopeng Peng, Zhaopan Xu, Jiaxin Ai, Yansheng Qiu, Chuanhao Li, Zhen Li, Ming Li, Yukang Feng, Jianwen Sun, Haoquan Zhang, Zizhen Li, Xiaofeng Mao, Wangbo Zhao, Kai Wang, Xiaojun Chang, Wenqi Shao, Yang You, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05782)  

**Abstract**: Multimodal reasoning, which integrates language and visual cues into problem solving and decision making, is a fundamental aspect of human intelligence and a crucial step toward artificial general intelligence. However, the evaluation of multimodal reasoning capabilities in Multimodal Large Language Models (MLLMs) remains inadequate. Most existing reasoning benchmarks are constrained by limited data size, narrow domain coverage, and unstructured knowledge distribution. To close these gaps, we introduce MDK12-Bench, a multi-disciplinary benchmark assessing the reasoning capabilities of MLLMs via real-world K-12 examinations. Spanning six disciplines (math, physics, chemistry, biology, geography, and information science), our benchmark comprises 140K reasoning instances across diverse difficulty levels from primary school to 12th grade. It features 6,827 instance-level knowledge point annotations based on a well-organized knowledge structure, detailed answer explanations, difficulty labels and cross-year partitions, providing a robust platform for comprehensive evaluation. Additionally, we present a novel dynamic evaluation framework to mitigate data contamination issues by bootstrapping question forms, question types, and image styles during evaluation. Extensive experiment on MDK12-Bench reveals the significant limitation of current MLLMs in multimodal reasoning. The findings on our benchmark provide insights into the development of the next-generation models. Our data and codes are available at this https URL. 

---
# Human Activity Recognition using RGB-Event based Sensors: A Multi-modal Heat Conduction Model and A Benchmark Dataset 

**Authors**: Shiao Wang, Xiao Wang, Bo Jiang, Lin Zhu, Guoqi Li, Yaowei Wang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05830)  

**Abstract**: Human Activity Recognition (HAR) primarily relied on traditional RGB cameras to achieve high-performance activity recognition. However, the challenging factors in real-world scenarios, such as insufficient lighting and rapid movements, inevitably degrade the performance of RGB cameras. To address these challenges, biologically inspired event cameras offer a promising solution to overcome the limitations of traditional RGB cameras. In this work, we rethink human activity recognition by combining the RGB and event cameras. The first contribution is the proposed large-scale multi-modal RGB-Event human activity recognition benchmark dataset, termed HARDVS 2.0, which bridges the dataset gaps. It contains 300 categories of everyday real-world actions with a total of 107,646 paired videos covering various challenging scenarios. Inspired by the physics-informed heat conduction model, we propose a novel multi-modal heat conduction operation framework for effective activity recognition, termed MMHCO-HAR. More in detail, given the RGB frames and event streams, we first extract the feature embeddings using a stem network. Then, multi-modal Heat Conduction blocks are designed to fuse the dual features, the key module of which is the multi-modal Heat Conduction Operation layer. We integrate RGB and event embeddings through a multi-modal DCT-IDCT layer while adaptively incorporating the thermal conductivity coefficient via FVEs into this module. After that, we propose an adaptive fusion module based on a policy routing strategy for high-performance classification. Comprehensive experiments demonstrate that our method consistently performs well, validating its effectiveness and robustness. The source code and benchmark dataset will be released on this https URL 

---
# Video Flow as Time Series: Discovering Temporal Consistency and Variability for VideoQA 

**Authors**: Zijie Song, Zhenzhen Hu, Yixiao Ma, Jia Li, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05783)  

**Abstract**: Video Question Answering (VideoQA) is a complex video-language task that demands a sophisticated understanding of both visual content and temporal dynamics. Traditional Transformer-style architectures, while effective in integrating multimodal data, often simplify temporal dynamics through positional encoding and fail to capture non-linear interactions within video sequences. In this paper, we introduce the Temporal Trio Transformer (T3T), a novel architecture that models time consistency and time variability. The T3T integrates three key components: Temporal Smoothing (TS), Temporal Difference (TD), and Temporal Fusion (TF). The TS module employs Brownian Bridge for capturing smooth, continuous temporal transitions, while the TD module identifies and encodes significant temporal variations and abrupt changes within the video content. Subsequently, the TF module synthesizes these temporal features with textual cues, facilitating a deeper contextual understanding and response accuracy. The efficacy of the T3T is demonstrated through extensive testing on multiple VideoQA benchmark datasets. Our results underscore the importance of a nuanced approach to temporal modeling in improving the accuracy and depth of video-based question answering. 

---
