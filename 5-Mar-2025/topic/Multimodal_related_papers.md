# Personalized Generation In Large Model Era: A Survey 

**Authors**: Yiyan Xu, Jinghao Zhang, Alireza Salemi, Xinting Hu, Wenjie Wang, Fuli Feng, Hamed Zamani, Xiangnan He, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2503.02614)  

**Abstract**: In the era of large models, content generation is gradually shifting to Personalized Generation (PGen), tailoring content to individual preferences and needs. This paper presents the first comprehensive survey on PGen, investigating existing research in this rapidly growing field. We conceptualize PGen from a unified perspective, systematically formalizing its key components, core objectives, and abstract workflows. Based on this unified perspective, we propose a multi-level taxonomy, offering an in-depth review of technical advancements, commonly used datasets, and evaluation metrics across multiple modalities, personalized contexts, and tasks. Moreover, we envision the potential applications of PGen and highlight open challenges and promising directions for future exploration. By bridging PGen research across multiple modalities, this survey serves as a valuable resource for fostering knowledge sharing and interdisciplinary collaboration, ultimately contributing to a more personalized digital landscape. 

---
# Evaluation of Architectural Synthesis Using Generative AI 

**Authors**: Jingfei Huang, Alexandros Haridis  

**Link**: [PDF](https://arxiv.org/pdf/2503.02861)  

**Abstract**: Recent advancements in multimodal Generative AI have the potential to democratize specialized architectural tasks, such as interpreting technical drawings and creating 3D CAD models, which traditionally require expert knowledge. This paper presents a comparative evaluation of two systems: GPT-4o and Claude 3.5, in the task of architectural 3D synthesis. We conduct a case study on two buildings from Palladio's Four Books of Architecture (1965): Villa Rotonda and Palazzo Porto. High-level architectural models and drawings of these buildings were prepared, inspired by Palladio's original texts and drawings. Through sequential text and image prompting, we assess the systems' abilities in (1) interpreting 2D and 3D representations of buildings from drawings, (2) encoding the buildings into a CAD software script, and (3) self-improving based on outputs. While both systems successfully generate individual parts, they struggle to accurately assemble these parts into the desired spatial relationships, with Claude 3.5 demonstrating better performance, particularly in self-correcting its output. This study contributes to ongoing research on benchmarking the strengths and weaknesses of off-the-shelf AI systems in performing intelligent human tasks that require discipline-specific knowledge. The findings highlight the potential of language-enabled AI systems to act as collaborative technical assistants in the architectural design process. 

---
# Attention Bootstrapping for Multi-Modal Test-Time Adaptation 

**Authors**: Yusheng Zhao, Junyu Luo, Xiao Luo, Jinsheng Huang, Jingyang Yuan, Zhiping Xiao, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02221)  

**Abstract**: Test-time adaptation aims to adapt a well-trained model to potential distribution shifts at test time using only unlabeled test data, without access to the original training data. While previous efforts mainly focus on a single modality, test-time distribution shift in the multi-modal setting is more complex and calls for new solutions. This paper tackles the problem of multi-modal test-time adaptation by proposing a novel method named Attention Bootstrapping with Principal Entropy Minimization (ABPEM). We observe that test-time distribution shift causes misalignment across modalities, leading to a large gap between intra-modality discrepancies (measured by self-attention) and inter-modality discrepancies (measured by cross-attention). We name this the attention gap. This attention gap widens with more severe distribution shifts, hindering effective modality fusion. To mitigate this attention gap and encourage better modality fusion, we propose attention bootstrapping that promotes cross-attention with the guidance of self-attention. Moreover, to reduce the gradient noise in the commonly-used entropy minimization, we adopt principal entropy minimization, a refinement of entropy minimization that reduces gradient noise by focusing on the principal parts of entropy, excluding less reliable gradient information. Extensive experiments on the benchmarks validate the effectiveness of the proposed ABPEM in comparison with competing baselines. 

---
# MindBridge: Scalable and Cross-Model Knowledge Editing via Memory-Augmented Modality 

**Authors**: Shuaike Li, Kai Zhang, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02701)  

**Abstract**: Knowledge editing is a technique for efficiently and accurately updating the knowledge of large language models (LLMs) to alleviate obsolescence and correct errors. However, most existing methods overfit to specific models, causing edited knowledge to be discarded during each LLM update and requiring frequent re-editing, which is particularly burdensome in today's rapidly evolving open-source community. To address this issue, we propose the problem of cross-model knowledge editing and introduce MindBridge, a scalable solution inspired by the low coupling between modality processing and LLMs in multi-modal models. MindBridge introduces the novel concept of memory modality, which encodes edited knowledge as an independent modality. It first performs LLM-agnostic pre-training of the memory modality and then integrates it with various LLMs. Extensive experiments on multiple LLMs and popular knowledge editing datasets demonstrate that MindBridge achieves superior performance even in editing tens of thousands of knowledge entries and can flexibly adapt to different LLMs. Our code is available at this https URL. 

---
# Multimodal Deep Learning for Subtype Classification in Breast Cancer Using Histopathological Images and Gene Expression Data 

**Authors**: Amin Honarmandi Shandiz  

**Link**: [PDF](https://arxiv.org/pdf/2503.02849)  

**Abstract**: Molecular subtyping of breast cancer is crucial for personalized treatment and prognosis. Traditional classification approaches rely on either histopathological images or gene expression profiling, limiting their predictive power. In this study, we propose a deep multimodal learning framework that integrates histopathological images and gene expression data to classify breast cancer into this http URL and this http URL / Her2 subtypes. Our approach employs a ResNet-50 model for image feature extraction and fully connected layers for gene expression processing, with a cross-attention fusion mechanism to enhance modality interaction. We conduct extensive experiments using five-fold cross-validation, demonstrating that our multimodal integration outperforms unimodal approaches in terms of classification accuracy, precision-recall AUC, and F1-score. Our findings highlight the potential of deep learning for robust and interpretable breast cancer subtype classification, paving the way for improved clinical decision-making. 

---
# Developing a PET/CT Foundation Model for Cross-Modal Anatomical and Functional Imaging 

**Authors**: Yujin Oh, Robert Seifert, Yihan Cao, Christoph Clement, Justin Ferdinandus, Constantin Lapa, Alessandro Liebich, Michelle Amon, Johanna Enke, Sifan Song, Runqi Meng, Fang Zeng, Ning Guo, Xiang Li, Pedram Heidari, Axel Rominger, Kuangyu Shi, Quanzheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02824)  

**Abstract**: In oncology, Positron Emission Tomography-Computed Tomography (PET/CT) is widely used in cancer diagnosis, staging, and treatment monitoring, as it combines anatomical details from CT with functional metabolic activity and molecular marker expression information from PET. However, existing artificial intelligence-driven PET/CT analyses rely predominantly on task-specific models trained from scratch or on limited datasets, limiting their generalizability and robustness. To address this, we propose a foundation model approach specifically designed for multimodal PET/CT imaging. We introduce the Cross-Fraternal Twin Masked Autoencoder (FratMAE), a novel framework that effectively integrates whole-body anatomical and functional or molecular information. FratMAE employs separate Vision Transformer (ViT) encoders for PET and CT scans, along with cross-attention decoders that enable synergistic interactions between modalities during masked autoencoder training. Additionally, it incorporates textual metadata to enhance PET representation learning. By pre-training on PET/CT datasets, FratMAE captures intricate cross-modal relationships and global uptake patterns, achieving superior performance on downstream tasks and demonstrating its potential as a generalizable foundation model. 

---
# Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024 

**Authors**: Nuria Alina Chandra, Ryan Murtfeldt, Lin Qiu, Arnab Karmakar, Hannah Lee, Emmanuel Tanumihardja, Kevin Farhat, Ben Caffee, Sejin Paik, Changyeon Lee, Jongwook Choi, Aerin Kim, Oren Etzioni  

**Link**: [PDF](https://arxiv.org/pdf/2503.02857)  

**Abstract**: In the age of increasingly realistic generative AI, robust deepfake detection is essential for mitigating fraud and disinformation. While many deepfake detectors report high accuracy on academic datasets, we show that these academic benchmarks are out of date and not representative of recent deepfakes. We introduce Deepfake-Eval-2024, a new deepfake detection benchmark consisting of in-the-wild deepfakes collected from social media and deepfake detection platform users in 2024. Deepfake-Eval-2024 consists of 44 hours of videos, 56.5 hours of audio, and 1,975 images, encompassing the latest manipulation technologies. The benchmark contains diverse media content from 88 different websites in 52 different languages. We find that the performance of open-source state-of-the-art deepfake detection models drops precipitously when evaluated on Deepfake-Eval-2024, with AUC decreasing by 50% for video, 48% for audio, and 45% for image models compared to previous benchmarks. We also evaluate commercial deepfake detection models and models finetuned on Deepfake-Eval-2024, and find that they have superior performance to off-the-shelf open-source models, but they do not yet reach the accuracy of human deepfake forensic analysts. The dataset is available at this https URL. 

---
# Nexus-O: An Omni-Perceptive And -Interactive Model for Language, Audio, And Vision 

**Authors**: Che Liu, Yingji Zhang, Dong Zhang, Weijie Zhang, Chenggong Gong, Haohan Li, Yu Lu, Shilin Zhou, Yue Lu, Ziliang Gan, Ziao Wang, Junwei Liao, Haipang Wu, Ji Liu, André Freitas, Qifan Wang, Zenglin Xu, Rongjuncheng Zhang, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.01879)  

**Abstract**: Human beings perceive the real world through a spectrum of sensory modalities, encompassing auditory, visual, and linguistic faculties. The journey towards achieving Artificial General Intelligence (AGI) necessitates the development of models that can emulate these multifaceted perceptual capabilities and comprehensively understand these diversified data. To this end, we introduce \textbf{Nexus-O}, an industry-level \textbf{omni-perceptive and -interactive} model capable of efficiently processing Audio, Image, Video, and Text data in any combination and output audio/text in an end-to-end way. We systematically investigate Nexus-O by addressing three key research questions: First, how can models be efficiently designed and trained to achieve tri-modal alignment, understanding and reasoning capabilities across multiple modalities? Second, what approaches can be implemented to evaluate tri-modal model robustness, ensuring reliable performance and applicability in real-world scenarios? Third, what strategies can be employed to curate and obtain high-quality, real-life scenario speech datasets? For the first question, we design and pre-train Nexus-O based on the vision-language model, rather than the language model. By pre-training the model over high-quality synthetic audio data, our model is capable of tri-modal perception and interaction. For the second question, we introduce a new audio testbed, Nexus-O-audio, comprising diverse Automatic Speech Recognition (ASR) samples, spanning various real-world scenarios, such as corporate meetings and live stream. For the third question, we design the speech data synthesis pipeline to obtain high-quality speech training datasets, covering various real-world scenarios. Comprehensive experimentation and an in-depth analysis of tri-modal alignment over latent space demonstrate the advantages of our model on downstream tasks. 

---
# A Multimodal Symphony: Integrating Taste and Sound through Generative AI 

**Authors**: Matteo Spanio, Massimiliano Zampini, Antonio Rodà, Franco Pierucci  

**Link**: [PDF](https://arxiv.org/pdf/2503.02823)  

**Abstract**: In recent decades, neuroscientific and psychological research has traced direct relationships between taste and auditory perceptions. This article explores multimodal generative models capable of converting taste information into music, building on this foundational research. We provide a brief review of the state of the art in this field, highlighting key findings and methodologies. We present an experiment in which a fine-tuned version of a generative music model (MusicGEN) is used to generate music based on detailed taste descriptions provided for each musical piece. The results are promising: according the participants' ($n=111$) evaluation, the fine-tuned model produces music that more coherently reflects the input taste descriptions compared to the non-fine-tuned model. This study represents a significant step towards understanding and developing embodied interactions between AI, sound, and taste, opening new possibilities in the field of generative AI. We release our dataset, code and pre-trained model at: this https URL. 

---
# BioD2C: A Dual-level Semantic Consistency Constraint Framework for Biomedical VQA 

**Authors**: Zhengyang Ji, Shang Gao, Li Liu, Yifan Jia, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2503.02476)  

**Abstract**: Biomedical visual question answering (VQA) has been widely studied and has demonstrated significant application value and potential in fields such as assistive medical diagnosis. Despite their success, current biomedical VQA models perform multimodal information interaction only at the model level within large language models (LLMs), leading to suboptimal multimodal semantic alignment when dealing with complex tasks. To address this issue, we propose BioD2C: a novel Dual-level Semantic Consistency Constraint Framework for Biomedical VQA, which achieves dual-level semantic interaction alignment at both the model and feature levels, enabling the model to adaptively learn visual features based on the question. Specifically, we firstly integrate textual features into visual features via an image-text fusion mechanism as feature-level semantic interaction, obtaining visual features conditioned on the given text; and then introduce a text-queue-based cross-modal soft semantic loss function to further align the image semantics with the question semantics. Specifically, in this work, we establish a new dataset, BioVGQ, to address inherent biases in prior datasets by filtering manually-altered images and aligning question-answer pairs with multimodal context, and train our model on this dataset. Extensive experimental results demonstrate that BioD2C achieves state-of-the-art (SOTA) performance across multiple downstream datasets, showcasing its robustness, generalizability, and potential to advance biomedical VQA research. 

---
# Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs 

**Authors**: Wei-Yao Wang, Zhao Wang, Helen Suzuki, Yoshiyuki Kobayashi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02597)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) have demonstrated significant progress in perceiving and reasoning over multimodal inquiries, ushering in a new research era for foundation models. However, vision-language misalignment in MLLMs has emerged as a critical challenge, where the textual responses generated by these models are not factually aligned with the given text-image inputs. Existing efforts to address vision-language misalignment have focused on developing specialized vision-language connectors or leveraging visual instruction tuning from diverse domains. In this paper, we tackle this issue from a fundamental yet unexplored perspective by revisiting the core architecture of MLLMs. Most MLLMs are typically built on decoder-only LLMs consisting of a causal attention mechanism, which limits the ability of earlier modalities (e.g., images) to incorporate information from later modalities (e.g., text). To address this problem, we propose AKI, a novel MLLM that unlocks causal attention into modality-mutual attention (MMA) to enable image tokens to attend to text tokens. This simple yet effective design allows AKI to achieve superior performance in 12 multimodal understanding benchmarks (+7.2% on average) without introducing additional parameters and increasing training time. Our MMA design is intended to be generic, allowing for application across various modalities, and scalable to accommodate diverse multimodal scenarios. The code is publicly available at this https URL, and we will release our AKI-4B model to encourage further advancements in MLLMs across various directions. 

---
# Are Large Vision Language Models Good Game Players? 

**Authors**: Xinyu Wang, Bohan Zhuang, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02358)  

**Abstract**: Large Vision Language Models (LVLMs) have demonstrated remarkable abilities in understanding and reasoning about both visual and textual information. However, existing evaluation methods for LVLMs, primarily based on benchmarks like Visual Question Answering and image captioning, often fail to capture the full scope of LVLMs' capabilities. These benchmarks are limited by issues such as inadequate assessment of detailed visual perception, data contamination, and a lack of focus on multi-turn reasoning. To address these challenges, we propose \method{}, a game-based evaluation framework designed to provide a comprehensive assessment of LVLMs' cognitive and reasoning skills in structured environments. \method{} uses a set of games to evaluate LVLMs on four core tasks: Perceiving, Question Answering, Rule Following, and End-to-End Playing, with each target task designed to assess specific abilities, including visual perception, reasoning, decision-making, etc. Based on this framework, we conduct extensive experiments that explore the limitations of current LVLMs, such as handling long structured outputs and perceiving detailed and dense elements. Code and data are publicly available at this https URL. 

---
# GRADEO: Towards Human-Like Evaluation for Text-to-Video Generation via Multi-Step Reasoning 

**Authors**: Zhun Mou, Bin Xia, Zhengchao Huang, Wenming Yang, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.02341)  

**Abstract**: Recent great advances in video generation models have demonstrated their potential to produce high-quality videos, bringing challenges to effective evaluation. Unlike human evaluation, existing automated evaluation metrics lack high-level semantic understanding and reasoning capabilities for video, thus making them infeasible and unexplainable. To fill this gap, we curate GRADEO-Instruct, a multi-dimensional T2V evaluation instruction tuning dataset, including 3.3k videos from over 10 existing video generation models and multi-step reasoning assessments converted by 16k human annotations. We then introduce GRADEO, one of the first specifically designed video evaluation models, which grades AI-generated videos for explainable scores and assessments through multi-step reasoning. Experiments show that our method aligns better with human evaluations than existing methods. Furthermore, our benchmarking reveals that current video generation models struggle to produce content that aligns with human reasoning and complex real-world scenarios. The models, datasets, and codes will be released soon. 

---
# Semi-Supervised Audio-Visual Video Action Recognition with Audio Source Localization Guided Mixup 

**Authors**: Seokun Kang, Taehwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.02284)  

**Abstract**: Video action recognition is a challenging but important task for understanding and discovering what the video does. However, acquiring annotations for a video is costly, and semi-supervised learning (SSL) has been studied to improve performance even with a small number of labeled data in the task. Prior studies for semi-supervised video action recognition have mostly focused on using single modality - visuals - but the video is multi-modal, so utilizing both visuals and audio would be desirable and improve performance further, which has not been explored well. Therefore, we propose audio-visual SSL for video action recognition, which uses both visual and audio together, even with quite a few labeled data, which is challenging. In addition, to maximize the information of audio and video, we propose a novel audio source localization-guided mixup method that considers inter-modal relations between video and audio modalities. In experiments on UCF-51, Kinetics-400, and VGGSound datasets, our model shows the superior performance of the proposed semi-supervised audio-visual action recognition framework and audio source localization-guided mixup. 

---
# Words or Vision: Do Vision-Language Models Have Blind Faith in Text? 

**Authors**: Ailin Deng, Tri Cao, Zhirui Chen, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02199)  

**Abstract**: Vision-Language Models (VLMs) excel in integrating visual and textual information for vision-centric tasks, but their handling of inconsistencies between modalities is underexplored. We investigate VLMs' modality preferences when faced with visual data and varied textual inputs in vision-centered settings. By introducing textual variations to four vision-centric tasks and evaluating ten Vision-Language Models (VLMs), we discover a \emph{``blind faith in text''} phenomenon: VLMs disproportionately trust textual data over visual data when inconsistencies arise, leading to significant performance drops under corrupted text and raising safety concerns. We analyze factors influencing this text bias, including instruction prompts, language model size, text relevance, token order, and the interplay between visual and textual certainty. While certain factors, such as scaling up the language model size, slightly mitigate text bias, others like token order can exacerbate it due to positional biases inherited from language models. To address this issue, we explore supervised fine-tuning with text augmentation and demonstrate its effectiveness in reducing text bias. Additionally, we provide a theoretical analysis suggesting that the blind faith in text phenomenon may stem from an imbalance of pure text and multi-modal data during training. Our findings highlight the need for balanced training and careful consideration of modality interactions in VLMs to enhance their robustness and reliability in handling multi-modal data inconsistencies. 

---
# DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models 

**Authors**: Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02175)  

**Abstract**: Large Multimodal Models (LMMs) have emerged as powerful models capable of understanding various data modalities, including text, images, and videos. LMMs encode both text and visual data into tokens that are then combined and processed by an integrated Large Language Model (LLM). Including visual tokens substantially increases the total token count, often by thousands. The increased input length for LLM significantly raises the complexity of inference, resulting in high latency in LMMs. To address this issue, token pruning methods, which remove part of the visual tokens, are proposed. The existing token pruning methods either require extensive calibration and fine-tuning or rely on suboptimal importance metrics which results in increased redundancy among the retained tokens. In this paper, we first formulate token pruning as Max-Min Diversity Problem (MMDP) where the goal is to select a subset such that the diversity among the selected {tokens} is maximized. Then, we solve the MMDP to obtain the selected subset and prune the rest. The proposed method, DivPrune, reduces redundancy and achieves the highest diversity of the selected tokens. By ensuring high diversity, the selected tokens better represent the original tokens, enabling effective performance even at high pruning ratios without requiring fine-tuning. Extensive experiments with various LMMs show that DivPrune achieves state-of-the-art accuracy over 16 image- and video-language datasets. Additionally, DivPrune reduces both the end-to-end latency and GPU memory usage for the tested models. The code is available $\href{this https URL}{\text{here}}$. 

---
# MedHEval: Benchmarking Hallucinations and Mitigation Strategies in Medical Large Vision-Language Models 

**Authors**: Aofei Chang, Le Huang, Parminder Bhatia, Taha Kass-Hout, Fenglong Ma, Cao Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02157)  

**Abstract**: Large Vision Language Models (LVLMs) are becoming increasingly important in the medical domain, yet Medical LVLMs (Med-LVLMs) frequently generate hallucinations due to limited expertise and the complexity of medical applications. Existing benchmarks fail to effectively evaluate hallucinations based on their underlying causes and lack assessments of mitigation strategies. To address this gap, we introduce MedHEval, a novel benchmark that systematically evaluates hallucinations and mitigation strategies in Med-LVLMs by categorizing them into three underlying causes: visual misinterpretation, knowledge deficiency, and context misalignment. We construct a diverse set of close- and open-ended medical VQA datasets with comprehensive evaluation metrics to assess these hallucination types. We conduct extensive experiments across 11 popular (Med)-LVLMs and evaluate 7 state-of-the-art hallucination mitigation techniques. Results reveal that Med-LVLMs struggle with hallucinations arising from different causes while existing mitigation methods show limited effectiveness, especially for knowledge- and context-based errors. These findings underscore the need for improved alignment training and specialized mitigation strategies to enhance Med-LVLMs' reliability. MedHEval establishes a standardized framework for evaluating and mitigating medical hallucinations, guiding the development of more trustworthy Med-LVLMs. 

---
# LLMs as Educational Analysts: Transforming Multimodal Data Traces into Actionable Reading Assessment Reports 

**Authors**: Eduardo Davalos, Yike Zhang, Namrata Srivastava, Jorge Alberto Salas, Sara McFadden, Sun-Joo Cho, Gautam Biswas, Amanda Goodwin  

**Link**: [PDF](https://arxiv.org/pdf/2503.02099)  

**Abstract**: Reading assessments are essential for enhancing students' comprehension, yet many EdTech applications focus mainly on outcome-based metrics, providing limited insights into student behavior and cognition. This study investigates the use of multimodal data sources -- including eye-tracking data, learning outcomes, assessment content, and teaching standards -- to derive meaningful reading insights. We employ unsupervised learning techniques to identify distinct reading behavior patterns, and then a large language model (LLM) synthesizes the derived information into actionable reports for educators, streamlining the interpretation process. LLM experts and human educators evaluate these reports for clarity, accuracy, relevance, and pedagogical usefulness. Our findings indicate that LLMs can effectively function as educational analysts, turning diverse data into teacher-friendly insights that are well-received by educators. While promising for automating insight generation, human oversight remains crucial to ensure reliability and fairness. This research advances human-centered AI in education, connecting data-driven analytics with practical classroom applications. 

---
# Abn-BLIP: Abnormality-aligned Bootstrapping Language-Image Pre-training for Pulmonary Embolism Diagnosis and Report Generation from CTPA 

**Authors**: Zhusi Zhong, Yuli Wang, Lulu Bi, Zhuoqi Ma, Sun Ho Ahn, Christopher J. Mullin, Colin F. Greineder, Michael K. Atalay, Scott Collins, Grayson L. Baird, Cheng Ting Lin, Webster Stayman, Todd M. Kolb, Ihab Kamel, Harrison X. Bai, Zhicheng Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02034)  

**Abstract**: Medical imaging plays a pivotal role in modern healthcare, with computed tomography pulmonary angiography (CTPA) being a critical tool for diagnosing pulmonary embolism and other thoracic conditions. However, the complexity of interpreting CTPA scans and generating accurate radiology reports remains a significant challenge. This paper introduces Abn-BLIP (Abnormality-aligned Bootstrapping Language-Image Pretraining), an advanced diagnosis model designed to align abnormal findings to generate the accuracy and comprehensiveness of radiology reports. By leveraging learnable queries and cross-modal attention mechanisms, our model demonstrates superior performance in detecting abnormalities, reducing missed findings, and generating structured reports compared to existing methods. Our experiments show that Abn-BLIP outperforms state-of-the-art medical vision-language models and 3D report generation methods in both accuracy and clinical relevance. These results highlight the potential of integrating multimodal learning strategies for improving radiology reporting. The source code is available at this https URL. 

---
# Recurrence-Enhanced Vision-and-Language Transformers for Robust Multimodal Document Retrieval 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2503.01980)  

**Abstract**: Cross-modal retrieval is gaining increasing efficacy and interest from the research community, thanks to large-scale training, novel architectural and learning designs, and its application in LLMs and multimodal LLMs. In this paper, we move a step forward and design an approach that allows for multimodal queries, composed of both an image and a text, and can search within collections of multimodal documents, where images and text are interleaved. Our model, ReT, employs multi-level representations extracted from different layers of both visual and textual backbones, both at the query and document side. To allow for multi-level and cross-modal understanding and feature extraction, ReT employs a novel Transformer-based recurrent cell that integrates both textual and visual features at different layers, and leverages sigmoidal gates inspired by the classical design of LSTMs. Extensive experiments on M2KR and M-BEIR benchmarks show that ReT achieves state-of-the-art performance across diverse settings. Our source code and trained models are publicly available at this https URL. 

---
# What are You Looking at? Modality Contribution in Multimodal Medical Deep Learning Methods 

**Authors**: Christian Gapp, Elias Tappeiner, Martin Welk, Karl Fritscher, Elke Ruth Gizewski, Rainer Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2503.01904)  

**Abstract**: Purpose High dimensional, multimodal data can nowadays be analyzed by huge deep neural networks with little effort. Several fusion methods for bringing together different modalities have been developed. Particularly, in the field of medicine with its presence of high dimensional multimodal patient data, multimodal models characterize the next step. However, what is yet very underexplored is how these models process the source information in detail. Methods To this end, we implemented an occlusion-based both model and performance agnostic modality contribution method that quantitatively measures the importance of each modality in the dataset for the model to fulfill its task. We applied our method to three different multimodal medical problems for experimental purposes. Results Herein we found that some networks have modality preferences that tend to unimodal collapses, while some datasets are imbalanced from the ground up. Moreover, we could determine a link between our metric and the performance of single modality trained nets. Conclusion The information gain through our metric holds remarkable potential to improve the development of multimodal models and the creation of datasets in the future. With our method we make a crucial contribution to the field of interpretability in deep learning based multimodal research and thereby notably push the integrability of multimodal AI into clinical practice. Our code is publicly available at this https URL. 

---
# When Continue Learning Meets Multimodal Large Language Model: A Survey 

**Authors**: Yukang Huo, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01887)  

**Abstract**: Recent advancements in Artificial Intelligence have led to the development of Multimodal Large Language Models (MLLMs). However, adapting these pre-trained models to dynamic data distributions and various tasks efficiently remains a challenge. Fine-tuning MLLMs for specific tasks often causes performance degradation in the model's prior knowledge domain, a problem known as 'Catastrophic Forgetting'. While this issue has been well-studied in the Continual Learning (CL) community, it presents new challenges for MLLMs. This review paper, the first of its kind in MLLM continual learning, presents an overview and analysis of 440 research papers in this this http URL review is structured into four sections. First, it discusses the latest research on MLLMs, covering model innovations, benchmarks, and applications in various fields. Second, it categorizes and overviews the latest studies on continual learning, divided into three parts: non-large language models unimodal continual learning (Non-LLM Unimodal CL), non-large language models multimodal continual learning (Non-LLM Multimodal CL), and continual learning in large language models (CL in LLM). The third section provides a detailed analysis of the current state of MLLM continual learning research, including benchmark evaluations, architectural innovations, and a summary of theoretical and empirical this http URL, the paper discusses the challenges and future directions of continual learning in MLLMs, aiming to inspire future research and development in the field. This review connects the foundational concepts, theoretical insights, method innovations, and practical applications of continual learning for multimodal large models, providing a comprehensive understanding of the research progress and challenges in this field, aiming to inspire researchers in the field and promote the advancement of related technologies. 

---
# Vision Language Models in Medicine 

**Authors**: Beria Chingnabe Kalpelbe, Angel Gabriel Adaambiik, Wei Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01863)  

**Abstract**: With the advent of Vision-Language Models (VLMs), medical artificial intelligence (AI) has experienced significant technological progress and paradigm shifts. This survey provides an extensive review of recent advancements in Medical Vision-Language Models (Med-VLMs), which integrate visual and textual data to enhance healthcare outcomes. We discuss the foundational technology behind Med-VLMs, illustrating how general models are adapted for complex medical tasks, and examine their applications in healthcare. The transformative impact of Med-VLMs on clinical practice, education, and patient care is highlighted, alongside challenges such as data scarcity, narrow task generalization, interpretability issues, and ethical concerns like fairness, accountability, and privacy. These limitations are exacerbated by uneven dataset distribution, computational demands, and regulatory hurdles. Rigorous evaluation methods and robust regulatory frameworks are essential for safe integration into healthcare workflows. Future directions include leveraging large-scale, diverse datasets, improving cross-modal generalization, and enhancing interpretability. Innovations like federated learning, lightweight architectures, and Electronic Health Record (EHR) integration are explored as pathways to democratize access and improve clinical relevance. This review aims to provide a comprehensive understanding of Med-VLMs' strengths and limitations, fostering their ethical and balanced adoption in healthcare. 

---
# A Review of Artificial Intelligence Impacting Statistical Process Monitoring and Future Directions 

**Authors**: Shing I Chang, Parviz Ghafariasl  

**Link**: [PDF](https://arxiv.org/pdf/2503.01858)  

**Abstract**: It has been 100 years since statistical process control (SPC) or statistical process monitoring (SPM) was first introduced for production processes and later applied to service, healthcare, and other industries. The techniques applied to SPM applications are mostly statistically oriented. Recent advances in Artificial Intelligence (AI) have reinvigorated the imagination of adopting AI for SPM applications. This manuscript begins with a concise review of the historical development of the statistically based SPM methods. Next, this manuscript explores AI and Machine Learning (ML) algorithms and methods applied in various SPM applications, addressing quality characteristics of univariate, multivariate, profile, and image. These AI methods can be classified into the following categories: classification, pattern recognition, time series applications, and generative AI. Specifically, different kinds of neural networks, such as artificial neural networks (ANN), convolutional neural networks (CNN), recurrent neural networks (RNN), and generative adversarial networks (GAN), are among the most implemented AI methods impacting SPM. Finally, this manuscript outlines a couple of future directions that harness the potential of the Large Multimodal Model (LMM) for advancing SPM research and applications in complex systems. The ultimate objective is to transform statistical process monitoring (SPM) into smart process control (SMPC), where corrective actions are autonomously implemented to either prevent quality issues or restore process performance. 

---
# FairSense-AI: Responsible AI Meets Sustainability 

**Authors**: Shaina Raza, Mukund Sayeeganesh Chettiar, Matin Yousefabadi, Tahniat Khan, Marcelo Lotif  

**Link**: [PDF](https://arxiv.org/pdf/2503.02865)  

**Abstract**: In this paper, we introduce FairSense-AI: a multimodal framework designed to detect and mitigate bias in both text and images. By leveraging Large Language Models (LLMs) and Vision-Language Models (VLMs), FairSense-AI uncovers subtle forms of prejudice or stereotyping that can appear in content, providing users with bias scores, explanatory highlights, and automated recommendations for fairness enhancements. In addition, FairSense-AI integrates an AI risk assessment component that aligns with frameworks like the MIT AI Risk Repository and NIST AI Risk Management Framework, enabling structured identification of ethical and safety concerns. The platform is optimized for energy efficiency via techniques such as model pruning and mixed-precision computation, thereby reducing its environmental footprint. Through a series of case studies and applications, we demonstrate how FairSense-AI promotes responsible AI use by addressing both the social dimension of fairness and the pressing need for sustainability in large-scale AI deployments. this https URL, this https URL 

---
# MciteBench: A Benchmark for Multimodal Citation Text Generation in MLLMs 

**Authors**: Caiyu Hu, Yikai Zhang, Tinghui Zhu, Yiwei Ye, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02589)  

**Abstract**: Multimodal Large Language Models (MLLMs) have advanced in integrating diverse modalities but frequently suffer from hallucination. A promising solution to mitigate this issue is to generate text with citations, providing a transparent chain for verification. However, existing work primarily focuses on generating citations for text-only content, overlooking the challenges and opportunities of multimodal contexts. To address this gap, we introduce MCiteBench, the first benchmark designed to evaluate and analyze the multimodal citation text generation ability of MLLMs. Our benchmark comprises data derived from academic papers and review-rebuttal interactions, featuring diverse information sources and multimodal content. We comprehensively evaluate models from multiple dimensions, including citation quality, source reliability, and answer accuracy. Through extensive experiments, we observe that MLLMs struggle with multimodal citation text generation. We also conduct deep analyses of models' performance, revealing that the bottleneck lies in attributing the correct sources rather than understanding the multimodal content. 

---
# MMSciBench: Benchmarking Language Models on Multimodal Scientific Problems 

**Authors**: Xinwu Ye, Chengfan Li, Siming Chen, Xiangru Tang, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.01891)  

**Abstract**: Recent advances in large language models (LLMs) and vision-language models (LVLMs) have shown promise across many tasks, yet their scientific reasoning capabilities remain untested, particularly in multimodal settings. We present MMSciBench, a benchmark for evaluating mathematical and physical reasoning through text-only and text-image formats, with human-annotated difficulty levels, solutions with detailed explanations, and taxonomic mappings. Evaluation of state-of-the-art models reveals significant limitations, with even the best model achieving only \textbf{63.77\%} accuracy and particularly struggling with visual reasoning tasks. Our analysis exposes critical gaps in complex reasoning and visual-textual integration, establishing MMSciBench as a rigorous standard for measuring progress in multimodal scientific understanding. The code for MMSciBench is open-sourced at GitHub, and the dataset is available at Hugging Face. 

---
