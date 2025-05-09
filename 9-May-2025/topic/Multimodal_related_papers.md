# Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging 

**Authors**: Shiqi Chen, Jinghan Zhang, Tongyao Zhu, Wei Liu, Siyang Gao, Miao Xiong, Manling Li, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05464)  

**Abstract**: Vision-Language Models (VLMs) combine visual perception with the general capabilities, such as reasoning, of Large Language Models (LLMs). However, the mechanisms by which these two abilities can be combined and contribute remain poorly understood. In this work, we explore to compose perception and reasoning through model merging that connects parameters of different models. Unlike previous works that often focus on merging models of the same kind, we propose merging models across modalities, enabling the incorporation of the reasoning capabilities of LLMs into VLMs. Through extensive experiments, we demonstrate that model merging offers a successful pathway to transfer reasoning abilities from LLMs to VLMs in a training-free manner. Moreover, we utilize the merged models to understand the internal mechanism of perception and reasoning and how merging affects it. We find that perception capabilities are predominantly encoded in the early layers of the model, whereas reasoning is largely facilitated by the middle-to-late layers. After merging, we observe that all layers begin to contribute to reasoning, whereas the distribution of perception abilities across layers remains largely unchanged. These observations shed light on the potential of model merging as a tool for multimodal integration and interpretation. 

---
# A Benchmark Dataset and a Framework for Urdu Multimodal Named Entity Recognition 

**Authors**: Hussain Ahmad, Qingyang Zeng, Jing Wan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05148)  

**Abstract**: The emergence of multimodal content, particularly text and images on social media, has positioned Multimodal Named Entity Recognition (MNER) as an increasingly important area of research within Natural Language Processing. Despite progress in high-resource languages such as English, MNER remains underexplored for low-resource languages like Urdu. The primary challenges include the scarcity of annotated multimodal datasets and the lack of standardized baselines. To address these challenges, we introduce the U-MNER framework and release the Twitter2015-Urdu dataset, a pioneering resource for Urdu MNER. Adapted from the widely used Twitter2015 dataset, it is annotated with Urdu-specific grammar rules. We establish benchmark baselines by evaluating both text-based and multimodal models on this dataset, providing comparative analyses to support future research on Urdu MNER. The U-MNER framework integrates textual and visual context using Urdu-BERT for text embeddings and ResNet for visual feature extraction, with a Cross-Modal Fusion Module to align and fuse information. Our model achieves state-of-the-art performance on the Twitter2015-Urdu dataset, laying the groundwork for further MNER research in low-resource languages. 

---
# Image-Text Relation Prediction for Multilingual Tweets 

**Authors**: Matīss Rikters, Edison Marrese-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2505.05040)  

**Abstract**: Various social networks have been allowing media uploads for over a decade now. Still, it has not always been clear what is their relation with the posted text or even if there is any at all. In this work, we explore how multilingual vision-language models tackle the task of image-text relation prediction in different languages, and construct a dedicated balanced benchmark data set from Twitter posts in Latvian along with their manual translations into English. We compare our results to previous work and show that the more recently released vision-language model checkpoints are becoming increasingly capable at this task, but there is still much room for further improvement. 

---
# Advancing Conversational Diagnostic AI with Multimodal Reasoning 

**Authors**: Khaled Saab, Jan Freyberg, Chunjong Park, Tim Strother, Yong Cheng, Wei-Hung Weng, David G.T. Barrett, David Stutz, Nenad Tomasev, Anil Palepu, Valentin Liévin, Yash Sharma, Roma Ruparel, Abdullah Ahmed, Elahe Vedadi, Kimberly Kanada, Cian Hughes, Yun Liu, Geoff Brown, Yang Gao, Sean Li, S. Sara Mahdavi, James Manyika, Katherine Chou, Yossi Matias, Avinatan Hassidim, Dale R. Webster, Pushmeet Kohli, S.M. Ali Eslami, Joëlle Barral, Adam Rodman, Vivek Natarajan, Mike Schaekermann, Tao Tu, Alan Karthikesalingam, Ryutaro Tanno  

**Link**: [PDF](https://arxiv.org/pdf/2505.04653)  

**Abstract**: Large Language Models (LLMs) have demonstrated great potential for conducting diagnostic conversations but evaluation has been largely limited to language-only interactions, deviating from the real-world requirements of remote care delivery. Instant messaging platforms permit clinicians and patients to upload and discuss multimodal medical artifacts seamlessly in medical consultation, but the ability of LLMs to reason over such data while preserving other attributes of competent diagnostic conversation remains unknown. Here we advance the conversational diagnosis and management performance of the Articulate Medical Intelligence Explorer (AMIE) through a new capability to gather and interpret multimodal data, and reason about this precisely during consultations. Leveraging Gemini 2.0 Flash, our system implements a state-aware dialogue framework, where conversation flow is dynamically controlled by intermediate model outputs reflecting patient states and evolving diagnoses. Follow-up questions are strategically directed by uncertainty in such patient states, leading to a more structured multimodal history-taking process that emulates experienced clinicians. We compared AMIE to primary care physicians (PCPs) in a randomized, blinded, OSCE-style study of chat-based consultations with patient actors. We constructed 105 evaluation scenarios using artifacts like smartphone skin photos, ECGs, and PDFs of clinical documents across diverse conditions and demographics. Our rubric assessed multimodal capabilities and other clinically meaningful axes like history-taking, diagnostic accuracy, management reasoning, communication, and empathy. Specialist evaluation showed AMIE to be superior to PCPs on 7/9 multimodal and 29/32 non-multimodal axes (including diagnostic accuracy). The results show clear progress in multimodal conversational diagnostic AI, but real-world translation needs further research. 

---
# REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM 

**Authors**: Madhur Jindal, Saurabh Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2505.04673)  

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.
We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$). 

---
# Rethinking Multimodal Sentiment Analysis: A High-Accuracy, Simplified Fusion Architecture 

**Authors**: Nischal Mandal, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04642)  

**Abstract**: Multimodal sentiment analysis, a pivotal task in affective computing, seeks to understand human emotions by integrating cues from language, audio, and visual signals. While many recent approaches leverage complex attention mechanisms and hierarchical architectures, we propose a lightweight, yet effective fusion-based deep learning model tailored for utterance-level emotion classification. Using the benchmark IEMOCAP dataset, which includes aligned text, audio-derived numeric features, and visual descriptors, we design a modality-specific encoder using fully connected layers followed by dropout regularization. The modality-specific representations are then fused using simple concatenation and passed through a dense fusion layer to capture cross-modal interactions. This streamlined architecture avoids computational overhead while preserving performance, achieving a classification accuracy of 92% across six emotion categories. Our approach demonstrates that with careful feature engineering and modular design, simpler fusion strategies can outperform or match more complex models, particularly in resource-constrained environments. 

---
# Adaptive Token Boundaries: Integrating Human Chunking Mechanisms into Multimodal LLMs 

**Authors**: Dongxing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04637)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have demonstrated remarkable capabilities in processing diverse data types, yet significant disparities persist between human cognitive processes and computational approaches to multimodal information integration. This research presents a systematic investigation into the parallels between human cross-modal chunking mechanisms and token representation methodologies in MLLMs. Through empirical studies comparing human performance patterns with model behaviors across visual-linguistic tasks, we demonstrate that conventional static tokenization schemes fundamentally constrain current models' capacity to simulate the dynamic, context-sensitive nature of human information processing. We propose a novel framework for dynamic cross-modal tokenization that incorporates adaptive boundaries, hierarchical representations, and alignment mechanisms grounded in cognitive science principles. Quantitative evaluations demonstrate that our approach yields statistically significant improvements over state-of-the-art models on benchmark tasks (+7.8% on Visual Question Answering, +5.3% on Complex Scene Description) while exhibiting more human-aligned error patterns and attention distributions. These findings contribute to the theoretical understanding of the relationship between human cognition and artificial intelligence, while providing empirical evidence for developing more cognitively plausible AI systems. 

---
# TokLIP: Marry Visual Tokens to CLIP for Multimodal Comprehension and Generation 

**Authors**: Haokun Lin, Teng Wang, Yixiao Ge, Yuying Ge, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05422)  

**Abstract**: Pioneering token-based works such as Chameleon and Emu3 have established a foundation for multimodal unification but face challenges of high training computational overhead and limited comprehension performance due to a lack of high-level semantics. In this paper, we introduce TokLIP, a visual tokenizer that enhances comprehension by semanticizing vector-quantized (VQ) tokens and incorporating CLIP-level semantics while enabling end-to-end multimodal autoregressive training with standard VQ tokens. TokLIP integrates a low-level discrete VQ tokenizer with a ViT-based token encoder to capture high-level continuous semantics. Unlike previous approaches (e.g., VILA-U) that discretize high-level features, TokLIP disentangles training objectives for comprehension and generation, allowing the direct application of advanced VQ tokenizers without the need for tailored quantization operations. Our empirical results demonstrate that TokLIP achieves exceptional data efficiency, empowering visual tokens with high-level semantic understanding while enhancing low-level generative capacity, making it well-suited for autoregressive Transformers in both comprehension and generation tasks. The code and models are available at this https URL. 

---
# Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models 

**Authors**: Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, Shouzheng Huang, Xinping Zhao, Borui Jiang, Lanqing Hong, Longyue Wang, Zhuotao Tian, Baoxing Huai, Wenhan Luo, Weihua Luo, Zheng Zhang, Baotian Hu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04921)  

**Abstract**: Reasoning lies at the heart of intelligence, shaping the ability to make decisions, draw conclusions, and generalize across domains. In artificial intelligence, as systems increasingly operate in open, uncertain, and multimodal environments, reasoning becomes essential for enabling robust and adaptive behavior. Large Multimodal Reasoning Models (LMRMs) have emerged as a promising paradigm, integrating modalities such as text, images, audio, and video to support complex reasoning capabilities and aiming to achieve comprehensive perception, precise understanding, and deep reasoning. As research advances, multimodal reasoning has rapidly evolved from modular, perception-driven pipelines to unified, language-centric frameworks that offer more coherent cross-modal understanding. While instruction tuning and reinforcement learning have improved model reasoning, significant challenges remain in omni-modal generalization, reasoning depth, and agentic behavior. To address these issues, we present a comprehensive and structured survey of multimodal reasoning research, organized around a four-stage developmental roadmap that reflects the field's shifting design philosophies and emerging capabilities. First, we review early efforts based on task-specific modules, where reasoning was implicitly embedded across stages of representation, alignment, and fusion. Next, we examine recent approaches that unify reasoning into multimodal LLMs, with advances such as Multimodal Chain-of-Thought (MCoT) and multimodal reinforcement learning enabling richer and more structured reasoning chains. Finally, drawing on empirical insights from challenging benchmarks and experimental cases of OpenAI O3 and O4-mini, we discuss the conceptual direction of native large multimodal reasoning models (N-LMRMs), which aim to support scalable, agentic, and adaptive reasoning and planning in complex, real-world environments. 

---
# SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models 

**Authors**: Shun Taguchi, Hideki Deguchi, Takumi Hamazaki, Hiroyuki Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2505.04911)  

**Abstract**: This study introduces SpatialPrompting, a novel framework that harnesses the emergent reasoning capabilities of off-the-shelf multimodal large language models to achieve zero-shot spatial reasoning in three-dimensional (3D) environments. Unlike existing methods that rely on expensive 3D-specific fine-tuning with specialized 3D inputs such as point clouds or voxel-based features, SpatialPrompting employs a keyframe-driven prompt generation strategy. This framework uses metrics such as vision-language similarity, Mahalanobis distance, field of view, and image sharpness to select a diverse and informative set of keyframes from image sequences and then integrates them with corresponding camera pose data to effectively abstract spatial relationships and infer complex 3D structures. The proposed framework not only establishes a new paradigm for flexible spatial reasoning that utilizes intuitive visual and positional cues but also achieves state-of-the-art zero-shot performance on benchmark datasets, such as ScanQA and SQA3D, across several metrics. The proposed method effectively eliminates the need for specialized 3D inputs and fine-tuning, offering a simpler and more scalable alternative to conventional approaches. 

---
# X-Driver: Explainable Autonomous Driving with Vision-Language Models 

**Authors**: Wei Liu, Jiyuan Zhang, Binxiong Zheng, Yufeng Hu, Yingzhan Lin, Zengfeng Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.05098)  

**Abstract**: End-to-end autonomous driving has advanced significantly, offering benefits such as system simplicity and stronger driving performance in both open-loop and closed-loop settings than conventional pipelines. However, existing frameworks still suffer from low success rates in closed-loop evaluations, highlighting their limitations in real-world deployment. In this paper, we introduce X-Driver, a unified multi-modal large language models(MLLMs) framework designed for closed-loop autonomous driving, leveraging Chain-of-Thought(CoT) and autoregressive modeling to enhance perception and decision-making. We validate X-Driver across multiple autonomous driving tasks using public benchmarks in CARLA simulation environment, including Bench2Drive[6]. Our experimental results demonstrate superior closed-loop performance, surpassing the current state-of-the-art(SOTA) while improving the interpretability of driving decisions. These findings underscore the importance of structured reasoning in end-to-end driving and establish X-Driver as a strong baseline for future research in closed-loop autonomous driving. 

---
# T2VTextBench: A Human Evaluation Benchmark for Textual Control in Video Generation Models 

**Authors**: Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, Jiale Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04946)  

**Abstract**: Thanks to recent advancements in scalable deep architectures and large-scale pretraining, text-to-video generation has achieved unprecedented capabilities in producing high-fidelity, instruction-following content across a wide range of styles, enabling applications in advertising, entertainment, and education. However, these models' ability to render precise on-screen text, such as captions or mathematical formulas, remains largely untested, posing significant challenges for applications requiring exact textual accuracy. In this work, we introduce T2VTextBench, the first human-evaluation benchmark dedicated to evaluating on-screen text fidelity and temporal consistency in text-to-video models. Our suite of prompts integrates complex text strings with dynamic scene changes, testing each model's ability to maintain detailed instructions across frames. We evaluate ten state-of-the-art systems, ranging from open-source solutions to commercial offerings, and find that most struggle to generate legible, consistent text. These results highlight a critical gap in current video generators and provide a clear direction for future research aimed at enhancing textual manipulation in video synthesis. 

---
# Towards Artificial Intelligence Research Assistant for Expert-Involved Learning 

**Authors**: Tianyu Liu, Simeng Han, Xiao Luo, Hanchen Wang, Pan Lu, Biqing Zhu, Yuge Wang, Keyi Li, Jiapeng Chen, Rihao Qu, Yufeng Liu, Xinyue Cui, Aviv Yaish, Yuhang Chen, Minsheng Hao, Chuhan Li, Kexing Li, Arman Cohan, Hua Xu, Mark Gerstein, James Zou, Hongyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04638)  

**Abstract**: Large Language Models (LLMs) and Large Multi-Modal Models (LMMs) have emerged as transformative tools in scientific research, yet their reliability and specific contributions to biomedical applications remain insufficiently characterized. In this study, we present \textbf{AR}tificial \textbf{I}ntelligence research assistant for \textbf{E}xpert-involved \textbf{L}earning (ARIEL), a multimodal dataset designed to benchmark and enhance two critical capabilities of LLMs and LMMs in biomedical research: summarizing extensive scientific texts and interpreting complex biomedical figures. To facilitate rigorous assessment, we create two open-source sets comprising biomedical articles and figures with designed questions. We systematically benchmark both open- and closed-source foundation models, incorporating expert-driven human evaluations conducted by doctoral-level experts. Furthermore, we improve model performance through targeted prompt engineering and fine-tuning strategies for summarizing research papers, and apply test-time computational scaling to enhance the reasoning capabilities of LMMs, achieving superior accuracy compared to human-expert corrections. We also explore the potential of using LMM Agents to generate scientific hypotheses from diverse multimodal inputs. Overall, our results delineate clear strengths and highlight significant limitations of current foundation models, providing actionable insights and guiding future advancements in deploying large-scale language and multi-modal models within biomedical research. 

---
# Multimodal Benchmarking and Recommendation of Text-to-Image Generation Models 

**Authors**: Kapil Wanaskar, Gaytri Jena, Magdalini Eirinaki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04650)  

**Abstract**: This work presents an open-source unified benchmarking and evaluation framework for text-to-image generation models, with a particular focus on the impact of metadata augmented prompts. Leveraging the DeepFashion-MultiModal dataset, we assess generated outputs through a comprehensive set of quantitative metrics, including Weighted Score, CLIP (Contrastive Language Image Pre-training)-based similarity, LPIPS (Learned Perceptual Image Patch Similarity), FID (Frechet Inception Distance), and retrieval-based measures, as well as qualitative analysis. Our results demonstrate that structured metadata enrichments greatly enhance visual realism, semantic fidelity, and model robustness across diverse text-to-image architectures. While not a traditional recommender system, our framework enables task-specific recommendations for model selection and prompt design based on evaluation metrics. 

---
# Learning Item Representations Directly from Multimodal Features for Effective Recommendation 

**Authors**: Xin Zhou, Xiaoxiong Zhang, Dusit Niyato, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04960)  

**Abstract**: Conventional multimodal recommender systems predominantly leverage Bayesian Personalized Ranking (BPR) optimization to learn item representations by amalgamating item identity (ID) embeddings with multimodal features. Nevertheless, our empirical and theoretical findings unequivocally demonstrate a pronounced optimization gradient bias in favor of acquiring representations from multimodal features over item ID embeddings. As a consequence, item ID embeddings frequently exhibit suboptimal characteristics despite the convergence of multimodal feature parameters. Given the rich informational content inherent in multimodal features, in this paper, we propose a novel model (i.e., LIRDRec) that learns item representations directly from these features to augment recommendation performance. Recognizing that features derived from each modality may capture disparate yet correlated aspects of items, we propose a multimodal transformation mechanism, integrated with modality-specific encoders, to effectively fuse features from all modalities. Moreover, to differentiate the influence of diverse modality types, we devise a progressive weight copying fusion module within LIRDRec. This module incrementally learns the weight assigned to each modality in synthesizing the final user or item representations. Finally, we utilize the powerful visual understanding of Multimodal Large Language Models (MLLMs) to convert the item images into texts and extract semantics embeddings upon the texts via LLMs. Empirical evaluations conducted on five real-world datasets validate the superiority of our approach relative to competing baselines. It is worth noting the proposed model, equipped with embeddings extracted from MLLMs and LLMs, can further improve the recommendation accuracy of NDCG@20 by an average of 4.21% compared to the original embeddings. 

---
# A Pain Assessment Framework based on multimodal data and Deep Machine Learning methods 

**Authors**: Stefanos Gkikas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05396)  

**Abstract**: From the original abstract:
This thesis initially aims to study the pain assessment process from a clinical-theoretical perspective while exploring and examining existing automatic approaches. Building on this foundation, the primary objective of this Ph.D. project is to develop innovative computational methods for automatic pain assessment that achieve high performance and are applicable in real clinical settings. A primary goal is to thoroughly investigate and assess significant factors, including demographic elements that impact pain perception, as recognized in pain research, through a computational standpoint. Within the limits of the available data in this research area, our goal was to design, develop, propose, and offer automatic pain assessment pipelines for unimodal and multimodal configurations that are applicable to the specific requirements of different scenarios. The studies published in this Ph.D. thesis showcased the effectiveness of the proposed methods, achieving state-of-the-art results. Additionally, they paved the way for exploring new approaches in artificial intelligence, foundation models, and generative artificial intelligence. 

---
# EcoAgent: An Efficient Edge-Cloud Collaborative Multi-Agent Framework for Mobile Automation 

**Authors**: Biao Yi, Xavier Hu, Yurun Chen, Shengyu Zhang, Hongxia Yang, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05440)  

**Abstract**: Cloud-based mobile agents powered by (multimodal) large language models ((M)LLMs) offer strong reasoning abilities but suffer from high latency and cost. While fine-tuned (M)SLMs enable edge deployment, they often lose general capabilities and struggle with complex tasks. To address this, we propose EcoAgent, an Edge-Cloud cOllaborative multi-agent framework for mobile automation. EcoAgent features a closed-loop collaboration among a cloud-based Planning Agent and two edge-based agents: the Execution Agent for action execution and the Observation Agent for verifying outcomes. The Observation Agent uses a Pre-Understanding Module to compress screen images into concise text, reducing token usage. In case of failure, the Planning Agent retrieves screen history and replans via a Reflection Module. Experiments on AndroidWorld show that EcoAgent maintains high task success rates while significantly reducing MLLM token consumption, enabling efficient and practical mobile automation. 

---
# Mapping User Trust in Vision Language Models: Research Landscape, Challenges, and Prospects 

**Authors**: Agnese Chiatti, Sara Bernardini, Lara Shibelski Godoy Piccolo, Viola Schiaffonati, Matteo Matteucci  

**Link**: [PDF](https://arxiv.org/pdf/2505.05318)  

**Abstract**: The rapid adoption of Vision Language Models (VLMs), pre-trained on large image-text and video-text datasets, calls for protecting and informing users about when to trust these systems. This survey reviews studies on trust dynamics in user-VLM interactions, through a multi-disciplinary taxonomy encompassing different cognitive science capabilities, collaboration modes, and agent behaviours. Literature insights and findings from a workshop with prospective VLM users inform preliminary requirements for future VLM trust studies. 

---
# Biomed-DPT: Dual Modality Prompt Tuning for Biomedical Vision-Language Models 

**Authors**: Wei Peng, Kang Liu, Jianchen Hu, Meng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05189)  

**Abstract**: Prompt learning is one of the most effective paradigms for adapting pre-trained vision-language models (VLMs) to the biomedical image classification tasks in few shot scenarios. However, most of the current prompt learning methods only used the text prompts and ignored the particular structures (such as the complex anatomical structures and subtle pathological features) in the biomedical images. In this work, we propose Biomed-DPT, a knowledge-enhanced dual modality prompt tuning technique. In designing the text prompt, Biomed-DPT constructs a dual prompt including the template-driven clinical prompts and the large language model (LLM)-driven domain-adapted prompts, then extracts the clinical knowledge from the domain-adapted prompts through the knowledge distillation technique. In designing the vision prompt, Biomed-DPT introduces the zero vector as a soft prompt to leverage attention re-weighting so that the focus on non-diagnostic regions and the recognition of non-critical pathological features are avoided. Biomed-DPT achieves an average classification accuracy of 66.14\% across 11 biomedical image datasets covering 9 modalities and 10 organs, with performance reaching 78.06\% in base classes and 75.97\% in novel classes, surpassing the Context Optimization (CoOp) method by 6.20\%, 3.78\%, and 8.04\%, respectively. Our code are available at \underline{this https URL}. 

---
# FG-CLIP: Fine-Grained Visual and Textual Alignment 

**Authors**: Chunyu Xie, Bin Wang, Fanjing Kong, Jincheng Li, Dawei Liang, Gengshen Zhang, Dawei Leng, Yuhui Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05071)  

**Abstract**: Contrastive Language-Image Pre-training (CLIP) excels in multimodal tasks such as image-text retrieval and zero-shot classification but struggles with fine-grained understanding due to its focus on coarse-grained short captions. To address this, we propose Fine-Grained CLIP (FG-CLIP), which enhances fine-grained understanding through three key innovations. First, we leverage large multimodal models to generate 1.6 billion long caption-image pairs for capturing global-level semantic details. Second, a high-quality dataset is constructed with 12 million images and 40 million region-specific bounding boxes aligned with detailed captions to ensure precise, context-rich representations. Third, 10 million hard fine-grained negative samples are incorporated to improve the model's ability to distinguish subtle semantic differences. Corresponding training methods are meticulously designed for these data. Extensive experiments demonstrate that FG-CLIP outperforms the original CLIP and other state-of-the-art methods across various downstream tasks, including fine-grained understanding, open-vocabulary object detection, image-text retrieval, and general multimodal benchmarks. These results highlight FG-CLIP's effectiveness in capturing fine-grained image details and improving overall model performance. The related data, code, and models are available at this https URL. 

---
