# Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs 

**Authors**: Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi-ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yunsheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Mengchen Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez-Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xia Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, Xiren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.01743)  

**Abstract**: We introduce Phi-4-Mini and Phi-4-Multimodal, compact yet highly capable language and multimodal models. Phi-4-Mini is a 3.8-billion-parameter language model trained on high-quality web and synthetic data, significantly outperforming recent open-source models of similar size and matching the performance of models twice its size on math and coding tasks requiring complex reasoning. This achievement is driven by a carefully curated synthetic data recipe emphasizing high-quality math and coding datasets. Compared to its predecessor, Phi-3.5-Mini, Phi-4-Mini features an expanded vocabulary size of 200K tokens to better support multilingual applications, as well as group query attention for more efficient long-sequence generation. Phi-4-Multimodal is a multimodal model that integrates text, vision, and speech/audio input modalities into a single model. Its novel modality extension approach leverages LoRA adapters and modality-specific routers to allow multiple inference modes combining various modalities without interference. For example, it now ranks first in the OpenASR leaderboard to date, although the LoRA component of the speech/audio modality has just 460 million parameters. Phi-4-Multimodal supports scenarios involving (vision + language), (vision + speech), and (speech/audio) inputs, outperforming larger vision-language and speech-language models on a wide range of tasks. Additionally, we experiment to further train Phi-4-Mini to enhance its reasoning capabilities. Despite its compact 3.8-billion-parameter size, this experimental version achieves reasoning performance on par with or surpassing significantly larger models, including DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B. 

---
# Scientific Reasoning: Assessment of Multimodal Generative LLMs 

**Authors**: Florian Dreyer, Ekaterina Kolos, Daria Matiash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01064)  

**Abstract**: Large language models (LLMs) can answer questions and reason about complex tasks, also from the scientific domain. We assess several multimodal LLMs (MLLMs) on ScienceQA and find that Gemini models show the highest accuracy with little context, and the highest textual similarity to human explanations with richer context. Adapter-tuning of smaller MLLMs did not lead to any reliable performance. Training from Gemini outputs consistently underperformed training from the original data. 

---
# How Deep is Love in LLMs' Hearts? Exploring Semantic Size in Human-like Cognition 

**Authors**: Yao Yao, Yifei Yang, Xinbei Ma, Dongjie Yang, Zhuosheng Zhang, Zuchao Li, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00330)  

**Abstract**: How human cognitive abilities are formed has long captivated researchers. However, a significant challenge lies in developing meaningful methods to measure these complex processes. With the advent of large language models (LLMs), which now rival human capabilities in various domains, we are presented with a unique testbed to investigate human cognition through a new lens. Among the many facets of cognition, one particularly crucial aspect is the concept of semantic size, the perceived magnitude of both abstract and concrete words or concepts. This study seeks to investigate whether LLMs exhibit similar tendencies in understanding semantic size, thereby providing insights into the underlying mechanisms of human cognition. We begin by exploring metaphorical reasoning, comparing how LLMs and humans associate abstract words with concrete objects of varying sizes. Next, we examine LLMs' internal representations to evaluate their alignment with human cognitive processes. Our findings reveal that multi-modal training is crucial for LLMs to achieve more human-like understanding, suggesting that real-world, multi-modal experiences are similarly vital for human cognitive development. Lastly, we examine whether LLMs are influenced by attention-grabbing headlines with larger semantic sizes in a real-world web shopping scenario. The results show that multi-modal LLMs are more emotionally engaged in decision-making, but this also introduces potential biases, such as the risk of manipulation through clickbait headlines. Ultimately, this study offers a novel perspective on how LLMs interpret and internalize language, from the smallest concrete objects to the most profound abstract concepts like love. The insights gained not only improve our understanding of LLMs but also provide new avenues for exploring the cognitive abilities that define human intelligence. 

---
# Zero-Shot Defense Against Toxic Images via Inherent Multimodal Alignment in LVLMs 

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00037)  

**Abstract**: Large Vision-Language Models (LVLMs) have made significant strides in multimodal comprehension, thanks to extensive pre-training and fine-tuning on large-scale visual datasets. However, despite their robust textual safety mechanisms, they remain vulnerable to harmful visual inputs. Existing safeguards-typically relying on pre-filtering or fine-tuning-incur high costs and diminish overall utility. To address this critical vulnerability, we introduce SafeCLIP, a lightweight method that leverages LVLMs inherent multimodal alignment for zero-shot toxic image detection. By projecting CLIPs discarded CLS token into its text space and matching it with toxic descriptors, SafeCLIP detects harmful content without any architectural changes-adding minimal latency and enabling dynamic safety corrections during inference and this http URL show that SafeCLIP achieves a 66.9% defense success rate with only 3.2% false positive rate and 7.2% overhead. In contrast, state-of-the-art methods achieve 52.9% success but have a 10.7% false positive rate and 210% overhead. Our work demonstrates that leveraging inherent multimodal alignment can yield efficient, low-cost LVLM safety. Code is available at this http URL. 

---
# Evaluating Large Language Models on the Spanish Medical Intern Resident (MIR) Examination 2024/2025:A Comparative Analysis of Clinical Reasoning and Knowledge Application 

**Authors**: Carlos Luengo Vera, Ignacio Ferro Picon, M. Teresa del Val Nunez, Jose Andres Gomez Gandia, Antonio de Lucas Ancillo, Victor Ramos Arroyo, Carlos Milan Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00025)  

**Abstract**: This study presents a comparative evaluation of 22 large language models LLMs on the Spanish Medical Intern Resident MIR examinations for 2024 and 2025 with a focus on clinical reasoning domain specific expertise and multimodal processing capabilities The MIR exam consisting of 210 multiple choice questions some requiring image interpretation serves as a stringent benchmark for assessing both factual recall and complex clinical problem solving skills Our investigation encompasses general purpose models such as GPT4 Claude LLaMA and Gemini as well as specialized fine tuned systems like Miri Pro which leverages proprietary Spanish healthcare data to excel in medical contexts
Recent market entries Deepseek and Grok have further enriched the evaluation landscape particularly for tasks that demand advanced visual and semantic analysis The findings indicate that while general purpose LLMs perform robustly overall fine tuned models consistently achieve superior accuracy especially in addressing nuanced domain specific challenges A modest performance decline observed between the two exam cycles appears attributable to the implementation of modified questions designed to mitigate reliance on memorization
The results underscore the transformative potential of domain specific fine tuning and multimodal integration in advancing medical AI applications They also highlight critical implications for the future integration of LLMs into medical education training and clinical decision making emphasizing the importance of balancing automated reasoning with ethical and context aware judgment 

---
# Advancing vision-language models in front-end development via data synthesis 

**Authors**: Tong Ge, Yashu Liu, Jieping Ye, Tianyi Li, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01619)  

**Abstract**: Modern front-end (FE) development, especially when leveraging the unique features of frameworks like React and Vue, presents distinctive challenges. These include managing modular architectures, ensuring synchronization between data and visual outputs for declarative rendering, and adapting reusable components to various scenarios. Such complexities make it particularly difficult for state-of-the-art large vision-language models (VLMs) to generate accurate and functional code directly from design images. To address these challenges, we propose a reflective agentic workflow that synthesizes high-quality image-text data to capture the diverse characteristics of FE development. This workflow automates the extraction of self-contained\footnote{A \textbf{self-contained} code snippet is one that encapsulates all necessary logic, styling, and dependencies, ensuring it functions independently without requiring external imports or context.} code snippets from real-world projects, renders the corresponding visual outputs, and generates detailed descriptions that link design elements to functional code. To further expand the scope and utility of the synthesis, we introduce three data synthesis strategies: Evolution-based synthesis, which enables scalable and diverse dataset expansion; Waterfall-Model-based synthesis, which generates logically coherent code derived from system requirements; and Additive Development synthesis, which iteratively increases the complexity of human-authored components. We build a large vision-language model, Flame, trained on the synthesized datasets and demonstrate its effectiveness in generating React code via the $\text{pass}@k$ metric. Our results suggest that a code VLM trained to interpret images before code generation may achieve better performance. 

---
# Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG 

**Authors**: Wenbin Wang, Yongcheng Jing, Liang Ding, Yingjie Wang, Li Shen, Yong Luo, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01222)  

**Abstract**: High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To overcome the limitations of existing methods, this paper shifts away from prior dedicated heuristic approaches and revisits the most fundamental idea to HR perception by enhancing the long-context capability of MLLMs, driven by recent advances in long-context techniques like retrieval-augmented generation (RAG) for general LLMs. Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43% improvement on $V^*$ Bench and 19% on HR-Bench. 

---
# PreMind: Multi-Agent Video Understanding for Advanced Indexing of Presentation-style Videos 

**Authors**: Kangda Wei, Zhengyu Zhou, Bingqing Wang, Jun Araki, Lukas Lange, Ruihong Huang, Zhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00162)  

**Abstract**: In recent years, online lecture videos have become an increasingly popular resource for acquiring new knowledge. Systems capable of effectively understanding/indexing lecture videos are thus highly desirable, enabling downstream tasks like question answering to help users efficiently locate specific information within videos. This work proposes PreMind, a novel multi-agent multimodal framework that leverages various large models for advanced understanding/indexing of presentation-style videos. PreMind first segments videos into slide-presentation segments using a Vision-Language Model (VLM) to enhance modern shot-detection techniques. Each segment is then analyzed to generate multimodal indexes through three key steps: (1) extracting slide visual content, (2) transcribing speech narratives, and (3) consolidating these visual and speech contents into an integrated understanding. Three innovative mechanisms are also proposed to improve performance: leveraging prior lecture knowledge to refine visual understanding, detecting/correcting speech transcription errors using a VLM, and utilizing a critic agent for dynamic iterative self-reflection in vision analysis. Compared to traditional video indexing methods, PreMind captures rich, reliable multimodal information, allowing users to search for details like abbreviations shown only on slides. Systematic evaluations on the public LPM dataset and an internal enterprise dataset are conducted to validate PreMind's effectiveness, supported by detailed analyses. 

---
# Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions 

**Authors**: Jia Chen, Qian Dong, Haitao Li, Xiaohui He, Yan Gao, Shaosheng Cao, Yi Wu, Ping Yang, Chen Xu, Yao Hu, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00501)  

**Abstract**: User-generated content (UGC) communities, especially those featuring multimodal content, improve user experiences by integrating visual and textual information into results (or items). The challenge of improving user experiences in complex systems with search and recommendation (S\&R) services has drawn significant attention from both academia and industry these years. However, the lack of high-quality datasets has limited the research progress on multimodal S\&R. To address the growing need for developing better S\&R services, we present a novel multimodal information retrieval dataset in this paper, namely Qilin. The dataset is collected from Xiaohongshu, a popular social platform with over 300 million monthly active users and an average search penetration rate of over 70\%. In contrast to existing datasets, \textsf{Qilin} offers a comprehensive collection of user sessions with heterogeneous results like image-text notes, video notes, commercial notes, and direct answers, facilitating the development of advanced multimodal neural retrieval models across diverse task settings. To better model user satisfaction and support the analysis of heterogeneous user behaviors, we also collect extensive APP-level contextual signals and genuine user feedback. Notably, Qilin contains user-favored answers and their referred results for search requests triggering the Deep Query Answering (DQA) module. This allows not only the training \& evaluation of a Retrieval-augmented Generation (RAG) pipeline, but also the exploration of how such a module would affect users' search behavior. Through comprehensive analysis and experiments, we provide interesting findings and insights for further improving S\&R systems. We hope that \textsf{Qilin} will significantly contribute to the advancement of multimodal content platforms with S\&R services in the future. 

---
# I see what you mean: Co-Speech Gestures for Reference Resolution in Multimodal Dialogue 

**Authors**: Esam Ghaleb, Bulat Khaertdinov, Aslı Özyürek, Raquel Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2503.00071)  

**Abstract**: In face-to-face interaction, we use multiple modalities, including speech and gestures, to communicate information and resolve references to objects. However, how representational co-speech gestures refer to objects remains understudied from a computational perspective. In this work, we address this gap by introducing a multimodal reference resolution task centred on representational gestures, while simultaneously tackling the challenge of learning robust gesture embeddings. We propose a self-supervised pre-training approach to gesture representation learning that grounds body movements in spoken language. Our experiments show that the learned embeddings align with expert annotations and have significant predictive power. Moreover, reference resolution accuracy further improves when (1) using multimodal gesture representations, even when speech is unavailable at inference time, and (2) leveraging dialogue history. Overall, our findings highlight the complementary roles of gesture and speech in reference resolution, offering a step towards more naturalistic models of human-machine interaction. 

---
# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning 

**Authors**: Nilay Yilmaz, Maitreya Patel, Yiran Lawrence Luo, Tejas Gokhale, Chitta Baral, Suren Jayasuriya, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00043)  

**Abstract**: Multimodal Large Language Models (MLLMs) have become a powerful tool for integrating visual and textual information. Despite their exceptional performance on visual understanding benchmarks, measuring their ability to reason abstractly across multiple images remains a significant challenge. To address this, we introduce VOILA, a large-scale, open-ended, dynamic benchmark designed to evaluate MLLMs' perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices. Our experiments demonstrate that the analogical reasoning tasks in VOILA present a challenge to MLLMs. Through multi-step analysis, we reveal that current MLLMs struggle to comprehend inter-image relationships and exhibit limited capabilities in high-level relational reasoning. Notably, we observe that performance improves when following a multi-step strategy of least-to-most prompting. Comprehensive evaluations on open-source models and GPT-4o show that on text-based answers, the best accuracy for challenging scenarios is 13% (LLaMa 3.2) and even for simpler tasks is only 29% (GPT-4o), while human performance is significantly higher at 70% across both difficulty levels. 

---
# PinLanding: Content-First Keyword Landing Page Generation via Multi-Modal AI for Web-Scale Discovery 

**Authors**: Faye Zhang, Jasmine Wan, Qianyu Cheng, Jinfeng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00619)  

**Abstract**: Online platforms like Pinterest hosting vast content collections traditionally rely on manual curation or user-generated search logs to create keyword landing pages (KLPs) -- topic-centered collection pages that serve as entry points for content discovery. While manual curation ensures quality, it doesn't scale to millions of collections, and search log approaches result in limited topic coverage and imprecise content matching. In this paper, we present PinLanding, a novel content-first architecture that transforms the way platforms create topical collections. Instead of deriving topics from user behavior, our system employs a multi-stage pipeline combining vision-language model (VLM) for attribute extraction, large language model (LLM) for topic generation, and a CLIP-based dual-encoder architecture for precise content matching. Our model achieves 99.7% Recall@10 on Fashion200K benchmark, demonstrating strong attribute understanding capabilities. In production deployment for search engine optimization with 4.2 million shopping landing pages, the system achieves a 4X increase in topic coverage and 14.29% improvement in collection attribute precision over the traditional search log-based approach via human evaluation. The architecture can be generalized beyond search traffic to power various user experiences, including content discovery and recommendations, providing a scalable solution to transform unstructured content into curated topical collections across any content domain. 

---
# Composed Multi-modal Retrieval: A Survey of Approaches and Applications 

**Authors**: Kun Zhang, Jingyu Li, Zhe Li, Jingjing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01334)  

**Abstract**: With the rapid growth of multi-modal data from social media, short video platforms, and e-commerce, content-based retrieval has become essential for efficiently searching and utilizing heterogeneous information. Over time, retrieval techniques have evolved from Unimodal Retrieval (UR) to Cross-modal Retrieval (CR) and, more recently, to Composed Multi-modal Retrieval (CMR). CMR enables users to retrieve images or videos by integrating a reference visual input with textual modifications, enhancing search flexibility and precision. This paper provides a comprehensive review of CMR, covering its fundamental challenges, technical advancements, and categorization into supervised, zero-shot, and semi-supervised learning paradigms. We discuss key research directions, including data augmentation, model architecture, and loss optimization in supervised CMR, as well as transformation frameworks and external knowledge integration in zero-shot CMR. Additionally, we highlight the application potential of CMR in composed image retrieval, video retrieval, and person retrieval, which have significant implications for e-commerce, online search, and public security. Given its ability to refine and personalize search experiences, CMR is poised to become a pivotal technology in next-generation retrieval systems. A curated list of related works and resources is available at: this https URL 

---
# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models 

**Authors**: Cansu Sancaktar, Christian Gumbsch, Andrii Zadaianchuk, Pavel Kolev, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.01584)  

**Abstract**: Exploration is a cornerstone of reinforcement learning (RL). Intrinsic motivation attempts to decouple exploration from external, task-based rewards. However, established approaches to intrinsic motivation that follow general principles such as information gain, often only uncover low-level interactions. In contrast, children's play suggests that they engage in meaningful high-level behavior by imitating or interacting with their caregivers. Recent work has focused on using foundation models to inject these semantic biases into exploration. However, these methods often rely on unrealistic assumptions, such as language-embedded environments or access to high-level actions. We propose SEmaNtically Sensible ExploratIon (SENSEI), a framework to equip model-based RL agents with an intrinsic motivation for semantically meaningful behavior. SENSEI distills a reward signal of interestingness from Vision Language Model (VLM) annotations, enabling an agent to predict these rewards through a world model. Using model-based RL, SENSEI trains an exploration policy that jointly maximizes semantic rewards and uncertainty. We show that in both robotic and video game-like simulations SENSEI discovers a variety of meaningful behaviors from image observations and low-level actions. SENSEI provides a general tool for learning from foundation model feedback, a crucial research direction, as VLMs become more powerful. 

---
# FAIR: Facilitating Artificial Intelligence Resilience in Manufacturing Industrial Internet 

**Authors**: Yingyan Zeng, Ismini Lourentzou, Xinwei Deng, Ran Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01086)  

**Abstract**: Artificial intelligence (AI) systems have been increasingly adopted in the Manufacturing Industrial Internet (MII). Investigating and enabling the AI resilience is very important to alleviate profound impact of AI system failures in manufacturing and Industrial Internet of Things (IIoT) operations, leading to critical decision making. However, there is a wide knowledge gap in defining the resilience of AI systems and analyzing potential root causes and corresponding mitigation strategies. In this work, we propose a novel framework for investigating the resilience of AI performance over time under hazard factors in data quality, AI pipelines, and the cyber-physical layer. The proposed method can facilitate effective diagnosis and mitigation strategies to recover AI performance based on a multimodal multi-head self latent attention model. The merits of the proposed method are elaborated using an MII testbed of connected Aerosol Jet Printing (AJP) machines, fog nodes, and Cloud with inference tasks via AI pipelines. 

---
# Distilled Prompt Learning for Incomplete Multimodal Survival Prediction 

**Authors**: Yingxue Xu, Fengtao Zhou, Chenyu Zhao, Yihui Wang, Can Yang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01653)  

**Abstract**: The integration of multimodal data including pathology images and gene profiles is widely applied in precise survival prediction. Despite recent advances in multimodal survival models, collecting complete modalities for multimodal fusion still poses a significant challenge, hindering their application in clinical settings. Current approaches tackling incomplete modalities often fall short, as they typically compensate for only a limited part of the knowledge of missing modalities. To address this issue, we propose a Distilled Prompt Learning framework (DisPro) to utilize the strong robustness of Large Language Models (LLMs) to missing modalities, which employs two-stage prompting for compensation of comprehensive information for missing modalities. In the first stage, Unimodal Prompting (UniPro) distills the knowledge distribution of each modality, preparing for supplementing modality-specific knowledge of the missing modality in the subsequent stage. In the second stage, Multimodal Prompting (MultiPro) leverages available modalities as prompts for LLMs to infer the missing modality, which provides modality-common information. Simultaneously, the unimodal knowledge acquired in the first stage is injected into multimodal inference to compensate for the modality-specific knowledge of the missing modality. Extensive experiments covering various missing scenarios demonstrated the superiority of the proposed method. The code is available at this https URL. 

---
# MINT: Multi-modal Chain of Thought in Unified Generative Models for Enhanced Image Generation 

**Authors**: Yi Wang, Mushui Liu, Wanggui He, Longxiang Zhang, Ziwei Huang, Guanghao Zhang, Fangxun Shu, Zhong Tao, Dong She, Zhelun Yu, Haoyuan Li, Weilong Dai, Mingli Song, Jie Song, Hao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01298)  

**Abstract**: Unified generative models have demonstrated extraordinary performance in both text and image generation. However, they tend to underperform when generating intricate images with various interwoven conditions, which is hard to solely rely on straightforward text-to-image generation. In response to this challenge, we introduce MINT, an innovative unified generative model, empowered with native multimodal chain of thought (MCoT) for enhanced image generation for the first time. Firstly, we design Mixture of Transformer Experts (MTXpert), an expert-parallel structure that effectively supports both natural language generation (NLG) and visual capabilities, while avoiding potential modality conflicts that could hinder the full potential of each modality. Building on this, we propose an innovative MCoT training paradigm, a step-by-step approach to multimodal thinking, reasoning, and reflection specifically designed to enhance image generation. This paradigm equips MINT with nuanced, element-wise decoupled alignment and a comprehensive understanding of textual and visual components. Furthermore, it fosters advanced multimodal reasoning and self-reflection, enabling the construction of images that are firmly grounded in the logical relationships between these elements. Notably, MINT has been validated to exhibit superior performance across multiple benchmarks for text-to-image (T2I) and image-to-text (I2T) tasks. 

---
# MedUnifier: Unifying Vision-and-Language Pre-training on Medical Data with Vision Generation Task using Discrete Visual Representations 

**Authors**: Ziyang Zhang, Yang Yu, Yucheng Chen, Xulei Yang, Si Yong Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01019)  

**Abstract**: Despite significant progress in Vision-Language Pre-training (VLP), current approaches predominantly emphasize feature extraction and cross-modal comprehension, with limited attention to generating or transforming visual content. This gap hinders the model's ability to synthesize coherent and novel visual representations from textual prompts, thereby reducing the effectiveness of multi-modal learning. In this work, we propose MedUnifier, a unified VLP framework tailored for medical data. MedUnifier seamlessly integrates text-grounded image generation capabilities with multi-modal learning strategies, including image-text contrastive alignment, image-text matching and image-grounded text generation. Unlike traditional methods that reply on continuous visual representations, our approach employs visual vector quantization, which not only facilitates a more cohesive learning strategy for cross-modal understanding but also enhances multi-modal generation quality by effectively leveraging discrete representations. Our framework's effectiveness is evidenced by the experiments on established benchmarks, including uni-modal tasks (supervised fine-tuning), cross-modal tasks (image-text retrieval and zero-shot image classification), and multi-modal tasks (medical report generation, image synthesis), where it achieves state-of-the-art performance across various tasks. MedUnifier also offers a highly adaptable tool for a wide range of language and vision tasks in healthcare, marking advancement toward the development of a generalizable AI model for medical applications. 

---
# LLM-Fusion: A Novel Multimodal Fusion Model for Accelerated Material Discovery 

**Authors**: Onur Boyar, Indra Priyadarsini, Seiji Takeda, Lisa Hamada  

**Link**: [PDF](https://arxiv.org/pdf/2503.01022)  

**Abstract**: Discovering materials with desirable properties in an efficient way remains a significant problem in materials science. Many studies have tackled this problem by using different sets of information available about the materials. Among them, multimodal approaches have been found to be promising because of their ability to combine different sources of information. However, fusion algorithms to date remain simple, lacking a mechanism to provide a rich representation of multiple modalities. This paper presents LLM-Fusion, a novel multimodal fusion model that leverages large language models (LLMs) to integrate diverse representations, such as SMILES, SELFIES, text descriptions, and molecular fingerprints, for accurate property prediction. Our approach introduces a flexible LLM-based architecture that supports multimodal input processing and enables material property prediction with higher accuracy than traditional methods. We validate our model on two datasets across five prediction tasks and demonstrate its effectiveness compared to unimodal and naive concatenation baselines. 

---
# Multimodal Distillation-Driven Ensemble Learning for Long-Tailed Histopathology Whole Slide Images Analysis 

**Authors**: Xitong Ling, Yifeng Ping, Jiawen Li, Jing Peng, Yuxuan Chen, Minxi Ouyang, Yizhi Wang, Yonghong He, Tian Guan, Xiaoping Liu, Lianghui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00915)  

**Abstract**: Multiple Instance Learning (MIL) plays a significant role in computational pathology, enabling weakly supervised analysis of Whole Slide Image (WSI) datasets. The field of WSI analysis is confronted with a severe long-tailed distribution problem, which significantly impacts the performance of classifiers. Long-tailed distributions lead to class imbalance, where some classes have sparse samples while others are abundant, making it difficult for classifiers to accurately identify minority class samples. To address this issue, we propose an ensemble learning method based on MIL, which employs expert decoders with shared aggregators and consistency constraints to learn diverse distributions and reduce the impact of class imbalance on classifier performance. Moreover, we introduce a multimodal distillation framework that leverages text encoders pre-trained on pathology-text pairs to distill knowledge and guide the MIL aggregator in capturing stronger semantic features relevant to class information. To ensure flexibility, we use learnable prompts to guide the distillation process of the pre-trained text encoder, avoiding limitations imposed by specific prompts. Our method, MDE-MIL, integrates multiple expert branches focusing on specific data distributions to address long-tailed issues. Consistency control ensures generalization across classes. Multimodal distillation enhances feature extraction. Experiments on Camelyon+-LT and PANDA-LT datasets show it outperforms state-of-the-art methods. 

---
# Cross Modality Medical Image Synthesis for Improving Liver Segmentation 

**Authors**: Muhammad Rafiq, Hazrat Ali, Ghulam Mujtaba, Zubair Shah, Shoaib Azmat  

**Link**: [PDF](https://arxiv.org/pdf/2503.00945)  

**Abstract**: Deep learning-based computer-aided diagnosis (CAD) of medical images requires large datasets. However, the lack of large publicly available labeled datasets limits the development of deep learning-based CAD systems. Generative Adversarial Networks (GANs), in particular, CycleGAN, can be used to generate new cross-domain images without paired training data. However, most CycleGAN-based synthesis methods lack the potential to overcome alignment and asymmetry between the input and generated data. We propose a two-stage technique for the synthesis of abdominal MRI using cross-modality translation of abdominal CT. We show that the synthetic data can help improve the performance of the liver segmentation network. We increase the number of abdominal MRI images through cross-modality image transformation of unpaired CT images using a CycleGAN inspired deformation invariant network called EssNet. Subsequently, we combine the synthetic MRI images with the original MRI images and use them to improve the accuracy of the U-Net on a liver segmentation task. We train the U-Net on real MRI images and then on real and synthetic MRI images. Consequently, by comparing both scenarios, we achieve an improvement in the performance of U-Net. In summary, the improvement achieved in the Intersection over Union (IoU) is 1.17%. The results show potential to address the data scarcity challenge in medical imaging. 

---
# DELST: Dual Entailment Learning for Hyperbolic Image-Gene Pretraining in Spatial Transcriptomics 

**Authors**: Xulin Chen, Junzhou Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00804)  

**Abstract**: Spatial transcriptomics (ST) maps gene expression within tissue at individual spots, making it a valuable resource for multimodal representation learning. Additionally, ST inherently contains rich hierarchical information both across and within modalities. For instance, different spots exhibit varying numbers of nonzero gene expressions, corresponding to different levels of cellular activity and semantic hierarchies. However, existing methods rely on contrastive alignment of image-gene pairs, failing to accurately capture the intricate hierarchical relationships in ST data. Here, we propose DELST, the first framework to embed hyperbolic representations while modeling hierarchy for image-gene pretraining at two levels: (1) Cross-modal entailment learning, which establishes an order relationship between genes and images to enhance image representation generalization; (2) Intra-modal entailment learning, which encodes gene expression patterns as hierarchical relationships, guiding hierarchical learning across different samples at a global scale and integrating biological insights into single-modal representations. Extensive experiments on ST benchmarks annotated by pathologists demonstrate the effectiveness of our framework, achieving improved predictive performance compared to existing methods. Our code and models are available at: this https URL. 

---
# CLEA: Closed-Loop Embodied Agent for Enhancing Task Execution in Dynamic Environments 

**Authors**: Mingcong Lei, Ge Wang, Yiming Zhao, Zhixin Mai, Qing Zhao, Yao Guo, Zhen Li, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.00729)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in the hierarchical decomposition of complex tasks through semantic reasoning. However, their application in embodied systems faces challenges in ensuring reliable execution of subtask sequences and achieving one-shot success in long-term task completion. To address these limitations in dynamic environments, we propose Closed-Loop Embodied Agent (CLEA) -- a novel architecture incorporating four specialized open-source LLMs with functional decoupling for closed-loop task management. The framework features two core innovations: (1) Interactive task planner that dynamically generates executable subtasks based on the environmental memory, and (2) Multimodal execution critic employing an evaluation framework to conduct a probabilistic assessment of action feasibility, triggering hierarchical re-planning mechanisms when environmental perturbations exceed preset thresholds. To validate CLEA's effectiveness, we conduct experiments in a real environment with manipulable objects, using two heterogeneous robots for object search, manipulation, and search-manipulation integration tasks. Across 12 task trials, CLEA outperforms the baseline model, achieving a 67.3% improvement in success rate and a 52.8% increase in task completion rate. These results demonstrate that CLEA significantly enhances the robustness of task planning and execution in dynamic environments. 

---
# Enhancing Monocular 3D Scene Completion with Diffusion Model 

**Authors**: Changlin Song, Jiaqi Wang, Liyun Zhu, He Weng  

**Link**: [PDF](https://arxiv.org/pdf/2503.00726)  

**Abstract**: 3D scene reconstruction is essential for applications in virtual reality, robotics, and autonomous driving, enabling machines to understand and interact with complex environments. Traditional 3D Gaussian Splatting techniques rely on images captured from multiple viewpoints to achieve optimal performance, but this dependence limits their use in scenarios where only a single image is available. In this work, we introduce FlashDreamer, a novel approach for reconstructing a complete 3D scene from a single image, significantly reducing the need for multi-view inputs. Our approach leverages a pre-trained vision-language model to generate descriptive prompts for the scene, guiding a diffusion model to produce images from various perspectives, which are then fused to form a cohesive 3D reconstruction. Extensive experiments show that our method effectively and robustly expands single-image inputs into a comprehensive 3D scene, extending monocular 3D reconstruction capabilities without further training. Our code is available this https URL. 

---
# Urban Safety Perception Through the Lens of Large Multimodal Models: A Persona-based Approach 

**Authors**: Ciro Beneduce, Bruno Lepri, Massimiliano Luca  

**Link**: [PDF](https://arxiv.org/pdf/2503.00610)  

**Abstract**: Understanding how urban environments are perceived in terms of safety is crucial for urban planning and policymaking. Traditional methods like surveys are limited by high cost, required time, and scalability issues. To overcome these challenges, this study introduces Large Multimodal Models (LMMs), specifically Llava 1.6 7B, as a novel approach to assess safety perceptions of urban spaces using street-view images. In addition, the research investigated how this task is affected by different socio-demographic perspectives, simulated by the model through Persona-based prompts. Without additional fine-tuning, the model achieved an average F1-score of 59.21% in classifying urban scenarios as safe or unsafe, identifying three key drivers of perceived unsafety: isolation, physical decay, and urban infrastructural challenges. Moreover, incorporating Persona-based prompts revealed significant variations in safety perceptions across the socio-demographic groups of age, gender, and nationality. Elder and female Personas consistently perceive higher levels of unsafety than younger or male Personas. Similarly, nationality-specific differences were evident in the proportion of unsafe classifications ranging from 19.71% in Singapore to 40.15% in Botswana. Notably, the model's default configuration aligned most closely with a middle-aged, male Persona. These findings highlight the potential of LMMs as a scalable and cost-effective alternative to traditional methods for urban safety perceptions. While the sensitivity of these models to socio-demographic factors underscores the need for thoughtful deployment, their ability to provide nuanced perspectives makes them a promising tool for AI-driven urban planning. 

---
# Bridging Spectral-wise and Multi-spectral Depth Estimation via Geometry-guided Contrastive Learning 

**Authors**: Ukcheol Shin, Kyunghyun Lee, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00793)  

**Abstract**: Deploying depth estimation networks in the real world requires high-level robustness against various adverse conditions to ensure safe and reliable autonomy. For this purpose, many autonomous vehicles employ multi-modal sensor systems, including an RGB camera, NIR camera, thermal camera, LiDAR, or Radar. They mainly adopt two strategies to use multiple sensors: modality-wise and multi-modal fused inference. The former method is flexible but memory-inefficient, unreliable, and vulnerable. Multi-modal fusion can provide high-level reliability, yet it needs a specialized architecture. In this paper, we propose an effective solution, named align-and-fuse strategy, for the depth estimation from multi-spectral images. In the align stage, we align embedding spaces between multiple spectrum bands to learn shareable representation across multi-spectral images by minimizing contrastive loss of global and spatially aligned local features with geometry cue. After that, in the fuse stage, we train an attachable feature fusion module that can selectively aggregate the multi-spectral features for reliable and robust prediction results. Based on the proposed method, a single-depth network can achieve both spectral-invariant and multi-spectral fused depth estimation while preserving reliability, memory efficiency, and flexibility. 

---
# MIRROR: Multi-Modal Pathological Self-Supervised Representation Learning via Modality Alignment and Retention 

**Authors**: Tianyi Wang, Jianan Fan, Dingxin Zhang, Dongnan Liu, Yong Xia, Heng Huang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.00374)  

**Abstract**: Histopathology and transcriptomics are fundamental modalities in oncology, encapsulating the morphological and molecular aspects of the disease. Multi-modal self-supervised learning has demonstrated remarkable potential in learning pathological representations by integrating diverse data sources. Conventional multi-modal integration methods primarily emphasize modality alignment, while paying insufficient attention to retaining the modality-specific structures. However, unlike conventional scenarios where multi-modal inputs share highly overlapping features, histopathology and transcriptomics exhibit pronounced heterogeneity, offering orthogonal yet complementary insights. Histopathology provides morphological and spatial context, elucidating tissue architecture and cellular topology, whereas transcriptomics delineates molecular signatures through gene expression patterns. This inherent disparity introduces a major challenge in aligning them while maintaining modality-specific fidelity. To address these challenges, we present MIRROR, a novel multi-modal representation learning method designed to foster both modality alignment and retention. MIRROR employs dedicated encoders to extract comprehensive features for each modality, which is further complemented by a modality alignment module to achieve seamless integration between phenotype patterns and molecular profiles. Furthermore, a modality retention module safeguards unique attributes from each modality, while a style clustering module mitigates redundancy and enhances disease-relevant information by modeling and aligning consistent pathological signatures within a clustering space. Extensive evaluations on TCGA cohorts for cancer subtyping and survival analysis highlight MIRROR's superior performance, demonstrating its effectiveness in constructing comprehensive oncological feature representations and benefiting the cancer diagnosis. 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

---
# Foundation-Model-Boosted Multimodal Learning for fMRI-based Neuropathic Pain Drug Response Prediction 

**Authors**: Wenrui Fan, L. M. Riza Rizky, Jiayang Zhang, Chen Chen, Haiping Lu, Kevin Teh, Dinesh Selvarajah, Shuo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00210)  

**Abstract**: Neuropathic pain, affecting up to 10% of adults, remains difficult to treat due to limited therapeutic efficacy and tolerability. Although resting-state functional MRI (rs-fMRI) is a promising non-invasive measurement of brain biomarkers to predict drug response in therapeutic development, the complexity of fMRI demands machine learning models with substantial capacity. However, extreme data scarcity in neuropathic pain research limits the application of high-capacity models. To address the challenge of data scarcity, we propose FMM$_{TC}$, a Foundation-Model-boosted Multimodal learning framework for fMRI-based neuropathic pain drug response prediction, which leverages both internal multimodal information in pain-specific data and external knowledge from large pain-agnostic data. Specifically, to maximize the value of limited pain-specific data, FMM$_{TC}$ integrates complementary information from two rs-fMRI modalities: Time series and functional Connectivity. FMM$_{TC}$ is further boosted by an fMRI foundation model with its external knowledge from extensive pain-agnostic fMRI datasets enriching limited pain-specific information. Evaluations with an in-house dataset and a public dataset from OpenNeuro demonstrate FMM$_{TC}$'s superior representation ability, generalizability, and cross-dataset adaptability over existing unimodal fMRI models that only consider one of the rs-fMRI modalities. The ablation study validates the effectiveness of multimodal learning and foundation-model-powered external knowledge transfer in FMM$_{TC}$. An integrated gradient-based interpretation study explains how FMM$_{TC}$'s cross-dataset dynamic behaviors enhance its adaptability. In conclusion, FMM$_{TC}$ boosts clinical trials in neuropathic pain therapeutic development by accurately predicting drug responses to improve the participant stratification efficiency. 

---
# PaliGemma-CXR: A Multi-task Multimodal Model for TB Chest X-ray Interpretation 

**Authors**: Denis Musinguzi, Andrew Katumba, Sudi Murindanyi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00171)  

**Abstract**: Tuberculosis (TB) is a infectious global health challenge. Chest X-rays are a standard method for TB screening, yet many countries face a critical shortage of radiologists capable of interpreting these images. Machine learning offers an alternative, as it can automate tasks such as disease diagnosis, and report generation. However, traditional approaches rely on task-specific models, which cannot utilize the interdependence between tasks. Building a multi-task model capable of performing multiple tasks poses additional challenges such as scarcity of multimodal data, dataset imbalance, and negative transfer. To address these challenges, we propose PaliGemma-CXR, a multi-task multimodal model capable of performing TB diagnosis, object detection, segmentation, report generation, and VQA. Starting with a dataset of chest X-ray images annotated with TB diagnosis labels and segmentation masks, we curated a multimodal dataset to support additional tasks. By finetuning PaliGemma on this dataset and sampling data using ratios of the inverse of the size of task datasets, we achieved the following results across all tasks: 90.32% accuracy on TB diagnosis and 98.95% on close-ended VQA, 41.3 BLEU score on report generation, and a mAP of 19.4 and 16.0 on object detection and segmentation, respectively. These results demonstrate that PaliGemma-CXR effectively leverages the interdependence between multiple image interpretation tasks to enhance performance. 

---
# Deciphering the complaint aspects: Towards an aspect-based complaint identification model with video complaint dataset in finance 

**Authors**: Sarmistha Das, Basha Mujavarsheik, R E Zera Lyngkhoi, Sriparna Saha, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2503.00054)  

**Abstract**: In today's competitive marketing landscape, effective complaint management is crucial for customer service and business success. Video complaints, integrating text and image content, offer invaluable insights by addressing customer grievances and delineating product benefits and drawbacks. However, comprehending nuanced complaint aspects within vast daily multimodal financial data remains a formidable challenge. Addressing this gap, we have curated a proprietary multimodal video complaint dataset comprising 433 publicly accessible instances. Each instance is meticulously annotated at the utterance level, encompassing five distinct categories of financial aspects and their associated complaint labels. To support this endeavour, we introduce Solution 3.0, a model designed for multimodal aspect-based complaint identification task. Solution 3.0 is tailored to perform three key tasks: 1) handling multimodal features ( audio and video), 2) facilitating multilabel aspect classification, and 3) conducting multitasking for aspect classifications and complaint identification parallelly. Solution 3.0 utilizes a CLIP-based dual frozen encoder with an integrated image segment encoder for global feature fusion, enhanced by contextual attention (ISEC) to improve accuracy and efficiency. Our proposed framework surpasses current multimodal baselines, exhibiting superior performance across nearly all metrics by opening new ways to strengthen appropriate customer care initiatives and effectively assisting individuals in resolving their problems. 

---
# Omni-SILA: Towards Omni-scene Driven Visual Sentiment Identifying, Locating and Attributing in Videos 

**Authors**: Jiamin Luo, Jingjing Wang, Junxiao Ma, Yujie Jin, Shoushan Li, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00049)  

**Abstract**: Prior studies on Visual Sentiment Understanding (VSU) primarily rely on the explicit scene information (e.g., facial expression) to judge visual sentiments, which largely ignore implicit scene information (e.g., human action, objection relation and visual background), while such information is critical for precisely discovering visual sentiments. Motivated by this, this paper proposes a new Omni-scene driven visual Sentiment Identifying, Locating and Attributing in videos (Omni-SILA) task, aiming to interactively and precisely identify, locate and attribute visual sentiments through both explicit and implicit scene information. Furthermore, this paper believes that this Omni-SILA task faces two key challenges: modeling scene and highlighting implicit scene beyond explicit. To this end, this paper proposes an Implicit-enhanced Causal MoE (ICM) approach for addressing the Omni-SILA task. Specifically, a Scene-Balanced MoE (SBM) and an Implicit-Enhanced Causal (IEC) blocks are tailored to model scene information and highlight the implicit scene information beyond explicit, respectively. Extensive experimental results on our constructed explicit and implicit Omni-SILA datasets demonstrate the great advantage of the proposed ICM approach over advanced Video-LLMs. 

---
