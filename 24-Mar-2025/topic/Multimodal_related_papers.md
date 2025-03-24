# CLIP-PING: Boosting Lightweight Vision-Language Models with Proximus Intrinsic Neighbors Guidance 

**Authors**: Chu Myaet Thwal, Ye Lin Tun, Minh N. H. Nguyen, Eui-Nam Huh, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2412.03871)  

**Abstract**: Beyond the success of Contrastive Language-Image Pre-training (CLIP), recent trends mark a shift toward exploring the applicability of lightweight vision-language models for resource-constrained scenarios. These models often deliver suboptimal performance when relying solely on a single image-text contrastive learning objective, spotlighting the need for more effective training mechanisms that guarantee robust cross-modal feature alignment. In this work, we propose CLIP-PING: Contrastive Language-Image Pre-training with Proximus Intrinsic Neighbors Guidance, a novel yet simple and efficient training paradigm designed to boost the performance of lightweight vision-language models with minimal computational overhead and lower data demands. CLIP-PING bootstraps unimodal features extracted from arbitrary pre-trained encoders to obtain intrinsic guidance of proximus neighbor samples, i.e., nearest-neighbor (NN) and cross nearest-neighbor (XNN). We find that extra contrastive supervision from these neighbors substantially boosts cross-modal alignment, enabling lightweight models to learn more generic features with rich semantic diversity. Extensive experiments reveal that CLIP-PING notably surpasses its peers in zero-shot generalization and cross-modal retrieval tasks. Specifically, a 5.5% gain on zero-shot ImageNet1K classification with 10.7% (I2T) and 5.7% (T2I) on Flickr30K retrieval, compared to the original CLIP when using ViT-XS image encoder trained on 3 million (image, text) pairs. Moreover, CLIP-PING showcases a strong transferability under the linear evaluation protocol across several downstream tasks. 

---
# Towards Agentic Recommender Systems in the Era of Multimodal Large Language Models 

**Authors**: Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang, Ryan A. Rossi, Branislav Kveton, Dongruo Zhou, Julian McAuley, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.16734)  

**Abstract**: Recent breakthroughs in Large Language Models (LLMs) have led to the emergence of agentic AI systems that extend beyond the capabilities of standalone models. By empowering LLMs to perceive external environments, integrate multimodal information, and interact with various tools, these agentic systems exhibit greater autonomy and adaptability across complex tasks. This evolution brings new opportunities to recommender systems (RS): LLM-based Agentic RS (LLM-ARS) can offer more interactive, context-aware, and proactive recommendations, potentially reshaping the user experience and broadening the application scope of RS. Despite promising early results, fundamental challenges remain, including how to effectively incorporate external knowledge, balance autonomy with controllability, and evaluate performance in dynamic, multimodal settings. In this perspective paper, we first present a systematic analysis of LLM-ARS: (1) clarifying core concepts and architectures; (2) highlighting how agentic capabilities -- such as planning, memory, and multimodal reasoning -- can enhance recommendation quality; and (3) outlining key research questions in areas such as safety, efficiency, and lifelong personalization. We also discuss open problems and future directions, arguing that LLM-ARS will drive the next wave of RS innovation. Ultimately, we foresee a paradigm shift toward intelligent, autonomous, and collaborative recommendation experiences that more closely align with users' evolving needs and complex decision-making processes. 

---
# The CASTLE 2024 Dataset: Advancing the Art of Multimodal Understanding 

**Authors**: Luca Rossetto, Werner Bailer, Duc-Tien Dang-Nguyen, Graham Healy, Björn Þór Jónsson, Onanong Kongmeesub, Hoang-Bao Le, Stevan Rudinac, Klaus Schöffmann, Florian Spiess, Allie Tran, Minh-Triet Tran, Quang-Linh Tran, Cathal Gurrin  

**Link**: [PDF](https://arxiv.org/pdf/2503.17116)  

**Abstract**: Egocentric video has seen increased interest in recent years, as it is used in a range of areas. However, most existing datasets are limited to a single perspective. In this paper, we present the CASTLE 2024 dataset, a multimodal collection containing ego- and exo-centric (i.e., first- and third-person perspective) video and audio from 15 time-aligned sources, as well as other sensor streams and auxiliary data. The dataset was recorded by volunteer participants over four days in a fixed location and includes the point of view of 10 participants, with an additional 5 fixed cameras providing an exocentric perspective. The entire dataset contains over 600 hours of UHD video recorded at 50 frames per second. In contrast to other datasets, CASTLE 2024 does not contain any partial censoring, such as blurred faces or distorted audio. The dataset is available via this https URL. 

---
# When Tom Eats Kimchi: Evaluating Cultural Bias of Multimodal Large Language Models in Cultural Mixture Contexts 

**Authors**: Jun Seong Kim, Kyaw Ye Thu, Javad Ismayilzada, Junyeong Park, Eunsu Kim, Huzama Ahmad, Na Min An, James Thorne, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2503.16826)  

**Abstract**: In a highly globalized world, it is important for multi-modal large language models (MLLMs) to recognize and respond correctly to mixed-cultural inputs. For example, a model should correctly identify kimchi (Korean food) in an image both when an Asian woman is eating it, as well as an African man is eating it. However, current MLLMs show an over-reliance on the visual features of the person, leading to misclassification of the entities. To examine the robustness of MLLMs to different ethnicity, we introduce MixCuBe, a cross-cultural bias benchmark, and study elements from five countries and four ethnicities. Our findings reveal that MLLMs achieve both higher accuracy and lower sensitivity to such perturbation for high-resource cultures, but not for low-resource cultures. GPT-4o, the best-performing model overall, shows up to 58% difference in accuracy between the original and perturbed cultural settings in low-resource cultures. Our dataset is publicly available at: this https URL. 

---
# When Words Outperform Vision: VLMs Can Self-Improve Via Text-Only Training For Human-Centered Decision Making 

**Authors**: Zhe Hu, Jing Li, Yu Yin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16965)  

**Abstract**: Embodied decision-making is fundamental for AI agents operating in real-world environments. While Visual Language Models (VLMs) have advanced this capability, they still struggle with complex decisions, particularly in human-centered situations that require deep reasoning about human needs and values. In this study, we systematically evaluate open-sourced VLMs on multimodal human-centered decision-making tasks. We find that LLMs receiving only textual descriptions unexpectedly outperform their VLM counterparts of similar scale that process actual images, suggesting that visual alignment may hinder VLM abilities. To address this challenge, we propose a novel text-only training approach with synthesized textual data. This method strengthens VLMs' language components and transfers the learned abilities to multimodal inference, eliminating the need for expensive image-text paired data. Furthermore, we show that VLMs can achieve substantial performance gains through self-improvement, using training data generated by their LLM counterparts rather than relying on larger teacher models like GPT-4. Our findings establish a more efficient and scalable approach to enhancing VLMs' human-centered decision-making capabilities, opening new avenues for optimizing VLMs through self-improvement mechanisms. 

---
# Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions 

**Authors**: Hadi Amini, Md Jueal Mia, Yasaman Saadati, Ahmed Imteaj, Seyedsina Nabavirazavi, Urmish Thakker, Md Zarif Hossain, Awal Ahmed Fime, S.S. Iyengar  

**Link**: [PDF](https://arxiv.org/pdf/2503.16585)  

**Abstract**: Language models (LMs) are machine learning models designed to predict linguistic patterns by estimating the probability of word sequences based on large-scale datasets, such as text. LMs have a wide range of applications in natural language processing (NLP) tasks, including autocomplete and machine translation. Although larger datasets typically enhance LM performance, scalability remains a challenge due to constraints in computational power and resources. Distributed computing strategies offer essential solutions for improving scalability and managing the growing computational demand. Further, the use of sensitive datasets in training and deployment raises significant privacy concerns. Recent research has focused on developing decentralized techniques to enable distributed training and inference while utilizing diverse computational resources and enabling edge AI. This paper presents a survey on distributed solutions for various LMs, including large language models (LLMs), vision language models (VLMs), multimodal LLMs (MLLMs), and small language models (SLMs). While LLMs focus on processing and generating text, MLLMs are designed to handle multiple modalities of data (e.g., text, images, and audio) and to integrate them for broader applications. To this end, this paper reviews key advancements across the MLLM pipeline, including distributed training, inference, fine-tuning, and deployment, while also identifying the contributions, limitations, and future areas of improvement. Further, it categorizes the literature based on six primary focus areas of decentralization. Our analysis describes gaps in current methodologies for enabling distributed solutions for LMs and outline future research directions, emphasizing the need for novel solutions to enhance the robustness and applicability of distributed LMs. 

---
# MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering 

**Authors**: Jialin Chen, Aosong Feng, Ziyu Zhao, Juan Garza, Gaukhar Nurbek, Cheng Qin, Ali Maatouk, Leandros Tassiulas, Yifeng Gao, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.16858)  

**Abstract**: Understanding the relationship between textual news and time-series evolution is a critical yet under-explored challenge in applied data science. While multimodal learning has gained traction, existing multimodal time-series datasets fall short in evaluating cross-modal reasoning and complex question answering, which are essential for capturing complex interactions between narrative information and temporal patterns. To bridge this gap, we introduce Multimodal Time Series Benchmark (MTBench), a large-scale benchmark designed to evaluate large language models (LLMs) on time series and text understanding across financial and weather domains. MTbench comprises paired time series and textual data, including financial news with corresponding stock price movements and weather reports aligned with historical temperature records. Unlike existing benchmarks that focus on isolated modalities, MTbench provides a comprehensive testbed for models to jointly reason over structured numerical trends and unstructured textual narratives. The richness of MTbench enables formulation of diverse tasks that require a deep understanding of both text and time-series data, including time-series forecasting, semantic and technical trend analysis, and news-driven question answering (QA). These tasks target the model's ability to capture temporal dependencies, extract key insights from textual context, and integrate cross-modal information. We evaluate state-of-the-art LLMs on MTbench, analyzing their effectiveness in modeling the complex relationships between news narratives and temporal patterns. Our findings reveal significant challenges in current models, including difficulties in capturing long-term dependencies, interpreting causality in financial and weather trends, and effectively fusing multimodal information. 

---
# Do Multimodal Large Language Models Understand Welding? 

**Authors**: Grigorii Khvatskii, Yong Suk Lee, Corey Angst, Maria Gibbs, Robert Landers, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2503.16537)  

**Abstract**: This paper examines the performance of Multimodal LLMs (MLLMs) in skilled production work, with a focus on welding. Using a novel data set of real-world and online weld images, annotated by a domain expert, we evaluate the performance of two state-of-the-art MLLMs in assessing weld acceptability across three contexts: RV \& Marine, Aeronautical, and Farming. While both models perform better on online images, likely due to prior exposure or memorization, they also perform relatively well on unseen, real-world weld images. Additionally, we introduce WeldPrompt, a prompting strategy that combines Chain-of-Thought generation with in-context learning to mitigate hallucinations and improve reasoning. WeldPrompt improves model recall in certain contexts but exhibits inconsistent performance across others. These results underscore the limitations and potentials of MLLMs in high-stakes technical domains and highlight the importance of fine-tuning, domain-specific data, and more sophisticated prompting strategies to improve model reliability. The study opens avenues for further research into multimodal learning in industry applications. 

---
# OpenVLThinker: An Early Exploration to Complex Vision-Language Reasoning via Iterative Self-Improvement 

**Authors**: Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17352)  

**Abstract**: Recent advancements demonstrated by DeepSeek-R1 have shown that complex reasoning abilities in large language models (LLMs), including sophisticated behaviors such as self-verification and self-correction, can be achieved by RL with verifiable rewards and significantly improves model performance on challenging tasks such as AIME. Motivated by these findings, our study investigates whether similar reasoning capabilities can be successfully integrated into large vision-language models (LVLMs) and assesses their impact on challenging multimodal reasoning tasks. We consider an approach that iteratively leverages supervised fine-tuning (SFT) on lightweight training data and Reinforcement Learning (RL) to further improve model generalization. Initially, reasoning capabilities were distilled from pure-text R1 models by generating reasoning steps using high-quality captions of the images sourced from diverse visual datasets. Subsequently, iterative RL training further enhance reasoning skills, with each iteration's RL-improved model generating refined SFT datasets for the next round. This iterative process yielded OpenVLThinker, a LVLM exhibiting consistently improved reasoning performance on challenging benchmarks such as MathVista, MathVerse, and MathVision, demonstrating the potential of our strategy for robust vision-language reasoning. The code, model and data are held at this https URL. 

---
# Chem42: a Family of chemical Language Models for Target-aware Ligand Generation 

**Authors**: Aahan Singh, Engin Tekin, Maryam Nadeem, Nancy A. ElNaker, Mohammad Amaan Sayeed, Natalia Vassilieva, Boulbaba Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2503.16563)  

**Abstract**: Revolutionizing drug discovery demands more than just understanding molecular interactions - it requires generative models that can design novel ligands tailored to specific biological targets. While chemical Language Models (cLMs) have made strides in learning molecular properties, most fail to incorporate target-specific insights, restricting their ability to drive de-novo ligand generation. Chem42, a cutting-edge family of generative chemical Language Models, is designed to bridge this gap. By integrating atomic-level interactions with multimodal inputs from Prot42, a complementary protein Language Model, Chem42 achieves a sophisticated cross-modal representation of molecular structures, interactions, and binding patterns. This innovative framework enables the creation of structurally valid, synthetically accessible ligands with enhanced target specificity. Evaluations across diverse protein targets confirm that Chem42 surpasses existing approaches in chemical validity, target-aware design, and predicted binding affinity. By reducing the search space of viable drug candidates, Chem42 could accelerate the drug discovery pipeline, offering a powerful generative AI tool for precision medicine. Our Chem42 models set a new benchmark in molecule property prediction, conditional molecule generation, and target-aware ligand design. The models are publicly available at this http URL. 

---
# EmpathyAgent: Can Embodied Agents Conduct Empathetic Actions? 

**Authors**: Xinyan Chen, Jiaxin Ge, Hongming Dai, Qiang Zhou, Qiuxuan Feng, Jingtong Hu, Yizhou Wang, Jiaming Liu, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16545)  

**Abstract**: Empathy is fundamental to human interactions, yet it remains unclear whether embodied agents can provide human-like empathetic support. Existing works have studied agents' tasks solving and social interactions abilities, but whether agents can understand empathetic needs and conduct empathetic behaviors remains overlooked. To address this, we introduce EmpathyAgent, the first benchmark to evaluate and enhance agents' empathetic actions across diverse scenarios. EmpathyAgent contains 10,000 multimodal samples with corresponding empathetic task plans and three different challenges. To systematically evaluate the agents' empathetic actions, we propose an empathy-specific evaluation suite that evaluates the agents' empathy process. We benchmark current models and found that exhibiting empathetic actions remains a significant challenge. Meanwhile, we train Llama3-8B using EmpathyAgent and find it can potentially enhance empathetic behavior. By establishing a standard benchmark for evaluating empathetic actions, we hope to advance research in empathetic embodied agents. Our code and data are publicly available at this https URL. 

---
# Multimodal Transformer Models for Turn-taking Prediction: Effects on Conversational Dynamics of Human-Agent Interaction during Cooperative Gameplay 

**Authors**: Young-Ho Bae, Casey C. Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2503.16432)  

**Abstract**: This study investigates multimodal turn-taking prediction within human-agent interactions (HAI), particularly focusing on cooperative gaming environments. It comprises both model development and subsequent user study, aiming to refine our understanding and improve conversational dynamics in spoken dialogue systems (SDSs). For the modeling phase, we introduce a novel transformer-based deep learning (DL) model that simultaneously integrates multiple modalities - text, vision, audio, and contextual in-game data to predict turn-taking events in real-time. Our model employs a Crossmodal Transformer architecture to effectively fuse information from these diverse modalities, enabling more comprehensive turn-taking predictions. The model demonstrates superior performance compared to baseline models, achieving 87.3% accuracy and 83.0% macro F1 score. A human user study was then conducted to empirically evaluate the turn-taking DL model in an interactive scenario with a virtual avatar while playing the game "Dont Starve Together", comparing a control condition without turn-taking prediction (n=20) to an experimental condition with our model deployed (n=40). Both conditions included a mix of English and Korean speakers, since turn-taking cues are known to vary by culture. We then analyzed the interaction quality, examining aspects such as utterance counts, interruption frequency, and participant perceptions of the avatar. Results from the user study suggest that our multimodal turn-taking model not only enhances the fluidity and naturalness of human-agent conversations, but also maintains a balanced conversational dynamic without significantly altering dialogue frequency. The study provides in-depth insights into the influence of turn-taking abilities on user perceptions and interaction quality, underscoring the potential for more contextually adaptive and responsive conversational agents. 

---
# Towards Automated Semantic Interpretability in Reinforcement Learning via Vision-Language Models 

**Authors**: Zhaoxin Li, Zhang Xi-Jia, Batuhan Altundas, Letian Chen, Rohan Paleja, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2503.16724)  

**Abstract**: Semantic Interpretability in Reinforcement Learning (RL) enables transparency, accountability, and safer deployment by making the agent's decisions understandable and verifiable. Achieving this, however, requires a feature space composed of human-understandable concepts, which traditionally rely on human specification and fail to generalize to unseen environments. In this work, we introduce Semantically Interpretable Reinforcement Learning with Vision-Language Models Empowered Automation (SILVA), an automated framework that leverages pre-trained vision-language models (VLM) for semantic feature extraction and interpretable tree-based models for policy optimization. SILVA first queries a VLM to identify relevant semantic features for an unseen environment, then extracts these features from the environment. Finally, it trains an Interpretable Control Tree via RL, mapping the extracted features to actions in a transparent and interpretable manner. To address the computational inefficiency of extracting features directly with VLMs, we develop a feature extraction pipeline that generates a dataset for training a lightweight convolutional network, which is subsequently used during RL. By leveraging VLMs to automate tree-based RL, SILVA removes the reliance on human annotation previously required by interpretable models while also overcoming the inability of VLMs alone to generate valid robot policies, enabling semantically interpretable reinforcement learning without human-in-the-loop. 

---
# MAPS: A Multi-Agent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving 

**Authors**: Jian Zhang, Zhiyuan Wang, Zhangqi Wang, Xinyu Zhang, Fangzhi Xu, Qika Lin, Rui Mao, Erik Cambria, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16905)  

**Abstract**: Multimodal scientific problems (MSPs) involve complex issues that require the integration of multiple modalities, such as text and diagrams, presenting a significant challenge in artificial intelligence. While progress has been made in addressing traditional scientific problems, MSPs still face two primary issues: the challenge of multi-modal comprehensive reasoning in scientific problem-solving and the lack of reflective and rethinking capabilities. To address these issues, we introduce a Multi-Agent framework based on the Big Seven Personality and Socratic guidance (MAPS). This framework employs seven distinct agents that leverage feedback mechanisms and the Socratic method to guide the resolution of MSPs. To tackle the first issue, we propose a progressive four-agent solving strategy, where each agent focuses on a specific stage of the problem-solving process. For the second issue, we introduce a Critic agent, inspired by Socratic questioning, which prompts critical thinking and stimulates autonomous learning. We conduct extensive experiments on the EMMA, Olympiad, and MathVista datasets, achieving promising results that outperform the current SOTA model by 15.84% across all tasks. Meanwhile, the additional analytical experiments also verify the model's progress as well as generalization ability. 

---
# GAIR: Improving Multimodal Geo-Foundation Model with Geo-Aligned Implicit Representations 

**Authors**: Zeping Liu, Fan Zhang, Junfeng Jiao, Ni Lao, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16683)  

**Abstract**: Advancements in vision and language foundation models have inspired the development of geo-foundation models (GeoFMs), enhancing performance across diverse geospatial tasks. However, many existing GeoFMs primarily focus on overhead remote sensing (RS) data while neglecting other data modalities such as ground-level imagery. A key challenge in multimodal GeoFM development is to explicitly model geospatial relationships across modalities, which enables generalizability across tasks, spatial scales, and temporal contexts. To address these limitations, we propose GAIR, a novel multimodal GeoFM architecture integrating overhead RS data, street view (SV) imagery, and their geolocation metadata. We utilize three factorized neural encoders to project an SV image, its geolocation, and an RS image into the embedding space. The SV image needs to be located within the RS image's spatial footprint but does not need to be at its geographic center. In order to geographically align the SV image and RS image, we propose a novel implicit neural representations (INR) module that learns a continuous RS image representation and looks up the RS embedding at the SV image's geolocation. Next, these geographically aligned SV embedding, RS embedding, and location embedding are trained with contrastive learning objectives from unlabeled data. We evaluate GAIR across 10 geospatial tasks spanning RS image-based, SV image-based, and location embedding-based benchmarks. Experimental results demonstrate that GAIR outperforms state-of-the-art GeoFMs and other strong baselines, highlighting its effectiveness in learning generalizable and transferable geospatial representations. 

---
# From Voices to Worlds: Developing an AI-Powered Framework for 3D Object Generation in Augmented Reality 

**Authors**: Majid Behravan, Denis Gracanin  

**Link**: [PDF](https://arxiv.org/pdf/2503.16474)  

**Abstract**: This paper presents Matrix, an advanced AI-powered framework designed for real-time 3D object generation in Augmented Reality (AR) environments. By integrating a cutting-edge text-to-3D generative AI model, multilingual speech-to-text translation, and large language models (LLMs), the system enables seamless user interactions through spoken commands. The framework processes speech inputs, generates 3D objects, and provides object recommendations based on contextual understanding, enhancing AR experiences. A key feature of this framework is its ability to optimize 3D models by reducing mesh complexity, resulting in significantly smaller file sizes and faster processing on resource-constrained AR devices. Our approach addresses the challenges of high GPU usage, large model output sizes, and real-time system responsiveness, ensuring a smoother user experience. Moreover, the system is equipped with a pre-generated object repository, further reducing GPU load and improving efficiency. We demonstrate the practical applications of this framework in various fields such as education, design, and accessibility, and discuss future enhancements including image-to-3D conversion, environmental object detection, and multimodal support. The open-source nature of the framework promotes ongoing innovation and its utility across diverse industries. 

---
# Enhancing Explainability with Multimodal Context Representations for Smarter Robots 

**Authors**: Anargh Viswanath, Lokesh Veeramacheneni, Hendrik Buschmeier  

**Link**: [PDF](https://arxiv.org/pdf/2503.16467)  

**Abstract**: Artificial Intelligence (AI) has significantly advanced in recent years, driving innovation across various fields, especially in robotics. Even though robots can perform complex tasks with increasing autonomy, challenges remain in ensuring explainability and user-centered design for effective interaction. A key issue in Human-Robot Interaction (HRI) is enabling robots to effectively perceive and reason over multimodal inputs, such as audio and vision, to foster trust and seamless collaboration. In this paper, we propose a generalized and explainable multimodal framework for context representation, designed to improve the fusion of speech and vision modalities. We introduce a use case on assessing 'Relevance' between verbal utterances from the user and visual scene perception of the robot. We present our methodology with a Multimodal Joint Representation module and a Temporal Alignment module, which can allow robots to evaluate relevance by temporally aligning multimodal inputs. Finally, we discuss how the proposed framework for context representation can help with various aspects of explainability in HRI. 

---
# OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents 

**Authors**: Pengzhou Cheng, Zheng Wu, Zongru Wu, Aston Zhang, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16465)  

**Abstract**: Autonomous graphical user interface (GUI) agents powered by multimodal large language models have shown great promise. However, a critical yet underexplored issue persists: over-execution, where the agent executes tasks in a fully autonomous way, without adequate assessment of its action confidence to compromise an adaptive human-agent collaboration. This poses substantial risks in complex scenarios, such as those involving ambiguous user instructions, unexpected interruptions, and environmental hijacks. To address the issue, we introduce OS-Kairos, an adaptive GUI agent capable of predicting confidence levels at each interaction step and efficiently deciding whether to act autonomously or seek human intervention. OS-Kairos is developed through two key mechanisms: (i) collaborative probing that annotates confidence scores at each interaction step; (ii) confidence-driven interaction that leverages these confidence scores to elicit the ability of adaptive interaction. Experimental results show that OS-Kairos substantially outperforms existing models on our curated dataset featuring complex scenarios, as well as on established benchmarks such as AITZ and Meta-GUI, with 24.59\%$\sim$87.29\% improvements in task success rate. OS-Kairos facilitates an adaptive human-agent collaboration, prioritizing effectiveness, generality, scalability, and efficiency for real-world GUI interaction. The dataset and codes are available at this https URL. 

---
# An Audio-Visual Fusion Emotion Generation Model Based on Neuroanatomical Alignment 

**Authors**: Haidong Wang, Qia Shan, JianHua Zhang, PengFei Xiao, Ao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.16454)  

**Abstract**: In the field of affective computing, traditional methods for generating emotions predominantly rely on deep learning techniques and large-scale emotion datasets. However, deep learning techniques are often complex and difficult to interpret, and standardizing large-scale emotional datasets are difficult and costly to establish. To tackle these challenges, we introduce a novel framework named Audio-Visual Fusion for Brain-like Emotion Learning(AVF-BEL). In contrast to conventional brain-inspired emotion learning methods, this approach improves the audio-visual emotion fusion and generation model through the integration of modular components, thereby enabling more lightweight and interpretable emotion learning and generation processes. The framework simulates the integration of the visual, auditory, and emotional pathways of the brain, optimizes the fusion of emotional features across visual and auditory modalities, and improves upon the traditional Brain Emotional Learning (BEL) model. The experimental results indicate a significant improvement in the similarity of the audio-visual fusion emotion learning generation model compared to single-modality visual and auditory emotion learning and generation model. Ultimately, this aligns with the fundamental phenomenon of heightened emotion generation facilitated by the integrated impact of visual and auditory stimuli. This contribution not only enhances the interpretability and efficiency of affective intelligence but also provides new insights and pathways for advancing affective computing technology. Our source code can be accessed here: this https URL}{this https URL. 

---
# ACE, Action and Control via Explanations: A Proposal for LLMs to Provide Human-Centered Explainability for Multimodal AI Assistants 

**Authors**: Elizabeth Anne Watkins, Emanuel Moss, Ramesh Manuvinakurike, Meng Shi, Richard Beckwith, Giuseppe Raffa  

**Link**: [PDF](https://arxiv.org/pdf/2503.16466)  

**Abstract**: In this short paper we address issues related to building multimodal AI systems for human performance support in manufacturing domains. We make two contributions: we first identify challenges of participatory design and training of such systems, and secondly, to address such challenges, we propose the ACE paradigm: "Action and Control via Explanations". Specifically, we suggest that LLMs can be used to produce explanations in the form of human interpretable "semantic frames", which in turn enable end users to provide data the AI system needs to align its multimodal models and representations, including computer vision, automatic speech recognition, and document inputs. ACE, by using LLMs to "explain" using semantic frames, will help the human and the AI system to collaborate, together building a more accurate model of humans activities and behaviors, and ultimately more accurate predictive outputs for better task support, and better outcomes for human users performing manual tasks. 

---
# Think-Then-React: Towards Unconstrained Human Action-to-Reaction Generation 

**Authors**: Wenhui Tan, Boyuan Li, Chuhao Jin, Wenbing Huang, Xiting Wang, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.16451)  

**Abstract**: Modeling human-like action-to-reaction generation has significant real-world applications, like human-robot interaction and games. Despite recent advancements in single-person motion generation, it is still challenging to well handle action-to-reaction generation, due to the difficulty of directly predicting reaction from action sequence without prompts, and the absence of a unified representation that effectively encodes multi-person motion. To address these challenges, we introduce Think-Then-React (TTR), a large language-model-based framework designed to generate human-like reactions. First, with our fine-grained multimodal training strategy, TTR is capable to unify two processes during inference: a thinking process that explicitly infers action intentions and reasons corresponding reaction description, which serve as semantic prompts, and a reacting process that predicts reactions based on input action and the inferred semantic prompts. Second, to effectively represent multi-person motion in language models, we propose a unified motion tokenizer by decoupling egocentric pose and absolute space features, which effectively represents action and reaction motion with same encoding. Extensive experiments demonstrate that TTR outperforms existing baselines, achieving significant improvements in evaluation metrics, such as reducing FID from 3.988 to 1.942. 

---
# DreamLLM-3D: Affective Dream Reliving using Large Language Model and 3D Generative AI 

**Authors**: Pinyao Liu, Keon Ju Lee, Alexander Steinmaurer, Claudia Picard-Deland, Michelle Carr, Alexandra Kitson  

**Link**: [PDF](https://arxiv.org/pdf/2503.16439)  

**Abstract**: We present DreamLLM-3D, a composite multimodal AI system behind an immersive art installation for dream re-experiencing. It enables automated dream content analysis for immersive dream-reliving, by integrating a Large Language Model (LLM) with text-to-3D Generative AI. The LLM processes voiced dream reports to identify key dream entities (characters and objects), social interaction, and dream sentiment. The extracted entities are visualized as dynamic 3D point clouds, with emotional data influencing the color and soundscapes of the virtual dream environment. Additionally, we propose an experiential AI-Dreamworker Hybrid paradigm. Our system and paradigm could potentially facilitate a more emotionally engaging dream-reliving experience, enhancing personal insights and creativity. 

---
# Interactive Sketchpad: An Interactive Multimodal System for Collaborative, Visual Problem-Solving 

**Authors**: Steven-Shine Chen, Jimin Lee, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16434)  

**Abstract**: Humans have long relied on visual aids like sketches and diagrams to support reasoning and problem-solving. Visual tools, like auxiliary lines in geometry or graphs in calculus, are essential for understanding complex ideas. However, many tutoring systems remain text-based, providing feedback only through natural language. Leveraging recent advances in Large Multimodal Models (LMMs), this paper introduces Interactive Sketchpad, a tutoring system that combines language-based explanations with interactive visualizations to enhance learning. Built on a pre-trained LMM, Interactive Sketchpad is fine-tuned to provide step-by-step guidance in both text and visuals, enabling natural multimodal interaction with the student. Accurate and robust diagrams are generated by incorporating code execution into the reasoning process. User studies conducted on math problems such as geometry, calculus, and trigonometry demonstrate that Interactive Sketchpad leads to improved task comprehension, problem-solving accuracy, and engagement levels, highlighting its potential for transforming educational technologies. 

---
