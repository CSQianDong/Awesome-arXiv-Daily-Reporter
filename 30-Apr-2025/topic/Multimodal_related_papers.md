# An Integrated Framework for Contextual Personalized LLM-Based Food Recommendation 

**Authors**: Ali Rostami  

**Link**: [PDF](https://arxiv.org/pdf/2504.20092)  

**Abstract**: Personalized food recommendation systems (Food-RecSys) critically underperform due to fragmented component understanding and the failure of conventional machine learning with vast, imbalanced food data. While Large Language Models (LLMs) offer promise, current generic Recommendation as Language Processing (RLP) strategies lack the necessary specialization for the food domain's complexity. This thesis tackles these deficiencies by first identifying and analyzing the essential components for effective Food-RecSys. We introduce two key innovations: a multimedia food logging platform for rich contextual data acquisition and the World Food Atlas, enabling unique geolocation-based food analysis previously unavailable. Building on this foundation, we pioneer the Food Recommendation as Language Processing (F-RLP) framework - a novel, integrated approach specifically architected for the food domain. F-RLP leverages LLMs in a tailored manner, overcoming the limitations of generic models and providing a robust infrastructure for effective, contextual, and truly personalized food recommendations. 

---
# UniversalRAG: Retrieval-Augmented Generation over Multiple Corpora with Diverse Modalities and Granularities 

**Authors**: Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.20734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing RAG approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, a novel RAG framework designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single combined corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose a modality-aware routing mechanism that dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it. Also, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 8 benchmarks spanning multiple modalities, showing its superiority over modality-specific and unified baselines. 

---
# A Multimodal Pipeline for Clinical Data Extraction: Applying Vision-Language Models to Scans of Transfusion Reaction Reports 

**Authors**: Henning Schäfer, Cynthia S. Schmidt, Johannes Wutzkowsky, Kamil Lorek, Lea Reinartz, Johannes Rückert, Christian Temme, Britta Böckmann, Peter A. Horn, Christoph M. Friedrich  

**Link**: [PDF](https://arxiv.org/pdf/2504.20220)  

**Abstract**: Despite the growing adoption of electronic health records, many processes still rely on paper documents, reflecting the heterogeneous real-world conditions in which healthcare is delivered. The manual transcription process is time-consuming and prone to errors when transferring paper-based data to digital formats. To streamline this workflow, this study presents an open-source pipeline that extracts and categorizes checkbox data from scanned documents. Demonstrated on transfusion reaction reports, the design supports adaptation to other checkbox-rich document types. The proposed method integrates checkbox detection, multilingual optical character recognition (OCR) and multilingual vision-language models (VLMs). The pipeline achieves high precision and recall compared against annually compiled gold-standards from 2017 to 2024. The result is a reduction in administrative workload and accurate regulatory reporting. The open-source availability of this pipeline encourages self-hosted parsing of checkbox forms. 

---
# mrCAD: Multimodal Refinement of Computer-aided Designs 

**Authors**: William P. McCarthy, Saujas Vaduguru, Karl D. D. Willis, Justin Matejka, Judith E. Fan, Daniel Fried, Yewen Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20294)  

**Abstract**: A key feature of human collaboration is the ability to iteratively refine the concepts we have communicated. In contrast, while generative AI excels at the \textit{generation} of content, it often struggles to make specific language-guided \textit{modifications} of its prior outputs. To bridge the gap between how humans and machines perform edits, we present mrCAD, a dataset of multimodal instructions in a communication game. In each game, players created computer aided designs (CADs) and refined them over several rounds to match specific target designs. Only one player, the Designer, could see the target, and they must instruct the other player, the Maker, using text, drawing, or a combination of modalities. mrCAD consists of 6,082 communication games, 15,163 instruction-execution rounds, played between 1,092 pairs of human players. We analyze the dataset and find that generation and refinement instructions differ in their composition of drawing and text. Using the mrCAD task as a benchmark, we find that state-of-the-art VLMs are better at following generation instructions than refinement instructions. These results lay a foundation for analyzing and modeling a multimodal language of refinement that is not represented in previous datasets. 

---
# Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains 

**Authors**: Juntian Zhang, Chuanqi cheng, Yuhan Liu, Wei Liu, Jian Luan, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.20199)  

**Abstract**: Vision-language models (VLMs) achieve remarkable success in single-image tasks. However, real-world scenarios often involve intricate multi-image inputs, leading to a notable performance decline as models struggle to disentangle critical information scattered across complex visual features. In this work, we propose Focus-Centric Visual Chain, a novel paradigm that enhances VLMs'perception, comprehension, and reasoning abilities in multi-image scenarios. To facilitate this paradigm, we propose Focus-Centric Data Synthesis, a scalable bottom-up approach for synthesizing high-quality data with elaborate reasoning paths. Through this approach, We construct VISC-150K, a large-scale dataset with reasoning data in the form of Focus-Centric Visual Chain, specifically designed for multi-image tasks. Experimental results on seven multi-image benchmarks demonstrate that our method achieves average performance gains of 3.16% and 2.24% across two distinct model architectures, without compromising the general vision-language capabilities. our study represents a significant step toward more robust and capable vision-language systems that can handle complex visual scenarios. 

---
# YoChameleon: Personalized Vision and Language Generation 

**Authors**: Thao Nguyen, Krishna Kumar Singh, Jing Shi, Trung Bui, Yong Jae Lee, Yuheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.20998)  

**Abstract**: Large Multimodal Models (e.g., GPT-4, Gemini, Chameleon) have evolved into powerful tools with millions of users. However, they remain generic models and lack personalized knowledge of specific user concepts. Previous work has explored personalization for text generation, yet it remains unclear how these methods can be adapted to new modalities, such as image generation. In this paper, we introduce Yo'Chameleon, the first attempt to study personalization for large multimodal models. Given 3-5 images of a particular concept, Yo'Chameleon leverages soft-prompt tuning to embed subject-specific information to (i) answer questions about the subject and (ii) recreate pixel-level details to produce images of the subject in new contexts. Yo'Chameleon is trained with (i) a self-prompting optimization mechanism to balance performance across multiple modalities, and (ii) a ``soft-positive" image generation approach to enhance image quality in a few-shot setting. 

---
# TAMO:Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data 

**Authors**: Qi Wang, Xiao Zhang, Mingyi Li, Yuan Yuan, Mengbai Xiao, Fuzhen Zhuang, Dongxiao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.20462)  

**Abstract**: With the development of distributed systems, microservices and cloud native technologies have become central to modern enterprise software development. Despite bringing significant advantages, these technologies also increase system complexity and operational challenges. Traditional root cause analysis (RCA) struggles to achieve automated fault response, heavily relying on manual intervention. In recent years, large language models (LLMs) have made breakthroughs in contextual inference and domain knowledge integration, providing new solutions for Artificial Intelligence for Operations (AIOps). However, Existing LLM-based approaches face three key challenges: text input constraints, dynamic service dependency hallucinations, and context window limitations. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data, namely TAMO, for fine-grained RCA. It unifies multi-modal observational data into time-aligned representations to extract consistent features and employs specialized root cause localization and fault classification tools for perceiving the contextual environment. This approach overcomes the limitations of LLM in handling real-time changing service dependencies and raw observational data and guides LLM to generate repair strategies aligned with system contexts by structuring key information into a prompt. Experimental results show that TAMO performs well in root cause analysis when dealing with public datasets characterized by heterogeneity and common fault types, demonstrating its effectiveness. 

---
# AlignDiT: Multimodal Aligned Diffusion Transformer for Synchronized Speech Generation 

**Authors**: Jeongsoo Choi, Ji-Hoon Kim, Kim Sung-Bin, Tae-Hyun Oh, Joon Son Chung  

**Link**: [PDF](https://arxiv.org/pdf/2504.20629)  

**Abstract**: In this paper, we address the task of multimodal-to-speech generation, which aims to synthesize high-quality speech from multiple input modalities: text, video, and reference audio. This task has gained increasing attention due to its wide range of applications, such as film production, dubbing, and virtual avatars. Despite recent progress, existing methods still suffer from limitations in speech intelligibility, audio-video synchronization, speech naturalness, and voice similarity to the reference speaker. To address these challenges, we propose AlignDiT, a multimodal Aligned Diffusion Transformer that generates accurate, synchronized, and natural-sounding speech from aligned multimodal inputs. Built upon the in-context learning capability of the DiT architecture, AlignDiT explores three effective strategies to align multimodal representations. Furthermore, we introduce a novel multimodal classifier-free guidance mechanism that allows the model to adaptively balance information from each modality during speech synthesis. Extensive experiments demonstrate that AlignDiT significantly outperforms existing methods across multiple benchmarks in terms of quality, synchronization, and speaker similarity. Moreover, AlignDiT exhibits strong generalization capability across various multimodal tasks, such as video-to-speech synthesis and visual forced alignment, consistently achieving state-of-the-art performance. The demo page is available at this https URL . 

---
# TrueFake: A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks 

**Authors**: Stefano Dell'Anna, Andrea Montibeller, Giulia Boato  

**Link**: [PDF](https://arxiv.org/pdf/2504.20658)  

**Abstract**: AI-generated synthetic media are increasingly used in real-world scenarios, often with the purpose of spreading misinformation and propaganda through social media platforms, where compression and other processing can degrade fake detection cues. Currently, many forensic tools fail to account for these in-the-wild challenges. In this work, we introduce TrueFake, a large-scale benchmarking dataset of 600,000 images including top notch generative techniques and sharing via three different social networks. This dataset allows for rigorous evaluation of state-of-the-art fake image detectors under very realistic and challenging conditions. Through extensive experimentation, we analyze how social media sharing impacts detection performance, and identify current most effective detection and training strategies. Our findings highlight the need for evaluating forensic models in conditions that mirror real-world use. 

---
# AutoP2C: An LLM-Based Agent Framework for Code Repository Generation from Multimodal Content in Academic Papers 

**Authors**: Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, Mingjun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.20115)  

**Abstract**: Machine Learning (ML) research is spread through academic papers featuring rich multimodal content, including text, diagrams, and tabular results. However, translating these multimodal elements into executable code remains a challenging and time-consuming process that requires substantial ML expertise. We introduce ``Paper-to-Code'' (P2C), a novel task that transforms the multimodal content of scientific publications into fully executable code repositories, which extends beyond the existing formulation of code generation that merely converts textual descriptions into isolated code snippets. To automate the P2C process, we propose AutoP2C, a multi-agent framework based on large language models that processes both textual and visual content from research papers to generate complete code repositories. Specifically, AutoP2C contains four stages: (1) repository blueprint extraction from established codebases, (2) multimodal content parsing that integrates information from text, equations, and figures, (3) hierarchical task decomposition for structured code generation, and (4) iterative feedback-driven debugging to ensure functionality and performance. Evaluation on a benchmark of eight research papers demonstrates the effectiveness of AutoP2C, which can successfully generate executable code repositories for all eight papers, while OpenAI-o1 or DeepSeek-R1 can only produce runnable code for one paper. The code is available at this https URL. 

---
