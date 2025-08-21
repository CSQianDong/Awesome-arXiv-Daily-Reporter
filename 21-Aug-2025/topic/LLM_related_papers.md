# MCP-Universe: Benchmarking Large Language Models with Real-World Model Context Protocol Servers 

**Authors**: Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14704)  

**Abstract**: The Model Context Protocol has emerged as a transformative standard for connecting large language models to external data sources and tools, rapidly gaining adoption across major AI providers and development platforms. However, existing benchmarks are overly simplistic and fail to capture real application challenges such as long-horizon reasoning and large, unfamiliar tool spaces. To address this critical gap, we introduce MCP-Universe, the first comprehensive benchmark specifically designed to evaluate LLMs in realistic and hard tasks through interaction with real-world MCP servers. Our benchmark encompasses 6 core domains spanning 11 different MCP servers: Location Navigation, Repository Management, Financial Analysis, 3D Design, Browser Automation, and Web Searching. To ensure rigorous evaluation, we implement execution-based evaluators, including format evaluators for agent format compliance, static evaluators for time-invariant content matching, and dynamic evaluators that automatically retrieve real-time ground truth for temporally sensitive tasks. Through extensive evaluation of leading LLMs, we find that even SOTA models such as GPT-5 (43.72%), Grok-4 (33.33%) and Claude-4.0-Sonnet (29.44%) exhibit significant performance limitations. In addition, our benchmark poses a significant long-context challenge for LLM agents, as the number of input tokens increases rapidly with the number of interaction steps. Moreover, it introduces an unknown-tools challenge, as LLM agents often lack familiarity with the precise usage of the MCP servers. Notably, enterprise-level agents like Cursor cannot achieve better performance than standard ReAct frameworks. Beyond evaluation, we open-source our extensible evaluation framework with UI support, enabling researchers and practitioners to seamlessly integrate new agents and MCP servers while fostering innovation in the rapidly evolving MCP ecosystem. 

---
# Privileged Self-Access Matters for Introspection in AI 

**Authors**: Siyuan Song, Harvey Lederman, Jennifer Hu, Kyle Mahowald  

**Link**: [PDF](https://arxiv.org/pdf/2508.14802)  

**Abstract**: Whether AI models can introspect is an increasingly important practical question. But there is no consensus on how introspection is to be defined. Beginning from a recently proposed ''lightweight'' definition, we argue instead for a thicker one. According to our proposal, introspection in AI is any process which yields information about internal states through a process more reliable than one with equal or lower computational cost available to a third party. Using experiments where LLMs reason about their internal temperature parameters, we show they can appear to have lightweight introspection while failing to meaningfully introspect per our proposed definition. 

---
# Automated Optimization Modeling through Expert-Guided Large Language Model Reasoning 

**Authors**: Beinuo Yang, Qishen Zhou, Junyi Li, Xingchen Su, Simon Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14410)  

**Abstract**: Optimization Modeling (OM) is essential for solving complex decision-making problems. However, the process remains time-consuming and error-prone, heavily relying on domain experts. While Large Language Models (LLMs) show promise in addressing these challenges through their natural language understanding and reasoning capabilities, current approaches face three critical limitations: high benchmark labeling error rates reaching up to 42\%, narrow evaluation scope that only considers optimal values, and computational inefficiency due to heavy reliance on multi-agent systems or model fine-tuning. In this work, we first enhance existing datasets through systematic error correction and more comprehensive annotation. Additionally, we introduce LogiOR, a new optimization modeling benchmark from the logistics domain, containing more complex problems with standardized annotations. Furthermore, we present ORThought, a novel framework that leverages expert-level optimization modeling principles through chain-of-thought reasoning to automate the OM process. Through extensive empirical evaluation, we demonstrate that ORThought outperforms existing approaches, including multi-agent frameworks, with particularly significant advantages on complex optimization problems. Finally, we provide a systematic analysis of our method, identifying critical success factors and failure modes, providing valuable insights for future research on LLM-based optimization modeling. 

---
# Long Chain-of-Thought Reasoning Across Languages 

**Authors**: Josh Barua, Seun Eisape, Kayo Yin, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2508.14828)  

**Abstract**: Scaling inference through long chains-of-thought (CoTs) has unlocked impressive reasoning capabilities in large language models (LLMs), yet the reasoning process remains almost exclusively English-centric. We construct translated versions of two popular English reasoning datasets, fine-tune Qwen 2.5 (7B) and Qwen 3 (8B) models, and present a systematic study of long CoT generation across French, Japanese, Latvian, and Swahili. Our experiments reveal three key findings. First, the efficacy of using English as a pivot language varies by language: it provides no benefit for French, improves performance when used as the reasoning language for Japanese and Latvian, and proves insufficient for Swahili where both task comprehension and reasoning remain poor. Second, extensive multilingual pretraining in Qwen 3 narrows but does not eliminate the cross-lingual performance gap. A lightweight fine-tune using only 1k traces still improves performance by over 30\% in Swahili. Third, data quality versus scale trade-offs are language dependent: small, carefully curated datasets suffice for English and French, whereas larger but noisier corpora prove more effective for Swahili and Latvian. Together, these results clarify when and why long CoTs transfer across languages and provide translated datasets to foster equitable multilingual reasoning research. 

---
# PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning 

**Authors**: Ruheng Wang, Hang Zhang, Trieu Nguyen, Shasha Feng, Hao-Wei Pang, Xiang Yu, Li Xiao, Peter Zhiping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14765)  

**Abstract**: Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery. 

---
# TransLLM: A Unified Multi-Task Foundation Framework for Urban Transportation via Learnable Prompting 

**Authors**: Jiaming Leng, Yunying Bi, Chuan Qin, Bing Yin, Yanyong Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14782)  

**Abstract**: Urban transportation systems encounter diverse challenges across multiple tasks, such as traffic forecasting, electric vehicle (EV) charging demand prediction, and taxi dispatch. Existing approaches suffer from two key limitations: small-scale deep learning models are task-specific and data-hungry, limiting their generalizability across diverse scenarios, while large language models (LLMs), despite offering flexibility through natural language interfaces, struggle with structured spatiotemporal data and numerical reasoning in transportation domains. To address these limitations, we propose TransLLM, a unified foundation framework that integrates spatiotemporal modeling with large language models through learnable prompt composition. Our approach features a lightweight spatiotemporal encoder that captures complex dependencies via dilated temporal convolutions and dual-adjacency graph attention networks, seamlessly interfacing with LLMs through structured embeddings. A novel instance-level prompt routing mechanism, trained via reinforcement learning, dynamically personalizes prompts based on input characteristics, moving beyond fixed task-specific templates. The framework operates by encoding spatiotemporal patterns into contextual representations, dynamically composing personalized prompts to guide LLM reasoning, and projecting the resulting representations through specialized output layers to generate task-specific predictions. Experiments across seven datasets and three tasks demonstrate the exceptional effectiveness of TransLLM in both supervised and zero-shot settings. Compared to ten baseline models, it delivers competitive performance on both regression and planning problems, showing strong generalization and cross-task adaptability. Our code is available at this https URL. 

---
# Large Language Models are Highly Aligned with Human Ratings of Emotional Stimuli 

**Authors**: Mattson Ogg, Chace Ashcraft, Ritwik Bose, Raphael Norman-Tenazas, Michael Wolmetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.14214)  

**Abstract**: Emotions exert an immense influence over human behavior and cognition in both commonplace and high-stress tasks. Discussions of whether or how to integrate large language models (LLMs) into everyday life (e.g., acting as proxies for, or interacting with, human agents), should be informed by an understanding of how these tools evaluate emotionally loaded stimuli or situations. A model's alignment with human behavior in these cases can inform the effectiveness of LLMs for certain roles or interactions. To help build this understanding, we elicited ratings from multiple popular LLMs for datasets of words and images that were previously rated for their emotional content by humans. We found that when performing the same rating tasks, GPT-4o responded very similarly to human participants across modalities, stimuli and most rating scales (r = 0.9 or higher in many cases). However, arousal ratings were less well aligned between human and LLM raters, while happiness ratings were most highly aligned. Overall LLMs aligned better within a five-category (happiness, anger, sadness, fear, disgust) emotion framework than within a two-dimensional (arousal and valence) organization. Finally, LLM ratings were substantially more homogenous than human ratings. Together these results begin to describe how LLM agents interpret emotional stimuli and highlight similarities and differences among biological and artificial intelligence in key behavioral domains. 

---
# Explaining Hitori Puzzles: Neurosymbolic Proof Staging for Sequential Decisions 

**Authors**: Maria Leonor Pacheco, Fabio Somenzi, Dananjay Srinivas, Ashutosh Trivedi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14294)  

**Abstract**: We propose a neurosymbolic approach to the explanation of complex sequences of decisions that combines the strengths of decision procedures and Large Language Models (LLMs). We demonstrate this approach by producing explanations for the solutions of Hitori puzzles. The rules of Hitori include local constraints that are effectively explained by short resolution proofs. However, they also include a connectivity constraint that is more suitable for visual explanations. Hence, Hitori provides an excellent testing ground for a flexible combination of SAT solvers and LLMs. We have implemented a tool that assists humans in solving Hitori puzzles, and we present experimental evidence of its effectiveness. 

---
# Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference 

**Authors**: Samir Abdaljalil, Erchin Serpedin, Khalid Qaraqe, Hasan Kurban  

**Link**: [PDF](https://arxiv.org/pdf/2508.14735)  

**Abstract**: Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: this https URL 

---
# Transplant Then Regenerate: A New Paradigm for Text Data Augmentation 

**Authors**: Guangzhan Wang, Hongyu Zhang, Beijun Shen, Xiaodong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14723)  

**Abstract**: Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows. 

---
# ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine 

**Authors**: Junying Chen, Zhenyang Cai, Zhiheng Liu, Yunjin Yang, Rongsheng Wang, Qingying Xiao, Xiangyi Feng, Zhan Su, Jing Guo, Xiang Wan, Guangjun Yu, Haizhou Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14706)  

**Abstract**: Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field. 

---
# Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs 

**Authors**: Haokun Lin, Haobo Xu, Yichen Wu, Ziyu Guo, Renrui Zhang, Zhichao Lu, Ying Wei, Qingfu Zhang, Zhenan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14896)  

**Abstract**: Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. All codes and experimental setups will be released to support the community. 

---
# Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination 

**Authors**: João Vitor de Carvalho Silva, Douglas G. Macharet  

**Link**: [PDF](https://arxiv.org/pdf/2508.14635)  

**Abstract**: The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements. 

---
# In2x at WMT25 Translation Task 

**Authors**: Lei Pang, Hanyi Mao, Quanjia Xiao, HaiXiao Liu, Xiangyi Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.14472)  

**Abstract**: This paper presents the open-system submission by the In2x research team for the WMT25 General Machine Translation Shared Task. Our submission focuses on Japanese-related translation tasks, aiming to explore a generalizable paradigm for extending large language models (LLMs) to other languages. This paradigm encompasses aspects such as data construction methods and reward model design. The ultimate goal is to enable large language model systems to achieve exceptional performance in low-resource or less commonly spoken languages. 

---
# Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs 

**Authors**: Yinghan Zhou, Weifeng Zhu, Juan Wen, Wanli Peng, Zhengxian Wu, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14408)  

**Abstract**: Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively. 

---
# DEPTH: Hallucination-Free Relation Extraction via Dependency-Aware Sentence Simplification and Two-tiered Hierarchical Refinement 

**Authors**: Yupei Yang, Fan Feng, Lin Yang, Wanxi Deng, Lin Qu, Biwei Huang, Shikui Tu, Lei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14391)  

**Abstract**: Relation extraction enables the construction of structured knowledge for many downstream applications. While large language models (LLMs) have shown great promise in this domain, most existing methods concentrate on relation classification, which predicts the semantic relation type between a related entity pair. However, we observe that LLMs often struggle to reliably determine whether a relation exists, especially in cases involving complex sentence structures or intricate semantics, which leads to spurious predictions. Such hallucinations can introduce noisy edges in knowledge graphs, compromising the integrity of structured knowledge and downstream reliability. To address these challenges, we propose DEPTH, a framework that integrates Dependency-aware sEntence simPlification and Two-tiered Hierarchical refinement into the relation extraction pipeline. Given a sentence and its candidate entity pairs, DEPTH operates in two stages: (1) the Grounding module extracts relations for each pair by leveraging their shortest dependency path, distilling the sentence into a minimal yet coherent relational context that reduces syntactic noise while preserving key semantics; (2) the Refinement module aggregates all local predictions and revises them based on a holistic understanding of the sentence, correcting omissions and inconsistencies. We further introduce a causality-driven reward model that mitigates reward hacking by disentangling spurious correlations, enabling robust fine-tuning via reinforcement learning with human feedback. Experiments on six benchmarks demonstrate that DEPTH reduces the average hallucination rate to 7.0\% while achieving a 17.2\% improvement in average F1 score over state-of-the-art baselines. 

---
# Credence Calibration Game? Calibrating Large Language Models through Structured Play 

**Authors**: Ke Fang, Tianyi Zhao, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.14390)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in decision-critical domains, it becomes essential to ensure that their confidence estimates faithfully correspond to their actual correctness. Existing calibration methods have primarily focused on post-hoc adjustments or auxiliary model training; however, many of these approaches necessitate additional supervision or parameter updates. In this work, we propose a novel prompt-based calibration framework inspired by the Credence Calibration Game. Our method establishes a structured interaction loop wherein LLMs receive feedback based on the alignment of their predicted confidence with correctness. Through feedback-driven prompting and natural language summaries of prior performance, our framework dynamically improves model calibration. Extensive experiments across models and game configurations demonstrate consistent improvements in evaluation metrics. Our results highlight the potential of game-based prompting as an effective strategy for LLM calibration. Code and data are available at this https URL. 

---
# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs 

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14817)  

**Abstract**: Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even state-of-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text. 

---
# ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities 

**Authors**: Wenhan Dong, Zhen Sun, Yuemeng Zhao, Zifan Peng, Jun Wu, Jingyi Zheng, Yule Liu, Xinlei He, Yu Wang, Ruiming Wang, Xinyi Huang, Lei Mo  

**Link**: [PDF](https://arxiv.org/pdf/2508.14377)  

**Abstract**: Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications. 

---
# Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency 

**Authors**: Aman Goel, Daniel Schwartz, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.14314)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages FINe-grained Cross-model consistency to detect and mitigate Hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves 7-8 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation across multiple models demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems. 

---
# Towards LLM-generated explanations for Component-based Knowledge Graph Question Answering Systems 

**Authors**: Dennis Schiese, Aleksandr Perevalov, Andreas Both  

**Link**: [PDF](https://arxiv.org/pdf/2508.14553)  

**Abstract**: Over time, software systems have reached a level of complexity that makes it difficult for their developers and users to explain particular decisions made by them. In this paper, we focus on the explainability of component-based systems for Question Answering (QA). These components often conduct processes driven by AI methods, in which behavior and decisions cannot be clearly explained or justified, s.t., even for QA experts interpreting the executed process and its results is hard. To address this challenge, we present an approach that considers the components' input and output data flows as a source for representing the behavior and provide explanations for the components, enabling users to comprehend what happened. In the QA framework used here, the data flows of the components are represented as SPARQL queries (inputs) and RDF triples (outputs). Hence, we are also providing valuable insights on verbalization regarding these data types. In our experiments, the approach generates explanations while following template-based settings (baseline) or via the use of Large Language Models (LLMs) with different configurations (automatic generation). Our evaluation shows that the explanations generated via LLMs achieve high quality and mostly outperform template-based approaches according to the users' ratings. Therefore, it enables us to automatically explain the behavior and decisions of QA components to humans while using RDF and SPARQL as a context for explanations. 

---
# Amortized Bayesian Meta-Learning for Low-Rank Adaptation of Large Language Models 

**Authors**: Liyi Zhang, Jake Snell, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2508.14285)  

**Abstract**: Fine-tuning large language models (LLMs) with low-rank adaptaion (LoRA) is a cost-effective way to incorporate information from a specific dataset. However, it is often unclear how well the fine-tuned LLM will generalize, i.e., how well it will perform on unseen datasets. Methods have been proposed to improve generalization by optimizing with in-context prompts, or by using meta-learning to fine-tune LLMs. However, these methods are expensive in memory and computation, requiring either long-context prompts or saving copies of parameters and using second-order gradient updates. To address these challenges, we propose Amortized Bayesian Meta-Learning for LoRA (ABMLL). This method builds on amortized Bayesian meta-learning for smaller models, adapting this approach to LLMs while maintaining its computational efficiency. We reframe task-specific and global parameters in the context of LoRA and use a set of new hyperparameters to balance reconstruction accuracy and the fidelity of task-specific parameters to the global ones. ABMLL provides effective generalization and scales to large models such as Llama3-8B. Furthermore, as a result of using a Bayesian framework, ABMLL provides improved uncertainty quantification. We test ABMLL on Unified-QA and CrossFit datasets and find that it outperforms existing methods on these benchmarks in terms of both accuracy and expected calibration error. 

---
# Organ-Agents: Virtual Human Physiology Simulator via LLMs 

**Authors**: Rihao Chang, He Jiao, Weizhi Nie, Honglin Guo, Keliang Xie, Zhenhua Wu, Lina Zhao, Yunpeng Bai, Yongtao Ma, Lanjun Wang, Yuting Su, Xi Gao, Weijie Wang, Nicu Sebe, Bruno Lepri, Bingwei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.14357)  

**Abstract**: Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care. 

---
# GLASS: Test-Time Acceleration for LLMs via Global-Local Neural Importance Aggregation 

**Authors**: Amirmohsen Sattarifard, Sepehr Lavasani, Ehsan Imani, Kunlin Zhang, Hanlin Xu, Fengyu Sun, Negar Hassanpour, Chao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14302)  

**Abstract**: Deploying Large Language Models (LLMs) on edge hardware demands aggressive, prompt-aware dynamic pruning to reduce computation without degrading quality. Static or predictor-based schemes either lock in a single sparsity pattern or incur extra runtime overhead, and recent zero-shot methods that rely on statistics from a single prompt fail on short prompt and/or long generation scenarios. We introduce A/I-GLASS: Activation- and Impact-based Global-Local neural importance Aggregation for feed-forward network SparSification, two training-free methods that dynamically select FFN units using a rank-aggregation of prompt local and model-intrinsic global neuron statistics. Empirical results across multiple LLMs and benchmarks demonstrate that GLASS significantly outperforms prior training-free methods, particularly in challenging long-form generation scenarios, without relying on auxiliary predictors or adding any inference overhead. 

---
# Entropy-Constrained Strategy Optimization in Urban Floods: A Multi-Agent Framework with LLM and Knowledge Graph Integration 

**Authors**: Peilin Ji, Xiao Xue, Simeng Wang, Wenhao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.14654)  

**Abstract**: In recent years, the increasing frequency of extreme urban rainfall events has posed significant challenges to emergency scheduling systems. Urban flooding often leads to severe traffic congestion and service disruptions, threatening public safety and mobility. However, effective decision making remains hindered by three key challenges: (1) managing trade-offs among competing goals (e.g., traffic flow, task completion, and risk mitigation) requires dynamic, context-aware strategies; (2) rapidly evolving environmental conditions render static rules inadequate; and (3) LLM-generated strategies frequently suffer from semantic instability and execution inconsistency. Existing methods fail to align perception, global optimization, and multi-agent coordination within a unified framework. To tackle these challenges, we introduce H-J, a hierarchical multi-agent framework that integrates knowledge-guided prompting, entropy-constrained generation, and feedback-driven optimization. The framework establishes a closed-loop pipeline spanning from multi-source perception to strategic execution and continuous refinement. We evaluate H-J on real-world urban topology and rainfall data under three representative conditions: extreme rainfall, intermittent bursts, and daily light rain. Experiments show that H-J outperforms rule-based and reinforcement-learning baselines in traffic smoothness, task success rate, and system robustness. These findings highlight the promise of uncertainty-aware, knowledge-constrained LLM-based approaches for enhancing resilience in urban flood response. 

---
# You Don't Know Until You Click:Automated GUI Testing for Production-Ready Software Evaluation 

**Authors**: Yutong Bian, Xianhao Lin, Yupeng Xie, Tianyang Liu, Mingchen Zhuge, Siyuan Lu, Haoming Tang, Jinlin Wang, Jiayi Zhang, Jiaqi Chen, Xiangru Tang, Yongxin Ni, Sirui Hong, Chenglin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14104)  

**Abstract**: Large Language Models (LLMs) and code agents in software development are rapidly evolving from generating isolated code snippets to producing full-fledged software applications with graphical interfaces, interactive logic, and dynamic behaviors. However, current benchmarks fall short in evaluating such production-ready software, as they often rely on static checks or binary pass/fail scripts, failing to capture the interactive behaviors and runtime dynamics that define real-world usability - qualities that only emerge when an application is actively used. This is the blind spot of current evaluation: you don't know if an app works until you click through it, interact with it, and observe how it responds. To bridge this gap, we introduce RealDevWorld, a novel evaluation framework for automated end-to-end assessment of LLMs' ability to generate production-ready repositories from scratch. It features two key components: (1) RealDevBench, a diverse collection of 194 open-ended software engineering tasks across multiple domains, incorporating multimodal elements to reflect real-world complexity; and (2) AppEvalPilot, a new agent-as-a-judge evaluation system that simulates realistic, GUI-based user interactions to automatically and holistically assess software functional correctness, visual fidelity, and runtime behavior. The framework delivers fine-grained, task-specific diagnostic feedback, supporting nuanced evaluation beyond simple success/failure judgments. Empirical results show that RealDevWorld delivers effective, automatic, and human-aligned evaluations, achieving an accuracy of 0.92 and a correlation of 0.85 with expert human assessments, while significantly reducing the reliance on manual review. This enables scalable, human-aligned assessment of production-level software generated by LLMs. Our code is available on GitHub. 

---
# PAPPL: Personalized AI-Powered Progressive Learning Platform 

**Authors**: Shayan Bafandkar, Sungyong Chung, Homa Khosravian, Alireza Talebpour  

**Link**: [PDF](https://arxiv.org/pdf/2508.14109)  

**Abstract**: Engineering education has historically been constrained by rigid, standardized frameworks, often neglecting students' diverse learning needs and interests. While significant advancements have been made in online and personalized education within K-12 and foundational sciences, engineering education at both undergraduate and graduate levels continues to lag in adopting similar innovations. Traditional evaluation methods, such as exams and homework assignments, frequently overlook individual student requirements, impeding personalized educational experiences. To address these limitations, this paper introduces the Personalized AI-Powered Progressive Learning (PAPPL) platform, an advanced Intelligent Tutoring System (ITS) designed specifically for engineering education. It highlights the development of a scalable, data-driven tutoring environment leveraging cutting-edge AI technology to enhance personalized learning across diverse academic disciplines, particularly in STEM fields. PAPPL integrates core ITS components including the expert module, student module, tutor module, and user interface, and utilizes GPT-4o, a sophisticated large language model (LLM), to deliver context-sensitive and pedagogically sound hints based on students' interactions. The system uniquely records student attempts, detects recurring misconceptions, and generates progressively targeted feedback, providing personalized assistance that adapts dynamically to each student's learning profile. Additionally, PAPPL offers instructors detailed analytics, empowering evidence-based adjustments to teaching strategies. This study provides a fundamental framework for the progression of Generative ITSs scalable to all education levels, delivering important perspectives on personalized progressive learning and the wider possibilities of Generative AI in the field of education. 

---
# Your Reward Function for RL is Your Best PRM for Search: Unifying RL and Search-Based TTS 

**Authors**: Can Jin, Yang Zhou, Qixin Zhang, Hongwu Peng, Di Zhang, Marco Pavone, Ligong Han, Zhang-Wei Hong, Tong Che, Dimitris N. Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2508.14313)  

**Abstract**: Test-time scaling (TTS) for large language models (LLMs) has thus far fallen into two largely separate paradigms: (1) reinforcement learning (RL) methods that optimize sparse outcome-based rewards, yet suffer from instability and low sample efficiency; and (2) search-based techniques guided by independently trained, static process reward models (PRMs), which require expensive human- or LLM-generated labels and often degrade under distribution shifts. In this paper, we introduce AIRL-S, the first natural unification of RL-based and search-based TTS. Central to AIRL-S is the insight that the reward function learned during RL training inherently represents the ideal PRM for guiding downstream search. Specifically, we leverage adversarial inverse reinforcement learning (AIRL) combined with group relative policy optimization (GRPO) to learn a dense, dynamic PRM directly from correct reasoning traces, entirely eliminating the need for labeled intermediate process data. At inference, the resulting PRM simultaneously serves as the critic for RL rollouts and as a heuristic to effectively guide search procedures, facilitating robust reasoning chain extension, mitigating reward hacking, and enhancing cross-task generalization. Experimental results across eight benchmarks, including mathematics, scientific reasoning, and code generation, demonstrate that our unified approach improves performance by 9 % on average over the base model, matching GPT-4o. Furthermore, when integrated into multiple search algorithms, our PRM consistently outperforms all baseline PRMs trained with labeled data. These results underscore that, indeed, your reward function for RL is your best PRM for search, providing a robust and cost-effective solution to complex reasoning tasks in LLMs. 

---
# Hard Examples Are All You Need: Maximizing GRPO Post-Training Under Annotation Budgets 

**Authors**: Benjamin Pikus, Pratyush Ranjan Tiwari, Burton Ye  

**Link**: [PDF](https://arxiv.org/pdf/2508.14094)  

**Abstract**: Collecting high-quality training examples for language model fine-tuning is expensive, with practical budgets limiting the amount of data that can be procured. We investigate a critical question for resource-constrained alignment: under a fixed acquisition budget, should practitioners prioritize examples that are easy, medium, hard, or of random difficulty? We study Group Relative Policy Optimization (GRPO) fine-tuning across different model sizes and families, comparing four subset selection policies chosen from the same unlabeled pool using base-model difficulty estimates obtained via multi-sample evaluation. Our experiments reveal that training on the hardest examples yields the largest performance gains, up to 47%, while training on easy examples yield the smallest gains. Analysis reveals that this effect arises from harder examples providing more learnable opportunities during GRPO training. These findings provide practical guidance for budget-constrained post-training: prioritizing hard examples yields substantial performance gains on reasoning tasks when using GRPO. 

---
# DLLMQuant: Quantizing Diffusion-based Large Language Models 

**Authors**: Chen Xu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14090)  

**Abstract**: Diffusion-based large language models (DLLMs) have shown promise for non-autoregressive text generation, but their deployment is constrained by large model sizes and heavy computational costs. Post-training quantization (PTQ), a widely used method for compressing and accelerating Large Language Models (LLMs), suffers from severe accuracy degradation and reduced generalization performance when directly applied to DLLMs (e.g., AWQ suffers a 16% accuracy drop on LLADA under W4A4). This paper explores how DLLMs' key mechanisms - dynamic masking, iterative generation, bidirectional attention - clash with quantization. We identify three core issues: 1) Iterative generation and dynamic masking ratios lead to distinct token distributions across decoding steps, which are not adequately captured by existing PTQ calibration methods; 2) Quantization errors are accumulated and amplified progressively during iteration in DLLMs, causing quantized models to perform worse as decoding steps progress; 3) Unmasked tokens stabilize while masked remain probabilistic, making overall feature distribution incompatible with existing PTQ methods. To address these issues, we propose DLLMQuant, a PTQ framework tailored for DLLMs, which incorporates three novel techniques: 1) Temporal-Mask Adaptive Sampling (TMAS), a calibration method that accounts for both time and mask factors, with the capacity to capture distributions across timesteps. 2) Interaction-Aware Activation Quantization (IA-AQ), which utilizes bidirectional attention's interaction signals to dynamically allocate quantization resources. 3) Certainty-Guided Quantization (CGQ), which integrates mask status and token scores as key weighting criteria into error compensation, making weight quantization more suitable for DLLMs. Experiments show that DLLMQuant achieves significant performance gains while enhancing efficiency. 

---
# RynnEC: Bringing MLLMs into Embodied World 

**Authors**: Ronghao Dang, Yuqian Yuan, Yunxuan Mao, Kehan Li, Jiangpin Liu, Zhikai Wang, Xin Li, Fan Wang, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.14160)  

**Abstract**: We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: this https URL 

---
# A Multi-Agent Approach to Neurological Clinical Reasoning 

**Authors**: Moran Sorka, Alon Gorenshtein, Dvir Aran, Shahar Shelly  

**Link**: [PDF](https://arxiv.org/pdf/2508.14063)  

**Abstract**: Large language models (LLMs) have shown promise in medical domains, but their ability to handle specialized neurological reasoning requires systematic evaluation. We developed a comprehensive benchmark using 305 questions from Israeli Board Certification Exams in Neurology, classified along three complexity dimensions: factual knowledge depth, clinical concept integration, and reasoning complexity. We evaluated ten LLMs using base models, retrieval-augmented generation (RAG), and a novel multi-agent system. Results showed significant performance variation. OpenAI-o1 achieved the highest base performance (90.9% accuracy), while specialized medical models performed poorly (52.9% for Meditron-70B). RAG provided modest benefits but limited effectiveness on complex reasoning questions. In contrast, our multi-agent framework, decomposing neurological reasoning into specialized cognitive functions including question analysis, knowledge retrieval, answer synthesis, and validation, achieved dramatic improvements, especially for mid-range models. The LLaMA 3.3-70B-based agentic system reached 89.2% accuracy versus 69.5% for its base model, with substantial gains on level 3 complexity questions. The multi-agent approach transformed inconsistent subspecialty performance into uniform excellence, addressing neurological reasoning challenges that persisted with RAG enhancement. We validated our approach using an independent dataset of 155 neurological cases from MedQA. Results confirm that structured multi-agent approaches designed to emulate specialized cognitive processes significantly enhance complex medical reasoning, offering promising directions for AI assistance in challenging clinical contexts. 

---
# T-REX: Table -- Refute or Entail eXplainer 

**Authors**: Tim Luka Horstmann, Baptiste Geisenberger, Mehwish Alam  

**Link**: [PDF](https://arxiv.org/pdf/2508.14055)  

**Abstract**: Verifying textual claims against structured tabular data is a critical yet challenging task in Natural Language Processing with broad real-world impact. While recent advances in Large Language Models (LLMs) have enabled significant progress in table fact-checking, current solutions remain inaccessible to non-experts. We introduce T-REX (T-REX: Table -- Refute or Entail eXplainer), the first live, interactive tool for claim verification over multimodal, multilingual tables using state-of-the-art instruction-tuned reasoning LLMs. Designed for accuracy and transparency, T-REX empowers non-experts by providing access to advanced fact-checking technology. The system is openly available online. 

---
# FinAgentBench: A Benchmark Dataset for Agentic Retrieval in Financial Question Answering 

**Authors**: Chanyeol Choi, Jihoon Kwon, Alejandro Lopez-Lira, Chaewoon Kim, Minjae Kim, Juneha Hwang, Jaeseon Ha, Hojun Choi, Suyeol Yun, Yongjin Kim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.14052)  

**Abstract**: Accurate information retrieval (IR) is critical in the financial domain, where investors must identify relevant information from large collections of documents. Traditional IR methods-whether sparse or dense-often fall short in retrieval accuracy, as it requires not only capturing semantic similarity but also performing fine-grained reasoning over document structure and domain-specific knowledge. Recent advances in large language models (LLMs) have opened up new opportunities for retrieval with multi-step reasoning, where the model ranks passages through iterative reasoning about which information is most relevant to a given query. However, there exists no benchmark to evaluate such capabilities in the financial domain. To address this gap, we introduce FinAgentBench, the first large-scale benchmark for evaluating retrieval with multi-step reasoning in finance -- a setting we term agentic retrieval. The benchmark consists of 3,429 expert-annotated examples on S&P-100 listed firms and assesses whether LLM agents can (1) identify the most relevant document type among candidates, and (2) pinpoint the key passage within the selected document. Our evaluation framework explicitly separates these two reasoning steps to address context limitations. This design enables to provide a quantitative basis for understanding retrieval-centric LLM behavior in finance. We evaluate a suite of state-of-the-art models and further demonstrated how targeted fine-tuning can significantly improve agentic retrieval performance. Our benchmark provides a foundation for studying retrieval-centric LLM behavior in complex, domain-specific tasks for finance. We will release the dataset publicly upon acceptance of the paper and plan to expand and share dataset for the full S&P 500 and beyond. 

---
# An automatic patent literature retrieval system based on LLM-RAG 

**Authors**: Yao Ding, Yuqing Wu, Ziyang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14064)  

**Abstract**: With the acceleration of technological innovation efficient retrieval and classification of patent literature have become essential for intellectual property management and enterprise RD Traditional keyword and rulebased retrieval methods often fail to address complex query intents or capture semantic associations across technical domains resulting in incomplete and lowrelevance results This study presents an automated patent retrieval framework integrating Large Language Models LLMs with RetrievalAugmented Generation RAG technology The system comprises three components: 1) a preprocessing module for patent data standardization, 2) a highefficiency vector retrieval engine leveraging LLMgenerated embeddings, and 3) a RAGenhanced query module that combines external document retrieval with contextaware response generation Evaluations were conducted on the Google Patents dataset 20062024 containing millions of global patent records with metadata such as filing date domain and status The proposed gpt35turbo0125RAG configuration achieved 805 semantic matching accuracy and 92.1% recall surpassing baseline LLM methods by 28 percentage points The framework also demonstrated strong generalization in crossdomain classification and semantic clustering tasks These results validate the effectiveness of LLMRAG integration for intelligent patent retrieval providing a foundation for nextgeneration AIdriven intellectual property analysis platforms 

---
# The Hidden Cost of Readability: How Code Formatting Silently Consumes Your LLM Budget 

**Authors**: Dangfeng Pan, Zhensu Sun, Cenyuan Zhang, David Lo, Xiaoning Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.13666)  

**Abstract**: Source code is usually formatted with elements like indentation and newlines to improve readability for human developers. However, these visual aids do not seem to be beneficial for large language models (LLMs) in the same way since the code is processed as a linear sequence of tokens. Furthermore, these additional tokens can lead to increased computational costs and longer response times for LLMs. If such formatting elements are non-essential to LLMs, we can reduce such costs by removing them from the code. To figure out the role played by formatting elements, we conduct a comprehensive empirical study to evaluate the impact of code formatting on LLM performance and efficiency. Through large-scale experiments on Fill-in-the-Middle Code Completion tasks across four programming languages (Java, Python, C++, C\#) and ten LLMs-including both commercial and open-source models-we systematically analyze token count and performance when formatting elements are removed. Key findings indicate that LLMs can maintain performance across formatted code and unformatted code, achieving an average input token reduction of 24.5\% with negligible output token reductions. This makes code format removal a practical optimization strategy for improving LLM efficiency. Further exploration reveals that both prompting and fine-tuning LLMs can lead to significant reductions (up to 36.1\%) in output code length without compromising correctness. To facilitate practical applications, we develop a bidirectional code transformation tool for format processing, which can be seamlessly integrated into existing LLM inference workflows, ensuring both human readability and LLM efficiency. 

---
# MAHL: Multi-Agent LLM-Guided Hierarchical Chiplet Design with Adaptive Debugging 

**Authors**: Jinwei Tang, Jiayin Qin, Nuo Xu, Pragnya Sudershan Nalla, Yu Cao, Yang, Zhao, Caiwen Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14053)  

**Abstract**: As program workloads (e.g., AI) increase in size and algorithmic complexity, the primary challenge lies in their high dimensionality, encompassing computing cores, array sizes, and memory hierarchies. To overcome these obstacles, innovative approaches are required. Agile chip design has already benefited from machine learning integration at various stages, including logic synthesis, placement, and routing. With Large Language Models (LLMs) recently demonstrating impressive proficiency in Hardware Description Language (HDL) generation, it is promising to extend their abilities to 2.5D integration, an advanced technique that saves area overhead and development costs. However, LLM-driven chiplet design faces challenges such as flatten design, high validation cost and imprecise parameter optimization, which limit its chiplet design capability. To address this, we propose MAHL, a hierarchical LLM-based chiplet design generation framework that features six agents which collaboratively enable AI algorithm-hardware mapping, including hierarchical description generation, retrieval-augmented code generation, diverseflow-based validation, and multi-granularity design space exploration. These components together enhance the efficient generation of chiplet design with optimized Power, Performance and Area (PPA). Experiments show that MAHL not only significantly improves the generation accuracy of simple RTL design, but also increases the generation accuracy of real-world chiplet design, evaluated by Pass@5, from 0 to 0.72 compared to conventional LLMs under the best-case scenario. Compared to state-of-the-art CLARIE (expert-based), MAHL achieves comparable or even superior PPA results under certain optimization objectives. 

---
# Improving in-context learning with a better scoring function 

**Authors**: Omar Naim, Swarnadeep Bhar, Jérôme Bolte, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2508.14685)  

**Abstract**: Large language models (LLMs) exhibit a remarkable capacity to learn by analogy, known as in-context learning (ICL). However, recent studies have revealed limitations in this ability. In this paper, we examine these limitations on tasks involving first-order quantifiers such as {\em all} and {\em some}, as well as on ICL with linear functions. We identify Softmax, the scoring function in attention mechanism, as a contributing factor to these constraints. To address this, we propose \textbf{scaled signed averaging (SSA)}, a novel alternative to Softmax. Empirical results show that SSA dramatically improves performance on our target tasks. Furthermore, we evaluate both encoder-only and decoder-only transformers models with SSA, demonstrating that they match or exceed their Softmax-based counterparts across a variety of linguistic probing tasks. 

---
# Knowledge Graph-Infused Fine-Tuning for Structured Reasoning in Large Language Models 

**Authors**: Wuyang Zhang, Yexin Tian, Xiandong Meng, Mengjie Wang, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.14427)  

**Abstract**: This paper addresses the problems of missing reasoning chains and insufficient entity-level semantic understanding in large language models when dealing with tasks that require structured knowledge. It proposes a fine-tuning algorithm framework based on knowledge graph injection. The method builds on pretrained language models and introduces structured graph information for auxiliary learning. A graph neural network is used to encode entities and their relations, constructing a graph-based semantic representation. A fusion mechanism is then designed to jointly model the knowledge graph embeddings with the contextual representations from the language model. To enhance the robustness of knowledge integration, a gating mechanism is introduced to dynamically balance the contributions of linguistic semantics and structural knowledge. This effectively mitigates conflicts between different representational spaces. During training, a joint loss function is constructed to account for both task performance and structural alignment objectives. This helps improve the accuracy of entity prediction and semantic reasoning. The study also includes a series of systematic sensitivity experiments. It evaluates the effects of learning rate, graph coverage, and structural perturbations on model performance. The results further validate the effectiveness and stability of the proposed method across tasks such as entity recognition, question answering, and language generation. Experimental findings show that the proposed structure-aware fine-tuning framework significantly enhances the model's ability to represent complex semantic units. It demonstrates better semantic consistency and contextual logic modeling in scenarios involving structural reasoning and entity extraction. 

---
# SurveyGen-I: Consistent Scientific Survey Generation with Evolving Plans and Memory-Guided Writing 

**Authors**: Jing Chen, Zhiheng Yang, Yixian Shen, Jie Liu, Adam Belloum, Chrysa Papagainni, Paola Grosso  

**Link**: [PDF](https://arxiv.org/pdf/2508.14317)  

**Abstract**: Survey papers play a critical role in scientific communication by consolidating progress across a field. Recent advances in Large Language Models (LLMs) offer a promising solution by automating key steps in the survey-generation pipeline, such as retrieval, structuring, and summarization. However, existing LLM-based approaches often struggle with maintaining coherence across long, multi-section surveys and providing comprehensive citation coverage. To address these limitations, we introduce SurveyGen-I, an automatic survey generation framework that combines coarse-to-fine retrieval, adaptive planning, and memory-guided generation. SurveyGen-I first performs survey-level retrieval to construct the initial outline and writing plan, and then dynamically refines both during generation through a memory mechanism that stores previously written content and terminology, ensuring coherence across subsections. When the system detects insufficient context, it triggers fine-grained subsection-level retrieval. During generation, SurveyGen-I leverages this memory mechanism to maintain coherence across subsections. Experiments across four scientific domains demonstrate that SurveyGen-I consistently outperforms previous works in content quality, consistency, and citation coverage. 

---
# Beyond Semantic Similarity: Reducing Unnecessary API Calls via Behavior-Aligned Retriever 

**Authors**: Yixin Chen, Ying Xiong, Shangyu Wu, Yufei Cui, Xue Liu, Nan Guan, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.14323)  

**Abstract**: Tool-augmented large language models (LLMs) leverage external functions to extend their capabilities, but inaccurate function calls can lead to inefficiencies and increased this http URL methods address this challenge by fine-tuning LLMs or using demonstration-based prompting, yet they often suffer from high training overhead and fail to account for inconsistent demonstration samples, which misguide the model's invocation behavior. In this paper, we trained a behavior-aligned retriever (BAR), which provides behaviorally consistent demonstrations to help LLMs make more accurate tool-using decisions. To train the BAR, we construct a corpus including different function-calling behaviors, i.e., calling or this http URL use the contrastive learning framework to train the BAR with customized positive/negative pairs and a dual-negative contrastive loss, ensuring robust retrieval of behaviorally consistent this http URL demonstrate that our approach significantly reduces erroneous function calls while maintaining high task performance, offering a cost-effective and efficient solution for tool-augmented LLMs. 

---
# Let's Use ChatGPT To Write Our Paper! Benchmarking LLMs To Write the Introduction of a Research Paper 

**Authors**: Krishna Garg, Firoz Shaikh, Sambaran Bandyopadhyay, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2508.14273)  

**Abstract**: As researchers increasingly adopt LLMs as writing assistants, generating high-quality research paper introductions remains both challenging and essential. We introduce Scientific Introduction Generation (SciIG), a task that evaluates LLMs' ability to produce coherent introductions from titles, abstracts, and related works. Curating new datasets from NAACL 2025 and ICLR 2025 papers, we assess five state-of-the-art models, including both open-source (DeepSeek-v3, Gemma-3-12B, LLaMA 4-Maverick, MistralAI Small 3.1) and closed-source GPT-4o systems, across multiple dimensions: lexical overlap, semantic similarity, content coverage, faithfulness, consistency, citation correctness, and narrative quality. Our comprehensive framework combines automated metrics with LLM-as-a-judge evaluations. Results demonstrate LLaMA-4 Maverick's superior performance on most metrics, particularly in semantic similarity and faithfulness. Moreover, three-shot prompting consistently outperforms fewer-shot approaches. These findings provide practical insights into developing effective research writing assistants and set realistic expectations for LLM-assisted academic writing. To foster reproducibility and future research, we will publicly release all code and datasets. 

---
# GRILE: A Benchmark for Grammar Reasoning and Explanation in Romanian LLMs 

**Authors**: Adrian-Marius Dumitran, Alexandra-Mihaela Danila, Angela-Liliana Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2508.14279)  

**Abstract**: LLMs (Large language models) have revolutionized NLP (Natural Language Processing), yet their pedagogical value for low-resource languages remains unclear. We present GRILE (Grammar Romanian Inference and Language Explanations) , the first open benchmark of 1,151 multiple-choice questions harvested from Romanian high-stakes exams (National Evaluation, Baccalaureate, university admissions). GRILE enables us to probe two complementary abilities of seven state-of-the-art multilingual and Romanian-specific LLMs: (i) selecting the correct answer, and (ii) producing linguistically accurate explanations. While Gemini 2.5 Pro reaches 83% accuracy, most open-weight models stay below 65%, and 48% of their explanations contain factual or pedagogical flaws according to expert review. A detailed error analysis pinpoints systematic weaknesses in morphology and in applying the latest DOOM3 orthographic norms. All data, code and a public web demo are released to catalyze future research. Our findings expose open challenges for trustworthy educational NLP in low-resource settings and establish GRILE as a new test-bed for controllable explanation generation and evaluation. 

---
# DPad: Efficient Diffusion Language Models with Suffix Dropout 

**Authors**: Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai "Hellen" Li, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.14148)  

**Abstract**: Diffusion-based Large Language Models (dLLMs) parallelize text generation by framing decoding as a denoising process, but suffer from high computational overhead since they predict all future suffix tokens at each step while retaining only a small fraction. We propose Diffusion Scratchpad (DPad), a training-free method that restricts attention to a small set of nearby suffix tokens, preserving fidelity while eliminating redundancy. DPad integrates two strategies: (i) a sliding window, which maintains a fixed-length suffix window, and (ii) distance-decay dropout, which deterministically removes distant suffix tokens before attention computation. This simple design is compatible with existing optimizations such as prefix caching and can be implemented with only a few lines of code. Comprehensive evaluations across multiple benchmarks on LLaDA-1.5 and Dream models demonstrate that DPad delivers up to $\mathbf{61.4\times}$ speedup over vanilla dLLMs while maintaining comparable accuracy, highlighting its potential for efficient and scalable long-sequence inference. Our code is available at this https URL. 

---
# Confidence Estimation for Text-to-SQL in Large Language Models 

**Authors**: Sepideh Entezari Maleki, Mohammadreza Pourreza, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2508.14056)  

**Abstract**: Confidence estimation for text-to-SQL aims to assess the reliability of model-generated SQL queries without having access to gold answers. We study this problem in the context of large language models (LLMs), where access to model weights and gradients is often constrained. We explore both black-box and white-box confidence estimation strategies, evaluating their effectiveness on cross-domain text-to-SQL benchmarks. Our evaluation highlights the superior performance of consistency-based methods among black-box models and the advantage of SQL-syntax-aware approaches for interpreting LLM logits in white-box settings. Furthermore, we show that execution-based grounding of queries provides a valuable supplementary signal, improving the effectiveness of both approaches. 

---
# Disentangling concept semantics via multilingual averaging in Sparse Autoencoders 

**Authors**: Cliff O'Reilly, Ernesto Jimenez-Ruiz, Tillman Weyde  

**Link**: [PDF](https://arxiv.org/pdf/2508.14275)  

**Abstract**: Connecting LLMs with formal knowledge representation and reasoning is a promising approach to address their shortcomings. Embeddings and sparse autoencoders are widely used to represent textual content, but the semantics are entangled with syntactic and language-specific information. We propose a method that isolates concept semantics in Large Langue Models by averaging concept activations derived via Sparse Autoencoders. We create English text representations from OWL ontology classes, translate the English into French and Chinese and then pass these texts as prompts to the Gemma 2B LLM. Using the open source Gemma Scope suite of Sparse Autoencoders, we obtain concept activations for each class and language version. We average the different language activations to derive a conceptual average. We then correlate the conceptual averages with a ground truth mapping between ontology classes. Our results give a strong indication that the conceptual average aligns to the true relationship between classes when compared with a single language by itself. The result hints at a new technique which enables mechanistic interpretation of internal network states with higher accuracy. 

---
# MedReseacher-R1: Expert-Level Medical Deep Researcher via A Knowledge-Informed Trajectory Synthesis Framework 

**Authors**: Ailing Yu, Lan Yao, Jingnan Liu, Zhe Chen, Jiajun Yin, Yuan Wang, Xinhao Liao, Zhiling Ye, Ji Li, Yun Yue, Hansong Xiao, Hualei Zhou, Chunxiao Guo, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14880)  

**Abstract**: Recent developments in Large Language Model (LLM)-based agents have shown impressive capabilities spanning multiple domains, exemplified by deep research systems that demonstrate superior performance on complex information-seeking and synthesis tasks. While general-purpose deep research agents have shown impressive capabilities, they struggle significantly with medical domain challenges, as evidenced by leading proprietary systems achieving limited accuracy on complex medical benchmarks. The key limitations are: (1) the model lacks sufficient dense medical knowledge for clinical reasoning, and (2) the framework is constrained by the absence of specialized retrieval tools tailored for medical this http URL present a medical deep research agent that addresses these challenges through two core innovations. First, we develop a novel data synthesis framework using medical knowledge graphs, extracting the longest chains from subgraphs around rare medical entities to generate complex multi-hop question-answer pairs. Second, we integrate a custom-built private medical retrieval engine alongside general-purpose tools, enabling accurate medical information synthesis. Our approach generates 2100+ diverse trajectories across 12 medical specialties, each averaging 4.2 tool this http URL a two-stage training paradigm combining supervised fine-tuning and online reinforcement learning with composite rewards, our MedResearcher-R1-32B model demonstrates exceptional performance, establishing new state-of-the-art results on medical benchmarks while maintaining competitive performance on general deep research tasks. Our work demonstrates that strategic domain-specific innovations in architecture, tool design, and training data construction can enable smaller open-source models to outperform much larger proprietary systems in specialized domains. 

---
# The Prompting Brain: Neurocognitive Markers of Expertise in Guiding Large Language Models 

**Authors**: Hend Al-Khalifa, Raneem Almansour, Layan Abdulrahman Alhuasini, Alanood Alsaleh, Mohamad-Hani Temsah, Mohamad-Hani_Temsah, Ashwag Rafea S Alruwaili  

**Link**: [PDF](https://arxiv.org/pdf/2508.14869)  

**Abstract**: Prompt engineering has rapidly emerged as a critical skill for effective interaction with large language models (LLMs). However, the cognitive and neural underpinnings of this expertise remain largely unexplored. This paper presents findings from a cross-sectional pilot fMRI study investigating differences in brain functional connectivity and network activity between experts and intermediate prompt engineers. Our results reveal distinct neural signatures associated with higher prompt engineering literacy, including increased functional connectivity in brain regions such as the left middle temporal gyrus and the left frontal pole, as well as altered power-frequency dynamics in key cognitive networks. These findings offer initial insights into the neurobiological basis of prompt engineering proficiency. We discuss the implications of these neurocognitive markers in Natural Language Processing (NLP). Understanding the neural basis of human expertise in interacting with LLMs can inform the design of more intuitive human-AI interfaces, contribute to cognitive models of LLM interaction, and potentially guide the development of AI systems that better align with human cognitive workflows. This interdisciplinary approach aims to bridge the gap between human cognition and machine intelligence, fostering a deeper understanding of how humans learn and adapt to complex AI systems. 

---
# DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization 

**Authors**: Shuaijie She, Yu Bao, Yu Lu, Lu Xu, Tao Li, Wenhao Zhu, Shujian Huang, Shanbo Cheng, Lu Lu, Yuxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.14460)  

**Abstract**: We present DuPO, a dual learning-based preference optimization framework that generates annotation-free feedback via a generalized duality. DuPO addresses two key limitations: Reinforcement Learning with Verifiable Rewards (RLVR)'s reliance on costly labels and applicability restricted to verifiable tasks, and traditional dual learning's restriction to strictly dual task pairs (e.g., translation and back-translation). Specifically, DuPO decomposes a primal task's input into known and unknown components, then constructs its dual task to reconstruct the unknown part using the primal output and known information (e.g., reversing math solutions to recover hidden variables), broadening applicability to non-invertible tasks. The quality of this reconstruction serves as a self-supervised reward to optimize the primal task, synergizing with LLMs' ability to instantiate both tasks via a single model. Empirically, DuPO achieves substantial gains across diverse tasks: it enhances the average translation quality by 2.13 COMET over 756 directions, boosts the mathematical reasoning accuracy by an average of 6.4 points on three challenge benchmarks, and enhances performance by 9.3 points as an inference-time reranker (trading computation for accuracy). These results position DuPO as a scalable, general, and annotation-free paradigm for LLM optimization. 

---
# Punctuation and Predicates in Language Models 

**Authors**: Sonakshi Chauhan, Maheep Chaudhary, Koby Choy, Samuel Nellessen, Nandi Schoots  

**Link**: [PDF](https://arxiv.org/pdf/2508.14067)  

**Abstract**: In this paper we explore where information is collected and how it is propagated throughout layers in large language models (LLMs). We begin by examining the surprising computational importance of punctuation tokens which previous work has identified as attention sinks and memory aids. Using intervention-based techniques, we evaluate the necessity and sufficiency (for preserving model performance) of punctuation tokens across layers in GPT-2, DeepSeek, and Gemma. Our results show stark model-specific differences: for GPT-2, punctuation is both necessary and sufficient in multiple layers, while this holds far less in DeepSeek and not at all in Gemma. Extending beyond punctuation, we ask whether LLMs process different components of input (e.g., subjects, adjectives, punctuation, full sentences) by forming early static summaries reused across the network, or if the model remains sensitive to changes in these components across layers. Extending beyond punctuation, we investigate whether different reasoning rules are processed differently by LLMs. In particular, through interchange intervention and layer-swapping experiments, we find that conditional statements (if, then), and universal quantification (for all) are processed very differently. Our findings offer new insight into the internal mechanisms of punctuation usage and reasoning in LLMs and have implications for interpretability. 

---
# RAG-Boost: Retrieval-Augmented Generation Enhanced LLM-based Speech Recognition 

**Authors**: Pengcheng Wang, Sheng Li, Takahiro Shinozaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.14048)  

**Abstract**: In this paper, we propose RAG-Boost (ST-ShinozakiLab Task I system), which enhances the baseline LLM-based ASR system of the MLC-SLM Challenge (task I) with a retrieval-augmented generation (RAG) module on the fly. Each partial ASR hypothesis queries a vector store of audio-text pairs and domain terms, and the retrieved results are fused with the live ASR hypotheses to fix recognition errors. The fused hypotheses are passed to the LLM, yielding improved responses. 

---
# Two Birds with One Stone: Multi-Task Detection and Attribution of LLM-Generated Text 

**Authors**: Zixin Rao, Youssef Mohamed, Shang Liu, Zeyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14190)  

**Abstract**: Large Language Models (LLMs), such as GPT-4 and Llama, have demonstrated remarkable abilities in generating natural language. However, they also pose security and integrity challenges. Existing countermeasures primarily focus on distinguishing AI-generated content from human-written text, with most solutions tailored for English. Meanwhile, authorship attribution--determining which specific LLM produced a given text--has received comparatively little attention despite its importance in forensic analysis. In this paper, we present DA-MTL, a multi-task learning framework that simultaneously addresses both text detection and authorship attribution. We evaluate DA-MTL on nine datasets and four backbone models, demonstrating its strong performance across multiple languages and LLM sources. Our framework captures each task's unique characteristics and shares insights between them, which boosts performance in both tasks. Additionally, we conduct a thorough analysis of cross-modal and cross-lingual patterns and assess the framework's robustness against adversarial obfuscation techniques. Our findings offer valuable insights into LLM behavior and the generalization of both detection and authorship attribution. 

---
# MMReview: A Multidisciplinary and Multimodal Benchmark for LLM-Based Peer Review Automation 

**Authors**: Xian Gao, Jiacheng Ruan, Zongyun Zhang, Jingsheng Gao, Ting Liu, Yuzhuo Fu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14146)  

**Abstract**: With the rapid growth of academic publications, peer review has become an essential yet time-consuming responsibility within the research community. Large Language Models (LLMs) have increasingly been adopted to assist in the generation of review comments; however, current LLM-based review tasks lack a unified evaluation benchmark to rigorously assess the models' ability to produce comprehensive, accurate, and human-aligned assessments, particularly in scenarios involving multimodal content such as figures and tables. To address this gap, we propose \textbf{MMReview}, a comprehensive benchmark that spans multiple disciplines and modalities. MMReview includes multimodal content and expert-written review comments for 240 papers across 17 research domains within four major academic disciplines: Artificial Intelligence, Natural Sciences, Engineering Sciences, and Social Sciences. We design a total of 13 tasks grouped into four core categories, aimed at evaluating the performance of LLMs and Multimodal LLMs (MLLMs) in step-wise review generation, outcome formulation, alignment with human preferences, and robustness to adversarial input manipulation. Extensive experiments conducted on 16 open-source models and 5 advanced closed-source models demonstrate the thoroughness of the benchmark. We envision MMReview as a critical step toward establishing a standardized foundation for the development of automated peer review systems. 

---
# Measuring LLM Code Generation Stability via Structural Entropy 

**Authors**: Yewei Song, Tiezhu Sun, Xunzhu Tang, Prateek Rajput, Tegawende F. Bissyande, Jacques Klein  

**Link**: [PDF](https://arxiv.org/pdf/2508.14288)  

**Abstract**: Assessing the stability of code generation from large language models (LLMs) is essential for judging their reliability in real-world development. We extend prior "structural-entropy concepts" to the program domain by pairing entropy with abstract syntax tree (AST) analysis. For any fixed prompt, we collect the multiset of depth-bounded subtrees of AST in each generated program and treat their relative frequencies as a probability distribution. We then measure stability in two complementary ways: (i) Jensen-Shannon divergence, a symmetric, bounded indicator of structural overlap, and (ii) a Structural Cross-Entropy ratio that highlights missing high-probability patterns. Both metrics admit structural-only and token-aware variants, enabling separate views on control-flow shape and identifier-level variability. Unlike pass@k, BLEU, or CodeBLEU, our metrics are reference-free, language-agnostic, and execution-independent. We benchmark several leading LLMs on standard code generation tasks, demonstrating that AST-driven structural entropy reveals nuances in model consistency and robustness. The method runs in O(n,d) time with no external tests, providing a lightweight addition to the code-generation evaluation toolkit. 

---
