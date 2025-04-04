# Generative Evaluation of Complex Reasoning in Large Language Models 

**Authors**: Haowei Lin, Xiangyu Wang, Ruilin Yan, Baizhou Huang, Haotian Ye, Jianhua Zhu, Zihao Wang, James Zou, Jianzhu Ma, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02810)  

**Abstract**: With powerful large language models (LLMs) demonstrating superhuman reasoning capabilities, a critical question arises: Do LLMs genuinely reason, or do they merely recall answers from their extensive, web-scraped training datasets? Publicly released benchmarks inevitably become contaminated once incorporated into subsequent LLM training sets, undermining their reliability as faithful assessments. To address this, we introduce KUMO, a generative evaluation framework designed specifically for assessing reasoning in LLMs. KUMO synergistically combines LLMs with symbolic engines to dynamically produce diverse, multi-turn reasoning tasks that are partially observable and adjustable in difficulty. Through an automated pipeline, KUMO continuously generates novel tasks across open-ended domains, compelling models to demonstrate genuine generalization rather than memorization. We evaluated 23 state-of-the-art LLMs on 5,000 tasks across 100 domains created by KUMO, benchmarking their reasoning abilities against university students. Our findings reveal that many LLMs have outperformed university-level performance on easy reasoning tasks, and reasoning-scaled LLMs reach university-level performance on complex reasoning challenges. Moreover, LLM performance on KUMO tasks correlates strongly with results on newly released real-world reasoning benchmarks, underscoring KUMO's value as a robust, enduring assessment tool for genuine LLM reasoning capabilities. 

---
# MegaMath: Pushing the Limits of Open Math Corpora 

**Authors**: Fan Zhou, Zengzhi Wang, Nikhil Ranjan, Zhoujun Cheng, Liping Tang, Guowei He, Zhengzhong Liu, Eric P. Xing  

**Link**: [PDF](https://arxiv.org/pdf/2504.02807)  

**Abstract**: Mathematical reasoning is a cornerstone of human intelligence and a key benchmark for advanced capabilities in large language models (LLMs). However, the research community still lacks an open, large-scale, high-quality corpus tailored to the demands of math-centric LLM pre-training. We present MegaMath, an open dataset curated from diverse, math-focused sources through following practices: (1) Revisiting web data: We re-extracted mathematical documents from Common Crawl with math-oriented HTML optimizations, fasttext-based filtering and deduplication, all for acquiring higher-quality data on the Internet. (2) Recalling Math-related code data: We identified high quality math-related code from large code training corpus, Stack-V2, further enhancing data diversity. (3) Exploring Synthetic data: We synthesized QA-style text, math-related code, and interleaved text-code blocks from web data or code data. By integrating these strategies and validating their effectiveness through extensive ablations, MegaMath delivers 371B tokens with the largest quantity and top quality among existing open math pre-training datasets. 

---
# A Survey of Large Language Models in Mental Health Disorder Detection on Social Media 

**Authors**: Zhuohan Ge, Nicole Hu, Darian Li, Yubo Wang, Shihao Qi, Yuming Xu, Han Shi, Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02800)  

**Abstract**: The detection and intervention of mental health issues represent a critical global research focus, and social media data has been recognized as an important resource for mental health research. However, how to utilize Large Language Models (LLMs) for mental health problem detection on social media poses significant challenges. Hence, this paper aims to explore the potential of LLM applications in social media data analysis, focusing not only on the most common psychological disorders such as depression and anxiety but also incorporating psychotic disorders and externalizing disorders, summarizing the application methods of LLM from different dimensions, such as text data analysis and detection of mental disorders, and revealing the major challenges and shortcomings of current research. In addition, the paper provides an overview of popular datasets, and evaluation metrics. The survey in this paper provides a comprehensive frame of reference for researchers in the field of mental health, while demonstrating the great potential of LLMs in mental health detection to facilitate the further application of LLMs in future mental health interventions. 

---
# A Framework for Robust Cognitive Evaluation of LLMs 

**Authors**: Karin de Langis, Jong Inn Park, Bin Hu, Khanh Chi Le, Andreas Schramm, Michael C. Mensink, Andrew Elfenbein, Dongyeop Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02789)  

**Abstract**: Emergent cognitive abilities in large language models (LLMs) have been widely observed, but their nature and underlying mechanisms remain poorly understood. A growing body of research draws on cognitive science to investigate LLM cognition, but standard methodologies and experimen-tal pipelines have not yet been established. To address this gap we develop CognitivEval, a framework for systematically evaluating the artificial cognitive capabilities of LLMs, with a particular emphasis on robustness in response collection. The key features of CognitivEval include: (i) automatic prompt permutations, and (ii) testing that gathers both generations and model probability estimates. Our experiments demonstrate that these features lead to more robust experimental outcomes. Using CognitivEval, we replicate five classic experiments in cognitive science, illustrating the framework's generalizability across various experimental tasks and obtaining a cognitive profile of several state of the art LLMs. CognitivEval will be released publicly to foster broader collaboration within the cognitive science community. 

---
# MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs 

**Authors**: Jaap Jumelet, Leonie Weissweiler, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2504.02768)  

**Abstract**: We introduce MultiBLiMP 1.0, a massively multilingual benchmark of linguistic minimal pairs, covering 101 languages, 6 linguistic phenomena and containing more than 125,000 minimal pairs. Our minimal pairs are created using a fully automated pipeline, leveraging the large-scale linguistic resources of Universal Dependencies and UniMorph. MultiBLiMP 1.0 evaluates abilities of LLMs at an unprecedented multilingual scale, and highlights the shortcomings of the current state-of-the-art in modelling low-resource languages. 

---
# Enhancing LLM Robustness to Perturbed Instructions: An Empirical Study 

**Authors**: Aryan Agrawal, Lisa Alazraki, Shahin Honarvar, Marek Rei  

**Link**: [PDF](https://arxiv.org/pdf/2504.02733)  

**Abstract**: Large Language Models (LLMs) are highly vulnerable to input perturbations, as even a small prompt change may result in a substantially different output. Existing methods to enhance LLM robustness are primarily focused on perturbed data samples, whereas improving resiliency to perturbations of task-level instructions has remained relatively underexplored. In this work, we focus on character- and word-level edits of task-specific instructions, which substantially degrade downstream performance. We experiment with a variety of techniques to enhance the robustness of LLMs, including self-denoising and representation alignment, testing different models (Llama 3 and Flan-T5), datasets (CoLA, QNLI, SST-2) and instructions (both task-oriented and role-oriented). We find that, on average, self-denoising -- whether performed by a frozen LLM or a fine-tuned model -- achieves substantially higher performance gains than alternative strategies, including more complex baselines such as ensembling and supervised methods. 

---
# Why do LLMs attend to the first token? 

**Authors**: Federico Barbero, Álvaro Arroyo, Xiangming Gu, Christos Perivolaropoulos, Michael Bronstein, Petar Veličkovi ć, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02732)  

**Abstract**: Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training. 

---
# ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization 

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.02725)  

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency. 

---
# The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual context 

**Authors**: Nikhil Verma, Manasa Bharadwaj  

**Link**: [PDF](https://arxiv.org/pdf/2504.02708)  

**Abstract**: Alignment tuning has enabled large language models to excel in reasoning, instruction-following, and minimizing harmful generations. However, despite their widespread deployment, these models exhibit a monolingual bias, raising concerns about the effectiveness of alignment across languages. Current alignment methods predominantly focus on English, leaving it unclear how alignment mechanism generalize to multilingual settings. To address this, we conduct a systematic analysis of distributional shifts in the embedding space of LLMs before and after alignment, uncovering its impact on model behavior across diverse languages. We leverage the alignment-induced separation in safety space as a quantitative tool to measure how alignment enforces safety constraints. Our study evaluates seven LLMs using balanced toxicity datasets and parallel text-detoxification benchmarks, revealing substantial disparities in the latent representation space between high-resource and low-resource languages. These findings underscore the need for language-specific fine-tuning to ensure fair, reliable and robust multilingual alignment. Our insights provide a foundation for developing truly safe multilingual LLMs, emphasizing the urgency of addressing alignment gaps in underrepresented languages. 

---
# Limitations of Religious Data and the Importance of the Target Domain: Towards Machine Translation for Guinea-Bissau Creole 

**Authors**: Jacqueline Rowe, Edward Gow-Smith, Mark Hepple  

**Link**: [PDF](https://arxiv.org/pdf/2504.02674)  

**Abstract**: We introduce a new dataset for machine translation of Guinea-Bissau Creole (Kiriol), comprising around 40 thousand parallel sentences to English and Portuguese. This dataset is made up of predominantly religious data (from the Bible and texts from the Jehovah's Witnesses), but also a small amount of general domain data (from a dictionary). This mirrors the typical resource availability of many low resource languages. We train a number of transformer-based models to investigate how to improve domain transfer from religious data to a more general domain. We find that adding even 300 sentences from the target domain when training substantially improves the translation performance, highlighting the importance and need for data collection for low-resource languages, even on a small-scale. We additionally find that Portuguese-to-Kiriol translation models perform better on average than other source and target language pairs, and investigate how this relates to the morphological complexity of the languages involved and the degree of lexical overlap between creoles and lexifiers. Overall, we hope our work will stimulate research into Kiriol and into how machine translation might better support creole languages in general. 

---
# LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems 

**Authors**: Zishuo Liu, Carlos Rabat Villarreal, Mostafa Rahgouy, Amit Das, Zheng Zhang, Chang Ren, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.02671)  

**Abstract**: Fermi Problems (FPs) are mathematical reasoning tasks that require human-like logic and numerical reasoning. Unlike other reasoning questions, FPs often involve real-world impracticalities or ambiguous concepts, making them challenging even for humans to solve. Despite advancements in AI, particularly with large language models (LLMs) in various reasoning tasks, FPs remain relatively under-explored. This work conducted an exploratory study to examine the capabilities and limitations of LLMs in solving FPs. We first evaluated the overall performance of three advanced LLMs using a publicly available FP dataset. We designed prompts according to the recently proposed TELeR taxonomy, including a zero-shot scenario. Results indicated that all three LLMs achieved a fp_score (range between 0 - 1) below 0.5, underscoring the inherent difficulty of these reasoning tasks. To further investigate, we categorized FPs into standard and specific questions, hypothesizing that LLMs would perform better on standard questions, which are characterized by clarity and conciseness, than on specific ones. Comparative experiments confirmed this hypothesis, demonstrating that LLMs performed better on standard FPs in terms of both accuracy and efficiency. 

---
# LinTO Audio and Textual Datasets to Train and Evaluate Automatic Speech Recognition in Tunisian Arabic Dialect 

**Authors**: Hedi Naouara, Jean-Pierre Lorré, Jérôme Louradour  

**Link**: [PDF](https://arxiv.org/pdf/2504.02604)  

**Abstract**: Developing Automatic Speech Recognition (ASR) systems for Tunisian Arabic Dialect is challenging due to the dialect's linguistic complexity and the scarcity of annotated speech datasets. To address these challenges, we propose the LinTO audio and textual datasets -- comprehensive resources that capture phonological and lexical features of Tunisian Arabic Dialect. These datasets include a variety of texts from numerous sources and real-world audio samples featuring diverse speakers and code-switching between Tunisian Arabic Dialect and English or French. By providing high-quality audio paired with precise transcriptions, the LinTO audio and textual datasets aim to provide qualitative material to build and benchmark ASR systems for the Tunisian Arabic Dialect.
Keywords -- Tunisian Arabic Dialect, Speech-to-Text, Low-Resource Languages, Audio Data Augmentation 

---
# LexPam: Legal Procedure Awareness-Guided Mathematical Reasoning 

**Authors**: Kepu Zhang, Guofu Xie, Weijie Yu, Mingyue Xu, Xu Tang, Yaxin Li, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02590)  

**Abstract**: The legal mathematical reasoning ability of LLMs is crucial when applying them to real-world scenarios, as it directly affects the credibility of the LLM. While existing legal LLMs can perform general judicial question answering, their legal mathematical reasoning capabilities have not been trained. Open-domain reasoning models, though able to generate detailed calculation steps, do not follow the reasoning logic required for legal scenarios. Additionally, there is currently a lack of legal mathematical reasoning datasets to help validate and enhance LLMs' reasoning abilities in legal contexts. To address these issues, we propose the first Chinese legal Mathematical Reasoning Dataset, LexNum, which includes three common legal mathematical reasoning scenarios: economic compensation, work injury compensation, and traffic accident compensation. Based on LexNum, we tested the performance of existing legal LLMs and reasoning LLMs, and introduced LexPam, a reinforcement learning algorithm guided by legal procedural awareness to train LLMs, enhancing their mathematical reasoning abilities in legal scenarios. Experiments on tasks in the three legal scenarios show that the performance of existing legal LLMs and reasoning models in legal mathematical reasoning tasks is unsatisfactory. LexPam can enhance the LLM's ability in these tasks. 

---
# Language Models reach higher Agreement than Humans in Historical Interpretation 

**Authors**: Fabio Celli, Georgios Spathulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02572)  

**Abstract**: This paper compares historical annotations by humans and Large Language Models. The findings reveal that both exhibit some cultural bias, but Large Language Models achieve a higher consensus on the interpretation of historical facts from short texts. While humans tend to disagree on the basis of their personal biases, Large Models disagree when they skip information or produce hallucinations. These findings have significant implications for digital humanities, enabling large-scale annotation and quantitative analysis of historical data. This offers new educational and research opportunities to explore historical interpretations from different Language Models, fostering critical thinking about bias. 

---
# Leveraging LLM For Synchronizing Information Across Multilingual Tables 

**Authors**: Siddharth Khincha, Tushar Kataria, Ankita Anand, Dan Roth, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.02559)  

**Abstract**: The vast amount of online information today poses challenges for non-English speakers, as much of it is concentrated in high-resource languages such as English and French. Wikipedia reflects this imbalance, with content in low-resource languages frequently outdated or incomplete. Recent research has sought to improve cross-language synchronization of Wikipedia tables using rule-based methods. These approaches can be effective, but they struggle with complexity and generalization. This paper explores large language models (LLMs) for multilingual information synchronization, using zero-shot prompting as a scalable solution. We introduce the Information Updation dataset, simulating the real-world process of updating outdated Wikipedia tables, and evaluate LLM performance. Our findings reveal that single-prompt approaches often produce suboptimal results, prompting us to introduce a task decomposition strategy that enhances coherence and accuracy. Our proposed method outperforms existing baselines, particularly in Information Updation (1.79%) and Information Addition (20.58%), highlighting the model strength in dynamically updating and enriching data across architectures 

---
# UNDO: Understanding Distillation as Optimization 

**Authors**: Kushal Jain, Piyushi Goyal, Kumar Shridhar  

**Link**: [PDF](https://arxiv.org/pdf/2504.02521)  

**Abstract**: Knowledge distillation has emerged as an effective strategy for compressing large language models' (LLMs) knowledge into smaller, more efficient student models. However, standard one-shot distillation methods often produce suboptimal results due to a mismatch between teacher-generated rationales and the student's specific learning requirements. In this paper, we introduce the UNDO: UNderstanding Distillation as Optimization framework, designed to bridge this gap by iteratively identifying the student's errors and prompting the teacher to refine its explanations accordingly. Each iteration directly targets the student's learning deficiencies, motivating the teacher to provide tailored and enhanced rationales that specifically address these weaknesses. Empirical evaluations on various challenging mathematical and commonsense reasoning tasks demonstrate that our iterative distillation method, UNDO, significantly outperforms standard one-step distillation methods, achieving performance gains of up to 20%. Additionally, we show that teacher-generated data refined through our iterative process remains effective even when applied to different student models, underscoring the broad applicability of our approach. Our work fundamentally reframes knowledge distillation as an iterative teacher-student interaction, effectively leveraging dynamic refinement by the teacher for better knowledge distillation. 

---
# Inference-Time Scaling for Generalist Reward Modeling 

**Authors**: Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02495)  

**Abstract**: Reinforcement learning (RL) has been widely adopted in post-training for large language models (LLMs) at scale. Recently, the incentivization of reasoning capabilities in LLMs from RL indicates that $\textit{proper learning methods could enable effective inference-time scalability}$. A key challenge of RL is to obtain accurate reward signals for LLMs in various domains beyond verifiable questions or artificial rules. In this work, we investigate how to improve reward modeling (RM) with more inference compute for general queries, i.e. the $\textbf{inference-time scalability of generalist RM}$, and further, how to improve the effectiveness of performance-compute scaling with proper learning methods. For the RM approach, we adopt pointwise generative reward modeling (GRM) to enable flexibility for different input types and potential for inference-time scaling. For the learning method, we propose Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL, to generate principles adaptively and critiques accurately, resulting in $\textbf{DeepSeek-GRM}$ models. Furthermore, for effective inference-time scaling, we use parallel sampling to expand compute usage, and introduce a meta RM to guide voting process for better scaling performance. Empirically, we show that SPCT significantly improves the quality and scalability of GRMs, outperforming existing methods and models in various RM benchmarks without severe biases, and could achieve better performance compared to training-time scaling. DeepSeek-GRM still meets challenges in some tasks, which we believe can be addressed by future efforts in generalist reward systems. The models will be released and open-sourced. 

---
# Cognitive Memory in Large Language Models 

**Authors**: Lianlei Shan, Shixian Luo, Zezhou Zhu, Yu Yuan, Yong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02441)  

**Abstract**: This paper examines memory mechanisms in Large Language Models (LLMs), emphasizing their importance for context-rich responses, reduced hallucinations, and improved efficiency. It categorizes memory into sensory, short-term, and long-term, with sensory memory corresponding to input prompts, short-term memory processing immediate context, and long-term memory implemented via external databases or structures. The text-based memory section covers acquisition (selection and summarization), management (updating, accessing, storing, and resolving conflicts), and utilization (full-text search, SQL queries, semantic search). The KV cache-based memory section discusses selection methods (regularity-based summarization, score-based approaches, special token embeddings) and compression techniques (low-rank compression, KV merging, multimodal compression), along with management strategies like offloading and shared attention mechanisms. Parameter-based memory methods (LoRA, TTT, MoE) transform memories into model parameters to enhance efficiency, while hidden-state-based memory approaches (chunk mechanisms, recurrent transformers, Mamba model) improve long-text processing by combining RNN hidden states with current methods. Overall, the paper offers a comprehensive analysis of LLM memory mechanisms, highlighting their significance and future research directions. 

---
# Scaling Video-Language Models to 10K Frames via Hierarchical Differential Distillation 

**Authors**: Chuanqi Cheng, Jian Guan, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02438)  

**Abstract**: Long-form video processing fundamentally challenges vision-language models (VLMs) due to the high computational costs of handling extended temporal sequences. Existing token pruning and feature merging methods often sacrifice critical temporal dependencies or dilute semantic information. We introduce differential distillation, a principled approach that systematically preserves task-relevant information while suppressing redundancy. Based on this principle, we develop ViLaMP, a hierarchical video-language model that processes hour-long videos at ``mixed precision'' through two key mechanisms: (1) differential keyframe selection that maximizes query relevance while maintaining temporal distinctiveness at the frame level and (2) differential feature merging that preserves query-salient features in non-keyframes at the patch level. Hence, ViLaMP retains full information in keyframes while reducing non-keyframes to their most salient features, resembling mixed-precision training. Extensive experiments demonstrate ViLaMP's superior performance across four video understanding benchmarks, particularly on long-form content. Notably, ViLaMP can process ultra-long videos (up to 10K frames) on a single NVIDIA A100 GPU, achieving substantial computational efficiency while maintaining state-of-the-art performance. 

---
# Adapting Large Language Models for Multi-Domain Retrieval-Augmented-Generation 

**Authors**: Alexandre Misrahi, Nadezhda Chirkova, Maxime Louis, Vassilina Nikoulina  

**Link**: [PDF](https://arxiv.org/pdf/2504.02411)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances LLM factuality, but multi-domain applications face challenges like lack of diverse benchmarks and poor out-of-domain generalization. The first contribution of this work is to introduce a diverse benchmark comprising a variety of question-answering tasks from 8 sources and covering 13 domains. Our second contribution consists in systematically testing out-of-domain generalization for typical RAG tuning strategies. While our findings reveal that standard fine-tuning fails to generalize effectively, we show that sequence-level distillation with teacher-generated labels improves out-of-domain performance by providing more coherent supervision. Our findings highlight key strategies for improving multi-domain RAG robustness. 

---
# AnesBench: Multi-Dimensional Evaluation of LLM Reasoning in Anesthesiology 

**Authors**: Xiang Feng, Wentao Jiang, Zengmao Wang, Yong Luo, Pingbo Xu, Baosheng Yu, Hua Jin, Bo Du, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02404)  

**Abstract**: The application of large language models (LLMs) in the medical field has gained significant attention, yet their reasoning capabilities in more specialized domains like anesthesiology remain underexplored. In this paper, we systematically evaluate the reasoning capabilities of LLMs in anesthesiology and analyze key factors influencing their performance. To this end, we introduce AnesBench, a cross-lingual benchmark designed to assess anesthesiology-related reasoning across three levels: factual retrieval (System 1), hybrid reasoning (System 1.x), and complex decision-making (System 2). Through extensive experiments, we first explore how model characteristics, including model scale, Chain of Thought (CoT) length, and language transferability, affect reasoning performance. Then, we further evaluate the effectiveness of different training strategies, leveraging our curated anesthesiology-related dataset, including continuous pre-training (CPT) and supervised fine-tuning (SFT). Additionally, we also investigate how the test-time reasoning techniques, such as Best-of-N sampling and beam search, influence reasoning performance, and assess the impact of reasoning-enhanced model distillation, specifically DeepSeek-R1. We will publicly release AnesBench, along with our CPT and SFT training datasets and evaluation code at this https URL. 

---
# DaKultur: Evaluating the Cultural Awareness of Language Models for Danish with Native Speakers 

**Authors**: Max Müller-Eberstein, Mike Zhang, Elisa Bassignana, Peter Brunsgaard Trolle, Rob van der Goot  

**Link**: [PDF](https://arxiv.org/pdf/2504.02403)  

**Abstract**: Large Language Models (LLMs) have seen widespread societal adoption. However, while they are able to interact with users in languages beyond English, they have been shown to lack cultural awareness, providing anglocentric or inappropriate responses for underrepresented language communities. To investigate this gap and disentangle linguistic versus cultural proficiency, we conduct the first cultural evaluation study for the mid-resource language of Danish, in which native speakers prompt different models to solve tasks requiring cultural awareness. Our analysis of the resulting 1,038 interactions from 63 demographically diverse participants highlights open challenges to cultural adaptation: Particularly, how currently employed automatically translated data are insufficient to train or measure cultural adaptation, and how training on native-speaker data can more than double response acceptance rates. We release our study data as DaKultur - the first native Danish cultural awareness dataset. 

---
# Scaling Analysis of Interleaved Speech-Text Language Models 

**Authors**: Gallil Maimon, Michael Hassid, Amit Roth, Yossi Adi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02398)  

**Abstract**: Existing Speech Language Model (SLM) scaling analysis paints a bleak picture. They predict that SLMs require much more compute and data compared to text, leading some to question the feasibility of training high-quality SLMs. However, modern SLMs are often initialised from pre-trained TextLMs using speech-text interleaving to allow knowledge transfer. This raises the question - Do interleaved SLMs scale more efficiently than textless-SLMs? In this paper we answer a resounding, yes! We conduct scaling analysis of interleaved SLMs by training several dozen and analysing the scaling trends. We see that under this setup SLMs scale more efficiently with compute. Additionally, our results indicate that the scaling-dynamics are significantly different than textless-SLMs, suggesting one should allocate notably more of the compute budget for increasing model size over training tokens. We also study the role of synthetic data and TextLM model families in unlocking this potential. Results suggest, that our scaled up model achieves comparable performance with leading models on speech semantic metrics while using less compute and data than other approaches. We open source models, samples, and data - this https URL. 

---
# The quasi-semantic competence of LLMs: a case study on the part-whole relation 

**Authors**: Mattia Proietti, Alessandro Lenci  

**Link**: [PDF](https://arxiv.org/pdf/2504.02395)  

**Abstract**: Understanding the extent and depth of the semantic competence of \emph{Large Language Models} (LLMs) is at the center of the current scientific agenda in Artificial Intelligence (AI) and Computational Linguistics (CL). We contribute to this endeavor by investigating their knowledge of the \emph{part-whole} relation, a.k.a. \emph{meronymy}, which plays a crucial role in lexical organization, but it is significantly understudied. We used data from ConceptNet relations \citep{speer2016conceptnet} and human-generated semantic feature norms \citep{McRae:2005} to explore the abilities of LLMs to deal with \textit{part-whole} relations. We employed several methods based on three levels of analysis: i.) \textbf{behavioral} testing via prompting, where we directly queried the models on their knowledge of meronymy, ii.) sentence \textbf{probability} scoring, where we tested models' abilities to discriminate correct (real) and incorrect (asymmetric counterfactual) \textit{part-whole} relations, and iii.) \textbf{concept representation} analysis in vector space, where we proved the linear organization of the \textit{part-whole} concept in the embedding and unembedding spaces. These analyses present a complex picture that reveals that the LLMs' knowledge of this relation is only partial. They have just a ``\emph{quasi}-semantic'' competence and still fall short of capturing deep inferential properties. 

---
# LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models 

**Authors**: Weibin Liao, Xin Gao, Tianyu Jia, Rihong Qiu, Yifan Zhu, Yang Lin, Xu Chu, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02327)  

**Abstract**: Natural Language to SQL (NL2SQL) has emerged as a critical task for enabling seamless interaction with databases. Recent advancements in Large Language Models (LLMs) have demonstrated remarkable performance in this domain. However, existing NL2SQL methods predominantly rely on closed-source LLMs leveraging prompt engineering, while open-source models typically require fine-tuning to acquire domain-specific knowledge. Despite these efforts, open-source LLMs struggle with complex NL2SQL tasks due to the indirect expression of user query objectives and the semantic gap between user queries and database schemas. Inspired by the application of reinforcement learning in mathematical problem-solving to encourage step-by-step reasoning in LLMs, we propose LearNAT (Learning NL2SQL with AST-guided Task Decomposition), a novel framework that improves the performance of open-source LLMs on complex NL2SQL tasks through task decomposition and reinforcement learning. LearNAT introduces three key components: (1) a Decomposition Synthesis Procedure that leverages Abstract Syntax Trees (ASTs) to guide efficient search and pruning strategies for task decomposition, (2) Margin-aware Reinforcement Learning, which employs fine-grained step-level optimization via DPO with AST margins, and (3) Adaptive Demonstration Reasoning, a mechanism for dynamically selecting relevant examples to enhance decomposition capabilities. Extensive experiments on two benchmark datasets, Spider and BIRD, demonstrate that LearNAT enables a 7B-parameter open-source LLM to achieve performance comparable to GPT-4, while offering improved efficiency and accessibility. 

---
# CoTAL: Human-in-the-Loop Prompt Engineering, Chain-of-Thought Reasoning, and Active Learning for Generalizable Formative Assessment Scoring 

**Authors**: Clayton Cohn, Nicole Hutchins, Ashwin T S, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2504.02323)  

**Abstract**: Large language models (LLMs) have created new opportunities to assist teachers and support student learning. Methods such as chain-of-thought (CoT) prompting enable LLMs to grade formative assessments in science, providing scores and relevant feedback to students. However, the extent to which these methods generalize across curricula in multiple domains (such as science, computing, and engineering) remains largely untested. In this paper, we introduce Chain-of-Thought Prompting + Active Learning (CoTAL), an LLM-based approach to formative assessment scoring that (1) leverages Evidence-Centered Design (ECD) principles to develop curriculum-aligned formative assessments and rubrics, (2) applies human-in-the-loop prompt engineering to automate response scoring, and (3) incorporates teacher and student feedback to iteratively refine assessment questions, grading rubrics, and LLM prompts for automated grading. Our findings demonstrate that CoTAL improves GPT-4's scoring performance, achieving gains of up to 24.5% over a non-prompt-engineered baseline. Both teachers and students view CoTAL as effective in scoring and explaining student responses, each providing valuable refinements to enhance grading accuracy and explanation quality. 

---
# Improving Harmful Text Detection with Joint Retrieval and External Knowledge 

**Authors**: Zidong Yu, Shuo Wang, Nan Jiang, Weiqiang Huang, Xu Han, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2504.02310)  

**Abstract**: Harmful text detection has become a crucial task in the development and deployment of large language models, especially as AI-generated content continues to expand across digital platforms. This study proposes a joint retrieval framework that integrates pre-trained language models with knowledge graphs to improve the accuracy and robustness of harmful text detection. Experimental results demonstrate that the joint retrieval approach significantly outperforms single-model baselines, particularly in low-resource training scenarios and multilingual environments. The proposed method effectively captures nuanced harmful content by leveraging external contextual information, addressing the limitations of traditional detection models. Future research should focus on optimizing computational efficiency, enhancing model interpretability, and expanding multimodal detection capabilities to better tackle evolving harmful content patterns. This work contributes to the advancement of AI safety, ensuring more trustworthy and reliable content moderation systems. 

---
# Measurement of LLM's Philosophies of Human Nature 

**Authors**: Minheng Ni, Ennan Wu, Zidong Gong, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Lijuan Wang, Wangmeng Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02304)  

**Abstract**: The widespread application of artificial intelligence (AI) in various tasks, along with frequent reports of conflicts or violations involving AI, has sparked societal concerns about interactions with AI systems. Based on Wrightsman's Philosophies of Human Nature Scale (PHNS), a scale empirically validated over decades to effectively assess individuals' attitudes toward human nature, we design the standardized psychological scale specifically targeting large language models (LLM), named the Machine-based Philosophies of Human Nature Scale (M-PHNS). By evaluating LLMs' attitudes toward human nature across six dimensions, we reveal that current LLMs exhibit a systemic lack of trust in humans, and there is a significant negative correlation between the model's intelligence level and its trust in humans. Furthermore, we propose a mental loop learning framework, which enables LLM to continuously optimize its value system during virtual interactions by constructing moral scenarios, thereby improving its attitude toward human nature. Experiments demonstrate that mental loop learning significantly enhances their trust in humans compared to persona or instruction prompts. This finding highlights the potential of human-based psychological assessments for LLM, which can not only diagnose cognitive biases but also provide a potential solution for ethical learning in artificial intelligence. We release the M-PHNS evaluation code and data at this https URL. 

---
# State-of-the-Art Translation of Text-to-Gloss using mBART : A case study of Bangla 

**Authors**: Sharif Md. Abdullah, Abhijit Paul, Shebuti Rayana, Ahmedul Kabir, Zarif Masud  

**Link**: [PDF](https://arxiv.org/pdf/2504.02293)  

**Abstract**: Despite a large deaf and dumb population of 1.7 million, Bangla Sign Language (BdSL) remains a understudied domain. Specifically, there are no works on Bangla text-to-gloss translation task. To address this gap, we begin by addressing the dataset problem. We take inspiration from grammatical rule based gloss generation used in Germany and American sign langauage (ASL) and adapt it for BdSL. We also leverage LLM to generate synthetic data and use back-translation, text generation for data augmentation. With dataset prepared, we started experimentation. We fine-tuned pretrained mBART-50 and mBERT-multiclass-uncased model on our dataset. We also trained GRU, RNN and a novel seq-to-seq model with multi-head attention. We observe significant high performance (ScareBLEU=79.53) with fine-tuning pretrained mBART-50 multilingual model from Facebook. We then explored why we observe such high performance with mBART. We soon notice an interesting property of mBART -- it was trained on shuffled and masked text data. And as we know, gloss form has shuffling property. So we hypothesize that mBART is inherently good at text-to-gloss tasks. To find support against this hypothesis, we trained mBART-50 on PHOENIX-14T benchmark and evaluated it with existing literature. Our mBART-50 finetune demonstrated State-of-the-Art performance on PHOENIX-14T benchmark, far outperforming existing models in all 6 metrics (ScareBLEU = 63.89, BLEU-1 = 55.14, BLEU-2 = 38.07, BLEU-3 = 27.13, BLEU-4 = 20.68, COMET = 0.624). Based on the results, this study proposes a new paradigm for text-to-gloss task using mBART models. Additionally, our results show that BdSL text-to-gloss task can greatly benefit from rule-based synthetic dataset. 

---
# LLMs as Deceptive Agents: How Role-Based Prompting Induces Semantic Ambiguity in Puzzle Tasks 

**Authors**: Seunghyun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02254)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have not only showcased impressive creative capabilities but also revealed emerging agentic behaviors that exploit linguistic ambiguity in adversarial settings. In this study, we investigate how an LLM, acting as an autonomous agent, leverages semantic ambiguity to generate deceptive puzzles that mislead and challenge human users. Inspired by the popular puzzle game "Connections", we systematically compare puzzles produced through zero-shot prompting, role-injected adversarial prompts, and human-crafted examples, with an emphasis on understanding the underlying agent decision-making processes. Employing computational analyses with HateBERT to quantify semantic ambiguity, alongside subjective human evaluations, we demonstrate that explicit adversarial agent behaviors significantly heighten semantic ambiguity -- thereby increasing cognitive load and reducing fairness in puzzle solving. These findings provide critical insights into the emergent agentic qualities of LLMs and underscore important ethical considerations for evaluating and safely deploying autonomous language systems in both educational technologies and entertainment. 

---
# Subasa -- Adapting Language Models for Low-resourced Offensive Language Detection in Sinhala 

**Authors**: Shanilka Haturusinghe, Tharindu Cyril Weerasooriya, Marcos Zampieri, Christopher M. Homan, S.R. Liyanage  

**Link**: [PDF](https://arxiv.org/pdf/2504.02178)  

**Abstract**: Accurate detection of offensive language is essential for a number of applications related to social media safety. There is a sharp contrast in performance in this task between low and high-resource languages. In this paper, we adapt fine-tuning strategies that have not been previously explored for Sinhala in the downstream task of offensive language detection. Using this approach, we introduce four models: "Subasa-XLM-R", which incorporates an intermediate Pre-Finetuning step using Masked Rationale Prediction. Two variants of "Subasa-Llama" and "Subasa-Mistral", are fine-tuned versions of Llama (3.2) and Mistral (v0.3), respectively, with a task-specific strategy. We evaluate our models on the SOLD benchmark dataset for Sinhala offensive language detection. All our models outperform existing baselines. Subasa-XLM-R achieves the highest Macro F1 score (0.84) surpassing state-of-the-art large language models like GPT-4o when evaluated on the same SOLD benchmark dataset under zero-shot settings. The models and code are publicly available. 

---
# LL4G: Self-Supervised Dynamic Optimization for Graph-Based Personality Detection 

**Authors**: Lingzhi Shen, Yunfei Long, Xiaohao Cai, Guanming Chen, Yuhan Wang, Imran Razzak, Shoaib Jameel  

**Link**: [PDF](https://arxiv.org/pdf/2504.02146)  

**Abstract**: Graph-based personality detection constructs graph structures from textual data, particularly social media posts. Current methods often struggle with sparse or noisy data and rely on static graphs, limiting their ability to capture dynamic changes between nodes and relationships. This paper introduces LL4G, a self-supervised framework leveraging large language models (LLMs) to optimize graph neural networks (GNNs). LLMs extract rich semantic features to generate node representations and to infer explicit and implicit relationships. The graph structure adaptively adds nodes and edges based on input data, continuously optimizing itself. The GNN then uses these optimized representations for joint training on node reconstruction, edge prediction, and contrastive learning tasks. This integration of semantic and structural information generates robust personality profiles. Experimental results on Kaggle and Pandora datasets show LL4G outperforms state-of-the-art models. 

---
# One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image 

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks  

**Link**: [PDF](https://arxiv.org/pdf/2504.02132)  

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings. 

---
# Overcoming Vocabulary Constraints with Pixel-level Fallback 

**Authors**: Jonas F. Lotz, Hendra Setiawan, Stephan Peitz, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2504.02122)  

**Abstract**: Subword tokenization requires balancing computational efficiency and vocabulary coverage, which often leads to suboptimal performance on languages and scripts not prioritized during training. We propose to augment pretrained language models with a vocabulary-free encoder that generates input embeddings from text rendered as pixels. Through experiments on English-centric language models, we demonstrate that our approach substantially improves machine translation performance and facilitates effective cross-lingual transfer, outperforming tokenizer-based methods. Furthermore, we find that pixel-based representations outperform byte-level approaches and standard vocabulary expansion. Our approach enhances the multilingual capabilities of monolingual language models without extensive retraining and reduces decoding latency via input compression. 

---
# Language Models at the Syntax-Semantics Interface: A Case Study of the Long-Distance Binding of Chinese Reflexive ziji 

**Authors**: Xiulin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02116)  

**Abstract**: This paper explores whether language models can effectively resolve the complex binding patterns of the Mandarin Chinese reflexive ziji, which are constrained by both syntactic and semantic factors. We construct a dataset of 240 synthetic sentences using templates and examples from syntactic literature, along with 320 natural sentences from the BCC corpus. Evaluating 21 language models against this dataset and comparing their performance to judgments from native Mandarin speakers, we find that none of the models consistently replicates human-like judgments. The results indicate that existing language models tend to rely heavily on sequential cues, though not always favoring the closest strings, and often overlooking subtle semantic and syntactic constraints. They tend to be more sensitive to noun-related than verb-related semantics. 

---
# ContrastScore: Towards Higher Quality, Less Biased, More Efficient Evaluation Metrics with Contrastive Evaluation 

**Authors**: Xiao Wang, Daniil Larionov, Siwei Wu, Yiqi Liu, Steffen Eger, Nafise Sadat Moosavi, Chenghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.02106)  

**Abstract**: Evaluating the quality of generated text automatically remains a significant challenge. Conventional reference-based metrics have been shown to exhibit relatively weak correlation with human evaluations. Recent research advocates the use of large language models (LLMs) as source-based metrics for natural language generation (NLG) assessment. While promising, LLM-based metrics, particularly those using smaller models, still fall short in aligning with human judgments. In this work, we introduce ContrastScore, a contrastive evaluation metric designed to enable higher-quality, less biased, and more efficient assessment of generated text. We evaluate ContrastScore on two NLG tasks: machine translation and summarization. Experimental results show that ContrastScore consistently achieves stronger correlation with human judgments than both single-model and ensemble-based baselines. Notably, ContrastScore based on Qwen 3B and 0.5B even outperforms Qwen 7B, despite having only half as many parameters, demonstrating its efficiency. Furthermore, it effectively mitigates common evaluation biases such as length and likelihood preferences, resulting in more robust automatic evaluation. 

---
# Increasing happiness through conversations with artificial intelligence 

**Authors**: Joseph Heffner, Chongyu Qin, Martin Chadwick, Chris Knutsen, Christopher Summerfield, Zeb Kurth-Nelson, Robb B. Rutledge  

**Link**: [PDF](https://arxiv.org/pdf/2504.02091)  

**Abstract**: Chatbots powered by artificial intelligence (AI) have rapidly become a significant part of everyday life, with over a quarter of American adults using them multiple times per week. While these tools offer potential benefits and risks, a fundamental question remains largely unexplored: How do conversations with AI influence subjective well-being? To investigate this, we conducted a study where participants either engaged in conversations with an AI chatbot (N = 334) or wrote journal entires (N = 193) on the same randomly assigned topics and reported their momentary happiness afterward. We found that happiness after AI chatbot conversations was higher than after journaling, particularly when discussing negative topics such as depression or guilt. Leveraging large language models for sentiment analysis, we found that the AI chatbot mirrored participants' sentiment while maintaining a consistent positivity bias. When discussing negative topics, participants gradually aligned their sentiment with the AI's positivity, leading to an overall increase in happiness. We hypothesized that the history of participants' sentiment prediction errors, the difference between expected and actual emotional tone when responding to the AI chatbot, might explain this happiness effect. Using computational modeling, we find the history of these sentiment prediction errors over the course of a conversation predicts greater post-conversation happiness, demonstrating a central role of emotional expectations during dialogue. Our findings underscore the effect that AI interactions can have on human well-being. 

---
# From Text to Graph: Leveraging Graph Neural Networks for Enhanced Explainability in NLP 

**Authors**: Fabio Yáñez-Romero, Andrés Montoyo, Armando Suárez, Yoan Gutiérrez, Ruslan Mitkov  

**Link**: [PDF](https://arxiv.org/pdf/2504.02064)  

**Abstract**: Researchers have relegated natural language processing tasks to Transformer-type models, particularly generative models, because these models exhibit high versatility when performing generation and classification tasks. As the size of these models increases, they achieve outstanding results. Given their widespread use, many explainability techniques are developed based on these models. However, this process becomes computationally expensive due to the large size of the models. Additionally, transformers interpret input information through tokens that fragment input words into sequences lacking inherent semantic meaning, complicating the explanation of the model from the very beginning. This study proposes a novel methodology to achieve explainability in natural language processing tasks by automatically converting sentences into graphs and maintaining semantics through nodes and relations that express fundamental linguistic concepts. It also allows the subsequent exploitation of this knowledge in subsequent tasks, making it possible to obtain trends and understand how the model associates the different elements inside the text with the explained task. The experiments delivered promising results in determining the most critical components within the text structure for a given classification. 

---
# Concept Lancet: Image Editing with Compositional Representation Transplant 

**Authors**: Jinqi Luo, Tianjiao Ding, Kwan Ho Ryan Chan, Hancheng Min, Chris Callison-Burch, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2504.02828)  

**Abstract**: Diffusion models are widely used for image editing tasks. Existing editing methods often design a representation manipulation procedure by curating an edit direction in the text embedding or score space. However, such a procedure faces a key challenge: overestimating the edit strength harms visual consistency while underestimating it fails the editing task. Notably, each source image may require a different editing strength, and it is costly to search for an appropriate strength via trial-and-error. To address this challenge, we propose Concept Lancet (CoLan), a zero-shot plug-and-play framework for principled representation manipulation in diffusion-based image editing. At inference time, we decompose the source input in the latent (text embedding or diffusion score) space as a sparse linear combination of the representations of the collected visual concepts. This allows us to accurately estimate the presence of concepts in each image, which informs the edit. Based on the editing task (replace/add/remove), we perform a customized concept transplant process to impose the corresponding editing direction. To sufficiently model the concept space, we curate a conceptual representation dataset, CoLan-150K, which contains diverse descriptions and scenarios of visual terms and phrases for the latent dictionary. Experiments on multiple diffusion-based image editing baselines show that methods equipped with CoLan achieve state-of-the-art performance in editing effectiveness and consistency preservation. 

---
# A Framework for Situating Innovations, Opportunities, and Challenges in Advancing Vertical Systems with Large AI Models 

**Authors**: Gaurav Verma, Jiawei Zhou, Mohit Chandra, Srijan Kumar, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.02793)  

**Abstract**: Large artificial intelligence (AI) models have garnered significant attention for their remarkable, often "superhuman", performance on standardized benchmarks. However, when these models are deployed in high-stakes verticals such as healthcare, education, and law, they often reveal notable limitations. For instance, they exhibit brittleness to minor variations in input data, present contextually uninformed decisions in critical settings, and undermine user trust by confidently producing or reproducing inaccuracies. These challenges in applying large models necessitate cross-disciplinary innovations to align the models' capabilities with the needs of real-world applications. We introduce a framework that addresses this gap through a layer-wise abstraction of innovations aimed at meeting users' requirements with large models. Through multiple case studies, we illustrate how researchers and practitioners across various fields can operationalize this framework. Beyond modularizing the pipeline of transforming large models into useful "vertical systems", we also highlight the dynamism that exists within different layers of the framework. Finally, we discuss how our framework can guide researchers and practitioners to (i) optimally situate their innovations (e.g., when vertical-specific insights can empower broadly impactful vertical-agnostic innovations), (ii) uncover overlooked opportunities (e.g., spotting recurring problems across verticals to develop practically useful foundation models instead of chasing benchmarks), and (iii) facilitate cross-disciplinary communication of critical challenges (e.g., enabling a shared vocabulary for AI developers, domain experts, and human-computer interaction scholars). 

---
# Affordable AI Assistants with Knowledge Graph of Thoughts 

**Authors**: Maciej Besta, Lorenzo Paleari, Jia Hao Andrea Jiang, Robert Gerstenberger, You Wu, Patrick Iff, Ales Kubicek, Piotr Nyczyk, Diana Khimey, Jón Gunnar Hannesson, Grzegorz Kwaśniewski, Marcin Copik, Hubert Niewiadomski, Torsten Hoefler  

**Link**: [PDF](https://arxiv.org/pdf/2504.02670)  

**Abstract**: Large Language Models (LLMs) are revolutionizing the development of AI assistants capable of performing diverse tasks across domains. However, current state-of-the-art LLM-driven agents face significant challenges, including high operational costs and limited success rates on complex benchmarks like GAIA. To address these issues, we propose the Knowledge Graph of Thoughts (KGoT), an innovative AI assistant architecture that integrates LLM reasoning with dynamically constructed knowledge graphs (KGs). KGoT extracts and structures task-relevant knowledge into a dynamic KG representation, iteratively enhanced through external tools such as math solvers, web crawlers, and Python scripts. Such structured representation of task-relevant knowledge enables low-cost models to solve complex tasks effectively. For example, KGoT achieves a 29% improvement in task success rates on the GAIA benchmark compared to Hugging Face Agents with GPT-4o mini, while reducing costs by over 36x compared to GPT-4o. Improvements for recent reasoning models are similar, e.g., 36% and 37.5% for Qwen2.5-32B and Deepseek-R1-70B, respectively. KGoT offers a scalable, affordable, and high-performing solution for AI assistants. 

---
# Efficient Model Editing with Task-Localized Sparse Fine-tuning 

**Authors**: Leonardo Iurada, Marco Ciccone, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02620)  

**Abstract**: Task arithmetic has emerged as a promising approach for editing models by representing task-specific knowledge as composable task vectors. However, existing methods rely on network linearization to derive task vectors, leading to computational bottlenecks during training and inference. Moreover, linearization alone does not ensure weight disentanglement, the key property that enables conflict-free composition of task vectors. To address this, we propose TaLoS which allows to build sparse task vectors with minimal interference without requiring explicit linearization and sharing information across tasks. We find that pre-trained models contain a subset of parameters with consistently low gradient sensitivity across tasks, and that sparsely updating only these parameters allows for promoting weight disentanglement during fine-tuning. Our experiments prove that TaLoS improves training and inference efficiency while outperforming current methods in task addition and negation. By enabling modular parameter editing, our approach fosters practical deployment of adaptable foundation models in real-world applications. 

---
# Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving 

**Authors**: Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Linhao Zhang, Shulin Xin, Lu Chen, Qi Liu, Xiaojian Zhong, Aoyan Li, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Kai Shen, Liang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02605)  

**Abstract**: The task of issue resolving is to modify a codebase to generate a patch that addresses a given issue. However, existing benchmarks, such as SWE-bench, focus almost exclusively on Python, making them insufficient for evaluating Large Language Models (LLMs) across diverse software ecosystems. To address this, we introduce a multilingual issue-resolving benchmark, called Multi-SWE-bench, covering Java, TypeScript, JavaScript, Go, Rust, C, and C++. It includes a total of 1,632 high-quality instances, which were carefully annotated from 2,456 candidates by 68 expert annotators, ensuring that the benchmark can provide an accurate and reliable evaluation. Based on Multi-SWE-bench, we evaluate a series of state-of-the-art models using three representative methods (Agentless, SWE-agent, and OpenHands) and present a comprehensive analysis with key empirical insights. In addition, we launch a Multi-SWE-RL open-source community, aimed at building large-scale reinforcement learning (RL) training datasets for issue-resolving tasks. As an initial contribution, we release a set of 4,723 well-structured instances spanning seven programming languages, laying a solid foundation for RL research in this domain. More importantly, we open-source our entire data production pipeline, along with detailed tutorials, encouraging the open-source community to continuously contribute and expand the dataset. We envision our Multi-SWE-bench and the ever-growing Multi-SWE-RL community as catalysts for advancing RL toward its full potential, bringing us one step closer to the dawn of AGI. 

---
# Rethinking RL Scaling for Vision Language Models: A Transparent, From-Scratch Framework and Comprehensive Evaluation Scheme 

**Authors**: Yan Ma, Steffi Chern, Xuyang Shen, Yiran Zhong, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.02587)  

**Abstract**: Reinforcement learning (RL) has recently shown strong potential in improving the reasoning capabilities of large language models and is now being actively extended to vision-language models (VLMs). However, existing RL applications in VLMs often rely on heavily engineered frameworks that hinder reproducibility and accessibility, while lacking standardized evaluation protocols, making it difficult to compare results or interpret training dynamics. This work introduces a transparent, from-scratch framework for RL in VLMs, offering a minimal yet functional four-step pipeline validated across multiple models and datasets. In addition, a standardized evaluation scheme is proposed to assess training dynamics and reflective behaviors. Extensive experiments on visual reasoning tasks uncover key empirical findings: response length is sensitive to random seeds, reflection correlates with output length, and RL consistently outperforms supervised fine-tuning (SFT) in generalization, even with high-quality data. These findings, together with the proposed framework, aim to establish a reproducible baseline and support broader engagement in RL-based VLM research. 

---
# Reasoning Inconsistencies and How to Mitigate Them in Deep Learning 

**Authors**: Erik Arakelyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.02577)  

**Abstract**: The recent advancements in Deep Learning models and techniques have led to significant strides in performance across diverse tasks and modalities. However, while the overall capabilities of models show promising growth, our understanding of their internal reasoning processes remains limited, particularly concerning systematic inconsistencies or errors patterns of logical or inferential flaws. These inconsistencies may manifest as contradictory outputs, failure to generalize across similar tasks, or erroneous conclusions in specific contexts. Even detecting and measuring such reasoning discrepancies is challenging, as they may arise from opaque internal procedures, biases and imbalances in training data, or the inherent complexity of the task. Without effective methods to detect, measure, and mitigate these errors, there is a risk of deploying models that are biased, exploitable, or logically unreliable. This thesis aims to address these issues by producing novel methods for deep learning models that reason over knowledge graphs, natural language, and images. The thesis contributes two techniques for detecting and quantifying predictive inconsistencies originating from opaque internal procedures in natural language and image processing models. To mitigate inconsistencies from biases in training data, this thesis presents a data efficient sampling method to improve fairness and performance and a synthetic dataset generation approach in low resource scenarios. Finally, the thesis offers two techniques to optimize the models for complex reasoning tasks. These methods enhance model performance while allowing for more faithful and interpretable exploration and exploitation during inference. Critically, this thesis provides a comprehensive framework to improve the robustness, fairness, and interpretability of deep learning models across diverse tasks and modalities. 

---
# ZClip: Adaptive Spike Mitigation for LLM Pre-Training 

**Authors**: Abhay Kumar, Louis Owen, Nilabhra Roy Chowdhury, Fabian Güra  

**Link**: [PDF](https://arxiv.org/pdf/2504.02507)  

**Abstract**: Training large language models (LLMs) presents numerous challenges, including gradient instability and loss spikes. These phenomena can lead to catastrophic divergence, requiring costly checkpoint restoration and data batch skipping. Traditional gradient clipping techniques, such as constant or norm-based methods, fail to address these issues effectively due to their reliance on fixed thresholds or heuristics, leading to inefficient learning and requiring frequent manual intervention. In this work, we propose ZClip, an adaptive gradient clipping algorithm that dynamically adjusts the clipping threshold based on statistical properties of gradient norms over time. Unlike prior reactive strategies, ZClip proactively adapts to training dynamics without making any prior assumptions on the scale and the temporal evolution of gradient norms. At its core, it leverages z-score-based anomaly detection to identify and mitigate large gradient spikes, preventing malignant loss spikes while not interfering with convergence otherwise. Our code is available at: this https URL. 

---
# Advancing Semantic Caching for LLMs with Domain-Specific Embeddings and Synthetic Data 

**Authors**: Waris Gill, Justin Cechmanek, Tyler Hutcherson, Srijith Rajamohan, Jen Agarwal, Muhammad Ali Gulzar, Manvinder Singh, Benoit Dion  

**Link**: [PDF](https://arxiv.org/pdf/2504.02268)  

**Abstract**: This report investigates enhancing semantic caching effectiveness by employing specialized, fine-tuned embedding models. Semantic caching relies on embedding similarity rather than exact key matching, presenting unique challenges in balancing precision, query latency, and computational efficiency. We propose leveraging smaller, domain-specific embedding models, fine-tuned with targeted real-world and synthetically generated datasets. Our empirical evaluations demonstrate that compact embedding models fine-tuned for just one epoch on specialized datasets significantly surpass both state-of-the-art open-source and proprietary alternatives in precision and recall. Moreover, we introduce a novel synthetic data generation pipeline for the semantic cache that mitigates the challenge of limited domain-specific annotated data, further boosting embedding performance. Our approach effectively balances computational overhead and accuracy, establishing a viable and efficient strategy for practical semantic caching implementations. 

---
# LLM Social Simulations Are a Promising Research Method 

**Authors**: Jacy Reese Anthis, Ryan Liu, Sean M. Richardson, Austin C. Kozlowski, Bernard Koch, James Evans, Erik Brynjolfsson, Michael Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2504.02234)  

**Abstract**: Accurate and verifiable large language model (LLM) simulations of human research subjects promise an accessible data source for understanding human behavior and training new AI systems. However, results to date have been limited, and few social scientists have adopted these methods. In this position paper, we argue that the promise of LLM social simulations can be achieved by addressing five tractable challenges. We ground our argument in a literature survey of empirical comparisons between LLMs and human research subjects, commentaries on the topic, and related work. We identify promising directions with prompting, fine-tuning, and complementary methods. We believe that LLM social simulations can already be used for exploratory research, such as pilot experiments for psychology, economics, sociology, and marketing. More widespread use may soon be possible with rapidly advancing LLM capabilities, and researchers should prioritize developing conceptual models and evaluations that can be iteratively deployed and refined at pace with ongoing AI advances. 

---
# Neural Style Transfer for Synthesising a Dataset of Ancient Egyptian Hieroglyphs 

**Authors**: Lewis Matheson Creed  

**Link**: [PDF](https://arxiv.org/pdf/2504.02163)  

**Abstract**: The limited availability of training data for low-resource languages makes applying machine learning techniques challenging. Ancient Egyptian is one such language with few resources. However, innovative applications of data augmentation methods, such as Neural Style Transfer, could overcome these barriers. This paper presents a novel method for generating datasets of ancient Egyptian hieroglyphs by applying NST to a digital typeface. Experimental results found that image classification models trained on NST-generated examples and photographs demonstrate equal performance and transferability to real unseen images of hieroglyphs. 

---
# Towards Interpretable Soft Prompts 

**Authors**: Oam Patel, Jason Wang, Nikhil Shivakumar Nayak, Suraj Srinivas, Himabindu Lakkaraju  

**Link**: [PDF](https://arxiv.org/pdf/2504.02144)  

**Abstract**: Soft prompts have been popularized as a cheap and easy way to improve task-specific LLM performance beyond few-shot prompts. Despite their origin as an automated prompting method, however, soft prompts and other trainable prompts remain a black-box method with no immediately interpretable connections to prompting. We create a novel theoretical framework for evaluating the interpretability of trainable prompts based on two desiderata: faithfulness and scrutability. We find that existing methods do not naturally satisfy our proposed interpretability criterion. Instead, our framework inspires a new direction of trainable prompting methods that explicitly optimizes for interpretability. To this end, we formulate and test new interpretability-oriented objective functions for two state-of-the-art prompt tuners: Hard Prompts Made Easy (PEZ) and RLPrompt. Our experiments with GPT-2 demonstrate a fundamental trade-off between interpretability and the task-performance of the trainable prompt, explicating the hardness of the soft prompt interpretability problem and revealing odd behavior that arises when one optimizes for an interpretability proxy. 

---
# Achieving Unanimous Consensus in Decision Making Using Multi-Agents 

**Authors**: Apurba Pokharel, Ram Dantu, Shakila Zaman, Sirisha Talapuru, Vinh Quach  

**Link**: [PDF](https://arxiv.org/pdf/2504.02128)  

**Abstract**: Blockchain consensus mechanisms have relied on algorithms such as Proof-of-Work (PoW) and Proof-of-Stake (PoS) to ensure network functionality and integrity. However, these approaches struggle with adaptability for decision-making where the opinions of each matter rather than reaching an agreement based on honest majority or weighted consensus. This paper introduces a novel deliberation-based consensus mechanism where Large Language Models (LLMs) act as rational agents engaging in structured discussions to reach a unanimous consensus. By leveraging graded consensus and a multi-round deliberation process, our approach ensures both unanimous consensus for definitive problems and graded confidence for prioritized decisions and policies. We provide a formalization of our system and use it to show that the properties of blockchains: consistency, agreement, liveness, and determinism are maintained. Moreover, experimental results demonstrate our system's feasibility, showcasing how our deliberation method's convergence, block properties, and accuracy enable decision-making on blockchain networks. We also address key challenges with this novel approach such as degeneration of thoughts, hallucinations, malicious models and nodes, resource consumption, and scalability. 

---
# Exploring LLM Reasoning Through Controlled Prompt Variations 

**Authors**: Giannis Chatziveroglou, Richard Yun, Maura Kelleher  

**Link**: [PDF](https://arxiv.org/pdf/2504.02111)  

**Abstract**: This study investigates the reasoning robustness of large language models (LLMs) on mathematical problem-solving tasks under systematically introduced input perturbations. Using the GSM8K dataset as a controlled testbed, we evaluate how well state-of-the-art models maintain logical consistency and correctness when confronted with four categories of prompt perturbations: irrelevant context, pathological instructions, factually relevant but non-essential context, and a combination of the latter two. Our experiments, conducted on thirteen open-source and closed-source LLMs, reveal that introducing irrelevant context within the model's context window significantly degrades performance, suggesting that distinguishing essential from extraneous details remains a pressing challenge. Surprisingly, performance regressions are relatively insensitive to the complexity of the reasoning task, as measured by the number of steps required, and are not strictly correlated with model size. Moreover, we observe that certain perturbations inadvertently trigger chain-of-thought-like reasoning behaviors, even without explicit prompting. Our findings highlight critical vulnerabilities in current LLMs and underscore the need for improved robustness against noisy, misleading, and contextually dense inputs, paving the way for more resilient and reliable reasoning in real-world applications. 

---
# TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining 

**Authors**: Jeffrey Li, Mohammadreza Armandpour, Iman Mirzadeh, Sachin Mehta, Vaishaal Shankar, Raviteja Vemulapalli, Samy Bengio, Oncel Tuzel, Mehrdad Farajtabar, Hadi Pouransari, Fartash Faghri  

**Link**: [PDF](https://arxiv.org/pdf/2504.02107)  

**Abstract**: Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) - orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains. 

---
# Self-Resource Allocation in Multi-Agent LLM Systems 

**Authors**: Alfonso Amayuelas, Jingbo Yang, Saaket Agashe, Ashwin Nagarajan, Antonis Antoniades, Xin Eric Wang, William Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02051)  

**Abstract**: With the development of LLMs as agents, there is a growing interest in connecting multiple agents into multi-agent systems to solve tasks concurrently, focusing on their role in task assignment and coordination. This paper explores how LLMs can effectively allocate computational tasks among multiple agents, considering factors such as cost, efficiency, and performance. In this work, we address key questions, including the effectiveness of LLMs as orchestrators and planners, comparing their effectiveness in task assignment and coordination. Our experiments demonstrate that LLMs can achieve high validity and accuracy in resource allocation tasks. We find that the planner method outperforms the orchestrator method in handling concurrent actions, resulting in improved efficiency and better utilization of agents. Additionally, we show that providing explicit information about worker capabilities enhances the allocation strategies of planners, particularly when dealing with suboptimal workers. 

---
# Urban Computing in the Era of Large Language Models 

**Authors**: Zhonghang Li, Lianghao Xia, Xubin Ren, Jiabin Tang, Tianyi Chen, Yong Xu, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02009)  

**Abstract**: Urban computing has emerged as a multidisciplinary field that harnesses data-driven technologies to address challenges and improve urban living. Traditional approaches, while beneficial, often face challenges with generalization, scalability, and contextual understanding. The advent of Large Language Models (LLMs) offers transformative potential in this domain. This survey explores the intersection of LLMs and urban computing, emphasizing the impact of LLMs in processing and analyzing urban data, enhancing decision-making, and fostering citizen engagement. We provide a concise overview of the evolution and core technologies of LLMs. Additionally, we survey their applications across key urban domains, such as transportation, public safety, and environmental monitoring, summarizing essential tasks and prior works in various urban contexts, while highlighting LLMs' functional roles and implementation patterns. Building on this, we propose potential LLM-based solutions to address unresolved challenges. To facilitate in-depth research, we compile a list of available datasets and tools applicable to diverse urban scenarios. Finally, we discuss the limitations of current approaches and outline future directions for advancing LLMs in urban computing. 

---
# LLMs Working in Harmony: A Survey on the Technological Aspects of Building Effective LLM-Based Multi Agent Systems 

**Authors**: R. M. Aratchige, W. M. K. S. Ilmini  

**Link**: [PDF](https://arxiv.org/pdf/2504.01963)  

**Abstract**: This survey investigates foundational technologies essential for developing effective Large Language Model (LLM)-based multi-agent systems. Aiming to answer how best to optimize these systems for collaborative, dynamic environments, we focus on four critical areas: Architecture, Memory, Planning, and Technologies/Frameworks. By analyzing recent advancements and their limitations - such as scalability, real-time response challenges, and agent coordination constraints, we provide a detailed view of the technological landscape. Frameworks like the Mixture of Agents architecture and the ReAct planning model exemplify current innovations, showcasing improvements in role assignment and decision-making. This review synthesizes key strengths and persistent challenges, offering practical recommendations to enhance system scalability, agent collaboration, and adaptability. Our findings provide a roadmap for future research, supporting the creation of robust, efficient multi-agent systems that advance both individual agent performance and collective system resilience. 

---
