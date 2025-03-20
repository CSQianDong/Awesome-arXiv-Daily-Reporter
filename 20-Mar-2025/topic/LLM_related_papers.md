# Pseudo-Relevance Feedback Can Improve Zero-Shot LLM-Based Dense Retrieval 

**Authors**: Hang Li, Xiao Wang, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2503.14887)  

**Abstract**: Pseudo-relevance feedback (PRF) refines queries by leveraging initially retrieved documents to improve retrieval effectiveness. In this paper, we investigate how large language models (LLMs) can facilitate PRF for zero-shot LLM-based dense retrieval, extending the recently proposed PromptReps method. Specifically, our approach uses LLMs to extract salient passage features-such as keywords and summaries-from top-ranked documents, which are then integrated into PromptReps to produce enhanced query representations. Experiments on passage retrieval benchmarks demonstrate that incorporating PRF significantly boosts retrieval performance. Notably, smaller rankers with PRF can match the effectiveness of larger rankers without PRF, highlighting PRF's potential to improve LLM-driven search while maintaining an efficient balance between effectiveness and resource usage. 

---
# RAGO: Systematic Performance Optimization for Retrieval-Augmented Generation Serving 

**Authors**: Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdanbakhsh, Vidushi Dadu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14649)  

**Abstract**: Retrieval-augmented generation (RAG), which combines large language models (LLMs) with retrievals from external knowledge databases, is emerging as a popular approach for reliable LLM serving. However, efficient RAG serving remains an open challenge due to the rapid emergence of many RAG variants and the substantial differences in workload characteristics across them. In this paper, we make three fundamental contributions to advancing RAG serving. First, we introduce RAGSchema, a structured abstraction that captures the wide range of RAG algorithms, serving as a foundation for performance optimization. Second, we analyze several representative RAG workloads with distinct RAGSchema, revealing significant performance variability across these workloads. Third, to address this variability and meet diverse performance requirements, we propose RAGO (Retrieval-Augmented Generation Optimizer), a system optimization framework for efficient RAG serving. Our evaluation shows that RAGO achieves up to a 2x increase in QPS per chip and a 55% reduction in time-to-first-token latency compared to RAG systems built on LLM-system extensions. 

---
# Do Chains-of-Thoughts of Large Language Models Suffer from Hallucinations, Cognitive Biases, or Phobias in Bayesian Reasoning? 

**Authors**: Roberto Araya  

**Link**: [PDF](https://arxiv.org/pdf/2503.15268)  

**Abstract**: Learning to reason and carefully explain arguments is central to students' cognitive, mathematical, and computational thinking development. This is particularly challenging in problems under uncertainty and in Bayesian reasoning. With the new generation of large language models (LLMs) capable of reasoning using Chain-of-Thought (CoT), there is an excellent opportunity to learn with them as they explain their reasoning through a dialogue with their artificial internal voice. It is an engaging and excellent opportunity to learn Bayesian reasoning. Furthermore, given that different LLMs sometimes arrive at opposite solutions, CoT generates opportunities for deep learning by detailed comparisons of reasonings. However, unlike humans, we found that they do not autonomously explain using ecologically valid strategies like natural frequencies, whole objects, and embodied heuristics. This is unfortunate, as these strategies help humans avoid critical mistakes and have proven pedagogical value in Bayesian reasoning. In order to overcome these biases and aid understanding and learning, we included prompts that induce LLMs to use these strategies. We found that LLMs with CoT incorporate them but not consistently. They show persistent biases towards symbolic reasoning and avoidance or phobia of ecologically valid strategies. 

---
# Aligning Crowd-sourced Human Feedback for Reinforcement Learning on Code Generation by Large Language Models 

**Authors**: Man Fai Wong, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15129)  

**Abstract**: This paper studies how AI-assisted programming and large language models (LLM) improve software developers' ability via AI tools (LLM agents) like Github Copilot and Amazon CodeWhisperer, while integrating human feedback to enhance reinforcement learning (RLHF) with crowd-sourced computation to enhance text-to-code generation. Additionally, we demonstrate that our Bayesian optimization framework supports AI alignment in code generation by distributing the feedback collection burden, highlighting the value of collecting human feedback of good quality. Our empirical evaluations demonstrate the efficacy of this approach, showcasing how LLM agents can be effectively trained for improved text-to-code generation. Our Bayesian optimization framework can be designed for general domain-specific languages, promoting the alignment of large language model capabilities with human feedback in AI-assisted programming for code generation. 

---
# From 1,000,000 Users to Every User: Scaling Up Personalized Preference for User-level Alignment 

**Authors**: Jia-Nan Li, Jian Guan, Songhao Wu, Wei Wu, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.15463)  

**Abstract**: Large language models (LLMs) have traditionally been aligned through one-size-fits-all approaches that assume uniform human preferences, fundamentally overlooking the diversity in user values and needs. This paper introduces a comprehensive framework for scalable personalized alignment of LLMs. We establish a systematic preference space characterizing psychological and behavioral dimensions, alongside diverse persona representations for robust preference inference in real-world scenarios. Building upon this foundation, we introduce \textsc{AlignX}, a large-scale dataset of over 1.3 million personalized preference examples, and develop two complementary alignment approaches: \textit{in-context alignment} directly conditioning on persona representations and \textit{preference-bridged alignment} modeling intermediate preference distributions. Extensive experiments demonstrate substantial improvements over existing methods, with an average 17.06\% accuracy gain across four benchmarks while exhibiting a strong adaptation capability to novel preferences, robustness to limited user data, and precise preference controllability. These results validate our framework's effectiveness, advancing toward truly user-adaptive AI systems. 

---
# Visual Position Prompt for MLLM based Visual Grounding 

**Authors**: Wei Tang, Yanpeng Sun, Qinying Gu, Zechao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15426)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) excel at various image-related tasks, they encounter challenges in precisely aligning coordinates with spatial information within images, particularly in position-aware tasks such as visual grounding. This limitation arises from two key factors. First, MLLMs lack explicit spatial references, making it difficult to associate textual descriptions with precise image locations. Second, their feature extraction processes prioritize global context over fine-grained spatial details, leading to weak localization capability. To address this issue, we introduce VPP-LLaVA, an MLLM equipped with Visual Position Prompt (VPP) to improve its grounding capability. VPP-LLaVA integrates two complementary mechanisms. The global VPP overlays learnable, axis-like embeddings onto the input image to provide structured spatial cues. The local VPP focuses on fine-grained localization by incorporating position-aware queries, which suggests probable object locations. We also introduce a VPP-SFT dataset with 0.6M samples, consolidating high-quality visual grounding data into a compact format for efficient model training. Training on this dataset with VPP enhances the model's performance, achieving state-of-the-art results on standard grounding benchmarks despite using fewer training samples compared to other MLLMs like MiniGPT-v2, which rely on much larger datasets ($\sim$21M samples). The code and VPP-SFT dataset will be available at this https URL upon acceptance. 

---
# MAMM-Refine: A Recipe for Improving Faithfulness in Generation with Multi-Agent Collaboration 

**Authors**: David Wan, Justin Chih-Yao Chen, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2503.15272)  

**Abstract**: Multi-agent collaboration among models has shown promise in reasoning tasks but is underexplored in long-form generation tasks like summarization and question-answering. We extend multi-agent multi-model reasoning to generation, specifically to improving faithfulness through refinement, i.e., revising model-generated outputs to remove factual inconsistencies. We investigate how iterative collaboration among multiple instances and types of large language models (LLMs) enhances subtasks in the refinement process, such as error detection, critiquing unfaithful sentences, and making corrections based on critiques. We design intrinsic evaluations for each subtask, with our findings indicating that both multi-agent (multiple instances) and multi-model (diverse LLM types) approaches benefit error detection and critiquing. Additionally, reframing critiquing and refinement as reranking rather than generation tasks improves multi-agent performance. We consolidate these insights into a final "recipe" called Multi-Agent Multi-Model Refinement (MAMM-Refine), where multi-agent and multi-model collaboration significantly boosts performance on three summarization datasets as well as on long-form question answering, demonstrating the effectiveness and generalizability of our recipe. 

---
# Automated Non-Functional Requirements Generation in Software Engineering with Large Language Models: A Comparative Study 

**Authors**: Jomar Thomas Almonte, Santhosh Anitha Boominathan, Nathalia Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2503.15248)  

**Abstract**: Neglecting non-functional requirements (NFRs) early in software development can lead to critical challenges. Despite their importance, NFRs are often overlooked or difficult to identify, impacting software quality. To support requirements engineers in eliciting NFRs, we developed a framework that leverages Large Language Models (LLMs) to derive quality-driven NFRs from functional requirements (FRs). Using a custom prompting technique within a Deno-based pipeline, the system identifies relevant quality attributes for each functional requirement and generates corresponding NFRs, aiding systematic integration. A crucial aspect is evaluating the quality and suitability of these generated requirements. Can LLMs produce high-quality NFR suggestions? Using 34 functional requirements - selected as a representative subset of 3,964 FRs-the LLMs inferred applicable attributes based on the ISO/IEC 25010:2023 standard, generating 1,593 NFRs. A horizontal evaluation covered three dimensions: NFR validity, applicability of quality attributes, and classification precision. Ten industry software quality evaluators, averaging 13 years of experience, assessed a subset for relevance and quality. The evaluation showed strong alignment between LLM-generated NFRs and expert assessments, with median validity and applicability scores of 5.0 (means: 4.63 and 4.59, respectively) on a 1-5 scale. In the classification task, 80.4% of LLM-assigned attributes matched expert choices, with 8.3% near misses and 11.3% mismatches. A comparative analysis of eight LLMs highlighted variations in performance, with gemini-1.5-pro exhibiting the highest attribute accuracy, while llama-3.3-70B achieved higher validity and applicability scores. These findings provide insights into the feasibility of using LLMs for automated NFR generation and lay the foundation for further exploration of AI-assisted requirements engineering. 

---
# Real-world validation of a multimodal LLM-powered pipeline for High-Accuracy Clinical Trial Patient Matching leveraging EHR data 

**Authors**: Anatole Callies, Quentin Bodinier, Philippe Ravaud, Kourosh Davarpanah  

**Link**: [PDF](https://arxiv.org/pdf/2503.15374)  

**Abstract**: Background: Patient recruitment in clinical trials is hindered by complex eligibility criteria and labor-intensive chart reviews. Prior research using text-only models have struggled to address this problem in a reliable and scalable way due to (1) limited reasoning capabilities, (2) information loss from converting visual records to text, and (3) lack of a generic EHR integration to extract patient data.
Methods: We introduce a broadly applicable, integration-free, LLM-powered pipeline that automates patient-trial matching using unprocessed documents extracted from EHRs. Our approach leverages (1) the new reasoning-LLM paradigm, enabling the assessment of even the most complex criteria, (2) visual capabilities of latest LLMs to interpret medical records without lossy image-to-text conversions, and (3) multimodal embeddings for efficient medical record search. The pipeline was validated on the n2c2 2018 cohort selection dataset (288 diabetic patients) and a real-world dataset composed of 485 patients from 30 different sites matched against 36 diverse trials.
Results: On the n2c2 dataset, our method achieved a new state-of-the-art criterion-level accuracy of 93\%. In real-world trials, the pipeline yielded an accuracy of 87\%, undermined by the difficulty to replicate human decision-making when medical records lack sufficient information. Nevertheless, users were able to review overall eligibility in under 9 minutes per patient on average, representing an 80\% improvement over traditional manual chart reviews.
Conclusion: This pipeline demonstrates robust performance in clinical trial patient matching without requiring custom integration with site systems or trial-specific tailoring, thereby enabling scalable deployment across sites seeking to leverage AI for patient matching. 

---
# Exploring Large Language Models for Word Games:Who is the Spy? 

**Authors**: Chentian Wei, Jiewei Chen, Jinzhu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15235)  

**Abstract**: Word games hold significant research value for natural language processing (NLP), game theory, and related fields due to their rule-based and situational nature. This study explores how large language models (LLMs) can be effectively involved in word games and proposes a training-free framework. "Shei Shi Wo Di" or "Who is the Spy" in English, is a classic word game. Using this game as an example, we introduce a Chain-of-Thought (CoT)-based scheduling framework to enable LLMs to achieve excellent performance in tasks such as inferring role words and disguising their identities. We evaluate the framework's performance based on game success rates and the accuracy of the LLM agents' analytical results. Experimental results affirm the framework's effectiveness, demonstrating notable improvements in LLM performance across multiple datasets. This work highlights the potential of LLMs in mastering situational reasoning and social interactions within structured game environments. Our code is publicly available at this https URL. 

---
# BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity? 

**Authors**: Pierre Chambon, Baptiste Roziere, Benoit Sagot, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2503.15242)  

**Abstract**: We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time. 

---
# Text-Derived Relational Graph-Enhanced Network for Skeleton-Based Action Segmentation 

**Authors**: Haoyu Ji, Bowen Chen, Weihong Ren, Wenze Huang, Zhihao Yang, Zhiyong Wang, Honghai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15126)  

**Abstract**: Skeleton-based Temporal Action Segmentation (STAS) aims to segment and recognize various actions from long, untrimmed sequences of human skeletal movements. Current STAS methods typically employ spatio-temporal modeling to establish dependencies among joints as well as frames, and utilize one-hot encoding with cross-entropy loss for frame-wise classification supervision. However, these methods overlook the intrinsic correlations among joints and actions within skeletal features, leading to a limited understanding of human movements. To address this, we propose a Text-Derived Relational Graph-Enhanced Network (TRG-Net) that leverages prior graphs generated by Large Language Models (LLM) to enhance both modeling and supervision. For modeling, the Dynamic Spatio-Temporal Fusion Modeling (DSFM) method incorporates Text-Derived Joint Graphs (TJG) with channel- and frame-level dynamic adaptation to effectively model spatial relations, while integrating spatio-temporal core features during temporal modeling. For supervision, the Absolute-Relative Inter-Class Supervision (ARIS) method employs contrastive learning between action features and text embeddings to regularize the absolute class distributions, and utilizes Text-Derived Action Graphs (TAG) to capture the relative inter-class relationships among action features. Additionally, we propose a Spatial-Aware Enhancement Processing (SAEP) method, which incorporates random joint occlusion and axial rotation to enhance spatial generalization. Performance evaluations on four public datasets demonstrate that TRG-Net achieves state-of-the-art results. 

---
# POSTA: A Go-to Framework for Customized Artistic Poster Generation 

**Authors**: Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14908)  

**Abstract**: Poster design is a critical medium for visual communication. Prior work has explored automatic poster design using deep learning techniques, but these approaches lack text accuracy, user customization, and aesthetic appeal, limiting their applicability in artistic domains such as movies and exhibitions, where both clear content delivery and visual impact are essential. To address these limitations, we present POSTA: a modular framework powered by diffusion models and multimodal large language models (MLLMs) for customized artistic poster generation. The framework consists of three modules. Background Diffusion creates a themed background based on user input. Design MLLM then generates layout and typography elements that align with and complement the background style. Finally, to enhance the poster's aesthetic appeal, ArtText Diffusion applies additional stylization to key text elements. The final result is a visually cohesive and appealing poster, with a fully modular process that allows for complete customization. To train our models, we develop the PosterArt dataset, comprising high-quality artistic posters annotated with layout, typography, and pixel-level stylized text segmentation. Our comprehensive experimental analysis demonstrates POSTA's exceptional controllability and design diversity, outperforming existing models in both text accuracy and aesthetic quality. 

---
# Mitigating Object Hallucinations in MLLMs via Multi-Frequency Perturbations 

**Authors**: Shuo Li, Jiajun Sun, Guodong Zheng, Xiaoran Fan, Yujiong Shen, Yi Lu, Zhiheng Xi, Yuming Yang, Wenming Tan, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14895)  

**Abstract**: Recently, multimodal large language models (MLLMs) have demonstrated remarkable performance in visual-language tasks. However, the authenticity of the responses generated by MLLMs is often compromised by object hallucinations. We identify that a key cause of these hallucinations is the model's over-susceptibility to specific image frequency features in detecting objects. In this paper, we introduce Multi-Frequency Perturbations (MFP), a simple, cost-effective, and pluggable method that leverages both low-frequency and high-frequency features of images to perturb visual feature representations and explicitly suppress redundant frequency-domain features during inference, thereby mitigating hallucinations. Experimental results demonstrate that our method significantly mitigates object hallucinations across various model architectures. Furthermore, as a training-time method, MFP can be combined with inference-time methods to achieve state-of-the-art performance on the CHAIR benchmark. 

---
# MetaLadder: Ascending Mathematical Solution Quality via Analogical-Problem Reasoning Transfer 

**Authors**: Honglin Lin, Zhuoshi Pan, Yu Li, Qizhi Pei, Xin Gao, Mengzhang Cai, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14891)  

**Abstract**: Large Language Models (LLMs) have demonstrated promising capabilities in solving mathematical reasoning tasks, leveraging Chain-of-Thought (CoT) data as a vital component in guiding answer generation. Current paradigms typically generate CoT and answers directly for a given problem, diverging from human problem-solving strategies to some extent. Humans often solve problems by recalling analogous cases and leveraging their solutions to reason about the current task. Inspired by this cognitive process, we propose \textbf{MetaLadder}, a novel framework that explicitly prompts LLMs to recall and reflect on meta-problems, those structurally or semantically analogous problems, alongside their CoT solutions before addressing the target problem. Additionally, we introduce a problem-restating mechanism to enhance the model's comprehension of the target problem by regenerating the original question, which further improves reasoning accuracy. Therefore, the model can achieve reasoning transfer from analogical problems, mimicking human-like "learning from examples" and generalization abilities. Extensive experiments on mathematical benchmarks demonstrate that our MetaLadder significantly boosts LLMs' problem-solving accuracy, largely outperforming standard CoT-based methods (\textbf{10.3\%} accuracy gain) and other methods. Our code and data has been released at this https URL. 

---
# MASS: Mathematical Data Selection via Skill Graphs for Pretraining Large Language Models 

**Authors**: Jiazheng Li, Lu Yu, Qing Cui, Zhiqiang Zhang, Jun Zhou, Yanfang Ye, Chuxu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14917)  

**Abstract**: High-quality data plays a critical role in the pretraining and fine-tuning of large language models (LLMs), even determining their performance ceiling to some degree. Consequently, numerous data selection methods have been proposed to identify subsets of data that can effectively and efficiently enhance model performance. However, most of these methods focus on general data selection and tend to overlook the specific nuances of domain-related data. In this paper, we introduce MASS, a \textbf{MA}thematical data \textbf{S}election framework using the \textbf{S}kill graph for pretraining LLMs in the mathematical reasoning domain. By taking into account the unique characteristics of mathematics and reasoning, we construct a skill graph that captures the mathematical skills and their interrelations from a reference dataset. This skill graph guides us in assigning quality scores to the target dataset, enabling us to select the top-ranked subset which is further used to pretrain LLMs. Experimental results demonstrate the efficiency and effectiveness of MASS across different model sizes (1B and 7B) and pretraining datasets (web data and synthetic data). Specifically, in terms of efficiency, models trained on subsets selected by MASS can achieve similar performance to models trained on the original datasets, with a significant reduction in the number of trained tokens - ranging from 50\% to 70\% fewer tokens. In terms of effectiveness, when trained on the same amount of tokens, models trained on the data selected by MASS outperform those trained on the original datasets by 3.3\% to 5.9\%. These results underscore the potential of MASS to improve both the efficiency and effectiveness of pretraining LLMs. 

---
# Reasoning Effort and Problem Complexity: A Scaling Analysis in LLMs 

**Authors**: Benjamin Estermann, Roger Wattenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2503.15113)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable text generation capabilities, and recent advances in training paradigms have led to breakthroughs in their reasoning performance. In this work, we investigate how the reasoning effort of such models scales with problem complexity. We use the infinitely scalable Tents puzzle, which has a known linear-time solution, to analyze this scaling behavior. Our results show that reasoning effort scales with problem size, but only up to a critical problem complexity. Beyond this threshold, the reasoning effort does not continue to increase, and may even decrease. This observation highlights a critical limitation in the logical coherence of current LLMs as problem complexity increases, and underscores the need for strategies to improve reasoning scalability. Furthermore, our results reveal significant performance differences between current state-of-the-art reasoning models when faced with increasingly complex logical puzzles. 

---
# TruthLens:A Training-Free Paradigm for DeepFake Detection 

**Authors**: Ritabrata Chakraborty, Rajatsubhra Chakraborty, Ali Khaleghi Rahimian, Thomas MacDougall  

**Link**: [PDF](https://arxiv.org/pdf/2503.15342)  

**Abstract**: The proliferation of synthetic images generated by advanced AI models poses significant challenges in identifying and understanding manipulated visual content. Current fake image detection methods predominantly rely on binary classification models that focus on accuracy while often neglecting interpretability, leaving users without clear insights into why an image is deemed real or fake. To bridge this gap, we introduce TruthLens, a novel training-free framework that reimagines deepfake detection as a visual question-answering (VQA) task. TruthLens utilizes state-of-the-art large vision-language models (LVLMs) to observe and describe visual artifacts and combines this with the reasoning capabilities of large language models (LLMs) like GPT-4 to analyze and aggregate evidence into informed decisions. By adopting a multimodal approach, TruthLens seamlessly integrates visual and semantic reasoning to not only classify images as real or fake but also provide interpretable explanations for its decisions. This transparency enhances trust and provides valuable insights into the artifacts that signal synthetic content. Extensive evaluations demonstrate that TruthLens outperforms conventional methods, achieving high accuracy on challenging datasets while maintaining a strong emphasis on explainability. By reframing deepfake detection as a reasoning-driven process, TruthLens establishes a new paradigm in combating synthetic media, combining cutting-edge performance with interpretability to address the growing threats of visual disinformation. 

---
# Assessing Large Language Models for Automated Feedback Generation in Learning Programming Problem Solving 

**Authors**: Priscylla Silva, Evandro Costa  

**Link**: [PDF](https://arxiv.org/pdf/2503.14630)  

**Abstract**: Providing effective feedback is important for student learning in programming problem-solving. In this sense, Large Language Models (LLMs) have emerged as potential tools to automate feedback generation. However, their reliability and ability to identify reasoning errors in student code remain not well understood. This study evaluates the performance of four LLMs (GPT-4o, GPT-4o mini, GPT-4-Turbo, and Gemini-1.5-pro) on a benchmark dataset of 45 student solutions. We assessed the models' capacity to provide accurate and insightful feedback, particularly in identifying reasoning mistakes. Our analysis reveals that 63\% of feedback hints were accurate and complete, while 37\% contained mistakes, including incorrect line identification, flawed explanations, or hallucinated issues. These findings highlight the potential and limitations of LLMs in programming education and underscore the need for improvements to enhance reliability and minimize risks in educational applications. 

---
# ConQuer: A Framework for Concept-Based Quiz Generation 

**Authors**: Yicheng Fu, Zikui Wang, Liuxin Yang, Meiqing Huo, Zhongdongming Dai  

**Link**: [PDF](https://arxiv.org/pdf/2503.14662)  

**Abstract**: Quizzes play a crucial role in education by reinforcing students' understanding of key concepts and encouraging self-directed exploration. However, compiling high-quality quizzes can be challenging and require deep expertise and insight into specific subject matter. Although LLMs have greatly enhanced the efficiency of quiz generation, concerns remain regarding the quality of these AI-generated quizzes and their educational impact on students. To address these issues, we introduce ConQuer, a concept-based quiz generation framework that leverages external knowledge sources. We employ comprehensive evaluation dimensions to assess the quality of the generated quizzes, using LLMs as judges. Our experiment results demonstrate a 4.8% improvement in evaluation scores and a 77.52% win rate in pairwise comparisons against baseline quiz sets. Ablation studies further underscore the effectiveness of each component in our framework. Code available at this https URL. 

---
# Policy Frameworks for Transparent Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Yihang Chen, Haikang Deng, Kaiqiao Han, Qingyue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.14521)  

**Abstract**: Chain-of-Thought (CoT) reasoning enhances large language models (LLMs) by decomposing complex problems into step-by-step solutions, improving performance on reasoning tasks. However, current CoT disclosure policies vary widely across different models in frontend visibility, API access, and pricing strategies, lacking a unified policy framework. This paper analyzes the dual-edged implications of full CoT disclosure: while it empowers small-model distillation, fosters trust, and enables error diagnosis, it also risks violating intellectual property, enabling misuse, and incurring operational costs. We propose a tiered-access policy framework that balances transparency, accountability, and security by tailoring CoT availability to academic, business, and general users through ethical licensing, structured reasoning outputs, and cross-tier safeguards. By harmonizing accessibility with ethical and operational considerations, this framework aims to advance responsible AI deployment while mitigating risks of misuse or misinterpretation. 

---
# Evaluating Bias in Retrieval-Augmented Medical Question-Answering Systems 

**Authors**: Yuelyu Ji, Hang Zhang, Yanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15454)  

**Abstract**: Medical QA systems powered by Retrieval-Augmented Generation (RAG) models support clinical decision-making but may introduce biases related to race, gender, and social determinants of health. We systematically evaluate biases in RAG-based LLM by examining demographic-sensitive queries and measuring retrieval discrepancies. Using datasets like MMLU and MedMCQA, we analyze retrieval overlap and correctness disparities. Our findings reveal substantial demographic disparities within RAG pipelines, emphasizing the critical need for retrieval methods that explicitly account for fairness to ensure equitable clinical decision-making. 

---
# SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation 

**Authors**: Thomas Pickard, Aline Villavicencio, Maggie Mi, Wei He, Dylan Phelps, Carolina Scarton, Marco Idiart  

**Link**: [PDF](https://arxiv.org/pdf/2503.15358)  

**Abstract**: Idiomatic expressions present a unique challenge in NLP, as their meanings are often not directly inferable from their constituent words. Despite recent advancements in Large Language Models (LLMs), idiomaticity remains a significant obstacle to robust semantic representation. We present datasets and tasks for SemEval-2025 Task 1: AdMiRe (Advancing Multimodal Idiomaticity Representation), which challenges the community to assess and improve models' ability to interpret idiomatic expressions in multimodal contexts and in multiple languages. Participants competed in two subtasks: ranking images based on their alignment with idiomatic or literal meanings, and predicting the next image in a sequence. The most effective methods achieved human-level performance by leveraging pretrained LLMs and vision-language models in mixture-of-experts settings, with multiple queries used to smooth over the weaknesses in these models' representations of idiomaticity. 

---
# Inside-Out: Hidden Factual Knowledge in LLMs 

**Authors**: Zorik Gekhman, Eyal Ben David, Hadas Orgad, Eran Ofek, Yonatan Belinkov, Idan Szpector, Jonathan Herzig, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2503.15299)  

**Abstract**: This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) puts a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first. 

---
# SPILL: Domain-Adaptive Intent Clustering based on Selection and Pooling with Large Language Models 

**Authors**: I-Fan Lin, Faegheh Hasibi, Suzan Verberne  

**Link**: [PDF](https://arxiv.org/pdf/2503.15351)  

**Abstract**: In this paper, we propose Selection and Pooling with Large Language Models (SPILL), an intuitive and domain-adaptive method for intent clustering without fine-tuning. Existing embeddings-based clustering methods rely on a few labeled examples or unsupervised fine-tuning to optimize results for each new dataset, which makes them less generalizable to multiple datasets. Our goal is to make these existing embedders more generalizable to new domain datasets without further fine-tuning. Inspired by our theoretical derivation and simulation results on the effectiveness of sampling and pooling techniques, we view the clustering task as a small-scale selection problem. A good solution to this problem is associated with better clustering performance. Accordingly, we propose a two-stage approach: First, for each utterance (referred to as the seed), we derive its embedding using an existing embedder. Then, we apply a distance metric to select a pool of candidates close to the seed. Because the embedder is not optimized for new datasets, in the second stage, we use an LLM to further select utterances from these candidates that share the same intent as the seed. Finally, we pool these selected candidates with the seed to derive a refined embedding for the seed. We found that our method generally outperforms directly using an embedder, and it achieves comparable results to other state-of-the-art studies, even those that use much larger models and require fine-tuning, showing its strength and efficiency. Our results indicate that our method enables existing embedders to be further improved without additional fine-tuning, making them more adaptable to new domain datasets. Additionally, viewing the clustering task as a small-scale selection problem gives the potential of using LLMs to customize clustering tasks according to the user's goals. 

---
# ELTEX: A Framework for Domain-Driven Synthetic Data Generation 

**Authors**: Arina Razmyslovich, Kseniia Murasheva, Sofia Sedlova, Julien Capitaine, Eugene Dmitriev  

**Link**: [PDF](https://arxiv.org/pdf/2503.15055)  

**Abstract**: We present ELTEX (Efficient LLM Token Extraction), a domain-driven framework for generating high-quality synthetic training data in specialized domains. While Large Language Models (LLMs) have shown impressive general capabilities, their performance in specialized domains like cybersecurity remains limited by the scarcity of domain-specific training data. ELTEX addresses this challenge by systematically integrating explicit domain indicator extraction with dynamic prompting to preserve critical domain knowledge throughout the generation process. We demonstrate ELTEX's effectiveness in the context of blockchain-related cyberattack detection, where we fine-tune Gemma-2B using various combinations of real and ELTEX-generated data. Our results show that the ELTEX-enhanced model achieves performance competitive with GPT-4 across both standard classification metrics and uncertainty calibration, while requiring significantly fewer computational resources. We release a curated synthetic dataset of social media texts for cyberattack detection in blockchain. Our work demonstrates that domain-driven synthetic data generation can effectively bridge the performance gap between resource-efficient models and larger architectures in specialized domains. 

---
# Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice Question Answering 

**Authors**: Francesco Maria Molfese, Luca Moroni, Luca Gioffrè, Alessandro Scirè, Simone Conia, Roberto Navigli  

**Link**: [PDF](https://arxiv.org/pdf/2503.14996)  

**Abstract**: One of the most widely used tasks to evaluate Large Language Models (LLMs) is Multiple-Choice Question Answering (MCQA). While open-ended question answering tasks are more challenging to evaluate, MCQA tasks are, in principle, easier to assess, as the model's answer is thought to be simple to extract and is directly compared to a set of predefined choices. However, recent studies have started to question the reliability of MCQA evaluation, showing that multiple factors can significantly impact the reported performance of LLMs, especially when the model generates free-form text before selecting one of the answer choices. In this work, we shed light on the inconsistencies of MCQA evaluation strategies, which can lead to inaccurate and misleading model comparisons. We systematically analyze whether existing answer extraction methods are aligned with human judgment, and how they are influenced by answer constraints in the prompt across different domains. Our experiments demonstrate that traditional evaluation strategies often underestimate LLM capabilities, while LLM-based answer extractors are prone to systematic errors. Moreover, we reveal a fundamental trade-off between including format constraints in the prompt to simplify answer extraction and allowing models to generate free-form text to improve reasoning. Our findings call for standardized evaluation methodologies and highlight the need for more reliable and consistent MCQA evaluation practices. 

---
# HaploVL: A Single-Transformer Baseline for Multi-Modal Understanding 

**Authors**: Rui Yang, Lin Song, Yicheng Xiao, Runhui Huang, Yixiao Ge, Ying Shan, Hengshuang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.14694)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly propelled the development of large multi-modal models (LMMs), highlighting the potential for general and intelligent assistants. However, most LMMs model visual and textual modalities separately, leading to recent efforts to develop native LMMs using a single transformer. Despite the promise, these native models are resource-intensive and often exhibit performance gaps compared to their compositional counterparts. To alleviate this issue, we propose a simple yet efficient method to construct a baseline for the native and end-to-end large multi-modal model in a single transformer. First, we propose a new early-fusion LMM that can fuse multi-modal inputs in the early stage and respond to visual instructions in an auto-regressive manner. Second, we devise an efficient training recipe for the proposed model, which harnesses the prior knowledge of the pre-trained models, addressing both the performance limitations and the challenge of resource consumption. The proposed model demonstrates superior performance compared to other LMMs using one transformer and significantly narrows the performance gap with compositional LMMs. 

---
# Generating Medically-Informed Explanations for Depression Detection using LLMs 

**Authors**: Xiangyong Chen, Xiaochuan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.14671)  

**Abstract**: Early detection of depression from social media data offers a valuable opportunity for timely intervention. However, this task poses significant challenges, requiring both professional medical knowledge and the development of accurate and explainable models. In this paper, we propose LLM-MTD (Large Language Model for Multi-Task Depression Detection), a novel approach that leverages a pre-trained large language model to simultaneously classify social media posts for depression and generate textual explanations grounded in medical diagnostic criteria. We train our model using a multi-task learning framework with a combined loss function that optimizes both classification accuracy and explanation quality. We evaluate LLM-MTD on the benchmark Reddit Self-Reported Depression Dataset (RSDD) and compare its performance against several competitive baseline methods, including traditional machine learning and fine-tuned BERT. Our experimental results demonstrate that LLM-MTD achieves state-of-the-art performance in depression detection, showing significant improvements in AUPRC and other key metrics. Furthermore, human evaluation of the generated explanations reveals their relevance, completeness, and medical accuracy, highlighting the enhanced interpretability of our approach. This work contributes a novel methodology for depression detection that combines the power of large language models with the crucial aspect of explainability. 

---
# Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and Culturally Aware Arabic LLM 

**Authors**: Yazeed Alnumay, Alexandre Barbet, Anna Bialas, William Darling, Shaan Desai, Joan Devassy, Kyle Duffy, Stephanie Howe, Olivia Lasche, Justin Lee, Anirudh Shrinivason, Jennifer Tracey  

**Link**: [PDF](https://arxiv.org/pdf/2503.14603)  

**Abstract**: Building high-quality large language models (LLMs) for enterprise Arabic applications remains challenging due to the limited availability of digitized Arabic data. In this work, we present a data synthesis and refinement strategy to help address this problem, namely, by leveraging synthetic data generation and human-in-the-loop annotation to expand our Arabic training corpus. We further present our iterative post training recipe that is essential to achieving state-of-the-art performance in aligning the model with human preferences, a critical aspect to enterprise use cases. The culmination of this effort is the release of a small, 7B, open-weight model that outperforms similarly sized peers in head-to-head comparisons and on Arabic-focused benchmarks covering cultural knowledge, instruction following, RAG, and contextual faithfulness. 

---
# Solla: Towards a Speech-Oriented LLM That Hears Acoustic Context 

**Authors**: Junyi Ao, Dekun Chen, Xiaohai Tian, Wenjie Feng, Jun Zhang, Lu Lu, Yuxuan Wang, Haizhou Li, Zhizheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15338)  

**Abstract**: Large Language Models (LLMs) have recently shown remarkable ability to process not only text but also multimodal inputs such as speech and audio. However, most existing models primarily focus on analyzing input signals using text instructions, overlooking scenarios in which speech instructions and audio are mixed and serve as inputs to the model. To address these challenges, we introduce Solla, a novel framework designed to understand speech-based questions and hear the acoustic context concurrently. Solla incorporates an audio tagging module to effectively identify and represent audio events, as well as an ASR-assisted prediction method to improve comprehension of spoken content. To rigorously evaluate Solla and other publicly available models, we propose a new benchmark dataset called SA-Eval, which includes three tasks: audio event classification, audio captioning, and audio question answering. SA-Eval has diverse speech instruction with various speaking styles, encompassing two difficulty levels, easy and hard, to capture the range of real-world acoustic conditions. Experimental results show that Solla performs on par with or outperforms baseline models on both the easy and hard test sets, underscoring its effectiveness in jointly understanding speech and audio. 

---
# Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification 

**Authors**: Shichen Li, Zhongqing Wang, Zheyu Zhao, Yue Zhang, Peifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15117)  

**Abstract**: Model editing aims at selectively updating a small subset of a neural model's parameters with an interpretable strategy to achieve desired modifications. It can significantly reduce computational costs to adapt to large language models (LLMs). Given its ability to precisely target critical components within LLMs, model editing shows great potential for efficient fine-tuning applications. In this work, we investigate model editing to serve an efficient method for adapting LLMs to solve aspect-based sentiment classification. Through causal interventions, we trace and determine which neuron hidden states are essential for the prediction of the model. By performing interventions and restorations on each component of an LLM, we identify the importance of these components for aspect-based sentiment classification. Our findings reveal that a distinct set of mid-layer representations is essential for detecting the sentiment polarity of given aspect words. Leveraging these insights, we develop a model editing approach that focuses exclusively on these critical parts of the LLM, leading to a more efficient method for adapting LLMs. Our in-domain and out-of-domain experiments demonstrate that this approach achieves competitive results compared to the currently strongest methods with significantly fewer trainable parameters, highlighting a more efficient and interpretable fine-tuning strategy. 

---
# A Review on Large Language Models for Visual Analytics 

**Authors**: Navya Sonal Agarwal, Sanjay Kumar Sonbhadra  

**Link**: [PDF](https://arxiv.org/pdf/2503.15176)  

**Abstract**: This paper provides a comprehensive review of the integration of Large Language Models (LLMs) with visual analytics, addressing their foundational concepts, capabilities, and wide-ranging applications. It begins by outlining the theoretical underpinnings of visual analytics and the transformative potential of LLMs, specifically focusing on their roles in natural language understanding, natural language generation, dialogue systems, and text-to-media transformations. The review further investigates how the synergy between LLMs and visual analytics enhances data interpretation, visualization techniques, and interactive exploration capabilities. Key tools and platforms including LIDA, Chat2VIS, Julius AI, and Zoho Analytics, along with specialized multimodal models such as ChartLlama and CharXIV, are critically evaluated. The paper discusses their functionalities, strengths, and limitations in supporting data exploration, visualization enhancement, automated reporting, and insight extraction. The taxonomy of LLM tasks, ranging from natural language understanding (NLU), natural language generation (NLG), to dialogue systems and text-to-media transformations, is systematically explored. This review provides a SWOT analysis of integrating Large Language Models (LLMs) with visual analytics, highlighting strengths like accessibility and flexibility, weaknesses such as computational demands and biases, opportunities in multimodal integration and user collaboration, and threats including privacy concerns and skill degradation. It emphasizes addressing ethical considerations and methodological improvements for effective integration. 

---
# Uncertainty Distillation: Teaching Language Models to Express Semantic Confidence 

**Authors**: Sophia Hager, David Mueller, Kevin Duh, Nicholas Andrews  

**Link**: [PDF](https://arxiv.org/pdf/2503.14749)  

**Abstract**: As large language models (LLMs) are increasingly used for factual question-answering, it becomes more important for LLMs to have the capability to communicate the likelihood that their answer is correct. For these verbalized expressions of uncertainty to be meaningful, they should reflect the error rates at the expressed level of confidence. However, when prompted to express confidence, the error rates of current LLMs are inconsistent with their communicated confidences, highlighting the need for uncertainty quantification methods. Many prior methods calculate lexical uncertainty, estimating a model's confidence in the specific string it generated. In some cases, however, it may be more useful to estimate semantic uncertainty, or the model's confidence in the answer regardless of how it is verbalized. We propose a simple procedure, uncertainty distillation, to teach an LLM to verbalize calibrated semantic confidences. Using held-out data to map initial uncertainty estimates to meaningful probabilities, we create examples annotated with verbalized probabilities for supervised fine-tuning. We demonstrate our method yields verbalized confidences that correlate with observed error rates with a small fine-tuned language model as well as with larger instruction-tuned models, and find that our semantic uncertainty correlates well with lexical uncertainty on short answers. 

---
