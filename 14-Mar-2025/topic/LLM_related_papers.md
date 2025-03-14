# Siege: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search 

**Authors**: Andy Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.10619)  

**Abstract**: We introduce Siege, a multi-turn adversarial framework that models the gradual erosion of Large Language Model (LLM) safety through a tree search perspective. Unlike single-turn jailbreaks that rely on one meticulously engineered prompt, Siege expands the conversation at each turn in a breadth-first fashion, branching out multiple adversarial prompts that exploit partial compliance from previous responses. By tracking these incremental policy leaks and re-injecting them into subsequent queries, Siege reveals how minor concessions can accumulate into fully disallowed outputs. Evaluations on the JailbreakBench dataset show that Siege achieves a 100% success rate on GPT-3.5-turbo and 97% on GPT-4 in a single multi-turn run, using fewer queries than baselines such as Crescendo or GOAT. This tree search methodology offers an in-depth view of how model safeguards degrade over successive dialogue turns, underscoring the urgency of robust multi-turn testing procedures for language models. 

---
# LLM Agents Display Human Biases but Exhibit Distinct Learning Patterns 

**Authors**: Idan Horowitz, Ori Plonsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.10248)  

**Abstract**: We investigate the choice patterns of Large Language Models (LLMs) in the context of Decisions from Experience tasks that involve repeated choice and learning from feedback, and compare their behavior to human participants. We find that on the aggregate, LLMs appear to display behavioral biases similar to humans: both exhibit underweighting rare events and correlation effects. However, more nuanced analyses of the choice patterns reveal that this happens for very different reasons. LLMs exhibit strong recency biases, unlike humans, who appear to respond in more sophisticated ways. While these different processes may lead to similar behavior on average, choice patterns contingent on recent events differ vastly between the two groups. Specifically, phenomena such as ``surprise triggers change" and the ``wavy recency effect of rare events" are robustly observed in humans, but entirely absent in LLMs. Our findings provide insights into the limitations of using LLMs to simulate and predict humans in learning environments and highlight the need for refined analyses of their behavior when investigating whether they replicate human decision making tendencies. 

---
# StepMathAgent: A Step-Wise Agent for Evaluating Mathematical Processes through Tree-of-Error 

**Authors**: Shu-Xun Yang, Cunxiang Wang, Yidong Wang, Xiaotao Gu, Minlie Huang, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10105)  

**Abstract**: Evaluating mathematical capabilities is critical for assessing the overall performance of large language models (LLMs). However, existing evaluation methods often focus solely on final answers, resulting in highly inaccurate and uninterpretable evaluation outcomes, as well as their failure to assess proof or open-ended problems. To address these issues, we propose a novel mathematical process evaluation agent based on Tree-of-Error, called StepMathAgent. This agent incorporates four internal core operations: logical step segmentation, step scoring, score aggregation and error tree generation, along with four external extension modules: difficulty calibration, simplicity evaluation, completeness validation and format assessment. Furthermore, we introduce StepMathBench, a benchmark comprising 1,000 step-divided process evaluation instances, derived from 200 high-quality math problems grouped by problem type, subject category and difficulty level. Experiments on StepMathBench show that our proposed StepMathAgent outperforms all state-of-the-art methods, demonstrating human-aligned evaluation preferences and broad applicability to various scenarios. Our data and code are available at this https URL. 

---
# OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model 

**Authors**: Bowen Zhang, Pengcheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.10009)  

**Abstract**: Operations Research (OR) has been widely applied in various fields such as resource allocation, production planning, and supply chain management. However, addressing real-world OR problems requires OR experts to perform mathematical modeling and programmers to develop solution algorithms. This traditional method, heavily reliant on experts, is costly and has long development cycles, severely limiting the widespread adoption of OR techniques. Few have considered using Artificial Intelligence (AI) to replace professionals to achieve fully automated solutions for OR problems. We propose OR-LLM-Agent, the first AI agent that enables end-to-end automation for solving real-world OR problems. OR-LLM-Agent leverages the Chain-of-Thought (CoT) reasoning capabilities of Large Language Models (LLMs) to translate natural language problem descriptions into formal mathematical models and automatically generate Gurobi solver code. In OR-LLM-Agent, OR-CodeAgent is designed to automate code execution and repair within a sandbox environment, facilitating the derivation of the final solution. Due to the lack of dedicated benchmark datasets for evaluating the automated solving of OR problems, we construct a benchmark dataset comprising 83 real-world OR problems described in natural language. We conduct comparative experiments with state-of-the-art (SOTA) reasoning LLMs, including GPT-o3-mini, DeepSeek-R1, and Gemini 2.0 Flash Thinking. The OR-LLM-Agent achieved the highest pass rate of 100% and the highest solution accuracy of 85%, demonstrating the feasibility of automated OR problem-solving. Data and code have been publicly available at this https URL. 

---
# A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1 

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10635)  

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at this https URL. 

---
# Compositional Subspace Representation Fine-tuning for Adaptive Large Language Models 

**Authors**: Andy Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.10617)  

**Abstract**: Adapting large language models to multiple tasks can cause cross-skill interference, where improvements for one skill degrade another. While methods such as LoRA impose orthogonality constraints at the weight level, they do not fully address interference in hidden-state representations. We propose Compositional Subspace Representation Fine-tuning (CS-ReFT), a novel representation-based approach that learns multiple orthonormal subspace transformations, each specializing in a distinct skill, and composes them via a lightweight router. By isolating these subspace edits in the hidden state, rather than weight matrices, CS-ReFT prevents cross-task conflicts more effectively. On the AlpacaEval benchmark, applying CS-ReFT to Llama-2-7B achieves a 93.94% win rate, surpassing GPT-3.5 Turbo (86.30%) while requiring only 0.0098% of model parameters. These findings show that specialized representation edits, composed via a simple router, significantly enhance multi-task instruction following with minimal overhead. 

---
# TruthPrInt: Mitigating LVLM Object Hallucination Via Latent Truthful-Guided Pre-Intervention 

**Authors**: Jinhao Duan, Fei Kong, Hao Cheng, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10602)  

**Abstract**: Object Hallucination (OH) has been acknowledged as one of the major trustworthy challenges in Large Vision-Language Models (LVLMs). Recent advancements in Large Language Models (LLMs) indicate that internal states, such as hidden states, encode the "overall truthfulness" of generated responses. However, it remains under-explored how internal states in LVLMs function and whether they could serve as "per-token" hallucination indicators, which is essential for mitigating OH. In this paper, we first conduct an in-depth exploration of LVLM internal states in relation to OH issues and discover that (1) LVLM internal states are high-specificity per-token indicators of hallucination behaviors. Moreover, (2) different LVLMs encode universal patterns of hallucinations in common latent subspaces, indicating that there exist "generic truthful directions" shared by various LVLMs. Based on these discoveries, we propose Truthful-Guided Pre-Intervention (TruthPrInt) that first learns the truthful direction of LVLM decoding and then applies truthful-guided inference-time intervention during LVLM decoding. We further propose ComnHallu to enhance both cross-LVLM and cross-data hallucination detection transferability by constructing and aligning hallucination latent subspaces. We evaluate TruthPrInt in extensive experimental settings, including in-domain and out-of-domain scenarios, over popular LVLMs and OH benchmarks. Experimental results indicate that TruthPrInt significantly outperforms state-of-the-art methods. Codes will be available at this https URL. 

---
# SciVerse: Unveiling the Knowledge Comprehension and Visual Reasoning of LMMs on Multi-modal Scientific Problems 

**Authors**: Ziyu Guo, Ray Zhang, Hao Chen, Jialin Gao, Dongzhi Jiang, Jiaze Wang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10627)  

**Abstract**: The rapid advancement of Large Multi-modal Models (LMMs) has enabled their application in scientific problem-solving, yet their fine-grained capabilities remain under-explored. In this paper, we introduce SciVerse, a multi-modal scientific evaluation benchmark to thoroughly assess LMMs across 5,735 test instances in five distinct versions. We aim to investigate three key dimensions of LMMs: scientific knowledge comprehension, multi-modal content interpretation, and Chain-of-Thought (CoT) reasoning. To unveil whether LMMs possess sufficient scientific expertise, we first transform each problem into three versions containing different levels of knowledge required for solving, i.e., Knowledge-free, -lite, and -rich. Then, to explore how LMMs interpret multi-modal scientific content, we annotate another two versions, i.e., Vision-rich and -only, marking more question information from texts to diagrams. Comparing the results of different versions, SciVerse systematically examines the professional knowledge stock and visual perception skills of LMMs in scientific domains. In addition, to rigorously assess CoT reasoning, we propose a new scientific CoT evaluation strategy, conducting a step-wise assessment on knowledge and logical errors in model outputs. Our extensive evaluation of different LMMs on SciVerse reveals critical limitations in their scientific proficiency and provides new insights into future developments. Project page: this https URL 

---
# Advanced Tool Learning and Selection System (ATLASS): A Closed-Loop Framework Using LLM 

**Authors**: Mohd Ariful Haque, Justin Williams, Sunzida Siddique, Md. Hujaifa Islam, Hasmot Ali, Kishor Datta Gupta, Roy George  

**Link**: [PDF](https://arxiv.org/pdf/2503.10071)  

**Abstract**: The combination of LLM agents with external tools enables models to solve complex tasks beyond their knowledge base. Human-designed tools are inflexible and restricted to solutions within the scope of pre-existing tools created by experts. To address this problem, we propose ATLASS, an advanced tool learning and selection system designed as a closed-loop framework. It enables the LLM to solve problems by dynamically generating external tools on demand. In this framework, agents play a crucial role in orchestrating tool selection, execution, and refinement, ensuring adaptive problem-solving capabilities. The operation of ATLASS follows three phases: The first phase, Understanding Tool Requirements, involves the Agents determining whether tools are required and specifying their functionality; the second phase, Tool Retrieval/Generation, involves the Agents retrieving or generating tools based on their availability; and the third phase, Task Solving, involves combining all the component tools necessary to complete the initial task. The Tool Dataset stores the generated tools, ensuring reusability and minimizing inference cost. Current LLM-based tool generation systems have difficulty creating complex tools that need APIs or external packages. In ATLASS, we solve the problem by automatically setting up the environment, fetching relevant API documentation online, and using a Python interpreter to create a reliable, versatile tool that works in a wider range of situations. OpenAI GPT-4.0 is used as the LLM agent, and safety and ethical concerns are handled through human feedback before executing generated code. By addressing the limitations of predefined toolsets and enhancing adaptability, ATLASS serves as a real-world solution that empowers users with dynamically generated tools for complex problem-solving. 

---
# PiSA: A Self-Augmented Data Engine and Training Strategy for 3D Understanding with Large Models 

**Authors**: Zilu Guo, Hongbin Lin, Zhihao Yuan, Chaoda Zheng, Pengshuo Qiu, Dongzhi Jiang, Renrui Zhang, Chun-Mei Feng, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10529)  

**Abstract**: 3D Multimodal Large Language Models (MLLMs) have recently made substantial advancements. However, their potential remains untapped, primarily due to the limited quantity and suboptimal quality of 3D datasets. Current approaches attempt to transfer knowledge from 2D MLLMs to expand 3D instruction data, but still face modality and domain gaps. To this end, we introduce PiSA-Engine (Point-Self-Augmented-Engine), a new framework for generating instruction point-language datasets enriched with 3D spatial semantics. We observe that existing 3D MLLMs offer a comprehensive understanding of point clouds for annotation, while 2D MLLMs excel at cross-validation by providing complementary information. By integrating holistic 2D and 3D insights from off-the-shelf MLLMs, PiSA-Engine enables a continuous cycle of high-quality data generation. We select PointLLM as the baseline and adopt this co-evolution training framework to develop an enhanced 3D MLLM, termed PointLLM-PiSA. Additionally, we identify limitations in previous 3D benchmarks, which often feature coarse language captions and insufficient category diversity, resulting in inaccurate evaluations. To address this gap, we further introduce PiSA-Bench, a comprehensive 3D benchmark covering six key aspects with detailed and diverse labels. Experimental results demonstrate PointLLM-PiSA's state-of-the-art performance in zero-shot 3D object captioning and generative classification on our PiSA-Bench, achieving significant improvements of 46.45% (+8.33%) and 63.75% (+16.25%), respectively. We will release the code, datasets, and benchmark. 

---
# DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation 

**Authors**: Wenhao Hu, Jinhao Duan, Chunchen Wei, Li Zhang, Yue Zhang, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10452)  

**Abstract**: The rapid advancement of large language models (LLMs) has significantly improved their performance in code generation tasks. However, existing code benchmarks remain static, consisting of fixed datasets with predefined problems. This makes them vulnerable to memorization during training, where LLMs recall specific test cases instead of generalizing to new problems, leading to data contamination and unreliable evaluation results. To address these issues, we introduce DynaCode, a dynamic, complexity-aware benchmark that overcomes the limitations of static datasets. DynaCode evaluates LLMs systematically using a complexity-aware metric, incorporating both code complexity and call-graph structures. DynaCode achieves large-scale diversity, generating up to 189 million unique nested code problems across four distinct levels of code complexity, referred to as units, and 16 types of call graphs. Results on 12 latest LLMs show an average performance drop of 16.8% to 45.7% compared to MBPP+, a static code generation benchmark, with performance progressively decreasing as complexity increases. This demonstrates DynaCode's ability to effectively differentiate LLMs. Additionally, by leveraging call graphs, we gain insights into LLM behavior, particularly their preference for handling subfunction interactions within nested code. 

---
# CINEMA: Coherent Multi-Subject Video Generation via MLLM-Based Guidance 

**Authors**: Yufan Deng, Xun Guo, Yizhi Wang, Jacob Zhiyuan Fang, Angtian Wang, Shenghai Yuan, Yiding Yang, Bo Liu, Haibin Huang, Chongyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10391)  

**Abstract**: Video generation has witnessed remarkable progress with the advent of deep generative models, particularly diffusion models. While existing methods excel in generating high-quality videos from text prompts or single images, personalized multi-subject video generation remains a largely unexplored challenge. This task involves synthesizing videos that incorporate multiple distinct subjects, each defined by separate reference images, while ensuring temporal and spatial consistency. Current approaches primarily rely on mapping subject images to keywords in text prompts, which introduces ambiguity and limits their ability to model subject relationships effectively. In this paper, we propose CINEMA, a novel framework for coherent multi-subject video generation by leveraging Multimodal Large Language Model (MLLM). Our approach eliminates the need for explicit correspondences between subject images and text entities, mitigating ambiguity and reducing annotation effort. By leveraging MLLM to interpret subject relationships, our method facilitates scalability, enabling the use of large and diverse datasets for training. Furthermore, our framework can be conditioned on varying numbers of subjects, offering greater flexibility in personalized content creation. Through extensive evaluations, we demonstrate that our approach significantly improves subject consistency, and overall video coherence, paving the way for advanced applications in storytelling, interactive media, and personalized video generation. 

---
# G-Boost: Boosting Private SLMs with General LLMs 

**Authors**: Yijiang Fan, Yuren Mao, Longbin Lai, Ying Zhang, Zhengping Qian, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10367)  

**Abstract**: Due to the limited computational resources, most Large Language Models (LLMs) developers can only fine-tune Small Language Models (SLMs) on their own data. These private SLMs typically have limited effectiveness. To boost the performance of private SLMs, this paper proposes to ask general LLMs for help. The general LLMs can be APIs or larger LLMs whose inference cost the developers can afford. Specifically, we propose the G-Boost framework where a private SLM adaptively performs collaborative inference with a general LLM under the guide of process reward. Experiments demonstrate that our framework can significantly boost the performance of private SLMs. 

---
# KV-Distill: Nearly Lossless Learnable Context Compression for LLMs 

**Authors**: Vivek Chari, Guanghui Qin, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2503.10337)  

**Abstract**: Sequence-to-sequence tasks often benefit from long contexts, but the quadratic complexity of self-attention in standard Transformers renders this non-trivial. During generation, temporary representations -stored in the so-called KV cache-account for a large portion of GPU memory usage and scale linearly with context length. We introduce KV-Distill, a Transformer compression framework that distills long context KV caches into significantly shorter representations in a question-independent fashion. KV-Distill can be trained as a parameter-efficient adaptor for pretrained models, and enables the compression of arbitrary spans of a context while preserving pre-trained model capabilities. We treat a compressed-uncompressed cache as a student-teacher pairing and apply a KL-type divergence to match the generated outputs. KV-Distill outperforms other compression techniques in worst-case extractive tasks and approaches uncompressed performance in long context question answering and summarization, and it can be fine-tuned on domain-specific contexts to reduce lengths by up to 99% while preserving downstream performance. We demonstrate the generalizability of KV-Distill across various model sizes and architectures. 

---
# Efficient Federated Fine-Tuning of Large Language Models with Layer Dropout 

**Authors**: Shilong Wang, Jianchun Liu, Hongli Xu, Jiaming Yan, Xianjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10217)  

**Abstract**: Fine-tuning plays a crucial role in enabling pre-trained LLMs to evolve from general language comprehension to task-specific expertise. To preserve user data privacy, federated fine-tuning is often employed and has emerged as the de facto paradigm. However, federated fine-tuning is prohibitively inefficient due to the tension between LLM complexity and the resource constraint of end devices, incurring unaffordable fine-tuning overhead. Existing literature primarily utilizes parameter-efficient fine-tuning techniques to mitigate communication costs, yet computational and memory burdens continue to pose significant challenges for developers. This work proposes DropPEFT, an innovative federated PEFT framework that employs a novel stochastic transformer layer dropout method, enabling devices to deactivate a considerable fraction of LLMs layers during training, thereby eliminating the associated computational load and memory footprint. In DropPEFT, a key challenge is the proper configuration of dropout ratios for layers, as overhead and training performance are highly sensitive to this setting. To address this challenge, we adaptively assign optimal dropout-ratio configurations to devices through an exploration-exploitation strategy, achieving efficient and effective fine-tuning. Extensive experiments show that DropPEFT can achieve a 1.3-6.3\times speedup in model convergence and a 40%-67% reduction in memory footprint compared to state-of-the-art methods. 

---
# LLMs in Disease Diagnosis: A Comparative Study of DeepSeek-R1 and O3 Mini Across Chronic Health Conditions 

**Authors**: Gaurav Kumar Gupta, Pranal Pande  

**Link**: [PDF](https://arxiv.org/pdf/2503.10486)  

**Abstract**: Large Language Models (LLMs) are revolutionizing medical diagnostics by enhancing both disease classification and clinical decision-making. In this study, we evaluate the performance of two LLM- based diagnostic tools, DeepSeek R1 and O3 Mini, using a structured dataset of symptoms and diagnoses. We assessed their predictive accuracy at both the disease and category levels, as well as the reliability of their confidence scores. DeepSeek R1 achieved a disease-level accuracy of 76% and an overall accuracy of 82%, outperforming O3 Mini, which attained 72% and 75% respectively. Notably, DeepSeek R1 demonstrated exceptional performance in Mental Health, Neurological Disorders, and Oncology, where it reached 100% accuracy, while O3 Mini excelled in Autoimmune Disease classification with 100% accuracy. Both models, however, struggled with Respiratory Disease classification, recording accuracies of only 40% for DeepSeek R1 and 20% for O3 Mini. Additionally, the analysis of confidence scores revealed that DeepSeek R1 provided high-confidence predictions in 92% of cases, compared to 68% for O3 Mini. Ethical considerations regarding bias, model interpretability, and data privacy are also discussed to ensure the responsible integration of LLMs into clinical practice. Overall, our findings offer valuable insights into the strengths and limitations of LLM-based diagnostic systems and provide a roadmap for future enhancements in AI-driven healthcare. 

---
# Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding 

**Authors**: Jinze Li, Yixing Xu, Haiduo Huang, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2503.10135)  

**Abstract**: Speculative decoding (SPD) aims to accelerate the auto-regressive token generation process of a target Large Language Model (LLM). Some approaches employ a draft model with multiple heads to predict a sequence of future tokens, where each head handles a token in the sequence. The target LLM verifies the predicted sequence and accepts aligned tokens, enabling efficient multi-token generation. However, existing methods assume that all tokens within a sequence are equally important, employing identical head structures and relying on a single-generation paradigm, either serial or parallel. To this end, we theoretically demonstrate that initial tokens in the draft sequence are more important than later ones. Building on this insight, we propose Gumiho, a hybrid model combining serial and parallel heads. Specifically, given the critical importance of early tokens, we employ a sophisticated Transformer architecture for the early draft heads in a serial configuration to improve accuracy. For later tokens, we utilize multiple lightweight MLP heads operating in parallel to enhance efficiency. By allocating more advanced model structures and longer running times to the early heads, Gumiho achieves improved overall performance. The experimental results demonstrate that our method outperforms existing approaches, fully validating its effectiveness. 

---
# Retrieval-Augmented Generation with Hierarchical Knowledge 

**Authors**: Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.10150)  

**Abstract**: Graph-based Retrieval-Augmented Generation (RAG) methods have significantly enhanced the performance of large language models (LLMs) in domain-specific tasks. However, existing RAG methods do not adequately utilize the naturally inherent hierarchical knowledge in human cognition, which limits the capabilities of RAG systems. In this paper, we introduce a new RAG approach, called HiRAG, which utilizes hierarchical knowledge to enhance the semantic understanding and structure capturing capabilities of RAG systems in the indexing and retrieval processes. Our extensive experiments demonstrate that HiRAG achieves significant performance improvements over the state-of-the-art baseline methods. The code of our proposed method is available at \href{this https URL}{this https URL}. 

---
# Cognitive-Mental-LLM: Leveraging Reasoning in Large Language Models for Mental Health Prediction via Online Text 

**Authors**: Avinash Patil, Amardeep Kour Gedhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10095)  

**Abstract**: Large Language Models (LLMs) have demonstrated potential in predicting mental health outcomes from online text, yet traditional classification methods often lack interpretability and robustness. This study evaluates structured reasoning techniques-Chain-of-Thought (CoT), Self-Consistency (SC-CoT), and Tree-of-Thought (ToT)-to improve classification accuracy across multiple mental health datasets sourced from Reddit. We analyze reasoning-driven prompting strategies, including Zero-shot CoT and Few-shot CoT, using key performance metrics such as Balanced Accuracy, F1 score, and Sensitivity/Specificity. Our findings indicate that reasoning-enhanced techniques improve classification performance over direct prediction, particularly in complex cases. Compared to baselines such as Zero Shot non-CoT Prompting, and fine-tuned pre-trained transformers such as BERT and Mental-RoBerta, and fine-tuned Open Source LLMs such as Mental Alpaca and Mental-Flan-T5, reasoning-driven LLMs yield notable gains on datasets like Dreaddit (+0.52\% over M-LLM, +0.82\% over BERT) and SDCNL (+4.67\% over M-LLM, +2.17\% over BERT). However, performance declines in Depression Severity, and CSSRS predictions suggest dataset-specific limitations, likely due to our using a more extensive test set. Among prompting strategies, Few-shot CoT consistently outperforms others, reinforcing the effectiveness of reasoning-driven LLMs. Nonetheless, dataset variability highlights challenges in model reliability and interpretability. This study provides a comprehensive benchmark of reasoning-based LLM techniques for mental health text classification. It offers insights into their potential for scalable clinical applications while identifying key challenges for future improvements. 

---
# ImageScope: Unifying Language-Guided Image Retrieval via Large Multimodal Model Collective Reasoning 

**Authors**: Pengfei Luo, Jingbo Zhou, Tong Xu, Yuan Xia, Linli Xu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10166)  

**Abstract**: With the proliferation of images in online content, language-guided image retrieval (LGIR) has emerged as a research hotspot over the past decade, encompassing a variety of subtasks with diverse input forms. While the development of large multimodal models (LMMs) has significantly facilitated these tasks, existing approaches often address them in isolation, requiring the construction of separate systems for each task. This not only increases system complexity and maintenance costs, but also exacerbates challenges stemming from language ambiguity and complex image content, making it difficult for retrieval systems to provide accurate and reliable results. To this end, we propose ImageScope, a training-free, three-stage framework that leverages collective reasoning to unify LGIR tasks. The key insight behind the unification lies in the compositional nature of language, which transforms diverse LGIR tasks into a generalized text-to-image retrieval process, along with the reasoning of LMMs serving as a universal verification to refine the results. To be specific, in the first stage, we improve the robustness of the framework by synthesizing search intents across varying levels of semantic granularity using chain-of-thought (CoT) reasoning. In the second and third stages, we then reflect on retrieval results by verifying predicate propositions locally, and performing pairwise evaluations globally. Experiments conducted on six LGIR datasets demonstrate that ImageScope outperforms competitive baselines. Comprehensive evaluations and ablation studies further confirm the effectiveness of our design. 

---
# Generative AI for Named Entity Recognition in Low-Resource Language Nepali 

**Authors**: Sameer Neupane, Jeevan Chapagain, Nobal B. Niraula, Diwa Koirala  

**Link**: [PDF](https://arxiv.org/pdf/2503.09822)  

**Abstract**: Generative Artificial Intelligence (GenAI), particularly Large Language Models (LLMs), has significantly advanced Natural Language Processing (NLP) tasks, such as Named Entity Recognition (NER), which involves identifying entities like person, location, and organization names in text. LLMs are especially promising for low-resource languages due to their ability to learn from limited data. However, the performance of GenAI models for Nepali, a low-resource language, has not been thoroughly evaluated. This paper investigates the application of state-of-the-art LLMs for Nepali NER, conducting experiments with various prompting techniques to assess their effectiveness. Our results provide insights into the challenges and opportunities of using LLMs for NER in low-resource settings and offer valuable contributions to the advancement of NLP research in languages like Nepali. 

---
# Exploiting Edited Large Language Models as General Scientific Optimizers 

**Authors**: Qitan Lv, Tianyu Liu, Hong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09620)  

**Abstract**: Large language models (LLMs) have been widely adopted in mathematical optimization in scientific scenarios for their extensive knowledge and advanced reasoning capabilities. Existing methods mainly focus on utilizing LLMs to solve optimization problems in a prompt-based manner, which takes observational feedback as additional textual descriptions. However, due to LLM's \textbf{high sensitivity to the prompts} and \textbf{tendency to get lost in lengthy prompts}, these methods struggle to effectively utilize the {observational} feedback from each optimization step, which severely hinders the applications for real-world scenarios. To address these challenges, we propose a conceptually simple and general {bi-level} optimization method, namely \textbf{G}eneral \textbf{S}cientific \textbf{O}ptimizers (GSO). Specifically, GSO first utilizes inner-level simulators as experimental platforms to evaluate the current solution and provide observational feedback. Then, LLMs serve as knowledgeable and versatile scientists, generating new solutions by refining potential errors from the feedback as the outer-level optimization. Finally, simulations together with the expert knowledge in LLMs are jointly updated with bi-level interactions via model editing. Extensive experiments show that GSO consistently outperforms existing state-of-the-art methods using \textit{six} different LLM backbones on \textit{seven} different tasks, demonstrating the effectiveness and a wide range of applications. 

---
# Can A Society of Generative Agents Simulate Human Behavior and Inform Public Health Policy? A Case Study on Vaccine Hesitancy 

**Authors**: Abe Bohan Hou, Hongru Du, Yichen Wang, Jingyu Zhang, Zixiao Wang, Paul Pu Liang, Daniel Khashabi, Lauren Gardner, Tianxing He  

**Link**: [PDF](https://arxiv.org/pdf/2503.09639)  

**Abstract**: Can we simulate a sandbox society with generative agents to model human behavior, thereby reducing the over-reliance on real human trials for assessing public policies? In this work, we investigate the feasibility of simulating health-related decision-making, using vaccine hesitancy, defined as the delay in acceptance or refusal of vaccines despite the availability of vaccination services (MacDonald, 2015), as a case study. To this end, we introduce the VacSim framework with 100 generative agents powered by Large Language Models (LLMs). VacSim simulates vaccine policy outcomes with the following steps: 1) instantiate a population of agents with demographics based on census data; 2) connect the agents via a social network and model vaccine attitudes as a function of social dynamics and disease-related information; 3) design and evaluate various public health interventions aimed at mitigating vaccine hesitancy. To align with real-world results, we also introduce simulation warmup and attitude modulation to adjust agents' attitudes. We propose a series of evaluations to assess the reliability of various LLM simulations. Experiments indicate that models like Llama and Qwen can simulate aspects of human behavior but also highlight real-world alignment challenges, such as inconsistent responses with demographic profiles. This early exploration of LLM-driven simulations is not meant to serve as definitive policy guidance; instead, it serves as a call for action to examine social simulation for policy development. 

---
# Conversational Gold: Evaluating Personalized Conversational Search System using Gold Nuggets 

**Authors**: Zahra Abbasiantaeb, Simon Lupart, Leif Azzopardi, Jeffery Dalton, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09902)  

**Abstract**: The rise of personalized conversational search systems has been driven by advancements in Large Language Models (LLMs), enabling these systems to retrieve and generate answers for complex information needs. However, the automatic evaluation of responses generated by Retrieval Augmented Generation (RAG) systems remains an understudied challenge. In this paper, we introduce a new resource for assessing the retrieval effectiveness and relevance of response generated by RAG systems, using a nugget-based evaluation framework. Built upon the foundation of TREC iKAT 2023, our dataset extends to the TREC iKAT 2024 collection, which includes 17 conversations and 20,575 relevance passage assessments, together with 2,279 extracted gold nuggets, and 62 manually written gold answers from NIST assessors. While maintaining the core structure of its predecessor, this new collection enables a deeper exploration of generation tasks in conversational settings. Key improvements in iKAT 2024 include: (1) ``gold nuggets'' -- concise, essential pieces of information extracted from relevant passages of the collection -- which serve as a foundation for automatic response evaluation; (2) manually written answers to provide a gold standard for response evaluation; (3) unanswerable questions to evaluate model hallucination; (4) expanded user personas, providing richer contextual grounding; and (5) a transition from Personal Text Knowledge Base (PTKB) ranking to PTKB classification and selection. Built on this resource, we provide a framework for long-form answer generation evaluation, involving nuggets extraction and nuggets matching, linked to retrieval. This establishes a solid resource for advancing research in personalized conversational search and long-form answer generation. Our resources are publicly available at this https URL. 

---
# Improving the Reusability of Conversational Search Test Collections 

**Authors**: Zahra Abbasiantaeb, Chuan Meng, Leif Azzopardi, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2503.09899)  

**Abstract**: Incomplete relevance judgments limit the reusability of test collections. When new systems are compared to previous systems that contributed to the pool, they often face a disadvantage. This is due to pockets of unjudged documents (called holes) in the test collection that the new systems return. The very nature of Conversational Search (CS) means that these holes are potentially larger and more problematic when evaluating systems. In this paper, we aim to extend CS test collections by employing Large Language Models (LLMs) to fill holes by leveraging existing judgments. We explore this problem using TREC iKAT 23 and TREC CAsT 22 collections, where information needs are highly dynamic and the responses are much more varied, leaving bigger holes to fill. Our experiments reveal that CS collections show a trend towards less reusability in deeper turns. Also, fine-tuning the Llama 3.1 model leads to high agreement with human assessors, while few-shot prompting the ChatGPT results in low agreement with humans. Consequently, filling the holes of a new system using ChatGPT leads to a higher change in the location of the new system. While regenerating the assessment pool with few-shot prompting the ChatGPT model and using it for evaluation achieves a high rank correlation with human-assessed pools. We show that filling the holes using few-shot training the Llama 3.1 model enables a fairer comparison between the new system and the systems contributed to the pool. Our hole-filling model based on few-shot training of the Llama 3.1 model can improve the reusability of test collections. 

---
# From TOWER to SPIRE: Adding the Speech Modality to a Text-Only LLM 

**Authors**: Kshitij Ambilduke, Ben Peters, Sonal Sannigrahi, Anil Keshwani, Tsz Kin Lam, Bruno Martins, Marcely Zanon Boito, André F.T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2503.10620)  

**Abstract**: Large language models (LLMs) have shown remarkable performance and generalization capabilities across multiple languages and tasks, making them very attractive targets for multi-modality integration (e.g., images or speech). In this work, we extend an existing LLM to the speech modality via speech discretization and continued pre-training. In particular, we are interested in multilingual LLMs, such as TOWER, as their pre-training setting allows us to treat discretized speech input as an additional translation language. The resulting open-source model, SPIRE, is able to transcribe and translate English speech input while maintaining TOWER's original performance on translation-related tasks, showcasing that discretized speech input integration as an additional language is feasible during LLM adaptation. We make our code and models available to the community. 

---
# Source-primed Multi-turn Conversation Helps Large Language Models Translate Documents 

**Authors**: Hanxu Hu, Jannis Vamvas, Rico Sennrich  

**Link**: [PDF](https://arxiv.org/pdf/2503.10494)  

**Abstract**: LLMs have paved the way for truly simple document-level machine translation, but challenges such as omission errors remain. In this paper, we study a simple method for handling document-level machine translation, by leveraging previous contexts in a multi-turn conversational manner. Specifically, by decomposing documents into segments and iteratively translating them while maintaining previous turns, this method ensures coherent translations without additional training, and can fully re-use the KV cache of previous turns thus minimizing computational overhead. We further propose a `source-primed' method that first provides the whole source document before multi-turn translation. We empirically show this multi-turn method outperforms both translating entire documents in a single turn and translating each segment independently according to multiple automatic metrics in representative LLMs, establishing a strong baseline for document-level translation using LLMs. 

---
# Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set 

**Authors**: Florian Eichin, Yang Janet Liu, Barbara Plank, Michael A. Hedderich  

**Link**: [PDF](https://arxiv.org/pdf/2503.10515)  

**Abstract**: Discourse understanding is essential for many NLP tasks, yet most existing work remains constrained by framework-dependent discourse representations. This work investigates whether large language models (LLMs) capture discourse knowledge that generalizes across languages and frameworks. We address this question along two dimensions: (1) developing a unified discourse relation label set to facilitate cross-lingual and cross-framework discourse analysis, and (2) probing LLMs to assess whether they encode generalizable discourse abstractions. Using multilingual discourse relation classification as a testbed, we examine a comprehensive set of 23 LLMs of varying sizes and multilingual capabilities. Our results show that LLMs, especially those with multilingual training corpora, can generalize discourse information across languages and frameworks. Further layer-wise analyses reveal that language generalization at the discourse level is most salient in the intermediate layers. Lastly, our error analysis provides an account of challenging relation classes. 

---
# Adaptive Inner Speech-Text Alignment for LLM-based Speech Translation 

**Authors**: Henglyu Liu, Andong Chen, Kehai Chen, Xuefeng Bai, Meizhi Zhong, Yuan Qiu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10211)  

**Abstract**: Recent advancement of large language models (LLMs) has led to significant breakthroughs across various tasks, laying the foundation for the development of LLM-based speech translation systems. Existing methods primarily focus on aligning inputs and outputs across modalities while overlooking deeper semantic alignment within model representations. To address this limitation, we propose an Adaptive Inner Speech-Text Alignment (AI-STA) method to bridge the modality gap by explicitly aligning speech and text representations at selected layers within LLMs. To achieve this, we leverage the optimal transport (OT) theory to quantify fine-grained representation discrepancies between speech and text. Furthermore, we utilize the cross-modal retrieval technique to identify the layers that are best suited for alignment and perform joint training on these layers. Experimental results on speech translation (ST) tasks demonstrate that AI-STA significantly improves the translation performance of large speech-text models (LSMs), outperforming previous state-of-the-art approaches. Our findings highlight the importance of inner-layer speech-text alignment in LLMs and provide new insights into enhancing cross-modal learning. 

---
# New Trends for Modern Machine Translation with Large Reasoning Models 

**Authors**: Sinuo Liu, Chenyang Lyu, Minghao Wu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10351)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs), particularly those leveraging Chain-of-Thought reasoning (CoT), have opened brand new possibility for Machine Translation (MT). This position paper argues that LRMs substantially transformed traditional neural MT as well as LLMs-based MT paradigms by reframing translation as a dynamic reasoning task that requires contextual, cultural, and linguistic understanding and reasoning. We identify three foundational shifts: 1) contextual coherence, where LRMs resolve ambiguities and preserve discourse structure through explicit reasoning over cross-sentence and complex context or even lack of context; 2) cultural intentionality, enabling models to adapt outputs by inferring speaker intent, audience expectations, and socio-linguistic norms; 3) self-reflection, LRMs can perform self-reflection during the inference time to correct the potential errors in translation especially extremely noisy cases, showing better robustness compared to simply mapping X->Y translation. We explore various scenarios in translation including stylized translation, document-level translation and multimodal translation by showcasing empirical examples that demonstrate the superiority of LRMs in translation. We also identify several interesting phenomenons for LRMs for MT including auto-pivot translation as well as the critical challenges such as over-localisation in translation and inference efficiency. In conclusion, we think that LRMs redefine translation systems not merely as text converters but as multilingual cognitive agents capable of reasoning about meaning beyond the text. This paradigm shift reminds us to think of problems in translation beyond traditional translation scenarios in a much broader context with LRMs - what we can achieve on top of it. 

---
# "Well, Keep Thinking": Enhancing LLM Reasoning with Adaptive Injection Decoding 

**Authors**: Hyunbin Jin, Je Won Yeom, Seunghyun Bae, Taesup Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.10167)  

**Abstract**: Large language models (LLMs) exhibit strong reasoning abilities, often attributed to few-shot or zero-shot chain-of-thought (CoT) prompting. While effective, these methods require labor-intensive prompt engineering, raising the question of whether reasoning can be induced without reliance on explicit prompts. In this work, we unlock the reasoning capabilities of LLMs without explicit prompting. Inspired by zero-shot CoT and CoT-decoding, we propose a novel decoding strategy that systematically nudges LLMs to continue reasoning, thereby preventing immature reasoning processes. Specifically, we monitor the model's generation and inject a designated phrase whenever it is likely to conclude its response prematurely, before completing the reasoning process. Our experimental evaluations on diverse reasoning benchmarks demonstrate that our proposed strategy substantially improves LLM reasoning capabilities, highlighting the potential of decoding-based interventions as an alternative to traditional prompting techniques. 

---
# Representation-based Reward Modeling for Efficient Safety Alignment of Large Language Model 

**Authors**: Qiyuan Deng, Xuefeng Bai, Kehai Chen, Yaowei Wang, Liqiang Nie, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10093)  

**Abstract**: Reinforcement Learning (RL) algorithms for safety alignment of Large Language Models (LLMs), such as Direct Preference Optimization (DPO), encounter the challenge of distribution shift. Current approaches typically address this issue through online sampling from the target policy, which requires significant computational resources. In this paper, we hypothesize that during off-policy training, while the ranking order of output generated by policy changes, their overall distribution remains relatively stable. This stability allows the transformation of the sampling process from the target policy into a re-ranking of preference data. Building on this hypothesis, We propose a new framework that leverages the model's intrinsic safety judgment capability to extract reward signals, which are then used to calculate label confidence for preferences reordering. Extensive experimental results and theoretical analysis demonstrate that the proposed method effectively addresses the distribution shift issue, remarkably enhancing the safety performance while reducing about 300x computational overheads. 

---
# Attention Reveals More Than Tokens: Training-Free Long-Context Reasoning with Attention-guided Retrieval 

**Authors**: Yuwei Zhang, Jayanth Srinivasa, Gaowen Liu, Jingbo Shang  

**Link**: [PDF](https://arxiv.org/pdf/2503.09819)  

**Abstract**: Large Language Models (LLMs) often exhibit substantially shorter effective context lengths than their claimed capacities, especially when handling complex reasoning tasks that require integrating information from multiple parts of a long context and performing multi-step reasoning. Although Chain-of-Thought (CoT) prompting has shown promise in reducing task complexity, our empirical analysis reveals that it does not fully resolve this limitation. Through controlled experiments, we identify poor recall of implicit facts as the primary cause of failure, which significantly hampers reasoning performance. Interestingly, we observe that the internal attention weights from the generated CoT tokens can effectively ground implicit facts, even when these facts are not explicitly recalled. Building on this insight, we propose a novel training-free algorithm, Attrieval, which leverages attention weights to retrieve relevant facts from the long context and incorporates them into the reasoning process. Additionally, we find that selecting context tokens from CoT tokens further improves performance. Our results demonstrate that Attrieval enhances long-context reasoning capability notably on both synthetic and real-world QA datasets with various models. 

---
# Why Does Your CoT Prompt (Not) Work? Theoretical Analysis of Prompt Space Complexity, its Interaction with Answer Space During CoT Reasoning with LLMs: A Recurrent Perspective 

**Authors**: Xiang Zhang, Juntai Cao, Jiaqi Wei, Chenyu You, Dujian Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.10084)  

**Abstract**: Despite the remarkable successes of Large Language Models (LLMs), their fundamental Transformer architecture possesses inherent theoretical limitations that restrict their capability to handle reasoning tasks with increasing computational complexity. Chain-of-Thought (CoT) prompting has emerged as a practical solution, supported by several theoretical studies. However, current CoT-based methods (including ToT, GoT, etc.) generally adopt a "one-prompt-fits-all" strategy, using fixed templates (e.g., "think step by step") across diverse reasoning tasks. This method forces models to navigate an extremely complex prompt space to identify effective reasoning paths. The current prompt designing research are also heavily relying on trial-and-error rather than theoretically informed guidance. In this paper, we provide a rigorous theoretical analysis of the complexity and interplay between two crucial spaces: the prompt space (the space of potential prompt structures) and the answer space (the space of reasoning solutions generated by LLMs) in CoT reasoning. We demonstrate how reliance on a single universal prompt (e.g. think step by step) can negatively impact the theoretical computability of LLMs, illustrating that prompt complexity directly influences the structure and effectiveness of the navigation in answer space. Our analysis highlights that sometimes human supervision is critical for efficiently navigating the prompt space. We theoretically and empirically show that task-specific prompting significantly outperforms unsupervised prompt generation, emphasizing the necessity of thoughtful human guidance in CoT prompting. 

---
# Probabilistic Reasoning with LLMs for k-anonymity Estimation 

**Authors**: Jonathan Zheng, Sauvik Das, Alan Ritter, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09674)  

**Abstract**: Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a novel numerical reasoning task under uncertainty, focusing on estimating the k-anonymity of user-generated documents containing privacy-sensitive information. We propose BRANCH, which uses LLMs to factorize a joint probability distribution to estimate the k-value-the size of the population matching the given information-by modeling individual pieces of textual information as random variables. The probability of each factor occurring within a population is estimated using standalone LLMs or retrieval-augmented generation systems, and these probabilities are combined into a final k-value. Our experiments show that this method successfully estimates the correct k-value 67% of the time, an 11% increase compared to GPT-4o chain-of-thought reasoning. Additionally, we leverage LLM uncertainty to develop prediction intervals for k-anonymity, which include the correct value in nearly 92% of cases. 

---
# Review GIDE -- Restaurant Review Gastrointestinal Illness Detection and Extraction with Large Language Models 

**Authors**: Timothy Laurence, Joshua Harris, Leo Loman, Amy Douglas, Yung-Wai Chan, Luke Hounsome, Lesley Larkin, Michael Borowitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.09743)  

**Abstract**: Foodborne gastrointestinal (GI) illness is a common cause of ill health in the UK. However, many cases do not interact with the healthcare system, posing significant challenges for traditional surveillance methods. The growth of publicly available online restaurant reviews and advancements in large language models (LLMs) present potential opportunities to extend disease surveillance by identifying public reports of GI illness. In this study, we introduce a novel annotation schema, developed with experts in GI illness, applied to the Yelp Open Dataset of reviews. Our annotations extend beyond binary disease detection, to include detailed extraction of information on symptoms and foods. We evaluate the performance of open-weight LLMs across these three tasks: GI illness detection, symptom extraction, and food extraction. We compare this performance to RoBERTa-based classification models fine-tuned specifically for these tasks. Our results show that using prompt-based approaches, LLMs achieve micro-F1 scores of over 90% for all three of our tasks. Using prompting alone, we achieve micro-F1 scores that exceed those of smaller fine-tuned models. We further demonstrate the robustness of LLMs in GI illness detection across three bias-focused experiments. Our results suggest that publicly available review text and LLMs offer substantial potential for public health surveillance of GI illness by enabling highly effective extraction of key information. While LLMs appear to exhibit minimal bias in processing, the inherent limitations of restaurant review data highlight the need for cautious interpretation of results. 

---
# What's In Your Field? Mapping Scientific Research with Knowledge Graphs and Large Language Models 

**Authors**: Abhipsha Das, Nicholas Lourie, Siavash Golkar, Mariel Pettee  

**Link**: [PDF](https://arxiv.org/pdf/2503.09894)  

**Abstract**: The scientific literature's exponential growth makes it increasingly challenging to navigate and synthesize knowledge across disciplines. Large language models (LLMs) are powerful tools for understanding scientific text, but they fail to capture detailed relationships across large bodies of work. Unstructured approaches, like retrieval augmented generation, can sift through such corpora to recall relevant facts; however, when millions of facts influence the answer, unstructured approaches become cost prohibitive. Structured representations offer a natural complement -- enabling systematic analysis across the whole corpus. Recent work enhances LLMs with unstructured or semistructured representations of scientific concepts; to complement this, we try extracting structured representations using LLMs. By combining LLMs' semantic understanding with a schema of scientific concepts, we prototype a system that answers precise questions about the literature as a whole. Our schema applies across scientific fields and we extract concepts from it using only 20 manually annotated abstracts. To demonstrate the system, we extract concepts from 30,000 papers on arXiv spanning astrophysics, fluid dynamics, and evolutionary biology. The resulting database highlights emerging trends and, by visualizing the knowledge graph, offers new ways to explore the ever-growing landscape of scientific knowledge. Demo: abby101/surveyor-0 on HF Spaces. Code: this https URL. 

---
# BeamLLM: Vision-Empowered mmWave Beam Prediction with Large Language Models 

**Authors**: Can Zheng, Jiguang He, Guofa Cai, Zitong Yu, Chung G. Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10432)  

**Abstract**: In this paper, we propose BeamLLM, a vision-aided millimeter-wave (mmWave) beam prediction framework leveraging large language models (LLMs) to address the challenges of high training overhead and latency in mmWave communication systems. By combining computer vision (CV) with LLMs' cross-modal reasoning capabilities, the framework extracts user equipment (UE) positional features from RGB images and aligns visual-temporal features with LLMs' semantic space through reprogramming techniques. Evaluated on a realistic vehicle-to-infrastructure (V2I) scenario, the proposed method achieves 61.01% top-1 accuracy and 97.39% top-3 accuracy in standard prediction tasks, significantly outperforming traditional deep learning models. In few-shot prediction scenarios, the performance degradation is limited to 12.56% (top-1) and 5.55% (top-3) from time sample 1 to 10, demonstrating superior prediction capability. 

---
# VisualPRM: An Effective Process Reward Model for Multimodal Reasoning 

**Authors**: Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10291)  

**Abstract**: We introduce VisualPRM, an advanced multimodal Process Reward Model (PRM) with 8B parameters, which improves the reasoning abilities of existing Multimodal Large Language Models (MLLMs) across different model scales and families with Best-of-N (BoN) evaluation strategies. Specifically, our model improves the reasoning performance of three types of MLLMs and four different model scales. Even when applied to the highly capable InternVL2.5-78B, it achieves a 5.9-point improvement across seven multimodal reasoning benchmarks. Experimental results show that our model exhibits superior performance compared to Outcome Reward Models and Self-Consistency during BoN evaluation. To facilitate the training of multimodal PRMs, we construct a multimodal process supervision dataset VisualPRM400K using an automated data pipeline. For the evaluation of multimodal PRMs, we propose VisualProcessBench, a benchmark with human-annotated step-wise correctness labels, to measure the abilities of PRMs to detect erroneous steps in multimodal reasoning tasks. We hope that our work can inspire more future research and contribute to the development of MLLMs. Our model, data, and benchmark are released in this https URL. 

---
# MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation 

**Authors**: Weihao Xuan, Rui Yang, Heli Qi, Qingcheng Zeng, Yunze Xiao, Yun Xing, Junjue Wang, Huitao Li, Xin Li, Kunyu Yu, Nan Liu, Qingyu Chen, Douglas Teodoro, Edison Marrese-Taylor, Shijian Lu, Yusuke Iwasawa, Yutaka Matsuo, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10497)  

**Abstract**: Traditional benchmarks struggle to evaluate increasingly sophisticated language models in multilingual and culturally diverse contexts. To address this gap, we introduce MMLU-ProX, a comprehensive multilingual benchmark covering 13 typologically diverse languages with approximately 11,829 questions per language. Building on the challenging reasoning-focused design of MMLU-Pro, our framework employs a semi-automatic translation process: translations generated by state-of-the-art large language models (LLMs) are rigorously evaluated by expert annotators to ensure conceptual accuracy, terminological consistency, and cultural relevance. We comprehensively evaluate 25 state-of-the-art LLMs using 5-shot chain-of-thought (CoT) and zero-shot prompting strategies, analyzing their performance across linguistic and cultural boundaries. Our experiments reveal consistent performance degradation from high-resource languages to lower-resource ones, with the best models achieving over 70% accuracy on English but dropping to around 40% for languages like Swahili, highlighting persistent gaps in multilingual capabilities despite recent advances. MMLU-ProX is an ongoing project; we are expanding our benchmark by incorporating additional languages and evaluating more language models to provide a more comprehensive assessment of multilingual capabilities. 

---
# ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content 

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2503.09964)  

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies. 

---
# Understanding the Logical Capabilities of Large Language Models via Out-of-Context Representation Learning 

**Authors**: Jonathan Shaki, Emanuele La Malfa, Michael Wooldridge, Sarit Kraus  

**Link**: [PDF](https://arxiv.org/pdf/2503.10408)  

**Abstract**: We study the capabilities of Large Language Models (LLM) on binary relations, a ubiquitous concept in math employed in most reasoning, math and logic benchmarks. This work focuses on equality, inequality, and inclusion, along with the properties they satisfy, such as ir/reflexivity, a/symmetry, transitivity, and logical complexity (e.g., number of reasoning ``hops''). We propose an alternative to in-context learning that trains only the representations of newly introduced tokens, namely out-of-context representation learning. This method mitigates linguistic biases already present in a model and, differently from in-context learning, does not rely on external information or illustrations. We argue out-of-context representation learning as a better alternative to in-context learning and fine-tuning to evaluate the capabilities of LLMs on logic tasks that are the building blocks of more complex reasoning benchmarks. 

---
# PluralLLM: Pluralistic Alignment in LLMs via Federated Learning 

**Authors**: Mahmoud Srewa, Tianyu Zhao, Salma Elmalaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.09925)  

**Abstract**: Ensuring Large Language Models (LLMs) align with diverse human preferences while preserving privacy and fairness remains a challenge. Existing methods, such as Reinforcement Learning from Human Feedback (RLHF), rely on centralized data collection, making them computationally expensive and privacy-invasive. We introduce PluralLLM a federated learning-based approach that enables multiple user groups to collaboratively train a transformer-based preference predictor without sharing sensitive data, which can also serve as a reward model for aligning LLMs. Our method leverages Federated Averaging (FedAvg) to aggregate preference updates efficiently, achieving 46% faster convergence, a 4% improvement in alignment scores, and nearly the same group fairness measure as in centralized training. Evaluated on a Q/A preference alignment task, PluralLLM demonstrates that federated preference learning offers a scalable and privacy-preserving alternative for aligning LLMs with diverse human values. 

---
# LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics 

**Authors**: Jialiang Tang, Shuo Chen, Chen Gong, Jing Zhang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.09656)  

**Abstract**: Time Series Forecasting (TSF) is critical in many real-world domains like financial planning and health monitoring. Recent studies have revealed that Large Language Models (LLMs), with their powerful in-contextual modeling capabilities, hold significant potential for TSF. However, existing LLM-based methods usually perform suboptimally because they neglect the inherent characteristics of time series data. Unlike the textual data used in LLM pre-training, the time series data is semantically sparse and comprises distinctive temporal patterns. To address this problem, we propose LLM-PS to empower the LLM for TSF by learning the fundamental \textit{Patterns} and meaningful \textit{Semantics} from time series data. Our LLM-PS incorporates a new multi-scale convolutional neural network adept at capturing both short-term fluctuations and long-term trends within the time series. Meanwhile, we introduce a time-to-text module for extracting valuable semantics across continuous time intervals rather than isolated time points. By integrating these patterns and semantics, LLM-PS effectively models temporal dependencies, enabling a deep comprehension of time series and delivering accurate forecasts. Intensive experimental results demonstrate that LLM-PS achieves state-of-the-art performance in both short- and long-term forecasting tasks, as well as in few- and zero-shot settings. 

---
# PRISM: Preference Refinement via Implicit Scene Modeling for 3D Vision-Language Preference-Based Reinforcement Learning 

**Authors**: Yirong Sun, Yanjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10177)  

**Abstract**: We propose PRISM, a novel framework designed to overcome the limitations of 2D-based Preference-Based Reinforcement Learning (PBRL) by unifying 3D point cloud modeling and future-aware preference refinement. At its core, PRISM adopts a 3D Point Cloud-Language Model (3D-PC-LLM) to mitigate occlusion and viewpoint biases, ensuring more stable and spatially consistent preference signals. Additionally, PRISM leverages Chain-of-Thought (CoT) reasoning to incorporate long-horizon considerations, thereby preventing the short-sighted feedback often seen in static preference comparisons. In contrast to conventional PBRL techniques, this integration of 3D perception and future-oriented reasoning leads to significant gains in preference agreement rates, faster policy convergence, and robust generalization across unseen robotic environments. Our empirical results, spanning tasks such as robotic manipulation and autonomous navigation, highlight PRISM's potential for real-world applications where precise spatial understanding and reliable long-term decision-making are critical. By bridging 3D geometric awareness with CoT-driven preference modeling, PRISM establishes a comprehensive foundation for scalable, human-aligned reinforcement learning. 

---
# Global Position Aware Group Choreography using Large Language Model 

**Authors**: Haozhou Pang, Tianwei Ding, Lanshan He, Qi Gan  

**Link**: [PDF](https://arxiv.org/pdf/2503.09645)  

**Abstract**: Dance serves as a profound and universal expression of human culture, conveying emotions and stories through movements synchronized with music. Although some current works have achieved satisfactory results in the task of single-person dance generation, the field of multi-person dance generation remains relatively novel. In this work, we present a group choreography framework that leverages recent advancements in Large Language Models (LLM) by modeling the group dance generation problem as a sequence-to-sequence translation task. Our framework consists of a tokenizer that transforms continuous features into discrete tokens, and an LLM that is fine-tuned to predict motion tokens given the audio tokens. We show that by proper tokenization of input modalities and careful design of the LLM training strategies, our framework can generate realistic and diverse group dances while maintaining strong music correlation and dancer-wise consistency. Extensive experiments and evaluations demonstrate that our framework achieves state-of-the-art performance. 

---
