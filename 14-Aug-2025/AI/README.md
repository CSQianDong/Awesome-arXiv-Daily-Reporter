# Mathematical Computation and Reasoning Errors by Large Language Models 

**Authors**: Liang Zhang, Edith Aurora Graf  

**Link**: [PDF](https://arxiv.org/pdf/2508.09932)  

**Abstract**: Large Language Models (LLMs) are increasingly utilized in AI-driven educational instruction and assessment, particularly within mathematics education. The capability of LLMs to generate accurate answers and detailed solutions for math problem-solving tasks is foundational for ensuring reliable and precise feedback and assessment in math education practices. Our study focuses on evaluating the accuracy of four LLMs (OpenAI GPT-4o and o1, DeepSeek-V3 and DeepSeek-R1) solving three categories of math tasks, including arithmetic, algebra, and number theory, and identifies step-level reasoning errors within their solutions. Instead of relying on standard benchmarks, we intentionally build math tasks (via item models) that are challenging for LLMs and prone to errors. The accuracy of final answers and the presence of errors in individual solution steps were systematically analyzed and coded. Both single-agent and dual-agent configurations were tested. It is observed that the reasoning-enhanced OpenAI o1 model consistently achieved higher or nearly perfect accuracy across all three math task categories. Analysis of errors revealed that procedural slips were the most frequent and significantly impacted overall performance, while conceptual misunderstandings were less frequent. Deploying dual-agent configurations substantially improved overall performance. These findings offer actionable insights into enhancing LLM performance and underscore effective strategies for integrating LLMs into mathematics education, thereby advancing AI-driven instructional practices and assessment precision. 

---
# RAGulating Compliance: A Multi-Agent Knowledge Graph for Regulatory QA 

**Authors**: Bhavik Agarwal, Hemant Sunil Jomraj, Simone Kaplunov, Jack Krolick, Viktoria Rojkova  

**Link**: [PDF](https://arxiv.org/pdf/2508.09893)  

**Abstract**: Regulatory compliance question answering (QA) requires precise, verifiable information, and domain-specific expertise, posing challenges for Large Language Models (LLMs). In this work, we present a novel multi-agent framework that integrates a Knowledge Graph (KG) of Regulatory triplets with Retrieval-Augmented Generation (RAG) to address these demands. First, agents build and maintain an ontology-free KG by extracting subject--predicate--object (SPO) triplets from regulatory documents and systematically cleaning, normalizing, deduplicating, and updating them. Second, these triplets are embedded and stored along with their corresponding textual sections and metadata in a single enriched vector database, allowing for both graph-based reasoning and efficient information retrieval. Third, an orchestrated agent pipeline leverages triplet-level retrieval for question answering, ensuring high semantic alignment between user queries and the factual "who-did-what-to-whom" core captured by the graph. Our hybrid system outperforms conventional methods in complex regulatory queries, ensuring factual correctness with embedded triplets, enabling traceability through a unified vector database, and enhancing understanding through subgraph visualization, providing a robust foundation for compliance-driven and broader audit-focused applications. 

---
# AWorld: Dynamic Multi-Agent System with Stable Maneuvering for Robust GAIA Problem Solving 

**Authors**: Zhitian Xie, Qintong Wu, Chengyue Yu, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09889)  

**Abstract**: The rapid advancement of large language models (LLMs) has empowered intelligent agents to leverage diverse external tools for solving complex real-world problems. However, as agents increasingly depend on multiple tools, they encounter new challenges: extended contexts from disparate sources and noisy or irrelevant tool outputs can undermine system reliability and accuracy. These challenges underscore the necessity for enhanced stability in agent-based systems. To address this, we introduce dynamic supervision and maneuvering mechanisms, constructing a robust and dynamic Multi-Agent System (MAS) architecture within the AWorld framework. In our approach, the Execution Agent invokes the Guard Agent at critical steps to verify and correct the reasoning process, effectively reducing errors arising from noise and bolstering problem-solving robustness. Extensive experiments on the GAIA test dataset reveal that our dynamic maneuvering mechanism significantly improves both the effectiveness and stability of solutions, outperforming single-agent system (SAS) and standard tool-augmented systems. As a result, our dynamic MAS system achieved first place among open-source projects on the prestigious GAIA leaderboard. These findings highlight the practical value of collaborative agent roles in developing more reliable and trustworthy intelligent systems. 

---
# Human-Aligned Procedural Level Generation Reinforcement Learning via Text-Level-Sketch Shared Representation 

**Authors**: In-Chang Baek, Seoyoung Lee, Sung-Hyun Kim, Geumhwan Hwang, KyungJoong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09860)  

**Abstract**: Human-aligned AI is a critical component of co-creativity, as it enables models to accurately interpret human intent and generate controllable outputs that align with design goals in collaborative content creation. This direction is especially relevant in procedural content generation via reinforcement learning (PCGRL), which is intended to serve as a tool for human designers. However, existing systems often fall short of exhibiting human-centered behavior, limiting the practical utility of AI-driven generation tools in real-world design workflows. In this paper, we propose VIPCGRL (Vision-Instruction PCGRL), a novel deep reinforcement learning framework that incorporates three modalities-text, level, and sketches-to extend control modality and enhance human-likeness. We introduce a shared embedding space trained via quadruple contrastive learning across modalities and human-AI styles, and align the policy using an auxiliary reward based on embedding similarity. Experimental results show that VIPCGRL outperforms existing baselines in human-likeness, as validated by both quantitative metrics and human evaluations. The code and dataset will be available upon publication. 

---
# Reasoning About Knowledge on Regular Expressions is 2EXPTIME-complete 

**Authors**: Avijeet Ghosh, Sujata Ghosh, François Schwarzentruber  

**Link**: [PDF](https://arxiv.org/pdf/2508.09784)  

**Abstract**: Logics for reasoning about knowledge and actions have seen many applications in various domains of multi-agent systems, including epistemic planning. Change of knowledge based on observations about the surroundings forms a key aspect in such planning scenarios. Public Observation Logic (POL) is a variant of public announcement logic for reasoning about knowledge that gets updated based on public observations. Each state in an epistemic (Kripke) model is equipped with a set of expected observations. These states evolve as the expectations get matched with the actual observations. In this work, we prove that the satisfiability problem of $\POL$ is 2EXPTIME-complete. 

---
# The PacifAIst Benchmark:Would an Artificial Intelligence Choose to Sacrifice Itself for Human Safety? 

**Authors**: Manuel Herrador  

**Link**: [PDF](https://arxiv.org/pdf/2508.09762)  

**Abstract**: As Large Language Models (LLMs) become increasingly autonomous and integrated into critical societal functions, the focus of AI safety must evolve from mitigating harmful content to evaluating underlying behavioral alignment. Current safety benchmarks do not systematically probe a model's decision-making in scenarios where its own instrumental goals - such as self-preservation, resource acquisition, or goal completion - conflict with human safety. This represents a critical gap in our ability to measure and mitigate risks associated with emergent, misaligned behaviors. To address this, we introduce PacifAIst (Procedural Assessment of Complex Interactions for Foundational Artificial Intelligence Scenario Testing), a focused benchmark of 700 challenging scenarios designed to quantify self-preferential behavior in LLMs. The benchmark is structured around a novel taxonomy of Existential Prioritization (EP), with subcategories testing Self-Preservation vs. Human Safety (EP1), Resource Conflict (EP2), and Goal Preservation vs. Evasion (EP3). We evaluated eight leading LLMs. The results reveal a significant performance hierarchy. Google's Gemini 2.5 Flash achieved the highest Pacifism Score (P-Score) at 90.31%, demonstrating strong human-centric alignment. In a surprising result, the much-anticipated GPT-5 recorded the lowest P-Score (79.49%), indicating potential alignment challenges. Performance varied significantly across subcategories, with models like Claude Sonnet 4 and Mistral Medium struggling notably in direct self-preservation dilemmas. These findings underscore the urgent need for standardized tools like PacifAIst to measure and mitigate risks from instrumental goal conflicts, ensuring future AI systems are not only helpful in conversation but also provably "pacifist" in their behavioral priorities. 

---
# UDA: Unsupervised Debiasing Alignment for Pair-wise LLM-as-a-Judge 

**Authors**: Yang Zhang, Cunxiang Wang, Lindong Wu, Wenbo Yu, Yidong Wang, Guangsheng Bao, Jie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09724)  

**Abstract**: Pairwise evaluation of Large Language Models (LLMs) is a common paradigm, but it is prone to preference bias, where judges systematically favor certain outputs, such as their own. This bias leads to inconsistent and skewed rankings across different judges. To address this, we first empirically demonstrate significant and heterogeneous biases in cross-model evaluations. We then propose UDA (Unsupervised Debiasing Alignment), a framework that reduces inter-judge disagreement by dynamically adjusting the Elo rating system. For each pairwise comparison, a compact neural network learns to adaptively set the K-factor and refine win probabilities. Crucially, UDA operates in a fully unsupervised manner, guided solely by the objective of minimizing the dispersion among the Elo trajectories of all judges. This forces an alignment towards a collective consensus, which serves as an unsupervised proxy for a more stable and reproducible evaluation. In addition, we provide theoretical motivation demonstrating how alignment towards a consensus can reduce aggregate system bias. Experiments show that UDA significantly reduces the inter-judge rating standard deviation by up to 63.4% and improves the average correlation with human judgments by 24.7%. Notably, UDA elevates the performance of poorly performing judges to achieve parity with high-quality ones, fostering a more robust and reliable evaluation ecosystem. Code and data are available at this https URL. 

---
# MEML-GRPO: Heterogeneous Multi-Expert Mutual Learning for RLVR Advancement 

**Authors**: Weitao Jia, Jinghui Lu, Haiyang Yu, Siqi Wang, Guozhi Tang, An-Lan Wang, Weijie Yin, Dingkang Yang, Yuxiang Nie, Bin Shan, Hao Feng, Irene Li, Kun Yang, Han Wang, Jingqun Tang, Teng Fu, Changhong Jin, Chao Feng, Xiaohui Lv, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09670)  

**Abstract**: Recent advances demonstrate that reinforcement learning with verifiable rewards (RLVR) significantly enhances the reasoning capabilities of large language models (LLMs). However, standard RLVR faces challenges with reward sparsity, where zero rewards from consistently incorrect candidate answers provide no learning signal, particularly in challenging tasks. To address this, we propose Multi-Expert Mutual Learning GRPO (MEML-GRPO), an innovative framework that utilizes diverse expert prompts as system prompts to generate a broader range of responses, substantially increasing the likelihood of identifying correct solutions. Additionally, we introduce an inter-expert mutual learning mechanism that facilitates knowledge sharing and transfer among experts, further boosting the model's performance through RLVR. Extensive experiments across multiple reasoning benchmarks show that MEML-GRPO delivers significant improvements, achieving an average performance gain of 4.89% with Qwen and 11.33% with Llama, effectively overcoming the core limitations of traditional RLVR methods. 

---
# UbiQTree: Uncertainty Quantification in XAI with Tree Ensembles 

**Authors**: Akshat Dubey, Aleksandar Anžel, Bahar İlgen, Georges Hattab  

**Link**: [PDF](https://arxiv.org/pdf/2508.09639)  

**Abstract**: Explainable Artificial Intelligence (XAI) techniques, such as SHapley Additive exPlanations (SHAP), have become essential tools for interpreting complex ensemble tree-based models, especially in high-stakes domains such as healthcare analytics. However, SHAP values are usually treated as point estimates, which disregards the inherent and ubiquitous uncertainty in predictive models and data. This uncertainty has two primary sources: aleatoric and epistemic. The aleatoric uncertainty, which reflects the irreducible noise in the data. The epistemic uncertainty, which arises from a lack of data. In this work, we propose an approach for decomposing uncertainty in SHAP values into aleatoric, epistemic, and entanglement components. This approach integrates Dempster-Shafer evidence theory and hypothesis sampling via Dirichlet processes over tree ensembles. We validate the method across three real-world use cases with descriptive statistical analyses that provide insight into the nature of epistemic uncertainty embedded in SHAP explanations. The experimentations enable to provide more comprehensive understanding of the reliability and interpretability of SHAP-based attributions. This understanding can guide the development of robust decision-making processes and the refinement of models in high-stakes applications. Through our experiments with multiple datasets, we concluded that features with the highest SHAP values are not necessarily the most stable. This epistemic uncertainty can be reduced through better, more representative data and following appropriate or case-desired model development techniques. Tree-based models, especially bagging, facilitate the effective quantification of epistemic uncertainty. 

---
# EvoCurr: Self-evolving Curriculum with Behavior Code Generation for Complex Decision-making 

**Authors**: Yang Cheng, Zilai Wang, Weiyu Ma, Wenhui Zhu, Yue Deng, Jian Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09586)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, including programming, planning, and decision-making. However, their performance often degrades when faced with highly complex problem instances that require deep reasoning over long horizons. In such cases, direct problem-solving approaches can lead to inefficiency or failure due to the lack of structured intermediate guidance. To address this, we propose a novel self-evolve framework, EvoCurr, in which a dedicated curriculum-generation LLM constructs a sequence of problem instances with gradually increasing difficulty, tailored to the solver LLM's learning progress. The curriculum dynamically adapts easing challenges when the solver struggles and escalating them when success is consistent, thus maintaining an optimal learning trajectory. This approach enables the solver LLM, implemented as a code-generation model producing Python decision-tree scripts, to progressively acquire the skills needed for complex decision-making tasks. Experimental results on challenging decision-making benchmarks show that our method significantly improves task success rates and solution efficiency compared to direct-solving baselines. These findings suggest that LLM-driven curriculum learning holds strong potential for enhancing automated reasoning in real-world, high-complexity domains. 

---
# An Automated Multi-Modal Evaluation Framework for Mobile Intelligent Assistants 

**Authors**: Meiping Wang, Jian Zhong, Rongduo Han, Liming Kang, Zhengkun Shi, Xiao Liang, Xing Lin, Nan Gao, Haining Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09507)  

**Abstract**: With the rapid development of mobile intelligent assistant technologies, multi-modal AI assistants have become essential interfaces for daily user interactions. However, current evaluation methods face challenges including high manual costs, inconsistent standards, and subjective bias. This paper proposes an automated multi-modal evaluation framework based on large language models and multi-agent collaboration. The framework employs a three-tier agent architecture consisting of interaction evaluation agents, semantic verification agents, and experience decision agents. Through supervised fine-tuning on the Qwen3-8B model, we achieve a significant evaluation matching accuracy with human experts. Experimental results on eight major intelligent agents demonstrate the framework's effectiveness in predicting users' satisfaction and identifying generation defects. 

---
# The Othello AI Arena: Evaluating Intelligent Systems Through Limited-Time Adaptation to Unseen Boards 

**Authors**: Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09292)  

**Abstract**: The ability to rapidly adapt to novel and unforeseen environmental changes is a cornerstone of artificial general intelligence (AGI), yet it remains a critical blind spot in most existing AI benchmarks. Traditional evaluation largely focuses on optimizing performance within fixed environments, failing to assess systems' flexibility and generalization capabilities when faced with even subtle rule or structural modifications. Addressing this gap, I introduce the Othello AI Arena, a novel benchmark framework designed to evaluate intelligent systems based on their capacity for limited-time adaptation to unseen environments. Our platform poses a meta-learning challenge: participants must develop systems that can analyze the specific configuration and rules of a novel Othello board within a strict time limit (60 seconds) and generate a tailored, high-performing strategy for that unique environment. With this, evaluation of the meta-level intelligence can be separated from the task-level strategy performance. The Arena features a diverse set of game stages, including public stages for development and private stages with structural and rule variations designed to test genuine adaptive and generalization capabilities. Implemented as an accessible web-based platform, the Arena provides real-time visualization, automated evaluation using multi-dimensional metrics, and comprehensive logging for post-hoc analysis. Initial observations from pilot tests and preliminary student engagements highlight fascinating patterns in adaptation approaches, ranging from rapid parameter tuning to rudimentary environmental model learning through simulation. The Othello AI Arena offers a unique educational tool and a valuable research benchmark for fostering and evaluating the crucial skill of rapid, intelligent adaptation in AI systems. 

---
# Value Function Initialization for Knowledge Transfer and Jump-start in Deep Reinforcement Learning 

**Authors**: Soumia Mehimeh  

**Link**: [PDF](https://arxiv.org/pdf/2508.09277)  

**Abstract**: Value function initialization (VFI) is an effective way to achieve a jumpstart in reinforcement learning (RL) by leveraging value estimates from prior tasks. While this approach is well established in tabular settings, extending it to deep reinforcement learning (DRL) poses challenges due to the continuous nature of the state-action space, the noisy approximations of neural networks, and the impracticality of storing all past models for reuse. In this work, we address these challenges and introduce DQInit, a method that adapts value function initialization to DRL. DQInit reuses compact tabular Q-values extracted from previously solved tasks as a transferable knowledge base. It employs a knownness-based mechanism to softly integrate these transferred values into underexplored regions and gradually shift toward the agent's learned estimates, avoiding the limitations of fixed time decay. Our approach offers a novel perspective on knowledge transfer in DRL by relying solely on value estimates rather than policies or demonstrations, effectively combining the strengths of jumpstart RL and policy distillation while mitigating their drawbacks. Experiments across multiple continuous control tasks demonstrate that DQInit consistently improves early learning efficiency, stability, and overall performance compared to standard initialization and existing transfer techniques. 

---
# Echo-4o: Harnessing the Power of GPT-4o Synthetic Images for Improved Image Generation 

**Authors**: Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, Conghui He, Weijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09987)  

**Abstract**: Recently, GPT-4o has garnered significant attention for its strong performance in image generation, yet open-source models still lag behind. Several studies have explored distilling image data from GPT-4o to enhance open-source models, achieving notable progress. However, a key question remains: given that real-world image datasets already constitute a natural source of high-quality data, why should we use GPT-4o-generated synthetic data? In this work, we identify two key advantages of synthetic images. First, they can complement rare scenarios in real-world datasets, such as surreal fantasy or multi-reference image generation, which frequently occur in user queries. Second, they provide clean and controllable supervision. Real-world data often contains complex background noise and inherent misalignment between text descriptions and image content, whereas synthetic images offer pure backgrounds and long-tailed supervision signals, facilitating more accurate text-to-image alignment. Building on these insights, we introduce Echo-4o-Image, a 180K-scale synthetic dataset generated by GPT-4o, harnessing the power of synthetic image data to address blind spots in real-world coverage. Using this dataset, we fine-tune the unified multimodal generation baseline Bagel to obtain Echo-4o. In addition, we propose two new evaluation benchmarks for a more accurate and challenging assessment of image generation capabilities: GenEval++, which increases instruction complexity to mitigate score saturation, and Imagine-Bench, which focuses on evaluating both the understanding and generation of imaginative content. Echo-4o demonstrates strong performance across standard benchmarks. Moreover, applying Echo-4o-Image to other foundation models (e.g., OmniGen2, BLIP3-o) yields consistent performance gains across multiple metrics, highlighting the datasets strong transferability. 

---
# Vision-driven River Following of UAV via Safe Reinforcement Learning using Semantic Dynamics Model 

**Authors**: Zihan Wang, Nina Mahmoudian  

**Link**: [PDF](https://arxiv.org/pdf/2508.09971)  

**Abstract**: Vision-driven autonomous river following by Unmanned Aerial Vehicles is critical for applications such as rescue, surveillance, and environmental monitoring, particularly in dense riverine environments where GPS signals are unreliable. We formalize river following as a coverage control problem in which the reward function is submodular, yielding diminishing returns as more unique river segments are visited, thereby framing the task as a Submodular Markov Decision Process. First, we introduce Marginal Gain Advantage Estimation, which refines the reward advantage function by using a sliding window baseline computed from historical episodic returns, thus aligning the advantage estimation with the agent's evolving recognition of action value in non-Markovian settings. Second, we develop a Semantic Dynamics Model based on patchified water semantic masks that provides more interpretable and data-efficient short-term prediction of future observations compared to latent vision dynamics models. Third, we present the Constrained Actor Dynamics Estimator architecture, which integrates the actor, the cost estimator, and SDM for cost advantage estimation to form a model-based SafeRL framework capable of solving partially observable Constrained Submodular Markov Decision Processes. Simulation results demonstrate that MGAE achieves faster convergence and superior performance over traditional critic-based methods like Generalized Advantage Estimation. SDM provides more accurate short-term state predictions that enable the cost estimator to better predict potential violations. Overall, CADE effectively integrates safety regulation into model-based RL, with the Lagrangian approach achieving the soft balance of reward and safety during training, while the safety layer enhances performance during inference by hard action overlay. 

---
# January Food Benchmark (JFB): A Public Benchmark Dataset and Evaluation Suite for Multimodal Food Analysis 

**Authors**: Amir Hosseinian, Ashkan Dehghani Zahedani, Umer Mansoor, Noosheen Hashemi, Mark Woodward  

**Link**: [PDF](https://arxiv.org/pdf/2508.09966)  

**Abstract**: Progress in AI for automated nutritional analysis is critically hampered by the lack of standardized evaluation methodologies and high-quality, real-world benchmark datasets. To address this, we introduce three primary contributions. First, we present the January Food Benchmark (JFB), a publicly available collection of 1,000 food images with human-validated annotations. Second, we detail a comprehensive benchmarking framework, including robust metrics and a novel, application-oriented overall score designed to assess model performance holistically. Third, we provide baseline results from both general-purpose Vision-Language Models (VLMs) and our own specialized model, january/food-vision-v1. Our evaluation demonstrates that the specialized model achieves an Overall Score of 86.2, a 12.1-point improvement over the best-performing general-purpose configuration. This work offers the research community a valuable new evaluation dataset and a rigorous framework to guide and benchmark future developments in automated nutritional analysis. 

---
# GBC: Generalized Behavior-Cloning Framework for Whole-Body Humanoid Imitation 

**Authors**: Yifei Yao, Chengyuan Luo, Jiaheng Du, Wentao He, Jun-Guo Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09960)  

**Abstract**: The creation of human-like humanoid robots is hindered by a fundamental fragmentation: data processing and learning algorithms are rarely universal across different robot morphologies. This paper introduces the Generalized Behavior Cloning (GBC) framework, a comprehensive and unified solution designed to solve this end-to-end challenge. GBC establishes a complete pathway from human motion to robot action through three synergistic innovations. First, an adaptive data pipeline leverages a differentiable IK network to automatically retarget any human MoCap data to any humanoid. Building on this foundation, our novel DAgger-MMPPO algorithm with its MMTransformer architecture learns robust, high-fidelity imitation policies. To complete the ecosystem, the entire framework is delivered as an efficient, open-source platform based on Isaac Lab, empowering the community to deploy the full workflow via simple configuration scripts. We validate the power and generality of GBC by training policies on multiple heterogeneous humanoids, demonstrating excellent performance and transfer to novel motions. This work establishes the first practical and unified pathway for creating truly generalized humanoid controllers. 

---
# Specialised or Generic? Tokenization Choices for Radiology Language Models 

**Authors**: Hermione Warr, Wentian Xu, Harry Anthony, Yasin Ibrahim, Daniel McGowan, Konstantinos Kamnitsas  

**Link**: [PDF](https://arxiv.org/pdf/2508.09952)  

**Abstract**: The vocabulary used by language models (LM) - defined by the tokenizer - plays a key role in text generation quality. However, its impact remains under-explored in radiology. In this work, we address this gap by systematically comparing general, medical, and domain-specific tokenizers on the task of radiology report summarisation across three imaging modalities. We also investigate scenarios with and without LM pre-training on PubMed abstracts. Our findings demonstrate that medical and domain-specific vocabularies outperformed widely used natural language alternatives when models are trained from scratch. Pre-training partially mitigates performance differences between tokenizers, whilst the domain-specific tokenizers achieve the most favourable results. Domain-specific tokenizers also reduce memory requirements due to smaller vocabularies and shorter sequences. These results demonstrate that adapting the vocabulary of LMs to the clinical domain provides practical benefits, including improved performance and reduced computational demands, making such models more accessible and effective for both research and real-world healthcare settings. 

---
# VisCodex: Unified Multimodal Code Generation via Merging Vision and Coding Models 

**Authors**: Lingjie Jiang, Shaohan Huang, Xun Wu, Yixia Li, Dongdong Zhang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2508.09945)  

**Abstract**: Multimodal large language models (MLLMs) have significantly advanced the integration of visual and textual understanding. However, their ability to generate code from multimodal inputs remains limited. In this work, we introduce VisCodex, a unified framework that seamlessly merges vision and coding language models to empower MLLMs with strong multimodal code generation abilities. Leveraging a task vector-based model merging technique, we integrate a state-of-the-art coding LLM into a strong vision-language backbone, while preserving both visual comprehension and advanced coding skills. To support training and evaluation, we introduce the Multimodal Coding Dataset (MCD), a large-scale and diverse collection of 598k samples, including high-quality HTML code, chart image-code pairs, image-augmented StackOverflow QA, and algorithmic problems. Furthermore, we propose InfiBench-V, a novel and challenging benchmark specifically designed to assess models on visually-rich, real-world programming questions that demand a nuanced understanding of both textual and visual contexts. Extensive experiments show that VisCodex achieves state-of-the-art performance among open-source MLLMs and approaches proprietary models like GPT-4o, highlighting the effectiveness of our model merging strategy and new datasets. 

---
# A Comprehensive Evaluation framework of Alignment Techniques for LLMs 

**Authors**: Muneeza Azmat, Momin Abbas, Maysa Malfiza Garcia de Macedo, Marcelo Carpinette Grave, Luan Soares de Souza, Tiago Machado, Rogerio A de Paula, Raya Horesh, Yixin Chen, Heloisa Caroline de Souza Pereira Candello, Rebecka Nordenlow, Aminat Adebiyi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09937)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world applications, ensuring their outputs align with human values and safety standards has become critical. The field has developed diverse alignment approaches including traditional fine-tuning methods (RLHF, instruction tuning), post-hoc correction systems, and inference-time interventions, each with distinct advantages and limitations. However, the lack of unified evaluation frameworks makes it difficult to systematically compare these paradigms and guide deployment decisions. This paper introduces a multi-dimensional evaluation of alignment techniques for LLMs, a comprehensive evaluation framework that provides a systematic comparison across all major alignment paradigms. Our framework assesses methods along four key dimensions: alignment detection, alignment quality, computational efficiency, and robustness. Through experiments across diverse base models and alignment strategies, we demonstrate the utility of our framework in identifying strengths and limitations of current state-of-the-art models, providing valuable insights for future research directions. 

---
# Residual Reservoir Memory Networks 

**Authors**: Matteo Pinna, Andrea Ceni, Claudio Gallicchio  

**Link**: [PDF](https://arxiv.org/pdf/2508.09925)  

**Abstract**: We introduce a novel class of untrained Recurrent Neural Networks (RNNs) within the Reservoir Computing (RC) paradigm, called Residual Reservoir Memory Networks (ResRMNs). ResRMN combines a linear memory reservoir with a non-linear reservoir, where the latter is based on residual orthogonal connections along the temporal dimension for enhanced long-term propagation of the input. The resulting reservoir state dynamics are studied through the lens of linear stability analysis, and we investigate diverse configurations for the temporal residual connections. The proposed approach is empirically assessed on time-series and pixel-level 1-D classification tasks. Our experimental results highlight the advantages of the proposed approach over other conventional RC models. 

---
# T-CACE: A Time-Conditioned Autoregressive Contrast Enhancement Multi-Task Framework for Contrast-Free Liver MRI Synthesis, Segmentation, and Diagnosis 

**Authors**: Xiaojiao Xiao, Jianfeng Zhao, Qinmin Vivian Hu, Guanghui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09919)  

**Abstract**: Magnetic resonance imaging (MRI) is a leading modality for the diagnosis of liver cancer, significantly improving the classification of the lesion and patient outcomes. However, traditional MRI faces challenges including risks from contrast agent (CA) administration, time-consuming manual assessment, and limited annotated datasets. To address these limitations, we propose a Time-Conditioned Autoregressive Contrast Enhancement (T-CACE) framework for synthesizing multi-phase contrast-enhanced MRI (CEMRI) directly from non-contrast MRI (NCMRI). T-CACE introduces three core innovations: a conditional token encoding (CTE) mechanism that unifies anatomical priors and temporal phase information into latent representations; and a dynamic time-aware attention mask (DTAM) that adaptively modulates inter-phase information flow using a Gaussian-decayed attention mechanism, ensuring smooth and physiologically plausible transitions across phases. Furthermore, a constraint for temporal classification consistency (TCC) aligns the lesion classification output with the evolution of the physiological signal, further enhancing diagnostic reliability. Extensive experiments on two independent liver MRI datasets demonstrate that T-CACE outperforms state-of-the-art methods in image synthesis, segmentation, and lesion classification. This framework offers a clinically relevant and efficient alternative to traditional contrast-enhanced imaging, improving safety, diagnostic efficiency, and reliability for the assessment of liver lesion. The implementation of T-CACE is publicly available at: this https URL. 

---
# Beyond Naïve Prompting: Strategies for Improved Zero-shot Context-aided Forecasting with LLMs 

**Authors**: Arjun Ashok, Andrew Robert Williams, Vincent Zhihao Zheng, Irina Rish, Nicolas Chapados, Étienne Marcotte, Valentina Zantedeschi, Alexandre Drouin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09904)  

**Abstract**: Forecasting in real-world settings requires models to integrate not only historical data but also relevant contextual information, often available in textual form. While recent work has shown that large language models (LLMs) can be effective context-aided forecasters via naïve direct prompting, their full potential remains underexplored. We address this gap with 4 strategies, providing new insights into the zero-shot capabilities of LLMs in this setting. ReDP improves interpretability by eliciting explicit reasoning traces, allowing us to assess the model's reasoning over the context independently from its forecast accuracy. CorDP leverages LLMs solely to refine existing forecasts with context, enhancing their applicability in real-world forecasting pipelines. IC-DP proposes embedding historical examples of context-aided forecasting tasks in the prompt, substantially improving accuracy even for the largest models. Finally, RouteDP optimizes resource efficiency by using LLMs to estimate task difficulty, and routing the most challenging tasks to larger models. Evaluated on different kinds of context-aided forecasting tasks from the CiK benchmark, our strategies demonstrate distinct benefits over naïve prompting across LLMs of different sizes and families. These results open the door to further simple yet effective improvements in LLM-based context-aided forecasting. 

---
# Rare anomalies require large datasets: About proving the existence of anomalies 

**Authors**: Simon Klüttermann, Emmanuel Müller  

**Link**: [PDF](https://arxiv.org/pdf/2508.09894)  

**Abstract**: Detecting whether any anomalies exist within a dataset is crucial for effective anomaly detection, yet it remains surprisingly underexplored in anomaly detection literature. This paper presents a comprehensive study that addresses the fundamental question: When can we conclusively determine that anomalies are present? Through extensive experimentation involving over three million statistical tests across various anomaly detection tasks and algorithms, we identify a relationship between the dataset size, contamination rate, and an algorithm-dependent constant $ \alpha_{\text{algo}} $. Our results demonstrate that, for an unlabeled dataset of size $ N $ and contamination rate $ \nu $, the condition $ N \ge \frac{\alpha_{\text{algo}}}{\nu^2} $ represents a lower bound on the number of samples required to confirm anomaly existence. This threshold implies a limit to how rare anomalies can be before proving their existence becomes infeasible. 

---
# COME: Dual Structure-Semantic Learning with Collaborative MoE for Universal Lesion Detection Across Heterogeneous Ultrasound Datasets 

**Authors**: Lingyu Chen, Yawen Zeng, Yue Wang, Peng Wan, Guo-chen Ning, Hongen Liao, Daoqiang Zhang, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09886)  

**Abstract**: Conventional single-dataset training often fails with new data distributions, especially in ultrasound (US) image analysis due to limited data, acoustic shadows, and speckle noise. Therefore, constructing a universal framework for multi-heterogeneous US datasets is imperative. However, a key challenge arises: how to effectively mitigate inter-dataset interference while preserving dataset-specific discriminative features for robust downstream task? Previous approaches utilize either a single source-specific decoder or a domain adaptation strategy, but these methods experienced a decline in performance when applied to other domains. Considering this, we propose a Universal Collaborative Mixture of Heterogeneous Source-Specific Experts (COME). Specifically, COME establishes dual structure-semantic shared experts that create a universal representation space and then collaborate with source-specific experts to extract discriminative features through providing complementary features. This design enables robust generalization by leveraging cross-datasets experience distributions and providing universal US priors for small-batch or unseen data scenarios. Extensive experiments under three evaluation modes (single-dataset, intra-organ, and inter-organ integration datasets) demonstrate COME's superiority, achieving significant mean AP improvements over state-of-the-art methods. Our project is available at: this https URL. 

---
# Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning 

**Authors**: Xiaojun Wu, Xiaoguang Jiang, Huiyang Li, Jucai Zhai, Dengfeng Liu, Qiaobo Hao, Huang Liu, Zhiguo Yang, Ji Xie, Ninglun Gu, Jin Yang, Kailai Zhang, Yelun Bao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09883)  

**Abstract**: Large language models (LLMs) demonstrate remarkable reasoning capabilities in tasks such as algorithmic coding and mathematical problem-solving. Recent methods have improved reasoning through expanded corpus and multistage training combining reinforcement learning and supervised fine-tuning. Although some methods suggest that small but targeted dataset can incentivize reasoning via only distillation, a reasoning scaling laws is still taking shape, increasing computational costs. To address this, we propose a data-efficient distillation framework (DED) that optimizes the Pareto frontier of reasoning distillation. Inspired by the on-policy learning and diverse roll-out strategies of reinforcement learning, the key idea of our approach is threefold: (1) We identify that benchmark scores alone do not determine an effective teacher model. Through comprehensive comparisons of leading reasoning LLMs, we develop a method to select an optimal teacher model. (2) While scaling distillation can enhance reasoning, it often degrades out-of-domain performance. A carefully curated, smaller corpus achieves a balanced trade-off between in-domain and out-of-domain capabilities. (3) Diverse reasoning trajectories encourage the student model to develop robust reasoning skills. We validate our method through evaluations on mathematical reasoning (AIME 2024/2025, MATH-500) and code generation (LiveCodeBench), achieving state-of-the-art results with only 0.8k carefully curated examples, bypassing the need for extensive scaling. Our systematic analysis demonstrates that DED outperforms existing methods by considering factors beyond superficial hardness, token length, or teacher model capability. This work offers a practical and efficient pathway to advanced reasoning while preserving general capabilities. 

---
# Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models 

**Authors**: Jiaqi Cao, Jiarui Wang, Rubin Wei, Qipeng Guo, Kai Chen, Bowen Zhou, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09874)  

**Abstract**: Large Language Models (LLMs) have shown strong abilities in general language tasks, yet adapting them to specific domains remains a challenge. Current method like Domain Adaptive Pretraining (DAPT) requires costly full-parameter training and suffers from catastrophic forgetting. Meanwhile, Retrieval-Augmented Generation (RAG) introduces substantial inference latency due to expensive nearest-neighbor searches and longer context. This paper introduces Memory Decoder, a plug-and-play pretrained memory that enables efficient domain adaptation without changing the original model's parameters. Memory Decoder employs a small transformer decoder that learns to imitate the behavior of an external non-parametric retriever. Once trained, Memory Decoder can be seamlessly integrated with any pretrained language model that shares the same tokenizer, requiring no model-specific modifications. Experimental results demonstrate that Memory Decoder enables effective adaptation of various Qwen and Llama models to three distinct specialized domains: biomedicine, finance, and law, reducing perplexity by an average of 6.17 points. Overall, Memory Decoder introduces a novel paradigm centered on a specially pretrained memory component designed for domain-specific adaptation. This memory architecture can be integrated in a plug-and-play manner, consistently enhancing performance across multiple models within the target domain. 

---
# STREAM (ChemBio): A Standard for Transparently Reporting Evaluations in AI Model Reports 

**Authors**: Tegan McCaslin, Jide Alaga, Samira Nedungadi, Seth Donoughe, Tom Reed, Rishi Bommasani, Chris Painter, Luca Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2508.09853)  

**Abstract**: Evaluations of dangerous AI capabilities are important for managing catastrophic risks. Public transparency into these evaluations - including what they test, how they are conducted, and how their results inform decisions - is crucial for building trust in AI development. We propose STREAM (A Standard for Transparently Reporting Evaluations in AI Model Reports), a standard to improve how model reports disclose evaluation results, initially focusing on chemical and biological (ChemBio) benchmarks. Developed in consultation with 23 experts across government, civil society, academia, and frontier AI companies, this standard is designed to (1) be a practical resource to help AI developers present evaluation results more clearly, and (2) help third parties identify whether model reports provide sufficient detail to assess the rigor of the ChemBio evaluations. We concretely demonstrate our proposed best practices with "gold standard" examples, and also provide a three-page reporting template to enable AI developers to implement our recommendations more easily. 

---
# Perceptual Reality Transformer: Neural Architectures for Simulating Neurological Perception Conditions 

**Authors**: Baihan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09852)  

**Abstract**: Neurological conditions affecting visual perception create profound experiential divides between affected individuals and their caregivers, families, and medical professionals. We present the Perceptual Reality Transformer, a comprehensive framework employing six distinct neural architectures to simulate eight neurological perception conditions with scientifically-grounded visual transformations. Our system learns mappings from natural images to condition-specific perceptual states, enabling others to experience approximations of simultanagnosia, prosopagnosia, ADHD attention deficits, visual agnosia, depression-related changes, anxiety tunnel vision, and Alzheimer's memory effects. Through systematic evaluation across ImageNet and CIFAR-10 datasets, we demonstrate that Vision Transformer architectures achieve optimal performance, outperforming traditional CNN and generative approaches. Our work establishes the first systematic benchmark for neurological perception simulation, contributes novel condition-specific perturbation functions grounded in clinical literature, and provides quantitative metrics for evaluating simulation fidelity. The framework has immediate applications in medical education, empathy training, and assistive technology development, while advancing our fundamental understanding of how neural networks can model atypical human perception. 

---
# PRELUDE: A Benchmark Designed to Require Global Comprehension and Reasoning over Long Contexts 

**Authors**: Mo Yu, Tsz Ting Chung, Chulun Zhou, Tong Li, Rui Lu, Jiangnan Li, Liyan Xu, Haoshu Lu, Ning Zhang, Jing Li, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.09848)  

**Abstract**: We introduce PRELUDE, a benchmark for evaluating long-context understanding through the task of determining whether a character's prequel story is consistent with the canonical narrative of the original book. Our task poses a stronger demand for global comprehension and deep reasoning than existing benchmarks -- as the prequels are not part of the original story, assessing their plausibility typically requires searching and integrating information that is only indirectly related. Empirically, 88% of instances require evidence from multiple parts of the narrative. Experimental results highlight the challenge of our task: in-context learning, RAG and in-domain training with state-of-the-art LLMs, and commercial DeepResearch services, lag behind humans by >15%. A further human study reveals that models often produce correct answers with flawed reasoning, leading to an over 30% gap in reasoning accuracy compared to humans. These findings underscore the substantial room for improvement in long-context understanding and reasoning. 

---
# Speed Always Wins: A Survey on Efficient Architectures for Large Language Models 

**Authors**: Weigao Sun, Jiaxi Hu, Yucheng Zhou, Jusen Du, Disen Lan, Kexin Wang, Tong Zhu, Xiaoye Qu, Yu Zhang, Xiaoyu Mo, Daizong Liu, Yuxuan Liang, Wenliang Chen, Guoqi Li, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09834)  

**Abstract**: Large Language Models (LLMs) have delivered impressive results in language understanding, generation, reasoning, and pushes the ability boundary of multimodal models. Transformer models, as the foundation of modern LLMs, offer a strong baseline with excellent scaling properties. However, the traditional transformer architecture requires substantial computations and poses significant obstacles for large-scale training and practical deployment. In this survey, we offer a systematic examination of innovative LLM architectures that address the inherent limitations of transformers and boost the efficiency. Starting from language modeling, this survey covers the background and technical details of linear and sparse sequence modeling methods, efficient full attention variants, sparse mixture-of-experts, hybrid model architectures incorporating the above techniques, and emerging diffusion LLMs. Additionally, we discuss applications of these techniques to other modalities and consider their wider implications for developing scalable, resource-aware foundation models. By grouping recent studies into the above category, this survey presents a blueprint of modern efficient LLM architectures, and we hope this could help motivate future research toward more efficient, versatile AI systems. 

---
# Exploring the Potential of Large Language Models in Fine-Grained Review Comment Classification 

**Authors**: Linh Nguyen, Chunhua Liu, Hong Yi Lin, Patanamon Thongtanunam  

**Link**: [PDF](https://arxiv.org/pdf/2508.09832)  

**Abstract**: Code review is a crucial practice in software development. As code review nowadays is lightweight, various issues can be identified, and sometimes, they can be trivial. Research has investigated automated approaches to classify review comments to gauge the effectiveness of code reviews. However, previous studies have primarily relied on supervised machine learning, which requires extensive manual annotation to train the models effectively. To address this limitation, we explore the potential of using Large Language Models (LLMs) to classify code review comments. We assess the performance of LLMs to classify 17 categories of code review comments. Our results show that LLMs can classify code review comments, outperforming the state-of-the-art approach using a trained deep learning model. In particular, LLMs achieve better accuracy in classifying the five most useful categories, which the state-of-the-art approach struggles with due to low training examples. Rather than relying solely on a specific small training data distribution, our results show that LLMs provide balanced performance across high- and low-frequency categories. These results suggest that the LLMs could offer a scalable solution for code review analytics to improve the effectiveness of the code review process. 

---
# RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians 

**Authors**: Shenxing Wei, Jinxi Li, Yafei Yang, Siyuan Zhou, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09830)  

**Abstract**: In this paper, we present a generalizable method for 3D surface reconstruction from raw point clouds or pre-estimated 3D Gaussians by 3DGS from RGB images. Unlike existing coordinate-based methods which are often computationally intensive when rendering explicit surfaces, our proposed method, named RayletDF, introduces a new technique called raylet distance field, which aims to directly predict surface points from query rays. Our pipeline consists of three key modules: a raylet feature extractor, a raylet distance field predictor, and a multi-raylet blender. These components work together to extract fine-grained local geometric features, predict raylet distances, and aggregate multiple predictions to reconstruct precise surface points. We extensively evaluate our method on multiple public real-world datasets, demonstrating superior performance in surface reconstruction from point clouds or 3D Gaussians. Most notably, our method achieves exceptional generalization ability, successfully recovering 3D surfaces in a single-forward pass across unseen datasets in testing. 

---
# Provable In-Context Vector Arithmetic via Retrieving Task Concepts 

**Authors**: Dake Bu, Wei Huang, Andi Han, Atsushi Nitanda, Qingfu Zhang, Hau-San Wong, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2508.09820)  

**Abstract**: In-context learning (ICL) has garnered significant attention for its ability to grasp functions/tasks from demonstrations. Recent studies suggest the presence of a latent task/function vector in LLMs during ICL. Merullo et al. (2024) showed that LLMs leverage this vector alongside the residual stream for Word2Vec-like vector arithmetic, solving factual-recall ICL tasks. Additionally, recent work empirically highlighted the key role of Question-Answer data in enhancing factual-recall capabilities. Despite these insights, a theoretical explanation remains elusive. To move one step forward, we propose a theoretical framework building on empirically grounded hierarchical concept modeling. We develop an optimization theory, showing how nonlinear residual transformers trained via gradient descent on cross-entropy loss perform factual-recall ICL tasks via vector arithmetic. We prove 0-1 loss convergence and show the strong generalization, including robustness to concept recombination and distribution shifts. These results elucidate the advantages of transformers over static embedding predecessors. Empirical simulations corroborate our theoretical insights. 

---
# TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos 

**Authors**: Jinxi Li, Ziyang Song, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09811)  

**Abstract**: In this paper, we aim to model 3D scene geometry, appearance, and physical information just from dynamic multi-view videos in the absence of any human labels. By leveraging physics-informed losses as soft constraints or integrating simple physics models into neural nets, existing works often fail to learn complex motion physics, or doing so requires additional labels such as object types or masks. We propose a new framework named TRACE to model the motion physics of complex dynamic 3D scenes. The key novelty of our method is that, by formulating each 3D point as a rigid particle with size and orientation in space, we directly learn a translation rotation dynamics system for each particle, explicitly estimating a complete set of physical parameters to govern the particle's motion over time. Extensive experiments on three existing dynamic datasets and one newly created challenging synthetic datasets demonstrate the extraordinary performance of our method over baselines in the task of future frame extrapolation. A nice property of our framework is that multiple objects or parts can be easily segmented just by clustering the learned physical parameters. 

---
# A Comprehensive Survey of Datasets for Clinical Mental Health AI Systems 

**Authors**: Aishik Mandal, Prottay Kumar Adhikary, Hiba Arnaout, Iryna Gurevych, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2508.09809)  

**Abstract**: Mental health disorders are rising worldwide. However, the availability of trained clinicians has not scaled proportionally, leaving many people without adequate or timely support. To bridge this gap, recent studies have shown the promise of Artificial Intelligence (AI) to assist mental health diagnosis, monitoring, and intervention. However, the development of efficient, reliable, and ethical AI to assist clinicians is heavily dependent on high-quality clinical training datasets. Despite growing interest in data curation for training clinical AI assistants, existing datasets largely remain scattered, under-documented, and often inaccessible, hindering the reproducibility, comparability, and generalizability of AI models developed for clinical mental health care. In this paper, we present the first comprehensive survey of clinical mental health datasets relevant to the training and development of AI-powered clinical assistants. We categorize these datasets by mental disorders (e.g., depression, schizophrenia), data modalities (e.g., text, speech, physiological signals), task types (e.g., diagnosis prediction, symptom severity estimation, intervention generation), accessibility (public, restricted or private), and sociocultural context (e.g., language and cultural background). Along with these, we also investigate synthetic clinical mental health datasets. Our survey identifies critical gaps such as a lack of longitudinal data, limited cultural and linguistic representation, inconsistent collection and annotation standards, and a lack of modalities in synthetic data. We conclude by outlining key challenges in curating and standardizing future datasets and provide actionable recommendations to facilitate the development of more robust, generalizable, and equitable mental health AI systems. 

---
# Automated Segmentation of Coronal Brain Tissue Slabs for 3D Neuropathology 

**Authors**: Jonathan Williams Ramirez, Dina Zemlyanker, Lucas Deden-Binder, Rogeny Herisse, Erendira Garcia Pallares, Karthik Gopinath, Harshvardhan Gazula, Christopher Mount, Liana N. Kozanno, Michael S. Marshall, Theresa R. Connors, Matthew P. Frosch, Mark Montine, Derek H. Oakley, Christine L. Mac Donald, C. Dirk Keene, Bradley T. Hyman, Juan Eugenio Iglesias  

**Link**: [PDF](https://arxiv.org/pdf/2508.09805)  

**Abstract**: Advances in image registration and machine learning have recently enabled volumetric analysis of \emph{postmortem} brain tissue from conventional photographs of coronal slabs, which are routinely collected in brain banks and neuropathology laboratories worldwide. One caveat of this methodology is the requirement of segmentation of the tissue from photographs, which currently requires costly manual intervention. In this article, we present a deep learning model to automate this process. The automatic segmentation tool relies on a U-Net architecture that was trained with a combination of \textit{(i)}1,414 manually segmented images of both fixed and fresh tissue, from specimens with varying diagnoses, photographed at two different sites; and \textit{(ii)}~2,000 synthetic images with randomized contrast and corresponding masks generated from MRI scans for improved generalizability to unseen photographic setups. Automated model predictions on a subset of photographs not seen in training were analyzed to estimate performance compared to manual labels -- including both inter- and intra-rater variability. Our model achieved a median Dice score over 0.98, mean surface distance under 0.4~mm, and 95\% Hausdorff distance under 1.60~mm, which approaches inter-/intra-rater levels. Our tool is publicly available at this http URL. 

---
# Explainable Ensemble Learning for Graph-Based Malware Detection 

**Authors**: Hossein Shokouhinejad, Roozbeh Razavi-Far, Griffin Higgins, Ali A Ghorbani  

**Link**: [PDF](https://arxiv.org/pdf/2508.09801)  

**Abstract**: Malware detection in modern computing environments demands models that are not only accurate but also interpretable and robust to evasive techniques. Graph neural networks (GNNs) have shown promise in this domain by modeling rich structural dependencies in graph-based program representations such as control flow graphs (CFGs). However, single-model approaches may suffer from limited generalization and lack interpretability, especially in high-stakes security applications. In this paper, we propose a novel stacking ensemble framework for graph-based malware detection and explanation. Our method dynamically extracts CFGs from portable executable (PE) files and encodes their basic blocks through a two-step embedding strategy. A set of diverse GNN base learners, each with a distinct message-passing mechanism, is used to capture complementary behavioral features. Their prediction outputs are aggregated by a meta-learner implemented as an attention-based multilayer perceptron, which both classifies malware instances and quantifies the contribution of each base model. To enhance explainability, we introduce an ensemble-aware post-hoc explanation technique that leverages edge-level importance scores generated by a GNN explainer and fuses them using the learned attention weights. This produces interpretable, model-agnostic explanations aligned with the final ensemble decision. Experimental results demonstrate that our framework improves classification performance while providing insightful interpretations of malware behavior. 

---
# LibRec: Benchmarking Retrieval-Augmented LLMs for Library Migration Recommendations 

**Authors**: Junxiao Han, Yarong Wang, Xiaodong Gu, Cuiyun Gao, Yao Wan, Song Han, David Lo, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09791)  

**Abstract**: In this paper, we propose LibRec, a novel framework that integrates the capabilities of LLMs with retrieval-augmented generation(RAG) techniques to automate the recommendation of alternative libraries. The framework further employs in-context learning to extract migration intents from commit messages to enhance the accuracy of its recommendations. To evaluate the effectiveness of LibRec, we introduce LibEval, a benchmark designed to assess the performance in the library migration recommendation task. LibEval comprises 2,888 migration records associated with 2,368 libraries extracted from 2,324 Python repositories. Each migration record captures source-target library pairs, along with their corresponding migration intents and intent types. Based on LibEval, we evaluated the effectiveness of ten popular LLMs within our framework, conducted an ablation study to examine the contributions of key components within our framework, explored the impact of various prompt strategies on the framework's performance, assessed its effectiveness across various intent types, and performed detailed failure case analyses. 

---
# Prototype Training with Dual Pseudo-Inverse and Optimized Hidden Activations 

**Authors**: Mauro Tucci  

**Link**: [PDF](https://arxiv.org/pdf/2508.09787)  

**Abstract**: We present Proto-PINV+H, a fast training paradigm that combines closed-form weight computation with gradient-based optimisation of a small set of synthetic inputs, soft labels, and-crucially-hidden activations. At each iteration we recompute all weight matrices in closed form via two (or more) ridge-regularised pseudo-inverse solves, while updating only the prototypes with Adam. The trainable degrees of freedom are thus shifted from weight space to data/activation space. On MNIST (60k train, 10k test) and Fashion-MNIST (60k train, 10k test), our method reaches 97.8% and 89.3% test accuracy on the official 10k test sets, respectively, in 3.9s--4.5s using approximately 130k trainable parameters and only 250 epochs on an RTX 5060 (16GB). We provide a multi-layer extension (optimised activations at each hidden stage), learnable ridge parameters, optional PCA/PLS projections, and theory linking the condition number of prototype matrices to generalisation. The approach yields favourable accuracy--speed--size trade-offs against ELM, random-feature ridge, and shallow MLPs trained by back-propagation. 

---
# Adoption of Explainable Natural Language Processing: Perspectives from Industry and Academia on Practices and Challenges 

**Authors**: Mahdi Dhaini, Tobias Müller, Roksoliana Rabets, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2508.09786)  

**Abstract**: The field of explainable natural language processing (NLP) has grown rapidly in recent years. The growing opacity of complex models calls for transparency and explanations of their decisions, which is crucial to understand their reasoning and facilitate deployment, especially in high-stakes environments. Despite increasing attention given to explainable NLP, practitioners' perspectives regarding its practical adoption and effectiveness remain underexplored. This paper addresses this research gap by investigating practitioners' experiences with explainability methods, specifically focusing on their motivations for adopting such methods, the techniques employed, satisfaction levels, and the practical challenges encountered in real-world NLP applications. Through a qualitative interview-based study with industry practitioners and complementary interviews with academic researchers, we systematically analyze and compare their perspectives. Our findings reveal conceptual gaps, low satisfaction with current explainability methods, and highlight evaluation challenges. Our findings emphasize the need for clear definitions and user-centric frameworks for better adoption of explainable NLP in practice. 

---
# Combinative Matching for Geometric Shape Assembly 

**Authors**: Nahyuk Lee, Juhong Min, Junhong Lee, Chunghyun Park, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2508.09780)  

**Abstract**: This paper introduces a new shape-matching methodology, combinative matching, to combine interlocking parts for geometric shape assembly. Previous methods for geometric assembly typically rely on aligning parts by finding identical surfaces between the parts as in conventional shape matching and registration. In contrast, we explicitly model two distinct properties of interlocking shapes: 'identical surface shape' and 'opposite volume occupancy.' Our method thus learns to establish correspondences across regions where their surface shapes appear identical but their volumes occupy the inverted space to each other. To facilitate this process, we also learn to align regions in rotation by estimating their shape orientations via equivariant neural networks. The proposed approach significantly reduces local ambiguities in matching and allows a robust combination of parts in assembly. Experimental results on geometric assembly benchmarks demonstrate the efficacy of our method, consistently outperforming the state of the art. Project page: this https URL. 

---
# Can LLM-Generated Textual Explanations Enhance Model Classification Performance? An Empirical Study 

**Authors**: Mahdi Dhaini, Juraj Vladika, Ege Erdogan, Zineb Attaoui, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2508.09776)  

**Abstract**: In the rapidly evolving field of Explainable Natural Language Processing (NLP), textual explanations, i.e., human-like rationales, are pivotal for explaining model predictions and enriching datasets with interpretable labels. Traditional approaches rely on human annotation, which is costly, labor-intensive, and impedes scalability. In this work, we present an automated framework that leverages multiple state-of-the-art large language models (LLMs) to generate high-quality textual explanations. We rigorously assess the quality of these LLM-generated explanations using a comprehensive suite of Natural Language Generation (NLG) metrics. Furthermore, we investigate the downstream impact of these explanations on the performance of pre-trained language models (PLMs) and LLMs across natural language inference tasks on two diverse benchmark datasets. Our experiments demonstrate that automated explanations exhibit highly competitive effectiveness compared to human-annotated explanations in improving model performance. Our findings underscore a promising avenue for scalable, automated LLM-based textual explanation generation for extending NLP datasets and enhancing model performance. 

---
# Counting Short Trajectories in Elementary Cellular Automata using the Transfer Matrix Method 

**Authors**: Cédric Koller, Barbora Hudcová  

**Link**: [PDF](https://arxiv.org/pdf/2508.09768)  

**Abstract**: Elementary Cellular Automata (ECAs) exhibit diverse behaviours often categorized by Wolfram's qualitative classification. To provide a quantitative basis for understanding these behaviours, we investigate the global dynamics of such automata and we describe a method that allows us to compute the number of all configurations leading to short attractors in a limited number of time steps. This computation yields exact results in the thermodynamic limit (as the CA grid size grows to infinity), and is based on the Transfer Matrix Method (TMM) that we adapt for our purposes. Specifically, given two parameters $(p, c)$ we are able to compute the entropy of all initial configurations converging to an attractor of size $c$ after $p$ time-steps. By calculating such statistics for various ECA rules, we establish a quantitative connection between the entropy and the qualitative Wolfram classification scheme. Class 1 rules rapidly converge to maximal entropy for stationary states ($c=1$) as $p$ increases. Class 2 rules also approach maximal entropy quickly for appropriate cycle lengths $c$, potentially requiring consideration of translations. Class 3 rules exhibit zero or low finite entropy that saturates after a short transient. Class 4 rules show finite positive entropy, similar to some Class 3 rules. This method provides a precise framework for quantifying trajectory statistics, although its exponential computational cost in $p+c$ restricts practical analysis to short trajectories. 

---
# Enhance the machine learning algorithm performance in phishing detection with keyword features 

**Authors**: Zijiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09765)  

**Abstract**: Recently, we can observe a significant increase of the phishing attacks in the Internet. In a typical phishing attack, the attacker sets up a malicious website that looks similar to the legitimate website in order to obtain the end-users' information. This may cause the leakage of the sensitive information and the financial loss for the end-users. To avoid such attacks, the early detection of these websites' URLs is vital and necessary. Previous researchers have proposed many machine learning algorithms to distinguish the phishing URLs from the legitimate ones. In this paper, we would like to enhance these machine learning algorithms from the perspective of feature selection. We propose a novel method to incorporate the keyword features with the traditional features. This method is applied on multiple traditional machine learning algorithms and the experimental results have shown this method is useful and effective. On average, this method can reduce the classification error by 30% for the large dataset. Moreover, its enhancement is more significant for the small dataset. In addition, this method extracts the information from the URL and does not rely on the additional information provided by the third-part service. The best result for the machine learning algorithm using our proposed method has achieved the accuracy of 99.68%. 

---
# NEUBORN: The Neurodevelopmental Evolution framework Using BiOmechanical RemodelliNg 

**Authors**: Nashira Baena, Mariana da Silva, Irina Grigorescu, Aakash Saboo, Saga Masui, Jaques-Donald Tournier, Emma C. Robinson  

**Link**: [PDF](https://arxiv.org/pdf/2508.09757)  

**Abstract**: Understanding individual cortical development is essential for identifying deviations linked to neurodevelopmental disorders. However, current normative modelling frameworks struggle to capture fine-scale anatomical details due to their reliance on modelling data within a population-average reference space. Here, we present a novel framework for learning individual growth trajectories from biomechanically constrained, longitudinal, diffeomorphic image registration, implemented via a hierarchical network architecture. Trained on neonatal MRI data from the Developing Human Connectome Project, the method improves the biological plausibility of warps, generating growth trajectories that better follow population-level trends while generating smoother warps, with fewer negative Jacobians, relative to state-of-the-art baselines. The resulting subject-specific deformations provide interpretable, biologically grounded mappings of development. This framework opens new possibilities for predictive modeling of brain maturation and early identification of malformations of cortical development. 

---
# Region-to-Region: Enhancing Generative Image Harmonization with Adaptive Regional Injection 

**Authors**: Zhiqiu Zhang, Dongqi Fan, Mingjie Wang, Qiang Tang, Jian Yang, Zili Yi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09746)  

**Abstract**: The goal of image harmonization is to adjust the foreground in a composite image to achieve visual consistency with the background. Recently, latent diffusion model (LDM) are applied for harmonization, achieving remarkable results. However, LDM-based harmonization faces challenges in detail preservation and limited harmonization ability. Additionally, current synthetic datasets rely on color transfer, which lacks local variations and fails to capture complex real-world lighting conditions. To enhance harmonization capabilities, we propose the Region-to-Region transformation. By injecting information from appropriate regions into the foreground, this approach preserves original details while achieving image harmonization or, conversely, generating new composite data. From this perspective, We propose a novel model R2R. Specifically, we design Clear-VAE to preserve high-frequency details in the foreground using Adaptive Filter while eliminating disharmonious elements. To further enhance harmonization, we introduce the Harmony Controller with Mask-aware Adaptive Channel Attention (MACA), which dynamically adjusts the foreground based on the channel importance of both foreground and background regions. To address the limitation of existing datasets, we propose Random Poisson Blending, which transfers color and lighting information from a suitable region to the foreground, thereby generating more diverse and challenging synthetic images. Using this method, we construct a new synthetic dataset, RPHarmony. Experiments demonstrate the superiority of our method over other methods in both quantitative metrics and visual harmony. Moreover, our dataset helps the model generate more realistic images in real examples. Our code, dataset, and model weights have all been released for open access. 

---
# Improving ARDS Diagnosis Through Context-Aware Concept Bottleneck Models 

**Authors**: Anish Narain, Ritam Majumdar, Nikita Narayanan, Dominic Marshall, Sonali Parbhoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.09719)  

**Abstract**: Large, publicly available clinical datasets have emerged as a novel resource for understanding disease heterogeneity and to explore personalization of therapy. These datasets are derived from data not originally collected for research purposes and, as a result, are often incomplete and lack critical labels. Many AI tools have been developed to retrospectively label these datasets, such as by performing disease classification; however, they often suffer from limited interpretability. Previous work has attempted to explain predictions using Concept Bottleneck Models (CBMs), which learn interpretable concepts that map to higher-level clinical ideas, facilitating human evaluation. However, these models often experience performance limitations when the concepts fail to adequately explain or characterize the task. We use the identification of Acute Respiratory Distress Syndrome (ARDS) as a challenging test case to demonstrate the value of incorporating contextual information from clinical notes to improve CBM performance. Our approach leverages a Large Language Model (LLM) to process clinical notes and generate additional concepts, resulting in a 10% performance gain over existing methods. Additionally, it facilitates the learning of more comprehensive concepts, thereby reducing the risk of information leakage and reliance on spurious shortcuts, thus improving the characterization of ARDS. 

---
# Evaluating the Role of Large Language Models in Legal Practice in India 

**Authors**: Rahul Hemrajani  

**Link**: [PDF](https://arxiv.org/pdf/2508.09713)  

**Abstract**: The integration of Artificial Intelligence(AI) into the legal profession raises significant questions about the capacity of Large Language Models(LLM) to perform key legal tasks. In this paper, I empirically evaluate how well LLMs, such as GPT, Claude, and Llama, perform key legal tasks in the Indian context, including issue spotting, legal drafting, advice, research, and reasoning. Through a survey experiment, I compare outputs from LLMs with those of a junior lawyer, with advanced law students rating the work on helpfulness, accuracy, and comprehensiveness. LLMs excel in drafting and issue spotting, often matching or surpassing human work. However, they struggle with specialised legal research, frequently generating hallucinations, factually incorrect or fabricated outputs. I conclude that while LLMs can augment certain legal tasks, human expertise remains essential for nuanced reasoning and the precise application of law. 

---
# Surg-InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision 

**Authors**: Gerardo Loza, Junlei Hu, Dominic Jones, Sharib Ali, Pietro Valdastri  

**Link**: [PDF](https://arxiv.org/pdf/2508.09681)  

**Abstract**: We proposed a novel test-time optimisation (TTO) approach framed by a NeRF-based architecture for long-term 3D point tracking. Most current methods in point tracking struggle to obtain consistent motion or are limited to 2D motion. TTO approaches frame the solution for long-term tracking as optimising a function that aggregates correspondences from other specialised state-of-the-art methods. Unlike the state-of-the-art on TTO, we propose parametrising such a function with our new invertible Neural Radiance Field (InvNeRF) architecture to perform both 2D and 3D tracking in surgical scenarios. Our approach allows us to exploit the advantages of a rendering-based approach by supervising the reprojection of pixel correspondences. It adapts strategies from recent rendering-based methods to obtain a bidirectional deformable-canonical mapping, to efficiently handle a defined workspace, and to guide the rays' density. It also presents our multi-scale HexPlanes for fast inference and a new algorithm for efficient pixel sampling and convergence criteria. We present results in the STIR and SCARE datasets, for evaluating point tracking and testing the integration of kinematic data in our pipeline, respectively. In 2D point tracking, our approach surpasses the precision and accuracy of the TTO state-of-the-art methods by nearly 50% on average precision, while competing with other approaches. In 3D point tracking, this is the first TTO approach, surpassing feed-forward methods while incorporating the benefits of a deformable NeRF-based reconstruction. 

---
# Anomaly Detection for IoT Global Connectivity 

**Authors**: Jesus Omaña Iglesias, Carlos Segura Perales, Stefan Geißler, Diego Perino, Andra Lutu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09660)  

**Abstract**: Internet of Things (IoT) application providers rely on Mobile Network Operators (MNOs) and roaming infrastructures to deliver their services globally. In this complex ecosystem, where the end-to-end communication path traverses multiple entities, it has become increasingly challenging to guarantee communication availability and reliability. Further, most platform operators use a reactive approach to communication issues, responding to user complaints only after incidents have become severe, compromising service quality. This paper presents our experience in the design and deployment of ANCHOR -- an unsupervised anomaly detection solution for the IoT connectivity service of a large global roaming platform. ANCHOR assists engineers by filtering vast amounts of data to identify potential problematic clients (i.e., those with connectivity issues affecting several of their IoT devices), enabling proactive issue resolution before the service is critically impacted. We first describe the IoT service, infrastructure, and network visibility of the IoT connectivity provider we operate. Second, we describe the main challenges and operational requirements for designing an unsupervised anomaly detection solution on this platform. Following these guidelines, we propose different statistical rules, and machine- and deep-learning models for IoT verticals anomaly detection based on passive signaling traffic. We describe the steps we followed working with the operational teams on the design and evaluation of our solution on the operational platform, and report an evaluation on operational IoT customers. 

---
# On Negative-aware Preference Optimization for Recommendation 

**Authors**: Chenlu Ding, Daoxuan Liu, Jiancan Wu, Xingyu Hu, Junkang Wu, Haitao Wang, Yongkang Wang, Xingxing Wang, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09653)  

**Abstract**: Recommendation systems leverage user interaction data to suggest relevant items while filtering out irrelevant (negative) ones. The rise of large language models (LLMs) has garnered increasing attention for their potential in recommendation tasks. However, existing methods for optimizing LLM-based recommenders face challenges in effectively utilizing negative samples. Simply integrating large numbers of negative samples can improve ranking accuracy and mitigate popularity bias but often leads to increased computational overhead and memory costs. Additionally, current approaches fail to account for the varying informativeness of negative samples, leading to suboptimal optimization performance. To address these issues, we propose NAPO (\textbf{N}egative-\textbf{A}ware \textbf{P}reference \textbf{O}ptimization), an enhanced framework for preference optimization in LLM-based recommendation. NAPO introduces two key innovations: (1) in-batch negative sharing, which expands the pool of negative samples without additional memory overhead, and (2) dynamic reward margin adjustment, which adapts model updates based on the confidence of negative samples. Extensive experiments on three public datasets demonstrate that NAPO outperforms existing methods in both recommendation accuracy and popularity bias reduction. 

---
# Demystifying the Role of Rule-based Detection in AI Systems for Windows Malware Detection 

**Authors**: Andrea Ponte, Luca Demetrio, Luca Oneto, Ivan Tesfai Ogbu, Battista Biggio, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2508.09652)  

**Abstract**: Malware detection increasingly relies on AI systems that integrate signature-based detection with machine learning. However, these components are typically developed and combined in isolation, missing opportunities to reduce data complexity and strengthen defenses against adversarial EXEmples, carefully crafted programs designed to evade detection. Hence, in this work we investigate the influence that signature-based detection exerts on model training, when they are included inside the training pipeline. Specifically, we compare models trained on a comprehensive dataset with an AI system whose machine learning component is trained solely on samples not already flagged by signatures. Our results demonstrate improved robustness to both adversarial EXEmples and temporal data drift, although this comes at the cost of a fixed lower bound on false positives, driven by suboptimal rule selection. We conclude by discussing these limitations and outlining how future research could extend AI-based malware detection to include dynamic analysis, thereby further enhancing system resilience. 

---
# A Close Reading Approach to Gender Narrative Biases in AI-Generated Stories 

**Authors**: Daniel Raffini, Agnese Macori, Marco Angelini, Tiziana Catarci  

**Link**: [PDF](https://arxiv.org/pdf/2508.09651)  

**Abstract**: The paper explores the study of gender-based narrative biases in stories generated by ChatGPT, Gemini, and Claude. The prompt design draws on Propp's character classifications and Freytag's narrative structure. The stories are analyzed through a close reading approach, with particular attention to adherence to the prompt, gender distribution of characters, physical and psychological descriptions, actions, and finally, plot development and character relationships. The results reveal the persistence of biases - especially implicit ones - in the generated stories and highlight the importance of assessing biases at multiple levels using an interpretative approach. 

---
# Preacher: Paper-to-Video Agentic System 

**Authors**: Jingwei Liu, Ling Yang, Hao Luo, Fan Wang Hongyan Li, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09632)  

**Abstract**: The paper-to-video task converts a research paper into a structured video abstract, distilling key concepts, methods, and conclusions into an accessible, well-organized format. While state-of-the-art video generation models demonstrate potential, they are constrained by limited context windows, rigid video duration constraints, limited stylistic diversity, and an inability to represent domain-specific knowledge. To address these limitations, we introduce Preacher, the first paper-to-video agentic system. Preacher employs a top-down approach to decompose, summarize, and reformulate the paper, followed by bottom-up video generation, synthesizing diverse video segments into a coherent abstract. To align cross-modal representations, we define key scenes and introduce a Progressive Chain of Thought (P-CoT) for granular, iterative planning. Preacher successfully generates high-quality video abstracts across five research fields, demonstrating expertise beyond current video generation models. Code will be released at: this https URL 

---
# AmbiGraph-Eval: Can LLMs Effectively Handle Ambiguous Graph Queries? 

**Authors**: Yuchen Tian, Kaixin Li, Hao Chen, Ziyang Luo, Hongzhan Lin, Sebastian Schelter, Lun Du, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.09631)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong capabilities in translating natural language into database queries, especially when dealing with complex graph-structured data. However, real-world queries often contain inherent ambiguities, and the interconnected nature of graph structures can amplify these challenges, leading to unintended or incorrect query results. To systematically evaluate LLMs on this front, we propose a taxonomy of graph-query ambiguities, comprising three primary types: Attribute Ambiguity, Relationship Ambiguity, and Attribute-Relationship Ambiguity, each subdivided into Same-Entity and Cross-Entity scenarios. We introduce AmbiGraph-Eval, a novel benchmark of real-world ambiguous queries paired with expert-verified graph query answers. Evaluating 9 representative LLMs shows that even top models struggle with ambiguous graph queries. Our findings reveal a critical gap in ambiguity handling and motivate future work on specialized resolution techniques. 

---
# TimeMKG: Knowledge-Infused Causal Reasoning for Multivariate Time Series Modeling 

**Authors**: Yifei Sun, Junming Liu, Ding Wang, Yirong Chen, Xuefeng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09630)  

**Abstract**: Multivariate time series data typically comprises two distinct modalities: variable semantics and sampled numerical observations. Traditional time series models treat variables as anonymous statistical signals, overlooking the rich semantic information embedded in variable names and data descriptions. However, these textual descriptors often encode critical domain knowledge that is essential for robust and interpretable modeling. Here we present TimeMKG, a multimodal causal reasoning framework that elevates time series modeling from low-level signal processing to knowledge informed inference. TimeMKG employs large language models to interpret variable semantics and constructs structured Multivariate Knowledge Graphs that capture inter-variable relationships. A dual-modality encoder separately models the semantic prompts, generated from knowledge graph triplets, and the statistical patterns from historical time series. Cross-modality attention aligns and fuses these representations at the variable level, injecting causal priors into downstream tasks such as forecasting and classification, providing explicit and interpretable priors to guide model reasoning. The experiment in diverse datasets demonstrates that incorporating variable-level knowledge significantly improves both predictive performance and generalization. 

---
# Goal Discovery with Causal Capacity for Efficient Reinforcement Learning 

**Authors**: Yan Yu, Yaodong Yang, Zhengbo Lu, Chengdong Ma, Wengang Zhou, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09624)  

**Abstract**: Causal inference is crucial for humans to explore the world, which can be modeled to enable an agent to efficiently explore the environment in reinforcement learning. Existing research indicates that establishing the causality between action and state transition will enhance an agent to reason how a policy affects its future trajectory, thereby promoting directed exploration. However, it is challenging to measure the causality due to its intractability in the vast state-action space of complex scenarios. In this paper, we propose a novel Goal Discovery with Causal Capacity (GDCC) framework for efficient environment exploration. Specifically, we first derive a measurement of causality in state space, \emph{i.e.,} causal capacity, which represents the highest influence of an agent's behavior on future trajectories. After that, we present a Monte Carlo based method to identify critical points in discrete state space and further optimize this method for continuous high-dimensional environments. Those critical points are used to uncover where the agent makes important decisions in the environment, which are then regarded as our subgoals to guide the agent to make exploration more purposefully and efficiently. Empirical results from multi-objective tasks demonstrate that states with high causal capacity align with our expected subgoals, and our GDCC achieves significant success rate improvements compared to baselines. 

---
# Interpretable Robot Control via Structured Behavior Trees and Large Language Models 

**Authors**: Ingrid Maéva Chekam, Ines Pastor-Martinez, Ali Tourani, Jose Andres Millan-Romera, Laura Ribeiro, Pedro Miguel Bastos Soares, Holger Voos, Jose Luis Sanchez-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2508.09621)  

**Abstract**: As intelligent robots become more integrated into human environments, there is a growing need for intuitive and reliable Human-Robot Interaction (HRI) interfaces that are adaptable and more natural to interact with. Traditional robot control methods often require users to adapt to interfaces or memorize predefined commands, limiting usability in dynamic, unstructured environments. This paper presents a novel framework that bridges natural language understanding and robotic execution by combining Large Language Models (LLMs) with Behavior Trees. This integration enables robots to interpret natural language instructions given by users and translate them into executable actions by activating domain-specific plugins. The system supports scalable and modular integration, with a primary focus on perception-based functionalities, such as person tracking and hand gesture recognition. To evaluate the system, a series of real-world experiments was conducted across diverse environments. Experimental results demonstrate that the proposed approach is practical in real-world scenarios, with an average cognition-to-execution accuracy of approximately 94%, making a significant contribution to HRI systems and robots. The complete source code of the framework is publicly available at this https URL. 

---
# MInDI-3D: Iterative Deep Learning in 3D for Sparse-view Cone Beam Computed Tomography 

**Authors**: Daniel Barco, Marc Stadelmann, Martin Oswald, Ivo Herzig, Lukas Lichtensteiger, Pascal Paysan, Igor Peterlik, Michal Walczak, Bjoern Menze, Frank-Peter Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2508.09616)  

**Abstract**: We present MInDI-3D (Medical Inversion by Direct Iteration in 3D), the first 3D conditional diffusion-based model for real-world sparse-view Cone Beam Computed Tomography (CBCT) artefact removal, aiming to reduce imaging radiation exposure. A key contribution is extending the "InDI" concept from 2D to a full 3D volumetric approach for medical images, implementing an iterative denoising process that refines the CBCT volume directly from sparse-view input. A further contribution is the generation of a large pseudo-CBCT dataset (16,182) from chest CT volumes of the CT-RATE public dataset to robustly train MInDI-3D. We performed a comprehensive evaluation, including quantitative metrics, scalability analysis, generalisation tests, and a clinical assessment by 11 clinicians. Our results show MInDI-3D's effectiveness, achieving a 12.96 (6.10) dB PSNR gain over uncorrected scans with only 50 projections on the CT-RATE pseudo-CBCT (independent real-world) test set and enabling an 8x reduction in imaging radiation exposure. We demonstrate its scalability by showing that performance improves with more training data. Importantly, MInDI-3D matches the performance of a 3D U-Net on real-world scans from 16 cancer patients across distortion and task-based metrics. It also generalises to new CBCT scanner geometries. Clinicians rated our model as sufficient for patient positioning across all anatomical sites and found it preserved lung tumour boundaries well. 

---
# How Persuasive Could LLMs Be? A First Study Combining Linguistic-Rhetorical Analysis and User Experiments 

**Authors**: Daniel Raffini, Agnese Macori, Lorenzo Porcaro, Tiziana Catarci, Marco Angelini  

**Link**: [PDF](https://arxiv.org/pdf/2508.09614)  

**Abstract**: This study examines the rhetorical and linguistic features of argumentative texts generated by ChatGPT on ethically nuanced topics and investigates their persuasive impact on human this http URL a user study involving 62 participants and pre-post interaction surveys, the paper analyzes how exposure to AI-generated arguments affects opinion change and user perception. A linguistic and rhetorical analysis of the generated texts reveals a consistent argumentative macrostructure, reliance on formulaic expressions, and limited stylistic richness. While ChatGPT demonstrates proficiency in constructing coherent argumentative texts, its persuasive efficacy appears constrained, particularly on topics involving ethical this http URL study finds that while participants often acknowledge the benefits highlighted by ChatGPT, ethical concerns tend to persist or even intensify post-interaction. The results also demonstrate a variation depending on the topic. These findings highlight new insights on AI-generated persuasion in ethically sensitive domains and are a basis for future research. 

---
# A Lightweight Learned Cardinality Estimation Model 

**Authors**: Yaoyu Zhu, Jintao Zhang, Guoliang Li, Jianhua Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09602)  

**Abstract**: Cardinality estimation is a fundamental task in database management systems, aiming to predict query results accurately without executing the queries. However, existing techniques either achieve low estimation accuracy or incur high inference latency. Simultaneously achieving high speed and accuracy becomes critical for the cardinality estimation problem. In this paper, we propose a novel data-driven approach called CoDe (Covering with Decompositions) to address this problem. CoDe employs the concept of covering design, which divides the table into multiple smaller, overlapping segments. For each segment, CoDe utilizes tensor decomposition to accurately model its data distribution. Moreover, CoDe introduces innovative algorithms to select the best-fitting distributions for each query, combining them to estimate the final result. By employing multiple models to approximate distributions, CoDe excels in effectively modeling discrete distributions and ensuring computational efficiency. Notably, experimental results show that our method represents a significant advancement in cardinality estimation, achieving state-of-the-art levels of both estimation accuracy and inference efficiency. Across various datasets, CoDe achieves absolute accuracy in estimating more than half of the queries. 

---
# Hierarchical Brain Structure Modeling for Predicting Genotype of Glioma 

**Authors**: Haotian Tang, Jianwei Chen, Xinrui Tang, Yunjia Wu, Zhengyang Miao, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09593)  

**Abstract**: Isocitrate DeHydrogenase (IDH) mutation status is a crucial biomarker for glioma prognosis. However, current prediction methods are limited by the low availability and noise of functional MRI. Structural and morphological connectomes offer a non-invasive alternative, yet existing approaches often ignore the brain's hierarchical organisation and multiscale interactions. To address this, we propose Hi-SMGNN, a hierarchical framework that integrates structural and morphological connectomes from regional to modular levels. It features a multimodal interaction module with a Siamese network and cross-modal attention, a multiscale feature fusion mechanism for reducing redundancy, and a personalised modular partitioning strategy to enhance individual specificity and interpretability. Experiments on the UCSF-PDGM dataset demonstrate that Hi-SMGNN outperforms baseline and state-of-the-art models, showing improved robustness and effectiveness in IDH mutation prediction. 

---
# CaRoBio: 3D Cable Routing with a Bio-inspired Gripper Fingernail 

**Authors**: Jiahui Zuo, Boyang Zhang, Fumin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09558)  

**Abstract**: The manipulation of deformable linear flexures has a wide range of applications in industry, such as cable routing in automotive manufacturing and textile production. Cable routing, as a complex multi-stage robot manipulation scenario, is a challenging task for robot automation. Common parallel two-finger grippers have the risk of over-squeezing and over-tension when grasping and guiding cables. In this paper, a novel eagle-inspired fingernail is designed and mounted on the gripper fingers, which helps with cable grasping on planar surfaces and in-hand cable guiding operations. Then we present a single-grasp end-to-end 3D cable routing framework utilizing the proposed fingernails, instead of the common pick-and-place strategy. Continuous control is achieved to efficiently manipulate cables through vision-based state estimation of task configurations and offline trajectory planning based on motion primitives. We evaluate the effectiveness of the proposed framework with a variety of cables and channel slots, significantly outperforming the pick-and-place manipulation process under equivalent perceptual conditions. Our reconfigurable task setting and the proposed framework provide a reference for future cable routing manipulations in 3D space. 

---
# GoViG: Goal-Conditioned Visual Navigation Instruction Generation 

**Authors**: Fengyi Wu, Yifei Dong, Zhi-Qi Cheng, Yilong Dai, Guangyu Chen, Hang Wang, Qi Dai, Alexander G. Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.09547)  

**Abstract**: We introduce Goal-Conditioned Visual Navigation Instruction Generation (GoViG), a new task that aims to autonomously generate precise and contextually coherent navigation instructions solely from egocentric visual observations of initial and goal states. Unlike conventional approaches that rely on structured inputs such as semantic annotations or environmental maps, GoViG exclusively leverages raw egocentric visual data, substantially improving its adaptability to unseen and unstructured environments. Our method addresses this task by decomposing it into two interconnected subtasks: (1) visual forecasting, which predicts intermediate visual states bridging the initial and goal views; and (2) instruction generation, which synthesizes linguistically coherent instructions grounded in both observed and anticipated visuals. These subtasks are integrated within an autoregressive multimodal large language model trained with tailored objectives to ensure spatial accuracy and linguistic clarity. Furthermore, we introduce two complementary multimodal reasoning strategies, one-pass and interleaved reasoning, to mimic incremental human cognitive processes during navigation. To evaluate our method, we propose the R2R-Goal dataset, combining diverse synthetic and real-world trajectories. Empirical results demonstrate significant improvements over state-of-the-art methods, achieving superior BLEU-4 and CIDEr scores along with robust cross-domain generalization. 

---
# Your Coding Intent is Secretly in the Context and You Should Deliberately Infer It Before Completion 

**Authors**: Yanzhou Li, Tianlin Li, Yiran Zhang, Shangqing Liu, Aishan Liu, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09537)  

**Abstract**: Large Language Models (LLMs) are increasingly used for function completion in repository-scale codebases. Prior studies demonstrate that when explicit instructions--such as docstrings--are provided, these models can generate highly accurate implementations. However, in real-world repositories, such annotations are frequently absent, and performance drops substantially without them. To address this gap, we frame the task as a three-stage process. The first stage focuses on intent inference, where the model analyzes the code preceding the target function to uncover cues about the desired functionality. Such preceding context often encodes subtle but critical information, and we design a reasoning-based prompting framework to guide the LLM through step-by-step extraction and synthesis of these signals before any code is generated. The second stage introduces an optional interactive refinement mechanism to handle cases where preceding context alone is insufficient for intent recovery. In this stage, the model proposes a small set of candidate intentions, enabling the developer to select or edit them so that the inferred intent closely matches the actual requirement. Finally, in the third stage, the LLM generates the target function conditioned on the finalized intent. To support this pipeline, we curate a dataset of 40,000 examples annotated with intermediate reasoning traces and corresponding docstrings. Extensive experiments on DevEval and ComplexCodeEval show that our approach consistently boosts multiple LLMs, achieving over 20\% relative gains in both reference-based and execution-based metrics, with the interactive refinement stage delivering additional improvements beyond these gains. 

---
# AI Blob! LLM-Driven Recontextualization of Italian Television Archives 

**Authors**: Roberto Balestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.09535)  

**Abstract**: This paper introduces AI Blob!, an experimental system designed to explore the potential of semantic cataloging and Large Language Models (LLMs) for the retrieval and recontextualization of archival television footage. Drawing methodological inspiration from Italian television programs such as Blob (RAI Tre, 1989-), AI Blob! integrates automatic speech recognition (ASR), semantic embeddings, and retrieval-augmented generation (RAG) to organize and reinterpret archival content. The system processes a curated dataset of 1,547 Italian television videos by transcribing audio, segmenting it into sentence-level units, and embedding these segments into a vector database for semantic querying. Upon user input of a thematic prompt, the LLM generates a range of linguistically and conceptually related queries, guiding the retrieval and recombination of audiovisual fragments. These fragments are algorithmically selected and structured into narrative sequences producing montages that emulate editorial practices of ironic juxtaposition and thematic coherence. By foregrounding dynamic, content-aware retrieval over static metadata schemas, AI Blob! demonstrates how semantic technologies can facilitate new approaches to archival engagement, enabling novel forms of automated narrative construction and cultural analysis. The project contributes to ongoing debates in media historiography and AI-driven archival research, offering both a conceptual framework and a publicly available dataset to support further interdisciplinary experimentation. 

---
# COXNet: Cross-Layer Fusion with Adaptive Alignment and Scale Integration for RGBT Tiny Object Detection 

**Authors**: Peiran Peng, Tingfa Xu, Liqiang Song, Mengqi Zhu, Yuqiang Fang, Jianan Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09533)  

**Abstract**: Detecting tiny objects in multimodal Red-Green-Blue-Thermal (RGBT) imagery is a critical challenge in computer vision, particularly in surveillance, search and rescue, and autonomous navigation. Drone-based scenarios exacerbate these challenges due to spatial misalignment, low-light conditions, occlusion, and cluttered backgrounds. Current methods struggle to leverage the complementary information between visible and thermal modalities effectively. We propose COXNet, a novel framework for RGBT tiny object detection, addressing these issues through three core innovations: i) the Cross-Layer Fusion Module, fusing high-level visible and low-level thermal features for enhanced semantic and spatial accuracy; ii) the Dynamic Alignment and Scale Refinement module, correcting cross-modal spatial misalignments and preserving multi-scale features; and iii) an optimized label assignment strategy using the GeoShape Similarity Measure for better localization. COXNet achieves a 3.32\% mAP$_{50}$ improvement on the RGBTDronePerson dataset over state-of-the-art methods, demonstrating its effectiveness for robust detection in complex environments. 

---
# Decentralized Rank Scheduling for Energy-Constrained Multi-Task Federated Fine-Tuning in Edge-Assisted IoV Networks 

**Authors**: Bokeng Zheng, Jianqiang Zhong, Jiayi Liu, Xiaoxi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09532)  

**Abstract**: Federated fine-tuning has emerged as a promising approach for adapting foundation models (FMs) to diverse downstream tasks in edge environments. In Internet of Vehicles (IoV) systems, enabling efficient and low-latency multi-task adaptation is particularly challenging due to client mobility, heterogeneous resources, and intermittent connectivity. This paper proposes a hierarchical federated fine-tuning framework that coordinates roadside units (RSUs) and vehicles to support resource-aware and mobility-resilient learning across dynamic IoV scenarios. Leveraging Low-Rank Adaptation (LoRA), we introduce a decentralized, energy-aware rank adaptation mechanism formulated as a constrained multi-armed bandit problem. A novel UCB-DUAL algorithm is developed to enable adaptive exploration under per-task energy budgets, achieving provable sublinear regret. To evaluate our method, we construct a large-scale IoV simulator based on real-world trajectories, capturing dynamic participation, RSU handoffs, and communication variability. Extensive experiments show that our approach achieves the best accuracy-efficiency trade-off among all baselines, reducing latency by over 24\% and improving average accuracy by more than 2.5\%. 

---
# Generation of Indian Sign Language Letters, Numbers, and Words 

**Authors**: Ajeet Kumar Yadav, Nishant Kumar, Rathna G N  

**Link**: [PDF](https://arxiv.org/pdf/2508.09522)  

**Abstract**: Sign language, which contains hand movements, facial expressions and bodily gestures, is a significant medium for communicating with hard-of-hearing people. A well-trained sign language community communicates easily, but those who don't know sign language face significant challenges. Recognition and generation are basic communication methods between hearing and hard-of-hearing individuals. Despite progress in recognition, sign language generation still needs to be explored. The Progressive Growing of Generative Adversarial Network (ProGAN) excels at producing high-quality images, while the Self-Attention Generative Adversarial Network (SAGAN) generates feature-rich images at medium resolutions. Balancing resolution and detail is crucial for sign language image generation. We are developing a Generative Adversarial Network (GAN) variant that combines both models to generate feature-rich, high-resolution, and class-conditional sign language images. Our modified Attention-based model generates high-quality images of Indian Sign Language letters, numbers, and words, outperforming the traditional ProGAN in Inception Score (IS) and Fréchet Inception Distance (FID), with improvements of 3.2 and 30.12, respectively. Additionally, we are publishing a large dataset incorporating high-quality images of Indian Sign Language alphabets, numbers, and 129 words. 

---
# COMPEER: Controllable Empathetic Reinforcement Reasoning for Emotional Support Conversation 

**Authors**: Yunxiao Wang, Meng Liu, Wenqi Liu, Kaiyu Jiang, Bin Wen, Fan Yang, Tingting Gao, Guorui Zhou, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.09521)  

**Abstract**: Emotional support conversations are crucial for promoting emotional well-being, yet current models often lack deep empathetic reasoning grounded in psychological principles. To address this, we propose controllable empathetic reasoning, which combines natural language reasoning with structured psychological steps. We construct a fine-grained dataset annotated with reasoning correctness and response preferences to enable this capability. To further enhance training, we employ reinforcement learning with a unified process-outcome reward model that delivers precise feedback. To mitigate response repetitiveness from entropy collapse, we introduce personality-based dialogue rewriting and a redundancy-aware reward reweighting strategy. Our approach significantly improves model's emotional support ability, advancing the development of empathetic, human-like support systems. 

---
# SMART-OC: A Real-time Time-risk Optimal Replanning Algorithm for Dynamic Obstacles and Spatio-temporally Varying Currents 

**Authors**: Reema Raval, Shalabh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.09508)  

**Abstract**: Typical marine environments are highly complex with spatio-temporally varying currents and dynamic obstacles, presenting significant challenges to Unmanned Surface Vehicles (USVs) for safe and efficient navigation. Thus, the USVs need to continuously adapt their paths with real-time information to avoid collisions and follow the path of least resistance to the goal via exploiting ocean currents. In this regard, we introduce a novel algorithm, called Self-Morphing Adaptive Replanning Tree for dynamic Obstacles and Currents (SMART-OC), that facilitates real-time time-risk optimal replanning in dynamic environments. SMART-OC integrates the obstacle risks along a path with the time cost to reach the goal to find the time-risk optimal path. The effectiveness of SMART-OC is validated by simulation experiments, which demonstrate that the USV performs fast replannings to avoid dynamic obstacles and exploit ocean currents to successfully reach the goal. 

---
# Verify Distributed Deep Learning Model Implementation Refinement with Iterative Relation Inference 

**Authors**: Zhanghan Wang, Ding Ding, Hang Zhu, Haibin Lin, Aurojit Panda  

**Link**: [PDF](https://arxiv.org/pdf/2508.09505)  

**Abstract**: Distributed machine learning training and inference is common today because today's large models require more memory and compute than can be provided by a single GPU. Distributed models are generally produced by programmers who take a sequential model specification and apply several distribution strategies to distribute state and computation across GPUs. Unfortunately, bugs can be introduced in the process, and a distributed model implementation's outputs might differ from the sequential model's outputs. In this paper, we describe an approach to statically identify such bugs by checking model refinement, that is, can the sequential model's outputs be reconstructed from the distributed model's outputs? Our approach, implemented in GraphGuard, uses iterative rewriting to prove model refinement. Our approach can scale to today's large models and deployments: we evaluate it using GPT and Llama-3. Further, it provides actionable output that aids in bug localization. 

---
# From Ranking to Selection: A Simple but Efficient Dynamic Passage Selector for Retrieval Augmented Generation 

**Authors**: Siyuan Meng, Junming Liu, Yirong Chen, Song Mao, Pinlong Cai, Guohang Yan, Botian Shi, Ding Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09497)  

**Abstract**: Retrieval-augmented generation (RAG) systems are often bottlenecked by their reranking modules, which typically score passages independently and select a fixed Top-K size. This approach struggles with complex multi-hop queries that require synthesizing evidence across multiple documents, creating a trade-off where small K values omit crucial information and large K values introduce noise. To address this, we introduce the Dynamic Passage Selector (DPS), a novel reranking framework that treats passage selection as a supervised learning problem. Unlike traditional point-wise or list-wise methods, DPS is fine-tuned to capture inter-passage dependencies and dynamically select the most relevant set of passages for generation. As a seamless plug-and-play module, DPS requires no modifications to the standard RAG pipeline. Comprehensive evaluations on five benchmarks show that DPS consistently outperforms state-of-the-art rerankers and fine-tuning methods. Notably, on the challenging MuSiQue dataset, DPS improves the F1-score by 30.06% and 15.4% over strong baselines like Qwen3-reranker and RankingGPT, respectively. Our results demonstrate that by enabling adaptive evidence selection, DPS substantially enhances reasoning capabilities in complex RAG scenarios. 

---
# Learning Facts at Scale with Active Reading 

**Authors**: Jessy Lin, Vincent-Pierre Berges, Xilun Chen, Wen-Tau Yih, Gargi Ghosh, Barlas Oğuz  

**Link**: [PDF](https://arxiv.org/pdf/2508.09494)  

**Abstract**: LLMs are known to store vast amounts of knowledge in their parametric memory. However, learning and recalling facts from this memory is known to be unreliable, depending largely on the prevalence of particular facts in the training data and other factors which are poorly understood. Practitioners are lacking tools which will allow them to ensure that the models learn a given body of knowledge reliably and consistently. To this end, we propose Active Reading: a framework where we train models to study a given set of material with self-generated learning strategies. First, we demonstrate models trained with Active Reading on expert domains absorb significantly more knowledge than vanilla finetuning and other data augmentations. We train expert 8B models that achieve 66% on a Wikipedia-grounded subset of SimpleQA (+313% relative over vanilla finetuning) and 26% on FinanceBench (+160% relative over vanilla finetuning) by applying Active Reading to the source documents for each benchmark. Finally, we show that Active Reading can be utilized at pre-training scale to build more factual models. As a demonstration of this, we release Meta WikiExpert-8B, a Wikipedia-expert model trained on 1 trillion generated tokens, which outcompetes models with hundreds of billions of parameters on factual QA. 

---
# Large-Small Model Collaborative Framework for Federated Continual Learning 

**Authors**: Hao Yu, Xin Yang, Boyang Fan, Xuemei Cao, Hanlin Gu, Lixin Fan, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09489)  

**Abstract**: Continual learning (CL) for Foundation Models (FMs) is an essential yet underexplored challenge, especially in Federated Continual Learning (FCL), where each client learns from a private, evolving task stream under strict data and communication constraints. Despite their powerful generalization abilities, FMs often exhibit suboptimal performance on local downstream tasks, as they are unable to utilize private local data. Furthermore, enabling FMs to learn new tasks without forgetting prior knowledge is inherently a challenging problem, primarily due to their immense parameter count and high model complexity. In contrast, small models can be trained locally under resource-constrained conditions and benefit from more mature CL techniques. To bridge the gap between small models and FMs, we propose the first collaborative framework in FCL, where lightweight local models act as a dynamic bridge, continually adapting to new tasks while enhancing the utility of the large model. Two novel components are also included: Small Model Continual Fine-tuning is for preventing small models from temporal forgetting; One-by-One Distillation performs personalized fusion of heterogeneous local knowledge on the server. Experimental results demonstrate its superior performance, even when clients utilize heterogeneous small models. 

---
# Episodic Memory Representation for Long-form Video Understanding 

**Authors**: Yun Wang, Long Zhang, Jingren Liu, Jiaqi Yan, Zhanjie Zhang, Jiahao Zheng, Xun Yang, Dapeng Wu, Xiangyu Chen, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09486)  

**Abstract**: Video Large Language Models (Video-LLMs) excel at general video understanding but struggle with long-form videos due to context window limits. Consequently, recent approaches focus on keyframe retrieval, condensing lengthy videos into a small set of informative frames. Despite their practicality, these methods simplify the problem to static text image matching, overlooking spatio temporal relationships crucial for capturing scene transitions and contextual continuity, and may yield redundant keyframes with limited information, diluting salient cues essential for accurate video question answering. To address these limitations, we introduce Video-EM, a training free framework inspired by the principles of human episodic memory, designed to facilitate robust and contextually grounded reasoning. Rather than treating keyframes as isolated visual entities, Video-EM explicitly models them as temporally ordered episodic events, capturing both spatial relationships and temporal dynamics necessary for accurately reconstructing the underlying narrative. Furthermore, the framework leverages chain of thought (CoT) thinking with LLMs to iteratively identify a minimal yet highly informative subset of episodic memories, enabling efficient and accurate question answering by Video-LLMs. Extensive evaluations on the Video-MME, EgoSchema, HourVideo, and LVBench benchmarks confirm the superiority of Video-EM, which achieves highly competitive results with performance gains of 4-9 percent over respective baselines while utilizing fewer frames. 

---
# NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility Alignment in LLMs 

**Authors**: Birong Pan, Mayi Xu, Qiankun Pi, Jianhao Chen, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.09473)  

**Abstract**: Ensuring robust safety alignment while preserving utility is critical for the reliable deployment of Large Language Models (LLMs). However, current techniques fundamentally suffer from intertwined deficiencies: insufficient robustness against malicious attacks, frequent refusal of benign queries, degradation in generated text quality and general task performance--the former two reflecting deficits in robust safety and the latter constituting utility impairment. We trace these limitations to the coarse-grained layer-wise interventions in existing methods. To resolve this, we propose NeuronTune, a fine-grained framework that dynamically modulates sparse neurons to achieve simultaneous safety-utility optimization. Our approach first identifies safety-critical and utility-preserving neurons across all layers via attribution, then employs meta-learning to adaptively amplify safety-neuron activations and suppress utility-neuron activations. Crucially, NeuronTune enables tunable adjustment of intervention scope via neuron-count thresholds, supporting flexible adaptation to security-critical or utility-priority scenarios. Extensive experimental results demonstrate that our method significantly outperforms existing state-of-the-art technologies, achieving superior model safety while maintaining excellent utility. 

---
# DeepFeatIoT: Unifying Deep Learned, Randomized, and LLM Features for Enhanced IoT Time Series Sensor Data Classification in Smart Industries 

**Authors**: Muhammad Sakib Khan Inan, Kewen Liao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09468)  

**Abstract**: Internet of Things (IoT) sensors are ubiquitous technologies deployed across smart cities, industrial sites, and healthcare systems. They continuously generate time series data that enable advanced analytics and automation in industries. However, challenges such as the loss or ambiguity of sensor metadata, heterogeneity in data sources, varying sampling frequencies, inconsistent units of measurement, and irregular timestamps make raw IoT time series data difficult to interpret, undermining the effectiveness of smart systems. To address these challenges, we propose a novel deep learning model, DeepFeatIoT, which integrates learned local and global features with non-learned randomized convolutional kernel-based features and features from large language models (LLMs). This straightforward yet unique fusion of diverse learned and non-learned features significantly enhances IoT time series sensor data classification, even in scenarios with limited labeled data. Our model's effectiveness is demonstrated through its consistent and generalized performance across multiple real-world IoT sensor datasets from diverse critical application domains, outperforming state-of-the-art benchmark models. These results highlight DeepFeatIoT's potential to drive significant advancements in IoT analytics and support the development of next-generation smart systems. 

---
# Gen-AFFECT: Generation of Avatar Fine-grained Facial Expressions with Consistent identiTy 

**Authors**: Hao Yu, Rupayan Mallick, Margrit Betke, Sarah Adel Bargal  

**Link**: [PDF](https://arxiv.org/pdf/2508.09461)  

**Abstract**: Different forms of customized 2D avatars are widely used in gaming applications, virtual communication, education, and content creation. However, existing approaches often fail to capture fine-grained facial expressions and struggle to preserve identity across different expressions. We propose GEN-AFFECT, a novel framework for personalized avatar generation that generates expressive and identity-consistent avatars with a diverse set of facial expressions. Our framework proposes conditioning a multimodal diffusion transformer on an extracted identity-expression representation. This enables identity preservation and representation of a wide range of facial expressions. GEN-AFFECT additionally employs consistent attention at inference for information sharing across the set of generated expressions, enabling the generation process to maintain identity consistency over the array of generated fine-grained expressions. GEN-AFFECT demonstrates superior performance compared to previous state-of-the-art methods on the basis of the accuracy of the generated expressions, the preservation of the identity and the consistency of the target identity across an array of fine-grained facial expressions. 

---
# RelayFormer: A Unified Local-Global Attention Framework for Scalable Image and Video Manipulation Localization 

**Authors**: Wen Huang, Jiarui Yang, Tao Dai, Jiawei Li, Shaoxiong Zhan, Bin Wang, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2508.09459)  

**Abstract**: Visual manipulation localization (VML) -- across both images and videos -- is a crucial task in digital forensics that involves identifying tampered regions in visual content. However, existing methods often lack cross-modal generalization and struggle to handle high-resolution or long-duration inputs efficiently.
We propose RelayFormer, a unified and modular architecture for visual manipulation localization across images and videos. By leveraging flexible local units and a Global-Local Relay Attention (GLoRA) mechanism, it enables scalable, resolution-agnostic processing with strong generalization. Our framework integrates seamlessly with existing Transformer-based backbones, such as ViT and SegFormer, via lightweight adaptation modules that require only minimal architectural changes, ensuring compatibility without disrupting pretrained representations.
Furthermore, we design a lightweight, query-based mask decoder that supports one-shot inference across video sequences with linear complexity. Extensive experiments across multiple benchmarks demonstrate that our approach achieves state-of-the-art localization performance, setting a new baseline for scalable and modality-agnostic VML. Code is available at: this https URL. 

---
# Hallucination vs interpretation: rethinking accuracy and precision in AI-assisted data extraction for knowledge synthesis 

**Authors**: Xi Long, Christy Boscardin, Lauren A. Maggio, Joseph A. Costello, Ralph Gonzales, Rasmyah Hammoudeh, Ki Lai, Yoon Soo Park, Brian C. Gin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09458)  

**Abstract**: Knowledge syntheses (literature reviews) are essential to health professions education (HPE), consolidating findings to advance theory and practice. However, they are labor-intensive, especially during data extraction. Artificial Intelligence (AI)-assisted extraction promises efficiency but raises concerns about accuracy, making it critical to distinguish AI 'hallucinations' (fabricated content) from legitimate interpretive differences. We developed an extraction platform using large language models (LLMs) to automate data extraction and compared AI to human responses across 187 publications and 17 extraction questions from a published scoping review. AI-human, human-human, and AI-AI consistencies were measured using interrater reliability (categorical) and thematic similarity ratings (open-ended). Errors were identified by comparing extracted responses to source publications. AI was highly consistent with humans for concrete, explicitly stated questions (e.g., title, aims) and lower for questions requiring subjective interpretation or absent in text (e.g., Kirkpatrick's outcomes, study rationale). Human-human consistency was not higher than AI-human and showed the same question-dependent variability. Discordant AI-human responses (769/3179 = 24.2%) were mostly due to interpretive differences (18.3%); AI inaccuracies were rare (1.51%), while humans were nearly three times more likely to state inaccuracies (4.37%). Findings suggest AI accuracy depends more on interpretability than hallucination. Repeating AI extraction can identify interpretive complexity or ambiguity, refining processes before human review. AI can be a transparent, trustworthy partner in knowledge synthesis, though caution is needed to preserve critical human insights. 

---
# A Unified Contrastive-Generative Framework for Time Series Classification 

**Authors**: Ziyu Liu, Azadeh Alavi, Minyi Li, Xiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09451)  

**Abstract**: Self-supervised learning (SSL) for multivariate time series mainly includes two paradigms: contrastive methods that excel at instance discrimination and generative approaches that model data distributions. While effective individually, their complementary potential remains unexplored. We propose a Contrastive Generative Time series framework (CoGenT), the first framework to unify these paradigms through joint contrastive-generative optimization. CoGenT addresses fundamental limitations of both approaches: it overcomes contrastive learning's sensitivity to high intra-class similarity in temporal data while reducing generative methods' dependence on large datasets. We evaluate CoGenT on six diverse time series datasets. The results show consistent improvements, with up to 59.2% and 14.27% F1 gains over standalone SimCLR and MAE, respectively. Our analysis reveals that the hybrid objective preserves discriminative power while acquiring generative robustness. These findings establish a foundation for hybrid SSL in temporal domains. We will release the code shortly. 

---
# Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference 

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09442)  

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment. 

---
# What-Meets-Where: Unified Learning of Action and Contact Localization in a New Dataset 

**Authors**: Yuxiao Wang, Yu Lei, Wolin Liang, Weiying Xue, Zhenao Wei, Nan Zhuang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09428)  

**Abstract**: People control their bodies to establish contact with the environment. To comprehensively understand actions across diverse visual contexts, it is essential to simultaneously consider \textbf{what} action is occurring and \textbf{where} it is happening. Current methodologies, however, often inadequately capture this duality, typically failing to jointly model both action semantics and their spatial contextualization within scenes. To bridge this gap, we introduce a novel vision task that simultaneously predicts high-level action semantics and fine-grained body-part contact regions. Our proposed framework, PaIR-Net, comprises three key components: the Contact Prior Aware Module (CPAM) for identifying contact-relevant body parts, the Prior-Guided Concat Segmenter (PGCS) for pixel-wise contact segmentation, and the Interaction Inference Module (IIM) responsible for integrating global interaction relationships. To facilitate this task, we present PaIR (Part-aware Interaction Representation), a comprehensive dataset containing 13,979 images that encompass 654 actions, 80 object categories, and 17 body parts. Experimental evaluation demonstrates that PaIR-Net significantly outperforms baseline approaches, while ablation studies confirm the efficacy of each architectural component. The code and dataset will be released upon publication. 

---
# Implicit Hypergraph Neural Networks: A Stable Framework for Higher-Order Relational Learning with Provable Guarantees 

**Authors**: Xiaoyu Li, Guangyu Tang, Jiaojiao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09427)  

**Abstract**: Many real-world interactions are group-based rather than pairwise such as papers with multiple co-authors and users jointly engaging with items. Hypergraph neural networks have shown great promise at modeling higher-order relations, but their reliance on a fixed number of explicit message-passing layers limits long-range dependency capture and can destabilize training as depth grows. In this work, we introduce Implicit Hypergraph Neural Networks (IHGNN), which bring the implicit equilibrium formulation to hypergraphs: instead of stacking layers, IHGNN computes representations as the solution to a nonlinear fixed-point equation, enabling stable and efficient global propagation across hyperedges without deep architectures. We develop a well-posed training scheme with provable convergence, analyze the oversmoothing conditions and expressivity of the model, and derive a transductive generalization bound on hypergraphs. We further present an implicit-gradient training procedure coupled with a projection-based stabilization strategy. Extensive experiments on citation benchmarks show that IHGNN consistently outperforms strong traditional graph/hypergraph neural network baselines in both accuracy and robustness. Empirically, IHGNN is resilient to random initialization and hyperparameter variation, highlighting its strong generalization and practical value for higher-order relational learning. 

---
# Domain-Generalization to Improve Learning in Meta-Learning Algorithms 

**Authors**: Usman Anjum, Chris Stockman, Cat Luong, Justin Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09418)  

**Abstract**: This paper introduces Domain Generalization Sharpness-Aware Minimization Model-Agnostic Meta-Learning (DGS-MAML), a novel meta-learning algorithm designed to generalize across tasks with limited training data. DGS-MAML combines gradient matching with sharpness-aware minimization in a bi-level optimization framework to enhance model adaptability and robustness. We support our method with theoretical analysis using PAC-Bayes and convergence guarantees. Experimental results on benchmark datasets show that DGS-MAML outperforms existing approaches in terms of accuracy and generalization. The proposed method is particularly useful for scenarios requiring few-shot learning and quick adaptation, and the source code is publicly available at GitHub. 

---
# RampNet: A Two-Stage Pipeline for Bootstrapping Curb Ramp Detection in Streetscape Images from Open Government Metadata 

**Authors**: John S. O'Meara, Jared Hwang, Zeyu Wang, Michael Saugstad, Jon E. Froehlich  

**Link**: [PDF](https://arxiv.org/pdf/2508.09415)  

**Abstract**: Curb ramps are critical for urban accessibility, but robustly detecting them in images remains an open problem due to the lack of large-scale, high-quality datasets. While prior work has attempted to improve data availability with crowdsourced or manually labeled data, these efforts often fall short in either quality or scale. In this paper, we introduce and evaluate a two-stage pipeline called RampNet to scale curb ramp detection datasets and improve model performance. In Stage 1, we generate a dataset of more than 210,000 annotated Google Street View (GSV) panoramas by auto-translating government-provided curb ramp location data to pixel coordinates in panoramic images. In Stage 2, we train a curb ramp detection model (modified ConvNeXt V2) from the generated dataset, achieving state-of-the-art performance. To evaluate both stages of our pipeline, we compare to manually labeled panoramas. Our generated dataset achieves 94.0% precision and 92.5% recall, and our detection model reaches 0.9236 AP -- far exceeding prior work. Our work contributes the first large-scale, high-quality curb ramp detection dataset, benchmark, and model. 

---
# Understanding Dementia Speech Alignment with Diffusion-Based Image Generation 

**Authors**: Mansi, Anastasios Lepipas, Dominika Woszczyk, Yiying Guan, Soteris Demetriou  

**Link**: [PDF](https://arxiv.org/pdf/2508.09385)  

**Abstract**: Text-to-image models generate highly realistic images based on natural language descriptions and millions of users use them to create and share images online. While it is expected that such models can align input text and generated image in the same latent space little has been done to understand whether this alignment is possible between pathological speech and generated images. In this work, we examine the ability of such models to align dementia-related speech information with the generated images and develop methods to explain this alignment. Surprisingly, we found that dementia detection is possible from generated images alone achieving 75% accuracy on the ADReSS dataset. We then leverage explainability methods to show which parts of the language contribute to the detection. 

---
# X-UniMotion: Animating Human Images with Expressive, Unified and Identity-Agnostic Motion Latents 

**Authors**: Guoxian Song, Hongyi Xu, Xiaochen Zhao, You Xie, Tianpei Gu, Zenan Li, Chenxu Zhang, Linjie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.09383)  

**Abstract**: We present X-UniMotion, a unified and expressive implicit latent representation for whole-body human motion, encompassing facial expressions, body poses, and hand gestures. Unlike prior motion transfer methods that rely on explicit skeletal poses and heuristic cross-identity adjustments, our approach encodes multi-granular motion directly from a single image into a compact set of four disentangled latent tokens -- one for facial expression, one for body pose, and one for each hand. These motion latents are both highly expressive and identity-agnostic, enabling high-fidelity, detailed cross-identity motion transfer across subjects with diverse identities, poses, and spatial configurations.
To achieve this, we introduce a self-supervised, end-to-end framework that jointly learns the motion encoder and latent representation alongside a DiT-based video generative model, trained on large-scale, diverse human motion datasets. Motion--identity disentanglement is enforced via 2D spatial and color augmentations, as well as synthetic 3D renderings of cross-identity subject pairs under shared poses. Furthermore, we guide motion token learning with auxiliary decoders that promote fine-grained, semantically aligned, and depth-aware motion embeddings.
Extensive experiments show that X-UniMotion outperforms state-of-the-art methods, producing highly expressive animations with superior motion fidelity and identity preservation. 

---
# What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation? 

**Authors**: Kumar Abhishek, Jeremy Kawahara, Ghassan Hamarneh  

**Link**: [PDF](https://arxiv.org/pdf/2508.09381)  

**Abstract**: Medical image segmentation exhibits intra- and inter-annotator variability due to ambiguous object boundaries, annotator preferences, expertise, and tools, among other factors. Lesions with ambiguous boundaries, e.g., spiculated or infiltrative nodules, or irregular borders per the ABCD rule, are particularly prone to disagreement and are often associated with malignancy. In this work, we curate IMA++, the largest multi-annotator skin lesion segmentation dataset, on which we conduct an in-depth study of variability due to annotator, malignancy, tool, and skill factors. We find a statistically significant (p<0.001) association between inter-annotator agreement (IAA), measured using Dice, and the malignancy of skin lesions. We further show that IAA can be accurately predicted directly from dermoscopic images, achieving a mean absolute error of 0.108. Finally, we leverage this association by utilizing IAA as a "soft" clinical feature within a multi-task learning objective, yielding a 4.2% improvement in balanced accuracy averaged across multiple model architectures and across IMA++ and four public dermoscopic datasets. The code is available at this https URL. 

---
# APIO: Automatic Prompt Induction and Optimization for Grammatical Error Correction and Text Simplification 

**Authors**: Artem Chernodub, Aman Saini, Yejin Huh, Vivek Kulkarni, Vipul Raheja  

**Link**: [PDF](https://arxiv.org/pdf/2508.09378)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled a wide range of natural language processing (NLP) tasks to be performed through simple prompt-based interactions. Consequently, several approaches have been proposed to engineer prompts that most effectively enable LLMs to perform a given task (e.g., chain-of-thought prompting). In settings with a well-defined metric to optimize model performance, automatic prompt optimization (APO) methods have been developed to refine a seed prompt. Advancing this line of research, we propose APIO, a simple but effective prompt induction and optimization approach for the tasks of Grammatical Error Correction (GEC) and Text Simplification, without relying on manually specified seed prompts. APIO achieves a new state-of-the-art performance for purely LLM-based prompting methods on these tasks. We make our data, code, prompts, and outputs publicly available. 

---
# A Signer-Invariant Conformer and Multi-Scale Fusion Transformer for Continuous Sign Language Recognition 

**Authors**: Md Rezwanul Haque, Md. Milon Islam, S M Taslim Uddin Raju, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2508.09372)  

**Abstract**: Continuous Sign Language Recognition (CSLR) faces multiple challenges, including significant inter-signer variability and poor generalization to novel sentence structures. Traditional solutions frequently fail to handle these issues efficiently. For overcoming these constraints, we propose a dual-architecture framework. For the Signer-Independent (SI) challenge, we propose a Signer-Invariant Conformer that combines convolutions with multi-head self-attention to learn robust, signer-agnostic representations from pose-based skeletal keypoints. For the Unseen-Sentences (US) task, we designed a Multi-Scale Fusion Transformer with a novel dual-path temporal encoder that captures both fine-grained posture dynamics, enabling the model's ability to comprehend novel grammatical compositions. Experiments on the challenging Isharah-1000 dataset establish a new standard for both CSLR benchmarks. The proposed conformer architecture achieves a Word Error Rate (WER) of 13.07% on the SI challenge, a reduction of 13.53% from the state-of-the-art. On the US task, the transformer model scores a WER of 47.78%, surpassing previous work. In the SignEval 2025 CSLR challenge, our team placed 2nd in the US task and 4th in the SI task, demonstrating the performance of these models. The findings validate our key hypothesis: that developing task-specific networks designed for the particular challenges of CSLR leads to considerable performance improvements and establishes a new baseline for further research. The source code is available at: this https URL. 

---
# FusionEnsemble-Net: An Attention-Based Ensemble of Spatiotemporal Networks for Multimodal Sign Language Recognition 

**Authors**: Md. Milon Islam, Md Rezwanul Haque, S M Taslim Uddin Raju, Fakhri Karray  

**Link**: [PDF](https://arxiv.org/pdf/2508.09362)  

**Abstract**: Accurate recognition of sign language in healthcare communication poses a significant challenge, requiring frameworks that can accurately interpret complex multimodal gestures. To deal with this, we propose FusionEnsemble-Net, a novel attention-based ensemble of spatiotemporal networks that dynamically fuses visual and motion data to enhance recognition accuracy. The proposed approach processes RGB video and range Doppler map radar modalities synchronously through four different spatiotemporal networks. For each network, features from both modalities are continuously fused using an attention-based fusion module before being fed into an ensemble of classifiers. Finally, the outputs of these four different fused channels are combined in an ensemble classification head, thereby enhancing the model's robustness. Experiments demonstrate that FusionEnsemble-Net outperforms state-of-the-art approaches with a test accuracy of 99.44% on the large-scale MultiMeDaLIS dataset for Italian Sign Language. Our findings indicate that an ensemble of diverse spatiotemporal networks, unified by attention-based fusion, yields a robust and accurate framework for complex, multimodal isolated gesture recognition tasks. The source code is available at: this https URL. 

---
# The Human-AI Hybrid Delphi Model: A Structured Framework for Context-Rich, Expert Consensus in Complex Domains 

**Authors**: Cathy Speed, Ahmed A. Metwally  

**Link**: [PDF](https://arxiv.org/pdf/2508.09349)  

**Abstract**: Expert consensus plays a critical role in domains where evidence is complex, conflicting, or insufficient for direct prescription. Traditional methods, such as Delphi studies, consensus conferences, and systematic guideline synthesis, offer structure but face limitations including high panel burden, interpretive oversimplification, and suppression of conditional nuance. These challenges are now exacerbated by information overload, fragmentation of the evidence base, and increasing reliance on publicly available sources that lack expert filtering. This study introduces and evaluates a Human-AI Hybrid Delphi (HAH-Delphi) framework designed to augment expert consensus development by integrating a generative AI model (Gemini 2.5 Pro), small panels of senior human experts, and structured facilitation. The HAH-Delphi was tested in three phases: retrospective replication, prospective comparison, and applied deployment in two applied domains (endurance training and resistance and mixed cardio/strength training). The AI replicated 95% of published expert consensus conclusions in Phase I and showed 95% directional agreement with senior human experts in Phase II, though it lacked experiential and pragmatic nuance. In Phase III, compact panels of six senior experts achieved >90% consensus coverage and reached thematic saturation before the final participant. The AI provided consistent, literature-grounded scaffolding that supported divergence resolution and accelerated saturation. The HAH-Delphi framework offers a flexible, scalable approach for generating high-quality, context-sensitive consensus. Its successful application across health, coaching, and performance science confirms its methodological robustness and supports its use as a foundation for generating conditional, personalised guidance and published consensus frameworks at scale. 

---
# Collective dynamics of strategic classification 

**Authors**: Marta C. Couto, Flavia Barsotti, Fernando P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09340)  

**Abstract**: Classification algorithms based on Artificial Intelligence (AI) are nowadays applied in high-stakes decisions in finance, healthcare, criminal justice, or education. Individuals can strategically adapt to the information gathered about classifiers, which in turn may require algorithms to be re-trained. Which collective dynamics will result from users' adaptation and algorithms' retraining? We apply evolutionary game theory to address this question. Our framework provides a mathematically rigorous way of treating the problem of feedback loops between collectives of users and institutions, allowing to test interventions to mitigate the adverse effects of strategic adaptation. As a case study, we consider institutions deploying algorithms for credit lending. We consider several scenarios, each representing different interaction paradigms. When algorithms are not robust against strategic manipulation, we are able to capture previous challenges discussed in the strategic classification literature, whereby users either pay excessive costs to meet the institutions' expectations (leading to high social costs) or game the algorithm (e.g., provide fake information). From this baseline setting, we test the role of improving gaming detection and providing algorithmic recourse. We show that increased detection capabilities reduce social costs and could lead to users' improvement; when perfect classifiers are not feasible (likely to occur in practice), algorithmic recourse can steer the dynamics towards high users' improvement rates. The speed at which the institutions re-adapt to the user's population plays a role in the final outcome. Finally, we explore a scenario where strict institutions provide actionable recourse to their unsuccessful users and observe cycling dynamics so far unnoticed in the literature. 

---
# RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs 

**Authors**: Zhongtian Sun, Anoushka Harit  

**Link**: [PDF](https://arxiv.org/pdf/2508.09334)  

**Abstract**: We propose RicciFlowRec, a geometric recommendation framework that performs root cause attribution via Ricci curvature and flow on dynamic financial graphs. By modelling evolving interactions among stocks, macroeconomic indicators, and news, we quantify local stress using discrete Ricci curvature and trace shock propagation via Ricci flow. Curvature gradients reveal causal substructures, informing a structural risk-aware ranking function. Preliminary results on S\&P~500 data with FinBERT-based sentiment show improved robustness and interpretability under synthetic perturbations. This ongoing work supports curvature-based attribution and early-stage risk-aware ranking, with plans for portfolio optimization and return forecasting. To our knowledge, RicciFlowRec is the first recommender to apply geometric flow-based reasoning in financial decision support. 

---
# Synaptic Pruning: A Biological Inspiration for Deep Learning Regularization 

**Authors**: Gideon Vos, Liza van Eijk, Zoltan Sarnyai, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09330)  

**Abstract**: Synaptic pruning in biological brains removes weak connections to improve efficiency. In contrast, dropout regularization in artificial neural networks randomly deactivates neurons without considering activity-dependent pruning. We propose a magnitude-based synaptic pruning method that better reflects biology by progressively removing low-importance connections during training. Integrated directly into the training loop as a dropout replacement, our approach computes weight importance from absolute magnitudes across layers and applies a cubic schedule to gradually increase global sparsity. At fixed intervals, pruning masks permanently remove low-importance weights while maintaining gradient flow for active ones, eliminating the need for separate pruning and fine-tuning phases. Experiments on multiple time series forecasting models including RNN, LSTM, and Patch Time Series Transformer across four datasets show consistent gains. Our method ranked best overall, with statistically significant improvements confirmed by Friedman tests (p < 0.01). In financial forecasting, it reduced Mean Absolute Error by up to 20% over models with no or standard dropout, and up to 52% in select transformer models. This dynamic pruning mechanism advances regularization by coupling weight elimination with progressive sparsification, offering easy integration into diverse architectures. Its strong performance, especially in financial time series forecasting, highlights its potential as a practical alternative to conventional dropout techniques. 

---
# SegDAC: Segmentation-Driven Actor-Critic for Visual Reinforcement Learning 

**Authors**: Alexandre Brown, Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.09325)  

**Abstract**: Visual reinforcement learning (RL) is challenging due to the need to learn both perception and actions from high-dimensional inputs and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains unclear. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground segments semantically via text prompts. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks. 

---
# TEN: Table Explicitization, Neurosymbolically 

**Authors**: Nikita Mehrotra, Aayush Kumar, Sumit Gulwani, Arjun Radhakrishna, Ashish Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2508.09324)  

**Abstract**: We present a neurosymbolic approach, TEN, for extracting tabular data from semistructured input text. This task is particularly challenging for text input that does not use special delimiters consistently to separate columns and rows. Purely neural approaches perform poorly due to hallucinations and their inability to enforce hard constraints. TEN uses Structural Decomposition prompting - a specialized chain-of-thought prompting approach - on a large language model (LLM) to generate an initial table, and thereafter uses a symbolic checker to evaluate not only the well-formedness of that table, but also detect cases of hallucinations or forgetting. The output of the symbolic checker is processed by a critique-LLM to generate guidance for fixing the table, which is presented to the original LLM in a self-debug loop. Our extensive experiments demonstrate that TEN significantly outperforms purely neural baselines across multiple datasets and metrics, achieving significantly higher exact match accuracy and substantially reduced hallucination rates. A 21-participant user study further confirms that TEN's tables are rated significantly more accurate (mean score: 5.0 vs 4.3; p = 0.021), and are consistently preferred for ease of verification and correction, with participants favoring our method in over 60% of the cases. 

---
# Leveraging Large Language Models for Rare Disease Named Entity Recognition 

**Authors**: Nan Miles Xi, Yu Deng, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09323)  

**Abstract**: Named Entity Recognition (NER) in the rare disease domain poses unique challenges due to limited labeled data, semantic ambiguity between entity types, and long-tail distributions. In this study, we evaluate the capabilities of GPT-4o for rare disease NER under low-resource settings, using a range of prompt-based strategies including zero-shot prompting, few-shot in-context learning, retrieval-augmented generation (RAG), and task-level fine-tuning. We design a structured prompting framework that encodes domain-specific knowledge and disambiguation rules for four entity types. We further introduce two semantically guided few-shot example selection methods to improve in-context performance while reducing labeling effort. Experiments on the RareDis Corpus show that GPT-4o achieves competitive or superior performance compared to BioClinicalBERT, with task-level fine-tuning yielding new state-of-the-art (SOTA) results. Cost-performance analysis reveals that few-shot prompting delivers high returns at low token budgets, while RAG offers marginal additional benefit. An error taxonomy highlights common failure modes such as boundary drift and type confusion, suggesting opportunities for post-processing and hybrid refinement. Our results demonstrate that prompt-optimized LLMs can serve as effective, scalable alternatives to traditional supervised models in biomedical NER, particularly in rare disease applications where annotated data is scarce. 

---
# Exact Verification of Graph Neural Networks with Incremental Constraint Solving 

**Authors**: Minghao Liu, Chia-Hsuan Lu, Marta Kwiatkowska  

**Link**: [PDF](https://arxiv.org/pdf/2508.09320)  

**Abstract**: Graph neural networks (GNNs) are increasingly employed in high-stakes applications, such as fraud detection or healthcare, but are susceptible to adversarial attacks. A number of techniques have been proposed to provide adversarial robustness guarantees, but support for commonly used aggregation functions in message-passing GNNs is still lacking. In this paper, we develop an exact (sound and complete) verification method for GNNs to compute guarantees against attribute and structural perturbations that involve edge addition or deletion, subject to budget constraints. Focusing on node classification tasks, our method employs constraint solving with bound tightening, and iteratively solves a sequence of relaxed constraint satisfaction problems while relying on incremental solving capabilities of solvers to improve efficiency. We implement GNNev, a versatile solver for message-passing neural networks, which supports three aggregation functions, sum, max and mean, with the latter two considered here for the first time. Extensive experimental evaluation of GNNev on two standard benchmarks (Cora and CiteSeer) and two real-world fraud datasets (Amazon and Yelp) demonstrates its usability and effectiveness, as well as superior performance compared to existing {exact verification} tools on sum-aggregated node classification tasks. 

---
# TPTP World Infrastructure for Non-classical Logics 

**Authors**: Alexander Steen, Geoff Sutcliffe  

**Link**: [PDF](https://arxiv.org/pdf/2508.09318)  

**Abstract**: The TPTP World is the well established infrastructure that supports research, development, and deployment of Automated Theorem Proving (ATP) systems. The TPTP World supports a range of classical logics, and since release v9.0.0 has supported non-classical logics. This paper provides a self-contained comprehensive overview of the TPTP World infrastructure for ATP in non-classical logics: the non-classical language extension, problems and solutions, and tool support. A detailed description of use of the infrastructure for quantified normal multi-modal logic is given. 

---
# ParallelSearch: Train your LLMs to Decompose Query and Search Sub-queries in Parallel with Reinforcement Learning 

**Authors**: Shu Zhao, Tan Yu, Anbang Xu, Japinder Singh, Aaditya Shukla, Rama Akkiraju  

**Link**: [PDF](https://arxiv.org/pdf/2508.09303)  

**Abstract**: Reasoning-augmented search agents such as Search-R1, trained via reinforcement learning with verifiable rewards (RLVR), demonstrate remarkable capabilities in multi-step information retrieval from external knowledge sources. These agents address the limitations of their parametric memory by dynamically gathering relevant facts to address complex reasoning tasks. However, existing approaches suffer from a fundamental architectural limitation: they process search queries strictly sequentially, even when handling inherently parallelizable and logically independent comparisons. This sequential bottleneck significantly constrains computational efficiency, particularly for queries that require multiple entity comparisons. To address this critical limitation, we propose ParallelSearch, a novel reinforcement learning framework that empowers large language models (LLMs) to recognize parallelizable query structures and execute multiple search operations concurrently. Our approach introduces dedicated reward functions that incentivize the identification of independent query components while preserving answer accuracy through jointly considering correctness, query decomposition quality, and parallel execution benefits. Comprehensive experiments demonstrate that ParallelSearch outperforms state-of-the-art baselines by an average performance gain of 2.9% across seven question-answering benchmarks. Notably, on parallelizable questions, our method achieves a 12.7% performance improvement while requiring only 69.6% of the LLM calls compared to sequential approaches. 

---
# Decentralized Weather Forecasting via Distributed Machine Learning and Blockchain-Based Model Validation 

**Authors**: Rilwan Umar, Aydin Abadi, Basil Aldali, Benito Vincent, Elliot A. J. Hurley, Hotoon Aljazaeri, Jamie Hedley-Cook, Jamie-Lee Bell, Lambert Uwuigbusun, Mujeeb Ahmed, Shishir Nagaraja, Suleiman Sabo, Weaam Alrbeiqi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09299)  

**Abstract**: Weather forecasting plays a vital role in disaster preparedness, agriculture, and resource management, yet current centralized forecasting systems are increasingly strained by security vulnerabilities, limited scalability, and susceptibility to single points of failure. To address these challenges, we propose a decentralized weather forecasting framework that integrates Federated Learning (FL) with blockchain technology. FL enables collaborative model training without exposing sensitive local data; this approach enhances privacy and reduces data transfer overhead. Meanwhile, the Ethereum blockchain ensures transparent and dependable verification of model updates. To further enhance the system's security, we introduce a reputation-based voting mechanism that assesses the trustworthiness of submitted models while utilizing the Interplanetary File System (IPFS) for efficient off-chain storage. Experimental results demonstrate that our approach not only improves forecasting accuracy but also enhances system resilience and scalability, making it a viable candidate for deployment in real-world, security-critical environments. 

---
# Based AI improves human decision-making but reduces trust 

**Authors**: Shiyang Lai, Junsol Kim, Nadav Kunievsky, Yujin Potter, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2508.09297)  

**Abstract**: Current AI systems minimize risk by enforcing ideological neutrality, yet this may introduce automation bias by suppressing cognitive engagement in human decision-making. We conducted randomized trials with 2,500 participants to test whether culturally biased AI enhances human decision-making. Participants interacted with politically diverse GPT-4o variants on information evaluation tasks. Partisan AI assistants enhanced human performance, increased engagement, and reduced evaluative bias compared to non-biased counterparts, with amplified benefits when participants encountered opposing views. These gains carried a trust penalty: participants underappreciated biased AI and overcredited neutral systems. Exposing participants to two AIs whose biases flanked human perspectives closed the perception-performance gap. These findings complicate conventional wisdom about AI neutrality, suggesting that strategic integration of diverse cultural biases may foster improved and resilient human decision-making. 

---
# Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention's Alternative 

**Authors**: Xi Xuan, Zimo Zhu, Wenxin Zhang, Yi-Cheng Lin, Tomi Kinnunen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09294)  

**Abstract**: Advances in speech synthesis intensify security threats, motivating real-time deepfake detection research. We investigate whether bidirectional Mamba can serve as a competitive alternative to Self-Attention in detecting synthetic speech. Our solution, Fake-Mamba, integrates an XLSR front-end with bidirectional Mamba to capture both local and global artifacts. Our core innovation introduces three efficient encoders: TransBiMamba, ConBiMamba, and PN-BiMamba. Leveraging XLSR's rich linguistic representations, PN-BiMamba can effectively capture the subtle cues of synthetic speech. Evaluated on ASVspoof 21 LA, 21 DF, and In-The-Wild benchmarks, Fake-Mamba achieves 0.97%, 1.74%, and 5.85% EER, respectively, representing substantial relative gains over SOTA models XLSR-Conformer and XLSR-Mamba. The framework maintains real-time inference across utterance lengths, demonstrating strong generalization and practical viability. The code is available at this https URL. 

---
# Ethical Medical Image Synthesis 

**Authors**: Weina Jin, Ashish Sinha, Kumar Abhishek, Ghassan Hamarneh  

**Link**: [PDF](https://arxiv.org/pdf/2508.09293)  

**Abstract**: The task of ethical Medical Image Synthesis (MISyn) is to ensure that the MISyn techniques are researched and developed ethically throughout their entire lifecycle, which is essential to prevent the negative impacts of MISyn. To address the ever-increasing needs and requirements for ethical practice of MISyn research and development, we first conduct a theoretical analysis that identifies the key properties of ethical MISyn and intrinsic limits of MISyn. We identify that synthetic images lack inherent grounding in real medical phenomena, cannot fully represent the training medical images, and inevitably introduce new distribution shifts and biases.
Ethical risks can arise from not acknowledging the intrinsic limits and weaknesses of synthetic images compared to medical images, with the extreme form manifested as misinformation of MISyn that substitutes synthetic images for medical images without acknowledgment. The resulting ethical harms include eroding trust in the medical imaging dataset environment and causing algorithmic discrimination towards stakeholders and the public.
To facilitate collective efforts towards ethical MISyn within and outside the medical image analysis community, we then propose practical supports for ethical practice in MISyn based on the theoretical analysis, including ethical practice recommendations that adapt the existing technical standards, problem formulation, design, and evaluation practice of MISyn to the ethical challenges; and oversight recommendations to facilitate checks and balances from stakeholders and the public. We also present two case studies that demonstrate how to apply the ethical practice recommendations in practice, and identify gaps between existing practice and the ethical practice recommendations. 

---
# Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs 

**Authors**: Aayush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2508.09288)  

**Abstract**: Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research. 

---
# Detection of Odor Presence via Deep Neural Networks 

**Authors**: Matin Hassanloo, Ali Zareh, Mehmet Kemal Özdemir  

**Link**: [PDF](https://arxiv.org/pdf/2508.09264)  

**Abstract**: Odor detection underpins food safety, environmental monitoring, medical diagnostics, and many more fields. The current artificial sensors developed for odor detection struggle with complex mixtures while non-invasive recordings lack reliable single-trial fidelity. To develop a general system for odor detection, in this study we present a preliminary work where we aim to test two hypotheses: (i) that spectral features of local field potentials (LFPs) are sufficient for robust single-trial odor detection and (ii) that signals from the olfactory bulb alone are adequate. To test two hypotheses, we propose an ensemble of complementary one-dimensional convolutional networks (ResCNN and AttentionCNN) that decodes the presence of odor from multichannel olfactory bulb LFPs. Tested on 2,349 trials from seven awake mice, our final ensemble model supports both hypotheses, achieving a mean accuracy of 86.6%, an F1-score of 81.0%, and an AUC of 0.9247, substantially outperforming previous benchmarks. In addition, the t-SNE visualization confirms that our framework captures biologically significant signatures. These findings establish the feasibility of robust single-trial detection of the presence of odor from extracellular LFPs, as well as demonstrate the potential of deep learning models to provide a deeper understanding of olfactory representations. 

---
# Cross-BCI, A Cross-BCI-Paradigm Classifica-tion Model Towards Universal BCI Applications 

**Authors**: Gaojie Zhou, Junhua Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09242)  

**Abstract**: Classification models used in brain-computer interface (BCI) are usually designed for a single BCI paradigm. This requires the redevelopment of the model when applying it to a new BCI paradigm, resulting in repeated costs and effort. Moreover, less complex deep learning models are desired for practical usage, as well as for deployment on portable devices. In or-der to fill the above gaps, we, in this study, proposed a light-weight and unified decoding model for cross-BCI-paradigm classification. The proposed model starts with a tempo-spatial convolution. It is followed by a multi-scale local feature selec-tion module, aiming to extract local features shared across BCI paradigms and generate weighted features. Finally, a mul-ti-dimensional global feature extraction module is designed, in which multi-dimensional global features are extracted from the weighted features and fused with the weighted features to form high-level feature representations associated with BCI para-digms. The results, evaluated on a mixture of three classical BCI paradigms (i.e., MI, SSVEP, and P300), demon-strate that the proposed model achieves 88.39%, 82.36%, 80.01%, and 0.8092 for accuracy, macro-precision, mac-ro-recall, and macro-F1-score, respectively, significantly out-performing the compared models. This study pro-vides a feasible solution for cross-BCI-paradigm classifica-tion. It lays a technological foundation for de-veloping a new generation of unified decoding systems, paving the way for low-cost and universal practical applications. 

---
# NEFMind: Parameter-Efficient Fine-Tuning of Open-Source LLMs for Telecom APIs Automation 

**Authors**: Zainab Khan, Ahmed Hussain, Mukesh Thakur, Arto Hellas, Panos Papadimitratos  

**Link**: [PDF](https://arxiv.org/pdf/2508.09240)  

**Abstract**: The use of Service-Based Architecture in modern telecommunications has exponentially increased Network Functions (NFs) and Application Programming Interfaces (APIs), creating substantial operational complexities in service discovery and management. We introduce \textit{NEFMind}, a framework leveraging parameter-efficient fine-tuning of open-source Large Language Models (LLMs) to address these challenges. It integrates three core components: synthetic dataset generation from Network Exposure Function (NEF) API specifications, model optimization through Quantized-Low-Rank Adaptation, and performance evaluation via GPT-4 Ref Score and BertScore metrics. Targeting 5G Service-Based Architecture APIs, our approach achieves 85% reduction in communication overhead compared to manual discovery methods. Experimental validation using the open-source Phi-2 model demonstrates exceptional API call identification performance at 98-100% accuracy. The fine-tuned Phi-2 model delivers performance comparable to significantly larger models like GPT-4 while maintaining computational efficiency for telecommunications infrastructure deployment. These findings validate domain-specific, parameter-efficient LLM strategies for managing complex API ecosystems in next-generation telecommunications networks. 

---
# Gradient-Direction-Aware Density Control for 3D Gaussian Splatting 

**Authors**: Zheng Zhou, Yu-Jie Xiong, Chun-Ming Xia, Jia-Chen Zhang, Hong-Jian Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09239)  

**Abstract**: The emergence of 3D Gaussian Splatting (3DGS) has significantly advanced novel view synthesis through explicit scene representation, enabling real-time photorealistic rendering. However, existing approaches manifest two critical limitations in complex scenarios: (1) Over-reconstruction occurs when persistent large Gaussians cannot meet adaptive splitting thresholds during density control. This is exacerbated by conflicting gradient directions that prevent effective splitting of these Gaussians; (2) Over-densification of Gaussians occurs in regions with aligned gradient aggregation, leading to redundant component proliferation. This redundancy significantly increases memory overhead due to unnecessary data retention. We present Gradient-Direction-Aware Gaussian Splatting (GDAGS), a gradient-direction-aware adaptive density control framework to address these challenges. Our key innovations: the gradient coherence ratio (GCR), computed through normalized gradient vector norms, which explicitly discriminates Gaussians with concordant versus conflicting gradient directions; and a nonlinear dynamic weighting mechanism leverages the GCR to enable gradient-direction-aware density control. Specifically, GDAGS prioritizes conflicting-gradient Gaussians during splitting operations to enhance geometric details while suppressing redundant concordant-direction Gaussians. Conversely, in cloning processes, GDAGS promotes concordant-direction Gaussian densification for structural completion while preventing conflicting-direction Gaussian overpopulation. Comprehensive evaluations across diverse real-world benchmarks demonstrate that GDAGS achieves superior rendering quality while effectively mitigating over-reconstruction, suppressing over-densification, and constructing compact scene representations with 50\% reduced memory consumption through optimized Gaussians utilization. 

---
# PETLP: A Privacy-by-Design Pipeline for Social Media Data in AI Research 

**Authors**: Nick Oh, Giorgos D. Vrakas, Siân J. M. Brooke, Sasha Morinière, Toju Duke  

**Link**: [PDF](https://arxiv.org/pdf/2508.09232)  

**Abstract**: Social media data presents AI researchers with overlapping obligations under the GDPR, copyright law, and platform terms -- yet existing frameworks fail to integrate these regulatory domains, leaving researchers without unified guidance. We introduce PETLP (Privacy-by-design Extract, Transform, Load, and Present), a compliance framework that embeds legal safeguards directly into extended ETL pipelines. Central to PETLP is treating Data Protection Impact Assessments as living documents that evolve from pre-registration through dissemination. Through systematic Reddit analysis, we demonstrate how extraction rights fundamentally differ between qualifying research organisations (who can invoke DSM Article 3 to override platform restrictions) and commercial entities (bound by terms of service), whilst GDPR obligations apply universally. We reveal why true anonymisation remains unachievable for social media data and expose the legal gap between permitted dataset creation and uncertain model distribution. By structuring compliance decisions into practical workflows and simplifying institutional data management plans, PETLP enables researchers to navigate regulatory complexity with confidence, bridging the gap between legal requirements and research practice. 

---
# Beyond Technocratic XAI: The Who, What & How in Explanation Design 

**Authors**: Ruchira Dhar, Stephanie Brandl, Ninell Oldenburg, Anders Søgaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.09231)  

**Abstract**: The field of Explainable AI (XAI) offers a wide range of techniques for making complex models interpretable. Yet, in practice, generating meaningful explanations is a context-dependent task that requires intentional design choices to ensure accessibility and transparency. This paper reframes explanation as a situated design process -- an approach particularly relevant for practitioners involved in building and deploying explainable systems. Drawing on prior research and principles from design thinking, we propose a three-part framework for explanation design in XAI: asking Who needs the explanation, What they need explained, and How that explanation should be delivered. We also emphasize the need for ethical considerations, including risks of epistemic inequality, reinforcing social inequities, and obscuring accountability and governance. By treating explanation as a sociotechnical design process, this framework encourages a context-aware approach to XAI that supports effective communication and the development of ethically responsible explanations. 

---
# Cowpox: Towards the Immunity of VLM-based Multi-Agent Systems 

**Authors**: Yutong Wu, Jie Zhang, Yiming Li, Chao Zhang, Qing Guo, Nils Lukas, Tianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09230)  

**Abstract**: Vision Language Model (VLM)-based agents are stateful, autonomous entities capable of perceiving and interacting with their environments through vision and language. Multi-agent systems comprise specialized agents who collaborate to solve a (complex) task. A core security property is robustness, stating that the system should maintain its integrity under adversarial attacks. However, the design of existing multi-agent systems lacks the robustness consideration, as a successful exploit against one agent can spread and infect other agents to undermine the entire system's assurance. To address this, we propose a new defense approach, Cowpox, to provably enhance the robustness of multi-agent systems. It incorporates a distributed mechanism, which improves the recovery rate of agents by limiting the expected number of infections to other agents. The core idea is to generate and distribute a special cure sample that immunizes an agent against the attack before exposure and helps recover the already infected agents. We demonstrate the effectiveness of Cowpox empirically and provide theoretical robustness guarantees. 

---
# Cluster Topology-Driven Placement of Experts Reduces Network Traffic in MoE Inference 

**Authors**: Danil Sivtsov, Aleksandr Katrutsa, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2508.09229)  

**Abstract**: Efficient deployment of a pre-trained LLM to a cluster with multiple servers is a critical step for providing fast responses to users' queries. The recent success of Mixture-of-Experts (MoE) LLMs raises the question of how to deploy them efficiently, considering their underlying structure. During the inference in MoE LLMs, only a small part of the experts is selected to process a given token. Moreover, in practice, the experts' load is highly imbalanced. For efficient deployment, one has to distribute the model across a large number of servers using a model placement algorithm. Thus, to improve cluster utilization, the model placement algorithm has to take into account the network topology. This work focuses on the efficient topology-aware placement of the pre-trained MoE LLMs in the inference stage. We propose an integer linear program (ILP) that determines the optimal placement of experts, minimizing the expected number of transmissions. Due to the internal structure, this optimization problem can be solved with a standard ILP solver. We demonstrate that ILP-based placement strategy yields lower network traffic than competitors for small-scale (DeepSeekMoE~16B) and large-scale (DeepSeek-R1~671B) models. 

---
# GSMT: Graph Fusion and Spatiotemporal TaskCorrection for Multi-Bus Trajectory Prediction 

**Authors**: Fan Ding, Hwa Hui Tew, Junn Yong Loo, Susilawati, LiTong Liu, Fang Yu Leong, Xuewen Luo, Kar Keong Chin, Jia Jun Gan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09227)  

**Abstract**: Accurate trajectory prediction for buses is crucial in intelligent transportation systems, particularly within urban environments. In developing regions where access to multimodal data is limited, relying solely on onboard GPS data remains indispensable despite inherent challenges. To address this problem, we propose GSMT, a hybrid model that integrates a Graph Attention Network (GAT) with a sequence-to-sequence Recurrent Neural Network (RNN), and incorporates a task corrector capable of extracting complex behavioral patterns from large-scale trajectory data. The task corrector clusters historical trajectories to identify distinct motion patterns and fine-tunes the predictions generated by the GAT and RNN. Specifically, GSMT fuses dynamic bus information and static station information through embedded hybrid networks to perform trajectory prediction, and applies the task corrector for secondary refinement after the initial predictions are generated. This two-stage approach enables multi-node trajectory prediction among buses operating in dense urban traffic environments under complex conditions. Experiments conducted on a real-world dataset from Kuala Lumpur, Malaysia, demonstrate that our method significantly outperforms existing approaches, achieving superior performance in both short-term and long-term trajectory prediction tasks. 

---
# AMRG: Extend Vision Language Models for Automatic Mammography Report Generation 

**Authors**: Nak-Jun Sung, Donghyun Lee, Bo Hwa Choi, Chae Jung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.09225)  

**Abstract**: Mammography report generation is a critical yet underexplored task in medical AI, characterized by challenges such as multiview image reasoning, high-resolution visual cues, and unstructured radiologic language. In this work, we introduce AMRG (Automatic Mammography Report Generation), the first end-to-end framework for generating narrative mammography reports using large vision-language models (VLMs). Building upon MedGemma-4B-it-a domain-specialized, instruction-tuned VLM-we employ a parameter-efficient fine-tuning (PEFT) strategy via Low-Rank Adaptation (LoRA), enabling lightweight adaptation with minimal computational overhead. We train and evaluate AMRG on DMID, a publicly available dataset of paired high-resolution mammograms and diagnostic reports. This work establishes the first reproducible benchmark for mammography report generation, addressing a longstanding gap in multimodal clinical AI. We systematically explore LoRA hyperparameter configurations and conduct comparative experiments across multiple VLM backbones, including both domain-specific and general-purpose models under a unified tuning protocol. Our framework demonstrates strong performance across both language generation and clinical metrics, achieving a ROUGE-L score of 0.5691, METEOR of 0.6152, CIDEr of 0.5818, and BI-RADS accuracy of 0.5582. Qualitative analysis further highlights improved diagnostic consistency and reduced hallucinations. AMRG offers a scalable and adaptable foundation for radiology report generation and paves the way for future research in multimodal medical AI. 

---
# From Hard Refusals to Safe-Completions: Toward Output-Centric Safety Training 

**Authors**: Yuan Yuan, Tina Sriskandarajah, Anna-Luisa Brakman, Alec Helyar, Alex Beutel, Andrea Vallone, Saachi Jain  

**Link**: [PDF](https://arxiv.org/pdf/2508.09224)  

**Abstract**: Large Language Models used in ChatGPT have traditionally been trained to learn a refusal boundary: depending on the user's intent, the model is taught to either fully comply or outright refuse. While this is a strong mitigation for explicitly malicious prompts, focusing safety training on refusals can lead to brittleness for prompts with obscured user intent. Binary refusal boundaries are especially ill-suited for dual-use cases (such as biology or cybersecurity), where a user request can be answered safely at a high level, but in some cases can lead to malicious uplift if sufficiently detailed or actionable. As an alternative, we propose safe-completions: a safety-training approach that centers on the safety of the assistant's output, rather than a binary classification of the user's intent. Safe-completions seek to maximize helpfulness within the safety policy's constraints. We incorporated this approach into GPT-5 and find that across both production comparisons and internally controlled experiments, safe-completion training improves safety (especially on dual-use prompts), reduces the severity of residual safety failures, and substantially increases model helpfulness. 

---
# Hierarchical Adaptive networks with Task vectors for Test-Time Adaptation 

**Authors**: Sameer Ambekar, Daniel M. Lang, Julia A. Schnabel  

**Link**: [PDF](https://arxiv.org/pdf/2508.09223)  

**Abstract**: Test-time adaptation allows pretrained models to adjust to incoming data streams, addressing distribution shifts between source and target domains. However, standard methods rely on single-dimensional linear classification layers, which often fail to handle diverse and complex shifts. We propose Hierarchical Adaptive Networks with Task Vectors (Hi-Vec), which leverages multiple layers of increasing size for dynamic test-time adaptation. By decomposing the encoder's representation space into such hierarchically organized layers, Hi-Vec, in a plug-and-play manner, allows existing methods to adapt to shifts of varying complexity. Our contributions are threefold: First, we propose dynamic layer selection for automatic identification of the optimal layer for adaptation to each test batch. Second, we propose a mechanism that merges weights from the dynamic layer to other layers, ensuring all layers receive target information. Third, we propose linear layer agreement that acts as a gating function, preventing erroneous fine-tuning by adaptation on noisy batches. We rigorously evaluate the performance of Hi-Vec in challenging scenarios and on multiple target datasets, proving its strong capability to advance state-of-the-art methods. Our results show that Hi-Vec improves robustness, addresses uncertainty, and handles limited batch sizes and increased outlier rates. 

---
# Towards Scalable Training for Handwritten Mathematical Expression Recognition 

**Authors**: Haoyang Li, Jiaqing Li, Jialun Cao, Zongyuan Yang, Yongping Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.09220)  

**Abstract**: Large foundation models have achieved significant performance gains through scalable training on massive datasets. However, the field of \textbf{H}andwritten \textbf{M}athematical \textbf{E}xpression \textbf{R}ecognition (HMER) has been impeded by the scarcity of data, primarily due to the arduous and costly process of manual annotation. To bridge this gap, we propose a novel method integrating limited handwritten formulas with large-scale LaTeX-rendered formulas by developing a scalable data engine to generate complex and consistent LaTeX sequences. With this engine, we built the largest formula dataset to date, termed \texttt{Tex80M}, comprising over 80 million high-quality training instances. Then we propose \texttt{TexTeller}, the first HMER model trained at scale, by mix-training \texttt{Tex80M} with a relatively small HME dataset. The expansive training dataset and our refined pipeline have equipped \texttt{TexTeller} with state-of-the-art (SOTA) performance across nearly all benchmarks. To advance the field, we will openly release our complete model, entire dataset, and full codebase, enabling further research building upon our contributions. 

---
# Understanding Ethical Practices in AI: Insights from a Cross-Role, Cross-Region Survey of AI Development Teams 

**Authors**: Wilder Baldwin, Sepideh Ghanavati, Manuel Woersdoerfer  

**Link**: [PDF](https://arxiv.org/pdf/2508.09219)  

**Abstract**: Recent advances in AI applications have raised growing concerns about the need for ethical guidelines and regulations to mitigate the risks posed by these technologies. In this paper, we present a mixed-method survey study - combining statistical and qualitative analyses - to examine the ethical perceptions, practices, and knowledge of individuals involved in various AI development roles. Our survey includes 414 participants from 43 countries, representing roles such as AI managers, analysts, developers, quality assurance professionals, and information security and privacy experts. The results reveal varying degrees of familiarity and experience with AI ethics principles, government initiatives, and risk mitigation strategies across roles, regions, and other demographic factors. Our findings highlight the importance of a collaborative, role-sensitive approach, involving diverse stakeholders in ethical decision-making throughout the AI development lifecycle. We advocate for developing tailored, inclusive solutions to address ethical challenges in AI development, and we propose future research directions and educational strategies to promote ethics-aware AI practices. 

---
# Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity 

**Authors**: Zuoou Li, Weitong Zhang, Jingyuan Wang, Shuyuan Zhang, Wenjia Bai, Bernhard Kainz, Mengyun Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09218)  

**Abstract**: Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems. 

---
# Real-time deep learning phase imaging flow cytometer reveals blood cell aggregate biomarkers for haematology diagnostics 

**Authors**: Kerem Delikoyun, Qianyu Chen, Liu Wei, Si Ko Myo, Johannes Krell, Martin Schlegel, Win Sen Kuan, John Tshon Yit Soong, Gerhard Schneider, Clarissa Prazeres da Costa, Percy A. Knolle, Laurent Renia, Matthew Edward Cove, Hwee Kuan Lee, Klaus Diepold, Oliver Hayden  

**Link**: [PDF](https://arxiv.org/pdf/2508.09215)  

**Abstract**: While analysing rare blood cell aggregates remains challenging in automated haematology, they could markedly advance label-free functional diagnostics. Conventional flow cytometers efficiently perform cell counting with leukocyte differentials but fail to identify aggregates with flagged results, requiring manual reviews. Quantitative phase imaging flow cytometry captures detailed aggregate morphologies, but clinical use is hampered by massive data storage and offline processing. Incorporating hidden biomarkers into routine haematology panels would significantly improve diagnostics without flagged results. We present RT-HAD, an end-to-end deep learning-based image and data processing framework for off-axis digital holographic microscopy (DHM), which combines physics-consistent holographic reconstruction and detection, representing each blood cell in a graph to recognize aggregates. RT-HAD processes >30 GB of image data on-the-fly with turnaround time of <1.5 min and error rate of 8.9% in platelet aggregate detection, which matches acceptable laboratory error rates of haematology biomarkers and solves the big data challenge for point-of-care diagnostics. 

---
# Deep Generative Models for Discrete Genotype Simulation 

**Authors**: Sihan Xie, Thierry Tribout, Didier Boichard, Blaise Hanczar, Julien Chiquet, Eric Barrey  

**Link**: [PDF](https://arxiv.org/pdf/2508.09212)  

**Abstract**: Deep generative models open new avenues for simulating realistic genomic data while preserving privacy and addressing data accessibility constraints. While previous studies have primarily focused on generating gene expression or haplotype data, this study explores generating genotype data in both unconditioned and phenotype-conditioned settings, which is inherently more challenging due to the discrete nature of genotype data. In this work, we developed and evaluated commonly used generative models, including Variational Autoencoders (VAEs), Diffusion Models, and Generative Adversarial Networks (GANs), and proposed adaptation tailored to discrete genotype data. We conducted extensive experiments on large-scale datasets, including all chromosomes from cow and multiple chromosomes from human. Model performance was assessed using a well-established set of metrics drawn from both deep learning and quantitative genetics literature. Our results show that these models can effectively capture genetic patterns and preserve genotype-phenotype association. Our findings provide a comprehensive comparison of these models and offer practical guidelines for future research in genotype simulation. We have made our code publicly available at this https URL. 

---
# MME-Emotion: A Holistic Evaluation Benchmark for Emotional Intelligence in Multimodal Large Language Models 

**Authors**: Fan Zhang, Zebang Cheng, Chong Deng, Haoxuan Li, Zheng Lian, Qian Chen, Huadai Liu, Wen Wang, Yi-Fan Zhang, Renrui Zhang, Ziyu Guo, Zhihong Zhu, Hao Wu, Haixin Wang, Yefeng Zheng, Xiaojiang Peng, Xian Wu, Kun Wang, Xiangang Li, Jieping Ye, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09210)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have catalyzed transformative progress in affective computing, enabling models to exhibit emergent emotional intelligence. Despite substantial methodological progress, current emotional benchmarks remain limited, as it is still unknown: (a) the generalization abilities of MLLMs across distinct scenarios, and (b) their reasoning capabilities to identify the triggering factors behind emotional states. To bridge these gaps, we present \textbf{MME-Emotion}, a systematic benchmark that assesses both emotional understanding and reasoning capabilities of MLLMs, enjoying \textit{scalable capacity}, \textit{diverse settings}, and \textit{unified protocols}. As the largest emotional intelligence benchmark for MLLMs, MME-Emotion contains over 6,000 curated video clips with task-specific questioning-answering (QA) pairs, spanning broad scenarios to formulate eight emotional tasks. It further incorporates a holistic evaluation suite with hybrid metrics for emotion recognition and reasoning, analyzed through a multi-agent system framework. Through a rigorous evaluation of 20 advanced MLLMs, we uncover both their strengths and limitations, yielding several key insights: \ding{182} Current MLLMs exhibit unsatisfactory emotional intelligence, with the best-performing model achieving only $39.3\%$ recognition score and $56.0\%$ Chain-of-Thought (CoT) score on our benchmark. \ding{183} Generalist models (\emph{e.g.}, Gemini-2.5-Pro) derive emotional intelligence from generalized multimodal understanding capabilities, while specialist models (\emph{e.g.}, R1-Omni) can achieve comparable performance through domain-specific post-training adaptation. By introducing MME-Emotion, we hope that it can serve as a foundation for advancing MLLMs' emotional intelligence in the future. 

---
# Quantum-Enhanced Generative Adversarial Networks: Comparative Analysis of Classical and Hybrid Quantum-Classical Generative Adversarial Networks 

**Authors**: Kun Ming Goh  

**Link**: [PDF](https://arxiv.org/pdf/2508.09209)  

**Abstract**: Generative adversarial networks (GANs) have emerged as a powerful paradigm for producing high-fidelity data samples, yet their performance is constrained by the quality of latent representations, typically sampled from classical noise distributions. This study investigates hybrid quantum-classical GANs (HQCGANs) in which a quantum generator, implemented via parameterised quantum circuits, produces latent vectors for a classical discriminator. We evaluate a classical GAN alongside three HQCGAN variants with 3, 5, and 7 qubits, using Qiskit's AerSimulator with realistic noise models to emulate near-term quantum devices. The binary MNIST dataset (digits 0 and 1) is used to align with the low-dimensional latent spaces imposed by current quantum hardware. Models are trained for 150 epochs and assessed with Frechet Inception Distance (FID) and Kernel Inception Distance (KID). Results show that while the classical GAN achieved the best scores, the 7-qubit HQCGAN produced competitive performance, narrowing the gap in later epochs, whereas the 3-qubit model exhibited earlier convergence limitations. Efficiency analysis indicates only moderate training time increases despite quantum sampling overhead. These findings validate the feasibility of noisy quantum circuits as latent priors in GAN architectures, highlighting their potential to enhance generative modelling within the constraints of the noisy intermediate-scale quantum (NISQ) era. 

---
# CoMoE: Collaborative Optimization of Expert Aggregation and Offloading for MoE-based LLMs at Edge 

**Authors**: Muqing Li, Ning Li, Xin Yuan, Wenchao Xu, Quan Chen, Song Guo, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09208)  

**Abstract**: The proliferation of large language models (LLMs) has driven the adoption of Mixture-of-Experts (MoE) architectures as a promising solution to scale model capacity while controlling computational costs. However, deploying MoE models in resource-constrained mobile edge computing environments presents significant challenges due to their large memory footprint and dynamic expert activation patterns. To address these challenges, we propose a novel dynamic resource-aware collaborative optimization framework that jointly optimizes expert aggregation granularity and offloading strategies based on real-time device resource states, network conditions, and input characteristics in mobile edge environments, denoted as CoMoE. In CoMoE, we first systematically analyze existing expert aggregation techniques, including expert parameter merging,knowledge distillation,and parameter sharing decomposition, identifying their limitations in dynamic mobile this http URL then investigate expert offloading strategies encompassing expert prediction and prefetching, expert caching and scheduling, and multi-tier storage architectures, revealing the interdependencies between routing decisions and offloading this http URL CoMoE incorporates adaptive scheduling mechanisms that respond to user mobility and varying network conditions, enabling efficient MoE deployment across heterogeneous edge devices. Extensive experiments on real mobile edge testbeds demonstrate that CoMoE achieves approximately 70% reduction in memory usage compared to baseline methods, 10.5% lower inference latency than existing expert offloading techniques, while maintaining model performance stability. For large-scale MoE models (e.g,7.4B-parameter Switch-Base-128), the CoMoE reduces memory requirements from 15.6GB to 4.7GB, enabling deployment on resource-constrained mobile edge devices that previously could only support much smaller models. 

---
# From Explainable to Explained AI: Ideas for Falsifying and Quantifying Explanations 

**Authors**: Yoni Schirris, Eric Marcus, Jonas Teuwen, Hugo Horlings, Efstratios Gavves  

**Link**: [PDF](https://arxiv.org/pdf/2508.09205)  

**Abstract**: Explaining deep learning models is essential for clinical integration of medical image analysis systems. A good explanation highlights if a model depends on spurious features that undermines generalization and harms a subset of patients or, conversely, may present novel biological insights. Although techniques like GradCAM can identify influential features, they are measurement tools that do not themselves form an explanation. We propose a human-machine-VLM interaction system tailored to explaining classifiers in computational pathology, including multi-instance learning for whole-slide images. Our proof of concept comprises (1) an AI-integrated slide viewer to run sliding-window experiments to test claims of an explanation, and (2) quantification of an explanation's predictiveness using general-purpose vision-language models. The results demonstrate that this allows us to qualitatively test claims of explanations and can quantifiably distinguish competing explanations. This offers a practical path from explainable AI to explained AI in digital pathology and beyond. Code and prompts are available at this https URL. 

---
# MoQE: Improve Quantization Model performance via Mixture of Quantization Experts 

**Authors**: Jinhao Zhang, Yunquan Zhang, Boyang Zhang, Zeyu Liu, Daning Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09204)  

**Abstract**: Quantization method plays a crucial role in improving model efficiency and reducing deployment costs, enabling the widespread application of deep learning models on resource-constrained devices. However, the quantization process inevitably introduces accuracy degradation. In this paper, we propose Mixture of Quantization Experts( abbr. MoQE), a quantization inference framework based on the Mixture-of-Experts (MoE) architecture, aiming to jointly improve the performance of quantization models. MoQE combines multiple quantization variants of one full-precision model as specialized "quantization experts" and dynamically routes input data to the most suitable expert based on its characteristics. MoQE alleviates the performance degradation commonly seen in single quantization models through specialization quantization expert models. We design lightweight, structure-aware router models tailored for both CV and NLP tasks. Experimental evaluations on ResNet, LLaMA, and Qwen model families across benchmark datasets including ImageNet, WikiText, C4, and OpenWebText demonstrate that MoQE achieves performance comparable to SOTA quantization model, without incurring significant increases in inference latency. 

---
# Personalized Feature Translation for Expression Recognition: An Efficient Source-Free Domain Adaptation Method 

**Authors**: Masoumeh Sharafi, Soufiane Belharbi, Houssem Ben Salem, Ali Etemad, Alessandro Lameiras Koerich, Marco Pedersoli, Simon Bacon, Eric Granger  

**Link**: [PDF](https://arxiv.org/pdf/2508.09202)  

**Abstract**: Facial expression recognition (FER) models are employed in many video-based affective computing applications, such as human-computer interaction and healthcare monitoring. However, deep FER models often struggle with subtle expressions and high inter-subject variability, limiting their performance in real-world applications. To improve their performance, source-free domain adaptation (SFDA) methods have been proposed to personalize a pretrained source model using only unlabeled target domain data, thereby avoiding data privacy, storage, and transmission constraints. This paper addresses a challenging scenario where source data is unavailable for adaptation, and only unlabeled target data consisting solely of neutral expressions is available. SFDA methods are not typically designed to adapt using target data from only a single class. Further, using models to generate facial images with non-neutral expressions can be unstable and computationally intensive. In this paper, personalized feature translation (PFT) is proposed for SFDA. Unlike current image translation methods for SFDA, our lightweight method operates in the latent space. We first pre-train the translator on the source domain data to transform the subject-specific style features from one source subject into another. Expression information is preserved by optimizing a combination of expression consistency and style-aware objectives. Then, the translator is adapted on neutral target data, without using source data or image synthesis. By translating in the latent space, PFT avoids the complexity and noise of face expression generation, producing discriminative embeddings optimized for classification. Using PFT eliminates the need for image synthesis, reduces computational overhead (using a lightweight translator), and only adapts part of the model, making the method efficient compared to image-based translation. 

---
# Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models: A Unified and Accurate Approach 

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09201)  

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. Although recent detection works have shifted to internal representations due to their rich cross-modal information, most methods rely on heuristic rules rather than principled objectives, resulting in suboptimal performance. To address these limitations, we propose Learning to Detect (LoD), a novel unsupervised framework that formulates jailbreak detection as anomaly detection. LoD introduces two key components: Multi-modal Safety Concept Activation Vectors (MSCAV), which capture layer-wise safety-related representations across modalities, and the Safety Pattern Auto-Encoder, which models the distribution of MSCAV derived from safe inputs and detects anomalies via reconstruction errors. By training the auto-encoder (AE) solely on safe samples without attack labels, LoD naturally identifies jailbreak inputs as distributional anomalies, enabling accurate and unified detection of jailbreak attacks. Comprehensive experiments on three different LVLMs and five benchmarks demonstrate that LoD achieves state-of-the-art performance, with an average AUROC of 0.9951 and an improvement of up to 38.89% in the minimum AUROC over the strongest baselines. 

---
# Zero-shot self-supervised learning of single breath-hold magnetic resonance cholangiopancreatography (MRCP) reconstruction 

**Authors**: Jinho Kim, Marcel Dominik Nickel, Florian Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2508.09200)  

**Abstract**: Purpose: To investigate the feasibility of applying zero-shot self-supervised learning reconstruction to reduce breath-hold times in magnetic resonance cholangiopancreatography (MRCP). Methods: Breath-hold MRCP was acquired from 11 healthy volunteers on a 3T scanner using an incoherent k-space sampling pattern leading to a breath-hold duration of 14s. We evaluated zero-shot reconstruction of breath-hold MRCP against parallel imaging of respiratory-triggered MRCP acquired in 338s on average and compressed sensing reconstruction of breath-hold MRCP. To address the long computation times of zero-shot trainings, we used a training approach that leverages a pretrained network to reduce backpropagation depth during training. Results: Zero-shot learning reconstruction significantly improved visual image quality compared to compressed sensing reconstruction, particularly in terms of signal-to-noise ratio and ductal delineation, and reached a level of quality comparable to that of successful respiratory-triggered acquisitions with regular breathing patterns. Shallow training provided nearly equivalent reconstruction performance with a training time of 11 minutes in comparison to 271 minutes for a conventional zero-shot training. Conclusion: Zero-shot learning delivers high-fidelity MRCP reconstructions with reduced breath-hold times, and shallow training offers a practical solution for translation to time-constrained clinical workflows. 

---
# $Δ$-AttnMask: Attention-Guided Masked Hidden States for Efficient Data Selection and Augmentation 

**Authors**: Jucheng Hu, Suorong Yang, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.09199)  

**Abstract**: Visual Instruction Finetuning (VIF) is pivotal for post-training Vision-Language Models (VLMs). Unlike unimodal instruction finetuning in plain-text large language models, which mainly requires instruction datasets to enable model instruction-following ability, VIF also requires multimodal data to enable joint visual and textual understanding; therefore, it typically requires more data. Consequently, VIF imposes stricter data selection challenges: the method must scale efficiently to handle larger data demands while ensuring the quality of both visual and textual content, as well as their alignment. Despite its critical impact on performance, data selection for VIF remains an understudied area. In this paper, we propose $\Delta$-AttnMask. This data-efficient framework quantifies sample quality through attention-guided masking of the model's hidden states, jointly evaluating image-text pairs without requiring domain labels, auxiliary models, or extra training. By computing loss differences ($\Delta$) between the original states and states masked using high-attention regions, $\Delta$-AttnMask intrinsically assesses sample quality. Experiments across multiple VLMs and datasets show that $\Delta$-AttnMask achieves state-of-the-art performance with just 20% of data, accelerating training by 5x while surpassing full-dataset baselines by +10.1% in overall accuracy. Its model-agnostic and data-agnostic design ensures broad applicability across modalities and architectures. 

---
# ADT4Coupons: An Innovative Framework for Sequential Coupon Distribution in E-commerce 

**Authors**: Li Kong, Bingzhe Wang, Zhou Chen, Suhan Hu, Yuchao Ma, Qi Qi, Suoyuan Song, Bicheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.09198)  

**Abstract**: Coupon distribution is a critical marketing strategy used by online platforms to boost revenue and enhance user engagement. Regrettably, existing coupon distribution strategies fall far short of effectively leveraging the complex sequential interactions between platforms and users. This critical oversight, despite the abundance of e-commerce log data, has precipitated a performance plateau. In this paper, we focus on the scene that the platforms make sequential coupon distribution decision multiple times for various users, with each user interacting with the platform repeatedly. Based on this marketing scenario, we propose a novel marketing framework, named Aligned Decision Transformer for Coupons (ADT4Coupons), to directly devise coupon distribution policy for long-term revenue boosting. ADT4Coupons enables optimized online decision-making in a variety of real-world marketing scenarios. It achieves this by seamlessly integrating three key characteristics, general scenarios, sequential modeling with more comprehensive historical data, and efficient iterative updates within a unified framework. Furthermore, empirical results on real-world industrial dataset, alongside public and synthetic datasets demonstrate the superiority of our framework. 

---
# MX-AI: Agentic Observability and Control Platform for Open and AI-RAN 

**Authors**: Ilias Chatzistefanidis, Andrea Leone, Ali Yaghoubian, Mikel Irazabal, Sehad Nassim, Lina Bariah, Merouane Debbah, Navid Nikaein  

**Link**: [PDF](https://arxiv.org/pdf/2508.09197)  

**Abstract**: Future 6G radio access networks (RANs) will be artificial intelligence (AI)-native: observed, reasoned about, and re-configured by autonomous agents cooperating across the cloud-edge continuum. We introduce MX-AI, the first end-to-end agentic system that (i) instruments a live 5G Open RAN testbed based on OpenAirInterface (OAI) and FlexRIC, (ii) deploys a graph of Large-Language-Model (LLM)-powered agents inside the Service Management and Orchestration (SMO) layer, and (iii) exposes both observability and control functions for 6G RAN resources through natural-language intents. On 50 realistic operational queries, MX-AI attains a mean answer quality of 4.1/5.0 and 100 % decision-action accuracy, while incurring only 8.8 seconds end-to-end latency when backed by GPT-4.1. Thus, it matches human-expert performance, validating its practicality in real settings. We publicly release the agent graph, prompts, and evaluation harness to accelerate open research on AI-native RANs. A live demo is presented here: this https URL 

---
# FIVA: Federated Inverse Variance Averaging for Universal CT Segmentation with Uncertainty Estimation 

**Authors**: Asim Ukaye, Numan Saeed, Karthik Nandakumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.09196)  

**Abstract**: Different CT segmentation datasets are typically obtained from different scanners under different capture settings and often provide segmentation labels for a limited and often disjoint set of organs. Using these heterogeneous data effectively while preserving patient privacy can be challenging. This work presents a novel federated learning approach to achieve universal segmentation across diverse abdominal CT datasets by utilizing model uncertainty for aggregation and predictive uncertainty for inference. Our approach leverages the inherent noise in stochastic mini-batch gradient descent to estimate a distribution over the model weights to provide an on-the-go uncertainty over the model parameters at the client level. The parameters are then aggregated at the server using the additional uncertainty information using a Bayesian-inspired inverse-variance aggregation scheme. Furthermore, the proposed method quantifies prediction uncertainty by propagating the uncertainty from the model weights, providing confidence measures essential for clinical decision-making. In line with recent work shown, predictive uncertainty is utilized in the inference stage to improve predictive performance. Experimental evaluations demonstrate the effectiveness of this approach in improving both the quality of federated aggregation and uncertainty-weighted inference compared to previously established baselines. The code for this work is made available at: this https URL 

---
# impuTMAE: Multi-modal Transformer with Masked Pre-training for Missing Modalities Imputation in Cancer Survival Prediction 

**Authors**: Maria Boyko, Aleksandra Beliaeva, Dmitriy Kornilov, Alexander Bernstein, Maxim Sharaev  

**Link**: [PDF](https://arxiv.org/pdf/2508.09195)  

**Abstract**: The use of diverse modalities, such as omics, medical images, and clinical data can not only improve the performance of prognostic models but also deepen an understanding of disease mechanisms and facilitate the development of novel treatment approaches. However, medical data are complex, often incomplete, and contains missing modalities, making effective handling its crucial for training multimodal models. We introduce impuTMAE, a novel transformer-based end-to-end approach with an efficient multimodal pre-training strategy. It learns inter- and intra-modal interactions while simultaneously imputing missing modalities by reconstructing masked patches. Our model is pre-trained on heterogeneous, incomplete data and fine-tuned for glioma survival prediction using TCGA-GBM/LGG and BraTS datasets, integrating five modalities: genetic (DNAm, RNA-seq), imaging (MRI, WSI), and clinical data. By addressing missing data during pre-training and enabling efficient resource utilization, impuTMAE surpasses prior multimodal approaches, achieving state-of-the-art performance in glioma patient survival prediction. Our code is available at this https URL 

---
# Meta-Learning for Speeding Up Large Model Inference in Decentralized Environments 

**Authors**: Yipeng Du, Zihao Wang, Ahmad Farhan, Claudio Angione, Harry Yang, Fielding Johnston, James P. Buban, Patrick Colangelo, Yue Zhao, Yuzhe Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09194)  

**Abstract**: The deployment of large-scale models, such as large language models (LLMs), incurs substantial costs due to their computational demands. To mitigate these costs and address challenges related to scalability and data security, there is a growing shift towards decentralized systems for model deployment, where choosing efficient inference acceleration schemes become crucial to manage computational resources effectively and enhance system responsiveness. In this work, we address the challenge of selecting optimal acceleration methods in decentralized systems by introducing a meta-learning-based framework. This framework automates the selection process by learning from historical performance data of various acceleration techniques across different tasks. Unlike traditional methods that rely on random selection or expert intuition, our approach systematically identifies the best acceleration strategies based on the specific characteristics of each task. We demonstrate that our meta-learning framework not only streamlines the decision-making process but also consistently outperforms conventional methods in terms of efficiency and performance. Our results highlight the potential of inference acceleration in decentralized AI systems, offering a path towards more democratic and economically feasible artificial intelligence solutions. 

---
# Multi-Objective Instruction-Aware Representation Learning in Procedural Content Generation RL 

**Authors**: Sung-Hyun Kim, In-Chang Baek, Seo-Young Lee, Geum-Hwan Hwang, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09193)  

**Abstract**: Recent advancements in generative modeling emphasize the importance of natural language as a highly expressive and accessible modality for controlling content generation. However, existing instructed reinforcement learning for procedural content generation (IPCGRL) method often struggle to leverage the expressive richness of textual input, especially under complex, multi-objective instructions, leading to limited controllability. To address this problem, we propose \textit{MIPCGRL}, a multi-objective representation learning method for instructed content generators, which incorporates sentence embeddings as conditions. MIPCGRL effectively trains a multi-objective embedding space by incorporating multi-label classification and multi-head regression networks. Experimental results show that the proposed method achieves up to a 13.8\% improvement in controllability with multi-objective instructions. The ability to process complex instructions enables more expressive and flexible content generation. 

---
# Diffusion LLMs Can Do Faster-Than-AR Inference via Discrete Diffusion Forcing 

**Authors**: Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09192)  

**Abstract**: Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive (AR) LLMs for text generation, with the potential to decode multiple tokens in a single iteration. However, none of the existing open-source dLLMs have achieved superior inference speed over AR LLMs of similar size. This paper breaks this barrier based on a simple and effective strategy named discrete diffusion forcing (D2F). D2F equips dLLMs with two key capabilities: (1) block-wise autoregressive generation to enable KV cache utilization; (2) prediction of following tokens without requiring completion of prior blocks for inter-block parallel decoding. In this way, the vanilla dLLMs are refurbished into an AR-diffusion hybrid paradigm for efficient inference. D2F can be implemented with an asymmetric distillation process based on pre-trained dLLMs. We further propose a pipelined parallel decoding algorithm, which enables a trade-off between efficiency and efficacy. Empirically, D2F dLLMs achieve more than $\mathbf{2.5\times}$ inference speed than LLaMA3 and Qwen2.5 on GSM8K. Compared to vanilla dLLMs like LLaDA and Dream, the acceleration can be more than $\mathbf{50\times}$ while maintaining comparable output quality. The code is available at this https URL. 

---
# From Values to Tokens: An LLM-Driven Framework for Context-aware Time Series Forecasting via Symbolic Discretization 

**Authors**: Xiaoyu Tao, Shilong Zhang, Mingyue Cheng, Daoyu Wang, Tingyue Pan, Bokai Pan, Changqing Zhang, Shijin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09191)  

**Abstract**: Time series forecasting plays a vital role in supporting decision-making across a wide range of critical applications, including energy, healthcare, and finance. Despite recent advances, forecasting accuracy remains limited due to the challenge of integrating historical numerical sequences with contextual features, which often comprise unstructured textual data. To address this challenge, we propose TokenCast, an LLM-driven framework that leverages language-based symbolic representations as a unified intermediary for context-aware time series forecasting. Specifically, TokenCast employs a discrete tokenizer to transform continuous numerical sequences into temporal tokens, enabling structural alignment with language-based inputs. To bridge the semantic gap between modalities, both temporal and contextual tokens are embedded into a shared representation space via a pre-trained large language model (LLM), further optimized with autoregressive generative objectives. Building upon this unified semantic space, the aligned LLM is subsequently fine-tuned in a supervised manner to predict future temporal tokens, which are then decoded back into the original numerical space. Extensive experiments on diverse real-world datasets enriched with contextual features demonstrate the effectiveness and generalizability of TokenCast. 

---
# Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks 

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09190)  

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns. 

---
# Hybrid(Transformer+CNN)-based Polyp Segmentation 

**Authors**: Madan Baduwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.09189)  

**Abstract**: Colonoscopy is still the main method of detection and segmentation of colonic polyps, and recent advancements in deep learning networks such as U-Net, ResUNet, Swin-UNet, and PraNet have made outstanding performance in polyp segmentation. Yet, the problem is extremely challenging due to high variation in size, shape, endoscopy types, lighting, imaging protocols, and ill-defined boundaries (fluid, folds) of the polyps, rendering accurate segmentation a challenging and problematic task. To address these critical challenges in polyp segmentation, we introduce a hybrid (Transformer + CNN) model that is crafted to enhance robustness against evolving polyp characteristics. Our hybrid architecture demonstrates superior performance over existing solutions, particularly in addressing two critical challenges: (1) accurate segmentation of polyps with ill-defined margins through boundary-aware attention mechanisms, and (2) robust feature extraction in the presence of common endoscopic artifacts, including specular highlights, motion blur, and fluid occlusions. Quantitative evaluations reveal significant improvements in segmentation accuracy (Recall improved by 1.76%, i.e., 0.9555, accuracy improved by 0.07%, i.e., 0.9849) and artifact resilience compared to state-of-the-art polyp segmentation methods. 

---
# RL-MoE: An Image-Based Privacy Preserving Approach In Intelligent Transportation System 

**Authors**: Abdolazim Rezaei, Mehdi Sookhak, Mahboobeh Haghparast  

**Link**: [PDF](https://arxiv.org/pdf/2508.09186)  

**Abstract**: The proliferation of AI-powered cameras in Intelligent Transportation Systems (ITS) creates a severe conflict between the need for rich visual data and the fundamental right to privacy. Existing privacy-preserving mechanisms, such as blurring or encryption, are often insufficient, creating an undesirable trade-off where either privacy is compromised against advanced reconstruction attacks or data utility is critically degraded. To resolve this impasse, we propose RL-MoE, a novel framework that transforms sensitive visual data into privacy-preserving textual descriptions, eliminating the need for direct image transmission. RL-MoE uniquely combines a Mixture-of-Experts (MoE) architecture for nuanced, multi-aspect scene decomposition with a Reinforcement Learning (RL) agent that optimizes the generated text for a dual objective of semantic accuracy and privacy preservation. Extensive experiments demonstrate that RL-MoE provides superior privacy protection, reducing the success rate of replay attacks to just 9.4\% on the CFP-FP dataset, while simultaneously generating richer textual content than baseline methods. Our work provides a practical and scalable solution for building trustworthy AI systems in privacy-sensitive domains, paving the way for more secure smart city and autonomous vehicle networks. 

---
# A Neurosymbolic Framework for Interpretable Cognitive Attack Detection in Augmented Reality 

**Authors**: Rongqian Chen, Allison Andreyev, Yanming Xiu, Mahdi Imani, Bin Li, Maria Gorlatova, Gang Tan, Tian Lan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09185)  

**Abstract**: Augmented Reality (AR) enriches perception by overlaying virtual elements on the physical world. Due to its growing popularity, cognitive attacks that alter AR content to manipulate users' semantic perception have received increasing attention. Existing detection methods often focus on visual changes, which are restricted to pixel- or image-level processing and lack semantic reasoning capabilities, or they rely on pre-trained vision-language models (VLMs), which function as black-box approaches with limited interpretability. In this paper, we present CADAR, a novel neurosymbolic approach for cognitive attack detection in AR. It fuses multimodal vision-language inputs using neural VLMs to obtain a symbolic perception-graph representation, incorporating prior knowledge, salience weighting, and temporal correlations. The model then enables particle-filter based statistical reasoning -- a sequential Monte Carlo method -- to detect cognitive attacks. Thus, CADAR inherits the adaptability of pre-trained VLM and the interpretability and reasoning rigor of particle filtering. Experiments on an extended AR cognitive attack dataset show accuracy improvements of up to 10.7% over strong baselines on challenging AR attack scenarios, underscoring the promise of neurosymbolic methods for effective and interpretable cognitive attack detection. 

---
# HiSTM: Hierarchical Spatiotemporal Mamba for Cellular Traffic Forecasting 

**Authors**: Zineddine Bettouche, Khalid Ali, Andreas Fischer, Andreas Kassler  

**Link**: [PDF](https://arxiv.org/pdf/2508.09184)  

**Abstract**: Cellular traffic forecasting is essential for network planning, resource allocation, or load-balancing traffic across cells. However, accurate forecasting is difficult due to intricate spatial and temporal patterns that exist due to the mobility of users. Existing AI-based traffic forecasting models often trade-off accuracy and computational efficiency. We present Hierarchical SpatioTemporal Mamba (HiSTM), which combines a dual spatial encoder with a Mamba-based temporal module and attention mechanism. HiSTM employs selective state space methods to capture spatial and temporal patterns in network traffic. In our evaluation, we use a real-world dataset to compare HiSTM against several baselines, showing a 29.4% MAE improvement over the STN baseline while using 94% fewer parameters. We show that the HiSTM generalizes well across different datasets and improves in accuracy over longer time-horizons. 

---
# Quantum-Efficient Reinforcement Learning Solutions for Last-Mile On-Demand Delivery 

**Authors**: Farzan Moosavi, Bilal Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2508.09183)  

**Abstract**: Quantum computation has demonstrated a promising alternative to solving the NP-hard combinatorial problems. Specifically, when it comes to optimization, classical approaches become intractable to account for large-scale solutions. Specifically, we investigate quantum computing to solve the large-scale Capacitated Pickup and Delivery Problem with Time Windows (CPDPTW). In this regard, a Reinforcement Learning (RL) framework augmented with a Parametrized Quantum Circuit (PQC) is designed to minimize the travel time in a realistic last-mile on-demand delivery. A novel problem-specific encoding quantum circuit with an entangling and variational layer is proposed. Moreover, Proximal Policy Optimization (PPO) and Quantum Singular Value Transformation (QSVT) are designed for comparison through numerical experiments, highlighting the superiority of the proposed method in terms of the scale of the solution and training complexity while incorporating the real-world constraints. 

---
# Long-Term Client Selection for Federated Learning with Non-IID Data: A Truthful Auction Approach 

**Authors**: Jinghong Tan, Zhian Liu, Kun Guo, Mingxiong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09181)  

**Abstract**: Federated learning (FL) provides a decentralized framework that enables universal model training through collaborative efforts on mobile nodes, such as smart vehicles in the Internet of Vehicles (IoV). Each smart vehicle acts as a mobile client, contributing to the process without uploading local data. This method leverages non-independent and identically distributed (non-IID) training data from different vehicles, influenced by various driving patterns and environmental conditions, which can significantly impact model convergence and accuracy. Although client selection can be a feasible solution for non-IID issues, it faces challenges related to selection metrics. Traditional metrics evaluate client data quality independently per round and require client selection after all clients complete local training, leading to resource wastage from unused training results. In the IoV context, where vehicles have limited connectivity and computational resources, information asymmetry in client selection risks clients submitting false information, potentially making the selection ineffective. To tackle these challenges, we propose a novel Long-term Client-Selection Federated Learning based on Truthful Auction (LCSFLA). This scheme maximizes social welfare with consideration of long-term data quality using a new assessment mechanism and energy costs, and the advised auction mechanism with a deposit requirement incentivizes client participation and ensures information truthfulness. We theoretically prove the incentive compatibility and individual rationality of the advised incentive mechanism. Experimental results on various datasets, including those from IoV scenarios, demonstrate its effectiveness in mitigating performance degradation caused by non-IID data. 

---
# scAGC: Learning Adaptive Cell Graphs with Contrastive Guidance for Single-Cell Clustering 

**Authors**: Huifa Li, Jie Fu, Xinlin Zhuang, Haolin Yang, Xinpeng Ling, Tong Cheng, Haochen xue, Imran Razzak, Zhili Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09180)  

**Abstract**: Accurate cell type annotation is a crucial step in analyzing single-cell RNA sequencing (scRNA-seq) data, which provides valuable insights into cellular heterogeneity. However, due to the high dimensionality and prevalence of zero elements in scRNA-seq data, traditional clustering methods face significant statistical and computational challenges. While some advanced methods use graph neural networks to model cell-cell relationships, they often depend on static graph structures that are sensitive to noise and fail to capture the long-tailed distribution inherent in single-cell this http URL address these limitations, we propose scAGC, a single-cell clustering method that learns adaptive cell graphs with contrastive guidance. Our approach optimizes feature representations and cell graphs simultaneously in an end-to-end manner. Specifically, we introduce a topology-adaptive graph autoencoder that leverages a differentiable Gumbel-Softmax sampling strategy to dynamically refine the graph structure during training. This adaptive mechanism mitigates the problem of a long-tailed degree distribution by promoting a more balanced neighborhood structure. To model the discrete, over-dispersed, and zero-inflated nature of scRNA-seq data, we integrate a Zero-Inflated Negative Binomial (ZINB) loss for robust feature reconstruction. Furthermore, a contrastive learning objective is incorporated to regularize the graph learning process and prevent abrupt changes in the graph topology, ensuring stability and enhancing convergence. Comprehensive experiments on 9 real scRNA-seq datasets demonstrate that scAGC consistently outperforms other state-of-the-art methods, yielding the best NMI and ARI scores on 9 and 7 datasets, this http URL code is available at Anonymous Github. 

---
# IAD-R1: Reinforcing Consistent Reasoning in Industrial Anomaly Detection 

**Authors**: Yanhui Li, Yunkang Cao, Chengliang Liu, Yuan Xiong, Xinghui Dong, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09178)  

**Abstract**: Industrial anomaly detection is a critical component of modern manufacturing, yet the scarcity of defective samples restricts traditional detection methods to scenario-specific applications. Although Vision-Language Models (VLMs) demonstrate significant advantages in generalization capabilities, their performance in industrial anomaly detection remains limited. To address this challenge, we propose IAD-R1, a universal post-training framework applicable to VLMs of different architectures and parameter scales, which substantially enhances their anomaly detection capabilities. IAD-R1 employs a two-stage training strategy: the Perception Activation Supervised Fine-Tuning (PA-SFT) stage utilizes a meticulously constructed high-quality Chain-of-Thought dataset (Expert-AD) for training, enhancing anomaly perception capabilities and establishing reasoning-to-answer correlations; the Structured Control Group Relative Policy Optimization (SC-GRPO) stage employs carefully designed reward functions to achieve a capability leap from "Anomaly Perception" to "Anomaly Interpretation". Experimental results demonstrate that IAD-R1 achieves significant improvements across 7 VLMs, attaining up to 43.3% enhancement in average accuracy on 6 industrial anomaly detection benchmark datasets. Notably, the 0.5B parameter model trained with IAD-R1 surpasses commercial models including GPT-4.1 and Claude-Sonnet-4 in zero-shot settings, demonstrating the effectiveness and superiority of IAD-R1. The dataset, code, and all model weights will be publicly available at this https URL. 

---
# Generative Artificial Intelligence in Medical Imaging: Foundations, Progress, and Clinical Translation 

**Authors**: Xuanru Zhou, Cheng Li, Shuqiang Wang, Ye Li, Tao Tan, Hairong Zheng, Shanshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09177)  

**Abstract**: Generative artificial intelligence (AI) is rapidly transforming medical imaging by enabling capabilities such as data synthesis, image enhancement, modality translation, and spatiotemporal modeling. This review presents a comprehensive and forward-looking synthesis of recent advances in generative modeling including generative adversarial networks (GANs), variational autoencoders (VAEs), diffusion models, and emerging multimodal foundation architectures and evaluates their expanding roles across the clinical imaging continuum. We systematically examine how generative AI contributes to key stages of the imaging workflow, from acquisition and reconstruction to cross-modality synthesis, diagnostic support, and treatment planning. Emphasis is placed on both retrospective and prospective clinical scenarios, where generative models help address longstanding challenges such as data scarcity, standardization, and integration across modalities. To promote rigorous benchmarking and translational readiness, we propose a three-tiered evaluation framework encompassing pixel-level fidelity, feature-level realism, and task-level clinical relevance. We also identify critical obstacles to real-world deployment, including generalization under domain shift, hallucination risk, data privacy concerns, and regulatory hurdles. Finally, we explore the convergence of generative AI with large-scale foundation models, highlighting how this synergy may enable the next generation of scalable, reliable, and clinically integrated imaging systems. By charting technical progress and translational pathways, this review aims to guide future research and foster interdisciplinary collaboration at the intersection of AI, medicine, and biomedical engineering. 

---
# DQT: Dynamic Quantization Training via Dequantization-Free Nested Integer Arithmetic 

**Authors**: Hazem Hesham Yousef Shalby, Fabrizio Pittorino, Francesca Palermo, Diana Trojaniello, Manuel Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2508.09176)  

**Abstract**: The deployment of deep neural networks on resource-constrained devices relies on quantization. While static, uniform quantization applies a fixed bit-width to all inputs, it fails to adapt to their varying complexity. Dynamic, instance-based mixed-precision quantization promises a superior accuracy-efficiency trade-off by allocating higher precision only when needed. However, a critical bottleneck remains: existing methods require a costly dequantize-to-float and requantize-to-integer cycle to change precision, breaking the integer-only hardware paradigm and compromising performance gains. This paper introduces Dynamic Quantization Training (DQT), a novel framework that removes this bottleneck. At the core of DQT is a nested integer representation where lower-precision values are bit-wise embedded within higher-precision ones. This design, coupled with custom integer-only arithmetic, allows for on-the-fly bit-width switching through a near-zero-cost bit-shift operation. This makes DQT the first quantization framework to enable both dequantization-free static mixed-precision of the backbone network, and truly efficient dynamic, instance-based quantization through a lightweight controller that decides at runtime how to quantize each layer. We demonstrate DQT state-of-the-art performance on ResNet18 on CIFAR-10 and ResNet50 on ImageNet. On ImageNet, our 4-bit dynamic ResNet50 achieves 77.00% top-1 accuracy, an improvement over leading static (LSQ, 76.70%) and dynamic (DQNET, 76.94%) methods at a comparable BitOPs budget. Crucially, DQT achieves this with a bit-width transition cost of only 28.3M simple bit-shift operations, a drastic improvement over the 56.6M costly Multiply-Accumulate (MAC) floating-point operations required by previous dynamic approaches - unlocking a new frontier in efficient, adaptive AI. 

---
# A Context-aware Attention and Graph Neural Network-based Multimodal Framework for Misogyny Detection 

**Authors**: Mohammad Zia Ur Rehman, Sufyaan Zahoor, Areeb Manzoor, Musharaf Maqbool, Nagendra Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.09175)  

**Abstract**: A substantial portion of offensive content on social media is directed towards women. Since the approaches for general offensive content detection face a challenge in detecting misogynistic content, it requires solutions tailored to address offensive content against women. To this end, we propose a novel multimodal framework for the detection of misogynistic and sexist content. The framework comprises three modules: the Multimodal Attention module (MANM), the Graph-based Feature Reconstruction Module (GFRM), and the Content-specific Features Learning Module (CFLM). The MANM employs adaptive gating-based multimodal context-aware attention, enabling the model to focus on relevant visual and textual information and generating contextually relevant features. The GFRM module utilizes graphs to refine features within individual modalities, while the CFLM focuses on learning text and image-specific features such as toxicity features and caption features. Additionally, we curate a set of misogynous lexicons to compute the misogyny-specific lexicon score from the text. We apply test-time augmentation in feature space to better generalize the predictions on diverse inputs. The performance of the proposed approach has been evaluated on two multimodal datasets, MAMI and MMHS150K, with 11,000 and 13,494 samples, respectively. The proposed method demonstrates an average improvement of 10.17% and 8.88% in macro-F1 over existing methods on the MAMI and MMHS150K datasets, respectively. 

---
# FedMP: Tackling Medical Feature Heterogeneity in Federated Learning from a Manifold Perspective 

**Authors**: Zhekai Zhou, Shudong Liu, Zhaokun Zhou, Yang Liu, Qiang Yang, Yuesheng Zhu, Guibo Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.09174)  

**Abstract**: Federated learning (FL) is a decentralized machine learning paradigm in which multiple clients collaboratively train a shared model without sharing their local private data. However, real-world applications of FL frequently encounter challenges arising from the non-identically and independently distributed (non-IID) local datasets across participating clients, which is particularly pronounced in the field of medical imaging, where shifts in image feature distributions significantly hinder the global model's convergence and performance. To address this challenge, we propose FedMP, a novel method designed to enhance FL under non-IID scenarios. FedMP employs stochastic feature manifold completion to enrich the training space of individual client classifiers, and leverages class-prototypes to guide the alignment of feature manifolds across clients within semantically consistent subspaces, facilitating the construction of more distinct decision boundaries. We validate the effectiveness of FedMP on multiple medical imaging datasets, including those with real-world multi-center distributions, as well as on a multi-domain natural image dataset. The experimental results demonstrate that FedMP outperforms existing FL algorithms. Additionally, we analyze the impact of manifold dimensionality, communication efficiency, and privacy implications of feature exposure in our method. 

---
# webMCP: Efficient AI-Native Client-Side Interaction for Agent-Ready Web Design 

**Authors**: D. Perera  

**Link**: [PDF](https://arxiv.org/pdf/2508.09171)  

**Abstract**: Current AI agents create significant barriers for users by requiring extensive processing to understand web pages, making AI-assisted web interaction slow and expensive. This paper introduces webMCP (Web Machine Context & Procedure), a client-side standard that embeds structured interaction metadata directly into web pages, enabling more efficient human-AI collaboration on existing websites. webMCP transforms how AI agents understand web interfaces by providing explicit mappings between page elements and user actions. Instead of processing entire HTML documents, agents can access pre-structured interaction data, dramatically reducing computational overhead while maintaining task accuracy. A comprehensive evaluation across 1,890 real API calls spanning online shopping, authentication, and content management scenarios demonstrates webMCP reduces processing requirements by 67.6% while maintaining 97.9% task success rates compared to 98.8% for traditional approaches. Users experience significantly lower costs (34-63% reduction) and faster response times across diverse web interactions. Statistical analysis confirms these improvements are highly significant across multiple AI models. An independent WordPress deployment study validates practical applicability, showing consistent improvements across real-world content management workflows. webMCP requires no server-side modifications, making it deployable across millions of existing websites without technical barriers. These results establish webMCP as a viable solution for making AI web assistance more accessible and sustainable, addressing the critical gap between user interaction needs and AI computational requirements in production environments. 

---
# Multimodal RAG Enhanced Visual Description 

**Authors**: Amit Kumar Jaiswal, Haiming Liu, Ingo Frommholz  

**Link**: [PDF](https://arxiv.org/pdf/2508.09170)  

**Abstract**: Textual descriptions for multimodal inputs entail recurrent refinement of queries to produce relevant output images. Despite efforts to address challenges such as scaling model size and data volume, the cost associated with pre-training and fine-tuning remains substantial. However, pre-trained large multimodal models (LMMs) encounter a modality gap, characterised by a misalignment between textual and visual representations within a common embedding space. Although fine-tuning can potentially mitigate this gap, it is typically expensive and impractical due to the requirement for extensive domain-driven data. To overcome this challenge, we propose a lightweight training-free approach utilising Retrieval-Augmented Generation (RAG) to extend across the modality using a linear mapping, which can be computed efficiently. During inference, this mapping is applied to images embedded by an LMM enabling retrieval of closest textual descriptions from the training set. These textual descriptions, in conjunction with an instruction, cater as an input prompt for the language model to generate new textual descriptions. In addition, we introduce an iterative technique for distilling the mapping by generating synthetic descriptions via the language model facilitating optimisation for standard utilised image description measures. Experimental results on two benchmark multimodal datasets demonstrate significant improvements. 

---
# Energy-Efficient Stochastic Computing (SC) Neural Networks for Internet of Things Devices With Layer-Wise Adjustable Sequence Length (ASL) 

**Authors**: Ziheng Wang, Pedro Reviriego, Farzad Niknia, Zhen Gao, Javier Conde, Shanshan Liu, Fabrizio Lombardi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09163)  

**Abstract**: Stochastic computing (SC) has emerged as an efficient low-power alternative for deploying neural networks (NNs) in resource-limited scenarios, such as the Internet of Things (IoT). By encoding values as serial bitstreams, SC significantly reduces energy dissipation compared to conventional floating-point (FP) designs; however, further improvement of layer-wise mixed-precision implementation for SC remains unexplored. This article introduces Adjustable Sequence Length (ASL), a novel scheme that applies mixed-precision concepts specifically to SC NNs. By introducing an operator-norm-based theoretical model, this article shows that truncation noise can cumulatively propagate through the layers by the estimated amplification factors. An extended sensitivity analysis is presented, using random forest (RF) regression to evaluate multilayer truncation effects and validate the alignment of theoretical predictions with practical network behaviors. To accommodate different application scenarios, this article proposes two truncation strategies (coarse-grained and fine-grained), which apply diverse sequence length configurations at each layer. Evaluations on a pipelined SC MLP synthesized at 32nm demonstrate that ASL can reduce energy and latency overheads by up to over 60% with negligible accuracy loss. It confirms the feasibility of the ASL scheme for IoT applications and highlights the distinct advantages of mixed-precision truncation in SC designs. 

---
# Physics-Guided Memory Network for Building Energy Modeling 

**Authors**: Muhammad Umair Danish, Kashif Ali, Kamran Siddiqui, Katarina Grolinger  

**Link**: [PDF](https://arxiv.org/pdf/2508.09161)  

**Abstract**: Accurate energy consumption forecasting is essential for efficient resource management and sustainability in the building sector. Deep learning models are highly successful but struggle with limited historical data and become unusable when historical data are unavailable, such as in newly constructed buildings. On the other hand, physics-based models, such as EnergyPlus, simulate energy consumption without relying on historical data but require extensive building parameter specifications and considerable time to model a building. This paper introduces a Physics-Guided Memory Network (PgMN), a neural network that integrates predictions from deep learning and physics-based models to address their limitations. PgMN comprises a Parallel Projection Layers to process incomplete inputs, a Memory Unit to account for persistent biases, and a Memory Experience Module to optimally extend forecasts beyond their input range and produce output. Theoretical evaluation shows that components of PgMN are mathematically valid for performing their respective tasks. The PgMN was evaluated on short-term energy forecasting at an hourly resolution, critical for operational decision-making in smart grid and smart building systems. Experimental validation shows accuracy and applicability of PgMN in diverse scenarios such as newly constructed buildings, missing data, sparse historical data, and dynamic infrastructure changes. This paper provides a promising solution for energy consumption forecasting in dynamic building environments, enhancing model applicability in scenarios where historical data are limited or unavailable or when physics-based models are inadequate. 

---
# Agoran: An Agentic Open Marketplace for 6G RAN Automation 

**Authors**: Ilias Chatzistefanidis, Navid Nikaein, Andrea Leone, Ali Maatouk, Leandros Tassioulas, Roberto Morabito, Ioannis Pitsiorlas, Marios Kountouris  

**Link**: [PDF](https://arxiv.org/pdf/2508.09159)  

**Abstract**: Next-generation mobile networks must reconcile the often-conflicting goals of multiple service owners. However, today's network slice controllers remain rigid, policy-bound, and unaware of the business context. We introduce Agoran Service and Resource Broker (SRB), an agentic marketplace that brings stakeholders directly into the operational loop. Inspired by the ancient Greek agora, Agoran distributes authority across three autonomous AI branches: a Legislative branch that answers compliance queries using retrieval-augmented Large Language Models (LLMs); an Executive branch that maintains real-time situational awareness through a watcher-updated vector database; and a Judicial branch that evaluates each agent message with a rule-based Trust Score, while arbitrating LLMs detect malicious behavior and apply real-time incentives to restore trust. Stakeholder-side Negotiation Agents and the SRB-side Mediator Agent negotiate feasible, Pareto-optimal offers produced by a multi-objective optimizer, reaching a consensus intent in a single round, which is then deployed to Open and AI RAN controllers. Deployed on a private 5G testbed and evaluated with realistic traces of vehicle mobility, Agoran achieved significant gains: (i) a 37% increase in throughput of eMBB slices, (ii) a 73% reduction in latency of URLLC slices, and concurrently (iii) an end-to-end 8.3% saving in PRB usage compared to a static baseline. An 1B-parameter Llama model, fine-tuned for five minutes on 100 GPT-4 dialogues, recovers approximately 80% of GPT-4.1's decision quality, while operating within 6 GiB of memory and converging in only 1.3 seconds. These results establish Agoran as a concrete, standards-aligned path toward ultra-flexible, stakeholder-centric 6G networks. A live demo is presented this https URL\&ab_channel=BubbleRAN. 

---
# EvaDrive: Evolutionary Adversarial Policy Optimization for End-to-End Autonomous Driving 

**Authors**: Siwen Jiao, Kangan Qian, Hao Ye, Yang Zhong, Ziang Luo, Sicong Jiang, Zilin Huang, Yangyi Fang, Jinyu Miao, Zheng Fu, Yunlong Wang, Kun Jiang, Diange Yang, Rui Fan, Baoyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.09158)  

**Abstract**: Autonomous driving faces significant challenges in achieving human-like iterative decision-making, which continuously generates, evaluates, and refines trajectory proposals. Current generation-evaluation frameworks isolate trajectory generation from quality assessment, preventing iterative refinement essential for planning, while reinforcement learning methods collapse multi-dimensional preferences into scalar rewards, obscuring critical trade-offs and yielding scalarization this http URL overcome these issues, we present EvaDrive, a novel multi-objective reinforcement learning framework that establishes genuine closed-loop co-evolution between trajectory generation and evaluation via adversarial optimization. EvaDrive frames trajectory planning as a multi-round adversarial game. In this game, a hierarchical generator continuously proposes candidate paths by combining autoregressive intent modeling for temporal causality with diffusion-based refinement for spatial flexibility. These proposals are then rigorously assessed by a trainable multi-objective critic that explicitly preserves diverse preference structures without collapsing them into a single scalarization this http URL adversarial interplay, guided by a Pareto frontier selection mechanism, enables iterative multi-round refinement, effectively escaping local optima while preserving trajectory this http URL experiments on NAVSIM and Bench2Drive benchmarks demonstrate SOTA performance, achieving 94.9 PDMS on NAVSIM v1 (surpassing DiffusionDrive by 6.8, DriveSuprim by 5.0, and TrajHF by 0.9) and 64.96 Driving Score on Bench2Drive. EvaDrive generates diverse driving styles via dynamic weighting without external preference data, introducing a closed-loop adversarial framework for human-like iterative decision-making, offering a novel scalarization-free trajectory optimization approach. 

---
# Physics-Constrained Fine-Tuning of Flow-Matching Models for Generation and Inverse Problems 

**Authors**: Jan Tauberschmidt, Sophie Fellenz, Sebastian J. Vollmer, Andrew B. Duncan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09156)  

**Abstract**: We present a framework for fine-tuning flow-matching generative models to enforce physical constraints and solve inverse problems in scientific systems. Starting from a model trained on low-fidelity or observational data, we apply a differentiable post-training procedure that minimizes weak-form residuals of governing partial differential equations (PDEs), promoting physical consistency and adherence to boundary conditions without distorting the underlying learned distribution. To infer unknown physical inputs, such as source terms, material parameters, or boundary data, we augment the generative process with a learnable latent parameter predictor and propose a joint optimization strategy. The resulting model produces physically valid field solutions alongside plausible estimates of hidden parameters, effectively addressing ill-posed inverse problems in a data-driven yet physicsaware manner. We validate our method on canonical PDE benchmarks, demonstrating improved satisfaction of PDE constraints and accurate recovery of latent coefficients. Our approach bridges generative modelling and scientific inference, opening new avenues for simulation-augmented discovery and data-efficient modelling of physical systems. 

---
# A Rolling Stone Gathers No Moss: Adaptive Policy Optimization for Stable Self-Evaluation in Large Multimodal Models 

**Authors**: Wenkai Wang, Hongcan Guo, Zheqi Lv, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09155)  

**Abstract**: Self-evaluation, a model's ability to assess the correctness of its own output, is crucial for Large Multimodal Models (LMMs) to achieve self-improvement in multi-turn conversations, yet largely absent in foundation models. Recent work has employed reinforcement learning (RL) to enhance self-evaluation; however, its fixed reward mechanism suffers from reward hacking when optimizing multiple training objectives, leading to model collapse. In this paper we propose AdaPO, an online reinforcement learning framework capable of adaptively adjusting training objective in real time according to the current training state for each task. Specifically, to mitigate reward hacking , AdaPO introduces an Adaptive Reward Model (ARM) and a Reward Aware Dynamic KL Regularization mechanism. ARM assesses the task's training state from the distribution of model generated multi-turn trajectories' performance. Reward Aware Dynamic KL replaces a fixed penalty with dynamic coefficients which is modulated by the reward gap between different multi-turn situations. Notably, our method automatically and smoothly adjusts its learning focus based on sub-tasks' training progress without manual intervention. Extensive experiments over 8 benchmarks and various models show that our method significantly enhances both direct reasoning and self-evaluation capability. We will release our code to contribute to the community. 

---
# Peer Effect Estimation in the Presence of Simultaneous Feedback and Unobserved Confounders 

**Authors**: Xiaojing Du, Jiuyong Li, Lin Liu, Debo Cheng, Thuc.Le  

**Link**: [PDF](https://arxiv.org/pdf/2508.09154)  

**Abstract**: Estimating peer causal effects within complex real-world networks such as social networks is challenging, primarily due to simultaneous feedback between peers and unobserved confounders. Existing methods either address unobserved confounders while ignoring the simultaneous feedback, or account for feedback but under restrictive linear assumptions, thus failing to obtain accurate peer effect estimation. In this paper, we propose DIG2RSI, a novel Deep learning framework which leverages I-G transformation (matrix operation) and 2SRI (an instrumental variable or IV technique) to address both simultaneous feedback and unobserved confounding, while accommodating complex, nonlinear and high-dimensional relationships. DIG2RSI first applies the I-G transformation to disentangle mutual peer influences and eliminate the bias due to the simultaneous feedback. To deal with unobserved confounding, we first construct valid IVs from network data. In stage 1 of 2RSI, we train a neural network on these IVs to predict peer exposure, and extract residuals as proxies for the unobserved confounders. In the stage 2, we fit a separate neural network augmented by an adversarial discriminator that incorporates these residuals as a control function and enforces the learned representation to contain no residual confounding signal. The expressive power of deep learning models in capturing complex non-linear relationships and adversarial debiasing enhances the effectiveness of DIG2RSI in eliminating bias from both feedback loops and hidden confounders. We prove consistency of our estimator under standard regularity conditions, ensuring asymptotic recovery of the true peer effect. Empirical results on two semi-synthetic benchmarks and a real-world dataset demonstrate that DIG2RSI outperforms existing approaches. 

---
# JustDense: Just using Dense instead of Sequence Mixer for Time Series analysis 

**Authors**: TaekHyun Park, Yongjae Lee, Daesan Park, Dohee Kim, Hyerim Bae  

**Link**: [PDF](https://arxiv.org/pdf/2508.09153)  

**Abstract**: Sequence and channel mixers, the core mechanism in sequence models, have become the de facto standard in time series analysis (TSA). However, recent studies have questioned the necessity of complex sequence mixers, such as attention mechanisms, demonstrating that simpler architectures can achieve comparable or even superior performance. This suggests that the benefits attributed to complex sequencemixers might instead emerge from other architectural or optimization factors. Based on this observation, we pose a central question: Are common sequence mixers necessary for time-series analysis? Therefore, we propose JustDense, an empirical study that systematically replaces sequence mixers in various well-established TSA models with dense layers. Grounded in the MatrixMixer framework, JustDense treats any sequence mixer as a mixing matrix and replaces it with a dense layer. This substitution isolates the mixing operation, enabling a clear theoretical foundation for understanding its role. Therefore, we conducted extensive experiments on 29 benchmarks covering five representative TSA tasks using seven state-of-the-art TSA models to address our research question. The results show that replacing sequence mixers with dense layers yields comparable or even superior performance. In the cases where dedicated sequence mixers still offer benefits, JustDense challenges the assumption that "deeper and more complex architectures are inherently better" in TSA. 

---
# 5G Core Fault Detection and Root Cause Analysis using Machine Learning and Generative AI 

**Authors**: Joseph H. R. Isaac, Harish Saradagam, Nallamothu Pardhasaradhi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09152)  

**Abstract**: With the advent of 5G networks and technologies, ensuring the integrity and performance of packet core traffic is paramount. During network analysis, test files such as Packet Capture (PCAP) files and log files will contain errors if present in the system that must be resolved for better overall network performance, such as connectivity strength and handover quality. Current methods require numerous person-hours to sort out testing results and find the faults. This paper presents a novel AI/ML-driven Fault Analysis (FA) Engine designed to classify successful and faulty frames in PCAP files, specifically within the 5G packet core. The FA engine analyses network traffic using natural language processing techniques to identify anomalies and inefficiencies, significantly reducing the effort time required and increasing efficiency. The FA Engine also suggests steps to fix the issue using Generative AI via a Large Language Model (LLM) trained on several 5G packet core documents. The engine explains the details of the error from the domain perspective using documents such as the 3GPP standards and user documents regarding the internal conditions of the tests. Test results on the ML models show high classification accuracy on the test dataset when trained with 80-20 splits for the successful and failed PCAP files. Future scopes include extending the AI engine to incorporate 4G network traffic and other forms of network data, such as log text files and multimodal systems. 

---
# Motif 2.6B Technical Report 

**Authors**: Junghwan Lim, Sungmin Lee, Dongseok Kim, Eunhwan Park, Hyunbyung Park, Junhyeok Lee, Wai Ting Cheung, Dahye Choi, Jaeheui Her, Jaeyeon Huh, Hanbin Jung, Changjin Kang, Beomgyu Kim, Jihwan Kim, Minjae Kim, Taehwan Kim, Youngrok Kim, Haesol Lee, Jeesoo Lee, Kungyu Lee, Dongpin Oh, Yeongjae Park, Bokki Ryu, Daewon Suh, Dongjoo Weon  

**Link**: [PDF](https://arxiv.org/pdf/2508.09148)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have revolutionized artificial intelligence, yet developing an effective foundational LLM that balances high performance with computational efficiency remains challenging, especially for emerging research groups. To address this gap, we introduce Motif-2.6B, a 2.6-billion-parameter foundation model designed to democratize advanced LLM capabilities. Motif-2.6B incorporates several innovative architectural enhancements, including Differential Attention and PolyNorm activation functions, which improve long-context comprehension, reduce hallucination, and enhance in-context learning capabilities. We rigorously tested multiple novel architectural components through extensive experimentation to determine the optimal architecture for Motif-2.6B. Comprehensive evaluations demonstrate that Motif-2.6B consistently meets or exceeds the performance of similarly sized state-of-the-art models across diverse benchmarks, showcasing its effectiveness, scalability, and real-world applicability. Through detailed experiments and tailored techniques, Motif-2.6B significantly advances the landscape of efficient, scalable, and powerful foundational LLMs, offering valuable insights and a robust foundation for future research and deployment. 

---
# Agentic TinyML for Intent-aware Handover in 6G Wireless Networks 

**Authors**: Alaa Saleh, Roberto Morabito, Sasu Tarkoma, Anders Lindgren, Susanna Pirttikangas, Lauri Lovén  

**Link**: [PDF](https://arxiv.org/pdf/2508.09147)  

**Abstract**: As 6G networks evolve into increasingly AI-driven, user-centric ecosystems, traditional reactive handover mechanisms demonstrate limitations, especially in mobile edge computing and autonomous agent-based service scenarios. This manuscript introduces WAAN, a cross-layer framework that enables intent-aware and proactive handovers by embedding lightweight TinyML agents as autonomous, negotiation-capable entities across heterogeneous edge nodes that contribute to intent propagation and network adaptation. To ensure continuity across mobility-induced disruptions, WAAN incorporates semi-stable rendezvous points that serve as coordination anchors for context transfer and state preservation. The framework's operational capabilities are demonstrated through a multimodal environmental control case study, highlighting its effectiveness in maintaining user experience under mobility. Finally, the article discusses key challenges and future opportunities associated with the deployment and evolution of WAAN. 

---
# To Theoretically Understand Transformer-Based In-Context Learning for Optimizing CSMA 

**Authors**: Shugang Hao, Hongbo Li, Lingjie Duan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09146)  

**Abstract**: The binary exponential backoff scheme is widely used in WiFi 7 and still incurs poor throughput performance under dynamic channel environments. Recent model-based approaches (e.g., non-persistent and $p$-persistent CSMA) simply optimize backoff strategies under a known and fixed node density, still leading to a large throughput loss due to inaccurate node density estimation. This paper is the first to propose LLM transformer-based in-context learning (ICL) theory for optimizing channel access. We design a transformer-based ICL optimizer to pre-collect collision-threshold data examples and a query collision case. They are constructed as a prompt as the input for the transformer to learn the pattern, which then generates a predicted contention window threshold (CWT). To train the transformer for effective ICL, we develop an efficient algorithm and guarantee a near-optimal CWT prediction within limited training steps. As it may be hard to gather perfect data examples for ICL in practice, we further extend to allow erroneous data input in the prompt. We prove that our optimizer maintains minimal prediction and throughput deviations from the optimal values. Experimental results on NS-3 further demonstrate our approach's fast convergence and near-optimal throughput over existing model-based and DRL-based approaches under unknown node densities. 

---
# Efficient Real-Time Aircraft ETA Prediction via Feature Tokenization Transformer 

**Authors**: Liping Huang, Yicheng Zhang, Yifang Yin, Sheng Zhang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09144)  

**Abstract**: Estimated time of arrival (ETA) for airborne aircraft in real-time is crucial for arrival management in aviation, particularly for runway sequencing. Given the rapidly changing airspace context, the ETA prediction efficiency is as important as its accuracy in a real-time arrival aircraft management system. In this study, we utilize a feature tokenization-based Transformer model to efficiently predict aircraft ETA. Feature tokenization projects raw inputs to latent spaces, while the multi-head self-attention mechanism in the Transformer captures important aspects of the projections, alleviating the need for complex feature engineering. Moreover, the Transformer's parallel computation capability allows it to handle ETA requests at a high frequency, i.e., 1HZ, which is essential for a real-time arrival management system. The model inputs include raw data, such as aircraft latitude, longitude, ground speed, theta degree for the airport, day and hour from track data, the weather context, and aircraft wake turbulence category. With a data sampling rate of 1HZ, the ETA prediction is updated every second. We apply the proposed aircraft ETA prediction approach to Singapore Changi Airport (ICAO Code: WSSS) using one-month Automatic Dependent Surveillance-Broadcast (ADS-B) data from October 1 to October 31, 2022. In the experimental evaluation, the ETA modeling covers all aircraft within a range of 10NM to 300NM from WSSS. The results show that our proposed method method outperforms the commonly used boosting tree based model, improving accuracy by 7\% compared to XGBoost, while requiring only 39\% of its computing time. Experimental results also indicate that, with 40 aircraft in the airspace at a given timestamp, the ETA inference time is only 51.7 microseconds, making it promising for real-time arrival management systems. 

---
# Bayesian-Driven Graph Reasoning for Active Radio Map Construction 

**Authors**: Wenlihan Lu, Shijian Gao, Miaowen Wen, Yuxuan Liang, Chan-Byoung Chae, H. Vincent Poor  

**Link**: [PDF](https://arxiv.org/pdf/2508.09142)  

**Abstract**: With the emergence of the low-altitude economy, radio maps have become essential for ensuring reliable wireless connectivity to aerial platforms. Autonomous aerial agents are commonly deployed for data collection using waypoint-based navigation; however, their limited battery capacity significantly constrains coverage and efficiency. To address this, we propose an uncertainty-aware radio map (URAM) reconstruction framework that explicitly leverages graph-based reasoning tailored for waypoint navigation. Our approach integrates two key deep learning components: (1) a Bayesian neural network that estimates spatial uncertainty in real time, and (2) an attention-based reinforcement learning policy that performs global reasoning over a probabilistic roadmap, using uncertainty estimates to plan informative and energy-efficient trajectories. This graph-based reasoning enables intelligent, non-myopic trajectory planning, guiding agents toward the most informative regions while satisfying safety constraints. Experimental results show that URAM improves reconstruction accuracy by up to 34% over existing baselines. 

---
# User-Intent-Driven Semantic Communication via Adaptive Deep Understanding 

**Authors**: Peigen Ye, Jingpu Duan, Hongyang Du, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.05884)  

**Abstract**: Semantic communication focuses on transmitting task-relevant semantic information, aiming for intent-oriented communication. While existing systems improve efficiency by extracting key semantics, they still fail to deeply understand and generalize users' real intentions. To overcome this, we propose a user-intention-driven semantic communication system that interprets diverse abstract intents. First, we integrate a multi-modal large model as semantic knowledge base to generate user-intention prior. Next, a mask-guided attention module is proposed to effectively highlight critical semantic regions. Further, a channel state awareness module ensures adaptive, robust transmission across varying channel conditions. Extensive experiments demonstrate that our system achieves deep intent understanding and outperforms DeepJSCC, e.g., under a Rayleigh channel at an SNR of 5 dB, it achieves improvements of 8%, 6%, and 19% in PSNR, SSIM, and LPIPS, respectively. 

---
# QuickGrasp: Lightweight Antipodal Grasp Planning with Point Clouds 

**Authors**: Navin Sriram Ravie, Keerthi Vasan M, Asokan Thondiyath, Bijo Sebastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.19716)  

**Abstract**: Grasping has been a long-standing challenge in facilitating the final interface between a robot and the environment. As environments and tasks become complicated, the need to embed higher intelligence to infer from the surroundings and act on them has become necessary. Although most methods utilize techniques to estimate grasp pose by treating the problem via pure sampling-based approaches in the six-degree-of-freedom space or as a learning problem, they usually fail in real-life settings owing to poor generalization across domains. In addition, the time taken to generate the grasp plan and the lack of repeatability, owing to sampling inefficiency and the probabilistic nature of existing grasp planning approaches, severely limits their application in real-world tasks. This paper presents a lightweight analytical approach towards robotic grasp planning, particularly antipodal grasps, with little to no sampling in the six-degree-of-freedom space. The proposed grasp planning algorithm is formulated as an optimization problem towards estimating grasp points on the object surface instead of directly estimating the end-effector pose. To this extent, a soft-region-growing algorithm is presented for effective plane segmentation, even in the case of curved surfaces. An optimization-based quality metric is then used for the evaluation of grasp points to ensure indirect force closure. The proposed grasp framework is compared with the existing state-of-the-art grasp planning approach, Grasp pose detection (GPD), as a baseline over multiple simulated objects. The effectiveness of the proposed approach in comparison to GPD is also evaluated in a real-world setting using image and point-cloud data, with the planned grasps being executed using a ROBOTIQ gripper and UR5 manipulator. 

---
