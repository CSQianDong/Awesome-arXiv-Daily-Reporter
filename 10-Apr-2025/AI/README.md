# AssistanceZero: Scalably Solving Assistance Games 

**Authors**: Cassidy Laidlaw, Eli Bronstein, Timothy Guo, Dylan Feng, Lukas Berglund, Justin Svegliato, Stuart Russell, Anca Dragan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07091)  

**Abstract**: Assistance games are a promising alternative to reinforcement learning from human feedback (RLHF) for training AI assistants. Assistance games resolve key drawbacks of RLHF, such as incentives for deceptive behavior, by explicitly modeling the interaction between assistant and user as a two-player game where the assistant cannot observe their shared goal. Despite their potential, assistance games have only been explored in simple settings. Scaling them to more complex environments is difficult because it requires both solving intractable decision-making problems under uncertainty and accurately modeling human users' behavior. We present the first scalable approach to solving assistance games and apply it to a new, challenging Minecraft-based assistance game with over $10^{400}$ possible goals. Our approach, AssistanceZero, extends AlphaZero with a neural network that predicts human actions and rewards, enabling it to plan under uncertainty. We show that AssistanceZero outperforms model-free RL algorithms and imitation learning in the Minecraft-based assistance game. In a human study, our AssistanceZero-trained assistant significantly reduces the number of actions participants take to complete building tasks in Minecraft. Our results suggest that assistance games are a tractable framework for training effective AI assistants in complex environments. Our code and models are available at this https URL. 

---
# SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills 

**Authors**: Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.07079)  

**Abstract**: To survive and thrive in complex environments, humans have evolved sophisticated self-improvement mechanisms through environment exploration, hierarchical abstraction of experiences into reuseable skills, and collaborative construction of an ever-growing skill repertoire. Despite recent advancements, autonomous web agents still lack crucial self-improvement capabilities, struggling with procedural knowledge abstraction, refining skills, and skill composition. In this work, we introduce SkillWeaver, a skill-centric framework enabling agents to self-improve by autonomously synthesizing reusable skills as APIs. Given a new website, the agent autonomously discovers skills, executes them for practice, and distills practice experiences into robust APIs. Iterative exploration continually expands a library of lightweight, plug-and-play APIs, significantly enhancing the agent's capabilities. Experiments on WebArena and real-world websites demonstrate the efficacy of SkillWeaver, achieving relative success rate improvements of 31.8% and 39.8%, respectively. Additionally, APIs synthesized by strong agents substantially enhance weaker agents through transferable skills, yielding improvements of up to 54.3% on WebArena. These results demonstrate the effectiveness of honing diverse website interactions into APIs, which can be seamlessly shared among various web agents. 

---
# $Π$-NeSy: A Possibilistic Neuro-Symbolic Approach 

**Authors**: Ismaïl Baaj, Pierre Marquis  

**Link**: [PDF](https://arxiv.org/pdf/2504.07055)  

**Abstract**: In this article, we introduce a neuro-symbolic approach that combines a low-level perception task performed by a neural network with a high-level reasoning task performed by a possibilistic rule-based system. The goal is to be able to derive for each input instance the degree of possibility that it belongs to a target (meta-)concept. This (meta-)concept is connected to intermediate concepts by a possibilistic rule-based system. The probability of each intermediate concept for the input instance is inferred using a neural network. The connection between the low-level perception task and the high-level reasoning task lies in the transformation of neural network outputs modeled by probability distributions (through softmax activation) into possibility distributions. The use of intermediate concepts is valuable for the explanation purpose: using the rule-based system, the classification of an input instance as an element of the (meta-)concept can be justified by the fact that intermediate concepts have been recognized.
From the technical side, our contribution consists of the design of efficient methods for defining the matrix relation and the equation system associated with a possibilistic rule-based system. The corresponding matrix and equation are key data structures used to perform inferences from a possibilistic rule-based system and to learn the values of the rule parameters in such a system according to a training data sample. Furthermore, leveraging recent results on the handling of inconsistent systems of fuzzy relational equations, an approach for learning rule parameters according to multiple training data samples is presented. Experiments carried out on the MNIST addition problems and the MNIST Sudoku puzzles problems highlight the effectiveness of our approach compared with state-of-the-art neuro-symbolic ones. 

---
# Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration 

**Authors**: Kostas Hatalis, Despina Christou, Vyshnavi Kondapalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.06943)  

**Abstract**: Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents. 

---
# FamilyTool: A Multi-hop Personalized Tool Use Benchmark 

**Authors**: Yuxin Wang, Yiran Guo, Yining Zheng, Zhangyue Yin, Shuo Chen, Jie Yang, Jiajun Chen, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06766)  

**Abstract**: The integration of tool learning with Large Language Models (LLMs) has expanded their capabilities in handling complex tasks by leveraging external tools. However, existing benchmarks for tool learning inadequately address critical real-world personalized scenarios, particularly those requiring multi-hop reasoning and inductive knowledge adaptation in dynamic environments. To bridge this gap, we introduce FamilyTool, a novel benchmark grounded in a family-based knowledge graph (KG) that simulates personalized, multi-hop tool use scenarios. FamilyTool challenges LLMs with queries spanning 1 to 3 relational hops (e.g., inferring familial connections and preferences) and incorporates an inductive KG setting where models must adapt to unseen user preferences and relationships without re-training, a common limitation in prior approaches that compromises generalization. We further propose KGETool: a simple KG-augmented evaluation pipeline to systematically assess LLMs' tool use ability in these settings. Experiments reveal significant performance gaps in state-of-the-art LLMs, with accuracy dropping sharply as hop complexity increases and inductive scenarios exposing severe generalization deficits. These findings underscore the limitations of current LLMs in handling personalized, evolving real-world contexts and highlight the urgent need for advancements in tool-learning frameworks. FamilyTool serves as a critical resource for evaluating and advancing LLM agents' reasoning, adaptability, and scalability in complex, dynamic environments. Code and dataset are available at Github. 

---
# Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis 

**Authors**: Umakanta Maharana, Sarthak Verma, Avarna Agarwal, Prakashini Mruthyunjaya, Dwarikanath Mahapatra, Sakir Ahmed, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2504.06581)  

**Abstract**: Large language models (LLMs) offer a promising pre-screening tool, improving early disease detection and providing enhanced healthcare access for underprivileged communities. The early diagnosis of various diseases continues to be a significant challenge in healthcare, primarily due to the nonspecific nature of early symptoms, the shortage of expert medical practitioners, and the need for prolonged clinical evaluations, all of which can delay treatment and adversely affect patient outcomes. With impressive accuracy in prediction across a range of diseases, LLMs have the potential to revolutionize clinical pre-screening and decision-making for various medical conditions. In this work, we study the diagnostic capability of LLMs for Rheumatoid Arthritis (RA) with real world patients data. Patient data was collected alongside diagnoses from medical experts, and the performance of LLMs was evaluated in comparison to expert diagnoses for RA disease prediction. We notice an interesting pattern in disease diagnosis and find an unexpected \textit{misalignment between prediction and explanation}. We conduct a series of multi-round analyses using different LLM agents. The best-performing model accurately predicts rheumatoid arthritis (RA) diseases approximately 95\% of the time. However, when medical experts evaluated the reasoning generated by the model, they found that nearly 68\% of the reasoning was incorrect. This study highlights a clear misalignment between LLMs high prediction accuracy and its flawed reasoning, raising important questions about relying on LLM explanations in clinical settings. \textbf{LLMs provide incorrect reasoning to arrive at the correct answer for RA disease diagnosis.} 

---
# Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill? 

**Authors**: Chenrui Fan, Ming Li, Lichao Sun, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06514)  

**Abstract**: We find that the response length of reasoning LLMs, whether trained by reinforcement learning or supervised learning, drastically increases for ill-posed questions with missing premises (MiP), ending up with redundant and ineffective thinking. This newly introduced scenario exacerbates the general overthinking issue to a large extent, which we name as the MiP-Overthinking. Such failures are against the ``test-time scaling law'' but have been widely observed on multiple datasets we curated with MiP, indicating the harm of cheap overthinking and a lack of critical thinking. Surprisingly, LLMs not specifically trained for reasoning exhibit much better performance on the MiP scenario, producing much shorter responses that quickly identify ill-posed queries. This implies a critical flaw of the current training recipe for reasoning LLMs, which does not encourage efficient thinking adequately, leading to the abuse of thinking patterns. To further investigate the reasons behind such failures, we conduct fine-grained analyses of the reasoning length, overthinking patterns, and location of critical thinking on different types of LLMs. Moreover, our extended ablation study reveals that the overthinking is contagious through the distillation of reasoning models' responses. These results improve the understanding of overthinking and shed novel insights into mitigating the problem. 

---
# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning 

**Authors**: Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2504.07097)  

**Abstract**: Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models. 

---
# Are We Done with Object-Centric Learning? 

**Authors**: Alexander Rubinstein, Ameya Prabhu, Matthias Bethge, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.07092)  

**Abstract**: Object-centric learning (OCL) seeks to learn representations that only encode an object, isolated from other objects or background cues in a scene. This approach underpins various aims, including out-of-distribution (OOD) generalization, sample-efficient composition, and modeling of structured environments. Most research has focused on developing unsupervised mechanisms that separate objects into discrete slots in the representation space, evaluated using unsupervised object discovery. However, with recent sample-efficient segmentation models, we can separate objects in the pixel space and encode them independently. This achieves remarkable zero-shot performance on OOD object discovery benchmarks, is scalable to foundation models, and can handle a variable number of slots out-of-the-box. Hence, the goal of OCL methods to obtain object-centric representations has been largely achieved. Despite this progress, a key question remains: How does the ability to separate objects within a scene contribute to broader OCL objectives, such as OOD generalization? We address this by investigating the OOD generalization challenge caused by spurious background cues through the lens of OCL. We propose a novel, training-free probe called $\textbf{Object-Centric Classification with Applied Masks (OCCAM)}$, demonstrating that segmentation-based encoding of individual objects significantly outperforms slot-based OCL methods. However, challenges in real-world applications remain. We provide the toolbox for the OCL community to use scalable object-centric representations, and focus on practical applications and fundamental questions, such as understanding object perception in human cognition. Our code is available $\href{this https URL}{here}$. 

---
# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs 

**Authors**: Elan Markowitz, Krupa Galiya, Greg Ver Steeg, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07087)  

**Abstract**: Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks. 

---
# Self-Steering Language Models 

**Authors**: Gabriel Grand, Joshua B. Tenenbaum, Vikash K. Mansinghka, Alexander K. Lew, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2504.07081)  

**Abstract**: While test-time reasoning enables language models to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its abstract structure--both how to verify solutions and how to search for them. This paper introduces DisCIPL, a method for "self-steering" LMs where a Planner model generates a task-specific inference program that is executed by a population of Follower models. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. In decoupling planning from execution, our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs. 

---
# DeduCE: Deductive Consistency as a Framework to Evaluate LLM Reasoning 

**Authors**: Atharva Pandey, Kshitij Dubey, Rahul Sharma, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07080)  

**Abstract**: Despite great performance on Olympiad-level reasoning problems, frontier large language models can still struggle on high school math when presented with novel problems outside standard benchmarks. Going beyond final accuracy, we propose a deductive consistency metric to analyze chain-of-thought output from language models (LMs).Formally, deductive reasoning involves two subtasks: understanding a set of input premises and inferring the conclusions that follow from them. The proposed metric studies LMs' performance on these subtasks, with the goal of explaining LMs' reasoning errors on novel problems: how well do LMs understand input premises with increasing context lengths, and how well can they infer conclusions over multiple reasoning hops? Since existing benchmarks may be memorized, we develop a pipeline to evaluate LMs' deductive consistency on novel, perturbed versions of benchmark problems. On novel grade school math problems (GSM-8k), we find that LMs are fairly robust to increasing number of input premises, but suffer significant accuracy decay as the number of reasoning hops is increased. Interestingly, these errors are masked in the original benchmark as all models achieve near 100% accuracy. As we increase the number of solution steps using a synthetic dataset, prediction over multiple hops still remains the major source of error compared to understanding input premises. Other factors, such as shifts in language style or natural propagation of early errors do not explain the trends. Our analysis provides a new view to characterize LM reasoning -- as computations over a window of input premises and reasoning hops -- that can provide unified evaluation across problem domains. 

---
# HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification 

**Authors**: Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07069)  

**Abstract**: This paper introduces a comprehensive system for detecting hallucinations in large language model (LLM) outputs in enterprise settings. We present a novel taxonomy of LLM responses specific to hallucination in enterprise applications, categorizing them into context-based, common knowledge, enterprise-specific, and innocuous statements. Our hallucination detection model HDM-2 validates LLM responses with respect to both context and generally known facts (common knowledge). It provides both hallucination scores and word-level annotations, enabling precise identification of problematic content. To evaluate it on context-based and common-knowledge hallucinations, we introduce a new dataset HDMBench. Experimental results demonstrate that HDM-2 out-performs existing approaches across RagTruth, TruthfulQA, and HDMBench datasets. This work addresses the specific challenges of enterprise deployment, including computational efficiency, domain specialization, and fine-grained error identification. Our evaluation dataset, model weights, and inference code are publicly available. 

---
# RayFronts: Open-Set Semantic Ray Frontiers for Online Scene Understanding and Exploration 

**Authors**: Omar Alama, Avigyan Bhattacharya, Haoyang He, Seungchan Kim, Yuheng Qiu, Wenshan Wang, Cherie Ho, Nikhil Keetha, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2504.06994)  

**Abstract**: Open-set semantic mapping is crucial for open-world robots. Current mapping approaches either are limited by the depth range or only map beyond-range entities in constrained settings, where overall they fail to combine within-range and beyond-range observations. Furthermore, these methods make a trade-off between fine-grained semantics and efficiency. We introduce RayFronts, a unified representation that enables both dense and beyond-range efficient semantic mapping. RayFronts encodes task-agnostic open-set semantics to both in-range voxels and beyond-range rays encoded at map boundaries, empowering the robot to reduce search volumes significantly and make informed decisions both within & beyond sensory range, while running at 8.84 Hz on an Orin AGX. Benchmarking the within-range semantics shows that RayFronts's fine-grained image encoding provides 1.34x zero-shot 3D semantic segmentation performance while improving throughput by 16.5x. Traditionally, online mapping performance is entangled with other system components, complicating evaluation. We propose a planner-agnostic evaluation framework that captures the utility for online beyond-range search and exploration, and show RayFronts reduces search volume 2.2x more efficiently than the closest online baselines. 

---
# Enhancing Metabolic Syndrome Prediction with Hybrid Data Balancing and Counterfactuals 

**Authors**: Sanyam Paresh Shah, Abdullah Mamun, Shovito Barua Soumma, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06987)  

**Abstract**: Metabolic Syndrome (MetS) is a cluster of interrelated risk factors that significantly increases the risk of cardiovascular diseases and type 2 diabetes. Despite its global prevalence, accurate prediction of MetS remains challenging due to issues such as class imbalance, data scarcity, and methodological inconsistencies in existing studies. In this paper, we address these challenges by systematically evaluating and optimizing machine learning (ML) models for MetS prediction, leveraging advanced data balancing techniques and counterfactual analysis. Multiple ML models, including XGBoost, Random Forest, TabNet, etc., were trained and compared under various data balancing techniques such as random oversampling (ROS), SMOTE, ADASYN, and CTGAN. Additionally, we introduce MetaBoost, a novel hybrid framework that integrates SMOTE, ADASYN, and CTGAN, optimizing synthetic data generation through weighted averaging and iterative weight tuning to enhance the model's performance (achieving a 1.14% accuracy improvement over individual balancing techniques). A comprehensive counterfactual analysis is conducted to quantify feature-level changes required to shift individuals from high-risk to low-risk categories. The results indicate that blood glucose (50.3%) and triglycerides (46.7%) were the most frequently modified features, highlighting their clinical significance in MetS risk reduction. Additionally, probabilistic analysis shows elevated blood glucose (85.5% likelihood) and triglycerides (74.9% posterior probability) as the strongest predictors. This study not only advances the methodological rigor of MetS prediction but also provides actionable insights for clinicians and researchers, highlighting the potential of ML in mitigating the public health burden of metabolic syndrome. 

---
# RNN-Transducer-based Losses for Speech Recognition on Noisy Targets 

**Authors**: Vladimir Bataev  

**Link**: [PDF](https://arxiv.org/pdf/2504.06963)  

**Abstract**: Training speech recognition systems on noisy transcripts is a significant challenge in industrial pipelines, where datasets are enormous and ensuring accurate transcription for every instance is difficult. In this work, we introduce novel loss functions to mitigate the impact of transcription errors in RNN-Transducer models. Our Star-Transducer loss addresses deletion errors by incorporating "skip frame" transitions in the loss lattice, restoring over 90% of the system's performance compared to models trained with accurate transcripts. The Bypass-Transducer loss uses "skip token" transitions to tackle insertion errors, recovering more than 60% of the quality. Finally, the Target-Robust Transducer loss merges these approaches, offering robust performance against arbitrary errors. Experimental results demonstrate that the Target-Robust Transducer loss significantly improves RNN-T performance on noisy data by restoring over 70% of the quality compared to well-transcribed data. 

---
# Efficient Self-Supervised Learning for Earth Observation via Dynamic Dataset Curation 

**Authors**: Thomas Kerdreux, Alexandre Tuel, Quentin Febvre, Alexis Mouche, Bertrand Chapron  

**Link**: [PDF](https://arxiv.org/pdf/2504.06962)  

**Abstract**: Self-supervised learning (SSL) has enabled the development of vision foundation models for Earth Observation (EO), demonstrating strong transferability across diverse remote sensing tasks. While prior work has focused on network architectures and training strategies, the role of dataset curation, especially in balancing and diversifying pre-training datasets, remains underexplored. In EO, this challenge is amplified by the redundancy and heavy-tailed distributions common in satellite imagery, which can lead to biased representations and inefficient training.
In this work, we propose a dynamic dataset pruning strategy designed to improve SSL pre-training by maximizing dataset diversity and balance. Our method iteratively refines the training set without requiring a pre-existing feature extractor, making it well-suited for domains where curated datasets are limited or unavailable. We demonstrate our approach on the Sentinel-1 Wave Mode (WV) Synthetic Aperture Radar (SAR) archive, a challenging dataset dominated by ocean observations. We train models from scratch on the entire Sentinel-1 WV archive spanning 10 years. Across three downstream tasks, our results show that dynamic pruning improves both computational efficiency and representation quality, leading to stronger transferability.
We also release the weights of Nereus-SAR-1, the first model in the Nereus family, a series of foundation models for ocean observation and analysis using SAR imagery, at this http URL. 

---
# Adaptive Computation Pruning for the Forgetting Transformer 

**Authors**: Zhixuan Lin, Johan Obando-Ceron, Xu Owen He, Aaron Courville  

**Link**: [PDF](https://arxiv.org/pdf/2504.06949)  

**Abstract**: The recently proposed Forgetting Transformer (FoX) incorporates a forget gate into softmax attention and has shown consistently better or on-par performance compared to the standard RoPE-based Transformer. Notably, many attention heads in FoX tend to forget quickly, causing their output at each timestep to rely primarily on the local context. Based on this observation, we propose Adaptive Computation Pruning (ACP) for FoX, a method that dynamically prunes computations involving input-output dependencies that are strongly decayed by the forget gate. This is achieved using a dynamically set pruning threshold that ensures that the pruned attention weights remain negligible. We apply ACP to language model pretraining with FoX and show it consistently reduces the number of FLOPs in softmax attention by around 70% across different model sizes and context lengths, resulting in a roughly 10% to 35% improvement in training throughput. Furthermore, longer context lengths yield greater computational savings. All these speed improvements are achieved without any performance degradation. We also perform several analyses to provide deeper insights into our method, such as examining the pruning patterns and analyzing the distribution of FLOP savings across different attention heads. Our code is available at this https URL. 

---
# Beyond Tools: Generative AI as Epistemic Infrastructure in Education 

**Authors**: Bodong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06928)  

**Abstract**: As generative AI rapidly integrates into educational infrastructures worldwide, it transforms how knowledge gets created, validated, and shared, yet current discourse inadequately addresses its implications as epistemic infrastructure mediating teaching and learning. This paper investigates how AI systems function as epistemic infrastructures in education and their impact on human epistemic agency. Adopting a situated cognition perspective and following a value-sensitive design approach, the study conducts a technical investigation of two representative AI systems in educational settings, analyzing their impact on teacher practice across three dimensions: affordances for skilled epistemic actions, support for epistemic sensitivity, and implications for long-term habit formation. The analysis reveals that current AI systems inadequately support teachers' skilled epistemic actions, insufficiently foster epistemic sensitivity, and potentially cultivate problematic habits that prioritize efficiency over epistemic agency. To address these challenges, the paper recommends recognizing the infrastructural transformation occurring in education, developing AI environments that stimulate skilled actions while upholding epistemic norms, and involving educators in AI design processes -- recommendations aimed at fostering AI integration that aligns with core educational values and maintains human epistemic agency. 

---
# Are Vision-Language Models Ready for Dietary Assessment? Exploring the Next Frontier in AI-Powered Food Image Recognition 

**Authors**: Sergio Romero-Tapiador, Ruben Tolosana, Blanca Lacruz-Pleguezuelos, Laura Judith Marcos Zambrano, Guadalupe X.Bazán, Isabel Espinosa-Salinas, Julian Fierrez, Javier Ortega-Garcia, Enrique Carrillo de Santa Pau, Aythami Morales  

**Link**: [PDF](https://arxiv.org/pdf/2504.06925)  

**Abstract**: Automatic dietary assessment based on food images remains a challenge, requiring precise food detection, segmentation, and classification. Vision-Language Models (VLMs) offer new possibilities by integrating visual and textual reasoning. In this study, we evaluate six state-of-the-art VLMs (ChatGPT, Gemini, Claude, Moondream, DeepSeek, and LLaVA), analyzing their capabilities in food recognition at different levels. For the experimental framework, we introduce the FoodNExTDB, a unique food image database that contains 9,263 expert-labeled images across 10 categories (e.g., "protein source"), 62 subcategories (e.g., "poultry"), and 9 cooking styles (e.g., "grilled"). In total, FoodNExTDB includes 50k nutritional labels generated by seven experts who manually annotated all images in the database. Also, we propose a novel evaluation metric, Expert-Weighted Recall (EWR), that accounts for the inter-annotator variability. Results show that closed-source models outperform open-source ones, achieving over 90% EWR in recognizing food products in images containing a single product. Despite their potential, current VLMs face challenges in fine-grained food recognition, particularly in distinguishing subtle differences in cooking styles and visually similar food items, which limits their reliability for automatic dietary assessment. The FoodNExTDB database is publicly available at this https URL. 

---
# Longitudinal Assessment of Lung Lesion Burden in CT 

**Authors**: Tejas Sudharshan Mathai, Benjamin Hou, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.06924)  

**Abstract**: In the U.S., lung cancer is the second major cause of death. Early detection of suspicious lung nodules is crucial for patient treatment planning, management, and improving outcomes. Many approaches for lung nodule segmentation and volumetric analysis have been proposed, but few have looked at longitudinal changes in total lung tumor burden. In this work, we trained two 3D models (nnUNet) with and without anatomical priors to automatically segment lung lesions and quantified total lesion burden for each patient. The 3D model without priors significantly outperformed ($p < .001$) the model trained with anatomy priors. For detecting clinically significant lesions $>$ 1cm, a precision of 71.3\%, sensitivity of 68.4\%, and F1-score of 69.8\% was achieved. For segmentation, a Dice score of 77.1 $\pm$ 20.3 and Hausdorff distance error of 11.7 $\pm$ 24.1 mm was obtained. The median lesion burden was 6.4 cc (IQR: 2.1, 18.1) and the median volume difference between manual and automated measurements was 0.02 cc (IQR: -2.8, 1.2). Agreements were also evaluated with linear regression and Bland-Altman plots. The proposed approach can produce a personalized evaluation of the total tumor burden for a patient and facilitate interval change tracking over time. 

---
# Leveraging Anatomical Priors for Automated Pancreas Segmentation on Abdominal CT 

**Authors**: Anisa V. Prasad, Tejas Sudharshan Mathai, Pritam Mukherjee, Jianfei Liu, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2504.06921)  

**Abstract**: An accurate segmentation of the pancreas on CT is crucial to identify pancreatic pathologies and extract imaging-based biomarkers. However, prior research on pancreas segmentation has primarily focused on modifying the segmentation model architecture or utilizing pre- and post-processing techniques. In this article, we investigate the utility of anatomical priors to enhance the segmentation performance of the pancreas. Two 3D full-resolution nnU-Net models were trained, one with 8 refined labels from the public PANORAMA dataset, and another that combined them with labels derived from the public TotalSegmentator (TS) tool. The addition of anatomical priors resulted in a 6\% increase in Dice score ($p < .001$) and a 36.5 mm decrease in Hausdorff distance for pancreas segmentation ($p < .001$). Moreover, the pancreas was always detected when anatomy priors were used, whereas there were 8 instances of failed detections without their use. The use of anatomy priors shows promise for pancreas segmentation and subsequent derivation of imaging biomarkers. 

---
# An Analysis of Temporal Dropout in Earth Observation Time Series for Regression Tasks 

**Authors**: Miro Miranda, Francisco Mena, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06915)  

**Abstract**: Missing instances in time series data impose a significant challenge to deep learning models, particularly in regression tasks. In the Earth Observation field, satellite failure or cloud occlusion frequently results in missing time-steps, introducing uncertainties in the predicted output and causing a decline in predictive performance. While many studies address missing time-steps through data augmentation to improve model robustness, the uncertainty arising at the input level is commonly overlooked. To address this gap, we introduce Monte Carlo Temporal Dropout (MC-TD), a method that explicitly accounts for input-level uncertainty by randomly dropping time-steps during inference using a predefined dropout ratio, thereby simulating the effect of missing data. To bypass the need for costly searches for the optimal dropout ratio, we extend this approach with Monte Carlo Concrete Temporal Dropout (MC-ConcTD), a method that learns the optimal dropout distribution directly. Both MC-TD and MC-ConcTD are applied during inference, leveraging Monte Carlo sampling for uncertainty quantification. Experiments on three EO time-series datasets demonstrate that MC-ConcTD improves predictive performance and uncertainty calibration compared to existing approaches. Additionally, we highlight the advantages of adaptive dropout tuning over manual selection, making uncertainty quantification more robust and accessible for EO applications. 

---
# MedSegFactory: Text-Guided Generation of Medical Image-Mask Pairs 

**Authors**: Jiawei Mao, Yuhan Wang, Yucheng Tang, Daguang Xu, Kang Wang, Yang Yang, Zongwei Zhou, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06897)  

**Abstract**: This paper presents MedSegFactory, a versatile medical synthesis framework that generates high-quality paired medical images and segmentation masks across modalities and tasks. It aims to serve as an unlimited data repository, supplying image-mask pairs to enhance existing segmentation tools. The core of MedSegFactory is a dual-stream diffusion model, where one stream synthesizes medical images and the other generates corresponding segmentation masks. To ensure precise alignment between image-mask pairs, we introduce Joint Cross-Attention (JCA), enabling a collaborative denoising paradigm by dynamic cross-conditioning between streams. This bidirectional interaction allows both representations to guide each other's generation, enhancing consistency between generated pairs. MedSegFactory unlocks on-demand generation of paired medical images and segmentation masks through user-defined prompts that specify the target labels, imaging modalities, anatomical regions, and pathological conditions, facilitating scalable and high-quality data generation. This new paradigm of medical image synthesis enables seamless integration into diverse medical imaging workflows, enhancing both efficiency and accuracy. Extensive experiments show that MedSegFactory generates data of superior quality and usability, achieving competitive or state-of-the-art performance in 2D and 3D segmentation tasks while addressing data scarcity and regulatory constraints. 

---
# Audio-visual Event Localization on Portrait Mode Short Videos 

**Authors**: Wuyang Liu, Yi Chai, Yongpeng Yan, Yanzhen Ren  

**Link**: [PDF](https://arxiv.org/pdf/2504.06884)  

**Abstract**: Audio-visual event localization (AVEL) plays a critical role in multimodal scene understanding. While existing datasets for AVEL predominantly comprise landscape-oriented long videos with clean and simple audio context, short videos have become the primary format of online video content due to the the proliferation of smartphones. Short videos are characterized by portrait-oriented framing and layered audio compositions (e.g., overlapping sound effects, voiceovers, and music), which brings unique challenges unaddressed by conventional methods. To this end, we introduce AVE-PM, the first AVEL dataset specifically designed for portrait mode short videos, comprising 25,335 clips that span 86 fine-grained categories with frame-level annotations. Beyond dataset creation, our empirical analysis shows that state-of-the-art AVEL methods suffer an average 18.66% performance drop during cross-mode evaluation. Further analysis reveals two key challenges of different video formats: 1) spatial bias from portrait-oriented framing introduces distinct domain priors, and 2) noisy audio composition compromise the reliability of audio modality. To address these issues, we investigate optimal preprocessing recipes and the impact of background music for AVEL on portrait mode videos. Experiments show that these methods can still benefit from tailored preprocessing and specialized model design, thus achieving improved performance. This work provides both a foundational benchmark and actionable insights for advancing AVEL research in the era of mobile-centric video content. Dataset and code will be released. 

---
# Compound and Parallel Modes of Tropical Convolutional Neural Networks 

**Authors**: Mingbo Li, Liying Liu, Ye Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.06881)  

**Abstract**: Convolutional neural networks have become increasingly deep and complex, leading to higher computational costs. While tropical convolutional neural networks (TCNNs) reduce multiplications, they underperform compared to standard CNNs. To address this, we propose two new variants - compound TCNN (cTCNN) and parallel TCNN (pTCNN)-that use combinations of tropical min-plus and max-plus kernels to replace traditional convolution kernels. This reduces multiplications and balances efficiency with performance. Experiments on various datasets show that cTCNN and pTCNN match or exceed the performance of other CNN methods. Combining these with conventional CNNs in deeper architectures also improves performance. We are further exploring simplified TCNN architectures that reduce parameters and multiplications with minimal accuracy loss, aiming for efficient and effective models. 

---
# Persona Dynamics: Unveiling the Impact of Personality Traits on Agents in Text-Based Games 

**Authors**: Seungwon Lim, Seungbeen Lee, Dongjun Min, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06868)  

**Abstract**: Artificial agents are increasingly central to complex interactions and decision-making tasks, yet aligning their behaviors with desired human values remains an open challenge. In this work, we investigate how human-like personality traits influence agent behavior and performance within text-based interactive environments. We introduce PANDA: PersonalityAdapted Neural Decision Agents, a novel method for projecting human personality traits onto agents to guide their behavior. To induce personality in a text-based game agent, (i) we train a personality classifier to identify what personality type the agent's actions exhibit, and (ii) we integrate the personality profiles directly into the agent's policy-learning pipeline. By deploying agents embodying 16 distinct personality types across 25 text-based games and analyzing their trajectories, we demonstrate that an agent's action decisions can be guided toward specific personality profiles. Moreover, certain personality types, such as those characterized by higher levels of Openness, display marked advantages in performance. These findings underscore the promise of personality-adapted agents for fostering more aligned, effective, and human-centric decision-making in interactive environments. 

---
# GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes 

**Authors**: Seunghyeok Back, Joosoon Lee, Kangmin Kim, Heeseon Rho, Geonhyup Lee, Raeyoung Kang, Sangbeom Lee, Sangjun Noh, Youngjin Lee, Taeyeop Lee, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.06866)  

**Abstract**: Robust grasping in cluttered environments remains an open challenge in robotics. While benchmark datasets have significantly advanced deep learning methods, they mainly focus on simplistic scenes with light occlusion and insufficient diversity, limiting their applicability to practical scenarios. We present GraspClutter6D, a large-scale real-world grasping dataset featuring: (1) 1,000 highly cluttered scenes with dense arrangements (14.1 objects/scene, 62.6\% occlusion), (2) comprehensive coverage across 200 objects in 75 environment configurations (bins, shelves, and tables) captured using four RGB-D cameras from multiple viewpoints, and (3) rich annotations including 736K 6D object poses and 9.3B feasible robotic grasps for 52K RGB-D images. We benchmark state-of-the-art segmentation, object pose estimation, and grasping detection methods to provide key insights into challenges in cluttered environments. Additionally, we validate the dataset's effectiveness as a training resource, demonstrating that grasping networks trained on GraspClutter6D significantly outperform those trained on existing datasets in both simulation and real-world experiments. The dataset, toolkit, and annotation tools are publicly available on our project website: this https URL. 

---
# EIDT-V: Exploiting Intersections in Diffusion Trajectories for Model-Agnostic, Zero-Shot, Training-Free Text-to-Video Generation 

**Authors**: Diljeet Jagpal, Xi Chen, Vinay P. Namboodiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.06861)  

**Abstract**: Zero-shot, training-free, image-based text-to-video generation is an emerging area that aims to generate videos using existing image-based diffusion models. Current methods in this space require specific architectural changes to image generation models, which limit their adaptability and scalability. In contrast to such methods, we provide a model-agnostic approach. We use intersections in diffusion trajectories, working only with the latent values. We could not obtain localized frame-wise coherence and diversity using only the intersection of trajectories. Thus, we instead use a grid-based approach. An in-context trained LLM is used to generate coherent frame-wise prompts; another is used to identify differences between frames. Based on these, we obtain a CLIP-based attention mask that controls the timing of switching the prompts for each grid cell. Earlier switching results in higher variance, while later switching results in more coherence. Therefore, our approach can ensure appropriate control between coherence and variance for the frames. Our approach results in state-of-the-art performance while being more flexible when working with diverse image-generation models. The empirical analysis using quantitative metrics and user studies confirms our model's superior temporal consistency, visual fidelity and user satisfaction, thus providing a novel way to obtain training-free, image-based text-to-video generation. 

---
# Integrating Cognitive Processing Signals into Language Models: A Review of Advances, Applications and Future Directions 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.06843)  

**Abstract**: Recently, the integration of cognitive neuroscience in Natural Language Processing (NLP) has gained significant attention. This article provides a critical and timely overview of recent advancements in leveraging cognitive signals, particularly Eye-tracking (ET) signals, to enhance Language Models (LMs) and Multimodal Large Language Models (MLLMs). By incorporating user-centric cognitive signals, these approaches address key challenges, including data scarcity and the environmental costs of training large-scale models. Cognitive signals enable efficient data augmentation, faster convergence, and improved human alignment. The review emphasises the potential of ET data in tasks like Visual Question Answering (VQA) and mitigating hallucinations in MLLMs, and concludes by discussing emerging challenges and research trends. 

---
# Adaptive Locally Linear Embedding 

**Authors**: Ali Goli, Mahdieh Alizadeh, Hadi Sadoghi Yazdi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06829)  

**Abstract**: Manifold learning techniques, such as Locally linear embedding (LLE), are designed to preserve the local neighborhood structures of high-dimensional data during dimensionality reduction. Traditional LLE employs Euclidean distance to define neighborhoods, which can struggle to capture the intrinsic geometric relationships within complex data. A novel approach, Adaptive locally linear embedding(ALLE), is introduced to address this limitation by incorporating a dynamic, data-driven metric that enhances topological preservation. This method redefines the concept of proximity by focusing on topological neighborhood inclusion rather than fixed distances. By adapting the metric based on the local structure of the data, it achieves superior neighborhood preservation, particularly for datasets with complex geometries and high-dimensional structures. Experimental results demonstrate that ALLE significantly improves the alignment between neighborhoods in the input and feature spaces, resulting in more accurate and topologically faithful embeddings. This approach advances manifold learning by tailoring distance metrics to the underlying data, providing a robust solution for capturing intricate relationships in high-dimensional datasets. 

---
# Learning in Spiking Neural Networks with a Calcium-based Hebbian Rule for Spike-timing-dependent Plasticity 

**Authors**: Willian Soares Girão, Nicoletta Risi, Elisabetta Chicca  

**Link**: [PDF](https://arxiv.org/pdf/2504.06796)  

**Abstract**: Understanding how biological neural networks are shaped via local plasticity mechanisms can lead to energy-efficient and self-adaptive information processing systems, which promises to mitigate some of the current roadblocks in edge computing systems. While biology makes use of spikes to seamless use both spike timing and mean firing rate to modulate synaptic strength, most models focus on one of the two. In this work, we present a Hebbian local learning rule that models synaptic modification as a function of calcium traces tracking neuronal activity. We show how the rule reproduces results from spike time and spike rate protocols from neuroscientific studies. Moreover, we use the model to train spiking neural networks on MNIST digit recognition to show and explain what sort of mechanisms are needed to learn real-world patterns. We show how our model is sensitive to correlated spiking activity and how this enables it to modulate the learning rate of the network without altering the mean firing rate of the neurons nor the hyparameters of the learning rule. To the best of our knowledge, this is the first work that showcases how spike timing and rate can be complementary in their role of shaping the connectivity of spiking neural networks. 

---
# Zero-Shot Image-Based Large Language Model Approach to Road Pavement Monitoring 

**Authors**: Shuoshuo Xu, Kai Zhao, James Loney, Zili Li, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06785)  

**Abstract**: Effective and rapid evaluation of pavement surface condition is critical for prioritizing maintenance, ensuring transportation safety, and minimizing vehicle wear and tear. While conventional manual inspections suffer from subjectivity, existing machine learning-based methods are constrained by their reliance on large and high-quality labeled datasets, which require significant resources and limit adaptability across varied road conditions. The revolutionary advancements in Large Language Models (LLMs) present significant potential for overcoming these challenges. In this study, we propose an innovative automated zero-shot learning approach that leverages the image recognition and natural language understanding capabilities of LLMs to assess road conditions effectively. Multiple LLM-based assessment models were developed, employing prompt engineering strategies aligned with the Pavement Surface Condition Index (PSCI) standards. These models' accuracy and reliability were evaluated against official PSCI results, with an optimized model ultimately selected. Extensive tests benchmarked the optimized model against evaluations from various levels experts using Google Street View road images. The results reveal that the LLM-based approach can effectively assess road conditions, with the optimized model -employing comprehensive and structured prompt engineering strategies -outperforming simpler configurations by achieving high accuracy and consistency, even surpassing expert evaluations. Moreover, successfully applying the optimized model to Google Street View images demonstrates its potential for future city-scale deployments. These findings highlight the transformative potential of LLMs in automating road damage evaluations and underscore the pivotal role of detailed prompt engineering in achieving reliable assessments. 

---
# AI, Help Me Think$\unicode{x2014}$but for Myself: Assisting People in Complex Decision-Making by Providing Different Kinds of Cognitive Support 

**Authors**: Leon Reicherts, Zelun Tony Zhang, Elisabeth von Oswald, Yuanting Liu, Yvonne Rogers, Mariam Hassib  

**Link**: [PDF](https://arxiv.org/pdf/2504.06771)  

**Abstract**: How can we design AI tools that effectively support human decision-making by complementing and enhancing users' reasoning processes? Common recommendation-centric approaches face challenges such as inappropriate reliance or a lack of integration with users' decision-making processes. Here, we explore an alternative interaction model in which the AI outputs build upon users' own decision-making rationales. We compare this approach, which we call ExtendAI, with a recommendation-based AI. Participants in our mixed-methods user study interacted with both AIs as part of an investment decision-making task. We found that the AIs had different impacts, with ExtendAI integrating better into the decision-making process and people's own thinking and leading to slightly better outcomes. RecommendAI was able to provide more novel insights while requiring less cognitive effort. We discuss the implications of these and other findings along with three tensions of AI-assisted decision-making which our study revealed. 

---
# Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception 

**Authors**: Yuankun Xie, Ruibo Fu, Zhiyong Wang, Xiaopeng Wang, Songjun Cao, Long Ma, Haonan Cheng, Long Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.06753)  

**Abstract**: The rapid advancement of audio generation technologies has escalated the risks of malicious deepfake audio across speech, sound, singing voice, and music, threatening multimedia security and trust. While existing countermeasures (CMs) perform well in single-type audio deepfake detection (ADD), their performance declines in cross-type scenarios. This paper is dedicated to studying the alltype ADD task. We are the first to comprehensively establish an all-type ADD benchmark to evaluate current CMs, incorporating cross-type deepfake detection across speech, sound, singing voice, and music. Then, we introduce the prompt tuning self-supervised learning (PT-SSL) training paradigm, which optimizes SSL frontend by learning specialized prompt tokens for ADD, requiring 458x fewer trainable parameters than fine-tuning (FT). Considering the auditory perception of different audio types,we propose the wavelet prompt tuning (WPT)-SSL method to capture type-invariant auditory deepfake information from the frequency domain without requiring additional training parameters, thereby enhancing performance over FT in the all-type ADD task. To achieve an universally CM, we utilize all types of deepfake audio for co-training. Experimental results demonstrate that WPT-XLSR-AASIST achieved the best performance, with an average EER of 3.58% across all evaluation sets. The code is available online. 

---
# EDIT: Enhancing Vision Transformers by Mitigating Attention Sink through an Encoder-Decoder Architecture 

**Authors**: Wenfeng Feng, Guoying Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.06738)  

**Abstract**: In this paper, we propose EDIT (Encoder-Decoder Image Transformer), a novel architecture designed to mitigate the attention sink phenomenon observed in Vision Transformer models. Attention sink occurs when an excessive amount of attention is allocated to the [CLS] token, distorting the model's ability to effectively process image patches. To address this, we introduce a layer-aligned encoder-decoder architecture, where the encoder utilizes self-attention to process image patches, while the decoder uses cross-attention to focus on the [CLS] token. Unlike traditional encoder-decoder framework, where the decoder depends solely on high-level encoder representations, EDIT allows the decoder to extract information starting from low-level features, progressively refining the representation layer by layer. EDIT is naturally interpretable demonstrated through sequential attention maps, illustrating the refined, layer-by-layer focus on key image features. Experiments on ImageNet-1k and ImageNet-21k, along with transfer learning tasks, show that EDIT achieves consistent performance improvements over DeiT3 models. These results highlight the effectiveness of EDIT's design in addressing attention sink and improving visual feature extraction. 

---
# Learning global control of underactuated systems with Model-Based Reinforcement Learning 

**Authors**: Niccolò Turcato, Marco Calì, Alberto Dalla Libera, Giulio Giacomuzzo, Ruggero Carli, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2504.06721)  

**Abstract**: This short paper describes our proposed solution for the third edition of the "AI Olympics with RealAIGym" competition, held at ICRA 2025. We employed Monte-Carlo Probabilistic Inference for Learning Control (MC-PILCO), an MBRL algorithm recognized for its exceptional data efficiency across various low-dimensional robotic tasks, including cart-pole, ball \& plate, and Furuta pendulum systems. MC-PILCO optimizes a system dynamics model using interaction data, enabling policy refinement through simulation rather than direct system data optimization. This approach has proven highly effective in physical systems, offering greater data efficiency than Model-Free (MF) alternatives. Notably, MC-PILCO has previously won the first two editions of this competition, demonstrating its robustness in both simulated and real-world environments. Besides briefly reviewing the algorithm, we discuss the most critical aspects of the MC-PILCO implementation in the tasks at hand: learning a global policy for the pendubot and acrobot systems. 

---
# Masked Scene Modeling: Narrowing the Gap Between Supervised and Self-Supervised Learning in 3D Scene Understanding 

**Authors**: Pedro Hermosilla, Christian Stippel, Leon Sick  

**Link**: [PDF](https://arxiv.org/pdf/2504.06719)  

**Abstract**: Self-supervised learning has transformed 2D computer vision by enabling models trained on large, unannotated datasets to provide versatile off-the-shelf features that perform similarly to models trained with labels. However, in 3D scene understanding, self-supervised methods are typically only used as a weight initialization step for task-specific fine-tuning, limiting their utility for general-purpose feature extraction. This paper addresses this shortcoming by proposing a robust evaluation protocol specifically designed to assess the quality of self-supervised features for 3D scene understanding. Our protocol uses multi-resolution feature sampling of hierarchical models to create rich point-level representations that capture the semantic capabilities of the model and, hence, are suitable for evaluation with linear probing and nearest-neighbor methods. Furthermore, we introduce the first self-supervised model that performs similarly to supervised models when only off-the-shelf features are used in a linear probing setup. In particular, our model is trained natively in 3D with a novel self-supervised approach based on a Masked Scene Modeling objective, which reconstructs deep features of masked patches in a bottom-up manner and is specifically tailored to hierarchical 3D models. Our experiments not only demonstrate that our method achieves competitive performance to supervised models, but also surpasses existing self-supervised approaches by a large margin. The model and training code can be found at our Github repository (this https URL). 

---
# Hyperparameter Optimisation with Practical Interpretability and Explanation Methods in Probabilistic Curriculum Learning 

**Authors**: Llewyn Salt, Marcus Gallagher  

**Link**: [PDF](https://arxiv.org/pdf/2504.06683)  

**Abstract**: Hyperparameter optimisation (HPO) is crucial for achieving strong performance in reinforcement learning (RL), as RL algorithms are inherently sensitive to hyperparameter settings. Probabilistic Curriculum Learning (PCL) is a curriculum learning strategy designed to improve RL performance by structuring the agent's learning process, yet effective hyperparameter tuning remains challenging and computationally demanding. In this paper, we provide an empirical analysis of hyperparameter interactions and their effects on the performance of a PCL algorithm within standard RL tasks, including point-maze navigation and DC motor control. Using the AlgOS framework integrated with Optuna's Tree-Structured Parzen Estimator (TPE), we present strategies to refine hyperparameter search spaces, enhancing optimisation efficiency. Additionally, we introduce a novel SHAP-based interpretability approach tailored specifically for analysing hyperparameter impacts, offering clear insights into how individual hyperparameters and their interactions influence RL performance. Our work contributes practical guidelines and interpretability tools that significantly improve the effectiveness and computational feasibility of hyperparameter optimisation in reinforcement learning. 

---
# NLP Security and Ethics, in the Wild 

**Authors**: Heather Lent, Erick Galinkin, Yiyi Chen, Jens Myrup Pedersen, Leon Derczynski, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2504.06669)  

**Abstract**: As NLP models are used by a growing number of end-users, an area of increasing importance is NLP Security (NLPSec): assessing the vulnerability of models to malicious attacks and developing comprehensive countermeasures against them. While work at the intersection of NLP and cybersecurity has the potential to create safer NLP for all, accidental oversights can result in tangible harm (e.g., breaches of privacy or proliferation of malicious models). In this emerging field, however, the research ethics of NLP have not yet faced many of the long-standing conundrums pertinent to cybersecurity, until now. We thus examine contemporary works across NLPSec, and explore their engagement with cybersecurity's ethical norms. We identify trends across the literature, ultimately finding alarming gaps on topics like harm minimization and responsible disclosure. To alleviate these concerns, we provide concrete recommendations to help NLP researchers navigate this space more ethically, bridging the gap between traditional cybersecurity and NLP ethics, which we frame as ``white hat NLP''. The goal of this work is to help cultivate an intentional culture of ethical research for those working in NLP Security. 

---
# Bridging the Gap Between Preference Alignment and Machine Unlearning 

**Authors**: Xiaohua Feng, Yuyuan Li, Huwei Ji, Jiaming Zhang, Li Zhang, Tianyu Du, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06659)  

**Abstract**: Despite advances in Preference Alignment (PA) for Large Language Models (LLMs), mainstream methods like Reinforcement Learning with Human Feedback (RLHF) face notable challenges. These approaches require high-quality datasets of positive preference examples, which are costly to obtain and computationally intensive due to training instability, limiting their use in low-resource scenarios. LLM unlearning technique presents a promising alternative, by directly removing the influence of negative examples. However, current research has primarily focused on empirical validation, lacking systematic quantitative analysis. To bridge this gap, we propose a framework to explore the relationship between PA and LLM unlearning. Specifically, we introduce a bi-level optimization-based method to quantify the impact of unlearning specific negative examples on PA performance. Our analysis reveals that not all negative examples contribute equally to alignment improvement when unlearned, and the effect varies significantly across examples. Building on this insight, we pose a crucial question: how can we optimally select and weight negative examples for unlearning to maximize PA performance? To answer this, we propose a framework called Unlearning to Align (U2A), which leverages bi-level optimization to efficiently select and unlearn examples for optimal PA performance. We validate the proposed method through extensive experiments, with results confirming its effectiveness. 

---
# A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty 

**Authors**: Xiaohua Feng, Yuyuan Li, Chengye Wang, Junlin Liu, Li Zhang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06658)  

**Abstract**: Driven by privacy protection laws and regulations, unlearning in Large Language Models (LLMs) is gaining increasing attention. However, current research often neglects the interpretability of the unlearning process, particularly concerning sample-level unlearning difficulty. Existing studies typically assume a uniform unlearning difficulty across samples. This simplification risks attributing the performance of unlearning algorithms to sample selection rather than the algorithm's design, potentially steering the development of LLM unlearning in the wrong direction. Thus, we investigate the relationship between LLM unlearning and sample characteristics, with a focus on unlearning difficulty. Drawing inspiration from neuroscience, we propose a Memory Removal Difficulty ($\mathrm{MRD}$) metric to quantify sample-level unlearning difficulty. Using $\mathrm{MRD}$, we analyze the characteristics of hard-to-unlearn versus easy-to-unlearn samples. Furthermore, we propose an $\mathrm{MRD}$-based weighted sampling method to optimize existing unlearning algorithms, which prioritizes easily forgettable samples, thereby improving unlearning efficiency and effectiveness. We validate the proposed metric and method using public benchmarks and datasets, with results confirming its effectiveness. 

---
# GRAIN: Multi-Granular and Implicit Information Aggregation Graph Neural Network for Heterophilous Graphs 

**Authors**: Songwei Zhao, Yuan Jiang, Zijing Zhang, Yang Yu, Hechang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06649)  

**Abstract**: Graph neural networks (GNNs) have shown significant success in learning graph representations. However, recent studies reveal that GNNs often fail to outperform simple MLPs on heterophilous graph tasks, where connected nodes may differ in features or labels, challenging the homophily assumption. Existing methods addressing this issue often overlook the importance of information granularity and rarely consider implicit relationships between distant nodes. To overcome these limitations, we propose the Granular and Implicit Graph Network (GRAIN), a novel GNN model specifically designed for heterophilous graphs. GRAIN enhances node embeddings by aggregating multi-view information at various granularity levels and incorporating implicit data from distant, non-neighboring nodes. This approach effectively integrates local and global information, resulting in smoother, more accurate node representations. We also introduce an adaptive graph information aggregator that efficiently combines multi-granularity and implicit data, significantly improving node representation quality, as shown by experiments on 13 datasets covering varying homophily and heterophily. GRAIN consistently outperforms 12 state-of-the-art models, excelling on both homophilous and heterophilous graphs. 

---
# AMAD: AutoMasked Attention for Unsupervised Multivariate Time Series Anomaly Detection 

**Authors**: Tiange Huang, Yongjun Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06643)  

**Abstract**: Unsupervised multivariate time series anomaly detection (UMTSAD) plays a critical role in various domains, including finance, networks, and sensor systems. In recent years, due to the outstanding performance of deep learning in general sequential tasks, many models have been specialized for deep UMTSAD tasks and have achieved impressive results, particularly those based on the Transformer and self-attention mechanisms. However, the sequence anomaly association assumptions underlying these models are often limited to specific predefined patterns and scenarios, such as concentrated or peak anomaly patterns. These limitations hinder their ability to generalize to diverse anomaly situations, especially where the lack of labels poses significant challenges. To address these issues, we propose AMAD, which integrates \textbf{A}uto\textbf{M}asked Attention for UMTS\textbf{AD} scenarios. AMAD introduces a novel structure based on the AutoMask mechanism and an attention mixup module, forming a simple yet generalized anomaly association representation framework. This framework is further enhanced by a Max-Min training strategy and a Local-Global contrastive learning approach. By combining multi-scale feature extraction with automatic relative association modeling, AMAD provides a robust and adaptable solution to UMTSAD challenges. Extensive experimental results demonstrate that the proposed model achieving competitive performance results compared to SOTA benchmarks across a variety of datasets. 

---
# Wanting to be Understood 

**Authors**: Chrisantha Fernando, Dylan Banarse, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2504.06611)  

**Abstract**: This paper explores an intrinsic motivation for mutual awareness, hypothesizing that humans possess a fundamental drive to understand \textit{and to be understood} even in the absence of extrinsic rewards. Through simulations of the perceptual crossing paradigm, we explore the effect of various internal reward functions in reinforcement learning agents. The drive to understand is implemented as an active inference type artificial curiosity reward, whereas the drive to be understood is implemented through intrinsic rewards for imitation, influence/impressionability, and sub-reaction time anticipation of the other. Results indicate that while artificial curiosity alone does not lead to a preference for social interaction, rewards emphasizing reciprocal understanding successfully drive agents to prioritize interaction. We demonstrate that this intrinsic motivation can facilitate cooperation in tasks where only one agent receives extrinsic reward for the behaviour of the other. 

---
# InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features 

**Authors**: Sujay Khandagale, Bhawna Juneja, Prabhat Agarwal, Aditya Subramanian, Jaewon Yang, Yuting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06609)  

**Abstract**: Modern search systems use a multi-stage architecture to deliver personalized results efficiently. Key stages include retrieval, pre-ranking, full ranking, and blending, which refine billions of items to top selections. The pre-ranking stage, vital for scoring and filtering hundreds of thousands of items down to a few thousand, typically relies on two tower models due to their computational efficiency, despite often lacking in capturing complex interactions. While query-item cross interaction features are paramount for full ranking, integrating them into pre-ranking models presents efficiency-related challenges. In this paper, we introduce InteractRank, a novel two tower pre-ranking model with robust cross interaction features used at Pinterest. By incorporating historical user engagement-based query-item interactions in the scoring function along with the two tower dot product, InteractRank significantly boosts pre-ranking performance with minimal latency and computation costs. In real-world A/B experiments at Pinterest, InteractRank improves the online engagement metric by 6.5% over a BM25 baseline and by 3.7% over a vanilla two tower baseline. We also highlight other components of InteractRank, like real-time user-sequence modeling, and analyze their contributions through offline ablation studies. The code for InteractRank is available at this https URL. 

---
# Automated Business Process Analysis: An LLM-Based Approach to Value Assessment 

**Authors**: William De Michele, Abel Armas Cervantes, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06600)  

**Abstract**: Business processes are fundamental to organizational operations, yet their optimization remains challenging due to the timeconsuming nature of manual process analysis. Our paper harnesses Large Language Models (LLMs) to automate value-added analysis, a qualitative process analysis technique that aims to identify steps in the process that do not deliver value. To date, this technique is predominantly manual, time-consuming, and subjective. Our method offers a more principled approach which operates in two phases: first, decomposing high-level activities into detailed steps to enable granular analysis, and second, performing a value-added analysis to classify each step according to Lean principles. This approach enables systematic identification of waste while maintaining the semantic understanding necessary for qualitative analysis. We develop our approach using 50 business process models, for which we collect and publish manual ground-truth labels. Our evaluation, comparing zero-shot baselines with more structured prompts reveals (a) a consistent benefit of structured prompting and (b) promising performance for both tasks. We discuss the potential for LLMs to augment human expertise in qualitative process analysis while reducing the time and subjectivity inherent in manual approaches. 

---
# Exploring Ordinal Bias in Action Recognition for Instructional Videos 

**Authors**: Joochan Kim, Minjoon Jung, Byoung-Tak Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06580)  

**Abstract**: Action recognition models have achieved promising results in understanding instructional videos. However, they often rely on dominant, dataset-specific action sequences rather than true video comprehension, a problem that we define as ordinal bias. To address this issue, we propose two effective video manipulation methods: Action Masking, which masks frames of frequently co-occurring actions, and Sequence Shuffling, which randomizes the order of action segments. Through comprehensive experiments, we demonstrate that current models exhibit significant performance drops when confronted with nonstandard action sequences, underscoring their vulnerability to ordinal bias. Our findings emphasize the importance of rethinking evaluation strategies and developing models capable of generalizing beyond fixed action patterns in diverse instructional videos. 

---
# Attributes-aware Visual Emotion Representation Learning 

**Authors**: Rahul Singh Maharjan, Marta Romeo, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06578)  

**Abstract**: Visual emotion analysis or recognition has gained considerable attention due to the growing interest in understanding how images can convey rich semantics and evoke emotions in human perception. However, visual emotion analysis poses distinctive challenges compared to traditional vision tasks, especially due to the intricate relationship between general visual features and the different affective states they evoke, known as the affective gap. Researchers have used deep representation learning methods to address this challenge of extracting generalized features from entire images. However, most existing methods overlook the importance of specific emotional attributes such as brightness, colorfulness, scene understanding, and facial expressions. Through this paper, we introduce A4Net, a deep representation network to bridge the affective gap by leveraging four key attributes: brightness (Attribute 1), colorfulness (Attribute 2), scene context (Attribute 3), and facial expressions (Attribute 4). By fusing and jointly training all aspects of attribute recognition and visual emotion analysis, A4Net aims to provide a better insight into emotional content in images. Experimental results show the effectiveness of A4Net, showcasing competitive performance compared to state-of-the-art methods across diverse visual emotion datasets. Furthermore, visualizations of activation maps generated by A4Net offer insights into its ability to generalize across different visual emotion datasets. 

---
# Societal Impacts Research Requires Benchmarks for Creative Composition Tasks 

**Authors**: Judy Hanwen Shen, Carlos Guestrin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06549)  

**Abstract**: Foundation models that are capable of automating cognitive tasks represent a pivotal technological shift, yet their societal implications remain unclear. These systems promise exciting advances, yet they also risk flooding our information ecosystem with formulaic, homogeneous, and potentially misleading synthetic content. Developing benchmarks grounded in real use cases where these risks are most significant is therefore critical. Through a thematic analysis using 2 million language model user prompts, we identify creative composition tasks as a prevalent usage category where users seek help with personal tasks that require everyday creativity. Our fine-grained analysis identifies mismatches between current benchmarks and usage patterns among these tasks. Crucially, we argue that the same use cases that currently lack thorough evaluations can lead to negative downstream impacts. This position paper argues that benchmarks focused on creative composition tasks is a necessary step towards understanding the societal harms of AI-generated content. We call for greater transparency in usage patterns to inform the development of new benchmarks that can effectively measure both the progress and the impacts of models with creative capabilities. 

---
# Polygon: Symbolic Reasoning for SQL using Conflict-Driven Under-Approximation Search 

**Authors**: Pinhan Zhao, Yuepeng Wang, Xinyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06542)  

**Abstract**: We present a novel symbolic reasoning engine for SQL which can efficiently generate an input $I$ for $n$ queries $P_1, \cdots, P_n$, such that their outputs on $I$ satisfy a given property (expressed in SMT). This is useful in different contexts, such as disproving equivalence of two SQL queries and disambiguating a set of queries. Our first idea is to reason about an under-approximation of each $P_i$ -- that is, a subset of $P_i$'s input-output behaviors. While it makes our approach both semantics-aware and lightweight, this idea alone is incomplete (as a fixed under-approximation might miss some behaviors of interest). Therefore, our second idea is to perform search over an expressive family of under-approximations (which collectively cover all program behaviors of interest), thereby making our approach complete. We have implemented these ideas in a tool, Polygon, and evaluated it on over 30,000 benchmarks across two tasks (namely, SQL equivalence refutation and query disambiguation). Our evaluation results show that Polygon significantly outperforms all prior techniques. 

---
# OPAL: Encoding Causal Understanding of Physical Systems for Robot Learning 

**Authors**: Daniel Tcheurekdjian, Joshua Klasmeier, Tom Cooney, Christopher McCann, Tyler Fenstermaker  

**Link**: [PDF](https://arxiv.org/pdf/2504.06538)  

**Abstract**: We present OPAL (Operant Physical Agent with Language), a novel vision-language-action architecture that introduces topological constraints to flow matching for robotic control. To do so, we further introduce topological attention. Our approach models action sequences as topologically-structured representations with non-trivial constraints. Experimental results across 10 complex manipulation tasks demonstrate OPAL's superior performance compared to previous approaches, including Octo, OpenVLA, and ${\pi}$0.
Our architecture achieves significant improvements in zero-shot performance without requiring task-specific fine-tuning, while reducing inference computational requirements by 42%. The theoretical guarantees provided by our topological approach result in more coherent long-horizon action sequences. Our results highlight the potential of constraining the search space of learning problems in robotics by deriving from fundamental physical laws, and the possibility of using topological attention to embed causal understanding into transformer architectures. 

---
# Lugha-Llama: Adapting Large Language Models for African Languages 

**Authors**: Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2504.06536)  

**Abstract**: Large language models (LLMs) have achieved impressive results in a wide range of natural language applications. However, they often struggle to recognize low-resource languages, in particular African languages, which are not well represented in large training corpora. In this paper, we consider how to adapt LLMs to low-resource African languages. We find that combining curated data from African languages with high-quality English educational texts results in a training mix that substantially improves the model's performance on these languages. On the challenging IrokoBench dataset, our models consistently achieve the best performance amongst similarly sized baselines, particularly on knowledge-intensive multiple-choice questions (AfriMMLU). Additionally, on the cross-lingual question answering benchmark AfriQA, our models outperform the base model by over 10%. To better understand the role of English data during training, we translate a subset of 200M tokens into Swahili language and perform an analysis which reveals that the content of these data is primarily responsible for the strong performance. We release our models and data to encourage future research on African languages. 

---
# Flexible Graph Similarity Computation With A Proactive Optimization Strategy 

**Authors**: Zhouyang Liu, Ning Liu, Yixin Chen, Jiezhong He, Dongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.06533)  

**Abstract**: Graph Edit Distance (GED) is an important similarity measure in graph retrieval, which quantifies the minimum cost of transforming one graph into another through edit operations, and offers flexibility by allowing customizable operation costs. Recent learning-based approaches approximate GEDs with the distances between representations in vector spaces. However, these methods often struggle with varying operation costs due to neglecting the impact of these costs on determining optimal graph mappings. Furthermore, they rely on isolated node distances as guidance, necessitating inefficient reactive refinements of mappings. To address these issues, we propose Graph Edit Network (GEN), a novel learning-based approach for flexible GED computation. By identifying the limitations of existing methods in capturing flexibility of GED, we introduce a principled yet simple solution that incorporates the operation costs before establishing mappings. To improve matching efficiency, we propose a strategy that proactively optimizes guidance from a graph perspective. This strategy initializes guidance as each node's alignment difficulty and captures the interdependencies between matches within and across graphs through a difficulty propagation mechanism, enabling more informed decisions. As a result, GEN selects optimal matches in a single step, minimizing the need for costly refinements. Results on real-world and synthetic datasets demonstrate the effectiveness, time efficiency, and adaptability of GEN, achieving up to 37.8\% error reduction and 72.7\% inference time reduction compared with state-of-the-art models, while performing robustly under varying cost settings and graph sizes. 

---
# WaveHiTS: Wavelet-Enhanced Hierarchical Time Series Modeling for Wind Direction Nowcasting in Eastern Inner Mongolia 

**Authors**: Hailong Shu, Weiwei Song, Yue Wang, Jiping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06532)  

**Abstract**: Wind direction forecasting plays a crucial role in optimizing wind energy production, but faces significant challenges due to the circular nature of directional data, error accumulation in multi-step forecasting, and complex meteorological interactions. This paper presents a novel model, WaveHiTS, which integrates wavelet transform with Neural Hierarchical Interpolation for Time Series to address these challenges. Our approach decomposes wind direction into U-V components, applies wavelet transform to capture multi-scale frequency patterns, and utilizes a hierarchical structure to model temporal dependencies at multiple scales, effectively mitigating error propagation. Experiments conducted on real-world meteorological data from Inner Mongolia, China demonstrate that WaveHiTS significantly outperforms deep learning models (RNN, LSTM, GRU), transformer-based approaches (TFT, Informer, iTransformer), and hybrid models (EMD-LSTM). The proposed model achieves RMSE values of approximately 19.2°-19.4° compared to 56°-64° for deep learning recurrent models, maintaining consistent accuracy across all forecasting steps up to 60 minutes ahead. Moreover, WaveHiTS demonstrates superior robustness with vector correlation coefficients (VCC) of 0.985-0.987 and hit rates of 88.5%-90.1%, substantially outperforming baseline models. Ablation studies confirm that each component-wavelet transform, hierarchical structure, and U-V decomposition-contributes meaningfully to overall performance. These improvements in wind direction nowcasting have significant implications for enhancing wind turbine yaw control efficiency and grid integration of wind energy. 

---
# Beyond Moore's Law: Harnessing the Redshift of Generative AI with Effective Hardware-Software Co-Design 

**Authors**: Amir Yazdanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2504.06531)  

**Abstract**: For decades, Moore's Law has served as a steadfast pillar in computer architecture and system design, promoting a clear abstraction between hardware and software. This traditional Moore's computing paradigm has deepened the rift between the two, enabling software developers to achieve near-exponential performance gains often without needing to delve deeply into hardware-specific optimizations. Yet today, Moore's Law -- with its once relentless performance gains now diminished to incremental improvements -- faces inevitable physical barriers. This stagnation necessitates a reevaluation of the conventional system design philosophy. The traditional decoupled system design philosophy, which maintains strict abstractions between hardware and software, is increasingly obsolete. The once-clear boundary between software and hardware is rapidly dissolving, replaced by co-design. It is imperative for the computing community to intensify its commitment to hardware-software co-design, elevating system abstractions to first-class citizens and reimagining design principles to satisfy the insatiable appetite of modern computing. Hardware-software co-design is not a recent innovation. To illustrate its historical evolution, I classify its development into five relatively distinct ``epochs''. This post also highlights the growing influence of the architecture community in interdisciplinary teams -- particularly alongside ML researchers -- and explores why current co-design paradigms are struggling in today's computing landscape. Additionally, I will examine the concept of the ``hardware lottery'' and explore directions to mitigate its constraining influence on the next era of computing innovation. 

---
# TSP-OCS: A Time-Series Prediction for Optimal Camera Selection in Multi-Viewpoint Surgical Video Analysis 

**Authors**: Xinyu Liu, Xiaoguang Lin, Xiang Liu, Yong Yang, Hongqian Wang, Qilong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.06527)  

**Abstract**: Recording the open surgery process is essential for educational and medical evaluation purposes; however, traditional single-camera methods often face challenges such as occlusions caused by the surgeon's head and body, as well as limitations due to fixed camera angles, which reduce comprehensibility of the video content. This study addresses these limitations by employing a multi-viewpoint camera recording system, capturing the surgical procedure from six different angles to mitigate occlusions. We propose a fully supervised learning-based time series prediction method to choose the best shot sequences from multiple simultaneously recorded video streams, ensuring optimal viewpoints at each moment. Our time series prediction model forecasts future camera selections by extracting and fusing visual and semantic features from surgical videos using pre-trained models. These features are processed by a temporal prediction network with TimeBlocks to capture sequential dependencies. A linear embedding layer reduces dimensionality, and a Softmax classifier selects the optimal camera view based on the highest probability. In our experiments, we created five groups of open thyroidectomy videos, each with simultaneous recordings from six different angles. The results demonstrate that our method achieves competitive accuracy compared to traditional supervised methods, even when predicting over longer time horizons. Furthermore, our approach outperforms state-of-the-art time series prediction techniques on our dataset. This manuscript makes a unique contribution by presenting an innovative framework that advances surgical video analysis techniques, with significant implications for improving surgical education and patient safety. 

---
# The Power of the Pareto Front: Balancing Uncertain Rewards for Adaptive Experimentation in scanning probe microscopy 

**Authors**: Yu Liu, Sergei V. Kalinin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06525)  

**Abstract**: Automated experimentation has the potential to revolutionize scientific discovery, but its effectiveness depends on well-defined optimization targets, which are often uncertain or probabilistic in real-world settings. In this work, we demonstrate the application of Multi-Objective Bayesian Optimization (MOBO) to balance multiple, competing rewards in autonomous experimentation. Using scanning probe microscopy (SPM) imaging, one of the most widely used and foundational SPM modes, we show that MOBO can optimize imaging parameters to enhance measurement quality, reproducibility, and efficiency. A key advantage of this approach is the ability to compute and analyze the Pareto front, which not only guides optimization but also provides physical insights into the trade-offs between different objectives. Additionally, MOBO offers a natural framework for human-in-the-loop decision-making, enabling researchers to fine-tune experimental trade-offs based on domain expertise. By standardizing high-quality, reproducible measurements and integrating human input into AI-driven optimization, this work highlights MOBO as a powerful tool for advancing autonomous scientific discovery. 

---
# Continuous-Variable Quantum Encoding Techniques: A Comparative Study of Embedding Techniques and Their Impact on Machine Learning Performance 

**Authors**: Minati Rath, Hema Date  

**Link**: [PDF](https://arxiv.org/pdf/2504.06497)  

**Abstract**: This study explores the intersection of continuous-variable quantum computing (CVQC) and classical machine learning, focusing on CVQC data encoding techniques, including Displacement encoding and squeezing encoding, alongside Instantaneous Quantum Polynomial (IQP) encoding from discrete quantum computing. We perform an extensive empirical analysis to assess the impact of these encoding methods on classical machine learning models, such as Logistic Regression, Support Vector Machines, K-Nearest Neighbors, and ensemble methods like Random Forest and LightGBM. Our findings indicate that CVQC-based encoding methods significantly enhance feature expressivity, resulting in improved classification accuracy and F1 scores, especially in high-dimensional and complex datasets. However, these improvements come with varying computational costs, which depend on the complexity of the encoding and the architecture of the machine learning models. Additionally, we examine the trade-off between quantum expressibility and classical learnability, offering valuable insights into the practical feasibility of incorporating these quantum encodings into real-world applications. This study contributes to the growing body of research on quantum-classical hybrid learning, emphasizing the role of CVQC in advancing quantum data representation and its integration into classical machine learning workflows. 

---
# Exploiting Meta-Learning-based Poisoning Attacks for Graph Link Prediction 

**Authors**: Mingchen Li, Di Zhuang, Keyu Chen, Dumindu Samaraweera, Morris Chang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06492)  

**Abstract**: Link prediction in graph data utilizes various algorithms and machine learning/deep learning models to predict potential relationships between graph nodes. This technique has found widespread use in numerous real-world applications, including recommendation systems, community networks, and biological structures. However, recent research has highlighted the vulnerability of link prediction models to adversarial attacks, such as poisoning and evasion attacks. Addressing the vulnerability of these models is crucial to ensure stable and robust performance in link prediction applications. While many works have focused on enhancing the robustness of the Graph Convolution Network (GCN) model, the Variational Graph Auto-Encoder (VGAE), a sophisticated model for link prediction, has not been thoroughly investigated in the context of graph adversarial attacks. To bridge this gap, this article proposes an unweighted graph poisoning attack approach using meta-learning techniques to undermine VGAE's link prediction performance. We conducted comprehensive experiments on diverse datasets to evaluate the proposed method and its parameters, comparing it with existing approaches in similar settings. Our results demonstrate that our approach significantly diminishes link prediction performance and outperforms other state-of-the-art methods. 

---
# AI-Assisted Transport of Radioactive Ion Beams 

**Authors**: Sergio Lopez-Caceres, Daniel Santiago-Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2504.06469)  

**Abstract**: Beams of radioactive heavy ions allow researchers to study rare and unstable atomic nuclei, shedding light into the internal structure of exotic nuclei and on how chemical elements are formed in stars. However, the extraction and transport of radioactive beams rely on time-consuming expert-driven tuning methods, where hundreds of parameters are manually optimized. Here, we introduce a system that uses Artificial Intelligence (AI) to assist in the radioactive beam transport process. We apply our methodology to real-life scenarios showing advantages when compared with standard tuning methods. Our method can be extended to other radioactive beam facilities around the world to improve operational efficiency and enhance scientific output. 

---
# Agent-Arena: A General Framework for Evaluating Control Algorithms 

**Authors**: Halid Abdulrahim Kadi, Kasim Terzić  

**Link**: [PDF](https://arxiv.org/pdf/2504.06468)  

**Abstract**: Robotic research is inherently challenging, requiring expertise in diverse environments and control algorithms. Adapting algorithms to new environments often poses significant difficulties, compounded by the need for extensive hyper-parameter tuning in data-driven methods. To address these challenges, we present Agent-Arena, a Python framework designed to streamline the integration, replication, development, and testing of decision-making policies across a wide range of benchmark environments. Unlike existing frameworks, Agent-Arena is uniquely generalised to support all types of control algorithms and is adaptable to both simulation and real-robot scenarios. Please see our GitHub repository this https URL. 

---
# Federated Neural Architecture Search with Model-Agnostic Meta Learning 

**Authors**: Xinyuan Huang, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06457)  

**Abstract**: Federated Learning (FL) often struggles with data heterogeneity due to the naturally uneven distribution of user data across devices. Federated Neural Architecture Search (NAS) enables collaborative search for optimal model architectures tailored to heterogeneous data to achieve higher accuracy. However, this process is time-consuming due to extensive search space and retraining. To overcome this, we introduce FedMetaNAS, a framework that integrates meta-learning with NAS within the FL context to expedite the architecture search by pruning the search space and eliminating the retraining stage. Our approach first utilizes the Gumbel-Softmax reparameterization to facilitate relaxation of the mixed operations in the search space. We then refine the local search process by incorporating Model-Agnostic Meta-Learning, where a task-specific learner adapts both weights and architecture parameters (alphas) for individual tasks, while a meta learner adjusts the overall model weights and alphas based on the gradient information from task learners. Following the meta-update, we propose soft pruning using the same trick on search space to gradually sparsify the architecture, ensuring that the performance of the chosen architecture remains robust after pruning which allows for immediate use of the model without retraining. Experimental evaluations demonstrate that FedMetaNAS significantly accelerates the search process by more than 50\% with higher accuracy compared to FedNAS. 

---
# Can you Finetune your Binoculars? Embedding Text Watermarks into the Weights of Large Language Models 

**Authors**: Fay Elhassan, Niccolò Ajroldi, Antonio Orvieto, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2504.06446)  

**Abstract**: The indistinguishability of AI-generated content from human text raises challenges in transparency and accountability. While several methods exist to watermark models behind APIs, embedding watermark strategies directly into model weights that are later reflected in the outputs of the model is challenging. In this study we propose a strategy to finetune a pair of low-rank adapters of a model, one serving as the text-generating model, and the other as the detector, so that a subtle watermark is embedded into the text generated by the first model and simultaneously optimized for detectability by the second. In this way, the watermarking strategy is fully learned end-to-end. This process imposes an optimization challenge, as balancing watermark robustness, naturalness, and task performance requires trade-offs. We discuss strategies on how to optimize this min-max objective and present results showing the effect of this modification to instruction finetuning. 

---
# Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning 

**Authors**: Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06438)  

**Abstract**: Large language models (LLMs) have shown substantial capacity for generating fluent, contextually appropriate responses. However, they can produce hallucinated outputs, especially when a user query includes one or more false premises-claims that contradict established facts. Such premises can mislead LLMs into offering fabricated or misleading details. Existing approaches include pretraining, fine-tuning, and inference-time techniques that often rely on access to logits or address hallucinations after they occur. These methods tend to be computationally expensive, require extensive training data, or lack proactive mechanisms to prevent hallucination before generation, limiting their efficiency in real-time applications. We propose a retrieval-based framework that identifies and addresses false premises before generation. Our method first transforms a user's query into a logical representation, then applies retrieval-augmented generation (RAG) to assess the validity of each premise using factual sources. Finally, we incorporate the verification results into the LLM's prompt to maintain factual consistency in the final output. Experiments show that this approach effectively reduces hallucinations, improves factual accuracy, and does not require access to model logits or large-scale fine-tuning. 

---
# Language-Dependent Political Bias in AI: A Study of ChatGPT and Gemini 

**Authors**: Dogus Yuksel, Mehmet Cem Catalbas, Bora Oc  

**Link**: [PDF](https://arxiv.org/pdf/2504.06436)  

**Abstract**: As leading examples of large language models, ChatGPT and Gemini claim to provide accurate and unbiased information, emphasizing their commitment to political neutrality and avoidance of personal bias. This research investigates the political tendency of large language models and the existence of differentiation according to the query language. For this purpose, ChatGPT and Gemini were subjected to a political axis test using 14 different languages. The findings of the study suggest that these large language models do exhibit political tendencies, with both models demonstrating liberal and leftist biases. A comparative analysis revealed that Gemini exhibited a more pronounced liberal and left-wing tendency compared to ChatGPT. The study also found that these political biases varied depending on the language used for inquiry. The study delves into the factors that constitute political tendencies and linguistic differentiation, exploring differences in the sources and scope of educational data, structural and grammatical features of languages, cultural and political contexts, and the model's response to linguistic features. From this standpoint, and an ethical perspective, it is proposed that artificial intelligence tools should refrain from asserting a lack of political tendencies and neutrality, instead striving for political neutrality and executing user queries by incorporating these tendencies. 

---
# Evaluating Mutation Techniques in Genetic Algorithm-Based Quantum Circuit Synthesis 

**Authors**: Michael Kölle, Tom Bintener, Maximilian Zorn, Gerhard Stenzel, Leo Sünkel, Thomas Gabor, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2504.06413)  

**Abstract**: Quantum computing leverages the unique properties of qubits and quantum parallelism to solve problems intractable for classical systems, offering unparalleled computational potential. However, the optimization of quantum circuits remains critical, especially for noisy intermediate-scale quantum (NISQ) devices with limited qubits and high error rates. Genetic algorithms (GAs) provide a promising approach for efficient quantum circuit synthesis by automating optimization tasks. This work examines the impact of various mutation strategies within a GA framework for quantum circuit synthesis. By analyzing how different mutations transform circuits, it identifies strategies that enhance efficiency and performance. Experiments utilized a fitness function emphasizing fidelity, while accounting for circuit depth and T operations, to optimize circuits with four to six qubits. Comprehensive hyperparameter testing revealed that combining delete and swap strategies outperformed other approaches, demonstrating their effectiveness in developing robust GA-based quantum circuit optimizers. 

---
# Understanding Machine Unlearning Through the Lens of Mode Connectivity 

**Authors**: Jiali Cheng, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.06407)  

**Abstract**: Machine Unlearning aims to remove undesired information from trained models without requiring full retraining from scratch. Despite recent advancements, their underlying loss landscapes and optimization dynamics received less attention. In this paper, we investigate and analyze machine unlearning through the lens of mode connectivity - the phenomenon where independently trained models can be connected by smooth low-loss paths in the parameter space. We define and study mode connectivity in unlearning across a range of overlooked conditions, including connections between different unlearning methods, models trained with and without curriculum learning, and models optimized with first-order and secondorder techniques. Our findings show distinct patterns of fluctuation of different evaluation metrics along the curve, as well as the mechanistic (dis)similarity between unlearning methods. To the best of our knowledge, this is the first study on mode connectivity in the context of machine unlearning. 

---
# Physical spline for denoising object trajectory data by combining splines, ML feature regression and model knowledge 

**Authors**: Jonas Torzewski  

**Link**: [PDF](https://arxiv.org/pdf/2504.06404)  

**Abstract**: This article presents a method for estimating the dynamic driving states (position, velocity, acceleration and heading) from noisy measurement data. The proposed approach is effective with both complete and partial observations, producing refined trajectory signals with kinematic consistency, ensuring that velocity is the integral of acceleration and position is the integral of velocity. Additionally, the method accounts for the constraint that vehicles can only move in the direction of their orientation. The method is implemented as a configurable python library that also enables trajectory estimation solely based on position data. Regularization is applied to prevent extreme state variations. A key application is enhancing recorded trajectory data for use as reference inputs in machine learning models. At the end, the article presents the results of the method along with a comparison to ground truth data. 

---
# Analyzing the Impact of Low-Rank Adaptation for Cross-Domain Few-Shot Object Detection in Aerial Images 

**Authors**: Hicham Talaoubrid, Anissa Mokraoui, Ismail Ben Ayed, Axel Prouvost, Sonimith Hang, Monit Korn, Rémi Harvey  

**Link**: [PDF](https://arxiv.org/pdf/2504.06330)  

**Abstract**: This paper investigates the application of Low-Rank Adaptation (LoRA) to small models for cross-domain few-shot object detection in aerial images. Originally designed for large-scale models, LoRA helps mitigate overfitting, making it a promising approach for resource-constrained settings. We integrate LoRA into DiffusionDet, and evaluate its performance on the DOTA and DIOR datasets. Our results show that LoRA applied after an initial fine-tuning slightly improves performance in low-shot settings (e.g., 1-shot and 5-shot), while full fine-tuning remains more effective in higher-shot configurations. These findings highlight LoRA's potential for efficient adaptation in aerial object detection, encouraging further research into parameter-efficient fine-tuning strategies for few-shot learning. Our code is available here: this https URL. 

---
# A Geometric-Aware Perspective and Beyond: Hybrid Quantum-Classical Machine Learning Methods 

**Authors**: Azadeh Alavia, Hossein Akhoundib, Fatemeh Kouchmeshkib, Mojtaba Mahmoodianc, Sanduni Jayasinghec, Yongli Rena, Abdolrahman Alavi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06328)  

**Abstract**: Geometric Machine Learning (GML) has shown that respecting non-Euclidean geometry in data spaces can significantly improve performance over naive Euclidean assumptions. In parallel, Quantum Machine Learning (QML) has emerged as a promising paradigm that leverages superposition, entanglement, and interference within quantum state manifolds for learning tasks. This paper offers a unifying perspective by casting QML as a specialized yet more expressive branch of GML. We argue that quantum states, whether pure or mixed, reside on curved manifolds (e.g., projective Hilbert spaces or density-operator manifolds), mirroring how covariance matrices inhabit the manifold of symmetric positive definite (SPD) matrices or how image sets occupy Grassmann manifolds. However, QML also benefits from purely quantum properties, such as entanglement-induced curvature, that can yield richer kernel structures and more nuanced data embeddings.
We illustrate these ideas with published and newly discussed results, including hybrid classical -quantum pipelines for diabetic foot ulcer classification and structural health monitoring. Despite near-term hardware limitations that constrain purely quantum solutions, hybrid architectures already demonstrate tangible benefits by combining classical manifold-based feature extraction with quantum embeddings. We present a detailed mathematical treatment of the geometrical underpinnings of quantum states, emphasizing parallels to classical Riemannian geometry and manifold-based optimization. Finally, we outline open research challenges and future directions, including Quantum Large Language Models (LLMs), quantum reinforcement learning, and emerging hardware approaches, demonstrating how synergizing GML and QML principles can unlock the next generation of machine intelligence. 

---
# MM-STFlowNet: A Transportation Hub-Oriented Multi-Mode Passenger Flow Prediction Method via Spatial-Temporal Dynamic Graph Modeling 

**Authors**: Ronghui Zhang, Wenbin Xing, Mengran Li, Zihan Wang, Junzhou Chen, Xiaolei Ma, Zhiyuan Liu, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2504.06325)  

**Abstract**: Accurate and refined passenger flow prediction is essential for optimizing the collaborative management of multiple collection and distribution modes in large-scale transportation hubs. Traditional methods often focus only on the overall passenger volume, neglecting the interdependence between different modes within the hub. To address this limitation, we propose MM-STFlowNet, a comprehensive multi-mode prediction framework grounded in dynamic spatial-temporal graph modeling. Initially, an integrated temporal feature processing strategy is implemented using signal decomposition and convolution techniques to address data spikes and high volatility. Subsequently, we introduce the Spatial-Temporal Dynamic Graph Convolutional Recurrent Network (STDGCRN) to capture detailed spatial-temporal dependencies across multiple traffic modes, enhanced by an adaptive channel attention mechanism. Finally, the self-attention mechanism is applied to incorporate various external factors, further enhancing prediction accuracy. Experiments on a real-world dataset from Guangzhounan Railway Station in China demonstrate that MM-STFlowNet achieves state-of-the-art performance, particularly during peak periods, providing valuable insight for transportation hub management. 

---
# From Stability to Inconsistency: A Study of Moral Preferences in LLMs 

**Authors**: Monika Jotautaite, Mary Phuong, Chatrik Singh Mangat, Maria Angelica Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2504.06324)  

**Abstract**: As large language models (LLMs) increasingly integrate into our daily lives, it becomes crucial to understand their implicit biases and moral tendencies. To address this, we introduce a Moral Foundations LLM dataset (MFD-LLM) grounded in Moral Foundations Theory, which conceptualizes human morality through six core foundations. We propose a novel evaluation method that captures the full spectrum of LLMs' revealed moral preferences by answering a range of real-world moral dilemmas. Our findings reveal that state-of-the-art models have remarkably homogeneous value preferences, yet demonstrate a lack of consistency. 

---
# Mosaic: Composite Projection Pruning for Resource-efficient LLMs 

**Authors**: Bailey J. Eccles, Leon Wong, Blesson Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2504.06323)  

**Abstract**: Extensive compute and memory requirements limit the deployment of large language models (LLMs) on any hardware. Compression methods, such as pruning, can reduce model size, which in turn reduces resource requirements. State-of-the-art pruning is based on coarse-grained methods. They are time-consuming and inherently remove critical model parameters, adversely impacting the quality of the pruned model. This paper introduces projection pruning, a novel fine-grained method for pruning LLMs. In addition, LLM projection pruning is enhanced by a new approach we refer to as composite projection pruning - the synergistic combination of unstructured pruning that retains accuracy and structured pruning that reduces model size. We develop Mosaic, a novel system to create and deploy pruned LLMs using composite projection pruning. Mosaic is evaluated using a range of performance and quality metrics on multiple hardware platforms, LLMs, and datasets. Mosaic is 7.19x faster in producing models than existing approaches. Mosaic models achieve up to 84.2% lower perplexity and 31.4% higher accuracy than models obtained from coarse-grained pruning. Up to 67% faster inference and 68% lower GPU memory use is noted for Mosaic models. 

---
# Assessing employment and labour issues implicated by using AI 

**Authors**: Thijs Willems, Darion Jin Hotan, Jiawen Cheryl Tang, Norakmal Hakim bin Norhashim, King Wang Poon, Zi An Galvyn Goh, Radha Vinod  

**Link**: [PDF](https://arxiv.org/pdf/2504.06322)  

**Abstract**: This chapter critiques the dominant reductionist approach in AI and work studies, which isolates tasks and skills as replaceable components. Instead, it advocates for a systemic perspective that emphasizes the interdependence of tasks, roles, and workplace contexts. Two complementary approaches are proposed: an ethnographic, context-rich method that highlights how AI reconfigures work environments and expertise; and a relational task-based analysis that bridges micro-level work descriptions with macro-level labor trends. The authors argue that effective AI impact assessments must go beyond predicting automation rates to include ethical, well-being, and expertise-related questions. Drawing on empirical case studies, they demonstrate how AI reshapes human-technology relations, professional roles, and tacit knowledge practices. The chapter concludes by calling for a human-centric, holistic framework that guides organizational and policy decisions, balancing technological possibilities with social desirability and sustainability of work. 

---
# Hybrid Temporal Differential Consistency Autoencoder for Efficient and Sustainable Anomaly Detection in Cyber-Physical Systems 

**Authors**: Michael Somma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06320)  

**Abstract**: Cyberattacks on critical infrastructure, particularly water distribution systems, have increased due to rapid digitalization and the integration of IoT devices and industrial control systems (ICS). These cyber-physical systems (CPS) introduce new vulnerabilities, requiring robust and automated intrusion detection systems (IDS) to mitigate potential threats. This study addresses key challenges in anomaly detection by leveraging time correlations in sensor data, integrating physical principles into machine learning models, and optimizing computational efficiency for edge applications. We build upon the concept of temporal differential consistency (TDC) loss to capture the dynamics of the system, ensuring meaningful relationships between dynamic states. Expanding on this foundation, we propose a hybrid autoencoder-based approach, referred to as hybrid TDC-AE, which extends TDC by incorporating both deterministic nodes and conventional statistical nodes. This hybrid structure enables the model to account for non-deterministic processes. Our approach achieves state-of-the-art classification performance while improving time to detect anomalies by 3%, outperforming the BATADAL challenge leader without requiring domain-specific knowledge, making it broadly applicable. Additionally, it maintains the computational efficiency of conventional autoencoders while reducing the number of fully connected layers, resulting in a more sustainable and efficient solution. The method demonstrates how leveraging physics-inspired consistency principles enhances anomaly detection and strengthens the resilience of cyber-physical systems. 

---
# Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching 

**Authors**: Yanhao Dong, Yubo Miao, Weinan Li, Xiao Zheng, Chao Wang, Feng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06319)  

**Abstract**: Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap. By strategically scheduling idle memory bandwidth during active computation windows, our method proactively prefetches required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for subsequent accesses and effectively hiding HBM access latency within computational cycles. Extensive experiments on NVIDIA H20 GPUs demonstrate that the proposed method achieves 2.15x improvement in attention kernel efficiency and up to 1.97x end-to-end throughput enhancement, surpassing state-of-the-art baseline FlashAttention-3. Notably, our solution maintains orthogonality to existing optimization techniques and can be integrated with current inference frameworks, providing a scalable latency-hiding solution for next-generation LLM inference engines. 

---
# DMol: A Schedule-Driven Diffusion Model for Highly Efficient and Versatile Molecule Generation 

**Authors**: Peizhi Niu, Yu-Hsiang Wang, Vishal Rana, Chetan Rupakheti, Abhishek Pandey, Olgica Milenkovic  

**Link**: [PDF](https://arxiv.org/pdf/2504.06312)  

**Abstract**: We introduce a new graph diffusion model for small molecule generation, \emph{DMol}, which outperforms the state-of-the-art DiGress model in terms of validity by roughly $1.5\%$ across all benchmarking datasets while reducing the number of diffusion steps by at least $10$-fold, and the running time to roughly one half. The performance improvements are a result of a careful change in the objective function and a ``graph noise" scheduling approach which, at each diffusion step, allows one to only change a subset of nodes of varying size in the molecule graph. Another relevant property of the method is that it can be easily combined with junction-tree-like graph representations that arise by compressing a collection of relevant ring structures into supernodes. Unlike classical junction-tree techniques that involve VAEs and require complicated reconstruction steps, compressed DMol directly performs graph diffusion on a graph that compresses only a carefully selected set of frequent carbon rings into supernodes, which results in straightforward sample generation. This compressed DMol method offers additional validity improvements over generic DMol of roughly $2\%$, increases the novelty of the method, and further improves the running time due to reductions in the graph size. 

---
# Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Encoding 

**Authors**: Haiping Liu, Hongpeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06308)  

**Abstract**: Rotary Position Embedding (RoPE) is widely adopted in Transformers due to its ability to encode relative positions with high efficiency and extrapolation capability. However, existing RoPE variants lack a unified theoretical foundation, especially in higher dimensions. In this paper, we propose a systematic mathematical framework for RoPE grounded in Lie group and Lie algebra theory. We identify two core properties of RoPE, named relativity and reversibility, and derive general constraints and constructions for valid RoPE in 1D, 2D, and N-dimensional (ND). We prove that RoPE must lie in the basis of a maximal abelian subalgebra (MASA) of the special orthogonal Lie algebra, and show that standard RoPE corresponds to the maximal toral subalgebra. Furthermore, we propose to model inter-dimensional interactions by learning an orthogonal basis transformation. Our framework unifies and explains existing RoPE designs, while enabling principled extensions to new modalities and tasks. 

---
# Optimizing Large Language Models: Metrics, Energy Efficiency, and Case Study Insights 

**Authors**: Tahniat Khan, Soroor Motie, Sedef Akinli Kocak, Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2504.06307)  

**Abstract**: The rapid adoption of large language models (LLMs) has led to significant energy consumption and carbon emissions, posing a critical challenge to the sustainability of generative AI technologies. This paper explores the integration of energy-efficient optimization techniques in the deployment of LLMs to address these environmental concerns. We present a case study and framework that demonstrate how strategic quantization and local inference techniques can substantially lower the carbon footprints of LLMs without compromising their operational effectiveness. Experimental results reveal that these methods can reduce energy consumption and carbon emissions by up to 45\% post quantization, making them particularly suitable for resource-constrained environments. The findings provide actionable insights for achieving sustainability in AI while maintaining high levels of accuracy and responsiveness. 

---
# Predicting Survivability of Cancer Patients with Metastatic Patterns Using Explainable AI 

**Authors**: Polycarp Nalela, Deepthi Rao, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06306)  

**Abstract**: Cancer remains a leading global health challenge and a major cause of mortality. This study leverages machine learning (ML) to predict the survivability of cancer patients with metastatic patterns using the comprehensive MSK-MET dataset, which includes genomic and clinical data from 25,775 patients across 27 cancer types. We evaluated five ML models-XGBoost, Naïve Bayes, Decision Tree, Logistic Regression, and Random Fores using hyperparameter tuning and grid search. XGBoost emerged as the best performer with an area under the curve (AUC) of 0.82. To enhance model interpretability, SHapley Additive exPlanations (SHAP) were applied, revealing key predictors such as metastatic site count, tumor mutation burden, fraction of genome altered, and organ-specific metastases. Further survival analysis using Kaplan-Meier curves, Cox Proportional Hazards models, and XGBoost Survival Analysis identified significant predictors of patient outcomes, offering actionable insights for clinicians. These findings could aid in personalized prognosis and treatment planning, ultimately improving patient care. 

---
# Well2Flow: Reconstruction of reservoir states from sparse wells using score-based generative models 

**Authors**: Shiqin Zeng, Haoyun Li, Abhinav Prakash Gahlot, Felix J. Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06305)  

**Abstract**: This study investigates the use of score-based generative models for reservoir simulation, with a focus on reconstructing spatially varying permeability and saturation fields in saline aquifers, inferred from sparse observations at two well locations. By modeling the joint distribution of permeability and saturation derived from high-fidelity reservoir simulations, the proposed neural network is trained to learn the complex spatiotemporal dynamics governing multiphase fluid flow in porous media. During inference, the framework effectively reconstructs both permeability and saturation fields by conditioning on sparse vertical profiles extracted from well log data. This approach introduces a novel methodology for incorporating physical constraints and well log guidance into generative models, significantly enhancing the accuracy and physical plausibility of the reconstructed subsurface states. Furthermore, the framework demonstrates strong generalization capabilities across varying geological scenarios, highlighting its potential for practical deployment in data-scarce reservoir management tasks. 

---
# On the Effectiveness and Generalization of Race Representations for Debiasing High-Stakes Decisions 

**Authors**: Dang Nguyen, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06303)  

**Abstract**: Understanding and mitigating biases is critical for the adoption of large language models (LLMs) in high-stakes decision-making. We introduce Admissions and Hiring, decision tasks with hypothetical applicant profiles where a person's race can be inferred from their name, as simplified test beds for racial bias. We show that Gemma 2B Instruct and LLaMA 3.2 3B Instruct exhibit strong biases. Gemma grants admission to 26% more White than Black applicants, and LLaMA hires 60% more Asian than White applicants. We demonstrate that these biases are resistant to prompt engineering: multiple prompting strategies all fail to promote fairness. In contrast, using distributed alignment search, we can identify "race subspaces" within model activations and intervene on them to debias model decisions. Averaging the representation across all races within the subspaces reduces Gemma's bias by 37-57%. Finally, we examine the generalizability of Gemma's race subspaces, and find limited evidence for generalization, where changing the prompt format can affect the race representation. Our work suggests mechanistic approaches may provide a promising venue for improving the fairness of LLMs, but a universal race representation remains elusive. 

---
# Resurrecting Socrates in the Age of AI: A Study Protocol for Evaluating a Socratic Tutor to Support Research Question Development in Higher Education 

**Authors**: Ben Degen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06294)  

**Abstract**: Formulating research questions is a foundational yet challenging academic skill, one that generative AI systems often oversimplify by offering instant answers at the expense of student reflection. This protocol lays out a study grounded in constructivist learning theory to evaluate a novel AI-based Socratic Tutor, designed to foster cognitive engagement and scaffold research question development in higher education. Anchored in dialogic pedagogy, the tutor engages students through iterative, reflective questioning, aiming to promote System 2 thinking and counteract overreliance on AI-generated outputs. In a quasi-experimental design, approximately 80 German pre-service biology teacher students will be randomly assigned to one of two groups: an AI Socratic Tutor condition and an uninstructed chatbot control. Across multiple cycles, students are expected to formulate research questions based on background texts, with quality assessed through double-blind expert review. The study also examines transfer of skills to novel phenomena and captures student perceptions through mixed-methods analysis, including surveys, interviews and reflective journals. This study aims to advance the understanding of how generative AI can be pedagogically aligned to support, not replace, human cognition and offers design principles for human-AI collaboration in education. 

---
# Temporal-contextual Event Learning for Pedestrian Crossing Intent Prediction 

**Authors**: Hongbin Liang, Hezhe Qiao, Wei Huang, Qizhou Wang, Mingsheng Shang, Lin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06292)  

**Abstract**: Ensuring the safety of vulnerable road users through accurate prediction of pedestrian crossing intention (PCI) plays a crucial role in the context of autonomous and assisted driving. Analyzing the set of observation video frames in ego-view has been widely used in most PCI prediction methods to forecast the cross intent. However, they struggle to capture the critical events related to pedestrian behaviour along the temporal dimension due to the high redundancy of the video frames, which results in the sub-optimal performance of PCI prediction. Our research addresses the challenge by introducing a novel approach called \underline{T}emporal-\underline{c}ontextual Event \underline{L}earning (TCL). The TCL is composed of the Temporal Merging Module (TMM), which aims to manage the redundancy by clustering the observed video frames into multiple key temporal events. Then, the Contextual Attention Block (CAB) is employed to adaptively aggregate multiple event features along with visual and non-visual data. By synthesizing the temporal feature extraction and contextual attention on the key information across the critical events, TCL can learn expressive representation for the PCI prediction. Extensive experiments are carried out on three widely adopted datasets, including PIE, JAAD-beh, and JAAD-all. The results show that TCL substantially surpasses the state-of-the-art methods. Our code can be accessed at this https URL. 

---
# Dynamic Evaluation Framework for Personalized and Trustworthy Agents: A Multi-Session Approach to Preference Adaptability 

**Authors**: Chirag Shah, Hideo Joho, Kirandeep Kaur, Preetam Prabhu Srikar Dammu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06277)  

**Abstract**: Recent advancements in generative AI have significantly increased interest in personalized agents. With increased personalization, there is also a greater need for being able to trust decision-making and action taking capabilities of these agents. However, the evaluation methods for these agents remain outdated and inadequate, often failing to capture the dynamic and evolving nature of user interactions. In this conceptual article, we argue for a paradigm shift in evaluating personalized and adaptive agents. We propose a comprehensive novel framework that models user personas with unique attributes and preferences. In this framework, agents interact with these simulated users through structured interviews to gather their preferences and offer customized recommendations. These recommendations are then assessed dynamically using simulations driven by Large Language Models (LLMs), enabling an adaptive and iterative evaluation process. Our flexible framework is designed to support a variety of agents and applications, ensuring a comprehensive and versatile evaluation of recommendation strategies that focus on proactive, personalized, and trustworthy aspects. 

---
# A Cascaded Architecture for Extractive Summarization of Multimedia Content via Audio-to-Text Alignment 

**Authors**: Tanzir Hossain, Ar-Rafi Islam, Md. Sabbir Hossain, Annajiat Alim Rasel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06275)  

**Abstract**: This study presents a cascaded architecture for extractive summarization of multimedia content via audio-to-text alignment. The proposed framework addresses the challenge of extracting key insights from multimedia sources like YouTube videos. It integrates audio-to-text conversion using Microsoft Azure Speech with advanced extractive summarization models, including Whisper, Pegasus, and Facebook BART XSum. The system employs tools such as Pytube, Pydub, and SpeechRecognition for content retrieval, audio extraction, and transcription. Linguistic analysis is enhanced through named entity recognition and semantic role labeling. Evaluation using ROUGE and F1 scores demonstrates that the cascaded architecture outperforms conventional summarization methods, despite challenges like transcription errors. Future improvements may include model fine-tuning and real-time processing. This study contributes to multimedia summarization by improving information retrieval, accessibility, and user experience. 

---
# Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06274)  

**Abstract**: Group recommender systems aim to generate recommendations that align with the collective preferences of a group, introducing challenges that differ significantly from those in individual recommendation scenarios. This paper presents Joint Group Profiling and Recommendation via Deep Neural Network-based Multi-Task Learning, a framework that unifies group profiling and recommendation tasks within a single model. By jointly learning these tasks, the model develops a deeper understanding of group dynamics, leading to improved recommendation accuracy. The shared representations between the two tasks facilitate the discovery of latent features essential to both, resulting in richer and more informative group embeddings. To further enhance performance, an attention mechanism is integrated to dynamically evaluate the relevance of different group features and item attributes, ensuring the model prioritizes the most impactful information. Experiments and evaluations on real-world datasets demonstrate that our multi-task learning approach consistently outperforms baseline models in terms of accuracy, validating its effectiveness and robustness. 

---
# A Diverse and Effective Retrieval-Based Debt Collection System with Expert Knowledge 

**Authors**: Jiaming Luo, Weiyi Luo, Guoqing Sun, Mengchen Zhu, Haifeng Tang, Kunyao Lan, Mengyue Wu, Kenny Q. Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06273)  

**Abstract**: Designing effective debt collection systems is crucial for improving operational efficiency and reducing costs in the financial industry. However, the challenges of maintaining script diversity, contextual relevance, and coherence make this task particularly difficult. This paper presents a debt collection system based on real debtor-collector data from a major commercial bank. We construct a script library from real-world debt collection conversations, and propose a two-stage retrieval based response system for contextual relevance. Experimental results show that our system improves script diversity, enhances response relevance, and achieves practical deployment efficiency through knowledge distillation. This work offers a scalable and automated solution, providing valuable insights for advancing debt collection practices in real-world applications. 

---
# RAVEN: An Agentic Framework for Multimodal Entity Discovery from Large-Scale Video Collections 

**Authors**: Kevin Dela Rosa  

**Link**: [PDF](https://arxiv.org/pdf/2504.06272)  

**Abstract**: We present RAVEN an adaptive AI agent framework designed for multimodal entity discovery and retrieval in large-scale video collections. Synthesizing information across visual, audio, and textual modalities, RAVEN autonomously processes video data to produce structured, actionable representations for downstream tasks. Key contributions include (1) a category understanding step to infer video themes and general-purpose entities, (2) a schema generation mechanism that dynamically defines domain-specific entities and attributes, and (3) a rich entity extraction process that leverages semantic retrieval and schema-guided prompting. RAVEN is designed to be model-agnostic, allowing the integration of different vision-language models (VLMs) and large language models (LLMs) based on application-specific requirements. This flexibility supports diverse applications in personalized search, content discovery, and scalable information retrieval, enabling practical applications across vast datasets. 

---
# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06271)  

**Abstract**: Large language models (LLMs) excel in question-answering (QA) tasks, and retrieval-augmented generation (RAG) enhances their precision by incorporating external evidence from diverse sources like web pages, databases, and knowledge graphs. However, current RAG methods rely on agent-specific strategies for individual data sources, posing challenges low-resource or black-box environments and complicates operations when evidence is fragmented across sources. To address these limitations, we propose ER-RAG, a framework that unifies evidence integration across heterogeneous data sources using the Entity-Relationship (ER) model. ER-RAG standardizes entity retrieval and relationship querying through ER-based APIs with GET and JOIN operations. It employs a two-stage generation process: first, a preference optimization module selects optimal sources; second, another module constructs API chains based on source schemas. This unified approach allows efficient fine-tuning and seamless integration across diverse data sources. ER-RAG demonstrated its effectiveness by winning all three tracks of the 2024 KDDCup CRAG Challenge, achieving performance on par with commercial RAG pipelines using an 8B LLM backbone. It outperformed hybrid competitors by 3.1% in LLM score and accelerated retrieval by 5.5X. 

---
# Addressing Cold-start Problem in Click-Through Rate Prediction via Supervised Diffusion Modeling 

**Authors**: Wenqiao Zhu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06270)  

**Abstract**: Predicting Click-Through Rates is a crucial function within recommendation and advertising platforms, as the output of CTR prediction determines the order of items shown to users. The Embedding \& MLP paradigm has become a standard approach for industrial recommendation systems and has been widely deployed. However, this paradigm suffers from cold-start problems, where there is either no or only limited user action data available, leading to poorly learned ID embeddings. The cold-start problem hampers the performance of new items. To address this problem, we designed a novel diffusion model to generate a warmed-up embedding for new items. Specifically, we define a novel diffusion process between the ID embedding space and the side information space. In addition, we can derive a sub-sequence from the diffusion steps to expedite training, given that our diffusion model is non-Markovian. Our diffusion model is supervised by both the variational inference and binary cross-entropy objectives, enabling it to generate warmed-up embeddings for items in both the cold-start and warm-up phases. Additionally, we have conducted extensive experiments on three recommendation datasets. The results confirmed the effectiveness of our approach. 

---
# EXCLAIM: An Explainable Cross-Modal Agentic System for Misinformation Detection with Hierarchical Retrieval 

**Authors**: Yin Wu, Zhengxuan Zhang, Fuling Wang, Yuyu Luo, Hui Xiong, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.06269)  

**Abstract**: Misinformation continues to pose a significant challenge in today's information ecosystem, profoundly shaping public perception and behavior. Among its various manifestations, Out-of-Context (OOC) misinformation is particularly obscure, as it distorts meaning by pairing authentic images with misleading textual narratives. Existing methods for detecting OOC misinformation predominantly rely on coarse-grained similarity metrics between image-text pairs, which often fail to capture subtle inconsistencies or provide meaningful explainability. While multi-modal large language models (MLLMs) demonstrate remarkable capabilities in visual reasoning and explanation generation, they have not yet demonstrated the capacity to address complex, fine-grained, and cross-modal distinctions necessary for robust OOC detection. To overcome these limitations, we introduce EXCLAIM, a retrieval-based framework designed to leverage external knowledge through multi-granularity index of multi-modal events and entities. Our approach integrates multi-granularity contextual analysis with a multi-agent reasoning architecture to systematically evaluate the consistency and integrity of multi-modal news content. Comprehensive experiments validate the effectiveness and resilience of EXCLAIM, demonstrating its ability to detect OOC misinformation with 4.3% higher accuracy compared to state-of-the-art approaches, while offering explainable and actionable insights. 

---
# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

---
# Leveraging LLMs for User Stories in AI Systems: UStAI Dataset 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00513)  

**Abstract**: AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use. 

---
# Multi-objective Optimization in CPU Design Space Exploration: Attention is All You Need 

**Authors**: Runzhen Xue, Hao Wu, Mingyu Yan, Ziheng Xiao, Xiaochun Ye, Dongrui Fan  

**Link**: [PDF](https://arxiv.org/pdf/2410.18368)  

**Abstract**: Design space exploration (DSE) enables architects to systematically evaluate various design options, guiding decisions on the most suitable configurations to meet specific objectives such as optimizing performance, power, and area. However, the growing complexity of modern CPUs has dramatically increased the number of micro-architectural parameters and expanded the overall design space, making DSE more challenging and time-consuming. Existing DSE frameworks struggle in large-scale design spaces due to inaccurate models and limited insights into parameter impact, hindering efficient identification of optimal micro-architectures within tight timeframes.
In this work, we introduce AttentionDSE. Its key idea is to use the attention mechanism to establish a direct mapping of micro-architectural parameters to their contributions to predicted performance. This approach enhances both the prediction accuracy and interpretability of the performance model. Furthermore, the weights are dynamically adjusted, enabling the model to respond to design changes and effectively pinpoint the key micro-architectural parameters/components responsible for performance bottlenecks. Thus, AttentionDSE accurately, purposefully, and rapidly discovers optimal designs. Experiments on SPEC 2017 demonstrate that AttentionDSE significantly reduces exploration time by over 80\% and achieves 3.9\% improvement in Pareto Hypervolume compared to state-of-the-art DSE frameworks while maintaining superior prediction accuracy and efficiency with an increasing number of parameters. 

---
# CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models 

**Authors**: Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, JingSong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2404.01663)  

**Abstract**: Open large language models (LLMs) have significantly advanced the field of natural language processing, showcasing impressive performance across various this http URL the significant advancements in LLMs, their effective operation still relies heavily on human input to accurately guide the dialogue flow, with agent tuning being a crucial optimization technique that involves human adjustments to the model for better response to such this http URL this dependency, our work introduces the TinyAgent model, trained on a meticulously curated high-quality dataset. We also present the Collaborative Multi-Agent Tuning (CMAT) framework, an innovative system designed to augment language agent capabilities through adaptive weight updates based on environmental feedback. This framework fosters collaborative learning and real-time adaptation among multiple intelligent agents, enhancing their context-awareness and long-term memory. In this research, we propose a new communication agent framework that integrates multi-agent systems with environmental feedback mechanisms, offering a scalable method to explore cooperative behaviors. Notably, our TinyAgent-7B model exhibits performance on par with GPT-3.5, despite having fewer parameters, signifying a substantial improvement in the efficiency and effectiveness of LLMs. 

---
