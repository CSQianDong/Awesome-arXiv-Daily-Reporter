# AI Judges in Design: Statistical Perspectives on Achieving Human Expert Equivalence With Vision-Language Models 

**Authors**: Kristen M. Edwards, Farnaz Tehranchi, Scarlett R. Miller, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00938)  

**Abstract**: The subjective evaluation of early stage engineering designs, such as conceptual sketches, traditionally relies on human experts. However, expert evaluations are time-consuming, expensive, and sometimes inconsistent. Recent advances in vision-language models (VLMs) offer the potential to automate design assessments, but it is crucial to ensure that these AI ``judges'' perform on par with human experts. However, no existing framework assesses expert equivalence. This paper introduces a rigorous statistical framework to determine whether an AI judge's ratings match those of human experts. We apply this framework in a case study evaluating four VLM-based judges on key design metrics (uniqueness, creativity, usefulness, and drawing quality). These AI judges employ various in-context learning (ICL) techniques, including uni- vs. multimodal prompts and inference-time reasoning. The same statistical framework is used to assess three trained novices for expert-equivalence. Results show that the top-performing AI judge, using text- and image-based ICL with reasoning, achieves expert-level agreement for uniqueness and drawing quality and outperforms or matches trained novices across all metrics. In 6/6 runs for both uniqueness and creativity, and 5/6 runs for both drawing quality and usefulness, its agreement with experts meets or exceeds that of the majority of trained novices. These findings suggest that reasoning-supported VLM models can achieve human-expert equivalence in design evaluation. This has implications for scaling design evaluation in education and practice, and provides a general statistical framework for validating AI judges in other domains requiring subjective content evaluation. 

---
# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning 

**Authors**: Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00907)  

**Abstract**: Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL. 

---
# Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents 

**Authors**: Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00906)  

**Abstract**: Computer use agents automate digital tasks by directly interacting with graphical user interfaces (GUIs) on computers and mobile devices, offering significant potential to enhance human productivity by completing an open-ended space of user queries. However, current agents face significant challenges: imprecise grounding of GUI elements, difficulties with long-horizon task planning, and performance bottlenecks from relying on single generalist models for diverse cognitive tasks. To this end, we introduce Agent S2, a novel compositional framework that delegates cognitive responsibilities across various generalist and specialist models. We propose a novel Mixture-of-Grounding technique to achieve precise GUI localization and introduce Proactive Hierarchical Planning, dynamically refining action plans at multiple temporal scales in response to evolving observations. Evaluations demonstrate that Agent S2 establishes new state-of-the-art (SOTA) performance on three prominent computer use benchmarks. Specifically, Agent S2 achieves 18.9% and 32.7% relative improvements over leading baseline agents such as Claude Computer Use and UI-TARS on the OSWorld 15-step and 50-step evaluation. Moreover, Agent S2 generalizes effectively to other operating systems and applications, surpassing previous best methods by 52.8% on WindowsAgentArena and by 16.52% on AndroidWorld relatively. Code available at this https URL. 

---
# Investigating Large Language Models in Diagnosing Students' Cognitive Skills in Math Problem-solving 

**Authors**: Hyoungwook Jin, Yoonsu Kim, Dongyun Jung, Seungju Kim, Kiyoon Choi, Jinho Son, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00843)  

**Abstract**: Mathematics learning entails mastery of both content knowledge and cognitive processing of knowing, applying, and reasoning with it. Automated math assessment primarily has focused on grading students' exhibition of content knowledge by finding textual evidence, such as specific numbers, formulas, and statements. Recent advancements in problem-solving, image recognition, and reasoning capabilities of large language models (LLMs) show promise for nuanced evaluation of students' cognitive skills. Diagnosing cognitive skills needs to infer students' thinking processes beyond textual evidence, which is an underexplored task in LLM-based automated assessment. In this work, we investigate how state-of-the-art LLMs diagnose students' cognitive skills in mathematics. We constructed MathCog, a novel benchmark dataset comprising 639 student responses to 110 expert-curated middle school math problems, each annotated with detailed teachers' diagnoses based on cognitive skill checklists. Using MathCog, we evaluated 16 closed and open LLMs of varying model sizes and vendors. Our evaluation reveals that even the state-of-the-art LLMs struggle with the task, all F1 scores below 0.5, and tend to exhibit strong false confidence for incorrect cases ($r_s=.617$). We also found that model size positively correlates with the diagnosis performance ($r_s=.771$). Finally, we discuss the implications of these findings, the overconfidence issue, and directions for improving automated cognitive skill diagnosis. 

---
# Example-Based Concept Analysis Framework for Deep Weather Forecast Models 

**Authors**: Soyeon Kim, Junho Choi, Subeen Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00831)  

**Abstract**: To improve the trustworthiness of an AI model, finding consistent, understandable representations of its inference process is essential. This understanding is particularly important in high-stakes operations such as weather forecasting, where the identification of underlying meteorological mechanisms is as critical as the accuracy of the predictions. Despite the growing literature that addresses this issue through explainable AI, the applicability of their solutions is often limited due to their AI-centric development. To fill this gap, we follow a user-centric process to develop an example-based concept analysis framework, which identifies cases that follow a similar inference process as the target instance in a target model and presents them in a user-comprehensible format. Our framework provides the users with visually and conceptually analogous examples, including the probability of concept assignment to resolve ambiguities in weather mechanisms. To bridge the gap between vector representations identified from models and human-understandable explanations, we compile a human-annotated concept dataset and implement a user interface to assist domain experts involved in the the framework development. 

---
# Explainable AI-Based Interface System for Weather Forecasting Model 

**Authors**: Soyeon Kim, Junho Choi, Yeji Choi, Subeen Lee, Artyom Stitsyuk, Minkyoung Park, Seongyeop Jeong, Youhyun Baek, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00795)  

**Abstract**: Machine learning (ML) is becoming increasingly popular in meteorological decision-making. Although the literature on explainable artificial intelligence (XAI) is growing steadily, user-centered XAI studies have not extend to this domain yet. This study defines three requirements for explanations of black-box models in meteorology through user studies: statistical model performance for different rainfall scenarios to identify model bias, model reasoning, and the confidence of model outputs. Appropriate XAI methods are mapped to each requirement, and the generated explanations are tested quantitatively and qualitatively. An XAI interface system is designed based on user feedback. The results indicate that the explanations increase decision utility and user trust. Users prefer intuitive explanations over those based on XAI algorithms even for potentially easy-to-recognize examples. These findings can provide evidence for future research on user-centered XAI algorithms, as well as a basis to improve the usability of AI systems in practice. 

---
# Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scale Test-Time Compute 

**Authors**: Jianhao Chen, Zishuo Xun, Bocheng Zhou, Han Qi, Qiaosheng Zhang, Yang Chen, Wei Hu, Yuzhong Qu, Wanli Ouyang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00762)  

**Abstract**: This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm. 

---
# Personality-Driven Decision-Making in LLM-Based Autonomous Agents 

**Authors**: Lewis Newsham, Daniel Prince  

**Link**: [PDF](https://arxiv.org/pdf/2504.00727)  

**Abstract**: The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies. 

---
# Towards Responsible and Trustworthy Educational Data Mining: Comparing Symbolic, Sub-Symbolic, and Neural-Symbolic AI Methods 

**Authors**: Danial Hooshyar, Eve Kikas, Yeongwook Yang, Gustav Šír, Raija Hämäläinen, Tommi Kärkkäinen, Roger Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00615)  

**Abstract**: Given the demand for responsible and trustworthy AI for education, this study evaluates symbolic, sub-symbolic, and neural-symbolic AI (NSAI) in terms of generalizability and interpretability. Our extensive experiments on balanced and imbalanced self-regulated learning datasets of Estonian primary school students predicting 7th-grade mathematics national test performance showed that symbolic and sub-symbolic methods performed well on balanced data but struggled to identify low performers in imbalanced datasets. Interestingly, symbolic and sub-symbolic methods emphasized different factors in their decision-making: symbolic approaches primarily relied on cognitive and motivational factors, while sub-symbolic methods focused more on cognitive aspects, learned knowledge, and the demographic variable of gender -- yet both largely overlooked metacognitive factors. The NSAI method, on the other hand, showed advantages by: (i) being more generalizable across both classes -- even in imbalanced datasets -- as its symbolic knowledge component compensated for the underrepresented class; and (ii) relying on a more integrated set of factors in its decision-making, including motivation, (meta)cognition, and learned knowledge, thus offering a comprehensive and theoretically grounded interpretability framework. These contrasting findings highlight the need for a holistic comparison of AI methods before drawing conclusions based solely on predictive performance. They also underscore the potential of hybrid, human-centered NSAI methods to address the limitations of other AI families and move us closer to responsible AI for education. Specifically, by enabling stakeholders to contribute to AI design, NSAI aligns learned patterns with theoretical constructs, incorporates factors like motivation and metacognition, and strengthens the trustworthiness and responsibility of educational data mining. 

---
# LLM-Guided Search for Deletion-Correcting Codes 

**Authors**: Franziska Weindel, Reinhard Heckel  

**Link**: [PDF](https://arxiv.org/pdf/2504.00613)  

**Abstract**: Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths \( n = 12, 13 \), and \( 16 \), establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes. 

---
# Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems? 

**Authors**: Kai Yan, Yufei Xu, Zhengyin Du, Xuesong Yao, Zheyu Wang, Xiaowen Guo, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00509)  

**Abstract**: The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surpassing human intelligence. However, is the LLMs' remarkable reasoning ability indeed comes from true intelligence by human standards, or are they simply reciting solutions witnessed during training at an Internet level? To study this problem, we propose RoR-Bench, a novel, multi-modal benchmark for detecting LLM's recitation behavior when asked simple reasoning problems but with conditions subtly shifted, and conduct empirical analysis on our benchmark. Surprisingly, we found existing cutting-edge LLMs unanimously exhibits extremely severe recitation behavior; by changing one phrase in the condition, top models such as OpenAI-o1 and DeepSeek-R1 can suffer $60\%$ performance loss on elementary school-level arithmetic and reasoning problems. Such findings are a wake-up call to the LLM community that compels us to re-evaluate the true intelligence level of cutting-edge LLMs. 

---
# Hawkeye:Efficient Reasoning with Model Collaboration 

**Authors**: Jianshu She, Zhuohao Li, Zhemin Huang, Qi Li, Peiran Xu, Haonan Li, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2504.00424)  

**Abstract**: Chain-of-Thought (CoT) reasoning has demonstrated remarkable effectiveness in enhancing the reasoning abilities of large language models (LLMs). However, its efficiency remains a challenge due to the generation of excessive intermediate reasoning tokens, which introduce semantic redundancy and overly detailed reasoning steps. Moreover, computational expense and latency are significant concerns, as the cost scales with the number of output tokens, including those intermediate steps. In this work, we observe that most CoT tokens are unnecessary, and retaining only a small portion of them is sufficient for producing high-quality responses. Inspired by this, we propose HAWKEYE, a novel post-training and inference framework where a large model produces concise CoT instructions to guide a smaller model in response generation. HAWKEYE quantifies redundancy in CoT reasoning and distills high-density information via reinforcement learning. By leveraging these concise CoTs, HAWKEYE is able to expand responses while reducing token usage and computational cost significantly. Our evaluation shows that HAWKEYE can achieve comparable response quality using only 35% of the full CoTs, while improving clarity, coherence, and conciseness by approximately 10%. Furthermore, HAWKEYE can accelerate end-to-end reasoning by up to 3.4x on complex math tasks while reducing inference cost by up to 60%. HAWKEYE will be open-sourced and the models will be available soon. 

---
# CyberBOT: Towards Reliable Cybersecurity Education via Ontology-Grounded Retrieval Augmented Generation 

**Authors**: Chengshuai Zhao, Riccardo De Maria, Tharindu Kumarage, Kumar Satvik Chaudhary, Garima Agrawal, Yiwen Li, Jongchan Park, Yuli Deng, Ying-Chih Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00389)  

**Abstract**: Advancements in large language models (LLMs) have enabled the development of intelligent educational tools that support inquiry-based learning across technical domains. In cybersecurity education, where accuracy and safety are paramount, systems must go beyond surface-level relevance to provide information that is both trustworthy and domain-appropriate. To address this challenge, we introduce CyberBOT, a question-answering chatbot that leverages a retrieval-augmented generation (RAG) pipeline to incorporate contextual information from course-specific materials and validate responses using a domain-specific cybersecurity ontology. The ontology serves as a structured reasoning layer that constrains and verifies LLM-generated answers, reducing the risk of misleading or unsafe guidance. CyberBOT has been deployed in a large graduate-level course at Arizona State University (ASU), where more than one hundred students actively engage with the system through a dedicated web-based platform. Computational evaluations in lab environments highlight the potential capacity of CyberBOT, and a forthcoming field study will evaluate its pedagogical impact. By integrating structured domain reasoning with modern generative capabilities, CyberBOT illustrates a promising direction for developing reliable and curriculum-aligned AI applications in specialized educational contexts. 

---
# Collaborative LLM Numerical Reasoning with Local Data Protection 

**Authors**: Min Zhang, Yuzhe Lu, Yun Zhou, Panpan Xu, Lin Lee Cheong, Chang-Tien Lu, Haozhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00299)  

**Abstract**: Numerical reasoning over documents, which demands both contextual understanding and logical inference, is challenging for low-capacity local models deployed on computation-constrained devices. Although such complex reasoning queries could be routed to powerful remote models like GPT-4, exposing local data raises significant data leakage concerns. Existing mitigation methods generate problem descriptions or examples for remote assistance. However, the inherent complexity of numerical reasoning hinders the local model from generating logically equivalent queries and accurately inferring answers with remote guidance. In this paper, we present a model collaboration framework with two key innovations: (1) a context-aware synthesis strategy that shifts the query domains while preserving logical consistency; and (2) a tool-based answer reconstruction approach that reuses the remote-generated problem-solving pattern with code snippets. Experimental results demonstrate that our method achieves better reasoning accuracy than solely using local models while providing stronger data protection than fully relying on remote models. Furthermore, our method improves accuracy by 16.2% - 43.6% while reducing data leakage by 2.3% - 44.6% compared to existing data protection approaches. 

---
# Exploration and Adaptation in Non-Stationary Tasks with Diffusion Policies 

**Authors**: Gunbir Singh Baveja  

**Link**: [PDF](https://arxiv.org/pdf/2504.00280)  

**Abstract**: This paper investigates the application of Diffusion Policy in non-stationary, vision-based RL settings, specifically targeting environments where task dynamics and objectives evolve over time. Our work is grounded in practical challenges encountered in dynamic real-world scenarios such as robotics assembly lines and autonomous navigation, where agents must adapt control strategies from high-dimensional visual inputs. We apply Diffusion Policy -- which leverages iterative stochastic denoising to refine latent action representations-to benchmark environments including Procgen and PointMaze. Our experiments demonstrate that, despite increased computational demands, Diffusion Policy consistently outperforms standard RL methods such as PPO and DQN, achieving higher mean and maximum rewards with reduced variability. These findings underscore the approach's capability to generate coherent, contextually relevant action sequences in continuously shifting conditions, while also highlighting areas for further improvement in handling extreme non-stationarity. 

---
# Rack Position Optimization in Large-Scale Heterogeneous Data Centers 

**Authors**: Chang-Lin Chen, Jiayu Chen, Tian Lan, Zhaoxia Zhao, Hongbo Dong, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2504.00277)  

**Abstract**: As rapidly growing AI computational demands accelerate the need for new hardware installation and maintenance, this work explores optimal data center resource management by balancing operational efficiency with fault tolerance through strategic rack positioning considering diverse resources and locations. Traditional mixed-integer programming (MIP) approaches often struggle with scalability, while heuristic methods may result in significant sub-optimality. To address these issues, this paper presents a novel two-tier optimization framework using a high-level deep reinforcement learning (DRL) model to guide a low-level gradient-based heuristic for local search. The high-level DRL agent employs Leader Reward for optimal rack type ordering, and the low-level heuristic efficiently maps racks to positions, minimizing movement counts and ensuring fault-tolerant resource distribution. This approach allows scalability to over 100,000 positions and 100 rack types. Our method outperformed the gradient-based heuristic by 7\% on average and the MIP solver by over 30\% in objective value. It achieved a 100\% success rate versus MIP's 97.5\% (within a 20-minute limit), completing in just 2 minutes compared to MIP's 1630 minutes (i.e., almost 4 orders of magnitude improvement). Unlike the MIP solver, which showed performance variability under time constraints and high penalties, our algorithm consistently delivered stable, efficient results - an essential feature for large-scale data center management. 

---
# Large Language Models in Numberland: A Quick Test of Their Numerical Reasoning Abilities 

**Authors**: Roussel Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2504.00226)  

**Abstract**: An essential element of human mathematical reasoning is our number sense -- an abstract understanding of numbers and their relationships -- which allows us to solve problems involving vast number spaces using limited computational resources. Mathematical reasoning of Large Language Models (LLMs) is often tested on high-level problems (such as Olympiad challenges, geometry, word problems, and puzzles), but their low-level number sense remains less explored. We introduce "Numberland," a 100-problem test to evaluate the numerical reasoning abilities of LLM-based agents. The tasks -- basic operations, advanced calculations (e.g., exponentiation, complex numbers), prime number checks, and the 24 game -- aim to test elementary skills and their integration in solving complex and uncertain problems. We evaluated five LLM-based agents: OpenAI's o1 and o1-mini, Google Gemini, Microsoft Copilot, and Anthropic Claude. They scored 74-95% on the first three tasks that allow deterministic steps to solutions. In the 24 game, which needs trial-and-error search, performance dropped to 10-73%. We tested the top 24 solver (o1 with 73% accuracy) on 25 harder problems, and its score fell to 27%, confirming search as a bottleneck. These results, along with the types of mistakes, suggest a fragile number of LLMs, which is a bit surprising given their prowess in challenging benchmarks. The limits of LLM numerical reasoning highlight the scope of simple, targeted tests to evaluate and explain LLM math skills to ensure safe use. 

---
# LLMs for Explainable AI: A Comprehensive Survey 

**Authors**: Ahsan Bilal, David Ebert, Beiyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00125)  

**Abstract**: Large Language Models (LLMs) offer a promising approach to enhancing Explainable AI (XAI) by transforming complex machine learning outputs into easy-to-understand narratives, making model predictions more accessible to users, and helping bridge the gap between sophisticated model behavior and human interpretability. AI models, such as state-of-the-art neural networks and deep learning models, are often seen as "black boxes" due to a lack of transparency. As users cannot fully understand how the models reach conclusions, users have difficulty trusting decisions from AI models, which leads to less effective decision-making processes, reduced accountabilities, and unclear potential biases. A challenge arises in developing explainable AI (XAI) models to gain users' trust and provide insights into how models generate their outputs. With the development of Large Language Models, we want to explore the possibilities of using human language-based models, LLMs, for model explainabilities. This survey provides a comprehensive overview of existing approaches regarding LLMs for XAI, and evaluation techniques for LLM-generated explanation, discusses the corresponding challenges and limitations, and examines real-world applications. Finally, we discuss future directions by emphasizing the need for more interpretable, automated, user-centric, and multidisciplinary approaches for XAI via LLMs. 

---
# The Axiom-Based Atlas: A Structural Mapping of Theorems via Foundational Proof Vectors 

**Authors**: Harim Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00063)  

**Abstract**: The Axiom-Based Atlas is a novel framework that structurally represents mathematical theorems as proof vectors over foundational axiom systems. By mapping the logical dependencies of theorems onto vectors indexed by axioms - such as those from Hilbert geometry, Peano arithmetic, or ZFC - we offer a new way to visualize, compare, and analyze mathematical knowledge. This vector-based formalism not only captures the logical foundation of theorems but also enables quantitative similarity metrics - such as cosine distance - between mathematical results, offering a new analytic layer for structural comparison. Using heatmaps, vector clustering, and AI-assisted modeling, this atlas enables the grouping of theorems by logical structure, not just by mathematical domain. We also introduce a prototype assistant (Atlas-GPT) that interprets natural language theorems and suggests likely proof vectors, supporting future applications in automated reasoning, mathematical education, and formal verification.
This direction is partially inspired by Terence Tao's recent reflections on the convergence of symbolic and structural mathematics. The Axiom-Based Atlas aims to provide a scalable, interpretable model of mathematical reasoning that is both human-readable and AI-compatible, contributing to the future landscape of formal mathematical systems. 

---
# GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors 

**Authors**: Tian-Xing Xu, Xiangjun Gao, Wenbo Hu, Xiaoyu Li, Song-Hai Zhang, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01016)  

**Abstract**: Despite remarkable advancements in video depth estimation, existing methods exhibit inherent limitations in achieving geometric fidelity through the affine-invariant predictions, limiting their applicability in reconstruction and other metrically grounded downstream tasks. We propose GeometryCrafter, a novel framework that recovers high-fidelity point map sequences with temporal coherence from open-world videos, enabling accurate 3D/4D reconstruction, camera parameter estimation, and other depth-based applications. At the core of our approach lies a point map Variational Autoencoder (VAE) that learns a latent space agnostic to video latent distributions for effective point map encoding and decoding. Leveraging the VAE, we train a video diffusion model to model the distribution of point map sequences conditioned on the input videos. Extensive evaluations on diverse datasets demonstrate that GeometryCrafter achieves state-of-the-art 3D accuracy, temporal consistency, and generalization capability. 

---
# IntrinsiX: High-Quality PBR Generation using Image Priors 

**Authors**: Peter Kocsis, Lukas Höllein, Matthias Nießner  

**Link**: [PDF](https://arxiv.org/pdf/2504.01008)  

**Abstract**: We introduce IntrinsiX, a novel method that generates high-quality intrinsic images from text description. In contrast to existing text-to-image models whose outputs contain baked-in scene lighting, our approach predicts physically-based rendering (PBR) maps. This enables the generated outputs to be used for content creation scenarios in core graphics applications that facilitate re-lighting, editing, and texture generation tasks. In order to train our generator, we exploit strong image priors, and pre-train separate models for each PBR material component (albedo, roughness, metallic, normals). We then align these models with a new cross-intrinsic attention formulation that concatenates key and value features in a consistent fashion. This allows us to exchange information between each output modality and to obtain semantically coherent PBR predictions. To ground each intrinsic component, we propose a rendering loss which provides image-space signals to constrain the model, thus facilitating sharp details also in the output BRDF properties. Our results demonstrate detailed intrinsic generation with strong generalization capabilities that outperforms existing intrinsic image decomposition methods used with generated images by a significant margin. Finally, we show a series of applications, including re-lighting, editing, and text-conditioned room-scale PBR texture generation. 

---
# When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning 

**Authors**: Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2504.01005)  

**Abstract**: Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at this https URL. 

---
# Token embeddings violate the manifold hypothesis 

**Authors**: Michael Robinson, Sourya Dey, Tony Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01002)  

**Abstract**: To fully understand the behavior of a large language model (LLM) requires our understanding of its input space. If this input space differs from our assumption, our understanding of and conclusions about the LLM is likely flawed, regardless of its architecture. Here, we elucidate the structure of the token embeddings, the input domain for LLMs, both empirically and theoretically. We present a generalized and statistically testable model where the neighborhood of each token splits into well-defined signal and noise dimensions.
This model is based on a generalization of a manifold called a fiber bundle, so we denote our hypothesis test as the ``fiber bundle null.'' Failing to reject the null is uninformative, but rejecting it at a specific token indicates that token has a statistically significant local structure, and so is of interest to us. By running our test over several open-source LLMs, each with unique token embeddings, we find that the null is frequently rejected, and so the token subspace is provably not a fiber bundle and hence also not a manifold. As a consequence of our findings, when an LLM is presented with two semantically equivalent prompts, and if one prompt contains a token implicated by our test, that prompt will likely exhibit more output variability proportional to the local signal dimension of the token. 

---
# Zero-shot Benchmarking: A Framework for Flexible and Scalable Automatic Evaluation of Language Models 

**Authors**: José Pombal, Nuno M. Guerreiro, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.01001)  

**Abstract**: As language models improve and become capable of performing more complex tasks across modalities, evaluating them automatically becomes increasingly challenging. Developing strong and robust task-specific automatic metrics gets harder, and human-annotated test sets -- which are expensive to create -- saturate more quickly. A compelling alternative is to design reliable strategies to automate the creation of test data and evaluation, but previous attempts either rely on pre-existing data, or focus solely on individual tasks. We present Zero-shot Benchmarking (ZSB), a framework for creating high-quality benchmarks for any task by leveraging language models for both synthetic test data creation and evaluation. ZSB is simple and flexible: it requires only the creation of a prompt for data generation and one for evaluation; it is scalable to tasks and languages where collecting real-world data is costly or impractical; it is model-agnostic, allowing the creation of increasingly challenging benchmarks as models improve. To assess the effectiveness of our framework, we create benchmarks for five text-only tasks and a multi-modal one: general capabilities in four languages (English, Chinese, French, and Korean), translation, and general vision-language capabilities in English. We then rank a broad range of open and closed systems on our benchmarks. ZSB rankings consistently correlate strongly with human rankings, outperforming widely-adopted standard benchmarks. Through ablations, we find that strong benchmarks can be created with open models, and that judge model size and dataset variety are crucial drivers of performance. We release all our benchmarks, and code to reproduce our experiments and to produce new benchmarks. 

---
# MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization 

**Authors**: Siyuan Li, Luyuan Zhang, Zedong Wang, Juanxi Tian, Cheng Tan, Zicheng Liu, Chang Yu, Qingsong Xie, Haonan Lu, Haoqian Wang, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00999)  

**Abstract**: Masked Image Modeling (MIM) with Vector Quantization (VQ) has achieved great success in both self-supervised pre-training and image generation. However, most existing methods struggle to address the trade-off in shared latent space for generation quality vs. representation learning and efficiency. To push the limits of this paradigm, we propose MergeVQ, which incorporates token merging techniques into VQ-based generative models to bridge the gap between image generation and visual representation learning in a unified architecture. During pre-training, MergeVQ decouples top-k semantics from latent space with the token merge module after self-attention blocks in the encoder for subsequent Look-up Free Quantization (LFQ) and global alignment and recovers their fine-grained details through cross-attention in the decoder for reconstruction. As for the second-stage generation, we introduce MergeAR, which performs KV Cache compression for efficient raster-order prediction. Extensive experiments on ImageNet verify that MergeVQ as an AR generative model achieves competitive performance in both visual representation learning and image generation tasks while maintaining favorable token efficiency and inference speed. The code and model will be available at this https URL. 

---
# MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs 

**Authors**: Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, Yihan Cao, Hui Ren, Xiang Li, Xiaoxiao Li, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00993)  

**Abstract**: Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available. 

---
# Accelerating drug discovery with Artificial: a whole-lab orchestration and scheduling system for self-driving labs 

**Authors**: Yao Fehlis, Paul Mandel, Charles Crain, Betty Liu, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2504.00986)  

**Abstract**: Self-driving labs are transforming drug discovery by enabling automated, AI-guided experimentation, but they face challenges in orchestrating complex workflows, integrating diverse instruments and AI models, and managing data efficiently. Artificial addresses these issues with a comprehensive orchestration and scheduling system that unifies lab operations, automates workflows, and integrates AI-driven decision-making. By incorporating AI/ML models like NVIDIA BioNeMo - which facilitates molecular interaction prediction and biomolecular analysis - Artificial enhances drug discovery and accelerates data-driven research. Through real-time coordination of instruments, robots, and personnel, the platform streamlines experiments, enhances reproducibility, and advances drug discovery. 

---
# WorldScore: A Unified Evaluation Benchmark for World Generation 

**Authors**: Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00983)  

**Abstract**: We introduce the WorldScore benchmark, the first unified benchmark for world generation. We decompose world generation into a sequence of next-scene generation tasks with explicit camera trajectory-based layout specifications, enabling unified evaluation of diverse approaches from 3D and 4D scene generation to video generation models. The WorldScore benchmark encompasses a curated dataset of 3,000 test examples that span diverse worlds: static and dynamic, indoor and outdoor, photorealistic and stylized. The WorldScore metrics evaluate generated worlds through three key aspects: controllability, quality, and dynamics. Through extensive evaluation of 19 representative models, including both open-source and closed-source ones, we reveal key insights and challenges for each category of models. Our dataset, evaluation code, and leaderboard can be found at this https URL 

---
# Resource Allocation for RIS-Assisted CoMP-NOMA Networks using Reinforcement Learning 

**Authors**: Muhammad Umer, Muhammad Ahmed Mohsin, Huma Ghafoor, Syed Ali Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00975)  

**Abstract**: This thesis delves into the forefront of wireless communication by exploring the synergistic integration of three transformative technologies: STAR-RIS, CoMP, and NOMA. Driven by the ever-increasing demand for higher data rates, improved spectral efficiency, and expanded coverage in the evolving landscape of 6G development, this research investigates the potential of these technologies to revolutionize future wireless networks.
The thesis analyzes the performance gains achievable through strategic deployment of STAR-RIS, focusing on mitigating inter-cell interference, enhancing signal strength, and extending coverage to cell-edge users. Resource sharing strategies for STAR-RIS elements are explored, optimizing both transmission and reflection functionalities. Analytical frameworks are developed to quantify the benefits of STAR-RIS assisted CoMP-NOMA networks under realistic channel conditions, deriving key performance metrics such as ergodic rates and outage probabilities. Additionally, the research delves into energy-efficient design approaches for CoMP-NOMA networks incorporating RIS, proposing novel RIS configurations and optimization algorithms to achieve a balance between performance and energy consumption. Furthermore, the application of Deep Reinforcement Learning (DRL) techniques for intelligent and adaptive optimization in aerial RIS-assisted CoMP-NOMA networks is explored, aiming to maximize network sum rate while meeting user quality of service requirements. Through a comprehensive investigation of these technologies and their synergistic potential, this thesis contributes valuable insights into the future of wireless communication, paving the way for the development of more efficient, reliable, and sustainable networks capable of meeting the demands of our increasingly connected world. 

---
# SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching 

**Authors**: Yuxuan Zhu, Ali Falahati, David H. Yang, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.00970)  

**Abstract**: Large language models face significant computational and memory challenges when processing long contexts. During inference, efficient management of the key-value (KV) cache, which stores intermediate activations for autoregressive generation, is critical to reducing memory overhead and improving computational efficiency. Traditional token-level efficient KV caching methods overlook semantic information, treating tokens independently without considering their semantic relationships. Meanwhile, existing semantic-preserving KV cache management approaches often suffer from substantial memory usage and high time-to-first-token. To address these limitations, we propose SentenceKV, a novel sentence-level semantic KV caching approach designed to enhance inference efficiency while preserving semantic coherence. During prefilling, SentenceKV groups tokens based on sentence-level semantic similarity, compressing sentence representations into concise semantic vectors stored directly on the GPU, while individual KV pairs are offloaded to CPU. During decoding, SentenceKV generates tokens by selectively retrieving semantically relevant sentence-level KV entries, leveraging the semantic similarity between the prefilling-stage semantic vectors and decoding-stage queries. This ensures efficient and contextually accurate predictions, minimizing the loading of redundant or irrelevant data into GPU memory and significantly reducing memory overhead while maintaining stable inference latency, even for extremely long contexts. Extensive evaluations on benchmarks including PG-19, LongBench, and Needle-In-A-Haystack demonstrate that SentenceKV significantly outperforms state-of-the-art methods in both efficiency and memory usage, without compromising model accuracy. 

---
# HDVIO2.0: Wind and Disturbance Estimation with Hybrid Dynamics VIO 

**Authors**: Giovanni Cioffi, Leonard Bauersfeld, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00969)  

**Abstract**: Visual-inertial odometry (VIO) is widely used for state estimation in autonomous micro aerial vehicles using onboard sensors. Current methods improve VIO by incorporating a model of the translational vehicle dynamics, yet their performance degrades when faced with low-accuracy vehicle models or continuous external disturbances, like wind. Additionally, incorporating rotational dynamics in these models is computationally intractable when they are deployed in online applications, e.g., in a closed-loop control system. We present HDVIO2.0, which models full 6-DoF, translational and rotational, vehicle dynamics and tightly incorporates them into a VIO with minimal impact on the runtime. HDVIO2.0 builds upon the previous work, HDVIO, and addresses these challenges through a hybrid dynamics model combining a point-mass vehicle model with a learning-based component, with access to control commands and IMU history, to capture complex aerodynamic effects. The key idea behind modeling the rotational dynamics is to represent them with continuous-time functions. HDVIO2.0 leverages the divergence between the actual motion and the predicted motion from the hybrid dynamics model to estimate external forces as well as the robot state. Our system surpasses the performance of state-of-the-art methods in experiments using public and new drone dynamics datasets, as well as real-world flights in winds up to 25 km/h. Unlike existing approaches, we also show that accurate vehicle dynamics predictions are achievable without precise knowledge of the full vehicle state. 

---
# Enabling Efficient Processing of Spiking Neural Networks with On-Chip Learning on Commodity Neuromorphic Processors for Edge AI Systems 

**Authors**: Rachmad Vidya Wicaksana Putra, Pasindu Wickramasinghe, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.00957)  

**Abstract**: The rising demand for energy-efficient edge AI systems (e.g., mobile agents/robots) has increased the interest in neuromorphic computing, since it offers ultra-low power/energy AI computation through spiking neural network (SNN) algorithms on neuromorphic processors. However, their efficient implementation strategy has not been comprehensively studied, hence limiting SNN deployments for edge AI systems. Toward this, we propose a design methodology to enable efficient SNN processing on commodity neuromorphic processors. To do this, we first study the key characteristics of targeted neuromorphic hardware (e.g., memory and compute budgets), and leverage this information to perform compatibility analysis for network selection. Afterward, we employ a mapping strategy for efficient SNN implementation on the targeted processor. Furthermore, we incorporate an efficient on-chip learning mechanism to update the systems' knowledge for adapting to new input classes and dynamic environments. The experimental results show that the proposed methodology leads the system to achieve low latency of inference (i.e., less than 50ms for image classification, less than 200ms for real-time object detection in video streaming, and less than 1ms in keyword recognition) and low latency of on-chip learning (i.e., less than 2ms for keyword recognition), while incurring less than 250mW of processing power and less than 15mJ of energy consumption across the respective different applications and scenarios. These results show the potential of the proposed methodology in enabling efficient edge AI systems for diverse application use-cases. 

---
# Unfair Learning: GenAI Exceptionalism and Copyright Law 

**Authors**: David Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2504.00955)  

**Abstract**: This paper challenges the argument that generative artificial intelligence (GenAI) is entitled to broad immunity from copyright law for reproducing copyrighted works without authorization due to a fair use defense. It examines fair use legal arguments and eight distinct substantive arguments, contending that every legal and substantive argument favoring fair use for GenAI applies equally, if not more so, to humans. Therefore, granting GenAI exceptional privileges in this domain is legally and logically inconsistent with withholding broad fair use exemptions from individual humans. It would mean no human would need to pay for virtually any copyright work again. The solution is to take a circumspect view of any fair use claim for mass copyright reproduction by any entity and focus on the first principles of whether permitting such exceptionalism for GenAI promotes science and the arts. 

---
# IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval 

**Authors**: Bangwei Liu, Yicheng Bao, Shaohui Lin, Xuhong Wang, Xin Tan, Yingchun Wang, Yuan Xie, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00954)  

**Abstract**: Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at this https URL. 

---
# Personalized Federated Training of Diffusion Models with Privacy Guarantees 

**Authors**: Kumar Kshitij Patel, Weitong Zhang, Lingxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00952)  

**Abstract**: The scarcity of accessible, compliant, and ethically sourced data presents a considerable challenge to the adoption of artificial intelligence (AI) in sensitive fields like healthcare, finance, and biomedical research. Furthermore, access to unrestricted public datasets is increasingly constrained due to rising concerns over privacy, copyright, and competition. Synthetic data has emerged as a promising alternative, and diffusion models -- a cutting-edge generative AI technology -- provide an effective solution for generating high-quality and diverse synthetic data. In this paper, we introduce a novel federated learning framework for training diffusion models on decentralized private datasets. Our framework leverages personalization and the inherent noise in the forward diffusion process to produce high-quality samples while ensuring robust differential privacy guarantees. Our experiments show that our framework outperforms non-collaborative training methods, particularly in settings with high data heterogeneity, and effectively reduces biases and imbalances in synthetic data, resulting in fairer downstream models. 

---
# QSViT: A Methodology for Quantizing Spiking Vision Transformers 

**Authors**: Rachmad Vidya Wicaksana Putra, Saad Iftikhar, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.00948)  

**Abstract**: Vision Transformer (ViT)-based models have shown state-of-the-art performance (e.g., accuracy) in vision-based AI tasks. However, realizing their capability in resource-constrained embedded AI systems is challenging due to their inherent large memory footprints and complex computations, thereby incurring high power/energy consumption. Recently, Spiking Vision Transformer (SViT)-based models have emerged as alternate low-power ViT networks. However, their large memory footprints still hinder their applicability for resource-constrained embedded AI systems. Therefore, there is a need for a methodology to compress SViT models without degrading the accuracy significantly. To address this, we propose QSViT, a novel design methodology to compress the SViT models through a systematic quantization strategy across different network layers. To do this, our QSViT employs several key steps: (1) investigating the impact of different precision levels in different network layers, (2) identifying the appropriate base quantization settings for guiding bit precision reduction, (3) performing a guided quantization strategy based on the base settings to select the appropriate quantization setting, and (4) developing an efficient quantized network based on the selected quantization setting. The experimental results demonstrate that, our QSViT methodology achieves 22.75% memory saving and 21.33% power saving, while also maintaining high accuracy within 2.1% from that of the original non-quantized SViT model on the ImageNet dataset. These results highlight the potential of QSViT methodology to pave the way toward the efficient SViT deployments on resource-constrained embedded AI systems. 

---
# Graph Classification and Radiomics Signature for Identification of Tuberculous Meningitis 

**Authors**: Snigdha Agarwal, Ganaraja V H, Neelam Sinha, Abhilasha Indoria, Netravathi M, Jitender Saini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00943)  

**Abstract**: Introduction: Tuberculous meningitis (TBM) is a serious brain infection caused by Mycobacterium tuberculosis, characterized by inflammation of the meninges covering the brain and spinal cord. Diagnosis often requires invasive lumbar puncture (LP) and cerebrospinal fluid (CSF) analysis. Objectives: This study aims to classify TBM patients using T1-weighted (T1w) non-contrast Magnetic Resonance Imaging (MRI) scans. We hypothesize that specific brain regions, such as the interpeduncular cisterns, bone, and corpus callosum, contain visual markers that can non-invasively distinguish TBM patients from healthy controls. We propose a novel Pixel-array Graphs Classifier (PAG-Classifier) that leverages spatial relationships between neighbouring 3D pixels in a graph-based framework to extract significant features through eigen decomposition. These features are then used to train machine learning classifiers for effective patient classification. We validate our approach using a radiomics-based methodology, classifying TBM patients based on relevant radiomics features. Results: We utilized an internal dataset consisting of 52 scans, 32 from confirmed TBM patients based on mycobacteria detection in CSF, and 20 from healthy individuals. We achieved a 5-fold cross-validated average F1 score of 85.71% for cistern regions with our PAG-Classifier and 92.85% with the radiomics features classifier, surpassing current state-of-the-art benchmarks by 15% and 22%, respectively. However, bone and corpus callosum regions showed poor classification effectiveness, with average F1 scores below 50%. Conclusion: Our study suggests that algorithms like the PAG-Classifier serve as effective tools for non-invasive TBM analysis, particularly by targeting the interpeduncular cistern. Findings indicate that the bone and corpus callosum regions lack distinctive patterns for differentiation. 

---
# Role and Use of Race in AI/ML Models Related to Health 

**Authors**: Martin C. Were, Ang Li, Bradley A. Malin, Zhijun Yin, Joseph R. Coco, Benjamin X. Collins, Ellen Wright Clayton, Laurie L. Novak, Rachele Hendricks-Sturrup, Abiodun Oluyomi, Shilo Anders, Chao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00899)  

**Abstract**: The role and use of race within health-related artificial intelligence and machine learning (AI/ML) models has sparked increasing attention and controversy. Despite the complexity and breadth of related issues, a robust and holistic framework to guide stakeholders in their examination and resolution remains lacking. This perspective provides a broad-based, systematic, and cross-cutting landscape analysis of race-related challenges, structured around the AI/ML lifecycle and framed through "points to consider" to support inquiry and decision-making. 

---
# Spectral Architecture Search for Neural Networks 

**Authors**: Gianluca Peri, Lorenzo Giambagli, Lorenzo Chicchi, Duccio Fanelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.00885)  

**Abstract**: Architecture design and optimization are challenging problems in the field of artificial neural networks. Working in this context, we here present SPARCS (SPectral ARchiteCture Search), a novel architecture search protocol which exploits the spectral attributes of the inter-layer transfer matrices. SPARCS allows one to explore the space of possible architectures by spanning continuous and differentiable manifolds, thus enabling for gradient-based optimization algorithms to be eventually employed. With reference to simple benchmark models, we show that the newly proposed method yields a self-emerging architecture with a minimal degree of expressivity to handle the task under investigation and with a reduced parameter count as compared to other viable alternatives. 

---
# Improved Visual-Spatial Reasoning via R1-Zero-Like Training 

**Authors**: Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00883)  

**Abstract**: Increasing attention has been placed on improving the reasoning capacities of multi-modal large language models (MLLMs). As the cornerstone for AI agents that function in the physical realm, video-based visual-spatial intelligence (VSI) emerges as one of the most pivotal reasoning capabilities of MLLMs. This work conducts a first, in-depth study on improving the visual-spatial reasoning of MLLMs via R1-Zero-like training. Technically, we first identify that the visual-spatial reasoning capacities of small- to medium-sized Qwen2-VL models cannot be activated via Chain of Thought (CoT) prompts. We then incorporate GRPO training for improved visual-spatial reasoning, using the carefully curated VSI-100k dataset, following DeepSeek-R1-Zero. During the investigation, we identify the necessity to keep the KL penalty (even with a small value) in GRPO. With just 120 GPU hours, our vsGRPO-2B model, fine-tuned from Qwen2-VL-2B, can outperform the base model by 12.1% and surpass GPT-4o. Moreover, our vsGRPO-7B model, fine-tuned from Qwen2-VL-7B, achieves performance comparable to that of the best open-source model LLaVA-NeXT-Video-72B. Additionally, we compare vsGRPO to supervised fine-tuning and direct preference optimization baselines and observe strong performance superiority. The code and dataset will be available soon. 

---
# CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models 

**Authors**: Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00882)  

**Abstract**: Dialect translation plays a key role in enabling seamless interaction across heterogeneous database systems. However, translating SQL queries between different dialects (e.g., from PostgreSQL to MySQL) remains a challenging task due to syntactic discrepancies and subtle semantic variations. Existing approaches including manual rewriting, rule-based systems, and large language model (LLM)-based techniques often involve high maintenance effort (e.g., crafting custom translation rules) or produce unreliable results (e.g., LLM generates non-existent functions), especially when handling complex queries. In this demonstration, we present CrackSQL, the first hybrid SQL dialect translation system that combines rule and LLM-based methods to overcome these limitations. CrackSQL leverages the adaptability of LLMs to minimize manual intervention, while enhancing translation accuracy by segmenting lengthy complex SQL via functionality-based query processing. To further improve robustness, it incorporates a novel cross-dialect syntax embedding model for precise syntax alignment, as well as an adaptive local-to-global translation strategy that effectively resolves interdependent query operations. CrackSQL supports three translation modes and offers multiple deployment and access options including a web console interface, a PyPI package, and a command-line prompt, facilitating adoption across a variety of real-world use cases 

---
# m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models 

**Authors**: Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00869)  

**Abstract**: Test-time scaling has emerged as a powerful technique for enhancing the reasoning capabilities of large language models. However, its effectiveness in medical reasoning remains uncertain, as the medical domain fundamentally differs from mathematical tasks in terms of knowledge representation and decision-making processes. In this paper, we provide the first comprehensive investigation of test-time scaling for medical reasoning and present m1, a simple yet effective approach that increases a model's medical reasoning capability at inference. Our evaluation across diverse medical tasks demonstrates that test-time scaling consistently enhances medical reasoning, enabling lightweight fine-tuned models under 10B parameters to establish new state-of-the-art performance, while our 32B model rivals previous 70B-scale medical LLMs. However, we identify an optimal reasoning token budget of approximately 4K, beyond which performance may degrade due to overthinking. Budget forcing, which extends test-time computation through iterative prompts, helps models double-check answers but does not necessarily improve the overall medical QA performance and, in some cases, even introduces errors into previously correct responses. Our case-by-case analysis identifies insufficient medical knowledge as a key bottleneck that prevents further performance gains through test-time scaling. We find that increasing data scale, improving data quality, and expanding model capacity consistently enhance medical knowledge grounding, enabling continued performance improvements, particularly on challenging medical benchmarks where smaller models reach saturation. These findings underscore fundamental differences between medical and mathematical reasoning in LLMs, highlighting that enriched medical knowledge, other than increased reasoning depth alone, is essential for realizing the benefits of test-time scaling. 

---
# Investigating the Capabilities and Limitations of Machine Learning for Identifying Bias in English Language Data with Information and Heritage Professionals 

**Authors**: Lucy Havens, Benjamin Bach, Melissa Terras, Beatrice Alex  

**Link**: [PDF](https://arxiv.org/pdf/2504.00860)  

**Abstract**: Despite numerous efforts to mitigate their biases, ML systems continue to harm already-marginalized people. While predominant ML approaches assume bias can be removed and fair models can be created, we show that these are not always possible, nor desirable, goals. We reframe the problem of ML bias by creating models to identify biased language, drawing attention to a dataset's biases rather than trying to remove them. Then, through a workshop, we evaluated the models for a specific use case: workflows of information and heritage professionals. Our findings demonstrate the limitations of ML for identifying bias due to its contextual nature, the way in which approaches to mitigating it can simultaneously privilege and oppress different communities, and its inevitability. We demonstrate the need to expand ML approaches to bias and fairness, providing a mixed-methods approach to investigating the feasibility of removing bias or achieving fairness in a given ML use case. 

---
# Exploring Personalized Federated Learning Architectures for Violence Detection in Surveillance Videos 

**Authors**: Mohammad Kassir, Siba Haidar, Antoun Yaacoub  

**Link**: [PDF](https://arxiv.org/pdf/2504.00857)  

**Abstract**: The challenge of detecting violent incidents in urban surveillance systems is compounded by the voluminous and diverse nature of video data. This paper presents a targeted approach using Personalized Federated Learning (PFL) to address these issues, specifically employing the Federated Learning with Personalization Layers method within the Flower framework. Our methodology adapts learning models to the unique data characteristics of each surveillance node, effectively managing the heterogeneous and non-IID nature of surveillance video data. Through rigorous experiments conducted on balanced and imbalanced datasets, our PFL models demonstrated enhanced accuracy and efficiency, achieving up to 99.3% accuracy. This study underscores the potential of PFL to significantly improve the scalability and effectiveness of surveillance systems, offering a robust, privacy-preserving solution for violence detection in complex urban environments. 

---
# ReaLitE: Enrichment of Relation Embeddings in Knowledge Graphs using Numeric Literals 

**Authors**: Antonis Klironomos, Baifan Zhou, Zhuoxun Zheng, Gad-Elrab Mohamed, Heiko Paulheim, Evgeny Kharlamov  

**Link**: [PDF](https://arxiv.org/pdf/2504.00852)  

**Abstract**: Most knowledge graph embedding (KGE) methods tailored for link prediction focus on the entities and relations in the graph, giving little attention to other literal values, which might encode important information. Therefore, some literal-aware KGE models attempt to either integrate numerical values into the embeddings of the entities or convert these numerics into entities during preprocessing, leading to information loss. Other methods concerned with creating relation-specific numerical features assume completeness of numerical data, which does not apply to real-world graphs. In this work, we propose ReaLitE, a novel relation-centric KGE model that dynamically aggregates and merges entities' numerical attributes with the embeddings of the connecting relations. ReaLitE is designed to complement existing conventional KGE methods while supporting multiple variations for numerical aggregations, including a learnable method.
We comprehensively evaluated the proposed relation-centric embedding using several benchmarks for link prediction and node classification tasks. The results showed the superiority of ReaLitE over the state of the art in both tasks. 

---
# Global Intervention and Distillation for Federated Out-of-Distribution Generalization 

**Authors**: Zhuang Qi, Runhui Zhang, Lei Meng, Wei Wu, Yachong Zhang, Xiangxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00850)  

**Abstract**: Attribute skew in federated learning leads local models to focus on learning non-causal associations, guiding them towards inconsistent optimization directions, which inevitably results in performance degradation and unstable convergence. Existing methods typically leverage data augmentation to enhance sample diversity or employ knowledge distillation to learn invariant representations. However, the instability in the quality of generated data and the lack of domain information limit their performance on unseen samples. To address these issues, this paper presents a global intervention and distillation method, termed FedGID, which utilizes diverse attribute features for backdoor adjustment to break the spurious association between background and label. It includes two main modules, where the global intervention module adaptively decouples objects and backgrounds in images, injects background information into random samples to intervene in the sample distribution, which links backgrounds to all categories to prevent the model from treating background-label associations as causal. The global distillation module leverages a unified knowledge base to guide the representation learning of client models, preventing local models from overfitting to client-specific attributes. Experimental results on three datasets demonstrate that FedGID enhances the model's ability to focus on the main subjects in unseen data and outperforms existing methods in collaborative modeling. 

---
# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights 

**Authors**: Yuchen Liu, Lino Lerch, Luigi Palmieri, Andrey Rudenko, Sebastian Koch, Timo Ropinski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2504.00839)  

**Abstract**: Predicting human behavior in shared environments is crucial for safe and efficient human-robot interaction. Traditional data-driven methods to that end are pre-trained on domain-specific datasets, activity types, and prediction horizons. In contrast, the recent breakthroughs in Large Language Models (LLMs) promise open-ended cross-domain generalization to describe various human activities and make predictions in any context. In particular, Multimodal LLMs (MLLMs) are able to integrate information from various sources, achieving more contextual awareness and improved scene understanding. The difficulty in applying general-purpose MLLMs directly for prediction stems from their limited capacity for processing large input sequences, sensitivity to prompt design, and expensive fine-tuning. In this paper, we present a systematic analysis of applying pre-trained MLLMs for context-aware human behavior prediction. To this end, we introduce a modular multimodal human activity prediction framework that allows us to benchmark various MLLMs, input variations, In-Context Learning (ICL), and autoregressive techniques. Our evaluation indicates that the best-performing framework configuration is able to reach 92.8% semantic similarity and 66.1% exact label accuracy in predicting human behaviors in the target frame. 

---
# A Survey on Music Generation from Single-Modal, Cross-Modal, and Multi-Modal Perspectives: Data, Methods, and Challenges 

**Authors**: Shuyu Li, Shulei Ji, Zihao Wang, Songruoyao Wu, Jiaxing Yu, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00837)  

**Abstract**: Multi-modal music generation, using multiple modalities like images, video, and text alongside musical scores and audio as guidance, is an emerging research area with broad applications. This paper reviews this field, categorizing music generation systems from the perspective of modalities. It covers modality representation, multi-modal data alignment, and their utilization to guide music generation. We also discuss current datasets and evaluation methods. Key challenges in this area include effective multi-modal integration, large-scale comprehensive datasets, and systematic evaluation methods. Finally, we provide an outlook on future research directions focusing on multi-modal fusion, alignment, data, and evaluation. 

---
# Conditional Temporal Neural Processes with Covariance Loss 

**Authors**: Boseon Yoo, Jiwoo Lee, Janghoon Ju, Seijun Chung, Soyeon Kim, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00794)  

**Abstract**: We introduce a novel loss function, Covariance Loss, which is conceptually equivalent to conditional neural processes and has a form of regularization so that is applicable to many kinds of neural networks. With the proposed loss, mappings from input variables to target variables are highly affected by dependencies of target variables as well as mean activation and mean dependencies of input and target variables. This nature enables the resulting neural networks to become more robust to noisy observations and recapture missing dependencies from prior information. In order to show the validity of the proposed loss, we conduct extensive sets of experiments on real-world datasets with state-of-the-art models and discuss the benefits and drawbacks of the proposed Covariance Loss. 

---
# Digitally Supported Analysis of Spontaneous Speech (DigiSpon): Benchmarking NLP-Supported Language Sample Analysis of Swiss Children's Speech 

**Authors**: Anja Ryser, Yingqiang Gao, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2504.00780)  

**Abstract**: Language sample analysis (LSA) is a process that complements standardized psychometric tests for diagnosing, for example, developmental language disorder (DLD) in children. However, its labor-intensive nature has limited its use in speech-language pathology practice. We introduce an approach that leverages natural language processing (NLP) methods not based on commercial large language models (LLMs) applied to transcribed speech data from 119 children in the German speaking part of Switzerland with typical and atypical language development. The study aims to identify optimal practices that support speech-language pathologists in diagnosing DLD more efficiently within a human-in-the-loop framework, without relying on potentially unethical implementations that leverage commercial LLMs. Preliminary findings underscore the potential of integrating locally deployed NLP methods into the process of semi-automatic LSA. 

---
# LLMs4SchemaDiscovery: A Human-in-the-Loop Workflow for Scientific Schema Mining with Large Language Models 

**Authors**: Sameer Sadruddin, Jennifer D'Souza, Eleni Poupaki, Alex Watkins, Hamed Babaei Giglou, Anisa Rula, Bora Karasulu, Sören Auer, Adrie Mackus, Erwin Kessels  

**Link**: [PDF](https://arxiv.org/pdf/2504.00752)  

**Abstract**: Extracting structured information from unstructured text is crucial for modeling real-world processes, but traditional schema mining relies on semi-structured data, limiting scalability. This paper introduces schema-miner, a novel tool that combines large language models with human feedback to automate and refine schema extraction. Through an iterative workflow, it organizes properties from text, incorporates expert input, and integrates domain-specific ontologies for semantic depth. Applied to materials science--specifically atomic layer deposition--schema-miner demonstrates that expert-guided LLMs generate semantically rich schemas suitable for diverse real-world applications. 

---
# Advancements in Multimodal Differential Evolution: A Comprehensive Review and Future Perspectives 

**Authors**: Dikshit Chauhan, Shivani, Donghwi Jung, Anupam Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2504.00717)  

**Abstract**: Multi-modal optimization involves identifying multiple global and local optima of a function, offering valuable insights into diverse optimal solutions within the search space. Evolutionary algorithms (EAs) excel at finding multiple solutions in a single run, providing a distinct advantage over classical optimization techniques that often require multiple restarts without guarantee of obtaining diverse solutions. Among these EAs, differential evolution (DE) stands out as a powerful and versatile optimizer for continuous parameter spaces. DE has shown significant success in multi-modal optimization by utilizing its population-based search to promote the formation of multiple stable subpopulations, each targeting different optima. Recent advancements in DE for multi-modal optimization have focused on niching methods, parameter adaptation, hybridization with other algorithms including machine learning, and applications across various domains. Given these developments, it is an opportune moment to present a critical review of the latest literature and identify key future research directions. This paper offers a comprehensive overview of recent DE advancements in multimodal optimization, including methods for handling multiple optima, hybridization with EAs, and machine learning, and highlights a range of real-world applications. Additionally, the paper outlines a set of compelling open problems and future research issues from multiple perspectives 

---
# Science Autonomy using Machine Learning for Astrobiology 

**Authors**: Victoria Da Poian, Bethany Theiling, Eric Lyness, David Burtt, Abigail R. Azari, Joey Pasterski, Luoth Chou, Melissa Trainer, Ryan Danell, Desmond Kaplan, Xiang Li, Lily Clough, Brett McKinney, Lukas Mandrake, Bill Diamond, Caroline Freissinet  

**Link**: [PDF](https://arxiv.org/pdf/2504.00709)  

**Abstract**: In recent decades, artificial intelligence (AI) including machine learning (ML) have become vital for space missions enabling rapid data processing, advanced pattern recognition, and enhanced insight extraction. These tools are especially valuable in astrobiology applications, where models must distinguish biotic patterns from complex abiotic backgrounds. Advancing the integration of autonomy through AI and ML into space missions is a complex challenge, and we believe that by focusing on key areas, we can make significant progress and offer practical recommendations for tackling these obstacles. 

---
# Energy Weighted Learning Progress Guided Interleaved Multi-Task Learning 

**Authors**: Hanne Say, Suzan Ece Ada, Emre Ugur, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2504.00707)  

**Abstract**: Humans can continuously acquire new skills and knowledge by exploiting existing ones for improved learning, without forgetting them. Similarly, 'continual learning' in machine learning aims to learn new information while preserving the previously acquired knowledge. Existing research often overlooks the nature of human learning, where tasks are interleaved due to human choice or environmental constraints. So, almost never do humans master one task before switching to the next. To investigate to what extent human-like learning can benefit the learner, we propose a method that interleaves tasks based on their 'learning progress' and energy consumption. From a machine learning perspective, our approach can be seen as a multi-task learning system that balances learning performance with energy constraints while mimicking ecologically realistic human task learning. To assess the validity of our approach, we consider a robot learning setting in simulation, where the robot learns the effect of its actions in different contexts. The conducted experiments show that our proposed method achieves better performance than sequential task learning and reduces energy consumption for learning the tasks. 

---
# Command A: An Enterprise-Ready Large Language Model 

**Authors**: Team Cohere, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, Neeral Beladia, Walter Beller-Morales, Alexandre Bérard, Andrew Berneshawi, Anna Bialas, Phil Blunsom, Matt Bobkin, Adi Bongale, Sam Braun, Maxime Brunet, Samuel Cahyawijaya, David Cairuz, Jon Ander Campos, Cassie Cao, Kris Cao, Roman Castagné, Julián Cendrero, Leila Chan Currie, Yash Chandak, Diane Chang, Giannis Chatziveroglou, Hongyu Chen, Claire Cheng, Alexis Chevalier, Justin T. Chiu, Eugene Cho, Eugene Choi, Eujeong Choi, Tim Chung, Volkan Cirik, Ana Cismaru, Pierre Clavier, Henry Conklin, Lucas Crawhall-Stein, Devon Crouse, Andres Felipe Cruz-Salinas, Ben Cyrus, Daniel D'souza, Hugo Dalla-Torre, John Dang, William Darling, Omar Darwiche Domingues, Saurabh Dash, Antoine Debugne, Théo Dehaze, Shaan Desai, Joan Devassy, Rishit Dholakia, Kyle Duffy, Ali Edalati, Ace Eldeib, Abdullah Elkady, Sarah Elsharkawy, Irem Ergün, Beyza Ermis, Marzieh Fadaee, Boyu Fan, Lucas Fayoux, Yannis Flet-Berliac, Nick Frosst, Matthias Gallé, Wojciech Galuba, Utsav Garg, Matthieu Geist, Mohammad Gheshlaghi Azar, Seraphina Goldfarb-Tarrant, Tomas Goldsack, Aidan Gomez, Victor Machado Gonzaga, Nithya Govindarajan, Manoj Govindassamy, Nathan Grinsztajn, Nikolas Gritsch, Patrick Gu, Shangmin Guo, Kilian Haefeli, Rod Hajjar, Tim Hawes, Jingyi He, Sebastian Hofstätter, Sungjin Hong, Sara Hooker, Tom Hosking  

**Link**: [PDF](https://arxiv.org/pdf/2504.00698)  

**Abstract**: In this report we describe the development of Command A, a powerful large language model purpose-built to excel at real-world enterprise use cases. Command A is an agent-optimised and multilingual-capable model, with support for 23 languages of global business, and a novel hybrid architecture balancing efficiency with top of the range performance. It offers best-in-class Retrieval Augmented Generation (RAG) capabilities with grounding and tool use to automate sophisticated business processes. These abilities are achieved through a decentralised training approach, including self-refinement algorithms and model merging techniques. We also include results for Command R7B which shares capability and architectural similarities to Command A. Weights for both models have been released for research purposes. This technical report details our original training pipeline and presents an extensive evaluation of our models across a suite of enterprise-relevant tasks and public benchmarks, demonstrating excellent performance and efficiency. 

---
# The HCI GenAI CO2ST Calculator: A Tool for Calculating the Carbon Footprint of Generative AI Use in Human-Computer Interaction Research 

**Authors**: Nanna Inie, Jeanette Falk, Raghavendra Selvan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00692)  

**Abstract**: Increased usage of generative AI (GenAI) in Human-Computer Interaction (HCI) research induces a climate impact from carbon emissions due to energy consumption of the hardware used to develop and run GenAI models and systems. The exact energy usage and and subsequent carbon emissions are difficult to estimate in HCI research because HCI researchers most often use cloud-based services where the hardware and its energy consumption are hidden from plain view. The HCI GenAI CO2ST Calculator is a tool designed specifically for the HCI research pipeline, to help researchers estimate the energy consumption and carbon footprint of using generative AI in their research, either a priori (allowing for mitigation strategies or experimental redesign) or post hoc (allowing for transparent documentation of carbon footprint in written reports of the research). 

---
# DynMoLE: Boosting Mixture of LoRA Experts Fine-Tuning with a Hybrid Routing Mechanism 

**Authors**: Dengchun Li, Naizheng Wang, Zihao Zhang, Haoyang Yin, Lei Duan, Meng Xiao, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00661)  

**Abstract**: Instruction-based fine-tuning of large language models (LLMs) has achieved remarkable success in various natural language processing (NLP) tasks. Parameter-efficient fine-tuning (PEFT) methods, such as Mixture of LoRA Experts (MoLE), combine the efficiency of Low-Rank Adaptation (LoRA) with the versatility of Mixture of Experts (MoE) models, demonstrating significant potential for handling multiple downstream tasks. However, the existing routing mechanisms for MoLE often involve a trade-off between computational efficiency and predictive accuracy, and they fail to fully address the diverse expert selection demands across different transformer layers. In this work, we propose DynMoLE, a hybrid routing strategy that dynamically adjusts expert selection based on the Tsallis entropy of the router's probability distribution. This approach mitigates router uncertainty, enhances stability, and promotes more equitable expert participation, leading to faster convergence and improved model performance. Additionally, we introduce an auxiliary loss based on Tsallis entropy to further guide the model toward convergence with reduced uncertainty, thereby improving training stability and performance. Our extensive experiments on commonsense reasoning benchmarks demonstrate that DynMoLE achieves substantial performance improvements, outperforming LoRA by 9.6% and surpassing the state-of-the-art MoLE method, MoLA, by 2.3%. We also conduct a comprehensive ablation study to evaluate the contributions of DynMoLE's key components. 

---
# Towards Adaptive AI Governance: Comparative Insights from the U.S., EU, and Asia 

**Authors**: Vikram Kulothungan, Deepti Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.00652)  

**Abstract**: Artificial intelligence (AI) trends vary significantly across global regions, shaping the trajectory of innovation, regulation, and societal impact. This variation influences how different regions approach AI development, balancing technological progress with ethical and regulatory considerations. This study conducts a comparative analysis of AI trends in the United States (US), the European Union (EU), and Asia, focusing on three key dimensions: generative AI, ethical oversight, and industrial applications. The US prioritizes market-driven innovation with minimal regulatory constraints, the EU enforces a precautionary risk-based framework emphasizing ethical safeguards, and Asia employs state-guided AI strategies that balance rapid deployment with regulatory oversight. Although these approaches reflect different economic models and policy priorities, their divergence poses challenges to international collaboration, regulatory harmonization, and the development of global AI standards. To address these challenges, this paper synthesizes regional strengths to propose an adaptive AI governance framework that integrates risk-tiered oversight, innovation accelerators, and strategic alignment mechanisms. By bridging governance gaps, this study offers actionable insights for fostering responsible AI development while ensuring a balance between technological progress, ethical imperatives, and regulatory coherence. 

---
# Impact of Data Duplication on Deep Neural Network-Based Image Classifiers: Robust vs. Standard Models 

**Authors**: Alireza Aghabagherloo, Aydin Abadi, Sumanta Sarkar, Vishnu Asutosh Dasu, Bart Preneel  

**Link**: [PDF](https://arxiv.org/pdf/2504.00638)  

**Abstract**: The accuracy and robustness of machine learning models against adversarial attacks are significantly influenced by factors such as training data quality, model architecture, the training process, and the deployment environment. In recent years, duplicated data in training sets, especially in language models, has attracted considerable attention. It has been shown that deduplication enhances both training performance and model accuracy in language models. While the importance of data quality in training image classifier Deep Neural Networks (DNNs) is widely recognized, the impact of duplicated images in the training set on model generalization and performance has received little attention.
In this paper, we address this gap and provide a comprehensive study on the effect of duplicates in image classification. Our analysis indicates that the presence of duplicated images in the training set not only negatively affects the efficiency of model training but also may result in lower accuracy of the image classifier. This negative impact of duplication on accuracy is particularly evident when duplicated data is non-uniform across classes or when duplication, whether uniform or non-uniform, occurs in the training set of an adversarially trained model. Even when duplicated samples are selected in a uniform way, increasing the amount of duplication does not lead to a significant improvement in accuracy. 

---
# CNOT-Optimal Clifford Synthesis as SAT 

**Authors**: Irfansha Shaik, Jaco van de Pol  

**Link**: [PDF](https://arxiv.org/pdf/2504.00634)  

**Abstract**: Clifford circuit optimization is an important step in the quantum compilation pipeline. Major compilers employ heuristic approaches. While they are fast, their results are often suboptimal. Minimization of noisy gates, like 2-qubit CNOT gates, is crucial for practical computing. Exact approaches have been proposed to fill the gap left by heuristic approaches. Among these are SAT based approaches that optimize gate count or depth, but they suffer from scalability issues. Further, they do not guarantee optimality on more important metrics like CNOT count or CNOT depth. A recent work proposed an exhaustive search only on Clifford circuits in a certain normal form to guarantee CNOT count optimality. But an exhaustive approach cannot scale beyond 6 qubits.
In this paper, we incorporate search restricted to Clifford normal forms in a SAT encoding to guarantee CNOT count optimality. By allowing parallel plans, we propose a second SAT encoding that optimizes CNOT depth. By taking advantage of flexibility in SAT based approaches, we also handle connectivity restrictions in hardware platforms, and allow for qubit relabeling. We have implemented the above encodings and variations in our open source tool Q-Synth.
In experiments, our encodings significantly outperform existing SAT approaches on random Clifford circuits. We consider practical VQE and Feynman benchmarks to compare with TKET and Qiskit compilers. In all-to-all connectivity, we observe reductions up to 32.1% in CNOT count and 48.1% in CNOT depth. Overall, we observe better results than TKET in the CNOT count and depth. We also experiment with connectivity restrictions of major quantum platforms. Compared to Qiskit, we observe up to 30.3% CNOT count and 35.9% CNOT depth further reduction. 

---
# Feature Subset Weighting for Distance-based Supervised Learning through Choquet Integration 

**Authors**: Adnan Theerens, Yvan Saeys, Chris Cornelis  

**Link**: [PDF](https://arxiv.org/pdf/2504.00624)  

**Abstract**: This paper introduces feature subset weighting using monotone measures for distance-based supervised learning. The Choquet integral is used to define a distance metric that incorporates these weights. This integration enables the proposed distances to effectively capture non-linear relationships and account for interactions both between conditional and decision attributes and among conditional attributes themselves, resulting in a more flexible distance measure. In particular, we show how this approach ensures that the distances remain unaffected by the addition of duplicate and strongly correlated features. Another key point of this approach is that it makes feature subset weighting computationally feasible, since only $m$ feature subset weights should be calculated each time instead of calculating all feature subset weights ($2^m$), where $m$ is the number of attributes. Next, we also examine how the use of the Choquet integral for measuring similarity leads to a non-equivalent definition of distance. The relationship between distance and similarity is further explored through dual measures. Additionally, symmetric Choquet distances and similarities are proposed, preserving the classical symmetry between similarity and distance. Finally, we introduce a concrete feature subset weighting distance, evaluate its performance in a $k$-nearest neighbors (KNN) classification setting, and compare it against Mahalanobis distances and weighted distance methods. 

---
# PLM4NDV: Minimizing Data Access for Number of Distinct Values Estimation with Pre-trained Language Models 

**Authors**: Xianghong Xu, Xiao He, Tieying Zhang, Lei Zhang, Rui Shi, Jianjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00608)  

**Abstract**: Number of Distinct Values (NDV) estimation of a multiset/column is a basis for many data management tasks, especially within databases. Despite decades of research, most existing methods require either a significant amount of samples through uniform random sampling or access to the entire column to produce estimates, leading to substantial data access costs and potentially ineffective estimations in scenarios with limited data access. In this paper, we propose leveraging semantic information, i.e., schema, to address these challenges. The schema contains rich semantic information that can benefit the NDV estimation. To this end, we propose PLM4NDV, a learned method incorporating Pre-trained Language Models (PLMs) to extract semantic schema information for NDV estimation. Specifically, PLM4NDV leverages the semantics of the target column and the corresponding table to gain a comprehensive understanding of the column's meaning. By using the semantics, PLM4NDV reduces data access costs, provides accurate NDV estimation, and can even operate effectively without any data access. Extensive experiments on a large-scale real-world dataset demonstrate the superiority of PLM4NDV over baseline methods. Our code is available at this https URL. 

---
# Data Cleansing for GANs 

**Authors**: Naoyuki Terashita, Hiroki Ohashi, Satoshi Hara  

**Link**: [PDF](https://arxiv.org/pdf/2504.00603)  

**Abstract**: As the application of generative adversarial networks (GANs) expands, it becomes increasingly critical to develop a unified approach that improves performance across various generative tasks. One effective strategy that applies to any machine learning task is identifying harmful instances, whose removal improves the performance. While previous studies have successfully estimated these harmful training instances in supervised settings, their approaches are not easily applicable to GANs. The challenge lies in two requirements of the previous approaches that do not apply to GANs. First, previous approaches require that the absence of a training instance directly affects the parameters. However, in the training for GANs, the instances do not directly affect the generator's parameters since they are only fed into the discriminator. Second, previous approaches assume that the change in loss directly quantifies the harmfulness of the instance to a model's performance, while common types of GAN losses do not always reflect the generative performance. To overcome the first challenge, we propose influence estimation methods that use the Jacobian of the generator's gradient with respect to the discriminator's parameters (and vice versa). Such a Jacobian represents the indirect effect between two models: how removing an instance from the discriminator's training changes the generator's parameters. Second, we propose an instance evaluation scheme that measures the harmfulness of each training instance based on how a GAN evaluation metric (e.g., Inception score) is expected to change by the instance's removal. Furthermore, we demonstrate that removing the identified harmful instances significantly improves the generative performance on various GAN evaluation metrics. 

---
# On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation 

**Authors**: Jirui Qi, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00597)  

**Abstract**: Retrieval-augmented generation (RAG) with large language models (LLMs) has demonstrated strong performance in multilingual question-answering (QA) tasks by leveraging relevant passages retrieved from corpora. In multilingual RAG (mRAG), the retrieved passages can be written in languages other than that of the query entered by the user, making it challenging for LLMs to effectively utilize the provided information. Recent research suggests that retrieving passages from multilingual corpora can improve RAG performance, particularly for low-resource languages. However, the extent to which LLMs can leverage different kinds of multilingual contexts to generate accurate answers, *independently from retrieval quality*, remains understudied. In this paper, we conduct an extensive assessment of LLMs' ability to (i) make consistent use of a relevant passage regardless of its language, (ii) respond in the expected language, and (iii) focus on the relevant passage even when multiple `distracting' passages in different languages are provided in the context. Our experiments with four LLMs across three QA datasets covering a total of 48 languages reveal a surprising ability of LLMs to extract the relevant information from out-language passages, but a much weaker ability to formulate a full answer in the correct language. Our analysis, based on both accuracy and feature attribution techniques, further shows that distracting passages negatively impact answer quality regardless of their language. However, distractors in the query language exert a slightly stronger influence. Taken together, our findings deepen the understanding of how LLMs utilize context in mRAG systems, providing directions for future improvements. 

---
# Enhancing Negation Awareness in Universal Text Embeddings: A Data-efficient and Computational-efficient Approach 

**Authors**: Hongliu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00584)  

**Abstract**: Negation plays an important role in various natural language processing tasks such as Natural Language Inference and Sentiment Analysis tasks. Numerous prior studies have found that contextual text embedding models such as BERT, ELMO, RoBERTa or XLNet face challenges in accurately understanding negation. Recent advancements in universal text embeddings have demonstrated superior performance over contextual text embeddings in various tasks. However, due to the bias in popular evaluation benchmarks, the negation awareness capacity of these models remains unclear. To bridge the gap in existing literature, an in-depth analysis is initiated in this work to study the negation awareness of cutting-edge universal text embedding models. Our findings reveal a significant lack of negation awareness in these models, often interpreting negated text pairs as semantically similar. To efficiently deal with the conflict that different tasks need different trade-offs between topic and negation information among other semantic information, a data-efficient and computational-efficient embedding re-weighting method is proposed without modifying the parameters of text embedding models. The proposed solution is able to improve text embedding models' negation awareness significantly on both simple negation understanding task and complex negation understanding task. Furthermore, the proposed solution can also significantly improve the negation awareness of Large Language Model based task-specific high dimensional universal text embeddings. 

---
# High-Quality Pseudo-Label Generation Based on Visual Prompt Assisted Cloud Model Update 

**Authors**: Xinrun Xu, Qiuhong Zhang, Jianwen Yang, Zhanbiao Lian, Jin Yan, Zhiming Ding, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00526)  

**Abstract**: Generating high-quality pseudo-labels on the cloud is crucial for cloud-edge object detection, especially in dynamic traffic monitoring where data distributions evolve. Existing methods often assume reliable cloud models, neglecting potential errors or struggling with complex distribution shifts. This paper proposes Cloud-Adaptive High-Quality Pseudo-label generation (CA-HQP), addressing these limitations by incorporating a learnable Visual Prompt Generator (VPG) and dual feature alignment into cloud model updates. The VPG enables parameter-efficient adaptation by injecting visual prompts, enhancing flexibility without extensive fine-tuning. CA-HQP mitigates domain discrepancies via two feature alignment techniques: global Domain Query Feature Alignment (DQFA) capturing scene-level shifts, and fine-grained Temporal Instance-Aware Feature Embedding Alignment (TIAFA) addressing instance variations. Experiments on the Bellevue traffic dataset demonstrate that CA-HQP significantly improves pseudo-label quality compared to existing methods, leading to notable performance gains for the edge model and showcasing CA-HQP's adaptation effectiveness. Ablation studies validate each component (DQFA, TIAFA, VPG) and the synergistic effect of combined alignment strategies, highlighting the importance of adaptive cloud updates and domain adaptation for robust object detection in evolving scenarios. CA-HQP provides a promising solution for enhancing cloud-edge object detection systems in real-world applications. 

---
# Automated detection of atomicity violations in large-scale systems 

**Authors**: Hang He, Yixing Luo, Chengcheng Wan, Ting Su, Haiying Sun, Geguang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00521)  

**Abstract**: Atomicity violations in interrupt-driven programs pose a significant threat to software safety in critical systems. These violations occur when the execution sequence of operations on shared resources is disrupted by asynchronous interrupts. Detecting atomicity violations is challenging due to the vast program state space, application-level code dependencies, and complex domain-specific knowledge. We propose Clover, a hybrid framework that integrates static analysis with large language model (LLM) agents to detect atomicity violations in real-world programs. Clover first performs static analysis to extract critical code snippets and operation information. It then initiates a multi-agent process, where the expert agent leverages domain-specific knowledge to detect atomicity violations, which are subsequently validated by the judge agent. Evaluations on RaceBench 2.1, SV-COMP, and RWIP demonstrate that Clover achieves a precision/recall of 92.3%/86.6%, outperforming existing approaches by 27.4-118.2% on F1-score. 

---
# Training Frozen Feature Pyramid DINOv2 for Eyelid Measurements with Infinite Encoding and Orthogonal Regularization 

**Authors**: Chun-Hung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00515)  

**Abstract**: Accurate measurement of eyelid parameters such as Margin Reflex Distances (MRD1, MRD2) and Levator Function (LF) is critical in oculoplastic diagnostics but remains limited by manual, inconsistent methods. This study evaluates deep learning models: SE-ResNet, EfficientNet, and the vision transformer-based DINOv2 for automating these measurements using smartphone-acquired images. We assess performance across frozen and fine-tuned settings, using MSE, MAE, and R2 metrics. DINOv2, pretrained through self-supervised learning, demonstrates superior scalability and robustness, especially under frozen conditions ideal for mobile deployment. Lightweight regressors such as MLP and Deep Ensemble offer high precision with minimal computational overhead. To address class imbalance and improve generalization, we integrate focal loss, orthogonal regularization, and binary encoding strategies. Our results show that DINOv2 combined with these enhancements delivers consistent, accurate predictions across all tasks, making it a strong candidate for real-world, mobile-friendly clinical applications. This work highlights the potential of foundation models in advancing AI-powered ophthalmic care. 

---
# Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving 

**Authors**: Jianing Huang, Kaixuan Zhang, Youjia Wu, Ze Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00510)  

**Abstract**: Neural operators have become increasingly popular in solving \textit{partial differential equations} (PDEs) due to their superior capability to capture intricate mappings between function spaces over complex domains. However, the data-hungry nature of operator learning inevitably poses a bottleneck for their widespread applications. At the core of the challenge lies the absence of transferability of neural operators to new geometries. To tackle this issue, we propose operator learning with domain decomposition, a local-to-global framework to solve PDEs on arbitrary geometries. Under this framework, we devise an iterative scheme \textit{Schwarz Neural Inference} (SNI). This scheme allows for partitioning of the problem domain into smaller subdomains, on which local problems can be solved with neural operators, and stitching local solutions to construct a global solution. Additionally, we provide a theoretical analysis of the convergence rate and error bound. We conduct extensive experiments on several representative PDEs with diverse boundary conditions and achieve remarkable geometry generalization compared to alternative methods. These analysis and experiments demonstrate the proposed framework's potential in addressing challenges related to geometry generalization and data efficiency. 

---
# Enhancing stroke disease classification through machine learning models via a novel voting system by feature selection techniques 

**Authors**: Mahade Hasan, Farhana Yasmin, Md. Mehedi Hassan, Xue Yu, Soniya Yeasmin, Herat Joshi, Sheikh Mohammed Shariful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2504.00485)  

**Abstract**: Heart disease remains a leading cause of mortality and morbidity worldwide, necessitating the development of accurate and reliable predictive models to facilitate early detection and intervention. While state of the art work has focused on various machine learning approaches for predicting heart disease, but they could not able to achieve remarkable accuracy. In response to this need, we applied nine machine learning algorithms XGBoost, logistic regression, decision tree, random forest, k-nearest neighbors (KNN), support vector machine (SVM), gaussian naïve bayes (NB gaussian), adaptive boosting, and linear regression to predict heart disease based on a range of physiological indicators. Our approach involved feature selection techniques to identify the most relevant predictors, aimed at refining the models to enhance both performance and interpretability. The models were trained, incorporating processes such as grid search hyperparameter tuning, and cross-validation to minimize overfitting. Additionally, we have developed a novel voting system with feature selection techniques to advance heart disease classification. Furthermore, we have evaluated the models using key performance metrics including accuracy, precision, recall, F1-score, and the area under the receiver operating characteristic curve (ROC AUC). Among the models, XGBoost demonstrated exceptional performance, achieving 99% accuracy, precision, F1-Score, 98% recall, and 100% ROC AUC. This study offers a promising approach to early heart disease diagnosis and preventive healthcare. 

---
# Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning 

**Authors**: Ruoxi Xu, Yunjie Ji, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Ben He, Yingfei Sun, Xiangang Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.00472)  

**Abstract**: Although large language models (LLMs) excel in knowledge recall and reasoning, their static nature leads to outdated information as the real world evolves or when adapting to domain-specific knowledge, highlighting the need for effective knowledge injection. However, current research on knowledge injection remains superficial, mainly focusing on knowledge memorization and retrieval. This paper proposes a four-tier knowledge injection framework that systematically defines the levels of knowledge injection: memorization, retrieval, reasoning, and association. Based on this framework, we introduce DeepKnowledge, a synthetic experimental testbed designed for fine-grained evaluation of the depth of knowledge injection across three knowledge types (novel, incremental, and updated). We then explore various knowledge injection scenarios and evaluate the depth of knowledge injection for each scenario on the benchmark. Experimental results reveal key factors to reach each level of knowledge injection for LLMs and establish a mapping between the levels of knowledge injection and the corresponding suitable injection methods, aiming to provide a comprehensive approach for efficient knowledge injection across various levels. 

---
# Learning-Based Approximate Nonlinear Model Predictive Control Motion Cueing 

**Authors**: Camilo Gonzalez Arango, Houshyar Asadi, Mohammad Reza Chalak Qazani, Chee Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00469)  

**Abstract**: Motion Cueing Algorithms (MCAs) encode the movement of simulated vehicles into movement that can be reproduced with a motion simulator to provide a realistic driving experience within the capabilities of the machine. This paper introduces a novel learning-based MCA for serial robot-based motion simulators. Building on the differentiable predictive control framework, the proposed method merges the advantages of Nonlinear Model Predictive Control (NMPC) - notably nonlinear constraint handling and accurate kinematic modeling - with the computational efficiency of machine learning. By shifting the computational burden to offline training, the new algorithm enables real-time operation at high control rates, thus overcoming the key challenge associated with NMPC-based motion cueing. The proposed MCA incorporates a nonlinear joint-space plant model and a policy network trained to mimic NMPC behavior while accounting for joint acceleration, velocity, and position limits. Simulation experiments across multiple motion cueing scenarios showed that the proposed algorithm performed on par with a state-of-the-art NMPC-based alternative in terms of motion cueing quality as quantified by the RMSE and correlation coefficient with respect to reference signals. However, the proposed algorithm was on average 400 times faster than the NMPC baseline. In addition, the algorithm successfully generalized to unseen operating conditions, including motion cueing scenarios on a different vehicle and real-time physics-based simulations. 

---
# MetaLoRA: Tensor-Enhanced Adaptive Low-Rank Fine-tuning 

**Authors**: Maolin Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00460)  

**Abstract**: There has been a significant increase in the deployment of neural network models, presenting substantial challenges in model adaptation and fine-tuning. Efficient adaptation is crucial in maintaining model performance across diverse tasks and domains. While Low-Rank Adaptation (LoRA) has emerged as a promising parameter-efficient fine-tuning method, its fixed parameter nature limits its ability to handle dynamic task requirements effectively. Adapting models to new tasks can be challenging due to the need for extensive fine-tuning. Current LoRA variants primarily focus on general parameter reduction while overlooking the importance of dynamic parameter adjustment and meta-learning capabilities. Moreover, existing approaches mainly address static adaptations, neglecting the potential benefits of task-aware parameter generation in handling diverse task distributions. To address these limitations, this Ph.D. research proposes a LoRA generation approach to model task relationships and introduces MetaLoRA, a novel parameter-efficient adaptation framework incorporating meta-learning principles. This work develops a comprehensive architecture that integrates meta-parameter generation with adaptive low-rank decomposition, enabling efficient handling of both task-specific and task-agnostic features. MetaLoRA accurately captures task patterns by incorporating meta-learning mechanisms and dynamic parameter adjustment strategies. To our knowledge, this research represents the first attempt to provide a meta-learning enhanced LoRA variant, offering improved adaptation capability while maintaining computational efficiency in model fine-tuning. 

---
# Distilling Multi-view Diffusion Models into 3D Generators 

**Authors**: Hao Qin, Luyuan Chen, Ming Kong, Mengxu Lu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00457)  

**Abstract**: We introduce DD3G, a formulation that Distills a multi-view Diffusion model (MV-DM) into a 3D Generator using gaussian splatting. DD3G compresses and integrates extensive visual and spatial geometric knowledge from the MV-DM by simulating its ordinary differential equation (ODE) trajectory, ensuring the distilled generator generalizes better than those trained solely on 3D data. Unlike previous amortized optimization approaches, we align the MV-DM and 3D generator representation spaces to transfer the teacher's probabilistic flow to the student, thus avoiding inconsistencies in optimization objectives caused by probabilistic sampling. The introduction of probabilistic flow and the coupling of various attributes in 3D Gaussians introduce challenges in the generation process. To tackle this, we propose PEPD, a generator consisting of Pattern Extraction and Progressive Decoding phases, which enables efficient fusion of probabilistic flow and converts a single image into 3D Gaussians within 0.06 seconds. Furthermore, to reduce knowledge loss and overcome sparse-view supervision, we design a joint optimization objective that ensures the quality of generated samples through explicit supervision and implicit verification. Leveraging existing 2D generation models, we compile 120k high-quality RGBA images for distillation. Experiments on synthetic and public datasets demonstrate the effectiveness of our method. Our project is available at: this https URL 

---
# No Free Lunch with Guardrails 

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00441)  

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.
Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and our own custom-built guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability. 

---
# Suite-IN++: A FlexiWear BodyNet Integrating Global and Local Motion Features from Apple Suite for Robust Inertial Navigation 

**Authors**: Lan Sun, Songpengcheng Xia, Jiarui Yang, Ling Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00438)  

**Abstract**: The proliferation of wearable technology has established multi-device ecosystems comprising smartphones, smartwatches, and headphones as critical enablers for ubiquitous pedestrian localization. However, traditional pedestrian dead reckoning (PDR) struggles with diverse motion modes, while data-driven methods, despite improving accuracy, often lack robustness due to their reliance on a single-device setup. Therefore, a promising solution is to fully leverage existing wearable devices to form a flexiwear bodynet for robust and accurate pedestrian localization. This paper presents Suite-IN++, a deep learning framework for flexiwear bodynet-based pedestrian localization. Suite-IN++ integrates motion data from wearable devices on different body parts, using contrastive learning to separate global and local motion features. It fuses global features based on the data reliability of each device to capture overall motion trends and employs an attention mechanism to uncover cross-device correlations in local features, extracting motion details helpful for accurate localization. To evaluate our method, we construct a real-life flexiwear bodynet dataset, incorporating Apple Suite (iPhone, Apple Watch, and AirPods) across diverse walking modes and device configurations. Experimental results demonstrate that Suite-IN++ achieves superior localization accuracy and robustness, significantly outperforming state-of-the-art models in real-life pedestrian tracking scenarios. 

---
# LLM-Assisted Proactive Threat Intelligence for Automated Reasoning 

**Authors**: Shuva Paul, Farhad Alemi, Richard Macwan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00428)  

**Abstract**: Successful defense against dynamically evolving cyber threats requires advanced and sophisticated techniques. This research presents a novel approach to enhance real-time cybersecurity threat detection and response by integrating large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems with continuous threat intelligence feeds. Leveraging recent advancements in LLMs, specifically GPT-4o, and the innovative application of RAG techniques, our approach addresses the limitations of traditional static threat analysis by incorporating dynamic, real-time data sources. We leveraged RAG to get the latest information in real-time for threat intelligence, which is not possible in the existing GPT-4o model. We employ the Patrowl framework to automate the retrieval of diverse cybersecurity threat intelligence feeds, including Common Vulnerabilities and Exposures (CVE), Common Weakness Enumeration (CWE), Exploit Prediction Scoring System (EPSS), and Known Exploited Vulnerabilities (KEV) databases, and integrate these with the all-mpnet-base-v2 model for high-dimensional vector embeddings, stored and queried in Milvus. We demonstrate our system's efficacy through a series of case studies, revealing significant improvements in addressing recently disclosed vulnerabilities, KEVs, and high-EPSS-score CVEs compared to the baseline GPT-4o. This work not only advances the role of LLMs in cybersecurity but also establishes a robust foundation for the development of automated intelligent cyberthreat information management systems, addressing crucial gaps in current cybersecurity practices. 

---
# Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents 

**Authors**: Gavin Greif, Niclas Griesshaber, Robin Greif  

**Link**: [PDF](https://arxiv.org/pdf/2504.00414)  

**Abstract**: We explore how multimodal Large Language Models (mLLMs) can help researchers transcribe historical documents, extract relevant historical information, and construct datasets from historical sources. Specifically, we investigate the capabilities of mLLMs in performing (1) Optical Character Recognition (OCR), (2) OCR Post-Correction, and (3) Named Entity Recognition (NER) tasks on a set of city directories published in German between 1754 and 1870. First, we benchmark the off-the-shelf transcription accuracy of both mLLMs and conventional OCR models. We find that the best-performing mLLM model significantly outperforms conventional state-of-the-art OCR models and other frontier mLLMs. Second, we are the first to introduce multimodal post-correction of OCR output using mLLMs. We find that this novel approach leads to a drastic improvement in transcription accuracy and consistently produces highly accurate transcriptions (<1% CER), without any image pre-processing or model fine-tuning. Third, we demonstrate that mLLMs can efficiently recognize entities in transcriptions of historical documents and parse them into structured dataset formats. Our findings provide early evidence for the long-term potential of mLLMs to introduce a paradigm shift in the approaches to historical data collection and document transcription. 

---
# Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding 

**Authors**: Mohanakrishnan Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00409)  

**Abstract**: Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding. 

---
# From Intuition to Understanding: Using AI Peers to Overcome Physics Misconceptions 

**Authors**: Ruben Weijers, Denton Wu, Hannah Betts, Tamara Jacod, Yuxiang Guan, Vidya Sujaya, Kushal Dev, Toshali Goel, William Delooze, Reihaneh Rabbany, Ying Wu, Jean-François Godbout, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2504.00408)  

**Abstract**: Generative AI has the potential to transform personalization and accessibility of education. However, it raises serious concerns about accuracy and helping students become independent critical thinkers. In this study, we designed a helpful AI "Peer" to help students correct fundamental physics misconceptions related to Newtonian mechanic concepts. In contrast to approaches that seek near-perfect accuracy to create an authoritative AI tutor or teacher, we directly inform students that this AI can answer up to 40% of questions incorrectly. In a randomized controlled trial with 165 students, those who engaged in targeted dialogue with the AI Peer achieved post-test scores that were, on average, 10.5 percentage points higher - with over 20 percentage points higher normalized gain - than a control group that discussed physics history. Qualitative feedback indicated that 91% of the treatment group's AI interactions were rated as helpful. Furthermore, by comparing student performance on pre- and post-test questions about the same concept, along with experts' annotations of the AI interactions, we find initial evidence suggesting the improvement in performance does not depend on the correctness of the AI. With further research, the AI Peer paradigm described here could open new possibilities for how we learn, adapt to, and grow with AI. 

---
# VerifiAgent: a Unified Verification Agent in Language Model Reasoning 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00406)  

**Abstract**: Large language models demonstrate remarkable reasoning capabilities but often produce unreliable or incorrect responses. Existing verification methods are typically model-specific or domain-restricted, requiring significant computational resources and lacking scalability across diverse reasoning tasks. To address these limitations, we propose VerifiAgent, a unified verification agent that integrates two levels of verification: meta-verification, which assesses completeness and consistency in model responses, and tool-based adaptive verification, where VerifiAgent autonomously selects appropriate verification tools based on the reasoning type, including mathematical, logical, or commonsense reasoning. This adaptive approach ensures both efficiency and robustness across different verification scenarios. Experimental results show that VerifiAgent outperforms baseline verification methods (e.g., deductive verifier, backward verifier) among all reasoning tasks. Additionally, it can further enhance reasoning accuracy by leveraging feedback from verification results. VerifiAgent can also be effectively applied to inference scaling, achieving better results with fewer generated samples and costs compared to existing process reward models in the mathematical reasoning domain. Code is available at this https URL 

---
# Beyond Wide-Angle Images: Unsupervised Video Portrait Correction via Spatiotemporal Diffusion Adaptation 

**Authors**: Wenbo Nie, Lang Nie, Chunyu Lin, Jingwen Chen, Ke Xing, Jiyuan Wang, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00401)  

**Abstract**: Wide-angle cameras, despite their popularity for content creation, suffer from distortion-induced facial stretching-especially at the edge of the lens-which degrades visual appeal. To address this issue, we propose an image portrait correction framework using diffusion models named ImagePD. It integrates the long-range awareness of transformer and multi-step denoising of diffusion models into a unified framework, achieving global structural robustness and local detail refinement. Besides, considering the high cost of obtaining video labels, we then repurpose ImagePD for unlabeled wide-angle videos (termed VideoPD), by spatiotemporal diffusion adaption with spatial consistency and temporal smoothness constraints. For the former, we encourage the denoised image to approximate pseudo labels following the wide-angle distortion distribution pattern, while for the latter, we derive rectification trajectories with backward optical flows and smooth them. Compared with ImagePD, VideoPD maintains high-quality facial corrections in space and mitigates the potential temporal shakes sequentially. Finally, to establish an evaluation benchmark and train the framework, we establish a video portrait dataset with a large diversity in people number, lighting conditions, and background. Experiments demonstrate that the proposed methods outperform existing solutions quantitatively and qualitatively, contributing to high-fidelity wide-angle videos with stable and natural portraits. The codes and dataset will be available. 

---
# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR) 

**Authors**: Mahak Agarwal, Divyam Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2504.00374)  

**Abstract**: In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation. 

---
# Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation 

**Authors**: Ting Liu, Siyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00356)  

**Abstract**: Recent advances in zero-shot referring image segmentation (RIS), driven by models such as the Segment Anything Model (SAM) and CLIP, have made substantial progress in aligning visual and textual information. Despite these successes, the extraction of precise and high-quality mask region representations remains a critical challenge, limiting the full potential of RIS tasks. In this paper, we introduce a training-free, hybrid global-local feature extraction approach that integrates detailed mask-specific features with contextual information from the surrounding area, enhancing mask region representation. To further strengthen alignment between mask regions and referring expressions, we propose a spatial guidance augmentation strategy that improves spatial coherence, which is essential for accurately localizing described areas. By incorporating multiple spatial cues, this approach facilitates more robust and precise referring segmentation. Extensive experiments on standard RIS benchmarks demonstrate that our method significantly outperforms existing zero-shot RIS models, achieving substantial performance gains. We believe our approach advances RIS tasks and establishes a versatile framework for region-text alignment, offering broader implications for cross-modal understanding and interaction. Code is available at this https URL . 

---
# Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments 

**Authors**: Joshua Moore, Aly Sabri Abdalla, Prabesh Khanal, Vuk Marojevic  

**Link**: [PDF](https://arxiv.org/pdf/2504.00341)  

**Abstract**: The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy. 

---
# VNJPTranslate: A comprehensive pipeline for Vietnamese-Japanese translation 

**Authors**: Hoang Hai Phan, Nguyen Duc Minh Vu, Nam Dang Phuong  

**Link**: [PDF](https://arxiv.org/pdf/2504.00339)  

**Abstract**: Neural Machine Translation (NMT) driven by Transformer architectures has advanced significantly, yet faces challenges with low-resource language pairs like Vietnamese-Japanese (Vi-Ja). Issues include sparse parallel data and handling linguistic/cultural nuances. Recent progress in Large Language Models (LLMs) with strong reasoning, often refined via Reinforcement Learning (RL), enables high-quality synthetic data generation. We introduce VNJPTranslate, a pipeline designed to systematically address the Vi-Ja translation task. It features a targeted data augmentation strategy using advanced LLMs with Chain-of-Thought prompting for challenging segments identified via corpus analysis. Subsequently, we employ efficient fine-tuning techniques (Unsloth with QLoRA) on a capable, low-parameter autoregressive model (specifically, a fine-tuned version of the 1.8B parameter Sailor model, which is based on the Qwen architecture) to create a practical and high-performing translation system. This integrated approach aims to improve Vi-Ja translation quality significantly over existing baselines. 

---
# Agentic Multimodal AI for Hyperpersonalized B2B and B2C Advertising in Competitive Markets: An AI-Driven Competitive Advertising Framework 

**Authors**: Sakhinana Sagar Srinivas, Akash Das, Shivam Gupta, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2504.00338)  

**Abstract**: The growing use of foundation models (FMs) in real-world applications demands adaptive, reliable, and efficient strategies for dynamic markets. In the chemical industry, AI-discovered materials drive innovation, but commercial success hinges on market adoption, requiring FM-driven advertising frameworks that operate in-the-wild. We present a multilingual, multimodal AI framework for autonomous, hyper-personalized advertising in B2B and B2C markets. By integrating retrieval-augmented generation (RAG), multimodal reasoning, and adaptive persona-based targeting, our system generates culturally relevant, market-aware ads tailored to shifting consumer behaviors and competition. Validation combines real-world product experiments with a Simulated Humanistic Colony of Agents to model consumer personas, optimize strategies at scale, and ensure privacy compliance. Synthetic experiments mirror real-world scenarios, enabling cost-effective testing of ad strategies without risky A/B tests. Combining structured retrieval-augmented reasoning with in-context learning (ICL), the framework boosts engagement, prevents market cannibalization, and maximizes ROAS. This work bridges AI-driven innovation and market adoption, advancing multimodal FM deployment for high-stakes decision-making in commercial marketing. 

---
# SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings 

**Authors**: Kerui Wu, Ziyue Zhao, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2504.00336)  

**Abstract**: Epilepsy is a common neurological disorder that affects around 65 million people worldwide. Detecting seizures quickly and accurately is vital, given the prevalence and severity of the associated complications. Recently, deep learning-based automated seizure detection methods have emerged as solutions; however, most existing methods require extensive post-processing and do not effectively handle the crucial long-range patterns in EEG data. In this work, we propose SeizureTransformer, a simple model comprised of (i) a deep encoder comprising 1D convolutions (ii) a residual CNN stack and a transformer encoder to embed previous output into high-level representation with contextual information, and (iii) streamlined decoder which converts these features into a sequence of probabilities, directly indicating the presence or absence of seizures at every time step. Extensive experiments on public and private EEG seizure detection datasets demonstrate that our model significantly outperforms existing approaches (ranked in the first place in the 2025 "seizure detection challenge" organized in the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders), underscoring its potential for real-time, precise seizure detection. 

---
# Detecting and Mitigating Bias in LLMs through Knowledge Graph-Augmented Training 

**Authors**: Rajeev Kumar, Harishankar Kumar, Kumari Shalini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00310)  

**Abstract**: Large language models have revolutionized natural language processing with their surprising capability to understand and generate human-like text. However, many of these models inherit and further amplify the biases present in their training data, raising ethical and fairness concerns. The detection and mitigation of such biases are vital to ensuring that LLMs act responsibly and equitably across diverse domains. This work investigates Knowledge Graph-Augmented Training (KGAT) as a novel method to mitigate bias in LLM. Using structured domain-specific knowledge from real-world knowledge graphs, we improve the understanding of the model and reduce biased output. Public datasets for bias assessment include Gender Shades, Bias in Bios, and FairFace, while metrics such as demographic parity and equal opportunity facilitate rigorous detection. We also performed targeted mitigation strategies to correct biased associations, leading to a significant drop in biased output and improved bias metrics. Equipped with real-world datasets and knowledge graphs, our framework is both scalable and effective, paving the way toward responsible deployment in sensitive and high-stakes applications. 

---
# FedPaI: Achieving Extreme Sparsity in Federated Learning via Pruning at Initialization 

**Authors**: Haonan Wang, Zeli Liu, Kajimusugura Hoshino, Tuo Zhang, John Paul Walters, Stephen Crago  

**Link**: [PDF](https://arxiv.org/pdf/2504.00308)  

**Abstract**: Federated Learning (FL) enables distributed training on edge devices but faces significant challenges due to resource constraints in edge environments, impacting both communication and computational efficiency. Existing iterative pruning techniques improve communication efficiency but are limited by their centralized design, which struggles with FL's decentralized and data-imbalanced nature, resulting in suboptimal sparsity levels. To address these issues, we propose FedPaI, a novel efficient FL framework that leverages Pruning at Initialization (PaI) to achieve extreme sparsity. FedPaI identifies optimal sparse connections at an early stage, maximizing model capacity and significantly reducing communication and computation overhead by fixing sparsity patterns at the start of training. To adapt to diverse hardware and software environments, FedPaI supports both structured and unstructured pruning. Additionally, we introduce personalized client-side pruning mechanisms for improved learning capacity and sparsity-aware server-side aggregation for enhanced efficiency. Experimental results demonstrate that FedPaI consistently outperforms existing efficient FL that applies conventional iterative pruning with significant leading in efficiency and model accuracy. For the first time, our proposed FedPaI achieves an extreme sparsity level of up to 98% without compromising the model accuracy compared to unpruned baselines, even under challenging non-IID settings. By employing our FedPaI with joint optimization of model learning capacity and sparsity, FL applications can benefit from faster convergence and accelerate the training by 6.4 to 7.9 times. 

---
# Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead 

**Authors**: Vidhisha Balachandran, Jingya Chen, Lingjiao Chen, Shivam Garg, Neel Joshi, Yash Lara, John Langford, Besmira Nushi, Vibhav Vineet, Yue Wu, Safoora Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00294)  

**Abstract**: Inference-time scaling can enhance the reasoning capabilities of large language models (LLMs) on complex problems that benefit from step-by-step problem solving. Although lengthening generated scratchpads has proven effective for mathematical tasks, the broader impact of this approach on other tasks remains less clear. In this work, we investigate the benefits and limitations of scaling methods across nine state-of-the-art models and eight challenging tasks, including math and STEM reasoning, calendar planning, NP-hard problems, navigation, and spatial reasoning. We compare conventional models (e.g., GPT-4o) with models fine-tuned for inference-time scaling (e.g., o1) through evaluation protocols that involve repeated model calls, either independently or sequentially with feedback. These evaluations approximate lower and upper performance bounds and potential for future performance improvements for each model, whether through enhanced training or multi-model inference systems. Our extensive empirical analysis reveals that the advantages of inference-time scaling vary across tasks and diminish as problem complexity increases. In addition, simply using more tokens does not necessarily translate to higher accuracy in these challenging regimes. Results from multiple independent runs with conventional models using perfect verifiers show that, for some tasks, these models can achieve performance close to the average performance of today's most advanced reasoning models. However, for other tasks, a significant performance gap remains, even in very high scaling regimes. Encouragingly, all models demonstrate significant gains when inference is further scaled with perfect verifiers or strong feedback, suggesting ample potential for future improvements. 

---
# Do Chinese models speak Chinese languages? 

**Authors**: Andrea W Wen-Yi, Unso Eun Seo Jo, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2504.00289)  

**Abstract**: The release of top-performing open-weight LLMs has cemented China's role as a leading force in AI development. Do these models support languages spoken in China? Or do they speak the same languages as Western models? Comparing multilingual capabilities is important for two reasons. First, language ability provides insights into pre-training data curation, and thus into resource allocation and development priorities. Second, China has a long history of explicit language policy, varying between inclusivity of minority languages and a Mandarin-first policy. To test whether Chinese LLMs today reflect an agenda about China's languages, we test performance of Chinese and Western open-source LLMs on Asian regional and Chinese minority languages. Our experiments on Information Parity and reading comprehension show Chinese models' performance across these languages correlates strongly (r=0.93) with Western models', with the sole exception being better Mandarin. Sometimes, Chinese models cannot identify languages spoken by Chinese minorities such as Kazakh and Uyghur, even though they are good at French and German. These results provide a window into current development priorities, suggest options for future development, and indicate guidance for end users. 

---
# Digital Twins in Biopharmaceutical Manufacturing: Review and Perspective on Human-Machine Collaborative Intelligence 

**Authors**: Mohammed Aatif Shahab, Francesco Destro, Richard D. Braatz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00286)  

**Abstract**: The biopharmaceutical industry is increasingly developing digital twins to digitalize and automate the manufacturing process in response to the growing market demands. However, this shift presents significant challenges for human operators, as the complexity and volume of information can overwhelm their ability to manage the process effectively. These issues are compounded when digital twins are designed without considering interaction and collaboration with operators, who are responsible for monitoring processes and assessing situations, particularly during abnormalities. Our review of current trends in biopharma digital twin development reveals a predominant focus on technology and often overlooks the critical role of human operators. To bridge this gap, this article proposes a collaborative intelligence framework that emphasizes the integration of operators with digital twins. Approaches to system design that can enhance operator trust and human-machine interface usability are presented. Moreover, innovative training programs for preparing operators to understand and utilize digital twins are discussed. The framework outlined in this article aims to enhance collaboration between operators and digital twins effectively by using their full capabilities to boost resilience and productivity in biopharmaceutical manufacturing. 

---
# SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers 

**Authors**: Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00255)  

**Abstract**: This study evaluates large language models (LLMs) in generating code from algorithm descriptions from recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a multi-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implement solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful Non-Reasoning LLMs and Reasoning LLMs as foundational models. The best-performing LLM using Sci-Reproducer achieves only 39% execution accuracy, highlighting the benchmark's this http URL analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We will open-source our benchmark, and code at this https URL. 

---
# ElaLoRA: Elastic & Learnable Low-Rank Adaptation for Efficient Model Fine-Tuning 

**Authors**: Huandong Chang, Zicheng Ma, Mingyuan Ma, Zhenting Qi, Andrew Sabot, Hong Jiang, H. T. Kung  

**Link**: [PDF](https://arxiv.org/pdf/2504.00254)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted technique for fine-tuning large-scale pre-trained models with minimal parameter updates. However, existing methods rely on fixed ranks or focus solely on either rank pruning or expansion, failing to adapt ranks dynamically to match the importance of different layers during training. In this work, we propose ElaLoRA, an adaptive low-rank adaptation framework that dynamically prunes and expands ranks based on gradient-derived importance scores. To the best of our knowledge, ElaLoRA is the first method that enables both rank pruning and expansion during fine-tuning. Experiments across multiple benchmarks demonstrate that ElaLoRA consistently outperforms existing PEFT methods across different parameter budgets. Furthermore, our studies validate that layers receiving higher rank allocations contribute more significantly to model performance, providing theoretical justification for our adaptive strategy. By introducing a principled and adaptive rank allocation mechanism, ElaLoRA offers a scalable and efficient fine-tuning solution, particularly suited for resource-constrained environments. 

---
# MultiMorph: On-demand Atlas Construction 

**Authors**: S. Mazdak Abulnaga, Andrew Hoopes, Neel Dey, Malte Hoffmann, Marianne Rakic, Bruce Fischl, John Guttag, Adrian Dalca  

**Link**: [PDF](https://arxiv.org/pdf/2504.00247)  

**Abstract**: We present MultiMorph, a fast and efficient method for constructing anatomical atlases on the fly. Atlases capture the canonical structure of a collection of images and are essential for quantifying anatomical variability across populations. However, current atlas construction methods often require days to weeks of computation, thereby discouraging rapid experimentation. As a result, many scientific studies rely on suboptimal, precomputed atlases from mismatched populations, negatively impacting downstream analyses. MultiMorph addresses these challenges with a feedforward model that rapidly produces high-quality, population-specific atlases in a single forward pass for any 3D brain dataset, without any fine-tuning or optimization. MultiMorph is based on a linear group-interaction layer that aggregates and shares features within the group of input images. Further, by leveraging auxiliary synthetic data, MultiMorph generalizes to new imaging modalities and population groups at test-time. Experimentally, MultiMorph outperforms state-of-the-art optimization-based and learning-based atlas construction methods in both small and large population settings, with a 100-fold reduction in time. This makes MultiMorph an accessible framework for biomedical researchers without machine learning expertise, enabling rapid, high-quality atlas generation for diverse studies. 

---
# Synthesizing Public Opinions with LLMs: Role Creation, Impacts, and the Future to eDemorcacy 

**Authors**: Rabimba Karanjai, Boris Shor, Amanda Austin, Ryan Kennedy, Yang Lu, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00241)  

**Abstract**: This paper investigates the use of Large Language Models (LLMs) to synthesize public opinion data, addressing challenges in traditional survey methods like declining response rates and non-response bias. We introduce a novel technique: role creation based on knowledge injection, a form of in-context learning that leverages RAG and specified personality profiles from the HEXACO model and demographic information, and uses that for dynamically generated prompts. This method allows LLMs to simulate diverse opinions more accurately than existing prompt engineering approaches. We compare our results with pre-trained models with standard few-shot prompts. Experiments using questions from the Cooperative Election Study (CES) demonstrate that our role-creation approach significantly improves the alignment of LLM-generated opinions with real-world human survey responses, increasing answer adherence. In addition, we discuss challenges, limitations and future research directions. 

---
# GazeLLM: Multimodal LLMs incorporating Human Visual Attention 

**Authors**: Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.00221)  

**Abstract**: Large Language Models (LLMs) are advancing into Multimodal LLMs (MLLMs), capable of processing image, audio, and video as well as text. Combining first-person video, MLLMs show promising potential for understanding human activities through video and audio, enabling many human-computer interaction and human-augmentation applications such as human activity support, real-world agents, and skill transfer to robots or other individuals. However, handling high-resolution, long-duration videos generates large latent representations, leading to substantial memory and processing demands, limiting the length and resolution MLLMs can manage. Reducing video resolution can lower memory usage but often compromises comprehension. This paper introduces a method that optimizes first-person video analysis by integrating eye-tracking data, and proposes a method that decomposes first-person vision video into sub areas for regions of gaze focus. By processing these selectively gazed-focused inputs, our approach achieves task comprehension equivalent to or even better than processing the entire image at full resolution, but with significantly reduced video data input (reduce the number of pixels to one-tenth), offering an efficient solution for using MLLMs to interpret and utilize human skills. 

---
# Can Diffusion Models Disentangle? A Theoretical Perspective 

**Authors**: Liming Wang, Muhammad Jehanzeb Mirza, Yishu Gong, Yuan Gong, Jiaqi Zhang, Brian H. Tracey, Katerina Placek, Marco Vilela, James R. Glass  

**Link**: [PDF](https://arxiv.org/pdf/2504.00220)  

**Abstract**: This paper presents a novel theoretical framework for understanding how diffusion models can learn disentangled representations. Within this framework, we establish identifiability conditions for general disentangled latent variable models, analyze training dynamics, and derive sample complexity bounds for disentangled latent subspace models. To validate our theory, we conduct disentanglement experiments across diverse tasks and modalities, including subspace recovery in latent subspace Gaussian mixture models, image colorization, image denoising, and voice conversion for speech classification. Additionally, our experiments show that training strategies inspired by our theory, such as style guidance regularization, consistently enhance disentanglement performance. 

---
# $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks 

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00218)  

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms. 

---
# RailGoerl24: Görlitz Rail Test Center CV Dataset 2024 

**Authors**: Rustam Tagiew, Ilkay Wunderlich, Mark Sastuba, Steffen Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00204)  

**Abstract**: Driverless train operation for open tracks on urban guided transport and mainline railways requires, among other things automatic detection of actual and potential obstacles, especially humans, in the danger zone of the train's path. Machine learning algorithms have proven to be powerful state-of-the-art tools for this task. However, these algorithms require large amounts of high-quality annotated data containing human beings in railway-specific environments as training data. Unfortunately, the amount of publicly available datasets is not yet sufficient and is significantly inferior to the datasets in the road domain. Therefore, this paper presents RailGoerl24, an on-board visual light Full HD camera dataset of 12205 frames recorded in a railway test center of TÜV SÜD Rail, in Görlitz, Germany. Its main purpose is to support the development of driverless train operation for guided transport. RailGoerl24 also includes a terrestrial LiDAR scan covering parts of the area used to acquire the RGB data. In addition to the raw data, the dataset contains 33556 boxwise annotations in total for the object class 'person'. The faces of recorded actors are not blurred or altered in any other way. RailGoerl24, soon available at this http URL, can also be used for tasks beyond collision prediction. 

---
# Identifying Sparsely Active Circuits Through Local Loss Landscape Decomposition 

**Authors**: Brianna Chrisman, Lucius Bushnaq, Lee Sharkey  

**Link**: [PDF](https://arxiv.org/pdf/2504.00194)  

**Abstract**: Much of mechanistic interpretability has focused on understanding the activation spaces of large neural networks. However, activation space-based approaches reveal little about the underlying circuitry used to compute features. To better understand the circuits employed by models, we introduce a new decomposition method called Local Loss Landscape Decomposition (L3D). L3D identifies a set of low-rank subnetworks: directions in parameter space of which a subset can reconstruct the gradient of the loss between any sample's output and a reference output vector. We design a series of progressively more challenging toy models with well-defined subnetworks and show that L3D can nearly perfectly recover the associated subnetworks. Additionally, we investigate the extent to which perturbing the model in the direction of a given subnetwork affects only the relevant subset of samples. Finally, we apply L3D to a real-world transformer model and a convolutional neural network, demonstrating its potential to identify interpretable and relevant circuits in parameter space. 

---
# Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified? 

**Authors**: Olawale Salaudeen, Nicole Chiou, Shiny Weng, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00186)  

**Abstract**: Spurious correlations are unstable statistical associations that hinder robust decision-making. Conventional wisdom suggests that models relying on such correlations will fail to generalize out-of-distribution (OOD), especially under strong distribution shifts. However, empirical evidence challenges this view as naive in-distribution empirical risk minimizers often achieve the best OOD accuracy across popular OOD generalization benchmarks. In light of these results, we propose a different perspective: many widely used benchmarks for evaluating robustness to spurious correlations are misspecified. Specifically, they fail to include shifts in spurious correlations that meaningfully impact OOD generalization, making them unsuitable for evaluating the benefit of removing such correlations. We establish conditions under which a distribution shift can reliably assess a model's reliance on spurious correlations. Crucially, under these conditions, we should not observe a strong positive correlation between in-distribution and OOD accuracy, often called "accuracy on the line." Yet, most state-of-the-art benchmarks exhibit this pattern, suggesting they do not effectively assess robustness. Our findings expose a key limitation in current benchmarks used to evaluate domain generalization algorithms, that is, models designed to avoid spurious correlations. We highlight the need to rethink how robustness to spurious correlations is assessed, identify well-specified benchmarks the field should prioritize, and enumerate strategies for designing future benchmarks that meaningfully reflect robustness under distribution shift. 

---
# Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency 

**Authors**: Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran  

**Link**: [PDF](https://arxiv.org/pdf/2504.00180)  

**Abstract**: Retrieval Augmented Generation (RAG) systems have emerged as a powerful method for enhancing large language models (LLMs) with up-to-date information. However, the retrieval step in RAG can sometimes surface documents containing contradictory information, particularly in rapidly evolving domains such as news. These contradictions can significantly impact the performance of LLMs, leading to inconsistent or erroneous outputs. This study addresses this critical challenge in two ways. First, we present a novel data generation framework to simulate different types of contradictions that may occur in the retrieval stage of a RAG system. Second, we evaluate the robustness of different LLMs in performing as context validators, assessing their ability to detect contradictory information within retrieved document sets. Our experimental results reveal that context validation remains a challenging task even for state-of-the-art LLMs, with performance varying significantly across different types of contradictions. While larger models generally perform better at contradiction detection, the effectiveness of different prompting strategies varies across tasks and model architectures. We find that chain-of-thought prompting shows notable improvements for some models but may hinder performance in others, highlighting the complexity of the task and the need for more robust approaches to context validation in RAG systems. 

---
# Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier 

**Authors**: Craig W. Schmidt, Varshini Reddy, Chris Tanner, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2504.00178)  

**Abstract**: Pre-tokenization, the initial step in many modern tokenization pipelines, segments text into smaller units called pretokens, typically splitting on whitespace and punctuation. While this process encourages having full, individual words as tokens, it introduces a fundamental limitation in most tokenization algorithms such as Byte Pair Encoding (BPE). Specifically, pre-tokenization causes the distribution of tokens in a corpus to heavily skew towards common, full-length words. This skewed distribution limits the benefits of expanding to larger vocabularies, since the additional tokens appear with progressively lower counts. To overcome this barrier, we propose BoundlessBPE, a modified BPE algorithm that relaxes the pretoken boundary constraint. Our approach selectively merges two complete pretokens into a larger unit we term a superword. Superwords are not necessarily semantically cohesive. For example, the pretokens " of" and " the" might be combined to form the superword " of the". This merging strategy results in a substantially more uniform distribution of tokens across a corpus than standard BPE, and compresses text more effectively, with an approximate 20% increase in bytes per token. 

---
# MetaCLBench: Meta Continual Learning Benchmark on Resource-Constrained Edge Devices 

**Authors**: Sijia Li, Young D. Kwon, Lik-Hang Lee, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2504.00174)  

**Abstract**: Meta-Continual Learning (Meta-CL) has emerged as a promising approach to minimize manual labeling efforts and system resource requirements by enabling Continual Learning (CL) with limited labeled samples. However, while existing methods have shown success in image-based tasks, their effectiveness remains unexplored for sequential time-series data from sensor systems, particularly audio inputs. To address this gap, we conduct a comprehensive benchmark study evaluating six representative Meta-CL approaches using three network architectures on five datasets from both image and audio modalities. We develop MetaCLBench, an end-to-end Meta-CL benchmark framework for edge devices to evaluate system overheads and investigate trade-offs among performance, computational costs, and memory requirements across various Meta-CL methods. Our results reveal that while many Meta-CL methods enable to learn new classes for both image and audio modalities, they impose significant computational and memory costs on edge devices. Also, we find that pre-training and meta-training procedures based on source data before deployment improve Meta-CL performance. Finally, to facilitate further research, we provide practical guidelines for researchers and machine learning practitioners implementing Meta-CL on resource-constrained environments and make our benchmark framework and tools publicly available, enabling fair evaluation across both accuracy and system-level metrics. 

---
# Backdoor Detection through Replicated Execution of Outsourced Training 

**Authors**: Hengrui Jia, Sierra Wyllie, Akram Bin Sediq, Ahmed Ibrahim, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2504.00170)  

**Abstract**: It is common practice to outsource the training of machine learning models to cloud providers. Clients who do so gain from the cloud's economies of scale, but implicitly assume trust: the server should not deviate from the client's training procedure. A malicious server may, for instance, seek to insert backdoors in the model. Detecting a backdoored model without prior knowledge of both the backdoor attack and its accompanying trigger remains a challenging problem. In this paper, we show that a client with access to multiple cloud providers can replicate a subset of training steps across multiple servers to detect deviation from the training procedure in a similar manner to differential testing. Assuming some cloud-provided servers are benign, we identify malicious servers by the substantial difference between model updates required for backdooring and those resulting from clean training. Perhaps the strongest advantage of our approach is its suitability to clients that have limited-to-no local compute capability to perform training; we leverage the existence of multiple cloud providers to identify malicious updates without expensive human labeling or heavy computation. We demonstrate the capabilities of our approach on an outsourced supervised learning task where $50\%$ of the cloud providers insert their own backdoor; our approach is able to correctly identify $99.6\%$ of them. In essence, our approach is successful because it replaces the signature-based paradigm taken by existing approaches with an anomaly-based detection paradigm. Furthermore, our approach is robust to several attacks from adaptive adversaries utilizing knowledge of our detection scheme. 

---
# Does "Reasoning" with Large Language Models Improve Recognizing, Generating, and Reframing Unhelpful Thoughts? 

**Authors**: Yilin Qi, Dong Won Lee, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.00163)  

**Abstract**: Cognitive Reframing, a core element of Cognitive Behavioral Therapy (CBT), helps individuals reinterpret negative experiences by finding positive meaning. Recent advances in Large Language Models (LLMs) have demonstrated improved performance through reasoning-based strategies. This inspires a promising direction of leveraging the reasoning capabilities of LLMs to improve CBT and mental reframing by simulating the process of critical thinking, potentially enabling more effective recognition, generation, and reframing of cognitive distortions. In this work, we investigate the role of various reasoning methods, including pre-trained reasoning LLMs and augmented reasoning strategies such as CoT and self-consistency in enhancing LLMs' ability to perform cognitive reframing tasks. We find that augmented reasoning methods, even when applied to "outdated" LLMs like GPT-3.5, consistently outperform state-of-the-art pretrained reasoning models on recognizing, generating and reframing unhelpful thoughts. 

---
# Towards Precise Action Spotting: Addressing Temporal Misalignment in Labels with Dynamic Label Assignment 

**Authors**: Masato Tamura  

**Link**: [PDF](https://arxiv.org/pdf/2504.00149)  

**Abstract**: Precise action spotting has attracted considerable attention due to its promising applications. While existing methods achieve substantial performance by employing well-designed model architecture, they overlook a significant challenge: the temporal misalignment inherent in ground-truth labels. This misalignment arises when frames labeled as containing events do not align accurately with the actual event times, often as a result of human annotation errors or the inherent difficulties in precisely identifying event boundaries across neighboring frames. To tackle this issue, we propose a novel dynamic label assignment strategy that allows predictions to have temporal offsets from ground-truth action times during training, ensuring consistent event spotting. Our method extends the concept of minimum-cost matching, which is utilized in the spatial domain for object detection, to the temporal domain. By calculating matching costs based on predicted action class scores and temporal offsets, our method dynamically assigns labels to the most likely predictions, even when the predicted times of these predictions deviate from ground-truth times, alleviating the negative effects of temporal misalignment in labels. We conduct extensive experiments and demonstrate that our method achieves state-of-the-art performance, particularly in conditions where events are visually distinct and temporal misalignment in labels is common. 

---
# Lorentzian Graph Isomorphic Network 

**Authors**: Srinitish Srinivasan, Omkumar CU  

**Link**: [PDF](https://arxiv.org/pdf/2504.00142)  

**Abstract**: We introduce the Lorentzian Graph Isomorphic Network (LGIN), a novel graph neural network (GNN) designed to operate in hyperbolic spaces, leveraging the Lorentzian model to enhance graph representation learning. Existing GNNs primarily operate in Euclidean spaces, which can limit their ability to capture hierarchical and multi-relational structures inherent to complex graphs. LGIN addresses this by incorporating curvature-aware aggregation functions that preserve the Lorentzian metric tensor, ensuring embeddings remain constrained within the hyperbolic space by proposing a new update rule that effectively captures both local neighborhood interactions and global structural properties, enabling LGIN to distinguish non-isomorphic graphs with expressiveness at least as powerful as the Weisfeiler-Lehman test. Through extensive evaluation across nine benchmark datasets, including molecular and protein structures, LGIN consistently outperforms or matches state-of-the-art GNNs, demonstrating its robustness and efficacy in modeling complex graph structures. To the best of our knowledge, this is the first study to extend the concept of a powerful graph neural network to Riemannian manifolds, paving the way for future advancements in hyperbolic graph learning. The code for our paper can be found at this https URL. 

---
# Data-driven Power Loss Identification through Physics-Based Thermal Model Backpropagation 

**Authors**: Mattia Scarpa, Francesco Pase, Ruggero Carli, Mattia Bruschetta, Franscesco Toso  

**Link**: [PDF](https://arxiv.org/pdf/2504.00133)  

**Abstract**: Digital twins for power electronics require accurate power losses whose direct measurements are often impractical or impossible in real-world applications. This paper presents a novel hybrid framework that combines physics-based thermal modeling with data-driven techniques to identify and correct power losses accurately using only temperature measurements. Our approach leverages a cascaded architecture where a neural network learns to correct the outputs of a nominal power loss model by backpropagating through a reduced-order thermal model. We explore two neural architectures, a bootstrapped feedforward network, and a recurrent neural network, demonstrating that the bootstrapped feedforward approach achieves superior performance while maintaining computational efficiency for real-time applications. Between the interconnection, we included normalization strategies and physics-guided training loss functions to preserve stability and ensure physical consistency. Experimental results show that our hybrid model reduces both temperature estimation errors (from 7.2+-6.8°C to 0.3+-0.3°C) and power loss prediction errors (from 5.4+-6.6W to 0.2+-0.3W) compared to traditional physics-based approaches, even in the presence of thermal model uncertainties. This methodology allows us to accurately estimate power losses without direct measurements, making it particularly helpful for real-time industrial applications where sensor placement is hindered by cost and physical limitations. 

---
# Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting 

**Authors**: Reza Nematirad, Anil Pahwa, Balasubramaniam Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00118)  

**Abstract**: Time series forecasting is an important application in various domains such as energy management, traffic planning, financial markets, meteorology, and medicine. However, real-time series data often present intricate temporal variability and sharp fluctuations, which pose significant challenges for time series forecasting. Previous models that rely on 1D time series representations usually struggle with complex temporal variations. To address the limitations of 1D time series, this study introduces the Times2D method that transforms the 1D time series into 2D space. Times2D consists of three main parts: first, a Periodic Decomposition Block (PDB) that captures temporal variations within a period and between the same periods by converting the time series into a 2D tensor in the frequency domain. Second, the First and Second Derivative Heatmaps (FSDH) capture sharp changes and turning points, respectively. Finally, an Aggregation Forecasting Block (AFB) integrates the output tensors from PDB and FSDH for accurate forecasting. This 2D transformation enables the utilization of 2D convolutional operations to effectively capture long and short characteristics of the time series. Comprehensive experimental results across large-scale data in the literature demonstrate that the proposed Times2D model achieves state-of-the-art performance in both short-term and long-term forecasting. The code is available in this repository: this https URL. 

---
# Assessing Code Understanding in LLMs 

**Authors**: Cosimo Laneve, Alvise Spanò, Dalila Ressi, Sabina Rossi, Michele Bugliesi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00065)  

**Abstract**: We present an empirical evaluation of Large Language Models in code understanding associated with non-trivial, semantic-preserving program transformations such as copy propagation or constant folding. Our findings show that LLMs fail to judge semantic equivalence in approximately 41\% of cases when no context is provided and in 29\% when given a simple generic context. To improve accuracy, we advocate integrating LLMs with code-optimization tools to enhance training and facilitate more robust program understanding. 

---
# Evaluating the Feasibility and Accuracy of Large Language Models for Medical History-Taking in Obstetrics and Gynecology 

**Authors**: Dou Liu, Ying Long, Sophia Zuoqiu, Tian Tang, Rong Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00061)  

**Abstract**: Effective physician-patient communications in pre-diagnostic environments, and most specifically in complex and sensitive medical areas such as infertility, are critical but consume a lot of time and, therefore, cause clinic workflows to become inefficient. Recent advancements in Large Language Models (LLMs) offer a potential solution for automating conversational medical history-taking and improving diagnostic accuracy. This study evaluates the feasibility and performance of LLMs in those tasks for infertility cases. An AI-driven conversational system was developed to simulate physician-patient interactions with ChatGPT-4o and ChatGPT-4o-mini. A total of 70 real-world infertility cases were processed, generating 420 diagnostic histories. Model performance was assessed using F1 score, Differential Diagnosis (DDs) Accuracy, and Accuracy of Infertility Type Judgment (ITJ). ChatGPT-4o-mini outperformed ChatGPT-4o in information extraction accuracy (F1 score: 0.9258 vs. 0.9029, p = 0.045, d = 0.244) and demonstrated higher completeness in medical history-taking (97.58% vs. 77.11%), suggesting that ChatGPT-4o-mini is more effective in extracting detailed patient information, which is critical for improving diagnostic accuracy. In contrast, ChatGPT-4o performed slightly better in differential diagnosis accuracy (2.0524 vs. 2.0048, p > 0.05). ITJ accuracy was higher in ChatGPT-4o-mini (0.6476 vs. 0.5905) but with lower consistency (Cronbach's $\alpha$ = 0.562), suggesting variability in classification reliability. Both models demonstrated strong feasibility in automating infertility history-taking, with ChatGPT-4o-mini excelling in completeness and extraction accuracy. In future studies, expert validation for accuracy and dependability in a clinical setting, AI model fine-tuning, and larger datasets with a mix of cases of infertility have to be prioritized. 

---
# CF-CAM: Gradient Perturbation Mitigation and Feature Stabilization for Reliable Interpretability 

**Authors**: Hongjie He, Xu Pan, Yudong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00060)  

**Abstract**: As deep learning continues to advance, the opacity of neural network decision-making remains a critical challenge, limiting trust and applicability in high-stakes domains. Class Activation Mapping (CAM) techniques have emerged as a key approach to visualizing model decisions, yet existing methods face inherent trade-offs. Gradient-based CAM variants suffer from sensitivity to gradient perturbations, leading to unstable and unreliable explanations. Conversely, gradient-free approaches mitigate gradient instability but incur significant computational overhead and inference latency. To address these limitations, we propose Cluster Filter Class Activation Map (CF-CAM), a novel framework that reintroduces gradient-based weighting while enhancing robustness against gradient noise. CF-CAM employs a hierarchical importance weighting strategy to balance discriminative feature preservation and noise elimination. A density-aware channel clustering via Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups semantically relevant feature channels and discard noise-prone activations. Additionally, cluster-conditioned gradient filtering leverages bilateral filters to refine gradient signals, preserving edge-aware localization while suppressing noise impact. Experiment results demonstrate that CF-CAM achieves superior interpretability performance while maintaining resilience to gradient perturbations, outperforming state-of-the-art CAM methods in faithfulness and robustness. By effectively mitigating gradient instability without excessive computational cost, CF-CAM provides a reliable solution for enhancing the interpretability of deep neural networks in critical applications such as medical diagnosis and autonomous driving. 

---
# GAL-MAD: Towards Explainable Anomaly Detection in Microservice Applications Using Graph Attention Networks 

**Authors**: Lahiru Akmeemana, Chamodya Attanayake, Husni Faiz, Sandareka Wickramanayake  

**Link**: [PDF](https://arxiv.org/pdf/2504.00058)  

**Abstract**: The transition to microservices has revolutionized software architectures, offering enhanced scalability and modularity. However, the distributed and dynamic nature of microservices introduces complexities in ensuring system reliability, making anomaly detection crucial for maintaining performance and functionality. Anomalies stemming from network and performance issues must be swiftly identified and addressed. Existing anomaly detection techniques often rely on statistical models or machine learning methods that struggle with the high-dimensional, interdependent data inherent in microservice applications. Current techniques and available datasets predominantly focus on system traces and logs, limiting their ability to support advanced detection models. This paper addresses these gaps by introducing the RS-Anomic dataset generated using the open-source RobotShop microservice application. The dataset captures multivariate performance metrics and response times under normal and anomalous conditions, encompassing ten types of anomalies. We propose a novel anomaly detection model called Graph Attention and LSTM-based Microservice Anomaly Detection (GAL-MAD), leveraging Graph Attention and Long Short-Term Memory architectures to capture spatial and temporal dependencies in microservices. We utilize SHAP values to localize anomalous services and identify root causes to enhance explainability. Experimental results demonstrate that GAL-MAD outperforms state-of-the-art models on the RS-Anomic dataset, achieving higher accuracy and recall across varying anomaly rates. The explanations provide actionable insights into service anomalies, which benefits system administrators. 

---
# Integrating Large Language Models with Human Expertise for Disease Detection in Electronic Health Records 

**Authors**: Jie Pan, Seungwon Lee, Cheligeer Cheligeer, Elliot A. Martin, Kiarash Riazi, Hude Quan, Na Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00053)  

**Abstract**: Objective: Electronic health records (EHR) are widely available to complement administrative data-based disease surveillance and healthcare performance evaluation. Defining conditions from EHR is labour-intensive and requires extensive manual labelling of disease outcomes. This study developed an efficient strategy based on advanced large language models to identify multiple conditions from EHR clinical notes. Methods: We linked a cardiac registry cohort in 2015 with an EHR system in Alberta, Canada. We developed a pipeline that leveraged a generative large language model (LLM) to analyze, understand, and interpret EHR notes by prompts based on specific diagnosis, treatment management, and clinical guidelines. The pipeline was applied to detect acute myocardial infarction (AMI), diabetes, and hypertension. The performance was compared against clinician-validated diagnoses as the reference standard and widely adopted International Classification of Diseases (ICD) codes-based methods. Results: The study cohort accounted for 3,088 patients and 551,095 clinical notes. The prevalence was 55.4%, 27.7%, 65.9% and for AMI, diabetes, and hypertension, respectively. The performance of the LLM-based pipeline for detecting conditions varied: AMI had 88% sensitivity, 63% specificity, and 77% positive predictive value (PPV); diabetes had 91% sensitivity, 86% specificity, and 71% PPV; and hypertension had 94% sensitivity, 32% specificity, and 72% PPV. Compared with ICD codes, the LLM-based method demonstrated improved sensitivity and negative predictive value across all conditions. The monthly percentage trends from the detected cases by LLM and reference standard showed consistent patterns. 

---
# The Cursive Transformer 

**Authors**: Sam Greydanus, Zachary Wimpee  

**Link**: [PDF](https://arxiv.org/pdf/2504.00051)  

**Abstract**: Transformers trained on tokenized text, audio, and images can generate high-quality autoregressive samples. But handwriting data, represented as sequences of pen coordinates, remains underexplored. We introduce a novel tokenization scheme that converts pen stroke offsets to polar coordinates, discretizes them into bins, and then turns them into sequences of tokens with which to train a standard GPT model. This allows us to capture complex stroke distributions without using any specialized architectures (eg. the mixture density network or the self-advancing ASCII attention head from Graves 2014). With just 3,500 handwritten words and a few simple data augmentations, we are able to train a model that can generate realistic cursive handwriting. Our approach is simpler and more performant than previous RNN-based methods. 

---
# JudgeLRM: Large Reasoning Models as a Judge 

**Authors**: Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00050)  

**Abstract**: The rise of Large Language Models (LLMs) as evaluators offers a scalable alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for judges approaches often fall short in domains requiring complex reasoning. In this work, we investigate whether LLM judges truly benefit from enhanced reasoning capabilities. Through a detailed analysis of reasoning requirements across evaluation tasks, we reveal a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples - highlighting the limitations of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM models consistently outperform both SFT-tuned and state-of-the-art reasoning models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks requiring deep reasoning. 

---
# Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs 

**Authors**: Cong Duy Vu Hoang, Gioacchino Tangari, Clemence Lanfranchi, Dalu Guo, Paul Cayet, Steve Siu, Don Dharmasiri, Yuan-Fang Li, Long Duong, Damien Hilloulin, Rhicheek Patra, Sungpack Hong, Hassan Chafi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00048)  

**Abstract**: The growing adoption of large language models (LLMs) in business applications has amplified interest in Natural Language to SQL (NL2SQL) solutions, in which there is competing demand for high performance and efficiency. Domain- and customer-specific requirements further complicate the problem. To address this conundrum, we introduce Distill-C, a distilled customization framework tailored for NL2SQL tasks. Distill-C utilizes large teacher LLMs to produce high-quality synthetic data through a robust and scalable pipeline. Finetuning smaller and open-source LLMs on this synthesized data enables them to rival or outperform teacher models an order of magnitude larger. Evaluated on multiple challenging benchmarks, Distill-C achieves an average improvement of 36% in execution accuracy compared to the base models from three distinct LLM families. Additionally, on three internal customer benchmarks, Distill-C demonstrates a 22.6% performance improvement over the base models. Our results demonstrate that Distill-C is an effective, high-performing and generalizable approach for deploying lightweight yet powerful NL2SQL models, delivering exceptional accuracies while maintaining low computational cost. 

---
# EAP4EMSIG -- Enhancing Event-Driven Microscopy for Microfluidic Single-Cell Analysis 

**Authors**: Nils Friederich, Angelo Jovin Yamachui Sitcheu, Annika Nassal, Erenus Yildiz, Matthias Pesch, Maximilian Beichter, Lukas Scholtes, Bahar Akbaba, Thomas Lautenschlager, Oliver Neumann, Dietrich Kohlheyer, Hanno Scharr, Johannes Seiffarth, Katharina Nöh, Ralf Mikut  

**Link**: [PDF](https://arxiv.org/pdf/2504.00047)  

**Abstract**: Microfluidic Live-Cell Imaging yields data on microbial cell factories. However, continuous acquisition is challenging as high-throughput experiments often lack realtime insights, delaying responses to stochastic events. We introduce three components in the Experiment Automation Pipeline for Event-Driven Microscopy to Smart Microfluidic Single-Cell Analysis: a fast, accurate Deep Learning autofocusing method predicting the focus offset, an evaluation of real-time segmentation methods and a realtime data analysis dashboard. Our autofocusing achieves a Mean Absolute Error of 0.0226\textmu m with inference times below 50~ms. Among eleven Deep Learning segmentation methods, Cellpose~3 reached a Panoptic Quality of 93.58\%, while a distance-based method is fastest (121~ms, Panoptic Quality 93.02\%). All six Deep Learning Foundation Models were unsuitable for real-time segmentation. 

---
# Multi-Stakeholder Disaster Insights from Social Media Using Large Language Models 

**Authors**: Loris Belcastro, Cristian Cosentino, Fabrizio Marozzo, Merve Gündüz-Cüre, Şule Öztürk-Birim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00046)  

**Abstract**: In recent years, social media has emerged as a primary channel for users to promptly share feedback and issues during disasters and emergencies, playing a key role in crisis management. While significant progress has been made in collecting and analyzing social media content, there remains a pressing need to enhance the automation, aggregation, and customization of this data to deliver actionable insights tailored to diverse stakeholders, including the press, police, EMS, and firefighters. This effort is essential for improving the coordination of activities such as relief efforts, resource distribution, and media communication. This paper presents a methodology that leverages the capabilities of LLMs to enhance disaster response and management. Our approach combines classification techniques with generative AI to bridge the gap between raw user feedback and stakeholder-specific reports. Social media posts shared during catastrophic events are analyzed with a focus on user-reported issues, service interruptions, and encountered challenges. We employ full-spectrum LLMs, using analytical models like BERT for precise, multi-dimensional classification of content type, sentiment, emotion, geolocation, and topic. Generative models such as ChatGPT are then used to produce human-readable, informative reports tailored to distinct audiences, synthesizing insights derived from detailed classifications. We compare standard approaches, which analyze posts directly using prompts in ChatGPT, to our advanced method, which incorporates multi-dimensional classification, sub-event selection, and tailored report generation. Our methodology demonstrates superior performance in both quantitative metrics, such as text coherence scores and latent representations, and qualitative assessments by automated tools and field experts, delivering precise insights for diverse disaster response stakeholders. 

---
# CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation 

**Authors**: Jixuan Leng, Chengsong Huang, Langlin Huang, Bill Yuchen Lin, William W. Cohen, Haohan Wang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00043)  

**Abstract**: Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly either assess text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles-a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in multiple formats (text and image) and offers different evaluation strategies ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs outperform non-reasoning models substantially by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings offer insights into the limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations. 

---
# Quantum Methods for Managing Ambiguity in Natural Language Processing 

**Authors**: Jurek Eisinger, Ward Gauderis, Lin de Huybrecht, Geraint A. Wiggins  

**Link**: [PDF](https://arxiv.org/pdf/2504.00040)  

**Abstract**: The Categorical Compositional Distributional (DisCoCat) framework models meaning in natural language using the mathematical framework of quantum theory, expressed as formal diagrams. DisCoCat diagrams can be associated with tensor networks and quantum circuits. DisCoCat diagrams have been connected to density matrices in various contexts in Quantum Natural Language Processing (QNLP). Previous use of density matrices in QNLP entails modelling ambiguous words as probability distributions over more basic words (the word \texttt{queen}, e.g., might mean the reigning queen or the chess piece). In this article, we investigate using probability distributions over processes to account for syntactic ambiguity in sentences. The meanings of these sentences are represented by density matrices. We show how to create probability distributions on quantum circuits that represent the meanings of sentences and explain how this approach generalises tasks from the literature. We conduct an experiment to validate the proposed theory. 

---
# Revisiting the Relationship between Adversarial and Clean Training: Why Clean Training Can Make Adversarial Training Better 

**Authors**: MingWei Zhou, Xiaobing Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00038)  

**Abstract**: Adversarial training (AT) is an effective technique for enhancing adversarial robustness, but it usually comes at the cost of a decline in generalization ability. Recent studies have attempted to use clean training to assist adversarial training, yet there are contradictions among the conclusions. We comprehensively summarize the representative strategies and, with a focus on the multi - view hypothesis, provide a unified explanation for the contradictory phenomena among different studies. In addition, we conduct an in - depth analysis of the knowledge combinations transferred from clean - trained models to adversarially - trained models in previous studies, and find that they can be divided into two categories: reducing the learning difficulty and providing correct guidance. Based on this finding, we propose a new idea of leveraging clean training to further improve the performance of advanced AT this http URL reveal that the problem of generalization degradation faced by AT partly stems from the difficulty of adversarial training in learning certain sample features, and this problem can be alleviated by making full use of clean training. 

---
# ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models 

**Authors**: Guoyizhe Wei, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2504.00037)  

**Abstract**: Vision Transformers (ViTs) have delivered remarkable progress through global self-attention, yet their quadratic complexity can become prohibitive for high-resolution inputs. In this work, we present ViT-Linearizer, a cross-architecture distillation framework that transfers rich ViT representations into a linear-time, recurrent-style model. Our approach leverages 1) activation matching, an intermediate constraint that encourages student to align its token-wise dependencies with those produced by the teacher, and 2) masked prediction, a contextual reconstruction objective that requires the student to predict the teacher's representations for unseen (masked) tokens, to effectively distill the quadratic self-attention knowledge into the student while maintaining efficient complexity. Empirically, our method provides notable speedups particularly for high-resolution tasks, significantly addressing the hardware challenges in inference. Additionally, it also elevates Mamba-based architectures' performance on standard vision benchmarks, achieving a competitive 84.3% top-1 accuracy on ImageNet with a base-sized model. Our results underscore the good potential of RNN-based solutions for large-scale visual tasks, bridging the gap between theoretical efficiency and real-world practice. 

---
# Improving Diseases Predictions Utilizing External Bio-Banks 

**Authors**: Hido Pinto, Eran Segal  

**Link**: [PDF](https://arxiv.org/pdf/2504.00036)  

**Abstract**: Machine learning has been successfully used in critical domains, such as medicine. However, extracting meaningful insights from biomedical data is often constrained by the lack of their available disease labels. In this research, we demonstrate how machine learning can be leveraged to enhance explainability and uncover biologically meaningful associations, even when predictive improvements in disease modeling are limited. We train LightGBM models from scratch on our dataset (10K) to impute metabolomics features and apply them to the UK Biobank (UKBB) for downstream analysis. The imputed metabolomics features are then used in survival analysis to assess their impact on disease-related risk factors. As a result, our approach successfully identified biologically relevant connections that were not previously known to the predictive models. Additionally, we applied a genome-wide association study (GWAS) on key metabolomics features, revealing a link between vascular dementia and smoking. Although being a well-established epidemiological relationship, this link was not embedded in the model's training data, which validated the method's ability to extract meaningful signals. Furthermore, by integrating survival models as inputs in the 10K data, we uncovered associations between metabolic substances and obesity, demonstrating the ability to infer disease risk for future patients without requiring direct outcome labels. These findings highlight the potential of leveraging external bio-banks to extract valuable biomedical insights, even in data-limited scenarios. Our results demonstrate that machine learning models trained on smaller datasets can still be used to uncover real biological associations when carefully integrated with survival analysis and genetic studies. 

---
# MiZero: The Shadowy Defender Against Text Style Infringements 

**Authors**: Ziwei Zhang, Juan Wen, Wanli Peng, Zhengxian Wu, Yinghan Zhou, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.00035)  

**Abstract**: In-Context Learning (ICL) and efficient fine-tuning methods significantly enhanced the efficiency of applying Large Language Models (LLMs) to downstream tasks. However, they also raise concerns about the imitation and infringement of personal creative data. Current methods for data copyright protection primarily focuses on content security but lacks effectiveness in protecting the copyrights of text styles. In this paper, we introduce a novel implicit zero-watermarking scheme, namely MiZero. This scheme establishes a precise watermark domain to protect the copyrighted style, surpassing traditional watermarking methods that distort the style characteristics. Specifically, we employ LLMs to extract condensed-lists utilizing the designed instance delimitation mechanism. These lists guide MiZero in generating the watermark. Extensive experiments demonstrate that MiZero effectively verifies text style copyright ownership against AI imitation. 

---
# Leaking LoRa: An Evaluation of Password Leaks and Knowledge Storage in Large Language Models 

**Authors**: Ryan Marinelli, Magnus Eckhoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.00031)  

**Abstract**: To effectively deploy Large Language Models (LLMs) in application-specific settings, fine-tuning techniques are applied to enhance performance on specialized tasks. This process often involves fine-tuning on user data data, which may contain sensitive information. Although not recommended, it is not uncommon for users to send passwords in messages, and fine-tuning models on this could result in passwords being leaked. In this study, a Large Language Model is fine-tuned with customer support data and passwords from the RockYou password wordlist using Low-Rank Adaptation (LoRA). Out of the first 200 passwords from the list, 37 were successfully recovered. Further, causal tracing is used to identify that password information is largely located in a few layers. Lastly, Rank One Model Editing (ROME) is used to remove the password information from the model, resulting in the number of passwords recovered going from 37 to 0. 

---
# Token-Driven GammaTune: Adaptive Calibration for Enchanced Speculative Decoding 

**Authors**: Aayush Gautam, Susav Shrestha, Narasimha Annapareddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.00030)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, selecting an optimal speculation length is critical for maximizing speedup while minimizing wasted computation. We introduce \textit{GammaTune} and \textit{GammaTune+}, training-free adaptive algorithms that dynamically adjust speculation length based on token acceptance rates using a heuristic-based switching mechanism. Evaluated on SpecBench across multiple tasks and model pairs, our method outperforms other heuristic-based approaches and fixed-length speculative decoding, achieving an average speedup of 15\% ($\pm$5\%) with \textit{GammaTune} and 16\% ($\pm$3\%) with \textit{GammaTune+}, while reducing performance variance. This makes \textit{GammaTune} a robust and efficient solution for real-world deployment. 

---
# Generating Structured Plan Representation of Procedures with LLMs 

**Authors**: Deepeka Garg, Sihan Zeng, Sumitra Ganesh, Leo Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2504.00029)  

**Abstract**: In this paper, we address the challenges of managing Standard Operating Procedures (SOPs), which often suffer from inconsistencies in language, format, and execution, leading to operational inefficiencies. Traditional process modeling demands significant manual effort, domain expertise, and familiarity with complex languages like Business Process Modeling Notation (BPMN), creating barriers for non-techincal users. We introduce SOP Structuring (SOPStruct), a novel approach that leverages Large Language Models (LLMs) to transform SOPs into decision-tree-based structured representations. SOPStruct produces a standardized representation of SOPs across different domains, reduces cognitive load, and improves user comprehension by effectively capturing task dependencies and ensuring sequential integrity. Our approach enables leveraging the structured information to automate workflows as well as empower the human users. By organizing procedures into logical graphs, SOPStruct facilitates backtracking and error correction, offering a scalable solution for process optimization. We employ a novel evaluation framework, combining deterministic methods with the Planning Domain Definition Language (PDDL) to verify graph soundness, and non-deterministic assessment by an LLM to ensure completeness. We empirically validate the robustness of our LLM-based structured SOP representation methodology across SOPs from different domains and varying levels of complexity. Despite the current lack of automation readiness in many organizations, our research highlights the transformative potential of LLMs to streamline process modeling, paving the way for future advancements in automated procedure optimization. 

---
# Opioid Named Entity Recognition (ONER-2025) from Reddit 

**Authors**: Muhammad Ahmad, Humaira Farid, Iqra Ameer, Muhammad Muzamil, Ameer Hamza Muhammad Jalal, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2504.00027)  

**Abstract**: The opioid overdose epidemic remains a critical public health crisis, particularly in the United States, leading to significant mortality and societal costs. Social media platforms like Reddit provide vast amounts of unstructured data that offer insights into public perceptions, discussions, and experiences related to opioid use. This study leverages Natural Language Processing (NLP), specifically Opioid Named Entity Recognition (ONER-2025), to extract actionable information from these platforms. Our research makes four key contributions. First, we created a unique, manually annotated dataset sourced from Reddit, where users share self-reported experiences of opioid use via different administration routes. This dataset contains 331,285 tokens and includes eight major opioid entity categories. Second, we detail our annotation process and guidelines while discussing the challenges of labeling the ONER-2025 dataset. Third, we analyze key linguistic challenges, including slang, ambiguity, fragmented sentences, and emotionally charged language, in opioid discussions. Fourth, we propose a real-time monitoring system to process streaming data from social media, healthcare records, and emergency services to identify overdose events. Using 5-fold cross-validation in 11 experiments, our system integrates machine learning, deep learning, and transformer-based language models with advanced contextual embeddings to enhance understanding. Our transformer-based models (bert-base-NER and roberta-base) achieved 97% accuracy and F1-score, outperforming baselines by 10.23% (RF=0.88). 

---
# Diffusion models applied to skin and oral cancer classification 

**Authors**: José J. M. Uliana, Renato A. Krohling  

**Link**: [PDF](https://arxiv.org/pdf/2504.00026)  

**Abstract**: This study investigates the application of diffusion models in medical image classification (DiffMIC), focusing on skin and oral lesions. Utilizing the datasets PAD-UFES-20 for skin cancer and P-NDB-UFES for oral cancer, the diffusion model demonstrated competitive performance compared to state-of-the-art deep learning models like Convolutional Neural Networks (CNNs) and Transformers. Specifically, for the PAD-UFES-20 dataset, the model achieved a balanced accuracy of 0.6457 for six-class classification and 0.8357 for binary classification (cancer vs. non-cancer). For the P-NDB-UFES dataset, it attained a balanced accuracy of 0.9050. These results suggest that diffusion models are viable models for classifying medical images of skin and oral lesions. In addition, we investigate the robustness of the model trained on PAD-UFES-20 for skin cancer but tested on the clinical images of the HIBA dataset. 

---
# A multi-locus predictiveness curve and its summary assessment for genetic risk prediction 

**Authors**: Changshuai Wei, Ming Li, Yalu Wen, Chengyin Ye, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00024)  

**Abstract**: With the advance of high-throughput genotyping and sequencing technologies, it becomes feasible to comprehensive evaluate the role of massive genetic predictors in disease prediction. There exists, therefore, a critical need for developing appropriate statistical measurements to access the combined effects of these genetic variants in disease prediction. Predictiveness curve is commonly used as a graphical tool to measure the predictive ability of a risk prediction model on a single continuous biomarker. Yet, for most complex diseases, risk prediciton models are formed on multiple genetic variants. We therefore propose a multi-marker predictiveness curve and provide a non-parametric method to construct the curve for case-control studies. We further introduce a global predictiveness U and a partial predictiveness U to summarize prediction curve across the whole population and sub-population of clinical interest, respectively. We also demonstrate the connections of predictiveness curve with ROC curve and Lorenz curve. Through simulation, we compared the performance of the predictiveness U to other three summary indices: R square, Total Gain, and Average Entropy, and showed that Predictiveness U outperformed the other three indexes in terms of unbiasedness and robustness. Moreover, we simulated a series of rare-variants disease model, found partial predictiveness U performed better than global predictiveness U. Finally, we conducted a real data analysis, using predictiveness curve and predictiveness U to evaluate a risk prediction model for Nicotine Dependence. 

---
# Celler:A Genomic Language Model for Long-Tailed Single-Cell Annotation 

**Authors**: Huan Zhao, Yiming Liu, Jina Yao, Ling Xiong, Zexin Zhou, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00020)  

**Abstract**: Recent breakthroughs in single-cell technology have ushered in unparalleled opportunities to decode the molecular intricacy of intricate biological systems, especially those linked to diseases unique to humans. However, these progressions have also ushered in novel obstacles-specifically, the efficient annotation of extensive, long-tailed single-cell data pertaining to disease conditions. To effectively surmount this challenge, we introduce Celler, a state-of-the-art generative pre-training model crafted specifically for the annotation of single-cell data. Celler incorporates two groundbreaking elements: First, we introduced the Gaussian Inflation (GInf) Loss function. By dynamically adjusting sample weights, GInf Loss significantly enhances the model's ability to learn from rare categories while reducing the risk of overfitting for common categories. Secondly, we introduce an innovative Hard Data Mining (HDM) strategy into the training process, specifically targeting the challenging-to-learn minority data samples, which significantly improved the model's predictive accuracy. Additionally, to further advance research in this field, we have constructed a large-scale single-cell dataset: Celler-75, which encompasses 40 million cells distributed across 80 human tissues and 75 specific diseases. This dataset provides critical support for comprehensively exploring the potential of single-cell technology in disease research. Our code is available at this https URL. 

---
# ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding 

**Authors**: Indraneil Paul, Haoyi Yang, Goran Glavaš, Kristian Kersting, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2504.00019)  

**Abstract**: Language models (LMs) have become a staple of the code-writing toolbox. Their pre-training recipe has, however, remained stagnant over recent years, barring the occasional changes in data sourcing and filtering strategies. In particular, research exploring modifications to Code-LMs' pre-training objectives, geared towards improving data efficiency and better disentangling between syntax and semantics, has been noticeably sparse, especially compared with corresponding efforts in natural language LMs. In this work, we examine grounding on obfuscated code as a means of helping Code-LMs look beyond the surface-form syntax and enhance their pre-training sample efficiency. To this end, we compile ObscuraX, a dataset of approximately 55M source and obfuscated code pairs in seven languages. Subsequently, we pre-train ObscuraCoder models, ranging in size from 255M to 2.8B parameters, on a 272B-token corpus that includes ObscuraX and demonstrate that our obfuscation-based pre-training recipe leads to consistent improvements in Code-LMs' abilities compared to both vanilla autoregressive pre-training as well as existing de-obfuscation (DOBF) objectives. ObscuraCoder demonstrates sizeable gains across multiple tests of syntactic and semantic code understanding, along with improved capabilities in multilingual code completion, multilingual code commit summarization, and multi-purpose library-oriented code generation. 

---
# Enhance Vision-based Tactile Sensors via Dynamic Illumination and Image Fusion 

**Authors**: Artemii Redkin, Zdravko Dugonjic, Mike Lambeta, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2504.00017)  

**Abstract**: Vision-based tactile sensors use structured light to measure deformation in their elastomeric interface. Until now, vision-based tactile sensors such as DIGIT and GelSight have been using a single, static pattern of structured light tuned to the specific form factor of the sensor. In this work, we investigate the effectiveness of dynamic illumination patterns, in conjunction with image fusion techniques, to improve the quality of sensing of vision-based tactile sensors. Specifically, we propose to capture multiple measurements, each with a different illumination pattern, and then fuse them together to obtain a single, higher-quality measurement. Experimental results demonstrate that this type of dynamic illumination yields significant improvements in image contrast, sharpness, and background difference. This discovery opens the possibility of retroactively improving the sensing quality of existing vision-based tactile sensors with a simple software update, and for new hardware designs capable of fully exploiting dynamic illumination. 

---
# Deep Learning-Based Hypoglycemia Classification Across Multiple Prediction Horizons 

**Authors**: Beyza Cinar, Jennifer Daniel Onwuchekwa, Maria Maleshkova  

**Link**: [PDF](https://arxiv.org/pdf/2504.00009)  

**Abstract**: Type 1 diabetes (T1D) management can be significantly enhanced through the use of predictive machine learning (ML) algorithms, which can mitigate the risk of adverse events like hypoglycemia. Hypoglycemia, characterized by blood glucose levels below 70 mg/dL, is a life-threatening condition typically caused by excessive insulin administration, missed meals, or physical activity. Its asymptomatic nature impedes timely intervention, making ML models crucial for early detection. This study integrates short- (up to 2h) and long-term (up to 24h) prediction horizons (PHs) within a single classification model to enhance decision support. The predicted times are 5-15 min, 15-30 min, 30 min-1h, 1-2h, 2-4h, 4-8h, 8-12h, and 12-24h before hypoglycemia. In addition, a simplified model classifying up to 4h before hypoglycemia is compared. We trained ResNet and LSTM models on glucose levels, insulin doses, and acceleration data. The results demonstrate the superiority of the LSTM models when classifying nine classes. In particular, subject-specific models yielded better performance but achieved high recall only for classes 0, 1, and 2 with 98%, 72%, and 50%, respectively. A population-based six-class model improved the results with at least 60% of events detected. In contrast, longer PHs remain challenging with the current approach and may be considered with different models. 

---
# Tensor Generalized Approximate Message Passing 

**Authors**: Yinchuan Li, Guangchen Lan, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00008)  

**Abstract**: We propose a tensor generalized approximate message passing (TeG-AMP) algorithm for low-rank tensor inference, which can be used to solve tensor completion and decomposition problems. We derive TeG-AMP algorithm as an approximation of the sum-product belief propagation algorithm in high dimensions where the central limit theorem and Taylor series approximations are applicable. As TeG-AMP is developed based on a general TR decomposition model, it can be directly applied to many low-rank tensor types. Moreover, our TeG-AMP can be simplified based on the CP decomposition model and a tensor simplified AMP is proposed for low CP-rank tensor inference problems. Experimental results demonstrate that the proposed methods significantly improve recovery performances since it takes full advantage of tensor structures. 

---
# Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices 

**Authors**: Xiao Yan, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.00002)  

**Abstract**: Recent advancements in large language models (LLMs) have prompted interest in deploying these models on mobile devices to enable new applications without relying on cloud connectivity. However, the efficiency constraints of deploying LLMs on resource-limited devices present significant challenges. In this paper, we conduct a comprehensive measurement study to evaluate the efficiency tradeoffs between mobile-based, edge-based, and cloud-based deployments for LLM applications. We implement AutoLife-Lite, a simplified LLM-based application that analyzes smartphone sensor data to infer user location and activity contexts. Our experiments reveal that: (1) Only small-size LLMs (<4B parameters) can run successfully on powerful mobile devices, though they exhibit quality limitations compared to larger models; (2) Model compression is effective in lower the hardware requirement, but may lead to significant performance degradation; (3) The latency to run LLMs on mobile devices with meaningful output is significant (>30 seconds), while cloud services demonstrate better time efficiency (<10 seconds); (4) Edge deployments offer intermediate tradeoffs between latency and model capabilities, with different results on CPU-based and GPU-based settings. These findings provide valuable insights for system designers on the current limitations and future directions for on-device LLM applications. 

---
