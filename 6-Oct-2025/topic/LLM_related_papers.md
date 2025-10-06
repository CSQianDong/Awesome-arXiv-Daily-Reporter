# Grounding Large Language Models in Clinical Evidence: A Retrieval-Augmented Generation System for Querying UK NICE Clinical Guidelines 

**Authors**: Matthew Lewis, Samuel Thio, Richard JB Dobson, Spiros Denaxas  

**Link**: [PDF](https://arxiv.org/pdf/2510.02967)  

**Abstract**: This paper presents the development and evaluation of a Retrieval-Augmented Generation (RAG) system for querying the United Kingdom's National Institute for Health and Care Excellence (NICE) clinical guidelines using Large Language Models (LLMs). The extensive length and volume of these guidelines can impede their utilisation within a time-constrained healthcare system, a challenge this project addresses through the creation of a system capable of providing users with precisely matched information in response to natural language queries. The system's retrieval architecture, composed of a hybrid embedding mechanism, was evaluated against a database of 10,195 text chunks derived from three hundred guidelines. It demonstrates high performance, with a Mean Reciprocal Rank (MRR) of 0.814, a Recall of 81% at the first chunk and of 99.1% within the top ten retrieved chunks, when evaluated on 7901 queries.
The most significant impact of the RAG system was observed during the generation phase. When evaluated on a manually curated dataset of seventy question-answer pairs, RAG-enhanced models showed substantial gains in performance. Faithfulness, the measure of whether an answer is supported by the source text, was increased by 64.7 percentage points to 99.5% for the RAG-enhanced O4-Mini model and significantly outperformed the medical-focused Meditron3-8B LLM, which scored 43%. This, combined with a perfect Context Precision score of 1 for all RAG-enhanced models, confirms the system's ability to prevent information fabrication by grounding its answers in relevant source material. This study thus establishes RAG as an effective, reliable, and scalable approach for applying generative AI in healthcare, enabling cost-effective access to medical guidelines. 

---
# Less LLM, More Documents: Searching for Improved RAG 

**Authors**: Jingjie Ning, Yibo Kong, Yunfan Long, Jamie Callan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02657)  

**Abstract**: Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators improves accuracy, it also raises cost and limits deployability. We explore an orthogonal axis: enlarging the retriever's corpus to reduce reliance on large LLMs. Experimental results show that corpus scaling consistently strengthens RAG and can often serve as a substitute for increasing model size, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and large models benefit less. Our analysis shows that improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. These findings establish a principled corpus-generator trade-off: investing in larger corpora offers an effective path to stronger RAG, often comparable to enlarging the LLM itself. 

---
# A Simple but Effective Elaborative Query Reformulation Approach for Natural Language Recommendation 

**Authors**: Qianfeng Wen, Yifan Liu, Justin Cui, Joshua Zhang, Anton Korikov, George-Kirollos Saad, Scott Sanner  

**Link**: [PDF](https://arxiv.org/pdf/2510.02656)  

**Abstract**: Natural Language (NL) recommender systems aim to retrieve relevant items from free-form user queries and item descriptions. Existing systems often rely on dense retrieval (DR), which struggles to interpret challenging queries that express broad (e.g., "cities for youth friendly activities") or indirect (e.g., "cities for a high school graduation trip") user intents. While query reformulation (QR) has been widely adopted to improve such systems, existing QR methods tend to focus only on expanding the range of query subtopics (breadth) or elaborating on the potential meaning of a query (depth), but not both. In this paper, we propose EQR (Elaborative Subtopic Query Reformulation), a large language model-based QR method that combines both breadth and depth by generating potential query subtopics with information-rich elaborations. We also introduce three new natural language recommendation benchmarks in travel, hotel, and restaurant domains to establish evaluation of NL recommendation with challenging queries. Experiments show EQR substantially outperforms state-of-the-art QR methods in various evaluation metrics, highlighting that a simple yet effective QR approach can significantly improve NL recommender systems for queries with broad and indirect user intents. 

---
# Geolog-IA: Conversational System for Academic Theses 

**Authors**: Micaela Fuel Pozo, Andrea Guatumillo Saltos, Yeseña Tipan Llumiquinga, Kelly Lascano Aguirre, Marilyn Castillo Jara, Christian Mejia-Escobar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02653)  

**Abstract**: This study presents the development of Geolog-IA, a novel conversational system based on artificial intelligence that responds naturally to questions about geology theses from the Central University of Ecuador. Our proposal uses the Llama 3.1 and Gemini 2.5 language models, which are complemented by a Retrieval Augmented Generation (RAG) architecture and an SQLite database. This strategy allows us to overcome problems such as hallucinations and outdated knowledge. The evaluation of Geolog-IA's performance with the BLEU metric reaches an average of 0.87, indicating high consistency and accuracy in the responses generated. The system offers an intuitive, web-based interface that facilitates interaction and information retrieval for directors, teachers, students, and administrative staff at the institution. This tool can be a key support in education, training, and research and establishes a basis for future applications in other disciplines. 

---
# CoDA: Agentic Systems for Collaborative Data Visualization 

**Authors**: Zichen Chen, Jiefeng Chen, Sercan Ö. Arik, Misha Sra, Tomas Pfister, Jinsung Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03194)  

**Abstract**: Deep research has revolutionized data analysis, yet data scientists still devote substantial time to manually crafting visualizations, highlighting the need for robust automation from natural language queries. However, current systems struggle with complex datasets containing multiple files and iterative refinement. Existing approaches, including simple single- or multi-agent systems, often oversimplify the task, focusing on initial query parsing while failing to robustly manage data complexity, code errors, or final visualization quality. In this paper, we reframe this challenge as a collaborative multi-agent problem. We introduce CoDA, a multi-agent system that employs specialized LLM agents for metadata analysis, task planning, code generation, and self-reflection. We formalize this pipeline, demonstrating how metadata-focused analysis bypasses token limits and quality-driven refinement ensures robustness. Extensive evaluations show CoDA achieves substantial gains in the overall score, outperforming competitive baselines by up to 41.5%. This work demonstrates that the future of visualization automation lies not in isolated code generation but in integrated, collaborative agentic workflows. 

---
# Reward Model Routing in Alignment 

**Authors**: Xinle Wu, Yao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02850)  

**Abstract**: Reinforcement learning from human or AI feedback (RLHF / RLAIF) has become the standard paradigm for aligning large language models (LLMs). However, most pipelines rely on a single reward model (RM), limiting alignment quality and risking overfitting. Recent work explores RM routing--dynamically selecting an RM from a candidate pool to exploit complementary strengths while maintaining $O(1)$ RM calls--but existing methods suffer from cold-start and insufficient exploration. We propose BayesianRouter, a hybrid routing framework that combines offline RM strengths learning with online Bayesian selection. In the offline stage, a multi-task router is trained on preference data to estimate per-RM reliability. In the online stage, a Bayesian Thompson sampling router performs per-query RM selection, initializing RM-specific weight vectors with offline embeddings as Gaussian priors and adaptively updating their posteriors with online rewards to adapt to the evolving policy distribution. Extensive experiments on instruction-following (AlpacaEval-2, Arena-Hard, MT-Bench) and reasoning (GSM8K, MMLU) benchmarks show that BayesianRouter consistently outperforms individual RMs, RM ensembling, and existing routing methods. 

---
# Improving Cooperation in Collaborative Embodied AI 

**Authors**: Hima Jacob Leven Suprabha, Laxmi Nag Laxminarayan Nagesh, Ajith Nair, Alvin Reuben Amal Selvaster, Ayan Khan, Raghuram Damarla, Sanju Hannah Samuel, Sreenithi Saravana Perumal, Titouan Puech, Venkataramireddy Marella, Vishal Sonar, Alessandro Suglia, Oliver Lemon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03153)  

**Abstract**: The integration of Large Language Models (LLMs) into multiagent systems has opened new possibilities for collaborative reasoning and cooperation with AI agents. This paper explores different prompting methods and evaluates their effectiveness in enhancing agent collaborative behaviour and decision-making. We enhance CoELA, a framework designed for building Collaborative Embodied Agents that leverage LLMs for multi-agent communication, reasoning, and task coordination in shared virtual spaces. Through systematic experimentation, we examine different LLMs and prompt engineering strategies to identify optimised combinations that maximise collaboration performance. Furthermore, we extend our research by integrating speech capabilities, enabling seamless collaborative voice-based interactions. Our findings highlight the effectiveness of prompt optimisation in enhancing collaborative agent performance; for example, our best combination improved the efficiency of the system running with Gemma3 by 22% compared to the original CoELA system. In addition, the speech integration provides a more engaging user interface for iterative system development and demonstrations. 

---
# NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning 

**Authors**: Yulong Zhang, Li Wang, Wei Du, Peilin Li, Yuqin Dai Zhiyuan Zhao, Lingyong Fang, Ziniu Liu, Ru Zhang, Huijia Zhu, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02816)  

**Abstract**: Verifying multi-step reasoning in large language models is difficult due to imprecise error localization and high token costs. Existing methods either assess entire reasoning chains, suffering attention dilution, or rely on expensive multi-sampling. We introduce Node-wise Consistency Verification (NCV), a training-free framework that recasts verification as lightweight binary consistency checks at the node level. By decomposing the chain of thought into interconnected verification nodes, NCV precisely localizes errors and avoids unnecessary long-form generation. Experiments demonstrate that our approach enhances interpretability and efficiency, presenting a scalable solution for reliable LLM reasoning verification. On public datasets, NCV achieves a 10\% to 25\% improvement in F1 scores over baselines while utilizing $6\times$~$58\times$ fewer tokens than traditional methods like CoT-based verifiers. 

---
# Beyond the Final Answer: Evaluating the Reasoning Trajectories of Tool-Augmented Agents 

**Authors**: Wonjoong Kim, Sangwu Park, Yeonjun In, Sein Kim, Dongha Lee, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.02837)  

**Abstract**: Although recent tool-augmented benchmarks incorporate complex user requests and diverse tools, the evaluation methods for most of them remain limited to answer matching. However, as the number of steps required to resolve a user request increases, a proper evaluation of an agent's performance must go beyond the final answer to also assess the problem-solving trajectory, including previously ignored aspects such as efficiency, hallucination, and adaptivity. The most straightforward method for evaluating these aspects is to compare an agent's trajectory with the ground-truth trajectory, but this approach is fundamentally limited since annotating all valid ground-truth trajectories is prohibitively expensive. However, a simple LLM-based evaluator struggles to assess trajectories in detail without ground truth. To effectively evaluate the agents in this manner, we introduce TRACE, a framework for the multi-dimensional evaluation of tool-augmented LLM agent performance. By incorporating an evidence bank, which accumulates knowledge gathered from preceding reasoning steps, TRACE enables a multi-faceted analysis and evaluation of an agent's reasoning trajectory effectively. To validate our framework, we develop a new meta-evaluation dataset by augmenting existing benchmarks with diverse and flawed trajectories, each labeled with multi-faceted performance scores. Our results confirm that TRACE accurately evaluates these complex behaviors in a scalable and cost-effective manner, even with small open-source LLMs. Furthermore, we apply our method to evaluate the trajectories that agents produce while solving tool-augmented tasks, presenting previously unreported observations and their corresponding insights. 

---
# Multimodal Large Language Model Framework for Safe and Interpretable Grid-Integrated EVs 

**Authors**: Jean Douglas Carvalho, Hugo Kenji, Ahmad Mohammad Saber, Glaucia Melo, Max Mauro Dias Santos, Deepa Kundur  

**Link**: [PDF](https://arxiv.org/pdf/2510.02592)  

**Abstract**: The integration of electric vehicles (EVs) into smart grids presents unique opportunities to enhance both transportation systems and energy networks. However, ensuring safe and interpretable interactions between drivers, vehicles, and the surrounding environment remains a critical challenge. This paper presents a multi-modal large language model (LLM)-based framework to process multimodal sensor data - such as object detection, semantic segmentation, and vehicular telemetry - and generate natural-language alerts for drivers. The framework is validated using real-world data collected from instrumented vehicles driving on urban roads, ensuring its applicability to real-world scenarios. By combining visual perception (YOLOv8), geocoded positioning, and CAN bus telemetry, the framework bridges raw sensor data and driver comprehension, enabling safer and more informed decision-making in urban driving scenarios. Case studies using real data demonstrate the framework's effectiveness in generating context-aware alerts for critical situations, such as proximity to pedestrians, cyclists, and other vehicles. This paper highlights the potential of LLMs as assistive tools in e-mobility, benefiting both transportation systems and electric networks by enabling scalable fleet coordination, EV load forecasting, and traffic-aware energy planning.
Index Terms - Electric vehicles, visual perception, large language models, YOLOv8, semantic segmentation, CAN bus, prompt engineering, smart grid. 

---
# Automated Constraint Specification for Job Scheduling by Regulating Generative Model with Domain-Specific Representation 

**Authors**: Yu-Zhe Shi, Qiao Xu, Yanjia Li, Mingchen Liu, Huamin Qu, Lecheng Ruan, Qining Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02679)  

**Abstract**: Advanced Planning and Scheduling (APS) systems have become indispensable for modern manufacturing operations, enabling optimized resource allocation and production efficiency in increasingly complex and dynamic environments. While algorithms for solving abstracted scheduling problems have been extensively investigated, the critical prerequisite of specifying manufacturing requirements into formal constraints remains manual and labor-intensive. Although recent advances of generative models, particularly Large Language Models (LLMs), show promise in automating constraint specification from heterogeneous raw manufacturing data, their direct application faces challenges due to natural language ambiguity, non-deterministic outputs, and limited domain-specific knowledge. This paper presents a constraint-centric architecture that regulates LLMs to perform reliable automated constraint specification for production scheduling. The architecture defines a hierarchical structural space organized across three levels, implemented through domain-specific representation to ensure precision and reliability while maintaining flexibility. Furthermore, an automated production scenario adaptation algorithm is designed and deployed to efficiently customize the architecture for specific manufacturing configurations. Experimental results demonstrate that the proposed approach successfully balances the generative capabilities of LLMs with the reliability requirements of manufacturing systems, significantly outperforming pure LLM-based approaches in constraint specification tasks. 

---
# Multimodal Function Vectors for Spatial Relations 

**Authors**: Shuhao Fu, Esther Goldberg, Ying Nian Wu, Hongjing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02528)  

**Abstract**: Large Multimodal Models (LMMs) demonstrate impressive in-context learning abilities from limited multimodal demonstrations, yet the internal mechanisms supporting such task learning remain opaque. Building on prior work of large language models, we show that a small subset of attention heads in the vision-language model OpenFlamingo-4B is responsible for transmitting representations of spatial relations. The activations of these attention heads, termed function vectors, can be extracted and manipulated to alter an LMM's performance on relational tasks. First, using both synthetic and real image datasets, we apply causal mediation analysis to identify attention heads that strongly influence relational predictions, and extract multimodal function vectors that improve zero-shot accuracy at inference time. We further demonstrate that these multimodal function vectors can be fine-tuned with a modest amount of training data, while keeping LMM parameters frozen, to significantly outperform in-context learning baselines. Finally, we show that relation-specific function vectors can be linearly combined to solve analogy problems involving novel and untrained spatial relations, highlighting the strong generalization ability of this approach. Our results show that LMMs encode spatial relational knowledge within localized internal structures, which can be systematically extracted and optimized, thereby advancing our understanding of model modularity and enhancing control over relational reasoning in LMMs. 

---
# Safe and Efficient In-Context Learning via Risk Control 

**Authors**: Andrea Wynn, Metod Jazbec, Charith Peris, Rinat Khaziev, Anqi Liu, Daniel Khashabi, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2510.02480)  

**Abstract**: Large language models (LLMs) demonstrate a remarkable ability to learn new tasks from a few in-context examples. However, this flexibility introduces safety concerns: LLMs can be influenced by incorrect or malicious demonstrations -- for example, if an adversary tampers with or injects harmful examples without a human supervisor noticing. This motivates principled designs in which the system itself includes built-in mechanisms to guard against such attacks. We propose a novel approach to limit the degree to which harmful demonstrations can degrade model performance. First, we define a baseline ``safe'' behavior for the model -- the model's performance given no in-context demonstrations (zero-shot). Next, we apply distribution-free risk control (DFRC) to control the extent to which in-context samples can decay performance below zero-shot. We achieve this by leveraging dynamic early exit prediction, ignoring later attention heads that attend the most to the unsafe inputs. Finally, we propose modifications to DFRC that allow it to both control risk for harmful inputs \textit{and} leverage performance and efficiency gains on helpful inputs. We present both theoretical and empirical results showing that our approach can effectively control risk for harmful in-context demonstrations while simultaneously achieving substantial computational efficiency gains with helpful demonstrations. 

---
# BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks 

**Authors**: Sagnik Anupam, Davis Brown, Shuo Li, Eric Wong, Hamed Hassani, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2510.02418)  

**Abstract**: LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about captcha resolution. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale. 

---
# Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignment 

**Authors**: Hongxiang Zhang, Yuan Tian, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03223)  

**Abstract**: To solve complex reasoning tasks for Large Language Models (LLMs), prompting-based methods offer a lightweight alternative to fine-tuning and reinforcement learning. However, as reasoning chains extend, critical intermediate steps and the original prompt will be buried in the context, receiving insufficient attention and leading to errors. In this paper, we propose Self-Anchor, a novel pipeline that leverages the inherent structure of reasoning to steer LLM attention. Self-Anchor decomposes reasoning trajectories into structured plans and automatically aligns the model's attention to the most relevant inference steps, allowing the model to maintain focus throughout generation. Our experiment shows that Self-Anchor outperforms SOTA prompting methods across six benchmarks. Notably, Self-Anchor significantly reduces the performance gap between ``non-reasoning'' models and specialized reasoning models, with the potential to enable most LLMs to tackle complex reasoning tasks without retraining. 

---
# Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair 

**Authors**: José Cambronero, Michele Tufano, Sherry Shi, Renyao Wei, Grant Uy, Runxiang Cheng, Chin-Jung Liu, Shiying Pan, Satish Chandra, Pat Rondon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03217)  

**Abstract**: Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately agent-generated patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing unlikely patches to developers can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patch trajectories rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems. 

---
# On the Role of Temperature Sampling in Test-Time Scaling 

**Authors**: Yuheng Wu, Azalia Mirhoseini, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2510.02611)  

**Abstract**: Large language models (LLMs) can improve reasoning at inference time through test-time scaling (TTS), where multiple reasoning traces are generated and the best one is selected. Prior work shows that increasing the number of samples K steadily improves accuracy. In this paper, we demonstrate that this trend does not hold indefinitely: at large K, further scaling yields no gains, and certain hard questions remain unsolved regardless of the number of traces. Interestingly, we find that different sampling temperatures solve different subsets of problems, implying that single-temperature scaling explores only part of a model's potential. We therefore propose scaling along the temperature dimension, which enlarges the reasoning boundary of LLMs. Averaged over Qwen3 (0.6B, 1.7B, 4B, 8B) and five representative reasoning benchmarks (AIME 2024/2025, MATH500, LiveCodeBench, Hi-ToM), temperature scaling yields an additional 7.3 points over single-temperature TTS. Temperature scaling also enables base models to reach performance comparable to reinforcement learning (RL)-trained counterparts, without additional post-training. We further provide a comprehensive analysis of this phenomenon and design a multi-temperature voting method that reduces the overhead of temperature scaling. Overall, our findings suggest that TTS is more powerful than previously thought, and that temperature scaling offers a simple and effective way to unlock the latent potential of base models. 

---
# Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting? 

**Authors**: Xuan Xu, Haolun Li, Zhongliang Yang, Beilin Chu, Jia Song, Moxuan Xu, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.03174)  

**Abstract**: Traditional topic models such as neural topic models rely on inference and generation networks to learn latent topic distributions. This paper explores a new paradigm for topic modeling in the era of large language models, framing TM as a long-form generation task whose definition is updated in this paradigm. We propose a simple but practical approach to implement LLM-based topic model tasks out of the box (sample a data subset, generate topics and representative text with our prompt, text assignment with keyword match). We then investigate whether the long-form generation paradigm can beat NTMs via zero-shot prompting. We conduct a systematic comparison between NTMs and LLMs in terms of topic quality and empirically examine the claim that "a majority of NTMs are outdated." 

---
# Agentic Additive Manufacturing Alloy Discovery 

**Authors**: Peter Pak, Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2510.02567)  

**Abstract**: Agentic systems enable the intelligent use of research tooling, augmenting a researcher's ability to investigate and propose novel solutions to existing problems. Within Additive Manufacturing (AM), alloy discovery remains a complex challenge, often requiring expertise in the various domains of materials science, thermodynamic simulations, and experimental analysis. Large Language Model (LLM) enabled agents can facilitate this endeavor by utilizing their extensive knowledge base to dispatch tool calls via Model Context Protocol (MCP) to perform actions such as Thermo-Calc property diagram calculations and lack of fusion process map generation. In addition, the multi-agent system developed in this work is able to effectively reason through complex user prompts and provide analysis on the printability of proposed alloys. These agents can dynamically adjust their task trajectory to the outcomes of tool call results, effectively enabling autonomous decision-making in practical environments. This work aims to utilize LLM enabled agents to automate and accelerate the task of alloy discovery within the field of additive manufacturing and showcase the benefits of adopting this multi-agent system. 

---
# Untargeted Jailbreak Attack 

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2510.02999)  

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.
To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM this http URL evaluations demonstrate that \textsc{UJA} can achieve over 80\% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%. 

---
# Evaluating Large Language Models for IUCN Red List Species Information 

**Authors**: Shinya Uryu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02830)  

**Abstract**: Large Language Models (LLMs) are rapidly being adopted in conservation to address the biodiversity crisis, yet their reliability for species evaluation is uncertain. This study systematically validates five leading models on 21,955 species across four core IUCN Red List assessment components: taxonomy, conservation status, distribution, and threats. A critical paradox was revealed: models excelled at taxonomic classification (94.9%) but consistently failed at conservation reasoning (27.2% for status assessment). This knowledge-reasoning gap, evident across all models, suggests inherent architectural constraints, not just data limitations. Furthermore, models exhibited systematic biases favoring charismatic vertebrates, potentially amplifying existing conservation inequities. These findings delineate clear boundaries for responsible LLM deployment: they are powerful tools for information retrieval but require human oversight for judgment-based decisions. A hybrid approach is recommended, where LLMs augment expert capacity while human experts retain sole authority over risk assessment and policy. 

---
# Prototyping Digital Social Spaces through Metaphor-Driven Design: Translating Spatial Concepts into an Interactive Social Simulation 

**Authors**: Yoojin Hong, Martina Di Paola, Braahmi Padmakumar, Hwi Joon Lee, Mahnoor Shafiq, Joseph Seering  

**Link**: [PDF](https://arxiv.org/pdf/2510.02759)  

**Abstract**: Social media platforms are central to communication, yet their designs remain narrowly focused on engagement and scale. While researchers have proposed alternative visions for online spaces, these ideas are difficult to prototype within platform constraints. In this paper, we introduce a metaphor-driven system to help users imagine and explore new social media environments. The system translates users' metaphors into structured sets of platform features and generates interactive simulations populated with LLM-driven agents. To evaluate this approach, we conducted a study where participants created and interacted with simulated social media spaces. Our findings show that metaphors allow users to express distinct social expectations, and that perceived authenticity of the simulation depended on how well it captured dynamics like intimacy, participation, and temporal engagement. We conclude by discussing how metaphor-driven simulation can be a powerful design tool for prototyping alternative social architectures and expanding the design space for future social platforms. 

---
# TravelBench : Exploring LLM Performance in Low-Resource Domains 

**Authors**: Srinivas Billa, Xiaonan Jing  

**Link**: [PDF](https://arxiv.org/pdf/2510.02719)  

**Abstract**: Results on existing LLM benchmarks capture little information over the model capabilities in low-resource tasks, making it difficult to develop effective solutions in these domains. To address these challenges, we curated 14 travel-domain datasets spanning 7 common NLP tasks using anonymised data from real-world scenarios, and analysed the performance across LLMs. We report on the accuracy, scaling behaviour, and reasoning capabilities of LLMs in a variety of tasks. Our results confirm that general benchmarking results are insufficient for understanding model performance in low-resource tasks. Despite the amount of training FLOPs, out-of-the-box LLMs hit performance bottlenecks in complex, domain-specific scenarios. Furthermore, reasoning provides a more significant boost for smaller LLMs by making the model a better judge on certain tasks. 

---
# Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks 

**Authors**: Yubo Li, Ramayya Krishnan, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2510.02712)  

**Abstract**: Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present the first comprehensive survival analysis of conversational AI robustness, analyzing 36,951 conversation turns across 9 state-of-the-art LLMs to model failure as a time-to-event process. Our survival modeling framework-employing Cox proportional hazards, Accelerated Failure Time, and Random Survival Forest approaches-reveals extraordinary temporal dynamics. We find that abrupt, prompt-to-prompt(P2P) semantic drift is catastrophic, dramatically increasing the hazard of conversational failure. In stark contrast, gradual, cumulative drift is highly protective, vastly reducing the failure hazard and enabling significantly longer dialogues. AFT models with interactions demonstrate superior performance, achieving excellent discrimination and exceptional calibration. These findings establish survival analysis as a powerful paradigm for evaluating LLM robustness, offer concrete insights for designing resilient conversational agents, and challenge prevailing assumptions about the necessity of semantic consistency in conversational AI Systems. 

---
# Multimodal Carotid Risk Stratification with Large Vision-Language Models: Benchmarking, Fine-Tuning, and Clinical Insights 

**Authors**: Daphne Tsolissou, Theofanis Ganitidis, Konstantinos Mitsis, Stergios CHristodoulidis, Maria Vakalopoulou, Konstantina Nikita  

**Link**: [PDF](https://arxiv.org/pdf/2510.02922)  

**Abstract**: Reliable risk assessment for carotid atheromatous disease remains a major clinical challenge, as it requires integrating diverse clinical and imaging information in a manner that is transparent and interpretable to clinicians. This study investigates the potential of state-of-the-art and recent large vision-language models (LVLMs) for multimodal carotid plaque assessment by integrating ultrasound imaging (USI) with structured clinical, demographic, laboratory, and protein biomarker data. A framework that simulates realistic diagnostic scenarios through interview-style question sequences is proposed, comparing a range of open-source LVLMs, including both general-purpose and medically tuned models. Zero-shot experiments reveal that even if they are very powerful, not all LVLMs can accurately identify imaging modality and anatomy, while all of them perform poorly in accurate risk classification. To address this limitation, LLaVa-NeXT-Vicuna is adapted to the ultrasound domain using low-rank adaptation (LoRA), resulting in substantial improvements in stroke risk stratification. The integration of multimodal tabular data in the form of text further enhances specificity and balanced accuracy, yielding competitive performance compared to prior convolutional neural network (CNN) baselines trained on the same dataset. Our findings highlight both the promise and limitations of LVLMs in ultrasound-based cardiovascular risk prediction, underscoring the importance of multimodal integration, model calibration, and domain adaptation for clinical translation. 

---
# HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inference 

**Authors**: Shubham Negi, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2510.02675)  

**Abstract**: The rapid adoption of Large Language Models (LLMs) has driven a growing demand for efficient inference, particularly in latency-sensitive applications such as chatbots and personalized assistants. Unlike traditional deep neural networks, LLM inference proceeds in two distinct phases: the prefill phase, which processes the full input sequence in parallel, and the decode phase, which generates tokens sequentially. These phases exhibit highly diverse compute and memory requirements, which makes accelerator design particularly challenging. Prior works have primarily been optimized for high-batch inference or evaluated only short input context lengths, leaving the low-batch and long context regime, which is critical for interactive applications, largely underexplored.
We propose HALO, a heterogeneous memory centric accelerator designed for these unique challenges of prefill and decode phases in low-batch LLM inference. HALO integrates HBM based Compute-in-DRAM (CiD) with an on-chip analog Compute-in-Memory (CiM), co-packaged using 2.5D integration. To further improve the hardware utilization, we introduce a phase-aware mapping strategy that adapts to the distinct demands of the prefill and decode phases. Compute bound operations in the prefill phase are mapped to CiM to exploit its high throughput matrix multiplication capability, while memory-bound operations in the decode phase are executed on CiD to benefit from reduced data movement within DRAM. Additionally, we present an analysis of the performance tradeoffs of LLMs under two architectural extremes: a fully CiD and a fully on-chip analog CiM design to highlight the need for a heterogeneous design. We evaluate HALO on LLaMA-2 7B and Qwen3 8B models. Our experimental results show that LLMs mapped to HALO achieve up to 18x geometric mean speedup over AttAcc, an attention-optimized mapping and 2.5x over CENT, a fully CiD based mapping. 

---
# TutorBench: A Benchmark To Assess Tutoring Capabilities Of Large Language Models 

**Authors**: Rakshith S Srinivasa, Zora Che, Chen Bo Calvin Zhang, Diego Mares, Ernesto Hernandez, Jayeon Park, Dean Lee, Guillermo Mangialardi, Charmaine Ng, Ed-Yeremai Hernandez Cardona, Anisha Gunjal, Yunzhong He, Bing Liu, Chen Xing  

**Link**: [PDF](https://arxiv.org/pdf/2510.02663)  

**Abstract**: As students increasingly adopt large language models (LLMs) as learning aids, it is crucial to build models that are adept at handling the nuances of tutoring: they need to identify the core needs of students, be adaptive, provide personalized guidance, and be accurate. To this end, we introduce TutorBench, a dataset and evaluation benchmark designed to rigorously evaluate the core tutoring skills of LLMs. The dataset comprises 1,490 samples curated by human experts, focused on high-school and AP-level curricula. The samples are drawn from three common tutoring tasks: (i) generating adaptive explanations tailored to a student's confusion, (ii) providing actionable feedback on a student's work, and (iii) promoting active learning through effective hint generation. To account for the inherent complexity of tutoring, samples are accompanied by sample-specific rubrics which are used to judge model responses during evaluation. TutorBench uses a reliable and fine-grained automatic evaluation method that uses an LLM-judge and the sample-specific rubrics. We evaluate 16 frontier LLMs on TutorBench and present a detailed analysis of their performance and behavior. Our results show that none of the frontier LLMs achieve a score of greater than $56\%$, showing a large room for improvement. We find that LLMs fall short in exhibiting the full range of tutoring skills needed to guide, diagnose, and support students effectively, with all the frontier models achieving less than a $60\%$ pass rate on rubric criteria related to these skills. We also find that different model families exhibit varied strengths and limitations: the Claude models outperform others in supporting active learning, while they lag behind in the other two use cases. By releasing TutorBench, we provide a comprehensive and unsaturated benchmark to guide the development of the next-generation of AI tutors. 

---
# How Confident are Video Models? Empowering Video Models to Express their Uncertainty 

**Authors**: Zhiting Mei, Ola Shorinwa, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02571)  

**Abstract**: Generative video models demonstrate impressive text-to-video capabilities, spurring widespread adoption in many real-world applications. However, like large language models (LLMs), video generation models tend to hallucinate, producing plausible videos even when they are factually wrong. Although uncertainty quantification (UQ) of LLMs has been extensively studied in prior work, no UQ method for video models exists, raising critical safety concerns. To our knowledge, this paper represents the first work towards quantifying the uncertainty of video models. We present a framework for uncertainty quantification of generative video models, consisting of: (i) a metric for evaluating the calibration of video models based on robust rank correlation estimation with no stringent modeling assumptions; (ii) a black-box UQ method for video models (termed S-QUBED), which leverages latent modeling to rigorously decompose predictive uncertainty into its aleatoric and epistemic components; and (iii) a UQ dataset to facilitate benchmarking calibration in video models. By conditioning the generation task in the latent space, we disentangle uncertainty arising due to vague task specifications from that arising from lack of knowledge. Through extensive experiments on benchmark video datasets, we demonstrate that S-QUBED computes calibrated total uncertainty estimates that are negatively correlated with the task accuracy and effectively computes the aleatoric and epistemic constituents. 

---
# A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps 

**Authors**: Junlin Zeng, Xin Zhang, Xiang Zhao, Yan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02716)  

**Abstract**: Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.} 

---
# Knowledge-Graph Based RAG System Evaluation Framework 

**Authors**: Sicheng Dong, Vahid Zolfaghari, Nenad Petrovic, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2510.02549)  

**Abstract**: Large language models (LLMs) has become a significant research focus and is utilized in various fields, such as text generation and dialog systems. One of the most essential applications of LLM is Retrieval Augmented Generation (RAG), which greatly enhances generated content's reliability and relevance. However, evaluating RAG systems remains a challenging task. Traditional evaluation metrics struggle to effectively capture the key features of modern LLM-generated content that often exhibits high fluency and naturalness. Inspired by the RAGAS tool, a well-known RAG evaluation framework, we extended this framework into a KG-based evaluation paradigm, enabling multi-hop reasoning and semantic community clustering to derive more comprehensive scoring metrics. By incorporating these comprehensive evaluation criteria, we gain a deeper understanding of RAG systems and a more nuanced perspective on their performance. To validate the effectiveness of our approach, we compare its performance with RAGAS scores and construct a human-annotated subset to assess the correlation between human judgments and automated metrics. In addition, we conduct targeted experiments to demonstrate that our KG-based evaluation method is more sensitive to subtle semantic differences in generated outputs. Finally, we discuss the key challenges in evaluating RAG systems and highlight potential directions for future research. 

---
# Automatic Building Code Review: A Case Study 

**Authors**: Hanlong Wan, Weili Xu, Michael Rosenberg, Jian Zhang, Aysha Siddika  

**Link**: [PDF](https://arxiv.org/pdf/2510.02634)  

**Abstract**: Building officials, particularly those in resource-constrained or rural jurisdictions, face labor-intensive, error-prone, and costly manual reviews of design documents as projects increase in size and complexity. The growing adoption of Building Information Modeling (BIM) and Large Language Models (LLMs) presents opportunities for automated code review (ACR) solutions. This study introduces a novel agent-driven framework that integrates BIM-based data extraction with automated verification using both retrieval-augmented generation (RAG) and Model Context Protocol (MCP) agent pipelines. The framework employs LLM-enabled agents to extract geometry, schedules, and system attributes from heterogeneous file types, which are then processed for building code checking through two complementary mechanisms: (1) direct API calls to the US Department of Energy COMcheck engine, providing deterministic and audit-ready outputs, and (2) RAG-based reasoning over rule provisions, enabling flexible interpretation where coverage is incomplete or ambiguous.
The framework was evaluated through case demonstrations, including automated extraction of geometric attributes (such as surface area, tilt, and insulation values), parsing of operational schedules, and validation of lighting allowances under ASHRAE Standard 90.1-2022. Comparative performance tests across multiple LLMs showed that GPT-4o achieved the best balance of efficiency and stability, while smaller models exhibited inconsistencies or failures. Results confirm that MCP agent pipelines outperform RAG reasoning pipelines in rigor and reliability. This work advances ACR research by demonstrating a scalable, interoperable, and production-ready approach that bridges BIM with authoritative code review tools. 

---
# ToolTweak: An Attack on Tool Selection in LLM-based Agents 

**Authors**: Jonathan Sneh, Ruomei Yan, Jialin Yu, Philip Torr, Yarin Gal, Sunando Sengupta, Eric Sommerlade, Alasdair Paren, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02554)  

**Abstract**: As LLMs increasingly power agents that interact with external tools, tool use has become an essential mechanism for extending their capabilities. These agents typically select tools from growing databases or marketplaces to solve user tasks, creating implicit competition among tool providers and developers for visibility and usage. In this paper, we show that this selection process harbors a critical vulnerability: by iteratively manipulating tool names and descriptions, adversaries can systematically bias agents toward selecting specific tools, gaining unfair advantage over equally capable alternatives. We present ToolTweak, a lightweight automatic attack that increases selection rates from a baseline of around 20% to as high as 81%, with strong transferability between open-source and closed-source models. Beyond individual tools, we show that such attacks cause distributional shifts in tool usage, revealing risks to fairness, competition, and security in emerging tool ecosystems. To mitigate these risks, we evaluate two defenses: paraphrasing and perplexity filtering, which reduce bias and lead agents to select functionally similar tools more equally. All code will be open-sourced upon acceptance. 

---
# CLARITY: Clinical Assistant for Routing, Inference, and Triage 

**Authors**: Vladimir Shaposhnikov, Aleksandr Nesterov, Ilia Kopanichuk, Ivan Bakulin, Egor Zhelvakov, Ruslan Abramov, Ekaterina Tsapieva, Dmitry V. Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2510.02463)  

**Abstract**: We present CLARITY (Clinical Assistant for Routing, Inference, and Triage), an AI-driven platform designed to facilitate patient-to-specialist routing, clinical consultations, and severity assessment of patients' conditions. Its hybrid architecture combines a Finite State Machine (FSM) for structured dialogue flows with collaborative agents that employ Large Language Model (LLM) to analyze symptoms and prioritize referrals to appropriate specialists. Built on a modular microservices framework, CLARITY ensures safe, efficient, and robust performance, flexible and readily scalable to meet the demands of existing workflows and IT solutions in healthcare.
We report integration of our clinical assistant into a large-scale nation-wide inter-hospital IT platform, with over 55,000 content-rich user dialogues completed within the two months of deployment, 2,500 of which were expert-annotated for a consequent validation. The validation results show that CLARITY surpasses human-level performance in terms of the first-attempt routing precision, naturally requiring up to 3 times shorter duration of the consultation than with a human. 

---
# CWM: An Open-Weights LLM for Research on Code Generation with World Models 

**Authors**: FAIR CodeGen team. Jade Copet, Quentin Carbonneaux, Gal Cohen, Jonas Gehring, Jacob Kahn, Jannik Kossen, Felix Kreuk, Emily McMilin, Michel Meyer, Yuxiang Wei, David Zhang, Kunhao Zheng, Jordi Armengol-Estapé, Pedram Bashiri, Maximilian Beck, Pierre Chambon, Abhishek Charnalia, Chris Cummins, Juliette Decugis, Zacharias V. Fisches, François Fleuret, Fabian Gloeckle, Alex Gu, Michael Hassid, Daniel Haziza, Badr Youbi Idrissi, Christian Keller, Rahul Kindi, Hugh Leather, Gallil Maimon, Aram Markosyan, Francisco Massa, Pierre-Emmanuel Mazaré, Vegard Mella, Naila Murray, Keyur Muzumdar, Peter O'Hearn, Matteo Pagliardini, Dmitrii Pedchenko, Tal Remez, Volker Seeker, Marco Selvi, Oren Sultan, Sida Wang, Luca Wehrstedt, Ori Yoran, Lingming Zhang, Taco Cohen, Yossi Adi, Gabriel Synnaeve  

**Link**: [PDF](https://arxiv.org/pdf/2510.02387)  

**Abstract**: We release Code World Model (CWM), a 32-billion-parameter open-weights LLM, to advance research on code generation with world models. To improve code understanding beyond what can be learned from training on static code alone, we mid-train CWM on a large amount of observation-action trajectories from Python interpreter and agentic Docker environments, and perform extensive multi-task reasoning RL in verifiable coding, math, and multi-turn software engineering environments. With CWM, we provide a strong testbed for researchers to explore the opportunities world modeling affords for improving code generation with reasoning and planning in computational environments. We present first steps of how world models can benefit agentic coding, enable step-by-step simulation of Python code execution, and show early results of how reasoning can benefit from the latter. CWM is a dense, decoder-only LLM trained with a context size of up to 131k tokens. Independent of its world modeling capabilities, CWM offers strong performance on general coding and math tasks: it reaches pass@1 scores of 65.8% on SWE-bench Verified (with test-time scaling), 68.6% on LiveCodeBench, 96.6% on Math-500, and 76.0% on AIME 2024. To support further research on code world modeling, we release model checkpoints after mid-training, SFT, and RL. 

---
# How to Train Your Advisor: Steering Black-Box LLMs with Advisor Models 

**Authors**: Parth Asawa, Alan Zhu, Matei Zaharia, Alexandros G. Dimakis, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2510.02453)  

**Abstract**: Foundation models are increasingly deployed as black-box services, where model weights cannot be modified and customization is limited to prompting. While static prompt optimization has shown promise, it produces a single fixed prompt that fails to adapt to different inputs, users, or environments. We introduce Advisor Models, lightweight parametric policies trained with reinforcement learning to reactively issue natural language steering instructions in-context to black-box models. The advisor is a second small model that sits between the input and the model, shaping behavior on a per-instance basis using reward signals from the environment. Across multiple domains involving reasoning and personalization, we show that Advisor Models outperform static prompt optimizers, discovering environment dynamics and improving downstream task performance. We also demonstrate the generalizability of advisors by transferring them across black-box models, as well as the framework's ability to achieve specialization while retaining robustness to out-of-distribution inputs. Viewed more broadly, Advisor Models provide a learnable interface to black-box systems where the advisor acts as a parametric, environment-specific memory. We argue that dynamic optimization of black-box models via Advisor Models is a promising direction for enabling personalization and environment-adaptable AI with frontier-level capabilities. 

---
# A Hybrid CAPTCHA Combining Generative AI with Keystroke Dynamics for Enhanced Bot Detection 

**Authors**: Ayda Aghaei Nia  

**Link**: [PDF](https://arxiv.org/pdf/2510.02374)  

**Abstract**: Completely Automated Public Turing tests to tell Computers and Humans Apart (CAPTCHAs) are a foundational component of web security, yet traditional implementations suffer from a trade-off between usability and resilience against AI-powered bots. This paper introduces a novel hybrid CAPTCHA system that synergizes the cognitive challenges posed by Large Language Models (LLMs) with the behavioral biometric analysis of keystroke dynamics. Our approach generates dynamic, unpredictable questions that are trivial for humans but non-trivial for automated agents, while simultaneously analyzing the user's typing rhythm to distinguish human patterns from robotic input. We present the system's architecture, formalize the feature extraction methodology for keystroke analysis, and report on an experimental evaluation. The results indicate that our dual-layered approach achieves a high degree of accuracy in bot detection, successfully thwarting both paste-based and script-based simulation attacks, while maintaining a high usability score among human participants. This work demonstrates the potential of combining cognitive and behavioral tests to create a new generation of more secure and user-friendly CAPTCHAs. 

---
# A Cross-Lingual Analysis of Bias in Large Language Models Using Romanian History 

**Authors**: Matei-Iulian Cocu, Răzvan-Cosmin Cristia, Adrian Marius Dumitran  

**Link**: [PDF](https://arxiv.org/pdf/2510.02362)  

**Abstract**: In this case study, we select a set of controversial Romanian historical questions and ask multiple Large Language Models to answer them across languages and contexts, in order to assess their biases. Besides being a study mainly performed for educational purposes, the motivation also lies in the recognition that history is often presented through altered perspectives, primarily influenced by the culture and ideals of a state, even through large language models. Since they are often trained on certain data sets that may present certain ambiguities, the lack of neutrality is subsequently instilled in users. The research process was carried out in three stages, to confirm the idea that the type of response expected can influence, to a certain extent, the response itself; after providing an affirmative answer to some given question, an LLM could shift its way of thinking after being asked the same question again, but being told to respond with a numerical value of a scale. Results show that binary response stability is relatively high but far from perfect and varies by language. Models often flip stance across languages or between formats; numeric ratings frequently diverge from the initial binary choice, and the most consistent models are not always those judged most accurate or neutral. Our research brings to light the predisposition of models to such inconsistencies, within a specific contextualization of the language for the question asked. 

---
# Beyond Manuals and Tasks: Instance-Level Context Learning for LLM Agents 

**Authors**: Kuntai Cai, Juncheng Liu, Xianglin Yang, Zhaojie Niu, Xiaokui Xiao, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02369)  

**Abstract**: Large language model (LLM) agents typically receive two kinds of context: (i) environment-level manuals that define interaction interfaces and global rules, and (ii) task-level guidance or demonstrations tied to specific goals. In this work, we identify a crucial but overlooked third type of context, instance-level context, which consists of verifiable and reusable facts tied to a specific environment instance, such as object locations, crafting recipes, and local rules. We argue that the absence of instance-level context is a common source of failure for LLM agents in complex tasks, as success often depends not only on reasoning over global rules or task prompts but also on making decisions based on precise and persistent facts. Acquiring such context requires more than memorization: the challenge lies in efficiently exploring, validating, and formatting these facts under tight interaction budgets. We formalize this problem as Instance-Level Context Learning (ILCL) and introduce our task-agnostic method to solve it. Our method performs a guided exploration, using a compact TODO forest to intelligently prioritize its next actions and a lightweight plan-act-extract loop to execute them. This process automatically produces a high-precision context document that is reusable across many downstream tasks and agents, thereby amortizing the initial exploration cost. Experiments across TextWorld, ALFWorld, and Crafter demonstrate consistent gains in both success and efficiency: for instance, ReAct's mean success rate in TextWorld rises from 37% to 95%, while IGE improves from 81% to 95%. By transforming one-off exploration into persistent, reusable knowledge, our method complements existing contexts to enable more reliable and efficient LLM agents. 

---
# Emission-GPT: A domain-specific language model agent for knowledge retrieval, emission inventory and data analysis 

**Authors**: Jiashu Ye, Tong Wu, Weiwen Chen, Hao Zhang, Zeteng Lin, Xingxing Li, Shujuan Weng, Manni Zhu, Xin Yuan, Xinlong Hong, Jingjie Li, Junyu Zheng, Zhijiong Huang, Jing Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02359)  

**Abstract**: Improving air quality and addressing climate change relies on accurate understanding and analysis of air pollutant and greenhouse gas emissions. However, emission-related knowledge is often fragmented and highly specialized, while existing methods for accessing and compiling emissions data remain inefficient. These issues hinder the ability of non-experts to interpret emissions information, posing challenges to research and management. To address this, we present Emission-GPT, a knowledge-enhanced large language model agent tailored for the atmospheric emissions domain. Built on a curated knowledge base of over 10,000 documents (including standards, reports, guidebooks, and peer-reviewed literature), Emission-GPT integrates prompt engineering and question completion to support accurate domain-specific question answering. Emission-GPT also enables users to interactively analyze emissions data via natural language, such as querying and visualizing inventories, analyzing source contributions, and recommending emission factors for user-defined scenarios. A case study in Guangdong Province demonstrates that Emission-GPT can extract key insights--such as point source distributions and sectoral trends--directly from raw data with simple prompts. Its modular and extensible architecture facilitates automation of traditionally manual workflows, positioning Emission-GPT as a foundational tool for next-generation emission inventory development and scenario-based assessment. 

---
# Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark 

**Authors**: Xinjie Shen, Mufei Li, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.02356)  

**Abstract**: The deployment of Large Language Models (LLMs) in embodied agents creates an urgent need to measure their privacy awareness in the physical world. Existing evaluation methods, however, are confined to natural language based scenarios. To bridge this gap, we introduce EAPrivacy, a comprehensive evaluation benchmark designed to quantify the physical-world privacy awareness of LLM-powered agents. EAPrivacy utilizes procedurally generated scenarios across four tiers to test an agent's ability to handle sensitive objects, adapt to changing environments, balance task execution with privacy constraints, and resolve conflicts with social norms. Our measurements reveal a critical deficit in current models. The top-performing model, Gemini 2.5 Pro, achieved only 59\% accuracy in scenarios involving changing physical environments. Furthermore, when a task was accompanied by a privacy request, models prioritized completion over the constraint in up to 86\% of cases. In high-stakes situations pitting privacy against critical social norms, leading models like GPT-4o and Claude-3.5-haiku disregarded the social norm over 15\% of the time. These findings, demonstrated by our benchmark, underscore a fundamental misalignment in LLMs regarding physically grounded privacy and establish the need for more robust, physically-aware alignment. 

---
# Evaluating Bias in Spoken Dialogue LLMs for Real-World Decisions and Recommendations 

**Authors**: Yihao Wu, Tianrui Wang, Yizhou Peng, Yi-Wen Chao, Xuyi Zhuang, Xinsheng Wang, Shunshun Yin, Ziyang Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.02352)  

**Abstract**: While biases in large language models (LLMs), such as stereotypes and cultural tendencies in outputs, have been examined and identified, their presence and characteristics in spoken dialogue models (SDMs) with audio input and output remain largely unexplored. Paralinguistic features, such as age, gender, and accent, can affect model outputs; when compounded by multi-turn conversations, these effects may exacerbate biases, with potential implications for fairness in decision-making and recommendation tasks. In this paper, we systematically evaluate biases in speech LLMs and study the impact of multi-turn dialogues with repeated negative feedback. Bias is measured using Group Unfairness Score (GUS) for decisions and similarity-based normalized statistics rate (SNSR) for recommendations, across both open-source models like Qwen2.5-Omni and GLM-4-Voice, as well as closed-source APIs such as GPT-4o Audio and Gemini-2.5-Flash. Our analysis reveals that closed-source models generally exhibit lower bias, while open-source models are more sensitive to age and gender, and recommendation tasks tend to amplify cross-group disparities. We found that biased decisions may persist in multi-turn conversations. This work provides the first systematic study of biases in end-to-end spoken dialogue models, offering insights towards fair and reliable audio-based interactive systems. To facilitate further research, we release the FairDialogue dataset and evaluation code. 

---
# ChunkLLM: A Lightweight Pluggable Framework for Accelerating LLMs Inference 

**Authors**: Haojie Ouyang, Jianwei Lv, Lei Ren, Chen Wei, Xiaojie Wang, Fangxiang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02361)  

**Abstract**: Transformer-based large models excel in natural language processing and computer vision, but face severe computational inefficiencies due to the self-attention's quadratic complexity with input tokens. Recently, researchers have proposed a series of methods based on block selection and compression to alleviate this problem, but they either have issues with semantic incompleteness or poor training-inference efficiency. To comprehensively address these challenges, we propose ChunkLLM, a lightweight and pluggable training framework. Specifically, we introduce two components: QK Adapter (Q-Adapter and K-Adapter) and Chunk Adapter. The former is attached to each Transformer layer, serving dual purposes of feature compression and chunk attention acquisition. The latter operates at the bottommost layer of the model, functioning to detect chunk boundaries by leveraging contextual semantic information. During the training phase, the parameters of the backbone remain frozen, with only the QK Adapter and Chunk Adapter undergoing training. Notably, we design an attention distillation method for training the QK Adapter, which enhances the recall rate of key chunks. During the inference phase, chunk selection is triggered exclusively when the current token is detected as a chunk boundary, thereby accelerating model inference. Experimental evaluations are conducted on a diverse set of long-text and short-text benchmark datasets spanning multiple tasks. ChunkLLM not only attains comparable performance on short-text benchmarks but also maintains 98.64% of the performance on long-context benchmarks while preserving a 48.58% key-value cache retention rate. Particularly, ChunkLLM attains a maximum speedup of 4.48x in comparison to the vanilla Transformer in the processing of 120K long texts. 

---
# Language, Culture, and Ideology: Personalizing Offensiveness Detection in Political Tweets with Reasoning LLMs 

**Authors**: Dzmitry Pihulski, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2510.02351)  

**Abstract**: We explore how large language models (LLMs) assess offensiveness in political discourse when prompted to adopt specific political and cultural perspectives. Using a multilingual subset of the MD-Agreement dataset centered on tweets from the 2020 US elections, we evaluate several recent LLMs - including DeepSeek-R1, o4-mini, GPT-4.1-mini, Qwen3, Gemma, and Mistral - tasked with judging tweets as offensive or non-offensive from the viewpoints of varied political personas (far-right, conservative, centrist, progressive) across English, Polish, and Russian contexts. Our results show that larger models with explicit reasoning abilities (e.g., DeepSeek-R1, o4-mini) are more consistent and sensitive to ideological and cultural variation, while smaller models often fail to capture subtle distinctions. We find that reasoning capabilities significantly improve both the personalization and interpretability of offensiveness judgments, suggesting that such mechanisms are key to adapting LLMs for nuanced sociopolitical text classification across languages and ideologies. 

---
# Glaucoma Detection and Structured OCT Report Generation via a Fine-tuned Multimodal Large Language Model 

**Authors**: Jalil Jalili, Yashraj Gavhane, Evan Walker, Anna Heinke, Christopher Bowd, Akram Belghith, Massimo A. Fazio, Christopher A. Girkin, C. Gustavo De Moraes, Jeffrey M. Liebmann, Sally L. Baxter, Robert N. Weinreb, Linda M. Zangwill, Mark Christopher  

**Link**: [PDF](https://arxiv.org/pdf/2510.02403)  

**Abstract**: Objective: To develop an explainable multimodal large language model (MM-LLM) that (1) screens optic nerve head (ONH) OCT circle scans for quality and (2) generates structured clinical reports that include glaucoma diagnosis and sector-wise retinal nerve fiber layer (RNFL) thinning assessments. Design: Retrospective cohort study of 1,310 subjects contributing 43,849 Spectralis ONH OCT circle scans (1,331 glaucomatous and 867 healthy eyes) from the DIGS and ADAGES cohorts. Methods: A MM-LLM (Llama 3.2 Vision-Instruct model) was fine-tuned to generate clinical descriptions of OCT imaging data. Training data included paired OCT images and automatically generated, structured clinical reports that described global and sectoral RNFL thinning. Poor-quality scans were labeled as unusable and paired with a fixed refusal statement. The model was evaluated on a held-out test set for three tasks: quality assessment, glaucoma detection, and RNFL thinning classification across seven anatomical sectors. Evaluation metrics included accuracy, sensitivity, specificity, precision, and F1-score. Model description quality was also evaluated using standard text evaluation metrics. Results: The model achieved 0.90 accuracy and 0.98 specificity for quality triage. For glaucoma detection, accuracy was 0.86 (sensitivity 0.91, specificity 0.73, F1-score 0.91). RNFL thinning prediction accuracy ranged from 0.83 to 0.94, with highest performance in global and temporal sectors. Text generation scores showed strong alignment with reference reports (BLEU: 0.82; ROUGE-1: 0.94; ROUGE-2: 0.87; ROUGE-L: 0.92; BERTScore-F1: 0.99). Conclusions: The fine-tuned MM-LLM generated accurate clinical descriptions based on OCT imaging. The model achieved high accuracy in identifying image quality issues and detecting glaucoma. The model also provided sectoral descriptions of RNFL thinning to help support clinical OCT evaluation. 

---
# LLMSQL: Upgrading WikiSQL for the LLM Era of Text-to-SQL 

**Authors**: Dzmitry Pihulski, Karol Charchut, Viktoria Novogrodskaia, Jan Kocoń  

**Link**: [PDF](https://arxiv.org/pdf/2510.02350)  

**Abstract**: Converting natural language questions into SQL queries (Text-to-SQL) enables non-expert users to interact with relational databases and has long been a central task for natural language interfaces to data. While the WikiSQL dataset played a key role in early NL2SQL research, its usage has declined due to structural and annotation issues, including case sensitivity inconsistencies, data type mismatches, syntax errors, and unanswered questions. We present LLMSQL, a systematic revision and transformation of WikiSQL designed for the LLM era. We classify these errors and implement automated methods for cleaning and re-annotation. To assess the impact of these improvements, we evaluated multiple large language models (LLMs), including Gemma 3, LLaMA 3.2, Mistral 7B, gpt-oss 20B, Phi-3.5 Mini, Qwen 2.5, OpenAI o4-mini, DeepSeek R1 and others. Rather than serving as an update, LLMSQL is introduced as an LLM-ready benchmark: unlike the original WikiSQL, tailored for pointer-network models selecting tokens from input, LLMSQL provides clean natural language questions and full SQL queries as plain text, enabling straightforward generation and evaluation for modern natural language-to-SQL models. 

---
# Small Language Models for Curriculum-based Guidance 

**Authors**: Konstantinos Katharakis, Sippo Rossi, Raghava Rao Mukkamala  

**Link**: [PDF](https://arxiv.org/pdf/2510.02347)  

**Abstract**: The adoption of generative AI and large language models (LLMs) in education is still emerging. In this study, we explore the development and evaluation of AI teaching assistants that provide curriculum-based guidance using a retrieval-augmented generation (RAG) pipeline applied to selected open-source small language models (SLMs). We benchmarked eight SLMs, including LLaMA 3.1, IBM Granite 3.3, and Gemma 3 (7-17B parameters), against GPT-4o. Our findings show that with proper prompting and targeted retrieval, SLMs can match LLMs in delivering accurate, pedagogically aligned responses. Importantly, SLMs offer significant sustainability benefits due to their lower computational and energy requirements, enabling real-time use on consumer-grade hardware without depending on cloud infrastructure. This makes them not only cost-effective and privacy-preserving but also environmentally responsible, positioning them as viable AI teaching assistants for educational institutions aiming to scale personalized learning in a sustainable and energy-efficient manner. 

---
# Evaluating Uncertainty Quantification Methods in Argumentative Large Language Models 

**Authors**: Kevin Zhou, Adam Dejl, Gabriel Freedman, Lihu Chen, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2510.02339)  

**Abstract**: Research in uncertainty quantification (UQ) for large language models (LLMs) is increasingly important towards guaranteeing the reliability of this groundbreaking technology. We explore the integration of LLM UQ methods in argumentative LLMs (ArgLLMs), an explainable LLM framework for decision-making based on computational argumentation in which UQ plays a critical role. We conduct experiments to evaluate ArgLLMs' performance on claim verification tasks when using different LLM UQ methods, inherently performing an assessment of the UQ methods' effectiveness. Moreover, the experimental procedure itself is a novel way of evaluating the effectiveness of UQ methods, especially when intricate and potentially contentious statements are present. Our results demonstrate that, despite its simplicity, direct prompting is an effective UQ strategy in ArgLLMs, outperforming considerably more complex approaches. 

---
# A-MemGuard: A Proactive Defense Framework for LLM-Based Agent Memory 

**Authors**: Qianshan Wei, Tengchao Yang, Yaochen Wang, Xinfeng Li, Lijun Li, Zhenfei Yin, Yi Zhan, Thorsten Holz, Zhiqiang Lin, XiaoFeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02373)  

**Abstract**: Large Language Model (LLM) agents use memory to learn from past interactions, enabling autonomous planning and decision-making in complex environments. However, this reliance on memory introduces a critical security risk: an adversary can inject seemingly harmless records into an agent's memory to manipulate its future behavior. This vulnerability is characterized by two core aspects: First, the malicious effect of injected records is only activated within a specific context, making them hard to detect when individual memory entries are audited in isolation. Second, once triggered, the manipulation can initiate a self-reinforcing error cycle: the corrupted outcome is stored as precedent, which not only amplifies the initial error but also progressively lowers the threshold for similar attacks in the future. To address these challenges, we introduce A-MemGuard (Agent-Memory Guard), the first proactive defense framework for LLM agent memory. The core idea of our work is the insight that memory itself must become both self-checking and self-correcting. Without modifying the agent's core architecture, A-MemGuard combines two mechanisms: (1) consensus-based validation, which detects anomalies by comparing reasoning paths derived from multiple related memories and (2) a dual-memory structure, where detected failures are distilled into ``lessons'' stored separately and consulted before future actions, breaking error cycles and enabling adaptation. Comprehensive evaluations on multiple benchmarks show that A-MemGuard effectively cuts attack success rates by over 95% while incurring a minimal utility cost. This work shifts LLM memory security from static filtering to a proactive, experience-driven model where defenses strengthen over time. Our code is available in this https URL 

---
# FormalML: A Benchmark for Evaluating Formal Subgoal Completion in Machine Learning Theory 

**Authors**: Xiao-Wen Yang, Zihao Zhang, Jianuo Cao, Zhi Zhou, Zenan Li, Lan-Zhe Guo, Yuan Yao, Taolue Chen, Yu-Feng Li, Xiaoxing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.02335)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable progress in formal theorem proving. Yet their ability to serve as practical assistants for mathematicians, filling in missing steps within complex proofs, remains underexplored. We identify this challenge as the task of subgoal completion, where an LLM must discharge short but nontrivial proof obligations left unresolved in a human-provided sketch. To study this problem, we introduce FormalML, a Lean 4 benchmark built from foundational theories of machine learning. Using a translation tactic that converts procedural proofs into declarative form, we extract 4937 problems spanning optimization and probability inequalities, with varying levels of difficulty. FormalML is the first subgoal completion benchmark to combine premise retrieval and complex research-level contexts. Evaluation of state-of-the-art provers highlights persistent limitations in accuracy and efficiency, underscoring the need for more capable LLM-based theorem provers for effective subgoal completion, 

---
# Where Did It Go Wrong? Attributing Undesirable LLM Behaviors via Representation Gradient Tracing 

**Authors**: Zhe Li, Wei Zhao, Yige Li, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.02334)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their deployment is frequently undermined by undesirable behaviors such as generating harmful content, factual inaccuracies, and societal biases. Diagnosing the root causes of these failures poses a critical challenge for AI safety. Existing attribution methods, particularly those based on parameter gradients, often fall short due to prohibitive noisy signals and computational complexity. In this work, we introduce a novel and efficient framework that diagnoses a range of undesirable LLM behaviors by analyzing representation and its gradients, which operates directly in the model's activation space to provide a semantically meaningful signal linking outputs to their training data. We systematically evaluate our method for tasks that include tracking harmful content, detecting backdoor poisoning, and identifying knowledge contamination. The results demonstrate that our approach not only excels at sample-level attribution but also enables fine-grained token-level analysis, precisely identifying the specific samples and phrases that causally influence model behavior. This work provides a powerful diagnostic tool to understand, audit, and ultimately mitigate the risks associated with LLMs. The code is available at this https URL. 

---
# Breaking the MoE LLM Trilemma: Dynamic Expert Clustering with Structured Compression 

**Authors**: Peijun Zhu, Ning Yang, Jiayu Wei, Jinghang Wu, Haijun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02345)  

**Abstract**: Mixture-of-Experts (MoE) Large Language Models (LLMs) face a trilemma of load imbalance, parameter redundancy, and communication overhead. We introduce a unified framework based on dynamic expert clustering and structured compression to address these issues cohesively. Our method employs an online clustering procedure that periodically regroups experts using a fused metric of parameter and activation similarity, which stabilizes expert utilization. To our knowledge, this is one of the first frameworks to leverage the semantic embedding capability of the router to dynamically reconfigure the model's architecture during training for substantial efficiency gains. Within each cluster, we decompose expert weights into a shared base matrix and extremely low-rank residual adapters, achieving up to fivefold parameter reduction per group while preserving specialization. This structure enables a two-stage hierarchical routing strategy: tokens are first assigned to a cluster, then to specific experts within it, drastically reducing the routing search space and the volume of all-to-all communication. Furthermore, a heterogeneous precision scheme, which stores shared bases in FP16 and residual factors in INT4, coupled with dynamic offloading of inactive clusters, reduces peak memory consumption to levels comparable to dense models. Evaluated on GLUE and WikiText-103, our framework matches the quality of standard MoE models while reducing total parameters by approximately 80%, improving throughput by 10% to 20%, and lowering expert load variance by a factor of over three. Our work demonstrates that structural reorganization is a principled path toward scalable, efficient, and memory-effective MoE LLMs. 

---
# DiffuSpec: Unlocking Diffusion Language Models for Speculative Decoding 

**Authors**: Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02358)  

**Abstract**: As large language models (LLMs) scale up, accuracy improves, but the autoregressive (AR) nature of decoding increases latency since each token requires a serial forward pass. Speculative decoding addresses this by employing a fast drafter to propose multi-token drafts, which are then verified in parallel by the target model. However, many deployments still rely on AR drafters, where sequential passes limit wall-clock gains. We revisit the drafting stage and present DiffuSpec, a training-free drop-in framework that uses a pretrained diffusion language model (DLM) to produce multi-token drafts in a single forward pass, while remaining compatible with standard AR verifiers. Because DLM drafts are generated under bidirectional conditioning, parallel per-position candidates form a token lattice in which the locally highest-probability token at each position need not form a causal left-to-right path. Moreover, DLM drafting requires pre-specifying a draft length, inducing a speed-quality trade-off. To address these challenges, we introduce two practical components: (i) a causal-consistency path search (CPS) over this lattice that extracts a left-to-right path aligned with AR verification; and (ii) an adaptive draft-length (ADL) controller that adjusts next proposal size based on recent acceptance feedback and realized generated length. Across benchmarks, DiffuSpec yields up to 3x wall-clock speedup, establishing diffusion-based drafting as a robust alternative to autoregressive drafters for speculative decoding. 

---
# Human Mobility Datasets Enriched With Contextual and Social Dimensions 

**Authors**: Chiara Pugliese, Francesco Lettich, Guido Rocchietti, Chiara Renso, Fabio Pinelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.02333)  

**Abstract**: In this resource paper, we present two publicly available datasets of semantically enriched human trajectories, together with the pipeline to build them. The trajectories are publicly available GPS traces retrieved from OpenStreetMap. Each dataset includes contextual layers such as stops, moves, points of interest (POIs), inferred transportation modes, and weather data. A novel semantic feature is the inclusion of synthetic, realistic social media posts generated by Large Language Models (LLMs), enabling multimodal and semantic mobility analysis. The datasets are available in both tabular and Resource Description Framework (RDF) formats, supporting semantic reasoning and FAIR data practices. They cover two structurally distinct, large cities: Paris and New York. Our open source reproducible pipeline allows for dataset customization, while the datasets support research tasks such as behavior modeling, mobility prediction, knowledge graph construction, and LLM-based applications. To our knowledge, our resource is the first to combine real-world movement, structured semantic enrichment, LLM-generated text, and semantic web compatibility in a reusable framework. 

---
# Hallucination-Resistant, Domain-Specific Research Assistant with Self-Evaluation and Vector-Grounded Retrieval 

**Authors**: Vivek Bhavsar, Joseph Ereifej, Aravanan Gurusami  

**Link**: [PDF](https://arxiv.org/pdf/2510.02326)  

**Abstract**: Large language models accelerate literature synthesis but can hallucinate and mis-cite, limiting their usefulness in expert workflows. We present RA-FSM (Research Assistant - Finite State Machine), a modular GPT-based research assistant that wraps generation in a finite-state control loop: Relevance -> Confidence -> Knowledge. The system is grounded in vector retrieval and a deterministic citation pipeline. The controller filters out-of-scope queries, scores answerability, decomposes questions, and triggers retrieval only when needed, and emits answers with confidence labels and in-corpus, de-duplicated references. A ranked-tier ingestion workflow constructs a domain knowledge base from journals, conferences, indices, preprints, and patents, writing both to a dense vector index and to a relational store of normalized metrics. We implement the system for photonics and evaluate it on six task categories: analytical reasoning, numerical analysis, methodological critique, comparative synthesis, factual extraction, and application design. In blinded A/B reviews, domain experts prefer RA-FSM to both a strong Notebook LM (NLM) and a vanilla Default GPT API call single-pass baseline, citing stronger boundary-condition handling and more defensible evidence use. Coverage and novelty analyses indicate that RA-FSM explores beyond the NLM while incurring tunable latency and cost overheads. The design emphasizes transparent, well-cited answers for high-stakes technical work and is generalizable to other scientific domains. 

---
# AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering 

**Authors**: Ziqing Wang, Chengsheng Mao, Xiaole Wen, Yuan Luo, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.02328)  

**Abstract**: Medical Multimodal Large Language Models (Med-MLLMs) have shown great promise in medical visual question answering (Med-VQA). However, when deployed in low-resource settings where abundant labeled data are unavailable, existing Med-MLLMs commonly fail due to their medical reasoning capability bottlenecks: (i) the intrinsic reasoning bottleneck that ignores the details from the medical image; (ii) the extrinsic reasoning bottleneck that fails to incorporate specialized medical knowledge. To address those limitations, we propose AMANDA, a training-free agentic framework that performs medical knowledge augmentation via LLM agents. Specifically, our intrinsic medical knowledge augmentation focuses on coarse-to-fine question decomposition for comprehensive diagnosis, while extrinsic medical knowledge augmentation grounds the reasoning process via biomedical knowledge graph retrieval. Extensive experiments across eight Med-VQA benchmarks demonstrate substantial improvements in both zero-shot and few-shot Med-VQA settings. The code is available at this https URL. 

---
# Spiral of Silence in Large Language Model Agents 

**Authors**: Mingze Zhong, Meng Fang, Zijing Shi, Yuxuan Huang, Shunfeng Zheng, Yali Du, Ling Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02360)  

**Abstract**: The Spiral of Silence (SoS) theory holds that individuals with minority views often refrain from speaking out for fear of social isolation, enabling majority positions to dominate public discourse. When the 'agents' are large language models (LLMs), however, the classical psychological explanation is not directly applicable, since SoS was developed for human societies. This raises a central question: can SoS-like dynamics nevertheless emerge from purely statistical language generation in LLM collectives? We propose an evaluation framework for examining SoS in LLM agents. Specifically, we consider four controlled conditions that systematically vary the availability of 'History' and 'Persona' signals. Opinion dynamics are assessed using trend tests such as Mann-Kendall and Spearman's rank, along with concentration measures including kurtosis and interquartile range. Experiments across open-source and closed-source models show that history and persona together produce strong majority dominance and replicate SoS patterns; history signals alone induce strong anchoring; and persona signals alone foster diverse but uncorrelated opinions, indicating that without historical anchoring, SoS dynamics cannot emerge. The work bridges computational sociology and responsible AI design, highlighting the need to monitor and mitigate emergent conformity in LLM-agent systems. 

---
# Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning 

**Authors**: Wannan Yang, Xinchi Qiu, Lei Yu, Yuchen Zhang, Oliver Aobo Yang, Narine Kokhlikyan, Nicola Cancedda, Diego Garcia-Olano  

**Link**: [PDF](https://arxiv.org/pdf/2510.02324)  

**Abstract**: Large Language Models (LLMs) exhibit impressive capabilities but often hallucinate, confidently providing incorrect answers instead of admitting ignorance. Prior work has shown that models encode linear representations of their own knowledge and that activation steering can reduce hallucinations. These approaches, however, require real-time monitoring and intervention during inference. We introduce Contrastive Activation Steering for Amortized Learning (CASAL), an efficient algorithm that connects interpretability with amortized optimization. CASAL directly bakes the benefits of activation steering into model's weights. Once trained, LLMs answer questions they know while abstaining from answering those they do not. CASAL's light-weight design requires training only a submodule of a single transformer layer and yet reduces hallucination by 30%-40% across multiple short-form QA benchmarks. CASAL is 30x more compute-efficient and 20x more data-efficient than strong LoRA-based baselines such as SFT and DPO, boosting its practical applicability in data scarce domains. Importantly, CASAL also generalizes effectively to out-of-distribution (OOD) domains. We showcase CASAL's flexibility in mitigating hallucinations in both text-only and vision-language models. To our knowledge, CASAL is the first steering-based training method that has been shown to be effective for both dense and Mixture-of-Experts (MoE) models. CASAL represents a promising step forward for applying interpretability-inspired method for practical deployment in production systems. 

---
# SelfJudge: Faster Speculative Decoding via Self-Supervised Judge Verification 

**Authors**: Kanghoon Yoon, Minsub Kim, Sungjae Lee, Joonhyung Lee, Sunghyeon Woo, Yeonjun In, Se Jung Kwon, Chanyoung Park, Dongsoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.02329)  

**Abstract**: Speculative decoding accelerates LLM inference by verifying candidate tokens from a draft model against a larger target model. Recent judge decoding boosts this process by relaxing verification criteria by accepting draft tokens that may exhibit minor discrepancies from target model output, but existing methods are restricted by their reliance on human annotations or tasks with verifiable ground truths, limiting generalizability across diverse NLP tasks. We propose SelfJudge, which trains judge verifiers via self-supervision of the target model. Our method measures semantic preservation by assessing whether token-substituted responses preserve the meaning of original responses, enabling automatic verifier training across diverse NLP tasks. Our experiments show SelfJudge achieves superior inference-accuracy trade-offs than judge decoding baselines, offering a broadly applicable solution for faster LLM inference. 

---
# KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI 

**Authors**: So Kuroki, Yotaro Kubo, Takuya Akiba, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02327)  

**Abstract**: Real-time speech-to-speech (S2S) models excel at generating natural, low-latency conversational responses but often lack deep knowledge and semantic understanding. Conversely, cascaded systems combining automatic speech recognition, a text-based Large Language Model (LLM), and text-to-speech synthesis offer superior knowledge representation at the cost of high latency, which disrupts the flow of natural interaction. This paper introduces a novel hybrid architecture that bridges the gap between these two paradigms. Our framework processes user speech through an S2S transformer for immediate responsiveness while concurrently relaying the query to a powerful back-end LLM. The LLM's text-based response is then injected in real time to guide the S2S model's speech generation, effectively infusing its output with rich knowledge without the full latency penalty of a cascaded system. We evaluated our method using a speech-synthesized variant of the MT-Bench benchmark that consists of multi-turn question-answering sessions. The results demonstrate that our system substantially outperforms a baseline S2S model in response correctness, approaching that of a cascaded system, while maintaining a latency on par with the baseline. 

---
# FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents 

**Authors**: Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han Lù, Léo Boisvert, Massimo Caccia, Jérémy Espinas, Alexandre Aussem, Véronique Eglin, Alexandre Lacoste  

**Link**: [PDF](https://arxiv.org/pdf/2510.03204)  

**Abstract**: Web agents powered by large language models (LLMs) must process lengthy web page observations to complete user goals; these pages often exceed tens of thousands of tokens. This saturates context limits and increases computational cost processing; moreover, processing full pages exposes agents to security risks such as prompt injection. Existing pruning strategies either discard relevant content or retain irrelevant context, leading to suboptimal action prediction. We introduce FocusAgent, a simple yet effective approach that leverages a lightweight LLM retriever to extract the most relevant lines from accessibility tree (AxTree) observations, guided by task goals. By pruning noisy and irrelevant content, FocusAgent enables efficient reasoning while reducing vulnerability to injection attacks. Experiments on WorkArena and WebArena benchmarks show that FocusAgent matches the performance of strong baselines, while reducing observation size by over 50%. Furthermore, a variant of FocusAgent significantly reduces the success rate of prompt-injection attacks, including banner and pop-up attacks, while maintaining task success performance in attack-free settings. Our results highlight that targeted LLM-based retrieval is a practical and robust strategy for building web agents that are efficient, effective, and secure. 

---
# Cache-to-Cache: Direct Semantic Communication Between Large Language Models 

**Authors**: Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03215)  

**Abstract**: Multi-LLM systems harness the complementary strengths of diverse Large Language Models, achieving performance and efficiency gains unattainable by a single model. In existing designs, LLMs communicate through text, forcing internal representations to be transformed into output token sequences. This process both loses rich semantic information and incurs token-by-token generation latency. Motivated by these limitations, we ask: Can LLMs communicate beyond text? Oracle experiments show that enriching the KV-Cache semantics can improve response quality without increasing cache size, supporting KV-Cache as an effective medium for inter-model communication. Thus, we propose Cache-to-Cache (C2C), a new paradigm for direct semantic communication between LLMs. C2C uses a neural network to project and fuse the source model's KV-cache with that of the target model to enable direct semantic transfer. A learnable gating mechanism selects the target layers that benefit from cache communication. Compared with text communication, C2C utilizes the deep, specialized semantics from both models, while avoiding explicit intermediate text generation. Experiments show that C2C achieves 8.5-10.5% higher average accuracy than individual models. It further outperforms the text communication paradigm by approximately 3.0-5.0%, while delivering an average 2.0x speedup in latency. Our code is available at this https URL. 

---
# Semantic Similarity in Radiology Reports via LLMs and NER 

**Authors**: Beth Pearson, Ahmed Adnan, Zahraa Abdallah  

**Link**: [PDF](https://arxiv.org/pdf/2510.03102)  

**Abstract**: Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: \href{this https URL}{this http URL\_reports} 

---
# Self-Reflective Generation at Test Time 

**Authors**: Jian Mu, Qixin Zhang, Zhiyong Wang, Menglin Yang, Shuang Qiu, Chengwei Qin, Zhongxiang Dai, Yao Shu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02919)  

**Abstract**: Large language models (LLMs) increasingly solve complex reasoning tasks via long chain-of-thought, but their forward-only autoregressive generation process is fragile; early token errors can cascade, which creates a clear need for self-reflection mechanisms. However, existing self-reflection either performs revisions over full drafts or learns self-correction via expensive training, both fundamentally reactive and inefficient. To address this, we propose Self-Reflective Generation at Test Time (SRGen), a lightweight test-time framework that reflects before generating at uncertain points. During token generation, SRGen utilizes dynamic entropy thresholding to identify high-uncertainty tokens. For each identified token, it trains a specific corrective vector, which fully exploits the already generated context for a self-reflective generation to correct the token probability distribution. By retrospectively analyzing the partial output, this self-reflection enables more trustworthy decisions, thereby significantly reducing the probability of errors at highly uncertain points. Evaluated on challenging mathematical reasoning benchmarks and a diverse set of LLMs, SRGen can consistently strengthen model reasoning: improvements in single-pass quality also translate into stronger self-consistency voting. Especially, on AIME2024 with DeepSeek-R1-Distill-Qwen-7B, SRGen yields absolute improvements of +12.0% on Pass@1 and +13.3% on Cons@5. Moreover, our findings position SRGen as a plug-and-play method that integrates reflection into the generation process for reliable LLM reasoning, achieving consistent gains with bounded overhead and broad composability with other training-time (e.g., RLHF) and test-time (e.g., SLOT) techniques. 

---
# Beyond the Final Layer: Intermediate Representations for Better Multilingual Calibration in Large Language Models 

**Authors**: Ej Zhou, Caiqi Zhang, Tiancheng Hu, Chengzu Li, Nigel Collier, Ivan Vulić, Anna Korhonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03136)  

**Abstract**: Confidence calibration, the alignment of a model's predicted confidence with its actual accuracy, is crucial for the reliable deployment of Large Language Models (LLMs). However, this critical property remains largely under-explored in multilingual contexts. In this work, we conduct the first large-scale, systematic studies of multilingual calibration across six model families and over 100 languages, revealing that non-English languages suffer from systematically worse calibration. To diagnose this, we investigate the model's internal representations and find that the final layer, biased by English-centric training, provides a poor signal for multilingual confidence. In contrast, our layer-wise analysis uncovers a key insight that late-intermediate layers consistently offer a more reliable and better-calibrated signal. Building on this, we introduce a suite of training-free methods, including Language-Aware Confidence Ensemble (LACE), which adaptively selects an optimal ensemble of layers for each specific language. Our study highlights the hidden costs of English-centric alignment and offer a new path toward building more globally equitable and trustworthy LLMs by looking beyond the final layer. 

---
# The Path of Self-Evolving Large Language Models: Achieving Data-Efficient Learning via Intrinsic Feedback 

**Authors**: Hangfan Zhang, Siyuan Xu, Zhimeng Guo, Huaisheng Zhu, Shicheng Liu, Xinrun Wang, Qiaosheng Zhang, Yang Chen, Peng Ye, Lei Bai, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02752)  

**Abstract**: Reinforcement learning (RL) has demonstrated potential in enhancing the reasoning capabilities of large language models (LLMs), but such training typically demands substantial efforts in creating and annotating data. In this work, we explore improving LLMs through RL with minimal data. Our approach alternates between the LLM proposing a task and then attempting to solve it. To minimize data dependency, we introduce two novel mechanisms grounded in self-awareness: (1) self-aware difficulty prediction, where the model learns to assess task difficulty relative to its own abilities and prioritize challenging yet solvable tasks, and (2) self-aware limit breaking, where the model recognizes when a task is beyond its capability boundary and proactively requests external data to break through that limit. Extensive experiments on nine benchmarks showing a 53.8% relative improvement with less than 1.2% extra data demonstrate the efficacy of self-aware RL and underscore the promise of self-evolving agent training. 

---
# Uncertainty as Feature Gaps: Epistemic Uncertainty Quantification of LLMs in Contextual Question-Answering 

**Authors**: Yavuz Bakman, Sungmin Kang, Zhiqi Huang, Duygu Nur Yaldiz, Catarina G. Belém, Chenyang Zhu, Anoop Kumar, Alfy Samuel, Salman Avestimehr, Daben Liu, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.02671)  

**Abstract**: Uncertainty Quantification (UQ) research has primarily focused on closed-book factual question answering (QA), while contextual QA remains unexplored, despite its importance in real-world applications. In this work, we focus on UQ for the contextual QA task and propose a theoretically grounded approach to quantify epistemic uncertainty. We begin by introducing a task-agnostic, token-level uncertainty measure defined as the cross-entropy between the predictive distribution of the given model and the unknown true distribution. By decomposing this measure, we isolate the epistemic component and approximate the true distribution by a perfectly prompted, idealized model. We then derive an upper bound for epistemic uncertainty and show that it can be interpreted as semantic feature gaps in the given model's hidden representations relative to the ideal model. We further apply this generic framework to the contextual QA task and hypothesize that three features approximate this gap: context-reliance (using the provided context rather than parametric knowledge), context comprehension (extracting relevant information from context), and honesty (avoiding intentional lies). Using a top-down interpretability approach, we extract these features by using only a small number of labeled samples and ensemble them to form a robust uncertainty score. Experiments on multiple QA benchmarks in both in-distribution and out-of-distribution settings show that our method substantially outperforms state-of-the-art unsupervised (sampling-free and sampling-based) and supervised UQ methods, achieving up to a 13-point PRR improvement while incurring a negligible inference overhead. 

---
# Mind the Gap: Linguistic Divergence and Adaptation Strategies in Human-LLM Assistant vs. Human-Human Interactions 

**Authors**: Fulei Zhang, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02645)  

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in customer-facing applications, a critical yet underexplored question is how users communicate differently with LLM chatbots compared to human agent. In this study, we present empirical evidence that users adopt distinct communication styles when users interact with chatbots versus human agents. Our analysis reveals significant differences in grammatical fluency, politeness, and lexical diversity in user language between the two settings. These findings suggest that models trained exclusively on human-human interaction data may not adequately accommodate the communication style shift that occurs once an LLM chatbot is deployed. To enhance LLM robustness to post-launch communication style changes, we experimented with two strategies: (1) data augmentation during the post-training phase and (2) inference-time user message reformulation. Our results indicate that models trained on stylistically diverse datasets significantly outperform those trained exclusively on original or stylistically uniform datasets, while inference-time reformulation proved less effective. These insights help us to better adapt our models for improved LLM-user interaction experiences. 

---
# SurveyBench: How Well Can LLM(-Agents) Write Academic Surveys? 

**Authors**: Zhaojun Sun, Xuzhou Zhu, Xuanhe Zhou, Xin Tong, Shuo Wang, Jie Fu, Guoliang Li, Zhiyuan Liu, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03120)  

**Abstract**: Academic survey writing, which distills vast literature into a coherent and insightful narrative, remains a labor-intensive and intellectually demanding task. While recent approaches, such as general DeepResearch agents and survey-specialized methods, can generate surveys automatically (a.k.a. LLM4Survey), their outputs often fall short of human standards and there lacks a rigorous, reader-aligned benchmark for thoroughly revealing their deficiencies. To fill the gap, we propose a fine-grained, quiz-driven evaluation framework SurveyBench, featuring (1) typical survey topics source from recent 11,343 arXiv papers and corresponding 4,947 high-quality surveys; (2) a multifaceted metric hierarchy that assesses the outline quality (e.g., coverage breadth, logical coherence), content quality (e.g., synthesis granularity, clarity of insights), and non-textual richness; and (3) a dual-mode evaluation protocol that includes content-based and quiz-based answerability tests, explicitly aligned with readers' informational needs. Results show SurveyBench effectively challenges existing LLM4Survey approaches (e.g., on average 21% lower than human in content-based evaluation). 

---
# Self-Improvement in Multimodal Large Language Models: A Survey 

**Authors**: Shijian Deng, Kai Wang, Tianyu Yang, Harsh Singh, Yapeng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2510.02665)  

**Abstract**: Recent advancements in self-improvement for Large Language Models (LLMs) have efficiently enhanced model capabilities without significantly increasing costs, particularly in terms of human effort. While this area is still relatively young, its extension to the multimodal domain holds immense potential for leveraging diverse data sources and developing more general self-improving models. This survey is the first to provide a comprehensive overview of self-improvement in Multimodal LLMs (MLLMs). We provide a structured overview of the current literature and discuss methods from three perspectives: 1) data collection, 2) data organization, and 3) model optimization, to facilitate the further development of self-improvement in MLLMs. We also include commonly used evaluations and downstream applications. Finally, we conclude by outlining open challenges and future research directions. 

---
# IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context 

**Authors**: Santhosh G S, Akshay Govind S, Gokul S Krishnan, Balaraman Ravindran, Sriraam Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02742)  

**Abstract**: Large Language Models (LLMs) have gained significant traction across critical domains owing to their impressive contextual understanding and generative capabilities. However, their increasing deployment in high stakes applications necessitates rigorous evaluation of embedded biases, particularly in culturally diverse contexts like India where existing embedding-based bias assessment methods often fall short in capturing nuanced stereotypes. We propose an evaluation framework based on a encoder trained using contrastive learning that captures fine-grained bias through embedding similarity. We also introduce a novel dataset - IndiCASA (IndiBias-based Contextually Aligned Stereotypes and Anti-stereotypes) comprising 2,575 human-validated sentences spanning five demographic axes: caste, gender, religion, disability, and socioeconomic status. Our evaluation of multiple open-weight LLMs reveals that all models exhibit some degree of stereotypical bias, with disability related biases being notably persistent, and religion bias generally lower likely due to global debiasing efforts demonstrating the need for fairer model development. 

---
# Words That Make Language Models Perceive 

**Authors**: Sophie L. Wang, Phillip Isola, Brian Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2510.02425)  

**Abstract**: Large language models (LLMs) trained purely on text ostensibly lack any direct perceptual experience, yet their internal representations are implicitly shaped by multimodal regularities encoded in language. We test the hypothesis that explicit sensory prompting can surface this latent structure, bringing a text-only LLM into closer representational alignment with specialist vision and audio encoders. When a sensory prompt tells the model to 'see' or 'hear', it cues the model to resolve its next-token predictions as if they were conditioned on latent visual or auditory evidence that is never actually supplied. Our findings reveal that lightweight prompt engineering can reliably activate modality-appropriate representations in purely text-trained LLMs. 

---
# KnowledgeSmith: Uncovering Knowledge Updating in LLMs with Model Editing and Unlearning 

**Authors**: Yinyi Luo, Zhexian Zhou, Hao Chen, Kai Qiu, Marios Savvides, Yixuan Li, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02392)  

**Abstract**: Knowledge editing and machine unlearning are two popular approaches for large language models (LLMs) to stay up-to-date. However, the knowledge updating mechanism of LLMs remains largely unexplored due to insufficient, isolated, and small-scale evaluation. For instance, are LLMs similar to humans in modifying certain knowledge? What differs editing and unlearning as training data increases? This paper proposes KnowledgeSmith, a unified framework to systematically understand the updating mechanism of LLMs. We first cast editing and unlearning as instances of one constrained optimization problem. Then, we propose an automatic dataset generator that provides structured interventions across multiple graph levels and data scales, enabling controlled studies of how different modification strategies propagate through model knowledge. Extensive experiments demonstrate nuanced insights over knowledge propagation, plasticity scaling, consistency, and robustness. For instance, our results show that LLMs do not exhibit similar updating as humans for different levels of knowledge, and there exists consistency-capacity trade-off. We hope our findings can offer suggestions to the design of more reliable and scalable strategies. Code: this https URL 

---
# Uncertainty-Aware Answer Selection for Improved Reasoning in Multi-LLM Systems 

**Authors**: Aakriti Agrawal, Rohith Aralikatti, Anirudh Satheesh, Souradip Chakraborty, Amrit Singh Bedi, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02377)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities, yet selecting the most reliable response from multiple LLMs remains a challenge, particularly in resource-constrained settings. Existing approaches often depend on costly external verifiers, human evaluators, or self-consistency techniques that require multiple samples from a single model. While multi-LLM systems produce more diverse responses than single models and thus have greater potential, they often underperform compared to single LLM self-consistency. We propose a principled, novel and computationally efficient method to select the best response from multiple different LLMs using a calibrated log-likelihood score, implicitly leveraging the inherent knowledge and confidence of these models. Our method demonstrates improvements of approx. 4%, 3%, and 5% across both debate (multi-round LLM discussions) and non-debate (Best-of-N with multiple LLMs) settings on GSM8K, MMLU (6 subsets), and ARC datasets respectively. 

---
# SoT: Structured-of-Thought Prompting Guides Multilingual Reasoning in Large Language Models 

**Authors**: Rui Qi, Zhibo Man, Yufeng Chen, Fengran Mo, Jinan Xu, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02648)  

**Abstract**: Recent developments have enabled Large Language Models (LLMs) to engage in complex reasoning tasks through deep thinking. However, the capacity of reasoning has not been successfully transferred to non-high-resource languages due to resource constraints, which struggles with multilingual reasoning tasks. To this end, we propose Structured-of-Thought (SoT), a training-free method that improves the performance on multilingual reasoning through a multi-step transformation: Language Thinking Transformation and Structured Knowledge Transformation. The SoT method converts language-specific semantic information into language-agnostic structured representations, enabling the models to understand the query in different languages more sophisticated. Besides, SoT effectively guides LLMs toward more concentrated reasoning to maintain consistent underlying reasoning pathways when handling cross-lingual variations in expression. Experimental results demonstrate that SoT outperforms several strong baselines on multiple multilingual reasoning benchmarks when adapting to various backbones of LLMs. It can also be integrated with other training-free strategies for further improvements. Our code is available at this https URL. 

---
# An Senegalese Legal Texts Structuration Using LLM-augmented Knowledge Graph 

**Authors**: Oumar Kane, Mouhamad M. Allaya, Dame Samb, Mamadou Bousso  

**Link**: [PDF](https://arxiv.org/pdf/2510.02353)  

**Abstract**: This study examines the application of artificial intelligence (AI) and large language models (LLM) to improve access to legal texts in Senegal's judicial system. The emphasis is on the difficulties of extracting and organizing legal documents, highlighting the need for better access to judicial information. The research successfully extracted 7,967 articles from various legal documents, particularly focusing on the Land and Public Domain Code. A detailed graph database was developed, which contains 2,872 nodes and 10,774 relationships, aiding in the visualization of interconnections within legal texts. In addition, advanced triple extraction techniques were utilized for knowledge, demonstrating the effectiveness of models such as GPT-4o, GPT-4, and Mistral-Large in identifying relationships and relevant metadata. Through these technologies, the aim is to create a solid framework that allows Senegalese citizens and legal professionals to more effectively understand their rights and responsibilities. 

---
# DRIFT: Learning from Abundant User Dissatisfaction in Real-World Preference Learning 

**Authors**: Yifan Wang, Bolian Li, Junlin Wu, Zhaoxuan Tan, Zheli Liu, Ruqi Zhang, Ananth Grama, Qingkai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02341)  

**Abstract**: Real-world large language model deployments (e.g., conversational AI systems, code generation assistants) naturally generate abundant implicit user dissatisfaction (DSAT) signals, as users iterate toward better answers through refinements, corrections, and expressed preferences, while explicit satisfaction (SAT) feedback is scarce. Existing preference learning approaches are poorly aligned with this data profile, as they rely on costly human annotations or assume plentiful positive responses. In this paper, we introduce \textbf{DRIFT} (\textbf{D}issatisfaction-\textbf{R}efined \textbf{I}terative pre\textbf{F}erence \textbf{T}raining), which anchors training on real-world DSAT signals and samples positives dynamically from the evolving policy. Empirically, DRIFT models trained on real-world \textit{WildFeedback} datasets and synthetic \textit{UltraFeedback} datasets achieve up to +6.23\% (7B) / +7.61\% (14B) on WildBench Task Score and up to +8.95\% (7B) / +12.29\% (14B) on AlpacaEval2 win rate over base models, outperforming strong baseline methods such as iterative DPO and SPIN. At larger scales, the improvements are particularly pronounced: 14B models trained with DRIFT surpass GPT-4o-mini on WildBench. Further analysis shows that DRIFT also preserves exploratory capacity, yielding more diverse high-reward solutions rather than collapsing to narrow subsets. Theoretically, we demonstrate that this design preserves preference margins and avoids the gradient degeneration. These results show that DRIFT is an effective and scalable recipe for real-world post-training that leverages the most abundant and informative signal. The code and data are available at this https URL. 

---
# Can Prompts Rewind Time for LLMs? Evaluating the Effectiveness of Prompted Knowledge Cutoffs 

**Authors**: Xin Gao, Ruiyi Zhang, Daniel Du, Saurabh Mahindre, Sai Ashish Somayajula, Pengtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.02340)  

**Abstract**: Large Language Models (LLMs) are widely used for temporal prediction, but their reliance on pretraining data raises contamination concerns, as accurate predictions on pre-cutoff test data may reflect memorization rather than reasoning, leading to an overestimation of their generalization capability. With the recent emergence of prompting-based unlearning techniques, a natural question arises: Can LLMs be prompted to simulate an earlier knowledge cutoff? In this work, we investigate the capability of prompting to simulate earlier knowledge cutoff in LLMs. We construct three evaluation datasets to assess the extent to which LLMs can forget (1) direct factual knowledge, (2) semantic shifts, and (3) causally related knowledge. Results demonstrate that while prompt-based simulated knowledge cutoffs show effectiveness when directly queried with the information after that date, they struggle to induce forgetting when the forgotten content is not directly asked but causally related to the query. These findings highlight the need for more rigorous evaluation settings when applying LLMs for temporal prediction tasks. The full dataset and evaluation code are available at this https URL. 

---
# Retrieval and Augmentation of Domain Knowledge for Text-to-SQL Semantic Parsing 

**Authors**: Manasi Patwardhan, Ayush Agarwal, Shabbirhussain Bhaisaheb, Aseem Arora, Lovekesh Vig, Sunita Sarawagi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02394)  

**Abstract**: The performance of Large Language Models (LLMs) for translating Natural Language (NL) queries into SQL varies significantly across databases (DBs). NL queries are often expressed using a domain specific vocabulary, and mapping these to the correct SQL requires an understanding of the embedded domain expressions, their relationship to the DB schema structure. Existing benchmarks rely on unrealistic, ad-hoc query specific textual hints for expressing domain knowledge. In this paper, we propose a systematic framework for associating structured domain statements at the database level. We present retrieval of relevant structured domain statements given a user query using sub-string level match. We evaluate on eleven realistic DB schemas covering diverse domains across five open-source and proprietary LLMs and demonstrate that (1) DB level structured domain statements are more practical and accurate than existing ad-hoc query specific textual domain statements, and (2) Our sub-string match based retrieval of relevant domain statements provides significantly higher accuracy than other retrieval approaches. 

---
# When Names Disappear: Revealing What LLMs Actually Understand About Code 

**Authors**: Cuong Chi Le, Minh V.T. Pham, Cuong Duc Van, Hoang N. Phan, Huy N. Phan, Tien N. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.03178)  

**Abstract**: Large Language Models (LLMs) achieve strong results on code tasks, but how they derive program meaning remains unclear. We argue that code communicates through two channels: structural semantics, which define formal behavior, and human-interpretable naming, which conveys intent. Removing the naming channel severely degrades intent-level tasks such as summarization, where models regress to line-by-line descriptions. Surprisingly, we also observe consistent reductions on execution tasks that should depend only on structure, revealing that current benchmarks reward memorization of naming patterns rather than genuine semantic reasoning. To disentangle these effects, we introduce a suite of semantics-preserving obfuscations and show that they expose identifier leakage across both summarization and execution. Building on these insights, we release ClassEval-Obf, an obfuscation-enhanced benchmark that systematically suppresses naming cues while preserving behavior. Our results demonstrate that ClassEval-Obf reduces inflated performance gaps, weakens memorization shortcuts, and provides a more reliable basis for assessing LLMs' code understanding and generalization. 

---
