# YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models 

**Authors**: Lei Wang, Heyang Gao, Xiaohe Bo, Xu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07581)  

**Abstract**: Leveraging large language model (LLM) based agents to simulate human social behaviors has recently gained significant attention. In this paper, we introduce a novel social simulator called YuLan-OneSim. Compared to previous works, YuLan-OneSim distinguishes itself in five key aspects: (1) Code-free scenario construction: Users can simply describe and refine their simulation scenarios through natural language interactions with our simulator. All simulation code is automatically generated, significantly reducing the need for programming expertise. (2) Comprehensive default scenarios: We implement 50 default simulation scenarios spanning 8 domains, including economics, sociology, politics, psychology, organization, demographics, law, and communication, broadening access for a diverse range of social researchers. (3) Evolvable simulation: Our simulator is capable of receiving external feedback and automatically fine-tuning the backbone LLMs, significantly enhancing the simulation quality. (4) Large-scale simulation: By developing a fully responsive agent framework and a distributed simulation architecture, our simulator can handle up to 100,000 agents, ensuring more stable and reliable simulation results. (5) AI social researcher: Leveraging the above features, we develop an AI social researcher. Users only need to propose a research topic, and the AI researcher will automatically analyze the input, construct simulation environments, summarize results, generate technical reports, review and refine the reports--completing the social science research loop. To demonstrate the advantages of YuLan-OneSim, we conduct experiments to evaluate the quality of the automatically generated scenarios, the reliability, efficiency, and scalability of the simulation process, as well as the performance of the AI social researcher. 

---
# A Survey on Collaborative Mechanisms Between Large and Small Language Models 

**Authors**: Yi Chen, JiaHao Zhao, HaoHao Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07460)  

**Abstract**: Large Language Models (LLMs) deliver powerful AI capabilities but face deployment challenges due to high resource costs and latency, whereas Small Language Models (SLMs) offer efficiency and deployability at the cost of reduced performance. Collaboration between LLMs and SLMs emerges as a crucial paradigm to synergistically balance these trade-offs, enabling advanced AI applications, especially on resource-constrained edge devices. This survey provides a comprehensive overview of LLM-SLM collaboration, detailing various interaction mechanisms (pipeline, routing, auxiliary, distillation, fusion), key enabling technologies, and diverse application scenarios driven by on-device needs like low latency, privacy, personalization, and offline operation. While highlighting the significant potential for creating more efficient, adaptable, and accessible AI, we also discuss persistent challenges including system overhead, inter-model consistency, robust task allocation, evaluation complexity, and security/privacy concerns. Future directions point towards more intelligent adaptive frameworks, deeper model fusion, and expansion into multimodal and embodied AI, positioning LLM-SLM collaboration as a key driver for the next generation of practical and ubiquitous artificial intelligence. 

---
# Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving 

**Authors**: Xinji Mai, Haotian Xu, Xing W, Weinong Wang, Yingying Zhang, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07773)  

**Abstract**: Large Language Models (LLMs) often struggle with mathematical reasoning tasks requiring precise, verifiable computation. While Reinforcement Learning (RL) from outcome-based rewards enhances text-based reasoning, understanding how agents autonomously learn to leverage external tools like code execution remains crucial. We investigate RL from outcome-based rewards for Tool-Integrated Reasoning, ZeroTIR, training base LLMs to spontaneously generate and execute Python code for mathematical problems without supervised tool-use examples. Our central contribution is we demonstrate that as RL training progresses, key metrics scale predictably. Specifically, we observe strong positive correlations where increased training steps lead to increases in the spontaneous code execution frequency, the average response length, and, critically, the final task accuracy. This suggests a quantifiable relationship between computational effort invested in training and the emergence of effective, tool-augmented reasoning strategies. We implement a robust framework featuring a decoupled code execution environment and validate our findings across standard RL algorithms and frameworks. Experiments show ZeroTIR significantly surpasses non-tool ZeroRL baselines on challenging math benchmarks. Our findings provide a foundational understanding of how autonomous tool use is acquired and scales within Agent RL, offering a reproducible benchmark for future studies. Code is released at \href{this https URL}{this https URL}. 

---
# How well do LLMs reason over tabular data, really? 

**Authors**: Cornelius Wolff, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.07453)  

**Abstract**: Large Language Models (LLMs) excel in natural language tasks, but less is known about their reasoning capabilities over tabular data. Prior analyses devise evaluation strategies that poorly reflect an LLM's realistic performance on tabular queries. Moreover, we have a limited understanding of the robustness of LLMs towards realistic variations in tabular inputs. Therefore, we ask: Can general-purpose LLMs reason over tabular data, really?, and focus on two questions 1) are tabular reasoning capabilities of general-purpose LLMs robust to real-world characteristics of tabular inputs, and 2) how can we realistically evaluate an LLM's performance on analytical tabular queries? Building on a recent tabular reasoning benchmark, we first surface shortcomings of its multiple-choice prompt evaluation strategy, as well as commonly used free-form text metrics such as SacreBleu and BERT-score. We show that an LLM-as-a-judge procedure yields more reliable performance insights and unveil a significant deficit in tabular reasoning performance of LLMs. We then extend the tabular inputs reflecting three common characteristics in practice: 1) missing values, 2) duplicate entities, and 3) structural variations. Experiments show that the tabular reasoning capabilities of general-purpose LLMs suffer from these variations, stressing the importance of improving their robustness for realistic tabular inputs. 

---
# Measuring General Intelligence with Generated Games 

**Authors**: Vivek Verma, David Huang, William Chen, Dan Klein, Nicholas Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07215)  

**Abstract**: We present gg-bench, a collection of game environments designed to evaluate general reasoning capabilities in language models. Unlike most static benchmarks, gg-bench is a data generating process where new evaluation instances can be generated at will. In particular, gg-bench is synthetically generated by (1) using a large language model (LLM) to generate natural language descriptions of novel games, (2) using the LLM to implement each game in code as a Gym environment, and (3) training reinforcement learning (RL) agents via self-play on the generated games. We evaluate language models by their winrate against these RL agents by prompting models with the game description, current board state, and a list of valid moves, after which models output the moves they wish to take. gg-bench is challenging: state-of-the-art LLMs such as GPT-4o and Claude 3.7 Sonnet achieve winrates of 7-9% on gg-bench using in-context learning, while reasoning models such as o1, o3-mini and DeepSeek-R1 achieve average winrates of 31-36%. We release the generated games, data generation process, and evaluation code in order to support future modeling work and expansion of our benchmark. 

---
# RefPentester: A Knowledge-Informed Self-Reflective Penetration Testing Framework Based on Large Language Models 

**Authors**: Hanzheng Dai, Yuanliang Li, Zhibo Zhang, Jun Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07089)  

**Abstract**: Automated penetration testing (AutoPT) powered by large language models (LLMs) has gained attention for its ability to automate ethical hacking processes and identify vulnerabilities in target systems by leveraging the intrinsic knowledge of LLMs. However, existing LLM-based AutoPT frameworks often underperform compared to human experts in challenging tasks for several reasons: the imbalanced knowledge used in LLM training, short-sighted planning in the planning process, and hallucinations during command generation. In addition, the penetration testing (PT) process, with its trial-and-error nature, is limited by existing frameworks that lack mechanisms to learn from previous failed operations, restricting adaptive improvement of PT strategies. To address these limitations, we propose a knowledge-informed self-reflective PT framework powered by LLMs, called RefPentester, which is an AutoPT framework designed to assist human operators in identifying the current stage of the PT process, selecting appropriate tactic and technique for the stage, choosing suggested action, providing step-by-step operational guidance, and learning from previous failed operations. We also modeled the PT process as a seven-state Stage Machine to integrate the proposed framework effectively. The evaluation shows that RefPentester can successfully reveal credentials on Hack The Box's Sau machine, outperforming the baseline GPT-4o model by 16.7\%. Across PT stages, RefPentester also demonstrates superior success rates on PT stage transitions. 

---
# Architectural Precedents for General Agents using Large Language Models 

**Authors**: Robert E. Wray, James R. Kirk, John E. Laird  

**Link**: [PDF](https://arxiv.org/pdf/2505.07087)  

**Abstract**: One goal of AI (and AGI) is to identify and understand specific mechanisms and representations sufficient for general intelligence. Often, this work manifests in research focused on architectures and many cognitive architectures have been explored in AI/AGI. However, different research groups and even different research traditions have somewhat independently identified similar/common patterns of processes and representations or cognitive design patterns that are manifest in existing architectures. Today, AI systems exploiting large language models (LLMs) offer a relatively new combination of mechanism and representation available for exploring the possibilities of general intelligence. In this paper, we summarize a few recurring cognitive design patterns that have appeared in various pre-transformer AI architectures. We then explore how these patterns are evident in systems using LLMs, especially for reasoning and interactive ("agentic") use cases. By examining and applying these recurring patterns, we can also predict gaps or deficiencies in today's Agentic LLM Systems and identify likely subjects of future research towards general intelligence using LLMs and other generative foundation models. 

---
# DialogueReason: Rule-Based RL Sparks Dialogue Reasoning in LLMs 

**Authors**: Yubo Shu, Zhewei Huang, Xin Wu, Chen Hu, Shuchang Zhou, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07049)  

**Abstract**: We propose DialogueReason, a reasoning paradigm that uncovers the lost roles in monologue-style reasoning models, aiming to boost diversity and coherency of the reasoning process. Recent advances in RL-based large reasoning models have led to impressive long CoT capabilities and high performance on math and science benchmarks. However, these reasoning models rely mainly on monologue-style reasoning, which often limits reasoning diversity and coherency, frequently recycling fixed strategies or exhibiting unnecessary shifts in attention. Our work consists of an analysis of monologue reasoning patterns and the development of a dialogue-based reasoning approach. We first introduce the Compound-QA task, which concatenates multiple problems into a single prompt to assess both diversity and coherency of reasoning. Our analysis shows that Compound-QA exposes weaknesses in monologue reasoning, evidenced by both quantitative metrics and qualitative reasoning traces. Building on the analysis, we propose a dialogue-based reasoning, named DialogueReason, structured around agents, environment, and interactions. Using PPO with rule-based rewards, we train open-source LLMs (Qwen-QWQ and Qwen-Base) to adopt dialogue reasoning. We evaluate trained models on MATH, AIME, and GPQA datasets, showing that the dialogue reasoning model outperforms monologue models under more complex compound questions. Additionally, we discuss how dialogue-based reasoning helps enhance interpretability, facilitate more intuitive human interaction, and inspire advances in multi-agent system design. 

---
# LLM-Augmented Chemical Synthesis and Design Decision Programs 

**Authors**: Haorui Wang, Jeff Guo, Lingkai Kong, Rampi Ramprasad, Philippe Schwaller, Yuanqi Du, Chao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07027)  

**Abstract**: Retrosynthesis, the process of breaking down a target molecule into simpler precursors through a series of valid reactions, stands at the core of organic chemistry and drug development. Although recent machine learning (ML) research has advanced single-step retrosynthetic modeling and subsequent route searches, these solutions remain restricted by the extensive combinatorial space of possible pathways. Concurrently, large language models (LLMs) have exhibited remarkable chemical knowledge, hinting at their potential to tackle complex decision-making tasks in chemistry. In this work, we explore whether LLMs can successfully navigate the highly constrained, multi-step retrosynthesis planning problem. We introduce an efficient scheme for encoding reaction pathways and present a new route-level search strategy, moving beyond the conventional step-by-step reactant prediction. Through comprehensive evaluations, we show that our LLM-augmented approach excels at retrosynthesis planning and extends naturally to the broader challenge of synthesizable molecular design. 

---
# From Knowledge to Reasoning: Evaluating LLMs for Ionic Liquids Research in Chemical and Biological Engineering 

**Authors**: Gaurab Sarkar, Sougata Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.06964)  

**Abstract**: Although Large Language Models (LLMs) have achieved remarkable performance in diverse general knowledge and reasoning tasks, their utility in the scientific domain of Chemical and Biological Engineering (CBE) is unclear. Hence, it necessitates challenging evaluation benchmarks that can measure LLM performance in knowledge- and reasoning-based tasks, which is lacking. As a foundational step, we empirically measure the reasoning capabilities of LLMs in CBE. We construct and share an expert-curated dataset of 5,920 examples for benchmarking LLMs' reasoning capabilities in the niche domain of Ionic Liquids (ILs) for carbon sequestration, an emergent solution to reducing global warming. The dataset presents different difficulty levels by varying along the dimensions of linguistic and domain-specific knowledge. Benchmarking three less than 10B parameter open-source LLMs on the dataset suggests that while smaller general-purpose LLMs are knowledgeable about ILs, they lack domain-specific reasoning capabilities. Based on our results, we further discuss considerations for leveraging LLMs for carbon capture research using ILs. Since LLMs have a high carbon footprint, gearing them for IL research can symbiotically benefit both fields and help reach the ambitious carbon neutrality target by 2050. Dataset link: this https URL 

---
# Towards Artificial General or Personalized Intelligence? A Survey on Foundation Models for Personalized Federated Intelligence 

**Authors**: Yu Qiao, Huy Q. Le, Avi Deb Raha, Phuong-Nam Tran, Apurba Adhikary, Mengchun Zhang, Loc X. Nguyen, Eui-Nam Huh, Dusit Niyato, Choong Seon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06907)  

**Abstract**: The rise of large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, has reshaped the artificial intelligence landscape. As prominent examples of foundational models (FMs) built on LLMs, these models exhibit remarkable capabilities in generating human-like content, bringing us closer to achieving artificial general intelligence (AGI). However, their large-scale nature, sensitivity to privacy concerns, and substantial computational demands present significant challenges to personalized customization for end users. To bridge this gap, this paper presents the vision of artificial personalized intelligence (API), focusing on adapting these powerful models to meet the specific needs and preferences of users while maintaining privacy and efficiency. Specifically, this paper proposes personalized federated intelligence (PFI), which integrates the privacy-preserving advantages of federated learning (FL) with the zero-shot generalization capabilities of FMs, enabling personalized, efficient, and privacy-protective deployment at the edge. We first review recent advances in both FL and FMs, and discuss the potential of leveraging FMs to enhance federated systems. We then present the key motivations behind realizing PFI and explore promising opportunities in this space, including efficient PFI, trustworthy PFI, and PFI empowered by retrieval-augmented generation (RAG). Finally, we outline key challenges and future research directions for deploying FM-powered FL systems at the edge with improved personalization, computational efficiency, and privacy guarantees. Overall, this survey aims to lay the groundwork for the development of API as a complement to AGI, with a particular focus on PFI as a key enabling technique. 

---
# Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks 

**Authors**: Kai Xu, YiWei Mao, XinYi Guan, ZiLong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07473)  

**Abstract**: The application of large language models (LLMs) in the field of coding is evolving rapidly: from code assistants, to autonomous coding agents, and then to generating complete projects through natural language. Early LLM code benchmarks primarily focused on code generation accuracy, but these benchmarks have gradually become saturated. Benchmark saturation weakens their guiding role for LLMs. For example, HumanEval Pass@1 has reached 99.4% and MBPP 94.2%. Among various attempts to address benchmark saturation, approaches based on software engineering have stood out, but the saturation of existing software engineering benchmarks is rapidly increasing. To address this, we propose a new benchmark, Web-Bench, which contains 50 projects, each consisting of 20 tasks with sequential dependencies. The tasks implement project features in sequence, simulating real-world human development workflows. When designing Web-Bench, we aim to cover the foundational elements of Web development: Web Standards and Web Frameworks. Given the scale and complexity of these projects, which were designed by engineers with 5 to 10 years of experience, each presents a significant challenge. On average, a single project takes 4 to 8 hours for a senior engineer to complete. On our given benchmark agent (Web-Agent), SOTA (Claude 3.7 Sonnet) achieves only 25.1% Pass@1, significantly lower (better) than SWE-Bench's Verified (65.4%) and Full (33.8%) scores. Finally, we discuss that in any development field, Standards and Frameworks represent foundational knowledge and efficiency tools, respectively, and LLMs require optimization tailored to them. 

---
# KCluster: An LLM-based Clustering Approach to Knowledge Component Discovery 

**Authors**: Yumou Wei, Paulo Carvalho, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2505.06469)  

**Abstract**: Educators evaluate student knowledge using knowledge component (KC) models that map assessment questions to KCs. Still, designing KC models for large question banks remains an insurmountable challenge for instructors who need to analyze each question by hand. The growing use of Generative AI in education is expected only to aggravate this chronic deficiency of expert-designed KC models, as course engineers designing KCs struggle to keep up with the pace at which questions are generated. In this work, we propose KCluster, a novel KC discovery algorithm based on identifying clusters of congruent questions according to a new similarity metric induced by a large language model (LLM). We demonstrate in three datasets that an LLM can create an effective metric of question similarity, which a clustering algorithm can use to create KC models from questions with minimal human effort. Combining the strengths of LLM and clustering, KCluster generates descriptive KC labels and discovers KC models that predict student performance better than the best expert-designed models available. In anticipation of future work, we illustrate how KCluster can reveal insights into difficult KCs and suggest improvements to instruction. 

---
# Reliable Collaborative Conversational Agent System Based on LLMs and Answer Set Programming 

**Authors**: Yankai Zeng, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06438)  

**Abstract**: As the Large-Language-Model-driven (LLM-driven) Artificial Intelligence (AI) bots became popular, people realized their strong potential in Task-Oriented Dialogue (TOD). However, bots relying wholly on LLMs are unreliable in their knowledge, and whether they can finally produce a correct result for the task is not guaranteed. The collaboration among these agents also remains a challenge, since the necessary information to convey is unclear, and the information transfer is by prompts -- unreliable, and malicious knowledge is easy to inject. With the help of logic programming tools such as Answer Set Programming (ASP), conversational agents can be built safely and reliably, and communication among the agents made more efficient and secure. We proposed an Administrator-Assistant Dual-Agent paradigm, where the two ASP-driven bots share the same knowledge base and complete their tasks independently, while the information can be passed by a Collaborative Rule Set (CRS). The knowledge and information conveyed are encapsulated and invisible to the users, ensuring the security of information transmission. We have constructed AutoManager, a dual-agent system for managing the drive-through window of a fast-food restaurant such as Taco Bell in the US. In AutoManager, the assistant bot takes the customer's order while the administrator bot manages the menu and food supply. We evaluated our AutoManager and compared it with the real-world Taco Bell Drive-Thru AI Order Taker, and the results show that our method is more reliable. 

---
# Learning Dynamics in Continual Pre-Training for Large Language Models 

**Authors**: Xingjin Wang, Howe Tissue, Lu Wang, Linjing Li, Daniel Dajun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07796)  

**Abstract**: Continual Pre-Training (CPT) has become a popular and effective method to apply strong foundation models to specific downstream tasks. In this work, we explore the learning dynamics throughout the CPT process for large language models. We specifically focus on how general and downstream domain performance evolves at each training step, with domain performance measured via validation losses. We have observed that the CPT loss curve fundamentally characterizes the transition from one curve to another hidden curve, and could be described by decoupling the effects of distribution shift and learning rate annealing. We derive a CPT scaling law that combines the two factors, enabling the prediction of loss at any (continual) training steps and across learning rate schedules (LRS) in CPT. Our formulation presents a comprehensive understanding of several critical factors in CPT, including loss potential, peak learning rate, training steps, replay ratio, etc. Moreover, our approach can be adapted to customize training hyper-parameters to different CPT goals such as balancing general and domain-specific performance. Extensive experiments demonstrate that our scaling law holds across various CPT datasets and training hyper-parameters. 

---
# Text-to-CadQuery: A New Paradigm for CAD Generation with Scalable Large Model Capabilities 

**Authors**: Haoyang Xie, Feng Ju  

**Link**: [PDF](https://arxiv.org/pdf/2505.06507)  

**Abstract**: Computer-aided design (CAD) is fundamental to modern engineering and manufacturing, but creating CAD models still requires expert knowledge and specialized software. Recent advances in large language models (LLMs) open up the possibility of generative CAD, where natural language is directly translated into parametric 3D models. However, most existing methods generate task-specific command sequences that pretrained models cannot directly handle. These sequences must be converted into CAD representations such as CAD vectors before a 3D model can be produced, which requires training models from scratch and adds unnecessary complexity. To tackle this issue, we propose generating CadQuery code directly from text, leveraging the strengths of pretrained LLMs to produce 3D models without intermediate representations, using this Python-based scripting language. Since LLMs already excel at Python generation and spatial reasoning, fine-tuning them on Text-to-CadQuery data proves highly effective. Given that these capabilities typically improve with scale, we hypothesize that larger models will perform better after fine-tuning. To enable this, we augment the Text2CAD dataset with 170,000 CadQuery annotations. We fine-tune six open-source LLMs of varying sizes and observe consistent improvements. Our best model achieves a top-1 exact match of 69.3%, up from 58.8%, and reduces Chamfer Distance by 48.6%. Project page: this https URL. 

---
# Enhancing Code Generation via Bidirectional Comment-Level Mutual Grounding 

**Authors**: Yifeng Di, Tianyi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07768)  

**Abstract**: Large Language Models (LLMs) have demonstrated unprecedented capability in code generation. However, LLM-generated code is still plagued with a wide range of functional errors, especially for complex programming tasks that LLMs have not seen before. Recent studies have shown that developers often struggle with inspecting and fixing incorrect code generated by LLMs, diminishing their productivity and trust in LLM-based code generation. Inspired by the mutual grounding theory in communication, we propose an interactive approach that leverages code comments as a medium for developers and LLMs to establish a shared understanding. Our approach facilitates iterative grounding by interleaving code generation, inline comment generation, and contextualized user feedback through editable comments to align generated code with developer intent. We evaluated our approach on two popular benchmarks and demonstrated that our approach significantly improved multiple state-of-the-art LLMs, e.g., 17.1% pass@1 improvement for code-davinci-002 on HumanEval. Furthermore, we conducted a user study with 12 participants in comparison to two baselines: (1) interacting with GitHub Copilot, and (2) interacting with a multi-step code generation paradigm called Multi-Turn Program Synthesis. Participants completed the given programming tasks 16.7% faster and with 10.5% improvement in task success rate when using our approach. Both results show that interactively refining code comments enables the collaborative establishment of mutual grounding, leading to more accurate code generation and higher developer confidence. 

---
# Circuit Partitioning Using Large Language Models for Quantum Compilation and Simulations 

**Authors**: Pranav Sinha, Sumit Kumar Jha, Sunny Raj  

**Link**: [PDF](https://arxiv.org/pdf/2505.07711)  

**Abstract**: We are in the midst of the noisy intermediate-scale quantum (NISQ) era, where quantum computers are limited by noisy gates, some of which are more error-prone than others and can render the final computation incomprehensible. Quantum circuit compilation algorithms attempt to minimize these noisy gates when mapping quantum algorithms onto quantum hardware but face computational challenges that restrict their application to circuits with no more than 5-6 qubits, necessitating the need to partition large circuits before the application of noisy quantum gate minimization algorithms. The existing generation of these algorithms is heuristic in nature and does not account for downstream gate minimization tasks. Large language models (LLMs) have the potential to change this and help improve quantum circuit partitions. This paper investigates the use of LLMs, such as Llama and Mistral, for partitioning quantum circuits by capitalizing on their abilities to understand and generate code, including QASM. Specifically, we teach LLMs to partition circuits using the quick partition approach of the Berkeley Quantum Synthesis Toolkit. Through experimental evaluations, we show that careful fine-tuning of open source LLMs enables us to obtain an accuracy of 53.4% for the partition task while over-the-shelf LLMs are unable to correctly partition circuits, using standard 1-shot and few-shot training approaches. 

---
# OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit 

**Authors**: Arun S. Maiya  

**Link**: [PDF](https://arxiv.org/pdf/2505.07672)  

**Abstract**: We present this http URL, a Python-based toolkit for applying large language models (LLMs) to sensitive, non-public data in offline or restricted environments. The system is designed for privacy-preserving use cases and provides prebuilt pipelines for document processing and storage, retrieval-augmented generation (RAG), information extraction, summarization, classification, and prompt/output processing with minimal configuration. this http URL supports multiple LLM backends -- including this http URL, Ollama, vLLM, and Hugging Face Transformers -- with quantized model support, GPU acceleration, and seamless backend switching. Although designed for fully local execution, this http URL also supports integration with a wide range of cloud LLM providers when permitted, enabling hybrid deployments that balance performance with data control. A no-code web interface extends accessibility to non-technical users. 

---
# A Case Study Investigating the Role of Generative AI in Quality Evaluations of Epics in Agile Software Development 

**Authors**: Werner Geyer, Jessica He, Daita Sarkar, Michelle Brachman, Chris Hammond, Jennifer Heins, Zahra Ashktorab, Carlos Rosemberg, Charlie Hill  

**Link**: [PDF](https://arxiv.org/pdf/2505.07664)  

**Abstract**: The broad availability of generative AI offers new opportunities to support various work domains, including agile software development. Agile epics are a key artifact for product managers to communicate requirements to stakeholders. However, in practice, they are often poorly defined, leading to churn, delivery delays, and cost overruns. In this industry case study, we investigate opportunities for large language models (LLMs) to evaluate agile epic quality in a global company. Results from a user study with 17 product managers indicate how LLM evaluations could be integrated into their work practices, including perceived values and usage in improving their epics. High levels of satisfaction indicate that agile epics are a new, viable application of AI evaluations. However, our findings also outline challenges, limitations, and adoption barriers that can inform both practitioners and researchers on the integration of such evaluations into future agile work practices. 

---
# MiMo: Unlocking the Reasoning Potential of Language Model -- From Pretraining to Posttraining 

**Authors**: Xiaomi LLM-Core Team, Bingquan Xia, Bowen Shen, Cici, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, Liang Zhao, Peidian Li, Peng Wang, Shihua Yu, Shimao Chen, Weikun Wang, Wenhan Ma, Xiangwei Deng, Yi Huang, Yifan Song, Zihan Jiang, Bowen Ye, Can Cai, Chenhong He, Dong Zhang, Duo Zhang, Guoan Wang, Hao Tian, Haochen Zhao, Heng Qu, Hongshen Xu, Jun Shi, Kainan Bao, QingKai Fang, Kang Zhou, Kangyang Zhou, Lei Li, Menghang Zhu, Nuo Chen, Qiantong Wang, Shaohui Liu, Shicheng Li, Shuhao Gu, Shuhuai Ren, Shuo Liu, Sirui Deng, Weiji Zhuang, Weiwei Lv, Wenyu Yang, Xin Zhang, Xing Yong, Xing Zhang, Xingchen Song, Xinzhe Xu, Xu Wang, Yihan Yan, Yu Tu, Yuanyuan Tian, Yudong Wang, Yue Yu, Zhenru Lin, Zhichao Song, Zihao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.07608)  

**Abstract**: We present MiMo-7B, a large language model born for reasoning tasks, with optimization across both pre-training and post-training stages. During pre-training, we enhance the data preprocessing pipeline and employ a three-stage data mixing strategy to strengthen the base model's reasoning potential. MiMo-7B-Base is pre-trained on 25 trillion tokens, with additional Multi-Token Prediction objective for enhanced performance and accelerated inference speed. During post-training, we curate a dataset of 130K verifiable mathematics and programming problems for reinforcement learning, integrating a test-difficulty-driven code-reward scheme to alleviate sparse-reward issues and employing strategic data resampling to stabilize training. Extensive evaluations show that MiMo-7B-Base possesses exceptional reasoning potential, outperforming even much larger 32B models. The final RL-tuned model, MiMo-7B-RL, achieves superior performance on mathematics, code and general reasoning tasks, surpassing the performance of OpenAI o1-mini. The model checkpoints are available at this https URL. 

---
# Concept-Level Explainability for Auditing & Steering LLM Responses 

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady  

**Link**: [PDF](https://arxiv.org/pdf/2505.07610)  

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior. 

---
# Characterizing the Investigative Methods of Fictional Detectives with Large Language Models 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Bruno Feijó, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2505.07601)  

**Abstract**: Detective fiction, a genre defined by its complex narrative structures and character-driven storytelling, presents unique challenges for computational narratology, a research field focused on integrating literary theory into automated narrative generation. While traditional literary studies have offered deep insights into the methods and archetypes of fictional detectives, these analyses often focus on a limited number of characters and lack the scalability needed for the extraction of unique traits that can be used to guide narrative generation methods. In this paper, we present an AI-driven approach for systematically characterizing the investigative methods of fictional detectives. Our multi-phase workflow explores the capabilities of 15 Large Language Models (LLMs) to extract, synthesize, and validate distinctive investigative traits of fictional detectives. This approach was tested on a diverse set of seven iconic detectives - Hercule Poirot, Sherlock Holmes, William Murdoch, Columbo, Father Brown, Miss Marple, and Auguste Dupin - capturing the distinctive investigative styles that define each character. The identified traits were validated against existing literary analyses and further tested in a reverse identification phase, achieving an overall accuracy of 91.43%, demonstrating the method's effectiveness in capturing the distinctive investigative approaches of each detective. This work contributes to the broader field of computational narratology by providing a scalable framework for character analysis, with potential applications in AI-driven interactive storytelling and automated narrative generation. 

---
# A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models 

**Authors**: Junjie Ye, Caishuang Huang, Zhuohan Chen, Wenjie Fu, Chenyuan Yang, Leyi Yang, Yilong Wu, Peng Wang, Meng Zhou, Xiaolong Yang, Tao Gui, Qi Zhang, Zhongchao Shi, Jianping Fan, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07591)  

**Abstract**: Instruction following evaluates large language models (LLMs) on their ability to generate outputs that adhere to user-defined constraints. However, existing benchmarks often rely on templated constraint prompts, which lack the diversity of real-world usage and limit fine-grained performance assessment. To fill this gap, we propose a multi-dimensional constraint framework encompassing three constraint patterns, four constraint categories, and four difficulty levels. Building on this framework, we develop an automated instruction generation pipeline that performs constraint expansion, conflict detection, and instruction rewriting, yielding 1,200 code-verifiable instruction-following test samples. We evaluate 19 LLMs across seven model families and uncover substantial variation in performance across constraint forms. For instance, average performance drops from 77.67% at Level I to 32.96% at Level IV. Furthermore, we demonstrate the utility of our approach by using it to generate data for reinforcement learning, achieving substantial gains in instruction following without degrading general performance. In-depth analysis indicates that these gains stem primarily from modifications in the model's attention modules parameters, which enhance constraint recognition and adherence. Code and data are available in this https URL. 

---
# GRADA: Graph-based Reranker against Adversarial Documents Attack 

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07546)  

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy. 

---
# LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning 

**Authors**: Xiaotian Lin, Yanlin Qi, Yizhang Zhu, Themis Palpanas, Chengliang Chai, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07437)  

**Abstract**: Instruction tuning has emerged as a critical paradigm for improving the capabilities and alignment of large language models (LLMs). However, existing iterative model-aware data selection methods incur significant computational overhead, as they rely on repeatedly performing full-dataset model inference to estimate sample utility for subsequent training iterations, creating a fundamental efficiency bottleneck. In this paper, we propose LEAD, an efficient iterative data selection framework that accurately estimates sample utility entirely within the standard training loop, eliminating the need for costly additional model inference. At its core, LEAD introduces Instance-Level Dynamic Uncertainty (IDU), a theoretically grounded utility function combining instantaneous training loss, gradient-based approximation of loss changes, and exponential smoothing of historical loss signals. To further scale efficiently to large datasets, LEAD employs a two-stage, coarse-to-fine selection strategy, adaptively prioritizing informative clusters through a multi-armed bandit mechanism, followed by precise fine-grained selection of high-utility samples using IDU. Extensive experiments across four diverse benchmarks show that LEAD significantly outperforms state-of-the-art methods, improving average model performance by 6.1%-10.8% while using only 2.5% of the training data and reducing overall training time by 5-10x. 

---
# Can Generative AI agents behave like humans? Evidence from laboratory market experiments 

**Authors**: R. Maria del Rio-Chanona, Marco Pangallo, Cars Hommes  

**Link**: [PDF](https://arxiv.org/pdf/2505.07457)  

**Abstract**: We explore the potential of Large Language Models (LLMs) to replicate human behavior in economic market experiments. Compared to previous studies, we focus on dynamic feedback between LLM agents: the decisions of each LLM impact the market price at the current step, and so affect the decisions of the other LLMs at the next step. We compare LLM behavior to market dynamics observed in laboratory settings and assess their alignment with human participants' behavior. Our findings indicate that LLMs do not adhere strictly to rational expectations, displaying instead bounded rationality, similarly to human participants. Providing a minimal context window i.e. memory of three previous time steps, combined with a high variability setting capturing response heterogeneity, allows LLMs to replicate broad trends seen in human experiments, such as the distinction between positive and negative feedback markets. However, differences remain at a granular level--LLMs exhibit less heterogeneity in behavior than humans. These results suggest that LLMs hold promise as tools for simulating realistic human behavior in economic contexts, though further research is needed to refine their accuracy and increase behavioral diversity. 

---
# Synthetic Code Surgery: Repairing Bugs and Vulnerabilities with LLMs and Synthetic Data 

**Authors**: David de-Fitero-Dominguez, Antonio Garcia-Cabot, Eva Garcia-Lopez  

**Link**: [PDF](https://arxiv.org/pdf/2505.07372)  

**Abstract**: This paper presents a novel methodology for enhancing Automated Program Repair (APR) through synthetic data generation utilizing Large Language Models (LLMs). Current APR systems are constrained by the limited availability of high-quality training data encompassing diverse bug types across multiple programming languages. The proposed approach addresses this limitation through a two-phase process: a synthetic sample generation followed by a rigorous quality assessment. Multiple state-of-the-art LLMs were employed to generate approximately 30,000 paired examples of buggy and fixed code across 12 programming languages and 13 bug categories. Subsequently, these samples underwent cross-model evaluation against five criteria: correctness, code quality, security, performance, and completeness. Experimental evaluation on the VulRepair test set dataset showed statistically significant improvements in Perfect Prediction rates, with the quality-filtered synthetic dataset outperforming both baseline and real-world commit data configurations in certain scenarios. The methodology was validated through rigorous statistical testing, including ANOVA and post-hoc Tukey's Honest Significant Difference analysis. Furthermore, the best-performing configurations surpassed existing systems despite using a less computationally intensive decoding strategy. This research establishes a self-bootstrapping paradigm in which LLMs generate and evaluate their own training data, potentially transforming approaches to data scarcity across software engineering tasks and advancing the development of robust, adaptable tools for automated code maintenance. 

---
# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent 

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07596)  

**Abstract**: Retrieval-augmented generation (RAG) is a common strategy to reduce hallucinations in Large Language Models (LLMs). While reinforcement learning (RL) can enable LLMs to act as search agents by activating retrieval capabilities, existing ones often underutilize their internal knowledge. This can lead to redundant retrievals, potential harmful knowledge conflicts, and increased inference latency. To address these limitations, an efficient and adaptive search agent capable of discerning optimal retrieval timing and synergistically integrating parametric (internal) and retrieved (external) knowledge is in urgent need. This paper introduces the Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could indentify its own knowledge boundary and prioritize the utilization of internal knowledge, resorting to external search only when internal knowledge is deemed insufficient. This is achieved using a novel knowledge-boundary aware reward function and a knowledge-boundary aware training dataset. These are designed for internal-external knowledge synergy oriented RL, incentivizing the model to deliver accurate answers, minimize unnecessary retrievals, and encourage appropriate external searches when its own knowledge is lacking. Evaluations across multiple knowledge reasoning tasks demonstrate that IKEA significantly outperforms baseline methods, reduces retrieval frequency significantly, and exhibits robust generalization capabilities. 

---
# No Query, No Access 

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07258)  

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.
Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at this https URL 

---
# SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models 

**Authors**: Peichao Lai, Kexuan Zhang, Yi Lin, Linyihan Zhang, Feiyang Ye, Jinhao Yan, Yanwei Xu, Conghui He, Yilei Wang, Wentao Zhang, Bin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.07247)  

**Abstract**: Subjective Answer Grading (SAG) plays a crucial role in education, standardized testing, and automated assessment systems, particularly for evaluating short-form responses in Short Answer Scoring (SAS). However, existing approaches often produce coarse-grained scores and lack detailed reasoning. Although large language models (LLMs) have demonstrated potential as zero-shot evaluators, they remain susceptible to bias, inconsistencies with human judgment, and limited transparency in scoring decisions. To overcome these limitations, we introduce SAS-Bench, a benchmark specifically designed for LLM-based SAS tasks. SAS-Bench provides fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams. This benchmark facilitates detailed evaluation of model reasoning processes and explainability. We also release an open-source dataset containing 1,030 questions and 4,109 student responses, each annotated by domain experts. Furthermore, we conduct comprehensive experiments with various LLMs, identifying major challenges in scoring science-related questions and highlighting the effectiveness of few-shot prompting in improving scoring accuracy. Our work offers valuable insights into the development of more robust, fair, and educationally meaningful LLM-based evaluation systems. 

---
# Comet: Accelerating Private Inference for Large Language Model by Predicting Activation Sparsity 

**Authors**: Guang Yan, Yuhui Zhang, Zimu Guo, Lutan Zhao, Xiaojun Chen, Chen Wang, Wenhao Wang, Dan Meng, Rui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07239)  

**Abstract**: With the growing use of large language models (LLMs) hosted on cloud platforms to offer inference services, privacy concerns about the potential leakage of sensitive information are escalating. Secure multi-party computation (MPC) is a promising solution to protect the privacy in LLM inference. However, MPC requires frequent inter-server communication, causing high performance overhead.
Inspired by the prevalent activation sparsity of LLMs, where most neuron are not activated after non-linear activation functions, we propose an efficient private inference system, Comet. This system employs an accurate and fast predictor to predict the sparsity distribution of activation function output. Additionally, we introduce a new private inference protocol. It efficiently and securely avoids computations involving zero values by exploiting the spatial locality of the predicted sparse distribution. While this computation-avoidance approach impacts the spatiotemporal continuity of KV cache entries, we address this challenge with a low-communication overhead cache refilling strategy that merges miss requests and incorporates a prefetching mechanism. Finally, we evaluate Comet on four common LLMs and compare it with six state-of-the-art private inference systems. Comet achieves a 1.87x-2.63x speedup and a 1.94x-2.64x communication reduction. 

---
# UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Faryal Batool, Jeffrin Sam, Artem Lykov, Chih-Yung Wen, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07236)  

**Abstract**: We present UAV-CodeAgents, a scalable multi-agent framework for autonomous UAV mission generation, built on large language and vision-language models (LLMs/VLMs). The system leverages the ReAct (Reason + Act) paradigm to interpret satellite imagery, ground high-level natural language instructions, and collaboratively generate UAV trajectories with minimal human supervision. A core component is a vision-grounded, pixel-pointing mechanism that enables precise localization of semantic targets on aerial maps. To support real-time adaptability, we introduce a reactive thinking loop, allowing agents to iteratively reflect on observations, revise mission goals, and coordinate dynamically in evolving environments.
UAV-CodeAgents is evaluated on large-scale mission scenarios involving industrial and environmental fire detection. Our results show that a lower decoding temperature (0.5) yields higher planning reliability and reduced execution time, with an average mission creation time of 96.96 seconds and a success rate of 93%. We further fine-tune Qwen2.5VL-7B on 9,000 annotated satellite images, achieving strong spatial grounding across diverse visual categories. To foster reproducibility and future research, we will release the full codebase and a novel benchmark dataset for vision-language-based UAV planning. 

---
# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07233)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker, which refines retrieved documents to enhance generation quality and explainability. The challenge of selecting the optimal number of documents (k) remains unsolved: too few may omit critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results. The model, data and code are available at this https URL 

---
# Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? 

**Authors**: Weixian Waylon Li, Hyeonjun Kim, Mihai Cucuringu, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07078)  

**Abstract**: Large Language Models (LLMs) have recently been leveraged for asset pricing tasks and stock trading applications, enabling AI agents to generate investment decisions from unstructured financial data. However, most evaluations of LLM timing-based investing strategies are conducted on narrow timeframes and limited stock universes, overstating effectiveness due to survivorship and data-snooping biases. We critically assess their generalizability and robustness by proposing FINSABER, a backtesting framework evaluating timing-based strategies across longer periods and a larger universe of symbols. Systematic backtests over two decades and 100+ symbols reveal that previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation. Our market regime analysis further demonstrates that LLM strategies are overly conservative in bull markets, underperforming passive benchmarks, and overly aggressive in bear markets, incurring heavy losses. These findings highlight the need to develop LLM strategies that are able to prioritise trend detection and regime-aware risk controls over mere scaling of framework complexity. 

---
# ParaView-MCP: An Autonomous Visualization Agent with Direct Tool Use 

**Authors**: Shusen Liu, Haichao Miao, Peer-Timo Bremer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07064)  

**Abstract**: While powerful and well-established, tools like ParaView present a steep learning curve that discourages many potential users. This work introduces ParaView-MCP, an autonomous agent that integrates modern multimodal large language models (MLLMs) with ParaView to not only lower the barrier to entry but also augment ParaView with intelligent decision support. By leveraging the state-of-the-art reasoning, command execution, and vision capabilities of MLLMs, ParaView-MCP enables users to interact with ParaView through natural language and visual inputs. Specifically, our system adopted the Model Context Protocol (MCP) - a standardized interface for model-application communication - that facilitates direct interaction between MLLMs with ParaView's Python API to allow seamless information exchange between the user, the language model, and the visualization tool itself. Furthermore, by implementing a visual feedback mechanism that allows the agent to observe the viewport, we unlock a range of new capabilities, including recreating visualizations from examples, closed-loop visualization parameter updates based on user-defined goals, and even cross-application collaboration involving multiple tools. Broadly, we believe such an agent-driven visualization paradigm can profoundly change the way we interact with visualization tools. We expect a significant uptake in the development of such visualization tools, in both visualization research and industry. 

---
# Semantic Retention and Extreme Compression in LLMs: Can We Have Both? 

**Authors**: Stanislas Laborde, Martin Cousseau, Antoun Yaacoub, Lionel Prevost  

**Link**: [PDF](https://arxiv.org/pdf/2505.07289)  

**Abstract**: The exponential growth in Large Language Model (LLM) deployment has intensified the need for efficient model compression techniques to reduce computational and memory costs. While pruning and quantization have shown promise, their combined potential remains largely unexplored. In this paper, we examine joint compression and how strategically combining pruning and quantization could yield superior performance-to-compression ratios compared to single-method approaches. Recognizing the challenges in accurately assessing LLM performance, we address key limitations of previous evaluation frameworks and introduce the Semantic Retention Compression Rate (SrCr), a novel metric that quantifies the trade-off between model compression and semantic preservation, facilitating the optimization of pruning-quantization configurations. Experiments demonstrate that our recommended combination achieves, on average, a 20% performance increase compared to an equivalent quantization-only model at the same theoretical compression rate. 

---
# Seed1.5-VL Technical Report 

**Authors**: Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, Jingji Chen, Jingjia Huang, Kang Lei, Liping Yuan, Lishu Luo, Pengfei Liu, Qinghao Ye, Rui Qian, Shen Yan, Shixiong Zhao, Shuai Peng, Shuangye Li, Sihang Yuan, Sijin Wu, Tianheng Cheng, Weiwei Liu, Wenqian Wang, Xianhan Zeng, Xiao Liu, Xiaobo Qin, Xiaohan Ding, Xiaojun Xiao, Xiaoying Zhang, Xuanwei Zhang, Xuehan Xiong, Yanghua Peng, Yangrui Chen, Yanwei Li, Yanxu Hu, Yi Lin, Yiyuan Hu, Yiyuan Zhang, Youbin Wu, Yu Li, Yudong Liu, Yue Ling, Yujia Qin, Zanbo Wang, Zhiwu He, Aoxue Zhang, Bairen Yi, Bencheng Liao, Can Huang, Can Zhang, Chaorui Deng, Chaoyi Deng, Cheng Lin, Cheng Yuan, Chenggang Li, Chenhui Gou, Chenwei Lou, Chengzhi Wei, Chundian Liu, Chunyuan Li, Deyao Zhu, Donghong Zhong, Feng Li, Feng Zhang, Gang Wu, Guodong Li, Guohong Xiao, Haibin Lin, Haihua Yang, Haoming Wang, Heng Ji, Hongxiang Hao, Hui Shen, Huixia Li, Jiahao Li, Jialong Wu, Jianhua Zhu, Jianpeng Jiao, Jiashi Feng, Jiaze Chen, Jianhui Duan, Jihao Liu, Jin Zeng, Jingqun Tang, Jingyu Sun, Joya Chen, Jun Long, Junda Feng, Junfeng Zhan, Junjie Fang, Junting Lu, Kai Hua, Kai Liu, Kai Shen, Kaiyuan Zhang, Ke Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07062)  

**Abstract**: We present Seed1.5-VL, a vision-language foundation model designed to advance general-purpose multimodal understanding and reasoning. Seed1.5-VL is composed with a 532M-parameter vision encoder and a Mixture-of-Experts (MoE) LLM of 20B active parameters. Despite its relatively compact architecture, it delivers strong performance across a wide spectrum of public VLM benchmarks and internal evaluation suites, achieving the state-of-the-art performance on 38 out of 60 public benchmarks. Moreover, in agent-centric tasks such as GUI control and gameplay, Seed1.5-VL outperforms leading multimodal systems, including OpenAI CUA and Claude 3.7. Beyond visual and video understanding, it also demonstrates strong reasoning abilities, making it particularly effective for multimodal reasoning challenges such as visual puzzles. We believe these capabilities will empower broader applications across diverse tasks. In this report, we mainly provide a comprehensive review of our experiences in building Seed1.5-VL across model design, data construction, and training at various stages, hoping that this report can inspire further research. Seed1.5-VL is now accessible at this https URL (Volcano Engine Model ID: doubao-1-5-thinking-vision-pro-250428) 

---
# Convert Language Model into a Value-based Strategic Planner 

**Authors**: Xiaoyu Wang, Yue Zhao, Qingqing Gu, Zhonglin Jiang, Xiaokai Chen, Yong Chen, Luo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.06987)  

**Abstract**: Emotional support conversation (ESC) aims to alleviate the emotional distress of individuals through effective conversations. Although large language models (LLMs) have obtained remarkable progress on ESC, most of these studies might not define the diagram from the state model perspective, therefore providing a suboptimal solution for long-term satisfaction. To address such an issue, we leverage the Q-learning on LLMs, and propose a framework called straQ*. Our framework allows a plug-and-play LLM to bootstrap the planning during ESC, determine the optimal strategy based on long-term returns, and finally guide the LLM to response. Substantial experiments on ESC datasets suggest that straQ* outperforms many baselines, including direct inference, self-refine, chain of thought, finetuning, and finite state machines. 

---
# ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification 

**Authors**: Dipayan Saha, Hasan Al Shaikh, Shams Tarek, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06821)  

**Abstract**: Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios. 

---
# MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG 

**Authors**: Woosang Lim, Zekun Li, Gyuwan Kim, Sungyoung Ji, HyeonJung Kim, Kyuri Choi, Jin Hyuk Lim, Kyungpyo Park, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06569)  

**Abstract**: Long-context (LC) Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained context windows, and fragmented information caused by suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical retrieval framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through chunk- and document-level expansions in real time. By starting from the finest-level retrieval and progressively incorporating higher-level and broader context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on the challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm that MacRAG consistently surpasses baseline RAG pipelines on single- and multi-step generation with Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at this https URL. 

---
# System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection 

**Authors**: Jiawei Guo, Haipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.06493)  

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning. 

---
# Enfoque Odychess: Un método dialéctico, constructivista y adaptativo para la enseñanza del ajedrez con inteligencias artificiales generativas 

**Authors**: Ernesto Giralt Hernandez, Lazaro Antonio Bueno Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.06652)  

**Abstract**: Chess teaching has evolved through different approaches, however, traditional methodologies, often based on memorization, contrast with the new possibilities offered by generative artificial intelligence, a technology still little explored in this field. This study seeks to empirically validate the effectiveness of the Odychess Approach in improving chess knowledge, strategic understanding, and metacognitive skills in students. A quasi-experimental study was conducted with a pre-test/post-test design and a control group (N=60). The experimental intervention implemented the Odychess Approach, incorporating a Llama 3.3 language model that was specifically adapted using Parameter-Efficient Fine-Tuning (PEFT) techniques to act as a Socratic chess tutor. Quantitative assessment instruments were used to measure chess knowledge, strategic understanding, and metacognitive skills before and after the intervention. The results of the quasi-experimental study showed significant improvements in the experimental group compared to the control group in the three variables analyzed: chess knowledge, strategic understanding, and metacognitive skills. The complementary qualitative analysis revealed greater analytical depth, more developed dialectical reasoning, and increased intrinsic motivation in students who participated in the Odychess method-based intervention. The Odychess Approach represents an effective pedagogical methodology for teaching chess, demonstrating the potential of the synergistic integration of constructivist and dialectical principles with generative artificial intelligence. The implications of this work are relevant for educators and institutions interested in adopting innovative pedagogical technologies and for researchers in the field of AI applied to education, highlighting the transferability of the language model adaptation methodology to other educational domains. 

---
# Towards AI-Driven Human-Machine Co-Teaming for Adaptive and Agile Cyber Security Operation Centers 

**Authors**: Massimiliano Albanese, Xinming Ou, Kevin Lybarger, Daniel Lende, Dmitry Goldgof  

**Link**: [PDF](https://arxiv.org/pdf/2505.06394)  

**Abstract**: Security Operations Centers (SOCs) face growing challenges in managing cybersecurity threats due to an overwhelming volume of alerts, a shortage of skilled analysts, and poorly integrated tools. Human-AI collaboration offers a promising path to augment the capabilities of SOC analysts while reducing their cognitive overload. To this end, we introduce an AI-driven human-machine co-teaming paradigm that leverages large language models (LLMs) to enhance threat intelligence, alert triage, and incident response workflows. We present a vision in which LLM-based AI agents learn from human analysts the tacit knowledge embedded in SOC operations, enabling the AI agents to improve their performance on SOC tasks through this co-teaming. We invite SOCs to collaborate with us to further develop this process and uncover replicable patterns where human-AI co-teaming yields measurable improvements in SOC productivity. 

---
# Camera Control at the Edge with Language Models for Scene Understanding 

**Authors**: Alexiy Buynitsky, Sina Ehsani, Bhanu Pallakonda, Pragyana Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.06402)  

**Abstract**: In this paper, we present Optimized Prompt-based Unified System (OPUS), a framework that utilizes a Large Language Model (LLM) to control Pan-Tilt-Zoom (PTZ) cameras, providing contextual understanding of natural environments. To achieve this goal, the OPUS system improves cost-effectiveness by generating keywords from a high-level camera control API and transferring knowledge from larger closed-source language models to smaller ones through Supervised Fine-Tuning (SFT) on synthetic data. This enables efficient edge deployment while maintaining performance comparable to larger models like GPT-4. OPUS enhances environmental awareness by converting data from multiple cameras into textual descriptions for language models, eliminating the need for specialized sensory tokens. In benchmark testing, our approach significantly outperformed both traditional language model techniques and more complex prompting methods, achieving a 35% improvement over advanced techniques and a 20% higher task accuracy compared to closed-source models like Gemini Pro. The system demonstrates OPUS's capability to simplify PTZ camera operations through an intuitive natural language interface. This approach eliminates the need for explicit programming and provides a conversational method for interacting with camera systems, representing a significant advancement in how users can control and utilize PTZ camera technology. 

---
# Prompting Large Language Models for Training-Free Non-Intrusive Load Monitoring 

**Authors**: Junyu Xue, Xudong Wang, Xiaoling He, Shicheng Liu, Yi Wang, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06330)  

**Abstract**: Non-intrusive Load Monitoring (NILM) aims to disaggregate aggregate household electricity consumption into individual appliance usage, enabling more effective energy management. While deep learning has advanced NILM, it remains limited by its dependence on labeled data, restricted generalization, and lack of interpretability. In this paper, we introduce the first prompt-based NILM framework that leverages Large Language Models (LLMs) with in-context learning. We design and evaluate prompt strategies that integrate appliance features, timestamps and contextual information, as well as representative time-series examples, using the REDD dataset. With optimized prompts, LLMs achieve competitive state detection accuracy, reaching an average F1-score of 0.676 on unseen households, and demonstrate robust generalization without the need for fine-tuning. LLMs also enhance interpretability by providing clear, human-readable explanations for their predictions. Our results show that LLMs can reduce data requirements, improve adaptability, and provide transparent energy disaggregation in NILM applications. 

---
# Document Attribution: Examining Citation Relationships using Large Language Models 

**Authors**: Vipula Rawte, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka  

**Link**: [PDF](https://arxiv.org/pdf/2505.06324)  

**Abstract**: As Large Language Models (LLMs) are increasingly applied to document-based tasks - such as document summarization, question answering, and information extraction - where user requirements focus on retrieving information from provided documents rather than relying on the model's parametric knowledge, ensuring the trustworthiness and interpretability of these systems has become a critical concern. A central approach to addressing this challenge is attribution, which involves tracing the generated outputs back to their source documents. However, since LLMs can produce inaccurate or imprecise responses, it is crucial to assess the reliability of these citations.
To tackle this, our work proposes two techniques. (1) A zero-shot approach that frames attribution as a straightforward textual entailment task. Our method using flan-ul2 demonstrates an improvement of 0.27% and 2.4% over the best baseline of ID and OOD sets of AttributionBench, respectively. (2) We also explore the role of the attention mechanism in enhancing the attribution process. Using a smaller LLM, flan-t5-small, the F1 scores outperform the baseline across almost all layers except layer 4 and layers 8 through 11. 

---
# Learn to Think: Bootstrapping LLM Reasoning Capability Through Graph Learning 

**Authors**: Hang Gao, Chenhao Zhang, Tie Wang, Junsuo Zhao, Fengge Wu, Changwen Zheng, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06321)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across various domains. However, they still face significant challenges, including high computational costs for training and limitations in solving complex reasoning problems. Although existing methods have extended the reasoning capabilities of LLMs through structured paradigms, these approaches often rely on task-specific prompts and predefined reasoning processes, which constrain their flexibility and generalizability. To address these limitations, we propose a novel framework that leverages graph learning to enable more flexible and adaptive reasoning capabilities for LLMs. Specifically, this approach models the reasoning process of a problem as a graph and employs LLM-based graph learning to guide the adaptive generation of each reasoning step. To further enhance the adaptability of the model, we introduce a Graph Neural Network (GNN) module to perform representation learning on the generated reasoning process, enabling real-time adjustments to both the model and the prompt. Experimental results demonstrate that this method significantly improves reasoning performance across multiple tasks without requiring additional training or task-specific prompt design. Code can be found in this https URL. 

---
# Quantum State Preparation via Large-Language-Model-Driven Evolution 

**Authors**: Qing-Hong Cao, Zong-Yue Hou, Ying-Ying Li, Xiaohui Liu, Zhuo-Yang Song, Liang-Qi Zhang, Shutao Zhang, Ke Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06347)  

**Abstract**: We propose an automated framework for quantum circuit design by integrating large-language models (LLMs) with evolutionary optimization to overcome the rigidity, scalability limitations, and expert dependence of traditional ones in variational quantum algorithms. Our approach (FunSearch) autonomously discovers hardware-efficient ansätze with new features of scalability and system-size-independent number of variational parameters entirely from scratch. Demonstrations on the Ising and XY spin chains with n = 9 qubits yield circuits containing 4 parameters, achieving near-exact energy extrapolation across system sizes. Implementations on quantum hardware (Zuchongzhi chip) validate practicality, where two-qubit quantum gate noises can be effectively mitigated via zero-noise extrapolations for a spin chain system as large as 20 sites. This framework bridges algorithmic design and experimental constraints, complementing contemporary quantum architecture search frameworks to advance scalable quantum simulations. 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

---
# Large Language Model-driven Security Assistant for Internet of Things via Chain-of-Thought 

**Authors**: Mingfei Zeng, Ming Xie, Xixi Zheng, Chunhai Li, Chuan Zhang, Liehuang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06307)  

**Abstract**: The rapid development of Internet of Things (IoT) technology has transformed people's way of life and has a profound impact on both production and daily activities. However, with the rapid advancement of IoT technology, the security of IoT devices has become an unavoidable issue in both research and applications. Although some efforts have been made to detect or mitigate IoT security vulnerabilities, they often struggle to adapt to the complexity of IoT environments, especially when dealing with dynamic security scenarios. How to automatically, efficiently, and accurately understand these vulnerabilities remains a challenge. To address this, we propose an IoT security assistant driven by Large Language Model (LLM), which enhances the LLM's understanding of IoT security vulnerabilities and related threats. The aim of the ICoT method we propose is to enable the LLM to understand security issues by breaking down the various dimensions of security vulnerabilities and generating responses tailored to the user's specific needs and expertise level. By incorporating ICoT, LLM can gradually analyze and reason through complex security scenarios, resulting in more accurate, in-depth, and personalized security recommendations and solutions. Experimental results show that, compared to methods relying solely on LLM, our proposed LLM-driven IoT security assistant significantly improves the understanding of IoT security issues through the ICoT approach and provides personalized solutions based on the user's identity, demonstrating higher accuracy and reliability. 

---
# User Behavior Analysis in Privacy Protection with Large Language Models: A Study on Privacy Preferences with Limited Data 

**Authors**: Haowei Yang, Qingyi Lu, Yang Wang, Sibei Liu, Jiayun Zheng, Ao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06305)  

**Abstract**: With the widespread application of large language models (LLMs), user privacy protection has become a significant research topic. Existing privacy preference modeling methods often rely on large-scale user data, making effective privacy preference analysis challenging in data-limited environments. This study explores how LLMs can analyze user behavior related to privacy protection in scenarios with limited data and proposes a method that integrates Few-shot Learning and Privacy Computing to model user privacy preferences. The research utilizes anonymized user privacy settings data, survey responses, and simulated data, comparing the performance of traditional modeling approaches with LLM-based methods. Experimental results demonstrate that, even with limited data, LLMs significantly improve the accuracy of privacy preference modeling. Additionally, incorporating Differential Privacy and Federated Learning further reduces the risk of user data exposure. The findings provide new insights into the application of LLMs in privacy protection and offer theoretical support for advancing privacy computing and user behavior analysis. 

---
# AKD : Adversarial Knowledge Distillation For Large Language Models Alignment on Coding tasks 

**Authors**: Ilyas Oulkadda, Julien Perez  

**Link**: [PDF](https://arxiv.org/pdf/2505.06267)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) for code generation, exemplified by GitHub Copilot\footnote{A coding extension powered by a Code-LLM to assist in code completion tasks} surpassing a million users, highlights the transformative potential of these tools in improving developer productivity. However, this rapid growth also underscores critical concerns regarding the quality, safety, and reliability of the code they generate. As Code-LLMs evolve, they face significant challenges, including the diminishing returns of model scaling and the scarcity of new, high-quality training data. To address these issues, this paper introduces Adversarial Knowledge Distillation (AKD), a novel approach that leverages adversarially generated synthetic datasets to distill the capabilities of larger models into smaller, more efficient ones. By systematically stress-testing and refining the reasoning capabilities of Code-LLMs, AKD provides a framework for enhancing model robustness, reliability, and security while improving their parameter-efficiency. We believe this work represents a critical step toward ensuring dependable automated code generation within the constraints of existing data and the cost-efficiency of model execution. 

---
# Defending against Indirect Prompt Injection by Instruction Detection 

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06311)  

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark. 

---
# QiMeng-TensorOp: Automatically Generating High-Performance Tensor Operators with Hardware Primitives 

**Authors**: Xuzhi Zhang, Shaohui Peng, Qirui Zhou, Yuanbo Wen, Qi Guo, Ruizhi Chen, Xinguo Zhu, Weiqiang Xiong, Haixin Chen, Congying Ma, Ke Gao, Chen Zhao, Yanjun Wu, Yunji Chen, Ling Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06302)  

**Abstract**: Computation-intensive tensor operators constitute over 90\% of the computations in Large Language Models (LLMs) and Deep Neural this http URL and efficiently generating high-performance tensor operators with hardware primitives is crucial for diverse and ever-evolving hardware architectures like RISC-V, ARM, and GPUs, as manually optimized implementation takes at least months and lacks this http URL excel at generating high-level language codes, but they struggle to fully comprehend hardware characteristics and produce high-performance tensor operators. We introduce a tensor-operator auto-generation framework with a one-line user prompt (QiMeng-TensorOp), which enables LLMs to automatically exploit hardware characteristics to generate tensor operators with hardware primitives, and tune parameters for optimal performance across diverse hardware. Experimental results on various hardware platforms, SOTA LLMs, and typical tensor operators demonstrate that QiMeng-TensorOp effectively unleashes the computing capability of various hardware platforms, and automatically generates tensor operators of superior performance. Compared with vanilla LLMs, QiMeng-TensorOp achieves up to $1291 \times$ performance improvement. Even compared with human experts, QiMeng-TensorOp could reach $251 \%$ of OpenBLAS on RISC-V CPUs, and $124 \%$ of cuBLAS on NVIDIA GPUs. Additionally, QiMeng-TensorOp also significantly reduces development costs by $200 \times$ compared with human experts. 

---
# Dialz: A Python Toolkit for Steering Vectors 

**Authors**: Zara Siddique, Liam D. Turner, Luis Espinosa-Anke  

**Link**: [PDF](https://arxiv.org/pdf/2505.06262)  

**Abstract**: We introduce Dialz, a framework for advancing research on steering vectors for open-source LLMs, implemented in Python. Steering vectors allow users to modify activations at inference time to amplify or weaken a 'concept', e.g. honesty or positivity, providing a more powerful alternative to prompting or fine-tuning. Dialz supports a diverse set of tasks, including creating contrastive pair datasets, computing and applying steering vectors, and visualizations. Unlike existing libraries, Dialz emphasizes modularity and usability, enabling both rapid prototyping and in-depth analysis. We demonstrate how Dialz can be used to reduce harmful outputs such as stereotypes, while also providing insights into model behaviour across different layers. We release Dialz with full documentation, tutorials, and support for popular open-source models to encourage further research in safe and controllable language generation. Dialz enables faster research cycles and facilitates insights into model interpretability, paving the way for safer, more transparent, and more reliable AI systems. 

---
# PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model 

**Authors**: Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06274)  

**Abstract**: Multi-objective test-time alignment aims to adapt large language models (LLMs) to diverse multi-dimensional user preferences during inference while keeping LLMs frozen. Recently, GenARM (Xu et al., 2025) first independently trains Autoregressive Reward Models (ARMs) for each preference dimension without awareness of each other, then combines their outputs based on user-specific preference vectors during inference to achieve multi-objective test-time alignment, leading to two key limitations: the need for \textit{multiple} ARMs increases the inference cost, and the separate training of ARMs causes the misalignment between the guided generation and the user preferences. To address these issues, we propose Preference-aware ARM (PARM), a single unified ARM trained across all preference dimensions. PARM uses our proposed Preference-Aware Bilinear Low-Rank Adaptation (PBLoRA), which employs a bilinear form to condition the ARM on preference vectors, enabling it to achieve precise control over preference trade-offs during inference. Experiments demonstrate that PARM reduces inference costs and achieves better alignment with preference vectors compared with existing methods. Additionally, PARM enables weak-to-strong guidance, allowing a smaller PARM to guide a larger frozen LLM without expensive training, making multi-objective alignment accessible with limited computing resources. The code is available at this https URL. 

---
# Spoken Language Understanding on Unseen Tasks With In-Context Learning 

**Authors**: Neeraj Agrawal, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.07731)  

**Abstract**: Spoken language understanding (SLU) tasks involve diverse skills that probe the information extraction, classification and/or generation capabilities of models. In this setting, task-specific training data may not always be available. While traditional task-specific SLU models are unable to cater to such requirements, the speech-text large language models (LLMs) offer a promising alternative with emergent abilities. However, out of-the-box, our evaluations indicate that the zero/few-shot performance of prominent open-source speech-text LLMs on SLU tasks are not up to the mark. In this paper, we introduce a novel approach to robust task-agnostic fine-tuning using randomized class labels. With this proposed fine-tuning, we illustrate that the performance of the speech-text LLMs on an unseen task is significantly improved over standard approaches. Critically, the proposed approach avoids the requirement of task-specific data annotations for enabling new tasks in speech-text LLMs. 

---
# Benchmarking Retrieval-Augmented Generation for Chemistry 

**Authors**: Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07671)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful framework for enhancing large language models (LLMs) with external knowledge, particularly in scientific domains that demand specialized and dynamic information. Despite its promise, the application of RAG in the chemistry domain remains underexplored, primarily due to the lack of high-quality, domain-specific corpora and well-curated evaluation benchmarks. In this work, we introduce ChemRAG-Bench, a comprehensive benchmark designed to systematically assess the effectiveness of RAG across a diverse set of chemistry-related tasks. The accompanying chemistry corpus integrates heterogeneous knowledge sources, including scientific literature, the PubChem database, PubMed abstracts, textbooks, and Wikipedia entries. In addition, we present ChemRAG-Toolkit, a modular and extensible RAG toolkit that supports five retrieval algorithms and eight LLMs. Using ChemRAG-Toolkit, we demonstrate that RAG yields a substantial performance gain -- achieving an average relative improvement of 17.4% over direct inference methods. We further conduct in-depth analyses on retriever architectures, corpus selection, and the number of retrieved passages, culminating in practical recommendations to guide future research and deployment of RAG systems in the chemistry domain. The code and data is available at this https URL. 

---
# JobHop: A Large-Scale Dataset of Career Trajectories 

**Authors**: Iman Johary, Raphael Romero, Alexandru C. Mara, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2505.07653)  

**Abstract**: Understanding labor market dynamics is essential for policymakers, employers, and job seekers. However, comprehensive datasets that capture real-world career trajectories are scarce. In this paper, we introduce JobHop, a large-scale public dataset derived from anonymized resumes provided by VDAB, the public employment service in Flanders, Belgium. Utilizing Large Language Models (LLMs), we process unstructured resume data to extract structured career information, which is then mapped to standardized ESCO occupation codes using a multi-label classification model. This results in a rich dataset of over 2.3 million work experiences, extracted from and grouped into more than 391,000 user resumes and mapped to standardized ESCO occupation codes, offering valuable insights into real-world occupational transitions. This dataset enables diverse applications, such as analyzing labor market mobility, job stability, and the effects of career breaks on occupational transitions. It also supports career path prediction and other data-driven decision-making processes. To illustrate its potential, we explore key dataset characteristics, including job distributions, career breaks, and job transitions, demonstrating its value for advancing labor market research. 

---
# Domain Regeneration: How well do LLMs match syntactic properties of text domains? 

**Authors**: Da Ju, Hagen Blix, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2505.07784)  

**Abstract**: Recent improvement in large language model performance have, in all likelihood, been accompanied by improvement in how well they can approximate the distribution of their training data. In this work, we explore the following question: which properties of text domains do LLMs faithfully approximate, and how well do they do so? Applying observational approaches familiar from corpus linguistics, we prompt a commonly used, opensource LLM to regenerate text from two domains of permissively licensed English text which are often contained in LLM training data -- Wikipedia and news text. This regeneration paradigm allows us to investigate whether LLMs can faithfully match the original human text domains in a fairly semantically-controlled setting. We investigate varying levels of syntactic abstraction, from more simple properties like sentence length, and article readability, to more complex and higher order properties such as dependency tag distribution, parse depth, and parse complexity. We find that the majority of the regenerated distributions show a shifted mean, a lower standard deviation, and a reduction of the long tail, as compared to the human originals. 

---
# QUPID: Quantified Understanding for Enhanced Performance, Insights, and Decisions in Korean Search Engines 

**Authors**: Ohjoon Kwon, Changsu Lee, Jihye Back, Lim Sun Suk, Inho Kang, Donghyeon Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07345)  

**Abstract**: Large language models (LLMs) have been widely used for relevance assessment in information retrieval. However, our study demonstrates that combining two distinct small language models (SLMs) with different architectures can outperform LLMs in this task. Our approach -- QUPID -- integrates a generative SLM with an embedding-based SLM, achieving higher relevance judgment accuracy while reducing computational costs compared to state-of-the-art LLM solutions. This computational efficiency makes QUPID highly scalable for real-world search systems processing millions of queries daily. In experiments across diverse document types, our method demonstrated consistent performance improvements (Cohen's Kappa of 0.646 versus 0.387 for leading LLMs) while offering 60x faster inference times. Furthermore, when integrated into production search pipelines, QUPID improved nDCG@5 scores by 1.9%. These findings underscore how architectural diversity in model combinations can significantly enhance both search relevance and operational efficiency in information retrieval systems. 

---
# Computational Fact-Checking of Online Discourse: Scoring scientific accuracy in climate change related news articles 

**Authors**: Tim Wittenborg, Constantin Sebastian Tremel, Markus Stocker, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07409)  

**Abstract**: Democratic societies need reliable information. Misinformation in popular media such as news articles or videos threatens to impair civic discourse. Citizens are, unfortunately, not equipped to verify this content flood consumed daily at increasing rates. This work aims to semi-automatically quantify scientific accuracy of online media. By semantifying media of unknown veracity, their statements can be compared against equally processed trusted sources. We implemented a workflow using LLM-based statement extraction and knowledge graph analysis. Our neurosymbolic system was able to evidently streamline state-of-the-art veracity quantification. Evaluated via expert interviews and a user survey, the tool provides a beneficial veracity indication. This indicator, however, is unable to annotate public media at the required granularity and scale. Further work towards a FAIR (Findable, Accessible, Interoperable, Reusable) ground truth and complementary metrics are required to scientifically support civic discourse. 

---
# AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection 

**Authors**: Kai Hua, Steven Wu, Ge Zhang, Ke Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07293)  

**Abstract**: Recently, there has been growing interest in collecting reasoning-intensive pretraining data to improve LLMs' complex reasoning ability. Prior approaches typically rely on supervised classifiers to identify such data, which requires labeling by humans or LLMs, often introducing domain-specific biases. Due to the attention heads being crucial to in-context reasoning, we propose AttentionInfluence, a simple yet effective, training-free method without supervision signal. Our approach enables a small pretrained language model to act as a strong data selector through a simple attention head masking operation. Specifically, we identify retrieval heads and compute the loss difference when masking these heads. We apply AttentionInfluence to a 1.3B-parameter dense model to conduct data selection on the SmolLM corpus of 241B tokens, and mix the SmolLM corpus with the selected subset comprising 73B tokens to pretrain a 7B-parameter dense model using 1T training tokens and WSD learning rate scheduling. Our experimental results demonstrate substantial improvements, ranging from 1.4pp to 3.5pp, across several knowledge-intensive and reasoning-heavy benchmarks (i.e., MMLU, MMLU-Pro, AGIEval-en, GSM8K, and HumanEval). This demonstrates an effective weak-to-strong scaling property, with small models improving the final performance of larger models-offering a promising and scalable path for reasoning-centric data selection. 

---
# Learning from Peers in Reasoning Models 

**Authors**: Tongxu Luo, Wenyu Du, Jiaxi Bi, Stephen Chung, Zhengyang Tang, Hao Yang, Min Zhang, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07787)  

**Abstract**: Large Reasoning Models (LRMs) have the ability to self-correct even when they make mistakes in their reasoning paths. However, our study reveals that when the reasoning process starts with a short but poor beginning, it becomes difficult for the model to recover. We refer to this phenomenon as the "Prefix Dominance Trap". Inspired by psychological findings that peer interaction can promote self-correction without negatively impacting already accurate individuals, we propose **Learning from Peers** (LeaP) to address this phenomenon. Specifically, every tokens, each reasoning path summarizes its intermediate reasoning and shares it with others through a routing mechanism, enabling paths to incorporate peer insights during inference. However, we observe that smaller models sometimes fail to follow summarization and reflection instructions effectively. To address this, we fine-tune them into our **LeaP-T** model series. Experiments on AIME 2024, AIME 2025, AIMO 2025, and GPQA Diamond show that LeaP provides substantial improvements. For instance, QwQ-32B with LeaP achieves nearly 5 absolute points higher than the baseline on average, and surpasses DeepSeek-R1-671B on three math benchmarks with an average gain of 3.3 points. Notably, our fine-tuned LeaP-T-7B matches the performance of DeepSeek-R1-Distill-Qwen-14B on AIME 2024. In-depth analysis reveals LeaP's robust error correction by timely peer insights, showing strong error tolerance and handling varied task difficulty. LeaP marks a milestone by enabling LRMs to collaborate during reasoning. Our code, datasets, and models are available at this https URL . 

---
# Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs 

**Authors**: Yifan Wei, Xiaoyan Yu, Tengfei Pan, Angsheng Li, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.07184)  

**Abstract**: Large language models (LLMs) have achieved unprecedented performance by leveraging vast pretraining corpora, yet their performance remains suboptimal in knowledge-intensive domains such as medicine and scientific research, where high factual precision is required. While synthetic data provides a promising avenue for augmenting domain knowledge, existing methods frequently generate redundant samples that do not align with the model's true knowledge gaps. To overcome this limitation, we propose a novel Structural Entropy-guided Knowledge Navigator (SENATOR) framework that addresses the intrinsic knowledge deficiencies of LLMs. Our approach employs the Structure Entropy (SE) metric to quantify uncertainty along knowledge graph paths and leverages Monte Carlo Tree Search (MCTS) to selectively explore regions where the model lacks domain-specific knowledge. Guided by these insights, the framework generates targeted synthetic data for supervised fine-tuning, enabling continuous self-improvement. Experimental results on LLaMA-3 and Qwen2 across multiple domain-specific benchmarks show that SENATOR effectively detects and repairs knowledge deficiencies, achieving notable performance improvements. The code and data for our methods and experiments are available at this https URL. 

---
# Benchmarking Ethical and Safety Risks of Healthcare LLMs in China-Toward Systemic Governance under Healthy China 2030 

**Authors**: Mouxiao Bian, Rongzhao Zhang, Chao Ding, Xinwei Peng, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07205)  

**Abstract**: Large Language Models (LLMs) are poised to transform healthcare under China's Healthy China 2030 initiative, yet they introduce new ethical and patient-safety challenges. We present a novel 12,000-item Q&A benchmark covering 11 ethics and 9 safety dimensions in medical contexts, to quantitatively evaluate these risks. Using this dataset, we assess state-of-the-art Chinese medical LLMs (e.g., Qwen 2.5-32B, DeepSeek), revealing moderate baseline performance (accuracy 42.7% for Qwen 2.5-32B) and significant improvements after fine-tuning on our data (up to 50.8% accuracy). Results show notable gaps in LLM decision-making on ethics and safety scenarios, reflecting insufficient institutional oversight. We then identify systemic governance shortfalls-including the lack of fine-grained ethical audit protocols, slow adaptation by hospital IRBs, and insufficient evaluation tools-that currently hinder safe LLM deployment. Finally, we propose a practical governance framework for healthcare institutions (embedding LLM auditing teams, enacting data ethics guidelines, and implementing safety simulation pipelines) to proactively manage LLM risks. Our study highlights the urgent need for robust LLM governance in Chinese healthcare, aligning AI innovation with patient safety and ethical standards. 

---
# KDH-MLTC: Knowledge Distillation for Healthcare Multi-Label Text Classification 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.07162)  

**Abstract**: The increasing volume of healthcare textual data requires computationally efficient, yet highly accurate classification approaches able to handle the nuanced and complex nature of medical terminology. This research presents Knowledge Distillation for Healthcare Multi-Label Text Classification (KDH-MLTC), a framework leveraging model compression and Large Language Models (LLMs). The proposed approach addresses conventional healthcare Multi-Label Text Classification (MLTC) challenges by integrating knowledge distillation and sequential fine-tuning, subsequently optimized through Particle Swarm Optimization (PSO) for hyperparameter tuning. KDH-MLTC transfers knowledge from a more complex teacher LLM (i.e., BERT) to a lighter student LLM (i.e., DistilBERT) through sequential training adapted to MLTC that preserves the teacher's learned information while significantly reducing computational requirements. As a result, the classification is enabled to be conducted locally, making it suitable for healthcare textual data characterized by sensitivity and, therefore, ensuring HIPAA compliance. The experiments conducted on three medical literature datasets of different sizes, sampled from the Hallmark of Cancer (HoC) dataset, demonstrate that KDH-MLTC achieves superior performance compared to existing approaches, particularly for the largest dataset, reaching an F1 score of 82.70%. Additionally, statistical validation and an ablation study are carried out, proving the robustness of KDH-MLTC. Furthermore, the PSO-based hyperparameter optimization process allowed the identification of optimal configurations. The proposed approach contributes to healthcare text classification research, balancing efficiency requirements in resource-constrained healthcare settings with satisfactory accuracy demands. 

---
# HAMLET: Healthcare-focused Adaptive Multilingual Learning Embedding-based Topic Modeling 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.07157)  

**Abstract**: Traditional topic models often struggle with contextual nuances and fail to adequately handle polysemy and rare words. This limitation typically results in topics that lack coherence and quality. Large Language Models (LLMs) can mitigate this issue by generating an initial set of topics. However, these raw topics frequently lack refinement and representativeness, which leads to redundancy without lexical similarity and reduced interpretability. This paper introduces HAMLET, a graph-driven architecture for cross-lingual healthcare topic modeling that uses LLMs. The proposed approach leverages neural-enhanced semantic fusion to refine the embeddings of topics generated by the LLM. Instead of relying solely on statistical co-occurrence or human interpretation to extract topics from a document corpus, this method introduces a topic embedding refinement that uses Bidirectional Encoder Representations from Transformers (BERT) and Graph Neural Networks (GNN). After topic generation, a hybrid technique that involves BERT and Sentence-BERT (SBERT) is employed for embedding. The topic representations are further refined using a GNN, which establishes connections between documents, topics, words, similar topics, and similar words. A novel method is introduced to compute similarities. Consequently, the topic embeddings are refined, and the top k topics are extracted. Experiments were conducted using two healthcare datasets, one in English and one in French, from which six sets were derived. The results demonstrate the effectiveness of HAMLET. 

---
# From Rankings to Insights: Evaluation Should Shift Focus from Leaderboard to Feedback 

**Authors**: Zongqi Wang, Tianle Gu, Chen Gong, Xin Tian, Siqi Bao, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06698)  

**Abstract**: Automatic evaluation benchmarks such as MT-Bench, Arena-Hard, and Auto-Arena are seeing growing adoption for the evaluation of Large Language Models (LLMs). Existing research has primarily focused on approximating human-based model rankings using limited data and LLM-as-a-Judge. However, the fundamental premise of these studies, which attempts to replicate human rankings, is flawed. Specifically, these benchmarks typically offer only overall scores, limiting their utility to leaderboard rankings, rather than providing feedback that can guide model optimization and support model profiling. Therefore, we advocate for an evaluation paradigm shift from approximating human-based model rankings to providing feedback with analytical value. To this end, we introduce Feedbacker, an evaluation framework that provides comprehensive and fine-grained results, thereby enabling thorough identification of a model's specific strengths and weaknesses. Such feedback not only supports the targeted optimization of the model but also enhances the understanding of its behavior. Feedbacker comprises three key components: an extensible tree-based query taxonomy builder, an automated query synthesis scheme, and a suite of visualization and analysis tools. Furthermore, we propose a novel LLM-as-a-Judge method: PC2 (Pre-Comparison-derived Criteria) pointwise evaluation. This method derives evaluation criteria by pre-comparing the differences between several auxiliary responses, achieving the accuracy of pairwise evaluation while maintaining the time complexity of pointwise evaluation. Finally, leveraging the evaluation results of 17 mainstream LLMs, we demonstrate the usage of Feedbacker and highlight its effectiveness and potential. Our homepage project is available at this https URL. 

---
# Boosting Neural Language Inference via Cascaded Interactive Reasoning 

**Authors**: Min Li, Chun Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06607)  

**Abstract**: Natural Language Inference (NLI) focuses on ascertaining the logical relationship (entailment, contradiction, or neutral) between a given premise and hypothesis. This task presents significant challenges due to inherent linguistic features such as diverse phrasing, semantic complexity, and contextual nuances. While Pre-trained Language Models (PLMs) built upon the Transformer architecture have yielded substantial advancements in NLI, prevailing methods predominantly utilize representations from the terminal layer. This reliance on final-layer outputs may overlook valuable information encoded in intermediate layers, potentially limiting the capacity to model intricate semantic interactions effectively. Addressing this gap, we introduce the Cascaded Interactive Reasoning Network (CIRN), a novel architecture designed for deeper semantic comprehension in NLI. CIRN implements a hierarchical feature extraction strategy across multiple network depths, operating within an interactive space where cross-sentence information is continuously integrated. This mechanism aims to mimic a process of progressive reasoning, transitioning from surface-level feature matching to uncovering more profound logical and semantic connections between the premise and hypothesis. By systematically mining latent semantic relationships at various representational levels, CIRN facilitates a more thorough understanding of the input pair. Comprehensive evaluations conducted on several standard NLI benchmark datasets reveal consistent performance gains achieved by CIRN over competitive baseline approaches, demonstrating the efficacy of leveraging multi-level interactive features for complex relational reasoning. 

---
# Is your multimodal large language model a good science tutor? 

**Authors**: Ming Liu, Liwen Wang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06418)  

**Abstract**: Multimodal large language models (MLLMs) demonstrate impressive performance on scientific reasoning tasks (e.g., ScienceQA). However, most existing benchmarks focus narrowly on the accuracy of the final answer while ignoring other metrics. In particular, when applying MLLMs to educational contexts, the goal is not only correctness but also the ability to teach. In this paper, we propose a framework that evaluates MLLMs as science tutors using a comprehensive educational rubric and a simulated student model that judges the teaching performance of the tutors. Given a list of candidate MLLM science tutors, we use rubric-based student judgments to produce a range of tutor performance scores, identifying both strong and weak tutors. Using the training section of the ScienceQA dataset, we then construct a data set of pairwise comparisons between the outputs of strong and weak tutors. This enables us to apply multiple preference optimization methods to fine-tune an underperforming tutor model (Qwen2-VL-2B) into more effective ones. Our results also show that strong problem-solving skills do not guarantee high-quality tutoring and that performance optimization-guided refinements can yield more educationally aligned tutor models. This approach opens avenues for building MLLMs that serve not only as problem solvers, but as genuinely helpful educational assistants. 

---
# Bridging the Gap: An Intermediate Language for Enhanced and Cost-Effective Grapheme-to-Phoneme Conversion with Homographs with Multiple Pronunciations Disambiguation 

**Authors**: Abbas Bertina, Shahab Beirami, Hossein Biniazian, Elham Esmaeilnia, Soheil Shahi, Mahdi Pirnia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06599)  

**Abstract**: Grapheme-to-phoneme (G2P) conversion for Persian presents unique challenges due to its complex phonological features, particularly homographs and Ezafe, which exist in formal and informal language contexts. This paper introduces an intermediate language specifically designed for Persian language processing that addresses these challenges through a multi-faceted approach. Our methodology combines two key components: Large Language Model (LLM) prompting techniques and a specialized sequence-to-sequence machine transliteration architecture. We developed and implemented a systematic approach for constructing a comprehensive lexical database for homographs with multiple pronunciations disambiguation often termed polyphones, utilizing formal concept analysis for semantic differentiation. We train our model using two distinct datasets: the LLM-generated dataset for formal and informal Persian and the B-Plus podcasts for informal language variants. The experimental results demonstrate superior performance compared to existing state-of-the-art approaches, particularly in handling the complexities of Persian phoneme conversion. Our model significantly improves Phoneme Error Rate (PER) metrics, establishing a new benchmark for Persian G2P conversion accuracy. This work contributes to the growing research in low-resource language processing and provides a robust solution for Persian text-to-speech systems and demonstrating its applicability beyond Persian. Specifically, the approach can extend to languages with rich homographic phenomena such as Chinese and Arabic 

---
# Direct Density Ratio Optimization: A Statistically Consistent Approach to Aligning Large Language Models 

**Authors**: Rei Higuchi, Taiji Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2505.07558)  

**Abstract**: Aligning large language models (LLMs) with human preferences is crucial for safe deployment, yet existing methods assume specific preference models like Bradley-Terry model. This assumption leads to statistical inconsistency, where more data doesn't guarantee convergence to true human preferences. To address this critical gap, we introduce a novel alignment method Direct Density Ratio Optimization (DDRO). DDRO directly estimates the density ratio between preferred and unpreferred output distributions, circumventing the need for explicit human preference modeling. We theoretically prove that DDRO is statistically consistent, ensuring convergence to the true preferred distribution as the data size grows, regardless of the underlying preference structure. Experiments demonstrate that DDRO achieves superior performance compared to existing methods on many major benchmarks. DDRO unlocks the potential for truly data-driven alignment, paving the way for more reliable and human-aligned LLMs. 

---
# Evaluating LLM-Generated Q&A Test: a Student-Centered Study 

**Authors**: Anna Wróblewska, Bartosz Grabek, Jakub Świstak, Daniel Dan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06591)  

**Abstract**: This research prepares an automatic pipeline for generating reliable question-answer (Q&A) tests using AI chatbots. We automatically generated a GPT-4o-mini-based Q&A test for a Natural Language Processing course and evaluated its psychometric and perceived-quality metrics with students and experts. A mixed-format IRT analysis showed that the generated items exhibit strong discrimination and appropriate difficulty, while student and expert star ratings reflect high overall quality. A uniform DIF check identified two items for review. These findings demonstrate that LLM-generated assessments can match human-authored tests in psychometric performance and user satisfaction, illustrating a scalable approach to AI-assisted assessment development. 

---
# One Trigger Token Is Enough: A Defense Strategy for Balancing Safety and Usability in Large Language Models 

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07167)  

**Abstract**: Large Language Models (LLMs) have been extensively used across diverse domains, including virtual assistants, automated code generation, and scientific research. However, they remain vulnerable to jailbreak attacks, which manipulate the models into generating harmful responses despite safety alignment. Recent studies have shown that current safety-aligned LLMs often undergo the shallow safety alignment, where the first few tokens largely determine whether the response will be harmful. Through comprehensive observations, we find that safety-aligned LLMs and various defense strategies generate highly similar initial tokens in their refusal responses, which we define as safety trigger tokens. Building on this insight, we propose \texttt{D-STT}, a simple yet effective defense algorithm that identifies and explicitly decodes safety trigger tokens of the given safety-aligned LLM to trigger the model's learned safety patterns. In this process, the safety trigger is constrained to a single token, which effectively preserves model usability by introducing minimum intervention in the decoding process. Extensive experiments across diverse jailbreak attacks and benign prompts demonstrate that \ours significantly reduces output harmfulness while preserving model usability and incurring negligible response time overhead, outperforming ten baseline methods. 

---
# Pre-training vs. Fine-tuning: A Reproducibility Study on Dense Retrieval Knowledge Acquisition 

**Authors**: Zheng Yao, Shuai Wang, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07166)  

**Abstract**: Dense retrievers utilize pre-trained backbone language models (e.g., BERT, LLaMA) that are fine-tuned via contrastive learning to perform the task of encoding text into sense representations that can be then compared via a shallow similarity operation, e.g. inner product. Recent research has questioned the role of fine-tuning vs. that of pre-training within dense retrievers, specifically arguing that retrieval knowledge is primarily gained during pre-training, meaning knowledge not acquired during pre-training cannot be sub-sequentially acquired via fine-tuning. We revisit this idea here as the claim was only studied in the context of a BERT-based encoder using DPR as representative dense retriever. We extend the previous analysis by testing other representation approaches (comparing the use of CLS tokens with that of mean pooling), backbone architectures (encoder-only BERT vs. decoder-only LLaMA), and additional datasets (MSMARCO in addition to Natural Questions). Our study confirms that in DPR tuning, pre-trained knowledge underpins retrieval performance, with fine-tuning primarily adjusting neuron activation rather than reorganizing knowledge. However, this pattern does not hold universally, such as in mean-pooled (Contriever) and decoder-based (LLaMA) models. We ensure full reproducibility and make our implementation publicly available at this https URL. 

---
# Reassessing Large Language Model Boolean Query Generation for Systematic Reviews 

**Authors**: Shuai Wang, Harrisen Scells, Bevan Koopman, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07155)  

**Abstract**: Systematic reviews are comprehensive literature reviews that address highly focused research questions and represent the highest form of evidence in medicine. A critical step in this process is the development of complex Boolean queries to retrieve relevant literature. Given the difficulty of manually constructing these queries, recent efforts have explored Large Language Models (LLMs) to assist in their formulation. One of the first studies,Wang et al., investigated ChatGPT for this task, followed by Staudinger et al., which evaluated multiple LLMs in a reproducibility study. However, the latter overlooked several key aspects of the original work, including (i) validation of generated queries, (ii) output formatting constraints, and (iii) selection of examples for chain-of-thought (Guided) prompting. As a result, its findings diverged significantly from the original study. In this work, we systematically reproduce both studies while addressing these overlooked factors. Our results show that query effectiveness varies significantly across models and prompt designs, with guided query formulation benefiting from well-chosen seed studies. Overall, prompt design and model selection are key drivers of successful query formulation. Our findings provide a clearer understanding of LLMs' potential in Boolean query generation and highlight the importance of model- and prompt-specific optimisations. The complex nature of systematic reviews adds to challenges in both developing and reproducing methods but also highlights the importance of reproducibility studies in this domain. 

---
# REFINE-AF: A Task-Agnostic Framework to Align Language Models via Self-Generated Instructions using Reinforcement Learning from Automated Feedback 

**Authors**: Aniruddha Roy, Pretam Ray, Abhilash Nandy, Somak Aditya, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06548)  

**Abstract**: Instruction-based Large Language Models (LLMs) have proven effective in numerous few-shot or zero-shot Natural Language Processing (NLP) tasks. However, creating human-annotated instruction data is time-consuming, expensive, and often limited in quantity and task diversity. Previous research endeavors have attempted to address this challenge by proposing frameworks capable of generating instructions in a semi-automated and task-agnostic manner directly from the model itself. Many of these efforts have relied on large API-only parameter-based models such as GPT-3.5 (175B), which are expensive, and subject to limits on a number of queries. This paper explores the performance of three open-source small LLMs such as LLaMA 2-7B, LLama 2-13B, and Mistral 7B, using a semi-automated framework, thereby reducing human intervention, effort, and cost required to generate an instruction dataset for fine-tuning LLMs. Furthermore, we demonstrate that incorporating a Reinforcement Learning (RL) based training algorithm into this LLMs-based framework leads to further enhancements. Our evaluation of the dataset reveals that these RL-based frameworks achieve a substantial improvements in 63-66% of the tasks compared to previous approaches. 

---
# Utilizing LLMs to Investigate the Disputed Role of Evidence in Electronic Cigarette Health Policy Formation in Australia and the UK 

**Authors**: Damian Curran, Brian Chapman, Mike Conway  

**Link**: [PDF](https://arxiv.org/pdf/2505.06782)  

**Abstract**: Australia and the UK have developed contrasting approaches to the regulation of electronic cigarettes, with - broadly speaking - Australia adopting a relatively restrictive approach and the UK adopting a more permissive approach. Notably, these divergent policies were developed from the same broad evidence base. In this paper, to investigate differences in how the two jurisdictions manage and present evidence, we developed and evaluated a Large Language Model-based sentence classifier to perform automated analyses of electronic cigarette-related policy documents drawn from official Australian and UK legislative processes (109 documents in total). Specifically, we utilized GPT-4 to automatically classify sentences based on whether they contained claims that e-cigarettes were broadly helpful or harmful for public health. Our LLM-based classifier achieved an F-score of 0.9. Further, when applying the classifier to our entire sentence-level corpus, we found that Australian legislative documents show a much higher proportion of harmful statements, and a lower proportion of helpful statements compared to the expected values, with the opposite holding for the UK. In conclusion, this work utilized an LLM-based approach to provide evidence to support the contention that - drawing on the same evidence base - Australian ENDS-related policy documents emphasize the harms associated with ENDS products and UK policy documents emphasize the benefits. Further, our approach provides a starting point for using LLM-based methods to investigate the complex relationship between evidence and health policy formation. 

---
# Web Page Classification using LLMs for Crawling Support 

**Authors**: Yuichi Sasazawa, Yasuhiro Sogawa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06972)  

**Abstract**: A web crawler is a system designed to collect web pages, and efficient crawling of new pages requires appropriate algorithms. While website features such as XML sitemaps and the frequency of past page updates provide important clues for accessing new pages, their universal application across diverse conditions is challenging. In this study, we propose a method to efficiently collect new pages by classifying web pages into two types, "Index Pages" and "Content Pages," using a large language model (LLM), and leveraging the classification results to select index pages as starting points for accessing new pages. We construct a dataset with automatically annotated web page types and evaluate our approach from two perspectives: the page type classification performance and coverage of new pages. Experimental results demonstrate that the LLM-based method outperformed baseline methods in both evaluation metrics. 

---
# Lossless Compression of Large Language Model-Generated Text via Next-Token Prediction 

**Authors**: Yu Mao, Holger Pirk, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.06297)  

**Abstract**: As large language models (LLMs) continue to be deployed and utilized across domains, the volume of LLM-generated data is growing rapidly. This trend highlights the increasing importance of effective and lossless compression for such data in modern text management systems. However, compressing LLM-generated data presents unique challenges compared to traditional human- or machine-generated content. Traditional machine-generated data is typically derived from computational processes or device outputs, often highly structured and limited to low-level elements like labels or numerical values. This structure enables conventional lossless compressors to perform efficiently. In contrast, LLM-generated data is more complex and diverse, requiring new approaches for effective compression. In this work, we conduct the first systematic investigation of lossless compression techniques tailored specifically to LLM-generated data. Notably, because LLMs are trained via next-token prediction, we find that LLM-generated data is highly predictable for the models themselves. This predictability enables LLMs to serve as efficient compressors of their own outputs. Through extensive experiments with 14 representative LLMs and 8 LLM-generated datasets from diverse domains, we show that LLM-based prediction methods achieve remarkable compression rates, exceeding 20x, far surpassing the 3x rate achieved by Gzip, a widely used general-purpose compressor. Furthermore, this advantage holds across different LLM sizes and dataset types, demonstrating the robustness and practicality of LLM-based methods in lossless text compression under generative AI workloads. 

---
# Bridging Ears and Eyes: Analyzing Audio and Visual Large Language Models to Humans in Visible Sound Recognition and Reducing Their Sensory Gap via Cross-Modal Distillation 

**Authors**: Xilin Jiang, Junkai Wu, Vishal Choudhari, Nima Mesgarani  

**Link**: [PDF](https://arxiv.org/pdf/2505.06803)  

**Abstract**: Audio large language models (LLMs) are considered experts at recognizing sound objects, yet their performance relative to LLMs in other sensory modalities, such as visual or audio-visual LLMs, and to humans using their ears, eyes, or both remains unexplored. To investigate this, we systematically evaluate audio, visual, and audio-visual LLMs, specifically Qwen2-Audio, Qwen2-VL, and Qwen2.5-Omni, against humans in recognizing sound objects of different classes from audio-only, silent video, or sounded video inputs. We uncover a performance gap between Qwen2-Audio and Qwen2-VL that parallels the sensory discrepancy between human ears and eyes. To reduce this gap, we introduce a cross-modal distillation framework, where an LLM in one modality serves as the teacher and another as the student, with knowledge transfer in sound classes predicted as more challenging to the student by a heuristic model. Distillation in both directions, from Qwen2-VL to Qwen2-Audio and vice versa, leads to notable improvements, particularly in challenging classes. This work highlights the sensory gap in LLMs from a human-aligned perspective and proposes a principled approach to enhancing modality-specific perception in multimodal LLMs. 

---
# Knowledge Distillation for Enhancing Walmart E-commerce Search Relevance Using Large Language Models 

**Authors**: Hongwei Shang, Nguyen Vo, Nitin Yadav, Tian Zhang, Ajit Puthenputhussery, Xunfan Cai, Shuyi Chen, Prijith Chandran, Changsung Kang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07105)  

**Abstract**: Ensuring the products displayed in e-commerce search results are relevant to users queries is crucial for improving the user experience. With their advanced semantic understanding, deep learning models have been widely used for relevance matching in search tasks. While large language models (LLMs) offer superior ranking capabilities, it is challenging to deploy LLMs in real-time systems due to the high-latency requirements. To leverage the ranking power of LLMs while meeting the low-latency demands of production systems, we propose a novel framework that distills a high performing LLM into a more efficient, low-latency student model. To help the student model learn more effectively from the teacher model, we first train the teacher LLM as a classification model with soft targets. Then, we train the student model to capture the relevance margin between pairs of products for a given query using mean squared error loss. Instead of using the same training data as the teacher model, we significantly expand the student model dataset by generating unlabeled data and labeling it with the teacher model predictions. Experimental results show that the student model performance continues to improve as the size of the augmented training data increases. In fact, with enough augmented data, the student model can outperform the teacher model. The student model has been successfully deployed in production at this http URL with significantly positive metrics. 

---
# Injecting Knowledge Graphs into Large Language Models 

**Authors**: Erica Coppolillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07554)  

**Abstract**: Integrating structured knowledge from Knowledge Graphs (KGs) into Large Language Models (LLMs) remains a key challenge for symbolic reasoning. Existing methods mainly rely on prompt engineering or fine-tuning, which lose structural fidelity or incur high computational costs. Building on recent encoding techniques which integrate graph embeddings within the LLM input as tokens, we extend this paradigm to the KG domain by leveraging Knowledge Graph Embedding (KGE) models, thus enabling graph-aware reasoning. Our approach is model-agnostic, resource-efficient, and compatible with any LLMs. Extensive experimentation on synthetic and real-world datasets shows that our method improves reasoning performance over established baselines, further achieving the best trade-off in terms of accuracy and efficiency against state-of-the-art LLMs. 

---
# Why Uncertainty Estimation Methods Fall Short in RAG: An Axiomatic Analysis 

**Authors**: Heydar Soudani, Evangelos Kanoulas, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07459)  

**Abstract**: Large Language Models (LLMs) are valued for their strong performance across various tasks, but they also produce inaccurate or misleading outputs. Uncertainty Estimation (UE) quantifies the model's confidence and helps users assess response reliability. However, existing UE methods have not been thoroughly examined in scenarios like Retrieval-Augmented Generation (RAG), where the input prompt includes non-parametric knowledge. This paper shows that current UE methods cannot reliably assess correctness in the RAG setting. We further propose an axiomatic framework to identify deficiencies in existing methods and guide the development of improved approaches. Our framework introduces five constraints that an effective UE method should meet after incorporating retrieved documents into the LLM's prompt. Experimental results reveal that no existing UE method fully satisfies all the axioms, explaining their suboptimal performance in RAG. We further introduce a simple yet effective calibration function based on our framework, which not only satisfies more axioms than baseline methods but also improves the correlation between uncertainty estimates and correctness. 

---
