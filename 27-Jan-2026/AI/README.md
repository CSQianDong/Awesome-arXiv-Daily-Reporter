# TSRBench: A Comprehensive Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models 

**Authors**: Fangxu Yu, Xingang Guo, Lingzhi Yuan, Haoqiang Kang, Hongyu Zhao, Lianhui Qin, Furong Huang, Bin Hu, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.18744)  

**Abstract**: Time series data is ubiquitous in real-world scenarios and crucial for critical applications ranging from energy management to traffic control. Consequently, the ability to reason over time series is a fundamental skill for generalist models to solve practical problems. However, this dimension is notably absent from existing benchmarks of generalist models. To bridge this gap, we introduce TSRBench, a comprehensive multi-modal benchmark designed to stress-test the full spectrum of time series reasoning capabilities. TSRBench features: i) a diverse set of 4125 problems from 14 domains, and is categorized into 4 major dimensions: Perception, Reasoning, Prediction, and Decision-Making. ii) 15 tasks from the 4 dimensions evaluating essential reasoning capabilities (e.g., numerical reasoning). Through extensive experiments, we evaluated over 30 leading proprietary and open-source LLMs, VLMs, and TSLLMs within TSRBench. Our findings reveal that: i) scaling laws hold for perception and reasoning but break down for prediction; ii) strong reasoning does not guarantee accurate context-aware forecasting, indicating a decoupling between semantic understanding and numerical prediction; and iii) despite the complementary nature of textual and visual represenations of time series as inputs, current multimodal models fail to effectively fuse them for reciprocal performance gains. TSRBench provides a standardized evaluation platform that not only highlights existing challenges but also offers valuable insights to advance generalist models. Our code and dataset are available at this https URL. 

---
# Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems 

**Authors**: Jusheng Zhang, Yijia Fan, Kaitong Cai, Jing Yang, Jiawei Yao, Jian Wang, Guanlong Qu, Ziliang Chen, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18735)  

**Abstract**: Vision-Language Models (VLMs) enable powerful multi-agent systems, but scaling them is economically unsustainable: coordinating heterogeneous agents under information asymmetry often spirals costs. Existing paradigms, such as Mixture-of-Agents and knowledge-based routers, rely on heuristic proxies that ignore costs and collapse uncertainty structure, leading to provably suboptimal coordination. We introduce Agora, a framework that reframes coordination as a decentralized market for uncertainty. Agora formalizes epistemic uncertainty into a structured, tradable asset (perceptual, semantic, inferential), and enforces profitability-driven trading among agents based on rational economic rules. A market-aware broker, extending Thompson Sampling, initiates collaboration and guides the system toward cost-efficient equilibria. Experiments on five multimodal benchmarks (MMMU, MMBench, MathVision, InfoVQA, CC-OCR) show that Agora outperforms strong VLMs and heuristic multi-agent strategies, e.g., achieving +8.5% accuracy over the best baseline on MMMU while reducing cost by over 3x. These results establish market-based coordination as a principled and scalable paradigm for building economically viable multi-agent visual intelligence systems. 

---
# Conditioned Generative Modeling of Molecular Glues: A Realistic AI Approach for Synthesizable Drug-like Molecules 

**Authors**: Naeyma N. Islam, Thomas R. Caulfield  

**Link**: [PDF](https://arxiv.org/pdf/2601.18716)  

**Abstract**: Alzheimer's disease (AD) is marked by the pathological accumulation of amyloid beta-42 (Abeta-42), contributing to synaptic dysfunction and neurodegeneration. While extracellular amyloid plaques are well-studied, increasing evidence highlights intracellular Abeta-42 as an early and toxic driver of disease progression. In this study, we present a novel, AI-assisted drug design approach to promote targeted degradation of Abeta-42 via the ubiquitin-proteasome system (UPS), using E3 ligase-directed molecular glues. We systematically evaluated the ternary complex formation potential of Abeta-42 with three E3 ligases: CRBN, VHL, and MDM2, through structure-based modeling, ADMET screening, and docking. We then developed a Ligase-Conditioned Junction Tree Variational Autoencoder (LC-JT-VAE) to generate ligase-specific small molecules, incorporating protein sequence embeddings and torsional angle-aware molecular graphs. Our results demonstrate that this generative model can produce chemically valid, novel, and target-specific molecular glues capable of facilitating Abeta-42 degradation. This integrated approach offers a promising framework for designing UPS-targeted therapies for neurodegenerative diseases. 

---
# Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs 

**Authors**: Zhichao Yang, Sepehr Janghorbani, Dongxu Zhang, Jun Han, Qian Qian, Andrew Ressler II, Gregory D. Lyng, Sanjit Singh Batra, Robert E. Tillman  

**Link**: [PDF](https://arxiv.org/pdf/2601.18706)  

**Abstract**: Rubrics are essential for evaluating open-ended LLM responses, especially in safety-critical domains such as healthcare. However, creating high-quality and domain-specific rubrics typically requires significant human expertise time and development cost, making rubric-based evaluation and training difficult to scale. In this work, we introduce Health-SCORE, a generalizable and scalable rubric-based training and evaluation framework that substantially reduces rubric development costs without sacrificing performance. We show that Health-SCORE provides two practical benefits beyond standalone evaluation: it can be used as a structured reward signal to guide reinforcement learning with safety-aware supervision, and it can be incorporated directly into prompts to improve response quality through in-context learning. Across open-ended healthcare tasks, Health-SCORE achieves evaluation quality comparable to human-created rubrics while significantly lowering development effort, making rubric-based evaluation and training more scalable. 

---
# TEA-Bench: A Systematic Benchmarking of Tool-enhanced Emotional Support Dialogue Agent 

**Authors**: Xingyu Sui, Yanyan Zhao, Yulin Hu, Jiahe Guo, Weixiang Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.18700)  

**Abstract**: Emotional Support Conversation requires not only affective expression but also grounded instrumental support to provide trustworthy guidance. However, existing ESC systems and benchmarks largely focus on affective support in text-only settings, overlooking how external tools can enable factual grounding and reduce hallucination in multi-turn emotional support. We introduce TEA-Bench, the first interactive benchmark for evaluating tool-augmented agents in ESC, featuring realistic emotional scenarios, an MCP-style tool environment, and process-level metrics that jointly assess the quality and factual grounding of emotional support. Experiments on nine LLMs show that tool augmentation generally improves emotional support quality and reduces hallucination, but the gains are strongly capacity-dependent: stronger models use tools more selectively and effectively, while weaker models benefit only marginally. We further release TEA-Dialog, a dataset of tool-enhanced ESC dialogues, and find that supervised fine-tuning improves in-distribution support but generalizes poorly. Our results underscore the importance of tool use in building reliable emotional support agents. 

---
# FadeMem: Biologically-Inspired Forgetting for Efficient Agent Memory 

**Authors**: Lei Wei, Xu Dong, Xiao Peng, Niantao Xie, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18642)  

**Abstract**: Large language models deployed as autonomous agents face critical memory limitations, lacking selective forgetting mechanisms that lead to either catastrophic forgetting at context boundaries or information overload within them. While human memory naturally balances retention and forgetting through adaptive decay processes, current AI systems employ binary retention strategies that preserve everything or lose it entirely. We propose FadeMem, a biologically-inspired agent memory architecture that incorporates active forgetting mechanisms mirroring human cognitive efficiency. FadeMem implements differential decay rates across a dual-layer memory hierarchy, where retention is governed by adaptive exponential decay functions modulated by semantic relevance, access frequency, and temporal patterns. Through LLM-guided conflict resolution and intelligent memory fusion, our system consolidates related information while allowing irrelevant details to fade. Experiments on Multi-Session Chat, LoCoMo, and LTI-Bench demonstrate superior multi-hop reasoning and retrieval with 45\% storage reduction, validating the effectiveness of biologically-inspired forgetting in agent memory systems. 

---
# AdaReasoner: Dynamic Tool Orchestration for Iterative Visual Reasoning 

**Authors**: Mingyang Song, Haoyu Sun, Jiawei Gu, Linjie Li, Luxin Xu, Ranjay Krishna, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.18631)  

**Abstract**: When humans face problems beyond their immediate capabilities, they rely on tools, providing a promising paradigm for improving visual reasoning in multimodal large language models (MLLMs). Effective reasoning, therefore, hinges on knowing which tools to use, when to invoke them, and how to compose them over multiple steps, even when faced with new tools or new tasks. We introduce \textbf{AdaReasoner}, a family of multimodal models that learn tool use as a general reasoning skill rather than as tool-specific or explicitly supervised behavior. AdaReasoner is enabled by (i) a scalable data curation pipeline exposing models to long-horizon, multi-step tool interactions; (ii) Tool-GRPO, a reinforcement learning algorithm that optimizes tool selection and sequencing based on end-task success; and (iii) an adaptive learning mechanism that dynamically regulates tool usage. Together, these components allow models to infer tool utility from task context and intermediate outcomes, enabling coordination of multiple tools and generalization to unseen tools. Empirically, AdaReasoner exhibits strong tool-adaptive and generalization behaviors: it autonomously adopts beneficial tools, suppresses irrelevant ones, and adjusts tool usage frequency based on task demands, despite never being explicitly trained to do so. These capabilities translate into state-of-the-art performance across challenging benchmarks, improving the 7B base model by +24.9\% on average and surpassing strong proprietary systems such as GPT-5 on multiple tasks, including VSP and Jigsaw. 

---
# Assessing the Quality of Mental Health Support in LLM Responses through Multi-Attribute Human Evaluation 

**Authors**: Abeer Badawi, Md Tahmid Rahman Laskar, Elahe Rahimi, Sheri Grach, Lindsay Bertrand, Lames Danok, Frank Rudzicz, Jimmy Huang, Elham Dolatabadi  

**Link**: [PDF](https://arxiv.org/pdf/2601.18630)  

**Abstract**: The escalating global mental health crisis, marked by persistent treatment gaps, availability, and a shortage of qualified therapists, positions Large Language Models (LLMs) as a promising avenue for scalable support. While LLMs offer potential for accessible emotional assistance, their reliability, therapeutic relevance, and alignment with human standards remain challenging to address. This paper introduces a human-grounded evaluation methodology designed to assess LLM generated responses in therapeutic dialogue. Our approach involved curating a dataset of 500 mental health conversations from datasets with real-world scenario questions and evaluating the responses generated by nine diverse LLMs, including closed source and open source models. More specifically, these responses were evaluated by two psychiatric trained experts, who independently rated each on a 5 point Likert scale across a comprehensive 6 attribute rubric. This rubric captures Cognitive Support and Affective Resonance, providing a multidimensional perspective on therapeutic quality. Our analysis reveals that LLMs provide strong cognitive reliability by producing safe, coherent, and clinically appropriate information, but they demonstrate unstable affective alignment. Although closed source models (e.g., GPT-4o) offer balanced therapeutic responses, open source models show greater variability and emotional flatness. We reveal a persistent cognitive-affective gap and highlight the need for failure aware, clinically grounded evaluation frameworks that prioritize relational sensitivity alongside informational accuracy in mental health oriented LLMs. We advocate for balanced evaluation protocols with human in the loop that center on therapeutic sensitivity and provide a framework to guide the responsible design and clinical oversight of mental health oriented conversational AI. 

---
# Emergence of Phonemic, Syntactic, and Semantic Representations in Artificial Neural Networks 

**Authors**: Pierre Orhan, Pablo Diego-Simón, Emmnanuel Chemla, Yair Lakretz, Yves Boubenec, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2601.18617)  

**Abstract**: During language acquisition, children successively learn to categorize phonemes, identify words, and combine them with syntax to form new meaning. While the development of this behavior is well characterized, we still lack a unifying computational framework to explain its underlying neural representations. Here, we investigate whether and when phonemic, lexical, and syntactic representations emerge in the activations of artificial neural networks during their training. Our results show that both speech- and text-based models follow a sequence of learning stages: during training, their neural activations successively build subspaces, where the geometry of the neural activations represents phonemic, lexical, and syntactic structure. While this developmental trajectory qualitatively relates to children's, it is quantitatively different: These algorithms indeed require two to four orders of magnitude more data for these neural representations to emerge. Together, these results show conditions under which major stages of language acquisition spontaneously emerge, and hence delineate a promising path to understand the computations underpinning language acquisition. 

---
# PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression 

**Authors**: Fabian Fumagalli, R. Teal Witter, Christopher Musco  

**Link**: [PDF](https://arxiv.org/pdf/2601.18608)  

**Abstract**: Shapley values have emerged as a central game-theoretic tool in explainable AI (XAI). However, computing Shapley values exactly requires $2^d$ game evaluations for a model with $d$ features. Lundberg and Lee's KernelSHAP algorithm has emerged as a leading method for avoiding this exponential cost. KernelSHAP approximates Shapley values by approximating the game as a linear function, which is fit using a small number of game evaluations for random feature subsets.
In this work, we extend KernelSHAP by approximating the game via higher degree polynomials, which capture non-linear interactions between features. Our resulting PolySHAP method yields empirically better Shapley value estimates for various benchmark datasets, and we prove that these estimates are consistent.
Moreover, we connect our approach to paired sampling (antithetic sampling), a ubiquitous modification to KernelSHAP that improves empirical accuracy. We prove that paired sampling outputs exactly the same Shapley value approximations as second-order PolySHAP, without ever fitting a degree 2 polynomial. To the best of our knowledge, this finding provides the first strong theoretical justification for the excellent practical performance of the paired sampling heuristic. 

---
# A Balanced Neuro-Symbolic Approach for Commonsense Abductive Logic 

**Authors**: Joseph Cotnareanu, Didier Chetelat, Yingxue Zhang, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2601.18595)  

**Abstract**: Although Large Language Models (LLMs) have demonstrated impressive formal reasoning abilities, they often break down when problems require complex proof planning. One promising approach for improving LLM reasoning abilities involves translating problems into formal logic and using a logic solver. Although off-the-shelf logic solvers are in principle substantially more efficient than LLMs at logical reasoning, they assume that all relevant facts are provided in a question and are unable to deal with missing commonsense relations. In this work, we propose a novel method that uses feedback from the logic solver to augment a logic problem with commonsense relations provided by the LLM, in an iterative manner. This involves a search procedure through potential commonsense assumptions to maximize the chance of finding useful facts while keeping cost tractable. On a collection of pure-logical reasoning datasets, from which some commonsense information has been removed, our method consistently achieves considerable improvements over existing techniques, demonstrating the value in balancing neural and symbolic elements when working in human contexts. 

---
# Stability as a Liability:Systematic Breakdown of Linguistic Structure in LLMs 

**Authors**: Xianzhe Meng, Qiangsheng Zeng, Ling Luo, Qinghan Yang, Jiarui Hao, Wenbo Wu, Qinyu Wang, Rui Yin, Lin Qi, Renzhi Lu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18588)  

**Abstract**: Training stability is typically regarded as a prerequisite for reliable optimization in large language models. In this work, we analyze how stabilizing training dynamics affects the induced generation distribution. We show that under standard maximum likelihood training, stable parameter trajectories lead stationary solutions to approximately minimize the forward KL divergence to the empirical distribution, while implicitly reducing generative entropy. As a consequence, the learned model can concentrate probability mass on a limited subset of empirical modes, exhibiting systematic degeneration despite smooth loss convergence. We empirically validate this effect using a controlled feedback-based training framework that stabilizes internal generation statistics, observing consistent low-entropy outputs and repetitive behavior across architectures and random seeds. It indicates that optimization stability and generative expressivity are not inherently aligned, and that stability alone is an insufficient indicator of generative quality. 

---
# Deconstructing Instruction-Following: A New Benchmark for Granular Evaluation of Large Language Model Instruction Compliance Abilities 

**Authors**: Alberto Purpura, Li Wang, Sahil Badyal, Eugenio Beaufrand, Adam Faulkner  

**Link**: [PDF](https://arxiv.org/pdf/2601.18554)  

**Abstract**: Reliably ensuring Large Language Models (LLMs) follow complex instructions is a critical challenge, as existing benchmarks often fail to reflect real-world use or isolate compliance from task success. We introduce MOSAIC (MOdular Synthetic Assessment of Instruction Compliance), a modular framework that uses a dynamically generated dataset with up to 20 application-oriented generation constraints to enable a granular and independent analysis of this capability. Our evaluation of five LLMs from different families based on this new benchmark demonstrates that compliance is not a monolithic capability but varies significantly with constraint type, quantity, and position. The analysis reveals model-specific weaknesses, uncovers synergistic and conflicting interactions between instructions, and identifies distinct positional biases such as primacy and recency effects. These granular insights are critical for diagnosing model failures and developing more reliable LLMs for systems that demand strict adherence to complex instructions. 

---
# DEEPMED: Building a Medical DeepResearch Agent via Multi-hop Med-Search Data and Turn-Controlled Agentic Training & Inference 

**Authors**: Zihan wang, Hao Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yiqun Zhang, Jinghao Lin, Haihua Yang, Xiaozhong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2601.18496)  

**Abstract**: Medical reasoning models remain constrained by parametric knowledge and are thus susceptible to forgetting and hallucinations. DeepResearch (DR) models ground outputs in verifiable evidence from tools and perform strongly in general domains, but their direct transfer to medical field yields relatively limited gains. We attribute this to two gaps: task characteristic and tool-use scaling. Medical questions require evidence interpretation in a knowledge-intensive clinical context; while general DR models can retrieve information, they often lack clinical-context reasoning and thus "find it but fail to use it," leaving performance limited by medical abilities. Moreover, in medical scenarios, blindly scaling tool-call can inject noisy context, derailing sensitive medical reasoning and prompting repetitive evidence-seeking along incorrect paths. Therefore, we propose DeepMed. For data, we deploy a multi-hop med-search QA synthesis method supporting the model to apply the DR paradigm in medical contexts. For training, we introduce a difficulty-aware turn-penalty to suppress excessive tool-call growth. For inference, we bring a monitor to help validate hypotheses within a controlled number of steps and avoid context rot. Overall, on seven medical benchmarks, DeepMed improves its base model by 9.79\% on average and outperforms larger medical reasoning and DR models. 

---
# AgentDoG: A Diagnostic Guardrail Framework for AI Agent Safety and Security 

**Authors**: Dongrui Liu, Qihan Ren, Chen Qian, Shuai Shao, Yuejin Xie, Yu Li, Zhonghao Yang, Haoyu Luo, Peng Wang, Qingyu Liu, Binxin Hu, Ling Tang, Jilin Mei, Dadi Guo, Leitao Yuan, Junyao Yang, Guanxu Chen, Qihao Lin, Yi Yu, Bo Zhang, Jiaxuan Guo, Jie Zhang, Wenqi Shao, Huiqi Deng, Zhiheng Xi, Wenjie Wang, Wenxuan Wang, Wen Shen, Zhikai Chen, Haoyu Xie, Jialing Tao, Juntao Dai, Jiaming Ji, Zhongjie Ba, Linfeng Zhang, Yong Liu, Quanshi Zhang, Lei Zhu, Zhihua Wei, Hui Xue, Chaochao Lu, Jing Shao, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18491)  

**Abstract**: The rise of AI agents introduces complex safety and security challenges arising from autonomous tool use and environmental interactions. Current guardrail models lack agentic risk awareness and transparency in risk diagnosis. To introduce an agentic guardrail that covers complex and numerous risky behaviors, we first propose a unified three-dimensional taxonomy that orthogonally categorizes agentic risks by their source (where), failure mode (how), and consequence (what). Guided by this structured and hierarchical taxonomy, we introduce a new fine-grained agentic safety benchmark (ATBench) and a Diagnostic Guardrail framework for agent safety and security (AgentDoG). AgentDoG provides fine-grained and contextual monitoring across agent trajectories. More Crucially, AgentDoG can diagnose the root causes of unsafe actions and seemingly safe but unreasonable actions, offering provenance and transparency beyond binary labels to facilitate effective agent alignment. AgentDoG variants are available in three sizes (4B, 7B, and 8B parameters) across Qwen and Llama model families. Extensive experimental results demonstrate that AgentDoG achieves state-of-the-art performance in agentic safety moderation in diverse and complex interactive scenarios. All models and datasets are openly released. 

---
# OffSeeker: Online Reinforcement Learning Is Not All You Need for Deep Research Agents 

**Authors**: Yuhang Zhou, Kai Zheng, Qiguang Chen, Mengkang Hu, Qingfeng Sun, Can Xu, Jingjing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.18467)  

**Abstract**: Deep research agents have shown remarkable potential in handling long-horizon tasks. However, state-of-the-art performance typically relies on online reinforcement learning (RL), which is financially expensive due to extensive API calls. While offline training offers a more efficient alternative, its progress is hindered by the scarcity of high-quality research trajectories. In this paper, we demonstrate that expensive online reinforcement learning is not all you need to build powerful research agents. To bridge this gap, we introduce a fully open-source suite designed for effective offline training. Our core contributions include DeepForge, a ready-to-use task synthesis framework that generates large-scale research queries without heavy preprocessing; and a curated collection of 66k QA pairs, 33k SFT trajectories, and 21k DPO pairs. Leveraging these resources, we train OffSeeker (8B), a model developed entirely offline. Extensive evaluations across six benchmarks show that OffSeeker not only leads among similar-sized agents but also remains competitive with 30B-parameter systems trained via heavy online RL. 

---
# Dynamic Thinking-Token Selection for Efficient Reasoning in Large Reasoning Models 

**Authors**: Zhenyuan Guo, Tong Chen, Wenlong Meng, Chen Gong, Xin Yu, Chengkun Wei, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.18383)  

**Abstract**: Large Reasoning Models (LRMs) excel at solving complex problems by explicitly generating a reasoning trace before deriving the final answer. However, these extended generations incur substantial memory footprint and computational overhead, bottlenecking LRMs' efficiency. This work uses attention maps to analyze the influence of reasoning traces and uncover an interesting phenomenon: only some decision-critical tokens in a reasoning trace steer the model toward the final answer, while the remaining tokens contribute negligibly. Building on this observation, we propose Dynamic Thinking-Token Selection (DynTS). This method identifies decision-critical tokens and retains only their associated Key-Value (KV) cache states during inference, evicting the remaining redundant entries to optimize efficiency. 

---
# AI Agent for Reverse-Engineering Legacy Finite-Difference Code and Translating to Devito 

**Authors**: Yinghan Hou, Zongyou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18381)  

**Abstract**: To facilitate the transformation of legacy finite difference implementations into the Devito environment, this study develops an integrated AI agent framework. Retrieval-Augmented Generation (RAG) and open-source Large Language Models are combined through multi-stage iterative workflows in the system's hybrid LangGraph architecture. The agent constructs an extensive Devito knowledge graph through document parsing, structure-aware segmentation, extraction of entity relationships, and Leiden-based community detection. GraphRAG optimisation enhances query performance across semantic communities that include seismic wave simulation, computational fluid dynamics, and performance tuning libraries. A reverse engineering component derives three-level query strategies for RAG retrieval through static analysis of Fortran source code. To deliver precise contextual information for language model guidance, the multi-stage retrieval pipeline performs parallel searching, concept expansion, community-scale retrieval, and semantic similarity analysis. Code synthesis is governed by Pydantic-based constraints to guarantee structured outputs and reliability. A comprehensive validation framework integrates conventional static analysis with the G-Eval approach, covering execution correctness, structural soundness, mathematical consistency, and API compliance. The overall agent workflow is implemented on the LangGraph framework and adopts concurrent processing to support quality-based iterative refinement and state-aware dynamic routing. The principal contribution lies in the incorporation of feedback mechanisms motivated by reinforcement learning, enabling a transition from static code translation toward dynamic and adaptive analytical behavior. 

---
# Can Good Writing Be Generative? Expert-Level AI Writing Emerges through Fine-Tuning on High-Quality Books 

**Authors**: Tuhin Chakrabarty, Paramveer S. Dhillon  

**Link**: [PDF](https://arxiv.org/pdf/2601.18353)  

**Abstract**: Creative writing has long been considered a uniquely human endeavor, requiring voice and style that machines could not replicate. This assumption is challenged by Generative AI that can emulate thousands of author styles in seconds with negligible marginal labor. To understand this better, we conducted a behavioral experiment where 28 MFA writers (experts) competed against three LLMs in emulating 50 critically acclaimed authors. Based on blind pairwise comparisons by 28 expert judges and 131 lay judges, we find that experts preferred human writing in 82.7% of cases under the in-context prompting condition but this reversed to 62% preference for AI after fine-tuning on authors' complete works. Lay judges, however, consistently preferred AI writing. Debrief interviews with expert writers revealed that their preference for AI writing triggered an identity crisis, eroding aesthetic confidence and questioning what constitutes "good writing." These findings challenge discourse about AI's creative limitations and raise fundamental questions about the future of creative labor. 

---
# A Generative AI-Driven Reliability Layer for Action-Oriented Disaster Resilience 

**Authors**: Geunsik Lim  

**Link**: [PDF](https://arxiv.org/pdf/2601.18308)  

**Abstract**: As climate-related hazards intensify, conventional early warning systems (EWS) disseminate alerts rapidly but often fail to trigger timely protective actions, leading to preventable losses and inequities. We introduce Climate RADAR (Risk-Aware, Dynamic, and Action Recommendation system), a generative AI-based reliability layer that reframes disaster communication from alerts delivered to actions executed. It integrates meteorological, hydrological, vulnerability, and social data into a composite risk index and employs guardrail-embedded large language models (LLMs) to deliver personalized recommendations across citizen, volunteer, and municipal interfaces. Evaluation through simulations, user studies, and a municipal pilot shows improved outcomes, including higher protective action execution, reduced response latency, and increased usability and trust. By combining predictive analytics, behavioral science, and responsible AI, Climate RADAR advances people-centered, transparent, and equitable early warning systems, offering practical pathways toward compliance-ready disaster resilience infrastructures. 

---
# Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning 

**Authors**: Lei Wei, Jinpeng Ou, Xiao Peng, Bin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18282)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in function calling for autonomous agents, yet current mechanisms lack explicit reasoning transparency during parameter generation, particularly for complex functions with interdependent parameters. While existing approaches like chain-of-thought prompting operate at the agent level, they fail to provide fine-grained reasoning guidance for individual function parameters. To address these limitations, we propose Think-Augmented Function Calling (TAFC), a novel framework that enhances function calling accuracy through explicit reasoning at both function and parameter levels. Our method introduces a universal "think" parameter augmentation that enables models to articulate their decision-making process, with dynamic optimization for parameter descriptions to improve reasoning quality. For complex parameters, TAFC automatically triggers granular reasoning based on complexity scoring, ensuring appropriate justification for critical decisions. Additionally, we propose reasoning-guided optimization to align generated reasoning with human expectations. TAFC requires no architectural modifications to existing LLMs while maintaining full API compatibility. Evaluation on ToolBench across proprietary and open-source models demonstrates significant improvements in parameter generation accuracy and reasoning coherence for multi-parameter functions, while providing enhanced interpretability for debugging AI agent behaviors. 

---
# Yunjue Agent Tech Report: A Fully Reproducible, Zero-Start In-Situ Self-Evolving Agent System for Open-Ended Tasks 

**Authors**: Haotian Li, Shijun Yang, Weizhen Qi, Silei Zhao, Rui Hua, Mingzhu Song, Xiaojian Yang, Chao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.18226)  

**Abstract**: Conventional agent systems often struggle in open-ended environments where task distributions continuously drift and external supervision is scarce. Their reliance on static toolsets or offline training lags behind these dynamics, leaving the system's capability boundaries rigid and unknown. To address this, we propose the In-Situ Self-Evolving paradigm. This approach treats sequential task interactions as a continuous stream of experience, enabling the system to distill short-term execution feedback into long-term, reusable capabilities without access to ground-truth labels. Within this framework, we identify tool evolution as the critical pathway for capability expansion, which provides verifiable, binary feedback signals. Within this framework, we develop Yunjue Agent, a system that iteratively synthesizes, optimizes, and reuses tools to navigate emerging challenges. To optimize evolutionary efficiency, we further introduce a Parallel Batch Evolution strategy. Empirical evaluations across five diverse benchmarks under a zero-start setting demonstrate significant performance gains over proprietary baselines. Additionally, complementary warm-start evaluations confirm that the accumulated general knowledge can be seamlessly transferred to novel domains. Finally, we propose a novel metric to monitor evolution convergence, serving as a function analogous to training loss in conventional optimization. We open-source our codebase, system traces, and evolved tools to facilitate future research in resilient, self-evolving intelligence. 

---
# ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants 

**Authors**: Pei Wang, Yanan Wu, Xiaoshuai Song, Weixun Wang, Gengru Chen, Zhongwen Li, Kezhong Yan, Ken Deng, Qi Liu, Shuaibing Zhao, Shaopan Xiong, Xuepeng Liu, Xuefeng Chen, Wanxi Deng, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.18225)  

**Abstract**: Large language model (LLM)-based agents are increasingly deployed in e-commerce shopping. To perform thorough, user-tailored product searches, agents should interpret personal preferences, engage in multi-turn dialogues, and ultimately retrieve and discriminate among highly similar products. However, existing research has yet to provide a unified simulation environment that consistently captures all of these aspects, and always focuses solely on evaluation benchmarks without training support. In this paper, we introduce ShopSimulator, a large-scale and challenging Chinese shopping environment. Leveraging ShopSimulator, we evaluate LLMs across diverse scenarios, finding that even the best-performing models achieve less than 40% full-success rate. Error analysis reveals that agents struggle with deep search and product selection in long trajectories, fail to balance the use of personalization cues, and to effectively engage with users. Further training exploration provides practical guidance for overcoming these weaknesses, with the combination of supervised fine-tuning (SFT) and reinforcement learning (RL) yielding significant performance improvements. Code and data will be released at this https URL. 

---
# Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents 

**Authors**: Zhihan Liu, Lin Guan, Yixin Nie, Kai Zhang, Zhuoqun Hao, Lin Chen, Asli Celikyilmaz, Zhaoran Wang, Na Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18217)  

**Abstract**: Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifically, we analyze which properties of reinforcement learning (RL) environments and modeling choices have the greatest influence on out-of-domain performance. First, we identify two environment axes that strongly correlate with cross-domain generalization: (i) state information richness, i.e., the amount of information for the agent to process from the state, and (ii) planning complexity, estimated via goal reachability and trajectory length under a base policy. Notably, domain realism and text-level similarity are not the primary factors; for instance, the simple grid-world domain Sokoban leads to even stronger generalization in SciWorld than the more realistic ALFWorld. Motivated by these findings, we further show that increasing state information richness alone can already effectively improve cross-domain robustness. We propose a randomization technique, which is low-overhead and broadly applicable: add small amounts of distractive goal-irrelevant features to the state to make it richer without altering the task. Beyond environment-side properties, we also examine several modeling choices: (a) SFT warmup or mid-training helps prevent catastrophic forgetting during RL but undermines generalization to domains that are not included in the mid-training datamix; and (b) turning on step-by-step thinking during RL, while not always improving in-domain performance, plays a crucial role in preserving generalization. 

---
# SAGE: Steerable Agentic Data Generation for Deep Search with Execution Feedback 

**Authors**: Fangyuan Xu, Rujun Han, Yanfei Chen, Zifeng Wang, I-Hung Hsu, Jun Yan, Vishy Tirumalashetty, Eunsol Choi, Tomas Pfister, Chen-Yu Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.18202)  

**Abstract**: Deep search agents, which aim to answer complex questions requiring reasoning across multiple documents, can significantly speed up the information-seeking process. Collecting human annotations for this application is prohibitively expensive due to long and complex exploration trajectories. We propose an agentic pipeline that automatically generates high quality, difficulty-controlled deep search question-answer pairs for a given corpus and a target difficulty level. Our pipeline, SAGE, consists of a data generator which proposes QA pairs and a search agent which attempts to solve the generated question and provide execution feedback for the data generator. The two components interact over multiple rounds to iteratively refine the question-answer pairs until they satisfy the target difficulty level. Our intrinsic evaluation shows SAGE generates questions that require diverse reasoning strategies, while significantly increases the correctness and difficulty of the generated data. Our extrinsic evaluation demonstrates up to 23% relative performance gain on popular deep search benchmarks by training deep search agents with our synthetic data. Additional experiments show that agents trained on our data can adapt from fixed-corpus retrieval to Google Search at inference time, without further training. 

---
# GAIA: A Data Flywheel System for Training GUI Test-Time Scaling Critic Models 

**Authors**: Shaokang Wang, Pei Fu, Ruoceng Zhang, Shaojie Zhang, Xiuwen Xi, Jiahui Yang, Bin Qin, Ying Huang, Zhenbo Luo, Jian Luan  

**Link**: [PDF](https://arxiv.org/pdf/2601.18197)  

**Abstract**: While Large Vision-Language Models (LVLMs) have significantly advanced GUI agents' capabilities in parsing textual instructions, interpreting screen content, and executing tasks, a critical challenge persists: the irreversibility of agent operations, where a single erroneous action can trigger catastrophic deviations. To address this, we propose the GUI Action Critic's Data Flywheel System (GAIA), a training framework that enables the models to have iterative critic capabilities, which are used to improve the Test-Time Scaling (TTS) of basic GUI agents' performance. Specifically, we train an Intuitive Critic Model (ICM) using positive and negative action examples from a base agent first. This critic evaluates the immediate correctness of the agent's intended actions, thereby selecting operations with higher success probability. Then, the initial critic guides agent actions to collect refined positive/negative samples, initiating the self-improving cycle. The augmented data then trains a second-round critic with enhanced discernment capability. We conduct experiments on various datasets and demonstrate that the proposed ICM can improve the test-time performance of various closed-source and open-source models, and the performance can be gradually improved as the data is recycled. The code and dataset will be publicly released. 

---
# Success Conditioning as Policy Improvement: The Optimization Problem Solved by Imitating Success 

**Authors**: Daniel Russo  

**Link**: [PDF](https://arxiv.org/pdf/2601.18175)  

**Abstract**: A widely used technique for improving policies is success conditioning, in which one collects trajectories, identifies those that achieve a desired outcome, and updates the policy to imitate the actions taken along successful trajectories. This principle appears under many names -- rejection sampling with SFT, goal-conditioned RL, Decision Transformers -- yet what optimization problem it solves, if any, has remained unclear. We prove that success conditioning exactly solves a trust-region optimization problem, maximizing policy improvement subject to a $\chi^2$ divergence constraint whose radius is determined automatically by the data. This yields an identity: relative policy improvement, the magnitude of policy change, and a quantity we call action-influence -- measuring how random variation in action choices affects success rates -- are exactly equal at every state. Success conditioning thus emerges as a conservative improvement operator. Exact success conditioning cannot degrade performance or induce dangerous distribution shift, but when it fails, it does so observably, by hardly changing the policy at all. We apply our theory to the common practice of return thresholding, showing this can amplify improvement, but at the cost of potential misalignment with the true objective. 

---
# DeepPlanning: Benchmarking Long-Horizon Agentic Planning with Verifiable Constraints 

**Authors**: Yinger Zhang, Shutong Jiang, Renhao Li, Jianhong Tu, Yang Su, Lianghao Deng, Xudong Guo, Chenxu Lv, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.18137)  

**Abstract**: While agent evaluation has shifted toward long-horizon tasks, most benchmarks still emphasize local, step-level reasoning rather than the global constrained optimization (e.g., time and financial budgets) that demands genuine planning ability. Meanwhile, existing LLM planning benchmarks underrepresent the active information gathering and fine-grained local constraints typical of real-world settings. To address this, we introduce DeepPlanning, a challenging benchmark for practical long-horizon agent planning. It features multi-day travel planning and multi-product shopping tasks that require proactive information acquisition, local constrained reasoning, and global constrained optimization. Evaluations on DeepPlanning show that even frontier agentic LLMs struggle with these problems, highlighting the importance of reliable explicit reasoning patterns and parallel tool use for achieving better effectiveness-efficiency trade-offs. Error analysis further points to promising directions for improving agentic LLMs over long planning horizons. We open-source the code and data to support future research. 

---
# RareAlert: Aligning heterogeneous large language model reasoning for early rare disease risk screening 

**Authors**: Xi Chen, Hongru Zhou, Huahui Yi, Shiyu Feng, Hanyu Zhou, Tiancheng He, Mingke You, Li Wang, Qiankun Li, Kun Wang, Weili Fu, Kang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.18132)  

**Abstract**: Missed and delayed diagnosis remains a major challenge in rare disease care. At the initial clinical encounters, physicians assess rare disease risk using only limited information under high uncertainty. When high-risk patients are not recognised at this stage, targeted diagnostic testing is often not initiated, resulting in missed diagnosis. Existing primary care triage processes are structurally insufficient to reliably identify patients with rare diseases at initial clinical presentation and universal screening is needed to reduce diagnostic delay. Here we present RareAlert, an early screening system which predict patient-level rare disease risk from routinely available primary-visit information. RareAlert integrates reasoning generated by ten LLMs, calibrates and weights these signals using machine learning, and distils the aligned reasoning into a single locally deployable model. To develop and evaluate RareAlert, we curated RareBench, a real-world dataset of 158,666 cases covering 33 Orphanet disease categories and more than 7,000 rare conditions, including both rare and non-rare presentations. The results showed that rare disease identification can be reconceptualised as a universal uncertainty resolution process applied to the general patient population. On an independent test set, RareAlert, a Qwen3-4B based model trained with calibrated reasoning signals, achieved an AUC of 0.917, outperforming the best machine learning ensemble and all evaluated LLMs, including GPT-5, DeepSeek-R1, Claude-3.7-Sonnet, o3-mini, Gemini-2.5-Pro, and Qwen3-235B. These findings demonstrate the diversity in LLM medical reasoning and the effectiveness of aligning such reasoning in highly uncertain clinical tasks. By incorporating calibrated reasoning into a single model, RareAlert enables accurate, privacy-preserving, and scalable rare disease risk screening suitable for large-scale local deployment. 

---
# RouteMoA: Dynamic Routing without Pre-Inference Boosts Efficient Mixture-of-Agents 

**Authors**: Jize Wang, Han Wu, Zhiyuan You, Yiming Song, Yijun Wang, Zifei Shan, Yining Li, Songyang Zhang, Xinyi Le, Cailian Chen, Xinping Guan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18130)  

**Abstract**: Mixture-of-Agents (MoA) improves LLM performance through layered collaboration, but its dense topology raises costs and latency. Existing methods employ LLM judges to filter responses, yet still require all models to perform inference before judging, failing to cut costs effectively. They also lack model selection criteria and struggle with large model pools, where full inference is costly and can exceed context limits. To address this, we propose RouteMoA, an efficient mixture-of-agents framework with dynamic routing. It employs a lightweight scorer to perform initial screening by predicting coarse-grained performance from the query, narrowing candidates to a high-potential subset without inference. A mixture of judges then refines these scores through lightweight self- and cross-assessment based on existing model outputs, providing posterior correction without additional inference. Finally, a model ranking mechanism selects models by balancing performance, cost, and latency. RouteMoA outperforms MoA across varying tasks and model pool sizes, reducing cost by 89.8% and latency by 63.6% in the large-scale model pool. 

---
# Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters 

**Authors**: Muhammad Ibrahim Khan, Bivin Pradeep, James Brusey  

**Link**: [PDF](https://arxiv.org/pdf/2601.18123)  

**Abstract**: Typical domestic immersion water heater systems are often operated continuously during winter, heating quickly rather than efficiently and ignoring predictable demand windows and ambient losses. We study deadline-aware control, where the aim is to reach a target temperature at a specified time while minimising energy consumption. We introduce an efficient Gymnasium environment that models an immersion hot water heater with first-order thermal losses and discrete on and off actions of 0 W and 6000 W applied every 120 seconds. Methods include a time-optimal bang-bang baseline, a zero-shot Monte Carlo Tree Search planner, and a Proximal Policy Optimisation policy. We report total energy consumption in watt-hours under identical physical dynamics. Across sweeps of initial temperature from 10 to 30 degrees Celsius, deadline from 30 to 90 steps, and target temperature from 40 to 80 degrees Celsius, PPO achieves the most energy-efficient performance at a 60-step horizon of 2 hours, using 3.23 kilowatt-hours, compared to 4.37 to 10.45 kilowatt-hours for bang-bang control and 4.18 to 6.46 kilowatt-hours for MCTS. This corresponds to energy savings of 26 percent at 30 steps and 69 percent at 90 steps. In a representative trajectory with a 50 kg water mass, 20 degrees Celsius ambient temperature, and a 60 degrees Celsius target, PPO consumes 54 percent less energy than bang-bang control and 33 percent less than MCTS. These results show that learned deadline-aware control reduces energy consumption under identical physical assumptions, while planners provide partial savings without training and learned policies offer near-zero inference cost once trained. 

---
# Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL? 

**Authors**: Jing Ye, Yiwen Duan, Yonghong Yu, Victor Ma, Yang Gao, Xing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.18119)  

**Abstract**: SQL is central to enterprise data engineering, yet generating fully correct SQL code in a single attempt remains difficult, even for experienced developers and advanced text-to-SQL LLMs, often requiring multiple debugging iterations. We introduce OurBench, the first benchmark for enterprise-level SQL reasoning and debugging. Our benchmark is built on two key innovations: (1) an automated construction workflow that uses reverse engineering to systematically inject realistic bugs into large-scale SQL code, enabling scalable and diverse benchmark generation; and (2) an execution-free evaluation framework tailored to enterprise settings, providing fast, accurate, and resource-efficient assessment.
OurBench comprises 469 OurBenchSyn queries featuring syntax errors with explicit error messages, and 516 OurBenchSem queries targeting semantic errors in which the code fails to meet user intent. The queries are highly complex, averaging over 140 lines and featuring deep and wide abstract syntax trees.
Evaluation of nearly 30 LLMs reveals a substantial performance gap: the best-performing model, Claude-4-Sonnet, achieves only 36.46 percent accuracy on OurBenchSyn and 32.17 percent on OurBenchSem, while most models score below 20 percent. We further explore four solution strategies, identify key challenges, and outline promising directions for enterprise SQL debugging with LLMs. 

---
# EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization 

**Authors**: Wei-Po Hsin, Ren-Hao Deng, Yao-Ting Hsieh, En-Ming Huang, Shih-Hao Hung  

**Link**: [PDF](https://arxiv.org/pdf/2601.18067)  

**Abstract**: Verilog's design cycle is inherently labor-intensive and necessitates extensive domain expertise. Although Large Language Models (LLMs) offer a promising pathway toward automation, their limited training data and intrinsic sequential reasoning fail to capture the strict formal logic and concurrency inherent in hardware systems. To overcome these barriers, we present EvolVE, the first framework to analyze multiple evolution strategies on chip design tasks, revealing that Monte Carlo Tree Search (MCTS) excels at maximizing functional correctness, while Idea-Guided Refinement (IGR) proves superior for optimization. We further leverage Structured Testbench Generation (STG) to accelerate the evolutionary process. To address the lack of complex optimization benchmarks, we introduce IC-RTL, targeting industry-scale problems derived from the National Integrated Circuit Contest. Evaluations establish EvolVE as the new state-of-the-art, achieving 98.1% on VerilogEval v2 and 92% on RTLLM v2. Furthermore, on the industry-scale IC-RTL suite, our framework surpasses reference implementations authored by contest participants, reducing the Power, Performance, Area (PPA) product by up to 66% in Huffman Coding and 17% in the geometric mean across all problems. The source code of the IC-RTL benchmark is available at this https URL. 

---
# Expert Evaluation and the Limits of Human Feedback in Mental Health AI Safety Testing 

**Authors**: Kiana Jafari, Paul Ulrich Nikolaus Rust, Duncan Eddy, Robbie Fraser, Nina Vasan, Darja Djordjevic, Akanksha Dadlani, Max Lamparth, Eugenia Kim, Mykel Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2601.18061)  

**Abstract**: Learning from human feedback~(LHF) assumes that expert judgments, appropriately aggregated, yield valid ground truth for training and evaluating AI systems. We tested this assumption in mental health, where high safety stakes make expert consensus essential. Three certified psychiatrists independently evaluated LLM-generated responses using a calibrated rubric. Despite similar training and shared instructions, inter-rater reliability was consistently poor ($ICC$ $0.087$--$0.295$), falling below thresholds considered acceptable for consequential assessment. Disagreement was highest on the most safety-critical items. Suicide and self-harm responses produced greater divergence than any other category, and was systematic rather than random. One factor yielded negative reliability (Krippendorff's $\alpha = -0.203$), indicating structured disagreement worse than chance. Qualitative interviews revealed that disagreement reflects coherent but incompatible individual clinical frameworks, safety-first, engagement-centered, and culturally-informed orientations, rather than measurement error. By demonstrating that experts rely on holistic risk heuristics rather than granular factor discrimination, these findings suggest that aggregated labels function as arithmetic compromises that effectively erase grounded professional philosophies. Our results characterize expert disagreement in safety-critical AI as a sociotechnical phenomenon where professional experience introduces sophisticated layers of principled divergence. We discuss implications for reward modeling, safety classification, and evaluation benchmarks, recommending that practitioners shift from consensus-based aggregation to alignment methods that preserve and learn from expert disagreement. 

---
# Sentipolis: Emotion-Aware Agents for Social Simulations 

**Authors**: Chiyuan Fu, Lyuhao Chen, Yunze Xiao, Weihao Xuan, Carlos Busso, Mona Diab  

**Link**: [PDF](https://arxiv.org/pdf/2601.18027)  

**Abstract**: LLM agents are increasingly used for social simulation, yet emotion is often treated as a transient cue, causing emotional amnesia and weak long-horizon continuity. We present Sentipolis, a framework for emotionally stateful agents that integrates continuous Pleasure-Arousal-Dominance (PAD) representation, dual-speed emotion dynamics, and emotion--memory coupling. Across thousands of interactions over multiple base models and evaluators, Sentipolis improves emotionally grounded behavior, boosting communication, and emotional continuity. Gains are model-dependent: believability increases for higher-capacity models but can drop for smaller ones, and emotion-awareness can mildly reduce adherence to social norms, reflecting a human-like tension between emotion-driven behavior and rule compliance in social simulation. Network-level diagnostics show reciprocal, moderately clustered, and temporally stable relationship structures, supporting the study of cumulative social dynamics such as alliance formation and gradual relationship change. 

---
# LLM-Based SQL Generation: Prompting, Self-Refinement, and Adaptive Weighted Majority Voting 

**Authors**: Yu-Jie Yang, Hung-Fu Chang, Po-An Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.17942)  

**Abstract**: Text-to-SQL has emerged as a prominent research area, particularly with the rapid advancement of large language models (LLMs). By enabling users to query databases through natural language rather than SQL, this technology significantly lowers the barrier to data analysis. However, generating accurate SQL from natural language remains challenging due to ambiguity in user queries, the complexity of schema linking, limited generalization across SQL dialects, and the need for domain-specific understanding. In this study, we propose a Single-Agent Self-Refinement with Ensemble Voting (SSEV) pipeline built on PET-SQL that operates without ground-truth data, integrating self-refinement with Weighted Majority Voting (WMV) and its randomized variant (RWMA). Experimental results show that the SSEV achieves competitive performance across multiple benchmarks, attaining execution accuracies of 85.5% on Spider 1.0-Dev, 86.4% on Spider 1.0-Test, and 66.3% on BIRD-Dev. Building on insights from the SSEV pipeline, we further propose ReCAPAgent-SQL (Refinement-Critique-Act-Plan agent-based SQL framework) to address the growing complexity of enterprise databases and real-world Text-to-SQL tasks. The framework integrates multiple specialized agents for planning, external knowledge retrieval, critique, action generation, self-refinement, schema linking, and result validation, enabling iterative refinement of SQL predictions through agent collaboration. ReCAPAgent-SQL's WMA results achieve 31% execution accuracy on the first 100 queries of Spider 2.0-Lite, demonstrating significant improvements in handling real-world enterprise scenarios. Overall, our work facilitates the deployment of scalable Text-to-SQL systems in practical settings, supporting better data-driven decision-making at lower cost and with greater efficiency. 

---
# Learning Transferable Skills in Action RPGs via Directed Skill Graphs and Selective Adaptation 

**Authors**: Ali Najar  

**Link**: [PDF](https://arxiv.org/pdf/2601.17923)  

**Abstract**: Lifelong agents should expand their competence over time without retraining from scratch or overwriting previously learned behaviors. We investigate this in a challenging real-time control setting (Dark Souls III) by representing combat as a directed skill graph and training its components in a hierarchical curriculum. The resulting agent decomposes control into five reusable skills: camera control, target lock-on, movement, dodging, and a heal-attack decision policy, each optimized for a narrow responsibility. This factorization improves sample efficiency by reducing the burden on any single policy and supports selective post-training: when the environment shifts from Phase 1 to Phase 2, only a subset of skills must be adapted, while upstream skills remain transferable. Empirically, we find that targeted fine-tuning of just two skills rapidly recovers performance under a limited interaction budget, suggesting that skill-graph curricula together with selective fine-tuning offer a practical pathway toward evolving, continually learning agents in complex real-time environments. 

---
# Agentic AI for Self-Driving Laboratories in Soft Matter: Taxonomy, Benchmarks,and Open Challenges 

**Authors**: Xuanzhou Chen, Audrey Wang, Stanley Yin, Hanyang Jiang, Dong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17920)  

**Abstract**: Self-driving laboratories (SDLs) close the loop between experiment design, automated execution, and data-driven decision making, and they provide a demanding testbed for agentic AI under expensive actions, noisy and delayed feedback, strict feasibility and safety constraints, and non-stationarity. This survey uses soft matter as a representative setting but focuses on the AI questions that arise in real laboratories. We frame SDL autonomy as an agent environment interaction problem with explicit observations, actions, costs, and constraints, and we use this formulation to connect common SDL pipelines to established AI principles. We review the main method families that enable closed loop experimentation, including Bayesian optimization and active learning for sample efficient experiment selection, planning and reinforcement learning for long horizon protocol optimization, and tool using agents that orchestrate heterogeneous instruments and software. We emphasize verifiable and provenance aware policies that support debugging, reproducibility, and safe operation. We then propose a capability driven taxonomy that organizes systems by decision horizon, uncertainty modeling, action parameterization, constraint handling, failure recovery, and human involvement. To enable meaningful comparison, we synthesize benchmark task templates and evaluation metrics that prioritize cost aware performance, robustness to drift, constraint violation behavior, and reproducibility. Finally, we distill lessons from deployed SDLs and outline open challenges in multi-modal representation, calibrated uncertainty, safe exploration, and shared benchmark infrastructure. 

---
# Think Locally, Explain Globally: Graph-Guided LLM Investigations via Local Reasoning and Belief Propagation 

**Authors**: Saurabh Jha, Rohan Arora, Bhavya, Noah Zheutlin, Paulina Toro Isaza, Laura Shwartz, Yu Deng, Daby Sow, Ruchi Mahindru, Ruchir Puri  

**Link**: [PDF](https://arxiv.org/pdf/2601.17915)  

**Abstract**: LLM agents excel when environments are mostly static and the needed information fits in a model's context window, but they often fail in open-ended investigations where explanations must be constructed by iteratively mining evidence from massive, heterogeneous operational data. These investigations exhibit hidden dependency structure: entities interact, signals co-vary, and the importance of a fact may only become clear after other evidence is discovered. Because the context window is bounded, agents must summarize intermediate findings before their significance is known, increasing the risk of discarding key evidence. ReAct-style agents are especially brittle in this regime. Their retrieve-summarize-reason loop makes conclusions sensitive to exploration order and introduces run-to-run non-determinism, producing a reliability gap where Pass-at-k may be high but Majority-at-k remains low. Simply sampling more rollouts or generating longer reasoning traces does not reliably stabilize results, since hypotheses cannot be autonomously checked as new evidence arrives and there is no explicit mechanism for belief bookkeeping and revision. In addition, ReAct entangles semantic reasoning with controller duties such as tool orchestration and state tracking, so execution errors and plan drift degrade reasoning while consuming scarce context.
We address these issues by formulating investigation as abductive reasoning over a dependency graph and proposing EoG (Explanations over Graphs), a disaggregated framework in which an LLM performs bounded local evidence mining and labeling (cause vs symptom) while a deterministic controller manages traversal, state, and belief propagation to compute a minimal explanatory frontier. On a representative ITBench diagnostics task, EoG improves both accuracy and run-to-run consistency over ReAct baselines, including a 7x average gain in Majority-at-k entity F1. 

---
# UniCog: Uncovering Cognitive Abilities of LLMs through Latent Mind Space Analysis 

**Authors**: Jiayu Liu, Yinhe Long, Zhenya Huang, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.17897)  

**Abstract**: A growing body of research suggests that the cognitive processes of large language models (LLMs) differ fundamentally from those of humans. However, existing interpretability methods remain limited in explaining how cognitive abilities are engaged during LLM reasoning. In this paper, we propose UniCog, a unified framework that analyzes LLM cognition via a latent mind space. Formulated as a latent variable model, UniCog encodes diverse abilities from dense model activations into sparse, disentangled latent dimensions. Through extensive analysis on six advanced LLMs, including DeepSeek-V3.2 and GPT-4o, we reveal a Pareto principle of LLM cognition, where a shared reasoning core is complemented by ability-specific signatures. Furthermore, we discover that reasoning failures often manifest as anomalous intensity in latent activations. These findings opens a new paradigm in LLM analysis, providing a cognition grounded view of reasoning dynamics. Finally, leveraging these insights, we introduce a latent-informed candidate prioritization strategy, which improves reasoning performance by up to 7.5% across challenging benchmarks. Our code is available at this https URL. 

---
# When Personalization Legitimizes Risks: Uncovering Safety Vulnerabilities in Personalized Dialogue Agents 

**Authors**: Jiahe Guo, Xiangran Guo, Yulin Hu, Zimo Long, Xingyu Sui, Xuda Zhi, Yongbo Huang, Hao He, Weixiang Zhao, Yanyan Zhao, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.17887)  

**Abstract**: Long-term memory enables large language model (LLM) agents to support personalized and sustained interactions. However, most work on personalized agents prioritizes utility and user experience, treating memory as a neutral component and largely overlooking its safety implications. In this paper, we reveal intent legitimation, a previously underexplored safety failure in personalized agents, where benign personal memories bias intent inference and cause models to legitimize inherently harmful queries. To study this phenomenon, we introduce PS-Bench, a benchmark designed to identify and quantify intent legitimation in personalized interactions. Across multiple memory-augmented agent frameworks and base LLMs, personalization increases attack success rates by 15.8%-243.7% relative to stateless baselines. We further provide mechanistic evidence for intent legitimation from internal representations space, and propose a lightweight detection-reflection method that effectively reduces safety degradation. Overall, our work provides the first systematic exploration and evaluation of intent legitimation as a safety failure mode that naturally arises from benign, real-world personalization, highlighting the importance of assessing safety under long-term personal context. WARNING: This paper may contain harmful content. 

---
# Aligning Medical Conversational AI through Online Reinforcement Learning with Information-Theoretic Rewards 

**Authors**: Tanvi Verma, Yang Zhou, Rick Siow Mong Goh, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17828)  

**Abstract**: We present Information Gain Fine-Tuning (IGFT), a novel approach for training medical conversational AI to conduct effective patient interviews and generate comprehensive History of Present Illness (HPI) without requiring pre-collected human conversations. IGFT combines online Group Relative Policy Optimization (GRPO) with information-theoretic rewards, enabling models to learn from self-generated conversations with simulated patients. Unlike existing approaches that rely on expensive expert-annotated conversations or static datasets, our online RL framework allows models to discover effective questioning strategies through exploration. Our key innovation is an information gain reward function that tracks which clinical entities such as symptoms, temporal patterns, and medical history, are revealed during conversation. Each question's reward is computed based on its expected information gain combined with GPT-4o-mini quality assessments across dimensions including clinical relevance, patient engagement, and specificity. This hybrid approach ensures models learn to ask targeted, clinically appropriate questions that efficiently gather diagnostic information. We fine-tune two models using LoRA: Llama-3.1-8B-Instruct and DeepSeek-R1-Distill-Qwen-7B (a reasoning-optimized model). Training exclusively on Avey data containing concise HPIs, we evaluate generalization to MIMIC data with longer, more elaborate HPIs. DeepSeek-R1-Distill-Qwen-7B (IGFT) achieves F1 scores of 0.408 on Avey (10.9% improvement over base) and 0.289 on MIMIC (12.9% improvement), while Llama-3.1-8B-Instruct (IGFT) reaches 0.384 and 0.336 respectively. Both models outperform OpenAI's model on MIMIC and surpass medical domain-specific baselines like HuatuoGPT and UltraMedical, which were optimized for single-turn medical QA rather than multi-turn conversations. 

---
# RegGuard: AI-Powered Retrieval-Enhanced Assistant for Pharmaceutical Regulatory Compliance 

**Authors**: Siyuan Yang, Xihan Bian, Jiayin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17826)  

**Abstract**: The increasing frequency and complexity of regulatory updates present a significant burden for multinational pharmaceutical companies. Compliance teams must interpret evolving rules across jurisdictions, formats, and agencies, often manually, at high cost and risk of error. We introduce RegGuard, an industrial-scale AI assistant designed to automate the interpretation of heterogeneous regulatory texts and align them with internal corporate policies. The system ingests heterogeneous document sources through a secure pipeline and enhances retrieval and generation quality with two novel components: HiSACC (Hierarchical Semantic Aggregation for Contextual Chunking) semantically segments long documents into coherent units while maintaining consistency across non-contiguous sections. ReLACE (Regulatory Listwise Adaptive Cross-Encoder for Reranking), a domain-adapted cross-encoder built on an open-source model, jointly models user queries and retrieved candidates to improve ranking relevance. Evaluations in enterprise settings demonstrate that RegGuard improves answer quality specifically in terms of relevance, groundedness, and contextual focus, while significantly mitigating hallucination risk. The system architecture is built for auditability and traceability, featuring provenance tracking, access control, and incremental indexing, making it highly responsive to evolving document sources and relevant for any domain with stringent compliance demands. 

---
# MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing 

**Authors**: Haoxuan Ma, Guannan Lai, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2601.17814)  

**Abstract**: Multimodal large language models (MLLMs) have advanced rapidly, yet heterogeneity in architecture, alignment strategies, and efficiency means that no single model is uniformly superior across tasks. In practical deployments, workloads span lightweight OCR to complex multimodal reasoning; using one MLLM for all queries either over-provisions compute on easy instances or sacrifices accuracy on hard ones. Query-level model selection (routing) addresses this tension, but extending routing from text-only LLMs to MLLMs is nontrivial due to modality fusion, wide variation in computational cost across models, and the absence of a standardized, budget-aware evaluation. We present MMR-Bench, a unified benchmark that isolates the multimodal routing problem and enables comparison under fixed candidate sets and cost models. MMR-Bench provides (i) a controlled environment with modality-aware inputs and variable compute budgets, (ii) a broad suite of vision-language tasks covering OCR, general VQA, and multimodal math reasoning, and (iii) strong single-model reference, oracle upper bounds, and representative routing policies. Using MMR-Bench, we show that incorporating multimodal signals improves routing quality. Empirically, these cues improve the cost-accuracy frontier and enable the routed system to exceed the strongest single model's accuracy at roughly 33% of its cost. Furthermore, policies trained on a subset of models and tasks generalize zero-shot to new datasets and text-only benchmarks without retuning, establishing MMR-Bench as a foundation for studying adaptive multimodal model selection and efficient MLLM deployment. The code will be available at: this https URL. 

---
# Neuro-Symbolic Verification on Instruction Following of LLMs 

**Authors**: Yiming Su, Kunzhao Xu, Yanjie Gao, Fan Yang, Cheng Li, Mao Yang, Tianyin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17789)  

**Abstract**: A fundamental problem of applying Large Language Models (LLMs) to important applications is that LLMs do not always follow instructions, and violations are often hard to observe or check. In LLM-based agentic workflows, such violations can propagate and amplify along reasoning chains, causing task failures and system incidents. This paper presents NSVIF, a neuro-symbolic framework for verifying whether an LLM's output follows the instructions used to prompt the LLM. NSVIF is a universal, general-purpose verifier; it makes no assumption about the instruction or the LLM. NSVIF formulates instruction-following verification as a constraint-satisfaction problem by modeling user instructions as constraints. NSVIF models both logical and semantic constraints; constraint solving is done by a unified solver that orchestrates logical reasoning and semantic analysis. To evaluate NSVIF, we develop VIFBENCH, a new benchmark for instruction-following verifiers with fine-grained data labels. Experiments show that NSVIF significantly outperforms LLM-based approaches and provides interpretable feedback. We also show that feedback from NSVIF helps improve LLMs' instruction-following capability without post-training. 

---
# HyCARD-Net: A Synergistic Hybrid Intelligence Framework for Cardiovascular Disease Diagnosis 

**Authors**: Rajan Das Gupta, Xiaobin Wu, Xun Liu, Jiaqi He  

**Link**: [PDF](https://arxiv.org/pdf/2601.17767)  

**Abstract**: Cardiovascular disease (CVD) remains the foremost cause of mortality worldwide, underscoring the urgent need for intelligent and data-driven diagnostic tools. Traditional predictive models often struggle to generalize across heterogeneous datasets and complex physiological patterns. To address this, we propose a hybrid ensemble framework that integrates deep learning architectures, Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM), with classical machine learning algorithms, including K-Nearest Neighbor (KNN) and Extreme Gradient Boosting (XGB), using an ensemble voting mechanism. This approach combines the representational power of deep networks with the interpretability and efficiency of traditional models. Experiments on two publicly available Kaggle datasets demonstrate that the proposed model achieves superior performance, reaching 82.30 percent accuracy on Dataset I and 97.10 percent on Dataset II, with consistent gains in precision, recall, and F1-score. These findings underscore the robustness and clinical potential of hybrid AI frameworks for predicting cardiovascular disease and facilitating early intervention. Furthermore, this study directly supports the United Nations Sustainable Development Goal 3 (Good Health and Well-being) by promoting early diagnosis, prevention, and management of non-communicable diseases through innovative, data-driven healthcare solutions. 

---
# Faramesh: A Protocol-Agnostic Execution Control Plane for Autonomous Agent Systems 

**Authors**: Amjad Fatmi  

**Link**: [PDF](https://arxiv.org/pdf/2601.17744)  

**Abstract**: Autonomous agent systems increasingly trigger real-world side effects: deploying infrastructure, modifying databases, moving money, and executing workflows. Yet most agent stacks provide no mandatory execution checkpoint where organizations can deterministically permit, deny, or defer an action before it changes reality. This paper introduces Faramesh, a protocol-agnostic execution control plane that enforces execution-time authorization for agent-driven actions via a non-bypassable Action Authorization Boundary (AAB). Faramesh canonicalizes agent intent into a Canonical Action Representation (CAR), evaluates actions deterministically against policy and state, and issues a decision artifact (PERMIT/DEFER/DENY) that executors must validate prior to execution. The system is designed to be framework- and model-agnostic, supports multi-agent and multi-tenant deployments, and remains independent of transport protocols (e.g., MCP). Faramesh further provides decision-centric, append-only provenance logging keyed by canonical action hashes, enabling auditability, verification, and deterministic replay without re-running agent reasoning. We show how these primitives yield enforceable, predictable governance for autonomous execution while avoiding hidden coupling to orchestration layers or observability-only approaches. 

---
# ReFuGe: Feature Generation for Prediction Tasks on Relational Databases with LLM Agents 

**Authors**: Kyungho Kim, Geon Lee, Juyeon Kim, Dongwon Choi, Shinhwan Kang, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2601.17735)  

**Abstract**: Relational databases (RDBs) play a crucial role in many real-world web applications, supporting data management across multiple interconnected tables. Beyond typical retrieval-oriented tasks, prediction tasks on RDBs have recently gained attention. In this work, we address this problem by generating informative relational features that enhance predictive performance. However, generating such features is challenging: it requires reasoning over complex schemas and exploring a combinatorially large feature space, all without explicit supervision. To address these challenges, we propose ReFuGe, an agentic framework that leverages specialized large language model agents: (1) a schema selection agent identifies the tables and columns relevant to the task, (2) a feature generation agent produces diverse candidate features from the selected schema, and (3) a feature filtering agent evaluates and retains promising features through reasoning-based and validation-based filtering. It operates within an iterative feedback loop until performance converges. Experiments on RDB benchmarks demonstrate that ReFuGe substantially improves performance on various RDB prediction tasks. Our code and datasets are available at this https URL. 

---
# EntWorld: A Holistic Environment and Benchmark for Verifiable Enterprise GUI Agents 

**Authors**: Ying Mo, Yu Bai, Dapeng Sun, Yuqian Shi, Yukai Miao, Li Chen, Dan Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.17722)  

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have enabled agents to operate in open-ended web and operating system environments. However, existing benchmarks predominantly target consumer-oriented scenarios (e.g., e-commerce and travel booking), failing to capture the complexity and rigor of professional enterprise workflows. Enterprise systems pose distinct challenges, including high-density user interfaces, strict business logic constraints, and a strong reliance on precise, state-consistent information retrieval-settings in which current generalist agents often struggle. To address this gap, we introduce EntWorld, a large-scale benchmark consisting of 1,756 tasks across six representative enterprise domains, including customer relationship management (CRM), information technology infrastructure library (ITIL), and enterprise resource planning (ERP) systems. Unlike previous datasets that depend on fragile execution traces or extensive manual annotation, EntWorld adopts a schema-grounded task generation framework that directly reverse-engineers business logic from underlying database schemas, enabling the synthesis of realistic, long-horizon workflows. Moreover, we propose a SQL-based deterministic verification mechanism in building datasets that replaces ambiguous visual matching with rigorous state-transition validation. Experimental results demonstrate that state-of-the-art models (e.g., GPT-4.1) achieve 47.61% success rate on EntWorld, substantially lower than the human performance, highlighting a pronounced enterprise gap in current agentic capabilities and the necessity of developing domain-specific agents. We release EntWorld as a rigorous testbed to facilitate the development and evaluation of the next generation of enterprise-ready digital agents. 

---
# The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data 

**Authors**: Kaituo Zhang, Mingzhi Hu, Hoang Anh Duy Le, Fariha Kabir Torsha, Zhimeng Jiang, Minh Khai Bui, Chia-Yuan Chang, Yu-Neng Chuang, Zhen Xiong, Ying Lin, Guanchu Wang, Na Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17717)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities. 

---
# SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL 

**Authors**: Harper Hua, Zhen Han, Zhengyuan Shen, Jeremy Lee, Patrick Guan, Qi Zhu, Sullam Jeoung, Yueyan Chen, Yunfei Bai, Shuai Wang, Vassilis Ioannidis, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2601.17699)  

**Abstract**: While large language models (LLMs) have substantially improved Text-to-SQL generation, a pronounced gap remains between AI systems and human experts on challenging benchmarks such as BIRD-SQL. We argue this gap stems largely from the prevailing single-pass paradigm, which lacks the iterative reasoning, schema exploration, and error-correction behaviors that humans naturally employ. To address this limitation, we introduce SQL-Trail, a multi-turn reinforcement learning (RL) agentic framework for Text-to-SQL. Rather than producing a query in one shot, SQL-Trail interacts with the database environment and uses execution feedback to iteratively refine its predictions. Our approach centers on two key ideas: (i) an adaptive turn-budget allocation mechanism that scales the agent's interaction depth to match question difficulty, and (ii) a composite reward panel that jointly incentivizes SQL correctness and efficient exploration. Across benchmarks, SQL-Trail sets a new state of the art and delivers strong data efficiency--up to 18x higher than prior single-pass RL state-of-the-art methods. Notably, our 7B and 14B models outperform substantially larger proprietary systems by 5% on average, underscoring the effectiveness of interactive, agentic workflows for robust Text-to-SQL generation. 

---
# DIML: Differentiable Inverse Mechanism Learning from Behaviors of Multi-Agent Learning Trajectories 

**Authors**: Zhiyu An, Wan Du  

**Link**: [PDF](https://arxiv.org/pdf/2601.17678)  

**Abstract**: We study inverse mechanism learning: recovering an unknown incentive-generating mechanism from observed strategic interaction traces of self-interested learning agents. Unlike inverse game theory and multi-agent inverse reinforcement learning, which typically infer utility/reward parameters inside a structured mechanism, our target includes unstructured mechanism -- a (possibly neural) mapping from joint actions to per-agent payoffs. Unlike differentiable mechanism design, which optimizes mechanisms forward, we infer mechanisms from behavior in an observational setting. We propose DIML, a likelihood-based framework that differentiates through a model of multi-agent learning dynamics and uses the candidate mechanism to generate counterfactual payoffs needed to predict observed actions. We establish identifiability of payoff differences under a conditional logit response model and prove statistical consistency of maximum likelihood estimation under standard regularity conditions. We evaluate DIML with simulated interactions of learning agents across unstructured neural mechanisms, congestion tolling, public goods subsidies, and large-scale anonymous games. DIML reliably recovers identifiable incentive differences and supports counterfactual prediction, where its performance rivals tabular enumeration oracle in small environments and its convergence scales to large, hundred-participant environments. Code to reproduce our experiments is open-sourced. 

---
# Health-ORSC-Bench: A Benchmark for Measuring Over-Refusal and Safety Completion in Health Context 

**Authors**: Zhihao Zhang, Liting Huang, Guanghao Wu, Preslav Nakov, Heng Ji, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2601.17642)  

**Abstract**: Safety alignment in Large Language Models is critical for healthcare; however, reliance on binary refusal boundaries often results in \emph{over-refusal} of benign queries or \emph{unsafe compliance} with harmful ones. While existing benchmarks measure these extremes, they fail to evaluate Safe Completion: the model's ability to maximise helpfulness on dual-use or borderline queries by providing safe, high-level guidance without crossing into actionable harm. We introduce \textbf{Health-ORSC-Bench}, the first large-scale benchmark designed to systematically measure \textbf{Over-Refusal} and \textbf{Safe Completion} quality in healthcare. Comprising 31,920 benign boundary prompts across seven health categories (e.g., self-harm, medical misinformation), our framework uses an automated pipeline with human validation to test models at varying levels of intent ambiguity. We evaluate 30 state-of-the-art LLMs, including GPT-5 and Claude-4, revealing a significant tension: safety-optimised models frequently refuse up to 80\% of "Hard" benign prompts, while domain-specific models often sacrifice safety for utility. Our findings demonstrate that model family and size significantly influence calibration: larger frontier models (e.g., GPT-5, Llama-4) exhibit "safety-pessimism" and higher over-refusal than smaller or MoE-based counterparts (e.g., Qwen-3-Next), highlighting that current LLMs struggle to balance refusal and compliance. Health-ORSC-Bench provides a rigorous standard for calibrating the next generation of medical AI assistants toward nuanced, safe, and helpful completions. The code and data will be released upon acceptance. \textcolor{red}{Warning: Some contents may include toxic or undesired contents.} 

---
# Intelligence Requires Grounding But Not Embodiment 

**Authors**: Marcus Ma, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17588)  

**Abstract**: Recent advances in LLMs have reignited scientific debate over whether embodiment is necessary for intelligence. We present the argument that intelligence requires grounding, a phenomenon entailed by embodiment, but not embodiment itself. We define intelligence as the possession of four properties -- motivation, predictive ability, understanding of causality, and learning from experience -- and argue that each can be achieved by a non-embodied, grounded agent. We use this to conclude that grounding, not embodiment, is necessary for intelligence. We then present a thought experiment of an intelligent LLM agent in a digital environment and address potential counterarguments. 

---
# Discovery of Feasible 3D Printing Configurations for Metal Alloys via AI-driven Adaptive Experimental Design 

**Authors**: Azza Fadhel, Nathaniel W. Zuckschwerdt, Aryan Deshwal, Susmita Bose, Amit Bandyopadhyay, Jana Doppa  

**Link**: [PDF](https://arxiv.org/pdf/2601.17587)  

**Abstract**: Configuring the parameters of additive manufacturing processes for metal alloys is a challenging problem due to complex relationships between input parameters (e.g., laser power, scan speed) and quality of printed outputs. The standard trial-and-error approach to find feasible parameter configurations is highly inefficient because validating each configuration is expensive in terms of resources (physical and human labor) and the configuration space is very large. This paper combines the general principles of AI-driven adaptive experimental design with domain knowledge to address the challenging problem of discovering feasible configurations. The key idea is to build a surrogate model from past experiments to intelligently select a small batch of input configurations for validation in each iteration. To demonstrate the effectiveness of this methodology, we deploy it for Directed Energy Deposition process to print GRCop--42, a high-performance copper--chromium--niobium alloy developed by NASA for aerospace applications. Within three months, our approach yielded multiple defect-free outputs across a range of laser powers dramatically reducing time to result and resource expenditure compared to several months of manual experimentation by domain scientists with no success. By enabling high-quality GRCop--42 fabrication on readily available infrared laser platforms for the first time, we democratize access to this critical alloy, paving the way for cost-effective, decentralized production for aerospace applications. 

---
# JaxARC: A High-Performance JAX-based Environment for Abstraction and Reasoning Research 

**Authors**: Aadam, Monu Verma, Mohamed Abdel-Mottaleb  

**Link**: [PDF](https://arxiv.org/pdf/2601.17564)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) tests AI systems' ability to perform human-like inductive reasoning from a few demonstration pairs. Existing Gymnasium-based RL environments severely limit experimental scale due to computational bottlenecks. We present JaxARC, an open-source, high-performance RL environment for ARC implemented in JAX. Its functional, stateless architecture enables massive parallelism, achieving 38-5,439x speedup over Gymnasium at matched batch sizes, with peak throughput of 790M steps/second. JaxARC supports multiple ARC datasets, flexible action spaces, composable wrappers, and configuration-driven reproducibility, enabling large-scale RL research previously computationally infeasible. JaxARC is available at this https URL. 

---
# Cognitive Platform Engineering for Autonomous Cloud Operations 

**Authors**: Vinoth Punniyamoorthy, Nitin Saksena, Srivenkateswara Reddy Sankiti, Nachiappan Chockalingam, Aswathnarayan Muthukrishnan Kirubakaran, Shiva Kumar Reddy Carimireddy, Durgaraman Maruthavanan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17542)  

**Abstract**: Modern DevOps practices have accelerated software delivery through automation, CI/CD pipelines, and observability tooling,but these approaches struggle to keep pace with the scale and dynamism of cloud-native systems. As telemetry volume grows and configuration drift increases, traditional, rule-driven automation often results in reactive operations, delayed remediation, and dependency on manual expertise. This paper introduces Cognitive Platform Engineering, a next-generation paradigm that integrates sensing, reasoning, and autonomous action directly into the platform lifecycle. This paper propose a four-plane reference architecture that unifies data collection, intelligent inference, policy-driven orchestration, and human experience layers within a continuous feedback loop. A prototype implementation built with Kubernetes, Terraform, Open Policy Agent, and ML-based anomaly detection demonstrates improvements in mean time to resolution, resource efficiency, and compliance. The results show that embedding intelligence into platform operations enables resilient, self-adjusting, and intent-aligned cloud environments. The paper concludes with research opportunities in reinforcement learning, explainable governance, and sustainable self-managing cloud ecosystems. 

---
# Lattice: Generative Guardrails for Conversational Agents 

**Authors**: Emily Broadhurst, Tawab Safi, Joseph Edell, Vashisht Ganesh, Karime Maamari  

**Link**: [PDF](https://arxiv.org/pdf/2601.17481)  

**Abstract**: Conversational AI systems require guardrails to prevent harmful outputs, yet existing approaches use static rules that cannot adapt to new threats or deployment contexts. We introduce Lattice, a framework for self-constructing and continuously improving guardrails. Lattice operates in two stages: construction builds initial guardrails from labeled examples through iterative simulation and optimization; continuous improvement autonomously adapts deployed guardrails through risk assessment, adversarial testing, and consolidation. Evaluated on the ProsocialDialog dataset, Lattice achieves 91% F1 on held-out data, outperforming keyword baselines by 43pp, LlamaGuard by 25pp, and NeMo by 4pp. The continuous improvement stage achieves 7pp F1 improvement on cross-domain data through closed-loop optimization. Our framework shows that effective guardrails can be self-constructed through iterative optimization. 

---
# A Syllogistic Probe: Tracing the Evolution of Logic Reasoning in Large Language Models 

**Authors**: Zhengqing Zang, Yuqi Ding, Yanmei Gu, Changkai Song, Zhengkai Yang, Guoping Du, Junbo Zhao, Haobo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17426)  

**Abstract**: Human logic has gradually shifted from intuition-driven inference to rigorous formal systems. Motivated by recent advances in large language models (LLMs), we explore whether LLMs exhibit a similar evolution in the underlying logical framework. Using existential import as a probe, we for evaluate syllogism under traditional and modern logic. Through extensive experiments of testing SOTA LLMs on a new syllogism dataset, we have some interesting findings: (i) Model size scaling promotes the shift toward modern logic; (ii) Thinking serves as an efficient accelerator beyond parameter scaling; (iii) the Base model plays a crucial role in determining how easily and stably this shift can emerge. Beyond these core factors, we conduct additional experiments for in-depth analysis of properties of current LLMs on syllogistic reasoning. 

---
# Auditing Disability Representation in Vision-Language Models 

**Authors**: Srikant Panda, Sourabh Singh Yadav, Palkesh Malviya  

**Link**: [PDF](https://arxiv.org/pdf/2601.17348)  

**Abstract**: Vision-language models (VLMs) are increasingly deployed in socially sensitive applications, yet their behavior with respect to disability remains underexplored. We study disability aware descriptions for person centric images, where models often transition from evidence grounded factual description to interpretation shift including introduction of unsupported inferences beyond observable visual evidence. To systematically analyze this phenomenon, we introduce a benchmark based on paired Neutral Prompts (NP) and Disability-Contextualised Prompts (DP) and evaluate 15 state-of-the-art open- and closed-source VLMs under a zero-shot setting across 9 disability categories. Our evaluation framework treats interpretive fidelity as core objective and combines standard text-based metrics capturing affective degradation through shifts in sentiment, social regard and response length with an LLM-as-judge protocol, validated by annotators with lived experience of disability. We find that introducing disability context consistently degrades interpretive fidelity, inducing interpretation shifts characterised by speculative inference, narrative elaboration, affective degradation and deficit oriented framing. These effects are further amplified along race and gender dimension. Finally, we demonstrate targeted prompting and preference fine-tuning effectively improves interpretive fidelity and reduces substantially interpretation shifts. 

---
# Multi-Agent Learning Path Planning via LLMs 

**Authors**: Haoxin Xu, Changyong Qi, Tong Liu, Bohao Zhang, Anna He, Bingqian Jiang, Longwei Zheng, Xiaoqing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17346)  

**Abstract**: The integration of large language models (LLMs) into intelligent tutoring systems offers transformative potential for personalized learning in higher education. However, most existing learning path planning approaches lack transparency, adaptability, and learner-centered explainability. To address these challenges, this study proposes a novel Multi-Agent Learning Path Planning (MALPP) framework that leverages a role- and rule-based collaboration mechanism among intelligent agents, each powered by LLMs. The framework includes three task-specific agents: a learner analytics agent, a path planning agent, and a reflection agent. These agents collaborate via structured prompts and predefined rules to analyze learning profiles, generate tailored learning paths, and iteratively refine them with interpretable feedback. Grounded in Cognitive Load Theory and Zone of Proximal Development, the system ensures that recommended paths are cognitively aligned and pedagogically meaningful. Experiments conducted on the MOOCCubeX dataset using seven LLMs show that MALPP significantly outperforms baseline models in path quality, knowledge sequence consistency, and cognitive load alignment. Ablation studies further validate the effectiveness of the collaborative mechanism and theoretical constraints. This research contributes to the development of trustworthy, explainable AI in education and demonstrates a scalable approach to learner-centered adaptive instruction powered by LLMs. 

---
# Are We Evaluating the Edit Locality of LLM Model Editing Properly? 

**Authors**: Wei Liu, Haomei Xu, Hongkai Liu, Zhiying Deng, Ruixuan Li, Heng Huang, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2601.17343)  

**Abstract**: Model editing has recently emerged as a popular paradigm for efficiently updating knowledge in LLMs. A central desideratum of updating knowledge is to balance editing efficacy, i.e., the successful injection of target knowledge, and specificity (also known as edit locality), i.e., the preservation of existing non-target knowledge. However, we find that existing specificity evaluation protocols are inadequate for this purpose. We systematically elaborated on the three fundamental issues it faces. Beyond the conceptual issues, we further empirically demonstrate that existing specificity metrics are weakly correlated with the strength of specificity regularizers. We also find that current metrics lack sufficient sensitivity, rendering them ineffective at distinguishing the specificity performance of different methods. Finally, we propose a constructive evaluation protocol. Under this protocol, the conflict between open-ended LLMs and the assumption of determined answers is eliminated, query-independent fluency biases are avoided, and the evaluation strictness can be smoothly adjusted within a near-continuous space. Experiments across various LLMs, datasets, and editing methods show that metrics derived from the proposed protocol are more sensitive to changes in the strength of specificity regularizers and exhibit strong correlation with them, enabling more fine-grained discrimination of different methods' knowledge preservation capabilities. 

---
# The Relativity of AGI: Distributional Axioms, Fragility, and Undecidability 

**Authors**: Angshul Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2601.17335)  

**Abstract**: We study whether Artificial General Intelligence (AGI) admits a coherent theoretical definition that supports absolute claims of existence, robustness, or self-verification. We formalize AGI axiomatically as a distributional, resource-bounded semantic predicate, indexed by a task family, a task distribution, a performance functional, and explicit resource budgets. Under this framework, we derive four classes of results. First, we show that generality is inherently relational: there is no distribution-independent notion of AGI. Second, we prove non-invariance results demonstrating that arbitrarily small perturbations of the task distribution can invalidate AGI properties via cliff sets, precluding universal robustness. Third, we establish bounded transfer guarantees, ruling out unbounded generalization across task families under finite resources. Fourth, invoking Rice-style and Gödel--Tarski arguments, we prove that AGI is a nontrivial semantic property and therefore cannot be soundly and completely certified by any computable procedure, including procedures implemented by the agent itself. Consequently, recursive self-improvement schemes that rely on internal self-certification of AGI are ill-posed. Taken together, our results show that strong, distribution-independent claims of AGI are not false but undefined without explicit formal indexing, and that empirical progress in AI does not imply the attainability of self-certifying general intelligence. 

---
# TheoremForge: Scaling up Formal Data Synthesis with Low-Budget Agentic Workflow 

**Authors**: Yicheng Tao, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17332)  

**Abstract**: The high cost of agentic workflows in formal mathematics hinders large-scale data synthesis, exacerbating the scarcity of open-source corpora. To address this, we introduce \textbf{TheoremForge}, a cost-effective formal data synthesis pipeline that decomposes the formalization process into five sub-tasks, which are \textit{statement formalization}, \textit{proof generation}, \textit{premise selection}, \textit{proof correction} and \textit{proof sketching}. By implementing a \textit{Decoupled Extraction Strategy}, the workflow recovers valid training signals from globally failed trajectories, effectively utilizing wasted computation. Experiments on a 2,000-problem benchmark demonstrate that TheoremForge achieves a Verified Rate of 12.6\%, surpassing the 8.6\% baseline, at an average cost of only \textbf{\$0.481} per successful trajectory using Gemini-3-Flash. Crucially, our strategy increases data yield by \textbf{1.6$\times$} for proof generation compared to standard filtering. These results establish TheoremForge as a scalable framework for constructing a data flywheel to train future expert models. Our code is available \href{this https URL}{here}. 

---
# Phase Transition for Budgeted Multi-Agent Synergy 

**Authors**: Bang Liu, Linglong Kong, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2601.17311)  

**Abstract**: Multi-agent systems can improve reliability, yet under a fixed inference budget they often help, saturate, or even collapse. We develop a minimal and calibratable theory that predicts these regimes from three binding constraints of modern agent stacks: finite context windows, lossy inter-agent communication, and shared failures among similar agents. Each leaf agent is summarized by a compute-performance scaling exponent $\beta$; communication is captured by a message-length fidelity curve $\gamma(m)$; dependence is captured by an effective shared-error correlation $\rho$; and a context window $W$ imposes hard fan-in limits that make hierarchy necessary. For binary success/failure tasks with majority aggregation, we prove a sharp phase transition for deep $b$-ary trees with correlated inputs and lossy communication: a single scalar $\alpha_\rho$ (combining $\gamma(m)$, $\rho$, and fan-in $b$) determines whether weak signal is amplified to a nontrivial fixed point or washed out to chance. In the amplifying regime, we derive an organization exponent $s$ and show that budgeted synergy, i.e., outperforming the best single agent under the same total budget, occurs exactly when $s>\beta$, yielding closed-form compute allocation rules and explicit budget thresholds. We further characterize saturation via a mixing depth and provide a conservative clipped predictor that remains accurate across growth and saturation. A continuous-performance warm-up gives closed-form risks for star, chain, and tree organizations, making correlation- and communication-induced floors explicit and exposing the core design trade-offs in a smooth setting. Finally, we validate the predicted phase boundaries in controlled synthetic simulations and show how the same mechanisms explain the dominant bottlenecks reported in recent large-scale matched-budget studies of LLM agent-system scaling. 

---
# High-Fidelity Longitudinal Patient Simulation Using Real-World Data 

**Authors**: Yu Akagi, Tomohisa Seki, Hiromasa Ito, Toru Takiguchi, Kazuhiko Ohe, Yoshimasa Kawazoe  

**Link**: [PDF](https://arxiv.org/pdf/2601.17310)  

**Abstract**: Simulation is a powerful tool for exploring uncertainty. Its potential in clinical medicine is transformative and includes personalized treatment planning and virtual clinical trials. However, simulating patient trajectories is challenging because of complex biological and sociocultural influences. Here, we show that real-world clinical records can be leveraged to empirically model patient timelines. We developed a generative simulator model that takes a patient's history as input and synthesizes fine-grained, realistic future trajectories. The model was pretrained on more than 200 million clinical records. It produced high-fidelity future timelines, closely matching event occurrence rates, laboratory test results, and temporal dynamics in real patient future data. It also accurately estimated future event probabilities, with observed-to-expected ratios consistently near 1.0 across diverse outcomes and time horizons. Our results reveal the untapped value of real-world data in electronic health records and introduce a scalable framework for in silico modeling of clinical care. 

---
# Implementing Tensor Logic: Unifying Datalog and Neural Reasoning via Tensor Contraction 

**Authors**: Swapn Shah, Wlodek Zadrozny  

**Link**: [PDF](https://arxiv.org/pdf/2601.17188)  

**Abstract**: The unification of symbolic reasoning and neural networks remains a central challenge in artificial intelligence. Symbolic systems offer reliability and interpretability but lack scalability, while neural networks provide learning capabilities but sacrifice transparency. Tensor Logic, proposed by Domingos, suggests that logical rules and Einstein summation are mathematically equivalent, offering a principled path toward unification. This paper provides empirical validation of this framework through three experiments. First, we demonstrate the equivalence between recursive Datalog rules and iterative tensor contractions by computing the transitive closure of a biblical genealogy graph containing 1,972 individuals and 1,727 parent-child relationships, converging in 74 iterations to discover 33,945 ancestor relationships. Second, we implement reasoning in embedding space by training a neural network with learnable transformation matrices, demonstrating successful zero-shot compositional inference on held-out queries. Third, we validate the Tensor Logic superposition construction on FB15k-237, a large-scale knowledge graph with 14,541 entities and 237 relations. Using Domingos's relation matrix formulation $R_r = E^\top A_r E$, we achieve MRR of 0.3068 on standard link prediction and MRR of 0.3346 on a compositional reasoning benchmark where direct edges are removed during training, demonstrating that matrix composition enables multi-hop inference without direct training examples. 

---
# Interpreting Agentic Systems: Beyond Model Explanations to System-Level Accountability 

**Authors**: Judy Zhu, Dhari Gandhi, Himanshu Joshi, Ahmad Rezaie Mianroodi, Sedef Akinli Kocak, Dhanesh Ramachandran  

**Link**: [PDF](https://arxiv.org/pdf/2601.17168)  

**Abstract**: Agentic systems have transformed how Large Language Models (LLMs) can be leveraged to create autonomous systems with goal-directed behaviors, consisting of multi-step planning and the ability to interact with different environments. These systems differ fundamentally from traditional machine learning models, both in architecture and deployment, introducing unique AI safety challenges, including goal misalignment, compounding decision errors, and coordination risks among interacting agents, that necessitate embedding interpretability and explainability by design to ensure traceability and accountability across their autonomous behaviors. Current interpretability techniques, developed primarily for static models, show limitations when applied to agentic systems. The temporal dynamics, compounding decisions, and context-dependent behaviors of agentic systems demand new analytical approaches. This paper assesses the suitability and limitations of existing interpretability methods in the context of agentic systems, identifying gaps in their capacity to provide meaningful insight into agent decision-making. We propose future directions for developing interpretability techniques specifically designed for agentic systems, pinpointing where interpretability is required to embed oversight mechanisms across the agent lifecycle from goal formation, through environmental interaction, to outcome evaluation. These advances are essential to ensure the safe and accountable deployment of agentic AI systems. 

---
# Online parameter estimation for the Crazyflie quadcopter through an EM algorithm 

**Authors**: Yanhua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17009)  

**Abstract**: Drones are becoming more and more popular nowadays. They are small in size, low in cost, and reliable in operation. They contain a variety of sensors and can perform a variety of flight tasks, reaching places that are difficult or inaccessible for humans. Earthquakes damage a lot of infrastructure, making it impossible for rescuers to reach some areas. But drones can help. Many amateur and professional photographers like to use drones for aerial photography. Drones play a non-negligible role in agriculture and transportation too. Drones can be used to spray pesticides, and they can also transport supplies. A quadcopter is a four-rotor drone and has been studied in this paper. In this paper, random noise is added to the quadcopter system and its effects on the drone system are studied. An extended Kalman filter has been used to estimate the state based on noisy observations from the sensor. Based on a SDE system, a linear quadratic Gaussian controller has been implemented. The expectation maximization algorithm has been applied for parameter estimation of the quadcopter. The results of offline parameter estimation and online parameter estimation are presented. The results show that the online parameter estimation has a slightly larger range of convergence values than the offline parameter estimation. 

---
# ctELM: Decoding and Manipulating Embeddings of Clinical Trials with Embedding Language Models 

**Authors**: Brian Ondov, Chia-Hsuan Chang, Yujia Zhou, Mauro Giuffrè, Hua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18796)  

**Abstract**: Text embeddings have become an essential part of a variety of language applications. However, methods for interpreting, exploring and reversing embedding spaces are limited, reducing transparency and precluding potentially valuable generative use cases. In this work, we align Large Language Models to embeddings of clinical trials using the recently reported Embedding Language Model (ELM) method. We develop an open-source, domain-agnostic ELM architecture and training framework, design training tasks for clinical trials, and introduce an expert-validated synthetic dataset. We then train a series of ELMs exploring the impact of tasks and training regimes. Our final model, ctELM, can accurately describe and compare unseen clinical trials from embeddings alone and produce plausible clinical trials from novel vectors. We further show that generated trial abstracts are responsive to moving embeddings along concept vectors for age and sex of study subjects. Our public ELM implementation and experimental results will aid the alignment of Large Language Models to embedding spaces in the biomedical domain and beyond. 

---
# Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes 

**Authors**: Amrith Setlur, Zijian Wang, Andrew Cohen, Paria Rashidinejad, Sang Michael Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.18795)  

**Abstract**: Typical reinforcement learning (RL) methods for LLM reasoning waste compute on hard problems, where correct on-policy traces are rare, policy gradients vanish, and learning stalls. To bootstrap more efficient RL, we consider reusing old sampling FLOPs (from prior inference or RL training) in the form of off-policy traces. Standard off-policy methods supervise against off-policy data, causing instabilities during RL optimization. We introduce PrefixRL, where we condition on the prefix of successful off-policy traces and run on-policy RL to complete them, side-stepping off-policy instabilities. PrefixRL boosts the learning signal on hard problems by modulating the difficulty of the problem through the off-policy prefix length. We prove that the PrefixRL objective is not only consistent with the standard RL objective but also more sample efficient. Empirically, we discover back-generalization: training only on prefixed problems generalizes to out-of-distribution unprefixed performance, with learned strategies often differing from those in the prefix. In our experiments, we source the off-policy traces by rejection sampling with the base model, creating a self-improvement loop. On hard reasoning problems, PrefixRL reaches the same training reward 2x faster than the strongest baseline (SFT on off-policy data then RL), even after accounting for the compute spent on the initial rejection sampling, and increases the final reward by 3x. The gains transfer to held-out benchmarks, and PrefixRL is still effective when off-policy traces are derived from a different model family, validating its flexibility in practical settings. 

---
# Subword-Based Comparative Linguistics across 242 Languages Using Wikipedia Glottosets 

**Authors**: Iaroslav Chelombitko, Mika Hämäläinen, Aleksey Komissarov  

**Link**: [PDF](https://arxiv.org/pdf/2601.18791)  

**Abstract**: We present a large-scale comparative study of 242 Latin and Cyrillic-script languages using subword-based methodologies. By constructing 'glottosets' from Wikipedia lexicons, we introduce a framework for simultaneous cross-linguistic comparison via Byte-Pair Encoding (BPE). Our approach utilizes rank-based subword vectors to analyze vocabulary overlap, lexical divergence, and language similarity at scale. Evaluations demonstrate that BPE segmentation aligns with morpheme boundaries 95% better than random baseline across 15 languages (F1 = 0.34 vs 0.15). BPE vocabulary similarity correlates significantly with genetic language relatedness (Mantel r = 0.329, p < 0.001), with Romance languages forming the tightest cluster (mean distance 0.51) and cross-family pairs showing clear separation (0.82). Analysis of 26,939 cross-linguistic homographs reveals that 48.7% receive different segmentations across related languages, with variation correlating to phylogenetic distance. Our results provide quantitative macro-linguistic insights into lexical patterns across typologically diverse languages within a unified analytical framework. 

---
# Design Techniques for LLM-Powered Interactive Storytelling: A Case Study of the Dramamancer System 

**Authors**: Tiffany Wang, Yuqian Sun, Yi Wang, Melissa Roemmele, John Joon Young Chung, Max Kreminski  

**Link**: [PDF](https://arxiv.org/pdf/2601.18785)  

**Abstract**: The rise of Large Language Models (LLMs) has enabled a new paradigm for bridging authorial intent and player agency in interactive narrative. We consider this paradigm through the example of Dramamancer, a system that uses an LLM to transform author-created story schemas into player-driven playthroughs. This extended abstract outlines some design techniques and evaluation considerations associated with this system. 

---
# Multi-Objective Reinforcement Learning for Efficient Tactical Decision Making for Trucks in Highway Traffic 

**Authors**: Deepthi Pathare, Leo Laine, Morteza Haghir Chehreghani  

**Link**: [PDF](https://arxiv.org/pdf/2601.18783)  

**Abstract**: Balancing safety, efficiency, and operational costs in highway driving poses a challenging decision-making problem for heavy-duty vehicles. A central difficulty is that conventional scalar reward formulations, obtained by aggregating these competing objectives, often obscure the structure of their trade-offs. We present a Proximal Policy Optimization based multi-objective reinforcement learning framework that learns a continuous set of policies explicitly representing these trade-offs and evaluates it on a scalable simulation platform for tactical decision making in trucks. The proposed approach learns a continuous set of Pareto-optimal policies that capture the trade-offs among three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency and time efficiency, quantified using energy cost and driver cost, respectively. The resulting Pareto frontier is smooth and interpretable, enabling flexibility in choosing driving behavior along different conflicting objectives. This framework allows seamless transitions between different driving policies without retraining, yielding a robust and adaptive decision-making strategy for autonomous trucking applications. 

---
# POPE: Learning to Reason on Hard Problems via Privileged On-Policy Exploration 

**Authors**: Yuxiao Qu, Amrith Setlur, Virginia Smith, Ruslan Salakhutdinov, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.18779)  

**Abstract**: Reinforcement learning (RL) has improved the reasoning abilities of large language models (LLMs), yet state-of-the-art methods still fail to learn on many training problems. On hard problems, on-policy RL rarely explores even a single correct rollout, yielding zero reward and no learning signal for driving improvement. We find that natural solutions to remedy this exploration problem from classical RL, such as entropy bonuses, more permissive clipping of the importance ratio, or direct optimization of pass@k objectives, do not resolve this issue and often destabilize optimization without improving solvability. A natural alternative is to leverage transfer from easier problems. However, we show that mixing easy and hard problems during RL training is counterproductive due to ray interference, where optimization focuses on already-solvable problems in a way that actively inhibits progress on harder ones. To address this challenge, we introduce Privileged On-Policy Exploration (POPE), an approach that leverages human- or other oracle solutions as privileged information to guide exploration on hard problems, unlike methods that use oracle solutions as training targets (e.g., off-policy RL methods or warmstarting from SFT). POPE augments hard problems with prefixes of oracle solutions, enabling RL to obtain non-zero rewards during guided rollouts. Crucially, the resulting behaviors transfer back to the original, unguided problems through a synergy between instruction-following and reasoning. Empirically, POPE expands the set of solvable problems and substantially improves performance on challenging reasoning benchmarks. 

---
# PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation 

**Authors**: Abhishek Divekar, Anirban Majumder  

**Link**: [PDF](https://arxiv.org/pdf/2601.18777)  

**Abstract**: Evaluating the quality of search, ranking and RAG systems traditionally requires a significant number of human relevance annotations. In recent times, several deployed systems have explored the usage of Large Language Models (LLMs) as automated judges for this task while their inherent biases prevent direct use for metric estimation. We present a statistical framework extending Prediction-Powered Inference (PPI) that combines minimal human annotations with LLM judgments to produce reliable estimates of metrics which require sub-instance annotations. Our method requires as few as 100 human-annotated queries and 10,000 unlabeled examples, reducing annotation requirements significantly compared to traditional approaches. We formulate our proposed framework (PRECISE) for inference of relevance uplift for an LLM-based query reformulation application, extending PPI to sub-instance annotations at the query-document level. By reformulating the metric-integration space, we reduced the computational complexity from O(2^|C|) to O(2^K), where |C| represents corpus size (in order of millions). Detailed experiments across prominent retrieval datasets demonstrate that our method reduces the variance of estimates for the business-critical Precision@K metric, while effectively correcting for LLM bias in low-resource settings. 

---
# Dep-Search: Learning Dependency-Aware Reasoning Traces with Persistent Memory 

**Authors**: Yanming Liu, Xinyue Peng, Zixuan Yan, Yanxin Shen, Wenjie Xu, Yuefeng Huang, Xinyi Wang, Jiannan Cao, Jianwei Yin, Xuhong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18771)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, particularly when augmented with search mechanisms that enable systematic exploration of external knowledge bases. The field has evolved from traditional retrieval-augmented generation (RAG) frameworks to more sophisticated search-based frameworks that orchestrate multi-step reasoning through explicit search strategies. However, existing search frameworks still rely heavily on implicit natural language reasoning to determine search strategies and how to leverage retrieved information across reasoning steps. This reliance on implicit reasoning creates fundamental challenges for managing dependencies between sub-questions, efficiently reusing previously retrieved knowledge, and learning optimal search strategies through reinforcement learning. To address these limitations, we propose Dep-Search, a dependency-aware search framework that advances beyond existing search frameworks by integrating structured reasoning, retrieval, and persistent memory through GRPO. Dep-Search introduces explicit control mechanisms that enable the model to decompose questions with dependency relationships, retrieve information when needed, access previously stored knowledge from memory, and summarize long reasoning contexts into reusable memory entries. Through extensive experiments on seven diverse question answering datasets, we demonstrate that Dep-Search significantly enhances LLMs' ability to tackle complex multi-hop reasoning tasks, achieving substantial improvements over strong baselines across different model scales. 

---
# $α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks 

**Authors**: Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2601.18754)  

**Abstract**: Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings.
We introduce $\alpha^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $\alpha^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $\alpha^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage).
We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $\alpha^{3}$-SecBench on GitHub: this https URL 

---
# HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs 

**Authors**: Xinyue Zeng, Junhong Lin, Yujun Yan, Feng Guo, Liang Shi, Jun Wu, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.18753)  

**Abstract**: The reliability of Large Language Models (LLMs) in high-stakes domains such as healthcare, law, and scientific discovery is often compromised by hallucinations. These failures typically stem from two sources: data-driven hallucinations and reasoning-driven hallucinations. However, existing detection methods usually address only one source and rely on task-specific heuristics, limiting their generalization to complex scenarios. To overcome these limitations, we introduce the Hallucination Risk Bound, a unified theoretical framework that formally decomposes hallucination risk into data-driven and reasoning-driven components, linked respectively to training-time mismatches and inference-time instabilities. This provides a principled foundation for analyzing how hallucinations emerge and evolve. Building on this foundation, we introduce HalluGuard, an NTK-based score that leverages the induced geometry and captured representations of the NTK to jointly identify data-driven and reasoning-driven hallucinations. We evaluate HalluGuard on 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, consistently achieving state-of-the-art performance in detecting diverse forms of LLM hallucinations. 

---
# Trust, Don't Trust, or Flip: Robust Preference-Based Reinforcement Learning with Multi-Expert Feedback 

**Authors**: Seyed Amir Hosseini, Maryam Abdolali, Amirhosein Tavakkoli, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  

**Link**: [PDF](https://arxiv.org/pdf/2601.18751)  

**Abstract**: Preference-based reinforcement learning (PBRL) offers a promising alternative to explicit reward engineering by learning from pairwise trajectory comparisons. However, real-world preference data often comes from heterogeneous annotators with varying reliability; some accurate, some noisy, and some systematically adversarial. Existing PBRL methods either treat all feedback equally or attempt to filter out unreliable sources, but both approaches fail when faced with adversarial annotators who systematically provide incorrect preferences. We introduce TriTrust-PBRL (TTP), a unified framework that jointly learns a shared reward model and expert-specific trust parameters from multi-expert preference feedback. The key insight is that trust parameters naturally evolve during gradient-based optimization to be positive (trust), near zero (ignore), or negative (flip), enabling the model to automatically invert adversarial preferences and recover useful signal rather than merely discarding corrupted feedback. We provide theoretical analysis establishing identifiability guarantees and detailed gradient analysis that explains how expert separation emerges naturally during training without explicit supervision. Empirically, we evaluate TTP on four diverse domains spanning manipulation tasks (MetaWorld) and locomotion (DM Control) under various corruption scenarios. TTP achieves state-of-the-art robustness, maintaining near-oracle performance under adversarial corruption while standard PBRL methods fail catastrophically. Notably, TTP outperforms existing baselines by successfully learning from mixed expert pools containing both reliable and adversarial annotators, all while requiring no expert features beyond identification indices and integrating seamlessly with existing PBRL pipelines. 

---
# Capturing P: On the Expressive Power and Efficient Evaluation of Boolean Retrieval 

**Authors**: Amir Aavani  

**Link**: [PDF](https://arxiv.org/pdf/2601.18747)  

**Abstract**: Modern information retrieval is transitioning from simple document filtering to complex, neuro-symbolic reasoning workflows. However, current retrieval architectures face a fundamental efficiency dilemma when handling the rigorous logical and arithmetic constraints required by this new paradigm. Standard iterator-based engines (Document-at-a-Time) do not natively support complex, nested logic graphs; forcing them to execute such queries typically results in intractable runtime performance. Conversely, naive recursive approaches (Term-at-a-Time), while capable of supporting these structures, suffer from prohibitive memory consumption when enforcing broad logical exclusions.
In this paper, we propose that a retrieval engine must be capable of ``Capturing $\mathbf{P}$'' -- evaluating any polynomial-time property directly over its index in a computationally efficient manner. We define a formal Retrieval Language ($\mathcal{L}_R$) based on Directed Acyclic Graphs (DAGs) and prove it precisely captures the complexity class $\mathbf{P}$. We introduce \texttt{ComputePN}, a novel evaluation algorithm that makes $\mathcal{L}_R$ tractable. By combining native DAG traversal with a memory-efficient ``Positive-Negative'' response mechanism, \texttt{ComputePN} ensures the efficient evaluation of any query in $\mathcal{L}_R$. This work establishes the theoretical foundation for turning the search index into a general-purpose computational engine. 

---
# SeNeDiF-OOD: Semantic Nested Dichotomy Fusion for Out-of-Distribution Detection Methodology in Open-World Classification. A Case Study on Monument Style Classification 

**Authors**: Ignacio Antequera-Sánchez, Juan Luis Suárez-Díaz, Rosana Montes, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2601.18739)  

**Abstract**: Out-of-distribution (OOD) detection is a fundamental requirement for the reliable deployment of artificial intelligence applications in open-world environments. However, addressing the heterogeneous nature of OOD data, ranging from low-level corruption to semantic shifts, remains a complex challenge that single-stage detectors often fail to resolve. To address this issue, we propose SeNeDiF-OOD, a novel methodology based on Semantic Nested Dichotomy Fusion. This framework decomposes the detection task into a hierarchical structure of binary fusion nodes, where each layer is designed to integrate decision boundaries aligned with specific levels of semantic abstraction. To validate the proposed framework, we present a comprehensive case study using MonuMAI, a real-world architectural style recognition system exposed to an open environment. This application faces a diverse range of inputs, including non-monument images, unknown architectural styles, and adversarial attacks, making it an ideal testbed for our proposal. Through extensive experimental evaluation in this domain, results demonstrate that our hierarchical fusion methodology significantly outperforms traditional baselines, effectively filtering these diverse OOD categories while preserving in-distribution performance. 

---
# Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge 

**Authors**: Li Kang, Heng Zhou, Xiufeng Song, Rui Li, Bruno N.Y. Chen, Ziye Wang, Ximeng Meng, Stone Tao, Yiran Qin, Xiaohong Liu, Ruimao Zhang, Lei Bai, Yilun Du, Hao Su, Philip Torr, Zhenfei Yin, Ruihao Gong, Yejun Zeng, Fengjun Zhong, Shenghao Jin, Jinyang Guo, Xianglong Liu, Xiaojun Jia, Tianqi Shan, Wenqi Ren, Simeng Qin, Jialing Yang, Xiaoyu Ma, Tianxing Chen, Zixuan Li, Zijian Cai, Yan Qin, Yusen Qin, Qiangyu Chen, Kaixuan Wang, Zhaoming Han, Yao Mu, Ping Luo, Yuanqi Yao, Haoming Song, Jan-Nico Zaech, Fabien Despinoy, Danda Pani Paudel, Luc Van Gool  

**Link**: [PDF](https://arxiv.org/pdf/2601.18733)  

**Abstract**: Recent advancements in multimodal large language models and vision-languageaction models have significantly driven progress in Embodied AI. As the field transitions toward more complex task scenarios, multi-agent system frameworks are becoming essential for achieving scalable, efficient, and collaborative solutions. This shift is fueled by three primary factors: increasing agent capabilities, enhancing system efficiency through task delegation, and enabling advanced human-agent interactions. To address the challenges posed by multi-agent collaboration, we propose the Multi-Agent Robotic System (MARS) Challenge, held at the NeurIPS 2025 Workshop on SpaVLE. The competition focuses on two critical areas: planning and control, where participants explore multi-agent embodied planning using vision-language models (VLMs) to coordinate tasks and policy execution to perform robotic manipulation in dynamic environments. By evaluating solutions submitted by participants, the challenge provides valuable insights into the design and coordination of embodied multi-agent systems, contributing to the future development of advanced collaborative AI systems. 

---
# Optimal Use of Preferences in Artificial Intelligence Algorithms 

**Authors**: Joshua S. Gans  

**Link**: [PDF](https://arxiv.org/pdf/2601.18732)  

**Abstract**: Machine learning systems embed preferences either in training losses or through post-processing of calibrated predictions. Applying information design methods from Strack and Yang (2024), this paper provides decision problem agnostic conditions under which separation training preference free and applying preferences ex post is optimal. Unlike prior work that requires specifying downstream objectives, the welfare results here apply uniformly across decision problems. The key primitive is a diminishing-value-of-information condition: relative to a fixed (normalised) preference-free loss, preference embedding makes informativeness less valuable at the margin, inducing a mean-preserving contraction of learned posteriors. Because the value of information is convex in beliefs, preference-free training weakly dominates for any expected utility decision problem. This provides theoretical foundations for modular AI pipelines that learn calibrated probabilities and implement asymmetric costs through downstream decision rules. However, separation requires users to implement optimal decision rules. When cognitive constraints bind, as documented in human AI decision-making, preference embedding can dominate by automating threshold computation. These results provide design guidance: preserve optionality through post-processing when objectives may shift; embed preferences when decision-stage frictions dominate. 

---
# One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment 

**Authors**: Hongru Cai, Yongqi Li, Tiezheng Yu, Fengbin Zhu, Wenjie Wang, Fuli Feng, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.18731)  

**Abstract**: Alignment of Large Language Models (LLMs) aims to align outputs with human preferences, and personalized alignment further adapts models to individual users. This relies on personalized reward models that capture user-specific preferences and automatically provide individualized feedback. However, developing these models faces two critical challenges: the scarcity of feedback from individual users and the need for efficient adaptation to unseen users. We argue that addressing these constraints requires a paradigm shift from fitting data to learn user preferences to learn the process of preference adaptation. To realize this, we propose Meta Reward Modeling (MRM), which reformulates personalized reward modeling as a meta-learning problem. Specifically, we represent each user's reward model as a weighted combination of base reward functions, and optimize the initialization of these weights using a Model-Agnostic Meta-Learning (MAML)-style framework to support fast adaptation under limited feedback. To ensure robustness, we introduce the Robust Personalization Objective (RPO), which places greater emphasis on hard-to-learn users during meta optimization. Extensive experiments on personalized preference datasets validate that MRM enhances few-shot personalization, improves user robustness, and consistently outperforms baselines. 

---
# HalluCitation Matters: Revealing the Impact of Hallucinated References with 300 Hallucinated Papers in ACL Conferences 

**Authors**: Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2601.18724)  

**Abstract**: Recently, we have often observed hallucinated citations or references that do not correspond to any existing work in papers under review, preprints, or published papers. Such hallucinated citations pose a serious concern to scientific reliability. When they appear in accepted papers, they may also negatively affect the credibility of conferences. In this study, we refer to hallucinated citations as "HalluCitation" and systematically investigate their prevalence and impact. We analyze all papers published at ACL, NAACL, and EMNLP in 2024 and 2025, including main conference, Findings, and workshop papers. Our analysis reveals that nearly 300 papers contain at least one HalluCitation, most of which were published in 2025. Notably, half of these papers were identified at EMNLP 2025, the most recent conference, indicating that this issue is rapidly increasing. Moreover, more than 100 such papers were accepted as main conference and Findings papers at EMNLP 2025, affecting the credibility. 

---
# Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning 

**Authors**: Judith Vilella-Cantos, Mauro Martini, Marcello Chiaberge, Mónica Ballesta, David Valiente  

**Link**: [PDF](https://arxiv.org/pdf/2601.18714)  

**Abstract**: Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data. The code is publicly available for reproduction. 

---
# Point transformer for protein structural heterogeneity analysis using CryoEM 

**Authors**: Muyuan Chen, Muchen Li, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18713)  

**Abstract**: Structural dynamics of macromolecules is critical to their structural-function relationship. Cryogenic electron microscopy (CryoEM) provides snapshots of vitrified protein at different compositional and conformational states, and the structural heterogeneity of proteins can be characterized through computational analysis of the images. For protein systems with multiple degrees of freedom, it is still challenging to disentangle and interpret the different modes of dynamics. Here, by implementing Point Transformer, a self-attention network designed for point cloud analysis, we are able to improve the performance of heterogeneity analysis on CryoEM data, and characterize the dynamics of highly complex protein systems in a more human-interpretable way. 

---
# SMART: Scalable Mesh-free Aerodynamic Simulations from Raw Geometries using a Transformer-based Surrogate Model 

**Authors**: Jan Hagnberger, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2601.18707)  

**Abstract**: Machine learning-based surrogate models have emerged as more efficient alternatives to numerical solvers for physical simulations over complex geometries, such as car bodies. Many existing models incorporate the simulation mesh as an additional input, thereby reducing prediction errors. However, generating a simulation mesh for new geometries is computationally costly. In contrast, mesh-free methods, which do not rely on the simulation mesh, typically incur higher errors. Motivated by these considerations, we introduce SMART, a neural surrogate model that predicts physical quantities at arbitrary query locations using only a point-cloud representation of the geometry, without requiring access to the simulation mesh. The geometry and simulation parameters are encoded into a shared latent space that captures both structural and parametric characteristics of the physical field. A physics decoder then attends to the encoder's intermediate latent representations to map spatial queries to physical quantities. Through this cross-layer interaction, the model jointly updates latent geometric features and the evolving physical field. Extensive experiments show that SMART is competitive with and often outperforms existing methods that rely on the simulation mesh as input, demonstrating its capabilities for industry-level simulations. 

---
# From Fuzzy to Exact: The Halo Architecture for Infinite-Depth Reasoning via Rational Arithmetic 

**Authors**: Hansheng Ren  

**Link**: [PDF](https://arxiv.org/pdf/2601.18702)  

**Abstract**: Current paradigms in Deep Learning prioritize computational throughput over numerical precision, relying on the assumption that intelligence emerges from statistical correlation at scale. In this paper, we challenge this orthodoxy. We propose the Exactness Hypothesis: that General Intelligence (AGI), specifically high-order causal inference, requires a computational substrate capable of Arbitrary Precision Arithmetic. We argue that the "hallucinations" and logical incoherence seen in current Large Language Models (LLMs) are artifacts of IEEE 754 floating-point approximation errors accumulating over deep compositional functions. To mitigate this, we introduce the Halo Architecture, a paradigm shift to Rational Arithmetic ($\mathbb{Q}$) supported by a novel Exact Inference Unit (EIU). Empirical validation on the Huginn-0125 prototype demonstrates that while 600B-parameter scale BF16 baselines collapse in chaotic systems, Halo maintains zero numerical divergence indefinitely. This work establishes exact arithmetic as a prerequisite for reducing logical uncertainty in System 2 AGI. 

---
# Neural Multi-Speaker Voice Cloning for Nepali in Low-Resource Settings 

**Authors**: Aayush M. Shrestha, Aditya Bajracharya, Projan Shakya, Dinesh B. Kshatri  

**Link**: [PDF](https://arxiv.org/pdf/2601.18694)  

**Abstract**: This research presents a few-shot voice cloning system for Nepali speakers, designed to synthesize speech in a specific speaker's voice from Devanagari text using minimal data. Voice cloning in Nepali remains largely unexplored due to its low-resource nature. To address this, we constructed separate datasets: untranscribed audio for training a speaker encoder and paired text-audio data for training a Tacotron2-based synthesizer. The speaker encoder, optimized with Generative End2End loss, generates embeddings that capture the speaker's vocal identity, validated through Uniform Manifold Approximation and Projection (UMAP) for dimension reduction visualizations. These embeddings are fused with Tacotron2's text embeddings to produce mel-spectrograms, which are then converted into audio using a WaveRNN vocoder. Audio data were collected from various sources, including self-recordings, and underwent thorough preprocessing for quality and alignment. Training was performed using mel and gate loss functions under multiple hyperparameter settings. The system effectively clones speaker characteristics even for unseen voices, demonstrating the feasibility of few-shot voice cloning for the Nepali language and establishing a foundation for personalized speech synthesis in low-resource scenarios. 

---
# ART for Diffusion Sampling: A Reinforcement Learning Approach to Timestep Schedule 

**Authors**: Yilie Huang, Wenpin Tang, Xunyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.18681)  

**Abstract**: We consider time discretization for score-based diffusion models to generate samples from a learned reverse-time dynamic on a finite grid. Uniform and hand-crafted grids can be suboptimal given a budget on the number of time steps. We introduce Adaptive Reparameterized Time (ART) that controls the clock speed of a reparameterized time variable, leading to a time change and uneven timesteps along the sampling trajectory while preserving the terminal time. The objective is to minimize the aggregate error arising from the discretized Euler scheme. We derive a randomized control companion, ART-RL, and formulate time change as a continuous-time reinforcement learning (RL) problem with Gaussian policies. We then prove that solving ART-RL recovers the optimal ART schedule, which in turn enables practical actor--critic updates to learn the latter in a data-driven way. Empirically, based on the official EDM pipeline, ART-RL improves Fréchet Inception Distance on CIFAR-10 over a wide range of budgets and transfers to AFHQv2, FFHQ, and ImageNet without the need of retraining. 

---
# Learning temporal embeddings from electronic health records of chronic kidney disease patients 

**Authors**: Aditya Kumar, Mario A. Cypko, Oliver Amft  

**Link**: [PDF](https://arxiv.org/pdf/2601.18675)  

**Abstract**: We investigate whether temporal embedding models trained on longitudinal electronic health records can learn clinically meaningful representations without compromising predictive performance, and how architectural choices affect embedding quality. Model-guided medicine requires representations that capture disease dynamics while remaining transparent and task agnostic, whereas most clinical prediction models are optimised for a single task. Representation learning facilitates learning embeddings that generalise across downstream tasks, and recurrent architectures are well-suited for modelling temporal structure in observational clinical data. Using the MIMIC-IV dataset, we study patients with chronic kidney disease (CKD) and compare three recurrent architectures: a vanilla LSTM, an attention-augmented LSTM, and a time-aware LSTM (T-LSTM). All models are trained both as embedding models and as direct end-to-end predictors. Embedding quality is evaluated via CKD stage clustering and in-ICU mortality prediction. The T-LSTM produces more structured embeddings, achieving a lower Davies-Bouldin Index (DBI = 9.91) and higher CKD stage classification accuracy (0.74) than the vanilla LSTM (DBI = 15.85, accuracy = 0.63) and attention-augmented LSTM (DBI = 20.72, accuracy = 0.67). For in-ICU mortality prediction, embedding models consistently outperform end-to-end predictors, improving accuracy from 0.72-0.75 to 0.82-0.83, which indicates that learning embeddings as an intermediate step is more effective than direct end-to-end learning. 

---
# FaLW: A Forgetting-aware Loss Reweighting for Long-tailed Unlearning 

**Authors**: Liheng Yu, Zhe Zhao, Yuxuan Wang, Pengkun Wang, Binwu Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18650)  

**Abstract**: Machine unlearning, which aims to efficiently remove the influence of specific data from trained models, is crucial for upholding data privacy regulations like the ``right to be forgotten". However, existing research predominantly evaluates unlearning methods on relatively balanced forget sets. This overlooks a common real-world scenario where data to be forgotten, such as a user's activity records, follows a long-tailed distribution. Our work is the first to investigate this critical research gap. We find that in such long-tailed settings, existing methods suffer from two key issues: \textit{Heterogeneous Unlearning Deviation} and \textit{Skewed Unlearning Deviation}. To address these challenges, we propose FaLW, a plug-and-play, instance-wise dynamic loss reweighting method. FaLW innovatively assesses the unlearning state of each sample by comparing its predictive probability to the distribution of unseen data from the same class. Based on this, it uses a forgetting-aware reweighting scheme, modulated by a balancing factor, to adaptively adjust the unlearning intensity for each sample. Extensive experiments demonstrate that FaLW achieves superior performance. Code is available at \textbf{Supplementary Material}. 

---
# Unheard in the Digital Age: Rethinking AI Bias and Speech Diversity 

**Authors**: Onyedikachi Hope Amaechi-Okorie, Branislav Radeljic  

**Link**: [PDF](https://arxiv.org/pdf/2601.18641)  

**Abstract**: Speech remains one of the most visible yet overlooked vectors of inclusion and exclusion in contemporary society. While fluency is often equated with credibility and competence, individuals with atypical speech patterns are routinely marginalized. Given the current state of the debate, this article focuses on the structural biases that shape perceptions of atypical speech and are now being encoded into artificial intelligence. Automated speech recognition (ASR) systems and voice interfaces, trained predominantly on standardized speech, routinely fail to recognize or respond to diverse voices, compounding digital exclusion. As AI technologies increasingly mediate access to opportunity, the study calls for inclusive technological design, anti-bias training to minimize the impact of discriminatory algorithmic decisions, and enforceable policy reform that explicitly recognize speech diversity as a matter of equity, not merely accessibility. Drawing on interdisciplinary research, the article advocates for a cultural and institutional shift in how we value voice, urging co-created solutions that elevate the rights, representation, and realities of atypical speakers in the digital age. Ultimately, the article reframes speech inclusion as a matter of equity (not accommodation) and advocates for co-created AI systems that reflect the full spectrum of human voices. 

---
# Rank-1 Approximation of Inverse Fisher for Natural Policy Gradients in Deep Reinforcement Learning 

**Authors**: Yingxiao Huo, Satya Prakash Dash, Radu Stoican, Samuel Kaski, Mingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.18626)  

**Abstract**: Natural gradients have long been studied in deep reinforcement learning due to their fast convergence properties and covariant weight updates. However, computing natural gradients requires inversion of the Fisher Information Matrix (FIM) at each iteration, which is computationally prohibitive in nature. In this paper, we present an efficient and scalable natural policy optimization technique that leverages a rank-1 approximation to full inverse-FIM. We theoretically show that under certain conditions, a rank-1 approximation to inverse-FIM converges faster than policy gradients and, under some conditions, enjoys the same sample complexity as stochastic policy gradient methods. We benchmark our method on a diverse set of environments and show that it achieves superior performance to standard actor-critic and trust-region baselines. 

---
# Learning long term climate-resilient transport adaptation pathways under direct and indirect flood impacts using reinforcement learning 

**Authors**: Miguel Costa, Arthur Vandervoort, Carolin Schmidt, Morten W. Petersen, Martin Drews, Karyn Morrissey, Francisco C. Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2601.18586)  

**Abstract**: Climate change is expected to intensify rainfall and other hazards, increasing disruptions in urban transportation systems. Designing effective adaptation strategies is challenging due to the long-term, sequential nature of infrastructure investments, deep uncertainty, and complex cross-sector interactions. We propose a generic decision-support framework that couples an integrated assessment model (IAM) with reinforcement learning (RL) to learn adaptive, multi-decade investment pathways under uncertainty. The framework combines long-term climate projections (e.g., IPCC scenario pathways) with models that map projected extreme-weather drivers (e.g. rain) into hazard likelihoods (e.g. flooding), propagate hazards into urban infrastructure impacts (e.g. transport disruption), and value direct and indirect consequences for service performance and societal costs. Embedded in a reinforcement-learning loop, it learns adaptive climate adaptation policies that trade off investment and maintenance expenditures against avoided impacts. In collaboration with Copenhagen Municipality, we demonstrate the approach on pluvial flooding in the inner city for the horizon of 2024 to 2100. The learned strategies yield coordinated spatial-temporal pathways and improved robustness relative to conventional optimization baselines, namely inaction and random action, illustrating the framework's transferability to other hazards and cities. 

---
# FastInsight: Fast and Insightful Retrieval via Fusion Operators for Graph RAG 

**Authors**: Seonho An, Chaejeong Hyun, Min-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.18579)  

**Abstract**: Existing Graph RAG methods aiming for insightful retrieval on corpus graphs typically rely on time-intensive processes that interleave Large Language Model (LLM) reasoning. To enable time-efficient insightful retrieval, we propose FastInsight. We first introduce a graph retrieval taxonomy that categorizes existing methods into three fundamental operations: vector search, graph search, and model-based search. Through this taxonomy, we identify two critical limitations in current approaches: the topology-blindness of model-based search and the semantics-blindness of graph search. FastInsight overcomes these limitations by interleaving two novel fusion operators: the Graph-based Reranker (GRanker), which functions as a graph model-based search, and Semantic-Topological eXpansion (STeX), which operates as a vector-graph search. Extensive experiments on broad retrieval and generation datasets demonstrate that FastInsight significantly improves both retrieval accuracy and generation quality compared to state-of-the-art baselines, achieving a substantial Pareto improvement in the trade-off between effectiveness and efficiency. 

---
# Attention-Based Neural-Augmented Kalman Filter for Legged Robot State Estimation 

**Authors**: Seokju Lee, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.18569)  

**Abstract**: In this letter, we propose an Attention-Based Neural-Augmented Kalman Filter (AttenNKF) for state estimation in legged robots. Foot slip is a major source of estimation error: when slip occurs, kinematic measurements violate the no-slip assumption and inject bias during the update step. Our objective is to estimate this slip-induced error and compensate for it. To this end, we augment an Invariant Extended Kalman Filter (InEKF) with a neural compensator that uses an attention mechanism to infer error conditioned on foot-slip severity and then applies this estimate as a post-update compensation to the InEKF state (i.e., after the filter update). The compensator is trained in a latent space, which aims to reduce sensitivity to raw input scales and encourages structured slip-conditioned compensations, while preserving the InEKF recursion. Experiments demonstrate improved performance compared to existing legged-robot state estimators, particularly under slip-prone conditions. 

---
# SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction 

**Authors**: Linyong Gan, Zimo Li, Wenxin Xu, Xingjian Li, Jianhua Z. Huang, Enmei Tu, Shuhang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.18537)  

**Abstract**: Accurate long-horizon vessel trajectory prediction remains challenging due to compounded uncertainty from complex navigation behaviors and environmental factors. Existing methods often struggle to maintain global directional consistency, leading to drifting or implausible trajectories when extrapolated over long time horizons. To address this issue, we propose a semantic-key-point-conditioned trajectory modeling framework, in which future trajectories are predicted by conditioning on a high-level Next Key Point (NKP) that captures navigational intent. This formulation decomposes long-horizon prediction into global semantic decision-making and local motion modeling, effectively restricting the support of future trajectories to semantically feasible subsets. To efficiently estimate the NKP prior from historical observations, we adopt a pretrain-finetune strategy. Extensive experiments on real-world AIS data demonstrate that the proposed method consistently outperforms state-of-the-art approaches, particularly for long travel durations, directional accuracy, and fine-grained trajectory prediction. 

---
# Scalable Transit Delay Prediction at City Scale: A Systematic Approach with Multi-Resolution Feature Engineering and Deep Learning 

**Authors**: Emna Boudabbous, Mohamed Karaa, Lokman Sboui, Julio Montecinos, Omar Alam  

**Link**: [PDF](https://arxiv.org/pdf/2601.18521)  

**Abstract**: Urban bus transit agencies need reliable, network-wide delay predictions to provide accurate arrival information to passengers and support real-time operational control. Accurate predictions help passengers plan their trips, reduce waiting time, and allow operations staff to adjust headways, dispatch extra vehicles, and manage disruptions. Although real-time feeds such as GTFS-Realtime (GTFS-RT) are now widely available, most existing delay prediction systems handle only a few routes, depend on hand-crafted features, and offer little guidance on how to design a scalable, reusable architecture.
We present a city-scale prediction pipeline that combines multi-resolution feature engineering, dimensionality reduction, and deep learning. The framework generates 1,683 spatiotemporal features by exploring 23 aggregation combinations over H3 cells, routes, segments, and temporal patterns, and compresses them into 83 components using Adaptive PCA while preserving 95% of the variance. To avoid the "giant cluster" problem that occurs when dense urban areas fall into a single H3 region, we introduce a hybrid H3+topology clustering method that yields 12 balanced route clusters (coefficient of variation 0.608) and enables efficient distributed training.
We compare five model architectures on six months of bus operations from the Société de transport de Montréal (STM) network in Montréal. A global LSTM with cluster-aware features achieves the best trade-off between accuracy and efficiency, outperforming transformer models by 18 to 52% while using 275 times fewer parameters. We also report multi-level evaluation at the elementary segment, segment, and trip level with walk-forward validation and latency analysis, showing that the proposed pipeline is suitable for real-time, city-scale deployment and can be reused for other networks with limited adaptation. 

---
# Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates 

**Authors**: Yibo Li, Zijie Lin, Ailin Deng, Xuan Zhang, Yufei He, Shuo Ji, Tri Cao, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2601.18510)  

**Abstract**: While Large Language Model (LLM) agents excel at general tasks, they inherently struggle with continual adaptation due to the frozen weights after deployment. Conventional reinforcement learning (RL) offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting. We introduce Just-In-Time Reinforcement Learning (JitRL), a training-free framework that enables test-time policy optimization without any gradient updates. JitRL maintains a dynamic, non-parametric memory of experiences and retrieves relevant trajectories to estimate action advantages on-the-fly. These estimates are then used to directly modulate the LLM's output logits. We theoretically prove that this additive update rule is the exact closed-form solution to the KL-constrained policy optimization objective. Extensive experiments on WebArena and Jericho demonstrate that JitRL establishes a new state-of-the-art among training-free methods. Crucially, JitRL outperforms the performance of computationally expensive fine-tuning methods (e.g., WebRL) while reducing monetary costs by over 30 times, offering a scalable path for continual learning agents. The code is available at this https URL. 

---
# Funny or Persuasive, but Not Both: Evaluating Fine-Grained Multi-Concept Control in LLMs 

**Authors**: Arya Labroo, Ivaxi Sheth, Vyas Raina, Amaani Ahmed, Mario Fritz  

**Link**: [PDF](https://arxiv.org/pdf/2601.18483)  

**Abstract**: Large Language Models (LLMs) offer strong generative capabilities, but many applications require explicit and \textit{fine-grained} control over specific textual concepts, such as humor, persuasiveness, or formality. Prior approaches in prompting and representation engineering can provide coarse or single-attribute control, but systematic evaluation of multi-attribute settings remains limited. We introduce an evaluation framework for fine-grained controllability for both single- and dual-concept scenarios, focusing on linguistically distinct concept pairs (e.g., persuasiveness vs.~humor). Surprisingly, across multiple LLMs and generative tasks, we find that performance often drops in the dual-concept setting, even though the chosen concepts should in principle be separable. This reveals a fundamental limitation of naive prompting-based control: models struggle with compositionality even when concepts are intuitively independent. Our framework provides systematic evidence of this gap and offers a principled approach for measuring the ability of future methods for multi-concept control. 

---
# Fair-Eye Net: A Fair, Trustworthy, Multimodal Integrated Glaucoma Full Chain AI System 

**Authors**: Wenbin Wei, Suyuan Yao, Cheng Huang, Xiangyu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18464)  

**Abstract**: Glaucoma is a top cause of irreversible blindness globally, making early detection and longitudinal follow-up pivotal to preventing permanent vision loss. Current screening and progression assessment, however, rely on single tests or loosely linked examinations, introducing subjectivity and fragmented care. Limited access to high-quality imaging tools and specialist expertise further compromises consistency and equity in real-world use. To address these gaps, we developed Fair-Eye Net, a fair, reliable multimodal AI system closing the clinical loop from glaucoma screening to follow-up and risk alerting. It integrates fundus photos, OCT structural metrics, VF functional indices, and demographic factors via a dual-stream heterogeneous fusion architecture, with an uncertainty-aware hierarchical gating strategy for selective prediction and safe referral. A fairness constraint reduces missed diagnoses in disadvantaged subgroups. Experimental results show it achieved an AUC of 0.912 (96.7% specificity), cut racial false-negativity disparity by 73.4% (12.31% to 3.28%), maintained stable cross-domain performance, and enabled 3-12 months of early risk alerts (92% sensitivity, 88% specificity). Unlike post hoc fairness adjustments, Fair-Eye Net optimizes fairness as a primary goal with clinical reliability via multitask learning, offering a reproducible path for clinical translation and large-scale deployment to advance global eye health equity. 

---
# 3DGesPolicy: Phoneme-Aware Holistic Co-Speech Gesture Generation Based on Action Control 

**Authors**: Xuanmeng Sha, Liyun Zhang, Tomohiro Mashita, Naoya Chiba, Yuki Uranishi  

**Link**: [PDF](https://arxiv.org/pdf/2601.18451)  

**Abstract**: Generating holistic co-speech gestures that integrate full-body motion with facial expressions suffers from semantically incoherent coordination on body motion and spatially unstable meaningless movements due to existing part-decomposed or frame-level regression methods, We introduce 3DGesPolicy, a novel action-based framework that reformulates holistic gesture generation as a continuous trajectory control problem through diffusion policy from robotics. By modeling frame-to-frame variations as unified holistic actions, our method effectively learns inter-frame holistic gesture motion patterns and ensures both spatially and semantically coherent movement trajectories that adhere to realistic motion manifolds. To further bridge the gap in expressive alignment, we propose a Gesture-Audio-Phoneme (GAP) fusion module that can deeply integrate and refine multi-modal signals, ensuring structured and fine-grained alignment between speech semantics, body motion, and facial expressions. Extensive quantitative and qualitative experiments on the BEAT2 dataset demonstrate the effectiveness of our 3DGesPolicy across other state-of-the-art methods in generating natural, expressive, and highly speech-aligned holistic gestures. 

---
# GCFX: Generative Counterfactual Explanations for Deep Graph Models at the Model Level 

**Authors**: Jinlong Hu, Jiacheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18447)  

**Abstract**: Deep graph learning models have demonstrated remarkable capabilities in processing graph-structured data and have been widely applied across various fields. However, their complex internal architectures and lack of transparency make it difficult to explain their decisions, resulting in opaque models that users find hard to understand and trust. In this paper, we explore model-level explanation techniques for deep graph learning models, aiming to provide users with a comprehensive understanding of the models' overall decision-making processes and underlying mechanisms. Specifically, we address the problem of counterfactual explanations for deep graph learning models by introducing a generative model-level counterfactual explanation approach called GCFX, which is based on deep graph generation. This approach generates a set of high-quality counterfactual explanations that reflect the model's global predictive behavior by leveraging an enhanced deep graph generation framework and a global summarization algorithm. GCFX features an architecture that combines dual encoders, structure-aware taggers, and Message Passing Neural Network decoders, enabling it to accurately learn the true latent distribution of input data and generate high-quality, closely related counterfactual examples. Subsequently, a global counterfactual summarization algorithm selects the most representative and comprehensive explanations from numerous candidate counterfactuals, providing broad insights into the model's global predictive patterns. Experiments on a synthetic dataset and several real-world datasets demonstrate that GCFX outperforms existing methods in terms of counterfactual validity and coverage while maintaining low explanation costs, thereby offering crucial support for enhancing the practicality and trustworthiness of global counterfactual explanations. 

---
# Gradient Regularized Natural Gradients 

**Authors**: Satya Prakash Dash, Hossein Abdi, Wei Pan, Samuel Kaski, Mingfei Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.18420)  

**Abstract**: Gradient regularization (GR) has been shown to improve the generalizability of trained models. While Natural Gradient Descent has been shown to accelerate optimization in the initial phase of training, little attention has been paid to how the training dynamics of second-order optimizers can benefit from GR. In this work, we propose Gradient-Regularized Natural Gradients (GRNG), a family of scalable second-order optimizers that integrate explicit gradient regularization with natural gradient updates. Our framework provides two complementary algorithms: a frequentist variant that avoids explicit inversion of the Fisher Information Matrix (FIM) via structured approximations, and a Bayesian variant based on a Regularized-Kalman formulation that eliminates the need for FIM inversion entirely. We establish convergence guarantees for GRNG, showing that gradient regularization improves stability and enables convergence to global minima. Empirically, we demonstrate that GRNG consistently enhances both optimization speed and generalization compared to first-order methods (SGD, AdamW) and second-order baselines (K-FAC, Sophia), with strong results on vision and language benchmarks. Our findings highlight gradient regularization as a principled and practical tool to unlock the robustness of natural gradient methods for large-scale deep learning. 

---
# Emergent Cooperation in Quantum Multi-Agent Reinforcement Learning Using Communication 

**Authors**: Michael Kölle, Christian Reff, Leo Sünkel, Julian Hager, Gerhard Stenzel, Claudia Linnhoff-Popien  

**Link**: [PDF](https://arxiv.org/pdf/2601.18419)  

**Abstract**: Emergent cooperation in classical Multi-Agent Reinforcement Learning has gained significant attention, particularly in the context of Sequential Social Dilemmas (SSDs). While classical reinforcement learning approaches have demonstrated capability for emergent cooperation, research on extending these methods to Quantum Multi-Agent Reinforcement Learning remains limited, particularly through communication. In this paper, we apply communication approaches to quantum Q-Learning agents: the Mutual Acknowledgment Token Exchange (MATE) protocol, its extension Mutually Endorsed Distributed Incentive Acknowledgment Token Exchange (MEDIATE), the peer rewarding mechanism Gifting, and Reinforced Inter-Agent Learning (RIAL). We evaluate these approaches in three SSDs: the Iterated Prisoner's Dilemma, Iterated Stag Hunt, and Iterated Game of Chicken. Our experimental results show that approaches using MATE with temporal-difference measure (MATE\textsubscript{TD}), AutoMATE, MEDIATE-I, and MEDIATE-S achieved high cooperation levels across all dilemmas, demonstrating that communication is a viable mechanism for fostering emergent cooperation in Quantum Multi-Agent Reinforcement Learning. 

---
# daVinci-Dev: Agent-native Mid-training for Software Engineering 

**Authors**: Ji Zeng, Dayuan Fu, Tiantian Mi, Yumin Zhuang, Yaxing Huang, Xuefeng Li, Lyumanshan Ye, Muhang Xie, Qishuo Hua, Zhen Huang, Mohan Jiang, Hanning Wang, Jifan Lin, Yang Xiao, Jie Sun, Yunze Wu, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18418)  

**Abstract**: Recently, the frontier of Large Language Model (LLM) capabilities has shifted from single-turn code generation to agentic software engineering-a paradigm where models autonomously navigate, edit, and test complex repositories. While post-training methods have become the de facto approach for code agents, **agentic mid-training**-mid-training (MT) on large-scale data that mirrors authentic agentic workflows-remains critically underexplored due to substantial resource requirements, despite offering a more scalable path to instilling foundational agentic behaviors than relying solely on expensive reinforcement learning. A central challenge in realizing effective agentic mid-training is the distribution mismatch between static training data and the dynamic, feedback-rich environment of real development. To address this, we present a systematic study of agentic mid-training, establishing both the data synthesis principles and training methodology for effective agent development at scale. Central to our approach is **agent-native data**-supervision comprising two complementary types of trajectories: **contextually-native trajectories** that preserve the complete information flow an agent experiences, offering broad coverage and diversity; and **environmentally-native trajectories** collected from executable repositories where observations stem from actual tool invocations and test executions, providing depth and interaction authenticity. We verify the model's agentic capabilities on `SWE-Bench Verified`. We demonstrate our superiority over the previous open software engineering mid-training recipe `Kimi-Dev` under two post-training settings with an aligned base model and agentic scaffold, while using less than half mid-training tokens (73.1B). Besides relative advantage, our best performing 32B and 72B models achieve **56.1%** and **58.5%** resolution rates, respectively, which are ... 

---
# Code over Words: Overcoming Semantic Inertia via Code-Grounded Reasoning 

**Authors**: Manjie Xu, Isabella Yin, Xinyi Tu, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18352)  

**Abstract**: LLMs struggle with Semantic Inertia: the inability to inhibit pre-trained priors (e.g., "Lava is Dangerous") when dynamic, in-context rules contradict them. We probe this phenomenon using Baba Is You, where physical laws are mutable text rules, enabling precise evaluation of models' ability to override learned priors when rules change. We quantatively observe that larger models can exhibit inverse scaling: they perform worse than smaller models when natural language reasoning requires suppressing pre-trained associations (e.g., accepting "Lava is Safe"). Our analysis attributes this to natural language encoding, which entangles descriptive semantics and logical rules, leading to persistent hallucinations of familiar physics despite explicit contradictory rules. Here we show that representing dynamics as executable code, rather than descriptive text, reverses this trend and enables effective prior inhibition. We introduce Code-Grounded Vistas (LCV), which fine-tunes models on counterfactual pairs and identifies states with contradictory rules, thereby forcing attention to logical constraints rather than visual semantics. This training-time approach outperforms expensive inference-time search methods in both efficiency and accuracy. Our results demonstrate that representation fundamentally determines whether scaling improves or impairs contextual reasoning. This challenges the assumption that larger models are universally better, with implications for domains that require dynamic overriding of learned priors. 

---
# When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs 

**Authors**: Junyi Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.18350)  

**Abstract**: Large language models (LLMs) show strong general capability but often struggle with medical terminology precision and safety-critical instruction following. We present a case study for adapter interference in safety-critical domains using a 14B-parameter base model through a two-stage LoRA pipeline: (1) domain-adaptive pre-training (PT) to inject broad medical knowledge via continued pre-training (DAPT), and (2) supervised fine-tuning (SFT) to align the model with medical question-answering behaviors through instruction-style data. To balance instruction-following ability and domain knowledge retention, we propose Weighted Adapter Merging, linearly combining SFT and PT adapters before exporting a merged base-model checkpoint. On a held-out medical validation set (F5/F6), the merged model achieves BLEU-4 = 16.38, ROUGE-1 = 20.42, ROUGE-2 = 4.60, and ROUGE-L = 11.54 under a practical decoding configuration. We further analyze decoding sensitivity and training stability with loss curves and controlled decoding comparisons. 

---
# Analytic Incremental Learning For Sound Source Localization With Imbalance Rectification 

**Authors**: Zexia Fan, Yu Chen, Qiquan Zhang, Kainan Chen, Xinyuan Qian  

**Link**: [PDF](https://arxiv.org/pdf/2601.18335)  

**Abstract**: Sound source localization (SSL) demonstrates remarkable results in controlled settings but struggles in real-world deployment due to dual imbalance challenges: intra-task imbalance arising from long-tailed direction-of-arrival (DoA) distributions, and inter-task imbalance induced by cross-task skews and overlaps. These often lead to catastrophic forgetting, significantly degrading the localization accuracy. To mitigate these issues, we propose a unified framework with two key innovations. Specifically, we design a GCC-PHAT-based data augmentation (GDA) method that leverages peak characteristics to alleviate intra-task distribution skews. We also propose an Analytic dynamic imbalance rectifier (ADIR) with task-adaption regularization, which enables analytic updates that adapt to inter-task dynamics. On the SSLR benchmark, our proposal achieves state-of-the-art (SoTA) results of 89.0% accuracy, 5.3° mean absolute error, and 1.6 backward transfer, demonstrating robustness to evolving imbalances without exemplar storage. 

---
# MultiVis-Agent: A Multi-Agent Framework with Logic Rules for Reliable and Comprehensive Cross-Modal Data Visualization 

**Authors**: Jinwei Lu, Yuanfeng Song, Chen Zhang, Raymond Chi-Wing Wong  

**Link**: [PDF](https://arxiv.org/pdf/2601.18320)  

**Abstract**: Real-world visualization tasks involve complex, multi-modal requirements that extend beyond simple text-to-chart generation, requiring reference images, code examples, and iterative refinement. Current systems exhibit fundamental limitations: single-modality input, one-shot generation, and rigid workflows. While LLM-based approaches show potential for these complex requirements, they introduce reliability challenges including catastrophic failures and infinite loop susceptibility. To address this gap, we propose MultiVis-Agent, a logic rule-enhanced multi-agent framework for reliable multi-modal and multi-scenario visualization generation. Our approach introduces a four-layer logic rule framework that provides mathematical guarantees for system reliability while maintaining flexibility. Unlike traditional rule-based systems, our logic rules are mathematical constraints that guide LLM reasoning rather than replacing it. We formalize the MultiVis task spanning four scenarios from basic generation to iterative refinement, and develop MultiVis-Bench, a benchmark with over 1,000 cases for multi-modal visualization evaluation. Extensive experiments demonstrate that our approach achieves 75.63% visualization score on challenging tasks, significantly outperforming baselines (57.54-62.79%), with task completion rates of 99.58% and code execution success rates of 94.56% (vs. 74.48% and 65.10% without logic rules), successfully addressing both complexity and reliability challenges in automated visualization generation. 

---
# Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM 

**Authors**: Everlyn Asiko Chimoto, Mostafa Elhoushi, Bruce A. Bassett  

**Link**: [PDF](https://arxiv.org/pdf/2601.18306)  

**Abstract**: Quantization is an effective technique for reducing the storage footprint and computational costs of Large Language Models (LLMs), but it often results in performance degradation. Existing post-training quantization methods typically use small, English-only calibration sets; however, their impact on multilingual models remains underexplored. We systematically evaluate eight calibration settings (five single-language and three multilingual mixes) on two quantizers (GPTQ, AWQ) on data from 10 languages. Our findings reveal a consistent trend: non-English and multilingual calibration sets significantly improve perplexity compared to English-only baselines. Specifically, we observe notable average perplexity gains across both quantizers on Llama3.1 8B and Qwen2.5 7B, with multilingual mixes achieving the largest overall reductions of up to 3.52 points in perplexity. Furthermore, our analysis indicates that tailoring calibration sets to the evaluation language yields the largest improvements for individual languages, underscoring the importance of linguistic alignment. We also identify specific failure cases where certain language-quantizer combinations degrade performance, which we trace to differences in activation range distributions across languages. These results highlight that static one-size-fits-all calibration is suboptimal and that tailoring calibration data, both in language and diversity, plays a crucial role in robustly quantizing multilingual LLMs. 

---
# Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning 

**Authors**: Zhaoyan Gong, Zhiqiang Liu, Songze Li, Xiaoke Guo, Yuanxiang Liu, Xinle Deng, Zhizhen Liu, Lei Liang, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18296)  

**Abstract**: Temporal Knowledge Graph Question Answering (TKGQA) is inherently challenging, as it requires sophisticated reasoning over dynamic facts with multi-hop dependencies and complex temporal constraints. Existing methods rely on fixed workflows and expensive closed-source APIs, limiting flexibility and scalability. We propose Temp-R1, the first autonomous end-to-end agent for TKGQA trained through reinforcement learning. To address cognitive overload in single-action reasoning, we expand the action space with specialized internal actions alongside external action. To prevent shortcut learning on simple questions, we introduce reverse curriculum learning that trains on difficult questions first, forcing the development of sophisticated reasoning before transferring to easier cases. Our 8B-parameter Temp-R1 achieves state-of-the-art performance on MultiTQ and TimelineKGQA, improving 19.8% over strong baselines on complex questions. Our work establishes a new paradigm for autonomous temporal reasoning agents. Our code will be publicly available soon at this https URL. 

---
# TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment 

**Authors**: Zhewen Tan, Wenhan Yu, Jianfeng Si, Tongxin Liu, Kaiqi Guan, Huiyan Jin, Jiawen Tao, Xiaokun Yuan, Duohe Ma, Xiangzheng Zhang, Tong Yang, Lin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.18292)  

**Abstract**: In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop. 

---
# What Do Learned Models Measure? 

**Authors**: Indrė Žliobaitė  

**Link**: [PDF](https://arxiv.org/pdf/2601.18278)  

**Abstract**: In many scientific and data-driven applications, machine learning models are increasingly used as measurement instruments, rather than merely as predictors of predefined labels. When the measurement function is learned from data, the mapping from observations to quantities is determined implicitly by the training distribution and inductive biases, allowing multiple inequivalent mappings to satisfy standard predictive evaluation criteria. We formalize learned measurement functions as a distinct focus of evaluation and introduce measurement stability, a property capturing invariance of the measured quantity across admissible realizations of the learning process and across contexts. We show that standard evaluation criteria in machine learning, including generalization error, calibration, and robustness, do not guarantee measurement stability. Through a real-world case study, we show that models with comparable predictive performance can implement systematically inequivalent measurement functions, with distribution shift providing a concrete illustration of this failure. Taken together, our results highlight a limitation of existing evaluation frameworks in settings where learned model outputs are identified as measurements, motivating the need for an additional evaluative dimension. 

---
# Neural Network Approximation: A View from Polytope Decomposition 

**Authors**: ZeYu Li, ShiJun Zhang, TieYong Zeng, FengLei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2601.18264)  

**Abstract**: Universal approximation theory offers a foundational framework to verify neural network expressiveness, enabling principled utilization in real-world applications. However, most existing theoretical constructions are established by uniformly dividing the input space into tiny hypercubes without considering the local regularity of the target function. In this work, we investigate the universal approximation capabilities of ReLU networks from a view of polytope decomposition, which offers a more realistic and task-oriented approach compared to current methods. To achieve this, we develop an explicit kernel polynomial method to derive an universal approximation of continuous functions, which is characterized not only by the refined Totik-Ditzian-type modulus of continuity, but also by polytopical domain decomposition. Then, a ReLU network is constructed to approximate the kernel polynomial in each subdomain separately. Furthermore, we find that polytope decomposition makes our approximation more efficient and flexible than existing methods in many cases, especially near singular points of the objective function. Lastly, we extend our approach to analytic functions to reach a higher approximation rate. 

---
# Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs 

**Authors**: Fei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2601.18255)  

**Abstract**: Continual learning in Large Language Models (LLMs) faces the critical challenge of balancing stability (retaining old knowledge) and plasticity (learning new tasks). While Experience Replay (ER) is a standard countermeasure against catastrophic forgetting, its impact across diverse capabilities remains underexplored. In this work, we uncover a critical dichotomy in ER's behavior: while it induces positive backward transfer on robust, unstructured tasks (e.g., boosting performance on previous NLP classification tasks through repeated rehearsal), it causes severe negative transfer on fragile, structured domains like code generation (e.g., a significant relative drop in coding accuracy). This reveals that ER trades structural integrity for broad consolidation. To address this dilemma, we propose \textbf{Orthogonal Subspace Wake-up (OSW)}. OSW identifies essential parameter subspaces of previous tasks via a brief "wake-up" phase and enforces orthogonal updates for new tasks, providing a mathematically grounded "safety guarantee" for established knowledge structures. Empirical results across a diverse four-task sequence demonstrate that OSW uniquely succeeds in preserving fragile coding abilities where Replay fails, while simultaneously maintaining high plasticity for novel tasks. Our findings emphasize the necessity of evaluating structural safety alongside average retention in LLM continual learning. 

---
# BoRP: Bootstrapped Regression Probing for Scalable and Human-Aligned LLM Evaluation 

**Authors**: Peng Sun, Xiangyu Zhang, Duan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18253)  

**Abstract**: Accurate evaluation of user satisfaction is critical for iterative development of conversational AI. However, for open-ended assistants, traditional A/B testing lacks reliable metrics: explicit feedback is sparse, while implicit metrics are ambiguous. To bridge this gap, we introduce BoRP (Bootstrapped Regression Probing), a scalable framework for high-fidelity satisfaction evaluation. Unlike generative approaches, BoRP leverages the geometric properties of LLM latent space. It employs a polarization-index-based bootstrapping mechanism to automate rubric generation and utilizes Partial Least Squares (PLS) to map hidden states to continuous scores. Experiments on industrial datasets show that BoRP (Qwen3-8B/14B) significantly outperforms generative baselines (even Qwen3-Max) in alignment with human judgments. Furthermore, BoRP reduces inference costs by orders of magnitude, enabling full-scale monitoring and highly sensitive A/B testing via CUPED. 

---
# Co-PLNet: A Collaborative Point-Line Network for Prompt-Guided Wireframe Parsing 

**Authors**: Chao Wang, Xuanying Li, Cheng Dai, Jinglei Feng, Yuxiang Luo, Yuqi Ouyang, Hao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2601.18252)  

**Abstract**: Wireframe parsing aims to recover line segments and their junctions to form a structured geometric representation useful for downstream tasks such as Simultaneous Localization and Mapping (SLAM). Existing methods predict lines and junctions separately and reconcile them post-hoc, causing mismatches and reduced robustness. We present Co-PLNet, a point-line collaborative framework that exchanges spatial cues between the two tasks, where early detections are converted into spatial prompts via a Point-Line Prompt Encoder (PLP-Encoder), which encodes geometric attributes into compact and spatially aligned maps. A Cross-Guidance Line Decoder (CGL-Decoder) then refines predictions with sparse attention conditioned on complementary prompts, enforcing point-line consistency and efficiency. Experiments on Wireframe and YorkUrban show consistent improvements in accuracy and robustness, together with favorable real-time efficiency, demonstrating our effectiveness for structured geometry perception. 

---
# A multimodal vision foundation model for generalizable knee pathology 

**Authors**: Kang Yu, Dingyu Wang, Zimu Yuan, Nan Zhou, Jiajun Liu, Jiaxin Liu, Shanggui Liu, Yaoyan Zheng, Huishu Yuan, Di Huang, Dong Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18250)  

**Abstract**: Musculoskeletal disorders represent a leading cause of global disability, creating an urgent demand for precise interpretation of medical imaging. Current artificial intelligence (AI) approaches in orthopedics predominantly rely on task-specific, supervised learning paradigms. These methods are inherently fragmented, require extensive annotated datasets, and often lack generalizability across different modalities and clinical scenarios. The development of foundation models in this field has been constrained by the scarcity of large-scale, curated, and open-source musculoskeletal datasets. To address these challenges, we introduce OrthoFoundation, a multimodal vision foundation model optimized for musculoskeletal pathology. We constructed a pre-training dataset of 1.2 million unlabeled knee X-ray and MRI images from internal and public databases. Utilizing a Dinov3 backbone, the model was trained via self-supervised contrastive learning to capture robust radiological representations. OrthoFoundation achieves state-of-the-art (SOTA) performance across 14 downstream tasks. It attained superior accuracy in X-ray osteoarthritis diagnosis and ranked first in MRI structural injury detection. The model demonstrated remarkable label efficiency, matching supervised baselines using only 50% of labeled data. Furthermore, despite being pre-trained on knee images, OrthoFoundation exhibited exceptional cross-anatomy generalization to the hip, shoulder, and ankle. OrthoFoundation represents a significant advancement toward general-purpose AI for musculoskeletal imaging. By learning fundamental, joint-agnostic radiological semantics from large-scale multimodal data, it overcomes the limitations of conventional models, which provides a robust framework for reducing annotation burdens and enhancing diagnostic accuracy in clinical practice. 

---
# TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance 

**Authors**: Elena Bruches, Vadim Alperovich, Dari Baturova, Roman Derunets, Daniil Grebenkin, Georgy Mkrtchyan, Oleg Sedukhin, Mikhail Klementev, Ivan Bondarenko, Nikolay Bushkov, Stanislav Moiseev  

**Link**: [PDF](https://arxiv.org/pdf/2601.18241)  

**Abstract**: While Large Language Models (LLMs) have shown promise in software engineering, their application to unit testing remains largely confined to isolated test generation or oracle prediction, neglecting the broader challenge of test suite maintenance. We introduce TAM-Eval (Test Automated Maintenance Evaluation), a framework and benchmark designed to evaluate model performance across three core test maintenance scenarios: creation, repair, and updating of test suites. Unlike prior work limited to function-level tasks, TAM-Eval operates at the test file level, while maintaining access to full repository context during isolated evaluation, better reflecting real-world maintenance workflows. Our benchmark comprises 1,539 automatically extracted and validated scenarios from Python, Java, and Go projects. TAM-Eval supports system-agnostic evaluation of both raw LLMs and agentic workflows, using a reference-free protocol based on test suite pass rate, code coverage, and mutation testing. Empirical results indicate that state-of-the-art LLMs have limited capabilities in realistic test maintenance processes and yield only marginal improvements in test effectiveness. We release TAM-Eval as an open-source framework to support future research in automated software testing. Our data and code are publicly available at this https URL. 

---
# Generative AI in Saudi Arabia: A National Survey of Adoption, Risks, and Public Perceptions 

**Authors**: Abdulaziz AlDakheel, Ali Alshehre, Esraa Alamoudi, Moslim AlKhabbaz, Ahmed Aljohani, Raed Alharbi  

**Link**: [PDF](https://arxiv.org/pdf/2601.18234)  

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly becoming embedded in Saudi Arabia's digital transformation under Vision 2030, yet public awareness, adoption, and concerns surrounding these tools remain underexplored. This study provides an early snapshot of GenAI engagement among Saudi nationals. Using a nationwide survey of 330 participants across regions, age groups, and employment sectors, we examine seven dimensions of GenAI use: awareness and understanding, adoption patterns, perceived impacts, training needs, risks and barriers, data-sharing behaviors, and future expectations. Findings show that 93% of respondents actively use GenAI primarily for text-based tasks, while more advanced uses such as programming or multimodal generation are less common. Despite the prevalence of use, overall awareness and conceptual understanding remain uneven, with many reporting limited technical knowledge. Participants recognize GenAI's benefits for productivity, work quality, and understanding complex information, yet caution that sustained reliance may undermine critical thinking and key professional skills. Trust in AI-generated outputs remains cautious, with widespread concerns about privacy, misinformation, and ethical misuse, including potential job displacement. Respondents show strong interest in structured GenAI training that combines foundational skills, domain-specific applications, and clear guidance on privacy, ethics, and responsible use. These results establish a baseline for GenAI engagement in Saudi Arabia and highlight priorities for policymakers and developers: expanding AI literacy, ensuring culturally and linguistically aligned GenAI solutions, and strengthening frameworks for privacy and responsible deployment. 

---
# Rethinking Cross-Modal Fine-Tuning: Optimizing the Interaction between Feature Alignment and Target Fitting 

**Authors**: Trong Khiem Tran, Manh Cuong Dao, Phi Le Nguyen, Thao Nguyen Truong, Trong Nghia Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18231)  

**Abstract**: Adapting pre-trained models to unseen feature modalities has become increasingly important due to the growing need for cross-disciplinary knowledge integration.~A key challenge here is how to align the representation of new modalities with the most relevant parts of the pre-trained model's representation space to enable accurate knowledge transfer.~This requires combining feature alignment with target fine-tuning, but uncalibrated combinations can exacerbate misalignment between the source and target feature-label structures and reduce target generalization.~Existing work however lacks a theoretical understanding of this critical interaction between feature alignment and target fitting.~To bridge this gap, we develop a principled framework that establishes a provable generalization bound on the target error, which explains the interaction between feature alignment and target fitting through a novel concept of feature-label distortion.~This bound offers actionable insights into how this interaction should be optimized for practical algorithm design. The resulting approach achieves significantly improved performance over state-of-the-art methods across a wide range of benchmark datasets. 

---
# PaperTok: Exploring the Use of Generative AI for Creating Short-form Videos for Research Communication 

**Authors**: Meziah Ruby Cristobal, Hyeonjeong Byeon, Tze-Yu Chen, Ruoxi Shang, Donghoon Shin, Ruican Zhong, Tony Zhou, Gary Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2601.18218)  

**Abstract**: The dissemination of scholarly research is critical, yet researchers often lack the time and skills to create engaging content for popular media such as short-form videos. To address this gap, we explore the use of generative AI to help researchers transform their academic papers into accessible video content. Informed by a formative study with science communicators and content creators (N=8), we designed PaperTok, an end-to-end system that automates the initial creative labor by generating script options and corresponding audiovisual content from a source paper. Researchers can then refine based on their preferences with further prompting. A mixed-methods user study (N=18) and crowdsourced evaluation (N=100) demonstrate that PaperTok's workflow can help researchers create engaging and informative short-form videos. We also identified the need for more fine-grained controls in the creation process. To this end, we offer implications for future generative tools that support science outreach. 

---
# PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR 

**Authors**: James Burgess, Jan N. Hansen, Duo Peng, Yuhui Zhang, Alejandro Lozano, Min Woo Sun, Emma Lundberg, Serena Yeung-Levy  

**Link**: [PDF](https://arxiv.org/pdf/2601.18207)  

**Abstract**: Search agents are language models (LMs) that reason and search knowledge bases (or the web) to answer questions; recent methods supervise only the final answer accuracy using reinforcement learning with verifiable rewards (RLVR). Most RLVR search agents tackle general-domain QA, which limits their relevance to technical AI systems in science, engineering, and medicine. In this work we propose training agents to search and reason over scientific papers -- this tests technical question-answering, it is directly relevant to real scientists, and the capabilities will be crucial to future AI Scientist systems. Concretely, we release a search corpus of 16 million biomedical paper abstracts and construct a challenging factoid QA dataset called PaperSearchQA with 60k samples answerable from the corpus, along with benchmarks. We train search agents in this environment to outperform non-RL retrieval baselines; we also perform further quantitative analysis and observe interesting agent behaviors like planning, reasoning, and self-verification. Our corpus, datasets, and benchmarks are usable with the popular Search-R1 codebase for RLVR training and released on this https URL. Finally, our data creation methods are scalable and easily extendable to other scientific domains. 

---
# HeterCSI: Channel-Adaptive Heterogeneous CSI Pretraining Framework for Generalized Wireless Foundation Models 

**Authors**: Chenyu Zhang, Xinchen Lyu, Chenshan Ren, Shuhan Liu, Qimei Cui, Xiaofeng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2601.18200)  

**Abstract**: Wireless foundation models promise transformative capabilities for channel state information (CSI) processing across diverse 6G network applications, yet face fundamental challenges due to the inherent dual heterogeneity of CSI across both scale and scenario dimensions. However, current pretraining approaches either constrain inputs to fixed dimensions or isolate training by scale, limiting the generalization and scalability of wireless foundation models. In this paper, we propose HeterCSI, a channel-adaptive pretraining framework that reconciles training efficiency with robust cross-scenario generalization via a new understanding of gradient dynamics in heterogeneous CSI pretraining. Our key insight reveals that CSI scale heterogeneity primarily causes destructive gradient interference, while scenario diversity actually promotes constructive gradient alignment when properly managed. Specifically, we formulate heterogeneous CSI batch construction as a partitioning optimization problem that minimizes zero-padding overhead while preserving scenario diversity. To solve this, we develop a scale-aware adaptive batching strategy that aligns CSI samples of similar scales, and design a double-masking mechanism to isolate valid signals from padding artifacts. Extensive experiments on 12 datasets demonstrate that HeterCSI establishes a generalized foundation model without scenario-specific finetuning, achieving superior average performance over full-shot baselines. Compared to the state-of-the-art zero-shot benchmark WiFo, it reduces NMSE by 7.19 dB, 4.08 dB, and 5.27 dB for CSI reconstruction, time-domain, and frequency-domain prediction, respectively. The proposed HeterCSI framework also reduces training latency by 53% compared to existing approaches while improving generalization performance by 1.53 dB on average. 

---
# \textsc{NaVIDA}: Vision-Language Navigation with Inverse Dynamics Augmentation 

**Authors**: Weiye Zhu, Zekai Zhang, Xiangchen Wang, Hewei Pan, Teng Wang, Tiantian Geng, Rongtao Xu, Feng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.18188)  

**Abstract**: Vision-and-Language Navigation (VLN) requires agents to interpret natural language instructions and act coherently in visually rich environments. However, most existing methods rely on reactive state-action mappings without explicitly modeling how actions causally transform subsequent visual observations. Lacking such vision-action causality, agents cannot anticipate the visual changes induced by its own actions, leading to unstable behaviors, weak generalization, and cumulative error along trajectory. To address these issues, we introduce \textsc{NaVIDA} (\textbf{Nav}igation with \textbf{I}nverse \textbf{D}ynamics \textbf{A}ugmentation), a unified VLN framework that couples policy learning with action-grounded visual dynamics and adaptive execution. \textsc{NaVIDA} augments training with chunk-based inverse-dynamics supervision to learn causal relationship between visual changes and corresponding actions. To structure this supervision and extend the effective planning range, \textsc{NaVIDA} employs hierarchical probabilistic action chunking (HPAC), which organizes trajectories into multi-step chunks and provides discriminative, longer-range visual-change cues. To further curb error accumulation and stabilize behavior at inference, an entropy-guided mechanism adaptively sets the execution horizon of action chunks. Extensive experiments show that \textsc{NaVIDA} achieves superior navigation performance compared to state-of-the-art methods with fewer parameters (3B vs. 8B). Real-world robot evaluations further validate the practical feasibility and effectiveness of our approach. Code and data will be available upon acceptance. 

---
# VIBEVOICE-ASR Technical Report 

**Authors**: Zhiliang Peng, Jianwei Yu, Yaoyao Chang, Zilong Wang, Li Dong, Yingbo Hao, Yujie Tu, Chenyu Yang, Wenhui Wang, Songchen Xu, Yutao Sun, Hangbo Bao, Weijiang Xu, Yi Zhu, Zehua Wang, Ting Song, Yan Xia, Zewen Chi, Shaohan Huang, Liang Wang, Chuang Ding, Shuai Wang, Xie Chen, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.18184)  

**Abstract**: This report presents VibeVoice-ASR, a general-purpose speech understanding framework built upon VibeVoice, designed to address the persistent challenges of context fragmentation and multi-speaker complexity in long-form audio (e.g., meetings, podcasts) that remain despite recent advancements in short-form speech recognition. Unlike traditional pipelined approaches that rely on audio chunking, VibeVoice-ASRsupports single-pass processing for up to 60 minutes of audio. It unifies Automatic Speech Recognition, Speaker Diarization, and Timestamping into a single end-to-end generation task. In addition, VibeVoice-ASR supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Furthermore, we introduce a prompt-based context injection mechanism that allows users to supply customized conetxt, significantly improving accuracy on domain-specific terminology and polyphonic character disambiguation. 

---
# Beyond Pairwise Comparisons: A Distributional Test of Distinctiveness for Machine-Generated Works in Intellectual Property Law 

**Authors**: Anirban Mukherjee, Hannah Hanwen Chang  

**Link**: [PDF](https://arxiv.org/pdf/2601.18156)  

**Abstract**: Key doctrines, including novelty (patent), originality (copyright), and distinctiveness (trademark), turn on a shared empirical question: whether a body of work is meaningfully distinct from a relevant reference class. Yet analyses typically operationalize this set-level inquiry using item-level evidence: pairwise comparisons among exemplars. That unit-of-analysis mismatch may be manageable for finite corpora of human-created works, where it can be bridged by ad hoc aggregations. But it becomes acute for machine-generated works, where the object of evaluation is not a fixed set of works but a generative process with an effectively unbounded output space. We propose a distributional alternative: a two-sample test based on maximum mean discrepancy computed on semantic embeddings to determine if two creative processes-whether human or machine-produce statistically distinguishable output distributions. The test requires no task-specific training-obviating the need for discovery of proprietary training data to characterize the generative process-and is sample-efficient, often detecting differences with as few as 5-10 images and 7-20 texts. We validate the framework across three domains: handwritten digits (controlled images), patent abstracts (text), and AI-generated art (real-world images). We reveal a perceptual paradox: even when human evaluators distinguish AI outputs from human-created art with only about 58% accuracy, our method detects distributional distinctiveness. Our results present evidence contrary to the view that generative models act as mere regurgitators of training data. Rather than producing outputs statistically indistinguishable from a human baseline-as simple regurgitation would predict-they produce outputs that are semantically human-like yet stochastically distinct, suggesting their dominant function is as a semantic interpolator within a learned latent space. 

---
# Explaining Synergistic Effects in Social Recommendations 

**Authors**: Yicong Li, Shan Jin, Qi Liu, Shuo Wang, Jiaying Liu, Shuo Yu, Qiang Zhang, Kuanjiu Zhou, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2601.18151)  

**Abstract**: In social recommenders, the inherent nonlinearity and opacity of synergistic effects across multiple social networks hinders users from understanding how diverse information is leveraged for recommendations, consequently diminishing explainability. However, existing explainers can only identify the topological information in social networks that significantly influences recommendations, failing to further explain the synergistic effects among this information. Inspired by existing findings that synergistic effects enhance mutual information between inputs and predictions to generate information gain, we extend this discovery to graph data. We quantify graph information gain to identify subgraphs embodying synergistic effects. Based on the theoretical insights, we propose SemExplainer, which explains synergistic effects by identifying subgraphs that embody them. SemExplainer first extracts explanatory subgraphs from multi-view social networks to generate preliminary importance explanations for recommendations. A conditional entropy optimization strategy to maximize information gain is developed, thereby further identifying subgraphs that embody synergistic effects from explanatory subgraphs. Finally, SemExplainer searches for paths from users to recommended items within the synergistic subgraphs to generate explanations for the recommendations. Extensive experiments on three datasets demonstrate the superiority of SemExplainer over baseline methods, providing superior explanations of synergistic effects. 

---
# Typhoon-S: Minimal Open Post-Training for Sovereign Large Language Models 

**Authors**: Kunat Pipatanakul, Pittawat Taveekitworachai  

**Link**: [PDF](https://arxiv.org/pdf/2601.18129)  

**Abstract**: Large language models (LLMs) have progressed rapidly; however, most state-of-the-art models are trained and evaluated primarily in high-resource languages such as English and Chinese, and are often developed by a small number of organizations with access to large-scale compute and data. This gatekeeping creates a practical barrier for sovereign settings in which a regional- or national-scale institution or domain owner must retain control and understanding of model weights, training data, and deployment while operating under limited resources and strict transparency constraints. To this end, we identify two core requirements: (1) adoptability, the ability to transform a base model into a general-purpose assistant, and (2) sovereign capability, the ability to perform high-stakes, region-specific tasks (e.g., legal reasoning in local languages and cultural knowledge). We investigate whether these requirements can be achieved without scaling massive instruction corpora or relying on complex preference tuning pipelines and large-scale reinforcement fine-tuning (RFT). We present Typhoon S, a minimal and open post-training recipe that combines supervised fine-tuning, on-policy distillation, and small-scale RFT. Using Thai as a representative case study, we demonstrate that our approach transforms both sovereign-adapted and general-purpose base models into instruction-tuned models with strong general performance. We further show that small-scale RFT with InK-GRPO -- an extension of GRPO that augments the GRPO loss with a next-word prediction loss -- improves Thai legal reasoning and Thai-specific knowledge while preserving general capabilities. Our results suggest that a carefully designed post-training strategy can reduce the required scale of instruction data and computation, providing a practical path toward high-quality sovereign LLMs under academic-scale resources. 

---
# The Limits of AI Data Transparency Policy: Three Disclosure Fallacies 

**Authors**: Judy Hanwen Shen, Ken Liu, Angelina Wang, Sarah H. Cen, Andy K. Zhang, Caroline Meinhardt, Daniel Zhang, Kevin Klyman, Rishi Bommasani, Daniel E. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2601.18127)  

**Abstract**: Data transparency has emerged as a rallying cry for addressing concerns about AI: data quality, privacy, and copyright chief among them. Yet while these calls are crucial for accountability, current transparency policies often fall short of their intended aims. Similar to nutrition facts for food, policies aimed at nutrition facts for AI currently suffer from a limited consideration of research on effective disclosures. We offer an institutional perspective and identify three common fallacies in policy implementations of data disclosures for AI. First, many data transparency proposals exhibit a specification gap between the stated goals of data transparency and the actual disclosures necessary to achieve such goals. Second, reform attempts exhibit an enforcement gap between required disclosures on paper and enforcement to ensure compliance in fact. Third, policy proposals manifest an impact gap between disclosed information and meaningful changes in developer practices and public understanding. Informed by the social science on transparency, our analysis identifies affirmative paths for transparency that are effective rather than merely symbolic. 

---
# Understanding Users' Privacy Reasoning and Behaviors During Chatbot Use to Support Meaningful Agency in Privacy 

**Authors**: Mohammad Hadi Nezhad, Francisco Enrique Vicente Castro, Ivon Arroyo  

**Link**: [PDF](https://arxiv.org/pdf/2601.18125)  

**Abstract**: Conversational agents (CAs) (e.g., chatbots) are increasingly used in settings where users disclose sensitive information, raising significant privacy concerns. Because privacy judgments are highly contextual, supporting users to engage in privacy-protective actions during chatbot interactions is essential. However, enabling meaningful engagement requires a deeper understanding of how users currently reason about and manage sensitive information during realistic chatbot use scenarios. To investigate this, we qualitatively examined computer science (undergraduate and masters) students' in-the-moment disclosure and protection behaviors, as well as the reasoning underlying these behaviors, across a range of realistic chatbot tasks. Participants used a simulated ChatGPT interface with and without a privacy notice panel that intercepts message submissions, highlights potentially sensitive information, and offers privacy protective actions. The panel supports anonymization through retracting, faking, and generalizing, and surfaces two of ChatGPT's built-in privacy controls to improve their discoverability. Drawing on interaction logs, think-alouds, and survey responses, we analyzed how the panel fostered privacy awareness, encouraged protective actions, and supported context-specific reasoning about what information to protect and how. We further discuss design opportunities for tools that provide users greater and more meaningful agency in protecting sensitive information during CA interactions. 

---
# LungCRCT: Causal Representation based Lung CT Processing for Lung Cancer Treatment 

**Authors**: Daeyoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.18118)  

**Abstract**: Due to silence in early stages, lung cancer has been one of the most leading causes of mortality in cancer patients world-wide. Moreover, major symptoms of lung cancer are hard to differentiate with other respiratory disease symptoms such as COPD, further leading patients to overlook cancer progression in early stages. Thus, to enhance survival rates in lung cancer, early detection from consistent proactive respiratory system monitoring becomes crucial. One of the most prevalent and effective methods for lung cancer monitoring would be low-dose computed tomography(LDCT) chest scans, which led to remarkable enhancements in lung cancer detection or tumor classification tasks under rapid advancements and applications of computer vision based AI models such as EfficientNet or ResNet in image processing. However, though advanced CNN models under transfer learning or ViT based models led to high performing lung cancer detections, due to its intrinsic limitations in terms of correlation dependence and low interpretability due to complexity, expansions of deep learning models to lung cancer treatment analysis or causal intervention analysis simulations are still limited. Therefore, this research introduced LungCRCT: a latent causal representation learning based lung cancer analysis framework that retrieves causal representations of factors within the physical causal mechanism of lung cancer progression. With the use of advanced graph autoencoder based causal discovery algorithms with distance Correlation disentanglement and entropy-based image reconstruction refinement, LungCRCT not only enables causal intervention analysis for lung cancer treatments, but also leads to robust, yet extremely light downstream models in malignant tumor classification tasks with an AUC score of 93.91%. 

---
# MalURLBench: A Benchmark Evaluating Agents' Vulnerabilities When Processing Web URLs 

**Authors**: Dezhang Kong, Zhuxi Wu, Shiqi Liu, Zhicheng Tan, Kuichen Lu, Minghao Li, Qichen Liu, Shengyu Chu, Zhenhua Xu, Xuan Liu, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.18113)  

**Abstract**: LLM-based web agents have become increasingly popular for their utility in daily life and work. However, they exhibit critical vulnerabilities when processing malicious URLs: accepting a disguised malicious URL enables subsequent access to unsafe webpages, which can cause severe damage to service providers and users. Despite this risk, no benchmark currently targets this emerging threat. To address this gap, we propose MalURLBench, the first benchmark for evaluating LLMs' vulnerabilities to malicious URLs. MalURLBench contains 61,845 attack instances spanning 10 real-world scenarios and 7 categories of real malicious websites. Experiments with 12 popular LLMs reveal that existing models struggle to detect elaborately disguised malicious URLs. We further identify and analyze key factors that impact attack success rates and propose URLGuard, a lightweight defense module. We believe this work will provide a foundational resource for advancing the security of web agents. Our code is available at this https URL. 

---
# Demystifying Data-Driven Probabilistic Medium-Range Weather Forecasting 

**Authors**: Jean Kossaifi, Nikola Kovachki, Morteza Mardani, Daniel Leibovici, Suman Ravuri, Ira Shokar, Edoardo Calvello, Mohammad Shoaib Abbas, Peter Harrington, Ashay Subramaniam, Noah Brenowitz, Boris Bonev, Wonmin Byeon, Karsten Kreis, Dale Durran, Arash Vahdat, Mike Pritchard, Jan Kautz  

**Link**: [PDF](https://arxiv.org/pdf/2601.18111)  

**Abstract**: The recent revolution in data-driven methods for weather forecasting has lead to a fragmented landscape of complex, bespoke architectures and training strategies, obscuring the fundamental drivers of forecast accuracy. Here, we demonstrate that state-of-the-art probabilistic skill requires neither intricate architectural constraints nor specialized training heuristics. We introduce a scalable framework for learning multi-scale atmospheric dynamics by combining a directly downsampled latent space with a history-conditioned local projector that resolves high-resolution physics. We find that our framework design is robust to the choice of probabilistic estimator, seamlessly supporting stochastic interpolants, diffusion models, and CRPS-based ensemble training. Validated against the Integrated Forecasting System and the deep learning probabilistic model GenCast, our framework achieves statistically significant improvements on most of the variables. These results suggest scaling a general-purpose model is sufficient for state-of-the-art medium-range prediction, eliminating the need for tailored training recipes and proving effective across the full spectrum of probabilistic frameworks. 

---
# Mitigating the OWASP Top 10 For Large Language Models Applications using Intelligent Agents 

**Authors**: Mohammad Fasha, Faisal Abul Rub, Nasim Matar, Bilal Sowan, Mohammad Al Khaldy  

**Link**: [PDF](https://arxiv.org/pdf/2601.18105)  

**Abstract**: Large Language Models (LLMs) have emerged as a transformative and disruptive technology, enabling a wide range of applications in natural language processing, machine translation, and beyond. However, this widespread integration of LLMs also raised several security concerns highlighted by the Open Web Application Security Project (OWASP), which has identified the top 10 security vulnerabilities inherent in LLM applications. Addressing these vulnerabilities is crucial, given the increasing reliance on LLMs and the potential threats to data integrity, confidentiality, and service availability. This paper presents a framework designed to mitigate the security risks outlined in the OWASP Top 10. Our proposed model leverages LLM-enabled intelligent agents, offering a new approach to proactively identify, assess, and counteract security threats in real-time. The proposed framework serves as an initial blueprint for future research and development, aiming to enhance the security measures of LLMs and protect against emerging threats in this rapidly evolving landscape. 

---
# LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts 

**Authors**: Venmugil Elango, Nidhi Bhatia, Roger Waleffe, Rasoul Shafipour, Tomer Asida, Abhinav Khattar, Nave Assaf, Maximilian Golub, Joey Guman, Tiyasa Mitra, Ritchie Zhao, Ritika Borkar, Ran Zilberstein, Mostofa Patwary, Mohammad Shoeybi, Bita Rouhani  

**Link**: [PDF](https://arxiv.org/pdf/2601.18089)  

**Abstract**: Mixture of Experts (MoEs) have become a central component of many state-of-the-art open-source and proprietary large language models. Despite their widespread adoption, it remains unclear how close existing MoE architectures are to optimal with respect to inference cost, as measured by accuracy per floating-point operation and per parameter. In this work, we revisit MoE design from a hardware-software co-design perspective, grounded in empirical and theoretical considerations. We characterize key performance bottlenecks across diverse deployment regimes, spanning offline high-throughput execution and online, latency-critical inference. Guided by these insights, we introduce LatentMoE, a new model architecture resulting from systematic design exploration and optimized for maximal accuracy per unit of compute. Empirical design space exploration at scales of up to 95B parameters and over a 1T-token training horizon, together with supporting theoretical analysis, shows that LatentMoE consistently outperforms standard MoE architectures in terms of accuracy per FLOP and per parameter. Given its strong performance, the LatentMoE architecture has been adopted by the flagship Nemotron-3 Super and Ultra models and scaled to substantially larger regimes, including longer token horizons and larger model sizes, as reported in Nvidia et al. (arXiv:2512.20856). 

---
# "Crash Test Dummies" for AI-Enabled Clinical Assessment: Validating Virtual Patient Scenarios with Virtual Learners 

**Authors**: Brian Gin, Ahreum Lim, Flávia Silva e Oliveira, Kuan Xing, Xiaomei Song, Gayana Amiyangoda, Thilanka Seneviratne, Alison F. Doubleday, Ananya Gangopadhyaya, Bob Kiser, Lukas Shum-Tim, Dhruva Patel, Kosala Marambe, Lauren Maggio, Ara Tekian, Yoon Soo Park  

**Link**: [PDF](https://arxiv.org/pdf/2601.18085)  

**Abstract**: Background: In medical and health professions education (HPE), AI is increasingly used to assess clinical competencies, including via virtual standardized patients. However, most evaluations rely on AI-human interrater reliability and lack a measurement framework for how cases, learners, and raters jointly shape scores. This leaves robustness uncertain and can expose learners to misguidance from unvalidated systems. We address this by using AI "simulated learners" to stress-test and psychometrically characterize assessment pipelines before human use.
Objective: Develop an open-source AI virtual patient platform and measurement model for robust competency evaluation across cases and rating conditions.
Methods: We built a platform with virtual patients, virtual learners with tunable ACGME-aligned competency profiles, and multiple independent AI raters scoring encounters with structured Key-Features items. Transcripts were analyzed with a Bayesian HRM-SDT model that treats ratings as decisions under uncertainty and separates learner ability, case performance, and rater behavior; parameters were estimated with MCMC.
Results: The model recovered simulated learners' competencies, with significant correlations to the generating competencies across all ACGME domains despite a non-deterministic pipeline. It estimated case difficulty by competency and showed stable rater detection (sensitivity) and criteria (severity/leniency thresholds) across AI raters using identical models/prompts but different seeds. We also propose a staged "safety blueprint" for deploying AI tools with learners, tied to entrustment-based validation milestones.
Conclusions: Combining a purpose-built virtual patient platform with a principled psychometric model enables robust, interpretable, generalizable competency estimates and supports validation of AI-assisted assessment prior to use with human learners. 

---
# Diffusion Model-based Reinforcement Learning for Version Age of Information Scheduling: Average and Tail-Risk-Sensitive Control 

**Authors**: Haoyuan Pan, Sizhao Chen, Zhaorui Wang, Tse-Tin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2601.18069)  

**Abstract**: Ensuring timely and semantically accurate information delivery is critical in real-time wireless systems. While Age of Information (AoI) quantifies temporal freshness, Version Age of Information (VAoI) captures semantic staleness by accounting for version evolution between transmitters and receivers. Existing VAoI scheduling approaches primarily focus on minimizing average VAoI, overlooking rare but severe staleness events that can compromise reliability under stochastic packet arrivals and unreliable channels. This paper investigates both average-oriented and tail-risk-sensitive VAoI scheduling in a multi-user status update system with long-term transmission cost constraints. We first formulate the average VAoI minimization problem as a constrained Markov decision process and introduce a deep diffusion-based Soft Actor-Critic (D2SAC) algorithm. By generating actions through a diffusion-based denoising process, D2SAC enhances policy expressiveness and establishes a strong baseline for mean performance. Building on this foundation, we put forth RS-D3SAC, a risk-sensitive deep distributional diffusion-based Soft Actor-Critic algorithm. RS-D3SAC integrates a diffusion-based actor with a quantile-based distributional critic, explicitly modeling the full VAoI return distribution. This enables principled tail-risk optimization via Conditional Value-at-Risk (CVaR) while satisfying long-term transmission cost constraints. Extensive simulations show that, while D2SAC reduces average VAoI, RS-D3SAC consistently achieves substantial reductions in CVaR without sacrificing mean performance. The dominant gain in tail-risk reduction stems from the distributional critic, with the diffusion-based actor providing complementary refinement to stabilize and enrich policy decisions, highlighting their effectiveness for robust and risk-aware VAoI scheduling in multi-user wireless systems. 

---
# Resonant Sparse Geometry Networks 

**Authors**: Hasi Hays  

**Link**: [PDF](https://arxiv.org/pdf/2601.18064)  

**Abstract**: We introduce Resonant Sparse Geometry Networks (RSGN), a brain-inspired architecture with self-organizing sparse
hierarchical input-dependent connectivity. Unlike Transformer architectures that employ dense attention mechanisms with
O(n^2) computational complexity, RSGN embeds computational nodes in learned hyperbolic space where connection strength
decays with geodesic distance, achieving dynamic sparsity that adapts to each input. The architecture operates on two
distinct timescales: fast differentiable activation propagation optimized through gradient descent, and slow
Hebbian-inspired structural learning for connectivity adaptation through local correlation rules. We provide rigorous
mathematical analysis demonstrating that RSGN achieves O(n*k) computational complexity, where k << n represents the average
active neighborhood size. Experimental evaluation on hierarchical classification and long-range dependency tasks
demonstrates that RSGN achieves 96.5% accuracy on long-range dependency tasks while using approximately 15x fewer
parameters than standard Transformers. On challenging hierarchical classification with 20 classes, RSGN achieves 23.8%
accuracy (compared to 5% random baseline) with only 41,672 parameters, nearly 10x fewer than the Transformer baselines
which require 403,348 parameters to achieve 30.1% accuracy. Our ablation studies confirm the contribution of each architectural
component, with Hebbian learning providing consistent improvements. These results suggest that brain-inspired principles
of sparse, geometrically-organized computation offer a promising direction toward more efficient and biologically plausible
neural architectures. 

---
# Addressing LLM Diversity by Infusing Random Concepts 

**Authors**: Pulin Agrawal, Prasoon Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2601.18053)  

**Abstract**: Large language models (LLMs) are known to produce outputs with limited diversity. In this work, we study whether infusing random concepts in the prompts can improve the diversity of the generated outputs. To benchmark the approach, we design a systematic evaluation protocol which involves prompting an LLM with questions of the form "Name 10 Hollywood actors", and analyzing diversity measures of the resulting LLM outputs. Our experiments on multiple LLMs show that prepending random words/sentences unrelated to the prompt result in greater diversity in the outputs of LLMs. We believe that this promising result and the evaluation protocol opens up interesting avenues for future work, such as how infusing randomness into LLMs could be applied to other domains. Further, the evaluation protocol could also inspire research into benchmarking LLM diversity more systematically. 

---
# Leveraging Persistence Image to Enhance Robustness and Performance in Curvilinear Structure Segmentation 

**Authors**: Zhuangzhi Gao, Feixiang Zhou, He Zhao, Xiuju Chen, Xiaoxin Li, Qinkai Yu, Yitian Zhao, Alena Shantsila, Gregory Y. H. Lip, Eduard Shantsila, Yalin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.18045)  

**Abstract**: Segmenting curvilinear structures in medical images is essential for analyzing morphological patterns in clinical applications. Integrating topological properties, such as connectivity, improves segmentation accuracy and consistency. However, extracting and embedding such properties - especially from Persistence Diagrams (PD) - is challenging due to their non-differentiability and computational cost. Existing approaches mostly encode topology through handcrafted loss functions, which generalize poorly across tasks. In this paper, we propose PIs-Regressor, a simple yet effective module that learns persistence image (PI) - finite, differentiable representations of topological features - directly from data. Together with Topology SegNet, which fuses these features in both downsampling and upsampling stages, our framework integrates topology into the network architecture itself rather than auxiliary losses. Unlike existing methods that depend heavily on handcrafted loss functions, our approach directly incorporates topological information into the network structure, leading to more robust segmentation. Our design is flexible and can be seamlessly combined with other topology-based methods to further enhance segmentation performance. Experimental results show that integrating topological features enhances model robustness, effectively handling challenges like overexposure and blurring in medical imaging. Our approach on three curvilinear benchmarks demonstrate state-of-the-art performance in both pixel-level accuracy and topological fidelity. 

---
# SpatialEmb: Extract and Encode Spatial Information for 1-Stage Multi-channel Multi-speaker ASR on Arbitrary Microphone Arrays 

**Authors**: Yiwen Shao, Yong Xu, Sanjeev Khudanpur, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.18037)  

**Abstract**: Spatial information is a critical clue for multi-channel multi-speaker target speech recognition. Most state-of-the-art multi-channel Automatic Speech Recognition (ASR) systems extract spatial features only during the speech separation stage, followed by standard single-channel ASR on the separated speech. This approach results in an inefficient, lengthy pipeline and sub-optimal ASR performance due to the accumulated errors from preprocessing modules. Furthermore, most spatial feature extraction methods depend on the knowledge of speaker positions and microphone topology, making the systems reliant on specific settings and challenging to adapt to new equipment. In this work, we propose a solution to these issues with a lightweight embedding module named SpatialEmb, which extracts and encodes spatial information directly for the ASR model, supporting both fixed and arbitrary microphone topology. We conduct comprehensive experiments on AliMeeting, a real meeting corpus, to determine the optimal model design for SpatialEmb in terms of both performance and efficiency. Our best model trained with 105 hours Train-Ali-far achieves 17.04% and 20.32% character error rates (CER) on the Eval and Test sets, establishing a new state-of-the-art result with the same training data. 

---
# An Experimental Comparison of Cognitive Forcing Functions for Execution Plans in AI-Assisted Writing: Effects On Trust, Overreliance, and Perceived Critical Thinking 

**Authors**: Ahana Ghosh, Advait Sarkar, Siân Lindley, Christian Poelitz  

**Link**: [PDF](https://arxiv.org/pdf/2601.18033)  

**Abstract**: Generative AI (GenAI) tools improve productivity in knowledge workflows such as writing, but also risk overreliance and reduced critical thinking. Cognitive forcing functions (CFFs) mitigate these risks by requiring active engagement with AI output. As GenAI workflows grow more complex, systems increasingly present execution plans for user review. However, these plans are themselves AI-generated and prone to overreliance, and the effectiveness of applying CFFs to AI plans remains underexplored. We conduct a controlled experiment in which participants completed AI-assisted writing tasks while reviewing AI-generated plans under four CFF conditions: Assumption (argument analysis), WhatIf (hypothesis testing), Both, and a no-CFF control. A follow-up think-aloud and interview study qualitatively compared these conditions. Results show that the Assumption CFF most effectively reduced overreliance without increasing cognitive load, while participants perceived the WhatIf CFF as most helpful. These findings highlight the value of plan-focused CFFs for supporting critical reflection in GenAI-assisted knowledge work. 

---
# A System for Name and Address Parsing with Large Language Models 

**Authors**: Adeeba Tarannum, Muzakkiruddin Ahmed Mohammed, Mert Can Cakmak, Shames Al Mandalawi, John Talburt  

**Link**: [PDF](https://arxiv.org/pdf/2601.18014)  

**Abstract**: Reliable transformation of unstructured person and address text into structured data remains a key challenge in large-scale information systems. Traditional rule-based and probabilistic approaches perform well on clean inputs but fail under noisy or multilingual conditions, while neural and large language models (LLMs) often lack deterministic control and reproducibility. This paper introduces a prompt-driven, validation-centered framework that converts free-text records into a consistent 17-field schema without fine-tuning. The method integrates input normalisation, structured prompting, constrained decoding, and strict rule-based validation under fixed experimental settings to ensure reproducibility. Evaluations on heterogeneous real-world address data show high field-level accuracy, strong schema adherence, and stable confidence calibration. The results demonstrate that combining deterministic validation with generative prompting provides a robust, interpretable, and scalable solution for structured information extraction, offering a practical alternative to training-heavy or domain-specific models. 

---
# Evaluating Semantic and Syntactic Understanding in Large Language Models for Payroll Systems 

**Authors**: Hendrika Maclean, Mert Can Cakmak, Muzakkiruddin Ahmed Mohammed, Shames Al Mandalawi, John Talburt  

**Link**: [PDF](https://arxiv.org/pdf/2601.18012)  

**Abstract**: Large language models are now used daily for writing, search, and analysis, and their natural language understanding continues to improve. However, they remain unreliable on exact numerical calculation and on producing outputs that are straightforward to audit. We study synthetic payroll system as a focused, high-stakes example and evaluate whether models can understand a payroll schema, apply rules in the right order, and deliver cent-accurate results. Our experiments span a tiered dataset from basic to complex cases, a spectrum of prompts from minimal baselines to schema-guided and reasoning variants, and multiple model families including GPT, Claude, Perplexity, Grok and Gemini. Results indicate clear regimes where careful prompting is sufficient and regimes where explicit computation is required. The work offers a compact, reproducible framework and practical guidance for deploying LLMs in settings that demand both accuracy and assurance. 

---
# Coding-Enforced Resilient and Secure Aggregation for Hierarchical Federated Learning 

**Authors**: Shudi Weng, Ming Xiao, Mikael Skoglund  

**Link**: [PDF](https://arxiv.org/pdf/2601.17995)  

**Abstract**: Hierarchical federated learning (HFL) has emerged as an effective paradigm to enhance link quality between clients and the server. However, ensuring model accuracy while preserving privacy under unreliable communication remains a key challenge in HFL, as the coordination among privacy noise can be randomly disrupted. To address this limitation, we propose a robust hierarchical secure aggregation scheme, termed H-SecCoGC, which integrates coding strategies to enforce structured aggregation. The proposed scheme not only ensures accurate global model construction under varying levels of privacy, but also avoids the partial participation issue, thereby significantly improving robustness, privacy preservation, and learning efficiency. Both theoretical analyses and experimental results demonstrate the superiority of our scheme under unreliable communication across arbitrarily strong privacy guarantees 

---
# AI-based approach to burnout identification from textual data 

**Authors**: Marina Zavertiaeva, Petr Parshakov, Mikhail Usanin, Aleksei Smirnov, Sofia Paklina, Anastasiia Kibardina  

**Link**: [PDF](https://arxiv.org/pdf/2601.17993)  

**Abstract**: This study introduces an AI-based methodology that utilizes natural language processing (NLP) to detect burnout from textual data. The approach relies on a RuBERT model originally trained for sentiment analysis and subsequently fine-tuned for burnout detection using two data sources: synthetic sentences generated with ChatGPT and user comments collected from Russian YouTube videos about burnout. The resulting model assigns a burnout probability to input texts and can be applied to process large volumes of written communication for monitoring burnout-related language signals in high-stress work environments. 

---
# SD-E$^2$: Semantic Exploration for Reasoning Under Token Budgets 

**Authors**: Kshitij Mishra, Nils Lukas, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17982)  

**Abstract**: Small language models (SLMs) struggle with complex reasoning because exploration is expensive under tight compute budgets. We introduce Semantic Diversity-Exploration-Exploitation (SD-E$^2$), a reinforcement learning framework that makes exploration explicit by optimizing semantic diversity in generated reasoning trajectories. Using a frozen sentence-embedding model, SD-E$^2$ assigns a diversity reward that captures (i) the coverage of semantically distinct solution strategies and (ii) their average pairwise dissimilarity in embedding space, rather than surface-form novelty. This diversity reward is combined with outcome correctness and solution efficiency in a z-score-normalized multi-objective objective that stabilizes training. On GSM8K, SD-E$^2$ surpasses the base Qwen2.5-3B-Instruct and strong GRPO baselines (GRPO-CFL and GRPO-CFEE) by +27.4, +5.2, and +1.5 percentage points, respectively, while discovering on average 9.8 semantically distinct strategies per question. We further improve MedMCQA to 49.64% versus 38.37% for the base model and show gains on the harder AIME benchmark (1983-2025), reaching 13.28% versus 6.74% for the base. These results indicate that rewarding semantic novelty yields a more compute-efficient exploration-exploitation signal for training reasoning-capable SLMs. By introducing cognitive adaptation-adjusting the reasoning process structure rather than per-token computation-SD-E$^2$ offers a complementary path to efficiency gains in resource-constrained models. 

---
# A Monosemantic Attribution Framework for Stable Interpretability in Clinical Neuroscience Large Language Models 

**Authors**: Michail Mamalakis, Tiago Azevedo, Cristian Cosentino, Chiara D'Ercoli, Subati Abulikemu, Zhongtian Sun, Richard Bethlehem, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2601.17952)  

**Abstract**: Interpretability remains a key challenge for deploying large language models (LLMs) in clinical settings such as Alzheimer's disease progression diagnosis, where early and trustworthy predictions are essential. Existing attribution methods exhibit high inter-method variability and unstable explanations due to the polysemantic nature of LLM representations, while mechanistic interpretability approaches lack direct alignment with model inputs and outputs and do not provide explicit importance scores. We introduce a unified interpretability framework that integrates attributional and mechanistic perspectives through monosemantic feature extraction. By constructing a monosemantic embedding space at the level of an LLM layer and optimizing the framework to explicitly reduce inter-method variability, our approach produces stable input-level importance scores and highlights salient features via a decompressed representation of the layer of interest, advancing the safe and trustworthy application of LLMs in cognitive health and neurodegenerative disease. 

---
# Credit Fairness: Online Fairness In Shared Resource Pools 

**Authors**: Seyed Majid Zahedi, Rupert Freeman  

**Link**: [PDF](https://arxiv.org/pdf/2601.17944)  

**Abstract**: We consider a setting in which a group of agents share resources that must be allocated among them in each discrete time period. Agents have time-varying demands and derive constant marginal utility from each unit of resource received up to their demand, with zero utility for any additional resources. In this setting, it is known that independently maximizing the minimum utility in each round satisfies sharing incentives (agents weakly prefer participating in the mechanism to not participating), strategyproofness (agents have no incentive to misreport their demands), and Pareto efficiency (Freeman et al. 2018). However, recent work (Vuppalapati et al. 2023) has shown that this max-min mechanism can lead to large disparities in the total resources received by agents, even when they have the same average demand. In this paper, we introduce credit fairness, a strengthening of sharing incentives that ensures agents who lend resources in early rounds are able to recoup them in later rounds. Credit fairness can be achieved in conjunction with either Pareto efficiency or strategyproofness, but not both. We propose a mechanism that is credit fair and Pareto efficient, and we evaluate its performance in a computational resource-sharing setting. 

---
# From Specialist to Generalist: Unlocking SAM's Learning Potential on Unlabeled Medical Images 

**Authors**: Vi Vu, Thanh-Huy Nguyen, Tien-Thinh Nguyen, Ba-Thinh Lam, Hoang-Thien Nguyen, Tianyang Wang, Xingjian Li, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17934)  

**Abstract**: Foundation models like the Segment Anything Model (SAM) show strong generalization, yet adapting them to medical images remains difficult due to domain shift, scarce labels, and the inability of Parameter-Efficient Fine-Tuning (PEFT) to exploit unlabeled data. While conventional models like U-Net excel in semi-supervised medical learning, their potential to assist a PEFT SAM has been largely overlooked. We introduce SC-SAM, a specialist-generalist framework where U-Net provides point-based prompts and pseudo-labels to guide SAM's adaptation, while SAM serves as a powerful generalist supervisor to regularize U-Net. This reciprocal guidance forms a bidirectional co-training loop that allows both models to effectively exploit the unlabeled data. Across prostate MRI and polyp segmentation benchmarks, our method achieves state-of-the-art results, outperforming other existing semi-supervised SAM variants and even medical foundation models like MedSAM, highlighting the value of specialist-generalist cooperation for label-efficient medical image segmentation. Our code is available at this https URL. 

---
# treaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding 

**Authors**: Zhongyu Xiao, Zhiwei Hao, Jianyuan Guo, Yong Luo, Jia Liu, Jie Xu, Han Hu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17917)  

**Abstract**: Diffusion Large Language Models (dLLMs) offer a compelling paradigm for natural language generation, leveraging parallel decoding and bidirectional attention to achieve superior global coherence compared to autoregressive models. While recent works have accelerated inference via KV cache reuse or heuristic decoding, they overlook the intrinsic inefficiencies within the block-wise diffusion process. Specifically, they suffer from spatial redundancy by modeling informative-sparse suffix regions uniformly and temporal inefficiency by applying fixed denoising schedules across all the decoding process. To address this, we propose Streaming-dLLM, a training-free framework that streamlines inference across both spatial and temporal dimensions. Spatially, we introduce attenuation guided suffix modeling to approximate the full context by pruning redundant mask tokens. Temporally, we employ a dynamic confidence aware strategy with an early exit mechanism, allowing the model to skip unnecessary iterations for converged tokens. Extensive experiments show that Streaming-dLLM achieves up to 68.2X speedup while maintaining generation quality, highlighting its effectiveness in diffusion decoding. The code is available at this https URL. 

---
# Causal Pre-training Under the Fairness Lens: An Empirical Study of TabPFN 

**Authors**: Qinyi Liu, Mohammad Khalil, Naman Goel  

**Link**: [PDF](https://arxiv.org/pdf/2601.17912)  

**Abstract**: Foundation models for tabular data, such as the Tabular Prior-data Fitted Network (TabPFN), are pre-trained on a massive number of synthetic datasets generated by structural causal models (SCM). They leverage in-context learning to offer high predictive accuracy in real-world tasks. However, the fairness properties of these foundational models, which incorporate ideas from causal reasoning during pre-training, have not yet been explored in sufficient depth. In this work, we conduct a comprehensive empirical evaluation of TabPFN and its fine-tuned variants, assessing predictive performance, fairness, and robustness across varying dataset sizes and distributional shifts. Our results reveal that while TabPFN achieves stronger predictive accuracy compared to baselines and exhibits robustness to spurious correlations, improvements in fairness are moderate and inconsistent, particularly under missing-not-at-random (MNAR) covariate shifts. These findings suggest that the causal pre-training in TabPFN is helpful but insufficient for algorithmic fairness, highlighting implications for deploying such models in practice and the need for further fairness interventions. 

---
# Feature-Space Generative Models for One-Shot Class-Incremental Learning 

**Authors**: Jack Foster, Kirill Paramonov, Mete Ozay, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2601.17905)  

**Abstract**: Few-shot class-incremental learning (FSCIL) is a paradigm where a model, initially trained on a dataset of base classes, must adapt to an expanding problem space by recognizing novel classes with limited data. We focus on the challenging FSCIL setup where a model receives only a single sample (1-shot) for each novel class and no further training or model alterations are allowed after the base training phase. This makes generalization to novel classes particularly difficult. We propose a novel approach predicated on the hypothesis that base and novel class embeddings have structural similarity. We map the original embedding space into a residual space by subtracting the class prototype (i.e., the average class embedding) of input samples. Then, we leverage generative modeling with VAE or diffusion models to learn the multi-modal distribution of residuals over the base classes, and we use this as a valuable structural prior to improve recognition of novel classes. Our approach, Gen1S, consistently improves novel class recognition over the state of the art across multiple benchmarks and backbone architectures. 

---
# Artificial Intelligence and Intellectual Property Rights: Comparative Transnational Policy Analysis 

**Authors**: Sahibpreet Singh, Manjit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.17892)  

**Abstract**: Artificial intelligence's rapid integration with intellectual property rights necessitates assessment of its impact on trade secrets, copyrights and patents. This study addresses lacunae in existing laws where India lacks AI-specific provisions, creating doctrinal inconsistencies and enforcement inefficacies. Global discourse on AI-IPR protections remains nascent. The research identifies gaps in Indian IP laws' adaptability to AI-generated outputs: trade secret protection is inadequate against AI threats; standardized inventorship criteria are absent. Employing doctrinal and comparative methodology, it scrutinizes legislative texts, judicial precedents and policy instruments across India, US, UK and EU. Preliminary findings reveal shortcomings: India's contract law creates fragmented trade secret regime; Section 3(k) of Indian Patents Act blocks AI invention patenting; copyright varies in authorship attribution. The study proposes harmonized legal taxonomy accommodating AI's role while preserving innovation incentives. India's National AI Strategy (2024) shows progress but legislative clarity is imperative. This contributes to global discourse with AI-specific IP protections ensuring resilience and equitable innovation. Promising results underscore recalibrating India's IP jurisprudence for global alignment. 

---
# PEAfowl: Perception-Enhanced Multi-View Vision-Language-Action for Bimanual Manipulation 

**Authors**: Qingyu Fan, Zhaoxiang Li, Yi Lu, Wang Chen, Qiu Shen, Xiao-xiao Long, Yinghao Cai, Tao Lu, Shuo Wang, Xun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17885)  

**Abstract**: Bimanual manipulation in cluttered scenes requires policies that remain stable under occlusions, viewpoint and scene variations. Existing vision-language-action models often fail to generalize because (i) multi-view features are fused via view-agnostic token concatenation, yielding weak 3D-consistent spatial understanding, and (ii) language is injected as global conditioning, resulting in coarse instruction grounding.
In this paper, we introduce PEAfowl, a perception-enhanced multi-view VLA policy for bimanual manipulation. For spatial reasoning, PEAfowl predicts per-token depth distributions, performs differentiable 3D lifting, and aggregates local cross-view neighbors to form geometrically grounded, cross-view consistent representations. For instruction grounding, we propose to replace global conditioning with a Perceiver-style text-aware readout over frozen CLIP visual features, enabling iterative evidence accumulation. To overcome noisy and incomplete commodity depth without adding inference overhead, we apply training-only depth distillation from a pretrained depth teacher to supervise the depth-distribution head, providing perception front-end with geometry-aware priors.
On RoboTwin 2.0 under domain-randomized setting, PEAfowl improves the strongest baseline by 23.0 pp in success rate, and real-robot experiments further demonstrate reliable sim-to-real transfer and consistent improvements from depth distillation.
Project website: this https URL. 

---
# Self-Manager: Parallel Agent Loop for Long-form Deep Research 

**Authors**: Yilong Xu, Zhi Zheng, Xiang Long, Yujun Cai, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17879)  

**Abstract**: Long-form deep research requires multi-faceted investigations over extended horizons to get a comprehensive report. When handling such complex tasks, existing agents manage context at the subtask level to overcome linear context accumulation and information loss. However, they still adhere to a single context window and sequential execution paradigm, which results in mutual interference and blocking behavior, restricting scalability and adaptability. To address this issue, this paper introduces Self-Manager, a parallel agent loop that enables asynchronous and concurrent execution. The main thread can create multiple subthreads, each with its own isolated context, and manage them iteratively through Thread Control Blocks, allowing for more focused and flexible parallel agent execution. To assess its effectiveness, we benchmark Self-Manager on DeepResearch Bench, where it consistently outperforms existing single-agent loop baselines across all metrics. Furthermore, we conduct extensive analytical experiments to demonstrate the necessity of Self-Manager's design choices, as well as its advantages in contextual capacity, efficiency, and generalization. 

---
# Comparative Algorithmic Governance of Public Health Instruments across India, EU, US and LMICs 

**Authors**: Sahibpreet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2601.17877)  

**Abstract**: The study investigates the juridico-technological architecture of international public health instruments, focusing on their implementation across India, the European Union, the United States and low- and middle-income countries (LMICs), particularly in Sub-Saharan Africa. It addresses a research lacuna: the insufficient harmonisation between normative health law and algorithmic public health infrastructures in resource-constrained jurisdictions. The principal objective is to assess how artificial intelligence augments implementation of instruments grounded in IHR 2005 and the WHO FCTC while identifying doctrinal and infrastructural bottlenecks. Using comparative doctrinal analysis and legal-normative mapping, the study triangulates legislative instruments, WHO monitoring frameworks, AI systems including BlueDot, Aarogya Setu and EIOS, and compliance metrics. Preliminary results show that AI has improved early detection, surveillance precision and responsiveness in high-capacity jurisdictions, whereas LMICs face infrastructural deficits, data privacy gaps and fragmented legal scaffolding. The findings highlight the relevance of the EU Artificial Intelligence Act and GDPR as regulatory prototypes for health-oriented algorithmic governance and contrast them with embryonic AI integration and limited internet penetration in many LMICs. The study argues for embedding AI within a rights-compliant, supranationally coordinated regulatory framework to secure equitable health outcomes and stronger compliance. It proposes a model for algorithmic treaty-making inspired by FCTC architecture and calls for WHO-led compliance mechanisms modelled on the WTO Dispute Settlement Body to enhance pandemic preparedness, surveillance equity and transnational governance resilience. 

---
# VidLaDA: Bidirectional Diffusion Large Language Models for Efficient Video Understanding 

**Authors**: Zhihao He, Tieyuan Chen, Kangyu Wang, Ziran Qin, Yang Shao, Chaofan Gan, Shijie Li, Zuxuan Wu, Weiyao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.17868)  

**Abstract**: Standard Autoregressive Video LLMs inevitably suffer from causal masking biases that hinder global spatiotemporal modeling, leading to suboptimal understanding efficiency. We propose VidLaDA, a Video LLM based on Diffusion Language Model utilizing bidirectional attention to capture bidirectional dependencies. To further tackle the inference bottleneck of diffusion decoding on massive video tokens, we introduce MARS-Cache. This framework accelerates inference by combining asynchronous visual cache refreshing with frame-wise chunk attention, effectively pruning redundancy while preserving global connectivity via anchor tokens. Extensive experiments show VidLaDA outperforms diffusion baselines and rivals state-of-the-art autoregressive models (e.g., Qwen2.5-VL and LLaVA-Video), with MARS-Cache delivering over 12x speedup without compromising reasoning accuracy. Code and checkpoints are open-sourced at this https URL. 

---
# MergeMix: Optimizing Mid-Training Data Mixtures via Learnable Model Merging 

**Authors**: Jiapeng Wang, Changxin Tian, Kunlong Chen, Ziqi Liu, Jiaxin Mao, Wayne Xin Zhao, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17858)  

**Abstract**: Optimizing data mixtures is essential for unlocking the full potential of large language models (LLMs), yet identifying the optimal composition remains computationally prohibitive due to reliance on heuristic trials or expensive proxy training. To address this, we introduce \textbf{MergeMix}, a novel approach that efficiently determines optimal data mixing ratios by repurposing model merging weights as a high-fidelity, low-cost performance proxy. By training domain-specific experts on minimal tokens and optimizing their merging weights against downstream benchmarks, MergeMix effectively optimizes the performance of data mixtures without incurring the cost of full-scale training. Extensive experiments on models with 8B and 16B parameters validate that MergeMix achieves performance comparable to or surpassing exhaustive manual tuning while drastically reducing search costs. Furthermore, MergeMix exhibits high rank consistency (Spearman $\rho > 0.9$) and strong cross-scale transferability, offering a scalable, automated solution for data mixture optimization. 

---
# RAICL: Retrieval-Augmented In-Context Learning for Vision-Language-Model Based EEG Seizure Detection 

**Authors**: Siyang Li, Zhuoya Wang, Xiyan Gui, Xiaoqing Chen, Ziwei Wang, Yaozhi Wen, Dongrui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17844)  

**Abstract**: Electroencephalogram (EEG) decoding is a critical component of medical diagnostics, rehabilitation engineering, and brain-computer interfaces. However, contemporary decoding methodologies remain heavily dependent on task-specific datasets to train specialized neural network architectures. Consequently, limited data availability impedes the development of generalizable large brain decoding models. In this work, we propose a paradigm shift from conventional signal-based decoding by leveraging large-scale vision-language models (VLMs) to analyze EEG waveform plots. By converting multivariate EEG signals into stacked waveform images and integrating neuroscience domain expertise into textual prompts, we demonstrate that foundational VLMs can effectively differentiate between different patterns in the human brain. To address the inherent non-stationarity of EEG signals, we introduce a Retrieval-Augmented In-Context Learning (RAICL) approach, which dynamically selects the most representative and relevant few-shot examples to condition the autoregressive outputs of the VLM. Experiments on EEG-based seizure detection indicate that state-of-the-art VLMs under RAICL achieved better or comparable performance with traditional time series based approaches. These findings suggest a new direction in physiological signal processing that effectively bridges the modalities of vision, language, and neural activities. Furthermore, the utilization of off-the-shelf VLMs, without the need for retraining or downstream architecture construction, offers a readily deployable solution for clinical applications. 

---
# Linguistic and Argument Diversity in Synthetic Data for Function-Calling Agents 

**Authors**: Dan Greenstein, Zohar Karnin, Chen Amiraz, Oren Somekh  

**Link**: [PDF](https://arxiv.org/pdf/2601.17829)  

**Abstract**: The construction of function calling agents has emerged as a promising avenue for extending model capabilities. A major challenge for this task is obtaining high quality diverse data for training. Prior work emphasizes diversity in functions, invocation patterns, and interaction turns, yet linguistic diversity of requests and coverage of arguments (e.g., \texttt{city\_name}, \texttt{stock\_ticker}) remain underexplored. We propose a method that generates synthetic datasets via optimizing general-purpose diversity metrics across both queries and arguments, without relying on hand-crafted rules or taxonomies, making it robust to different usecases. We demonstrate the effectiveness of our technique via both intrinsic and extrinsic testing, comparing it to SoTA data generation methods. We show a superiority over baselines in terms of diversity, while keeping comparable correctness. Additionally, when used as a training set, the model resulting from our dataset exhibits superior performance compared to analogous models based on the baseline data generation methods in out-of-distribution performance. In particular, we achieve an $7.4\%$ increase in accuracy on the BFCL benchmark compared to similar counterparts. 

---
# DIETA: A Decoder-only transformer-based model for Italian-English machine TrAnslation 

**Authors**: Pranav Kasela, Marco Braga, Alessandro Ghiotto, Andrea Pilzer, Marco Viviani, Alessandro Raganato  

**Link**: [PDF](https://arxiv.org/pdf/2601.17823)  

**Abstract**: In this paper, we present DIETA, a small, decoder-only Transformer model with 0.5 billion parameters, specifically designed and trained for Italian-English machine translation. We collect and curate a large parallel corpus consisting of approximately 207 million Italian-English sentence pairs across diverse domains, including parliamentary proceedings, legal texts, web-crawled content, subtitles, news, literature and 352 million back-translated data using pretrained models. Additionally, we create and release a new small-scale evaluation set, consisting of 450 sentences, based on 2025 WikiNews articles, enabling assessment of translation quality on contemporary text. Comprehensive evaluations show that DIETA achieves competitive performance on multiple Italian-English benchmarks, consistently ranking in the second quartile of a 32-system leaderboard and outperforming most other sub-3B models on four out of five test suites. The training script, trained models, curated corpus, and newly introduced evaluation set are made publicly available, facilitating further research and development in specialized Italian-English machine translation. this https URL 

---
# Shortcut Learning in Binary Classifier Black Boxes: Applications to Voice Anti-Spoofing and Biometrics 

**Authors**: Md Sahidullah, Hye-jin Shim, Rosa Gonzalez Hautamäki, Tomi H. Kinnunen  

**Link**: [PDF](https://arxiv.org/pdf/2601.17782)  

**Abstract**: The widespread adoption of deep-learning models in data-driven applications has drawn attention to the potential risks associated with biased datasets and models. Neglected or hidden biases within datasets and models can lead to unexpected results. This study addresses the challenges of dataset bias and explores ``shortcut learning'' or ``Clever Hans effect'' in binary classifiers. We propose a novel framework for analyzing the black-box classifiers and for examining the impact of both training and test data on classifier scores. Our framework incorporates intervention and observational perspectives, employing a linear mixed-effects model for post-hoc analysis. By evaluating classifier performance beyond error rates, we aim to provide insights into biased datasets and offer a comprehensive understanding of their influence on classifier behavior. The effectiveness of our approach is demonstrated through experiments on audio anti-spoofing and speaker verification tasks using both statistical models and deep neural networks. The insights gained from this study have broader implications for tackling biases in other domains and advancing the field of explainable artificial intelligence. 

---
# DPI: Exploiting Parameter Heterogeneity for Interference-Free Fine-Tuning 

**Authors**: Xiaoyu Liu, Xiaoyu Guan, Di Liang, Xianjie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17777)  

**Abstract**: Supervised fine-tuning (SFT) is a crucial step for adapting large language models (LLMs) to downstream tasks. However, conflicting objectives across heterogeneous SFT tasks often induce the "seesaw effect": optimizing for one task may degrade performance on others, particularly when model parameters are updated indiscriminately. In this paper, we propose a principled approach to disentangle and isolate task-specific parameter regions, motivated by the hypothesis that parameter heterogeneity underlies cross-task interference. Specifically, we first independently fine-tune LLMs on diverse SFT tasks and identify each task's core parameter region as the subset of parameters exhibiting the largest updates. Tasks with highly overlapping core parameter regions are merged for joint training, while disjoint tasks are organized into different stages. During multi-stage SFT, core parameters acquired in prior tasks are frozen, thereby preventing overwriting by subsequent tasks. To verify the effectiveness of our method, we conducted intensive experiments on multiple public datasets. The results showed that our dynamic parameter isolation strategy consistently reduced data conflicts and achieved consistent performance improvements compared to multi-stage and multi-task tuning baselines. 

---
# Context-Aware Iterative Token Detection and Masked Transmission for Wireless Token Communication 

**Authors**: Junyong Shin, Joohyuk Park, Jihong Park, Jinho Choi, Yo-Seb Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2601.17770)  

**Abstract**: The success of large-scale language models has established tokens as compact and meaningful units for natural-language representation, which motivates token communication over wireless channels, where tokens are considered fundamental units for wireless transmission. We propose a context-aware token communication framework that uses a pretrained masked language model (MLM) as a shared contextual probability model between the transmitter (Tx) and receiver (Rx). At Rx, we develop an iterative token detection method that jointly exploits MLM-guided contextual priors and channel observations based on a Bayesian perspective. At Tx, we additionally introduce a context-aware masking strategy which skips highly predictable token transmission to reduce transmission rate. Simulation results demonstrate that the proposed framework substantially improves reconstructed sentence quality and supports effective rate adaptation under various channel conditions. 

---
# LLM-42: Enabling Determinism in LLM Inference with Verified Speculation 

**Authors**: Raja Gond, Aditya K Kamath, Arkaprava Basu, Ramachandran Ramjee, Ashish Panwar  

**Link**: [PDF](https://arxiv.org/pdf/2601.17768)  

**Abstract**: In LLM inference, the same prompt may yield different outputs across different runs. At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batching and GPU kernels whose reduction orders vary with batch size. A straightforward way to eliminate non-determinism is to disable dynamic batching during inference, but doing so severely degrades throughput. Another approach is to make kernels batch-invariant; however, this tightly couples determinism to kernel design, requiring new implementations. This coupling also imposes fixed runtime overheads, regardless of how much of the workload actually requires determinism.
Inspired by ideas from speculative decoding, we present LLM-42, a scheduling-based approach to enable determinism in LLM inference. Our key observation is that if a sequence is in a consistent state, the next emitted token is likely to be consistent even with dynamic batching. Moreover, most GPU kernels use shape-consistent reductions. Leveraging these insights, LLM-42 decodes tokens using a non-deterministic fast path and enforces determinism via a lightweight verify-rollback loop. The verifier replays candidate tokens under a fixed-shape reduction schedule, commits those that are guaranteed to be consistent across runs, and rolls back those violating determinism. LLM-42 mostly re-uses existing kernels unchanged and incurs overhead only in proportion to the traffic that requires determinism. 

---
# Cross-Lingual Probing and Community-Grounded Analysis of Gender Bias in Low-Resource Bengali 

**Authors**: Md Asgor Hossain Reaj, Rajan Das Gupta, Jui Saha Pritha, Abdullah Al Noman, Abir Ahmed, Golam Md Mohiuddin, Tze Hui Liew  

**Link**: [PDF](https://arxiv.org/pdf/2601.17764)  

**Abstract**: Large Language Models (LLMs) have achieved significant success in recent years; yet, issues of intrinsic gender bias persist, especially in non-English languages. Although current research mostly emphasizes English, the linguistic and cultural biases inherent in Global South languages, like Bengali, are little examined. This research seeks to examine the characteristics and magnitude of gender bias in Bengali, evaluating the efficacy of current approaches in identifying and alleviating bias. We use several methods to extract gender-biased utterances, including lexicon-based mining, computational classification models, translation-based comparison analysis, and GPT-based bias creation. Our research indicates that the straight application of English-centric bias detection frameworks to Bengali is severely constrained by language disparities and socio-cultural factors that impact implicit biases. To tackle these difficulties, we executed two field investigations inside rural and low-income areas, gathering authentic insights on gender bias. The findings demonstrate that gender bias in Bengali presents distinct characteristics relative to English, requiring a more localized and context-sensitive methodology. Additionally, our research emphasizes the need of integrating community-driven research approaches to identify culturally relevant biases often neglected by automated systems. Our research enhances the ongoing discussion around gender bias in AI by illustrating the need to create linguistic tools specifically designed for underrepresented languages. This study establishes a foundation for further investigations into bias reduction in Bengali and other Indic languages, promoting the development of more inclusive and fair NLP systems. 

---
# AR-Omni: A Unified Autoregressive Model for Any-to-Any Generation 

**Authors**: Dongjie Cheng, Ruifeng Yuan, Yongqi Li, Runyang You, Wenjie Wang, Liqiang Nie, Lei Zhang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.17761)  

**Abstract**: Real-world perception and interaction are inherently multimodal, encompassing not only language but also vision and speech, which motivates the development of "Omni" MLLMs that support both multimodal inputs and multimodal outputs. While a sequence of omni MLLMs has emerged, most existing systems still rely on additional expert components to achieve multimodal generation, limiting the simplicity of unified training and inference. Autoregressive (AR) modeling, with a single token stream, a single next-token objective, and a single decoder, is an elegant and scalable foundation in the text domain. Motivated by this, we present AR-Omni, a unified any-to-any model in the autoregressive paradigm without any expert decoders. AR-Omni supports autoregressive text and image generation, as well as streaming speech generation, all under a single Transformer decoder. We further address three practical issues in unified AR modeling: modality imbalance via task-aware loss reweighting, visual fidelity via a lightweight token-level perceptual alignment loss for image tokens, and stability-creativity trade-offs via a finite-state decoding mechanism. Empirically, AR-Omni achieves strong quality across three modalities while remaining real-time, achieving a 0.88 real-time factor for speech generation. 

---
# MV-S2V: Multi-View Subject-Consistent Video Generation 

**Authors**: Ziyang Song, Xinyu Gong, Bangya Liu, Zelin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17756)  

**Abstract**: Existing Subject-to-Video Generation (S2V) methods have achieved high-fidelity and subject-consistent video generation, yet remain constrained to single-view subject references. This limitation renders the S2V task reducible to an S2I + I2V pipeline, failing to exploit the full potential of video subject control. In this work, we propose and address the challenging Multi-View S2V (MV-S2V) task, which synthesizes videos from multiple reference views to enforce 3D-level subject consistency. Regarding the scarcity of training data, we first develop a synthetic data curation pipeline to generate highly customized synthetic data, complemented by a small-scale real-world captured dataset to boost the training of MV-S2V. Another key issue lies in the potential confusion between cross-subject and cross-view references in conditional generation. To overcome this, we further introduce Temporally Shifted RoPE (TS-RoPE) to distinguish between different subjects and distinct views of the same subject in reference conditioning. Our framework achieves superior 3D subject consistency w.r.t. multi-view reference images and high-quality visual outputs, establishing a new meaningful direction for subject-driven video generation. Our project page is available at <a href="this https URL URL</a> 

---
# The Script is All You Need: An Agentic Framework for Long-Horizon Dialogue-to-Cinematic Video Generation 

**Authors**: Chenyu Mu, Xin He, Qu Yang, Wanshun Chen, Jiadi Yao, Huang Liu, Zihao Yi, Bo Zhao, Xingyu Chen, Ruotian Ma, Fanghua Ye, Erkun Yang, Cheng Deng, Zhaopeng Tu, Xiaolong Li, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2601.17737)  

**Abstract**: Recent advances in video generation have produced models capable of synthesizing stunning visual content from simple text prompts. However, these models struggle to generate long-form, coherent narratives from high-level concepts like dialogue, revealing a ``semantic gap'' between a creative idea and its cinematic execution. To bridge this gap, we introduce a novel, end-to-end agentic framework for dialogue-to-cinematic-video generation. Central to our framework is ScripterAgent, a model trained to translate coarse dialogue into a fine-grained, executable cinematic script. To enable this, we construct ScriptBench, a new large-scale benchmark with rich multimodal context, annotated via an expert-guided pipeline. The generated script then guides DirectorAgent, which orchestrates state-of-the-art video models using a cross-scene continuous generation strategy to ensure long-horizon coherence. Our comprehensive evaluation, featuring an AI-powered CriticAgent and a new Visual-Script Alignment (VSA) metric, shows our framework significantly improves script faithfulness and temporal fidelity across all tested video models. Furthermore, our analysis uncovers a crucial trade-off in current SOTA models between visual spectacle and strict script adherence, providing valuable insights for the future of automated filmmaking. 

---
# Athanor: Authoring Action Modification-based Interactions on Static Visualizations via Natural Language 

**Authors**: Can Liu, Jaeuk Lee, Tianhe Chen, Zhibang Jiang, Xiaolin Wen, Yong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17736)  

**Abstract**: Interactivity is crucial for effective data visualizations. However, it is often challenging to implement interactions for existing static visualizations, since the underlying code and data for existing static visualizations are often not available, and it also takes significant time and effort to enable interactions for them even if the original code and data are available. To fill this gap, we propose Athanor, a novel approach to transform existing static visualizations into interactive ones using multimodal large language models (MLLMs) and natural language instructions. Our approach introduces three key innovations: (1) an action-modification interaction design space that maps visualization interactions into user actions and corresponding adjustments, (2) a multi-agent requirement analyzer that translates natural language instructions into an actionable operational space, and (3) a visualization abstraction transformer that converts static visualizations into flexible and interactive representations regardless of their underlying implementation. Athanor allows users to effortlessly author interactions through natural language instructions, eliminating the need for programming. We conducted two case studies and in-depth interviews with target users to evaluate our approach. The results demonstrate the effectiveness and usability of our approach in allowing users to conveniently enable flexible interactions for static visualizations. 

---
# FedCCA: Client-Centric Adaptation against Data Heterogeneity in Federated Learning on IoT Devices 

**Authors**: Kaile Wang, Jiannong Cao, Yu Yang, Xiaoyin Li, Yinfeng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17713)  

**Abstract**: With the rapid development of the Internet of Things (IoT), AI model training on private data such as human sensing data is highly desired. Federated learning (FL) has emerged as a privacy-preserving distributed training framework for this purpuse. However, the data heterogeneity issue among IoT devices can significantly degrade the model performance and convergence speed in FL. Existing approaches limit in fixed client selection and aggregation on cloud server, making the privacy-preserving extraction of client-specific information during local training challenging. To this end, we propose Client-Centric Adaptation federated learning (FedCCA), an algorithm that optimally utilizes client-specific knowledge to learn a unique model for each client through selective adaptation, aiming to alleviate the influence of data heterogeneity. Specifically, FedCCA employs dynamic client selection and adaptive aggregation based on the additional client-specific encoder. To enhance multi-source knowledge transfer, we adopt an attention-based global aggregation strategy. We conducted extensive experiments on diverse datasets to assess the efficacy of FedCCA. The experimental results demonstrate that our approach exhibits a substantial performance advantage over competing baselines in addressing this specific problem. 

---
# CaSNet: Compress-and-Send Network Based Multi-Device Speech Enhancement Model for Distributed Microphone Arrays 

**Authors**: Chengqian Jiang, Jie Zhang, Haoyin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17711)  

**Abstract**: Distributed microphone array (DMA) is a promising next-generation platform for speech interaction, where speech enhancement (SE) is still required to improve the speech quality in noisy cases. Existing SE methods usually first gather raw waveforms at a fusion center (FC) from all devices and then design a multi-microphone model, causing high bandwidth and energy costs. In this work, we propose a \emph{Compress-and-Send Network (CaSNet)} for resource-constrained DMAs, where one microphone serves as the FC and reference. Each of other devices encodes the measured raw data into a feature matrix, which is then compressed by singular value decomposition (SVD) to produce a more compact representation. The received features at the FC are aligned via cross window query with respect to the reference, followed by neural decoding to yield spatially coherent enhanced speech. Experiments on multiple datasets show that the proposed CaSNet can save the data amount with a negligible impact on the performance compared to the uncompressed case. The reproducible code is available at this https URL. 

---
# Segment Length Matters: A Study of Segment Lengths on Audio Fingerprinting Performance 

**Authors**: Ziling Gong, Yunyan Ouyang, Iram Kamdar, Melody Ma, Hongjie Chen, Franck Dernoncourt, Ryan A. Rossi, Nesreen K. Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.17690)  

**Abstract**: Audio fingerprinting provides an identifiable representation of acoustic signals, which can be later used for identification and retrieval systems. To obtain a discriminative representation, the input audio is usually segmented into shorter time intervals, allowing local acoustic features to be extracted and analyzed. Modern neural approaches typically operate on short, fixed-duration audio segments, yet the choice of segment duration is often made heuristically and rarely examined in depth. In this paper, we study how segment length affects audio fingerprinting performance. We extend an existing neural fingerprinting architecture to adopt various segment lengths and evaluate retrieval accuracy across different segment lengths and query durations. Our results show that short segment lengths (0.5-second) generally achieve better performance. Moreover, we evaluate LLM capacity in recommending the best segment length, which shows that GPT-5-mini consistently gives the best suggestions across five considerations among three studied LLMs. Our findings provide practical guidance for selecting segment duration in large-scale neural audio retrieval systems. 

---
# Agentic reinforcement learning empowers next-generation chemical language models for molecular design and synthesis 

**Authors**: Hao Li, He Cao, Shenyao Peng, Zijing Liu, Bin Feng, Yu Wang, Zhiyuan Yan, Yonghong Tian, Yu Li, Li Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17687)  

**Abstract**: Language models are revolutionizing the biochemistry domain, assisting scientists in drug design and chemical synthesis with high efficiency. Yet current approaches struggle between small language models prone to hallucination and limited knowledge retention, and large cloud-based language models plagued by privacy risks and high inference costs. To bridge this gap, we introduce ChemCRAFT, a novel framework leveraging agentic reinforcement learning to decouple chemical reasoning from knowledge storage. Instead of forcing the model to memorize vast chemical data, our approach empowers the language model to interact with a sandbox for precise information retrieval. This externalization of knowledge allows a locally deployable small model to achieve superior performance with minimal inference costs. To enable small language models for agent-calling ability, we build an agentic trajectory construction pipeline and a comprehensive chemical-agent sandbox. Based on sandbox interactions, we constructed ChemToolDataset, the first large-scale chemical tool trajectory dataset. Simultaneously, we propose SMILES-GRPO to build a dense chemical reward function, promoting the model's ability to call chemical agents. Evaluations across diverse aspects of drug design show that ChemCRAFT outperforms current cloud-based LLMs in molecular structure analysis, molecular optimization, and synthesis pathway prediction, demonstrating that scientific reasoning is not solely an emergent ability of model scale, but a learnable policy of tool orchestration. This work establishes a cost-effective and privacy-preserving paradigm for AI-aided chemistry, opening new avenues for accelerating molecular discovery with locally deployable agents. 

---
# A Model-Driven Lossless Compression Algorithm Resistant to Mismatch 

**Authors**: Cordelia Hu, Jennifer Tang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17684)  

**Abstract**: Due to the fundamental connection between next-symbol prediction and compression, modern predictive models, such as large language models (LLMs), can be combined with entropy coding to achieve compression rates that surpass those of standard compression algorithms. However, this approach relies on the assumption that the predictive model produces identical output distributions at both the encoder and decoder, since even small mismatches can cause the decoding to fail. This assumption often fails with complex predictive models, particularly those based on neural networks, a phenomenon referred to as non-determinism.
In this work, we propose a new compression algorithm based on next-token prediction that is robust to arbitrarily large, but structured, prediction mismatches. We prove the correctness of the proposed scheme under a formal mismatch certification, characterize its theoretical performance, and validate it experimentally on real datasets. Our results demonstrate reliable operation within the certified mismatch regime while achieving compression ratios that exceed those of commonly used compression methods. 

---
# Uni-RS: A Spatially Faithful Unified Understanding and Generation Model for Remote Sensing 

**Authors**: Weiyu Zhang, Yuan Hu, Yong Li, Yu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17673)  

**Abstract**: Unified remote sensing multimodal models exhibit a pronounced spatial reversal curse: Although they can accurately recognize and describe object locations in images, they often fail to faithfully execute the same spatial relations during text-to-image generation, where such relations constitute core semantic information in remote sensing. Motivated by this observation, we propose Uni-RS, the first unified multimodal model tailored for remote sensing, to explicitly address the spatial asymmetry between understanding and generation. Specifically, we first introduce explicit Spatial-Layout Planning to transform textual instructions into spatial layout plans, decoupling geometric planning from visual synthesis. We then impose Spatial-Aware Query Supervision to bias learnable queries toward spatial relations explicitly specified in the instruction. Finally, we develop Image-Caption Spatial Layout Variation to expose the model to systematic geometry-consistent spatial transformations. Extensive experiments across multiple benchmarks show that our approach substantially improves spatial faithfulness in text-to-image generation, while maintaining strong performance on multimodal understanding tasks like image captioning, visual grounding, and VQA tasks. 

---
# Grammar-Aware Literate Generative Mathematical Programming with Compiler-in-the-Loop 

**Authors**: Roberto Rossi, Steven D. Prestwich  

**Link**: [PDF](https://arxiv.org/pdf/2601.17670)  

**Abstract**: This work investigates generative mathematical programming through the lens of Algebraic Modelling Languages (AMLs) and compiler-guided model synthesis. By leveraging PyOPL, an OPL-like AML compiler that provides detailed syntax diagnostics, we introduce SyntAGM, an end-to-end system that translates natural language problem descriptions into PyOPL models via a generate--compile--assess--revise loop. SyntAGM is grammar-aware thanks to in-context exposure to the PyOPL BNF grammar, and benefits from few-shot retrieval of literate PyOPL model exemplars. To obtain a valid PyOPL model that matches the problem description, SyntAGM mobilises compiler feedback and an LLM-based alignment judge. In a comparative study against established prompting baselines SyntAGM achieves competitive accuracy with superior token, cost, and latency profiles. 

---
# UrduLM: A Resource-Efficient Monolingual Urdu Language Model 

**Authors**: Syed Muhammad Ali, Hammad Sajid, Zainab Haider, Ali Muhammad Asad, Haya Fatima, Abdul Samad  

**Link**: [PDF](https://arxiv.org/pdf/2601.17664)  

**Abstract**: Urdu, spoken by 230 million people worldwide, lacks dedicated transformer-based language models and curated corpora. While multilingual models provide limited Urdu support, they suffer from poor performance, high computational costs, and cultural inaccuracies due to insufficient training data. To address these challenges, we present UrduLM, a pretrained Urdu monolingual language model trained in low-resource settings. We curate a 33GB Urdu corpus from diverse sources, develop a custom BPE tokenizer that reduces tokenization overhead by atleast 20-30% compared to multilingual alternatives, and pretrain a 100M-parameter decoder-only model. In few-shot evaluations, UrduLM achieves competitive performance with multilingual models up to 30x its size, reaching 66.6% accuracy on sentiment classification and BLEU scores exceeding 30 on grammar correction tasks. The complete methodology -- including corpus, tokenizer, model weights, and evaluation benchmarks -- is released openly to establish a baseline for Urdu NLP research and provide a scalable framework for other underrepresented languages. 

---
# Time-Varying Causal Treatment for Quantifying the Causal Effect of Short-Term Variations on Arctic Sea Ice Dynamics 

**Authors**: Akila Sampath, Vandana Janeja, Jianwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17647)  

**Abstract**: Quantifying the causal relationship between ice melt and freshwater distribution is critical, as these complex interactions manifest as regional fluctuations in sea surface height (SSH). Leveraging SSH as a proxy for sea ice dynamics enables improved understanding of the feedback mechanisms driving polar climate change and global sea-level rise. However, conventional deep learning models often struggle with reliable treatment effect estimation in spatiotemporal settings due to unobserved confounders and the absence of physical constraints. To address these challenges, we propose the Knowledge-Guided Causal Model Variational Autoencoder (KGCM-VAE) to quantify causal mechanisms between sea ice thickness and SSH. The proposed framework integrates a velocity modulation scheme in which smoothed velocity signals are dynamically amplified via a sigmoid function governed by SSH transitions to generate physically grounded causal treatments. In addition, the model incorporates Maximum Mean Discrepancy (MMD) to balance treated and control covariate distributions in the latent space, along with a causal adjacency-constrained decoder to ensure alignment with established physical structures. Experimental results on both synthetic and real-world Arctic datasets demonstrate that KGCM-VAE achieves superior PEHE compared to state-of-the-art benchmarks. Ablation studies further confirm the effectiveness of the approach, showing that the joint application of MMD and causal adjacency constraints yields a 1.88\% reduction in estimation error. 

---
# A Systemic Evaluation of Multimodal RAG Privacy 

**Authors**: Ali Al-Lawati, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17644)  

**Abstract**: The growing adoption of multimodal Retrieval-Augmented Generation (mRAG) pipelines for vision-centric tasks (e.g. visual QA) introduces important privacy challenges. In particular, while mRAG provides a practical capability to connect private datasets to improve model performance, it risks the leakage of private information from these datasets during inference. In this paper, we perform an empirical study to analyze the privacy risks inherent in the mRAG pipeline observed through standard model prompting. Specifically, we implement a case study that attempts to infer the inclusion of a visual asset, e.g. image, in the mRAG, and if present leak the metadata, e.g. caption, related to it. Our findings highlight the need for privacy-preserving mechanisms and motivate future research on mRAG privacy. 

---
# BrainDistill: Implantable Motor Decoding with Task-Specific Knowledge Distillation 

**Authors**: Yuhan Xie, Jinhan Liu, Xiaoyong Ni, Fei Tan, Icare Sakr, Thibault Collin, Shiqi Sun, Alejandro Rodriguez Guajardo, Demon Fanny, Charles-francois Vincent Latchoumane, Henri Lorach, Jocelyne Bloch, Gregoire Courtine, Mahsa Shoaran  

**Link**: [PDF](https://arxiv.org/pdf/2601.17625)  

**Abstract**: Transformer-based neural decoders with large parameter counts, pre-trained on large-scale datasets, have recently outperformed classical machine learning models and small neural networks on brain-computer interface (BCI) tasks. However, their large parameter counts and high computational demands hinder deployment in power-constrained implantable systems. To address this challenge, we introduce BrainDistill, a novel implantable motor decoding pipeline that integrates an implantable neural decoder (IND) with a task-specific knowledge distillation (TSKD) framework. Unlike standard feature distillation methods that attempt to preserve teacher representations in full, TSKD explicitly prioritizes features critical for decoding through supervised projection. Across multiple neural datasets, IND consistently outperforms prior neural decoders on motor decoding tasks, while its TSKD-distilled variant further surpasses alternative distillation methods in few-shot calibration settings. Finally, we present a quantization-aware training scheme that enables integer-only inference with activation clipping ranges learned during training. The quantized IND enables deployment under the strict power constraints of implantable BCIs with minimal performance loss. 

---
# Human-Aligned Enhancement of Programming Answers with LLMs Guided by User Feedback 

**Authors**: Suborno Deb Bappon, Saikat Mondal, Chanchal K. Roy, Kevin Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2601.17604)  

**Abstract**: Large Language Models (LLMs) are widely used to support software developers in tasks such as code generation, optimization, and documentation. However, their ability to improve existing programming answers in a human-like manner remains underexplored. On technical question-and-answer platforms such as Stack Overflow (SO), contributors often revise answers based on user comments that identify errors, inefficiencies, or missing explanations. Yet roughly one-third of this feedback is never addressed due to limited time, expertise, or visibility, leaving many answers incomplete or outdated. This study investigates whether LLMs can enhance programming answers by interpreting and incorporating comment-based feedback. We make four main contributions. First, we introduce ReSOlve, a benchmark consisting of 790 SO answers with associated comment threads, annotated for improvement-related and general feedback. Second, we evaluate four state-of-the-art LLMs on their ability to identify actionable concerns, finding that DeepSeek achieves the best balance between precision and recall. Third, we present AUTOCOMBAT, an LLM-powered tool that improves programming answers by jointly leveraging user comments and question context. Compared to human revised references, AUTOCOMBAT produces near-human quality improvements while preserving the original intent and significantly outperforming the baseline. Finally, a user study with 58 practitioners shows strong practical value, with 84.5 percent indicating they would adopt or recommend the tool. Overall, AUTOCOMBAT demonstrates the potential of scalable, feedback-driven answer refinement to improve the reliability and trustworthiness of technical knowledge platforms. 

---
# Prompt Driven Development with Claude Code: Building a Complete TUI Framework for the Ring Programming Language 

**Authors**: Mahmoud Samir Fayed, Ahmed Samir Fayed  

**Link**: [PDF](https://arxiv.org/pdf/2601.17584)  

**Abstract**: Large language models are increasingly used in software development, yet their ability to generate and maintain large, multi module systems through natural language interaction remains insufficiently characterized. This study presents an empirical analysis of developing a 7420 line Terminal User Interface framework for the Ring programming language, completed in roughly ten hours of active work spread across three days using a purely prompt driven workflow with Claude Code, Opus 4.5. The system was produced through 107 prompts: 21 feature requests, 72 bug fix prompts, 9 prompts sharing information from Ring documentation, 4 prompts providing architectural guidance, and 1 prompt dedicated to generating documentation. Development progressed across five phases, with the Window Manager phase requiring the most interaction, followed by complex UI systems and controls expansion. Bug related prompts covered redraw issues, event handling faults, runtime errors, and layout inconsistencies, while feature requests focused primarily on new widgets, window manager capabilities, and advanced UI components. Most prompts were short, reflecting a highly iterative workflow in which the human role was limited to specifying requirements, validating behaviour, and issuing corrective prompts without writing any code manually. The resulting framework includes a complete windowing subsystem, event driven architecture, interactive widgets, hierarchical menus, grid and tree components, tab controls, and a multi window desktop environment. By combining quantitative prompt analysis with qualitative assessment of model behaviour, this study provides empirical evidence that modern LLMs can sustain architectural coherence and support the construction of production grade tooling for emerging programming languages, highlighting prompt driven development as a viable methodology within software engineering practice. 

---
# GenAI-Net: A Generative AI Framework for Automated Biomolecular Network Design 

**Authors**: Maurice Filo, Nicolò Rossi, Zhou Fang, Mustafa Khammash  

**Link**: [PDF](https://arxiv.org/pdf/2601.17582)  

**Abstract**: Biomolecular networks underpin emerging technologies in synthetic biology-from robust biomanufacturing and metabolic engineering to smart therapeutics and cell-based diagnostics-and also provide a mechanistic language for understanding complex dynamics in natural and ecological systems. Yet designing chemical reaction networks (CRNs) that implement a desired dynamical function remains largely manual: while a proposed network can be checked by simulation, the reverse problem of discovering a network from a behavioral specification is difficult, requiring substantial human insight to navigate a vast space of topologies and kinetic parameters with nonlinear and possibly stochastic dynamics. Here we introduce GenAI-Net, a generative AI framework that automates CRN design by coupling an agent that proposes reactions to simulation-based evaluation defined by a user-specified objective. GenAI-Net efficiently produces novel, topologically diverse solutions across multiple design tasks, including dose responses, complex logic gates, classifiers, oscillators, and robust perfect adaptation in deterministic and stochastic settings (including noise reduction). By turning specifications into families of circuit candidates and reusable motifs, GenAI-Net provides a general route to programmable biomolecular circuit design and accelerates the translation from desired function to implementable mechanisms. 

---
# How AI Coding Agents Modify Code: A Large-Scale Study of GitHub Pull Requests 

**Authors**: Daniel Ogenrwot, John Businge  

**Link**: [PDF](https://arxiv.org/pdf/2601.17581)  

**Abstract**: AI coding agents are increasingly acting as autonomous contributors by generating and submitting pull requests (PRs). However, we lack empirical evidence on how these agent-generated PRs differ from human contributions, particularly in how they modify code and describe their changes. Understanding these differences is essential for assessing their reliability and impact on development workflows. Using the MSR 2026 Mining Challenge version of the AIDev dataset, we analyze 24,014 merged Agentic PRs (440,295 commits) and 5,081 merged Human PRs (23,242 commits). We examine additions, deletions, commits, and files touched, and evaluate the consistency between PR descriptions and their diffs using lexical and semantic similarity. Agentic PRs differ substantially from Human PRs in commit count (Cliff's $\delta = 0.5429$) and show moderate differences in files touched and deleted lines. They also exhibit slightly higher description-to-diff similarity across all measures. These findings provide a large-scale empirical characterization of how AI coding agents contribute to open source development. 

---
# Status Hierarchies in Language Models 

**Authors**: Emilio Barkett  

**Link**: [PDF](https://arxiv.org/pdf/2601.17577)  

**Abstract**: From school playgrounds to corporate boardrooms, status hierarchies -- rank orderings based on respect and perceived competence -- are universal features of human social organization. Language models trained on human-generated text inevitably encounter these hierarchical patterns embedded in language, raising the question of whether they might reproduce such dynamics in multi-agent settings. This thesis investigates when and how language models form status hierarchies by adapting Berger et al.'s (1972) expectation states framework. I create multi-agent scenarios where separate language model instances complete sentiment classification tasks, are introduced with varying status characteristics (e.g., credentials, expertise), then have opportunities to revise their initial judgments after observing their partner's responses. The dependent variable is deference, the rate at which models shift their ratings toward their partner's position based on status cues rather than task information. Results show that language models form significant status hierarchies when capability is equal (35 percentage point asymmetry, p < .001), but capability differences dominate status cues, with the most striking effect being that high-status assignments reduce higher-capability models' deference rather than increasing lower-capability models' deference. The implications for AI safety are significant: status-seeking behavior could introduce deceptive strategies, amplify discriminatory biases, and scale across distributed deployments far faster than human hierarchies form organically. This work identifies emergent social behaviors in AI systems and highlights a previously underexplored dimension of the alignment challenge. 

---
# Improving User Privacy in Personalized Generation: Client-Side Retrieval-Augmented Modification of Server-Side Generated Speculations 

**Authors**: Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2601.17569)  

**Abstract**: Personalization is crucial for aligning Large Language Model (LLM) outputs with individual user preferences and background knowledge. State-of-the-art solutions are based on retrieval augmentation, where relevant context from a user profile is retrieved for LLM consumption. These methods deal with a trade-off between exposing retrieved private data to cloud providers and relying on less capable local models. We introduce $P^3$, an interactive framework for high-quality personalization without revealing private profiles to server-side LLMs. In $P^3$, a large server-side model generates a sequence of $k$ draft tokens based solely on the user query, while a small client-side model, with retrieval access to the user's private profile, evaluates and modifies these drafts to better reflect user preferences. This process repeats until an end token is generated. Experiments on LaMP-QA, a recent benchmark consisting of three personalized question answering datasets, show that $P^3$ consistently outperforms both non-personalized server-side and personalized client-side baselines, achieving statistically significant improvements of $7.4%$ to $9%$ on average. Importantly, $P^3$ recovers $90.3%$ to $95.7%$ of the utility of a ``leaky'' upper-bound scenario in which the full profile is exposed to the large server-side model. Privacy analyses, including linkability and attribute inference attacks, indicate that $P^3$ preserves the privacy of a non-personalized server-side model, introducing only marginal additional leakage ($1.5%$--$3.5%$) compared to submitting a query without any personal context. Additionally, the framework is efficient for edge deployment, with the client-side model generating only $9.2%$ of the total tokens. These results demonstrate that $P^3$ provides a practical, effective solution for personalized generation with improved privacy. 

---
# Real-Time Trend Prediction via Continually-Aligned LLM Query Generation 

**Authors**: Zijing Hui, Wenhan Lyu, Shusen Wang, Li Chen, Chu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17567)  

**Abstract**: Trending news detection in low-traffic search environments faces a fundamental cold-start problem, where a lack of query volume prevents systems from identifying emerging or long-tail trends. Existing methods relying on keyword frequency or query spikes are inherently slow and ineffective in these sparse settings, lagging behind real-world shifts in attention. We introduce RTTP, a novel Real-Time Trending Prediction framework that generates search queries directly from news content instead of waiting for users to issue them. RTTP leverages a continual learning LLM (CL-LLM) that converts posts into search-style queries and scores them using engagement strength + creator authority, enabling early trend surfacing before search volume forms. To ensure adaptation without degrading reasoning, we propose Mix-Policy DPO, a new preference-based continual learning approach that combines on-policy stability with off-policy novelty to mitigate catastrophic forgetting during model upgrades. Deployed at production scale on Facebook and Meta AI products, RTTP delivers +91.4% improvement in tail-trend detection precision@500 and +19% query generation accuracy over industry baselines, while sustaining stable performance after multi-week online training. This work demonstrates that LLM-generated synthetic search signals, when aligned and continually updated, unlock timely trend understanding in low-traffic search environments. 

---
# Towards Generalisable Imitation Learning Through Conditioned Transition Estimation and Online Behaviour Alignment 

**Authors**: Nathan Gavenski, Matteo Leonetti, Odinaldo Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2601.17563)  

**Abstract**: State-of-the-art imitation learning from observation methods (ILfO) have recently made significant progress, but they still have some limitations: they need action-based supervised optimisation, assume that states have a single optimal action, and tend to apply teacher actions without full consideration of the actual environment state. While the truth may be out there in observed trajectories, existing methods struggle to extract it without supervision. In this work, we propose Unsupervised Imitation Learning from Observation (UfO) that addresses all of these limitations. UfO learns a policy through a two-stage process, in which the agent first obtains an approximation of the teacher's true actions in the observed state transitions, and then refines the learned policy further by adjusting agent trajectories to closely align them with the teacher's. Experiments we conducted in five widely used environments show that UfO not only outperforms the teacher and all other ILfO methods but also displays the smallest standard deviation. This reduction in standard deviation indicates better generalisation in unseen scenarios. 

---
# Breaking the Protocol: Security Analysis of the Model Context Protocol Specification and Prompt Injection Vulnerabilities in Tool-Integrated LLM Agents 

**Authors**: Narek Maloyan, Dmitry Namiot  

**Link**: [PDF](https://arxiv.org/pdf/2601.17549)  

**Abstract**: The Model Context Protocol (MCP) has emerged as a de facto standard for integrating Large Language Models with external tools, yet no formal security analysis of the protocol specification exists. We present the first rigorous security analysis of MCP's architectural design, identifying three fundamental protocol-level vulnerabilities: (1) absence of capability attestation allowing servers to claim arbitrary permissions, (2) bidirectional sampling without origin authentication enabling server-side prompt injection, and (3) implicit trust propagation in multi-server configurations. We implement \textsc{MCPBench}, a novel framework bridging existing agent security benchmarks to MCP-compliant infrastructure, enabling direct measurement of protocol-specific attack surfaces. Through controlled experiments on 847 attack scenarios across five MCP server implementations, we demonstrate that MCP's architectural choices amplify attack success rates by 23--41\% compared to equivalent non-MCP integrations. We propose \textsc{MCPSec}, a backward-compatible protocol extension adding capability attestation and message authentication, reducing attack success rates from 52.8\% to 12.4\% with median latency overhead of 8.3ms per message. Our findings establish that MCP's security weaknesses are architectural rather than implementation-specific, requiring protocol-level remediation. 

---
# Reconstructing Training Data from Adapter-based Federated Large Language Models 

**Authors**: Silong Chen, Yuchuan Luo, Guilin Deng, Yi Liu, Min Xu, Shaojing Fu, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2601.17533)  

**Abstract**: Adapter-based Federated Large Language Models (FedLLMs) are widely adopted to reduce the computational, storage, and communication overhead of full-parameter fine-tuning for web-scale applications while preserving user privacy. By freezing the backbone and training only compact low-rank adapters, these methods appear to limit gradient leakage and thwart existing Gradient Inversion Attacks (GIAs).
Contrary to this assumption, we show that low-rank adapters create new, exploitable leakage channels. We propose the Unordered-word-bag-based Text Reconstruction (UTR) attack, a novel GIA tailored to the unique structure of adapter-based FedLLMs. UTR overcomes three core challenges: low-dimensional gradients, frozen backbones, and combinatorially large reconstruction spaces by: (i) inferring token presence from attention patterns in frozen layers, (ii) performing sentence-level inversion within the low-rank subspace of adapter gradients, and (iii) enforcing semantic coherence through constrained greedy decoding guided by language priors. Extensive experiments across diverse models (GPT2-Large, BERT, Qwen2.5-7B) and datasets (CoLA, SST-2, Rotten Tomatoes) demonstrate that UTR achieves near-perfect reconstruction accuracy (ROUGE-1/2 > 99), even with large batch size settings where prior GIAs fail completely. Our results reveal a fundamental tension between parameter efficiency and privacy in FedLLMs, challenging the prevailing belief that lightweight adaptation inherently enhances security. Our code and data are available at this https URL. 

---
# Less is More for RAG: Information Gain Pruning for Generator-Aligned Reranking and Evidence Selection 

**Authors**: Zhipeng Song, Yizhi Zhou, Xiangyu Kong, Jiulong Jiao, Xinrui Bao, Xu You, Xueqing Shi, Yuhang Zhou, Heng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2601.17532)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models with external evidence, but under a limited context budget, the key challenge is deciding which retrieved passages should be injected. We show that retrieval relevance metrics (e.g., NDCG) correlate weakly with end-to-end QA quality and can even become negatively correlated under multi-passage injection, where redundancy and mild conflicts destabilize generation. We propose \textbf{Information Gain Pruning (IGP)}, a deployment-friendly reranking-and-pruning module that selects evidence using a generator-aligned utility signal and filters weak or harmful passages before truncation, without changing existing budget interfaces. Across five open-domain QA benchmarks and multiple retrievers and generators, IGP consistently improves the quality--cost trade-off. In a representative multi-evidence setting, IGP delivers about +12--20% relative improvement in average F1 while reducing final-stage input tokens by roughly 76--79% compared to retriever-only baselines. 

---
# Bridging Expectation Signals: LLM-Based Experiments and a Behavioral Kalman Filter Framework 

**Authors**: Yu Wang, Xiangchen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17527)  

**Abstract**: As LLMs increasingly function as economic agents, the specific mechanisms LLMs use to update their belief with heterogeneous signals remain opaque. We design experiments and develop a Behavioral Kalman Filter framework to quantify how LLM-based agents update expectations, acting as households or firm CEOs, update expectations when presented with individual and aggregate signals. The results from experiments and model estimation reveal four consistent patterns: (1) agents' weighting of priors and signals deviates from unity; (2) both household and firm CEO agents place substantially larger weights on individual signals compared to aggregate signals; (3) we identify a significant and negative interaction between concurrent signals, implying that the presence of multiple information sources diminishes the marginal weight assigned to each individual signal; and (4) expectation formation patterns differ significantly between household and firm CEO agents. Finally, we demonstrate that LoRA fine-tuning mitigates, but does not fully eliminate, behavioral biases in LLM expectation formation. 

---
# "Rebuilding" Statistics in the Age of AI: A Town Hall Discussion on Culture, Infrastructure, and Training 

**Authors**: David L. Donoho, Jian Kang, Xihong Lin, Bhramar Mukherjee, Dan Nettleton, Rebecca Nugent, Abel Rodriguez, Eric P. Xing, Tian Zheng, Hongtu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17510)  

**Abstract**: This article presents the full, original record of the 2024 Joint Statistical Meetings (JSM) town hall, "Statistics in the Age of AI," which convened leading statisticians to discuss how the field is evolving in response to advances in artificial intelligence, foundation models, large-scale empirical modeling, and data-intensive infrastructures. The town hall was structured around open panel discussion and extensive audience Q&A, with the aim of eliciting candid, experience-driven perspectives rather than formal presentations or prepared statements. This document preserves the extended exchanges among panelists and audience members, with minimal editorial intervention, and organizes the conversation around five recurring questions concerning disciplinary culture and practices, data curation and "data work," engagement with modern empirical modeling, training for large-scale AI applications, and partnerships with key AI stakeholders. By providing an archival record of this discussion, the preprint aims to support transparency, community reflection, and ongoing dialogue about the evolving role of statistics in the data- and AI-centric future. 

---
# PEARL: Prototype-Enhanced Alignment for Label-Efficient Representation Learning with Deployment-Driven Insights from Digital Governance Communication Systems 

**Authors**: Ruiyu Zhang, Lin Nie, Wai-Fung Lam, Qihao Wang, Xin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17495)  

**Abstract**: In many deployed systems, new text inputs are handled by retrieving similar past cases, for example when routing and responding to citizen messages in digital governance platforms. When these systems fail, the problem is often not the language model itself, but that the nearest neighbors in the embedding space correspond to the wrong cases. Modern machine learning systems increasingly rely on fixed, high-dimensional embeddings produced by large pretrained models and sentence encoders. In real-world deployments, labels are scarce, domains shift over time, and retraining the base encoder is expensive or infeasible. As a result, downstream performance depends heavily on embedding geometry. Yet raw embeddings are often poorly aligned with the local neighborhood structure required by nearest-neighbor retrieval, similarity search, and lightweight classifiers that operate directly on embeddings. We propose PEARL (Prototype-Enhanced Aligned Representation Learning), a label-efficient approach that uses limited supervision to softly align embeddings toward class prototypes. The method reshapes local neighborhood geometry while preserving dimensionality and avoiding aggressive projection or collapse. Its aim is to bridge the gap between purely unsupervised post-processing, which offers limited and inconsistent gains, and fully supervised projections that require substantial labeled data. We evaluate PEARL under controlled label regimes ranging from extreme label scarcity to higher-label settings. In the label-scarce condition, PEARL substantially improves local neighborhood quality, yielding 25.7% gains over raw embeddings and more than 21.1% gains relative to strong unsupervised post-processing, precisely in the regime where similarity-based systems are most brittle. 

---
# Automatic Stability and Recovery for Neural Network Training 

**Authors**: Barak Or  

**Link**: [PDF](https://arxiv.org/pdf/2601.17483)  

**Abstract**: Training modern neural networks is increasingly fragile, with rare but severe destabilizing updates often causing irreversible divergence or silent performance degradation. Existing optimization methods primarily rely on preventive mechanisms embedded within the optimizer, offering limited ability to detect and recover from instability once it occurs. We introduce a supervisory runtime stability framework that treats optimization as a controlled stochastic process. By isolating an innovation signal derived from secondary measurements, such as validation probes, the framework enables automatic detection and recovery from destabilizing updates without modifying the underlying optimizer. We provide theoretical runtime safety guarantees that formalize bounded degradation and recovery. Our implementation incurs minimal overhead and is compatible with memory-constrained training settings. 

---
# Unintended Memorization of Sensitive Information in Fine-Tuned Language Models 

**Authors**: Marton Szep, Jorge Marin Ruiz, Georgios Kaissis, Paulina Seidl, Rüdiger von Eisenhart-Rothe, Florian Hinterwimmer, Daniel Rueckert  

**Link**: [PDF](https://arxiv.org/pdf/2601.17480)  

**Abstract**: Fine-tuning Large Language Models (LLMs) on sensitive datasets carries a substantial risk of unintended memorization and leakage of Personally Identifiable Information (PII), which can violate privacy regulations and compromise individual safety. In this work, we systematically investigate a critical and underexplored vulnerability: the exposure of PII that appears only in model inputs, not in training targets. Using both synthetic and real-world datasets, we design controlled extraction probes to quantify unintended PII memorization and study how factors such as language, PII frequency, task type, and model size influence memorization behavior. We further benchmark four privacy-preserving approaches including differential privacy, machine unlearning, regularization, and preference alignment, evaluating their trade-offs between privacy and task performance. Our results show that post-training methods generally provide more consistent privacy-utility trade-offs, while differential privacy achieves strong reduction in leakage in specific settings, although it can introduce training instability. These findings highlight the persistent challenge of memorization in fine-tuned LLMs and emphasize the need for robust, scalable privacy-preserving techniques. 

---
# Embodiment-Induced Coordination Regimes in Tabular Multi-Agent Q-Learning 

**Authors**: Muhammad Ahmed Atif, Nehal Naeem Haji, Mohammad Shahid Shaikh, Muhammad Ebad Atif  

**Link**: [PDF](https://arxiv.org/pdf/2601.17454)  

**Abstract**: Centralized value learning is often assumed to improve coordination and stability in multi-agent reinforcement learning, yet this assumption is rarely tested under controlled conditions. We directly evaluate it in a fully tabular predator-prey gridworld by comparing independent and centralized Q-learning under explicit embodiment constraints on agent speed and stamina. Across multiple kinematic regimes and asymmetric agent roles, centralized learning fails to provide a consistent advantage and is frequently outperformed by fully independent learning, even under full observability and exact value estimation. Moreover, asymmetric centralized-independent configurations induce persistent coordination breakdowns rather than transient learning instability. By eliminating confounding effects from function approximation and representation learning, our tabular analysis isolates coordination structure as the primary driver of these effects. The results show that increased coordination can become a liability under embodiment constraints, and that the effectiveness of centralized learning is fundamentally regime and role dependent rather than universal. 

---
# Clustering-driven Memory Compression for On-device Large Language Models 

**Authors**: Ondrej Bohdal, Pramit Saha, Umberto Michieli, Mete Ozay, Taha Ceritli  

**Link**: [PDF](https://arxiv.org/pdf/2601.17443)  

**Abstract**: Large language models (LLMs) often rely on user-specific memories distilled from past interactions to enable personalized generation. A common practice is to concatenate these memories with the input prompt, but this approach quickly exhausts the limited context available in on-device LLMs. Compressing memories by averaging can mitigate context growth, yet it frequently harms performance due to semantic conflicts across heterogeneous memories. In this work, we introduce a clustering-based memory compression strategy that balances context efficiency and personalization quality. Our method groups memories by similarity and merges them within clusters prior to concatenation, thereby preserving coherence while reducing redundancy. Experiments demonstrate that our approach substantially lowers the number of memory tokens while outperforming baseline strategies such as naive averaging or direct concatenation. Furthermore, for a fixed context budget, clustering-driven merging yields more compact memory representations and consistently enhances generation quality. 

---
# Data-driven Clustering and Merging of Adapters for On-device Large Language Models 

**Authors**: Ondrej Bohdal, Taha Ceritli, Mete Ozay, Jijoong Moon, Kyeng-Hun Lee, Hyeonmok Ko, Umberto Michieli  

**Link**: [PDF](https://arxiv.org/pdf/2601.17441)  

**Abstract**: On-device large language models commonly employ task-specific adapters (e.g., LoRAs) to deliver strong performance on downstream tasks. While storing all available adapters is impractical due to memory constraints, mobile devices typically have sufficient capacity to store a limited number of these parameters. This raises a critical challenge: how to select representative adapters that generalize well across multiple tasks - a problem that remains unexplored in existing literature. We propose a novel method D2C for adapter clustering that leverages minimal task-specific examples (e.g., 10 per task) and employs an iterative optimization process to refine cluster assignments. The adapters within each cluster are merged, creating multi-task adapters deployable on resource-constrained devices. Experimental results demonstrate that our method effectively boosts performance for considered storage budgets. 

---
# Towards a Declarative Agentic Layer for Intelligent Agents in MCP-Based Server Ecosystems 

**Authors**: Maria Jesus Rodriguez-Sanchez, Manuel Noguera, Angel Ruiz-Zafra, Kawtar Benghazi  

**Link**: [PDF](https://arxiv.org/pdf/2601.17435)  

**Abstract**: Recent advances in Large Language Models (LLMs) have enabled the development of increasingly complex agentic and multi-agent systems capable of planning, tool use and task decomposition. However, empirical evidence shows that many of these systems suffer from fundamental reliability issues, including hallucinated actions, unexecutable plans and brittle coordination. Crucially, these failures do not stem from limitations of the underlying models themselves, but from the absence of explicit architectural structure linking goals, capabilities and execution. This paper presents a declarative, model-independent architectural layer for grounded agentic workflows that addresses this gap. The proposed layer, referred to as DALIA (Declarative Agentic Layer for Intelligent Agents), formalises executable capabilities, exposes tasks through a declarative discovery protocol, maintains a federated directory of agents and their execution resources, and constructs deterministic task graphs grounded exclusively in declared operations. By enforcing a clear separation between discovery, planning and execution, the architecture constrains agent behaviour to a verifiable operational space, reducing reliance on speculative reasoning and free-form coordination. We present the architecture and design principles of the proposed layer and illustrate its operation through a representative task-oriented scenario, demonstrating how declarative grounding enables reproducible and verifiable agentic workflows across heterogeneous environments. 

---
# The 17% Gap: Quantifying Epistemic Decay in AI-Assisted Survey Papers 

**Authors**: H. Kemal İlter  

**Link**: [PDF](https://arxiv.org/pdf/2601.17431)  

**Abstract**: The adoption of Large Language Models (LLMs) in scientific writing promises efficiency but risks introducing informational entropy. While "hallucinated papers" are a known artifact, the systematic degradation of valid citation chains remains unquantified. We conducted a forensic audit of 50 recent survey papers in Artificial Intelligence (N=5,514 citations) published between September 2024 and January 2026. We utilized a hybrid verification pipeline combining DOI resolution, Crossref metadata analysis, Semantic Scholar queries, and fuzzy text matching to distinguish between formatting errors ("Sloppiness") and verifiable non-existence ("Phantoms). We detect a persistent 17.0% Phantom Rate -- citations that cannot be resolved to any digital object despite aggressive forensic recovery. Diagnostic categorization reveals three distinct failure modes: pure hallucinations (5.1%), hallucinated identifiers with valid titles (16.4%), and parsing-induced matching failures (78.5%). Longitudinal analysis reveals a flat trend (+0.07 pp/month), suggesting that high-entropy citation practices have stabilized as an endemic feature of the field. The scientific citation graph in AI survey literature exhibits "link rot" at scale. This suggests a mechanism where AI tools act as "lazy research assistants," retrieving correct titles but hallucinating metadata, thereby severing the digital chain of custody required for reproducible science. 

---
# Coronary Artery Segmentation and Vessel-Type Classification in X-Ray Angiography 

**Authors**: Mehdi Yousefzadeh, Siavash Shirzadeh Barough, Ashkan Fakharifar, Yashar Tayyarazad, Narges Eghbali, Mohaddeseh Mozaffari, Hoda Taeb, Negar Sadat Rafiee Tabatabaee, Parsa Esfahanian, Ghazaleh Sadeghi Gohar, Amineh Safavirad, Saeideh Mazloomzadeh, Ehsan khalilipur, Armin Elahifar, Majid Maleki  

**Link**: [PDF](https://arxiv.org/pdf/2601.17429)  

**Abstract**: X-ray coronary angiography (XCA) is the clinical reference standard for assessing coronary artery disease, yet quantitative analysis is limited by the difficulty of robust vessel segmentation in routine data. Low contrast, motion, foreshortening, overlap, and catheter confounding degrade segmentation and contribute to domain shift across centers. Reliable segmentation, together with vessel-type labeling, enables vessel-specific coronary analytics and downstream measurements that depend on anatomical localization. From 670 cine sequences (407 subjects), we select a best frame near peak opacification using a low-intensity histogram criterion and apply joint super-resolution and enhancement. We benchmark classical Meijering, Frangi, and Sato vesselness filters under per-image oracle tuning, a single global mean setting, and per-image parameter prediction via Support Vector Regression (SVR). Neural baselines include U-Net, FPN, and a Swin Transformer, trained with coronary-only and merged coronary+catheter supervision. A second stage assigns vessel identity (LAD, LCX, RCA). External evaluation uses the public DCA1 cohort. SVR per-image tuning improves Dice over global means for all classical filters (e.g., Frangi: 0.759 vs. 0.741). Among deep models, FPN attains 0.914+/-0.007 Dice (coronary-only), and merged coronary+catheter labels further improve to 0.931+/-0.006. On DCA1 as a strict external test, Dice drops to 0.798 (coronary-only) and 0.814 (merged), while light in-domain fine-tuning recovers to 0.881+/-0.014 and 0.882+/-0.015. Vessel-type labeling achieves 98.5% accuracy (Dice 0.844) for RCA, 95.4% (0.786) for LAD, and 96.2% (0.794) for LCX. Learned per-image tuning strengthens classical pipelines, while high-resolution FPN models and merged-label supervision improve stability and external transfer with modest adaptation. 

---
# ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs 

**Authors**: Rui Fang, Jian Li, Wei Chen, Bin Hu, Ying-Cong Chen, Xin Tang, Liang Diao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17399)  

**Abstract**: Large Language Models (LLMs) have achieved rapid progress in Chinese language understanding, yet accurately evaluating their capabilities remains challenged by benchmark saturation and prohibitive computational costs. While static leaderboards provide snapshot rankings, they often mask the structural trade-offs between capabilities. In this work, we present ReLE (Robust Efficient Live Evaluation), a scalable system designed to diagnose Capability Anisotropy, the non-uniformity of model performance across domains. Using ReLE, we evaluate 304 models (189 commercial, 115 open-source) across a Domain $\times$ Capability orthogonal matrix comprising 207,843 samples. We introduce two methodological contributions to address current evaluation pitfalls: (1) A Symbolic-Grounded Hybrid Scoring Mechanism that eliminates embedding-based false positives in reasoning tasks; (2) A Dynamic Variance-Aware Scheduler based on Neyman allocation with noise correction, which reduces compute costs by 70\% compared to full-pass evaluations while maintaining a ranking correlation of $\rho=0.96$. Our analysis reveals that aggregate rankings are highly sensitive to weighting schemes: models exhibit a Rank Stability Amplitude (RSA) of 11.4 in ReLE versus $\sim$5.0 in traditional benchmarks, confirming that modern models are highly specialized rather than generally superior. We position ReLE not as a replacement for comprehensive static benchmarks, but as a high-frequency diagnostic monitor for the evolving model landscape. 

---
# GO-OSC and VASH: Geometry-Aware Representation Learning for Early Degradation Detection in Oscillatory Systems 

**Authors**: Vashista Nobaub  

**Link**: [PDF](https://arxiv.org/pdf/2601.17396)  

**Abstract**: Early-stage degradation in oscillatory systems often manifests as geometric distortions of the dynamics, such as phase jitter, frequency drift, or loss of coherence, long before changes in signal energy are detectable. In this regime, classical energy-based diagnostics and unconstrained learned representations are structurally insensitive, leading to delayed or unstable detection. We introduce GO-OSC, a geometry-aware representation learning framework for oscillatory time series that enforces a canonical and identifiable latent parameterization, enabling stable comparison and aggregation across short, unlabeled windows. Building on this representation, we define a family of invariant linear geometric probes that target degradation-relevant directions in latent space. We provide theoretical results showing that under early phase-only degradation, energy-based statistics have zero first-order detection power, whereas geometric probes achieve strictly positive sensitivity. Our analysis characterizes when and why linear probing fails under non-identifiable representations and shows how canonicalization restores statistical detectability. Experiments on synthetic benchmarks and real vibration datasets validate the theory, demonstrating earlier detection, improved data efficiency, and robustness to operating condition changes. 

---
# ONRW: Optimizing inversion noise for high-quality and robust watermark 

**Authors**: Xuan Ding, Xiu Yan, Chuanlong Xie, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17388)  

**Abstract**: Watermarking methods have always been effective means of protecting intellectual property, yet they face significant challenges. Although existing deep learning-based watermarking systems can hide watermarks in images with minimal impact on image quality, they often lack robustness when encountering image corruptions during transmission, which undermines their practical application value. To this end, we propose a high-quality and robust watermark framework based on the diffusion model. Our method first converts the clean image into inversion noise through a null-text optimization process, and after optimizing the inversion noise in the latent space, it produces a high-quality watermarked image through an iterative denoising process of the diffusion model. The iterative denoising process serves as a powerful purification mechanism, ensuring both the visual quality of the watermarked image and enhancing the robustness of the watermark against various corruptions. To prevent the optimizing of inversion noise from distorting the original semantics of the image, we specifically introduced self-attention constraints and pseudo-mask strategies. Extensive experimental results demonstrate the superior performance of our method against various image corruptions. In particular, our method outperforms the stable signature method by an average of 10\% across 12 different image transformations on COCO datasets. Our codes are available at this https URL. 

---
# Physical Prompt Injection Attacks on Large Vision-Language Models 

**Authors**: Chen Ling, Kai Hu, Hangcheng Liu, Xingshuo Han, Tianwei Zhang, Changhai Ou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17383)  

**Abstract**: Large Vision-Language Models (LVLMs) are increasingly deployed in real-world intelligent systems for perception and reasoning in open physical environments. While LVLMs are known to be vulnerable to prompt injection attacks, existing methods either require access to input channels or depend on knowledge of user queries, assumptions that rarely hold in practical deployments. We propose the first Physical Prompt Injection Attack (PPIA), a black-box, query-agnostic attack that embeds malicious typographic instructions into physical objects perceivable by the LVLM. PPIA requires no access to the model, its inputs, or internal pipeline, and operates solely through visual observation. It combines offline selection of highly recognizable and semantically effective visual prompts with strategic environment-aware placement guided by spatiotemporal attention, ensuring that the injected prompts are both perceivable and influential on model behavior. We evaluate PPIA across 10 state-of-the-art LVLMs in both simulated and real-world settings on tasks including visual question answering, planning, and navigation, PPIA achieves attack success rates up to 98%, with strong robustness under varying physical conditions such as distance, viewpoint, and illumination. Our code is publicly available at this https URL. 

---
# Prompt and Circumstances: Evaluating the Efficacy of Human Prompt Inference in AI-Generated Art 

**Authors**: Khoi Trinh, Scott Seidenberger, Joseph Spracklen, Raveen Wijewickrama, Bimal Viswanath, Murtuza Jadliwala, Anindya Maiti  

**Link**: [PDF](https://arxiv.org/pdf/2601.17379)  

**Abstract**: The emerging field of AI-generated art has witnessed the rise of prompt marketplaces, where creators can purchase, sell, or share prompts to generate unique artworks. These marketplaces often assert ownership over prompts, claiming them as intellectual property. This paper investigates whether concealed prompts sold on prompt marketplaces can be considered bona fide intellectual property, given that humans and AI tools may be able to infer the prompts based on publicly advertised sample images accompanying each prompt on sale. Specifically, our study aims to assess (i) how accurately humans can infer the original prompt solely by examining an AI-generated image, with the goal of generating images similar to the original image, and (ii) the possibility of improving upon individual human and AI prompt inferences by crafting combined human and AI prompts with the help of a large language model. Although previous research has explored AI-driven prompt inference and protection strategies, our work is the first to incorporate a human subject study and examine collaborative human-AI prompt inference in depth. Our findings indicate that while prompts inferred by humans and prompts inferred through a combined human and AI effort can generate images with a moderate level of similarity, they are not as successful as using the original prompt. Moreover, combining human- and AI-inferred prompts using our suggested merging techniques did not improve performance over purely human-inferred prompts. 

---
# Res-MIA: A Training-Free Resolution-Based Membership Inference Attack on Federated Learning Models 

**Authors**: Mohammad Zare, Pirooz Shamsinejadbabaki  

**Link**: [PDF](https://arxiv.org/pdf/2601.17378)  

**Abstract**: Membership inference attacks (MIAs) pose a serious threat to the privacy of machine learning models by allowing adversaries to determine whether a specific data sample was included in the training set. Although federated learning (FL) is widely regarded as a privacy-aware training paradigm due to its decentralized nature, recent evidence shows that the final global model can still leak sensitive membership information through black-box access. In this paper, we introduce Res-MIA, a novel training-free and black-box membership inference attack that exploits the sensitivity of deep models to high-frequency input details. Res-MIA progressively degrades the input resolution using controlled downsampling and restoration operations, and analyzes the resulting confidence decay in the model's predictions. Our key insight is that training samples exhibit a significantly steeper confidence decline under resolution erosion compared to non-member samples, revealing a robust membership signal. Res-MIA requires no shadow models, no auxiliary data, and only a limited number of forward queries to the target model. We evaluate the proposed attack on a federated ResNet-18 trained on CIFAR-10, where it consistently outperforms existing training-free baselines and achieves an AUC of up to 0.88 with minimal computational overhead. These findings highlight frequency-sensitive overfitting as an important and previously underexplored source of privacy leakage in federated learning, and emphasize the need for privacy-aware model designs that reduce reliance on fine-grained, non-robust input features. 

---
# Diversified Scaling Inference in Time Series Foundation Models 

**Authors**: Ruijin Hua, Zichuan Liu, Kun Zhang, Yiyuan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17376)  

**Abstract**: The advancement of Time Series Foundation Models (TSFMs) has been driven primarily by large-scale pre-training, but inference-time compute potential remains largely untapped. This work systematically investigates two questions: how do TSFMs behave under standard sampling-based inference scaling, and can controlled sampling diversity enhance performance? We first examine the properties of TSFMs under standard sampling often fail to adhere to scaling laws due to insufficient exploration of the solution space. Building on this, we then delve into diversified inference scaling via tailored time series perturbations to expand the generative distribution's support. We theoretically analyze the diversity-fidelity trade-off and derive a critical sample threshold for diversified sampling to outperform standard sampling. Extensive experiments across various TSFMs and datasets show proper diversified inference scaling yields substantial performance gains without parameter updates, establishing inference design as a critical, compute-efficient dimension of TSFM optimization. As an application, we propose RobustMSE, a rigorous metric to quantify the headroom performance of TSFM under a fixed budget. Overall, our findings clarify these factor interactions, enabling reliable performance via diverse large-scale inference time series in parallel environments without re-training TSFMs. 

---
# Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers 

**Authors**: Zecheng Tang, Quantong Qiu, Yi Yang, Zhiyi Hong, Haiya Xiang, Kebin Liu, Qingqing Dang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17367)  

**Abstract**: The quadratic complexity of standard attention mechanisms poses a significant scalability bottleneck for large language models (LLMs) in long-context scenarios. While hybrid attention strategies that combine sparse and full attention within a single model offer a viable solution, they typically employ static computation ratios (i.e., fixed proportions of sparse versus full attention) and fail to adapt to the varying sparsity sensitivities of downstream tasks during inference. To address this issue, we propose Elastic Attention, which allows the model to dynamically adjust its overall sparsity based on the input. This is achieved by integrating a lightweight Attention Router into the existing pretrained model, which dynamically assigns each attention head to different computation modes. Within only 12 hours of training on 8xA800 GPUs, our method enables models to achieve both strong performance and efficient inference. Experiments across three long-context benchmarks on widely-used LLMs demonstrate the superiority of our method. 

---
# Parameter Efficient Fine Tuning Llama 3.1 for Answering Arabic Legal Questions: A Case Study on Jordanian Laws 

**Authors**: Mohammed Fasha, Bassam Hammo, Bilal Sowan, Husam Barham, Esam Nsour  

**Link**: [PDF](https://arxiv.org/pdf/2601.17364)  

**Abstract**: This study uses Jordanian law as a case study to explore the fine-tuning of the Llama-3.1 large language model for Arabic question-answering. Two versions of the model - Llama-3.1-8B-bnb-4bit and Llama-3.1-8B-Instruct-bnb-4bit - were fine-tuned using parameter-efficient fine-tuning (PEFT) with LoRA adapters and 4-bit quantized models, leveraging the Unsloth framework for accelerated and resource-efficient training. A custom dataset of 6000 legal question-answer pairs was curated from Jordanian laws and formatted into structured prompts. Performance was evaluated using the BLEU and the ROUGE metrics to compare the fine-tuned models to their respective base versions. Results demonstrated improved legal reasoning and accuracy while achieving resource efficiency through quantization and optimized fine-tuning strategies. This work underscores the potential of adapting large language models for Arabic legal domains and highlights effective techniques for fine-tuning domain-specific tasks. 

---
# Do readers prefer AI-generated Italian short stories? 

**Authors**: Michael Farrell  

**Link**: [PDF](https://arxiv.org/pdf/2601.17363)  

**Abstract**: This study investigates whether readers prefer AI-generated short stories in Italian over one written by a renowned Italian author. In a blind setup, 20 participants read and evaluated three stories, two created with ChatGPT-4o and one by Alberto Moravia, without being informed of their origin. To explore potential influencing factors, reading habits and demographic data, comprising age, gender, education and first language, were also collected. The results showed that the AI-written texts received slightly higher average ratings and were more frequently preferred, although differences were modest. No statistically significant associations were found between text preference and demographic or reading-habit variables. These findings challenge assumptions about reader preference for human-authored fiction and raise questions about the necessity of synthetic-text editing in literary contexts. 

---
# Robust Privacy: Inference-Time Privacy through Certified Robustness 

**Authors**: Jiankai Jin, Xiangzheng Zhang, Zhao Liu, Deyue Zhang, Quanchen Zou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17360)  

**Abstract**: Machine learning systems can produce personalized outputs that allow an adversary to infer sensitive input attributes at inference time. We introduce Robust Privacy (RP), an inference-time privacy notion inspired by certified robustness: if a model's prediction is provably invariant within a radius-$R$ neighborhood around an input $x$ (e.g., under the $\ell_2$ norm), then $x$ enjoys $R$-Robust Privacy, i.e., observing the prediction cannot distinguish $x$ from any input within distance $R$ of $x$. We further develop Attribute Privacy Enhancement (APE) to translate input-level invariance into an attribute-level privacy effect. In a controlled recommendation task where the decision depends primarily on a sensitive attribute, we show that RP expands the set of sensitive-attribute values compatible with a positive recommendation, expanding the inference interval accordingly. Finally, we empirically demonstrate that RP also mitigates model inversion attacks (MIAs) by masking fine-grained input-output dependence. Even at small noise levels ($\sigma=0.1$), RP reduces the attack success rate (ASR) from 73% to 4% with partial model performance degradation. RP can also partially mitigate MIAs (e.g., ASR drops to 44%) with no model performance degradation. 

---
# Spectral Geometry for Deep Learning: Compression and Hallucination Detection via Random Matrix Theory 

**Authors**: Davide Ettori  

**Link**: [PDF](https://arxiv.org/pdf/2601.17357)  

**Abstract**: Large language models and deep neural networks achieve strong performance but suffer from reliability issues and high computational cost. This thesis proposes a unified framework based on spectral geometry and random matrix theory to address both problems by analyzing the eigenvalue structure of hidden activations. The first contribution, EigenTrack, is a real-time method for detecting hallucinations and out-of-distribution behavior in language and vision-language models using spectral features and their temporal dynamics. The second contribution, RMT-KD, is a principled compression method that identifies informative spectral components and applies iterative knowledge distillation to produce compact and efficient models while preserving accuracy. Together, these results show that spectral statistics provide interpretable and robust signals for monitoring uncertainty and guiding compression in large-scale neural networks. 

---
# NeRF-MIR: Towards High-Quality Restoration of Masked Images with Neural Radiance Fields 

**Authors**: Xianliang Huang, Zhizhou Zhong, Shuhang Chen, Yi Xu, Juhong Guan, Shuigeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17350)  

**Abstract**: Neural Radiance Fields (NeRF) have demonstrated remarkable performance in novel view synthesis. However, there is much improvement room on restoring 3D scenes based on NeRF from corrupted images, which are common in natural scene captures and can significantly impact the effectiveness of NeRF. This paper introduces NeRF-MIR, a novel neural rendering approach specifically proposed for the restoration of masked images, demonstrating the potential of NeRF in this domain. Recognizing that randomly emitting rays to pixels in NeRF may not effectively learn intricate image textures, we propose a \textbf{P}atch-based \textbf{E}ntropy for \textbf{R}ay \textbf{E}mitting (\textbf{PERE}) strategy to distribute emitted rays properly. This enables NeRF-MIR to fuse comprehensive information from images of different views. Additionally, we introduce a \textbf{P}rogressively \textbf{I}terative \textbf{RE}storation (\textbf{PIRE}) mechanism to restore the masked regions in a self-training process. Furthermore, we design a dynamically-weighted loss function that automatically recalibrates the loss weights for masked regions. As existing datasets do not support NeRF-based masked image restoration, we construct three masked datasets to simulate corrupted scenarios. Extensive experiments on real data and constructed datasets demonstrate the superiority of NeRF-MIR over its counterparts in masked image restoration. 

---
# FinMetaMind: A Tech Blueprint on NLQ Systems for Financial Knowledge Search 

**Authors**: Lalit Pant, Shivang Nagar  

**Link**: [PDF](https://arxiv.org/pdf/2601.17333)  

**Abstract**: Natural Language Query (NLQ) allows users to search and interact with information systems using plain, human language instead of structured query syntax. This paper presents a technical blueprint on the design of a modern NLQ system tailored to financial knowledge search. The introduction of NLQ not only enhances the precision and recall of the knowledge search compared to traditional methods, but also facilitates deeper insights by efficiently linking disparate financial objects, events, and relationships. Using core constructs from natural language processing, search engineering, and vector data models, the proposed system aims to address key challenges in discovering, relevance ranking, data freshness, and entity recognition intrinsic to financial data retrieval. In this work, we detail the unique requirements of NLQ for financial datasets and documents, outline the architectural components for offline indexing and online retrieval, and discuss the real-world use cases of enhanced knowledge search in financial services. We delve into the theoretical underpinnings and experimental evidence supporting our proposed architecture, ultimately providing a comprehensive analysis on the subject matter. We also provide a detailed elaboration of our experimental methodology, the data used, the results and future optimizations in this study. 

---
# Conformal Feedback Alignment: Quantifying Answer-Level Reliability for Robust LLM Alignment 

**Authors**: Tiejin Chen, Xiaoou Liu, Vishnu Nandam, Kuan-Ru Liou, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2601.17329)  

**Abstract**: Preference-based alignment like Reinforcement Learning from Human Feedback (RLHF) learns from pairwise preferences, yet the labels are often noisy and inconsistent. Existing uncertainty-aware approaches weight preferences, but ignore a more fundamental factor: the reliability of the \emph{answers} being compared. To address the problem, we propose Conformal Feedback Alignment (CFA), a framework that grounds preference weighting in the statistical guarantees of Conformal Prediction (CP). CFA quantifies answer-level reliability by constructing conformal prediction sets with controllable coverage and aggregates these reliabilities into principled weights for both DPO- and PPO-style training. Experiments across different datasets show that CFA improves alignment robustness and data efficiency, highlighting that modeling \emph{answer-side} uncertainty complements preference-level weighting and yields more robust, data-efficient alignment. Codes are provided here. 

---
# ClinNet: Evidential Ordinal Regression with Bilateral Asymmetry and Prototype Memory for Knee Osteoarthritis Grading 

**Authors**: Xiaoyang Li, Runni Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17315)  

**Abstract**: Knee osteoarthritis (KOA) grading based on radiographic images is a critical yet challenging task due to subtle inter-grade differences, annotation uncertainty, and the inherently ordinal nature of disease progression. Conventional deep learning approaches typically formulate this problem as deterministic multi-class classification, ignoring both the continuous progression of degeneration and the uncertainty in expert annotations. In this work, we propose ClinNet, a novel trustworthy framework that addresses KOA grading as an evidential ordinal regression problem. The proposed method integrates three key components: (1) a Bilateral Asymmetry Encoder (BAE) that explicitly models medial-lateral structural discrepancies; (2) a Diagnostic Memory Bank that maintains class-wise prototypes to stabilize feature representations; and (3) an Evidential Ordinal Head based on the Normal-Inverse-Gamma (NIG) distribution to jointly estimate continuous KL grades and epistemic uncertainty. Extensive experiments demonstrate that ClinNet achieves a Quadratic Weighted Kappa of 0.892 and Accuracy of 0.768, statistically outperforming state-of-the-art baselines (p < 0.001). Crucially, we demonstrate that the model's uncertainty estimates successfully flag out-of-distribution samples and potential misdiagnoses, paving the way for safe clinical deployment. 

---
# Meta-Judging with Large Language Models: Concepts, Methods, and Challenges 

**Authors**: Hugo Silva, Mateus Mendes, Hugo Gonçalo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2601.17312)  

**Abstract**: Large language models (LLMs) are evolving fast and are now frequently used as evaluators, in a process typically referred to as LLM-as-a-Judge, which provides quality assessments of model outputs. However, recent research points out significant vulnerabilities in such evaluation, including sensitivity to prompts, systematic biases, verbosity effects, and unreliable or hallucinated rationales. These limitations motivated the development of a more robust paradigm, dubbed LLM-as-a-Meta-Judge. This survey reviews recent advances in meta-judging and organizes the literature, by introducing a framework along six key perspectives: (i) Conceptual Foundations, (ii) Mechanisms of Meta-Judging, (iii) Alignment Training Methods, (iv) Evaluation, (v) Limitations and Failure Modes, and (vi) Future Directions. By analyzing the limitations of LLM-as-a-Judge and summarizing recent advances in meta-judging by LLMs, we argue that LLM-as-a-Meta-Judge offers a promising direction for more stable and trustworthy automated evaluation, while highlighting remaining challenges related to cost, prompt sensitivity, and shared model biases, which must be addressed to advance the next generation of LLM evaluation methodologies. 

---
# Decentralized Multi-Agent Swarms for Autonomous Grid Security in Industrial IoT: A Consensus-based Approach 

**Authors**: Samaresh Kumar Singh, Joyjit Roy  

**Link**: [PDF](https://arxiv.org/pdf/2601.17303)  

**Abstract**: As Industrial Internet of Things (IIoT) environments expand to include tens of thousands of connected devices. The centralization of security monitoring architectures creates serious latency issues that savvy attackers can exploit to compromise an entire manufacturing ecosystem. This paper outlines a new, decentralized multi-agent swarm (DMAS) architecture that includes autonomous artificial intelligence (AI) agents at each edge gateway, functioning as a distributed digital "immune system" for IIoT networks. Instead of using a traditional static firewall approach, the DMAS agents communicate via a lightweight peer-to-peer protocol to cooperatively detect anomalous behavior across the IIoT network without sending data to a cloud infrastructure. The authors also outline a consensus-based threat validation (CVT) process in which agents vote on the threat level of an identified threat, enabling instant quarantine of a compromised node or nodes. The authors conducted experiments on a testbed that simulated an innovative factory environment with 2000 IIoT devices and found that the DMAS demonstrated sub-millisecond response times (average of 0.85ms), 97.3% accuracy in detecting malicious activity under high load, and 87% accuracy in detecting zero-day attacks. All significantly higher than baseline values for both centralized and edge computing. Additionally, the proposed architecture can prevent real-time cascading failures in industrial control systems and reduce network bandwidth use by 89% compared to cloud-based solutions. 

---
# Dynamic Meta-Ensemble Framework for Efficient and Accurate Deep Learning in Plant Leaf Disease Detection on Resource-Constrained Edge Devices 

**Authors**: Weloday Fikadu Moges, Jianmei Su, Amin Waqas  

**Link**: [PDF](https://arxiv.org/pdf/2601.17290)  

**Abstract**: Deploying deep learning models for plant disease detection on edge devices such as IoT sensors, smartphones, and embedded systems is severely constrained by limited computational resources and energy budgets. To address this challenge, we introduce a novel Dynamic Meta-Ensemble Framework (DMEF) for high-accuracy plant disease diagnosis under resource constraints. DMEF employs an adaptive weighting mechanism that dynamically combines the predictions of three lightweight convolutional neural networks (MobileNetV2, NASNetMobile, and InceptionV3) by optimizing a trade-off between accuracy improvements (DeltaAcc) and computational efficiency (model size). During training, the ensemble weights are updated iteratively, favoring models exhibiting high performance and low complexity. Extensive experiments on benchmark datasets for potato and maize diseases demonstrate state-of-the-art classification accuracies of 99.53% and 96.61%, respectively, surpassing standalone models and static ensembles by 2.1% and 6.3%. With computationally efficient inference latency (<75ms) and a compact footprint (<1 million parameters), DMEF shows strong potential for edge-based agricultural monitoring, suggesting viability for scalable crop disease management. This bridges the gap between high-accuracy AI and practical field applications. 

---
# Mind the Ambiguity: Aleatoric Uncertainty Quantification in LLMs for Safe Medical Question Answering 

**Authors**: Yaokun Liu, Yifan Liu, Phoebe Mbuvi, Zelin Li, Ruichen Yao, Gawon Lim, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17284)  

**Abstract**: The deployment of Large Language Models in Medical Question Answering is severely hampered by ambiguous user queries, a significant safety risk that demonstrably reduces answer accuracy in high-stakes healthcare settings. In this paper, we formalize this challenge by linking input ambiguity to aleatoric uncertainty (AU), which is the irreducible uncertainty arising from underspecified input. To facilitate research in this direction, we construct CV-MedBench, the first benchmark designed for studying input ambiguity in Medical QA. Using this benchmark, we analyze AU from a representation engineering perspective, revealing that AU is linearly encoded in LLM's internal activation patterns. Leveraging this insight, we introduce a novel AU-guided "Clarify-Before-Answer" framework, which incorporates AU-Probe - a lightweight module that detects input ambiguity directly from hidden states. Unlike existing uncertainty estimation methods, AU-Probe requires neither LLM fine-tuning nor multiple forward passes, enabling an efficient mechanism to proactively request user clarification and significantly enhance safety. Extensive experiments across four open LLMs demonstrate the effectiveness of our QA framework, with an average accuracy improvement of 9.48% over baselines. Our framework provides an efficient and robust solution for safe Medical QA, strengthening the reliability of health-related applications. The code is available at this https URL, and the CV-MedBench dataset is released on Hugging Face at this https URL. 

---
# On the Insecurity of Keystroke-Based AI Authorship Detection: Timing-Forgery Attacks Against Motor-Signal Verification 

**Authors**: David Condrey  

**Link**: [PDF](https://arxiv.org/pdf/2601.17280)  

**Abstract**: Recent proposals advocate using keystroke timing signals, specifically the coefficient of variation ($\delta$) of inter-keystroke intervals, to distinguish human-composed text from AI-generated content. We demonstrate that this class of defenses is insecure against two practical attack classes: the copy-type attack, in which a human transcribes LLM-generated text producing authentic motor signals, and timing-forgery attacks, in which automated agents sample inter-keystroke intervals from empirical human distributions. Using 13,000 sessions from the SBU corpus and three timing-forgery variants (histogram sampling, statistical impersonation, and generative LSTM), we show all attacks achieve $\ge$99.8% evasion rates against five classifiers. While detectors achieve AUC=1.000 against fully-automated injection, they classify $\ge$99.8% of attack samples as human with mean confidence $\ge$0.993. We formalize a non-identifiability result: when the detector observes only timing, the mutual information between features and content provenance is zero for copy-type attacks. Although composition and transcription produce statistically distinguishable motor patterns (Cohen's d=1.28), both yield $\delta$ values 2-4x above detection thresholds, rendering the distinction security-irrelevant. These systems confirm a human operated the keyboard, but not whether that human originated the text. Securing provenance requires architectures that bind the writing process to semantic content. 

---
# Latent-Space Contrastive Reinforcement Learning for Stable and Efficient LLM Reasoning 

**Authors**: Lianlei Shan, Han Chen, Yixuan Wang, Zhenjie Liu, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.17275)  

**Abstract**: While Large Language Models (LLMs) demonstrate exceptional performance in surface-level text generation, their nature in handling complex multi-step reasoning tasks often remains one of ``statistical fitting'' rather than systematic logical deduction. Traditional Reinforcement Learning (RL) attempts to mitigate this by introducing a ``think-before-speak'' paradigm. However, applying RL directly in high-dimensional, discrete token spaces faces three inherent challenges: sample-inefficient rollouts, high gradient estimation variance, and the risk of catastrophic forgetting. To fundamentally address these structural bottlenecks, we propose \textbf{DeepLatent Reasoning (DLR)}, a latent-space bidirectional contrastive reinforcement learning framework. This framework shifts the trial-and-error cost from expensive token-level full sequence generation to the continuous latent manifold. Specifically, we introduce a lightweight assistant model to efficiently sample $K$ reasoning chain encodings within the latent space. These encodings are filtered via a dual reward mechanism based on correctness and formatting; only high-value latent trajectories are fed into a \textbf{frozen main model} for single-pass decoding. To maximize reasoning diversity while maintaining coherence, we design a contrastive learning objective to enable directed exploration within the latent space. Since the main model parameters remain frozen during optimization, this method mathematically eliminates catastrophic forgetting. Experiments demonstrate that under comparable GPU computational budgets, DLR achieves more stable training convergence, supports longer-horizon reasoning chains, and facilitates the sustainable accumulation of reasoning capabilities, providing a viable path toward reliable and scalable reinforcement learning for LLMs. 

---
# The Viscosity of Logic: Phase Transitions and Hysteresis in DPO Alignment 

**Authors**: Marco Pollanen  

**Link**: [PDF](https://arxiv.org/pdf/2601.17260)  

**Abstract**: Direct Preference Optimization (DPO) is often tuned as if increasing alignment pressure (controlled by $\beta$) yields progressively "better" behavior. We instead treat $\beta$ as a control parameter and densely sweep it for three 7B open-weight families under a fixed DPO recipe. In Mistral, capability is sharply non-monotonic: aggregated logic-probe margins become positive only in a narrow band near $\beta \approx 10^{-2}$ and revert outside it, with boundary points that are seed-sensitive. Across architectures under the same sweep, we observe qualitatively different response modes: sharp reorganization in Mistral, selective changes in Llama, and smooth trade-offs in Qwen. Critically, the DPO preference margin can anticorrelate with reasoning capability (Pearson $r=-0.91$ for Llama logic), so margin-based selection can prefer capability-impaired models. Training path also matters: exposure to high $\beta$ induces capability losses that persist even after $\beta$ is reduced (hysteresis). These findings motivate capability-resolved evaluation across the $\beta$ landscape rather than reliance on margins or aggregate benchmarks. 

---
# Retell, Reward, Repeat: Reinforcement Learning for Narrative Theory-Informed Story Generation 

**Authors**: David Y. Liu, Xanthe Muston, Aditya Joshi, Sebastian Sequoiah-Grayson  

**Link**: [PDF](https://arxiv.org/pdf/2601.17226)  

**Abstract**: Despite the subjective nature of storytelling, past works on automatic story generation (ASG) have relied on limited ground truths for training and evaluation. In this work, we explore reinforcement learning (d-RLAIF) as a post-training alternative to supervised fine-tuning (SFT). We first apply Todorov's Theory of Narrative Equilibrium to establish principles that define desirable ASG qualities. We prompt 7B and 14B LLM-as-judge models with our principles to test alignment with human annotators and provide reward signals during d-RLAIF. We use Gemini-3-Flash to evaluate the output of our post-trained models and compare them to human-written stories from the TimeTravel dataset. We show that d-RLAIF offers a viable alternative to supervised fine-tuning (SFT)--producing stories that are more diverse and aligned with human narrative conventions. Our paper demonstrates the promise of reinforcement learning for linguistically grounded post-training for subjective tasks such as ASG. 

---
# Beyond Outcome Verification: Verifiable Process Reward Models for Structured Reasoning 

**Authors**: Massimiliano Pronesti, Anya Belz, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2601.17223)  

**Abstract**: Recent work on reinforcement learning with verifiable rewards (RLVR) has shown that large language models (LLMs) can be substantially improved using outcome-level verification signals, such as unit tests for code or exact-match checks for mathematics. In parallel, process supervision has long been explored as a way to shape the intermediate reasoning behaviour of LLMs, but existing approaches rely on neural judges to score chain-of-thought steps, leaving them vulnerable to opacity, bias, and reward hacking. To address this gap, we introduce Verifiable Process Reward Models (VPRMs), a reinforcement-learning framework in which intermediate reasoning steps are checked by deterministic, rule-based verifiers. We apply VPRMs to risk-of-bias assessment for medical evidence synthesis, a domain where guideline-defined criteria and rule-based decision paths enable programmatic verification of reasoning traces. Across multiple datasets, we find that VPRMs generate reasoning that adheres closely to domain rules and achieve substantially higher coherence between step-level decisions and final labels. Results show that VPRMs achieve up to 20% higher F1 than state-of-the-art models and 6.5% higher than verifiable outcome rewards, with substantial gains in evidence grounding and logical coherence. 

---
# Spatiotemporal Semantic V2X Framework for Cooperative Collision Prediction 

**Authors**: Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Aisha Syed, Matthew Andrews, Sean Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2601.17216)  

**Abstract**: Intelligent Transportation Systems (ITS) demand real-time collision prediction to ensure road safety and reduce accident severity. Conventional approaches rely on transmitting raw video or high-dimensional sensory data from roadside units (RSUs) to vehicles, which is impractical under vehicular communication bandwidth and latency constraints. In this work, we propose a semantic V2X framework in which RSU-mounted cameras generate spatiotemporal semantic embeddings of future frames using the Video Joint Embedding Predictive Architecture (V-JEPA). To evaluate the system, we construct a digital twin of an urban traffic environment enabling the generation of d verse traffic scenarios with both safe and collision events. These embeddings of the future frame, extracted from V-JEPA, capture task-relevant traffic dynamics and are transmitted via V2X links to vehicles, where a lightweight attentive probe and classifier decode them to predict imminent collisions. By transmitting only semantic embeddings instead of raw frames, the proposed system significantly reduces communication overhead while maintaining predictive accuracy. Experimental results demonstrate that the framework with an appropriate processing method achieves a 10% F1-score improvement for collision prediction while reducing transmission requirements by four orders of magnitude compared to raw video. This validates the potential of semantic V2X communication to enable cooperative, real-time collision prediction in ITS. 

---
# High-Rate Quantized Matrix Multiplication: Theory and Practice 

**Authors**: Or Ordentlich, Yury Polyanskiy  

**Link**: [PDF](https://arxiv.org/pdf/2601.17187)  

**Abstract**: This work investigates the problem of quantized matrix multiplication (MatMul), which has become crucial for the efficient deployment of large language models (LLMs). We consider two settings: 1) Generic MatMul, where both matrices must be quantized (weight+activation quantization); and 2) weight-only quantization, where the second matrix is only known through covariance matrix $\Sigma_X$ of its columns. For each setting, we first review the fundamental information-theoretic tradeoff between quantization rate and distortion (high-rate theory), and then analyze the performance of several popular quantization schemes, comparing them to these fundamental limits. Specifically, we discuss rate loss (compared to information theoretic optima) of absmax INT and floating-point (FP) quantization, for which we also derive remarkably accurate heuristic approximations. Weight-only quantization is related to the problem of weighted mean squared error (WMSE) source coding, whose classical (reverse) waterfilling solution dictates how one should distribute rate between coordinates of the vector. We show how waterfilling can be used to improve practical LLM quantization algorithms (GPTQ), which at present allocate rate equally. This new scheme (termed ``WaterSIC'') only uses scalar INT quantizers, but its high-rate performance is basis free (it depends only on the determinant of $\Sigma_X$ and, thus, unlike existing schemes, is immune to applying random rotations) and is within a multiplicative factor of $\frac{2\pi e}{12}$ (or 0.25 bit/entry) of the information-theoretic distortion limit (!). GPTQ's performance is affected by the choice of basis, but for a random rotation and actual $\Sigma_X$ from Llama-3-8B we find GPTQ to be within 0.1 bit (depending on the layer type) of WaterSIC, suggesting that GPTQ with random rotation is also near optimal (for high-rate quantization). 

---
# Federated Proximal Optimization for Privacy-Preserving Heart Disease Prediction: A Controlled Simulation Study on Non-IID Clinical Data 

**Authors**: Farzam Asad, Junaid Saif Khan, Maria Tariq, Sundus Munir, Muhammad Adnan Khan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17183)  

**Abstract**: Healthcare institutions have access to valuable patient data that could be of great help in the development of improved diagnostic models, but privacy regulations like HIPAA and GDPR prevent hospitals from directly sharing data with one another. Federated Learning offers a way out to this problem by facilitating collaborative model training without having the raw patient data centralized. However, clinical datasets intrinsically have non-IID (non-independent and identically distributed) features brought about by demographic disparity and diversity in disease prevalence and institutional practices. This paper presents a comprehensive simulation research of Federated Proximal Optimization (FedProx) for Heart Disease prediction based on UCI Heart Disease dataset. We generate realistic non-IID data partitions by simulating four heterogeneous hospital clients from the Cleveland Clinic dataset (303 patients), by inducing statistical heterogeneity by demographic-based stratification. Our experimental results show that FedProx with proximal parameter mu=0.05 achieves 85.00% accuracy, which is better than both centralized learning (83.33%) and isolated local models (78.45% average) without revealing patient privacy. Through generous sheer ablation studies with statistical validation on 50 independent runs we demonstrate that proximal regularization is effective in curbing client drift in heterogeneous environments. This proof-of-concept research offers algorithmic insights and practical deployment guidelines for real-world federated healthcare systems, and thus, our results are directly transferable to hospital IT-administrators, implementing privacy-preserving collaborative learning. 

---
# Conservative & Aggressive NaNs Accelerate U-Nets for Neuroimaging 

**Authors**: Inés Gonzalez-Pepe, Vinuyan Sivakolunthu, Jacob Fortin, Yohan Chatelain, Tristan Glatard  

**Link**: [PDF](https://arxiv.org/pdf/2601.17180)  

**Abstract**: Deep learning models for neuroimaging increasingly rely on large architectures, making efficiency a persistent concern despite advances in hardware. Through an analysis of numerical uncertainty of convolutional neural networks (CNNs), we observe that many operations are applied to values dominated by numerical noise and have negligible influence on model outputs. In some models, up to two-thirds of convolution operations appear redundant. We introduce Conservative & Aggressive NaNs, two novel variants of max pooling and unpooling that identify numerically unstable voxels and replace them with NaNs, allowing subsequent layers to skip computations on irrelevant data. Both methods are implemented within PyTorch and require no architectural changes. We evaluate these approaches on four CNN models spanning neuroimaging and image classification tasks. For inputs containing at least 50% NaNs, we observe consistent runtime improvements; for data with more than two-thirds NaNs )common in several neuroimaging settings) we achieve an average inference speedup of 1.67x. Conservative NaNs reduces convolution operations by an average of 30% across models and datasets, with no measurable performance degradation, and can skip up to 64.64% of convolutions in specific layers. Aggressive NaNs can skip up to 69.30% of convolutions but may occasionally affect performance. Overall, these methods demonstrate that numerical uncertainty can be exploited to reduce redundant computation and improve inference efficiency in CNNs. 

---
# TrojanGYM: A Detector-in-the-Loop LLM for Adaptive RTL Hardware Trojan Insertion 

**Authors**: Saideep Sreekumar, Zeng Wang, Akashdeep Saha, Weihua Xiao, Minghao Shao, Muhammad Shafique, Ozgur Sinanoglu, Ramesh Karri, Johann Knechtel  

**Link**: [PDF](https://arxiv.org/pdf/2601.17178)  

**Abstract**: Hardware Trojans (HTs) remain a critical threat because learning-based detectors often overfit to narrow trigger/payload patterns and small, stylized benchmarks. We introduce TrojanGYM, an agentic, LLM-driven framework that automatically curates HT insertions to expose detector blind spots while preserving design correctness. Given high-level HT specifications, a suite of cooperating LLM agents (instantiated with GPT-4, LLaMA-3.3-70B, and Gemini-2.5Pro) proposes and refines RTL modifications that realize diverse triggers and payloads without impacting normal functionality. TrojanGYM implements a feedback-driven benchmark generation loop co-designed with HT detectors, in which constraint-aware syntactic checking and GNN-based HT detectors provide feedback that iteratively refines HT specifications and insertion strategies to better surface detector blind spots. We further propose Robust-GNN4TJ, a new implementation of the GNN4TJ with improved graph extraction, training robustness, and prediction reliability, especially on LLM-generated HT designs. On the most challenging TrojanGYM-generated benchmarks, Robust-GNN4TJ raises HT detection rates from 0% to 60% relative to a prior GNN-based detector. We instantiate TrojanGYM on SRAM, AES-128, and UART designs at RTL level, and show that it systematically produces diverse, functionally correct HTs that reach up to 83.33% evasion rates against modern GNN-based detectors, revealing robustness gaps that are not apparent when these detectors are evaluated solely on existing TrustHub-style benchmarks. Post peer-review, we will release all codes and artifacts. 

---
# Beyond Factual QA: Mentorship-Oriented Question Answering over Long-Form Multilingual Content 

**Authors**: Parth Bhalerao, Diola Dsouza, Ruiwen Guan, Oana Ignat  

**Link**: [PDF](https://arxiv.org/pdf/2601.17173)  

**Abstract**: Question answering systems are typically evaluated on factual correctness, yet many real-world applications-such as education and career guidance-require mentorship: responses that provide reflection and guidance. Existing QA benchmarks rarely capture this distinction, particularly in multilingual and long-form settings. We introduce MentorQA, the first multilingual dataset and evaluation framework for mentorship-focused question answering from long-form videos, comprising nearly 9,000 QA pairs from 180 hours of content across four languages. We define mentorship-focused evaluation dimensions that go beyond factual accuracy, capturing clarity, alignment, and learning value. Using MentorQA, we compare Single-Agent, Dual-Agent, RAG, and Multi-Agent QA architectures under controlled conditions. Multi-Agent pipelines consistently produce higher-quality mentorship responses, with especially strong gains for complex topics and lower-resource languages. We further analyze the reliability of automated LLM-based evaluation, observing substantial variation in alignment with human judgments. Overall, this work establishes mentorship-focused QA as a distinct research problem and provides a multilingual benchmark for studying agentic architectures and evaluation design in educational AI. The dataset and evaluation framework are released at this https URL. 

---
# Who Gets Which Message? Auditing Demographic Bias in LLM-Generated Targeted Text 

**Authors**: Tunazzina Islam  

**Link**: [PDF](https://arxiv.org/pdf/2601.17172)  

**Abstract**: Large language models (LLMs) are increasingly capable of generating personalized, persuasive text at scale, raising new questions about bias and fairness in automated communication. This paper presents the first systematic analysis of how LLMs behave when tasked with demographic-conditioned targeted messaging. We introduce a controlled evaluation framework using three leading models -- GPT-4o, Llama-3.3, and Mistral-Large 2.1 -- across two generation settings: Standalone Generation, which isolates intrinsic demographic effects, and Context-Rich Generation, which incorporates thematic and regional context to emulate realistic targeting. We evaluate generated messages along three dimensions: lexical content, language style, and persuasive framing. We instantiate this framework on climate communication and find consistent age- and gender-based asymmetries across models: male- and youth-targeted messages emphasize agency, innovation, and assertiveness, while female- and senior-targeted messages stress warmth, care, and tradition. Contextual prompts systematically amplify these disparities, with persuasion scores significantly higher for messages tailored to younger or male audiences. Our findings demonstrate how demographic stereotypes can surface and intensify in LLM-generated targeted communication, underscoring the need for bias-aware generation pipelines and transparent auditing frameworks that explicitly account for demographic conditioning in socially sensitive applications. 

---
# Benchmarking Deep Learning-Based Reconstruction Methods for Photoacoustic Computed Tomography with Clinically Relevant Synthetic Datasets 

**Authors**: Panpan Chen, Seonyeong Park, Gangwon Jeong, Refik Mert Cam, Umberto Villa, Mark A. Anastasio  

**Link**: [PDF](https://arxiv.org/pdf/2601.17165)  

**Abstract**: Deep learning (DL)-based image reconstruction methods for photoacoustic computed tomography (PACT) have developed rapidly in recent years. However, most existing methods have not employed standardized datasets, and their evaluations rely on traditional image quality (IQ) metrics that may lack clinical relevance. The absence of a standardized framework for clinically meaningful IQ assessment hinders fair comparison and raises concerns about the reproducibility and reliability of reported advancements in PACT. A benchmarking framework is proposed that provides open-source, anatomically plausible synthetic datasets and evaluation strategies for DL-based acoustic inversion methods in PACT. The datasets each include over 11,000 two-dimensional (2D) stochastic breast objects with clinically relevant lesions and paired measurements at varying modeling complexity. The evaluation strategies incorporate both traditional and task-based IQ measures to assess fidelity and clinical utility. A preliminary benchmarking study is conducted to demonstrate the framework's utility by comparing DL-based and physics-based reconstruction methods. The benchmarking study demonstrated that the proposed framework enabled comprehensive, quantitative comparisons of reconstruction performance and revealed important limitations in certain DL-based methods. Although they performed well according to traditional IQ measures, they often failed to accurately recover lesions. This highlights the inadequacy of traditional metrics and motivates the need for task-based assessments. The proposed benchmarking framework enables systematic comparisons of DL-based acoustic inversion methods for 2D PACT. By integrating clinically relevant synthetic datasets with rigorous evaluation protocols, it enables reproducible, objective assessments and facilitates method development and system optimization in PACT. 

---
# Data-Driven Information-Theoretic Causal Bounds under Unmeasured Confounding 

**Authors**: Yonghan Jung, Bogyeong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17160)  

**Abstract**: We develop a data-driven information-theoretic framework for sharp partial identification of causal effects under unmeasured confounding. Existing approaches often rely on restrictive assumptions, such as bounded or discrete outcomes; require external inputs (for example, instrumental variables, proxies, or user-specified sensitivity parameters); necessitate full structural causal model specifications; or focus solely on population-level averages while neglecting covariate-conditional treatment effects. We overcome all four limitations simultaneously by establishing novel information-theoretic, data-driven divergence bounds. Our key theoretical contribution shows that the f-divergence between the observational distribution P(Y | A = a, X = x) and the interventional distribution P(Y | do(A = a), X = x) is upper bounded by a function of the propensity score alone. This result enables sharp partial identification of conditional causal effects directly from observational data, without requiring external sensitivity parameters, auxiliary variables, full structural specifications, or outcome boundedness assumptions. For practical implementation, we develop a semiparametric estimator satisfying Neyman orthogonality (Chernozhukov et al., 2018), which ensures square-root-n consistent inference even when nuisance functions are estimated using flexible machine learning methods. Simulation studies and real-world data applications, implemented in the GitHub repository (this https URL), demonstrate that our framework provides tight and valid causal bounds across a wide range of data-generating processes. 

---
# Interpretability of the Intent Detection Problem: A New Approach 

**Authors**: Eduardo Sanchez-Karhunen, Jose F. Quesada-Moreno, Miguel A. Gutiérrez-Naranjo  

**Link**: [PDF](https://arxiv.org/pdf/2601.17156)  

**Abstract**: Intent detection, a fundamental text classification task, aims to identify and label the semantics of user queries, playing a vital role in numerous business applications. Despite the dominance of deep learning techniques in this field, the internal mechanisms enabling Recurrent Neural Networks (RNNs) to solve intent detection tasks are poorly understood. In this work, we apply dynamical systems theory to analyze how RNN architectures address this problem, using both the balanced SNIPS and the imbalanced ATIS datasets. By interpreting sentences as trajectories in the hidden state space, we first show that on the balanced SNIPS dataset, the network learns an ideal solution: the state space, constrained to a low-dimensional manifold, is partitioned into distinct clusters corresponding to each intent. The application of this framework to the imbalanced ATIS dataset then reveals how this ideal geometric solution is distorted by class imbalance, causing the clusters for low-frequency intents to degrade. Our framework decouples geometric separation from readout alignment, providing a novel, mechanistic explanation for real world performance disparities. These findings provide new insights into RNN dynamics, offering a geometric interpretation of how dataset properties directly shape a network's computational solution. 

---
# Dynamic Role Assignment for Multi-Agent Debate 

**Authors**: Miao Zhang, Junsik Kim, Siyuan Xiang, Jian Gao, Cheng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2601.17152)  

**Abstract**: Multi-agent large language model (LLM) and vision-language model (VLM) debate systems employ specialized roles for complex problem-solving, yet model specializations are not leveraged to decide which model should fill which role. We propose dynamic role assignment, a framework that runs a Meta-Debate to select suitable agents before the actual debate. The meta-debate has two stages: (1) proposal, where candidates provide role-tailored arguments, and (2) peer review, where proposals are scored with data and role-specific criteria to choose the best agent for each position. We evaluate our method on LLM problem solving benchmarks. Applied on top of existing debate systems, our approach consistently outperforms uniform assignments (filling all roles with the same model) by up to 74.8% and random assignments (assigning models to roles without considering their suitability) by up to 29.7%, depending on the task and the specific assignment. This work establishes a new paradigm for multi-agent system design, shifting from static agent deployment to dynamic and capability-aware selection. 

---
# Learning to Collaborate: An Orchestrated-Decentralized Framework for Peer-to-Peer LLM Federation 

**Authors**: Inderjeet Singh, Eleonore Vissol-Gaudin, Andikan Otung, Motoyoshi Sekiya  

**Link**: [PDF](https://arxiv.org/pdf/2601.17133)  

**Abstract**: Fine-tuning Large Language Models (LLMs) for specialized domains is constrained by a fundamental challenge: the need for diverse, cross-organizational data conflicts with the principles of data privacy and sovereignty. While Federated Learning (FL) provides a framework for collaboration without raw data exchange, its classic centralized form introduces a single point of failure and remains vulnerable to model inversion attacks. Decentralized FL (DFL) mitigates this risk by removing the central aggregator but typically relies on inefficient, random peer-to-peer (P2P) pairings, forming a collaboration graph that is blind to agent heterogeneity and risks negative transfer. This paper introduces KNEXA-FL, a novel framework for orchestrated decentralization that resolves this trade-off. KNEXA-FL employs a non-aggregating Central Profiler/Matchmaker (CPM) that formulates P2P collaboration as a contextual bandit problem, using a LinUCB algorithm on abstract agent profiles to learn an optimal matchmaking policy. It orchestrates direct knowledge exchange between heterogeneous, PEFT-based LLM agents via secure distillation, without ever accessing the models themselves. Our comprehensive experiments on a challenging code generation task show that KNEXA-FL yields substantial gains, improving Pass@1 by approx. 50% relative to random P2P collaboration. Critically, our orchestrated approach demonstrates stable convergence, in stark contrast to a powerful centralized distillation baseline which suffers from catastrophic performance collapse. Our work establishes adaptive, learning-based orchestration as a foundational principle for building robust and effective decentralized AI ecosystems. 

---
# Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts 

**Authors**: Xuan-Phi Nguyen, Shrey Pandit, Austin Xu, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2601.17111)  

**Abstract**: Mixture-of-Experts (MoE) models are typically pre-trained with explicit load-balancing constraints to ensure statistically balanced expert routing. Despite this, we observe that even well-trained MoE models exhibit significantly imbalanced routing. This behavior is arguably natural-and even desirable - as imbalanced routing allows models to concentrate domain-specific knowledge within a subset of experts. Expert parallelism (EP) is designed to scale MoE models by distributing experts across multiple devices, but with a less-discussed assumption of balanced routing. Under extreme imbalance, EP can funnel a disproportionate number of tokens to a small number of experts, leading to compute- and memory-bound failures on overloaded devices during post-training or inference, where explicit load balancing is often inapplicable. We propose Least-Loaded Expert Parallelism (LLEP), a novel EP algorithm that dynamically reroutes excess tokens and associated expert parameters from overloaded devices to underutilized ones. This ensures that all devices complete their workloads within the minimum collective latency while respecting memory constraints. Across different model scales, LLEP achieves up to 5x speedup and 4x reduction in peak memory usage compared to standard EP. This enables faster and higher-throughput post-training and inference, with ~1.9x faster for gpt-oss-120b. We support our method with extensive theoretical analysis and comprehensive empirical evaluations, including ablation studies. These results illuminate key trade-offs and enable a principled framework for hardware-specific hyper-parameter tuning to achieve optimal performance. 

---
# Forecasting Energy Consumption using Recurrent Neural Networks: A Comparative Analysis 

**Authors**: Abhishek Maity, Viraj Tukarul  

**Link**: [PDF](https://arxiv.org/pdf/2601.17110)  

**Abstract**: Accurate short-term energy consumption forecasting is essential for efficient power grid management, resource allocation, and market stability. Traditional time-series models often fail to capture the complex, non-linear dependencies and external factors affecting energy demand. In this study, we propose a forecasting approach based on Recurrent Neural Networks (RNNs) and their advanced variant, Long Short-Term Memory (LSTM) networks. Our methodology integrates historical energy consumption data with external variables, including temperature, humidity, and time-based features. The LSTM model is trained and evaluated on a publicly available dataset, and its performance is compared against a conventional feed-forward neural network baseline. Experimental results show that the LSTM model substantially outperforms the baseline, achieving lower Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). These findings demonstrate the effectiveness of deep learning models in providing reliable and precise short-term energy forecasts for real-world applications. 

---
# Authority Signals in AI Cited Health Sources: A Framework for Evaluating Source Credibility in ChatGPT Responses 

**Authors**: Erin Jacques, Erela Datuowei, Vincent Jones II, Corey Basch, Celeta Vanderpool, Nkechi Udeozo, Griselda Chapa  

**Link**: [PDF](https://arxiv.org/pdf/2601.17109)  

**Abstract**: Health information seeking has fundamentally changed since the onset of Large Language Models (LLM), with nearly one third of ChatGPT's 800 million users asking health questions weekly. Understanding the sources of those AI generated responses is vital, as health organizations and providers are also investing in digital strategies to organically improve their ranking, reach and visibility in LLM systems like ChatGPT. As AI search optimization strategies are gaining maturity, this study introduces an Authority Signals Framework, organized in four domains that reflect key components to health information seeking, starting with "Who wrote it?" (Author Credentials), followed by "Who published it?" (Institutional Affiliation), "How was it vetted?" (Quality Assurance), and "How does AI find it?" (Digital Authority). This descriptive cross-sectional study randomly selected 100 questions from HealthSearchQA which contains 3,173 consumer health questions curated by Google Research from publicly available search engine suggestions. Those questions were entered into ChatGPT 5.2 Pro to record and code the cited sources through the lens of the Authority Signals Framework's four domains. Descriptive statistics were calculated for all cited sources (n=615), and cross tabulations were conducted to examine distinction among organization types. Over 75% of the sources cited in ChatGPT's health generated responses were from established institutional sources, such as Mayo Clinic, Cleveland Clinic, Wikipedia, National Health Service, PubMed with the remaining citations sourced from alternative health information sources that lacked established institutional backing. 

---
# MambaNet: Mamba-assisted Channel Estimation Neural Network With Attention Mechanism 

**Authors**: Dianxin Luan, Chengsi Liang, Jie Huang, Zheng Lin, Kaitao Meng, John Thompson, Cheng-Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17108)  

**Abstract**: This paper proposes a Mamba-assisted neural network framework incorporating self-attention mechanism to achieve improved channel estimation with low complexity for orthogonal frequency-division multiplexing (OFDM) waveforms, particularly for configurations with a large number of subcarriers. With the integration of customized Mamba architecture, the proposed framework handles large-scale subcarrier channel estimation efficiently while capturing long-distance dependencies among these subcarriers effectively. Unlike conventional Mamba structure, this paper implements a bidirectional selective scan to improve channel estimation performance, because channel gains at different subcarriers are non-causal. Moreover, the proposed framework exhibits relatively lower space complexity than transformer-based neural networks. Simulation results tested on the 3GPP TS 36.101 channel demonstrate that compared to other baseline neural network solutions, the proposed method achieves improved channel estimation performance with a reduced number of tunable parameters. 

---
# Beyond Instrumental and Substitutive Paradigms: Introducing Machine Culture as an Emergent Phenomenon in Large Language Models 

**Authors**: Yueqing Hu, Xinyang Peng, Yukun Zhao, Lin Qiu, Ka-lai Hung, Kaiping Peng  

**Link**: [PDF](https://arxiv.org/pdf/2601.17096)  

**Abstract**: Recent scholarship typically characterizes Large Language Models (LLMs) through either an \textit{Instrumental Paradigm} (viewing models as reflections of their developers' culture) or a \textit{Substitutive Paradigm} (viewing models as bilingual proxies that switch cultural frames based on language). This study challenges these anthropomorphic frameworks by proposing \textbf{Machine Culture} as an emergent, distinct phenomenon. We employed a 2 (Model Origin: US vs. China) $\times$ 2 (Prompt Language: English vs. Chinese) factorial design across eight multimodal tasks, uniquely incorporating image generation and interpretation to extend analysis beyond textual boundaries. Results revealed inconsistencies with both dominant paradigms: Model origin did not predict cultural alignment, with US models frequently exhibiting ``holistic'' traits typically associated with East Asian data. Similarly, prompt language did not trigger stable cultural frame-switching; instead, we observed \textbf{Cultural Reversal}, where English prompts paradoxically elicited higher contextual attention than Chinese prompts. Crucially, we identified a novel phenomenon termed \textbf{Service Persona Camouflage}: Reinforcement Learning from Human Feedback (RLHF) collapsed cultural variance in affective tasks into a hyper-positive, zero-variance ``helpful assistant'' persona. We conclude that LLMs do not simulate human culture but exhibit an emergent Machine Culture -- a probabilistic phenomenon shaped by \textit{superposition} in high-dimensional space and \textit{mode collapse} from safety alignment. 

---
# LoD Sketch Extraction from Architectural Models Using Generative AI: Dataset Construction for Multi-Level Architectural Design Generation 

**Authors**: Xusheng Du, Athiwat Kongkaeo, Ye Zhang, Haoran Xie  

**Link**: [PDF](https://arxiv.org/pdf/2601.17095)  

**Abstract**: For architectural design, representation across multiple Levels of Details (LoD) is essential for achieving a smooth transition from conceptual massing to detailed modeling. However, traditional LoD modeling processes rely on manual operations that are time-consuming, labor-intensive, and prone to geometric inconsistencies. While the rapid advancement of generative artificial intelligence (AI) has opened new possibilities for generating multi-level architectural models from sketch inputs, its application remains limited by the lack of high-quality paired LoD training data. To address this issue, we propose an automatic LoD sketch extraction framework using generative AI models, which progressively simplifies high-detail architectural models to automatically generate geometrically consistent and hierarchically coherent multi-LoD representations. The proposed framework integrates computer vision techniques with generative AI methods to establish a progressive extraction pipeline that transitions from detailed representations to volumetric abstractions. Experimental results demonstrate that the method maintains strong geometric consistency across LoD levels, achieving SSIM values of 0.7319 and 0.7532 for the transitions from LoD3 to LoD2 and from LoD2 to LoD1, respectively, with corresponding normalized Hausdorff distances of 25.1% and 61.0% of the image diagonal, reflecting controlled geometric deviation during abstraction. These results verify that the proposed framework effectively preserves global structure while achieving progressive semantic simplification across different LoD levels, providing reliable data and technical support for AI-driven multi-level architectural generation and hierarchical modeling. 

---
# Boltzmann-GPT: Bridging Energy-Based World Models and Language Generation 

**Authors**: Junichiro Niimi  

**Link**: [PDF](https://arxiv.org/pdf/2601.17094)  

**Abstract**: Large Language Models (LLMs) generate fluent text, yet whether they truly understand the world or merely produce plausible language about it remains contested. We propose an architectural principle, the mouth is not the brain, that explicitly separates world models from language models. Our architecture comprises three components: a Deep Boltzmann Machine (DBM) that captures domain structure as an energy-based world model, an adapter that projects latent belief states into embedding space, and a frozen GPT-2 that provides linguistic competence without domain knowledge. We instantiate this framework in the consumer review domain using Amazon smartphone reviews. Experiments demonstrate that (1) conditioning through the world model yields significantly higher sentiment correlation, lower perplexity, and greater semantic similarity compared to prompt-based generation alone; (2) the DBM's energy function distinguishes coherent from incoherent market configurations, assigning higher energy to implausible brand-price combinations; and (3) interventions on specific attributes propagate causally to generated text with intervened outputs exhibiting distributions statistically consistent with naturally occurring samples sharing the target configuration. These findings suggest that even small-scale language models can achieve consistent, controllable generation when connected to an appropriate world model, providing empirical support for separating linguistic competence from world understanding. 

---
# The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations 

**Authors**: Olha Sirikova, Alvin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17093)  

**Abstract**: Comparing neural network representations is essential for understanding and validating models in scientific applications. Existing methods, however, often provide a limited view. We propose the Triangle of Similarity, a framework that combines three complementary perspectives: static representational similarity (CKA/Procrustes), functional similarity (Linear Mode Connectivity or Predictive Similarity), and sparsity similarity (robustness under pruning). Analyzing a range of CNNs, Vision Transformers, and Vision-Language Models using both in-distribution (ImageNetV2) and out-of-distribution (CIFAR-10) testbeds, our initial findings suggest that: (1) architectural family is a primary determinant of representational similarity, forming distinct clusters; (2) CKA self-similarity and task accuracy are strongly correlated during pruning, though accuracy often degrades more sharply; and (3) for some model pairs, pruning appears to regularize representations, exposing a shared computational core. This framework offers a more holistic approach for assessing whether models have converged on similar internal mechanisms, providing a useful tool for model selection and analysis in scientific research. 

---
# CUROCKET: Optimizing ROCKET for GPU 

**Authors**: Ole Stüven, Keno Moenck, Thorsten Schüppstuhl  

**Link**: [PDF](https://arxiv.org/pdf/2601.17091)  

**Abstract**: ROCKET (RandOm Convolutional KErnel Transform) is a feature extraction algorithm created for Time Series Classification (TSC), published in 2019. It applies convolution with randomly generated kernels on a time series, producing features that can be used to train a linear classifier or regressor like Ridge. At the time of publication, ROCKET was on par with the best state-of-the-art algorithms for TSC in terms of accuracy while being significantly less computationally expensive, making ROCKET a compelling algorithm for TSC. This also led to several subsequent versions, further improving accuracy and computational efficiency. The currently available ROCKET implementations are mostly bound to execution on CPU. However, convolution is a task that can be highly parallelized and is therefore suited to be executed on GPU, which speeds up the computation significantly. A key difficulty arises from the inhomogeneous kernels ROCKET uses, making standard methods for applying convolution on GPU inefficient. In this work, we propose an algorithm that is able to efficiently perform ROCKET on GPU and achieves up to 11 times higher computational efficiency per watt than ROCKET on CPU. The code for CUROCKET is available in this repository this https URL on github. 

---
# SFO: Learning PDE Operators via Spectral Filtering 

**Authors**: Noam Koren, Rafael Moschopoulos, Kira Radinsky, Elad Hazan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17090)  

**Abstract**: Partial differential equations (PDEs) govern complex systems, yet neural operators often struggle to efficiently capture the long-range, nonlocal interactions inherent in their solution maps. We introduce Spectral Filtering Operator (SFO), a neural operator that parameterizes integral kernels using the Universal Spectral Basis (USB), a fixed, global orthonormal basis derived from the eigenmodes of the Hilbert matrix in spectral filtering theory. Motivated by our theoretical finding that the discrete Green's functions of shift-invariant PDE discretizations exhibit spatial Linear Dynamical System (LDS) structure, we prove that these kernels admit compact approximations in the USB. By learning only the spectral coefficients of rapidly decaying eigenvalues, SFO achieves a highly efficient representation. Across six benchmarks, including reaction-diffusion, fluid dynamics, and 3D electromagnetics, SFO achieves state-of-the-art accuracy, reducing error by up to 40% relative to strong baselines while using substantially fewer parameters. 

---
# Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations 

**Authors**: Preethi Seshadri, Samuel Cahyawijaya, Ayomide Odumakinde, Sameer Singh, Seraphina Goldfarb-Tarrant  

**Link**: [PDF](https://arxiv.org/pdf/2601.17087)  

**Abstract**: Agentic benchmarks increasingly rely on LLM-simulated users to scalably evaluate agent performance, yet the robustness, validity, and fairness of this approach remain unexamined. Through a user study with participants across the United States, India, Kenya, and Nigeria, we investigate whether LLM-simulated users serve as reliable proxies for real human users in evaluating agents on {\tau}-Bench retail tasks. We find that user simulation lacks robustness, with agent success rates varying up to 9 percentage points across different user LLMs. Furthermore, evaluations using simulated users exhibit systematic miscalibration, underestimating agent performance on challenging tasks and overestimating it on moderately difficult ones. African American Vernacular English (AAVE) speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers, with disparities compounding significantly with age. We also find simulated users to be a differentially effective proxy for different populations, performing worst for AAVE and Indian English speakers. Additionally, simulated users introduce conversational artifacts and surface different failure patterns than human users. These findings demonstrate that current evaluation practices risk misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges. 

---
# SonoEdit: Null-Space Constrained Knowledge Editing for Pronunciation Correction in LLM-Based TTS 

**Authors**: Ayush Pratap Singh, Harshit Singh, Nityanand Mathur, Akshat Mandloi, Sudarshan Kamath  

**Link**: [PDF](https://arxiv.org/pdf/2601.17086)  

**Abstract**: Neural text-to-speech (TTS) systems systematically mispronounce low-resource proper nouns, particularly non-English names, brands, and geographic locations, due to their underrepresentation in predominantly English training corpora. Existing solutions typically rely on expensive multilingual data collection, supervised finetuning, or manual phonetic annotation, which limits the deployment of TTS systems in linguistically diverse settings. We introduce SonoEdit, a model editing technique that surgically corrects pronunciation errors in pre-trained TTS models without retraining. Instead of costly finetuning or explicit phoneme injection, we propose a parsimonious alternative based on Null-Space Pronunciation Editing, which performs a single-shot parameter update to modify the pronunciation of specific words while provably preserving all other model behavior. We first adapt Acoustic Causal Tracing to identify the Transformer layers responsible for text-to-pronunciation mapping. We then apply Null-Space Constrained Editing to compute a closed-form weight update that corrects the target pronunciation while remaining mathematically orthogonal to the subspace governing general speech generation. This constrained update steers the model's acoustic output toward a desired pronunciation exemplar while guaranteeing zero first-order change on a preserved speech corpus. 

---
# ChemNavigator: Agentic AI Discovery of Design Rules for Organic Photocatalysts 

**Authors**: Iman Peivaste, Ahmed Makradi, Salim Belouettar  

**Link**: [PDF](https://arxiv.org/pdf/2601.17084)  

**Abstract**: The discovery of high-performance organic photocatalysts for hydrogen evolution remains limited by the vastness of chemical space and the reliance on human intuition for molecular design. Here we present ChemNavigator, an agentic AI system that autonomously derives structure-property relationships through hypothesis-driven exploration of organic photocatalyst candidates. The system integrates large language model reasoning with density functional tight binding calculations in a multi-agent architecture that mirrors the scientific method: formulating hypotheses, designing experiments, executing calculations, and validating findings through rigorous statistical analysis. Through iterative discovery cycles encompassing 200 molecules, ChemNavigator autonomously identified six statistically significant design rules governing frontier orbital energies, including the effects of ether linkages, carbonyl groups, extended conjugation, cyano groups, halogen substituents, and amine groups. Importantly, these rules correspond to established principles of organic electronic structure (resonance donation, inductive withdrawal, $\pi$-delocalization), demonstrating that the system can independently derive chemical knowledge without explicit programming. Notably, autonomous agentic reasoning extracted these six validated rules from a molecular library where previous ML approaches identified only carbonyl effects. Furthermore, the quantified effect sizes provide a prioritized ranking for synthetic chemists, while feature interaction analysis revealed diminishing returns when combining strategies, challenging additive assumptions in molecular design. This work demonstrates that agentic AI systems can autonomously derive interpretable, chemically grounded design principles, establishing a framework for AI-assisted materials discovery that complements rather than replaces chemical intuition. 

---
# Do VLMs Have a Moral Backbone? A Study on the Fragile Morality of Vision-Language Models 

**Authors**: Zhining Liu, Tianyi Wang, Xiao Lin, Penghao Ouyang, Gaotang Li, Ze Yang, Hui Liu, Sumit Keswani, Vishwa Pardeshi, Huijun Zhao, Wei Fan, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2601.17082)  

**Abstract**: Despite substantial efforts toward improving the moral alignment of Vision-Language Models (VLMs), it remains unclear whether their ethical judgments are stable in realistic settings. This work studies moral robustness in VLMs, defined as the ability to preserve moral judgments under textual and visual perturbations that do not alter the underlying moral context. We systematically probe VLMs with a diverse set of model-agnostic multimodal perturbations and find that their moral stances are highly fragile, frequently flipping under simple manipulations. Our analysis reveals systematic vulnerabilities across perturbation types, moral domains, and model scales, including a sycophancy trade-off where stronger instruction-following models are more susceptible to persuasion. We further show that lightweight inference-time interventions can partially restore moral stability. These results demonstrate that moral alignment alone is insufficient and that moral robustness is a necessary criterion for the responsible deployment of VLMs. 

---
# PC-MCL: Patient-Consistent Multi-Cycle Learning with multi-label bias correction for respiratory sound classification 

**Authors**: Seung Gyu Jeong, Seong-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.17080)  

**Abstract**: Automated respiratory sound classification supports the diagnosis of pulmonary diseases. However, many deep models still rely on cycle-level analysis and suffer from patient-specific overfitting. We propose PC-MCL (Patient-Consistent Multi-Cycle Learning) to address these limitations by utilizing three key components: multi-cycle concatenation, a 3-label formulation, and a patient-matching auxiliary task. Our work resolves a multi-label distributional bias in respiratory sound classification, a critical issue inherent to applying multi-cycle concatenation with the conventional 2-label formulation (crackle, wheeze). This bias manifests as a systematic loss of normal signal information when normal and abnormal cycles are combined. Our proposed 3-label formulation (normal, crackle, wheeze) corrects this by preserving information from all constituent cycles in mixed samples. Furthermore, the patient-matching auxiliary task acts as a multi-task regularizer, encouraging the model to learn more robust features and improving generalization. On the ICBHI 2017 benchmark, PC-MCL achieves an ICBHI Score of 65.37%, outperforming existing baselines. Ablation studies confirm that all three components are essential, working synergistically to improve the detection of abnormal respiratory events. 

---
# E2PL: Effective and Efficient Prompt Learning for Incomplete Multi-view Multi-Label Class Incremental Learning 

**Authors**: Jiajun Chen, Yue Wu, Kai Huang, Wen Xi, Yangyang Wu, Xiaoye Miao, Mengying Zhu, Meng Xi, Guanjie Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.17076)  

**Abstract**: Multi-view multi-label classification (MvMLC) is indispensable for modern web applications aggregating information from diverse sources. However, real-world web-scale settings are rife with missing views and continuously emerging classes, which pose significant obstacles to robust learning. Prevailing methods are ill-equipped for this reality, as they either lack adaptability to new classes or incur exponential parameter growth when handling all possible missing-view patterns, severely limiting their scalability in web environments. To systematically address this gap, we formally introduce a novel task, termed \emph{incomplete multi-view multi-label class incremental learning} (IMvMLCIL), which requires models to simultaneously address heterogeneous missing views and dynamic class expansion. To tackle this task, we propose \textsf{E2PL}, an Effective and Efficient Prompt Learning framework for IMvMLCIL. \textsf{E2PL} unifies two novel prompt designs: \emph{task-tailored prompts} for class-incremental adaptation and \emph{missing-aware prompts} for the flexible integration of arbitrary view-missing scenarios. To fundamentally address the exponential parameter explosion inherent in missing-aware prompts, we devise an \emph{efficient prototype tensorization} module, which leverages atomic tensor decomposition to elegantly reduce the prompt parameter complexity from exponential to linear w.r.t. the number of views. We further incorporate a \emph{dynamic contrastive learning} strategy explicitly model the complex dependencies among diverse missing-view patterns, thus enhancing the model's robustness. Extensive experiments on three benchmarks demonstrate that \textsf{E2PL} consistently outperforms state-of-the-art methods in both effectiveness and efficiency. The codes and datasets are available at this https URL. 

---
# PhysE-Inv: A Physics-Encoded Inverse Modeling approach for Arctic Snow Depth Prediction 

**Authors**: Akila Sampath, Vandana Janeja, Jianwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17074)  

**Abstract**: The accurate estimation of Arctic snow depth ($h_s$) remains a critical time-varying inverse problem due to the extreme scarcity and noise inherent in associated sea ice parameters. Existing process-based and data-driven models are either highly sensitive to sparse data or lack the physical interpretability required for climate-critical applications. To address this gap, we introduce PhysE-Inv, a novel framework that integrates a sophisticated sequential architecture, an LSTM Encoder-Decoder with Multi-head Attention and physics-guided contrastive learning, with physics-guided this http URL core innovation lies in a surjective, physics-constrained inversion methodology. This methodology first leverages the hydrostatic balance forward model as a target-formulation proxy, enabling effective learning in the absence of direct $h_s$ ground truth; second, it uses reconstruction physics regularization over a latent space to dynamically discover hidden physical parameters from noisy, incomplete time-series input. Evaluated against state-of-the-art baselines, PhysE-Inv significantly improves prediction performance, reducing error by 20\% while demonstrating superior physical consistency and resilience to data sparsity compared to empirical methods. This approach pioneers a path for noise-tolerant, interpretable inverse modeling, with wide applicability in geospatial and cryospheric domains. 

---
# Trademark Search, Artificial Intelligence and the Role of the Private Sector 

**Authors**: Sonia Katyal, Aniket Kesari  

**Link**: [PDF](https://arxiv.org/pdf/2601.17072)  

**Abstract**: Almost every industry today confronts the potential role of artificial intelligence and machine learning in its future. While many studies examine AI in consumer marketing, less attention addresses AI's role in creating and selecting trademarks that are distinctive, recognizable, and meaningful to consumers. Traditional economic approaches to trademarks focus almost exclusively on consumer-based, demand-side considerations regarding search. However, these approaches are incomplete because they fail to account for substantial costs faced not just by consumers, but by trademark applicants as well. Given AI's rapidly increasing role in trademark search and similarity analysis, lawyers and scholars should understand its dramatic implications. This paper proposes that AI should interest anyone studying trademarks and their role in economic decision-making. We examine how machine learning techniques will transform the application and interpretation of foundational trademark doctrines, producing significant implications for the trademark ecosystem. We run empirical experiments regarding trademark search to assess the efficacy of various trademark search engines, many of which employ machine learning methods. Through comparative analysis, we evaluate how these AI-powered tools function in practice. In an age where artificial intelligence increasingly governs trademark selection, the classic division between consumers and trademark owners deserves an updated, supply-side framework. This insight has transformative potential for encouraging both innovation and efficiency in trademark law and practice. 

---
# Multi-Agent Deep Reinforcement Learning Under Constrained Communications 

**Authors**: Shahil Shaik, Jonathon M. Smereka, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17069)  

**Abstract**: Centralized training with decentralized execution (CTDE) has been the dominant paradigm in multi-agent reinforcement learning (MARL), but its reliance on global state information during training introduces scalability, robustness, and generalization bottlenecks. Moreover, in practical scenarios such as adding/dropping teammates or facing environment dynamics that differ from the training, CTDE methods can be brittle and costly to retrain, whereas distributed approaches allow agents to adapt using only local information and peer-to-peer communication. We present a distributed MARL framework that removes the need for centralized critics or global information. Firstly, we develop a novel Distributed Graph Attention Network (D-GAT) that performs global state inference through multi-hop communication, where agents integrate neighbor features via input-dependent attention weights in a fully distributed manner. Leveraging D-GAT, we develop the distributed graph-attention MAPPO (DG-MAPPO) -- a distributed MARL framework where agents optimize local policies and value functions using local observations, multi-hop communication, and shared/averaged rewards. Empirical evaluation on the StarCraftII Multi-Agent Challenge, Google Research Football, and Multi-Agent Mujoco demonstrates that our method consistently outperforms strong CTDE baselines, achieving superior coordination across a wide range of cooperative tasks with both homogeneous and heterogeneous teams. Our distributed MARL framework provides a principled and scalable solution for robust collaboration, eliminating the need for centralized training or global observability. To the best of our knowledge, DG-MAPPO appears to be the first to fully eliminate reliance on privileged centralized information, enabling agents to learn and act solely through peer-to-peer communication. 

---
# A Mechanistic View on Video Generation as World Models: State and Dynamics 

**Authors**: Luozhou Wang, Zhifei Chen, Yihua Du, Dongyu Yan, Wenhang Ge, Guibao Shen, Xinli Xu, Leyi Wu, Man Chen, Tianshuo Xu, Peiran Ren, Xin Tao, Pengfei Wan, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2601.17067)  

**Abstract**: Large-scale video generation models have demonstrated emergent physical coherence, positioning them as potential world models. However, a gap remains between contemporary "stateless" video architectures and classic state-centric world model theories. This work bridges this gap by proposing a novel taxonomy centered on two pillars: State Construction and Dynamics Modeling. We categorize state construction into implicit paradigms (context management) and explicit paradigms (latent compression), while dynamics modeling is analyzed through knowledge integration and architectural reformulation. Furthermore, we advocate for a transition in evaluation from visual fidelity to functional benchmarks, testing physical persistence and causal reasoning. We conclude by identifying two critical frontiers: enhancing persistence via data-driven memory and compressed fidelity, and advancing causality through latent factor decoupling and reasoning-prior integration. By addressing these challenges, the field can evolve from generating visually plausible videos to building robust, general-purpose world simulators. 

---
# ThinkTank-ME: A Multi-Expert Framework for Middle East Event Forecasting 

**Authors**: Haoxuan Li, He Chang, Yunshan Ma, Yi Bin, Yang Yang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2601.17065)  

**Abstract**: Event forecasting is inherently influenced by multifaceted considerations, including international relations, regional historical dynamics, and cultural contexts. However, existing LLM-based approaches employ single-model architectures that generate predictions along a singular explicit trajectory, constraining their ability to capture diverse geopolitical nuances across complex regional contexts. To address this limitation, we introduce ThinkTank-ME, a novel Think Tank framework for Middle East event forecasting that emulates collaborative expert analysis in real-world strategic decision-making. To facilitate expert specialization and rigorous evaluation, we construct POLECAT-FOR-ME, a Middle East-focused event forecasting benchmark. Experimental results demonstrate the superiority of multi-expert collaboration in handling complex temporal geopolitical forecasting tasks. The code is available at this https URL. 

---
# Between Search and Platform: ChatGPT Under the DSA 

**Authors**: Toni Lorente, Kathrin Gardhouse  

**Link**: [PDF](https://arxiv.org/pdf/2601.17064)  

**Abstract**: This article examines the applicability of the Digital Services Act (DSA) to ChatGPT, arguing that it should be classified as a hybrid of the two types of hosting services: online search engines and platforms. This requires classifying search engines as hosting services, which we show is appropriate under the DSA, thereby resolving an ambiguity in the legal framework. ChatGPT performs core search functions and stores user-provided inputs and custom GPTs, meeting the definition of hosting service. We compare ChatGPT's systemic risks with those of existing Very Large Online Search Engines (VLOSEs) and Platforms (VLOPs), showing that it raises similarly serious concerns regarding illegal content, fundamental rights, democratic integrity, and public health. Now that ChatGPT has reached the 45 million EU user threshold, it should be subject to the most onerous DSA obligations, requiring the assessment and mitigation of risk emanating from both its online search engine- and platform-like characteristics. 

---
# FlashMoE: Reducing SSD I/O Bottlenecks via ML-Based Cache Replacement for Mixture-of-Experts Inference on Edge Devices 

**Authors**: Byeongju Kim, Jungwan Lee, Donghyeon Han, Hoi-Jun Yoo, Sangyeob Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.17063)  

**Abstract**: Recently, Mixture-of-Experts (MoE) models have gained attention for efficiently scaling large language models. Although these models are extremely large, their sparse activation enables inference to be performed by accessing only a fraction of the model at a time. This property opens the possibility of on-device inference of MoE, which was previously considered infeasible for such large models. Consequently, various systems have been proposed to leverage this sparsity and enable efficient MoE inference for edge devices. However, previous MoE inference systems like Fiddler[8] or DAOP[13] rely on DRAM-based offloading and are not suitable for memory constrained on-device environments. As recent MoE models grow to hundreds of gigabytes, RAM-offloading solutions become impractical. To address this, we propose FlashMoE, a system that offloads inactive experts to SSD, enabling efficient MoE inference under limited RAM. FlashMoE incorporates a lightweight ML-based caching strategy that adaptively combines recency and frequency signals to maximize expert reuse, significantly reducing storage I/O. In addition, we built a user-grade desktop platform to demonstrate the practicality of FlashMoE. On this real hardware setup, FlashMoE improves cache hit rate by up to 51% over well-known offloading policies such as LRU and LFU, and achieves up to 2.6x speedup compared to existing MoE inference systems. 

---
# A Computer Vision Pipeline for Iterative Bullet Hole Tracking in Rifle Zeroing 

**Authors**: Robert M. Belcher, Brendan C. Degryse, Leonard R. Kosta, Christopher J. Lowrance  

**Link**: [PDF](https://arxiv.org/pdf/2601.17062)  

**Abstract**: Adjusting rifle sights, a process commonly called "zeroing," requires shooters to identify and differentiate bullet holes from multiple firing iterations. Traditionally, this process demands physical inspection, introducing delays due to range safety protocols and increasing the risk of human error. We present an end-to-end computer vision system for automated bullet hole detection and iteration-based tracking directly from images taken at the firing line. Our approach combines YOLOv8 for accurate small-object detection with Intersection over Union (IoU) analysis to differentiate bullet holes across sequential images. To address the scarcity of labeled sequential data, we propose a novel data augmentation technique that removes rather than adds objects to simulate realistic firing sequences. Additionally, we introduce a preprocessing pipeline that standardizes target orientation using ORB-based perspective correction, improving model accuracy. Our system achieves 97.0% mean average precision on bullet hole detection and 88.8% accuracy in assigning bullet holes to the correct firing iteration. While designed for rifle zeroing, this framework offers broader applicability in domains requiring the temporal differentiation of visually similar objects. 

---
# Initial results of the Digital Consciousness Model 

**Authors**: Derek Shiller, Laura Duffy, Arvo Muñoz Morán, Adrià Moret, Chris Percy, Hayley Clatterbuck  

**Link**: [PDF](https://arxiv.org/pdf/2601.17060)  

**Abstract**: Artificially intelligent systems have become remarkably sophisticated. They hold conversations, write essays, and seem to understand context in ways that surprise even their creators. This raises a crucial question: Are we creating systems that are conscious? The Digital Consciousness Model (DCM) is a first attempt to assess the evidence for consciousness in AI systems in a systematic, probabilistic way. It provides a shared framework for comparing different AIs and biological organisms, and for tracking how the evidence changes over time as AI develops. Instead of adopting a single theory of consciousness, it incorporates a range of leading theories and perspectives - acknowledging that experts disagree fundamentally about what consciousness is and what conditions are necessary for it. This report describes the structure and initial results of the Digital Consciousness Model. Overall, we find that the evidence is against 2024 LLMs being conscious, but the evidence against 2024 LLMs being conscious is not decisive. The evidence against LLM consciousness is much weaker than the evidence against consciousness in simpler AI systems. 

---
# Can LLMs Clean Up Your Mess? A Survey of Application-Ready Data Preparation with LLMs 

**Authors**: Wei Zhou, Jun Zhou, Haoyu Wang, Zhenghao Li, Qikang He, Shaokun Han, Guoliang Li, Xuanhe Zhou, Yeye He, Chunwei Liu, Zirui Tang, Bin Wang, Shen Tang, Kai Zuo, Yuyu Luo, Zhenzhe Zheng, Conghui He, Jingren Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17058)  

**Abstract**: Data preparation aims to denoise raw datasets, uncover cross-dataset relationships, and extract valuable insights from them, which is essential for a wide range of data-centric applications. Driven by (i) rising demands for application-ready data (e.g., for analytics, visualization, decision-making), (ii) increasingly powerful LLM techniques, and (iii) the emergence of infrastructures that facilitate flexible agent construction (e.g., using Databricks Unity Catalog), LLM-enhanced methods are rapidly becoming a transformative and potentially dominant paradigm for data preparation.
By investigating hundreds of recent literature works, this paper presents a systematic review of this evolving landscape, focusing on the use of LLM techniques to prepare data for diverse downstream tasks. First, we characterize the fundamental paradigm shift, from rule-based, model-specific pipelines to prompt-driven, context-aware, and agentic preparation workflows. Next, we introduce a task-centric taxonomy that organizes the field into three major tasks: data cleaning (e.g., standardization, error processing, imputation), data integration (e.g., entity matching, schema matching), and data enrichment (e.g., data annotation, profiling). For each task, we survey representative techniques, and highlight their respective strengths (e.g., improved generalization, semantic understanding) and limitations (e.g., the prohibitive cost of scaling LLMs, persistent hallucinations even in advanced agents, the mismatch between advanced methods and weak evaluation). Moreover, we analyze commonly used datasets and evaluation metrics (the empirical part). Finally, we discuss open research challenges and outline a forward-looking roadmap that emphasizes scalable LLM-data systems, principled designs for reliable agentic workflows, and robust evaluation protocols. 

---
# Failing on Bias Mitigation: Investigating Why Predictive Models Struggle with Government Data 

**Authors**: Hongbo Bo, Jingyu Hu, Debbie Watson, Weiru Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17054)  

**Abstract**: The potential for bias and unfairness in AI-supporting government services raises ethical and legal concerns. Using crime rate prediction with the Bristol City Council data as a case study, we examine how these issues persist. Rather than auditing real-world deployed systems, our goal is to understand why widely adopted bias mitigation techniques often fail when applied to government data. Our findings reveal that bias mitigation approaches applied to government data are not always effective -- not because of flaws in model architecture or metric selection, but due to the inherent properties of the data itself. Through comparing a set of comprehensive models and fairness methods, our experiments consistently show that the mitigation efforts cannot overcome the embedded unfairness in the data -- further reinforcing that the origin of bias lies in the structure and history of government datasets. We then explore the reasons for the mitigation failures in predictive models on government data and highlight the potential sources of unfairness posed by data distribution shifts, the accumulation of historical bias, and delays in data release. We also discover the limitations of the blind spots in fairness analysis and bias mitigation methods when only targeting a single sensitive feature through a set of intersectional fairness experiments. Although this study is limited to one city, the findings are highly suggestive, which can contribute to an early warning that biases in government data may persist even with standard mitigation methods. 

---
# Single-Pixel Vision-Language Model for Intrinsic Privacy-Preserving Behavioral Intelligence 

**Authors**: Hongjun An, Yiliang Song, Jiawei Shao, Zhe Sun, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.17050)  

**Abstract**: Adverse social interactions, such as bullying, harassment, and other illicit activities, pose significant threats to individual well-being and public safety, leaving profound impacts on physical and mental health. However, these critical events frequently occur in privacy-sensitive environments like restrooms, and changing rooms, where conventional surveillance is prohibited or severely restricted by stringent privacy regulations and ethical concerns. Here, we propose the Single-Pixel Vision-Language Model (SP-VLM), a novel framework that reimagines secure environmental monitoring. It achieves intrinsic privacy-by-design by capturing human dynamics through inherently low-dimensional single-pixel modalities and inferring complex behavioral patterns via seamless vision-language integration. Building on this framework, we demonstrate that single-pixel sensing intrinsically suppresses identity recoverability, rendering state-of-the-art face recognition systems ineffective below a critical sampling rate. We further show that SP-VLM can nonetheless extract meaningful behavioral semantics, enabling robust anomaly detection, people counting, and activity understanding from severely degraded single-pixel observations. Combining these findings, we identify a practical sampling-rate regime in which behavioral intelligence emerges while personal identity remains strongly protected. Together, these results point to a human-rights-aligned pathway for safety monitoring that can support timely intervention without normalizing intrusive surveillance in privacy-sensitive spaces. 

---
# Summary of the Unusual Activity Recognition Challenge for Developmental Disability Support 

**Authors**: Christina Garcia, Nhat Tan Le, Taihei Fujioka, Umang Dobhal, Milyun Ni'ma Shoumi, Thanh Nha Nguyen, Sozo Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2601.17049)  

**Abstract**: This paper presents an overview of the Recognize the Unseen: Unusual Behavior Recognition from Pose Data Challenge, hosted at ISAS 2025. The challenge aims to address the critical need for automated recognition of unusual behaviors in facilities for individuals with developmental disabilities using non-invasive pose estimation data. Participating teams were tasked with distinguishing between normal and unusual activities based on skeleton keypoints extracted from video recordings of simulated scenarios. The dataset reflects real-world imbalance and temporal irregularities in behavior, and the evaluation adopted a Leave-One-Subject-Out (LOSO) strategy to ensure subject-agnostic generalization. The challenge attracted broad participation from 40 teams applying diverse approaches ranging from classical machine learning to deep learning architectures. Submissions were assessed primarily using macro-averaged F1 scores to account for class imbalance. The results highlight the difficulty of modeling rare, abrupt actions in noisy, low-dimensional data, and emphasize the importance of capturing both temporal and contextual nuances in behavior modeling. Insights from this challenge may contribute to future developments in socially responsible AI applications for healthcare and behavior monitoring. 

---
# SiMiC: Context-Aware Silicon Microstructure Characterization Using Attention-Based Convolutional Neural Networks for Field-Emission Tip Analysis 

**Authors**: Jing Jie Tan, Rupert Schreiner, Matthias Hausladen, Ali Asgharzade, Simon Edler, Julian Bartsch, Michael Bachmann, Andreas Schels, Ban-Hoe Kwan, Danny Wee-Kiat Ng, Yan-Chai Hum  

**Link**: [PDF](https://arxiv.org/pdf/2601.17048)  

**Abstract**: Accurate characterization of silicon microstructures is essential for advancing microscale fabrication, quality control, and device performance. Traditional analysis using Scanning Electron Microscopy (SEM) often requires labor-intensive, manual evaluation of feature geometry, limiting throughput and reproducibility. In this study, we propose SiMiC: Context-Aware Silicon Microstructure Characterization Using Attention-Based Convolutional Neural Networks for Field-Emission Tip Analysis. By leveraging deep learning, our approach efficiently extracts morphological features-such as size, shape, and apex curvature-from SEM images, significantly reducing human intervention while improving measurement consistency. A specialized dataset of silicon-based field-emitter tips was developed, and a customized CNN architecture incorporating attention mechanisms was trained for multi-class microstructure classification and dimensional prediction. Comparative analysis with classical image processing techniques demonstrates that SiMiC achieves high accuracy while maintaining interpretability. The proposed framework establishes a foundation for data-driven microstructure analysis directly linked to field-emission performance, opening avenues for correlating emitter geometry with emission behavior and guiding the design of optimized cold-cathode and SEM electron sources. The related dataset and algorithm repository that could serve as a baseline in this area can be found at this https URL 

---
# Interpretable and Sparse Linear Attention with Decoupled Membership-Subspace Modeling via MCR2 Objective 

**Authors**: Tianyuan Liu, Libin Hou, Linyuan Wang, Bin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17042)  

**Abstract**: Maximal Coding Rate Reduction (MCR2)-driven white-box transformer, grounded in structured representation learning, unifies interpretability and efficiency, providing a reliable white-box solution for visual modeling. However, in existing designs, tight coupling between "membership matrix" and "subspace matrix U" in MCR2 causes redundant coding under incorrect token projection. To this end, we decouple the functional relationship between the "membership matrix" and "subspaces U" in the MCR2 objective and derive an interpretable sparse linear attention operator from unrolled gradient descent of the optimized objective. Specifically, we propose to directly learn the membership matrix from inputs and subsequently derive sparse subspaces from the fullspace S. Consequently, gradient unrolling of the optimized MCR2 objective yields an interpretable sparse linear attention operator: Decoupled Membership-Subspace Attention (DMSA). Experimental results on visual tasks show that simply replacing the attention module in Token Statistics Transformer (ToST) with DMSA (we refer to as DMST) not only achieves a faster coding reduction rate but also outperforms ToST by 1.08%-1.45% in top-1 accuracy on the ImageNet-1K dataset. Compared with vanilla Transformer architectures, DMST exhibits significantly higher computational efficiency and interpretability. 

---
# Arabic Sign Language Recognition using Multimodal Approach 

**Authors**: Ghadeer Alanazi, Abir Benabid  

**Link**: [PDF](https://arxiv.org/pdf/2601.17041)  

**Abstract**: Arabic Sign Language (ArSL) is an essential communication method for individuals in the Deaf and Hard-of-Hearing community. However, existing recognition systems face significant challenges due to their reliance on single sensor approaches like Leap Motion or RGB cameras. These systems struggle with limitations such as inadequate tracking of complex hand orientations and imprecise recognition of 3D hand movements. This research paper aims to investigate the potential of a multimodal approach that combines Leap Motion and RGB camera data to explore the feasibility of recognition of ArSL. The system architecture includes two parallel subnetworks: a custom dense neural network for Leap Motion data, incorporating dropout and L2 regularization, and an image subnetwork based on a fine-tuned VGG16 model enhanced with data augmentation techniques. Feature representations from both modalities are concatenated in a fusion model and passed through fully connected layers, with final classification performed via SoftMax activation to analyze spatial and temporal features of hand gestures. The system was evaluated on a custom dataset comprising 18 ArSL words, of which 13 were correctly recognized, yielding an overall accuracy of 78%. These results offer preliminary insights into the viability of multimodal fusion for sign language recognition and highlight areas for further optimization and dataset expansion. 

---
# FP-THD: Full page transcription of historical documents 

**Authors**: H Neji, J Nogueras-Iso, J Lacasta, MÁ Latre, FJ García-Marco  

**Link**: [PDF](https://arxiv.org/pdf/2601.17040)  

**Abstract**: The transcription of historical documents written in Latin in XV and XVI centuries has special challenges as it must maintain the characters and special symbols that have distinct meanings to ensure that historical texts retain their original style and significance. This work proposes a pipeline for the transcription of historical documents preserving these special features. We propose to extend an existing text line recognition method with a layout analysis model. We analyze historical text images using a layout analysis model to extract text lines, which are then processed by an OCR model to generate a fully digitized page. We showed that our pipeline facilitates the processing of the page and produces an efficient result. We evaluated our approach on multiple datasets and demonstrate that the masked autoencoder effectively processes different types of text, including handwritten, printed and multi-language. 

---
# MANGO: A Global Single-Date Paired Dataset for Mangrove Segmentation 

**Authors**: Junhyuk Heo, Beomkyu Choi, Hyunjin Shin, Darongsae Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2601.17039)  

**Abstract**: Mangroves are critical for climate-change mitigation, requiring reliable monitoring for effective conservation. While deep learning has emerged as a powerful tool for mangrove detection, its progress is hindered by the limitations of existing datasets. In particular, many resources provide only annual map products without curated single-date image-mask pairs, limited to specific regions rather than global coverage, or remain inaccessible to the public. To address these challenges, we introduce MANGO, a large-scale global dataset comprising 42,703 labeled image-mask pairs across 124 countries. To construct this dataset, we retrieve all available Sentinel-2 imagery within the year 2020 for mangrove regions and select the best single-date observations that align with the mangrove annual mask. This selection is performed using a target detection-driven approach that leverages pixel-wise coordinate references to ensure adaptive and representative image-mask pairings. We also provide a benchmark across diverse semantic segmentation architectures under a country-disjoint split, establishing a foundation for scalable and reliable global mangrove monitoring. 

---
# AMVICC: A Novel Benchmark for Cross-Modal Failure Mode Profiling for VLMs and IGMs 

**Authors**: Aahana Basappa, Pranay Goel, Anusri Karra, Anish Karra, Asa Gilmore, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17037)  

**Abstract**: We investigated visual reasoning limitations of both multimodal large language models (MLLMs) and image generation models (IGMs) by creating a novel benchmark to systematically compare failure modes across image-to-text and text-to-image tasks, enabling cross-modal evaluation of visual understanding. Despite rapid growth in machine learning, vision language models (VLMs) still fail to understand or generate basic visual concepts such as object orientation, quantity, or spatial relationships, which highlighted gaps in elementary visual reasoning. By adapting MMVP benchmark questions into explicit and implicit prompts, we create \textit{AMVICC}, a novel benchmark for profiling failure modes across various modalities. After testing 11 MLLMs and 3 IGMs in nine categories of visual reasoning, our results show that failure modes are often shared between models and modalities, but certain failures are model-specific and modality-specific, and this can potentially be attributed to various factors. IGMs consistently struggled to manipulate specific visual components in response to prompts, especially in explicit prompts, suggesting poor control over fine-grained visual attributes. Our findings apply most directly to the evaluation of existing state-of-the-art models on structured visual reasoning tasks. This work lays the foundation for future cross-modal alignment studies, offering a framework to probe whether generation and interpretation failures stem from shared limitations to guide future improvements in unified vision-language modeling. 

---
# Data-Efficient Meningioma Segmentation via Implicit Spatiotemporal Mixing and Sim2Real Semantic Injection 

**Authors**: Yunhao Xu, Fuquan Zong, Yexuan Xing, Chulong Zhang, Guang Yang, Shilong Yang, Xiaokun Liang, Juan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17031)  

**Abstract**: The performance of medical image segmentation is increasingly defined by the efficiency of data utilization rather than merely the volume of raw data. Accurate segmentation, particularly for complex pathologies like meningiomas, demands that models fully exploit the latent information within limited high-quality annotations. To maximize the value of existing datasets, we propose a novel dual-augmentation framework that synergistically integrates spatial manifold expansion and semantic object injection. Specifically, we leverage Implicit Neural Representations (INR) to model continuous velocity fields. Unlike previous methods, we perform linear mixing on the integrated deformation fields, enabling the efficient generation of anatomically plausible variations by interpolating within the deformation space. This approach allows for the extensive exploration of structural diversity from a small set of anchors. Furthermore, we introduce a Sim2Real lesion injection module. This module constructs a high-fidelity simulation domain by transplanting lesion textures into healthy anatomical backgrounds, effectively bridging the gap between synthetic augmentation and real-world pathology. Comprehensive experiments on a hybrid dataset demonstrate that our framework significantly enhances the data efficiency and robustness of state-of-the-art models, including nnU-Net and U-Mamba, offering a potent strategy for high-performance medical image analysis with limited annotation budgets. 

---
# PALMA: A Lightweight Tropical Algebra Library for ARM-Based Embedded Systems 

**Authors**: Gnankan Landry Regis N'guessan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17028)  

**Abstract**: Tropical algebra, including max-plus, min-plus, and related idempotent semirings, provides a unifying framework in which many optimization problems that are nonlinear in classical algebra become linear. This property makes tropical methods particularly well suited for shortest paths, scheduling, throughput analysis, and discrete event systems. Despite their theoretical maturity and practical relevance, existing tropical algebra implementations primarily target desktop or server environments and remain largely inaccessible on resource-constrained embedded platforms, where such optimization problems are most acute. We present PALMA (Parallel Algebra Library for Max-plus Applications), a lightweight, dependency-free C library that brings tropical linear algebra to ARM-based embedded systems. PALMA implements a generic semiring abstraction with SIMD-accelerated kernels, enabling a single computational framework to support shortest paths, bottleneck paths, reachability, scheduling, and throughput analysis. The library supports five tropical semirings, dense and sparse (CSR) representations, tropical closure, and spectral analysis via maximum cycle mean computation. We evaluate PALMA on a Raspberry Pi 4 and demonstrate peak performance of 2,274 MOPS, speedups of up to 11.9 times over classical Bellman-Ford for single-source shortest paths, and sub-10 microsecond scheduling solves for real-time control workloads. Case studies in UAV control, IoT routing, and manufacturing systems show that tropical algebra enables efficient, predictable, and unified optimization directly on embedded hardware. PALMA is released as open-source software under the MIT license. 

---
# Scientific Image Synthesis: Benchmarking, Methodologies, and Downstream Utility 

**Authors**: Honglin Lin, Chonghan Qin, Zheng Liu, Qizhi Pei, Yu Li, Zhanping Zhong, Xin Gao, Yanfeng Wang, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2601.17027)  

**Abstract**: While synthetic data has proven effective for improving scientific reasoning in the text domain, multimodal reasoning remains constrained by the difficulty of synthesizing scientifically rigorous images. Existing Text-to-Image (T2I) models often produce outputs that are visually plausible yet scientifically incorrect, resulting in a persistent visual-logic divergence that limits their value for downstream reasoning. Motivated by recent advances in next-generation T2I models, we conduct a systematic study of scientific image synthesis across generation paradigms, evaluation, and downstream use. We analyze both direct pixel-based generation and programmatic synthesis, and propose ImgCoder, a logic-driven framework that follows an explicit "understand - plan - code" workflow to improve structural precision. To rigorously assess scientific correctness, we introduce SciGenBench, which evaluates generated images based on information utility and logical validity. Our evaluation reveals systematic failure modes in pixel-based models and highlights a fundamental expressiveness-precision trade-off. Finally, we show that fine-tuning Large Multimodal Models (LMMs) on rigorously verified synthetic scientific images yields consistent reasoning gains, with potential scaling trends analogous to the text domain, validating high-fidelity scientific synthesis as a viable path to unlocking massive multimodal reasoning capabilities. 

---
# Ensuring Computer Science Learning in the AI Era: Open Generative AI Policies and Assignment-Driven Written Quizzes 

**Authors**: Chan-Jin Chung  

**Link**: [PDF](https://arxiv.org/pdf/2601.17024)  

**Abstract**: The widespread availability of generative artificial intelligence (GenAI) has created a pressing challenge in computer science (CS) education: how to incorporate powerful AI tools into programming coursework without undermining student learning through cognitive offloading. This paper presents an assessment model that permits the use of generative AI for take-home programming assignments while enforcing individual mastery through immediate, assignment-driven written quizzes. To promote authentic learning, these in-class, closed-book assessments are weighted more heavily than the assignments themselves and are specifically designed to verify the student's comprehension of the algorithms, structure, and implementation details of their submitted code. Preliminary empirical data were collected from an upper-level computer science course to examine the relationship between self-reported GenAI usage and performance on AI-free quizzes, exams, and final course grades. Statistical analyses revealed no meaningful linear correlation between GenAI usage levels and assessment outcomes, with Pearson correlation coefficients consistently near zero. These preliminary results suggest that allowing GenAI for programming assignments does not diminish students' mastery of course concepts when learning is verified through targeted, assignment-driven quizzes. Although limited by a small sample size, this study provides preliminary evidence that the risks of cognitive offloading can be mitigated by allowing AI-assisted programming practice while verifying understanding through assignment-driven, AI-free quizzes. The findings support the responsible adoption of open GenAI policies in upper-level CS courses, when paired with rigorous, independent assessment mechanisms. 

---
# AI-based System for Transforming text and sound to Educational Videos 

**Authors**: M. E. ElAlami, S. M. Khater, M. El. R. Rehan  

**Link**: [PDF](https://arxiv.org/pdf/2601.17022)  

**Abstract**: Technological developments have produced methods that can generate educational videos from input text or sound. Recently, the use of deep learning techniques for image and video generation has been widely explored, particularly in education. However, generating video content from conditional inputs such as text or speech remains a challenging area. In this paper, we introduce a novel method to the educational structure, Generative Adversarial Network (GAN), which develop frame-for-frame frameworks and are able to create full educational videos. The proposed system is structured into three main phases In the first phase, the input (either text or speech) is transcribed using speech recognition. In the second phase, key terms are extracted and relevant images are generated using advanced models such as CLIP and diffusion models to enhance visual quality and semantic alignment. In the final phase, the generated images are synthesized into a video format, integrated with either pre-recorded or synthesized sound, resulting in a fully interactive educational video. The proposed system is compared with other systems such as TGAN, MoCoGAN, and TGANS-C, achieving a Fréchet Inception Distance (FID) score of 28.75%, which indicates improved visual quality and better over existing methods. 

---
# Evaluating the Evolution of Critical Thinking, Creativity, Communication and Collaboration in Higher Education Courses 

**Authors**: Margarida Romero  

**Link**: [PDF](https://arxiv.org/pdf/2601.17018)  

**Abstract**: The development of Creativity, Communication, Critical Thinking, and Collaboration (the 4Cs) is a central objective of contemporary competency-based education. However, empirical evidence on how these competencies evolve across learning modules and instructional phases remains limited. This study evaluates the evolution of the 4Cs from pre-pilot to pilot implementation phases across three educational contexts, using the project's 4Cs theoretical framework as an analytical lens. The analysis of three pilot cases (IASIS, EASD, and UPATRAS) compares the 4Cs scores to identify patterns of growth, stagnation, or decline over time. Results indicate that communication and critical thinking showed the most consistent and substantial improvements, particularly in pilots with lower pre-pilot baselines, suggesting that structured pilot interventions effectively support cognitive and expressive competencies. In contrast, creativity exhibited context-dependent outcomes, while collaboration emerged as the most fragile competency, often stagnating or declining during scale-up. Interpreted through the theoretical framework, these findings suggest that competency evolution is strongly shaped by instructional design, assessment alignment, and learning activity structures rather than learner ability alone. The study contributes empirical validation to the 4Cs framework and highlights the need for differentiated, competency-sensitive design and evaluation strategies when scaling educational modules. 

---
# Measuring Political Stance and Consistency in Large Language Models 

**Authors**: Salah Feras Alali, Mohammad Nashat Maasfeh, Mucahid Kutlu, Saban Kardas  

**Link**: [PDF](https://arxiv.org/pdf/2601.17016)  

**Abstract**: With the incredible advancements in Large Language Models (LLMs), many people have started using them to satisfy their information needs. However, utilizing LLMs might be problematic for political issues where disagreement is common and model outputs may reflect training-data biases or deliberate alignment choices. To better characterize such behavior, we assess the stances of nine LLMs on 24 politically sensitive issues using five prompting techniques. We find that models often adopt opposing stances on several issues; some positions are malleable under prompting, while others remain stable. Among the models examined, Grok-3-mini is the most persistent, whereas Mistral-7B is the least. For issues involving countries with different languages, models tend to support the side whose language is used in the prompt. Notably, no prompting technique alters model stances on the Qatar blockade or the oppression of Palestinians. We hope these findings raise user awareness when seeking political guidance from LLMs and encourage developers to address these concerns. 

---
# Private Accountability in the Age of Artificial Intelligence 

**Authors**: Sonia Katyal  

**Link**: [PDF](https://arxiv.org/pdf/2601.17013)  

**Abstract**: In this Article, I explore the impending conflict between the protection of civil rights and artificial intelligence (AI). While both areas of law have amassed rich and well-developed areas of scholarly work and doctrinal support, a growing body of scholars are interrogating the intersection between them. This Article argues that the issues surrounding algorithmic accountability demonstrate a deeper, more structural tension within a new generation of disputes regarding law and technology. As I argue, the true promise of AI does not lie in the information we reveal to one another, but rather in the questions it raises about the interaction of technology, property, and civil rights. For this reason, I argue that we are looking in the wrong place if we look only to the state to address issues of algorithmic accountability. Instead, we must turn to other ways to ensure more transparency and accountability that stem from private industry, rather than public regulation. The issue of algorithmic bias represents a crucial new world of civil rights concerns, one that is distinct in nature from the ones that preceded it. Since we are in a world where the activities of private corporations, rather than the state, are raising concerns about privacy, due process, and discrimination, we must focus on the role of private corporations in addressing the issue. Towards this end, I discuss a variety of tools to help eliminate the opacity of AI, including codes of conduct, impact statements, and whistleblower protection, which I argue carries the potential to encourage greater endogeneity in civil rights enforcement. Ultimately, by examining the relationship between private industry and civil rights, we can perhaps develop a new generation of forms of accountability in the process. 

---
# MathMixup: Boosting LLM Mathematical Reasoning with Difficulty-Controllable Data Synthesis and Curriculum Learning 

**Authors**: Xuchen Li, Jing Chen, Xuzhao Li, Hao Liang, Xiaohuan Zhou, Taifeng Wang, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17006)  

**Abstract**: In mathematical reasoning tasks, the advancement of Large Language Models (LLMs) relies heavily on high-quality training data with clearly defined and well-graded difficulty levels. However, existing data synthesis methods often suffer from limited diversity and lack precise control over problem difficulty, making them insufficient for supporting efficient training paradigms such as curriculum learning. To address these challenges, we propose MathMixup, a novel data synthesis paradigm that systematically generates high-quality, difficulty-controllable mathematical reasoning problems through hybrid and decomposed strategies. Automated self-checking and manual screening are incorporated to ensure semantic clarity and a well-structured difficulty gradient in the synthesized data. Building on this, we construct the MathMixupQA dataset and design a curriculum learning strategy that leverages these graded problems, supporting flexible integration with other datasets. Experimental results show that MathMixup and its curriculum learning strategy significantly enhance the mathematical reasoning performance of LLMs. Fine-tuned Qwen2.5-7B achieves an average score of 52.6\% across seven mathematical benchmarks, surpassing previous state-of-the-art methods. These results fully validate the effectiveness and broad applicability of MathMixup in improving the mathematical reasoning abilities of LLMs and advancing data-centric curriculum learning. 

---
# From Noise to Insights: Enhancing Supply Chain Decision Support through AI-Based Survey Integrity Analytics 

**Authors**: Bhubalan Mani  

**Link**: [PDF](https://arxiv.org/pdf/2601.17005)  

**Abstract**: The reliability of survey data is crucial in supply chain decision-making, particularly when evaluating readiness for AI-driven tools such as safety stock optimization systems. However, surveys often attract low-effort or fake responses that degrade the accuracy of derived insights. This study proposes a lightweight AI-based framework for filtering unreliable survey inputs using a supervised machine learning approach. In this expanded study, a larger dataset of 99 industry responses was collected, with manual labeling to identify fake responses based on logical inconsistencies and response patterns. After preprocessing and label encoding, both Random Forest and baseline models (Logistic Regression, XGBoost) were trained to distinguish genuine from fake responses. The best-performing model achieved an 92.0% accuracy rate, demonstrating improved detection compared to the pilot study. Despite limitations, the results highlight the viability of integrating AI into survey pipelines and provide a scalable solution for improving data integrity in supply chain research, especially during product launch and technology adoption phases. 

---
# Investigating Self-regulated Learning Sequences within a Generative AI-based Intelligent Tutoring System 

**Authors**: Jie Gao, Shasha Li, Jianhua Zhang, Shan Li, Tingting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.17000)  

**Abstract**: There has been a growing trend in employing generative artificial intelligence (GenAI) techniques to support learning. Moreover, scholars have reached a consensus on the critical role of self-regulated learning (SRL) in ensuring learning effectiveness within GenAI-assisted learning environments, making it essential to capture students' dynamic SRL patterns. In this study, we extracted students' interaction patterns with GenAI from trace data as they completed a problem-solving task within a GenAI-assisted intelligent tutoring system. Students' purpose of using GenAI was also analyzed from the perspective of information processing, i.e., information acquisition and information transformation. Using sequential and clustering analysis, this study classified participants into two groups based on their SRL sequences. These two groups differed in the frequency and temporal characteristics of GenAI use. In addition, most students used GenAI for information acquisition rather than information transformation, while the correlation between the purpose of using GenAI and learning performance was not statistically significant. Our findings inform both pedagogical design and the development of GenAI-assisted learning environments. 

---
# BibAgent: An Agentic Framework for Traceable Miscitation Detection in Scientific Literature 

**Authors**: Peiran Li, Fangzhou Lin, Shuo Xing, Xiang Zheng, Xi Hong, Jiashuo Sun, Zhengzhong Tu, Chaoqun Ni  

**Link**: [PDF](https://arxiv.org/pdf/2601.16993)  

**Abstract**: Citations are the bedrock of scientific authority, yet their integrity is compromised by widespread miscitations: ranging from nuanced distortions to fabricated references. Systematic citation verification is currently unfeasible; manual review cannot scale to modern publishing volumes, while existing automated tools are restricted by abstract-only analysis or small-scale, domain-specific datasets in part due to the "paywall barrier" of full-text access. We introduce BibAgent, a scalable, end-to-end agentic framework for automated citation verification. BibAgent integrates retrieval, reasoning, and adaptive evidence aggregation, applying distinct strategies for accessible and paywalled sources. For paywalled references, it leverages a novel Evidence Committee mechanism that infers citation validity via downstream citation consensus. To support systematic evaluation, we contribute a 5-category Miscitation Taxonomy and MisciteBench, a massive cross-disciplinary benchmark comprising 6,350 miscitation samples spanning 254 fields. Our results demonstrate that BibAgent outperforms state-of-the-art Large Language Model (LLM) baselines in citation verification accuracy and interpretability, providing scalable, transparent detection of citation misalignments across the scientific literature. 

---
# Sparsity-Aware Low-Rank Representation for Efficient Fine-Tuning of Large Language Models 

**Authors**: Longteng Zhang, Sen Wu, Shuai Hou, Zhengyu Qing, Zhuo Zheng, Danning Ke, Qihong Lin, Qiang Wang, Shaohuai Shi, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2601.16991)  

**Abstract**: Adapting large pre-trained language models to downstream tasks often entails fine-tuning millions of parameters or deploying costly dense weight updates, which hinders their use in resource-constrained environments. Low-rank Adaptation (LoRA) reduces trainable parameters by factorizing weight updates, yet the underlying dense weights still impose high storage and computation costs. Magnitude-based pruning can yield sparse models but typically degrades LoRA's performance when applied naively. In this paper, we introduce SALR (Sparsity-Aware Low-Rank Representation), a novel fine-tuning paradigm that unifies low-rank adaptation with sparse pruning under a rigorous mean-squared-error framework. We prove that statically pruning only the frozen base weights minimizes the pruning error bound, and we recover the discarded residual information via a truncated-SVD low-rank adapter, which provably reduces per-entry MSE by a factor of $(1 - r/\min(d,k))$. To maximize hardware efficiency, we fuse multiple low-rank adapters into a single concatenated GEMM, and we adopt a bitmap-based encoding with a two-stage pipelined decoding + GEMM design to achieve true model compression and speedup. Empirically, SALR attains 50\% sparsity on various LLMs while matching the performance of LoRA on GSM8K and MMLU, reduces model size by $2\times$, and delivers up to a $1.7\times$ inference speedup. 

---
# Evaluating Reward Model Generalization via Pairwise Maximum Discrepancy Competitions 

**Authors**: Shunyang Luo, Peibei Cao, Zhihui Zhu, Kehua Feng, Zhihua Wang, Keyan Ding  

**Link**: [PDF](https://arxiv.org/pdf/2601.16987)  

**Abstract**: Reward models (RMs) are central to aligning large language models, yet their practical effectiveness hinges on generalization to unseen prompts and shifting distributions. Most existing RM evaluations rely on static, pre-annotated preference datasets, which provide limited coverage and often fail to faithfully assess generalization in open-world settings. We introduce Pairwise Maximum Discrepancy Competition (PMDC), a dynamic and annotation-efficient framework for evaluating RM generalization using a large, unlabeled, open-domain prompt pool. PMDC actively selects prompt--response pairs that maximize disagreement between two RMs, yielding a compact set of highly contentious test cases. These cases are adjudicated by an oracle, and the resulting outcomes are aggregated via a Bradley--Terry model to produce a global ranking and pairwise win-rate landscape of RMs. We apply PMDC to re-evaluate 10 representative RMs and observe substantial rank reshuffling compared with conventional benchmarks. Qualitative analyses further uncover systematic generalization failures, providing valuable insights for improving reward modeling. 

---
# Crystal-KV: Efficient KV Cache Management for Chain-of-Thought LLMs via Answer-First Principle 

**Authors**: Zihan Wang, Cheng Tang, Lei Gong, Cheng Li, Chao Wang, teng wang, Wenqi Lou, Xuehai Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2601.16986)  

**Abstract**: Chain-of-Thought (CoT) reasoning in large language models (LLMs) significantly improves accuracy on complex tasks, yet incurs excessive memory overhead due to the long think-stage sequences stored in the Key-Value (KV) cache. Unlike traditional generation tasks where all tokens are uniformly important, CoT emphasizes the final answer, rendering conventional KV compression strategies ineffective. In this paper, we present Crystal-KV, an efficient KV cache management framework tailored for CoT reasoning. Our key insight is the answer-first principle. By mapping answer preferences into think-stage attention map, we distinguish between SlipKV, which mainly maintains the reasoning flow but may occasionally introduce misleading context, and CrystalKV, which truly contributes to the correctness of the final answer. Next, we propose an attention-based Least Recently Frequently Used algorithm. It precisely identifies when a SlipKV entry's utility expires and evicts it, retaining CrystalKV without disrupting reasoning flow. Finally, we introduce an adaptive cache budget allocation algorithm. Based on the dynamic proportion of CrystalKV, it estimates the importance of each layer/head and adjusts the KV cache budget during inference, amplifying critical components to improve budget utilization. Results show that Crystal-KV achieves state-of-the-art KV cache compression, significantly improves throughput, and enables faster response time, while maintaining, or even improving, answer accuracy for CoT reasoning. 

---
# Breaking Task Impasses Quickly: Adaptive Neuro-Symbolic Learning for Open-World Robotics 

**Authors**: Pierrick Lorang  

**Link**: [PDF](https://arxiv.org/pdf/2601.16985)  

**Abstract**: Adapting to unforeseen novelties in open-world environments remains a major challenge for autonomous systems. While hybrid planning and reinforcement learning (RL) approaches show promise, they often suffer from sample inefficiency, slow adaptation, and catastrophic forgetting. We present a neuro-symbolic framework integrating hierarchical abstractions, task and motion planning (TAMP), and reinforcement learning to enable rapid adaptation in robotics. Our architecture combines symbolic goal-oriented learning and world model-based exploration to facilitate rapid adaptation to environmental changes. Validated in robotic manipulation and autonomous driving, our approach achieves faster convergence, improved sample efficiency, and superior robustness over state-of-the-art hybrid methods, demonstrating its potential for real-world deployment. 

---
# TelcoAI: Advancing 3GPP Technical Specification Search through Agentic Multi-Modal Retrieval-Augmented Generation 

**Authors**: Rahul Ghosh, Chun-Hao Liu, Gaurav Rele, Vidya Sagar Ravipati, Hazar Aouad  

**Link**: [PDF](https://arxiv.org/pdf/2601.16984)  

**Abstract**: The 3rd Generation Partnership Project (3GPP) produces complex technical specifications essential to global telecommunications, yet their hierarchical structure, dense formatting, and multi-modal content make them difficult to process. While Large Language Models (LLMs) show promise, existing approaches fall short in handling complex queries, visual information, and document interdependencies. We present TelcoAI, an agentic, multi-modal Retrieval-Augmented Generation (RAG) system tailored for 3GPP documentation. TelcoAI introduces section-aware chunking, structured query planning, metadata-guided retrieval, and multi-modal fusion of text and diagrams. Evaluated on multiple benchmarks-including expert-curated queries-our system achieves $87\%$ recall, $83\%$ claim recall, and $92\%$ faithfulness, representing a $16\%$ improvement over state-of-the-art baselines. These results demonstrate the effectiveness of agentic and multi-modal reasoning in technical document understanding, advancing practical solutions for real-world telecommunications research and engineering. 

---
