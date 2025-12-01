# Hierarchical AI-Meteorologist: LLM-Agent System for Multi-Scale and Explainable Weather Forecast Reporting 

**Authors**: Daniil Sukhorukov, Andrei Zakharov, Nikita Glazkov, Katsiaryna Yanchanka, Vladimir Kirilin, Maxim Dubovitsky, Roman Sultimov, Yuri Maksimov, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2511.23387)  

**Abstract**: We present the Hierarchical AI-Meteorologist, an LLM-agent system that generates explainable weather reports using a hierarchical forecast reasoning and weather keyword generation. Unlike standard approaches that treat forecasts as flat time series, our framework performs multi-scale reasoning across hourly, 6-hour, and daily aggregations to capture both short-term dynamics and long-term trends. Its core reasoning agent converts structured meteorological inputs into coherent narratives while simultaneously extracting a few keywords effectively summarizing the dominant meteorological events. These keywords serve as semantic anchors for validating consistency, temporal coherence and factual alignment of the generated reports. Using OpenWeather and Meteostat data, we demonstrate that hierarchical context and keyword-based validation substantially improve interpretability and robustness of LLM-generated weather narratives, offering a reproducible framework for semantic evaluation of automated meteorological reporting and advancing agent-based scientific reasoning. 

---
# Thinking by Doing: Building Efficient World Model Reasoning in LLMs via Multi-turn Interaction 

**Authors**: Bao Shu, Yan Cai, Jianjian Sun, Chunrui Han, En Yu, Liang Zhao, Jingcheng Hu, Yinmin Zhang, Haoran Lv, Yuang Peng, Zheng Ge, Xiangyu Zhang, Daxin Jiang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2511.23476)  

**Abstract**: Developing robust world model reasoning is crucial for large language model (LLM) agents to plan and interact in complex environments. While multi-turn interaction offers a superior understanding of environmental dynamics via authentic feedback, current approaches often impose a rigid reasoning process, which constrains the model's active learning, ultimately hindering efficient world model reasoning. To address these issues, we explore world-model internalization through efficient interaction and active reasoning (WMAct), which liberates the model from structured reasoning, allowing the model to shape thinking directly through its doing, and achieves effective and efficient world model reasoning with two key mechanisms: (1) a reward rescaling mechanism adjusting outcome reward based on action efficacy to incentivize redundancy reduction and purposeful interaction; (2) an interaction frequency annealing strategy to progressively reduce the maximum allowed interaction turns, which compels the model to condense its learning and internalize environmental dynamics rather than over-relying on environmental cues. Our experiments on Sokoban, Maze, and Taxi show that WMAct yields effective world model reasoning capable of resolving tasks in a single turn that previously required multiple interactions and fosters strong transferability to complex environments, improving performance on a suite of reasoning benchmarks. 

---
# TIM-PRM: Verifying multimodal reasoning with Tool-Integrated PRM 

**Authors**: Peng Kuang, Xiangxiang Wang, Wentao Liu, Jian Dong, Kaidi Xu, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22998)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performances in mathematical reasoning, yet they remain vulnerable to visual hallucinations and logical inconsistencies that standard outcome-based supervision fails to mitigate. While Process Reward Models (PRMs) promise step-by-step verification, current approaches typically operate as scalar scorers or generative critics that suffer from sycophancy, blindly validating the flawed hypotheses rather than grounding them in visual reality. To bridge this gap, we introduce TIM-PRM (Tool-Integrated Multimodal PRM), a novel agentic framework that transforms verification from a passive classification task into an active, tool-augmented investigation. TIM-PRM is trained to explicitly plan verification strategies and utilizes a mechanism of Independent Question Asking to query evidence via external tools, effectively decoupling verification from the reasoning context to eliminate confirmation bias. We instantiate this method by curating a high-quality dataset of tool-integrated verification trajectories. Extensive experiments on VisualProcessBench demonstrate that our 8B parameter model surpasses existing open-source multimodal PRMs, significantly outperforming much larger models like Qwen2.5-72B and InternVL-78B, while offering interpretable insights into the verification process. 

---
# InsightEval: An Expert-Curated Benchmark for Assessing Insight Discovery in LLM-Driven Data Agents 

**Authors**: Zhenghao Zhu, Yuanfeng Song, Xin Chen, Chengzhong Liu, Yakun Cui, Caleb Chen Cao, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.22884)  

**Abstract**: Data analysis has become an indispensable part of scientific research. To discover the latent knowledge and insights hidden within massive datasets, we need to perform deep exploratory analysis to realize their full value. With the advent of large language models (LLMs) and multi-agent systems, more and more researchers are making use of these technologies for insight discovery. However, there are few benchmarks for evaluating insight discovery capabilities. As one of the most comprehensive existing frameworks, InsightBench also suffers from many critical flaws: format inconsistencies, poorly conceived objectives, and redundant insights. These issues may significantly affect the quality of data and the evaluation of agents. To address these issues, we thoroughly investigate shortcomings in InsightBench and propose essential criteria for a high-quality insight benchmark. Regarding this, we develop a data-curation pipeline to construct a new dataset named InsightEval. We further introduce a novel metric to measure the exploratory performance of agents. Through extensive experiments on InsightEval, we highlight prevailing challenges in automated insight discovery and raise some key findings to guide future research in this promising direction. 

---
# ORION: Teaching Language Models to Reason Efficiently in the Language of Thought 

**Authors**: Kumar Tanmay, Kriti Aggarwal, Paul Pu Liang, Subhabrata Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2511.22891)  

**Abstract**: Large Reasoning Models (LRMs) achieve strong performance in mathematics, code generation, and task planning, but their reliance on long chains of verbose "thinking" tokens leads to high latency, redundancy, and incoherent reasoning paths. Inspired by the Language of Thought Hypothesis, which posits that human reasoning operates over a symbolic, compositional mental language called Mentalese, we introduce a framework that trains models to reason in a similarly compact style. Mentalese encodes abstract reasoning as ultra-compressed, structured tokens, enabling models to solve complex problems with far fewer steps. To improve both efficiency and accuracy, we propose SHORTER LENGTH PREFERENCE OPTIMIZATION (SLPO), a reinforcement learning method that rewards concise solutions that stay correct, while still allowing longer reasoning when needed. Applied to Mentalese-aligned models, SLPO yields significantly higher compression rates by enabling concise reasoning that preserves the benefits of detailed thinking without the computational overhead. Across benchmarks including AIME 2024 and 2025, MinervaMath, OlympiadBench, Math500, and AMC, our ORION models produce reasoning traces with 4-16x fewer tokens, achieve up to 5x lower inference latency, and reduce training costs by 7-9x relative to the DeepSeek R1 Distilled model, while maintaining 90-98% of its accuracy. ORION also surpasses Claude and ChatGPT-4o by up to 5% in accuracy while maintaining 2x compression. These results show that Mentalese-style compressed reasoning offers a step toward human-like cognitive efficiency, enabling real-time, cost-effective reasoning without sacrificing accuracy. 

---
# DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning 

**Authors**: Zhihong Shao, Yuxiang Luo, Chengda Lu, Z.Z. Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, Xiaokang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22570)  

**Abstract**: Large language models have made significant progress in mathematical reasoning, which serves as an important testbed for AI and could impact scientific research if further advanced. By scaling reasoning with reinforcement learning that rewards correct final answers, LLMs have improved from poor performance to saturating quantitative reasoning competitions like AIME and HMMT in one year. However, this approach faces fundamental limitations. Pursuing higher final answer accuracy doesn't address a key issue: correct answers don't guarantee correct reasoning. Moreover, many mathematical tasks like theorem proving require rigorous step-by-step derivation rather than numerical answers, making final answer rewards inapplicable. To push the limits of deep reasoning, we believe it is necessary to verify the comprehensiveness and rigor of mathematical reasoning. Self-verification is particularly important for scaling test-time compute, especially for open problems without known solutions. Towards self-verifiable mathematical reasoning, we investigate how to train an accurate and faithful LLM-based verifier for theorem proving. We then train a proof generator using the verifier as the reward model, and incentivize the generator to identify and resolve as many issues as possible in their own proofs before finalizing them. To maintain the generation-verification gap as the generator becomes stronger, we propose to scale verification compute to automatically label new hard-to-verify proofs, creating training data to further improve the verifier. Our resulting model, DeepSeekMath-V2, demonstrates strong theorem-proving capabilities, achieving gold-level scores on IMO 2025 and CMO 2024 and a near-perfect 118/120 on Putnam 2024 with scaled test-time compute. 

---
# Evolutionary Discovery of Heuristic Policies for Traffic Signal Control 

**Authors**: Ruibing Wang, Shuhan Guo, Zeen Li, Zhen Wang, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.23122)  

**Abstract**: Traffic Signal Control (TSC) involves a challenging trade-off: classic heuristics are efficient but oversimplified, while Deep Reinforcement Learning (DRL) achieves high performance yet suffers from poor generalization and opaque policies. Online Large Language Models (LLMs) provide general reasoning but incur high latency and lack environment-specific optimization. To address these issues, we propose Temporal Policy Evolution for Traffic (\textbf{\method{}}), which uses LLMs as an evolution engine to derive specialized heuristic policies. The framework introduces two key modules: (1) Structured State Abstraction (SSA), converting high-dimensional traffic data into temporal-logical facts for reasoning; and (2) Credit Assignment Feedback (CAF), tracing flawed micro-decisions to poor macro-outcomes for targeted critique. Operating entirely at the prompt level without training, \method{} yields lightweight, robust policies optimized for specific traffic environments, outperforming both heuristics and online LLM actors. 

---
# Towards Continuous Intelligence Growth: Self-Training, Continual Learning, and Dual-Scale Memory in SuperIntelliAgent 

**Authors**: Jianzhe Lin, Zeyu Pan, Yun Zhu, Ruiqi Song, Jining Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.23436)  

**Abstract**: We introduce SuperIntelliAgent, an agentic learning framework that couples a trainable small diffusion model (the learner) with a frozen large language model (the verifier) to enable continual intelligence growth through self-supervised interaction. Unlike conventional supervised fine-tuning, SuperIntelliAgent learns autonomously without annotation: the learner generates candidate outputs, the verifier evaluates them through step-by-step reasoning, and their interaction produces chosen/rejected pairs for Direct Preference Optimization (DPO). This converts each input into a pseudo-training signal for continual improvement. The framework integrates dual-scale memory: short-term in-context memory that preserves reasoning traces across refinement cycles, and long-term memory that consolidates acquired knowledge through lightweight on-the-fly fine-tuning. A replay buffer retains samples that show verifiable progress and replays them as auxiliary supervision, reinforcing recent learning while forming adaptive curricula. SuperIntelliAgent is infrastructure-agnostic and can be plugged into existing agentic frameworks while turning ordinary inference loops into a lifelong optimization process. We posit that pairing a trainable learner with a reasoning-capable verifier forms a minimal reliable unit of growing intelligence, as paired feedback and partial-history replay yield richer learning curricula and stronger preference alignment. With a small number of automatically generated DPO pairs, the learner improves across all benchmarks, indicating that this mechanism provides a promising direction for continual intelligence accumulation and real-world deployment. 

---
# Enhanced Conditional Generation of Double Perovskite by Knowledge-Guided Language Model Feedback 

**Authors**: Inhyo Lee, Junhyeong Lee, Jongwon Park, KyungTae Lim, Seunghwa Ryu  

**Link**: [PDF](https://arxiv.org/pdf/2511.22307)  

**Abstract**: Double perovskites (DPs) are promising candidates for sustainable energy technologies due to their compositional tunability and compatibility with low-energy fabrication, yet their vast design space poses a major challenge for conditional materials discovery. This work introduces a multi-agent, text gradient-driven framework that performs DP composition generation under natural-language conditions by integrating three complementary feedback sources: LLM-based self-evaluation, DP-specific domain knowledge-informed feedback, and ML surrogate-based feedback. Analogous to how knowledge-informed machine learning improves the reliability of conventional data-driven models, our framework incorporates domain-informed text gradients to guide the generative process toward physically meaningful regions of the DP composition space. Systematic comparison of three incremental configurations, (i) pure LLM generation, (ii) LLM generation with LLM reasoning-based feedback, and (iii) LLM generation with domain knowledge-guided feedback, shows that iterative guidance from knowledge-informed gradients improves stability-condition satisfaction without additional training data, achieving over 98% compositional validity and up to 54% stable or metastable candidates, surpassing both the LLM-only baseline (43%) and prior GAN-based results (27%). Analyses of ML-based gradients further reveal that they enhance performance in in-distribution (ID) regions but become unreliable in out-of-distribution (OOD) regimes. Overall, this work provides the first systematic analysis of multi-agent, knowledge-guided text gradients for DP discovery and establishes a generalizable blueprint for MAS-driven generative materials design aimed at advancing sustainable technologies. 

---
# Solving Context Window Overflow in AI Agents 

**Authors**: Anton Bulle Labate, Valesca Moura de Sousa, Sandro Rama Fiorini, Leonardo Guerreiro Azevedo, Raphael Melo Thiago, Viviane Torres da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2511.22729)  

**Abstract**: Large Language Models (LLMs) have become increasingly capable of interacting with external tools, granting access to specialized knowledge beyond their training data - critical in dynamic, knowledge-intensive domains such as Chemistry and Materials Science. However, large tool outputs can overflow the LLMs' context window, preventing task completion. Existing solutions such as truncation or summarization fail to preserve complete outputs, making them unsuitable for workflows requiring the full data. This work introduces a method that enables LLMs to process and utilize tool responses of arbitrary length without loss of information. By shifting the model's interaction from raw data to memory pointers, the method preserves tool functionality, allows seamless integration into agentic workflows, and reduces token usage and execution time. The proposed method is validated on a real-world Materials Science application that cannot be executed with conventional workflows, and its effectiveness is demonstrated via a comparative analysis where both methods succeed. In this experiment, the proposed approach consumed approximately seven times fewer tokens than the traditional workflow. 

---
# Evaluating Strategies for Synthesizing Clinical Notes for Medical Multimodal AI 

**Authors**: Niccolo Marini, Zhaohui Liang, Sivaramakrishnan Rajaraman, Zhiyun Xue, Sameer Antani  

**Link**: [PDF](https://arxiv.org/pdf/2511.21827)  

**Abstract**: Multimodal (MM) learning is emerging as a promising paradigm in biomedical artificial intelligence (AI) applications, integrating complementary modality, which highlight different aspects of patient health. The scarcity of large heterogeneous biomedical MM data has restrained the development of robust models for medical AI applications. In the dermatology domain, for instance, skin lesion datasets typically include only images linked to minimal metadata describing the condition, thereby limiting the benefits of MM data integration for reliable and generalizable predictions. Recent advances in Large Language Models (LLMs) enable the synthesis of textual description of image findings, potentially allowing the combination of image and text representations. However, LLMs are not specifically trained for use in the medical domain, and their naive inclusion has raised concerns about the risk of hallucinations in clinically relevant contexts. This work investigates strategies for generating synthetic textual clinical notes, in terms of prompt design and medical metadata inclusion, and evaluates their impact on MM architectures toward enhancing performance in classification and cross-modal retrieval tasks. Experiments across several heterogeneous dermatology datasets demonstrate that synthetic clinical notes not only enhance classification performance, particularly under domain shift, but also unlock cross-modal retrieval capabilities, a downstream task that is not explicitly optimized during training. 

---
# Swarms of Large Language Model Agents for Protein Sequence Design with Experimental Validation 

**Authors**: Fiona Y. Wang, Di Sheng Lee, David L. Kaplan, Markus J. Buehler  

**Link**: [PDF](https://arxiv.org/pdf/2511.22311)  

**Abstract**: Designing proteins de novo with tailored structural, physicochemical, and functional properties remains a grand challenge in biotechnology, medicine, and materials science, due to the vastness of sequence space and the complex coupling between sequence, structure, and function. Current state-of-the-art generative methods, such as protein language models (PLMs) and diffusion-based architectures, often require extensive fine-tuning, task-specific data, or model reconfiguration to support objective-directed design, thereby limiting their flexibility and scalability. To overcome these limitations, we present a decentralized, agent-based framework inspired by swarm intelligence for de novo protein design. In this approach, multiple large language model (LLM) agents operate in parallel, each assigned to a specific residue position. These agents iteratively propose context-aware mutations by integrating design objectives, local neighborhood interactions, and memory and feedback from previous iterations. This position-wise, decentralized coordination enables emergent design of diverse, well-defined sequences without reliance on motif scaffolds or multiple sequence alignments, validated with experiments on proteins with alpha helix and coil structures. Through analyses of residue conservation, structure-based metrics, and sequence convergence and embeddings, we demonstrate that the framework exhibits emergent behaviors and effective navigation of the protein fitness landscape. Our method achieves efficient, objective-directed designs within a few GPU-hours and operates entirely without fine-tuning or specialized training, offering a generalizable and adaptable solution for protein design. Beyond proteins, the approach lays the groundwork for collective LLM-driven design across biomolecular systems and other scientific discovery tasks. 

---
# RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems 

**Authors**: Mengfan Li, Xuanhua Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2511.22275)  

**Abstract**: Large Language models are revolutionizing the conversational recommender systems through their impressive capabilities in instruction comprehension, reasoning, and human interaction. A core factor underlying effective recommendation dialogue is the ability to infer and reason about users' mental states (such as desire, intention, and belief), a cognitive capacity commonly referred to as Theory of Mind. Despite growing interest in evaluating ToM in LLMs, current benchmarks predominantly rely on synthetic narratives inspired by Sally-Anne test, which emphasize physical perception and fail to capture the complexity of mental state inference in realistic conversational settings. Moreover, existing benchmarks often overlook a critical component of human ToM: behavioral prediction, the ability to use inferred mental states to guide strategic decision-making and select appropriate conversational actions for future interactions. To better align LLM-based ToM evaluation with human-like social reasoning, we propose RecToM, a novel benchmark for evaluating ToM abilities in recommendation dialogues. RecToM focuses on two complementary dimensions: Cognitive Inference and Behavioral Prediction. The former focus on understanding what has been communicated by inferring the underlying mental states. The latter emphasizes what should be done next, evaluating whether LLMs can leverage these inferred mental states to predict, select, and assess appropriate dialogue strategies. Extensive experiments on state-of-the-art LLMs demonstrate that RecToM poses a significant challenge. While the models exhibit partial competence in recognizing mental states, they struggle to maintain coherent, strategic ToM reasoning throughout dynamic recommendation dialogues, particularly in tracking evolving intentions and aligning conversational strategies with inferred mental states. 

---
# Evaluating LLMs for One-Shot Patching of Real and Artificial Vulnerabilities 

**Authors**: Aayush Garg, Zanis Ali Khan, Renzo Degiovanni, Qiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2511.23408)  

**Abstract**: Automated vulnerability patching is crucial for software security, and recent advancements in Large Language Models (LLMs) present promising capabilities for automating this task. However, existing research has primarily assessed LLMs using publicly disclosed vulnerabilities, leaving their effectiveness on related artificial vulnerabilities largely unexplored. In this study, we empirically evaluate the patching effectiveness and complementarity of several prominent LLMs, such as OpenAI's GPT variants, LLaMA, DeepSeek, and Mistral models, using both real and artificial vulnerabilities. Our evaluation employs Proof-of-Vulnerability (PoV) test execution to concretely assess whether LLM-generated source code successfully patches vulnerabilities. Our results reveal that LLMs patch real vulnerabilities more effectively compared to artificial ones. Additionally, our analysis reveals significant variability across LLMs in terms of overlapping (multiple LLMs patching the same vulnerabilities) and complementarity (vulnerabilities patched exclusively by a single LLM), emphasizing the importance of selecting appropriate LLMs for effective vulnerability patching. 

---
# Towards Improving Interpretability of Language Model Generation through a Structured Knowledge Discovery Approach 

**Authors**: Shuqi Liu, Han Wu, Guanzhi Deng, Jianshu Chen, Xiaoyang Wang, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2511.23335)  

**Abstract**: Knowledge-enhanced text generation aims to enhance the quality of generated text by utilizing internal or external knowledge sources. While language models have demonstrated impressive capabilities in generating coherent and fluent text, the lack of interpretability presents a substantial obstacle. The limited interpretability of generated text significantly impacts its practical usability, particularly in knowledge-enhanced text generation tasks that necessitate reliability and explainability. Existing methods often employ domain-specific knowledge retrievers that are tailored to specific data characteristics, limiting their generalizability to diverse data types and tasks. To overcome this limitation, we directly leverage the two-tier architecture of structured knowledge, consisting of high-level entities and low-level knowledge triples, to design our task-agnostic structured knowledge hunter. Specifically, we employ a local-global interaction scheme for structured knowledge representation learning and a hierarchical transformer-based pointer network as the backbone for selecting relevant knowledge triples and entities. By combining the strong generative ability of language models with the high faithfulness of the knowledge hunter, our model achieves high interpretability, enabling users to comprehend the model output generation process. Furthermore, we empirically demonstrate the effectiveness of our model in both internal knowledge-enhanced table-to-text generation on the RotoWireFG dataset and external knowledge-enhanced dialogue response generation on the KdConv dataset. Our task-agnostic model outperforms state-of-the-art methods and corresponding language models, setting new standards on the benchmark. 

---
# Multi-chain Graph Refinement and Selection for Reliable Reasoning in Large Language Models 

**Authors**: Yujiao Yang, Jing Lian, Linhui Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.23136)  

**Abstract**: The complex reasoning ability of Large Language Models (LLMs) poses a critical bottleneck for their practical applications. Test-time expansion methods such as Tree-of-Thought (ToT) and Graph-of-Thought (GoT) enhance reasoning by introducing intermediate reasoning structures, tree search, or graph-based exploration mechanisms. However, their reasoning strategies suffer from limited diversity, redundant search branches, and inadequate integration and error correction across heterogeneous reasoning paths. To address these limitations, we propose a novel reasoning framework called Multi-chain Graph Refinement & Selection (MGRS), which first generates multiple diverse reasoning trajectories for a given problem, refines candidate responses using a composite self- and cross-verification strategy, then constructs a reasoning relation graph and estimates the success rate of intermediate nodes, and finally computes cumulative success rates to select the most reliable answer and corresponding reasoning trajectory. Experimental results demonstrate that MGRS significantly advances both the reasoning capability and computational efficiency of reasoning enhancement methods. Across six benchmark datasets spanning four distinct tasks, MGRS achieves an average accuracy of 82.9%, outperforming state-of-the-art baselines by a clear margin of 2.1%. Remarkably, on the 24-point game, MGRS attains 100% accuracy for the first time, while delivering a 13.6x speed-up compared to the leading Forest of Thoughts framework. 

---
# Conveying Imagistic Thinking in TCM Translation: A Prompt Engineering and LLM-Based Evaluation Framework 

**Authors**: Jiatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.23059)  

**Abstract**: Traditional Chinese Medicine theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis. Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient and replicable HITL methodological pathway for translation of ancient, concept-dense texts like TCM. 

---
# Mind Reading or Misreading? LLMs on the Big Five Personality Test 

**Authors**: Francesco Di Cursi, Chiara Boldrini, Marco Conti, Andrea Passarella  

**Link**: [PDF](https://arxiv.org/pdf/2511.23101)  

**Abstract**: We evaluate large language models (LLMs) for automatic personality prediction from text under the binary Five Factor Model (BIG5). Five models -- including GPT-4 and lightweight open-source alternatives -- are tested across three heterogeneous datasets (Essays, MyPersonality, Pandora) and two prompting strategies (minimal vs. enriched with linguistic and psychological cues). Enriched prompts reduce invalid outputs and improve class balance, but also introduce a systematic bias toward predicting trait presence. Performance varies substantially: Openness and Agreeableness are relatively easier to detect, while Extraversion and Neuroticism remain challenging. Although open-source models sometimes approach GPT-4 and prior benchmarks, no configuration yields consistently reliable predictions in zero-shot binary settings. Moreover, aggregate metrics such as accuracy and macro-F1 mask significant asymmetries, with per-class recall offering clearer diagnostic value. These findings show that current out-of-the-box LLMs are not yet suitable for APPT, and that careful coordination of prompt design, trait framing, and evaluation metrics is essential for interpretable results. 

---
# Automated Generation of MDPs Using Logic Programming and LLMs for Robotic Applications 

**Authors**: Enrico Saccon, Davide De Martini, Matteo Saveriano, Edoardo Lamon, Luigi Palopoli, Marco Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2511.23143)  

**Abstract**: We present a novel framework that integrates Large Language Models (LLMs) with automated planning and formal verification to streamline the creation and use of Markov Decision Processes (MDP). Our system leverages LLMs to extract structured knowledge in the form of a Prolog knowledge base from natural language (NL) descriptions. It then automatically constructs an MDP through reachability analysis, and synthesises optimal policies using the Storm model checker. The resulting policy is exported as a state-action table for execution. We validate the framework in three human-robot interaction scenarios, demonstrating its ability to produce executable policies with minimal manual effort. This work highlights the potential of combining language models with formal methods to enable more accessible and scalable probabilistic planning in robotics. 

---
# Adversarial Training for Process Reward Models 

**Authors**: Gurusha Juneja, Deepak Nathani, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22888)  

**Abstract**: Process Reward Models (PRMs) enhance reasoning ability of LLMs by providing step-level supervision. However, their widespread adoption is limited due to expensive manual step-level annotation and poor generalization of static training data to novel errors. We introduce Adversarially Trained PRMs (\texttt{APRM}), where a Generator ($G$) learns to produce reasoning errors to deceive a PRM ($R$), while $R$ concurrently learns to detect them. This interaction yields progressively harder negatives for $R$, improving its robustness and generalization to novel errors without requiring manual step-level labels. Averaged across diverse mathematical reasoning benchmarks, \texttt{APRM} improves solver accuracy by $+3.4$ percentage points (pp) over the strongest PRM baseline. \texttt{APRM} achieves gains of $+5.3$ pp on out-of-distribution tasks. 

---
# SpaceMind: Camera-Guided Modality Fusion for Spatial Reasoning in Vision-Language Models 

**Authors**: Ruosen Zhao, Zhikang Zhang, Jialei Xu, Jiahao Chang, Dong Chen, Lingyun Li, Weijian Sun, Zizhuang Wei  

**Link**: [PDF](https://arxiv.org/pdf/2511.23075)  

**Abstract**: Large vision-language models (VLMs) show strong multimodal understanding but still struggle with 3D spatial reasoning, such as distance estimation, size comparison, and cross-view consistency. Existing 3D-aware methods either depend on auxiliary 3D information or enhance RGB-only VLMs with geometry encoders through shallow feature fusion. We propose SpaceMind, a multimodal large language model explicitly designed for spatial reasoning solely from RGB inputs. The model adopts a dual-encoder architecture, integrating VGGT as a spatial understanding encoder and InternViT as a 2D visual encoder. The key idea is to treat the camera representation as an active guiding modality rather than passive metadata. Specifically, SpaceMind introduces a lightweight Camera-Guided Modality Fusion module before the language model to replace shallow fusion. It applies camera-conditioned biasing to spatial tokens, assigns query-independent weights reflecting their geometric importance, and uses the camera embedding to gate the fused representation. Empirically, SpaceMind establishes new state-of-the-art results on VSI-Bench, SQA3D and SPBench, surpassing both open and proprietary systems on VSI-Bench and SPBench by large margins and achieving state-of-the-art performance on SQA3D. These results demonstrate that camera-guided modality fusion is an effective and practical inductive bias for equipping VLMs with genuinely spatially grounded intelligence. We will release code and model checkpoints to support future research. 

---
# AgentShield: Make MAS more secure and efficient 

**Authors**: Kaixiang Wang, Zhaojiacheng Zhou, Bunyod Suvonov, Jiong Lou, Jie LI  

**Link**: [PDF](https://arxiv.org/pdf/2511.22924)  

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) offer powerful cooperative reasoning but remain vulnerable to adversarial attacks, where compromised agents can undermine the system's overall performance. Existing defenses either depend on single trusted auditors, creating single points of failure, or sacrifice efficiency for robustness. To resolve this tension, we propose \textbf{AgentShield}, a distributed framework for efficient, decentralized auditing. AgentShield introduces a novel three-layer defense: \textbf{(i) Critical Node Auditing} prioritizes high-influence agents via topological analysis; \textbf{(ii) Light Token Auditing} implements a cascade protocol using lightweight sentry models for rapid discriminative verification; and \textbf{(iii) Two-Round Consensus Auditing} triggers heavyweight arbiters only upon uncertainty to ensure global agreement. This principled design optimizes the robustness-efficiency trade-off. Experiments demonstrate that AgentShield achieves a 92.5\% recovery rate and reduces auditing overhead by over 70\% compared to existing methods, maintaining high collaborative accuracy across diverse MAS topologies and adversarial scenarios. 

---
# VeriDispatcher: Multi-Model Dispatching through Pre-Inference Difficulty Prediction for RTL Generation Optimization 

**Authors**: Zeng Wang, Weihua Xiao, Minghao Shao, Raghu Vamshi Hemadri, Ozgur Sinanoglu, Muhammad Shafique, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2511.22749)  

**Abstract**: Large Language Models (LLMs) show strong performance in RTL generation, but different models excel on different tasks because of architecture and training differences. Prior work mainly prompts or finetunes a single model. What remains not well studied is how to coordinate multiple different LLMs so they jointly improve RTL quality while also reducing cost, instead of running all models and choosing the best output. We define this as the multi-LLM RTL generation problem. We propose VeriDispatcher, a multi-LLM RTL generation framework that dispatches each RTL task to suitable LLMs based on pre-inference difficulty prediction. For each model, we train a compact classifier over semantic embeddings of task descriptions, using difficulty scores derived from benchmark variants that combine syntax, structural similarity, and functional correctness. At inference, VeriDispatcher uses these predictors to route tasks to a selected subset of LLMs. Across 10 diverse LLMs on RTLLM and VerilogEval, VeriDispatcher achieves up to 18% accuracy improvement on RTLLM using only 40% of commercial calls, and on VerilogEval maintains accuracy while reducing commercial usage by 25%, enabling cost-effective, high-quality LLM deployment in hardware design automation. 

---
# Automated Design Optimization via Strategic Search with Large Language Models 

**Authors**: Anthony Carreon, Vansh Sharma, Venkat Raman  

**Link**: [PDF](https://arxiv.org/pdf/2511.22651)  

**Abstract**: Traditional optimization methods excel in well-defined search spaces but struggle with design problems where transformations and design parameters are difficult to define. Large language models (LLMs) offer a promising alternative by dynamically interpreting design spaces and leveraging encoded domain knowledge. To this end, we introduce AUTO, an LLM agent framework that treats design optimization as a gradient-free search problem guided by strategic LLM reasoning. The framework employs two collaborative agents: a Strategist that selects between exploration and exploitation strategies, and an Implementor that executes detailed designs. Applied to GPU code optimization -- a domain critical to fields from machine learning to scientific computing -- AUTO generates solutions competitive with expert implementations for chemical kinetics integration and dense matrix multiplication. The framework achieves 50-70% search efficiency relative to Bayesian optimization methodologies. It completes optimizations in approximately 8 hours at an estimated cost of up to \$159 per run, compared to an estimated cost of up to \$480 with median-wage software developers. These findings open the door to automating design optimization in ill-defined search spaces with limited prior information. 

---
# Mapping Clinical Doubt: Locating Linguistic Uncertainty in LLMs 

**Authors**: Srivarshinee Sridhar, Raghav Kaushik Ravi, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2511.22402)  

**Abstract**: Large Language Models (LLMs) are increasingly used in clinical settings, where sensitivity to linguistic uncertainty can influence diagnostic interpretation and decision-making. Yet little is known about where such epistemic cues are internally represented within these models. Distinct from uncertainty quantification, which measures output confidence, this work examines input-side representational sensitivity to linguistic uncertainty in medical text. We curate a contrastive dataset of clinical statements varying in epistemic modality (e.g., 'is consistent with' vs. 'may be consistent with') and propose Model Sensitivity to Uncertainty (MSU), a layerwise probing metric that quantifies activation-level shifts induced by uncertainty cues. Our results show that LLMs exhibit structured, depth-dependent sensitivity to clinical uncertainty, suggesting that epistemic information is progressively encoded in deeper layers. These findings reveal how linguistic uncertainty is internally represented in LLMs, offering insight into their interpretability and epistemic reliability. 

---
# A Theoretically Grounded Hybrid Ensemble for Reliable Detection of LLM-Generated Text 

**Authors**: Sepyan Purnama Kristanto, Lutfi Hakim  

**Link**: [PDF](https://arxiv.org/pdf/2511.22153)  

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has blurred the line between human and machine authorship, creating practical risks for academic integrity and information reliability. Existing text detectors typically rely on a single methodological paradigm and suffer from poor generalization and high false positive rates (FPR), especially on high-stakes academic text. We propose a theoretically grounded hybrid ensemble that systematically fuses three complementary detection paradigms: (i) a RoBERTa-based transformer classifier for deep semantic feature extraction, (ii) a GPT-2-based probabilistic detector using perturbation-induced likelihood curvature, and (iii) a statistical linguistic feature analyzer capturing stylometric patterns. The core novelty lies in an optimized weighted voting framework, where ensemble weights are learned on the probability simplex to maximize F1-score rather than set heuristically. We provide a bias-variance analysis and empirically demonstrate low inter-model correlation (rho ~ 0.35-0.42), a key condition for variance reduction. Evaluated on a large-scale, multigenerator corpus of 30,000 documents, our system achieves 94.2% accuracy and an AUC of 0.978, with a 35% relative reduction in false positives on academic text. This yields a more reliable and ethically responsible detector for real-world deployment in education and other high-stakes domains. 

---
# From Compound Figures to Composite Understanding: Developing a Multi-Modal LLM from Biomedical Literature with Medical Multiple-Image Benchmarking and Validation 

**Authors**: Zhen Chen, Yihang Fu, Gabriel Madera, Mauro Giuffre, Serina Applebaum, Hyunjae Kim, Hua Xu, Qingyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.22232)  

**Abstract**: Multi-modal large language models (MLLMs) have shown promise in advancing healthcare. However, most existing models remain confined to single-image understanding, which greatly limits their applicability in clinical workflows. In practice, medical diagnosis and progression often require synthesizing information across multiple images from different modalities or time points. The development of medical MLLMs capable of such multi-image understanding has been hindered by the lack of large-scale, high-quality annotated training data. To address this limitation, we propose a novel framework that leverages license-permissive compound images in biomedical literature, as a rich yet underutilized data source for multi-image analysis. Specifically, we design a five-stage, context-aware instruction generation paradigm underpinned by a divide-and-conquer strategy. By decomposing multi-image analysis into manageable sub-tasks, this paradigm empowers MLLMs to move beyond single-panel analysis and provide a composite understanding by learning the complex spatial, temporal, and cross-modal relationships inherent in these compound figures. By parsing over 237,000 compound figures and their contextual text for instruction generation, we develop M3LLM, a medical multi-image multi-modal large language model. For benchmarking, we construct PMC-MI-Bench for composite understanding, manually validated by medical experts. Extensive experiments show that M3LLM significantly outperforms both general-purpose and specialized medical MLLMs across multi-image, single-image, text-only, and multi-choice scenarios. Notably, M3LLM exhibits strong generalization to longitudinal chest X-ray analysis using the MIMIC dataset. This work establishes a scalable and efficient paradigm for developing medical MLLMs capable of composite reasoning, bridging the gap between biomedical literature and real-world clinical applications. 

---
# Distillability of LLM Security Logic: Predicting Attack Success Rate of Outline Filling Attack via Ranking Regression 

**Authors**: Tianyu Zhang, Zihang Xi, Jingyu Hua, Sheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.22044)  

**Abstract**: In the realm of black-box jailbreak attacks on large language models (LLMs), the feasibility of constructing a narrow safety proxy, a lightweight model designed to predict the attack success rate (ASR) of adversarial prompts, remains underexplored. This work investigates the distillability of an LLM's core security logic. We propose a novel framework that incorporates an improved outline filling attack to achieve dense sampling of the model's security boundaries. Furthermore, we introduce a ranking regression paradigm that replaces standard regression and trains the proxy model to predict which prompt yields a higher ASR. Experimental results show that our proxy model achieves an accuracy of 91.1 percent in predicting the relative ranking of average long response (ALR), and 69.2 percent in predicting ASR. These findings confirm the predictability and distillability of jailbreak behaviors, and demonstrate the potential of leveraging such distillability to optimize black-box attacks. 

---
# Focused Chain-of-Thought: Efficient LLM Reasoning via Structured Input Information 

**Authors**: Lukas Struppek, Dominik Hintersdorf, Hannah Struppek, Daniel Neider, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2511.22176)  

**Abstract**: Recent large language models achieve strong reasoning performance by generating detailed chain-of-thought traces, but this often leads to excessive token use and high inference latency. Existing efficiency approaches typically focus on model-centric interventions, such as reinforcement learning or supervised fine-tuning, to reduce verbosity. In contrast, we propose a training-free, input-centric approach. Inspired by cognitive psychology, we introduce Focused Chain-of-Thought (F-CoT), which separates information extraction from the reasoning process. F-CoT first organizes the essential information from a query into a concise, structured context and then guides the model to reason exclusively over this context. By preventing attention to irrelevant details, F-CoT naturally produces shorter reasoning paths. On arithmetic word problems, F-CoT reduces generated tokens by 2-3x while maintaining accuracy comparable to standard zero-shot CoT. These results highlight structured input as a simple yet effective lever for more efficient LLM reasoning. 

---
# BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands 

**Authors**: Seongwon Cho, Daechul Ahn, Donghyun Shin, Hyeonbeom Choi, San Kim, Jonghyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.22364)  

**Abstract**: Open-vocabulary mobile manipulation (OVMM) requires robots to follow language instructions, navigate, and manipulate while updating their world representation under dynamic environmental changes. However, most prior approaches update their world representation only at discrete update points such as navigation targets, waypoints, or the end of an action step, leaving robots blind between updates and causing cascading failures: overlooked objects, late error detection, and delayed replanning. To address this limitation, we propose BINDER (Bridging INstant and DEliberative Reasoning), a dual process framework that decouples strategic planning from continuous environment monitoring. Specifically, BINDER integrates a Deliberative Response Module (DRM, a multimodal LLM for task planning) with an Instant Response Module (IRM, a VideoLLM for continuous monitoring). The two modules play complementary roles: the DRM performs strategic planning with structured 3D scene updates and guides what the IRM attends to, while the IRM analyzes video streams to update memory, correct ongoing actions, and trigger replanning when necessary. Through this bidirectional coordination, the modules address the trade off between maintaining awareness and avoiding costly updates, enabling robust adaptation under dynamic conditions. Evaluated in three real world environments with dynamic object placement, BINDER achieves substantially higher success and efficiency than SoTA baselines, demonstrating its effectiveness for real world deployment. 

---
# Toward Automated and Trustworthy Scientific Analysis and Visualization with LLM-Generated Code 

**Authors**: Apu Kumar Chakroborti, Yi Ding, Lipeng Wan  

**Link**: [PDF](https://arxiv.org/pdf/2511.21920)  

**Abstract**: As modern science becomes increasingly data-intensive, the ability to analyze and visualize large-scale, complex datasets is critical to accelerating discovery. However, many domain scientists lack the programming expertise required to develop custom data analysis workflows, creating barriers to timely and effective insight. Large language models (LLMs) offer a promising solution by generating executable code from natural language descriptions. In this paper, we investigate the trustworthiness of open-source LLMs in autonomously producing Python scripts for scientific data analysis and visualization. We construct a benchmark suite of domain-inspired prompts that reflect real-world research tasks and systematically evaluate the executability and correctness of the generated code. Our findings show that, without human intervention, the reliability of LLM-generated code is limited, with frequent failures caused by ambiguous prompts and the models' insufficient understanding of domain-specific contexts. To address these challenges, we design and assess three complementary strategies: data-aware prompt disambiguation, retrieval-augmented prompt enhancement, and iterative error repair. While these methods significantly improve execution success rates and output quality, further refinement is needed. This work highlights both the promise and current limitations of LLM-driven automation in scientific workflows and introduces actionable techniques and a reusable benchmark for building more inclusive, accessible, and trustworthy AI-assisted research tools. 

---
# Prompted Policy Search: Reinforcement Learning through Linguistic and Numerical Reasoning in LLMs 

**Authors**: Yifan Zhou, Sachin Grover, Mohamed El Mistiri, Kamalesh Kalirathnam, Pratyush Kerhalkar, Swaroop Mishra, Neelesh Kumar, Sanket Gaurav, Oya Aran, Heni Ben Amor  

**Link**: [PDF](https://arxiv.org/pdf/2511.21928)  

**Abstract**: Reinforcement Learning (RL) traditionally relies on scalar reward signals, limiting its ability to leverage the rich semantic knowledge often available in real-world tasks. In contrast, humans learn efficiently by combining numerical feedback with language, prior knowledge, and common sense. We introduce Prompted Policy Search (ProPS), a novel RL method that unifies numerical and linguistic reasoning within a single framework. Unlike prior work that augment existing RL components with language, ProPS places a large language model (LLM) at the center of the policy optimization loop-directly proposing policy updates based on both reward feedback and natural language input. We show that LLMs can perform numerical optimization in-context, and that incorporating semantic signals, such as goals, domain knowledge, and strategy hints can lead to more informed exploration and sample-efficient learning. ProPS is evaluated across fifteen Gymnasium tasks, spanning classic control, Atari games, and MuJoCo environments, and compared to seven widely-adopted RL algorithms (e.g., PPO, SAC, TRPO). It outperforms all baselines on eight out of fifteen tasks and demonstrates substantial gains when provided with domain knowledge. These results highlight the potential of unifying semantics and numerics for transparent, generalizable, and human-aligned RL. 

---
# Improving Score Reliability of Multiple Choice Benchmarks with Consistency Evaluation and Altered Answer Choices 

**Authors**: Paulo Cavalin, Cassia Sanctos, Marcelo Grave, Claudio Pinhanez, Yago Primerano  

**Link**: [PDF](https://arxiv.org/pdf/2511.21860)  

**Abstract**: In this work we present the Consistency-Rebalanced Accuracy (CoRA) metric, improving the reliability of Large Language Model (LLM) scores computed on multiple choice (MC) benchmarks. Our metric explores the response consistency of the LLMs, taking advantage of synthetically-generated questions with altered answer choices. With two intermediate scores, i.e. Bare-Minimum-Consistency Accuracy (BMCA) and Consistency Index (CI), CoRA is computed by adjusting the multiple-choice question answering (MCQA) scores to better reflect the level of consistency of the LLM. We present evaluations in different benchmarks using diverse LLMs, and not only demonstrate that LLMs can present low response consistency even when they present high MCQA scores, but also that CoRA can successfully scale down the scores of inconsistent models. 

---
# FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers 

**Authors**: Sarina Xi, Vishisht Rao, Justin Payan, Nihar B. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2511.21843)  

**Abstract**: The identification and localization of errors is a core task in peer review, yet the exponential growth of scientific output has made it increasingly difficult for human reviewers to reliably detect errors given the limited pool of experts. Recent advances in Large Language Models (LLMs) have sparked interest in their potential to support such evaluation tasks, from academic peer review to automated scientific assessment. However, despite the growing use of LLMs in review systems, their capabilities to pinpoint errors remain underexplored. In this work, we introduce Fault Localization Across Writing in Science (FLAWS), an automated benchmark consisting of 713 paper-error pairs designed to evaluate how effectively LLMs detect errors that undermine key claims in research papers. We construct the benchmark by systematically inserting claim-invalidating errors into peer-reviewed papers using LLMs, paired with an automated evaluation metric that measures whether models can identify and localize these errors. Developing such a benchmark presents unique challenges that we overcome: ensuring that the inserted errors are well-defined, challenging, and relevant to the content of the paper, avoiding artifacts that would make identification trivial, and designing a scalable, automated evaluation metric. On the resulting benchmark, we evaluate five frontier LLMs: Claude Sonnet 4.5, DeepSeek Reasoner v3.1, Gemini 2.5 Pro, GPT 5, and Grok 4. Among these, GPT 5 is the top-performing model, achieving 39.1% identification accuracy when k=10, where k is the number of top-ranked error text candidates generated by the LLM. 

---
# LLM-Empowered Event-Chain Driven Code Generation for ADAS in SDV systems 

**Authors**: Nenad Petrovic, Norbert Kroth, Axel Torschmied, Yinglei Song, Fengjunjie Pan, Vahid Zolfaghari, Nils Purschke, Sven Kirchner, Chengdong Wu, Andre Schamschurko, Yi Zhang, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2511.21877)  

**Abstract**: This paper presents an event-chain-driven, LLM-empowered workflow for generating validated, automotive code from natural-language requirements. A Retrieval-Augmented Generation (RAG) layer retrieves relevant signals from large and evolving Vehicle Signal Specification (VSS) catalogs as code generation prompt context, reducing hallucinations and ensuring architectural correctness. Retrieved signals are mapped and validated before being transformed into event chains that encode causal and timing constraints. These event chains guide and constrain LLM-based code synthesis, ensuring behavioral consistency and real-time feasibility. Based on our initial findings from the emergency braking case study, with the proposed approach, we managed to achieve valid signal usage and consistent code generation without LLM retraining. 

---
# fMRI-LM: Towards a Universal Foundation Model for Language-Aligned fMRI Understanding 

**Authors**: Yuxiang Wei, Yanteng Zhang, Xi Xiao, Chengxuan Qian, Tianyang Wang, Vince D. Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2511.21760)  

**Abstract**: Recent advances in multimodal large language models (LLMs) have enabled unified reasoning across images, audio, and video, but extending such capability to brain imaging remains largely unexplored. Bridging this gap is essential to link neural activity with semantic cognition and to develop cross-modal brain representations. To this end, we present fMRI-LM, a foundational model that bridges functional MRI (fMRI) and language through a three-stage framework. In Stage 1, we learn a neural tokenizer that maps fMRI into discrete tokens embedded in a language-consistent space. In Stage 2, a pretrained LLM is adapted to jointly model fMRI tokens and text, treating brain activity as a sequence that can be temporally predicted and linguistically described. To overcome the lack of natural fMRI-text pairs, we construct a large descriptive corpus that translates diverse imaging-based features into structured textual descriptors, capturing the low-level organization of fMRI signals. In Stage 3, we perform multi-task, multi-paradigm instruction tuning to endow fMRI-LM with high-level semantic understanding, supporting diverse downstream applications. Across various benchmarks, fMRI-LM achieves strong zero-shot and few-shot performance, and adapts efficiently with parameter-efficient tuning (LoRA), establishing a scalable pathway toward a language-aligned, universal model for structural and semantic understanding of fMRI. 

---
# Extracting Disaster Impacts and Impact Related Locations in Social Media Posts Using Large Language Models 

**Authors**: Sameeah Noreen Hameed, Surangika Ranathunga, Raj Prasanna, Kristin Stock, Christopher B. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2511.21753)  

**Abstract**: Large-scale disasters can often result in catastrophic consequences on people and infrastructure. Situation awareness about such disaster impacts generated by authoritative data from in-situ sensors, remote sensing imagery, and/or geographic data is often limited due to atmospheric opacity, satellite revisits, and time limitations. This often results in geo-temporal information gaps. In contrast, impact-related social media posts can act as "geo-sensors" during a disaster, where people describe specific impacts and locations. However, not all locations mentioned in disaster-related social media posts relate to an impact. Only the impacted locations are critical for directing resources effectively. e.g., "The death toll from a fire which ripped through the Greek coastal town of #Mati stood at 80, with dozens of people unaccounted for as forensic experts tried to identify victims who were burned alive #Greecefires #AthensFires #Athens #Greece." contains impacted location "Mati" and non-impacted locations "Greece" and "Athens". This research uses Large Language Models (LLMs) to identify all locations, impacts and impacted locations mentioned in disaster-related social media posts. In the process, LLMs are fine-tuned to identify only impacts and impacted locations (as distinct from other, non-impacted locations), including locations mentioned in informal expressions, abbreviations, and short forms. Our fine-tuned model demonstrates efficacy, achieving an F1-score of 0.69 for impact and 0.74 for impacted location extraction, substantially outperforming the pre-trained baseline. These robust results confirm the potential of fine-tuned language models to offer a scalable solution for timely decision-making in resource allocation, situational awareness, and post-disaster recovery planning for responders. 

---
# SO-Bench: A Structural Output Evaluation of Multimodal LLMs 

**Authors**: Di Feng, Kaixin Ma, Feng Nan, Haofeng Chen, Bohan Zhai, David Griffiths, Mingfei Gao, Zhe Gan, Eshan Verma, Yinfei Yang, Zhifeng Chen, Afshin Dehghan  

**Link**: [PDF](https://arxiv.org/pdf/2511.21750)  

**Abstract**: Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We plan to make the benchmark available to the community. 

---
# QuantumChem-200K: A Large-Scale Open Organic Molecular Dataset for Quantum-Chemistry Property Screening and Language Model Benchmarking 

**Authors**: Yinqi Zeng, Renjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.21747)  

**Abstract**: The discovery of next-generation photoinitiators for two-photon polymerization (TPP) is hindered by the absence of large, open datasets containing the quantum-chemical and photophysical properties required to model photodissociation and excited-state behavior. Existing molecular datasets typically provide only basic physicochemical descriptors and therefore cannot support data-driven screening or AI-assisted design of photoinitiators. To address this gap, we introduce QuantumChem-200K, a large-scale dataset of over 200,000 organic molecules annotated with eleven quantum-chemical properties, including two-photon absorption (TPA) cross sections, TPA spectral ranges, singlet-triplet intersystem crossing (ISC) energies, toxicity and synthetic accessibility scores, hydrophilicity, solubility, boiling point, molecular weight, and aromaticity. These values are computed using a hybrid workflow that integrates density function theory (DFT), semi-empirical excited-state methods, atomistic quantum solvers, and neural-network predictors. Using QuantumChem-200K, we fine tune the open-source Qwen2.5-32B large language model to create a chemistry AI assistant capable of forward property prediction from SMILES. Benchmarking on 3000 unseen molecules from VQM24 and ZINC20 demonstrates that domain-specific fine-tuning significantly improves accuracy over GPT-4o, Llama-3.1-70B, and the base Qwen2.5-32B model, particularly for TPA and ISC predictions central to photoinitiator design. QuantumChem-200K and the corresponding AI assistant together provide the first scalable platform for high-throughput, LLM-driven photoinitiator screening and accelerated discovery of photosensitive materials. 

---
# EduMod-LLM: A Modular Approach for Designing Flexible and Transparent Educational Assistants 

**Authors**: Meenakshi Mittal, Rishi Khare, Mihran Miroyan, Chancharik Mitra, Narges Norouzi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21742)  

**Abstract**: With the growing use of Large Language Model (LLM)-based Question-Answering (QA) systems in education, it is critical to evaluate their performance across individual pipeline components. In this work, we introduce {\model}, a modular function-calling LLM pipeline, and present a comprehensive evaluation along three key axes: function calling strategies, retrieval methods, and generative language models. Our framework enables fine-grained analysis by isolating and assessing each component. We benchmark function-calling performance across LLMs, compare our novel structure-aware retrieval method to vector-based and LLM-scoring baselines, and evaluate various LLMs for response synthesis. This modular approach reveals specific failure modes and performance patterns, supporting the development of interpretable and effective educational QA systems. Our findings demonstrate the value of modular function calling in improving system transparency and pedagogical alignment. Website and Supplementary Material: this https URL 

---
# Decoding inner speech with an end-to-end brain-to-text neural interface 

**Authors**: Yizi Zhang, Linyang He, Chaofei Fan, Tingkai Liu, Han Yu, Trung Le, Jingyuan Li, Scott Linderman, Lea Duncker, Francis R Willett, Nima Mesgarani, Liam Paninski  

**Link**: [PDF](https://arxiv.org/pdf/2511.21740)  

**Abstract**: Speech brain-computer interfaces (BCIs) aim to restore communication for people with paralysis by translating neural activity into text. Most systems use cascaded frameworks that decode phonemes before assembling sentences with an n-gram language model (LM), preventing joint optimization of all stages simultaneously. Here, we introduce an end-to-end Brain-to-Text (BIT) framework that translates neural activity into coherent sentences using a single differentiable neural network. Central to our approach is a cross-task, cross-species pretrained neural encoder, whose representations transfer to both attempted and imagined speech. In a cascaded setting with an n-gram LM, the pretrained encoder establishes a new state-of-the-art (SOTA) on the Brain-to-Text '24 and '25 benchmarks. Integrated end-to-end with audio large language models (LLMs) and trained with contrastive learning for cross-modal alignment, BIT reduces the word error rate (WER) of the prior end-to-end method from 24.69% to 10.22%. Notably, we find that small-scale audio LLMs markedly improve end-to-end decoding. Beyond record-setting performance, BIT aligns attempted and imagined speech embeddings to enable cross-task generalization. Altogether, our approach advances the integration of large, diverse neural datasets, paving the way for an end-to-end decoding framework that supports seamless, differentiable optimization. 

---
# R2Q: Towards Robust 2-Bit Large Language Models via Residual Refinement Quantization 

**Authors**: Jiayi Chen, Jieqi Shi, Jing Huo, Chen Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21736)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has brought substantial computational and memory demands, spurring the adoption of low-bit quantization. While 8-bit and 4-bit formats have become prevalent, extending quantization to 2 bits remains challenging due to severe accuracy degradation. To address this, we propose Residual Refinement Quantization (R2Q)-a novel 2-bit quantization framework that decomposes the process into two sequential 1-bit sub-quantizations, forming an adaptive quantization lattice. Extensive evaluations on Llama, OPT, and Qwen across diverse benchmarks-covering question answering, commonsense reasoning, and language modeling-demonstrate that R2Q consistently outperforms existing 2-bit quantization methods in both fine-grained and coarse-grained settings. By refining quantization through a residual learning mechanism, R2Q enhances performance, improves training stability, and accelerates convergence under extreme compression. Furthermore, its modular design enables seamless integration with existing quantization-aware training (QAT) frameworks. 

---
# Identifying Quantum Structure in AI Language: Evidence for Evolutionary Convergence of Human and Artificial Cognition 

**Authors**: Diederik Aerts, Jonito Aerts Arguëlles, Lester Beltran, Suzette Geriente, Roberto Leporini, Massimiliano Sassoli de Bianchi, Sandro Sozzo  

**Link**: [PDF](https://arxiv.org/pdf/2511.21731)  

**Abstract**: We present the results of cognitive tests on conceptual combinations, performed using specific Large Language Models (LLMs) as test subjects. In the first test, performed with ChatGPT and Gemini, we show that Bell's inequalities are significantly violated, which indicates the presence of 'quantum entanglement' in the tested concepts. In the second test, also performed using ChatGPT and Gemini, we instead identify the presence of 'Bose-Einstein statistics', rather than the intuitively expected 'Maxwell-Boltzmann statistics', in the distribution of the words contained in large-size texts. Interestingly, these findings mirror the results previously obtained in both cognitive tests with human participants and information retrieval tests on large corpora. Taken together, they point to the 'systematic emergence of quantum structures in conceptual-linguistic domains', regardless of whether the cognitive agent is human or artificial. Although LLMs are classified as neural networks for historical reasons, we believe that a more essential form of knowledge organization takes place in the distributive semantic structure of vector spaces built on top of the neural network. It is this meaning-bearing structure that lends itself to a phenomenon of evolutionary convergence between human cognition and language, slowly established through biological evolution, and LLM cognition and language, emerging much more rapidly as a result of self-learning and training. We analyze various aspects and examples that contain evidence supporting the above hypothesis. We also advance a unifying framework that explains the pervasive quantum organization of meaning that we identify. 

---
# Affective Multimodal Agents with Proactive Knowledge Grounding for Emotionally Aligned Marketing Dialogue 

**Authors**: Lin Yu, Xiaofei Han, Yifei Kang, Chiung-Yi Tseng, Danyang Zhang, Ziqian Bi, Zhimo Han  

**Link**: [PDF](https://arxiv.org/pdf/2511.21728)  

**Abstract**: Recent advances in large language models (LLMs) have enabled fluent dialogue systems, but most remain reactive and struggle in emotionally rich, goal-oriented settings such as marketing conversations. To address this limitation, we propose AffectMind, a multimodal affective dialogue agent that performs proactive reasoning and dynamic knowledge grounding to sustain emotionally aligned and persuasive interactions. AffectMind combines three components: a Proactive Knowledge Grounding Network (PKGN) that continuously updates factual and affective context from text, vision, and prosody; an Emotion--Intent Alignment Model (EIAM) that jointly models user emotion and purchase intent to adapt persuasion strategies; and a Reinforced Discourse Loop (RDL) that optimizes emotional coherence and engagement via reinforcement signals from user responses. Experiments on two newly curated marketing dialogue datasets, MM-ConvMarket and AffectPromo, show that AffectMind outperforms strong LLM-based baselines in emotional consistency (+26\%), persuasive success rate (+19\%), and long-term user engagement (+23\%), highlighting emotion-grounded proactivity as a key capability for commercial multimodal agents. 

---
# Medical Malice: A Dataset for Context-Aware Safety in Healthcare LLMs 

**Authors**: Andrew Maranhão Ventura D'addario  

**Link**: [PDF](https://arxiv.org/pdf/2511.21757)  

**Abstract**: The integration of Large Language Models (LLMs) into healthcare demands a safety paradigm rooted in \textit{primum non nocere}. However, current alignment techniques rely on generic definitions of harm that fail to capture context-dependent violations, such as administrative fraud and clinical discrimination. To address this, we introduce Medical Malice: a dataset of 214,219 adversarial prompts calibrated to the regulatory and ethical complexities of the Brazilian Unified Health System (SUS). Crucially, the dataset includes the reasoning behind each violation, enabling models to internalize ethical boundaries rather than merely memorizing a fixed set of refusals. Using an unaligned agent (Grok-4) within a persona-driven pipeline, we synthesized high-fidelity threats across seven taxonomies, ranging from procurement manipulation and queue-jumping to obstetric violence. We discuss the ethical design of releasing these "vulnerability signatures" to correct the information asymmetry between malicious actors and AI developers. Ultimately, this work advocates for a shift from universal to context-aware safety, providing the necessary resources to immunize healthcare AI against the nuanced, systemic threats inherent to high-stakes medical environments -- vulnerabilities that represent the paramount risk to patient safety and the successful integration of AI in healthcare systems. 

---
# Asking LLMs to Verify First is Almost Free Lunch 

**Authors**: Shiguang Wu, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21734)  

**Abstract**: To enhance the reasoning capabilities of Large Language Models (LLMs) without high costs of training, nor extensive test-time sampling, we introduce Verification-First (VF), a strategy that prompts models to verify a provided candidate answer, even a trivial or random one, before generating a solution. This approach triggers a "reverse reasoning" process that is cognitively easier and complementary to standard forward Chain-of-Thought (CoT), effectively invoking the model's critical thinking to reduce logical errors. We further generalize the VF strategy to Iter-VF, a sequential test-time scaling (TTS) method that iteratively cycles the verification-generation process using the model's previous answer. Extensive experiments across various benchmarks (from mathematical reasoning to coding and agentic tasks) and various LLMs (from open-source 1B to cutting-edge commercial ones) confirm that VF with random answer consistently outperforms standard CoT with minimal computational overhead, and Iter-VF outperforms existing TTS strategies. 

---
# GPS: General Per-Sample Prompter 

**Authors**: Pawel Batorski, Paul Swoboda  

**Link**: [PDF](https://arxiv.org/pdf/2511.21714)  

**Abstract**: LLMs are sensitive to prompting, with task performance often hinging on subtle, sometimes imperceptible variations in phrasing. As a result, crafting effective prompts manually remains challenging and time-consuming. Recent automatic prompting methods mitigate this difficulty but face three key limitations: (i) for each new task, they require large datasets to train good prompts;(ii) they rely on costly optimization loops that may take hours; (iii)they typically produce a single task-level prompt that does not adapt to the individual input problem to be solved.
We propose GPS, the first general-purpose, per-sample prompting method. Without any task-specific tuning, GPS generates a tailored prompt for each unseen input, improving performance across diverse tasks. The prompter is trained with reinforcement learning on a suite of training tasks and includes a novel regularization for effectively adapting to per-sample prompting. Finally, we employ Minimum Bayes Risk decoding to stabilize inference.
Empirically, GPS demonstrates competitive performance: we attain second best results among baselines on text simplification, third best results on summarization and on-par results on classification, while not training on any of these tasks, in contrast to the baselines. For in-domain prompting, we obtain sota on GSM8K. Our work shows the potential of a novel and effective paradigm for automatic prompting: generating adaptive, input-specific prompts without extensive optimization and without access to a task-specific training set. Our code is available at this https URL. 

---
# Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA Fine-Tuning and Efficient Majority Voting Approach 

**Authors**: Blessed Guda, Lawrence Francis, Gabrial Zencha Ashungafac, Carlee Joe-Wong, Moise Busogi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21709)  

**Abstract**: Multiple Choice Question (MCQ) answering is a widely used method for evaluating the performance of Large Language Models (LLMs). However, LLMs often exhibit selection bias in MCQ tasks, where their choices are influenced by factors like answer position or option symbols rather than the content. This bias undermines the reliability of MCQ as an evaluation framework. Most existing selection bias metrics require answer labels and measure divergences between prediction and answer distributions, but do not fully capture the consistency of a model's predictions across different orderings of answer choices. Existing selection bias mitigation strategies have notable limitations: majority voting, though effective, is computationally prohibitive; calibration-based methods require validation sets and often fail to generalize across datasets. To address these gaps, we propose three key contributions: (1) a new unsupervised label-free Permutation Bias Metric (PBM) that directly quantifies inconsistencies in model predictions across answer permutations, providing a more precise measure of selection bias, (2) an efficient majority voting approach called Batch Question-Context KV caching (BaQCKV), to significantly reduce computational costs while preserving bias mitigation effectiveness, and (3) an unsupervised Low-Rank Adaptation (LoRA-1) fine-tuning strategy based on our proposed metric and the BaQCKV that mitigates selection bias, providing a computationally efficient alternative that maintains model generalizability. Experiments across multiple MCQ benchmarks demonstrate that our approaches reduce bias, increasing consistency in accuracy while minimizing computational costs. 

---
# Lost in the Pipeline: How Well Do Large Language Models Handle Data Preparation? 

**Authors**: Matteo Spreafico, Ludovica Tassini, Camilla Sancricca, Cinzia Cappiello  

**Link**: [PDF](https://arxiv.org/pdf/2511.21708)  

**Abstract**: Large language models have recently demonstrated their exceptional capabilities in supporting and automating various tasks. Among the tasks worth exploring for testing large language model capabilities, we considered data preparation, a critical yet often labor-intensive step in data-driven processes. This paper investigates whether large language models can effectively support users in selecting and automating data preparation tasks. To this aim, we considered both general-purpose and fine-tuned tabular large language models. We prompted these models with poor-quality datasets and measured their ability to perform tasks such as data profiling and cleaning. We also compare the support provided by large language models with that offered by traditional data preparation tools. To evaluate the capabilities of large language models, we developed a custom-designed quality model that has been validated through a user study to gain insights into practitioners' expectations. 

---
# Factors That Support Grounded Responses in LLM Conversations: A Rapid Review 

**Authors**: Gabriele Cesar Iwashima, Claudia Susie Rodrigues, Claudio Dipolitto, Geraldo Xexéo  

**Link**: [PDF](https://arxiv.org/pdf/2511.21762)  

**Abstract**: Large language models (LLMs) may generate outputs that are misaligned with user intent, lack contextual grounding, or exhibit hallucinations during conversation, which compromises the reliability of LLM-based applications. This review aimed to identify and analyze techniques that align LLM responses with conversational goals, ensure grounding, and reduce hallucination and topic drift. We conducted a Rapid Review guided by the PRISMA framework and the PICO strategy to structure the search, filtering, and selection processes. The alignment strategies identified were categorized according to the LLM lifecycle phase in which they operate: inference-time, post-training, and reinforcement learning-based methods. Among these, inference-time approaches emerged as particularly efficient, aligning outputs without retraining while supporting user intent, contextual grounding, and hallucination mitigation. The reviewed techniques provided structured mechanisms for improving the quality and reliability of LLM responses across key alignment objectives. 

---
# EulerESG: Automating ESG Disclosure Analysis with LLMs 

**Authors**: Yi Ding, Xushuo Tang, Zhengyi Yang, Wenqian Zhang, Simin Wu, Yuxin Huang, Lingjing Lan, Weiyuan Li, Yin Chen, Mingchen Ju, Wenke Yang, Thong Hoang, Mykhailo Klymenko, Xiwei Zu, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21712)  

**Abstract**: Environmental, Social, and Governance (ESG) reports have become central to how companies communicate climate risk, social impact, and governance practices, yet they are still published primarily as long, heterogeneous PDF documents. This makes it difficult to systematically answer seemingly simple questions. Existing tools either rely on brittle rule-based extraction or treat ESG reports as generic text, without explicitly modelling the underlying reporting standards. We present \textbf{EulerESG}, an LLM-powered system for automating ESG disclosure analysis with explicit awareness of ESG frameworks. EulerESG combines (i) dual-channel retrieval and LLM-driven disclosure analysis over ESG reports, and (ii) an interactive dashboard and chatbot for exploration, benchmarking, and explanation. Using four globally recognised companies and twelve SASB sub-industries, we show that EulerESG can automatically populate standard-aligned metric tables with high fidelity (up to 0.95 average accuracy) while remaining practical in end-to-end runtime, and we compare several recent LLM models in this setting. The full implementation, together with a demonstration video, is publicly available at this https URL. 

---
# A General Highly Accurate Online Planning Method Integrating Large Language Models into Nested Rollout Policy Adaptation for Dialogue Tasks 

**Authors**: Hui Wang, Fafa Zhang, Xiaoyu Zhang, Chaoxu Mu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21706)  

**Abstract**: In goal-oriented dialogue tasks, the main challenge is to steer the interaction towards a given goal within a limited number of turns. Existing approaches either rely on elaborate prompt engineering, whose effectiveness is heavily dependent on human experience, or integrate policy networks and pre-trained policy models, which are usually difficult to adapt to new dialogue scenarios and costly to train. Therefore, in this paper, we present Nested Rollout Policy Adaptation for Goal-oriented Dialogue (NRPA-GD), a novel dialogue policy planning method that completely avoids specific model training by utilizing a Large Language Model (LLM) to simulate behaviors of user and system at the same time. Specifically, NRPA-GD constructs a complete evaluation mechanism for dialogue trajectories and employs an optimization framework of nested Monte Carlo simulation and policy self-adaptation to dynamically adjust policies during the dialogue process. The experimental results on four typical goal-oriented dialogue datasets show that NRPA-GD outperforms both existing prompt engineering and specifically pre-trained model-based methods. Impressively, NRPA-GD surpasses ChatGPT and pre-trained policy models with only a 0.6-billion-parameter LLM. The proposed approach further demonstrates the advantages and novelty of employing planning methods on LLMs to solve practical planning tasks. 

---
# Evaluating Embedding Generalization: How LLMs, LoRA, and SLERP Shape Representational Geometry 

**Authors**: Siyaxolisa Kabane  

**Link**: [PDF](https://arxiv.org/pdf/2511.21703)  

**Abstract**: We investigate the generalization properties of dense text embeddings when the embedding backbone is a large language model (LLM) versus when it is a non-LLM encoder, and we study the extent to which spherical linear interpolation (SLERP) model-merging mitigates over-specialization introduced by task-specific adaptation (e.g., LoRA). To make the comparison concrete and domain-agnostic, we design a controlled suite of experiments in which models embed short numerical sequences and are evaluated on their ability to cluster and classify those sequences according to well-defined number-theoretic properties. Our experimental protocol compares four families of models: (1) non-LLM encoders trained from scratch or fine-tuned for embeddings, (2) LLM-based encoders adapted with parameter-efficient methods (LoRA), (3) LLM-based encoders with LoRA followed by model souping merging into the base weights, and (4) the same LoRA-adapted LLMs merged using SLERP across checkpoints or stages. We evaluate representational quality with clustering indices (Silhouette and Davies Bouldin). We additionally analyze the use of kmeans labels to see if the embeddings encode any other information besides the one we are testing for. Empirically, we find that LLM-based backbones produce embeddings that better capture higher-order, compositional numeric patterns, but are prone to adapter dominance that degrades balanced generalization; SLERP merging consistently recovers base-model structure while retaining most task gains, yielding superior tradeoffs in clustering separability, and robustness compared to model souping or models that were not merged. 

---
# PromptTailor: Multi-turn Intent-Aligned Prompt Synthesis for Lightweight LLMs 

**Authors**: Yizhou Xu, Janet Davis  

**Link**: [PDF](https://arxiv.org/pdf/2511.21725)  

**Abstract**: Lightweight language models remain attractive for on-device and privacy-sensitive applications, but their responses are highly sensitive to prompt quality. For open-ended generation, non-expert users often lack the knowledge or time to consistently craft high-quality prompts, leading them to rely on prompt optimization tools. However, a key challenge is ensuring the optimized prompts genuinely align with users' original intents and preferences. We introduce PromptTailor, a system for controllable prompt generation for open-ended text that improves model output quality by intent-aligned prompt synthesis. PromptTailor expands minimal user instructions into rich, domain-aware prompts while preserving the user's stated preferences. The system is a quantized Llama3-8B model fine-tuned with a lightweight LoRA adapter on 12,300 prompt-refinement dialogues spanning 41 everyday domains, distilled from three stronger LLMs. The adapter attaches to any Llama3-8B base, enabling edge deployment. In human and LLM-judge evaluations across multiple target models and optimization baselines, PromptTailor yields higher preference rates than chain-of-thought prompting and matches or surpasses state-of-the-art prompt optimization methods while requiring fewer model calls (e.g., 3 vs. 9). These results show that a compact student, guided by powerful teachers, can learn effective prompt-generation strategies that enhance response quality while maintaining alignment with user intent. 

---
# Cacheback: Speculative Decoding With Nothing But Cache 

**Authors**: Zhiyao Ma, In Gim, Lin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2511.21699)  

**Abstract**: We present Cacheback Decoding, a training-free and model-agnostic speculative decoding method that exploits the locality in language to accelerate Large Language Model (LLM) inference. Cacheback leverages only Least Recently Used (LRU) cache tables of token n-grams to generate draft sequences. Cacheback achieves state-of-the-art performance among comparable methods despite its minimalist design, and its simplicity allows easy integration into existing systems. Cacheback also shows potential for fast adaptation to new domains. 

---
# Temporal Consistency for LLM Reasoning Process Error Identification 

**Authors**: Jiacheng Guo, Yue Wu, Jiahao Qiu, Kaixuan Huang, Xinzhe Juan, Ling Yang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.14495)  

**Abstract**: Verification is crucial for effective mathematical reasoning. We present a new temporal consistency method where verifiers iteratively refine their judgments based on the previous assessment. Unlike one-round verification or multi-model debate approaches, our method leverages consistency in a sequence of self-reflection actions to improve verification accuracy. Empirical evaluations across diverse mathematical process error identification benchmarks (Mathcheck, ProcessBench, and PRM800K) show consistent performance improvements over baseline methods. When applied to the recent DeepSeek R1 distilled models, our method demonstrates strong performance, enabling 7B/8B distilled models to outperform all 70B/72B models and GPT-4o on ProcessBench. Notably, the distilled 14B model with our method achieves performance comparable to Deepseek-R1. Our codes are available at this https URL 

---
# On the Role of Preference Variance in Preference Optimization 

**Authors**: Jiacheng Guo, Zihao Li, Jiahao Qiu, Yue Wu, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13022)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an important approach for learning from human preferences in aligning large language models (LLMs). However, collecting human preference data is costly and inefficient, motivating methods to reduce the required annotations. In this work, we investigate the impact of \emph{preference variance} (PVar), which measures the variance in model preferences when comparing pairs of responses, on the effectiveness of DPO training. We provide a theoretical insight by establishing an upper bound on the DPO gradient norm for any given prompt, showing it is controlled by the PVar of that prompt. This implies that prompts with low PVar can only produce small gradient updates, making them less valuable for learning. We validate this finding by fine-tuning LLMs with preferences generated by a reward model, evaluating on two benchmarks (AlpacaEval 2.0 and Arena-Hard). Experimental results demonstrate that prompts with higher PVar outperform randomly selected prompts or those with lower PVar. We also show that our PVar-based selection method is robust, when using smaller reward models (1B, 3B) for selection. Notably, in a separate experiment using the original human annotations from the UltraFeedback dataset, we found that training on only the top 10\% of prompts with the highest PVar yields better evaluation performance than training on the full dataset, highlighting the importance of preference variance in identifying informative examples for efficient LLM alignment. 

---
# Optimizing Multimodal Language Models through Attention-based Interpretability 

**Authors**: Alexander Sergeev, Evgeny Kotelnikov  

**Link**: [PDF](https://arxiv.org/pdf/2511.23375)  

**Abstract**: Modern large language models become multimodal, analyzing various data formats like text and images. While fine-tuning is effective for adapting these multimodal language models (MLMs) to downstream tasks, full fine-tuning is computationally expensive. Parameter-Efficient Fine-Tuning (PEFT) methods address this by training only a small portion of model weights. However, MLMs are difficult to interpret, making it challenging to identify which components are most effective for training to balance efficiency and performance. We propose an attention-based interpretability method for MLMs by analyzing attention scores relative to image tokens. The core idea is to identify attention heads that focus on image key objects. We utilize this information to select optimal model components for PEFT in multimodal models. Our contributions include a method for identifying attention heads associated with image key objects, its application to PEFT for image captioning, and the creation of a new dataset containing images, key object masks, and their textual descriptions. We conducted experiments on MLMs with 2-3 billion parameters to validate the method's effectiveness. By calculating Head Impact (HI) scores we quantify an attention head's focus on key objects, indicating its significance in image understanding. Our fine-tuning experiments demonstrate that adapting layers with the highest HI scores leads to the most significant shifts in metrics compared to pre-trained, randomly selected, or lowest-HI-score layers. This indicates that fine-tuning a small percentage (around 0.01%) of parameters in these crucial layers can substantially influence image understanding capabilities. 

---
# MCP vs RAG vs NLWeb vs HTML: A Comparison of the Effectiveness and Efficiency of Different Agent Interfaces to the Web (Technical Report) 

**Authors**: Aaron Steiner, Ralph Peeters, Christian Bizer  

**Link**: [PDF](https://arxiv.org/pdf/2511.23281)  

**Abstract**: Large language model agents are increasingly used to automate web tasks such as product search, offer comparison, and checkout. Current research explores different interfaces through which these agents interact with websites, including traditional HTML browsing, retrieval-augmented generation (RAG) over pre-crawled content, communication via Web APIs using the Model Context Protocol (MCP), and natural-language querying through the NLWeb interface. However, no prior work has compared these four architectures within a single controlled environment using identical tasks.
To address this gap, we introduce a testbed consisting of four simulated e-shops, each offering its products via HTML, MCP, and NLWeb interfaces. For each interface (HTML, RAG, MCP, and NLWeb) we develop specialized agents that perform the same sets of tasks, ranging from simple product searches and price comparisons to complex queries for complementary or substitute products and checkout processes. We evaluate the agents using GPT 4.1, GPT 5, GPT 5 mini, and Claude Sonnet 4 as underlying LLM. Our evaluation shows that the RAG, MCP and NLWeb agents outperform HTML on both effectiveness and efficiency. Averaged over all tasks, F1 rises from 0.67 for HTML to between 0.75 and 0.77 for the other agents. Token usage falls from about 241k for HTML to between 47k and 140k per task. The runtime per task drops from 291 seconds to between 50 and 62 seconds. The best overall configuration is RAG with GPT 5 achieving an F1 score of 0.87 and a completion rate of 0.79. Also taking cost into consideration, RAG with GPT 5 mini offers a good compromise between API usage fees and performance. Our experiments show the choice of the interaction interface has a substantial impact on both the effectiveness and efficiency of LLM-based web agents. 

---
# ShoppingComp: Are LLMs Really Ready for Your Shopping Cart? 

**Authors**: Huaixiao Tou, Ying Zeng, Cong Ma, Muzhi Li, Minghao Li, Weijie Yuan, He Zhang, Kai Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.22978)  

**Abstract**: We present ShoppingComp, a challenging real-world benchmark for rigorously evaluating LLM-powered shopping agents on three core capabilities: precise product retrieval, expert-level report generation, and safety critical decision making. Unlike prior e-commerce benchmarks, ShoppingComp introduces highly complex tasks under the principle of guaranteeing real products and ensuring easy verifiability, adding a novel evaluation dimension for identifying product safety hazards alongside recommendation accuracy and report quality. The benchmark comprises 120 tasks and 1,026 scenarios, curated by 35 experts to reflect authentic shopping needs. Results reveal stark limitations of current LLMs: even state-of-the-art models achieve low performance (e.g., 11.22% for GPT-5, 3.92% for Gemini-2.5-Flash). These findings highlight a substantial gap between research benchmarks and real-world deployment, where LLMs make critical errors such as failure to identify unsafe product usage or falling for promotional misinformation, leading to harmful recommendations. ShoppingComp fills the gap and thus establishes a new standard for advancing reliable and practical agents in e-commerce. 

---
# Visual Puns from Idioms: An Iterative LLM-T2IM-MLLM Framework 

**Authors**: Kelaiti Xiao, Liang Yang, Dongyu Zhang, Paerhati Tulajiang, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.22943)  

**Abstract**: We study idiom-based visual puns--images that align an idiom's literal and figurative meanings--and present an iterative framework that coordinates a large language model (LLM), a text-to-image model (T2IM), and a multimodal LLM (MLLM) for automatic generation and evaluation. Given an idiom, the system iteratively (i) generates detailed visual prompts, (ii) synthesizes an image, (iii) infers the idiom from the image, and (iv) refines the prompt until recognition succeeds or a step limit is reached. Using 1,000 idioms as inputs, we synthesize a corresponding dataset of visual pun images with paired prompts, enabling benchmarking of both generation and understanding. Experiments across 10 LLMs, 10 MLLMs, and one T2IM (Qwen-Image) show that MLLM choice is the primary performance driver: GPT achieves the highest accuracies, Gemini follows, and the best open-source MLLM (Gemma) is competitive with some closed models. On the LLM side, Claude attains the strongest average performance for prompt generation. 

---
# Social Perceptions of English Spelling Variation on Twitter: A Comparative Analysis of Human and LLM Responses 

**Authors**: Dong Nguyen, Laura Rosseel  

**Link**: [PDF](https://arxiv.org/pdf/2511.23041)  

**Abstract**: Spelling variation (e.g. funnnn vs. fun) can influence the social perception of texts and their writers: we often have various associations with different forms of writing (is the text informal? does the writer seem young?). In this study, we focus on the social perception of spelling variation in online writing in English and study to what extent this perception is aligned between humans and large language models (LLMs). Building on sociolinguistic methodology, we compare LLM and human ratings on three key social attributes of spelling variation (formality, carefulness, age). We find generally strong correlations in the ratings between humans and LLMs. However, notable differences emerge when we analyze the distribution of ratings and when comparing between different types of spelling variation. 

---
# FEANEL: A Benchmark for Fine-Grained Error Analysis in K-12 English Writing 

**Authors**: Jingheng Ye, Shen Wang, Jiaqi Chen, Hebin Wang, Deqing Zou, Yanyu Zhu, Jiwei Tang, Hai-Tao Zheng, Ruitong Liu, Haoyang Li, Yanfeng Wang, Qingsong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2511.22883)  

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence, offering profound opportunities for educational applications. However, their ability to provide fine-grained educational feedback for K-12 English writing remains underexplored. In this paper, we challenge the error analysis and pedagogical skills of LLMs by introducing the problem of Fine-grained Error Analysis for English Learners and present the Fine-grained Error ANalysis for English Learners (FEANEL) Benchmark. The benchmark comprises 1,000 essays written by elementary and secondary school students, and a well-developed English writing error taxonomy. Each error is annotated by language education experts and categorized by type, severity, and explanatory feedback, using a part-of-speech-based taxonomy they co-developed. We evaluate state-of-the-art LLMs on the FEANEL Benchmark to explore their error analysis and pedagogical abilities. Experimental results reveal significant gaps in current LLMs' ability to perform fine-grained error analysis, highlighting the need for advancements in particular methods for educational applications. 

---
# Mitigating Semantic Drift: Evaluating LLMs' Efficacy in Psychotherapy through MI Dialogue Summarization 

**Authors**: Vivek Kumar, Pushpraj Singh Rajawat, Eirini Ntoutsi  

**Link**: [PDF](https://arxiv.org/pdf/2511.22818)  

**Abstract**: Recent advancements in large language models (LLMs) have shown their potential across both general and domain-specific tasks. However, there is a growing concern regarding their lack of sensitivity, factual incorrectness in responses, inconsistent expressions of empathy, bias, hallucinations, and overall inability to capture the depth and complexity of human understanding, especially in low-resource and sensitive domains such as psychology. To address these challenges, our study employs a mixed-methods approach to evaluate the efficacy of LLMs in psychotherapy. We use LLMs to generate precise summaries of motivational interviewing (MI) dialogues and design a two-stage annotation scheme based on key components of the Motivational Interviewing Treatment Integrity (MITI) framework, namely evocation, collaboration, autonomy, direction, empathy, and a non-judgmental attitude. Using expert-annotated MI dialogues as ground truth, we formulate multi-class classification tasks to assess model performance under progressive prompting techniques, incorporating one-shot and few-shot prompting. Our results offer insights into LLMs' capacity for understanding complex psychological constructs and highlight best practices to mitigate ``semantic drift" in therapeutic settings. Our work contributes not only to the MI community by providing a high-quality annotated dataset to address data scarcity in low-resource domains but also critical insights for using LLMs for precise contextual interpretation in complex behavioral therapy. 

---
# Training-Free Loosely Speculative Decoding: Accepting Semantically Correct Drafts Beyond Exact Match 

**Authors**: Jinze Li, Yixing Xu, Guanchen Li, Shuo Yang, Jinfeng Xu, Xuanwu Yin, Dong Li, Edith C.H.Ngai, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2511.22972)  

**Abstract**: Large language models (LLMs) achieve strong performance across diverse tasks but suffer from high inference latency due to their autoregressive generation. Speculative Decoding (SPD) mitigates this issue by verifying candidate tokens in parallel from a smaller draft model, yet its strict exact-match verification discards many semantically valid continuations. Moreover, existing training-based SPD methods often suffer from performance degradation on out-of-distribution (OOD) tasks. To this end, we propose Training-Free Loosely Speculative Decoding (FLy), a novel method that loosens the rigid verification criterion by leveraging the target model's self-corrective behavior to judge whether a draft-target mismatch remains semantically valid. FLy introduces a two-tier mechanism: an entropy-level gate that identifies whether the current token allows multiple plausible alternatives or is nearly deterministic, and a token-level deferred window that distinguishes genuine errors from differently worded yet semantically correct variants. To further reduce latency, we design a multi-level acceleration strategy that accelerates not only the target model but also the drafter itself. Owing to its training-free design, FLy composes seamlessly with arbitrary draft-target pairs and generalizes across models and domains without hyperparameter re-tuning. Experiments show that FLy preserves more than 99% of the target model's accuracy while achieving an average 2.81x speedup on Llama-3.1-70B-Instruct and 5.07x speedup on the 405B variant. Notably, on out-of-domain datasets, our method remains highly effective and outperforms the training-based method EAGLE-3 by 1.62x. 

---
# Improving LLM-based Ontology Matching with fine-tuning on synthetic data 

**Authors**: Guilherme Sousa, Rinaldo Lima, Cassia Trojahn  

**Link**: [PDF](https://arxiv.org/pdf/2511.22612)  

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into various components of Ontology Matching pipelines. This paper investigates the capability of LLMs to perform ontology matching directly on ontology modules and generate the corresponding alignments. Furthermore, it is explored how a dedicated fine-tuning strategy can enhance the model's matching performance in a zero-shot setting. The proposed method incorporates a search space reduction technique to select relevant subsets from both source and target ontologies, which are then used to automatically construct prompts. Recognizing the scarcity of reference alignments for training, a novel LLM-based approach is introduced for generating a synthetic dataset. This process creates a corpus of ontology submodule pairs and their corresponding reference alignments, specifically designed to fine-tune an LLM for the ontology matching task. The proposed approach was evaluated on the Conference, Geolink, Enslaved, Taxon, and Hydrography datasets from the OAEI complex track. The results demonstrate that the LLM fine-tuned on the synthetically generated data exhibits superior performance compared to the non-fine-tuned base model. The key contribution is a strategy that combines automatic dataset generation with fine-tuning to effectively adapt LLMs for ontology matching tasks. 

---
# RAG System for Supporting Japanese Litigation Procedures: Faithful Response Generation Complying with Legal Norms 

**Authors**: Yuya Ishihara, Atsushi Keyaki, Hiroaki Yamada, Ryutaro Ohara, Mihoko Sumida  

**Link**: [PDF](https://arxiv.org/pdf/2511.22858)  

**Abstract**: This study discusses the essential components that a Retrieval-Augmented Generation (RAG)-based LLM system should possess in order to support Japanese medical litigation procedures complying with legal norms. In litigation, expert commissioners, such as physicians, architects, accountants, and engineers, provide specialized knowledge to help judges clarify points of dispute. When considering the substitution of these expert roles with a RAG-based LLM system, the constraint of strict adherence to legal norms is imposed. Specifically, three requirements arise: (1) the retrieval module must retrieve appropriate external knowledge relevant to the disputed issues in accordance with the principle prohibiting the use of private knowledge, (2) the responses generated must originate from the context provided by the RAG and remain faithful to that context, and (3) the retrieval module must reference external knowledge with appropriate timestamps corresponding to the issues at hand. This paper discusses the design of a RAG-based LLM system that satisfies these requirements. 

---
# Token-Level Marginalization for Multi-Label LLM Classifiers 

**Authors**: Anjaneya Praharaj, Jaykumar Kasundra  

**Link**: [PDF](https://arxiv.org/pdf/2511.22312)  

**Abstract**: This paper addresses the critical challenge of deriving interpretable confidence scores from generative language models (LLMs) when applied to multi-label content safety classification. While models like LLaMA Guard are effective for identifying unsafe content and its categories, their generative architecture inherently lacks direct class-level probabilities, which hinders model confidence assessment and performance interpretation. This limitation complicates the setting of dynamic thresholds for content moderation and impedes fine-grained error analysis. This research proposes and evaluates three novel token-level probability estimation approaches to bridge this gap. The aim is to enhance model interpretability and accuracy, and evaluate the generalizability of this framework across different instruction-tuned models. Through extensive experimentation on a synthetically generated, rigorously annotated dataset, it is demonstrated that leveraging token logits significantly improves the interpretability and reliability of generative classifiers, enabling more nuanced content safety moderation. 

---
# Smarter, not Bigger: Fine-Tuned RAG-Enhanced LLMs for Automotive HIL Testing 

**Authors**: Chao Feng, Zihan Liu, Siddhant Gupta, Gongpei Cui, Jan von der Assen, Burkhard Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2511.22584)  

**Abstract**: Hardware-in-the-Loop (HIL) testing is essential for automotive validation but suffers from fragmented and underutilized test artifacts. This paper presents HIL-GPT, a retrieval-augmented generation (RAG) system integrating domain-adapted large language models (LLMs) with semantic retrieval. HIL-GPT leverages embedding fine-tuning using a domain-specific dataset constructed via heuristic mining and LLM-assisted synthesis, combined with vector indexing for scalable, traceable test case and requirement retrieval. Experiments show that fine-tuned compact models, such as \texttt{bge-base-en-v1.5}, achieve a superior trade-off between accuracy, latency, and cost compared to larger models, challenging the notion that bigger is always better. An A/B user study further confirms that RAG-enhanced assistants improve perceived helpfulness, truthfulness, and satisfaction over general-purpose LLMs. These findings provide insights for deploying efficient, domain-aligned LLM-based assistants in industrial HIL environments. 

---
# A Hybrid Theory and Data-driven Approach to Persuasion Detection with Large Language Models 

**Authors**: Gia Bao Hoang, Keith J Ransom, Rachel Stephens, Carolyn Semmler, Nicolas Fay, Lewis Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2511.22109)  

**Abstract**: Traditional psychological models of belief revision focus on face-to-face interactions, but with the rise of social media, more effective models are needed to capture belief revision at scale, in this rich text-based online discourse. Here, we use a hybrid approach, utilizing large language models (LLMs) to develop a model that predicts successful persuasion using features derived from psychological experiments.
Our approach leverages LLM generated ratings of features previously examined in the literature to build a random forest classification model that predicts whether a message will result in belief change. Of the eight features tested, \textit{epistemic emotion} and \textit{willingness to share} were the top-ranking predictors of belief change in the model. Our findings provide insights into the characteristics of persuasive messages and demonstrate how LLMs can enhance models of successful persuasion based on psychological theory. Given these insights, this work has broader applications in fields such as online influence detection and misinformation mitigation, as well as measuring the effectiveness of online narratives. 

---
# LLMs for Low-Resource Dialect Translation Using Context-Aware Prompting: A Case Study on Sylheti 

**Authors**: Tabia Tanzin Prama, Christopher M. Danforth, Peter Sheridan Dodds  

**Link**: [PDF](https://arxiv.org/pdf/2511.21761)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong translation abilities through prompting, even without task-specific training. However, their effectiveness in dialectal and low-resource contexts remains underexplored. This study presents the first systematic investigation of LLM-based machine translation (MT) for Sylheti, a dialect of Bangla that is itself low-resource. We evaluate five advanced LLMs (GPT-4.1, GPT-4.1, LLaMA 4, Grok 3, and DeepSeek V3.2) across both translation directions (Bangla $\Leftrightarrow$ Sylheti), and find that these models struggle with dialect-specific vocabulary. To address this, we introduce Sylheti-CAP (Context-Aware Prompting), a three-step framework that embeds a linguistic rulebook, a dictionary (2{,}260 core vocabulary items and idioms), and an authenticity check directly into prompts. Extensive experiments show that Sylheti-CAP consistently improves translation quality across models and prompting strategies. Both automatic metrics and human evaluations confirm its effectiveness, while qualitative analysis reveals notable reductions in hallucinations, ambiguities, and awkward phrasing, establishing Sylheti-CAP as a scalable solution for dialectal and low-resource MT. Dataset link: \href{this https URL}{this https URL} 

---
# Building Domain-Specific Small Language Models via Guided Data Generation 

**Authors**: Aman Kumar, Ekant Muljibhai Amin, Xian Yeow Lee, Lasitha Vidyaratne, Ahmed K. Farahat, Dipanjan D. Ghosh, Yuta Koreeda, Chetan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2511.21748)  

**Abstract**: Large Language Models (LLMs) have shown remarkable success in supporting a wide range of knowledge-intensive tasks. In specialized domains, there is growing interest in leveraging LLMs to assist subject matter experts with domain-specific challenges. However, deploying LLMs as SaaS solutions raises data privacy concerns, while many open-source models demand significant computational resources for effective domain adaptation and deployment. A promising alternative is to develop smaller, domain-specialized LLMs, though this approach is often constrained by the lack of high-quality domain-specific training data. In this work, we address these limitations by presenting a cost-efficient and scalable training pipeline that combines guided synthetic data generation from a small seed corpus with bottom-up domain data curation. Our pipeline integrates Domain-Adaptive Pretraining (DAPT), Domain-specific Supervised Fine-tuning (DSFT), and Direct Preference Optimization (DPO) to train effective small-scale models for specialized use cases. We demonstrate this approach through DiagnosticSLM, a 3B-parameter domain-specific model tailored for fault diagnosis, root cause analysis, and repair recommendation in industrial settings. To evaluate model performance, we introduce four domain-specific benchmarks: multiple-choice questions (DiagnosticMCQ), question answering (DiagnosticQA), sentence completion (DiagnosticComp), and summarization (DiagnosticSum). DiagnosticSLM achieves up to 25% accuracy improvement over open-source models of comparable or larger size (2B-9B) on the MCQ task, while also outperforming or matching them in other tasks, demonstrating effective domain-specific reasoning and generalization capabilities. 

---
# Dissecting the Ledger: Locating and Suppressing "Liar Circuits" in Financial Large Language Models 

**Authors**: Soham Mirajkar  

**Link**: [PDF](https://arxiv.org/pdf/2511.21756)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-stakes financial domains, yet they suffer from specific, reproducible hallucinations when performing arithmetic operations. Current mitigation strategies often treat the model as a black box. In this work, we propose a mechanistic approach to intrinsic hallucination detection. By applying Causal Tracing to the GPT-2 XL architecture on the ConvFinQA benchmark, we identify a dual-stage mechanism for arithmetic reasoning: a distributed computational scratchpad in middle layers (L12-L30) and a decisive aggregation circuit in late layers (specifically Layer 46). We verify this mechanism via an ablation study, demonstrating that suppressing Layer 46 reduces the model's confidence in hallucinatory outputs by 81.8%. Furthermore, we demonstrate that a linear probe trained on this layer generalizes to unseen financial topics with 98% accuracy, suggesting a universal geometry of arithmetic deception. 

---
# C$^2$DLM: Causal Concept-Guided Diffusion Large Language Models 

**Authors**: Kairong Han, Nuanqiao Shan, Ziyu Zhao, Zijing Hu, Xinpeng Dong, Junjian Ye, Lujia Pan, Fei Wu, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22146)  

**Abstract**: Autoregressive (AR) language models and Diffusion Language Models (DLMs) constitute the two principal paradigms of large language models. However, both paradigms suffer from insufficient reasoning capabilities. Human reasoning inherently relies on causal knowledge and thought, which are reflected in natural language. But in the AR paradigm, language is modeled as next token prediction (a strictly left-to-right, token-by-token order), whereas natural language itself exhibits more flexible causal structures. In the DLM paradigm, the attention mechanism is fully connected, which entirely disregards causal order. To fill this gap, we propose a \underline{\textbf{C}}ausal \underline{\textbf{C}}oncept-Guided \underline{\textbf{D}}iffusion \underline{\textbf{L}}anguage \underline{\textbf{M}}odel (C$^2$DLM). Starting from DLM's fully connected attention, C$^2$DLM first obtains a concept-level causal graph from the teacher model, and then explicitly guides attention to learn causal relationships between concepts. By focusing on causal relationships and avoiding interference from difficult subgoals involving causal inversion, C$^2$DLM improves 12\% with about 3.2 times training speedup in the COT-OrderPerturb task, and achieves an average gain of 1.31\% across six downstream reasoning tasks. More details in the repository ~\href{this https URL}{here}. 

---
# Scaling Competence, Shrinking Reasoning: Cognitive Signatures in Language Model Learning 

**Authors**: Mukul Singh, Ananya Singha, Arjun Radhakrishna, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2511.21743)  

**Abstract**: We analyze reasoning in language models during task-specific fine-tuning and draws parallel between reasoning tokens--intermediate steps generated while solving problem and the human working memory. Drawing from cognitive science, we align training dynamics with the Four Stages of Competence: models initially produce incorrect outputs without reasoning, then begin reasoning (but still fail), eventually reason effectively, and finally solve tasks without explicit reasoning. We find that reasoning token length expands as performance improves, peaks at the stage of conscious competence, then declines as the model internalizes the task. Notably, after training, models retain performance even when reasoning is removed--suggesting it scaffolded learning but is no longer needed. This progression offers actionable insights: reasoning token dynamics can serve as a signal for diagnosing training stage, identifying convergence, and guiding early stopping. We propose metrics to track this trajectory and argue that reasoning behavior is valuable for understanding and optimizing reasoning model training. 

---
# PeerCoPilot: A Language Model-Powered Assistant for Behavioral Health Organizations 

**Authors**: Gao Mo, Naveen Raman, Megan Chai, Cindy Peng, Shannon Pagdon, Nev Jones, Hong Shen, Peggy Swarbrick, Fei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2511.21721)  

**Abstract**: Behavioral health conditions, which include mental health and substance use disorders, are the leading disease burden in the United States. Peer-run behavioral health organizations (PROs) critically assist individuals facing these conditions by combining mental health services with assistance for needs such as income, employment, and housing. However, limited funds and staffing make it difficult for PROs to address all service user needs. To assist peer providers at PROs with their day-to-day tasks, we introduce PeerCoPilot, a large language model (LLM)-powered assistant that helps peer providers create wellness plans, construct step-by-step goals, and locate organizational resources to support these goals. PeerCoPilot ensures information reliability through a retrieval-augmented generation pipeline backed by a large database of over 1,300 vetted resources. We conducted human evaluations with 15 peer providers and 6 service users and found that over 90% of users supported using PeerCoPilot. Moreover, we demonstrated that PeerCoPilot provides more reliable and specific information than a baseline LLM. PeerCoPilot is now used by a group of 5-10 peer providers at CSPNJ, a large behavioral health organization serving over 10,000 service users, and we are actively expanding PeerCoPilot's use. 

---
# When Harmless Words Harm: A New Threat to LLM Safety via Conceptual Triggers 

**Authors**: Zhaoxin Zhang, Borui Chen, Yiming Hu, Youyang Qu, Tianqing Zhu, Longxiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21718)  

**Abstract**: Recent research on large language model (LLM) jailbreaks has primarily focused on techniques that bypass safety mechanisms to elicit overtly harmful outputs. However, such efforts often overlook attacks that exploit the model's capacity for abstract generalization, creating a critical blind spot in current alignment strategies. This gap enables adversaries to induce objectionable content by subtly manipulating the implicit social values embedded in model outputs. In this paper, we introduce MICM, a novel, model-agnostic jailbreak method that targets the aggregate value structure reflected in LLM responses. Drawing on conceptual morphology theory, MICM encodes specific configurations of nuanced concepts into a fixed prompt template through a predefined set of phrases. These phrases act as conceptual triggers, steering model outputs toward a specific value stance without triggering conventional safety filters. We evaluate MICM across five advanced LLMs, including GPT-4o, Deepseek-R1, and Qwen3-8B. Experimental results show that MICM consistently outperforms state-of-the-art jailbreak techniques, achieving high success rates with minimal rejection. Our findings reveal a critical vulnerability in commercial LLMs: their safety mechanisms remain susceptible to covert manipulation of underlying value alignment. 

---
# Insight-A: Attribution-aware for Multimodal Misinformation Detection 

**Authors**: Junjie Wu, Yumeng Fu, Chen Gong, Guohong Fu  

**Link**: [PDF](https://arxiv.org/pdf/2511.21705)  

**Abstract**: AI-generated content (AIGC) technology has emerged as a prevalent alternative to create multimodal misinformation on social media platforms, posing unprecedented threats to societal safety. However, standard prompting leverages multimodal large language models (MLLMs) to identify the emerging misinformation, which ignores the misinformation attribution. To this end, we present Insight-A, exploring attribution with MLLM insights for detecting multimodal misinformation. Insight-A makes two efforts: I) attribute misinformation to forgery sources, and II) an effective pipeline with hierarchical reasoning that detects distortions across modalities. Specifically, to attribute misinformation to forgery traces based on generation patterns, we devise cross-attribution prompting (CAP) to model the sophisticated correlations between perception and reasoning. Meanwhile, to reduce the subjectivity of human-annotated prompts, automatic attribution-debiased prompting (ADP) is used for task adaptation on MLLMs. Additionally, we design image captioning (IC) to achieve visual details for enhancing cross-modal consistency checking. Extensive experiments demonstrate the superiority of our proposal and provide a new paradigm for multimodal misinformation detection in the era of AIGC. 

---
# 47B Mixture-of-Experts Beats 671B Dense Models on Chinese Medical Examinations 

**Authors**: Chiung-Yi Tseng, Danyang Zhang, Tianyang Wang, Hongying Luo, Lu Chen, Junming Huang, Jibin Guan, Junfeng Hao, Junhao Song, Ziqian Bi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21701)  

**Abstract**: The rapid advancement of large language models(LLMs) has prompted significant interest in their potential applications in medical domains. This paper presents a comprehensive benchmark evaluation of 27 state-of-the-art LLMs on Chinese medical examination questions, encompassing seven medical specialties across two professional levels. We introduce a robust evaluation framework that assesses model performance on 2,800 carefully curated questions from cardiovascular, gastroenterology, hematology, infectious diseases, nephrology, neurology, and respiratory medicine domains. Our dataset distinguishes between attending physician and senior physician difficulty levels, providing nuanced insights into model capabilities across varying complexity. Our empirical analysis reveals substantial performance variations among models, with Mixtral-8x7B achieving the highest overall accuracy of 74.25%, followed by DeepSeek-R1-671B at 64.07%. Notably, we observe no consistent correlation between model size and performance, as evidenced by the strong performance of smaller mixture-of-experts architectures. The evaluation demonstrates significant performance gaps between medical specialties, with models generally performing better on cardiovascular and neurology questions compared to gastroenterology and nephrology domains. Furthermore, our analysis indicates minimal performance degradation between attending and senior physician levels for top-performing models, suggesting robust generalization capabilities. This benchmark provides critical insights for the deployment of LLMs in medical education and clinical decision support systems, highlighting both the promise and current limitations of these technologies in specialized medical contexts. 

---
# Addressing Stereotypes in Large Language Models: A Critical Examination and Mitigation 

**Authors**: Fatima Kazi  

**Link**: [PDF](https://arxiv.org/pdf/2511.21711)  

**Abstract**: Large Language models (LLMs), such as ChatGPT, have gained popularity in recent years with the advancement of Natural Language Processing (NLP), with use cases spanning many disciplines and daily lives as well. LLMs inherit explicit and implicit biases from the datasets they were trained on; these biases can include social, ethical, cultural, religious, and other prejudices and stereotypes. It is important to comprehensively examine such shortcomings by identifying the existence and extent of such biases, recognizing the origin, and attempting to mitigate such biased outputs to ensure fair outputs to reduce harmful stereotypes and misinformation. This study inspects and highlights the need to address biases in LLMs amid growing generative Artificial Intelligence (AI). We utilize bias-specific benchmarks such StereoSet and CrowSPairs to evaluate the existence of various biases in many different generative models such as BERT, GPT 3.5, and ADA. To detect both explicit and implicit biases, we adopt a three-pronged approach for thorough and inclusive analysis. Results indicate fine-tuned models struggle with gender biases but excel at identifying and avoiding racial biases. Our findings also illustrated that despite some cases of success, LLMs often over-rely on keywords in prompts and its outputs. This demonstrates the incapability of LLMs to attempt to truly understand the accuracy and authenticity of its outputs. Finally, in an attempt to bolster model performance, we applied an enhancement learning strategy involving fine-tuning, models using different prompting techniques, and data augmentation of the bias benchmarks. We found fine-tuned models to exhibit promising adaptability during cross-dataset testing and significantly enhanced performance on implicit bias benchmarks, with performance gains of up to 20%. 

---
# ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering 

**Authors**: Alberto Compagnoni, Marco Morini, Sara Sarto, Federico Cocchi, Davide Caffagni, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2511.22715)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown impressive capabilities in jointly understanding text, images, and videos, often evaluated via Visual Question Answering (VQA). However, even state-of-the-art MLLMs struggle with domain-specific or knowledge-intensive queries, where relevant information is underrepresented in pre-training data. Knowledge-based VQA (KB-VQA) addresses this by retrieving external documents to condition answer generation, but current retrieval-augmented approaches suffer from low precision, noisy passages, and limited reasoning. To address this, we propose ReAG, a novel Reasoning-Augmented Multimodal RAG approach that combines coarse- and fine-grained retrieval with a critic model that filters irrelevant passages, ensuring high-quality additional context. The model follows a multi-stage training strategy leveraging reinforcement learning to enhance reasoning over retrieved content, while supervised fine-tuning serves only as a cold start. Extensive experiments on Encyclopedic-VQA and InfoSeek demonstrate that ReAG significantly outperforms prior methods, improving answer accuracy and providing interpretable reasoning grounded in retrieved evidence. Our source code is publicly available at: this https URL. 

---
# ThetaEvolve: Test-time Learning on Open Problems 

**Authors**: Yiping Wang, Shao-Rong Su, Zhiyuan Zeng, Eva Xu, Liliang Ren, Xinyu Yang, Zeyi Huang, Xuehai He, Luyao Ma, Baolin Peng, Hao Cheng, Pengcheng He, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.23473)  

**Abstract**: Recent advances in large language models (LLMs) have enabled breakthroughs in mathematical discovery, exemplified by AlphaEvolve, a closed-source system that evolves programs to improve bounds on open problems. However, it relies on ensembles of frontier LLMs to achieve new bounds and is a pure inference system that models cannot internalize the evolving strategies. We introduce ThetaEvolve, an open-source framework that simplifies and extends AlphaEvolve to efficiently scale both in-context learning and Reinforcement Learning (RL) at test time, allowing models to continually learn from their experiences in improving open optimization problems. ThetaEvolve features a single LLM, a large program database for enhanced exploration, batch sampling for higher throughput, lazy penalties to discourage stagnant outputs, and optional reward shaping for stable training signals, etc. ThetaEvolve is the first evolving framework that enable a small open-source model, like DeepSeek-R1-0528-Qwen3-8B, to achieve new best-known bounds on open problems (circle packing and first auto-correlation inequality) mentioned in AlphaEvolve. Besides, across two models and four open tasks, we find that ThetaEvolve with RL at test-time consistently outperforms inference-only baselines, and the model indeed learns evolving capabilities, as the RL-trained checkpoints demonstrate faster progress and better final performance on both trained target task and other unseen tasks. We release our code publicly: this https URL 

---
# PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel 

**Authors**: Jinjun Yi, Zhixin Zhao, Yitao Hu, Ke Yan, Weiwei Sun, Hao Wang, Laiping Zhao, Yuhao Zhang, Wenxin Li, Keqiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.22333)  

**Abstract**: LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: *one-query-per-CTA* execution repeatedly loads shared prefix KV cache, while *one-size-fits-all* tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention.
This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 67.4% on average and TPOT by 13.6-83.4% under the same configurations against state-of-the-art attention kernels. 

---
# Selecting User Histories to Generate LLM Users for Cold-Start Item Recommendation 

**Authors**: Nachiket Subbaraman, Jaskinder Sarai, Aniruddh Nath, Lichan Hong, Lukasz Heldt, Li Wei, Zhe Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.21989)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning, generalization, and simulating human-like behavior across a wide range of tasks. These strengths present new opportunities to enhance traditional recommendation systems (RS), especially in the cold-start item scenario where newly introduced items lack interactions. Existing works have used LLMs to address cold-start issues in traditional RS through data augmentation, but they have limitations. One recent work directly addresses this issue by prompting LLMs to generate augmented interaction data between randomly sampled users and cold-start items. Then, they train the traditional RS with augmented data, incorporating collaborative signals for cold-start items. Although they use LLMs to provide cold-start items with feedback, they use partial user histories, which does not allow the LLM to fully emulate the user. Furthermore, randomly selecting users is not optimal for augmentation. To address these challenges, we leverage the LLM as a user and develop a reinforcement learning (RL) framework that trains a policy to select users for augmentation, optimizing for cold-start item performance after augmented training. The policy model learns to select users for cold-start item data augmentation based on their behavioral features and histories. To optimize user selection for cold-start item performance, we employ a policy gradient method that updates the policy in the direction of actions that lead to high rewards. Experiments on Amazon Product Review datasets show substantial gains in cold-start item recall, demonstrating the effectiveness of our method as a scalable, serving-efficient augmentation strategy for modern RS. 

---
# Do LLM-judges Align with Human Relevance in Cranfield-style Recommender Evaluation? 

**Authors**: Gustavo Penha, Aleksandr V. Petrov, Claudia Hauff, Enrico Palumbo, Ali Vardasbi, Edoardo D'Amico, Francesco Fabbri, Alice Wang, Praveen Chandar, Henrik Lindstrom, Hugues Bouchard, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2511.23312)  

**Abstract**: Evaluating recommender systems remains a long-standing challenge, as offline methods based on historical user interactions and train-test splits often yield unstable and inconsistent results due to exposure bias, popularity bias, sampled evaluations, and missing-not-at-random patterns. In contrast, textual document retrieval benefits from robust, standardized evaluation via Cranfield-style test collections, which combine pooled relevance judgments with controlled setups. While recent work shows that adapting this methodology to recommender systems is feasible, constructing such collections remains costly due to the need for manual relevance judgments, thus limiting scalability. This paper investigates whether Large Language Models (LLMs) can serve as reliable automatic judges to address these scalability challenges. Using the ML-32M-ext Cranfield-style movie recommendation collection, we first examine the limitations of existing evaluation methodologies. Then we explore the alignment and the recommender systems ranking agreement between the LLM-judge and human provided relevance labels. We find that incorporating richer item metadata and longer user histories improves alignment, and that LLM-judge yields high agreement with human-based rankings (Kendall's tau = 0.87). Finally, an industrial case study in the podcast recommendation domain demonstrates the practical value of LLM-judge for model selection. Overall, our results show that LLM-judge is a viable and scalable approach for evaluating recommender systems. 

---
# Evaluating Embedding Models and Pipeline Optimization for AI Search Quality 

**Authors**: Philip Zhong, Kent Chen, Don Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.22240)  

**Abstract**: We evaluate the performance of various text embedding models and pipeline configurations for AI-driven search systems. We compare sentence-transformer and generative embedding models (e.g., All-MPNet, BGE, GTE, and Qwen) at different dimensions, indexing methods (Milvus HNSW/IVF), and chunking strategies. A custom evaluation dataset of 11,975 query-chunk pairs was synthesized from US City Council meeting transcripts using a local large language model (LLM). The data pipeline includes preprocessing, automated question generation per chunk, manual validation, and continuous integration/continuous deployment (CI/CD) integration. We measure retrieval accuracy using reference-based metrics: Top-K Accuracy and Normalized Discounted Cumulative Gain (NDCG). Our results demonstrate that higher-dimensional embeddings significantly boost search quality (e.g., Qwen3-Embedding-8B/4096 achieves Top-3 accuracy about 0.571 versus 0.412 for GTE-large/1024), and that neural re-rankers (e.g., a BGE cross-encoder) further improve ranking accuracy (Top-3 up to 0.527). Finer-grained chunking (512 characters versus 2000 characters) also improves accuracy. We discuss the impact of these factors and outline future directions for pipeline automation and evaluation. 

---
# How Does A Text Preprocessing Pipeline Affect Ontology Matching? 

**Authors**: Zhangcheng Qiang, Kerry Taylor, Weiqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2411.03962)  

**Abstract**: The classical text preprocessing pipeline, comprising Tokenisation, Normalisation, Stop Words Removal, and Stemming/Lemmatisation, has been implemented in many systems for ontology matching (OM). However, the lack of standardisation in text preprocessing creates diversity in the mapping results. In this paper, we investigate the effect of the text preprocessing pipeline on 8 Ontology Alignment Evaluation Initiative (OAEI) tracks with 49 distinct alignments. We find that Tokenisation and Normalisation (categorised as Phase 1 text preprocessing) are more effective than Stop Words Removal and Stemming/Lemmatisation (categorised as Phase 2 text preprocessing). We propose two novel approaches to repair unwanted false mappings that occur in Phase 2 text preprocessing. One is an ad hoc logic-based repair approach used before text preprocessing, employing an ontology-specific check to find common words that cause false mappings. The other repair approach is the post hoc large language model (LLM)-based approach, used after text preprocessing, which utilises the strong background knowledge provided by LLMs to repair non-existent and counter-intuitive false mappings. The experimental results indicate that these two approaches can significantly improve the matching correctness and the overall matching performance. 

---
