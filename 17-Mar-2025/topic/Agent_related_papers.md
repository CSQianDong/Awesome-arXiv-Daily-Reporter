# AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation 

**Authors**: Fengyu Li, Yilin Li, Junhao Zhu, Lu Chen, Yanfei Zhang, Jia Zhou, Hui Zu, Jingwen Zhao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11346)  

**Abstract**: Huawei has always been committed to exploring the AI application in historical research. Biography generation, as a specialized form of abstractive summarization, plays a crucial role in historical research but faces unique challenges that existing large language models (LLMs) struggle to address. These challenges include maintaining stylistic adherence to historical writing conventions, ensuring factual fidelity, and handling fragmented information across multiple documents. We present AIstorian, a novel end-to-end agentic system featured with a knowledge graph (KG)-powered retrieval-augmented generation (RAG) and anti-hallucination multi-agents. Specifically, AIstorian introduces an in-context learning based chunking strategy and a KG-based index for accurate and efficient reference retrieval. Meanwhile, AIstorian orchestrates multi-agents to conduct on-the-fly hallucination detection and error-type-aware correction. Additionally, to teach LLMs a certain language style, we finetune LLMs based on a two-step training approach combining data augmentation-enhanced supervised fine-tuning with stylistic preference optimization. Extensive experiments on a real-life historical Jinshi dataset demonstrate that AIstorian achieves a 3.8x improvement in factual accuracy and a 47.6% reduction in hallucination rate compared to existing baselines. The data and code are available at: this https URL. 

---
# GNNs as Predictors of Agentic Workflow Performances 

**Authors**: Yuanshuo Zhang, Yuchen Hou, Bohan Tang, Shuo Chen, Muhan Zhang, Xiaowen Dong, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.11301)  

**Abstract**: Agentic workflows invoked by Large Language Models (LLMs) have achieved remarkable success in handling complex tasks. However, optimizing such workflows is costly and inefficient in real-world applications due to extensive invocations of LLMs. To fill this gap, this position paper formulates agentic workflows as computational graphs and advocates Graph Neural Networks (GNNs) as efficient predictors of agentic workflow performances, avoiding repeated LLM invocations for evaluation. To empirically ground this position, we construct FLORA-Bench, a unified platform for benchmarking GNNs for predicting agentic workflow performances. With extensive experiments, we arrive at the following conclusion: GNNs are simple yet effective predictors. This conclusion supports new applications of GNNs and a novel direction towards automating agentic workflow optimization. All codes, models, and data are available at this https URL. 

---
# DeskVision: Large Scale Desktop Region Captioning for Advanced GUI Agents 

**Authors**: Yibin Xu, Liang Yang, Hao Chen, Hua Wang, Zhi Chen, Yaohua Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11170)  

**Abstract**: The limitation of graphical user interface (GUI) data has been a significant barrier to the development of GUI agents today, especially for the desktop / computer use scenarios. To address this, we propose an automated GUI data generation pipeline, AutoCaptioner, which generates data with rich descriptions while minimizing human effort. Using AutoCaptioner, we created a novel large-scale desktop GUI dataset, DeskVision, along with the largest desktop test benchmark, DeskVision-Eval, which reflects daily usage and covers diverse systems and UI elements, each with rich descriptions. With DeskVision, we train a new GUI understanding model, GUIExplorer. Results show that GUIExplorer achieves state-of-the-art (SOTA) performance in understanding/grounding visual elements without the need for complex architectural designs. We further validated the effectiveness of the DeskVision dataset through ablation studies on various large visual language models (LVLMs). We believe that AutoCaptioner and DeskVision will significantly advance the development of GUI agents, and will open-source them for the community. 

---
# Thinking Machines: A Survey of LLM based Reasoning Strategies 

**Authors**: Dibyanayan Bandyopadhyay, Soham Bhattacharjee, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2503.10814)  

**Abstract**: Large Language Models (LLMs) are highly proficient in language-based tasks. Their language capabilities have positioned them at the forefront of the future AGI (Artificial General Intelligence) race. However, on closer inspection, Valmeekam et al. (2024); Zecevic et al. (2023); Wu et al. (2024) highlight a significant gap between their language proficiency and reasoning abilities. Reasoning in LLMs and Vision Language Models (VLMs) aims to bridge this gap by enabling these models to think and re-evaluate their actions and responses. Reasoning is an essential capability for complex problem-solving and a necessary step toward establishing trust in Artificial Intelligence (AI). This will make AI suitable for deployment in sensitive domains, such as healthcare, banking, law, defense, security etc. In recent times, with the advent of powerful reasoning models like OpenAI O1 and DeepSeek R1, reasoning endowment has become a critical research topic in LLMs. In this paper, we provide a detailed overview and comparison of existing reasoning techniques and present a systematic survey of reasoning-imbued language models. We also study current challenges and present our findings. 

---
# SciFi-Benchmark: How Would AI-Powered Robots Behave in Science Fiction Literature? 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10706)  

**Abstract**: Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in SciFi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers, in addition to a smaller human-labeled evaluation set. Data is available at this https URL 

---
# Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents 

**Authors**: Dongjun Lee, Juyong Lee, Kyuyoung Kim, Jihoon Tack, Jinwoo Shin, Yee Whye Teh, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10689)  

**Abstract**: Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: this https URL. 

---
# Cerebrum (AIOS SDK): A Platform for Agent Development, Deployment, Distribution, and Discovery 

**Authors**: Balaji Rama, Kai Mei, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11444)  

**Abstract**: Autonomous LLM-based agents have emerged as a powerful paradigm for complex task execution, yet the field lacks standardized tools for development, deployment, distribution and discovery of agents. We present Cerebrum, an Agent SDK for AIOS that addresses this gap through three key components: (1) a comprehensive SDK featuring a modular four-layer architecture for agent development, encompassing LLM, memory, storage, and tool management; (2) a community-driven Agent Hub for sharing and discovering agents, complete with version control and dependency management; (3) an interactive web interface for testing and evaluating agents. The platform's effectiveness is demonstrated through implementations of various agent architectures, including Chain of Thought (CoT), ReAct, and tool-use agents. Cerebrum advances the field by providing a unified framework that standardizes agent development while maintaining flexibility for researchers and developers to innovate and distribute their agents. The live website is at this https URL, the code is at this https URL, and video is at this https URL. 

---
# Prompt Injection Detection and Mitigation via AI Multi-Agent NLP Frameworks 

**Authors**: Diego Gosmar, Deborah A. Dahl, Dario Gosmar  

**Link**: [PDF](https://arxiv.org/pdf/2503.11517)  

**Abstract**: Prompt injection constitutes a significant challenge for generative AI systems by inducing unintended outputs. We introduce a multi-agent NLP framework specifically designed to address prompt injection vulnerabilities through layered detection and enforcement mechanisms. The framework orchestrates specialized agents for generating responses, sanitizing outputs, and enforcing policy compliance. Evaluation on 500 engineered injection prompts demonstrates a marked reduction in injection success and policy breaches. Novel metrics, including Injection Success Rate (ISR), Policy Override Frequency (POF), Prompt Sanitization Rate (PSR), and Compliance Consistency Score (CCS), are proposed to derive a composite Total Injection Vulnerability Score (TIVS). The system utilizes the OVON (Open Voice Network) framework for inter-agent communication via structured JSON messages, extending a previously established multi-agent architecture from hallucination mitigation to address the unique challenges of prompt injection. 

---
# Large Reasoning Models in Agent Scenarios: Exploring the Necessity of Reasoning Capabilities 

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Weidong Wang, Zhigang Zuo, Di Wu, Duanfeng Chu, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11074)  

**Abstract**: The rise of Large Reasoning Models (LRMs) signifies a paradigm shift toward advanced computational reasoning. Yet, this progress disrupts traditional agent frameworks, traditionally anchored by execution-oriented Large Language Models (LLMs). To explore this transformation, we propose the LaRMA framework, encompassing nine tasks across Tool Usage, Plan Design, and Problem Solving, assessed with three top LLMs (e.g., Claude3.5-sonnet) and five leading LRMs (e.g., DeepSeek-R1). Our findings address four research questions: LRMs surpass LLMs in reasoning-intensive tasks like Plan Design, leveraging iterative reflection for superior outcomes; LLMs excel in execution-driven tasks such as Tool Usage, prioritizing efficiency; hybrid LLM-LRM configurations, pairing LLMs as actors with LRMs as reflectors, optimize agent performance by blending execution speed with reasoning depth; and LRMs' enhanced reasoning incurs higher computational costs, prolonged processing, and behavioral challenges, including overthinking and fact-ignoring tendencies. This study fosters deeper inquiry into LRMs' balance of deep thinking and overthinking, laying a critical foundation for future agent design advancements. 

---
# Collaboration is all you need: LLM Assisted Safe Code Translation 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2503.11237)  

**Abstract**: This paper introduces UniTranslator, a visionary framework that re-imagines code translation as a collaborative endeavor among multiple, compact LLMs. By orchestrating the interaction of specialized agents, each focused on different aspects of the translation process and grounded in a deep understanding of programming concepts, UniTranslator achieves a level of accuracy and efficiency that rivals larger, monolithic models. Our preliminary evaluation demonstrates the potential of UniTranslator to overcome the limitations of existing approaches and unlock the power of smaller LLMs for complex code translation tasks. We explore the effectiveness of this dynamic multi-agent paradigm in handling diverse language pairs, including low-resource languages, and in mitigating common issues such as code artifacts and hallucinations through the use of Natural Language Inference (NLI) grounding and iterative feedback mechanisms 

---
# TxAgent: An AI Agent for Therapeutic Reasoning Across a Universe of Tools 

**Authors**: Shanghua Gao, Richard Zhu, Zhenglun Kong, Ayush Noori, Xiaorui Su, Curtis Ginder, Theodoros Tsiligkaridis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2503.10970)  

**Abstract**: Precision therapeutics require multimodal adaptive models that generate personalized treatment recommendations. We introduce TxAgent, an AI agent that leverages multi-step reasoning and real-time biomedical knowledge retrieval across a toolbox of 211 tools to analyze drug interactions, contraindications, and patient-specific treatment strategies. TxAgent evaluates how drugs interact at molecular, pharmacokinetic, and clinical levels, identifies contraindications based on patient comorbidities and concurrent medications, and tailors treatment strategies to individual patient characteristics. It retrieves and synthesizes evidence from multiple biomedical sources, assesses interactions between drugs and patient conditions, and refines treatment recommendations through iterative reasoning. It selects tools based on task objectives and executes structured function calls to solve therapeutic tasks that require clinical reasoning and cross-source validation. The ToolUniverse consolidates 211 tools from trusted sources, including all US FDA-approved drugs since 1939 and validated clinical insights from Open Targets. TxAgent outperforms leading LLMs, tool-use models, and reasoning agents across five new benchmarks: DrugPC, BrandPC, GenericPC, TreatmentPC, and DescriptionPC, covering 3,168 drug reasoning tasks and 456 personalized treatment scenarios. It achieves 92.1% accuracy in open-ended drug reasoning tasks, surpassing GPT-4o and outperforming DeepSeek-R1 (671B) in structured multi-step reasoning. TxAgent generalizes across drug name variants and descriptions. By integrating multi-step inference, real-time knowledge grounding, and tool-assisted decision-making, TxAgent ensures that treatment recommendations align with established clinical guidelines and real-world evidence, reducing the risk of adverse events and improving therapeutic decision-making. 

---
# Learning to reset in target search problems 

**Authors**: Gorka Muñoz-Gil, Hans J. Briegel, Michele Caraglio  

**Link**: [PDF](https://arxiv.org/pdf/2503.11330)  

**Abstract**: Target search problems are central to a wide range of fields, from biological foraging to the optimization algorithms. Recently, the ability to reset the search has been shown to significantly improve the searcher's efficiency. However, the optimal resetting strategy depends on the specific properties of the search problem and can often be challenging to determine. In this work, we propose a reinforcement learning (RL)-based framework to train agents capable of optimizing their search efficiency in environments by learning how to reset. First, we validate the approach in a well-established benchmark: the Brownian search with resetting. There, RL agents consistently recover strategies closely resembling the sharp resetting distribution, known to be optimal in this scenario. We then extend the framework by allowing agents to control not only when to reset, but also their spatial dynamics through turning actions. In this more complex setting, the agents discover strategies that adapt both resetting and turning to the properties of the environment, outperforming the proposed benchmarks. These results demonstrate how reinforcement learning can serve both as an optimization tool and a mechanism for uncovering new, interpretable strategies in stochastic search processes with resetting. 

---
# Observation-Graph Interaction and Key-Detail Guidance for Vision and Language Navigation 

**Authors**: Yifan Xie, Binkai Ou, Fei Ma, Yaohua Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11006)  

**Abstract**: Vision and Language Navigation (VLN) requires an agent to navigate through environments following natural language instructions. However, existing methods often struggle with effectively integrating visual observations and instruction details during navigation, leading to suboptimal path planning and limited success rates. In this paper, we propose OIKG (Observation-graph Interaction and Key-detail Guidance), a novel framework that addresses these limitations through two key components: (1) an observation-graph interaction module that decouples angular and visual information while strengthening edge representations in the navigation space, and (2) a key-detail guidance module that dynamically extracts and utilizes fine-grained location and object information from instructions. By enabling more precise cross-modal alignment and dynamic instruction interpretation, our approach significantly improves the agent's ability to follow complex navigation instructions. Extensive experiments on the R2R and RxR datasets demonstrate that OIKG achieves state-of-the-art performance across multiple evaluation metrics, validating the effectiveness of our method in enhancing navigation precision through better observation-instruction alignment. 

---
# Zero-Shot Subject-Centric Generation for Creative Application Using Entropy Fusion 

**Authors**: Kaifeng Zou, Xiaoyi Feng, Peng Wang, Tao Huang, Zizhou Huang, Zhang Haihang, Yuntao Zou, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10697)  

**Abstract**: Generative models are widely used in visual content creation. However, current text-to-image models often face challenges in practical applications-such as textile pattern design and meme generation-due to the presence of unwanted elements that are difficult to separate with existing methods. Meanwhile, subject-reference generation has emerged as a key research trend, highlighting the need for techniques that can produce clean, high-quality subject images while effectively removing extraneous components. To address this challenge, we introduce a framework for reliable subject-centric image generation. In this work, we propose an entropy-based feature-weighted fusion method to merge the informative cross-attention features obtained from each sampling step of the pretrained text-to-image model FLUX, enabling a precise mask prediction and subject-centric generation. Additionally, we have developed an agent framework based on Large Language Models (LLMs) that translates users' casual inputs into more descriptive prompts, leading to highly detailed image generation. Simultaneously, the agents extract primary elements of prompts to guide the entropy-based feature fusion, ensuring focused primary element generation without extraneous components. Experimental results and user studies demonstrate our methods generates high-quality subject-centric images, outperform existing methods or other possible pipelines, highlighting the effectiveness of our approach. 

---
# IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models 

**Authors**: Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.10110)  

**Abstract**: Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g., brushing a soft pillow) to more dangerous (e.g., toppling a glass vase). Due to this diversity, it is difficult to characterize which contacts may be acceptable or unacceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach uses the VLM's outputs to produce a dense 3D "cost map" that encodes contact tolerances and seamlessly integrates with standard motion planners. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3620 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Supplementary material is available at this https URL. 

---
