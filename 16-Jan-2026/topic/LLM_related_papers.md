# Development of Ontological Knowledge Bases by Leveraging Large Language Models 

**Authors**: Le Ngoc Luyen, Marie-Hélène Abel, Philippe Gouspillou  

**Link**: [PDF](https://arxiv.org/pdf/2601.10436)  

**Abstract**: Ontological Knowledge Bases (OKBs) play a vital role in structuring domain-specific knowledge and serve as a foundation for effective knowledge management systems. However, their traditional manual development poses significant challenges related to scalability, consistency, and adaptability. Recent advancements in Generative AI, particularly Large Language Models (LLMs), offer promising solutions for automating and enhancing OKB development. This paper introduces a structured, iterative methodology leveraging LLMs to optimize knowledge acquisition, automate ontology artifact generation, and enable continuous refinement cycles. We demonstrate this approach through a detailed case study focused on developing a user context profile ontology within the vehicle sales domain. Key contributions include significantly accelerated ontology construction processes, improved ontological consistency, effective bias mitigation, and enhanced transparency in the ontology engineering process. Our findings highlight the transformative potential of integrating LLMs into ontology development, notably improving scalability, integration capabilities, and overall efficiency in knowledge management systems. 

---
# An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit 

**Authors**: Warren Jouanneau, Emma Jouffroy, Marc Palyart  

**Link**: [PDF](https://arxiv.org/pdf/2601.10321)  

**Abstract**: Finding the most relevant person for a job proposal in real time is challenging, especially when resumes are long, structured, and multilingual. In this paper, we propose a re-ranking model based on a new generation of late cross-attention architecture, that decomposes both resumes and project briefs to efficiently handle long-context inputs with minimal computational overhead. To mitigate historical data biases, we use a generative large language model (LLM) as a teacher, generating fine-grained, semantically grounded supervision. This signal is distilled into our student model via an enriched distillation loss function. The resulting model produces skill-fit scores that enable consistent and interpretable person-job matching. Experiments on relevance, ranking, and calibration metrics demonstrate that our approach outperforms state-of-the-art baselines. 

---
# Continuum Memory Architectures for Long-Horizon LLM Agents 

**Authors**: Joe Logan  

**Link**: [PDF](https://arxiv.org/pdf/2601.09913)  

**Abstract**: Retrieval-augmented generation (RAG) has become the default strategy for providing large language model (LLM) agents with contextual knowledge. Yet RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent. We define the \textit{Continuum Memory Architecture} (CMA), a class of systems that maintain and update internal state across interactions through persistent storage, selective retention, associative routing, temporal chaining, and consolidation into higher-order abstractions. Rather than disclosing implementation specifics, we specify the architectural requirements CMA imposes and show consistent behavioral advantages on tasks that expose RAG's structural inability to accumulate, mutate, or disambiguate memory. The empirical probes (knowledge updates, temporal association, associative recall, contextual disambiguation) demonstrate that CMA is a necessary architectural primitive for long-horizon agents while highlighting open challenges around latency, drift, and interpretability. 

---
# iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification 

**Authors**: Zhuoxuan Huang, Yunshan Ma, Hongyu Zhang, Hua Ma, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.10609)  

**Abstract**: Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored. To bridge this gap, we formally define the itinerary modification task and introduce iTIMO, a dataset specifically tailored for this purpose. We identify the lack of {\itshape need-to-modify} itinerary data as the critical bottleneck hindering research on this task and propose a general pipeline to overcome it. This pipeline frames the generation of such data as an intent-driven perturbation task. It instructs large language models to perturb real world itineraries using three atomic editing operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents, including disruptions of popularity, spatial distance, and category diversity. Furthermore, a hybrid evaluation metric is designed to ensure perturbation effectiveness. We conduct comprehensive experiments on iTIMO, revealing the limitations of current LLMs and lead to several valuable directions for future research. Dataset and corresponding code are available at this https URL. 

---
# SciNets: Graph-Constrained Multi-Hop Reasoning for Scientific Literature Synthesis 

**Authors**: Sauhard Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.09727)  

**Abstract**: Cross-domain scientific synthesis requires connecting mechanistic explanations across fragmented literature, a capability that remains challenging for both retrieval-based systems and unconstrained language models. While recent work has applied large language models to scientific summarization and question answering, these approaches provide limited control over reasoning depth and structural grounding. We frame mechanistic synthesis as a graph-constrained multi-hop reasoning problem over literature-derived concept graphs. Given a scientific query and a compact, query-local corpus, SciNets constructs a directed concept graph and synthesizes mechanistic explanations by identifying multi-hop reasoning paths that connect concepts that rarely co-occur within individual papers. We systematically compare shortest-path reasoning, k-shortest paths with diversity constraints, stochastic random walks, and a retrieval-augmented language model baseline. Rather than evaluating correctness, which is often indeterminate when synthesizing connections across distributed sources, we introduce a behavioral framework that measures symbolic reasoning depth, mechanistic diversity, and grounding stability. Across machine learning, biology, and climate science tasks, explicit graph constraints enable controllable multi-hop reasoning while revealing a consistent trade-off: deeper and more diverse symbolic reasoning increases grounding instability, whereas shortest-path reasoning remains highly stable but structurally conservative. These findings provide a systematic behavioral characterization of the limits and capabilities of current graph-LLM integration for scientific synthesis. 

---
# Introducing Axlerod: An LLM-based Chatbot for Assisting Independent Insurance Agents 

**Authors**: Adam Bradley, John Hastings, Khandaker Mamun Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.09715)  

**Abstract**: The insurance industry is undergoing a paradigm shift through the adoption of artificial intelligence (AI) technologies, particularly in the realm of intelligent conversational agents. Chatbots have evolved into sophisticated AI-driven systems capable of automating complex workflows, including policy recommendation and claims triage, while simultaneously enabling dynamic, context-aware user engagement. This paper presents the design, implementation, and empirical evaluation of Axlerod, an AI-powered conversational interface designed to improve the operational efficiency of independent insurance agents. Leveraging natural language processing (NLP), retrieval-augmented generation (RAG), and domain-specific knowledge integration, Axlerod demonstrates robust capabilities in parsing user intent, accessing structured policy databases, and delivering real-time, contextually relevant responses. Experimental results underscore Axlerod's effectiveness, achieving an overall accuracy of 93.18% in policy retrieval tasks while reducing the average search time by 2.42 seconds. This work contributes to the growing body of research on enterprise-grade AI applications in insurtech, with a particular focus on agent-assistive rather than consumer-facing architectures. 

---
# Detecting Winning Arguments with Large Language Models and Persuasion Strategies 

**Authors**: Tiziano Labruna, Arkadiusz Modzelewski, Giorgio Satta, Giovanni Da San Martino  

**Link**: [PDF](https://arxiv.org/pdf/2601.10660)  

**Abstract**: Detecting persuasion in argumentative text is a challenging task with important implications for understanding human communication. This work investigates the role of persuasion strategies - such as Attack on reputation, Distraction, and Manipulative wording - in determining the persuasiveness of a text. We conduct experiments on three annotated argument datasets: Winning Arguments (built from the Change My View subreddit), Anthropic/Persuasion, and Persuasion for Good. Our approach leverages large language models (LLMs) with a Multi-Strategy Persuasion Scoring approach that guides reasoning over six persuasion strategies. Results show that strategy-guided reasoning improves the prediction of persuasiveness. To better understand the influence of content, we organize the Winning Argument dataset into broad discussion topics and analyze performance across them. We publicly release this topic-annotated version of the dataset to facilitate future research. Overall, our methodology demonstrates the value of structured, strategy-aware prompting for enhancing interpretability and robustness in argument quality assessment. 

---
# Influential Training Data Retrieval for Explaining Verbalized Confidence of LLMs 

**Authors**: Yuxi Xia, Loris Schoenegger, Benjamin Roth  

**Link**: [PDF](https://arxiv.org/pdf/2601.10645)  

**Abstract**: Large language models (LLMs) can increase users' perceived trust by verbalizing confidence in their outputs. However, prior work has shown that LLMs are often overconfident, making their stated confidence unreliable since it does not consistently align with factual accuracy. To better understand the sources of this verbalized confidence, we introduce TracVC (\textbf{Trac}ing \textbf{V}erbalized \textbf{C}onfidence), a method that builds on information retrieval and influence estimation to trace generated confidence expressions back to the training data. We evaluate TracVC on OLMo and Llama models in a question answering setting, proposing a new metric, content groundness, which measures the extent to which an LLM grounds its confidence in content-related training examples (relevant to the question and answer) versus in generic examples of confidence verbalization. Our analysis reveals that OLMo2-13B is frequently influenced by confidence-related data that is lexically unrelated to the query, suggesting that it may mimic superficial linguistic expressions of certainty rather than rely on genuine content grounding. These findings point to a fundamental limitation in current training regimes: LLMs may learn how to sound confident without learning when confidence is justified. Our analysis provides a foundation for improving LLMs' trustworthiness in expressing more reliable confidence. 

---
# MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching 

**Authors**: Changle Qu, Sunhao Dai, Hengyi Cai, Jun Xu, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.10712)  

**Abstract**: Tool-Integrated Reasoning (TIR) empowers large language models (LLMs) to tackle complex tasks by interleaving reasoning steps with external tool interactions. However, existing reinforcement learning methods typically rely on outcome- or trajectory-level rewards, assigning uniform advantages to all steps within a trajectory. This coarse-grained credit assignment fails to distinguish effective tool calls from redundant or erroneous ones, particularly in long-horizon multi-turn scenarios. To address this, we propose MatchTIR, a framework that introduces fine-grained supervision via bipartite matching-based turn-level reward assignment and dual-level advantage estimation. Specifically, we formulate credit assignment as a bipartite matching problem between predicted and ground-truth traces, utilizing two assignment strategies to derive dense turn-level rewards. Furthermore, to balance local step precision with global task success, we introduce a dual-level advantage estimation scheme that integrates turn-level and trajectory-level signals, assigning distinct advantage values to individual interaction turns. Extensive experiments on three benchmarks demonstrate the superiority of MatchTIR. Notably, our 4B model surpasses the majority of 8B competitors, particularly in long-horizon and multi-turn tasks. Our codes are available at this https URL. 

---
# PERM: Psychology-grounded Empathetic Reward Modeling for Large Language Models 

**Authors**: Chengbing Wang, Wuqiang Zheng, Yang Zhang, Fengbin Zhu, Junyi Cheng, Yi Xie, Wenjie Wang, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2601.10532)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in human-centric applications, yet they often fail to provide substantive emotional support. While Reinforcement Learning (RL) has been utilized to enhance empathy of LLMs, existing reward models typically evaluate empathy from a single perspective, overlooking the inherently bidirectional interaction nature of empathy between the supporter and seeker as defined by Empathy Cycle theory. To address this limitation, we propose Psychology-grounded Empathetic Reward Modeling (PERM). PERM operationalizes empathy evaluation through a bidirectional decomposition: 1) Supporter perspective, assessing internal resonation and communicative expression; 2) Seeker perspective, evaluating emotional reception. Additionally, it incorporates a bystander perspective to monitor overall interaction quality. Extensive experiments on a widely-used emotional intelligence benchmark and an industrial daily conversation dataset demonstrate that PERM outperforms state-of-the-art baselines by over 10\%. Furthermore, a blinded user study reveals a 70\% preference for our approach, highlighting its efficacy in generating more empathetic responses. Our code, dataset, and models are available at this https URL. 

---
# The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models 

**Authors**: Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey  

**Link**: [PDF](https://arxiv.org/pdf/2601.10387)  

**Abstract**: Large language models can represent a variety of personas but typically default to a helpful Assistant identity cultivated during post-training. We investigate the structure of the space of model personas by extracting activation directions corresponding to diverse character archetypes. Across several different models, we find that the leading component of this persona space is an "Assistant Axis," which captures the extent to which a model is operating in its default Assistant mode. Steering towards the Assistant direction reinforces helpful and harmless behavior; steering away increases the model's tendency to identify as other entities. Moreover, steering away with more extreme values often induces a mystical, theatrical speaking style. We find this axis is also present in pre-trained models, where it primarily promotes helpful human archetypes like consultants and coaches and inhibits spiritual ones. Measuring deviations along the Assistant Axis predicts "persona drift," a phenomenon where models slip into exhibiting harmful or bizarre behaviors that are uncharacteristic of their typical persona. We find that persona drift is often driven by conversations demanding meta-reflection on the model's processes or featuring emotionally vulnerable users. We show that restricting activations to a fixed region along the Assistant Axis can stabilize model behavior in these scenarios -- and also in the face of adversarial persona-based jailbreaks. Our results suggest that post-training steers models toward a particular region of persona space but only loosely tethers them to it, motivating work on training and steering strategies that more deeply anchor models to a coherent persona. 

---
# DR-Arena: an Automated Evaluation Framework for Deep Research Agents 

**Authors**: Yiwen Gao, Ruochen Zhao, Yang Deng, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10504)  

**Abstract**: As Large Language Models (LLMs) increasingly operate as Deep Research (DR) Agents capable of autonomous investigation and information synthesis, reliable evaluation of their task performance has become a critical bottleneck. Current benchmarks predominantly rely on static datasets, which suffer from several limitations: limited task generality, temporal misalignment, and data contamination. To address these, we introduce DR-Arena, a fully automated evaluation framework that pushes DR agents to their capability limits through dynamic investigation. DR-Arena constructs real-time Information Trees from fresh web trends to ensure the evaluation rubric is synchronized with the live world state, and employs an automated Examiner to generate structured tasks testing two orthogonal capabilities: Deep reasoning and Wide coverage. DR-Arena further adopts Adaptive Evolvement Loop, a state-machine controller that dynamically escalates task complexity based on real-time performance, demanding deeper deduction or wider aggregation until a decisive capability boundary emerges. Experiments with six advanced DR agents demonstrate that DR-Arena achieves a Spearman correlation of 0.94 with the LMSYS Search Arena leaderboard. This represents the state-of-the-art alignment with human preferences without any manual efforts, validating DR-Arena as a reliable alternative for costly human adjudication. 

---
# GeoSteer: Faithful Chain-of-Thought Steering via Latent Manifold Gradients 

**Authors**: Kentaro Kazama, Daiki Shirafuji, Tatsuhiko Saito  

**Link**: [PDF](https://arxiv.org/pdf/2601.10229)  

**Abstract**: Recent advances in Large Language Models (LLMs) have improved multi-step reasoning. Most approaches rely on Chain-of-Thought (CoT) rationales. Previous studies have shown that LLMs often generate logically inconsistent reasoning steps even when their final answers are correct. These inconsistencies reduce the reliability of step-level reasoning. We propose GeoSteer, a manifold-based framework that improves the quality of intermediate reasoning. The method consists of: (1) constructing a CoT dataset with segment-level scores, (2) training a Variational Autoencoder (VAE) model and a quality estimation model to learn a low-dimensional manifold of high-quality CoT trajectories, and (3) steering hidden states of target LLMs toward higher-quality regions in the latent space. This update in a latent space behaves like a natural-gradient adjustment in the original hidden-state space. It ensures geometrically coherent steering. We evaluate GeoSteer on the GSM8k dataset using the Qwen3 series. We measure via answer accuracy and overall reasoning performance. GeoSteer improved the exact match accuracy by up to 2.6 points. It also enhanced the pairwise win rate by 5.3 points. These results indicate that GeoSteer provides an effective and controllable mechanism for improving the quality of intermediate reasoning in LLMs. 

---
# HUMANLLM: Benchmarking and Reinforcing LLM Anthropomorphism via Human Cognitive Patterns 

**Authors**: Xintao Wang, Jian Yang, Weiyuan Li, Rui Xie, Jen-tse Huang, Jun Gao, Shuai Huang, Yueping Kang, Liyuan Gou, Hongwei Feng, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10198)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and generation, serving as the foundation for advanced persona simulation and Role-Playing Language Agents (RPLAs). However, achieving authentic alignment with human cognitive and behavioral patterns remains a critical challenge for these agents. We present HUMANLLM, a framework treating psychological patterns as interacting causal forces. We construct 244 patterns from ~12,000 academic papers and synthesize 11,359 scenarios where 2-5 patterns reinforce, conflict, or modulate each other, with multi-turn conversations expressing inner thoughts, actions, and dialogue. Our dual-level checklists evaluate both individual pattern fidelity and emergent multi-pattern dynamics, achieving strong human alignment (r=0.91) while revealing that holistic metrics conflate simulation accuracy with social desirability. HUMANLLM-8B outperforms Qwen3-32B on multi-pattern dynamics despite 4x fewer parameters, demonstrating that authentic anthropomorphism requires cognitive modeling--simulating not just what humans do, but the psychological processes generating those behaviors. 

---
# Loop as a Bridge: Can Looped Transformers Truly Link Representation Space and Natural Language Outputs? 

**Authors**: Guanxu Chen, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10242)  

**Abstract**: Large Language Models (LLMs) often exhibit a gap between their internal knowledge and their explicit linguistic outputs. In this report, we empirically investigate whether Looped Transformers (LTs)--architectures that increase computational depth by iterating shared layers--can bridge this gap by utilizing their iterative nature as a form of introspection. Our experiments reveal that while increasing loop iterations narrows the gap, it is partly driven by a degradation of their internal knowledge carried by representations. Moreover, another empirical analysis suggests that current LTs' ability to perceive representations does not improve across loops; it is only present in the final loop. These results suggest that while LTs offer a promising direction for scaling computational depth, they have yet to achieve the introspection required to truly link representation space and natural language. 

---
# Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment 

**Authors**: Cameron Tice, Puria Radmard, Samuel Ratnam, Andy Kim, David Africa, Kyle O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2601.10160)  

**Abstract**: Pretraining corpora contain extensive discourse about AI systems, yet the causal influence of this discourse on downstream alignment remains poorly understood. If prevailing descriptions of AI behaviour are predominantly negative, LLMs may internalise corresponding behavioural priors, giving rise to self-fulfilling misalignment. This paper provides the first controlled study of this hypothesis by pretraining 6.9B-parameter LLMs with varying amounts of (mis)alignment discourse. We find that discussion of AI contributes to misalignment. Upsampling synthetic training documents about AI misalignment leads to a notable increase in misaligned behaviour. Conversely, upsampling documents about aligned behaviour reduces misalignment scores from 45% to 9%. We consider this evidence of self-fulfilling alignment. These effects are dampened, but persist through post-training. Our findings establish the study of how pretraining data shapes alignment priors, or alignment pretraining, as a complement to post-training. We recommend practitioners pretrain for alignment as well as capabilities. Our models and datasets are available at this http URL 

---
# ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback 

**Authors**: Yutao Mou, Zhangchi Xue, Lijun Li, Peiyang Liu, Shikun Zhang, Wei Ye, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10156)  

**Abstract**: While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks. 

---
# Role-Playing Agents Driven by Large Language Models: Current Status, Challenges, and Future Trends 

**Authors**: Ye Wang, Jiaxing Chen, Hongjiang Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10122)  

**Abstract**: In recent years, with the rapid advancement of large language models (LLMs), role-playing language agents (RPLAs) have emerged as a prominent research focus at the intersection of natural language processing (NLP) and human-computer interaction. This paper systematically reviews the current development and key technologies of RPLAs, delineating the technological evolution from early rule-based template paradigms, through the language style imitation stage, to the cognitive simulation stage centered on personality modeling and memory mechanisms. It summarizes the critical technical pathways supporting high-quality role-playing, including psychological scale-driven character modeling, memory-augmented prompting mechanisms, and motivation-situation-based behavioral decision control. At the data level, the paper further analyzes the methods and challenges of constructing role-specific corpora, focusing on data sources, copyright constraints, and structured annotation processes. In terms of evaluation, it collates multi-dimensional assessment frameworks and benchmark datasets covering role knowledge, personality fidelity, value alignment, and interactive hallucination, while commenting on the advantages and disadvantages of methods such as human evaluation, reward models, and LLM-based scoring. Finally, the paper outlines future development directions of role-playing agents, including personality evolution modeling, multi-agent collaborative narrative, multimodal immersive interaction, and integration with cognitive neuroscience, aiming to provide a systematic perspective and methodological insights for subsequent research. 

---
# Credit C-GPT: A Domain-Specialized Large Language Model for Conversational Understanding in Vietnamese Debt Collection 

**Authors**: Nhung Nguyen Thi Hong, Cuong Nguyen Dang, Tri Le Ngoc  

**Link**: [PDF](https://arxiv.org/pdf/2601.10167)  

**Abstract**: Debt collection is a critical function within the banking, financial services, and insurance (BFSI) sector, relying heavily on large-scale human-to-human conversational interactions conducted primarily in Vietnamese contact centers. These conversations involve informal spoken language, emotional variability, and complex domain-specific reasoning, which pose significant challenges for traditional natural language processing systems. This paper introduces Credit C-GPT, a domain-specialized large language model with seven billion parameters, fine-tuned for conversational understanding in Vietnamese debt collection scenarios. The proposed model integrates multiple conversational intelligence tasks, including dialogue understanding, sentiment recognition, intent detection, call stage classification, and structured slot-value extraction, within a single reasoning-based framework. We describe the data construction process, annotation strategy, and training methodology, and evaluate the model on proprietary human-annotated datasets. Experimental results show consistent improvements over traditional pipeline-based approaches, indicating that domain-specialized conversational language models provide a scalable and privacy-aware solution for real-time assistance and post-call analytics in enterprise contact centers. 

---
# HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning 

**Authors**: Ziang Cui, Mengran Yu, Tianjiao Li, Chenyu Shi, Yingxuan Shi, Lusheng Zhang, Hongwei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2601.10187)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy. 

---
# Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation 

**Authors**: Lechen Zhang, Yunxiang Zhang, Wei Hu, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10109)  

**Abstract**: Large reasoning models such as DeepSeek-R1 and their distilled variants achieve strong performance on complex reasoning tasks. Yet, distilling these models often demands large-scale data for supervised fine-tuning (SFT), motivating the pursuit of data-efficient training methods. To address this, we propose a skill-centric distillation framework that efficiently transfers reasoning ability to weaker models with two components: (1) Skill-based data selection, which prioritizes examples targeting the student model's weaker skills, and (2) Skill-aware fine-tuning, which encourages explicit skill decomposition during problem solving. With only 1,000 training examples selected from a 100K teacher-generated corpus, our method surpasses random SFT baselines by +1.6% on Qwen3-4B and +1.4% on Qwen3-8B across five mathematical reasoning benchmarks. Further analysis confirms that these gains concentrate on skills emphasized during training, highlighting the effectiveness of skill-centric training for efficient reasoning distillation. 

---
# CALM-IT: Generating Realistic Long-Form Motivational Interviewing Dialogues with Dual-Actor Conversational Dynamics Tracking 

**Authors**: Viet Cuong Nguyen, Nhi Yen Nguyen, Kristin A. Candan, Mary Conlon, Vanessa Rumie, Kristen Risola, Srijan Kumar, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2601.10085)  

**Abstract**: Large Language Models (LLMs) are increasingly used in mental health-related settings, yet they struggle to sustain realistic, goal-directed dialogue over extended interactions. While LLMs generate fluent responses, they optimize locally for the next turn rather than maintaining a coherent model of therapeutic progress, leading to brittleness and long-horizon drift. We introduce CALM-IT, a framework for generating and evaluating long-form Motivational Interviewing (MI) dialogues that explicitly models dual-actor conversational dynamics. CALM-IT represents therapist-client interaction as a bidirectional state-space process, in which both agents continuously update inferred alignment, mental states, and short-term goals to guide strategy selection and utterance generation. Across large-scale evaluations, CALM-IT consistently outperforms strong baselines in Effectiveness and Goal Alignment and remains substantially more stable as conversation length increases. Although CALM-IT initiates fewer therapist redirections, it achieves the highest client acceptance rate (64.3%), indicating more precise and therapeutically aligned intervention timing. Overall, CALM-IT provides evidence for modeling evolving conversational state being essential for generating high-quality long-form synthetic conversations. 

---
# Long-Chain Reasoning Distillation via Adaptive Prefix Alignment 

**Authors**: Zhenghao Liu, Zhuoyang Wu, Xinze Li, Yukun Yan, Shuo Wang, Zulong Chen, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.10064)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities, particularly in solving complex mathematical problems. Recent studies show that distilling long reasoning trajectories can effectively enhance the reasoning performance of small-scale student models. However, teacher-generated reasoning trajectories are often excessively long and structurally complex, making them difficult for student models to learn. This mismatch leads to a gap between the provided supervision signal and the learning capacity of the student model. To address this challenge, we propose Prefix-ALIGNment distillation (P-ALIGN), a framework that fully exploits teacher CoTs for distillation through adaptive prefix alignment. Specifically, P-ALIGN adaptively truncates teacher-generated reasoning trajectories by determining whether the remaining suffix is concise and sufficient to guide the student model. Then, P-ALIGN leverages the teacher-generated prefix to supervise the student model, encouraging effective prefix alignment. Experiments on multiple mathematical reasoning benchmarks demonstrate that P-ALIGN outperforms all baselines by over 3%. Further analysis indicates that the prefixes constructed by P-ALIGN provide more effective supervision signals, while avoiding the negative impact of redundant and uncertain reasoning components. All code is available at this https URL. 

---
# Take Out Your Calculators: Estimating the Real Difficulty of Question Items with LLM Student Simulations 

**Authors**: Christabel Acquaye, Yi Ting Huang, Marine Carpuat, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2601.09953)  

**Abstract**: Standardized math assessments require expensive human pilot studies to establish the difficulty of test items. We investigate the predictive value of open-source large language models (LLMs) for evaluating the difficulty of multiple-choice math questions for real-world students. We show that, while LLMs are poor direct judges of problem difficulty, simulation-based approaches with LLMs yield promising results under the right conditions. Under the proposed approach, we simulate a "classroom" of 4th, 8th, or 12th grade students by prompting the LLM to role-play students of varying proficiency levels. We use the outcomes of these simulations to fit Item Response Theory (IRT) models, comparing learned difficulty parameters for items to their real-world difficulties, as determined by item-level statistics furnished by the National Assessment of Educational Progress (NAEP). We observe correlations as high as 0.75, 0.76, and 0.82 for grades 4, 8, and 12, respectively. In our simulations, we experiment with different "classroom sizes," showing tradeoffs between computation size and accuracy. We find that role-plays with named students improves predictions (compared to student ids), and stratifying names across gender and race further improves predictions. Our results show that LLMs with relatively weaker mathematical abilities (Gemma) actually yield better real-world difficulty predictions than mathematically stronger models (Llama and Qwen), further underscoring the suitability of open-source models for the task. 

---
# EmplifAI: a Fine-grained Dataset for Japanese Empathetic Medical Dialogues in 28 Emotion Labels 

**Authors**: Wan Jou She, Lis Kanashiro Pereira, Fei Cheng, Sakiko Yahata, Panote Siriaraya, Eiji Aramaki  

**Link**: [PDF](https://arxiv.org/pdf/2601.10033)  

**Abstract**: This paper introduces EmplifAI, a Japanese empathetic dialogue dataset designed to support patients coping with chronic medical conditions. They often experience a wide range of positive and negative emotions (e.g., hope and despair) that shift across different stages of disease management. EmplifAI addresses this complexity by providing situation-based dialogues grounded in 28 fine-grained emotion categories, adapted and validated from the GoEmotions taxonomy. The dataset includes 280 medically contextualized situations and 4125 two-turn dialogues, collected through crowdsourcing and expert review. To evaluate emotional alignment in empathetic dialogues, we assessed model predictions on situation--dialogue pairs using BERTScore across multiple large language models (LLMs), achieving F1 scores of 0.83. Fine-tuning a baseline Japanese LLM (LLM-jp-3.1-13b-instruct4) with EmplifAI resulted in notable improvements in fluency, general empathy, and emotion-specific empathy. Furthermore, we compared the scores assigned by LLM-as-a-Judge and human raters on dialogues generated by multiple LLMs to validate our evaluation pipeline and discuss the insights and potential risks derived from the correlation analysis. 

---
# MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication 

**Authors**: Sraavya Sambara, Yuan Pu, Ayman Ali, Vishala Mishra, Lionel Wong, Monica Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2601.09853)  

**Abstract**: Real-world health questions from patients often unintentionally embed false assumptions or premises. In such cases, safe medical communication typically involves redirection: addressing the implicit misconception and then responding to the underlying patient context, rather than the original question. While large language models (LLMs) are increasingly being used by lay users for medical advice, they have not yet been tested for this crucial competency. Therefore, in this work, we investigate how LLMs react to false premises embedded within real-world health questions. We develop a semi-automated pipeline to curate MedRedFlag, a dataset of 1100+ questions sourced from Reddit that require redirection. We then systematically compare responses from state-of-the-art LLMs to those from clinicians. Our analysis reveals that LLMs often fail to redirect problematic questions, even when the problematic premise is detected, and provide answers that could lead to suboptimal medical decision making. Our benchmark and results reveal a novel and substantial gap in how LLMs perform under the conditions of real-world health communication, highlighting critical safety concerns for patient-facing medical AI systems. Code and dataset are available at this https URL. 

---
# From Detection to Diagnosis: Advancing Hallucination Analysis with Automated Data Synthesis 

**Authors**: Yanyi Liu, Qingwen Yang, Tiezheng Guo, Feiyu Qu, Jun Liu, Yingyou Wen  

**Link**: [PDF](https://arxiv.org/pdf/2601.09734)  

**Abstract**: Hallucinations in Large Language Models (LLMs), defined as the generation of content inconsistent with facts or context, represent a core obstacle to their reliable deployment in critical domains. Current research primarily focuses on binary "detection" approaches that, while capable of identifying hallucinations, fail to provide interpretable and actionable feedback for model improvement, thus limiting practical utility. To address this limitation, a new research paradigm is proposed, shifting from "detection" to "diagnosis". The Hallucination Diagnosis Task is introduced, a task which requires models to not only detect hallucinations, but also perform error localization, causal explanation, and content correction. We develop the Hallucination Diagnosis Generator (HDG), an automated pipeline that systematically generates high-quality training samples with rich diagnostic metadata from raw corpora through multi-dimensional augmentation strategies including controlled fact fabrication and reasoning chain perturbation. Using HDG-generated data, we train HDM-4B-RL, a 4-billion-parameter hallucination diagnosis model, employing Group Relative Policy Optimization (GRPO) with a comprehensive reward function incorporating structural, accuracy, and localization signals. Experimental results demonstrate that our model surpasses previous state-of-the-art detection models on the HaluEval benchmark while achieving comparable performance to advanced general-purpose models. In comprehensive diagnosis tasks, HDM-4B-RL matches the capabilities of larger general models while maintaining a smaller size. This work validates the feasibility and value of hallucination diagnosis, providing an effective methodology for building more trustworthy and reliable generative AI systems. 

---
# Stable and Explainable Personality Trait Evaluation in Large Language Models with Internal Activations 

**Authors**: Xiaoxu Ma, Xiangbo Zhang, Zhenyu Weng  

**Link**: [PDF](https://arxiv.org/pdf/2601.09833)  

**Abstract**: Evaluating personality traits in Large Language Models (LLMs) is key to model interpretation, comparison, and responsible deployment. However, existing questionnaire-based evaluation methods exhibit limited stability and offer little explainability, as their results are highly sensitive to minor variations in prompt phrasing or role-play configurations. To address these limitations, we propose an internal-activation-based approach, termed Persona-Vector Neutrality Interpolation (PVNI), for stable and explainable personality trait evaluation in LLMs. PVNI extracts a persona vector associated with a target personality trait from the model's internal activations using contrastive prompts. It then estimates the corresponding neutral score by interpolating along the persona vector as an anchor axis, enabling an interpretable comparison between the neutral prompt representation and the persona direction. We provide a theoretical analysis of the effectiveness and generalization properties of PVNI. Extensive experiments across diverse LLMs demonstrate that PVNI yields substantially more stable personality trait evaluations than existing methods, even under questionnaire and role-play variants. 

---
# Eliminating Agentic Workflow for Introduction Generation with Parametric Stage Tokens 

**Authors**: Meicong Zhang, Tiancheng su, Guoxiu He  

**Link**: [PDF](https://arxiv.org/pdf/2601.09728)  

**Abstract**: In recent years, using predefined agentic workflows to guide large language models (LLMs) for literature classification and review has become a research focus. However, writing research introductions is more challenging. It requires rigorous logic, coherent structure, and abstract summarization. Existing workflows often suffer from long reasoning chains, error accumulation, and reduced textual coherence. To address these limitations, we propose eliminating external agentic workflows. Instead, we directly parameterize their logical structure into the LLM. This allows the generation of a complete introduction in a single inference. To this end, we introduce the Stage Token for Introduction Generation (STIG). STIG converts the multiple stages of the original workflow into explicit stage signals. These signals guide the model to follow different logical roles and functions during generation. Through instruction tuning, the model learns the mapping between stage tokens and text functions. It also learns the logical order and transition patterns between stages, encoding this knowledge into the model parameters. Experimental results show that STIG can generate multi-stage text in a single inference. It does not require explicit workflow calls. STIG outperforms traditional agentic workflows and other baselines on metrics of semantic similarity and sentence-level structural rationality. The code is provided in the Supplementary Materials. 

---
# Clinical Document Metadata Extraction: A Scoping Review 

**Authors**: Kurt Miller, Qiuhao Lu, William Hersh, Kirk Roberts, Steven Bedrick, Andrew Wen, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09730)  

**Abstract**: Clinical document metadata, such as document type, structure, author role, medical specialty, and encounter setting, is essential for accurate interpretation of information captured in clinical documents. However, vast documentation heterogeneity and drift over time challenge harmonization of document metadata. Automated extraction methods have emerged to coalesce metadata from disparate practices into target schema. This scoping review aims to catalog research on clinical document metadata extraction, identify methodological trends and applications, and highlight gaps. We followed the PRISMA-ScR (Preferred Reporting Items for Systematic Reviews and Meta-Analyses Extension for Scoping Reviews) guidelines to identify articles that perform clinical document metadata extraction. We initially found and screened 266 articles published between January 2011 and August 2025, then comprehensively reviewed 67 we deemed relevant to our study. Among the articles included, 45 were methodological, 17 used document metadata as features in a downstream application, and 5 analyzed document metadata composition. We observe myriad purposes for methodological study and application types. Available labelled public data remains sparse except for structural section datasets. Methods for extracting document metadata have progressed from largely rule-based and traditional machine learning with ample feature engineering to transformer-based architectures with minimal feature engineering. The emergence of large language models has enabled broader exploration of generalizability across tasks and datasets, allowing the possibility of advanced clinical text processing systems. We anticipate that research will continue to expand into richer document metadata representations and integrate further into clinical applications and workflows. 

---
# Assessing and Improving Punctuation Robustness in English-Marathi Machine Translation 

**Authors**: Kaustubh Shivshankar Shejole, Sourabh Deoghare, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2601.09725)  

**Abstract**: Punctuation plays a critical role in resolving semantic and structural ambiguity in written language. Machine Translation (MT) systems are now widely applied across diverse domains and languages, including many low-resource settings. In this work, we focus on Marathi, a low- to middle-resource language. We introduce Virām, the first diagnostic benchmark for assessing punctuation robustness in English-to-Marathi machine translation, consisting of 54 manually curated, punctuation-ambiguous instances. We evaluate two primary strategies for enhancing reliability: a pipeline-based restore-then-translate approach and direct fine-tuned on punctuation-varied data. Our results demonstrate that specialized fine-tuned models and pipeline systems significantly improve translation quality over standard baselines on the Virām benchmark. Qualitative analysis reveals that the original model may result in wrong translations leading to wrong interpretations, while fine-tuned models significantly improve overall reliability. Furthermore, we find that current Large Language Models (LLMs) lag behind these task-specific approaches in preserving meaning for punctuation-ambiguous text, thus necessitating further research in this area. 

---
# Forgetting as a Feature: Cognitive Alignment of Large Language Models 

**Authors**: Hien Tran, Quinten Steenhuis, Alexandros Christoforos, Chadbourne Davis  

**Link**: [PDF](https://arxiv.org/pdf/2601.09726)  

**Abstract**: Large Language Models (LLMs) are often evaluated against ideals of perfect Bayesian inference, yet growing evidence suggests that their in-context reasoning exhibits systematic forgetting of past information. Rather than viewing this behavior as a limitation, we reinterpret forgetting as a functional cognitive mechanism. Drawing inspiration from human memory dynamics, we model LLM inference as a probabilistic memory process governed by exponential decay. We introduce a benchmark suite that evaluates temporal reasoning, concept drift adaptation, and associative recall, enabling direct comparison between model behavior and human cognitive patterns. Our empirical results reveal that LLMs demonstrate forgetting rates analogous to human memory efficiency trade-offs between stability and adaptability. Building on these observations, we propose probabilistic memory prompting, a lightweight strategy that shapes evidence integration to mimic human-like memory decay, leading to improved long-horizon reasoning performance. Our findings position forgetting not as a failure mode, but as a principled mechanism for adaptive intelligence. 

---
# Syntactic Framing Fragility: An Audit of Robustness in LLM Ethical Decisions 

**Authors**: Katherine Elkins, Jon Chun  

**Link**: [PDF](https://arxiv.org/pdf/2601.09724)  

**Abstract**: Large language models (LLMs) are increasingly deployed in consequential decision-making settings, yet their robustness to benign prompt variation remains underexplored. In this work, we study whether LLMs maintain consistent ethical judgments across logically equivalent but syntactically different prompts, focusing on variations involving negation and conditional structure. We introduce Syntactic Framing Fragility (SFF), a robustness evaluation framework that isolates purely syntactic effects via Logical Polarity Normalization (LPN), enabling direct comparison of decisions across positive and negative framings without semantic drift. Auditing 23 state-of-the-art models spanning the U.S. and China as well as small U.S. open-source software models over 14 ethical scenarios and four controlled framings (39,975 decisions), we find widespread and statistically significant inconsistency: many models reverse ethical endorsements solely due to syntactic polarity, with open-source models exhibiting over twice the fragility of commercial counterparts. We further uncover extreme negation sensitivity, where some models endorse actions in 80-97% of cases when explicitly prompted with "should not." We show that eliciting chain-of-thought reasoning substantially reduces fragility, identifying a practical mitigation lever, and we map fragility across scenarios, finding higher risk in financial and business contexts than in medical scenarios. Our results demonstrate that syntactic consistency constitutes a distinct and critical dimension of ethical robustness, and we argue that SFF-style audits should be a standard component of safety evaluation for deployed LLMs. Code and results will be available on this http URL. 

---
# SALP-CG: Standard-Aligned LLM Pipeline for Classifying and Grading Large Volumes of Online Conversational Health Data 

**Authors**: Yiwei Yan, Hao Li, Hua He, Gong Kai, Zhengyi Yang, Guanfeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09717)  

**Abstract**: Online medical consultations generate large volumes of conversational health data that often embed protected health information, requiring robust methods to classify data categories and assign risk levels in line with policies and practice. However, existing approaches lack unified standards and reliable automated methods to fulfill sensitivity classification for such conversational health data. This study presents a large language model-based extraction pipeline, SALP-CG, for classifying and grading privacy risks in online conversational health data. We concluded health-data classification and grading rules in accordance with GB/T 39725-2020. Combining few-shot guidance, JSON Schema constrained decoding, and deterministic high-risk rules, the backend-agnostic extraction pipeline achieves strong category compliance and reliable sensitivity across diverse LLMs. On the MedDialog-CN benchmark, models yields robust entity counts, high schema compliance, and accurate sensitivity grading, while the strongest model attains micro-F1=0.900 for maximum-level prediction. The category landscape stratified by sensitivity shows that Level 2-3 items dominate, enabling re-identification when combined; Level 4-5 items are less frequent but carry outsize harm. SALP-CG reliably helps classify categories and grading sensitivity in online conversational health data across LLMs, offering a practical method for health data governance. Code is available at this https URL. 

---
# Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models 

**Authors**: Hoyoon Byun, Youngjun Choi, Taero Kim, Sungrae Park, Kyungwoo Song  

**Link**: [PDF](https://arxiv.org/pdf/2601.09719)  

**Abstract**: Pre-Layer Normalization (Pre-LN) is the de facto choice for large language models (LLMs) and is crucial for stable pretraining and effective transfer learning. However, Pre-LN is inefficient due to repeated statistical calculations and suffers from the curse of depth. As layers grow, the magnitude and variance of the hidden state escalate, destabilizing training. Efficiency-oriented normalization-free methods such as Dynamic Tanh (DyT) improve speed but remain fragile at depth. To jointly address stability and efficiency, we propose Bounded Hyperbolic Tanh (BHyT), a drop-in replacement for Pre-LN. BHyT couples a tanh nonlinearity with explicit, data-driven input bounding to keep activations within a non-saturating range. It prevents depth-wise growth in activation magnitude and variance and comes with a theoretical stability guarantee. For efficiency, BHyT computes exact statistics once per block and replaces a second normalization with a lightweight variance approximation, enhancing efficiency. Empirically, BHyT demonstrates improved stability and efficiency during pretraining, achieving an average of 15.8% faster training and an average of 4.2% higher token generation throughput compared to RMSNorm., while matching or surpassing its inference performance and robustness across language understanding and reasoning benchmarks. Our code is available at: this https URL 

---
# Evaluating Novelty in AI-Generated Research Plans Using Multi-Workflow LLM Pipelines 

**Authors**: Devesh Saraogi, Rohit Singhee, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.09714)  

**Abstract**: The integration of Large Language Models (LLMs) into the scientific ecosystem raises fundamental questions about the creativity and originality of AI-generated research. Recent work has identified ``smart plagiarism'' as a concern in single-step prompting approaches, where models reproduce existing ideas with terminological shifts. This paper investigates whether agentic workflows -- multi-step systems employing iterative reasoning, evolutionary search, and recursive decomposition -- can generate more novel and feasible research plans. We benchmark five reasoning architectures: Reflection-based iterative refinement, Sakana AI v2 evolutionary algorithms, Google Co-Scientist multi-agent framework, GPT Deep Research (GPT-5.1) recursive decomposition, and Gemini~3 Pro multimodal long-context pipeline. Using evaluations from thirty proposals each on novelty, feasibility, and impact, we find that decomposition-based and long-context workflows achieve mean novelty of 4.17/5, while reflection-based approaches score significantly lower (2.33/5). Results reveal varied performance across research domains, with high-performing workflows maintaining feasibility without sacrificing creativity. These findings support the view that carefully designed multi-stage agentic workflows can advance AI-assisted research ideation. 

---
# StatLLaMA: A multi-stage training framework for building a domain-optimized statistical language model 

**Authors**: Jing-Yi Zeng, Guan-Hua Huang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09718)  

**Abstract**: This study investigates how to efficiently build a domain-specialized large language model (LLM) for statistics using the lightweight LLaMA-3.2-3B family as the foundation model (FM). We systematically compare three multi-stage training pipelines, starting from a base FM with no instruction-following capability, a base FM augmented with post-hoc instruction tuning, and an instruction-tuned FM with strong general reasoning abilities across continual pretraining, supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF) preference alignment, and downstream task adaptation. Results show that pipelines beginning with a base FM fail to develop meaningful statistical reasoning, even after extensive instruction tuning, SFT, or RLHF alignment. In contrast, starting from LLaMA-3.2-3B-Instruct enables effective domain specialization. A comprehensive evaluation of SFT variants reveals clear trade-offs between domain expertise and general reasoning ability. We further demonstrate that direct preference optimization provides stable and effective RLHF preference alignment. Finally, we show that downstream fine-tuning must be performed with extremely low intensity to avoid catastrophic forgetting in highly optimized models. The final model, StatLLaMA, achieves strong and balanced performance on benchmarks of mathematical reasoning, common-sense reasoning, and statistical expertise, offering a practical blueprint for developing resource-efficient statistical LLMs. The code is available at this https URL. 

---
# SuS: Strategy-aware Surprise for Intrinsic Exploration 

**Authors**: Mark Kashirskiy, Ilya Makarov  

**Link**: [PDF](https://arxiv.org/pdf/2601.10349)  

**Abstract**: We propose Strategy-aware Surprise (SuS), a novel intrinsic motivation framework that uses pre-post prediction mismatch as a novelty signal for exploration in reinforcement learning. Unlike traditional curiosity-driven methods that rely solely on state prediction error, SuS introduces two complementary components: Strategy Stability (SS) and Strategy Surprise (SuS). SS measures consistency in behavioral strategy across temporal steps, while SuS captures unexpected outcomes relative to the agent's current strategy representation. Our combined reward formulation leverages both signals through learned weighting coefficients. We evaluate SuS on mathematical reasoning tasks using large language models, demonstrating significant improvements in both accuracy and solution diversity. Ablation studies confirm that removing either component results in at least 10% performance degradation, validating the synergistic nature of our approach. SuS achieves 17.4% improvement in Pass@1 and 26.4% improvement in Pass@5 compared to baseline methods, while maintaining higher strategy diversity throughout training. 

---
# LLM-Driven Preference Data Synthesis for Proactive Prediction of the Next User Utterance in Human-Machine Dialogue 

**Authors**: Jinqiang Wang, Huansheng Ning, Jianguo Ding, Tao Zhu, Liming Chen, Chris Nugent  

**Link**: [PDF](https://arxiv.org/pdf/2601.09713)  

**Abstract**: Proactively predicting a users next utterance in human-machine dialogue can streamline interaction and improve user experience. Existing commercial API-based solutions are subject to privacy concerns while deploying general-purpose LLMs locally remains computationally expensive. As such, training a compact, task-specific LLM provides a practical alternative. Although user simulator methods can predict a user's next utterance, they mainly imitate their speaking style rather than advancing the dialogue. Preference data synthesis has been investigated to generate data for proactive next utterance prediction and help align LLMs with user preferences. Yet existing methods lack the ability to explicitly model the intent reasoning that leads to the user's next utterance and to define and synthesize preference and non-preference reasoning processes for predicting the user's next this http URL address these challenges, we propose ProUtt, an LLM-driven preference data synthesis method for proactive next utterance prediction. ProUtt converts dialogue history into an intent tree and explicitly models intent reasoning trajectories by predicting the next plausible path from both exploitation and exploration perspectives. It then constructs preference and non-preference reasoning processes by perturbing or revising intent tree paths at different future turns. Extensive evaluations using LLM-as-a-judge and human judgments demonstrate that ProUtt consistently outperforms existing data synthesis methods, user simulators, and commercial LLM APIs across four benchmark datasets. We release both the code and the synthesized datasets to facilitate future research. 

---
# Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing 

**Authors**: Yinzhi Zhao, Ming Wang, Shi Feng, Xiaocui Yang, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10543)  

**Abstract**: Large language models (LLMs) have achieved impressive performance across natural language tasks and are increasingly deployed in real-world applications. Despite extensive safety alignment efforts, recent studies show that such alignment is often shallow and remains vulnerable to jailbreak attacks. Existing defense mechanisms, including decoding-based constraints and post-hoc content detectors, struggle against sophisticated jailbreaks, often intervening robust detection or excessively degrading model utility. In this work, we examine the decoding process of LLMs and make a key observation: even when successfully jailbroken, models internally exhibit latent safety-related signals during generation. However, these signals are overridden by the model's drive for fluent continuation, preventing timely self-correction or refusal. Building on this observation, we propose a simple yet effective approach that explicitly surfaces and leverages these latent safety signals for early detection of unsafe content during decoding. Experiments across diverse jailbreak attacks demonstrate that our approach significantly enhances safety, while maintaining low over-refusal rates on benign inputs and preserving response quality. Our results suggest that activating intrinsic safety-awareness during decoding offers a promising and complementary direction for defending against jailbreak attacks. Code is available at: this https URL. 

---
# A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5 

**Authors**: Xingjun Ma, Yixu Wang, Hengyuan Xu, Yutao Wu, Yifan Ding, Yunhan Zhao, Zilong Wang, Jiabin Hua, Ming Wen, Jianan Liu, Ranjie Duan, Yifeng Gao, Yingshui Tan, Yunhao Chen, Hui Xue, Xin Wang, Wei Cheng, Jingjing Chen, Zuxuan Wu, Bo Li, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10527)  

**Abstract**: The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has produced substantial gains in reasoning, perception, and generative capability across language and vision. However, whether these advances yield commensurate improvements in safety remains unclear, in part due to fragmented evaluation practices limited to single modalities or threat models. In this report, we present an integrated safety evaluation of 7 frontier models: GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5. We evaluate each model across language, vision-language, and image generation settings using a unified protocol that integrates benchmark evaluation, adversarial evaluation, multilingual evaluation, and compliance evaluation. Aggregating our evaluations into safety leaderboards and model safety profiles across multiple evaluation modes reveals a sharply heterogeneous safety landscape. While GPT-5.2 demonstrates consistently strong and balanced safety performance across evaluations, other models exhibit pronounced trade-offs among benchmark safety, adversarial alignment, multilingual generalization, and regulatory compliance. Both language and vision-language modalities show significant vulnerability under adversarial evaluation, with all models degrading substantially despite strong results on standard benchmarks. Text-to-image models achieve relatively stronger alignment in regulated visual risk categories, yet remain brittle under adversarial or semantically ambiguous prompts. Overall, these results show that safety in frontier models is inherently multidimensional--shaped by modality, language, and evaluation scheme, underscoring the need for standardized safety evaluations to accurately assess real-world risk and guide responsible model development and deployment. 

---
# TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks 

**Authors**: Vansh Kapoor, Aman Gupta, Hao Chen, Anurag Beniwal, Jing Huang, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2601.10245)  

**Abstract**: Multi-step reasoning tasks like mathematical problem solving are vulnerable to cascading failures, where a single incorrect step leads to complete solution breakdown. Current LLM routing methods assign entire queries to one model, treating all reasoning steps as equal. We propose TRIM (Targeted routing in multi-step reasoning tasks), which routes only critical steps$\unicode{x2013}$those likely to derail the solution$\unicode{x2013}$to larger models while letting smaller models handle routine continuations. Our key insight is that targeted step-level interventions can fundamentally transform inference efficiency by confining expensive calls to precisely those steps where stronger models prevent cascading errors. TRIM operates at the step-level: it uses process reward models to identify erroneous steps and makes routing decisions based on step-level uncertainty and budget constraints. We develop several routing strategies within TRIM, ranging from a simple threshold-based policy to more expressive policies that reason about long-horizon accuracy-cost trade-offs and uncertainty in step-level correctness estimates. On MATH-500, even the simplest thresholding strategy surpasses prior routing methods with 5x higher cost efficiency, while more advanced policies match the strong, expensive model's performance using 80% fewer expensive model tokens. On harder benchmarks such as AIME, TRIM achieves up to 6x higher cost efficiency. All methods generalize effectively across math reasoning tasks, demonstrating that step-level difficulty represents fundamental characteristics of reasoning. 

---
# SPRInG: Continual LLM Personalization via Selective Parametric Adaptation and Retrieval-Interpolated Generation 

**Authors**: Seoyeon Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2601.09974)  

**Abstract**: Personalizing Large Language Models typically relies on static retrieval or one-time adaptation, assuming user preferences remain invariant over time. However, real-world interactions are dynamic, where user interests continuously evolve, posing a challenge for models to adapt to preference drift without catastrophic forgetting. Standard continual learning approaches often struggle in this context, as they indiscriminately update on noisy interaction streams, failing to distinguish genuine preference shifts from transient contexts. To address this, we introduce SPRInG, a novel semi-parametric framework designed for effective continual personalization. During training, SPRInG employs drift-driven selective adaptation, which utilizes a likelihood-based scoring function to identify high-novelty interactions. This allows the model to selectively update the user-specific adapter on drift signals while preserving hard-to-learn residuals in a replay buffer. During inference, we apply strict relevance gating and fuse parametric knowledge with retrieved history via logit interpolation. Experiments on the long-form personalized generation benchmark demonstrate that SPRInG outperforms existing baselines, validating its robustness for real-world continual personalization. 

---
# ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding 

**Authors**: Xueyun Tian, Wei Li, Bingbing Xu, Heng Dong, Yuanzhuo Wang, Huawei Shen  

**Link**: [PDF](https://arxiv.org/pdf/2601.10323)  

**Abstract**: Recent Omni-multimodal Large Language Models show promise in unified audio, vision, and text modeling. However, streaming audio-video understanding remains challenging, as existing approaches suffer from disjointed capabilities: they typically exhibit incomplete modality support or lack autonomous proactive monitoring. To address this, we present ROMA, a real-time omni-multimodal assistant for unified reactive and proactive interaction. ROMA processes continuous inputs as synchronized multimodal units, aligning dense audio with discrete video frames to handle granularity mismatches. For online decision-making, we introduce a lightweight speak head that decouples response initiation from generation to ensure precise triggering without task conflict. We train ROMA with a curated streaming dataset and a two-stage curriculum that progressively optimizes for streaming format adaptation and proactive responsiveness. To standardize the fragmented evaluation landscape, we reorganize diverse benchmarks into a unified suite covering both proactive (alert, narration) and reactive (QA) settings. Extensive experiments across 12 benchmarks demonstrate ROMA achieves state-of-the-art performance on proactive tasks while competitive in reactive settings, validating its robustness in unified real-time omni-multimodal understanding. 

---
# Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts 

**Authors**: Sijia Luo, Xiaokang Zhang, Yuxuan Hu, Bohan Zhang, Ke Wang, Jinbo Su, Mengshu Sun, Lei Liang, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10079)  

**Abstract**: Reinforcement Learning (RL) has become essential for eliciting complex reasoning capabilities in Large Language Models (LLMs). However, the substantial memory overhead of storing Key-Value (KV) caches during long-horizon rollouts acts as a critical bottleneck, often prohibiting efficient training on limited hardware. While existing KV compression techniques offer a remedy for inference, directly applying them to RL training induces a severe policy mismatch, leading to catastrophic performance collapse. To address this, we introduce Sparse-RL empowers stable RL training under sparse rollouts. We show that instability arises from a fundamental policy mismatch among the dense old policy, the sparse sampler policy, and the learner policy. To mitigate this issue, Sparse-RL incorporates Sparsity-Aware Rejection Sampling and Importance-based Reweighting to correct the off-policy bias introduced by compression-induced information loss. Experimental results show that Sparse-RL reduces rollout overhead compared to dense baselines while preserving the performance. Furthermore, Sparse-RL inherently implements sparsity-aware training, significantly enhancing model robustness during sparse inference deployment. 

---
# MATRIX AS PLAN: Structured Logical Reasoning with Feedback-Driven Replanning 

**Authors**: Ke Chen, Jiandian Zeng, Zihao Peng, Guo Li, Guangxue Zhang, Tian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10101)  

**Abstract**: As knowledge and semantics on the web grow increasingly complex, enhancing Large Language Models (LLMs) comprehension and reasoning capabilities has become particularly important. Chain-of-Thought (CoT) prompting has been shown to enhance the reasoning capabilities of LLMs. However, it still falls short on logical reasoning tasks that rely on symbolic expressions and strict deductive rules. Neuro-symbolic methods address this gap by enforcing formal correctness through external solvers. Yet these solvers are highly format-sensitive, and small instabilities in model outputs can lead to frequent processing failures. LLM-driven approaches avoid parsing brittleness, but they lack structured representations and process-level error-correction mechanisms. To further enhance the logical reasoning capabilities of LLMs, we propose MatrixCoT, a structured CoT framework with a matrix-based plan. Specifically, we normalize and type natural language expressions, attach explicit citation fields, and introduce a matrix-based planning method to preserve global relations among steps. The plan becomes a verifiable artifact, making execution more stable. For verification, we also add a feedback-driven replanning mechanism. Under semantic-equivalence constraints, it identifies omissions and defects, rewrites and compresses the dependency matrix, and produces a more trustworthy final answer. Experiments on five logical-reasoning benchmarks and five LLMs show that, without relying on external solvers, MatrixCoT enhances both robustness and interpretability when tackling complex symbolic reasoning tasks, while maintaining competitive performance. 

---
# ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack 

**Authors**: Hao Li, Yankai Yang, G. Edward Suh, Ning Zhang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2601.10173)  

**Abstract**: Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at this https URL. 

---
# Self-reflection in Automated Qualitative Coding: Improving Text Annotation through Secondary LLM Critique 

**Authors**: Zackary Okun Dunivin, Mobina Noori, Seth Frey, Curtis Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2601.09905)  

**Abstract**: Large language models (LLMs) allow for sophisticated qualitative coding of large datasets, but zero- and few-shot classifiers can produce an intolerable number of errors, even with careful, validated prompting. We present a simple, generalizable two-stage workflow: an LLM applies a human-designed, LLM-adapted codebook; a secondary LLM critic performs self-reflection on each positive label by re-reading the source text alongside the first model's rationale and issuing a final decision. We evaluate this approach on six qualitative codes over 3,000 high-content emails from Apache Software Foundation project evaluation discussions. Our human-derived audit of 360 positive annotations (60 passages by six codes) found that the first-line LLM had a false-positive rate of 8% to 54%, despite F1 scores of 0.74 and 1.00 in testing. Subsequent recoding of all stage-one annotations via a second self-reflection stage improved F1 by 0.04 to 0.25, bringing two especially poor performing codes up to 0.69 and 0.79 from 0.52 and 0.55 respectively. Our manual evaluation identified two recurrent error classes: misinterpretation (violations of code definitions) and meta-discussion (debate about a project evaluation criterion mistaken for its use as a decision justification). Code-specific critic clauses addressing observed failure modes were especially effective with testing and refinement, replicating the codebook-adaption process for LLM interpretation in stage-one. We explain how favoring recall in first-line LLM annotation combined with secondary critique delivers precision-first, compute-light control. With human guidance and validation, self-reflection slots into existing LLM-assisted annotation pipelines to reduce noise and potentially salvage unusable classifiers. 

---
# PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary 

**Authors**: Jiarui Yao, Ruida Wang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.10201)  

**Abstract**: Improving the reasoning abilities of Large Language Models (LLMs) has been a continuous topic recently. But most relevant works are based on outcome rewards at the trajectory level, missing fine-grained supervision during the reasoning process. Other existing training frameworks that try to combine process signals together to optimize LLMs also rely heavily on tedious additional steps like MCTS, training a separate reward model, etc., doing harm to the training efficiency. Moreover, the intuition behind the process signals design lacks rigorous theoretical support, leaving the understanding of the optimization mechanism opaque. In this paper, we propose Process Reward Learning (PRL), which decomposes the entropy regularized reinforcement learning objective into intermediate steps, with rigorous process rewards that could be assigned to models accordingly. Starting from theoretical motivation, we derive the formulation of PRL that is essentially equivalent to the objective of reward maximization plus a KL-divergence penalty term between the policy model and a reference model. However, PRL could turn the outcome reward into process supervision signals, which helps better guide the exploration during RL optimization. From our experiment results, we demonstrate that PRL not only improves the average performance for LLMs' reasoning ability measured by average @ n, but also broadens the reasoning boundary by improving the pass @ n metric. Extensive experiments show the effectiveness of PRL could be verified and generalized. 

---
# Generative AI collective behavior needs an interactionist paradigm 

**Authors**: Laura Ferrarotti, Gian Maria Campedelli, Roberto Dessì, Andrea Baronchelli, Giovanni Iacca, Kathleen M. Carley, Alex Pentland, Joel Z. Leibo, James Evans, Bruno Lepri  

**Link**: [PDF](https://arxiv.org/pdf/2601.10567)  

**Abstract**: In this article, we argue that understanding the collective behavior of agents based on large language models (LLMs) is an essential area of inquiry, with important implications in terms of risks and benefits, impacting us as a society at many levels. We claim that the distinctive nature of LLMs--namely, their initialization with extensive pre-trained knowledge and implicit social priors, together with their capability of adaptation through in-context learning--motivates the need for an interactionist paradigm consisting of alternative theoretical foundations, methodologies, and analytical tools, in order to systematically examine how prior knowledge and embedded values interact with social context to shape emergent phenomena in multi-agent generative AI systems. We propose and discuss four directions that we consider crucial for the development and deployment of LLM-based collectives, focusing on theory, methods, and trans-disciplinary dialogue. 

---
# Diagnosing Generalization Failures in Fine-Tuned LLMs: A Cross-Architectural Study on Phishing Detection 

**Authors**: Frank Bobe III, Gregory D. Vetaw, Chase Pavlick, Darshan Bryner, Matthew Cook, Jose Salas-Vernis  

**Link**: [PDF](https://arxiv.org/pdf/2601.10524)  

**Abstract**: The practice of fine-tuning Large Language Models (LLMs) has achieved state-of-the-art performance on specialized tasks, yet diagnosing why these models become brittle and fail to generalize remains a critical open problem. To address this, we introduce and apply a multi-layered diagnostic framework to a cross-architectural study. We fine-tune Llama 3.1 8B, Gemma 2 9B, and Mistral models on a high-stakes phishing detection task and use SHAP analysis and mechanistic interpretability to uncover the root causes of their generalization failures. Our investigation reveals three critical findings: (1) Generalization is driven by a powerful synergy between architecture and data diversity. The Gemma 2 9B model achieves state-of-the-art performance (>91\% F1), but only when trained on a stylistically diverse ``generalist'' dataset. (2) Generalization is highly architecture-dependent. We diagnose a specific failure mode in Llama 3.1 8B, which performs well on a narrow domain but cannot integrate diverse data, leading to a significant performance drop. (3) Some architectures are inherently more generalizable. The Mistral model proves to be a consistent and resilient performer across multiple training paradigms. By pinpointing the flawed heuristics responsible for these failures, our work provides a concrete methodology for diagnosing and understanding generalization failures, underscoring that reliable AI requires deep validation of the interplay between architecture, data, and training strategy. 

---
# Structure and Diversity Aware Context Bubble Construction for Enterprise Retrieval Augmented Systems 

**Authors**: Amir Khurshid, Abhishek Sehgal  

**Link**: [PDF](https://arxiv.org/pdf/2601.10681)  

**Abstract**: Large language model (LLM) contexts are typically constructed using retrieval-augmented generation (RAG), which involves ranking and selecting the top-k passages. The approach causes fragmentation in information graphs in document structures, over-retrieval, and duplication of content alongside insufficient query context, including 2nd and 3rd order facets. In this paper, a structure-informed and diversity-constrained context bubble construction framework is proposed that assembles coherent, citable bundles of spans under a strict token budget. The method preserves and exploits inherent document structure by organising multi-granular spans (e.g., sections and rows) and using task-conditioned structural priors to guide retrieval. Starting from high-relevance anchor spans, a context bubble is constructed through constrained selection that balances query relevance, marginal coverage, and redundancy penalties. It will explicitly constrain diversity and budget, producing compact and informative context sets, unlike top-k retrieval. Moreover, a full retrieval is emitted that traces the scoring and selection choices of the records, thus providing auditability and deterministic tuning. Experiments on enterprise documents demonstrate the efficiency of context bubble as it significantly reduces redundant context, is better able to cover secondary facets and has a better answer quality and citation faithfulness within a limited context window. Ablation studies demonstrate that both structural priors as well as diversity constraint selection are necessary; removing either component results in a decline in coverage and an increase in redundant or incomplete context. 

---
# LLMdoctor: Token-Level Flow-Guided Preference Optimization for Efficient Test-Time Alignment of Large Language Models 

**Authors**: Tiesunlong Shen, Rui Mao, Jin Wang, Heming Sun, Jian Zhang, Xuejie Zhang, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2601.10416)  

**Abstract**: Aligning Large Language Models (LLMs) with human preferences is critical, yet traditional fine-tuning methods are computationally expensive and inflexible. While test-time alignment offers a promising alternative, existing approaches often rely on distorted trajectory-level signals or inefficient sampling, fundamentally capping performance and failing to preserve the generative diversity of the base model. This paper introduces LLMdoctor, a novel framework for efficient test-time alignment that operates via a patient-doctor paradigm. It integrates token-level reward acquisition with token-level flow-guided preference optimization (TFPO) to steer a large, frozen patient LLM with a smaller, specialized doctor model. Unlike conventional methods that rely on trajectory-level rewards, LLMdoctor first extracts fine-grained, token-level preference signals from the patient model's behavioral variations. These signals then guide the training of the doctor model via TFPO, which establishes flow consistency across all subtrajectories, enabling precise token-by-token alignment while inherently preserving generation diversity. Extensive experiments demonstrate that LLMdoctor significantly outperforms existing test-time alignment methods and even surpasses the performance of full fine-tuning approaches like DPO. 

---
# Breaking Up with Normatively Monolithic Agency with GRACE: A Reason-Based Neuro-Symbolic Architecture for Safe and Ethical AI Alignment 

**Authors**: Felix Jahn, Yannic Muskalla, Lisa Dargasz, Patrick Schramowski, Kevin Baum  

**Link**: [PDF](https://arxiv.org/pdf/2601.10520)  

**Abstract**: As AI agents become increasingly autonomous, widely deployed in consequential contexts, and efficacious in bringing about real-world impacts, ensuring that their decisions are not only instrumentally effective but also normatively aligned has become critical. We introduce a neuro-symbolic reason-based containment architecture, Governor for Reason-Aligned ContainmEnt (GRACE), that decouples normative reasoning from instrumental decision-making and can contain AI agents of virtually any design. GRACE restructures decision-making into three modules: a Moral Module (MM) that determines permissible macro actions via deontic logic-based reasoning; a Decision-Making Module (DMM) that encapsulates the target agent while selecting instrumentally optimal primitive actions in accordance with derived macro actions; and a Guard that monitors and enforces moral compliance. The MM uses a reason-based formalism providing a semantic foundation for deontic logic, enabling interpretability, contestability, and justifiability. Its symbolic representation enriches the DMM's informational context and supports formal verification and statistical guarantees of alignment enforced by the Guard. We demonstrate GRACE on an example of a LLM therapy assistant, showing how it enables stakeholders to understand, contest, and refine agent behavior. 

---
# NSR-Boost: A Neuro-Symbolic Residual Boosting Framework for Industrial Legacy Models 

**Authors**: Ziming Dai, Dabiao Ma, Jinle Tong, Mengyuan Han, Jian Yang, Haojun Fei  

**Link**: [PDF](https://arxiv.org/pdf/2601.10457)  

**Abstract**: Although the Gradient Boosted Decision Trees (GBDTs) dominate industrial tabular applications, upgrading legacy models in high-concurrency production environments still faces prohibitive retraining costs and systemic risks. To address this problem, we present NSR-Boost, a neuro-symbolic residual boosting framework designed specifically for industrial scenarios. Its core advantage lies in being "non-intrusive". It treats the legacy model as a frozen model and performs targeted repairs on "hard regions" where predictions fail. The framework comprises three key stages: first, finding hard regions through residuals, then generating interpretable experts by generating symbolic code structures using Large Language Model (LLM) and fine-tuning parameters using Bayesian optimization, and finally dynamically integrating experts with legacy model output through a lightweight aggregator. We report on the successful deployment of NSR-Boost within the core financial risk control system at Qfin Holdings. This framework not only significantly outperforms state-of-the-art (SOTA) baselines across six public datasets and one private dataset, more importantly, shows excellent performance gains on real-world online data. In conclusion, it effectively captures long-tail risks missed by traditional models and offers a safe, low-cost evolutionary paradigm for industry. 

---
# LADFA: A Framework of Using Large Language Models and Retrieval-Augmented Generation for Personal Data Flow Analysis in Privacy Policies 

**Authors**: Haiyue Yuan, Nikolay Matyunin, Ali Raza, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.10413)  

**Abstract**: Privacy policies help inform people about organisations' personal data processing practices, covering different aspects such as data collection, data storage, and sharing of personal data with third parties. Privacy policies are often difficult for people to fully comprehend due to the lengthy and complex legal language used and inconsistent practices across different sectors and organisations. To help conduct automated and large-scale analyses of privacy policies, many researchers have studied applications of machine learning and natural language processing techniques, including large language models (LLMs). While a limited number of prior studies utilised LLMs for extracting personal data flows from privacy policies, our approach builds on this line of work by combining LLMs with retrieval-augmented generation (RAG) and a customised knowledge base derived from existing studies. This paper presents the development of LADFA, an end-to-end computational framework, which can process unstructured text in a given privacy policy, extract personal data flows and construct a personal data flow graph, and conduct analysis of the data flow graph to facilitate insight discovery. The framework consists of a pre-processor, an LLM-based processor, and a data flow post-processor. We demonstrated and validated the effectiveness and accuracy of the proposed approach by conducting a case study that involved examining ten selected privacy policies from the automotive industry. Moreover, it is worth noting that LADFA is designed to be flexible and customisable, making it suitable for a range of text-based analysis tasks beyond privacy policy analysis. 

---
# LatentRefusal: Latent-Signal Refusal for Unanswerable Text-to-SQL Queries 

**Authors**: Xuancheng Ren, Shijing Hu, Zhihui Lu, Jiangqi Huang, Qiang Duan  

**Link**: [PDF](https://arxiv.org/pdf/2601.10398)  

**Abstract**: In LLM-based text-to-SQL systems, unanswerable and underspecified user queries may generate not only incorrect text but also executable programs that yield misleading results or violate safety constraints, posing a major barrier to safe deployment. Existing refusal strategies for such queries either rely on output-level instruction following, which is brittle due to model hallucinations, or estimate output uncertainty, which adds complexity and overhead. To address this challenge, we formalize safe refusal in text-to-SQL systems as an answerability-gating problem and propose LatentRefusal, a latent-signal refusal mechanism that predicts query answerability from intermediate hidden activations of a large language model. We introduce the Tri-Residual Gated Encoder, a lightweight probing architecture, to suppress schema noise and amplify sparse, localized cues of question-schema mismatch that indicate unanswerability. Extensive empirical evaluations across diverse ambiguous and unanswerable settings, together with ablation studies and interpretability analyses, demonstrate the effectiveness of the proposed approach and show that LatentRefusal provides an attachable and efficient safety layer for text-to-SQL systems. Across four benchmarks, LatentRefusal improves average F1 to 88.5 percent on both backbones while adding approximately 2 milliseconds of probe overhead. 

---
# DecisionLLM: Large Language Models for Long Sequence Decision Exploration 

**Authors**: Xiaowei Lv, Zhilin Zhang, Yijun Li, Yusen Huo, Siyuan Ju, Xuyan Li, Chunxiang Hong, Tianyu Wang, Yongcai Wang, Peng Sun, Chuan Yu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2601.10148)  

**Abstract**: Long-sequence decision-making, which is usually addressed through reinforcement learning (RL), is a critical component for optimizing strategic operations in dynamic environments, such as real-time bidding in computational advertising. The Decision Transformer (DT) introduced a powerful paradigm by framing RL as an autoregressive sequence modeling problem. Concurrently, Large Language Models (LLMs) have demonstrated remarkable success in complex reasoning and planning tasks. This inspires us whether LLMs, which share the same Transformer foundation, but operate at a much larger scale, can unlock new levels of performance in long-horizon sequential decision-making problem. This work investigates the application of LLMs to offline decision making tasks. A fundamental challenge in this domain is the LLMs' inherent inability to interpret continuous values, as they lack a native understanding of numerical magnitude and order when values are represented as text strings. To address this, we propose treating trajectories as a distinct modality. By learning to align trajectory data with natural language task descriptions, our model can autoregressively predict future decisions within a cohesive framework we term DecisionLLM. We establish a set of scaling laws governing this paradigm, demonstrating that performance hinges on three factors: model scale, data volume, and data quality. In offline experimental benchmarks and bidding scenarios, DecisionLLM achieves strong performance. Specifically, DecisionLLM-3B outperforms the traditional Decision Transformer (DT) by 69.4 on Maze2D umaze-v1 and by 0.085 on AuctionNet. It extends the AIGB paradigm and points to promising directions for future exploration in online bidding. 

---
# Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction 

**Authors**: Yanan Cao, Farnaz Fallahi, Murali Mohana Krishna Dandu, Lalitesh Morishetti, Kai Zhao, Luyi Ma, Sinduja Subramaniam, Jianpeng Xu, Evren Korpeoglu, Kaushiki Nag, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2601.10132)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning and prediction across different domains. Yet, their ability to infer temporal regularities from structured behavioral data remains underexplored. This paper presents a systematic study investigating whether LLMs can predict time intervals between recurring user actions, such as repeated purchases, and how different levels of contextual information shape their predictive behavior. Using a simple but representative repurchase scenario, we benchmark state-of-the-art LLMs in zero-shot settings against both statistical and machine-learning models. Two key findings emerge. First, while LLMs surpass lightweight statistical baselines, they consistently underperform dedicated machine-learning models, showing their limited ability to capture quantitative temporal structure. Second, although moderate context can improve LLM accuracy, adding further user-level detail degrades performance. These results challenge the assumption that "more context leads to better reasoning". Our study highlights fundamental limitations of today's LLMs in structured temporal inference and offers guidance for designing future context-aware hybrid models that integrate statistical precision with linguistic flexibility. 

---
# State of AI: An Empirical 100 Trillion Token Study with OpenRouter 

**Authors**: Malika Aubakirova, Alex Atallah, Chris Clark, Justin Summerville, Anjney Midha  

**Link**: [PDF](https://arxiv.org/pdf/2601.10088)  

**Abstract**: The past year has marked a turning point in the evolution and real-world use of large language models (LLMs). With the release of the first widely adopted reasoning model, o1, on December 5th, 2024, the field shifted from single-pass pattern generation to multi-step deliberation inference, accelerating deployment, experimentation, and new classes of applications. As this shift unfolded at a rapid pace, our empirical understanding of how these models have actually been used in practice has lagged behind. In this work, we leverage the OpenRouter platform, which is an AI inference provider across a wide variety of LLMs, to analyze over 100 trillion tokens of real-world LLM interactions across tasks, geographies, and time. In our empirical study, we observe substantial adoption of open-weight models, the outsized popularity of creative roleplay (beyond just the productivity tasks many assume dominate) and coding assistance categories, plus the rise of agentic inference. Furthermore, our retention analysis identifies foundational cohorts: early users whose engagement persists far longer than later cohorts. We term this phenomenon the Cinderella "Glass Slipper" effect. These findings underscore that the way developers and end-users engage with LLMs "in the wild" is complex and multifaceted. We discuss implications for model builders, AI developers, and infrastructure providers, and outline how a data-driven understanding of usage can inform better design and deployment of LLM systems. 

---
# Hallucination Detection and Mitigation in Large Language Models 

**Authors**: Ahmad Pesaranghader, Erin Li  

**Link**: [PDF](https://arxiv.org/pdf/2601.09929)  

**Abstract**: Large Language Models (LLMs) and Large Reasoning Models (LRMs) offer transformative potential for high-stakes domains like finance and law, but their tendency to hallucinate, generating factually incorrect or unsupported content, poses a critical reliability risk. This paper introduces a comprehensive operational framework for hallucination management, built on a continuous improvement cycle driven by root cause awareness. We categorize hallucination sources into model, data, and context-related factors, allowing targeted interventions over generic fixes. The framework integrates multi-faceted detection methods (e.g., uncertainty estimation, reasoning consistency) with stratified mitigation strategies (e.g., knowledge grounding, confidence calibration). We demonstrate its application through a tiered architecture and a financial data extraction case study, where model, context, and data tiers form a closed feedback loop for progressive reliability enhancement. This approach provides a systematic, scalable methodology for building trustworthy generative AI systems in regulated environments. 

---
# Improving Chain-of-Thought for Logical Reasoning via Attention-Aware Intervention 

**Authors**: Nguyen Minh Phuong, Dang Huu Tien, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2601.09805)  

**Abstract**: Modern logical reasoning with LLMs primarily relies on employing complex interactive frameworks that decompose the reasoning process into subtasks solved through carefully designed prompts or requiring external resources (e.g., symbolic solvers) to exploit their strong logical structures. While interactive approaches introduce additional overhead, hybrid approaches depend on external components, which limit their scalability. A non-interactive, end-to-end framework enables reasoning to emerge within the model itself -- improving generalization while preserving analyzability without any external resources. In this work, we introduce a non-interactive, end-to-end framework for reasoning tasks. We show that introducing structural information into the few-shot prompt activates a subset of attention heads that patterns aligned with logical reasoning operators. Building on this insight, we propose Attention-Aware Intervention (AAI), an inference-time intervention method that reweights attention scores across selected heads identified by their logical patterns. AAI offers an efficient way to steer the model's reasoning toward leveraging prior knowledge through attention modulation. Extensive experiments show that AAI enhances logical reasoning performance across diverse benchmarks and model architectures, while incurring negligible additional computational overhead. Code is available at this https URL. 

---
# PCN-Rec: Agentic Proof-Carrying Negotiation for Reliable Governance-Constrained Recommendation 

**Authors**: Aradhya Dixit, Shreem Dixit  

**Link**: [PDF](https://arxiv.org/pdf/2601.09771)  

**Abstract**: Modern LLM-based recommenders can generate compelling ranked lists, but they struggle to reliably satisfy governance constraints such as minimum long-tail exposure or diversity requirements. We present PCN-Rec, a proof-carrying negotiation pipeline that separates natural-language reasoning from deterministic enforcement. A base recommender (MF/CF) produces a candidate window of size W, which is negotiated by two agents: a User Advocate optimizing relevance and a Policy Agent enforcing constraints. A mediator LLM synthesizes a top-N slate together with a structured certificate (JSON) describing the claimed constraint satisfaction. A deterministic verifier recomputes all constraints from the slate and accepts only verifier-checked certificates; if verification fails, a deterministic constrained-greedy repair produces a compliant slate for re-verification, yielding an auditable trace. On MovieLens-100K with governance constraints, PCN-Rec achieves a 98.55% pass rate on feasible users (n = 551, W = 80) versus a one-shot single-LLM baseline without verification/repair, while preserving utility with only a 0.021 absolute drop in NDCG@10 (0.403 vs. 0.424); differences are statistically significant (p < 0.05). 

---
# Model See, Model Do? Exposure-Aware Evaluation of Bug-vs-Fix Preference in Code LLMs 

**Authors**: Ali Al-Kaswan, Claudio Spiess, Prem Devanbu, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2601.10496)  

**Abstract**: Large language models are increasingly used for code generation and debugging, but their outputs can still contain bugs, that originate from training data. Distinguishing whether an LLM prefers correct code, or a familiar incorrect version might be influenced by what it's been exposed to during training. We introduce an exposure-aware evaluation framework that quantifies how prior exposure to buggy versus fixed code influences a model's preference. Using the ManySStuBs4J benchmark, we apply Data Portraits for membership testing on the Stack-V2 corpus to estimate whether each buggy and fixed variant was seen during training. We then stratify examples by exposure and compare model preference using code completion as well as multiple likelihood-based scoring metrics We find that most examples (67%) have neither variant in the training data, and when only one is present, fixes are more frequently present than bugs. In model generations, models reproduce buggy lines far more often than fixes, with bug-exposed examples amplifying this tendency and fix-exposed examples showing only marginal improvement. In likelihood scoring, minimum and maximum token-probability metrics consistently prefer the fixed code across all conditions, indicating a stable bias toward correct fixes. In contrast, metrics like the Gini coefficient reverse preference when only the buggy variant was seen. Our results indicate that exposure can skew bug-fix evaluations and highlight the risk that LLMs may propagate memorised errors in practice. 

---
# Untangling Input Language from Reasoning Language: A Diagnostic Framework for Cross-Lingual Moral Alignment in LLMs 

**Authors**: Nan Li, Bo Kang, Tijl De Bie  

**Link**: [PDF](https://arxiv.org/pdf/2601.10257)  

**Abstract**: When LLMs judge moral dilemmas, do they reach different conclusions in different languages, and if so, why? Two factors could drive such differences: the language of the dilemma itself, or the language in which the model reasons. Standard evaluation conflates these by testing only matched conditions (e.g., English dilemma with English reasoning). We introduce a methodology that separately manipulates each factor, covering also mismatched conditions (e.g., English dilemma with Chinese reasoning), enabling decomposition of their contributions. To study \emph{what} changes, we propose an approach to interpret the moral judgments in terms of Moral Foundations Theory. As a side result, we identify evidence for splitting the Authority dimension into a family-related and an institutional dimension. Applying this methodology to English-Chinese moral judgment with 13 LLMs, we demonstrate its diagnostic power: (1) the framework isolates reasoning-language effects as contributing twice the variance of input-language effects; (2) it detects context-dependency in nearly half of models that standard evaluation misses; and (3) a diagnostic taxonomy translates these patterns into deployment guidance. We release our code and datasets at this https URL. 

---
# VERHallu: Evaluating and Mitigating Event Relation Hallucination in Video Large Language Models 

**Authors**: Zefan Zhang, Kehua Zhu, Shijie Jiang, Hongyuan Lu, Shengkai Sun, Tian Bai  

**Link**: [PDF](https://arxiv.org/pdf/2601.10010)  

**Abstract**: Video Large Language Models (VideoLLMs) exhibit various types of hallucinations. Existing research has primarily focused on hallucinations involving the presence of events, objects, and scenes in videos, while largely neglecting event relation hallucination. In this paper, we introduce a novel benchmark for evaluating the Video Event Relation Hallucination, named VERHallu. This benchmark focuses on causal, temporal, and subevent relations between events, encompassing three types of tasks: relation classification, question answering, and counterfactual question answering, for a comprehensive evaluation of event relation hallucination. Additionally, it features counterintuitive video scenarios that deviate from typical pretraining distributions, with each sample accompanied by human-annotated candidates covering both vision-language and pure language biases. Our analysis reveals that current state-of-the-art VideoLLMs struggle with dense-event relation reasoning, often relying on prior knowledge due to insufficient use of frame-level cues. Although these models demonstrate strong grounding capabilities for key events, they often overlook the surrounding subevents, leading to an incomplete and inaccurate understanding of event relations. To tackle this, we propose a Key-Frame Propagating (KFP) strategy, which reallocates frame-level attention within intermediate layers to enhance multi-event understanding. Experiments show it effectively mitigates the event relation hallucination without affecting inference speed. 

---
# Context Volume Drives Performance: Tackling Domain Shift in Extremely Low-Resource Translation via RAG 

**Authors**: David Samuel Setiawan, Raphaël Merx, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2601.09982)  

**Abstract**: Neural Machine Translation (NMT) models for low-resource languages suffer significant performance degradation under domain shift. We quantify this challenge using Dhao, an indigenous language of Eastern Indonesia with no digital footprint beyond the New Testament (NT). When applied to the unseen Old Testament (OT), a standard NMT model fine-tuned on the NT drops from an in-domain score of 36.17 chrF++ to 27.11 chrF++. To recover this loss, we introduce a hybrid framework where a fine-tuned NMT model generates an initial draft, which is then refined by a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG). The final system achieves 35.21 chrF++ (+8.10 recovery), effectively matching the original in-domain quality. Our analysis reveals that this performance is driven primarily by the number of retrieved examples rather than the choice of retrieval algorithm. Qualitative analysis confirms the LLM acts as a robust "safety net," repairing severe failures in zero-shot domains. 

---
# Performance of AI agents based on reasoning language models on ALD process optimization tasks 

**Authors**: Angel Yanguas-Gil  

**Link**: [PDF](https://arxiv.org/pdf/2601.09980)  

**Abstract**: In this work we explore the performance and behavior of reasoning large language models to autonomously optimize atomic layer deposition (ALD) processes. In the ALD process optimization task, an agent built on top of a reasoning LLM has to find optimal dose times for an ALD precursor and a coreactant without any prior knowledge on the process, including whether it is actually self-limited. The agent is meant to interact iteratively with an ALD reactor in a fully unsupervised way. We evaluate this agent using a simple model of an ALD tool that incorporates ALD processes with different self-limited surface reaction pathways as well as a non self-limited component. Our results show that agents based on reasoning models like OpenAI's o3 and GPT5 consistently succeeded at completing this optimization task. However, we observed significant run-to-run variability due to the non deterministic nature of the model's response. In order to understand the logic followed by the reasoning model, the agent uses a two step process in which the model first generates an open response detailing the reasoning process. This response is then transformed into a structured output. An analysis of these reasoning traces showed that the logic of the model was sound and that its reasoning was based on the notions of self-limited process and saturation expected in the case of ALD. However, the agent can sometimes be misled by its own prior choices when exploring the optimization space. 

---
# Investigating Tool-Memory Conflicts in Tool-Augmented LLMs 

**Authors**: Jiali Cheng, Rui Pan, Hadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2601.09760)  

**Abstract**: Tool-augmented large language models (LLMs) have powered many applications. However, they are likely to suffer from knowledge conflict. In this paper, we propose a new type of knowledge conflict -- Tool-Memory Conflict (TMC), where the internal parametric knowledge contradicts with the external tool knowledge for tool-augmented LLMs. We find that existing LLMs, though powerful, suffer from TMC, especially on STEM-related tasks. We also uncover that under different conditions, tool knowledge and parametric knowledge may be prioritized differently. We then evaluate existing conflict resolving techniques, including prompting-based and RAG-based methods. Results show that none of these approaches can effectively resolve tool-memory conflicts. 

---
# Explicating Tacit Regulatory Knowledge from LLMs to Auto-Formalize Requirements for Compliance Test Case Generation 

**Authors**: Zhiyi Xue, Xiaohong Chen, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.09762)  

**Abstract**: Compliance testing in highly regulated domains is crucial but largely manual, requiring domain experts to translate complex regulations into executable test cases. While large language models (LLMs) show promise for automation, their susceptibility to hallucinations limits reliable application. Existing hybrid approaches mitigate this issue by constraining LLMs with formal models, but still rely on costly manual modeling. To solve this problem, this paper proposes RAFT, a framework for requirements auto-formalization and compliance test generation via explicating tacit regulatory knowledge from multiple LLMs. RAFT employs an Adaptive Purification-Aggregation strategy to explicate tacit regulatory knowledge from multiple LLMs and integrate it into three artifacts: a domain meta-model, a formal requirements representation, and testability constraints. These artifacts are then dynamically injected into prompts to guide high-precision requirement formalization and automated test generation. Experiments across financial, automotive, and power domains show that RAFT achieves expert-level performance, substantially outperforms state-of-the-art (SOTA) methods while reducing overall generation and review time. 

---
