# PlanQA: A Benchmark for Spatial Reasoning in LLMs using Structured Representations 

**Authors**: Fedor Rodionov, Abdelrahman Eldesokey, Michael Birsak, John Femiani, Bernard Ghanem, Peter Wonka  

**Link**: [PDF](https://arxiv.org/pdf/2507.07644)  

**Abstract**: We introduce PlanQA, a diagnostic benchmark for evaluating geometric and spatial reasoning in large-language models (LLMs). PlanQA is grounded in structured representations of indoor scenes, such as kitchens, living rooms, and bedrooms, encoded in a symbolic format (e.g., JSON, XML layouts). The benchmark includes diverse question types that test not only metric and topological reasoning (e.g., distance, visibility, shortest paths) but also interior design constraints such as affordance, clearance, balance, and usability. Our results across a variety of frontier open-source and commercial LLMs show that while models may succeed in shallow queries, they often fail to simulate physical constraints, preserve spatial coherence, or generalize under layout perturbation. PlanQA uncovers a clear blind spot in today's LLMs: they do not consistently reason about real-world layouts. We hope that this benchmark inspires new work on language models that can accurately infer and manipulate spatial and geometric properties in practical settings. 

---
# Measuring AI Alignment with Human Flourishing 

**Authors**: Elizabeth Hilliard, Akshaya Jagadeesh, Alex Cook, Steele Billings, Nicholas Skytland, Alicia Llewellyn, Jackson Paull, Nathan Paull, Nolan Kurylo, Keatra Nesbitt, Robert Gruenewald, Anthony Jantzi, Omar Chavez  

**Link**: [PDF](https://arxiv.org/pdf/2507.07787)  

**Abstract**: This paper introduces the Flourishing AI Benchmark (FAI Benchmark), a novel evaluation framework that assesses AI alignment with human flourishing across seven dimensions: Character and Virtue, Close Social Relationships, Happiness and Life Satisfaction, Meaning and Purpose, Mental and Physical Health, Financial and Material Stability, and Faith and Spirituality. Unlike traditional benchmarks that focus on technical capabilities or harm prevention, the FAI Benchmark measures AI performance on how effectively models contribute to the flourishing of a person across these dimensions. The benchmark evaluates how effectively LLM AI systems align with current research models of holistic human well-being through a comprehensive methodology that incorporates 1,229 objective and subjective questions. Using specialized judge Large Language Models (LLMs) and cross-dimensional evaluation, the FAI Benchmark employs geometric mean scoring to ensure balanced performance across all flourishing dimensions. Initial testing of 28 leading language models reveals that while some models approach holistic alignment (with the highest-scoring models achieving 72/100), none are acceptably aligned across all dimensions, particularly in Faith and Spirituality, Character and Virtue, and Meaning and Purpose. This research establishes a framework for developing AI systems that actively support human flourishing rather than merely avoiding harm, offering significant implications for AI development, ethics, and evaluation. 

---
# On the Impossibility of Separating Intelligence from Judgment: The Computational Intractability of Filtering for AI Alignment 

**Authors**: Sarah Ball, Greg Gluch, Shafi Goldwasser, Frauke Kreuter, Omer Reingold, Guy N. Rothblum  

**Link**: [PDF](https://arxiv.org/pdf/2507.07341)  

**Abstract**: With the increased deployment of large language models (LLMs), one concern is their potential misuse for generating harmful content. Our work studies the alignment challenge, with a focus on filters to prevent the generation of unsafe information. Two natural points of intervention are the filtering of the input prompt before it reaches the model, and filtering the output after generation. Our main results demonstrate computational challenges in filtering both prompts and outputs. First, we show that there exist LLMs for which there are no efficient prompt filters: adversarial prompts that elicit harmful behavior can be easily constructed, which are computationally indistinguishable from benign prompts for any efficient filter. Our second main result identifies a natural setting in which output filtering is computationally intractable. All of our separation results are under cryptographic hardness assumptions. In addition to these core findings, we also formalize and study relaxed mitigation approaches, demonstrating further computational barriers. We conclude that safety cannot be achieved by designing filters external to the LLM internals (architecture and weights); in particular, black-box access to the LLM will not suffice. Based on our technical results, we argue that an aligned AI system's intelligence cannot be separated from its judgment. 

---
# Application of LLMs to Multi-Robot Path Planning and Task Allocation 

**Authors**: Ashish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.07302)  

**Abstract**: Efficient exploration is a well known problem in deep reinforcement learning and this problem is exacerbated in multi-agent reinforcement learning due the intrinsic complexities of such algorithms. There are several approaches to efficiently explore an environment to learn to solve tasks by multi-agent operating in that environment, of which, the idea of expert exploration is investigated in this work. More specifically, this work investigates the application of large-language models as expert planners for efficient exploration in planning based tasks for multiple agents. 

---
# Open Source Planning & Control System with Language Agents for Autonomous Scientific Discovery 

**Authors**: Licong Xu, Milind Sarkar, Anto I. Lonappan, Íñigo Zubeldia, Pablo Villanueva-Domingo, Santiago Casas, Christian Fidler, Chetana Amancharla, Ujjwal Tiwari, Adrian Bayer, Chadi Ait Ekiou, Miles Cranmer, Adrian Dimitrov, James Fergusson, Kahaan Gandhi, Sven Krippendorf, Andrew Laverick, Julien Lesgourgues, Antony Lewis, Thomas Meier, Blake Sherwin, Kristen Surrao, Francisco Villaescusa-Navarro, Chi Wang, Xueqing Xu, Boris Bolliet  

**Link**: [PDF](https://arxiv.org/pdf/2507.07257)  

**Abstract**: We present a multi-agent system for automation of scientific research tasks, cmbagent. The system is formed by about 30 Large Language Model (LLM) agents and implements a Planning & Control strategy to orchestrate the agentic workflow, with no human-in-the-loop at any point. Each agent specializes in a different task (performing retrieval on scientific papers and codebases, writing code, interpreting results, critiquing the output of other agents) and the system is able to execute code locally. We successfully apply cmbagent to carry out a PhD level cosmology task (the measurement of cosmological parameters using supernova data) and evaluate its performance on two benchmark sets, finding superior performance over state-of-the-art LLMs. The source code is available on GitHub, demonstration videos are also available, and the system is deployed on HuggingFace and will be available on the cloud. 

---
# ViDove: A Translation Agent System with Multimodal Context and Memory-Augmented Reasoning 

**Authors**: Yichen Lu, Wei Dai, Jiaen Liu, Ching Wing Kwok, Zongheng Wu, Xudong Xiao, Ao Sun, Sheng Fu, Jianyuan Zhan, Yian Wang, Takatomo Saito, Sicheng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2507.07306)  

**Abstract**: LLM-based translation agents have achieved highly human-like translation results and are capable of handling longer and more complex contexts with greater efficiency. However, they are typically limited to text-only inputs. In this paper, we introduce ViDove, a translation agent system designed for multimodal input. Inspired by the workflow of human translators, ViDove leverages visual and contextual background information to enhance the translation process. Additionally, we integrate a multimodal memory system and long-short term memory modules enriched with domain-specific knowledge, enabling the agent to perform more accurately and adaptively in real-world scenarios. As a result, ViDove achieves significantly higher translation quality in both subtitle generation and general translation tasks, with a 28% improvement in BLEU scores and a 15% improvement in SubER compared to previous state-of-the-art baselines. Moreover, we introduce DoveBench, a new benchmark for long-form automatic video subtitling and translation, featuring 17 hours of high-quality, human-annotated data. Our code is available here: this https URL 

---
# Neurosymbolic Feature Extraction for Identifying Forced Labor in Supply Chains 

**Authors**: Zili Wang, Frank Montabon, Kristin Yvonne Rozier  

**Link**: [PDF](https://arxiv.org/pdf/2507.07217)  

**Abstract**: Supply chain networks are complex systems that are challenging to analyze; this problem is exacerbated when there are illicit activities involved in the supply chain, such as counterfeit parts, forced labor, or human trafficking. While machine learning (ML) can find patterns in complex systems like supply chains, traditional ML techniques require large training data sets. However, illicit supply chains are characterized by very sparse data, and the data that is available is often (purposely) corrupted or unreliable in order to hide the nature of the activities. We need to be able to automatically detect new patterns that correlate with such illegal activity over complex, even temporal data, without requiring large training data sets. We explore neurosymbolic methods for identifying instances of illicit activity in supply chains and compare the effectiveness of manual and automated feature extraction from news articles accurately describing illicit activities uncovered by authorities. We propose a question tree approach for querying a large language model (LLM) to identify and quantify the relevance of articles. This enables a systematic evaluation of the differences between human and machine classification of news articles related to forced labor in supply chains. 

---
# DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search 

**Authors**: Zerui Yang, Yuwei Wan, Yinqiao Li, Yudai Matsuda, Tong Xie, Linqi Song  

**Link**: [PDF](https://arxiv.org/pdf/2507.07426)  

**Abstract**: Recent advances in large language models have demonstrated considerable potential in scientific domains such as drug discovery. However, their effectiveness remains constrained when reasoning extends beyond the knowledge acquired during pretraining. Conventional approaches, such as fine-tuning or retrieval-augmented generation, face limitations in either imposing high computational overhead or failing to fully exploit structured scientific data. To overcome these challenges, we propose DrugMCTS, a novel framework that synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree Search for drug repurposing. The framework employs five specialized agents tasked with retrieving and analyzing molecular and protein information, thereby enabling structured and iterative reasoning. Without requiring domain-specific fine-tuning, DrugMCTS empowers Qwen2.5-7B-Instruct to outperform Deepseek-R1 by over 20\%. Extensive experiments on the DrugBank and KIBA datasets demonstrate that DrugMCTS achieves substantially higher recall and robustness compared to both general-purpose LLMs and deep learning baselines. Our results highlight the importance of structured reasoning, agent-based collaboration, and feedback-driven search mechanisms in advancing LLM applications for drug discovery. 

---
# Agentic Retrieval of Topics and Insights from Earnings Calls 

**Authors**: Anant Gupta, Rajarshi Bhowmik, Geoffrey Gunow  

**Link**: [PDF](https://arxiv.org/pdf/2507.07906)  

**Abstract**: Tracking the strategic focus of companies through topics in their earnings calls is a key task in financial analysis. However, as industries evolve, traditional topic modeling techniques struggle to dynamically capture emerging topics and their relationships. In this work, we propose an LLM-agent driven approach to discover and retrieve emerging topics from quarterly earnings calls. We propose an LLM-agent to extract topics from documents, structure them into a hierarchical ontology, and establish relationships between new and existing topics through a topic ontology. We demonstrate the use of extracted topics to infer company-level insights and emerging trends over time. We evaluate our approach by measuring ontology coherence, topic evolution accuracy, and its ability to surface emerging financial trends. 

---
# Performance and Practical Considerations of Large and Small Language Models in Clinical Decision Support in Rheumatology 

**Authors**: Sabine Felde, Rüdiger Buchkremer, Gamal Chehab, Christian Thielscher, Jörg HW Distler, Matthias Schneider, Jutta G. Richter  

**Link**: [PDF](https://arxiv.org/pdf/2507.07983)  

**Abstract**: Large language models (LLMs) show promise for supporting clinical decision-making in complex fields such as rheumatology. Our evaluation shows that smaller language models (SLMs), combined with retrieval-augmented generation (RAG), achieve higher diagnostic and therapeutic performance than larger models, while requiring substantially less energy and enabling cost-efficient, local deployment. These features are attractive for resource-limited healthcare. However, expert oversight remains essential, as no model consistently reached specialist-level accuracy in rheumatology. 

---
# Position: We Need An Algorithmic Understanding of Generative AI 

**Authors**: Oliver Eberle, Thomas McGee, Hamza Giaffar, Taylor Webb, Ida Momennejad  

**Link**: [PDF](https://arxiv.org/pdf/2507.07544)  

**Abstract**: What algorithms do LLMs actually learn and use to solve problems? Studies addressing this question are sparse, as research priorities are focused on improving performance through scale, leaving a theoretical and empirical gap in understanding emergent algorithms. This position paper proposes AlgEval: a framework for systematic research into the algorithms that LLMs learn and use. AlgEval aims to uncover algorithmic primitives, reflected in latent representations, attention, and inference-time compute, and their algorithmic composition to solve task-specific problems. We highlight potential methodological paths and a case study toward this goal, focusing on emergent search algorithms. Our case study illustrates both the formation of top-down hypotheses about candidate algorithms, and bottom-up tests of these hypotheses via circuit-level analysis of attention patterns and hidden states. The rigorous, systematic evaluation of how LLMs actually solve tasks provides an alternative to resource-intensive scaling, reorienting the field toward a principled understanding of underlying computations. Such algorithmic explanations offer a pathway to human-understandable interpretability, enabling comprehension of the model's internal reasoning performance measures. This can in turn lead to more sample-efficient methods for training and improving performance, as well as novel architectures for end-to-end and multi-agent systems. 

---
# From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on Retrieval-Augmented Generation systems 

**Authors**: Youngjoon Jang, Seongtae Hong, Junyoung Son, Sungjin Park, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.07847)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a crucial framework in natural language processing (NLP), improving factual consistency and reducing hallucinations by integrating external document retrieval with large language models (LLMs). However, the effectiveness of RAG is often hindered by coreferential complexity in retrieved documents, introducing ambiguity that disrupts in-context learning. In this study, we systematically investigate how entity coreference affects both document retrieval and generative performance in RAG-based systems, focusing on retrieval relevance, contextual understanding, and overall response quality. We demonstrate that coreference resolution enhances retrieval effectiveness and improves question-answering (QA) performance. Through comparative analysis of different pooling strategies in retrieval tasks, we find that mean pooling demonstrates superior context capturing ability after applying coreference resolution. In QA tasks, we discover that smaller models benefit more from the disambiguation process, likely due to their limited inherent capacity for handling referential ambiguity. With these findings, this study aims to provide a deeper understanding of the challenges posed by coreferential complexity in RAG, providing guidance for improving retrieval and generation in knowledge-intensive AI applications. 

---
# Autonomous Control Leveraging LLMs: An Agentic Framework for Next-Generation Industrial Automation 

**Authors**: Javal Vyas, Mehmet Mercangoz  

**Link**: [PDF](https://arxiv.org/pdf/2507.07115)  

**Abstract**: The increasing complexity of modern chemical processes, coupled with workforce shortages and intricate fault scenarios, demands novel automation paradigms that blend symbolic reasoning with adaptive control. In this work, we introduce a unified agentic framework that leverages large language models (LLMs) for both discrete fault-recovery planning and continuous process control within a single architecture. We adopt Finite State Machines (FSMs) as interpretable operating envelopes: an LLM-driven planning agent proposes recovery sequences through the FSM, a Simulation Agent executes and checks each transition, and a Validator-Reprompting loop iteratively refines invalid plans. In Case Study 1, across 180 randomly generated FSMs of varying sizes (4-25 states, 4-300 transitions), GPT-4o and GPT-4o-mini achieve 100% valid-path success within five reprompts-outperforming open-source LLMs in both accuracy and latency. In Case Study 2, the same framework modulates dual-heater inputs on a laboratory TCLab platform (and its digital twin) to maintain a target average temperature under persistent asymmetric disturbances. Compared to classical PID control, our LLM-based controller attains similar performance, while ablation of the prompting loop reveals its critical role in handling nonlinear dynamics. We analyze key failure modes-such as instruction following lapses and coarse ODE approximations. Our results demonstrate that, with structured feedback and modular agents, LLMs can unify high-level symbolic planningand low-level continuous control, paving the way towards resilient, language-driven automation in chemical engineering. 

---
# KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities 

**Authors**: Hruday Markondapatnaikuni, Basem Suleiman, Abdelkarim Erradi, Shijing Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.07695)  

**Abstract**: Fine-tuning is an immensely resource-intensive process when retraining Large Language Models (LLMs) to incorporate a larger body of knowledge. Although many fine-tuning techniques have been developed to reduce the time and computational cost involved, the challenge persists as LLMs continue to grow in size and complexity. To address this, a new approach to knowledge expansion in LLMs is needed. Retrieval-Augmented Generation (RAG) offers one such alternative by storing external knowledge in a database and retrieving relevant chunks to support question answering. However, naive implementations of RAG face significant limitations in scalability and answer accuracy. This paper introduces KeyKnowledgeRAG (K2RAG), a novel framework designed to overcome these limitations. Inspired by the divide-and-conquer paradigm, K2RAG integrates dense and sparse vector search, knowledge graphs, and text summarization to improve retrieval quality and system efficiency. The framework also includes a preprocessing step that summarizes the training data, significantly reducing the training time. K2RAG was evaluated using the MultiHopRAG dataset, where the proposed pipeline was trained on the document corpus and tested on a separate evaluation set. Results demonstrated notable improvements over common naive RAG implementations. K2RAG achieved the highest mean answer similarity score of 0.57, and reached the highest third quartile (Q3) similarity of 0.82, indicating better alignment with ground-truth answers. In addition to improved accuracy, the framework proved highly efficient. The summarization step reduced the average training time of individual components by 93%, and execution speed was up to 40% faster than traditional knowledge graph-based RAG systems. K2RAG also demonstrated superior scalability, requiring three times less VRAM than several naive RAG implementations tested in this study. 

---
# Bayesian Discrete Diffusion Beats Autoregressive Perplexity 

**Authors**: Cooper Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2507.07586)  

**Abstract**: We reveal a hidden Bayesian core of discrete-diffusion language models by showing that the expected denoiser output under the forward masking distribution recovers the exact posterior over clean tokens. Under minimal assumptions, Monte Carlo marginalization over K independent corruptions converges to this posterior at rate O(1/sqrt(K)), yielding a simple proof of consistency and finite-sample error bounds. Building on this insight, we introduce a lightweight inference-time ensemble that averages K mask-and-denoise passes to obtain posterior-aware token probabilities and uncertainty estimates at no extra training cost. On WikiText-2, our method achieves test perplexity 8.8 with K=8, versus 20.3 for GPT-2 Small, despite using a model of comparable size. Code is available at this https URL. 

---
# Hallucination Stations: On Some Basic Limitations of Transformer-Based Language Models 

**Authors**: Varin Sikka, Vishal Sikka  

**Link**: [PDF](https://arxiv.org/pdf/2507.07505)  

**Abstract**: With widespread adoption of transformer-based language models in AI, there is significant interest in the limits of LLMs capabilities, specifically so-called hallucinations, occurrences in which LLMs provide spurious, factually incorrect or nonsensical information when prompted on certain subjects. Furthermore, there is growing interest in agentic uses of LLMs - that is, using LLMs to create agents that act autonomously or semi-autonomously to carry out various tasks, including tasks with applications in the real world. This makes it important to understand the types of tasks LLMs can and cannot perform. We explore this topic from the perspective of the computational complexity of LLM inference. We show that LLMs are incapable of carrying out computational and agentic tasks beyond a certain complexity, and further that LLMs are incapable of verifying the accuracy of tasks beyond a certain complexity. We present examples of both, then discuss some consequences of this work. 

---
# PLAN-TUNING: Post-Training Language Models to Learn Step-by-Step Planning for Complex Problem Solving 

**Authors**: Mihir Parmar, Palash Goyal, Xin Liu, Yiwen Song, Mingyang Ling, Chitta Baral, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2507.07495)  

**Abstract**: Recently, decomposing complex problems into simple subtasks--a crucial part of human-like natural planning--to solve the given problem has significantly boosted the performance of large language models (LLMs). However, leveraging such planning structures during post-training to boost the performance of smaller open-source LLMs remains underexplored. Motivated by this, we introduce PLAN-TUNING, a unified post-training framework that (i) distills synthetic task decompositions (termed "planning trajectories") from large-scale LLMs and (ii) fine-tunes smaller models via supervised and reinforcement-learning objectives designed to mimic these planning processes to improve complex reasoning. On GSM8k and the MATH benchmarks, plan-tuned models outperform strong baselines by an average $\sim7\%$. Furthermore, plan-tuned models show better generalization capabilities on out-of-domain datasets, with average $\sim10\%$ and $\sim12\%$ performance improvements on OlympiadBench and AIME 2024, respectively. Our detailed analysis demonstrates how planning trajectories improves complex reasoning capabilities, showing that PLAN-TUNING is an effective strategy for improving task-specific performance of smaller LLMs. 

---
# CEA-LIST at CheckThat! 2025: Evaluating LLMs as Detectors of Bias and Opinion in Text 

**Authors**: Akram Elbouanani, Evan Dufraisse, Aboubacar Tuo, Adrian Popescu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07539)  

**Abstract**: This paper presents a competitive approach to multilingual subjectivity detection using large language models (LLMs) with few-shot prompting. We participated in Task 1: Subjectivity of the CheckThat! 2025 evaluation campaign. We show that LLMs, when paired with carefully designed prompts, can match or outperform fine-tuned smaller language models (SLMs), particularly in noisy or low-quality data settings. Despite experimenting with advanced prompt engineering techniques, such as debating LLMs and various example selection strategies, we found limited benefit beyond well-crafted standard few-shot prompts. Our system achieved top rankings across multiple languages in the CheckThat! 2025 subjectivity detection task, including first place in Arabic and Polish, and top-four finishes in Italian, English, German, and multilingual tracks. Notably, our method proved especially robust on the Arabic dataset, likely due to its resilience to annotation inconsistencies. These findings highlight the effectiveness and adaptability of LLM-based few-shot learning for multilingual sentiment tasks, offering a strong alternative to traditional fine-tuning, particularly when labeled data is scarce or inconsistent. 

---
# When Large Language Models Meet Law: Dual-Lens Taxonomy, Technical Advances, and Ethical Governance 

**Authors**: Peizhang Shao, Linrui Xu, Jinxi Wang, Wei Zhou, Xingyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07748)  

**Abstract**: This paper establishes the first comprehensive review of Large Language Models (LLMs) applied within the legal domain. It pioneers an innovative dual lens taxonomy that integrates legal reasoning frameworks and professional ontologies to systematically unify historical research and contemporary breakthroughs. Transformer-based LLMs, which exhibit emergent capabilities such as contextual reasoning and generative argumentation, surmount traditional limitations by dynamically capturing legal semantics and unifying evidence reasoning. Significant progress is documented in task generalization, reasoning formalization, workflow integration, and addressing core challenges in text processing, knowledge integration, and evaluation rigor via technical innovations like sparse attention mechanisms and mixture-of-experts architectures. However, widespread adoption of LLM introduces critical challenges: hallucination, explainability deficits, jurisdictional adaptation difficulties, and ethical asymmetry. This review proposes a novel taxonomy that maps legal roles to NLP subtasks and computationally implements the Toulmin argumentation framework, thus systematizing advances in reasoning, retrieval, prediction, and dispute resolution. It identifies key frontiers including low-resource systems, multimodal evidence integration, and dynamic rebuttal handling. Ultimately, this work provides both a technical roadmap for researchers and a conceptual framework for practitioners navigating the algorithmic future, laying a robust foundation for the next era of legal artificial intelligence. We have created a GitHub repository to index the relevant papers: this https URL. 

---
# Towards Interpretable Time Series Foundation Models 

**Authors**: Matthieu Boileau, Philippe Helluy, Jeremy Pawlus, Svitlana Vyetrenko  

**Link**: [PDF](https://arxiv.org/pdf/2507.07439)  

**Abstract**: In this paper, we investigate the distillation of time series reasoning capabilities into small, instruction-tuned language models as a step toward building interpretable time series foundation models. Leveraging a synthetic dataset of mean-reverting time series with systematically varied trends and noise levels, we generate natural language annotations using a large multimodal model and use these to supervise the fine-tuning of compact Qwen models. We introduce evaluation metrics that assess the quality of the distilled reasoning - focusing on trend direction, noise intensity, and extremum localization - and show that the post-trained models acquire meaningful interpretive capabilities. Our results highlight the feasibility of compressing time series understanding into lightweight, language-capable models suitable for on-device or privacy-sensitive deployment. This work contributes a concrete foundation toward developing small, interpretable models that explain temporal patterns in natural language. 

---
# MedReadCtrl: Personalizing medical text generation with readability-controlled instruction learning 

**Authors**: Hieu Tran, Zonghai Yao, Won Seok Jang, Sharmin Sultana, Allen Chang, Yuan Zhang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07419)  

**Abstract**: Generative AI has demonstrated strong potential in healthcare, from clinical decision support to patient-facing chatbots that improve outcomes. A critical challenge for deployment is effective human-AI communication, where content must be both personalized and understandable. We introduce MedReadCtrl, a readability-controlled instruction tuning framework that enables LLMs to adjust output complexity without compromising meaning. Evaluations of nine datasets and three tasks across medical and general domains show that MedReadCtrl achieves significantly lower readability instruction-following errors than GPT-4 (e.g., 1.39 vs. 1.59 on ReadMe, p<0.001) and delivers substantial gains on unseen clinical tasks (e.g., +14.7 ROUGE-L, +6.18 SARI on MTSamples). Experts consistently preferred MedReadCtrl (71.7% vs. 23.3%), especially at low literacy levels. These gains reflect MedReadCtrl's ability to restructure clinical content into accessible, readability-aligned language while preserving medical intent, offering a scalable solution to support patient education and expand equitable access to AI-enabled care. 

---
# SynthEHR-Eviction: Enhancing Eviction SDoH Detection with LLM-Augmented Synthetic EHR Data 

**Authors**: Zonghai Yao, Youxia Zhao, Avijit Mitra, David A. Levy, Emily Druhl, Jack Tsai, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07421)  

**Abstract**: Eviction is a significant yet understudied social determinants of health (SDoH), linked to housing instability, unemployment, and mental health. While eviction appears in unstructured electronic health records (EHRs), it is rarely coded in structured fields, limiting downstream applications. We introduce SynthEHR-Eviction, a scalable pipeline combining LLMs, human-in-the-loop annotation, and automated prompt optimization (APO) to extract eviction statuses from clinical notes. Using this pipeline, we created the largest public eviction-related SDoH dataset to date, comprising 14 fine-grained categories. Fine-tuned LLMs (e.g., Qwen2.5, LLaMA3) trained on SynthEHR-Eviction achieved Macro-F1 scores of 88.8% (eviction) and 90.3% (other SDoH) on human validated data, outperforming GPT-4o-APO (87.8%, 87.3%), GPT-4o-mini-APO (69.1%, 78.1%), and BioBERT (60.7%, 68.3%), while enabling cost-effective deployment across various model sizes. The pipeline reduces annotation effort by over 80%, accelerates dataset creation, enables scalable eviction detection, and generalizes to other information extraction tasks. 

---
# Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models 

**Authors**: Jikesh Thapa, Gurrehmat Chahal, Serban Voinea Gabreanu, Yazan Otoum  

**Link**: [PDF](https://arxiv.org/pdf/2507.07406)  

**Abstract**: Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks. 

---
# Hybrid LLM-Enhanced Intrusion Detection for Zero-Day Threats in IoT Networks 

**Authors**: Mohammad F. Al-Hammouri, Yazan Otoum, Rasha Atwa, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2507.07413)  

**Abstract**: This paper presents a novel approach to intrusion detection by integrating traditional signature-based methods with the contextual understanding capabilities of the GPT-2 Large Language Model (LLM). As cyber threats become increasingly sophisticated, particularly in distributed, heterogeneous, and resource-constrained environments such as those enabled by the Internet of Things (IoT), the need for dynamic and adaptive Intrusion Detection Systems (IDSs) becomes increasingly urgent. While traditional methods remain effective for detecting known threats, they often fail to recognize new and evolving attack patterns. In contrast, GPT-2 excels at processing unstructured data and identifying complex semantic relationships, making it well-suited to uncovering subtle, zero-day attack vectors. We propose a hybrid IDS framework that merges the robustness of signature-based techniques with the adaptability of GPT-2-driven semantic analysis. Experimental evaluations on a representative intrusion dataset demonstrate that our model enhances detection accuracy by 6.3%, reduces false positives by 9.0%, and maintains near real-time responsiveness. These results affirm the potential of language model integration to build intelligent, scalable, and resilient cybersecurity defences suited for modern connected environments. 

---
# GNN-CNN: An Efficient Hybrid Model of Convolutional and Graph Neural Networks for Text Representation 

**Authors**: Fardin Rastakhiz  

**Link**: [PDF](https://arxiv.org/pdf/2507.07414)  

**Abstract**: Time, cost, and energy efficiency are critical considerations in Deep-Learning (DL), particularly when processing long texts. Transformers, which represent the current state of the art, exhibit quadratic computational complexity relative to input length, making them inefficient for extended documents. This study introduces a novel model architecture that combines Graph Neural Networks (GNNs) and Convolutional Neural Networks (CNNs), integrated with a real-time, end-to-end graph generation mechanism. The model processes compact batches of character-level inputs without requiring padding or truncation. To enhance performance while maintaining high speed and efficiency, the model incorporates information from Large Language Models (LLMs), such as token embeddings and sentiment polarities, through efficient dictionary lookups. It captures local contextual patterns using CNNs, expands local receptive fields via lattice-based graph structures, and employs small-world graphs to aggregate document-level information. The generated graphs exhibit structural properties indicative of meaningful semantic organization, with an average clustering coefficient of approximately 0.45 and an average shortest path length ranging between 4 and 5. The model is evaluated across multiple text classification tasks, including sentiment analysis and news-categorization, and is compared against state-of-the-art models. Experimental results confirm the proposed model's efficiency and competitive performance. 

---
# Toward Real-World Chinese Psychological Support Dialogues: CPsDD Dataset and a Co-Evolving Multi-Agent System 

**Authors**: Yuanchen Shi, Longyin Zhang, Fang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07509)  

**Abstract**: The growing need for psychological support due to increasing pressures has exposed the scarcity of relevant datasets, particularly in non-English languages. To address this, we propose a framework that leverages limited real-world data and expert knowledge to fine-tune two large language models: Dialog Generator and Dialog Modifier. The Generator creates large-scale psychological counseling dialogues based on predefined paths, which guide system response strategies and user interactions, forming the basis for effective support. The Modifier refines these dialogues to align with real-world data quality. Through both automated and manual review, we construct the Chinese Psychological support Dialogue Dataset (CPsDD), containing 68K dialogues across 13 groups, 16 psychological problems, 13 causes, and 12 support focuses. Additionally, we introduce the Comprehensive Agent Dialogue Support System (CADSS), where a Profiler analyzes user characteristics, a Summarizer condenses dialogue history, a Planner selects strategies, and a Supporter generates empathetic responses. The experimental results of the Strategy Prediction and Emotional Support Conversation (ESC) tasks demonstrate that CADSS achieves state-of-the-art performance on both CPsDD and ESConv datasets. 

---
# Bridging the Plausibility-Validity Gap by Fine-Tuning a Reasoning-Enhanced LLM for Chemical Synthesis and Discovery 

**Authors**: Malikussaid, Hilal Hudan Nuha  

**Link**: [PDF](https://arxiv.org/pdf/2507.07328)  

**Abstract**: Large Language Models (LLMs) often generate scientifically plausible but factually invalid information, a challenge we term the "plausibility-validity gap," particularly in specialized domains like chemistry. This paper presents a systematic methodology to bridge this gap by developing a specialized scientific assistant. We utilized the Magistral Small model, noted for its integrated reasoning capabilities, and fine-tuned it using Low-Rank Adaptation (LoRA). A key component of our approach was the creation of a "dual-domain dataset," a comprehensive corpus curated from various sources encompassing both molecular properties and chemical reactions, which was standardized to ensure quality. Our evaluation demonstrates that the fine-tuned model achieves significant improvements over the baseline model in format adherence, chemical validity of generated molecules, and the feasibility of proposed synthesis routes. The results indicate a hierarchical learning pattern, where syntactic correctness is learned more readily than chemical possibility and synthesis feasibility. While a comparative analysis with human experts revealed competitive performance in areas like chemical creativity and reasoning, it also highlighted key limitations, including persistent errors in stereochemistry, a static knowledge cutoff, and occasional reference hallucination. This work establishes a viable framework for adapting generalist LLMs into reliable, specialized tools for chemical research, while also delineating critical areas for future improvement. 

---
# Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought 

**Authors**: Shin'ya Yamaguchi, Kosuke Nishida, Daiki Chijiwa  

**Link**: [PDF](https://arxiv.org/pdf/2507.07685)  

**Abstract**: Large vision-language models (LVLMs) have demonstrated remarkable capabilities by integrating pre-trained vision encoders with large language models (LLMs). Similar to single-modal LLMs, chain-of-thought (CoT) prompting has been adapted for LVLMs to enhance multi-modal reasoning by generating intermediate rationales based on visual and textual inputs. While CoT is assumed to improve grounding and accuracy in LVLMs, our experiments reveal a key challenge: existing LVLMs often ignore the contents of generated rationales in CoT reasoning. To address this, we re-formulate multi-modal CoT reasoning as a KL-constrained reward maximization focused on rationale-conditional log-likelihood. As the optimal solution, we propose rationale-enhanced decoding (RED), a novel plug-and-play inference-time decoding strategy. RED harmonizes visual and rationale information by multiplying distinct image-conditional and rationale-conditional next token distributions. Extensive experiments show that RED consistently and significantly improves reasoning over standard CoT and other decoding methods across multiple benchmarks and LVLMs. Our work offers a practical and effective approach to improve both the faithfulness and accuracy of CoT reasoning in LVLMs, paving the way for more reliable rationale-grounded multi-modal systems. 

---
# Behave Your Motion: Habit-preserved Cross-category Animal Motion Transfer 

**Authors**: Zhimin Zhang, Bi'an Du, Caoyuan Ma, Zheng Wang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07394)  

**Abstract**: Animal motion embodies species-specific behavioral habits, making the transfer of motion across categories a critical yet complex task for applications in animation and virtual reality. Existing motion transfer methods, primarily focused on human motion, emphasize skeletal alignment (motion retargeting) or stylistic consistency (motion style transfer), often neglecting the preservation of distinct habitual behaviors in animals. To bridge this gap, we propose a novel habit-preserved motion transfer framework for cross-category animal motion. Built upon a generative framework, our model introduces a habit-preservation module with category-specific habit encoder, allowing it to learn motion priors that capture distinctive habitual characteristics. Furthermore, we integrate a large language model (LLM) to facilitate the motion transfer to previously unobserved species. To evaluate the effectiveness of our approach, we introduce the DeformingThings4D-skl dataset, a quadruped dataset with skeletal bindings, and conduct extensive experiments and quantitative analyses, which validate the superiority of our proposed model. 

---
# Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation 

**Authors**: Yupu Liang, Yaping Zhang, Zhiyang Zhang, Yang Zhao, Lu Xiang, Chengqing Zong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.07572)  

**Abstract**: Document Image Machine Translation (DIMT) aims to translate text within document images, facing generalization challenges due to limited training data and the complex interplay between visual and textual information. To address these challenges, we introduce M4Doc, a novel single-to-mix modality alignment framework leveraging Multimodal Large Language Models (MLLMs). M4Doc aligns an image-only encoder with the multimodal representations of an MLLM, pre-trained on large-scale document image datasets. This alignment enables a lightweight DIMT model to learn crucial visual-textual correlations during training. During inference, M4Doc bypasses the MLLM, maintaining computational efficiency while benefiting from its multimodal knowledge. Comprehensive experiments demonstrate substantial improvements in translation quality, especially in cross-domain generalization and challenging document image scenarios. 

---
# An Information-Theoretic Perspective on Multi-LLM Uncertainty Estimation 

**Authors**: Maya Kruse, Majid Afshar, Saksham Khatwani, Anoop Mayampurath, Guanhua Chen, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.07236)  

**Abstract**: Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and naive ensemble baselines. 

---
# Planted in Pretraining, Swayed by Finetuning: A Case Study on the Origins of Cognitive Biases in LLMs 

**Authors**: Itay Itzhak, Yonatan Belinkov, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2507.07186)  

**Abstract**: Large language models (LLMs) exhibit cognitive biases -- systematic tendencies of irrational decision-making, similar to those seen in humans. Prior work has found that these biases vary across models and can be amplified by instruction tuning. However, it remains unclear if these differences in biases stem from pretraining, finetuning, or even random noise due to training stochasticity. We propose a two-step causal experimental approach to disentangle these factors. First, we finetune models multiple times using different random seeds to study how training randomness affects over $30$ cognitive biases. Second, we introduce \emph{cross-tuning} -- swapping instruction datasets between models to isolate bias sources. This swap uses datasets that led to different bias patterns, directly testing whether biases are dataset-dependent. Our findings reveal that while training randomness introduces some variability, biases are mainly shaped by pretraining: models with the same pretrained backbone exhibit more similar bias patterns than those sharing only finetuning data. These insights suggest that understanding biases in finetuned models requires considering their pretraining origins beyond finetuning effects. This perspective can guide future efforts to develop principled strategies for evaluating and mitigating bias in LLMs. 

---
# Machine Bullshit: Characterizing the Emergent Disregard for Truth in Large Language Models 

**Authors**: Kaiqu Liang, Haimin Hu, Xuandong Zhao, Dawn Song, Thomas L. Griffiths, Jaime Fernández Fisac  

**Link**: [PDF](https://arxiv.org/pdf/2507.07484)  

**Abstract**: Bullshit, as conceptualized by philosopher Harry Frankfurt, refers to statements made without regard to their truth value. While previous work has explored large language model (LLM) hallucination and sycophancy, we propose machine bullshit as an overarching conceptual framework that can allow researchers to characterize the broader phenomenon of emergent loss of truthfulness in LLMs and shed light on its underlying mechanisms. We introduce the Bullshit Index, a novel metric quantifying LLMs' indifference to truth, and propose a complementary taxonomy analyzing four qualitative forms of bullshit: empty rhetoric, paltering, weasel words, and unverified claims. We conduct empirical evaluations on the Marketplace dataset, the Political Neutrality dataset, and our new BullshitEval benchmark (2,400 scenarios spanning 100 AI assistants) explicitly designed to evaluate machine bullshit. Our results demonstrate that model fine-tuning with reinforcement learning from human feedback (RLHF) significantly exacerbates bullshit and inference-time chain-of-thought (CoT) prompting notably amplify specific bullshit forms, particularly empty rhetoric and paltering. We also observe prevalent machine bullshit in political contexts, with weasel words as the dominant strategy. Our findings highlight systematic challenges in AI alignment and provide new insights toward more truthful LLM behavior. 

---
# Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses 

**Authors**: Jens Rupprecht, Georg Ahnert, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2507.07188)  

**Abstract**: Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts -- we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also reveal that all tested models exhibit a consistent \textit{recency bias} varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data. 

---
# Robust Multimodal Large Language Models Against Modality Conflict 

**Authors**: Zongmeng Zhang, Wengang Zhou, Jie Zhao, Houqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.07151)  

**Abstract**: Despite the impressive capabilities of multimodal large language models (MLLMs) in vision-language tasks, they are prone to hallucinations in real-world scenarios. This paper investigates the hallucination phenomenon in MLLMs from the perspective of modality conflict. Unlike existing works focusing on the conflicts between model responses and inputs, we study the inherent conflicts in inputs from different modalities that place MLLMs in a dilemma and directly lead to hallucinations. We formally define the modality conflict and construct a dataset named Multimodal Modality Conflict (MMMC) to simulate this phenomenon in vision-language tasks. Three methods based on prompt engineering, supervised fine-tuning, and reinforcement learning are proposed to alleviate the hallucination caused by modality conflict. Extensive experiments are conducted on the MMMC dataset to analyze the merits and demerits of these methods. Our results show that the reinforcement learning method achieves the best performance in mitigating the hallucination under modality conflict, while the supervised fine-tuning method shows promising and stable performance. Our work sheds light on the unnoticed modality conflict that leads to hallucinations and provides more insights into the robustness of MLLMs. 

---
# Multi-level Mixture of Experts for Multimodal Entity Linking 

**Authors**: Zhiwei Hu, Víctor Gutiérrez-Basulto, Zhiliang Xiang, Ru Li, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.07108)  

**Abstract**: Multimodal Entity Linking (MEL) aims to link ambiguous mentions within multimodal contexts to associated entities in a multimodal knowledge base. Existing approaches to MEL introduce multimodal interaction and fusion mechanisms to bridge the modality gap and enable multi-grained semantic matching. However, they do not address two important problems: (i) mention ambiguity, i.e., the lack of semantic content caused by the brevity and omission of key information in the mention's textual context; (ii) dynamic selection of modal content, i.e., to dynamically distinguish the importance of different parts of modal information. To mitigate these issues, we propose a Multi-level Mixture of Experts (MMoE) model for MEL. MMoE has four components: (i) the description-aware mention enhancement module leverages large language models to identify the WikiData descriptions that best match a mention, considering the mention's textual context; (ii) the multimodal feature extraction module adopts multimodal feature encoders to obtain textual and visual embeddings for both mentions and entities; (iii)-(iv) the intra-level mixture of experts and inter-level mixture of experts modules apply a switch mixture of experts mechanism to dynamically and adaptively select features from relevant regions of information. Extensive experiments demonstrate the outstanding performance of MMoE compared to the state-of-the-art. MMoE's code is available at: this https URL. 

---
# A Language-Driven Framework for Improving Personalized Recommendations: Merging LLMs with Traditional Algorithms 

**Authors**: Aaron Goldstein, Ayan Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2507.07251)  

**Abstract**: Traditional recommendation algorithms are not designed to provide personalized recommendations based on user preferences provided through text, e.g., "I enjoy light-hearted comedies with a lot of humor". Large Language Models (LLMs) have emerged as one of the most promising tools for natural language processing in recent years. This research proposes a novel framework that mimics how a close friend would recommend items based on their knowledge of an individual's tastes. We leverage LLMs to enhance movie recommendation systems by refining traditional algorithm outputs and integrating them with language-based user preference inputs. We employ Singular Value Decomposition (SVD) or SVD++ algorithms to generate initial movie recommendations, implemented using the Surprise Python library and trained on the MovieLens-Latest-Small dataset. We compare the performance of the base algorithms with our LLM-enhanced versions using leave-one-out validation hit rates and cumulative hit rates. Additionally, to compare the performance of our framework against the current state-of-the-art recommendation systems, we use rating and ranking metrics with an item-based stratified 0.75 train, 0.25 test split. Our framework can generate preference profiles automatically based on users' favorite movies or allow manual preference specification for more personalized results. Using an automated approach, our framework overwhelmingly surpassed SVD and SVD++ on every evaluation metric used (e.g., improvements of up to ~6x in cumulative hit rate, ~3.7x in NDCG, etc.), albeit at the cost of a slight increase in computational overhead. 

---
# DocCHA: Towards LLM-Augmented Interactive Online diagnosis System 

**Authors**: Xinyi Liu, Dachun Sun, Yi R. Fung, Dilek Hakkani-Tür, Tarek Abdelzaher  

**Link**: [PDF](https://arxiv.org/pdf/2507.07870)  

**Abstract**: Despite the impressive capabilities of Large Language Models (LLMs), existing Conversational Health Agents (CHAs) remain static and brittle, incapable of adaptive multi-turn reasoning, symptom clarification, or transparent decision-making. This hinders their real-world applicability in clinical diagnosis, where iterative and structured dialogue is essential. We propose DocCHA, a confidence-aware, modular framework that emulates clinical reasoning by decomposing the diagnostic process into three stages: (1) symptom elicitation, (2) history acquisition, and (3) causal graph construction. Each module uses interpretable confidence scores to guide adaptive questioning, prioritize informative clarifications, and refine weak reasoning links.
Evaluated on two real-world Chinese consultation datasets (IMCS21, DX), DocCHA consistently outperforms strong prompting-based LLM baselines (GPT-3.5, GPT-4o, LLaMA-3), achieving up to 5.18 percent higher diagnostic accuracy and over 30 percent improvement in symptom recall, with only modest increase in dialogue turns. These results demonstrate the effectiveness of DocCHA in enabling structured, transparent, and efficient diagnostic conversations -- paving the way for trustworthy LLM-powered clinical assistants in multilingual and resource-constrained settings. 

---
# Understanding and Controlling Repetition Neurons and Induction Heads in In-Context Learning 

**Authors**: Nhi Hoai Doan, Tatsuya Hiraoka, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2507.07810)  

**Abstract**: This paper investigates the relationship between large language models' (LLMs) ability to recognize repetitive input patterns and their performance on in-context learning (ICL). In contrast to prior work that has primarily focused on attention heads, we examine this relationship from the perspective of skill neurons, specifically repetition neurons. Our experiments reveal that the impact of these neurons on ICL performance varies depending on the depth of the layer in which they reside. By comparing the effects of repetition neurons and induction heads, we further identify strategies for reducing repetitive outputs while maintaining strong ICL capabilities. 

---
# StreamUni: Achieving Streaming Speech Translation with a Unified Large Speech-Language Model 

**Authors**: Shoutao Guo, Xiang Li, Shaolei Zhang, Mengge Liu, Wei Chen, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.07803)  

**Abstract**: Streaming speech translation (StreamST) requires determining appropriate timing, known as policy, to generate translations while continuously receiving source speech inputs, balancing low latency with high translation quality. However, existing StreamST methods typically operate on sentence-level speech segments, referred to as simultaneous speech translation (SimulST). In practice, they require collaboration with segmentation models to accomplish StreamST, where the truncated speech segments constrain SimulST models to make policy decisions and generate translations based on limited contextual information. Moreover, SimulST models struggle to learn effective policies due to the complexity of speech inputs and cross-lingual generation. To address these challenges, we propose StreamUni, which achieves StreamST through a unified Large Speech-Language Model (LSLM). Specifically, StreamUni incorporates speech Chain-of-Thought (CoT) in guiding the LSLM to generate multi-stage outputs. Leveraging these multi-stage outputs, StreamUni simultaneously accomplishes speech segmentation, policy decision, and translation generation, completing StreamST without requiring massive policy-specific training. Additionally, we propose a streaming CoT training method that enhances low-latency policy decisions and generation capabilities using limited CoT data. Experiments demonstrate that our approach achieves state-of-the-art performance on StreamST tasks. 

---
# Automating MD simulations for Proteins using Large language Models: NAMD-Agent 

**Authors**: Achuth Chandrasekhar, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2507.07887)  

**Abstract**: Molecular dynamics simulations are an essential tool in understanding protein structure, dynamics, and function at the atomic level. However, preparing high quality input files for MD simulations can be a time consuming and error prone process. In this work, we introduce an automated pipeline that leverages Large Language Models (LLMs), specifically Gemini 2.0 Flash, in conjunction with python scripting and Selenium based web automation to streamline the generation of MD input files. The pipeline exploits CHARMM GUI's comprehensive web-based interface for preparing simulation-ready inputs for NAMD. By integrating Gemini's code generation and iterative refinement capabilities, simulation scripts are automatically written, executed, and revised to navigate CHARMM GUI, extract appropriate parameters, and produce the required NAMD input files. Post processing is performed using additional software to further refine the simulation outputs, thereby enabling a complete and largely hands free workflow. Our results demonstrate that this approach reduces setup time, minimizes manual errors, and offers a scalable solution for handling multiple protein systems in parallel. This automated framework paves the way for broader application of LLMs in computational structural biology, offering a robust and adaptable platform for future developments in simulation automation. 

---
# Not All Preferences are What You Need for Post-Training: Selective Alignment Strategy for Preference Optimization 

**Authors**: Zhijin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07725)  

**Abstract**: Post-training alignment of large language models (LLMs) is a critical challenge, as not all tokens contribute equally to model performance. This paper introduces a selective alignment strategy that prioritizes high-impact tokens within preference pairs, leveraging token-level log-probability differences between the current policy and a reference model. By focusing on these informative tokens, our approach reduces computational overhead and enhances alignment fidelity. We further explore the role of reference model quality, demonstrating that stronger reference models significantly improve token selection accuracy and overall optimization effectiveness. Comprehensive experiments on benchmarks such as Arena-Hard and MT-Bench validate the superiority of our Selective-DPO method over standard DPO and distillation-based baselines. Our findings highlight the importance of token-level optimization and reference model selection in advancing preference alignment for LLMs. The code is available at this https URL. 

---
# Exploring the Limits of Model Compression in LLMs: A Knowledge Distillation Study on QA Tasks 

**Authors**: Joyeeta Datta, Niclas Doll, Qusai Ramadan, Zeyd Boukhers  

**Link**: [PDF](https://arxiv.org/pdf/2507.07630)  

**Abstract**: Large Language Models (LLMs) have demonstrated outstanding performance across a range of NLP tasks, however, their computational demands hinder their deployment in real-world, resource-constrained environments. This work investigates the extent to which LLMs can be compressed using Knowledge Distillation (KD) while maintaining strong performance on Question Answering (QA) tasks. We evaluate student models distilled from the Pythia and Qwen2.5 families on two QA benchmarks, SQuAD and MLQA, under zero-shot and one-shot prompting conditions. Results show that student models retain over 90% of their teacher models' performance while reducing parameter counts by up to 57.1%. Furthermore, one-shot prompting yields additional performance gains over zero-shot setups for both model families. These findings underscore the trade-off between model efficiency and task performance, demonstrating that KD, combined with minimal prompting, can yield compact yet capable QA systems suitable for resource-constrained applications. 

---
# Teaching LLM to Reason: Reinforcement Learning from Algorithmic Problems without Code 

**Authors**: Keqin Bao, Nuo Chen, Xiaoyuan Li, Binyuan Hui, Bowen Yu, Fuli Feng, Junyang Lin, Xiangnan He, Dayiheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.07498)  

**Abstract**: Enhancing reasoning capabilities remains a central focus in the LLM reasearch community. A promising direction involves requiring models to simulate code execution step-by-step to derive outputs for given inputs. However, as code is often designed for large-scale systems, direct application leads to over-reliance on complex data structures and algorithms, even for simple cases, resulting in overfitting to algorithmic patterns rather than core reasoning structures. To address this, we propose TeaR, which aims at teaching LLMs to reason better. TeaR leverages careful data curation and reinforcement learning to guide models in discovering optimal reasoning paths through code-related tasks, thereby improving general reasoning abilities. We conduct extensive experiments using two base models and three long-CoT distillation models, with model sizes ranging from 1.5 billion to 32 billion parameters, and across 17 benchmarks spanning Math, Knowledge, Code, and Logical Reasoning. The results consistently show significant performance improvements. Notably, TeaR achieves a 35.9% improvement on Qwen2.5-7B and 5.9% on R1-Distilled-7B. 

---
# SAND: Boosting LLM Agents with Self-Taught Action Deliberation 

**Authors**: Yu Xia, Yiran Jenny Shen, Junda Wu, Tong Yu, Sungchul Kim, Ryan A. Rossi, Lina Yao, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.07441)  

**Abstract**: Large Language Model (LLM) agents are commonly tuned with supervised finetuning on ReAct-style expert trajectories or preference optimization over pairwise rollouts. Most of these methods focus on imitating specific expert behaviors or promoting chosen reasoning thoughts and actions over rejected ones. However, without reasoning and comparing over alternatives actions, LLM agents finetuned with these methods may over-commit towards seemingly plausible but suboptimal actions due to limited action space exploration. To address this, in this paper we propose Self-taught ActioN Deliberation (SAND) framework, enabling LLM agents to explicitly deliberate over candidate actions before committing to one. To tackle the challenges of when and what to deliberate given large action space and step-level action evaluation, we incorporate self-consistency action sampling and execution-guided action critique to help synthesize step-wise action deliberation thoughts using the base model of the LLM agent. In an iterative manner, the deliberation trajectories are then used to finetune the LLM agent itself. Evaluating on two representative interactive agent tasks, SAND achieves an average 20% improvement over initial supervised finetuning and also outperforms state-of-the-art agent tuning approaches. 

---
# Multi-Agent Retrieval-Augmented Framework for Evidence-Based Counterspeech Against Health Misinformation 

**Authors**: Anirban Saha Anik, Xiaoying Song, Elliott Wang, Bryan Wang, Bengisu Yarimbas, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.07307)  

**Abstract**: Large language models (LLMs) incorporated with Retrieval-Augmented Generation (RAG) have demonstrated powerful capabilities in generating counterspeech against misinformation. However, current studies rely on limited evidence and offer less control over final outputs. To address these challenges, we propose a Multi-agent Retrieval-Augmented Framework to generate counterspeech against health misinformation, incorporating multiple LLMs to optimize knowledge retrieval, evidence enhancement, and response refinement. Our approach integrates both static and dynamic evidence, ensuring that the generated counterspeech is relevant, well-grounded, and up-to-date. Our method outperforms baseline approaches in politeness, relevance, informativeness, and factual accuracy, demonstrating its effectiveness in generating high-quality counterspeech. To further validate our approach, we conduct ablation studies to verify the necessity of each component in our framework. Furthermore, human evaluations reveal that refinement significantly enhances counterspeech quality and obtains human preference. 

---
# SynthTextEval: Synthetic Text Data Generation and Evaluation for High-Stakes Domains 

**Authors**: Krithika Ramesh, Daniel Smolyak, Zihao Zhao, Nupoor Gandhi, Ritu Agarwal, Margrét Bjarnadóttir, Anjalie Field  

**Link**: [PDF](https://arxiv.org/pdf/2507.07229)  

**Abstract**: We present SynthTextEval, a toolkit for conducting comprehensive evaluations of synthetic text. The fluency of large language model (LLM) outputs has made synthetic text potentially viable for numerous applications, such as reducing the risks of privacy violations in the development and deployment of AI systems in high-stakes domains. Realizing this potential, however, requires principled consistent evaluations of synthetic data across multiple dimensions: its utility in downstream systems, the fairness of these systems, the risk of privacy leakage, general distributional differences from the source text, and qualitative feedback from domain experts. SynthTextEval allows users to conduct evaluations along all of these dimensions over synthetic data that they upload or generate using the toolkit's generation module. While our toolkit can be run over any data, we highlight its functionality and effectiveness over datasets from two high-stakes domains: healthcare and law. By consolidating and standardizing evaluation metrics, we aim to improve the viability of synthetic text, and in-turn, privacy-preservation in AI development. 

---
# GuardVal: Dynamic Large Language Model Jailbreak Evaluation for Comprehensive Safety Testing 

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07735)  

**Abstract**: Jailbreak attacks reveal critical vulnerabilities in Large Language Models (LLMs) by causing them to generate harmful or unethical content. Evaluating these threats is particularly challenging due to the evolving nature of LLMs and the sophistication required in effectively probing their vulnerabilities. Current benchmarks and evaluation methods struggle to fully address these challenges, leaving gaps in the assessment of LLM vulnerabilities. In this paper, we review existing jailbreak evaluation practices and identify three assumed desiderata for an effective jailbreak evaluation protocol. To address these challenges, we introduce GuardVal, a new evaluation protocol that dynamically generates and refines jailbreak prompts based on the defender LLM's state, providing a more accurate assessment of defender LLMs' capacity to handle safety-critical situations. Moreover, we propose a new optimization method that prevents stagnation during prompt refinement, ensuring the generation of increasingly effective jailbreak prompts that expose deeper weaknesses in the defender LLMs. We apply this protocol to a diverse set of models, from Mistral-7b to GPT-4, across 10 safety domains. Our findings highlight distinct behavioral patterns among the models, offering a comprehensive view of their robustness. Furthermore, our evaluation process deepens the understanding of LLM behavior, leading to insights that can inform future research and drive the development of more secure models. 

---
# SpatialViz-Bench: Automatically Generated Spatial Visualization Reasoning Tasks for MLLMs 

**Authors**: Siting Wang, Luoyang Sun, Cheng Deng, Kun Shao, Minnan Pei, Zheng Tian, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07610)  

**Abstract**: Humans can directly imagine and manipulate visual images in their minds, a capability known as spatial visualization. While multi-modal Large Language Models (MLLMs) support imagination-based reasoning, spatial visualization remains insufficiently evaluated, typically embedded within broader mathematical and logical assessments. Existing evaluations often rely on IQ tests or math competitions that may overlap with training data, compromising assessment reliability. To this end, we introduce SpatialViz-Bench, a comprehensive multi-modal benchmark for spatial visualization with 12 tasks across 4 sub-abilities, comprising 1,180 automatically generated problems. Our evaluation of 33 state-of-the-art MLLMs not only reveals wide performance variations and demonstrates the benchmark's strong discriminative power, but also uncovers counter-intuitive findings: models exhibit unexpected behaviors by showing difficulty perception that misaligns with human intuition, displaying dramatic 2D-to-3D performance cliffs, and defaulting to formula derivation despite spatial tasks requiring visualization alone. SpatialVizBench empirically demonstrates that state-of-the-art MLLMs continue to exhibit deficiencies in spatial visualization tasks, thereby addressing a significant lacuna in the field. The benchmark is publicly available. 

---
# May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks 

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes  

**Link**: [PDF](https://arxiv.org/pdf/2507.07417)  

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning the model to separate instructions and data, so that the LLM does not follow instructions that might be present with data. There are several academic systems and production-level implementations of this idea. We evaluate the robustness of this class of prompt injection defenses in the whitebox setting by constructing strong optimization-based attacks and showing that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for text-based LLMs and apply it to two recent whitebox defenses SecAlign (CCS 2025) and StruQ (USENIX Security 2025), showing attacks with success rates of up to 70% with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at this https URL 

---
# Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.07129)  

**Abstract**: The prevailing paradigm for scaling large language models (LLMs) involves monolithic, end-to-end training, a resource-intensive process that lacks flexibility. This paper explores an alternative, constructive approach to model development, built upon the foundation of non-trainable, deterministic input embeddings. In prior [1], we established that high-level semantic reasoning can emerge in Transformers using frozen embeddings derived from the visual structure of Unicode glyphs. Here, we demonstrate that this fixed representational substrate acts as a universal "docking port," enabling two powerful and efficient scaling paradigms: seamless modular composition and progressive layer-wise growth.
First, we show that specialist models trained on disparate datasets (e.g., Russian and Chinese text) can be merged into a single, more capable Mixture-of-Experts (MoE) model, post-training, with zero architectural modification. This is achieved by simply averaging their output logits. The resulting MoE model exhibits immediate performance improvements on reasoning benchmarks like MMLU, surpassing its constituent experts without catastrophic forgetting. Second, we introduce a layer-wise constructive training methodology, where a deep Transformer is "grown" by progressively stacking and training one layer at a time. This method demonstrates stable convergence and a clear correlation between model depth and the emergence of complex reasoning abilities, such as those required for SQuAD.
Our findings suggest a paradigm shift from monolithic optimization towards a more biological or constructive model of AI development, where complexity is built incrementally and modules can be composed freely. This opens new avenues for resource-efficient scaling, continual learning, and a more democratized ecosystem for building powerful AI systems. We release all code and models to facilitate further research. 

---
# RLEP: Reinforcement Learning with Experience Replay for LLM Reasoning 

**Authors**: Hongzhi Zhang, Jia Fu, Jingyuan Zhang, Kai Fu, Qi Wang, Fuzheng Zhang, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.07451)  

**Abstract**: Reinforcement learning (RL) for large language models is an energy-intensive endeavor: training can be unstable, and the policy may gradually drift away from its pretrained weights. We present \emph{RLEP}\, -- \,Reinforcement Learning with Experience rePlay\, -- \,a two-phase framework that first collects verified trajectories and then replays them during subsequent training. At every update step, the policy is optimized on mini-batches that blend newly generated rollouts with these replayed successes. By replaying high-quality examples, RLEP steers the model away from fruitless exploration, focuses learning on promising reasoning paths, and delivers both faster convergence and stronger final performance. On the Qwen2.5-Math-7B base model, RLEP reaches baseline peak accuracy with substantially fewer updates and ultimately surpasses it, improving accuracy on AIME-2024 from 38.2% to 39.9%, on AIME-2025 from 19.8% to 22.3%, and on AMC-2023 from 77.0% to 82.2%. Our code, datasets, and checkpoints are publicly available at this https URL to facilitate reproducibility and further research. 

---
# Automating Expert-Level Medical Reasoning Evaluation of Large Language Models 

**Authors**: Shuang Zhou, Wenya Xie, Jiaxi Li, Zaifu Zhan, Meijia Song, Han Yang, Cheyenna Espinoza, Lindsay Welton, Xinnie Mai, Yanwei Jin, Zidu Xu, Yuen-Hei Chung, Yiyun Xing, Meng-Han Tsai, Emma Schaffer, Yucheng Shi, Ninghao Liu, Zirui Liu, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.07988)  

**Abstract**: As large language models (LLMs) become increasingly integrated into clinical decision-making, ensuring transparent and trustworthy reasoning is essential. However, existing evaluation strategies of LLMs' medical reasoning capability either suffer from unsatisfactory assessment or poor scalability, and a rigorous benchmark remains lacking. To address this, we introduce MedThink-Bench, a benchmark designed for rigorous, explainable, and scalable assessment of LLMs' medical reasoning. MedThink-Bench comprises 500 challenging questions across ten medical domains, each annotated with expert-crafted step-by-step rationales. Building on this, we propose LLM-w-Ref, a novel evaluation framework that leverages fine-grained rationales and LLM-as-a-Judge mechanisms to assess intermediate reasoning with expert-level fidelity while maintaining scalability. Experiments show that LLM-w-Ref exhibits a strong positive correlation with expert judgments. Benchmarking twelve state-of-the-art LLMs, we find that smaller models (e.g., MedGemma-27B) can surpass larger proprietary counterparts (e.g., OpenAI-o3). Overall, MedThink-Bench offers a foundational tool for evaluating LLMs' medical reasoning, advancing their safe and responsible deployment in clinical practice. 

---
