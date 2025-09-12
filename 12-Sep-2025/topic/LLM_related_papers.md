# Towards Adaptive ML Benchmarks: Web-Agent-Driven Construction, Domain Expansion, and Metric Optimization 

**Authors**: Hangyi Jia, Yuxi Qian, Hanwen Tong, Xinhui Wu, Lin Chen, Feng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.09321)  

**Abstract**: Recent advances in large language models (LLMs) have enabled the emergence of general-purpose agents for automating end-to-end machine learning (ML) workflows, including data analysis, feature engineering, model training, and competition solving. However, existing benchmarks remain limited in task coverage, domain diversity, difficulty modeling, and evaluation rigor, failing to capture the full capabilities of such agents in realistic settings. We present TAM Bench, a diverse, realistic, and structured benchmark for evaluating LLM-based agents on end-to-end ML tasks. TAM Bench features three key innovations: (1) A browser automation and LLM-based task acquisition system that automatically collects and structures ML challenges from platforms such as Kaggle, AIcrowd, and Biendata, spanning multiple task types and data modalities (e.g., tabular, text, image, graph, audio); (2) A leaderboard-driven difficulty modeling mechanism that estimates task complexity using participant counts and score dispersion, enabling scalable and objective task calibration; (3) A multi-dimensional evaluation framework incorporating performance, format compliance, constraint adherence, and task generalization. Based on 150 curated AutoML tasks, we construct three benchmark subsets of different sizes -- Lite, Medium, and Full -- designed for varying evaluation scenarios. The Lite version, with 18 tasks and balanced coverage across modalities and difficulty levels, serves as a practical testbed for daily benchmarking and comparative studies. 

---
# TORSO: Template-Oriented Reasoning Towards General Tasks 

**Authors**: Minhyuk Kim, Seungyoon Lee, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.09448)  

**Abstract**: The approaches that guide Large Language Models (LLMs) to emulate human reasoning during response generation have emerged as an effective method for enabling them to solve complex problems in a step-by-step manner, thereby achieving superior performance. However, most existing approaches using few-shot prompts to generate responses heavily depend on the provided examples, limiting the utilization of the model's inherent reasoning capabilities. Moreover, constructing task-specific few-shot prompts is often costly and may lead to inconsistencies across different tasks. In this work, we introduce Template-Oriented Reasoning (TORSO), which elicits the model to utilize internal reasoning abilities to generate proper responses across various tasks without the need for manually crafted few-shot examples. Our experimental results demonstrate that TORSO achieves strong performance on diverse LLMs benchmarks with reasonable rationales. 

---
# Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs 

**Authors**: Vaibhav Chaudhary, Neha Soni, Narotam Singh, Amita Kapoor  

**Link**: [PDF](https://arxiv.org/pdf/2509.09272)  

**Abstract**: Knowledge graphs, a powerful tool for structuring information through relational triplets, have recently become the new front-runner in enhancing question-answering systems. While traditional Retrieval Augmented Generation (RAG) approaches are proficient in fact-based and local context-based extraction from concise texts, they encounter limitations when addressing the thematic and holistic understanding of complex, extensive texts, requiring a deeper analysis of both text and context. This paper presents a comprehensive technical comparative study of three different methodologies for constructing knowledge graph triplets and integrating them with Large Language Models (LLMs) for question answering: spaCy, Stanford CoreNLP-OpenIE, and GraphRAG, all leveraging open source technologies. We evaluate the effectiveness, feasibility, and adaptability of these methods by analyzing their capabilities, state of development, and their impact on the performance of LLM-based question answering. Experimental results indicate that while OpenIE provides the most comprehensive coverage of triplets, GraphRAG demonstrates superior reasoning abilities among the three. We conclude with a discussion on the strengths and limitations of each method and provide insights into future directions for improving knowledge graph-based question answering. 

---
# Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search 

**Authors**: Shuocheng Li, Yihao Liu, Silin Du, Wenxuan Zeng, Zhe Xu, Mengyu Zhou, Yeye He, Haoyu Dong, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09245)  

**Abstract**: Large language models (LLMs) have shown great promise in automating data science workflows, but existing models still struggle with multi-step reasoning and tool use, which limits their effectiveness on complex data analysis tasks. To address this, we propose a scalable pipeline that extracts high-quality, tool-based data analysis tasks and their executable multi-step solutions from real-world Jupyter notebooks and associated data files. Using this pipeline, we introduce NbQA, a large-scale dataset of standardized task-solution pairs that reflect authentic tool-use patterns in practical data science scenarios. To further enhance multi-step reasoning, we present Jupiter, a framework that formulates data analysis as a search problem and applies Monte Carlo Tree Search (MCTS) to generate diverse solution trajectories for value model learning. During inference, Jupiter combines the value model and node visit counts to efficiently collect executable multi-step plans with minimal search steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench, respectively-matching or surpassing GPT-4o and advanced agent frameworks. Further evaluations demonstrate improved generalization and stronger tool-use reasoning across diverse multi-step reasoning tasks. 

---
# Global Constraint LLM Agents for Text-to-Model Translation 

**Authors**: Junyang Cai, Serdar Kadioglu, Bistra Dilkina  

**Link**: [PDF](https://arxiv.org/pdf/2509.08970)  

**Abstract**: Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce a framework that addresses this challenge with an agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement. 

---
# Automated Unity Game Template Generation from GDDs via NLP and Multi-Modal LLMs 

**Authors**: Amna Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2509.08847)  

**Abstract**: This paper presents a novel framework for automated game template generation by transforming Game Design Documents (GDDs) into functional Unity game prototypes using Natural Language Processing (NLP) and multi-modal Large Language Models (LLMs). We introduce an end-to-end system that parses GDDs, extracts structured game specifications, and synthesizes Unity-compatible C# code that implements the core mechanics, systems, and architecture defined in the design documentation. Our approach combines a fine-tuned LLaMA-3 model specialized for Unity code generation with a custom Unity integration package that streamlines the implementation process. Evaluation results demonstrate significant improvements over baseline models, with our fine-tuned model achieving superior performance (4.8/5.0 average score) compared to state-of-the-art LLMs across compilation success, GDD adherence, best practices adoption, and code modularity metrics. The generated templates demonstrate high adherence to GDD specifications across multiple game genres. Our system effectively addresses critical gaps in AI-assisted game development, positioning LLMs as valuable tools in streamlining the transition from game design to implementation. 

---
# LoCoBench: A Benchmark for Long-Context Large Language Models in Complex Software Engineering 

**Authors**: Jielin Qiu, Zuxin Liu, Zhiwei Liu, Rithesh Murthy, Jianguo Zhang, Haolin Chen, Shiyu Wang, Ming Zhu, Liangwei Yang, Juntao Tan, Zhepeng Cen, Cheng Qian, Shelby Heinecke, Weiran Yao, Silvio Savarese, Caiming Xiong, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09614)  

**Abstract**: The emergence of long-context language models with context windows extending to millions of tokens has created new opportunities for sophisticated code understanding and software development evaluation. We propose LoCoBench, a comprehensive benchmark specifically designed to evaluate long-context LLMs in realistic, complex software development scenarios. Unlike existing code evaluation benchmarks that focus on single-function completion or short-context tasks, LoCoBench addresses the critical evaluation gap for long-context capabilities that require understanding entire codebases, reasoning across multiple files, and maintaining architectural consistency across large-scale software systems. Our benchmark provides 8,000 evaluation scenarios systematically generated across 10 programming languages, with context lengths spanning 10K to 1M tokens, a 100x variation that enables precise assessment of long-context performance degradation in realistic software development settings. LoCoBench introduces 8 task categories that capture essential long-context capabilities: architectural understanding, cross-file refactoring, multi-session development, bug investigation, feature implementation, code comprehension, integration testing, and security analysis. Through a 5-phase pipeline, we create diverse, high-quality scenarios that challenge LLMs to reason about complex codebases at unprecedented scale. We introduce a comprehensive evaluation framework with 17 metrics across 4 dimensions, including 8 new evaluation metrics, combined in a LoCoBench Score (LCBS). Our evaluation of state-of-the-art long-context models reveals substantial performance gaps, demonstrating that long-context understanding in complex software development represents a significant unsolved challenge that demands more attention. LoCoBench is released at: this https URL. 

---
# The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs 

**Authors**: Akshit Sinha, Arvindh Arun, Shashwat Goel, Steffen Staab, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2509.09677)  

**Abstract**: Does continued scaling of large language models (LLMs) yield diminishing returns? Real-world value often stems from the length of task an agent can complete. We start this work by observing the simple but counterintuitive fact that marginal gains in single-step accuracy can compound into exponential improvements in the length of a task a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. We propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. We find that larger models can correctly execute significantly more turns even when small models have 100\% single-turn accuracy. We observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. In contrast, recent thinking models do not self-condition, and can also execute much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of task they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks. 

---
# Fluent but Unfeeling: The Emotional Blind Spots of Language Models 

**Authors**: Bangzhao Shu, Isha Joshi, Melissa Karnaze, Anh C. Pham, Ishita Kakkar, Sindhu Kothe, Arpine Hovasapian, Mai ElSherief  

**Link**: [PDF](https://arxiv.org/pdf/2509.09593)  

**Abstract**: The versatility of Large Language Models (LLMs) in natural language understanding has made them increasingly popular in mental health research. While many studies explore LLMs' capabilities in emotion recognition, a critical gap remains in evaluating whether LLMs align with human emotions at a fine-grained level. Existing research typically focuses on classifying emotions into predefined, limited categories, overlooking more nuanced expressions. To address this gap, we introduce EXPRESS, a benchmark dataset curated from Reddit communities featuring 251 fine-grained, self-disclosed emotion labels. Our comprehensive evaluation framework examines predicted emotion terms and decomposes them into eight basic emotions using established emotion theories, enabling a fine-grained comparison. Systematic testing of prevalent LLMs under various prompt settings reveals that accurately predicting emotions that align with human self-disclosed emotions remains challenging. Qualitative analysis further shows that while certain LLMs generate emotion terms consistent with established emotion theories and definitions, they sometimes fail to capture contextual cues as effectively as human self-disclosures. These findings highlight the limitations of LLMs in fine-grained emotion alignment and offer insights for future research aimed at enhancing their contextual understanding. 

---
# Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users 

**Authors**: Haowei Yang, Yushang Zhao, Sitao Min, Bo Su, Chao Yao, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09066)  

**Abstract**: The cold-start user issue further compromises the effectiveness of recommender systems in limiting access to the historical behavioral information. It is an effective pipeline to optimize instructional prompts on a few-shot large language model (LLM) used in recommender tasks. We introduce a context-conditioned prompt formulation method P(u,\ Ds)\ \rightarrow\ R\widehat, where u is a cold-start user profile, Ds is a curated support set, and R\widehat is the predicted ranked list of items. Based on systematic experimentation with transformer-based autoregressive LLMs (BioGPT, LLaMA-2, GPT-4), we provide empirical evidence that optimal exemplar injection and instruction structuring can significantly improve the precision@k and NDCG scores of such models in low-data settings. The pipeline uses token-level alignments and embedding space regularization with a greater semantic fidelity. Our findings not only show that timely composition is not merely syntactic but also functional as it is in direct control of attention scales and decoder conduct through inference. This paper shows that prompt-based adaptation may be considered one of the ways to address cold-start recommendation issues in LLM-based pipelines. 

---
# LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations 

**Authors**: Harry Mayne, Ryan Othniel Kearns, Yushi Yang, Andrew M. Bean, Eoin Delaney, Chris Russell, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2509.09396)  

**Abstract**: To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at this https URL. 

---
# Can Multimodal LLMs See Materials Clearly? A Multimodal Benchmark on Materials Characterization 

**Authors**: Zhengzhao Lai, Youbin Zheng, Zhenyang Cai, Haonan Lyu, Jinpu Yang, Hongqing Liang, Yan Hu, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09307)  

**Abstract**: Materials characterization is fundamental to acquiring materials information, revealing the processing-microstructure-property relationships that guide material design and optimization. While multimodal large language models (MLLMs) have recently shown promise in generative and predictive tasks within materials science, their capacity to understand real-world characterization imaging data remains underexplored. To bridge this gap, we present MatCha, the first benchmark for materials characterization image understanding, comprising 1,500 questions that demand expert-level domain expertise. MatCha encompasses four key stages of materials research comprising 21 distinct tasks, each designed to reflect authentic challenges faced by materials scientists. Our evaluation of state-of-the-art MLLMs on MatCha reveals a significant performance gap compared to human experts. These models exhibit degradation when addressing questions requiring higher-level expertise and sophisticated visual perception. Simple few-shot and chain-of-thought prompting struggle to alleviate these limitations. These findings highlight that existing MLLMs still exhibit limited adaptability to real-world materials characterization scenarios. We hope MatCha will facilitate future research in areas such as new material discovery and autonomous scientific agents. MatCha is available at this https URL. 

---
# ENSI: Efficient Non-Interactive Secure Inference for Large Language Models 

**Authors**: Zhiyu He, Maojiang Wang, Xinwen Gao, Yuchuan Luo, Lin Liu, Shaojing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09424)  

**Abstract**: Secure inference enables privacy-preserving machine learning by leveraging cryptographic protocols that support computations on sensitive user data without exposing it. However, integrating cryptographic protocols with large language models (LLMs) presents significant challenges, as the inherent complexity of these protocols, together with LLMs' massive parameter scale and sophisticated architectures, severely limits practical usability. In this work, we propose ENSI, a novel non-interactive secure inference framework for LLMs, based on the principle of co-designing the cryptographic protocols and LLM architecture. ENSI employs an optimized encoding strategy that seamlessly integrates CKKS scheme with a lightweight LLM variant, BitNet, significantly reducing the computational complexity of encrypted matrix multiplications. In response to the prohibitive computational demands of softmax under homomorphic encryption (HE), we pioneer the integration of the sigmoid attention mechanism with HE as a seamless, retraining-free alternative. Furthermore, by embedding the Bootstrapping operation within the RMSNorm process, we efficiently refresh ciphertexts while markedly decreasing the frequency of costly bootstrapping invocations. Experimental evaluations demonstrate that ENSI achieves approximately an 8x acceleration in matrix multiplications and a 2.6x speedup in softmax inference on CPU compared to state-of-the-art method, with the proportion of bootstrapping is reduced to just 1%. 

---
# On Integrating Large Language Models and Scenario-Based Programming for Improving Software Reliability 

**Authors**: Ayelet Berzack, Guy Katz  

**Link**: [PDF](https://arxiv.org/pdf/2509.09194)  

**Abstract**: Large Language Models (LLMs) are fast becoming indispensable tools for software developers, assisting or even partnering with them in crafting complex programs. The advantages are evident -- LLMs can significantly reduce development time, generate well-organized and comprehensible code, and occasionally suggest innovative ideas that developers might not conceive on their own. However, despite their strengths, LLMs will often introduce significant errors and present incorrect code with persuasive confidence, potentially misleading developers into accepting flawed solutions.
In order to bring LLMs into the software development cycle in a more reliable manner, we propose a methodology for combining them with ``traditional'' software engineering techniques in a structured way, with the goal of streamlining the development process, reducing errors, and enabling users to verify crucial program properties with increased confidence. Specifically, we focus on the Scenario-Based Programming (SBP) paradigm -- an event-driven, scenario-based approach for software engineering -- to allow human developers to pour their expert knowledge into the LLM, as well as to inspect and verify its outputs.
To evaluate our methodology, we conducted a significant case study, and used it to design and implement the Connect4 game. By combining LLMs and SBP we were able to create a highly-capable agent, which could defeat various strong existing agents. Further, in some cases, we were able to formally verify the correctness of our agent. Finally, our experience reveals interesting insights regarding the ease-of-use of our proposed approach. The full code of our case-study will be made publicly available with the final version of this paper. 

---
# MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization 

**Authors**: Mohammed Tiouti, Mohamed Bal-Ghaoui  

**Link**: [PDF](https://arxiv.org/pdf/2509.09387)  

**Abstract**: Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines. 

---
# DP-FedLoRA: Privacy-Enhanced Federated Fine-Tuning for On-Device Large Language Models 

**Authors**: Honghui Xu, Shiva Shrestha, Wei Chen, Zhiyuan Li, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.09097)  

**Abstract**: As on-device large language model (LLM) systems become increasingly prevalent, federated fine-tuning enables advanced language understanding and generation directly on edge devices; however, it also involves processing sensitive, user-specific data, raising significant privacy concerns within the federated learning framework. To address these challenges, we propose DP-FedLoRA, a privacy-enhanced federated fine-tuning framework that integrates LoRA-based adaptation with differential privacy in a communication-efficient setting. Each client locally clips and perturbs its LoRA matrices using Gaussian noise to satisfy ($\epsilon$, $\delta$)-differential privacy. We further provide a theoretical analysis demonstrating the unbiased nature of the updates and deriving bounds on the variance introduced by noise, offering practical guidance for privacy-budget calibration. Experimental results across mainstream benchmarks show that DP-FedLoRA delivers competitive performance while offering strong privacy guarantees, paving the way for scalable and privacy-preserving LLM deployment in on-device environments. 

---
# Character-Level Perturbations Disrupt LLM Watermarks 

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09112)  

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.
To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms. 

---
# Improving LLM Safety and Helpfulness using SFT and DPO: A Study on OPT-350M 

**Authors**: Piyush Pant  

**Link**: [PDF](https://arxiv.org/pdf/2509.09055)  

**Abstract**: This research investigates the effectiveness of alignment techniques, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and a combined SFT+DPO approach on improving the safety and helpfulness of the OPT-350M language model. Utilizing the Anthropic Helpful-Harmless RLHF dataset, we train and evaluate four models: the base OPT350M, an SFT model, a DPO model, and a model trained with both SFT and DPO. We introduce three key evaluation metrics: Harmlessness Rate (HmR), Helpfulness Rate (HpR), and a Combined Alignment Score (CAS), all derived from reward model outputs. The results show that while SFT outperforms DPO, The combined SFT+DPO model outperforms all others across all metrics, demonstrating the complementary nature of these techniques. Our findings also highlight challenges posed by noisy data, limited GPU resources, and training constraints. This study offers a comprehensive view of how fine-tuning strategies affect model alignment and provides a foundation for more robust alignment pipelines in future work. 

---
# Towards Confidential and Efficient LLM Inference with Dual Privacy Protection 

**Authors**: Honglan Yu, Yibin Wang, Feifei Dai, Dong Liu, Haihui Fan, Xiaoyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09091)  

**Abstract**: CPU-based trusted execution environments (TEEs) and differential privacy (DP) have gained wide applications for private inference. Due to high inference latency in TEEs, researchers use partition-based approaches that offload linear model components to GPUs. However, dense nonlinear layers of large language models (LLMs) result in significant communication overhead between TEEs and GPUs. DP-based approaches apply random noise to protect data privacy, but this compromises LLM performance and semantic understanding. To overcome the above drawbacks, this paper proposes CMIF, a Confidential and efficient Model Inference Framework. CMIF confidentially deploys the embedding layer in the client-side TEE and subsequent layers on GPU servers. Meanwhile, it optimizes the Report-Noisy-Max mechanism to protect sensitive inputs with a slight decrease in model performance. Extensive experiments on Llama-series models demonstrate that CMIF reduces additional inference overhead in TEEs while preserving user data privacy. 

---
# Stated Preference for Interaction and Continued Engagement (SPICE): Evaluating an LLM's Willingness to Re-engage in Conversation 

**Authors**: Thomas Manuel Rost, Martina Figlia, Bernd Wallraff  

**Link**: [PDF](https://arxiv.org/pdf/2509.09043)  

**Abstract**: We introduce and evaluate Stated Preference for Interaction and Continued Engagement (SPICE), a simple diagnostic signal elicited by asking a Large Language Model a YES or NO question about its willingness to re-engage with a user's behavior after reviewing a short transcript. In a study using a 3-tone (friendly, unclear, abusive) by 10-interaction stimulus set, we tested four open-weight chat models across four framing conditions, resulting in 480 trials. Our findings show that SPICE sharply discriminates by user tone. Friendly interactions yielded a near-unanimous preference to continue (97.5% YES), while abusive interactions yielded a strong preference to discontinue (17.9% YES), with unclear interactions falling in between (60.4% YES). This core association remains decisive under multiple dependence-aware statistical tests, including Rao-Scott adjustment and cluster permutation tests. Furthermore, we demonstrate that SPICE provides a distinct signal from abuse classification. In trials where a model failed to identify abuse, it still overwhelmingly stated a preference not to continue the interaction (81% of the time). An exploratory analysis also reveals a significant interaction effect: a preamble describing the study context significantly impacts SPICE under ambiguity, but only when transcripts are presented as a single block of text rather than a multi-turn chat. The results validate SPICE as a robust, low-overhead, and reproducible tool for auditing model dispositions, complementing existing metrics by offering a direct, relational signal of a model's state. All stimuli, code, and analysis scripts are released to support replication. 

---
# All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens 

**Authors**: Siddarth Mamidanna, Daking Rai, Ziyu Yao, Yilun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.09650)  

**Abstract**: Large language models (LLMs) demonstrate proficiency across numerous computational tasks, yet their inner workings remain unclear. In theory, the combination of causal self-attention and multilayer perceptron layers allows every token to access and compute information based on all preceding tokens. In practice, to what extent are such operations present? In this paper, on mental math tasks (i.e., direct math calculation via next-token prediction without explicit reasoning), we investigate this question in three steps: inhibiting input-specific token computations in the initial layers, restricting the routes of information transfer across token positions in the next few layers, and forcing all computation to happen at the last token in the remaining layers. With two proposed techniques, Context-Aware Mean Ablation (CAMA) and Attention-Based Peeking (ABP), we identify an All-for-One subgraph (AF1) with high accuracy on a wide variety of mental math tasks, where meaningful computation occurs very late (in terms of layer depth) and only at the last token, which receives information of other tokens in few specific middle layers. Experiments on a variety of models and arithmetic expressions show that this subgraph is sufficient and necessary for high model performance, transfers across different models, and works on a variety of input styles. Ablations on different CAMA and ABP alternatives reveal their unique advantages over other methods, which may be of independent interest. 

---
# Steering MoE LLMs via Expert (De)Activation 

**Authors**: Mohsen Fayyaz, Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Ryan Rossi, Trung Bui, Hinrich Schütze, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.09660)  

**Abstract**: Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts. 

---
# LAVA: Language Model Assisted Verbal Autopsy for Cause-of-Death Determination 

**Authors**: Yiqun T. Chen, Tyler H. McCormick, Li Liu, Abhirup Datta  

**Link**: [PDF](https://arxiv.org/pdf/2509.09602)  

**Abstract**: Verbal autopsy (VA) is a critical tool for estimating causes of death in resource-limited settings where medical certification is unavailable. This study presents LA-VA, a proof-of-concept pipeline that combines Large Language Models (LLMs) with traditional algorithmic approaches and embedding-based classification for improved cause-of-death prediction. Using the Population Health Metrics Research Consortium (PHMRC) dataset across three age categories (Adult: 7,580; Child: 1,960; Neonate: 2,438), we evaluate multiple approaches: GPT-5 predictions, LCVA baseline, text embeddings, and meta-learner ensembles. Our results demonstrate that GPT-5 achieves the highest individual performance with average test site accuracies of 48.6% (Adult), 50.5% (Child), and 53.5% (Neonate), outperforming traditional statistical machine learning baselines by 5-10%. Our findings suggest that simple off-the-shelf LLM-assisted approaches could substantially improve verbal autopsy accuracy, with important implications for global health surveillance in low-resource settings. 

---
# GrACE: A Generative Approach to Better Confidence Elicitation in Large Language Models 

**Authors**: Zhaohan Zhang, Ziquan Liu, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2509.09438)  

**Abstract**: Assessing the reliability of Large Language Models (LLMs) by confidence elicitation is a prominent approach to AI safety in high-stakes applications, such as healthcare and finance. Existing methods either require expensive computational overhead or suffer from poor calibration, making them impractical and unreliable for real-world deployment. In this work, we propose GrACE, a Generative Approach to Confidence Elicitation that enables scalable and reliable confidence elicitation for LLMs. GrACE adopts a novel mechanism in which the model expresses confidence by the similarity between the last hidden state and the embedding of a special token appended to the vocabulary, in real-time. We fine-tune the model for calibrating the confidence with calibration targets associated with accuracy. Experiments with three LLMs and two benchmark datasets show that the confidence produced by GrACE achieves the best discriminative capacity and calibration on open-ended generation tasks, outperforming six competing methods without resorting to additional sampling or an auxiliary model. Moreover, we propose two strategies for improving test-time scaling based on confidence induced by GrACE. Experimental results show that using GrACE not only improves the accuracy of the final decision but also significantly reduces the number of required samples in the test-time scaling scheme, indicating the potential of GrACE as a practical solution for deploying LLMs with scalable, reliable, and real-time confidence estimation. 

---
# From scratch to silver: Creating trustworthy training data for patent-SDG classification using Large Language Models 

**Authors**: Grazia Sveva Ascione, Nicolò Tamagnone  

**Link**: [PDF](https://arxiv.org/pdf/2509.09303)  

**Abstract**: Classifying patents by their relevance to the UN Sustainable Development Goals (SDGs) is crucial for tracking how innovation addresses global challenges. However, the absence of a large, labeled dataset limits the use of supervised learning. Existing methods, such as keyword searches, transfer learning, and citation-based heuristics, lack scalability and generalizability. This paper frames patent-to-SDG classification as a weak supervision problem, using citations from patents to SDG-tagged scientific publications (NPL citations) as a noisy initial signal. To address its sparsity and noise, we develop a composite labeling function (LF) that uses large language models (LLMs) to extract structured concepts, namely functions, solutions, and applications, from patents and SDG papers based on a patent ontology. Cross-domain similarity scores are computed and combined using a rank-based retrieval approach. The LF is calibrated via a custom positive-only loss that aligns with known NPL-SDG links without penalizing discovery of new SDG associations. The result is a silver-standard, soft multi-label dataset mapping patents to SDGs, enabling the training of effective multi-label regression models. We validate our approach through two complementary strategies: (1) internal validation against held-out NPL-based labels, where our method outperforms several baselines including transformer-based models, and zero-shot LLM; and (2) external validation using network modularity in patent citation, co-inventor, and co-applicant graphs, where our labels reveal greater thematic, cognitive, and organizational coherence than traditional technological classifications. These results show that weak supervision and semantic alignment can enhance SDG classification at scale. 

---
# DeMeVa at LeWiDi-2025: Modeling Perspectives with In-Context Learning and Label Distribution Learning 

**Authors**: Daniil Ignatev, Nan Li, Hugh Mee Wong, Anh Dang, Shane Kaszefski Yaschuk  

**Link**: [PDF](https://arxiv.org/pdf/2509.09524)  

**Abstract**: This system paper presents the DeMeVa team's approaches to the third edition of the Learning with Disagreements shared task (LeWiDi 2025; Leonardelli et al., 2025). We explore two directions: in-context learning (ICL) with large language models, where we compare example sampling strategies; and label distribution learning (LDL) methods with RoBERTa (Liu et al., 2019b), where we evaluate several fine-tuning methods. Our contributions are twofold: (1) we show that ICL can effectively predict annotator-specific annotations (perspectivist annotations), and that aggregating these predictions into soft labels yields competitive performance; and (2) we argue that LDL methods are promising for soft label predictions and merit further exploration by the perspectivist community. 

---
# Agentic LLMs for Question Answering over Tabular Data 

**Authors**: Rishit Tyagi, Mohit Gupta, Rahul Bouri  

**Link**: [PDF](https://arxiv.org/pdf/2509.09234)  

**Abstract**: Question Answering over Tabular Data (Table QA) presents unique challenges due to the diverse structure, size, and data types of real-world tables. The SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale, domain-diverse datasets to evaluate the ability of models to accurately answer structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a multi-stage pipeline involving example selection, SQL query generation, answer extraction, verification, and iterative refinement. Experiments demonstrate the effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and 71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\% and 27\% respectively. This paper details our methodology, experimental results, and alternative approaches, providing insights into the strengths and limitations of LLM-driven Table QA. 

---
# Reading Between the Lines: Classifying Resume Seniority with Large Language Models 

**Authors**: Matan Cohen, Shira Shani, Eden Menahem, Yehudit Aperstein, Alexander Apartsin  

**Link**: [PDF](https://arxiv.org/pdf/2509.09229)  

**Abstract**: Accurately assessing candidate seniority from resumes is a critical yet challenging task, complicated by the prevalence of overstated experience and ambiguous self-presentation. In this study, we investigate the effectiveness of large language models (LLMs), including fine-tuned BERT architectures, for automating seniority classification in resumes. To rigorously evaluate model performance, we introduce a hybrid dataset comprising both real-world resumes and synthetically generated hard examples designed to simulate exaggerated qualifications and understated seniority. Using the dataset, we evaluate the performance of Large Language Models in detecting subtle linguistic cues associated with seniority inflation and implicit expertise. Our findings highlight promising directions for enhancing AI-driven candidate evaluation systems and mitigating bias introduced by self-promotional language. The dataset is available for the research community at this https URL 

---
# MR-UIE: Multi-Perspective Reasoning with Reinforcement Learning for Universal Information Extraction 

**Authors**: Zhongqiu Li, Shiquan Wang, Ruiyu Fang, Mengjiao Bao, Zhenhe Wu, Shuangyong Song, Yongxiang Li, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.09082)  

**Abstract**: Large language models (LLMs) demonstrate robust capabilities across diverse research domains. However, their performance in universal information extraction (UIE) remains insufficient, especially when tackling structured output scenarios that involve complex schema descriptions and require multi-step reasoning. While existing approaches enhance the performance of LLMs through in-context learning and instruction tuning, significant limitations nonetheless persist. To enhance the model's generalization ability, we propose integrating reinforcement learning (RL) with multi-perspective reasoning for information extraction (IE) tasks. Our work transitions LLMs from passive extractors to active reasoners, enabling them to understand not only what to extract but also how to reason. Experiments conducted on multiple IE benchmarks demonstrate that MR-UIE consistently elevates extraction accuracy across domains and surpasses state-of-the-art methods on several datasets. Furthermore, incorporating multi-perspective reasoning into RL notably enhances generalization in complex IE tasks, underscoring the critical role of reasoning in challenging scenarios. 

---
# Documents Are People and Words Are Items: A Psychometric Approach to Textual Data with Contextual Embeddings 

**Authors**: Jinsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.08920)  

**Abstract**: This research introduces a novel psychometric method for analyzing textual data using large language models. By leveraging contextual embeddings to create contextual scores, we transform textual data into response data suitable for psychometric analysis. Treating documents as individuals and words as items, this approach provides a natural psychometric interpretation under the assumption that certain keywords, whose contextual meanings vary significantly across documents, can effectively differentiate documents within a corpus. The modeling process comprises two stages: obtaining contextual scores and performing psychometric analysis. In the first stage, we utilize natural language processing techniques and encoder based transformer models to identify common keywords and generate contextual scores. In the second stage, we employ various types of factor analysis, including exploratory and bifactor models, to extract and define latent factors, determine factor correlations, and identify the most significant words associated with each factor. Applied to the Wiki STEM corpus, our experimental results demonstrate the method's potential to uncover latent knowledge dimensions and patterns within textual data. This approach not only enhances the psychometric analysis of textual data but also holds promise for applications in fields rich in textual information, such as education, psychology, and law. 

---
# BRoverbs -- Measuring how much LLMs understand Portuguese proverbs 

**Authors**: Thales Sales Almeida, Giovana Kerche Bonás, João Guilherme Alves Santos  

**Link**: [PDF](https://arxiv.org/pdf/2509.08960)  

**Abstract**: Large Language Models (LLMs) exhibit significant performance variations depending on the linguistic and cultural context in which they are applied. This disparity signals the necessity of mature evaluation frameworks that can assess their capabilities in specific regional settings. In the case of Portuguese, existing evaluations remain limited, often relying on translated datasets that may not fully capture linguistic nuances or cultural references. Meanwhile, native Portuguese-language datasets predominantly focus on structured national exams or sentiment analysis of social media interactions, leaving gaps in evaluating broader linguistic understanding. To address this limitation, we introduce BRoverbs, a dataset specifically designed to assess LLM performance through Brazilian proverbs. Proverbs serve as a rich linguistic resource, encapsulating cultural wisdom, figurative expressions, and complex syntactic structures that challenge the model comprehension of regional expressions. BRoverbs aims to provide a new evaluation tool for Portuguese-language LLMs, contributing to advancing regionally informed benchmarking. The benchmark is available at this https URL. 

---
# Noise or Nuance: An Investigation Into Useful Information and Filtering For LLM Driven AKBC 

**Authors**: Alex Clay, Ernesto Jiménez-Ruiz, Pranava Madhyastha  

**Link**: [PDF](https://arxiv.org/pdf/2509.08903)  

**Abstract**: RAG and fine-tuning are prevalent strategies for improving the quality of LLM outputs. However, in constrained situations, such as that of the 2025 LM-KBC challenge, such techniques are restricted. In this work we investigate three facets of the triple completion task: generation, quality assurance, and LLM response parsing. Our work finds that in this constrained setting: additional information improves generation quality, LLMs can be effective at filtering poor quality triples, and the tradeoff between flexibility and consistency with LLM response parsing is setting dependent. 

---
# PerFairX: Is There a Balance Between Fairness and Personality in Large Language Model Recommendations? 

**Authors**: Chandan Kumar Sah  

**Link**: [PDF](https://arxiv.org/pdf/2509.08829)  

**Abstract**: The integration of Large Language Models (LLMs) into recommender systems has enabled zero-shot, personality-based personalization through prompt-based interactions, offering a new paradigm for user-centric recommendations. However, incorporating user personality traits via the OCEAN model highlights a critical tension between achieving psychological alignment and ensuring demographic fairness. To address this, we propose PerFairX, a unified evaluation framework designed to quantify the trade-offs between personalization and demographic equity in LLM-generated recommendations. Using neutral and personality-sensitive prompts across diverse user profiles, we benchmark two state-of-the-art LLMs, ChatGPT and DeepSeek, on movie (MovieLens 10M) and music (this http URL 360K) datasets. Our results reveal that personality-aware prompting significantly improves alignment with individual traits but can exacerbate fairness disparities across demographic groups. Specifically, DeepSeek achieves stronger psychological fit but exhibits higher sensitivity to prompt variations, while ChatGPT delivers stable yet less personalized outputs. PerFairX provides a principled benchmark to guide the development of LLM-based recommender systems that are both equitable and psychologically informed, contributing to the creation of inclusive, user-centric AI applications in continual learning contexts. 

---
# Constructing a Question-Answering Simulator through the Distillation of LLMs 

**Authors**: Haipeng Liu, Ting Long, Jing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09226)  

**Abstract**: The question-answering (QA) simulator is a model that mimics real student learning behaviors and predicts their correctness of their responses to questions. QA simulators enable educational recommender systems (ERS) to collect large amounts of training data without interacting with real students, thereby preventing harmful recommendations made by an undertrained ERS from undermining actual student learning. Given the QA history, there are two categories of solutions to predict the correctness, conducting the simulation: (1) LLM-free methods, which apply a traditional sequential model to transfer the QA history into a vector representation first, and make predictions based on the representation; (2) LLM-based methods, which leverage the domain knowledge and reasoning capability of LLM to enhence the prediction. LLM-free methods offer fast inference but generally yield suboptimal performance. In contrast, most LLM-based methods achieve better results, but at the cost of slower inference speed and higher GPU memory consumption. In this paper, we propose a method named LLM Distillation based Simulator (LDSim), which distills domain knowledge and reasoning capability from an LLM to better assist prediction, thereby improving simulation performance. Extensive experiments demonstrate that our LDSim achieves strong results on both the simulation task and the knowledge tracing (KT) task. Our code is publicly available at this https URL. 

---
