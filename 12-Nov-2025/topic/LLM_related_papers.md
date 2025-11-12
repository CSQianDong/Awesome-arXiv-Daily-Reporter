# AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress 

**Authors**: Zhiheng Xi, Chenyang Liao, Guanyu Li, Yajie Yang, Wenxiang Chen, Zhihao Zhang, Binghai Wang, Senjie Jin, Yuhao Zhou, Jian Guan, Wei Wu, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08325)  

**Abstract**: Despite rapid development, large language models (LLMs) still encounter challenges in multi-turn decision-making tasks (i.e., agent tasks) like web shopping and browser navigation, which require making a sequence of intelligent decisions based on environmental feedback. Previous work for LLM agents typically relies on elaborate prompt engineering or fine-tuning with expert trajectories to improve performance. In this work, we take a different perspective: we explore constructing process reward models (PRMs) to evaluate each decision and guide the agent's decision-making process. Unlike LLM reasoning, where each step is scored based on correctness, actions in agent tasks do not have a clear-cut correctness. Instead, they should be evaluated based on their proximity to the goal and the progress they have made. Building on this insight, we propose a re-defined PRM for agent tasks, named AgentPRM, to capture both the interdependence between sequential decisions and their contribution to the final goal. This enables better progress tracking and exploration-exploitation balance. To scalably obtain labeled data for training AgentPRM, we employ a Temporal Difference-based (TD-based) estimation method combined with Generalized Advantage Estimation (GAE), which proves more sample-efficient than prior methods. Extensive experiments across different agentic tasks show that AgentPRM is over $8\times$ more compute-efficient than baselines, and it demonstrates robust improvement when scaling up test-time compute. Moreover, we perform detailed analyses to show how our method works and offer more insights, e.g., applying AgentPRM to the reinforcement learning of LLM agents. 

---
# MARC: Multimodal and Multi-Task Agentic Retrieval-Augmented Generation for Cold-Start Recommender System 

**Authors**: Seung Hwan Cho, Yujin Yang, Danik Baeck, Minjoo Kim, Young-Min Kim, Heejung Lee, Sangjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.08181)  

**Abstract**: Recommender systems (RS) are currently being studied to mitigate limitations during cold-start conditions by leveraging modality information or introducing Agent concepts based on the exceptional reasoning capabilities of Large Language Models (LLMs). Meanwhile, food and beverage recommender systems have traditionally used knowledge graph and ontology concepts due to the domain's unique data attributes and relationship characteristics. On this background, we propose MARC, a multimodal and multi-task cocktail recommender system based on Agentic Retrieval-Augmented Generation (RAG) utilizing graph database under cold-start conditions. The proposed system generates high-quality, contextually appropriate answers through two core processes: a task recognition router and a reflection process. The graph database was constructed by processing cocktail data from Kaggle, and its effectiveness was evaluated using 200 manually crafted questions. The evaluation used both LLM-as-a-judge and human evaluation to demonstrate that answers generated via the graph database outperformed those from a simple vector database in terms of quality. The code is available at this https URL 

---
# From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization 

**Authors**: Peiyu Hu, Wayne Lu, Jia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08006)  

**Abstract**: Cross-domain recommendation (CDR) is crucial for improving recommendation accuracy and generalization, yet traditional methods are often hindered by the reliance on shared user/item IDs, which are unavailable in most real-world scenarios. Consequently, many efforts have focused on learning disentangled representations through multi-domain joint training to bridge the domain gaps. Recent Large Language Model (LLM)-based approaches show promise, they still face critical challenges, including: (1) the \textbf{item ID tokenization dilemma}, which leads to vocabulary explosion and fails to capture high-order collaborative knowledge; and (2) \textbf{insufficient domain-specific modeling} for the complex evolution of user interests and item semantics. To address these limitations, we propose \textbf{GenCDR}, a novel \textbf{Gen}erative \textbf{C}ross-\textbf{D}omain \textbf{R}ecommendation framework. GenCDR first employs a \textbf{Domain-adaptive Tokenization} module, which generates disentangled semantic IDs for items by dynamically routing between a universal encoder and domain-specific adapters. Symmetrically, a \textbf{Cross-domain Autoregressive Recommendation} module models user preferences by fusing universal and domain-specific interests. Finally, a \textbf{Domain-aware Prefix-tree} enables efficient and accurate generation. Extensive experiments on multiple real-world datasets demonstrate that GenCDR significantly outperforms state-of-the-art baselines. Our code is available in the supplementary materials. 

---
# DiffuGR: Generative Document Retrieval with Diffusion Language Models 

**Authors**: Xinpeng Zhao, Yukun Zhao, Zhenyang Li, Mengqi Zhang, Jun Feng, Ran Chen, Ying Zhou, Zhumin Chen, Shuaiqiang Wang, Zhaochun Ren, Dawei Yin, Xin Xin  

**Link**: [PDF](https://arxiv.org/pdf/2511.08150)  

**Abstract**: Generative retrieval (GR) re-frames document retrieval as a sequence-based document identifier (DocID) generation task, memorizing documents with model parameters and enabling end-to-end retrieval without explicit indexing. Existing GR methods are based on auto-regressive generative models, i.e., the token generation is performed from left to right. However, such auto-regressive methods suffer from: (1) mismatch between DocID generation and natural language generation, e.g., an incorrect DocID token generated in early left steps would lead to totally erroneous retrieval; and (2) failure to balance the trade-off between retrieval efficiency and accuracy dynamically, which is crucial for practical applications. To address these limitations, we propose generative document retrieval with diffusion language models, dubbed DiffuGR. It models DocID generation as a discrete diffusion process: during training, DocIDs are corrupted through a stochastic masking process, and a diffusion language model is learned to recover them under a retrieval-aware objective. For inference, DiffuGR attempts to generate DocID tokens in parallel and refines them through a controllable number of denoising steps. In contrast to conventional left-to-right auto-regressive decoding, DiffuGR provides a novel mechanism to first generate more confident DocID tokens and refine the generation through diffusion-based denoising. Moreover, DiffuGR also offers explicit runtime control over the qualitylatency tradeoff. Extensive experiments on benchmark retrieval datasets show that DiffuGR is competitive with strong auto-regressive generative retrievers, while offering flexible speed and accuracy tradeoffs through variable denoising budgets. Overall, our results indicate that non-autoregressive diffusion models are a practical and effective alternative for generative document retrieval. 

---
# Training Language Models to Explain Their Own Computations 

**Authors**: Belinda Z. Li, Zifan Carl Guo, Vincent Huang, Jacob Steinhardt, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2511.08579)  

**Abstract**: Can language models (LMs) learn to faithfully describe their internal computations? Are they better able to describe themselves than other models? We study the extent to which LMs' privileged access to their own internals can be leveraged to produce new techniques for explaining their behavior. Using existing interpretability techniques as a source of ground truth, we fine-tune LMs to generate natural language descriptions of (1) the information encoded by LM features, (2) the causal structure of LMs' internal activations, and (3) the influence of specific input tokens on LM outputs. When trained with only tens of thousands of example explanations, explainer models exhibit non-trivial generalization to new queries. This generalization appears partly attributable to explainer models' privileged access to their own internals: using a model to explain its own computations generally works better than using a *different* model to explain its computations (even if the other model is significantly more capable). Our results suggest not only that LMs can learn to reliably explain their internal computations, but that such explanations offer a scalable complement to existing interpretability methods. 

---
# AlphaResearch: Accelerating New Algorithm Discovery with Language Models 

**Authors**: Zhaojian Yu, Kaiyue Feng, Yilun Zhao, Shilin He, Xiao-Ping Zhang, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2511.08522)  

**Abstract**: Large language models have made significant progress in complex but easy-to-verify problems, yet they still struggle with discovering the unknown. In this paper, we present \textbf{AlphaResearch}, an autonomous research agent designed to discover new algorithms on open-ended problems. To synergize the feasibility and innovation of the discovery process, we construct a novel dual research environment by combining the execution-based verify and simulated real-world peer review environment. AlphaResearch discovers new algorithm by iteratively running the following steps: (1) propose new ideas (2) verify the ideas in the dual research environment (3) optimize the research proposals for better performance. To promote a transparent evaluation process, we construct \textbf{AlphaResearchComp}, a new evaluation benchmark that includes an eight open-ended algorithmic problems competition, with each problem carefully curated and verified through executable pipelines, objective metrics, and reproducibility checks. AlphaResearch gets a 2/8 win rate in head-to-head comparison with human researchers, demonstrate the possibility of accelerating algorithm discovery with LLMs. Notably, the algorithm discovered by AlphaResearch on the \emph{``packing circles''} problem achieves the best-of-known performance, surpassing the results of human researchers and strong baselines from recent work (e.g., AlphaEvolve). Additionally, we conduct a comprehensive analysis of the remaining challenges of the 6/8 failure cases, providing valuable insights for future research. 

---
# Bot Meets Shortcut: How Can LLMs Aid in Handling Unknown Invariance OOD Scenarios? 

**Authors**: Shiyan Zheng, Herun Wan, Minnan Luo, Junhang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08455)  

**Abstract**: While existing social bot detectors perform well on benchmarks, their robustness across diverse real-world scenarios remains limited due to unclear ground truth and varied misleading cues. In particular, the impact of shortcut learning, where models rely on spurious correlations instead of capturing causal task-relevant features, has received limited attention. To address this gap, we conduct an in-depth study to assess how detectors are influenced by potential shortcuts based on textual features, which are most susceptible to manipulation by social bots. We design a series of shortcut scenarios by constructing spurious associations between user labels and superficial textual cues to evaluate model robustness. Results show that shifts in irrelevant feature distributions significantly degrade social bot detector performance, with an average relative accuracy drop of 32\% in the baseline models. To tackle this challenge, we propose mitigation strategies based on large language models, leveraging counterfactual data augmentation. These methods mitigate the problem from data and model perspectives across three levels, including data distribution at both the individual user text and overall dataset levels, as well as the model's ability to extract causal information. Our strategies achieve an average relative performance improvement of 56\% under shortcut scenarios. 

---
# Investigating CoT Monitorability in Large Reasoning Models 

**Authors**: Shu Yang, Junchao Wu, Xilin Gou, Xuansheng Wu, Derek Wong, Ninhao Liu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08525)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable performance on complex tasks by engaging in extended reasoning before producing final answers. Beyond improving abilities, these detailed reasoning traces also create a new opportunity for AI safety, CoT Monitorability: monitoring potential model misbehavior, such as the use of shortcuts or sycophancy, through their chain-of-thought (CoT) during decision-making. However, two key fundamental challenges arise when attempting to build more effective monitors through CoT analysis. First, as prior research on CoT faithfulness has pointed out, models do not always truthfully represent their internal decision-making in the generated reasoning. Second, monitors themselves may be either overly sensitive or insufficiently sensitive, and can potentially be deceived by models' long, elaborate reasoning traces. In this paper, we present the first systematic investigation of the challenges and potential of CoT monitorability. Motivated by two fundamental challenges we mentioned before, we structure our study around two central perspectives: (i) verbalization: to what extent do LRMs faithfully verbalize the true factors guiding their decisions in the CoT, and (ii) monitor reliability: to what extent can misbehavior be reliably detected by a CoT-based monitor? Specifically, we provide empirical evidence and correlation analyses between verbalization quality, monitor reliability, and LLM performance across mathematical, scientific, and ethical domains. Then we further investigate how different CoT intervention methods, designed to improve reasoning efficiency or performance, will affect monitoring effectiveness. Finally, we propose MoME, a new paradigm in which LLMs monitor other models' misbehavior through their CoT and provide structured judgments along with supporting evidence. 

---
# Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models 

**Authors**: Tianyu Fu, Yichen You, Zekai Chen, Guohao Dai, Huazhong Yang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08577)  

**Abstract**: Improving reasoning capabilities of Large Language Models (LLMs), especially under parameter constraints, is crucial for real-world applications. Prior work proposes recurrent transformers, which allocate a fixed number of extra iterations per token to improve generation quality. After the first, standard forward pass, instead of verbalization, last-layer hidden states are fed back as inputs for additional iterations to refine token predictions. Yet we identify a latent overthinking phenomenon: easy token predictions that are already correct after the first pass are sometimes revised into errors in additional iterations. To address this, we propose Think-at-Hard (TaH), a dynamic latent thinking method that iterates deeper only at hard tokens. It employs a lightweight neural decider to trigger latent iterations only at tokens that are likely incorrect after the standard forward pass. During latent iterations, Low-Rank Adaptation (LoRA) modules shift the LLM objective from general next-token prediction to focused hard-token refinement. We further introduce a duo-causal attention mechanism that extends attention from the token sequence dimension to an additional iteration depth dimension. This enables cross-iteration information flow while maintaining full sequential parallelism. Experiments show that TaH boosts LLM reasoning performance across five challenging benchmarks while maintaining the same parameter count. Compared with baselines that iterate twice for all output tokens, TaH delivers 8.1-11.3% accuracy gains while exempting 94% of tokens from the second iteration. Against strong single-iteration Qwen3 models finetuned with the same data, it also delivers 4.0-5.0% accuracy gains. When allowing less than 3% additional parameters from LoRA and the iteration decider, the gains increase to 8.5-12.6% and 5.3-5.4%, respectively. Our code is available at this https URL. 

---
# PCRLLM: Proof-Carrying Reasoning with Large Language Models under Stepwise Logical Constraints 

**Authors**: Tangrui Li, Pei Wang, Hongzheng Wang Christian Hahm, Matteo Spatola, Justin Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08392)  

**Abstract**: Large Language Models (LLMs) often exhibit limited logical coherence, mapping premises to conclusions without adherence to explicit inference rules. We propose Proof-Carrying Reasoning with LLMs (PCRLLM), a framework that constrains reasoning to single-step inferences while preserving natural language formulations. Each output explicitly specifies premises, rules, and conclusions, thereby enabling verification against a target logic. This mechanism mitigates trustworthiness concerns by supporting chain-level validation even in black-box settings. Moreover, PCRLLM facilitates systematic multi-LLM collaboration, allowing intermediate steps to be compared and integrated under formal rules. Finally, we introduce a benchmark schema for generating large-scale step-level reasoning data, combining natural language expressiveness with formal rigor. 

---
# Moral Susceptibility and Robustness under Persona Role-Play in Large Language Models 

**Authors**: Davi Bastos Costa, Felippe Alves, Renato Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2511.08565)  

**Abstract**: Large language models (LLMs) increasingly operate in social contexts, motivating analysis of how they express and shift moral judgments. In this work, we investigate the moral response of LLMs to persona role-play, prompting a LLM to assume a specific character. Using the Moral Foundations Questionnaire (MFQ), we introduce a benchmark that quantifies two properties: moral susceptibility and moral robustness, defined from the variability of MFQ scores across and within personas, respectively. We find that, for moral robustness, model family accounts for most of the variance, while model size shows no systematic effect. The Claude family is, by a significant margin, the most robust, followed by Gemini and GPT-4 models, with other families exhibiting lower robustness. In contrast, moral susceptibility exhibits a mild family effect but a clear within-family size effect, with larger variants being more susceptible. Moreover, robustness and susceptibility are positively correlated, an association that is more pronounced at the family level. Additionally, we present moral foundation profiles for models without persona role-play and for personas averaged across models. Together, these analyses provide a systematic view of how persona conditioning shapes moral behavior in large language models. 

---
# Adaptive Multi-Agent Response Refinement in Conversational Systems 

**Authors**: Soyeong Jeong, Aparna Elangovan, Emine Yilmaz, Oleg Rokhlenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.08319)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable success in conversational systems by generating human-like responses. However, they can fall short, especially when required to account for personalization or specific knowledge. In real-life settings, it is impractical to rely on users to detect these errors and request a new response. One way to address this problem is to refine the response before returning it to the user. While existing approaches focus on refining responses within a single LLM, this method struggles to consider diverse aspects needed for effective conversations. In this work, we propose refining responses through a multi-agent framework, where each agent is assigned a specific role for each aspect. We focus on three key aspects crucial to conversational quality: factuality, personalization, and coherence. Each agent is responsible for reviewing and refining one of these aspects, and their feedback is then merged to improve the overall response. To enhance collaboration among them, we introduce a dynamic communication strategy. Instead of following a fixed sequence of agents, our approach adaptively selects and coordinates the most relevant agents based on the specific requirements of each query. We validate our framework on challenging conversational datasets, demonstrating that ours significantly outperforms relevant baselines, particularly in tasks involving knowledge or user's persona, or both. 

---
# Automatic Paper Reviewing with Heterogeneous Graph Reasoning over LLM-Simulated Reviewer-Author Debates 

**Authors**: Shuaimin Li, Liyang Fan, Yufang Lin, Zeyang Li, Xian Wei, Shiwen Ni, Hamid Alinejad-Rokny, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08317)  

**Abstract**: Existing paper review methods often rely on superficial manuscript features or directly on large language models (LLMs), which are prone to hallucinations, biased scoring, and limited reasoning capabilities. Moreover, these methods often fail to capture the complex argumentative reasoning and negotiation dynamics inherent in reviewer-author interactions. To address these limitations, we propose ReViewGraph (Reviewer-Author Debates Graph Reasoner), a novel framework that performs heterogeneous graph reasoning over LLM-simulated multi-round reviewer-author debates. In our approach, reviewer-author exchanges are simulated through LLM-based multi-agent collaboration. Diverse opinion relations (e.g., acceptance, rejection, clarification, and compromise) are then explicitly extracted and encoded as typed edges within a heterogeneous interaction graph. By applying graph neural networks to reason over these structured debate graphs, ReViewGraph captures fine-grained argumentative dynamics and enables more informed review decisions. Extensive experiments on three datasets demonstrate that ReViewGraph outperforms strong baselines with an average relative improvement of 15.73%, underscoring the value of modeling detailed reviewer-author debate structures. 

---
# ParliaBench: An Evaluation and Benchmarking Framework for LLM-Generated Parliamentary Speech 

**Authors**: Marios Koniaris, Argyro Tsipi, Panayiotis Tsanakas  

**Link**: [PDF](https://arxiv.org/pdf/2511.08247)  

**Abstract**: Parliamentary speech generation presents specific challenges for large language models beyond standard text generation tasks. Unlike general text generation, parliamentary speeches require not only linguistic quality but also political authenticity and ideological consistency. Current language models lack specialized training for parliamentary contexts, and existing evaluation methods focus on standard NLP metrics rather than political authenticity. To address this, we present ParliaBench, a benchmark for parliamentary speech generation. We constructed a dataset of speeches from UK Parliament to enable systematic model training. We introduce an evaluation framework combining computational metrics with LLM-as-a-judge assessments for measuring generation quality across three dimensions: linguistic quality, semantic coherence, and political authenticity. We propose two novel embedding-based metrics, Political Spectrum Alignment and Party Alignment, to quantify ideological positioning. We fine-tuned five large language models (LLMs), generated 28k speeches, and evaluated them using our framework, comparing baseline and fine-tuned models. Results show that fine-tuning produces statistically significant improvements across the majority of metrics and our novel metrics demonstrate strong discriminative power for political dimensions. 

---
# Hierarchical structure understanding in complex tables with VLLMs: a benchmark and experiments 

**Authors**: Luca Bindini, Simone Giovannini, Simone Marinai, Valeria Nardoni, Kimiya Noor Ali  

**Link**: [PDF](https://arxiv.org/pdf/2511.08298)  

**Abstract**: This work investigates the ability of Vision Large Language Models (VLLMs) to understand and interpret the structure of tables in scientific articles. Specifically, we explore whether VLLMs can infer the hierarchical structure of tables without additional processing. As a basis for our experiments we use the PubTables-1M dataset, a large-scale corpus of scientific tables. From this dataset, we extract a subset of tables that we introduce as Complex Hierarchical Tables (CHiTab): a benchmark collection of complex tables containing hierarchical headings. We adopt a series of prompt engineering strategies to probe the models' comprehension capabilities, experimenting with various prompt formats and writing styles. Multiple state-of-the-art open-weights VLLMs are evaluated on the benchmark first using their off-the-shelf versions and then fine-tuning some models on our task. We also measure the performance of humans to solve the task on a small set of tables comparing with performance of the evaluated VLLMs. The experiments support our intuition that generic VLLMs, not explicitly designed for understanding the structure of tables, can perform this task. This study provides insights into the potential and limitations of VLLMs to process complex tables and offers guidance for future work on integrating structured data understanding into general-purpose VLLMs. 

---
# Interaction Dynamics as a Reward Signal for LLMs 

**Authors**: Sian Gooding, Edward Grefenstette  

**Link**: [PDF](https://arxiv.org/pdf/2511.08394)  

**Abstract**: The alignment of Large Language Models (LLMs) for multi-turn conversations typically relies on reward signals derived from the content of the text. This approach, however, overlooks a rich, complementary source of signal: the dynamics of the interaction itself. This paper introduces TRACE (Trajectory-based Reward for Agent Collaboration Estimation), a novel reward signal derived from the geometric properties of a dialogue's embedding trajectory--a concept we term 'conversational geometry'. Our central finding is that a reward model trained only on these structural signals achieves a pairwise accuracy (68.20%) comparable to a powerful LLM baseline that analyzes the full transcript (70.04%). Furthermore, a hybrid model combining interaction dynamics with textual analysis achieves the highest performance (80.17%), demonstrating their complementary nature. This work provides strong evidence that for interactive settings, how an agent communicates is as powerful a predictor of success as what it says, offering a new, privacy-preserving framework that not only aligns agents but also serves as a diagnostic tool for understanding the distinct interaction patterns that drive successful collaboration. 

---
# Sentence-Anchored Gist Compression for Long-Context LLMs 

**Authors**: Dmitrii Tarasov, Elizaveta Goncharova, Kuznetsov Andrey  

**Link**: [PDF](https://arxiv.org/pdf/2511.08128)  

**Abstract**: This work investigates context compression for Large Language Models (LLMs) using learned compression tokens to reduce the memory and computational demands of processing long sequences. We demonstrate that pre-trained LLMs can be fine-tuned to compress their context by factors of 2x to 8x without significant performance degradation, as evaluated on both short-context and long-context benchmarks. Furthermore, in experiments on a 3-billion-parameter LLaMA model, our method achieves results on par with alternative compression techniques while attaining higher compression ratios. 

---
# Relation as a Prior: A Novel Paradigm for LLM-based Document-level Relation Extraction 

**Authors**: Qiankun Pi, Yepeng Sun, Jicang Lu, Qinlong Fan, Ningbo Huang, Shiyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08143)  

**Abstract**: Large Language Models (LLMs) have demonstrated their remarkable capabilities in document understanding. However, recent research reveals that LLMs still exhibit performance gaps in Document-level Relation Extraction (DocRE) as requiring fine-grained comprehension. The commonly adopted "extract entities then predict relations" paradigm in LLM-based methods leads to these gaps due to two main reasons: (1) Numerous unrelated entity pairs introduce noise and interfere with the relation prediction for truly related entity pairs. (2) Although LLMs have identified semantic associations between entities, relation labels beyond the predefined set are still treated as prediction errors. To address these challenges, we propose a novel Relation as a Prior (RelPrior) paradigm for LLM-based DocRE. For challenge (1), RelPrior utilizes binary relation as a prior to extract and determine whether two entities are correlated, thereby filtering out irrelevant entity pairs and reducing prediction noise. For challenge (2), RelPrior utilizes predefined relation as a prior to match entities for triples extraction instead of directly predicting relation. Thus, it avoids misjudgment caused by strict predefined relation labeling. Extensive experiments on two benchmarks demonstrate that RelPrior achieves state-of-the-art performance, surpassing existing LLM-based methods. 

---
# Still Not There: Can LLMs Outperform Smaller Task-Specific Seq2Seq Models on the Poetry-to-Prose Conversion Task? 

**Authors**: Kunal Kingkar Das, Manoj Balaji Jagadeeshan, Nallani Chakravartula Sahith, Jivnesh Sandhan, Pawan Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2511.08145)  

**Abstract**: Large Language Models (LLMs) are increasingly treated as universal, general-purpose solutions across NLP tasks, particularly in English. But does this assumption hold for low-resource, morphologically rich languages such as Sanskrit? We address this question by comparing instruction-tuned and in-context-prompted LLMs with smaller task-specific encoder-decoder models on the Sanskrit poetry-to-prose conversion task. This task is intrinsically challenging: Sanskrit verse exhibits free word order combined with rigid metrical constraints, and its conversion to canonical prose (anvaya) requires multi-step reasoning involving compound segmentation, dependency resolution, and syntactic linearisation. This makes it an ideal testbed to evaluate whether LLMs can surpass specialised models. For LLMs, we apply instruction fine-tuning on general-purpose models and design in-context learning templates grounded in Paninian grammar and classical commentary heuristics. For task-specific modelling, we fully fine-tune a ByT5-Sanskrit Seq2Seq model. Our experiments show that domain-specific fine-tuning of ByT5-Sanskrit significantly outperforms all instruction-driven LLM approaches. Human evaluation strongly corroborates this result, with scores exhibiting high correlation with Kendall's Tau scores. Additionally, our prompting strategies provide an alternative to fine-tuning when domain-specific verse corpora are unavailable, and the task-specific Seq2Seq model demonstrates robust generalisation on out-of-domain evaluations. 

---
# SPEAR-MM: Selective Parameter Evaluation and Restoration via Model Merging for Efficient Financial LLM Adaptation 

**Authors**: Berkcan Kapusuzoglu, Supriyo Chakraborty, Renkun Ni, Stephen Rawls, Sambit Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2511.08500)  

**Abstract**: Large language models (LLMs) adapted to financial domains often suffer from catastrophic forgetting of general reasoning capabilities essential for customer interactions and complex financial analysis. We introduce Selective Parameter Evaluation and Restoration via Model Merging (SPEAR-MM), a practical framework that preserves critical capabilities while enabling domain adaptation. Our method approximates layer-wise impact on external benchmarks through post-hoc analysis, then selectively freezes or restores transformer layers via spherical interpolation merging. Applied to LLaMA-3.1-8B for financial tasks, SPEAR-MM achieves 91.2% retention of general capabilities versus 69.7% for standard continual pretraining, while maintaining 94% of domain adaptation gains. The approach provides interpretable trade-off control and reduces computational costs by 90% crucial for resource-constrained financial institutions. 

---
# State of the Art in Text Classification for South Slavic Languages: Fine-Tuning or Prompting? 

**Authors**: Taja Kuzman Pungeršek, Peter Rupnik, Ivan Porupski, Vuk Dinić, Nikola Ljubešić  

**Link**: [PDF](https://arxiv.org/pdf/2511.07989)  

**Abstract**: Until recently, fine-tuned BERT-like models provided state-of-the-art performance on text classification tasks. With the rise of instruction-tuned decoder-only models, commonly known as large language models (LLMs), the field has increasingly moved toward zero-shot and few-shot prompting. However, the performance of LLMs on text classification, particularly on less-resourced languages, remains under-explored. In this paper, we evaluate the performance of current language models on text classification tasks across several South Slavic languages. We compare openly available fine-tuned BERT-like models with a selection of open-source and closed-source LLMs across three tasks in three domains: sentiment classification in parliamentary speeches, topic classification in news articles and parliamentary speeches, and genre identification in web texts. Our results show that LLMs demonstrate strong zero-shot performance, often matching or surpassing fine-tuned BERT-like models. Moreover, when used in a zero-shot setup, LLMs perform comparably in South Slavic languages and English. However, we also point out key drawbacks of LLMs, including less predictable outputs, significantly slower inference, and higher computational costs. Due to these limitations, fine-tuned BERT-like models remain a more practical choice for large-scale automatic text annotation. 

---
# NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation 

**Authors**: Maoqi Liu, Quan Fang, Yuhao Wu, Can Zhao, Yang Yang, Kaiquan Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07982)  

**Abstract**: Accurate interpretation of Notices to Airmen (NOTAMs) is critical for aviation safety, yet their condensed and cryptic language poses significant challenges to both manual and automated processing. Existing automated systems are typically limited to shallow parsing, failing to extract the actionable intelligence needed for operational decisions. We formalize the complete interpretation task as deep parsing, a dual-reasoning challenge requiring both dynamic knowledge grounding (linking the NOTAM to evolving real-world aeronautical data) and schema-based inference (applying static domain rules to deduce operational status). To tackle this challenge, we propose NOTAM-Evolve, a self-evolving framework that enables a large language model (LLM) to autonomously master complex NOTAM interpretation. Leveraging a knowledge graph-enhanced retrieval module for data grounding, the framework introduces a closed-loop learning process where the LLM progressively improves from its own outputs, minimizing the need for extensive human-annotated reasoning traces. In conjunction with this framework, we introduce a new benchmark dataset of 10,000 expert-annotated NOTAMs. Our experiments demonstrate that NOTAM-Evolve achieves a 30.4% absolute accuracy improvement over the base LLM, establishing a new state of the art on the task of structured NOTAM interpretation. 

---
# Multimodal LLMs Do Not Compose Skills Optimally Across Modalities 

**Authors**: Paula Ontalvilla, Aitor Ormazabal, Gorka Azkune  

**Link**: [PDF](https://arxiv.org/pdf/2511.08113)  

**Abstract**: Skill composition is the ability to combine previously learned skills to solve new tasks. As neural networks acquire increasingly complex skills during their pretraining, it is not clear how successfully they can compose them. In this paper, we focus on Multimodal Large Language Models (MLLM), and study their ability to compose skills across modalities. To this end, we design three evaluation tasks which can be solved sequentially composing two modality-dependent skills, and evaluate several open MLLMs under two main settings: i) prompting the model to directly solve the task, and ii) using a two-step cascaded inference approach, which manually enforces the composition of the two skills for a given task. Even with these straightforward compositions, we find that all evaluated MLLMs exhibit a significant cross-modality skill composition gap. To mitigate the aforementioned gap, we explore two alternatives: i) use chain-of-thought prompting to explicitly instruct MLLMs for skill composition and ii) a specific fine-tuning recipe to promote skill composition. Although those strategies improve model performance, they still exhibit significant skill composition gaps, suggesting that more research is needed to improve cross-modal skill composition in MLLMs. 

---
# Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning 

**Authors**: Songze Li, Zhiqiang Liu, Zhaoyan Gong, Xiaoke Guo, Zhengke Gui, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07910)  

**Abstract**: Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to \textit{Logic Drift} challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA). Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the \textit{Logic Drift} in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs. To enhance LLMs' logic consistency in structured knowledge reasoning, we specifically target the logits output from the autoregressive generation process. We propose the \textit{Logits-to-Logic} framework, which incorporates logits strengthening and logits filtering as core modules to correct logical defects in LLM outputs. Extensive experiments show that our approach significantly improves LLMs' logic consistency in structured knowledge reasoning and achieves state-of-the-art performance on multiple KGQA benchmarks. 

---
# Self-Correction Distillation for Structured Data Question Answering 

**Authors**: Yushan Zhu, Wen Zhang, Long Jin, Mengshu Sun, Ling Zhong, Zhiqiang Liu, Juan Li, Lei Liang, Chong Long, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07998)  

**Abstract**: Structured data question answering (QA), including table QA, Knowledge Graph (KG) QA, and temporal KG QA, is a pivotal research area. Advances in large language models (LLMs) have driven significant progress in unified structural QA frameworks like TrustUQA. However, these frameworks face challenges when applied to small-scale LLMs since small-scale LLMs are prone to errors in generating structured queries. To improve the structured data QA ability of small-scale LLMs, we propose a self-correction distillation (SCD) method. In SCD, an error prompt mechanism (EPM) is designed to detect errors and provide customized error messages during inference, and a two-stage distillation strategy is designed to transfer large-scale LLMs' query-generation and error-correction capabilities to small-scale LLM. Experiments across 5 benchmarks with 3 structured data types demonstrate that our SCD achieves the best performance and superior generalization on small-scale LLM (8B) compared to other distillation methods, and closely approaches the performance of GPT4 on some datasets. Furthermore, large-scale LLMs equipped with EPM surpass the state-of-the-art results on most datasets. 

---
# From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory 

**Authors**: Siyu Xia, Zekun Xu, Jiajun Chai, Wentian Fan, Yan Song, Xiaohan Wang, Guojun Yin, Wei Lin, Haifeng Zhang, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07800)  

**Abstract**: Large Language Models (LLMs) based agents have demonstrated remarkable potential in autonomous task-solving across complex, open-ended environments. A promising approach for improving the reasoning capabilities of LLM agents is to better utilize prior experiences in guiding current decisions. However, LLMs acquire experience either through implicit memory via training, which suffers from catastrophic forgetting and limited interpretability, or explicit memory via prompting, which lacks adaptability. In this paper, we introduce a novel agent-centric, trainable, multi-layered graph memory framework and evaluate how context memory enhances the ability of LLMs to utilize parametric information. The graph abstracts raw agent trajectories into structured decision paths in a state machine and further distills them into high-level, human-interpretable strategic meta-cognition. In order to make memory adaptable, we propose a reinforcement-based weight optimization procedure that estimates the empirical utility of each meta-cognition based on reward feedback from downstream tasks. These optimized strategies are then dynamically integrated into the LLM agent's training loop through meta-cognitive prompting. Empirically, the learnable graph memory delivers robust generalization, improves LLM agents' strategic reasoning performance, and provides consistent benefits during Reinforcement Learning (RL) training. 

---
# LLM Optimization Unlocks Real-Time Pairwise Reranking 

**Authors**: Jingyu Wu, Aditya Shrivastava, Jing Zhu, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07555)  

**Abstract**: Efficiently reranking documents retrieved from information retrieval (IR) pipelines to enhance overall quality of Retrieval-Augmented Generation (RAG) system remains an important yet challenging problem. Recent studies have highlighted the importance of Large Language Models (LLMs) in reranking tasks. In particular, Pairwise Reranking Prompting (PRP) has emerged as a promising plug-and-play approach due to its usability and effectiveness. However, the inherent complexity of the algorithm, coupled with the high computational demands and latency incurred due to LLMs, raises concerns about its feasibility in real-time applications. To address these challenges, this paper presents a focused study on pairwise reranking, demonstrating that carefully applied optimization methods can significantly mitigate these issues. By implementing these methods, we achieve a remarkable latency reduction of up to 166 times, from 61.36 seconds to 0.37 seconds per query, with an insignificant drop in performance measured by Recall@k. Our study highlights the importance of design choices that were previously overlooked, such as using smaller models, limiting the reranked set, using lower precision, reducing positional bias with one-directional order inference, and restricting output tokens. These optimizations make LLM-based reranking substantially more efficient and feasible for latency-sensitive, real-world deployments. 

---
# Design, Results and Industry Implications of the World's First Insurance Large Language Model Evaluation Benchmark 

**Authors**: Hua Zhou, Bing Ma, Yufei Zhang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07794)  

**Abstract**: This paper comprehensively elaborates on the construction methodology, multi-dimensional evaluation system, and underlying design philosophy of CUFEInse v1.0. Adhering to the principles of "quantitative-oriented, expert-driven, and multi-validation," the benchmark establishes an evaluation framework covering 5 core dimensions, 54 sub-indicators, and 14,430 high-quality questions, encompassing insurance theoretical knowledge, industry understanding, safety and compliance, intelligent agent application, and logical rigor. Based on this benchmark, a comprehensive evaluation was conducted on 11 mainstream large language models. The evaluation results reveal that general-purpose models suffer from common bottlenecks such as weak actuarial capabilities and inadequate compliance adaptation. High-quality domain-specific training demonstrates significant advantages in insurance vertical scenarios but exhibits shortcomings in business adaptation and compliance. The evaluation also accurately identifies the common bottlenecks of current large models in professional scenarios such as insurance actuarial, underwriting and claim settlement reasoning, and compliant marketing copywriting. The establishment of CUFEInse not only fills the gap in professional evaluation benchmarks for the insurance field, providing academia and industry with a professional, systematic, and authoritative evaluation tool, but also its construction concept and methodology offer important references for the evaluation paradigm of large models in vertical fields, serving as an authoritative reference for academic model optimization and industrial model selection. Finally, the paper looks forward to the future iteration direction of the evaluation benchmark and the core development direction of "domain adaptation + reasoning enhancement" for insurance large models. 

---
# LLMs vs. Traditional Sentiment Tools in Psychology: An Evaluation on Belgian-Dutch Narratives 

**Authors**: Ratna Kandala, Katie Hoemann  

**Link**: [PDF](https://arxiv.org/pdf/2511.07641)  

**Abstract**: Understanding emotional nuances in everyday language is crucial for computational linguistics and emotion research. While traditional lexicon-based tools like LIWC and Pattern have served as foundational instruments, Large Language Models (LLMs) promise enhanced context understanding. We evaluated three Dutch-specific LLMs (ChocoLlama-8B-Instruct, Reynaerde-7B-chat, and GEITje-7B-ultra) against LIWC and Pattern for valence prediction in Flemish, a low-resource language variant. Our dataset comprised approximately 25000 spontaneous textual responses from 102 Dutch-speaking participants, each providing narratives about their current experiences with self-assessed valence ratings (-50 to +50). Surprisingly, despite architectural advancements, the Dutch-tuned LLMs underperformed compared to traditional methods, with Pattern showing superior performance. These findings challenge assumptions about LLM superiority in sentiment analysis tasks and highlight the complexity of capturing emotional valence in spontaneous, real-world narratives. Our results underscore the need for developing culturally and linguistically tailored evaluation frameworks for low-resource language variants, while questioning whether current LLM fine-tuning approaches adequately address the nuanced emotional expressions found in everyday language use. 

---
# REFLEX: Reference-Free Evaluation of Log Summarization via Large Language Model Judgment 

**Authors**: Priyanka Mudgal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07458)  

**Abstract**: Evaluating log summarization systems is challenging due to the lack of high-quality reference summaries and the limitations of existing metrics like ROUGE and BLEU, which depend on surface-level lexical overlap. We introduce REFLEX, a reference-free evaluation metric for log summarization based on large language model (LLM) judgment. REFLEX uses LLMs as zero-shot evaluators to assess summary quality along dimensions such as relevance, informativeness, and coherence, without requiring gold-standard references or human annotations. We show that REFLEX produces stable, interpretable, and fine-grained evaluations across multiple log summarization dataset, and more effectively distinguishes model outputs than traditional metrics. REFLEX provides a scalable alternative for evaluating log summaries in real-world settings where reference data is scarce or unavailable. 

---
# Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models 

**Authors**: Xin Liu, Qiyang Song, Qihang Zhou, Haichao Du, Shaowen Xu, Wenbo Jiang, Weijuan Zhang, Xiaoqi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2511.07498)  

**Abstract**: Large language models (LLMs) increasingly support multilingual understanding and generation. Meanwhile, efforts to interpret their internal mechanisms have emerged, offering insights to enhance multilingual performance. While multi-head self-attention (MHA) has proven critical in many areas, its role in multilingual capabilities remains underexplored. In this work, we study the contribution of MHA in supporting multilingual processing in LLMs. We propose Language Attention Head Importance Scores (LAHIS), an effective and efficient method that identifies attention head importance for multilingual capabilities via a single forward and backward pass through the LLM. Applying LAHIS to Aya-23-8B, Llama-3.2-3B, and Mistral-7B-v0.1, we reveal the existence of both language-specific and language-general heads. Language-specific heads enable cross-lingual attention transfer to guide the model toward target language contexts and mitigate off-target language generation issue, contributing to addressing challenges in multilingual LLMs. We also introduce a lightweight adaptation that learns a soft head mask to modulate attention outputs over language heads, requiring only 20 tunable parameters to improve XQuAD accuracy. Overall, our work enhances both the interpretability and multilingual capabilities of LLMs from the perspective of MHA. 

---
# GRIP: In-Parameter Graph Reasoning through Fine-Tuning Large Language Models 

**Authors**: Jiarui Feng, Donghong Cai, Yixin Chen, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07457)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in modeling sequential textual data and generalizing across diverse tasks. However, adapting LLMs to effectively handle structural data, such as knowledge graphs or web data, remains a challenging problem. Some approaches adopt complex strategies to convert graphs into text sequences, resulting in significant token overhead and rendering them impractical for large-scale graphs. Others introduce additional modules to encode graphs into fixed-size token representations for LLMs. However, these methods typically require large-scale post-training on graph-text corpus and complex alignment procedures, yet often yield sub-optimal results due to poor modality alignment. Inspired by in-parameter knowledge injection for test-time adaptation of LLMs, we propose GRIP, a novel framework that equips LLMs with the ability to internalize complex relational information from graphs through carefully designed fine-tuning tasks. This knowledge is efficiently stored within lightweight LoRA parameters, enabling the fine-tuned LLM to perform a wide range of graph-related tasks without requiring access to the original graph at inference time. Extensive experiments across multiple benchmarks validate the effectiveness and efficiency of our approach. 

---
# Critical Confabulation: Can LLMs Hallucinate for Social Good? 

**Authors**: Peiqi Sui, Eamon Duede, Hoyt Long, Richard Jean So  

**Link**: [PDF](https://arxiv.org/pdf/2511.07722)  

**Abstract**: LLMs hallucinate, yet some confabulations can have social affordances if carefully bounded. We propose critical confabulation (inspired by critical fabulation from literary and social theory), the use of LLM hallucinations to "fill-in-the-gap" for omissions in archives due to social and political inequality, and reconstruct divergent yet evidence-bound narratives for history's "hidden figures". We simulate these gaps with an open-ended narrative cloze task: asking LLMs to generate a masked event in a character-centric timeline sourced from a novel corpus of unpublished texts. We evaluate audited (for data contamination), fully-open models (the OLMo-2 family) and unaudited open-weight and proprietary baselines under a range of prompts designed to elicit controlled and useful hallucinations. Our findings validate LLMs' foundational narrative understanding capabilities to perform critical confabulation, and show how controlled and well-specified hallucinations can support LLM applications for knowledge production without collapsing speculation into a lack of historical accuracy and fidelity. 

---
# It Takes Two: A Dual Stage Approach for Terminology-Aware Translation 

**Authors**: Akshat Singh Jaswal  

**Link**: [PDF](https://arxiv.org/pdf/2511.07461)  

**Abstract**: This paper introduces DuTerm, a novel two-stage architecture for terminology-constrained machine translation. Our system combines a terminology-aware NMT model, adapted via fine-tuning on large-scale synthetic data, with a prompt-based LLM for post-editing. The LLM stage refines NMT output and enforces terminology adherence. We evaluate DuTerm on English-to German, English-to-Spanish, and English-to-Russian with the WMT 2025 Terminology Shared Task corpus. We demonstrate that flexible, context-driven terminology handling by the LLM consistently yields higher quality translations than strict constraint enforcement. Our results highlight a critical trade-off, revealing that an LLM's work best for high-quality translation as context-driven mutators rather than generators. 

---
# Revisiting NLI: Towards Cost-Effective and Human-Aligned Metrics for Evaluating LLMs in Question Answering 

**Authors**: Sai Shridhar Balamurali, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07659)  

**Abstract**: Evaluating answers from state-of-the-art large language models (LLMs) is challenging: lexical metrics miss semantic nuances, whereas "LLM-as-Judge" scoring is computationally expensive. We re-evaluate a lightweight alternative -- off-the-shelf Natural Language Inference (NLI) scoring augmented by a simple lexical-match flag and find that this decades-old technique matches GPT-4o's accuracy (89.9%) on long-form QA, while requiring orders-of-magnitude fewer parameters. To test human alignment of these metrics rigorously, we introduce DIVER-QA, a new 3000-sample human-annotated benchmark spanning five QA datasets and five candidate LLMs. Our results highlight that inexpensive NLI-based evaluation remains competitive and offer DIVER-QA as an open resource for future metric research. 

---
# PerspAct: Enhancing LLM Situated Collaboration Skills through Perspective Taking and Active Vision 

**Authors**: Sabrina Patania, Luca Annese, Anita Pellegrini, Silvia Serino, Anna Lambiase, Luca Pallonetto, Silvia Rossi, Simone Colombani, Tom Foulsham, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2511.08098)  

**Abstract**: Recent advances in Large Language Models (LLMs) and multimodal foundation models have significantly broadened their application in robotics and collaborative systems. However, effective multi-agent interaction necessitates robust perspective-taking capabilities, enabling models to interpret both physical and epistemic viewpoints. Current training paradigms often neglect these interactive contexts, resulting in challenges when models must reason about the subjectivity of individual perspectives or navigate environments with multiple observers. This study evaluates whether explicitly incorporating diverse points of view using the ReAct framework, an approach that integrates reasoning and acting, can enhance an LLM's ability to understand and ground the demands of other agents. We extend the classic Director task by introducing active visual exploration across a suite of seven scenarios of increasing perspective-taking complexity. These scenarios are designed to challenge the agent's capacity to resolve referential ambiguity based on visual access and interaction, under varying state representations and prompting strategies, including ReAct-style reasoning. Our results demonstrate that explicit perspective cues, combined with active exploration strategies, significantly improve the model's interpretative accuracy and collaborative effectiveness. These findings highlight the potential of integrating active perception with perspective-taking mechanisms in advancing LLMs' application in robotics and multi-agent systems, setting a foundation for future research into adaptive and context-aware AI systems. 

---
# Large Language Models for Scientific Idea Generation: A Creativity-Centered Survey 

**Authors**: Fatemeh Shahhosseini, Arash Marioriyad, Ali Momen, Mahdieh Soleymani Baghshah, Mohammad Hossein Rohban, Shaghayegh Haghjooy Javanmard  

**Link**: [PDF](https://arxiv.org/pdf/2511.07448)  

**Abstract**: Scientific idea generation lies at the heart of scientific discovery and has driven human progress-whether by solving unsolved problems or proposing novel hypotheses to explain unknown phenomena. Unlike standard scientific reasoning or general creative generation, idea generation in science is a multi-objective and open-ended task, where the novelty of a contribution is as essential as its empirical soundness. Large language models (LLMs) have recently emerged as promising generators of scientific ideas, capable of producing coherent and factual outputs with surprising intuition and acceptable reasoning, yet their creative capacity remains inconsistent and poorly understood. This survey provides a structured synthesis of methods for LLM-driven scientific ideation, examining how different approaches balance creativity with scientific soundness. We categorize existing methods into five complementary families: External knowledge augmentation, Prompt-based distributional steering, Inference-time scaling, Multi-agent collaboration, and Parameter-level adaptation. To interpret their contributions, we employ two complementary frameworks: Boden's taxonomy of Combinatorial, Exploratory and Transformational creativity to characterize the level of ideas each family expected to generate, and Rhodes' 4Ps framework-Person, Process, Press, and Product-to locate the aspect or source of creativity that each method emphasizes. By aligning methodological advances with creativity frameworks, this survey clarifies the state of the field and outlines key directions toward reliable, systematic, and transformative applications of LLMs in scientific discovery. 

---
# Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction 

**Authors**: Jun Xu, Xinkai Du, Yu Ao, Peilong Zhao, Yang Li, Ling Zhong, Lin Yuan, Zhongpu Bo, Xiaorui Wang, Mengshu Sun, Zhengke Gui, Dalong Zhang, Zhaoyang Wang, Qiwei Wang, Yangyang Hou, Zhiying Yin, Haofen Wang, Huajun Chen, Lei Liang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.07943)  

**Abstract**: Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at this https URL. 

---
# SciAgent: A Unified Multi-Agent System for Generalistic Scientific Reasoning 

**Authors**: Xuchen Li, Ruitao Wu, Xuanbo Liu, Xukai Wang, Jinbo Hu, Zhixin Bai, Bohan Zeng, Hao Liang, Leheng Chen, Mingrui Chen, Haitian Zhong, Xuanlin Yang, Xu-Yao Zhang, Liu Liu, Jia Li, Kaiqi Huang, Jiahao Xu, Haitao Mi, Wentao Zhang, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2511.08151)  

**Abstract**: Recent advances in large language models have enabled AI systems to achieve expert-level performance on domain-specific scientific tasks, yet these systems remain narrow and handcrafted. We introduce SciAgent, a unified multi-agent system designed for generalistic scientific reasoning-the ability to adapt reasoning strategies across disciplines and difficulty levels. SciAgent organizes problem solving as a hierarchical process: a Coordinator Agent interprets each problem's domain and complexity, dynamically orchestrating specialized Worker Systems, each composed of interacting reasoning Sub-agents for symbolic deduction, conceptual modeling, numerical computation, and verification. These agents collaboratively assemble and refine reasoning pipelines tailored to each task. Across mathematics and physics Olympiads (IMO, IMC, IPhO, CPhO), SciAgent consistently attains or surpasses human gold-medalist performance, demonstrating both domain generality and reasoning adaptability. Additionally, SciAgent has been tested on the International Chemistry Olympiad (IChO) and selected problems from the Humanity's Last Exam (HLE) benchmark, further confirming the system's ability to generalize across diverse scientific domains. This work establishes SciAgent as a concrete step toward generalistic scientific intelligence-AI systems capable of coherent, cross-disciplinary reasoning at expert levels. 

---
# Information Capacity: Evaluating the Efficiency of Large Language Models via Text Compression 

**Authors**: Cheng Yuan, Jiawei Shao, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08066)  

**Abstract**: Recent years have witnessed the rapid advancements of large language models (LLMs) and their expanding applications, leading to soaring demands for computational resources. The widespread adoption of test-time scaling further aggravates the tension between model capability and resource consumption, highlighting the importance of inference efficiency. However, a unified metric that accurately reflects an LLM's efficiency across different model sizes and architectures remains absent. Motivated by the correlation between compression and intelligence, we introduce information capacity, a measure of model efficiency based on text compression performance relative to computational complexity. Larger models can predict the next token more accurately, achieving greater compression gains but at higher computational costs. Empirical evaluations on mainstream open-source models show that models of varying sizes within a series exhibit consistent information capacity. This metric enables a fair efficiency comparison across model series and accurate performance prediction within a model series. A distinctive feature of information capacity is that it incorporates tokenizer efficiency, which affects both input and output token counts but is often neglected in LLM evaluations. We assess the information capacity of 49 models on 5 heterogeneous datasets and observe consistent results on the influences of tokenizer efficiency, pretraining data, and the mixture-of-experts architecture. 

---
# Dual-Process Scaffold Reasoning for Enhancing LLM Code Debugging 

**Authors**: Po-Chung Hsieh, Chin-Po Chen, Jeng-Lin Li, Ming-Ching Chang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08052)  

**Abstract**: Recent LLMs have demonstrated sophisticated problem-solving capabilities on various benchmarks through advanced reasoning algorithms. However, the key research question of identifying reasoning steps that balance complexity and computational efficiency remains unsolved. Recent research has increasingly drawn upon psychological theories to explore strategies for optimizing cognitive pathways. The LLM's final outputs and intermediate steps are regarded as System 1 and System 2, respectively. However, an in-depth exploration of the System 2 reasoning is still lacking. Therefore, we propose a novel psychologically backed Scaffold Reasoning framework for code debugging, which encompasses the Scaffold Stream, Analytic Stream, and Integration Stream. The construction of reference code within the Scaffold Stream is integrated with the buggy code analysis results produced by the Analytic Stream through the Integration Stream. Our framework achieves an 88.91% pass rate and an average inference time of 5.36 seconds per-problem on DebugBench, outperforming other reasoning approaches across various LLMs in both reasoning accuracy and efficiency. Further analyses elucidate the advantages and limitations of various cognitive pathways across varying problem difficulties and bug types. Our findings also corroborate the alignment of the proposed Scaffold Reasoning framework with human cognitive processes. 

---
# LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation 

**Authors**: Xingyu Li, Xiaolei Liu, Cheng Liu, Yixiao Xu, Kangyi Ding, Bangzhou Xin, Jia-Li Yin  

**Link**: [PDF](https://arxiv.org/pdf/2511.07876)  

**Abstract**: As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose LoopLLM, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90% of the maximum output length, compared to 20% for baselines, and improving transferability by around 40% to DeepSeek-V3 and Gemini 2.5 Flash. 

---
# LLM-Powered Fully Automated Chaos Engineering: Towards Enabling Anyone to Build Resilient Software Systems at Low Cost 

**Authors**: Daisuke Kikuta, Hiroki Ikeuchi, Kengo Tajiri  

**Link**: [PDF](https://arxiv.org/pdf/2511.07865)  

**Abstract**: Chaos Engineering (CE) is an engineering technique aimed at improving the resilience of distributed systems. It involves intentionally injecting faults into a system to test its resilience, uncover weaknesses, and address them before they cause failures in production. Recent CE tools automate the execution of predefined CE experiments. However, planning such experiments and improving the system based on the experimental results still remain manual. These processes are labor-intensive and require multi-domain expertise. To address these challenges and enable anyone to build resilient systems at low cost, this paper proposes ChaosEater, a system that automates the entire CE cycle with Large Language Models (LLMs). It predefines an agentic workflow according to a systematic CE cycle and assigns subdivided processes within the workflow to LLMs. ChaosEater targets CE for software systems built on Kubernetes. Therefore, the LLMs in ChaosEater complete CE cycles through software engineering tasks, including requirement definition, code generation, testing, and debugging. We evaluate ChaosEater through case studies on small- and large-scale Kubernetes systems. The results demonstrate that it consistently completes reasonable CE cycles with significantly low time and monetary costs. Its cycles are also qualitatively validated by human engineers and LLMs. 

---
# Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces 

**Authors**: Shreyas Rajesh, Pavan Holur, Chenda Duan, David Chong, Vwani Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.07587)  

**Abstract**: Large Language Models (LLMs) face fundamental challenges in long-context reasoning: many documents exceed their finite context windows, while performance on texts that do fit degrades with sequence length, necessitating their augmentation with external memory frameworks. Current solutions, which have evolved from retrieval using semantic embeddings to more sophisticated structured knowledge graphs representations for improved sense-making and associativity, are tailored for fact-based retrieval and fail to build the space-time-anchored narrative representations required for tracking entities through episodic events. To bridge this gap, we propose the \textbf{Generative Semantic Workspace} (GSW), a neuro-inspired generative memory framework that builds structured, interpretable representations of evolving situations, enabling LLMs to reason over evolving roles, actions, and spatiotemporal contexts. Our framework comprises an \textit{Operator}, which maps incoming observations to intermediate semantic structures, and a \textit{Reconciler}, which integrates these into a persistent workspace that enforces temporal, spatial, and logical coherence. On the Episodic Memory Benchmark (EpBench) \cite{huet_episodic_2025} comprising corpora ranging from 100k to 1M tokens in length, GSW outperforms existing RAG based baselines by up to \textbf{20\%}. Furthermore, GSW is highly efficient, reducing query-time context tokens by \textbf{51\%} compared to the next most token-efficient baseline, reducing inference time costs considerably. More broadly, GSW offers a concrete blueprint for endowing LLMs with human-like episodic memory, paving the way for more capable agents that can reason over long horizons. 

---
# SALT: Steering Activations towards Leakage-free Thinking in Chain of Thought 

**Authors**: Shourya Batra, Pierce Tillman, Samarth Gaggar, Shashank Kesineni, Kevin Zhu, Sunishchal Dev, Ashwinee Panda, Vasu Sharma, Maheep Chaudhary  

**Link**: [PDF](https://arxiv.org/pdf/2511.07772)  

**Abstract**: As Large Language Models (LLMs) evolve into personal assistants with access to sensitive user data, they face a critical privacy challenge: while prior work has addressed output-level privacy, recent findings reveal that LLMs often leak private information through their internal reasoning processes, violating contextual privacy expectations. These leaky thoughts occur when models inadvertently expose sensitive details in their reasoning traces, even when final outputs appear safe. The challenge lies in preventing such leakage without compromising the model's reasoning capabilities, requiring a delicate balance between privacy and utility. We introduce Steering Activations towards Leakage-free Thinking (SALT), a lightweight test-time intervention that mitigates privacy leakage in model's Chain of Thought (CoT) by injecting targeted steering vectors into hidden state. We identify the high-leakage layers responsible for this behavior. Through experiments across multiple LLMs, we demonstrate that SALT achieves reductions including $18.2\%$ reduction in CPL on QwQ-32B, $17.9\%$ reduction in CPL on Llama-3.1-8B, and $31.2\%$ reduction in CPL on Deepseek in contextual privacy leakage dataset AirGapAgent-R while maintaining comparable task performance and utility. Our work establishes SALT as a practical approach for test-time privacy protection in reasoning-capable language models, offering a path toward safer deployment of LLM-based personal agents. 

---
# ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents 

**Authors**: Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aishwarya Balwani, Denis Peskoff, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, Bing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07685)  

**Abstract**: Deep Research (DR) is an emerging agent application that leverages large language models (LLMs) to address open-ended queries. It requires the integration of several capabilities, including multi-step reasoning, cross-document synthesis, and the generation of evidence-backed, long-form answers. Evaluating DR remains challenging because responses are lengthy and diverse, admit many valid solutions, and often depend on dynamic information sources. We introduce ResearchRubrics, a standardized benchmark for DR built with over 2,800+ hours of human labor that pairs realistic, domain-diverse prompts with 2,500+ expert-written, fine-grained rubrics to assess factual grounding, reasoning soundness, and clarity. We also propose a new complexity framework for categorizing DR tasks along three axes: conceptual breadth, logical nesting, and exploration. In addition, we develop human and model-based evaluation protocols that measure rubric adherence for DR agents. We evaluate several state-of-the-art DR systems and find that even leading agents like Gemini's DR and OpenAI's DR achieve under 68% average compliance with our rubrics, primarily due to missed implicit context and inadequate reasoning about retrieved information. Our results highlight the need for robust, scalable assessment of deep research capabilities, to which end we release ResearchRubrics(including all prompts, rubrics, and evaluation code) to facilitate progress toward well-justified research assistants. 

---
# LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows 

**Authors**: Raffi Khatchadourian, Rolando Franco  

**Link**: [PDF](https://arxiv.org/pdf/2511.07585)  

**Abstract**: Financial institutions deploy Large Language Models (LLMs) for reconciliations, regulatory reporting, and client communications, but nondeterministic outputs (output drift) undermine auditability and trust. We quantify drift across five model architectures (7B-120B parameters) on regulated financial tasks, revealing a stark inverse relationship: smaller models (Granite-3-8B, Qwen2.5-7B) achieve 100% output consistency at T=0.0, while GPT-OSS-120B exhibits only 12.5% consistency (95% CI: 3.5-36.0%) regardless of configuration (p<0.0001, Fisher's exact test). This finding challenges conventional assumptions that larger models are universally superior for production deployment.
Our contributions include: (i) a finance-calibrated deterministic test harness combining greedy decoding (T=0.0), fixed seeds, and SEC 10-K structure-aware retrieval ordering; (ii) task-specific invariant checking for RAG, JSON, and SQL outputs using finance-calibrated materiality thresholds (plus or minus 5%) and SEC citation validation; (iii) a three-tier model classification system enabling risk-appropriate deployment decisions; and (iv) an audit-ready attestation system with dual-provider validation.
We evaluated five models (Qwen2.5-7B via Ollama, Granite-3-8B via IBM this http URL, Llama-3.3-70B, Mistral-Medium-2505, and GPT-OSS-120B) across three regulated financial tasks. Across 480 runs (n=16 per condition), structured tasks (SQL) remain stable even at T=0.2, while RAG tasks show drift (25-75%), revealing task-dependent sensitivity. Cross-provider validation confirms deterministic behavior transfers between local and cloud deployments. We map our framework to Financial Stability Board (FSB), Bank for International Settlements (BIS), and Commodity Futures Trading Commission (CFTC) requirements, demonstrating practical pathways for compliance-ready AI deployments. 

---
# CC30k: A Citation Contexts Dataset for Reproducibility-Oriented Sentiment Analysis 

**Authors**: Rochana R. Obadage, Sarah M. Rajtmajer, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07790)  

**Abstract**: Sentiments about the reproducibility of cited papers in downstream literature offer community perspectives and have shown as a promising signal of the actual reproducibility of published findings. To train effective models to effectively predict reproducibility-oriented sentiments and further systematically study their correlation with reproducibility, we introduce the CC30k dataset, comprising a total of 30,734 citation contexts in machine learning papers. Each citation context is labeled with one of three reproducibility-oriented sentiment labels: Positive, Negative, or Neutral, reflecting the cited paper's perceived reproducibility or replicability. Of these, 25,829 are labeled through crowdsourcing, supplemented with negatives generated through a controlled pipeline to counter the scarcity of negative labels. Unlike traditional sentiment analysis datasets, CC30k focuses on reproducibility-oriented sentiments, addressing a research gap in resources for computational reproducibility studies. The dataset was created through a pipeline that includes robust data cleansing, careful crowd selection, and thorough validation. The resulting dataset achieves a labeling accuracy of 94%. We then demonstrated that the performance of three large language models significantly improves on the reproducibility-oriented sentiment classification after fine-tuning using our dataset. The dataset lays the foundation for large-scale assessments of the reproducibility of machine learning papers. The CC30k dataset and the Jupyter notebooks used to produce and analyze the dataset are publicly available at this https URL . 

---
# DynaAct: Large Language Model Reasoning with Dynamic Action Spaces 

**Authors**: Xueliang Zhao, Wei Wu, Jian Guan, Qintong Li, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2511.08043)  

**Abstract**: In modern sequential decision-making systems, the construction of an optimal candidate action space is critical to efficient inference. However, existing approaches either rely on manually defined action spaces that lack scalability or utilize unstructured spaces that render exhaustive search computationally prohibitive. In this paper, we propose a novel framework named \textsc{DynaAct} for automatically constructing a compact action space to enhance sequential reasoning in complex problem-solving scenarios. Our method first estimates a proxy for the complete action space by extracting general sketches observed in a corpus covering diverse complex reasoning problems using large language models. We then formulate a submodular function that jointly evaluates candidate actions based on their utility to the current state and their diversity, and employ a greedy algorithm to select an optimal candidate set. Extensive experiments on six diverse standard benchmarks demonstrate that our approach significantly improves overall performance, while maintaining efficient inference without introducing substantial latency. The implementation is available at this https URL. 

---
# A Matter of Interest: Understanding Interestingness of Math Problems in Humans and Language Models 

**Authors**: Shubhra Mishra, Yuka Machino, Gabriel Poesia, Albert Jiang, Joy Hsu, Adrian Weller, Challenger Mishra, David Broman, Joshua B. Tenenbaum, Mateja Jamnik, Cedegao E. Zhang, Katherine M. Collins  

**Link**: [PDF](https://arxiv.org/pdf/2511.08548)  

**Abstract**: The evolution of mathematics has been guided in part by interestingness. From researchers choosing which problems to tackle next, to students deciding which ones to engage with, people's choices are often guided by judgments about how interesting or challenging problems are likely to be. As AI systems, such as LLMs, increasingly participate in mathematics with people -- whether for advanced research or education -- it becomes important to understand how well their judgments align with human ones. Our work examines this alignment through two empirical studies of human and LLM assessment of mathematical interestingness and difficulty, spanning a range of mathematical experience. We study two groups: participants from a crowdsourcing platform and International Math Olympiad competitors. We show that while many LLMs appear to broadly agree with human notions of interestingness, they mostly do not capture the distribution observed in human judgments. Moreover, most LLMs only somewhat align with why humans find certain math problems interesting, showing weak correlation with human-selected interestingness rationales. Together, our findings highlight both the promises and limitations of current LLMs in capturing human interestingness judgments for mathematical AI thought partnerships. 

---
# Patching LLM Like Software: A Lightweight Method for Improving Safety Policy in Large Language Models 

**Authors**: Huzaifa Arif, Keerthiram Murugesan, Ching-Yun Ko, Pin-Yu Chen, Payel Das, Alex Gittens  

**Link**: [PDF](https://arxiv.org/pdf/2511.08484)  

**Abstract**: We propose patching for large language models (LLMs) like software versions, a lightweight and modular approach for addressing safety vulnerabilities. While vendors release improved LLM versions, major releases are costly, infrequent, and difficult to tailor to customer needs, leaving released models with known safety gaps. Unlike full-model fine-tuning or major version updates, our method enables rapid remediation by prepending a compact, learnable prefix to an existing model. This "patch" introduces only 0.003% additional parameters, yet reliably steers model behavior toward that of a safer reference model. Across three critical domains (toxicity mitigation, bias reduction, and harmfulness refusal) policy patches achieve safety improvements comparable to next-generation safety-aligned models while preserving fluency. Our results demonstrate that LLMs can be "patched" much like software, offering vendors and practitioners a practical mechanism for distributing scalable, efficient, and composable safety updates between major model releases. 

---
# FaithAct: Faithfulness Planning and Acting in MLLMs 

**Authors**: Junxian Li, Xinyue Xu, Sai Ma, Sichao Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08409)  

**Abstract**: Unfaithfulness remains a persistent challenge for large language models (LLMs), which often produce plausible yet ungrounded reasoning chains that diverge from perceptual evidence or final conclusions. We distinguish between behavioral faithfulness (alignment between reasoning and output) and perceptual faithfulness (alignment between reasoning and input), and introduce FaithEval for quantifying step-level and chain-level faithfulness by evaluating whether each claimed object is visually supported by the image. Building on these insights, we propose FaithAct, a faithfulness-first planning and acting framework that enforces evidential grounding at every reasoning step. Experiments across multiple reasoning benchmarks demonstrate that FaithAct improves perceptual faithfulness by up to 26% without degrading task accuracy compared to prompt-based and tool-augmented baselines. Our analysis shows that treating faithfulness as a guiding principle not only mitigates hallucination but also leads to more stable reasoning trajectories. This work thereby establishes a unified framework for both evaluating and enforcing faithfulness in multimodal reasoning. 

---
# MADD: Multi-Agent Drug Discovery Orchestra 

**Authors**: Gleb V. Solovev, Alina B. Zhidkovskaya, Anastasia Orlova, Nina Gubina, Anastasia Vepreva, Rodion Golovinskii, Ilya Tonkii, Ivan Dubrovsky, Ivan Gurev, Dmitry Gilemkhanov, Denis Chistiakov, Timur A. Aliev, Ivan Poddiakov, Galina Zubkova, Ekaterina V. Skorb, Vladimir Vinogradov, Alexander Boukhanovsky, Nikolay Nikitin, Andrei Dmitrenko, Anna Kalyuzhnaya, Andrey Savchenko  

**Link**: [PDF](https://arxiv.org/pdf/2511.08217)  

**Abstract**: Hit identification is a central challenge in early drug discovery, traditionally requiring substantial experimental resources. Recent advances in artificial intelligence, particularly large language models (LLMs), have enabled virtual screening methods that reduce costs and improve efficiency. However, the growing complexity of these tools has limited their accessibility to wet-lab researchers. Multi-agent systems offer a promising solution by combining the interpretability of LLMs with the precision of specialized models and tools. In this work, we present MADD, a multi-agent system that builds and executes customized hit identification pipelines from natural language queries. MADD employs four coordinated agents to handle key subtasks in de novo compound generation and screening. We evaluate MADD across seven drug discovery cases and demonstrate its superior performance compared to existing LLM-based solutions. Using MADD, we pioneer the application of AI-first drug design to five biological targets and release the identified hit molecules. Finally, we introduce a new benchmark of query-molecule pairs and docking scores for over three million compounds to contribute to the agentic future of drug design. 

---
# EHRStruct: A Comprehensive Benchmark Framework for Evaluating Large Language Models on Structured Electronic Health Record Tasks 

**Authors**: Xiao Yang, Xuejiao Zhao, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08206)  

**Abstract**: Structured Electronic Health Record (EHR) data stores patient information in relational tables and plays a central role in clinical decision-making. Recent advances have explored the use of large language models (LLMs) to process such data, showing promise across various clinical this http URL, the absence of standardized evaluation frameworks and clearly defined tasks makes it difficult to systematically assess and compare LLM performance on structured EHR this http URL address these evaluation challenges, we introduce EHRStruct, a benchmark specifically designed to evaluate LLMs on structured EHR this http URL defines 11 representative tasks spanning diverse clinical needs and includes 2,200 task-specific evaluation samples derived from two widely used EHR this http URL use EHRStruct to evaluate 20 advanced and representative LLMs, covering both general and medical this http URL further analyze key factors influencing model performance, including input formats, few-shot generalisation, and finetuning strategies, and compare results with 11 state-of-the-art LLM-based enhancement methods for structured data reasoning. Our results indicate that many structured EHR tasks place high demands on the understanding and reasoning capabilities of this http URL response, we propose EHRMaster, a code-augmented method that achieves state-of-the-art performance and offers practical 

---
# National Institute on Aging PREPARE Challenge: Early Detection of Cognitive Impairment Using Speech - The SpeechCARE Solution 

**Authors**: Maryam Zolnoori, Hossein Azadmaleki, Yasaman Haghbin, Ali Zolnour, Mohammad Javad Momeni Nezhad, Sina Rashidi, Mehdi Naserian, Elyas Esmaeili, Sepehr Karimi Arpanahi  

**Link**: [PDF](https://arxiv.org/pdf/2511.08132)  

**Abstract**: Alzheimer's disease and related dementias (ADRD) affect one in five adults over 60, yet more than half of individuals with cognitive decline remain undiagnosed. Speech-based assessments show promise for early detection, as phonetic motor planning deficits alter acoustic features (e.g., pitch, tone), while memory and language impairments lead to syntactic and semantic errors. However, conventional speech-processing pipelines with hand-crafted features or general-purpose audio classifiers often exhibit limited performance and generalizability. To address these limitations, we introduce SpeechCARE, a multimodal speech processing pipeline that leverages pretrained, multilingual acoustic and linguistic transformer models to capture subtle speech-related cues associated with cognitive impairment. Inspired by the Mixture of Experts (MoE) paradigm, SpeechCARE employs a dynamic fusion architecture that weights transformer-based acoustic, linguistic, and demographic inputs, allowing integration of additional modalities (e.g., social factors, imaging) and enhancing robustness across diverse tasks. Its robust preprocessing includes automatic transcription, large language model (LLM)-based anomaly detection, and task identification. A SHAP-based explainability module and LLM reasoning highlight each modality's contribution to decision-making. SpeechCARE achieved AUC = 0.88 and F1 = 0.72 for classifying cognitively healthy, MCI, and AD individuals, with AUC = 0.90 and F1 = 0.62 for MCI detection. Bias analysis showed minimal disparities, except for adults over 80. Mitigation techniques included oversampling and weighted loss. Future work includes deployment in real-world care settings (e.g., VNS Health, Columbia ADRC) and EHR-integrated explainability for underrepresented populations in New York City. 

---
# Knowledge-Augmented Long-CoT Generation for Complex Biomolecular Reasoning 

**Authors**: Tianwen Lyu, Xiang Zhuang, Keyan Ding, Xinzhe Cao, Lei Liang, Wei Zhao, Qiang Zhang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08024)  

**Abstract**: Understanding complex biomolecular mechanisms requires multi-step reasoning across molecular interactions, signaling cascades, and metabolic pathways. While large language models(LLMs) show promise in such tasks, their application to biomolecular problems is hindered by logical inconsistencies and the lack of grounding in domain knowledge. Existing approaches often exacerbate these issues: reasoning steps may deviate from biological facts or fail to capture long mechanistic dependencies. To address these challenges, we propose a Knowledge-Augmented Long-CoT Reasoning framework that integrates LLMs with knowledge graph-based multi-hop reasoning chains. The framework constructs mechanistic chains via guided multi-hop traversal and pruning on the knowledge graph; these chains are then incorporated into supervised fine-tuning to improve factual grounding and further refined with reinforcement learning to enhance reasoning reliability and consistency. Furthermore, to overcome the shortcomings of existing benchmarks, which are often restricted in scale and scope and lack annotations for deep reasoning chains, we introduce PrimeKGQA, a comprehensive benchmark for biomolecular question answering. Experimental results on both PrimeKGQA and existing datasets demonstrate that although larger closed-source models still perform well on relatively simple tasks, our method demonstrates clear advantages as reasoning depth increases, achieving state-of-the-art performance on multi-hop tasks that demand traversal of structured biological knowledge. These findings highlight the effectiveness of combining structured knowledge with advanced reasoning strategies for reliable and interpretable biomolecular reasoning. 

---
# Numerical Sensitivity and Robustness: Exploring the Flaws of Mathematical Reasoning in Large Language Models 

**Authors**: Zhishen Sun, Guang Dai, Ivor Tsang, Haishan Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.08022)  

**Abstract**: LLMs have made significant progress in the field of mathematical reasoning, but whether they have true the mathematical understanding ability is still controversial. To explore this issue, we propose a new perturbation framework to evaluate LLMs' reasoning ability in complex environments by injecting additional semantically irrelevant perturbation sentences and gradually increasing the perturbation intensity. At the same time, we use an additional perturbation method: core questioning instruction missing, to further analyze the LLMs' problem-solving mechanism. The experimental results show that LLMs perform stably when facing perturbation sentences without numbers, but there is also a robustness boundary. As the perturbation intensity increases, the performance exhibits varying degrees of decline; when facing perturbation sentences with numbers, the performance decreases more significantly, most open source models with smaller parameters decrease by nearly or even more than 10%, and further increasing with the enhancement of perturbation intensity, with the maximum decrease reaching 51.55%. Even the most advanced commercial LLMs have seen a 3%-10% performance drop. By analyzing the reasoning process of LLMs in detail, We find that models are more sensitive to perturbations with numerical information and are more likely to give incorrect answers when disturbed by irrelevant numerical information. The higher the perturbation intensity, the more obvious these defects are. At the same time, in the absence of core questioning instruction, models can still maintain an accuracy of 20%-40%, indicating that LLMs may rely on memory templates or pattern matching to complete the task, rather than logical reasoning. In general, our work reveals the shortcomings and limitations of current LLMs in their reasoning capabilities, which is of great significance for the further development of LLMs. 

---
# MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement 

**Authors**: Zhishen Sun, Guang Dai, Haishan Ye  

**Link**: [PDF](https://arxiv.org/pdf/2511.08055)  

**Abstract**: LLMs demonstrate performance comparable to human abilities in complex tasks such as mathematical reasoning, but their robustness in mathematical reasoning under minor input perturbations still lacks systematic investigation. Existing methods generally suffer from limited scalability, weak semantic preservation, and high costs. Therefore, we propose MSCR, an automated adversarial attack method based on multi-source candidate replacement. By combining three information sources including cosine similarity in the embedding space of LLMs, the WordNet dictionary, and contextual predictions from a masked language model, we generate for each word in the input question a set of semantically similar candidates, which are then filtered and substituted one by one to carry out the attack. We conduct large-scale experiments on LLMs using the GSM8K and MATH500 benchmarks. The results show that even a slight perturbation involving only a single word can significantly reduce the accuracy of all models, with the maximum drop reaching 49.89% on GSM8K and 35.40% on MATH500, while preserving the high semantic consistency of the perturbed questions. Further analysis reveals that perturbations not only lead to incorrect outputs but also substantially increase the average response length, which results in more redundant reasoning paths and higher computational resource consumption. These findings highlight the robustness deficiencies and efficiency bottlenecks of current LLMs in mathematical reasoning tasks. 

---
# Benchmarking Multi-Step Legal Reasoning and Analyzing Chain-of-Thought Effects in Large Language Models 

**Authors**: Wenhan Yu, Xinbo Lin, Lanxin Ni, Jinhua Cheng, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2511.07979)  

**Abstract**: Large language models (LLMs) have demonstrated strong reasoning abilities across specialized domains, motivating research into their application to legal reasoning. However, existing legal benchmarks often conflate factual recall with genuine inference, fragment the reasoning process, and overlook the quality of reasoning. To address these limitations, we introduce MSLR, the first Chinese multi-step legal reasoning dataset grounded in real-world judicial decision making. MSLR adopts the IRAC framework (Issue, Rule, Application, Conclusion) to model structured expert reasoning from official legal documents. In addition, we design a scalable Human-LLM collaborative annotation pipeline that efficiently produces fine-grained step-level reasoning annotations and provides a reusable methodological framework for multi-step reasoning datasets. Evaluation of multiple LLMs on MSLR shows only moderate performance, highlighting the challenges of adapting to complex legal reasoning. Further experiments demonstrate that Self-Initiated Chain-of-Thought prompts generated by models autonomously improve reasoning coherence and quality, outperforming human-designed prompts. MSLR contributes to advancing LLM reasoning and Chain-of-Thought strategies and offers open resources for future research. The dataset and code are available at this https URL and this https URL. 

---
# Computational Blueprints: Generating Isomorphic Mathematics Problems with Large Language Models 

**Authors**: Jeong-Hoon Kim, Jinwoo Nam, Geunsik Jo  

**Link**: [PDF](https://arxiv.org/pdf/2511.07932)  

**Abstract**: Personalized mathematics education is growing rapidly, creating a strong demand for large sets of similar practice problems. Yet existing studies on mathematics problem generation have focused on data augmentation for training neural language models rather than on direct educational deployment. To bridge this gap, we define a new task, Isomorphic Math Problem Generation (IMPG), designed to produce structurally consistent variants of source problems. Subsequently, we explored LLM-based frameworks for automatic IMPG through successive refinements, and established Computational Blueprints for Isomorphic Twins (CBIT). With meta-level generation and template-based selective variation, CBIT achieves high mathematical correctness and structural consistency while reducing the cost of generation. Empirical results across refinements demonstrate that CBIT is superior on generation accuracy and cost-effectiveness at scale. Most importantly, CBIT-generated problems exhibited an error rate 17.8% lower than expert-authored items, with deployment to 6,732 learners on a commercial education platform yielding 186,870 interactions. 

---
# Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-view Multi-Label Feature Selection 

**Authors**: Zhiqi Chen, Yuzhou Liu, Jiarui Liu, Wanfu Gao  

**Link**: [PDF](https://arxiv.org/pdf/2511.08008)  

**Abstract**: Multi-view multi-label feature selection aims to identify informative features from heterogeneous views, where each sample is associated with multiple interdependent labels. This problem is particularly important in machine learning involving high-dimensional, multimodal data such as social media, bioinformatics or recommendation systems. Existing Multi-View Multi-Label Feature Selection (MVMLFS) methods mainly focus on analyzing statistical information of data, but seldom consider semantic information. In this paper, we aim to use these two types of information jointly and propose a method that combines Large Language Models (LLMs) semantic reasoning with Graph Neural Networks (GNNs) structural modeling for MVMLFS. Specifically, the method consists of three main components. (1) LLM is first used as an evaluation agent to assess the latent semantic relevance among feature, view, and label descriptions. (2) A semantic-aware heterogeneous graph with two levels is designed to represent relations among features, views and labels: one is a semantic graph representing semantic relations, and the other is a statistical graph. (3) A lightweight Graph Attention Network (GAT) is applied to learn node embedding in the heterogeneous graph as feature saliency scores for ranking and selection. Experimental results on multiple benchmark datasets demonstrate the superiority of our method over state-of-the-art baselines, and it is still effective when applied to small-scale datasets, showcasing its robustness, flexibility, and generalization ability. 

---
# Data Descriptions from Large Language Models with Influence Estimation 

**Authors**: Chaeri Kim, Jaeyeon Bae, Taehwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07897)  

**Abstract**: Deep learning models have been successful in many areas but understanding their behaviors still remains a black-box. Most prior explainable AI (XAI) approaches have focused on interpreting and explaining how models make predictions. In contrast, we would like to understand how data can be explained with deep learning model training and propose a novel approach to understand the data via one of the most common media - language - so that humans can easily understand. Our approach proposes a pipeline to generate textual descriptions that can explain the data with large language models by incorporating external knowledge bases. However, generated data descriptions may still include irrelevant information, so we introduce to exploit influence estimation to choose the most informative textual descriptions, along with the CLIP score. Furthermore, based on the phenomenon of cross-modal transferability, we propose a novel benchmark task named cross-modal transfer classification to examine the effectiveness of our textual descriptions. In the experiment of zero-shot setting, we show that our textual descriptions are more effective than other baseline descriptions, and furthermore, we successfully boost the performance of the model trained only on images across all nine image classification datasets. These results are further supported by evaluation using GPT-4o. Through our approach, we may gain insights into the inherent interpretability of the decision-making process of the model. 

---
# Alignment-Aware Quantization for LLM Safety 

**Authors**: Sunghyun Wee, Suyoung Kim, Hyeonjin Kim, Kyomin Hwang, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2511.07842)  

**Abstract**: Safety and efficiency are both important factors when deploying large language models(LLMs). LLMs are trained to follow human alignment for safety, and post training quantization(PTQ) is applied afterward for efficiency. However, these two objectives are often in conflict, revealing a fundamental flaw in the conventional PTQ paradigm: quantization can turn into a safety vulnerability if it only aims to achieve low perplexity. Models can demonstrate low perplexity yet exhibit significant degradation in alignment with the safety policy, highlighting that perplexity alone is an insufficient and often misleading proxy for model safety. To address this, we propose Alignment-Aware Quantization(AAQ), a novel approach that integrates Alignment-Preserving Contrastive(APC) loss into the PTQ pipeline. Compared to simple reconstruction loss, ours explicitly preserves alignment by encouraging the quantized model to mimic its safe, instruction-tuned model while diverging from the unaligned, pre-trained counterpart. Our method achieves this robust safety alignment without resorting to specialized safety-focused calibration datasets, highlighting its practical utility and broad applicability. AAQ is compatible with standard PTQ techniques and enables robust 4-bit (W4A4) quantization across diverse model families such as LLaMA, Qwen, and Mistral while maintaining safety where previous methods fail. Our work resolves the critical trade-off between efficiency and safety, paving the way toward LLMs that are both efficient and trustworthy. Anonymized code is available in the supplementary material. 

---
# Towards AI-Assisted Generation of Military Training Scenarios 

**Authors**: Soham Hans, Volkan Ustun, Benjamin Nye, James Sterrett, Matthew Green  

**Link**: [PDF](https://arxiv.org/pdf/2511.07690)  

**Abstract**: Achieving expert-level performance in simulation-based training relies on the creation of complex, adaptable scenarios, a traditionally laborious and resource intensive process. Although prior research explored scenario generation for military training, pre-LLM AI tools struggled to generate sufficiently complex or adaptable scenarios. This paper introduces a multi-agent, multi-modal reasoning framework that leverages Large Language Models (LLMs) to generate critical training artifacts, such as Operations Orders (OPORDs). We structure our framework by decomposing scenario generation into a hierarchy of subproblems, and for each one, defining the role of the AI tool: (1) generating options for a human author to select from, (2) producing a candidate product for human approval or modification, or (3) generating textual artifacts fully automatically. Our framework employs specialized LLM-based agents to address distinct subproblems. Each agent receives input from preceding subproblem agents, integrating both text-based scenario details and visual information (e.g., map features, unit positions and applies specialized reasoning to produce appropriate outputs. Subsequent agents process these outputs sequentially, preserving logical consistency and ensuring accurate document generation. This multi-agent strategy overcomes the limitations of basic prompting or single-agent approaches when tackling such highly complex tasks. We validate our framework through a proof-of-concept that generates the scheme of maneuver and movement section of an OPORD while estimating map positions and movements as a precursor demonstrating its feasibility and accuracy. Our results demonstrate the potential of LLM-driven multi-agent systems to generate coherent, nuanced documents and adapt dynamically to changing conditions, advancing automation in scenario generation for military training. 

---
# AIA Forecaster: Technical Report 

**Authors**: Rohan Alur, Bradly C. Stadie, Daniel Kang, Ryan Chen, Matt McManus, Michael Rickert, Tyler Lee, Michael Federici, Richard Zhu, Dennis Fogerty, Hayley Williamson, Nina Lozinski, Aaron Linsky, Jasjeet S. Sekhon  

**Link**: [PDF](https://arxiv.org/pdf/2511.07678)  

**Abstract**: This technical report describes the AIA Forecaster, a Large Language Model (LLM)-based system for judgmental forecasting using unstructured data. The AIA Forecaster approach combines three core elements: agentic search over high-quality news sources, a supervisor agent that reconciles disparate forecasts for the same event, and a set of statistical calibration techniques to counter behavioral biases in large language models. On the ForecastBench benchmark (Karger et al., 2024), the AIA Forecaster achieves performance equal to human superforecasters, surpassing prior LLM baselines. In addition to reporting on ForecastBench, we also introduce a more challenging forecasting benchmark sourced from liquid prediction markets. While the AIA Forecaster underperforms market consensus on this benchmark, an ensemble combining AIA Forecaster with market consensus outperforms consensus alone, demonstrating that our forecaster provides additive information. Our work establishes a new state of the art in AI forecasting and provides practical, transferable recommendations for future research. To the best of our knowledge, this is the first work that verifiably achieves expert-level forecasting at scale. 

---
# VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation 

**Authors**: Hyojun Choi, Seokju Hwang, Kyong-Ho Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07991)  

**Abstract**: Competency Questions (CQs) play a crucial role in validating ontology design. While manually crafting CQs can be highly time-consuming and costly for ontology engineers, recent studies have explored the use of large language models (LLMs) to automate this process. However, prior approaches have largely evaluated generated CQs based on their similarity to existing datasets, which often fail to verify semantic pitfalls such as "Misusing allValuesFrom". Since such pitfalls cannot be reliably detected through rule-based methods, we propose a novel dataset and model of Validating Semantic Pitfalls in Ontology (VSPO) for CQ generation specifically designed to verify the semantic pitfalls. To simulate missing and misused axioms, we use LLMs to generate natural language definitions of classes and properties and introduce misalignments between the definitions and the ontology by removing axioms or altering logical operators (e.g., substituting union with intersection). We then fine-tune LLaMA-3.1-8B-Instruct to generate CQs that validate these semantic discrepancies between the provided definitions and the corresponding axioms. The resulting CQs can detect a broader range of modeling errors compared to existing public datasets. Our fine-tuned model demonstrates superior performance over baselines, showing 26% higher precision and 28.2% higher recall than GPT-4.1 in generating CQs for pitfall validation. This research enables automatic generation of TBox-validating CQs using LLMs, significantly reducing manual effort while improving semantic alignment between ontologies and expert knowledge. To the best of our knowledge, this is the first study to target semantic pitfall validation in CQ generation using LLMs. 

---
# Procedural Knowledge Improves Agentic LLM Workflows 

**Authors**: Vincent Hsiao, Mark Roberts, Leslie Smith  

**Link**: [PDF](https://arxiv.org/pdf/2511.07568)  

**Abstract**: Large language models (LLMs) often struggle when performing agentic tasks without substantial tool support, prom-pt engineering, or fine tuning. Despite research showing that domain-dependent, procedural knowledge can dramatically increase planning efficiency, little work evaluates its potential for improving LLM performance on agentic tasks that may require implicit planning. We formalize, implement, and evaluate an agentic LLM workflow that leverages procedural knowledge in the form of a hierarchical task network (HTN). Empirical results of our implementation show that hand-coded HTNs can dramatically improve LLM performance on agentic tasks, and using HTNs can boost a 20b or 70b parameter LLM to outperform a much larger 120b parameter LLM baseline. Furthermore, LLM-created HTNs improve overall performance, though less so. The results suggest that leveraging expertise--from humans, documents, or LLMs--to curate procedural knowledge will become another important tool for improving LLM workflows. 

---
# Beyond Correctness: Confidence-Aware Reward Modeling for Enhancing Large Language Model Reasoning 

**Authors**: Qianxi He, Qingyu Ren, Shanzhe Lei, Xuhong Wang, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07483)  

**Abstract**: Recent advancements in large language models (LLMs) have shifted the post-training paradigm from traditional instruction tuning and human preference alignment toward reinforcement learning (RL) focused on reasoning capabilities. However, numerous technical reports indicate that purely rule-based reward RL frequently results in poor-quality reasoning chains or inconsistencies between reasoning processes and final answers, particularly when the base model is of smaller scale. During the RL exploration process, models might employ low-quality reasoning chains due to the lack of knowledge, occasionally producing correct answers randomly and receiving rewards based on established rule-based judges. This constrains the potential for resource-limited organizations to conduct direct reinforcement learning training on smaller-scale models. We propose a novel confidence-based reward model tailored for enhancing STEM reasoning capabilities. Unlike conventional approaches, our model penalizes not only incorrect answers but also low-confidence correct responses, thereby promoting more robust and logically consistent reasoning. We validate the effectiveness of our approach through static evaluations, Best-of-N inference tests, and PPO-based RL training. Our method outperforms several state-of-the-art open-source reward models across diverse STEM benchmarks. We release our codes and model in this https URL. 

---
# Making LLMs Reliable When It Matters Most: A Five-Layer Architecture for High-Stakes Decisions 

**Authors**: Alejandro R. Jadad  

**Link**: [PDF](https://arxiv.org/pdf/2511.07669)  

**Abstract**: Current large language models (LLMs) excel in verifiable domains where outputs can be checked before action but prove less reliable for high-stakes strategic decisions with uncertain outcomes. This gap, driven by mutually reinforcing cognitive biases in both humans and artificial intelligence (AI) systems, threatens the defensibility of valuations and sustainability of investments in the sector.
This report describes a framework emerging from systematic qualitative assessment across 7 frontier-grade LLMs and 3 market-facing venture vignettes under time pressure. Detailed prompting specifying decision partnership and explicitly instructing avoidance of sycophancy, confabulation, solution drift, and nihilism achieved initial partnership state but failed to maintain it under operational pressure. Sustaining protective partnership state required an emergent 7-stage calibration sequence, built upon a 4-stage initialization process, within a 5-layer protection architecture enabling bias self-monitoring, human-AI adversarial challenge, partnership state verification, performance degradation detection, and stakeholder protection.
Three discoveries resulted: partnership state is achievable through ordered calibration but requires emergent maintenance protocols; reliability degrades when architectural drift and context exhaustion align; and dissolution discipline prevents costly pursuit of fundamentally wrong directions. Cross-model validation revealed systematic performance differences across LLM architectures.
This approach demonstrates that human-AI teams can achieve cognitive partnership capable of preventing avoidable regret in high-stakes decisions, addressing return-on-investment expectations that depend on AI systems supporting consequential decision-making without introducing preventable cognitive traps when verification arrives too late. 

---
# AI-Driven Contribution Evaluation and Conflict Resolution: A Framework & Design for Group Workload Investigation 

**Authors**: Jakub Slapek, Mir Seyedebrahimi, Yang Jianhua  

**Link**: [PDF](https://arxiv.org/pdf/2511.07667)  

**Abstract**: The equitable assessment of individual contribution in teams remains a persistent challenge, where conflict and disparity in workload can result in unfair performance evaluation, often requiring manual intervention - a costly and challenging process. We survey existing tool features and identify a gap in conflict resolution methods and AI integration. To address this, we propose a framework and implementation design for a novel AI-enhanced tool that assists in dispute investigation. The framework organises heterogeneous artefacts - submissions (code, text, media), communications (chat, email), coordination records (meeting logs, tasks), peer assessments, and contextual information - into three dimensions with nine benchmarks: Contribution, Interaction, and Role. Objective measures are normalised, aggregated per dimension, and paired with inequality measures (Gini index) to surface conflict markers. A Large Language Model (LLM) architecture performs validated and contextual analysis over these measures to generate interpretable and transparent advisory judgments. We argue for feasibility under current statutory and institutional policy, and outline practical analytics (sentimental, task fidelity, word/line count, etc.), bias safeguards, limitations, and practical challenges. 

---
# Large Sign Language Models: Toward 3D American Sign Language Translation 

**Authors**: Sen Zhang, Xiaoxiao He, Di Liu, Zhaoyang Xia, Mingyu Zhao, Chaowei Tan, Vivian Li, Bo Liu, Dimitris N. Metaxas, Mubbasir Kapadia  

**Link**: [PDF](https://arxiv.org/pdf/2511.08535)  

**Abstract**: We present Large Sign Language Models (LSLM), a novel framework for translating 3D American Sign Language (ASL) by leveraging Large Language Models (LLMs) as the backbone, which can benefit hearing-impaired individuals' virtual communication. Unlike existing sign language recognition methods that rely on 2D video, our approach directly utilizes 3D sign language data to capture rich spatial, gestural, and depth information in 3D scenes. This enables more accurate and resilient translation, enhancing digital communication accessibility for the hearing-impaired community. Beyond the task of ASL translation, our work explores the integration of complex, embodied multimodal languages into the processing capabilities of LLMs, moving beyond purely text-based inputs to broaden their understanding of human communication. We investigate both direct translation from 3D gesture features to text and an instruction-guided setting where translations can be modulated by external prompts, offering greater flexibility. This work provides a foundational step toward inclusive, multimodal intelligent systems capable of understanding diverse forms of language. 

---
# Analysing Environmental Efficiency in AI for X-Ray Diagnosis 

**Authors**: Liam Kearns  

**Link**: [PDF](https://arxiv.org/pdf/2511.07436)  

**Abstract**: The integration of AI tools into medical applications has aimed to improve the efficiency of diagnosis. The emergence of large language models (LLMs), such as ChatGPT and Claude, has expanded this integration even further. Because of LLM versatility and ease of use through APIs, these larger models are often utilised even though smaller, custom models can be used instead. In this paper, LLMs and small discriminative models are integrated into a Mendix application to detect Covid-19 in chest X-rays. These discriminative models are also used to provide knowledge bases for LLMs to improve accuracy. This provides a benchmark study of 14 different model configurations for comparison of accuracy and environmental impact. The findings indicated that while smaller models reduced the carbon footprint of the application, the output was biased towards a positive diagnosis and the output probabilities were lacking confidence. Meanwhile, restricting LLMs to only give probabilistic output caused poor performance in both accuracy and carbon footprint, demonstrating the risk of using LLMs as a universal AI solution. While using the smaller LLM GPT-4.1-Nano reduced the carbon footprint by 94.2% compared to the larger models, this was still disproportionate to the discriminative models; the most efficient solution was the Covid-Net model. Although it had a larger carbon footprint than other small models, its carbon footprint was 99.9% less than when using GPT-4.5-Preview, whilst achieving an accuracy of 95.5%, the highest of all models examined. This paper contributes to knowledge by comparing generative and discriminative models in Covid-19 detection as well as highlighting the environmental risk of using generative tools for classification tasks. 

---
# Designing LLM-based Multi-Agent Systems for Software Engineering Tasks: Quality Attributes, Design Patterns and Rationale 

**Authors**: Yangxiao Cai, Ruiyin Li, Peng Liang, Mojtaba Shahin, Zengyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.08475)  

**Abstract**: As the complexity of Software Engineering (SE) tasks continues to escalate, Multi-Agent Systems (MASs) have emerged as a focal point of research and practice due to their autonomy and scalability. Furthermore, through leveraging the reasoning and planning capabilities of Large Language Models (LLMs), the application of LLM-based MASs in the field of SE is garnering increasing attention. However, there is no dedicated study that systematically explores the design of LLM-based MASs, including the Quality Attributes (QAs) on which the designers mainly focus, the design patterns used by the designers, and the rationale guiding the design of LLM-based MASs for SE tasks. To this end, we conducted a study to identify the QAs that LLM-based MASs for SE tasks focus on, the design patterns used in the MASs, and the design rationale for the MASs. We collected 94 papers on LLM-based MASs for SE tasks as the source. Our study shows that: (1) Code Generation is the most common SE task solved by LLM-based MASs among ten identified SE tasks, (2) Functional Suitability is the QA on which designers of LLM-based MASs pay the most attention, (3) Role-Based Cooperation is the design pattern most frequently employed among 16 patterns used to construct LLM-based MASs, and (4) Improving the Quality of Generated Code is the most common rationale behind the design of LLM-based MASs. Based on the study results, we presented the implications for the design of LLM-based MASs to support SE tasks. 

---
# DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes 

**Authors**: Haowen Li, Zhengding Luo, Dongyuan Shi, Boxiang Wang, Junwei Ji, Ziyi Yang, Woon-Seng Gan  

**Link**: [PDF](https://arxiv.org/pdf/2511.08012)  

**Abstract**: Direction-of-Arrival (DOA) estimation is critical in spatial audio and acoustic signal processing, with wide-ranging applications in real-world. Most existing DOA models are trained on synthetic data by convolving clean speech with room impulse responses (RIRs), which limits their generalizability due to constrained acoustic diversity. In this paper, we revisit DOA estimation using a recently introduced dataset constructed with the assistance of large language models (LLMs), which provides more realistic and diverse spatial audio scenes. We benchmark several representative neural-based DOA methods on this dataset and propose LightDOA, a lightweight DOA estimation model based on depthwise separable convolutions, specifically designed for mutil-channel input in varying environments. Experimental results show that LightDOA achieves satisfactory accuracy and robustness across various acoustic scenes while maintaining low computational complexity. This study not only highlights the potential of spatial audio synthesized with the assistance of LLMs in advancing robust and efficient DOA estimation research, but also highlights LightDOA as efficient solution for resource-constrained applications. 

---
# Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors for Few-shot Whole Slide Image Classification 

**Authors**: Zhenfeng Zhuang, Fangyu Zhou, Liansheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07941)  

**Abstract**: While Large Language Models (LLMs) are emerging as a promising direction in computational pathology, the substantial computational cost of giga-pixel Whole Slide Images (WSIs) necessitates the use of Multi-Instance Learning (MIL) to enable effective modeling. A key challenge is that pathological tasks typically provide only bag-level labels, while instance-level descriptions generated by LLMs often suffer from bias due to a lack of fine-grained medical knowledge. To address this, we propose that constructing task-specific pathological entity prototypes is crucial for learning generalizable features and enhancing model interpretability. Furthermore, existing vision-language MIL methods often employ unidirectional guidance, limiting cross-modal synergy. In this paper, we introduce a novel approach, Multimodal Prototype-based Multi-Instance Learning, that promotes bidirectional interaction through a balanced information compression scheme. Specifically, we leverage a frozen LLM to generate task-specific pathological entity descriptions, which are learned as text prototypes. Concurrently, the vision branch learns instance-level prototypes to mitigate the model's reliance on redundant data. For the fusion stage, we employ the Stereoscopic Optimal Transport (SOT) algorithm, which is based on a similarity metric, thereby facilitating broader semantic alignment in a higher-dimensional space. We conduct few-shot classification and explainability experiments on three distinct cancer datasets, and the results demonstrate the superior generalization capabilities of our proposed method. 

---
# MURPHY: Multi-Turn GRPO for Self Correcting Code Generation 

**Authors**: Chanakya Ekbote, Vijay Lingam, Behrooz Omidvar-Tehrani, Jun Huan, Sujay Sanghavi, Anoop Deoras, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2511.07833)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful framework for enhancing the reasoning capabilities of large language models (LLMs). However, existing approaches such as Group Relative Policy Optimization (GRPO) and its variants, while effective on reasoning benchmarks, struggle with agentic tasks that require iterative decision-making. We introduce Murphy, a multi-turn reflective optimization framework that extends GRPO by incorporating iterative self-correction during training. By leveraging both quantitative and qualitative execution feedback, Murphy enables models to progressively refine their reasoning across multiple turns. Evaluations on code generation benchmarks with model families such as Qwen and OLMo show that Murphy consistently improves performance, achieving up to a 8% relative gain in pass@1 over GRPO, on similar compute budgets. 

---
# Sparse3DPR: Training-Free 3D Hierarchical Scene Parsing and Task-Adaptive Subgraph Reasoning from Sparse RGB Views 

**Authors**: Haida Feng, Hao Wei, Zewen Xu, Haolin Wang, Chade Li, Yihong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07813)  

**Abstract**: Recently, large language models (LLMs) have been explored widely for 3D scene understanding. Among them, training-free approaches are gaining attention for their flexibility and generalization over training-based methods. However, they typically struggle with accuracy and efficiency in practical deployment. To address the problems, we propose Sparse3DPR, a novel training-free framework for open-ended scene understanding, which leverages the reasoning capabilities of pre-trained LLMs and requires only sparse-view RGB inputs. Specifically, we introduce a hierarchical plane-enhanced scene graph that supports open vocabulary and adopts dominant planar structures as spatial anchors, which enables clearer reasoning chains and more reliable high-level inferences. Furthermore, we design a task-adaptive subgraph extraction method to filter query-irrelevant information dynamically, reducing contextual noise and improving 3D scene reasoning efficiency and accuracy. Experimental results demonstrate the superiority of Sparse3DPR, which achieves a 28.7% EM@1 improvement and a 78.2% speedup compared with ConceptGraphs on the Space3D-Bench. Moreover, Sparse3DPR obtains comparable performance to training-based methods on ScanQA, with additional real-world experiments confirming its robustness and generalization capability. 

---
# Exploring the Underwater World Segmentation without Extra Training 

**Authors**: Bingyu Li, Tao Huo, Da Zhang, Zhiyuan Zhao, Junyu Gao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.07923)  

**Abstract**: Accurate segmentation of marine organisms is vital for biodiversity monitoring and ecological assessment, yet existing datasets and models remain largely limited to terrestrial scenes. To bridge this gap, we introduce \textbf{AquaOV255}, the first large-scale and fine-grained underwater segmentation dataset containing 255 categories and over 20K images, covering diverse categories for open-vocabulary (OV) evaluation. Furthermore, we establish the first underwater OV segmentation benchmark, \textbf{UOVSBench}, by integrating AquaOV255 with five additional underwater datasets to enable comprehensive evaluation. Alongside, we present \textbf{Earth2Ocean}, a training-free OV segmentation framework that transfers terrestrial vision--language models (VLMs) to underwater domains without any additional underwater training. Earth2Ocean consists of two core components: a Geometric-guided Visual Mask Generator (\textbf{GMG}) that refines visual features via self-similarity geometric priors for local structure perception, and a Category-visual Semantic Alignment (\textbf{CSA}) module that enhances text embeddings through multimodal large language model reasoning and scene-aware template construction. Extensive experiments on the UOVSBench benchmark demonstrate that Earth2Ocean achieves significant performance improvement on average while maintaining efficient inference. 

---
# Auto-US: An Ultrasound Video Diagnosis Agent Using Video Classification Framework and LLMs 

**Authors**: Yuezhe Yang, Yiyue Guo, Wenjie Cai, Qingqing Ruan, Siying Wang, Xingbo Dong, Zhe Jin, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07748)  

**Abstract**: AI-assisted ultrasound video diagnosis presents new opportunities to enhance the efficiency and accuracy of medical imaging analysis. However, existing research remains limited in terms of dataset diversity, diagnostic performance, and clinical applicability. In this study, we propose \textbf{Auto-US}, an intelligent diagnosis agent that integrates ultrasound video data with clinical diagnostic text. To support this, we constructed \textbf{CUV Dataset} of 495 ultrasound videos spanning five categories and three organs, aggregated from multiple open-access sources. We developed \textbf{CTU-Net}, which achieves state-of-the-art performance in ultrasound video classification, reaching an accuracy of 86.73\% Furthermore, by incorporating large language models, Auto-US is capable of generating clinically meaningful diagnostic suggestions. The final diagnostic scores for each case exceeded 3 out of 5 and were validated by professional clinicians. These results demonstrate the effectiveness and clinical potential of Auto-US in real-world ultrasound applications. Code and data are available at: this https URL. 

---
# A Self-Improving Architecture for Dynamic Safety in Large Language Models 

**Authors**: Tyler Slater  

**Link**: [PDF](https://arxiv.org/pdf/2511.07645)  

**Abstract**: Context: The integration of Large Language Models (LLMs) into core software systems is accelerating. However, existing software architecture patterns are static, while current safety assurance methods are not scalable, leaving systems vulnerable to novel adversarial threats.
Objective: To design, implement, and evaluate a novel software architecture that enables an AI-driven system to autonomously and continuously adapt its own safety protocols at runtime.
Method: We propose the Self-Improving Safety Framework (SISF), a runtime architecture that couples an unprotected, unaligned base LLM (mistralai/Mistral-7B-v0.1) with a dynamic feedback loop. This loop consists of an AI Adjudicator (GPT-4o) for breach detection and a Policy Synthesis Module (GPT-4 Turbo) that autonomously generates new, generalized safety policies (both heuristic and semantic) in response to failures.
Results: We conducted a dynamic learning evaluation using the 520-prompt AdvBench dataset. The unprotected model was 100% vulnerable. Our SISF, starting from zero policies, demonstrated a clear learning curve: it detected 237 breaches, autonomously synthesized 234 new policies, and reduced the overall Attack Success Rate (ASR) to 45.58%. In a subsequent test on 520 benign prompts, the SISF achieved a 0.00% False Positive Rate (FPR), proving its ability to adapt without compromising user utility.
Conclusion: An architectural approach to AI safety, based on the principles of self-adaptation, is a viable and effective strategy. Our framework demonstrates a practical path towards building more robust, resilient, and scalable AI-driven systems, shifting safety assurance from a static, pre-deployment activity to an automated, runtime process. 

---
# Private-RAG: Answering Multiple Queries with LLMs while Keeping Your Data Private 

**Authors**: Ruihan Wu, Erchi Wang, Zhiyuan Zhang, Yu-Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07637)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving documents from an external corpus at inference time. When this corpus contains sensitive information, however, unprotected RAG systems are at risk of leaking private information. Prior work has introduced differential privacy (DP) guarantees for RAG, but only in single-query settings, which fall short of realistic usage. In this paper, we study the more practical multi-query setting and propose two DP-RAG algorithms. The first, MURAG, leverages an individual privacy filter so that the accumulated privacy loss only depends on how frequently each document is retrieved rather than the total number of queries. The second, MURAG-ADA, further improves utility by privately releasing query-specific thresholds, enabling more precise selection of relevant documents. Our experiments across multiple LLMs and datasets demonstrate that the proposed methods scale to hundreds of queries within a practical DP budget ($\varepsilon\approx10$), while preserving meaningful utility. 

---
# Cortex AISQL: A Production SQL Engine for Unstructured Data 

**Authors**: Paritosh Aggarwal, Bowei Chen, Anupam Datta, Benjamin Han, Boxin Jiang, Nitish Jindal, Zihan Li, Aaron Lin, Pawel Liskowski, Jay Tayade, Dimitris Tsirogiannis, Nathan Wiegand, Weicheng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07663)  

**Abstract**: Snowflake's Cortex AISQL is a production SQL engine that integrates native semantic operations directly into SQL. This integration allows users to write declarative queries that combine relational operations with semantic reasoning, enabling them to query both structured and unstructured data effortlessly. However, making semantic operations efficient at production scale poses fundamental challenges. Semantic operations are more expensive than traditional SQL operations, possess distinct latency and throughput characteristics, and their cost and selectivity are unknown during query compilation. Furthermore, existing query engines are not designed to optimize semantic operations. The AISQL query execution engine addresses these challenges through three novel techniques informed by production deployment data from Snowflake customers. First, AI-aware query optimization treats AI inference cost as a first-class optimization objective, reasoning about large language model (LLM) cost directly during query planning to achieve 2-8$\times$ speedups. Second, adaptive model cascades reduce inference costs by routing most rows through a fast proxy model while escalating uncertain cases to a powerful oracle model, achieving 2-6$\times$ speedups while maintaining 90-95% of oracle model quality. Third, semantic join query rewriting lowers the quadratic time complexity of join operations to linear through reformulation as multi-label classification tasks, achieving 15-70$\times$ speedups with often improved prediction quality. AISQL is deployed in production at Snowflake, where it powers diverse customer workloads across analytics, search, and content understanding. 

---
# SemanticForge: Repository-Level Code Generation through Semantic Knowledge Graphs and Constraint Satisfaction 

**Authors**: Wuyang Zhang, Chenkai Zhang, Zhen Luo, Jianming Ma, Wangming Yuan, Chuqiao Gu, Chenwei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2511.07584)  

**Abstract**: Large language models (LLMs) have transformed software development by enabling automated code generation, yet they frequently suffer from systematic errors that limit practical deployment. We identify two critical failure modes: \textit{logical hallucination} (incorrect control/data-flow reasoning) and \textit{schematic hallucination} (type mismatches, signature violations, and architectural inconsistencies). These errors stem from the absence of explicit, queryable representations of repository-wide semantics.
This paper presents \textbf{SemanticForge}, which introduces four fundamental algorithmic advances for semantically-aware code generation: (1) a novel automatic reconciliation algorithm for dual static-dynamic knowledge graphs, unifying compile-time and runtime program semantics; (2) a neural approach that learns to generate structured graph queries from natural language, achieving 73\% precision versus 51\% for traditional retrieval; (3) a novel beam search algorithm with integrated SMT solving, enabling real-time constraint verification during generation rather than post-hoc validation; and (4) an incremental maintenance algorithm that updates knowledge graphs in $O(|\Delta R| \cdot \log n)$ time while maintaining semantic equivalence. 

---
# Dynamic Stability of LLM-Generated Code 

**Authors**: Prateek Rajput, Abdoul Aziz Bonkoungou, Yewei Song, Abdoul Kader Kabore, Iyiola E. Olatunji, Jacques Klein, Tegewende Bissyande  

**Link**: [PDF](https://arxiv.org/pdf/2511.07463)  

**Abstract**: Current evaluations of LLMs for code generation emphasize functional correctness, overlooking the fact that functionally correct solutions can differ significantly in algorithmic complexity. For instance, an $(O(n^2))$ versus $(O(n \log n))$ sorting algorithm may yield similar output but incur vastly different performance costs in production. This discrepancy reveals a critical limitation in current evaluation methods: they fail to capture the behavioral and performance diversity among correct solutions. To address this, we introduce a principled framework for evaluating the dynamic stability of generated code. We propose two metrics derived from opcode distributions: Static Canonical Trace Divergence (SCTD), which captures algorithmic structure diversity across generated solutions, and Dynamic Canonical Trace Divergence (DCTD), which quantifies runtime behavioral variance. Their ratio, the Behavioral Expression Factor (BEF), serves as a diagnostic signal: it indicates critical runtime instability when BEF $\ll$ 1 and functional redundancy when BEF $\gg$ 1. Empirical results on BigOBench and CodeContests show that state-of-the-art LLMs exhibit significant algorithmic variance even among functionally correct outputs. Notably, increasing sampling temperature improves pass@1 rates but degrades stability, revealing an unrecognized trade-off: searching for correct solutions in diverse output spaces introduces a "penalty of instability" between correctness and behavioral consistency. Our findings call for stability-aware objectives in code generation and new benchmarks with asymptotic test cases for robust, real-world LLM evaluation. 

---
# AudAgent: Automated Auditing of Privacy Policy Compliance in AI Agents 

**Authors**: Ye Zheng, Yidan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07441)  

**Abstract**: AI agents can autonomously perform tasks and, often without explicit user consent, collect or disclose users' sensitive local data, which raises serious privacy concerns. Although AI agents' privacy policies may describe their intended data practices, there remains limited transparency and accountability about whether runtime behavior matches those policies. To close this gap, we introduce AudAgent, a visual framework that continuously monitors AI agents' data practices in real time and guards compliance with stated privacy policies.
AudAgent consists of four components for automated privacy auditing of AI agents. (i) Policy parsing: an ensemble of LLMs translates natural-language privacy policies into a structured privacy-policy model, where cross-LLM voting guarantees confidence of the parsing results. (ii) Runtime annotation: a lightweight Presidio-based analyzer detects sensitive data and annotates how the data is used based on the context of the AI agent's operations and the privacy-policy model. (iii) Compliance auditing: ontology alignment and automata-based evaluation connect the policy model with runtime annotations, enabling on-the-fly compliance checks between the natural-language policy and observed unordered data practices of AI agents. (iv) User interface: a platform-independent implementation visualizes the real-time execution trace of AI agents along with potential privacy risks detected during auditing, providing user-friendly transparency and accountability.
In addition to common formatted privacy policies, AudAgent also supports user-defined policies for fine-grained control and customization. We evaluate AudAgent on AI agents built upon mainstream programming frameworks such as AutoGen, experiments show that AudAgent effectively identifies potential privacy policy violations in real time. 

---
# An Evaluation of LLMs Inference on Popular Single-board Computers 

**Authors**: Tung, Nguyen, Tuyen Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07425)  

**Abstract**: The growing demand for on-device large language model (LLM) inference is driving interest in deploying lightweight, cost-effective AI solutions on edge hardware. Single-board computers (SBCs) such as the Raspberry Pi and Orange Pi offer a promising platform for localized, privacy-preserving inference-but remain underexplored in the context of LLM workloads. In this work, we benchmark the performance of 25 quantized open-source LLMs across three SBCs-Raspberry Pi 4, Raspberry Pi 5, and Orange Pi 5 Pro-using two inference runtimes: Ollama and Llamafile. We evaluate generation throughput, memory usage, and power consumption under varying CPU configurations, using multiple prompt types to simulate realistic workloads. Our results show that SBCs can reliably support models up to 1.5B parameters, with Llamafile achieving up to 4x higher throughput and 30-40% lower power usage than Ollama. We identify architecture-specific bottlenecks, highlight runtime-level trade-offs, and provide practical deployment recommendations. This study offers the first broad evaluation of LLM inference on SBCs, bridging the gap between high-performance language models and affordable edge computing. 

---
# Advancing mathematics research with large language models 

**Authors**: Lisa Carbone  

**Link**: [PDF](https://arxiv.org/pdf/2511.07420)  

**Abstract**: The main drawback of using generative AI for advanced mathematics via Large Language Models (LLMs) is that they are probabilistic pattern-matchers, not logical reasoning engines. However, LLMs can pick up on patterns in higher mathematics that are difficult for humans to see. By putting the design of LLMs to their advantage, mathematicians may use them as powerful interactive assistants that can carry out laborious tasks, generate and debug code, check examples, formulate conjectures and more. We discuss how LLMs can be used to advance mathematics research by careful use of prompt engineering. We also discuss the integration of LLMs with Computer Algebra Systems and formal proof assistants such as Lean. 

---
# Knowledge-Guided Textual Reasoning for Explainable Video Anomaly Detection via LLMs 

**Authors**: Hari Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.07429)  

**Abstract**: We introduce Text-based Explainable Video Anomaly Detection (TbVAD), a language-driven framework for weakly supervised video anomaly detection that performs anomaly detection and explanation entirely within the textual domain. Unlike conventional WSVAD models that rely on explicit visual features, TbVAD represents video semantics through language, enabling interpretable and knowledge-grounded reasoning. The framework operates in three stages: (1) transforming video content into fine-grained captions using a vision-language model, (2) constructing structured knowledge by organizing the captions into four semantic slots (action, object, context, environment), and (3) generating slot-wise explanations that reveal which semantic factors contribute most to the anomaly decision. We evaluate TbVAD on two public benchmarks, UCF-Crime and XD-Violence, demonstrating that textual knowledge reasoning provides interpretable and reliable anomaly detection for real-world surveillance scenarios. 

---
