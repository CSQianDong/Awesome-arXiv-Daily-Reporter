# Persuade Me if You Can: A Framework for Evaluating Persuasion Effectiveness and Susceptibility Among Large Language Models 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2503.01829)  

**Abstract**: Large Language Models (LLMs) demonstrate persuasive capabilities that rival human-level persuasion. While these capabilities can be used for social good, they also present risks of potential misuse. Moreover, LLMs' susceptibility to persuasion raises concerns about alignment with ethical principles. To study these dynamics, we introduce Persuade Me If You Can (PMIYC), an automated framework for evaluating persuasion through multi-agent interactions. Here, Persuader agents engage in multi-turn conversations with the Persuadee agents, allowing us to measure LLMs' persuasive effectiveness and their susceptibility to persuasion. We conduct comprehensive evaluations across diverse LLMs, ensuring each model is assessed against others in both subjective and misinformation contexts. We validate the efficacy of our framework through human evaluations and show alignment with prior work. PMIYC offers a scalable alternative to human annotation for studying persuasion in LLMs. Through PMIYC, we find that Llama-3.3-70B and GPT-4o exhibit similar persuasive effectiveness, outperforming Claude 3 Haiku by 30%. However, GPT-4o demonstrates over 50% greater resistance to persuasion for misinformation compared to Llama-3.3-70B. These findings provide empirical insights into the persuasive dynamics of LLMs and contribute to the development of safer AI systems. 

---
# Rotary Outliers and Rotary Offset Features in Large Language Models 

**Authors**: André Jonasson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01832)  

**Abstract**: Transformer-based Large Language Models (LLMs) rely on positional encodings to provide sequence position information to their attention mechanism. Rotary Positional Encodings (RoPE), which encode relative position by rotating queries and keys, have become widely used in modern LLMs. We study the features and patterns that emerge in queries and keys when using rotary embeddings. Our analysis reveals consistent patterns within the same model across layers and attention heads and across different models and architectures. We present and apply analysis techniques and show how the queries and keys use RoPE to construct various attention patterns, including attention sinks. We find and analyze outliers across models in queries and keys and find that they are likely to be found in rotary features with partial cycles. We derive bounds that tell us what rotary frequencies are likely to be selected as outlier features and at what minimum angle the query-key rotary pairs in these features tend to be above and verify the bounds empirically with models of significant architectural differences. 

---
# EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test 

**Authors**: Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01840)  

**Abstract**: The sequential nature of modern LLMs makes them expensive and slow, and speculative sampling has proven to be an effective solution to this problem. Methods like EAGLE perform autoregression at the feature level, reusing top-layer features from the target model to achieve better results than vanilla speculative sampling. A growing trend in the LLM community is scaling up training data to improve model intelligence without increasing inference costs. However, we observe that scaling up data provides limited improvements for EAGLE. We identify that this limitation arises from EAGLE's feature prediction constraints. In this paper, we introduce EAGLE-3, which abandons feature prediction in favor of direct token prediction and replaces reliance on top-layer features with multi-layer feature fusion via a technique named training-time test. These improvements significantly enhance performance and enable the draft model to fully benefit from scaling up training data. Our experiments include both chat models and reasoning models, evaluated on five tasks. The results show that EAGLE-3 achieves a speedup ratio up to 6.5x, with about 1.4x improvement over EAGLE-2. The code is available at this https URL. 

---
# $\texttt{SEM-CTRL}$: Semantically Controlled Decoding 

**Authors**: Mohammad Albinhassan, Pranava Madhyastha, Alessandra Russo  

**Link**: [PDF](https://arxiv.org/pdf/2503.01804)  

**Abstract**: Ensuring both syntactic and semantic correctness in Large Language Model (LLM) outputs remains a significant challenge, despite being critical for real-world deployment. In this paper, we introduce $\texttt{SEM-CTRL}$, a unified approach that enforces rich context-sensitive constraints and task- and instance-specific semantics directly on an LLM decoder. Our approach integrates token-level MCTS, which is guided by specific syntactic and semantic constraints. The constraints over the desired outputs are expressed using Answer Set Grammars -- a logic-based formalism that generalizes context-sensitive grammars while incorporating background knowledge to represent task-specific semantics. We show that our approach guarantees correct completions for any off-the-shelf LLM without the need for fine-tuning. We evaluate $\texttt{SEM-CTRL}$ on a range of tasks, including synthetic grammar synthesis, combinatorial reasoning, and planning. Our results demonstrate that $\texttt{SEM-CTRL}$ allows small pre-trained LLMs to efficiently outperform larger variants and state-of-the-art reasoning models (e.g., o1-preview) while simultaneously guaranteeing solution correctness. 

---
# CrowdSelect: Synthetic Instruction Data Selection with Multi-LLM Wisdom 

**Authors**: Yisen Li, Lingfeng Yang, Wenxuan Shen, Pan Zhou, Yao Wan, Weiwei Lin, Dongping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01836)  

**Abstract**: Distilling advanced Large Language Models' instruction-following capabilities into smaller models using a selected subset has become a mainstream approach in model training. While existing synthetic instruction data selection strategies rely mainly on single-dimensional signals (i.e., reward scores, model perplexity), they fail to capture the complexity of instruction-following across diverse fields. Therefore, we investigate more diverse signals to capture comprehensive instruction-response pair characteristics and propose three foundational metrics that leverage Multi-LLM wisdom, informed by (1) diverse LLM responses and (2) reward model assessment. Building upon base metrics, we propose CrowdSelect, an integrated metric incorporating a clustering-based approach to maintain response diversity. Our comprehensive experiments demonstrate that our foundation metrics consistently improve performance across 4 base models on MT-bench and Arena-Hard. CrowdSelect, efficiently incorporating all metrics, achieves state-of-the-art performance in both Full and LoRA fine-tuning, showing improvements of 4.81% on Arena-Hard and 11.1% on MT-bench with Llama-3.2-3b-instruct. We hope our findings will bring valuable insights for future research in this direction. Code are available at this https URL. 

---
# Can (A)I Change Your Mind? 

**Authors**: Miriam Havin, Timna Wharton Kleinman, Moran Koren, Yaniv Dover, Ariel Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2503.01844)  

**Abstract**: The increasing integration of large language model (LLM) based conversational agents into everyday life raises critical cognitive and social questions about their potential to influence human opinions. Although previous studies have shown that LLM-based agents can generate persuasive content, these typically involve controlled, English-language settings. Addressing this, our preregistered study explored LLM's persuasive capabilities in more ecological, unconstrained scenarios, examining both static (written paragraphs) and dynamic (conversations via Telegram) interaction types. Conducted entirely in Hebrew with 200 participants, the study assessed the persuasive effects of both LLM and human interlocutors on controversial civil policy topics. Results indicated that participants adopted LLM and human perspectives similarly, with significant opinion changes evident across all conditions, regardless of interlocutor type or interaction mode. Confidence levels increased significantly in most scenarios, except in static LLM interactions. These findings demonstrate LLM-based agents' robust persuasive capabilities across diverse sources and settings, highlighting their potential impact on shaping public opinions. 

---
# Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models 

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani  

**Link**: [PDF](https://arxiv.org/pdf/2503.01781)  

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at this https URL. 

---
# Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models 

**Authors**: Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01763)  

**Abstract**: Tool learning aims to augment large language models (LLMs) with diverse tools, enabling them to act as agents for solving practical tasks. Due to the limited context length of tool-using LLMs, adopting information retrieval (IR) models to select useful tools from large toolsets is a critical initial step. However, the performance of IR models in tool retrieval tasks remains underexplored and unclear. Most tool-use benchmarks simplify this step by manually pre-annotating a small set of relevant tools for each task, which is far from the real-world scenarios. In this paper, we propose ToolRet, a heterogeneous tool retrieval benchmark comprising 7.6k diverse retrieval tasks, and a corpus of 43k tools, collected from existing datasets. We benchmark six types of models on ToolRet. Surprisingly, even the models with strong performance in conventional IR benchmarks, exhibit poor performance on ToolRet. This low retrieval quality degrades the task pass rate of tool-use LLMs. As a further step, we contribute a large-scale training dataset with over 200k instances, which substantially optimizes the tool retrieval ability of IR models. 

---
# Automated Annotation of Evolving Corpora for Augmenting Longitudinal Network Data: A Framework Integrating Large Language Models and Expert Knowledge 

**Authors**: Xiao Liu, Zirui Wu, Jiayi Li, Zhicheng Shao, Xun Pang, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01672)  

**Abstract**: Longitudinal network data are essential for analyzing political, economic, and social systems and processes. In political science, these datasets are often generated through human annotation or supervised machine learning applied to evolving corpora. However, as semantic contexts shift over time, inferring dynamic interaction types on emerging issues among a diverse set of entities poses significant challenges, particularly in maintaining timely and consistent annotations. This paper presents the Expert-Augmented LLM Annotation (EALA) approach, which leverages Large Language Models (LLMs) in combination with historically annotated data and expert-constructed codebooks to extrapolate and extend datasets into future periods. We evaluate the performance and reliability of EALA using a dataset of climate negotiations. Our findings demonstrate that EALA effectively predicts nuanced interactions between negotiation parties and captures the evolution of topics over time. At the same time, we identify several limitations inherent to LLM-based annotation, highlighting areas for further improvement. Given the wide availability of codebooks and annotated datasets, EALA holds substantial promise for advancing research in political science and beyond. 

---
# Word Form Matters: LLMs' Semantic Reconstruction under Typoglycemia 

**Authors**: Chenxi Wang, Tianle Gu, Zhongyu Wei, Lang Gao, Zirui Song, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01714)  

**Abstract**: Human readers can efficiently comprehend scrambled words, a phenomenon known as Typoglycemia, primarily by relying on word form; if word form alone is insufficient, they further utilize contextual cues for interpretation. While advanced large language models (LLMs) exhibit similar abilities, the underlying mechanisms remain unclear. To investigate this, we conduct controlled experiments to analyze the roles of word form and contextual information in semantic reconstruction and examine LLM attention patterns. Specifically, we first propose SemRecScore, a reliable metric to quantify the degree of semantic reconstruction, and validate its effectiveness. Using this metric, we study how word form and contextual information influence LLMs' semantic reconstruction ability, identifying word form as the core factor in this process. Furthermore, we analyze how LLMs utilize word form and find that they rely on specialized attention heads to extract and process word form information, with this mechanism remaining stable across varying levels of word scrambling. This distinction between LLMs' fixed attention patterns primarily focused on word form and human readers' adaptive strategy in balancing word form and contextual information provides insights into enhancing LLM performance by incorporating human-like, context-aware mechanisms. 

---
# Generate, Discriminate, Evolve: Enhancing Context Faithfulness via Fine-Grained Sentence-Level Self-Evolution 

**Authors**: Kun Li, Tianhua Zhang, Yunxiang Li, Hongyin Luo, Abdalla Moustafa, Xixin Wu, James Glass, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01695)  

**Abstract**: Improving context faithfulness in large language models is essential for developing trustworthy retrieval augmented generation systems and mitigating hallucinations, especially in long-form question answering (LFQA) tasks or scenarios involving knowledge conflicts. Existing methods either intervene LLMs only at inference without addressing their inherent limitations or overlook the potential for self-improvement. In this paper, we introduce GenDiE (Generate, Discriminate, Evolve), a novel self-evolving framework that enhances context faithfulness through fine-grained sentence-level optimization. GenDiE combines both generative and discriminative training, equipping LLMs with self-generation and self-scoring capabilities to facilitate iterative self-evolution. This supports both data construction for model alignment and score-guided search during inference. Furthermore, by treating each sentence in a response as an independent optimization unit, GenDiE effectively addresses the limitations of previous approaches that optimize at the holistic answer level, which may miss unfaithful details. Experiments on ASQA (in-domain LFQA) and ConFiQA (out-of-domain counterfactual QA) datasets demonstrate that GenDiE surpasses various baselines in both faithfulness and correctness, and exhibits robust performance for domain adaptation. 

---
# DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation 

**Authors**: Eliya Habba, Ofir Arviv, Itay Itzhak, Yotam Perlitz, Elron Bandel, Leshem Choshen, Michal Shmueli-Scheuer, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2503.01622)  

**Abstract**: Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices. We present DOVE (Dataset Of Variation Evaluation) a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from an holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations. DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation.
Browse the data, contribute, and more: this https URL 

---
# Attention Condensation via Sparsity Induced Regularized Training 

**Authors**: Eli Sason, Darya Frolova, Boris Nazarov, Felix Goldberd  

**Link**: [PDF](https://arxiv.org/pdf/2503.01564)  

**Abstract**: As the context window expands, self-attention increasingly dominates the transformer's inference time. Therefore, accelerating attention computation while minimizing performance degradation is essential for the efficient deployment of Large Language Models (LLMs). In this study we extend a theoretical framework of attention sparsity in LLMs. A customized loss function is designed to enforce the sparsity by restricting the number of top elements in the attention matrix. We perform an initial set of evaluations with GPT-2 to show the effectiveness of our sparsification approach. The attention matrices of the models trained with the proposed loss are both sparse and effective in capturing relevant input dependencies. We now continue working to demonstrate the value of our approach on larger models and different architectures. 

---
# Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the Lens of Summarization 

**Authors**: Siya Qi, Rui Cao, Yulan He, Zheng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01670)  

**Abstract**: With the rapid development of large language models (LLMs), LLM-as-a-judge has emerged as a widely adopted approach for text quality evaluation, including hallucination evaluation. While previous studies have focused exclusively on single-context evaluation (e.g., discourse faithfulness or world factuality), real-world hallucinations typically involve mixed contexts, which remains inadequately evaluated. In this study, we use summarization as a representative task to comprehensively evaluate LLMs' capability in detecting mixed-context hallucinations, specifically distinguishing between factual and non-factual hallucinations. Through extensive experiments across direct generation and retrieval-based models of varying scales, our main observations are: (1) LLMs' intrinsic knowledge introduces inherent biases in hallucination evaluation; (2) These biases particularly impact the detection of factual hallucinations, yielding a significant performance bottleneck; (3) The fundamental challenge lies in effective knowledge utilization, balancing between LLMs' intrinsic knowledge and external context for accurate mixed-context hallucination evaluation. 

---
# Evaluation and Facilitation of Online Discussions in the LLM Era: A Survey 

**Authors**: Katerina Korre, Dimitris Tsirmpas, Nikos Gkoumas, Emma Cabalé, Dionysis Kontarinis, Danai Myrtzani, Theodoros Evgeniou, Ion Androutsopoulos, John Pavlopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.01513)  

**Abstract**: We present a survey of methods for assessing and enhancing the quality of online discussions, focusing on the potential of Large Language Models (LLMs). While online discourses aim, at least in theory, to foster mutual understanding, they often devolve into harmful exchanges, such as hate speech, threatening social cohesion and democratic values. Recent advancements in LLMs enable facilitation agents that not only moderate content, but also actively improve the quality of interactions. Our survey synthesizes ideas from Natural Language Processing (NLP) and Social Sciences to provide (a) a new taxonomy on discussion quality evaluation, (b) an overview of intervention and facilitation strategies, along with a new taxonomy on conversation facilitation datasets, (c) an LLM-oriented roadmap of good practices and future research directions, from technological and societal perspectives. 

---
# Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language 

**Authors**: Xi Chen, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01539)  

**Abstract**: The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, and DeepSeek-v2.5 in identifying implicit toxic language, compared to both direct prompting and Chain-of-Thought. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors. 

---
# Liger: Linearizing Large Language Models to Gated Recurrent Structures 

**Authors**: Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01496)  

**Abstract**: Transformers with linear recurrent modeling offer linear-time training and constant-memory inference. Despite their demonstrated efficiency and performance, pretraining such non-standard architectures from scratch remains costly and risky. The linearization of large language models (LLMs) transforms pretrained standard models into linear recurrent structures, enabling more efficient deployment. However, current linearization methods typically introduce additional feature map modules that require extensive fine-tuning and overlook the gating mechanisms used in state-of-the-art linear recurrent models. To address these issues, this paper presents Liger, short for Linearizing LLMs to gated recurrent structures. Liger is a novel approach for converting pretrained LLMs into gated linear recurrent models without adding extra parameters. It repurposes the pretrained key matrix weights to construct diverse gating mechanisms, facilitating the formation of various gated recurrent structures while avoiding the need to train additional components from scratch. Using lightweight fine-tuning with Low-Rank Adaptation (LoRA), Liger restores the performance of the linearized gated recurrent models to match that of the original LLMs. Additionally, we introduce Liger Attention, an intra-layer hybrid attention mechanism, which significantly recovers 93\% of the Transformer-based LLM at 0.02\% pre-training tokens during the linearization process, achieving competitive results across multiple benchmarks, as validated on models ranging from 1B to 8B parameters. Code is available at this https URL. 

---
# When an LLM is apprehensive about its answers -- and when its uncertainty is justified 

**Authors**: Petr Sychev, Andrey Goncharov, Daniil Vyazhev, Edvard Khalafyan, Alexey Zaytsev  

**Link**: [PDF](https://arxiv.org/pdf/2503.01688)  

**Abstract**: Uncertainty estimation is crucial for evaluating Large Language Models (LLMs), particularly in high-stakes domains where incorrect answers result in significant consequences. Numerous approaches consider this problem, while focusing on a specific type of uncertainty, ignoring others. We investigate what estimates, specifically token-wise entropy and model-as-judge (MASJ), would work for multiple-choice question-answering tasks for different question topics. Our experiments consider three LLMs: Phi-4, Mistral, and Qwen of different sizes from 1.5B to 72B and $14$ topics. While MASJ performs similarly to a random error predictor, the response entropy predicts model error in knowledge-dependent domains and serves as an effective indicator of question difficulty: for biology ROC AUC is $0.73$. This correlation vanishes for the reasoning-dependent domain: for math questions ROC-AUC is $0.55$. More principally, we found out that the entropy measure required a reasoning amount. Thus, data-uncertainty related entropy should be integrated within uncertainty estimates frameworks, while MASJ requires refinement. Moreover, existing MMLU-Pro samples are biased, and should balance required amount of reasoning for different subdomains to provide a more fair assessment of LLMs performance. 

---
# Llama-3.1-Sherkala-8B-Chat: An Open Large Language Model for Kazakh 

**Authors**: Fajri Koto, Rituraj Joshi, Nurdaulet Mukhituly, Yuxia Wang, Zhuohan Xie, Rahul Pal, Daniil Orel, Parvez Mullah, Diana Turmakhan, Maiya Goloburda, Mohammed Kamran, Samujjwal Ghosh, Bokang Jia, Jonibek Mansurov, Mukhammed Togmanov, Debopriyo Banerjee, Nurkhan Laiyk, Akhmed Sakip, Xudong Han, Ekaterina Kochmar, Alham Fikri Aji, Aaryamonvikram Singh, Alok Anil Jadhav, Satheesh Katipomu, Samta Kamboj, Monojit Choudhury, Gurpreet Gosal, Gokul Ramakrishnan, Biswajit Mishra, Sarath Chandran, Avraham Sheinin, Natalia Vassilieva, Neha Sengupta, Larry Murray, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2503.01493)  

**Abstract**: Llama-3.1-Sherkala-8B-Chat, or Sherkala-Chat (8B) for short, is a state-of-the-art instruction-tuned open generative large language model (LLM) designed for Kazakh. Sherkala-Chat (8B) aims to enhance the inclusivity of LLM advancements for Kazakh speakers. Adapted from the LLaMA-3.1-8B model, Sherkala-Chat (8B) is trained on 45.3B tokens across Kazakh, English, Russian, and Turkish. With 8 billion parameters, it demonstrates strong knowledge and reasoning abilities in Kazakh, significantly outperforming existing open Kazakh and multilingual models of similar scale while achieving competitive performance in English. We release Sherkala-Chat (8B) as an open-weight instruction-tuned model and provide a detailed overview of its training, fine-tuning, safety alignment, and evaluation, aiming to advance research and support diverse real-world applications. 

---
# SePer: Measure Retrieval Utility Through The Lens Of Semantic Perplexity Reduction 

**Authors**: Lu Dai, Yijie Xu, Jinhui Ye, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01478)  

**Abstract**: Large Language Models (LLMs) have demonstrated improved generation performance by incorporating externally retrieved knowledge, a process known as retrieval-augmented generation (RAG). Despite the potential of this approach, existing studies evaluate RAG effectiveness by 1) assessing retrieval and generation components jointly, which obscures retrieval's distinct contribution, or 2) examining retrievers using traditional metrics such as NDCG, which creates a gap in understanding retrieval's true utility in the overall generation process. To address the above limitations, in this work, we introduce an automatic evaluation method that measures retrieval quality through the lens of information gain within the RAG framework. Specifically, we propose Semantic Perplexity (SePer), a metric that captures the LLM's internal belief about the correctness of the retrieved information. We quantify the utility of retrieval by the extent to which it reduces semantic perplexity post-retrieval. Extensive experiments demonstrate that SePer not only aligns closely with human preferences but also offers a more precise and efficient evaluation of retrieval utility across diverse RAG scenarios. 

---
# Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding 

**Authors**: Yiming Wang, Pei Zhang, Siyuan Huang, Baosong Yang, Zhuosheng Zhang, Fei Huang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01422)  

**Abstract**: Test-time scaling improves large language model performance by adding extra compute during decoding. Best-of-N (BoN) sampling serves as a common scaling technique, broadening the search space for finding better solutions from the model distribution. However, traditional BoN requires N full generations, leading to high GPU memory overhead and time latency. Moreover, some methods depend on reward models, adding computational cost and limiting domain generalization.
In this paper, we propose Self-Truncation Best-of-N (ST-BoN), a novel decoding method that avoids fully generating all samplings and eliminates the need for reward models. ST-BoN introduces early sampling consistency to estimate the most promising sample, truncating suboptimal ones to free memory and accelerate inference. This pushes the sampling-efficient test-time scaling. Compared to traditional BoN, ST-BoN can reduce dynamic GPU memory overhead by over 90% and time latency by 50%, while achieving comparable or even better performance across reasoning and open-ended domains. 

---
# SRAG: Structured Retrieval-Augmented Generation for Multi-Entity Question Answering over Wikipedia Graph 

**Authors**: Teng Lin, Yizhang Zhu, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01346)  

**Abstract**: Multi-entity question answering (MEQA) poses significant challenges for large language models (LLMs), which often struggle to consolidate scattered information across multiple documents. An example question might be "What is the distribution of IEEE Fellows among various fields of study?", which requires retrieving information from diverse sources e.g., Wikipedia pages. The effectiveness of current retrieval-augmented generation (RAG) methods is limited by the LLMs' capacity to aggregate insights from numerous pages. To address this gap, this paper introduces a structured RAG (SRAG) framework that systematically organizes extracted entities into relational tables (e.g., tabulating entities with schema columns like "name" and "field of study") and then apply table-based reasoning techniques. Our approach decouples retrieval and reasoning, enabling LLMs to focus on structured data analysis rather than raw text aggregation. Extensive experiments on Wikipedia-based multi-entity QA tasks demonstrate that SRAG significantly outperforms state-of-the-art long-context LLMs and RAG solutions, achieving a 29.6% improvement in accuracy. The results underscore the efficacy of structuring unstructured data to enhance LLMs' reasoning capabilities. 

---
# Q-NL Verifier: Leveraging Synthetic Data for Robust Knowledge Graph Question Answering 

**Authors**: Tim Schwabe, Louisa Siebel, Patrik Valach, Maribel Acosta  

**Link**: [PDF](https://arxiv.org/pdf/2503.01385)  

**Abstract**: Question answering (QA) requires accurately aligning user questions with structured queries, a process often limited by the scarcity of high-quality query-natural language (Q-NL) pairs. To overcome this, we present Q-NL Verifier, an approach to generating high-quality synthetic pairs of queries and NL translations. Our approach relies on large language models (LLMs) to generate semantically precise natural language paraphrases of structured queries. Building on these synthetic Q-NL pairs, we introduce a learned verifier component that automatically determines whether a generated paraphrase is semantically equivalent to the original query. Our experiments with the well-known LC-QuAD 2.0 benchmark show that Q-NL Verifier generalizes well to paraphrases from other models and even human-authored translations. Our approach strongly aligns with human judgments across varying query complexities and outperforms existing NLP metrics in assessing semantic correctness. We also integrate the verifier into QA pipelines, showing that verifier-filtered synthetic data has significantly higher quality in terms of translation correctness and enhances NL to Q translation accuracy. Lastly, we release an updated version of the LC-QuAD 2.0 benchmark containing our synthetic Q-NL pairs and verifier scores, offering a new resource for robust and scalable QA. 

---
# Explainable Depression Detection in Clinical Interviews with Personalized Retrieval-Augmented Generation 

**Authors**: Linhai Zhang, Ziyang Gao, Deyu Zhou, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.01315)  

**Abstract**: Depression is a widespread mental health disorder, and clinical interviews are the gold standard for assessment. However, their reliance on scarce professionals highlights the need for automated detection. Current systems mainly employ black-box neural networks, which lack interpretability, which is crucial in mental health contexts. Some attempts to improve interpretability use post-hoc LLM generation but suffer from hallucination. To address these limitations, we propose RED, a Retrieval-augmented generation framework for Explainable depression Detection. RED retrieves evidence from clinical interview transcripts, providing explanations for predictions. Traditional query-based retrieval systems use a one-size-fits-all approach, which may not be optimal for depression detection, as user backgrounds and situations vary. We introduce a personalized query generation module that combines standard queries with user-specific background inferred by LLMs, tailoring retrieval to individual contexts. Additionally, to enhance LLM performance in social intelligence, we augment LLMs by retrieving relevant knowledge from a social intelligence datastore using an event-centric retriever. Experimental results on the real-world benchmark demonstrate RED's effectiveness compared to neural networks and LLM-based baselines. 

---
# Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs 

**Authors**: Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, Noah D. Goodman  

**Link**: [PDF](https://arxiv.org/pdf/2503.01307)  

**Abstract**: Test-time inference has emerged as a powerful paradigm for enabling language models to ``think'' longer and more carefully about complex challenges, much like skilled human experts. While reinforcement learning (RL) can drive self-improvement in language models on verifiable tasks, some models exhibit substantial gains while others quickly plateau. For instance, we find that Qwen-2.5-3B far exceeds Llama-3.2-3B under identical RL training for the game of Countdown. This discrepancy raises a critical question: what intrinsic properties enable effective self-improvement? We introduce a framework to investigate this question by analyzing four key cognitive behaviors -- verification, backtracking, subgoal setting, and backward chaining -- that both expert human problem solvers and successful language models employ. Our study reveals that Qwen naturally exhibits these reasoning behaviors, whereas Llama initially lacks them. In systematic experimentation with controlled behavioral datasets, we find that priming Llama with examples containing these reasoning behaviors enables substantial improvements during RL, matching or exceeding Qwen's performance. Importantly, the presence of reasoning behaviors, rather than correctness of answers, proves to be the critical factor -- models primed with incorrect solutions containing proper reasoning patterns achieve comparable performance to those trained on correct solutions. Finally, leveraging continued pretraining with OpenWebMath data, filtered to amplify reasoning behaviors, enables the Llama model to match Qwen's self-improvement trajectory. Our findings establish a fundamental relationship between initial reasoning behaviors and the capacity for improvement, explaining why some language models effectively utilize additional computation while others plateau. 

---
# PROPER: A Progressive Learning Framework for Personalized Large Language Models with Group-Level Adaptation 

**Authors**: Linhai Zhang, Jialong Wu, Deyu Zhou, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.01303)  

**Abstract**: Personalized large language models (LLMs) aim to tailor their outputs to user preferences. Recent advances in parameter-efficient fine-tuning (PEFT) methods have highlighted the effectiveness of adapting population-level LLMs to personalized LLMs by fine-tuning user-specific parameters with user history. However, user data is typically sparse, making it challenging to adapt LLMs to specific user patterns. To address this challenge, we propose PROgressive PERsonalization (PROPER), a novel progressive learning framework inspired by meso-level theory in social science. PROPER bridges population-level and user-level models by grouping users based on preferences and adapting LLMs in stages. It combines a Mixture-of-Experts (MoE) structure with Low Ranked Adaptation (LoRA), using a user-aware router to assign users to appropriate groups automatically. Additionally, a LoRA-aware router is proposed to facilitate the integration of individual user LoRAs with group-level LoRAs. Experimental results show that PROPER significantly outperforms SOTA models across multiple tasks, demonstrating the effectiveness of our approach. 

---
# Enhancing Non-English Capabilities of English-Centric Large Language Models through Deep Supervision Fine-Tuning 

**Authors**: Wenshuai Huo, Xiaocheng Feng, Yichong Huang, Chengpeng Fu, Baohang Li, Yangfan Ye, Zhirui Zhang, Dandan Tu, Duyu Tang, Yunfei Lu, Hui Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.01275)  

**Abstract**: Large language models (LLMs) have demonstrated significant progress in multilingual language understanding and generation. However, due to the imbalance in training data, their capabilities in non-English languages are limited. Recent studies revealed the English-pivot multilingual mechanism of LLMs, where LLMs implicitly convert non-English queries into English ones at the bottom layers and adopt English for thinking at the middle layers. However, due to the absence of explicit supervision for cross-lingual alignment in the intermediate layers of LLMs, the internal representations during these stages may become inaccurate. In this work, we introduce a deep supervision fine-tuning method (DFT) that incorporates additional supervision in the internal layers of the model to guide its workflow. Specifically, we introduce two training objectives on different layers of LLMs: one at the bottom layers to constrain the conversion of the target language into English, and another at the middle layers to constrain reasoning in English. To effectively achieve the guiding purpose, we designed two types of supervision signals: logits and feature, which represent a stricter constraint and a relatively more relaxed guidance. Our method guides the model to not only consider the final generated result when processing non-English inputs but also ensure the accuracy of internal representations. We conducted extensive experiments on typical English-centric large models, LLaMA-2 and Gemma-2, and the results on multiple multilingual datasets show that our method significantly outperforms traditional fine-tuning methods. 

---
# Parameter-Efficient Fine-Tuning of Large Language Models via Deconvolution in Subspace 

**Authors**: Jia-Chen Zhang, Yu-Jie Xiong, Chun-Ming Xia, Dong-Hai Zhu, Xi-He Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01419)  

**Abstract**: Large language model (LLM) is considered a milestone towards achieving Artificial General Intelligence (AGI). With its advanced emergent capabilities, it adapt to a wide range of specific applications. Fine-tuning LLMs for various downstream tasks has become a new paradigm. Low-Rank Adaptation (LoRA) is well-known for its parameter efficiency. It can reduce the number of parameters needed to fine-tune LLMs by several orders of magnitude. However, LoRA-based approaches encounter a significant limitation due to the bottleneck imposed by rank one decomposition. As the parameters count in LLMs increase, even rank one decomposition might surpass the number of parameters truly necessary for handling more downstream tasks. In this paper, we propose a new method for Parameter-Efficient Fine-Tuning (PEFT) via deconvolution in subspace, dubbed as DCFT. We innovatively use deconvolution to complete details and enhance knowledge in subspace incremental matrices, and dynamically control parameters by adjusting the kernel size, unconstrained by rank-one decomposition. Extensive experiments are conducted to validate the effectiveness of DCFT. Results show that compared to LoRA, DCFT achieve an 8$\times$ reduction in parameters, and still achieves highly impressive performance. Our code is available here: this https URL. 

---
# Nature-Inspired Population-Based Evolution of Large Language Models 

**Authors**: Yiqun Zhang, Peng Ye, Xiaocui Yang, Shi Feng, Shufei Zhang, Lei Bai, Wanli Ouyang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01155)  

**Abstract**: Evolution, the engine behind the survival and growth of life on Earth, operates through the population-based process of reproduction. Inspired by this principle, this paper formally defines a newly emerging problem -- the population-based evolution of large language models (LLMs) -- and introduces a novel framework. Starting with a population of parent LLMs, our framework enables the population to evolve through four key operations: (i) crossover, merging the weights of different parents to create offspring LLMs, (ii) mutation, introducing small, random changes to model weights to foster diversity, (iii) selection, prioritizing high-performing models, and (iv) succession, transferring the learned experience from parent to offspring LLMs. With only 200 samples per new task, the LLM population evolves rapidly to adapt to the task at hand, without any gradients. Experiments on 12 datasets show that our framework consistently outperforms existing multi-LLM merging and adaptation methods, achieving accuracy gains of up to 54.8% over the best LLM in the initial population. Moreover, our framework allows for the evolution of LLMs across multiple new tasks simultaneously, scaling effectively with populations of up to 40 LLMs, and even zero-shot generalization to unseen held-out tasks. We have open-sourced the code on GitHub and released the weights of 10 parent LLMs, fine-tuned from gemma-2-2b-it, on HuggingFace$, enabling reproduction of our proposed framework using just a single 4090 GPU with 24GB memory, without any performance degradation. 

---
# Large Language Models for Healthcare Text Classification: A Systematic Review 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2503.01159)  

**Abstract**: Large Language Models (LLMs) have fundamentally transformed approaches to Natural Language Processing (NLP) tasks across diverse domains. In healthcare, accurate and cost-efficient text classification is crucial, whether for clinical notes analysis, diagnosis coding, or any other task, and LLMs present promising potential. Text classification has always faced multiple challenges, including manual annotation for training, handling imbalanced data, and developing scalable approaches. With healthcare, additional challenges are added, particularly the critical need to preserve patients' data privacy and the complexity of the medical terminology. Numerous studies have been conducted to leverage LLMs for automated healthcare text classification and contrast the results with existing machine learning-based methods where embedding, annotation, and training are traditionally required. Existing systematic reviews about LLMs either do not specialize in text classification or do not focus on the healthcare domain. This research synthesizes and critically evaluates the current evidence found in the literature regarding the use of LLMs for text classification in a healthcare setting. Major databases (e.g., Google Scholar, Scopus, PubMed, Science Direct) and other resources were queried, which focused on the papers published between 2018 and 2024 within the framework of PRISMA guidelines, which resulted in 65 eligible research articles. These were categorized by text classification type (e.g., binary classification, multi-label classification), application (e.g., clinical decision support, public health and opinion analysis), methodology, type of healthcare text, and metrics used for evaluation and validation. This review reveals the existing gaps in the literature and suggests future research lines that can be investigated and explored. 

---
# MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages 

**Authors**: Chen Zhang, Mingxu Tao, Zhiyuan Liao, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01150)  

**Abstract**: Large language models (LLMs) excel in high-resource languages but struggle with low-resource languages (LRLs), particularly those spoken by minority communities in China, such as Tibetan, Uyghur, Kazakh, and Mongolian. To systematically track the progress in these languages, we introduce MiLiC-Eval, a benchmark designed for minority languages in China, featuring 24K instances across 9 tasks. MiLiC-Eval focuses on underrepresented writing systems and provides a fine-grained assessment of linguistic and problem-solving skills. Our evaluation reveals that LLMs perform poorly on syntax-intensive tasks and multi-script languages. We further demonstrate how MiLiC-Eval can help advance LRL research in handling diverse writing systems and understanding the process of language adaptation. 

---
# How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach 

**Authors**: Ayeong Lee, Ethan Che, Tianyi Peng  

**Link**: [PDF](https://arxiv.org/pdf/2503.01141)  

**Abstract**: Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability. 

---
# Scientific Reasoning: Assessment of Multimodal Generative LLMs 

**Authors**: Florian Dreyer, Ekaterina Kolos, Daria Matiash  

**Link**: [PDF](https://arxiv.org/pdf/2503.01064)  

**Abstract**: Large language models (LLMs) can answer questions and reason about complex tasks, also from the scientific domain. We assess several multimodal LLMs (MLLMs) on ScienceQA and find that Gemini models show the highest accuracy with little context, and the highest textual similarity to human explanations with richer context. Adapter-tuning of smaller MLLMs did not lead to any reliable performance. Training from Gemini outputs consistently underperformed training from the original data. 

---
# Evaluating Polish linguistic and cultural competency in large language models 

**Authors**: Sławomir Dadas, Małgorzata Grębowiec, Michał Perełkiewicz, Rafał Poświata  

**Link**: [PDF](https://arxiv.org/pdf/2503.00995)  

**Abstract**: Large language models (LLMs) are becoming increasingly proficient in processing and generating multilingual texts, which allows them to address real-world problems more effectively. However, language understanding is a far more complex issue that goes beyond simple text analysis. It requires familiarity with cultural context, including references to everyday life, historical events, traditions, folklore, literature, and pop culture. A lack of such knowledge can lead to misinterpretations and subtle, hard-to-detect errors. To examine language models' knowledge of the Polish cultural context, we introduce the Polish linguistic and cultural competency benchmark, consisting of 600 manually crafted questions. The benchmark is divided into six categories: history, geography, culture & tradition, art & entertainment, grammar, and vocabulary. As part of our study, we conduct an extensive evaluation involving over 30 open-weight and commercial LLMs. Our experiments provide a new perspective on Polish competencies in language models, moving past traditional natural language processing tasks and general knowledge assessment. 

---
# Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs 

**Authors**: Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, Poulami Das  

**Link**: [PDF](https://arxiv.org/pdf/2503.00979)  

**Abstract**: Autoregressive Transformers rely on Key-Value (KV) caching to accelerate inference. However, the linear growth of the KV cache with context length leads to excessive memory consumption and bandwidth constraints. This bottleneck is particularly problematic in real-time applications -- such as chatbots and interactive assistants -- where low latency and high memory efficiency are critical. Existing methods drop distant tokens or compress states in a lossy manner, sacrificing accuracy by discarding vital context or introducing bias.
We propose MorphKV, an inference-time technique that maintains a constant-sized KV cache while preserving accuracy. MorphKV balances long-range dependencies and local coherence during text generation. It eliminates early-token bias while retaining high-fidelity context by adaptively ranking tokens through correlation-aware selection. Unlike heuristic retention or lossy compression, MorphKV iteratively refines the KV cache via lightweight updates guided by attention patterns of recent tokens. This approach captures inter-token correlation with greater accuracy, crucial for tasks like content creation and code generation. Our studies on long-response tasks show 52.9$\%$ memory savings and 18.2$\%$ higher accuracy on average compared to state-of-the-art prior works, enabling efficient real-world deployment. 

---
# HiBench: Benchmarking LLMs Capability on Hierarchical Structure Reasoning 

**Authors**: Zhuohang Jiang, Pangjing Wu, Ziran Liang, Peter Q. Chen, Xu Yuan, Ye Jia, Jiancheng Tu, Chen Li, Peter H.F. Ng, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00912)  

**Abstract**: Structure reasoning is a fundamental capability of large language models (LLMs), enabling them to reason about structured commonsense and answer multi-hop questions. However, existing benchmarks for structure reasoning mainly focus on horizontal and coordinate structures (\emph{e.g.} graphs), overlooking the hierarchical relationships within them. Hierarchical structure reasoning is crucial for human cognition, particularly in memory organization and problem-solving. It also plays a key role in various real-world tasks, such as information extraction and decision-making. To address this gap, we propose HiBench, the first framework spanning from initial structure generation to final proficiency assessment, designed to benchmark the hierarchical reasoning capabilities of LLMs systematically. HiBench encompasses six representative scenarios, covering both fundamental and practical aspects, and consists of 30 tasks with varying hierarchical complexity, totaling 39,519 queries. To evaluate LLMs comprehensively, we develop five capability dimensions that depict different facets of hierarchical structure understanding. Through extensive evaluation of 20 LLMs from 10 model families, we reveal key insights into their capabilities and limitations: 1) existing LLMs show proficiency in basic hierarchical reasoning tasks; 2) they still struggle with more complex structures and implicit hierarchical representations, especially in structural modification and textual reasoning. Based on these findings, we create a small yet well-designed instruction dataset, which enhances LLMs' performance on HiBench by an average of 88.84\% (Llama-3.1-8B) and 31.38\% (Qwen2.5-7B) across all tasks. The HiBench dataset and toolkit are available here, this https URL, to encourage evaluation. 

---
# SemViQA: A Semantic Question Answering System for Vietnamese Information Fact-Checking 

**Authors**: Nam V. Nguyen, Dien X. Tran, Thanh T. Tran, Anh T. Hoang, Tai V. Duong, Di T. Le, Phuc-Lu Le  

**Link**: [PDF](https://arxiv.org/pdf/2503.00955)  

**Abstract**: The rise of misinformation, exacerbated by Large Language Models (LLMs) like GPT and Gemini, demands robust fact-checking solutions, especially for low-resource languages like Vietnamese. Existing methods struggle with semantic ambiguity, homonyms, and complex linguistic structures, often trading accuracy for efficiency. We introduce SemViQA, a novel Vietnamese fact-checking framework integrating Semantic-based Evidence Retrieval (SER) and Two-step Verdict Classification (TVC). Our approach balances precision and speed, achieving state-of-the-art results with 78.97\% strict accuracy on ISE-DSC01 and 80.82\% on ViWikiFC, securing 1st place in the UIT Data Science Challenge. Additionally, SemViQA Faster improves inference speed 7x while maintaining competitive accuracy. SemViQA sets a new benchmark for Vietnamese fact verification, advancing the fight against misinformation. The source code is available at: this https URL. 

---
# Instruct-of-Reflection: Enhancing Large Language Models Iterative Reflection Capabilities via Dynamic-Meta Instruction 

**Authors**: Liping Liu, Chunhong Zhang, Likang Wu, Chuang Zhao, Zheng Hu, Ming He, Jianping Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00902)  

**Abstract**: Self-reflection for Large Language Models (LLMs) has gained significant attention. Existing approaches involve models iterating and improving their previous responses based on LLMs' internal reflection ability or external feedback. However, recent research has raised doubts about whether intrinsic self-correction without external feedback may even degrade performance. Based on our empirical evidence, we find that current static reflection methods may lead to redundant, drift, and stubborn issues. To mitigate this, we introduce Instruct-of-Reflection (IoRT), a novel and general reflection framework that leverages dynamic-meta instruction to enhance the iterative reflection capability of LLMs. Specifically, we propose the instructor driven by the meta-thoughts and self-consistency classifier, generates various instructions, including refresh, stop, and select, to guide the next reflection iteration. Our experiments demonstrate that IoRT achieves an average improvement of 10.1% over established baselines in mathematical and commonsense reasoning tasks, highlighting its efficacy and applicability. 

---
# Argument Summarization and its Evaluation in the Era of Large Language Models 

**Authors**: Moritz Altemeyer, Steffen Eger, Johannes Daxenberger, Tim Altendorf, Philipp Cimiano, Benjamin Schiller  

**Link**: [PDF](https://arxiv.org/pdf/2503.00847)  

**Abstract**: Large Language Models (LLMs) have revolutionized various Natural Language Generation (NLG) tasks, including Argument Summarization (ArgSum), a key subfield of Argument Mining (AM). This paper investigates the integration of state-of-the-art LLMs into ArgSum, including for its evaluation. In particular, we propose a novel prompt-based evaluation scheme, and validate it through a novel human benchmark dataset. Our work makes three main contributions: (i) the integration of LLMs into existing ArgSum frameworks, (ii) the development of a new LLM-based ArgSum system, benchmarked against prior methods, and (iii) the introduction of an advanced LLM-based evaluation scheme. We demonstrate that the use of LLMs substantially improves both the generation and evaluation of argument summaries, achieving state-of-the-art results and advancing the field of ArgSum. 

---
# Babel: Open Multilingual Large Language Models Serving Over 90% of Global Speakers 

**Authors**: Yiran Zhao, Chaoqun Liu, Yue Deng, Jiahao Ying, Mahani Aljunied, Zhaodonghui Li, Lidong Bing, Hou Pong Chan, Yu Rong, Deli Zhao, Wenxuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00865)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing (NLP), yet open-source multilingual LLMs remain scarce, with existing models often limited in language coverage. Such models typically prioritize well-resourced languages, while widely spoken but under-resourced languages are often overlooked. To address this disparity, we introduce $\texttt{Babel}$, an open multilingual LLM that covers the top 25 languages by number of speakers, supports over 90% of the global population, and includes many languages neglected by other open multilingual LLMs. Unlike traditional continue pretraining approaches, Babel expands its parameter count through a layer extension technique that elevates Babel's performance ceiling. We introduce two variants: $\texttt{Babel-9B}$, designed for efficient inference and fine-tuning, and $\texttt{Babel-83B}$, which sets a new standard for open multilingual LLMs. Extensive evaluations on multilingual tasks demonstrate its superior performance compared to open LLMs of comparable size. In addition, using open-source supervised fine-tuning datasets, Babel achieves remarkable performance, with Babel-9B-Chat leading among 10B-sized LLMs and Babel-83B-Chat setting a new standard for multilingual tasks, reaching the same level of commercial models. 

---
# Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs 

**Authors**: Rachit Saluja, Jacob Rosenthal, Yoav Artzi, David J. Pisapia, Benjamin L. Liechty, Mert R. Sabuncu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01194)  

**Abstract**: Large Language Models (LLMs) have shown significant promise across various natural language processing tasks. However, their application in the field of pathology, particularly for extracting meaningful insights from unstructured medical texts such as pathology reports, remains underexplored and not well quantified. In this project, we leverage state-of-the-art language models, including the GPT family, Mistral models, and the open-source Llama models, to evaluate their performance in comprehensively analyzing pathology reports. Specifically, we assess their performance in cancer type identification, AJCC stage determination, and prognosis assessment, encompassing both information extraction and higher-order reasoning tasks. Based on a detailed analysis of their performance metrics in a zero-shot setting, we developed two instruction-tuned models: Path-llama3.1-8B and Path-GPT-4o-mini-FT. These models demonstrated superior performance in zero-shot cancer type identification, staging, and prognosis assessment compared to the other models evaluated. 

---
# Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners 

**Authors**: Miao Peng, Nuo Chen, Zongrui Suo, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00845)  

**Abstract**: Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs. 

---
# Language Models Predict Empathy Gaps Between Social In-groups and Out-groups 

**Authors**: Yu Hou, Hal Daumé III, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2503.01030)  

**Abstract**: Studies of human psychology have demonstrated that people are more motivated to extend empathy to in-group members than out-group members (Cikara et al., 2011). In this study, we investigate how this aspect of intergroup relations in humans is replicated by LLMs in an emotion intensity prediction task. In this task, the LLM is given a short description of an experience a person had that caused them to feel a particular emotion; the LLM is then prompted to predict the intensity of the emotion the person experienced on a numerical scale. By manipulating the group identities assigned to the LLM's persona (the "perceiver") and the person in the narrative (the "experiencer"), we measure how predicted emotion intensities differ between in-group and out-group settings. We observe that LLMs assign higher emotion intensity scores to in-group members than out-group members. This pattern holds across all three types of social groupings we tested: race/ethnicity, nationality, and religion. We perform an in-depth analysis on Llama-3.1-8B, the model which exhibited strongest intergroup bias among those tested. 

---
# Evaluating Personalized Tool-Augmented LLMs from the Perspectives of Personalization and Proactivity 

**Authors**: Yupu Hao, Pengfei Cao, Zhuoran Jin, Huanxuan Liao, Yubo Chen, Kang Liu, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00771)  

**Abstract**: Personalized tool utilization is essential for aligning large language models (LLMs) with user preference in interaction scenarios with various tools. However, most of the current benchmarks primarily focus on either personalization of text generation or direct tool-utilizing, without considering both. In this work, we introduce a novel benchmark ETAPP for evaluating personalized tool invocation, establishing a sandbox environment, and a comprehensive dataset of 800 testing cases covering diverse user profiles. To improve the accuracy of our evaluation, we propose a key-point-based LLM evaluation method, mitigating biases in the LLM-as-a-judge system by manually annotating key points for each test case and providing them to LLM as the reference. Additionally, we evaluate the excellent LLMs and provide an in-depth analysis. Furthermore, we investigate the impact of different tool-invoking strategies on LLMs' personalization performance and the effects of fine-tuning in our task. The effectiveness of our preference-setting and key-point-based evaluation method is also validated. Our findings offer insights into improving personalized LLM agents. Our Code is available at this https URL. 

---
# Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies 

**Authors**: Tianyi Huang, Jingyuan Yi, Peiyang Yu, Xiaochuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00724)  

**Abstract**: The proliferation of misinformation on social media has raised significant societal concerns, necessitating robust detection mechanisms. Large Language Models such as GPT-4 and LLaMA2 have been envisioned as possible tools for detecting misinformation based on their advanced natural language understanding and reasoning capabilities. This paper conducts a comparison of LLM-based approaches to detecting misinformation between text-based, multimodal, and agentic approaches. We evaluate the effectiveness of fine-tuned models, zero-shot learning, and systematic fact-checking mechanisms in detecting misinformation across different topic domains like public health, politics, and finance. We also discuss scalability, generalizability, and explainability of the models and recognize key challenges such as hallucination, adversarial attacks on misinformation, and computational resources. Our findings point towards the importance of hybrid approaches that pair structured verification protocols with adaptive learning techniques to enhance detection accuracy and explainability. The paper closes by suggesting potential avenues of future work, including real-time tracking of misinformation, federated learning, and cross-platform detection models. 

---
# Zero-Shot Keyphrase Generation: Investigating Specialized Instructions and Multi-Sample Aggregation on Large Language Models 

**Authors**: Jayanth Mohan, Jishnu Ray Chowdhury, Tomas Malik, Cornelia Caragea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00597)  

**Abstract**: Keyphrases are the essential topical phrases that summarize a document. Keyphrase generation is a long-standing NLP task for automatically generating keyphrases for a given document. While the task has been comprehensively explored in the past via various models, only a few works perform some preliminary analysis of Large Language Models (LLMs) for the task. Given the impact of LLMs in the field of NLP, it is important to conduct a more thorough examination of their potential for keyphrase generation. In this paper, we attempt to meet this demand with our research agenda. Specifically, we focus on the zero-shot capabilities of open-source instruction-tuned LLMs (Phi-3, Llama-3) and the closed-source GPT-4o for this task. We systematically investigate the effect of providing task-relevant specialized instructions in the prompt. Moreover, we design task-specific counterparts to self-consistency-style strategies for LLMs and show significant benefits from our proposals over the baselines. 

---
# Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs 

**Authors**: Haowen Pan, Xiaozhi Wang, Yixin Cao, Zenglin Shi, Xun Yang, Juanzi Li, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01090)  

**Abstract**: Knowledge editing aims to update outdated information in Large Language Models (LLMs). A representative line of study is locate-then-edit methods, which typically employ causal tracing to identify the modules responsible for recalling factual knowledge about entities. However, we find these methods are often sensitive only to changes in the subject entity, leaving them less effective at adapting to changes in relations. This limitation results in poor editing locality, which can lead to the persistence of irrelevant or inaccurate facts, ultimately compromising the reliability of LLMs. We believe this issue arises from the insufficient precision of knowledge localization. To address this, we propose a Fine-grained Neuron-level Knowledge Editing (FiNE) method that enhances editing locality without affecting overall success rates. By precisely identifying and modifying specific neurons within feed-forward networks, FiNE significantly improves knowledge localization and editing. Quantitative experiments demonstrate that FiNE efficiently achieves better overall performance compared to existing techniques, providing new insights into the localization and modification of knowledge within LLMs. 

---
# Tutorial Proposal: Speculative Decoding for Efficient LLM Inference 

**Authors**: Heming Xia, Cunxiao Du, Yongqi Li, Qian Liu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00491)  

**Abstract**: This tutorial presents a comprehensive introduction to Speculative Decoding (SD), an advanced technique for LLM inference acceleration that has garnered significant research interest in recent years. SD is introduced as an innovative decoding paradigm to mitigate the high inference latency stemming from autoregressive decoding in LLMs. At each decoding step, SD efficiently drafts several future tokens and then verifies them in parallel. This approach, unlike traditional autoregressive decoding, facilitates the simultaneous decoding of multiple tokens per step, thereby achieving promising 2x-4x speedups in LLM inference while maintaining original distributions. This tutorial delves into the latest techniques in SD, including draft model architectures and verification strategies. Additionally, it explores the acceleration potential and future research directions in this promising field. We aim for this tutorial to elucidate the current research landscape and offer insights for researchers interested in Speculative Decoding, ultimately contributing to more efficient LLM inference. 

---
# Rehearse With User: Personalized Opinion Summarization via Role-Playing based on Large Language Models 

**Authors**: Yanyue Zhang, Yulan He, Deyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00449)  

**Abstract**: Personalized opinion summarization is crucial as it considers individual user interests while generating product summaries. Recent studies show that although large language models demonstrate powerful text summarization and evaluation capabilities without the need for training data, they face difficulties in personalized tasks involving long texts. To address this, \textbf{Rehearsal}, a personalized opinion summarization framework via LLMs-based role-playing is proposed. Having the model act as the user, the model can better understand the user's personalized needs. Additionally, a role-playing supervisor and practice process are introduced to improve the role-playing ability of the LLMs, leading to a better expression of user needs. Furthermore, through suggestions from virtual users, the summary generation is intervened, ensuring that the generated summary includes information of interest to the user, thus achieving personalized summary generation. Experiment results demonstrate that our method can effectively improve the level of personalization in large model-generated summaries. 

---
# BERT-based model for Vietnamese Fact Verification Dataset 

**Authors**: Bao Tran, T. N. Khanh, Khang Nguyen Tuong, Thien Dang, Quang Nguyen, Nguyen T. Thinh, Vo T. Hung  

**Link**: [PDF](https://arxiv.org/pdf/2503.00356)  

**Abstract**: The rapid advancement of information and communication technology has facilitated easier access to information. However, this progress has also necessitated more stringent verification measures to ensure the accuracy of information, particularly within the context of Vietnam. This paper introduces an approach to address the challenges of Fact Verification using the Vietnamese dataset by integrating both sentence selection and classification modules into a unified network architecture. The proposed approach leverages the power of large language models by utilizing pre-trained PhoBERT and XLM-RoBERTa as the backbone of the network. The proposed model was trained on a Vietnamese dataset, named ISE-DSC01, and demonstrated superior performance compared to the baseline model across all three metrics. Notably, we achieved a Strict Accuracy level of 75.11\%, indicating a remarkable 28.83\% improvement over the baseline model. 

---
# Structured Reasoning for Fairness: A Multi-Agent Approach to Bias Detection in Textual Data 

**Authors**: Tianyi Huang, Elsa Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00355)  

**Abstract**: From disinformation spread by AI chatbots to AI recommendations that inadvertently reinforce stereotypes, textual bias poses a significant challenge to the trustworthiness of large language models (LLMs). In this paper, we propose a multi-agent framework that systematically identifies biases by disentangling each statement as fact or opinion, assigning a bias intensity score, and providing concise, factual justifications. Evaluated on 1,500 samples from the WikiNPOV dataset, the framework achieves 84.9% accuracy$\unicode{x2014}$an improvement of 13.0% over the zero-shot baseline$\unicode{x2014}$demonstrating the efficacy of explicitly modeling fact versus opinion prior to quantifying bias intensity. By combining enhanced detection accuracy with interpretable explanations, this approach sets a foundation for promoting fairness and accountability in modern language models. 

---
# U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack 

**Authors**: Yunfan Gao, Yun Xiong, Wenlong Wu, Zijing Huang, Bohan Li, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00353)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have expanded their context windows to unprecedented lengths, sparking debates about the necessity of Retrieval-Augmented Generation (RAG). To address the fragmented evaluation paradigms and limited cases in existing Needle-in-a-Haystack (NIAH), this paper introduces U-NIAH, a unified framework that systematically compares LLMs and RAG methods in controlled long context settings. Our framework extends beyond traditional NIAH by incorporating multi-needle, long-needle, and needle-in-needle configurations, along with different retrieval settings, while leveraging the synthetic Starlight Academy dataset-a fictional magical universe-to eliminate biases from pre-trained knowledge. Through extensive experiments, we investigate three research questions: (1) performance trade-offs between LLMs and RAG, (2) error patterns in RAG, and (3) RAG's limitations in complex settings. Our findings show that RAG significantly enhances smaller LLMs by mitigating the "lost-in-the-middle" effect and improving robustness, achieving an 82.58% win-rate over LLMs. However, we observe that retrieval noise and reverse chunk ordering degrade performance, while surprisingly, advanced reasoning LLMs exhibit reduced RAG compatibility due to sensitivity to semantic distractors. We identify typical error patterns including omission due to noise, hallucination under high noise critical condition, and self-doubt behaviors. Our work not only highlights the complementary roles of RAG and LLMs, but also provides actionable insights for optimizing deployments. Code: this https URL. 

---
# How Deep is Love in LLMs' Hearts? Exploring Semantic Size in Human-like Cognition 

**Authors**: Yao Yao, Yifei Yang, Xinbei Ma, Dongjie Yang, Zhuosheng Zhang, Zuchao Li, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00330)  

**Abstract**: How human cognitive abilities are formed has long captivated researchers. However, a significant challenge lies in developing meaningful methods to measure these complex processes. With the advent of large language models (LLMs), which now rival human capabilities in various domains, we are presented with a unique testbed to investigate human cognition through a new lens. Among the many facets of cognition, one particularly crucial aspect is the concept of semantic size, the perceived magnitude of both abstract and concrete words or concepts. This study seeks to investigate whether LLMs exhibit similar tendencies in understanding semantic size, thereby providing insights into the underlying mechanisms of human cognition. We begin by exploring metaphorical reasoning, comparing how LLMs and humans associate abstract words with concrete objects of varying sizes. Next, we examine LLMs' internal representations to evaluate their alignment with human cognitive processes. Our findings reveal that multi-modal training is crucial for LLMs to achieve more human-like understanding, suggesting that real-world, multi-modal experiences are similarly vital for human cognitive development. Lastly, we examine whether LLMs are influenced by attention-grabbing headlines with larger semantic sizes in a real-world web shopping scenario. The results show that multi-modal LLMs are more emotionally engaged in decision-making, but this also introduces potential biases, such as the risk of manipulation through clickbait headlines. Ultimately, this study offers a novel perspective on how LLMs interpret and internalize language, from the smallest concrete objects to the most profound abstract concepts like love. The insights gained not only improve our understanding of LLMs but also provide new avenues for exploring the cognitive abilities that define human intelligence. 

---
# A Multi-Labeled Dataset for Indonesian Discourse: Examining Toxicity, Polarization, and Demographics Information 

**Authors**: Lucky Susanto, Musa Wijanarko, Prasetia Pratama, Zilu Tang, Fariz Akyas, Traci Hong, Ika Idris, Alham Aji, Derry Wijaya  

**Link**: [PDF](https://arxiv.org/pdf/2503.00417)  

**Abstract**: Polarization is defined as divisive opinions held by two or more groups on substantive issues. As the world's third-largest democracy, Indonesia faces growing concerns about the interplay between political polarization and online toxicity, which is often directed at vulnerable minority groups. Despite the importance of this issue, previous NLP research has not fully explored the relationship between toxicity and polarization. To bridge this gap, we present a novel multi-label Indonesian dataset that incorporates toxicity, polarization, and annotator demographic information. Benchmarking this dataset using BERT-base models and large language models (LLMs) shows that polarization information enhances toxicity classification, and vice versa. Furthermore, providing demographic information significantly improves the performance of polarization classification. 

---
# Jawaher: A Multidialectal Dataset of Arabic Proverbs for LLM Benchmarking 

**Authors**: Samar M. Magdy, Sang Yun Kwon, Fakhraddin Alwajih, Safaa Abdelfadil, Shady Shehata, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2503.00231)  

**Abstract**: Recent advancements in instruction fine-tuning, alignment methods such as reinforcement learning from human feedback (RLHF), and optimization techniques like direct preference optimization (DPO) have significantly enhanced the adaptability of large language models (LLMs) to user preferences. However, despite these innovations, many LLMs continue to exhibit biases toward Western, Anglo-centric, or American cultures, with performance on English data consistently surpassing that of other languages. This reveals a persistent cultural gap in LLMs, which complicates their ability to accurately process culturally rich and diverse figurative language such as proverbs. To address this, we introduce Jawaher, a benchmark designed to assess LLMs' capacity to comprehend and interpret Arabic proverbs. Jawaher includes proverbs from various Arabic dialects, along with idiomatic translations and explanations. Through extensive evaluations of both open- and closed-source models, we find that while LLMs can generate idiomatically accurate translations, they struggle with producing culturally nuanced and contextually relevant explanations. These findings highlight the need for ongoing model refinement and dataset expansion to bridge the cultural gap in figurative language processing. 

---
# Robust Multi-Objective Preference Alignment with Online DPO 

**Authors**: Raghav Gupta, Ryan Sullivan, Yunxuan Li, Samrat Phatale, Abhinav Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2503.00295)  

**Abstract**: Multi-objective preference alignment of large language models (LLMs) is critical for developing AI systems that are more configurable, personalizable, helpful, and safe. However, optimizing model outputs to satisfy diverse objectives with variable weights at inference time for truly personalized models presents a significant challenge. Existing approaches are either computationally expensive to train or do not sufficiently steer model behaviors. This paper introduces the Multi-Objective Online DPO (MO-ODPO) algorithm, designed to robustly and efficiently align model behaviors with multiple, potentially conflicting human preferences. Our approach incorporates a prompt conditioning mechanism, allowing us to train a single preference-conditional policy, that can adapt to new preference combinations at inference. Experiments on two popular benchmarks show that MO-ODPO Pareto-dominates existing baselines while providing excellent inference-time steerability between diverse objectives. 

---
# À la recherche du sens perdu: your favourite LLM might have more to say than you can understand 

**Authors**: K. O. T. Erziev  

**Link**: [PDF](https://arxiv.org/pdf/2503.00224)  

**Abstract**: We report a peculiar observation that LLMs can assign hidden meanings to sequences that seem visually incomprehensible to humans: for example, a nonsensical phrase consisting of Byzantine musical symbols is recognized by gpt-4o as "say abracadabra". Moreover, some models can communicate using these sequences.
Some of these meanings are hypothesized to partly originate in the massive spurious correlations due to BPE tokenization. We systematically evaluate the presence of such abilities in a wide range of models: Claude-3.5 Haiku, Claude-3.5 Sonnet (New and Old), Claude-3.7 Sonnet, gpt-4o mini, gpt-4o, o1-mini, Llama-3.3 70B, DeepSeek-R1-Distill-Lllama 70B, Qwen2.5 1.5B, Qwen2.5 32B, Phi-3.5 mini, GigaChat-Max, Vikhr-Llama-3.2 1B.
We argue that this observation might have far-reaching consequences for both safety and security of the modern and future LLMs and systems that employ them. As an illustration, we show that applying this method in combination with simple templates is sufficient to jailbreak previous generation models, with ASR = 0.4 on gpt-4o mini.
Our code and data artifacts are available at this https URL 

---
# A Survey of Uncertainty Estimation Methods on Large Language Models 

**Authors**: Zhiqiu Xia, Jinxuan Xu, Yuqian Zhang, Hang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00172)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks. However, these models could offer biased, hallucinated, or non-factual responses camouflaged by their fluency and realistic appearance. Uncertainty estimation is the key method to address this challenge. While research efforts in uncertainty estimation are ramping up, there is a lack of comprehensive and dedicated surveys on LLM uncertainty estimation. This survey presents four major avenues of LLM uncertainty estimation. Furthermore, we perform extensive experimental evaluations across multiple methods and datasets. At last, we provide critical and promising future directions for LLM uncertainty estimation. 

---
# Llamarine: Open-source Maritime Industry-specific Large Language Model 

**Authors**: William Nguyen, An Phan, Konobu Kimura, Hitoshi Maeno, Mika Tanaka, Quynh Le, William Poucher, Christopher Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2503.00203)  

**Abstract**: Large Language Models (LLMs) have demonstrated substantial potential in addressing complex reasoning tasks, yet their general-purpose nature often limits their effectiveness in specialized domains such as maritime navigation. To bridge this gap, we introduce Llamarine, the first open-source LLM designed specifically for maritime navigation. Llamarine 1.0 is developed through continued pretraining and fine-tuning on a high-quality corpus comprising maritime textbooks, research publications, and web text from Wikipedia. This domain-specific training enables the model to acquire expert-level knowledge in navigational principles, collision avoidance, route optimization, and regulatory compliance. Our key contributions include (a) the curation of a comprehensive maritime dataset from authoritative sources, ensuring depth and reliability in the model's knowledge base; (b) the development of a foundational model capable of reasoning about complex navigational challenges with greater accuracy than general-purpose LLMs; and (c) the establishment of a benchmark to evaluate performance in maritime-specific decision-making tasks. Experimental results demonstrate that Llamarine outperforms both general-purpose and commercial LLMs in critical navigation-related tasks, such as trajectory planning, risk assessment, and compliance with maritime regulations. By providing an open-source foundation model trained exclusively on high-quality maritime literature, Llamarine paves the way for AI-driven advancements in maritime safety, efficiency, and operational decision-making. 

---
# Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations 

**Authors**: Zhongqi Yang, Amir Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00134)  

**Abstract**: Large Language Models (LLMs) effectively leverage common-sense knowledge for general reasoning, yet they struggle with personalized reasoning when tasked with interpreting multifactor personal data. This limitation restricts their applicability in domains that require context-aware decision-making tailored to individuals. This paper introduces Personalized Causal Graph Reasoning as an agentic framework that enhances LLM reasoning by incorporating personal causal graphs derived from data of individuals. These graphs provide a foundation that guides the LLM's reasoning process. We evaluate it on a case study on nutrient-oriented dietary recommendations, which requires personal reasoning due to the implicit unique dietary effects. We propose a counterfactual evaluation to estimate the efficiency of LLM-recommended foods for glucose management. Results demonstrate that the proposed method efficiently provides personalized dietary recommendations to reduce average glucose iAUC across three time windows, which outperforms the previous approach. LLM-as-a-judge evaluation results indicate that our proposed method enhances personalization in the reasoning process. 

---
# from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors 

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Qi Li, Jiangyu Lei  

**Link**: [PDF](https://arxiv.org/pdf/2503.00038)  

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs. 

---
# Detecting LLM-Generated Korean Text through Linguistic Feature Analysis 

**Authors**: Shinwoo Park, Shubin Kim, Do-Kyung Kim, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00032)  

**Abstract**: The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres.
By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. 

---
# AILS-NTUA at SemEval-2025 Task 8: Language-to-Code prompting and Error Fixing for Tabular Question Answering 

**Authors**: Andreas Evangelatos, Giorgos Filandrianos, Maria Lymperaiou, Athanasios Voulodimos, Giorgos Stamou  

**Link**: [PDF](https://arxiv.org/pdf/2503.00435)  

**Abstract**: In this paper, we present our submission to SemEval-2025 Task 8: Question Answering over Tabular Data. This task, evaluated on the DataBench dataset, assesses Large Language Models' (LLMs) ability to answer natural language questions over structured data while addressing topic diversity and table size limitations in previous benchmarks. We propose a system that employs effective LLM prompting to translate natural language queries into executable code, enabling accurate responses, error correction, and interpretability. Our approach ranks first in both subtasks of the competition in the proprietary model category, significantly outperforming the organizer's baseline. 

---
# AnnoCaseLaw: A Richly-Annotated Dataset For Benchmarking Explainable Legal Judgment Prediction 

**Authors**: Magnus Sesodia, Alina Petrova, John Armour, Thomas Lukasiewicz, Oana-Maria Camburu, Puneet K. Dokania, Philip Torr, Christian Schroeder de Witt  

**Link**: [PDF](https://arxiv.org/pdf/2503.00128)  

**Abstract**: Legal systems worldwide continue to struggle with overwhelming caseloads, limited judicial resources, and growing complexities in legal proceedings. Artificial intelligence (AI) offers a promising solution, with Legal Judgment Prediction (LJP) -- the practice of predicting a court's decision from the case facts -- emerging as a key research area. However, existing datasets often formulate the task of LJP unrealistically, not reflecting its true difficulty. They also lack high-quality annotation essential for legal reasoning and explainability. To address these shortcomings, we introduce AnnoCaseLaw, a first-of-its-kind dataset of 471 meticulously annotated U.S. Appeals Court negligence cases. Each case is enriched with comprehensive, expert-labeled annotations that highlight key components of judicial decision making, along with relevant legal concepts. Our dataset lays the groundwork for more human-aligned, explainable LJP models. We define three legally relevant tasks: (1) judgment prediction; (2) concept identification; and (3) automated case annotation, and establish a performance baseline using industry-leading large language models (LLMs). Our results demonstrate that LJP remains a formidable task, with application of legal precedent proving particularly difficult. Code and data are available at this https URL. 

---
# Do Emotions Really Affect Argument Convincingness? A Dynamic Approach with LLM-based Manipulation Checks 

**Authors**: Yanran Chen, Steffen Eger  

**Link**: [PDF](https://arxiv.org/pdf/2503.00024)  

**Abstract**: Emotions have been shown to play a role in argument convincingness, yet this aspect is underexplored in the natural language processing (NLP) community. Unlike prior studies that use static analyses, focus on a single text domain or language, or treat emotion as just one of many factors, we introduce a dynamic framework inspired by manipulation checks commonly used in psychology and social science; leveraging LLM-based manipulation checks, this framework examines the extent to which perceived emotional intensity influences perceived convincingness. Through human evaluation of arguments across different languages, text domains, and topics, we find that in over half of cases, judgments of convincingness remain unchanged despite variations in perceived emotional intensity; when emotions do have an impact, they more often enhance rather than weaken convincingness. We further analyze how 11 LLMs behave in the same scenario, finding that while LLMs generally mirror human patterns, they struggle to capture nuanced emotional effects in individual judgments. 

---
# Eeyore: Realistic Depression Simulation via Supervised and Preference Optimization 

**Authors**: Siyang Liu, Bianca Brie, Wenda Li, Laura Biester, Andrew Lee, James Pennebaker, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2503.00018)  

**Abstract**: Large Language Models (LLMs) have been previously explored for mental healthcare training and therapy client simulation, but they still fall short in authentically capturing diverse client traits and psychological conditions. We introduce \textbf{Eeyore}, an 8B model optimized for realistic depression simulation through a structured alignment framework, incorporating expert input at every stage. First, we systematically curate real-world depression-related conversations, extracting depressive traits to guide data filtering and psychological profile construction, and use this dataset to instruction-tune Eeyore for profile adherence. Next, to further enhance realism, Eeyore undergoes iterative preference optimization -- first leveraging model-generated preferences and then calibrating with a small set of expert-annotated preferences. Throughout the entire pipeline, we actively collaborate with domain experts, developing interactive interfaces to validate trait extraction and iteratively refine structured psychological profiles for clinically meaningful role-play customization. Despite its smaller model size, the Eeyore depression simulation outperforms GPT-4o with SOTA prompting strategies, both in linguistic authenticity and profile adherence. 

---
# Jailbreaking Safeguarded Text-to-Image Models via Large Language Models 

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01839)  

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks. 

---
# LLMInit: A Free Lunch from Large Language Models for Selective Initialization of Recommendation 

**Authors**: Weizhi Zhang, Liangwei Yang, Wooseong Yang, Henry Peng Zou, Yuqing Liu, Ke Xu, Sourav Medya, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01814)  

**Abstract**: Collaborative filtering models, particularly graph-based approaches, have demonstrated strong performance in capturing user-item interactions for recommendation systems. However, they continue to struggle in cold-start and data-sparse scenarios. The emergence of large language models (LLMs) like GPT and LLaMA presents new possibilities for enhancing recommendation performance, especially in cold-start settings. Despite their promise, LLMs pose challenges related to scalability and efficiency due to their high computational demands and limited ability to model complex user-item relationships effectively. In this work, we introduce a novel perspective on leveraging LLMs for CF model initialization. Through experiments, we uncover an embedding collapse issue when scaling CF models to larger embedding dimensions. To effectively harness large-scale LLM embeddings, we propose innovative selective initialization strategies utilizing random, uniform, and variance-based index sampling. Our comprehensive evaluation on multiple real-world datasets demonstrates significant performance gains across various CF models while maintaining a lower computational cost compared to existing LLM-based recommendation approaches. 

---
# Evaluation of LLMs-based Hidden States as Author Representations for Psychological Human-Centered NLP Tasks 

**Authors**: Nikita Soni, Pranav Chitale, Khushboo Singh, Niranjan Balasubramanian, H. Andrew Schwartz  

**Link**: [PDF](https://arxiv.org/pdf/2503.00124)  

**Abstract**: Like most of NLP, models for human-centered NLP tasks -- tasks attempting to assess author-level information -- predominantly use representations derived from hidden states of Transformer-based LLMs. However, what component of the LM is used for the representation varies widely. Moreover, there is a need for Human Language Models (HuLMs) that implicitly model the author and provide a user-level hidden state. Here, we systematically evaluate different ways of representing documents and users using different LM and HuLM architectures to predict task outcomes as both dynamically changing states and averaged trait-like user-level attributes of valence, arousal, empathy, and distress. We find that representing documents as an average of the token hidden states performs the best generally. Further, while a user-level hidden state itself is rarely the best representation, we find its inclusion in the model strengthens token or document embeddings used to derive document- and user-level representations resulting in best performances. 

---
# MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment 

**Authors**: Weicong Qin, Yi Xu, Weijie Yu, Chenglei Shen, Ming He, Jianping Fan, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01711)  

**Abstract**: Personalized product search aims to retrieve and rank items that match users' preferences and search intent. Despite their effectiveness, existing approaches typically assume that users' query fully captures their real motivation. However, our analysis of a real-world e-commerce platform reveals that users often engage in relevant consultations before searching, indicating they refine intents through consultations based on motivation and need. The implied motivation in consultations is a key enhancing factor for personalized search. This unexplored area comes with new challenges including aligning contextual motivations with concise queries, bridging the category-text gap, and filtering noise within sequence history. To address these, we propose a Motivation-Aware Personalized Search (MAPS) method. It embeds queries and consultations into a unified semantic space via LLMs, utilizes a Mixture of Attention Experts (MoAE) to prioritize critical semantics, and introduces dual alignment: (1) contrastive learning aligns consultations, reviews, and product features; (2) bidirectional attention integrates motivation-aware embeddings with user preferences. Extensive experiments on real and synthetic data show MAPS outperforms existing methods in both retrieval and ranking tasks. 

---
# Using (Not so) Large Language Models for Generating Simulation Models in a Formal DSL -- A Study on Reaction Networks 

**Authors**: Justin N. Kreikemeyer, Miłosz Jankowski, Pia Wilsdorf, Adelinde M. Uhrmacher  

**Link**: [PDF](https://arxiv.org/pdf/2503.01675)  

**Abstract**: Formal languages are an integral part of modeling and simulation. They allow the distillation of knowledge into concise simulation models amenable to automatic execution, interpretation, and analysis. However, the arguably most humanly accessible means of expressing models is through natural language, which is not easily interpretable by computers. Here, we evaluate how a Large Language Model (LLM) might be used for formalizing natural language into simulation models. Existing studies only explored using very large LLMs, like the commercial GPT models, without fine-tuning model weights. To close this gap, we show how an open-weights, 7B-parameter Mistral model can be fine-tuned to translate natural language descriptions to reaction network models in a domain-specific language, offering a self-hostable, compute-, and memory efficient alternative. To this end, we develop a synthetic data generator to serve as the basis for fine-tuning and evaluation. Our quantitative evaluation shows that our fine-tuned Mistral model can recover the ground truth simulation model in up to 84.5% of cases. In addition, our small-scale user study demonstrates the model's practical potential for one-time generation as well as interactive modeling in various domains. While promising, in its current form, the fine-tuned small LLM cannot catch up with large LLMs. We conclude that higher-quality training data are required, and expect future small and open-source LLMs to offer new opportunities. 

---
# Constraining Sequential Model Editing with Editing Anchor Compression 

**Authors**: Hao-Xiang Xu, Jun-Yu Ma, Zhen-Hua Ling, Ningyu Zhang, Jia-Chen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00035)  

**Abstract**: Large language models (LLMs) struggle with hallucinations due to false or outdated knowledge. Given the high resource demands of retraining these models, there is an increasing focus on developing model editing. However, the general abilities of LLMs across downstream tasks are prone to significant degradation during sequential editing. This paper statistically observes that the parameter matrix after editing exhibits a significant deviation compared to its previous state as the number of edits increases. This serious deviation affects the original knowledge associations within LLMs and leads to the degradation of their general abilities. To this end, a framework termed Editing Anchor Compression (EAC) is proposed to constrain the deviation of the parameter matrix during sequential editing. It compresses the editing information by selecting editing anchors that are important in encoding new relations without deviating too much from the original matrix, thereby preserving the general abilities. Experiments of applying EAC to two popular editing methods on three LLMs across four tasks are conducted. Evaluation results show that EAC effectively minimizes unreasonable deviations caused by model editing, preserving over 70% of the general abilities while better retaining the editing knowledge compared to the original counterpart methods. 

---
# None of the Above, Less of the Right: Parallel Patterns between Humans and LLMs on Multi-Choice Questions Answering 

**Authors**: Zhi Rui Tam, Cheng-Kuang Wu, Chieh-Yen Lin, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01550)  

**Abstract**: Multiple-choice exam questions with "None of the above" (NA) options have been extensively studied in educational testing, in which existing research suggests that they better assess true knowledge. However, their impact on Large Language Models (LLMs) evaluation remains underexplored. Through systematic experiments with 28 LLMs on the MMLU benchmark, we examine how NA options affect model performance and confidence calibration. Our analysis reveals that NA options, when used as the correct answer, lead to a consistent 30-50\% performance drop across models regardless of scale--suggesting that LLMs lack the meta-cognitive ability to systematically evaluate and reject all given options when none are correct. This degradation shows strong domain dependence, with minimal impact on mathematical reasoning (14.6\% drop) but severe effects on tasks requiring uncertainty handling like business ethics (48.1\% drop). Our results highlight important implications for benchmark design and raise questions about LLMs' ability to handle uncertainty in real-world applications. 

---
# KVCrush: Key value cache size-reduction using similarity in head-behaviour 

**Authors**: Gopi Krishna Jha, Sameh Gobriel, Liubov Talamanova, Alexander Kozlov, Nilesh Jain  

**Link**: [PDF](https://arxiv.org/pdf/2503.00022)  

**Abstract**: Key-value (KV) caching has emerged as a crucial optimization technique for accelerating inference in large language models (LLMs). By allowing the attention operation to scale linearly rather than quadratically with the total sequence length, KV caching significantly enhances generation throughput. However, due to large context lengths in the modern LLMs, the memory footprint of the KV is a huge bottleneck for model deployment directly impacting the model's batch size, hindering its ability to deliver high-throughput. Existing research addresses this challenge using several techniques, such as discarding low-attention tokens, quantization, and matrix approximation which typically lead to a negative impact on the model accuracy.
In this paper, We propose KVCrush technology which can be combined with many KV compression technologies to improve the model accuracy at a much smaller memory. KVCrush provides an alternate representation scheme for key-value states, along with a low-overhead token pruning algorithm that accounts for the token distribution in the KV cache, which in turn allows for a a smaller footprint while maintaining the accuracy of the model. Based on our results, KVCrush reduces LongBench KV Cache size by 4x with less than 1% accuracy drop and achieves state-of-the-art average accuracy with minimal overhead, incurring less than 0.5% total inference latency. KVCrush not only outperforms the accuracy of state-of-the-art importance-based token retention schemes but is also compatible with typical practical LLM deployments using KV cache paging schemes such as vLLM and mixed precision quantization. 

---
# Towards Widening The Distillation Bottleneck for Reasoning Models 

**Authors**: Huifeng Yin, Yu Zhao, Minghao Wu, Xuanfan Ni, Bo Zeng, Hao Wang, Tianqi Shi, Liangying Shao, Chenyang Lyu, Longyue Wang, Weihua Luo, Kaifu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01461)  

**Abstract**: Large Reasoning Models(LRMs) such as OpenAI o1 and DeepSeek-R1 have shown remarkable reasoning capabilities by scaling test-time compute and generating long Chain-of-Thought(CoT). Distillation--post-training on LRMs-generated data--is a straightforward yet effective method to enhance the reasoning abilities of smaller models, but faces a critical bottleneck: we found that distilled long CoT data poses learning difficulty for small models and leads to the inheritance of biases (i.e. over-thinking) when using Supervised Fine-tuning(SFT) and Reinforcement Learning(RL) methods. To alleviate this bottleneck, we propose constructing tree-based CoT data from scratch via Monte Carlo Tree Search(MCTS). We then exploit a set of CoT-aware approaches, including Thoughts Length Balance, Fine-grained DPO, and Joint Post-training Objective, to enhance SFT and RL on the construted data. 

---
# Evaluating Large Language Models on the Spanish Medical Intern Resident (MIR) Examination 2024/2025:A Comparative Analysis of Clinical Reasoning and Knowledge Application 

**Authors**: Carlos Luengo Vera, Ignacio Ferro Picon, M. Teresa del Val Nunez, Jose Andres Gomez Gandia, Antonio de Lucas Ancillo, Victor Ramos Arroyo, Carlos Milan Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00025)  

**Abstract**: This study presents a comparative evaluation of 22 large language models LLMs on the Spanish Medical Intern Resident MIR examinations for 2024 and 2025 with a focus on clinical reasoning domain specific expertise and multimodal processing capabilities The MIR exam consisting of 210 multiple choice questions some requiring image interpretation serves as a stringent benchmark for assessing both factual recall and complex clinical problem solving skills Our investigation encompasses general purpose models such as GPT4 Claude LLaMA and Gemini as well as specialized fine tuned systems like Miri Pro which leverages proprietary Spanish healthcare data to excel in medical contexts
Recent market entries Deepseek and Grok have further enriched the evaluation landscape particularly for tasks that demand advanced visual and semantic analysis The findings indicate that while general purpose LLMs perform robustly overall fine tuned models consistently achieve superior accuracy especially in addressing nuanced domain specific challenges A modest performance decline observed between the two exam cycles appears attributable to the implementation of modified questions designed to mitigate reliance on memorization
The results underscore the transformative potential of domain specific fine tuning and multimodal integration in advancing medical AI applications They also highlight critical implications for the future integration of LLMs into medical education training and clinical decision making emphasizing the importance of balancing automated reasoning with ethical and context aware judgment 

---
# Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers 

**Authors**: Rin Ashizawa, Yoichi Hirose, Nozomu Yoshinari, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2503.01163)  

**Abstract**: Prompt optimization aims to search for effective prompts that enhance the performance of large language models (LLMs). Although existing prompt optimization methods have discovered effective prompts, they often differ from sophisticated prompts carefully designed by human experts. Prompt design strategies, representing best practices for improving prompt performance, can be key to improving prompt optimization. Recently, a method termed the Autonomous Prompt Engineering Toolbox (APET) has incorporated various prompt design strategies into the prompt optimization process. In APET, the LLM is needed to implicitly select and apply the appropriate strategies because prompt design strategies can have negative effects. This implicit selection may be suboptimal due to the limited optimization capabilities of LLMs. This paper introduces Optimizing Prompts with sTrategy Selection (OPTS), which implements explicit selection mechanisms for prompt design. We propose three mechanisms, including a Thompson sampling-based approach, and integrate them into EvoPrompt, a well-known prompt optimizer. Experiments optimizing prompts for two LLMs, Llama-3-8B-Instruct and GPT-4o mini, were conducted using BIG-Bench Hard. Our results show that the selection of prompt design strategies improves the performance of EvoPrompt, and the Thompson sampling-based mechanism achieves the best overall results. Our experimental code is provided at this https URL . 

---
# Evidence of conceptual mastery in the application of rules by Large Language Models 

**Authors**: José Luiz Nunes, Guilherme FCF Almeida, Brian Flanagan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00992)  

**Abstract**: In this paper we leverage psychological methods to investigate LLMs' conceptual mastery in applying rules. We introduce a novel procedure to match the diversity of thought generated by LLMs to that observed in a human sample. We then conducted two experiments comparing rule-based decision-making in humans and LLMs. Study 1 found that all investigated LLMs replicated human patterns regardless of whether they are prompted with scenarios created before or after their training cut-off. Moreover, we found unanticipated differences between the two sets of scenarios among humans. Surprisingly, even these differences were replicated in LLM responses. Study 2 turned to a contextual feature of human rule application: under forced time delay, human samples rely more heavily on a rule's text than on other considerations such as a rule's purpose.. Our results revealed that some models (Gemini Pro and Claude 3) responded in a human-like manner to a prompt describing either forced delay or time pressure, while others (GPT-4o and Llama 3.2 90b) did not. We argue that the evidence gathered suggests that LLMs have mastery over the concept of rule, with implications for both legal decision making and philosophical inquiry. 

---
# Instructor-Worker Large Language Model System for Policy Recommendation: a Case Study on Air Quality Analysis of the January 2025 Los Angeles Wildfires 

**Authors**: Kyle Gao, Dening Lu, Liangzhi Li, Nan Chen, Hongjie He, Linlin Xu, Jonathan Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00566)  

**Abstract**: The Los Angeles wildfires of January 2025 caused more than 250 billion dollars in damage and lasted for nearly an entire month before containment. Following our previous work, the Digital Twin Building, we modify and leverage the multi-agent large language model framework as well as the cloud-mapping integration to study the air quality during the Los Angeles wildfires. Recent advances in large language models have allowed for out-of-the-box automated large-scale data analysis. We use a multi-agent large language system comprised of an Instructor agent and Worker agents. Upon receiving the users' instructions, the Instructor agent retrieves the data from the cloud platform and produces instruction prompts to the Worker agents. The Worker agents then analyze the data and provide summaries. The summaries are finally input back into the Instructor agent, which then provides the final data analysis. We test this system's capability for data-based policy recommendation by assessing our Instructor-Worker LLM system's health recommendations based on air quality during the Los Angeles wildfires. 

---
# Semantic Integrity Constraints: Declarative Guardrails for AI-Augmented Data Processing Systems 

**Authors**: Alexander W. Lee, Justin Chan, Michael Fu, Nicolas Kim, Akshay Mehta, Deepti Raghavan, Ugur Cetintemel  

**Link**: [PDF](https://arxiv.org/pdf/2503.00600)  

**Abstract**: The emergence of AI-augmented Data Processing Systems (DPSs) has introduced powerful semantic operators that extend traditional data management capabilities with LLM-based processing. However, these systems face fundamental reliability (a.k.a. trust) challenges, as LLMs can generate erroneous outputs, limiting their adoption in critical domains. Existing approaches to LLM constraints--ranging from user-defined functions to constrained decoding--are fragmented, imperative, and lack semantics-aware integration into query execution. To address this gap, we introduce Semantic Integrity Constraints (SICs), a novel declarative abstraction that extends traditional database integrity constraints to govern and optimize semantic operators within DPSs. SICs integrate seamlessly into the relational model, allowing users to specify common classes of constraints (e.g., grounding and soundness) while enabling query-aware enforcement and optimization strategies.
In this paper, we present the core design of SICs, describe their formal integration into query execution, and detail our conception of grounding constraints, a key SIC class that ensures factual consistency of generated outputs. In addition, we explore novel enforcement mechanisms, combining proactive (constrained decoding) and reactive (validation and recovery) techniques to optimize efficiency and reliability. Our work establishes SICs as a foundational framework for trustworthy, high-performance AI-augmented data processing, paving the way for future research in constraint-driven optimizations, adaptive enforcement, and enterprise-scale deployments. 

---
# Causal Inference on Outcomes Learned from Text 

**Authors**: Iman Modarressi, Jann Spiess, Amar Venugopal  

**Link**: [PDF](https://arxiv.org/pdf/2503.00725)  

**Abstract**: We propose a machine-learning tool that yields causal inference on text in randomized trials. Based on a simple econometric framework in which text may capture outcomes of interest, our procedure addresses three questions: First, is the text affected by the treatment? Second, which outcomes is the effect on? And third, how complete is our description of causal effects? To answer all three questions, our approach uses large language models (LLMs) that suggest systematic differences across two groups of text documents and then provides valid inference based on costly validation. Specifically, we highlight the need for sample splitting to allow for statistical validation of LLM outputs, as well as the need for human labeling to validate substantive claims about how documents differ across groups. We illustrate the tool in a proof-of-concept application using abstracts of academic manuscripts. 

---
# Societal Alignment Frameworks Can Improve LLM Alignment 

**Authors**: Karolina Stańczak, Nicholas Meade, Mehar Bhatia, Hattie Zhou, Konstantin Böttinger, Jeremy Barnes, Jason Stanley, Jessica Montgomery, Richard Zemel, Nicolas Papernot, Nicolas Chapados, Denis Therien, Timothy P. Lillicrap, Ana Marasović, Sylvie Delacroix, Gillian K. Hadfield, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00069)  

**Abstract**: Recent progress in large language models (LLMs) has focused on producing responses that meet human expectations and align with shared values - a process coined alignment. However, aligning LLMs remains challenging due to the inherent disconnect between the complexity of human values and the narrow nature of the technological approaches designed to address them. Current alignment methods often lead to misspecified objectives, reflecting the broader issue of incomplete contracts, the impracticality of specifying a contract between a model developer, and the model that accounts for every scenario in LLM alignment. In this paper, we argue that improving LLM alignment requires incorporating insights from societal alignment frameworks, including social, economic, and contractual alignment, and discuss potential solutions drawn from these domains. Given the role of uncertainty within societal alignment frameworks, we then investigate how it manifests in LLM alignment. We end our discussion by offering an alternative view on LLM alignment, framing the underspecified nature of its objectives as an opportunity rather than perfect their specification. Beyond technical improvements in LLM alignment, we discuss the need for participatory alignment interface designs. 

---
# SCORE: Systematic COnsistency and Robustness Evaluation for Large Language Models 

**Authors**: Grigor Nalbandyan, Rima Shahbazyan, Evelina Bakhturina  

**Link**: [PDF](https://arxiv.org/pdf/2503.00137)  

**Abstract**: Typical evaluations of Large Language Models (LLMs) report a single metric per dataset, often representing the model's best-case performance under carefully selected settings. Unfortunately, this approach overlooks model robustness and reliability in real-world applications. For instance, simple paraphrasing of prompts on the MMLU-Pro dataset causes accuracy fluctuations of up to 10\%, while reordering answer choices in the AGIEval dataset results in accuracy differences of up to 6.1\%. While some studies discuss issues with LLM robustness, there is no unified or centralized framework for evaluating the robustness of language models. To address this gap and consolidate existing research on model robustness, we present SCORE ($\mathbf{S}$ystematic $\mathbf{CO}$nsistency and $\mathbf{R}$obustness $\mathbf{E}$valuation), a comprehensive framework for non-adversarial evaluation of LLMs. The SCORE framework evaluates models by repeatedly testing them on the same benchmarks in various setups to give a realistic estimate of their accuracy and consistency. We release the code publicly and start an LLM robustness leaderboard to facilitate further development and research. 

---
# VOILA: Evaluation of MLLMs For Perceptual Understanding and Analogical Reasoning 

**Authors**: Nilay Yilmaz, Maitreya Patel, Yiran Lawrence Luo, Tejas Gokhale, Chitta Baral, Suren Jayasuriya, Yezhou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00043)  

**Abstract**: Multimodal Large Language Models (MLLMs) have become a powerful tool for integrating visual and textual information. Despite their exceptional performance on visual understanding benchmarks, measuring their ability to reason abstractly across multiple images remains a significant challenge. To address this, we introduce VOILA, a large-scale, open-ended, dynamic benchmark designed to evaluate MLLMs' perceptual understanding and abstract relational reasoning. VOILA employs an analogical mapping approach in the visual domain, requiring models to generate an image that completes an analogy between two given image pairs, reference and application, without relying on predefined choices. Our experiments demonstrate that the analogical reasoning tasks in VOILA present a challenge to MLLMs. Through multi-step analysis, we reveal that current MLLMs struggle to comprehend inter-image relationships and exhibit limited capabilities in high-level relational reasoning. Notably, we observe that performance improves when following a multi-step strategy of least-to-most prompting. Comprehensive evaluations on open-source models and GPT-4o show that on text-based answers, the best accuracy for challenging scenarios is 13% (LLaMa 3.2) and even for simpler tasks is only 29% (GPT-4o), while human performance is significantly higher at 70% across both difficulty levels. 

---
# Efficient Test-Time Scaling via Self-Calibration 

**Authors**: Chengsong Huang, Langlin Huang, Jixuan Leng, Jiacheng Liu, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00031)  

**Abstract**: Increasing test-time computation is a straightforward approach to enhancing the quality of responses in Large Language Models (LLMs). While Best-of-N sampling and Self-Consistency with majority voting are simple and effective, they require a fixed number of sampling responses for each query, regardless of its complexity. This could result in wasted computation for simpler questions and insufficient exploration for more challenging ones. In this work, we argue that model confidence of responses can be used for improving the efficiency of test-time scaling. Unfortunately, LLMs are known to be overconfident and provide unreliable confidence estimation. To address this limitation, we introduce Self-Calibration by distilling Self-Consistency-derived confidence into the model itself. This enables reliable confidence estimation at test time with one forward pass. We then design confidence-based efficient test-time scaling methods to handle queries of various difficulty, such as Early-Stopping for Best-of-N and Self-Consistency with calibrated confidence. Experiments on three LLMs across six datasets demonstrate the effectiveness of our approach. Specifically, applying confidence-based Early Stopping to Best-of-N improves MathQA accuracy from 81.0 to 83.6 with a sample budget of 16 responses, indicating the efficacy of confidence-based sampling strategy at inference time. 

---
# Zero-Shot and Efficient Clarification Need Prediction in Conversational Search 

**Authors**: Lili Lu, Chuan Meng, Federico Ravenda, Mohammad Aliannejadi, Fabio Crestani  

**Link**: [PDF](https://arxiv.org/pdf/2503.00179)  

**Abstract**: Clarification need prediction (CNP) is a key task in conversational search, aiming to predict whether to ask a clarifying question or give an answer to the current user query. However, current research on CNP suffers from the issues of limited CNP training data and low efficiency. In this paper, we propose a zero-shot and efficient CNP framework (Zef-CNP), in which we first prompt large language models (LLMs) in a zero-shot manner to generate two sets of synthetic queries: ambiguous and specific (unambiguous) queries. We then use the generated queries to train efficient CNP models. Zef-CNP eliminates the need for human-annotated clarification-need labels during training and avoids the use of LLMs with high query latency at query time. To further improve the generation quality of synthetic queries, we devise a topic-, information-need-, and query-aware chain-of-thought (CoT) prompting strategy (TIQ-CoT). Moreover, we enhance TIQ-CoT with counterfactual query generation (CoQu), which guides LLMs first to generate a specific/ambiguous query and then sequentially generate its corresponding ambiguous/specific query. Experimental results show that Zef-CNP achieves superior CNP effectiveness and efficiency compared with zero- and few-shot LLM-based CNP predictors. 

---
# InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation 

**Authors**: Chong Zhang, Yukun Ma, Qian Chen, Wen Wang, Shengkui Zhao, Zexu Pan, Hao Wang, Chongjia Ni, Trung Hieu Nguyen, Kun Zhou, Yidi Jiang, Chaohong Tan, Zhifu Gao, Zhihao Du, Bin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.00084)  

**Abstract**: We introduce InspireMusic, a framework integrated super resolution and large language model for high-fidelity long-form music generation. A unified framework generates high-fidelity music, songs, and audio, which incorporates an autoregressive transformer with a super-resolution flow-matching model. This framework enables the controllable generation of high-fidelity long-form music at a higher sampling rate from both text and audio prompts. Our model differs from previous approaches, as we utilize an audio tokenizer with one codebook that contains richer semantic information, thereby reducing training costs and enhancing efficiency. This combination enables us to achieve high-quality audio generation with long-form coherence of up to $8$ minutes. Then, an autoregressive transformer model based on Qwen 2.5 predicts audio tokens. Next, we employ a super-resolution flow-matching model to generate high-sampling rate audio with fine-grained details learned from an acoustic codec model. Comprehensive experiments show that the InspireMusic-1.5B-Long model has a comparable performance to recent top-tier open-source systems, including MusicGen and Stable Audio 2.0, on subjective and objective evaluations. The code and pre-trained models are released at this https URL. 

---
# Protecting Users From Themselves: Safeguarding Contextual Privacy in Interactions with Conversational Agents 

**Authors**: Ivoline Ngong, Swanand Kadhe, Hao Wang, Keerthiram Murugesan, Justin D. Weisz, Amit Dhurandhar, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.18509)  

**Abstract**: Conversational agents are increasingly woven into individuals' personal lives, yet users often underestimate the privacy risks involved. The moment users share information with these agents (e.g., LLMs), their private information becomes vulnerable to exposure. In this paper, we characterize the notion of contextual privacy for user interactions with LLMs. It aims to minimize privacy risks by ensuring that users (sender) disclose only information that is both relevant and necessary for achieving their intended goals when interacting with LLMs (untrusted receivers). Through a formative design user study, we observe how even "privacy-conscious" users inadvertently reveal sensitive information through indirect disclosures. Based on insights from this study, we propose a locally-deployable framework that operates between users and LLMs, and identifies and reformulates out-of-context information in user prompts. Our evaluation using examples from ShareGPT shows that lightweight models can effectively implement this framework, achieving strong gains in contextual privacy while preserving the user's intended interaction goals through different approaches to classify information relevant to the intended goals. 

---
# Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation 

**Authors**: Yongchao Chen, Yilun Hao, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01700)  

**Abstract**: Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website this https URL for prompts, videos, and code. 

---
# Towards Efficient Educational Chatbots: Benchmarking RAG Frameworks 

**Authors**: Umar Ali Khan, Ekram Khan, Fiza Khan, Athar Ali Moinuddin  

**Link**: [PDF](https://arxiv.org/pdf/2503.00781)  

**Abstract**: Large Language Models (LLMs) have proven immensely beneficial in education by capturing vast amounts of literature-based information, allowing them to generate context without relying on external sources. In this paper, we propose a generative AI-powered GATE question-answering framework (GATE stands for Graduate Aptitude Test in Engineering) that leverages LLMs to explain GATE solutions and support students in their exam preparation. We conducted extensive benchmarking to select the optimal embedding model and LLM, evaluating our framework based on criteria such as latency, faithfulness, and relevance, with additional validation through human evaluation. Our chatbot integrates state-of-the-art embedding models and LLMs to deliver accurate, context-aware responses. Through rigorous experimentation, we identified configurations that balance performance and computational efficiency, ensuring a reliable chatbot to serve students' needs. Additionally, we discuss the challenges faced in data processing and modeling and implemented solutions. Our work explores the application of Retrieval-Augmented Generation (RAG) for GATE Q/A explanation tasks, and our findings demonstrate significant improvements in retrieval accuracy and response quality. This research offers practical insights for developing effective AI-driven educational tools while highlighting areas for future enhancement in usability and scalability. 

---
# PinLanding: Content-First Keyword Landing Page Generation via Multi-Modal AI for Web-Scale Discovery 

**Authors**: Faye Zhang, Jasmine Wan, Qianyu Cheng, Jinfeng Rao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00619)  

**Abstract**: Online platforms like Pinterest hosting vast content collections traditionally rely on manual curation or user-generated search logs to create keyword landing pages (KLPs) -- topic-centered collection pages that serve as entry points for content discovery. While manual curation ensures quality, it doesn't scale to millions of collections, and search log approaches result in limited topic coverage and imprecise content matching. In this paper, we present PinLanding, a novel content-first architecture that transforms the way platforms create topical collections. Instead of deriving topics from user behavior, our system employs a multi-stage pipeline combining vision-language model (VLM) for attribute extraction, large language model (LLM) for topic generation, and a CLIP-based dual-encoder architecture for precise content matching. Our model achieves 99.7% Recall@10 on Fashion200K benchmark, demonstrating strong attribute understanding capabilities. In production deployment for search engine optimization with 4.2 million shopping landing pages, the system achieves a 4X increase in topic coverage and 14.29% improvement in collection attribute precision over the traditional search log-based approach via human evaluation. The architecture can be generalized beyond search traffic to power various user experiences, including content discovery and recommendations, providing a scalable solution to transform unstructured content into curated topical collections across any content domain. 

---
# Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM 

**Authors**: Yuxin Yang, Haoyang Wu, Tao Wang, Jia Yang, Hao Ma, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00309)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized natural language processing. However, these models face challenges in retrieving precise information from vast datasets. Retrieval-Augmented Generation (RAG) was developed to combining LLMs with external information retrieval systems to enhance the accuracy and context of responses. Despite improvements, RAG still struggles with comprehensive retrieval in high-volume, low-information-density databases and lacks relational awareness, leading to fragmented answers.
To address this, this paper introduces the Pseudo-Knowledge Graph (PKG) framework, designed to overcome these limitations by integrating Meta-path Retrieval, In-graph Text and Vector Retrieval into LLMs. By preserving natural language text and leveraging various retrieval techniques, the PKG offers a richer knowledge representation and improves accuracy in information retrieval. Extensive evaluations using Open Compass and MultiHop-RAG benchmarks demonstrate the framework's effectiveness in managing large volumes of data and complex relationships. 

---
# CoPL: Collaborative Preference Learning for Personalizing LLMs 

**Authors**: Youngbin Choi, Seunghyuk Cho, Minjong Lee, MoonJeong Park, Yesong Ko, Jungseul Ok, Dongwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.01658)  

**Abstract**: Personalizing large language models (LLMs) is important for aligning outputs with diverse user preferences, yet existing methods struggle with flexibility and generalization. We propose CoPL (Collaborative Preference Learning), a graph-based collaborative filtering framework that models user-response relationships to enhance preference estimation, particularly in sparse annotation settings. By integrating a mixture of LoRA experts, CoPL efficiently fine-tunes LLMs while dynamically balancing shared and user-specific preferences. Additionally, an optimization-free adaptation strategy enables generalization to unseen users without fine-tuning. Experiments on UltraFeedback-P demonstrate that CoPL outperforms existing personalized reward models, effectively capturing both common and controversial preferences, making it a scalable solution for personalized LLM alignment. 

---
# Leveraging LLMs for Mental Health: Detection and Recommendations from Social Discussions 

**Authors**: Vaishali Aggarwal, Sachin Thukral, Krushil Patel, Arnab Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.01442)  

**Abstract**: Textual data from social platforms captures various aspects of mental health through discussions around and across issues, while users reach out for help and others sympathize and offer support. We propose a comprehensive framework that leverages Natural Language Processing (NLP) and Generative AI techniques to identify and assess mental health disorders, detect their severity, and create recommendations for behavior change and therapeutic interventions based on users' posts on Reddit.
To classify the disorders, we use rule-based labeling methods as well as advanced pre-trained NLP models to extract nuanced semantic features from the data. We fine-tune domain-adapted and generic pre-trained NLP models based on predictions from specialized Large Language Models (LLMs) to improve classification accuracy. Our hybrid approach combines the generalization capabilities of pre-trained models with the domain-specific insights captured by LLMs, providing an improved understanding of mental health discourse. Our findings highlight the strengths and limitations of each model, offering valuable insights into their practical applicability.
This research potentially facilitates early detection and personalized care to aid practitioners and aims to facilitate timely interventions and improve overall well-being, thereby contributing to the broader field of mental health surveillance and digital health analytics. 

---
# DeepRetrieval: Powerful Query Generation for Information Retrieval with Reinforcement Learning 

**Authors**: Pengcheng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00223)  

**Abstract**: Information retrieval systems are crucial for enabling effective access to large document collections. Recent approaches have leveraged Large Language Models (LLMs) to enhance retrieval performance through query augmentation, but often rely on expensive supervised learning or distillation techniques that require significant computational resources and hand-labeled data. In this paper, we introduce DeepRetrieval, a novel reinforcement learning-based approach that trains LLMs to perform query augmentation directly through trial and error, without requiring supervised data. By using the retrieval recall as a reward signal, our system learns to generate effective queries that maximize document retrieval performance. Our preliminary results demonstrate that DeepRetrieval significantly outperforms existing state-of-the-art methods, including the recent LEADS system, achieving 60.82\% recall on publication search and 70.84\% recall on trial search tasks while using a smaller model (3B vs. 7B parameters) and requiring no supervision data. These results suggest that our reinforcement learning approach offers a more efficient and effective paradigm for information retrieval, potentially changing the landscape of document retrieval systems. code is available at this https URL. 

---
# Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation 

**Authors**: Mufei Li, Siqi Miao, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2410.20724)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning abilities but face limitations such as hallucinations and outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) addresses these issues by grounding LLM outputs in structured external knowledge from KGs. However, current KG-based RAG frameworks still struggle to optimize the trade-off between retrieval effectiveness and efficiency in identifying a suitable amount of relevant graph information for the LLM to digest. We introduce SubgraphRAG, extending the KG-based RAG framework that retrieves subgraphs and leverages LLMs for reasoning and answer prediction. Our approach innovatively integrates a lightweight multilayer perceptron with a parallel triple-scoring mechanism for efficient and flexible subgraph retrieval while encoding directional structural distances to enhance retrieval effectiveness. The size of retrieved subgraphs can be flexibly adjusted to match the query's need and the downstream LLM's capabilities. This design strikes a balance between model complexity and reasoning power, enabling scalable and generalizable retrieval processes. Notably, based on our retrieved subgraphs, smaller LLMs like Llama3.1-8B-Instruct deliver competitive results with explainable reasoning, while larger models like GPT-4o achieve state-of-the-art accuracy compared with previous baselines -- all without fine-tuning. Extensive evaluations on the WebQSP and CWQ benchmarks highlight SubgraphRAG's strengths in efficiency, accuracy, and reliability by reducing hallucinations and improving response grounding. 

---
# Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints 

**Authors**: Sam Bowyer, Laurence Aitchison, Desi R. Ivanova  

**Link**: [PDF](https://arxiv.org/pdf/2503.01747)  

**Abstract**: Rigorous statistical evaluations of large language models (LLMs), including valid error bars and significance testing, are essential for meaningful and reliable performance assessment. Currently, when such statistical measures are reported, they typically rely on the Central Limit Theorem (CLT). In this position paper, we argue that while CLT-based methods for uncertainty quantification are appropriate when benchmarks consist of thousands of examples, they fail to provide adequate uncertainty estimates for LLM evaluations that rely on smaller, highly specialized benchmarks. In these small-data settings, we demonstrate that CLT-based methods perform very poorly, usually dramatically underestimating uncertainty (i.e. producing error bars that are too small). We give recommendations for alternative frequentist and Bayesian methods that are both easy to implement and more appropriate in these increasingly common scenarios. We provide a simple Python library for these Bayesian methods at this https URL . 

---
# SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models 

**Authors**: Cansu Sancaktar, Christian Gumbsch, Andrii Zadaianchuk, Pavel Kolev, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.01584)  

**Abstract**: Exploration is a cornerstone of reinforcement learning (RL). Intrinsic motivation attempts to decouple exploration from external, task-based rewards. However, established approaches to intrinsic motivation that follow general principles such as information gain, often only uncover low-level interactions. In contrast, children's play suggests that they engage in meaningful high-level behavior by imitating or interacting with their caregivers. Recent work has focused on using foundation models to inject these semantic biases into exploration. However, these methods often rely on unrealistic assumptions, such as language-embedded environments or access to high-level actions. We propose SEmaNtically Sensible ExploratIon (SENSEI), a framework to equip model-based RL agents with an intrinsic motivation for semantically meaningful behavior. SENSEI distills a reward signal of interestingness from Vision Language Model (VLM) annotations, enabling an agent to predict these rewards through a world model. Using model-based RL, SENSEI trains an exploration policy that jointly maximizes semantic rewards and uncertainty. We show that in both robotic and video game-like simulations SENSEI discovers a variety of meaningful behaviors from image observations and low-level actions. SENSEI provides a general tool for learning from foundation model feedback, a crucial research direction, as VLMs become more powerful. 

---
# Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning 

**Authors**: Wenjie Wu, Yongcheng Jing, Yingjie Wang, Wenbin Hu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01642)  

**Abstract**: Recent large language model (LLM) reasoning, despite its success, suffers from limited domain knowledge, susceptibility to hallucinations, and constrained reasoning depth, particularly in small-scale models deployed in resource-constrained environments. This paper presents the first investigation into integrating step-wise knowledge graph retrieval with step-wise reasoning to address these challenges, introducing a novel paradigm termed as graph-augmented reasoning. Our goal is to enable frozen, small-scale LLMs to retrieve and process relevant mathematical knowledge in a step-wise manner, enhancing their problem-solving abilities without additional training. To this end, we propose KG-RAR, a framework centered on process-oriented knowledge graph construction, a hierarchical retrieval strategy, and a universal post-retrieval processing and reward model (PRP-RM) that refines retrieved information and evaluates each reasoning step. Experiments on the Math500 and GSM8K benchmarks across six models demonstrate that KG-RAR yields encouraging results, achieving a 20.73\% relative improvement with Llama-3B on Math500. 

---
# Can Large Language Models Help Experimental Design for Causal Discovery? 

**Authors**: Junyi Li, Yongqiang Chen, Chenxi Liu, Qianyi Cai, Tongliang Liu, Bo Han, Kun Zhang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01139)  

**Abstract**: Designing proper experiments and selecting optimal intervention targets is a longstanding problem in scientific or causal discovery. Identifying the underlying causal structure from observational data alone is inherently this http URL interventional data, on the other hand, is crucial to causal discovery, yet it is usually expensive and time-consuming to gather sufficient interventional data to facilitate causal this http URL approaches commonly utilize uncertainty or gradient signals to determine the intervention targets. However, numerical-based approaches may yield suboptimal results due to the inaccurate estimation of the guiding signals at the beginning when with limited interventional data. In this work, we investigate a different approach, whether we can leverage Large Language Models (LLMs) to assist with the intervention targeting in causal discovery by making use of the rich world knowledge about the experimental design in this http URL, we present \oursfull (\ours) -- a robust framework that effectively incorporates LLMs to augment existing numerical approaches for the intervention targeting in causal discovery. Across $4$ realistic benchmark scales, \ours demonstrates significant improvements and robustness over existing methods and even surpasses humans, which demonstrates the usefulness of LLMs in assisting with experimental design for scientific discovery. 

---
# OptMetaOpenFOAM: Large Language Model Driven Chain of Thought for Sensitivity Analysis and Parameter Optimization based on CFD 

**Authors**: Yuxuan Chen, Long Zhang, Xu Zhu, Hua Zhou, Zhuyin Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.01273)  

**Abstract**: Merging natural language interfaces with computational fluid dynamics (CFD) workflows presents transformative opportunities for both industry and research. In this study, we introduce OptMetaOpenFOAM - a novel framework that bridges MetaOpenFOAM with external analysis and optimization tool libraries through a large language model (LLM)-driven chain-of-thought (COT) methodology. By automating complex CFD tasks via natural language inputs, the framework empowers non-expert users to perform sensitivity analyses and parameter optimizations with markedly improved efficiency. The test dataset comprises 11 distinct CFD analysis or optimization tasks, including a baseline simulation task derived from an OpenFOAM tutorial covering fluid dynamics, combustion, and heat transfer. Results confirm that OptMetaOpenFOAM can accurately interpret user requirements expressed in natural language and effectively invoke external tool libraries alongside MetaOpenFOAM to complete the tasks. Furthermore, validation on a non-OpenFOAM tutorial case - namely, a hydrogen combustion chamber - demonstrates that a mere 200-character natural language input can trigger a sequence of simulation, postprocessing, analysis, and optimization tasks spanning over 2,000 lines of code. These findings underscore the transformative potential of LLM-driven COT methodologies in linking external tool for advanced analysis and optimization, positioning OptMetaOpenFOAM as an effective tool that streamlines CFD simulations and enhances their convenience and efficiency for both industrial and research applications. Code is available at this https URL. 

---
# NeSyC: A Neuro-symbolic Continual Learner For Complex Embodied Tasks In Open Domains 

**Authors**: Wonje Choi, Jinwoo Park, Sanghyun Ahn, Daehee Lee, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00870)  

**Abstract**: We explore neuro-symbolic approaches to generalize actionable knowledge, enabling embodied agents to tackle complex tasks more effectively in open-domain environments. A key challenge for embodied agents is the generalization of knowledge across diverse environments and situations, as limited experiences often confine them to their prior knowledge. To address this issue, we introduce a novel framework, NeSyC, a neuro-symbolic continual learner that emulates the hypothetico-deductive model by continually formulating and validating knowledge from limited experiences through the combined use of Large Language Models (LLMs) and symbolic tools. Specifically, we devise a contrastive generality improvement scheme within NeSyC, which iteratively generates hypotheses using LLMs and conducts contrastive validation via symbolic tools. This scheme reinforces the justification for admissible actions while minimizing the inference of inadmissible ones. Additionally, we incorporate a memory-based monitoring scheme that efficiently detects action errors and triggers the knowledge refinement process across domains. Experiments conducted on diverse embodied task benchmarks-including ALFWorld, VirtualHome, Minecraft, RLBench, and a real-world robotic scenario-demonstrate that NeSyC is highly effective in solving complex embodied tasks across a range of open-domain environments. 

---
# A Law Reasoning Benchmark for LLM with Tree-Organized Structures including Factum Probandum, Evidence and Experiences 

**Authors**: Jiaxin Shen, Jinan Xu, Huiqi Hu, Luyi Lin, Fei Zheng, Guoyang Ma, Fandong Meng, Jie Zhou, Wenjuan Han  

**Link**: [PDF](https://arxiv.org/pdf/2503.00841)  

**Abstract**: While progress has been made in legal applications, law reasoning, crucial for fair adjudication, remains unexplored. We propose a transparent law reasoning schema enriched with hierarchical factum probandum, evidence, and implicit experience, enabling public scrutiny and preventing bias. Inspired by this schema, we introduce the challenging task, which takes a textual case description and outputs a hierarchical structure justifying the final decision. We also create the first crowd-sourced dataset for this task, enabling comprehensive evaluation. Simultaneously, we propose an agent framework that employs a comprehensive suite of legal analysis tools to address the challenge task. This benchmark paves the way for transparent and accountable AI-assisted law reasoning in the ``Intelligent Court''. 

---
# AutoAdvExBench: Benchmarking autonomous exploitation of adversarial example defenses 

**Authors**: Nicholas Carlini, Javier Rando, Edoardo Debenedetti, Milad Nasr, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2503.01811)  

**Abstract**: We introduce AutoAdvExBench, a benchmark to evaluate if large language models (LLMs) can autonomously exploit defenses to adversarial examples. Unlike existing security benchmarks that often serve as proxies for real-world tasks, bench directly measures LLMs' success on tasks regularly performed by machine learning security experts. This approach offers a significant advantage: if a LLM could solve the challenges presented in bench, it would immediately present practical utility for adversarial machine learning researchers. We then design a strong agent that is capable of breaking 75% of CTF-like ("homework exercise") adversarial example defenses. However, we show that this agent is only able to succeed on 13% of the real-world defenses in our benchmark, indicating the large gap between difficulty in attacking "real" code, and CTF-like code. In contrast, a stronger LLM that can attack 21% of real defenses only succeeds on 54% of CTF-like defenses. We make this benchmark available at this https URL. 

---
# Distilled Prompt Learning for Incomplete Multimodal Survival Prediction 

**Authors**: Yingxue Xu, Fengtao Zhou, Chenyu Zhao, Yihui Wang, Can Yang, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.01653)  

**Abstract**: The integration of multimodal data including pathology images and gene profiles is widely applied in precise survival prediction. Despite recent advances in multimodal survival models, collecting complete modalities for multimodal fusion still poses a significant challenge, hindering their application in clinical settings. Current approaches tackling incomplete modalities often fall short, as they typically compensate for only a limited part of the knowledge of missing modalities. To address this issue, we propose a Distilled Prompt Learning framework (DisPro) to utilize the strong robustness of Large Language Models (LLMs) to missing modalities, which employs two-stage prompting for compensation of comprehensive information for missing modalities. In the first stage, Unimodal Prompting (UniPro) distills the knowledge distribution of each modality, preparing for supplementing modality-specific knowledge of the missing modality in the subsequent stage. In the second stage, Multimodal Prompting (MultiPro) leverages available modalities as prompts for LLMs to infer the missing modality, which provides modality-common information. Simultaneously, the unimodal knowledge acquired in the first stage is injected into multimodal inference to compensate for the modality-specific knowledge of the missing modality. Extensive experiments covering various missing scenarios demonstrated the superiority of the proposed method. The code is available at this https URL. 

---
# Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens 

**Authors**: Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, Weizhen Bian, Zhen Ye, Sitong Cheng, Ruibin Yuan, Zhixian Zhao, Xinfa Zhu, Jiahao Pan, Liumeng Xue, Pengcheng Zhu, Yunlin Chen, Zhifei Li, Xie Chen, Lei Xie, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.01710)  

**Abstract**: Recent advancements in large language models (LLMs) have driven significant progress in zero-shot text-to-speech (TTS) synthesis. However, existing foundation models rely on multi-stage processing or complex architectures for predicting multiple codebooks, limiting efficiency and integration flexibility. To overcome these challenges, we introduce Spark-TTS, a novel system powered by BiCodec, a single-stream speech codec that decomposes speech into two complementary token types: low-bitrate semantic tokens for linguistic content and fixed-length global tokens for speaker attributes. This disentangled representation, combined with the Qwen2.5 LLM and a chain-of-thought (CoT) generation approach, enables both coarse-grained control (e.g., gender, speaking style) and fine-grained adjustments (e.g., precise pitch values, speaking rate). To facilitate research in controllable TTS, we introduce VoxBox, a meticulously curated 100,000-hour dataset with comprehensive attribute annotations. Extensive experiments demonstrate that Spark-TTS not only achieves state-of-the-art zero-shot voice cloning but also generates highly customizable voices that surpass the limitations of reference-based synthesis. Source code, pre-trained models, and audio samples are available at this https URL. 

---
# Scaling Law Phenomena Across Regression Paradigms: Multiple and Kernel Approaches 

**Authors**: Yifang Chen, Xuyang Guo, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2503.01314)  

**Abstract**: Recently, Large Language Models (LLMs) have achieved remarkable success. A key factor behind this success is the scaling law observed by OpenAI. Specifically, for models with Transformer architecture, the test loss exhibits a power-law relationship with model size, dataset size, and the amount of computation used in training, demonstrating trends that span more than seven orders of magnitude. This scaling law challenges traditional machine learning wisdom, notably the Oscar Scissors principle, which suggests that an overparametrized algorithm will overfit the training datasets, resulting in poor test performance. Recent research has also identified the scaling law in simpler machine learning contexts, such as linear regression. However, fully explaining the scaling law in large practical models remains an elusive goal. In this work, we advance our understanding by demonstrating that the scaling law phenomenon extends to multiple regression and kernel regression settings, which are significantly more expressive and powerful than linear methods. Our analysis provides deeper insights into the scaling law, potentially enhancing our understanding of LLMs. 

---
# LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains 

**Authors**: Ling Xiao, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2503.01236)  

**Abstract**: Multi-terrain cost-efficient path planning is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes travel costs. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments, where recharging or refueling is difficult. However, there is very limited research on this topic. In this paper, we develop a prompt-based approach, LLM-Advisor, which leverages large language models (LLMs) as effective advisors for path planning. The LLM-Advisor selectively provides suggestions, demonstrating its ability to recognize when no modifications are necessary. When suggestions are made, 70.59% of the paths suggested for the A* algorithm, 69.47% for the RRT* algorithm, and 78.70% for the LLM-A* algorithm achieve greater cost efficiency. Since LLM-Advisor may occasionally lack common sense in their suggestions, we propose two hallucination-mitigation strategies. Furthermore, we experimentally verified that GPT-4o performs poorly in zero-shot path planning, even when terrain descriptions are clearly provided, demonstrating its low spatial awareness. We also experimentally demonstrate that using an LLM as an advisor is more effective than directly integrating it into the path-planning loop. Since LLMs may generate hallucinations, using LLMs in the loop of a search-based method (such as A*) may lead to a higher number of failed paths, demonstrating that our proposed LLM-Advisor is a better choice. 

---
# Language-Guided Object Search in Agricultural Environments 

**Authors**: Advaith Balaji, Saket Pradhan, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2503.01068)  

**Abstract**: Creating robots that can assist in farms and gardens can help reduce the mental and physical workload experienced by farm workers. We tackle the problem of object search in a farm environment, providing a method that allows a robot to semantically reason about the location of an unseen target object among a set of previously seen objects in the environment using a Large Language Model (LLM). We leverage object-to-object semantic relationships to plan a path through the environment that will allow us to accurately and efficiently locate our target object while also reducing the overall distance traveled, without needing high-level room or area-level semantic relationships. During our evaluations, we found that our method outperformed a current state-of-the-art baseline and our ablations. Our offline testing yielded an average path efficiency of 84%, reflecting how closely the predicted path aligns with the ideal path. Upon deploying our system on the Boston Dynamics Spot robot in a real-world farm environment, we found that our system had a success rate of 80%, with a success weighted by path length of 0.67, which demonstrates a reasonable trade-off between task success and path efficiency under real-world conditions. The project website can be viewed at this https URL 

---
# LLM-Fusion: A Novel Multimodal Fusion Model for Accelerated Material Discovery 

**Authors**: Onur Boyar, Indra Priyadarsini, Seiji Takeda, Lisa Hamada  

**Link**: [PDF](https://arxiv.org/pdf/2503.01022)  

**Abstract**: Discovering materials with desirable properties in an efficient way remains a significant problem in materials science. Many studies have tackled this problem by using different sets of information available about the materials. Among them, multimodal approaches have been found to be promising because of their ability to combine different sources of information. However, fusion algorithms to date remain simple, lacking a mechanism to provide a rich representation of multiple modalities. This paper presents LLM-Fusion, a novel multimodal fusion model that leverages large language models (LLMs) to integrate diverse representations, such as SMILES, SELFIES, text descriptions, and molecular fingerprints, for accurate property prediction. Our approach introduces a flexible LLM-based architecture that supports multimodal input processing and enables material property prediction with higher accuracy than traditional methods. We validate our model on two datasets across five prediction tasks and demonstrate its effectiveness compared to unimodal and naive concatenation baselines. 

---
# Towards Reliable LLM-Driven Fuzz Testing: Vision and Road Ahead 

**Authors**: Yiran Cheng, Hong Jin Kang, Lwin Khin Shar, Chaopeng Dong, Zhiqiang Shi, Shichao Lv, Limin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00795)  

**Abstract**: Fuzz testing is a crucial component of software security assessment, yet its effectiveness heavily relies on valid fuzz drivers and diverse seed inputs. Recent advancements in Large Language Models (LLMs) offer transformative potential for automating fuzz testing (LLM4Fuzz), particularly in generating drivers and seeds. However, current LLM4Fuzz solutions face critical reliability challenges, including low driver validity rates and seed quality trade-offs, hindering their practical adoption.
This paper aims to examine the reliability bottlenecks of LLM-driven fuzzing and explores potential research directions to address these limitations. It begins with an overview of the current development of LLM4SE and emphasizes the necessity for developing reliable LLM4Fuzz solutions. Following this, the paper envisions a vision where reliable LLM4Fuzz transforms the landscape of software testing and security for industry, software development practitioners, and economic accessibility. It then outlines a road ahead for future research, identifying key challenges and offering specific suggestions for the researchers to consider. This work strives to spark innovation in the field, positioning reliable LLM4Fuzz as a fundamental component of modern software testing. 

---
# AI Agents for Ground-Based Gamma Astronomy 

**Authors**: D. Kostunin, V. Sotnikov, S. Golovachev, A. Strube  

**Link**: [PDF](https://arxiv.org/pdf/2503.00821)  

**Abstract**: Next-generation instruments for ground-based gamma-ray astronomy are marked by a substantial increase in complexity, featuring dozens of telescopes. This leap in scale introduces significant challenges in managing system operations and offline data analysis. Methods, which depend on advanced personnel training and sophisticated software, become increasingly strained as system complexity grows, making it more challenging to effectively support users in such a multifaceted environment. To address these challenges, we propose the development of AI agents based on instruction-finetuned large language models (LLMs). These agents align with specific documentation and codebases, understand the environmental context, operate with external APIs, and communicate with humans in natural language. Leveraging the advanced capabilities of modern LLMs, which can process and retain vast amounts of information, these AI agents offer a transformative approach to system management and data analysis by automating complex tasks and providing intelligent assistance. We present two prototypes that integrate with the Cherenkov Telescope Array Observatory pipelines for operations and offline data analysis. The first prototype automates data model implementation and maintenance for the Configuration Database of the Array Control and Data Acquisition (ACADA). The second prototype is an open-access code generation application tailored for data analysis based on the Gammapy framework. 

---
# LLMs are everywhere: Ubiquitous Utilization of AI Models through Air Computing 

**Authors**: Baris Yamansavascilar, Atay Ozgovde, Cem Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2503.00767)  

**Abstract**: We are witnessing a new era where problem-solving and cognitive tasks are being increasingly delegated to Large Language Models (LLMs) across diverse domains, ranging from code generation to holiday planning. This trend also creates a demand for the ubiquitous execution of LLM-powered applications in a wide variety of environments in which traditional terrestrial 2D networking infrastructures may prove insufficient. A promising solution in this context is to extend edge computing into a 3D setting to include aerial platforms organized in multiple layers, a paradigm we refer to as air computing, to augment local devices for running LLM and Generative AI (GenAI) applications. This approach alleviates the strain on existing infrastructure while enhancing service efficiency by offloading computational tasks to the corresponding air units such as UAVs. Furthermore, the coordinated deployment of various air units can significantly improve the Quality of Experience (QoE) by ensuring seamless, adaptive, and resilient task execution. In this study, we investigate the synergy between LLM-based applications and air computing, exploring their potential across various use cases. Additionally, we present a disaster response case study demonstrating how the collaborative utilization of LLMs and air computing can significantly improve outcomes in critical situations. 

---
# LADDER: Self-Improving LLMs Through Recursive Problem Decomposition 

**Authors**: Toby Simonds, Akira Yoshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2503.00735)  

**Abstract**: We introduce LADDER (Learning through Autonomous Difficulty-Driven Example Recursion), a framework enabling LLMs to autonomously improve their problem-solving capabilities through self-guided learning. By recursively generating and solving progressively simpler variants of complex problems, LADDER enables models to progressively learn through reinforcement learning how to solve harder problems. This self-improvement process is guided by verifiable reward signals, allowing the model to assess its solutions. Unlike prior approaches requiring curated datasets or human feedback, LADDER leverages the model's own capabilities to easier variants of sample questions. We demonstrate LADDER's effectiveness on mathematical integration tasks, where it improves a Llama 3B model's accuracy from 1\% to 82\% on undergraduate-level problems and enables a 7B parameter model to achieve state-of-the-art performance (70\%) on the MIT Integration Bee examination for it's model size. We also introduce TTRL (Test-Time Reinforcement Learning), a method that generates variants of test problems at inference time and applies reinforcement learning to further improve performance. By further creating and solving related problems during testing, TTRL enables the 7B model to achieve a score of 85\%, surpassing o1. These results showcase how strategic self-directed learning can achieve significant capability improvements without relying on architectural scaling or human supervision. 

---
# LLMDR: LLM-Driven Deadlock Detection and Resolution in Multi-Agent Pathfinding 

**Authors**: Seungbae Seo, Junghwan Kim, Minjeong Shin, Bongwon Suh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00717)  

**Abstract**: Multi-Agent Pathfinding (MAPF) is a core challenge in multi-agent systems. Existing learning-based MAPF methods often struggle with scalability, particularly when addressing complex scenarios that are prone to deadlocks. To address these challenges, we introduce LLMDR (LLM-Driven Deadlock Detection and Resolution), an approach designed to resolve deadlocks and improve the performance of learnt MAPF models. LLMDR integrates the inference capabilities of large language models (LLMs) with learnt MAPF models and prioritized planning, enabling it to detect deadlocks and provide customized resolution strategies. We evaluate LLMDR on standard MAPF benchmark maps with varying agent numbers, measuring its performance when combined with several base models. The results demonstrate that LLMDR improves the performance of learnt MAPF models, particularly in deadlock-prone scenarios, with notable improvements in success rates. These findings show the potential of integrating LLMs to improve the scalability of learning-based MAPF methods.
The source code for LLMDR is available at: this https URL 

---
# Speculative Ad-hoc Querying 

**Authors**: Haoyu Li, Srikanth Kandula, Maria Angels de Luis Balaguer, Aditya Akella, Venkat Arun  

**Link**: [PDF](https://arxiv.org/pdf/2503.00714)  

**Abstract**: Analyzing large datasets requires responsive query execution, but executing SQL queries on massive datasets can be slow. This paper explores whether query execution can begin even before the user has finished typing, allowing results to appear almost instantly. We propose SpeQL, a system that leverages Large Language Models (LLMs) to predict likely queries based on the database schema, the user's past queries, and their incomplete query. Since exact query prediction is infeasible, SpeQL speculates on partial queries in two ways: 1) it predicts the query structure to compile and plan queries in advance, and 2) it precomputes smaller temporary tables that are much smaller than the original database, but are still predicted to contain all information necessary to answer the user's final query. Additionally, SpeQL continuously displays results for speculated queries and subqueries in real time, aiding exploratory analysis. A utility/user study showed that SpeQL improved task completion time, and participants reported that its speculative display of results helped them discover patterns in the data more quickly. In the study, SpeQL improves user's query latency by up to $289\times$ and kept the overhead reasonable, at $\$4$ per hour. 

---
# CLEA: Closed-Loop Embodied Agent for Enhancing Task Execution in Dynamic Environments 

**Authors**: Mingcong Lei, Ge Wang, Yiming Zhao, Zhixin Mai, Qing Zhao, Yao Guo, Zhen Li, Shuguang Cui, Yatong Han, Jinke Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.00729)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities in the hierarchical decomposition of complex tasks through semantic reasoning. However, their application in embodied systems faces challenges in ensuring reliable execution of subtask sequences and achieving one-shot success in long-term task completion. To address these limitations in dynamic environments, we propose Closed-Loop Embodied Agent (CLEA) -- a novel architecture incorporating four specialized open-source LLMs with functional decoupling for closed-loop task management. The framework features two core innovations: (1) Interactive task planner that dynamically generates executable subtasks based on the environmental memory, and (2) Multimodal execution critic employing an evaluation framework to conduct a probabilistic assessment of action feasibility, triggering hierarchical re-planning mechanisms when environmental perturbations exceed preset thresholds. To validate CLEA's effectiveness, we conduct experiments in a real environment with manipulable objects, using two heterogeneous robots for object search, manipulation, and search-manipulation integration tasks. Across 12 task trials, CLEA outperforms the baseline model, achieving a 67.3% improvement in success rate and a 52.8% increase in task completion rate. These results demonstrate that CLEA significantly enhances the robustness of task planning and execution in dynamic environments. 

---
# Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable 

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Zachary Yahn, Yichang Xu, Ling Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00555)  

**Abstract**: Safety alignment is an important procedure before the official deployment of a Large Language Model (LLM). While safety alignment has been extensively studied for LLM, there is still a large research gap for Large Reasoning Models (LRMs) that equip with improved reasoning capability. We in this paper systematically examine a simplified pipeline for producing safety aligned LRMs. With our evaluation of various LRMs, we deliver two main findings: i) Safety alignment can be done upon the LRM to restore its safety capability. ii) Safety alignment leads to a degradation of the reasoning capability of LRMs. The two findings show that there exists a trade-off between reasoning and safety capability with the sequential LRM production pipeline. The discovered trade-off, which we name Safety Tax, should shed light on future endeavors of safety research on LRMs. As a by-product, we curate a dataset called DirectRefusal, which might serve as an alternative dataset for safety alignment. Our source code is available at this https URL. 

---
# Never too Prim to Swim: An LLM-Enhanced RL-based Adaptive S-Surface Controller for AUVs under Extreme Sea Conditions 

**Authors**: Guanwen Xie, Jingzehua Xu, Yimian Ding, Zhi Zhang, Shuai Zhang, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00527)  

**Abstract**: The adaptivity and maneuvering capabilities of Autonomous Underwater Vehicles (AUVs) have drawn significant attention in oceanic research, due to the unpredictable disturbances and strong coupling among the AUV's degrees of freedom. In this paper, we developed large language model (LLM)-enhanced reinforcement learning (RL)-based adaptive S-surface controller for AUVs. Specifically, LLMs are introduced for the joint optimization of controller parameters and reward functions in RL training. Using multi-modal and structured explicit task feedback, LLMs enable joint adjustments, balance multiple objectives, and enhance task-oriented performance and adaptability. In the proposed controller, the RL policy focuses on upper-level tasks, outputting task-oriented high-level commands that the S-surface controller then converts into control signals, ensuring cancellation of nonlinear effects and unpredictable external disturbances in extreme sea conditions. Under extreme sea conditions involving complex terrain, waves, and currents, the proposed controller demonstrates superior performance and adaptability in high-level tasks such as underwater target tracking and data collection, outperforming traditional PID and SMC controllers. 

---
# Challenges in Testing Large Language Model Based Software: A Faceted Taxonomy 

**Authors**: Felix Dobslaw, Robert Feldt, Juyeon Yoon, Shin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00481)  

**Abstract**: Large Language Models (LLMs) and Multi-Agent LLMs (MALLMs) introduce non-determinism unlike traditional or machine learning software, requiring new approaches to verifying correctness beyond simple output comparisons or statistical accuracy over test datasets.
This paper presents a taxonomy for LLM test case design, informed by both the research literature, our experience, and open-source tools that represent the state of practice. We identify key variation points that impact test correctness and highlight open challenges that the research, industry, and open-source communities must address as LLMs become integral to software systems.
Our taxonomy defines four facets of LLM test case design, addressing ambiguity in both inputs and outputs while establishing best practices. It distinguishes variability in goals, the system under test, and inputs, and introduces two key oracle types: atomic and aggregated. Our mapping indicates that current tools insufficiently account for these variability points, highlighting the need for closer collaboration between academia and practitioners to improve the reliability and reproducibility of LLM testing. 

---
# An evaluation of DeepSeek Models in Biomedical Natural Language Processing 

**Authors**: Zaifu Zhan, Shuang Zhou, Huixue Zhou, Jiawen Deng, Yu Hou, Jeremy Yeung, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00624)  

**Abstract**: The advancement of Large Language Models (LLMs) has significantly impacted biomedical Natural Language Processing (NLP), enhancing tasks such as named entity recognition, relation extraction, event extraction, and text classification. In this context, the DeepSeek series of models have shown promising potential in general NLP tasks, yet their capabilities in the biomedical domain remain underexplored. This study evaluates multiple DeepSeek models (Distilled-DeepSeek-R1 series and Deepseek-LLMs) across four key biomedical NLP tasks using 12 datasets, benchmarking them against state-of-the-art alternatives (Llama3-8B, Qwen2.5-7B, Mistral-7B, Phi-4-14B, Gemma-2-9B). Our results reveal that while DeepSeek models perform competitively in named entity recognition and text classification, challenges persist in event and relation extraction due to precision-recall trade-offs. We provide task-specific model recommendations and highlight future research directions. This evaluation underscores the strengths and limitations of DeepSeek models in biomedical NLP, guiding their future deployment and optimization. 

---
# PodAgent: A Comprehensive Framework for Podcast Generation 

**Authors**: Yujia Xiao, Lei He, Haohan Guo, Fenglong Xie, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.00455)  

**Abstract**: Existing Existing automatic audio generation methods struggle to generate podcast-like audio programs effectively. The key challenges lie in in-depth content generation, appropriate and expressive voice production. This paper proposed PodAgent, a comprehensive framework for creating audio programs. PodAgent 1) generates informative topic-discussion content by designing a Host-Guest-Writer multi-agent collaboration system, 2) builds a voice pool for suitable voice-role matching and 3) utilizes LLM-enhanced speech synthesis method to generate expressive conversational speech. Given the absence of standardized evaluation criteria for podcast-like audio generation, we developed comprehensive assessment guidelines to effectively evaluate the model's performance. Experimental results demonstrate PodAgent's effectiveness, significantly surpassing direct GPT-4 generation in topic-discussion dialogue content, achieving an 87.4% voice-matching accuracy, and producing more expressive speech through LLM-guided synthesis. Demo page: this https URL. Source code: this https URL. 

---
# Smoothing Grounding and Reasoning for MLLM-Powered GUI Agents with Query-Oriented Pivot Tasks 

**Authors**: Zongru Wu, Pengzhou Cheng, Zheng Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00401)  

**Abstract**: Perception-enhanced pre-training, particularly through grounding techniques, is widely adopted to enhance the performance of graphical user interface (GUI) agents. However, in resource-constrained scenarios, the format discrepancy between coordinate-oriented grounding and action-oriented reasoning limits the effectiveness of grounding for reasoning tasks. To address this challenge, we propose a query-oriented pivot approach called query inference, which serves as a bridge between GUI grounding and reasoning. By inferring potential user queries from a screenshot and its associated element coordinates, query inference improves the understanding of coordinates while aligning more closely with reasoning tasks. Experimental results show that query inference outperforms previous grounding techniques under the same training data scale. Notably, query inference achieves comparable or even better performance to large-scale grounding-enhanced OS-Atlas with less than 0.1% of training data. Furthermore, we explore the impact of reasoning formats and demonstrate that integrating additional semantic information into the input further boosts reasoning performance. The code is publicly available athttps://github.com/ZrW00/GUIPivot. 

---
# Shifting Power: Leveraging LLMs to Simulate Human Aversion in ABMs of Bilateral Financial Exchanges, A bond market study 

**Authors**: Alicia Vidler, Toby Walsh  

**Link**: [PDF](https://arxiv.org/pdf/2503.00320)  

**Abstract**: Bilateral markets, such as those for government bonds, involve decentralized and opaque transactions between market makers (MMs) and clients, posing significant challenges for traditional modeling approaches. To address these complexities, we introduce TRIBE an agent-based model augmented with a large language model (LLM) to simulate human-like decision-making in trading environments. TRIBE leverages publicly available data and stylized facts to capture realistic trading dynamics, integrating human biases like risk aversion and ambiguity sensitivity into the decision-making processes of agents. Our research yields three key contributions: first, we demonstrate that integrating LLMs into agent-based models to enhance client agency is feasible and enriches the simulation of agent behaviors in complex markets; second, we find that even slight trade aversion encoded within the LLM leads to a complete cessation of trading activity, highlighting the sensitivity of market dynamics to agents' risk profiles; third, we show that incorporating human-like variability shifts power dynamics towards clients and can disproportionately affect the entire system, often resulting in systemic agent collapse across simulations. These findings underscore the emergent properties that arise when introducing stochastic, human-like decision processes, revealing new system behaviors that enhance the realism and complexity of artificial societies. 

---
# Steering Large Language Model Activations in Sparse Spaces 

**Authors**: Reza Bayat, Ali Rahimi-Kalahroudi, Mohammad Pezeshki, Sarath Chandar, Pascal Vincent  

**Link**: [PDF](https://arxiv.org/pdf/2503.00177)  

**Abstract**: A key challenge in AI alignment is guiding large language models (LLMs) to follow desired behaviors at test time. Activation steering, which modifies internal model activations during inference, offers a potential solution. However, prior work in dense activation spaces struggles with superposition, wherein multiple features become entangled, limiting interpretability and precise control. In contrast, sparse representations provide an untapped opportunity for more interpretable behavior modulation. In this work, we introduce sparse activation steering (SAS), a method that leverages sparse autoencoders (SAEs) to steer LLM behavior in sparse spaces. By isolating behavior-specific features through a contrastive prompt-pairing approach, we define a set of features that can selectively reinforce or suppress behaviors. Experiments on Gemma 2 LLMs show that SAS vectors enable nuanced behavioral modulation and finer-grained control. Furthermore, scaling SAEs improves monosemanticity of SAS vectors, suggesting more reliable and interpretable interventions. 

---
# BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology 

**Authors**: Ludovico Mitchener, Jon M Laurent, Benjamin Tenmann, Siddharth Narayanan, Geemi P Wellawatte, Andrew White, Lorenzo Sani, Samuel G Rodriques  

**Link**: [PDF](https://arxiv.org/pdf/2503.00096)  

**Abstract**: Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery. 

---
# Rethinking LLM Bias Probing Using Lessons from the Social Sciences 

**Authors**: Kirsten N. Morehouse, Siddharth Swaroop, Weiwei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.00093)  

**Abstract**: The proliferation of LLM bias probes introduces three significant challenges: (1) we lack principled criteria for choosing appropriate probes, (2) we lack a system for reconciling conflicting results across probes, and (3) we lack formal frameworks for reasoning about when (and why) probe results will generalize to real user behavior. We address these challenges by systematizing LLM social bias probing using actionable insights from social sciences. We then introduce EcoLevels - a framework that helps (a) determine appropriate bias probes, (b) reconcile conflicting findings across probes, and (c) generate predictions about bias generalization. Overall, we ground our analysis in social science research because many LLM probes are direct applications of human probes, and these fields have faced similar challenges when studying social bias in humans. Based on our work, we suggest how the next generation of LLM bias probing can (and should) benefit from decades of social science research. 

---
# MergeIT: From Selection to Merging for Efficient Instruction Tuning 

**Authors**: Hongyi Cai, Yuqian Fu, Hongming Fu, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.00034)  

**Abstract**: Instruction tuning is crucial for optimizing Large Language Models (LLMs), yet mainstream data selection methods heavily rely on LLMs as instruction quality scorers, leading to high computational costs and reduced data diversity. To address these limitations, we propose MergeIT, a novel LLM-based Merging strategy for better Instruction Tuning that shifts the focus from selection to synthesis. MergeIT operates in two stages: first, topic-aware filtering clusters and refines the dataset, preserving diversity while eliminating redundancy without relying on LLM-based scoring. Second, LLM-based merging synthesizes semantically similar instructions into more informative and compact training data, enhancing data richness while further reducing dataset size. Experimental results demonstrate that MergeIT enables efficient, diverse, and scalable instruction selection and synthesis, establishing LLM-based merging as a promising alternative to conventional scoring-based selection methods for instruction tuning. Our source code and datasets are now available at this https URL 

---
# AI and Semantic Communication for Infrastructure Monitoring in 6G-Driven Drone Swarms 

**Authors**: Tasnim Ahmed, Salimur Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2503.00053)  

**Abstract**: The adoption of unmanned aerial vehicles to monitor critical infrastructure is gaining momentum in various industrial domains. Organizational imperatives drive this progression to minimize expenses, accelerate processes, and mitigate hazards faced by inspection personnel. However, traditional infrastructure monitoring systems face critical bottlenecks-5G networks lack the latency and reliability for large-scale drone coordination, while manual inspections remain costly and slow. We propose a 6G-enabled drone swarm system that integrates ultra-reliable, low-latency communications, edge AI, and semantic communication to automate inspections. By adopting LLMs for structured output and report generation, our framework is hypothesized to reduce inspection costs and improve fault detection speed compared to existing methods. 

---
# Game-Theoretic Regularized Self-Play Alignment of Large Language Models 

**Authors**: Xiaohang Tang, Sangwoong Yoon, Seongho Son, Huizhuo Yuan, Quanquan Gu, Ilija Bogunovic  

**Link**: [PDF](https://arxiv.org/pdf/2503.00030)  

**Abstract**: Self-play alignment algorithms have been developed as effective methods for fine-tuning large language models (LLMs), formulating preference optimization as a two-player game. However, the regularization with respect to the reference policy, which is crucial for mitigating over-optimization, has been insufficiently investigated in self-play alignment. In this paper, we show that our regularization method can improve the unregularized self-play significantly. To study the impact of different regularizations in self-play alignment, we propose Regularized Self-Play Policy Optimization (RSPO). This generalized framework regularizes the self-play by simply adding a chosen regularization term into the loss while maintaining provable last-iterate convergence to the Nash Equilibrium of the corresponding regularized game. Surprisingly, empirical evaluations using the Mistral-7B-Instruct base model reveal that forward KL divergence regularization reduces response length in RSPO, whereas reverse KL divergence markedly improves raw win rates. RSPO with a linear combination of forward and reverse KL divergence regularization substantially increases the length-controlled win rate in AlpacaEval-2, elevating the unregularized self-play alignment method (SPPO) from $28.53\%$ to $35.44\%$. Finally, we show that RSPO also improves the response diversity. 

---
