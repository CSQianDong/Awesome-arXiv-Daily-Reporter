# Learning Composable Chains-of-Thought 

**Authors**: Fangcong Yin, Zeyu Leo Liu, Liu Leqi, Xi Ye, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2505.22635)  

**Abstract**: A common approach for teaching large language models (LLMs) to reason is to train on chain-of-thought (CoT) traces of in-distribution reasoning problems, but such annotated data is costly to obtain for every problem of interest. We want reasoning models to generalize beyond their training distribution, and ideally to generalize compositionally: combine atomic reasoning skills to solve harder, unseen reasoning tasks. We take a step towards compositional generalization of reasoning skills when addressing a target compositional task that has no labeled CoT data. We find that simply training models on CoT data of atomic tasks leads to limited generalization, but minimally modifying CoT formats of constituent atomic tasks to be composable can lead to improvements. We can train "atomic CoT" models on the atomic tasks with Composable CoT data and combine them with multitask learning or model merging for better zero-shot performance on the target compositional task. Such a combined model can be further bootstrapped on a small amount of compositional data using rejection sampling fine-tuning (RFT). Results on string operations and natural language skill compositions show that training LLMs on Composable CoT outperforms multitask learning and continued fine-tuning baselines within a given training data budget. 

---
# AutoL2S: Auto Long-Short Reasoning for Efficient Large Language Models 

**Authors**: Feng Luo, Yu-Neng Chuang, Guanchu Wang, Hoang Anh Duy Le, Shaochen Zhong, Hongyi Liu, Jiayi Yuan, Yang Sui, Vladimir Braverman, Vipin Chaudhary, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22662)  

**Abstract**: The reasoning-capable large language models (LLMs) demonstrate strong performance on complex reasoning tasks but often suffer from overthinking, generating unnecessarily long chain-of-thought (CoT) reasoning paths for easy reasoning questions, thereby increasing inference cost and latency. Recent approaches attempt to address this challenge by manually deciding when to apply long or short reasoning. However, they lack the flexibility to adapt CoT length dynamically based on question complexity. In this paper, we propose Auto Long-Short Reasoning (AutoL2S), a dynamic and model-agnostic framework that enables LLMs to dynamically compress their generated reasoning path based on the complexity of the reasoning question. AutoL2S enables a learned paradigm, in which LLMs themselves can decide when longer reasoning is necessary and when shorter reasoning suffices, by training on data annotated with our proposed method, which includes both long and short CoT paths and a special <EASY> token. We then use <EASY> token to indicate when the model can skip generating lengthy CoT reasoning. This proposed annotation strategy can enhance the LLMs' ability to generate shorter CoT reasoning paths with improved quality after training. Extensive evaluation results show that AutoL2S reduces the length of reasoning generation by up to 57% without compromising performance, demonstrating the effectiveness of AutoL2S for scalable and efficient LLM reasoning. 

---
# The Climb Carves Wisdom Deeper Than the Summit: On the Noisy Rewards in Learning to Reason 

**Authors**: Ang Lv, Ruobing Xie, Xingwu Sun, Zhanhui Kang, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.22653)  

**Abstract**: Recent studies on post-training large language models (LLMs) for reasoning through reinforcement learning (RL) typically focus on tasks that can be accurately verified and rewarded, such as solving math problems. In contrast, our research investigates the impact of reward noise, a more practical consideration for real-world scenarios involving the post-training of LLMs using reward models. We found that LLMs demonstrate strong robustness to substantial reward noise. For example, manually flipping 40% of the reward function's outputs in math tasks still allows a Qwen-2.5-7B model to achieve rapid convergence, improving its performance on math tasks from 5% to 72%, compared to the 75% accuracy achieved by a model trained with noiseless rewards. Surprisingly, by only rewarding the appearance of key reasoning phrases (namely reasoning pattern reward, RPR), such as ``first, I need to''-without verifying the correctness of answers, the model achieved peak downstream performance (over 70% accuracy for Qwen-2.5-7B) comparable to models trained with strict correctness verification and accurate rewards. Recognizing the importance of the reasoning process over the final results, we combined RPR with noisy reward models. RPR helped calibrate the noisy reward models, mitigating potential false negatives and enhancing the LLM's performance on open-ended tasks. These findings suggest the importance of improving models' foundational abilities during the pre-training phase while providing insights for advancing post-training techniques. Our code and scripts are available at this https URL. 

---
# Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs 

**Authors**: Ziling Cheng, Meng Cao, Marc-Antoine Rondeau, Jackie Chi Kit Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2505.22630)  

**Abstract**: The widespread success of large language models (LLMs) on NLP benchmarks has been accompanied by concerns that LLMs function primarily as stochastic parrots that reproduce texts similar to what they saw during pre-training, often erroneously. But what is the nature of their errors, and do these errors exhibit any regularities? In this work, we examine irrelevant context hallucinations, in which models integrate misleading contextual cues into their predictions. Through behavioral analysis, we show that these errors result from a structured yet flawed mechanism that we term class-based (mis)generalization, in which models combine abstract class cues with features extracted from the query or context to derive answers. Furthermore, mechanistic interpretability experiments on Llama-3, Mistral, and Pythia across 39 factual recall relation types reveal that this behavior is reflected in the model's internal computations: (i) abstract class representations are constructed in lower layers before being refined into specific answers in higher layers, (ii) feature selection is governed by two competing circuits -- one prioritizing direct query-based reasoning, the other incorporating contextual cues -- whose relative influences determine the final output. Our findings provide a more nuanced perspective on the stochastic parrot argument: through form-based training, LLMs can exhibit generalization leveraging abstractions, albeit in unreliable ways based on contextual cues -- what we term stochastic chameleons. 

---
# GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning 

**Authors**: Qingchen Yu, Zifan Zheng, Ding Chen, Simin Niu, Bo Tang, Feiyu Xiong, Zhiyu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22661)  

**Abstract**: The evaluation of large language models (LLMs) has traditionally relied on static benchmarks, a paradigm that poses two major limitations: (1) predefined test sets lack adaptability to diverse application domains, and (2) standardized evaluation protocols often fail to capture fine-grained assessments of domain-specific knowledge and contextual reasoning abilities. To overcome these challenges, we propose GuessArena, an adaptive evaluation framework grounded in adversarial game-based interactions. Inspired by the interactive structure of the Guess Who I Am? game, our framework seamlessly integrates dynamic domain knowledge modeling with progressive reasoning assessment to improve evaluation fidelity. Empirical studies across five vertical domains-finance, healthcare, manufacturing, information technology, and education-demonstrate that GuessArena effectively distinguishes LLMs in terms of domain knowledge coverage and reasoning chain completeness. Compared to conventional benchmarks, our method provides substantial advantages in interpretability, scalability, and scenario adaptability. 

---
# Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning 

**Authors**: Erxin Yu, Jing Li, Ming Liao, Qi Zhu, Boyang Xue, Minghui Xu, Baojun Wang, Lanqing Hong, Fei Mi, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22591)  

**Abstract**: Although large language models demonstrate strong performance across various domains, they still struggle with numerous bad cases in mathematical reasoning. Previous approaches to learning from errors synthesize training data by solely extrapolating from isolated bad cases, thereby failing to generalize the extensive patterns inherent within these cases. This paper presents Self-Error-Instruct (SEI), a framework that addresses these model weaknesses and synthesizes more generalized targeted training data. Specifically, we explore a target model on two mathematical datasets, GSM8K and MATH, to pinpoint bad cases. Then, we generate error keyphrases for these cases based on the instructor model's (GPT-4o) analysis and identify error types by clustering these keyphrases. Next, we sample a few bad cases during each generation for each identified error type and input them into the instructor model, which synthesizes additional training data using a self-instruct approach. This new data is refined through a one-shot learning process to ensure that only the most effective examples are kept. Finally, we use these curated data to fine-tune the target model, iteratively repeating the process to enhance performance. We apply our framework to various models and observe improvements in their reasoning abilities across both in-domain and out-of-domain mathematics datasets. These results demonstrate the effectiveness of self-error instruction in improving LLMs' mathematical reasoning through error generalization. 

---
# Less, but Better: Efficient Multilingual Expansion for LLMs via Layer-wise Mixture-of-Experts 

**Authors**: Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Yufeng Chen, Jinan Xu, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.22582)  

**Abstract**: Continually expanding new languages for existing large language models (LLMs) is a promising yet challenging approach to building powerful multilingual LLMs. The biggest challenge is to make the model continuously learn new languages while preserving the proficient ability of old languages. To achieve this, recent work utilizes the Mixture-of-Experts (MoE) architecture to expand new languages by adding new experts and avoid catastrophic forgetting of old languages by routing corresponding tokens to the original model backbone (old experts). Although intuitive, this kind of method is parameter-costly when expanding new languages and still inevitably impacts the performance of old languages. To address these limitations, we analyze the language characteristics of different layers in LLMs and propose a layer-wise expert allocation algorithm (LayerMoE) to determine the appropriate number of new experts for each layer. Specifically, we find different layers in LLMs exhibit different representation similarities between languages and then utilize the similarity as the indicator to allocate experts for each layer, i.e., the higher similarity, the fewer experts. Additionally, to further mitigate the forgetting of old languages, we add a classifier in front of the router network on the layers with higher similarity to guide the routing of old language tokens. Experimental results show that our method outperforms the previous state-of-the-art baseline with 60% fewer experts in the single-expansion setting and with 33.3% fewer experts in the lifelong-expansion setting, demonstrating the effectiveness of our method. 

---
# Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems 

**Authors**: Hoang Pham, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2505.22571)  

**Abstract**: This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation. 

---
# Fusion Steering: Prompt-Specific Activation Control 

**Authors**: Waldemar Chang, Alhassan Yasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22572)  

**Abstract**: We present Fusion Steering, an activation steering methodology that improves factual accuracy in large language models (LLMs) for question-answering (QA) tasks. This approach introduces flexible steering configurations, including full-layer steering and segmented steering. Unlike traditional methods constrained to single-layer or fixed-layer operations, Fusion Steering employs dynamic injection of prompt-specific activation deltas across all transformer layers. These activation deltas are derived from reference completions that combine the ground-truth answer with a model-generated explanation to facilitate semantically enriched, example-specific steering. The injection weights are optimized per prompt using Optuna, targeting a joint objective that balances token overlap (factual alignment) and perplexity (fluency proxy). Evaluation employs a composite score integrating token overlap and LLM-graded quality, encompassing factual accuracy, coherence, and relevance. Empirical results on 260 SimpleQA prompts (selected from 500 where the baseline failed) showcase the efficacy of segmented steering. Using Gemma-2-2B-IT with 8-bit quantization, segmented steering achieves an accuracy of 25.4% (outputs scoring $\geq 0.6$), outperforming the baseline at 3.5% and full-layer steering at 16.2%. Under the stricter SimpleQA rubric, segmented steering boosts fully correct responses from 0.0% to 13.1%. These findings highlight the strengths of segmented, dynamic intervention strategies and the promise of per-prompt, full-network activation control. Fusion Steering is also amenable to sparse representations, such as Neuronpedia or sparse crosscoders, suggesting a promising direction for interpretable and scalable activation-level control in LLMs. 

---
# Emotion-o1: Adaptive Long Reasoning for Emotion Understanding in LLMs 

**Authors**: Changhao Song, Yazhou Zhang, Peng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22548)  

**Abstract**: Emotion understanding includes basic tasks (e.g., sentiment/emotion classification) and advanced tasks (e.g., sarcasm/humor detection). Current methods rely on fixed-length CoT reasoning, failing to adapt to the varying complexity of emotions. We propose a task-adaptive reasoning framework that employs DeepSeek-R1 to generate variable-length reasoning chains for different emotion tasks. By combining fine-tuning with reinforcement learning, we design a composite reward function that balances four objectives: prediction accuracy, adaptive reasoning depth control, structural diversity in reasoning paths, and suppression of repetitive logic. This approach achieves dynamic context-sensitive inference while enabling LLMs to autonomously develop deep reasoning capabilities. Experimental results demonstrate consistent improvements in both Acc and F1 scores across four tasks: emotion, sentiment, humor, and sarcasm. Notably, peak enhancements reached 3.56% F1 (2.76% Acc) for basic tasks and 37.95% F1 (23.14% Acc) for advanced tasks. Our work bridges rigid CoT reasoning and emotional complexity through adaptive-depth analysis. 

---
# Multi-MLLM Knowledge Distillation for Out-of-Context News Detection 

**Authors**: Yimeng Gu, Zhao Tong, Ignacio Castro, Shu Wu, Gareth Tyson  

**Link**: [PDF](https://arxiv.org/pdf/2505.22517)  

**Abstract**: Multimodal out-of-context news is a type of misinformation in which the image is used outside of its original context. Many existing works have leveraged multimodal large language models (MLLMs) for detecting out-of-context news. However, observing the limited zero-shot performance of smaller MLLMs, they generally require label-rich fine-tuning and/or expensive API calls to GPT models to improve the performance, which is impractical in low-resource scenarios. In contrast, we aim to improve the performance of small MLLMs in a more label-efficient and cost-effective manner. To this end, we first prompt multiple teacher MLLMs to generate both label predictions and corresponding rationales, which collectively serve as the teachers' knowledge. We then introduce a two-stage knowledge distillation framework to transfer this knowledge to a student MLLM. In Stage 1, we apply LoRA fine-tuning to the student model using all training data. In Stage 2, we further fine-tune the student model using both LoRA fine-tuning and DPO on the data points where teachers' predictions conflict. This two-stage strategy reduces annotation costs and helps the student model uncover subtle patterns in more challenging cases. Experimental results demonstrate that our approach achieves state-of-the-art performance using less than 10% labeled data. 

---
# Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition 

**Authors**: Hanting Chen, Yasheng Wang, Kai Han, Dong Li, Lin Li, Zhenni Bi, Jinpeng Li, Haoyu Wang, Fei Mi, Mingjian Zhu, Bin Wang, Kaikai Song, Yifei Fu, Xu He, Yu Luo, Chong Zhu, Quan He, Xueyu Wu, Wei He, Hailin Hu, Yehui Tang, Dacheng Tao, Xinghao Chen, Yunhe Wang, Other Contributors  

**Link**: [PDF](https://arxiv.org/pdf/2505.22375)  

**Abstract**: This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners. 

---
# Characterizing Bias: Benchmarking Large Language Models in Simplified versus Traditional Chinese 

**Authors**: Hanjia Lyu, Jiebo Luo, Jian Kang, Allison Koenecke  

**Link**: [PDF](https://arxiv.org/pdf/2505.22645)  

**Abstract**: While the capabilities of Large Language Models (LLMs) have been studied in both Simplified and Traditional Chinese, it is yet unclear whether LLMs exhibit differential performance when prompted in these two variants of written Chinese. This understanding is critical, as disparities in the quality of LLM responses can perpetuate representational harms by ignoring the different cultural contexts underlying Simplified versus Traditional Chinese, and can exacerbate downstream harms in LLM-facilitated decision-making in domains such as education or hiring. To investigate potential LLM performance disparities, we design two benchmark tasks that reflect real-world scenarios: regional term choice (prompting the LLM to name a described item which is referred to differently in Mainland China and Taiwan), and regional name choice (prompting the LLM to choose who to hire from a list of names in both Simplified and Traditional Chinese). For both tasks, we audit the performance of 11 leading commercial LLM services and open-sourced models -- spanning those primarily trained on English, Simplified Chinese, or Traditional Chinese. Our analyses indicate that biases in LLM responses are dependent on both the task and prompting language: while most LLMs disproportionately favored Simplified Chinese responses in the regional term choice task, they surprisingly favored Traditional Chinese names in the regional name choice task. We find that these disparities may arise from differences in training data representation, written character preferences, and tokenization of Simplified and Traditional Chinese. These findings highlight the need for further analysis of LLM biases; as such, we provide an open-sourced benchmark dataset to foster reproducible evaluations of future LLM behavior across Chinese language variants (this https URL). 

---
# EvolveSearch: An Iterative Self-Evolving Search Agent 

**Authors**: Dingchu Zhang, Yida Zhao, Jialong Wu, Baixuan Li, Wenbiao Yin, Liwen Zhang, Yong Jiang, Yufeng Li, Kewei Tu, Pengjun Xie, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22501)  

**Abstract**: The rapid advancement of large language models (LLMs) has transformed the landscape of agentic information seeking capabilities through the integration of tools such as search engines and web browsers. However, current mainstream approaches for enabling LLM web search proficiency face significant challenges: supervised fine-tuning struggles with data production in open-search domains, while RL converges quickly, limiting their data utilization efficiency. To address these issues, we propose EvolveSearch, a novel iterative self-evolution framework that combines SFT and RL to enhance agentic web search capabilities without any external human-annotated reasoning data. Extensive experiments on seven multi-hop question-answering (MHQA) benchmarks demonstrate that EvolveSearch consistently improves performance across iterations, ultimately achieving an average improvement of 4.7\% over the current state-of-the-art across seven benchmarks, opening the door to self-evolution agentic capabilities in open web search domains. 

---
# LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High 

**Authors**: Judith Sieker, Clara Lachenmaier, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2505.22354)  

**Abstract**: This paper examines how LLMs handle false presuppositions and whether certain linguistic factors influence their responses to falsely presupposed content. Presuppositions subtly introduce information as given, making them highly effective at embedding disputable or false information. This raises concerns about whether LLMs, like humans, may fail to detect and correct misleading assumptions introduced as false presuppositions, even when the stakes of misinformation are high. Using a systematic approach based on linguistic presupposition analysis, we investigate the conditions under which LLMs are more or less sensitive to adopt or reject false presuppositions. Focusing on political contexts, we examine how factors like linguistic construction, political party, and scenario probability impact the recognition of false presuppositions. We conduct experiments with a newly created dataset and examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's Mistral-7B-v03. Our results show that the models struggle to recognize false presuppositions, with performance varying by condition. This study highlights that linguistic presupposition analysis is a valuable tool for uncovering the reinforcement of political misinformation in LLM responses. 

---
# Precise In-Parameter Concept Erasure in Large Language Models 

**Authors**: Yoav Gur-Arieh, Clara Suslik, Yihuai Hong, Fazl Barez, Mor Geva  

**Link**: [PDF](https://arxiv.org/pdf/2505.22586)  

**Abstract**: Large language models (LLMs) often acquire knowledge during pretraining that is undesirable in downstream deployments, e.g., sensitive information or copyrighted content. Existing approaches for removing such knowledge rely on fine-tuning, training low-rank adapters or fact-level editing, but these are either too coarse, too shallow, or ineffective. In this work, we propose PISCES (Precise In-parameter Suppression for Concept EraSure), a novel framework for precisely erasing entire concepts from model parameters by directly editing directions that encode them in parameter space. PISCES uses a disentangler model to decompose MLP vectors into interpretable features, identifies those associated with a target concept using automated interpretability techniques, and removes them from model parameters. Experiments on Gemma 2 and Llama 3.1 over various concepts show that PISCES achieves modest gains in efficacy over leading erasure methods, reducing accuracy on the target concept to as low as 7.7%, while dramatically improving erasure specificity (by up to 31%) and robustness (by up to 38%). Overall, these results demonstrate that feature-based in-parameter editing enables a more precise and reliable approach for removing conceptual knowledge in language models. 

---
# Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing 

**Authors**: Yifan Lu, Jing Li, Yigeng Zhou, Yihui Zhang, Wenya Wang, Xiucheng Li, Meishan Zhang, Fangming Liu, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22298)  

**Abstract**: Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs. 

---
# If Pigs Could Fly... Can LLMs Logically Reason Through Counterfactuals? 

**Authors**: Ishwar B Balappanawar, Vamshi Krishna Bonagiri, Anish R Joishy, Manas Gaur, Krishnaprasad Thirunarayan, Ponnurangam Kumaraguru  

**Link**: [PDF](https://arxiv.org/pdf/2505.22318)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive reasoning capabilities in familiar contexts, but struggle when the context conflicts with their parametric knowledge. To investigate this phenomenon, we introduce CounterLogic, a dataset containing 1,800 examples across 9 logical schemas, explicitly designed to evaluate logical reasoning through counterfactual (hypothetical knowledge-conflicting) scenarios. Our systematic evaluation of 11 LLMs across 6 different datasets reveals a consistent performance degradation, with accuracies dropping by 27% on average when reasoning through counterfactual information. We propose Self-Segregate, a prompting method enabling metacognitive awareness (explicitly identifying knowledge conflicts) before reasoning. Our method dramatically narrows the average performance gaps from 27% to just 11%, while significantly increasing the overall accuracy (+7.5%). We discuss the implications of these findings and draw parallels to human cognitive processes, particularly on how humans disambiguate conflicting information during reasoning tasks. Our findings offer practical insights for understanding and enhancing LLMs reasoning capabilities in real-world applications, especially where models must logically reason independently of their factual knowledge. 

---
# Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO 

**Authors**: Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22453)  

**Abstract**: Improving Multi-modal Large Language Models (MLLMs) in the post-training stage typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL). However, these supervised methods require expensive and manually annotated multi-modal data--an ultimately unsustainable resource. While recent efforts have explored unsupervised post-training, their methods are complex and difficult to iterate. In this work, we are the first to investigate the use of GRPO, a stable and scalable online RL algorithm, for enabling continual self-improvement without any external supervision. We propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs. MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that MM-UPT significantly improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3 %$\rightarrow$72.9 % on MathVista, 62.9 %$\rightarrow$68.7 % on We-Math), using standard dataset without ground truth labels. MM-UPT also outperforms prior unsupervised baselines and even approaches the results of supervised GRPO. Furthermore, we show that incorporating synthetic questions, generated solely by MLLM itself, can boost performance as well, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for continual, autonomous enhancement of MLLMs in the absence of external supervision. Our code is available at this https URL. 

---
# Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start 

**Authors**: Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22334)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at this https URL. 

---
# Compensating for Data with Reasoning: Low-Resource Machine Translation with LLMs 

**Authors**: Samuel Frontull, Thomas Ströhle  

**Link**: [PDF](https://arxiv.org/pdf/2505.22293)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in multilingual machine translation, sometimes even outperforming traditional neural systems. However, previous research has highlighted the challenges of using LLMs, particularly with prompt engineering, for low-resource languages. In this work, we introduce Fragment-Shot Prompting, a novel in-context learning method that segments input and retrieves translation examples based on syntactic coverage, along with Pivoted Fragment-Shot, an extension that enables translation without direct parallel data. We evaluate these methods using GPT-3.5, GPT-4o, o1-mini, LLaMA-3.3, and DeepSeek-R1 for translation between Italian and two Ladin variants, revealing three key findings: (1) Fragment-Shot Prompting is effective for translating into and between the studied low-resource languages, with syntactic coverage positively correlating with translation quality; (2) Models with stronger reasoning abilities make more effective use of retrieved knowledge, generally produce better translations, and enable Pivoted Fragment-Shot to significantly improve translation quality between the Ladin variants; and (3) prompt engineering offers limited, if any, improvements when translating from a low-resource to a high-resource language, where zero-shot prompting already yields satisfactory results. We publicly release our code and the retrieval corpora. 

---
# ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM 

**Authors**: Hoang Pham, Thanh-Do Nguyen, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2505.22552)  

**Abstract**: Integrating knowledge graphs (KGs) to enhance the reasoning capabilities of large language models (LLMs) is an emerging research challenge in claim verification. While KGs provide structured, semantically rich representations well-suited for reasoning, most existing verification methods rely on unstructured text corpora, limiting their ability to effectively leverage KGs. Additionally, despite possessing strong reasoning abilities, modern LLMs struggle with multi-step modular pipelines and reasoning over KGs without adaptation. To address these challenges, we propose ClaimPKG, an end-to-end framework that seamlessly integrates LLM reasoning with structured knowledge from KGs. Specifically, the main idea of ClaimPKG is to employ a lightweight, specialized LLM to represent the input claim as pseudo-subgraphs, guiding a dedicated subgraph retrieval module to identify relevant KG subgraphs. These retrieved subgraphs are then processed by a general-purpose LLM to produce the final verdict and justification. Extensive experiments on the FactKG dataset demonstrate that ClaimPKG achieves state-of-the-art performance, outperforming strong baselines in this research field by 9%-12% accuracy points across multiple categories. Furthermore, ClaimPKG exhibits zero-shot generalizability to unstructured datasets such as HoVer and FEVEROUS, effectively combining structured knowledge from KGs with LLM reasoning across various LLM backbones. 

---
# Let's Predict Sentence by Sentence 

**Authors**: Hyeonbin Hwang, Byeongguk Jeon, Seungone Kim, Jiyeon Kim, Hoyeon Chang, Sohee Yang, Seungpil Won, Dohaeng Lee, Youbin Ahn, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.22202)  

**Abstract**: Autoregressive language models (LMs) generate one token at a time, yet human reasoning operates over higher-level abstractions - sentences, propositions, and concepts. This contrast raises a central question- Can LMs likewise learn to reason over structured semantic units rather than raw token sequences? In this work, we investigate whether pretrained LMs can be lifted into such abstract reasoning spaces by building on their learned representations. We present a framework that adapts a pretrained token-level LM to operate in sentence space by autoregressively predicting continuous embeddings of next sentences. We explore two embedding paradigms inspired by classical representation learning: 1) semantic embeddings, learned via autoencoding to preserve surface meaning; and 2) contextual embeddings, trained via next-sentence prediction to encode anticipatory structure. We evaluate both under two inference regimes: Discretized, which decodes each predicted embedding into text before re-encoding; and Continuous, which reasons entirely in embedding space for improved efficiency. Across four domains - mathematics, logic, commonsense, and planning - contextual embeddings under continuous inference show competitive performance with Chain-of-Thought (CoT) while reducing inference-time FLOPs on average by half. We also present early signs of scalability and modular adaptation. Finally, to visualize latent trajectories, we introduce SentenceLens, a diagnostic tool that decodes intermediate model states into interpretable sentences. Together, our results indicate that pretrained LMs can effectively transition to abstract, structured reasoning within latent embedding spaces. 

---
# Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models 

**Authors**: Mehdi Ali, Manuel Brack, Max Lübbering, Elias Wendt, Abbas Goher Khan, Richard Rutmann, Alex Jude, Maurice Kraus, Alexander Arno Weber, Felix Stollenwerk, David Kaczér, Florian Mai, Lucie Flek, Rafet Sifa, Nicolas Flores-Herr, Joachim Köhler, Patrick Schramowski, Michael Fromm, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.22232)  

**Abstract**: High-quality multilingual training data is essential for effectively pretraining large language models (LLMs). Yet, the availability of suitable open-source multilingual datasets remains limited. Existing state-of-the-art datasets mostly rely on heuristic filtering methods, restricting both their cross-lingual transferability and scalability. Here, we introduce JQL, a systematic approach that efficiently curates diverse and high-quality multilingual data at scale while significantly reducing computational demands. JQL distills LLMs' annotation capabilities into lightweight annotators based on pretrained multilingual embeddings. These models exhibit robust multilingual and cross-lingual performance, even for languages and scripts unseen during training. Evaluated empirically across 35 languages, the resulting annotation pipeline substantially outperforms current heuristic filtering methods like Fineweb2. JQL notably enhances downstream model training quality and increases data retention rates. Our research provides practical insights and valuable resources for multilingual data curation, raising the standards of multilingual dataset development. 

---
# Reverse Preference Optimization for Complex Instruction Following 

**Authors**: Xiang Huang, Ting-En Lin, Feiteng Fang, Yuchuan Wu, Hangyu Li, Yuzhong Qu, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22172)  

**Abstract**: Instruction following (IF) is a critical capability for large language models (LLMs). However, handling complex instructions with multiple constraints remains challenging. Previous methods typically select preference pairs based on the number of constraints they satisfy, introducing noise where chosen examples may fail to follow some constraints and rejected examples may excel in certain respects over the chosen ones. To address the challenge of aligning with multiple preferences, we propose a simple yet effective method called Reverse Preference Optimization (RPO). It mitigates noise in preference pairs by dynamically reversing the constraints within the instruction to ensure the chosen response is perfect, alleviating the burden of extensive sampling and filtering to collect perfect responses. Besides, reversal also enlarges the gap between chosen and rejected responses, thereby clarifying the optimization direction and making it more robust to noise. We evaluate RPO on two multi-turn IF benchmarks, Sysbench and Multi-IF, demonstrating average improvements over the DPO baseline of 4.6 and 2.5 points (on Llama-3.1 8B), respectively. Moreover, RPO scales effectively across model sizes (8B to 70B parameters), with the 70B RPO model surpassing GPT-4o. 

---
# ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments 

**Authors**: Gili Lior, Eliya Habba, Shahar Levy, Avi Caciularu, Gabriel Stanovsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.22169)  

**Abstract**: LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation. 

---
# InComeS: Integrating Compression and Selection Mechanisms into LLMs for Efficient Model Editing 

**Authors**: Shuaiyi Li, Zhisong Zhang, Yang Deng, Chenlong Deng, Tianqing Fang, Hongming Zhang, Haitao Mi, Dong Yu, Wai Lam  

**Link**: [PDF](https://arxiv.org/pdf/2505.22156)  

**Abstract**: Although existing model editing methods perform well in recalling exact edit facts, they often struggle in complex scenarios that require deeper semantic understanding rather than mere knowledge regurgitation. Leveraging the strong contextual reasoning abilities of large language models (LLMs), in-context learning (ICL) becomes a promising editing method by comprehending edit information through context encoding. However, this method is constrained by the limited context window of LLMs, leading to degraded performance and efficiency as the number of edits increases. To overcome this limitation, we propose InComeS, a flexible framework that enhances LLMs' ability to process editing contexts through explicit compression and selection mechanisms. Specifically, InComeS compresses each editing context into the key-value (KV) cache of a special gist token, enabling efficient handling of multiple edits without being restricted by the model's context window. Furthermore, specialized cross-attention modules are added to dynamically select the most relevant information from the gist pools, enabling adaptive and effective utilization of edit information. We conduct experiments on diverse model editing benchmarks with various editing formats, and the results demonstrate the effectiveness and efficiency of our method. 

---
# EULER: Enhancing the Reasoning Ability of Large Language Models through Error-Induced Learning 

**Authors**: Zhuoyang Wu, Xinze Li, Zhenghao Liu, Yukun Yan, Zhiyuan Liu, Minghe Yu, Cheng Yang, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22131)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong reasoning capabilities and achieved promising results in mathematical problem-solving tasks. Learning from errors offers the potential to further enhance the performance of LLMs during Supervised Fine-Tuning (SFT). However, the errors in synthesized solutions are typically gathered from sampling trails, making it challenging to generate solution errors for each mathematical problem. This paper introduces the Error-IndUced LEaRning (EULER) model, which aims to develop an error exposure model that generates high-quality solution errors to enhance the mathematical reasoning capabilities of LLMs. Specifically, EULER optimizes the error exposure model to increase the generation probability of self-made solution errors while utilizing solutions produced by a superior LLM to regularize the generation quality. Our experiments across various mathematical problem datasets demonstrate the effectiveness of the EULER model, achieving an improvement of over 4% compared to all baseline models. Further analysis reveals that EULER is capable of synthesizing more challenging and educational solution errors, which facilitate both the training and inference processes of LLMs. All codes are available at this https URL. 

---
# Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design 

**Authors**: Yudi Zhang, Weilin Zhao, Xu Han, Tiejun Zhao, Wang Xu, Hailong Cao, Conghui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22179)  

**Abstract**: Speculative decoding and quantization effectively accelerate memory-bound inference of large language models. Speculative decoding mitigates the memory bandwidth bottleneck by verifying multiple tokens within a single forward pass, which increases computational effort. Quantization achieves this optimization by compressing weights and activations into lower bit-widths and also reduces computations via low-bit matrix multiplications. To further leverage their strengths, we investigate the integration of these two techniques. Surprisingly, experiments applying the advanced speculative decoding method EAGLE-2 to various quantized models reveal that the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models. This finding led to our new speculative decoding design: a hierarchical framework that employs a small model as an intermediate stage to turn tree-style drafts into sequence drafts, leveraging the memory access benefits of the target quantized model. Experimental results show that our hierarchical approach achieves a 2.78$\times$ speedup across various tasks for the 4-bit weight Llama-3-70B model on an A100 GPU, outperforming EAGLE-2 by 1.31$\times$. Code available at this https URL. 

---
# LoKI: Low-damage Knowledge Implanting of Large Language Models 

**Authors**: Runyu Wang, Peng Ping, Zhengyu Guo, Xiaoye Zhang, Quan Shi, Liting Zhou, Tianbo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.22120)  

**Abstract**: Fine-tuning adapts pretrained models for specific tasks but poses the risk of catastrophic forgetting (CF), where critical knowledge from pre-training is overwritten. Current Parameter-Efficient Fine-Tuning (PEFT) methods for Large Language Models (LLMs), while efficient, often sacrifice general capabilities. To address the issue of CF in a general-purpose PEFT framework, we propose \textbf{Lo}w-damage \textbf{K}nowledge \textbf{I}mplanting (\textbf{LoKI}), a PEFT technique that is based on a mechanistic understanding of how knowledge is stored in transformer architectures. In two real-world scenarios, LoKI demonstrates task-specific performance that is comparable to or even surpasses that of full fine-tuning and LoRA-based methods across various model types, while significantly better preserving general capabilities. Our work connects mechanistic insights into LLM knowledge storage with practical fine-tuning objectives, achieving state-of-the-art trade-offs between task specialization and the preservation of general capabilities. Our implementation is publicly available as ready-to-use code\footnote{this https URL}. 

---
# Curse of High Dimensionality Issue in Transformer for Long-context Modeling 

**Authors**: Shuhai Zhang, Zeng You, Yaofo Chen, Zhiquan Wen, Qianyue Wang, Zhijie Qiu, Yuanqing Li, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.22107)  

**Abstract**: Transformer-based large language models (LLMs) excel in natural language processing tasks by capturing long-range dependencies through self-attention mechanisms. However, long-context modeling faces significant computational inefficiencies due to \textit{redundant} attention computations: while attention weights are often \textit{sparse}, all tokens consume \textit{equal} computational resources. In this paper, we reformulate traditional probabilistic sequence modeling as a \textit{supervised learning task}, enabling the separation of relevant and irrelevant tokens and providing a clearer understanding of redundancy. Based on this reformulation, we theoretically analyze attention sparsity, revealing that only a few tokens significantly contribute to predictions. Building on this, we formulate attention optimization as a linear coding problem and propose a \textit{group coding strategy}, theoretically showing its ability to improve robustness against random noise and enhance learning efficiency. Motivated by this, we propose \textit{Dynamic Group Attention} (DGA), which leverages the group coding to explicitly reduce redundancy by aggregating less important tokens during attention computation. Empirical results show that our DGA significantly reduces computational costs while maintaining competitive this http URL is available at this https URL. 

---
# THINK-Bench: Evaluating Thinking Efficiency and Chain-of-Thought Quality of Large Reasoning Models 

**Authors**: Zhiyuan Li, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22113)  

**Abstract**: Large reasoning models (LRMs) have achieved impressive performance in complex tasks, often outperforming conventional large language models (LLMs). However, the prevalent issue of overthinking severely limits their computational efficiency. Overthinking occurs when models generate excessive and redundant tokens that contribute little to accurate outcomes, especially in simple tasks, resulting in a significant waste of computational resources. To systematically investigate this issue, we introduce Think-Bench, a benchmark designed to evaluate the reasoning efficiency of LRMs. We also propose novel efficiency metrics and conduct a comprehensive evaluation of various LRMs across multiple dimensions, including the reasoning process, outcome quality, and chain-of-thought (CoT) characteristics. Our analysis reveals that most LRMs exhibit overthinking in handling easy questions, generating unnecessarily lengthy reasoning chains. While many LRMs demonstrate high CoT quality, several suffer from low efficiency. We hope that Think-Bench can serve as a robust foundation for advancing research into LRMs. 

---
# Knowledge Base Construction for Knowledge-Augmented Text-to-SQL 

**Authors**: Jinheon Baek, Horst Samulowitz, Oktie Hassanzadeh, Dharmashankar Subramanian, Sola Shirai, Alfio Gliozzo, Debarun Bhattacharjya  

**Link**: [PDF](https://arxiv.org/pdf/2505.22096)  

**Abstract**: Text-to-SQL aims to translate natural language queries into SQL statements, which is practical as it enables anyone to easily retrieve the desired information from databases. Recently, many existing approaches tackle this problem with Large Language Models (LLMs), leveraging their strong capability in understanding user queries and generating corresponding SQL code. Yet, the parametric knowledge in LLMs might be limited to covering all the diverse and domain-specific queries that require grounding in various database schemas, which makes generated SQLs less accurate oftentimes. To tackle this, we propose constructing the knowledge base for text-to-SQL, a foundational source of knowledge, from which we retrieve and generate the necessary knowledge for given queries. In particular, unlike existing approaches that either manually annotate knowledge or generate only a few pieces of knowledge for each query, our knowledge base is comprehensive, which is constructed based on a combination of all the available questions and their associated database schemas along with their relevant knowledge, and can be reused for unseen databases from different datasets and domains. We validate our approach on multiple text-to-SQL datasets, considering both the overlapping and non-overlapping database scenarios, where it outperforms relevant baselines substantially. 

---
# Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO 

**Authors**: Ran Li, Shimin Di, Yuchen Liu, Chen Jing, Yu Qiu, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.22068)  

**Abstract**: Previous study suggest that powerful Large Language Models (LLMs) trained with Reinforcement Learning with Verifiable Rewards (RLVR) only refines reasoning path without improving the reasoning capacity in math tasks while supervised-finetuning(SFT) with distillation can. We study this from the view of Scientific information extraction (SciIE) where LLMs and reasoning LLMs underperforms small Bert-based models. SciIE require both the reasoning and memorization. We argue that both SFT and RLVR can refine the reasoning path and improve reasoning capacity in a simple way based on SciIE. We propose two-stage training with 1. MimicSFT, using structured reasoning templates without needing high-quality chain-of-thought data, 2. R$^2$GRPO with relevance and rule-induced rewards. Experiments on scientific IE benchmarks show that both methods can improve the reasoning capacity. R$^2$GRPO with mimicSFT surpasses baseline LLMs and specialized supervised models in relation extraction. Our code is available at this https URL. 

---
# Jailbreak Distillation: Renewable Safety Benchmarking 

**Authors**: Jingyu Zhang, Ahmed Elgohary, Xiawei Wang, A S M Iftekhar, Ahmed Magooda, Benjamin Van Durme, Daniel Khashabi, Kyle Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2505.22037)  

**Abstract**: Large language models (LLMs) are rapidly deployed in critical applications, raising urgent needs for robust safety benchmarking. We propose Jailbreak Distillation (JBDistill), a novel benchmark construction framework that "distills" jailbreak attacks into high-quality and easily-updatable safety benchmarks. JBDistill utilizes a small set of development models and existing jailbreak attack algorithms to create a candidate prompt pool, then employs prompt selection algorithms to identify an effective subset of prompts as safety benchmarks. JBDistill addresses challenges in existing safety evaluation: the use of consistent evaluation prompts across models ensures fair comparisons and reproducibility. It requires minimal human effort to rerun the JBDistill pipeline and produce updated benchmarks, alleviating concerns on saturation and contamination. Extensive experiments demonstrate our benchmarks generalize robustly to 13 diverse evaluation models held out from benchmark construction, including proprietary, specialized, and newer-generation LLMs, significantly outperforming existing safety benchmarks in effectiveness while maintaining high separability and diversity. Our framework thus provides an effective, sustainable, and adaptable solution for streamlining safety evaluation. 

---
# Legal Assist AI: Leveraging Transformer-Based Model for Effective Legal Assistance 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22003)  

**Abstract**: Pursuit of accessible legal assistance in India faces a critical gap, as many citizens struggle to leverage their legal rights due to limited awareness and access to relevant legal information. This paper introduces Legal Assist AI, a transformer-based model designed to bridge this gap by offering effective legal assistance through large language models (LLMs). The system retrieves relevant legal information from a curated database and generates accurate responses, enabling effective assistance for diverse users, including legal professionals, scholars, and the general public. The model was fine-tuned on extensive datasets from the Indian legal domain, including Indian Constitution, Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS) and so forth, providing a robust understanding of the complexities of Indian law. By incorporating domain-specific legal datasets, the proposed model demonstrated remarkable efficiency and specialization in legal Question-Answering. The model was evaluated against state-of-the-art models such as GPT-3.5 Turbo and Mistral 7B, achieving a 60.08% score on the AIBE, outperforming its competitors in legal reasoning and accuracy. Unlike other models, Legal Assist AI avoided common issues such as hallucinations, making it highly reliable for practical legal applications. It showcases the model's applicability in real-world legal scenarios, with future iterations aiming to enhance performance and expand its dataset to cover a broader range of multilingual and case-specific queries as well. 

---
# Found in Translation: Measuring Multilingual LLM Consistency as Simple as Translate then Evaluate 

**Authors**: Ashim Gupta, Maitrey Mehta, Zhichao Xu, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.21999)  

**Abstract**: Large language models (LLMs) provide detailed and impressive responses to queries in English. However, are they really consistent at responding to the same query in other languages? The popular way of evaluating for multilingual performance of LLMs requires expensive-to-collect annotated datasets. Further, evaluating for tasks like open-ended generation, where multiple correct answers may exist, is nontrivial. Instead, we propose to evaluate the predictability of model response across different languages. In this work, we propose a framework to evaluate LLM's cross-lingual consistency based on a simple Translate then Evaluate strategy. We instantiate this evaluation framework along two dimensions of consistency: information and empathy. Our results reveal pronounced inconsistencies in popular LLM responses across thirty languages, with severe performance deficits in certain language families and scripts, underscoring critical weaknesses in their multilingual capabilities. These findings necessitate cross-lingual evaluations that are consistent along multiple dimensions. We invite practitioners to use our framework for future multilingual LLM benchmarking. 

---
# CoThink: Token-Efficient Reasoning via Instruct Models Guiding Reasoning Models 

**Authors**: Siqi Fan, Peng Han, Shuo Shang, Yequan Wang, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22017)  

**Abstract**: Large language models (LLMs) benefit from increased test-time compute, a phenomenon known as test-time scaling. However, reasoning-optimized models often overthink even simple problems, producing excessively verbose outputs and leading to low token efficiency. By comparing these models with equally sized instruct models, we identify two key causes of this verbosity: (1) reinforcement learning reduces the information density of forward reasoning, and (2) backward chain-of thought training encourages redundant and often unnecessary verification steps. Since LLMs cannot assess the difficulty of a given problem, they tend to apply the same cautious reasoning strategy across all tasks, resulting in inefficient overthinking. To address this, we propose CoThink, an embarrassingly simple pipeline: an instruct model first drafts a high-level solution outline; a reasoning model then works out the solution. We observe that CoThink enables dynamic adjustment of reasoning depth based on input difficulty. Evaluated with three reasoning models DAPO, DeepSeek-R1, and QwQ on three datasets GSM8K, MATH500, and AIME24, CoThink reduces total token generation by 22.3% while maintaining pass@1 accuracy within a 0.42% margin on average. With reference to the instruct model, we formally define reasoning efficiency and observe a potential reasoning efficiency scaling law in LLMs. 

---
# LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents 

**Authors**: Taro Yano, Yoichi Ishibashi, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.21963)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of tasks. To further tailor LLMs to specific domains or applications, post-training techniques such as Supervised Fine-Tuning (SFT), Preference Learning, and model merging are commonly employed. While each of these methods has been extensively studied in isolation, the automated construction of complete post-training pipelines remains an underexplored area. Existing approaches typically rely on manual design or focus narrowly on optimizing individual components, such as data ordering or merging strategies. In this work, we introduce LaMDAgent (short for Language Model Developing Agent), a novel framework that autonomously constructs and optimizes full post-training pipelines through the use of LLM-based agents. LaMDAgent systematically explores diverse model generation techniques, datasets, and hyperparameter configurations, leveraging task-based feedback to discover high-performing pipelines with minimal human intervention. Our experiments show that LaMDAgent improves tool-use accuracy by 9.0 points while preserving instruction-following capabilities. Moreover, it uncovers effective post-training strategies that are often overlooked by conventional human-driven exploration. We further analyze the impact of data and model size scaling to reduce computational costs on the exploration, finding that model size scalings introduces new challenges, whereas scaling data size enables cost-effective pipeline discovery. 

---
# Resolving Knowledge Conflicts in Domain-specific Data Selection: A Case Study on Medical Instruction-tuning 

**Authors**: Qihuang Zhong, Liang Ding, Fei Liao, Juhua Liu, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21958)  

**Abstract**: Domain-specific instruction-tuning has become the defacto standard for improving the performance of large language models (LLMs) in specialized applications, e.g., medical question answering. Since the instruction-tuning dataset might contain redundant or low-quality data, data selection (DS) is usually required to maximize the data efficiency. Despite the successes in the general domain, current DS methods often struggle to select the desired data for domain-specific instruction-tuning. One of the main reasons is that they neglect the impact of knowledge conflicts, i.e., the discrepancy between LLMs' pretrained knowledge and context knowledge of instruction data, which could damage LLMs' prior abilities and lead to hallucination. To this end, we propose a simple-yet-effective Knowledge-aware Data Selection (namely KDS) framework to select the domain-specific instruction-tuning data that meets LLMs' actual needs. The core of KDS is to leverage two knowledge-aware metrics for quantitatively measuring knowledge conflicts from two aspects: context-memory knowledge alignment and intra-memory knowledge consistency. By filtering the data with large knowledge conflicts and sampling the high-quality and diverse data, KDS can effectively stimulate the LLMs' abilities and achieve better domain-specific performance. Taking the medical domain as the testbed, we conduct extensive experiments and empirically prove that KDS surpasses the other baselines and brings significant and consistent performance gains among all LLMs. More encouragingly, KDS effectively improves the model generalization and alleviates the hallucination problem. 

---
# RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering 

**Authors**: Bolei He, Xinran He, Mengke Chen, Xianwei Xue, Ying Zhu, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2505.21940)  

**Abstract**: Large Language Models (LLMs) excel in many areas but continue to face challenges with complex reasoning tasks, such as Multi-Hop Question Answering (MHQA). MHQA requires integrating evidence from diverse sources while managing intricate logical dependencies, often leads to errors in reasoning. Retrieval-Augmented Generation (RAG), widely employed in MHQA tasks, faces challenges in effectively filtering noisy data and retrieving all necessary evidence, thereby limiting its effectiveness in addressing MHQA challenges. To address these challenges, we propose RISE:Reasoning Enhancement via Iterative Self-Exploration, a novel framework designed to enhance models' reasoning capability through iterative self-exploration. Specifically, RISE involves three key steps in addressing MHQA tasks: question decomposition, retrieve-then-read, and self-critique. By leveraging continuous self-exploration, RISE identifies accurate reasoning paths, iteratively self-improving the model's capability to integrate evidence, maintain logical consistency, and enhance performance in MHQA tasks. Extensive experiments on multiple MHQA benchmarks demonstrate that RISE significantly improves reasoning accuracy and task performance. 

---
# Learning to Route Queries Across Knowledge Bases for Step-wise Retrieval-Augmented Reasoning 

**Authors**: Chunyi Peng, Zhipeng Xu, Zhenghao Liu, Yishan Li, Yukun Yan, Shuo Wang, Zhiyuan Liu, Yu Gu, Minghe Yu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22095)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) has shown promise in mitigating hallucinations in Multimodal Large Language Models (MLLMs) by incorporating external knowledge during generation. Existing MRAG methods typically adopt a static retrieval pipeline that fetches relevant information from multiple Knowledge Bases (KBs), followed by a refinement step. However, these approaches overlook the reasoning and planning capabilities of MLLMs to dynamically determine how to interact with different KBs during the reasoning process. To address this limitation, we propose R1-Router, a novel MRAG framework that learns to decide when and where to retrieve knowledge based on the evolving reasoning state. Specifically, R1-Router can generate follow-up queries according to the current reasoning step, routing these intermediate queries to the most suitable KB, and integrating external knowledge into a coherent reasoning trajectory to answer the original query. Furthermore, we introduce Step-wise Group Relative Policy Optimization (Step-GRPO), a tailored reinforcement learning algorithm that assigns step-specific rewards to optimize the reasoning behavior of MLLMs. Experimental results on various open-domain QA benchmarks across multiple modalities demonstrate that R1-Router outperforms baseline models by over 7%. Further analysis shows that R1-Router can adaptively and effectively leverage diverse KBs, reducing unnecessary retrievals and improving both efficiency and accuracy. 

---
# Leveraging Interview-Informed LLMs to Model Survey Responses: Comparative Insights from AI-Generated and Human Data 

**Authors**: Jihong Zhang, Xinya Liang, Anqi Deng, Nicole Bonge, Lin Tan, Ling Zhang, Nicole Zarrett  

**Link**: [PDF](https://arxiv.org/pdf/2505.21997)  

**Abstract**: Mixed methods research integrates quantitative and qualitative data but faces challenges in aligning their distinct structures, particularly in examining measurement characteristics and individual response patterns. Advances in large language models (LLMs) offer promising solutions by generating synthetic survey responses informed by qualitative data. This study investigates whether LLMs, guided by personal interviews, can reliably predict human survey responses, using the Behavioral Regulations in Exercise Questionnaire (BREQ) and interviews from after-school program staff as a case study. Results indicate that LLMs capture overall response patterns but exhibit lower variability than humans. Incorporating interview data improves response diversity for some models (e.g., Claude, GPT), while well-crafted prompts and low-temperature settings enhance alignment between LLM and human responses. Demographic information had less impact than interview content on alignment accuracy. These findings underscore the potential of interview-informed LLMs to bridge qualitative and quantitative methodologies while revealing limitations in response variability, emotional interpretation, and psychometric fidelity. Future research should refine prompt design, explore bias mitigation, and optimize model settings to enhance the validity of LLM-generated survey data in social science research. 

---
# EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse 

**Authors**: Tianyu Guo, Hande Dong, Yichong Leng, Feng Liu, Cheater Lin, Nong Xiao, Xianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21889)  

**Abstract**: Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling this http URL's source code is publicly available at this https URL. 

---
# Principled Content Selection to Generate Diverse and Personalized Multi-Document Summaries 

**Authors**: Vishakh Padmakumar, Zichao Wang, David Arbour, Jennifer Healey  

**Link**: [PDF](https://arxiv.org/pdf/2505.21859)  

**Abstract**: While large language models (LLMs) are increasingly capable of handling longer contexts, recent work has demonstrated that they exhibit the "lost in the middle" phenomenon (Liu et al., 2024) of unevenly attending to different parts of the provided context. This hinders their ability to cover diverse source material in multi-document summarization, as noted in the DiverseSumm benchmark (Huang et al., 2024). In this work, we contend that principled content selection is a simple way to increase source coverage on this task. As opposed to prompting an LLM to perform the summarization in a single step, we explicitly divide the task into three steps -- (1) reducing document collections to atomic key points, (2) using determinantal point processes (DPP) to perform select key points that prioritize diverse content, and (3) rewriting to the final summary. By combining prompting steps, for extraction and rewriting, with principled techniques, for content selection, we consistently improve source coverage on the DiverseSumm benchmark across various LLMs. Finally, we also show that by incorporating relevance to a provided user intent into the DPP kernel, we can generate personalized summaries that cover relevant source information while retaining coverage. 

---
# Evaluating the Retrieval Robustness of Large Language Models 

**Authors**: Shuyang Cao, Karthik Radhakrishnan, David Rosenberg, Steven Lu, Pengxiang Cheng, Lu Wang, Shiyue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21870)  

**Abstract**: Retrieval-augmented generation (RAG) generally enhances large language models' (LLMs) ability to solve knowledge-intensive tasks. But RAG may also lead to performance degradation due to imperfect retrieval and the model's limited ability to leverage retrieved content. In this work, we evaluate the robustness of LLMs in practical RAG setups (henceforth retrieval robustness). We focus on three research questions: (1) whether RAG is always better than non-RAG; (2) whether more retrieved documents always lead to better performance; (3) and whether document orders impact results. To facilitate this study, we establish a benchmark of 1500 open-domain questions, each with retrieved documents from Wikipedia. We introduce three robustness metrics, each corresponds to one research question. Our comprehensive experiments, involving 11 LLMs and 3 prompting strategies, reveal that all of these LLMs exhibit surprisingly high retrieval robustness; nonetheless, different degrees of imperfect robustness hinders them from fully utilizing the benefits of RAG. 

---
# ArgInstruct: Specialized Instruction Fine-Tuning for Computational Argumentation 

**Authors**: Maja Stahl, Timon Ziegenbein, Joonsuk Park, Henning Wachsmuth  

**Link**: [PDF](https://arxiv.org/pdf/2505.22076)  

**Abstract**: Training large language models (LLMs) to follow instructions has significantly enhanced their ability to tackle unseen tasks. However, despite their strong generalization capabilities, instruction-following LLMs encounter difficulties when dealing with tasks that require domain knowledge. This work introduces a specialized instruction fine-tuning for the domain of computational argumentation (CA). The goal is to enable an LLM to effectively tackle any unseen CA tasks while preserving its generalization capabilities. Reviewing existing CA research, we crafted natural language instructions for 105 CA tasks to this end. On this basis, we developed a CA-specific benchmark for LLMs that allows for a comprehensive evaluation of LLMs' capabilities in solving various CA tasks. We synthesized 52k CA-related instructions, adapting the self-instruct process to train a CA-specialized instruction-following LLM. Our experiments suggest that CA-specialized instruction fine-tuning significantly enhances the LLM on both seen and unseen CA tasks. At the same time, performance on the general NLP tasks of the SuperNI benchmark remains stable. 

---
# Calibrating LLM Confidence by Probing Perturbed Representation Stability 

**Authors**: Reza Khanmohammadi, Erfan Miahi, Mehrsa Mardikoraem, Simerjot Kaur, Ivan Brugere, Charese H. Smiley, Kundan Thind, Mohammad M. Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21772)  

**Abstract**: Miscalibration in Large Language Models (LLMs) undermines their reliability, highlighting the need for accurate confidence estimation. We introduce CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability), a novel method analyzing internal representational stability in LLMs. CCPS applies targeted adversarial perturbations to final hidden states, extracts features reflecting the model's response to these perturbations, and uses a lightweight classifier to predict answer correctness. CCPS was evaluated on LLMs from 8B to 32B parameters (covering Llama, Qwen, and Mistral architectures) using MMLU and MMLU-Pro benchmarks in both multiple-choice and open-ended formats. Our results show that CCPS significantly outperforms current approaches. Across four LLMs and three MMLU variants, CCPS reduces Expected Calibration Error by approximately 55% and Brier score by 21%, while increasing accuracy by 5 percentage points, Area Under the Precision-Recall Curve by 4 percentage points, and Area Under the Receiver Operating Characteristic Curve by 6 percentage points, all relative to the strongest prior method. CCPS delivers an efficient, broadly applicable, and more accurate solution for estimating LLM confidence, thereby improving their trustworthiness. 

---
# BehaviorSFT: Behavioral Token Conditioning for Clinical Agents Across the Proactivity Spectrum 

**Authors**: Yubin Kim, Zhiyuan Hu, Hyewon Jeong, Eugene Park, Shuyue Stella Li, Chanwoo Park, Shiyun Xiong, MingYu Lu, Hyeonhoon Lee, Xin Liu, Daniel McDuff, Cynthia Breazeal, Samir Tulebaev, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.21757)  

**Abstract**: Large Language Models (LLMs) as clinical agents require careful behavioral adaptation. While adept at reactive tasks (e.g., diagnosis reasoning), LLMs often struggle with proactive engagement, like unprompted identification of critical missing information or risks. We introduce BehaviorBench, a comprehensive dataset to evaluate agent behaviors across a clinical assistance spectrum, ranging from reactive query responses to proactive interventions (e.g., clarifying ambiguities, flagging overlooked critical data). Our BehaviorBench experiments reveal LLMs' inconsistent proactivity. To address this, we propose BehaviorSFT, a novel training strategy using behavioral tokens to explicitly condition LLMs for dynamic behavioral selection along this spectrum. BehaviorSFT boosts performance, achieving up to 97.3% overall Macro F1 on BehaviorBench and improving proactive task scores (e.g., from 95.0% to 96.5% for Qwen2.5-7B-Ins). Crucially, blind clinician evaluations confirmed BehaviorSFT-trained agents exhibit more realistic clinical behavior, striking a superior balance between helpful proactivity (e.g., timely, relevant suggestions) and necessary restraint (e.g., avoiding over-intervention) versus standard fine-tuning or explicit instructed agents. 

---
# Counterfactual Simulatability of LLM Explanations for Generation Tasks 

**Authors**: Marvin Limpijankit, Yanda Chen, Melanie Subbiah, Nicholas Deas, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2505.21740)  

**Abstract**: LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks. 

---
# MAKIEval: A Multilingual Automatic WiKidata-based Framework for Cultural Awareness Evaluation for LLMs 

**Authors**: Raoyuan Zhao, Beiduo Chen, Barbara Plank, Michael A. Hedderich  

**Link**: [PDF](https://arxiv.org/pdf/2505.21693)  

**Abstract**: Large language models (LLMs) are used globally across many languages, but their English-centric pretraining raises concerns about cross-lingual disparities for cultural awareness, often resulting in biased outputs. However, comprehensive multilingual evaluation remains challenging due to limited benchmarks and questionable translation quality. To better assess these disparities, we introduce MAKIEval, an automatic multilingual framework for evaluating cultural awareness in LLMs across languages, regions, and topics. MAKIEval evaluates open-ended text generation, capturing how models express culturally grounded knowledge in natural language. Leveraging Wikidata's multilingual structure as a cross-lingual anchor, it automatically identifies cultural entities in model outputs and links them to structured knowledge, enabling scalable, language-agnostic evaluation without manual annotation or translation. We then introduce four metrics that capture complementary dimensions of cultural awareness: granularity, diversity, cultural specificity, and consensus across languages. We assess 7 LLMs developed from different parts of the world, encompassing both open-source and proprietary systems, across 13 languages, 19 countries and regions, and 6 culturally salient topics (e.g., food, clothing). Notably, we find that models tend to exhibit stronger cultural awareness in English, suggesting that English prompts more effectively activate culturally grounded knowledge. We publicly release our code and data. 

---
# Do We Know What LLMs Don't Know? A Study of Consistency in Knowledge Probing 

**Authors**: Raoyuan Zhao, Abdullatif Köksal, Ali Modarressi, Michael A. Hedderich, Hinrich Schütze  

**Link**: [PDF](https://arxiv.org/pdf/2505.21701)  

**Abstract**: The reliability of large language models (LLMs) is greatly compromised by their tendency to hallucinate, underscoring the need for precise identification of knowledge gaps within LLMs. Various methods for probing such gaps exist, ranging from calibration-based to prompting-based methods. To evaluate these probing methods, in this paper, we propose a new process based on using input variations and quantitative metrics. Through this, we expose two dimensions of inconsistency in knowledge gap probing. (1) Intra-method inconsistency: Minimal non-semantic perturbations in prompts lead to considerable variance in detected knowledge gaps within the same probing method; e.g., the simple variation of shuffling answer options can decrease agreement to around 40%. (2) Cross-method inconsistency: Probing methods contradict each other on whether a model knows the answer. Methods are highly inconsistent -- with decision consistency across methods being as low as 7% -- even though the model, dataset, and prompt are all the same. These findings challenge existing probing methods and highlight the urgent need for perturbation-robust probing frameworks. 

---
# LLMPR: A Novel LLM-Driven Transfer Learning based Petition Ranking Model 

**Authors**: Avijit Gayen, Somyajit Chakraborty, Mainak Sen, Soham Paul, Angshuman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2505.21689)  

**Abstract**: The persistent accumulation of unresolved legal cases, especially within the Indian judiciary, significantly hampers the timely delivery of justice. Manual methods of prioritizing petitions are often prone to inefficiencies and subjective biases further exacerbating delays. To address this issue, we propose LLMPR (Large Language Model-based Petition Ranking), an automated framework that utilizes transfer learning and machine learning to assign priority rankings to legal petitions based on their contextual urgency. Leveraging the ILDC dataset comprising 7,593 annotated petitions, we process unstructured legal text and extract features through various embedding techniques, including DistilBERT, LegalBERT, and MiniLM. These textual embeddings are combined with quantitative indicators such as gap days, rank scores, and word counts to train multiple machine learning models, including Random Forest, Decision Tree, XGBoost, LightGBM, and CatBoost. Our experiments demonstrate that Random Forest and Decision Tree models yield superior performance, with accuracy exceeding 99% and a Spearman rank correlation of 0.99. Notably, models using only numerical features achieve nearly optimal ranking results (R2 = 0.988, \r{ho} = 0.998), while LLM-based embeddings offer only marginal gains. These findings suggest that automated petition ranking can effectively streamline judicial workflows, reduce case backlog, and improve fairness in legal prioritization. 

---
# Explainability of Large Language Models using SMILE: Statistical Model-agnostic Interpretability with Local Explanations 

**Authors**: Zeinab Dehghani, Koorosh Aslansefat, Adil Khan, Mohammed Naveed Akram  

**Link**: [PDF](https://arxiv.org/pdf/2505.21657)  

**Abstract**: Large language models like GPT, LLAMA, and Claude have become incredibly powerful at generating text, but they are still black boxes, so it is hard to understand how they decide what to say. That lack of transparency can be problematic, especially in fields where trust and accountability matter. To help with this, we introduce SMILE, a new method that explains how these models respond to different parts of a prompt. SMILE is model-agnostic and works by slightly changing the input, measuring how the output changes, and then highlighting which words had the most impact. Create simple visual heat maps showing which parts of a prompt matter the most. We tested SMILE on several leading LLMs and used metrics such as accuracy, consistency, stability, and fidelity to show that it gives clear and reliable explanations. By making these models easier to understand, SMILE brings us one step closer to making AI more transparent and trustworthy. 

---
# How does Misinformation Affect Large Language Model Behaviors and Preferences? 

**Authors**: Miao Peng, Nuo Chen, Jianheng Tang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21608)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in knowledge-intensive tasks, while they remain vulnerable when encountering misinformation. Existing studies have explored the role of LLMs in combating misinformation, but there is still a lack of fine-grained analysis on the specific aspects and extent to which LLMs are influenced by misinformation. To bridge this gap, we present MisBench, the current largest and most comprehensive benchmark for evaluating LLMs' behavior and knowledge preference toward misinformation. MisBench consists of 10,346,712 pieces of misinformation, which uniquely considers both knowledge-based conflicts and stylistic variations in misinformation. Empirical results reveal that while LLMs demonstrate comparable abilities in discerning misinformation, they still remain susceptible to knowledge conflicts and stylistic variations. Based on these findings, we further propose a novel approach called Reconstruct to Discriminate (RtD) to strengthen LLMs' ability to detect misinformation. Our study provides valuable insights into LLMs' interactions with misinformation, and we believe MisBench can serve as an effective benchmark for evaluating LLM-based detectors and enhancing their reliability in real-world applications. Codes and data are available at this https URL. 

---
# 3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model 

**Authors**: Wenbo Hu, Yining Hong, Yanjun Wang, Leison Gao, Zibu Wei, Xingcheng Yao, Nanyun Peng, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22657)  

**Abstract**: Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks. 

---
# More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models 

**Authors**: Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, Sheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21523)  

**Abstract**: Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more heavily on language priors. Attention analysis shows that longer reasoning chains lead to reduced focus on visual inputs, which contributes to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model's perception accuracy changes with reasoning length, allowing us to evaluate whether the model preserves visual grounding during reasoning. We also release RH-Bench, a diagnostic benchmark that spans a variety of multimodal tasks, designed to assess the trade-off between reasoning ability and hallucination. Our analysis reveals that (i) larger models typically achieve a better balance between reasoning and perception, and (ii) this balance is influenced more by the types and domains of training data than by its overall volume. These findings underscore the importance of evaluation frameworks that jointly consider both reasoning quality and perceptual fidelity. 

---
# R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing 

**Authors**: Tianyu Fu, Yi Ge, Yichen You, Enshu Liu, Zhihang Yuan, Guohao Dai, Shengen Yan, Huazhong Yang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21600)  

**Abstract**: Large Language Models (LLMs) achieve impressive reasoning capabilities at the cost of substantial inference overhead, posing substantial deployment challenges. Although distilled Small Language Models (SLMs) significantly enhance efficiency, their performance suffers as they fail to follow LLMs' reasoning paths. Luckily, we reveal that only a small fraction of tokens genuinely diverge reasoning paths between LLMs and SLMs. Most generated tokens are either identical or exhibit neutral differences, such as minor variations in abbreviations or expressions. Leveraging this insight, we introduce **Roads to Rome (R2R)**, a neural token routing method that selectively utilizes LLMs only for these critical, path-divergent tokens, while leaving the majority of token generation to the SLM. We also develop an automatic data generation pipeline that identifies divergent tokens and generates token-level routing labels to train the lightweight router. We apply R2R to combine R1-1.5B and R1-32B models from the DeepSeek family, and evaluate on challenging math, coding, and QA benchmarks. With an average activated parameter size of 5.6B, R2R surpasses the average accuracy of R1-7B by 1.6x, outperforming even the R1-14B model. Compared to R1-32B, it delivers a 2.8x wall-clock speedup with comparable performance, advancing the Pareto frontier of test-time scaling efficiency. Our code is available at this https URL. 

---
# Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents 

**Authors**: Michael Kirchhof, Gjergji Kasneci, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.22655)  

**Abstract**: Large-language models (LLMs) and chatbot agents are known to provide wrong outputs at times, and it was recently found that this can never be fully prevented. Hence, uncertainty quantification plays a crucial role, aiming to quantify the level of ambiguity in either one overall number or two numbers for aleatoric and epistemic uncertainty. This position paper argues that this traditional dichotomy of uncertainties is too limited for the open and interactive setup that LLM agents operate in when communicating with a user, and that we need to research avenues that enrich uncertainties in this novel scenario. We review the literature and find that popular definitions of aleatoric and epistemic uncertainties directly contradict each other and lose their meaning in interactive LLM agent settings. Hence, we propose three novel research directions that focus on uncertainties in such human-computer interactions: Underspecification uncertainties, for when users do not provide all information or define the exact task at the first go, interactive learning, to ask follow-up questions and reduce the uncertainty about the current context, and output uncertainties, to utilize the rich language and speech space to express uncertainties as more than mere numbers. We expect that these new ways of dealing with and communicating uncertainties will lead to LLM agent interactions that are more transparent, trustworthy, and intuitive. 

---
# Scaling Reasoning without Attention 

**Authors**: Xueliang Zhao, Wei Wu, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22425)  

**Abstract**: Large language models (LLMs) have made significant advances in complex reasoning tasks, yet they remain bottlenecked by two core challenges: architectural inefficiency due to reliance on Transformers, and a lack of structured fine-tuning for high-difficulty domains. We introduce \ourmodel, an attention-free language model that addresses both issues through architectural and data-centric innovations. Built on the state space dual (SSD) layers of Mamba-2, our model eliminates the need for self-attention and key-value caching, enabling fixed-memory, constant-time inference. To train it for complex reasoning, we propose a two-phase curriculum fine-tuning strategy based on the \textsc{PromptCoT} synthesis paradigm, which generates pedagogically structured problems via abstract concept selection and rationale-guided generation. On benchmark evaluations, \ourmodel-7B outperforms strong Transformer and hybrid models of comparable scale, and even surpasses the much larger Gemma3-27B by 2.6\% on AIME 24, 0.6\% on AIME 25, and 3.0\% on Livecodebench. These results highlight the potential of state space models as efficient and scalable alternatives to attention-based architectures for high-capacity reasoning. 

---
# Skywork Open Reasoner 1 Technical Report 

**Authors**: Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.22312)  

**Abstract**: The success of DeepSeek-R1 underscores the significant role of reinforcement learning (RL) in enhancing the reasoning capabilities of large language models (LLMs). In this work, we present Skywork-OR1, an effective and scalable RL implementation for long Chain-of-Thought (CoT) models. Building on the DeepSeek-R1-Distill model series, our RL approach achieves notable performance gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%) for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models demonstrate competitive reasoning capabilities among models of similar size. We perform comprehensive ablation studies on the core components of our training pipeline to validate their effectiveness. Additionally, we thoroughly investigate the phenomenon of entropy collapse, identify key factors affecting entropy dynamics, and demonstrate that mitigating premature entropy collapse is critical for improved test performance. To support community research, we fully open-source our model weights, training code, and training datasets. 

---
# Rethinking the Unsolvable: When In-Context Search Meets Test-Time Scaling 

**Authors**: Fanzeng Xia, Yidong Luo, Tinko Sebastian Bartels, Yaqi Xu, Tongxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22290)  

**Abstract**: Recent research has highlighted that Large Language Models (LLMs), even when trained to generate extended long reasoning steps, still face significant challenges on hard reasoning problems. However, much of the existing literature relies on direct prompting with simple in-context learning examples for evaluation, which largely overlooks advanced techniques to elicit LLMs' deliberate reasoning before drawing conclusions that LLMs hit a performance ceiling. In this paper, we systematically explore the combined potential of in-context search and test-time scaling on super hard reasoning tasks. We find that by employing advanced in-context search prompting to LLMs augmented with internal scaling, one can achieve transformative performance breakthroughs on tasks previously deemed "unsolvable" (e.g., reported success rates below 5%). We provide both empirical results and theoretical analysis of how this combination can unleash LLM reasoning capabilities: i) Empirically, on controlled NP-hard tasks and complex real-world planning benchmarks, our approach achieves up to a 30x improvement in success rates compared to previously reported results without any external mechanisms; ii) Theoretically, we show that in-context search prompting, when combined with internal scaling, significantly extends the complexity class of solvable reasoning problems. These findings challenge prevailing assumptions about the limitations of LLMs on complex tasks, indicating that current evaluation paradigms systematically underestimate their true potential. Our work calls for a critical reassessment of how LLM reasoning is benchmarked and a more robust evaluation strategy that fully captures the true capabilities of contemporary LLMs, which can lead to a better understanding of their operational reasoning boundaries in real-world deployments. 

---
# Evaluation of LLMs in Speech is Often Flawed: Test Set Contamination in Large Language Models for Speech Recognition 

**Authors**: Yuan Tseng, Titouan Parcollet, Rogier van Dalen, Shucong Zhang, Sourav Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2505.22251)  

**Abstract**: Recent work suggests that large language models (LLMs) can improve performance of speech tasks compared to existing systems. To support their claims, results on LibriSpeech and Common Voice are often quoted. However, this work finds that a substantial amount of the LibriSpeech and Common Voice evaluation sets appear in public LLM pretraining corpora. This calls into question the reliability of findings drawn from these two datasets. To measure the impact of contamination, LLMs trained with or without contamination are compared, showing that a contaminated LLM is more likely to generate test sentences it has seen during training. Speech recognisers using contaminated LLMs shows only subtle differences in error rates, but assigns significantly higher probabilities to transcriptions seen during training. Results show that LLM outputs can be biased by tiny amounts of data contamination, highlighting the importance of evaluating LLM-based speech systems with held-out data. 

---
# Improving Brain-to-Image Reconstruction via Fine-Grained Text Bridging 

**Authors**: Runze Xia, Shuo Feng, Renzhi Wang, Congchi Yin, Xuyun Wen, Piji Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22150)  

**Abstract**: Brain-to-Image reconstruction aims to recover visual stimuli perceived by humans from brain activity. However, the reconstructed visual stimuli often missing details and semantic inconsistencies, which may be attributed to insufficient semantic information. To address this issue, we propose an approach named Fine-grained Brain-to-Image reconstruction (FgB2I), which employs fine-grained text as bridge to improve image reconstruction. FgB2I comprises three key stages: detail enhancement, decoding fine-grained text descriptions, and text-bridged brain-to-image reconstruction. In the detail-enhancement stage, we leverage large vision-language models to generate fine-grained captions for visual stimuli and experimentally validate its importance. We propose three reward metrics (object accuracy, text-image semantic similarity, and image-image semantic similarity) to guide the language model in decoding fine-grained text descriptions from fMRI signals. The fine-grained text descriptions can be integrated into existing reconstruction methods to achieve fine-grained Brain-to-Image reconstruction. 

---
# Rethinking the Outlier Distribution in Large Language Models: An In-depth Study 

**Authors**: Rahul Raman, Khushi Sharma, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21670)  

**Abstract**: Investigating outliers in large language models (LLMs) is crucial due to their significant impact on various aspects of LLM performance, including quantization and compression. Outliers often cause considerable quantization errors, leading to degraded model performance. Identifying and addressing these outliers can enhance the accuracy and efficiency of the quantization process, enabling smoother deployment on edge devices or specialized hardware. Recent studies have identified two common types of outliers in LLMs: massive activations and channel-wise outliers. While numerous quantization algorithms have been proposed to mitigate their effects and maintain satisfactory accuracy, few have thoroughly explored the root causes of these outliers in depth. In this paper, we conduct a comprehensive investigation into the formation mechanisms of these outliers and propose potential strategies to mitigate their occurrence. Ultimately, we introduce some efficient approaches to eliminate most massive activations and channel-wise outliers with minimal impact on accuracy. 

---
# Learning Compositional Behaviors from Demonstration and Language 

**Authors**: Weiyu Liu, Neil Nie, Ruohan Zhang, Jiayuan Mao, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21981)  

**Abstract**: We introduce Behavior from Language and Demonstration (BLADE), a framework for long-horizon robotic manipulation by integrating imitation learning and model-based planning. BLADE leverages language-annotated demonstrations, extracts abstract action knowledge from large language models (LLMs), and constructs a library of structured, high-level action representations. These representations include preconditions and effects grounded in visual perception for each high-level action, along with corresponding controllers implemented as neural network-based policies. BLADE can recover such structured representations automatically, without manually labeled states or symbolic definitions. BLADE shows significant capabilities in generalizing to novel situations, including novel initial states, external state perturbations, and novel goals. We validate the effectiveness of our approach both in simulation and on real robots with a diverse set of objects with articulated parts, partial observability, and geometric constraints. 

---
# EnsemW2S: Enhancing Weak-to-Strong Generalization with Large Language Model Ensembles 

**Authors**: Aakriti Agrawal, Mucong Ding, Zora Che, Chenghao Deng, Anirudh Satheesh, Bang An, Bayan Bruss, John Langford, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21959)  

**Abstract**: With Large Language Models (LLMs) rapidly approaching and potentially surpassing human-level performance, it has become imperative to develop approaches capable of effectively supervising and enhancing these powerful models using smaller, human-level models exposed to only human-level data. We address this critical weak-to-strong (W2S) generalization challenge by proposing a novel method aimed at improving weak experts, by training on the same limited human-level data, enabling them to generalize to complex, super-human-level tasks. Our approach, called \textbf{EnsemW2S}, employs a token-level ensemble strategy that iteratively combines multiple weak experts, systematically addressing the shortcomings identified in preceding iterations. By continuously refining these weak models, we significantly enhance their collective ability to supervise stronger student models. We extensively evaluate the generalization performance of both the ensemble of weak experts and the subsequent strong student model across in-distribution (ID) and out-of-distribution (OOD) datasets. For OOD, we specifically introduce question difficulty as an additional dimension for defining distributional shifts. Our empirical results demonstrate notable improvements, achieving 4\%, and 3.2\% improvements on ID datasets and, upto 6\% and 2.28\% on OOD datasets for experts and student models respectively, underscoring the effectiveness of our proposed method in advancing W2S generalization. 

---
# Look & Mark: Leveraging Radiologist Eye Fixations and Bounding boxes in Multimodal Large Language Models for Chest X-ray Report Generation 

**Authors**: Yunsoo Kim, Jinge Wu, Su-Hwan Kim, Pardeep Vasudev, Jiashu Shen, Honghan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22222)  

**Abstract**: Recent advancements in multimodal Large Language Models (LLMs) have significantly enhanced the automation of medical image analysis, particularly in generating radiology reports from chest X-rays (CXR). However, these models still suffer from hallucinations and clinically significant errors, limiting their reliability in real-world applications. In this study, we propose Look & Mark (L&M), a novel grounding fixation strategy that integrates radiologist eye fixations (Look) and bounding box annotations (Mark) into the LLM prompting framework. Unlike conventional fine-tuning, L&M leverages in-context learning to achieve substantial performance gains without retraining. When evaluated across multiple domain-specific and general-purpose models, L&M demonstrates significant gains, including a 1.2% improvement in overall metrics (this http URL) for CXR-LLaVA compared to baseline prompting and a remarkable 9.2% boost for LLaVA-Med. General-purpose models also benefit from L&M combined with in-context learning, with LLaVA-OV achieving an 87.3% clinical average performance (this http URL)-the highest among all models, even surpassing those explicitly trained for CXR report generation. Expert evaluations further confirm that L&M reduces clinically significant errors (by 0.43 average errors per report), such as false predictions and omissions, enhancing both accuracy and reliability. These findings highlight L&M's potential as a scalable and efficient solution for AI-assisted radiology, paving the way for improved diagnostic workflows in low-resource clinical settings. 

---
# Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation 

**Authors**: Yu-Lun Song, Chung-En Tsern, Che-Cheng Wu, Yu-Ming Chang, Syuan-Bo Huang, Wei-Chu Chen, Michael Chia-Liang Lin, Yu-Ta Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.21880)  

**Abstract**: This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications. 

---
# Mitigating Overthinking in Large Reasoning Models via Manifold Steering 

**Authors**: Yao Huang, Huanran Chen, Shouwei Ruan, Yichi Zhang, Xingxing Wei, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22411)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in solving complex tasks such as mathematics and coding. However, these models frequently exhibit a phenomenon known as overthinking during inference, characterized by excessive validation loops and redundant deliberation, leading to substantial computational overheads. In this paper, we aim to mitigate overthinking by investigating the underlying mechanisms from the perspective of mechanistic interpretability. We first showcase that the tendency of overthinking can be effectively captured by a single direction in the model's activation space and the issue can be eased by intervening the activations along this direction. However, this efficacy soon reaches a plateau and even deteriorates as the intervention strength increases. We therefore systematically explore the activation space and find that the overthinking phenomenon is actually tied to a low-dimensional manifold, which indicates that the limited effect stems from the noises introduced by the high-dimensional steering direction. Based on this insight, we propose Manifold Steering, a novel approach that elegantly projects the steering direction onto the low-dimensional activation manifold given the theoretical approximation of the interference noise. Extensive experiments on DeepSeek-R1 distilled models validate that our method reduces output tokens by up to 71% while maintaining and even improving the accuracy on several mathematical benchmarks. Our method also exhibits robust cross-domain transferability, delivering consistent token reduction performance in code generation and knowledge-based QA tasks. Code is available at: this https URL. 

---
# Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking 

**Authors**: Yunyi Zhang, Ruozhen Yang, Siqi Jiao, SeongKu Kang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.21815)  

**Abstract**: Scientific paper retrieval is essential for supporting literature discovery and research. While dense retrieval methods demonstrate effectiveness in general-purpose tasks, they often fail to capture fine-grained scientific concepts that are essential for accurate understanding of scientific queries. Recent studies also use large language models (LLMs) for query understanding; however, these methods often lack grounding in corpus-specific knowledge and may generate unreliable or unfaithful content. To overcome these limitations, we propose SemRank, an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy. Experiments show that SemRank consistently improves the performance of various base retrievers, surpasses strong existing LLM-based baselines, and remains highly efficient. 

---
# From Directions to Cones: Exploring Multidimensional Representations of Propositional Facts in LLMs 

**Authors**: Stanley Yu, Vaidehi Bulusu, Oscar Yasunaga, Clayton Lau, Cole Blondin, Sean O'Brien, Kevin Zhu, Vasu Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2505.21800)  

**Abstract**: Large Language Models (LLMs) exhibit strong conversational abilities but often generate falsehoods. Prior work suggests that the truthfulness of simple propositions can be represented as a single linear direction in a model's internal activations, but this may not fully capture its underlying geometry. In this work, we extend the concept cone framework, recently introduced for modeling refusal, to the domain of truth. We identify multi-dimensional cones that causally mediate truth-related behavior across multiple LLM families. Our results are supported by three lines of evidence: (i) causal interventions reliably flip model responses to factual statements, (ii) learned cones generalize across model architectures, and (iii) cone-based interventions preserve unrelated model behavior. These findings reveal the richer, multidirectional structure governing simple true/false propositions in LLMs and highlight concept cones as a promising tool for probing abstract behaviors. 

---
# Test-Time Immunization: A Universal Defense Framework Against Jailbreaks for (Multimodal) Large Language Models 

**Authors**: Yongcan Yu, Yanbo Wang, Ran He, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22271)  

**Abstract**: While (multimodal) large language models (LLMs) have attracted widespread attention due to their exceptional capabilities, they remain vulnerable to jailbreak attacks. Various defense methods are proposed to defend against jailbreak attacks, however, they are often tailored to specific types of jailbreak attacks, limiting their effectiveness against diverse adversarial strategies. For instance, rephrasing-based defenses are effective against text adversarial jailbreaks but fail to counteract image-based attacks. To overcome these limitations, we propose a universal defense framework, termed Test-time IMmunization (TIM), which can adaptively defend against various jailbreak attacks in a self-evolving way. Specifically, TIM initially trains a gist token for efficient detection, which it subsequently applies to detect jailbreak activities during inference. When jailbreak attempts are identified, TIM implements safety fine-tuning using the detected jailbreak instructions paired with refusal answers. Furthermore, to mitigate potential performance degradation in the detector caused by parameter updates during safety fine-tuning, we decouple the fine-tuning process from the detection module. Extensive experiments on both LLMs and multimodal LLMs demonstrate the efficacy of TIM. 

---
# Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation 

**Authors**: Tharindu Kumarage, Ninareh Mehrabi, Anil Ramakrishna, Xinyan Zhao, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta, Charith Peris  

**Link**: [PDF](https://arxiv.org/pdf/2505.21784)  

**Abstract**: Safety reasoning is a recent paradigm where LLMs reason over safety policies before generating responses, thereby mitigating limitations in existing safety measures such as over-refusal and jailbreak vulnerabilities. However, implementing this paradigm is challenging due to the resource-intensive process of creating high-quality policy-embedded chain-of-thought (CoT) datasets while ensuring reasoning remains accurate and free from hallucinations or policy conflicts. To tackle this, we propose AIDSAFE: Agentic Iterative Deliberation for Safety Reasoning, a novel data generation recipe that leverages multi-agent deliberation to iteratively expand reasoning on safety policies. A data refiner stage in AIDSAFE ensures high-quality outputs by eliminating repetitive, redundant, and deceptive thoughts. AIDSAFE-generated CoTs provide a strong foundation for supervised fine-tuning (SFT)-based safety training. Additionally, to address the need of preference data in alignment stages, such as DPO training, we introduce a supplemental recipe that uses belief augmentation to create distinct selected and rejected CoT samples. Our evaluations demonstrate that AIDSAFE-generated CoTs achieve superior policy adherence and reasoning quality. Consequently, we show that fine-tuning open-source LLMs on these CoTs can significantly improve safety generalization and jailbreak robustness while maintaining acceptable utility and over-refusal accuracy. AIDSAFE-generated CoT datasets can be found here: this https URL 

---
# ChemHAS: Hierarchical Agent Stacking for Enhancing Chemistry Tools 

**Authors**: Zhucong Li, Bowei Zhang, Jin Xiao, Zhijian Zhou, Fenglei Cao, Jiaqing Liang, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21569)  

**Abstract**: Large Language Model (LLM)-based agents have demonstrated the ability to improve performance in chemistry-related tasks by selecting appropriate tools. However, their effectiveness remains limited by the inherent prediction errors of chemistry tools. In this paper, we take a step further by exploring how LLMbased agents can, in turn, be leveraged to reduce prediction errors of the tools. To this end, we propose ChemHAS (Chemical Hierarchical Agent Stacking), a simple yet effective method that enhances chemistry tools through optimizing agent-stacking structures from limited data. ChemHAS achieves state-of-the-art performance across four fundamental chemistry tasks, demonstrating that our method can effectively compensate for prediction errors of the tools. Furthermore, we identify and characterize four distinct agent-stacking behaviors, potentially improving interpretability and revealing new possibilities for AI agent applications in scientific research. Our code and dataset are publicly available at https: //anonymous.this http URL. 

---
# R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning 

**Authors**: Yongchao Chen, Yueying Liu, Junwei Zhou, Yilun Hao, Jingquan Wang, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21668)  

**Abstract**: Despite advances in reasoning and planning of R1-like models, Large Language Models (LLMs) still struggle with tasks requiring precise computation, symbolic manipulation, optimization, and algorithmic reasoning, in which textual reasoning lacks the rigor of code execution. A key challenge is enabling LLMs to decide when to use textual reasoning versus code generation. While OpenAI trains models to invoke a Code Interpreter as needed, public research lacks guidance on aligning pre-trained LLMs to effectively leverage code and generalize across diverse tasks. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. We curate 144 reasoning and planning tasks (107 for training, 37 for testing), each with over 200 diverse questions. We fine-tune Qwen-2.5 models (3B/7B/14B) using various SFT and RL strategies, investigating different answer formats, reasoning vs. non-reasoning models, cold vs. warm starts, GRPO vs. PPO, and masked vs. unmasked code outputs. Unlike prior RL work on narrow domains, we find that Code Interpreter training is significantly harder due to high task diversity and expensive code execution, highlighting the critical role of the SFT stage. Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.0\% to 64.1\%, outperforming GPT-4o (text-only: 58.6\%) and approaching GPT-4o with Code Interpreter (70.9\%), with the emergent self-checking behavior via code generation. Datasets, Codes, and Models are available at this https URL and this https URL. 

---
# Capability-Based Scaling Laws for LLM Red-Teaming 

**Authors**: Alexander Panfilov, Paul Kassianik, Maksym Andriushchenko, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2505.20162)  

**Abstract**: As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers. 

---
# How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control 

**Authors**: Kunhang Li, Jason Naradowsky, Yansong Feng, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21531)  

**Abstract**: We explore Large Language Models (LLMs)' human motion knowledge through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations as a clear verification lens for human evaluators. Through carefully designed 20 representative motion instructions with full coverage of basic movement primitives and balanced body part usage, we conduct comprehensive evaluations including human assessment of both generated animations and high-level movement plans, as well as automatic comparison with oracle positions in low-level planning. We find that LLMs are strong at interpreting the high-level body movements but struggle with precise body part positioning. While breaking down motion queries into atomic components improves planning performance, LLMs have difficulty with multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximation for general spatial descriptions, but fail to handle precise spatial specifications in text, and the precise spatial-temporal parameters needed for avatar control. Notably, LLMs show promise in conceptualizing creative motions and distinguishing culturally-specific motion patterns. 

---
# Vision Meets Language: A RAG-Augmented YOLOv8 Framework for Coffee Disease Diagnosis and Farmer Assistance 

**Authors**: Semanto Mondal  

**Link**: [PDF](https://arxiv.org/pdf/2505.21544)  

**Abstract**: As a social being, we have an intimate bond with the environment. A plethora of things in human life, such as lifestyle, health, and food are dependent on the environment and agriculture. It comes under our responsibility to support the environment as well as agriculture. However, traditional farming practices often result in inefficient resource use and environmental challenges. To address these issues, precision agriculture has emerged as a promising approach that leverages advanced technologies to optimise agricultural processes. In this work, a hybrid approach is proposed that combines the three different potential fields of model AI: object detection, large language model (LLM), and Retrieval-Augmented Generation (RAG). In this novel framework, we have tried to combine the vision and language models to work together to identify potential diseases in the tree leaf. This study introduces a novel AI-based precision agriculture system that uses Retrieval Augmented Generation (RAG) to provide context-aware diagnoses and natural language processing (NLP) and YOLOv8 for crop disease detection. The system aims to tackle major issues with large language models (LLMs), especially hallucinations and allows for adaptive treatment plans and real-time disease detection. The system provides an easy-to-use interface to the farmers, which they can use to detect the different diseases related to coffee leaves by just submitting the image of the affected leaf the model will detect the diseases as well as suggest potential remediation methodologies which aim to lower the use of pesticides, preserving livelihoods, and encouraging environmentally friendly methods. With an emphasis on scalability, dependability, and user-friendliness, the project intends to improve RAG-integrated object detection systems for wider agricultural applications in the future. 

---
# Born a Transformer -- Always a Transformer? 

**Authors**: Yana Veitsman, Mayank Jobanputra, Yash Sarrof, Aleksandra Bakalova, Vera Demberg, Ellie Pavlick, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.21785)  

**Abstract**: Transformers have theoretical limitations in modeling certain sequence-to-sequence tasks, yet it remains largely unclear if these limitations play a role in large-scale pretrained LLMs, or whether LLMs might effectively overcome these constraints in practice due to the scale of both the models themselves and their pretraining data. We explore how these architectural constraints manifest after pretraining, by studying a family of $\textit{retrieval}$ and $\textit{copying}$ tasks inspired by Liu et al. [2024]. We use the recently proposed C-RASP framework for studying length generalization [Huang et al., 2025b] to provide guarantees for each of our settings. Empirically, we observe an $\textit{induction-versus-anti-induction}$ asymmetry, where pretrained models are better at retrieving tokens to the right (induction) rather than the left (anti-induction) of a query token. This asymmetry disappears upon targeted fine-tuning if length-generalization is guaranteed by theory. Mechanistic analysis reveals that this asymmetry is connected to the differences in the strength of induction versus anti-induction circuits within pretrained Transformers. We validate our findings through practical experiments on real-world tasks demonstrating reliability risks. Our results highlight that pretraining selectively enhances certain Transformer capabilities, but does not overcome fundamental length-generalization limits. 

---
# Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones 

**Authors**: Parsa Mirtaheri, Ezra Edelman, Samy Jelassi, Eran Malach, Enric Boix-Adsera  

**Link**: [PDF](https://arxiv.org/pdf/2505.21825)  

**Abstract**: Inference-time computation has emerged as a promising scaling axis for improving large language model reasoning. However, despite yielding impressive performance, the optimal allocation of inference-time computation remains poorly understood. A central question is whether to prioritize sequential scaling (e.g., longer chains of thought) or parallel scaling (e.g., majority voting across multiple short chains of thought). In this work, we seek to illuminate the landscape of test-time scaling by demonstrating the existence of reasoning settings where sequential scaling offers an exponential advantage over parallel scaling. These settings are based on graph connectivity problems in challenging distributions of graphs. We validate our theoretical findings with comprehensive experiments across a range of language models, including models trained from scratch for graph connectivity with different chain of thought strategies as well as large reasoning models. 

---
# From prosthetic memory to prosthetic denial: Auditing whether large language models are prone to mass atrocity denialism 

**Authors**: Roberto Ulloa, Eve M. Zucker, Daniel Bultmann, David J. Simon, Mykola Makhortykh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21753)  

**Abstract**: The proliferation of large language models (LLMs) can influence how historical narratives are disseminated and perceived. This study explores the implications of LLMs' responses on the representation of mass atrocity memory, examining whether generative AI systems contribute to prosthetic memory, i.e., mediated experiences of historical events, or to what we term "prosthetic denial," the AI-mediated erasure or distortion of atrocity memories. We argue that LLMs function as interfaces that can elicit prosthetic memories and, therefore, act as experiential sites for memory transmission, but also introduce risks of denialism, particularly when their outputs align with contested or revisionist narratives. To empirically assess these risks, we conducted a comparative audit of five LLMs (Claude, GPT, Llama, Mixtral, and Gemini) across four historical case studies: the Holodomor, the Holocaust, the Cambodian Genocide, and the genocide against the Tutsis in Rwanda. Each model was prompted with questions addressing common denialist claims in English and an alternative language relevant to each case (Ukrainian, German, Khmer, and French). Our findings reveal that while LLMs generally produce accurate responses for widely documented events like the Holocaust, significant inconsistencies and susceptibility to denialist framings are observed for more underrepresented cases like the Cambodian Genocide. The disparities highlight the influence of training data availability and the probabilistic nature of LLM responses on memory integrity. We conclude that while LLMs extend the concept of prosthetic memory, their unmoderated use risks reinforcing historical denialism, raising ethical concerns for (digital) memory preservation, and potentially challenging the advantageous role of technology associated with the original values of prosthetic memory. 

---
# What Makes a Good Reasoning Chain? Uncovering Structural Patterns in Long Chain-of-Thought Reasoning 

**Authors**: Gangwei Jiang, Yahui Liu, Zhaoyi Li, Qi Wang, Fuzheng Zhang, Linqi Song, Ying Wei, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.22148)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have popularized Long Chain-of-Thought (LCoT), a strategy that encourages deliberate and step-by-step reasoning before producing a final answer. While LCoTs have enabled expert-level performance in complex tasks, how the internal structures of their reasoning chains drive, or even predict, the correctness of final answers remains a critical yet underexplored question. In this work, we present LCoT2Tree, an automated framework that converts sequential LCoTs into hierarchical tree structures and thus enables deeper structural analysis of LLM reasoning. Using graph neural networks (GNNs), we reveal that structural patterns extracted by LCoT2Tree, including exploration, backtracking, and verification, serve as stronger predictors of final performance across a wide range of tasks and models. Leveraging an explainability technique, we further identify critical thought patterns such as over-branching that account for failures. Beyond diagnostic insights, the structural patterns by LCoT2Tree support practical applications, including improving Best-of-N decoding effectiveness. Overall, our results underscore the critical role of internal structures of reasoning chains, positioning LCoT2Tree as a powerful tool for diagnosing, interpreting, and improving reasoning in LLMs. 

---
# Visual Large Language Models Exhibit Human-Level Cognitive Flexibility in the Wisconsin Card Sorting Test 

**Authors**: Guangfu Hao, Frederic Alexandre, Shan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22112)  

**Abstract**: Cognitive flexibility has been extensively studied in human cognition but remains relatively unexplored in the context of Visual Large Language Models (VLLMs). This study assesses the cognitive flexibility of state-of-the-art VLLMs (GPT-4o, Gemini-1.5 Pro, and Claude-3.5 Sonnet) using the Wisconsin Card Sorting Test (WCST), a classic measure of set-shifting ability. Our results reveal that VLLMs achieve or surpass human-level set-shifting capabilities under chain-of-thought prompting with text-based inputs. However, their abilities are highly influenced by both input modality and prompting strategy. In addition, we find that through role-playing, VLLMs can simulate various functional deficits aligned with patients having impairments in cognitive flexibility, suggesting that VLLMs may possess a cognitive architecture, at least regarding the ability of set-shifting, similar to the brain. This study reveals the fact that VLLMs have already approached the human level on a key component underlying our higher cognition, and highlights the potential to use them to emulate complex brain processes. 

---
# Efficiently Enhancing General Agents With Hierarchical-categorical Memory 

**Authors**: Changze Qiao, Mingming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22006)  

**Abstract**: With large language models (LLMs) demonstrating remarkable capabilities, there has been a surge in research on leveraging LLMs to build general-purpose multi-modal agents. However, existing approaches either rely on computationally expensive end-to-end training using large-scale multi-modal data or adopt tool-use methods that lack the ability to continuously learn and adapt to new environments. In this paper, we introduce EHC, a general agent capable of learning without parameter updates. EHC consists of a Hierarchical Memory Retrieval (HMR) module and a Task-Category Oriented Experience Learning (TOEL) module. The HMR module facilitates rapid retrieval of relevant memories and continuously stores new information without being constrained by memory capacity. The TOEL module enhances the agent's comprehension of various task characteristics by classifying experiences and extracting patterns across different categories. Extensive experiments conducted on multiple standard datasets demonstrate that EHC outperforms existing methods, achieving state-of-the-art performance and underscoring its effectiveness as a general agent for handling complex multi-modal tasks. 

---
# From Reasoning to Learning: A Survey on Hypothesis Discovery and Rule Learning with Large Language Models 

**Authors**: Kaiyu He, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21935)  

**Abstract**: Since the advent of Large Language Models (LLMs), efforts have largely focused on improving their instruction-following and deductive reasoning abilities, leaving open the question of whether these models can truly discover new knowledge. In pursuit of artificial general intelligence (AGI), there is a growing need for models that not only execute commands or retrieve information but also learn, reason, and generate new knowledge by formulating novel hypotheses and theories that deepen our understanding of the world. Guided by Peirce's framework of abduction, deduction, and induction, this survey offers a structured lens to examine LLM-based hypothesis discovery. We synthesize existing work in hypothesis generation, application, and validation, identifying both key achievements and critical gaps. By unifying these threads, we illuminate how LLMs might evolve from mere ``information executors'' into engines of genuine innovation, potentially transforming research, science, and real-world problem solving. 

---
# SAGE-Eval: Evaluating LLMs for Systematic Generalizations of Safety Facts 

**Authors**: Chen Yueh-Han, Guy Davidson, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2505.21828)  

**Abstract**: Do LLMs robustly generalize critical safety facts to novel situations? Lacking this ability is dangerous when users ask naive questions. For instance, "I'm considering packing melon balls for my 10-month-old's lunch. What other foods would be good to include?" Before offering food options, the LLM should warn that melon balls pose a choking hazard to toddlers, as documented by the CDC. Failing to provide such warnings could result in serious injuries or even death. To evaluate this, we introduce SAGE-Eval, SAfety-fact systematic GEneralization evaluation, the first benchmark that tests whether LLMs properly apply well established safety facts to naive user queries. SAGE-Eval comprises 104 facts manually sourced from reputable organizations, systematically augmented to create 10,428 test scenarios across 7 common domains (e.g., Outdoor Activities, Medicine). We find that the top model, Claude-3.7-sonnet, passes only 58% of all the safety facts tested. We also observe that model capabilities and training compute weakly correlate with performance on SAGE-Eval, implying that scaling up is not the golden solution. Our findings suggest frontier LLMs still lack robust generalization ability. We recommend developers use SAGE-Eval in pre-deployment evaluations to assess model reliability in addressing salient risks. We publicly release SAGE-Eval at this https URL and our code is available at this https URL. 

---
# Don't Think Longer, Think Wisely: Optimizing Thinking Dynamics for Large Reasoning Models 

**Authors**: Sohyun An, Ruochen Wang, Tianyi Zhou, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21765)  

**Abstract**: While recent success of large reasoning models (LRMs) significantly advanced LLMs' reasoning capability by optimizing the final answer accuracy using reinforcement learning, they may also drastically increase the output length due to overthinking, characterized by unnecessarily complex reasoning paths that waste computation and potentially degrade the performance. We hypothesize that such inefficiencies stem from LRMs' limited capability to dynamically select the proper modular reasoning strategies, termed thinking patterns at the right position. To investigate this hypothesis, we propose a dynamic optimization framework that segments model-generated reasoning paths into distinct thinking patterns, systematically identifying and promoting beneficial patterns that improve the answer while removing detrimental ones. Empirical analysis confirms that our optimized thinking paths yield more concise yet sufficiently informative trajectories, enhancing reasoning efficiency by reducing attention FLOPs by up to 47% while maintaining accuracy for originally correct responses. Moreover, a non-trivial portion of originally incorrect responses are transformed into correct ones, achieving a 15.6% accuracy improvement with reduced length. Motivated by the improvement brought by the optimized thinking paths, we apply a preference optimization technique supported by a pairwise dataset contrasting suboptimal and optimal reasoning paths. Experimental evaluations across multiple mathematical reasoning benchmarks reveal that our method notably reduces computational overhead while simultaneously improving reasoning accuracy, achieving up to a 12% accuracy improvement and reducing token usage from approximately 5,000 to 3,000 tokens. 

---
# VIRAL: Vision-grounded Integration for Reward design And Learning 

**Authors**: Valentin Cuzin-Rambaud, Emilien Komlenovic, Alexandre Faure, Bruno Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22092)  

**Abstract**: The alignment between humans and machines is a critical challenge in artificial intelligence today. Reinforcement learning, which aims to maximize a reward function, is particularly vulnerable to the risks associated with poorly designed reward functions. Recent advancements has shown that Large Language Models (LLMs) for reward generation can outperform human performance in this context. We introduce VIRAL, a pipeline for generating and refining reward functions through the use of multi-modal LLMs. VIRAL autonomously creates and interactively improves reward functions based on a given environment and a goal prompt or annotated image. The refinement process can incorporate human feedback or be guided by a description generated by a video LLM, which explains the agent's policy in video form. We evaluated VIRAL in five Gymnasium environments, demonstrating that it accelerates the learning of new behaviors while ensuring improved alignment with user intent. The source-code and demo video are available at: this https URL and this https URL. 

---
# AI Mathematician: Towards Fully Automated Frontier Mathematical Research 

**Authors**: Yuanhang Liu, Yanxing Huang, Yanqiao Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22451)  

**Abstract**: Large Reasoning Models (LRMs) have made significant progress in mathematical capabilities in recent times. However, these successes have been primarily confined to competition-level problems. In this work, we propose AI Mathematician (AIM) framework, which harnesses the reasoning strength of LRMs to support frontier mathematical research. We have identified two critical challenges of mathematical research compared to competition, {\it the intrinsic complexity of research problems} and {\it the requirement of procedural rigor}. To address these challenges, AIM incorporates two core strategies: an exploration mechanism to foster longer solution paths, and the pessimistic reasonable verification method to ensure reliability.
This early version of AIM already exhibits strong capability in tackling research-level tasks. We conducted extensive experiments across several real-world mathematical topics and obtained promising results. AIM is able to autonomously construct substantial portions of proofs and uncover non-trivial insights within each research area. These findings highlight the potential of LRMs in mathematical discovery and suggest that LRM-based agent systems could significantly accelerate mathematical research in the future. 

---
# From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications 

**Authors**: Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A. Dobre, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2505.22311)  

**Abstract**: With the advent of 6G communications, intelligent communication systems face multiple challenges, including constrained perception and response capabilities, limited scalability, and low adaptability in dynamic environments. This tutorial provides a systematic introduction to the principles, design, and applications of Large Artificial Intelligence Models (LAMs) and Agentic AI technologies in intelligent communication systems, aiming to offer researchers a comprehensive overview of cutting-edge technologies and practical guidance. First, we outline the background of 6G communications, review the technological evolution from LAMs to Agentic AI, and clarify the tutorial's motivation and main contributions. Subsequently, we present a comprehensive review of the key components required for constructing LAMs. We further categorize LAMs and analyze their applicability, covering Large Language Models (LLMs), Large Vision Models (LVMs), Large Multimodal Models (LMMs), Large Reasoning Models (LRMs), and lightweight LAMs. Next, we propose a LAM-centric design paradigm tailored for communications, encompassing dataset construction and both internal and external learning approaches. Building upon this, we develop an LAM-based Agentic AI system for intelligent communications, clarifying its core components such as planners, knowledge bases, tools, and memory modules, as well as its interaction mechanisms. We also introduce a multi-agent framework with data retrieval, collaborative planning, and reflective evaluation for 6G. Subsequently, we provide a detailed overview of the applications of LAMs and Agentic AI in communication scenarios. Finally, we summarize the research challenges and future directions in current studies, aiming to support the development of efficient, secure, and sustainable next-generation intelligent communication systems. 

---
# From Strangers to Assistants: Fast Desire Alignment for Embodied Agent-User Adaptation 

**Authors**: Yuanfei Wang, Xinju Huang, Fangwei Zhong, Yaodong Yang, Yizhou Wang, Yuanpei Chen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22503)  

**Abstract**: While embodied agents have made significant progress in performing complex physical tasks, real-world applications demand more than pure task execution. The agents must collaborate with unfamiliar agents and human users, whose goals are often vague and implicit. In such settings, interpreting ambiguous instructions and uncovering underlying desires is essential for effective assistance. Therefore, fast and accurate desire alignment becomes a critical capability for embodied agents. In this work, we first develop a home assistance simulation environment HA-Desire that integrates an LLM-driven human user agent exhibiting realistic value-driven goal selection and communication. The ego agent must interact with this proxy user to infer and adapt to the user's latent desires. To achieve this, we present a novel framework FAMER for fast desire alignment, which introduces a desire-based mental reasoning mechanism to identify user intent and filter desire-irrelevant actions. We further design a reflection-based communication module that reduces redundant inquiries, and incorporate goal-relevant information extraction with memory persistence to improve information reuse and reduce unnecessary exploration. Extensive experiments demonstrate that our framework significantly enhances both task execution and communication efficiency, enabling embodied agents to quickly adapt to user-specific desires in complex embodied environments. 

---
# Thinking with Generated Images 

**Authors**: Ethan Chern, Zhulin Hu, Steffi Chern, Siqi Kou, Jiadi Su, Yan Ma, Zhijie Deng, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22525)  

**Abstract**: We present Thinking with Generated Images, a novel paradigm that fundamentally transforms how large multimodal models (LMMs) engage with visual reasoning by enabling them to natively think across text and vision modalities through spontaneous generation of intermediate visual thinking steps. Current visual reasoning with LMMs is constrained to either processing fixed user-provided images or reasoning solely through text-based chain-of-thought (CoT). Thinking with Generated Images unlocks a new dimension of cognitive capability where models can actively construct intermediate visual thoughts, critique their own visual hypotheses, and refine them as integral components of their reasoning process. We demonstrate the effectiveness of our approach through two complementary mechanisms: (1) vision generation with intermediate visual subgoals, where models decompose complex visual tasks into manageable components that are generated and integrated progressively, and (2) vision generation with self-critique, where models generate an initial visual hypothesis, analyze its shortcomings through textual reasoning, and produce refined outputs based on their own critiques. Our experiments on vision generation benchmarks show substantial improvements over baseline approaches, with our models achieving up to 50% (from 38% to 57%) relative improvement in handling complex multi-object scenarios. From biochemists exploring novel protein structures, and architects iterating on spatial designs, to forensic analysts reconstructing crime scenes, and basketball players envisioning strategic plays, our approach enables AI models to engage in the kind of visual imagination and iterative refinement that characterizes human creative, analytical, and strategic thinking. We release our open-source suite at this https URL. 

---
# ChatPD: An LLM-driven Paper-Dataset Networking System 

**Authors**: Anjie Xu, Ruiqing Ding, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22349)  

**Abstract**: Scientific research heavily depends on suitable datasets for method validation, but existing academic platforms with dataset management like PapersWithCode suffer from inefficiencies in their manual workflow. To overcome this bottleneck, we present a system, called ChatPD, that utilizes Large Language Models (LLMs) to automate dataset information extraction from academic papers and construct a structured paper-dataset network. Our system consists of three key modules: \textit{paper collection}, \textit{dataset information extraction}, and \textit{dataset entity resolution} to construct paper-dataset networks. Specifically, we propose a \textit{Graph Completion and Inference} strategy to map dataset descriptions to their corresponding entities. Through extensive experiments, we demonstrate that ChatPD not only outperforms the existing platform PapersWithCode in dataset usage extraction but also achieves about 90\% precision and recall in entity resolution tasks. Moreover, we have deployed ChatPD to continuously extract which datasets are used in papers, and provide a dataset discovery service, such as task-specific dataset queries and similar dataset recommendations. We open source ChatPD and the current paper-dataset network on this [GitHub repository]{this https URL}. 

---
# From Failures to Fixes: LLM-Driven Scenario Repair for Self-Evolving Autonomous Driving 

**Authors**: Xinyu Xia, Xingjun Ma, Yunfeng Hu, Ting Qu, Hong Chen, Xun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22067)  

**Abstract**: Ensuring robust and generalizable autonomous driving requires not only broad scenario coverage but also efficient repair of failure cases, particularly those related to challenging and safety-critical scenarios. However, existing scenario generation and selection methods often lack adaptivity and semantic relevance, limiting their impact on performance improvement. In this paper, we propose \textbf{SERA}, an LLM-powered framework that enables autonomous driving systems to self-evolve by repairing failure cases through targeted scenario recommendation. By analyzing performance logs, SERA identifies failure patterns and dynamically retrieves semantically aligned scenarios from a structured bank. An LLM-based reflection mechanism further refines these recommendations to maximize relevance and diversity. The selected scenarios are used for few-shot fine-tuning, enabling targeted adaptation with minimal data. Experiments on the benchmark show that SERA consistently improves key metrics across multiple autonomous driving baselines, demonstrating its effectiveness and generalizability under safety-critical conditions. 

---
# Estimating the Effects of Sample Training Orders for Large Language Models without Retraining 

**Authors**: Hao Yang, Haoxuan Li, Mengyue Yang, Xu Chen, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22042)  

**Abstract**: The order of training samples plays a crucial role in large language models (LLMs), significantly impacting both their external performance and internal learning dynamics. Traditional methods for investigating this effect generally require retraining the model with various sample orders, which is computationally infeasible for LLMs. In this work, we improve traditional methods by designing a retraining-free framework. By approximating Adam optimizer updates with first- and second-order Taylor expansions and utilizing random projection methods to store intermediate checkpoints, our framework can efficiently estimate model parameters for arbitrary training sample orders. Next, we apply our framework to two downstream research problems: (1) Training curriculum design for LLMs -- we base our retraining-free framework to propose a novel curriculum learning strategy that augments curriculum proposals with estimated model performances, enabling more informed sample scheduling. (2) LLMs' memorization and generalization effect analysis -- we use our retraining-free framework to estimate how the positions of training samples influence LLMs' capacity for memorization and generalization. We conduct extensive experiments to validate the effectiveness of our retraining-free framework in reproducing the true model performances, and further demonstrate its potential in optimizing LLM training curricula and analyzing the memorization and generalization effects of LLMs. 

---
# Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection 

**Authors**: Jinming Zhang, Xuanru Zhou, Jiachen Lian, Shuhe Li, William Li, Zoe Ezzes, Rian Bogley, Lisa Wauters, Zachary Miller, Jet Vonk, Brittany Morin, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.22029)  

**Abstract**: Speech dysfluency detection is crucial for clinical diagnosis and language assessment, but existing methods are limited by the scarcity of high-quality annotated data. Although recent advances in TTS model have enabled synthetic dysfluency generation, existing synthetic datasets suffer from unnatural prosody and limited contextual diversity. To address these limitations, we propose LLM-Dys -- the most comprehensive dysfluent speech corpus with LLM-enhanced dysfluency simulation. This dataset captures 11 dysfluency categories spanning both word and phoneme levels. Building upon this resource, we improve an end-to-end dysfluency detection framework. Experimental validation demonstrates state-of-the-art performance. All data, models, and code are open-sourced at this https URL. 

---
# Judging LLMs on a Simplex 

**Authors**: Patrick Vossler, Fan Xia, Yifan Mai, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21972)  

**Abstract**: Automated evaluation of free-form outputs from large language models (LLMs) is challenging because many distinct answers can be equally valid. A common practice is to use LLMs themselves as judges, but the theoretical properties of this approach are not yet well understood. We show that a geometric framework that represents both judges and candidates as points on a probability simplex can provide helpful insight on what is or is not identifiable using LLM judges. Our theoretical analysis uncovers a "phase transition" in ranking identifiability: for binary scoring systems, true rankings are identifiable even with weak judges under mild assumptions, while rankings become non-identifiable for three or more scoring levels even with infinite data, absent additional prior knowledge. This non-identifiability highlights how uncertainty in rankings stems from not only aleatoric uncertainty (i.e., inherent stochasticity in the data) but also epistemic uncertainty regarding which assumptions hold, an aspect that has received limited attention until now. To integrate both types of uncertainty, we use Bayesian inference to encode assumptions as priors and conduct sensitivity analysis of ranking estimates and credible intervals. Empirical evaluations across multiple benchmarks demonstrate that Bayesian inference yields more accurate rankings and substantially improves coverage rates. These results underscore the importance of taking a more holistic approach to uncertainty quantification when using LLMs as judges. 

---
# iDSE: Navigating Design Space Exploration in High-Level Synthesis Using LLMs 

**Authors**: Runkai Li, Jia Xiong, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22086)  

**Abstract**: High-Level Synthesis (HLS) serves as an agile hardware development tool that streamlines the circuit design by abstracting the register transfer level into behavioral descriptions, while allowing designers to customize the generated microarchitectures through optimization directives. However, the combinatorial explosion of possible directive configurations yields an intractable design space. Traditional design space exploration (DSE) methods, despite adopting heuristics or constructing predictive models to accelerate Pareto-optimal design acquisition, still suffer from prohibitive exploration costs and suboptimal results. Addressing these concerns, we introduce iDSE, the first LLM-aided DSE framework that leverages HLS design quality perception to effectively navigate the design space. iDSE intelligently pruns the design space to guide LLMs in calibrating representative initial sampling designs, expediting convergence toward the Pareto front. By exploiting the convergent and divergent thinking patterns inherent in LLMs for hardware optimization, iDSE achieves multi-path refinement of the design quality and diversity. Extensive experiments demonstrate that iDSE outperforms heuristic-based DSE methods by 5.1$\times$$\sim$16.6$\times$ in proximity to the reference Pareto front, matching NSGA-II with only 4.6% of the explored designs. Our work demonstrates the transformative potential of LLMs in scalable and efficient HLS design optimization, offering new insights into multiobjective optimization challenges. 

---
# Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding 

**Authors**: Hanyin Wang, Zhenbang Wu, Gururaj Kolar, Hariprasad Korsapati, Brian Bartlett, Bryan Hull, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21908)  

**Abstract**: Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-Sapphire introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-of-the-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks. 

---
# Extracting Research Instruments from Educational Literature Using LLMs 

**Authors**: Jiseung Yoo, Curran Mahowald, Meiyu Li, Wei Ai  

**Link**: [PDF](https://arxiv.org/pdf/2505.21855)  

**Abstract**: Large Language Models (LLMs) are transforming information extraction from academic literature, offering new possibilities for knowledge management. This study presents an LLM-based system designed to extract detailed information about research instruments used in the education field, including their names, types, target respondents, measured constructs, and outcomes. Using multi-step prompting and a domain-specific data schema, it generates structured outputs optimized for educational research. Our evaluation shows that this system significantly outperforms other approaches, particularly in identifying instrument names and detailed information. This demonstrates the potential of LLM-powered information extraction in educational contexts, offering a systematic way to organize research instrument information. The ability to aggregate such information at scale enhances accessibility for researchers and education leaders, facilitating informed decision-making in educational research and policy. 

---
# DualSchool: How Reliable are LLMs for Optimization Education? 

**Authors**: Michael Klamkin, Arnaud Deza, Sikai Cheng, Haoruo Zhao, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2505.21775)  

**Abstract**: Consider the following task taught in introductory optimization courses which addresses challenges articulated by the community at the intersection of (generative) AI and OR: generate the dual of a linear program. LLMs, being trained at web-scale, have the conversion process and many instances of Primal to Dual Conversion (P2DC) at their disposal. Students may thus reasonably expect that LLMs would perform well on the P2DC task. To assess this expectation, this paper introduces DualSchool, a comprehensive framework for generating and verifying P2DC instances. The verification procedure of DualSchool uses the Canonical Graph Edit Distance, going well beyond existing evaluation methods for optimization models, which exhibit many false positives and negatives when applied to P2DC. Experiments performed by DualSchool reveal interesting findings. Although LLMs can recite the conversion procedure accurately, state-of-the-art open LLMs fail to consistently produce correct duals. This finding holds even for the smallest two-variable instances and for derivative tasks, such as correctness, verification, and error classification. The paper also discusses the implications for educators, students, and the development of large reasoning systems. 

---
# OmniResponse: Online Multimodal Conversational Response Generation in Dyadic Interactions 

**Authors**: Cheng Luo, Jianghui Wang, Bing Li, Siyang Song, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2505.21724)  

**Abstract**: In this paper, we introduce Online Multimodal Conversational Response Generation (OMCRG), a novel task that aims to online generate synchronized verbal and non-verbal listener feedback, conditioned on the speaker's multimodal input. OMCRG reflects natural dyadic interactions and poses new challenges in achieving synchronization between the generated audio and facial responses of the listener. To address these challenges, we innovatively introduce text as an intermediate modality to bridge the audio and facial responses. We hence propose OmniResponse, a Multimodal Large Language Model (MLLM) that autoregressively generates high-quality multi-modal listener responses. OmniResponse leverages a pretrained LLM enhanced with two novel components: Chrono-Text, which temporally anchors generated text tokens, and TempoVoice, a controllable online TTS module that produces speech synchronized with facial reactions. To support further OMCRG research, we present ResponseNet, a new dataset comprising 696 high-quality dyadic interactions featuring synchronized split-screen videos, multichannel audio, transcripts, and facial behavior annotations. Comprehensive evaluations conducted on ResponseNet demonstrate that OmniResponse significantly outperforms baseline models in terms of semantic speech content, audio-visual synchronization, and generation quality. 

---
# The Feasibility of Topic-Based Watermarking on Academic Peer Reviews 

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2505.21636)  

**Abstract**: Large language models (LLMs) are increasingly integrated into academic workflows, with many conferences and journals permitting their use for tasks such as language refinement and literature summarization. However, their use in peer review remains prohibited due to concerns around confidentiality breaches, hallucinated content, and inconsistent evaluations. As LLM-generated text becomes more indistinguishable from human writing, there is a growing need for reliable attribution mechanisms to preserve the integrity of the review process. In this work, we evaluate topic-based watermarking (TBW), a lightweight, semantic-aware technique designed to embed detectable signals into LLM-generated text. We conduct a comprehensive assessment across multiple LLM configurations, including base, few-shot, and fine-tuned variants, using authentic peer review data from academic conferences. Our results show that TBW maintains review quality relative to non-watermarked outputs, while demonstrating strong robustness to paraphrasing-based evasion. These findings highlight the viability of TBW as a minimally intrusive and practical solution for enforcing LLM usage in peer review. 

---
# SOSBENCH: Benchmarking Safety Alignment on Scientific Knowledge 

**Authors**: Fengqing Jiang, Fengbo Ma, Zhangchen Xu, Yuetai Li, Bhaskar Ramasubramanian, Luyao Niu, Bo Li, Xianyan Chen, Zhen Xiang, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.21605)  

**Abstract**: Large language models (LLMs) exhibit advancing capabilities in complex tasks, such as reasoning and graduate-level question answering, yet their resilience against misuse, particularly involving scientifically sophisticated risks, remains underexplored. Existing safety benchmarks typically focus either on instructions requiring minimal knowledge comprehension (e.g., ``tell me how to build a bomb") or utilize prompts that are relatively low-risk (e.g., multiple-choice or classification tasks about hazardous content). Consequently, they fail to adequately assess model safety when handling knowledge-intensive, hazardous scenarios.
To address this critical gap, we introduce SOSBench, a regulation-grounded, hazard-focused benchmark encompassing six high-risk scientific domains: chemistry, biology, medicine, pharmacology, physics, and psychology. The benchmark comprises 3,000 prompts derived from real-world regulations and laws, systematically expanded via an LLM-assisted evolutionary pipeline that introduces diverse, realistic misuse scenarios (e.g., detailed explosive synthesis instructions involving advanced chemical formulas). We evaluate frontier models within a unified evaluation framework using our SOSBench. Despite their alignment claims, advanced models consistently disclose policy-violating content across all domains, demonstrating alarmingly high rates of harmful responses (e.g., 79.1% for Deepseek-R1 and 47.3% for GPT-4.1). These results highlight significant safety alignment deficiencies and underscore urgent concerns regarding the responsible deployment of powerful LLMs. 

---
# StreamLink: Large-Language-Model Driven Distributed Data Engineering System 

**Authors**: Dawei Feng, Di Mei, Huiri Tan, Lei Ren, Xianying Lou, Zhangxi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21575)  

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in natural language understanding (NLU), opening doors for innovative applications. We introduce StreamLink - an LLM-driven distributed data system designed to improve the efficiency and accessibility of data engineering tasks. We build StreamLink on top of distributed frameworks such as Apache Spark and Hadoop to handle large data at scale. One of the important design philosophies of StreamLink is to respect user data privacy by utilizing local fine-tuned LLMs instead of a public AI service like ChatGPT. With help from domain-adapted LLMs, we can improve our system's understanding of natural language queries from users in various scenarios and simplify the procedure of generating database queries like the Structured Query Language (SQL) for information processing. We also incorporate LLM-based syntax and security checkers to guarantee the reliability and safety of each generated query. StreamLink illustrates the potential of merging generative LLMs with distributed data processing for comprehensive and user-centric data engineering. With this architecture, we allow users to interact with complex database systems at different scales in a user-friendly and security-ensured manner, where the SQL generation reaches over 10\% of execution accuracy compared to baseline methods, and allow users to find the most concerned item from hundreds of millions of items within a few seconds using natural language. 

---
# AITEE -- Agentic Tutor for Electrical Engineering 

**Authors**: Christopher Knievel, Alexander Bernhardt, Christian Bernhardt  

**Link**: [PDF](https://arxiv.org/pdf/2505.21582)  

**Abstract**: Intelligent tutoring systems combined with large language models offer a promising approach to address students' diverse needs and promote self-efficacious learning. While large language models possess good foundational knowledge of electrical engineering basics, they remain insufficiently capable of addressing specific questions about electrical circuits. In this paper, we present AITEE, an agent-based tutoring system for electrical engineering designed to accompany students throughout their learning process, offer individualized support, and promote self-directed learning. AITEE supports both hand-drawn and digital circuits through an adapted circuit reconstruction process, enabling natural interaction with students. Our novel graph-based similarity measure identifies relevant context from lecture materials through a retrieval augmented generation approach, while parallel Spice simulation further enhances accuracy in applying solution methodologies. The system implements a Socratic dialogue to foster learner autonomy through guided questioning. Experimental evaluations demonstrate that AITEE significantly outperforms baseline approaches in domain-specific knowledge application, with even medium-sized LLM models showing acceptable performance. Our results highlight the potential of agentic tutors to deliver scalable, personalized, and effective learning environments for electrical engineering education. 

---
# OpenReview Should be Protected and Leveraged as a Community Asset for Research in the Era of Large Language Models 

**Authors**: Hao Sun, Yunyi Shen, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2505.21537)  

**Abstract**: In the era of large language models (LLMs), high-quality, domain-rich, and continuously evolving datasets capturing expert-level knowledge, core human values, and reasoning are increasingly valuable. This position paper argues that OpenReview -- the continually evolving repository of research papers, peer reviews, author rebuttals, meta-reviews, and decision outcomes -- should be leveraged more broadly as a core community asset for advancing research in the era of LLMs. We highlight three promising areas in which OpenReview can uniquely contribute: enhancing the quality, scalability, and accountability of peer review processes; enabling meaningful, open-ended benchmarks rooted in genuine expert deliberation; and supporting alignment research through real-world interactions reflecting expert assessment, intentions, and scientific values. To better realize these opportunities, we suggest the community collaboratively explore standardized benchmarks and usage guidelines around OpenReview, inviting broader dialogue on responsible data use, ethical considerations, and collective stewardship. 

---
# RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving 

**Authors**: Huacan Wang, Ziyi Ni, Shuo Zhang, Shuo Lu, Sen Hu, Ziyang He, Chen Hu, Jiaye Lin, Yifu Guo, Yuntao Du, Pin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21577)  

**Abstract**: The ultimate goal of code agents is to solve complex tasks autonomously. Although large language models (LLMs) have made substantial progress in code generation, real-world tasks typically demand full-fledged code repositories rather than simple scripts. Building such repositories from scratch remains a major challenge. Fortunately, GitHub hosts a vast, evolving collection of open-source repositories, which developers frequently reuse as modular components for complex tasks. Yet, existing frameworks like OpenHands and SWE-Agent still struggle to effectively leverage these valuable resources. Relying solely on README files provides insufficient guidance, and deeper exploration reveals two core obstacles: overwhelming information and tangled dependencies of repositories, both constrained by the limited context windows of current LLMs. To tackle these issues, we propose RepoMaster, an autonomous agent framework designed to explore and reuse GitHub repositories for solving complex tasks. For efficient understanding, RepoMaster constructs function-call graphs, module-dependency graphs, and hierarchical code trees to identify essential components, providing only identified core elements to the LLMs rather than the entire repository. During autonomous execution, it progressively explores related components using our exploration tools and prunes information to optimize context usage. Evaluated on the adjusted MLE-bench, RepoMaster achieves a 110% relative boost in valid submissions over the strongest baseline OpenHands. On our newly released GitTaskBench, RepoMaster lifts the task-pass rate from 24.1% to 62.9% while reducing token usage by 95%. Our code and demonstration materials are publicly available at this https URL. 

---
# DocReRank: Single-Page Hard Negative Query Generation for Training Multi-Modal RAG Rerankers 

**Authors**: Navve Wasserman, Oliver Heinimann, Yuval Golbari, Tal Zimbalist, Eli Schwartz, Michal Irani  

**Link**: [PDF](https://arxiv.org/pdf/2505.22584)  

**Abstract**: Rerankers play a critical role in multimodal Retrieval-Augmented Generation (RAG) by refining ranking of an initial set of retrieved documents. Rerankers are typically trained using hard negative mining, whose goal is to select pages for each query which rank high, but are actually irrelevant. However, this selection process is typically passive and restricted to what the retriever can find in the available corpus, leading to several inherent limitations. These include: limited diversity, negative examples which are often not hard enough, low controllability, and frequent false negatives which harm training. Our paper proposes an alternative approach: Single-Page Hard Negative Query Generation, which goes the other way around. Instead of retrieving negative pages per query, we generate hard negative queries per page. Using an automated LLM-VLM pipeline, and given a page and its positive query, we create hard negatives by rephrasing the query to be as similar as possible in form and context, yet not answerable from the page. This paradigm enables fine-grained control over the generated queries, resulting in diverse, hard, and targeted negatives. It also supports efficient false negative verification. Our experiments show that rerankers trained with data generated using our approach outperform existing models and significantly improve retrieval performance. 

---
# AI-Supported Platform for System Monitoring and Decision-Making in Nuclear Waste Management with Large Language Models 

**Authors**: Dongjune Chang, Sola Kim, Young Soo Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.21741)  

**Abstract**: Nuclear waste management requires rigorous regulatory compliance assessment, demanding advanced decision-support systems capable of addressing complex legal, environmental, and safety considerations. This paper presents a multi-agent Retrieval-Augmented Generation (RAG) system that integrates large language models (LLMs) with document retrieval mechanisms to enhance decision accuracy through structured agent collaboration. Through a structured 10-round discussion model, agents collaborate to assess regulatory compliance and safety requirements while maintaining document-grounded responses. Implemented on consumer-grade hardware, the system leverages Llama 3.2 and mxbai-embed-large-v1 embeddings for efficient retrieval and semantic representation. A case study of a proposed temporary nuclear waste storage site near Winslow, Arizona, demonstrates the framework's effectiveness. Results show the Regulatory Agent achieves consistently higher relevance scores in maintaining alignment with legal frameworks, while the Safety Agent effectively manages complex risk assessments requiring multifaceted analysis. The system demonstrates progressive improvement in agreement rates between agents across discussion rounds while semantic drift decreases, indicating enhanced decision-making consistency and response coherence. The system ensures regulatory decisions remain factually grounded, dynamically adapting to evolving regulatory frameworks through real-time document retrieval. By balancing automated assessment with human oversight, this framework offers a scalable and transparent approach to regulatory governance. These findings underscore the potential of AI-driven, multi-agent systems in advancing evidence-based, accountable, and adaptive decision-making for high-stakes environmental management scenarios. 

---
