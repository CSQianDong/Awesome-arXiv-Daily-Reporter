# Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models 

**Authors**: Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09138)  

**Abstract**: Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon, temporal oscillation, where correct answers often emerge in the middle process, but are overwritten in later denoising steps. To address this issue, we introduce two complementary methods that exploit temporal consistency: 1) Temporal Self-Consistency Voting, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; and 2) a post-training method termed Temporal Consistency Reinforcement, which uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. Empirical results across multiple benchmarks demonstrate the effectiveness of our approach. Using the negative TSE reward alone, we observe a remarkable average improvement of 24.7% on the Countdown dataset over an existing dLLM. Combined with the accuracy reward, we achieve absolute gains of 2.0% on GSM8K, 4.3% on MATH500, 6.6% on SVAMP, and 25.3% on Countdown, respectively. Our findings underscore the untapped potential of temporal dynamics in dLLMs and offer two simple yet effective tools to harness them. 

---
# Complex Logical Instruction Generation 

**Authors**: Mian Zhang, Shujian Liu, Sixun Dong, Ming Yin, Yebowen Hu, Xun Wang, Steven Ma, Song Wang, Sathish Reddy Indurthi, Haoyun Deng, Zhiyu Zoey Chen, Kaiqiang Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.09125)  

**Abstract**: Instruction following has catalyzed the recent era of Large Language Models (LLMs) and is the foundational skill underpinning more advanced capabilities such as reasoning and agentic behaviors. As tasks grow more challenging, the logic structures embedded in natural language instructions becomes increasingly intricate. However, how well LLMs perform on such logic-rich instructions remains under-explored. We propose LogicIFGen and LogicIFEval. LogicIFGen is a scalable, automated framework for generating verifiable instructions from code functions, which can naturally express rich logic such as conditionals, nesting, recursion, and function calls. We further curate a collection of complex code functions and use LogicIFGen to construct LogicIFEval, a benchmark comprising 426 verifiable logic-rich instructions. Our experiments demonstrate that current state-of-the-art LLMs still struggle to correctly follow the instructions in LogicIFEval. Most LLMs can only follow fewer than 60% of the instructions, revealing significant deficiencies in the instruction-following ability. Code and Benchmark: this https URL 

---
# OdysseyBench: Evaluating LLM Agents on Long-Horizon Complex Office Application Workflows 

**Authors**: Weixuan Wang, Dongge Han, Daniel Madrigal Diaz, Jin Xu, Victor Rühle, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09124)  

**Abstract**: Autonomous agents powered by large language models (LLMs) are increasingly deployed in real-world applications requiring complex, long-horizon workflows. However, existing benchmarks predominantly focus on atomic tasks that are self-contained and independent, failing to capture the long-term contextual dependencies and multi-interaction coordination required in realistic scenarios. To address this gap, we introduce OdysseyBench, a comprehensive benchmark for evaluating LLM agents on long-horizon workflows across diverse office applications including Word, Excel, PDF, Email, and Calendar. Our benchmark comprises two complementary splits: OdysseyBench+ with 300 tasks derived from real-world use cases, and OdysseyBench-Neo with 302 newly synthesized complex tasks. Each task requires agent to identify essential information from long-horizon interaction histories and perform multi-step reasoning across various applications. To enable scalable benchmark creation, we propose HomerAgents, a multi-agent framework that automates the generation of long-horizon workflow benchmarks through systematic environment exploration, task generation, and dialogue synthesis. Our extensive evaluation demonstrates that OdysseyBench effectively challenges state-of-the-art LLM agents, providing more accurate assessment of their capabilities in complex, real-world contexts compared to existing atomic task benchmarks. We believe that OdysseyBench will serve as a valuable resource for advancing the development and evaluation of LLM agents in real-world productivity scenarios. In addition, we release OdysseyBench and HomerAgents to foster research along this line. 

---
# SinLlama - A Large Language Model for Sinhala 

**Authors**: H.W.K.Aravinda, Rashad Sirajudeen, Samith Karunathilake, Nisansa de Silva, Surangika Ranathunga, Rishemjit Kaur  

**Link**: [PDF](https://arxiv.org/pdf/2508.09115)  

**Abstract**: Low-resource languages such as Sinhala are often overlooked by open-source Large Language Models (LLMs). In this research, we extend an existing multilingual LLM (Llama-3-8B) to better serve Sinhala. We enhance the LLM tokenizer with Sinhala specific vocabulary and perform continual pre-training on a cleaned 10 million Sinhala corpus, resulting in the SinLlama model. This is the very first decoder-based open-source LLM with explicit Sinhala support. When SinLlama was instruction fine-tuned for three text classification tasks, it outperformed base and instruct variants of Llama-3-8B by a significant margin. 

---
# AutoCodeBench: Large Language Models are Automatic Code Benchmark Generators 

**Authors**: Jason Chou, Ao Liu, Yuchi Deng, Zhiying Zeng, Tao Zhang, Haotian Zhu, Jianwei Cai, Yue Mao, Chenchen Zhang, Lingyun Tan, Ziyan Xu, Bohui Zhai, Hengyi Liu, Speed Zhu, Wiggin Zhou, Fengzong Lian  

**Link**: [PDF](https://arxiv.org/pdf/2508.09101)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains, with code generation emerging as a key area of focus. While numerous benchmarks have been proposed to evaluate their code generation abilities, these benchmarks face several critical limitations. First, they often rely on manual annotations, which are time-consuming and difficult to scale across different programming languages and problem complexities. Second, most existing benchmarks focus primarily on Python, while the few multilingual benchmarks suffer from limited difficulty and uneven language distribution. To address these challenges, we propose AutoCodeGen, an automated method for generating high-difficulty multilingual code generation datasets without manual annotations. AutoCodeGen ensures the correctness and completeness of test cases by generating test inputs with LLMs and obtaining test outputs through a multilingual sandbox, while achieving high data quality through reverse-order problem generation and multiple filtering steps. Using this novel method, we introduce AutoCodeBench, a large-scale code generation benchmark comprising 3,920 problems evenly distributed across 20 programming languages. It is specifically designed to evaluate LLMs on challenging, diverse, and practical multilingual tasks. We evaluate over 30 leading open-source and proprietary LLMs on AutoCodeBench and its simplified version AutoCodeBench-Lite. The results show that even the most advanced LLMs struggle with the complexity, diversity, and multilingual nature of these tasks. Besides, we introduce AutoCodeBench-Complete, specifically designed for base models to assess their few-shot code generation capabilities. We hope the AutoCodeBench series will serve as a valuable resource and inspire the community to focus on more challenging and practical multilingual code generation scenarios. 

---
# Link Prediction for Event Logs in the Process Industry 

**Authors**: Anastasia Zhukova, Thomas Walton, Christian E. Matt, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2508.09096)  

**Abstract**: Knowledge management (KM) is vital in the process industry for optimizing operations, ensuring safety, and enabling continuous improvement through effective use of operational data and past insights. A key challenge in this domain is the fragmented nature of event logs in shift books, where related records, e.g., entries documenting issues related to equipment or processes and the corresponding solutions, may remain disconnected. This fragmentation hinders the recommendation of previous solutions to the users. To address this problem, we investigate record linking (RL) as link prediction, commonly studied in graph-based machine learning, by framing it as a cross-document coreference resolution (CDCR) task enhanced with natural language inference (NLI) and semantic text similarity (STS) by shifting it into the causal inference (CI). We adapt CDCR, traditionally applied in the news domain, into an RL model to operate at the passage level, similar to NLI and STS, while accommodating the process industry's specific text formats, which contain unstructured text and structured record attributes. Our RL model outperformed the best versions of NLI- and STS-driven baselines by 28% (11.43 points) and 27% (11.21 points), respectively. Our work demonstrates how domain adaptation of the state-of-the-art CDCR models, enhanced with reasoning capabilities, can be effectively tailored to the process industry, improving data quality and connectivity in shift logs. 

---
# Utilizing Multilingual Encoders to Improve Large Language Models for Low-Resource Languages 

**Authors**: Imalsha Puranegedara, Themira Chathumina, Nisal Ranathunga, Nisansa de Silva, Surangika Ranathunga, Mokanarangan Thayaparan  

**Link**: [PDF](https://arxiv.org/pdf/2508.09091)  

**Abstract**: Large Language Models (LLMs) excel in English, but their performance degrades significantly on low-resource languages (LRLs) due to English-centric training. While methods like LangBridge align LLMs with multilingual encoders such as the Massively Multilingual Text-to-Text Transfer Transformer (mT5), they typically use only the final encoder layer. We propose a novel architecture that fuses all intermediate layers, enriching the linguistic information passed to the LLM. Our approach features two strategies: (1) a Global Softmax weighting for overall layer importance, and (2) a Transformer Softmax model that learns token-specific weights. The fused representations are mapped into the LLM's embedding space, enabling it to process multilingual inputs. The model is trained only on English data, without using any parallel or multilingual data. Evaluated on XNLI, IndicXNLI, Sinhala News Classification, and Amazon Reviews, our Transformer Softmax model significantly outperforms the LangBridge baseline. We observe strong performance gains in LRLs, improving Sinhala classification accuracy from 71.66% to 75.86% and achieving clear improvements across Indic languages such as Tamil, Bengali, and Malayalam. These specific gains contribute to an overall boost in average XNLI accuracy from 70.36% to 71.50%. This approach offers a scalable, data-efficient path toward more capable and equitable multilingual LLMs. 

---
# CPO: Addressing Reward Ambiguity in Role-playing Dialogue via Comparative Policy Optimization 

**Authors**: Xinge Ye, Rui Wang, Yuchuan Wu, Victor Ma, Feiteng Fang, Fei Huang, Yongbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.09074)  

**Abstract**: Reinforcement Learning Fine-Tuning (RLFT) has achieved notable success in tasks with objectively verifiable answers (e.g., code generation, mathematical reasoning), yet struggles with open-ended subjective tasks like role-playing dialogue. Traditional reward modeling approaches, which rely on independent sample-wise scoring, face dual challenges: subjective evaluation criteria and unstable reward this http URL by the insight that human evaluation inherently combines explicit criteria with implicit comparative judgments, we propose Comparative Policy Optimization (CPO). CPO redefines the reward evaluation paradigm by shifting from sample-wise scoring to comparative group-wise this http URL on the same principle, we introduce the CharacterArena evaluation framework, which comprises two stages:(1) Contextualized Multi-turn Role-playing Simulation, and (2) Trajectory-level Comparative Evaluation. By operationalizing subjective scoring via objective trajectory comparisons, CharacterArena minimizes contextual bias and enables more robust and fair performance evaluation. Empirical results on CharacterEval, CharacterBench, and CharacterArena confirm that CPO effectively mitigates reward ambiguity and leads to substantial improvements in dialogue quality. 

---
# READER: Retrieval-Assisted Drafter for Efficient LLM Inference 

**Authors**: Maxim Divilkovskiy, Vitaly Malygin, Sergey Zlobin, Sultan Isali, Vasily Kalugin, Stanislav Ilyushin, Nuriza Aitassova, Yi Fei, Zeng Weidi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09072)  

**Abstract**: Large Language Models (LLMs) generate tokens autoregressively, with each token depending on the preceding context. This sequential nature makes the inference process inherently difficult to accelerate, posing a significant challenge for efficient deployment. In recent years, various methods have been proposed to address this issue, with the most effective approaches often involving the training of additional draft models. In this paper, we introduce READER (Retrieval-Assisted Drafter for Efficient LLM Inference), a novel lossless speculative decoding method that enhances model-based approaches by leveraging self-repetitions in the text. Our algorithm expands the speculative decoding tree using tokens obtained through statistical search. This work focuses on large batch sizes (>= 8), an underexplored yet important area for industrial applications. We also analyze the key-value (KV) cache size during speculative decoding and propose an optimization to improve performance for large batches. As a result, READER outperforms existing speculative decoding methods. Notably, READER requires no additional training and can reuse pre-trained speculator models, increasing the speedup by over 40\%. Our method demonstrates particularly strong performance on search-based tasks, such as retrieval-augmented generation, where we achieve more than 10x speedup. 

---
# MVISU-Bench: Benchmarking Mobile Agents for Real-World Tasks by Multi-App, Vague, Interactive, Single-App and Unethical Instructions 

**Authors**: Zeyu Huang, Juyuan Wang, Longfeng Chen, Boyi Xiao, Leng Cai, Yawen Zeng, Jin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09057)  

**Abstract**: Given the significant advances in Large Vision Language Models (LVLMs) in reasoning and visual understanding, mobile agents are rapidly emerging to meet users' automation needs. However, existing evaluation benchmarks are disconnected from the real world and fail to adequately address the diverse and complex requirements of users. From our extensive collection of user questionnaire, we identified five tasks: Multi-App, Vague, Interactive, Single-App, and Unethical Instructions. Around these tasks, we present \textbf{MVISU-Bench}, a bilingual benchmark that includes 404 tasks across 137 mobile applications. Furthermore, we propose Aider, a plug-and-play module that acts as a dynamic prompt prompter to mitigate risks and clarify user intent for mobile agents. Our Aider is easy to integrate into several frameworks and has successfully improved overall success rates by 19.55\% compared to the current state-of-the-art (SOTA) on MVISU-Bench. Specifically, it achieves success rate improvements of 53.52\% and 29.41\% for unethical and interactive instructions, respectively. Through extensive experiments and analysis, we highlight the gap between existing mobile agents and real-world user expectations. 

---
# LLM-as-a-Supervisor: Mistaken Therapeutic Behaviors Trigger Targeted Supervisory Feedback 

**Authors**: Chen Xu, Zhenyu Lv, Tian Lan, Xianyang Wang, Luyao Ji, Leyang Cui, Minqiang Yang, Jian Shen, Qunxi Dong, Xiuling Liu, Juan Wang, Bin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09042)  

**Abstract**: Although large language models (LLMs) hold significant promise in psychotherapy, their direct application in patient-facing scenarios raises ethical and safety concerns. Therefore, this work shifts towards developing an LLM as a supervisor to train real therapists. In addition to the privacy of clinical therapist training data, a fundamental contradiction complicates the training of therapeutic behaviors: clear feedback standards are necessary to ensure a controlled training system, yet there is no absolute "gold standard" for appropriate therapeutic behaviors in practice. In contrast, many common therapeutic mistakes are universal and identifiable, making them effective triggers for targeted feedback that can serve as clearer evidence. Motivated by this, we create a novel therapist-training paradigm: (1) guidelines for mistaken behaviors and targeted correction strategies are first established as standards; (2) a human-in-the-loop dialogue-feedback dataset is then constructed, where a mistake-prone agent intentionally makes standard mistakes during interviews naturally, and a supervisor agent locates and identifies mistakes and provides targeted feedback; (3) after fine-tuning on this dataset, the final supervisor model is provided for real therapist training. The detailed experimental results of automated, human and downstream assessments demonstrate that models fine-tuned on our dataset MATE, can provide high-quality feedback according to the clinical guideline, showing significant potential for the therapist training scenario. 

---
# A Survey on Training-free Alignment of Large Language Models 

**Authors**: Birong Pan, Yongqi Li, Weiyu Zhang, Wenpeng Lu, Mayi Xu, Shen Zhou, Yuanyuan Zhu, Ming Zhong, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.09016)  

**Abstract**: The alignment of large language models (LLMs) aims to ensure their outputs adhere to human values, ethical standards, and legal norms. Traditional alignment methods often rely on resource-intensive fine-tuning (FT), which may suffer from knowledge degradation and face challenges in scenarios where the model accessibility or computational resources are constrained. In contrast, training-free (TF) alignment techniques--leveraging in-context learning, decoding-time adjustments, and post-generation corrections--offer a promising alternative by enabling alignment without heavily retraining LLMs, making them adaptable to both open-source and closed-source environments. This paper presents the first systematic review of TF alignment methods, categorizing them by stages of pre-decoding, in-decoding, and post-decoding. For each stage, we provide a detailed examination from the viewpoint of LLMs and multimodal LLMs (MLLMs), highlighting their mechanisms and limitations. Furthermore, we identify key challenges and future directions, paving the way for more inclusive and effective TF alignment techniques. By synthesizing and organizing the rapidly growing body of research, this survey offers a guidance for practitioners and advances the development of safer and more reliable LLMs. 

---
# LyS at SemEval 2025 Task 8: Zero-Shot Code Generation for Tabular QA 

**Authors**: Adrián Gude, Roi Santos-Ríos, Francisco Prado-Valiño, Ana Ezquerro, Jesús Vilares  

**Link**: [PDF](https://arxiv.org/pdf/2508.09012)  

**Abstract**: This paper describes our participation in SemEval 2025 Task 8, focused on Tabular Question Answering. We developed a zero-shot pipeline that leverages an Large Language Model to generate functional code capable of extracting the relevant information from tabular data based on an input question. Our approach consists of a modular pipeline where the main code generator module is supported by additional components that identify the most relevant columns and analyze their data types to improve extraction accuracy. In the event that the generated code fails, an iterative refinement process is triggered, incorporating the error feedback into a new generation prompt to enhance robustness. Our results show that zero-shot code generation is a valid approach for Tabular QA, achieving rank 33 of 53 in the test phase despite the lack of task-specific fine-tuning. 

---
# Retrospective Sparse Attention for Efficient Long-Context Generation 

**Authors**: Seonghwan Choi, Beomseok Kang, Dongwon Jo, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09001)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in long-context tasks such as reasoning, code generation, and multi-turn dialogue. However, inference over extended contexts is bottlenecked by the Key-Value (KV) cache, whose memory footprint grows linearly with sequence length and dominates latency at each decoding step. While recent KV cache compression methods identify and load important tokens, they focus predominantly on input contexts and fail to address the cumulative attention errors that arise during long decoding. In this paper, we introduce RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using newly arrived KV entries from subsequent decoding steps. By maintaining a lightweight output cache, RetroAttention enables past queries to efficiently access more relevant context, while incurring minimal latency overhead. This breaks the fixed-attention-output paradigm and allows continual correction of prior approximations. Extensive experiments on long-generation benchmarks show that RetroAttention consistently outperforms state-of-the-art (SOTA) KV compression methods, increasing effective KV exposure by up to 1.6$\times$ and accuracy by up to 21.9\%. 

---
# Jointly Generating and Attributing Answers using Logits of Document-Identifier Tokens 

**Authors**: Lucas Albarede, Jose Moreno, Lynda Tamine, Luce Lefeuvre  

**Link**: [PDF](https://arxiv.org/pdf/2508.08942)  

**Abstract**: Despite their impressive performances, Large Language Models (LLMs) remain prone to hallucination, which critically undermines their trustworthiness. While most of the previous work focused on tackling answer and attribution correctness, a recent line of work investigated faithfulness, with a focus on leveraging internal model signals to reflect a model's actual decision-making process while generating the answer. Nevertheless, these methods induce additional latency and have shown limitations in directly aligning token generation with attribution generation. In this paper, we introduce LoDIT, a method that jointly generates and faithfully attributes answers in RAG by leveraging specific token logits during generation. It consists of two steps: (1) marking the documents with specific token identifiers and then leveraging the logits of these tokens to estimate the contribution of each document to the answer during generation, and (2) aggregating these contributions into document attributions. Experiments on a trustworthiness-focused attributed text-generation benchmark, Trust-Align, show that LoDIT significantly outperforms state-of-the-art models on several metrics. Finally, an in-depth analysis of LoDIT shows both its efficiency in terms of latency and its robustness in different settings. 

---
# Train Long, Think Short: Curriculum Learning for Efficient Reasoning 

**Authors**: Hasan Abed Al Kader Hammoud, Kumail Alhamoud, Abed Hammoud, Elie Bou-Zeid, Marzyeh Ghassemi, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08940)  

**Abstract**: Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: this https URL. 

---
# Reveal-Bangla: A Dataset for Cross-Lingual Multi-Step Reasoning Evaluation 

**Authors**: Khondoker Ittehadul Islam, Gabriele Sarti  

**Link**: [PDF](https://arxiv.org/pdf/2508.08933)  

**Abstract**: Language models have demonstrated remarkable performance on complex multi-step reasoning tasks. However, their evaluation has been predominantly confined to high-resource languages such as English. In this paper, we introduce a manually translated Bangla multi-step reasoning dataset derived from the English Reveal dataset, featuring both binary and non-binary question types. We conduct a controlled evaluation of English-centric and Bangla-centric multilingual small language models on the original dataset and our translated version to compare their ability to exploit relevant reasoning steps to produce correct answers. Our results show that, in comparable settings, reasoning context is beneficial for more challenging non-binary questions, but models struggle to employ relevant Bangla reasoning steps effectively. We conclude by exploring how reasoning steps contribute to models' predictions, highlighting different trends across models and languages. 

---
# Munsit at NADI 2025 Shared Task 2: Pushing the Boundaries of Multidialectal Arabic ASR with Weakly Supervised Pretraining and Continual Supervised Fine-tuning 

**Authors**: Mahmoud Salhab, Shameed Sait, Mohammad Abusheikh, Hasan Abusheikh  

**Link**: [PDF](https://arxiv.org/pdf/2508.08912)  

**Abstract**: Automatic speech recognition (ASR) plays a vital role in enabling natural human-machine interaction across applications such as virtual assistants, industrial automation, customer support, and real-time transcription. However, developing accurate ASR systems for low-resource languages like Arabic remains a significant challenge due to limited labeled data and the linguistic complexity introduced by diverse dialects. In this work, we present a scalable training pipeline that combines weakly supervised learning with supervised fine-tuning to develop a robust Arabic ASR model. In the first stage, we pretrain the model on 15,000 hours of weakly labeled speech covering both Modern Standard Arabic (MSA) and various Dialectal Arabic (DA) variants. In the subsequent stage, we perform continual supervised fine-tuning using a mixture of filtered weakly labeled data and a small, high-quality annotated dataset. Our approach achieves state-of-the-art results, ranking first in the multi-dialectal Arabic ASR challenge. These findings highlight the effectiveness of weak supervision paired with fine-tuning in overcoming data scarcity and delivering high-quality ASR for low-resource, dialect-rich languages. 

---
# ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs 

**Authors**: Keyu Chen, Zhifeng Shen, Daohai Yu, Haoqian Wu, Wei Wen, Jianfeng He, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.08895)  

**Abstract**: The increasing scale and complexity of large language models (LLMs) pose significant inference latency challenges, primarily due to their autoregressive decoding paradigm characterized by the sequential nature of next-token prediction. By re-examining the outputs of autoregressive models, we observed that some segments exhibit parallelizable structures, which we term intrinsic parallelism. Decoding each parallelizable branch simultaneously (i.e. parallel decoding) can significantly improve the overall inference speed of LLMs. In this paper, we propose an Adaptive Serial-Parallel Decoding (ASPD), which addresses two core challenges: automated construction of parallelizable data and efficient parallel decoding mechanism. More specifically, we introduce a non-invasive pipeline that automatically extracts and validates parallelizable structures from the responses of autoregressive models. To empower efficient adaptive serial-parallel decoding, we implement a Hybrid Decoding Engine which enables seamless transitions between serial and parallel decoding modes while maintaining a reusable KV cache, maximizing computational efficiency. Extensive evaluations across General Tasks, Retrieval-Augmented Generation, Mathematical Reasoning, demonstrate that ASPD achieves unprecedented performance in both effectiveness and efficiency. Notably, on Vicuna Bench, our method achieves up to 3.19x speedup (1.85x on average) while maintaining response quality within 1% difference compared to autoregressive models, realizing significant acceleration without compromising generation quality. Our framework sets a groundbreaking benchmark for efficient LLM parallel inference, paving the way for its deployment in latency-sensitive applications such as AI-powered customer service bots and answer retrieval engines. 

---
# Entangled in Representations: Mechanistic Investigation of Cultural Biases in Large Language Models 

**Authors**: Haeun Yu, Seogyeong Jeong, Siddhesh Pawar, Jisu Shin, Jiho Jin, Junho Myung, Alice Oh, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.08879)  

**Abstract**: The growing deployment of large language models (LLMs) across diverse cultural contexts necessitates a better understanding of how the overgeneralization of less documented cultures within LLMs' representations impacts their cultural understanding. Prior work only performs extrinsic evaluation of LLMs' cultural competence, without accounting for how LLMs' internal mechanisms lead to cultural (mis)representation. To bridge this gap, we propose Culturescope, the first mechanistic interpretability-based method that probes the internal representations of LLMs to elicit the underlying cultural knowledge space. CultureScope utilizes a patching method to extract the cultural knowledge. We introduce a cultural flattening score as a measure of the intrinsic cultural biases. Additionally, we study how LLMs internalize Western-dominance bias and cultural flattening, which allows us to trace how cultural biases emerge within LLMs. Our experimental results reveal that LLMs encode Western-dominance bias and cultural flattening in their cultural knowledge space. We find that low-resource cultures are less susceptible to cultural biases, likely due to their limited training resources. Our work provides a foundation for future research on mitigating cultural biases and enhancing LLMs' cultural understanding. Our codes and data used for experiments are publicly available. 

---
# Weakly Supervised Fine-grained Span-Level Framework for Chinese Radiology Report Quality Assurance 

**Authors**: Kaiyu Wang, Lin Mu, Zhiyao Yang, Ximing Li, Xiaotang Zhou Wanfu Gao, Huimao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08876)  

**Abstract**: Quality Assurance (QA) for radiology reports refers to judging whether the junior reports (written by junior doctors) are qualified. The QA scores of one junior report are given by the senior doctor(s) after reviewing the image and junior report. This process requires intensive labor costs for senior doctors. Additionally, the QA scores may be inaccurate for reasons like diagnosis bias, the ability of senior doctors, and so on. To address this issue, we propose a Span-level Quality Assurance EvaluaTOR (Sqator) to mark QA scores automatically. Unlike the common document-level semantic comparison method, we try to analyze the semantic difference by exploring more fine-grained text spans. Unlike the common document-level semantic comparison method, we try to analyze the semantic difference by exploring more fine-grained text spans. Specifically, Sqator measures QA scores by measuring the importance of revised spans between junior and senior reports, and outputs the final QA scores by merging all revised span scores. We evaluate Sqator using a collection of 12,013 radiology reports. Experimental results show that Sqator can achieve competitive QA scores. Moreover, the importance scores of revised spans can be also consistent with the judgments of senior doctors. 

---
# BiasGym: Fantastic Biases and How to Find (and Remove) Them 

**Authors**: Sekh Mainul Islam, Nadav Borenstein, Siddhesh Milind Pawar, Haeun Yu, Arnav Arora, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.08855)  

**Abstract**: Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. Biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce BiasGym, a simple, cost-effective, and generalizable framework for reliably injecting, analyzing, and mitigating conceptual associations within LLMs. BiasGym consists of two components: BiasInject, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and BiasScope, which leverages these injected signals to identify and steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during training. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from a country being `reckless drivers') and in probing fictional associations (e.g., people from a country having `blue skin'), showing its utility for both safety interventions and interpretability research. 

---
# Steering Towards Fairness: Mitigating Political Bias in LLMs 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08846)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases, particularly along political and economic dimensions. In this paper, we propose a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), our method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions. 

---
# An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems 

**Authors**: Yuren Hao, Xiang Wan, Chengxiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08833)  

**Abstract**: In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 49 % on the originals but drops by 4 percentage points on surface variants, and by 10.5 percentage points on core-step-based variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities. 

---
# TiMoE: Time-Aware Mixture of Language Experts 

**Authors**: Robin Faro, Dongyang Fan, Tamar Alphaidze, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08827)  

**Abstract**: Large language models (LLMs) are typically trained on fixed snapshots of the web, which means that their knowledge becomes stale and their predictions risk temporal leakage: relying on information that lies in the future relative to a query. We tackle this problem by pre-training from scratch a set of GPT-style experts on disjoint two-year slices of a 2013-2024 corpus and combining them through TiMoE, a Time-aware Mixture of Language Experts. At inference time, TiMoE masks all experts whose training window ends after the query timestamp and merges the remaining log-probabilities in a shared space, guaranteeing strict causal validity while retaining the breadth of multi-period knowledge. We also release TSQA, a 10k-question benchmark whose alternatives are explicitly labelled as past, future or irrelevant, allowing fine-grained measurement of temporal hallucinations. Experiments on eight standard NLP tasks plus TSQA show that a co-adapted TiMoE variant matches or exceeds the best single-period expert and cuts future-knowledge errors by up to 15%. Our results demonstrate that modular, time-segmented pre-training paired with causal routing is a simple yet effective path toward LLMs that stay chronologically grounded without sacrificing general performance much. We open source our code at TiMoE (Github): this https URL 

---
# Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments 

**Authors**: Junjie Ye, Changhao Jiang, Zhengyin Du, Yufei Xu, Xuesong Yao, Zhiheng Xi, Xiaoran Fan, Qi Zhang, Xuanjing Huang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08791)  

**Abstract**: Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models' tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models. 

---
# Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering 

**Authors**: Yunfeng Ning, Mayi Xu, Jintao Wen, Qiankun Pi, Yuanyuan Zhu, Ming Zhong, Jiawei Jiang, Tieyun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.08785)  

**Abstract**: LLMs often suffer from hallucinations and outdated or incomplete knowledge. RAG is proposed to address these issues by integrating external knowledge like that in KGs into LLMs. However, leveraging private KGs in RAG systems poses significant privacy risks due to the black-box nature of LLMs and potential insecure data transmission, especially when using third-party LLM APIs lacking transparency and control. In this paper, we investigate the privacy-protected RAG scenario for the first time, where entities in KGs are anonymous for LLMs, thus preventing them from accessing entity semantics. Due to the loss of semantics of entities, previous RAG systems cannot retrieve question-relevant knowledge from KGs by matching questions with the meaningless identifiers of anonymous entities. To realize an effective RAG system in this scenario, two key challenges must be addressed: (1) How can anonymous entities be converted into retrievable information. (2) How to retrieve question-relevant anonymous entities. Hence, we propose a novel ARoG framework including relation-centric abstraction and structure-oriented abstraction strategies. For challenge (1), the first strategy abstracts entities into high-level concepts by dynamically capturing the semantics of their adjacent relations. It supplements meaningful semantics which can further support the retrieval process. For challenge (2), the second strategy transforms unstructured natural language questions into structured abstract concept paths. These paths can be more effectively aligned with the abstracted concepts in KGs, thereby improving retrieval performance. To guide LLMs to effectively retrieve knowledge from KGs, the two strategies strictly protect privacy from being exposed to LLMs. Experiments on three datasets demonstrate that ARoG achieves strong performance and privacy-robustness. 

---
# DevNous: An LLM-Based Multi-Agent System for Grounding IT Project Management in Unstructured Conversation 

**Authors**: Stavros Doropoulos, Stavros Vologiannidis, Ioannis Magnisalis  

**Link**: [PDF](https://arxiv.org/pdf/2508.08761)  

**Abstract**: The manual translation of unstructured team dialogue into the structured artifacts required for Information Technology (IT) project governance is a critical bottleneck in modern information systems management. We introduce DevNous, a Large Language Model-based (LLM) multi-agent expert system, to automate this unstructured-to-structured translation process. DevNous integrates directly into team chat environments, identifying actionable intents from informal dialogue and managing stateful, multi-turn workflows for core administrative tasks like automated task formalization and progress summary synthesis. To quantitatively evaluate the system, we introduce a new benchmark of 160 realistic, interactive conversational turns. The dataset was manually annotated with a multi-label ground truth and is publicly available. On this benchmark, DevNous achieves an exact match turn accuracy of 81.3\% and a multiset F1-Score of 0.845, providing strong evidence for its viability. The primary contributions of this work are twofold: (1) a validated architectural pattern for developing ambient administrative agents, and (2) the introduction of the first robust empirical baseline and public benchmark dataset for this challenging problem domain. 

---
# SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs 

**Authors**: Haotian Chen, Qingqing Long, Meng Xiao, Xiao Luo, Wei Ju, Chengrui Wang, Xuezhi Wang, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08742)  

**Abstract**: Scientific literature question answering is a pivotal step towards new scientific discoveries. Recently, \textit{two-stage} retrieval-augmented generated large language models (RAG-LLMs) have shown impressive advancements in this domain. Such a two-stage framework, especially the second stage (reranker), is particularly essential in the scientific domain, where subtle differences in terminology may have a greatly negative impact on the final factual-oriented or knowledge-intensive answers. Despite this significant progress, the potential and limitations of these works remain unexplored. In this work, we present a Scientific Rerank-oriented RAG Benchmark (SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning five scientific subjects. To rigorously assess the reranker performance in terms of noise resilience, relevance disambiguation, and factual consistency, we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI), and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely used rerankers on five families of LLMs, we provide detailed insights into their relative strengths and limitations. To the best of our knowledge, SciRerankBench is the first benchmark specifically developed to evaluate rerankers within RAG-LLMs, which provides valuable observations and guidance for their future development. 

---
# Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation 

**Authors**: Weibin Liao, Tianlong Wang, Yinghao Zhu, Yasha Wang, Junyi Gao, Liantao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08730)  

**Abstract**: Medical Lay Language Generation (MLLG) plays a vital role in improving the accessibility of complex scientific content for broader audiences. Recent literature to MLLG commonly employ parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) to fine-tuning large language models (LLMs) using paired expert-lay language datasets. However, LoRA struggles with the challenges posed by multi-source heterogeneous MLLG datasets. Specifically, through a series of exploratory experiments, we reveal that standard LoRA fail to meet the requirement for semantic fidelity and diverse lay-style generation in MLLG task. To address these limitations, we propose Magical, an asymmetric LoRA architecture tailored for MLLG under heterogeneous data scenarios. Magical employs a shared matrix $A$ for abstractive summarization, along with multiple isolated matrices $B$ for diverse lay-style generation. To preserve semantic fidelity during the lay language generation process, Magical introduces a Semantic Invariance Constraint to mitigate semantic subspace shifts on matrix $A$. Furthermore, to better adapt to diverse lay-style generation, Magical incorporates the Recommendation-guided Switch, an externally interface to prompt the LLM to switch between different matrices $B$. Experimental results on three real-world lay language generation datasets demonstrate that Magical consistently outperforms prompt-based methods, vanilla LoRA, and its recent variants, while also reducing trainable parameters by 31.66%. 

---
# IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization 

**Authors**: Yuzhuo Bai, Shitong Duan, Muhua Huang, Jing Yao, Zhenghao Liu, Peng Zhang, Tun Lu, Xiaoyuan Yi, Maosong Sun, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.08719)  

**Abstract**: Trained on various human-authored corpora, Large Language Models (LLMs) have demonstrated a certain capability of reflecting specific human-like traits (e.g., personality or values) by prompting, benefiting applications like personalized LLMs and social simulations. However, existing methods suffer from the superficial elicitation problem: LLMs can only be steered to mimic shallow and unstable stylistic patterns, failing to embody the desired traits precisely and consistently across diverse tasks like humans. To address this challenge, we propose IROTE, a novel in-context method for stable and transferable trait elicitation. Drawing on psychological theories suggesting that traits are formed through identity-related reflection, our method automatically generates and optimizes a textual self-reflection within prompts, which comprises self-perceived experience, to stimulate LLMs' trait-driven behavior. The optimization is performed by iteratively maximizing an information-theoretic objective that enhances the connections between LLMs' behavior and the target trait, while reducing noisy redundancy in reflection without any fine-tuning, leading to evocative and compact trait reflection. Extensive experiments across three human trait systems manifest that one single IROTE-generated self-reflection can induce LLMs' stable impersonation of the target trait across diverse downstream tasks beyond simple questionnaire answering, consistently outperforming existing strong baselines. 

---
# A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models 

**Authors**: Lingzhe Zhang, Liancheng Fang, Chiming Duan, Minghua He, Leyi Pan, Pei Xiao, Shiyu Huang, Yunpeng Zhai, Xuming Hu, Philip S. Yu, Aiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08712)  

**Abstract**: As text generation has become a core capability of modern Large Language Models (LLMs), it underpins a wide range of downstream applications. However, most existing LLMs rely on autoregressive (AR) generation, producing one token at a time based on previously generated context-resulting in limited generation speed due to the inherently sequential nature of the process. To address this challenge, an increasing number of researchers have begun exploring parallel text generation-a broad class of techniques aimed at breaking the token-by-token generation bottleneck and improving inference efficiency. Despite growing interest, there remains a lack of comprehensive analysis on what specific techniques constitute parallel text generation and how they improve inference performance. To bridge this gap, we present a systematic survey of parallel text generation methods. We categorize existing approaches into AR-based and Non-AR-based paradigms, and provide a detailed examination of the core techniques within each category. Following this taxonomy, we assess their theoretical trade-offs in terms of speed, quality, and efficiency, and examine their potential for combination and comparison with alternative acceleration strategies. Finally, based on our findings, we highlight recent advancements, identify open challenges, and outline promising directions for future research in parallel text generation. 

---
# Out of the Box, into the Clinic? Evaluating State-of-the-Art ASR for Clinical Applications for Older Adults 

**Authors**: Bram van Dijk, Tiberon Kuiper, Sirin Aoulad si Ahmed, Armel Levebvre, Jake Johnson, Jan Duin, Simon Mooijaart, Marco Spruit  

**Link**: [PDF](https://arxiv.org/pdf/2508.08684)  

**Abstract**: Voice-controlled interfaces can support older adults in clinical contexts, with chatbots being a prime example, but reliable Automatic Speech Recognition (ASR) for underrepresented groups remains a bottleneck. This study evaluates state-of-the-art ASR models on language use of older Dutch adults, who interacted with the this http URL chatbot designed for geriatric contexts. We benchmark generic multilingual ASR models, and models fine-tuned for Dutch spoken by older adults, while also considering processing speed. Our results show that generic multilingual models outperform fine-tuned models, which suggests recent ASR models can generalise well out of the box to realistic datasets. Furthermore, our results suggest that truncating existing architectures is helpful in balancing the accuracy-speed trade-off, though we also identify some cases with high WER due to hallucinations. 

---
# TopXGen: Topic-Diverse Parallel Data Generation for Low-Resource Machine Translation 

**Authors**: Armel Zebaze, Benoît Sagot, Rachel Bawden  

**Link**: [PDF](https://arxiv.org/pdf/2508.08680)  

**Abstract**: LLMs have been shown to perform well in machine translation (MT) with the use of in-context learning (ICL), rivaling supervised models when translating into high-resource languages (HRLs). However, they lag behind when translating into low-resource language (LRLs). Example selection via similarity search and supervised fine-tuning help. However the improvements they give are limited by the size, quality and diversity of existing parallel datasets. A common technique in low-resource MT is synthetic parallel data creation, the most frequent of which is backtranslation, whereby existing target-side texts are automatically translated into the source language. However, this assumes the existence of good quality and relevant target-side texts, which are not readily available for many LRLs. In this paper, we present \textsc{TopXGen}, an LLM-based approach for the generation of high quality and topic-diverse data in multiple LRLs, which can then be backtranslated to produce useful and diverse parallel texts for ICL and fine-tuning. Our intuition is that while LLMs struggle to translate into LRLs, their ability to translate well into HRLs and their multilinguality enable them to generate good quality, natural-sounding target-side texts, which can be translated well into a high-resource source language. We show that \textsc{TopXGen} boosts LLM translation performance during fine-tuning and in-context learning. Code and outputs are available at this https URL. 

---
# LLM driven Text-to-Table Generation through Sub-Tasks Guidance and Iterative Refinement 

**Authors**: Rajmohan C, Sarthak Harne, Arvind Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.08653)  

**Abstract**: Transforming unstructured text into structured data is a complex task, requiring semantic understanding, reasoning, and structural comprehension. While Large Language Models (LLMs) offer potential, they often struggle with handling ambiguous or domain-specific data, maintaining table structure, managing long inputs, and addressing numerical reasoning. This paper proposes an efficient system for LLM-driven text-to-table generation that leverages novel prompting techniques. Specifically, the system incorporates two key strategies: breaking down the text-to-table task into manageable, guided sub-tasks and refining the generated tables through iterative self-feedback. We show that this custom task decomposition allows the model to address the problem in a stepwise manner and improves the quality of the generated table. Furthermore, we discuss the benefits and potential risks associated with iterative self-feedback on the generated tables while highlighting the trade-offs between enhanced performance and computational cost. Our methods achieve strong results compared to baselines on two complex text-to-table generation datasets available in the public domain. 

---
# Prompt-Based Approach for Czech Sentiment Analysis 

**Authors**: Jakub Šmíd, Pavel Přibáň  

**Link**: [PDF](https://arxiv.org/pdf/2508.08651)  

**Abstract**: This paper introduces the first prompt-based methods for aspect-based sentiment analysis and sentiment classification in Czech. We employ the sequence-to-sequence models to solve the aspect-based tasks simultaneously and demonstrate the superiority of our prompt-based approach over traditional fine-tuning. In addition, we conduct zero-shot and few-shot learning experiments for sentiment classification and show that prompting yields significantly better results with limited training examples compared to traditional fine-tuning. We also demonstrate that pre-training on data from the target domain can lead to significant improvements in a zero-shot scenario. 

---
# UWB at WASSA-2024 Shared Task 2: Cross-lingual Emotion Detection 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.08650)  

**Abstract**: This paper presents our system built for the WASSA-2024 Cross-lingual Emotion Detection Shared Task. The task consists of two subtasks: first, to assess an emotion label from six possible classes for a given tweet in one of five languages, and second, to predict words triggering the detected emotions in binary and numerical formats. Our proposed approach revolves around fine-tuning quantized large language models, specifically Orca~2, with low-rank adapters (LoRA) and multilingual Transformer-based models, such as XLM-R and mT5. We enhance performance through machine translation for both subtasks and trigger word switching for the second subtask. The system achieves excellent performance, ranking 1st in numerical trigger words detection, 3rd in binary trigger words detection, and 7th in emotion detection. 

---
# LLaMA-Based Models for Aspect-Based Sentiment Analysis 

**Authors**: Jakub Šmíd, Pavel Přibáň, Pavel Král  

**Link**: [PDF](https://arxiv.org/pdf/2508.08649)  

**Abstract**: While large language models (LLMs) show promise for various tasks, their performance in compound aspect-based sentiment analysis (ABSA) tasks lags behind fine-tuned models. However, the potential of LLMs fine-tuned for ABSA remains unexplored. This paper examines the capabilities of open-source LLMs fine-tuned for ABSA, focusing on LLaMA-based models. We evaluate the performance across four tasks and eight English datasets, finding that the fine-tuned Orca~2 model surpasses state-of-the-art results in all tasks. However, all models struggle in zero-shot and few-shot scenarios compared to fully fine-tuned ones. Additionally, we conduct error analysis to identify challenges faced by fine-tuned models. 

---
# Quick on the Uptake: Eliciting Implicit Intents from Human Demonstrations for Personalized Mobile-Use Agents 

**Authors**: Zheng Wu, Heyuan Huang, Yanjia Yang, Yuanyi Song, Xingyu Lou, Weiwen Liu, Weinan Zhang, Jun Wang, Zhuosheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08645)  

**Abstract**: As multimodal large language models advance rapidly, the automation of mobile tasks has become increasingly feasible through the use of mobile-use agents that mimic human interactions from graphical user interface. To further enhance mobile-use agents, previous studies employ demonstration learning to improve mobile-use agents from human demonstrations. However, these methods focus solely on the explicit intention flows of humans (e.g., step sequences) while neglecting implicit intention flows (e.g., personal preferences), which makes it difficult to construct personalized mobile-use agents. In this work, to evaluate the \textbf{I}ntention \textbf{A}lignment \textbf{R}ate between mobile-use agents and humans, we first collect \textbf{MobileIAR}, a dataset containing human-intent-aligned actions and ground-truth actions. This enables a comprehensive assessment of the agents' understanding of human intent. Then we propose \textbf{IFRAgent}, a framework built upon \textbf{I}ntention \textbf{F}low \textbf{R}ecognition from human demonstrations. IFRAgent analyzes explicit intention flows from human demonstrations to construct a query-level vector library of standard operating procedures (SOP), and analyzes implicit intention flows to build a user-level habit repository. IFRAgent then leverages a SOP extractor combined with retrieval-augmented generation and a query rewriter to generate personalized query and SOP from a raw ambiguous query, enhancing the alignment between mobile-use agents and human intent. Experimental results demonstrate that IFRAgent outperforms baselines by an average of 6.79\% (32.06\% relative improvement) in human intention alignment rate and improves step completion rates by an average of 5.30\% (26.34\% relative improvement). The codes are available at this https URL. 

---
# InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling 

**Authors**: Peiji Li, Jiasheng Ye, Yongkang Chen, Yichuan Ma, Zijie Yu, Kedi Chen, Ganqu Cui, Haozhan Li, Jiacheng Chen, Chengqi Lyu, Wenwei Zhang, Linyang Li, Qipeng Guo, Dahua Lin, Bowen Zhou, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08636)  

**Abstract**: Large language models (LLMs) have revolutionized artificial intelligence by enabling complex reasoning capabilities. While recent advancements in reinforcement learning (RL) have primarily focused on domain-specific reasoning tasks (e.g., mathematics or code generation), real-world reasoning scenarios often require models to handle diverse and complex environments that narrow-domain benchmarks cannot fully capture. To address this gap, we present InternBootcamp, an open-source framework comprising 1000+ domain-diverse task environments specifically designed for LLM reasoning research. Our codebase offers two key functionalities: (1) automated generation of unlimited training/testing cases with configurable difficulty levels, and (2) integrated verification modules for objective response evaluation. These features make InternBootcamp fundamental infrastructure for RL-based model optimization, synthetic data generation, and model evaluation. Although manually developing such a framework with enormous task coverage is extremely cumbersome, we accelerate the development procedure through an automated agent workflow supplemented by manual validation protocols, which enables the task scope to expand rapidly. % With these bootcamps, we further establish Bootcamp-EVAL, an automatically generated benchmark for comprehensive performance assessment. Evaluation reveals that frontier models still underperform in many reasoning tasks, while training with InternBootcamp provides an effective way to significantly improve performance, leading to our 32B model that achieves state-of-the-art results on Bootcamp-EVAL and excels on other established benchmarks. In particular, we validate that consistent performance gains come from including more training tasks, namely \textbf{task scaling}, over two orders of magnitude, offering a promising route towards capable reasoning generalist. 

---
# Optimizing Retrieval-Augmented Generation (RAG) for Colloquial Cantonese: A LoRA-Based Systematic Review 

**Authors**: David Santandreu Calonge, Linda Smail  

**Link**: [PDF](https://arxiv.org/pdf/2508.08610)  

**Abstract**: This review examines recent advances in Parameter-Efficient Fine-Tuning (PEFT), with a focus on Low-Rank Adaptation (LoRA), to optimize Retrieval-Augmented Generation (RAG) systems like Qwen3, DeepSeek, and Kimi. These systems face challenges in understanding and generating authentic Cantonese colloquial expressions due to limited annotated data and linguistic variability. The review evaluates the integration of LoRA within RAG frameworks, benchmarks PEFT methods for retrieval and generation accuracy, identify domain adaptation strategies under limited data, and compares fine-tuning techniques aimed at improving semantic fidelity under data-scarce conditions. A systematic analysis of recent studies employing diverse LoRA variants, synthetic data generation, user feedback integration, and adaptive parameter allocation was conducted to assess their impact on computational efficiency, retrieval precision, linguistic authenticity, and scalability. Findings reveal that dynamic and ensemble LoRA adaptations significantly reduce trainable parameters without sacrificing retrieval accuracy and generation quality in dialectal contexts. However, limitations remain in fully preserving fine-grained linguistic nuances, especially for low-resource settings like Cantonese. The integration of real-time user feedback and domain-specific data remains underdeveloped, limiting model adaptability and personalization. While selective parameter freezing and nonlinear adaptation methods offer better trade-offs between efficiency and accuracy, their robustness at scale remains an open challenge. This review highlights the promise of PEFT-enhanced RAG systems for domain-specific language tasks and calls for future work targeting dialectal authenticity, dynamic adaptation, and scalable fine-tuning pipelines. 

---
# DepressLLM: Interpretable domain-adapted language model for depression detection from real-world narratives 

**Authors**: Sehwan Moon, Aram Lee, Jeong Eun Kim, Hee-Ju Kang, Il-Seon Shin, Sung-Wan Kim, Jae-Min Kim, Min Jhon, Ju-Wan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.08591)  

**Abstract**: Advances in large language models (LLMs) have enabled a wide range of applications. However, depression prediction is hindered by the lack of large-scale, high-quality, and rigorously annotated datasets. This study introduces DepressLLM, trained and evaluated on a novel corpus of 3,699 autobiographical narratives reflecting both happiness and distress. DepressLLM provides interpretable depression predictions and, via its Score-guided Token Probability Summation (SToPS) module, delivers both improved classification performance and reliable confidence estimates, achieving an AUC of 0.789, which rises to 0.904 on samples with confidence $\geq$ 0.95. To validate its robustness to heterogeneous data, we evaluated DepressLLM on in-house datasets, including an Ecological Momentary Assessment (EMA) corpus of daily stress and mood recordings, and on public clinical interview data. Finally, a psychiatric review of high-confidence misclassifications highlighted key model and data limitations that suggest directions for future refinements. These findings demonstrate that interpretable AI can enable earlier diagnosis of depression and underscore the promise of medical AI in psychiatry. 

---
# DeCAL Tokenwise Compression 

**Authors**: Sameer Panwar  

**Link**: [PDF](https://arxiv.org/pdf/2508.08514)  

**Abstract**: This paper introduces DeCAL, a new method for tokenwise compression. DeCAL uses an encoder-decoder language model pretrained with denoising to learn to produce high-quality, general-purpose compressed representations by the encoder. DeCAL applies small modifications to the encoder, with the emphasis on maximizing compression quality, even at the expense of compute. We show that DeCAL at 2x compression can match uncompressed on many downstream tasks, with usually only minor dropoff in metrics up to 8x compression, among question-answering, summarization, and multi-vector retrieval tasks. DeCAL offers significant savings where pre-computed dense representations can be utilized, and we believe the approach can be further developed to be more broadly applicable. 

---
# Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression 

**Authors**: Jadie Adams, Brian Hu, Emily Veenhuis, David Joy, Bharadwaj Ravichandran, Aaron Bray, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2508.08509)  

**Abstract**: Large language models (LLMs) are currently aligned using techniques such as reinforcement learning from human feedback (RLHF). However, these methods use scalar rewards that can only reflect user preferences on average. Pluralistic alignment instead seeks to capture diverse user preferences across a set of attributes, moving beyond just helpfulness and harmlessness. Toward this end, we propose a steerable pluralistic model based on few-shot comparative regression that can adapt to individual user preferences. Our approach leverages in-context learning and reasoning, grounded in a set of fine-grained attributes, to compare response options and make aligned choices. To evaluate our algorithm, we also propose two new steerable pluralistic benchmarks by adapting the Moral Integrity Corpus (MIC) and the HelpSteer2 datasets, demonstrating the applicability of our approach to value-aligned decision-making and reward modeling, respectively. Our few-shot comparative regression approach is interpretable and compatible with different attributes and LLMs, while outperforming multiple baseline and state-of-the-art methods. Our work provides new insights and research directions in pluralistic alignment, enabling a more fair and representative use of LLMs and advancing the state-of-the-art in ethical AI. 

---
# Momentum Point-Perplexity Mechanics in Large Language Models 

**Authors**: Lorenzo Tomaz, Judd Rosenblatt, Thomas Berry Jones, Diogo Schwerz de Lucena  

**Link**: [PDF](https://arxiv.org/pdf/2508.08492)  

**Abstract**: We take a physics-based approach to studying how the internal hidden states of large language models change from token to token during inference. Across 20 open-source transformer models (135M-3B parameters), we find that a quantity combining the rate of change in hidden states and the model's next-token certainty, analogous to energy in physics, remains nearly constant. Random-weight models conserve this "energy" more tightly than pre-trained ones, while training shifts models into a faster, more decisive regime with greater variability. Using this "log-Lagrangian" view, we derive a control method called Jacobian steering, which perturbs hidden states in the minimal way needed to favor a target token. This approach maintained near-constant energy in two tested models and produced continuations rated higher in semantic quality than the models' natural outputs. Viewing transformers through this mechanics lens offers a principled basis for interpretability, anomaly detection, and low-risk steering. This could help make powerful models more predictable and aligned with human intent. 

---
# Enhancing Small LLM Alignment through Margin-Based Objective Modifications under Resource Constraints 

**Authors**: Daren Yao, Jinsong Yuan, Ruike Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08466)  

**Abstract**: Small large language models (LLMs) often face difficulties in aligning output to human preferences, particularly when operating under severe performance gaps. In this work, we propose two lightweight DPO-based variants -- Adaptive Margin-Sigmoid Loss and APO-hinge-zero -- to better address underperformance scenarios by introducing margin-based objectives and selective update mechanisms.
Our APO-hinge-zero method, which combines hinge-induced hard-example mining with the chosen-focused optimization of APO-zero, achieves strong results. In AlpacaEval, APO-hinge-zero improves the win rate by +2.0 points and the length-controlled win rate by +1.4 points compared to the APO-zero baseline. In MT-Bench, our methods maintain competitive performance in diverse categories, particularly excelling in STEM and Humanities tasks.
These results demonstrate that simple modifications to preference-based objectives can significantly enhance small LLM alignment under resource constraints, offering a practical path toward more efficient deployment. 

---
# Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment 

**Authors**: Saketh Reddy Vemula, Dipti Mishra Sharma, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2508.08424)  

**Abstract**: Prior work on language modeling showed conflicting findings about whether morphologically aligned approaches to tokenization improve performance, particularly for languages with complex morphology. To investigate this, we select a typologically diverse set of languages: Telugu (agglutinative), Hindi (primarily fusional with some agglutination), and English (fusional). We conduct a comprehensive evaluation of language models -- starting from tokenizer training and extending through the finetuning and downstream task evaluation. To account for the consistent performance differences observed across tokenizer variants, we focus on two key factors: morphological alignment and tokenization quality. To assess morphological alignment of tokenizers in Telugu, we create a dataset containing gold morpheme segmentations of 600 derivational and 7000 inflectional word forms.
Our experiments reveal that better morphological alignment correlates positively -- though moderately -- with performance in syntax-based tasks such as Parts-of-Speech tagging, Named Entity Recognition and Dependency Parsing. However, we also find that the tokenizer algorithm (Byte-pair Encoding vs. Unigram) plays a more significant role in influencing downstream performance than morphological alignment alone. Naive Unigram tokenizers outperform others across most settings, though hybrid tokenizers that incorporate morphological segmentation significantly improve performance within the BPE framework. In contrast, intrinsic metrics like Corpus Token Count (CTC) and Rényi entropy showed no correlation with downstream performance. 

---
# Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery 

**Authors**: Jiatong Li, Weida Wang, Qinggang Zhang, Junxian Li, Di Zhang, Changmeng Zheng, Shufei Zhang, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08401)  

**Abstract**: Large language models (LLMs), especially Explicit Long Chain-of-Thought (CoT) reasoning models like DeepSeek-R1 and QWQ, have demonstrated powerful reasoning capabilities, achieving impressive performance in commonsense reasoning and mathematical inference. Despite their effectiveness, Long-CoT reasoning models are often criticized for their limited ability and low efficiency in knowledge-intensive domains such as molecule discovery. Success in this field requires a precise understanding of domain knowledge, including molecular structures and chemical principles, which is challenging due to the inherent complexity of molecular data and the scarcity of high-quality expert annotations. To bridge this gap, we introduce Mol-R1, a novel framework designed to improve explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in text-based molecule generation. Our approach begins with a high-quality reasoning dataset curated through Prior Regulation via In-context Distillation (PRID), a dedicated distillation strategy to effectively generate paired reasoning traces guided by prior regulations. Building upon this, we introduce MoIA, Molecular Iterative Adaptation, a sophisticated training strategy that iteratively combines Supervised Fine-tuning (SFT) with Reinforced Policy Optimization (RPO), tailored to boost the reasoning performance of R1-like reasoning models for molecule discovery. Finally, we examine the performance of Mol-R1 in the text-based molecule reasoning generation task, showing superior performance against existing baselines. 

---
# CoDAE: Adapting Large Language Models for Education via Chain-of-Thought Data Augmentation 

**Authors**: Shuzhou Yuan, William LaCroix, Hardik Ghoshal, Ercong Nie, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2508.08386)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as AI tutors due to their scalability and potential for personalized instruction. However, off-the-shelf LLMs often underperform in educational settings: they frequently reveal answers too readily, fail to adapt their responses to student uncertainty, and remain vulnerable to emotionally manipulative prompts. To address these challenges, we introduce CoDAE, a framework that adapts LLMs for educational use through Chain-of-Thought (CoT) data augmentation. We collect real-world dialogues between students and a ChatGPT-based tutor and enrich them using CoT prompting to promote step-by-step reasoning and pedagogically aligned guidance. Furthermore, we design targeted dialogue cases to explicitly mitigate three key limitations: over-compliance, low response adaptivity, and threat vulnerability. We fine-tune four open-source LLMs on different variants of the augmented datasets and evaluate them in simulated educational scenarios using both automatic metrics and LLM-as-a-judge assessments. Our results show that models fine-tuned with CoDAE deliver more pedagogically appropriate guidance, better support reasoning processes, and effectively resist premature answer disclosure. 

---
# Putnam-AXIOM: A Functional and Static Benchmark 

**Authors**: Aryan Gulati, Brando Miranda, Eric Chen, Emily Xia, Kai Fronsdal, Bruno Dumont, Elyas Obbad, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08292)  

**Abstract**: Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at this https URL. 

---
# Sacred or Synthetic? Evaluating LLM Reliability and Abstention for Religious Questions 

**Authors**: Farah Atif, Nursultan Askarbekuly, Kareem Darwish, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.08287)  

**Abstract**: Despite the increasing usage of Large Language Models (LLMs) in answering questions in a variety of domains, their reliability and accuracy remain unexamined for a plethora of domains including the religious domains. In this paper, we introduce a novel benchmark FiqhQA focused on the LLM generated Islamic rulings explicitly categorized by the four major Sunni schools of thought, in both Arabic and English. Unlike prior work, which either overlooks the distinctions between religious school of thought or fails to evaluate abstention behavior, we assess LLMs not only on their accuracy but also on their ability to recognize when not to answer. Our zero-shot and abstention experiments reveal significant variation across LLMs, languages, and legal schools of thought. While GPT-4o outperforms all other models in accuracy, Gemini and Fanar demonstrate superior abstention behavior critical for minimizing confident incorrect answers. Notably, all models exhibit a performance drop in Arabic, highlighting the limitations in religious reasoning for languages other than English. To the best of our knowledge, this is the first study to benchmark the efficacy of LLMs for fine-grained Islamic school of thought specific ruling generation and to evaluate abstention for Islamic jurisprudence queries. Our findings underscore the need for task-specific evaluation and cautious deployment of LLMs in religious applications. 

---
# The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs 

**Authors**: Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Schwartz-Ziv, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.08285)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their tendency to hallucinate poses serious challenges for reliable deployment. Despite numerous hallucination detection methods, their evaluations often rely on ROUGE, a metric based on lexical overlap that misaligns with human judgments. Through comprehensive human studies, we demonstrate that while ROUGE exhibits high recall, its extremely low precision leads to misleading performance estimates. In fact, several established detection methods show performance drops of up to 45.9\% when assessed using human-aligned metrics like LLM-as-Judge. Moreover, our analysis reveals that simple heuristics based on response length can rival complex detection techniques, exposing a fundamental flaw in current evaluation practices. We argue that adopting semantically aware and robust evaluation frameworks is essential to accurately gauge the true performance of hallucination detection methods, ultimately ensuring the trustworthiness of LLM outputs. 

---
# MinionsLLM: a Task-adaptive Framework For The Training and Control of Multi-Agent Systems Through Natural Language 

**Authors**: Andres Garcia Rincon, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2508.08283)  

**Abstract**: This paper presents MinionsLLM, a novel framework that integrates Large Language Models (LLMs) with Behavior Trees (BTs) and Formal Grammars to enable natural language control of multi-agent systems within arbitrary, user-defined environments. MinionsLLM provides standardized interfaces for defining environments, agents, and behavioral primitives, and introduces two synthetic dataset generation methods (Method A and Method B) to fine-tune LLMs for improved syntactic validity and semantic task relevance. We validate our approach using Google's Gemma 3 model family at three parameter scales (1B, 4B, and 12B) and demonstrate substantial gains: Method B increases syntactic validity to 92.6% and achieves a mean task performance improvement of 33% over baseline. Notably, our experiments show that smaller models benefit most from fine-tuning, suggesting promising directions for deploying compact, locally hosted LLMs in resource-constrained multi-agent control scenarios. The framework and all resources are released open-source to support reproducibility and future research. 

---
# Objective Metrics for Evaluating Large Language Models Using External Data Sources 

**Authors**: Haoze Du, Richard Li, Edward Gehringer  

**Link**: [PDF](https://arxiv.org/pdf/2508.08277)  

**Abstract**: Evaluating the performance of Large Language Models (LLMs) is a critical yet challenging task, particularly when aiming to avoid subjective assessments. This paper proposes a framework for leveraging subjective metrics derived from the class textual materials across different semesters to assess LLM outputs across various tasks. By utilizing well-defined benchmarks, factual datasets, and structured evaluation pipelines, the approach ensures consistent, reproducible, and bias-minimized measurements. The framework emphasizes automation and transparency in scoring, reducing reliance on human interpretation while ensuring alignment with real-world applications. This method addresses the limitations of subjective evaluation methods, providing a scalable solution for performance assessment in educational, scientific, and other high-stakes domains. 

---
# Evaluating Contrast Localizer for Identifying Causal Unitsin Social & Mathematical Tasks in Language Models 

**Authors**: Yassine Jamaa, Badr AlKhamissi, Satrajit Ghosh, Martin Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2508.08276)  

**Abstract**: This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive stimulus sets and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the causal relevance of contrast-based localizers and highlight the need for broader stimulus sets and more accurately capture task-specific units. 

---
# MLLM-CBench:A Comprehensive Benchmark for Continual Instruction Tuning of Multimodal LLMs with Chain-of-Thought Reasoning Analysis 

**Authors**: Haiyun Guo, ZhiYan Hou, Yu Chen, Jinghan He, Yandu Sun, Yuzhe Zhou, Shujing Guo, Kuan Zhu, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08275)  

**Abstract**: Multimodal Large Language Models (MLLMs) rely on continual instruction tuning to adapt to the evolving demands of real-world applications. However, progress in this area is hindered by the lack of rigorous and systematic benchmarks. To address this gap, we present MLLM-CTBench, a comprehensive evaluation benchmark with three key contributions: (1) Multidimensional Evaluation: We combine final answer accuracy with fine-grained CoT reasoning quality assessment, enabled by a specially trained CoT evaluator; (2) Comprehensive Evaluation of Algorithms and Training Paradigms: We benchmark eight continual learning algorithms across four major categories and systematically compare reinforcement learning with supervised fine-tuning paradigms; (3) Carefully Curated Tasks: We select and organize 16 datasets from existing work, covering six challenging domains. Our key findings include: (i) Models with stronger general capabilities exhibit greater robustness to forgetting during continual learning; (ii) Reasoning chains degrade more slowly than final answers, supporting the hierarchical forgetting hypothesis; (iii) The effectiveness of continual learning algorithms is highly dependent on both model capability and task order; (iv) In reinforcement learning settings, incorporating KL-divergence constraints helps maintain policy stability and plays a crucial role in mitigating forgetting. MLLM-CTBench establishes a rigorous standard for continual instruction tuning of MLLMs and offers practical guidance for algorithm design and evaluation. 

---
# Distilling Knowledge from Large Language Models: A Concept Bottleneck Model for Hate and Counter Speech Recognition 

**Authors**: Roberto Labadie-Tamayo, Djordje Slijepčević, Xihui Chen, Adrian Jaques Böck, Andreas Babic, Liz Freimann, Christiane Atzmüller Matthias Zeppelzauer  

**Link**: [PDF](https://arxiv.org/pdf/2508.08274)  

**Abstract**: The rapid increase in hate speech on social media has exposed an unprecedented impact on society, making automated methods for detecting such content important. Unlike prior black-box models, we propose a novel transparent method for automated hate and counter speech recognition, i.e., "Speech Concept Bottleneck Model" (SCBM), using adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to map input texts to an abstract adjective-based representation, which is then sent to a light-weight classifier for downstream tasks. Across five benchmark datasets spanning multiple languages and platforms (e.g., Twitter, Reddit, YouTube), SCBM achieves an average macro-F1 score of 0.69 which outperforms the most recently reported results from the literature on four out of five datasets. Aside from high recognition accuracy, SCBM provides a high level of both local and global interpretability. Furthermore, fusing our adjective-based concept representation with transformer embeddings, leads to a 1.8% performance increase on average across all datasets, showing that the proposed representation captures complementary information. Our results demonstrate that adjective-based concept representations can serve as compact, interpretable, and effective encodings for hate and counter speech recognition. With adapted adjectives, our method can also be applied to other NLP tasks. 

---
# TT-XAI: Trustworthy Clinical Text Explanations via Keyword Distillation and LLM Reasoning 

**Authors**: Kristian Miok, Blaz Škrlj, Daniela Zaharie, Marko Robnik Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2508.08273)  

**Abstract**: Clinical language models often struggle to provide trustworthy predictions and explanations when applied to lengthy, unstructured electronic health records (EHRs). This work introduces TT-XAI, a lightweight and effective framework that improves both classification performance and interpretability through domain-aware keyword distillation and reasoning with large language models (LLMs). First, we demonstrate that distilling raw discharge notes into concise keyword representations significantly enhances BERT classifier performance and improves local explanation fidelity via a focused variant of LIME. Second, we generate chain-of-thought clinical explanations using keyword-guided prompts to steer LLMs, producing more concise and clinically relevant reasoning. We evaluate explanation quality using deletion-based fidelity metrics, self-assessment via LLaMA-3 scoring, and a blinded human study with domain experts. All evaluation modalities consistently favor the keyword-augmented method, confirming that distillation enhances both machine and human interpretability. TT-XAI offers a scalable pathway toward trustworthy, auditable AI in clinical decision support. 

---
# Real-time News Story Identification 

**Authors**: Tadej Škvorc, Nikola Ivačič, Sebastjan Hribar, Marko Robnik-Šikonja  

**Link**: [PDF](https://arxiv.org/pdf/2508.08272)  

**Abstract**: To improve the reading experience, many news sites organize news into topical collections, called stories. In this work, we present an approach for implementing real-time story identification for a news monitoring system that automatically collects news articles as they appear online and processes them in various ways. Story identification aims to assign each news article to a specific story that the article is covering. The process is similar to text clustering and topic modeling, but requires that articles be grouped based on particular events, places, and people, rather than general text similarity (as in clustering) or general (predefined) topics (as in topic modeling). We present an approach to story identification that is capable of functioning in real time, assigning articles to stories as they are published online. In the proposed approach, we combine text representation techniques, clustering algorithms, and online topic modeling methods. We combine various text representation methods to extract specific events and named entities necessary for story identification, showing that a mixture of online topic-modeling approaches such as BERTopic, DBStream, and TextClust can be adapted for story discovery. We evaluate our approach on a news dataset from Slovene media covering a period of 1 month. We show that our real-time approach produces sensible results as judged by human evaluators. 

---
# Heartificial Intelligence: Exploring Empathy in Language Models 

**Authors**: Victoria Williams, Benjamin Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2508.08271)  

**Abstract**: Large language models have become increasingly common, used by millions of people worldwide in both professional and personal contexts. As these models continue to advance, they are frequently serving as virtual assistants and companions. In human interactions, effective communication typically involves two types of empathy: cognitive empathy (understanding others' thoughts and emotions) and affective empathy (emotionally sharing others' feelings). In this study, we investigated both cognitive and affective empathy across several small (SLMs) and large (LLMs) language models using standardized psychological tests. Our results revealed that LLMs consistently outperformed humans - including psychology students - on cognitive empathy tasks. However, despite their cognitive strengths, both small and large language models showed significantly lower affective empathy compared to human participants. These findings highlight rapid advancements in language models' ability to simulate cognitive empathy, suggesting strong potential for providing effective virtual companionship and personalized emotional support. Additionally, their high cognitive yet lower affective empathy allows objective and consistent emotional support without running the risk of emotional fatigue or bias. 

---
# TurQUaz at CheckThat! 2025: Debating Large Language Models for Scientific Web Discourse Detection 

**Authors**: Tarık Saraç, Selin Mergen, Mucahid Kutlu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08265)  

**Abstract**: In this paper, we present our work developed for the scientific web discourse detection task (Task 4a) of CheckThat! 2025. We propose a novel council debate method that simulates structured academic discussions among multiple large language models (LLMs) to identify whether a given tweet contains (i) a scientific claim, (ii) a reference to a scientific study, or (iii) mentions of scientific entities. We explore three debating methods: i) single debate, where two LLMs argue for opposing positions while a third acts as a judge; ii) team debate, in which multiple models collaborate within each side of the debate; and iii) council debate, where multiple expert models deliberate together to reach a consensus, moderated by a chairperson model. We choose council debate as our primary model as it outperforms others in the development test set. Although our proposed method did not rank highly for identifying scientific claims (8th out of 10) or mentions of scientific entities (9th out of 10), it ranked first in detecting references to scientific studies. 

---
# Argument Quality Annotation and Gender Bias Detection in Financial Communication through Large Language Models 

**Authors**: Alaa Alhamzeh, Mays Al Rebdawi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08262)  

**Abstract**: Financial arguments play a critical role in shaping investment decisions and public trust in financial institutions. Nevertheless, assessing their quality remains poorly studied in the literature. In this paper, we examine the capabilities of three state-of-the-art LLMs GPT-4o, Llama 3.1, and Gemma 2 in annotating argument quality within financial communications, using the FinArgQuality dataset. Our contributions are twofold. First, we evaluate the consistency of LLM-generated annotations across multiple runs and benchmark them against human annotations. Second, we introduce an adversarial attack designed to inject gender bias to analyse models responds and ensure model's fairness and robustness. Both experiments are conducted across three temperature settings to assess their influence on annotation stability and alignment with human labels. Our findings reveal that LLM-based annotations achieve higher inter-annotator agreement than human counterparts, though the models still exhibit varying degrees of gender bias. We provide a multifaceted analysis of these outcomes and offer practical recommendations to guide future research toward more reliable, cost-effective, and bias-aware annotation methodologies. 

---
# P/D-Device: Disaggregated Large Language Model between Cloud and Devices 

**Authors**: Yibo Jin, Yixu Xu, Yue Chen, Chengbin Wang, Tao Wang, Jiaqi Huang, Rongfei Zhang, Yiming Dong, Yuting Yan, Ke Cheng, Yingjie Zhu, Shulan Wang, Qianqian Tang, Shuaishuai Meng, Guanxin Cheng, Ze Wang, Shuyan Miao, Ketao Wang, Wen Liu, Yifan Yang, Tong Zhang, Anran Wang, Chengzhou Lu, Tiantian Dong, Yongsheng Zhang, Zhe Wang, Hefei Guo, Hongjie Liu, Wei Lu, Zhengyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09035)  

**Abstract**: Serving disaggregated large language models has been widely adopted in industrial practice for enhanced performance. However, too many tokens generated in decoding phase, i.e., occupying the resources for a long time, essentially hamper the cloud from achieving a higher throughput. Meanwhile, due to limited on-device resources, the time to first token (TTFT), i.e., the latency of prefill phase, increases dramatically with the growth on prompt length. In order to concur with such a bottleneck on resources, i.e., long occupation in cloud and limited on-device computing capacity, we propose to separate large language model between cloud and devices. That is, the cloud helps a portion of the content for each device, only in its prefill phase. Specifically, after receiving the first token from the cloud, decoupling with its own prefill, the device responds to the user immediately for a lower TTFT. Then, the following tokens from cloud are presented via a speed controller for smoothed TPOT (the time per output token), until the device catches up with the progress. On-device prefill is then amortized using received tokens while the resource usage in cloud is controlled. Moreover, during cloud prefill, the prompt can be refined, using those intermediate data already generated, to further speed up on-device inference. We implement such a scheme P/D-Device, and confirm its superiority over other alternatives. We further propose an algorithm to decide the best settings. Real-trace experiments show that TTFT decreases at least 60%, maximum TPOT is about tens of milliseconds, and cloud throughput increases by up to 15x. 

---
# E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence,and Efficiency 

**Authors**: Dongjie Xu, Yue Cui, Weijie Shi, Qingzhi Ma, Hanghui Guo, Jiaming Li, Yao Zhao, Ruiyuan Zhang, Shimin Di, Jia Zhu, Kai Zheng, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09023)  

**Abstract**: SQL query rewriting aims to reformulate a query into a more efficient form while preserving equivalence. Most existing methods rely on predefined rewrite rules. However, such rule-based approaches face fundamental limitations: (1) fixed rule sets generalize poorly to novel query patterns and struggle with complex queries; (2) a wide range of effective rewriting strategies cannot be fully captured by declarative rules. To overcome these issues, we propose using large language models (LLMs) to generate rewrites. LLMs can capture complex strategies, such as evaluation reordering and CTE rewriting. Despite this potential, directly applying LLMs often results in suboptimal or non-equivalent rewrites due to a lack of execution awareness and semantic grounding. To address these challenges, We present E3-Rewrite, an LLM-based SQL rewriting framework that produces executable, equivalent, and efficient queries. It integrates two core components: a context construction module and a reinforcement learning framework. First, the context module leverages execution plans and retrieved demonstrations to build bottleneck-aware prompts that guide inference-time rewriting. Second, we design a reward function targeting executability, equivalence, and efficiency, evaluated via syntax checks, equivalence verification, and cost estimation. Third, to ensure stable multi-objective learning, we adopt a staged curriculum that first emphasizes executability and equivalence, then gradually incorporates efficiency. Extensive experiments show that E3-Rewrite achieves up to a 25.6\% reduction in query execution time compared to state-of-the-art methods across multiple SQL benchmarks. Moreover, it delivers up to 24.4\% more successful rewrites, expanding coverage to complex queries that previous systems failed to handle. 

---
# Revealing the Role of Audio Channels in ASR Performance Degradation 

**Authors**: Kuan-Tang Huang, Li-Wei Chen, Hung-Shin Lee, Berlin Chen, Hsin-Min Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08967)  

**Abstract**: Pre-trained automatic speech recognition (ASR) models have demonstrated strong performance on a variety of tasks. However, their performance can degrade substantially when the input audio comes from different recording channels. While previous studies have demonstrated this phenomenon, it is often attributed to the mismatch between training and testing corpora. This study argues that variations in speech characteristics caused by different recording channels can fundamentally harm ASR performance. To address this limitation, we propose a normalization technique designed to mitigate the impact of channel variation by aligning internal feature representations in the ASR model with those derived from a clean reference channel. This approach significantly improves ASR performance on previously unseen channels and languages, highlighting its ability to generalize across channel and language differences. 

---
# A Dual-Axis Taxonomy of Knowledge Editing for LLMs: From Mechanisms to Functions 

**Authors**: Amir Mohammad Salehoof, Ali Ramezani, Yadollah Yaghoobzadeh, Majid Nili Ahmadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08795)  

**Abstract**: Large language models (LLMs) acquire vast knowledge from large text corpora, but this information can become outdated or inaccurate. Since retraining is computationally expensive, knowledge editing offers an efficient alternative -- modifying internal knowledge without full retraining. These methods aim to update facts precisely while preserving the model's overall capabilities. While existing surveys focus on the mechanism of editing (e.g., parameter changes vs. external memory), they often overlook the function of the knowledge being edited. This survey introduces a novel, complementary function-based taxonomy to provide a more holistic view. We examine how different mechanisms apply to various knowledge types -- factual, temporal, conceptual, commonsense, and social -- highlighting how editing effectiveness depends on the nature of the target knowledge. By organizing our review along these two axes, we map the current landscape, outline the strengths and limitations of existing methods, define the problem formally, survey evaluation tasks and datasets, and conclude with open challenges and future directions. 

---
# Designing Memory-Augmented AR Agents for Spatiotemporal Reasoning in Personalized Task Assistance 

**Authors**: Dongwook Choi, Taeyoon Kwon, Dongil Yang, Hyojun Kim, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08774)  

**Abstract**: Augmented Reality (AR) systems are increasingly integrating foundation models, such as Multimodal Large Language Models (MLLMs), to provide more context-aware and adaptive user experiences. This integration has led to the development of AR agents to support intelligent, goal-directed interactions in real-world environments. While current AR agents effectively support immediate tasks, they struggle with complex multi-step scenarios that require understanding and leveraging user's long-term experiences and preferences. This limitation stems from their inability to capture, retain, and reason over historical user interactions in spatiotemporal contexts. To address these challenges, we propose a conceptual framework for memory-augmented AR agents that can provide personalized task assistance by learning from and adapting to user-specific experiences over time. Our framework consists of four interconnected modules: (1) Perception Module for multimodal sensor processing, (2) Memory Module for persistent spatiotemporal experience storage, (3) Spatiotemporal Reasoning Module for synthesizing past and present contexts, and (4) Actuator Module for effective AR communication. We further present an implementation roadmap, a future evaluation strategy, a potential target application and use cases to demonstrate the practical applicability of our framework across diverse domains. We aim for this work to motivate future research toward developing more intelligent AR systems that can effectively bridge user's interaction history with adaptive, context-aware task assistance. 

---
# MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs 

**Authors**: Xiaoxue Gao, Huayun Zhang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08715)  

**Abstract**: Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods. 

---
# $\text{M}^{2}$LLM: Multi-view Molecular Representation Learning with Large Language Models 

**Authors**: Jiaxin Ju, Yizhen Zheng, Huan Yee Koh, Can Wang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08657)  

**Abstract**: Accurate molecular property prediction is a critical challenge with wide-ranging applications in chemistry, materials science, and drug discovery. Molecular representation methods, including fingerprints and graph neural networks (GNNs), achieve state-of-the-art results by effectively deriving features from molecular structures. However, these methods often overlook decades of accumulated semantic and contextual knowledge. Recent advancements in large language models (LLMs) demonstrate remarkable reasoning abilities and prior knowledge across scientific domains, leading us to hypothesize that LLMs can generate rich molecular representations when guided to reason in multiple perspectives. To address these gaps, we propose $\text{M}^{2}$LLM, a multi-view framework that integrates three perspectives: the molecular structure view, the molecular task view, and the molecular rules view. These views are fused dynamically to adapt to task requirements, and experiments demonstrate that $\text{M}^{2}$LLM achieves state-of-the-art performance on multiple benchmarks across classification and regression tasks. Moreover, we demonstrate that representation derived from LLM achieves exceptional performance by leveraging two core functionalities: the generation of molecular embeddings through their encoding capabilities and the curation of molecular features through advanced reasoning processes. 

---
# MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time 

**Authors**: Peter Phan, Dhruv Agarwal, Kavitha Srinivas, Horst Samulowitz, Pavan Kapanipathi, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2508.08641)  

**Abstract**: Large language models (LLMs) are increasingly being applied to black-box optimization tasks, from program synthesis to molecule design. Prior work typically leverages in-context learning to iteratively guide the model towards better solutions. Such methods, however, often struggle to balance exploration of new solution spaces with exploitation of high-reward ones. Recently, test-time training (TTT) with synthetic data has shown promise in improving solution quality. However, the need for hand-crafted training data tailored to each task limits feasibility and scalability across domains. To address this problem, we introduce MiGrATe-a method for online TTT that uses GRPO as a search algorithm to adapt LLMs at inference without requiring external training data. MiGrATe operates via a mixed-policy group construction procedure that combines on-policy sampling with two off-policy data selection techniques: greedy sampling, which selects top-performing past completions, and neighborhood sampling (NS), which generates completions structurally similar to high-reward ones. Together, these components bias the policy gradient towards exploitation of promising regions in solution space, while preserving exploration through on-policy sampling. We evaluate MiGrATe on three challenging domains-word search, molecule optimization, and hypothesis+program induction on the Abstraction and Reasoning Corpus (ARC)-and find that it consistently outperforms both inference-only and TTT baselines, demonstrating the potential of online TTT as a solution for complex search tasks without external supervision. 

---
# Adaptive Personalized Conversational Information Retrieval 

**Authors**: Fengran Mo, Yuchen Hui, Yuxing Tian, Zhaoxuan Tan, Chuan Meng, Zhan Su, Kaiyu Huang, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.08634)  

**Abstract**: Personalized conversational information retrieval (CIR) systems aim to satisfy users' complex information needs through multi-turn interactions by considering user profiles. However, not all search queries require personalization. The challenge lies in appropriately incorporating personalization elements into search when needed. Most existing studies implicitly incorporate users' personal information and conversational context using large language models without distinguishing the specific requirements for each query turn. Such a ``one-size-fits-all'' personalization strategy might lead to sub-optimal results. In this paper, we propose an adaptive personalization method, in which we first identify the required personalization level for a query and integrate personalized queries with other query reformulations to produce various enhanced queries. Then, we design a personalization-aware ranking fusion approach to assign fusion weights dynamically to different reformulated queries, depending on the required personalization level. The proposed adaptive personalized conversational information retrieval framework APCIR is evaluated on two TREC iKAT datasets. The results confirm the effectiveness of adaptive personalization of APCIR by outperforming state-of-the-art methods. 

---
# Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization 

**Authors**: Chaoqun Cui, Liangbin Huang, Shijing Wang, Zhe Tong, Zhaolong Huang, Xiao Zeng, Xiaofeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08550)  

**Abstract**: Video dubbing aims to translate original speech in visual media programs from the source language to the target language, relying on neural machine translation and text-to-speech technologies. Due to varying information densities across languages, target speech often mismatches the source speech duration, causing audio-video synchronization issues that significantly impact viewer experience. In this study, we approach duration alignment in LLM-based video dubbing machine translation as a preference optimization problem. We propose the Segment Supervised Preference Optimization (SSPO) method, which employs a segment-wise sampling strategy and fine-grained loss to mitigate duration mismatches between source and target lines. Experimental results demonstrate that SSPO achieves superior performance in duration alignment tasks. 

---
# Re:Verse -- Can Your VLM Read a Manga? 

**Authors**: Aaditya Baranwal, Madhav Kataria, Naitik Agrawal, Yogesh S Rawat, Shruti Vyas  

**Link**: [PDF](https://arxiv.org/pdf/2508.08508)  

**Abstract**: Current Vision Language Models (VLMs) demonstrate a critical gap between surface-level recognition and deep narrative reasoning when processing sequential visual storytelling. Through a comprehensive investigation of manga narrative understanding, we reveal that while recent large multimodal models excel at individual panel interpretation, they systematically fail at temporal causality and cross-panel cohesion, core requirements for coherent story comprehension. We introduce a novel evaluation framework that combines fine-grained multimodal annotation, cross-modal embedding analysis, and retrieval-augmented assessment to systematically characterize these limitations.
Our methodology includes (i) a rigorous annotation protocol linking visual elements to narrative structure through aligned light novel text, (ii) comprehensive evaluation across multiple reasoning paradigms, including direct inference and retrieval-augmented generation, and (iii) cross-modal similarity analysis revealing fundamental misalignments in current VLMs' joint representations. Applying this framework to Re:Zero manga across 11 chapters with 308 annotated panels, we conduct the first systematic study of long-form narrative understanding in VLMs through three core evaluation axes: generative storytelling, contextual dialogue grounding, and temporal reasoning. Our findings demonstrate that current models lack genuine story-level intelligence, struggling particularly with non-linear narratives, character consistency, and causal inference across extended sequences. This work establishes both the foundation and practical methodology for evaluating narrative intelligence, while providing actionable insights into the capability of deep sequential understanding of Discrete Visual Narratives beyond basic recognition in Multimodal Models. 

---
# Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning 

**Authors**: Masataro Asai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08385)  

**Abstract**: We study an efficient implementation of Multi-Armed Bandit (MAB)-based Monte-Carlo Tree Search (MCTS) for classical planning. One weakness of MCTS is that it spends a significant time deciding which node to expand next. While selecting a node from an OPEN list with $N$ nodes has $O(1)$ runtime complexity with traditional array-based priority-queues for dense integer keys, the tree-based OPEN list used by MCTS requires $O(\log N)$, which roughly corresponds to the search depth $d$. In classical planning, $d$ is arbitrarily large (e.g., $2^k-1$ in $k$-disk Tower-of-Hanoi) and the runtime for node selection is significant, unlike in game tree search, where the cost is negligible compared to the node evaluation (rollouts) because $d$ is inherently limited by the game (e.g., $d\leq 361$ in Go). To improve this bottleneck, we propose a bilevel modification to MCTS that runs a best-first search from each selected leaf node with an expansion budget proportional to $d$, which achieves amortized $O(1)$ runtime for node selection, equivalent to the traditional queue-based OPEN list. In addition, we introduce Tree Collapsing, an enhancement that reduces action selection steps and further improves the performance. 

---
# Exploring the Technical Knowledge Interaction of Global Digital Humanities: Three-decade Evidence from Bibliometric-based perspectives 

**Authors**: Jiayi Li, Chengxi Yan, Yurong Zeng, Zhichao Fang, Huiru Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08347)  

**Abstract**: Digital Humanities (DH) is an interdisciplinary field that integrates computational methods with humanities scholarship to investigate innovative topics. Each academic discipline follows a unique developmental path shaped by the topics researchers investigate and the methods they employ. With the help of bibliometric analysis, most of previous studies have examined DH across multiple dimensions such as research hotspots, co-author networks, and institutional rankings. However, these studies have often been limited in their ability to provide deep insights into the current state of technological advancements and topic development in DH. As a result, their conclusions tend to remain superficial or lack interpretability in understanding how methods and topics interrelate in the field. To address this gap, this study introduced a new concept of Topic-Method Composition (TMC), which refers to a hybrid knowledge structure generated by the co-occurrence of specific research topics and the corresponding method. Especially by analyzing the interaction between TMCs, we can see more clearly the intersection and integration of digital technology and humanistic subjects in DH. Moreover, this study developed a TMC-based workflow combining bibliometric analysis, topic modeling, and network analysis to analyze the development characteristics and patterns of research disciplines. By applying this workflow to large-scale bibliometric data, it enables a detailed view of the knowledge structures, providing a tool adaptable to other fields. 

---
# Maximizing GPU Efficiency via Optimal Adapter Caching: An Analytical Approach for Multi-Tenant LLM Serving 

**Authors**: Ferran Agullo, Joan Oliveras, Chen Wang, Alberto Gutierrez-Torre, Olivier Tardieu, Alaa Youssef, Jordi Torres, Josep Ll. Berral  

**Link**: [PDF](https://arxiv.org/pdf/2508.08343)  

**Abstract**: Serving LLM adapters has gained significant attention as an effective approach to adapt general-purpose language models to diverse, task-specific use cases. However, serving a wide range of adapters introduces several and substantial overheads, leading to performance degradation and challenges in optimal placement. To address these challenges, we present an analytical, AI-driven pipeline that accurately determines the optimal allocation of adapters in single-node setups. This allocation maximizes performance, effectively using GPU resources, while preventing request starvation. Crucially, the proposed allocation is given based on current workload patterns. These insights in single-node setups can be leveraged in multi-replica deployments for overall placement, load balancing and server configuration, ultimately enhancing overall performance and improving resource efficiency. Our approach builds on an in-depth analysis of LLM adapter serving, accounting for overheads and performance variability, and includes the development of the first Digital Twin capable of replicating online LLM-adapter serving systems with matching key performance metrics. The experimental results demonstrate that the Digital Twin achieves a SMAPE difference of no more than 5.5% in throughput compared to real results, and the proposed pipeline accurately predicts the optimal placement with minimal latency. 

---
# Doctor Sun: A Bilingual Multimodal Large Language Model for Biomedical AI 

**Authors**: Dong Xue, Ziyao Shao, Zhaoyang Duan, Fangzhou Liu, Bing Li, Zhongheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08270)  

**Abstract**: Large multimodal models (LMMs) have demonstrated significant potential in providing innovative solutions for various biomedical tasks, including pathology analysis, radiology report generation, and biomedical assistance. However, the existing multimodal biomedical AI is typically based on foundation LLMs, thus hindering the understanding of intricate medical concepts with limited medical training data. Moreover, recent LLaVA-induced medical LMMs struggle to effectively capture the intricate relationship between the texts and the images. Therefore, we introduce Doctor Sun, a large multimodal generative model specialized in medicine, developed to encode, integrate, and interpret diverse biomedical data modalities such as text and images. In particular, Doctor Sun integrates a pre-trained vision encoder with a medical LLM and conducts two-stage training on various medical datasets, focusing on feature alignment and instruction tuning. Moreover, we release SunMed-VL, a wide-range bilingual medical multimodal dataset, along with all associated models, code, and resources, to freely support the advancement of biomedical multimodal research. 

---
# Benchmarking Large Language Models for Geolocating Colonial Virginia Land Grants 

**Authors**: Ryan Mioduski  

**Link**: [PDF](https://arxiv.org/pdf/2508.08266)  

**Abstract**: Virginia's seventeenth- and eighteenth-century land patents survive primarily as narrative metes-and-bounds descriptions, limiting spatial analysis. This study systematically evaluates current-generation large language models (LLMs) in converting these prose abstracts into geographically accurate latitude/longitude coordinates within a focused evaluation context. A digitized corpus of 5,471 Virginia patent abstracts (1695-1732) is released, with 43 rigorously verified test cases serving as an initial, geographically focused benchmark. Six OpenAI models across three architectures (o-series, GPT-4-class, and GPT-3.5) were tested under two paradigms: direct-to-coordinate and tool-augmented chain-of-thought invoking external geocoding APIs. Results were compared with a GIS-analyst baseline, the Stanford NER geoparser, Mordecai-3, and a county-centroid heuristic.
The top single-call model, o3-2025-04-16, achieved a mean error of 23 km (median 14 km), outperforming the median LLM (37.4 km) by 37.5%, the weakest LLM (50.3 km) by 53.5%, and external baselines by 67% (GIS analyst) and 70% (Stanford NER). A five-call ensemble further reduced errors to 19 km (median 12 km) at minimal additional cost (approx. USD 0.20 per grant), outperforming the median LLM by 48.6%. A patentee-name-redaction ablation increased error by about 9%, indicating reliance on textual landmark and adjacency descriptions rather than memorization. The cost-efficient gpt-4o-2024-08-06 model maintained a 28 km mean error at USD 1.09 per 1,000 grants, establishing a strong cost-accuracy benchmark; external geocoding tools offered no measurable benefit in this evaluation.
These findings demonstrate the potential of LLMs for scalable, accurate, and cost-effective historical georeferencing. 

---
